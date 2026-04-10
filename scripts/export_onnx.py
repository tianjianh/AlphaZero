#!/usr/bin/env python3
"""
Export PyTorch AlphaZero model to ONNX format.

The ONNX model is the universal format used by all C++ inference backends.
Both the ONNX Runtime backend and the Eigen backend load .onnx files.

Usage:
  python export_onnx.py --checkpoint ../training/checkpoints/training.pt --output ../models/model.onnx
  python export_onnx.py --init --board 9 --output ../models/model.onnx  # random weights
"""

import argparse
import os
import sys

import torch
import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto

sys.path.insert(0, os.path.dirname(__file__))
from model import AlphaZeroNet, GoViT, create_model


def _embed_state_dict(onnx_path, model):
    """Embed the full PyTorch state_dict as ONNX initializer tensors.

    ONNX export may fold BatchNorm into Conv, losing separate BN params.
    We add all state_dict tensors as extra initializers with their original
    PyTorch names. The Eigen backend reads these; the ONNX Runtime backend
    ignores them (it uses the graph ops which have BN folded).
    """
    onnx_model = onnx.load(onnx_path)
    sd = model.state_dict()

    existing_names = {init.name for init in onnx_model.graph.initializer}
    # TorchScript exporter may also use param names as node outputs (SSA)
    for node in onnx_model.graph.node:
        existing_names.update(node.output)

    for name, tensor in sd.items():
        if name not in existing_names:
            np_data = tensor.detach().cpu().numpy()
            onnx_tensor = numpy_helper.from_array(np_data, name=name)
            onnx_model.graph.initializer.append(onnx_tensor)

    onnx.save(onnx_model, onnx_path)


def _append_postprocessing(onnx_path, board_size):
    """Append score post-processing to ONNX graph for C++ inference.

    Value already has tanh in the model — no post-processing needed.
    Score: softmax(logits) @ bin_values → raw points [B, 1]
    """
    model = onnx.load(onnx_path)
    graph = model.graph
    board_area = board_size * board_size
    num_bins = board_area * 2 + 1

    # Rename score output to make room for post-processed version
    for node in graph.node:
        new_outputs = list(node.output)
        for i, o in enumerate(new_outputs):
            if o == "score":
                new_outputs[i] = "score_logits"
        del node.output[:]
        node.output.extend(new_outputs)

    # --- Score: softmax → expected value ---
    graph.node.append(helper.make_node("Softmax", ["score_logits"], ["score_probs"], axis=1))
    bin_vals = (np.arange(num_bins, dtype=np.float32) - board_area).reshape(num_bins, 1)
    bv = numpy_helper.from_array(bin_vals, "score_bin_values")
    graph.initializer.append(bv)
    # MatMul: [B, num_bins] @ [num_bins, 1] → [B, 1]
    graph.node.append(helper.make_node("MatMul", ["score_probs", "score_bin_values"], ["score"]))

    # Update score output shape: [batch, num_bins] → [batch, 1]
    new_outputs = []
    for output in graph.output:
        if output.name == "score":
            new_outputs.append(
                helper.make_tensor_value_info("score", TensorProto.FLOAT, ["batch", 1]))
        else:
            new_outputs.append(output)
    del graph.output[:]
    graph.output.extend(new_outputs)

    onnx.save(model, onnx_path)


def export_to_onnx(model, output_path, board_size=9, input_channels=17, arch="resnet"):
    """Export a PyTorch model to ONNX format with dynamic batch axis.

    The ONNX file contains:
    1. The optimized graph (BN folded into Conv) for ONNX Runtime / TensorRT
    2. All raw state_dict tensors as extra initializers for the Eigen backend
    3. Post-processing ops: score softmax→expected_value
    """
    # Convert te.Linear → nn.Linear for ONNX compatibility
    from model import convert_te_to_nn
    convert_te_to_nn(model)

    model.eval()
    dummy = torch.randn(1, input_channels, board_size, board_size)

    # Run once to populate batch norm running stats
    with torch.no_grad():
        model(dummy)

    torch.onnx.export(
        model,
        dummy,
        output_path,
        input_names=["state"],
        output_names=["policy_logits", "value", "score"],
        dynamic_axes={
            "state": {0: "batch"},
            "policy_logits": {0: "batch"},
            "value": {0: "batch"},
            "score": {0: "batch"},
        },
        opset_version=18,
        do_constant_folding=True,
        external_data=False,
        dynamo=False,
    )

    # Append post-processing for C++ inference
    _append_postprocessing(output_path, board_size)

    # Embed full state_dict for Eigen backend
    _embed_state_dict(output_path, model)

    file_size = os.path.getsize(output_path)
    action_size = board_size * board_size + 1
    print(f"Exported ONNX model to {output_path}")
    print(f"  Size: {file_size / 1024:.1f} KB")
    print(f"  Input: [batch, {input_channels}, {board_size}, {board_size}]")
    print(f"  Output: policy_logits [batch, {action_size}], value [batch, 1], score [batch, 1]")


def main():
    parser = argparse.ArgumentParser(
        description="Export PyTorch model to ONNX",
        epilog="""Examples:
  # ResNet (default)
  python3 export_onnx.py --init --output ../models/model.onnx

  # ViT
  python3 export_onnx.py --init --arch vit --output ../models/vit.onnx
""",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", default="../training/checkpoints/training.pt")
    parser.add_argument("--output", default="../models/model.onnx")
    parser.add_argument("--board", type=int, default=9)
    parser.add_argument("--arch", default="resnet", choices=["resnet", "vit"])
    # ResNet params
    parser.add_argument("--filters", type=int, default=64)
    parser.add_argument("--blocks", type=int, default=5)
    # ViT params
    parser.add_argument("--d-model", type=int, default=192)
    parser.add_argument("--depth", type=int, default=8)
    parser.add_argument("--heads", type=int, default=6)
    parser.add_argument("--kv-groups", type=int, default=2)
    parser.add_argument("--mlp-ratio", type=int, default=4)
    parser.add_argument("--init", action="store_true",
                        help="Export an untrained (random) model")
    args = parser.parse_args()

    model = create_model(
        arch=args.arch, board_size=args.board, input_channels=17,
        num_filters=args.filters, num_res_blocks=args.blocks,
        d_model=args.d_model, depth=args.depth, heads=args.heads,
        kv_groups=args.kv_groups, mlp_ratio=args.mlp_ratio,
    )

    if not args.init:
        if os.path.exists(args.checkpoint):
            ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
            if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
                model.load_state_dict(ckpt["model_state_dict"], strict=False)
            else:
                model.load_state_dict(ckpt, strict=False)
            print(f"Loaded checkpoint: {args.checkpoint}")
        else:
            print(f"No checkpoint at {args.checkpoint}, exporting random weights")

    export_to_onnx(model, args.output, board_size=args.board, arch=args.arch)


if __name__ == "__main__":
    main()
