#!/usr/bin/env python3
"""
Export PyTorch AlphaZero model to ONNX format.

The ONNX model is the universal format used by all C++ inference backends.

Exports 5 inference heads:
  1. policy_logits [B, action_size]
  2. value [B, 1]         — P(win) - P(loss) after softmax
  3. score_mean [B, 1]    — raw regression output
  4. score_stdev [B, 1]   — softplus output (positive)
  5. ownership [B, board²] — per-intersection sigmoid

Usage:
  python export_onnx.py --checkpoint ../training/checkpoints/training.pt --output ../models/model.onnx
  python export_onnx.py --init --board 9 --output ../models/model.onnx  # random weights
"""

import argparse
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import onnx
from onnx import numpy_helper

sys.path.insert(0, os.path.dirname(__file__))
from model import create_model


class _InferenceWrapper(nn.Module):
    """Wrapper that calls forward_inference() and post-processes value logits.

    Value: softmax([B, 3]) → P(win) - P(loss) → [B, 1]
    All other outputs passed through unchanged.  Handles both input
    kinds: single (MiniGo 17-plane x) and dual (KataGo V7 spatial +
    global) — forward(*inputs) dispatches on arity.
    """

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, *inputs):
        policy, value_logits, score_mean, score_stdev, ownership = \
            self.model.forward_inference(*inputs)
        value_probs = F.softmax(value_logits, dim=1)
        value = value_probs[:, 0:1] - value_probs[:, 1:2]  # P(win) - P(loss) → [B, 1]
        return policy, value, score_mean, score_stdev, ownership


def _embed_state_dict(onnx_path, model):
    """Embed the full PyTorch state_dict as ONNX initializer tensors.

    ONNX export may fold BatchNorm into Conv, losing separate BN params,
    and can also reuse the original PyTorch names for the folded results.
    Prefix every embedded tensor with `_sd_` so the Eigen backend reads
    the un-folded weights regardless of what the optimizer did to the
    graph initializers. ONNX Runtime / TensorRT ignore these (they use
    the optimized graph).
    """
    onnx_model = onnx.load(onnx_path)
    for name, tensor in model.state_dict().items():
        np_data = tensor.detach().cpu().numpy()
        onnx_tensor = numpy_helper.from_array(np_data, name="_sd_" + name)
        onnx_model.graph.initializer.append(onnx_tensor)
    onnx.save(onnx_model, onnx_path)


def export_to_onnx(model, output_path, board_size=9, input_channels=17):
    """Export a PyTorch model to ONNX format with dynamic batch axis.

    Arch-agnostic: the model object carries its own architecture and
    input kind.  Single-input models (AlphaZeroNet, GoViT) export as
    `state [B,17,H,W]`; dual-input models (KataGoNet) export as
    `state_spatial [B,22,H,W]` + `state_global [B,19]` — the same
    contract as a converted kata1 network, so the C++ loader treats
    them identically (ModelFormat::KataGo).

    The ONNX file contains:
    1. The optimized graph (BN folded into Conv) for ONNX Runtime / TensorRT
    2. All raw state_dict tensors as extra initializers for the Eigen backend
    3. Value post-processing: softmax → P(win) - P(loss)
    """
    from model import convert_te_to_nn
    convert_te_to_nn(model)

    model.eval()
    wrapper = _InferenceWrapper(model)
    wrapper.eval()

    dual = getattr(model, "input_kind", "single") == "dual"
    if dual:
        dummy = (torch.randn(1, model.SPATIAL_CHANNELS, board_size, board_size),
                 torch.randn(1, model.GLOBAL_CHANNELS))
        input_names = ["state_spatial", "state_global"]
    else:
        dummy = (torch.randn(1, input_channels, board_size, board_size),)
        input_names = ["state"]

    # Run once to populate batch norm running stats
    with torch.no_grad():
        wrapper(*dummy)

    board_area = board_size * board_size
    output_names = ["policy_logits", "value", "score_mean", "score_stdev", "ownership"]

    torch.onnx.export(
        wrapper,
        dummy,
        output_path,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes={**{name: {0: "batch"} for name in input_names},
                      **{name: {0: "batch"} for name in output_names}},
        opset_version=18,
        do_constant_folding=True,
        external_data=False,
        dynamo=False,
    )

    # Embed full state_dict for Eigen backend
    _embed_state_dict(output_path, model)

    file_size = os.path.getsize(output_path)
    action_size = board_area + 1
    print(f"Exported ONNX model to {output_path}")
    print(f"  Size: {file_size / 1024:.1f} KB")
    if dual:
        print(f"  Inputs: state_spatial [batch, {model.SPATIAL_CHANNELS}, "
              f"{board_size}, {board_size}], state_global [batch, {model.GLOBAL_CHANNELS}]")
    else:
        print(f"  Input: [batch, {input_channels}, {board_size}, {board_size}]")
    print(f"  Outputs:")
    print(f"    policy_logits [batch, {action_size}]")
    print(f"    value [batch, 1]  (P(win) - P(loss))")
    print(f"    score_mean [batch, 1]  (raw points)")
    print(f"    score_stdev [batch, 1]  (uncertainty)")
    print(f"    ownership [batch, {board_area}]  (per-intersection)")


def main():
    parser = argparse.ArgumentParser(
        description="Export PyTorch model to ONNX",
        epilog="""Examples:
  # ResNet (default)
  python3 export_onnx.py --init --output ../models/model.onnx

  # ViT
  python3 export_onnx.py --init --arch vit --output ../models/vit.onnx

  # KataGo-V7 architecture (dual input, sized via --filters/--blocks)
  python3 export_onnx.py --init --arch katago --filters 128 --blocks 10 \\
      --output ../models/kata.onnx
""",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", default="../training/checkpoints/training.pt")
    parser.add_argument("--output", default="../models/model.onnx")
    parser.add_argument("--board", type=int, default=9)
    parser.add_argument("--arch", default="resnet",
                        choices=["resnet", "vit", "katago"])
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

    export_to_onnx(model, args.output, board_size=args.board)


if __name__ == "__main__":
    main()
