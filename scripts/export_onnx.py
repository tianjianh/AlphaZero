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
from onnx import numpy_helper, TensorProto

sys.path.insert(0, os.path.dirname(__file__))
from model import AlphaZeroNet


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

    for name, tensor in sd.items():
        if name not in existing_names:
            np_data = tensor.detach().cpu().numpy()
            onnx_tensor = numpy_helper.from_array(np_data, name=name)
            onnx_model.graph.initializer.append(onnx_tensor)

    onnx.save(onnx_model, onnx_path)


def export_to_onnx(model, output_path, board_size=9, input_channels=17):
    """Export a PyTorch AlphaZeroNet to ONNX format with dynamic batch axis.

    The ONNX file contains:
    1. The optimized graph (BN folded into Conv) for ONNX Runtime
    2. All raw state_dict tensors as extra initializers for the Eigen backend
    """
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
        output_names=["policy_logits", "value"],
        dynamic_axes={
            "state": {0: "batch"},
            "policy_logits": {0: "batch"},
            "value": {0: "batch"},
        },
        opset_version=17,
        do_constant_folding=True,
    )

    # Embed full state_dict for Eigen backend
    _embed_state_dict(output_path, model)

    file_size = os.path.getsize(output_path)
    action_size = board_size * board_size + 1
    print(f"Exported ONNX model to {output_path}")
    print(f"  Size: {file_size / 1024:.1f} KB")
    print(f"  Input: [batch, {input_channels}, {board_size}, {board_size}]")
    print(f"  Output: policy_logits [batch, {action_size}], value [batch, 1]")


def main():
    parser = argparse.ArgumentParser(
        description="Export PyTorch model to ONNX",
        epilog="""Examples:
  # Default small model (64 filters, 5 blocks)
  python3 export_onnx.py --init --output ../models/model.onnx

  # Large model for GPU benchmarking (128 filters, 10 blocks)
  python3 export_onnx.py --init --filters 128 --blocks 10 --output ../models/large.onnx
""",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", default="../training/checkpoints/training.pt",
                        help="PyTorch checkpoint path")
    parser.add_argument("--output", default="../models/model.onnx",
                        help="Output ONNX file")
    parser.add_argument("--board", type=int, default=9,
                        help="Board size (default: 9)")
    parser.add_argument("--filters", type=int, default=64,
                        help="Conv filters / model width (default: 64)")
    parser.add_argument("--blocks", type=int, default=5,
                        help="Residual blocks / model depth (default: 5)")
    parser.add_argument("--init", action="store_true",
                        help="Export an untrained (random) model")
    args = parser.parse_args()

    model = AlphaZeroNet(
        board_size=args.board,
        input_channels=17,
        num_filters=args.filters,
        num_res_blocks=args.blocks,
    )

    if not args.init:
        if os.path.exists(args.checkpoint):
            ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
            # Support both full checkpoint and state_dict-only formats
            if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
                model.load_state_dict(ckpt["model_state_dict"])
            else:
                model.load_state_dict(ckpt)
            print(f"Loaded checkpoint: {args.checkpoint}")
        else:
            print(f"No checkpoint at {args.checkpoint}, exporting random weights")

    export_to_onnx(model, args.output, board_size=args.board)


if __name__ == "__main__":
    main()
