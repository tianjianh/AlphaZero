#!/usr/bin/env python3
"""Export the Xiangqi network to ONNX.

The exported graph exposes two outputs:
  - policy_logits: [batch, rows * cols * rows * cols]
  - value:         [batch, 1]   (P(win) - P(loss) after WDL softmax)

The wrapper collapses the internal 3-class WDL head into a single scalar
value so existing C++ backends can keep reading NNOutput.value as a float.
"""

from __future__ import annotations

import argparse
import os
import sys

import onnx
import torch
from onnx import numpy_helper

sys.path.insert(0, os.path.dirname(__file__))
from model import create_model


class ExportWrapper(torch.nn.Module):
    """Thin inference wrapper used by torch.onnx.export."""

    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x):
        return self.model.forward_inference(x)


def _embed_state_dict(onnx_path: str, model: torch.nn.Module) -> None:
    """Append PyTorch state_dict tensors to the ONNX graph as extra initializers.

    The graph exporter may fuse BatchNorm into the preceding Conv (always under
    the Dynamo exporter, sometimes under TorchScript), which drops the original
    `*.bn.weight / bias / running_mean / running_var` names from the graph.
    The project's lightweight ONNX loader (src/loaded_model.cpp) looks up those
    names directly and re-fuses BN itself.

    Attaching the raw state_dict as inert extra initializers lets the loader
    recover every weight by its PyTorch name regardless of what the graph
    optimizer did to the compute ops.  Graph-execution backends (TRT, ONNX
    Runtime) ignore the extras.
    """
    onnx_model = onnx.load(onnx_path)

    existing = {init.name for init in onnx_model.graph.initializer}
    for node in onnx_model.graph.node:
        existing.update(node.output)

    for name, tensor in model.state_dict().items():
        if name in existing:
            continue
        np_data = tensor.detach().cpu().numpy()
        onnx_model.graph.initializer.append(numpy_helper.from_array(np_data, name=name))

    onnx.save(onnx_model, onnx_path)


def export_to_onnx(
    model: torch.nn.Module,
    output_path: str,
    board_rows: int = 10,
    board_cols: int = 9,
    input_channels: int = 57,
):
    model.eval()
    wrapper = ExportWrapper(model)
    dummy = torch.randn(1, input_channels, board_rows, board_cols)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    torch.onnx.export(
        wrapper,
        dummy,
        output_path,
        input_names=["input"],
        output_names=["policy_logits", "value"],
        export_params=True,
        do_constant_folding=True,
        dynamic_axes={
            "input": {0: "batch"},
            "policy_logits": {0: "batch"},
            "value": {0: "batch"},
        },
        opset_version=17,
    )

    # Re-attach raw PyTorch weights so the C++ loader can find BN params
    # even if the exporter folded them into Conv.
    _embed_state_dict(output_path, model)

    action_size = board_rows * board_cols * board_rows * board_cols
    print(f"Exported ONNX model to {output_path}")
    print(f"  Input:  [batch, {input_channels}, {board_rows}, {board_cols}]")
    print(f"  Policy: [batch, {action_size}]")
    print("  Value:  [batch, 1]  (P(win) - P(loss))")


def main():
    parser = argparse.ArgumentParser(description="Export Xiangqi model to ONNX")
    parser.add_argument("--checkpoint", default="training/checkpoints/training.pt",
                        help="Checkpoint to export. If missing, exports a freshly initialized model.")
    parser.add_argument("--output", default="models/model.onnx",
                        help="Destination ONNX file")
    parser.add_argument("--rows", type=int, default=10, help="Board row count")
    parser.add_argument("--cols", type=int, default=9, help="Board column count")
    parser.add_argument("--history-length", type=int, default=4,
                        help="History snapshots encoded into the input tensor")
    parser.add_argument("--filters", type=int, default=128, help="Residual tower channel count")
    parser.add_argument("--blocks", type=int, default=10, help="Residual block count")
    args = parser.parse_args()

    input_channels = args.history_length * 14 + 1
    model = create_model(
        board_rows=args.rows,
        board_cols=args.cols,
        input_channels=input_channels,
        num_filters=args.filters,
        num_res_blocks=args.blocks,
    )

    if os.path.exists(args.checkpoint):
        ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        state_dict = ckpt.get("model_state_dict", ckpt)
        # Strip DDP "module." prefixes.
        state_dict = {k.removeprefix("module."): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict, strict=False)

    export_to_onnx(
        model,
        args.output,
        board_rows=args.rows,
        board_cols=args.cols,
        input_channels=input_channels,
    )


if __name__ == "__main__":
    main()
