#!/usr/bin/env python3
"""Convert a KataGo .bin.gz / .txt.gz network to an ONNX file with
MiniGo-compatible output names.

Outputs (all post-processed in-graph):
  - policy_logits  [N, H*W + 1]      spatial(0..H*W-1) + pass(H*W)
  - value          [N, 1]            P(W) - P(L)
  - score_mean     [N, 1]            raw points (× 20 baked in)
  - score_stdev    [N, 1]            raw points (softplus + × 20 baked in)
  - ownership      [N, H*W]          [0, 1]  ((tanh + 1) / 2 baked in)

Inputs:
  - state_spatial  [N, 22, H, W]
  - state_global   [N, 19]

Usage:
    python tools/katago_to_onnx.py \
        --katago-bin /tmp/kata1-b10c128-s1141046784-d204142634.txt.gz \
        --board 9 --output models/kata1-b10c128.onnx
"""

import argparse
import os
import sys
from pathlib import Path

import torch

# Same-directory imports — make sure tools/ is on sys.path.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from warm_init_from_katago import parse_katago_model  # noqa: E402
from katago_arch import KataGoNet                      # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--katago-bin", required=True,
                    help=".bin.gz / .txt.gz file from media.katagotraining.org")
    ap.add_argument("--board", type=int, required=True,
                    help="Board edge length used at export time. The exported "
                         "ONNX has dynamic batch but a fixed H/W (KataGo's "
                         "graph has board-size-dependent constants in the "
                         "global-pool stages). Use the board size you'll run "
                         "inference at — engines for different sizes need "
                         "separate exports.")
    ap.add_argument("--output", required=True,
                    help="Output .onnx path")
    ap.add_argument("--opset", type=int, default=17,
                    help="ONNX opset version (default 17)")
    args = ap.parse_args()

    print(f"Loading KataGo network from {args.katago_bin} ...")
    kmodel = parse_katago_model(args.katago_bin)
    print(f"  name: {kmodel.name}")
    print(f"  model_version: {kmodel.model_version}")
    print(f"  spatial channels: {kmodel.num_input_channels}")
    print(f"  global channels: {kmodel.num_input_global_channels}")
    print(f"  trunk: {kmodel.trunk.num_blocks} blocks, "
          f"trunk_c={kmodel.trunk.trunk_num_channels}, "
          f"regular_c={kmodel.trunk.regular_num_channels}, "
          f"gpool_c={kmodel.trunk.gpool_num_channels}")
    print(f"  block kinds: {kmodel.trunk.block_kinds}")

    if kmodel.policy_head is None or kmodel.value_head is None:
        raise RuntimeError(
            "Parser did not retain policy/value head. Check warm_init "
            "extension (Phase 0.1) ran successfully.")

    print("Building KataGoNet ...")
    net = KataGoNet(kmodel).eval()
    n_params = sum(p.numel() for p in net.parameters())
    print(f"  total params: {n_params:,}")

    # Forward pass for shape validation
    H = args.board
    print(f"Running shape-check forward pass at board={H} ...")
    sp = torch.zeros(1, kmodel.num_input_channels, H, H)
    gl = torch.zeros(1, kmodel.num_input_global_channels)
    with torch.no_grad():
        outs = net(sp, gl)
    expected_shapes = [
        (1, H * H + 1),    # policy_logits
        (1, 1),            # value
        (1, 1),            # score_mean
        (1, 1),            # score_stdev
        (1, H * H),        # ownership
    ]
    names = ["policy_logits", "value", "score_mean", "score_stdev", "ownership"]
    for tensor, expected, name in zip(outs, expected_shapes, names):
        assert tuple(tensor.shape) == expected, (
            f"{name}: expected {expected}, got {tuple(tensor.shape)}")
    print("  shapes OK")

    # Export
    out_path = args.output
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    print(f"Exporting ONNX to {out_path} ...")
    # Use the legacy TorchScript exporter (`dynamo=False`). The new torch.export
    # exporter aggressively folds BatchNorm into the preceding Conv and reuses
    # the original conv-weight initializer name for the folded weight, which
    # collides with the embedded state_dict step below: the C++ Eigen backend
    # would end up reading folded weights and then double-applying mid_bn.
    torch.onnx.export(
        net,
        (sp, gl),
        out_path,
        input_names=["state_spatial", "state_global"],
        output_names=["policy_logits", "value", "score_mean",
                      "score_stdev", "ownership"],
        dynamic_axes={
            "state_spatial":  {0: "batch"},
            "state_global":   {0: "batch"},
            "policy_logits":  {0: "batch"},
            "value":          {0: "batch"},
            "score_mean":     {0: "batch"},
            "score_stdev":    {0: "batch"},
            "ownership":      {0: "batch"},
        },
        opset_version=args.opset,
        do_constant_folding=True,
        dynamo=False,
    )

    # PyTorch's exporter writes weights to an external `.onnx.data`
    # sidecar by default for large models. Inline those weights so the
    # ONNX file is self-contained (TensorRT can read either form, but
    # one file is simpler to distribute and cache).
    try:
        import onnx
        from onnx import numpy_helper
        from onnx.external_data_helper import load_external_data_for_model
        out_dir = os.path.dirname(out_path) or "."
        sidecar = out_path + ".data"
        if os.path.exists(sidecar):
            mdl = onnx.load(out_path)
            load_external_data_for_model(mdl, out_dir)
            for t in mdl.graph.initializer:
                t.ClearField("external_data")
                t.data_location = onnx.TensorProto.DEFAULT
            onnx.save(mdl, out_path, save_as_external_data=False)
            os.remove(sidecar)
            print(f"  inlined external weights ({os.path.getsize(out_path):,} bytes)")

        # Embed full state_dict so the C++ Eigen backend can read weights
        # by their PyTorch names. ONNX export folds BatchNorm into Conv
        # and may insert Identity passthroughs that collide with the
        # state_dict names, so prefix every embedded tensor with `_sd_`
        # to keep the optimized graph (TensorRT path) untouched. The C++
        # Eigen loader expects the same prefix.
        mdl = onnx.load(out_path)
        added = 0
        for name, tensor in net.state_dict().items():
            arr = tensor.detach().cpu().numpy()
            embedded_name = "_sd_" + name
            mdl.graph.initializer.append(
                numpy_helper.from_array(arr, name=embedded_name))
            added += 1
        onnx.save(mdl, out_path)
        print(f"  embedded {added} state_dict tensors for Eigen "
              f"(_sd_ prefix; {os.path.getsize(out_path):,} bytes total)")

        # Validate
        onnx_model = onnx.load(out_path)
        onnx.checker.check_model(onnx_model)
        print("  onnx.checker.check_model: OK")
    except ImportError:
        print("  WARN: onnx package unavailable, skipping checker")
    except Exception as e:
        raise RuntimeError(f"ONNX validation failed: {e}")

    print("Done.")


if __name__ == "__main__":
    main()
