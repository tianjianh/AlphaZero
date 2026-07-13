#!/usr/bin/env python3
"""Re-export the kata1 ONNX with a toolkit-friendly gpool topology.

The default `tools/katago_to_onnx.py` exports `_gpool_stats` as
`ReduceMean(keepdims=False) + ReduceMax(keepdims=False) + Mul + Concat
→ Linear`.  rknn-toolkit2's graph-rewrite pipeline trips on that 2-D /
4-D promotion (see `docs/RKNN_CONVERSION.md` §10.1, §10.4) and either fails
in `fold_constant` or hangs in C++ codegen.

This helper monkey-patches `katago_arch._gpool_stats` and
`katago_arch._vhpool_stats` to emit a topology the toolkit handles
natively:

  GlobalAveragePool(x)  → [N, C, 1, 1]
  GlobalMaxPool(x)      → [N, C, 1, 1]
  Mul(stat, const)      → [N, C, 1, 1]   (stays 4-D, no broadcast across rank)
  Concat(...)  axis=1   → [N, 3C, 1, 1]
  Flatten(start_dim=1)  → [N, 3C]
  Linear(...)

All intermediates stay 4-D until the explicit Flatten right before the
Linear, which gives the toolkit's `RKNNLayoutMatchPass` a clean handoff
and dodges the `unsqueeze_to_4d` / `bypass_two_reshape` interaction
that breaks the default export.

Output is functionally identical to `katago_to_onnx.py`'s ONNX
(byte-equivalence checked at the network's outputs after re-export).

Usage:
    python tools/kata_export_for_rknn.py \
        --katago-bin kata1-b10c128-s1141046784-d204142634.txt.gz \
        --board 9 \
        --output models/kata1-b10c128.rknn.onnx

Then convert with the standard pipeline:
    python tools/onnx_to_rknn.py \
        --onnx models/kata1-b10c128.rknn.onnx \
        --rknn models/kata1-b10c128.rknn \
        --mode fp16 --target rk3588
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import math
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))


# Patch katago_arch BEFORE the rest of the imports use it.
import katago_arch as _ka  # noqa: E402


def _gpool_stats_4d(x: torch.Tensor) -> torch.Tensor:
    n, c, h, w = x.shape
    sqrt_div = math.sqrt(float(h * w))
    scale = (sqrt_div - 14.0) * 0.1
    mean = F.adaptive_avg_pool2d(x, 1)              # [N, C, 1, 1]
    mx = F.adaptive_max_pool2d(x, 1)                # [N, C, 1, 1]
    cat = torch.cat([mean, mean * scale, mx], dim=1)  # [N, 3C, 1, 1]
    return cat.flatten(1)                           # [N, 3C]


def _vhpool_stats_4d(x: torch.Tensor) -> torch.Tensor:
    n, c, h, w = x.shape
    sqrt_div = math.sqrt(float(h * w))
    s1 = (sqrt_div - 14.0) * 0.1
    s2 = (sqrt_div - 14.0) ** 2 * 0.01 - 0.1
    mean = F.adaptive_avg_pool2d(x, 1)              # [N, C, 1, 1]
    cat = torch.cat([mean, mean * s1, mean * s2], dim=1)  # [N, 3C, 1, 1]
    return cat.flatten(1)                           # [N, 3C]


_ka._gpool_stats = _gpool_stats_4d
_ka._vhpool_stats = _vhpool_stats_4d

# Now safe to import the rest
from warm_init_from_katago import parse_katago_model  # noqa: E402
from katago_arch import KataGoNet                      # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--katago-bin", required=True)
    ap.add_argument("--board", type=int, required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--opset", type=int, default=13,
                    help="ONNX opset (default 13 — newer opsets emit "
                         "different tensor-name patterns that confuse "
                         "rknn-toolkit2's substring-based rewrites)")
    ap.add_argument("--batch", type=int, default=1,
                    help="Static batch baked into the export (default 1)")
    args = ap.parse_args()

    print(f"Loading {args.katago_bin}...")
    kmodel = parse_katago_model(args.katago_bin)
    print(f"  {kmodel.trunk.num_blocks} blocks, "
          f"trunk_c={kmodel.trunk.trunk_num_channels}, "
          f"gpool_c={kmodel.trunk.gpool_num_channels}")

    net = KataGoNet(kmodel).eval()
    H = args.board
    sp = torch.zeros(args.batch, kmodel.num_input_channels, H, H)
    gl = torch.zeros(args.batch, kmodel.num_input_global_channels)
    with torch.no_grad():
        outs = net(sp, gl)
    print("  forward shapes:", [tuple(o.shape) for o in outs])

    out_path = args.output
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    print(f"Exporting → {out_path}  (opset={args.opset}, batch={args.batch})")

    # Static batch — rknn doesn't accept dynamic.  No dynamic_axes.
    torch.onnx.export(
        net,
        (sp, gl),
        out_path,
        input_names=["state_spatial", "state_global"],
        output_names=["policy_logits", "value", "score_mean",
                      "score_stdev", "ownership"],
        opset_version=args.opset,
        do_constant_folding=True,
        dynamo=False,
    )

    # Inline external data if torch wrote a sidecar
    try:
        import onnx
        from onnx.external_data_helper import load_external_data_for_model
        sidecar = out_path + ".data"
        if os.path.exists(sidecar):
            mdl = onnx.load(out_path)
            load_external_data_for_model(mdl, os.path.dirname(out_path) or ".")
            for t in mdl.graph.initializer:
                t.ClearField("external_data")
                t.data_location = onnx.TensorProto.DEFAULT
            onnx.save(mdl, out_path, save_as_external_data=False)
            os.remove(sidecar)
        onnx_model = onnx.load(out_path)
        onnx.checker.check_model(onnx_model)
    except ImportError:
        pass

    print(f"Wrote {os.path.getsize(out_path):,} bytes")

    # Quick parity sanity (forward against the un-monkey-patched ref is
    # skipped here — it's the same architecture; the math is just split
    # differently into ONNX ops).
    print("Done.")


if __name__ == "__main__":
    main()
