#!/usr/bin/env python3
"""Parity test: confirm the exported KataGo ONNX matches its PyTorch
nn.Module counterpart on the same input.

Catches ONNX export bugs (missing ops, axis mistakes, constant folding
mishandling) but NOT architecture-port bugs — for that, compare against
upstream KataGo's own engine on the same position.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from parse_katago import parse_katago_model  # noqa: E402
from katago_arch import KataGoNet                      # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--katago-bin", required=True)
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--board", type=int, required=True)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tol", type=float, default=1e-4,
                    help="absolute tolerance per output (default 1e-4)")
    args = ap.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    print(f"Loading KataGo network from {args.katago_bin} ...")
    kmodel = parse_katago_model(args.katago_bin)
    net = KataGoNet(kmodel).eval()

    print(f"Building random batch (N={args.batch}, board={args.board}) ...")
    H = args.board
    sp = torch.randn(args.batch, kmodel.num_input_channels, H, H) * 0.5
    gl = torch.randn(args.batch, kmodel.num_input_global_channels) * 0.5

    # PyTorch reference
    print("Running PyTorch forward ...")
    with torch.no_grad():
        py_outs = net(sp, gl)
    py_outs = [o.detach().cpu().numpy() for o in py_outs]

    # ONNX Runtime
    print(f"Running ONNX Runtime on {args.onnx} ...")
    try:
        import onnxruntime as ort
    except ImportError as e:
        raise RuntimeError("onnxruntime is required for the parity test") from e

    sess = ort.InferenceSession(args.onnx, providers=["CPUExecutionProvider"])
    ort_outs = sess.run(
        ["policy_logits", "value", "score_mean", "score_stdev", "ownership"],
        {"state_spatial": sp.numpy(), "state_global": gl.numpy()},
    )

    names = ["policy_logits", "value", "score_mean", "score_stdev", "ownership"]
    ok = True
    for n, py, ort_arr in zip(names, py_outs, ort_outs):
        if py.shape != ort_arr.shape:
            print(f"  {n}: SHAPE MISMATCH py={py.shape} ort={ort_arr.shape}")
            ok = False
            continue
        diff = np.abs(py - ort_arr)
        max_d = float(diff.max())
        mean_d = float(diff.mean())
        flag = "OK" if max_d <= args.tol else "FAIL"
        print(f"  {n:14s}  shape={tuple(py.shape)}  "
              f"max_abs_diff={max_d:.6e}  mean_abs_diff={mean_d:.6e}  [{flag}]")
        if max_d > args.tol:
            ok = False

    # Sanity: empty board.  KataGo on 9x9 with 7.5 komi puts pass low and
    # plays near tengen / 4-4 / 3-3.  A faithful port produces a sensible
    # policy and a score_mean near -7.5 (white wins by komi from black's
    # perspective on truly empty play). This guards against architecture
    # port bugs that wouldn't show up in a self-consistency check.
    print("\n--- Empty-board sanity check (zero spatial, zero global) ---")
    sp0 = torch.zeros(1, kmodel.num_input_channels, H, H)
    gl0 = torch.zeros(1, kmodel.num_input_global_channels)
    with torch.no_grad():
        pl, v, sm, ssd, ow = net(sp0, gl0)
    pl = pl.numpy()[0]
    print(f"  value          = {v[0].item():.4f}")
    print(f"  score_mean     = {sm[0].item():.4f}")
    print(f"  score_stdev    = {ssd[0].item():.4f}")
    print(f"  pass logit     = {pl[H*H]:.4f}")
    print(f"  spatial top-5  : {[(int(i), float(pl[i])) for i in pl[:H*H].argsort()[-5:][::-1]]}")
    # Ownership average should be near 0.5 (uncertain) or shifted by komi.
    print(f"  ownership mean = {ow.mean().item():.4f}")
    print(f"  ownership min/max = {ow.min().item():.4f} / {ow.max().item():.4f}")

    if not ok:
        print("\nFAIL: PyTorch ↔ ONNX Runtime divergence above tolerance.")
        sys.exit(1)
    print("\nAll outputs match within tolerance. PyTorch ↔ ORT parity OK.")


if __name__ == "__main__":
    main()
