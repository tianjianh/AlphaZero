#!/usr/bin/env python3
"""On-board parity check: feed canonical kata1 / MiniGo inputs through
both ONNX Runtime (CPU) and rknnlite (NPU), report saturation status and
per-output divergence.

Run this **on the aarch64 Rockchip board** (RK3576 / RK3588), not on the
x86 conversion host.  The toolkit's host simulator stores fp16 in fp32
internally and masks the on-NPU saturation bug documented in
`rknnissue.md`; only running on real hardware tells you whether a
`.rknn` actually works.

Setup:
    # On the board, in the alphazero conda env:
    pip install rknn-toolkit-lite2==2.3.2 onnxruntime numpy

Usage:
    python tools/rknn_onboard_test.py \\
        --onnx models/kata1-b10c128.rknn.bs1.onnx \\
        --rknn models/kata1-b10c128.rk3576.bs1.rknn

Optional: pass several --rknn flags to compare multiple variants
(e.g. fp16 vs hybrid vs input-scaled) in one run.

Exit code:
    0  every .rknn passes saturation + parity checks
    1  at least one .rknn flagged
"""

from __future__ import annotations
import argparse
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np


# ── Canonical test inputs ───────────────────────────────────────────────
# Three fixtures that exercise the heads end-to-end:
#   1. empty board with realistic komi globals (matches kata1's training
#      input prior on move 1)
#   2. all-zero (probes whether biases alone saturate)
#   3. a mid-game position from the calibration set, if available
#
# For MiniGo single-input models we use just plane-0=on-board for the
# empty case.

def _katago_empty_board(H: int = 9, C: int = 22, G: int = 19,
                         komi: float = 6.5) -> Tuple[np.ndarray, np.ndarray]:
    sp = np.zeros((1, C, H, H), dtype=np.float32)
    gl = np.zeros((1, G), dtype=np.float32)
    sp[:, 0, :, :] = 1.0          # plane 0 = on-board
    self_komi = -komi if True else komi
    gl[:, 5] = self_komi / 20.0
    gl[:, 15] = -0.5              # parity wave for komi=6.5
    return sp, gl


def _katago_zero(H: int = 9, C: int = 22, G: int = 19) -> Tuple[np.ndarray, np.ndarray]:
    return (np.zeros((1, C, H, H), dtype=np.float32),
            np.zeros((1, G), dtype=np.float32))


def _minigo_empty_board(H: int = 9, C: int = 17) -> np.ndarray:
    s = np.zeros((1, C, H, H), dtype=np.float32)
    s[:, -1, :, :] = 1.0          # color plane = 1 (BLACK to play)
    return s


def _minigo_zero(H: int = 9, C: int = 17) -> np.ndarray:
    return np.zeros((1, C, H, H), dtype=np.float32)


# ── Saturation thresholds ───────────────────────────────────────────────
# Sane ranges for kata1 outputs on real positions (from rknnissue.md §3
# and §4 — the simulator gives values within these; saturated NPU runs
# blow past them).

_SANE_RANGES = {
    "policy_logits": (-30.0, 30.0),     # raw logits, no softmax
    "value":         (-1.001, 1.001),   # P(W)-P(L), strictly in [-1,+1]
    "score_mean":    (-100.0, 100.0),   # points; 9x9 caps around ±50
    "score_stdev":   (0.0, 50.0),       # softplus output, 9x9 max ~30
    "ownership":     (-0.001, 1.001),   # sigmoid in [0,1]
}


def _check_saturation(name: str, arr: np.ndarray) -> Tuple[bool, str]:
    a = np.asarray(arr, dtype=np.float32)
    if not np.all(np.isfinite(a)):
        return False, f"non-finite values (inf/nan) — count={int((~np.isfinite(a)).sum())}"
    lo_ok, hi_ok = _SANE_RANGES.get(name, (-1e9, 1e9))
    a_min, a_max = float(a.min()), float(a.max())
    if a_min < lo_ok or a_max > hi_ok:
        return False, f"out of range — got [{a_min:.3g}, {a_max:.3g}], expected ⊂ [{lo_ok}, {hi_ok}]"
    return True, f"[{a_min:.3g}, {a_max:.3g}]"


# ── ORT runner ──────────────────────────────────────────────────────────


def _run_ort(onnx_path: str):
    """Returns a callable run(feeds_dict) -> list of arrays in graph-output order."""
    import onnxruntime as ort
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    in_names = [i.name for i in sess.get_inputs()]
    out_names = [o.name for o in sess.get_outputs()]

    def run(*input_arrays):
        feeds = {n: a for n, a in zip(in_names, input_arrays)}
        return sess.run(None, feeds), out_names
    return run


def _run_rknn(rknn_path: str):
    """Returns a callable run(*inputs) -> (outputs, names_or_None)."""
    from rknnlite.api import RKNNLite
    r = RKNNLite()
    if r.load_rknn(rknn_path) != 0:
        raise SystemExit(f"rknn_load failed: {rknn_path}")
    if r.init_runtime() != 0:
        raise SystemExit(f"rknn_init_runtime failed: {rknn_path}")

    def run(*inputs):
        # rknnlite rejects `data_format=None` for 2-D inputs even though
        # the format string is meaningless for non-spatial buffers.
        # Pass "nchw" for every input — the runtime takes 2-D buffers
        # as-is regardless of the format tag.
        fmts = ["nchw"] * len(inputs)
        out = r.inference(inputs=list(inputs), data_format=fmts)
        return out, None
    return run, r


# ── Format detection (matches onnx_to_rknn._detect_format) ──────────────


def _detect_format(onnx_path: str):
    import onnx
    m = onnx.load(onnx_path)
    inputs = {i.name: [d.dim_value or d.dim_param
                       for d in i.type.tensor_type.shape.dim]
              for i in m.graph.input}
    outs = [o.name for o in m.graph.output]
    if "state_spatial" in inputs and "state_global" in inputs:
        sp_dims = inputs["state_spatial"]
        gl_dims = inputs["state_global"]
        # Baked batch: integer if static, else 1 (dynamic).
        baked_batch = sp_dims[0] if isinstance(sp_dims[0], int) else 1
        return dict(format="katago",
                    H=int(sp_dims[2]),
                    C=int(sp_dims[1]),
                    G=int(gl_dims[1]),
                    baked_batch=int(baked_batch),
                    output_names=outs)
    if "state" in inputs:
        sp_dims = inputs["state"]
        baked_batch = sp_dims[0] if isinstance(sp_dims[0], int) else 1
        return dict(format="minigo",
                    H=int(sp_dims[2]),
                    C=int(sp_dims[1]),
                    baked_batch=int(baked_batch),
                    output_names=outs)
    raise SystemExit(f"unknown ONNX layout: {list(inputs)}")


def _tile_to_batch(arr: np.ndarray, baked_batch: int) -> np.ndarray:
    """Tile a [1, ...]-shaped array along axis 0 up to `baked_batch`."""
    if arr.shape[0] == baked_batch:
        return arr
    if arr.shape[0] != 1:
        raise SystemExit(f"can't tile {arr.shape} to batch {baked_batch}")
    return np.broadcast_to(arr, (baked_batch,) + arr.shape[1:]).copy()


# ── One pass of comparison ──────────────────────────────────────────────


def compare_one(label: str, run_ort, run_rknn, inputs, output_names,
                tiled_batch: int = 1) -> bool:
    """tiled_batch>1 means we fed the same per-slot input replicated across
    `tiled_batch` lanes; in that case slot 0 should equal slot k for any
    deterministic NN. Slot-vs-slot disagreement on identical inputs is a
    per-lane codegen bug (see rknnissue_next.md §3)."""
    ort_outs, _ = run_ort(*inputs)
    rk_outs, _ = run_rknn(*inputs)

    print(f"\n  Test: {label}")
    print(f"  {'output':16s}  {'sane?':5s}  {'ORT range':25s}  {'RKNN range':25s}  max-abs-diff")
    print(f"  {'-'*16}  {'-'*5}  {'-'*25}  {'-'*25}  -----------")

    all_ok = True
    for name, ort_a, rk_a in zip(output_names, ort_outs, rk_outs):
        ort_a = np.asarray(ort_a, dtype=np.float32)
        rk_a = np.asarray(rk_a, dtype=np.float32)

        ok, rk_status = _check_saturation(name, rk_a)
        ort_lo, ort_hi = float(ort_a.min()), float(ort_a.max())
        ort_range = f"[{ort_lo:.3g}, {ort_hi:.3g}]"

        if rk_a.shape != ort_a.shape:
            print(f"  {name:16s}  shape mismatch: ORT {ort_a.shape} vs RKNN {rk_a.shape}")
            all_ok = False
            continue

        if np.all(np.isfinite(rk_a)) and np.all(np.isfinite(ort_a)):
            diff = float(np.abs(ort_a - rk_a).max())
        else:
            diff = float("inf")

        flag = "OK   " if ok else "FAIL "
        print(f"  {name:16s}  {flag}  {ort_range:25s}  {rk_status:25s}  {diff:.3e}")
        if not ok:
            all_ok = False

    # ── Per-slot consistency check ───────────────────────────────────
    # If we tiled the same input across all batch lanes, slot 0 should
    # equal slot k for any deterministic NN.  Disagreement here is the
    # bs=4 per-lane codegen bug from rknnissue_next.md §3.
    if tiled_batch > 1:
        max_slot_diff = 0.0
        worst_name = None
        for name, rk_a in zip(output_names, rk_outs):
            rk_a = np.asarray(rk_a, dtype=np.float32)
            if rk_a.shape[0] != tiled_batch or not np.all(np.isfinite(rk_a)):
                continue
            slot0 = rk_a[0:1]
            for k in range(1, tiled_batch):
                d = float(np.abs(slot0 - rk_a[k:k+1]).max())
                if d > max_slot_diff:
                    max_slot_diff = d
                    worst_name = name
        consistent = max_slot_diff < 0.1
        flag = "OK   " if consistent else "FAIL "
        print(f"  per-slot:        {flag}  max slot-vs-slot diff = "
              f"{max_slot_diff:.3e}  on '{worst_name}'")
        if not consistent:
            all_ok = False

    return all_ok


def main():
    ap = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--onnx", required=True, help="Source ONNX (for ORT reference)")
    ap.add_argument("--rknn", action="append", required=True,
                    help="Compiled .rknn to test. Repeatable to compare variants.")
    args = ap.parse_args()

    info = _detect_format(args.onnx)
    bb = info["baked_batch"]
    print(f"ONNX format: {info['format']}, H={info['H']}, "
          f"baked batch={bb}, outputs={info['output_names']}")

    # Build fixtures (per-slot, batch-1 shapes — tiled below to baked_batch)
    if info["format"] == "katago":
        fixtures = [
            ("empty board (komi=6.5)", _katago_empty_board(info["H"], info["C"], info["G"], 6.5)),
            ("all-zero",                _katago_zero(info["H"], info["C"], info["G"])),
        ]
        calib_dir = Path("calib/kata1")
        if (calib_dir / "state_spatial_0050.npy").exists():
            sp = np.load(calib_dir / "state_spatial_0050.npy").astype(np.float32)[None]
            gl = np.load(calib_dir / "state_global_0050.npy").astype(np.float32)[None]
            fixtures.append(("mid-game (calib_0050)", (sp, gl)))
    else:
        fixtures = [
            ("empty board",  (_minigo_empty_board(info["H"], info["C"]),)),
            ("all-zero",     (_minigo_zero(info["H"], info["C"]),)),
        ]

    # Tile each fixture's inputs along axis 0 to match the baked batch.
    # If baked_batch > 1 we feed the same per-slot input to every lane;
    # the per-slot consistency check then compares slot 0 vs slots 1..k.
    if bb > 1:
        fixtures = [(label, tuple(_tile_to_batch(a, bb) for a in ins))
                    for label, ins in fixtures]
        print(f"  tiling fixtures to batch={bb} (slot-consistency check enabled)")

    run_ort = _run_ort(args.onnx)
    overall_ok = True
    for rknn_path in args.rknn:
        print(f"\n{'='*72}")
        print(f"Testing: {rknn_path}")
        print('='*72)
        try:
            run_rknn, handle = _run_rknn(rknn_path)
        except Exception as e:
            print(f"  failed to load: {e}")
            overall_ok = False
            continue
        try:
            this_ok = True
            for label, ins in fixtures:
                ok = compare_one(label, run_ort, run_rknn, ins,
                                 info["output_names"], tiled_batch=bb)
                this_ok &= ok
            print(f"\n  → {rknn_path}: {'PASS' if this_ok else 'FAIL'}")
            overall_ok &= this_ok
        finally:
            try:
                handle.release()
            except Exception:
                pass

    print(f"\n{'='*72}")
    print(f"Overall: {'PASS' if overall_ok else 'FAIL — at least one .rknn flagged'}")
    sys.exit(0 if overall_ok else 1)


if __name__ == "__main__":
    main()
