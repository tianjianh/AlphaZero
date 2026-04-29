#!/usr/bin/env python3
"""Numerical sanity check for the A733 NPU conversion.

Loads the source ONNX into both ONNX Runtime (ground truth) and
acuitylib's Inference path (which models VIP9000 fp16 dataflow), feeds
canonical KataGo positions, and prints per-output max-abs-diff.

Usage:
    python tools/a733_verify.py \\
        --onnx models/kata1-b10c128.a733.bs1.unshared.onnx
"""

from __future__ import annotations

import argparse
import os
import numpy as np

# Set NPU target before any acuitylib import so the inference path
# matches the codegen target.
os.environ.setdefault("VSIMULATOR_SHADER_CORE_COUNT", "1")
os.environ.setdefault("VSIMULATOR_CONFIG", "VIP9000NANODI_PLUS_PID0X1000003B")


def _detect_io(onnx_path):
    import onnx
    m = onnx.load(onnx_path)
    inputs = [(x.name, [d.dim_value for d in x.type.tensor_type.shape.dim])
              for x in m.graph.input]
    outputs = [o.name for o in m.graph.output]
    return inputs, outputs


def _make_fixtures(input_specs, seed=0):
    """Three canonical KataGo fixtures: empty board, all-zero, mid-game."""
    rng = np.random.default_rng(seed)
    spatial_name, spatial_shape = input_specs[0]
    global_name, global_shape = input_specs[1]
    fixtures = []
    # 1) all-zero (empty board)
    sp_zero = np.zeros(spatial_shape, dtype=np.float32)
    gl_zero = np.zeros(global_shape, dtype=np.float32)
    fixtures.append(("all_zero", {spatial_name: sp_zero, global_name: gl_zero}))
    # 2) "empty board" — V7 plane 0 = empty (1 everywhere), plane 18 = komi
    sp_empty = np.zeros(spatial_shape, dtype=np.float32)
    sp_empty[0, 0, :, :] = 1.0
    gl_empty = np.zeros(global_shape, dtype=np.float32)
    gl_empty[0, 5] = 0.5  # komi-like
    fixtures.append(("empty_board", {spatial_name: sp_empty, global_name: gl_empty}))
    # 3) mid-game — sparse stones, random globals
    sp_mid = np.zeros(spatial_shape, dtype=np.float32)
    H, W = spatial_shape[-2], spatial_shape[-1]
    for _ in range(H * W // 4):
        ch = int(rng.integers(1, 4))
        i, j = int(rng.integers(0, H)), int(rng.integers(0, W))
        sp_mid[0, ch, i, j] = 1.0
    gl_mid = rng.standard_normal(global_shape).astype(np.float32) * 0.3
    fixtures.append(("mid_game", {spatial_name: sp_mid, global_name: gl_mid}))
    return fixtures


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    inputs, outputs = _detect_io(args.onnx)
    print(f"[verify] inputs: {inputs}")
    print(f"[verify] outputs: {outputs}")

    fixtures = _make_fixtures(inputs, seed=args.seed)

    # ── ONNX Runtime ground truth ───────────────────────────────────────
    import onnxruntime as ort
    ort_sess = ort.InferenceSession(args.onnx, providers=["CPUExecutionProvider"])
    ort_outs = []
    for name, feed in fixtures:
        out = ort_sess.run(None, feed)
        ort_outs.append((name, out))

    # ── Acuity simulator path (VIP9000 fp16 modeling) ───────────────────
    from acuitylib.interface.importer import OnnxLoader
    from acuitylib.interface.inference import Inference

    print("[verify] loading ONNX into Acuity for inference...")
    model = OnnxLoader(args.onnx).load(
        inputs=[n for n, _ in inputs],
        input_size_list=[s for _, s in inputs],
        outputs=outputs,
    )
    inf = Inference(model)
    inf.build_session()

    # ── Compare ─────────────────────────────────────────────────────────
    print()
    print(f"{'fixture':<14} {'output':<16} {'max_abs_diff':>14}")
    print(f"{'-'*14} {'-'*16} {'-'*14}")
    for (name, feed), (_, ort_out) in zip(fixtures, ort_outs):
        # Acuity wants inputs in IR order; pass as a list of arrays
        sim_pair = inf.run_session([feed[n] for n, _ in inputs])
        # run_session returns [(input0, input1), (output0, output1, ...)]
        sim_out = list(sim_pair[1])
        for out_name, ort_t, sim_t in zip(outputs, ort_out, sim_out):
            ort_arr = np.asarray(ort_t).astype(np.float32)
            sim_arr = np.asarray(sim_t).astype(np.float32)
            if ort_arr.shape != sim_arr.shape:
                # Acuity NCHW/NHWC layouts can mismatch; flatten compare
                ort_arr = ort_arr.reshape(-1)
                sim_arr = sim_arr.reshape(-1)
            d = float(np.abs(ort_arr - sim_arr).max())
            tag = "OK   " if d < 5e-3 else ("WARN " if d < 5e-2 else "FAIL ")
            print(f"{name:<14} {out_name:<16} {d:>14.3e}  {tag}")
    print()
    print("[verify] OK<5e-3 — fp16 noise floor.  WARN<5e-2 tolerable for fp16 path.")


if __name__ == "__main__":
    main()
