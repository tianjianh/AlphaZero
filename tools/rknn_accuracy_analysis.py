#!/usr/bin/env python3
"""Drive rknn-toolkit2's accuracy_analysis() to find which layer overflows
in fp16 and (when a board is connected) which layer's hardware output
diverges from the simulator.

Per RKNN SDK Guide §7.1.1: an `inf` in the `simulator_error` column of
the analysis report indicates fp16 overflow on the simulator.  A
discrepancy between `simulator_error` and `runtime_error` indicates
on-device divergence — the artefact §7.2 of the guide says to attach
to a Rockchip bug report.

Usage on x86 host (simulator-only, no board needed):
    python tools/rknn_accuracy_analysis.py \\
        --onnx  models/kata1-b10c128.rknn.bs1.onnx \\
        --inputs calib/kata1/state_spatial_0050.npy \\
                 calib/kata1/state_global_0050.npy \\
        --output-dir snapshot/kata1_bs1_sim

Usage with a board connected via USB-ADB (reports both simulator AND
runtime error per layer — finds where hardware diverges):
    python tools/rknn_accuracy_analysis.py \\
        --onnx  models/kata1-b10c128.rknn.bs1.onnx \\
        --inputs calib/kata1/state_spatial_0050.npy \\
                 calib/kata1/state_global_0050.npy \\
        --target rk3576 \\
        --output-dir snapshot/kata1_bs1_hw

The output-dir contains per-layer fp32/quant snapshots and an analysis
table; the script also prints a digest pointing at the first overflowing
or divergent layer (the diagnostic Rockchip wants in a bug report).
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import List, Optional


def _find_analysis_table(output_dir: str) -> Optional[Path]:
    """The toolkit emits a CSV-ish summary at `<output_dir>/analysis.txt`
    in v2.3.x, or under various per-version filenames.  Find any sane
    candidate."""
    p = Path(output_dir)
    for name in ("snapshot.txt", "analysis.txt", "error_analysis.txt"):
        candidate = p / name
        if candidate.exists():
            return candidate
    # Fallback: any .txt file at the root
    txts = sorted(p.glob("*.txt"))
    return txts[0] if txts else None


def main():
    ap = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--onnx", required=True, help="Source ONNX model (must "
                    "match the .rknn that was deployed).")
    ap.add_argument("--inputs", nargs="+", required=True,
                    help="Per-input .npy files (one per graph input). "
                         "For KataGo dual-input: state_spatial + state_global.")
    ap.add_argument("--target", default=None,
                    choices=["rk3562","rk3566","rk3568","rk3576","rk3588"],
                    help="Connect to this on-device target via USB-ADB and "
                         "compare simulator vs runtime per layer.  Without "
                         "this flag, the analysis runs simulator-only.")
    ap.add_argument("--device-id", default=None,
                    help="USB device ID (only when multiple boards attached).")
    ap.add_argument("--output-dir", default="snapshot",
                    help="Where to drop per-layer snapshots + the analysis "
                         "table (default ./snapshot).")
    ap.add_argument("--batch", type=int, default=1,
                    help="Static batch baked into the .rknn (default 1).")
    ap.add_argument("--quant-mode", default="fp16",
                    choices=["fp16","int8"],
                    help="What to compare against fp32.  fp16 if you're "
                         "diagnosing fp16 overflow (default); int8 for "
                         "post-quant accuracy analysis.")
    ap.add_argument("--dataset", default=None,
                    help="Calibration manifest (required for --quant-mode int8 "
                         "and recommended even for fp16 to exercise more "
                         "tensor ranges).")
    ap.add_argument("--quant-method", default="channel",
                    choices=["channel","layer"],
                    help="(int8 mode) per-layer quant uniformly hurts Go nets; "
                         "leave as 'channel'.")
    args = ap.parse_args()

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from onnx_to_rknn import (_detect_format, _resolve_input_shapes, _new_rknn,
                              _config)

    info = _detect_format(args.onnx)
    inputs, shapes = _resolve_input_shapes(info, args.batch)

    if len(args.inputs) != len(inputs):
        raise SystemExit(
            f"--inputs count ({len(args.inputs)}) must match ONNX input "
            f"count ({len(inputs)}: {inputs})")

    print(f"target_platform: {args.target or '(simulator only)'}")
    print(f"input_shapes:    {shapes}")
    print(f"output_dir:      {args.output_dir}")

    rknn = _new_rknn(verbose=True)
    try:
        # auto_hybrid is irrelevant here; we want plain fp16 (or plain int8)
        # so that accuracy_analysis sees the unmodified path.
        _config(rknn, args.target or "rk3576",
                optimization_level=3,
                quantized_method=args.quant_method,
                quantized_algorithm="normal",
                quantized_dtype="w8a8")
        if rknn.load_onnx(model=args.onnx, inputs=inputs,
                          input_size_list=shapes) != 0:
            raise SystemExit("load_onnx failed")
        if args.quant_mode == "fp16":
            ret = rknn.build(do_quantization=False)
        else:
            if not args.dataset:
                raise SystemExit("--quant-mode int8 requires --dataset")
            ret = rknn.build(do_quantization=True, dataset=args.dataset)
        if ret != 0:
            raise SystemExit("build failed")

        os.makedirs(args.output_dir, exist_ok=True)
        ret = rknn.accuracy_analysis(
            inputs=args.inputs,
            output_dir=args.output_dir,
            target=args.target,
            device_id=args.device_id,
        )
        if ret != 0:
            raise SystemExit("accuracy_analysis failed")
    finally:
        rknn.release()

    # ── Digest the analysis table ────────────────────────────────────
    table = _find_analysis_table(args.output_dir)
    if table is None:
        print(f"\n(no analysis table found in {args.output_dir} — open the "
              f"directory manually)")
        return
    print(f"\n=== analysis table: {table} ===\n")

    text = table.read_text()
    print(text)

    # Heuristic digest: any line with 'inf' or 'nan' in it is a smoking gun.
    print("\n=== digest ===")
    flags = []
    for i, line in enumerate(text.splitlines()):
        low = line.lower()
        if "inf" in low or "nan" in low:
            flags.append((i, line))
    if flags:
        print(f"⚠️  {len(flags)} line(s) mention 'inf' / 'nan' — these are the "
              f"layers SDK §7.1.1 says are overflowing in fp16:")
        for i, line in flags[:20]:
            print(f"  L{i}: {line}")
    else:
        print(f"  no 'inf' / 'nan' found in simulator analysis (the "
              f"x86 simulator doesn't reproduce hardware fp16 overflow — "
              f"see rknnissue_next.md).")
        if args.target is None:
            print(f"  Re-run with --target {info.get('target','rk3576')} "
                  f"on a USB-attached board to capture runtime_error.")


if __name__ == "__main__":
    main()
