#!/usr/bin/env python3
"""ONNX → Allwinner A733 (Cubie A7A) NPU NBG converter using ACUITY Toolkit.

The A733 carries a VeriSilicon VIP9000 NanoDI+ NPU.  Conversion runs offline
on x86 via the `acuitylib` Python API (pip-installable as `acuitylite`); the
resulting `.nb` (Network Binary Graph) is consumed on-device by Allwinner's
`awnn` / VIPLite runtime.

Pipeline (mirrors Allwinner's `pegasus_*` shell scripts):
    OnnxLoader.load(...)
      → (optional) Quantization.quantize('float16', 'float16')
      → OvxlibExporter.export(<dir>, dtype='float', pack_nbg_unify=True)

For fp16 we skip explicit quantization — the VIP9000 fp pipeline is fp16
natively, so an `--dtype float` build with no quantize step runs in fp16
on the NPU MAC array.

Usage:
    # 1) Re-export KataGo to ONNX with the toolkit-friendly gpool topology
    python tools/kata_export_for_rknn.py \
        --katago-bin kata1-b10c128-s1141046784-d204142634.txt.gz \
        --board 9 --batch 1 --opset 13 \
        --output models/kata1-b10c128.a733.bs1.onnx
    python -c "import onnx; from onnxsim import simplify; \
        m,_=simplify(onnx.load('models/kata1-b10c128.a733.bs1.onnx')); \
        onnx.save(m,'models/kata1-b10c128.a733.bs1.onnx')"

    # 2) Convert to NBG for A733
    python tools/onnx_to_a733.py \
        --onnx models/kata1-b10c128.a733.bs1.onnx \
        --output-dir models/kata1-b10c128.a733.bs1.fp16 \
        --mode fp16

The NBG file ends up at:
    <output-dir>/<base>_nbg/network_binary.nb
"""

from __future__ import annotations

import argparse
import os
import sys
import shutil
from pathlib import Path

# Set NPU target for A733 (NPU v3 = VIP9000 NanoDI+) before any acuitylib
# import — these env vars influence simulator + codegen.
os.environ.setdefault("VSIMULATOR_SHADER_CORE_COUNT", "1")
os.environ.setdefault("VSIMULATOR_CONFIG", "VIP9000NANODI_PLUS_PID0X1000003B")

import onnx  # noqa: E402


def _detect_io(onnx_path: str):
    """Return (input_names, input_shapes, output_names) from the ONNX graph.

    Static-batch ONNX is required (the NPU codegen needs a fixed shape).
    """
    m = onnx.load(onnx_path)
    inputs = []
    shapes = []
    for x in m.graph.input:
        dims = []
        for d in x.type.tensor_type.shape.dim:
            if d.dim_value <= 0:
                raise SystemExit(
                    f"input '{x.name}' has dynamic dimension; A733 codegen "
                    f"needs static shape — re-export with a fixed batch.")
            dims.append(d.dim_value)
        inputs.append(x.name)
        shapes.append(dims)
    outputs = [o.name for o in m.graph.output]
    return inputs, shapes, outputs


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--onnx", required=True, help="Source ONNX (static batch).")
    ap.add_argument("--output-dir", required=True,
                    help="Output directory; the .nb lands inside this dir.")
    ap.add_argument("--mode", choices=["fp16"], default="fp16",
                    help="Quantization mode (only fp16 supported here; the "
                         "VIP9000 fp pipeline is fp16-native).")
    ap.add_argument("--prefix", default=None,
                    help="Output prefix (default: ONNX basename).")
    ap.add_argument("--keep-workspace", action="store_true",
                    help="Don't strip intermediate ovxlib C code; useful for "
                         "inspecting the toolkit's generated graph.")
    args = ap.parse_args()

    onnx_path = os.path.abspath(args.onnx)
    out_dir = os.path.abspath(args.output_dir)
    prefix = args.prefix or Path(onnx_path).stem

    print(f"[a733] target: VIP9000 NanoDI+ (Allwinner A733)")
    print(f"[a733]   VSIMULATOR_CONFIG={os.environ['VSIMULATOR_CONFIG']}")
    print(f"[a733] source ONNX: {onnx_path}")
    print(f"[a733] output dir: {out_dir}")

    inputs, shapes, outputs = _detect_io(onnx_path)
    print(f"[a733] inputs: {list(zip(inputs, shapes))}")
    print(f"[a733] outputs: {outputs}")

    # acuitylib is heavy; import after env vars are set.
    from acuitylib.interface.importer import OnnxLoader
    from acuitylib.interface.exporter import OvxlibExporter

    print("[a733] loading ONNX into Acuity IR...")
    model = OnnxLoader(onnx_path).load(
        inputs=inputs,
        input_size_list=shapes,
        outputs=outputs,
    )

    os.makedirs(out_dir, exist_ok=True)
    output_prefix = os.path.join(out_dir, prefix)

    print(f"[a733] exporting (dtype=float, pack_nbg_only=True) → {output_prefix}_*")
    exporter = OvxlibExporter(model)
    # dtype='float' keeps weights fp32 in the IR; the NPU MAC executes them
    # in fp16 at runtime (VIP9000's primary FP path).
    # pack_nbg_only=True asks the toolkit to emit only the .nb file (Network
    # Binary Graph), bypassing the ovxlib C scaffolding generation that
    # needs --viv-sdk.  The .nb is what awnn / viplite loads on-device.
    exporter.export(
        output_prefix,
        dtype="float",
        target_ide_project="linux64",
        pack_nbg_only=True,
    )

    # The exporter writes a directory tree under output_prefix; locate the .nb
    found = []
    for root, _dirs, files in os.walk(out_dir):
        for f in files:
            if f.endswith(".nb"):
                found.append(os.path.join(root, f))
    if not found:
        raise SystemExit(
            f"[a733] no .nb produced under {out_dir} — toolkit failed silently?")

    print(f"[a733] produced {len(found)} NBG file(s):")
    for p in found:
        print(f"  {p}  ({os.path.getsize(p):,} bytes)")

    if not args.keep_workspace:
        # Strip the generated C-code workspace, keep only the .nb and the
        # accompanying nbg_meta.json (input/output names, shapes, dtypes —
        # the on-board awnn loader needs these).
        keep = (".nb", "nbg_meta.json")
        for root, dirs, files in list(os.walk(out_dir, topdown=False)):
            for f in files:
                if not any(f.endswith(s) for s in keep):
                    os.remove(os.path.join(root, f))
            for d in dirs:
                p = os.path.join(root, d)
                try:
                    os.rmdir(p)
                except OSError:
                    pass

    print("[a733] done.")


if __name__ == "__main__":
    main()
