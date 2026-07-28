#!/usr/bin/env python3
"""Quantise an exported fp32 ONNX model to INT8 for the K3 backend.

Why INT8
--------
The K3's A100 Tensor Cores peak at 15 TOPS dense INT8 vs 7.5 TFLOPS
fp16 (K3 paper, Table 1 — the INT8 row is annotated "accelerates
convolution").  Measured on-board, b10c128 9x9 at batch 16:

    fp32   22.7 evals/s      (RVV vector unit — no Tensor Core path)
    fp16 1954   evals/s
    int8 2982   evals/s      (3287 at batch 32)

and ResNet50 batch 1 / 4 threads reaches 151 inf/s, against the 129
the vendor reports in the paper's Table 8.

Four traps, all of them silent
------------------------------
1. **Fold BatchNorm first.**  Exported graphs keep BN as its own node.
   Quantising without folding leaves every conv wrapped in
   dequantise -> BN -> requantise, and the model then runs at exactly
   fp16 speed while looking like int8.  `quant_pre_process` folds it.
   Pass --skip_symbolic_shape: the global-pooling heads' Shape/Concat
   gymnastics crash symbolic inference ('NoneType' is not iterable).

2. **Weights must be SIGNED int8.**  With QUInt8 the EP refuses the
   graph outright:
       cannot find kernel config for this vlen 1024 and weight type u8
   There is no u8 kernel for VLEN=1024.  Use QuantType.QInt8.

3. **QDQ format, not QOperator.**  QLinearConv is not implemented by
   the EP — a QOperator graph lands 100% on the CPU EP and runs ~15x
   SLOWER than fp16.  QDQ nodes get absorbed into the fused subgraph.

4. **Keep the batch axis symbolic.**  The calibrator bakes the
   calibration batch into every value_info; the result is a graph
   frozen at batch 1.  We strip value_info and restore the symbolic
   axis afterwards.

We also re-attach the `_sd_` state_dict tensors, which the quantiser
drops as unused initializers — LoadedModel reads them to identify the
architecture, so without them every binary fails with "Unknown model
format".

Calibration
-----------
--calib DIR reads .npy files (one encoded state each, shape
[C,H,W] or [C*H*W]); dump them with `build/encode_dump`.  Without it we
fall back to synthetic states, which is fine for benchmarking but NOT
for play: the activation scales come out wrong.  per-channel is not
supported by the EP (the graph fails to compile), so we stay
per-tensor.

Usage
-----
    python3 tools/onnx_to_int8.py models/best.onnx --calib calib_states/
    python3 tools/onnx_to_int8.py models/best.onnx -o out.onnx
"""

import argparse
import glob
import os
import subprocess
import sys
import tempfile

import numpy as np
import onnx

try:
    from onnxruntime.quantization import (CalibrationDataReader, QuantFormat,
                                          QuantType, quantize_static)
except ImportError:
    sys.exit("needs onnxruntime with the quantization tools installed")

SD_PREFIX = "_sd_"


class StateReader(CalibrationDataReader):
    """Feeds encoded board states to the calibrator, one batch-1 dict at a time."""

    def __init__(self, input_name, shape, calib_dir=None, count=64):
        c, h, w = shape
        if calib_dir:
            files = sorted(glob.glob(os.path.join(calib_dir, "*.npy")))[:count]
            if not files:
                sys.exit(f"no .npy files in {calib_dir}")
            states = [np.load(f).astype(np.float32).reshape(1, c, h, w) for f in files]
            print(f"  calibrating on {len(states)} real states from {calib_dir}")
        else:
            print("  WARNING: no --calib dir; using synthetic states.  Fine for "
                  "benchmarking, wrong scales for play.")
            rng = np.random.default_rng(0)
            # Board planes are mostly one-hot/binary — mimic that occupancy
            # rather than uniform noise, or the activation scales blow up.
            states = [(rng.random((1, c, h, w)) > 0.7).astype(np.float32)
                      for _ in range(count)]
        self._it = iter([{input_name: s} for s in states])

    def get_next(self):
        return next(self._it, None)


def spatial_input(model):
    for inp in model.graph.input:
        dims = inp.type.tensor_type.shape.dim
        if len(dims) == 4:
            return inp.name, (dims[1].dim_value, dims[2].dim_value, dims[3].dim_value)
    sys.exit("no 4-D spatial input found")


def restore_dynamic_batch(model):
    del model.graph.value_info[:]           # calibration-baked shapes
    for t in list(model.graph.input) + list(model.graph.output):
        d = t.type.tensor_type.shape.dim[0]
        d.ClearField("dim_value")
        d.dim_param = "batch"


def reattach_metadata(model, source_path):
    have = {i.name for i in model.graph.initializer}
    src = onnx.load(source_path)
    n = 0
    for init in src.graph.initializer:
        if init.name.startswith(SD_PREFIX) and init.name not in have:
            model.graph.initializer.append(init)
            n += 1
    return n


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model")
    ap.add_argument("-o", "--output")
    ap.add_argument("--calib", help="directory of .npy encoded states")
    ap.add_argument("--count", type=int, default=64)
    args = ap.parse_args()

    src = args.model
    dst = args.output or (
        (src[:-len(".onnx")] if src.endswith(".onnx") else src) + ".int8.onnx")

    with tempfile.TemporaryDirectory() as tmp:
        pre = os.path.join(tmp, "pre.onnx")
        print("[1/4] folding BatchNorm (quant_pre_process)")
        subprocess.run(
            [sys.executable, "-m", "onnxruntime.quantization.preprocess",
             "--input", src, "--output", pre,
             "--skip_symbolic_shape", "True"],
            check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        name, shape = spatial_input(onnx.load(pre))
        print(f"[2/4] quantising (QDQ, signed int8, per-tensor), input {name} {shape}")
        q = os.path.join(tmp, "q.onnx")
        quantize_static(
            pre, q, StateReader(name, shape, args.calib, args.count),
            quant_format=QuantFormat.QDQ,
            activation_type=QuantType.QInt8,
            weight_type=QuantType.QInt8,   # u8 has no VLEN=1024 kernel
            per_channel=False)             # per-channel fails to compile on the EP

        model = onnx.load(q)

    print("[3/4] restoring symbolic batch axis")
    restore_dynamic_batch(model)

    print("[4/4] re-attaching _sd_ metadata")
    n = reattach_metadata(model, src)

    onnx.checker.check_model(model)
    onnx.save(model, dst)
    print(f"wrote {dst}  ({n} _sd_ tensors re-attached)")
    print(f"  {os.path.getsize(src) / 1e6:.1f} MB -> {os.path.getsize(dst) / 1e6:.1f} MB")
    print("  NOTE: the K3 backend compiles one fixed batch — set --max-batch "
          "to the batch you will actually run (16-32 is the sweet spot).")


if __name__ == "__main__":
    main()
