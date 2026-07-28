#!/usr/bin/env python3
"""Convert an exported fp32 ONNX model to fp16 for the K3 backend.

Why this exists
---------------
The SpacemiT K3's A100 Tensor Cores implement int4 / int8 / fp16 /
bf16 / fp8 MMA — there is **no fp32 path** (K3 paper, Table 1).  An
fp32 graph still runs, but only on the RVV vector unit, which costs
roughly an order of magnitude.  Measured on-board:

    2048^3 GEMM   fp32  292 GFLOPS   |  1024^3 GEMM  fp16  2733 GFLOPS
    b10c128 9x9   fp32  22.7 evals/s |  b10c128 9x9  fp16  447 evals/s

Accuracy is not the tradeoff it sounds like: verified against the
PyTorch reference vectors, fp16 lands within ~6e-4 on every head
(scripts/make_test_vectors.py + build/verify), because the Tensor
Core accumulates fp16 inputs into fp32 (paper, Table 3).

The Cast fixup
--------------
`onnxconverter_common.float16` retypes tensors but leaves any
pre-existing `Cast` node's `to` attribute pointing at FLOAT, so the
model fails to load with:

    Type (tensor(float16)) of output arg (...) of node (.../Cast)
    does not match expected type (tensor(float))

MiniGo's score_stdev head emits exactly such Casts.  We reconcile each
Cast's `to` with the declared type of its output before saving.

Usage
-----
    python3 tools/onnx_to_fp16.py models/best.onnx            # -> models/best.fp16.onnx
    python3 tools/onnx_to_fp16.py models/best.onnx out.onnx
"""

import os
import sys
import warnings

import onnx

try:
    from onnxconverter_common import float16
except ImportError:
    sys.exit("needs onnxconverter-common:  pip install onnxconverter-common")


def convert(src, dst):
    warnings.filterwarnings("ignore")          # fp16 range warnings are expected
    model = onnx.load(src)

    # keep_io_types: the engine feeds/reads fp32 buffers (the encoder and
    # NNOutput are fp32); only the trunk's storage/compute goes fp16.
    model16 = float16.convert_float_to_float16(model, keep_io_types=True)

    declared = {}
    for vi in list(model16.graph.value_info) + list(model16.graph.output) \
            + list(model16.graph.input):
        declared[vi.name] = vi.type.tensor_type.elem_type

    fixed = 0
    for node in model16.graph.node:
        if node.op_type != "Cast":
            continue
        want = declared.get(node.output[0])
        if want is None:
            continue
        for attr in node.attribute:
            if attr.name == "to" and attr.i != want:
                attr.i = want
                fixed += 1

    onnx.checker.check_model(model16)
    onnx.save(model16, dst)
    return fixed


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__.strip().split("Usage\n-----\n")[-1])
    src = sys.argv[1]
    if len(sys.argv) > 2:
        dst = sys.argv[2]
    else:
        base = src[:-len(".onnx")] if src.endswith(".onnx") else src
        dst = base + ".fp16.onnx"

    fixed = convert(src, dst)
    print(f"wrote {dst}"
          f"{f'  (reconciled {fixed} Cast node(s))' if fixed else ''}")
    print(f"  {os.path.getsize(src) / 1e6:.1f} MB -> "
          f"{os.path.getsize(dst) / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
