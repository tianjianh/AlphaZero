#!/usr/bin/env python3
"""Math-equivalent ONNX rewrites for the kata1 fp16-on-RK3576/3588
saturation bug documented in `rknnissue.md`.

Two transformations are available, individually or combined:

  --input-scale N         Halve the dynamic range visible to the trunk's
                          first-line ops without changing the network's
                          output.  Inserts `Mul(input, 1/N)` immediately
                          before each `Conv`/`MatMul` that consumes a
                          graph input, and multiplies that op's weight
                          tensor by N.  The forward pass is bit-identical
                          (modulo fp16 rounding); the toolkit's fp16
                          codegen, however, sees a different graph and
                          may schedule the MAC array differently — worth
                          trying as a black-box workaround when fp16
                          saturates on hardware.

  --unshare-initializers  Give each multi-consumer initializer its own
                          dedicated copy.  PyTorch ONNX export dedupes
                          identical-value tensors (e.g. kata1's
                          BN.weight=1 / running_var=1 are both the same
                          all-ones initializer used by 22 BN nodes); the
                          toolkit's fp16 codegen may handle these
                          aliases inconsistently.  Un-sharing produces a
                          larger ONNX (one copy per consumer slot) but
                          forward output is bit-identical, and the
                          downstream `.rknn` build sees a clean graph.

Both transformations preserve ORT output bit-exactly.  The intent is to
nudge the toolkit's codegen path without changing the model.

Usage:
    python tools/onnx_rknn_mitigations.py \\
        --input  models/kata1-b10c128.rknn.bs1.onnx \\
        --output models/kata1-b10c128.rknn.bs1.scaled2.onnx \\
        --input-scale 2 --unshare-initializers

Then re-convert the new ONNX:
    python tools/onnx_to_rknn.py \\
        --onnx models/kata1-b10c128.rknn.bs1.scaled2.onnx \\
        --rknn models/kata1-b10c128.rk3576.bs1.scaled2.rknn \\
        --mode fp16 --target rk3576

And run `tools/rknn_onboard_test.py` on the board to check whether the
saturation cleared.
"""

from __future__ import annotations

import argparse
import os
from typing import Dict, List, Tuple

import numpy as np
import onnx
from onnx import helper, numpy_helper


# ────────────────────────────────────────────────────────────────────────
#  Mitigation 1 — input scaling (math-equivalent)
# ────────────────────────────────────────────────────────────────────────


def apply_input_scale(model: onnx.ModelProto, scale: float) -> onnx.ModelProto:
    """Insert Mul(input, 1/scale) before each Conv/MatMul that consumes a
    graph input, and multiply that op's weight initializer by `scale`.

    The forward pass is unchanged: input scaled by 1/scale × weight
    scaled by scale → product unchanged.  But the graph topology
    differs, which can shake out fp16 codegen issues.

    Constraints:
      * Only applies to Conv / MatMul whose **first input** is a graph
        input (not an intermediate tensor).  Other ops are left alone.
      * Weight tensor must be an initializer (not a runtime-computed
        tensor) — for kata1 / MiniGo this is always true.
    """
    g = model.graph
    inits = {i.name: i for i in g.initializer}
    graph_inputs = {i.name for i in g.input}

    # Find all Conv / MatMul nodes whose first input is a graph input
    targets = [n for n in g.node
               if n.op_type in ("Conv", "MatMul")
               and n.input[0] in graph_inputs
               and len(n.input) >= 2
               and n.input[1] in inits]

    if not targets:
        print("[input-scale] no input-consuming Conv/MatMul nodes found — no-op")
        return model

    # Per graph input, insert one Mul(input, 1/scale) shared across all
    # downstream Conv/MatMul.  This produces a cleaner graph than one
    # Mul per consumer.
    inv_scale = 1.0 / float(scale)
    inv_scale_init_name = "__rknn_input_scale_inv"
    inv_scale_init = numpy_helper.from_array(
        np.array([inv_scale], dtype=np.float32), name=inv_scale_init_name
    )
    g.initializer.append(inv_scale_init)

    new_nodes = []
    rewired_inputs: Dict[str, str] = {}    # original input name -> scaled tensor name
    for target_node in targets:
        orig_input = target_node.input[0]
        weight_name = target_node.input[1]

        # Insert one Mul per unique graph input
        if orig_input not in rewired_inputs:
            scaled_name = f"{orig_input}__scaled_by_{scale}"
            mul_node = helper.make_node(
                "Mul",
                inputs=[orig_input, inv_scale_init_name],
                outputs=[scaled_name],
                name=f"__rknn_input_scale_{orig_input.replace('/', '_').replace('.', '_')}",
            )
            new_nodes.append(mul_node)
            rewired_inputs[orig_input] = scaled_name

        # Re-wire the Conv/MatMul to consume the scaled input
        target_node.input[0] = rewired_inputs[orig_input]

        # Multiply the weight initializer in place
        w_init = inits[weight_name]
        w = numpy_helper.to_array(w_init).copy()
        w *= float(scale)
        new_w_init = numpy_helper.from_array(w, name=weight_name)
        # Replace by name
        for i, x in enumerate(g.initializer):
            if x.name == weight_name:
                g.initializer.remove(x)
                g.initializer.insert(i, new_w_init)
                inits[weight_name] = new_w_init
                break

    # Insert all Mul nodes at the front of the graph (topological-correct)
    old = list(g.node)
    g.ClearField("node")
    g.node.extend(new_nodes)
    g.node.extend(old)

    print(f"[input-scale] scale={scale}, inserted {len(new_nodes)} Mul nodes, "
          f"rescaled {len(targets)} weight tensors by ×{scale}")
    return model


# ────────────────────────────────────────────────────────────────────────
#  Mitigation 2 — un-share multi-consumer initializers
# ────────────────────────────────────────────────────────────────────────


def apply_unshare_initializers(model: onnx.ModelProto) -> onnx.ModelProto:
    """Give every (consumer-node, slot) pair its own private copy of any
    initializer it shares with another consumer.

    Forward output is bit-identical (data is duplicated, not modified).
    The graph grows by ~5-10× the size of multi-consumer initializers
    (kata1 grows from 12 MB to ~13 MB).
    """
    g = model.graph
    inits = {i.name: i for i in g.initializer}

    # Collect (node, slot) per initializer use
    occurrences: Dict[str, List[Tuple["onnx.NodeProto", int]]] = {}
    for n in g.node:
        for slot, inp in enumerate(n.input):
            if inp in inits:
                occurrences.setdefault(inp, []).append((n, slot))

    new_inits = []
    n_split = 0
    n_aliases = 0
    for name, occs in occurrences.items():
        if len(occs) <= 1:
            continue
        n_split += 1
        orig = inits[name]
        # First occurrence keeps the original; alias each subsequent
        for i, (node, slot) in enumerate(occs[1:], 1):
            new_name = f"{name}__rknn_alias_{i}"
            new_init = onnx.TensorProto()
            new_init.CopyFrom(orig)
            new_init.name = new_name
            new_inits.append(new_init)
            node.input[slot] = new_name
            n_aliases += 1
    g.initializer.extend(new_inits)
    print(f"[unshare] split {n_split} multi-consumer initializers into "
          f"{n_split + n_aliases} private copies")
    return model


# ────────────────────────────────────────────────────────────────────────
#  CLI
# ────────────────────────────────────────────────────────────────────────


def main():
    ap = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input",  required=True, help="Source ONNX file")
    ap.add_argument("--output", required=True, help="Modified ONNX path")
    ap.add_argument("--input-scale", type=float, default=None,
                    help="Apply input-scaling mitigation with the given factor "
                         "(N=2 halves input dynamic range; recommended).")
    ap.add_argument("--unshare-initializers", action="store_true",
                    help="Give each multi-consumer initializer its own copy.")
    ap.add_argument("--no-verify", action="store_true",
                    help="Skip ORT forward-parity check after rewriting "
                         "(faster but less safe).")
    args = ap.parse_args()

    if args.input_scale is None and not args.unshare_initializers:
        raise SystemExit("must request at least one mitigation: "
                         "--input-scale and/or --unshare-initializers")

    model = onnx.load(args.input)

    if args.input_scale is not None:
        model = apply_input_scale(model, args.input_scale)
    if args.unshare_initializers:
        model = apply_unshare_initializers(model)

    # Strip stale value_info — shape inference may have leftover entries
    # that contradict the rewritten graph.
    model.graph.ClearField("value_info")

    onnx.save(model, args.output)
    print(f"wrote {args.output} ({os.path.getsize(args.output):,} bytes)")

    if not args.no_verify:
        # Forward parity check: feed the same random input to both ONNXes
        import onnxruntime as ort
        np.random.seed(0)

        # Build feeds matching the source ONNX's input shapes
        src = ort.InferenceSession(args.input, providers=["CPUExecutionProvider"])
        feeds = {}
        for i in src.get_inputs():
            shape = [1 if (s is None or isinstance(s, str) or s <= 0) else s
                     for s in i.shape]
            feeds[i.name] = np.random.randn(*shape).astype(np.float32) * 0.5
        src_out = src.run(None, feeds)

        dst = ort.InferenceSession(args.output, providers=["CPUExecutionProvider"])
        dst_out = dst.run(None, feeds)

        for sv, dv, info in zip(src_out, dst_out, src.get_outputs()):
            d = float(np.abs(sv - dv).max())
            tag = "OK   " if d < 5e-3 else "WARN "
            print(f"  parity {tag}{info.name:16s} max_abs_diff = {d:.3e}")


if __name__ == "__main__":
    main()
