#!/usr/bin/env python3
"""Convert a KataGo or MiniGo ONNX model to a Rockchip .rknn file.

Modes
-----
  fp16    → no quantisation, weights stored fp16. No calibration needed.
  hybrid  → trunk in int8, sensitive heads (softmax / value / score) in fp16.
            Requires a calibration manifest (see tools/rknn_calibration.py).
  int8    → full int8 quantisation. Requires a calibration manifest.

The hybrid path is the recommended config for Go on Rockchip NPUs: trunk
operators are ~95% of the FLOPs and survive int8 well, while a small numeric
error in policy / value / score logits becomes a large probability shift
after softmax — those layers stay fp16.

Auto-detection
--------------
We inspect the ONNX graph inputs to figure out which encoder layout the model
expects:

  * KataGo dual-input  (`state_spatial` + `state_global`)
  * MiniGo single-input (`state`)

`rknn.load_onnx` requires a fixed input shape, so we always materialise one
explicitly. Default batch is 1 (live play). For self-play, recompile with
`--batch 4` to pack multiple positions per `rknn_run` call.

Usage examples
--------------
    # 1) Fast fp16 conversion (no calibration)
    python tools/onnx_to_rknn.py \\
        --onnx models/kata1-b10c128.onnx \\
        --rknn models/kata1-b10c128.rknn \\
        --mode fp16 --target rk3588

    # 2) Hybrid int8 (trunk int8, heads fp16) — needs calibration
    python tools/rknn_calibration.py \\
        --onnx models/kata1-b10c128.onnx \\
        --output calib/kata1 --num-positions 200
    python tools/onnx_to_rknn.py \\
        --onnx models/kata1-b10c128.onnx \\
        --rknn models/kata1-b10c128.hybrid.rknn \\
        --mode hybrid --target rk3588 \\
        --dataset calib/kata1/dataset.txt

    # 3) Full int8 (max throughput, may degrade)
    python tools/onnx_to_rknn.py \\
        --onnx models/best.onnx \\
        --rknn models/best.int8.rknn \\
        --mode int8 --target rk3576 --dataset calib/best/dataset.txt

The .rknn file produced sits next to the .onnx (the C++ runtime resolves the
.rknn path by extension swap on `LoadedModel::model_path`).
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

# Same-directory imports
sys.path.insert(0, str(Path(__file__).resolve().parent))
from rknn_calibration import _detect_format  # noqa: E402


SUPPORTED_TARGETS = (
    "rk3562", "rk3566", "rk3568", "rk3576", "rk3588",
)


# ────────────────────────────────────────────────────────────────────────
#  Head-tensor discovery
# ────────────────────────────────────────────────────────────────────────


def _trace_head_tensors(onnx_path: str, output_names: Sequence[str],
                        depth: int) -> Tuple[Set[str], Dict[str, str]]:
    """For each output, walk back `depth` ops in the ONNX graph and collect
    every tensor name in the producer chain. Then **subtract the trunk**:
    tensors that appear in every output's chain are shared trunk layers,
    not head layers.

    Returns (head_tensor_names, tensor_to_op_type_map). The map lets the cfg
    patcher skip op types whose quantization scale the toolkit refuses to
    modify (ReduceMean / Reshape / Concat / etc.).

    The `custom_quantize_layers` entries in the rknn cfg use *output tensor
    names* of the rewritten graph (e.g. `add_613`, `softmax`). The original
    ONNX tensor names usually survive into the cfg, so the head set is the
    superset of names we flag as float16 to keep heads in high precision
    while leaving the trunk int8.
    """
    import onnx

    m = onnx.load(onnx_path)

    producer: Dict[str, "onnx.NodeProto"] = {}
    tensor_op_types: Dict[str, str] = {}
    for n in m.graph.node:
        for o in n.output:
            producer[o] = n
            tensor_op_types[o] = n.op_type

    chains: Dict[str, Set[str]] = {}
    for out_name in output_names:
        seen: Set[str] = set()
        frontier = [(out_name, 0)]
        while frontier:
            name, d = frontier.pop()
            if name in seen or d > depth:
                continue
            seen.add(name)
            n = producer.get(name)
            if n is None:
                continue
            for o in n.output:
                seen.add(o)
            for inp in n.input:
                if inp in producer:
                    frontier.append((inp, d + 1))
        chains[out_name] = seen

    # Heads = union of per-output chains, minus shared trunk.
    if not chains:
        return set(), tensor_op_types
    union: Set[str] = set()
    for s in chains.values():
        union |= s
    intersection = set.intersection(*chains.values()) if len(chains) >= 2 else set()
    head = union - intersection
    return head, tensor_op_types


# ────────────────────────────────────────────────────────────────────────
#  Hybrid cfg patching
# ────────────────────────────────────────────────────────────────────────


def _patch_hybrid_cfg(cfg_path: str, head_tensors: Set[str],
                      tensor_op_types: Optional[Dict[str, str]] = None) -> List[str]:
    """Edit the rknn-toolkit-generated quantization.cfg to force `head_tensors`
    (and any pattern-matched extras) into float16.

    The cfg is YAML with this structure:

        custom_quantize_layers:
          tensor_name_a: float16
        quantize_parameters:
          tensor_name_a:
            qtype: asym
            dtype: int8
            ...
          tensor_name_b: ...

    The set of valid `custom_quantize_layers` keys is exactly the keys of
    `quantize_parameters` — the toolkit rejects anything else with
    "Invalid operands name '<x>'". Original-ONNX initializer names (e.g.
    `val_22`, `policy_head.p2_conv.weight`) typically don't survive into
    the cfg's IR-level layer set, so we filter the traced head set to
    keep only names the cfg actually knows about.

    Two ways a layer ends up float16:
      1) traced from an ONNX graph output (intersection with cfg layers)
      2) substring match against head-related keywords (softmax / value /
         score_mean / score_stdev / ownership / policy_logits etc.)

    Outputs of the model often come pre-set to float32 by the toolkit; we
    leave those alone (they're effectively float32, which is finer than
    float16 already).

    Returns the list of layer names we forced to float16.
    """
    from ruamel.yaml import YAML

    yaml = YAML()
    yaml.preserve_quotes = True
    with open(cfg_path, "r") as f:
        cfg = yaml.load(f) or {}

    custom = cfg.get("custom_quantize_layers")
    if custom is None:
        cfg["custom_quantize_layers"] = {}
        custom = cfg["custom_quantize_layers"]

    quant_params = cfg.get("quantize_parameters", {}) or {}
    valid_layers = set(quant_params.keys())

    keywords = (
        "softmax", "ownership", "score_mean", "score_stdev",
        "value", "policy_logit", "tanh", "softplus", "policy_fc",
        "score_mean_head", "score_stdev_head", "value_head",
        "ownership_conv", "policy_conv",
    )

    # Toolkit-generated layer names use suffixes like `_sw`, `_mm`, `_rs`,
    # `_expand`, `_int8`, `_to_int8`, `_to_float16` — these are intermediate
    # IR tensors created by graph rewrites, not original ONNX outputs.
    # The toolkit refuses to override their dtype with
    #   "is not allowed to be modified!"
    # so we must skip them.
    TOOLKIT_SUFFIXES = (
        "_sw", "_mm", "_rs", "_rs_sw", "_expand", "_int8",
        "_to_int8", "_to_float16", "_float16",
    )

    def _is_toolkit_artifact(name: str) -> bool:
        if "#" in name:                              # `_rs#1`, `_rs#2`
            return True
        if "-rs" in name or "rs-" in name:           # `cat-rs`, `_rs-...`
            return True
        if "__" in name:                             # `getitem__cvt_*`
            return True
        for suf in TOOLKIT_SUFFIXES:
            if name.endswith(suf):
                return True
        return False

    # Op types we won't try to override. The toolkit's hybrid_quantization_step2
    # rejects modifying these layers' scale factors with "is not allowed to be
    # modified" — they're either reduction ops (ReduceMean/Max) whose output
    # scale is fixed by the quantization range, or shape ops that don't carry
    # numeric precision info themselves.
    UNMODIFIABLE_OP_TYPES = {
        "ReduceMean", "ReduceMax", "ReduceSum", "GlobalAveragePool",
        "GlobalMaxPool", "Reshape", "Squeeze", "Unsqueeze", "Transpose",
        "Concat", "Slice", "Gather", "Split",
    }

    def _is_head_layer(name: str) -> bool:
        # Skip layers the toolkit already declared float32 / float16 — forcing
        # them to float16 would either be a no-op or a precision downgrade.
        params = quant_params.get(name) or {}
        if params.get("dtype") in ("float32", "float16"):
            return False
        if _is_toolkit_artifact(name):
            return False
        # Skip op types whose scale parameter is locked.
        if tensor_op_types is not None:
            op = tensor_op_types.get(name, "")
            if op in UNMODIFIABLE_OP_TYPES:
                return False
        return True

    forced: Set[str] = set()

    # 1) Traced head tensors that are real cfg layers
    for name in head_tensors & valid_layers:
        if _is_head_layer(name):
            custom[name] = "float16"
            forced.add(name)

    # 2) Substring keyword sweep over the cfg's known layers
    for name in valid_layers:
        if not _is_head_layer(name):
            continue
        lname = str(name).lower()
        if any(k in lname for k in keywords):
            custom[name] = "float16"
            forced.add(str(name))

    cfg["custom_quantize_layers"] = custom
    with open(cfg_path, "w") as f:
        yaml.dump(cfg, f)
    return sorted(forced)


# ────────────────────────────────────────────────────────────────────────
#  RKNN driver
# ────────────────────────────────────────────────────────────────────────


def _resolve_input_shapes(info: Dict, batch: int) -> Tuple[List[str], List[List[int]]]:
    n = info["board_size"]
    if info["format"] == "katago":
        return (
            ["state_spatial", "state_global"],
            [
                [batch, info["input_channels"], n, n],
                [batch, info["input_global_channels"]],
            ],
        )
    return (
        ["state"],
        [[batch, info["input_channels"], n, n]],
    )


def _new_rknn(verbose: bool):
    from rknn.api import RKNN
    return RKNN(verbose=verbose)


def _config(rknn, target: str, optimization_level: int) -> None:
    rknn.config(
        target_platform=target,
        mean_values=None,           # features are already normalised
        std_values=None,
        optimization_level=optimization_level,
        # Toolkit rule-disables.
        #   * unsqueeze_to_4d_reshape_with_elementwise_op:
        #       The dual-input KataGo graph contains an unsqueeze→reshape
        #       pattern that the toolkit's default rule mis-rewrites under
        #       quantisation. Disabling restores correctness.
        #   * fuse_conv_gather:
        #       The KataGo policy head's `Conv → Slice` (channel selection
        #       for the side-to-move logits) gets pattern-matched into the
        #       toolkit's Conv+Gather fusion path, but the slicer's index
        #       isn't shape-typed in v8-v14 KataGo exports — the toolkit
        #       crashes inside `_p_fuse_conv_gather` with
        #       `TypeError: len() of unsized object`. Skipping the rule
        #       leaves the unfused Conv+Slice intact (no perf penalty —
        #       this op is on the policy head, <1% of FLOPs).
        disable_rules=[
            "unsqueeze_to_4d_reshape_with_elementwise_op",
            "fuse_conv_gather",
        ],
    )


def convert_fp16(onnx_path: str, rknn_path: str, target: str,
                 batch: int, optimization_level: int, verbose: bool) -> None:
    info = _detect_format(onnx_path)
    inputs, shapes = _resolve_input_shapes(info, batch)
    print(f"[fp16] target={target} input_shapes={shapes}")
    rknn = _new_rknn(verbose)
    try:
        _config(rknn, target, optimization_level)
        if rknn.load_onnx(model=onnx_path, inputs=inputs, input_size_list=shapes) != 0:
            raise RuntimeError("load_onnx failed")
        if rknn.build(do_quantization=False) != 0:
            raise RuntimeError("build failed")
        if rknn.export_rknn(rknn_path) != 0:
            raise RuntimeError("export_rknn failed")
        size = os.path.getsize(rknn_path)
        print(f"[fp16] wrote {rknn_path} ({size/1024:.1f} KB)")
    finally:
        rknn.release()


def convert_int8(onnx_path: str, rknn_path: str, target: str,
                 batch: int, dataset: str, optimization_level: int,
                 verbose: bool) -> None:
    if not os.path.exists(dataset):
        raise SystemExit(f"--dataset {dataset!r} does not exist")
    info = _detect_format(onnx_path)
    inputs, shapes = _resolve_input_shapes(info, batch)
    print(f"[int8] target={target} input_shapes={shapes} dataset={dataset}")
    rknn = _new_rknn(verbose)
    try:
        _config(rknn, target, optimization_level)
        if rknn.load_onnx(model=onnx_path, inputs=inputs, input_size_list=shapes) != 0:
            raise RuntimeError("load_onnx failed")
        if rknn.build(do_quantization=True, dataset=dataset) != 0:
            raise RuntimeError("build failed")
        if rknn.export_rknn(rknn_path) != 0:
            raise RuntimeError("export_rknn failed")
        size = os.path.getsize(rknn_path)
        print(f"[int8] wrote {rknn_path} ({size/1024:.1f} KB)")
    finally:
        rknn.release()


def convert_hybrid(onnx_path: str, rknn_path: str, target: str,
                   batch: int, dataset: str, proposal_dataset_size: int,
                   trace_depth: int, optimization_level: int,
                   keep_intermediates: bool, verbose: bool,
                   use_proposal: bool) -> None:
    """Two-step hybrid quantisation: trunk int8, heads fp16.

    Step 1: rknn.hybrid_quantization_step1(dataset, proposal=True).
        Generates `<base>.quantization.cfg`, `<base>.model`, `<base>.data`
        in a working directory.

    Step 2 (this tool): patch the cfg to force every head-related layer to
        float16. We do this by tracing back from each ONNX output for
        `--trace-depth` ops and collecting the producer-chain tensor names
        (KataGo/MiniGo cleanly localise their heads to ≤10 ops past the
        trunk, so depth=12 is plenty); plus a keyword sweep over any
        proposal layers that mention softmax/value/score/etc.

    Step 3: rknn.hybrid_quantization_step2(...) re-quantises with the
        patched cfg and exports the .rknn.
    """
    if not os.path.exists(dataset):
        raise SystemExit(f"--dataset {dataset!r} does not exist")
    info = _detect_format(onnx_path)
    inputs, shapes = _resolve_input_shapes(info, batch)

    # The two-step API drops files in CWD by default; isolate them in a
    # temp dir to avoid leaking generated state into the user's tree.
    workdir = tempfile.mkdtemp(prefix="rknn_hybrid_")
    print(f"[hybrid] target={target} input_shapes={shapes} dataset={dataset}")
    print(f"[hybrid] working dir: {workdir}")

    rknn = _new_rknn(verbose)
    try:
        _config(rknn, target, optimization_level)
        if rknn.load_onnx(model=onnx_path, inputs=inputs, input_size_list=shapes) != 0:
            raise RuntimeError("load_onnx failed")

        # Resolve dataset path to absolute (toolkit re-opens it from CWD)
        dataset_abs = os.path.abspath(dataset)

        old_cwd = os.getcwd()
        os.chdir(workdir)
        try:
            step1_kwargs = {"dataset": dataset_abs}
            if use_proposal:
                step1_kwargs.update(
                    proposal=True, proposal_dataset_size=proposal_dataset_size
                )
            ret = rknn.hybrid_quantization_step1(**step1_kwargs)
            if ret != 0:
                raise RuntimeError("hybrid_quantization_step1 failed")
        finally:
            os.chdir(old_cwd)

        # Find generated cfg/model/data files. The toolkit uses a deterministic
        # prefix (often the model basename or "torchjitexport"); pick whichever
        # set of three exists.
        cfgs = sorted(Path(workdir).glob("*.quantization.cfg"))
        if not cfgs:
            raise RuntimeError(
                "step1 didn't write a .quantization.cfg — was the toolkit happy?"
            )
        prefix = cfgs[0].name[: -len(".quantization.cfg")]
        cfg_path = str(cfgs[0])
        model_path = str(Path(workdir) / f"{prefix}.model")
        data_path = str(Path(workdir) / f"{prefix}.data")
        for p in (model_path, data_path):
            if not os.path.exists(p):
                raise RuntimeError(f"step1 didn't produce expected file: {p}")
        print(f"[hybrid] step1 ok — patching {cfg_path}")

        # Trace heads through the ONNX graph (head = per-output chains minus
        # shared trunk) so we don't accidentally float16 the trunk layers.
        head_tensors, tensor_ops = _trace_head_tensors(
            onnx_path, info["output_names"], depth=trace_depth
        )
        print(f"[hybrid] traced {len(head_tensors)} head-only tensors "
              f"(trace-depth={trace_depth})")

        forced = _patch_hybrid_cfg(cfg_path, head_tensors, tensor_ops)
        print(f"[hybrid] forced {len(forced)} layers to float16")
        if verbose:
            for name in forced:
                print(f"           - {name}")

        if rknn.hybrid_quantization_step2(
            model_input=model_path,
            data_input=data_path,
            model_quantization_cfg=cfg_path,
        ) != 0:
            raise RuntimeError("hybrid_quantization_step2 failed")

        if rknn.export_rknn(rknn_path) != 0:
            raise RuntimeError("export_rknn failed")
        size = os.path.getsize(rknn_path)
        print(f"[hybrid] wrote {rknn_path} ({size/1024:.1f} KB)")
    finally:
        rknn.release()
        if keep_intermediates:
            print(f"[hybrid] kept intermediates in {workdir}")
        else:
            shutil.rmtree(workdir, ignore_errors=True)


# ────────────────────────────────────────────────────────────────────────
#  CLI
# ────────────────────────────────────────────────────────────────────────


def main():
    ap = argparse.ArgumentParser(
        description="Convert a KataGo / MiniGo ONNX model to RKNN",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("--onnx", required=True, help="Source .onnx model")
    ap.add_argument("--rknn", required=True, help="Output .rknn path")
    ap.add_argument("--mode", choices=["fp16", "hybrid", "int8"], default="fp16",
                    help="Quantisation mode (default fp16)")
    ap.add_argument("--target", default="rk3588", choices=SUPPORTED_TARGETS,
                    help="Rockchip SoC name (default rk3588)")
    ap.add_argument("--batch", type=int, default=1,
                    help="Batch size baked into the .rknn (default 1). "
                         "Use 4 for self-play workloads to amortise per-call overhead.")
    ap.add_argument("--dataset", default=None,
                    help="Calibration manifest (only for --mode hybrid / int8). "
                         "Generated by tools/rknn_calibration.py.")
    ap.add_argument("--proposal-size", type=int, default=16,
                    help="Calibration samples used by step1's auto-proposal (hybrid). "
                         "Toolkit default 16; raise if proposal looks under-confident.")
    ap.add_argument("--use-proposal", action="store_true",
                    help="Run rknn-toolkit2's own proposal pass during hybrid step1. "
                         "Off by default — we trace heads from the ONNX ourselves, "
                         "and proposal mode incompatibly errors with `does not support "
                         "expand batch` on graphs whose batch dim depends on a Reshape "
                         "(common in our exports).")
    ap.add_argument("--trace-depth", type=int, default=6,
                    help="ops to walk back from each ONNX output to flag as fp16 "
                         "in hybrid mode (default 6). Heads are typically 3-6 ops "
                         "deep past the trunk; setting too high pulls trunk layers "
                         "into the fp16 set and defeats the int8 trunk goal.")
    ap.add_argument("--optimization-level", type=int, default=3,
                    help="rknn-toolkit2 optimization_level (0-3, default 3)")
    ap.add_argument("--keep-intermediates", action="store_true",
                    help="Don't delete the hybrid step1 working directory")
    ap.add_argument("--verbose", action="store_true",
                    help="Verbose RKNN logging")
    args = ap.parse_args()

    if args.mode in ("hybrid", "int8") and not args.dataset:
        raise SystemExit(
            f"--mode {args.mode} requires --dataset (use tools/rknn_calibration.py to generate one)"
        )

    out_dir = os.path.dirname(args.rknn) or "."
    os.makedirs(out_dir, exist_ok=True)

    if args.mode == "fp16":
        convert_fp16(args.onnx, args.rknn, args.target, args.batch,
                     args.optimization_level, args.verbose)
    elif args.mode == "int8":
        convert_int8(args.onnx, args.rknn, args.target, args.batch,
                     args.dataset, args.optimization_level, args.verbose)
    elif args.mode == "hybrid":
        convert_hybrid(args.onnx, args.rknn, args.target, args.batch,
                       args.dataset, args.proposal_size, args.trace_depth,
                       args.optimization_level, args.keep_intermediates,
                       args.verbose, args.use_proposal)


if __name__ == "__main__":
    main()
