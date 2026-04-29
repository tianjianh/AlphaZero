# A733 Conversion — ONNX → Allwinner A733 NPU (Vivante VIP9000)

This doc covers converting the kata1 ONNX model to a deployable
**Network Binary Graph** (`.nb`) for the Allwinner A733 SoC (Cubie A7A
SBC), targeting **fp16** inference on the on-die VeriSilicon VIP9000
NanoDI+ NPU (≈3 TOPS).

It pairs with the existing on-host RKNN pipeline (see
[`RKNN_CONVERSION.md`](RKNN_CONVERSION.md)) — the source ONNX is the
same; only the offline conversion target differs.  The runtime side on
the board is **`awnn` / VIPLite**, not the C++ `ComputeHandle` used by
the existing repo backends — runtime integration is out of scope for
this document.

The converter lives at `tools/onnx_to_a733.py`; the host-side parity
sanity check is `tools/a733_verify.py`.

> **⚠️ Simulator ≠ hardware.**  All host-side parity numbers here are
> from the acuitylite simulator, which models VIP9000 fp16 dataflow but
> stores intermediates in fp32 internally and rounds at op boundaries.
> It does not see fp16 saturation events that the actual NPU MAC array
> would.  The same caveat documented in
> [`RKNN_CONVERSION.md` §16](RKNN_CONVERSION.md) applies; on-board
> verification with `awnn` is needed before deployment.

---

## 1. Quick start

The conversion runs on an **x86_64 Linux host** in the existing
`alphazero` conda env — no Docker, no `ACUITY_PATH` env var, no
separate VIV-SDK install.  `acuitylite` is pip-installable and ships
its own bundled SDK in the wheel.

```bash
# 1) one-time install (already done in this repo's alphazero env)
conda activate alphazero
pip install acuitylite

# 2) re-export KataGo .txt.gz to ONNX with the toolkit-friendly gpool topology
python tools/kata_export_for_rknn.py \
    --katago-bin kata1-b10c128-s1141046784-d204142634.txt.gz \
    --board 9 --batch 1 --opset 13 \
    --output models/kata1-b10c128.a733.bs1.onnx

# 3) simplify (fold constants, dedupe identity chains)
python -c "import onnx; from onnxsim import simplify; \
    m,_=simplify(onnx.load('models/kata1-b10c128.a733.bs1.onnx')); \
    onnx.save(m,'models/kata1-b10c128.a733.bs1.onnx')"

# 4) un-share initializers — required for kata1 (see §3.2)
python tools/onnx_rknn_mitigations.py \
    --input  models/kata1-b10c128.a733.bs1.onnx \
    --output models/kata1-b10c128.a733.bs1.unshared.onnx \
    --unshare-initializers

# 5) convert to NBG
python tools/onnx_to_a733.py \
    --onnx models/kata1-b10c128.a733.bs1.unshared.onnx \
    --output-dir models/kata1-b10c128.a733.bs1.fp16 \
    --mode fp16

# 6) sanity check (host simulator vs ONNX Runtime)
python tools/a733_verify.py \
    --onnx models/kata1-b10c128.a733.bs1.unshared.onnx
```

Output:

```
models/kata1-b10c128.a733.bs1.fp16/
  network_binary.nb     # 6.2 MB — the deployable artifact
  nbg_meta.json         # input/output names, shapes, dtypes (fp16)
```

---

## 2. Why this path

| Choice | Reason |
|---|---|
| **Vivante VIP9000 NanoDI+ target** | A733's on-die NPU.  Datasheet TOPS rating ≈3 INT8.  The fp pipeline is fp16-native — fp16 ops run at full rate, no slowdown vs int8 in the way RK3576/3588 has int16 at ⅓ rate. |
| **`acuitylite` (pip)** | VeriSilicon's official Acuity Toolkit, packaged as a Python wheel.  Avoids the Docker-image install path Radxa documents (`ubuntu-npu:v2.0.10.1`) — the wheel's bundled `vsi_sdk` includes the host-side OpenVX libs needed for codegen. |
| **`pack_nbg_only=True`** | Skip the ovxlib C-code scaffolding and the `--viv-sdk` requirement; emit just the `.nb` plus its metadata.  This matches what `awnn` loads on-device; it does not match Allwinner's `pegasus_export_ovx_nbg.sh` "float" branch (which targets a JIT-compiled application path). |
| **fp16, no quantize step** | VIP9000's fp pipeline is already fp16; passing `dtype='float'` with no calibration is the recommended fp16 path.  No `Quantization(...)` call is needed.  See §4 for the int8/quantized routes (not used here). |
| **Static batch (1 or 4)** | NPU codegen needs a fixed input shape — mirrors RKNN's constraint. |
| **`--unshare-initializers`** | kata1's BN exports a `weight==running_var` alias (both initializers identically all-ones; the ONNX exporter dedupes them).  acuitylite's matcher trips on this with `TypeError: 'NoneType' is not iterable` on the **first** BN whose weight collides with running_var.  Splitting the alias into per-consumer copies is math-equivalent (ORT-bit-identical) and bypasses the matcher bug. |

---

## 3. The converter — `tools/onnx_to_a733.py`

### 3.1 Pipeline (Python API)

```python
import os
os.environ["VSIMULATOR_CONFIG"] = "VIP9000NANODI_PLUS_PID0X1000003B"  # NPU v3
os.environ["VSIMULATOR_SHADER_CORE_COUNT"] = "1"

from acuitylib.interface.importer import OnnxLoader
from acuitylib.interface.exporter import OvxlibExporter

model = OnnxLoader(onnx_path).load(
    inputs=["state_spatial", "state_global"],
    input_size_list=[[1, 22, 9, 9], [1, 19]],
    outputs=["policy_logits", "value", "score_mean", "score_stdev", "ownership"],
)
OvxlibExporter(model).export(
    "<output-dir>/<prefix>",
    dtype="float",            # fp16 path; no Quantization step needed
    target_ide_project="linux64",
    pack_nbg_only=True,       # emit .nb directly, skip ovxlib C scaffolding
)
```

The two `VSIMULATOR_*` env vars come from Allwinner's
`pegasus_setup.sh v3` (the A733 SoC variant — A723 / A737 et al. would
take v1 / v2 settings).  They steer the simulator backend and
codegen-time tile sizing toward NanoDI+'s shader-core count.

### 3.2 Why un-share initializers (`onnx_rknn_mitigations.py`)

KataGo's BN layers are exported with `running_var=1` baked in (the
network was trained with no running-stat tracking; var defaults to 1).
The PyTorch ONNX exporter dedupes multiple identical-value initializers
into one — both `pre_bn.weight` (gamma=1) and `pre_bn.running_var`
(=1) become the same all-ones initializer used by 22+ BN nodes.

acuitylite's ruler-matcher walks BN inputs by slot expecting distinct
tensors per slot.  When weight ≡ running_var, it pushes the same
tensor twice and one of them ends up `None` mid-walk.  The traceback
is:

```
File "acuitylib/converter/onnx/convert_onnx.py", line 1529, in
    acuitylib.converter.onnx.convert_onnx.OnnxRulerMatcher._onnx_push_ready_tensor
TypeError: 'NoneType' object is not iterable
```

This is the **same** dedup pattern the RKNN side documented in
[`RKNN_CONVERSION.md` §16](RKNN_CONVERSION.md) (where it surfaced as
fp16 saturation rather than a matcher crash).
`onnx_rknn_mitigations.py --unshare-initializers` clones each
multi-consumer initializer so every (node, slot) gets its private copy.
Forward-output is bit-identical (the data didn't change, just the
indexing); the resulting ONNX is ~50 KB larger.  Both paths (RKNN bs=1
and A733) now feed off the same un-shared ONNX.

### 3.3 fp16 dataflow (no calibration)

VIP9000's MAC array runs fp16 natively.  The converter:

1. Loads weights into the IR at fp32 precision.
2. Compiles to NBG with no quantize pass.  The `nbg_meta.json` records
   `"dtype": "float16"` for every input and output — the runtime
   converts host-side fp32 buffers to fp16 on `awnn_set_input_buffer`,
   runs the graph in fp16, and converts back to fp32 on
   `awnn_get_output_buffer`.

No calibration data, no quantize-config dictionaries.  This is the
analog of `tools/onnx_to_rknn.py --mode fp16` for the RKNN target —
just simpler because acuitylite's hybrid / per-channel knobs aren't
relevant to fp16.

### 3.4 Bundled SDK

`pip install acuitylite` ships with a prebuilt OpenVX SDK at
`<site-packages>/acuitylib/vsi_sdk/prebuilt-sdk/x86_64_linux/{lib,include}`.
The `pack_nbg_only=True` path links a small driver (`gen_nbg`) against
those libs, runs the simulator on dummy zero input to generate the NBG
binary, and tears down.  No external Docker image, no separate
`--viv-sdk` flag, nothing to mount or path-set.

The bundled `vsi_sdk.tar.gz` (TIM-VX headers only) is **not** needed
for this path — it's there for a separate TIM-VX export route
(`TimVxExporter`, not used here).

---

## 4. Quantisation modes (full enumeration)

For the record — the toolkit also supports int8 / int16 / pcq via
`Quantization(model).quantize(quantizer=..., qtype=...)` before export.
We don't use any of them for kata1 because:

* **fp16 is essentially lossless** on this network (the parity numbers
  in §6 are 2 orders of magnitude inside fp16 noise floor) and
* the VIP9000 fp pipeline runs fp16 at full rate, so there's no
  throughput incentive to drop to int8 the way there is on RK3576/3588.

The available quantizers (`from acuitylib.quantization import QuantizerType`):

| Quantizer | Comment |
|---|---|
| `FLOAT16` | Pure fp16 — equivalent to skipping the quantize call entirely on this toolkit (recommended). |
| `BFLOAT16` | bfloat16 — VIP9000 supports it on newer revs; not validated here. |
| `ASYMMETRIC_AFFINE` | int8 with zero-point.  Standard "uint8" path; needs calibration. |
| `SYMMETRIC_AFFINE` | int8 with no zero-point.  Smaller, slightly less accurate; needs calibration. |
| `PERCHANNEL_SYMMETRIC_AFFINE` | per-channel int8.  Better accuracy on Conv-heavy models. |
| `DYNAMIC_FIXED_POINT` | int16 path. |
| `FLOAT8`, `MXFP8`, `PERCHANNEL_FLOAT8` | newer fp8 variants; not advertised on A733. |

If you ever do need int8, the calibration plumbing from
`tools/rknn_calibration.py` (Go-position self-play dump → `.npy` files)
is reusable — wrap the file list in an `input_generator_func` returning
`[spatial_arr, global_arr]` per call and pass it to `quantize()`.
Skipping for kata1.

---

## 5. Multi-batch builds

`.nb` files bake in a fixed batch — recompile per batch size.  bs=1 is
for live play (lowest single-move latency); bs=4 amortises per-call
host overhead for self-play workloads.

```bash
for bs in 1 4; do
    python tools/kata_export_for_rknn.py \
        --katago-bin kata1-b10c128-s1141046784-d204142634.txt.gz \
        --board 9 --batch $bs --opset 13 \
        --output models/kata1-b10c128.a733.bs${bs}.onnx
    python -c "import onnx; from onnxsim import simplify; \
        m,_=simplify(onnx.load('models/kata1-b10c128.a733.bs${bs}.onnx')); \
        onnx.save(m,'models/kata1-b10c128.a733.bs${bs}.onnx')"
    python tools/onnx_rknn_mitigations.py \
        --input  models/kata1-b10c128.a733.bs${bs}.onnx \
        --output models/kata1-b10c128.a733.bs${bs}.unshared.onnx \
        --unshare-initializers
    python tools/onnx_to_a733.py \
        --onnx models/kata1-b10c128.a733.bs${bs}.unshared.onnx \
        --output-dir models/kata1-b10c128.a733.bs${bs}.fp16 \
        --mode fp16
done
```

| variant | NBG size | source ONNX size | shape (in/out) |
|---|---|---|---|
| bs=1 | 6.2 MB | 12 MB | `state_spatial[1,22,9,9]` / `state_global[1,19]` → `[1,82] [1,1] [1,1] [1,1] [1,81]` |
| bs=4 | 6.6 MB | 12 MB | `state_spatial[4,22,9,9]` / `state_global[4,19]` → `[4,82] [4,1] [4,1] [4,1] [4,81]` |

The ~7% NBG-size delta between bs=1 and bs=4 is from the per-batch
shape constants the toolkit serialises — weights themselves are shared.

> **bs>1 caveat.**  The RKNN pipeline hit a per-batch-lane codegen bug
> on `bs=4` (slot 0 correct, slots 1-3 saturated to different values
> on byte-identical input — see
> [`RKNN_CONVERSION.md` §16.2](RKNN_CONVERSION.md)).  This is a
> Rockchip-specific bug; VeriSilicon's codegen is independent and not
> known to share it, but it has not yet been verified on real A733
> hardware.  Run `tools/a733_onboard_test.py`-equivalent (TBD; see §8)
> against `bs=4` before relying on it for self-play throughput.

---

## 6. Numerical fidelity (host simulator)

`tools/a733_verify.py` runs three canonical fixtures (all-zero, empty
board, mid-game) through both ONNX Runtime (CPU, fp32 reference) and
acuitylite's VIP9000 simulator path.  Per-output `max_abs_diff`:

| variant | policy_logits | value | score_mean | score_stdev | ownership |
|---|---|---|---|---|---|
| bs=1 | 1.4e-05 | 0 | 5.7e-06 | 5.7e-06 | 8.3e-07 |
| bs=4 | 1.4e-05 | 1.0e-06 | 2.3e-05 | 1.5e-05 | 1.0e-06 |

All results are **two orders of magnitude inside the fp16 noise floor**
(5e-3) — the conversion is numerically clean.

The host simulator stores intermediates in fp32 and rounds at op
boundaries, so it does **not** see hardware fp16 saturation events.
The same warning applies as on the RKNN side; the RKNN bs=1 fp16
saturation (which `--unshare-initializers` happened to fix) was
invisible in the simulator and only surfaced on real hardware.

---

## 7. Pre-built artifacts

The pre-built archives live in the repo root (`.7z`, ignored by git):

| Archive | Contents | Size |
|---|---|---|
| `kata1-b10c128.a733.bs1.fp16.7z` | `models/kata1-b10c128.a733.bs1.unshared.onnx` + `models/kata1-b10c128.a733.bs1.fp16/{network_binary.nb,nbg_meta.json}` | 16 MB |
| `kata1-b10c128.a733.bs4.fp16.7z` | `models/kata1-b10c128.a733.bs4.unshared.onnx` + `models/kata1-b10c128.a733.bs4.fp16/{network_binary.nb,nbg_meta.json}` | 16 MB |

Build them yourself with `7za a -t7z -mx=9 -m0=lzma2 -ms=on <name>.7z
models/<base>.unshared.onnx models/<base>.fp16/`.  The archives are
not checked into git — they're regenerable from the `.txt.gz` source
weights in seconds via the Quick Start in §1.

For board deployment:

```bash
# on host
scp kata1-b10c128.a733.bs1.fp16.7z cubie:~/
# on Cubie A7A
7za x kata1-b10c128.a733.bs1.fp16.7z   # extracts to models/...
# then point awnn at models/kata1-b10c128.a733.bs1.fp16/network_binary.nb
```

---

## 8. Future work

* **On-board parity test.**  RKNN has `tools/rknn_onboard_test.py` that
  feeds canonical fixtures through `rknnlite` on the aarch64 device and
  compares with ORT (CPU) on the same board.  An A733 equivalent
  (`tools/a733_onboard_test.py`) would feed through `awnn` /
  `viplite-tina`.  Same fixture set; same pass/fail criteria
  (`max_abs_diff < 5e-3`, value/score saturation report, per-slot
  consistency for bs > 1).  Not built yet — needs a Cubie A7A on the
  bench.
* **Runtime backend.**  The C++ `ComputeHandle` family in `src/`
  currently has CUDA / TensorRT / OpenCL / RKNN / Eigen / Metal
  backends.  An `awnn_compute.cpp` backend that wraps `awnn_create`,
  `awnn_set_input_buffer`, `awnn_run`, `awnn_get_output_buffer` would
  let the rest of the project reach the A733 NPU through the same
  `nn_evaluator.cpp` dispatch the RKNN backend already uses.  Out of
  scope for the conversion side; tracked separately.
* **bs=4 on-board verification.**  The per-batch-lane bug class
  documented for RKNN (§16.2) needs to be checked-not-assumed on
  VeriSilicon's codegen.  Run a tiled-fixture test on real hardware
  before shipping bs=4 to self-play.
* **Hybrid quant / int8 path.**  Skipped here because fp16 is
  lossless and equally fast on this NPU — but the calibration plumbing
  from `rknn_calibration.py` is portable if a future smaller network
  (or a larger one whose weights don't fit on-die) needs the
  ~50% memory reduction.

---

## 9. File reference

| File | Role |
|---|---|
| `tools/onnx_to_a733.py` | ONNX → NBG converter (fp16-only at the moment). |
| `tools/a733_verify.py` | Host-side ORT vs acuitylite simulator parity check. |
| `tools/kata_export_for_rknn.py` | Re-exports KataGo `.txt.gz` to ONNX with toolkit-friendly 4D gpool topology — **shared** with the RKNN path; the same source ONNX feeds both targets. |
| `tools/onnx_rknn_mitigations.py` | `--unshare-initializers` mitigation; same script as the RKNN bs=1 fix (the kata1 BN weight≡running_var alias affects both targets). |
| `models/<base>.a733.bs<N>.onnx` | Re-exported source ONNX (intermediate; deletable after step 4). |
| `models/<base>.a733.bs<N>.unshared.onnx` | Un-shared source ONNX — the canonical reference for downstream parity checks. |
| `models/<base>.a733.bs<N>.fp16/network_binary.nb` | Deployable NBG. |
| `models/<base>.a733.bs<N>.fp16/nbg_meta.json` | Per-NBG metadata: input/output names, shapes, dtypes (fp16 throughout). |
