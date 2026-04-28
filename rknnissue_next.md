# RKNN Saturation — Follow-Up After Mitigation Round 1

## TL;DR

Round 1 mitigations from the converter side
(`--input-scale 2` and `--unshare-initializers`) **partially fix** the
on-device fp16 saturation reported in `rknnissue.md`:

| variant | bs=1 | bs=4 |
|---|---|---|
| baseline `fp16` | FAIL | FAIL |
| `--input-scale 2` (`scaled2`) | FAIL | FAIL |
| **`--unshare-initializers` (`unshared`)** | **✅ PASS** (max-abs-diff ≈ 1e-4 = fp16 noise) | partial — slot 0 PASS, slots 1–3 still saturate |
| `--input-scale 2 --unshare-initializers` | ✅ PASS (identical to `unshared`) | partial — same slot-0-only behaviour |

We can ship **bs=1 unshared** today. **bs=4 has a second, independent
bug** at the per-batch-lane level that `unshare-initializers` doesn't
touch.

While re-reading the v2.3.2 SDK guide (`02_Rockchip_RKNPU_User_Guide_RKNN_SDK_V2.3.2_EN.pdf`,
pp. 125–135) we also found that **Rockchip ships an official one-line
fix for this exact class of overflow bug** — `rknn.build(...,
auto_hybrid=True)` — which we hadn't tried yet. That should be
attempted before any more graph rewrites.

---

## 1. What we tested

On RK3576 board (ArmSoM Sige5, librknnrt 2.3.2), against the four
mitigation `.rknn` variants you produced:

* `kata1-b10c128.rk3576.bs{1,4}.rknn` (baseline, no rewrite)
* `kata1-b10c128.rk3576.bs{1,4}.scaled2.rknn` (`--input-scale 2`)
* `kata1-b10c128.rk3576.bs{1,4}.unshared.rknn` (`--unshare-initializers`)
* `kata1-b10c128.rk3576.bs{1,4}.scaled2.unshared.rknn` (both)

Driver: `tools/rknn_onboard_test.py`, with three small fixes (see §6).
Reference: ORT against the matching source `.onnx`, on the same input
fixtures (empty 9×9 / komi=6.5 / BLACK and all-zero).

## 2. Per-output detail — bs=1 `unshared` passes cleanly

Empty 9×9 / komi=6.5 / BLACK to play, ORT vs RKNN:

| output | ORT | RKNN bs=1 unshared | max-abs-diff |
|---|---|---|---|
| `policy_logits` | range [−7.26, 3.20] | range [−7.25, 3.20] | **1.7 × 10⁻²** |
| `value` | 0.0608 | 0.0615 | **7.1 × 10⁻⁴** |
| `score_mean` | 0.371 | 0.371 | **4.8 × 10⁻⁴** |
| `score_stdev` | 10.09 | 10.09 | **3.2 × 10⁻⁴** |
| `ownership` | range [0.49, 0.605] | range [0.49, 0.605] | **4.6 × 10⁻⁴** |

Diffs are at the fp16 noise floor. `softmax(policy)` produces a
near-uniform distribution over legal moves (≈82 effective moves) — exactly
what MCTS needs.

End-to-end through the production `NNEvaluator` queue + dual-input C++
runtime + bs=1 unshared `.rknn`: same numbers.

## 3. The new bug — slot-0-only correctness on bs=4

bs=4 `unshared` and `scaled2.unshared` "pass" the tensor-range saturation
check, but only because **slot 0** is correct. Per-slot inspection on
**four identical inputs** (the same empty board fed to all 4 slots):

| slot | value (ORT=0.0608) | score (ORT=0.371) | policy max (ORT=3.20) |
|------|---:|---:|---:|
| 0 | **0.0615** ✓ | **0.371** ✓ | **3.20** ✓ |
| 1 | 0.996 | 5.25 | −0.43 |
| 2 | 1.000 | 7.05 | 2.21 |
| 3 | 0.999 | 6.45 | −0.14 |

Slots 1–3 are saturated, **with each slot saturating to a different
value** even though the inputs were byte-identical. This is a
deterministic NN — the batch-level disagreement on identical inputs
means there's per-batch-lane state in the rknn-toolkit2 fp16 codegen path
that only lane 0 initializes correctly.

`scaled2` alone (no unshare) gives all 4 slots saturated:

```
slot 0: value=1, score=51100, policy_max=5892
slot 1: value=1, score= 9744, policy_max=−158
slot 2: value=1, score=12580, policy_max=6076
slot 3: value=1, score=12300, policy_max=4976
```

So we have **two layered bugs on bs=4**:

1. **Initializer-aliasing fp16 overflow** — fixed by `--unshare-initializers`
   for slot 0.
2. **Per-batch-lane state divergence** — slots 1..K-1 still saturate even
   after (1) is fixed.

## 4. Other relevant data

* **Config that's verified clean is `bs=1 unshared`.** That's a viable
  production deploy *now*: rename
  `kata1-b10c128.rk3576.bs1.unshared.rknn → kata1-b10c128.rknn.bs1.rknn`
  (matching the existing `<base>.rknn.bs1.onnx`) and the runtime's
  extension-swap will pick it up.

* **`scaled2` alone is a no-op for our problem.** `scaled2` and baseline
  saturate identically. `scaled2.unshared` is bit-identical to
  `unshared` (the scaling vanishes). So future iterations can drop
  `--input-scale` from the matrix unless there's another reason for it.

* **MiniGo `v0000.rknn` (single-input, 17 ch) on the same board, through
  the same runtime, gives sane outputs** — the bug is kata1-trunk
  specific, not "all rknn .rknn files on this SoC."

## 5. The SDK guide says we missed an official fix

In re-reading
`02_Rockchip_RKNPU_User_Guide_RKNN_SDK_V2.3.2_EN.pdf` we found
two relevant sections, both of which describe our exact symptom and
prescribe a different procedure than what we did:

### §6.3.3 Automatic Hybrid Quantization (p. 125) — the documented one-liner

> "To simplify the usage of hybrid quantization and **solve model
> overflow issues**, RKNN-Toolkit2 provides an automatic hybrid
> quantization feature. … For **non-quantized models**, it will check
> each layer of the model for **fp16 overflow and convert the
> overflowing layers to int16 computation**. At this time, you need to
> provide a dataset to calculate the numerical range of each layer."

```python
ret = rknn.build(do_quantization=False, dataset='./dataset.txt',
                 auto_hybrid=True)
```

**This is a documented fix for "non-quantized fp16 model overflows on
hardware."** It's the same shape as our bug. The previous fp16 builds
went through `do_quantization=False, auto_hybrid=False`; switching
`auto_hybrid=True` and pointing it at the existing kata1 calibration set
(`tools/rknn_calibration.py` → `calib/kata1/dataset.txt`) should
auto-detect every overflowing layer and bump it to int16.

This may be enough on its own — and crucially, it operates per-layer with
real per-tensor numerical ranges from calibration data, which is more
principled than our `--input-scale` (uniform 0.5× shrink, no
information about which layer needs it) or `--unshare-initializers`
(graph rewrite that happens to dodge the toolkit codegen bug).

### §7.1.1 (p. 132–133) — they tell you to expect fp16 overflow

> "When converted from FP32 to FP16, the intermediate tensors of the
> model may encounter overflow issues. … if tensors in the model during
> inference have values **exceeding the FP16 expression range (−65504 ~
> 65504), the tensor will overflow, resulting in abnormal model
> inference results.**"
> "For overflow issues, users can use the `rknn.accuracy_analysis(...,
> target=None)` interface for simulator FP16 accuracy analysis. **If the
> entire column or single column of `simulator_error` in the analysis
> results has an abnormal value (If words such as 'inf' appear), an FP16
> overflow may occur.**"
> "In this case, users can try **modifying the model structure to
> ensure that all tensors in the model do not overflow in FP16 (such as
> adding some BN layers, etc.).**"

This is the diagnostic we should have run first — it would have told us
which layer in kata1's trunk overflows.

### §7.2 (p. 134) — Rockchip admits sim/hardware can diverge

> "When the accuracy of the simulator is normal, abnormal inference
> results may still occur when deploying the board-side C API. There are
> generally three reasons … **The first is caused by a bug in the
> runtime of the board.**"
> "(because the **simulator does not strictly simulate NPU hardware, the
> results may not be completely consistent with the simulator**)."

And the prescribed escalation path:

> "If the inference results in the above steps are significantly
> different from the simulator inference results, it can be initially
> determined that there is a bug in the runtime of the board. **In this
> case, the analysis results and reproduced models can be fed back to
> the Rockchip NPU team for repair.**"

This validates our `rknnissue.md` approach (forensic report with
reproduction). For the bs=4 per-slot bug specifically, the SDK guide's
own §7.2 procedure is the right next step.

## 6. Recommended next iteration

In rough priority order:

### (a) **Try `auto_hybrid=True` first**, with the existing calibration set

```python
# in tools/onnx_to_rknn.py, the fp16 path
rknn.build(
    do_quantization=False,
    dataset='calib/kata1/dataset.txt',   # already produced by tools/rknn_calibration.py
    auto_hybrid=True,
    # Optional knobs (defaults shown in the doc):
    #   auto_hybrid_cos_thresh=0.98     # int8→fp16 promotion threshold (not used in non-quant path)
    #   auto_hybrid_euc_thresh=None     # euclidean threshold (None = disabled)
)
```

Build for both bs=1 and bs=4. If the per-slot bug on bs=4 was caused by
the same fp16 overflow (just expressed differently across MAC lanes),
this may fix bs=4 too — the int16 promotion on the offending layers
removes the lane-dependent saturation event.

### (b) **Run accuracy_analysis to confirm the overflow location**

```python
rknn.accuracy_analysis(
    inputs=['calib/kata1/state_spatial_0000.npy',
            'calib/kata1/state_global_0000.npy'],
    target=None)            # simulator
```

Look at `simulator_error`. Any layer with `inf` in the error column is
where fp16 overflows. (Per the SDK doc, having `inf` in *any*
`simulator_error` column is the textbook signature of our problem.)

For the runtime divergence:

```python
rknn.accuracy_analysis(
    inputs=[...],
    target='rk3576')        # board-connected via USB
```

The `runtime_error` column compared against `simulator_error` will name
the layer where on-device numerics diverge from the simulator. That's
the artefact §7.2 of the SDK guide tells us to attach to a Rockchip bug
report.

### (c) **If `auto_hybrid` doesn't fix bs=4, file with Rockchip**

The bs=4 slot-0-only behaviour is novel (we couldn't find it in the
existing `airockchip/rknn-toolkit2` issue tracker — we did find
[#274](https://github.com/airockchip/rknn-toolkit2/issues/274),
[#425](https://github.com/airockchip/rknn-toolkit2/issues/425),
[#463](https://github.com/airockchip/rknn-toolkit2/issues/463),
[#504](https://github.com/airockchip/rknn-toolkit2/issues/504),
[#220](https://github.com/airockchip/rknn-toolkit2/issues/220),
[#460](https://github.com/airockchip/rknn-toolkit2/issues/460), and
[#444](https://github.com/airockchip/rknn-toolkit2/issues/444) all
describing related sim-vs-hardware fp16 saturation symptoms, none with
the per-batch-slot signature). Worth a fresh issue with our reproduction.

## 7. Three small fixes to `tools/rknn_onboard_test.py` (already applied locally)

In running the drill we hit three bugs in the test script that I patched
locally:

1. **`data_format=None` for 2-D inputs** — `rknnlite` rejects `None`
   for the non-spatial input. Pass `"nchw"` for every input; the runtime
   takes 2-D buffers as-is regardless of the format string.

2. **Inputs always batch=1, but bs=4 ONNX expects batch=4** — the
   script crashes on bs=4 ORT with `Got: 1 Expected: 4`. Patched to tile
   the test input along the batch axis to match the ONNX baked batch.

3. **No per-slot consistency check** — the original script's range-based
   sanity ("value ∈ [−1, 1]?") returns `OK` on bs=4 unshared because
   slot 0's `value=0.0615` falls in range, masking the fact that slots
   1–3 are at 1.0. Patched to add a per-slot consistency check: when
   batch > 1 and inputs are tiled (i.e. all slots fed identical inputs),
   max slot-vs-slot diff must be < 0.1, otherwise FAIL.

The script diff is +31/−6 lines on top of `tools/rknn_onboard_test.py`.
With those three fixes, the script correctly:

* PASSES bs=1 unshared / scaled2.unshared
* FAILS bs=1 baseline / scaled2 (saturated)
* FAILS bs=4 baseline / scaled2 / unshared / scaled2.unshared
  (saturated, or per-slot inconsistent on the unshared variants)

This makes it a real CI gate for "is this `.rknn` actually usable on the
board?"

## 8. What's blocking what

* **bs=1 production**: nothing — `kata1-b10c128.rk3576.bs1.unshared.rknn`
  is clean, just needs the extension-swap symlink (or a rename).
* **bs=4 production**: blocked on the per-slot bug. Try `auto_hybrid=True`
  first; if that doesn't fix it, file a Rockchip bug with the
  `accuracy_analysis(target='rk3576')` artefact.
* **Runtime side**: nothing — the dual-input C++ backend (commit
  `8c28211`) already takes whichever `.rknn` is sane; it produces
  bit-identical output to `rknnlite` on every variant we've tried.

## 9. Reproduction (copy-paste)

```bash
# Confirm baseline saturates (sanity)
python tools/rknn_onboard_test.py \
    --onnx models/kata1-b10c128.rknn.bs1.onnx \
    --rknn models/kata1-b10c128.rk3576.bs1.rknn

# Confirm bs=1 unshared is clean
python tools/rknn_onboard_test.py \
    --onnx models/kata1-b10c128.rknn.bs1.onnx \
    --rknn models/kata1-b10c128.rk3576.bs1.unshared.rknn

# Confirm bs=4 unshared has the per-slot bug
python tools/rknn_onboard_test.py \
    --onnx models/kata1-b10c128.rknn.bs4.onnx \
    --rknn models/kata1-b10c128.rk3576.bs4.unshared.rknn

# Once you have an auto_hybrid build, drop it next to the .onnx and:
python tools/rknn_onboard_test.py \
    --onnx models/kata1-b10c128.rknn.bs4.onnx \
    --rknn models/kata1-b10c128.rk3576.bs4.auto_hybrid.rknn
```

For the slot-by-slot view that tensor-range checks miss:

```python
import numpy as np, onnxruntime as ort
from rknnlite.api import RKNNLite

H,W,C,G = 9,9,22,19
sp = np.zeros((4,C,H,W), np.float32); sp[:,0,:,:] = 1.0
gl = np.zeros((4,G), np.float32); gl[:,5] = -6.5/20.0; gl[:,15] = -0.5

r = RKNNLite()
r.load_rknn('models/kata1-b10c128.rk3576.bs4.unshared.rknn')
r.init_runtime()
out = r.inference(inputs=[sp, gl], data_format=['nchw', 'nchw'])
for s in range(4):
    print(f"slot {s}: value={out[1][s,0]:.4g}  "
          f"score={out[2][s,0]:.4g}  policy_max={out[0][s].max():.4g}")
```

Expected on the buggy bs=4 unshared: slot 0 sane, slots 1–3 saturated to
different values.
