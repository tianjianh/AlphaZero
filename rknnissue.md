# RKNN On-Device Output Saturation — kata1-b10c128 fp16

## TL;DR

The `kata1-b10c128` fp16 `.rknn` files (both `bs=1` and `bs=4`) produce wildly
out-of-range outputs when run on the **actual RK3576 NPU via librknnrt**:
policy logits ~3 000× ONNX Runtime's range, value head saturated at +1.0,
score head at the fp16 ceiling (or `+inf`). MCTS faithfully amplifies the
one-hot priors into a single-line forced descent.

The runtime backend has been ruled out (output is bit-identical to Rockchip's
official `rknnlite` Python wrapper). The fault is at the **conversion** layer,
on the actual hardware. The "fp16 100% top-1 vs ORT" fidelity number in
`RKNN_CONVERSION.md §11.1` was measured against the **rknn-toolkit2
simulator on x86**, which does not predict on-device behaviour for this
network.

---

## 1. Environment

| Item | Value |
|------|-------|
| Board | ArmSoM Sige5 |
| SoC | Rockchip RK3576 (2-core NPU) |
| Kernel | `6.1.115-vendor-rk35xx` |
| `librknnrt` | `2.3.2 (429f97ae6b @ 2025-04-09T09:09:27)` |
| NPU driver | `0.9.8` |
| Toolkit (per `.rknn` metadata) | `2.3.0 (c949ad889d @ 2024-11-07T11:39:30)` |
| Conversion target | `rk3576`, `--mode fp16` |

Runtime backend: `src/rknn_compute.cpp` (commit `8c28211`, dual-input
KataGo support).

## 2. Files tested

| File | Size | Role |
|------|-----:|------|
| `kata1-b10c128.rknn.bs1.onnx` | 12 007 844 B | RKNN-friendly ONNX, batch=1 |
| `kata1-b10c128.rknn.bs4.onnx` | 12 007 844 B | RKNN-friendly ONNX, batch=4 |
| `kata1-b10c128.rk3576.bs1.rknn` | 6 746 417 B | Compiled RKNN, batch=1 |
| `kata1-b10c128.rk3576.bs4.rknn` | 7 319 729 B | Compiled RKNN, batch=4 |

Both `.rknn` files were exported from the matching `.onnx` with
`--mode fp16 --target rk3576` per the documented pipeline. The two `.onnx`
files are byte-identical except for the baked batch dim; the `.rknn`
artifacts diverge in the predictable batched-buffer ways but otherwise
share the same trunk/head topology.

## 3. Test input

A realistic empty-board kata1 input matching `encode_for_katago`:

```python
H, W, C, G = 9, 9, 22, 19
sp = np.zeros((1, C, H, W), dtype=np.float32)
gl = np.zeros((1, G),       dtype=np.float32)
sp[:, 0, :, :] = 1.0          # plane 0 = on-board
gl[:, 5]  = -6.5 / 20.0       # self-komi / 20 (BLACK to play, komi=6.5)
gl[:, 15] = -0.5              # komi parity wave for komi=6.5
```

ORT through `kata1-b10c128.rknn.bs{1,4}.onnx` is the ground truth.

## 4. Symptom — saturated on-device output

| Metric | ORT (truth) | RKNN `bs=4` on-device | RKNN `bs=1` on-device |
|--------|------------:|---------------------:|---------------------:|
| `policy_logits` max | **3.20** | **10 940** | **8 888** |
| `policy_logits` min | **−7.26** | **−18 300** | **−33 500** |
| `value` | **0.061** | **1.000** (saturated) | **1.000** (saturated) |
| `score_mean` | **0.37** | **+inf** | **51 550** (≈ fp16 max 65 504) |
| `score_stdev` | **10.09** | **6.1 × 10⁻⁴** | **6.1 × 10⁻⁴** |
| `ownership` | **≈ 0.5** | **0.998** | **0.998** |
| top-1 vs ORT | — | **no** | **no** |
| top-5 overlap vs ORT | — | **0 / 5** | **0 / 5** |
| `softmax(policy)` effective moves | ~82 | **1.00** | **1.00** |

Policy logits are ~3 000× ORT's magnitude — not a clean scale factor, and
the *minimum* logit's magnitude grows from `bs=4` (−18 k) to `bs=1` (−33 k),
so the divergence isn't even monotone in the batch dimension.

After `softmax(logits − max)` the prior collapses to **one-hot on a
single move**. MCTS then picks that move every descent, expands the line to
terminal in ~150 NN calls (≈1 second wall-time on the NPU), and from then
on every "playout" walks the cached forced line to the terminal node and
backprops without an NN call. We observed `N = 2 502 237` visits accumulate
in well under a minute on a single position — entirely from terminal-walk
spinning, since real NN-bound throughput is ~673 states/s on this `.rknn`
(`benchmark` test 3, batch=4).

## 5. Even all-zero input saturates

```cpp
std::vector<float> all_zero(C*H*W + G, 0.0f);   // every spatial + global feature = 0
auto r = handle->predict_batch({ all_zero });
// policy[0..3] = -6 580, -28 200, -24 900, -25 600
// value       = 1.0
// score_mean  = +inf
// ownership   = 0.998
```

A net fed only zeros should output the bias-only forward pass — a single
deterministic small vector. The fact that all-zero already saturates rules
out any input-side issue (encoder, layout transpose, scale) and points at
the trunk's intermediate activations themselves overflowing fp16.

## 6. What's been ruled out

### 6.1 Not the runtime

Output of the C++ runtime (`src/rknn_compute.cpp`) is **bit-for-bit
identical** to Rockchip's official `rknnlite` Python wrapper on the same
inputs, both for `bs=1` and `bs=4`. Test:

```python
from rknnlite.api import RKNNLite
r = RKNNLite()
r.load_rknn('kata1-b10c128.rk3576.bs1.rknn')
r.init_runtime()
out = r.inference(inputs=[sp, gl], data_format=['nchw', 'nchw'])
# policy[0, :5] = [-33504, -6304, -7680, -13320, -14648]   ← matches C++
```

Same logits to fp16 precision in every slot. So the bug is **not** in the
NCHW→NHWC transpose, the dual-input demux, or the `rknn_inputs_set` /
`rknn_run` / `rknn_outputs_get` sequence.

### 6.2 Not batching

`bs=1` and `bs=4` exhibit the **same** qualitative pathology
(value=1, ownership=0.998, score saturating, one-hot policy). Only the
exact magnitude of policy logits differs. Reducing the batch did not
help — the cause is per-position, not per-batch.

### 6.3 Not the encoder

Three different input fixtures all saturate identically at the head:

1. Real `encode_for_katago(empty 9×9, komi=6.5)`
2. Synthetic minimal input (plane 0 = on-board, `gl[5] = komi/20`)
3. All-zero input

ORT on the same encoded input returns sensible per-test numbers
(value=0.06, score=0.37 for input 1; value=1.0, score=30.94 for input 3 — etc).
So the encoder produces what the `.onnx` expects.

### 6.4 Not a wrong-target `.rknn`

The `.rknn` metadata reports `target platform: rk3576`; our SoC is RK3576;
`rknn_init` succeeds; the kernel reports the rknn0/rknn1 cores are alive.
A SoC mismatch would error out at init, not produce silent garbage.

### 6.5 Not "every kata-class .rknn" — single-input MiniGo `.rknn` works fine

A separate `v0000.rknn` (single-input MiniGo, 17-channel ResNet, 10×128)
on the same board, through the same `RKNNComputeHandle::predict_batch`,
produces sensible outputs:

```
policy max = 0.118, min = -0.123     (matches ORT's similar small-logit range)
value      = 0.0066                  (close to 0, near-even position)
softmax top-5 priors: 0.014 0.013 0.013 0.013 0.013     (≈ uniform over 82)
```

So the runtime path is exercised correctly. The kata1 fp16 conversion
specifically produces broken on-device outputs.

## 7. Probable cause — fp16 trunk overflow on hardware

fp16 max is 65 504. kata1's trunk (10 blocks × 128 channels with global-pool
concat widening intermediate widths to 256+) likely has a Conv-BN-ReLU
intermediate whose magnitude exceeds 65 k on real positions, propagates as
`inf` through downstream `Softmax / Tanh / Sub / Mul` in the heads, and
appears at the output as the saturated values we see.

Indirect evidence (all consistent with the overflow story):

* `score_mean = 51 550` in the `bs=1` output is one `Mul` scaling factor
  away from the fp16 ceiling.
* `score_stdev = 6.1e-4 ≈ 1 / 2^11` is fp16's smallest normal positive
  number — the `Softplus / Mul` chain has under-flowed at the other extreme.
* `ownership = 0.998` is precisely what `tanh(large_positive)` saturates to.
* `value = 1.0` from `softmax(WLD_logits)[W] − softmax(WLD_logits)[L]`
  saturates at +1 whenever one of the three WLD logits dominates by ≳20.
* The `bs=1` `policy_min = −33 500` and `bs=4` `policy_min = −18 300` are
  both well past where fp32 → fp16 saturation flushes to `−inf`; a single
  upstream Conv-BN producing one extreme channel could explain both
  numbers (different batch padding may zero different post-overflow neighbours).

The toolkit's host-x86 **simulator** presumably stores fp16 values in fp32
internally and only rounds at op boundaries — the network never experiences
a true fp16 saturation event in simulation. On the **actual NPU** fp16 MAC
array, intermediate accumulators *are* fp16, and once any single op
overflows, downstream ops feed `inf` forward unchecked. This would explain
the simulator-vs-hardware split that the §11.1 numbers don't reflect.

## 8. Suggested experiments for the converter side

These all happen on the converter (x86 host) side; the runtime needs no
changes.

1. **Re-measure the §11.1 fidelity matrix on real hardware**, not the
   simulator. Drive `rknnlite` on a target SoC (RK3576 or RK3588), feed it
   the same calibration positions you currently feed the simulator, and
   record `top-K / KL / max_abs_diff` against ORT. We expect fp16 row to
   move from "100% top-1" to "≪ 100%" for kata1.

2. **Try `--mode hybrid --preset kata1`** before fp16. RKNN_CONVERSION.md
   §11.4 reports 87% top-1 for this preset on the simulator. Even at 87%
   on hardware, MCTS will recover real moves. The int8 trunk explicitly
   bounds intermediate dynamic range, which is exactly the property we need
   if fp16 overflow is the cause.

3. **Try `--mode int8` (full quant)** as a comparison. If int8 *also*
   saturates, the bug is graph-level (a fold or rewriter); if int8 is
   clean, that confirms fp16 dynamic range as the root cause and locks in
   hybrid as the production path.

4. **Diagnostic: insert intermediate output operators in the ONNX export**
   and inspect them on-device. Tap points:

    * after the input conv (stem output)
    * mid-trunk: block 5 output
    * end-trunk: block 9 output
    * before each head split

   Run `rknnlite` against the empty-board input. The first tap whose
   magnitude exceeds ~30 k is where the fp16 overflow originates — likely
   one specific BN scale or pre-activation that is anomalously large for
   this checkpoint.

5. **Mitigation A — add input-side scaling.** If trunk overflow can be
   pulled back below 65 k by shrinking the input by 2×:

   ```python
   rknn.config(target_platform='rk3576',
               mean_values=[[0]*22],
               std_values=[[2]*22])     # input/2 — math-equivalent if the
                                        # first conv weights are doubled
   ```

   This should be lossless when paired with a doubling of the first conv's
   weights (or just absorbed into the first BN scale).

6. **Mitigation B — pin specific Convs to fp16 / int8 by name.** If the
   tap-point diagnostic in (4) finds *one* offending block, use
   `--keep-fp16-pattern '<regex>'` to flag it. The converter already has
   the per-output graph-trace + cfg-patcher infrastructure (§7).

7. **Toolkit version sanity.** The `.rknn` says built with toolkit 2.3.0;
   §10.1 already pins 2.3.0 for an unrelated `fold_constant` bug in 2.3.2.
   Worth confirming the host build environment really used 2.3.0
   (`pip show rknn-toolkit2`) and not a transitively-upgraded 2.3.2 — and
   maybe trying 2.2.0 as a control.

8. **NC1HWC2 channel padding.** The verbose dump shows the spatial input
   landing as `OrigShape=(4,22,9,9)` `NativeShape=(4,12,9,9,8)` — i.e. the
   toolkit packed 22 channels into `C1=12 × C2=8 = 96` channel slots.
   That's a 4× over-allocation. Worth verifying whether the padded slots
   are zero-filled or whether the toolkit relies on subsequent ops masking
   them — a stale value in slot 23..95 fed into the first conv would be
   a candidate for the saturating bias we observe.

## 9. Reproduction

On the RK3576 board, with this repo's `build-rknn/`:

```bash
# Make the runtime extension-swap happy: <base>.rknn.bs1.onnx → <base>.rknn.bs1.rknn
ln -sf kata1-b10c128.rk3576.bs1.rknn  kata1-b10c128.rknn.bs1.rknn

# Symptom: value=1.0, score=+inf, N sky-rockets
./build-rknn/play --model kata1-b10c128.rknn.bs1.onnx --max-batch 1
```

Headless parity check (Python on the same board, in the `alphazero` conda env
with `pip install rknn-toolkit-lite2`):

```python
import numpy as np, onnxruntime as ort
from rknnlite.api import RKNNLite

H,W,C,G = 9,9,22,19
sp = np.zeros((1,C,H,W), np.float32); sp[:,0,:,:] = 1.0
gl = np.zeros((1,G),     np.float32); gl[:,5] = -6.5/20.0; gl[:,15] = -0.5

ort_out = ort.InferenceSession(
    'kata1-b10c128.rknn.bs1.onnx',
    providers=['CPUExecutionProvider']
).run(None, {'state_spatial': sp, 'state_global': gl})
print(f"ORT  value={ort_out[1][0,0]:.4g}  score={ort_out[2][0,0]:.4g}")

r = RKNNLite()
r.load_rknn('kata1-b10c128.rk3576.bs1.rknn')
r.init_runtime()
rk_out = r.inference(inputs=[sp, gl], data_format=['nchw','nchw'])
print(f"RKNN value={rk_out[1][0,0]:.4g}  score={rk_out[2][0,0]:.4g}")
```

Expected (and what we see):

```
ORT  value=0.06082  score=0.3706
RKNN value=1        score=5.155e+04   ← bs=1
                          inf         ← bs=4
```

Anything within fp16 noise (≈ 1e-3 on value, < 2 points on score) would
indicate a fixed conversion. The 16× value error and 140 000× score error
make play unusable.

## 10. What this implies for the runtime

Nothing changes in `src/rknn_compute.cpp`. The dual-input KataGo support
(commit `8c28211`) is correct — confirmed by:

* bit-for-bit match with the official `rknnlite` Python wrapper on
  identical inputs;
* a different `.rknn` (single-input MiniGo `v0000.rknn`) producing
  sensible outputs through the same code path;
* per-batch-slot output diversity confirming `state_global` is correctly
  routed to its own input bucket.

As soon as the converter produces a `.rknn` that does not saturate on
real hardware, the runtime will pick it up unchanged.
