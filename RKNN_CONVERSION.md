# RKNN Conversion — ONNX → Rockchip NPU

This doc covers converting MiniGo and KataGo ONNX models to Rockchip's
`.rknn` format for inference on RK3562 / RK3566 / RK3568 / RK3576 / RK3588
NPUs.  It pairs with the existing C++ `RKNNComputeHandle` (described in
[`README.md`](README.md#rknn-npu-backend-rockchip-aarch64-linux)); this
document focuses **purely on the offline conversion pipeline** that
produces the `.rknn` file.

The conversion tools live in `tools/`:

| Tool                         | Purpose |
|------------------------------|---------|
| `tools/onnx_to_rknn.py`      | ONNX → RKNN converter (fp16 / int8 / hybrid). |
| `tools/rknn_calibration.py`  | Generate calibration `.npy` files for int8/hybrid via self-play. |

Neither tool modifies the inference code paths — they sit alongside the
existing `tools/katago_to_onnx.py` / `scripts/export_onnx.py` flow and
emit a `.rknn` that drops in next to the source `.onnx` (the C++ runtime
resolves the `.rknn` path by extension swap on `LoadedModel::model_path`).

---

## 1. Quick start

The conversion runs on an **x86_64 Ubuntu host** — `rknn-toolkit2` is x86
only.  `librknnrt`-based inference happens on the aarch64 board.

```bash
# 1) Set up an rknn-conversion conda env (~150 MB)
conda create -n rknn python=3.10 -y
conda activate rknn
pip install 'rknn-toolkit2==2.3.0' 'setuptools<81' 'onnx<1.18'

# 2) Convert ONNX → fp16 .rknn (no calibration needed)
python tools/onnx_to_rknn.py \
    --onnx models/best.onnx \
    --rknn models/best.rknn \
    --mode fp16 --target rk3588

# 3) Hybrid int8 (trunk int8, sensitive heads fp16) — needs calibration
python tools/rknn_calibration.py \
    --onnx models/best.onnx \
    --output calib/best \
    --num-positions 200
python tools/onnx_to_rknn.py \
    --onnx models/best.onnx \
    --rknn models/best.rknn \
    --mode hybrid --target rk3588 \
    --dataset calib/best/dataset.txt

# 4) ship to board
scp models/best.rknn armsom:~/minigo/models/
```

---

## 2. Quantisation modes

Three modes are exposed via `--mode`:

| Mode      | Weights / activations | Calibration | Speedup vs fp16 | Quality cost |
|-----------|----------------------|-------------|-----------------|--------------|
| `fp16`    | fp16 / fp16          | none        | 1×              | ~0           |
| `hybrid`  | int8 trunk + fp16 heads | required    | ~1.7–1.9×       | minimal      |
| `int8`    | int8 / int8          | required    | ~2×             | head logits drift; softmax probs shift; MCTS strength can drop |

**Why hybrid is the default recommendation for Go on NPUs.**
Trunk operators (Conv + BN + ReLU) are ~95% of the FLOPs and survive int8
well — they consume one-hot stone planes and produce smoothly-distributed
feature maps where 256 quant bins is plenty.  The heads, though, are
small and end in nonlinearities (`softmax`, `tanh`, `softplus`) where
small numeric errors in logits become big probability shifts.  Examples:

* Policy logits in [−10, +10]: 1 logit unit of int8 quant noise (~0.08
  in absolute scale) becomes a >2× probability ratio after softmax.
* Value head: P(W) − P(L) crosses zero when the position is even.  int8
  on the value Linear pulls the decision boundary by several percent of
  win-rate.
* Score head: `softplus(stdev)` and `Mul(scoreMean, 20)` produce
  outputs in [−40, +40] points; one int8 quant bin (~0.3 points) is
  fine for stdev but visible in score utility blends.

Hybrid keeps these heads fp16 while leaving the trunk int8.  Heads are
<5% of FLOPs, so the throughput gap vs full int8 is small (often <5%).

---

## 3. Format auto-detection

The converter inspects ONNX `graph.input` to figure out which encoder the
model expects:

| Input layout                                   | Detected as | Encoder used by `tools/rknn_calibration.py` |
|------------------------------------------------|-------------|---------------------------------------------|
| `state_spatial: [B, C, H, W]`, `state_global: [B, G]` | `katago`    | V7 (22 spatial + 19 global), mirrors `src/katago_inputs.cpp` |
| `state: [B, C, H, W]`                          | `minigo`    | MiniGo (2·history+1 planes), mirrors `src/game.cpp::GoGame::encode` |

`board_size` is read from the spatial input's last two dims (must be
square).  `input_channels` and `input_global_channels` come from the
ONNX directly.  No model-side metadata is consulted; the same converter
handles both formats from a single binary.

`rknn-toolkit2` requires a fixed input shape.  We materialise one
explicitly with `--batch N` (default 1):

* `--batch 1` for live play (lowest single-move latency)
* `--batch 4` (or higher) for self-play workloads — multiple positions
  pack into one `rknn_run` call, amortising per-call host overhead.

You can ship both: the C++ runtime picks the `.rknn` file derived from
the loaded `.onnx` path by extension swap, so two builds (`best.rknn`
and `best_bs4.rknn`) coexist as long as you load the matching `.onnx`.

---

## 4. Calibration data generation

Quantisation needs a representative sample of input activations; the
toolkit computes per-tensor scales from the values it sees while
replaying calibration inputs.  For Go networks the input distribution
is dominated by:

* sparse one-hot stone planes (mostly 0, occasionally 1)
* liberty / area planes that vary with game phase (opening → endgame)
* board-history planes that encode the last N moves

Random-noise calibration would over-estimate plane variance and
under-estimate the prior of "empty intersection" — we'd quantise the
trunk against a distribution the network never actually sees.  Instead
`tools/rknn_calibration.py` **drives self-play with the model itself**
and dumps real positions across a spread of game phases.

### 4.1 What the helper does

For each of `--games` games:

1. Reset board.
2. At each ply, encode the current position (V7 for KataGo, MiniGo
   format otherwise) and run the ONNX model through `onnxruntime` to
   get a policy distribution.
3. Sample a move using softmax(policy / temperature), masked by
   simple-ko / suicide / pass legality.
4. Apply the move (full Tromp-Taylor simple-ko Go simulator, ported
   from `src/game.cpp`); update history; repeat until pass-pass or
   `2·board_size²` plies elapse.
5. Dump the encoded state every `--every` plies after the first
   `--skip-first` plies (defaults: skip 2, dump every 2).

The temperature is annealed from `--temperature` (default 1.0) toward
0.05 over the course of each game, so we get diverse openings and
focused endgames.  The skip+stride keeps the calibration set spread
across game phases rather than oversampling the empty board.

### 4.2 Encoder fidelity

The Python encoders mirror the C++ ones bit-for-bit on representable
positions:

| C++ source                          | Python equivalent                          |
|-------------------------------------|--------------------------------------------|
| `src/katago_inputs.cpp::encode_for_katago` | `tools/rknn_calibration.py::encode_katago_v7` |
| `src/game.cpp::GoGame::encode`     | `tools/rknn_calibration.py::encode_minigo`     |
| `src/game.cpp::GoGame::play / is_legal / is_ko_ban` | `tools/rknn_calibration.py::GoGame.{play, is_legal, is_ko_ban}` |
| Tromp-Taylor area flood-fill (`compute_area`) | `_compute_area` |

The same simplifications carry over — V7 ladder planes (14–17) and
encore start colors (20–21) are zeroed; the Tromp-Taylor area
approximation matches `src/katago_inputs.cpp`.  These planes are also
zero in the runtime, so calibration matches what the NPU sees in
production.

### 4.3 Output layout

Given `--output calib/best/`:

```
calib/best/
  state_0000.npy           # MiniGo: shape (input_channels, H, W)
  state_0001.npy
  ...
  dataset.txt              # one path per line (relative to cwd)
```

For KataGo dual-input models, each sample is **two** files:

```
calib/best/
  state_spatial_0000.npy   # shape (22, 9, 9)
  state_global_0000.npy    # shape (19,)
  ...
  dataset.txt              # one line per sample: "state_spatial_NNNN.npy state_global_NNNN.npy"
```

`rknn-toolkit2` expects this exact two-tokens-per-line format for
multi-input models — the converter passes the manifest through
unchanged.

### 4.4 How many samples?

Rockchip's docs suggest "≥100" for int8 calibration.  In practice:

| Sample count | Notes |
|-------------:|-------|
| 50           | Lower bound; trunk scales can be noisy.  Try 200 first. |
| 200          | Sweet spot for 9×9 Go; generation takes ~1–2 min on CPU. |
| 500+         | Marginal improvement; calibration time grows linearly. |

```bash
python tools/rknn_calibration.py \
    --onnx models/best.onnx \
    --output calib/best \
    --num-positions 200 \
    --games 32             # cap on simulator games if positions per game varies
```

---

## 5. fp16 conversion (no calibration)

Pipeline:

1. Detect input layout from ONNX.
2. `RKNN.config(target_platform=..., disable_rules=[...])`.
3. `rknn.load_onnx(model=onnx, inputs=..., input_size_list=...)`.
4. `rknn.build(do_quantization=False)` — weights stored fp16.
5. `rknn.export_rknn(rknn_path)`.

```bash
python tools/onnx_to_rknn.py \
    --onnx models/best.onnx --rknn models/best.rknn \
    --mode fp16 --target rk3588
```

Validation (run on the host):

```python
import onnxruntime as ort, numpy as np
from rknn.api import RKNN
np.random.seed(0)
x = np.random.randn(1, 17, 9, 9).astype(np.float32)

ort_out = ort.InferenceSession('models/best.onnx',
    providers=['CPUExecutionProvider']).run(None, {'state': x})

r = RKNN(verbose=False)
r.config(target_platform='rk3588')
r.load_onnx(model='models/best.onnx', inputs=['state'], input_size_list=[[1,17,9,9]])
r.build(do_quantization=False)
r.init_runtime()                            # x86 simulator
sim_out = r.inference(inputs=[x], data_format='nchw')
print(np.abs(ort_out[0] - sim_out[0]).max())   # should be O(1e-4)
r.release()
```

Expected `max_abs_diff` on `policy_logits` is around **2 × 10⁻⁴**
(pure fp16 quant error).

---

## 6. int8 conversion (full quant)

```bash
python tools/onnx_to_rknn.py \
    --onnx models/best.onnx --rknn models/best.rknn \
    --mode int8 --target rk3588 \
    --dataset calib/best/dataset.txt
```

Pipeline:

1. Detect inputs.
2. `RKNN.config(...)`.
3. `rknn.load_onnx(...)`.
4. `rknn.build(do_quantization=True, dataset=manifest)` — toolkit reads
   each calibration sample, runs it through the network in float, and
   derives per-tensor scales from the observed activation ranges.
5. `rknn.export_rknn(rknn_path)`.

Expected `max_abs_diff` on a 2-block toy model: **~1.4 × 10⁻¹** for
policy, **~9 × 10⁻²** for value.  These are realistic int8 errors and
are the reason hybrid mode exists — these errors compound through the
softmax / Sub of the value head.

---

## 7. Hybrid quantisation (the recommended path)

Hybrid is a two-step API:

* `rknn.hybrid_quantization_step1(dataset=...)` writes three files into
  the cwd: `<base>.quantization.cfg`, `<base>.model`, `<base>.data`.
* The user (or our tool) edits the cfg's `custom_quantize_layers` map,
  setting selected layer names to `float16`.
* `rknn.hybrid_quantization_step2(model_input=..., data_input=...,
  model_quantization_cfg=...)` re-quantises with the patched cfg.

The non-trivial part is **deciding which layers to flag as `float16`**.
The cfg generated by step1 lists every quantizable tensor as a key in
`quantize_parameters`, but RKNN's hybrid step2 refuses to re-quantise
some of them (the toolkit's "frozen scale" policy — see §10).  Our tool
handles this with a three-stage filter.

### 7.1 Stage 1 — graph trace

Walk back from each model output (`policy_logits`, `value`,
`score_mean`, `score_stdev`, `ownership`) up to `--trace-depth` ops
(default 6 — heads are typically 3–6 ops past the trunk).  Take the
**union of per-output chains, minus the intersection of all chains** —
this is the head set, with shared trunk subtracted.

```python
chains = {out: bfs_back(out, depth=6) for out in graph.outputs}
head_tensors = union(chains.values()) - intersect(chains.values())
```

The per-output chains overlap on every layer above the heads (the
trunk).  Subtracting the intersection leaves only tensors that
distinguish one head from another — exactly the layers we want fp16.

### 7.2 Stage 2 — keyword sweep over cfg layers

The cfg's tensor names sometimes don't match ONNX node output names
exactly (the toolkit rewrites the graph during step1).  We additionally
substring-match cfg layer names against a whitelist:

```
softmax, ownership, score_mean, score_stdev, value,
policy_logit, tanh, softplus, policy_fc, score_mean_head,
score_stdev_head, value_head, ownership_conv, policy_conv
```

Any match is added to the float16 set.

### 7.3 Stage 3 — exclusion filters

Three filters drop layers we know the toolkit will reject:

1. **Already float**: skip layers whose `quantize_parameters[layer].dtype`
   is `float32` / `float16` — forcing fp16 is a no-op or a downgrade.
2. **Toolkit artifacts**: skip names with toolkit-internal suffixes —
   `_sw`, `_mm`, `_rs`, `_expand`, `_int8`, `_to_int8`, `_to_float16`,
   `_float16`, names containing `#` (e.g. `_rs#1`), `__` (cvt
   intermediates), or `-rs` / `rs-` (reshape variants).  These are
   IR-level tensors that the rewriter materialised; their scales are
   derived from neighbours and refusing modification is by design.
3. **Frozen-op types**: skip outputs of `ReduceMean`, `ReduceMax`,
   `ReduceSum`, `GlobalAveragePool`, `GlobalMaxPool`, `Reshape`,
   `Squeeze`, `Unsqueeze`, `Transpose`, `Concat`, `Slice`, `Gather`,
   `Split` — the toolkit hardcodes their scale to the producing
   layer's scale.

The filtered set goes into `custom_quantize_layers: float16` and is
written back to the cfg.

```bash
python tools/onnx_to_rknn.py \
    --onnx models/best.onnx --rknn models/best.rknn \
    --mode hybrid --target rk3588 \
    --dataset calib/best/dataset.txt
```

Useful flags:

| Flag                  | Default | Effect |
|-----------------------|---------|--------|
| `--trace-depth N`     | 6       | Lower if you see too many trunk layers in the head set; raise if heads are nested deeper than usual. |
| `--proposal-size N`   | 16      | Sample size for `proposal=True` step1 (only if `--use-proposal`). |
| `--use-proposal`      | off     | Run rknn-toolkit2's auto-proposal during step1.  Off by default — proposal can incompatibly error with `does not support expand batch` on graphs whose batch is reshape-driven. |
| `--keep-intermediates`| off     | Don't delete the temp dir — useful for inspecting the generated cfg. |

### 7.4 Inspecting what the patcher did

```bash
python tools/onnx_to_rknn.py ... --keep-intermediates --verbose
# … outputs include …
[hybrid] working dir: /tmp/rknn_hybrid_XXXXXX
[hybrid] traced 6 head-only tensors (trace-depth=6)
[hybrid] forced 3 layers to float16
[hybrid]   - conv2d_2
[hybrid]   - value_int8
[hybrid]   - value_mm
```

In the working dir, the patched cfg shows the float16 set:

```yaml
custom_quantize_layers:
  conv2d_2: float16
  value_int8: float16
  value_mm: float16
quantize_parameters:
  state:
    qtype: asym
    dtype: float32      # toolkit auto-keeps inputs/outputs float32
    ...
  conv2d_2:
    qtype: asym
    dtype: int8         # but our patch overrides it to float16 below
    ...
```

The runtime then loads the .rknn with mixed precision —
[`rknn_compute.cpp`](src/rknn_compute.cpp) does not need any changes;
the toolkit bakes the per-tensor dtype into the file format.

---

## 8. Multi-batch builds

The `.rknn` file embeds a fixed batch size.  Recompile if you need
multiple batch sizes:

```bash
# bs=1 for live play
python tools/onnx_to_rknn.py --onnx models/best.onnx \
    --rknn models/best.rknn --mode fp16 --target rk3588 --batch 1

# bs=4 for self-play (better NPU utilisation)
python tools/onnx_to_rknn.py --onnx models/best.onnx \
    --rknn models/best_bs4.rknn --mode fp16 --target rk3588 --batch 4
```

The runtime reads the baked batch size from the rknn input-attr
(`input_attrs[0].dims[0]`) and pads single-sample requests up to the
batch.  See [README.md §"How batching works"](README.md#how-batching-works-gpu-backends-vs-rknn-npu)
for the trade-offs (latency vs throughput, padding cost on under-filled
batches, etc.).

---

## 9. Targets and SoC compatibility

The conversion cross-compiles for any single SoC:

| `--target`   | NPU cores | TOPS (int8) |
|--------------|----------:|------------:|
| `rk3562`     | 1         | 1           |
| `rk3566`     | 1         | 0.8         |
| `rk3568`     | 1         | 0.8         |
| `rk3576`     | 2         | 6           |
| `rk3588`     | 3         | 6           |

The C++ runtime auto-detects the actual SoC at startup
([`src/rknn_compute.cpp::detect_soc`](src/rknn_compute.cpp)) and
distributes server threads round-robin across cores via
`rknn_set_core_mask`.  A `.rknn` compiled for one SoC won't load on
another — recompile per SoC.

---

## 10. Known toolkit limitations

`rknn-toolkit2` (Rockchip's, not ours) has bugs that surface on certain
graph topologies.  These are **upstream** issues; this section
documents the workarounds we've baked into the converter.

### 10.1 `fold_constant` rank mismatch (v2.3.2)

**Symptom**

```
ValueError: Invalid rank for input: mean Got: 4 Expected: 2
  in graph_optimizer.fold_constant → onnxruntime.run
```

**Trigger**: any model where `ReduceMean(keepdims=False)` produces a
2-D tensor that fans out to both a broadcast `Mul` and a `Concat`,
followed by a `Gemm` (i.e., the KataGo / MiniGo global-pool path, the
SE block path).

**Cause**: the toolkit's `unsqueeze_to_4d_*` rules promote the 2-D
`mean` to 4-D before `Mul`/`Concat`, but `bypass_two_reshape` later
removes the 4-D→2-D reshape that would feed the `Gemm`.  `fold_constant`
then runs a sub-graph through ONNX Runtime where the `Gemm` sees a 4-D
input.

**Workaround**: use `rknn-toolkit2==2.3.0`.  The bug was introduced in
2.3.2.

```bash
pip install rknn-toolkit2==2.3.0 --force-reinstall --no-deps
```

### 10.2 Hybrid quant `not allowed to be modified` (v2.3.0 and v2.3.2)

**Symptom**

```
ValueError: The quantize_parameters['/value_head/ReduceMean_2_output_0_rs_sw']['scale']
            is not allowed to be modified!
  in hybrid_quantization_step2 → quant_utils.apply_hybrid_cfg
```

**Trigger**: hybrid mode on any model with a softmax-bearing value head
or an SE block (`Sigmoid` after `ReduceMean`).

**Cause**: the toolkit's `_sw` / `-rs` artifacts have scales derived
from neighbours.  When step2 reloads the cfg and re-derives them, it
asserts they're untouched — but the act of marking a parent layer
fp16 propagates scale changes.

**Workaround**: our tool's `_patch_hybrid_cfg` filters out
`ReduceMean` / `ReduceMax` / `Reshape` / `Concat` / `Slice` / `Gather`
outputs and toolkit-suffixed names (see §7.3).  This works for
truly-simple models.  For models with a softmax-bearing value head,
hybrid mode currently fails — fall back to **fp16 mode** until
Rockchip ships a fix, or use the fp16-keep flag carefully picked by
hand.

### 10.3 `fuse_conv_gather` `len() of unsized object`

**Symptom**

```
TypeError: len() of unsized object
  in graph_optimizer.fuse_ops → rules.reduce._p_fuse_conv_gather
```

**Trigger**: KataGo's policy head ends with `Conv → Slice` (selects
channel 0 of a 2-channel conv output for v12+ networks).  The
toolkit's `fuse_conv_gather` rule mis-handles the slice index.

**Workaround**: our tool's `RKNN.config(disable_rules=['fuse_conv_gather', ...])`
turns the rule off.  The unfused `Conv + Slice` runs natively at no
measurable cost (it's <1% of FLOPs).

### 10.4 KataGo (kata1-class) full-graph codegen hang

**Symptom**: `I rknn building ...` followed by ~indefinite spinning in
`initComputeZoneMapByStepsVector` — the toolkit's C++ codegen layer
gets stuck on KataGo's 10-block trunk with multiple parallel global-pool
residuals.

**Workaround**: as of toolkit v2.3.0/2.3.2 there is no fix.  Smaller
networks (4-block ResNet-with-SE) convert fine.  KataGo `kata1-b10c128`
(included in this repo at `models/kata1-b10c128.onnx`) is currently a
known-failing case.

Note that the runtime side has its own work to do too — at the moment
[`src/rknn_compute.cpp`](src/rknn_compute.cpp) throws `"KataGo format
requires the TensorRT backend"` at handle creation for dual-input ONNX
files (the runtime expects 1 input tensor; KataGo has 2).  Both gaps
are intended to close: this converter already produces valid
dual-input `.rknn` artifacts on smaller graphs, and the runtime
support is planned (see §14).

### 10.5 Toolkit/Python compatibility quirks

| Issue                                              | Fix |
|----------------------------------------------------|-----|
| `ModuleNotFoundError: No module named 'pkg_resources'` | `pip install 'setuptools<81'` (setuptools 81 dropped pkg_resources) |
| `AttributeError: module 'onnx' has no attribute 'mapping'` | `pip install 'onnx<1.18'` (onnx 1.18 dropped the legacy mapping) |
| `ValueError: This model does not support expand batch, Therefore, the 'proposal' function cannot be used!` | Don't pass `--use-proposal`; our default uses graph-trace head detection instead. |

These pins are baked into the quick-start `pip install` line in §1.

---

## 11. End-to-end validation

Run inference on three things and compare:

1. PyTorch (the source of truth for the model's intended outputs)
2. ONNX Runtime on the .onnx (catches export bugs)
3. RKNN simulator on the .rknn (catches conversion bugs)

```bash
# Reference: PyTorch ↔ ORT (existing in the repo for KataGo)
python tools/katago_parity_test.py \
    --katago-bin kata1-b10c128-s1141046784-d204142634.txt.gz \
    --onnx models/kata1-b10c128.onnx --board 9
```

For the .rknn → ORT comparison, use this snippet:

```python
import numpy as np, onnxruntime as ort
from rknn.api import RKNN

np.random.seed(0)
ONNX = 'models/best.onnx'
DATASET = 'calib/best/dataset.txt'
B, C, H, W = 1, 17, 9, 9   # adjust for KataGo

x = np.random.randn(B, C, H, W).astype(np.float32)
ort_out = ort.InferenceSession(ONNX, providers=['CPUExecutionProvider']).run(None, {'state': x})

for mode, build in [
    ('fp16', lambda r: r.build(do_quantization=False)),
    ('int8', lambda r: r.build(do_quantization=True, dataset=DATASET)),
]:
    r = RKNN(verbose=False)
    r.config(target_platform='rk3588')
    r.load_onnx(model=ONNX, inputs=['state'], input_size_list=[[B, C, H, W]])
    build(r); r.init_runtime()
    sim_out = r.inference(inputs=[x], data_format='nchw')
    for name, ort_t, sim_t in zip(['policy', 'value'], ort_out, sim_out):
        diff = float(np.abs(ort_t - sim_t).max())
        print(f'{mode:6s} {name:6s} max_abs_diff = {diff:.4e}')
    r.release()
```

Expected (a 2-block, 32-filter MiniGo toy model on 9×9):

```
fp16   policy max_abs_diff = 2.16e-04
fp16   value  max_abs_diff = 1.04e-04
int8   policy max_abs_diff = 1.42e-01
int8   value  max_abs_diff = 9.30e-02
```

The simulator output is bit-equivalent to what the NPU emits for the
same inputs (modulo NPU-specific layout transforms that the toolkit
applies symmetrically to both paths), so a green simulator check is a
strong signal that the NPU run will be correct.

---

## 12. Troubleshooting

| Symptom                                     | Likely cause / fix |
|---------------------------------------------|-------------------|
| `RKNN error -1 in rknn_init`                | `.rknn` was compiled for a different SoC.  Recompile with the right `--target`. |
| `[rknn_compute] expected 1 input tensor, got 2` | KataGo dual-input model; the C++ runtime currently expects single-input.  Until dual-input support lands in the runtime (see §14), run kata1 on TensorRT. |
| `fp16 numbers look right but quantised diverges` | Calibration set is too small / not representative.  Try `--num-positions 500`, `--temperature 1.0`, `--every 1`. |
| `proposal=True step1 fails with 'expand batch'` | Set `--use-proposal` off (it is by default in our tool); our trace+keyword head detection runs without proposal. |
| `Custom layer name not found in cfg`        | Toolkit renamed the tensor during graph rewriting.  Run with `--keep-intermediates --verbose` and inspect `<workdir>/<base>.quantization.cfg` for the actual layer name; pass it to the keyword sweep. |
| `pkg_resources` / `onnx.mapping` import errors | See §10.5 — pin `setuptools<81` and `onnx<1.18`. |
| `Permission denied` writing `.rknn`         | Check the parent dir exists (`mkdir -p`); the converter creates only the file, not parent dirs. |
| `Invalid rank for input: mean ...`          | Toolkit v2.3.2 fold_constant bug.  Downgrade to 2.3.0 (see §10.1). |
| `quantize_parameters[...] is not allowed to be modified` | Toolkit hybrid mode rejects modifying frozen scales.  Falls back to fp16 mode for now (§10.2). |

---

## 13. File reference

| File                                            | What it does |
|-------------------------------------------------|--------------|
| `tools/onnx_to_rknn.py`                         | Main converter; modes fp16/int8/hybrid; format auto-detect; cfg patcher. |
| `tools/rknn_calibration.py`                     | Self-play position dumper; ports `katago_inputs.cpp` + `game.cpp` to Python. |
| `models/*.onnx`                                 | Source ONNX models (KataGo or MiniGo). |
| `models/*.rknn`                                 | Compiled RKNN; sits next to the .onnx, found by extension swap. |
| `src/rknn_compute.cpp` / `include/rknn_compute.h` | Runtime backend (aarch64). |
| `third_party/rknn/rknn_api.h`                   | Vendored Rockchip C API header. |

---

## 14. Future work

* **kata1 conversion** — blocked on Rockchip's codegen fix
  (§10.4).  Workaround paths considered (graph rewriting to Conv-based
  GlobalAvg/Max + Flatten, weight-slicing instead of channel Gather,
  ORT pre-optimization) all clear `fold_constant` but still hang in
  the toolkit's C++ codegen.  Test with each new toolkit release.
* **Hybrid mode for SE/value-softmax models** — blocked on §10.2.
  Once Rockchip relaxes the frozen-scale assertion, our existing
  `_patch_hybrid_cfg` should produce working hybrid `.rknn` files for
  the full MiniGo `AlphaZeroNet` family.
* **Performance benchmarking on board** — the conversion side is in
  place; pair with `build/benchmark` runs on real Rockchip hardware
  to populate the throughput tables in `README.md` for hybrid vs fp16
  vs int8 across SoCs.
* **KataGo dual-input runtime path** — `src/rknn_compute.cpp`
  currently rejects dual-input ONNX (`expected 1 input tensor, got 2`)
  and throws `"KataGo format requires the TensorRT backend"` at
  handle creation.  Adding dual-input support — bind both
  `state_spatial` and `state_global` via `rknn_inputs_set`, demux the
  KataGo encoder output into the two buffers — is planned and
  separate from this conversion work.  The converter already emits
  valid dual-input `.rknn` files (on smaller graphs; kata1 itself is
  blocked on §10.4), so once the runtime lands the two pieces meet
  at the file boundary.
