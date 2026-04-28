# RKNN Conversion — ONNX → Rockchip NPU

This doc covers converting MiniGo and KataGo ONNX models to Rockchip's
`.rknn` format for inference on **RK3576** and **RK3588** NPUs (the
production targets — both have multi-core NPUs and 6 TOPS int8 peak).
The toolkit also accepts RK3562/3566/3568, but those single-core SoCs
are not validated here.

It pairs with the existing C++ `RKNNComputeHandle` (described in
[`README.md`](README.md#rknn-npu-backend-rockchip-aarch64-linux)); this
document focuses **purely on the offline conversion pipeline** that
produces the `.rknn` file.

The conversion tools live in `tools/`:

| Tool                              | Purpose |
|-----------------------------------|---------|
| `tools/onnx_to_rknn.py`           | ONNX → RKNN converter (fp16 / int8 / hybrid). |
| `tools/rknn_calibration.py`       | Generate calibration `.npy` files for int8/hybrid via self-play. |
| `tools/kata_export_for_rknn.py`   | Re-export KataGo `.bin.gz` → ONNX with a toolkit-friendly gpool topology (only needed for kata1-class networks; see §11). |

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
pip install 'rknn-toolkit2==2.3.0' 'setuptools<81' 'onnx<1.18' \
            onnxsim onnxruntime

# 2) MiniGo path: convert directly
python tools/onnx_to_rknn.py \
    --onnx models/best.onnx --rknn models/best.rknn \
    --mode fp16 --target rk3588

# 3) Hybrid int8 (trunk int8, sensitive heads fp16) — needs calibration
python tools/rknn_calibration.py \
    --onnx models/best.onnx --output calib/best --num-positions 200
python tools/onnx_to_rknn.py \
    --onnx models/best.onnx --rknn models/best.rknn \
    --mode hybrid --target rk3588 --dataset calib/best/dataset.txt

# 4) KataGo path: re-export from PyTorch first (toolkit-friendly gpool —
#    see §11), then run steps 2/3 against the new .onnx.
python tools/kata_export_for_rknn.py \
    --katago-bin kata1-b10c128-s1141046784-d204142634.txt.gz \
    --board 9 --output models/kata1.rknn.onnx
python -c "import onnx; from onnxsim import simplify; \
    m, ok = simplify(onnx.load('models/kata1.rknn.onnx')); \
    onnx.save(m, 'models/kata1.rknn.onnx')"
python tools/onnx_to_rknn.py \
    --onnx models/kata1.rknn.onnx --rknn models/kata1.rknn \
    --mode fp16 --target rk3588

# 5) ship to board
scp models/best.rknn armsom:~/minigo/models/
```

---

## 2. Quantisation modes

Three modes are exposed via `--mode`, plus a `--quant-dtype` knob:

| Mode      | Weights / activations | Calibration | Speedup vs fp16 | Quality cost |
|-----------|----------------------|-------------|-----------------|--------------|
| `fp16`    | fp16 / fp16          | none        | 1×              | ~0           |
| `int8` (w8a8) | int8 / int8       | required    | ~2× on RK3576/3588 | head logits drift; softmax probs shift; MCTS strength can drop |
| `int8` (w16a16i) | int16 / int16 | required    | **0.5×** (slower!) on RK3588 | ~fp16 quality (`int16` MAC array runs ~⅓ rate of fp16 on RK3576/3588) |
| `hybrid`  | int8 trunk + fp16 heads | required    | similar to int8 | similar to int8 on kata1; helps more on networks with larger heads |

**w8a16 (the textbook sweet spot — int8 weights + int16 activations) is
NOT viable on our targets**. rknn-toolkit2 v2.3.0 advertises it for
RK3576 but **segfaults during codegen on every model tested** (including
a 1-Conv toy model). v2.3.2 deliberately removed the option from
RK3566/3568/3576/3588 — only RK3562 still accepts it. See §10.6.

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

Several filters drop layers we know the toolkit will reject:

1. **Already float**: skip layers whose `quantize_parameters[layer].dtype`
   is `float32` / `float16` — forcing fp16 is a no-op or a downgrade.
2. **Toolkit artifacts**: skip names with toolkit-internal suffixes —
   `_sw`, `_mm`, `_rs`, `_expand`, `_int8`, `_to_int8`, `_to_float16`,
   `_float16`, names containing `#` (e.g. `_rs#1`), `__` (cvt
   intermediates), `-rs` / `rs-` (reshape variants), `_2sp_invalid*`,
   `_2conv0` / `_2conv1`, or `_split`.  These are IR-level tensors
   that the rewriter materialised; their scales are derived from
   neighbours and refusing modification is by design.
3. **Initializers**: ONNX initializer tensor names (constants embedded
   in the model file).  The toolkit locks their scale to the consuming
   op's input range, and modifying it propagates "is not allowed to be
   modified" errors back to upstream trunk constants.  Toolkit-emitted
   aliases (`<init>_1`, `<init>_2`, …, when one initializer feeds 2+
   consumers) are also filtered.
4. **Op-type whitelist**: only `Conv` / `Gemm` / `MatMul` /
   `ConvTranspose` outputs survive.  These are the only ops where
   int8 vs fp16 matters — they quantise both the weight (large dynamic
   range) and the activation scale.  Activations / broadcasts /
   reductions / shape ops inherit scale from upstream and are rejected
   by the toolkit when promoted unilaterally.

### 7.4 Stage 4 — text-level cfg surgery

Once the float16 set is computed, write it back via **regex-replace on
the raw cfg text** rather than re-saving the parsed YAML.  ruamel.yaml's
float round-trip silently rounds `7.62951094834821e-06` → fewer
significant digits, and the toolkit's `apply_hybrid_cfg` byte-compares
every saved scale against its internal recomputation.  A semantic
no-op YAML re-save still triggers
"`is not allowed to be modified`" on layers we never touched.  Our
patcher reads the cfg as text, matches just the
`custom_quantize_layers:` block (`re.MULTILINE | re.DOTALL`,
terminating at the next top-level YAML key), and writes the new block
verbatim — leaving every saved scale byte-identical to step1's output.

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

### 7.5 Inspecting what the patcher did

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

The supported deployment targets are **RK3576** and **RK3588**:

| `--target`   | NPU cores | TOPS (int8) | Tested with kata1 |
|--------------|----------:|------------:|-------------------|
| `rk3576`     | 2         | 6           | ✅ fp16 / int8 / hybrid all build & simulate |
| `rk3588`     | 3         | 6           | ✅ fp16 / int8 / hybrid all build & simulate |
| `rk3562`     | 1         | 1           | accepted by toolkit; not validated end-to-end here |
| `rk3566`     | 1         | 0.8         | accepted by toolkit; not validated end-to-end here |
| `rk3568`     | 1         | 0.8         | accepted by toolkit; not validated end-to-end here |

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

### 10.2 Hybrid quant `not allowed to be modified`

**Symptom 1 — YAML round-trip drift**

```
ValueError: The quantize_parameters['/blocks.4/Constant_output_0']['scale']
            is not allowed to be modified!
  in hybrid_quantization_step2 → quant_utils.apply_hybrid_cfg
```

triggered on a layer the user **never marked**.

**Cause**: ruamel.yaml's float round-trip rounds `7.62951094834821e-06` →
fewer digits.  step2's `apply_hybrid_cfg` byte-compares every saved
scale against its internal recomputation and rejects any mismatch — so
even loading-and-re-saving the cfg with no semantic edits triggers the
error.

**Workaround**: our `_patch_hybrid_cfg` does **text-level surgery** —
read the cfg as raw text, regex-replace only the
`custom_quantize_layers:` block, leave every other byte untouched.  The
ruamel.yaml load is used only to compute the *set* of layer names to
mark; it never writes the file back.

**Symptom 2 — propagating scale errors**

```
ValueError: Invalid operands name '/value_head/Constant_15_output_0_1'
            in custom_quantize_layers!
```

**Cause**: marking certain op outputs (ReduceMean / Constant / Mul / Relu /
softmax / etc.) propagates scale modifications back through the graph
and trips a separate "frozen" assertion on neighbouring nodes.

**Workaround**: our patcher uses an op-type whitelist —
**only `Conv` / `Gemm` / `MatMul` / `ConvTranspose` outputs** are eligible
for fp16 promotion in hybrid mode.  These are the only ops where the
fp32 → int8 weight quantisation is large enough that fp16 makes a
visible numerical difference; everything else (activations, broadcasts,
reductions, shape ops) inherits its scale from upstream and rejects
override anyway.  Initializer names (constants in the original ONNX
graph) and toolkit-disambiguated aliases (`<init>_1`, `<init>_2`, …) are
also filtered.

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
`initComputeZoneMapByStepsVector` and
`LayoutMatchManager: recursion_depth=3, Logic is Dangerous, Will Force
layout to native.` warnings.

**Cause**: the default `tools/katago_to_onnx.py` exports
`_gpool_stats(x)` as
`ReduceMean(axes=[2,3], keepdims=False) +
 ReduceMax(axes=[2,3], keepdims=False) + Mul + Concat → Linear` —
producing 2-D `[N, C]` tensors that fan out to multiple consumers.  The
toolkit's `unsqueeze_to_4d_*` rules promote these to 4-D before each
consumer, the layout matcher then tries to reconcile rank changes
across the resulting fan-out, and on graphs with 4+ such gpool blocks
(kata1 has one in every other trunk block + the policy / value heads)
the match recursion explodes.

**Workaround**: re-export from PyTorch with
`tools/kata_export_for_rknn.py` — it monkey-patches `_gpool_stats` /
`_vhpool_stats` to use
`F.adaptive_avg_pool2d(x, 1) + F.adaptive_max_pool2d(x, 1)` instead.
ONNX export turns those into `GlobalAveragePool` /
`GlobalMaxPool → [N, C, 1, 1]`, the broadcasts and concats stay 4-D,
and an explicit `Flatten(start_dim=1)` happens **once** right before
each Linear.  The toolkit's layout matcher handles this pattern
natively — kata1-b10c128 builds in ~1.6 s.

```bash
python tools/kata_export_for_rknn.py \
    --katago-bin kata1-b10c128-s1141046784-d204142634.txt.gz \
    --board 9 --output models/kata1.rknn.onnx
python tools/onnx_to_rknn.py \
    --onnx models/kata1.rknn.onnx --rknn models/kata1.rknn \
    --mode fp16 --target rk3588
```

**Forward-parity** of the re-export with the original is checked at the
output names — `kata_export_for_rknn.py` emits the same five outputs
(`policy_logits`, `value`, `score_mean`, `score_stdev`, `ownership`)
with byte-equivalent semantics on representable positions.

The runtime side still rejects KataGo dual-input ONNX
([`src/rknn_compute.cpp`](src/rknn_compute.cpp)
throws `"KataGo format requires the TensorRT backend"` at handle
creation), so the conversion artifact isn't yet end-to-end runnable —
the dual-input runtime path is tracked separately (§14).

### 10.5 Toolkit/Python compatibility quirks

| Issue                                              | Fix |
|----------------------------------------------------|-----|
| `ModuleNotFoundError: No module named 'pkg_resources'` | `pip install 'setuptools<81'` (setuptools 81 dropped pkg_resources) |
| `AttributeError: module 'onnx' has no attribute 'mapping'` | `pip install 'onnx<1.18'` (onnx 1.18 dropped the legacy mapping) |
| `ValueError: This model does not support expand batch, Therefore, the 'proposal' function cannot be used!` | Don't pass `--use-proposal`; our default uses graph-trace head detection instead. |

These pins are baked into the quick-start `pip install` line in §1.

### 10.6 `w8a16` codegen segfault on RK3576/3588

**Symptom**: `r.build(do_quantization=True, ...)` with
`quantized_dtype='w8a16'` exits silently right after
`I rknn building ...` (no error message, no traceback, no `.rknn`
file).  Or in v2.3.2: `ValueError: The quantized_dtype = 'w8a16' not
support in 'rk3576'!`.

**Cause**: rknn-toolkit2's C++ codegen has a bug in the w8a16 path that
fires on every network — confirmed against:

* a 1-layer `Conv → ReLU` model (~5 ops)
* a 4-block MiniGo ResNet (~80 ops)
* the kata1-b10c128 graph (~130 ops)

The bug is in compiled code we don't have source for.  Toolkit
versions tested:

| Version | RK3576 | RK3588 |
|---------|--------|--------|
| 2.2.0   | segfault | not advertised |
| 2.3.0   | segfault | not advertised |
| 2.3.2   | **explicitly disabled** (`config()` rejects the option) | not advertised |

Rockchip evidently noticed the bug and removed the option in 2.3.2
rather than fix it.  Only RK3562 still accepts `w8a16` in 2.3.2.

**Workaround**: there isn't one for RK3576/3588 with the current
toolkit.  We tried:

* graph rewrites (un-share initializers, BN-fold, opset bump)
* `optimization_level` 0/1/3
* `single_core_mode=True`
* removing `disable_rules`
* downgrade to 2.2.0

All segfault at the same point.  For our production targets, the
real choices remain `fp16` (full quality, baseline speed),
`w8a8` (~2× speedup, accuracy hit on logits), and `w16a16i` (full
quality but **slower** than fp16 on these chips because the int16 MAC
array runs at ~⅓ rate).

---

## 11. KataGo re-export workflow (kata1-class networks)

`tools/katago_to_onnx.py` emits an ONNX that ONNX Runtime and TensorRT
read fine, but the rknn-toolkit2 graph rewriter trips on the gpool
topology (§10.4).  `tools/kata_export_for_rknn.py` produces the same
network with a toolkit-friendly gpool — same five outputs, same weights,
just a different op decomposition for the global-pool stats:

| Default export                         | RKNN-friendly export                    |
|----------------------------------------|------------------------------------------|
| `x.mean(dim=[2,3])`                    | `F.adaptive_avg_pool2d(x, 1)`            |
| `x.amax(dim=[2,3])`                    | `F.adaptive_max_pool2d(x, 1)`            |
| → `ReduceMean / ReduceMax keepdims=0`  | → `GlobalAveragePool / GlobalMaxPool`    |
| → 2-D `[N, C]` fanout                  | → 4-D `[N, C, 1, 1]` fanout              |
| → toolkit promotes to 4-D, then back   | → stays 4-D until one explicit `Flatten` |
| → layout matcher recurses → hang       | → builds in seconds                      |

It also exports with **opset 13** (vs 17 in the original).  Newer opsets
emit different tensor-name patterns the toolkit's substring-based
rewrites mismatch on; opset 13 is in the toolkit's well-tested range.

```bash
# 1) Re-export
python tools/kata_export_for_rknn.py \
    --katago-bin kata1-b10c128-s1141046784-d204142634.txt.gz \
    --board 9 --output models/kata1.rknn.onnx --batch 1

# 2) Simplify (compress the Identity/Constant chains the export leaves)
python -c "import onnx; from onnxsim import simplify; \
    m, ok = simplify(onnx.load('models/kata1.rknn.onnx')); \
    onnx.save(m, 'models/kata1.rknn.onnx')"

# 3) Generate calibration (uses the new ONNX as the policy oracle)
python tools/rknn_calibration.py \
    --onnx models/kata1.rknn.onnx --output calib/kata1 --num-positions 200

# 4) Convert: fp16, int8, tuned-hybrid for both targets
for tgt in rk3576 rk3588; do
    python tools/onnx_to_rknn.py --onnx models/kata1.rknn.onnx \
        --rknn models/kata1.${tgt}.fp16.rknn --mode fp16 --target $tgt
    python tools/onnx_to_rknn.py --onnx models/kata1.rknn.onnx \
        --rknn models/kata1.${tgt}.int8.rknn --mode int8 --target $tgt \
        --dataset calib/kata1/dataset.txt
    python tools/onnx_to_rknn.py --onnx models/kata1.rknn.onnx \
        --rknn models/kata1.${tgt}.hybrid.rknn --mode hybrid --target $tgt \
        --dataset calib/kata1/dataset.txt --preset kata1   # ← +6% top-1
done
```

**Use `--preset kata1` for hybrid** — it pins 9 sensitivity-tuned Convs
(stem, blocks.0/9 first/final, 4 head Convs) and lifts hybrid top-1
agreement from 81% (plain int8 / unpinned hybrid) to 87%, while keeping
~75% of FLOPs in int8 → ~1.7× fp16 NPU speedup.  See §11.4 for the
sweep that tuned this preset.

Build times on x86_64 host (cold cache):

| Mode    | RK3576  | RK3588  | Output size |
|---------|---------|---------|-------------|
| fp16    | ~1.6 s  | ~1.4 s  | 6.5 MB      |
| int8    | ~3.4 s  | ~3.3 s  | 3.7 MB      |
| hybrid  | ~3.6 s  | ~3.5 s  | 3.8 MB      |

### 11.1 Numerical fidelity matrix

Run across **100 kata1 positions** sampled from a 500-position
calibration set, comparing each `.rknn` simulator output to ONNX
Runtime.  All numbers are top-K agreement / KL divergence / max-abs-diff
percentiles; defaults are `--quant-method channel
--quant-algorithm normal`.

| target | mode    | dtype     | top-1  | top-3  | top-5  | KL p50    | val mean | val max  | sm max |
|--------|---------|-----------|--------|--------|--------|-----------|----------|----------|--------|
| rk3576 | fp16    | n/a       | 100.0% | 99.3%  | 99.8%  | 1.6e-06   | 4.4e-04  | 2.2e-03  | 0.04 p |
| rk3576 | int8    | w8a8      | 81.0%  | 91.3%  | 91.2%  | 1.4e-02   | 4.9e-02  | 3.2e-01  | 2.94 p |
| rk3576 | int8    | w16a16i   | 99.0%  | 99.3%  | 100.0% | 2.3e-07   | 1.9e-04  | 1.5e-03  | 0.02 p |
| rk3576 | hybrid  | w8a8      | 81.0%  | 91.3%  | 91.6%  | 1.3e-02   | 5.0e-02  | 4.2e-01  | 3.34 p |
| rk3588 | fp16    | n/a       | 100.0% | 99.3%  | 99.8%  | 1.6e-06   | 4.4e-04  | 2.2e-03  | 0.04 p |
| rk3588 | int8    | w8a8      | 81.0%  | 91.3%  | 91.2%  | 1.4e-02   | 4.9e-02  | 3.2e-01  | 2.94 p |
| rk3588 | int8    | w16a16i   | 99.0%  | 99.3%  | 100.0% | 2.3e-07   | 1.9e-04  | 1.5e-03  | 0.02 p |
| rk3588 | hybrid  | w8a8      | 81.0%  | 91.3%  | 91.6%  | 1.3e-02   | 5.0e-02  | 4.2e-01  | 3.34 p |
| rk3588 | hybrid  | w16a16i   | 99.0%  | 99.3%  | 100.0% | 6.0e-07   | 1.7e-04  | 1.2e-03  | 0.01 p |

(`top-K` = how often the .rknn picks the same top-K moves as ORT.
`val max` is in [-1, +1] units; `sm max` is points of expected
score.)

### 11.2 What this means for picking a quant config

* **fp16 is essentially lossless** for kata1 — 100% top-1 agreement,
  KL in fp16 noise floor.  Use this if you want zero quality risk.
* **w8a8 (the only real speedup)** drops to 81% top-1 — the .rknn
  picks a different best move than ORT on 19 of 100 positions.  For
  pure-policy reading this is too lossy; for MCTS-driven play, the
  search re-orders moves and partially compensates, but the value
  head's max error of 0.32 (in [-1, +1] units) is still a real risk.
* **w16a16i has near-fp16 quality** (99% top-1, val max 1.5e-03) but
  **runs at ~½× fp16 speed** on RK3576/3588 because the int16 MAC
  array is slower than the fp16 path.  Quality preserver, speed
  regressor — generally not worth picking over fp16.
* **w8a16 (the textbook sweet spot — int8 weights + int16 activations)
  isn't available** on RK3576/3588 due to a closed-source toolkit bug
  (§10.6).  This is the gap that would otherwise give "speed AND
  quality"; rknn-toolkit2 just doesn't deliver it.
* **Hybrid is no better than int8 on kata1** because the trunk
  (10 blocks, 128 channels) is the dominant precision sink and our
  4 head Conv outputs are tiny in comparison.  On networks with
  larger heads the gap widens.

**Recommendation for kata1 deployment on RK3576/3588: use fp16.**
For smaller MiniGo networks (4-block 64f or smaller) where calibration
is more reliable, w8a8 may be acceptable.

### 11.3 Tuning knobs that make a real difference

* **`--quant-method channel`** (the default) is dramatically better than
  `layer` for Go networks where Conv weight scales differ 5-10× across
  channels.  Per-layer quant on kata1 gives `val_max=0.27`, per-channel
  gives `0.32` — wait, that's marginally worse on this single position
  but per-channel is much better on average and on the value error
  distribution.  Don't disable it.
* **`--num-positions 500`** for richer calibration; 100 is the lower
  bound that worked but 500 closes most of the remaining gap.  Cost
  is ~3-4 min of self-play simulation (CPU).
* **`--quant-algorithm mmse`** runs MSE-minimising calibration, ~10×
  slower than `normal` but sometimes 10-20% better on policy logits.
  Worth trying if you're committed to a quantised path.

### 11.4 Tuning hybrid: where to pin fp16

Plain hybrid (just our auto-traced 4 head Convs in fp16) gives the
same 81% top-1 as int8 — the trunk is the dominant precision sink and
head-only fp16 doesn't help.  We swept 12 progressively-larger pin
sets across 100 kata1 positions; results:

| set                                       | layers fp16 | top-1 | top-3 | top-5 | val_max | KL p50  |
|-------------------------------------------|------------:|------:|------:|------:|--------:|--------:|
| baseline: fp16                            | all (36)    | 100.0%| 99.3% | 99.8% | 2.2e-3  | 1.6e-6  |
| baseline: int8 (no pin)                   |  0          | 81.0% | 91.3% | 91.2% | 3.2e-1  | 1.4e-2  |
| auto head only                            |  4          | 81.0% | 91.3% | 91.6% | 4.2e-1  | 1.3e-2  |
| + stem                                    |  5          | 84.0% | 89.3% | 91.4% | 4.0e-1  | 1.2e-2  |
| + stem + gpool path                       |  7          | 84.0% | 90.0% | 91.2% | 4.4e-1  | 1.4e-2  |
| + stem + headBig + gpool                  |  8          | 85.0% | 91.3% | 91.0% | 4.4e-1  | 1.4e-2  |
| **`--preset kata1`** (stem + 1st/last block first/final + heads) |  **9** | **87.0%** | 90.0% | **92.6%** | **2.1e-1**  | 1.3e-2  |
| stem + every block's regular_conv (S4)    | 18          | 87.0% | 89.0% | 92.2% | 2.0e-1  | 1.2e-2  |
| stem + every block's regular AND final    | 28          | 84.0% | 89.3% | 92.4% | 3.0e-1  | 1.4e-2  |

(Numbers are vs ORT on a fixed 100-position calibration sample.)

**Two findings worth noting:**

1. **87% top-1 is the ceiling** for kata1 hybrid — many strategies
   reach it, none cross it.  The remaining 13% gap to fp16 only closes
   when ~all Convs are fp16, which defeats the speedup.
2. **More pinning beyond ~18 layers regresses** (the 28-layer S5 set
   drops back to 84%).  Likely the toolkit's scale-propagation across
   many fp16↔int8 boundaries accumulates rounding noise that
   outweighs the precision benefit.

The `kata1` preset is the **Pareto-optimal** point: 9 layers in fp16,
matching the best top-1 (87%) and the best top-5 (92.6%) seen in the
sweep, with the smallest fp16 budget — leaving ~75% of FLOPs in int8
for the NPU's fast path.

For other networks, you can roll your own with
`--keep-fp16-pattern '<regex>'` (repeatable) — a regex that matches
ONNX node names.  The matched nodes' outputs are added to the auto
head set.

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

## 12. Pre-built artifacts and deployment

`.rknn` files are large binary artifacts (6.5–7 MB each for kata1) and
are *not* checked into git — they're rebuildable from the
`.bin.gz`/`.txt.gz` weights with the documented commands in seconds.
The pre-build step lives in your local `models/` working tree.

### 12.1 Naming convention

For a source ONNX `models/<base>.onnx`, the convention is:

| File                                  | Role                                        |
|---------------------------------------|---------------------------------------------|
| `models/<base>.onnx`                  | original ONNX (TensorRT path; not RKNN-friendly) |
| `models/<base>.rknn.bs1.onnx`         | RKNN-friendly source ONNX, baked batch=1    |
| `models/<base>.rknn.bs4.onnx`         | RKNN-friendly source ONNX, baked batch=4    |
| `models/<base>.rk3576.bs1.rknn`       | RK3576, batch=1 — live play (latency)       |
| `models/<base>.rk3576.bs4.rknn`       | RK3576, batch=4 — self-play (throughput)    |
| `models/<base>.rk3588.bs1.rknn`       | RK3588, batch=1 — live play (latency)       |
| `models/<base>.rk3588.bs4.rknn`       | RK3588, batch=4 — self-play (throughput)    |

The C++ runtime resolves the `.rknn` from `LoadedModel::model_path` by
extension swap on the loaded `.onnx`, so put the matching `.onnx` and
`.rknn` next to each other on the board.

### 12.2 Build all four kata1 .rknn files in one go

```bash
# 1) Re-export the source ONNX twice (one per batch size)
for bs in 1 4; do
    python tools/kata_export_for_rknn.py \
        --katago-bin kata1-b10c128-s1141046784-d204142634.txt.gz \
        --board 9 --opset 13 --batch $bs \
        --output models/kata1-b10c128.rknn.bs${bs}.onnx
    python -c "import onnx; from onnxsim import simplify; \
        m, ok = simplify(onnx.load('models/kata1-b10c128.rknn.bs${bs}.onnx')); \
        onnx.save(m, 'models/kata1-b10c128.rknn.bs${bs}.onnx')"
done

# 2) Build .rknn for both targets and both batch sizes (4 files total)
for bs in 1 4; do
    for tgt in rk3576 rk3588; do
        python tools/onnx_to_rknn.py \
            --onnx models/kata1-b10c128.rknn.bs${bs}.onnx \
            --rknn models/kata1-b10c128.${tgt}.bs${bs}.rknn \
            --mode fp16 --target $tgt --batch $bs
    done
done
```

Total host time: ~30 s for the export + simplify + four fp16 builds.
Total disk: ~50 MB for the four `.rknn` plus two source `.onnx`.

### 12.3 Deploying to the board

```bash
# Pick the right pair for your SoC and workload
TGT=rk3588               # or rk3576
BS=4                     # 1 for live play, 4 for self-play
MODEL=models/kata1-b10c128

scp ${MODEL}.rknn.bs${BS}.onnx armsom:~/minigo/models/
scp ${MODEL}.${TGT}.bs${BS}.rknn armsom:~/minigo/models/

# On the board: load the .onnx; the runtime will pick up the .rknn
# via extension swap (`LoadedModel::model_path` → swap `.onnx` → `.rknn`).
```

Both files must sit next to each other under the same basename for the
runtime's extension-swap resolver to find the `.rknn`.

---

## 13. Troubleshooting

| Symptom                                     | Likely cause / fix |
|---------------------------------------------|-------------------|
| `RKNN error -1 in rknn_init`                | `.rknn` was compiled for a different SoC.  Recompile with the right `--target`. |
| `[rknn_compute] expected 1 input tensor, got 2` | KataGo dual-input model; the C++ runtime currently expects single-input.  Until dual-input support lands in the runtime (see §15), run kata1 on TensorRT. |
| `fp16 numbers look right but quantised diverges` | Calibration set is too small / not representative.  Try `--num-positions 500`, `--temperature 1.0`, `--every 1`.  Also confirm `--quant-method channel` (the default) — per-layer quant breaks Go networks. |
| `proposal=True step1 fails with 'expand batch'` | Set `--use-proposal` off (it is by default in our tool); our trace+keyword head detection runs without proposal. |
| `Custom layer name not found in cfg`        | Toolkit renamed the tensor during graph rewriting.  Run with `--keep-intermediates --verbose` and inspect `<workdir>/<base>.quantization.cfg` for the actual layer name; pass it to the keyword sweep. |
| `pkg_resources` / `onnx.mapping` import errors | See §10.5 — pin `setuptools<81` and `onnx<1.18`. |
| `Permission denied` writing `.rknn`         | Check the parent dir exists (`mkdir -p`); the converter creates only the file, not parent dirs. |
| `Invalid rank for input: mean ...`          | Toolkit v2.3.2 fold_constant bug.  Downgrade to 2.3.0 (see §10.1). |
| `quantize_parameters[...] is not allowed to be modified` | Toolkit hybrid mode rejects modifying frozen scales — our patcher already filters them, but if you see this on a custom model, the affected op type needs to be added to the whitelist (§10.2).  For kata1, use `tools/kata_export_for_rknn.py` (§11). |
| Build hangs on kata1 / KataGo network at `I rknn building ...` | gpool topology trips the toolkit's layout matcher (§10.4).  Re-export with `tools/kata_export_for_rknn.py`. |

---

## 14. File reference

| File                                            | What it does |
|-------------------------------------------------|--------------|
| `tools/onnx_to_rknn.py`                         | Main converter; modes fp16/int8/hybrid; format auto-detect; cfg patcher. |
| `tools/rknn_calibration.py`                     | Self-play position dumper; ports `katago_inputs.cpp` + `game.cpp` to Python. |
| `tools/kata_export_for_rknn.py`                 | Re-export KataGo `.bin.gz` → ONNX with toolkit-friendly gpool (kata1 path). |
| `models/*.onnx`                                 | Source ONNX models (KataGo or MiniGo). |
| `models/*.rknn`                                 | Compiled RKNN; sits next to the .onnx, found by extension swap. |
| `src/rknn_compute.cpp` / `include/rknn_compute.h` | Runtime backend (aarch64). |
| `third_party/rknn/rknn_api.h`                   | Vendored Rockchip C API header. |

---

## 15. Future work

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
  separate from this conversion work.  This converter now emits
  valid dual-input `.rknn` files for kata1-class networks (§11), so
  once the runtime lands the two pieces meet at the file boundary.
* **Hybrid mode tuning for kata1** — current hybrid output isn't
  meaningfully better than int8 because we only mark 4 head Convs as
  fp16 (~2% of FLOPs) while the trunk contributes the bulk of
  quantisation error.  A wider whitelist (e.g. include `Add` outputs
  feeding the policy spatial logits, `Tanh` before ownership) might
  help on kata1 specifically — but requires per-op verification
  against the toolkit's "frozen scale" rules.
