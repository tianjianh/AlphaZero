# Inference Backends

Per-backend implementation notes, the shared Context/Handle design,
measured performance, and platform notes.  The engine-side architecture
(MCTS / NNEvaluator) is in [ENGINE.md](ENGINE.md).


### CUDA GPU Backend (NVIDIA)

`CUDAComputeHandle` (`src/cuda_compute.cu`) implements the forward pass using
FP16 Tensor Cores via **CUTLASS GEMM** for conv3x3 (with im2col precompute)
and hand-written WMMA kernels for FC/head layers.  FP32 fallback for SM < 7.0.

| Kernel | Purpose |
|---|---|
| `transpose_nchw_fp32_to_fp16` | Fused transpose + FP32→FP16 conversion |
| `CutlassGemm` (im2col + GEMM) | **CUTLASS** conv3x3: software-pipelined tensor core GEMM |
| `conv3x3_wmma_bn` | Legacy WMMA fallback (used for FP32 path) |
| `conv1x1_bn_relu_reshape_fp16` | FP16 1×1 conv + BN + ReLU + layout reshape |
| `fc_bias_relu_fp16` | FP16 FC + bias + ReLU |
| `fc_bias_softmax_fp16_to_fp32` | FP16→FP32 FC + softmax (policy head) |
| `fc_bias_tanh_fp16_to_fp32` | FP16→FP32 FC + tanh (value head) |
| `score_softmax_ev_fp16` | FP16→FP32 softmax over bins → expected value (score head) |

- **FP16 weights & activations**: halves memory bandwidth for all buffers
- **FP32 BN scale/bias and FC bias**: small per-channel params stored natively as float (avoids conversion overhead)
- **FP32 accumulator**: WMMA accumulates in FP32 for numerical stability, converts to FP16 on write-back
- **Softmax/tanh in FP32**: final outputs computed in full precision
- Multi-arch: SASS for SM 75 (Turing), 80/86 (Ampere), 89 (Ada) + compute_90 PTX (Hopper/Blackwell)

### TensorRT GPU Backend (NVIDIA)

`TensorRTComputeHandle` (`src/tensorrt_compute.cpp`) uses NVIDIA TensorRT for
optimized inference.  TensorRT parses the ONNX model directly using its own
ONNX parser and applies automatic optimizations:

- **FP16 precision**: enabled automatically when the GPU supports it
- **Layer fusion**: TensorRT fuses conv+BN+ReLU, eliminating intermediate buffers
- **Kernel auto-tuning**: TensorRT benchmarks multiple kernel implementations
  at engine build time and selects the fastest for each layer on the target GPU
- **Engine caching**: the compiled engine is serialized to disk
  (`<model>.trt_<gpu_name>_b<N>.engine`) and reloaded on subsequent runs,
  skipping the build step (which can take 10-60s)
- **Dynamic batching**: optimization profile covers batch sizes 1 to `max_batch`

The design follows the same Context/Handle pattern:
- **`TRTDeviceState`** (per GPU): `ICudaEngine*` + `cudaStream_t` + `IRuntime*`
- **`TensorRTComputeHandle`** (per server thread): `IExecutionContext*` + I/O buffers

### OpenCL GPU Backend

`OpenCLComputeHandle` (`src/opencl_compute.cpp`, kernels in
`src/opencl_kernels.h`) runs **all three model architectures** — the
MiniGo KataGo-style ResNet (SE + GPool blocks), the GoViT transformer
(GQA + directional relative bias), and KataGo-V7 dual-input networks
(both trainable-KataGoNet exports and converted stock kata1 nets,
mish/relu autodetected from the graph ops).

Three precision tiers, chosen per device at context init
(`MINIGO_OPENCL_PRECISION` = `fp32` | `fp16` | `fp16-portable` | `auto`):

| Tier | Storage | Math | Hardware |
|---|---|---|---|
| `fp32` | float | float | any OpenCL 1.2 device |
| `fp16` (portable) | half via core `vload_half`/`vstore_half` | fp32 | any OpenCL 1.2 device — **no `cl_khr_fp16` needed** |
| `fp16` + MMA | half | tensor cores, fp32 accumulate | NVIDIA (inline-PTX `mma.sync.m16n8k16`, probe-compiled) |

Kernel design:

- **One unified implicit-GEMM kernel** covers every conv (k=1/3/5,
  im2col gathered on the fly — never materialized) and every FC.  Its
  fused epilogue applies pre-fused BN, row bias, the per-(channel,image)
  global-pool bias injection (pre- or post-BN), residual add and the
  activation — so a whole residual block is 3-4 launches and a KataGo
  gpool block needs **zero** separate element-wise kernels.
- The gather hoists all per-column index math out of the K-loop and
  builds the K→(ic,kh,kw) decomposition in a tiny local-memory LUT once
  per tile (integer division is ~25 emulated instructions on GPUs;
  doing it per element measurably dominated the kernel).
- The tensor-core version stages packed-half tiles in local memory with
  a bank-conflict-free 36-uint row stride and issues
  `mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32` from inline PTX —
  fp16 inputs, **fp32 accumulation**.  Compiled as its own program and
  probe-tested at init; any failure falls back to portable fp16.
- **Flash-style fused attention** for ViT: one work-group per
  (image, head), online softmax in fp32, K/V tiles staged in local
  memory, and the directional rel-position bucket recomputed from token
  coordinates — the [hw, hw] bias matrix is never materialized.
- LayerNorm / softmax / global-pool statistics always accumulate in
  fp32; outputs pack GPU-side into one buffer read back once per batch.

Measured on an NVIDIA A40 (9×9 board, batch 256, evals/s):

| Model | fp32 | fp16+MMA |
|---|---|---|
| resnet b10c128 | 11.1k | **25.4k** |
| katago b10c128 | 11.0k | **24.1k** |
| vit d192×8 | 6.0k | 7.0k |

`MINIGO_OPENCL_PROFILE=1` prints a per-kernel GPU-time summary after
every batch (uses `CL_QUEUE_PROFILING_ENABLE`).

Numerical verification against PyTorch reference outputs
(`scripts/make_test_vectors.py` + `build/verify`): fp32 matches to
~1e-7 on all five model formats; fp16 policy logits within 4e-2.

### Metal GPU Backend (macOS)

`MetalComputeHandle` (`src/metal_compute.mm`) uses **MPSGraph** (Metal
Performance Shaders Graph) to build the entire forward pass as a computation
graph.  Each `predict_batch()` call feeds inputs through the pre-compiled
graph via `graph.run()`.

- **FP16 compute**: weights and activations in half-precision with FP32
  accumulation.  Softmax and tanh run in FP32 for numerical stability.
- **Kernel fusion**: MPSGraph fuses conv+BN+ReLU automatically
- **`@autoreleasepool`**: wraps each `graph.run()` call, matching KataGo's
  Metal backend pattern for correct ObjC object lifecycle on server threads
- **Unified memory**: CPU and GPU share the same memory (no explicit copies)

### RKNN NPU Backend (Rockchip, aarch64 Linux)

`RKNNComputeHandle` (`src/rknn_compute.cpp`) runs inference on Rockchip's
on-chip NPU via the `librknnrt` runtime.  Unlike the GPU backends, which compile
or build their kernels at program start, the RKNN backend loads a **pre-compiled
`.rknn` file** produced by `rknn-toolkit2` on an x86_64 host.

The design follows the Context/Handle pattern:
- **`RKNNComputeContext`** (process-wide): holds the master `rknn_context` (weights)
  plus cached I/O tensor attrs.  Analogous to `ICudaEngine` in TensorRT.
- **`RKNNComputeHandle`** (per server thread): holds a `rknn_dup_context`'d
  context — its own inference state (input/output buffers, scheduler).  Weights
  are shared across dups by the runtime, so the cost is per-thread scratch
  only.  Analogous to `IExecutionContext` in TensorRT.

Why dup and not one shared context?  `rknn_inputs_set` / `rknn_run` /
`rknn_outputs_get` mutate per-inference state inside the context and are not
thread-safe.  `rknn_dup_context` is Rockchip's documented primitive for
concurrent inference across threads.

**Multi-core NPU distribution**

The physical NPU exposes 1–3 cores depending on the SoC (auto-detected from
`/proc/device-tree/compatible`):

| SoC | NPU cores | Peak | Typical thread mask |
|---|---:|---|---|
| RK3562 / RK3566 / RK3568 | 1 | ~0.8–1 TOPS | `AUTO` |
| RK3576 | 2 | ~6 TOPS (INT8) | `CORE_0`, `CORE_1` |
| RK3588 / RK3588s | 3 | ~6 TOPS (INT8) | `CORE_0`, `CORE_1`, `CORE_2` |

`pick_core_mask(thread_index, num_cores)` round-robins handles across cores;
if you set `--nn-server-threads 2 --nn-device-ids 0,0` on an RK3576, thread 0
pins to core 0 and thread 1 pins to core 1 via `rknn_set_core_mask`.  The
`gpu_id` parameter is always 0 on NPU systems (single logical NPU device).

**Precision and quantisation**

The `.rknn` file can be compiled in several modes (chosen at conversion time):
- **fp16** (default, no quantisation needed): `do_quantization=False`.  Simple,
  no calibration data required.  Achieves ~19% of NPU peak on 9×9/128f ResNet.
- **w8a16** (weights int8, activations int16): modest speedup with good
  accuracy.  Requires a calibration dataset.
- **int8 (w8a8)**: maximum throughput, ~2× fp16.  Sensitive logit heads
  (policy, value) can degrade; use **hybrid quantisation** to keep those
  in fp16 via `rknn.hybrid_quantization_step1/step2`.  See
  [ONNX → RKNN conversion](#onnx--rknn-conversion) below.

**Resolving the .rknn file from the ONNX path**

The RKNN backend still uses `LoadedModel::load()` on the `.onnx` to get
board size / channel count / model type (the ONNX is the single source of
truth for architecture metadata).  It then derives the `.rknn` path by
swapping the extension: `models/best.onnx` → `models/best.rknn`.  Both files
must sit side-by-side.

#### ONNX → RKNN conversion

Run on an **x86_64 Ubuntu 22.04 host** (conversion is not supported on
aarch64 — the board only runs models, doesn't compile them).

**Step 1: install rknn-toolkit2 (one-time setup).**  See [Linux (aarch64)
prerequisites](#linux-aarch64--rockchip-npu-board-eg-armsom-sige5--orange-pi-5)
above for the pip install.

**Step 2: convert.**  Save as `convert.py`:

```python
import sys
from rknn.api import RKNN

onnx_path, rknn_path = sys.argv[1], sys.argv[2]
TARGET = "rk3576"   # or rk3588 / rk3568 / rk3566 / rk3562

rknn = RKNN(verbose=True)
rknn.config(
    target_platform=TARGET,
    mean_values=None, std_values=None,    # identity: features are already normalised
    disable_rules=['unsqueeze_to_4d_reshape_with_elementwise_op'],
)

# Fixed input shape — the ONNX has a dynamic batch dim that RKNN rejects.
# Use a batched shape (e.g. [4, 17, 9, 9]) for better NPU utilisation on
# self-play workloads; use [1, 17, 9, 9] for live play / single-move latency.
assert rknn.load_onnx(
    model=onnx_path,
    inputs=["state"],
    input_size_list=[[1, 17, 9, 9]],
) == 0
assert rknn.build(do_quantization=False) == 0     # fp16 weights, no calibration
assert rknn.export_rknn(rknn_path) == 0
rknn.release()
```

```bash
python convert.py best.onnx best.rknn
scp best.rknn armsom:/path/next/to/best.onnx
```

**Step 3 (optional): int8 with hybrid quantisation.**  The Go logit heads
(policy, value, score) don't survive full int8 well — small numeric errors
in logits become big probability shifts after softmax.  The fix is to int8
the trunk and keep the heads fp16:

```python
# Calibration set: dump 100–500 selfplay encodings to .npy files,
# one `(17, 9, 9)` float32 per file.  calib.txt lists their paths.
rknn.hybrid_quantization_step1(
    dataset="calib.txt",
    proposal=True, proposal_dataset_size=16,
)
# → writes v0000.quantization.cfg / v0000.model / v0000.data
```

Edit `v0000.quantization.cfg` → `custom_quantize_layers:`:
```yaml
custom_quantize_layers:
  /policy_conv/Conv_output_0:          float16
  /policy_fc/Gemm_output_0:            float16
  /value_head/fc2/Gemm_output_0:       float16
  /score_mean_head/fc2/Gemm_output_0:  float16
  /score_stdev_head/fc2/Gemm_output_0: float16
  /ownership_conv/Conv_output_0:       float16
```
(Use the exact node names the toolkit emitted in your cfg — they reflect
RKNN's rewritten graph.)

```python
rknn.hybrid_quantization_step2(
    model_input="v0000.model",
    data_input="v0000.data",
    model_quantization_cfg="v0000.quantization.cfg",
)
rknn.export_rknn("v0000_hybrid.rknn")
```

Heads are <1% of FLOPs, so keeping them fp16 costs almost nothing;
expected throughput ≈ full int8, expected MCTS strength ≈ fp16.

### VIP9000 NPU Backend (VeriSilicon, aarch64 Linux — Allwinner A733)

`VIP9000ComputeHandle` (`src/vip9000_compute.cpp`) runs inference on the
VeriSilicon Vivante VIP9000 NanoDI+ NPU embedded in the Allwinner A733
SoC, via VIPLite v2.0 (`libNBGlinker.so` + `libVIPhal.so`).  Like RKNN,
the runtime consumes a **pre-compiled `.nb` (Network Binary Graph)**
produced by VeriSilicon's Acuity Toolkit (specifically v6.30.22 inside
Allwinner's `ubuntu-npu:v2.0.10.1` Docker image — the pip `acuitylite`
wheel is verified non-functional for the A733 PID, see
[A733_CONVERSION.md](A733_CONVERSION.md)) on an x86_64 host.

The design follows the Context/Handle pattern:
- **`VIP9000ComputeContext`** (process-wide): refcounts `vip_init` /
  `vip_destroy`, queries the hardware chip ID once
  (`vip_query_hardware(VIP_QUERY_HW_PROP_CID)`), holds a master
  `vip_network` created lazily on the first handle.  The master exists
  primarily so the chip-ID validation has somewhere to cache the I/O
  metadata (see "NBG header pre-flight" below); it is **not** dup'd.
- **`VIP9000ComputeHandle`** (per server thread): creates its own
  `vip_network` from the cached NBG bytes (`vip_create_network(...,
  VIP_CREATE_NETWORK_FROM_MEMORY)`), allocates per-input/per-output
  `vip_buffer`s via `vip_create_buffer`, then prepares + binds them
  in this exact order:
  ```
  vip_create_network                (per-thread, from cached NBG bytes)
  vip_create_buffer × n_inputs      (input buffer per input tensor)
  vip_create_buffer × n_outputs     (output buffer per output tensor)
  vip_prepare_network               (allocates command-buffer + memory pool)
  vip_set_input  × n_inputs         (must come AFTER prepare — order matters)
  vip_set_output × n_outputs
  ```
  Then per inference: map → fp32→fp16 convert + zero-pad → `vip_flush_buffer(FLUSH)`
  → `vip_run_network` → `vip_flush_buffer(INVALIDATE)` → fp16→fp32 → unmap.

Why one network per thread instead of `vip_dup_network(VIP_DUP_FOR_CMD_BY_NETWORK)`?
The dup primitive shares weight memory across handles — useful on memory-tight
embedded devices.  But it requires the master to be `vip_prepare_network`'d
first (which means the master also needs its own input/output buffers
attached), and on the A733 the duplicated weight memory is ~6 MB per
thread — a rounding error against the hundreds of MB of CMA the kernel
already reserves for NPU activations.  Per-thread `vip_create_network`
is simpler and keeps lifecycles independent.

**Single-core hardware, single-thread server**

The A733 NPU has one VIP9000 NanoDI+ core.  The VIPLite kernel driver
serializes hardware command submission internally, so multiple server
threads don't run in parallel — they queue on the same core, each
paying full inference latency.  Measured on a Cubie A7A with kata1-b10c128
at 9×9 (via `build/vip9000_smoke`), fp16 vs int8 (revised calibration):

| precision | NBG  | batch | server threads | ms/call          | states/s         | notes                                |
|-----------|------|------:|---------------:|-----------------:|-----------------:|--------------------------------------|
| fp16      | bs=1 |     1 |              1 |  80.78           | 12               | single stream, fp16-native           |
| fp16      | bs=4 |     4 |              1 | 338.68           | 11               | bs=4 doesn't amortize on fp16 path   |
| fp16      | bs=1 |     1 |              2 | 159.97 (each)    | 11 (aggregate)   | 2 threads — no aggregate gain        |
| **int8**  | bs=1 |     1 |              1 |   **0.82**       | **1213**         | 491 µs hardware + ~330 µs host       |
| **int8**  | bs=4 |     4 |              1 |   **3.43**       | **1167**         | bs=4 amortizes cleanly on int8 path  |
| **int8**  | bs=1 |     1 |              2 |   **1.49**       | **1284** (agg)   | int8 is fast enough that 2t helps    |

INT8 is **~110× faster** than fp16 on this hardware/model. This is much
larger than the typical ~3× int8/fp16 ratio because kata1-b10c128's
small spatials (9×9) and skinny channels (128) badly under-utilise
VIP9000's fp16 tiler (it's optimised for ≥56×56 inputs); the int8 path
hits a much better-tuned tile config on the MAC array.  For calibration:
ResNet-50 INT8 on this NPU clocks ~8 ms/call (~1 TOPS achieved) — at
similar absolute compute the same hardware delivers very different
numbers depending on how well the workload maps to the tiler.

Practical implications on the A733:

* **Use INT8** — `tools/onnx_to_a733_docker.sh int8 1` /
  `... int8 4`.  Backend auto-prefers `models/<base>.a733.bs<K>.int8/`
  over `.fp16/` when both are present.
* **`--nn-server-threads` ≤ 2** — on the int8 path 2 threads gives a
  ~17 % aggregate gain (1610 vs 1382 states/s); 3+ threads hits driver
  serialisation and doesn't help.  On the fp16 path, stay at 1.
* **bs=4 int8 amortises** — per-state throughput holds (1288 vs 1382
  states/s) while per-call latency is 4× higher, exactly matching the
  batch dimension.  On fp16 it doesn't, so bs=1 was better there.

**Numerical quality of the int8 NBGs** (200 random mid-game 9×9 positions,
fp16 NBG treated as ground truth — `build/vip9000_accuracy --positions 200`,
revised-calibration NBGs from `kata1-b10c128.a733.int8.revised.zip`):

| precision | top-1 | top-3 | top-5 | value MAE | score MAE | score_sd MAE | policy logits MAE | own MAE |
|-----------|------:|------:|------:|----------:|----------:|-------------:|------------------:|--------:|
| **int8 bs=1** |  7.5 % | 28.0 % | 34.0 % | 0.0337 | 1.03 pts | 0.80 | 4.17  | 0.0148 |
| **int8 bs=4** | **79.0 %** | **91.5 %** | **100.0 %** | 0.0412 | 1.00 pts | 0.78 | 0.38  | 0.0163 |

* **bs=4 int8 is production-quality** — top-1 79 % with 100 % top-5
  means MCTS visits the same candidate set as fp16 every time; value
  MAE 0.04 (range [−1, 1]) and score MAE 1.0 point are negligible
  strength regressions.  Safe for play / self-play / gating
  evaluations.
* **bs=1 int8 has a strong scalar-head, weak policy** — value /
  score / ownership are all close to fp16 (score MAE 1.0 pt, value
  MAE 0.03), but policy top-1 is only 7.5 %; policy logits MAE 4.17
  is much higher than bs=4's 0.38.  MCTS strength would suffer because
  the policy prior is the worst quantised head here.  Acceptable for
  pure value-driven workloads; not recommended for play.
* **Quality history** — the original (pre-revision) int8 NBGs had
  bs=1 top-1 = 3 % / score MAE 27 pts and bs=4 top-1 = 22.5 % / score
  MAE 4.5 pts.  The revision narrowed the per-tensor calibration ranges
  via more representative fixtures; bs=4 jumped from "MCTS-only safe"
  to "drop-in for fp16."  See A733_CONVERSION.md §4 for the
  calibration data flow.

**Precision and quantisation**

The `.nb` baked precision is whatever you compiled with.
`tools/onnx_to_a733_docker.sh` drives Acuity v6.30.22's `pegasus.py`
with `dtype="float"` (`OvxlibExporter`), which produces fp16 buffers
throughout (the VIP9000 fp pipeline is fp16-native — full MAC rate, no
fp32-fallback slowdown). The on-board runtime exposes input/output
tensors as `VIP_BUFFER_FORMAT_FP16` and the C++ backend round-trips fp32
↔ fp16 on host with vendored conversion routines (vendored from
VeriSilicon's `vpm_run.c` to avoid linking the SDK's helper libs).

INT8 quantisation is supported by the toolkit (`Quantization(model).quantize(
'uint8',...)` with calibration data) and would land at roughly **3× fp16
throughput** on this NPU; not currently used because kata1 is small
enough that fp16 is already lossless.  The C++ backend handles either
format transparently — it switches on `data_format` at every map/unmap.

**Resolving the .nb file from the ONNX path**

The VIP9000 backend uses `LoadedModel::load()` on the `.onnx` to get
board size / channel count / model type (the ONNX is the architecture
truth, same as RKNN).  It then derives the NBG path:

| ONNX path                                      | Resolved NBG (with `--max-batch K`)                           |
|------------------------------------------------|---------------------------------------------------------------|
| `models/foo.onnx`                              | `models/foo.a733.bs<K>.{int8,fp16}/network_binary.nb`         |
| `models/foo.unshared.onnx`                     | `models/foo.a733.bs<K>.{int8,fp16}/network_binary.nb`         |
| `models/foo.a733.bs1.unshared.onnx`            | `models/foo.a733.bs<K>.{int8,fp16}/network_binary.nb`         |
| `models/foo.a733.bs1.int8.unshared.onnx`       | `models/foo.a733.bs<K>.{int8,fp16}/network_binary.nb`         |
| `models/foo.a733.bs1.int8/network_binary.nb`   | itself (explicit override)                                    |
| `models/foo.a733.bs1.int8/`                    | `<dir>/network_binary.nb`                                     |

`K` is picked from `--max-batch`: 1 if `max-batch ≤ 1`, else 4.  Larger
batches throw at create time — re-convert with the desired bs.  Within
a `K`, the resolver tries `.int8/` first and falls back to `.fp16/`;
override with `VIP9000_FORCE_PRECISION=fp16` (or `int8`) to pin one.

**NBG header pre-flight**

Before calling `vip_create_network`, the backend reads the first 12
bytes of the `.nb` and validates:
- magic == `VPMN`
- target chip ID matches `vip_query_hardware(VIP_QUERY_HW_PROP_CID)`
  (low byte must match — full PID match preferred)

If the chip ID mismatches, the backend throws with a concrete
remediation message *before* the VIPLite runtime would otherwise fail
with a generic `status=-4` and the NN server thread would die — which
in turn would hang `evaluate_*()` callers forever (the NNEvaluator's
queue waits for a server that no longer exists).  This is a real
failure mode: the very first batch of NBGs delivered for kata1-b10c128
targeted chip `0x15` instead of the A733's `0x1000003B`, because the
acuitylite version on the conversion host didn't recognize
`VIP9000NANODI_PID0X1000003B` and silently fell back to a generic
VIP9000 default (low byte `0x15`).  See
[A733_CONVERSION.md](A733_CONVERSION.md) for the full diagnosis and
the verification checklist when re-converting.

#### ONNX → VIP9000 NBG conversion

Run on a **real x86_64 Linux host with Docker** (not aarch64 — Acuity
ships only x86_64 binaries; not a nested container — see
A733_CONVERSION.md §6.3 for why Docker-in-Docker fails the
simulator's `vsi_nn_CreateGraph()` call).

**Step 1: install Docker + Allwinner's `ubuntu-npu:v2.0.10.1` image
(one-time setup).**  The image is on Allwinner's Synology netdisk
(not Docker Hub) — A733_CONVERSION.md §3.0/§3.2 has the retrieval
recipe.  The pip `acuitylite` wheel is verified non-functional for
this NPU (chip table missing PID `0x1000003B` in 6.42–6.51) and
`tools/onnx_to_a733.py` is now a deprecation banner that exits 2.

**Step 2: convert.**  `tools/onnx_to_a733_docker.sh` runs the full
ONNX export → unshare-initializers → Acuity import → NBG export
chain inside the Docker image.  Setting `VSIMULATOR_CONFIG=VIP9000NANODI_PID0X1000003B`
(no `_PLUS_`, despite what `pegasus_setup.sh v3` claims — the actual
shipped config file is named without `_PLUS_`):

```bash
# Both batches in one go.  Reads kata1-b10c128-*.txt.gz and writes
# models/kata1-b10c128.a733.bs{1,4}.fp16/network_binary.nb.
bash tools/onnx_to_a733_docker.sh 1
bash tools/onnx_to_a733_docker.sh 4
```

**Step 3: verify before shipping.**  `xxd <nbg> | head -1` must show:
- bytes 0..3: `5650 4d4e` (`VPMN` magic)
- bytes 8..11: `3b00 0010` (target chip `0x1000003B`)

Bytes 4..7 are the NBG format version (`00 00 02 00` = `0x20000` from
v6.30.22, but v1 versions like `0x1001E` / `0x10020` also work — the
runtime accepts both as long as the target byte is right).  Then
sanity-check on the device with `build/vip9000_smoke`.  Full procedure
with all gotchas — including the chip-ID mismatch failure mode and how
to distinguish a real backend bug from a converter bug — in
[A733_CONVERSION.md](A733_CONVERSION.md) §7.

See also [A733_CONVERSION.md](A733_CONVERSION.md) for the detailed
rationale (why fp16, why Docker-only, why un-share initializers, why
the `_PLUS_` config name doesn't exist in v6.30.22, host-side parity
numbers).

### Modular Backend Design (KataGo pattern)

The architecture has three layers:

1. **`LoadedModel`** (`include/loaded_model.h`) — parses ONNX once, holds
   pre-fused BN weights in CPU memory.  Shared (const) across all threads.

2. **`ComputeContext`** (`include/compute_context.h`) — per-device GPU state.
   For TensorRT: one `ICudaEngine` + `cudaStream` per unique GPU (engine built lazily, cached to disk).
   For CUDA: one `cudaStream` per unique GPU.
   For OpenCL: one `cl_context` + `cl_queue` + `cl_program` per unique GPU
   (avoids NVIDIA serialization).  Created on the main thread.

3. **`ComputeHandle`** — per-server-thread GPU state.  Created ON the server
   thread.  Uploads weights from `LoadedModel` and owns workspace buffers.
   Implements `predict_batch()`.

Adding a new backend: implement `ComputeContext` + `ComputeHandle`,
add to the factory in `compute_context.cpp`, add CMake detection.  The
NNEvaluator, MCTS, game engine, and training pipeline are completely
backend-agnostic.

### Backend Lifecycle (example: 2 GPUs, 4 server threads, `--nn-device-ids 0,0,1,1`)

| | **TensorRT** | **CUDA** | **OpenCL** | **Metal** |
|---|---|---|---|---|
| **Context created** | Main thread | Main thread | Main thread | Main thread |
| **Context holds** | Runtime per GPU (no shared stream) | Device ID + precision per GPU (no shared stream) | Context + queue + **compiled kernels** per GPU | MTLDevice + command queue |
| **Handle created** | Server thread | Server thread | Server thread | Server thread |
| **Handles** | 4 (1 per thread) | 4 (1 per thread) | 4 (1 per thread) | 4 (1 per thread) |
| **CUDA stream** | Per-thread via `cudaStreamPerThread` | Per-thread via `cudaStreamPerThread` | N/A (OpenCL queue) | N/A (Metal queue) |
| **Engine/kernel build** | 1st server thread per GPU, mutex-guarded, cached to disk | N/A (hand-written kernels) | Main thread (1 `clBuildProgram` per GPU) | Server thread (1 MPSGraph per handle) |
| **Weight copies on GPU** | 1 per GPU (inside TRT engine, shared by handles) | 2 per GPU (each handle uploads own FP16 copy) | 2 per GPU (each handle uploads own copy) | 4 total (embedded in MPSGraph) |
| **Per-handle state** | Execution context + I/O buffers | Weights + workspace + CUDA graph cache | Kernel handles + weights + workspace | MPSGraph with embedded weights |

#### Per-thread CUDA streams (KataGo pattern)

Both the **TensorRT** and **CUDA** backends use `cudaStreamPerThread` — CUDA's
built-in per-thread implicit stream.  Each server thread automatically gets its
own independent CUDA stream with no explicit creation or destruction.

This means two server threads on the same GPU (e.g. `--nn-device-ids 0,0`)
submit inference work to **separate streams**.  The GPU hardware scheduler
interleaves their kernels with zero host-side contention — no shared stream,
no mutex around `predict_batch`.

Why this matters:
- **TensorRT**: `enqueueV3()` configures internal workspace and launches
  kernels.  With a shared stream, two concurrent `enqueueV3` calls from
  different execution contexts could race on TRT's host-side memory pool
  management, corrupting heap metadata (the root cause of the sporadic
  `malloc_consolidate` crash at process exit on multi-GPU systems).
- **CUDA**: `cudaStreamBeginCapture()` puts a stream into capture mode.
  If two threads shared a stream, one thread's kernel launches during
  another thread's capture would be pulled into the wrong CUDA graph.

With per-thread streams, each thread's capture, launch, and sync are
completely isolated.  No shared mutable state during inference.

#### Per-engine lock for context lifecycle (TensorRT)

Per-thread streams make **inference** lock-free, but `IExecutionContext`
**lifecycle** operations on a shared engine still need serialization.
The TRT headers and guide give **no thread-safety guarantee** for
`ICudaEngine::createExecutionContext()` / `~IExecutionContext`
(`NvInferRuntime.h` documents thread-safety requirements only for the
logger/allocator callbacks), and the engine tracks its live contexts
internally.

Empirically: with `--nn-device-ids 0,0,1,1`, two server threads share
one engine per GPU.  At process exit, `~NNEvaluator` wakes all server
threads at once; each called `delete exec_ctx` on contexts belonging to
the same engine concurrently, and the corruption surfaced later when
`~ICudaEngine` ran — `double free or corruption (out)` after
`Done! N games`.  Serializing the lifecycle fixed it.

Fix: one `engine_mutex` per `TRTDeviceState` held around (a) engine
build, (b) `createExecutionContext()`, and (c) `delete exec_ctx`.
Inference (`enqueueV3`, `setTensorAddress`, `setInputShape`) stays
unlocked — each thread still owns its own `exec_ctx` and stream, so
the GPU scheduler interleaves kernels across threads as before.

Note: upstream **KataGo does not need this lock** — its `trtbackend.cpp`
builds a **separate engine per server thread** (weights duplicated on
GPU per thread), so no `ICudaEngine` is ever shared; its only trtbackend
mutex is inside `TRTErrorRecorder`, which the TRT API requires to be
thread-safe.  MiniGo instead shares one engine per device (one weight
copy, per-thread exec contexts — a pattern TRT explicitly supports for
inference) and pays one mutex on the rare lifecycle path for it.

The **OpenCL** backend follows the analogous rule: the
`cl_command_queue` (the OpenCL analogue of a CUDA stream) is created
**per handle**, not per device — only the immutable `cl_context` +
compiled `cl_program` live in shared device state.

The **Metal** backend shares one `MTLCommandQueue` per device.  Apple
explicitly guarantees thread safety for Metal command queue submission,
so no additional synchronization is needed.

Each handle destructor calls `cudaStreamSynchronize(cudaStreamPerThread)`
before freeing device buffers, ensuring any in-flight async work from a
prior `predict_batch` (that may have thrown before reaching its own sync)
is drained before `cudaFree`.

#### Engine Build Serialization

TensorRT engine build is serialized per cache path (mutex) to prevent concurrent
writes to the same cache file.  With identical GPUs, only **1 build** occurs across
all 4 threads — the remaining 3 load from cache or reuse `dev.engine` in memory.

The cache filename includes the TensorRT version (`trt10.8.0_...`) so upgrading
TRT automatically invalidates stale cached engines.

### Handle Initialization Timeline

**Selfplay** (1 model, 2 GPUs, 4 server threads, `--nn-device-ids 0,0,1,1`):

```
Main thread:  LoadedModel::load()  →  create_compute_context({0,0,1,1})
              │                        │→ DeviceState[GPU0]: runtime (no stream)
              │                        │→ DeviceState[GPU1]: runtime (no stream)
              └→ NNEvaluator(model, ctx, {0,0,1,1}) → spawns 4 threads, returns

Thread 0 (GPU0): ──lock engine_mutex──→ deserialize engine ──→ unlock ──→ lock ──→ ExecCtx #0 ──→ unlock
Thread 1 (GPU0): ──lock engine_mutex── WAIT ─────────────────→ engine exists ─────→ ExecCtx #1 ──→ unlock
Thread 2 (GPU1): ──lock engine_mutex──→ deserialize engine ──→ unlock ──→ lock ──→ ExecCtx #2 ──→ unlock
Thread 3 (GPU1): ──lock engine_mutex── WAIT ─────────────────→ engine exists ─────→ ExecCtx #3 ──→ unlock
                  ↑ parallel (different GPUs)     ↑ serialized (same GPU) — build AND context lifecycle

Runtime inference (after all handles are ready):
Thread 0: predict_batch on cudaStreamPerThread[0]  ← independent stream
Thread 1: predict_batch on cudaStreamPerThread[1]  ← independent stream
Thread 2: predict_batch on cudaStreamPerThread[2]  ← independent stream
Thread 3: predict_batch on cudaStreamPerThread[3]  ← independent stream
           ↑ fully parallel, no shared state during inference
```

Threads on different GPUs run in parallel.  Threads on the same GPU are serialized
by `dev.engine_mutex` during three brief points: (1) engine build (first thread
deserializes, others wait then skip), (2) `createExecutionContext()` at handle
construction, and (3) `delete exec_ctx` at handle destruction.  After
construction all threads run inference fully in parallel — `enqueueV3` is lock-free
because each thread owns its own `IExecutionContext` and `cudaStreamPerThread`.

**Evaluation** (2 models, 2 GPUs, 4 server threads each):

Each model gets a **separate** `ComputeContext` with its own `DeviceState` per GPU.
Without serialization, both models' server threads would call `deserializeCudaEngine()`
concurrently on the same GPU through different `IRuntime` objects — causing a CUDA
driver-level race (SIGSEGV ~70% of the time).

Fix: `eval1->wait_ready()` blocks until all of model 1's handles are created before
model 2's `NNEvaluator` is constructed.  This adds ~1-2s to eval startup but
eliminates the race.  Runtime inference is fully parallel (both models' server
threads use independent per-thread streams).

### Context vs Handle vs Stream

Three levels of resource ownership, from long-lived shared infrastructure
down to per-thread mutable state:

**Context** — one per unique GPU, lives for the lifetime of the process.
Created on the main thread.  Holds resources that are expensive to create
once and immutable (or read-only) during inference:

| Backend | Context holds |
|---|---|
| **TensorRT** | `IRuntime*`, `ICudaEngine*` (built/deserialized once, immutable) |
| **CUDA** | Device ID, precision flag (FP16 or FP32) |
| **OpenCL** | `cl_context`, `cl_program` (compiled kernels) |
| **Metal** | `MTLDevice`, `MTLCommandQueue` |

**Handle** — one per server thread, created ON that thread.  Holds mutable
per-inference state that must not be shared between threads:

| Backend | Handle holds |
|---|---|
| **TensorRT** | `IExecutionContext*`, device I/O buffers (`d_input`, `d_policy`, ...) |
| **CUDA** | Weight copies (FP16), workspace buffers, CUDA graph cache |
| **OpenCL** | `cl_kernel` objects, weight buffers, workspace buffers |
| **Metal** | `MPSGraph` with baked-in weights and tensors |

**Stream** — one per server thread (via `cudaStreamPerThread` for TRT/CUDA).
An ordered queue of GPU operations.  The server thread submits memcpy →
inference → readback to its stream, then syncs.  Streams on the same GPU
can run in parallel — the GPU hardware scheduler interleaves their kernels.

```
Process
├── ComputeContext (GPU 0)
│   ├── TRT engine (shared, immutable, weights baked in)
│   ├── Handle #0 (thread 0)  ← exec_ctx + buffers + cudaStreamPerThread[0]
│   └── Handle #1 (thread 1)  ← exec_ctx + buffers + cudaStreamPerThread[1]
│
└── ComputeContext (GPU 1)
    ├── TRT engine (shared, immutable, weights baked in)
    ├── Handle #2 (thread 2)  ← exec_ctx + buffers + cudaStreamPerThread[2]
    └── Handle #3 (thread 3)  ← exec_ctx + buffers + cudaStreamPerThread[3]
```

The context is the shared read-only infrastructure.  The handle is the
per-thread mutable workspace.  The stream is the per-thread GPU command
queue.  Nothing is shared between threads during inference — handles and
streams are fully independent.

### Where Weights Live

Weights flow from CPU → GPU differently in each backend:

```
                    TensorRT          CUDA/OpenCL        Metal           Eigen
                    ─────────         ───────────        ─────           ─────
LoadedModel (CPU)   [weights]         [weights]          [weights]       [weights]
                       │                 │ │                │ │             │
                       ▼                 │ │                │ │             │
Context (GPU)       ICudaEngine          │ │                │ │             │
                    [weights ×1]         │ │                │ │             │
                       │                 │ │                │ │             │
              ┌────────┤                 │ │                │ │             │
              ▼        ▼                 ▼ ▼                ▼ ▼             ▼
Handle 0    ExecCtx  ExecCtx         [wt copy] [wt copy]  MPSGraph MPSGraph  (ref)
Handle 1    + bufs   + bufs          + bufs    + bufs     [weights] [weights] (ref)
```

| Backend | Weights on GPU | Sharing | Memory per GPU (128f/10b, 4 threads) |
|---|---|---|---|
| **TensorRT** | 1 copy per GPU (baked into `ICudaEngine`) | Shared — engine is immutable, handles get own `IExecutionContext` + I/O buffers | **~6 MB** |
| **CUDA+CUTLASS** | 1 copy per handle (FP16 upload) | None — each handle owns its weight buffers | ~24 MB |
| **OpenCL** | 1 copy per handle (`cl_mem` upload) | None — each handle creates own buffers | ~24 MB |
| **Metal** | 1 copy per handle (embedded in `MPSGraph`) | None — weights are graph constants | ~24 MB |
| **RKNN** | 1 copy in NPU DMA memory (from master `rknn_context`) | Shared — `rknn_dup_context` shares weights across duplicates, per-thread state only | **~6 MB** |
| **Eigen** | CPU only (in `LoadedModel`) | Shared by pointer — no GPU copies | 0 |

TensorRT is the most memory-efficient because the compiled engine separates
immutable weights (shared) from mutable execution state (per-thread).  For
the current 128f/10b model (~6MB weights), 4 server threads on 1 GPU use
24MB with CUDA vs 6MB with TensorRT.  For larger models this gap grows
proportionally.


## Performance

### Batch NN inference throughput (9×9, states/s)

**Small model** (64 filters, 5 blocks):

| Batch | TensorRT FP16 (RTX 2080 Ti) | CUDA+CUTLASS FP16 (RTX 2080 Ti) | OpenCL FP32 (RTX 2080 Ti) | Metal FP16 (M1 Max) |
|------:|----------------------------:|-------------------------------:|---------------------------:|--------------------:|
| 1     | **3,609**                   | 2,620                          | 934                        | 750                 |
| 8     | **26,039**                  | 20,986                         | 7,336                      | 7,500               |
| 32    | **85,254**                  | 45,083                         | 19,886                     | 26,000              |
| 64    | **136,814**                 | 55,482                         | 27,383                     | 28,000              |
| 128   | **175,102**                 | 66,495                         | 34,005                     | 44,000              |

**Large model** (128 filters, 10 blocks):

| Batch | TensorRT FP16 (RTX 2080 Ti) | CUDA FP16+WMMA (RTX 2080 Ti) | OpenCL FP32 (RTX 2080 Ti) | RKNN fp16 bs=1 model (RK3576, 1 core) | RKNN fp16 bs=4 model (RK3576, 1 core) | ORT CPU 4×A72 (RK3576) |
|------:|----------------------------:|-----------------------------:|---------------------------:|--------------------------------------:|--------------------------------------:|-----------------------:|
| 1     | **1,107**                   | 354                          | 324                        | 334                                   | 133¹                                  | 24                     |
| 8     | —                           | —                            | —                          | 348                                   | 521                                   | 27                     |
| 32    | **31,965**                  | 8,145                        | 4,723                      | 343                                   | 543                                   | 27                     |
| 64    | **56,726**                  | 9,347                        | 5,712                      | 356                                   | 544                                   | 27                     |
| 128   | **82,781**                  | 10,821                       | 6,012                      | 344                                   | 528                                   | 27                     |

¹ The bs=4 compiled RKNN model pads single-sample requests to batch=4 — three
pad slots are wasted, so bs=1 throughput is worse than the bs=1 compiled model.

TensorRT is **2.6×** faster than CUDA+CUTLASS at batch-128 (small model) due
to whole-graph layer fusion.  CUDA+CUTLASS is **2.0×** faster than OpenCL FP32.
At single inference, CUDA+CUTLASS closes to within **1.4×** of TensorRT
(2,620 vs 3,609) thanks to CUTLASS's optimized software pipelining.

**NPU notes:** at 9×9 / 128 filters / fp16, the RK3576 NPU tops out at ~543
states/s per core (≈ 279 GFLOPs effective, ~19% of the ~3 TFLOPs fp16 peak).
Single-sample latency is 2.66 ms on the bs=1 model and ~1.84 ms per-sample
on the bs=4 model.  Doubling to 2 cores gives ~1,100 states/s total.  The
NPU is 13–14× faster than the board's A72 CPU running the same ONNX through
ONNX Runtime, and ~150× slower than a desktop RTX 2080 Ti running TensorRT.
Quantising the trunk to int8 (with fp16 heads, via hybrid quantisation) is
expected to roughly double these numbers (~35% of peak).

### How batching works (GPU backends vs RKNN NPU)

The `NNEvaluator` queue, `--max-batch`, and `--search-threads` flags work the
same on every backend — but what happens **inside `predict_batch()`** differs
fundamentally between the GPU backends and RKNN.

**GPU backends (TensorRT, CUDA, OpenCL, Metal) — dynamic batch.**  The ONNX
is compiled (or kernels launched) with the batch dimension left as a free
variable.  Each `predict_batch()` call passes the runtime batch size as a
parameter:

- TensorRT: builds the engine with an optimisation profile covering
  `[MIN=1, OPT=max_batch/2, MAX=max_batch]`; `setInputShape(N)` before
  `enqueueV3` selects the shape for this call.
- CUDA/OpenCL: hand-written kernels take `N` as a kernel argument; GEMM
  tile counts scale with `N`.
- Metal: MPSGraph rebuilds the graph lazily for each new batch size it
  sees (cached after first use).

Effect: any batch size `1 ≤ N ≤ max_batch` runs in a single kernel launch,
and the per-sample cost drops as `N` grows (batching amortises launch
overhead and fills MAC arrays).  `max_batch` is a soft ceiling — setting
it higher just means the GPU can absorb bigger bursts.

**RKNN — static batch baked into the `.rknn` file.**  The `.rknn` is
compiled offline with *one specific batch shape* (the `input_size_list`
argument of `rknn.load_onnx`).  At runtime the NPU accepts only that exact
shape — no dynamic `N`.

**Contract with the server loop (same mental model as every other backend):**

- Each `predict_batch(states)` call emits **exactly one `rknn_run`**.
- The caller must keep `states.size() <= K` (i.e. set `--max-batch <= K`
  on the CLI).  The backend asserts this at runtime and throws with a
  hint if violated — no silent truncation, no internal chunking loop.
- If `states.size() < K`, the backend zero-pads slots `[N..K)` and
  discards their outputs.

```text
K = 4, --max-batch = 4, drain = 3:
    [s0 s1 s2 0] → rknn_run  (3 real + 1 pad, pad output discarded)

K = 4, --max-batch = 4, drain = 4:
    [s0 s1 s2 s3] → rknn_run  (fully packed)

K = 4, --max-batch = 8  →  RUNTIME ERROR (bump --max-batch down to 4).
```

The `model_batch` value is read from the rknn input-attr at handle init
(`dev.model_batch = input_attrs[0].dims[0]`) and logged on startup, so
picking `--max-batch` is just: "check the log, set the flag ≤ that."

**Consequences — completely different tuning rules:**

| Aspect | GPU backends | RKNN NPU |
|---|---|---|
| Batch size at runtime | Anything `1..max_batch` | Exactly `K` (pad if fewer) |
| Meaning of `--max-batch` | Max N per kernel; any value ≤ engine max is fine | Max N per kernel **and** must be ≤ compiled `K` |
| Setting `--max-batch > model_max` | soft cap — effectively ignored above engine max | **hard error** at runtime |
| Batch-1 live play | Fast (kernel specialises for N=1) | **Slow if compiled with K > 1** (you pay for K samples per move) |
| Self-play throughput | Scales sub-linearly with burst size | Scales with `K` *if the queue consistently fills K slots*; otherwise padding eats the win |
| Retuning | Change one flag | **Recompile the `.rknn`** on the x86 host |

**Practical rules on the NPU:**

1. **Set `--max-batch == K`** (the compiled `model_batch` shown in the
   startup log).  Nothing else makes sense.

2. **Pick `K` for the *dominant workload*:**
   - **Live play / `play` / `evaluate`:** compile `[[1, C, H, W]]` —
     single-move latency matters, nothing to batch.
   - **Self-play with 1 worker:** compile `[[1, C, H, W]]` — MCTS
     virtual-loss rarely generates enough concurrent leaves to keep a
     bs=4 model's slots full, so padding eats the win (observed on
     v0000: the bs=4 model was slightly *slower* than bs=1 with one
     worker).
   - **Self-play with multiple workers or heavy `--search-threads`:**
     compile `[[4, C, H, W]]` — queue stays full, padding is rare,
     peak per-core throughput rises ~55 % (334 → 543 states/s on v0000).

3. **Multi-core distribution is orthogonal to `K`.**  On a multi-core NPU,
   set `--nn-server-threads` to the number of NPU cores and
   `--nn-device-ids 0,0,...` (all zeros — there's only one logical NPU).
   Each server thread runs its own dup'd context pinned to one core via
   `rknn_set_core_mask`.  Effective throughput ≈ single-core throughput
   × core count, regardless of `K`.  You can also try `2 × num_cores`
   threads with two contexts per core — the driver serialises same-core
   `rknn_run`s but a second thread can overlap its host-side prep
   (memcpy, NHWC transpose, submission) with the first thread's NPU
   compute.  Typical gain: 5–15 %.

4. **You can ship both.**  Compile two `.rknn` files with different K
   (`best.rknn` for live play, `best_bs4.rknn` for self-play) and swap
   by renaming.  The backend picks up whichever file sits next to the
   `.onnx` and logs the detected `model_batch` on startup.

### Self-play throughput (800 sims/move)

| Config | RTX 5070 Ti (OpenCL) | RTX 2080 Ti (OpenCL) |
|---|---|---|
| 64 games, 8 threads, 32 search-threads | **2.17 s/game** | 3.13 s/game |

**Tuning guide**: increase `--threads` (more concurrent games) and
`--search-threads` (more threads per game's search) until GPU utilization
plateaus.  Batch size adapts naturally to the total concurrent search
threads: `total = min(games, threads) × search_threads`.


## Platform Notes

### macOS (Apple Silicon) — recommended: Metal

- **Metal** (default): uses MPSGraph for GPU inference with FP16 compute.
  The computation graph is compiled once at model load time (~200ms).
  2.8× faster than OpenCL on the same chip.
- **OpenCL**: available and functional; slower than Metal.
- **Eigen**: uses Apple **Accelerate** for hardware-tuned BLAS.

### macOS (Intel)

- Metal not available (requires Apple Silicon).
- OpenCL 1.2 available via Intel HD/Iris GPU.
- Accelerate provides vecLib BLAS for Eigen.

### Linux — recommended: TensorRT or CUDA (NVIDIA)

- **TensorRT** (default when installed): optimized inference via TensorRT's
  ONNX parser + kernel auto-tuning + FP16. Requires CUDA toolkit +
  `libnvinfer-dev` + `libnvonnxparsers-dev`. Auto-detected when libraries
  are present.
- **CUDA** (default without TensorRT): FP16 Tensor Core inference via WMMA.
  Requires CUDA toolkit (nvcc). 1.7-1.8× faster than OpenCL on the same GPU.
- **OpenCL** (fallback): hand-written implicit GEMM kernels.  Available on
  NVIDIA (CUDA toolkit), AMD (ROCm/Mesa), Intel (NEO).
- Metal not available on Linux.
- For faster Eigen: `sudo apt install libopenblas-dev`

