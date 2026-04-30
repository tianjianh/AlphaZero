# MiniGo C++ — Miniature KataGo AlphaZero

Miniature AlphaZero Go engine modelled on KataGo's architecture:
multi-threaded MCTS with per-leaf blocking evaluation + a KataGo-style
`NNEvaluator` server that batches leaf evaluations into one GPU call.
KataGo-style 7-headed neural network with 5 inference heads (policy,
value W/L/D, scoreMean, scoreStdev, ownership) and 2 training-only
auxiliary heads (score belief, opponent policy).
Training runs in Python/PyTorch. Works on **Linux** and **macOS** (Intel + Apple Silicon).

**Inference backends** (compile-time selectable):
- **TensorRT** (default on Linux with NVIDIA GPU + TensorRT installed) — optimized inference via NVIDIA TensorRT; automatic FP16, layer fusion, and kernel auto-tuning. Engine cached to disk after first build
- **CUDA** (default on Linux with NVIDIA GPU, no TensorRT) — FP16 Tensor Core inference via WMMA; hand-written implicit GEMM kernels. Supports Turing, Ampere, Ada, Hopper, Blackwell
- **Metal** (default on macOS Apple Silicon) — GPU inference via MPSGraph with FP16 compute; 2-3× faster than OpenCL on the same hardware
- **OpenCL** — GPU inference on any OpenCL 1.2+ device (NVIDIA, AMD, Intel); hand-written implicit GEMM kernels with fused BN/ReLU
- **RKNN** (aarch64 Linux with Rockchip NPU — RK3562/RK3566/RK3568/RK3576/RK3588) — NPU inference via Rockchip's `librknnrt`; fp16 or int8/hybrid quantisation, multi-core NPU support (auto-distributed across NPU cores). Requires offline ONNX → .rknn conversion on an x86_64 host with `rknn-toolkit2`.
- **VIP9000** (aarch64 Linux with VeriSilicon Vivante VIP9000 NPU — Allwinner A733 / V853 / similar) — NPU inference via the VIPLite v2.0 runtime (`libNBGlinker.so` + `libVIPhal.so`); fp16-native, single core. Requires offline ONNX → `.nb` (Network Binary Graph) conversion on an x86_64 Linux host running Allwinner's official Acuity Toolkit Docker image (`ubuntu-npu:v2.0.10.1`); the pip `acuitylite` wheel is a dead end — its bundled chip table doesn't contain A733's PID `0x1000003B` (verified across 6.42–6.51).
- **Eigen** (always available) — CPU inference using Apple Accelerate / OpenBLAS

**Multi-GPU / Multi-core support**: KataGo-style architecture with N server threads, each owning
a `ComputeHandle` on its assigned GPU (or NPU core).  All threads drain from a single shared queue
— whichever device finishes first picks up the next batch (self-balancing).  On Rockchip NPUs, the
single physical NPU exposes 1–3 cores (SoC-dependent); one server thread per core pins to each core
via `rknn_set_core_mask`, giving the same topology as one-thread-per-GPU.  On Allwinner A733
(VIP9000 NanoDI+) the NPU is single-core — the driver serializes hardware command submission.
On fp16, `--nn-server-threads 1` is the right setting (extra threads add latency without
throughput).  On the int8 path, `--nn-server-threads 2 --nn-device-ids 0,0` gives a small
(~10 %) aggregate gain; 3+ threads hits driver serialisation and doesn't help.

Each search thread pre-allocates one `NNResultBuf` (mutex + condvar), matching KataGo's pattern.

**Model format:** `.onnx` (universal — loaded by all backends via a built-in minimal protobuf parser, no external protobuf dependency).  The RKNN backend additionally consumes a pre-compiled `.rknn` file that sits next to the `.onnx` (e.g. `models/best.onnx` → `models/best.rknn`): the ONNX is still parsed for metadata (board size, channel count), while the weights come from the `.rknn`.  The VIP9000 backend consumes a pre-compiled `.nb` (Network Binary Graph) bundled in a sibling directory; the resolver picks `.a733.bs<K>.int8/network_binary.nb` first and falls back to `.fp16/` (where `K = 1` for `--max-batch 1`, `4` for `--max-batch ≤ 4`).  Set `VIP9000_FORCE_PRECISION=fp16` (or `int8`) to pin one or the other; otherwise int8 wins when both directories exist.

This build can also load **KataGo** networks (`kata1` and similar) for inference.
See [KataGo inference](#katago-inference-tensorrt-only) below for the conversion + run workflow.

### Neural network architecture

KataGo-style 7-headed design.  Two architectures (ResNet, ViT) share the
same head structure.  5 heads are exported to ONNX for inference; 2 are
training-only auxiliaries that shape the trunk's internal representations.

**Inference heads (in ONNX):**

| Head | Shape | Activation | Loss | Target |
|------|-------|------------|------|--------|
| Policy | `[B, action_size]` | (logits) | soft CE | MCTS visit distribution |
| Value | `[B, 3]` → `[B, 1]` | softmax → P(W)-P(L) | CE | one-hot {win=0, loss=1, draw=2} |
| ScoreMean | `[B, 1]` | none | MSE in points² | actual game score (signed, current player) |
| ScoreStdev | `[B, 1]` | softplus | MSE in points² | `\|actual − pred_mean.detach()\|` |
| Ownership | `[B, board²]` | sigmoid | mean BCE | per-intersection {0, 1} from Tromp-Taylor scoring |

**Training-only heads (in .pt checkpoint, stripped from ONNX):**

| Head | Shape | Activation | Loss | Target |
|------|-------|------------|------|--------|
| Score Belief | `[B, num_bins]` | softmax | soft CE | Gaussian centered on score, σ=3 (computed from score, not stored) |
| Opponent Policy | `[B, action_size]` | (logits) | CE | next move from trajectory (masked if `-1`) |

**Loss weights and target contributions**

Weights are calibrated to match KataGo's proportions: policy dominant
(~55%), value strong secondary (~18%), ownership moderate (~10%),
score total ~5%, opponent ~9%.  `value_weight` ramps across plan stages
(1.5 → 5.0) because raw value CE drops from ~0.8 (init) to ~0.08
(converged) — a fixed weight would make value either dominant at init
or negligible once converged.  Table shows late-stage (Overnight extend):

| Head | Late weight | Loss function | Raw loss | Weighted | % |
|------|------------:|---------------|----------|---------:|---:|
| Policy | 1.0 | soft CE | ~1.3 | **1.27** | 55% |
| Value | 5.0 (ramped 1.5→5.0) | CE (W/L/D) | ~0.08 | **0.40** | 17% |
| Ownership | 0.85 | BCE mean | ~0.27 | **0.23** | 10% |
| Opponent Policy | 0.1 | CE | ~2.0 | **0.20** | 9% |
| Score Belief | 0.035 | soft CE (163 bins) | ~2.8 | **0.10** | 4% |
| ScoreMean | 0.015 (ramped 0.004→0.015) | **Huber(δ=12)** | ~5.7 | **0.09** | 4% |
| ScoreStdev | 0.006 | **Huber(δ=10)** | ~2.6 | **0.02** | 1% |
| **Total** | | | | **2.30** | |

`value_weight` ramp: 1.5 (bootstrap) → 1.5 → 2.0 → 3.0 → 4.0 → 5.0.
This keeps value at ~15-21% of total loss across all training stages,
matching KataGo's value proportion despite our raw value loss being
much smaller (0.08 vs KataGo's typical ~0.5 with larger models).

**KataGo comparison (loss functions).**  Score losses now use the same
Huber formulation as KataGo.  Weights are higher than KataGo's because
we lack their additional score-related heads (TD score ×3, lead,
scoring — ~5 extra heads that contribute score gradient through the
shared trunk):

| | KataGo | MiniGo |
|---|---|---|
| Score mean | Huber(δ=12) | **Huber(δ=12)** (same) |
| Score stdev | Huber(δ=10) | **Huber(δ=10)** (same) |
| Score belief | CDF MSE + PDF CE, weight 0.04 total | soft CE, weight **0.035** |
| scoreMean weight | `0.0015` | `0.004 → 0.015` (higher to compensate for lacking TD/lead) |
| scoreStdev weight | `0.001` | `0.006` |
| Value weight | `1.20` | `1.5 → 5.0` (ramped; higher because our val_raw is 10× smaller) |
| Ownership weight | `1.5` | `0.85` (lower to give room for value ramp) |

**Why ownership weight is 0.85 (not KataGo's 1.5):**
`F.binary_cross_entropy_with_logits` averages BCE over all 81
intersections, returning ~0.27 mid-training.  KataGo uses weight 1.5
but their value proportion is naturally ~20% from a higher raw value
loss.  Our raw value loss is tiny (0.08), so we need weight 5.0 on
value — giving ownership a lower weight (0.85 × 0.27 = 0.23, ~10%)
keeps the total budget balanced.

**MCTS utility formula:**
```
utility = win_loss_weight × (P(win) - P(loss))
        + score_weight × atan(scoreMean / score_scale) / (π/2)
```

| Param | Default | KataGo | Meaning |
|-------|---------|--------|---------|
| `win_loss_weight` | 1.0 | 1.0 | multiplier on P(win)-P(loss) term |
| `score_weight` | 0.0 → 0.30 (ramped) | 0.30 (fixed) | how much MCTS values score predictions |
| `score_scale` | 18.0 | `2×√boardArea` = 18 for 9×9 | atan compression: 10pt lead → `atan(10/18)/(π/2) ≈ 0.32` |

KataGo additionally integrates score utility over the score distribution
`(scoreMean, scoreStdev)` so uncertain scores are dampened.  We use a
point estimate on `scoreMean` — a reasonable approximation once the model
is trained and stdev is small.  Stdev integration is a future improvement.

**Configurability**

All 7 loss weights and 3 MCTS weights are individually configurable
three ways, with this precedence (high → low):

1. **Per-stage override** in `plan.json` stages[]:
   ```json
   { "name": "Steady improve", ...,
     "score_weight": 0.05, "ownership_weight": 2.0,
     "score_mean_weight": 0.008, "win_loss_weight": 1.2 }
   ```
2. **CLI flag** to `run_loop.py train` (overrides plan default but not stage override):
   ```bash
   python run_loop.py train --policy-weight 2.0 --ownership-weight 3.0 \
                            --score-scale 15.0
   ```
   Available flags: `--policy-weight`, `--value-weight`,
   `--score-mean-weight`, `--score-stdev-weight`, `--ownership-weight`,
   `--score-belief-weight`, `--opp-policy-weight`, `--win-loss-weight`,
   `--score-weight`, `--score-scale`.
3. **Plan defaults** in `plan.json` `training` / `mcts` sections.

### Selfplay data format

Binary V2 format with ownership and opponent action:
```
Header: [magic: 0x4D47] [version: 2] [count: i32] [board_size: i32]
Per record:
  [state_size: i32] [state: f32×S]
  [policy_size: i32] [policy: f32×P]
  [value: f32] [score: f32]
  [ownership: f32×board²]
  [opponent_action: i32]
```

## Prerequisites

### macOS (Apple Silicon — recommended)

```bash
xcode-select --install      # provides Metal frameworks + OpenCL
brew install cmake eigen     # ncurses ships with macOS
```

Metal (MPSGraph) is auto-detected.  No additional GPU libraries needed.

### macOS (Intel)

```bash
xcode-select --install      # provides OpenCL
brew install cmake eigen     # ncurses ships with macOS
```

Metal is not available on Intel Macs; OpenCL or Eigen is used.

### Linux (Ubuntu / Debian)

```bash
sudo apt install cmake g++ libeigen3-dev libncurses-dev

# For NVIDIA GPU (TensorRT — recommended, fastest):
# 1. Install CUDA toolkit: https://developer.nvidia.com/cuda-downloads
# 2. Install TensorRT (from the same NVIDIA apt repo):
sudo apt install libnvinfer-dev libnvonnxparsers-dev
# IMPORTANT: TensorRT must match your CUDA version. Check with:
#   nvidia-smi              # shows driver CUDA version (e.g. 12.8)
#   dpkg -l libnvinfer10    # shows TensorRT CUDA version
# To install a specific version matching your CUDA:
#   apt-cache madison libnvinfer-dev | grep cuda12.8
#   sudo apt install libnvinfer-dev=<version>+cuda12.8 ...
# Verify: dpkg -l | grep libnvinfer-dev

# For NVIDIA GPU (CUDA — alternative, no TensorRT dependency):
# Install CUDA toolkit only (provides nvcc compiler + runtime)
# CUTLASS headers (header-only, for optimized GEMM):
#   git clone --depth 1 https://github.com/NVIDIA/cutlass.git /tmp/cutlass
#   sudo cp -r /tmp/cutlass/include/cutlass /usr/local/include/

# For NVIDIA GPU (OpenCL — portable alternative):
sudo apt install ocl-icd-opencl-dev
# + CUDA toolkit (provides the NVIDIA OpenCL ICD)

# For AMD GPU:    sudo apt install mesa-opencl-icd
# For Intel GPU:  sudo apt install intel-opencl-icd

# Optional: faster Eigen with OpenBLAS
sudo apt install libopenblas-dev
```

### Linux (aarch64 — Rockchip NPU board, e.g. ArmSoM Sige5 / Orange Pi 5)

```bash
sudo apt install cmake g++ libeigen3-dev libncurses-dev

# Runtime library.  Already present on stock ArmSoM / Radxa / Orange Pi images;
# otherwise fetch it from airockchip/rknn-toolkit2 (rknpu2/runtime/Linux/
# librknn_api/aarch64/librknnrt.so) and drop into /usr/lib/.
ls /usr/lib/librknnrt.so   # verify

# The build embeds third_party/rknn/rknn_api.h (vendored from upstream);
# no header install needed.
```

CMake auto-detects the NPU when `librknnrt.so` is present on aarch64; alternatively
force it with `cmake .. -DMINIGO_BACKEND=rknn`.

**Conversion toolkit** (x86_64 host only — the converter does NOT run on aarch64):
```bash
# On an x86_64 Ubuntu 22.04 machine with Python 3.10:
python3 -m venv ~/.venv/rknn && source ~/.venv/rknn/bin/activate
pip install "setuptools<81" "numpy==1.26.4" "onnx==1.14.1"
git clone --depth 1 --branch v2.3.2 https://github.com/airockchip/rknn-toolkit2
pip install rknn-toolkit2/rknn-toolkit2/packages/x86_64/rknn_toolkit2-2.3.2-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
```
`rknn-toolkit-lite2` runs on aarch64 but can only *execute* `.rknn` files, not
compile ONNX → RKNN.  Conversion must happen on x86_64.

### Linux (aarch64 — Allwinner A733 / VeriSilicon VIP9000 NPU board, e.g. Radxa Cubie A7A / A7Z)

```bash
sudo apt install cmake g++ libeigen3-dev libncurses-dev git

# Runtime libs.  Not in any apt repo on Allwinner — fetch the bundled
# viplite-tina SDK from the Allwinner-published ai-sdk repo:
git clone https://github.com/ZIFENG278/ai-sdk.git /root/proj/ai-sdk

# Verify the aarch64 v2.0 libs and header are present:
ls /root/proj/ai-sdk/viplite-tina/lib/aarch64-none-linux-gnu/v2.0/
#   debug/  inc/  libNBGlinker.so  libVIPhal.so

# Verify the kernel driver is loaded (vipcore module + /dev/vipcore):
ls /dev/vipcore && lsmod | grep vipcore
```

CMake auto-detects the NPU when both the bundled SDK and `/dev/vipcore` are
present on aarch64; alternatively force it with `cmake .. -DMINIGO_BACKEND=vip9000`.
If the SDK lives elsewhere, point CMake at it with
`-DVIPLITE_ROOT=/path/to/viplite-tina` (default: `/root/proj/ai-sdk/viplite-tina`).

**Conversion toolkit** (x86_64 host only — the converter does NOT run on aarch64).
The only working path is Allwinner's official Acuity Toolkit v6.30.22 inside
the `ubuntu-npu:v2.0.10.1` Docker image; pip `acuitylite` does NOT work (its
bundled chip table is missing A733's PID `0x1000003B`).  Use the wrapper
script in this repo:
```bash
# On a real x86_64 Linux host with Docker installed and ubuntu-npu:v2.0.10.1
# loaded (image is hosted on Allwinner's Synology netdisk, not Docker Hub —
# see A733_CONVERSION.md §3.2 for the retrieval recipe):
bash tools/onnx_to_a733_docker.sh 1   # produces models/<...>.a733.bs1.fp16/network_binary.nb
bash tools/onnx_to_a733_docker.sh 4   # produces models/<...>.a733.bs4.fp16/network_binary.nb
```
Conversion must happen on x86_64; aarch64 has no Acuity binaries.  See
[A733_CONVERSION.md](A733_CONVERSION.md) for the full ONNX → `.nb` flow,
and the verification checklist in [A733_CONVERSION.md §7](A733_CONVERSION.md)
for the chip-ID gotchas to spot-check after each re-conversion.

### Python (both platforms — only needed for training)

```bash
pip install torch numpy onnx onnxscript zstandard

# Optional: FP8 training on Blackwell+ GPUs (SM 10.0)
pip install transformer_engine
```

Python is NOT required for inference — only for training (`train.py`)
and model export (`export_onnx.py`).  Multi-GPU training uses PyTorch
DistributedDataParallel via `torchrun` (included with PyTorch).

### Build dependencies summary

| Component | Required | Package |
|---|---|---|
| CMake, C++17 | All | `cmake`, `g++` |
| Eigen3 | All | `libeigen3-dev` / `brew install eigen` |
| ncurses | Play UI | `libncurses-dev` (Linux) / ships with macOS |
| CUDA toolkit | CUDA/TensorRT backends | [nvidia.com](https://developer.nvidia.com/cuda-downloads) |
| TensorRT | TensorRT backend | `libnvinfer-dev`, `libnvonnxparsers-dev` |
| CUTLASS | CUDA backend (headers only) | [github.com/NVIDIA/cutlass](https://github.com/NVIDIA/cutlass) |
| OpenCL | OpenCL backend | `ocl-icd-opencl-dev` |
| librknnrt | RKNN backend (aarch64 Linux) | `/usr/lib/librknnrt.so` from [airockchip/rknn-toolkit2](https://github.com/airockchip/rknn-toolkit2) (`rknpu2/runtime/Linux/librknn_api/aarch64/`) |
| rknn-toolkit2 | ONNX → .rknn conversion (x86_64 host only) | `pip install rknn-toolkit2==2.3.2` |
| viplite-tina SDK | VIP9000 backend (aarch64 Linux) | `libNBGlinker.so` + `libVIPhal.so` from [ZIFENG278/ai-sdk](https://github.com/ZIFENG278/ai-sdk) (`viplite-tina/lib/aarch64-none-linux-gnu/v2.0/`) |
| Acuity Toolkit v6.30.22 | ONNX → .nb conversion (x86_64 Linux + Docker only) | Allwinner's `ubuntu-npu:v2.0.10.1` Docker image (Synology netdisk — see A733_CONVERSION.md §3.2). pip `acuitylite` does NOT work — chip table missing A733 PID. |
| PyTorch | Training | `pip install torch` |
| Transformer Engine | FP8 training (optional) | `pip install transformer_engine` |

## Build

```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc 2>/dev/null || sysctl -n hw.ncpu)
cd ..
```

CMake auto-detects the best backend for your platform:
```
-- Backend:    metal      (macOS Apple Silicon)
-- Backend:    tensorrt   (Linux with NVIDIA GPU + CUDA + TensorRT)
-- Backend:    cuda       (Linux with NVIDIA GPU + CUDA, no TensorRT)
-- Backend:    rknn       (aarch64 Linux with librknnrt.so — Rockchip NPU)
-- Backend:    vip9000    (aarch64 Linux with viplite-tina SDK — Allwinner A733 / VIP9000)
-- Backend:    opencl     (Linux with GPU, no CUDA)
-- Backend:    eigen      (no GPU available)
```

Force a specific backend:
```bash
cmake .. -DMINIGO_BACKEND=tensorrt # TensorRT (NVIDIA, fastest — requires libnvinfer-dev)
cmake .. -DMINIGO_BACKEND=cuda     # CUDA FP16 Tensor Cores (NVIDIA)
cmake .. -DMINIGO_BACKEND=metal    # Metal/MPSGraph (macOS Apple Silicon)
cmake .. -DMINIGO_BACKEND=opencl   # OpenCL (Linux, macOS)
cmake .. -DMINIGO_BACKEND=rknn     # Rockchip NPU (aarch64 Linux — RK3562/66/68/76/88)
cmake .. -DMINIGO_BACKEND=vip9000  # VeriSilicon VIP9000 NPU (aarch64 Linux — Allwinner A733)
cmake .. -DMINIGO_BACKEND=eigen    # CPU only (no GPU)
```

> **Tip (conda environments):** If CUDA was installed via conda, the conda
> GCC toolchain's sysroot can conflict with the host's glibc headers
> (`__time64_t` errors).  Fix by pointing CMake at the system compiler:
> ```bash
> cmake .. -DCMAKE_C_COMPILER=/usr/bin/gcc -DCMAKE_CXX_COMPILER=/usr/bin/g++
> ```

Compile with clang instead of gcc:
```bash
cmake .. -DMINIGO_USE_CLANG=ON      # resolves clang/clang++ from PATH
```
Must be passed on a **fresh build directory** — CMake caches the compiler
after the first configure, so switching gcc ↔ clang on an existing build
requires `rm -rf build/*` (or a new build dir).  Only selects C/C++;
nvcc's host compiler for `.cu` files is controlled separately via
`CMAKE_CUDA_HOST_COMPILER`.

CUDA and TensorRT share the same `CMAKE_CUDA_ARCHITECTURES` default:
Turing/Ampere/Ada SASS + Hopper PTX (forward-compat with Blackwell and beyond).
Note: this flag only affects `.cu` files (CUDA backend); TensorRT has no `.cu`
files — it JIT-compiles its own kernels at engine build time on the target GPU.
For faster local builds targeting one GPU:
```bash
cmake .. -DMINIGO_BACKEND=cuda -DCMAKE_CUDA_ARCHITECTURES=75      # Turing (RTX 2080 Ti)
cmake .. -DMINIGO_BACKEND=cuda -DCMAKE_CUDA_ARCHITECTURES=89      # Ada (RTX 4090)
```

The backend is selected at **compile time** — no `--backend` flag at runtime.
All binaries automatically use whichever backend was compiled.

## Quick Start

### Train (automated pipeline)

The training pipeline uses two commands: `init` (choose network, generate
training plan) and `train` (run or resume training).

```bash
# 1. Initialize — pick a preset or custom architecture.  Training plan and
# default model size are both set by the preset, but you can override either
# with --filters/--blocks (or --d-model/--depth) to decouple them.
python run_loop.py init small                # 9x9 ResNet 64f/5b,   48 iters (~24M  sim-games)
python run_loop.py init small --arch vit     # 9x9 ViT  d192/8L,    48 iters
python run_loop.py init large                # 9x9 ResNet 128f/10b, 72 iters (~47M  sim-games, 2x small)
python run_loop.py init xlarge               # 9x9 ResNet 128f/10b, 200 iters (~670M sim-games, deep run)
python run_loop.py init quick                # 5x5 ResNet 32f/3b,   5 iters (test)
python run_loop.py init quick --arch vit     # 5x5 ViT (test)

# 2. Train — GPUs are auto-detected, just run:
python run_loop.py train

# Or with explicit hardware settings:
python run_loop.py train --threads 64 --nn-device-ids 0,0,1,1 --max-batch 512

# Run a limited number of iterations then pause:
python run_loop.py train --iterations 20

# 3. Check progress:
python run_loop.py status
```

**Resumable** — stop at any time (Ctrl+C) and re-run `python run_loop.py train`
to continue from where it left off.  Pipeline state, selfplay data,
checkpoints, and training logs are all preserved.

#### Training pipeline

Each iteration runs three phases:

1. **Self-play**: generate games with the current best model (C++, multi-GPU)
2. **Train**: train on a sliding window of recent data (Python/PyTorch, multi-GPU DDP)
3. **Evaluate & gate**: play games between candidate and best model; promote
   if candidate wins ≥ 55% (configurable)

The pipeline is controlled by a **training plan** (`training/plan.json` file)
generated during `init`.  The plan defines staged training with escalating
parameters.  Example (ResNet, `large` preset — 72 iters, ~47M sim-games):

```
  Stage               Iters    Games  Sims  Epoch      LR   Gate
  ──────────────────────────────────────────────────────────────
  Bootstrap           1-4       400   200     3    1.2e-3   off
  Warm up             5-8       600   300     3      9e-4   off
  Early gated         9-14      900   400     3      6e-4   100g
  Consolidate        15-24     1100   450     4    4.5e-4   200g
  Steady improve     25-40     1300   550     5      3e-4   200g
  Overnight extend   41-72     1400   600     5      2e-4   200g
```

Each stage defines: selfplay games per iteration, MCTS simulations per move,
training epochs, learning rate, and evaluation games for gating.  Early stages
use fewer sims and no gating for fast exploration; later stages increase data
quality and enable gating to ensure only stronger models are promoted.

Per-stage overrides also control all **loss weights** (7 training head
weights) and **MCTS utility weights** (`win_loss_weight`, `score_weight`,
`score_scale`).  These escalate across stages — e.g. `score_weight` starts
at 0 (ignore score during bootstrap) and increases to 0.1+ in late stages
once the score head is reliable.

The three main presets are different training schedules — all three work
with any model size (override with `--filters`/`--blocks`):

| Plan     | Iters | Sim-games | vs small | Default model |
|----------|------:|----------:|---------:|---------------|
| `small`  |    48 |    ~24M   |     1.0× | 64f/5b        |
| `large`  |    72 |    ~47M   |     2.0× | 128f/10b      |
| `xlarge` |   200 |   ~670M   |    28.4× | 128f/10b      |

The plan is a JSON file — edit it to customize the schedule.

#### Model version management

All models live in `models/` with a simple versioning scheme:
- `models/v0000.onnx` — initial random model (created by `init`)
- `models/v0001.onnx` ... `models/v0060.onnx` — candidates from each iteration
- `models/best.onnx` — copy of the current best (used for self-play)
- `training/checkpoints/v0001.pt` ... — training checkpoints with optimizer state

Only models that pass evaluation gating are promoted to `best.onnx`.
The `training/state` file tracks progress for resume.

#### Logging

All training metrics are written to `training/logs/train.log` in real-time:
per-iteration parameters, per-epoch losses, selfplay timing, evaluation
win rates, and promotion decisions.  This single file captures the full
training history for debugging and tuning.

#### Value-head diagnostics (per epoch)

Raw `val` loss alone hides three very different failure modes for the
value head (W/L/D classifier): genuine good fit, easy-position-dominated
average (most 9×9 samples are decided endgame positions where any model
gets CE≈0.02), and overconfident collapse where predicted probabilities
no longer track empirical accuracy.  Each epoch therefore emits an
extra `val-diag` line to stdout AND `train.log`:

```
Epoch 1/3  loss=2.2274  pol=1.2676  val=0.0787  smn=10.8004 ...
    val-diag  phase CE: early=0.412(16%) mid=0.134(24%) late=0.018(60%)
              ECE=0.037  calib[0.33-0.50:conf=0.421/acc=0.398@5% ...]
```

The diagnostic is **value-head-only** — policy, score, and ownership
heads aren't classifiers with "confidence", so their loss numbers are
already informative.  Two metrics:

**Phase CE** — value cross-entropy bucketed by game phase, where phase
is estimated from stone count on the board (channel 0 + channel 8 of
the encoded state = current-snapshot stones for both colors):

| Bin | Stones | Corresponds to | Healthy CE |
|---|---|---|---|
| `early` | 0–15 | Opening; position fluid | 0.3–0.6 |
| `mid` | 16–40 | Middle game; decisive fights | 0.1–0.3 |
| `late` | 41+ | Endgame; usually decided | 0.01–0.05 |

The `(XX%)` annotation is the fraction of samples in that bin.  For
typical 9×9 training (80–130 move games), expect roughly **15/25/60**.
`late%` near 0 signals games too short to reach endgame (rare after
iter ~4).

**ECE (Expected Calibration Error)** — measures whether predicted
confidence tracks empirical accuracy.  Formally: Σ (|B|/N) · |acc−conf|
across 4 confidence buckets `[0.33,0.5)`, `[0.5,0.7)`, `[0.7,0.9)`,
`[0.9,1.0]`.  The `calib[...]` field shows per-bucket `conf/acc/@%`
so you can see *where* miscalibration lives.

#### How to read the diagnostic

**Healthy output — do nothing:**
```
val-diag  phase CE: early=0.42(15%) mid=0.15(25%) late=0.02(60%)
          ECE=0.03  calib[... conf≈acc in every bucket ...]
```
- Monotone drop early → late: position info is being used.
- ECE < 0.05: predictions are calibrated.
- `conf ≈ acc` in every bucket with ≥10% samples.

**Red-flag patterns and what to do:**

| Pattern | Likely cause | Action |
|---|---|---|
| `early ≈ mid ≈ late` (flat) | Model ignores position features; trunk underfit or value head bottlenecked | Increase `value_weight`, check trunk capacity, verify training data isn't corrupted |
| `early > 1.0` (very high) | Value head not learning opening at all | Raise `value_weight`; check that `value_target` derivation is correct |
| `early < 0.1` but `ECE` large | Overconfident on opening positions (dangerous — MCTS will over-commit at root) | Lower `value_weight`, add label smoothing, or regularize |
| `late% > 80%` | Games very long / opening samples scarce | Usually fine — selfplay just produces long games; consider data augmentation if opening play is weak |
| `late% ≈ 0%` | Games too short to reach endgame | Usually only iter 1–3; self-heals as model improves |
| `ECE > 0.10` | Confidence doesn't match accuracy | Value head is miscalibrated; MCTS will make bad decisions.  Lower `value_weight` or add KL-to-uniform regularization |
| `[0.9–1.0]` bucket: `conf=0.95, acc=0.70` | Overconfident on "easy" positions | Dangerous; model is assigning near-certainty to positions that aren't.  Check for value-head collapse or insufficient data variety |
| `[0.7–0.9]` bucket: `conf=0.80, acc=0.50` | Overconfident on hard middle-game positions | Most actionable — these are the positions MCTS actually searches.  Increase diversity in selfplay, lower temperature threshold, or bump `value_weight` |

**When to bump `value_weight`:** if `early CE > 0.5` for 3+ iterations
in a row while policy loss keeps dropping — value is lagging behind
policy.  Default `1.5` may be too low for small models; try `2.5–3.0`.

**When to lower `value_weight`:** if `ECE > 0.10` or overconfidence in
the `[0.9-1.0]` bucket persists — too much gradient on value is
collapsing the head.  Try `1.0` and add score-head supervision via a
higher `score_mean_weight`.

Diagnostic cost: one softmax + one cross-entropy with `reduction='none'`
per batch under `torch.no_grad()` — well under 1% of step time.

DDP note: each rank computes on its own data shard; rank-0 prints.
For 2-GPU configs this is close to global; for 4+ GPU configs the
per-bucket counts may be noisier.

#### GPU auto-detection

The `train` command auto-detects NVIDIA GPUs and configures:
- **C++ selfplay/evaluate**: 2 NN server threads per GPU with pipelining
  - 1 GPU → `--nn-server-threads 2 --nn-device-ids 0,0`
  - 2 GPUs → `--nn-server-threads 4 --nn-device-ids 0,0,1,1`
- **Python training**: auto-launches via `torchrun` with DDP (DistributedDataParallel)
  when multiple GPUs are detected.  Each GPU runs its own process with NCCL
  gradient synchronization.  Data is split across GPUs; effective batch size
  scales with GPU count.

Override with explicit flags if needed.

#### Evaluation binary

The `evaluate` binary plays match games between two models to determine
which is stronger.  Games are saved as SGF files for review:

```bash
./build/evaluate --model1 candidate.onnx --model2 baseline.onnx \
    --games 100 --sims 400 --threshold 0.55 --output eval_games/
# Exit code 0 = model1 wins (above threshold)
# Exit code 1 = model1 fails
# SGF game records saved to eval_games/game_*.sgf
```

Each model gets its own NNEvaluator with separate compute contexts.
Games alternate which model plays Black.  Temperature is 0 (deterministic)
with no Dirichlet noise for clean evaluation.  During training, evaluation
games are saved to `training/eval/iter_NNNN/`.

#### Visualizing games

Review selfplay or evaluation games with the visualizer:

```bash
# Selfplay game (binary format, supports .bin / .bin.zst / .bin.gz)
python scripts/visualize.py training/selfplay/iter_0001/game_0.bin.zst

# Evaluation game (SGF format)
python scripts/visualize.py training/eval/iter_0006/game_0.sgf

# All games in a directory
python scripts/visualize.py training/eval/iter_0006/
```

The visualizer uses ncurses with the same board style as the play UI.
Controls: Arrow keys or Enter = next/prev, `s` = skip to end, `q` = quit.
For `.bin` files: shows value (V) and score (S) per move from training data.

### Play

```bash
# Against trained model (backend selected at compile time)
./build/play --model models/best.onnx --sims 800

# Human vs Human (with optional analysis)
./build/play --model models/best.onnx  # choose H at mode prompt

# Against random bot (no model needed)
./build/play --random --board 9
```

The play UI uses **ncurses** with ACS line-drawing for a clean terminal board.
Arrow keys or coordinate typing (e.g., `D4`) to place stones — no Enter needed
for movement.

**Modes**: (B)lack vs AI, (W)hite vs AI, (H)uman vs Human.

#### Hotkeys

| Key | Action |
|---|---|
| Arrows / WASD | Move cursor |
| Enter / Space | Place stone at cursor |
| Typed coord (e.g. `D4`) | Jump cursor + place stone |
| `p` (lowercase) | Pass |
| `a` | Toggle analysis HUD (live evaluation display) |
| `o` | Toggle ownership overlay (+/- territory markers on the board) |
| `P` (Shift+P) | Toggle pondering (background search) |
| `r` | Restart game / refresh screen |
| `q` | Quit |

#### Pondering and analysis (two independent toggles)

The play binary exposes two orthogonal user preferences:

- **P — Pondering**: a background MCTS search runs while the engine is
  otherwise idle (your think time during AI-vs-human, or any time during
  human-vs-human).
- **A — Analysis**: a 500ms callback fires and the HUD displays the tree's
  live evaluation — win rate, score, top moves with visit counts.

**Coupling rule**: A requires P.  You can't display the analysis of a tree
that isn't being searched, so turning A on auto-enables P, and turning P off
auto-clears A.  Three reachable states:

| State | P | A | Meaning |
|---|---|---|---|
| **off** | ❌ | ❌ | Bot idle; no background search, no HUD |
| **ponder** | ✅ | ❌ | Silent background search; engine thinks without HUD clutter |
| **analyze** | ✅ | ✅ | Background search + live HUD (numbers update every 500ms) |

**State transitions**:

| From | press `a` → | press `P` → |
|---|---|---|
| off | analyze (A on, P auto-on) | ponder (P on) |
| ponder | analyze (A on, P stays on) | off (P off) |
| analyze | ponder (A off, P stays on) | off (P off, A auto-off) |

**Usefulness per game mode**:

| Game Mode | off | ponder | analyze |
|---|---|---|---|
| **Human vs AI** | Engine only thinks on its own turn (classic Go GUI behavior) | Tournament-style: engine also searches during your turn, no HUD clutter | Engine thinks both turns + live HUD shows what it's computing, including **during the AI's own move** |
| **Human vs Human** | Pure manual play, no engine | *Wasted compute* — search runs but nothing is displayed | Live evaluation of the current position (study mode) |

Notes:

- In human-vs-AI, **ponder** alone (P only) is genuinely useful: the engine
  exploits your idle time to search ahead, and its next move comes out faster.
- In human-vs-human, **ponder** alone is technically valid but pointless
  (nobody sees the tree).  You'll typically go `off` → `analyze` directly.
- **analyze** shows live updates through BOTH pondering and the AI's move
  selection — `gen_move` runs through the same AsyncBot worker thread as
  `ponder`, so the callback fires throughout.  You literally see what the
  AI is thinking while it thinks.

**Example HUD output** (analyze state, HvAI, mid-game):
```
WR 65.0%  Score +5.3 +/- 2.1  N=1234
| D5   65.0% n=523
| E3   22.1% n=234
| C6   12.4% n=128
```

The score shows the expected point margin ± standard deviation from the
neural network's scoreMean and scoreStdev heads.
The number of top moves shown is set by `--pvs K` (default 5).

When pondering without analysis (P only, A off), the panel shows a
placeholder: `[pondering — press a for HUD]`.

#### Tree preservation guarantees

The MCTS tree is a persistent object inside the `AsyncBot` and **is preserved
across most state changes**:

- **Toggling `a` or `P`**: tree is preserved.  The current search is stopped
  briefly to reconfigure the callback; the next search reuses the existing
  root via `reuse_tree=true`.
- **Making a move** (human via `play_move`, AI via `gen_move`): the tree is
  **re-rooted** to the chosen move's child.  The chosen subtree is preserved;
  unexplored siblings are discarded.  Still "tree reuse" in the sense that
  no NN re-evaluation of the new root is needed.
- **Toggling off → on**: tree persists across the off state.  `stop()` stops
  the search thread but doesn't touch the tree.  Re-enabling picks up where
  it left off.

The tree is **destroyed and rebuilt from scratch** only in these cases:

- **Starting a new game** (`r` for restart, or selecting a new mode): explicit
  `reset_tree()`.
- **Game over → restart**: same as above.
- **Playing a move into an unexplored branch**: if you play a move that the
  previous search never visited (so `root_->children[action]` is null),
  `make_move` drops the tree.  Rare in practice — it can only happen if you
  make a move immediately after enabling ponder, before the first playout
  has even completed.
- **Worker exception**: if the background search throws (e.g. a backend
  failure), the worker logs and exits the iteration.  Tree state is retained
  but may be partially incomplete; the next search rebuilds from what's left.

Toggles never rebuild the tree — not even if you hit `a` within microseconds
of `P`, catching the worker mid-NN-eval.  The root NN evaluation inside
`search()` is a synchronous call to `evaluator_->evaluate(...)` that cannot
be interrupted by `request_stop()`; it runs to completion and installs the
fresh root *before* the playout phase checks the stop flag.  By the time
the interrupted search returns, `root_` is already EXPANDED, and the next
search reuses it.  The only thing lost is the handful of playouts that
would have run after the stop was raised.

### Benchmark

```bash
# Benchmark the current best model
./build/benchmark --model models/best.onnx \
    --games 10 --threads 10 --search-threads 16 --max-batch 512

# Generate a standalone model for benchmarking
cd scripts
python3 export_onnx.py --init --filters 128 --blocks 10 --output ../models/bench_large.onnx
cd ..
./build/benchmark --model models/bench_large.onnx \
    --games 10 --threads 10 --search-threads 16
```

### KataGo inference (TensorRT only)

You can run a stock **KataGo** network (`kata1` and similar) inside this engine's
MCTS, for human play, benchmark, and match games. KataGo weights are
**inference-only** in this build — they cannot be used to generate selfplay
training records or fine-tuned. See `KATAGO_INFERENCE.md` for full details and
known limitations (ladder features and a few encore-only signals are zeroed).

#### What accepts KataGo weights

| Tool | Input format | Purpose | KataGo accepted? |
|---|---|---|---|
| `tools/katago_to_onnx.py` | `.txt.gz` / `.bin.gz` | one-shot conversion to ONNX | yes — required first step |
| `tools/katago_parity_test.py` | `.txt.gz` + `.onnx` | validate the converted ONNX | yes |
| `tools/warm_init_from_katago.py` | `.txt.gz` / `.bin.gz` | warm-init MiniGo's trunk (training prep) | yes (existing tool — unrelated to inference) |
| `build/play` | `.onnx` (KataGo or MiniGo) | interactive play | yes |
| `build/evaluate` | `.onnx` × 2 | match games (kata1 vs MiniGo, kata1 vs kata1, …) | yes |
| `build/benchmark` | `.onnx` (KataGo or MiniGo) | NN/MCTS throughput | yes (sections 1–4; section 5 selfplay is auto-skipped) |
| `build/selfplay` | `.onnx` | generate training records | **no — refuses KataGo with `return 2`** |
| `scripts/train*.py`, `run_continuous.py` | `.pt` / `.onnx` | training pipeline | no — KataGo never enters training |

Backend support: **TensorRT** (NVIDIA, x86_64), **RKNN** (Rockchip NPU,
aarch64), **VIP9000** (VeriSilicon NPU on Allwinner A733, aarch64).
Eigen / CUDA / OpenCL / Metal throw `"KataGo format requires the TensorRT
backend"` at handle creation; the NPU backends consume their respective
pre-compiled artifacts (`.rknn` / `.nb`) generated alongside the ONNX.

> ⚠ I tested `selfplay` to **confirm the rejection guard fires**, not to use it.
> Selfplay refuses KataGo models on purpose — the V2 record format and the
> 8-fold augmentation are MiniGo-shaped, so feeding KataGo states through them
> would produce corrupt training data. Use `play` / `evaluate` / `benchmark`
> for KataGo runs.

#### How to use it (5 steps)

```bash
# 1.  Download a KataGo network (or any kata1 .bin.gz / .txt.gz).
wget https://media.katagotraining.org/uploaded/networks/models/kata1/kata1-b10c128-s1141046784-d204142634.txt.gz

# 2.  Convert to ONNX. The exporter bakes post-processing (softmax → P(W)−P(L),
#     score × 20, ownership (tanh+1)/2) into the graph so the C++ side reads
#     the same shapes whether the model is MiniGo or KataGo. Inputs differ:
#     MiniGo has one input; KataGo has state_spatial [N,22,H,W] + state_global [N,19].
#     Pick the board size you want to run at — the engine is per-board-size.
python tools/katago_to_onnx.py \
    --katago-bin kata1-b10c128-s1141046784-d204142634.txt.gz \
    --board 9 \
    --output models/kata1-b10c128.onnx

# 3.  (optional) Validate PyTorch ↔ ONNX Runtime parity.
python tools/katago_parity_test.py \
    --katago-bin kata1-b10c128-s1141046784-d204142634.txt.gz \
    --onnx models/kata1-b10c128.onnx --board 9

# 4.  Build with the TensorRT backend (KataGo is TRT-only).
cmake -B build -DMINIGO_BACKEND=tensorrt
make -C build -j

# 5.  Run.  First run on each (GPU × max-batch × precision) builds a TRT
#     engine and caches it under trt_cache/; later runs reuse it.
./build/play       --model models/kata1-b10c128.onnx --sims 800 --komi 7.0
./build/benchmark  --model models/kata1-b10c128.onnx --max-batch 256 --sims 256 --komi 7.0
./build/evaluate   --model1 models/kata1-b10c128.onnx \
                   --model2 models/kata1-b10c128.onnx \
                   --games 50 --sims 200 --komi 7.0
```

`--komi` matters: kata1's typical 9×9 komi is `7.0` or `7.5`, and the value
flows into KataGo's global feature vector. A mismatched komi silently degrades
strength rather than erroring.

#### KataGo on the Allwinner A733 (VIP9000 backend)

```bash
# 1. Convert the kata1 weights once on an x86_64 host (see A733_CONVERSION.md).
#    The result is a pair of directories per batch size — int8 (preferred,
#    quantised against a calibration fixture) and fp16 (lossless reference):
#      models/kata1-b10c128.a733.bs4.unshared.onnx
#      models/kata1-b10c128.a733.bs4.int8/network_binary.nb     ← used at runtime
#      models/kata1-b10c128.a733.bs4.fp16/network_binary.nb     ← reference / fallback
#      (and bs=1 variants, if you also need single-state latency)

# 2. Build with the VIP9000 backend (auto-detected on the Cubie A7A).
cmake -B build -DMINIGO_BACKEND=vip9000
make -C build -j

# 3. Live play.  --max-batch 4 selects the bs=4 NBG (and the int8 sibling
#    if both .int8/ and .fp16/ exist); --search-threads 16 keeps each
#    bs=4 batch full so the NPU isn't padding 3 of 4 slots with zeros.
./build/play       --model models/kata1-b10c128.a733.bs4.unshared.onnx \
                   --max-batch 4 --search-threads 16 \
                   --sims 800 --komi 7.0

# 4. Benchmark (sections 1, 2, 4 work; section 3's batch sweep includes
#    8/32/... which exceed the compiled bs=4 — use vip9000_smoke for that).
./build/benchmark  --model models/kata1-b10c128.a733.bs4.unshared.onnx \
                   --max-batch 4 --search-threads 16 \
                   --nn-iters 100 --sims 256 --komi 7.0

# 5. Match games (kata1 vs kata1, etc.).
./build/evaluate   --model1 models/kata1-b10c128.a733.bs4.unshared.onnx \
                   --model2 models/kata1-b10c128.a733.bs4.unshared.onnx \
                   --games 50 --sims 200 --komi 7.0 \
                   --max-batch 4 --search-threads 16
```

**Why `--search-threads 16` matters with bs=4**: the bs=4 NBG always
processes 4 batch slots regardless of how many leaves MCTS has ready.
With `--search-threads 1`, you'd burn ~3.4 ms/call for 1 useful state
(≈290 effective inf/s) instead of ~1167 inf/s.  KataGo's recommendation
of 8–32 search threads applies — 16 is a good default.

**Bs=1 latency mode** (single-state, lower per-call latency, slightly
weaker policy quality on the current bs=1 int8 calibration):

```bash
./build/play --model models/kata1-b10c128.a733.bs1.unshared.onnx \
             --max-batch 1 --search-threads 1 \
             --sims 800 --komi 7.0
```

**Standalone tools** (built only with this backend):

```bash
# Time a fresh NBG end-to-end, bypassing MCTS / NNEvaluator
./build/vip9000_smoke    --model models/kata1-b10c128.a733.bs4.unshared.onnx \
                         --batch 4 --iters 500 --threads 1

# Compare int8 vs fp16 outputs on N random positions (top-1/3/5 +
# value/score/ownership MAE).  Use after each new int8 calibration.
./build/vip9000_accuracy --model models/kata1-b10c128.a733.bs4.unshared.onnx \
                         --batch 4 --positions 200 --seed 42
```

`vip9000_smoke` prints `policy[0..2]`, `value`, `score`, `score_sd`,
`ownership[0..1]` from the first call so you can eyeball sanity, then
times `iters` calls back-to-back.  `vip9000_accuracy` runs the same model
under both precisions in one process and reports policy top-K agreement
plus per-head MAE (treating fp16 as ground truth).

**A733-specific tuning** (single-core VIP9000 NanoDI+):
- **bs=4 int8 is the production default** — top-1 79 % / top-5 100 %
  vs fp16 on the revised-calibration NBG, ~1167 states/s with
  `--search-threads 16`.  See the perf and quality tables in the
  "VIP9000 NPU Backend" architecture section for the full numbers.
- **`--nn-server-threads 1`** for play (one MCTS in flight feeds the
  queue at search-thread rate — a second server thread doesn't help).
  For self-play with multiple parallel games, `--nn-server-threads 2
  --nn-device-ids 0,0` gives ~10 % more aggregate throughput on int8.
- **bs=1 is for latency-sensitive single-state callers only** — the
  current bs=1 int8 NBG has top-1 = 7.5 % vs fp16, so MCTS-driven
  workloads (play/evaluate) would lose strength.  Use bs=4 unless
  you really need 0.82 ms/call.
- **Memory**: each bs=4 int8 NBG is ~6.6 MB on disk and ~10–15 MB
  resident after `vip_prepare_network`.  No user tuning needed.

## Architecture

```
            LoadedModel (1, shared, main thread)
                │ CPU weights (ONNX parsed once)
                ▼
          ComputeContext (1, shared, main thread)
          ┌─────┴──────────────────┐
          │                        │
     DeviceState[GPU 0]      DeviceState[GPU 1]
          │                        │
     ┌────┼────┐              ┌────┼────┐
     │         │              │         │
  Handle_0  Handle_1       Handle_2  Handle_3
  (server0) (server1)      (server2) (server3)
  own bufs  own bufs        own bufs  own bufs
     │         │              │         │
     └─────────┴──────────────┴─────────┘
                      │
              NNEvaluator (1 instance)
     ┌────────────────────────────────────┐
     │  Shared Queue (NNResultBuf*)       │ ← search threads push
     │  Competing consumers               │ → N server threads pop
     └────────────────────────────────────┘
                      │
              MCTS Search Threads
     ┌────────────────────────────────────┐
     │  Worker 1 (game 1) ─┐             │
     │  Worker 2 (game 2) ─┤  search()   │
     │  Worker T (game T) ─┘  spawns N   │
     │    threads per move               │
     └────────────────────────────────────┘
                      │
              Self-Play Data (.bin)
                      ▼
          ┌─── Python ───────────┐
          │  train.py (PyTorch)  │
          │  export_onnx.py      │──▶ models/*.onnx
          └──────────────────────┘
```

**All backends use the same NNEvaluator architecture** — even Eigen (CPU).
Each server thread creates its own `ComputeHandle` on its assigned GPU
(KataGo pattern).  The backend is selected at compile time via
`cmake -DMINIGO_BACKEND=...`.

### Multi-Threaded MCTS (KataGo pattern)

Each `MCTS::search()` call spawns `--search-threads` internal threads, all
working on the same tree with virtual loss for synchronization.  Each search
thread pre-allocates one `NNResultBuf` (mutex + condvar, created once,
reused for all evaluations during that thread's lifetime — matching KataGo's
`SearchThread` pattern):

1. **Descend**: walk the tree from root, applying virtual loss at each node.
   Only descend through `EXPANDED` nodes; stop at `UNEVALUATED` leaves.

2. **Evaluate**: call `evaluator->evaluate_with_buf(buf, state)` — pushes
   the pre-allocated `NNResultBuf` to the server queue and **blocks** on
   its condvar until the server processes the batch containing this leaf.

3. **Expand**: CAS `UNEVALUATED → EXPANDING → EXPANDED`.  Only one thread
   expands each node; others that collide revert their virtual losses,
   `yield()`, and retry from the root.

4. **Backprop**: undo virtual loss, increment visit count, update value
   (all via atomics — `std::atomic<int>` for counts, CAS loop for float value).

**Batch size adapts naturally**: while the GPU processes batch N, search
threads descend and submit leaves for batch N+1.  Steady-state batch size
≈ total concurrent search threads across all games.

```
total_search_threads = min(games, threads) × search_threads
```

### AsyncBot — persistent worker + tree reuse (KataGo pattern)

The `AsyncBot` class (`include/async_bot.h`) wraps `MCTS + GoGame` and runs
all searches on **one persistent worker thread** that idle-waits on a
condvar between requests — matching KataGo's `internalSearchThreadLoop`
design.  A single `worker_loop()` handles all search modes (GENMOVE for
AI moves, PONDER for background search), so there's only one state
machine to reason about.

**Public API** (single-writer contract):

```cpp
// Callback configuration (persistent; applies to any subsequent search)
void set_callback(AnalysisCallback cb, int interval_ms, int max_pv);
void clear_callback();

// Synchronous operations (block until done)
int  gen_move(Stone color, int sims, float temp, bool noise);
bool play_move(Stone color, int action);

// Asynchronous operations (return immediately)
void start_ponder();                         // background search, no cb
void start_analyze(cb, interval_ms, max_pv); // set_callback + start_ponder
void stop();                                  // interrupt, wait for idle

// Queries (safe from any thread)
MCTS::AnalysisInfo get_analysis(int max_moves);
bool is_searching() const;
```

**Internals**: `pending_mode_` / `current_mode_` ∈ `{IDLE, GENMOVE,
PONDER, SHUTDOWN}` protected by `control_mutex_`.  Callers submit a
request by setting `pending_mode_` and notifying `worker_cv_`, then wait
on `done_cv_` for `current_mode_` to return to IDLE.  The worker spawns
a transient callback thread per iteration when a callback is configured,
joins it at the end of the iteration, then loops back to wait.

**Stop-flag ownership**: `MCTS::should_stop_` is set by `request_stop()`
(called from `stop_locked()`) and checked inside the playout loop.
Critically, it is **NOT** cleared inside `MCTS::search()` — the clear
happens in `AsyncBot::worker_loop` via `mcts_->reset_stop_flag()`
*inside* the same critical section as the `current_mode_` transition.
This avoids a race where a stop signal raised after the worker released
the lock but before it entered `search()` would be silently overwritten
by search()'s own clear.  Non-AsyncBot callers (selfplay, eval,
benchmark) never set the flag, so leaving it default-false is safe.

**Tree reuse**: the `MCTS` instance owns one persistent `root_` that
survives across `search()` calls.  `MCTS::make_move(action)` re-roots
the tree to the played child (preserving its subtree, discarding
siblings outside the lock).  Every search call passes `reuse_tree=true`
so subsequent searches accumulate visits rather than rebuilding.  See
`COMPARISON_WITH_KATAGO.md` §13 for the point-by-point comparison with
KataGo's `Search::makeMove` + `Search::beginSearch`.

**Live analysis flow** (used by the play UI):

1. `bot->set_callback(cb, 500, 10)` — register a callback that pushes
   an `AnalysisInfo` snapshot to a mutex-protected UI struct every 500ms.
2. `bot->start_ponder()` — spawns a GENMOVE or PONDER search via the
   worker; the worker spawns a callback thread for the duration.
3. UI thread reads the latest snapshot each frame under the same mutex
   and redraws.
4. On human move: `bot->play_move(color, action)` internally stops the
   ponder (worker's search returns via `request_stop`), advances game +
   tree, returns.  UI loop restarts ponder at the top of the next
   iteration via `maintain_bot_state`.
5. On AI move: `bot->gen_move(color)` runs a GENMOVE search through the
   same worker — the callback thread fires throughout, so the user sees
   the AI's search tree evolve live.

`MCTS::get_analysis(max_moves)` itself is a simple reader — it holds
`tree_mutex_` briefly, walks `root_->children`, and returns:

```cpp
struct AnalysisInfo {
    vector<MoveInfo> moves;   // top moves sorted by visits
    int   total_visits;       // root visit count
    float root_utility;       // mean Q (blended value + score)
    float root_score;         // NN raw score estimate (points)
};
```

Each `MoveInfo` contains: action, visits, prior (policy), utility
(Q-value).  Visit counts are atomic; priors are stable after the root
is expanded; `nn_score` is stored per-node so re-rooted trees carry
their own correct score without needing re-evaluation.

### KataGo-style NNEvaluator

The `NNEvaluator` (in `include/nn_evaluator.h`) is a server thread that:
1. Waits for at least one `NNResultBuf*` in the shared queue
2. Greedy drains up to `max_batch_size` items (no timeout, no threshold)
3. Runs one `predict_batch()` GPU call
4. Signals each client's condvar with the result

Each search thread's `NNResultBuf` is pre-allocated at the start of the
thread and reused across all evaluations — zero mutex/condvar creation
in the hot path.

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

`OpenCLComputeHandle` (`src/opencl_compute.cpp`) implements the full AlphaZero
forward pass using hand-written OpenCL kernels:

| Kernel | Purpose |
|---|---|
| `transpose_nchw_to_cnhw` | GPU-side NCHW → channel-major transpose |
| `conv3x3_sgemm_bn` | Implicit GEMM: fused im2col + register-blocked SGEMM + BN/residual/ReLU |
| `conv1x1_bn_relu_reshape` | Fused 1×1 conv + BN + ReLU + layout reshape for FC input |
| `fc_bias_relu` | Fused FC GEMM + bias + optional ReLU |
| `fc_bias_softmax` | Fused FC + bias + softmax (policy head) |
| `fc_bias_tanh` | Fused FC + bias + tanh (value head) |

The implicit GEMM kernel (`conv3x3_sgemm_bn`) computes im2col indices
on-the-fly during B-tile loading, eliminating the separate im2col scratch
buffer.  Register blocking (WPT_M=2, WPT_N=4) gives 8 outputs per work-item
with shared-memory tiling.

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

Per-thread streams make **inference** lock-free, but they do NOT make
`IExecutionContext` **lifecycle** operations thread-safe.  Per NVIDIA
TRT 10 docs, `ICudaEngine::createExecutionContext()` and
`~IExecutionContext` are not thread-safe with respect to other context
creation/destruction on the same engine — the engine keeps an internal
list of live contexts that both operations mutate.

With `--nn-device-ids 0,0,1,1`, two server threads share one engine per
GPU.  At process exit, `~NNEvaluator` calls `notify_all()` and all four
threads tear down their handles in parallel → each calls `delete
exec_ctx` on contexts that point into the same engine → concurrent
mutation of the internal list corrupts it → the damage only surfaces
when `~ICudaEngine` walks the list at teardown, manifesting as
`double free or corruption (out)` after `Done! N games`.

Fix: one `engine_mutex` per `TRTDeviceState` held around (a) engine
build, (b) `createExecutionContext()`, and (c) `delete exec_ctx`.
Inference (`enqueueV3`, `setTensorAddress`, `setInputShape`) stays
unlocked — each thread still owns its own `exec_ctx` and stream, so
the GPU scheduler interleaves kernels across threads as before.  This
matches KataGo's `trtbackend.cpp`: one per-engine lock for lifecycle,
zero locks for inference.

The **OpenCL** backend has a shared `cl_command_queue` per device but is
currently disabled (the KataGo-style ResNet requires SE/GPool kernels not
yet implemented in OpenCL).  When re-enabled, it should follow the same
pattern: one queue per handle, not per device.

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

## Resumable Training

The pipeline is fully resumable at every phase boundary.  Run
`python run_loop.py train` after any interruption to continue:

- **Pipeline state** (`training/state`): tracks current iteration, best model
  version, total games played, and promotion count
- **Selfplay resume**: skips iterations that already have enough game files
- **Training resume**: skips iterations whose versioned ONNX + checkpoint exist
- **Checkpoints** (`training/checkpoints/training.pt`): model weights + optimizer (Adam for ResNet, AdamW for ViT)
  state (momentum buffers) for smooth continuation.  Mixed precision
  auto-detected (FP8/BF16/FP16, see Precision table above)
- **Selfplay data**: accumulates in per-iteration directories
  (`training/selfplay/iter_0001/`, etc.) and is never deleted

Training uses a **sliding window** — only data from the last N iterations
is loaded (configurable via `PLAN_WINDOW_SIZE` in the training plan),
keeping training focused on recent, stronger games.  This is the standard
approach used by AlphaGo Zero and KataGo.

Selfplay data is compressed with **zstd** after generation (~10x smaller on
disk).  The training DataLoader streams and decompresses one file at a time
with 8 prefetch workers, keeping GPU utilization high with minimal memory.

## Neural Network

Two architectures (`scripts/model.py`), selected with `--arch`:

**ResNet** (default): KataGo-style trunk with alternating residual blocks —
block 0 is an SE-attention block (3x3 → 3x3 → squeeze-excitation → residual),
block 1 is a global-pooling block (parallel 3x3 main + 3x3 pool branches,
where the pool branch is globally mean+max pooled and projected to per-channel
biases that are added into the main branch before the second 3x3), and so on.
Value and score heads use global-pooled heads (1x1 conv → small channel count
→ mean+max pool → MLP) instead of the classic AlphaZero 1x1-to-1-channel +
flattened-FC design, which collapses all channel information before the FC.
**ViT**: Vision Transformer with one token per intersection, GQA, and
directional positional encoding (factorized row/col embedding + signed
relative bias for full spatial and directional awareness).

**Backend support for the new ResNet is currently TensorRT-only.**  The
Eigen/CUDA/OpenCL/Metal backends still contain the old AlphaZero-ResNet
forward pass but throw at handle creation until hand-written kernels for
SE/GPool blocks and global-pool heads are added (see `TODO.md`).

Both architectures share 7 output heads (KataGo-style).  4 drive MCTS at
inference time and are exported to ONNX; 3 are training-only auxiliaries
that shape the trunk's internal representations but are never evaluated
during inference (stripped from the `.onnx` file to save model size and
compute).

### Heads used by MCTS (exported to ONNX)

| Head | Model output | ONNX output | Training loss |
|------|-------------|-------------|---------------|
| **Policy** | `[B, 82]` logits | `policy_logits [B, 82]` | CE with soft MCTS visit distribution |
| **Value** | `[B, 3]` logits (W/L/D) | `value [B, 1]` = P(win)−P(loss) | CE over {win, loss, draw}, weight 1.5 |
| **ScoreMean** | `[B, 1]` raw float | `score_mean [B, 1]` (points) | MSE vs actual game score, weight 0.5 |
| **ScoreStdev** | `[B, 1]` softplus | `score_stdev [B, 1]` (points) | MSE vs |actual−predicted|, weight 0.5 |

The value head predicts a 3-class distribution: P(win), P(loss), P(draw).
The ONNX export appends `softmax → P(win) − P(loss)` post-processing so
C++ receives `value [B, 1]` in [-1, +1] as before.  ScoreMean is a direct
regression of the final score margin (in points, from the current player's
perspective) — replaces the old 163-bin classification, which suffered from
hard one-hot targets and stagnant training loss.  ScoreStdev captures the
model's uncertainty about its score estimate.

### Training-only auxiliary heads (NOT in ONNX)

| Head | Model output | Training loss | Weight |
|------|-------------|---------------|--------|
| **Ownership** | `[B, board²]` sigmoid | BCE per intersection | 1.5 / board² |
| **Score Belief** | `[B, num_bins]` logits | CE with soft Gaussian target (σ≈3) | 0.15 |
| **Opponent Policy** | `[B, action_size]` logits | CE vs opponent's next move | 0.15 |

**Ownership** is the most impactful auxiliary: it gives every trunk block
per-intersection territory supervision ("you own D4 but lost E7"), which
is far richer than the single-scalar game-outcome signal the value head
provides.  Without it the model struggles to learn endgame territory
concepts.  The ownership target comes from the game-end board state
(reusing `GoGame::score()`'s existing flood-fill logic).

**Score Belief** forces the trunk to represent score *uncertainty* — wide
distributions in sharp tactical positions, narrow peaks in settled endgames.
The target is a Gaussian centered on the actual game score, not a hard
one-hot bin.  The resulting features help ScoreMean and Value be more
accurate even though MCTS never reads the belief distribution itself.

**Opponent Policy** teaches threat-awareness by predicting what the
opponent plays next (available from the selfplay trajectory).  Lowest-
impact auxiliary (~10-20 Elo in KataGo ablations) but nearly free.

### MCTS Utility

Each leaf evaluation produces a blended utility that MCTS backpropagates:

```
utility = winLossWeight × value
        + scoreWeight   × atan(scoreMean / scoreScale) / (π/2)
```

- `value = P(win) − P(loss)` from the 3-class value head
- `scoreMean` is the regression output (points, current player's perspective)
- `atan` compression maps ±∞ → [-1, +1], preventing extreme scores from
  dominating the utility
- `winLossWeight` (default 1.0), `scoreWeight` (default 0.02, ramped up in
  later training stages), `scoreScale` (default 10.0) are configurable

The score signal teaches MCTS to prefer moves that **win by more points**,
which prevents aimless play in won positions and teaches the model to close
out games decisively.  Without it, MCTS treats "win by 1" and "win by 30"
identically.

### .pt vs .onnx

The PyTorch checkpoint (`.pt`) contains all 7 heads — needed for training
continuation.  The ONNX file (`.onnx`) contains only the 4 MCTS heads —
this keeps inference fast and the model file small.  The 3 training-only
heads add ~55K params to `.pt` but are stripped from `.onnx`.

### Precision

Auto-detected per GPU — best available precision used for both training
and inference:

| GPU | Training | Inference (TensorRT) |
|---|---|---|
| **Blackwell** (SM 10.0+) | FP8 via Transformer Engine (`te.Linear`, E4M3 fwd / E5M2 bwd) | FP8 (`kFP8` builder flag) |
| **Ampere/Ada** (SM 8.0+) | BF16 (`torch.amp.autocast`) | FP16 |
| **Turing** (SM 7.5) | FP16 + GradScaler | FP16 |
| **Pascal** (SM 6.x) | FP16 + GradScaler | FP16 |
| **CPU / older** | FP32 | N/A |

FP8 training requires `pip install transformer_engine` and `"fp8": true` in
the training plan.  Without it, Blackwell falls back to BF16.  FP8 inference
via TensorRT is automatic (no extra config needed).

**ViT positional encoding** — directional (D4 symmetry via data augmentation):
- *Factorized position*: `row_embed[r] + col_embed[c]` — 9+9=18 learned embeddings, full spatial resolution
- *Directional relative bias*: signed `(dx, dy)` offsets — 289 buckets for 9×9. Each direction is unique
  (north ≠ south ≠ east ≠ west), enabling the model to learn directional attention for captures,
  ladders, and edge awareness
- *GQA*: 6 query heads, 2 KV groups (3 queries share each K/V group)

See the **MCTS Utility** section above for how the score head drives search.
`score_weight` (default 0.02) and `score_scale` (default 10.0) are
configurable via CLI flags.  Set `score_weight` to 0 to disable score utility.

**Komi** (compensation for white) defaults to **6.5** for 9×9 and is configurable
via `--komi` in all executables and `PLAN_KOMI` in the training plan.

## Model Format

All backends use **ONNX** (`.onnx`) as the universal model format.
The project includes a built-in minimal protobuf parser (`onnx_loader.cpp`)
— **no external protobuf library required**.

- `LoadedModel::load()` parses the ONNX protobuf once and pre-fuses BN parameters
- Each `ComputeHandle` uploads these CPU weights to its own GPU buffers

The model has dynamic batch dimensions, so the same `.onnx` file works for
batch sizes 1 through N.  GPU workspace buffers auto-grow to fit the batch.

Generate standalone models of different sizes (for benchmarking):
```bash
cd scripts
python3 export_onnx.py --init --board 9 --filters 64  --blocks 5  --output ../models/small.onnx
python3 export_onnx.py --init --board 9 --filters 128 --blocks 10 --output ../models/large.onnx
python3 export_onnx.py --init --board 9 --filters 256 --blocks 20 --output ../models/xlarge.onnx
```

## Files

```
minigo-cpp/
├── CMakeLists.txt              # Build (Eigen required, OpenCL/Metal optional)
├── run_loop.py                 # Training pipeline (init/train/status)
│   ├── plan.json               #   Generated training schedule (editable)
├── models/                     # ONNX model files
│   ├── best.onnx               #   Current best (used for selfplay)
│   └── v0001.onnx ...          #   Version snapshots
├── trt_cache/                  # TensorRT compiled engine cache
├── training/                   # All training artifacts
│   ├── selfplay/               #   Game data (iter_0001/, iter_0002/, ...)
│   ├── eval/                   #   Evaluation match SGFs (iter_0006/, iter_0007/, ...)
│   ├── checkpoints/            #   PyTorch checkpoints (training.pt, v0001.pt, ...)
│   ├── logs/                   #   train.log (structured), pipeline.log
│   └── state                   #   Pipeline resume state
├── test_multi_gpu.sh           # Multi-GPU test suite
├── include/
│   ├── config.h                # Hyperparameters
│   ├── game.h                  # Go engine (ring buffer history, fast is_legal)
│   ├── loaded_model.h          # Shared CPU weights (ONNX parsed once)
│   ├── compute_context.h       # ComputeContext + ComputeHandle base classes
│   ├── batch_evaluator.h       # BatchEvaluator interface + NNResultBuf
│   ├── nn_evaluator.h          # KataGo-style batching server (N server threads)
│   ├── async_bot.h             # Persistent worker wrapper (ponder + analyze)
│   ├── eigen_compute.h         # Eigen CPU backend (context + handle)
│   ├── opencl_compute.h        # OpenCL GPU backend (context + handle)
│   ├── cuda_compute.h          # CUDA GPU backend (context + handle)
│   ├── tensorrt_compute.h      # TensorRT GPU backend (context + handle)
│   ├── metal_compute.h         # Metal/MPSGraph GPU backend (macOS)
│   ├── onnx_loader.h           # Built-in minimal ONNX protobuf parser
│   └── mcts.h                  # Multi-threaded MCTS (atomic MCTSNode)
├── src/
│   ├── game.cpp                # Full Go rules (captures, ko, scoring)
│   ├── onnx_loader.cpp         # ONNX weight parser (no external dependency)
│   ├── loaded_model.cpp        # ONNX parsing + BN pre-fusion
│   ├── compute_context.cpp     # Backend factory
│   ├── eigen_compute.cpp       # Eigen context + handle
│   ├── opencl_compute.cpp      # OpenCL context + handle + embedded kernels
│   ├── cuda_compute.cu         # CUDA context + handle + FP16 WMMA kernels
│   ├── tensorrt_compute.cpp    # TensorRT context + handle (ONNX→engine)
│   ├── metal_compute.mm        # Metal/MPSGraph context + handle (Obj-C++)
│   ├── nn_evaluator.cpp        # NNEvaluator N server threads (KataGo pattern)
│   ├── mcts.cpp                # Multi-threaded MCTS + tree reuse (make_move)
│   ├── async_bot.cpp           # Persistent worker thread + callback reporting
│   ├── main_play.cpp           # Human vs AI (uses AsyncBot for ponder/analyze)
│   ├── main_selfplay.cpp       # Multi-threaded data generation
│   ├── main_evaluate.cpp       # Model vs model evaluation matches
│   └── main_benchmark.cpp      # Performance tests
└── scripts/
    ├── model.py                # PyTorch model definition
    ├── export_onnx.py          # PyTorch → ONNX export
    ├── train.py                # Train on self-play data (DDP, streaming)
    └── visualize.py            # Selfplay/eval game viewer (.bin/.zst/.sgf)
```

## CLI Reference

### run_loop.py

```
python run_loop.py init <preset>      Initialize training (clears previous state)
  Presets: quick, small, large
  Custom:  init --board 9 --filters 96 --blocks 8

python run_loop.py train [options]    Start or resume training
  --threads N             Worker threads (default: all cores)
  --search-threads N      MCTS search threads per move (default: 16)
  --selfplay-instances N  Parallel selfplay processes (default: 1)
  --nn-server-threads N   NN server threads (default: auto-detect)
  --nn-device-ids IDS     GPU indices, comma-sep (default: auto-detect)
  --max-batch N           Max GPU batch size for NN server (default: 256)
  --iterations N          Max iterations this session (default: all)

python run_loop.py status             Show training progress
```

### selfplay

```
./build/selfplay [options]
  --model PATH           Model file (default: models/best.onnx)
  --games N              Number of games (default: 100)
  --threads N            Parallel workers (default: 1)
  --search-threads N     MCTS search threads per move (default: 16)
  --max-batch N          Max GPU batch size (default: 256)
  --output DIR           Output directory (default: training/selfplay)
  --sims N               MCTS simulations per move (default: 800)
  --c-puct F             UCB exploration constant (default: 1.5)
  --dirichlet-alpha F    Root noise concentration (default: 0.15)
  --dirichlet-epsilon F  Root noise weight (default: 0.25)
  --temp-threshold N     Moves of stochastic play (default: 15)
  --score-scale F        Score atan compression scale (default: 10.0)
  --nn-server-threads N  NN server threads (default: 1)
  --nn-device-ids IDS    Comma-separated GPU indices (default: "0")
```

### play

```
./build/play [options]
  --model PATH           Model file (default: models/best.onnx)
  --sims N               MCTS simulations per move (default: 800)
  --search-threads N     MCTS search threads (default: 16)
  --max-batch N          Max GPU batch size (default: 256)
  --c-puct F             UCB exploration constant (default: 1.5)
  --komi F               Komi value (default: 6.5)
  --score-weight F       Score utility weight (default: 0.0)
  --score-scale F        Score atan compression scale (default: 10.0)
  --nn-server-threads N  NN server threads (default: 1)
  --nn-device-ids IDS    GPU indices (default: "0")
  --pvs K                Top K moves shown in analysis HUD (default: 5)
  --random               Use random bot (no model needed)
  --board N              Board size (for --random mode)
```

In-game hotkeys: `a` analysis HUD, `P` (shift+p) pondering,
`p` pass, `r` restart, `q` quit.  See the Play section above for
the full state machine.

### evaluate

```
./build/evaluate [options]
  --model1 PATH          Candidate model (required)
  --model2 PATH          Baseline model (required)
  --games N              Games to play (default: 100)
  --threads N            Parallel game workers (default: 1)
  --search-threads N     MCTS threads per move (default: 16)
  --sims N               MCTS simulations per move (default: 800)
  --max-batch N          Max GPU batch size (default: 256)
  --threshold FLOAT      Win rate to pass (default: 0.55)
  --c-puct F             UCB exploration constant (default: 1.5)
  --score-scale F        Score atan compression scale (default: 10.0)
  --output DIR           Save game records as SGF files
  --nn-server-threads N  NN server threads per model (default: 1)
  --nn-device-ids IDS    GPU indices (default: "0")
```

Exit code 0 = model1 wins (above threshold), 1 = model1 fails.
SGF files can be reviewed with `python scripts/visualize.py`.

### benchmark

```
./build/benchmark [options]
  --model PATH           Model file (default: models/best.onnx)
  --sims N               MCTS simulations
  --nn-iters N           NN inference iterations (default: 1000)
  --games N              Self-play games (default: 5)
  --threads N            Self-play worker threads (default: 1)
  --search-threads N     MCTS search threads per move (default: 16)
  --max-batch N          Max GPU batch size (default: 256)
  --score-scale F        Score atan compression scale (default: 10.0)
  --nn-server-threads N  NN server threads (default: 1)
  --nn-device-ids IDS    Comma-separated GPU indices (default: "0")
```

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

## Troubleshooting

**cmake can't find Eigen3**: `brew list eigen` or `dpkg -l libeigen3-dev`

**OpenCL not found**: On Linux install `ocl-icd-opencl-dev` + a GPU driver ICD;
on macOS OpenCL ships with Xcode Command Line Tools.

**Metal not found**: Requires macOS with Apple Silicon.  Check that
Xcode Command Line Tools are installed (`xcode-select --install`).

**TensorRT not found**: Install with `sudo apt install libnvinfer-dev libnvonnxparsers-dev`.
Requires NVIDIA's CUDA apt repository. Verify with `dpkg -l | grep libnvinfer-dev`.

**TensorRT CUDA version mismatch**: TensorRT must match your CUDA driver version.
Check `nvidia-smi` (driver CUDA version) vs `dpkg -l libnvinfer10` (TensorRT CUDA version).
If mismatched, install the TensorRT package built for your driver's CUDA version.

**TensorRT engine rebuild**: Cached engines are stored in `trt_cache/` beside the model directory
and are GPU-specific and TensorRT-version-specific.
Delete the `trt_cache/` directory to force a rebuild after upgrading TensorRT or switching GPUs.

**Build without GPU**: `cmake .. -DMINIGO_BACKEND=eigen` (CPU-only)

**OpenCL kernel compile error**: shown in the exception message; usually means
the GPU doesn't support the feature used.  File a bug with the error text.

**Training interrupted**: Just re-run `python run_loop.py train` — it resumes automatically
from the last completed iteration.  Check `python run_loop.py status` to see progress.

**Slow on macOS with OpenCL**: Rebuild with Metal backend:
`cmake .. -DMINIGO_BACKEND=metal && make -j$(sysctl -n hw.ncpu)`.
Metal with MPSGraph FP16 is 2-3× faster than OpenCL on Apple Silicon.
