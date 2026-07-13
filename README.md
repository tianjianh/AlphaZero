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
- **OpenCL** — GPU inference on any OpenCL 1.2+ device (NVIDIA, AMD, Intel); all three model architectures (resnet / vit / katago-V7) with fp32, portable-fp16 and NVIDIA tensor-core (inline-PTX `mma.sync`) precision tiers
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

This build also supports the **KataGo V7 model format** end to end: converted
`kata1` networks run in every binary (including selfplay — V3 game records are
engine-neutral), and the katago architecture is trainable from scratch with
`--arch katago`.  See [KataGo V7 models](#katago-v7-models--running-and-training)
below and `FORMATS.md`.

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

Weights are calibrated toward KataGo's proportions: policy dominant,
value strong secondary, ownership moderate, score small.  In the
continuous pipeline two weights ramp **by training step** (not by
phase): `value_weight` ramps `1.0 → 2.0` over `--value-ramp-steps`
(default 30k) because raw value CE shrinks as the model converges, and
`score_mean_weight` ramps `0.004 → 0.008` over `--score-ramp-steps`
(default 50k) so the noisy early score signal doesn't distort the trunk.

| Head | Weight (continuous defaults) | Loss function |
|------|------------------------------|---------------|
| Policy | 1.0 | soft CE |
| Value | 1.0 → 2.0 (step ramp) | CE (W/L/D) |
| Ownership | 0.85 | BCE mean |
| Opponent Policy | 0.1 | CE |
| Score Belief | 0.035 | soft CE (163 bins) |
| ScoreMean | 0.004 → 0.008 (step ramp) | **Huber(δ=12)** |
| ScoreStdev | 0.006 | **Huber(δ=10)** |

**KataGo comparison (loss functions).**  Score losses use the same
Huber formulation as KataGo.  Weights differ because we lack their
additional score-related heads (TD score ×3, lead, scoring — ~5 extra
heads that contribute score gradient through the shared trunk):

| | KataGo | MiniGo |
|---|---|---|
| Score mean | Huber(δ=12) | **Huber(δ=12)** (same) |
| Score stdev | Huber(δ=10) | **Huber(δ=10)** (same) |
| Score belief | CDF MSE + PDF CE, weight 0.04 total | soft CE, weight **0.035** |
| scoreMean weight | `0.0015` | `0.004 → 0.008` (higher to compensate for lacking TD/lead) |
| scoreStdev weight | `0.001` | `0.006` |
| Value weight | `1.20` | `1.0 → 2.0` (step-ramped) |
| Ownership weight | `1.5` | `0.85` (BCE averaged over 81 points is already dense signal) |

**MCTS utility formula:**
```
utility = win_loss_weight × (P(win) - P(loss))
        + score_weight × atan(scoreMean / score_scale) / (π/2)
```

| Param | Default | KataGo | Meaning |
|-------|---------|--------|---------|
| `win_loss_weight` | 1.0 | 1.0 | multiplier on P(win)-P(loss) term |
| `score_weight` | 0 → `--score-weight-max` (0.04), ramped by trainer's `score_ramp` | 0.30 (fixed) | how much MCTS values score predictions |
| `score_scale` | 18.0 | `2×√boardArea` = 18 for 9×9 | atan compression: 10pt lead → `atan(10/18)/(π/2) ≈ 0.32` |

The selfplay driver reads `score_ramp` from `training/status.json` each
batch and passes `score_weight = score_weight_max × score_ramp` to the
C++ engine, so search only starts caring about score once the score head
has had gradient steps to become meaningful.  KataGo additionally
integrates score utility over the score distribution
`(scoreMean, scoreStdev)` so uncertain scores are dampened.  We use a
point estimate on `scoreMean` — a reasonable approximation once the model
is trained and stdev is small.  Stdev integration is a future improvement.

**Configurability**

All 7 loss weights and the MCTS weights are flags on
`run_continuous.py run` (forwarded to the trainer / selfplay driver):
`--policy-weight`, `--value-weight-start/-end`,
`--score-mean-weight-start/-end`, `--score-stdev-weight`,
`--ownership-weight`, `--score-belief-weight`, `--opp-policy-weight`,
`--score-weight-max`, `--score-scale`, plus `--value-ramp-steps` /
`--score-ramp-steps` for the ramp lengths.

### Selfplay data format (V3 — engine-neutral game records)

One file per game.  Records store the GAME (moves + per-move MCTS
policy + outcome + final ownership), **not encoded states and no
pre-baked augmentation** — the trainer replays the moves and encodes
positions on the fly for whichever architecture it trains (MiniGo
17-plane or KataGo V7), applying a random dihedral transform per
sample.  That makes one record pool serve all three architectures,
lets ANY model format generate selfplay data (including converted
kata1 nets), and shrinks records ~60×.

```
Header: [magic:u16 'MG'] [version:u16 = 3] [board_size:i32] [komi:f32]
        [n_moves:i32] [winner:i8 0/1/2] [black_score:f32]
Per move: [action:i16  (hw = pass)] [policy: f32×(hw+1)]
Footer:   [owner: i8×hw  (0 empty/dame, 1 black, 2 white)]
```

Parser, replay engine, both encoders, and augmentation live in
`scripts/gamedata.py`; per-position targets (value/score/ownership/
opponent action) are derived at load time.  See `FORMATS.md`.

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

# NOTE (NVIDIA + CUDA "compat" packages): if /usr/local/cuda-*/compat
# shadows the driver libs in ldconfig with a NEWER version than the
# kernel driver, NVIDIA's OpenCL JIT loads a mixed 560/575-style chain
# and clBuildProgram segfaults non-deterministically (or errors with
# "Unsupported .version"). Force the driver-matched chain, e.g.:
#   export LD_PRELOAD="/usr/lib/x86_64-linux-gnu/libcuda.so.<drv> \
#     /usr/lib/x86_64-linux-gnu/libnvidia-nvvm.so.<drv> \
#     /usr/lib/x86_64-linux-gnu/libnvidia-ptxjitcompiler.so.<drv>"
# (CUDA/PyTorch in other processes keep using the compat chain.)

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

### Train (continuous pipeline)

Training is **continuous, KataGo-style**: selfplay, training, gatekeeping,
and (optionally) rating run as concurrent processes coordinating through
the filesystem — no phased iterations.  Two commands:

```bash
# 1. Initialize — choose the architecture ONCE.  It is recorded in
#    training/run_config.json; `run` reads it back, so you never
#    re-specify --filters/--blocks (contradicting flags are an error).
python scripts/run_continuous.py init                        # 9x9 ResNet 128f/10b, komi 7.5
python scripts/run_continuous.py init --filters 64 --blocks 5
python scripts/run_continuous.py init --arch vit --d-model 192 --depth 8

# 2. Run — launches all workers under one supervisor:
python scripts/run_continuous.py run \
    --train-gpus 0 --selfplay-gpus 1 --gate-gpus 1 \
    --selfplay-nn-device-ids 0,0 --gate-nn-device-ids 0

# 3. Check progress any time:
python scripts/run_continuous.py status
tail -f logs/current/supervisor.log      # HEARTBEAT lines roll up all workers
```

Stop with Ctrl-C (workers finish their current batch/match, trainer
checkpoints, everything resumes on the next `run`).  A second Ctrl-C
force-kills.

See **[CONTINUOUS_TRAINING.md](CONTINUOUS_TRAINING.md)** for the full
design.  The essentials:

#### The four workers

| Worker | GPU use | What it does |
|---|---|---|
| `selfplay_driver.py` | selfplay GPUs | Loops `build/selfplay` batches with `models/accepted/latest`; compresses games to `training/selfplay/g_<id>.bin.zst`; prunes the pool to `--window-games` |
| `train_continuous.py` | train GPUs (DDP via torchrun) | One continuous step loop over a sliding window; exports a candidate ONNX every `--export-every` steps |
| `gatekeeper.py` | gate GPUs | Evaluates the newest candidate vs `accepted/latest` with `build/evaluate`; promotes on `score > threshold` (default 0.5); stale-drops older candidates |
| `rate.py` (opt-in `--rating`) | rate GPUs | Periodic round-robin Elo over recent accepted models (monitoring only) |

#### Rollout ↔ training synchronization (two-sided valve)

The trainer paces itself with a KataGo-style **replay bucket**
(`python/train.py` `max_train_bucket_per_new_data` in upstream): every new
selfplay row (one unique position in V3) credits `replay_target` samples of budget;
each step drains `global_batch`.  Empty bucket → trainer sleeps
(`BUDGET_SLEEP`) instead of over-replaying stale data.

The reverse direction — **selfplay outpacing training** — is handled by
backpressure that KataGo's distributed setup doesn't need but a single
host does: the trainer publishes `bucket_fill` in `training/status.json`,
and the selfplay driver **pauses between batches** when the bucket
saturates (`THROTTLE_PAUSE` at ≥ 0.9 fill) and resumes once training
drains it (< 0.5).  Without this, games beyond the bucket cap would burn
selfplay GPU time on credit that gets discarded.  Tune with
`--throttle-high/--throttle-low` (0 disables).

#### Model lifecycle

```
models/
├── accepted/            # promoted models; latest -> v<step>.onnx symlink
│   ├── v000000000.onnx  # seed from init (random or warm-init)
│   └── latest           # what selfplay plays with (atomic symlink swap)
├── candidates/          # trainer exports, awaiting gate
└── rejected/            # failed gate or stale-dropped
training/
├── run_config.json      # architecture + komi, written by init
├── checkpoints/training.pt   # weights + optimizer + bucket/watermark state
├── selfplay/g_*.bin.zst      # game pool (sliding window)
└── status.json          # trainer → selfplay/supervisor contract
```

**Resumable** — the trainer checkpoints weights, optimizer, step, bucket
level, and scanner watermark on every export and on shutdown; selfplay IDs
are recovered from directory state.  Re-running `run` continues everywhere
it left off.

#### Warm-starting from KataGo weights

`tools/warm_init_from_katago.py` can seed the run with kata1 b10c128 trunk
weights instead of random init — see the tool's docstring for the exact
workflow (replace `accepted/v000000000.onnx` and optionally
`training/checkpoints/training.pt` between `init` and `run`).

#### Monitoring

- `logs/current/supervisor.log` — one `HEARTBEAT` line per interval:
  trainer step/state, bucket fill, pool size, candidates/accepted counts.
- `logs/current/train.log` + `train_metrics.csv` — per-step losses, LR,
  ramps, bucket, ring occupancy.
- `logs/current/selfplay.log` + `selfplay_batches.csv` — per-batch games,
  positions, throttle waits.
- `logs/current/gatekeeper.log` + `gate_decisions.csv` — match results.
- `training/status.json` — live trainer state (`state`, `step`,
  `bucket_fill`, `score_ramp`, ...).

#### Evaluation binary

The `evaluate` binary plays match games between two models (used by the
gatekeeper, and directly for ad-hoc matches).  Games can be saved as SGF:

```bash
./build/evaluate --model1 candidate.onnx --model2 baseline.onnx \
    --games 100 --sims 400 --threshold 0.55 --output eval_games/
# Exit code 0 = model1 wins (above threshold)
# Exit code 1 = model1 fails
# SGF game records saved to eval_games/game_*.sgf
```

Each model gets its own NNEvaluator with separate compute contexts.
Games alternate which model plays Black.  Temperature is 0 (deterministic)
with no Dirichlet noise for clean evaluation.

#### Visualizing games

Review selfplay or evaluation games with the visualizer:

```bash
# Selfplay game (binary format, supports .bin / .bin.zst / .bin.gz)
python scripts/visualize.py training/selfplay/g_00000000000000001.bin.zst

# Evaluation game (SGF format)
python scripts/visualize.py eval_games/game_0.sgf

# All games in a directory
python scripts/visualize.py eval_games/
```

The visualizer uses ncurses with the same board style as the play UI.
Controls: Arrow keys or Enter = next/prev, `s` = skip to end, `q` = quit.
For `.bin` files: shows value (V) and score (S) per move from training data.

### Play

```bash
# Against your trained model (backend selected at compile time)
./build/play --model models/accepted/latest --sims 800

# Human vs Human (with optional analysis)
./build/play --model models/accepted/latest  # choose H at mode prompt

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
./build/benchmark --model models/accepted/latest \
    --games 10 --threads 10 --search-threads 16 --max-batch 512

# Generate a standalone model for benchmarking
cd scripts
python3 export_onnx.py --init --filters 128 --blocks 10 --output ../models/bench_large.onnx
cd ..
./build/benchmark --model models/bench_large.onnx \
    --games 10 --threads 10 --search-threads 16
```

### KataGo V7 models — running AND training

The KataGo V7 model format (dual input `state_spatial [B,22,H,W]` +
`state_global [B,19]`, same 5 inference outputs as ours) is a
first-class citizen:

- **Run a stock kata1 network**: convert once with
  `tools/katago_to_onnx.py`, then use it in `play`, `evaluate`,
  `benchmark` — and `selfplay`: V3 game records are encoding-free, so
  a kata1 net can generate training data for ANY architecture.
- **Train the katago architecture from scratch**: `--arch katago`
  everywhere (`run_continuous.py init --arch katago`,
  `train_continuous.py`, `export_onnx.py`).  `KataGoNet`
  (scripts/model.py) is a KataGo-style pre-activation trunk with
  MiniGo's 7-head set, trained by the same loop as resnet/vit from the
  same V3 record pool, exported to the same dual-input ONNX contract.
- Stock kata1 *checkpoints* are not resumable by the trainer (their
  heads differ from our 7-head set); they run as-is or warm-init.

See `KATAGO_INFERENCE.md` for encoder fidelity notes (ladder planes
and encore-only signals are zeroed) and **`FORMATS.md`** for the full
format-support matrix.

| Tool | KataGo-format ONNX accepted? |
|---|---|
| `build/play` / `build/evaluate` / `build/benchmark` | yes (all sections) |
| `build/selfplay` | **yes — V3 records are engine-neutral** |
| `scripts/train_continuous.py` | yes via `--arch katago` (fresh or resumed `KataGoNet`) |
| `tools/katago_to_onnx.py` / `katago_parity_test.py` / `warm_init_from_katago.py` | stock-weight conversion / validation / warm-init |

Backend support for the dual-input format: **TensorRT** (full),
**OpenCL** (full — both the converted-kata1 and trainable-KataGoNet
namings, mish/relu autodetected), **Eigen** (CPU forward,
converted-kata1 naming), **RKNN / VIP9000** (via their pre-compiled
artifacts).  CUDA / Metal accept it at the interface but their kernels
are TODO placeholders (as for the MiniGo formats).

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

3. **Expand**: claim `UNEVALUATED → EXPANDING` with a single `fetch_or`
   (states encoded so the claimed bit is idempotent on `EXPANDED`), publish
   with a release store.  Only one thread expands each node; others that
   collide revert their virtual losses, `yield()`, and retry from the root.
   No `compare_exchange` anywhere on the search path — everything lowers to
   single AMOs, so it stays cheap on Zaamo-only RISC-V, x86, and ARM alike.

4. **Backprop**: undo virtual loss, increment visit count, update value
   (all via atomics — `std::atomic<int>` `fetch_add` for counts, fixed-point
   `std::atomic<int64_t>` `fetch_add` for the value sum — exact and
   interleaving-independent, unlike float accumulation).

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

## Resumable Training

The continuous pipeline resumes cleanly after any interruption — just
re-run `python scripts/run_continuous.py run`:

- **Trainer checkpoint** (`training/checkpoints/training.pt`): model
  weights + optimizer state (Adam for ResNet, AdamW for ViT) + step +
  replay-bucket level + scanner watermark, saved atomically on every
  export AND on graceful shutdown (SIGTERM/Ctrl-C).  Mixed precision is
  auto-detected (FP8/BF16/FP16, see Precision table above).
- **Selfplay IDs**: monotonic `g_<id>.bin.zst` IDs are recovered from
  directory state (`max(existing) + 1`) — no counter file to corrupt.
  Orphaned `.tmp` files and staging dirs are cleaned at startup.
- **Ring warm-up**: on resume, each rank's in-RAM ring rehydrates from
  the NEWEST ~`--ring-games` pool files (not the oldest), so training
  restarts on fresh data.
- **Gatekeeper / models**: promotion state is the filesystem itself
  (`accepted/`, `candidates/`, `rejected/`, `latest` symlink) — nothing
  else to restore.

Training samples from a **sliding window**: the pool keeps the newest
`--window-games` games on disk; each rank's ring holds the newest
`--ring-games` in RAM (compressed; decompressed lazily per batch).
This is the standard recent-window approach used by AlphaGo Zero and
KataGo, bounded by host RAM instead of a shuffle-daemon.

Selfplay data is compressed with **zstd** at publish time (~10× smaller
on disk); the ring stores the compressed bytes and pays ~tens of ms of
decompression per batch instead of ~20× the host RAM.

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
├── CMakeLists.txt              # Build (backend auto-detect: tensorrt/cuda/metal/…)
├── models/                     # Model lifecycle (created by run_continuous init)
│   ├── accepted/               #   Promoted models + `latest` symlink (selfplay reads)
│   ├── candidates/             #   Trainer exports awaiting the gatekeeper
│   ├── rejected/               #   Failed gate / stale-dropped
│   └── trt_cache/              #   Shared TensorRT engine cache (MINIGO_TRT_CACHE)
├── training/
│   ├── run_config.json         #   Architecture + komi (written by init, read by run)
│   ├── selfplay/               #   Game pool: g_<id>.bin.zst (sliding window)
│   ├── checkpoints/training.pt #   Trainer checkpoint (weights+optimizer+bucket)
│   └── status.json             #   Trainer → selfplay/supervisor live state
├── logs/<timestamp>/           # Per-run logs; logs/current symlink
├── include/
│   ├── config.h                # Hyperparameters
│   ├── game.h                  # Go engine (ring buffer history, fast is_legal)
│   ├── loaded_model.h          # Shared CPU weights (ONNX parsed once)
│   ├── compute_context.h       # ComputeContext + ComputeHandle base classes
│   ├── batch_evaluator.h       # BatchEvaluator interface + NNResultBuf
│   ├── nn_request_queue.h      # Ring-buffer request queue (KataGo semantics)
│   ├── nn_evaluator.h          # KataGo-style batching server (N server threads)
│   ├── async_bot.h             # Persistent worker wrapper (ponder + analyze)
│   ├── katago_inputs.h         # KataGo V7 input encoder (22 spatial + 19 global)
│   ├── eigen_compute.h         # Eigen CPU backend (context + handle)
│   ├── opencl_compute.h        # OpenCL GPU backend (context + handle)
│   ├── cuda_compute.h          # CUDA GPU backend (context + handle)
│   ├── tensorrt_compute.h      # TensorRT GPU backend (context + handle)
│   ├── metal_compute.h         # Metal/MPSGraph GPU backend (macOS)
│   ├── rknn_compute.h          # Rockchip NPU backend (aarch64)
│   ├── vip9000_compute.h       # VeriSilicon NPU backend (aarch64)
│   ├── onnx_loader.h           # Built-in minimal ONNX protobuf parser
│   └── mcts.h                  # Multi-threaded MCTS (atomic MCTSNode)
├── src/                        # Implementations of the above + 4 binaries:
│   ├── main_play.cpp           # Human vs AI (uses AsyncBot for ponder/analyze)
│   ├── main_selfplay.cpp       # Multi-threaded data generation (V2 .bin)
│   ├── main_evaluate.cpp       # Model vs model evaluation matches (+SGF)
│   └── main_benchmark.cpp      # Performance tests
├── scripts/                    # The continuous pipeline + tooling
│   ├── run_continuous.py       # Supervisor: init / run / status
│   ├── train_continuous.py     # Continuous trainer (DDP, bucket, ring)
│   ├── selfplay_driver.py      # Selfplay loop + publish + throttle
│   ├── gatekeeper.py           # Candidate gating vs accepted/latest
│   ├── rate.py                 # Optional Elo rating loop
│   ├── model.py                # PyTorch model definition
│   ├── export_onnx.py          # PyTorch → ONNX export
│   └── visualize.py            # Selfplay/eval game viewer (.bin/.zst/.sgf)
└── tools/                      # KataGo conversion + NPU toolchain scripts
```

## CLI Reference

### run_continuous.py

```
python scripts/run_continuous.py init [options]
  --arch {resnet,vit}     Architecture (default: resnet)
  --board N               Board size (default: 9)
  --filters N --blocks N  ResNet size (default: 128/10)
  --d-model/--depth/--heads/--kv-groups/--mlp-ratio   ViT size
  --komi F                Komi for the whole run (default: 7.5)
  -y                      Skip confirmation (archives any previous run)
  → writes training/run_config.json + seeds models/accepted/v000000000.onnx

python scripts/run_continuous.py run [options]
  Architecture + komi come from run_config.json — do NOT re-specify.
  Per-worker GPUs:   --train-gpus/--selfplay-gpus/--gate-gpus/--rate-gpus
  Per-worker NN:     --<proc>-nn-device-ids (+ optional --<proc>-nn-server-threads)
  Selfplay:          --selfplay-batch-games --selfplay-sims --window-games
                     --score-weight-max --throttle-high --throttle-low
  Training:          --batch-size --base-lr --replay-target --ring-games
                     --export-every --value-ramp-steps --score-ramp-steps ...
  Gating:            --gate-games --gate-sims --gate-threshold
  Rating (opt-in):   --rating --rating-games --rating-interval ...
  Shutdown:          --graceful-timeout (Ctrl-C = graceful, 2nd = kill)

python scripts/run_continuous.py status    Show pool/models/step summary
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
  --score-scale F        Score atan compression scale (default: 18.0)
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
  --komi F               Komi value (default: 7.5)
  --score-weight F       Score utility weight (default: 0.0)
  --score-scale F        Score atan compression scale (default: 18.0)
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
  --score-scale F        Score atan compression scale (default: 18.0)
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
  --score-scale F        Score atan compression scale (default: 18.0)
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

**Training interrupted**: Just re-run `python scripts/run_continuous.py run ...` —
the trainer resumes from its checkpoint (step, optimizer, replay bucket) and the
other workers pick up from the filesystem.  Check
`python scripts/run_continuous.py status` or `logs/current/supervisor.log`.

**Selfplay logs THROTTLE_PAUSE and idles**: working as intended — the trainer's
replay bucket is full (training is the bottleneck).  It resumes automatically
when the trainer drains the bucket below `--throttle-low`.  If the trainer is
DEAD (check `logs/current/train.stdio.log`), fix that instead; pausing selfplay
while nothing trains is exactly what stops GPU waste.

**Trainer logs BUDGET_SLEEP**: the inverse — selfplay is the bottleneck and the
trainer has exhausted its replay budget (`--replay-target` × new rows).  Give
selfplay more GPU, more `--selfplay-threads`, or accept the pacing.

**Slow on macOS with OpenCL**: Rebuild with Metal backend:
`cmake .. -DMINIGO_BACKEND=metal && make -j$(sysctl -n hw.ncpu)`.
Metal with MPSGraph FP16 is 2-3× faster than OpenCL on Apple Silicon.
