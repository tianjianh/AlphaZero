# MiniGo C++ — Miniature KataGo AlphaZero

A miniature AlphaZero Go engine modelled on KataGo's architecture:
multi-threaded MCTS with per-leaf blocking evaluation, a KataGo-style
`NNEvaluator` server that batches leaf evaluations into single GPU
calls, and a fully continuous KataGo-style training pipeline (selfplay,
training, gatekeeping, and rating as concurrent supervised processes).
The 7-headed network has 5 inference heads (policy, value W/L/D,
scoreMean, scoreStdev, ownership) and 2 training-only auxiliaries
(score belief, opponent policy).  Training runs in Python/PyTorch;
inference is C++.  Works on **Linux** and **macOS** (Intel + Apple
Silicon), plus two embedded NPU targets.

**Three trainable architectures**, one engine-neutral game-record pool:

| `--arch` | Input | Trunk |
|---|---|---|
| `resnet` | 17-plane board tensor | KataGo-style ResNet: alternating SE + global-pooling residual blocks, global-pooled value/score heads |
| `vit` | 17-plane board tensor | Vision Transformer: one token per intersection, GQA attention, directional relative position bias |
| `katago` | KataGo V7 dual input (22 spatial planes + 19 globals) | KataGo pre-activation trunk; converted stock `kata1` networks are drop-in for every binary |

**Inference backends** (compile-time selectable; `auto` never picks a stub):

| Backend | Hardware | Status |
|---|---|---|
| **TensorRT** | NVIDIA (Turing+) | Full — parses the ONNX natively, auto FP16, engine cached to disk |
| **OpenCL** | any OpenCL 1.2 GPU (NVIDIA / AMD / Intel) | Full, all three architectures — fp32, portable fp16, and NVIDIA tensor-core (inline-PTX `mma.sync`) precision tiers |
| **Metal** | macOS Apple Silicon | MPSGraph FP16 (legacy AlphaZero arch; current archs pending port) |
| **Eigen** | any CPU | resnet + converted-kata1 (debugging-grade speed) |
| **RKNN** | Rockchip NPU (RK356x/3576/3588, aarch64) | fp16 / int8 via pre-compiled `.rknn` ([conversion guide](docs/RKNN_CONVERSION.md)) |
| **VIP9000** | VeriSilicon NPU (Allwinner A733, aarch64) | fp16 / int8 via pre-compiled `.nb` ([conversion guide](docs/A733_CONVERSION.md)) |
| CUDA | NVIDIA | **stub** — throws at handle creation; kernels not ported to the current architectures (see [roadmap](docs/ROADMAP.md)) |

**Multi-GPU / multi-core**: N server threads, each owning a
`ComputeHandle` on its assigned device, all draining one shared queue —
whichever device finishes first picks up the next batch
(self-balancing).  On multi-core NPUs one server thread pins to each
core, giving the same topology as one-thread-per-GPU.

**Model format**: `.onnx` everywhere (built-in minimal protobuf parser
— no external protobuf dependency).  NPU backends additionally consume
their pre-compiled sibling artifacts (`.rknn` / `.nb`) while still
reading the ONNX for metadata.  Details: [docs/NETWORK.md](docs/NETWORK.md)
and [docs/FORMATS.md](docs/FORMATS.md).

## Documentation map

This README is the guide: install, build, verify, train, play, CLI
reference, troubleshooting.  Deep-dive material lives in `docs/`:

| Doc | Contents |
|---|---|
| [docs/CONTINUOUS_TRAINING.md](docs/CONTINUOUS_TRAINING.md) | Full training-pipeline design: workers, train bucket, throttle, ring buffer, cold start, ops playbook, every CLI knob |
| [docs/TRAINING_STRATEGY.md](docs/TRAINING_STRATEGY.md) | Tuning rationale and postmortems from real runs (score-head feedback loop, ring staleness, ...) |
| [docs/NETWORK.md](docs/NETWORK.md) | The 7-headed network, loss functions/weights, MCTS utility formula, ONNX model contract |
| [docs/FORMATS.md](docs/FORMATS.md) | Support matrix: model architectures × file formats × backends; V3 game-record format |
| [docs/ENGINE.md](docs/ENGINE.md) | Engine internals: Go board engine, multi-threaded MCTS, AsyncBot, NNEvaluator batching server |
| [docs/BACKENDS.md](docs/BACKENDS.md) | Per-backend implementation details, Context/Handle lifecycle, measured performance, platform notes |
| [docs/KATAGO_INFERENCE.md](docs/KATAGO_INFERENCE.md) | Running stock KataGo networks: conversion, parity validation, caveats |
| [docs/COMPARISON_WITH_KATAGO.md](docs/COMPARISON_WITH_KATAGO.md) | Line-by-line engine audit against upstream KataGo semantics |
| [docs/RKNN_CONVERSION.md](docs/RKNN_CONVERSION.md) | Rockchip NPU: ONNX → `.rknn` conversion, quantisation, on-board bring-up |
| [docs/A733_CONVERSION.md](docs/A733_CONVERSION.md) | Allwinner A733 / VIP9000 NPU: ONNX → `.nb` conversion, int8 calibration, bring-up |
| [docs/ROADMAP.md](docs/ROADMAP.md) | Open work items and known limitations |

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
# see docs/A733_CONVERSION.md §3.2 for the retrieval recipe):
bash tools/onnx_to_a733_docker.sh 1   # produces models/<...>.a733.bs1.fp16/network_binary.nb
bash tools/onnx_to_a733_docker.sh 4   # produces models/<...>.a733.bs4.fp16/network_binary.nb
```
Conversion must happen on x86_64; aarch64 has no Acuity binaries.  See
[docs/A733_CONVERSION.md](docs/A733_CONVERSION.md) for the full ONNX → `.nb` flow,
and the verification checklist in [docs/A733_CONVERSION.md §7](docs/A733_CONVERSION.md)
for the chip-ID gotchas to spot-check after each re-conversion.

### Python (both platforms — only needed for training)

```bash
pip install -r scripts/requirements.txt   # torch, numpy, zstandard, onnx

# Optional: FP8 training on Blackwell+ GPUs (SM 10.0)
pip install transformer_engine
```

Python is NOT required for inference — only for training (`train_continuous.py`)
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
| Acuity Toolkit v6.30.22 | ONNX → .nb conversion (x86_64 Linux + Docker only) | Allwinner's `ubuntu-npu:v2.0.10.1` Docker image (Synology netdisk — see docs/A733_CONVERSION.md §3.2). pip `acuitylite` does NOT work — chip table missing A733 PID. |
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
-- Backend:    opencl     (Linux with a GPU, no TensorRT — incl. NVIDIA)
-- Backend:    rknn       (aarch64 Linux with librknnrt.so — Rockchip NPU)
-- Backend:    vip9000    (aarch64 Linux with viplite-tina SDK — Allwinner A733 / VIP9000)
-- Backend:    eigen      (no GPU available)

# `auto` never selects the CUDA backend — its kernels are stubs for the
# current architectures (docs/ROADMAP.md); pass -DMINIGO_BACKEND=cuda
# explicitly only if you are developing it.
```

Force a specific backend:
```bash
cmake .. -DMINIGO_BACKEND=tensorrt # TensorRT (NVIDIA, fastest — requires libnvinfer-dev)
cmake .. -DMINIGO_BACKEND=cuda     # CUDA (STUB for current archs — developers only)
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

See **[docs/CONTINUOUS_TRAINING.md](docs/CONTINUOUS_TRAINING.md)** for the full
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
# Selfplay game (binary format, supports .bin / .bin.zst)
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

See [docs/KATAGO_INFERENCE.md](docs/KATAGO_INFERENCE.md) for encoder fidelity notes (ladder
planes are fully computed via a port of upstream's ladder search;
encore-only signals are zeroed) and [docs/FORMATS.md](docs/FORMATS.md) for the full
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

# 4.  Build with a backend that runs the dual-input format natively
#     (tensorrt and opencl both do; eigen runs it on CPU).
cmake -B build -DMINIGO_BACKEND=tensorrt   # or -DMINIGO_BACKEND=opencl
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
# 1. Convert the kata1 weights once on an x86_64 host (see docs/A733_CONVERSION.md).
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

## Model formats & network

Full details: [docs/NETWORK.md](docs/NETWORK.md) (heads, losses, MCTS
utility) and [docs/FORMATS.md](docs/FORMATS.md) (support matrix, V3
record format).  The short version:

- All three architectures train under one 7-head loss and export the
  same 5-output inference ONNX (`policy_logits, value, score_mean,
  score_stdev, ownership`).
- Selfplay records (V3) store moves + targets, not encoded states —
  one record pool serves every architecture, and any model format
  (including converted kata1 nets) can generate training data.
- The C++ loader auto-detects the format from the ONNX graph inputs;
  BN is pre-fused at load; the `_sd_`-prefixed embedded state_dict
  carries raw weights for the kernel backends.

## Architecture & performance

Engine internals (MCTS, AsyncBot, NNEvaluator) are documented in
[docs/ENGINE.md](docs/ENGINE.md); backend implementations, the
Context/Handle lifecycle, and full benchmark tables in
[docs/BACKENDS.md](docs/BACKENDS.md).  Headline numbers:

- **OpenCL on NVIDIA A40** (9x9, batch 256, evals/s): resnet b10c128
  **25.4k** fp16+tensor-core vs 11.1k fp32; katago b10c128 24.1k;
  vit d192x8 7.0k.  Numerically verified against PyTorch on all five
  model-format variants (`build/verify`, 15/15 tiers passing).
- **TensorRT on RTX 2080 Ti** (9x9, batch 128): 175k states/s small
  model, 83k large model.
- **RK3576 NPU**: ~543 states/s per core fp16 (b10c128); ~2x with int8
  hybrid quantisation; A733 VIP9000 int8 ~110x faster than its fp16.

Environment knobs: `MINIGO_OPENCL_PRECISION=fp32|fp16|fp16-portable|auto`,
`MINIGO_OPENCL_PROFILE=1` (per-kernel GPU times),
`MINIGO_TRT_CACHE=<dir>` (shared engine cache),
`VIP9000_FORCE_PRECISION=fp16|int8`, `MINIGO_BACKEND=<backend>`
(build-time selection for `run_continuous.py`'s auto-build).

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
├── docs/                       # Deep-dive documentation (see map above)
├── src/                        # Implementations of the above + 5 binaries:
│   ├── main_play.cpp           # Human vs AI (uses AsyncBot for ponder/analyze)
│   ├── main_selfplay.cpp       # Multi-threaded data generation (V3 records)
│   ├── main_evaluate.cpp       # Model vs model evaluation matches (+SGF)
│   ├── main_benchmark.cpp      # Performance tests
│   └── main_verify.cpp         # Backend numerical verification vs PyTorch
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
  --sims N               MCTS simulations per move (default: 600)
  --c-puct F             UCB exploration constant (default: 1.25)
  --dirichlet-alpha F    Root noise concentration (default: 0.15)
  --dirichlet-epsilon F  Root noise weight (default: 0.22)
  --temp-threshold N     Moves of stochastic play (default: 12)
  --score-scale F        Score atan compression scale (default: 18.0)
  --nn-server-threads N  NN server threads (default: 1)
  --nn-device-ids IDS    Comma-separated GPU indices (default: "0")
```

### play

```
./build/play [options]
  --model PATH           Model file (default: models/best.onnx)
  --sims N               MCTS simulations per move (default: 600)
  --search-threads N     MCTS search threads (default: 16)
  --max-batch N          Max GPU batch size (default: 256)
  --c-puct F             UCB exploration constant (default: 1.25)
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
  --sims N               MCTS simulations per move (default: 600)
  --max-batch N          Max GPU batch size (default: 256)
  --threshold FLOAT      Win rate to pass (default: 0.5)
  --c-puct F             UCB exploration constant (default: 1.25)
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

### verify

Numerical verification of the compiled backend against PyTorch
reference vectors, plus a raw inference micro-benchmark:

```
# One-time: build reference models + vectors for all five formats
python scripts/make_test_vectors.py build/test_vectors

./build/verify --model build/test_vectors/resnet.onnx \
               --vectors build/test_vectors/resnet.vec        # PASS/FAIL + max diffs
./build/verify --model some_model.onnx --bench                # evals/s at B=1..256
  --tol-pol/--tol-val/--tol-score/--tol-own   per-field tolerances
  --max-batch N                               bench batch cap (default: 256)
```

Exit code 0 = within tolerance.  For the OpenCL backend, combine with
`MINIGO_OPENCL_PRECISION=fp32|fp16|fp16-portable` to test each tier.

Related: `tools/encoder_parity_test.py` proves the C++ and Python
KataGo-V7 encoders (incl. the ladder solver) bit-identical — run it
after touching `src/katago_inputs.cpp`, `src/game.cpp`, or
`scripts/gamedata.py`:

```bash
make -C build encode_dump
python3 tools/encoder_parity_test.py --records 'training/selfplay/g_*.bin.zst'
```

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

## Roadmap & known limitations

Open items (CUDA/Metal kernel ports, FP8-on-Blackwell, positional
superko, board-adaptive score scale, ...) are tracked in
[docs/ROADMAP.md](docs/ROADMAP.md).
