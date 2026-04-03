# MiniGo C++ — Miniature KataGo AlphaZero

Miniature AlphaZero Go engine modelled on KataGo's architecture:
multi-threaded MCTS with per-leaf blocking evaluation + a KataGo-style
`NNEvaluator` server that batches leaf evaluations into one GPU call.
Training runs in Python/PyTorch. Works on **Linux** and **macOS** (Intel + Apple Silicon).

**Inference backends** (compile-time selectable):
- **TensorRT** (default on Linux with NVIDIA GPU + TensorRT installed) — optimized inference via NVIDIA TensorRT; automatic FP16, layer fusion, and kernel auto-tuning. Engine cached to disk after first build
- **CUDA** (default on Linux with NVIDIA GPU, no TensorRT) — FP16 Tensor Core inference via WMMA; hand-written implicit GEMM kernels. Supports Turing, Ampere, Ada, Hopper, Blackwell
- **Metal** (default on macOS Apple Silicon) — GPU inference via MPSGraph with FP16 compute; 2-3× faster than OpenCL on the same hardware
- **OpenCL** — GPU inference on any OpenCL 1.2+ device (NVIDIA, AMD, Intel); hand-written implicit GEMM kernels with fused BN/ReLU
- **Eigen** (always available) — CPU inference using Apple Accelerate / OpenBLAS

**Multi-GPU support**: KataGo-style architecture with N server threads, each owning
a `ComputeHandle` on its assigned GPU.  All threads drain from a single shared queue
— whichever GPU finishes first picks up the next batch (self-balancing).

Each search thread pre-allocates one `NNResultBuf` (mutex + condvar), matching KataGo's pattern.

**Model format:** `.onnx` (universal — loaded by all backends via a built-in minimal protobuf parser, no external protobuf dependency)

## Prerequisites

### macOS (Apple Silicon — recommended)

```bash
xcode-select --install      # provides Metal frameworks + OpenCL
brew install cmake eigen
```

Metal (MPSGraph) is auto-detected.  No additional GPU libraries needed.

### macOS (Intel)

```bash
xcode-select --install      # provides OpenCL
brew install cmake eigen
```

Metal is not available on Intel Macs; OpenCL or Eigen is used.

### Linux (Ubuntu / Debian)

```bash
sudo apt install cmake g++ libeigen3-dev

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

# For NVIDIA GPU (OpenCL — portable alternative):
sudo apt install ocl-icd-opencl-dev
# + CUDA toolkit (provides the NVIDIA OpenCL ICD)

# For AMD GPU:    sudo apt install mesa-opencl-icd
# For Intel GPU:  sudo apt install intel-opencl-icd

# Optional: faster Eigen with OpenBLAS
sudo apt install libopenblas-dev
```

### Python (both platforms — only needed for training)

```bash
pip install torch numpy onnx
```

Python is NOT required for inference — only for training (`train.py`)
and model export (`export_onnx.py`).

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
-- Backend:    opencl     (Linux with GPU, no CUDA)
-- Backend:    eigen      (no GPU available)
```

Force a specific backend:
```bash
cmake .. -DMINIGO_BACKEND=tensorrt # TensorRT (NVIDIA, fastest — requires libnvinfer-dev)
cmake .. -DMINIGO_BACKEND=cuda     # CUDA FP16 Tensor Cores (NVIDIA)
cmake .. -DMINIGO_BACKEND=metal    # Metal/MPSGraph (macOS Apple Silicon)
cmake .. -DMINIGO_BACKEND=opencl   # OpenCL (Linux, macOS)
cmake .. -DMINIGO_BACKEND=eigen    # CPU only (no GPU)
```

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
# 1. Initialize — pick a preset or custom architecture
./run_loop.sh init small       # 9x9, 64f/5b,  60 iters (~2-4 hours)
./run_loop.sh init large       # 9x9, 128f/10b, 200 iters (~12-24 hours)
./run_loop.sh init quick       # 5x5, 32f/3b,  5 iters (pipeline test)
./run_loop.sh init --board 9 --filters 96 --blocks 8   # custom arch

# 2. Train — GPUs are auto-detected, just run:
./run_loop.sh train

# Or with explicit hardware settings:
./run_loop.sh train --threads 64 --nn-device-ids 0,0,1,1 --max-batch 512

# Run a limited number of iterations then pause:
./run_loop.sh train --iterations 20

# 3. Check progress:
./run_loop.sh status
```

**Resumable** — stop at any time (Ctrl+C) and re-run `./run_loop.sh train`
to continue from where it left off.  Pipeline state, selfplay data,
checkpoints, and training logs are all preserved.

#### Training pipeline

Each iteration runs three phases:

1. **Self-play**: generate games with the current best model (C++, multi-GPU)
2. **Train**: train on a sliding window of recent data (Python/PyTorch, multi-GPU DataParallel)
3. **Evaluate & gate**: play games between candidate and best model; promote
   if candidate wins ≥ 55% (configurable)

The pipeline is controlled by a **training plan** (`training_plan` file)
generated during `init`.  The plan defines staged training with escalating
parameters:

```
  Stage             Iters   Games   Sims  Epoch      LR   Gate
  ────────────────────────────────────────────────────────────
  Warm up           1-5     200    400     15    2e-3    off
  Explore           6-15    400    600     15    1e-3    50g
  Strengthen       16-35    400    600     20    5e-4   100g
  Polish           36-60    500    800     20    1e-4   100g
```

Each stage defines: selfplay games per iteration, MCTS simulations per move,
training epochs, learning rate, and evaluation games for gating.  Early stages
use fewer sims and no gating for fast exploration; later stages increase data
quality and enable gating to ensure only stronger models are promoted.

The plan is a plain text file — edit it to customize the schedule.

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

#### GPU auto-detection

The `train` command auto-detects NVIDIA GPUs and configures pipelining
(2 NN server threads per GPU):
- 1 GPU → `--nn-server-threads 2 --nn-device-ids 0,0`
- 2 GPUs → `--nn-server-threads 4 --nn-device-ids 0,0,1,1`

Override with explicit flags if needed.  PyTorch training also uses
`DataParallel` automatically when multiple GPUs are available.

#### Evaluation binary

The `evaluate` binary plays match games between two models to determine
which is stronger:

```bash
./build/evaluate --model1 candidate.onnx --model2 baseline.onnx \
    --games 100 --sims 400 --threshold 0.55
# Exit code 0 = model1 wins (above threshold)
# Exit code 1 = model1 fails
```

Each model gets its own NNEvaluator with separate compute contexts.
Games alternate which model plays Black.  Temperature is 0 (deterministic)
with no Dirichlet noise for clean evaluation.

### Play

```bash
# Against trained model (backend selected at compile time)
./build/play --model models/best.onnx --sims 800

# Multi-GPU with larger batch
./build/play --model models/best.onnx --sims 800 \
    --max-batch 512 --nn-server-threads 2 --nn-device-ids 0,0

# Against random bot (no model needed)
./build/play --random --board 9
```

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
          │  export_onnx.py      │──▶ model.onnx
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
FP16 Tensor Cores (WMMA) for maximum throughput on NVIDIA GPUs:

| Kernel | Purpose |
|---|---|
| `transpose_nchw_fp32_to_fp16` | Fused transpose + FP32→FP16 conversion |
| `conv3x3_wmma_bn` | **WMMA Tensor Core** implicit GEMM: fused im2col + 16×16×16 MMA + BN/residual/ReLU |
| `conv1x1_bn_relu_reshape_fp16` | FP16 1×1 conv + BN + ReLU + layout reshape |
| `fc_bias_relu_fp16` | FP16 FC + bias + ReLU |
| `fc_bias_softmax_fp16_to_fp32` | FP16→FP32 FC + softmax (policy head) |
| `fc_bias_tanh_fp16_to_fp32` | FP16→FP32 FC + tanh (value head) |

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

## Performance

### Batch NN inference throughput (9×9, states/s)

**Small model** (64 filters, 5 blocks):

| Batch | TensorRT FP16 (RTX 2080 Ti) | CUDA FP16+WMMA (RTX 2080 Ti) | OpenCL FP32 (RTX 2080 Ti) | Metal FP16 (M1 Max) |
|------:|----------------------------:|-----------------------------:|---------------------------:|--------------------:|
| 1     | **3,609**                   | 1,250                        | 934                        | 750                 |
| 8     | **26,039**                  | 9,730                        | 7,336                      | 7,500               |
| 32    | **85,254**                  | 31,466                       | 19,886                     | 26,000              |
| 64    | **136,814**                 | 50,561                       | 27,383                     | 28,000              |
| 128   | **175,102**                 | 59,933                       | 34,005                     | 44,000              |

**Large model** (128 filters, 10 blocks):

| Batch | TensorRT FP16 (RTX 2080 Ti) | CUDA FP16+WMMA (RTX 2080 Ti) | OpenCL FP32 (RTX 2080 Ti) |
|------:|----------------------------:|-----------------------------:|---------------------------:|
| 1     | **1,107**                   | 354                          | 324                        |
| 32    | **31,965**                  | 8,145                        | 4,723                      |
| 64    | **56,726**                  | 9,347                        | 5,712                      |
| 128   | **82,781**                  | 10,821                       | 6,012                      |

TensorRT is **2.9×** faster than CUDA WMMA at batch-128 (small model) and
**7.6×** faster for the large model, where TensorRT's layer fusion and kernel
auto-tuning dominate.  CUDA WMMA is **1.8×** faster than OpenCL FP32.

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
`./run_loop.sh train` after any interruption to continue:

- **Pipeline state** (`pipeline_state`): tracks current iteration, best model
  version, total games played, and promotion count
- **Selfplay resume**: skips iterations that already have enough game files
- **Training resume**: skips iterations whose versioned ONNX + checkpoint exist
- **Checkpoints** (`checkpoints/training.pt`): model weights + Adam optimizer
  state (momentum buffers) for smooth continuation
- **Selfplay data**: accumulates in per-iteration directories
  (`selfplay_data/iter_0001/`, etc.) and is never deleted

Training uses a **sliding window** — only data from the last N iterations
is loaded (configurable via `PLAN_WINDOW_SIZE` in the training plan),
keeping training focused on recent, stronger games and bounding memory usage.

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
├── run_loop.sh                 # Training pipeline (init/train/status)
├── training_plan               # Generated training schedule (editable)
├── models/                     # ONNX model files
│   ├── best.onnx               #   Current best (used for selfplay)
│   └── v0001.onnx ...          #   Version snapshots
├── trt_cache/                  # TensorRT compiled engine cache
├── training/                   # All training artifacts
│   ├── selfplay/               #   Game data (iter_0001/, iter_0002/, ...)
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
│   ├── mcts.cpp                # Multi-threaded MCTS + data augmentation
│   ├── main_play.cpp           # Human vs AI
│   ├── main_selfplay.cpp       # Multi-threaded data generation
│   ├── main_evaluate.cpp       # Model vs model evaluation matches
│   └── main_benchmark.cpp      # Performance tests
└── scripts/
    ├── model.py                # PyTorch model definition
    ├── export_onnx.py          # PyTorch → ONNX export
    └── train.py                # Train on self-play data (resumable)
```

## CLI Reference

### run_loop.sh

```
./run_loop.sh init <preset>      Initialize training (clears previous state)
  Presets: quick, small, large
  Custom:  init --board 9 --filters 96 --blocks 8

./run_loop.sh train [options]    Start or resume training
  --threads N             Worker threads (default: all cores)
  --search-threads N      MCTS search threads per move (default: 16)
  --selfplay-instances N  Parallel selfplay processes (default: 1)
  --nn-server-threads N   NN server threads (default: auto-detect)
  --nn-device-ids IDS     GPU indices, comma-sep (default: auto-detect)
  --max-batch N           Max GPU batch size for NN server (default: 256)
  --iterations N          Max iterations this session (default: all)

./run_loop.sh status             Show training progress
```

### selfplay

```
./build/selfplay [options]
  --model PATH           Model file (default: model.onnx)
  --games N              Number of games (default: 100)
  --threads N            Parallel workers (default: 1)
  --search-threads N     MCTS search threads per move (default: 16)
  --max-batch N          Max GPU batch size (default: 256)
  --output DIR           Output directory (default: selfplay_data)
  --sims N               MCTS simulations per move (default: 800)
  --nn-server-threads N  NN server threads (default: 1)
  --nn-device-ids IDS    Comma-separated GPU indices (default: "0")
```

### play

```
./build/play [options]
  --model PATH           Model file (default: model.onnx)
  --sims N               MCTS simulations per move (default: 800)
  --search-threads N     MCTS search threads (default: 16)
  --max-batch N          Max GPU batch size (default: 256)
  --komi F               Komi value (default: 7.5)
  --nn-server-threads N  NN server threads (default: 1)
  --nn-device-ids IDS    GPU indices (default: "0")
  --random               Use random bot (no model needed)
  --board N              Board size (for --random mode)
```

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
  --nn-server-threads N  NN server threads per model (default: 1)
  --nn-device-ids IDS    GPU indices (default: "0")
```

Exit code 0 = model1 wins (above threshold), 1 = model1 fails.

### benchmark

```
./build/benchmark [options]
  --model PATH           Model file (default: model.onnx)
  --sims N               MCTS simulations
  --nn-iters N           NN inference iterations (default: 1000)
  --games N              Self-play games (default: 5)
  --threads N            Self-play worker threads (default: 1)
  --search-threads N     MCTS search threads per move (default: 16)
  --max-batch N          Max GPU batch size (default: 256)
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

**Training interrupted**: Just re-run `./run_loop.sh train` — it resumes automatically
from the last completed iteration.  Check `./run_loop.sh status` to see progress.

**Slow on macOS with OpenCL**: Rebuild with Metal backend:
`cmake .. -DMINIGO_BACKEND=metal && make -j$(sysctl -n hw.ncpu)`.
Metal with MPSGraph FP16 is 2-3× faster than OpenCL on Apple Silicon.
