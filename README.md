# MiniGo C++ — Miniature KataGo AlphaZero

Miniature AlphaZero Go engine modelled on KataGo's architecture:
multi-threaded MCTS with per-leaf blocking evaluation + a KataGo-style
`NNEvaluator` server that batches leaf evaluations into one GPU call.
Training runs in Python/PyTorch. Works on **Linux** and **macOS** (Intel + Apple Silicon).

**Inference backends** (compile-time selectable):
- **CUDA** (default on Linux with NVIDIA GPU) — FP16 Tensor Core inference via WMMA; hand-written implicit GEMM kernels. Supports Turing, Ampere, Ada, Hopper, Blackwell
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

# For NVIDIA GPU (CUDA — recommended):
# Install CUDA toolkit (provides nvcc compiler + runtime)
# https://developer.nvidia.com/cuda-downloads

# For NVIDIA GPU (OpenCL — alternative):
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
-- Backend:    cuda       (Linux with NVIDIA GPU + CUDA toolkit)
-- Backend:    opencl     (Linux with GPU, no CUDA)
-- Backend:    eigen      (no GPU available)
```

Force a specific backend:
```bash
cmake .. -DMINIGO_BACKEND=cuda     # CUDA FP16 Tensor Cores (NVIDIA)
cmake .. -DMINIGO_BACKEND=metal    # Metal/MPSGraph (macOS Apple Silicon)
cmake .. -DMINIGO_BACKEND=opencl   # OpenCL (Linux, macOS)
cmake .. -DMINIGO_BACKEND=eigen    # CPU only (no GPU)
```

For CUDA, multi-arch is built by default (Turing/Ampere/Ada SASS + Hopper PTX
for Blackwell forward-compat). For faster local builds targeting one GPU:
```bash
cmake .. -DMINIGO_BACKEND=cuda -DCMAKE_CUDA_ARCHITECTURES=75   # Turing (RTX 2080 Ti)
cmake .. -DMINIGO_BACKEND=cuda -DCMAKE_CUDA_ARCHITECTURES=89   # Ada (RTX 4090)
```

The backend is selected at **compile time** — no `--backend` flag at runtime.
All binaries automatically use whichever backend was compiled.

## Quick Start

### Train (automated loop)

```bash
# Quick test (~5 min, 5x5 board, small net)
chmod +x run_loop.sh
./run_loop.sh --quick

# Full training (9x9 board, 16 search threads per move)
./run_loop.sh --iterations 30 --games 200 --sims 800

# Multi-GPU: 2 server threads across 2 GPUs
./run_loop.sh --iterations 30 --games 200 --nn-server-threads 2 --nn-device-ids 0,1

# 4 server threads on 2 GPUs (2 threads per GPU)
./run_loop.sh --iterations 30 --games 200 --nn-server-threads 4 --nn-device-ids 0,0,1,1

# Maximize GPU utilization with 2 parallel selfplay instances
./run_loop.sh --iterations 30 --games 200 --selfplay-instances 2

# CPU-only: rebuild with Eigen backend first
# cmake .. -DMINIGO_BACKEND=eigen && make -j$(nproc)
./run_loop.sh --iterations 30 --games 200
```

Training is **resumable** — stop at any time (Ctrl+C) and restart
`./run_loop.sh` to continue from the last checkpoint.  Self-play data
accumulates across iterations; the optimizer state is preserved.

### Train (manual steps)

```bash
# 1. Export initial (random) ONNX model
cd scripts
python3 export_onnx.py --init --board 9 --output ../model.onnx
cd ..

# 2. Generate self-play data (GPU)
./build/selfplay --model model.onnx \
    --games 100 --threads 8 --search-threads 16 --sims 400 \
    --nn-server-threads 1 --nn-device-ids 0

# 3. Train in Python (auto-exports updated model.onnx)
cd scripts
python3 train.py --data ../selfplay_data --epochs 20 --board 9
cd ..

# Repeat from step 2 with the updated model
```

### Play

```bash
# Against trained model (backend selected at compile time)
./build/play --model model.onnx --sims 800

# Against random bot (no model needed)
./build/play --random --board 9
```

### Benchmark

```bash
# Benchmark (backend selected at compile time)
./build/benchmark --model model.onnx \
    --games 10 --threads 10 --search-threads 8

# Generate larger models for GPU benchmarking
cd scripts
python3 export_onnx.py --init --filters 128 --blocks 10 --output ../model_large.onnx
cd ..
./build/benchmark --model model_large.onnx \
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

| Batch | CUDA FP16+WMMA (RTX 2080 Ti) | OpenCL FP32 (RTX 2080 Ti) | Metal FP16 (M1 Max) |
|------:|-----------------------------:|---------------------------:|--------------------:|
| 1     | 1,013                        | 934                        | 750                 |
| 8     | 7,908                        | 7,336                      | 7,500               |
| 32    | **27,060**                   | 19,886                     | 26,000              |
| 64    | **45,247**                   | 27,383                     | 28,000              |
| 128   | **56,452**                   | 34,005                     | 44,000              |

**Large model** (128 filters, 10 blocks):

| Batch | CUDA FP16+WMMA (RTX 2080 Ti) | OpenCL FP32 (RTX 2080 Ti) |
|------:|-----------------------------:|---------------------------:|
| 1     | 330                          | 324                        |
| 32    | **7,766**                    | 4,723                      |
| 64    | **9,160**                    | 5,712                      |
| 128   | **11,007**                   | 6,012                      |

CUDA FP16+WMMA is **1.66×** faster than OpenCL FP32 at batch-128 (small model)
and **1.83×** faster for the large model, where tensor core utilization is higher.

### Self-play throughput (800 sims/move)

| Config | RTX 5070 Ti (OpenCL) | RTX 2080 Ti (OpenCL) |
|---|---|---|
| 64 games, 8 threads, 32 search-threads | **2.17 s/game** | 3.13 s/game |

**Tuning guide**: increase `--threads` (more concurrent games) and
`--search-threads` (more threads per game's search) until GPU utilization
plateaus.  Batch size adapts naturally to the total concurrent search
threads: `total = min(games, threads) × search_threads`.

## Resumable Training

Training state is fully saved in checkpoints:
- Model weights + optimizer state (Adam momentum buffers)
- List of already-trained data files
- Iteration counter

On resume, `train.py` skips previously-trained data files and restores
the optimizer state, so training continues smoothly from where it left off.

Self-play data accumulates in per-iteration subdirectories
(`selfplay_data/iter_0001/`, etc.) and is not deleted between iterations.

## Model Format

All backends use **ONNX** (`.onnx`) as the universal model format.
The project includes a built-in minimal protobuf parser (`onnx_loader.cpp`)
— **no external protobuf library required**.

- `LoadedModel::load()` parses the ONNX protobuf once and pre-fuses BN parameters
- Each `ComputeHandle` uploads these CPU weights to its own GPU buffers

The model has dynamic batch dimensions, so the same `.onnx` file works for
batch sizes 1 through N.  GPU workspace buffers auto-grow to fit the batch.

Generate models of different sizes:
```bash
cd scripts
python3 export_onnx.py --init --board 9 --filters 64  --blocks 5  --output ../model.onnx        # small (default)
python3 export_onnx.py --init --board 9 --filters 128 --blocks 10 --output ../model_large.onnx   # large
python3 export_onnx.py --init --board 9 --filters 256 --blocks 20 --output ../model_xlarge.onnx  # extra-large
```

## Files

```
minigo-cpp/
├── CMakeLists.txt              # Build (Eigen required, OpenCL/Metal optional)
├── run_loop.sh                 # Automated training loop (multi-GPU selfplay)
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
│   ├── metal_compute.mm        # Metal/MPSGraph context + handle (Obj-C++)
│   ├── nn_evaluator.cpp        # NNEvaluator N server threads (KataGo pattern)
│   ├── mcts.cpp                # Multi-threaded MCTS + data augmentation
│   ├── main_play.cpp           # Human vs AI
│   ├── main_selfplay.cpp       # Multi-threaded data generation
│   └── main_benchmark.cpp      # Performance tests
└── scripts/
    ├── model.py                # PyTorch model definition
    ├── export_onnx.py          # PyTorch → ONNX export
    └── train.py                # Train on self-play data (resumable)
```

## CLI Reference

### run_loop.sh

```
./run_loop.sh [options]
  --iterations N         Training iterations (default: 30)
  --games N              Games per iteration (default: 100)
  --sims N               MCTS simulations per move (default: 400)
  --board N              Board size (default: 9)
  --epochs N             Training epochs per iteration (default: 15)
  --batch N              Training batch size (default: 256)
  --threads N            Total selfplay worker threads (default: all cores)
  --search-threads N     MCTS search threads per move (default: 16)
  --selfplay-instances N Parallel selfplay processes (default: 1)
  --nn-server-threads N  NN server threads (default: 1)
  --nn-device-ids IDS    Comma-separated GPU indices (default: "0")
  --quick                Fast test mode (5x5 board, small net)
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
  --komi F               Komi value (default: 7.5)
  --random               Use random bot (no model needed)
  --board N              Board size (for --random mode)
```

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

### Linux — recommended: CUDA (NVIDIA) or OpenCL

- **CUDA** (default on NVIDIA): FP16 Tensor Core inference via WMMA. Requires
  CUDA toolkit (nvcc). Auto-detected when `nvcc` is on PATH. 1.7-1.8× faster
  than OpenCL on the same GPU.
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

**Build without GPU**: `cmake .. -DMINIGO_BACKEND=eigen` (CPU-only)

**OpenCL kernel compile error**: shown in the exception message; usually means
the GPU doesn't support the feature used.  File a bug with the error text.

**Training interrupted**: Just restart `./run_loop.sh` — it resumes automatically.

**Slow on macOS with OpenCL**: Rebuild with Metal backend:
`cmake .. -DMINIGO_BACKEND=metal && make -j$(sysctl -n hw.ncpu)`.
Metal with MPSGraph FP16 is 2-3× faster than OpenCL on Apple Silicon.
