# MiniGo C++ — Miniature KataGo AlphaZero

Miniature AlphaZero Go engine modelled on KataGo's architecture:
multi-threaded MCTS with per-leaf blocking evaluation + a KataGo-style
`NNEvaluator` server that batches leaf evaluations into one GPU call.
Training runs in Python/PyTorch. Works on **Linux** and **macOS** (Intel + Apple Silicon).

**Inference backends** (compile-time selectable, runtime switchable):
- **Metal** (default on macOS Apple Silicon) — GPU inference via MPSGraph with FP16 compute; 2-3× faster than OpenCL on the same hardware
- **OpenCL** (default on Linux) — GPU inference on any OpenCL 1.2+ device (NVIDIA, AMD, Intel); hand-written implicit GEMM kernels with fused BN/ReLU
- **Eigen** (always available) — CPU inference using Apple Accelerate / OpenBLAS; one engine per thread

All GPU backends use the same `NNEvaluator` server thread for batching across MCTS threads.
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

# For NVIDIA GPU (OpenCL):
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
-- Backend:    opencl     (Linux with GPU)
-- Backend:    eigen      (no GPU available)
```

Force a specific backend:
```bash
cmake .. -DMINIGO_BACKEND=metal    # Metal/MPSGraph (macOS Apple Silicon)
cmake .. -DMINIGO_BACKEND=opencl   # OpenCL (Linux, macOS)
cmake .. -DMINIGO_BACKEND=eigen    # CPU only (no GPU)
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
    --games 100 --threads 8 --search-threads 16 --sims 400

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
┌──────────────────────────────────────────────────────────────────┐
│                    Self-Play (C++)                                │
│                                                                  │
│  Worker 1 (game 1) ─┐  Each worker plays one game at a time.    │
│  Worker 2 (game 2) ─┤  Per move, MCTS::search() spawns N        │
│  Worker 3 (game 3) ─┤  search threads (KataGo pattern):         │
│         ...         ─┤                                           │
│  Worker T (game T) ─┘  Each search thread owns one NNResultBuf  │
│                        (pre-allocated, reused for all evals):   │
│                        descend → evaluate_with_buf(BLOCK) →     │
│                        expand → backprop → repeat               │
│                                     │                            │
│         ┌───────────────────────────▼──────────────────┐         │
│         │          NNEvaluator (server thread)          │         │
│         │  Shared queue of NNResultBuf* pointers.       │         │
│         │  Pops up to max_batch_size, fires one         │         │
│         │  predict_batch() GPU call, signals each       │         │
│         │  client's condvar. No timeout, no threshold.  │         │
│         └───────────────────────────┬──────────────────┘         │
│                                     │                            │
│                    ┌────────────────▼────────────┐               │
│                    │     InferenceEngine          │               │
│                    │  ┌────────┬────────┬───────┐ │               │
│                    │  │ Eigen  │ OpenCL │ Metal │ │               │
│                    │  │ (CPU)  │ (GPU)  │ (GPU) │ │               │
│                    │  └────────┴────────┴───────┘ │               │
│                    └─────────────────────────────┘               │
│                                     │                            │
│                        Self-Play Data (.bin)                      │
└─────────────────────────────────────┬────────────────────────────┘
                                      │
                                      ▼
                          ┌─── Python ───────────┐
                          │  train.py (PyTorch)  │
                          │  export_onnx.py      │──▶ model.onnx
                          └──────────────────────┘
```

**All backends use the same NNEvaluator architecture** — even Eigen (CPU).
The NNEvaluator server thread serializes all `predict_batch()` calls, so
the engine doesn't need to be thread-safe.  The backend is selected at
compile time via `cmake -DMINIGO_BACKEND=...`.

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

### OpenCL GPU Backend

`OpenCLEngine` (`src/opencl_engine.cpp`) implements the full AlphaZero forward
pass using hand-written OpenCL kernels:

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

`MetalEngine` (`src/metal_engine.mm`) uses **MPSGraph** (Metal Performance
Shaders Graph) to build the entire forward pass as a computation graph at
`load_model()` time.  Each `predict_batch()` call feeds inputs through the
pre-compiled graph via `graph.run()`.

- **FP16 compute**: weights and activations in half-precision with FP32
  accumulation.  Softmax and tanh run in FP32 for numerical stability.
- **Kernel fusion**: MPSGraph fuses conv+BN+ReLU automatically
- **`@autoreleasepool`**: wraps each `graph.run()` call, matching KataGo's
  Metal backend pattern for correct ObjC object lifecycle on server threads
- **Unified memory**: CPU and GPU share the same memory (no explicit copies)

### Modular Backend Design

The `InferenceEngine` interface (`include/inference_engine.h`) defines:
- `load_model(path)` — load a `.onnx` model
- `predict(state)` — single inference
- `predict_batch(states)` — batch inference

Adding a new backend (e.g., CUDA): implement `InferenceEngine`, add to the
factory in `inference_engine.cpp`, add CMake detection.  The NNEvaluator,
MCTS, game engine, and training pipeline are completely backend-agnostic.

`BatchEvaluator` (`include/batch_evaluator.h`) is what MCTS sees.
All backends use `NNEvaluator` — the KataGo-style server that batches
across threads.  There is no `DirectEvaluator`; the architecture is
fully unified.

## Performance

### Batch NN inference throughput (9×9, 64 filters, 5 blocks)

| Batch | Metal FP16 (M1 Max) | OpenCL (M1 Max) | OpenCL (RTX 5070 Ti) |
|---|---|---|---|
| 1 | **750/s** | 303/s | 1,100/s |
| 8 | **7,500/s** | 2,400/s | 8,700/s |
| 32 | **26,000/s** | 7,700/s | 28,300/s |
| 64 | **28,000/s** | 10,800/s | 41,300/s |
| 128 | **44,000/s** | 15,800/s | 55,200/s |

Metal FP16 is **2.8× faster** than OpenCL on the same Apple Silicon chip.

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

- The OpenCL backend parses ONNX protobuf to extract weights → uploads to GPU
- The Metal backend parses the same way → builds MPSGraph with FP16 weights
- The Eigen backend uses the same parser → stores weights as Eigen matrices

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
├── run_loop.sh                 # Automated training loop (multi-instance selfplay)
├── include/
│   ├── config.h                # Hyperparameters
│   ├── game.h                  # Go engine (ring buffer history, fast is_legal)
│   ├── inference_engine.h      # Abstract inference interface + factory
│   ├── batch_evaluator.h       # BatchEvaluator interface + NNResultBuf
│   ├── nn_evaluator.h          # KataGo-style batching server
│   ├── eigen_engine.h          # Eigen CPU inference
│   ├── opencl_engine.h         # OpenCL GPU inference
│   ├── metal_engine.h          # Metal/MPSGraph GPU inference (macOS)
│   ├── onnx_loader.h           # Built-in minimal ONNX protobuf parser
│   └── mcts.h                  # Multi-threaded MCTS (atomic MCTSNode)
├── src/
│   ├── game.cpp                # Full Go rules (captures, ko, scoring)
│   ├── onnx_loader.cpp         # ONNX weight parser (no external dependency)
│   ├── eigen_engine.cpp        # Eigen inference
│   ├── opencl_engine.cpp       # OpenCL inference + embedded kernels
│   ├── metal_engine.mm         # Metal/MPSGraph inference (Obj-C++)
│   ├── inference_engine.cpp    # Backend factory
│   ├── nn_evaluator.cpp        # NNEvaluator server thread (KataGo pattern)
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
```

## Platform Notes

### macOS (Apple Silicon) — recommended: `--backend metal`

- **Metal** (default): uses MPSGraph for GPU inference with FP16 compute.
  The computation graph is compiled once at model load time (~200ms).
  2.8× faster than OpenCL on the same chip.
- **OpenCL**: available and functional; slower than Metal.
- **Eigen**: uses Apple **Accelerate** for hardware-tuned BLAS.

### macOS (Intel)

- Metal not available (requires Apple Silicon).
- OpenCL 1.2 available via Intel HD/Iris GPU.
- Accelerate provides vecLib BLAS for Eigen.

### Linux — recommended: `--backend opencl`

- **OpenCL** (default): hand-written implicit GEMM kernels optimized for
  NVIDIA GPUs.  Available on NVIDIA (CUDA toolkit), AMD (ROCm/Mesa), Intel (NEO).
- Metal not available on Linux.
- For faster Eigen: `sudo apt install libopenblas-dev`

## Troubleshooting

**cmake can't find Eigen3**: `brew list eigen` or `dpkg -l libeigen3-dev`

**OpenCL not found**: On Linux install `ocl-icd-opencl-dev` + a GPU driver ICD;
on macOS OpenCL ships with Xcode Command Line Tools.

**Metal not found**: Requires macOS with Apple Silicon.  Check that
Xcode Command Line Tools are installed (`xcode-select --install`).

**Build without GPU**: `cmake .. -DENABLE_OPENCL=OFF -DENABLE_METAL=OFF` (Eigen-only)

**OpenCL kernel compile error**: shown in the exception message; usually means
the GPU doesn't support the feature used.  File a bug with the error text.

**Training interrupted**: Just restart `./run_loop.sh` — it resumes automatically.

**Slow on macOS with OpenCL**: Use `--backend metal` instead.  Metal with
MPSGraph FP16 is 2-3× faster than OpenCL on Apple Silicon.
