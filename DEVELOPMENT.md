# MiniGo C++ — Development Progress & Plan

## Branch Structure

```
main          — initial commit (original single-GPU, monolithic engine)
single-gpu    — same as main (preserved as stable baseline)
multi-gpu     — active development branch (KataGo-style multi-server architecture)
```

## What Has Been Done (Across All Sessions)

### Session 1: OpenCL Kernel Optimization
- **Implicit GEMM** (`conv3x3_sgemm_bn`): replaced separate im2col + SGEMM with a single kernel that computes im2col on-the-fly during B-tile loading. Eliminates 24MB scratch buffer.
- **Register blocking**: WPT_M=2, WPT_N=4, TS=16 → each work-item computes 8 outputs
- **Fused kernels**: BN + residual add + ReLU fused into SGEMM epilogue
- **Fused head kernels**: conv1x1_bn_relu_reshape, fc_bias_relu, fc_bias_softmax, fc_bias_tanh
- **GPU-side transpose**: NCHW → channel-major on GPU instead of CPU
- **Fixed global work size bug**: was launching 8× too many workgroups

### Session 2: Game Engine Optimization
- **Fast `is_legal()`**: empty-neighbor fast path skips BFS for ~80-90% of moves
- **Ring buffer history**: O(1) update_history() replacing O(n) vector erase
- **Flat vector MCTSNode children**: replaced unordered_map with vector indexed by action

### Session 3: KataGo-style MCTS Architecture
- **Multi-threaded MCTS**: `search()` spawns N search threads (KataGo pattern)
- **VLP=1 per thread**: each thread descends → evaluate_single → expand → backprop → repeat
- **Collision handling**: revert virtual losses, yield(), retry from root (KataGo's revertVirtualLosses)
- **Atomic MCTSNode**: std::atomic<int> for visit_count/virtual_loss_count, CAS loop for float value
- **Three-state expansion**: UNEVALUATED → EXPANDING → EXPANDED via CAS
- **Pre-allocated NNResultBuf per search thread**: fixes macOS ARM pthread mutex corruption

### Session 4: NNEvaluator (KataGo Pattern)
- **Shared queue**: NNResultBuf* pointers, competing consumers
- **No timeout, no threshold**: greedy drain, fire immediately (matches KataGo waitPopUpToN)
- **Per-leaf condvar**: each NNResultBuf has its own mutex+condvar
- **evaluate_with_buf()**: search thread reuses buf across all evaluations

### Session 5: Metal Backend (MPSGraph)
- **MPSGraph**: builds entire forward pass as computation graph at load_model() time
- **FP16 compute**: weights/activations in half-precision, softmax/tanh in FP32
- **@autoreleasepool**: wraps each graph.run() call (KataGo Metal pattern)
- **Performance**: batch-128 at 44K states/s on M1 Max (2.8× faster than OpenCL)

### Session 6: Unified Architecture
- **Removed DirectEvaluator**: all backends use NNEvaluator
- **Single-backend CMake**: `cmake -DMINIGO_BACKEND=auto|metal|opencl|eigen`
- **Removed --backend flag**: backend is compile-time only
- **Fixed GPU buffer overflows**: buf_pol_out_ and buf_val_h1_ sizes were swapped

### Session 7: Multi-GPU Refactor (Part 1)
- **LoadedModel**: extracted ONNX parsing + BN pre-fusion into shared CPU-side class
- **ComputeContext / ComputeHandle**: base classes matching KataGo's architecture
- **EigenComputeContext/Handle**: working, tested with 2 server threads
- **MetalComputeContext/Handle**: working, tested with 2 server threads
- **NNEvaluator refactored**: takes LoadedModel + ComputeContext + gpu_ids, spawns N server threads
- **ComputeHandle created ON server thread**: matches KataGo pattern exactly
- **New CLI**: `--nn-server-threads N`, `--nn-device-ids 0,1`

### Session 8: OpenCL Refactor + Cleanup
- **OpenCLComputeContext**: per-GPU device state (cl_context + cl_queue + cl_program), device selection by index
- **OpenCLComputeHandle**: uploads weights from LoadedModel CPU data, per-thread kernel handles + workspace
- **Kernel handles per-thread**: each ComputeHandle creates its own cl_kernel objects from the shared cl_program
- **Old engine files deleted**: inference_engine.h/.cpp, opencl_engine.h/.cpp, metal_engine.h/.mm, eigen_engine.h/.cpp
- **run_loop.sh updated**: `--nn-server-threads N`, `--nn-device-ids 0,1` flags forwarded to selfplay
- **Both Metal and OpenCL builds verified**: compile clean on macOS

### Session 9: CUDA Backend with FP16 Tensor Cores
- **CUDAComputeContext / CUDAComputeHandle**: same architecture as OpenCL/Metal — context per unique GPU, handle per server thread
- **Opaque Impl pattern**: header (`cuda_compute.h`) has no CUDA includes — all CUDA types live in the `.cu` file, so `compute_context.cpp` compiles as plain C++
- **FP16 weights & activations**: all intermediate buffers in half precision, halving memory bandwidth
- **WMMA Tensor Core GEMM** (`conv3x3_wmma_bn`): 16×16×16 `nvcuda::wmma` fragments with FP32 accumulator, implicit im2col on-the-fly. Block: 256 threads = 8 warps (2×4 layout), tile: 32×64 output
- **FP32 BN scale/bias and FC bias**: small per-channel params stored as float (direct upload, no GPU conversion), eliminates `__half2float` in every kernel's hot path
- **Fused transpose + FP32→FP16**: single kernel converts host input and transposes to channel-major FP16 layout
- **FP32 softmax/tanh output**: final head outputs computed in full precision for numerical stability
- **Multi-architecture CMake**: default builds SASS for SM 75 (Turing), 80/86 (Ampere), 89 (Ada) + compute_90 PTX (forward-compat with Hopper/Blackwell). Override with `-DCMAKE_CUDA_ARCHITECTURES=75`
- **Auto-detection**: `cmake -DMINIGO_BACKEND=auto` tries CUDA first on Linux (checks for nvcc), falls back to OpenCL
- **Performance**: 1.66× faster than OpenCL FP32 at batch-128 (small model), 1.83× for large model on RTX 2080 Ti

## What Remains (multi-gpu branch)

### ~~Priority 1: OpenCL ComputeContext/Handle Refactor~~ ✓ DONE

### ~~Priority 2: Clean Up Old Files~~ ✓ DONE

### ~~Priority 3: run_loop.sh Updates~~ ✓ DONE

### Priority 4: test_multi_gpu.sh

Create a test script for the remote Linux machine with 2× RTX 5070 Ti:
- Test 1: 1 server, GPU 0 (baseline)
- Test 2: 2 servers, same GPU 0,0 (should be ~same as baseline)
- Test 3: 2 servers, 2 GPUs 0,1 (should be ~1.8× baseline)
- Test 4: 4 servers, 2 GPUs 0,0,1,1
- Test 5: Large model comparison
- Test 6: 10-run stability test

### Priority 5: README Update

- Multi-GPU section with examples
- Updated architecture diagram showing LoadedModel → ComputeContext → ComputeHandle
- Updated CLI reference for all binaries
- Updated file listing

## Architecture Diagram (New)

```
                LoadedModel (1, shared, main thread)
                    │ CPU weights
                    ▼
              ComputeContext (1, shared, main thread)
              ┌─────┴──────────────────┐
              │                        │
         DeviceState[GPU 0]      DeviceState[GPU 1]
         (CUDA: cudaStream)       (CUDA: cudaStream)
         (OpenCL: cl_context+     (OpenCL: cl_context+
          cl_queue+cl_program)     cl_queue+cl_program)
              │                        │
     ┌────────┼────────┐      ┌────────┼────────┐
     │        │        │      │        │        │
  Handle_0  Handle_1        Handle_2  Handle_3
  (thread0) (thread1)       (thread2) (thread3)
  own bufs  own bufs        own bufs  own bufs
     │        │               │        │
     └────────┴───────────────┴────────┘
                      │
              NNEvaluator (1 instance)
              ┌───────────────────┐
              │  Shared Queue     │ ← search threads push NNResultBuf*
              │  (competing       │
              │   consumers)      │ → 4 server threads pop and process
              └───────────────────┘
```

## Key Design Decisions

1. **LoadedModel is shared (const)**: Parsed once, CPU-side. Each ComputeHandle reads from it to upload weights to GPU. No copies of the ONNX file per thread.

2. **ComputeContext holds per-device state**: One cl_context per unique GPU (KataGo pattern to avoid NVIDIA serialization). Two threads on the same GPU share the same context/queue.

3. **ComputeHandle created ON server thread**: Matches KataGo's pattern. GPU buffer allocation happens on the thread that will use them.

4. **NNResultBuf pre-allocated per search thread**: One mutex+condvar per thread lifetime, not per evaluation call. Fixes macOS ARM pthread corruption.

5. **Single shared queue**: All server threads drain from the same queue. Self-balancing — whichever GPU finishes first picks up the next batch.

## File Map (New Architecture)

```
include/
  loaded_model.h          NEW — shared CPU weights
  compute_context.h       NEW — ComputeContext + ComputeHandle base classes
  eigen_compute.h         NEW — Eigen backend (context + handle)
  opencl_compute.h        NEW — OpenCL backend (context + handle)
  cuda_compute.h          NEW — CUDA backend (context + handle, opaque Impl)
  metal_compute.h         NEW — Metal backend (context + handle)
  batch_evaluator.h       UNCHANGED — NNResultBuf + BatchEvaluator
  nn_evaluator.h          MODIFIED — takes LoadedModel + ComputeContext + gpu_ids
  config.h                UNCHANGED
  game.h                  UNCHANGED
  mcts.h                  UNCHANGED
  onnx_loader.h           UNCHANGED

src/
  loaded_model.cpp        NEW — ONNX parsing + BN pre-fusion
  compute_context.cpp     NEW — factory for backend-specific context
  eigen_compute.cpp       NEW — Eigen context + handle
  opencl_compute.cpp      NEW — OpenCL context + handle (refactored from opencl_engine.cpp)
  cuda_compute.cu         NEW — CUDA context + handle + FP16 WMMA kernels
  metal_compute.mm        NEW — Metal/MPSGraph context + handle
  nn_evaluator.cpp        MODIFIED — N server threads, ComputeHandle per thread
  game.cpp                UNCHANGED
  mcts.cpp                UNCHANGED
  onnx_loader.cpp         UNCHANGED
  main_selfplay.cpp       MODIFIED — uses new LoadedModel + ComputeContext flow
  main_benchmark.cpp      MODIFIED — same
  main_play.cpp           MODIFIED — same

DELETED (old engine files removed):
  include/inference_engine.h, include/opencl_engine.h, include/metal_engine.h, include/eigen_engine.h
  src/inference_engine.cpp, src/opencl_engine.cpp, src/metal_engine.mm, src/eigen_engine.cpp
```

## Test Results

### macOS M1 Max (Metal FP16 backend)

| Test | Result |
|---|---|
| Selfplay, 1 server thread | 10/10 pass |
| Selfplay, 2 server threads (0,0) | 3/3 pass |
| Benchmark, Metal FP16 batch-128 | 44K states/s |
| run_loop.sh --quick | Completes full training loop |

### Linux RTX 5070 Ti (OpenCL backend — single-gpu branch)

| Test | Result |
|---|---|
| Selfplay, 64 games, 8 threads, 32 search | 2.17 s/game |
| Batch-128 throughput | 55K states/s |

### macOS M1 Max (OpenCL backend — multi-gpu branch)

| Test | Result |
|---|---|
| Selfplay, 1 server thread, GPU 0 | 2 games pass (2.86 s/game) |
| Selfplay, 2 server threads, GPU 0,0 | 4 games pass (4.12 s/game) |
| Benchmark batch-128 throughput | 15K states/s |
| Benchmark single inference | 306 inf/s (3.3 ms/call) |

### Linux 2× RTX 5070 Ti (OpenCL backend — multi-gpu branch)

All 9 tests pass (`test_multi_gpu.sh`).

| Test | Result |
|---|---|
| Selfplay, 1 server, GPU 0 | 4 games pass (1.47 s/game) |
| Selfplay, 2 servers, GPU 0,0 | 4 games pass (1.07 s/game, 1.4× baseline) |
| Selfplay, 2 servers, GPU 0,1 | 4 games pass (0.56 s/game, **2.6× baseline**) |
| Selfplay, 4 servers, GPU 0,0,1,1 | 8 games pass (0.36 s/game, **4.1× baseline**) |
| Benchmark batch-128 throughput | 52K states/s |
| Large model (128 filters, 10 blocks), 2 GPUs | 4 games pass (2.35 s/game) |
| Stability (10 sequential runs, 2 GPUs) | 10/10 pass |

### Linux 2× RTX 2080 Ti (CUDA FP16+WMMA backend)

Batch NN inference throughput (states/s):

| Batch | CUDA FP16+WMMA | OpenCL FP32 | Speedup |
|------:|---------------:|------------:|--------:|
| **Small model (64f, 5b)** | | | |
| 1     | 1,013          | 934         | 1.08× |
| 32    | 27,060         | 19,886      | 1.36× |
| 64    | 45,247         | 27,383      | **1.65×** |
| 128   | 56,452         | 34,005      | **1.66×** |
| **Large model (128f, 10b)** | | | |
| 1     | 330            | 324         | 1.02× |
| 32    | 7,766          | 4,723       | 1.64× |
| 64    | 9,160          | 5,712       | 1.60× |
| 128   | 11,007         | 6,012       | **1.83×** |

Multi-GPU self-play (CUDA, small model, 800 sims):

| Config | Wall time |
|---|---|
| 1 server, GPU 0 | 39.2s (5 games) |
| 2 servers, GPU 0,1 | 15.9s (5 games, **2.5× speedup**) |
