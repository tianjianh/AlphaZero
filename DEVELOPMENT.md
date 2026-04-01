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

### Session 7 (current): Multi-GPU Refactor
- **LoadedModel**: extracted ONNX parsing + BN pre-fusion into shared CPU-side class
- **ComputeContext / ComputeHandle**: base classes matching KataGo's architecture
- **EigenComputeContext/Handle**: working, tested with 2 server threads
- **MetalComputeContext/Handle**: working, tested with 2 server threads
- **NNEvaluator refactored**: takes LoadedModel + ComputeContext + gpu_ids, spawns N server threads
- **ComputeHandle created ON server thread**: matches KataGo pattern exactly
- **New CLI**: `--nn-server-threads N`, `--nn-device-ids 0,1`

## What Remains (multi-gpu branch)

### Priority 1: OpenCL ComputeContext/Handle Refactor

The 866-line `src/opencl_engine.cpp` needs to be refactored into `src/opencl_compute.cpp`:

**OpenCLComputeContext** (created on main thread, shared):
- `init_opencl(device_id)`: select platform + device by index
- `compile_kernels()`: build program from embedded kernel source string
- One cl_context + cl_command_queue + cl_program per unique GPU device
- KataGo pattern: separate contexts per GPU to avoid NVIDIA serialization
- The `OPENCL_KERNELS` string and all 6 kernel definitions stay verbatim

**OpenCLComputeHandle** (created ON server thread):
- Upload weights from `LoadedModel` CPU data → `cl_mem` buffers via `clCreateBuffer(CL_MEM_COPY_HOST_PTR)`
- Allocate workspace buffers (buf_main_, buf_temp_, buf_skip_, etc.)
- `predict_batch()`: same forward pass logic as current opencl_engine.cpp
- References shared cl_context/cl_command_queue from OpenCLComputeContext

**Key code to extract from opencl_engine.cpp:**
- Lines 18-307: `OPENCL_KERNELS` string → stays in opencl_compute.cpp (static const)
- Lines 305-318: `CL_CHECK` macro, `new_buf`, `release_buf` → utility helpers
- Lines 327-393: `init_opencl()` → moves to `OpenCLComputeContext` constructor
- Lines 398-427: `compile_kernels()` → moves to `OpenCLComputeContext` constructor
- Lines 432-478: `allocate_workspace()`, `free_workspace()` → `OpenCLComputeHandle`
- Lines 483-621: `load_model()` → REPLACED by `LoadedModel::load()` + weight upload in `OpenCLComputeHandle` constructor
- Lines 626-866: `predict_batch()` + helper methods → `OpenCLComputeHandle::predict_batch()`

**The weight upload change:**
```cpp
// OLD (opencl_engine.cpp): parsed ONNX, uploaded directly
auto& wt = get("input_conv.weight");
g.weight = upload(wt.get_floats());

// NEW (opencl_compute.cpp): upload from LoadedModel CPU data
g.weight = upload(model->input_conv.weight);
```

### Priority 2: Clean Up Old Files

After OpenCL refactor is complete, remove:
- `include/opencl_engine.h` → replaced by `include/opencl_compute.h`
- `src/opencl_engine.cpp` → replaced by `src/opencl_compute.cpp`
- `include/metal_engine.h` → replaced by `include/metal_compute.h`
- `src/metal_engine.mm` → replaced by `src/metal_compute.mm`
- `include/eigen_engine.h` → replaced by `include/eigen_compute.h`
- `src/eigen_engine.cpp` → replaced by `src/eigen_compute.cpp`
- `include/inference_engine.h` → replaced by `include/compute_context.h` + `include/loaded_model.h`
- `src/inference_engine.cpp` → replaced by `src/compute_context.cpp`

### Priority 3: run_loop.sh Updates

Add new flags:
```bash
NN_SERVER_THREADS=1
NN_DEVICE_IDS="0"

--nn-server-threads) NN_SERVER_THREADS=$2; shift 2;;
--nn-device-ids)     NN_DEVICE_IDS=$2; shift 2;;

# Pass to selfplay
"${BUILD_DIR}/selfplay" \
    --nn-server-threads ${NN_SERVER_THREADS} \
    --nn-device-ids ${NN_DEVICE_IDS} \
    ...
```

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
         cl_context_0             cl_context_1
         cl_queue_0               cl_queue_1
         cl_program_0             cl_program_1
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
  opencl_compute.cpp      WIP — needs full refactor from opencl_engine.cpp
  metal_compute.mm        NEW — Metal/MPSGraph context + handle
  nn_evaluator.cpp        MODIFIED — N server threads, ComputeHandle per thread
  game.cpp                UNCHANGED
  mcts.cpp                UNCHANGED
  onnx_loader.cpp         UNCHANGED
  main_selfplay.cpp       MODIFIED — uses new LoadedModel + ComputeContext flow
  main_benchmark.cpp      MODIFIED — same
  main_play.cpp           MODIFIED — same

TO DELETE (after OpenCL refactor):
  include/inference_engine.h
  include/opencl_engine.h
  include/metal_engine.h
  include/eigen_engine.h
  src/inference_engine.cpp
  src/opencl_engine.cpp
  src/metal_engine.mm
  src/eigen_engine.cpp
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

### Multi-GPU testing (TODO — needs OpenCL refactor first)

To be tested on 2× RTX 5070 Ti with `test_multi_gpu.sh`.
