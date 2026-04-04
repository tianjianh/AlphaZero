# MiniGo C++ vs KataGo — Detailed Architecture Comparison

Side-by-side comparison of implementation details across all major subsystems.

## 1. Neural Network Evaluator

| Aspect | KataGo | MiniGo C++ |
|---|---|---|
| **Server loop function** | `NNEvaluator::serve()` | `NNEvaluator::server_loop()` |
| **Queue type** | `ThreadSafeQueue<NNResultBuf*>` (custom, dual-vector swap) | `std::vector<NNResultBuf*>` + mutex |
| **Queue drain** | `waitPopUpToN(buf, N)` — atomic multi-pop under one lock | `queue_.erase(begin, begin+n)` — O(n) copy |
| **Queue capacity** | Pre-allocated `maxBatchSize * 4 * numGPUs` | Pre-allocated `max_batch_size` |
| **Queue bounded?** | Yes — `notFullCondVar` blocks pushers when full | No — unbounded vector |
| **Batch sizing** | Dynamic — `setCurrentBatchSize()` adjustable at runtime | Static — `min(queue.size(), max_batch_size)` |
| **Batch timeout** | None (blocks indefinitely) | None (blocks indefinitely) |
| **GPU call** | `NeuralNet::getOutput()` — synchronous, blocks | `handle->predict_batch()` — synchronous, blocks |
| **NN cache** | `NNCacheTable` with `MutexPool`, `Hash128` keys | None |
| **Symmetry** | Evaluator-level: random symmetry applied per eval, averaging support | Game-level: dihedral augmentation during training data generation |
| **Post-eval processing** | Softmax scaling, policy temperature, ownership maps | Raw policy + scalar value only |

## 2. Result Buffer and Delivery

| Aspect | KataGo | MiniGo C++ |
|---|---|---|
| **Struct name** | `NNResultBuf` | `NNResultBuf` |
| **Condvar** | `clientWaitingForResult` (per-buf) | `cv` (per-buf) |
| **Mutex** | `resultMutex` (per-buf) | `mu` (per-buf) |
| **Done flag** | `bool hasResult` | `bool done` |
| **Result type** | `std::shared_ptr<NNOutput>` (policy, winloss, score, ownership) | `std::vector<float> policy` + `float value` |
| **Server notification** | `notify_all()` | `notify_one()` |
| **Input storage** | Multiple vectors (spatial rows, global rows) | Raw pointer `state_data` + `state_size` |
| **Allocation** | Pre-allocated per search thread, reused | Pre-allocated per search thread, reused |
| **Heap alloc per eval** | Zero (reused buf) | Zero (reused buf) |

## 3. MCTS Node

| Aspect | KataGo | MiniGo C++ |
|---|---|---|
| **Visit count** | `std::atomic<int64_t>` | `std::atomic<int>` |
| **Virtual losses** | `std::atomic<int32_t>` | `std::atomic<int>` |
| **Value accumulator** | `std::atomic<double>` (multiple: winloss, score, lead, utility) | `std::atomic<int32_t>` bit-pattern CAS on float |
| **Node states** | 7 states: UNEVALUATED → EVALUATING → EXPANDED0 → GROWING1 → EXPANDED1 → GROWING2 → EXPANDED2 | 3 states: UNEVALUATED → EXPANDING → EXPANDED |
| **Children storage** | Progressive arrays: 8 → 64 → MAX_POLICY_SIZE (no reallocation) | `std::vector<std::unique_ptr<MCTSNode>>` |
| **Transposition table** | Yes — `SearchNodeTable` with hash lookup | No — pure tree (no sharing) |
| **Memory order** | `acquire`/`release` for state; mixed for stats | `relaxed` for everything except state CAS |
| **Prior** | Stored in edge (`SearchChildPointer`) | Stored in node (`float prior`) |

## 4. Virtual Loss and Search

| Aspect | KataGo | MiniGo C++ |
|---|---|---|
| **Apply vloss** | `fetch_add(1)` during descent | `fetch_add(1)` during descent |
| **Revert vloss** | `fetch_sub(1)` on collision or backprop | `fetch_sub(1)` on collision or backprop |
| **Q-value formula** | `(winlossAvg - vloss) / (visits + vloss)` (plus score, lead, utility terms) | `(total_value - vloss) / (visits + vloss)` |
| **UCB exploration** | PUCT with multiple value components | PUCT: `c_puct * prior * sqrt(parent_N) / (1 + child_N)` |
| **Collision handling** | Revert vloss → yield → retry from root | Revert vloss → yield → retry from root |
| **Root noise** | Dirichlet noise on root prior | Dirichlet noise on root prior |
| **Temperature** | Configurable with threshold | Greedy after `temperature_threshold` moves |
| **Cycle detection** | `graphPath` set tracks visited nodes | No — tree is acyclic by construction |
| **Search thread stack** | Default OS thread stack | 8MB via `pthread_attr_setstacksize` |

## 5. Multi-GPU Architecture

| Aspect | KataGo | MiniGo C++ |
|---|---|---|
| **Server threads** | N threads, one per GPU assignment | N threads, one per GPU assignment |
| **GPU assignment** | `gpuIdxByServerThread` vector | `gpu_ids` vector |
| **Handle creation** | `NeuralNet::createComputeHandle()` on server thread | `context_->create_handle()` on server thread |
| **Queue topology** | Single shared queue, competing consumers | Single shared queue, competing consumers |
| **Load balancing** | Self-balancing (fastest GPU drains next) | Self-balancing (fastest GPU drains next) |
| **Per-GPU isolation** | Separate compute handle per thread; shared `cl_context` per GPU | Same: separate `ComputeHandle` per thread; shared `cl_context`/`cl_queue` per GPU |
| **Default servers per GPU** | 1 | 1 (but 2 can help — see below) |

## 6. Compute Backend

| Aspect | KataGo | MiniGo C++ |
|---|---|---|
| **Backend abstraction** | `NeuralNet::getOutput()` free function dispatch | `ComputeHandle::predict_batch()` virtual method |
| **Supported backends** | OpenCL, CUDA, TensorRT, Eigen, Metal | TensorRT, CUDA, OpenCL, Metal, Eigen |
| **Backend selection** | Compile-time (`#define`) | Compile-time (`cmake -DMINIGO_BACKEND=`) |
| **TensorRT** | ONNX parser, FP16, kernel auto-tune, engine cached to disk | Same: ONNX parser, FP16, auto-tune, engine cached in `trt_cache/` |
| **CUDA** | cuDNN + cuBLAS, FP16 | Hand-written FP16 WMMA Tensor Core implicit GEMM kernels |
| **OpenCL kernels** | Tuned with auto-tuning pass | Hand-written implicit GEMM with register blocking |
| **Weight format** | Custom binary (gzipped, SHA256 verified) | ONNX (built-in minimal protobuf parser) |
| **BN fusion** | During model conversion | During `LoadedModel::load()` — `scale = gamma/sqrt(var+eps)` |
| **Precision** | FP16/FP32 configurable per layer type | TensorRT/CUDA/Metal: FP16; OpenCL/Eigen: FP32 |
| **Weight sharing** | `ModelDesc` shared (const), each handle uploads to GPU | `LoadedModel` shared (const), each handle uploads to GPU |

## 7. Model Architecture

| Aspect | KataGo | MiniGo C++ |
|---|---|---|
| **Input encoding** | Multiple versions (V3-V7), 13-22 spatial + 19 global features | 17 channels: 8 history x 2 colors + 1 color plane |
| **Trunk** | Residual + global pooling blocks, bottleneck support | Simple residual blocks (conv3x3 + BN + ReLU) |
| **Policy head** | Spatial policy + pass prediction, score belief | 1x1 conv → FC → softmax (board_size^2 + 1 actions) |
| **Value head** | Win/loss/draw + score mean/variance + lead + ownership | 1x1 conv → FC → ReLU → FC → tanh (single scalar) |
| **Typical size** | 256 filters, 40 blocks (b40) | 64 filters, 5 blocks |
| **Board sizes** | 9x9 through 29x29 | Configurable (default 9x9) |

## 8. Input Encoding

| Aspect | KataGo | MiniGo C++ |
|---|---|---|
| **History depth** | 5 moves | 8 moves |
| **Feature planes** | 22 spatial (V7): stones, liberties, ko, area, ladder, etc | 17: current stones x8, opponent stones x8, color |
| **Global features** | 19 (V7): komi, rules, ko state, scoring area | None |
| **Encoding location** | `NNInputs::fillRowV7()` | `GoGame::encode()` |
| **History storage** | Board + move history objects | Ring buffer (`std::array` with head/size) |
| **History update** | Vector operations | O(1) ring buffer rotation |

## 9. Synchronization Primitives

| Aspect | KataGo | MiniGo C++ |
|---|---|---|
| **Queue lock** | `std::mutex` inside `ThreadSafeQueue` | `std::mutex queue_mutex_` |
| **Queue signal** | `notEmptyCondVar` + `notFullCondVar` | Single `queue_cv_` |
| **Shutdown** | `isKilled` flag + queue `close()`/`setReadOnly()` | `bool stop_` flag |
| **Atomic float** | `std::atomic<double>` (platform support) | CAS loop on `std::atomic<int32_t>` bit pattern |
| **Per-node lock** | `MutexPool` indexed by hash (for expansion) | Lock-free CAS on state enum |
| **Stats lock** | `statsLock` mutex for virtual loss writes | Lock-free `fetch_add`/`fetch_sub` |

## 10. Key Design Differences

### Things KataGo has that MiniGo doesn't:
- **NN cache** — avoids re-evaluating positions seen before
- **Transposition table** — shared nodes across search paths
- **Global pooling blocks** — richer trunk architecture
- **Multiple value outputs** — win/loss, score, lead, ownership
- **Auto-tuned OpenCL** — per-device kernel parameter tuning
- **Dynamic batch sizing** — adjusts batch cap at runtime
- **Progressive child arrays** — avoids reallocation (8 → 64 → max)
- **7-state node expansion** — supports growing children arrays

### Things MiniGo does differently:
- **ONNX model format** — industry standard, no custom converter needed
- **ONNX model format** — industry standard, no custom converter needed (KataGo uses custom binary)
- **Implicit GEMM kernel** — fused im2col computed on-the-fly in register-blocked SGEMM
- **CUDA WMMA kernels** — hand-written FP16 Tensor Core implicit GEMM (no cuDNN dependency)
- **CAS float accumulation** — portable C++17, no `std::atomic<double>` dependency
- **Ring buffer history** — O(1) update vs vector operations
- **Simpler node states** — 3 states vs 7 (no progressive resizing)
- **`notify_one` delivery** — avoids thundering herd on result notification
- **Evaluation gating** — dedicated `evaluate` binary plays model-vs-model matches with SGF output
- **TRT engine cache per version** — pipeline uses versioned model files so TRT cache persists across iterations

## 11. Training Pipeline

| Aspect | KataGo | MiniGo C++ |
|---|---|---|
| **Pipeline orchestration** | Custom C++ `SelfplayManager` + Python training | Bash `run_loop.sh` (init/train/status) + Python training |
| **Selfplay data format** | Custom binary (gzipped) | Binary `.bin` compressed with zstd |
| **Training framework** | Custom C++ training loop (TF or PyTorch) | PyTorch with DDP via `torchrun` |
| **Multi-GPU training** | Custom data-parallel or single GPU | PyTorch DistributedDataParallel (NCCL) |
| **Data loading** | Streaming from compressed files | Streaming IterableDataset (zstd decompress per file, 8 workers) |
| **Data window** | Sliding window of recent games (weighted sampling) | Sliding window of last N iterations (uniform sampling) |
| **Evaluation gating** | Continuous Elo tracking, no hard gate | Match-based: candidate must beat best by ≥55% win rate |
| **Evaluation games** | Ongoing Elo estimation from selfplay results | Dedicated `evaluate` binary, games saved as SGF |
| **Model versioning** | Sequential network files, best tracked separately | `models/v0000.onnx` ... `v0100.onnx` + `best.onnx` |
| **LR schedule** | Configured externally, manual changes | Staged plan: per-stage LR defined in `training_plan` |
| **Resume** | Checkpoint-based, manual restart | Automatic: `run_loop.sh train` detects state and continues |
| **Training plan** | Manual configuration files | Generated by `init`, editable text file with stage definitions |
| **GPU auto-detection** | Manual `--config` | Auto: detects GPU count, sets NN servers + DDP processes |
| **Selfplay compression** | gzip | zstd (faster decompress, similar ratio) |
| **Game visualization** | External SGF viewers | Built-in `visualize.py` (reads .bin.zst and .sgf) |

### Key training differences:
- **KataGo uses Elo tracking** from selfplay results (no separate evaluation matches). MiniGo plays explicit head-to-head matches between candidate and best model.
- **KataGo weights recent data higher** via exponential weighting. MiniGo uses a flat sliding window (all games in window are equal).
- **KataGo runs selfplay continuously** with training happening in parallel. MiniGo runs sequential phases: selfplay → train → evaluate per iteration.
- **KataGo's training is in C++** (with optional PyTorch). MiniGo uses Python/PyTorch exclusively for training, with DDP for multi-GPU.

---

## 12. NNEvaluator Server — Full Technical Detail

### Initialization flow (MiniGo)

```
main thread:
  LoadedModel::load("models/v0003.onnx") ← parse ONNX, pre-fuse BN, store CPU weights
       │
       ▼
  create_compute_context(device_ids)     ← backend auto-detected at compile time:
       │                                    TensorRT: cudaStream per GPU, engine built lazily
       │                                    CUDA: cudaStream per GPU
       │                                    OpenCL: cl_context + cl_queue + cl_program per GPU
       │                                    Metal: MTLDevice + compiled MPSGraph
       ▼
  NNEvaluator(model, context, gpu_ids, max_batch)
       │
       └── for each gpu_id in gpu_ids:
             spawn server_thread(i, gpu_id)
                │
                └── ON server thread:
                      context_->create_handle(model, gpu_id, max_batch)
                        │
                        ├── TensorRT: build/load engine from ONNX (cached in trt_cache/),
                        │             create IExecutionContext, allocate I/O buffers
                        ├── CUDA: upload weights as FP16, allocate workspace
                        ├── OpenCL: create cl_kernels, upload weights, allocate workspace
                        └── Metal: buffers from shared MPSGraph
```

### Initialization flow (KataGo)

```
main thread:
  NeuralNet::globalInitialize()          ← platform-level init
  LoadedModel = NeuralNet::loadModelFile()  ← custom binary format, SHA256 verified
       │
       ▼
  NNEvaluator(loadedModel, nnCacheTable, logger, config, ...)
       │
       └── for each server thread:
             spawn serve(buf, rand, gpuIdx, threadIdx)
                │
                └── ON server thread:
                      NeuralNet::createComputeHandle(context, loadedModel, logger, ...)
                        │
                        └── per-handle: allocate GPU buffers, upload weights,
                            compile kernels (with auto-tuning if first run)
```

**Key similarity**: Both MiniGo and KataGo share `cl_context` per GPU — deliberately creating
separate contexts per GPU (not per platform) to avoid NVIDIA serialization where one GPU
locks out others.  Each `ComputeHandle` references the shared context and command queue
from its GPU's device state. Handles own their own kernel objects and GPU buffers.

**Key difference**: Weight format. MiniGo uses ONNX (parsed by a built-in minimal protobuf
parser); KataGo uses a custom binary format (gzipped, SHA256 verified). Both parse the
model file once on the main thread into shared CPU weights, then each handle uploads
its own copy to GPU via `clCreateBuffer`.

### Server loop — step by step (MiniGo)

```cpp
void NNEvaluator::server_loop(int thread_id, int gpu_id) {
    // ── Step 0: Create GPU handle ON this thread ──
    auto handle = context_->create_handle(model_.get(), gpu_id, max_batch_size_);

    while (true) {
        // ── Step 1: Block until queue has items ──
        // Lock queue_mutex_, wait on queue_cv_ until !queue_.empty() || stop_
        // If stop_ && empty: exit thread

        // ── Step 2: Drain up to max_batch_size items ──
        // batch.assign(queue_.begin(), queue_.begin() + n)
        // queue_.erase(queue_.begin(), queue_.begin() + n)   ← O(n) shift
        // Release queue_mutex_

        // ── Step 3: Flatten states (CPU work) ──
        // For each NNResultBuf* in batch:
        //   Copy state_data[0..state_size] into vector<vector<float>>
        // This is a full copy of all input tensors

        // ── Step 4: GPU inference (BLOCKS) ──
        // handle->predict_batch(all_states)
        //   Inside predict_batch (OpenCL):
        //     a. Flatten all states into one contiguous buffer
        //     b. clEnqueueWriteBuffer → upload to buf_flat_in_
        //     c. Enqueue transpose kernel (NCHW → CNHW)
        //     d. Enqueue input conv (implicit GEMM + BN + ReLU)
        //     e. For each res block: 2× implicit GEMM (conv1 + conv2+residual)
        //     f. Enqueue policy head (1x1 conv + FC + softmax)
        //     g. Enqueue value head (1x1 conv + FC + ReLU + FC + tanh)
        //     h. clEnqueueReadBuffer × 2 (policy + value)
        //     i. clFinish()  ← BLOCKS until all GPU work completes
        //     j. Pack results into vector<pair<vector<float>, float>>

        // ── Step 5: Deliver results ──
        // For each buf in batch:
        //   Lock buf->mu
        //   Move policy vector, set value float
        //   Set buf->done = true
        //   Unlock buf->mu
        //   buf->cv.notify_one()  ← wake the specific search thread
    }
}
```

### Server loop — step by step (KataGo)

```cpp
void NNEvaluator::serve(NNServerBuf& buf, Rand& rand, int gpuIdx, int threadIdx) {
    // ── Step 0: Create GPU handle ON this thread ──
    ComputeHandle* gpuHandle = NeuralNet::createComputeHandle(...);

    while (true) {
        // ── Step 1: Block until queue has items ──
        // queryQueue.waitPopUpToN(resultBufs, desiredBatchSize)
        // Uses ThreadSafeQueue: dual-vector swap, notEmptyCondVar
        // Returns false when queue closed → exit thread

        // ── Step 2: Drain is atomic inside waitPopUpToN ──
        // Under one lock: pop min(size, N) items via swap-vector pattern
        // No O(n) erase — items are already in the output vector

        // ── Step 3: Prepare inputs ──
        // For each NNResultBuf: apply symmetry transform, fill spatial/global rows
        // Symmetry can be randomized or fixed

        // ── Step 4: GPU inference (BLOCKS) ──
        // NeuralNet::getOutput(gpuHandle, inputBuffers, numRows, resultBufs, outputBuf)
        // Backend-specific blocking call (same pattern as MiniGo)

        // ── Step 5: Deliver results ──
        // For each buf:
        //   Move NNOutput* into buf->result (shared_ptr)
        //   Set buf->hasResult = true
        //   buf->clientWaitingForResult.notify_all()
        //
        // Then: lock bufferMutex, decrement numOngoingEvals,
        //       notify waitingForFinish if anyone is waiting
    }
}
```

### Client-side (search thread → server) — MiniGo

```
Search thread calls evaluate_with_buf(buf, state):

  1. buf.done = false
  2. buf.state_data = state.data()     ← pointer, no copy
  3. buf.state_size = state.size()
  4. Lock queue_mutex_
  5. queue_.push_back(&buf)            ← push pointer
  6. Unlock queue_mutex_
  7. queue_cv_.notify_all()            ← wake all server threads
  8. Lock buf.mu
  9. buf.cv.wait(lock, [&]{ return buf.done; })  ← BLOCK here
  10. Return {buf.policy, buf.value}
```

### Client-side (search thread → server) — KataGo

```
Search thread calls evaluate(board, history, player, params, buf, ...):

  1. Check NN cache (Hash128 lookup) → if hit, return cached NNOutput
  2. Fill spatial input rows from board state (fillRowV3..V7)
  3. Fill global input rows (komi, rules, ko, etc.)
  4. buf.hasResult = false
  5. queryQueue.waitPush(&buf)          ← may block if queue full (bounded)
  6. Lock buf.resultMutex
  7. buf.clientWaitingForResult.wait(lock, [&]{ return buf.hasResult; })
  8. Store result in NN cache for future lookups
  9. Return buf.result (shared_ptr<NNOutput>)
```

**Key differences**:
- KataGo checks cache before queuing — avoids GPU call for repeated positions
- KataGo's queue is bounded — pushers can block if queue is full (backpressure)
- MiniGo's queue is unbounded — pushers never block on queue (only on result)
- KataGo fills input features inline; MiniGo passes a pre-encoded state pointer

### Resource ownership diagram (MiniGo, OpenCL backend)

```
MAIN THREAD creates:                 SERVER THREAD 0 creates:        SERVER THREAD 1 creates:
─────────────────────               ────────────────────────        ────────────────────────
LoadedModel (shared const)          OpenCLComputeHandle             OpenCLComputeHandle
  ├── input_conv.weight  ─────────▶   ├── cl_kernel × 6              ├── cl_kernel × 6
  ├── res_conv1[i].weight ────────▶   ├── ConvBNGPU weights          ├── ConvBNGPU weights
  ├── ...                             │    (cl_mem, own copy)         │    (cl_mem, own copy)
  └── value_fc2.bias     ────────▶   ├── workspace bufs             ├── workspace bufs
                                      │    buf_main_, buf_temp_...    │    buf_main_, buf_temp_...
OpenCLComputeContext (shared)         │                               │
  └── devices_ map:                   │ References:                   │ References:
       gpu_id=0:                      │  dev_.context (shared) ◄──────┤  dev_.context (shared)
         ├── cl_context    ◄──────────┤  dev_.queue   (shared) ◄──────┤  dev_.queue   (shared)
         ├── cl_command_queue ◄───────┤  dev_.program (shared) ◄──────┤  dev_.program (shared)
         └── cl_program    ◄──────────┘                               └──
       gpu_id=1:
         ├── cl_context        (only used by threads assigned to GPU 1)
         ├── cl_command_queue
         └── cl_program

NNEvaluator (shared)
  ├── queue_mutex_         ← all server threads + all search threads contend
  ├── queue_cv_            ← notify_all wakes all servers
  ├── queue_<NNResultBuf*> ← competing consumers drain
  └── server_threads_[]    ← joined on destructor
```

### GPU pipeline timing (why 2 servers on 1 GPU helps)

With 1 server thread, the server's CPU work creates gaps where the GPU is idle:

```
Server:  [drain 0.5ms][flatten 1.5ms][  GPU 5ms (clFinish blocks)  ][pack 0.5ms][deliver 1ms][drain...
GPU:      idle idle idle idle idle     ████████████████████████████████  idle idle idle idle idle ...
                                       ▲                            ▲
                                       GPU starts                   GPU done
```

With 2 server threads on the same GPU, server 1 prepares while server 0's batch runs.
Both threads share `dev_.queue` (in-order OpenCL queue), so GPU batches execute back-to-back:

```
Server0: [drain+flatten 2ms][enqueue][clFinish blocks 5ms][deliver 1.5ms][drain+flatten 2ms]...
Server1:    [drain+flatten 2ms][enqueue][  clFinish blocks ~7ms  ][deliver 1.5ms]...
GPU:     idle████████batch_A████████████████batch_B████████████████batch_C████████████████...
```

The `clEnqueue*` calls are async (just add work to the queue). `clFinish` blocks until
all *previously* enqueued commands complete. With an in-order queue, server 1's enqueued
kernels execute after server 0's — and server 0's `clFinish` only waits for its own
commands (those enqueued before its `clFinish` call).

Measured improvement: 1.24x on single GPU (2.03 → 1.64 s/game at 16 threads, 24 search).

### Why more servers on same GPU hurts at high thread count

With 4 competing consumers on the shared queue vs 2:
- Each server grabs ~1/4 of pending items instead of ~1/2
- Smaller batches → worse GPU throughput (batch-64: 39K states/s vs batch-128: 52K states/s)
- The CPU pipelining benefit saturates at 2 servers (CPU gaps already hidden)
- Net effect: throughput loss from batch fragmentation > gain from pipelining

Optimal config: **1 server per GPU** when total search threads are high enough to fill
batches. Use **2 servers per GPU** only when GPU time per batch is short relative to CPU
overhead (small models, small batches).

### Shutdown sequence (MiniGo)

```
NNEvaluator destructor:
  1. Lock queue_mutex_, set stop_ = true, unlock
  2. queue_cv_.notify_all()              ← wake all servers
  3. Each server thread sees stop_ && queue_.empty() → breaks out of loop
  4. ComputeHandle destructor:
       - free workspace (clReleaseMemObject × 10)
       - free weights (clReleaseMemObject for all conv/fc buffers)
       - clReleaseKernel × 6
  5. join() all server threads
  6. ComputeContext destructor (later, when shared_ptr refcount → 0):
       - clReleaseProgram, clReleaseCommandQueue, clReleaseContext per GPU
```

Sources:
- [KataGo nneval.cpp](https://github.com/lightvector/KataGo/blob/master/cpp/neuralnet/nneval.cpp)
- [KataGo nneval.h](https://github.com/lightvector/KataGo/blob/master/cpp/neuralnet/nneval.h)
- [KataGo searchnode.h](https://github.com/lightvector/KataGo/blob/master/cpp/search/searchnode.h)
- [KataGo threadsafequeue.h](https://github.com/lightvector/KataGo/blob/master/cpp/core/threadsafequeue.h)
