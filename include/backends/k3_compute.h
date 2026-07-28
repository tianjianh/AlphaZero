#pragma once
#ifdef MINIGO_HAS_K3

#include "nn/compute_context.h"
#include "model/loaded_model.h"
#include <memory>
#include <string>
#include <vector>

namespace minigo {

// Forward-declared — defined in k3_compute.cpp (avoids onnxruntime headers here)
struct K3DeviceState;

// ================================================================
// K3 backend (SpacemiT K3, riscv64 Linux — ONNX Runtime based)
//
// The K3 has two 8-core clusters: X100 application cores (2.4 GHz,
// RVV VLEN=256) and A100 AI cores (2.0 GHz, RVV VLEN=1024 with
// per-core-pair TCM).  The A100 cluster is fenced off from normal
// scheduling — a thread must write "0" to /proc/set_ai_thread
// before the kernel accepts an affinity mask touching cpus 8-15
// (one-way: an AI thread can never re-bind to the X100 cluster).
//
// Inference runs through ONNX Runtime on the SAME .onnx file the
// other backends read (dynamic batch axis).  Two execution
// providers are supported per handle:
//   * spacemit — SpacemiT's EP (libspacemit_ep).  Its internal
//     thread pool marks itself via /proc/set_ai_thread, pins one
//     thread per A100 core and stages tiles through TCM.  The ORT
//     intra-op pool is left at 1 thread (the server thread only
//     does pre/post-processing).
//   * cpu — vanilla ORT CPU EP, with the intra-op pool pinned to a
//     slice of the X100 cluster.  Useful standalone on non-K3
//     machines, or alongside a spacemit handle to put BOTH clusters
//     to work (the NNEvaluator queue self-balances).
//
// Topology: one logical device; multiple server threads each hold
// their own Ort::Session (weights are re-read per session — Go nets
// are small).  gpu_id is ignored (RKNN pattern); handles are
// assigned a slice index from an atomic counter.
//
// Environment knobs (all optional):
//   MINIGO_K3_EP        spacemit | cpu | hetero   (default: spacemit if
//                       compiled in and the EP loads, else cpu.
//                       hetero: handle 0 = spacemit on the A100s,
//                       handles 1.. = cpu on X100 slices)
//   MINIGO_K3_THREADS   compute threads per handle (default: AI-core
//                       count / #servers for spacemit; X100-slice
//                       size for cpu)
//   MINIGO_K3_CPUSET    cpu list for cpu-EP handles, e.g. "0-7"
//                       (default: the non-AI cores)
//   MINIGO_K3_BUCKET    1 = pad batches up to power-of-two buckets
//                       (default 1 — steadies EP kernel caching)
//   MINIGO_K3_SPIN      ORT intra-op spinning for cpu handles (default 1)
//   MINIGO_K3_VERBOSE   1 = per-handle placement / timing chatter
// The EP's own SPACEMIT_EP_* variables pass straight through.
// ================================================================
class K3ComputeContext : public ComputeContext {
public:
    explicit K3ComputeContext(const std::vector<int>& device_ids);
    ~K3ComputeContext();

    std::unique_ptr<ComputeHandle>
    create_handle(const LoadedModel* model, int gpu_id, int max_batch_size) override;

    std::string backend_name() const override { return "k3"; }

    K3DeviceState& device_state();

private:
    struct Impl;
    Impl* impl_ = nullptr;
};

class K3ComputeHandle : public ComputeHandle {
public:
    K3ComputeHandle(K3DeviceState& dev, const LoadedModel* model,
                    int max_batch_size, int thread_index);
    ~K3ComputeHandle();

    std::vector<Result>
    predict_batch(const std::vector<std::vector<float>>& states) override;

private:
    struct Impl;
    Impl* impl_ = nullptr;
};

}  // namespace minigo
#endif  // MINIGO_HAS_K3
