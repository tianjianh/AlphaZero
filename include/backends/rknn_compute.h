#pragma once
#ifdef MINIGO_HAS_RKNN

#include "nn/compute_context.h"
#include "model/loaded_model.h"
#include <memory>
#include <string>
#include <vector>

namespace minigo {

// Forward-declared — defined in rknn_compute.cpp (avoids rknn_api.h here)
struct RKNNDeviceState;

// ================================================================
// RKNN backend (Rockchip NPU, RK3562/RK3566/RK3568/RK3576/RK3588)
//
// Topology: exactly one NPU device (gpu_id is always 0).  Multiple
// server threads share the NPU by each holding a duplicated
// rknn_context (weights are shared across dups by the runtime).
//
// Model file: the RKNN runtime consumes pre-compiled .rknn files.
// Resolution: `<model_path>` is used if it already ends in ".rknn";
// otherwise the ".onnx" extension is replaced with ".rknn".
//
// Core distribution: each handle pins its dup'd context to a single
// NPU core (round-robin across cores 0/1/2).  With N server threads
// on an N-core NPU, each thread gets its own core — maximal
// parallelism with no scheduler contention.
// ================================================================
class RKNNComputeContext : public ComputeContext {
public:
    explicit RKNNComputeContext(const std::vector<int>& device_ids);
    ~RKNNComputeContext();

    std::unique_ptr<ComputeHandle>
    create_handle(const LoadedModel* model, int gpu_id, int max_batch_size) override;

    std::string backend_name() const override { return "rknn"; }

    RKNNDeviceState& device_state();

private:
    struct Impl;
    Impl* impl_ = nullptr;
};

class RKNNComputeHandle : public ComputeHandle {
public:
    RKNNComputeHandle(RKNNDeviceState& dev, const LoadedModel* model,
                      int max_batch_size, int thread_index);
    ~RKNNComputeHandle();

    std::vector<Result>
    predict_batch(const std::vector<std::vector<float>>& states) override;

private:
    struct Impl;
    Impl* impl_ = nullptr;
};

}  // namespace minigo
#endif  // MINIGO_HAS_RKNN
