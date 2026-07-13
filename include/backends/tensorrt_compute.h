#pragma once
#ifdef MINIGO_HAS_TENSORRT

#include "nn/compute_context.h"
#include "model/loaded_model.h"
#include <memory>
#include <string>
#include <vector>

namespace minigo {

// Forward-declared — defined in tensorrt_compute.cpp (avoids TRT headers here)
struct TRTDeviceState;

class TensorRTComputeContext : public ComputeContext {
public:
    // Initialize one CUDA stream per unique GPU; engines are built lazily
    // on first create_handle() call per device.
    explicit TensorRTComputeContext(const std::vector<int>& device_ids);
    ~TensorRTComputeContext();

    std::unique_ptr<ComputeHandle>
    create_handle(const LoadedModel* model, int gpu_id, int max_batch_size) override;

    std::string backend_name() const override { return "tensorrt"; }

    // Get device state for a given GPU index
    TRTDeviceState& device_state(int gpu_id);

private:
    struct Impl;
    Impl* impl_ = nullptr;
};

class TensorRTComputeHandle : public ComputeHandle {
public:
    TensorRTComputeHandle(TRTDeviceState& dev, const LoadedModel* model, int max_batch_size);
    ~TensorRTComputeHandle();

    std::vector<Result>
    predict_batch(const std::vector<std::vector<float>>& states) override;

private:
    struct Impl;
    Impl* impl_ = nullptr;
};

}  // namespace minigo
#endif  // MINIGO_HAS_TENSORRT
