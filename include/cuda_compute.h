#pragma once
#ifdef MINIGO_HAS_CUDA

#include "compute_context.h"
#include "loaded_model.h"
#include <memory>
#include <string>
#include <vector>

namespace minigo {

// Forward-declared — defined in cuda_compute.cu (avoids cuda_runtime.h in header)
struct CUDADeviceState;

class CUDAComputeContext : public ComputeContext {
public:
    // Initialize one cudaStream per unique GPU
    explicit CUDAComputeContext(const std::vector<int>& device_ids);
    ~CUDAComputeContext();

    std::unique_ptr<ComputeHandle>
    create_handle(const LoadedModel* model, int gpu_id, int max_batch_size) override;

    std::string backend_name() const override { return "cuda"; }

    // Get device state for a given GPU index
    CUDADeviceState& device_state(int gpu_id);

private:
    struct Impl;
    Impl* impl_ = nullptr;
};

class CUDAComputeHandle : public ComputeHandle {
public:
    CUDAComputeHandle(CUDADeviceState& dev, const LoadedModel* model, int max_batch_size);
    ~CUDAComputeHandle();

    std::vector<Result>
    predict_batch(const std::vector<std::vector<float>>& states) override;

private:
    struct Impl;
    Impl* impl_ = nullptr;
};

}  // namespace minigo
#endif  // MINIGO_HAS_CUDA
