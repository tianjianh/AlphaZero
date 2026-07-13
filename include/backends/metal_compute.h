#pragma once
#ifdef MINIGO_HAS_METAL

#include "nn/compute_context.h"
#include "model/loaded_model.h"
#include <memory>
#include <string>
#include <vector>

namespace minigo {

class MetalComputeContext : public ComputeContext {
public:
    explicit MetalComputeContext(const std::vector<int>& device_ids);
    ~MetalComputeContext();

    std::unique_ptr<ComputeHandle>
    create_handle(const LoadedModel* model, int gpu_id, int max_batch_size) override;

    std::string backend_name() const override { return "metal"; }

    // Opaque pointer to Obj-C implementation
    struct Impl;
    Impl* impl_ = nullptr;
};

class MetalComputeHandle : public ComputeHandle {
public:
    MetalComputeHandle(MetalComputeContext::Impl* ctx_impl, const LoadedModel* model);
    ~MetalComputeHandle();

    std::vector<Result>
    predict_batch(const std::vector<std::vector<float>>& states) override;

    // Opaque pointer to Obj-C implementation
    struct Impl;
    Impl* impl_ = nullptr;
};

}  // namespace minigo
#endif  // MINIGO_HAS_METAL
