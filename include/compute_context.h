#pragma once

#include "loaded_model.h"
#include <memory>
#include <string>
#include <vector>

namespace minigo {

// ================================================================
// ComputeHandle — per-server-thread GPU state (KataGo pattern)
//
// Created ON the server thread.  Owns GPU buffers, workspace, and
// any per-thread mutable state.  References the shared ComputeContext
// for device contexts and compiled kernels.
//
// Destroyed when the server thread exits.
// ================================================================
// ── NN output: policy + value + score + ownership ────────────
struct NNOutput {
    std::vector<float> policy;
    float value    = 0.0f;   // P(win) - P(loss), post-processed in ONNX
    float score    = 0.0f;   // scoreMean (raw points)
    float score_sd = 0.0f;   // scoreStdev (uncertainty in points)
    std::vector<float> ownership;  // [board²] per-intersection ownership (sigmoid)
};

class ComputeHandle {
public:
    virtual ~ComputeHandle() = default;

    // Run batch inference.  Called exclusively from the owning server thread.
    using Result = NNOutput;
    virtual std::vector<Result>
    predict_batch(const std::vector<std::vector<float>>& states) = 0;
};

// ================================================================
// ComputeContext — shared device state (KataGo pattern)
//
// Created once on the main thread.  Holds device contexts, compiled
// kernels, and any immutable state shared across server threads.
//
// For TensorRT: one ICudaEngine + cudaStream per unique GPU device
//               (engine built lazily, cached to disk).
// For OpenCL:   one cl_context + cl_command_queue + cl_program per
//               unique GPU device (avoids NVIDIA serialization).
// For CUDA:     one cudaStream per unique GPU device.
// For Metal:    one MTLDevice + compiled MPSGraph.
// For Eigen:    trivial (no GPU resources).
// ================================================================
class ComputeContext {
public:
    virtual ~ComputeContext() = default;

    // Create a ComputeHandle for a specific GPU device.
    // Called ON the server thread (KataGo pattern).
    virtual std::unique_ptr<ComputeHandle>
    create_handle(const LoadedModel* model, int gpu_id, int max_batch_size) = 0;

    // Backend name for display
    virtual std::string backend_name() const = 0;
};

// Factory: create the ComputeContext for the compile-time selected backend.
// device_ids: list of GPU indices that will be used (for pre-initializing devices).
std::unique_ptr<ComputeContext> create_compute_context(const std::vector<int>& device_ids);

// Returns the name of the compile-time selected backend.
std::string backend_name();

}  // namespace minigo
