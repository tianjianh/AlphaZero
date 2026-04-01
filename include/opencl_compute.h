#pragma once
#ifdef MINIGO_HAS_OPENCL

#ifdef __APPLE__
#  include <OpenCL/cl.h>
#else
#  include <CL/cl.h>
#endif

#include "compute_context.h"
#include "loaded_model.h"
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace minigo {

// Per-unique-GPU device state (shared by all threads on that GPU)
struct OpenCLDeviceState {
    cl_platform_id  platform = nullptr;
    cl_device_id    device   = nullptr;
    cl_context      context  = nullptr;
    cl_command_queue queue    = nullptr;
    cl_program      program  = nullptr;
};

class OpenCLComputeContext : public ComputeContext {
public:
    // Initialize one cl_context + cl_queue + compiled cl_program per unique GPU
    explicit OpenCLComputeContext(const std::vector<int>& device_ids);
    ~OpenCLComputeContext();

    std::unique_ptr<ComputeHandle>
    create_handle(const LoadedModel* model, int gpu_id, int max_batch_size) override;

    std::string backend_name() const override { return "opencl"; }

    // Get device state for a given GPU index
    OpenCLDeviceState& device_state(int gpu_id);

private:
    std::map<int, OpenCLDeviceState> devices_;  // gpu_id → device state
};

class OpenCLComputeHandle : public ComputeHandle {
public:
    OpenCLComputeHandle(OpenCLDeviceState& dev, const LoadedModel* model, int max_batch_size);
    ~OpenCLComputeHandle();

    std::vector<Result>
    predict_batch(const std::vector<std::vector<float>>& states) override;

private:
    OpenCLDeviceState& dev_;

    // ── Kernel handles ─────────────────────────────────────────
    cl_kernel k_transpose_nchw_          = nullptr;
    cl_kernel k_conv3x3_sgemm_bn_        = nullptr;
    cl_kernel k_conv1x1_bn_relu_reshape_ = nullptr;
    cl_kernel k_fc_bias_relu_            = nullptr;
    cl_kernel k_fc_bias_softmax_         = nullptr;
    cl_kernel k_fc_bias_tanh_            = nullptr;

    // ── Weight buffers on GPU ──────────────────────────────────
    struct ConvBNGPU {
        cl_mem weight = nullptr, bn_scale = nullptr, bn_bias = nullptr;
        int c_out = 0, c_in = 0, k = 0;
    };
    struct FCGPU {
        cl_mem weight = nullptr, bias = nullptr;
        int out_features = 0, in_features = 0;
    };

    ConvBNGPU              input_conv_gpu_;
    std::vector<ConvBNGPU> res_conv1_gpu_, res_conv2_gpu_;
    ConvBNGPU              policy_conv_gpu_, value_conv_gpu_;
    FCGPU                  policy_fc_gpu_, value_fc1_gpu_, value_fc2_gpu_;

    // ── Workspace buffers ──────────────────────────────────────
    cl_mem buf_flat_in_ = nullptr, buf_input_ = nullptr;
    cl_mem buf_main_ = nullptr, buf_temp_ = nullptr, buf_skip_ = nullptr;
    cl_mem buf_pol_out_ = nullptr, buf_pol_feat_ = nullptr;
    cl_mem buf_val_h1_ = nullptr, buf_val_feat_ = nullptr, buf_val_out_ = nullptr;
    int alloc_batch_ = 0;

    // Model metadata
    int board_size, input_channels, num_filters, num_res_blocks;

    // Helpers
    cl_mem upload(const std::vector<float>& data);
    void release_buf(cl_mem& buf);
    void allocate_workspace(int batch);
    void free_workspace();
    void free_weights();

    void run_conv3x3(cl_mem input_buf, cl_mem output_buf,
                     const ConvBNGPU& conv, cl_mem residual_buf,
                     int N, int H, int W, int mode, bool relu);
    void run_conv1x1_bn_relu_reshape(cl_mem input_buf, cl_mem output_buf,
                                      const ConvBNGPU& conv, int N, int HW);
    void run_fc_bias_relu(cl_mem input_buf, cl_mem output_buf,
                          const FCGPU& fc, int N, bool relu);
};

}  // namespace minigo
#endif  // MINIGO_HAS_OPENCL
