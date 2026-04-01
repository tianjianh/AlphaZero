#pragma once
#ifdef MINIGO_HAS_OPENCL

#ifdef __APPLE__
#  include <OpenCL/cl.h>
#else
#  include <CL/cl.h>
#endif

#include "inference_engine.h"
#include <string>
#include <vector>

namespace minigo {

// ----------------------------------------------------------------
// OpenCL inference backend
//
// Runs the full AlphaZero forward pass on the GPU using OpenCL.
// Weights are loaded from the same .onnx file used by EigenEngine.
// Batch inference (predict_batch) runs a single GPU kernel sequence
// covering all N input states simultaneously.
//
// NOT thread-safe: use NNEvaluator to multiplex across threads.
// ----------------------------------------------------------------
class OpenCLEngine : public InferenceEngine {
public:
    OpenCLEngine();
    ~OpenCLEngine() override;

    void load_model(const std::string& path) override;

    std::pair<std::vector<float>, float>
    predict(const std::vector<float>& state) override;

    std::vector<std::pair<std::vector<float>, float>>
    predict_batch(const std::vector<std::vector<float>>& states) override;

    std::string backend_name()   const override { return "opencl"; }

private:
    // ── OpenCL objects ───────────────────────────────────────────
    cl_platform_id    platform_ = nullptr;
    cl_device_id      device_   = nullptr;
    cl_context        context_  = nullptr;
    cl_command_queue  queue_    = nullptr;
    cl_program        program_  = nullptr;

    // Kernels
    cl_kernel k_transpose_nchw_          = nullptr;
    cl_kernel k_conv3x3_sgemm_bn_        = nullptr;  // implicit GEMM: fused im2col+sgemm+BN
    cl_kernel k_conv1x1_bn_relu_reshape_ = nullptr;
    cl_kernel k_fc_bias_relu_            = nullptr;
    cl_kernel k_fc_bias_softmax_         = nullptr;
    cl_kernel k_fc_bias_tanh_            = nullptr;

    // ── Weight buffers on GPU ────────────────────────────────────
    struct ConvBNGPU {
        cl_mem weight   = nullptr;  // [C_out, C_in*kH*kW]  row-major
        cl_mem bn_scale = nullptr;  // [C_out] pre-fused BN scale
        cl_mem bn_bias  = nullptr;  // [C_out] pre-fused BN bias
        int c_out = 0, c_in = 0, k = 0;  // k = kernel size (1 or 3)
    };
    struct FCGPU {
        cl_mem weight = nullptr;  // [out_features, in_features]
        cl_mem bias   = nullptr;  // [out_features]
        int out_features = 0, in_features = 0;
    };

    ConvBNGPU              input_conv_gpu_;
    std::vector<ConvBNGPU> res_conv1_gpu_, res_conv2_gpu_;
    ConvBNGPU              policy_conv_gpu_, value_conv_gpu_;
    FCGPU                  policy_fc_gpu_, value_fc1_gpu_, value_fc2_gpu_;

    // ── Workspace buffers (pre-allocated for MAX_BATCH) ──────────
    static constexpr int MAX_BATCH = 256;

    cl_mem buf_flat_in_  = nullptr;  // [N * in_channels * H * W] staging for upload
    cl_mem buf_input_    = nullptr;  // [in_channels, N*H*W] channel-major
    cl_mem buf_main_     = nullptr;  // [num_filters,  N*H*W]
    cl_mem buf_temp_     = nullptr;  // same — first conv in residual block
    cl_mem buf_skip_     = nullptr;  // same — saved residual connection
    // buf_col_ removed: implicit GEMM computes im2col on-the-fly, no scratch buffer needed

    cl_mem buf_pol_feat_ = nullptr;  // [2*H*W, N]  policy features reshaped
    cl_mem buf_pol_out_  = nullptr;  // [action_size, N]
    cl_mem buf_val_feat_ = nullptr;  // [H*W, N]    value features reshaped
    cl_mem buf_val_h1_   = nullptr;  // [num_filters, N]
    cl_mem buf_val_out_  = nullptr;  // [1, N]

    int alloc_batch_ = 0;  // batch size for which workspace was allocated

    // ── Private helpers ──────────────────────────────────────────
    void init_opencl();
    void compile_kernels();
    void allocate_workspace(int batch);
    void free_workspace();
    void free_weights();

    // Upload a flat float vector to a new GPU buffer
    cl_mem upload(const std::vector<float>& data);

    // Release and zero a cl_mem handle
    static void release_buf(cl_mem& buf);

    // im2col + register-blocked SGEMM + BN + optional ReLU (2 kernel launches)
    // mode: 1 = BN+ReLU, 2 = BN+residual_add+ReLU
    void run_conv3x3(cl_mem input_buf, cl_mem output_buf,
                     const ConvBNGPU& conv, cl_mem residual_buf,
                     int N, int H, int W, int mode, bool relu);

    // Fused 1×1 conv + BN + ReLU + reshape to [C_out*HW, N]
    void run_conv1x1_bn_relu_reshape(cl_mem input_buf, cl_mem output_buf,
                                     const ConvBNGPU& conv,
                                     int N, int HW);

    // Fused FC + bias + optional ReLU (single kernel launch)
    void run_fc_bias_relu(cl_mem input_buf, cl_mem output_buf,
                          const FCGPU& fc, int N, bool relu);
};

}  // namespace minigo
#endif  // MINIGO_HAS_OPENCL
