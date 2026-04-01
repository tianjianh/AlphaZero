#pragma once

#include "compute_context.h"
#include "loaded_model.h"
#include <Eigen/Dense>
#include <vector>
#include <memory>

namespace minigo {

// ================================================================
// Eigen CPU backend — ComputeContext + ComputeHandle
// ================================================================

class EigenComputeContext : public ComputeContext {
public:
    EigenComputeContext() = default;

    std::unique_ptr<ComputeHandle>
    create_handle(const LoadedModel* model, int gpu_id, int max_batch_size) override;

    std::string backend_name() const override { return "eigen"; }
};

class EigenComputeHandle : public ComputeHandle {
public:
    EigenComputeHandle(const LoadedModel* model);

    std::vector<Result>
    predict_batch(const std::vector<std::vector<float>>& states) override;

private:
    Result predict_single(const std::vector<float>& state);

    // Model metadata
    int board_size, input_channels, num_filters, num_res_blocks;

    // Weight matrices (Eigen format, from LoadedModel CPU data)
    using MatF = Eigen::MatrixXf;
    using VecF = Eigen::VectorXf;

    struct ConvBN {
        MatF weight;
        VecF bn_scale, bn_bias;
        int c_out, c_in, k;
    };
    struct FC {
        MatF weight;
        VecF bias;
        int out_features, in_features;
    };

    ConvBN input_conv_;
    std::vector<ConvBN> res_conv1_, res_conv2_;
    ConvBN policy_conv_, value_conv_;
    FC policy_fc_, value_fc1_, value_fc2_;

    // Workspace
    std::vector<float> im2col_buf_;

    // Helpers
    void im2col(const float* input, int C, int H, int W, int kH, int kW,
                int padH, int padW, float* col);
};

}  // namespace minigo
