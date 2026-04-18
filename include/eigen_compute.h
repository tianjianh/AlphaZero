#pragma once
#ifdef MINIGO_HAS_EIGEN

#include "compute_context.h"
#include "loaded_model.h"
#include <Eigen/Dense>
#include <memory>
#include <vector>

namespace minigo {

// ================================================================
// Eigen CPU backend - KataGo-style ResNet (SE + GPool + WLD value)
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
    explicit EigenComputeHandle(const LoadedModel* model);

    std::vector<Result>
    predict_batch(const std::vector<std::vector<float>>& states) override;

    // Nested weight wrappers exposed so free helpers in the .cpp can load them.
    using MatF = Eigen::MatrixXf;
    using VecF = Eigen::VectorXf;

    struct ConvBN {
        MatF weight;                 // [c_out, c_in * k * k]
        VecF bn_scale, bn_bias;      // fused BN
        int c_out = 0, c_in = 0, k = 0;
    };
    struct FC {
        MatF weight;                 // [out_features, in_features]
        VecF bias;
        int out_features = 0, in_features = 0;
    };
    struct Block {
        BlockKind kind = BlockKind::Plain;
        ConvBN conv1, conv2;
        FC se_fc1, se_fc2;           // kind == SE
        ConvBN pool_conv;            // kind == GPool
        FC pool_fc;                  // kind == GPool
    };

private:
    Result predict_single(const std::vector<float>& state);

    int board_rows_ = 0, board_cols_ = 0;
    int input_channels_ = 0, num_filters_ = 0;
    int action_size_ = 0;
    int value_head_size_ = 3;

    ConvBN input_conv_;
    std::vector<Block> blocks_;
    ConvBN policy_conv_, value_conv_;
    FC policy_fc_, value_fc1_, value_fc2_;

    // Scratch - resized lazily inside predict_single.
    std::vector<float> im2col_buf_;

    static void im2col(const float* input, int C, int H, int W,
                       int kH, int kW, int padH, int padW, float* col);
    static void conv3x3(const MatF& input, MatF& output, const ConvBN& conv,
                        int H, int W, std::vector<float>& scratch);
    static void conv1x1(const MatF& input, MatF& output, const ConvBN& conv);
    static void bn(MatF& x, const ConvBN& conv);
    static void bn_relu(MatF& x, const ConvBN& conv);
};

}  // namespace minigo
#endif  // MINIGO_HAS_EIGEN
