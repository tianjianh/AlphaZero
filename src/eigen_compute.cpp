#ifdef MINIGO_HAS_EIGEN
#include "eigen_compute.h"
#include <cmath>
#include <cstring>
#include <iostream>
#include <algorithm>
#include <stdexcept>

namespace minigo {

// ================================================================
// EigenComputeContext — trivial for CPU
// ================================================================

std::unique_ptr<ComputeHandle>
EigenComputeContext::create_handle(const LoadedModel* model, int /*gpu_id*/, int /*max_batch_size*/) {
    return std::make_unique<EigenComputeHandle>(model);
}

// ================================================================
// EigenComputeHandle — loads weights from LoadedModel into Eigen matrices
// ================================================================

static Eigen::MatrixXf to_matrix(const std::vector<float>& data, int rows, int cols) {
    return Eigen::Map<const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(
        data.data(), rows, cols);
}

static Eigen::VectorXf to_vector(const std::vector<float>& data) {
    return Eigen::Map<const Eigen::VectorXf>(data.data(), (int)data.size());
}

EigenComputeHandle::EigenComputeHandle(const LoadedModel* model) {
    if (model->model_type == "vit")
        throw std::runtime_error("Eigen backend does not support ViT models. Use TensorRT.");
    board_size     = model->board_size;
    input_channels = model->input_channels;
    num_filters    = model->num_filters;
    num_res_blocks = model->num_res_blocks;

    auto load_conv = [](ConvBN& dst, const ConvBNWeights& src) {
        dst.c_out = src.c_out;
        dst.c_in  = src.c_in;
        dst.k     = src.k;
        dst.weight   = to_matrix(src.weight, src.c_out, src.c_in * src.k * src.k);
        dst.bn_scale = to_vector(src.bn_scale);
        dst.bn_bias  = to_vector(src.bn_bias);
    };

    auto load_fc = [](FC& dst, const FCWeights& src) {
        dst.out_features = src.out_features;
        dst.in_features  = src.in_features;
        dst.weight = to_matrix(src.weight, src.out_features, src.in_features);
        dst.bias   = to_vector(src.bias);
    };

    load_conv(input_conv_, model->input_conv);

    res_conv1_.resize(num_res_blocks);
    res_conv2_.resize(num_res_blocks);
    for (int i = 0; i < num_res_blocks; i++) {
        load_conv(res_conv1_[i], model->res_conv1[i]);
        load_conv(res_conv2_[i], model->res_conv2[i]);
    }

    load_conv(policy_conv_, model->policy_conv);
    load_conv(value_conv_,  model->value_conv);
    load_fc(policy_fc_, model->policy_fc);
    load_fc(value_fc1_, model->value_fc1);
    load_fc(value_fc2_, model->value_fc2);

    load_conv(score_conv_, model->score_conv);
    load_fc(score_fc1_, model->score_fc1);
    load_fc(score_fc2_, model->score_fc2);
}

// ================================================================
// Im2col for 3x3 convolution with padding=1, stride=1
// ================================================================

void EigenComputeHandle::im2col(const float* input, int C, int H, int W,
                                 int kH, int kW, int padH, int padW, float* col) {
    int patch_size = C * kH * kW;
    int hw = H * W;

    for (int ic = 0; ic < C; ic++) {
        for (int kh = 0; kh < kH; kh++) {
            for (int kw = 0; kw < kW; kw++) {
                int row = ic * kH * kW + kh * kW + kw;
                for (int oh = 0; oh < H; oh++) {
                    int ih = oh + kh - padH;
                    for (int ow = 0; ow < W; ow++) {
                        int iw = ow + kw - padW;
                        int c = oh * W + ow;
                        if (ih >= 0 && ih < H && iw >= 0 && iw < W)
                            col[row * hw + c] = input[ic * hw + ih * W + iw];
                        else
                            col[row * hw + c] = 0.0f;
                    }
                }
            }
        }
    }
}

// ================================================================
// Forward pass (single sample)
// ================================================================

EigenComputeHandle::Result
EigenComputeHandle::predict_single(const std::vector<float>& state) {
    int n  = board_size;
    int hw = n * n;
    int action_size = hw + 1;

    using MatF = Eigen::MatrixXf;
    using VecF = Eigen::VectorXf;

    MatF x = Eigen::Map<const MatF>(state.data(), input_channels, hw);

    // Im2col workspace
    int max_patch = num_filters * 9;
    im2col_buf_.resize(max_patch * hw);

    // Helper lambdas
    auto conv3x3 = [&](const MatF& input, MatF& output, const ConvBN& conv) {
        im2col(input.data(), conv.c_in, n, n, 3, 3, 1, 1, im2col_buf_.data());
        Eigen::Map<MatF> col(im2col_buf_.data(), conv.c_in * 9, hw);
        output.noalias() = conv.weight * col;
    };

    auto conv1x1 = [&](const MatF& input, MatF& output, const ConvBN& conv) {
        output.noalias() = conv.weight * input;
    };

    auto bn_relu = [&](MatF& x, const ConvBN& conv) {
        for (int c = 0; c < (int)x.rows(); c++)
            x.row(c) = x.row(c) * conv.bn_scale(c) +
                        Eigen::RowVectorXf::Constant(x.cols(), conv.bn_bias(c));
        x = x.cwiseMax(0.0f);
    };

    // Input conv + BN + ReLU
    MatF trunk;
    conv3x3(x, trunk, input_conv_);
    bn_relu(trunk, input_conv_);

    // Residual blocks
    for (int i = 0; i < num_res_blocks; i++) {
        MatF residual = trunk;
        MatF tmp;
        conv3x3(trunk, tmp, res_conv1_[i]);
        bn_relu(tmp, res_conv1_[i]);

        MatF tmp2;
        conv3x3(tmp, tmp2, res_conv2_[i]);
        // BN (no ReLU yet)
        for (int c = 0; c < (int)tmp2.rows(); c++)
            tmp2.row(c) = tmp2.row(c) * res_conv2_[i].bn_scale(c) +
                           Eigen::RowVectorXf::Constant(tmp2.cols(), res_conv2_[i].bn_bias(c));
        trunk = (tmp2 + residual).cwiseMax(0.0f);  // residual add + ReLU
    }

    // Policy head
    MatF p_conv;
    conv1x1(trunk, p_conv, policy_conv_);
    bn_relu(p_conv, policy_conv_);

    Eigen::Map<VecF> p_flat(p_conv.data(), policy_conv_.c_out * hw);
    VecF p_logits = policy_fc_.weight * p_flat + policy_fc_.bias;

    // Softmax
    float max_logit = p_logits.maxCoeff();
    VecF p_exp = (p_logits.array() - max_logit).exp();
    VecF p_probs = p_exp / p_exp.sum();

    std::vector<float> policy(action_size);
    for (int i = 0; i < action_size; i++)
        policy[i] = p_probs(i);

    // Value head
    MatF v_conv;
    conv1x1(trunk, v_conv, value_conv_);
    bn_relu(v_conv, value_conv_);

    Eigen::Map<VecF> v_flat(v_conv.data(), value_conv_.c_out * hw);
    VecF v_hidden = (value_fc1_.weight * v_flat + value_fc1_.bias).cwiseMax(0.0f);
    VecF v_out = value_fc2_.weight * v_hidden + value_fc2_.bias;
    float value = std::tanh(v_out(0));

    // Score head (same structure as value head)
    MatF s_conv;
    conv1x1(trunk, s_conv, score_conv_);
    bn_relu(s_conv, score_conv_);

    Eigen::Map<VecF> s_flat(s_conv.data(), score_conv_.c_out * hw);
    VecF s_hidden = (score_fc1_.weight * s_flat + score_fc1_.bias).cwiseMax(0.0f);
    VecF s_out = score_fc2_.weight * s_hidden + score_fc2_.bias;
    float score = std::tanh(s_out(0));

    return { policy, value, score };
}

// ================================================================
// Batch predict (loops over single predict)
// ================================================================

std::vector<EigenComputeHandle::Result>
EigenComputeHandle::predict_batch(const std::vector<std::vector<float>>& states) {
    std::vector<Result> results;
    results.reserve(states.size());
    for (auto& s : states)
        results.push_back(predict_single(s));
    return results;
}

}  // namespace minigo
#endif  // MINIGO_HAS_EIGEN
