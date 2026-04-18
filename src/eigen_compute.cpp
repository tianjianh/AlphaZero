#ifdef MINIGO_HAS_EIGEN
#include "eigen_compute.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <stdexcept>

namespace minigo {

// ────────────────────────────────────────────────────────────────
// Helpers for pushing LoadedModel weights into Eigen matrices
// ────────────────────────────────────────────────────────────────

static Eigen::MatrixXf to_matrix(const std::vector<float>& data, int rows, int cols) {
    return Eigen::Map<const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(
        data.data(), rows, cols);
}

static Eigen::VectorXf to_vector(const std::vector<float>& data) {
    return Eigen::Map<const Eigen::VectorXf>(data.data(), static_cast<int>(data.size()));
}

static void load_conv(EigenComputeHandle::ConvBN& dst, const ConvBNWeights& src) {
    dst.c_out = src.c_out;
    dst.c_in = src.c_in;
    dst.k = src.k;
    dst.weight = to_matrix(src.weight, src.c_out, src.c_in * src.k * src.k);
    dst.bn_scale = to_vector(src.bn_scale);
    dst.bn_bias = to_vector(src.bn_bias);
}

static void load_fc(EigenComputeHandle::FC& dst, const FCWeights& src) {
    dst.out_features = src.out_features;
    dst.in_features = src.in_features;
    dst.weight = to_matrix(src.weight, src.out_features, src.in_features);
    dst.bias = to_vector(src.bias);
}

// ────────────────────────────────────────────────────────────────
// EigenComputeContext
// ────────────────────────────────────────────────────────────────

std::unique_ptr<ComputeHandle>
EigenComputeContext::create_handle(const LoadedModel* model,
                                   int /*gpu_id*/,
                                   int /*max_batch_size*/) {
    return std::make_unique<EigenComputeHandle>(model);
}

// ────────────────────────────────────────────────────────────────
// EigenComputeHandle - weight loading
// ────────────────────────────────────────────────────────────────

EigenComputeHandle::EigenComputeHandle(const LoadedModel* model) {
    if (!model) throw std::runtime_error("EigenComputeHandle: null model");

    board_rows_      = model->board_rows;
    board_cols_      = model->board_cols;
    input_channels_  = model->input_channels;
    num_filters_     = model->num_filters;
    action_size_     = model->action_size;
    value_head_size_ = model->value_head_size;

    load_conv(input_conv_, model->input_conv);

    blocks_.resize(model->blocks.size());
    for (size_t i = 0; i < model->blocks.size(); ++i) {
        const BlockWeights& src = model->blocks[i];
        Block& dst = blocks_[i];
        dst.kind = src.kind;
        load_conv(dst.conv1, src.conv1);
        load_conv(dst.conv2, src.conv2);
        if (src.kind == BlockKind::SE) {
            load_fc(dst.se_fc1, src.se_fc1);
            load_fc(dst.se_fc2, src.se_fc2);
        } else if (src.kind == BlockKind::GPool) {
            load_conv(dst.pool_conv, src.pool_conv);
            load_fc(dst.pool_fc, src.pool_fc);
        }
    }

    load_conv(policy_conv_, model->policy_conv);
    load_conv(value_conv_, model->value_conv);
    load_fc(policy_fc_, model->policy_fc);
    load_fc(value_fc1_, model->value_fc1);
    load_fc(value_fc2_, model->value_fc2);

    std::cout << "[eigen] handle ready: "
              << board_rows_ << "x" << board_cols_
              << " filters=" << num_filters_
              << " blocks=" << blocks_.size() << "\n";
}

// ────────────────────────────────────────────────────────────────
// Kernels
// ────────────────────────────────────────────────────────────────

void EigenComputeHandle::im2col(const float* input, int C, int H, int W,
                                int kH, int kW, int padH, int padW, float* col) {
    int hw = H * W;
    for (int ic = 0; ic < C; ++ic) {
        for (int kh = 0; kh < kH; ++kh) {
            for (int kw = 0; kw < kW; ++kw) {
                int row = ic * kH * kW + kh * kW + kw;
                for (int oh = 0; oh < H; ++oh) {
                    int ih = oh + kh - padH;
                    for (int ow = 0; ow < W; ++ow) {
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

void EigenComputeHandle::conv3x3(const MatF& input, MatF& output, const ConvBN& conv,
                                 int H, int W, std::vector<float>& scratch) {
    int patch = conv.c_in * 9;
    int hw = H * W;
    scratch.resize(static_cast<size_t>(patch) * hw);
    im2col(input.data(), conv.c_in, H, W, 3, 3, 1, 1, scratch.data());
    Eigen::Map<MatF> col(scratch.data(), patch, hw);
    output.noalias() = conv.weight * col;
}

void EigenComputeHandle::conv1x1(const MatF& input, MatF& output, const ConvBN& conv) {
    // 1x1 conv is just a GEMM on the channel dimension.
    output.noalias() = conv.weight * input;
}

void EigenComputeHandle::bn(MatF& x, const ConvBN& conv) {
    for (int c = 0; c < static_cast<int>(x.rows()); ++c) {
        x.row(c) = x.row(c) * conv.bn_scale(c) +
                   Eigen::RowVectorXf::Constant(x.cols(), conv.bn_bias(c));
    }
}

void EigenComputeHandle::bn_relu(MatF& x, const ConvBN& conv) {
    bn(x, conv);
    x = x.cwiseMax(0.0f);
}

// ────────────────────────────────────────────────────────────────
// Forward pass (single sample)
// ────────────────────────────────────────────────────────────────

EigenComputeHandle::Result
EigenComputeHandle::predict_single(const std::vector<float>& state) {
    const int H = board_rows_;
    const int W = board_cols_;
    const int hw = H * W;

    MatF trunk = Eigen::Map<const MatF>(state.data(), input_channels_, hw);

    {
        MatF tmp;
        conv3x3(trunk, tmp, input_conv_, H, W, im2col_buf_);
        bn_relu(tmp, input_conv_);
        trunk = std::move(tmp);
    }

    MatF tmp, tmp2, pool_branch;
    for (const Block& blk : blocks_) {
        MatF residual = trunk;

        // First 3x3 + BN + ReLU.
        conv3x3(trunk, tmp, blk.conv1, H, W, im2col_buf_);
        bn_relu(tmp, blk.conv1);

        if (blk.kind == BlockKind::GPool) {
            // Parallel pool branch from the block input.
            conv3x3(residual, pool_branch, blk.pool_conv, H, W, im2col_buf_);
            bn_relu(pool_branch, blk.pool_conv);

            int C = static_cast<int>(pool_branch.rows());
            VecF mean = pool_branch.rowwise().mean();
            VecF mx(C);
            for (int c = 0; c < C; ++c) mx(c) = pool_branch.row(c).maxCoeff();
            VecF stats(2 * C);
            stats.head(C) = mean;
            stats.tail(C) = mx;
            VecF bias = blk.pool_fc.weight * stats + blk.pool_fc.bias;

            // Broadcast bias per-channel into tmp before conv2.
            for (int c = 0; c < C; ++c)
                tmp.row(c).array() += bias(c);
        }

        // Second 3x3 + BN (no ReLU yet).
        conv3x3(tmp, tmp2, blk.conv2, H, W, im2col_buf_);
        bn(tmp2, blk.conv2);

        if (blk.kind == BlockKind::SE) {
            int C = static_cast<int>(tmp2.rows());
            VecF pooled = tmp2.rowwise().mean();                           // [C]
            VecF h = (blk.se_fc1.weight * pooled + blk.se_fc1.bias).cwiseMax(0.0f);
            VecF gate_logits = blk.se_fc2.weight * h + blk.se_fc2.bias;    // [C]
            VecF gate(C);
            for (int c = 0; c < C; ++c) gate(c) = 1.0f / (1.0f + std::exp(-gate_logits(c)));
            for (int c = 0; c < C; ++c) tmp2.row(c) *= gate(c);
        }

        trunk = (tmp2 + residual).cwiseMax(0.0f);
    }

    // ── Policy head ────────────────────────────────────────────
    MatF p_conv;
    conv1x1(trunk, p_conv, policy_conv_);
    bn_relu(p_conv, policy_conv_);
    Eigen::Map<VecF> p_flat(p_conv.data(), policy_conv_.c_out * hw);
    VecF p_logits = policy_fc_.weight * p_flat + policy_fc_.bias;
    float max_logit = p_logits.maxCoeff();
    VecF p_exp = (p_logits.array() - max_logit).exp();
    VecF p_probs = p_exp / p_exp.sum();

    std::vector<float> policy(action_size_);
    for (int i = 0; i < action_size_; ++i) policy[i] = p_probs(i);

    // ── Value head (WLD -> scalar P(win) - P(loss)) ────────────
    MatF v_conv;
    conv1x1(trunk, v_conv, value_conv_);
    bn_relu(v_conv, value_conv_);
    Eigen::Map<VecF> v_flat(v_conv.data(), value_conv_.c_out * hw);
    VecF v_hidden = (value_fc1_.weight * v_flat + value_fc1_.bias).cwiseMax(0.0f);
    VecF wdl = value_fc2_.weight * v_hidden + value_fc2_.bias;      // [value_head_size]

    float value_scalar = 0.0f;
    if (value_head_size_ >= 3) {
        float m = wdl.maxCoeff();
        float ew = std::exp(wdl(0) - m);
        float ed = std::exp(wdl(1) - m);
        float el = std::exp(wdl(2) - m);
        float z = ew + ed + el;
        value_scalar = (ew - el) / z;
    } else if (value_head_size_ == 1) {
        // Legacy scalar-tanh model (export compatibility).
        value_scalar = std::tanh(wdl(0));
    } else {
        float m = wdl.maxCoeff();
        VecF ex = (wdl.array() - m).exp();
        float z = ex.sum();
        value_scalar = (ex(0) - ex(value_head_size_ - 1)) / z;
    }

    Result r;
    r.policy = std::move(policy);
    r.value = value_scalar;
    r.score = 0.0f;
    r.score_sd = 0.0f;
    return r;
}

// ────────────────────────────────────────────────────────────────
// Batch predict (loops over single predict - simpler, still plenty fast
// for a CPU fallback used at test-time).
// ────────────────────────────────────────────────────────────────

std::vector<EigenComputeHandle::Result>
EigenComputeHandle::predict_batch(const std::vector<std::vector<float>>& states) {
    std::vector<Result> results;
    results.reserve(states.size());
    for (const auto& s : states) results.push_back(predict_single(s));
    return results;
}

}  // namespace minigo
#endif  // MINIGO_HAS_EIGEN
