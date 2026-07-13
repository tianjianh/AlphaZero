#ifdef MINIGO_HAS_EIGEN
#include "eigen_compute.h"
#include "onnx_loader.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <unordered_map>

namespace minigo {

// ================================================================
// EigenComputeContext — thin factory; ViT is unsupported.
// ================================================================
std::unique_ptr<ComputeHandle>
EigenComputeContext::create_handle(const LoadedModel* model,
                                   int /*gpu_id*/, int /*max_batch_size*/) {
    // Interface contract: all three model formats (MiniGo resnet/vit
    // single-input, KataGo-V7 dual-input) are accepted here; the
    // handle constructor throws its own placeholder (TODO) for the
    // architectures whose CPU forward pass isn't implemented yet.
    return std::make_unique<EigenComputeHandle>(model);
}

// ================================================================
// Tensor lookup helpers
// ================================================================
namespace {

using TensorMap = std::unordered_map<std::string, const onnx_parser::OnnxTensor*>;

// All state_dict tensors are embedded with this prefix so they don't
// collide with whatever the ONNX optimizer renamed in the graph
// (folded conv weights, Identity passthroughs, etc.). See
// `_embed_state_dict` in scripts/export_onnx.py and tools/katago_to_onnx.py.
constexpr const char* SD_PREFIX = "_sd_";

const onnx_parser::OnnxTensor* find(const TensorMap& tm, const std::string& name) {
    auto it = tm.find(SD_PREFIX + name);
    if (it != tm.end()) return it->second;
    // Fallback to the un-prefixed name for older ONNX files.
    it = tm.find(name);
    return it == tm.end() ? nullptr : it->second;
}

const onnx_parser::OnnxTensor& must(const TensorMap& tm, const std::string& name) {
    auto* t = find(tm, name);
    if (!t) throw std::runtime_error("Eigen backend: missing tensor '" + name + "'");
    return *t;
}

EigenComputeHandle::MatF row_major_matrix(const std::vector<float>& data,
                                           int rows, int cols) {
    if ((int)data.size() != rows * cols)
        throw std::runtime_error("matrix shape mismatch (" +
                                 std::to_string(data.size()) + " vs " +
                                 std::to_string(rows * cols) + ")");
    return Eigen::Map<const EigenComputeHandle::MatF>(data.data(), rows, cols);
}

Eigen::VectorXf to_vector(const std::vector<float>& data) {
    return Eigen::Map<const Eigen::VectorXf>(data.data(), (int)data.size());
}

// Pre-fuse PyTorch BN parameters: scale = gamma/sqrt(var+eps),
// bias = beta - gamma*mean/sqrt(var+eps).
//
// Epsilon defaults differ by training framework:
//   * MiniGo (PyTorch nn.BatchNorm2d default): 1e-5
//   * KataGo binary format (warm_init_from_katago):     1e-20
// The embedded state_dict carries running_mean/var but NOT epsilon, so the
// caller passes it explicitly based on the model format.
void load_bn_params(EigenComputeHandle::BN& dst, const TensorMap& tm,
                    const std::string& prefix, float eps) {
    auto& gamma = must(tm, prefix + ".weight");
    auto& beta  = must(tm, prefix + ".bias");
    auto& mean  = must(tm, prefix + ".running_mean");
    auto& var   = must(tm, prefix + ".running_var");
    auto g = gamma.get_floats();
    auto b = beta.get_floats();
    auto m = mean.get_floats();
    auto v = var.get_floats();
    int n = (int)g.size();
    Eigen::VectorXf scale(n), bias(n);
    for (int i = 0; i < n; ++i) {
        float inv_std = 1.0f / std::sqrt(v[i] + eps);
        scale(i) = g[i] * inv_std;
        bias(i)  = b[i] - g[i] * m[i] * inv_std;
    }
    dst.scale = std::move(scale);
    dst.bias  = std::move(bias);
    dst.channels = n;
}

void load_convbn(EigenComputeHandle::ConvBN& dst, const TensorMap& tm,
                 const std::string& conv_name, const std::string& bn_prefix,
                 float eps) {
    auto& w = must(tm, conv_name);
    if (w.dims.size() != 4)
        throw std::runtime_error("expected 4D conv weight: " + conv_name);
    int co = (int)w.dims[0];
    int ci = (int)w.dims[1];
    int kh = (int)w.dims[2];
    int kw = (int)w.dims[3];
    if (kh != kw) throw std::runtime_error("non-square kernel: " + conv_name);
    dst.c_out = co; dst.c_in = ci; dst.k = kh;
    dst.weight = row_major_matrix(w.get_floats(), co, ci * kh * kw);

    EigenComputeHandle::BN bn;
    load_bn_params(bn, tm, bn_prefix, eps);
    if (bn.channels != co)
        throw std::runtime_error("BN/conv channel mismatch: " + bn_prefix);
    dst.bn_scale = std::move(bn.scale);
    dst.bn_bias  = std::move(bn.bias);
}

void load_conv(EigenComputeHandle::Conv& dst, const TensorMap& tm,
               const std::string& conv_name, bool require_bias) {
    auto& w = must(tm, conv_name);
    if (w.dims.size() != 4)
        throw std::runtime_error("expected 4D conv weight: " + conv_name);
    int co = (int)w.dims[0];
    int ci = (int)w.dims[1];
    int kh = (int)w.dims[2];
    int kw = (int)w.dims[3];
    if (kh != kw) throw std::runtime_error("non-square kernel: " + conv_name);
    dst.c_out = co; dst.c_in = ci; dst.k = kh;
    dst.weight = row_major_matrix(w.get_floats(), co, ci * kh * kw);

    // Conv bias: PyTorch stores as <prefix>.bias when bias=True.
    std::string bias_name = conv_name;
    auto pos = bias_name.rfind(".weight");
    if (pos != std::string::npos) bias_name.replace(pos, 7, ".bias");
    auto* b = find(tm, bias_name);
    if (b) {
        dst.bias = to_vector(b->get_floats());
        dst.has_bias = true;
    } else if (require_bias) {
        throw std::runtime_error("expected bias on conv: " + conv_name);
    } else {
        dst.bias = Eigen::VectorXf::Zero(co);
        dst.has_bias = false;
    }
}

void load_fc(EigenComputeHandle::FC& dst, const TensorMap& tm,
             const std::string& weight_name, bool with_bias) {
    auto& w = must(tm, weight_name);
    if (w.dims.size() != 2)
        throw std::runtime_error("expected 2D fc weight: " + weight_name);
    int out = (int)w.dims[0];
    int in  = (int)w.dims[1];
    dst.out_features = out;
    dst.in_features  = in;
    dst.weight = row_major_matrix(w.get_floats(), out, in);
    std::string bias_name = weight_name;
    auto pos = bias_name.rfind(".weight");
    if (pos != std::string::npos) bias_name.replace(pos, 7, ".bias");
    auto* b = find(tm, bias_name);
    if (b) {
        dst.bias = to_vector(b->get_floats());
        dst.has_bias = true;
    } else {
        dst.bias = Eigen::VectorXf::Zero(out);
        dst.has_bias = with_bias;  // record caller's expectation
    }
}

void load_gpool_head(EigenComputeHandle::GPoolHead& head, const TensorMap& tm,
                     const std::string& prefix, float eps) {
    load_convbn(head.conv, tm, prefix + ".conv.weight", prefix + ".bn", eps);
    load_fc(head.fc1, tm, prefix + ".fc1.weight", true);
    load_fc(head.fc2, tm, prefix + ".fc2.weight", true);
}

}  // namespace

// ================================================================
// EigenComputeHandle — load weights from the model's ONNX file
// ================================================================
EigenComputeHandle::EigenComputeHandle(const LoadedModel* model) {
    if (model->model_type == "vit")
        throw std::runtime_error(
            "Eigen backend: ViT forward pass is a placeholder (TODO) — "
            "resnet and KataGo-V7 models run on CPU; use TensorRT for ViT.");

    format_                = model->format;
    board_size_            = model->board_size;
    input_channels_        = model->input_channels;
    input_global_channels_ = model->input_global_channels;
    num_filters_           = model->num_filters;
    num_blocks_            = model->num_res_blocks;

    // Re-parse the ONNX file to get all initializer tensors by name.
    // For MiniGo this includes the embedded state_dict (export_onnx.py
    // calls `_embed_state_dict` after the optimized graph is built).
    // For KataGo the initializers are exactly the PyTorch state_dict
    // tensors (katago_to_onnx.py exports the raw model with no fusion).
    auto tensors = onnx_parser::parse_onnx_file(model->model_path);
    TensorMap tm;
    tm.reserve(tensors.size());
    for (const auto& t : tensors) tm[t.name] = &t;

    if (format_ == ModelFormat::KataGo) {
        // ── KataGo path ────────────────────────────────────────
        // Graph-op-based detection: a plain "Mish" byte-scan misses
        // opset<=17 exports, where each Mish is decomposed into
        // Softplus+Tanh+Mul and the literal never appears.
        katago_use_mish_ = onnx_parser::graph_uses_mish(model->model_path);
        // KataGo's binary format stores BN epsilon (1e-20 in stock kata1
        // networks) and katago_arch.py copies it into PyTorch BN modules
        // before export. The embedded state_dict carries running stats
        // but not eps, so use the upstream value here.
        const float K_EPS = 1e-20f;

        // Stem
        load_conv(k_initial_conv_, tm, "stem.initial_conv.weight", false);
        load_fc(k_initial_matmul_, tm, "stem.initial_matmul.weight", false);
        if (k_initial_matmul_.in_features != input_global_channels_)
            throw std::runtime_error(
                "KataGo stem.initial_matmul: expected in=" +
                std::to_string(input_global_channels_) + " got " +
                std::to_string(k_initial_matmul_.in_features));
        num_filters_ = k_initial_conv_.c_out;

        // Trunk
        int b = 0;
        while (find(tm, "blocks." + std::to_string(b) + ".pre_bn.weight")) {
            std::string pre = "blocks." + std::to_string(b);
            Block blk;
            bool has_gpool = (find(tm, pre + ".gpool_conv.weight") != nullptr);
            blk.kind = has_gpool ? BlockKind::KataGPool : BlockKind::KataRegular;
            load_bn_params(blk.k_pre_bn, tm, pre + ".pre_bn", K_EPS);
            load_conv(blk.k_regular_conv, tm, pre + ".regular_conv.weight", false);
            load_bn_params(blk.k_mid_bn, tm, pre + ".mid_bn", K_EPS);
            load_conv(blk.k_final_conv, tm, pre + ".final_conv.weight", false);
            if (has_gpool) {
                load_conv(blk.k_gpool_conv, tm, pre + ".gpool_conv.weight", false);
                load_bn_params(blk.k_gpool_bn, tm, pre + ".gpool_bn", K_EPS);
                load_fc(blk.k_gpool_to_bias, tm, pre + ".gpool_to_bias.weight", false);
                blk.k_gpool_channels = blk.k_gpool_conv.c_out;
            }
            blocks_.push_back(std::move(blk));
            ++b;
        }
        num_blocks_ = (int)blocks_.size();
        if (num_blocks_ == 0)
            throw std::runtime_error("KataGo: no trunk blocks found");

        // Trunk tip
        load_bn_params(k_trunk_tip_bn_, tm, "trunk_tip_bn", K_EPS);

        // Policy head
        load_conv(k_p1_conv_, tm, "policy_head.p1_conv.weight", false);
        load_conv(k_g1_conv_, tm, "policy_head.g1_conv.weight", false);
        load_bn_params(k_g1_bn_, tm, "policy_head.g1_bn", K_EPS);
        load_fc(k_gpool_to_bias_, tm, "policy_head.gpool_to_bias.weight", false);
        load_bn_params(k_p1_bn_, tm, "policy_head.p1_bn", K_EPS);
        load_conv(k_p2_conv_, tm, "policy_head.p2_conv.weight", false);
        k_p2_out_channels_ = k_p2_conv_.c_out;
        load_fc(k_gpool_to_pass_, tm, "policy_head.gpool_to_pass.weight", false);

        // Value head
        load_conv(k_v1_conv_, tm, "value_head.v1_conv.weight", false);
        load_bn_params(k_v1_bn_, tm, "value_head.v1_bn", K_EPS);
        load_fc(k_v2_mul_, tm, "value_head.v2_mul.weight", false);
        load_fc(k_v3_mul_, tm, "value_head.v3_mul.weight", false);
        load_fc(k_sv3_mul_, tm, "value_head.sv3_mul.weight", false);
        // v2/v3/sv3 biases are stored as nn.Parameter, not nn.Linear.bias —
        // PyTorch serialises them at the same flat names "<head>.v2_bias", etc.
        k_v2_bias_  = to_vector(must(tm, "value_head.v2_bias").get_floats());
        k_v3_bias_  = to_vector(must(tm, "value_head.v3_bias").get_floats());
        k_sv3_bias_ = to_vector(must(tm, "value_head.sv3_bias").get_floats());
        load_conv(k_v_ownership_conv_, tm, "value_head.v_ownership_conv.weight", false);

        std::cout << "Eigen backend: KataGo, blocks=" << num_blocks_
                  << " trunk_c=" << num_filters_
                  << " act=" << (katago_use_mish_ ? "mish" : "relu") << "\n";

    } else {
        // ── MiniGo path ────────────────────────────────────────
        // PyTorch nn.BatchNorm2d default eps; matches scripts/model.py.
        const float MG_EPS = 1e-5f;
        load_convbn(mg_input_conv_, tm, "input_conv.weight", "input_bn", MG_EPS);
        if (mg_input_conv_.c_out != num_filters_)
            num_filters_ = mg_input_conv_.c_out;

        // Trunk: alternating SE / GPool blocks. Even index → SE, odd → GPool.
        int b = 0;
        while (find(tm, "trunk." + std::to_string(b) + ".conv2.weight")) {
            std::string pre = "trunk." + std::to_string(b);
            Block blk;
            // SE blocks have `conv1.weight`; GPool blocks have `conv_main.weight`.
            bool is_se = (find(tm, pre + ".conv1.weight") != nullptr);
            if (is_se) {
                blk.kind = BlockKind::MiniGoSE;
                load_convbn(blk.se_conv1, tm, pre + ".conv1.weight", pre + ".bn1", MG_EPS);
                load_convbn(blk.se_conv2, tm, pre + ".conv2.weight", pre + ".bn2", MG_EPS);
                load_fc(blk.se_fc1, tm, pre + ".se.fc1.weight", true);
                load_fc(blk.se_fc2, tm, pre + ".se.fc2.weight", true);
            } else {
                blk.kind = BlockKind::MiniGoGPool;
                load_convbn(blk.mg_conv_main, tm, pre + ".conv_main.weight", pre + ".bn_main", MG_EPS);
                load_convbn(blk.mg_conv_pool, tm, pre + ".conv_pool.weight", pre + ".bn_pool", MG_EPS);
                load_fc(blk.mg_pool_fc, tm, pre + ".pool_fc.weight", true);
                load_convbn(blk.mg_conv2, tm, pre + ".conv2.weight", pre + ".bn2", MG_EPS);
                blk.mg_pool_channels = blk.mg_conv_pool.c_out;
            }
            blocks_.push_back(std::move(blk));
            ++b;
        }
        num_blocks_ = (int)blocks_.size();
        if (num_blocks_ == 0)
            throw std::runtime_error("MiniGo: no trunk blocks found");

        // Heads
        load_convbn(mg_policy_conv_, tm, "policy_conv.weight", "policy_bn", MG_EPS);
        load_fc(mg_policy_fc_, tm, "policy_fc.weight", true);

        load_gpool_head(mg_value_head_,        tm, "value_head", MG_EPS);
        load_gpool_head(mg_score_mean_head_,   tm, "score_mean_head", MG_EPS);
        load_gpool_head(mg_score_stdev_head_,  tm, "score_stdev_head", MG_EPS);

        load_conv(mg_ownership_conv_, tm, "ownership_conv.weight", false);

        std::cout << "Eigen backend: MiniGo ResNet, blocks=" << num_blocks_
                  << " filters=" << num_filters_ << "\n";
    }
}

// ================================================================
// Batched im2col, stride=1, square kernel of size K, padding=K/2.
//
// Input  in  [C, B*HW] row-major: in[c * B*HW + b * HW + (h*W + w)]
// Output out [C*K*K, B*HW] row-major: out[(ic*K*K + kh*K + kw) * B*HW + b*HW + (oh*W+ow)]
//             = in[ic, b, oh+kh-pad, ow+kw-pad]   (or 0 outside the board)
//
// Producing one [C*K*K, B*HW] matrix lets the conv collapse to a single
// GEMM `weight × col` whose right operand grows with the batch — large
// enough for Eigen's OpenMP path to actually spread across cores.
// ================================================================
void EigenComputeHandle::im2col_batched(const float* in, int C, int H, int W,
                                        int K, int pad, int B, float* out) {
    const int HW  = H * W;
    const int BHW = B * HW;
    for (int ic = 0; ic < C; ++ic) {
        const float* in_c = in + (size_t)ic * BHW;
        for (int kh = 0; kh < K; ++kh) {
            for (int kw = 0; kw < K; ++kw) {
                int row = ic * K * K + kh * K + kw;
                float* out_row = out + (size_t)row * BHW;
                for (int b = 0; b < B; ++b) {
                    const float* in_cb = in_c + (size_t)b * HW;
                    float* out_cb = out_row + (size_t)b * HW;
                    for (int oh = 0; oh < H; ++oh) {
                        int ih = oh + kh - pad;
                        if (ih < 0 || ih >= H) {
                            std::memset(out_cb + oh * W, 0, sizeof(float) * W);
                            continue;
                        }
                        for (int ow = 0; ow < W; ++ow) {
                            int iw = ow + kw - pad;
                            if (iw >= 0 && iw < W)
                                out_cb[oh * W + ow] = in_cb[ih * W + iw];
                            else
                                out_cb[oh * W + ow] = 0.0f;
                        }
                    }
                }
            }
        }
    }
}

namespace {

// In-place ReLU: x = max(x, 0).
inline void relu_inplace(EigenComputeHandle::MatF& x) {
    x = x.cwiseMax(0.0f);
}

// In-place Mish: x * tanh(softplus(x)).
inline void mish_inplace(EigenComputeHandle::MatF& x) {
    x = x.array() * (((x.array().max(-30.0f)).exp() + 1.0f).log()).tanh();
}

inline float softplus(float x) {
    if (x > 20.0f) return x;
    return std::log1p(std::exp(x));
}

// BN(x): channel-wise affine using pre-fused (scale, bias).
inline void bn_inplace(EigenComputeHandle::MatF& x,
                       const Eigen::VectorXf& scale,
                       const Eigen::VectorXf& bias) {
    int C = (int)x.rows();
    int W = (int)x.cols();
    for (int c = 0; c < C; ++c) {
        x.row(c) = x.row(c) * scale(c) +
                   Eigen::RowVectorXf::Constant(W, bias(c));
    }
}

}  // namespace

void EigenComputeHandle::apply_act_inplace(MatF& x) {
    if (format_ == ModelFormat::KataGo && katago_use_mish_) mish_inplace(x);
    else relu_inplace(x);
}

// ================================================================
// MiniGo forward pass — batched (whole drained batch in one shot)
// ================================================================
std::vector<EigenComputeHandle::Result>
EigenComputeHandle::predict_batch_minigo(const std::vector<std::vector<float>>& states) {
    const int B = (int)states.size();
    if (B == 0) return {};

    const int n   = board_size_;
    const int hw  = n * n;
    const int BHW = B * hw;
    const int action_size = hw + 1;
    const int Cin = input_channels_;

    if ((int)states[0].size() != Cin * hw)
        throw std::runtime_error("MiniGo state size mismatch (" +
            std::to_string(states[0].size()) + " vs " +
            std::to_string(Cin * hw) + ")");

    // Pack input: [Cin, B*HW] row-major.  Each state is C contiguous
    // HW-sized rows already, so per-channel we memcpy hw floats from
    // s[c*hw .. (c+1)*hw] into row c at column-offset b*hw.
    MatF x(Cin, BHW);
    for (int b = 0; b < B; ++b) {
        const auto& s = states[b];
        for (int c = 0; c < Cin; ++c)
            std::memcpy(&x(c, b * hw), &s[c * hw], hw * sizeof(float));
    }

    // Per-(channel, batch) reductions over each batch's HW slice. Output
    // is a [C, B] matrix used as the right operand of FC matmuls so the
    // FC step is a real GEMM `weight @ stats` (not B independent matvecs).
    auto reduce_mean = [&](const MatF& m) -> MatF {
        int C = (int)m.rows();
        MatF out(C, B);
        for (int c = 0; c < C; ++c)
            for (int b = 0; b < B; ++b)
                out(c, b) = m.row(c).segment(b * hw, hw).mean();
        return out;
    };
    auto reduce_max = [&](const MatF& m) -> MatF {
        int C = (int)m.rows();
        MatF out(C, B);
        for (int c = 0; c < C; ++c)
            for (int b = 0; b < B; ++b)
                out(c, b) = m.row(c).segment(b * hw, hw).maxCoeff();
        return out;
    };
    auto reduce_std = [&](const MatF& m) -> MatF {  // unbiased (PyTorch default)
        int C = (int)m.rows();
        const int denom = std::max(1, hw - 1);
        MatF out(C, B);
        for (int c = 0; c < C; ++c) {
            for (int b = 0; b < B; ++b) {
                auto seg = m.row(c).segment(b * hw, hw);
                float mean = seg.mean();
                float ss = (seg.array() - mean).square().sum();
                out(c, b) = std::sqrt(ss / (float)denom);
            }
        }
        return out;
    };
    auto add_per_batch = [&](MatF& dst, const MatF& src_cb) {
        int C = (int)dst.rows();
        for (int c = 0; c < C; ++c)
            for (int b = 0; b < B; ++b)
                dst.row(c).segment(b * hw, hw).array() += src_cb(c, b);
    };
    auto mul_per_batch = [&](MatF& dst, const MatF& src_cb) {
        int C = (int)dst.rows();
        for (int c = 0; c < C; ++c)
            for (int b = 0; b < B; ++b)
                dst.row(c).segment(b * hw, hw).array() *= src_cb(c, b);
    };

    auto conv_bn = [&](const ConvBN& cv, const MatF& in, MatF& out) {
        int patch = cv.c_in * cv.k * cv.k;
        int pad = cv.k / 2;
        if ((int)im2col_buf_.size() < (size_t)patch * BHW)
            im2col_buf_.resize((size_t)patch * BHW);
        im2col_batched(in.data(), cv.c_in, n, n, cv.k, pad, B, im2col_buf_.data());
        Eigen::Map<const MatF> col(im2col_buf_.data(), patch, BHW);
        out.noalias() = cv.weight * col;                // ← the big GEMM
        bn_inplace(out, cv.bn_scale, cv.bn_bias);
    };
    auto conv_only = [&](const Conv& cv, const MatF& in, MatF& out) {
        int patch = cv.c_in * cv.k * cv.k;
        int pad = cv.k / 2;
        if ((int)im2col_buf_.size() < (size_t)patch * BHW)
            im2col_buf_.resize((size_t)patch * BHW);
        im2col_batched(in.data(), cv.c_in, n, n, cv.k, pad, B, im2col_buf_.data());
        Eigen::Map<const MatF> col(im2col_buf_.data(), patch, BHW);
        out.noalias() = cv.weight * col;
        if (cv.has_bias)
            for (int c = 0; c < cv.c_out; ++c)
                out.row(c).array() += cv.bias(c);
    };

    // Stem
    MatF trunk;
    conv_bn(mg_input_conv_, x, trunk);
    relu_inplace(trunk);

    // Trunk blocks
    for (auto& blk : blocks_) {
        if (blk.kind == BlockKind::MiniGoSE) {
            MatF residual = trunk;
            MatF tmp;
            conv_bn(blk.se_conv1, trunk, tmp); relu_inplace(tmp);
            MatF tmp2;
            conv_bn(blk.se_conv2, tmp, tmp2);
            // SE: per-(c,b) GAP → fc1 → ReLU → fc2 → sigmoid → broadcast multiply
            MatF gap = reduce_mean(tmp2);                                      // [C, B]
            MatF h; h.noalias() = blk.se_fc1.weight * gap;                     // [hidden, B]
            h.colwise() += blk.se_fc1.bias;
            relu_inplace(h);
            MatF s; s.noalias() = blk.se_fc2.weight * h;                       // [C, B]
            s.colwise() += blk.se_fc2.bias;
            s = (1.0f / (1.0f + (-s.array()).exp())).matrix();
            mul_per_batch(tmp2, s);
            trunk = (tmp2 + residual).cwiseMax(0.0f);
        } else {  // MiniGoGPool
            MatF residual = trunk;
            MatF main;  conv_bn(blk.mg_conv_main, trunk, main);
            MatF pool;  conv_bn(blk.mg_conv_pool, trunk, pool);
            relu_inplace(pool);
            int pc = blk.mg_pool_channels;
            MatF stats(2 * pc, B);
            stats.topRows(pc)    = reduce_mean(pool);
            stats.bottomRows(pc) = reduce_max(pool);
            MatF bias_per_batch; bias_per_batch.noalias() = blk.mg_pool_fc.weight * stats;  // [F, B]
            bias_per_batch.colwise() += blk.mg_pool_fc.bias;
            add_per_batch(main, bias_per_batch);
            relu_inplace(main);
            MatF out2; conv_bn(blk.mg_conv2, main, out2);
            trunk = (out2 + residual).cwiseMax(0.0f);
        }
    }

    // ── Heads ───────────────────────────────────────────────
    auto run_gpool_head = [&](const GPoolHead& head) -> MatF {
        MatF h; conv_bn(head.conv, trunk, h); relu_inplace(h);
        int hc = head.conv.c_out;
        MatF pooled(3 * hc, B);
        pooled.topRows(hc)         = reduce_mean(h);
        pooled.middleRows(hc, hc)  = reduce_max(h);
        pooled.bottomRows(hc)      = reduce_std(h);
        MatF h1; h1.noalias() = head.fc1.weight * pooled;   // [mlp, B]
        h1.colwise() += head.fc1.bias;
        relu_inplace(h1);
        MatF out; out.noalias() = head.fc2.weight * h1;     // [out, B]
        out.colwise() += head.fc2.bias;
        return out;
    };

    // Policy: conv→bn→ReLU→flatten-per-batch→fc.  Flatten reshapes
    // [c_out=2, B*HW] to [c_out*HW, B] so the FC is one [act, 2*HW]·[2*HW, B] GEMM.
    MatF p; conv_bn(mg_policy_conv_, trunk, p); relu_inplace(p);
    int p_in = mg_policy_conv_.c_out * hw;
    MatF P_flat(p_in, B);
    // Row-major MatF: column b is strided through memory (stride = B), so
    // element-wise assignment is required — memcpy along (c*hw, b) would
    // overwrite neighbouring batches.
    for (int b = 0; b < B; ++b)
        for (int c = 0; c < mg_policy_conv_.c_out; ++c)
            for (int i = 0; i < hw; ++i)
                P_flat(c * hw + i, b) = p(c, b * hw + i);
    MatF p_logits; p_logits.noalias() = mg_policy_fc_.weight * P_flat;
    p_logits.colwise() += mg_policy_fc_.bias;

    MatF v_logits = run_gpool_head(mg_value_head_);         // [3, B]
    MatF score_mean = run_gpool_head(mg_score_mean_head_);  // [1, B]
    MatF score_sd_raw = run_gpool_head(mg_score_stdev_head_); // [1, B]

    // Ownership: 1x1 conv (with bias) → sigmoid
    MatF own; conv_only(mg_ownership_conv_, trunk, own);    // [1, B*HW]

    // Pack per-batch results.  Row-major MatF stores rows contiguously, so a
    // column (the per-batch slice we want) is at stride B — read element-wise.
    std::vector<Result> results(B);
    for (int b = 0; b < B; ++b) {
        Result& r = results[b];
        r.policy.resize(action_size);
        for (int i = 0; i < action_size; ++i) r.policy[i] = p_logits(i, b);

        float vmax = std::max({v_logits(0, b), v_logits(1, b), v_logits(2, b)});
        float ew = std::exp(v_logits(0, b) - vmax);
        float el = std::exp(v_logits(1, b) - vmax);
        float ed = std::exp(v_logits(2, b) - vmax);
        float sum = ew + el + ed;
        r.value    = (ew - el) / sum;
        r.score    = score_mean(0, b);
        r.score_sd = softplus(score_sd_raw(0, b));

        r.ownership.resize(hw);
        for (int i = 0; i < hw; ++i)
            r.ownership[i] = 1.0f / (1.0f + std::exp(-own(0, b * hw + i)));
    }
    return results;
}

// ================================================================
// KataGo forward pass — batched (whole drained batch in one shot)
// ================================================================
std::vector<EigenComputeHandle::Result>
EigenComputeHandle::predict_batch_katago(const std::vector<std::vector<float>>& states) {
    const int B = (int)states.size();
    if (B == 0) return {};

    const int n   = board_size_;
    const int hw  = n * n;
    const int BHW = B * hw;
    const int action_size = hw + 1;
    const int Cin = input_channels_;
    const int Gin = input_global_channels_;
    const int sp_per = Cin * hw;
    const int gl_per = Gin;

    if ((int)states[0].size() != sp_per + gl_per)
        throw std::runtime_error("KataGo state size mismatch (" +
            std::to_string(states[0].size()) + " vs " +
            std::to_string(sp_per + gl_per) + ")");

    // Pack inputs.  Spatial: [Cin, B*HW] row-major.  Global: [Gin, B] (each
    // batch element's global vector is a column → matmul gives a per-batch
    // bias for the stem with one GEMM).
    MatF x(Cin, BHW);
    MatF G(Gin, B);
    for (int b = 0; b < B; ++b) {
        const auto& s = states[b];
        for (int c = 0; c < Cin; ++c)
            std::memcpy(&x(c, b * hw), &s[c * hw], hw * sizeof(float));
        for (int c = 0; c < Gin; ++c)
            G(c, b) = s[sp_per + c];
    }

    auto conv_only = [&](const Conv& cv, const MatF& in, MatF& out) {
        int patch = cv.c_in * cv.k * cv.k;
        int pad = cv.k / 2;
        if ((int)im2col_buf_.size() < (size_t)patch * BHW)
            im2col_buf_.resize((size_t)patch * BHW);
        im2col_batched(in.data(), cv.c_in, n, n, cv.k, pad, B, im2col_buf_.data());
        Eigen::Map<const MatF> col(im2col_buf_.data(), patch, BHW);
        out.noalias() = cv.weight * col;                // ← the big GEMM
        if (cv.has_bias)
            for (int c = 0; c < cv.c_out; ++c)
                out.row(c).array() += cv.bias(c);
    };

    auto act = [&](MatF& m) { apply_act_inplace(m); };

    auto add_per_batch = [&](MatF& dst, const MatF& src_cb) {
        int C = (int)dst.rows();
        for (int c = 0; c < C; ++c)
            for (int b = 0; b < B; ++b)
                dst.row(c).segment(b * hw, hw).array() += src_cb(c, b);
    };

    // KataGo gpool stats: [mean, mean*(sqrt(N)-14)*0.1, max] per channel,
    // computed per-batch → output [3*C, B] (column = batch).
    auto gpool_stats = [&](const MatF& h) -> MatF {
        int C = (int)h.rows();
        float sqrt_div = std::sqrt((float)hw);
        float scale = (sqrt_div - 14.0f) * 0.1f;
        MatF out(3 * C, B);
        for (int c = 0; c < C; ++c) {
            for (int b = 0; b < B; ++b) {
                auto seg = h.row(c).segment(b * hw, hw);
                float mean = seg.mean();
                float mx   = seg.maxCoeff();
                out(c, b)         = mean;
                out(C + c, b)     = mean * scale;
                out(2 * C + c, b) = mx;
            }
        }
        return out;
    };
    // Value head pool: [mean, mean*a, mean*b] per channel → [3*C, B].
    auto vh_stats = [&](const MatF& h) -> MatF {
        int C = (int)h.rows();
        float sqrt_div = std::sqrt((float)hw);
        float a = (sqrt_div - 14.0f) * 0.1f;
        float b_coef = (sqrt_div - 14.0f) * (sqrt_div - 14.0f) * 0.01f - 0.1f;
        MatF out(3 * C, B);
        for (int c = 0; c < C; ++c) {
            for (int b = 0; b < B; ++b) {
                float mean = h.row(c).segment(b * hw, hw).mean();
                out(c, b)         = mean;
                out(C + c, b)     = mean * a;
                out(2 * C + c, b) = mean * b_coef;
            }
        }
        return out;
    };

    // Stem: trunk = initial_conv(spatial) + initial_matmul(global) broadcast.
    MatF trunk;
    conv_only(k_initial_conv_, x, trunk);                              // [trunk_c, B*HW]
    MatF stem_bias; stem_bias.noalias() = k_initial_matmul_.weight * G; // [trunk_c, B]
    add_per_batch(trunk, stem_bias);

    // Trunk blocks (KataGo pre-activation).
    for (auto& blk : blocks_) {
        MatF skip = trunk;
        MatF h = trunk;
        bn_inplace(h, blk.k_pre_bn.scale, blk.k_pre_bn.bias);
        act(h);

        if (blk.kind == BlockKind::KataRegular) {
            MatF reg; conv_only(blk.k_regular_conv, h, reg);
            bn_inplace(reg, blk.k_mid_bn.scale, blk.k_mid_bn.bias);
            act(reg);
            MatF out; conv_only(blk.k_final_conv, reg, out);
            trunk = skip + out;
        } else {  // KataGPool
            MatF reg; conv_only(blk.k_regular_conv, h, reg);
            MatF gp;  conv_only(blk.k_gpool_conv,  h, gp);
            bn_inplace(gp, blk.k_gpool_bn.scale, blk.k_gpool_bn.bias);
            act(gp);
            MatF gs = gpool_stats(gp);                                  // [3*gpool_c, B]
            MatF bias_per_batch; bias_per_batch.noalias() =
                blk.k_gpool_to_bias.weight * gs;                        // [regular_c, B]
            add_per_batch(reg, bias_per_batch);
            bn_inplace(reg, blk.k_mid_bn.scale, blk.k_mid_bn.bias);
            act(reg);
            MatF out; conv_only(blk.k_final_conv, reg, out);
            trunk = skip + out;
        }
    }

    // Trunk tip
    bn_inplace(trunk, k_trunk_tip_bn_.scale, k_trunk_tip_bn_.bias);
    act(trunk);

    // ── Policy head ───────────────────────────────────────
    MatF p1; conv_only(k_p1_conv_, trunk, p1);                          // [p1_c, B*HW]
    MatF g1; conv_only(k_g1_conv_, trunk, g1);
    bn_inplace(g1, k_g1_bn_.scale, k_g1_bn_.bias);
    act(g1);
    MatF g1s = gpool_stats(g1);                                          // [3*g1_c, B]
    MatF bias_pol; bias_pol.noalias() = k_gpool_to_bias_.weight * g1s;   // [p1_c, B]
    add_per_batch(p1, bias_pol);
    bn_inplace(p1, k_p1_bn_.scale, k_p1_bn_.bias);
    act(p1);
    MatF p2; conv_only(k_p2_conv_, p1, p2);                              // [out_ch, B*HW]
    MatF pass; pass.noalias() = k_gpool_to_pass_.weight * g1s;           // [1, B]

    // ── Value head ────────────────────────────────────────
    MatF v1; conv_only(k_v1_conv_, trunk, v1);
    bn_inplace(v1, k_v1_bn_.scale, k_v1_bn_.bias);
    act(v1);
    MatF pooled = vh_stats(v1);                                          // [3*v1_c, B]
    MatF v2; v2.noalias() = k_v2_mul_.weight * pooled;                   // [v2_c, B]
    v2.colwise() += k_v2_bias_;
    if (format_ == ModelFormat::KataGo && katago_use_mish_) {
        v2 = v2.array() * (((v2.array().max(-30.0f)).exp() + 1.0f).log()).tanh();
    } else {
        v2 = v2.cwiseMax(0.0f);
    }
    MatF v3_logits; v3_logits.noalias() = k_v3_mul_.weight * v2;         // [3, B]
    v3_logits.colwise() += k_v3_bias_;
    MatF sv3; sv3.noalias() = k_sv3_mul_.weight * v2;                    // [sv3_c, B]
    sv3.colwise() += k_sv3_bias_;

    MatF own; conv_only(k_v_ownership_conv_, v1, own);                   // [1, B*HW]

    // Pack per-batch results.
    std::vector<Result> results(B);
    for (int b = 0; b < B; ++b) {
        Result& r = results[b];
        r.policy.resize(action_size);
        for (int i = 0; i < hw; ++i) r.policy[i] = p2(0, b * hw + i);
        r.policy[hw] = pass(0, b);

        float vmax = std::max({v3_logits(0, b), v3_logits(1, b), v3_logits(2, b)});
        float ew = std::exp(v3_logits(0, b) - vmax);
        float el = std::exp(v3_logits(1, b) - vmax);
        float ed = std::exp(v3_logits(2, b) - vmax);
        float sum = ew + el + ed;
        r.value    = (ew - el) / sum;
        r.score    = sv3(k_score_mean_idx_, b) * 20.0f;
        r.score_sd = softplus(sv3(k_score_stdev_idx_, b)) * 20.0f;

        r.ownership.resize(hw);
        for (int i = 0; i < hw; ++i)
            r.ownership[i] = (std::tanh(own(0, b * hw + i)) + 1.0f) * 0.5f;
    }
    return results;
}

// ================================================================
// Batch predict — loops over single predict
// ================================================================
std::vector<EigenComputeHandle::Result>
EigenComputeHandle::predict_batch(const std::vector<std::vector<float>>& states) {
    if (format_ == ModelFormat::KataGo)
        return predict_batch_katago(states);
    return predict_batch_minigo(states);
}

}  // namespace minigo
#endif  // MINIGO_HAS_EIGEN
