#include "eigen_engine.h"
#include "onnx_loader.h"
#include <cmath>
#include <cstring>
#include <iostream>
#include <algorithm>
#include <numeric>
#include <unordered_map>
#include <stdexcept>

namespace minigo {

// ============================================================
// ONNX weight loading
// ============================================================

void EigenEngine::load_from_onnx(const std::string& path) {
    using namespace onnx_parser;

    auto tensors = parse_onnx_file(path);

    // Build name -> tensor map
    std::unordered_map<std::string, OnnxTensor*> tensor_map;
    for (auto& t : tensors)
        tensor_map[t.name] = &t;

    auto get_tensor = [&](const std::string& name) -> OnnxTensor& {
        auto it = tensor_map.find(name);
        if (it == tensor_map.end())
            throw std::runtime_error("Missing tensor: " + name);
        return *it->second;
    };

    // Load conv weight: (C_out, C_in, kH, kW) -> Eigen (C_out, C_in*kH*kW)
    auto load_conv = [&](Conv2dParams& conv, const std::string& name) {
        auto& t  = get_tensor(name);
        auto  fl = t.get_floats();
        int c_out = (int)t.dims[0];
        int c_in  = (int)t.dims[1];
        int kh    = (int)t.dims[2];
        int kw    = (int)t.dims[3];
        int cols  = c_in * kh * kw;

        conv.c_out = c_out; conv.c_in = c_in; conv.kh = kh; conv.kw = kw;

        Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
            src(fl.data(), c_out, cols);
        conv.weight = src;
    };

    // Load BN params and pre-fuse: y = scale * x + bias
    const float eps = 1e-5f;
    auto load_bn = [&](BNParams& bn, const std::string& prefix) {
        auto gamma_f = get_tensor(prefix + ".weight").get_floats();
        auto beta_f  = get_tensor(prefix + ".bias").get_floats();
        auto mean_f  = get_tensor(prefix + ".running_mean").get_floats();
        auto var_f   = get_tensor(prefix + ".running_var").get_floats();

        int channels = (int)gamma_f.size();
        bn.scale.resize(channels);
        bn.bias.resize(channels);
        for (int i = 0; i < channels; i++) {
            float inv_std = 1.0f / std::sqrt(var_f[i] + eps);
            bn.scale(i) = gamma_f[i] * inv_std;
            bn.bias(i)  = beta_f[i] - gamma_f[i] * mean_f[i] * inv_std;
        }
    };

    // Load linear layer
    auto load_linear = [&](LinearParams& fc, const std::string& prefix) {
        auto& wt     = get_tensor(prefix + ".weight");
        auto  w_fl   = wt.get_floats();
        int rows = (int)wt.dims[0];
        int cols = (int)wt.dims[1];

        Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
            src(w_fl.data(), rows, cols);
        fc.weight = src;

        auto b_fl = get_tensor(prefix + ".bias").get_floats();
        fc.bias = Eigen::Map<Vector>(b_fl.data(), (int)b_fl.size());
    };

    // Infer architecture
    auto& input_w = get_tensor("input_conv.weight");
    input_channels = (int)input_w.dims[1];
    num_filters    = (int)input_w.dims[0];

    num_res_blocks = 0;
    while (tensor_map.count("res_blocks." + std::to_string(num_res_blocks) + ".conv1.weight"))
        num_res_blocks++;

    auto& pfc_w = get_tensor("policy_fc.weight");
    int action_size = (int)pfc_w.dims[0];
    board_size = (int)std::sqrt((float)(action_size - 1));

    std::cout << "ONNX model: board=" << board_size
              << " filters=" << num_filters
              << " blocks="  << num_res_blocks
              << " channels=" << input_channels << "\n";

    // Load all weights
    load_conv(input_conv_, "input_conv.weight");
    load_bn  (input_bn_,   "input_bn");

    res_blocks_.resize(num_res_blocks);
    for (int i = 0; i < num_res_blocks; i++) {
        std::string prefix = "res_blocks." + std::to_string(i);
        load_conv(res_blocks_[i].conv1, prefix + ".conv1.weight");
        load_bn  (res_blocks_[i].bn1,  prefix + ".bn1");
        load_conv(res_blocks_[i].conv2, prefix + ".conv2.weight");
        load_bn  (res_blocks_[i].bn2,  prefix + ".bn2");
    }

    load_conv  (policy_conv_, "policy_conv.weight");
    load_bn    (policy_bn_,   "policy_bn");
    load_linear(policy_fc_,   "policy_fc");

    load_conv  (value_conv_,  "value_conv.weight");
    load_bn    (value_bn_,    "value_bn");
    load_linear(value_fc1_,   "value_fc1");
    load_linear(value_fc2_,   "value_fc2");

    std::cout << "Loaded ONNX weights from " << path << " (eigen backend)\n";
}

// ============================================================
// Model loading
// ============================================================

void EigenEngine::load_model(const std::string& model_path) {
    load_from_onnx(model_path);
}

// ============================================================
// Im2col for 3x3 convolution with padding=1, stride=1
// ============================================================

void EigenEngine::im2col_3x3(const Tensor& input, int h, int w) {
    int c_in       = (int)input.rows();
    int patch_size = c_in * 9;
    int hw         = h * w;

    im2col_buf_.resize(patch_size, hw);

    for (int ic = 0; ic < c_in; ic++) {
        for (int kh = 0; kh < 3; kh++) {
            for (int kw = 0; kw < 3; kw++) {
                int row      = ic * 9 + kh * 3 + kw;
                int offset_r = kh - 1;
                int offset_c = kw - 1;

                for (int oh = 0; oh < h; oh++) {
                    int ih = oh + offset_r;
                    for (int ow = 0; ow < w; ow++) {
                        int iw  = ow + offset_c;
                        int col = oh * w + ow;

                        if (ih >= 0 && ih < h && iw >= 0 && iw < w)
                            im2col_buf_(row, col) = input(ic, ih * w + iw);
                        else
                            im2col_buf_(row, col) = 0.0f;
                    }
                }
            }
        }
    }
}

// ============================================================
// Layer operations
// ============================================================

void EigenEngine::conv2d_3x3(const Tensor& input, Tensor& output,
                               const Conv2dParams& params, int h, int w) {
    im2col_3x3(input, h, w);
    output.noalias() = params.weight * im2col_buf_;
}

void EigenEngine::conv2d_1x1(const Tensor& input, Tensor& output,
                               const Conv2dParams& params) {
    output.noalias() = params.weight * input;
}

void EigenEngine::batch_norm_apply(Tensor& x, const BNParams& params) {
    for (int c = 0; c < (int)x.rows(); c++) {
        x.row(c) = x.row(c) * params.scale(c) +
                    Eigen::RowVectorXf::Constant(x.cols(), params.bias(c));
    }
}

void EigenEngine::relu(Tensor& x) {
    x = x.cwiseMax(0.0f);
}

void EigenEngine::linear(const Vector& input, Vector& output,
                          const LinearParams& params) {
    output.noalias() = params.weight * input + params.bias;
}

// ============================================================
// Forward pass
// ============================================================

std::pair<std::vector<float>, float>
EigenEngine::predict(const std::vector<float>& state) {
    int n          = board_size;
    int hw         = n * n;
    int action_size = hw + 1;

    Tensor x = Eigen::Map<const Tensor>(state.data(), input_channels, hw);

    // Input convolution
    Tensor trunk;
    conv2d_3x3(x, trunk, input_conv_, n, n);
    batch_norm_apply(trunk, input_bn_);
    relu(trunk);

    // Residual blocks
    for (auto& rb : res_blocks_) {
        Tensor residual = trunk;

        Tensor tmp;
        conv2d_3x3(residual, tmp, rb.conv1, n, n);
        batch_norm_apply(tmp, rb.bn1);
        relu(tmp);

        Tensor tmp2;
        conv2d_3x3(tmp, tmp2, rb.conv2, n, n);
        batch_norm_apply(tmp2, rb.bn2);

        trunk = tmp2 + residual;
        relu(trunk);
    }

    // Policy head
    Tensor p_conv;
    conv2d_1x1(trunk, p_conv, policy_conv_);
    batch_norm_apply(p_conv, policy_bn_);
    relu(p_conv);

    Eigen::Map<Vector> p_flat(p_conv.data(), 2 * hw);
    Vector p_logits;
    linear(p_flat, p_logits, policy_fc_);

    // Softmax
    float max_logit = p_logits.maxCoeff();
    Vector p_exp    = (p_logits.array() - max_logit).exp();
    float  sum_exp  = p_exp.sum();
    Vector p_probs  = p_exp / sum_exp;

    std::vector<float> policy(action_size);
    for (int i = 0; i < action_size; i++)
        policy[i] = p_probs(i);

    // Value head
    Tensor v_conv;
    conv2d_1x1(trunk, v_conv, value_conv_);
    batch_norm_apply(v_conv, value_bn_);
    relu(v_conv);

    Eigen::Map<Vector> v_flat(v_conv.data(), hw);
    Vector v_hidden;
    linear(v_flat, v_hidden, value_fc1_);
    v_hidden = v_hidden.cwiseMax(0.0f);

    Vector v_out;
    linear(v_hidden, v_out, value_fc2_);
    float value = std::tanh(v_out(0));

    return {policy, value};
}

}  // namespace minigo
