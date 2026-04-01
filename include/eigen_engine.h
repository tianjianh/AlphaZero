#pragma once

#include "inference_engine.h"
#include <Eigen/Dense>
#include <string>
#include <vector>

namespace minigo {

using Tensor = Eigen::MatrixXf;
using Vector = Eigen::VectorXf;

struct Conv2dParams {
    Tensor weight;   // (C_out, C_in * kH * kW)
    int c_in, c_out, kh, kw;
};

struct BNParams {
    Vector scale;    // gamma / sqrt(var + eps)
    Vector bias;     // beta - gamma * mean / sqrt(var + eps)
};

struct LinearParams {
    Tensor weight;   // (out_features, in_features)
    Vector bias;     // (out_features)
};

struct ResBlockParams {
    Conv2dParams conv1, conv2;
    BNParams bn1, bn2;
};

class EigenEngine : public InferenceEngine {
public:
    EigenEngine() = default;

    void load_model(const std::string& model_path) override;
    std::pair<std::vector<float>, float>
    predict(const std::vector<float>& state) override;
    std::string backend_name() const override { return "eigen"; }

private:
    // Network parameters
    Conv2dParams input_conv_;
    BNParams input_bn_;
    std::vector<ResBlockParams> res_blocks_;

    Conv2dParams policy_conv_;
    BNParams policy_bn_;
    LinearParams policy_fc_;

    Conv2dParams value_conv_;
    BNParams value_bn_;
    LinearParams value_fc1_;
    LinearParams value_fc2_;

    // Workspace (preallocated, not thread-safe)
    Tensor im2col_buf_;

    // Layer operations
    void conv2d_3x3(const Tensor& input, Tensor& output,
                    const Conv2dParams& params, int h, int w);
    void conv2d_1x1(const Tensor& input, Tensor& output,
                    const Conv2dParams& params);
    void batch_norm_apply(Tensor& x, const BNParams& params);
    void relu(Tensor& x);
    void linear(const Vector& input, Vector& output, const LinearParams& params);
    void im2col_3x3(const Tensor& input, int h, int w);

    // ONNX loading
    void load_from_onnx(const std::string& path);
};

}  // namespace minigo
