#pragma once

#include <cmath>
#include <memory>
#include <string>
#include <vector>
#include <stdexcept>
#include <unordered_map>

namespace minigo {

// ================================================================
// LoadedModel — parsed ONNX weights in CPU memory (KataGo pattern)
//
// Loaded once on the main thread.  Shared (const) across all
// NNEvaluator server threads.  Each ComputeHandle uploads these
// CPU weights to its own GPU buffers.
//
// BN parameters are pre-fused: scale[i] = gamma[i] / sqrt(var[i] + eps),
// bias[i] = beta[i] - gamma[i] * mean[i] / sqrt(var[i] + eps).
// ================================================================

struct ConvBNWeights {
    std::vector<float> weight;      // [C_out, C_in*kH*kW] row-major
    std::vector<float> bn_scale;    // [C_out] pre-fused
    std::vector<float> bn_bias;     // [C_out] pre-fused
    int c_out = 0, c_in = 0, k = 0;  // k = kernel size (1 or 3)
};

struct FCWeights {
    std::vector<float> weight;  // [out_features, in_features]
    std::vector<float> bias;    // [out_features]
    int out_features = 0, in_features = 0;
};

class LoadedModel {
public:
    // Load and parse an ONNX model file.  Pre-fuses BN parameters.
    static std::shared_ptr<LoadedModel> load(const std::string& model_path);

    // Original ONNX file path (needed by TensorRT backend)
    std::string model_path;

    // Model architecture metadata
    std::string model_type = "resnet";  // "resnet" or "vit"
    int board_size = 9;
    int input_channels = 17;
    int num_filters = 64;
    int num_res_blocks = 5;

    // Weights (all pre-fused BN, CPU-side)
    ConvBNWeights              input_conv;
    std::vector<ConvBNWeights> res_conv1, res_conv2;
    ConvBNWeights              policy_conv, value_conv;
    FCWeights                  policy_fc, value_fc1, value_fc2;

    // Score head (always present in current models)
    bool                       has_score_head = true;
    ConvBNWeights              score_conv;
    FCWeights                  score_fc1, score_fc2;
};

}  // namespace minigo
