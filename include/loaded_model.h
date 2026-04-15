#pragma once

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace minigo {

struct ConvBNWeights {
    std::vector<float> weight;
    std::vector<float> bn_scale;
    std::vector<float> bn_bias;
    int c_out = 0;
    int c_in = 0;
    int k = 0;
};

struct FCWeights {
    std::vector<float> weight;
    std::vector<float> bias;
    int out_features = 0;
    int in_features = 0;
};

class LoadedModel {
public:
    static std::shared_ptr<LoadedModel> load(const std::string& model_path);

    std::string model_path;
    std::string model_type;
    int board_rows = 10;
    int board_cols = 9;
    int input_channels = 0;
    int num_filters = 0;
    int num_res_blocks = 0;
    int vit_depth = 0;
    int vit_heads = 0;
    int vit_kv_groups = 0;
    int action_size = 0;

    ConvBNWeights input_conv;
    std::vector<ConvBNWeights> res_conv1;
    std::vector<ConvBNWeights> res_conv2;
    ConvBNWeights policy_conv;
    ConvBNWeights value_conv;
    FCWeights policy_fc;
    FCWeights value_fc1;
    FCWeights value_fc2;
};

}  // namespace minigo
