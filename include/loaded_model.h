#pragma once

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace minigo {

struct ConvBNWeights {
    std::vector<float> weight;
    std::vector<float> bn_scale;   // fused BN scale:  gamma / sqrt(var+eps)
    std::vector<float> bn_bias;    // fused BN bias:   beta - gamma*mean / sqrt(var+eps)
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

// ── KataGo-style residual block weights ───────────────────────────
// One of three block kinds, driven by which auxiliary weights are
// present in the exported ONNX.  Even indices default to SE blocks,
// odd indices to GPool - matching the Python XiangqiNet layout.
enum class BlockKind : int { Plain = 0, SE = 1, GPool = 2 };

struct BlockWeights {
    BlockKind kind = BlockKind::Plain;
    ConvBNWeights conv1;
    ConvBNWeights conv2;
    // SE path (only valid if kind == SE).
    FCWeights se_fc1;
    FCWeights se_fc2;
    // GPool path (only valid if kind == GPool).
    ConvBNWeights pool_conv;
    FCWeights pool_fc;
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
    // Value head emits 3-class WLD logits; kept here for backends that need it.
    int value_head_size = 3;

    ConvBNWeights input_conv;
    std::vector<BlockWeights> blocks;
    ConvBNWeights policy_conv;
    ConvBNWeights value_conv;
    FCWeights policy_fc;
    FCWeights value_fc1;
    FCWeights value_fc2;   // out_features = 3 (W / D / L)
};

}  // namespace minigo
