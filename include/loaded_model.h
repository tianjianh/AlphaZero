#pragma once

#include <cmath>
#include <memory>
#include <mutex>
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

// Detected at load time from the ONNX graph inputs.
enum class ModelFormat {
    MiniGo = 0,   // MiniGo's own ResNet/ViT (single state input)
    KataGo = 1,   // KataGo-V7 dual input (state_spatial + state_global):
                  // either a converted kata1 net (tools/katago_to_onnx.py)
                  // or a trainable KataGoNet (scripts/model.py --arch katago)
};

// ================================================================
// Full weight bundle for backends that run their own kernels
// (OpenCL / CUDA / Metal / Eigen).  Populated lazily by
// LoadedModel::weights() — TensorRT never pays for this (it parses
// the ONNX graph itself).
//
// All tensors are raw fp32 row-major, exactly as stored in the
// embedded `_sd_` state_dict; BN is pre-fused with the epsilon that
// matches the producing framework (1e-5 PyTorch, 1e-20 kata1).
// ================================================================

struct BNParamsW {                       // pre-fused BatchNorm
    std::vector<float> scale, bias;      // [C]
    int channels = 0;
};

struct ConvW {
    std::vector<float> weight;           // [c_out, c_in*k*k]
    std::vector<float> bias;             // [c_out], empty when !has_bias
    int c_out = 0, c_in = 0, k = 0;
    bool has_bias = false;
};

struct FCW {
    std::vector<float> weight;           // [out, in]
    std::vector<float> bias;             // [out], empty when !has_bias
    int out_features = 0, in_features = 0;
    bool has_bias = false;
};

struct GPoolHeadW {                      // scripts/model.py GPoolHead
    ConvW conv;                          // 1x1, no bias
    BNParamsW bn;
    FCW fc1, fc2;                        // fc1: [mlp, 3*head_ch], both biased
};

struct TrunkBlockW {
    enum Kind {
        SE = 0,          // ResBlockSE:    conv1+bn1+relu → conv2+bn2 → SE → +res → relu
        MgGPool,         // GPoolResBlock: bn(conv_main)+poolbias → relu → conv2+bn2 → +res → relu
        KataRegular,     // pre-activation: pre_bn→act→conv→mid_bn→act→conv→+skip
        KataGPool,       // pre-activation with gpool bias branch
    } kind = SE;

    // ── MiniGo SE fields ─────────────────────────────────────
    ConvW conv1; BNParamsW bn1;
    ConvW conv2; BNParamsW bn2;          // conv2/bn2 shared with MgGPool
    FCW se_fc1, se_fc2;

    // ── MiniGo GPool fields ──────────────────────────────────
    ConvW conv_main; BNParamsW bn_main;
    ConvW conv_pool; BNParamsW bn_pool;
    FCW pool_fc;                         // [C, 2*pool_ch], biased

    // ── KataGo fields (both namings normalized here) ─────────
    BNParamsW pre_bn, mid_bn;
    ConvW regular_conv, final_conv;
    ConvW gpool_conv; BNParamsW gpool_bn;
    FCW gpool_to_bias;                   // [C, 3*pool_ch]

    int pool_channels = 0;
};

struct ViTBlockW {
    std::vector<float> ln1_g, ln1_b;     // norm1 [d]
    std::vector<float> ln2_g, ln2_b;     // norm2 [d]
    FCW qkv;                             // [(H+2G)*head_dim, d], packed [Q|K|V]
    FCW out_proj;                        // [d, H*head_dim]
    FCW mlp1, mlp2;                      // [4d, d], [d, 4d]
};

struct ViTW {
    FCW token_proj;                      // [d, C_in]
    std::vector<float> row_embed, col_embed;  // [board, d] row-major
    std::vector<float> rel_bias;         // [H, (2*board-1)^2]
    std::vector<ViTBlockW> blocks;
    std::vector<float> final_ln_g, final_ln_b;
    FCW policy_proj;                     // [1, d]
    float pass_logit = 0.0f;
    FCW value_fc1, value_fc2;            // [d, d] gelu [3, d]
    FCW score_mean_fc1, score_mean_fc2;
    FCW score_stdev_fc1, score_stdev_fc2;
    FCW ownership_proj;                  // [1, d]
    int num_rel_buckets = 0;
};

struct ModelWeights {
    enum TrunkStyle { MiniGoTrunk, KataTrunk, ViTTrunk } trunk_style = MiniGoTrunk;
    enum HeadStyle  { MiniGoHeads, Kata1Heads }          head_style  = MiniGoHeads;
    bool use_mish = false;               // kata1 activation (else relu)

    // ── MiniGo ResNet trunk ─────────────────────────────────
    ConvW input_conv; BNParamsW input_bn;

    // ── KataGo trunk (trainable KataGoNet or converted kata1) ─
    ConvW stem_conv;                     // 3x3 (KataGoNet) or 3x3/5x5 (kata1)
    FCW stem_global;                     // [C, n_global]; biased for KataGoNet
    BNParamsW tip_bn;

    std::vector<TrunkBlockW> blocks;     // trunk blocks, all styles

    // ── MiniGo-style heads (resnet / vit-none / KataGoNet) ───
    ConvW policy_conv; BNParamsW policy_bn; FCW policy_fc;
    GPoolHeadW value_head, score_mean_head, score_stdev_head;
    ConvW ownership_conv;                // 1x1, biased

    // ── kata1-style heads (converted stock networks) ─────────
    ConvW p1_conv, g1_conv, p2_conv;
    BNParamsW g1_bn, p1_bn;
    FCW k_gpool_to_bias, k_gpool_to_pass;
    ConvW v1_conv; BNParamsW v1_bn;
    FCW v2_mul, v3_mul, sv3_mul;         // no bias in the Linear itself…
    std::vector<float> v2_bias, v3_bias, sv3_bias;  // …bias is a raw Parameter
    ConvW vown_conv;
    int score_mean_idx = 0, score_stdev_idx = 1;

    // ── ViT ──────────────────────────────────────────────────
    ViTW vit;
};

class LoadedModel {
public:
    // Load and parse an ONNX model file (metadata only — cheap).
    static std::shared_ptr<LoadedModel> load(const std::string& model_path);

    // Original ONNX file path (needed by TensorRT backend)
    std::string model_path;

    // Architecture metadata (all inferred from ONNX weights by load())
    ModelFormat format = ModelFormat::MiniGo;
    std::string model_type;   // "resnet", "vit", or "katago"
    int board_size = 0;
    int input_channels = 0;
    int input_global_channels = 0;  // KataGo only (0 for MiniGo)
    int num_filters = 0;      // ResNet: conv filters; ViT: d_model
    int num_res_blocks = 0;   // ResNet only
    int vit_depth = 0;        // ViT: transformer blocks
    int vit_heads = 0;        // ViT: Q heads
    int vit_kv_groups = 0;    // ViT: KV groups (GQA)
    int vit_head_dim = 0;     // ViT: per-head dim

    // Full weights for kernel backends.  Lazy: first call re-parses the
    // ONNX file and builds the bundle; thread-safe (server threads create
    // handles concurrently).  Throws std::runtime_error on malformed files.
    const ModelWeights& weights() const;

    // Legacy per-op fields (kept for source compat; unpopulated)
    ConvBNWeights              input_conv;
    std::vector<ConvBNWeights> res_conv1, res_conv2;
    ConvBNWeights              policy_conv, value_conv;
    FCWeights                  policy_fc, value_fc1, value_fc2;
    ConvBNWeights              score_conv;
    FCWeights                  score_fc1, score_fc2;

private:
    mutable std::once_flag weights_once_;
    mutable std::shared_ptr<ModelWeights> weights_;
};

}  // namespace minigo
