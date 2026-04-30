#pragma once
#ifdef MINIGO_HAS_EIGEN

#include "compute_context.h"
#include "loaded_model.h"
#include <Eigen/Dense>
#include <vector>
#include <memory>

namespace minigo {

// ================================================================
// Eigen CPU backend — ComputeContext + ComputeHandle
//
// Supports two model formats:
//   * MiniGo:  KataGo-style ResNet built in scripts/model.py
//              (single state input; alternating ResBlockSE +
//              GPoolResBlock trunk; GPool value/score heads).
//   * KataGo:  kata1-style network exported by tools/katago_to_onnx.py
//              (state_spatial + state_global inputs; pre-activation
//              regular + gpool blocks; KataGo policy/value heads with
//              post-processing baked into the graph).
//
// ViT models are not supported on this backend — use TensorRT.
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
    EigenComputeHandle(const LoadedModel* model);

    std::vector<Result>
    predict_batch(const std::vector<std::vector<float>>& states) override;

    // ── Weight containers (public so the file-scope loaders in
    //   eigen_compute.cpp can populate them — they're plain data
    //   holders, not part of the inference API). ────────────────
    //
    // The whole forward pass is row-major: weight rows are output channels,
    // im2col rows are patch components, activation rows are channels and
    // columns are flattened spatial positions. This lets us map the raw
    // state buffer (laid out as [C, H, W] row-major in memory) directly.
    using MatF = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
    using VecF = Eigen::VectorXf;

    // Conv with pre-fused BN absorbed into a (scale, bias) per output channel.
    struct ConvBN {
        MatF weight;             // [c_out, c_in * k * k]
        VecF bn_scale, bn_bias;  // [c_out]
        int c_out = 0, c_in = 0, k = 0;
    };

    // Conv without BN (optional bias).
    struct Conv {
        MatF weight;     // [c_out, c_in * k * k]
        VecF bias;       // [c_out] (empty when no bias)
        int c_out = 0, c_in = 0, k = 0;
        bool has_bias = false;
    };

    struct FC {
        MatF weight;     // [out, in]
        VecF bias;       // [out] (empty when no bias)
        int out_features = 0, in_features = 0;
        bool has_bias = false;
    };

    struct BN {
        VecF scale, bias;        // pre-fused gamma / sqrt(var+eps), beta - gamma*mean/sqrt
        int channels = 0;
    };

    // GPool head used by MiniGo (mean + max + std → fc1 → fc2).
    struct GPoolHead {
        ConvBN conv;             // 1x1
        FC fc1;                  // [mlp_hidden, 3 * head_ch]
        FC fc2;                  // [out_features, mlp_hidden]
    };

private:
    // One trunk block. The same struct holds both MiniGo and KataGo blocks;
    // only the relevant fields are populated based on `kind`.
    enum class BlockKind {
        MiniGoSE,        // ResBlockSE: conv1 → bn1 → relu → conv2 → bn2 → SE → +residual → relu
        MiniGoGPool,     // GPoolResBlock: conv_main + bn_main; conv_pool + bn_pool + relu + (mean,max) → pool_fc → bias; + relu; conv2 + bn2; +residual; relu
        KataRegular,     // pre_bn → act → regular_conv → mid_bn → act → final_conv → +skip
        KataGPool,       // pre_bn → act → (regular_conv); (gpool_conv → gpool_bn → act → gstats → gpool_to_bias) → bias added; mid_bn → act → final_conv → +skip
    };

    struct Block {
        BlockKind kind;

        // MiniGo SE block fields
        ConvBN se_conv1, se_conv2;
        FC se_fc1, se_fc2;

        // MiniGo GPool block fields
        ConvBN mg_conv_main, mg_conv_pool, mg_conv2;
        FC mg_pool_fc;
        int mg_pool_channels = 0;

        // KataGo regular / gpool block fields
        BN k_pre_bn, k_mid_bn;
        Conv k_regular_conv, k_final_conv;
        // KataGo gpool extras (only for KataGPool)
        Conv k_gpool_conv;
        BN k_gpool_bn;
        FC k_gpool_to_bias;
        int k_gpool_channels = 0;
    };

    // ── Format / metadata ───────────────────────────────────────
    ModelFormat format_ = ModelFormat::MiniGo;
    int board_size_ = 0;
    int input_channels_ = 0;
    int input_global_channels_ = 0;
    int num_filters_ = 0;        // MiniGo: filters; KataGo: trunk_c
    int num_blocks_ = 0;
    bool katago_use_mish_ = true;  // detected; defaults to mish for kata1

    // ── MiniGo weights ──────────────────────────────────────────
    ConvBN mg_input_conv_;       // input_conv + input_bn
    std::vector<Block> blocks_;  // populated for both formats
    // Heads
    ConvBN mg_policy_conv_;      // policy_conv (1x1, 2 ch) + policy_bn
    FC mg_policy_fc_;
    GPoolHead mg_value_head_;
    GPoolHead mg_score_mean_head_;
    GPoolHead mg_score_stdev_head_;
    Conv mg_ownership_conv_;     // 1x1, with bias

    // ── KataGo weights ──────────────────────────────────────────
    Conv k_initial_conv_;        // no bias, 5x5 typically (kernel from weight shape)
    FC k_initial_matmul_;        // no bias, [trunk_c, num_global]
    BN k_trunk_tip_bn_;
    // Policy head
    Conv k_p1_conv_, k_g1_conv_, k_p2_conv_;
    BN k_g1_bn_, k_p1_bn_;
    FC k_gpool_to_bias_, k_gpool_to_pass_;  // no bias
    int k_p2_out_channels_ = 0;
    // Value head
    Conv k_v1_conv_;
    BN k_v1_bn_;
    FC k_v2_mul_, k_v3_mul_, k_sv3_mul_;    // no bias (PyTorch Linears with bias=False)
    VecF k_v2_bias_, k_v3_bias_, k_sv3_bias_;
    Conv k_v_ownership_conv_;
    int k_score_mean_idx_ = 0;
    int k_score_stdev_idx_ = 1;

    // ── Workspace ───────────────────────────────────────────────
    std::vector<float> im2col_buf_;

    // ── Forward passes ──────────────────────────────────────────
    // Both predict_batch_* operate on the whole drained batch at once:
    // activations are kept as row-major [C, B*HW] matrices so each
    // layer is a single GEMM (weight × col) instead of B small GEMMs.
    // Eigen's matrix multiply is multi-threaded via OpenMP when the
    // backend is built with `find_package(OpenMP)` succeeding.
    std::vector<Result>
    predict_batch_minigo(const std::vector<std::vector<float>>& states);
    std::vector<Result>
    predict_batch_katago(const std::vector<std::vector<float>>& states);

    // ── Helpers ─────────────────────────────────────────────────
    // Batched im2col: in [C, B*HW] → out [C*K*K, B*HW] (row-major).
    void im2col_batched(const float* in, int C, int H, int W,
                        int K, int pad, int B, float* out);
    void apply_act_inplace(MatF& x);  // KataGo activation (mish or relu)
};

}  // namespace minigo
#endif  // MINIGO_HAS_EIGEN
