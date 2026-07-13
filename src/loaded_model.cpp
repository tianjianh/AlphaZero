#include "loaded_model.h"
#include "onnx_loader.h"
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>

namespace minigo {

namespace {

using TensorMap = std::unordered_map<std::string, const onnx_parser::OnnxTensor*>;

// All state_dict tensors are embedded with this prefix so they don't
// collide with whatever the ONNX optimizer renamed in the graph
// (folded conv weights, Identity passthroughs, etc.). See
// `_embed_state_dict` in scripts/export_onnx.py and tools/katago_to_onnx.py.
constexpr const char* SD_PREFIX = "_sd_";

const onnx_parser::OnnxTensor* find_t(const TensorMap& tm, const std::string& name) {
    auto it = tm.find(SD_PREFIX + name);
    if (it != tm.end()) return it->second;
    it = tm.find(name);  // fallback for older ONNX files
    return it == tm.end() ? nullptr : it->second;
}

const onnx_parser::OnnxTensor& must_t(const TensorMap& tm, const std::string& name) {
    auto* t = find_t(tm, name);
    if (!t) throw std::runtime_error("LoadedModel: missing tensor '" + name + "'");
    return *t;
}

// Pre-fuse BN.  Epsilon differs by producing framework:
//   PyTorch nn.BatchNorm2d default: 1e-5 (MiniGo resnet/vit, KataGoNet)
//   kata1 binary format:            1e-20 (tools/katago_arch.py copies it)
void load_bn(BNParamsW& dst, const TensorMap& tm, const std::string& prefix, float eps) {
    auto g = must_t(tm, prefix + ".weight").get_floats();
    auto b = must_t(tm, prefix + ".bias").get_floats();
    auto m = must_t(tm, prefix + ".running_mean").get_floats();
    auto v = must_t(tm, prefix + ".running_var").get_floats();
    int n = (int)g.size();
    dst.scale.resize(n);
    dst.bias.resize(n);
    for (int i = 0; i < n; i++) {
        float inv_std = 1.0f / std::sqrt(v[i] + eps);
        dst.scale[i] = g[i] * inv_std;
        dst.bias[i]  = b[i] - g[i] * m[i] * inv_std;
    }
    dst.channels = n;
}

void load_conv(ConvW& dst, const TensorMap& tm, const std::string& wname,
               bool expect_bias) {
    auto& w = must_t(tm, wname);
    if (w.dims.size() != 4)
        throw std::runtime_error("expected 4D conv weight: " + wname);
    dst.c_out = (int)w.dims[0];
    dst.c_in  = (int)w.dims[1];
    if (w.dims[2] != w.dims[3])
        throw std::runtime_error("non-square kernel: " + wname);
    dst.k = (int)w.dims[2];
    dst.weight = w.get_floats();

    std::string bname = wname;
    auto pos = bname.rfind(".weight");
    if (pos != std::string::npos) bname.replace(pos, 7, ".bias");
    auto* b = find_t(tm, bname);
    if (b) {
        dst.bias = b->get_floats();
        dst.has_bias = true;
    } else if (expect_bias) {
        throw std::runtime_error("expected bias on conv: " + wname);
    } else {
        dst.has_bias = false;
    }
}

void load_fc(FCW& dst, const TensorMap& tm, const std::string& wname) {
    auto& w = must_t(tm, wname);
    if (w.dims.size() != 2)
        throw std::runtime_error("expected 2D fc weight: " + wname);
    dst.out_features = (int)w.dims[0];
    dst.in_features  = (int)w.dims[1];
    dst.weight = w.get_floats();
    std::string bname = wname;
    auto pos = bname.rfind(".weight");
    if (pos != std::string::npos) bname.replace(pos, 7, ".bias");
    auto* b = find_t(tm, bname);
    if (b) {
        dst.bias = b->get_floats();
        dst.has_bias = true;
    } else {
        dst.has_bias = false;
    }
}

void load_gpool_head(GPoolHeadW& h, const TensorMap& tm,
                     const std::string& prefix, float eps) {
    load_conv(h.conv, tm, prefix + ".conv.weight", false);
    load_bn(h.bn, tm, prefix + ".bn", eps);
    load_fc(h.fc1, tm, prefix + ".fc1.weight");
    load_fc(h.fc2, tm, prefix + ".fc2.weight");
}

}  // namespace

std::shared_ptr<LoadedModel> LoadedModel::load(const std::string& model_path) {
    using namespace onnx_parser;

    auto tensors = parse_onnx_file(model_path);
    TensorMap tm;
    for (auto& t : tensors) tm[t.name] = &t;

    auto has = [&](const std::string& name) -> bool {
        return find_t(tm, name) != nullptr;
    };
    auto get = [&](const std::string& name) -> const OnnxTensor& {
        return must_t(tm, name);
    };

    auto model = std::make_shared<LoadedModel>();
    model->model_path = model_path;

    // ── KataGo detection: graph has two named inputs ─────────────
    // Both a converted kata1 net (tools/katago_to_onnx.py) and a
    // trainable KataGoNet export (scripts/export_onnx.py --arch katago)
    // use input names "state_spatial" and "state_global".
    {
        auto inputs = onnx_parser::parse_onnx_graph_inputs(model_path);
        const onnx_parser::OnnxGraphIO* sp = nullptr;
        const onnx_parser::OnnxGraphIO* gl = nullptr;
        for (auto& i : inputs) {
            if (i.name == "state_spatial") sp = &i;
            else if (i.name == "state_global") gl = &i;
        }
        if (sp && gl) {
            model->format = ModelFormat::KataGo;
            model->model_type = "katago";
            if (sp->dims.size() != 4)
                throw std::runtime_error(
                    "KataGo state_spatial: expected 4D shape, got "
                    + std::to_string(sp->dims.size()) + "D");
            model->input_channels = (int)sp->dims[1];
            int h = (int)sp->dims[2];
            int w = (int)sp->dims[3];
            if (h != w)
                throw std::runtime_error(
                    "KataGo non-square board not supported: H=" +
                    std::to_string(h) + " W=" + std::to_string(w));
            model->board_size = h;
            if (gl->dims.size() != 2)
                throw std::runtime_error(
                    "KataGo state_global: expected 2D shape, got "
                    + std::to_string(gl->dims.size()) + "D");
            model->input_global_channels = (int)gl->dims[1];

            // Trunk metadata from the embedded state_dict (either naming).
            const char* stem = has("stem_conv.weight") ? "stem_conv.weight"
                             : has("stem.initial_conv.weight") ? "stem.initial_conv.weight"
                             : nullptr;
            if (stem) {
                model->num_filters = (int)get(stem).dims[0];
                int b = 0;
                while (has("blocks." + std::to_string(b) + ".conv2.weight") ||
                       has("blocks." + std::to_string(b) + ".final_conv.weight"))
                    b++;
                model->num_res_blocks = b;
            }

            std::cout << "Model loaded: type=katago board=" << model->board_size
                      << " channels=" << model->input_channels
                      << "+" << model->input_global_channels
                      << " trunk_c=" << model->num_filters
                      << " blocks=" << model->num_res_blocks << "\n";
            return model;
        }
    }

    // ── Detect model type and infer architecture ─────────────────
    bool is_vit = has("token_proj.weight");

    if (is_vit) {
        model->model_type = "vit";
        auto& tp = get("token_proj.weight");
        model->input_channels = (int)tp.dims[1];
        model->num_filters    = (int)tp.dims[0];  // d_model

        // board_size from row_embed: [board_size, d_model]
        if (has("row_embed.weight")) {
            model->board_size = (int)get("row_embed.weight").dims[0];
        } else if (has("orbit_ids")) {   // very old exports
            auto& oi = get("orbit_ids");
            int hw = 1;
            for (auto d : oi.dims) hw *= (int)d;
            model->board_size = (int)std::round(std::sqrt((double)hw));
        } else {
            model->board_size = 9;
        }

        // Depth: packed qkv_proj (current) or split q_proj (pre-migration).
        model->vit_depth = 0;
        while (has("blocks." + std::to_string(model->vit_depth) + ".attn.qkv_proj.weight") ||
               has("blocks." + std::to_string(model->vit_depth) + ".attn.q_proj.weight"))
            model->vit_depth++;

        // Exact head geometry:
        //   rel_bias  [H, buckets]           → num_heads
        //   out_proj  [d_model, H*head_dim]  → head_dim
        //   qkv_proj  [H*hd + 2*G*hd, d]     → kv_groups
        if (model->vit_depth > 0) {
            int H = (int)get("blocks.0.attn.rel_bias").dims[0];
            int q_dim = (int)get("blocks.0.attn.out_proj.weight").dims[1];
            int qkv_out;
            if (has("blocks.0.attn.qkv_proj.weight"))
                qkv_out = (int)get("blocks.0.attn.qkv_proj.weight").dims[0];
            else
                qkv_out = q_dim + (int)get("blocks.0.attn.kv_proj.weight").dims[0];
            model->vit_heads     = H;
            model->vit_head_dim  = q_dim / H;
            model->vit_kv_groups = (qkv_out - q_dim) / (2 * model->vit_head_dim);
        }

        std::cout << "Model loaded: type=vit board=" << model->board_size
                  << " d_model=" << model->num_filters
                  << " depth=" << model->vit_depth
                  << " heads=" << model->vit_heads
                  << " kv_groups=" << model->vit_kv_groups
                  << " head_dim=" << model->vit_head_dim << "\n";

    } else {
        // ResNet — only the new KataGo-style layout (alternating SE +
        // GPool blocks + global-pool heads) is supported.
        bool is_new_resnet = has("trunk.0.conv2.weight");
        bool is_legacy_resnet = has("res_blocks.0.conv1.weight");
        if (!is_new_resnet) {
            if (is_legacy_resnet) {
                throw std::runtime_error(
                    "Legacy AlphaZero ResNet format detected (res_blocks.*). "
                    "This architecture has been replaced with a KataGo-style ResNet "
                    "(alternating SE + GPool residual blocks, global-pool value/score heads). "
                    "Please retrain from scratch using the new architecture.");
            }
            throw std::runtime_error(
                "Unknown model format: no `token_proj.weight` (ViT) and no "
                "`trunk.0.conv2.weight` (new ResNet).  Is this an ONNX file "
                "from minigo-cpp?");
        }

        model->model_type = "resnet";
        auto& iw = get("input_conv.weight");
        model->input_channels = (int)iw.dims[1];
        model->num_filters    = (int)iw.dims[0];

        model->num_res_blocks = 0;
        while (has("trunk." + std::to_string(model->num_res_blocks) + ".conv2.weight"))
            model->num_res_blocks++;

        auto& pfw = get("policy_fc.weight");
        int action_size = (int)pfw.dims[0];
        model->board_size = (int)std::round(std::sqrt((double)(action_size - 1)));

        std::cout << "Model loaded: type=resnet board=" << model->board_size
                  << " filters=" << model->num_filters
                  << " blocks=" << model->num_res_blocks
                  << " channels=" << model->input_channels << "\n";
    }
    return model;
}

// ================================================================
// Full weight bundle — lazy, shared by the kernel backends.
// ================================================================
const ModelWeights& LoadedModel::weights() const {
    std::call_once(weights_once_, [this] {
        auto w = std::make_shared<ModelWeights>();

        auto tensors = onnx_parser::parse_onnx_file(model_path);
        TensorMap tm;
        for (auto& t : tensors) tm[t.name] = &t;
        auto has = [&](const std::string& n) { return find_t(tm, n) != nullptr; };

        if (model_type == "vit") {
            // ── GoViT ────────────────────────────────────────
            w->trunk_style = ModelWeights::ViTTrunk;
            w->head_style  = ModelWeights::MiniGoHeads;
            ViTW& v = w->vit;

            load_fc(v.token_proj, tm, "token_proj.weight");
            v.row_embed = must_t(tm, "row_embed.weight").get_floats();
            v.col_embed = must_t(tm, "col_embed.weight").get_floats();

            v.blocks.resize(vit_depth);
            for (int i = 0; i < vit_depth; i++) {
                std::string pre = "blocks." + std::to_string(i) + ".";
                ViTBlockW& b = v.blocks[i];
                b.ln1_g = must_t(tm, pre + "norm1.weight").get_floats();
                b.ln1_b = must_t(tm, pre + "norm1.bias").get_floats();
                b.ln2_g = must_t(tm, pre + "norm2.weight").get_floats();
                b.ln2_b = must_t(tm, pre + "norm2.bias").get_floats();
                if (has(pre + "attn.qkv_proj.weight")) {
                    load_fc(b.qkv, tm, pre + "attn.qkv_proj.weight");
                } else {
                    // Pre-migration split naming: concat [Q | K | V] rows.
                    FCW q, kv;
                    load_fc(q, tm, pre + "attn.q_proj.weight");
                    load_fc(kv, tm, pre + "attn.kv_proj.weight");
                    b.qkv.out_features = q.out_features + kv.out_features;
                    b.qkv.in_features  = q.in_features;
                    b.qkv.weight = q.weight;
                    b.qkv.weight.insert(b.qkv.weight.end(),
                                        kv.weight.begin(), kv.weight.end());
                    b.qkv.bias = q.bias;
                    b.qkv.bias.insert(b.qkv.bias.end(),
                                      kv.bias.begin(), kv.bias.end());
                    b.qkv.has_bias = q.has_bias;
                }
                // rel_bias lives per block: [H, buckets]
                if (i == 0) {
                    auto& rb = must_t(tm, pre + "attn.rel_bias");
                    v.num_rel_buckets = (int)rb.dims[1];
                }
                load_fc(b.out_proj, tm, pre + "attn.out_proj.weight");
                load_fc(b.mlp1, tm, pre + "mlp.0.weight");
                load_fc(b.mlp2, tm, pre + "mlp.2.weight");
            }
            // Per-block rel_bias concatenated: [depth, H, buckets]
            v.rel_bias.clear();
            for (int i = 0; i < vit_depth; i++) {
                auto rb = must_t(tm, "blocks." + std::to_string(i) +
                                     ".attn.rel_bias").get_floats();
                v.rel_bias.insert(v.rel_bias.end(), rb.begin(), rb.end());
            }
            v.final_ln_g = must_t(tm, "final_norm.weight").get_floats();
            v.final_ln_b = must_t(tm, "final_norm.bias").get_floats();

            load_fc(v.policy_proj, tm, "policy_proj.weight");
            v.pass_logit = must_t(tm, "pass_logit").get_floats().at(0);
            load_fc(v.value_fc1, tm, "value_fc1.weight");
            load_fc(v.value_fc2, tm, "value_fc2.weight");
            load_fc(v.score_mean_fc1, tm, "score_mean_fc1.weight");
            load_fc(v.score_mean_fc2, tm, "score_mean_fc2.weight");
            load_fc(v.score_stdev_fc1, tm, "score_stdev_fc1.weight");
            load_fc(v.score_stdev_fc2, tm, "score_stdev_fc2.weight");
            load_fc(v.ownership_proj, tm, "ownership_proj.weight");

        } else if (format == ModelFormat::KataGo && has("stem.initial_conv.weight")) {
            // ── Converted kata1 network (tools/katago_to_onnx.py) ──
            // kata1 stores BN with its own epsilon (1e-20 upstream).
            const float K_EPS = 1e-20f;
            w->trunk_style = ModelWeights::KataTrunk;
            w->head_style  = ModelWeights::Kata1Heads;
            // kata1 networks use Mish; older ones ReLU.  Opset<=17 exports
            // decompose Mish, so detection counts graph node ops.
            w->use_mish = onnx_parser::graph_uses_mish(model_path);

            load_conv(w->stem_conv, tm, "stem.initial_conv.weight", false);
            load_fc(w->stem_global, tm, "stem.initial_matmul.weight");

            int b = 0;
            while (has("blocks." + std::to_string(b) + ".pre_bn.weight")) {
                std::string pre = "blocks." + std::to_string(b) + ".";
                TrunkBlockW blk;
                bool gp = has(pre + "gpool_conv.weight");
                blk.kind = gp ? TrunkBlockW::KataGPool : TrunkBlockW::KataRegular;
                load_bn(blk.pre_bn, tm, pre + "pre_bn", K_EPS);
                load_conv(blk.regular_conv, tm, pre + "regular_conv.weight", false);
                load_bn(blk.mid_bn, tm, pre + "mid_bn", K_EPS);
                load_conv(blk.final_conv, tm, pre + "final_conv.weight", false);
                if (gp) {
                    load_conv(blk.gpool_conv, tm, pre + "gpool_conv.weight", false);
                    load_bn(blk.gpool_bn, tm, pre + "gpool_bn", K_EPS);
                    load_fc(blk.gpool_to_bias, tm, pre + "gpool_to_bias.weight");
                    blk.pool_channels = blk.gpool_conv.c_out;
                }
                w->blocks.push_back(std::move(blk));
                b++;
            }
            if (w->blocks.empty())
                throw std::runtime_error("kata1 model: no trunk blocks found");
            load_bn(w->tip_bn, tm, "trunk_tip_bn", K_EPS);

            load_conv(w->p1_conv, tm, "policy_head.p1_conv.weight", false);
            load_conv(w->g1_conv, tm, "policy_head.g1_conv.weight", false);
            load_bn(w->g1_bn, tm, "policy_head.g1_bn", K_EPS);
            load_fc(w->k_gpool_to_bias, tm, "policy_head.gpool_to_bias.weight");
            load_bn(w->p1_bn, tm, "policy_head.p1_bn", K_EPS);
            load_conv(w->p2_conv, tm, "policy_head.p2_conv.weight", false);
            load_fc(w->k_gpool_to_pass, tm, "policy_head.gpool_to_pass.weight");

            load_conv(w->v1_conv, tm, "value_head.v1_conv.weight", false);
            load_bn(w->v1_bn, tm, "value_head.v1_bn", K_EPS);
            load_fc(w->v2_mul, tm, "value_head.v2_mul.weight");
            load_fc(w->v3_mul, tm, "value_head.v3_mul.weight");
            load_fc(w->sv3_mul, tm, "value_head.sv3_mul.weight");
            w->v2_bias  = must_t(tm, "value_head.v2_bias").get_floats();
            w->v3_bias  = must_t(tm, "value_head.v3_bias").get_floats();
            w->sv3_bias = must_t(tm, "value_head.sv3_bias").get_floats();
            load_conv(w->vown_conv, tm, "value_head.v_ownership_conv.weight", false);

        } else if (format == ModelFormat::KataGo) {
            // ── Trainable KataGoNet (scripts/model.py --arch katago) ──
            // PyTorch-native: BN eps 1e-5, ReLU, MiniGo-style heads.
            const float EPS = 1e-5f;
            w->trunk_style = ModelWeights::KataTrunk;
            w->head_style  = ModelWeights::MiniGoHeads;
            w->use_mish = false;

            if (!has("stem_conv.weight"))
                throw std::runtime_error(
                    "KataGo-format ONNX with neither kata1 naming "
                    "(stem.initial_conv.*) nor KataGoNet naming (stem_conv.*) — "
                    "unsupported dual-input model");

            load_conv(w->stem_conv, tm, "stem_conv.weight", false);
            load_fc(w->stem_global, tm, "stem_global.weight");

            int b = 0;
            while (has("blocks." + std::to_string(b) + ".conv2.weight")) {
                std::string pre = "blocks." + std::to_string(b) + ".";
                TrunkBlockW blk;
                bool gp = has(pre + "conv_gpool.weight");
                blk.kind = gp ? TrunkBlockW::KataGPool : TrunkBlockW::KataRegular;
                load_bn(blk.pre_bn, tm, pre + "bn1", EPS);
                load_bn(blk.mid_bn, tm, pre + "bn2", EPS);
                if (gp) {
                    load_conv(blk.regular_conv, tm, pre + "conv_regular.weight", false);
                    load_conv(blk.gpool_conv, tm, pre + "conv_gpool.weight", false);
                    load_bn(blk.gpool_bn, tm, pre + "gpool_bn", EPS);
                    load_fc(blk.gpool_to_bias, tm, pre + "gpool_to_bias.weight");
                    blk.pool_channels = blk.gpool_conv.c_out;
                } else {
                    load_conv(blk.regular_conv, tm, pre + "conv1.weight", false);
                }
                load_conv(blk.final_conv, tm, pre + "conv2.weight", false);
                w->blocks.push_back(std::move(blk));
                b++;
            }
            if (w->blocks.empty())
                throw std::runtime_error("KataGoNet model: no trunk blocks found");
            load_bn(w->tip_bn, tm, "tip_bn", EPS);

            load_conv(w->policy_conv, tm, "policy_conv.weight", false);
            load_bn(w->policy_bn, tm, "policy_bn", EPS);
            load_fc(w->policy_fc, tm, "policy_fc.weight");
            load_gpool_head(w->value_head, tm, "value_head", EPS);
            load_gpool_head(w->score_mean_head, tm, "score_mean_head", EPS);
            load_gpool_head(w->score_stdev_head, tm, "score_stdev_head", EPS);
            load_conv(w->ownership_conv, tm, "ownership_conv.weight", true);

        } else {
            // ── MiniGo KataGo-style ResNet (scripts/model.py resnet) ──
            const float EPS = 1e-5f;
            w->trunk_style = ModelWeights::MiniGoTrunk;
            w->head_style  = ModelWeights::MiniGoHeads;

            load_conv(w->input_conv, tm, "input_conv.weight", false);
            load_bn(w->input_bn, tm, "input_bn", EPS);

            int b = 0;
            while (has("trunk." + std::to_string(b) + ".conv2.weight")) {
                std::string pre = "trunk." + std::to_string(b) + ".";
                TrunkBlockW blk;
                bool is_se = has(pre + "conv1.weight");
                if (is_se) {
                    blk.kind = TrunkBlockW::SE;
                    load_conv(blk.conv1, tm, pre + "conv1.weight", false);
                    load_bn(blk.bn1, tm, pre + "bn1", EPS);
                    load_conv(blk.conv2, tm, pre + "conv2.weight", false);
                    load_bn(blk.bn2, tm, pre + "bn2", EPS);
                    load_fc(blk.se_fc1, tm, pre + "se.fc1.weight");
                    load_fc(blk.se_fc2, tm, pre + "se.fc2.weight");
                } else {
                    blk.kind = TrunkBlockW::MgGPool;
                    load_conv(blk.conv_main, tm, pre + "conv_main.weight", false);
                    load_bn(blk.bn_main, tm, pre + "bn_main", EPS);
                    load_conv(blk.conv_pool, tm, pre + "conv_pool.weight", false);
                    load_bn(blk.bn_pool, tm, pre + "bn_pool", EPS);
                    load_fc(blk.pool_fc, tm, pre + "pool_fc.weight");
                    load_conv(blk.conv2, tm, pre + "conv2.weight", false);
                    load_bn(blk.bn2, tm, pre + "bn2", EPS);
                    blk.pool_channels = blk.conv_pool.c_out;
                }
                w->blocks.push_back(std::move(blk));
                b++;
            }
            if (w->blocks.empty())
                throw std::runtime_error("resnet model: no trunk blocks found");

            load_conv(w->policy_conv, tm, "policy_conv.weight", false);
            load_bn(w->policy_bn, tm, "policy_bn", EPS);
            load_fc(w->policy_fc, tm, "policy_fc.weight");
            load_gpool_head(w->value_head, tm, "value_head", EPS);
            load_gpool_head(w->score_mean_head, tm, "score_mean_head", EPS);
            load_gpool_head(w->score_stdev_head, tm, "score_stdev_head", EPS);
            load_conv(w->ownership_conv, tm, "ownership_conv.weight", true);
        }

        weights_ = std::move(w);
    });
    if (!weights_)
        throw std::runtime_error("LoadedModel::weights(): previous load failed");
    return *weights_;
}

}  // namespace minigo
