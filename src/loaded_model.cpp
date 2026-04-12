#include "loaded_model.h"
#include "onnx_loader.h"
#include <cmath>
#include <iostream>

namespace minigo {

std::shared_ptr<LoadedModel> LoadedModel::load(const std::string& model_path) {
    using namespace onnx_parser;

    auto tensors = parse_onnx_file(model_path);
    std::unordered_map<std::string, OnnxTensor*> tm;
    for (auto& t : tensors) tm[t.name] = &t;

    auto get = [&](const std::string& name) -> OnnxTensor& {
        auto it = tm.find(name);
        if (it == tm.end())
            throw std::runtime_error("Missing tensor: " + name);
        return *it->second;
    };

    auto model = std::make_shared<LoadedModel>();
    model->model_path = model_path;

    // ── Detect model type and infer architecture ─────────────────
    bool is_vit = tm.count("token_proj.weight") > 0;

    if (is_vit) {
        // ViT model — infer from token_proj weights
        model->model_type = "vit";
        auto& tp = get("token_proj.weight");
        model->input_channels = (int)tp.dims[1];
        model->num_filters    = (int)tp.dims[0];  // d_model
        model->num_res_blocks = 0;

        // board_size from orbit_ids buffer: length = board_size^2
        if (tm.count("orbit_ids")) {
            auto& oi = get("orbit_ids");
            int hw = 1;
            for (auto d : oi.dims) hw *= (int)d;
            model->board_size = (int)std::round(std::sqrt((double)hw));
        } else {
            model->board_size = 9;
        }

        // Infer depth: count blocks.N.attn.q_proj.weight
        model->vit_depth = 0;
        while (tm.count("blocks." + std::to_string(model->vit_depth) + ".attn.q_proj.weight"))
            model->vit_depth++;

        // Infer heads from q_proj: shape [num_heads*head_dim, d_model]
        // and kv_groups from kv_proj: shape [2*kv_groups*head_dim, d_model]
        if (model->vit_depth > 0) {
            auto& qw = get("blocks.0.attn.q_proj.weight");
            auto& kvw = get("blocks.0.attn.kv_proj.weight");
            int d_model = model->num_filters;
            int q_out = (int)qw.dims[0];   // num_heads * head_dim
            int kv_out = (int)kvw.dims[0];  // 2 * kv_groups * head_dim
            // head_dim = d_model / num_heads, but num_heads = q_out / head_dim
            // kv_out = 2 * kv_groups * head_dim → kv_groups = kv_out / (2 * head_dim)
            // Since q_out == d_model (full heads), head_dim candidates: 32, 64
            int head_dim = 32;  // standard default
            if (d_model % 64 == 0 && q_out == d_model) head_dim = d_model > 256 ? 64 : 32;
            model->vit_heads = q_out / head_dim;
            model->vit_kv_groups = kv_out / (2 * head_dim);
        }

    } else {
        // ResNet model — only the new KataGo-style layout (alternating SE +
        // GPool blocks + global-pool heads) is supported.  The legacy
        // AlphaZero layout (`res_blocks.*` plain blocks + flat FC heads)
        // has been replaced wholesale; old .onnx files must be retrained.
        bool is_new_resnet = tm.count("trunk.0.conv2.weight") > 0;
        bool is_legacy_resnet = tm.count("res_blocks.0.conv1.weight") > 0;
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
        while (tm.count("trunk." + std::to_string(model->num_res_blocks) + ".conv2.weight"))
            model->num_res_blocks++;

        auto& pfw = get("policy_fc.weight");
        int action_size = (int)pfw.dims[0];
        model->board_size = (int)std::round(std::sqrt((double)(action_size - 1)));
    }

    const float eps = 1e-5f;

    // ── Helper: load conv weights ────────────────────────────────
    auto load_conv = [&](ConvBNWeights& g, const std::string& wname) {
        auto& wt = get(wname);
        g.c_out = (int)wt.dims[0];
        g.c_in  = (int)wt.dims[1];
        g.k     = (int)wt.dims[2];
        g.weight = wt.get_floats();
    };

    // ── Helper: pre-fuse BN into (scale, bias) ──────────────────
    auto load_bn = [&](ConvBNWeights& g, const std::string& prefix) {
        auto gamma = get(prefix + ".weight").get_floats();
        auto beta  = get(prefix + ".bias").get_floats();
        auto mean  = get(prefix + ".running_mean").get_floats();
        auto var   = get(prefix + ".running_var").get_floats();
        int ch = (int)gamma.size();
        g.bn_scale.resize(ch);
        g.bn_bias.resize(ch);
        for (int i = 0; i < ch; i++) {
            float inv_std = 1.0f / std::sqrt(var[i] + eps);
            g.bn_scale[i] = gamma[i] * inv_std;
            g.bn_bias[i]  = beta[i] - gamma[i] * mean[i] * inv_std;
        }
    };

    // ── Helper: load FC weights ─────────────────────────────────
    auto load_fc = [&](FCWeights& g, const std::string& prefix) {
        auto& wt = get(prefix + ".weight");
        g.out_features = (int)wt.dims[0];
        g.in_features  = (int)wt.dims[1];
        g.weight = wt.get_floats();
        g.bias = get(prefix + ".bias").get_floats();
    };

    // ── Metadata-only load ───────────────────────────────────────
    // Both ViT and the new KataGo-style ResNet use TensorRT (which parses
    // the ONNX graph directly and owns its own weight upload).  The per-op
    // helpers load_conv/load_bn/load_fc are kept for the TODO re-enable of
    // Eigen/CUDA/OpenCL/Metal after those backends gain SE + GPool kernels.
    (void)load_conv; (void)load_bn; (void)load_fc;

    if (is_vit) {
        std::cout << "Model loaded: type=vit board=" << model->board_size
                  << " d_model=" << model->num_filters
                  << " depth=" << model->vit_depth
                  << " heads=" << model->vit_heads
                  << " kv_groups=" << model->vit_kv_groups << "\n";
    } else {
        std::cout << "Model loaded: type=resnet board=" << model->board_size
                  << " filters=" << model->num_filters
                  << " blocks=" << model->num_res_blocks
                  << " channels=" << model->input_channels << "\n";
    }
    return model;
}

}  // namespace minigo
