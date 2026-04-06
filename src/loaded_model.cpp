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
        model->num_res_blocks = 0;  // not applicable

        // board_size from orbit_ids buffer: length = board_size^2
        if (tm.count("orbit_ids")) {
            auto& oi = get("orbit_ids");
            int hw = 1;
            for (auto d : oi.dims) hw *= (int)d;
            model->board_size = (int)std::round(std::sqrt((double)hw));
        } else {
            model->board_size = 9;
        }

    } else {
        // ResNet model
        model->model_type = "resnet";
        auto& iw = get("input_conv.weight");
        model->input_channels = (int)iw.dims[1];
        model->num_filters    = (int)iw.dims[0];

        model->num_res_blocks = 0;
        while (tm.count("res_blocks." + std::to_string(model->num_res_blocks) + ".conv1.weight"))
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

    // ── Load all weights ─────────────────────────────────────────
    // ViT models use TensorRT (ONNX graph directly) — no CPU weight loading needed
    if (is_vit) {
        std::cout << "Model loaded: type=vit board=" << model->board_size
                  << " d_model=" << model->num_filters
                  << " channels=" << model->input_channels << "\n";
        return model;
    }

    load_conv(model->input_conv, "input_conv.weight");
    load_bn  (model->input_conv, "input_bn");

    model->res_conv1.resize(model->num_res_blocks);
    model->res_conv2.resize(model->num_res_blocks);
    for (int i = 0; i < model->num_res_blocks; i++) {
        std::string pfx = "res_blocks." + std::to_string(i);
        load_conv(model->res_conv1[i], pfx + ".conv1.weight");
        load_bn  (model->res_conv1[i], pfx + ".bn1");
        load_conv(model->res_conv2[i], pfx + ".conv2.weight");
        load_bn  (model->res_conv2[i], pfx + ".bn2");
    }

    load_conv(model->policy_conv, "policy_conv.weight");
    load_bn  (model->policy_conv, "policy_bn");
    load_conv(model->value_conv,  "value_conv.weight");
    load_bn  (model->value_conv,  "value_bn");

    load_fc(model->policy_fc, "policy_fc");
    load_fc(model->value_fc1, "value_fc1");
    load_fc(model->value_fc2, "value_fc2");

    // Score head (always present)
    load_conv(model->score_conv, "score_conv.weight");
    load_bn  (model->score_conv, "score_bn");
    load_fc(model->score_fc1, "score_fc1");
    load_fc(model->score_fc2, "score_fc2");

    std::cout << "Model loaded: type=resnet board=" << model->board_size
              << " filters=" << model->num_filters
              << " blocks=" << model->num_res_blocks
              << " channels=" << model->input_channels << "\n";

    return model;
}

}  // namespace minigo
