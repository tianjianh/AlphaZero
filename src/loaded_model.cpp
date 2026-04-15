#include "loaded_model.h"
#include "onnx_loader.h"

#include <cmath>
#include <iostream>
#include <stdexcept>

namespace minigo {

std::shared_ptr<LoadedModel> LoadedModel::load(const std::string& model_path) {
    using namespace onnx_parser;

    auto tensors = parse_onnx_file(model_path);
    std::unordered_map<std::string, OnnxTensor*> tm;
    for (auto& t : tensors) tm[t.name] = &t;

    auto get = [&](const std::string& name) -> OnnxTensor& {
        auto it = tm.find(name);
        if (it == tm.end()) it = tm.find("model." + name);
        if (it == tm.end()) it = tm.find("module." + name);
        if (it == tm.end())
            throw std::runtime_error("Missing tensor: " + name);
        return *it->second;
    };

    auto has = [&](const std::string& name) {
        return tm.count(name) || tm.count("model." + name) || tm.count("module." + name);
    };

    auto load_conv = [&](ConvBNWeights& g, const std::string& wname) {
        auto& wt = get(wname);
        g.c_out = static_cast<int>(wt.dims[0]);
        g.c_in = static_cast<int>(wt.dims[1]);
        g.k = static_cast<int>(wt.dims[2]);
        g.weight = wt.get_floats();
    };

    auto load_bn = [&](ConvBNWeights& g, const std::string& prefix) {
        constexpr float eps = 1e-5f;
        auto gamma = get(prefix + ".weight").get_floats();
        auto beta = get(prefix + ".bias").get_floats();
        auto mean = get(prefix + ".running_mean").get_floats();
        auto var = get(prefix + ".running_var").get_floats();
        int ch = static_cast<int>(gamma.size());
        g.bn_scale.resize(ch);
        g.bn_bias.resize(ch);
        for (int i = 0; i < ch; ++i) {
            float inv_std = 1.0f / std::sqrt(var[i] + eps);
            g.bn_scale[i] = gamma[i] * inv_std;
            g.bn_bias[i] = beta[i] - gamma[i] * mean[i] * inv_std;
        }
    };

    auto load_fc = [&](FCWeights& g, const std::string& prefix) {
        auto& wt = get(prefix + ".weight");
        g.out_features = static_cast<int>(wt.dims[0]);
        g.in_features = static_cast<int>(wt.dims[1]);
        g.weight = wt.get_floats();
        g.bias = get(prefix + ".bias").get_floats();
    };

    if (!has("input_conv.weight") || !has("res_blocks.0.conv1.weight")) {
        throw std::runtime_error(
            "Unsupported ONNX model. Expected the Xiangqi residual network export "
            "with input_conv/res_blocks/policy/value weights.");
    }

    auto model = std::make_shared<LoadedModel>();
    model->model_path = model_path;
    model->model_type = "xiangqi-resnet";

    load_conv(model->input_conv, "input_conv.weight");
    load_bn(model->input_conv, "input_bn");

    model->input_channels = model->input_conv.c_in;
    model->num_filters = model->input_conv.c_out;

    while (has("res_blocks." + std::to_string(model->num_res_blocks) + ".conv1.weight")) {
        model->res_conv1.emplace_back();
        model->res_conv2.emplace_back();
        int idx = model->num_res_blocks;
        load_conv(model->res_conv1.back(), "res_blocks." + std::to_string(idx) + ".conv1.weight");
        load_bn(model->res_conv1.back(), "res_blocks." + std::to_string(idx) + ".bn1");
        load_conv(model->res_conv2.back(), "res_blocks." + std::to_string(idx) + ".conv2.weight");
        load_bn(model->res_conv2.back(), "res_blocks." + std::to_string(idx) + ".bn2");
        model->num_res_blocks++;
    }

    load_conv(model->policy_conv, "policy_conv.weight");
    load_bn(model->policy_conv, "policy_bn");
    load_fc(model->policy_fc, "policy_fc");

    load_conv(model->value_conv, "value_conv.weight");
    load_bn(model->value_conv, "value_bn");
    load_fc(model->value_fc1, "value_fc1");
    load_fc(model->value_fc2, "value_fc2");

    model->action_size = model->policy_fc.out_features;

    std::cout << "Model loaded: type=" << model->model_type
              << " board=" << model->board_rows << "x" << model->board_cols
              << " filters=" << model->num_filters
              << " blocks=" << model->num_res_blocks
              << " channels=" << model->input_channels
              << " actions=" << model->action_size << "\n";
    return model;
}

}  // namespace minigo
