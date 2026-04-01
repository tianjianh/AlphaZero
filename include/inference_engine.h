#pragma once

#include <memory>
#include <string>
#include <vector>
#include <utility>

namespace minigo {

struct Config;

// Abstract inference backend interface.
// All backends load .onnx model files as the universal format.
class InferenceEngine {
public:
    virtual ~InferenceEngine() = default;

    virtual void load_model(const std::string& model_path) = 0;

    // Single inference: state (flat C*H*W) -> (softmaxed policy, value in [-1,1])
    virtual std::pair<std::vector<float>, float>
    predict(const std::vector<float>& state) = 0;

    // Batch inference. Default implementation loops over predict().
    virtual std::vector<std::pair<std::vector<float>, float>>
    predict_batch(const std::vector<std::vector<float>>& states) {
        std::vector<std::pair<std::vector<float>, float>> results;
        results.reserve(states.size());
        for (auto& s : states)
            results.push_back(predict(s));
        return results;
    }

    virtual std::string backend_name() const = 0;

    // Model metadata (set after load_model)
    int board_size = 9;
    int input_channels = 17;
    int num_filters = 64;
    int num_res_blocks = 5;
};

// Factory: create the engine for the compile-time selected backend.
// The backend (metal/opencl/eigen) is determined at build time via
// cmake -DMINIGO_BACKEND=... — no runtime selection needed.
std::unique_ptr<InferenceEngine> create_engine(const std::string& model_path);

// Returns the name of the compile-time selected backend ("metal", "opencl", "eigen").
std::string backend_name();

}  // namespace minigo
