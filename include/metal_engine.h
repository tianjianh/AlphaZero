#pragma once
#ifdef MINIGO_HAS_METAL

#include "inference_engine.h"
#include <string>
#include <vector>

namespace minigo {

// ----------------------------------------------------------------
// Metal inference backend using MPSGraph (macOS Apple Silicon)
//
// Builds the AlphaZero network as an MPSGraph at load_model() time.
// Each predict_batch() call feeds inputs through the pre-compiled graph
// via graph.run() — which internally manages Metal command buffers,
// kernel fusion, and memory planning.
//
// Uses Objective-C++ (.mm) for MPSGraph API access.  The autorelease
// pool is handled by wrapping each graph.run() in @autoreleasepool,
// matching KataGo's Metal backend pattern.
//
// NOT thread-safe: use NNEvaluator to multiplex across threads.
// ----------------------------------------------------------------
class MetalEngine : public InferenceEngine {
public:
    MetalEngine();
    ~MetalEngine() override;

    void load_model(const std::string& path) override;

    std::pair<std::vector<float>, float>
    predict(const std::vector<float>& state) override;

    std::vector<std::pair<std::vector<float>, float>>
    predict_batch(const std::vector<std::vector<float>>& states) override;

    std::string backend_name()   const override { return "metal"; }

private:
    // Opaque pointer to the Objective-C implementation
    // (avoids exposing MPSGraph/Metal headers to C++ consumers)
    struct Impl;
    Impl* impl_ = nullptr;
};

}  // namespace minigo
#endif  // MINIGO_HAS_METAL
