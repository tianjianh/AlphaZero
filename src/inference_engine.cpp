#include "inference_engine.h"
#include "eigen_engine.h"

#ifdef MINIGO_HAS_OPENCL
#include "opencl_engine.h"
#endif

#ifdef MINIGO_HAS_METAL
#include "metal_engine.h"
#endif

#include <stdexcept>

namespace minigo {

std::unique_ptr<InferenceEngine> create_engine(const std::string& model_path) {
    std::unique_ptr<InferenceEngine> engine;

#ifdef MINIGO_HAS_METAL
    engine = std::make_unique<MetalEngine>();
#elif defined(MINIGO_HAS_OPENCL)
    engine = std::make_unique<OpenCLEngine>();
#else
    engine = std::make_unique<EigenEngine>();
#endif

    engine->load_model(model_path);
    return engine;
}

std::string backend_name() {
#ifdef MINIGO_HAS_METAL
    return "metal";
#elif defined(MINIGO_HAS_OPENCL)
    return "opencl";
#else
    return "eigen";
#endif
}

}  // namespace minigo
