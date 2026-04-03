#include "compute_context.h"
#include "eigen_compute.h"

#ifdef MINIGO_HAS_OPENCL
#include "opencl_compute.h"
#endif

#ifdef MINIGO_HAS_METAL
#include "metal_compute.h"
#endif

#ifdef MINIGO_HAS_CUDA
#include "cuda_compute.h"
#endif

#ifdef MINIGO_HAS_TENSORRT
#include "tensorrt_compute.h"
#endif

#include <stdexcept>

namespace minigo {

std::unique_ptr<ComputeContext> create_compute_context(const std::vector<int>& device_ids) {
#ifdef MINIGO_HAS_METAL
    return std::make_unique<MetalComputeContext>(device_ids);
#elif defined(MINIGO_HAS_TENSORRT)
    return std::make_unique<TensorRTComputeContext>(device_ids);
#elif defined(MINIGO_HAS_CUDA)
    return std::make_unique<CUDAComputeContext>(device_ids);
#elif defined(MINIGO_HAS_OPENCL)
    return std::make_unique<OpenCLComputeContext>(device_ids);
#else
    (void)device_ids;
    return std::make_unique<EigenComputeContext>();
#endif
}

std::string backend_name() {
#ifdef MINIGO_HAS_METAL
    return "metal";
#elif defined(MINIGO_HAS_TENSORRT)
    return "tensorrt";
#elif defined(MINIGO_HAS_CUDA)
    return "cuda";
#elif defined(MINIGO_HAS_OPENCL)
    return "opencl";
#else
    return "eigen";
#endif
}

}  // namespace minigo
