#include "nn/compute_context.h"

#ifdef MINIGO_HAS_EIGEN
#include "backends/eigen_compute.h"
#endif

#ifdef MINIGO_HAS_OPENCL
#include "backends/opencl_compute.h"
#endif

#ifdef MINIGO_HAS_METAL
#include "backends/metal_compute.h"
#endif

#ifdef MINIGO_HAS_CUDA
#include "backends/cuda_compute.h"
#endif

#ifdef MINIGO_HAS_TENSORRT
#include "backends/tensorrt_compute.h"
#endif

#ifdef MINIGO_HAS_RKNN
#include "backends/rknn_compute.h"
#endif

#ifdef MINIGO_HAS_VIP9000
#include "backends/vip9000_compute.h"
#endif

#ifdef MINIGO_HAS_K3
#include "backends/k3_compute.h"
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
#elif defined(MINIGO_HAS_RKNN)
    return std::make_unique<RKNNComputeContext>(device_ids);
#elif defined(MINIGO_HAS_VIP9000)
    return std::make_unique<VIP9000ComputeContext>(device_ids);
#elif defined(MINIGO_HAS_K3)
    return std::make_unique<K3ComputeContext>(device_ids);
#elif defined(MINIGO_HAS_OPENCL)
    return std::make_unique<OpenCLComputeContext>(device_ids);
#elif defined(MINIGO_HAS_EIGEN)
    (void)device_ids;
    return std::make_unique<EigenComputeContext>();
#else
    #error "No inference backend compiled. Use cmake -DMINIGO_BACKEND=..."
#endif
}

std::string backend_name() {
#ifdef MINIGO_HAS_METAL
    return "metal";
#elif defined(MINIGO_HAS_TENSORRT)
    return "tensorrt";
#elif defined(MINIGO_HAS_CUDA)
    return "cuda";
#elif defined(MINIGO_HAS_RKNN)
    return "rknn";
#elif defined(MINIGO_HAS_VIP9000)
    return "vip9000";
#elif defined(MINIGO_HAS_K3)
    return "k3";
#elif defined(MINIGO_HAS_OPENCL)
    return "opencl";
#elif defined(MINIGO_HAS_EIGEN)
    return "eigen";
#else
    return "none";
#endif
}

}  // namespace minigo
