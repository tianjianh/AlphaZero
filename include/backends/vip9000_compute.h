#pragma once
#ifdef MINIGO_HAS_VIP9000

#include "nn/compute_context.h"
#include "model/loaded_model.h"
#include <memory>
#include <string>
#include <vector>

namespace minigo {

// Forward-declared — defined in vip9000_compute.cpp (avoids vip_lite.h here).
struct VIP9000DeviceState;

// ================================================================
// VIP9000 backend (VeriSilicon Vivante VIP9000 NPU, e.g. Allwinner
// A733 NanoDI+ / V853 PICO / RV1106 NanoSI+).
//
// Topology: one NPU core (multi-core variants share one logical
// device).  Multiple server threads share the NPU by each holding
// its own vip_network created from the same NBG bytes; the VIPLite
// driver serializes hardware command submission internally, so
// concurrent vip_run_network() calls on different network handles
// queue cleanly on the single NPU core.
//
// Model file: VIPLite consumes pre-compiled .nb (Network Binary
// Graph) files — one per static batch size.  Path resolution from
// LoadedModel::model_path:
//   1. If the path ends in ".nb", use as-is (single-bs mode).
//   2. If the path is a directory containing "network_binary.nb",
//      use that file.
//   3. Else, strip the ".onnx" / ".unshared.onnx" / ".a733.bsX.fp16"
//      suffixes and look for "<base>.a733.bs<K>.fp16/network_binary.nb"
//      where K is the smallest pre-converted batch ≥ max_batch_size.
//
// The runtime currently accepts max_batch_size ∈ {1, 2, 3, 4} and
// picks bs=1 for max=1, bs=4 for 2..4.  Calls beyond model_batch
// throw — re-convert the ONNX with the desired batch size.
// ================================================================
class VIP9000ComputeContext : public ComputeContext {
public:
    explicit VIP9000ComputeContext(const std::vector<int>& device_ids);
    ~VIP9000ComputeContext();

    std::unique_ptr<ComputeHandle>
    create_handle(const LoadedModel* model, int gpu_id, int max_batch_size) override;

    std::string backend_name() const override { return "vip9000"; }

    VIP9000DeviceState& device_state();

private:
    struct Impl;
    Impl* impl_ = nullptr;
};

class VIP9000ComputeHandle : public ComputeHandle {
public:
    VIP9000ComputeHandle(VIP9000DeviceState& dev, const LoadedModel* model,
                         int max_batch_size, int thread_index);
    ~VIP9000ComputeHandle();

    std::vector<Result>
    predict_batch(const std::vector<std::vector<float>>& states) override;

private:
    struct Impl;
    Impl* impl_ = nullptr;
};

}  // namespace minigo
#endif  // MINIGO_HAS_VIP9000
