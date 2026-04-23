#ifdef MINIGO_HAS_RKNN

#include "rknn_compute.h"
#include "loaded_model.h"

#include "rknn_api.h"

#include <sys/stat.h>

#include <atomic>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace minigo {

// ================================================================
// Error checking
// ================================================================
#define RKNN_CHECK(expr, what)                                                 \
    do {                                                                       \
        int _rc = (expr);                                                      \
        if (_rc != RKNN_SUCC) {                                                \
            std::ostringstream _os;                                            \
            _os << "RKNN error " << _rc << " in " << (what)                    \
                << " at " << __FILE__ << ":" << __LINE__;                      \
            throw std::runtime_error(_os.str());                               \
        }                                                                      \
    } while (0)

// ================================================================
// Helpers
// ================================================================

static std::vector<uint8_t> read_file_bytes(const std::string& path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f)
        throw std::runtime_error("RKNN: cannot open model file: " + path);
    auto sz = f.tellg();
    if (sz <= 0)
        throw std::runtime_error("RKNN: empty model file: " + path);
    std::vector<uint8_t> buf((size_t)sz);
    f.seekg(0);
    f.read(reinterpret_cast<char*>(buf.data()), sz);
    if (!f)
        throw std::runtime_error("RKNN: failed to read model file: " + path);
    return buf;
}

static bool file_exists(const std::string& p) {
    struct stat st;
    return stat(p.c_str(), &st) == 0;
}

// Detect Rockchip SoC and its NPU core count by reading the device
// tree's `compatible` string.  Falls back to 1 core on unknown chips,
// which is safe — pick_core_mask() degenerates to CORE_AUTO.
//
//   rk3562 / rk3566 / rk3568        → 1 NPU core  (0.8–1 TOPS)
//   rk3576                          → 2 NPU cores (6 TOPS, 2×3)
//   rk3588 / rk3588s                → 3 NPU cores (6 TOPS, 3×2)
struct SocInfo {
    std::string name;
    int         num_cores = 1;
};

static SocInfo detect_soc() {
    SocInfo info;
    std::ifstream f("/proc/device-tree/compatible", std::ios::binary);
    if (!f) return info;
    std::string s((std::istreambuf_iterator<char>(f)),
                   std::istreambuf_iterator<char>());
    // /proc/device-tree/compatible is a NUL-separated list of strings.
    auto has = [&](const char* needle) {
        size_t nl = std::strlen(needle);
        for (size_t i = 0; i + nl <= s.size(); i++)
            if (std::memcmp(s.data() + i, needle, nl) == 0) return true;
        return false;
    };
    if      (has("rockchip,rk3588")) { info.name = "rk3588"; info.num_cores = 3; }
    else if (has("rockchip,rk3576")) { info.name = "rk3576"; info.num_cores = 2; }
    else if (has("rockchip,rk3568")) { info.name = "rk3568"; info.num_cores = 1; }
    else if (has("rockchip,rk3566")) { info.name = "rk3566"; info.num_cores = 1; }
    else if (has("rockchip,rk3562")) { info.name = "rk3562"; info.num_cores = 1; }
    return info;
}

// Resolve the .rknn model path from a LoadedModel whose `model_path`
// typically points at the .onnx file.  Policy:
//   1. If the path already ends in ".rknn", use as-is.
//   2. Else, strip any ".onnx" suffix and append ".rknn".
//   3. Verify the resulting file exists.
static std::string resolve_rknn_path(const std::string& model_path) {
    const std::string rknn_ext = ".rknn";
    const std::string onnx_ext = ".onnx";

    auto ends_with = [](const std::string& s, const std::string& suf) {
        return s.size() >= suf.size() &&
               s.compare(s.size() - suf.size(), suf.size(), suf) == 0;
    };

    std::string rknn_path;
    if (ends_with(model_path, rknn_ext)) {
        rknn_path = model_path;
    } else if (ends_with(model_path, onnx_ext)) {
        rknn_path = model_path.substr(0, model_path.size() - onnx_ext.size()) + rknn_ext;
    } else {
        rknn_path = model_path + rknn_ext;
    }

    if (!file_exists(rknn_path)) {
        std::ostringstream os;
        os << "RKNN: compiled model not found at " << rknn_path
           << ".  Convert the ONNX with rknn-toolkit2 (on x86_64 host) "
           << "— see the RKNN section in the README for the conversion command.";
        throw std::runtime_error(os.str());
    }
    return rknn_path;
}

static const char* tensor_type_name(rknn_tensor_type t) {
    switch (t) {
        case RKNN_TENSOR_FLOAT32: return "fp32";
        case RKNN_TENSOR_FLOAT16: return "fp16";
        case RKNN_TENSOR_INT8:    return "i8";
        case RKNN_TENSOR_UINT8:   return "u8";
        case RKNN_TENSOR_INT16:   return "i16";
        case RKNN_TENSOR_INT32:   return "i32";
        default:                  return "?";
    }
}

// ================================================================
// Device state (one per RKNNComputeContext)
//
// Holds the master rknn_context + cached I/O metadata.  The master
// context is created once on the main thread; each server thread
// calls rknn_dup_context() to obtain its own context for inference.
// The runtime shares the model weights across duplicates.
// ================================================================
struct RKNNDeviceState {
    // Model file bytes — kept alive for rknn_init's lifetime of the
    // master context.  rknn_init copies internally but keeping the
    // buffer costs nothing and documents ownership.
    std::vector<uint8_t> model_bytes;
    std::string          rknn_path;

    // Master context (weights live here; duplicates share them).
    rknn_context         master_ctx = 0;
    bool                 master_owned = false;

    // I/O metadata (populated once after master rknn_init).
    uint32_t                        n_inputs  = 0;
    uint32_t                        n_outputs = 0;
    std::vector<rknn_tensor_attr>   input_attrs;
    std::vector<rknn_tensor_attr>   output_attrs;

    // Model shape (from input attr).
    int board_size     = 0;
    int input_channels = 0;
    int action_size    = 0;
    // true if input dim 1 is channels (NCHW), false if dim 3 is channels (NHWC).
    bool input_is_nchw = true;
    int  model_batch   = 1;   // fixed batch dim from the compiled model

    // Output tensor indices, mapped by name.  -1 if not found.
    int policy_idx    = -1;   // policy_logits
    int value_idx     = -1;   // value
    int score_idx     = -1;   // score_mean
    int score_sd_idx  = -1;   // score_stdev
    int ownership_idx = -1;   // ownership

    // SoC identity + NPU core count (for round-robin core assignment),
    // populated from /proc/device-tree/compatible at context init.
    std::string soc_name;
    int         num_cores = 1;

    // Serializes rknn_dup_context calls — the runtime isn't documented as
    // thread-safe for context lifecycle operations.
    std::mutex lifecycle_mu;
};

// ================================================================
// Map output tensors by ONNX export name
// ================================================================
static void map_output_names(RKNNDeviceState& dev) {
    auto name_matches = [](const char* tname, const char* want) {
        // RKNN preserves ONNX output names but may append "/" or "_"
        // separators in some versions.  Match on prefix.
        size_t wl = std::strlen(want);
        return std::strncmp(tname, want, wl) == 0;
    };

    for (uint32_t i = 0; i < dev.n_outputs; i++) {
        const char* name = dev.output_attrs[i].name;
        if      (name_matches(name, "policy_logits")) dev.policy_idx    = (int)i;
        else if (name_matches(name, "value"))         dev.value_idx     = (int)i;
        else if (name_matches(name, "score_mean"))    dev.score_idx     = (int)i;
        else if (name_matches(name, "score_stdev"))   dev.score_sd_idx  = (int)i;
        else if (name_matches(name, "ownership"))     dev.ownership_idx = (int)i;
        else if (name_matches(name, "score") && dev.score_idx < 0)
            dev.score_idx = (int)i;   // legacy "score" alias
    }

    if (dev.policy_idx < 0 || dev.value_idx < 0) {
        // Fallback: identify by output size — policy is the largest
        // output (action_size); value is the smallest (size 1).
        int biggest = -1, biggest_elems = -1;
        int smallest = -1, smallest_elems = INT32_MAX;
        for (uint32_t i = 0; i < dev.n_outputs; i++) {
            int n = (int)dev.output_attrs[i].n_elems;
            if (n > biggest_elems)  { biggest_elems = n; biggest = (int)i; }
            if (n < smallest_elems) { smallest_elems = n; smallest = (int)i; }
        }
        if (dev.policy_idx < 0) dev.policy_idx = biggest;
        if (dev.value_idx  < 0) dev.value_idx  = smallest;
    }
}

// ================================================================
// Init the master context and query I/O
// ================================================================
static void init_master(RKNNDeviceState& dev, const LoadedModel* model) {
    SocInfo soc = detect_soc();
    dev.soc_name  = soc.name;
    dev.num_cores = soc.num_cores;
    std::cout << "RKNN SoC: "
              << (dev.soc_name.empty() ? "unknown" : dev.soc_name)
              << " (" << dev.num_cores << " NPU core"
              << (dev.num_cores == 1 ? "" : "s") << ")\n";

    dev.rknn_path   = resolve_rknn_path(model->model_path);
    dev.model_bytes = read_file_bytes(dev.rknn_path);

    int rc = rknn_init(&dev.master_ctx,
                       dev.model_bytes.data(),
                       (uint32_t)dev.model_bytes.size(),
                       0, nullptr);
    if (rc != RKNN_SUCC) {
        std::ostringstream os;
        os << "RKNN: rknn_init failed (" << rc << ") for " << dev.rknn_path;
        throw std::runtime_error(os.str());
    }
    dev.master_owned = true;

    // SDK version
    rknn_sdk_version sdkv{};
    if (rknn_query(dev.master_ctx, RKNN_QUERY_SDK_VERSION, &sdkv, sizeof(sdkv)) == RKNN_SUCC) {
        std::cout << "RKNN SDK " << sdkv.api_version
                  << " (driver " << sdkv.drv_version << ")\n";
    }

    // I/O counts
    rknn_input_output_num io{};
    RKNN_CHECK(rknn_query(dev.master_ctx, RKNN_QUERY_IN_OUT_NUM, &io, sizeof(io)),
               "RKNN_QUERY_IN_OUT_NUM");
    dev.n_inputs  = io.n_input;
    dev.n_outputs = io.n_output;

    dev.input_attrs.resize(dev.n_inputs);
    dev.output_attrs.resize(dev.n_outputs);
    for (uint32_t i = 0; i < dev.n_inputs; i++) {
        dev.input_attrs[i].index = i;
        RKNN_CHECK(rknn_query(dev.master_ctx, RKNN_QUERY_INPUT_ATTR,
                              &dev.input_attrs[i], sizeof(rknn_tensor_attr)),
                   "RKNN_QUERY_INPUT_ATTR");
    }
    for (uint32_t i = 0; i < dev.n_outputs; i++) {
        dev.output_attrs[i].index = i;
        RKNN_CHECK(rknn_query(dev.master_ctx, RKNN_QUERY_OUTPUT_ATTR,
                              &dev.output_attrs[i], sizeof(rknn_tensor_attr)),
                   "RKNN_QUERY_OUTPUT_ATTR");
    }

    if (dev.n_inputs != 1) {
        std::ostringstream os;
        os << "RKNN: expected 1 input tensor, got " << dev.n_inputs;
        throw std::runtime_error(os.str());
    }

    // Infer [batch, channels, board, board] from the input attr.  The RKNN
    // compiler preserves the ONNX layout (NCHW in our case) but some
    // toolchains reshape to NHWC internally — the `fmt` field tells us.
    const rknn_tensor_attr& in = dev.input_attrs[0];
    if (in.n_dims != 4) {
        std::ostringstream os;
        os << "RKNN: expected 4-D input, got " << in.n_dims << "-D";
        throw std::runtime_error(os.str());
    }
    if (in.fmt == RKNN_TENSOR_NHWC) {
        dev.input_is_nchw   = false;
        dev.model_batch     = (int)in.dims[0];
        dev.board_size      = (int)in.dims[1];
        dev.input_channels  = (int)in.dims[3];
    } else {
        dev.input_is_nchw   = true;
        dev.model_batch     = (int)in.dims[0];
        dev.input_channels  = (int)in.dims[1];
        dev.board_size      = (int)in.dims[2];
    }

    // model_batch > 1 is fine — predict_batch chunks + pads accordingly.

    // Sanity-check against LoadedModel metadata (ONNX-derived).
    if (dev.board_size != model->board_size ||
        dev.input_channels != model->input_channels) {
        std::ostringstream os;
        os << "RKNN: model shape mismatch — .rknn has "
           << dev.input_channels << "C " << dev.board_size << "x" << dev.board_size
           << " but .onnx has " << model->input_channels << "C "
           << model->board_size << "x" << model->board_size
           << ".  Re-convert the ONNX to RKNN.";
        throw std::runtime_error(os.str());
    }
    dev.action_size = dev.board_size * dev.board_size + 1;

    map_output_names(dev);

    std::cout << "RKNN model: " << dev.rknn_path
              << " batch=" << dev.model_batch
              << " C=" << dev.input_channels
              << " board=" << dev.board_size << "x" << dev.board_size
              << " input_fmt=" << (dev.input_is_nchw ? "NCHW" : "NHWC")
              << " input_type=" << tensor_type_name(in.type)
              << "\n";
    std::cout << "RKNN outputs:";
    for (uint32_t i = 0; i < dev.n_outputs; i++) {
        std::cout << " " << dev.output_attrs[i].name
                  << "[" << dev.output_attrs[i].n_elems
                  << "," << tensor_type_name(dev.output_attrs[i].type) << "]";
    }
    std::cout << "\n";
    std::cout << "RKNN mapped: policy=" << dev.policy_idx
              << " value="      << dev.value_idx
              << " score="      << dev.score_idx
              << " score_sd="   << dev.score_sd_idx
              << " ownership="  << dev.ownership_idx << "\n";
}

// ================================================================
// RKNNComputeContext — shared state
// ================================================================

struct RKNNComputeContext::Impl {
    RKNNDeviceState dev;
    std::atomic<int> next_thread_index{0};
};

RKNNComputeContext::RKNNComputeContext(const std::vector<int>& device_ids) {
    impl_ = new Impl();
    // device_ids is informational — RKNN exposes exactly one logical
    // NPU device.  We init the master lazily on first create_handle()
    // since we need a LoadedModel pointer to resolve the .rknn path.
    (void)device_ids;
    for (int id : device_ids) {
        if (id != 0) {
            std::cerr << "RKNN: warning — ignoring non-zero device id "
                      << id << " (only one NPU is exposed)\n";
        }
    }
}

RKNNComputeContext::~RKNNComputeContext() {
    if (impl_) {
        if (impl_->dev.master_owned && impl_->dev.master_ctx)
            rknn_destroy(impl_->dev.master_ctx);
        delete impl_;
    }
}

RKNNDeviceState& RKNNComputeContext::device_state() {
    return impl_->dev;
}

// ================================================================
// RKNNComputeHandle — per-server-thread state
// ================================================================

struct RKNNComputeHandle::Impl {
    RKNNDeviceState& dev;
    rknn_context     ctx = 0;
    bool             ctx_owned = false;

    // Staging buffer sized for one full compiled batch (model_batch × C × H × W),
    // in model-native layout (NCHW or NHWC).  Filled by predict_batch per chunk.
    std::vector<float> input_stage;

    Impl(RKNNDeviceState& d) : dev(d) {}

    ~Impl() {
        if (ctx_owned && ctx) {
            std::lock_guard<std::mutex> lk(dev.lifecycle_mu);
            rknn_destroy(ctx);
        }
    }
};

static rknn_core_mask pick_core_mask(int thread_index, int num_cores) {
    if (num_cores <= 1) return RKNN_NPU_CORE_AUTO;
    switch (thread_index % num_cores) {
        case 0: return RKNN_NPU_CORE_0;
        case 1: return RKNN_NPU_CORE_1;
        case 2: return RKNN_NPU_CORE_2;
        default: return RKNN_NPU_CORE_AUTO;
    }
}

RKNNComputeHandle::RKNNComputeHandle(RKNNDeviceState& dev,
                                     const LoadedModel* model,
                                     int max_batch_size,
                                     int thread_index) {
    (void)max_batch_size;
    impl_ = new Impl(dev);
    auto& I = *impl_;

    // Lazily init the master context on the first handle.  Serialized
    // by lifecycle_mu — only one thread builds, others see it ready.
    {
        std::lock_guard<std::mutex> lk(dev.lifecycle_mu);
        if (!dev.master_owned)
            init_master(dev, model);
    }

    // Duplicate the master context for this thread.  Weights are
    // shared across duplicates; each context holds its own inference
    // state so concurrent rknn_run calls on different dups run in
    // parallel on different NPU cores.
    {
        std::lock_guard<std::mutex> lk(dev.lifecycle_mu);
        int rc = rknn_dup_context(&dev.master_ctx, &I.ctx);
        if (rc != RKNN_SUCC) {
            std::ostringstream os;
            os << "RKNN: rknn_dup_context failed (" << rc << ")";
            throw std::runtime_error(os.str());
        }
        I.ctx_owned = true;
    }

    // Pin this dup to one NPU core — round-robin across cores.
    rknn_core_mask mask = pick_core_mask(thread_index, dev.num_cores);
    int rc = rknn_set_core_mask(I.ctx, mask);
    if (rc != RKNN_SUCC) {
        // Non-fatal: some RKNN platforms don't implement per-core masks
        // (single-core NPUs like RK3566).  Fall back to AUTO.
        std::cerr << "RKNN: rknn_set_core_mask(thread " << thread_index
                  << ") returned " << rc << " — using AUTO core\n";
    }

    int HW = dev.board_size * dev.board_size;
    I.input_stage.resize((size_t)dev.model_batch * dev.input_channels * HW);

    std::cout << "RKNN handle ready: thread=" << thread_index
              << " core_mask=0x" << std::hex << (int)mask << std::dec
              << " board=" << dev.board_size
              << " C=" << dev.input_channels
              << " model_bs=" << dev.model_batch << "\n";
}

RKNNComputeHandle::~RKNNComputeHandle() {
    if (impl_) delete impl_;
}

std::unique_ptr<ComputeHandle>
RKNNComputeContext::create_handle(const LoadedModel* model,
                                  int gpu_id, int max_batch_size) {
    (void)gpu_id;
    int tid = impl_->next_thread_index.fetch_add(1);
    return std::make_unique<RKNNComputeHandle>(
        impl_->dev, model, max_batch_size, tid);
}

// ================================================================
// Inference — one rknn_run per predict_batch call.
//
// The compiled model has a fixed batch dim (model_batch, aka K).
// Contract: caller must keep states.size() <= K (i.e. set
// --max-batch <= K on the CLI).  We pad slots [N..K) with zeros and
// discard their outputs.  Anything > K throws with a clear hint so
// misconfiguration can't silently truncate results.
//
// Parallelism across samples comes from multiple server threads
// pinned to different NPU cores (see rknn_set_core_mask above), not
// from any in-kernel chunking here.
// ================================================================
std::vector<RKNNComputeHandle::Result>
RKNNComputeHandle::predict_batch(
        const std::vector<std::vector<float>>& states) {
    if (states.empty()) return {};

    auto& I = *impl_;
    RKNNDeviceState& dev = I.dev;
    const int N  = (int)states.size();
    const int HW = dev.board_size * dev.board_size;
    const int C  = dev.input_channels;
    const int B  = dev.board_size;
    const int K  = dev.model_batch;
    const int action_size = dev.action_size;
    const size_t sample_floats = (size_t)C * HW;
    const size_t chunk_bytes   = K * sample_floats * sizeof(float);

    if (N > K) {
        std::ostringstream os;
        os << "RKNN: got " << N << " states but compiled model_batch=" << K
           << ".  Lower --max-batch to " << K
           << " or re-convert the ONNX with input_size_list=[["
           << N << "," << C << "," << B << "," << B << "]].";
        throw std::runtime_error(os.str());
    }

    if (I.input_stage.size() < K * sample_floats)
        I.input_stage.resize(K * sample_floats);

    // Fill slots [0..N) with real states; leave [N..K) as zeros.
    std::memset(I.input_stage.data(), 0, chunk_bytes);
    if (dev.input_is_nchw) {
        for (int n = 0; n < N; n++) {
            const auto& s = states[n];
            if ((int)s.size() != C * HW)
                throw std::runtime_error("RKNN: state size mismatch");
            std::memcpy(I.input_stage.data() + n * sample_floats,
                        s.data(), sample_floats * sizeof(float));
        }
    } else {
        // NCHW (C,H,W) → NHWC (H,W,C) into each slot.
        for (int n = 0; n < N; n++) {
            const auto& s = states[n];
            if ((int)s.size() != C * HW)
                throw std::runtime_error("RKNN: state size mismatch");
            const float* src = s.data();
            float* dst = I.input_stage.data() + n * sample_floats;
            for (int h = 0; h < B; h++)
                for (int w = 0; w < B; w++)
                    for (int c = 0; c < C; c++)
                        dst[(h * B + w) * C + c] = src[c * HW + h * B + w];
        }
    }

    rknn_input inputs[1];
    std::memset(inputs, 0, sizeof(inputs));
    inputs[0].index        = 0;
    inputs[0].buf          = I.input_stage.data();
    inputs[0].size         = (uint32_t)chunk_bytes;
    inputs[0].pass_through = 0;
    inputs[0].type         = RKNN_TENSOR_FLOAT32;
    inputs[0].fmt          = dev.input_is_nchw ? RKNN_TENSOR_NCHW
                                               : RKNN_TENSOR_NHWC;

    int rc = rknn_inputs_set(I.ctx, dev.n_inputs, inputs);
    if (rc != RKNN_SUCC)
        throw std::runtime_error("RKNN: rknn_inputs_set failed ("
                                 + std::to_string(rc) + ")");

    rc = rknn_run(I.ctx, nullptr);
    if (rc != RKNN_SUCC)
        throw std::runtime_error("RKNN: rknn_run failed ("
                                 + std::to_string(rc) + ")");

    std::vector<rknn_output> outputs(dev.n_outputs);
    std::memset(outputs.data(), 0, outputs.size() * sizeof(rknn_output));
    for (uint32_t i = 0; i < dev.n_outputs; i++) {
        outputs[i].index       = i;
        outputs[i].want_float  = 1;
        outputs[i].is_prealloc = 0;
    }
    rc = rknn_outputs_get(I.ctx, dev.n_outputs, outputs.data(), nullptr);
    if (rc != RKNN_SUCC)
        throw std::runtime_error("RKNN: rknn_outputs_get failed ("
                                 + std::to_string(rc) + ")");

    // Per-sample stride within each output tensor (total floats / K).
    auto stride_of = [&](int idx) -> int {
        if (idx < 0) return 0;
        int total = (int)(outputs[idx].size / sizeof(float));
        return total / K;
    };
    const int pol_stride = stride_of(dev.policy_idx);
    const int own_stride = stride_of(dev.ownership_idx);

    auto slot_ptr = [&](int idx, int slot) -> const float* {
        return reinterpret_cast<const float*>(outputs[idx].buf)
               + slot * stride_of(idx);
    };

    std::vector<Result> results(N);
    for (int n = 0; n < N; n++) {
        Result& r = results[n];

        if (dev.policy_idx >= 0 && pol_stride > 0) {
            const float* p = slot_ptr(dev.policy_idx, n);
            int count = std::min(action_size, pol_stride);
            r.policy.assign(p, p + count);
            if (count < action_size) r.policy.resize(action_size, 0.0f);
        } else {
            r.policy.assign(action_size, 0.0f);
        }

        r.value    = dev.value_idx    >= 0 ? slot_ptr(dev.value_idx,    n)[0] : 0.0f;
        r.score    = dev.score_idx    >= 0 ? slot_ptr(dev.score_idx,    n)[0] : 0.0f;
        r.score_sd = dev.score_sd_idx >= 0 ? slot_ptr(dev.score_sd_idx, n)[0] : 0.0f;

        if (dev.ownership_idx >= 0 && own_stride > 0) {
            const float* p = slot_ptr(dev.ownership_idx, n);
            int count = std::min(HW, own_stride);
            r.ownership.assign(p, p + count);
            if (count < HW) r.ownership.resize(HW, 0.0f);
        } else {
            r.ownership.assign(HW, 0.0f);
        }
    }

    rknn_outputs_release(I.ctx, dev.n_outputs, outputs.data());
    return results;
}

}  // namespace minigo
#endif  // MINIGO_HAS_RKNN
