#ifdef MINIGO_HAS_TENSORRT

#include "backends/tensorrt_compute.h"
#include "model/loaded_model.h"

#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <cuda_runtime.h>

#include <sys/stat.h>

#include <filesystem>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace minigo {

// ================================================================
// Error checking
// ================================================================
#define CUDA_CHECK(expr)                                                       \
    do {                                                                       \
        cudaError_t _err = (expr);                                             \
        if (_err != cudaSuccess) {                                             \
            std::ostringstream _os;                                            \
            _os << "CUDA error " << cudaGetErrorString(_err)                   \
                << " at " << __FILE__ << ":" << __LINE__;                      \
            throw std::runtime_error(_os.str());                               \
        }                                                                      \
    } while (0)

// ================================================================
// TensorRT logger (singleton)
// ================================================================
class TRTLogger : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING)
            std::cerr << "[TensorRT] " << msg << "\n";
    }
};

static TRTLogger& get_trt_logger() {
    static TRTLogger logger;
    return logger;
}

// ================================================================
// Device state (defined here, forward-declared in header)
//
// One per unique GPU.  Holds the CUDA stream and — once built —
// the compiled TensorRT engine for that device.
// ================================================================
struct TRTDeviceState {
    int          device_id = -1;

    // Serializes ALL engine-level host-side operations on the same
    // ICudaEngine: (1) lazy engine build on first create_handle(),
    // (2) createExecutionContext(), (3) delete exec_ctx.
    //
    // Why: the TRT headers/guide give NO thread-safety guarantee for
    // engine lifecycle operations (NvInferRuntime.h documents thread-
    // safety requirements only for logger/allocator callbacks), and we
    // empirically hit "double free or corruption (out)" at process
    // teardown when ~NNEvaluator woke two same-GPU server threads that
    // then destroyed their exec_ctxs concurrently (the engine tracks
    // its live contexts internally; --nn-device-ids 0,0 / 0,0,1,1).
    // Serializing lifecycle ops fixed it (commit 8465041) and costs
    // nothing off the startup/shutdown path.
    //
    // Note upstream KataGo does NOT need this lock — its trtbackend
    // builds a SEPARATE engine per server thread (weights duplicated
    // per thread on GPU), so no ICudaEngine is ever shared.  We share
    // one engine per device (one weight copy, per-thread exec contexts,
    // which TRT explicitly supports for inference) and therefore pay
    // one mutex on the rare lifecycle path instead.  Inference
    // (enqueueV3, setTensorAddress, setInputShape) stays lock-free —
    // each thread owns its exec_ctx and cudaStreamPerThread.
    std::mutex                              engine_mutex;
    nvinfer1::ICudaEngine*                  engine  = nullptr;
    nvinfer1::IRuntime*                     runtime = nullptr;

    // Cached tensor names + sizes (populated after engine build)
    std::string input_name;            // MiniGo: "state"
    std::string input_spatial_name;    // KataGo: "state_spatial"
    std::string input_global_name;     // KataGo: "state_global"
    std::string policy_name;
    std::string value_name;
    std::string score_name;       // "score_mean"
    std::string score_sd_name;    // "score_stdev"
    std::string ownership_name;   // "ownership"
    std::string precision;        // "FP8", "FP16", or "FP32"

    // NOTE: no shared cudaStream_t.  Each ComputeHandle uses
    // cudaStreamPerThread — CUDA's built-in per-thread implicit stream
    // (matching KataGo's trtbackend.cpp pattern).  Two server threads
    // on the same GPU get independent streams automatically, allowing
    // the GPU hardware scheduler to interleave their inference work
    // with zero host-side contention.  No predict_mutex needed.
};

// ================================================================
// Engine cache helpers
// ================================================================

static std::string make_cache_path(const std::string& model_path,
                                   const std::string& gpu_name,
                                   int max_batch_size,
                                   const std::string& precision) {
    // Sanitize GPU name for filesystem
    std::string safe_name;
    for (char c : gpu_name) {
        if (std::isalnum(c) || c == '_' || c == '-')
            safe_name += c;
        else if (c == ' ')
            safe_name += '_';
    }

    // Extract model filename; cache in a directory shared across workers.
    // MINIGO_TRT_CACHE (if set) is an absolute-or-relative directory path
    // used as the cache root; otherwise fall back to the historical
    // relative "trt_cache/" next to the launching cwd.  The continuous
    // pipeline's supervisor sets MINIGO_TRT_CACHE=<project>/models/trt_cache
    // so gatekeeper + selfplay share plans regardless of their launch cwd.
    std::string base = model_path;
    auto slash = model_path.find_last_of('/');
    if (slash != std::string::npos)
        base = model_path.substr(slash + 1);

    const char* env_cache = std::getenv("MINIGO_TRT_CACHE");
    std::string cache_dir = (env_cache && env_cache[0]) ? env_cache : "trt_cache";
    std::error_code ec;
    std::filesystem::create_directories(cache_dir, ec);  // best-effort; load/store below reports real failures

    // Include precision so FP16/BF16/FP8 engines don't collide
    std::string prec_tag;
    for (char c : precision) prec_tag += (char)std::tolower(c);

    // Include TRT version: serialized engines are not portable across versions
    std::string trt_ver = std::to_string(NV_TENSORRT_MAJOR) + "."
                        + std::to_string(NV_TENSORRT_MINOR) + "."
                        + std::to_string(NV_TENSORRT_PATCH);

    return cache_dir + "/" + base + ".trt" + trt_ver + "_" + safe_name
           + "_b" + std::to_string(max_batch_size)
           + "_" + prec_tag + ".engine";
}

static std::vector<char> read_file(const std::string& path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) return {};
    auto sz = f.tellg();
    if (sz <= 0) return {};
    std::vector<char> buf((size_t)sz);
    f.seekg(0);
    f.read(buf.data(), sz);
    return buf;
}

static bool write_file(const std::string& path, const void* data, size_t size) {
    std::ofstream f(path, std::ios::binary);
    if (!f) return false;
    f.write(static_cast<const char*>(data), (std::streamsize)size);
    return f.good();
}

// Check if cache file is newer than the model file
static bool cache_is_valid(const std::string& cache_path,
                           const std::string& model_path) {
    struct stat model_stat, cache_stat;
    if (stat(model_path.c_str(), &model_stat) != 0) return false;
    if (stat(cache_path.c_str(), &cache_stat) != 0) return false;
    return cache_stat.st_mtime >= model_stat.st_mtime;
}

// ================================================================
// Build or load a TensorRT engine for a given device
//
// Multiple server threads may call this concurrently for the same
// model+GPU.  A per-cache-path mutex ensures only one thread builds
// the engine; the rest wait and load the cached result.
// ================================================================

static std::mutex build_registry_mutex;
static std::map<std::string, std::mutex> build_mutexes;

static std::mutex& get_build_mutex(const std::string& cache_path) {
    std::lock_guard<std::mutex> lock(build_registry_mutex);
    return build_mutexes[cache_path];
}

static nvinfer1::ICudaEngine* build_or_load_engine(
        TRTDeviceState& dev,
        const LoadedModel* model,
        int max_batch_size) {

    CUDA_CHECK(cudaSetDevice(dev.device_id));

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, dev.device_id));
    std::string gpu_name(prop.name);

    std::string cache_path = make_cache_path(model->model_path, gpu_name, max_batch_size, dev.precision);

    // Serialize build per cache path — second thread waits for first to finish
    std::lock_guard<std::mutex> build_lock(get_build_mutex(cache_path));

    // Try loading cached engine (may have been built by another thread)
    if (cache_is_valid(cache_path, model->model_path)) {
        auto plan = read_file(cache_path);
        if (!plan.empty()) {
            auto* engine = dev.runtime->deserializeCudaEngine(plan.data(), plan.size());
            if (engine) {
                std::cout << "TensorRT: loaded cached engine from " << cache_path << "\n";
                return engine;
            }
            std::cerr << "TensorRT: cached engine invalid, rebuilding\n";
        }
    }

    // Build engine from ONNX
    std::cout << "TensorRT: building engine from " << model->model_path
              << " for " << gpu_name << " (max_batch=" << max_batch_size << ")...\n";

    auto* builder = nvinfer1::createInferBuilder(get_trt_logger());
    if (!builder)
        throw std::runtime_error("TensorRT: failed to create builder");

    auto* network = builder->createNetworkV2(0);
    if (!network) {
        delete builder;
        throw std::runtime_error("TensorRT: failed to create network");
    }

    auto* parser = nvonnxparser::createParser(*network, get_trt_logger());
    if (!parser) {
        delete network;
        delete builder;
        throw std::runtime_error("TensorRT: failed to create ONNX parser");
    }

    if (!parser->parseFromFile(model->model_path.c_str(),
                               static_cast<int>(nvinfer1::ILogger::Severity::kWARNING))) {
        std::ostringstream os;
        os << "TensorRT: ONNX parsing failed";
        for (int i = 0; i < parser->getNbErrors(); i++)
            os << "\n  " << parser->getError(i)->desc();
        delete parser;
        delete network;
        delete builder;
        throw std::runtime_error(os.str());
    }

    // Configure builder
    auto* config = builder->createBuilderConfig();
    // Workspace for tactic selection and runtime scratch memory.
    // Freed after build; generous allocation lets TRT pick faster tactics.
    size_t free_mem = 0, total_mem = 0;
    CUDA_CHECK(cudaMemGetInfo(&free_mem, &total_mem));
    config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, free_mem / 2);

    // Enable best available precision: FP8 > BF16 > FP16 > FP32.
    // Detect from SM version (set in init_device) and TRT capability.
    //
    // TRT >= 11 removed weakly-typed networks: BuilderFlag::kFP16/kBF16/
    // kFP8 and platformHasFastFp16() no longer exist, and precision is
    // dictated by the tensor dtypes in the ONNX itself.  Our exporters
    // emit FP32 graphs, so on TRT 11 the engine builds FP32 (TF32 for
    // matmul/conv via the default kTF32 flag).  Reduced-precision on
    // TRT 11 requires exporting FP16/BF16 ONNX — until then, prefer a
    // TRT 10.x install (see README) for FP16/BF16 engines.
    std::string build_prec = "FP32";
#if NV_TENSORRT_MAJOR >= 11
    std::cout << "TensorRT " << NV_TENSORRT_MAJOR
              << ": strongly-typed build — engine precision follows the "
                 "ONNX dtypes (FP32 graph -> FP32/TF32 engine)\n";
#else
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
    if (dev.precision == "FP8") {
        config->setFlag(nvinfer1::BuilderFlag::kFP8);
        config->setFlag(nvinfer1::BuilderFlag::kFP16);  // FP8 needs FP16 fallback layers
        build_prec = "FP8";
    } else if (dev.precision == "BF16") {
        config->setFlag(nvinfer1::BuilderFlag::kBF16);
        build_prec = "BF16";
    } else if (builder->platformHasFastFp16()) {
        config->setFlag(nvinfer1::BuilderFlag::kFP16);
        build_prec = "FP16";
    }
#pragma GCC diagnostic pop
#endif
    std::cout << "TensorRT: building with " << build_prec << " precision\n";

    // Optimization profile for dynamic batch size [1, max_batch_size].
    // KataGo networks have 2 inputs (spatial + global); MiniGo has 1.
    // Set min/opt/max dims for every input.
    auto* profile = builder->createOptimizationProfile();
    int nb_inputs = network->getNbInputs();
    int opt_batch = std::max(1, max_batch_size / 2);
    for (int i = 0; i < nb_inputs; ++i) {
        auto* inp = network->getInput(i);
        auto dims = inp->getDimensions();
        // dims.d[0] is the dynamic batch (-1); rest are fixed.
        nvinfer1::Dims min_dims = dims, opt_dims = dims, max_dims = dims;
        min_dims.d[0] = 1;
        opt_dims.d[0] = opt_batch;
        max_dims.d[0] = max_batch_size;
        profile->setDimensions(inp->getName(),
                               nvinfer1::OptProfileSelector::kMIN, min_dims);
        profile->setDimensions(inp->getName(),
                               nvinfer1::OptProfileSelector::kOPT, opt_dims);
        profile->setDimensions(inp->getName(),
                               nvinfer1::OptProfileSelector::kMAX, max_dims);
    }
    config->addOptimizationProfile(profile);

    // Build serialized engine
    auto* serialized = builder->buildSerializedNetwork(*network, *config);
    if (!serialized) {
        delete config;
        delete parser;
        delete network;
        delete builder;
        throw std::runtime_error("TensorRT: engine build failed");
    }

    // Deserialize into engine
    auto* engine = dev.runtime->deserializeCudaEngine(
        serialized->data(), serialized->size());

    // Cache to disk
    if (engine) {
        if (write_file(cache_path, serialized->data(), serialized->size()))
            std::cout << "TensorRT: cached engine to " << cache_path << "\n";
    }

    delete serialized;
    delete config;
    delete parser;
    delete network;
    delete builder;

    if (!engine)
        throw std::runtime_error("TensorRT: engine deserialization failed");

    return engine;
}

// ================================================================
// TensorRTComputeContext
// ================================================================

struct TensorRTComputeContext::Impl {
    std::map<int, TRTDeviceState> devices;
};

static void init_device(TRTDeviceState& ds, int device_id) {
    int device_count = 0;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));
    if (device_count == 0)
        throw std::runtime_error("No CUDA devices found");
    if (device_id < 0 || device_id >= device_count)
        throw std::runtime_error("CUDA device_id " + std::to_string(device_id) +
                                 " out of range [0, " + std::to_string(device_count) + ")");

    ds.device_id = device_id;
    CUDA_CHECK(cudaSetDevice(device_id));

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device_id));
    // Detect best precision from SM version.
    // BF16 preferred on Ampere+ (SM 8.0+): same throughput as FP16,
    // but 8 exponent bits (same as FP32) so no overflow at 65504.
    // FP16 has only 5 exponent bits → score head can overflow to NaN.
    // FP8 is NOT auto-selected — it requires calibration and explicit
    // opt-in.  Use BF16 as the default for SM 8.0+ (Ampere/Blackwell).
    if (prop.major >= 8)
        ds.precision = "BF16";
    else if (prop.major >= 7 || (prop.major == 6 && prop.minor >= 0))
        ds.precision = "FP16";
    else
        ds.precision = "FP32";

    std::cout << "TensorRT device " << device_id << ": " << prop.name
              << " (SM " << prop.major << "." << prop.minor
              << ", " << ds.precision << ")\n";

    // No stream created here — each ComputeHandle uses cudaStreamPerThread.

    ds.runtime = nvinfer1::createInferRuntime(get_trt_logger());
    if (!ds.runtime)
        throw std::runtime_error("TensorRT: failed to create runtime for device " +
                                 std::to_string(device_id));
}

TensorRTComputeContext::TensorRTComputeContext(const std::vector<int>& device_ids) {
    impl_ = new Impl();
    for (int id : device_ids) {
        if (impl_->devices.count(id)) continue;
        auto& ds = impl_->devices[id];
        init_device(ds, id);
    }
}

TensorRTComputeContext::~TensorRTComputeContext() {
    if (impl_) {
        for (auto& [id, ds] : impl_->devices) {
            cudaSetDevice(ds.device_id);
            if (ds.engine)  delete ds.engine;
            if (ds.runtime) delete ds.runtime;
            // No stream to destroy — handles use cudaStreamPerThread.
        }
        delete impl_;
    }
}

TRTDeviceState& TensorRTComputeContext::device_state(int gpu_id) {
    auto it = impl_->devices.find(gpu_id);
    if (it == impl_->devices.end())
        throw std::runtime_error("TensorRT device " + std::to_string(gpu_id) + " not initialized");
    return it->second;
}

std::unique_ptr<ComputeHandle>
TensorRTComputeContext::create_handle(const LoadedModel* model,
                                      int gpu_id, int max_batch_size) {
    return std::make_unique<TensorRTComputeHandle>(
        device_state(gpu_id), model, max_batch_size);
}

// ================================================================
// TensorRTComputeHandle::Impl — per-thread execution state
// ================================================================

struct TensorRTComputeHandle::Impl {
    TRTDeviceState& dev;
    nvinfer1::IExecutionContext* exec_ctx = nullptr;

    ModelFormat format = ModelFormat::MiniGo;
    int board_size            = 0;
    int input_channels        = 0;       // MiniGo: 17; KataGo: 22 (spatial)
    int input_global_channels = 0;       // KataGo: 19
    int max_batch_size = 0;

    // I/O device buffers (256-byte aligned as required by TRT 10)
    float* d_input          = nullptr;   // MiniGo: [max_batch, C, H, W]
    float* d_input_spatial  = nullptr;   // KataGo: [max_batch, 22, H, W]
    float* d_input_global   = nullptr;   // KataGo: [max_batch, 19]
    float* d_policy    = nullptr;   // [max_batch, action_size]
    float* d_value     = nullptr;   // [max_batch, 1]
    float* d_score     = nullptr;   // [max_batch, 1]
    float* d_score_sd  = nullptr;   // [max_batch, 1]
    float* d_ownership = nullptr;   // [max_batch, board²]

    // Tensor names from engine
    std::string input_name;
    std::string input_spatial_name;
    std::string input_global_name;
    std::string policy_name;
    std::string value_name;
    std::string score_name;
    std::string score_sd_name;
    std::string ownership_name;

    // Persistent host staging buffers, sized once for max_batch_size
    // (KataGo pattern: NNServerBuf/InputBuffers allocated per server
    // thread, reused for every batch — trtbackend.cpp InputBuffers).
    // Without these, every predict_batch call heap-allocates ~3 MB of
    // std::vectors on the hot path.
    std::vector<float> h_input;          // MiniGo flat input / KataGo spatial
    std::vector<float> h_input_global;   // KataGo global
    std::vector<float> h_policy;
    std::vector<float> h_value;
    std::vector<float> h_score;
    std::vector<float> h_score_sd;
    std::vector<float> h_ownership;

    Impl(TRTDeviceState& d) : dev(d) {}

    ~Impl() {
        cudaSetDevice(dev.device_id);
        // Drain this thread's implicit stream before freeing device
        // buffers.  If a prior predict_batch threw mid-inference, the
        // stream may still have pending ops referencing d_input etc.
        // NOTE: the handle destructor runs on the same server thread
        // that created the handle (local variable in server_loop), so
        // cudaStreamPerThread refers to this thread's own stream.
        cudaStreamSynchronize(cudaStreamPerThread);
        // delete exec_ctx must be serialized per engine — ~NNEvaluator
        // notifies all server threads at once, so two threads on the
        // same GPU race to tear down exec_ctxs that share an engine.
        // Without this lock the engine's internal context list is
        // corrupted and ~ICudaEngine later crashes with a "double
        // free or corruption (out)" after "Done!".  Stream sync above
        // stays outside the lock (it's per-thread, not per-engine).
        if (exec_ctx) {
            std::lock_guard<std::mutex> lock(dev.engine_mutex);
            delete exec_ctx;
        }
        if (d_input)         cudaFree(d_input);
        if (d_input_spatial) cudaFree(d_input_spatial);
        if (d_input_global)  cudaFree(d_input_global);
        if (d_policy)    cudaFree(d_policy);
        if (d_value)     cudaFree(d_value);
        if (d_score)     cudaFree(d_score);
        if (d_score_sd)  cudaFree(d_score_sd);
        if (d_ownership) cudaFree(d_ownership);
    }
};

TensorRTComputeHandle::TensorRTComputeHandle(TRTDeviceState& dev,
                                             const LoadedModel* model,
                                             int max_batch_size) {
    CUDA_CHECK(cudaSetDevice(dev.device_id));

    if (max_batch_size <= 0) max_batch_size = 32;

    // Build engine if not yet built for this device (thread-safe)
    {
        std::lock_guard<std::mutex> lock(dev.engine_mutex);
        if (!dev.engine) {
            dev.engine = build_or_load_engine(dev, model, max_batch_size);

            // Discover I/O tensor names by matching ONNX export names.
            // KataGo (two inputs): "state_spatial", "state_global".
            // MiniGo (single input): single nameless / "state" tensor.
            int nb = dev.engine->getNbIOTensors();
            for (int i = 0; i < nb; i++) {
                const char* name = dev.engine->getIOTensorName(i);
                auto mode = dev.engine->getTensorIOMode(name);
                std::string sname(name);
                if (mode == nvinfer1::TensorIOMode::kINPUT) {
                    if      (sname == "state_spatial") dev.input_spatial_name = name;
                    else if (sname == "state_global")  dev.input_global_name  = name;
                    else                                dev.input_name         = name;
                } else {
                    if (sname == "policy_logits")
                        dev.policy_name = name;
                    else if (sname == "value")
                        dev.value_name = name;
                    else if (sname == "score_mean" || sname == "score")
                        dev.score_name = name;
                    else if (sname == "score_stdev")
                        dev.score_sd_name = name;
                    else if (sname == "ownership")
                        dev.ownership_name = name;
                    else {
                        // Fallback: identify by shape for unknown names
                        auto dims = dev.engine->getTensorShape(name);
                        if (dims.nbDims >= 2 && dims.d[dims.nbDims - 1] > 1)
                            dev.policy_name = name;
                        else if (dev.value_name.empty())
                            dev.value_name = name;
                    }
                }
            }

            bool is_katago_engine =
                !dev.input_spatial_name.empty() && !dev.input_global_name.empty();
            bool is_minigo_engine = !dev.input_name.empty();

            if (model->format == ModelFormat::KataGo && !is_katago_engine)
                throw std::runtime_error(
                    "TensorRT: KataGo model loaded but engine is missing "
                    "state_spatial / state_global inputs.");
            if (model->format == ModelFormat::MiniGo && !is_minigo_engine)
                throw std::runtime_error(
                    "TensorRT: MiniGo model loaded but engine has no "
                    "single-input tensor.");

            if (dev.policy_name.empty() || dev.value_name.empty()
                || (!is_katago_engine && !is_minigo_engine)) {
                std::ostringstream os;
                os << "TensorRT: could not identify I/O tensors. Found " << nb << " tensors:";
                for (int i = 0; i < nb; i++) {
                    const char* name = dev.engine->getIOTensorName(i);
                    auto dims = dev.engine->getTensorShape(name);
                    auto mode = dev.engine->getTensorIOMode(name);
                    os << "\n  " << name << " ["
                       << (mode == nvinfer1::TensorIOMode::kINPUT ? "INPUT" : "OUTPUT") << "] (";
                    for (int d = 0; d < dims.nbDims; d++) {
                        if (d > 0) os << ",";
                        os << dims.d[d];
                    }
                    os << ")";
                }
                throw std::runtime_error(os.str());
            }

            if (is_katago_engine) {
                std::cout << "TensorRT I/O: input_spatial=\"" << dev.input_spatial_name
                          << "\" input_global=\"" << dev.input_global_name
                          << "\" policy=\"" << dev.policy_name
                          << "\" value=\"" << dev.value_name
                          << "\" score=\"" << dev.score_name
                          << "\" score_sd=\"" << dev.score_sd_name
                          << "\" ownership=\"" << dev.ownership_name << "\"\n";
            } else {
                std::cout << "TensorRT I/O: input=\"" << dev.input_name
                          << "\" policy=\"" << dev.policy_name
                          << "\" value=\"" << dev.value_name
                          << "\" score=\"" << dev.score_name
                          << "\" score_sd=\"" << dev.score_sd_name
                          << "\" ownership=\"" << dev.ownership_name << "\"\n";
            }
        }
    }

    impl_ = new Impl(dev);
    // From here on, any throw must release Impl ourselves: the object
    // isn't constructed yet, so ~TensorRTComputeHandle won't run.
    // ~Impl frees whatever was allocated before the failure.
    try {
    auto& I = *impl_;

    I.format                = model->format;
    I.board_size            = model->board_size;
    I.input_channels        = model->input_channels;
    I.input_global_channels = model->input_global_channels;
    I.max_batch_size        = max_batch_size;
    I.input_name            = dev.input_name;
    I.input_spatial_name    = dev.input_spatial_name;
    I.input_global_name     = dev.input_global_name;
    I.policy_name           = dev.policy_name;
    I.value_name            = dev.value_name;
    I.score_name            = dev.score_name;
    I.score_sd_name         = dev.score_sd_name;
    I.ownership_name        = dev.ownership_name;

    // Create per-thread execution context.  Serialized per engine via
    // dev.engine_mutex — see TRTDeviceState comment above for why.
    {
        std::lock_guard<std::mutex> lock(dev.engine_mutex);
        I.exec_ctx = dev.engine->createExecutionContext();
    }
    if (!I.exec_ctx)
        throw std::runtime_error("TensorRT: failed to create execution context");

    // Allocate I/O buffers
    int HW = I.board_size * I.board_size;
    int action_size = HW + 1;

    size_t policy_bytes = (size_t)max_batch_size * action_size * sizeof(float);
    size_t value_bytes  = (size_t)max_batch_size * 1 * sizeof(float);

    if (I.format == ModelFormat::KataGo) {
        size_t spatial_bytes = (size_t)max_batch_size * I.input_channels * HW * sizeof(float);
        size_t global_bytes  = (size_t)max_batch_size * I.input_global_channels * sizeof(float);
        CUDA_CHECK(cudaMalloc(&I.d_input_spatial, spatial_bytes));
        CUDA_CHECK(cudaMalloc(&I.d_input_global,  global_bytes));
    } else {
        size_t input_bytes  = (size_t)max_batch_size * I.input_channels * HW * sizeof(float);
        CUDA_CHECK(cudaMalloc(&I.d_input,  input_bytes));
    }
    CUDA_CHECK(cudaMalloc(&I.d_policy, policy_bytes));
    CUDA_CHECK(cudaMalloc(&I.d_value,  value_bytes));
    CUDA_CHECK(cudaMalloc(&I.d_score,     (size_t)max_batch_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&I.d_score_sd,  (size_t)max_batch_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&I.d_ownership, (size_t)max_batch_size * HW * sizeof(float)));

    // Host staging buffers — allocated once, reused every predict_batch.
    if (I.format == ModelFormat::KataGo) {
        I.h_input.resize((size_t)max_batch_size * I.input_channels * HW);
        I.h_input_global.resize((size_t)max_batch_size * I.input_global_channels);
    } else {
        I.h_input.resize((size_t)max_batch_size * I.input_channels * HW);
    }
    I.h_policy.resize((size_t)max_batch_size * action_size);
    I.h_value.resize((size_t)max_batch_size);
    I.h_score.resize((size_t)max_batch_size);
    I.h_score_sd.resize((size_t)max_batch_size);
    I.h_ownership.resize((size_t)max_batch_size * HW);

    std::cout << "TensorRT handle ready (" << I.dev.precision << "): board=" << I.board_size;
    if (model->model_type == "vit")
        std::cout << " d_model=" << model->num_filters
                  << " depth=" << model->vit_depth
                  << " heads=" << model->vit_heads
                  << " kv=" << model->vit_kv_groups;
    else
        std::cout << " filters=" << model->num_filters
                  << " blocks=" << model->num_res_blocks;
    std::cout << "\n";
    } catch (...) {
        delete impl_;
        impl_ = nullptr;
        throw;
    }
}

TensorRTComputeHandle::~TensorRTComputeHandle() {
    if (impl_) {
        cudaSetDevice(impl_->dev.device_id);
        delete impl_;
    }
}

// ================================================================
// Inference
// ================================================================

std::vector<TensorRTComputeHandle::Result>
TensorRTComputeHandle::predict_batch(
        const std::vector<std::vector<float>>& states) {
    if (states.empty()) return {};

    auto& I = *impl_;
    CUDA_CHECK(cudaSetDevice(I.dev.device_id));

    // Each server thread uses cudaStreamPerThread — its own implicit
    // CUDA stream.  No mutex needed; two threads on the same GPU
    // submit to independent streams and the GPU interleaves them.

    int N = (int)states.size();
    int H = I.board_size, W = I.board_size;
    int HW = H * W;
    int action_size = HW + 1;

    if (N > I.max_batch_size)
        throw std::runtime_error(
            "TensorRT: batch " + std::to_string(N) + " exceeds max_batch_size "
            + std::to_string(I.max_batch_size));

    if (I.format == ModelFormat::KataGo) {
        // Encoder packs each state as: [22*H*W spatial floats][19 global floats].
        const int sp_per = I.input_channels * HW;
        const int gl_per = I.input_global_channels;
        for (int n = 0; n < N; ++n) {
            const auto& s = states[n];
            if ((int)s.size() != sp_per + gl_per)
                throw std::runtime_error(
                    "TensorRT KataGo: state size mismatch (" +
                    std::to_string(s.size()) + " vs expected " +
                    std::to_string(sp_per + gl_per) + ")");
            std::memcpy(I.h_input.data() + (size_t)n * sp_per,
                        s.data(), sp_per * sizeof(float));
            std::memcpy(I.h_input_global.data() + (size_t)n * gl_per,
                        s.data() + sp_per, gl_per * sizeof(float));
        }
        CUDA_CHECK(cudaMemcpyAsync(I.d_input_spatial, I.h_input.data(),
            (size_t)N * sp_per * sizeof(float),
            cudaMemcpyHostToDevice, cudaStreamPerThread));
        CUDA_CHECK(cudaMemcpyAsync(I.d_input_global, I.h_input_global.data(),
            (size_t)N * gl_per * sizeof(float),
            cudaMemcpyHostToDevice, cudaStreamPerThread));

        nvinfer1::Dims4 sp_dims(N, I.input_channels, H, W);
        nvinfer1::Dims2 gl_dims(N, I.input_global_channels);
        if (!I.exec_ctx->setInputShape(I.input_spatial_name.c_str(), sp_dims))
            throw std::runtime_error("TensorRT: setInputShape(state_spatial) failed");
        if (!I.exec_ctx->setInputShape(I.input_global_name.c_str(), gl_dims))
            throw std::runtime_error("TensorRT: setInputShape(state_global) failed");
        if (!I.exec_ctx->setTensorAddress(I.input_spatial_name.c_str(), I.d_input_spatial))
            throw std::runtime_error("TensorRT: setTensorAddress(state_spatial) failed");
        if (!I.exec_ctx->setTensorAddress(I.input_global_name.c_str(), I.d_input_global))
            throw std::runtime_error("TensorRT: setTensorAddress(state_global) failed");
    } else {
        // Flatten input states and upload (MiniGo single input)
        const int in_per = I.input_channels * HW;
        for (int n = 0; n < N; ++n) {
            const auto& s = states[n];
            if ((int)s.size() != in_per)
                throw std::runtime_error(
                    "TensorRT: state size mismatch (" +
                    std::to_string(s.size()) + " vs expected " +
                    std::to_string(in_per) + ")");
            std::memcpy(I.h_input.data() + (size_t)n * in_per,
                        s.data(), in_per * sizeof(float));
        }
        CUDA_CHECK(cudaMemcpyAsync(I.d_input, I.h_input.data(),
            (size_t)N * in_per * sizeof(float),
            cudaMemcpyHostToDevice, cudaStreamPerThread));

        nvinfer1::Dims4 input_dims(N, I.input_channels, H, W);
        if (!I.exec_ctx->setInputShape(I.input_name.c_str(), input_dims))
            throw std::runtime_error("TensorRT: setInputShape failed");
        if (!I.exec_ctx->setTensorAddress(I.input_name.c_str(),  I.d_input))
            throw std::runtime_error("TensorRT: setTensorAddress(input) failed");
    }

    if (!I.exec_ctx->setTensorAddress(I.policy_name.c_str(), I.d_policy))
        throw std::runtime_error("TensorRT: setTensorAddress(policy) failed");
    if (!I.exec_ctx->setTensorAddress(I.value_name.c_str(),  I.d_value))
        throw std::runtime_error("TensorRT: setTensorAddress(value) failed");
    if (!I.score_name.empty()) {
        if (!I.exec_ctx->setTensorAddress(I.score_name.c_str(), I.d_score))
            throw std::runtime_error("TensorRT: setTensorAddress(score_mean) failed");
    }
    if (!I.score_sd_name.empty()) {
        if (!I.exec_ctx->setTensorAddress(I.score_sd_name.c_str(), I.d_score_sd))
            throw std::runtime_error("TensorRT: setTensorAddress(score_stdev) failed");
    }
    if (!I.ownership_name.empty()) {
        if (!I.exec_ctx->setTensorAddress(I.ownership_name.c_str(), I.d_ownership))
            throw std::runtime_error("TensorRT: setTensorAddress(ownership) failed");
    }

    // Run inference
    if (!I.exec_ctx->enqueueV3(cudaStreamPerThread))
        throw std::runtime_error("TensorRT: enqueueV3 failed");

    // Read back results into the persistent staging buffers
    CUDA_CHECK(cudaMemcpyAsync(I.h_policy.data(), I.d_policy,
        (size_t)N * action_size * sizeof(float), cudaMemcpyDeviceToHost, cudaStreamPerThread));
    CUDA_CHECK(cudaMemcpyAsync(I.h_value.data(), I.d_value,
        (size_t)N * sizeof(float), cudaMemcpyDeviceToHost, cudaStreamPerThread));

    if (!I.score_name.empty())
        CUDA_CHECK(cudaMemcpyAsync(I.h_score.data(), I.d_score,
            (size_t)N * sizeof(float), cudaMemcpyDeviceToHost, cudaStreamPerThread));
    if (!I.score_sd_name.empty())
        CUDA_CHECK(cudaMemcpyAsync(I.h_score_sd.data(), I.d_score_sd,
            (size_t)N * sizeof(float), cudaMemcpyDeviceToHost, cudaStreamPerThread));
    if (!I.ownership_name.empty())
        CUDA_CHECK(cudaMemcpyAsync(I.h_ownership.data(), I.d_ownership,
            (size_t)N * HW * sizeof(float), cudaMemcpyDeviceToHost, cudaStreamPerThread));

    CUDA_CHECK(cudaStreamSynchronize(cudaStreamPerThread));

    // Pack results — TRT outputs are row-major [N, action_size] and [N, 1].
    // Missing optional heads read zeros: the staging buffers are
    // zero-initialised at allocation and never written for absent heads.
    std::vector<TensorRTComputeHandle::Result> results(N);
    for (int n = 0; n < N; n++) {
        results[n].policy.assign(I.h_policy.begin() + (size_t)n * action_size,
                                 I.h_policy.begin() + (size_t)(n + 1) * action_size);
        results[n].value    = I.h_value[n];
        results[n].score    = I.h_score[n];
        results[n].score_sd = I.h_score_sd[n];
        results[n].ownership.assign(I.h_ownership.begin() + (size_t)n * HW,
                                    I.h_ownership.begin() + (size_t)(n + 1) * HW);
    }
    return results;
}

}  // namespace minigo
#endif  // MINIGO_HAS_TENSORRT
