#ifdef MINIGO_HAS_TENSORRT

#include "tensorrt_compute.h"
#include "loaded_model.h"

#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <cuda_runtime.h>

#include <sys/stat.h>

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

    // Engine is built lazily on first create_handle() for this device.
    // Protected by build_mutex so only the first thread builds it.
    std::mutex                              build_mutex;
    nvinfer1::ICudaEngine*                  engine  = nullptr;
    nvinfer1::IRuntime*                     runtime = nullptr;

    // Cached tensor names + sizes (populated after engine build)
    std::string input_name;
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

    // Extract model filename; cache in a fixed trt_cache/ directory
    std::string base = model_path;
    auto slash = model_path.find_last_of('/');
    if (slash != std::string::npos)
        base = model_path.substr(slash + 1);

    std::string cache_dir = "trt_cache";
    system(("mkdir -p " + cache_dir).c_str());

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
    cudaMemGetInfo(&free_mem, &total_mem);
    config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, free_mem / 2);

    // Enable best available precision: FP8 > FP16 > FP32
    // Detect from SM version (set in init_device) and TRT capability.
    std::string build_prec = "FP32";
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
    std::cout << "TensorRT: building with " << build_prec << " precision\n";

    // Optimization profile for dynamic batch size [1, max_batch_size]
    auto* profile = builder->createOptimizationProfile();
    auto* input_tensor = network->getInput(0);
    auto input_dims = input_tensor->getDimensions();
    // input_dims is [N, C, H, W] — N is dynamic (-1)
    nvinfer1::Dims4 min_dims(1, input_dims.d[1], input_dims.d[2], input_dims.d[3]);
    nvinfer1::Dims4 opt_dims(std::max(1, max_batch_size / 2),
                             input_dims.d[1], input_dims.d[2], input_dims.d[3]);
    nvinfer1::Dims4 max_dims(max_batch_size,
                             input_dims.d[1], input_dims.d[2], input_dims.d[3]);
    profile->setDimensions(input_tensor->getName(),
                           nvinfer1::OptProfileSelector::kMIN, min_dims);
    profile->setDimensions(input_tensor->getName(),
                           nvinfer1::OptProfileSelector::kOPT, opt_dims);
    profile->setDimensions(input_tensor->getName(),
                           nvinfer1::OptProfileSelector::kMAX, max_dims);
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

    int board_size     = 0;
    int input_channels = 0;
    int max_batch_size = 0;

    // I/O device buffers (256-byte aligned as required by TRT 10)
    float* d_input     = nullptr;   // [max_batch, C, H, W]
    float* d_policy    = nullptr;   // [max_batch, action_size]
    float* d_value     = nullptr;   // [max_batch, 1]
    float* d_score     = nullptr;   // [max_batch, 1]
    float* d_score_sd  = nullptr;   // [max_batch, 1]
    float* d_ownership = nullptr;   // [max_batch, board²]

    // Tensor names from engine
    std::string input_name;
    std::string policy_name;
    std::string value_name;
    std::string score_name;
    std::string score_sd_name;
    std::string ownership_name;

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
        if (exec_ctx)   delete exec_ctx;
        if (d_input)     cudaFree(d_input);
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
        std::lock_guard<std::mutex> lock(dev.build_mutex);
        if (!dev.engine) {
            dev.engine = build_or_load_engine(dev, model, max_batch_size);

            // Discover I/O tensor names by matching ONNX export names
            int nb = dev.engine->getNbIOTensors();
            for (int i = 0; i < nb; i++) {
                const char* name = dev.engine->getIOTensorName(i);
                auto mode = dev.engine->getTensorIOMode(name);
                if (mode == nvinfer1::TensorIOMode::kINPUT) {
                    dev.input_name = name;
                } else {
                    std::string sname(name);
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

            if (dev.input_name.empty() || dev.policy_name.empty() || dev.value_name.empty()) {
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

            std::cout << "TensorRT I/O: input=\"" << dev.input_name
                      << "\" policy=\"" << dev.policy_name
                      << "\" value=\"" << dev.value_name
                      << "\" score=\"" << dev.score_name
                      << "\" score_sd=\"" << dev.score_sd_name
                      << "\" ownership=\"" << dev.ownership_name << "\"\n";
        }
    }

    impl_ = new Impl(dev);
    auto& I = *impl_;

    I.board_size     = model->board_size;
    I.input_channels = model->input_channels;
    I.max_batch_size = max_batch_size;
    I.input_name      = dev.input_name;
    I.policy_name     = dev.policy_name;
    I.value_name      = dev.value_name;
    I.score_name      = dev.score_name;
    I.score_sd_name   = dev.score_sd_name;
    I.ownership_name  = dev.ownership_name;

    // Create per-thread execution context
    I.exec_ctx = dev.engine->createExecutionContext();
    if (!I.exec_ctx)
        throw std::runtime_error("TensorRT: failed to create execution context");

    // Allocate I/O buffers
    int HW = I.board_size * I.board_size;
    int action_size = HW + 1;

    size_t input_bytes  = (size_t)max_batch_size * I.input_channels * HW * sizeof(float);
    size_t policy_bytes = (size_t)max_batch_size * action_size * sizeof(float);
    size_t value_bytes  = (size_t)max_batch_size * 1 * sizeof(float);

    CUDA_CHECK(cudaMalloc(&I.d_input,  input_bytes));
    CUDA_CHECK(cudaMalloc(&I.d_policy, policy_bytes));
    CUDA_CHECK(cudaMalloc(&I.d_value,  value_bytes));
    CUDA_CHECK(cudaMalloc(&I.d_score,     (size_t)max_batch_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&I.d_score_sd,  (size_t)max_batch_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&I.d_ownership, (size_t)max_batch_size * HW * sizeof(float)));

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

    // Flatten input states and upload
    size_t input_floats = (size_t)N * I.input_channels * HW;
    std::vector<float> flat_input;
    flat_input.reserve(input_floats);
    for (auto& s : states)
        flat_input.insert(flat_input.end(), s.begin(), s.end());

    CUDA_CHECK(cudaMemcpyAsync(I.d_input, flat_input.data(),
        input_floats * sizeof(float), cudaMemcpyHostToDevice, cudaStreamPerThread));

    // Set dynamic input shape for this batch
    nvinfer1::Dims4 input_dims(N, I.input_channels, H, W);
    if (!I.exec_ctx->setInputShape(I.input_name.c_str(), input_dims))
        throw std::runtime_error("TensorRT: setInputShape failed");

    // Bind I/O tensor addresses
    if (!I.exec_ctx->setTensorAddress(I.input_name.c_str(),  I.d_input))
        throw std::runtime_error("TensorRT: setTensorAddress(input) failed");
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

    // Read back results
    std::vector<float> pol_flat((size_t)N * action_size);
    std::vector<float> val_flat((size_t)N);
    std::vector<float> scr_flat((size_t)N);
    std::vector<float> scr_sd_flat((size_t)N);
    std::vector<float> own_flat((size_t)N * HW);

    CUDA_CHECK(cudaMemcpyAsync(pol_flat.data(), I.d_policy,
        pol_flat.size() * sizeof(float), cudaMemcpyDeviceToHost, cudaStreamPerThread));
    CUDA_CHECK(cudaMemcpyAsync(val_flat.data(), I.d_value,
        val_flat.size() * sizeof(float), cudaMemcpyDeviceToHost, cudaStreamPerThread));

    if (!I.score_name.empty())
        CUDA_CHECK(cudaMemcpyAsync(scr_flat.data(), I.d_score,
            scr_flat.size() * sizeof(float), cudaMemcpyDeviceToHost, cudaStreamPerThread));
    if (!I.score_sd_name.empty())
        CUDA_CHECK(cudaMemcpyAsync(scr_sd_flat.data(), I.d_score_sd,
            scr_sd_flat.size() * sizeof(float), cudaMemcpyDeviceToHost, cudaStreamPerThread));
    if (!I.ownership_name.empty())
        CUDA_CHECK(cudaMemcpyAsync(own_flat.data(), I.d_ownership,
            own_flat.size() * sizeof(float), cudaMemcpyDeviceToHost, cudaStreamPerThread));

    CUDA_CHECK(cudaStreamSynchronize(cudaStreamPerThread));

    // Pack results — TRT outputs are row-major [N, action_size] and [N, 1]
    std::vector<TensorRTComputeHandle::Result> results(N);
    for (int n = 0; n < N; n++) {
        results[n].policy.assign(pol_flat.begin() + n * action_size,
                                 pol_flat.begin() + (n + 1) * action_size);
        results[n].value    = val_flat[n];
        results[n].score    = scr_flat[n];
        results[n].score_sd = scr_sd_flat[n];
        results[n].ownership.assign(own_flat.begin() + n * HW,
                                    own_flat.begin() + (n + 1) * HW);
    }
    return results;
}

}  // namespace minigo
#endif  // MINIGO_HAS_TENSORRT
