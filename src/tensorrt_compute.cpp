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
    cudaStream_t stream    = nullptr;

    // Engine is built lazily on first create_handle() for this device.
    // Protected by build_mutex so only the first thread builds it.
    std::mutex                              build_mutex;
    nvinfer1::ICudaEngine*                  engine  = nullptr;
    nvinfer1::IRuntime*                     runtime = nullptr;

    // Cached tensor names + sizes (populated after engine build)
    std::string input_name;
    std::string policy_name;
    std::string value_name;
    std::string score_name;
};

// ================================================================
// Engine cache helpers
// ================================================================

static std::string make_cache_path(const std::string& model_path,
                                   const std::string& gpu_name,
                                   int max_batch_size) {
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

    return cache_dir + "/" + base + ".trt_" + safe_name + "_b" +
           std::to_string(max_batch_size) + ".engine";
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
// ================================================================

static nvinfer1::ICudaEngine* build_or_load_engine(
        TRTDeviceState& dev,
        const LoadedModel* model,
        int max_batch_size) {

    CUDA_CHECK(cudaSetDevice(dev.device_id));

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, dev.device_id));
    std::string gpu_name(prop.name);

    std::string cache_path = make_cache_path(model->model_path, gpu_name, max_batch_size);

    // Try loading cached engine
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
    config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, 256ULL << 20);  // 256 MiB

    // Enable FP16 if the device supports it
    // (platformHasFastFp16/kFP16 deprecated in TRT 10.12 in favour of strong
    //  typing, but still functional and the simplest path for our use case)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
    if (builder->platformHasFastFp16())
        config->setFlag(nvinfer1::BuilderFlag::kFP16);
#pragma GCC diagnostic pop

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
    std::cout << "TensorRT device " << device_id << ": " << prop.name
              << " (SM " << prop.major << "." << prop.minor << ")\n";

    CUDA_CHECK(cudaStreamCreate(&ds.stream));

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
            if (ds.engine) delete ds.engine;
            if (ds.runtime) delete ds.runtime;
            if (ds.stream) {
                cudaSetDevice(ds.device_id);
                cudaStreamDestroy(ds.stream);
            }
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
    float* d_input   = nullptr;   // [max_batch, C, H, W]
    float* d_policy  = nullptr;   // [max_batch, action_size]
    float* d_value   = nullptr;   // [max_batch, 1]
    float* d_score   = nullptr;   // [max_batch, 1]  (optional)

    // Tensor names from engine
    std::string input_name;
    std::string policy_name;
    std::string value_name;
    std::string score_name;

    Impl(TRTDeviceState& d) : dev(d) {}

    ~Impl() {
        cudaSetDevice(dev.device_id);
        if (exec_ctx) delete exec_ctx;
        if (d_input)  cudaFree(d_input);
        if (d_policy) cudaFree(d_policy);
        if (d_value)  cudaFree(d_value);
        if (d_score)  cudaFree(d_score);
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

            // Discover I/O tensor names
            int nb = dev.engine->getNbIOTensors();
            for (int i = 0; i < nb; i++) {
                const char* name = dev.engine->getIOTensorName(i);
                auto mode = dev.engine->getTensorIOMode(name);
                if (mode == nvinfer1::TensorIOMode::kINPUT) {
                    dev.input_name = name;
                } else {
                    // Check for explicitly-named score head first
                    std::string sname(name);
                    if (sname == "score") {
                        dev.score_name = name;
                    } else {
                        // Identify policy vs value by shape:
                        // policy is [N, action_size], value is [N, 1]
                        auto dims = dev.engine->getTensorShape(name);
                        // Last dim > 1 → policy, else value
                        if (dims.nbDims >= 2 && dims.d[dims.nbDims - 1] > 1)
                            dev.policy_name = name;
                        else
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
                      << "\" score=\"" << dev.score_name << "\"\n";
        }
    }

    impl_ = new Impl(dev);
    auto& I = *impl_;

    I.board_size     = model->board_size;
    I.input_channels = model->input_channels;
    I.max_batch_size = max_batch_size;
    I.input_name     = dev.input_name;
    I.policy_name    = dev.policy_name;
    I.value_name     = dev.value_name;
    I.score_name     = dev.score_name;

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
    CUDA_CHECK(cudaMalloc(&I.d_score, (size_t)max_batch_size * sizeof(float)));

    std::cout << "TensorRT handle ready: board=" << I.board_size;
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
        input_floats * sizeof(float), cudaMemcpyHostToDevice, I.dev.stream));

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
    if (!I.exec_ctx->setTensorAddress(I.score_name.c_str(), I.d_score))
        throw std::runtime_error("TensorRT: setTensorAddress(score) failed");

    // Run inference
    if (!I.exec_ctx->enqueueV3(I.dev.stream))
        throw std::runtime_error("TensorRT: enqueueV3 failed");

    // Read back results
    std::vector<float> pol_flat((size_t)N * action_size);
    std::vector<float> val_flat((size_t)N);

    CUDA_CHECK(cudaMemcpyAsync(pol_flat.data(), I.d_policy,
        pol_flat.size() * sizeof(float), cudaMemcpyDeviceToHost, I.dev.stream));
    CUDA_CHECK(cudaMemcpyAsync(val_flat.data(), I.d_value,
        val_flat.size() * sizeof(float), cudaMemcpyDeviceToHost, I.dev.stream));

    std::vector<float> scr_flat((size_t)N);
    CUDA_CHECK(cudaMemcpyAsync(scr_flat.data(), I.d_score,
        scr_flat.size() * sizeof(float), cudaMemcpyDeviceToHost, I.dev.stream));

    CUDA_CHECK(cudaStreamSynchronize(I.dev.stream));

    // Pack results — TRT outputs are row-major [N, action_size] and [N, 1]
    std::vector<TensorRTComputeHandle::Result> results(N);
    for (int n = 0; n < N; n++) {
        std::vector<float> pol(pol_flat.begin() + n * action_size,
                               pol_flat.begin() + (n + 1) * action_size);
        results[n].policy = std::move(pol);
        results[n].value  = val_flat[n];
        results[n].score  = scr_flat[n];
    }
    return results;
}

}  // namespace minigo
#endif  // MINIGO_HAS_TENSORRT
