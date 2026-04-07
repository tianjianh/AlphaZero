#ifdef MINIGO_HAS_CUDA

#include "cuda_compute.h"
#include "loaded_model.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include <cudnn.h>
#include <cmath>
#include <cstring>
#include <iostream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <algorithm>
#include <vector>

namespace minigo {

// ================================================================
// Error checking
// ================================================================
#define CUDA_CHECK(expr) do {                                                  \
    cudaError_t _e = (expr);                                                   \
    if (_e != cudaSuccess) {                                                   \
        std::ostringstream _os;                                                \
        _os << "CUDA: " << cudaGetErrorString(_e)                              \
            << " at " << __FILE__ << ":" << __LINE__;                          \
        throw std::runtime_error(_os.str());                                   \
    }                                                                          \
} while (0)

#define CUBLAS_CHECK(expr) do {                                                \
    cublasStatus_t _s = (expr);                                                \
    if (_s != CUBLAS_STATUS_SUCCESS) {                                         \
        std::ostringstream _os;                                                \
        _os << "cuBLAS error " << (int)_s                                      \
            << " at " << __FILE__ << ":" << __LINE__;                          \
        throw std::runtime_error(_os.str());                                   \
    }                                                                          \
} while (0)

#define CUDNN_CHECK(expr) do {                                                 \
    cudnnStatus_t _s = (expr);                                                 \
    if (_s != CUDNN_STATUS_SUCCESS) {                                          \
        std::ostringstream _os;                                                \
        _os << "cuDNN: " << cudnnGetErrorString(_s)                            \
            << " at " << __FILE__ << ":" << __LINE__;                          \
        throw std::runtime_error(_os.str());                                   \
    }                                                                          \
} while (0)

// ================================================================
// Device state
// ================================================================
struct CUDADeviceState {
    int            device_id = -1;
    cudaStream_t   stream    = nullptr;
    cublasHandle_t cublas    = nullptr;
    cudnnHandle_t  cudnn     = nullptr;
    bool           use_fp16  = false;
    int            sm_major  = 0;
};

// ================================================================
// Small utility kernels
// ================================================================

// FP32→FP16 conversion (for weight upload)
__global__ void convert_fp32_to_fp16(const float* in, half* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = __float2half(in[i]);
}

// Add per-channel bias + ReLU to NCHW tensor (works for FP16 or FP32)
// act: 0=none, 1=relu
template<typename T>
__global__ void bias_act_nchw(T* data, const float* bias, int C, int HW, int total, int act) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    int c = (i / HW) % C;
    float v;
    if constexpr (std::is_same_v<T, half>)
        v = __half2float(data[i]) + bias[c];
    else
        v = data[i] + bias[c];
    if (act == 1 && v < 0.0f) v = 0.0f;
    if constexpr (std::is_same_v<T, half>)
        data[i] = __float2half(v);
    else
        data[i] = v;
}

// Add per-channel bias + residual + ReLU to NCHW tensor
template<typename T>
__global__ void bias_res_relu_nchw(T* data, const float* bias, const T* residual,
                                    int C, int HW, int total) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    int c = (i / HW) % C;
    float v, r;
    if constexpr (std::is_same_v<T, half>) {
        v = __half2float(data[i]) + bias[c] + __half2float(residual[i]);
    } else {
        v = data[i] + bias[c] + residual[i];
    }
    if (v < 0.0f) v = 0.0f;
    if constexpr (std::is_same_v<T, half>)
        data[i] = __float2half(v);
    else
        data[i] = v;
}

// Add bias + activation to FC output [N, M] (row-major)
// act: 0=none, 1=relu, 2=tanh
template<typename T>
__global__ void bias_act_fc(T* data, const float* bias, int M, int total, int act) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    int m = i % M;
    float v;
    if constexpr (std::is_same_v<T, half>)
        v = __half2float(data[i]) + bias[m];
    else
        v = data[i] + bias[m];
    if (act == 1 && v < 0.0f) v = 0.0f;
    else if (act == 2) v = tanhf(v);
    if constexpr (std::is_same_v<T, half>)
        data[i] = __float2half(v);
    else
        data[i] = v;
}

// Policy softmax: read T input [N, M], write float output [N, M]
template<typename T>
__global__ void softmax_kernel(const T* input, float* output, int N, int M) {
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= N) return;
    float mx = -1e30f;
    for (int i = 0; i < M; i++) {
        float v;
        if constexpr (std::is_same_v<T, half>)
            v = __half2float(input[n * M + i]);
        else
            v = input[n * M + i];
        mx = fmaxf(mx, v);
    }
    float sum = 0.0f;
    for (int i = 0; i < M; i++) {
        float v;
        if constexpr (std::is_same_v<T, half>)
            v = __half2float(input[n * M + i]);
        else
            v = input[n * M + i];
        float e = expf(v - mx);
        output[n * M + i] = e;
        sum += e;
    }
    float inv = 1.0f / sum;
    for (int i = 0; i < M; i++)
        output[n * M + i] *= inv;
}

// Value tanh: read T input [N, 1], write float output [N]
template<typename T>
__global__ void tanh_out_kernel(const T* input, float* output, const float* bias, int N) {
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= N) return;
    float v;
    if constexpr (std::is_same_v<T, half>)
        v = __half2float(input[n]);
    else
        v = input[n];
    output[n] = tanhf(v + bias[0]);
}

// Score: softmax over bins → expected value
template<typename T>
__global__ void score_softmax_ev_kernel(const T* input, float* output,
                                         const float* bin_values, int N, int num_bins) {
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= N) return;
    const T* row = input + n * num_bins;
    float mx = -1e30f;
    for (int i = 0; i < num_bins; i++) {
        float v;
        if constexpr (std::is_same_v<T, half>)
            v = __half2float(row[i]);
        else
            v = row[i];
        mx = fmaxf(mx, v);
    }
    float sum = 0.0f, ev = 0.0f;
    for (int i = 0; i < num_bins; i++) {
        float v;
        if constexpr (std::is_same_v<T, half>)
            v = __half2float(row[i]);
        else
            v = row[i];
        float e = expf(v - mx);
        sum += e;
        ev += e * bin_values[i];
    }
    output[n] = ev / sum;
}

// ================================================================
// CUDAComputeContext
// ================================================================

struct CUDAComputeContext::Impl {
    std::map<int, CUDADeviceState> devices;
};

static void init_device(CUDADeviceState& ds, int device_id) {
    int count = 0;
    CUDA_CHECK(cudaGetDeviceCount(&count));
    if (device_id < 0 || device_id >= count)
        throw std::runtime_error("CUDA device " + std::to_string(device_id) + " out of range");

    ds.device_id = device_id;
    CUDA_CHECK(cudaSetDevice(device_id));

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device_id));
    ds.sm_major = prop.major;
    ds.use_fp16 = (prop.major > 6 || (prop.major == 6 && prop.minor >= 0));

    CUDA_CHECK(cudaStreamCreate(&ds.stream));
    CUBLAS_CHECK(cublasCreate(&ds.cublas));
    CUBLAS_CHECK(cublasSetStream(ds.cublas, ds.stream));
    CUBLAS_CHECK(cublasSetMathMode(ds.cublas, CUBLAS_TENSOR_OP_MATH));
    CUDNN_CHECK(cudnnCreate(&ds.cudnn));
    CUDNN_CHECK(cudnnSetStream(ds.cudnn, ds.stream));

    const char* prec = ds.use_fp16 ? "FP16" : "FP32";
    const char* tc = (prop.major >= 7) ? " + Tensor Cores" : "";
    std::cout << "CUDA device " << device_id << ": " << prop.name
              << " (SM " << prop.major << "." << prop.minor
              << ", " << prec << tc << ")\n";
}

CUDAComputeContext::CUDAComputeContext(const std::vector<int>& device_ids) {
    impl_ = new Impl();
    for (int id : device_ids) {
        if (impl_->devices.count(id)) continue;
        init_device(impl_->devices[id], id);
    }
}

CUDAComputeContext::~CUDAComputeContext() {
    if (impl_) {
        for (auto& [id, ds] : impl_->devices) {
            cudaSetDevice(ds.device_id);
            if (ds.cublas) cublasDestroy(ds.cublas);
            if (ds.cudnn) cudnnDestroy(ds.cudnn);
            if (ds.stream) cudaStreamDestroy(ds.stream);
        }
        delete impl_;
    }
}

CUDADeviceState& CUDAComputeContext::device_state(int gpu_id) {
    auto it = impl_->devices.find(gpu_id);
    if (it == impl_->devices.end())
        throw std::runtime_error("CUDA device " + std::to_string(gpu_id) + " not initialized");
    return it->second;
}

std::unique_ptr<ComputeHandle>
CUDAComputeContext::create_handle(const LoadedModel* model, int gpu_id, int max_batch) {
    return std::make_unique<CUDAComputeHandle>(device_state(gpu_id), model, max_batch);
}

// ================================================================
// CUDAComputeHandle::Impl
// ================================================================

struct CUDAComputeHandle::Impl {
    CUDADeviceState& dev;
    bool use_fp16;
    int board_size, input_channels, num_filters, num_res_blocks;
    int num_score_bins;
    cudnnDataType_t dt;      // CUDNN_DATA_HALF or CUDNN_DATA_FLOAT
    cudaDataType_t  cuda_dt; // CUDA_R_16F or CUDA_R_32F
    size_t elem;             // sizeof(half) or sizeof(float)

    // Per-conv-layer GPU data
    struct ConvGPU {
        void*  weight   = nullptr; // pre-scaled by bn_scale, FP16 or FP32
        float* bias     = nullptr; // bn_bias, always FP32
        cudnnFilterDescriptor_t filter_desc = nullptr;
        cudnnConvolutionDescriptor_t conv_desc = nullptr;
        cudnnTensorDescriptor_t bias_desc = nullptr;  // [1, c_out, 1, 1]
        int c_out, c_in, k;
    };
    struct FCGPU {
        void*  weight = nullptr;
        float* bias   = nullptr;
        int out_features, in_features;
    };

    ConvGPU              input_conv;
    std::vector<ConvGPU> res_conv1, res_conv2;
    ConvGPU              policy_conv, value_conv, score_conv;
    FCGPU                policy_fc, value_fc1, value_fc2;
    FCGPU                score_fc1, score_fc2;

    // Score bin values on GPU [-board_area, ..., +board_area]
    float* bin_values_gpu = nullptr;

    // Shared activation descriptor + cached tensor descriptors
    cudnnActivationDescriptor_t relu_act = nullptr;
    cudnnTensorDescriptor_t desc_input = nullptr; // [N, C_in, H, W]
    cudnnTensorDescriptor_t desc_trunk = nullptr; // [N, F, H, W]
    cudnnTensorDescriptor_t desc_pol   = nullptr; // [N, 2, H, W]
    cudnnTensorDescriptor_t desc_vs    = nullptr; // [N, 1, H, W]
    int desc_batch = 0;

    // Workspace buffers (void* — actual type depends on use_fp16)
    float* buf_input_f32 = nullptr; // pre-allocated FP32 staging
    void*  buf_input  = nullptr;  // [N, C_in, H, W] compute type
    void*  buf_main   = nullptr;  // [N, F, H, W]
    void*  buf_temp   = nullptr;  // [N, F, H, W]
    void*  buf_head   = nullptr;  // [N, max_head_channels, H, W]
    void*  buf_fc     = nullptr;  // [N, max_fc_features]
    void*  buf_fc2    = nullptr;  // [N, max_fc2_features]

    // FP32 output buffers
    float* buf_pol_out = nullptr; // [N, action_size]
    float* buf_val_out = nullptr; // [N]
    float* buf_scr_out = nullptr; // [N]

    // cuDNN workspace
    void*  cudnn_ws    = nullptr;
    size_t cudnn_ws_sz = 0;

    int alloc_batch = 0;

    explicit Impl(CUDADeviceState& d) : dev(d), use_fp16(d.use_fp16) {
        if (use_fp16) { dt = CUDNN_DATA_HALF; cuda_dt = CUDA_R_16F; elem = sizeof(half); }
        else          { dt = CUDNN_DATA_FLOAT; cuda_dt = CUDA_R_32F; elem = sizeof(float); }
    }

    // ── Weight upload ────────────────────────────────────────
    // Pre-scale conv weights by bn_scale, upload as FP16 or FP32
    void upload_conv(ConvGPU& g, const ConvBNWeights& src) {
        g.c_out = src.c_out; g.c_in = src.c_in; g.k = src.k;
        int n = (int)src.weight.size();

        // Pre-scale: weight[co][...] *= bn_scale[co]
        std::vector<float> scaled(n);
        int per_filter = n / src.c_out;
        for (int co = 0; co < src.c_out; co++)
            for (int j = 0; j < per_filter; j++)
                scaled[co * per_filter + j] = src.weight[co * per_filter + j] * src.bn_scale[co];

        if (use_fp16) {
            float* tmp; CUDA_CHECK(cudaMalloc(&tmp, n * sizeof(float)));
            CUDA_CHECK(cudaMemcpy(tmp, scaled.data(), n * sizeof(float), cudaMemcpyHostToDevice));
            half* h; CUDA_CHECK(cudaMalloc(&h, n * sizeof(half)));
            int blk = (n + 255) / 256;
            convert_fp32_to_fp16<<<blk, 256, 0, dev.stream>>>(tmp, h, n);
            CUDA_CHECK(cudaStreamSynchronize(dev.stream));
            cudaFree(tmp);
            g.weight = h;
        } else {
            float* buf; CUDA_CHECK(cudaMalloc(&buf, n * sizeof(float)));
            CUDA_CHECK(cudaMemcpy(buf, scaled.data(), n * sizeof(float), cudaMemcpyHostToDevice));
            g.weight = buf;
        }
        // Bias = bn_bias (always FP32)
        CUDA_CHECK(cudaMalloc(&g.bias, src.c_out * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(g.bias, src.bn_bias.data(), src.c_out * sizeof(float),
                               cudaMemcpyHostToDevice));
        // cuDNN descriptors
        CUDNN_CHECK(cudnnCreateFilterDescriptor(&g.filter_desc));
        CUDNN_CHECK(cudnnSetFilter4dDescriptor(g.filter_desc, dt, CUDNN_TENSOR_NCHW,
                                                g.c_out, g.c_in, g.k, g.k));
        CUDNN_CHECK(cudnnCreateConvolutionDescriptor(&g.conv_desc));
        int pad = (g.k == 3) ? 1 : 0;
        CUDNN_CHECK(cudnnSetConvolution2dDescriptor(g.conv_desc, pad, pad, 1, 1, 1, 1,
                                                     CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));
        if (dev.sm_major >= 7)
            CUDNN_CHECK(cudnnSetConvolutionMathType(g.conv_desc, CUDNN_TENSOR_OP_MATH));
        CUDNN_CHECK(cudnnCreateTensorDescriptor(&g.bias_desc));
        CUDNN_CHECK(cudnnSetTensor4dDescriptor(g.bias_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                                                1, g.c_out, 1, 1));
    }

    void upload_fc(FCGPU& g, const FCWeights& src) {
        g.out_features = src.out_features; g.in_features = src.in_features;
        int n = (int)src.weight.size();
        if (use_fp16) {
            float* tmp; CUDA_CHECK(cudaMalloc(&tmp, n * sizeof(float)));
            CUDA_CHECK(cudaMemcpy(tmp, src.weight.data(), n * sizeof(float), cudaMemcpyHostToDevice));
            half* h; CUDA_CHECK(cudaMalloc(&h, n * sizeof(half)));
            int blk = (n + 255) / 256;
            convert_fp32_to_fp16<<<blk, 256, 0, dev.stream>>>(tmp, h, n);
            CUDA_CHECK(cudaStreamSynchronize(dev.stream));
            cudaFree(tmp);
            g.weight = h;
        } else {
            float* buf; CUDA_CHECK(cudaMalloc(&buf, n * sizeof(float)));
            CUDA_CHECK(cudaMemcpy(buf, src.weight.data(), n * sizeof(float), cudaMemcpyHostToDevice));
            g.weight = buf;
        }
        CUDA_CHECK(cudaMalloc(&g.bias, src.out_features * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(g.bias, src.bias.data(), src.out_features * sizeof(float),
                               cudaMemcpyHostToDevice));
    }

    void free_conv(ConvGPU& c) {
        if (c.weight) cudaFree(c.weight);
        if (c.bias) cudaFree(c.bias);
        if (c.filter_desc) cudnnDestroyFilterDescriptor(c.filter_desc);
        if (c.conv_desc) cudnnDestroyConvolutionDescriptor(c.conv_desc);
        if (c.bias_desc) cudnnDestroyTensorDescriptor(c.bias_desc);
        c = {};
    }
    void free_fc(FCGPU& f) {
        if (f.weight) cudaFree(f.weight);
        if (f.bias) cudaFree(f.bias);
        f = {};
    }

    // ── Workspace allocation ─────────────────────────────────
    void ensure_descs(int N) {
        if (N == desc_batch) return;
        int H = board_size, W = board_size;
        auto mk = [&]() { cudnnTensorDescriptor_t d; CUDNN_CHECK(cudnnCreateTensorDescriptor(&d)); return d; };
        if (!desc_input) { desc_input = mk(); desc_trunk = mk(); desc_pol = mk(); desc_vs = mk(); }
        CUDNN_CHECK(cudnnSetTensor4dDescriptor(desc_input, CUDNN_TENSOR_NCHW, dt, N, input_channels, H, W));
        CUDNN_CHECK(cudnnSetTensor4dDescriptor(desc_trunk, CUDNN_TENSOR_NCHW, dt, N, num_filters, H, W));
        CUDNN_CHECK(cudnnSetTensor4dDescriptor(desc_pol,   CUDNN_TENSOR_NCHW, dt, N, 2, H, W));
        CUDNN_CHECK(cudnnSetTensor4dDescriptor(desc_vs,    CUDNN_TENSOR_NCHW, dt, N, 1, H, W));
        desc_batch = N;
    }

    void allocate(int batch) {
        if (batch <= alloc_batch) return;
        free_workspace();
        int H = board_size, W = board_size, HW = H * W;
        int F = num_filters, as = HW + 1;
        int max_head_ch = std::max(2, 1); // policy=2, value/score=1
        int max_fc_feat = std::max({F, (int)value_fc1.out_features,
                                    (int)score_fc1.out_features});
        int max_fc2     = std::max({as, num_score_bins, 1});

        CUDA_CHECK(cudaMalloc(&buf_input_f32, (size_t)batch * input_channels * HW * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&buf_input, (size_t)batch * input_channels * HW * elem));
        CUDA_CHECK(cudaMalloc(&buf_main, (size_t)batch * F * HW * elem));
        CUDA_CHECK(cudaMalloc(&buf_temp, (size_t)batch * F * HW * elem));
        CUDA_CHECK(cudaMalloc(&buf_head, (size_t)batch * max_head_ch * HW * elem));
        CUDA_CHECK(cudaMalloc(&buf_fc,   (size_t)batch * max_fc_feat * elem));
        CUDA_CHECK(cudaMalloc(&buf_fc2,  (size_t)batch * max_fc2 * elem));
        CUDA_CHECK(cudaMalloc(&buf_pol_out, (size_t)batch * as * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&buf_val_out, (size_t)batch * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&buf_scr_out, (size_t)batch * sizeof(float)));

        // Query max cuDNN workspace across all conv configs
        ensure_descs(batch);
        cudnn_ws_sz = 0;
        auto query_ws = [&](ConvGPU& conv, cudnnTensorDescriptor_t in_d, cudnnTensorDescriptor_t out_d) {
            size_t sz = 0;
            cudnnGetConvolutionForwardWorkspaceSize(dev.cudnn, in_d, conv.filter_desc,
                conv.conv_desc, out_d, CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM, &sz);
            cudnn_ws_sz = std::max(cudnn_ws_sz, sz);
        };
        query_ws(input_conv, desc_input, desc_trunk);
        if (!res_conv1.empty()) query_ws(res_conv1[0], desc_trunk, desc_trunk);
        query_ws(policy_conv, desc_trunk, desc_pol);
        query_ws(value_conv, desc_trunk, desc_vs);

        if (cudnn_ws_sz > 0)
            CUDA_CHECK(cudaMalloc(&cudnn_ws, cudnn_ws_sz));
        alloc_batch = batch;
    }

    void free_workspace() {
        auto fr = [](void*& p) { if (p) { cudaFree(p); p = nullptr; } };
        auto frf = [](float*& p) { if (p) { cudaFree(p); p = nullptr; } };
        frf(buf_input_f32); fr(buf_input);
        fr(buf_main); fr(buf_temp); fr(buf_head); fr(buf_fc); fr(buf_fc2);
        frf(buf_pol_out); frf(buf_val_out); frf(buf_scr_out);
        fr(cudnn_ws);
        cudnn_ws_sz = 0;
        alloc_batch = 0;
    }

    // ── cuDNN conv + bias + relu (cached descriptors, no per-call alloc) ──
    void conv_bias_relu(cudnnTensorDescriptor_t in_d, void* input,
                        cudnnTensorDescriptor_t out_d, void* output,
                        ConvGPU& conv) {
        float a1 = 1.0f, b = 0.0f;
        CUDNN_CHECK(cudnnConvolutionForward(dev.cudnn, &a1,
            in_d, input, conv.filter_desc, conv.weight, conv.conv_desc,
            CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM,
            cudnn_ws, cudnn_ws_sz, &b, out_d, output));
        int total = desc_batch * conv.c_out * board_size * board_size;
        int HW = board_size * board_size;
        int blk = (total + 255) / 256;
        if (use_fp16)
            bias_act_nchw<half><<<blk, 256, 0, dev.stream>>>((half*)output, conv.bias, conv.c_out, HW, total, 1);
        else
            bias_act_nchw<float><<<blk, 256, 0, dev.stream>>>((float*)output, conv.bias, conv.c_out, HW, total, 1);
        CUDA_CHECK(cudaGetLastError());
    }

    void conv_bias_res_relu(cudnnTensorDescriptor_t in_d, void* input,
                            cudnnTensorDescriptor_t out_d, void* output,
                            ConvGPU& conv, void* residual) {
        float a1 = 1.0f, b = 0.0f;
        CUDNN_CHECK(cudnnConvolutionForward(dev.cudnn, &a1,
            in_d, input, conv.filter_desc, conv.weight, conv.conv_desc,
            CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM,
            cudnn_ws, cudnn_ws_sz, &b, out_d, output));
        int total = desc_batch * conv.c_out * board_size * board_size;
        int HW = board_size * board_size;
        int blk = (total + 255) / 256;
        if (use_fp16)
            bias_res_relu_nchw<half><<<blk, 256, 0, dev.stream>>>(
                (half*)output, conv.bias, (const half*)residual, conv.c_out, HW, total);
        else
            bias_res_relu_nchw<float><<<blk, 256, 0, dev.stream>>>(
                (float*)output, conv.bias, (const float*)residual, conv.c_out, HW, total);
        CUDA_CHECK(cudaGetLastError());
    }

    // ── cuBLAS FC ────────────────────────────────────────────
    // input [N, K] row-major, weight [M, K] row-major → output [N, M] row-major
    void run_fc(void* input, void* output, FCGPU& fc, int N) {
        int M = fc.out_features, K = fc.in_features;
        float alpha = 1.0f, beta = 0.0f;
        // In column-major: C[M,N] = A^T[M,K] × B[K,N]
        // A = weight stored [M,K] row-major = [K,M] col-major → need transpose
        // B = input stored [N,K] row-major = [K,N] col-major → no transpose
        CUBLAS_CHECK(cublasGemmEx(dev.cublas,
            CUBLAS_OP_T, CUBLAS_OP_N,
            M, N, K,
            &alpha,
            fc.weight, cuda_dt, K,   // A: [K, M] col-major, lda=K
            input, cuda_dt, K,        // B: [K, N] col-major, ldb=K
            &beta,
            output, cuda_dt, M,       // C: [M, N] col-major, ldc=M
            CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    }

    // FC + bias + activation (result stays in compute type)
    void fc_bias_act(void* input, void* output, FCGPU& fc, int N, int act) {
        run_fc(input, output, fc, N);
        int total = N * fc.out_features;
        int blk = (total + 255) / 256;
        if (use_fp16)
            bias_act_fc<half><<<blk, 256, 0, dev.stream>>>((half*)output, fc.bias, fc.out_features, total, act);
        else
            bias_act_fc<float><<<blk, 256, 0, dev.stream>>>((float*)output, fc.bias, fc.out_features, total, act);
        CUDA_CHECK(cudaGetLastError());
    }
};

// ================================================================
// CUDAComputeHandle — public API
// ================================================================

CUDAComputeHandle::CUDAComputeHandle(CUDADeviceState& dev,
                                     const LoadedModel* model,
                                     int max_batch_size) {
    CUDA_CHECK(cudaSetDevice(dev.device_id));
    impl_ = new Impl(dev);
    auto& I = *impl_;

    I.board_size     = model->board_size;
    I.input_channels = model->input_channels;
    I.num_filters    = model->num_filters;
    I.num_res_blocks = model->num_res_blocks;
    I.num_score_bins = model->score_fc2.out_features;

    CUDNN_CHECK(cudnnCreateActivationDescriptor(&I.relu_act));
    CUDNN_CHECK(cudnnSetActivationDescriptor(I.relu_act, CUDNN_ACTIVATION_RELU,
                                              CUDNN_PROPAGATE_NAN, 0.0));

    // Upload weights
    I.upload_conv(I.input_conv, model->input_conv);
    I.res_conv1.resize(I.num_res_blocks);
    I.res_conv2.resize(I.num_res_blocks);
    for (int i = 0; i < I.num_res_blocks; i++) {
        I.upload_conv(I.res_conv1[i], model->res_conv1[i]);
        I.upload_conv(I.res_conv2[i], model->res_conv2[i]);
    }
    I.upload_conv(I.policy_conv, model->policy_conv);
    I.upload_conv(I.value_conv,  model->value_conv);
    I.upload_conv(I.score_conv,  model->score_conv);
    I.upload_fc(I.policy_fc, model->policy_fc);
    I.upload_fc(I.value_fc1, model->value_fc1);
    I.upload_fc(I.value_fc2, model->value_fc2);
    I.upload_fc(I.score_fc1, model->score_fc1);
    I.upload_fc(I.score_fc2, model->score_fc2);

    // Score bin values: [-board_area, ..., +board_area]
    int ba = I.board_size * I.board_size;
    std::vector<float> bv(I.num_score_bins);
    for (int i = 0; i < I.num_score_bins; i++) bv[i] = (float)(i - ba);
    CUDA_CHECK(cudaMalloc(&I.bin_values_gpu, I.num_score_bins * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(I.bin_values_gpu, bv.data(), I.num_score_bins * sizeof(float),
                           cudaMemcpyHostToDevice));

    I.allocate(max_batch_size > 0 ? max_batch_size : 32);

    std::cout << "CUDA handle ready (cuDNN+cuBLAS, "
              << (I.use_fp16 ? "FP16" : "FP32") << "): board=" << I.board_size
              << " filters=" << I.num_filters
              << " blocks="  << I.num_res_blocks
              << " score_bins=" << I.num_score_bins << "\n";
}

CUDAComputeHandle::~CUDAComputeHandle() {
    if (impl_) {
        cudaSetDevice(impl_->dev.device_id);
        impl_->free_workspace();
        impl_->free_conv(impl_->input_conv);
        for (auto& c : impl_->res_conv1) impl_->free_conv(c);
        for (auto& c : impl_->res_conv2) impl_->free_conv(c);
        impl_->free_conv(impl_->policy_conv);
        impl_->free_conv(impl_->value_conv);
        impl_->free_conv(impl_->score_conv);
        impl_->free_fc(impl_->policy_fc);
        impl_->free_fc(impl_->value_fc1);
        impl_->free_fc(impl_->value_fc2);
        impl_->free_fc(impl_->score_fc1);
        impl_->free_fc(impl_->score_fc2);
        if (impl_->bin_values_gpu) cudaFree(impl_->bin_values_gpu);
        if (impl_->relu_act) cudnnDestroyActivationDescriptor(impl_->relu_act);
        if (impl_->desc_input) cudnnDestroyTensorDescriptor(impl_->desc_input);
        if (impl_->desc_trunk) cudnnDestroyTensorDescriptor(impl_->desc_trunk);
        if (impl_->desc_pol) cudnnDestroyTensorDescriptor(impl_->desc_pol);
        if (impl_->desc_vs) cudnnDestroyTensorDescriptor(impl_->desc_vs);
        impl_->res_conv1.clear();
        impl_->res_conv2.clear();
        delete impl_;
    }
}

std::vector<CUDAComputeHandle::Result>
CUDAComputeHandle::predict_batch(const std::vector<std::vector<float>>& states) {
    if (states.empty()) return {};

    auto& I = *impl_;
    CUDA_CHECK(cudaSetDevice(I.dev.device_id));

    int N = (int)states.size();
    int H = I.board_size, W = I.board_size, HW = H * W;
    int F = I.num_filters;
    int action_size = HW + 1;

    I.allocate(N);
    I.ensure_descs(N);

    // Upload input to pre-allocated buffers (no per-batch malloc)
    size_t input_floats = (size_t)N * I.input_channels * HW;
    std::vector<float> flat;
    flat.reserve(input_floats);
    for (auto& s : states)
        flat.insert(flat.end(), s.begin(), s.end());

    CUDA_CHECK(cudaMemcpyAsync(I.buf_input_f32, flat.data(), input_floats * sizeof(float),
                                cudaMemcpyHostToDevice, I.dev.stream));
    if (I.use_fp16) {
        int blk = (int)((input_floats + 255) / 256);
        convert_fp32_to_fp16<<<blk, 256, 0, I.dev.stream>>>(I.buf_input_f32, (half*)I.buf_input, (int)input_floats);
        CUDA_CHECK(cudaGetLastError());
    } else {
        CUDA_CHECK(cudaMemcpyAsync(I.buf_input, I.buf_input_f32, input_floats * sizeof(float),
                                    cudaMemcpyDeviceToDevice, I.dev.stream));
    }

    // ── Input conv + BN + ReLU ─────────────────────────────────
    I.conv_bias_relu(I.desc_input, I.buf_input, I.desc_trunk, I.buf_main, I.input_conv);

    // ── Residual blocks ──────────────────────────────────────
    for (int i = 0; i < I.num_res_blocks; i++) {
        I.conv_bias_relu(I.desc_trunk, I.buf_main, I.desc_trunk, I.buf_temp, I.res_conv1[i]);
        I.conv_bias_res_relu(I.desc_trunk, I.buf_temp, I.desc_trunk, I.buf_main, I.res_conv2[i], I.buf_main);
    }

    // ── Policy head ──────────────────────────────────────────
    I.conv_bias_relu(I.desc_trunk, I.buf_main, I.desc_pol, I.buf_head, I.policy_conv);
    // FC → [N, action_size] in compute type, then softmax → FP32
    I.run_fc(I.buf_head, I.buf_fc2, I.policy_fc, N);
    // Add bias (no activation) then softmax
    {
        int total = N * action_size;
        int blk = (total + 255) / 256;
        if (I.use_fp16)
            bias_act_fc<half><<<blk, 256, 0, I.dev.stream>>>((half*)I.buf_fc2, I.policy_fc.bias, action_size, total, 0);
        else
            bias_act_fc<float><<<blk, 256, 0, I.dev.stream>>>((float*)I.buf_fc2, I.policy_fc.bias, action_size, total, 0);
        CUDA_CHECK(cudaGetLastError());
    }
    {
        int blk = (N + 63) / 64;
        if (I.use_fp16)
            softmax_kernel<half><<<blk, 64, 0, I.dev.stream>>>((const half*)I.buf_fc2, I.buf_pol_out, N, action_size);
        else
            softmax_kernel<float><<<blk, 64, 0, I.dev.stream>>>((const float*)I.buf_fc2, I.buf_pol_out, N, action_size);
        CUDA_CHECK(cudaGetLastError());
    }

    // ── Value head ───────────────────────────────────────────
    I.conv_bias_relu(I.desc_trunk, I.buf_main, I.desc_vs, I.buf_head, I.value_conv);
    // FC1 + bias + ReLU → [N, 64]
    I.fc_bias_act(I.buf_head, I.buf_fc, I.value_fc1, N, 1);
    // FC2 (out=1) + bias + tanh → FP32
    I.run_fc(I.buf_fc, I.buf_fc2, I.value_fc2, N);
    {
        int blk = (N + 63) / 64;
        if (I.use_fp16)
            tanh_out_kernel<half><<<blk, 64, 0, I.dev.stream>>>((const half*)I.buf_fc2, I.buf_val_out, I.value_fc2.bias, N);
        else
            tanh_out_kernel<float><<<blk, 64, 0, I.dev.stream>>>((const float*)I.buf_fc2, I.buf_val_out, I.value_fc2.bias, N);
        CUDA_CHECK(cudaGetLastError());
    }

    // ── Score head ───────────────────────────────────────────
    I.conv_bias_relu(I.desc_trunk, I.buf_main, I.desc_vs, I.buf_head, I.score_conv);
    // FC1 + bias + ReLU → [N, 64]
    I.fc_bias_act(I.buf_head, I.buf_fc, I.score_fc1, N, 1);
    // FC2 → [N, num_bins] + bias (no activation)
    I.fc_bias_act(I.buf_fc, I.buf_fc2, I.score_fc2, N, 0);
    // Softmax → expected value → FP32
    {
        int blk = (N + 63) / 64;
        if (I.use_fp16)
            score_softmax_ev_kernel<half><<<blk, 64, 0, I.dev.stream>>>(
                (const half*)I.buf_fc2, I.buf_scr_out, I.bin_values_gpu, N, I.num_score_bins);
        else
            score_softmax_ev_kernel<float><<<blk, 64, 0, I.dev.stream>>>(
                (const float*)I.buf_fc2, I.buf_scr_out, I.bin_values_gpu, N, I.num_score_bins);
        CUDA_CHECK(cudaGetLastError());
    }

    // ── Read back results ────────────────────────────────────
    std::vector<float> pol_flat((size_t)N * action_size);
    std::vector<float> val_flat((size_t)N);
    std::vector<float> scr_flat((size_t)N);

    CUDA_CHECK(cudaMemcpyAsync(pol_flat.data(), I.buf_pol_out,
        pol_flat.size() * sizeof(float), cudaMemcpyDeviceToHost, I.dev.stream));
    CUDA_CHECK(cudaMemcpyAsync(val_flat.data(), I.buf_val_out,
        val_flat.size() * sizeof(float), cudaMemcpyDeviceToHost, I.dev.stream));
    CUDA_CHECK(cudaMemcpyAsync(scr_flat.data(), I.buf_scr_out,
        scr_flat.size() * sizeof(float), cudaMemcpyDeviceToHost, I.dev.stream));

    CUDA_CHECK(cudaStreamSynchronize(I.dev.stream));

    // Pack results
    std::vector<Result> results(N);
    for (int n = 0; n < N; n++) {
        results[n].policy.resize(action_size);
        for (int a = 0; a < action_size; a++)
            results[n].policy[a] = pol_flat[n * action_size + a];
        results[n].value = val_flat[n];
        results[n].score = scr_flat[n];
    }
    return results;
}

}  // namespace minigo
#endif  // MINIGO_HAS_CUDA
