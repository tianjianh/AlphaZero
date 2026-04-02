#ifdef MINIGO_HAS_CUDA

#include "cuda_compute.h"
#include "loaded_model.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <cmath>
#include <cstring>
#include <iostream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <algorithm>

using namespace nvcuda;

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
// Device state (defined here, forward-declared in header)
// ================================================================
struct CUDADeviceState {
    int          device_id = -1;
    cudaStream_t stream    = nullptr;
};

// ================================================================
// Utility kernels
// ================================================================

// Convert FP32 buffer to FP16 on GPU (used for one-time weight upload)
__global__ void convert_fp32_to_fp16(const float* input, half* output, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) output[i] = __float2half(input[i]);
}

// Fused transpose + FP32→FP16: NCHW [N,C,HW] → channel-major FP16 [C, N*HW]
__global__ void transpose_nchw_fp32_to_fp16(
    const float* input, half* output,
    int N, int C, int HW
) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * C * HW;
    if (gid >= total) return;
    int n  = gid / (C * HW);
    int c  = (gid / HW) % C;
    int hw = gid % HW;
    output[c * (N * HW) + n * HW + hw] = __float2half(input[gid]);
}

// ================================================================
// Conv 3×3: WMMA Tensor Core kernel + fused BN + optional ReLU
//
// FP16 weights & activations, FP32 accumulator via Tensor Cores.
//
// A (weight):  half [C_out, K], K = C_in * 9
// B (virtual): im2col of input half [C_in, NHW] — computed on-the-fly
// C (output):  half [C_out, NHW]
//
// Block: 256 threads = 8 warps
// Warp layout: 2 (M) × 4 (N)
// Output tile: TILE_M × TILE_N = 32 × 64
// WMMA tile: 16 × 16 × 16
//
// mode: 0 = plain, 1 = BN + optional ReLU, 2 = BN + residual-add + ReLU
// Grid: { ceil(NHW/TILE_N), ceil(C_out/TILE_M) }
// ================================================================

#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16
#define WARPS_M 2
#define WARPS_N 4
#define TILE_M (WARPS_M * WMMA_M)  // 32
#define TILE_N (WARPS_N * WMMA_N)  // 64
#define SMEM_A_STRIDE (WMMA_K + 8) // 24, avoids bank conflicts
#define SMEM_B_STRIDE (TILE_N + 8) // 72

__global__ void conv3x3_wmma_bn(
    const half* __restrict__ A,
    const half* __restrict__ input,
    half*       __restrict__ output,
    const float* __restrict__ bn_scale,
    const float* __restrict__ bn_bias,
    const half* __restrict__ residual,
    int C_out, int NHW, int K,
    int H, int W,
    int mode, int do_relu
) {
    __shared__ half smem_a[TILE_M][SMEM_A_STRIDE];  // [32][24]
    __shared__ half smem_b[WMMA_K][SMEM_B_STRIDE];  // [16][72]

    int warp_id = threadIdx.x / 32;
    int warp_m  = warp_id / WARPS_N;  // 0 or 1
    int warp_n  = warp_id % WARPS_N;  // 0..3

    int gm_base = blockIdx.y * TILE_M;
    int gn_base = blockIdx.x * TILE_N;

    if (gm_base >= C_out || gn_base >= NHW) return;

    int HW = H * W;

    // Accumulator fragment (FP32)
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc;
    wmma::fill_fragment(acc, 0.0f);

    int numTiles = (K + WMMA_K - 1) / WMMA_K;

    for (int t = 0; t < numTiles; t++) {
        int k_off = t * WMMA_K;

        // Cooperatively load A tile [TILE_M, WMMA_K] from weight[C_out, K]
        // 256 threads, 32 * 16 = 512 elements → 2 per thread
        for (int idx = threadIdx.x; idx < TILE_M * WMMA_K; idx += 256) {
            int row = idx / WMMA_K;
            int col = idx % WMMA_K;
            int gm  = gm_base + row;
            int gk  = k_off + col;
            smem_a[row][col] = (gm < C_out && gk < K) ? A[gm * K + gk] : __float2half(0.0f);
        }

        // Cooperatively load B tile [WMMA_K, TILE_N] via implicit im2col
        // 256 threads, 16 * 64 = 1024 elements → 4 per thread
        for (int idx = threadIdx.x; idx < WMMA_K * TILE_N; idx += 256) {
            int kr  = idx / TILE_N;   // row in tile (0..15)
            int nc  = idx % TILE_N;   // col in tile (0..63)
            int k_row = k_off + kr;
            int gn    = gn_base + nc;

            half val = __float2half(0.0f);
            if (k_row < K && gn < NHW) {
                int c_in = k_row / 9;
                int rem  = k_row % 9;
                int kh   = rem / 3;
                int kw   = rem % 3;
                int n    = gn / HW;
                int hw   = gn % HW;
                int ih   = hw / W + kh - 1;  // same-padding: pad=1
                int iw   = hw % W + kw - 1;
                if (ih >= 0 && ih < H && iw >= 0 && iw < W)
                    val = input[c_in * NHW + n * HW + ih * W + iw];
            }
            smem_b[kr][nc] = val;
        }

        __syncthreads();

        // Each warp loads its fragment and does MMA
        wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> frag_a;
        wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> frag_b;

        wmma::load_matrix_sync(frag_a, &smem_a[warp_m * WMMA_M][0], SMEM_A_STRIDE);
        wmma::load_matrix_sync(frag_b, &smem_b[0][warp_n * WMMA_N], SMEM_B_STRIDE);

        wmma::mma_sync(acc, frag_a, frag_b, acc);

        __syncthreads();
    }

    // Store accumulator to shared memory (reuse smem_b region as float)
    // Need 32 × 64 floats = 8192 bytes; smem_a + smem_b = 1536 + 2304 = 3840 bytes (half)
    // Reinterpret shared memory as float for output staging
    extern __shared__ float smem_out[];  // [TILE_M][TILE_N + 8], declared via launch config
    const int OUT_STRIDE = TILE_N + 8;   // 72

    wmma::store_matrix_sync(&smem_out[(warp_m * WMMA_M) * OUT_STRIDE + warp_n * WMMA_N],
                            acc, OUT_STRIDE, wmma::mem_row_major);

    __syncthreads();

    // Apply BN + optional residual + ReLU, write FP16 to global
    for (int idx = threadIdx.x; idx < TILE_M * TILE_N; idx += 256) {
        int row = idx / TILE_N;
        int col = idx % TILE_N;
        int gm  = gm_base + row;
        int gn  = gn_base + col;
        if (gm >= C_out || gn >= NHW) continue;

        float v = smem_out[row * OUT_STRIDE + col];
        if (mode >= 1) {
            v = bn_scale[gm] * v + bn_bias[gm];
        }
        if (mode == 2) v += __half2float(residual[gm * NHW + gn]);
        if (do_relu && v < 0.0f) v = 0.0f;
        output[gm * NHW + gn] = __float2half(v);
    }
}

// ================================================================
// Fused 1×1 conv + BN + ReLU + reshape (FP16)
//
// input  half [C_in, N*HW]   channel-major
// output half [C_out*HW, N]   FC-ready layout
// ================================================================
__global__ void conv1x1_bn_relu_reshape_fp16(
    const half* input, half* output,
    const half* weight, const float* bn_scale, const float* bn_bias,
    int C_in, int C_out, int N, int HW
) {
    int n  = blockIdx.x * blockDim.x + threadIdx.x;
    int hw = blockIdx.y * blockDim.y + threadIdx.y;
    if (n >= N || hw >= HW) return;

    int NHW = N * HW;
    for (int co = 0; co < C_out; co++) {
        float acc = 0.0f;
        for (int ci = 0; ci < C_in; ci++)
            acc += __half2float(weight[co * C_in + ci]) *
                   __half2float(input[ci * NHW + n * HW + hw]);
        float v = bn_scale[co] * acc + bn_bias[co];
        if (v < 0.0f) v = 0.0f;
        output[(co * HW + hw) * N + n] = __float2half(v);
    }
}

// ================================================================
// Fused FC + bias + optional ReLU (FP16 in/out)
// ================================================================
__global__ void fc_bias_relu_fp16(
    const half* weight, const half* input, half* output,
    const float* bias, int M, int N, int K, int do_relu
) {
    int m = blockIdx.x * blockDim.x + threadIdx.x;
    int n = blockIdx.y * blockDim.y + threadIdx.y;
    if (m >= M || n >= N) return;

    float acc = 0.0f;
    for (int k = 0; k < K; k++)
        acc += __half2float(weight[m * K + k]) * __half2float(input[k * N + n]);
    acc += bias[m];
    if (do_relu && acc < 0.0f) acc = 0.0f;
    output[m * N + n] = __float2half(acc);
}

// ================================================================
// Fused FC + bias + softmax (FP16 input → FP32 output)
// ================================================================
__global__ void fc_bias_softmax_fp16_to_fp32(
    const half* weight, const half* input, float* output,
    const float* bias, int M, int N, int K
) {
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= N) return;

    float mx = -1e30f;
    for (int m = 0; m < M; m++) {
        float acc = bias[m];
        for (int k = 0; k < K; k++)
            acc += __half2float(weight[m * K + k]) * __half2float(input[k * N + n]);
        output[m * N + n] = acc;
        mx = fmaxf(mx, acc);
    }

    float sum = 0.0f;
    for (int m = 0; m < M; m++) {
        float e = expf(output[m * N + n] - mx);
        output[m * N + n] = e;
        sum += e;
    }
    float inv_sum = 1.0f / sum;
    for (int m = 0; m < M; m++)
        output[m * N + n] *= inv_sum;
}

// ================================================================
// Fused FC + bias + tanh (FP16 input → FP32 output)
// ================================================================
__global__ void fc_bias_tanh_fp16_to_fp32(
    const half* weight, const half* input, float* output,
    const float* bias, int N, int K
) {
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= N) return;

    float acc = bias[0];
    for (int k = 0; k < K; k++)
        acc += __half2float(weight[k]) * __half2float(input[k * N + n]);
    output[n] = tanhf(acc);
}

// ================================================================
// CUDAComputeContext
// ================================================================

struct CUDAComputeContext::Impl {
    std::map<int, CUDADeviceState> devices;
};

static void init_device(CUDADeviceState& ds, int device_id) {
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
    std::cout << "CUDA device " << device_id << ": " << prop.name
              << " (SM " << prop.major << "." << prop.minor << ")\n";

    CUDA_CHECK(cudaStreamCreate(&ds.stream));
}

CUDAComputeContext::CUDAComputeContext(const std::vector<int>& device_ids) {
    impl_ = new Impl();
    for (int id : device_ids) {
        if (impl_->devices.count(id)) continue;
        auto& ds = impl_->devices[id];
        init_device(ds, id);
    }
}

CUDAComputeContext::~CUDAComputeContext() {
    if (impl_) {
        for (auto& [id, ds] : impl_->devices) {
            if (ds.stream) {
                cudaSetDevice(ds.device_id);
                cudaStreamDestroy(ds.stream);
            }
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
CUDAComputeContext::create_handle(const LoadedModel* model, int gpu_id, int max_batch_size) {
    return std::make_unique<CUDAComputeHandle>(device_state(gpu_id), model, max_batch_size);
}

// ================================================================
// CUDAComputeHandle::Impl — FP16 weights, WMMA tensor-core compute
// ================================================================

struct CUDAComputeHandle::Impl {
    CUDADeviceState& dev;

    struct ConvBNGPU {
        half* weight = nullptr;      // FP16 (bulk data, bandwidth-sensitive)
        float* bn_scale = nullptr;   // FP32 (small per-channel, avoids conversions)
        float* bn_bias = nullptr;    // FP32
        int c_out = 0, c_in = 0, k = 0;
    };
    struct FCGPU {
        half* weight = nullptr;      // FP16
        float* bias = nullptr;       // FP32
        int out_features = 0, in_features = 0;
    };

    ConvBNGPU              input_conv_gpu;
    std::vector<ConvBNGPU> res_conv1_gpu, res_conv2_gpu;
    ConvBNGPU              policy_conv_gpu, value_conv_gpu;
    FCGPU                  policy_fc_gpu, value_fc1_gpu, value_fc2_gpu;

    // Input buffer (FP32 from host)
    float* buf_flat_in = nullptr;

    // Internal workspace (all FP16)
    half* buf_input    = nullptr;   // [C_in, N*HW] after transpose
    half* buf_main     = nullptr;   // [F, N*HW]
    half* buf_temp     = nullptr;   // [F, N*HW]
    half* buf_skip     = nullptr;   // [F, N*HW]
    half* buf_pol_out  = nullptr;   // [2*HW, N] after 1×1 conv reshape
    half* buf_val_h1   = nullptr;   // [HW, N]
    half* buf_val_feat = nullptr;   // [F, N]

    // Output buffers (FP32 for softmax/tanh numerical stability)
    float* buf_pol_feat = nullptr;  // [action_size, N]
    float* buf_val_out  = nullptr;  // [N]

    int alloc_batch = 0;
    int board_size, input_channels, num_filters, num_res_blocks;

    explicit Impl(CUDADeviceState& d) : dev(d) {}

    // Upload FP32 vector → FP16 GPU buffer (convert on GPU)
    half* upload_half(const std::vector<float>& data) {
        size_t n = data.size();
        float* tmp = nullptr;
        CUDA_CHECK(cudaMalloc(&tmp, n * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(tmp, data.data(), n * sizeof(float), cudaMemcpyHostToDevice));
        half* buf = nullptr;
        CUDA_CHECK(cudaMalloc(&buf, n * sizeof(half)));
        int threads = 256;
        int blocks = (int)((n + threads - 1) / threads);
        convert_fp32_to_fp16<<<blocks, threads, 0, dev.stream>>>(tmp, buf, (int)n);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaStreamSynchronize(dev.stream));
        CUDA_CHECK(cudaFree(tmp));
        return buf;
    }

    // Upload FP32 vector → FP32 GPU buffer (direct copy, no conversion)
    float* upload_float(const std::vector<float>& data) {
        float* buf = nullptr;
        CUDA_CHECK(cudaMalloc(&buf, data.size() * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(buf, data.data(), data.size() * sizeof(float),
                               cudaMemcpyHostToDevice));
        return buf;
    }

    template<typename T>
    void release_buf(T*& buf) { if (buf) { cudaFree(buf); buf = nullptr; } }

    void upload_conv(ConvBNGPU& g, const ConvBNWeights& src) {
        g.c_out = src.c_out;  g.c_in = src.c_in;  g.k = src.k;
        g.weight   = upload_half(src.weight);    // FP16 (bandwidth-sensitive)
        g.bn_scale = upload_float(src.bn_scale); // FP32 (small, avoids conversions)
        g.bn_bias  = upload_float(src.bn_bias);  // FP32
    }
    void upload_fc(FCGPU& g, const FCWeights& src) {
        g.out_features = src.out_features;
        g.in_features  = src.in_features;
        g.weight = upload_half(src.weight);   // FP16
        g.bias   = upload_float(src.bias);    // FP32
    }
    void free_conv(ConvBNGPU& c) {
        release_buf(c.weight); release_buf(c.bn_scale); release_buf(c.bn_bias);
    }
    void free_fc(FCGPU& f) {
        release_buf(f.weight); release_buf(f.bias);
    }

    void free_weights() {
        free_conv(input_conv_gpu);
        for (auto& c : res_conv1_gpu) free_conv(c);
        for (auto& c : res_conv2_gpu) free_conv(c);
        free_conv(policy_conv_gpu);
        free_conv(value_conv_gpu);
        free_fc(policy_fc_gpu);
        free_fc(value_fc1_gpu);
        free_fc(value_fc2_gpu);
        res_conv1_gpu.clear();
        res_conv2_gpu.clear();
    }

    void allocate_workspace(int batch) {
        if (batch <= alloc_batch) return;
        free_workspace();

        int H = board_size, W = board_size;
        int NHW  = batch * H * W;
        int filt = num_filters;
        int inch = input_channels;
        int as   = H * W + 1;

        auto alloc_h = [](size_t floats) -> half* {
            half* b = nullptr;
            CUDA_CHECK(cudaMalloc(&b, floats * sizeof(half)));
            return b;
        };
        auto alloc_f = [](size_t floats) -> float* {
            float* b = nullptr;
            CUDA_CHECK(cudaMalloc(&b, floats * sizeof(float)));
            return b;
        };

        buf_flat_in  = alloc_f((size_t)inch * NHW);         // FP32 from host
        buf_input    = alloc_h((size_t)inch * NHW);          // FP16
        buf_main     = alloc_h((size_t)filt * NHW);
        buf_temp     = alloc_h((size_t)filt * NHW);
        buf_skip     = alloc_h((size_t)filt * NHW);
        buf_pol_out  = alloc_h((size_t)2 * H * W * batch);
        buf_pol_feat = alloc_f((size_t)as * batch);          // FP32 output
        buf_val_h1   = alloc_h((size_t)H * W * batch);
        buf_val_feat = alloc_h((size_t)filt * batch);
        buf_val_out  = alloc_f((size_t)batch);               // FP32 output

        alloc_batch = batch;
    }

    void free_workspace() {
        release_buf(buf_flat_in);  release_buf(buf_input);
        release_buf(buf_main);     release_buf(buf_temp);
        release_buf(buf_skip);     release_buf(buf_pol_out);
        release_buf(buf_pol_feat); release_buf(buf_val_h1);
        release_buf(buf_val_feat); release_buf(buf_val_out);
        alloc_batch = 0;
    }

    // Dynamic shared memory: TILE_M * (TILE_N + 8) * sizeof(float) for WMMA output staging
    static constexpr int WMMA_SMEM_BYTES = TILE_M * (TILE_N + 8) * (int)sizeof(float);

    void run_conv3x3(half* input_buf, half* output_buf,
                     const ConvBNGPU& conv, half* residual_buf,
                     int N, int H, int W, int mode, bool relu) {
        int NHW   = N * H * W;
        int C_out = conv.c_out;
        int K     = conv.c_in * 9;
        int do_relu_i = relu ? 1 : 0;

        half* res_arg = (mode == 2 && residual_buf) ? residual_buf : output_buf;

        dim3 block(256);
        dim3 grid((NHW + TILE_N - 1) / TILE_N, (C_out + TILE_M - 1) / TILE_M);

        conv3x3_wmma_bn<<<grid, block, WMMA_SMEM_BYTES, dev.stream>>>(
            conv.weight, input_buf, output_buf,
            conv.bn_scale, conv.bn_bias, res_arg,
            C_out, NHW, K, H, W, mode, do_relu_i);
        CUDA_CHECK(cudaGetLastError());
    }

    void run_conv1x1_bn_relu_reshape(half* input_buf, half* output_buf,
                                      const ConvBNGPU& conv, int N, int HW) {
        dim3 block(16, 16);
        dim3 grid((N + block.x - 1) / block.x, (HW + block.y - 1) / block.y);

        conv1x1_bn_relu_reshape_fp16<<<grid, block, 0, dev.stream>>>(
            input_buf, output_buf,
            conv.weight, conv.bn_scale, conv.bn_bias,
            conv.c_in, conv.c_out, N, HW);
        CUDA_CHECK(cudaGetLastError());
    }

    void run_fc_bias_relu(half* input_buf, half* output_buf,
                          const FCGPU& fc, int N, bool relu) {
        int M = fc.out_features;
        int K = fc.in_features;
        int do_relu_i = relu ? 1 : 0;

        dim3 block(16, 16);
        dim3 grid((M + block.x - 1) / block.x, (N + block.y - 1) / block.y);

        fc_bias_relu_fp16<<<grid, block, 0, dev.stream>>>(
            fc.weight, input_buf, output_buf, fc.bias,
            M, N, K, do_relu_i);
        CUDA_CHECK(cudaGetLastError());
    }
};

// ================================================================
// CUDAComputeHandle public API
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

    I.upload_conv(I.input_conv_gpu, model->input_conv);
    I.res_conv1_gpu.resize(I.num_res_blocks);
    I.res_conv2_gpu.resize(I.num_res_blocks);
    for (int i = 0; i < I.num_res_blocks; i++) {
        I.upload_conv(I.res_conv1_gpu[i], model->res_conv1[i]);
        I.upload_conv(I.res_conv2_gpu[i], model->res_conv2[i]);
    }
    I.upload_conv(I.policy_conv_gpu, model->policy_conv);
    I.upload_conv(I.value_conv_gpu,  model->value_conv);
    I.upload_fc(I.policy_fc_gpu, model->policy_fc);
    I.upload_fc(I.value_fc1_gpu, model->value_fc1);
    I.upload_fc(I.value_fc2_gpu, model->value_fc2);

    I.allocate_workspace(max_batch_size > 0 ? max_batch_size : 32);

    std::cout << "CUDA handle ready (FP16+WMMA): board=" << I.board_size
              << " filters=" << I.num_filters
              << " blocks="  << I.num_res_blocks << "\n";
}

CUDAComputeHandle::~CUDAComputeHandle() {
    if (impl_) {
        cudaSetDevice(impl_->dev.device_id);
        impl_->free_workspace();
        impl_->free_weights();
        delete impl_;
    }
}

std::vector<std::pair<std::vector<float>, float>>
CUDAComputeHandle::predict_batch(const std::vector<std::vector<float>>& states) {
    if (states.empty()) return {};

    auto& I = *impl_;
    CUDA_CHECK(cudaSetDevice(I.dev.device_id));

    int N = (int)states.size();
    int H = I.board_size, W = I.board_size;
    int HW = H * W;
    int action_size = HW + 1;

    I.allocate_workspace(N);

    // Upload FP32 inputs from host
    size_t input_floats = (size_t)N * I.input_channels * HW;
    std::vector<float> flat_input;
    flat_input.reserve(input_floats);
    for (auto& s : states)
        flat_input.insert(flat_input.end(), s.begin(), s.end());

    CUDA_CHECK(cudaMemcpyAsync(I.buf_flat_in, flat_input.data(),
        input_floats * sizeof(float), cudaMemcpyHostToDevice, I.dev.stream));

    // Fused transpose + FP32→FP16: [N, C, HW] → [C, N*HW] as half
    {
        int C = I.input_channels;
        size_t total = input_floats;
        int threads = 256;
        int blocks = (int)((total + threads - 1) / threads);
        transpose_nchw_fp32_to_fp16<<<blocks, threads, 0, I.dev.stream>>>(
            I.buf_flat_in, I.buf_input, N, C, HW);
        CUDA_CHECK(cudaGetLastError());
    }

    // Input conv 3×3 + BN + ReLU (WMMA)
    I.run_conv3x3(I.buf_input, I.buf_main,
                  I.input_conv_gpu, nullptr,
                  N, H, W, 1, true);

    // Residual blocks (WMMA)
    for (int i = 0; i < I.num_res_blocks; i++) {
        I.run_conv3x3(I.buf_main, I.buf_temp,
                      I.res_conv1_gpu[i], nullptr,
                      N, H, W, 1, true);

        I.run_conv3x3(I.buf_temp, I.buf_skip,
                      I.res_conv2_gpu[i], I.buf_main,
                      N, H, W, 2, true);

        std::swap(I.buf_main, I.buf_skip);
    }

    // Policy head: 1×1 conv FP16, softmax→FP32
    I.run_conv1x1_bn_relu_reshape(I.buf_main, I.buf_pol_out,
                                  I.policy_conv_gpu, N, HW);
    {
        int M = I.policy_fc_gpu.out_features;
        int K = I.policy_fc_gpu.in_features;
        int threads = 64;
        int blocks = (N + threads - 1) / threads;
        fc_bias_softmax_fp16_to_fp32<<<blocks, threads, 0, I.dev.stream>>>(
            I.policy_fc_gpu.weight, I.buf_pol_out, I.buf_pol_feat,
            I.policy_fc_gpu.bias, M, N, K);
        CUDA_CHECK(cudaGetLastError());
    }

    // Value head: 1×1 conv FP16, FC1 FP16, tanh→FP32
    I.run_conv1x1_bn_relu_reshape(I.buf_main, I.buf_val_h1,
                                  I.value_conv_gpu, N, HW);
    I.run_fc_bias_relu(I.buf_val_h1, I.buf_val_feat,
                       I.value_fc1_gpu, N, true);
    {
        int K = I.value_fc2_gpu.in_features;
        int threads = 64;
        int blocks = (N + threads - 1) / threads;
        fc_bias_tanh_fp16_to_fp32<<<blocks, threads, 0, I.dev.stream>>>(
            I.value_fc2_gpu.weight, I.buf_val_feat, I.buf_val_out,
            I.value_fc2_gpu.bias, N, K);
        CUDA_CHECK(cudaGetLastError());
    }

    // Read back FP32 results
    std::vector<float> pol_flat((size_t)action_size * N);
    CUDA_CHECK(cudaMemcpyAsync(pol_flat.data(), I.buf_pol_feat,
        pol_flat.size() * sizeof(float), cudaMemcpyDeviceToHost, I.dev.stream));

    std::vector<float> val_flat((size_t)N);
    CUDA_CHECK(cudaMemcpyAsync(val_flat.data(), I.buf_val_out,
        val_flat.size() * sizeof(float), cudaMemcpyDeviceToHost, I.dev.stream));

    CUDA_CHECK(cudaStreamSynchronize(I.dev.stream));

    // Pack results (same column-major→row-major unpacking)
    std::vector<std::pair<std::vector<float>, float>> results(N);
    for (int n = 0; n < N; n++) {
        std::vector<float> pol(action_size);
        for (int a = 0; a < action_size; a++)
            pol[a] = pol_flat[a * N + n];
        results[n] = { std::move(pol), val_flat[n] };
    }
    return results;
}

}  // namespace minigo
#endif  // MINIGO_HAS_CUDA
