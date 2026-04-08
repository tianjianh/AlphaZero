#ifdef MINIGO_HAS_CUDA

#include "cuda_compute.h"
#include "loaded_model.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>

// CUTLASS GEMM for optimized convolution
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/layout/matrix.h>
#include <cutlass/numeric_types.h>
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
    bool         use_fp16  = false;  // SM >= 7.0 for WMMA tensor cores
};

// ================================================================
// FP16 utility kernels
// ================================================================

__global__ void convert_fp32_to_fp16(const float* input, half* output, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) output[i] = __float2half(input[i]);
}

// Fused transpose + FP32→FP16: NCHW [N,C,HW] → channel-major FP16 [C, N*HW]
__global__ void transpose_nchw_fp32_to_fp16(
    const float* input, half* output, int N, int C, int HW
) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * C * HW;
    if (gid >= total) return;
    int n  = gid / (C * HW);
    int c  = (gid / HW) % C;
    int hw = gid % HW;
    output[c * (N * HW) + n * HW + hw] = __float2half(input[gid]);
}

// FP32 transpose: NCHW [N,C,HW] → channel-major FP32 [C, N*HW]
__global__ void transpose_nchw_fp32(
    const float* input, float* output, int N, int C, int HW
) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * C * HW;
    if (gid >= total) return;
    int n  = gid / (C * HW);
    int c  = (gid / HW) % C;
    int hw = gid % HW;
    output[c * (N * HW) + n * HW + hw] = input[gid];
}

// ================================================================
// Conv 3×3: WMMA Tensor Core kernel + fused BN + optional ReLU (FP16)
// (unchanged from original)
// ================================================================

#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16
#define WARPS_M 2
#define WARPS_N 4
#define TILE_M (WARPS_M * WMMA_M)  // 32
#define TILE_N (WARPS_N * WMMA_N)  // 64
#define SMEM_A_STRIDE (WMMA_K + 8) // 24
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
    __shared__ half smem_a[TILE_M][SMEM_A_STRIDE];
    __shared__ half smem_b[WMMA_K][SMEM_B_STRIDE];

    int warp_id = threadIdx.x / 32;
    int warp_m  = warp_id / WARPS_N;
    int warp_n  = warp_id % WARPS_N;
    int gm_base = blockIdx.y * TILE_M;
    int gn_base = blockIdx.x * TILE_N;
    if (gm_base >= C_out || gn_base >= NHW) return;
    int HW = H * W;

    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc;
    wmma::fill_fragment(acc, 0.0f);
    int numTiles = (K + WMMA_K - 1) / WMMA_K;

    for (int t = 0; t < numTiles; t++) {
        int k_off = t * WMMA_K;
        for (int idx = threadIdx.x; idx < TILE_M * WMMA_K; idx += 256) {
            int row = idx / WMMA_K, col = idx % WMMA_K;
            int gm = gm_base + row, gk = k_off + col;
            smem_a[row][col] = (gm < C_out && gk < K) ? A[gm * K + gk] : __float2half(0.0f);
        }
        for (int idx = threadIdx.x; idx < WMMA_K * TILE_N; idx += 256) {
            int kr = idx / TILE_N, nc = idx % TILE_N;
            int k_row = k_off + kr, gn = gn_base + nc;
            half val = __float2half(0.0f);
            if (k_row < K && gn < NHW) {
                int c_in = k_row / 9, rem = k_row % 9;
                int kh = rem / 3, kw = rem % 3;
                int n = gn / HW, hw = gn % HW;
                int ih = hw / W + kh - 1, iw = hw % W + kw - 1;
                if (ih >= 0 && ih < H && iw >= 0 && iw < W)
                    val = input[c_in * NHW + n * HW + ih * W + iw];
            }
            smem_b[kr][nc] = val;
        }
        __syncthreads();

        wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> frag_a;
        wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> frag_b;
        wmma::load_matrix_sync(frag_a, &smem_a[warp_m * WMMA_M][0], SMEM_A_STRIDE);
        wmma::load_matrix_sync(frag_b, &smem_b[0][warp_n * WMMA_N], SMEM_B_STRIDE);
        wmma::mma_sync(acc, frag_a, frag_b, acc);
        __syncthreads();
    }

    extern __shared__ float smem_out[];
    const int OUT_STRIDE = TILE_N + 8;
    wmma::store_matrix_sync(&smem_out[(warp_m * WMMA_M) * OUT_STRIDE + warp_n * WMMA_N],
                            acc, OUT_STRIDE, wmma::mem_row_major);
    __syncthreads();

    for (int idx = threadIdx.x; idx < TILE_M * TILE_N; idx += 256) {
        int row = idx / TILE_N, col = idx % TILE_N;
        int gm = gm_base + row, gn = gn_base + col;
        if (gm >= C_out || gn >= NHW) continue;
        float v = smem_out[row * OUT_STRIDE + col];
        if (mode >= 1) v = bn_scale[gm] * v + bn_bias[gm];
        if (mode == 2) v += __half2float(residual[gm * NHW + gn]);
        if (do_relu && v < 0.0f) v = 0.0f;
        output[gm * NHW + gn] = __float2half(v);
    }
}

// ================================================================
// Conv 3×3: FP32 fallback (no tensor cores)
//
// Simple per-element kernel with implicit im2col.
// Each thread computes one (c_out, nhw) output element.
// ================================================================
__global__ void conv3x3_bn_fp32(
    const float* __restrict__ weight,
    const float* __restrict__ input,
    float*       __restrict__ output,
    const float* __restrict__ bn_scale,
    const float* __restrict__ bn_bias,
    const float* __restrict__ residual,
    int C_out, int C_in, int NHW,
    int H, int W,
    int mode, int do_relu
) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= C_out * NHW) return;
    int co  = gid / NHW;
    int nhw = gid % NHW;
    int HW = H * W;
    int n  = nhw / HW;
    int hw = nhw % HW;
    int oh = hw / W, ow = hw % W;

    float acc = 0.0f;
    for (int ci = 0; ci < C_in; ci++) {
        for (int kh = 0; kh < 3; kh++) {
            int ih = oh + kh - 1;
            if (ih < 0 || ih >= H) continue;
            for (int kw = 0; kw < 3; kw++) {
                int iw = ow + kw - 1;
                if (iw < 0 || iw >= W) continue;
                acc += weight[co * (C_in * 9) + ci * 9 + kh * 3 + kw]
                     * input[ci * NHW + n * HW + ih * W + iw];
            }
        }
    }
    if (mode >= 1) acc = bn_scale[co] * acc + bn_bias[co];
    if (mode == 2) acc += residual[co * NHW + nhw];
    if (do_relu && acc < 0.0f) acc = 0.0f;
    output[co * NHW + nhw] = acc;
}

// ================================================================
// ================================================================
// Im2col kernel: [C_in, NHW] → [C_in*9, NHW] for 3×3 conv pad=1
// ================================================================
__global__ void im2col_3x3_fp16(
    const half* __restrict__ input, half* __restrict__ col,
    int C_in, int NHW, int H, int W
) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int K = C_in * 9;
    int total = K * NHW;
    if (gid >= total) return;
    int k_row = gid / NHW;
    int nhw   = gid % NHW;
    int HW = H * W;
    int ci = k_row / 9, rem = k_row % 9;
    int kh = rem / 3, kw = rem % 3;
    int n = nhw / HW, hw = nhw % HW;
    int ih = hw / W + kh - 1, iw = hw % W + kw - 1;
    col[gid] = (ih >= 0 && ih < H && iw >= 0 && iw < W)
        ? input[ci * NHW + n * HW + ih * W + iw] : __float2half(0.0f);
}

// BN + optional residual + ReLU on GEMM output [C_out, NHW]
__global__ void bn_relu_fp16(
    half* __restrict__ data, const float* __restrict__ bn_scale,
    const float* __restrict__ bn_bias, const half* __restrict__ residual,
    int C_out, int NHW, int mode
) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= C_out * NHW) return;
    int c = gid / NHW;
    float v = bn_scale[c] * __half2float(data[gid]) + bn_bias[c];
    if (mode == 2) v += __half2float(residual[gid]);
    if (v < 0.0f) v = 0.0f;
    data[gid] = __float2half(v);
}

// ================================================================
// CUTLASS GEMM conv3x3 — uses im2col + optimized GEMM
//
// Tile sizes tuned for M=64 (small filter count):
// - ThreadblockShape: 64×128×32 — covers full M in one tile
// - WarpShape: 32×64×32 — 2×2 warp arrangement
// - InstructionShape: 16×8×16 — WMMA on Turing+
// ================================================================

// CUTLASS GEMM type for FP16 Tensor Core with FP32 accumulation
// Alignment=1 to support any N (e.g., N=81 for single-sample inference)
using CutlassGemm = cutlass::gemm::device::Gemm<
    cutlass::half_t,                           // A type
    cutlass::layout::RowMajor,                 // A layout
    cutlass::half_t,                           // B type
    cutlass::layout::RowMajor,                 // B layout
    cutlass::half_t,                           // C type
    cutlass::layout::RowMajor,                 // C layout
    float,                                      // accumulator
    cutlass::arch::OpClassTensorOp,             // use tensor cores
    cutlass::arch::Sm75,                        // Turing
    cutlass::gemm::GemmShape<64, 64, 32>,       // threadblock tile
    cutlass::gemm::GemmShape<32, 32, 32>,       // warp tile
    cutlass::gemm::GemmShape<16, 8, 8>,         // Turing mma instruction
    cutlass::epilogue::thread::LinearCombination<
        cutlass::half_t, 1, float, float>,      // alignment=1 for arbitrary N
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
    2,                                          // pipeline stages
    1,                                          // A alignment
    1                                           // B alignment
>;

// Fused 1×1 conv + BN + ReLU + reshape — FP16 and FP32 versions
// ================================================================
__global__ void conv1x1_bn_relu_reshape_fp16(
    const half* __restrict__ input, half* __restrict__ output,
    const half* __restrict__ weight, const float* __restrict__ bn_scale,
    const float* __restrict__ bn_bias,
    int C_in, int C_out, int N, int HW
) {
    extern __shared__ char smem_raw[];
    half*  sw = (half*)smem_raw;
    float* s_scale = (float*)(sw + C_out * C_in);
    float* s_bias  = s_scale + C_out;

    int total_w = C_out * C_in;
    for (int i = threadIdx.x; i < total_w; i += blockDim.x) sw[i] = weight[i];
    if (threadIdx.x < C_out) {
        s_scale[threadIdx.x] = bn_scale[threadIdx.x];
        s_bias[threadIdx.x]  = bn_bias[threadIdx.x];
    }
    __syncthreads();

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int n  = idx % N, hw = idx / N;
    if (hw >= HW || n >= N) return;
    int NHW = N * HW;
    int C_in_half2 = C_in / 2;

    for (int co = 0; co < C_out; co++) {
        float acc = 0.0f;
        const half* w_row = sw + co * C_in;
        for (int ci2 = 0; ci2 < C_in_half2; ci2++) {
            half2 wv = ((const half2*)w_row)[ci2];
            int ci = ci2 * 2;
            half2 iv = __halves2half2(input[ci * NHW + n * HW + hw],
                                      input[(ci + 1) * NHW + n * HW + hw]);
            float2 wf = __half22float2(wv), xf = __half22float2(iv);
            acc += wf.x * xf.x + wf.y * xf.y;
        }
        if (C_in & 1) {
            int ci = C_in - 1;
            acc += __half2float(w_row[ci]) * __half2float(input[ci * NHW + n * HW + hw]);
        }
        float v = s_scale[co] * acc + s_bias[co];
        if (v < 0.0f) v = 0.0f;
        output[(co * HW + hw) * N + n] = __float2half(v);
    }
}

__global__ void conv1x1_bn_relu_reshape_fp32(
    const float* __restrict__ input, float* __restrict__ output,
    const float* __restrict__ weight, const float* __restrict__ bn_scale,
    const float* __restrict__ bn_bias,
    int C_in, int C_out, int N, int HW
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int n  = idx % N, hw = idx / N;
    if (hw >= HW || n >= N) return;
    int NHW = N * HW;

    for (int co = 0; co < C_out; co++) {
        float acc = 0.0f;
        for (int ci = 0; ci < C_in; ci++)
            acc += weight[co * C_in + ci] * input[ci * NHW + n * HW + hw];
        float v = bn_scale[co] * acc + bn_bias[co];
        if (v < 0.0f) v = 0.0f;
        output[(co * HW + hw) * N + n] = v;
    }
}

// ================================================================
// FC + bias + optional ReLU — FP16 and FP32
// ================================================================
#define FC_WARPS_PER_BLOCK 8

__global__ void fc_bias_relu_fp16(
    const half* __restrict__ weight, const half* __restrict__ input,
    half* __restrict__ output,
    const float* __restrict__ bias, int M, int N, int K, int do_relu
) {
    int lane = threadIdx.x, warp = threadIdx.y;
    int m = blockIdx.x, n = blockIdx.y * FC_WARPS_PER_BLOCK + warp;
    if (m >= M || n >= N) return;
    float acc = 0.0f;
    for (int k = lane; k < K; k += 32)
        acc += __half2float(weight[m * K + k]) * __half2float(input[k * N + n]);
    for (int offset = 16; offset > 0; offset >>= 1)
        acc += __shfl_down_sync(0xffffffff, acc, offset);
    if (lane == 0) {
        acc += bias[m];
        if (do_relu && acc < 0.0f) acc = 0.0f;
        output[m * N + n] = __float2half(acc);
    }
}

__global__ void fc_bias_relu_fp32(
    const float* __restrict__ weight, const float* __restrict__ input,
    float* __restrict__ output,
    const float* __restrict__ bias, int M, int N, int K, int do_relu
) {
    int lane = threadIdx.x, warp = threadIdx.y;
    int m = blockIdx.x, n = blockIdx.y * FC_WARPS_PER_BLOCK + warp;
    if (m >= M || n >= N) return;
    float acc = 0.0f;
    for (int k = lane; k < K; k += 32)
        acc += weight[m * K + k] * input[k * N + n];
    for (int offset = 16; offset > 0; offset >>= 1)
        acc += __shfl_down_sync(0xffffffff, acc, offset);
    if (lane == 0) {
        acc += bias[m];
        if (do_relu && acc < 0.0f) acc = 0.0f;
        output[m * N + n] = acc;
    }
}

// ================================================================
// FC + bias + softmax — FP16 and FP32 → FP32 output
// ================================================================
__global__ void fc_bias_softmax_fp16_to_fp32(
    const half* __restrict__ weight, const half* __restrict__ input,
    float* __restrict__ output,
    const float* __restrict__ bias, int M, int N, int K
) {
    int n = blockIdx.x;
    if (n >= N) return;
    int lane = threadIdx.x;
    extern __shared__ half s_input_h[];
    for (int k = lane; k < K; k += 32) s_input_h[k] = input[k * N + n];
    __syncwarp();

    float mx = -1e30f;
    for (int m = 0; m < M; m++) {
        float acc = 0.0f;
        for (int k = lane; k < K; k += 32)
            acc += __half2float(weight[m * K + k]) * __half2float(s_input_h[k]);
        for (int off = 16; off > 0; off >>= 1) acc += __shfl_down_sync(0xffffffff, acc, off);
        if (lane == 0) { acc += bias[m]; output[m * N + n] = acc; mx = fmaxf(mx, acc); }
    }
    mx = __shfl_sync(0xffffffff, mx, 0);
    if (lane == 0) {
        float sum = 0.0f;
        for (int m = 0; m < M; m++) { float e = expf(output[m * N + n] - mx); output[m * N + n] = e; sum += e; }
        float inv = 1.0f / sum;
        for (int m = 0; m < M; m++) output[m * N + n] *= inv;
    }
}

__global__ void fc_bias_softmax_fp32(
    const float* __restrict__ weight, const float* __restrict__ input,
    float* __restrict__ output,
    const float* __restrict__ bias, int M, int N, int K
) {
    int n = blockIdx.x;
    if (n >= N) return;
    int lane = threadIdx.x;
    extern __shared__ float s_input_f[];
    for (int k = lane; k < K; k += 32) s_input_f[k] = input[k * N + n];
    __syncwarp();

    float mx = -1e30f;
    for (int m = 0; m < M; m++) {
        float acc = 0.0f;
        for (int k = lane; k < K; k += 32)
            acc += weight[m * K + k] * s_input_f[k];
        for (int off = 16; off > 0; off >>= 1) acc += __shfl_down_sync(0xffffffff, acc, off);
        if (lane == 0) { acc += bias[m]; output[m * N + n] = acc; mx = fmaxf(mx, acc); }
    }
    mx = __shfl_sync(0xffffffff, mx, 0);
    if (lane == 0) {
        float sum = 0.0f;
        for (int m = 0; m < M; m++) { float e = expf(output[m * N + n] - mx); output[m * N + n] = e; sum += e; }
        float inv = 1.0f / sum;
        for (int m = 0; m < M; m++) output[m * N + n] *= inv;
    }
}

// ================================================================
// FC + bias + tanh → FP32 output (value head)
// ================================================================
__global__ void fc_bias_tanh_fp16_to_fp32(
    const half* __restrict__ weight, const half* __restrict__ input,
    float* __restrict__ output,
    const float* __restrict__ bias, int N, int K
) {
    int n = blockIdx.x;
    if (n >= N) return;
    int lane = threadIdx.x;
    float acc = 0.0f;
    for (int k = lane; k < K; k += 32)
        acc += __half2float(weight[k]) * __half2float(input[k * N + n]);
    for (int off = 16; off > 0; off >>= 1) acc += __shfl_down_sync(0xffffffff, acc, off);
    if (lane == 0) output[n] = tanhf(acc + bias[0]);
}

__global__ void fc_bias_tanh_fp32(
    const float* __restrict__ weight, const float* __restrict__ input,
    float* __restrict__ output,
    const float* __restrict__ bias, int N, int K
) {
    int n = blockIdx.x;
    if (n >= N) return;
    int lane = threadIdx.x;
    float acc = 0.0f;
    for (int k = lane; k < K; k += 32)
        acc += weight[k] * input[k * N + n];
    for (int off = 16; off > 0; off >>= 1) acc += __shfl_down_sync(0xffffffff, acc, off);
    if (lane == 0) output[n] = tanhf(acc + bias[0]);
}

// ================================================================
// Score head: softmax over bins → expected value → FP32 output
//
// Input: [num_bins, N] channel-major (from FC output, with bias already added)
// Output: [N] float — expected score in raw points
// One warp per batch element, warp-shuffle for reduction.
// ================================================================
__global__ void score_softmax_ev_fp16(
    const half* __restrict__ input, float* __restrict__ output,
    const float* __restrict__ bin_values, int N, int num_bins
) {
    int n = blockIdx.x;
    if (n >= N) return;
    int lane = threadIdx.x;
    float mx = -1e30f;
    for (int b = lane; b < num_bins; b += 32)
        mx = fmaxf(mx, __half2float(input[b * N + n]));
    for (int off = 16; off > 0; off >>= 1) mx = fmaxf(mx, __shfl_down_sync(0xffffffff, mx, off));
    mx = __shfl_sync(0xffffffff, mx, 0);

    float sum = 0.0f, ev = 0.0f;
    for (int b = lane; b < num_bins; b += 32) {
        float e = expf(__half2float(input[b * N + n]) - mx);
        sum += e; ev += e * bin_values[b];
    }
    for (int off = 16; off > 0; off >>= 1) {
        sum += __shfl_down_sync(0xffffffff, sum, off);
        ev  += __shfl_down_sync(0xffffffff, ev, off);
    }
    if (lane == 0) output[n] = ev / sum;
}

__global__ void score_softmax_ev_fp32(
    const float* __restrict__ input, float* __restrict__ output,
    const float* __restrict__ bin_values, int N, int num_bins
) {
    int n = blockIdx.x;
    if (n >= N) return;
    int lane = threadIdx.x;
    float mx = -1e30f;
    for (int b = lane; b < num_bins; b += 32)
        mx = fmaxf(mx, input[b * N + n]);
    for (int off = 16; off > 0; off >>= 1) mx = fmaxf(mx, __shfl_down_sync(0xffffffff, mx, off));
    mx = __shfl_sync(0xffffffff, mx, 0);

    float sum = 0.0f, ev = 0.0f;
    for (int b = lane; b < num_bins; b += 32) {
        float e = expf(input[b * N + n] - mx);
        sum += e; ev += e * bin_values[b];
    }
    for (int off = 16; off > 0; off >>= 1) {
        sum += __shfl_down_sync(0xffffffff, sum, off);
        ev  += __shfl_down_sync(0xffffffff, ev, off);
    }
    if (lane == 0) output[n] = ev / sum;
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
    if (device_id < 0 || device_id >= device_count)
        throw std::runtime_error("CUDA device_id " + std::to_string(device_id) + " out of range");

    ds.device_id = device_id;
    CUDA_CHECK(cudaSetDevice(device_id));
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device_id));
    ds.use_fp16 = (prop.major >= 7);  // WMMA requires Volta+

    CUDA_CHECK(cudaStreamCreate(&ds.stream));
    const char* prec = ds.use_fp16 ? "FP16+WMMA" : "FP32";
    std::cout << "CUDA device " << device_id << ": " << prop.name
              << " (SM " << prop.major << "." << prop.minor << ", " << prec << ")\n";
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
            if (ds.stream) { cudaSetDevice(ds.device_id); cudaStreamDestroy(ds.stream); }
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
// CUDAComputeHandle::Impl — dual FP16/FP32 paths
// ================================================================

struct CUDAComputeHandle::Impl {
    CUDADeviceState& dev;
    bool use_fp16;
    size_t elem;  // sizeof(half) or sizeof(float)

    struct ConvBNGPU {
        void*  weight   = nullptr;  // FP16 or FP32
        float* bn_scale = nullptr;
        float* bn_bias  = nullptr;
        int c_out = 0, c_in = 0, k = 0;
    };
    struct FCGPU {
        void*  weight = nullptr;   // FP16 or FP32
        float* bias   = nullptr;
        int out_features = 0, in_features = 0;
    };

    ConvBNGPU              input_conv_gpu;
    std::vector<ConvBNGPU> res_conv1_gpu, res_conv2_gpu;
    ConvBNGPU              policy_conv_gpu, value_conv_gpu, score_conv_gpu;
    FCGPU                  policy_fc_gpu, value_fc1_gpu, value_fc2_gpu;
    FCGPU                  score_fc1_gpu, score_fc2_gpu;
    float*                 bin_values_gpu = nullptr;

    half*  buf_im2col  = nullptr;  // [C_in*9, N*HW] for CUTLASS GEMM im2col workspace
    float* buf_flat_in = nullptr;  // FP32 from host
    void*  buf_input   = nullptr;  // [C_in, N*HW]
    void*  buf_main    = nullptr;  // [F, N*HW]
    void*  buf_temp    = nullptr;  // [F, N*HW]
    void*  buf_skip    = nullptr;  // [F, N*HW]
    void*  buf_pol_out = nullptr;  // [2*HW, N]
    void*  buf_val_h1  = nullptr;  // [HW, N]
    void*  buf_val_feat= nullptr;  // [64, N]
    float* buf_pol_feat= nullptr;  // [action_size, N] FP32
    float* buf_val_out = nullptr;  // [N] FP32
    void*  buf_scr_h1  = nullptr;  // [HW, N]
    void*  buf_scr_feat= nullptr;  // [64, N]
    void*  buf_scr_bins= nullptr;  // [num_bins, N]
    float* buf_scr_out = nullptr;  // [N] FP32

    // CUDA graph cache — replays the entire compute pipeline without
    // per-kernel launch overhead (~5-10µs × 21 kernels eliminated)
    cudaGraph_t     graph      = nullptr;
    cudaGraphExec_t graph_exec = nullptr;
    int             graph_batch = 0;

    // Host-side pre-allocated buffers (avoid per-call std::vector alloc)
    std::vector<float> host_input;
    std::vector<float> host_pol, host_val, host_scr;

    int alloc_batch = 0;
    int board_size, input_channels, num_filters, num_res_blocks, num_score_bins;

    explicit Impl(CUDADeviceState& d) : dev(d), use_fp16(d.use_fp16) {
        elem = use_fp16 ? sizeof(half) : sizeof(float);
    }

    // Upload FP32 vector → GPU (FP16 or FP32 based on use_fp16)
    void* upload_compute(const std::vector<float>& data) {
        size_t n = data.size();
        if (use_fp16) {
            float* tmp; CUDA_CHECK(cudaMalloc(&tmp, n * sizeof(float)));
            CUDA_CHECK(cudaMemcpy(tmp, data.data(), n * sizeof(float), cudaMemcpyHostToDevice));
            half* buf; CUDA_CHECK(cudaMalloc(&buf, n * sizeof(half)));
            convert_fp32_to_fp16<<<((int)n + 255) / 256, 256, 0, dev.stream>>>(tmp, buf, (int)n);
            CUDA_CHECK(cudaStreamSynchronize(dev.stream));
            cudaFree(tmp);
            return buf;
        } else {
            float* buf; CUDA_CHECK(cudaMalloc(&buf, n * sizeof(float)));
            CUDA_CHECK(cudaMemcpy(buf, data.data(), n * sizeof(float), cudaMemcpyHostToDevice));
            return buf;
        }
    }
    float* upload_float(const std::vector<float>& data) {
        float* buf; CUDA_CHECK(cudaMalloc(&buf, data.size() * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(buf, data.data(), data.size() * sizeof(float), cudaMemcpyHostToDevice));
        return buf;
    }

    template<typename T> void release_buf(T*& buf) { if (buf) { cudaFree(buf); buf = nullptr; } }

    void upload_conv(ConvBNGPU& g, const ConvBNWeights& src) {
        g.c_out = src.c_out; g.c_in = src.c_in; g.k = src.k;
        g.weight   = upload_compute(src.weight);
        g.bn_scale = upload_float(src.bn_scale);
        g.bn_bias  = upload_float(src.bn_bias);
    }
    void upload_fc(FCGPU& g, const FCWeights& src) {
        g.out_features = src.out_features; g.in_features = src.in_features;
        g.weight = upload_compute(src.weight);
        g.bias   = upload_float(src.bias);
    }
    void free_conv(ConvBNGPU& c) {
        if (c.weight) cudaFree(c.weight); release_buf(c.bn_scale); release_buf(c.bn_bias); c = {};
    }
    void free_fc(FCGPU& f) {
        if (f.weight) cudaFree(f.weight); release_buf(f.bias); f = {};
    }

    void allocate_workspace(int batch) {
        if (batch <= alloc_batch) return;
        free_workspace();
        int H = board_size, W = board_size, HW = H * W;
        int F = num_filters, as = HW + 1;

        // Im2col workspace for CUTLASS (max K = F*9 for res convs)
        if (use_fp16)
            CUDA_CHECK(cudaMalloc(&buf_im2col, (size_t)num_filters * 9 * batch * HW * sizeof(half)));
        CUDA_CHECK(cudaMalloc(&buf_flat_in, (size_t)input_channels * batch * HW * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&buf_input,   (size_t)input_channels * batch * HW * elem));
        CUDA_CHECK(cudaMalloc(&buf_main,    (size_t)F * batch * HW * elem));
        CUDA_CHECK(cudaMalloc(&buf_temp,    (size_t)F * batch * HW * elem));
        CUDA_CHECK(cudaMalloc(&buf_skip,    (size_t)F * batch * HW * elem));
        CUDA_CHECK(cudaMalloc(&buf_pol_out, (size_t)2 * HW * batch * elem));
        CUDA_CHECK(cudaMalloc(&buf_pol_feat,(size_t)as * batch * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&buf_val_h1,  (size_t)HW * batch * elem));
        CUDA_CHECK(cudaMalloc(&buf_val_feat,(size_t)F * batch * elem));
        CUDA_CHECK(cudaMalloc(&buf_val_out, (size_t)batch * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&buf_scr_h1,  (size_t)HW * batch * elem));
        CUDA_CHECK(cudaMalloc(&buf_scr_feat,(size_t)F * batch * elem));
        CUDA_CHECK(cudaMalloc(&buf_scr_bins,(size_t)num_score_bins * batch * elem));
        CUDA_CHECK(cudaMalloc(&buf_scr_out, (size_t)batch * sizeof(float)));
        alloc_batch = batch;
    }

    void free_workspace() {
        auto fr = [](auto*& p) { if (p) { cudaFree(p); p = nullptr; } };
        fr(buf_im2col); fr(buf_flat_in); fr(buf_input); fr(buf_main); fr(buf_temp); fr(buf_skip);
        fr(buf_pol_out); fr(buf_pol_feat); fr(buf_val_h1); fr(buf_val_feat); fr(buf_val_out);
        fr(buf_scr_h1); fr(buf_scr_feat); fr(buf_scr_bins); fr(buf_scr_out);
        if (graph_exec) { cudaGraphExecDestroy(graph_exec); graph_exec = nullptr; }
        if (graph) { cudaGraphDestroy(graph); graph = nullptr; }
        graph_batch = 0;
        alloc_batch = 0;
    }

    static constexpr int WMMA_SMEM_BYTES = TILE_M * (TILE_N + 8) * (int)sizeof(float);

    // Launch the entire compute pipeline (transpose + trunk + heads).
    // Called during graph capture AND during non-graph execution.
    // Uses local pointer copies for ping-pong (doesn't modify members).
    void launch_compute(int N) {
        int H = board_size, W = board_size, HW = H * W;
        int action_size = HW + 1;

        // Transpose
        size_t input_floats = (size_t)N * input_channels * HW;
        int thr = 256, blk = (int)((input_floats + thr - 1) / thr);
        if (use_fp16)
            transpose_nchw_fp32_to_fp16<<<blk, thr, 0, dev.stream>>>(
                buf_flat_in, (half*)buf_input, N, input_channels, HW);
        else
            transpose_nchw_fp32<<<blk, thr, 0, dev.stream>>>(
                buf_flat_in, (float*)buf_input, N, input_channels, HW);

        // Input conv
        run_conv3x3(buf_input, buf_main, input_conv_gpu, nullptr, N, H, W, 1, true);

        // Residual blocks — use local pointers for ping-pong
        void* cur = buf_main;
        void* alt = buf_skip;
        for (int i = 0; i < num_res_blocks; i++) {
            run_conv3x3(cur, buf_temp, res_conv1_gpu[i], nullptr, N, H, W, 1, true);
            run_conv3x3(buf_temp, alt, res_conv2_gpu[i], cur, N, H, W, 2, true);
            std::swap(cur, alt);
        }
        // cur now points to the trunk output

        // Policy head
        run_conv1x1(cur, buf_pol_out, policy_conv_gpu, N, HW);
        {
            int M = policy_fc_gpu.out_features, K = policy_fc_gpu.in_features;
            int smem = use_fp16 ? K * (int)sizeof(half) : K * (int)sizeof(float);
            if (use_fp16)
                fc_bias_softmax_fp16_to_fp32<<<N, 32, smem, dev.stream>>>(
                    (const half*)policy_fc_gpu.weight, (const half*)buf_pol_out, buf_pol_feat,
                    policy_fc_gpu.bias, M, N, K);
            else
                fc_bias_softmax_fp32<<<N, 32, smem, dev.stream>>>(
                    (const float*)policy_fc_gpu.weight, (const float*)buf_pol_out, buf_pol_feat,
                    policy_fc_gpu.bias, M, N, K);
        }

        // Value head
        run_conv1x1(cur, buf_val_h1, value_conv_gpu, N, HW);
        run_fc(buf_val_h1, buf_val_feat, value_fc1_gpu, N, true);
        {
            int K = value_fc2_gpu.in_features;
            if (use_fp16)
                fc_bias_tanh_fp16_to_fp32<<<N, 32, 0, dev.stream>>>(
                    (const half*)value_fc2_gpu.weight, (const half*)buf_val_feat, buf_val_out,
                    value_fc2_gpu.bias, N, K);
            else
                fc_bias_tanh_fp32<<<N, 32, 0, dev.stream>>>(
                    (const float*)value_fc2_gpu.weight, (const float*)buf_val_feat, buf_val_out,
                    value_fc2_gpu.bias, N, K);
        }

        // Score head
        run_conv1x1(cur, buf_scr_h1, score_conv_gpu, N, HW);
        run_fc(buf_scr_h1, buf_scr_feat, score_fc1_gpu, N, true);
        run_fc(buf_scr_feat, buf_scr_bins, score_fc2_gpu, N, false);
        {
            if (use_fp16)
                score_softmax_ev_fp16<<<N, 32, 0, dev.stream>>>(
                    (const half*)buf_scr_bins, buf_scr_out, bin_values_gpu, N, num_score_bins);
            else
                score_softmax_ev_fp32<<<N, 32, 0, dev.stream>>>(
                    (const float*)buf_scr_bins, buf_scr_out, bin_values_gpu, N, num_score_bins);
        }
    }

    void ensure_graph(int N) {
        if (N == graph_batch) return;
        if (graph_exec) { cudaGraphExecDestroy(graph_exec); graph_exec = nullptr; }
        if (graph) { cudaGraphDestroy(graph); graph = nullptr; }

        CUDA_CHECK(cudaStreamBeginCapture(dev.stream, cudaStreamCaptureModeGlobal));
        launch_compute(N);
        CUDA_CHECK(cudaStreamEndCapture(dev.stream, &graph));
        CUDA_CHECK(cudaGraphInstantiate(&graph_exec, graph, nullptr, nullptr, 0));
        graph_batch = N;
    }

    // ── Dispatch helpers ────────────────────────────────────

    // CUTLASS conv3x3: im2col + GEMM + BN/ReLU (FP16 tensor core path)
    void run_conv3x3_cutlass(half* in, half* out, const ConvBNGPU& conv, half* residual,
                              int N, int H, int W, int mode) {
        int NHW = N * H * W;
        int C_out = conv.c_out, C_in = conv.c_in, K = C_in * 9;

        // Im2col: [C_in, NHW] → [K, NHW]
        int total_im2col = K * NHW;
        im2col_3x3_fp16<<<(total_im2col + 255) / 256, 256, 0, dev.stream>>>(
            in, buf_im2col, C_in, NHW, H, W);

        // CUTLASS GEMM: C[C_out, NHW] = A[C_out, K] × B[K, NHW]
        CutlassGemm gemm_op;
        cutlass::half_t alpha(1.0f), beta(0.0f);
        CutlassGemm::Arguments args(
            {C_out, NHW, K},
            {(cutlass::half_t*)conv.weight, K},
            {(cutlass::half_t*)buf_im2col, NHW},
            {(cutlass::half_t*)out, NHW},
            {(cutlass::half_t*)out, NHW},
            {alpha, beta}
        );
        cutlass::Status status = gemm_op.can_implement(args);
        if (status != cutlass::Status::kSuccess) {
            std::cerr << "CUTLASS: can_implement failed (" << (int)status
                      << ") M=" << C_out << " N=" << NHW << " K=" << K << "\n";
            throw std::runtime_error("CUTLASS GEMM cannot implement");
        }
        size_t ws_size = CutlassGemm::get_workspace_size(args);
        void* ws = nullptr;
        if (ws_size > 0) CUDA_CHECK(cudaMalloc(&ws, ws_size));
        status = gemm_op(args, ws, dev.stream);
        if (ws) cudaFree(ws);
        if (status != cutlass::Status::kSuccess)
            throw std::runtime_error("CUTLASS GEMM launch failed");
        CUDA_CHECK(cudaGetLastError());

        // BN + optional residual + ReLU
        int total = C_out * NHW;
        bn_relu_fp16<<<(total + 255) / 256, 256, 0, dev.stream>>>(
            out, conv.bn_scale, conv.bn_bias,
            (mode == 2 && residual) ? residual : nullptr,
            C_out, NHW, mode);
    }

    void run_conv3x3(void* in, void* out, const ConvBNGPU& conv, void* residual,
                     int N, int H, int W, int mode, bool relu) {
        int NHW = N * H * W, do_relu = relu ? 1 : 0;
        if (use_fp16) {
            // Use CUTLASS GEMM for optimized tensor core performance
            run_conv3x3_cutlass((half*)in, (half*)out, conv, (half*)residual,
                                N, H, W, mode);
        } else {
            int total = conv.c_out * NHW;
            conv3x3_bn_fp32<<<(total + 255) / 256, 256, 0, dev.stream>>>(
                (const float*)conv.weight, (const float*)in, (float*)out,
                conv.bn_scale, conv.bn_bias,
                (mode == 2 && residual) ? (const float*)residual : (const float*)out,
                conv.c_out, conv.c_in, NHW, H, W, mode, do_relu);
        }
        CUDA_CHECK(cudaGetLastError());
    }

    void run_conv1x1(void* in, void* out, const ConvBNGPU& conv, int N, int HW) {
        int total = N * HW;
        if (use_fp16) {
            int smem = conv.c_out * conv.c_in * (int)sizeof(half) + 2 * conv.c_out * (int)sizeof(float);
            conv1x1_bn_relu_reshape_fp16<<<(total + 255) / 256, 256, smem, dev.stream>>>(
                (const half*)in, (half*)out, (const half*)conv.weight, conv.bn_scale, conv.bn_bias,
                conv.c_in, conv.c_out, N, HW);
        } else {
            conv1x1_bn_relu_reshape_fp32<<<(total + 255) / 256, 256, 0, dev.stream>>>(
                (const float*)in, (float*)out, (const float*)conv.weight, conv.bn_scale, conv.bn_bias,
                conv.c_in, conv.c_out, N, HW);
        }
        CUDA_CHECK(cudaGetLastError());
    }

    void run_fc(void* in, void* out, const FCGPU& fc, int N, bool relu) {
        int M = fc.out_features, K = fc.in_features;
        dim3 block(32, FC_WARPS_PER_BLOCK);
        dim3 grid(M, (N + FC_WARPS_PER_BLOCK - 1) / FC_WARPS_PER_BLOCK);
        if (use_fp16)
            fc_bias_relu_fp16<<<grid, block, 0, dev.stream>>>(
                (const half*)fc.weight, (const half*)in, (half*)out, fc.bias, M, N, K, relu ? 1 : 0);
        else
            fc_bias_relu_fp32<<<grid, block, 0, dev.stream>>>(
                (const float*)fc.weight, (const float*)in, (float*)out, fc.bias, M, N, K, relu ? 1 : 0);
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

    I.board_size      = model->board_size;
    I.input_channels  = model->input_channels;
    I.num_filters     = model->num_filters;
    I.num_res_blocks  = model->num_res_blocks;
    I.num_score_bins  = model->score_fc2.out_features;

    I.upload_conv(I.input_conv_gpu, model->input_conv);
    I.res_conv1_gpu.resize(I.num_res_blocks);
    I.res_conv2_gpu.resize(I.num_res_blocks);
    for (int i = 0; i < I.num_res_blocks; i++) {
        I.upload_conv(I.res_conv1_gpu[i], model->res_conv1[i]);
        I.upload_conv(I.res_conv2_gpu[i], model->res_conv2[i]);
    }
    I.upload_conv(I.policy_conv_gpu, model->policy_conv);
    I.upload_conv(I.value_conv_gpu,  model->value_conv);
    I.upload_conv(I.score_conv_gpu,  model->score_conv);
    I.upload_fc(I.policy_fc_gpu, model->policy_fc);
    I.upload_fc(I.value_fc1_gpu, model->value_fc1);
    I.upload_fc(I.value_fc2_gpu, model->value_fc2);
    I.upload_fc(I.score_fc1_gpu, model->score_fc1);
    I.upload_fc(I.score_fc2_gpu, model->score_fc2);

    // Score bin values: [-board_area, ..., +board_area]
    int ba = I.board_size * I.board_size;
    std::vector<float> bv(I.num_score_bins);
    for (int i = 0; i < I.num_score_bins; i++) bv[i] = (float)(i - ba);
    I.bin_values_gpu = I.upload_float(bv);

    I.allocate_workspace(max_batch_size > 0 ? max_batch_size : 32);

    std::cout << "CUDA handle ready (" << (I.use_fp16 ? "FP16+WMMA" : "FP32")
              << "): board=" << I.board_size
              << " filters=" << I.num_filters
              << " blocks="  << I.num_res_blocks
              << " score_bins=" << I.num_score_bins << "\n";
}

CUDAComputeHandle::~CUDAComputeHandle() {
    if (impl_) {
        cudaSetDevice(impl_->dev.device_id);
        impl_->free_workspace();
        impl_->free_conv(impl_->input_conv_gpu);
        for (auto& c : impl_->res_conv1_gpu) impl_->free_conv(c);
        for (auto& c : impl_->res_conv2_gpu) impl_->free_conv(c);
        impl_->free_conv(impl_->policy_conv_gpu);
        impl_->free_conv(impl_->value_conv_gpu);
        impl_->free_conv(impl_->score_conv_gpu);
        impl_->free_fc(impl_->policy_fc_gpu);
        impl_->free_fc(impl_->value_fc1_gpu);
        impl_->free_fc(impl_->value_fc2_gpu);
        impl_->free_fc(impl_->score_fc1_gpu);
        impl_->free_fc(impl_->score_fc2_gpu);
        if (impl_->bin_values_gpu) cudaFree(impl_->bin_values_gpu);
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
    int action_size = HW + 1;

    I.allocate_workspace(N);

    // Flatten input into pre-allocated host buffer
    size_t input_floats = (size_t)N * I.input_channels * HW;
    I.host_input.resize(input_floats);
    float* dst = I.host_input.data();
    for (auto& s : states) {
        std::memcpy(dst, s.data(), s.size() * sizeof(float));
        dst += s.size();
    }

    // Upload input (not part of graph — source address changes)
    CUDA_CHECK(cudaMemcpyAsync(I.buf_flat_in, I.host_input.data(),
        input_floats * sizeof(float), cudaMemcpyHostToDevice, I.dev.stream));

    // Execute compute pipeline via CUDA graph (or capture on first call / batch change)
    I.ensure_graph(N);
    CUDA_CHECK(cudaGraphLaunch(I.graph_exec, I.dev.stream));

    // Read back results into pre-allocated host buffers
    I.host_pol.resize((size_t)action_size * N);
    I.host_val.resize(N);
    I.host_scr.resize(N);

    CUDA_CHECK(cudaMemcpyAsync(I.host_pol.data(), I.buf_pol_feat,
        I.host_pol.size() * sizeof(float), cudaMemcpyDeviceToHost, I.dev.stream));
    CUDA_CHECK(cudaMemcpyAsync(I.host_val.data(), I.buf_val_out,
        N * sizeof(float), cudaMemcpyDeviceToHost, I.dev.stream));
    CUDA_CHECK(cudaMemcpyAsync(I.host_scr.data(), I.buf_scr_out,
        N * sizeof(float), cudaMemcpyDeviceToHost, I.dev.stream));

    CUDA_CHECK(cudaStreamSynchronize(I.dev.stream));

    // Pack results
    std::vector<CUDAComputeHandle::Result> results(N);
    for (int n = 0; n < N; n++) {
        std::vector<float> pol(action_size);
        for (int a = 0; a < action_size; a++)
            pol[a] = I.host_pol[a * N + n];
        results[n].policy = std::move(pol);
        results[n].value  = I.host_val[n];
        results[n].score  = I.host_scr[n];
    }
    return results;
}

}  // namespace minigo
#endif  // MINIGO_HAS_CUDA
