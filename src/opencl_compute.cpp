#ifdef MINIGO_HAS_OPENCL

#include "opencl_compute.h"
#include "loaded_model.h"
#include <cmath>
#include <cstring>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <algorithm>

namespace minigo {

// ================================================================
// OpenCL kernel source (embedded as a string)
// ================================================================
static const char* OPENCL_KERNELS = R"CL(

// ================================================================
// Transpose:  NCHW → channel-major  [C, N*HW]
// input[n, c, hw]  →  output[c, n*HW + hw]
// ================================================================
__kernel void transpose_nchw_to_cnhw(
    __global const float* input,
    __global       float* output,
    const int N, const int C, const int HW
) {
    const int gid = get_global_id(0);
    const int total = N * C * HW;
    if (gid >= total) return;
    const int n  = gid / (C * HW);
    const int c  = (gid / HW) % C;
    const int hw = gid % HW;
    output[c * (N * HW) + n * HW + hw] = input[gid];
}

// ================================================================
// Implicit GEMM for 3×3 convolution + fused BN + optional ReLU
//
// Replaces the separate im2col + SGEMM two-kernel sequence with a
// single kernel that computes im2col indices on-the-fly during B-tile
// loading, so the 24 MB col buffer is never written to global memory.
// The 2.7 MB input tensor (C_in=64, NHW=10368 for batch=128, 9×9 board)
// fits in L2 cache, turning ~576 MB of GDDR7 traffic into L2 hits.
//
// Layout:
//   A (weight):  [C_out, K]     K = C_in * 9
//   B (virtual): im2col of input[C_in, NHW]  — computed on-the-fly
//   C (output):  [C_out, NHW]
//
// Tiling: same register-blocked scheme as the old sgemm_bn.
//   Work-group: TS × TS = 16 × 16 = 256 threads
//   Each thread computes WPT_M × WPT_N = 2 × 4 = 8 output elements
//   Tile covers TSM × TSN = 32 × 64 outputs per work-group
//
// mode: 0 = plain GEMM,  1 = BN + optional ReLU,
//       2 = BN + residual-add + ReLU
//
// Global work size: { ceil(C_out/TSM)*TS,  ceil(NHW/TSN)*TS }
// Local  work size: { TS, TS }
// ================================================================
#define TS    16
#define WPT_M  2
#define WPT_N  4
#define TSM   (TS * WPT_M)   // 32
#define TSN   (TS * WPT_N)   // 64

__kernel __attribute__((reqd_work_group_size(TS, TS, 1)))
void conv3x3_sgemm_bn(
    __global const float* A,        // weight [C_out, K], K = C_in*9
    __global const float* input,    // [C_in, NHW] channel-major
    __global       float* C,        // output [C_out, NHW]
    __global const float* bn_scale,
    __global const float* bn_bias,
    __global const float* residual,
    const int C_out, const int NHW, const int K,  // K = C_in * 9
    const int H, const int W,
    const int mode,                 // 0=plain, 1=BN+relu, 2=BN+residual+relu
    const int do_relu
) {
    __local float As[TSM][TS + 1];   // +1 avoids bank conflicts
    __local float Bs[TS][TSN + 1];

    const int lm = get_local_id(0);
    const int ln = get_local_id(1);
    const int gm_base = get_group_id(0) * TSM;
    const int gn_base = get_group_id(1) * TSN;

    if (gm_base >= C_out || gn_base >= NHW) return;

    // ── Precompute nhw decompositions for this thread's B columns ──
    const int HW = H * W;
    int b_nhw[WPT_N], b_n[WPT_N], b_oh[WPT_N], b_ow[WPT_N];
    for (int wn = 0; wn < WPT_N; wn++) {
        int nhw = gn_base + ln + wn * TS;
        b_nhw[wn] = nhw;
        if (nhw < NHW) {
            int n_   = nhw / HW;
            int hw_  = nhw % HW;
            b_n[wn]  = n_;
            b_oh[wn] = hw_ / W;
            b_ow[wn] = hw_ % W;
        }
    }

    // ── Register accumulators ──────────────────────────────────────
    float Creg[WPT_M][WPT_N];
    for (int wm = 0; wm < WPT_M; wm++)
        for (int wn = 0; wn < WPT_N; wn++)
            Creg[wm][wn] = 0.0f;

    // ── Tile loop over K = C_in * 9 ───────────────────────────────
    const int numTiles = (K + TS - 1) / TS;
    for (int t = 0; t < numTiles; t++) {
        const int k_off = t * TS;

        // Load A tile [TSM, TS]: weight[C_out, K] — standard row-major load
        for (int wm = 0; wm < WPT_M; wm++) {
            const int row = gm_base + lm + wm * TS;
            const int col = k_off + ln;
            As[lm + wm * TS][ln] = (row < C_out && col < K) ? A[row * K + col] : 0.0f;
        }

        // Load B tile [TS, TSN]: implicit im2col from input[C_in, NHW]
        {
            int k_row = k_off + lm;
            int c_in_k = -1, kh_k = -1, kw_k = -1;
            if (k_row < K) {
                int rem = k_row % 9;
                c_in_k  = k_row / 9;
                kh_k    = rem / 3;
                kw_k    = rem % 3;
            }
            for (int wn = 0; wn < WPT_N; wn++) {
                float val = 0.0f;
                if (k_row < K && b_nhw[wn] < NHW) {
                    int ih = b_oh[wn] + kh_k - 1;  // same-padding: pad=1
                    int iw = b_ow[wn] + kw_k - 1;
                    if (ih >= 0 && ih < H && iw >= 0 && iw < W)
                        val = input[c_in_k * NHW + b_n[wn] * HW + ih * W + iw];
                }
                Bs[lm][ln + wn * TS] = val;
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        // Compute: WPT_M × WPT_N register blocking
        for (int ki = 0; ki < TS; ki++) {
            float a[WPT_M], b[WPT_N];
            for (int wm = 0; wm < WPT_M; wm++) a[wm] = As[lm + wm * TS][ki];
            for (int wn = 0; wn < WPT_N; wn++) b[wn] = Bs[ki][ln + wn * TS];
            for (int wm = 0; wm < WPT_M; wm++)
                for (int wn = 0; wn < WPT_N; wn++)
                    Creg[wm][wn] += a[wm] * b[wn];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // ── Write back with optional BN / residual-add / ReLU ─────────
    for (int wm = 0; wm < WPT_M; wm++) {
        const int row = gm_base + lm + wm * TS;
        if (row >= C_out) continue;
        const float sc = (mode >= 1) ? bn_scale[row] : 1.0f;
        const float bi = (mode >= 1) ? bn_bias[row]  : 0.0f;
        for (int wn = 0; wn < WPT_N; wn++) {
            const int col = gn_base + ln + wn * TS;
            if (col >= NHW) continue;
            float v = sc * Creg[wm][wn] + bi;
            if (mode == 2) v += residual[row * NHW + col];
            if (do_relu && v < 0.0f) v = 0.0f;
            C[row * NHW + col] = v;
        }
    }
}

// ================================================================
// Fused 1×1 conv + BN + ReLU + reshape for FC
//
// input [C_in, N*HW]  channel-major
// output[C_out*HW, N] FC-ready layout (reshaped)
// ================================================================
__kernel void conv1x1_bn_relu_reshape(
    __global const float* input,
    __global       float* output,
    __global const float* weight,
    __global const float* bn_scale,
    __global const float* bn_bias,
    const int C_in, const int C_out,
    const int N, const int HW
) {
    const int n  = get_global_id(0);
    const int hw = get_global_id(1);
    if (n >= N || hw >= HW) return;

    const int NHW = N * HW;

    for (int co = 0; co < C_out; co++) {
        float acc = 0.0f;
        for (int ci = 0; ci < C_in; ci++) {
            acc += weight[co * C_in + ci] * input[ci * NHW + n * HW + hw];
        }
        float v = bn_scale[co] * acc + bn_bias[co];
        if (v < 0.0f) v = 0.0f;
        output[(co * HW + hw) * N + n] = v;
    }
}

// ================================================================
// Fused FC + bias + optional ReLU
//
// output[M, N] = weight[M, K] × input[K, N] + bias[M]
// ================================================================
__kernel void fc_bias_relu(
    __global const float* weight,
    __global const float* input,
    __global       float* output,
    __global const float* bias,
    const int M, const int N, const int K,
    const int do_relu
) {
    const int m = get_global_id(0);
    const int n = get_global_id(1);
    if (m >= M || n >= N) return;

    float acc = 0.0f;
    for (int k = 0; k < K; k++)
        acc += weight[m * K + k] * input[k * N + n];
    acc += bias[m];
    if (do_relu && acc < 0.0f) acc = 0.0f;
    output[m * N + n] = acc;
}

// ================================================================
// Fused FC + bias + softmax   (for policy head final stage)
// ================================================================
__kernel void fc_bias_softmax(
    __global const float* weight,
    __global const float* input,
    __global       float* output,
    __global const float* bias,
    const int M, const int N, const int K
) {
    const int n = get_global_id(0);
    if (n >= N) return;

    float mx = -1e30f;
    for (int m = 0; m < M; m++) {
        float acc = bias[m];
        for (int k = 0; k < K; k++)
            acc += weight[m * K + k] * input[k * N + n];
        output[m * N + n] = acc;
        mx = fmax(mx, acc);
    }

    float sum = 0.0f;
    for (int m = 0; m < M; m++) {
        float e = exp(output[m * N + n] - mx);
        output[m * N + n] = e;
        sum += e;
    }

    float inv_sum = 1.0f / sum;
    for (int m = 0; m < M; m++)
        output[m * N + n] *= inv_sum;
}

// ================================================================
// Fused FC + bias + tanh  (for value head final stage: M=1)
// ================================================================
__kernel void fc_bias_tanh(
    __global const float* weight,
    __global const float* input,
    __global       float* output,
    __global const float* bias,
    const int N, const int K
) {
    const int n = get_global_id(0);
    if (n >= N) return;

    float acc = bias[0];
    for (int k = 0; k < K; k++)
        acc += weight[k] * input[k * N + n];
    output[n] = tanh(acc);
}

)CL";

// ================================================================
// Helper macros
// ================================================================
#define CL_CHECK(expr)                                                        \
    do {                                                                      \
        cl_int _err = (expr);                                                 \
        if (_err != CL_SUCCESS) {                                             \
            std::ostringstream _os;                                           \
            _os << "OpenCL error " << _err << " at " << __FILE__             \
                << ":" << __LINE__;                                           \
            throw std::runtime_error(_os.str());                              \
        }                                                                     \
    } while (0)

static cl_mem new_buf(cl_context ctx, size_t bytes, cl_int* err) {
    return clCreateBuffer(ctx, CL_MEM_READ_WRITE, bytes, nullptr, err);
}

static void release_buf(cl_mem& buf) {
    if (buf) { clReleaseMemObject(buf); buf = nullptr; }
}

static size_t round_up(size_t n, size_t tile) {
    return ((n + tile - 1) / tile) * tile;
}

// ================================================================
// OpenCLComputeContext — one cl_context + cl_queue + cl_program per
// unique GPU device (KataGo pattern: avoids NVIDIA serialization)
// ================================================================

static void init_device(OpenCLDeviceState& ds, int device_id) {
    cl_uint num_platforms = 0;
    CL_CHECK(clGetPlatformIDs(0, nullptr, &num_platforms));
    if (num_platforms == 0)
        throw std::runtime_error("No OpenCL platforms found");

    std::vector<cl_platform_id> platforms(num_platforms);
    CL_CHECK(clGetPlatformIDs(num_platforms, platforms.data(), nullptr));

    // Collect all GPU devices across all platforms
    std::vector<std::pair<cl_platform_id, cl_device_id>> all_gpus;
    for (auto plat : platforms) {
        cl_uint nd = 0;
        if (clGetDeviceIDs(plat, CL_DEVICE_TYPE_GPU, 0, nullptr, &nd) != CL_SUCCESS || nd == 0)
            continue;
        std::vector<cl_device_id> devs(nd);
        if (clGetDeviceIDs(plat, CL_DEVICE_TYPE_GPU, nd, devs.data(), nullptr) != CL_SUCCESS)
            continue;
        for (auto d : devs)
            all_gpus.push_back({plat, d});
    }

    // Fall back to any device if no GPUs found
    if (all_gpus.empty()) {
        for (auto plat : platforms) {
            cl_uint nd = 0;
            if (clGetDeviceIDs(plat, CL_DEVICE_TYPE_ALL, 0, nullptr, &nd) != CL_SUCCESS || nd == 0)
                continue;
            std::vector<cl_device_id> devs(nd);
            if (clGetDeviceIDs(plat, CL_DEVICE_TYPE_ALL, nd, devs.data(), nullptr) != CL_SUCCESS)
                continue;
            for (auto d : devs)
                all_gpus.push_back({plat, d});
        }
    }

    if (all_gpus.empty())
        throw std::runtime_error("No OpenCL devices found");

    if (device_id < 0 || device_id >= (int)all_gpus.size())
        throw std::runtime_error("OpenCL device_id " + std::to_string(device_id) +
                                 " out of range [0, " + std::to_string(all_gpus.size()) + ")");

    ds.platform = all_gpus[device_id].first;
    ds.device   = all_gpus[device_id].second;

    char name[256] = {};
    clGetDeviceInfo(ds.device, CL_DEVICE_NAME, sizeof(name), name, nullptr);
    std::cout << "OpenCL device " << device_id << ": " << name << "\n";

    cl_int err;
    ds.context = clCreateContext(nullptr, 1, &ds.device, nullptr, nullptr, &err);
    CL_CHECK(err);

#ifdef CL_VERSION_2_0
    cl_queue_properties props[] = { 0 };
    ds.queue = clCreateCommandQueueWithProperties(ds.context, ds.device, props, &err);
#else
    ds.queue = clCreateCommandQueue(ds.context, ds.device, 0, &err);
#endif
    CL_CHECK(err);
}

static void compile_kernels(OpenCLDeviceState& ds) {
    cl_int err;
    ds.program = clCreateProgramWithSource(ds.context, 1, &OPENCL_KERNELS,
                                           nullptr, &err);
    CL_CHECK(err);

    err = clBuildProgram(ds.program, 1, &ds.device, "-cl-mad-enable", nullptr, nullptr);
    if (err != CL_SUCCESS) {
        size_t log_size = 0;
        clGetProgramBuildInfo(ds.program, ds.device, CL_PROGRAM_BUILD_LOG,
                              0, nullptr, &log_size);
        std::string log(log_size, ' ');
        clGetProgramBuildInfo(ds.program, ds.device, CL_PROGRAM_BUILD_LOG,
                              log_size, &log[0], nullptr);
        throw std::runtime_error("OpenCL build error:\n" + log);
    }
}

OpenCLComputeContext::OpenCLComputeContext(const std::vector<int>& device_ids) {
    for (int id : device_ids) {
        if (devices_.count(id)) continue;  // already initialized this GPU
        auto& ds = devices_[id];
        init_device(ds, id);
        compile_kernels(ds);
    }
}

OpenCLComputeContext::~OpenCLComputeContext() {
    for (auto& [id, ds] : devices_) {
        if (ds.program) clReleaseProgram(ds.program);
        if (ds.queue)   clReleaseCommandQueue(ds.queue);
        if (ds.context) clReleaseContext(ds.context);
    }
}

OpenCLDeviceState& OpenCLComputeContext::device_state(int gpu_id) {
    auto it = devices_.find(gpu_id);
    if (it == devices_.end())
        throw std::runtime_error("OpenCL device " + std::to_string(gpu_id) + " not initialized");
    return it->second;
}

std::unique_ptr<ComputeHandle>
OpenCLComputeContext::create_handle(const LoadedModel* model, int gpu_id, int max_batch_size) {
    return std::make_unique<OpenCLComputeHandle>(device_state(gpu_id), model, max_batch_size);
}

// ================================================================
// OpenCLComputeHandle — per-server-thread GPU state
//
// Created ON the server thread.  Uploads weights from LoadedModel
// CPU data → cl_mem buffers.  Owns workspace buffers.
// ================================================================

OpenCLComputeHandle::OpenCLComputeHandle(OpenCLDeviceState& dev,
                                         const LoadedModel* model,
                                         int max_batch_size)
    : dev_(dev)
{
    board_size     = model->board_size;
    input_channels = model->input_channels;
    num_filters    = model->num_filters;
    num_res_blocks = model->num_res_blocks;

    // Create kernel handles from the shared compiled program
    cl_int err;
    auto mk = [&](const char* name) -> cl_kernel {
        cl_kernel k = clCreateKernel(dev_.program, name, &err);
        CL_CHECK(err);
        return k;
    };
    k_transpose_nchw_          = mk("transpose_nchw_to_cnhw");
    k_conv3x3_sgemm_bn_        = mk("conv3x3_sgemm_bn");
    k_conv1x1_bn_relu_reshape_ = mk("conv1x1_bn_relu_reshape");
    k_fc_bias_relu_            = mk("fc_bias_relu");
    k_fc_bias_softmax_         = mk("fc_bias_softmax");
    k_fc_bias_tanh_            = mk("fc_bias_tanh");

    // Upload weights from LoadedModel CPU data → GPU buffers
    auto upload_conv = [&](ConvBNGPU& g, const ConvBNWeights& src) {
        g.c_out = src.c_out;
        g.c_in  = src.c_in;
        g.k     = src.k;
        g.weight   = upload(src.weight);
        g.bn_scale = upload(src.bn_scale);
        g.bn_bias  = upload(src.bn_bias);
    };

    auto upload_fc = [&](FCGPU& g, const FCWeights& src) {
        g.out_features = src.out_features;
        g.in_features  = src.in_features;
        g.weight = upload(src.weight);
        g.bias   = upload(src.bias);
    };

    upload_conv(input_conv_gpu_, model->input_conv);

    res_conv1_gpu_.resize(num_res_blocks);
    res_conv2_gpu_.resize(num_res_blocks);
    for (int i = 0; i < num_res_blocks; i++) {
        upload_conv(res_conv1_gpu_[i], model->res_conv1[i]);
        upload_conv(res_conv2_gpu_[i], model->res_conv2[i]);
    }

    upload_conv(policy_conv_gpu_, model->policy_conv);
    upload_conv(value_conv_gpu_,  model->value_conv);
    upload_fc(policy_fc_gpu_, model->policy_fc);
    upload_fc(value_fc1_gpu_, model->value_fc1);
    upload_fc(value_fc2_gpu_, model->value_fc2);

    upload_conv(score_conv_gpu_, model->score_conv);
    upload_fc(score_fc1_gpu_, model->score_fc1);
    upload_fc(score_fc2_gpu_, model->score_fc2);

    // Pre-allocate workspace
    allocate_workspace(max_batch_size > 0 ? max_batch_size : 32);

    std::cout << "OpenCL handle ready: board=" << board_size
              << " filters=" << num_filters
              << " blocks="  << num_res_blocks << "\n";
}

OpenCLComputeHandle::~OpenCLComputeHandle() {
    free_workspace();
    free_weights();
    if (k_transpose_nchw_)          clReleaseKernel(k_transpose_nchw_);
    if (k_conv3x3_sgemm_bn_)        clReleaseKernel(k_conv3x3_sgemm_bn_);
    if (k_conv1x1_bn_relu_reshape_) clReleaseKernel(k_conv1x1_bn_relu_reshape_);
    if (k_fc_bias_relu_)            clReleaseKernel(k_fc_bias_relu_);
    if (k_fc_bias_softmax_)         clReleaseKernel(k_fc_bias_softmax_);
    if (k_fc_bias_tanh_)            clReleaseKernel(k_fc_bias_tanh_);
}

// ================================================================
// Weight upload / free helpers
// ================================================================

cl_mem OpenCLComputeHandle::upload(const std::vector<float>& data) {
    cl_int err;
    cl_mem buf = clCreateBuffer(dev_.context,
                                CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                data.size() * sizeof(float),
                                (void*)data.data(), &err);
    CL_CHECK(err);
    return buf;
}

void OpenCLComputeHandle::release_buf(cl_mem& buf) {
    if (buf) { clReleaseMemObject(buf); buf = nullptr; }
}

void OpenCLComputeHandle::free_weights() {
    auto free_conv = [this](ConvBNGPU& c) {
        release_buf(c.weight);
        release_buf(c.bn_scale);
        release_buf(c.bn_bias);
    };
    auto free_fc = [this](FCGPU& f) {
        release_buf(f.weight);
        release_buf(f.bias);
    };
    free_conv(input_conv_gpu_);
    for (auto& c : res_conv1_gpu_) free_conv(c);
    for (auto& c : res_conv2_gpu_) free_conv(c);
    free_conv(policy_conv_gpu_);
    free_conv(value_conv_gpu_);
    free_fc(policy_fc_gpu_);
    free_fc(value_fc1_gpu_);
    free_fc(value_fc2_gpu_);
    free_conv(score_conv_gpu_);
    free_fc(score_fc1_gpu_);
    free_fc(score_fc2_gpu_);
    res_conv1_gpu_.clear();
    res_conv2_gpu_.clear();
}

// ================================================================
// Workspace allocation
// ================================================================

void OpenCLComputeHandle::allocate_workspace(int batch) {
    if (batch <= alloc_batch_) return;
    free_workspace();

    int H = board_size, W = board_size;
    int NHW   = batch * H * W;
    int filt  = num_filters;
    int inch  = input_channels;
    int as    = H * W + 1;  // action_size

    cl_int err;
    auto alloc = [&](size_t floats) -> cl_mem {
        cl_mem b = new_buf(dev_.context, floats * sizeof(float), &err);
        CL_CHECK(err);
        return b;
    };

    buf_flat_in_  = alloc((size_t)inch * NHW);
    buf_input_    = alloc((size_t)inch * NHW);
    buf_main_     = alloc((size_t)filt * NHW);
    buf_temp_     = alloc((size_t)filt * NHW);
    buf_skip_     = alloc((size_t)filt * NHW);

    buf_pol_out_  = alloc((size_t)2 * H * W * batch);
    buf_pol_feat_ = alloc((size_t)as * batch);
    buf_val_h1_   = alloc((size_t)H * W * batch);
    buf_val_feat_ = alloc((size_t)filt * batch);
    buf_val_out_  = alloc((size_t)batch);

    buf_scr_h1_   = alloc((size_t)H * W * batch);
    buf_scr_feat_ = alloc((size_t)filt * batch);
    buf_scr_out_  = alloc((size_t)batch);

    alloc_batch_ = batch;
}

void OpenCLComputeHandle::free_workspace() {
    release_buf(buf_flat_in_);
    release_buf(buf_input_);
    release_buf(buf_main_);
    release_buf(buf_temp_);
    release_buf(buf_skip_);
    release_buf(buf_pol_feat_);
    release_buf(buf_pol_out_);
    release_buf(buf_val_feat_);
    release_buf(buf_val_h1_);
    release_buf(buf_val_out_);
    release_buf(buf_scr_h1_);
    release_buf(buf_scr_feat_);
    release_buf(buf_scr_out_);
    alloc_batch_ = 0;
}

// ================================================================
// Kernel launch helpers
// ================================================================

void OpenCLComputeHandle::run_conv3x3(cl_mem input_buf, cl_mem output_buf,
                                       const ConvBNGPU& conv, cl_mem residual_buf,
                                       int N, int H, int W, int mode, bool relu) {
    int NHW   = N * H * W;
    int C_in  = conv.c_in, C_out = conv.c_out;
    int K     = C_in * 9;
    int do_relu_i = relu ? 1 : 0;

    cl_mem res_arg = (mode == 2 && residual_buf) ? residual_buf : output_buf;

    size_t gsX = ((size_t)(C_out + 31) / 32) * 16;
    size_t gsY = ((size_t)(NHW   + 63) / 64) * 16;
    size_t gs2[2] = { gsX, gsY };
    size_t ls2[2] = { 16, 16 };

    CL_CHECK(clSetKernelArg(k_conv3x3_sgemm_bn_,  0, sizeof(cl_mem), &conv.weight));
    CL_CHECK(clSetKernelArg(k_conv3x3_sgemm_bn_,  1, sizeof(cl_mem), &input_buf));
    CL_CHECK(clSetKernelArg(k_conv3x3_sgemm_bn_,  2, sizeof(cl_mem), &output_buf));
    CL_CHECK(clSetKernelArg(k_conv3x3_sgemm_bn_,  3, sizeof(cl_mem), &conv.bn_scale));
    CL_CHECK(clSetKernelArg(k_conv3x3_sgemm_bn_,  4, sizeof(cl_mem), &conv.bn_bias));
    CL_CHECK(clSetKernelArg(k_conv3x3_sgemm_bn_,  5, sizeof(cl_mem), &res_arg));
    CL_CHECK(clSetKernelArg(k_conv3x3_sgemm_bn_,  6, sizeof(int),    &C_out));
    CL_CHECK(clSetKernelArg(k_conv3x3_sgemm_bn_,  7, sizeof(int),    &NHW));
    CL_CHECK(clSetKernelArg(k_conv3x3_sgemm_bn_,  8, sizeof(int),    &K));
    CL_CHECK(clSetKernelArg(k_conv3x3_sgemm_bn_,  9, sizeof(int),    &H));
    CL_CHECK(clSetKernelArg(k_conv3x3_sgemm_bn_, 10, sizeof(int),    &W));
    CL_CHECK(clSetKernelArg(k_conv3x3_sgemm_bn_, 11, sizeof(int),    &mode));
    CL_CHECK(clSetKernelArg(k_conv3x3_sgemm_bn_, 12, sizeof(int),    &do_relu_i));
    CL_CHECK(clEnqueueNDRangeKernel(dev_.queue, k_conv3x3_sgemm_bn_, 2,
                                    nullptr, gs2, ls2, 0, nullptr, nullptr));
}

void OpenCLComputeHandle::run_conv1x1_bn_relu_reshape(cl_mem input_buf, cl_mem output_buf,
                                                       const ConvBNGPU& conv,
                                                       int N, int HW) {
    int C_in  = conv.c_in;
    int C_out = conv.c_out;

    CL_CHECK(clSetKernelArg(k_conv1x1_bn_relu_reshape_, 0, sizeof(cl_mem), &input_buf));
    CL_CHECK(clSetKernelArg(k_conv1x1_bn_relu_reshape_, 1, sizeof(cl_mem), &output_buf));
    CL_CHECK(clSetKernelArg(k_conv1x1_bn_relu_reshape_, 2, sizeof(cl_mem), &conv.weight));
    CL_CHECK(clSetKernelArg(k_conv1x1_bn_relu_reshape_, 3, sizeof(cl_mem), &conv.bn_scale));
    CL_CHECK(clSetKernelArg(k_conv1x1_bn_relu_reshape_, 4, sizeof(cl_mem), &conv.bn_bias));
    CL_CHECK(clSetKernelArg(k_conv1x1_bn_relu_reshape_, 5, sizeof(int), &C_in));
    CL_CHECK(clSetKernelArg(k_conv1x1_bn_relu_reshape_, 6, sizeof(int), &C_out));
    CL_CHECK(clSetKernelArg(k_conv1x1_bn_relu_reshape_, 7, sizeof(int), &N));
    CL_CHECK(clSetKernelArg(k_conv1x1_bn_relu_reshape_, 8, sizeof(int), &HW));

    size_t gs[2] = { round_up((size_t)N,  16),
                     round_up((size_t)HW, 16) };
    CL_CHECK(clEnqueueNDRangeKernel(dev_.queue, k_conv1x1_bn_relu_reshape_, 2,
                                    nullptr, gs, nullptr, 0, nullptr, nullptr));
}

void OpenCLComputeHandle::run_fc_bias_relu(cl_mem input_buf, cl_mem output_buf,
                                            const FCGPU& fc, int N, bool relu) {
    int M  = fc.out_features;
    int K  = fc.in_features;
    int do_relu_i = relu ? 1 : 0;

    CL_CHECK(clSetKernelArg(k_fc_bias_relu_, 0, sizeof(cl_mem), &fc.weight));
    CL_CHECK(clSetKernelArg(k_fc_bias_relu_, 1, sizeof(cl_mem), &input_buf));
    CL_CHECK(clSetKernelArg(k_fc_bias_relu_, 2, sizeof(cl_mem), &output_buf));
    CL_CHECK(clSetKernelArg(k_fc_bias_relu_, 3, sizeof(cl_mem), &fc.bias));
    CL_CHECK(clSetKernelArg(k_fc_bias_relu_, 4, sizeof(int), &M));
    CL_CHECK(clSetKernelArg(k_fc_bias_relu_, 5, sizeof(int), &N));
    CL_CHECK(clSetKernelArg(k_fc_bias_relu_, 6, sizeof(int), &K));
    CL_CHECK(clSetKernelArg(k_fc_bias_relu_, 7, sizeof(int), &do_relu_i));

    size_t gs[2] = { round_up((size_t)M, 16),
                     round_up((size_t)N, 16) };
    CL_CHECK(clEnqueueNDRangeKernel(dev_.queue, k_fc_bias_relu_, 2,
                                    nullptr, gs, nullptr, 0, nullptr, nullptr));
}

// ================================================================
// Forward pass (batched)
// ================================================================

std::vector<OpenCLComputeHandle::Result>
OpenCLComputeHandle::predict_batch(const std::vector<std::vector<float>>& states) {
    if (states.empty()) return {};

    int N = (int)states.size();
    int H = board_size, W = board_size;
    int HW = H * W;
    int NHW = N * HW;
    int action_size = HW + 1;

    allocate_workspace(N);

    // ── Upload inputs [N, C, HW] flat, then transpose on GPU ─────
    size_t input_floats = (size_t)N * input_channels * HW;
    std::vector<float> flat_input;
    flat_input.reserve(input_floats);
    for (auto& s : states)
        flat_input.insert(flat_input.end(), s.begin(), s.end());

    CL_CHECK(clEnqueueWriteBuffer(dev_.queue, buf_flat_in_, CL_FALSE, 0,
        input_floats * sizeof(float), flat_input.data(),
        0, nullptr, nullptr));

    // GPU transpose: [N, C, HW] → [C, N*HW]
    {
        int C = input_channels;
        size_t total = input_floats;
        size_t gs = round_up(total, 256);
        CL_CHECK(clSetKernelArg(k_transpose_nchw_, 0, sizeof(cl_mem), &buf_flat_in_));
        CL_CHECK(clSetKernelArg(k_transpose_nchw_, 1, sizeof(cl_mem), &buf_input_));
        CL_CHECK(clSetKernelArg(k_transpose_nchw_, 2, sizeof(int), &N));
        CL_CHECK(clSetKernelArg(k_transpose_nchw_, 3, sizeof(int), &C));
        CL_CHECK(clSetKernelArg(k_transpose_nchw_, 4, sizeof(int), &HW));
        CL_CHECK(clEnqueueNDRangeKernel(dev_.queue, k_transpose_nchw_, 1,
                                        nullptr, &gs, nullptr, 0, nullptr, nullptr));
    }

    // ── Input conv: in_channels → num_filters, 3×3 + BN + ReLU ──
    run_conv3x3(buf_input_, buf_main_,
                input_conv_gpu_, nullptr,
                N, H, W, /*mode=*/1, /*relu=*/true);

    // ── Residual blocks ──────────────────────────────────────────
    for (int i = 0; i < num_res_blocks; i++) {
        run_conv3x3(buf_main_, buf_temp_,
                    res_conv1_gpu_[i], nullptr,
                    N, H, W, /*mode=*/1, /*relu=*/true);

        run_conv3x3(buf_temp_, buf_skip_,
                    res_conv2_gpu_[i], buf_main_,
                    N, H, W, /*mode=*/2, /*relu=*/true);

        std::swap(buf_main_, buf_skip_);
    }

    // ── Policy head ──────────────────────────────────────────────
    run_conv1x1_bn_relu_reshape(buf_main_, buf_pol_out_,
                                policy_conv_gpu_, N, HW);

    {
        int M = policy_fc_gpu_.out_features;
        int K = policy_fc_gpu_.in_features;
        size_t gs = round_up((size_t)N, 64);
        CL_CHECK(clSetKernelArg(k_fc_bias_softmax_, 0, sizeof(cl_mem), &policy_fc_gpu_.weight));
        CL_CHECK(clSetKernelArg(k_fc_bias_softmax_, 1, sizeof(cl_mem), &buf_pol_out_));
        CL_CHECK(clSetKernelArg(k_fc_bias_softmax_, 2, sizeof(cl_mem), &buf_pol_feat_));
        CL_CHECK(clSetKernelArg(k_fc_bias_softmax_, 3, sizeof(cl_mem), &policy_fc_gpu_.bias));
        CL_CHECK(clSetKernelArg(k_fc_bias_softmax_, 4, sizeof(int), &M));
        CL_CHECK(clSetKernelArg(k_fc_bias_softmax_, 5, sizeof(int), &N));
        CL_CHECK(clSetKernelArg(k_fc_bias_softmax_, 6, sizeof(int), &K));
        CL_CHECK(clEnqueueNDRangeKernel(dev_.queue, k_fc_bias_softmax_, 1,
                                        nullptr, &gs, nullptr, 0, nullptr, nullptr));
    }

    // ── Value head ───────────────────────────────────────────────
    run_conv1x1_bn_relu_reshape(buf_main_, buf_val_h1_,
                                value_conv_gpu_, N, HW);

    run_fc_bias_relu(buf_val_h1_, buf_val_feat_,
                     value_fc1_gpu_, N, /*relu=*/true);

    {
        int K = value_fc2_gpu_.in_features;
        size_t gs = round_up((size_t)N, 64);
        CL_CHECK(clSetKernelArg(k_fc_bias_tanh_, 0, sizeof(cl_mem), &value_fc2_gpu_.weight));
        CL_CHECK(clSetKernelArg(k_fc_bias_tanh_, 1, sizeof(cl_mem), &buf_val_feat_));
        CL_CHECK(clSetKernelArg(k_fc_bias_tanh_, 2, sizeof(cl_mem), &buf_val_out_));
        CL_CHECK(clSetKernelArg(k_fc_bias_tanh_, 3, sizeof(cl_mem), &value_fc2_gpu_.bias));
        CL_CHECK(clSetKernelArg(k_fc_bias_tanh_, 4, sizeof(int), &N));
        CL_CHECK(clSetKernelArg(k_fc_bias_tanh_, 5, sizeof(int), &K));
        CL_CHECK(clEnqueueNDRangeKernel(dev_.queue, k_fc_bias_tanh_, 1,
                                        nullptr, &gs, nullptr, 0, nullptr, nullptr));
    }

    // ── Score head ────────────────────────────────────────────────
    run_conv1x1_bn_relu_reshape(buf_main_, buf_scr_h1_,
                                score_conv_gpu_, N, HW);
    run_fc_bias_relu(buf_scr_h1_, buf_scr_feat_,
                     score_fc1_gpu_, N, /*relu=*/true);
    {
        int K = score_fc2_gpu_.in_features;
        size_t gs = round_up((size_t)N, 64);
        CL_CHECK(clSetKernelArg(k_fc_bias_tanh_, 0, sizeof(cl_mem), &score_fc2_gpu_.weight));
        CL_CHECK(clSetKernelArg(k_fc_bias_tanh_, 1, sizeof(cl_mem), &buf_scr_feat_));
        CL_CHECK(clSetKernelArg(k_fc_bias_tanh_, 2, sizeof(cl_mem), &buf_scr_out_));
        CL_CHECK(clSetKernelArg(k_fc_bias_tanh_, 3, sizeof(cl_mem), &score_fc2_gpu_.bias));
        CL_CHECK(clSetKernelArg(k_fc_bias_tanh_, 4, sizeof(int), &N));
        CL_CHECK(clSetKernelArg(k_fc_bias_tanh_, 5, sizeof(int), &K));
        CL_CHECK(clEnqueueNDRangeKernel(dev_.queue, k_fc_bias_tanh_, 1,
                                        nullptr, &gs, nullptr, 0, nullptr, nullptr));
    }

    // ── Read back results ────────────────────────────────────────
    std::vector<float> pol_flat((size_t)action_size * N);
    CL_CHECK(clEnqueueReadBuffer(dev_.queue, buf_pol_feat_, CL_FALSE, 0,
        pol_flat.size() * sizeof(float), pol_flat.data(), 0, nullptr, nullptr));

    std::vector<float> val_flat((size_t)N);
    CL_CHECK(clEnqueueReadBuffer(dev_.queue, buf_val_out_, CL_FALSE, 0,
        val_flat.size() * sizeof(float), val_flat.data(), 0, nullptr, nullptr));

    std::vector<float> scr_flat((size_t)N);
    CL_CHECK(clEnqueueReadBuffer(dev_.queue, buf_scr_out_, CL_FALSE, 0,
        scr_flat.size() * sizeof(float), scr_flat.data(), 0, nullptr, nullptr));

    CL_CHECK(clFinish(dev_.queue));

    // ── Pack results ─────────────────────────────────────────────
    std::vector<OpenCLComputeHandle::Result> results(N);
    for (int n = 0; n < N; n++) {
        std::vector<float> pol(action_size);
        for (int a = 0; a < action_size; a++)
            pol[a] = pol_flat[a * N + n];
        results[n].policy = std::move(pol);
        results[n].value  = val_flat[n];
        results[n].score  = scr_flat[n];
    }
    return results;
}

}  // namespace minigo
#endif  // MINIGO_HAS_OPENCL
