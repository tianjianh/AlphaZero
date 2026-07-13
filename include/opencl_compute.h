#pragma once
#ifdef MINIGO_HAS_OPENCL

#ifdef __APPLE__
#  include <OpenCL/cl.h>
#else
#  include <CL/cl.h>
#endif

#include "compute_context.h"
#include "loaded_model.h"
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace minigo {

// Precision tier, chosen per device at context init.
//   FP32     — float storage + float math; any OpenCL 1.2 device.
//   FP16     — half storage (vload_half/vstore_half, core CL) + float
//              math; any OpenCL 1.2 device, halves bandwidth.
//   FP16_MMA — FP16 plus NVIDIA tensor-core GEMMs via inline-PTX
//              mma.sync (f16 inputs, f32 accumulate); probed at init.
// Env override: MINIGO_OPENCL_PRECISION = fp32 | fp16 | auto (default).
enum class OpenCLPrecision { FP32, FP16, FP16_MMA };

// Per-unique-GPU device state (shared by all threads on that GPU).
// Deliberately holds NO cl_command_queue: an in-order command queue is
// per-thread execution state (the OpenCL analogue of a CUDA stream), so
// each ComputeHandle creates its own.  (KataGo likewise gives every
// ComputeHandle its own queue — cpp/neuralnet/openclbackend.cpp.)
struct OpenCLDeviceState {
    cl_platform_id  platform = nullptr;
    cl_device_id    device   = nullptr;
    cl_context      context  = nullptr;
    cl_program      program      = nullptr;   // portable kernels (tier-typed)
    cl_program      program_mma  = nullptr;   // NVIDIA tensor-core gemm (or null)
    OpenCLPrecision precision = OpenCLPrecision::FP32;
    std::string     name;
};

class OpenCLComputeContext : public ComputeContext {
public:
    // Initialize one cl_context + compiled programs per unique GPU.
    explicit OpenCLComputeContext(const std::vector<int>& device_ids);
    ~OpenCLComputeContext();

    std::unique_ptr<ComputeHandle>
    create_handle(const LoadedModel* model, int gpu_id, int max_batch_size) override;

    std::string backend_name() const override { return "opencl"; }

    OpenCLDeviceState& device_state(int gpu_id);

private:
    std::map<int, OpenCLDeviceState> devices_;  // gpu_id → device state
};

class OpenCLComputeHandle : public ComputeHandle {
public:
    OpenCLComputeHandle(OpenCLDeviceState& dev, const LoadedModel* model,
                        int max_batch_size);
    ~OpenCLComputeHandle();

    std::vector<Result>
    predict_batch(const std::vector<std::vector<float>>& states) override;

private:
    // ── GPU-side mirrors of the LoadedModel weight structs ─────
    struct GLayer {                     // conv or FC weight (+ optional bias)
        cl_mem w = nullptr;             // XT [out, in*k*k]
        cl_mem b = nullptr;             // fp32 [out] (row-bias) or null
        int out = 0, in = 0, k = 1;
        bool has_b = false;
    };
    struct GBN {                        // pre-fused BN
        cl_mem scale = nullptr, bias = nullptr;
        int ch = 0;
    };
    struct GGHead { GLayer conv; GBN bn; GLayer fc1, fc2; };
    struct GBlock {
        int kind = 0;                   // TrunkBlockW::Kind
        GLayer conv1, conv2;  GBN bn1, bn2;          // SE / (MgGPool conv2+bn2)
        GLayer se_fc1, se_fc2;
        GLayer conv_main, conv_pool, pool_fc;  GBN bn_main, bn_pool;
        GLayer regular, final, gpool, g2b;     GBN pre, mid, gp_bn;
        int pc = 0;
    };
    struct GViTBlock {
        cl_mem ln1_g = nullptr, ln1_b = nullptr, ln2_g = nullptr, ln2_b = nullptr;
        GLayer qkv, out_proj, mlp1, mlp2;
    };

    // ── A workspace buffer with its geometry ───────────────────
    struct Buf {
        cl_mem mem = nullptr;
        int rows = 0, ld = 0;           // elements; storage type = tier XT
    };

    OpenCLDeviceState& dev_;
    const LoadedModel* model_;
    cl_command_queue queue_ = nullptr;
    bool fp16_ = false;                 // storage is half
    bool mma_  = false;                 // tensor-core gemm available
    bool profile_ = false;              // MINIGO_OPENCL_PROFILE=1
    std::vector<std::pair<std::string, cl_event>> events_;

    // Kernels (from the tier's portable program; k_gemm_mma_ separate)
    cl_kernel k_xpose_ = nullptr, k_gemm_ = nullptr, k_gemm_mma_ = nullptr;
    cl_kernel k_bnact_ = nullptr, k_resadd_ = nullptr, k_stats_ = nullptr;
    cl_kernel k_ln_ = nullptr, k_pos_ = nullptr, k_att_ = nullptr;
    cl_kernel k_meantok_ = nullptr, k_flat_ = nullptr;
    cl_kernel k_pack_mg_ = nullptr, k_pack_sp_ = nullptr;

    // ── Weights on GPU ──────────────────────────────────────────
    std::vector<cl_mem> owned_;         // everything to release
    // MiniGo trunk
    GLayer input_conv_; GBN input_bn_;
    // Kata trunk
    GLayer stem_conv_, stem_global_; GBN tip_bn_;
    std::vector<GBlock> blocks_;
    // MiniGo heads
    GLayer policy_conv_; GBN policy_bn_; GLayer policy_fc_;
    GGHead value_head_, score_mean_head_, score_stdev_head_;
    GLayer ownership_conv_;
    // Kata1 heads
    GLayer p1_conv_, g1_conv_, p2_conv_; GBN g1_bn_, p1_bn_;
    GLayer k_g2b_, k_g2pass_;
    GLayer v1_conv_; GBN v1_bn_;
    GLayer v2_mul_, v3_mul_, sv3_mul_;
    GLayer vown_conv_;
    // ViT
    GLayer vt_token_; cl_mem vt_row_ = nullptr, vt_col_ = nullptr, vt_rel_ = nullptr;
    std::vector<GViTBlock> vt_blocks_;
    cl_mem vt_fin_g_ = nullptr, vt_fin_b_ = nullptr;
    GLayer vt_policy_, vt_val1_, vt_val2_, vt_sm1_, vt_sm2_, vt_sd1_, vt_sd2_, vt_own_;

    // ── Workspace ───────────────────────────────────────────────
    cl_mem staging_ = nullptr;          // fp32 [maxB, state_len]
    cl_mem out_pack_ = nullptr;         // fp32 [maxB, out_stride]
    Buf x_in_, g_in_;
    Buf buf_a_, buf_b_, buf_c_, buf_d_, buf_e_;
    Buf stats_, cb_, o_mlp_, o_pol_, flat_;
    Buf o_v_, o_sm_, o_sd_, o_sv3_, o_pass_, o_own_, p_sp_;
    std::vector<float> staging_host_;

    // Geometry
    int max_batch_ = 0, board_ = 0, hw_ = 0, state_len_ = 0, out_stride_ = 0;
    int ldT_ = 0, ldN_ = 0;             // leading dims for [.., B*hw] / [.., B]

    // ── Helpers ─────────────────────────────────────────────────
    cl_mem up_raw(const void* data, size_t bytes);
    cl_mem up_f32(const std::vector<float>& v);
    cl_mem up_xt(const std::vector<float>& v);       // tier storage type
    GLayer up_conv(const ConvW& c);
    GLayer up_fc(const FCW& f);
    GLayer up_fc_bias(const FCW& f, const std::vector<float>& bias);
    GBN    up_bn(const BNParamsW& b);
    GGHead up_ghead(const GPoolHeadW& h);
    Buf    make_buf(int rows, int ld);

    void upload_weights();
    void alloc_workspace();

    // Kernel enqueue helpers
    void enqueue(const char* label, cl_kernel k, int dims,
                 const size_t* gws, const size_t* lws);
    void profile_dump();
    void run_gemm(const GLayer& A, cl_mem Bsrc, int ldB, const Buf& C,
                  int M, int N, int K, int KS,
                  const GBN* bn, bool rowbias,
                  cl_mem cb, int ldcb, int cb_mode,   // 0 none, 1 pre, 2 post
                  cl_mem res, int act);
    void run_bnact(const Buf& src, const Buf& dst, const GBN& bn, int C, int T, int act);
    void run_resadd(const Buf& a, const Buf& b, const Buf* mul, const Buf& dst,
                    int C, int T, int act);
    void run_stats(const Buf& src, const Buf& dst, int C, int N, int variant);
    void run_xpose(int C, int HW, int N, int src_off, const Buf& dst);

    // Forward passes (enqueue full graph for one chunk of B images)
    void forward_resnet_trunk(int B);
    void forward_kata_trunk(int B);
    void forward_minigo_heads(int B, const Buf& trunk);
    void forward_kata1_heads(int B);
    void forward_vit(int B);

    void predict_chunk(const std::vector<std::vector<float>>& states,
                       size_t lo, size_t hi, std::vector<Result>& out);
};

}  // namespace minigo
#endif  // MINIGO_HAS_OPENCL
