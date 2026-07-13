#ifdef MINIGO_HAS_OPENCL

// ================================================================
// OpenCL inference backend — all three model architectures:
//
//   * resnet  — MiniGo KataGo-style ResNet (SE + GPool blocks,
//               global-pooled heads), single 17-plane input
//   * vit     — GoViT transformer (GQA + directional rel-bias),
//               single 17-plane input
//   * katago  — KataGo-V7 dual input (22 spatial + 19 global):
//               trainable KataGoNet naming AND converted kata1
//               naming (mish/relu autodetected)
//
// Two precision paths (see include/opencl_compute.h):
//   fp32       portable OpenCL C 1.2
//   fp16(+MMA) half storage everywhere, fp32 accumulation, NVIDIA
//              tensor cores via inline-PTX mma.sync where available
//
// Every convolution / FC is one fused implicit-GEMM launch whose
// epilogue applies BN, biases, per-(channel,image) gpool bias,
// residual add and activation — a full residual block is 3-4 kernel
// launches, and the whole output is packed GPU-side into a single
// fp32 buffer read back once per batch.
// ================================================================

#include "backends/opencl_compute.h"
#include "opencl_kernels.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <sstream>
#include <stdexcept>

namespace minigo {

// ================================================================
// Small utilities
// ================================================================
#define CL_CHECK(expr)                                                        \
    do {                                                                      \
        cl_int _err = (expr);                                                 \
        if (_err != CL_SUCCESS) {                                             \
            std::ostringstream _os;                                           \
            _os << "OpenCL error " << _err << " at " << __FILE__              \
                << ":" << __LINE__;                                           \
            throw std::runtime_error(_os.str());                              \
        }                                                                     \
    } while (0)

namespace {

int round_up(int n, int a) { return ((n + a - 1) / a) * a; }

// fp32 -> fp16 (round to nearest even), for weight/host conversions.
uint16_t f2h(float f) {
    uint32_t x;
    std::memcpy(&x, &f, 4);
    uint32_t sign = (x >> 16) & 0x8000u;
    int32_t  exp  = (int32_t)((x >> 23) & 0xFF) - 127 + 15;
    uint32_t man  = x & 0x7FFFFFu;
    if (((x >> 23) & 0xFF) == 0xFF)               // inf/nan
        return (uint16_t)(sign | 0x7C00u | (man ? 0x200u : 0u));
    if (exp >= 31) return (uint16_t)(sign | 0x7C00u);   // overflow -> inf
    if (exp <= 0) {                                // subnormal / zero
        if (exp < -10) return (uint16_t)sign;
        man |= 0x800000u;
        uint32_t shift = (uint32_t)(14 - exp);
        uint32_t h = man >> shift;
        uint32_t rem = man & ((1u << shift) - 1);
        uint32_t half = 1u << (shift - 1);
        if (rem > half || (rem == half && (h & 1))) h++;
        return (uint16_t)(sign | h);
    }
    uint32_t h = sign | ((uint32_t)exp << 10) | (man >> 13);
    uint32_t rem = man & 0x1FFFu;
    if (rem > 0x1000u || (rem == 0x1000u && (h & 1))) h++;
    return (uint16_t)h;
}

float host_softplus(float x) {
    if (x > 20.0f) return x;
    return std::log1p(std::exp(x));
}

const char* precision_name(OpenCLPrecision p) {
    switch (p) {
        case OpenCLPrecision::FP32:     return "fp32";
        case OpenCLPrecision::FP16:     return "fp16";
        case OpenCLPrecision::FP16_MMA: return "fp16 + tensor-core MMA";
    }
    return "?";
}

// Epilogue flag bits / activation codes — must match opencl_kernels.h.
enum { EF_ROWBIAS = 1, EF_BN = 2, EF_CB_PRE = 4, EF_CB_POST = 8, EF_RES = 16 };
enum { ACT_NONE = 0, ACT_RELU = 1, ACT_MISH = 2, ACT_GELU = 3, ACT_SIGMOID = 4 };

// Sequential kernel-argument setter.
struct Args {
    cl_kernel k;
    cl_uint i = 0;
    explicit Args(cl_kernel kk) : k(kk) {}
    Args& mem(cl_mem m) { CL_CHECK(clSetKernelArg(k, i++, sizeof(cl_mem), &m)); return *this; }
    Args& i32(int v)    { CL_CHECK(clSetKernelArg(k, i++, sizeof(int), &v));    return *this; }
    Args& f32(float v)  { CL_CHECK(clSetKernelArg(k, i++, sizeof(float), &v));  return *this; }
};

cl_program build_program(cl_context ctx, cl_device_id dev,
                         const char* src, const std::string& options,
                         std::string* err_log) {
    cl_int err;
    cl_program prog = clCreateProgramWithSource(ctx, 1, &src, nullptr, &err);
    CL_CHECK(err);
    err = clBuildProgram(prog, 1, &dev, options.c_str(), nullptr, nullptr);
    if (err != CL_SUCCESS) {
        size_t log_size = 0;
        clGetProgramBuildInfo(prog, dev, CL_PROGRAM_BUILD_LOG, 0, nullptr, &log_size);
        std::string log(log_size, ' ');
        clGetProgramBuildInfo(prog, dev, CL_PROGRAM_BUILD_LOG, log_size, &log[0], nullptr);
        clReleaseProgram(prog);
        if (err_log) *err_log = log;
        return nullptr;
    }
    return prog;
}

}  // namespace

// ================================================================
// OpenCLComputeContext
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
    ds.name = name;

    cl_int err;
    ds.context = clCreateContext(nullptr, 1, &ds.device, nullptr, nullptr, &err);
    CL_CHECK(err);
    // No command queue here — queues are per-handle (per server thread).
}

static bool is_nvidia(const OpenCLDeviceState& ds) {
    char vendor[256] = {};
    clGetDeviceInfo(ds.device, CL_DEVICE_VENDOR, sizeof(vendor), vendor, nullptr);
    std::string v(vendor), n(ds.name);
    return v.find("NVIDIA") != std::string::npos ||
           n.find("NVIDIA") != std::string::npos;
}

static void compile_device_programs(OpenCLDeviceState& ds) {
    // Precision request: fp32 | fp16 | fp16-portable | auto.
    // auto/fp16 pick tensor-core MMA on NVIDIA; fp16-portable skips the
    // MMA probe (debugging aid + the exact path non-NVIDIA devices run).
    const char* env = std::getenv("MINIGO_OPENCL_PRECISION");
    std::string req = env ? env : "auto";
    for (auto& c : req) c = (char)std::tolower(c);

    bool want_fp16 = (req != "fp32");
    bool allow_mma = (req != "fp16-portable");
    const std::string base_opts = "-cl-mad-enable -cl-denorms-are-zero";

    std::string log;
    if (want_fp16) {
        ds.program = build_program(ds.context, ds.device, OPENCL_PORTABLE_SRC,
                                   base_opts + " -DSTORE_HALF=1", &log);
        if (!ds.program)
            throw std::runtime_error("OpenCL build error (portable fp16):\n" + log);
        ds.precision = OpenCLPrecision::FP16;

        // NVIDIA tensor-core GEMM: probe-compile; fall back silently if the
        // driver rejects inline PTX (non-NVIDIA or exotic stacks).
        if (allow_mma && is_nvidia(ds)) {
            ds.program_mma = build_program(ds.context, ds.device, OPENCL_MMA_SRC,
                                           base_opts, &log);
            if (ds.program_mma) {
                ds.precision = OpenCLPrecision::FP16_MMA;
            } else {
                std::cerr << "OpenCL: tensor-core MMA program rejected, "
                          << "falling back to portable fp16.\n";
            }
        }
    } else {
        ds.program = build_program(ds.context, ds.device, OPENCL_PORTABLE_SRC,
                                   base_opts + " -DSTORE_HALF=0", &log);
        if (!ds.program)
            throw std::runtime_error("OpenCL build error (portable fp32):\n" + log);
        ds.precision = OpenCLPrecision::FP32;
    }
}

OpenCLComputeContext::OpenCLComputeContext(const std::vector<int>& device_ids) {
    for (int id : device_ids) {
        if (devices_.count(id)) continue;
        auto& ds = devices_[id];
        init_device(ds, id);
        compile_device_programs(ds);
        std::cout << "OpenCL device " << id << ": " << ds.name
                  << " — " << precision_name(ds.precision) << "\n";
    }
}

OpenCLComputeContext::~OpenCLComputeContext() {
    for (auto& [id, ds] : devices_) {
        if (ds.program)     clReleaseProgram(ds.program);
        if (ds.program_mma) clReleaseProgram(ds.program_mma);
        if (ds.context)     clReleaseContext(ds.context);
    }
}

OpenCLDeviceState& OpenCLComputeContext::device_state(int gpu_id) {
    auto it = devices_.find(gpu_id);
    if (it == devices_.end())
        throw std::runtime_error("OpenCL device " + std::to_string(gpu_id) +
                                 " not initialized");
    return it->second;
}

std::unique_ptr<ComputeHandle>
OpenCLComputeContext::create_handle(const LoadedModel* model, int gpu_id,
                                    int max_batch_size) {
    return std::make_unique<OpenCLComputeHandle>(device_state(gpu_id), model,
                                                 max_batch_size);
}

// ================================================================
// OpenCLComputeHandle — construction
// ================================================================

OpenCLComputeHandle::OpenCLComputeHandle(OpenCLDeviceState& dev,
                                         const LoadedModel* model,
                                         int max_batch_size)
    : dev_(dev), model_(model)
{
    fp16_ = dev_.precision != OpenCLPrecision::FP32;
    mma_  = dev_.precision == OpenCLPrecision::FP16_MMA;

    max_batch_ = std::max(1, max_batch_size);
    board_     = model->board_size;
    hw_        = board_ * board_;
    state_len_ = (model->format == ModelFormat::KataGo)
               ? model->input_channels * hw_ + model->input_global_channels
               : model->input_channels * hw_;
    out_stride_ = 2 * hw_ + 6;   // A + 3 + 1 + 1 + hw, A = hw+1
    ldT_ = round_up(max_batch_ * hw_, 16);
    ldN_ = round_up(max_batch_, 16);

    const char* prof_env = std::getenv("MINIGO_OPENCL_PROFILE");
    profile_ = prof_env && prof_env[0] && prof_env[0] != '0';

    cl_int err;
    // Per-handle command queue on this server thread (per-thread execution
    // state — see OpenCLDeviceState comment in the header).
    queue_ = clCreateCommandQueue(dev_.context, dev_.device,
                                  profile_ ? CL_QUEUE_PROFILING_ENABLE : 0, &err);
    CL_CHECK(err);

    auto kern = [&](cl_program p, const char* name) {
        cl_int e;
        cl_kernel k = clCreateKernel(p, name, &e);
        CL_CHECK(e);
        return k;
    };
    k_xpose_   = kern(dev_.program, "xpose_in");
    k_gemm_    = kern(dev_.program, "gemm");
    k_bnact_   = kern(dev_.program, "bn_act_ew");
    k_resadd_  = kern(dev_.program, "resadd_act");
    k_stats_   = kern(dev_.program, "gstats");
    k_ln_      = kern(dev_.program, "layernorm");
    k_pos_     = kern(dev_.program, "pos_embed_add");
    k_att_     = kern(dev_.program, "attention");
    k_meantok_ = kern(dev_.program, "mean_tokens");
    k_flat_    = kern(dev_.program, "flatten_pol");
    k_pack_mg_ = kern(dev_.program, "pack_mg");
    k_pack_sp_ = kern(dev_.program, "pack_sp");
    if (mma_)
        k_gemm_mma_ = kern(dev_.program_mma, "gemm");

    upload_weights();
    alloc_workspace();
}

OpenCLComputeHandle::~OpenCLComputeHandle() {
    if (queue_) clFinish(queue_);
    for (cl_mem m : owned_)
        if (m) clReleaseMemObject(m);
    cl_kernel ks[] = { k_xpose_, k_gemm_, k_gemm_mma_, k_bnact_, k_resadd_,
                       k_stats_, k_ln_, k_pos_, k_att_, k_meantok_, k_flat_,
                       k_pack_mg_, k_pack_sp_ };
    for (cl_kernel k : ks)
        if (k) clReleaseKernel(k);
    if (queue_) clReleaseCommandQueue(queue_);
}

// ── Upload helpers ───────────────────────────────────────────────
cl_mem OpenCLComputeHandle::up_raw(const void* data, size_t bytes) {
    cl_int err;
    cl_mem m = clCreateBuffer(dev_.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                              std::max<size_t>(bytes, 4), const_cast<void*>(data), &err);
    CL_CHECK(err);
    owned_.push_back(m);
    return m;
}

cl_mem OpenCLComputeHandle::up_f32(const std::vector<float>& v) {
    static const float zero = 0.0f;
    if (v.empty()) return up_raw(&zero, 4);
    return up_raw(v.data(), v.size() * 4);
}

cl_mem OpenCLComputeHandle::up_xt(const std::vector<float>& v) {
    if (!fp16_) return up_f32(v);
    std::vector<uint16_t> h(std::max<size_t>(v.size(), 2));
    for (size_t i = 0; i < v.size(); i++) h[i] = f2h(v[i]);
    return up_raw(h.data(), h.size() * 2);
}

OpenCLComputeHandle::GLayer OpenCLComputeHandle::up_conv(const ConvW& c) {
    GLayer g;
    g.w = up_xt(c.weight);
    g.out = c.c_out; g.in = c.c_in; g.k = c.k;
    g.has_b = c.has_bias;
    if (c.has_bias) g.b = up_f32(c.bias);
    return g;
}

OpenCLComputeHandle::GLayer OpenCLComputeHandle::up_fc(const FCW& f) {
    GLayer g;
    g.w = up_xt(f.weight);
    g.out = f.out_features; g.in = f.in_features; g.k = 1;
    g.has_b = f.has_bias;
    if (f.has_bias) g.b = up_f32(f.bias);
    return g;
}

OpenCLComputeHandle::GLayer
OpenCLComputeHandle::up_fc_bias(const FCW& f, const std::vector<float>& bias) {
    GLayer g = up_fc(f);
    g.has_b = true;
    g.b = up_f32(bias);
    return g;
}

OpenCLComputeHandle::GBN OpenCLComputeHandle::up_bn(const BNParamsW& b) {
    GBN g;
    g.scale = up_f32(b.scale);
    g.bias  = up_f32(b.bias);
    g.ch = b.channels;
    return g;
}

OpenCLComputeHandle::GGHead OpenCLComputeHandle::up_ghead(const GPoolHeadW& h) {
    GGHead g;
    g.conv = up_conv(h.conv);
    g.bn   = up_bn(h.bn);
    g.fc1  = up_fc(h.fc1);
    g.fc2  = up_fc(h.fc2);
    return g;
}

OpenCLComputeHandle::Buf OpenCLComputeHandle::make_buf(int rows, int ld) {
    Buf b;
    b.rows = std::max(rows, 1);
    b.ld = ld;
    size_t elt = fp16_ ? 2 : 4;
    cl_int err;
    b.mem = clCreateBuffer(dev_.context, CL_MEM_READ_WRITE,
                           std::max<size_t>((size_t)b.rows * ld * elt, 16), nullptr, &err);
    CL_CHECK(err);
    owned_.push_back(b.mem);
    return b;
}

void OpenCLComputeHandle::upload_weights() {
    const ModelWeights& w = model_->weights();

    if (w.trunk_style == ModelWeights::ViTTrunk) {
        const ViTW& v = w.vit;
        vt_token_ = up_fc(v.token_proj);
        vt_row_ = up_f32(v.row_embed);
        vt_col_ = up_f32(v.col_embed);
        vt_rel_ = up_f32(v.rel_bias);
        vt_blocks_.resize(v.blocks.size());
        for (size_t i = 0; i < v.blocks.size(); i++) {
            const ViTBlockW& b = v.blocks[i];
            GViTBlock& g = vt_blocks_[i];
            g.ln1_g = up_f32(b.ln1_g); g.ln1_b = up_f32(b.ln1_b);
            g.ln2_g = up_f32(b.ln2_g); g.ln2_b = up_f32(b.ln2_b);
            g.qkv = up_fc(b.qkv);
            g.out_proj = up_fc(b.out_proj);
            g.mlp1 = up_fc(b.mlp1);
            g.mlp2 = up_fc(b.mlp2);
        }
        vt_fin_g_ = up_f32(v.final_ln_g);
        vt_fin_b_ = up_f32(v.final_ln_b);
        vt_policy_ = up_fc(v.policy_proj);
        vt_val1_ = up_fc(v.value_fc1);  vt_val2_ = up_fc(v.value_fc2);
        vt_sm1_ = up_fc(v.score_mean_fc1);  vt_sm2_ = up_fc(v.score_mean_fc2);
        vt_sd1_ = up_fc(v.score_stdev_fc1); vt_sd2_ = up_fc(v.score_stdev_fc2);
        vt_own_ = up_fc(v.ownership_proj);
        return;
    }

    if (w.trunk_style == ModelWeights::KataTrunk) {
        stem_conv_ = up_conv(w.stem_conv);
        stem_global_ = up_fc(w.stem_global);
        tip_bn_ = up_bn(w.tip_bn);
    } else {
        input_conv_ = up_conv(w.input_conv);
        input_bn_ = up_bn(w.input_bn);
    }

    blocks_.resize(w.blocks.size());
    for (size_t i = 0; i < w.blocks.size(); i++) {
        const TrunkBlockW& b = w.blocks[i];
        GBlock& g = blocks_[i];
        g.kind = (int)b.kind;
        g.pc = b.pool_channels;
        switch (b.kind) {
            case TrunkBlockW::SE:
                g.conv1 = up_conv(b.conv1); g.bn1 = up_bn(b.bn1);
                g.conv2 = up_conv(b.conv2); g.bn2 = up_bn(b.bn2);
                g.se_fc1 = up_fc(b.se_fc1); g.se_fc2 = up_fc(b.se_fc2);
                break;
            case TrunkBlockW::MgGPool:
                g.conv_main = up_conv(b.conv_main); g.bn_main = up_bn(b.bn_main);
                g.conv_pool = up_conv(b.conv_pool); g.bn_pool = up_bn(b.bn_pool);
                g.pool_fc = up_fc(b.pool_fc);
                g.conv2 = up_conv(b.conv2); g.bn2 = up_bn(b.bn2);
                break;
            case TrunkBlockW::KataRegular:
                g.pre = up_bn(b.pre_bn); g.mid = up_bn(b.mid_bn);
                g.regular = up_conv(b.regular_conv);
                g.final = up_conv(b.final_conv);
                break;
            case TrunkBlockW::KataGPool:
                g.pre = up_bn(b.pre_bn); g.mid = up_bn(b.mid_bn);
                g.regular = up_conv(b.regular_conv);
                g.final = up_conv(b.final_conv);
                g.gpool = up_conv(b.gpool_conv); g.gp_bn = up_bn(b.gpool_bn);
                g.g2b = up_fc(b.gpool_to_bias);
                break;
        }
    }

    if (w.head_style == ModelWeights::MiniGoHeads) {
        policy_conv_ = up_conv(w.policy_conv);
        policy_bn_ = up_bn(w.policy_bn);
        policy_fc_ = up_fc(w.policy_fc);
        value_head_ = up_ghead(w.value_head);
        score_mean_head_ = up_ghead(w.score_mean_head);
        score_stdev_head_ = up_ghead(w.score_stdev_head);
        ownership_conv_ = up_conv(w.ownership_conv);
    } else {
        p1_conv_ = up_conv(w.p1_conv);
        g1_conv_ = up_conv(w.g1_conv);
        p2_conv_ = up_conv(w.p2_conv);
        g1_bn_ = up_bn(w.g1_bn);
        p1_bn_ = up_bn(w.p1_bn);
        k_g2b_ = up_fc(w.k_gpool_to_bias);
        k_g2pass_ = up_fc(w.k_gpool_to_pass);
        v1_conv_ = up_conv(w.v1_conv);
        v1_bn_ = up_bn(w.v1_bn);
        v2_mul_ = up_fc_bias(w.v2_mul, w.v2_bias);
        v3_mul_ = up_fc_bias(w.v3_mul, w.v3_bias);
        sv3_mul_ = up_fc_bias(w.sv3_mul, w.sv3_bias);
        vown_conv_ = up_conv(w.vown_conv);
    }
}

void OpenCLComputeHandle::alloc_workspace() {
    const ModelWeights& w = model_->weights();
    cl_int err;

    staging_ = clCreateBuffer(dev_.context, CL_MEM_READ_ONLY,
                              (size_t)max_batch_ * state_len_ * 4, nullptr, &err);
    CL_CHECK(err);
    owned_.push_back(staging_);
    out_pack_ = clCreateBuffer(dev_.context, CL_MEM_WRITE_ONLY,
                               (size_t)max_batch_ * out_stride_ * 4, nullptr, &err);
    CL_CHECK(err);
    owned_.push_back(out_pack_);
    staging_host_.resize((size_t)max_batch_ * state_len_);

    int A = hw_ + 1;

    if (w.trunk_style == ModelWeights::ViTTrunk) {
        const ViTW& v = w.vit;
        int d = model_->num_filters;
        int q_dim = model_->vit_heads * model_->vit_head_dim;
        int kv_dim = model_->vit_kv_groups * model_->vit_head_dim;
        x_in_  = make_buf(model_->input_channels, ldT_);
        buf_a_ = make_buf(d, ldT_);                        // X
        buf_b_ = make_buf(d, ldT_);                        // Xn
        buf_c_ = make_buf(v.blocks.empty() ? d : v.blocks[0].mlp1.out_features, ldT_);
        buf_d_ = make_buf(q_dim + 2 * kv_dim, ldT_);       // QKV
        buf_e_ = make_buf(q_dim, ldT_);                    // attention out
        stats_ = make_buf(d, ldN_);                        // pooled tokens
        o_mlp_ = make_buf(std::max(d, 64), ldN_);
        o_v_   = make_buf(3, ldN_);
        o_sm_  = make_buf(1, ldN_);
        o_sd_  = make_buf(1, ldN_);
        o_own_ = make_buf(1, ldT_);
        p_sp_  = make_buf(1, ldT_);
        return;
    }

    int F = model_->num_filters;
    int Cin = model_->input_channels;
    x_in_ = make_buf(Cin, ldT_);
    if (w.trunk_style == ModelWeights::KataTrunk)
        g_in_ = make_buf(model_->input_global_channels, ldN_);

    buf_a_ = make_buf(F, ldT_);   // trunk
    buf_b_ = make_buf(F, ldT_);   // h (kata pre-activation)
    buf_c_ = make_buf(F, ldT_);   // t1
    buf_d_ = make_buf(F, ldT_);   // t2

    int e_rows = 2, mlp_rows = 64, stat_c = F, sv3c = 1, pass_rows = 1, cb_rows = F;
    if (w.head_style == ModelWeights::MiniGoHeads) {
        e_rows = std::max({2, w.value_head.conv.c_out,
                           w.score_mean_head.conv.c_out,
                           w.score_stdev_head.conv.c_out});
        mlp_rows = std::max({64, w.value_head.fc1.out_features,
                             w.score_mean_head.fc1.out_features,
                             w.score_stdev_head.fc1.out_features});
        for (auto& b : w.blocks) {
            if (b.kind == TrunkBlockW::SE)
                mlp_rows = std::max(mlp_rows, b.se_fc1.out_features);
        }
        flat_  = make_buf(w.policy_conv.c_out * hw_, ldN_);
        o_pol_ = make_buf(A, ldN_);
    } else {
        e_rows = std::max({w.p1_conv.c_out, w.g1_conv.c_out,
                           w.v1_conv.c_out, w.p2_conv.c_out, 2});
        mlp_rows = std::max(64, w.v2_mul.out_features);
        sv3c = w.sv3_mul.out_features;
        pass_rows = w.k_gpool_to_pass.out_features;
        stat_c = std::max({F, w.g1_conv.c_out, w.v1_conv.c_out});
        cb_rows = std::max(F, w.p1_conv.c_out);
    }
    buf_e_ = make_buf(e_rows, ldT_);
    stats_ = make_buf(3 * stat_c, ldN_);
    cb_    = make_buf(cb_rows, ldN_);
    o_mlp_ = make_buf(mlp_rows, ldN_);
    o_v_   = make_buf(3, ldN_);
    o_sm_  = make_buf(1, ldN_);
    o_sd_  = make_buf(1, ldN_);
    o_sv3_ = make_buf(sv3c, ldN_);
    o_pass_= make_buf(pass_rows, ldN_);
    o_own_ = make_buf(1, ldT_);
    p_sp_  = make_buf(std::max(2, w.head_style == ModelWeights::Kata1Heads
                                   ? w.p2_conv.c_out : 2), ldT_);
}

// ================================================================
// Kernel enqueue helpers
// ================================================================

void OpenCLComputeHandle::enqueue(const char* label, cl_kernel k, int dims,
                                  const size_t* gws, const size_t* lws) {
    cl_event ev = nullptr;
    CL_CHECK(clEnqueueNDRangeKernel(queue_, k, (cl_uint)dims, nullptr, gws, lws,
                                    0, nullptr, profile_ ? &ev : nullptr));
    if (profile_) events_.emplace_back(label, ev);
}

// Per-kernel GPU-time summary (MINIGO_OPENCL_PROFILE=1), grouped by label.
void OpenCLComputeHandle::profile_dump() {
    if (!profile_ || events_.empty()) return;
    struct Acc { double ms = 0; int n = 0; };
    std::vector<std::pair<std::string, Acc>> order;
    double total = 0, prev_end = 0, gaps = 0;
    bool first = true;
    for (auto& [label, ev] : events_) {
        cl_ulong t0 = 0, t1 = 0;
        clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_START, sizeof(t0), &t0, nullptr);
        clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_END, sizeof(t1), &t1, nullptr);
        double ms = (double)(t1 - t0) / 1e6;
        if (!first) gaps += std::max(0.0, ((double)t0 - prev_end) / 1e6);
        prev_end = (double)t1;
        first = false;
        total += ms;
        auto it = std::find_if(order.begin(), order.end(),
                               [&](auto& p) { return p.first == label; });
        if (it == order.end()) order.push_back({label, {ms, 1}});
        else { it->second.ms += ms; it->second.n++; }
        clReleaseEvent(ev);
    }
    events_.clear();
    std::cerr << "[opencl profile] kernels " << total << " ms, inter-kernel gaps "
              << gaps << " ms\n";
    for (auto& [label, acc] : order)
        std::cerr << "  " << label << ": " << acc.ms << " ms / " << acc.n << "\n";
}

void OpenCLComputeHandle::run_gemm(const GLayer& A, cl_mem Bsrc, int ldB,
                                   const Buf& C, int M, int N, int K, int KS,
                                   const GBN* bn, bool rowbias,
                                   cl_mem cb, int ldcb, int cb_mode,
                                   cl_mem res, int act) {
    int flags = 0;
    cl_mem p0 = nullptr, p1 = nullptr;
    if (bn) { flags |= EF_BN; p0 = bn->scale; p1 = bn->bias; }
    if (rowbias) { flags |= EF_ROWBIAS; p1 = A.b; }
    if (cb_mode == 1) flags |= EF_CB_PRE;
    if (cb_mode == 2) flags |= EF_CB_POST;
    if (res) flags |= EF_RES;

    cl_kernel k = mma_ ? k_gemm_mma_ : k_gemm_;
    Args a(k);
    a.mem(A.w).mem(Bsrc).mem(C.mem).mem(p0).mem(p1).mem(cb).mem(res)
     .i32(M).i32(N).i32(K)
     .i32(ldB).i32(C.ld).i32(ldcb)
     .i32(KS).i32(KS / 2).i32(board_).i32(board_).i32(hw_)
     .i32(flags).i32(act);

    const char* label = KS > 1 ? (mma_ ? "conv(mma)" : "conv") : (mma_ ? "fc(mma)" : "fc");
    if (mma_) {
        size_t gws[2] = { (size_t)((M + 63) / 64) * 128, (size_t)((N + 63) / 64) };
        size_t lws[2] = { 128, 1 };
        enqueue(label, k, 2, gws, lws);
    } else {
        size_t gws[2] = { (size_t)((M + 63) / 64) * 16, (size_t)((N + 63) / 64) * 16 };
        size_t lws[2] = { 16, 16 };
        enqueue(label, k, 2, gws, lws);
    }
}

void OpenCLComputeHandle::run_bnact(const Buf& src, const Buf& dst, const GBN& bn,
                                    int C, int T, int act) {
    Args a(k_bnact_);
    a.mem(src.mem).mem(dst.mem).mem(bn.scale).mem(bn.bias)
     .i32(C).i32(T).i32(src.ld).i32(act);
    size_t lws = 256, gws = ((size_t)C * T + lws - 1) / lws * lws;
    enqueue("bn_act", k_bnact_, 1, &gws, &lws);
}

void OpenCLComputeHandle::run_resadd(const Buf& a, const Buf& b, const Buf* mul,
                                     const Buf& dst, int C, int T, int act) {
    Args ar(k_resadd_);
    ar.mem(a.mem).mem(b.mem).mem(mul ? mul->mem : nullptr).mem(dst.mem)
      .i32(C).i32(T).i32(a.ld).i32(mul ? mul->ld : 1).i32(hw_)
      .i32(mul ? 1 : 0).i32(act);
    size_t lws = 256, gws = ((size_t)C * T + lws - 1) / lws * lws;
    enqueue("resadd", k_resadd_, 1, &gws, &lws);
}

void OpenCLComputeHandle::run_stats(const Buf& src, const Buf& dst,
                                    int C, int N, int variant) {
    float c1 = (std::sqrt((float)hw_) - 14.0f) * 0.1f;
    float d14 = std::sqrt((float)hw_) - 14.0f;
    float c2 = d14 * d14 * 0.01f - 0.1f;
    Args a(k_stats_);
    a.mem(src.mem).mem(dst.mem)
     .i32(C).i32(N).i32(hw_).i32(src.ld).i32(dst.ld)
     .i32(variant).f32(c1).f32(c2);
    size_t lws = 64, gws = (size_t)64 * C * N;
    enqueue("gstats", k_stats_, 1, &gws, &lws);
}

void OpenCLComputeHandle::run_xpose(int C, int HW, int N, int src_off, const Buf& dst) {
    Args a(k_xpose_);
    a.mem(staging_).mem(dst.mem)
     .i32(state_len_).i32(src_off).i32(C).i32(HW).i32(N).i32(dst.ld);
    size_t lws = 256, gws = ((size_t)C * N * HW + lws - 1) / lws * lws;
    enqueue("xpose", k_xpose_, 1, &gws, &lws);
}

// ================================================================
// Forward passes
// ================================================================

// Stats-kernel variants (must match gstats in opencl_kernels.h)
enum { ST_MEAN = 0, ST_MEAN_MAX = 1, ST_MEAN_MAX_STD = 2, ST_KATA = 3, ST_KATA_VH = 4 };

void OpenCLComputeHandle::forward_resnet_trunk(int B) {
    const int T = B * hw_;
    const int F = model_->num_filters;

    run_xpose(model_->input_channels, hw_, B, 0, x_in_);
    // trunk = relu(bn(conv3x3(x)))
    run_gemm(input_conv_, x_in_.mem, x_in_.ld, buf_a_,
             F, T, input_conv_.in * 9, 3,
             &input_bn_, false, nullptr, 1, 0, nullptr, ACT_RELU);

    for (auto& g : blocks_) {
        if (g.kind == (int)TrunkBlockW::SE) {
            // t1 = relu(bn1(conv1(trunk)))
            run_gemm(g.conv1, buf_a_.mem, buf_a_.ld, buf_c_, F, T, F * 9, 3,
                     &g.bn1, false, nullptr, 1, 0, nullptr, ACT_RELU);
            // t2 = bn2(conv2(t1))
            run_gemm(g.conv2, buf_c_.mem, buf_c_.ld, buf_d_, F, T, F * 9, 3,
                     &g.bn2, false, nullptr, 1, 0, nullptr, ACT_NONE);
            // SE gate: gap -> fc1(relu) -> fc2(sigmoid)
            run_stats(buf_d_, stats_, F, B, ST_MEAN);
            run_gemm(g.se_fc1, stats_.mem, stats_.ld, o_mlp_,
                     g.se_fc1.out, B, F, 1, nullptr, true, nullptr, 1, 0,
                     nullptr, ACT_RELU);
            run_gemm(g.se_fc2, o_mlp_.mem, o_mlp_.ld, cb_,
                     F, B, g.se_fc1.out, 1, nullptr, true, nullptr, 1, 0,
                     nullptr, ACT_SIGMOID);
            // trunk = relu(t2 * se + trunk)
            run_resadd(buf_d_, buf_a_, &cb_, buf_a_, F, T, ACT_RELU);
        } else {  // MgGPool
            const int pc = g.pc;
            // pool branch: relu(bn_pool(conv_pool(trunk))) -> [mean;max] -> pool_fc
            run_gemm(g.conv_pool, buf_a_.mem, buf_a_.ld, buf_c_, pc, T, F * 9, 3,
                     &g.bn_pool, false, nullptr, 1, 0, nullptr, ACT_RELU);
            run_stats(buf_c_, stats_, pc, B, ST_MEAN_MAX);
            run_gemm(g.pool_fc, stats_.mem, stats_.ld, cb_,
                     F, B, 2 * pc, 1, nullptr, true, nullptr, 1, 0,
                     nullptr, ACT_NONE);
            // main = relu(bn_main(conv_main(trunk)) + cb)
            run_gemm(g.conv_main, buf_a_.mem, buf_a_.ld, buf_d_, F, T, F * 9, 3,
                     &g.bn_main, false, cb_.mem, cb_.ld, 2, nullptr, ACT_RELU);
            // trunk = relu(bn2(conv2(main)) + trunk)
            run_gemm(g.conv2, buf_d_.mem, buf_d_.ld, buf_a_, F, T, F * 9, 3,
                     &g.bn2, false, nullptr, 1, 0, buf_a_.mem, ACT_RELU);
        }
    }
}

void OpenCLComputeHandle::forward_kata_trunk(int B) {
    const ModelWeights& w = model_->weights();
    const int T = B * hw_;
    const int F = model_->num_filters;
    const int act = w.use_mish ? ACT_MISH : ACT_RELU;

    run_xpose(model_->input_channels, hw_, B, 0, x_in_);
    run_xpose(model_->input_global_channels, 1, B,
              model_->input_channels * hw_, g_in_);

    // cb = stem_global(global); trunk = stem_conv(spatial) + cb (broadcast)
    run_gemm(stem_global_, g_in_.mem, g_in_.ld, cb_,
             F, B, stem_global_.in, 1, nullptr, stem_global_.has_b,
             nullptr, 1, 0, nullptr, ACT_NONE);
    run_gemm(stem_conv_, x_in_.mem, x_in_.ld, buf_a_,
             F, T, stem_conv_.in * stem_conv_.k * stem_conv_.k, stem_conv_.k,
             nullptr, false, cb_.mem, cb_.ld, 1, nullptr, ACT_NONE);

    for (auto& g : blocks_) {
        // h = act(pre_bn(trunk))
        run_bnact(buf_a_, buf_b_, g.pre, F, T, act);
        if (g.kind == (int)TrunkBlockW::KataRegular) {
            run_gemm(g.regular, buf_b_.mem, buf_b_.ld, buf_c_, F, T, F * 9, 3,
                     &g.mid, false, nullptr, 1, 0, nullptr, act);
            run_gemm(g.final, buf_c_.mem, buf_c_.ld, buf_a_, F, T, F * 9, 3,
                     nullptr, false, nullptr, 1, 0, buf_a_.mem, ACT_NONE);
        } else {  // KataGPool
            const int pc = g.pc;
            run_gemm(g.gpool, buf_b_.mem, buf_b_.ld, buf_d_, pc, T, F * 9, 3,
                     &g.gp_bn, false, nullptr, 1, 0, nullptr, act);
            run_stats(buf_d_, stats_, pc, B, ST_KATA);
            run_gemm(g.g2b, stats_.mem, stats_.ld, cb_,
                     F, B, 3 * pc, 1, nullptr, g.g2b.has_b, nullptr, 1, 0,
                     nullptr, ACT_NONE);
            // regular = act(mid_bn(conv(h) + cb))
            run_gemm(g.regular, buf_b_.mem, buf_b_.ld, buf_c_, F, T, F * 9, 3,
                     &g.mid, false, cb_.mem, cb_.ld, 1, nullptr, act);
            run_gemm(g.final, buf_c_.mem, buf_c_.ld, buf_a_, F, T, F * 9, 3,
                     nullptr, false, nullptr, 1, 0, buf_a_.mem, ACT_NONE);
        }
    }
    // trunk tip: act(bn(trunk)) in place
    run_bnact(buf_a_, buf_a_, tip_bn_, F, T, act);
}

void OpenCLComputeHandle::forward_minigo_heads(int B, const Buf& trunk) {
    const ModelWeights& w = model_->weights();
    const int T = B * hw_;
    const int F = model_->num_filters;
    const int A = hw_ + 1;

    // ── Policy: 1x1 conv(2ch)+bn+relu -> flatten -> fc ──
    run_gemm(policy_conv_, trunk.mem, trunk.ld, buf_e_,
             policy_conv_.out, T, F, 1, &policy_bn_, false,
             nullptr, 1, 0, nullptr, ACT_RELU);
    {
        Args a(k_flat_);
        a.mem(buf_e_.mem).mem(flat_.mem)
         .i32(policy_conv_.out).i32(B).i32(hw_).i32(buf_e_.ld).i32(flat_.ld);
        size_t lws = 256, gws = ((size_t)policy_conv_.out * T + lws - 1) / lws * lws;
        enqueue("flatten", k_flat_, 1, &gws, &lws);
    }
    run_gemm(policy_fc_, flat_.mem, flat_.ld, o_pol_,
             A, B, policy_conv_.out * hw_, 1, nullptr, true,
             nullptr, 1, 0, nullptr, ACT_NONE);

    // ── GPool heads: conv+bn+relu -> [mean;max;std] -> fc1(relu) -> fc2 ──
    auto ghead = [&](const GGHead& h, const Buf& out, int out_rows) {
        run_gemm(h.conv, trunk.mem, trunk.ld, buf_e_,
                 h.conv.out, T, F, 1, &h.bn, false, nullptr, 1, 0,
                 nullptr, ACT_RELU);
        run_stats(buf_e_, stats_, h.conv.out, B, ST_MEAN_MAX_STD);
        run_gemm(h.fc1, stats_.mem, stats_.ld, o_mlp_,
                 h.fc1.out, B, 3 * h.conv.out, 1, nullptr, true,
                 nullptr, 1, 0, nullptr, ACT_RELU);
        run_gemm(h.fc2, o_mlp_.mem, o_mlp_.ld, out,
                 out_rows, B, h.fc1.out, 1, nullptr, true,
                 nullptr, 1, 0, nullptr, ACT_NONE);
    };
    ghead(value_head_, o_v_, 3);
    ghead(score_mean_head_, o_sm_, 1);
    ghead(score_stdev_head_, o_sd_, 1);

    // ── Ownership: 1x1 conv with bias (raw logits; host applies sigmoid) ──
    run_gemm(ownership_conv_, trunk.mem, trunk.ld, o_own_,
             1, T, F, 1, nullptr, true, nullptr, 1, 0, nullptr, ACT_NONE);

    // ── Pack ──
    {
        Args a(k_pack_mg_);
        a.mem(o_pol_.mem).mem(o_v_.mem).mem(o_sm_.mem).mem(o_sd_.mem)
         .mem(o_own_.mem).mem(out_pack_)
         .i32(A).i32(hw_).i32(o_pol_.ld).i32(o_v_.ld).i32(o_own_.ld).i32(B);
        size_t lws[2] = { 32, 8 };
        size_t gws[2] = { (size_t)round_up(out_stride_, 32), (size_t)round_up(B, 8) };
        enqueue("pack", k_pack_mg_, 2, gws, lws);
    }
    (void)w;
}

void OpenCLComputeHandle::forward_kata1_heads(int B) {
    const ModelWeights& w = model_->weights();
    const int T = B * hw_;
    const int F = model_->num_filters;
    const int act = w.use_mish ? ACT_MISH : ACT_RELU;

    // ── Policy head ──
    // g1 = act(bn(g1_conv(trunk))); gstats -> cb = gpool_to_bias @ gstats
    run_gemm(g1_conv_, buf_a_.mem, buf_a_.ld, buf_d_,
             g1_conv_.out, T, F, 1, &g1_bn_, false, nullptr, 1, 0, nullptr, act);
    run_stats(buf_d_, stats_, g1_conv_.out, B, ST_KATA);
    run_gemm(k_g2b_, stats_.mem, stats_.ld, cb_,
             p1_conv_.out, B, 3 * g1_conv_.out, 1, nullptr, k_g2b_.has_b,
             nullptr, 1, 0, nullptr, ACT_NONE);
    // p1 = act(p1_bn(p1_conv(trunk) + cb))
    run_gemm(p1_conv_, buf_a_.mem, buf_a_.ld, buf_c_,
             p1_conv_.out, T, F, 1, &p1_bn_, false, cb_.mem, cb_.ld, 1,
             nullptr, act);
    // p2 spatial logits + pass from gstats
    run_gemm(p2_conv_, buf_c_.mem, buf_c_.ld, p_sp_,
             p2_conv_.out, T, p1_conv_.out, 1, nullptr, false,
             nullptr, 1, 0, nullptr, ACT_NONE);
    run_gemm(k_g2pass_, stats_.mem, stats_.ld, o_pass_,
             k_g2pass_.out, B, 3 * g1_conv_.out, 1, nullptr, k_g2pass_.has_b,
             nullptr, 1, 0, nullptr, ACT_NONE);

    // ── Value head ──
    run_gemm(v1_conv_, buf_a_.mem, buf_a_.ld, buf_d_,
             v1_conv_.out, T, F, 1, &v1_bn_, false, nullptr, 1, 0, nullptr, act);
    run_stats(buf_d_, stats_, v1_conv_.out, B, ST_KATA_VH);
    run_gemm(v2_mul_, stats_.mem, stats_.ld, o_mlp_,
             v2_mul_.out, B, 3 * v1_conv_.out, 1, nullptr, true,
             nullptr, 1, 0, nullptr, act);
    run_gemm(v3_mul_, o_mlp_.mem, o_mlp_.ld, o_v_,
             3, B, v2_mul_.out, 1, nullptr, true, nullptr, 1, 0,
             nullptr, ACT_NONE);
    run_gemm(sv3_mul_, o_mlp_.mem, o_mlp_.ld, o_sv3_,
             sv3_mul_.out, B, v2_mul_.out, 1, nullptr, true, nullptr, 1, 0,
             nullptr, ACT_NONE);
    // ownership operates on v1 activations (buf_d_), raw logits out
    run_gemm(vown_conv_, buf_d_.mem, buf_d_.ld, o_own_,
             1, T, v1_conv_.out, 1, nullptr, false, nullptr, 1, 0,
             nullptr, ACT_NONE);

    // ── Pack (policy spatial row 0 of p2, pass row 0 of o_pass) ──
    {
        Args a(k_pack_sp_);
        a.mem(p_sp_.mem).mem(o_pass_.mem).mem(o_v_.mem)
         .mem(o_sv3_.mem).mem(o_sv3_.mem).mem(o_own_.mem).mem(out_pack_)
         .i32(hw_).i32(0).i32(1).f32(0.0f).i32(0)
         .i32(w.score_mean_idx).i32(w.score_stdev_idx)
         .i32(p_sp_.ld).i32(o_v_.ld).i32(B);
        size_t lws[2] = { 32, 8 };
        size_t gws[2] = { (size_t)round_up(out_stride_, 32), (size_t)round_up(B, 8) };
        enqueue("pack", k_pack_sp_, 2, gws, lws);
    }
}

void OpenCLComputeHandle::forward_vit(int B) {
    const ModelWeights& w = model_->weights();
    const ViTW& v = w.vit;
    const int T = B * hw_;
    const int d = model_->num_filters;
    const int Hq = model_->vit_heads, G = model_->vit_kv_groups;
    const int dh = model_->vit_head_dim;
    const int q_dim = Hq * dh, kv_dim = G * dh;
    const int span = 2 * board_ - 1;
    const int buckets = span * span;
    const float ln_eps = 1e-5f;

    if (dh > 64)
        throw std::runtime_error("OpenCL ViT: head_dim > 64 not supported");

    run_xpose(model_->input_channels, hw_, B, 0, x_in_);

    // X = token_proj(x) + pos_embed
    run_gemm(vt_token_, x_in_.mem, x_in_.ld, buf_a_,
             d, T, vt_token_.in, 1, nullptr, true, nullptr, 1, 0,
             nullptr, ACT_NONE);
    {
        Args a(k_pos_);
        a.mem(buf_a_.mem).mem(vt_row_).mem(vt_col_)
         .i32(d).i32(T).i32(buf_a_.ld).i32(board_).i32(hw_);
        size_t lws = 256, gws = ((size_t)d * T + lws - 1) / lws * lws;
        enqueue("pos_embed", k_pos_, 1, &gws, &lws);
    }

    const size_t att_lt = std::min<size_t>(round_up(hw_, 32), 1024);
    for (size_t bi = 0; bi < vt_blocks_.size(); bi++) {
        GViTBlock& g = vt_blocks_[bi];
        // Xn = LN1(X)
        {
            Args a(k_ln_);
            a.mem(buf_a_.mem).mem(buf_b_.mem).mem(g.ln1_g).mem(g.ln1_b)
             .i32(d).i32(buf_a_.ld).f32(ln_eps);
            size_t lws = 128, gws = 128 * (size_t)T;
            enqueue("layernorm", k_ln_, 1, &gws, &lws);
        }
        // QKV = qkv_proj(Xn)
        run_gemm(g.qkv, buf_b_.mem, buf_b_.ld, buf_d_,
                 q_dim + 2 * kv_dim, T, d, 1, nullptr, true, nullptr, 1, 0,
                 nullptr, ACT_NONE);
        // AO = attention(QKV)
        {
            Args a(k_att_);
            a.mem(buf_d_.mem).mem(buf_e_.mem).mem(vt_rel_)
             .i32(Hq).i32(G).i32(dh).i32(hw_).i32(board_)
             .i32(buf_d_.ld).i32(buf_e_.ld)
             .i32(q_dim).i32(q_dim + kv_dim)
             .f32(1.0f / std::sqrt((float)dh))
             .i32((int)bi * Hq * buckets).i32(span);
            size_t gws = att_lt * (size_t)B * Hq, lws = att_lt;
            enqueue("attention", k_att_, 1, &gws, &lws);
        }
        // X = X + out_proj(AO)
        run_gemm(g.out_proj, buf_e_.mem, buf_e_.ld, buf_a_,
                 d, T, q_dim, 1, nullptr, true, nullptr, 1, 0,
                 buf_a_.mem, ACT_NONE);
        // Xn = LN2(X); X = X + mlp2(gelu(mlp1(Xn)))
        {
            Args a(k_ln_);
            a.mem(buf_a_.mem).mem(buf_b_.mem).mem(g.ln2_g).mem(g.ln2_b)
             .i32(d).i32(buf_a_.ld).f32(ln_eps);
            size_t lws = 128, gws = 128 * (size_t)T;
            enqueue("layernorm", k_ln_, 1, &gws, &lws);
        }
        run_gemm(g.mlp1, buf_b_.mem, buf_b_.ld, buf_c_,
                 g.mlp1.out, T, d, 1, nullptr, true, nullptr, 1, 0,
                 nullptr, ACT_GELU);
        run_gemm(g.mlp2, buf_c_.mem, buf_c_.ld, buf_a_,
                 d, T, g.mlp1.out, 1, nullptr, true, nullptr, 1, 0,
                 buf_a_.mem, ACT_NONE);
    }

    // Xn = final_norm(X)
    {
        Args a(k_ln_);
        a.mem(buf_a_.mem).mem(buf_b_.mem).mem(vt_fin_g_).mem(vt_fin_b_)
         .i32(d).i32(buf_a_.ld).f32(ln_eps);
        size_t lws = 128, gws = 128 * (size_t)T;
        enqueue("layernorm", k_ln_, 1, &gws, &lws);
    }
    // pooled = mean over tokens
    {
        Args a(k_meantok_);
        a.mem(buf_b_.mem).mem(stats_.mem)
         .i32(d).i32(B).i32(hw_).i32(buf_b_.ld).i32(stats_.ld);
        size_t lws = 256, gws = ((size_t)d * B + lws - 1) / lws * lws;
        enqueue("mean_tokens", k_meantok_, 1, &gws, &lws);
    }

    // Heads
    run_gemm(vt_policy_, buf_b_.mem, buf_b_.ld, p_sp_,
             1, T, d, 1, nullptr, true, nullptr, 1, 0, nullptr, ACT_NONE);
    run_gemm(vt_own_, buf_b_.mem, buf_b_.ld, o_own_,
             1, T, d, 1, nullptr, true, nullptr, 1, 0, nullptr, ACT_NONE);

    auto fc_head = [&](const GLayer& f1, const GLayer& f2, const Buf& out, int rows) {
        run_gemm(f1, stats_.mem, stats_.ld, o_mlp_,
                 f1.out, B, d, 1, nullptr, true, nullptr, 1, 0, nullptr, ACT_GELU);
        run_gemm(f2, o_mlp_.mem, o_mlp_.ld, out,
                 rows, B, f1.out, 1, nullptr, true, nullptr, 1, 0, nullptr, ACT_NONE);
    };
    fc_head(vt_val1_, vt_val2_, o_v_, 3);
    fc_head(vt_sm1_, vt_sm2_, o_sm_, 1);
    fc_head(vt_sd1_, vt_sd2_, o_sd_, 1);

    // Pack (policy spatial row 0, pass = learned constant)
    {
        Args a(k_pack_sp_);
        a.mem(p_sp_.mem).mem(nullptr).mem(o_v_.mem)
         .mem(o_sm_.mem).mem(o_sd_.mem).mem(o_own_.mem).mem(out_pack_)
         .i32(hw_).i32(0).i32(0).f32(v.pass_logit).i32(0)
         .i32(0).i32(0)
         .i32(p_sp_.ld).i32(o_v_.ld).i32(B);
        size_t lws[2] = { 32, 8 };
        size_t gws[2] = { (size_t)round_up(out_stride_, 32), (size_t)round_up(B, 8) };
        enqueue("pack", k_pack_sp_, 2, gws, lws);
    }
}

// ================================================================
// predict_batch
// ================================================================

void OpenCLComputeHandle::predict_chunk(
        const std::vector<std::vector<float>>& states,
        size_t lo, size_t hi, std::vector<Result>& out) {
    const ModelWeights& w = model_->weights();
    const int B = (int)(hi - lo);
    const int A = hw_ + 1;

    // Host-side flatten + upload
    for (int i = 0; i < B; i++) {
        const auto& s = states[lo + i];
        if ((int)s.size() != state_len_)
            throw std::runtime_error("OpenCL backend: state size mismatch (" +
                std::to_string(s.size()) + " vs " + std::to_string(state_len_) + ")");
        std::memcpy(&staging_host_[(size_t)i * state_len_], s.data(),
                    (size_t)state_len_ * 4);
    }
    CL_CHECK(clEnqueueWriteBuffer(queue_, staging_, CL_FALSE, 0,
                                  (size_t)B * state_len_ * 4,
                                  staging_host_.data(), 0, nullptr, nullptr));

    // Enqueue the graph
    if (w.trunk_style == ModelWeights::ViTTrunk) {
        forward_vit(B);
    } else if (w.trunk_style == ModelWeights::KataTrunk) {
        forward_kata_trunk(B);
        if (w.head_style == ModelWeights::Kata1Heads)
            forward_kata1_heads(B);
        else
            forward_minigo_heads(B, buf_a_);
    } else {
        forward_resnet_trunk(B);
        forward_minigo_heads(B, buf_a_);
    }

    // One blocking read of the packed outputs
    std::vector<float> host_out((size_t)B * out_stride_);
    CL_CHECK(clEnqueueReadBuffer(queue_, out_pack_, CL_TRUE, 0,
                                 host_out.size() * 4, host_out.data(),
                                 0, nullptr, nullptr));
    profile_dump();

    // Post-process per the model contract (policy stays raw logits)
    const bool kata1 = w.head_style == ModelWeights::Kata1Heads;
    for (int i = 0; i < B; i++) {
        const float* p = &host_out[(size_t)i * out_stride_];
        Result r;
        r.policy.assign(p, p + A);

        float vmax = std::max({p[A], p[A + 1], p[A + 2]});
        float ew = std::exp(p[A] - vmax);
        float el = std::exp(p[A + 1] - vmax);
        float ed = std::exp(p[A + 2] - vmax);
        r.value = (ew - el) / (ew + el + ed);

        if (kata1) {
            r.score    = p[A + 3] * 20.0f;
            r.score_sd = host_softplus(p[A + 4]) * 20.0f;
        } else {
            r.score    = p[A + 3];
            r.score_sd = host_softplus(p[A + 4]);
        }

        r.ownership.resize(hw_);
        const float* own = p + A + 5;
        if (kata1) {
            for (int j = 0; j < hw_; j++)
                r.ownership[j] = (std::tanh(own[j]) + 1.0f) * 0.5f;
        } else {
            for (int j = 0; j < hw_; j++)
                r.ownership[j] = 1.0f / (1.0f + std::exp(-own[j]));
        }
        out.push_back(std::move(r));
    }
}

std::vector<OpenCLComputeHandle::Result>
OpenCLComputeHandle::predict_batch(const std::vector<std::vector<float>>& states) {
    std::vector<Result> results;
    results.reserve(states.size());
    for (size_t lo = 0; lo < states.size(); lo += (size_t)max_batch_) {
        size_t hi = std::min(states.size(), lo + (size_t)max_batch_);
        predict_chunk(states, lo, hi, results);
    }
    return results;
}

}  // namespace minigo
#endif  // MINIGO_HAS_OPENCL
