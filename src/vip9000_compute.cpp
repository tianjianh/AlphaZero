#ifdef MINIGO_HAS_VIP9000

#include "vip9000_compute.h"
#include "loaded_model.h"

#include "vip_lite.h"

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
#define VIP_CHECK(expr, what)                                                  \
    do {                                                                       \
        vip_status_e _rc = (expr);                                             \
        if (_rc != VIP_SUCCESS) {                                              \
            std::ostringstream _os;                                            \
            _os << "VIP9000 error " << (int)_rc << " in " << (what)            \
                << " at " << __FILE__ << ":" << __LINE__;                      \
            throw std::runtime_error(_os.str());                               \
        }                                                                      \
    } while (0)

// ================================================================
// Helpers — file I/O and path resolution
// ================================================================
static std::vector<uint8_t> read_file_bytes(const std::string& path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f)
        throw std::runtime_error("VIP9000: cannot open NBG file: " + path);
    auto sz = f.tellg();
    if (sz <= 0)
        throw std::runtime_error("VIP9000: empty NBG file: " + path);
    std::vector<uint8_t> buf((size_t)sz);
    f.seekg(0);
    f.read(reinterpret_cast<char*>(buf.data()), sz);
    if (!f)
        throw std::runtime_error("VIP9000: failed to read NBG file: " + path);
    return buf;
}

static bool file_exists(const std::string& p) {
    struct stat st;
    return stat(p.c_str(), &st) == 0 && S_ISREG(st.st_mode);
}

static bool dir_exists(const std::string& p) {
    struct stat st;
    return stat(p.c_str(), &st) == 0 && S_ISDIR(st.st_mode);
}

static bool ends_with(const std::string& s, const std::string& suf) {
    return s.size() >= suf.size() &&
           s.compare(s.size() - suf.size(), suf.size(), suf) == 0;
}

// Pick the smallest pre-converted NBG batch ≥ max_batch_size.  We ship
// bs=1 (live play) and bs=4 (self-play), per A733_CONVERSION.md §5.
static int pick_compiled_batch(int max_batch_size) {
    if (max_batch_size <= 1) return 1;
    if (max_batch_size <= 4) return 4;
    std::ostringstream os;
    os << "VIP9000: only bs=1 and bs=4 NBGs are pre-converted for the A733 "
       << "(max_batch_size=" << max_batch_size << " requested).  Re-convert "
       << "the ONNX with `tools/onnx_to_a733.py` at the desired bs, or pass "
       << "--max-batch <= 4.";
    throw std::runtime_error(os.str());
}

// Resolve the .nb path from LoadedModel::model_path + chosen batch.
//
//   models/foo.onnx                 → models/foo.a733.bs<K>.fp16/network_binary.nb
//   models/foo.unshared.onnx        → models/foo.a733.bs<K>.fp16/network_binary.nb
//   models/foo.a733.bs1.unshared.onnx
//                                   → models/foo.a733.bs<K>.fp16/network_binary.nb
//   models/foo.a733.bs1.fp16/network_binary.nb
//                                   → use as-is (overrides chosen K — also lets
//                                     advanced callers force a specific NBG).
//   <dir>/                          → <dir>/network_binary.nb
static std::string resolve_nbg_path(const std::string& model_path, int chosen_bs) {
    // Direct .nb file.
    if (ends_with(model_path, ".nb")) return model_path;

    // Directory containing network_binary.nb.
    if (dir_exists(model_path)) {
        std::string p = model_path;
        if (!p.empty() && p.back() != '/') p += '/';
        p += "network_binary.nb";
        if (file_exists(p)) return p;
    }

    // Strip ONNX-related suffixes to get a "base" name, then build the
    // canonical NBG directory path.
    std::string base = model_path;
    if (ends_with(base, ".onnx"))
        base = base.substr(0, base.size() - 5);
    if (ends_with(base, ".unshared"))
        base = base.substr(0, base.size() - 9);

    // Strip ".a733.bsN.{fp16,int8}" / ".a733.bsN" if present so re-runs
    // with a different max_batch_size pick the right NBG without re-typing.
    auto strip_suffix_tag = [&]() {
        size_t pos = base.rfind(".a733.bs");
        if (pos == std::string::npos) return;
        size_t i = pos + 8;
        while (i < base.size() && std::isdigit((unsigned char)base[i])) ++i;
        if (i == pos + 8) return;  // no digits — leave alone
        // Optional ".fp16" / ".int8" precision tag.
        if (i + 5 <= base.size() && base.compare(i, 5, ".fp16") == 0) i += 5;
        else if (i + 5 <= base.size() && base.compare(i, 5, ".int8") == 0) i += 5;
        if (i == base.size()) base = base.substr(0, pos);
    };
    strip_suffix_tag();

    // Try INT8 first (faster on the VIP9000 fp pipeline when available),
    // fall back to FP16.  This matches the convention from the converter:
    // models/<base>.a733.bs<K>.{int8,fp16}/network_binary.nb.
    const char* candidates[] = { ".int8/network_binary.nb",
                                 ".fp16/network_binary.nb" };
    for (const char* suffix : candidates) {
        std::string p = base + ".a733.bs" + std::to_string(chosen_bs) + suffix;
        if (file_exists(p)) return p;
    }

    std::ostringstream os;
    os << "VIP9000: pre-converted NBG not found at "
       << base << ".a733.bs" << chosen_bs << ".{int8,fp16}/network_binary.nb."
       << "  Convert the ONNX with `bash tools/onnx_to_a733_docker.sh "
       << chosen_bs << "` (see A733_CONVERSION.md).";
    throw std::runtime_error(os.str());
}

static const char* fmt_name(vip_enum f) {
    switch (f) {
        case VIP_BUFFER_FORMAT_FP32:  return "fp32";
        case VIP_BUFFER_FORMAT_FP16:  return "fp16";
        case VIP_BUFFER_FORMAT_BFP16: return "bf16";
        case VIP_BUFFER_FORMAT_UINT8: return "u8";
        case VIP_BUFFER_FORMAT_INT8:  return "i8";
        case VIP_BUFFER_FORMAT_INT16: return "i16";
        case VIP_BUFFER_FORMAT_INT32: return "i32";
        default:                      return "?";
    }
}

// ================================================================
// IEEE-754 fp32 ↔ fp16 conversion (binary16 / "half")
//
// Vendored from VeriSilicon's vpm_run.c so the backend has no
// dependency on VIPLite's helper code (those routines aren't exposed
// in libVIPhal.so).  Round-to-nearest-even, infinities clamped to
// fp16 max (no INF emit), denormals supported.
// ================================================================
static inline uint16_t fp32_to_fp16(float in) {
    uint32_t fp32;
    std::memcpy(&fp32, &in, sizeof(fp32));
    uint32_t t1 = (fp32 & 0x80000000u) >> 16;   // sign
    uint32_t t2 = (fp32 & 0x7F800000u) >> 13;   // exponent
    uint32_t t3 = (fp32 & 0x007FE000u) >> 13;   // mantissa (no rounding)
    uint32_t t4 = (fp32 & 0x00001000u) >> 12;   // round bit
    uint16_t fp16;

    if ((fp32 & 0x7FFFFFFFu) == 0) {
        fp16 = (uint16_t)t1;                    // ±0
    } else if ((fp32 & 0x7F800000u) == 0x7F800000u) {
        // INF or NaN → preserve NaN by setting mantissa, INF clamped to max.
        if ((fp32 & 0x007FFFFFu) != 0) {
            fp16 = (uint16_t)(t1 | 0x7E00 | (t3 & 0x03FF));   // NaN
        } else {
            fp16 = (uint16_t)(t1 | 0x7BFF);                  // ±max (no INF)
        }
    } else {
        int32_t e = (int32_t)((fp32 & 0x7F800000u) >> 23) - 127 + 15;
        if (e >= 0x1F) {
            fp16 = (uint16_t)(t1 | 0x7BFF);     // overflow → ±max
        } else if (e <= 0) {
            // Denormal or underflow.
            int32_t shift = 1 - e;
            if (shift > 24) {
                fp16 = (uint16_t)t1;
            } else {
                uint32_t m = (fp32 & 0x007FFFFFu) | 0x00800000u;
                uint32_t m_shifted = m >> (shift + 13);
                uint32_t round_bit = (m >> (shift + 12)) & 1u;
                fp16 = (uint16_t)(t1 | (m_shifted + round_bit));
            }
        } else {
            uint16_t mant_round = (uint16_t)(t3 + t4);
            uint16_t exp = (uint16_t)((e << 10) & 0x7C00);
            // mant_round can carry into exp on the 0x3FF→0x400 boundary;
            // OR-add lets the carry bump exp by one tick correctly.
            fp16 = (uint16_t)(t1 | exp | mant_round);
        }
    }
    return fp16;
}

static inline float fp16_to_fp32(uint16_t in) {
    uint32_t sign = (uint32_t)(in & 0x8000u) << 16;
    uint32_t exp  = (uint32_t)(in & 0x7C00u) >> 10;
    uint32_t mant = (uint32_t)(in & 0x03FFu);
    uint32_t fp32;

    if (exp == 0) {
        if (mant == 0) {
            fp32 = sign;                        // ±0
        } else {
            // Denormal — normalize.
            int e = -1;
            do { e++; mant <<= 1; } while ((mant & 0x0400) == 0);
            mant &= 0x03FF;
            fp32 = sign | ((uint32_t)(127 - 15 - e) << 23) | (mant << 13);
        }
    } else if (exp == 0x1F) {
        // INF or NaN.
        fp32 = sign | 0x7F800000u | (mant << 13);
    } else {
        fp32 = sign | ((uint32_t)(exp + 127 - 15) << 23) | (mant << 13);
    }
    float out;
    std::memcpy(&out, &fp32, sizeof(out));
    return out;
}

// ================================================================
// Tensor I/O metadata cached after master vip_create_network.
// ================================================================
struct TensorMeta {
    uint32_t                       num_of_dims = 0;
    uint32_t                       sizes[6]   = {1,1,1,1,1,1};
    vip_enum                       data_format = 0;
    vip_enum                       quant_format = 0;
    int32_t                        fixed_point_pos = 0;
    float                          tf_scale = 1.0f;
    int32_t                        tf_zero_point = 0;
    std::string                    name;
    uint64_t                       n_elems = 0;     // product of sizes
};

// ================================================================
// Device state (one per VIP9000ComputeContext).
//
// vip_init() / vip_destroy() are process-level — we call vip_init()
// once on the first context, never destroy in the typical case (the
// driver auto-cleans on process exit).  See VIPLite v2.0 docs:
// "vip_init can be called multiple times, but should be paired with
// vip_destroy."  We use a process-level init counter so multiple
// ComputeContext instances share one driver init.
// ================================================================
struct VIP9000DeviceState {
    std::vector<uint8_t>  nbg_bytes;          // model file kept alive
    std::string           nbg_path;
    int                   model_batch = 1;    // K — fixed batch in NBG
    int                   chosen_bs   = 1;    // K we asked for at handle time

    // Master network — created once on the first handle and kept alive
    // for vip_dup_network's lifetime contract.  Each server thread
    // duplicates this for its own inference state.
    vip_network           master = nullptr;
    bool                  master_owned = false;

    uint32_t              n_inputs  = 0;
    uint32_t              n_outputs = 0;
    std::vector<TensorMeta> in_meta;
    std::vector<TensorMeta> out_meta;

    // Resolved input slot indices.
    int                   spatial_input_idx = 0;
    int                   global_input_idx  = 1;

    // Resolved output indices (index into out_meta).
    int                   policy_idx    = -1;
    int                   value_idx     = -1;
    int                   score_idx     = -1;
    int                   score_sd_idx  = -1;
    int                   ownership_idx = -1;

    // Model architecture — cross-checked against LoadedModel.
    int                   board_size            = 0;
    int                   input_channels        = 0;   // spatial C (22)
    int                   input_global_channels = 0;   // global G (19)
    int                   action_size           = 0;   // board² + 1

    // Serializes lifecycle ops (vip_create_network / vip_dup_network /
    // vip_destroy_network) — driver isn't documented as thread-safe
    // for context lifecycle.
    std::mutex            lifecycle_mu;
};

// vip_init refcount — bumped per ComputeContext, drained at destruction.
static std::mutex          g_init_mu;
static int                 g_init_count = 0;
static uint32_t            g_hw_cid     = 0;     // queried once after first init

static void global_vip_init_inc() {
    std::lock_guard<std::mutex> lk(g_init_mu);
    if (g_init_count == 0) {
        VIP_CHECK(vip_init(), "vip_init");
        uint32_t v = vip_get_version();
        std::cout << "VIPLite driver version=0x" << std::hex << v << std::dec << "\n";
        // Cache the hardware chip ID so we can validate NBG headers without
        // ping-ponging vip_query_hardware in the hot path.
        if (vip_query_hardware(VIP_QUERY_HW_PROP_CID, sizeof(g_hw_cid), &g_hw_cid)
            != VIP_SUCCESS) {
            g_hw_cid = 0;   // unknown; pre-flight check is then skipped
        } else {
            std::cout << "VIP9000 hardware chip ID=0x" << std::hex << g_hw_cid
                      << std::dec << "\n";
        }
    }
    ++g_init_count;
}
static void global_vip_init_dec() {
    std::lock_guard<std::mutex> lk(g_init_mu);
    if (--g_init_count == 0) {
        // vip_destroy is safe to call here — no networks remain since
        // ComputeContext destruction tore them down already.
        vip_destroy();
    }
}

// ================================================================
// Pre-flight NBG header validation.
//
// VIPLite returns a very generic "create_network failed status=-4"
// when an NBG was compiled for a different VIP9000 variant — and
// then NNEvaluator hangs forever because the server thread dies
// while search threads queue work that never gets drained.
//
// Catch the chip-ID mismatch up-front by parsing the NBG header
// (the `.nb` format has a stable 12-byte preamble: 4-byte magic
// "VPMN", 4-byte format version, 4-byte target chip ID).  Compare
// against the vip_query_hardware()-reported CID and fail with a
// concrete remediation pointer if they disagree.
// ================================================================
struct NbgHeader {
    uint32_t magic     = 0;
    uint32_t version   = 0;
    uint32_t target_id = 0;
    bool     valid     = false;
};

static NbgHeader parse_nbg_header(const std::vector<uint8_t>& bytes) {
    NbgHeader h;
    if (bytes.size() < 12) return h;
    auto rd = [&](size_t off) {
        return (uint32_t)bytes[off]       |
               ((uint32_t)bytes[off + 1] << 8) |
               ((uint32_t)bytes[off + 2] << 16) |
               ((uint32_t)bytes[off + 3] << 24);
    };
    h.magic     = rd(0);
    h.version   = rd(4);
    h.target_id = rd(8);
    h.valid     = (h.magic == 0x4E4D5056u);  // "VPMN" little-endian
    return h;
}

// Returns true if the NBG can plausibly run on the cached g_hw_cid.
// VeriSilicon writes the full 32-bit PID on newer toolkits and the
// short low-byte family ID (e.g. 0x15 for an older VIP9000 Nano) on
// older ones; we accept either as long as the low byte matches.
static bool nbg_target_compatible(const NbgHeader& h, uint32_t hw_cid) {
    if (!h.valid || hw_cid == 0) return true;   // unknown — let driver decide
    if (h.target_id == hw_cid)             return true;
    if ((h.target_id & 0xFFu) == (hw_cid & 0xFFu)) return true;
    return false;
}

// ================================================================
// Query and cache a tensor's metadata (input or output).
// ================================================================
static void query_tensor_meta(vip_network net, bool is_input, uint32_t idx,
                              TensorMeta& m) {
    auto Q = [&](vip_enum prop, void* val) {
        return is_input ? vip_query_input(net, idx, prop, val)
                        : vip_query_output(net, idx, prop, val);
    };

    VIP_CHECK(Q(VIP_BUFFER_PROP_NUM_OF_DIMENSION, &m.num_of_dims),   "query num_of_dims");
    VIP_CHECK(Q(VIP_BUFFER_PROP_SIZES_OF_DIMENSION, m.sizes),        "query sizes");
    VIP_CHECK(Q(VIP_BUFFER_PROP_DATA_FORMAT, &m.data_format),        "query data_format");
    VIP_CHECK(Q(VIP_BUFFER_PROP_QUANT_FORMAT, &m.quant_format),      "query quant_format");

    char name_buf[64] = {0};
    if (Q(VIP_BUFFER_PROP_NAME, name_buf) == VIP_SUCCESS) {
        name_buf[sizeof(name_buf) - 1] = 0;
        m.name = name_buf;
    }

    if (m.quant_format == VIP_BUFFER_QUANTIZE_DYNAMIC_FIXED_POINT) {
        Q(VIP_BUFFER_PROP_FIXED_POINT_POS, &m.fixed_point_pos);
    } else if (m.quant_format == VIP_BUFFER_QUANTIZE_TF_ASYMM) {
        Q(VIP_BUFFER_PROP_TF_SCALE,      &m.tf_scale);
        Q(VIP_BUFFER_PROP_TF_ZERO_POINT, &m.tf_zero_point);
    }

    m.n_elems = 1;
    for (uint32_t d = 0; d < m.num_of_dims; ++d)
        m.n_elems *= m.sizes[d];
}

// Build a vip_buffer_create_params_t from cached meta + memory_type.
static vip_buffer_create_params_t make_create_params(const TensorMeta& m) {
    vip_buffer_create_params_t p{};
    p.num_of_dims  = m.num_of_dims;
    for (uint32_t d = 0; d < m.num_of_dims; ++d) p.sizes[d] = m.sizes[d];
    p.data_format  = m.data_format;
    p.quant_format = m.quant_format;
    p.memory_type  = VIP_BUFFER_MEMORY_TYPE_DEFAULT;
    if (m.quant_format == VIP_BUFFER_QUANTIZE_DYNAMIC_FIXED_POINT) {
        p.quant_data.dfp.fixed_point_pos = m.fixed_point_pos;
    } else if (m.quant_format == VIP_BUFFER_QUANTIZE_TF_ASYMM) {
        p.quant_data.affine.scale     = m.tf_scale;
        p.quant_data.affine.zeroPoint = m.tf_zero_point;
    }
    return p;
}

// ================================================================
// Map outputs by name with substring matching, then fall back to
// ONNX-export order (kata_export_for_rknn.py order).
// ================================================================
static void map_outputs(VIP9000DeviceState& dev) {
    auto contains_ci = [](const std::string& s, const char* needle) {
        std::string ls = s;
        for (auto& c : ls) c = (char)std::tolower((unsigned char)c);
        std::string n = needle;
        return ls.find(n) != std::string::npos;
    };

    for (uint32_t i = 0; i < dev.n_outputs; ++i) {
        const std::string& nm = dev.out_meta[i].name;
        if (contains_ci(nm, "policy"))         dev.policy_idx    = (int)i;
        else if (contains_ci(nm, "ownership")) dev.ownership_idx = (int)i;
        else if (contains_ci(nm, "score_stdev") ||
                 contains_ci(nm, "score_sd"))  dev.score_sd_idx  = (int)i;
        else if (contains_ci(nm, "score_mean") ||
                (contains_ci(nm, "score") && dev.score_idx < 0))
                                                dev.score_idx     = (int)i;
        else if (contains_ci(nm, "value"))     dev.value_idx     = (int)i;
    }

    // Anything still unmapped — fall back to ONNX-export order
    // (kata_export_for_rknn.py): 0=policy, 1=value, 2=score_mean,
    // 3=score_stdev, 4=ownership.
    auto fallback = [&](int idx, int& out) {
        if (out < 0 && idx < (int)dev.n_outputs) out = idx;
    };
    fallback(0, dev.policy_idx);
    fallback(1, dev.value_idx);
    fallback(2, dev.score_idx);
    fallback(3, dev.score_sd_idx);
    fallback(4, dev.ownership_idx);
}

// Identify spatial vs global input.  Prefer name match ("state_spatial"
// / "state_global"), fall back to dim count (4-D = spatial, 2-D = global).
static void map_inputs(VIP9000DeviceState& dev) {
    if (dev.n_inputs == 1) {
        dev.spatial_input_idx = 0;
        dev.global_input_idx  = -1;
        return;
    }
    if (dev.n_inputs != 2)
        throw std::runtime_error("VIP9000: expected 1 or 2 input tensors, got "
                                 + std::to_string(dev.n_inputs));

    int sp = -1, gl = -1;
    for (uint32_t i = 0; i < dev.n_inputs; ++i) {
        const auto& m = dev.in_meta[i];
        if (m.name.find("state_spatial") != std::string::npos) sp = (int)i;
        else if (m.name.find("state_global") != std::string::npos) gl = (int)i;
    }
    if (sp < 0 || gl < 0) {
        // Fall back to dim count.
        sp = -1; gl = -1;
        for (uint32_t i = 0; i < dev.n_inputs; ++i) {
            if (dev.in_meta[i].num_of_dims == 4) sp = (int)i;
            else if (dev.in_meta[i].num_of_dims == 2) gl = (int)i;
        }
    }
    if (sp < 0 || gl < 0)
        throw std::runtime_error(
            "VIP9000: could not identify state_spatial / state_global inputs");
    dev.spatial_input_idx = sp;
    dev.global_input_idx  = gl;
}

// ================================================================
// Init the master network and query I/O.  Called under lifecycle_mu.
// ================================================================
static void init_master(VIP9000DeviceState& dev, const LoadedModel* model,
                        int max_batch_size) {
    dev.chosen_bs = pick_compiled_batch(max_batch_size);
    dev.nbg_path  = resolve_nbg_path(model->model_path, dev.chosen_bs);
    dev.nbg_bytes = read_file_bytes(dev.nbg_path);

    NbgHeader hdr = parse_nbg_header(dev.nbg_bytes);
    if (!hdr.valid) {
        std::ostringstream os;
        os << "VIP9000: " << dev.nbg_path
           << " is not a VIPLite NBG (bad magic — expected 'VPMN').";
        throw std::runtime_error(os.str());
    }
    std::cout << "VIP9000 NBG header: target=0x" << std::hex << hdr.target_id
              << " version=0x" << hdr.version << std::dec
              << " bytes=" << dev.nbg_bytes.size() << "\n";
    if (!nbg_target_compatible(hdr, g_hw_cid)) {
        std::ostringstream os;
        os << "VIP9000: NBG target chip 0x" << std::hex << hdr.target_id
           << " does not match this hardware's CID 0x" << g_hw_cid << std::dec
           << ".\n  NBG path: " << dev.nbg_path
           << "\n  This usually means the NBG was generated by a stale "
              "acuitylite version against a different VIP9000 variant.  "
              "Re-convert on x86 with the toolkit that matches the on-board "
              "viplite runtime (current: 2.0.3.2-AW-2024-08-30) — see "
              "A733_CONVERSION.md.  vip_create_network would otherwise fail "
              "with status=-4 and the NN server thread would die, causing "
              "evaluate_*() to hang forever.";
        throw std::runtime_error(os.str());
    }

    VIP_CHECK(vip_create_network(dev.nbg_bytes.data(),
                                 (uint32_t)dev.nbg_bytes.size(),
                                 VIP_CREATE_NETWORK_FROM_MEMORY,
                                 &dev.master),
              "vip_create_network(master)");
    dev.master_owned = true;

    VIP_CHECK(vip_query_network(dev.master, VIP_NETWORK_PROP_INPUT_COUNT,
                                &dev.n_inputs),
              "query INPUT_COUNT");
    VIP_CHECK(vip_query_network(dev.master, VIP_NETWORK_PROP_OUTPUT_COUNT,
                                &dev.n_outputs),
              "query OUTPUT_COUNT");

    dev.in_meta.assign(dev.n_inputs, TensorMeta{});
    for (uint32_t i = 0; i < dev.n_inputs; ++i)
        query_tensor_meta(dev.master, true, i, dev.in_meta[i]);
    dev.out_meta.assign(dev.n_outputs, TensorMeta{});
    for (uint32_t i = 0; i < dev.n_outputs; ++i)
        query_tensor_meta(dev.master, false, i, dev.out_meta[i]);

    map_inputs(dev);
    map_outputs(dev);

    // Resolve model shape from the spatial input.  VIPLite reports
    // sizes in [w, h, c, n] (innermost first) for a 4-D NCHW tensor.
    const auto& sp = dev.in_meta[dev.spatial_input_idx];
    if (sp.num_of_dims != 4)
        throw std::runtime_error("VIP9000: spatial input is not 4-D");
    int W   = (int)sp.sizes[0];
    int H   = (int)sp.sizes[1];
    int C   = (int)sp.sizes[2];
    int Nbg = (int)sp.sizes[3];
    if (H != W)
        throw std::runtime_error("VIP9000: non-square spatial input "
                                 + std::to_string(H) + "x" + std::to_string(W));
    dev.board_size     = H;
    dev.input_channels = C;
    dev.model_batch    = Nbg;
    dev.action_size    = H * W + 1;

    if (dev.global_input_idx >= 0) {
        const auto& gl = dev.in_meta[dev.global_input_idx];
        if (gl.num_of_dims != 2)
            throw std::runtime_error("VIP9000: global input is not 2-D");
        if ((int)gl.sizes[1] != dev.model_batch) {
            std::ostringstream os;
            os << "VIP9000: global input batch " << gl.sizes[1]
               << " != spatial input batch " << dev.model_batch;
            throw std::runtime_error(os.str());
        }
        dev.input_global_channels = (int)gl.sizes[0];
    }

    // Cross-check against ONNX-derived shape.
    if (dev.board_size != model->board_size ||
        dev.input_channels != model->input_channels) {
        std::ostringstream os;
        os << "VIP9000: NBG shape mismatch — NBG has " << dev.input_channels
           << "C " << dev.board_size << "x" << dev.board_size
           << " but ONNX has " << model->input_channels << "C "
           << model->board_size << "x" << model->board_size
           << ".  Re-convert with onnx_to_a733.py.";
        throw std::runtime_error(os.str());
    }
    if (dev.global_input_idx >= 0 &&
        dev.input_global_channels != model->input_global_channels) {
        std::ostringstream os;
        os << "VIP9000: KataGo global-channel mismatch — NBG has "
           << dev.input_global_channels << " but ONNX has "
           << model->input_global_channels;
        throw std::runtime_error(os.str());
    }

    std::cout << "VIP9000 NBG: " << dev.nbg_path << " bytes=" << dev.nbg_bytes.size()
              << " batch=" << dev.model_batch
              << " C=" << dev.input_channels;
    if (dev.global_input_idx >= 0) std::cout << "+G" << dev.input_global_channels;
    std::cout << " board=" << dev.board_size << "x" << dev.board_size
              << " spatial_dtype=" << fmt_name(sp.data_format)
              << "\n";
    std::cout << "VIP9000 outputs:";
    for (uint32_t i = 0; i < dev.n_outputs; ++i) {
        std::cout << " [" << i << "]"
                  << (dev.out_meta[i].name.empty() ? "(noname)" : dev.out_meta[i].name)
                  << "(" << dev.out_meta[i].n_elems
                  << "," << fmt_name(dev.out_meta[i].data_format) << ")";
    }
    std::cout << "\n";
    std::cout << "VIP9000 mapped: policy=" << dev.policy_idx
              << " value="     << dev.value_idx
              << " score="     << dev.score_idx
              << " score_sd="  << dev.score_sd_idx
              << " ownership=" << dev.ownership_idx << "\n";
}

// ================================================================
// VIP9000ComputeContext — shared state
// ================================================================
struct VIP9000ComputeContext::Impl {
    VIP9000DeviceState dev;
    std::atomic<int>   next_thread_index{0};
};

VIP9000ComputeContext::VIP9000ComputeContext(const std::vector<int>& device_ids) {
    impl_ = new Impl();
    // device_ids is informational — VIPLite exposes one logical NPU.
    // We accept multiple ids (e.g. "0,0") so --nn-server-threads N
    // can spawn N concurrent server threads on the same hardware.
    for (int id : device_ids) {
        if (id != 0) {
            std::cerr << "VIP9000: warning — ignoring non-zero device id "
                      << id << " (only one NPU is exposed)\n";
        }
    }
    global_vip_init_inc();
}

VIP9000ComputeContext::~VIP9000ComputeContext() {
    if (impl_) {
        if (impl_->dev.master_owned && impl_->dev.master) {
            std::lock_guard<std::mutex> lk(impl_->dev.lifecycle_mu);
            vip_destroy_network(impl_->dev.master);
            impl_->dev.master = nullptr;
            impl_->dev.master_owned = false;
        }
        delete impl_;
    }
    global_vip_init_dec();
}

VIP9000DeviceState& VIP9000ComputeContext::device_state() {
    return impl_->dev;
}

// ================================================================
// VIP9000ComputeHandle — per-server-thread state
// ================================================================
struct VIP9000ComputeHandle::Impl {
    VIP9000DeviceState&        dev;
    int                        thread_index = 0;

    vip_network                net = nullptr;
    bool                       net_owned = false;

    // Per-input/per-output VIP buffers (allocated once, reused per call).
    std::vector<vip_buffer>    input_buffers;
    std::vector<vip_buffer>    output_buffers;

    // Staging scratch for fp16 conversion on the host side, keyed by
    // input/output index.  Buffers are sized for a full compiled batch
    // (model_batch × elements-per-sample).  Inputs need the staging
    // for the fp32→fp16 conversion (mapped buffer is fp16); we then
    // memcpy from staging into the mapped buffer.  Skipped if the
    // tensor is already fp32.
    Impl(VIP9000DeviceState& d, int tid)
        : dev(d), thread_index(tid) {}

    ~Impl() {
        std::lock_guard<std::mutex> lk(dev.lifecycle_mu);
        for (auto b : input_buffers)  if (b) vip_destroy_buffer(b);
        for (auto b : output_buffers) if (b) vip_destroy_buffer(b);
        if (net_owned && net) vip_destroy_network(net);
    }
};

VIP9000ComputeHandle::VIP9000ComputeHandle(VIP9000DeviceState& dev,
                                           const LoadedModel* model,
                                           int max_batch_size,
                                           int thread_index) {
    impl_ = new Impl(dev, thread_index);
    auto& I = *impl_;

    // Lazy-init the master under lifecycle_mu — first handle wins, the
    // rest see master_owned=true and skip.
    {
        std::lock_guard<std::mutex> lk(dev.lifecycle_mu);
        if (!dev.master_owned)
            init_master(dev, model, max_batch_size);
    }

    // Per-thread network from the cached NBG bytes.  vip_dup_network
    // would save the ~6 MB weight footprint per thread but requires
    // the master to be vip_prepare_network'd first — which means the
    // master would also need its own input/output buffers attached.
    // Skipping the optimisation: with N=1 or 2 server threads the
    // duplicate weight RAM is negligible vs. activation memory the
    // driver allocates on prepare anyway.
    {
        std::lock_guard<std::mutex> lk(dev.lifecycle_mu);
        VIP_CHECK(vip_create_network(dev.nbg_bytes.data(),
                                     (uint32_t)dev.nbg_bytes.size(),
                                     VIP_CREATE_NETWORK_FROM_MEMORY,
                                     &I.net),
                  "vip_create_network(handle)");
        I.net_owned = true;
    }

    // Allocate input/output buffers using cached metadata.  These don't
    // depend on prepare and can be created up-front.
    I.input_buffers.assign(dev.n_inputs, nullptr);
    for (uint32_t i = 0; i < dev.n_inputs; ++i) {
        auto p = make_create_params(dev.in_meta[i]);
        VIP_CHECK(vip_create_buffer(&p, sizeof(p), &I.input_buffers[i]),
                  "vip_create_buffer(input)");
    }
    I.output_buffers.assign(dev.n_outputs, nullptr);
    for (uint32_t i = 0; i < dev.n_outputs; ++i) {
        auto p = make_create_params(dev.out_meta[i]);
        VIP_CHECK(vip_create_buffer(&p, sizeof(p), &I.output_buffers[i]),
                  "vip_create_buffer(output)");
    }

    // VIPLite contract (matches vpm_run.c): vip_prepare_network must be
    // called BEFORE vip_set_input / vip_set_output — prepare allocates
    // the command buffer and internal memory pool that set_input then
    // patches with buffer addresses.  Calling them in the wrong order
    // returns nbglk_set_input "pls prepare network firstly" / status=-9.
    VIP_CHECK(vip_prepare_network(I.net), "vip_prepare_network");

    for (uint32_t i = 0; i < dev.n_inputs; ++i)
        VIP_CHECK(vip_set_input(I.net, i, I.input_buffers[i]),
                  "vip_set_input");
    for (uint32_t i = 0; i < dev.n_outputs; ++i)
        VIP_CHECK(vip_set_output(I.net, i, I.output_buffers[i]),
                  "vip_set_output");

    std::cout << "VIP9000 handle ready: thread=" << thread_index
              << " model_bs=" << dev.model_batch << "\n";
}

VIP9000ComputeHandle::~VIP9000ComputeHandle() {
    if (impl_) delete impl_;
}

std::unique_ptr<ComputeHandle>
VIP9000ComputeContext::create_handle(const LoadedModel* model,
                                     int gpu_id, int max_batch_size) {
    (void)gpu_id;
    int tid = impl_->next_thread_index.fetch_add(1);
    return std::make_unique<VIP9000ComputeHandle>(
        impl_->dev, model, max_batch_size, tid);
}

// ================================================================
// Inference — one vip_run_network per predict_batch call.
//
// The compiled NBG has a fixed batch dim K (1 or 4).  We pad slots
// [N..K) with zeros and discard their outputs.  Anything > K throws.
//
// Parallelism across samples: only via multiple server threads each
// pinned to its own dup'd network (the NPU has one core; the driver
// queues run requests).
// ================================================================
std::vector<VIP9000ComputeHandle::Result>
VIP9000ComputeHandle::predict_batch(
        const std::vector<std::vector<float>>& states) {
    if (states.empty()) return {};

    auto& I = *impl_;
    VIP9000DeviceState& dev = I.dev;
    const int N  = (int)states.size();
    const int K  = dev.model_batch;
    const int B  = dev.board_size;
    const int HW = B * B;
    const int C  = dev.input_channels;
    const int G  = dev.input_global_channels;
    const int action_size = dev.action_size;

    if (N > K) {
        std::ostringstream os;
        os << "VIP9000: got " << N << " states but compiled model_batch="
           << K << ".  Lower --max-batch to " << K
           << " or re-convert with bs=" << N << ".";
        throw std::runtime_error(os.str());
    }

    const size_t spatial_per_sample = (size_t)C * HW;
    const size_t global_per_sample  = (size_t)G;
    const size_t per_state_floats   = spatial_per_sample +
                                      (dev.global_input_idx >= 0 ? global_per_sample : 0);

    // ── Write inputs (fp32 → fp16, packed K × per-sample, zero-pad) ─
    auto write_input = [&](int slot_idx, bool is_spatial) {
        if (slot_idx < 0) return;
        const TensorMeta& tm = dev.in_meta[slot_idx];
        vip_buffer buf = I.input_buffers[slot_idx];
        const size_t buf_bytes = vip_get_buffer_size(buf);
        void* mapped = vip_map_buffer(buf);
        if (!mapped) throw std::runtime_error("VIP9000: vip_map_buffer(input) failed");

        // Per-sample float counts (host side).
        const size_t per_sample = is_spatial ? spatial_per_sample : global_per_sample;

        if (tm.data_format == VIP_BUFFER_FORMAT_FP16) {
            uint16_t* dst = static_cast<uint16_t*>(mapped);
            const size_t total_h = (size_t)K * per_sample;
            if (buf_bytes < total_h * sizeof(uint16_t)) {
                vip_unmap_buffer(buf);
                throw std::runtime_error("VIP9000: input fp16 buffer smaller than expected");
            }
            // Real slots — convert & copy.
            for (int n = 0; n < N; ++n) {
                if (states[n].size() != per_state_floats) {
                    vip_unmap_buffer(buf);
                    std::ostringstream os;
                    os << "VIP9000: state size mismatch — got " << states[n].size()
                       << " expected " << per_state_floats;
                    throw std::runtime_error(os.str());
                }
                const float* src = states[n].data() + (is_spatial ? 0 : spatial_per_sample);
                uint16_t*    d   = dst + (size_t)n * per_sample;
                for (size_t k = 0; k < per_sample; ++k)
                    d[k] = fp32_to_fp16(src[k]);
            }
            // Pad slots [N, K) — fp16 zero is just 0x0000.
            if (N < K) {
                std::memset(dst + (size_t)N * per_sample, 0,
                            (size_t)(K - N) * per_sample * sizeof(uint16_t));
            }
        } else if (tm.data_format == VIP_BUFFER_FORMAT_FP32) {
            float* dst = static_cast<float*>(mapped);
            const size_t total_f = (size_t)K * per_sample;
            if (buf_bytes < total_f * sizeof(float)) {
                vip_unmap_buffer(buf);
                throw std::runtime_error("VIP9000: input fp32 buffer smaller than expected");
            }
            for (int n = 0; n < N; ++n) {
                const float* src = states[n].data() + (is_spatial ? 0 : spatial_per_sample);
                std::memcpy(dst + (size_t)n * per_sample, src, per_sample * sizeof(float));
            }
            if (N < K)
                std::memset(dst + (size_t)N * per_sample, 0,
                            (size_t)(K - N) * per_sample * sizeof(float));
        } else if (tm.data_format == VIP_BUFFER_FORMAT_INT8 ||
                   tm.data_format == VIP_BUFFER_FORMAT_UINT8) {
            // INT8/UINT8 with TF asymmetric_affine: q = round(x/scale)+zp,
            // clamped to the dtype range.  Pad slots [N..K) with the
            // quantised zero (encodes fp32 0 → -zp/scale rounded; for the
            // typical pad region the encoder already wrote 0.0f, but the
            // input side of kata1 has zp ≠ 0 for state_global so we have
            // to encode 0.0 explicitly rather than memset-zero the bytes).
            const float scale = tm.tf_scale > 0 ? tm.tf_scale : 1.0f;
            const int   zp    = tm.tf_zero_point;
            const bool  is_signed = (tm.data_format == VIP_BUFFER_FORMAT_INT8);
            const int   q_lo = is_signed ?  -128 : 0;
            const int   q_hi = is_signed ?   127 : 255;
            const size_t total_b = (size_t)K * per_sample;
            if (buf_bytes < total_b) {
                vip_unmap_buffer(buf);
                throw std::runtime_error("VIP9000: input int8 buffer smaller than expected");
            }
            auto quantize = [&](float x) -> int {
                int q = (int)std::lrintf(x / scale) + zp;
                if (q < q_lo) q = q_lo;
                if (q > q_hi) q = q_hi;
                return q;
            };
            const int q_zero = quantize(0.0f);
            uint8_t* dst = static_cast<uint8_t*>(mapped);
            for (int n = 0; n < N; ++n) {
                if (states[n].size() != per_state_floats) {
                    vip_unmap_buffer(buf);
                    std::ostringstream os;
                    os << "VIP9000: state size mismatch — got " << states[n].size()
                       << " expected " << per_state_floats;
                    throw std::runtime_error(os.str());
                }
                const float* src = states[n].data() + (is_spatial ? 0 : spatial_per_sample);
                uint8_t*    d   = dst + (size_t)n * per_sample;
                if (is_signed) {
                    int8_t* ds = reinterpret_cast<int8_t*>(d);
                    for (size_t k = 0; k < per_sample; ++k) ds[k] = (int8_t)quantize(src[k]);
                } else {
                    for (size_t k = 0; k < per_sample; ++k) d[k] = (uint8_t)quantize(src[k]);
                }
            }
            if (N < K) {
                if (is_signed)
                    std::memset(dst + (size_t)N * per_sample, (int8_t)q_zero,
                                (size_t)(K - N) * per_sample);
                else
                    std::memset(dst + (size_t)N * per_sample, (uint8_t)q_zero,
                                (size_t)(K - N) * per_sample);
            }
        } else {
            vip_unmap_buffer(buf);
            std::ostringstream os;
            os << "VIP9000: unsupported input data_format " << (int)tm.data_format
               << " (" << fmt_name(tm.data_format) << ") — only fp16/fp32/int8/uint8 supported";
            throw std::runtime_error(os.str());
        }
        vip_flush_buffer(buf, VIP_BUFFER_OPER_TYPE_FLUSH);
        vip_unmap_buffer(buf);
    };

    write_input(dev.spatial_input_idx, /*is_spatial=*/true);
    if (dev.global_input_idx >= 0)
        write_input(dev.global_input_idx, /*is_spatial=*/false);

    // ── Run inference (blocking) ──────────────────────────────────
    VIP_CHECK(vip_run_network(I.net), "vip_run_network");

    // ── Read outputs ──────────────────────────────────────────────
    // Per-sample float strides per output, taken from cached meta.
    auto stride_of = [&](int idx) -> size_t {
        if (idx < 0) return 0;
        const TensorMeta& tm = dev.out_meta[idx];
        return tm.n_elems / (size_t)K;     // K is the batch dim of every output
    };
    const size_t pol_stride = stride_of(dev.policy_idx);
    const size_t own_stride = stride_of(dev.ownership_idx);

    // Map an output and pull `count` floats for sample `slot` into `dst`.
    auto read_output = [&](int idx, int slot, float* dst, size_t count) {
        if (idx < 0 || count == 0) return;
        const TensorMeta& tm = dev.out_meta[idx];
        const size_t per_sample = tm.n_elems / (size_t)K;
        vip_buffer buf = I.output_buffers[idx];
        vip_flush_buffer(buf, VIP_BUFFER_OPER_TYPE_INVALIDATE);
        void* mapped = vip_map_buffer(buf);
        if (!mapped) throw std::runtime_error("VIP9000: vip_map_buffer(output) failed");

        size_t want = std::min(count, per_sample);
        if (tm.data_format == VIP_BUFFER_FORMAT_FP16) {
            const uint16_t* src = static_cast<const uint16_t*>(mapped)
                                + (size_t)slot * per_sample;
            for (size_t k = 0; k < want; ++k) dst[k] = fp16_to_fp32(src[k]);
        } else if (tm.data_format == VIP_BUFFER_FORMAT_FP32) {
            const float* src = static_cast<const float*>(mapped)
                             + (size_t)slot * per_sample;
            std::memcpy(dst, src, want * sizeof(float));
        } else if (tm.data_format == VIP_BUFFER_FORMAT_INT8 ||
                   tm.data_format == VIP_BUFFER_FORMAT_UINT8) {
            // Dequantize: x = (q - zero_point) * scale
            const float scale = tm.tf_scale;   // exact value as quantised
            const int   zp    = tm.tf_zero_point;
            if (tm.data_format == VIP_BUFFER_FORMAT_INT8) {
                const int8_t* src = static_cast<const int8_t*>(mapped)
                                  + (size_t)slot * per_sample;
                for (size_t k = 0; k < want; ++k)
                    dst[k] = (float)(src[k] - zp) * scale;
            } else {
                const uint8_t* src = static_cast<const uint8_t*>(mapped)
                                   + (size_t)slot * per_sample;
                for (size_t k = 0; k < want; ++k)
                    dst[k] = (float)((int)src[k] - zp) * scale;
            }
        } else {
            vip_unmap_buffer(buf);
            std::ostringstream os;
            os << "VIP9000: unsupported output data_format " << (int)tm.data_format
               << " (" << fmt_name(tm.data_format) << ") — only fp16/fp32/int8/uint8 supported";
            throw std::runtime_error(os.str());
        }
        // Zero anything we didn't fill (count > per_sample).
        for (size_t k = want; k < count; ++k) dst[k] = 0.0f;
        vip_unmap_buffer(buf);
    };

    auto read_scalar = [&](int idx, int slot) -> float {
        if (idx < 0) return 0.0f;
        float v = 0.0f;
        read_output(idx, slot, &v, 1);
        return v;
    };

    std::vector<Result> results(N);
    for (int n = 0; n < N; ++n) {
        Result& r = results[n];
        if (pol_stride > 0) {
            r.policy.assign((size_t)action_size, 0.0f);
            read_output(dev.policy_idx, n, r.policy.data(), (size_t)action_size);
        } else {
            r.policy.assign((size_t)action_size, 0.0f);
        }
        r.value    = read_scalar(dev.value_idx,    n);
        r.score    = read_scalar(dev.score_idx,    n);
        r.score_sd = read_scalar(dev.score_sd_idx, n);
        if (own_stride > 0) {
            r.ownership.assign((size_t)HW, 0.0f);
            read_output(dev.ownership_idx, n, r.ownership.data(), (size_t)HW);
        } else {
            r.ownership.assign((size_t)HW, 0.0f);
        }
    }
    return results;
}

}  // namespace minigo
#endif  // MINIGO_HAS_VIP9000
