#ifdef MINIGO_HAS_K3

#include "backends/k3_compute.h"
#include "model/loaded_model.h"
#include "model/onnx_loader.h"

#include <onnxruntime_cxx_api.h>

#include <sched.h>

#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef MINIGO_K3_HAS_SPACEMIT_EP
// SpacemiT execution provider registration (libspacemit_ep.so).
// Declared here so the backend builds against the plain ORT headers;
// signature matches /usr/include/spacemit_ort_env_c_api.h.
extern "C" OrtStatus* OrtSessionOptionsSpaceMITEnvInit(
    OrtSessionOptions* options,
    const char* const* provider_options_keys,
    const char* const* provider_options_values,
    size_t num_keys);
#endif

namespace minigo {

namespace {

// ── env helpers ─────────────────────────────────────────────────
std::string env_str(const char* name, const std::string& dflt) {
    const char* v = std::getenv(name);
    return (v && *v) ? std::string(v) : dflt;
}

int env_int(const char* name, int dflt) {
    const char* v = std::getenv(name);
    if (!v || !*v) return dflt;
    try { return std::stoi(v); } catch (...) { return dflt; }
}

// Parse "0-3,8,10-11" into a cpu list.
std::vector<int> parse_cpu_list(const std::string& s) {
    std::vector<int> cpus;
    std::istringstream ss(s);
    std::string tok;
    while (std::getline(ss, tok, ',')) {
        auto dash = tok.find('-');
        if (dash == std::string::npos) {
            cpus.push_back(std::stoi(tok));
        } else {
            int lo = std::stoi(tok.substr(0, dash));
            int hi = std::stoi(tok.substr(dash + 1));
            for (int c = lo; c <= hi; c++) cpus.push_back(c);
        }
    }
    return cpus;
}

// Classify the machine's cpus by cluster.  On the K3, /proc/cpuinfo
// carries "model name : Spacemit(R) A100" for the AI cores and
// "Spacemit(R) X100" for the application cores.  On non-K3 hosts both
// lists come back empty and callers fall back to generic behaviour.
struct CpuClusters {
    std::vector<int> ai;    // A100
    std::vector<int> app;   // X100 (or anything that isn't an A100)
};

CpuClusters detect_clusters() {
    CpuClusters cl;
    std::ifstream f("/proc/cpuinfo");
    if (!f) return cl;
    std::string line;
    int cur = -1;
    while (std::getline(f, line)) {
        if (line.rfind("processor", 0) == 0) {
            auto colon = line.find(':');
            if (colon != std::string::npos)
                cur = std::atoi(line.c_str() + colon + 1);
        } else if (line.rfind("model name", 0) == 0 && cur >= 0) {
            if (line.find("A100") != std::string::npos)      cl.ai.push_back(cur);
            else                                             cl.app.push_back(cur);
            cur = -1;
        }
    }
    return cl;
}

// Is the model's bulk weight storage fp16?  The A100's Tensor Cores
// implement int4/int8/fp16/bf16/fp8 only — there is NO fp32 MMA path
// (SpacemiT K3 paper, Table 1).  An fp32 graph therefore runs on the
// RVV vector unit and leaves ~10-20x on the table (measured: 2.7
// TFLOPS fp16 GEMM vs 292 GFLOPS fp32; 447 vs 22.7 evals/s on
// b10c128 9x9).  Decide from the largest initializer, so the fp32
// scalars that `keep_io_types` leaves behind don't sway the verdict.
// A quantised graph carries int8 weights; an fp16 graph carries fp16.
// Either reaches the Tensor Cores, so only a *float* verdict is a
// problem.  Skip `_sd_` tensors: that embedded state_dict is fp32
// metadata for the kernel backends and is not executed here — judging
// by raw size alone would let it outvote the real weights.
bool model_compute_is_fp32(const std::string& path) {
    constexpr int ONNX_FLOAT = 1, ONNX_FLOAT16 = 10;
    try {
        auto tensors = onnx_parser::parse_onnx_file(path);
        int64_t best_elems = -1;
        int     best_type  = 0;
        for (const auto& t : tensors) {
            if (t.name.rfind("_sd_", 0) == 0) continue;   // metadata, not compute
            int64_t n = t.total_elements();
            if (n > best_elems) { best_elems = n; best_type = t.data_type; }
        }
        if (best_elems < 0) return false;                 // nothing to judge
        (void)ONNX_FLOAT16;
        return best_type == ONNX_FLOAT;
    } catch (...) {
        return false;   // unknown — stay quiet rather than cry wolf
    }
}

}  // namespace

// ================================================================
// Device state — shared across handles (one per K3ComputeContext)
// ================================================================
struct K3DeviceState {
    // Shared ORT environment (thread pools per session, not global).
    std::unique_ptr<Ort::Env> env;

    std::string model_path;

    // EP plan
    enum class EP { SpaceMIT, CPU };
    EP  handle0_ep   = EP::CPU;    // EP for handle 0
    EP  handleN_ep   = EP::CPU;    // EP for handles 1..
    bool hetero      = false;

    int num_servers      = 1;
    int threads_per_handle = 0;    // 0 = derive per EP at handle creation

    CpuClusters clusters;
    std::vector<int> cpu_ep_cpuset;   // where cpu-EP intra-op pools live

    bool bucket_batches = true;
    bool allow_spin     = true;
    bool verbose        = false;

    // Model I/O metadata (filled by the first handle, mirrored by the rest;
    // sessions agree because they load the same file).
    std::mutex meta_mu;
    bool  meta_ready = false;
    bool  is_katago  = false;
    std::string spatial_name, global_name;
    std::vector<std::string> output_names;   // session order
    int board_size = 0, input_channels = 0, input_global_channels = 0;
    int action_size = 0;
    int policy_idx = -1, value_idx = -1, score_idx = -1,
        score_sd_idx = -1, ownership_idx = -1;

    std::atomic<int> next_thread_index{0};
};

// ================================================================
// Context
// ================================================================
struct K3ComputeContext::Impl {
    K3DeviceState dev;
};

K3ComputeContext::K3ComputeContext(const std::vector<int>& device_ids) {
    impl_ = new Impl();
    auto& dev = impl_->dev;

    dev.env = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "minigo-k3");
    dev.num_servers = std::max<size_t>(1, device_ids.size());
    dev.clusters    = detect_clusters();

    // ── EP selection ────────────────────────────────────────────
    std::string ep = env_str("MINIGO_K3_EP", "");
#ifdef MINIGO_K3_HAS_SPACEMIT_EP
    const bool have_spacemit = true;
#else
    const bool have_spacemit = false;
#endif
    if (ep.empty()) ep = have_spacemit ? "spacemit" : "cpu";

    if (ep == "spacemit" && !have_spacemit) {
        std::cerr << "K3: MINIGO_K3_EP=spacemit but built without the "
                     "SpacemiT EP — falling back to cpu\n";
        ep = "cpu";
    }
    if (ep == "hetero" && !have_spacemit) {
        std::cerr << "K3: MINIGO_K3_EP=hetero but built without the "
                     "SpacemiT EP — falling back to cpu\n";
        ep = "cpu";
    }

    using EP = K3DeviceState::EP;
    if (ep == "spacemit")      { dev.handle0_ep = EP::SpaceMIT; dev.handleN_ep = EP::SpaceMIT; }
    else if (ep == "hetero")   { dev.handle0_ep = EP::SpaceMIT; dev.handleN_ep = EP::CPU; dev.hetero = true; }
    else if (ep == "cpu")      { dev.handle0_ep = EP::CPU;      dev.handleN_ep = EP::CPU; }
    else {
        throw std::runtime_error("K3: unknown MINIGO_K3_EP '" + ep +
                                 "' (use spacemit, cpu, or hetero)");
    }

    // ── thread budget ───────────────────────────────────────────
    dev.threads_per_handle = env_int("MINIGO_K3_THREADS", 0);

    // Default thread count for SpaceMIT handles: split the AI cores
    // across the spacemit handles.  The EP enforces (ai threads <=
    // AI cores), so derive from the detected cluster.
    int n_ai = (int)dev.clusters.ai.size();
    if (dev.handle0_ep == EP::SpaceMIT && dev.threads_per_handle <= 0 && n_ai > 0) {
        int spacemit_handles = dev.hetero ? 1 : dev.num_servers;
        dev.threads_per_handle = std::max(1, n_ai / std::max(1, spacemit_handles));
    }

    // The EP sizes its AI pool from SPACEMIT_EP_INTRA_THREAD_NUM (or the
    // session's intra-op count).  Set it process-wide unless the user did.
    if (dev.handle0_ep == EP::SpaceMIT && dev.threads_per_handle > 0 &&
        !std::getenv("SPACEMIT_EP_INTRA_THREAD_NUM")) {
        setenv("SPACEMIT_EP_INTRA_THREAD_NUM",
               std::to_string(dev.threads_per_handle).c_str(), 0);
    }

    // cpu-EP pools live on the application cores by default.
    std::string cpuset = env_str("MINIGO_K3_CPUSET", "");
    if (!cpuset.empty())                dev.cpu_ep_cpuset = parse_cpu_list(cpuset);
    else if (!dev.clusters.app.empty()) dev.cpu_ep_cpuset = dev.clusters.app;

    dev.bucket_batches = env_int("MINIGO_K3_BUCKET", 1) != 0;
    dev.allow_spin     = env_int("MINIGO_K3_SPIN", 1) != 0;
    dev.verbose        = env_int("MINIGO_K3_VERBOSE", 0) != 0;

    for (int id : device_ids) {
        if (id != 0)
            std::cerr << "K3: warning — ignoring non-zero device id " << id
                      << " (single logical device; servers are sliced internally)\n";
    }

    std::cout << "K3 backend: onnxruntime " << Ort::GetVersionString()
              << ", ep=" << ep
              << ", servers=" << dev.num_servers
              << ", threads/handle=" << (dev.threads_per_handle > 0
                                         ? std::to_string(dev.threads_per_handle)
                                         : std::string("auto"))
              << ", ai_cores=" << n_ai
              << ", bucket=" << (dev.bucket_batches ? "on" : "off") << "\n";
}

K3ComputeContext::~K3ComputeContext() {
    delete impl_;
}

K3DeviceState& K3ComputeContext::device_state() { return impl_->dev; }

// ================================================================
// Handle
// ================================================================
struct K3ComputeHandle::Impl {
    K3DeviceState& dev;
    std::unique_ptr<Ort::Session> session;
    Ort::MemoryInfo mem_info{nullptr};

    K3DeviceState::EP ep = K3DeviceState::EP::CPU;
    int thread_index = 0;
    int max_batch    = 1;

    // I/O names as C strings for Run() (point into dev after meta init).
    std::vector<const char*> input_names;
    std::vector<const char*> output_names;

    // Staging buffers (sized to the largest bucket seen).
    std::vector<float> stage_spatial;
    std::vector<float> stage_global;

    explicit Impl(K3DeviceState& d) : dev(d) {}
};

// Extract model I/O metadata from a session (first handle only).
static void init_metadata(K3DeviceState& dev, Ort::Session& session,
                          const LoadedModel* model) {
    Ort::AllocatorWithDefaultOptions alloc;

    size_t n_in = session.GetInputCount();
    if (n_in != 1 && n_in != 2) {
        throw std::runtime_error("K3: expected 1 (MiniGo) or 2 (KataGo) "
                                 "graph inputs, got " + std::to_string(n_in));
    }

    int spatial_i = -1, global_i = -1;
    std::vector<std::string> in_names(n_in);
    std::vector<std::vector<int64_t>> in_dims(n_in);
    for (size_t i = 0; i < n_in; i++) {
        in_names[i] = session.GetInputNameAllocated(i, alloc).get();
        in_dims[i]  = session.GetInputTypeInfo(i)
                          .GetTensorTypeAndShapeInfo().GetShape();
        if (in_dims[i].size() == 4)      spatial_i = (int)i;
        else if (in_dims[i].size() == 2) global_i  = (int)i;
    }
    if (spatial_i < 0)
        throw std::runtime_error("K3: no 4-D spatial input tensor found");

    dev.spatial_name = in_names[spatial_i];
    const auto& sd = in_dims[spatial_i];   // [N, C, H, W]
    dev.input_channels = sd[1] > 0 ? (int)sd[1] : model->input_channels;
    dev.board_size     = sd[2] > 0 ? (int)sd[2] : model->board_size;

    dev.is_katago = (n_in == 2);
    if (dev.is_katago) {
        if (global_i < 0)
            throw std::runtime_error("K3: dual-input model but no 2-D "
                                     "global input tensor found");
        dev.global_name = in_names[global_i];
        const auto& gd = in_dims[global_i];  // [N, G]
        dev.input_global_channels =
            gd[1] > 0 ? (int)gd[1] : model->input_global_channels;
    }

    // Sanity-check against the ONNX-parser metadata (same file).
    if (dev.board_size != model->board_size ||
        dev.input_channels != model->input_channels) {
        std::ostringstream os;
        os << "K3: model shape mismatch — session sees "
           << dev.input_channels << "C " << dev.board_size << "x" << dev.board_size
           << " but LoadedModel has " << model->input_channels << "C "
           << model->board_size << "x" << model->board_size;
        throw std::runtime_error(os.str());
    }
    dev.action_size = dev.board_size * dev.board_size + 1;

    // Outputs: map by exported name, with size fallback for the two
    // essentials (policy = largest, value = smallest).
    size_t n_out = session.GetOutputCount();
    dev.output_names.resize(n_out);
    std::vector<int64_t> out_elems(n_out, 0);
    for (size_t i = 0; i < n_out; i++) {
        dev.output_names[i] = session.GetOutputNameAllocated(i, alloc).get();
        auto dims = session.GetOutputTypeInfo(i)
                        .GetTensorTypeAndShapeInfo().GetShape();
        int64_t elems = 1;
        for (auto d : dims) elems *= (d > 0 ? d : 1);
        out_elems[i] = elems;

        const std::string& nm = dev.output_names[i];
        auto starts = [&](const char* p) { return nm.rfind(p, 0) == 0; };
        if      (starts("policy_logits")) dev.policy_idx    = (int)i;
        else if (starts("value"))         dev.value_idx     = (int)i;
        else if (starts("score_mean"))    dev.score_idx     = (int)i;
        else if (starts("score_stdev"))   dev.score_sd_idx  = (int)i;
        else if (starts("ownership"))     dev.ownership_idx = (int)i;
        else if (starts("score") && dev.score_idx < 0) dev.score_idx = (int)i;
    }
    if (dev.policy_idx < 0 || dev.value_idx < 0) {
        int biggest = -1, smallest = -1;
        int64_t bmax = -1, smin = INT64_MAX;
        for (size_t i = 0; i < n_out; i++) {
            if (out_elems[i] > bmax) { bmax = out_elems[i]; biggest  = (int)i; }
            if (out_elems[i] < smin) { smin = out_elems[i]; smallest = (int)i; }
        }
        if (dev.policy_idx < 0) dev.policy_idx = biggest;
        if (dev.value_idx  < 0) dev.value_idx  = smallest;
    }

    if (model_compute_is_fp32(dev.model_path)) {
        std::cerr
            << "K3: WARNING — this model computes in fp32.  The A100 Tensor Cores\n"
               "    implement int4/int8/fp16/bf16/fp8 only, so an fp32 graph runs\n"
               "    on the RVV vector unit and leaves ~10-100x on the table\n"
               "    (b10c128 9x9 @ B=16: fp32 22.7, fp16 1954, int8 2982 evals/s).\n"
               "    Convert once with:\n"
               "        python3 tools/onnx_to_fp16.py " << dev.model_path << "\n"
               "        python3 tools/onnx_to_int8.py " << dev.model_path
            << " --calib <dir>\n";
    }

    std::cout << "K3 model: " << dev.model_path
              << " C=" << dev.input_channels;
    if (dev.is_katago) std::cout << "+G" << dev.input_global_channels;
    std::cout << " board=" << dev.board_size << "x" << dev.board_size
              << " format=" << (dev.is_katago ? "katago" : "minigo") << "\n";
    std::cout << "K3 outputs:";
    for (auto& n : dev.output_names) std::cout << " " << n;
    std::cout << "  (policy=" << dev.policy_idx
              << " value=" << dev.value_idx
              << " score=" << dev.score_idx
              << " score_sd=" << dev.score_sd_idx
              << " ownership=" << dev.ownership_idx << ")\n";
}

K3ComputeHandle::K3ComputeHandle(K3DeviceState& dev, const LoadedModel* model,
                                 int max_batch_size, int thread_index) {
    impl_ = new Impl(dev);
    auto& I = *impl_;
    I.thread_index = thread_index;
    I.max_batch    = std::max(1, max_batch_size);

    using EP = K3DeviceState::EP;
    I.ep = (thread_index == 0) ? dev.handle0_ep : dev.handleN_ep;

    if (dev.model_path.empty()) dev.model_path = model->model_path;

    // ── session options ─────────────────────────────────────────
    auto build_session = [&](EP ep) -> std::unique_ptr<Ort::Session> {
        Ort::SessionOptions so;
        so.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        so.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);
        so.SetInterOpNumThreads(1);

        // Pin the batch axis to a literal BEFORE the session is built.
        //
        // Handing the EP a concrete tensor at Run time is not enough: it
        // plans its tile graph from the *declared* shape, and a symbolic
        // batch makes it fall back to a batch-1 plan it then loops
        // (measured 4x: 491 vs 1954 evals/s on b10c128 9x9 fp16 @ B=16).
        // Overriding the free dimension bakes the real batch into the
        // graph, so the planner tiles for it.
        //
        // The axis name comes from the exporter: scripts/export_onnx.py
        // and tools/katago_to_onnx.py emit "batch"; the ONNX model-zoo
        // nets use "N".  Overriding a name the graph does not have is a
        // no-op, so listing the known spellings is safe.
        //
        // Only when bucketing (the default), where predict_batch always
        // presents exactly max_batch.  With MINIGO_K3_BUCKET=0 the batch
        // varies per call, so pinning it here would make ORT reject every
        // batch != max_batch ("Got invalid dimensions for input").
        if (dev.bucket_batches)
            for (const char* dim : {"batch", "batch_size", "N"})
                so.AddFreeDimensionOverrideByName(dim, I.max_batch);

        if (ep == EP::SpaceMIT) {
#ifdef MINIGO_K3_HAS_SPACEMIT_EP
            // Size the EP's AI-thread pool from the SESSION, not the
            // environment.  SPACEMIT_EP_INTRA_THREAD_NUM is read during
            // the EP library's static init — long before any
            // ComputeContext exists — so a setenv() here would be
            // ignored, and the EP would quietly run on ONE A100 core
            // (measured 4x: 510 vs 1954 evals/s on b10c128 9x9 fp16).
            // The EP pins one pool thread per A100 core itself, via
            // /proc/set_ai_thread.
            int threads = dev.threads_per_handle > 0
                        ? dev.threads_per_handle
                        : std::max(1, (int)dev.clusters.ai.size());
            so.SetIntraOpNumThreads(threads);
            Ort::ThrowOnError(OrtSessionOptionsSpaceMITEnvInit(
                so, nullptr, nullptr, 0));
#else
            throw std::runtime_error("K3: SpaceMIT EP not compiled in");
#endif
        } else {
            // cpu EP: intra-op pool on an X100 slice.  The calling
            // (server) thread is pool member 0 — pin it to the slice's
            // first cpu and hand the rest to ORT's affinity string.
            int threads = dev.threads_per_handle;
            const auto& cpuset = dev.cpu_ep_cpuset;
            std::vector<int> slice;
            if (!cpuset.empty()) {
                // Slice the cpuset across cpu-EP handles (hetero: handles
                // 1.. are the cpu ones; plain cpu: all handles).
                int cpu_handles = dev.hetero
                    ? std::max(1, dev.num_servers - 1)
                    : dev.num_servers;
                int slice_idx   = dev.hetero ? thread_index - 1 : thread_index;
                if (slice_idx < 0) slice_idx = 0;
                int per = std::max(1, (int)cpuset.size() / std::max(1, cpu_handles));
                int lo  = (slice_idx * per) % (int)cpuset.size();
                for (int k = 0; k < per; k++)
                    slice.push_back(cpuset[(lo + k) % (int)cpuset.size()]);
                if (threads <= 0) threads = (int)slice.size();
            }
            if (threads <= 0) threads = 4;
            so.SetIntraOpNumThreads(threads);
            so.AddConfigEntry("session.intra_op.allow_spinning",
                              dev.allow_spin ? "1" : "0");

            if (!slice.empty()) {
                // Bind the server thread itself to the slice.
                cpu_set_t cs;
                CPU_ZERO(&cs);
                for (int c : slice) CPU_SET(c, &cs);
                if (sched_setaffinity(0, sizeof(cs), &cs) != 0)
                    std::cerr << "K3: warning — could not pin server thread "
                              << thread_index << " to its cpu slice\n";
                // Pool threads 1..threads-1 (main thread is not listed).
                if (threads > 1) {
                    std::ostringstream aff;
                    for (int t = 1; t < threads; t++) {
                        if (t > 1) aff << ";";
                        aff << slice[t % slice.size()];
                    }
                    so.AddConfigEntry("session.intra_op_thread_affinities",
                                      aff.str().c_str());
                }
            }
        }
        return std::make_unique<Ort::Session>(*dev.env, dev.model_path.c_str(), so);
    };

    try {
        I.session = build_session(I.ep);
    } catch (const std::exception& e) {
        if (I.ep == EP::SpaceMIT) {
            std::cerr << "K3: SpaceMIT EP session failed (" << e.what()
                      << ") — retrying with cpu EP\n";
            I.ep = EP::CPU;
            I.session = build_session(I.ep);
        } else {
            throw;
        }
    }

    // ── metadata (once) ─────────────────────────────────────────
    {
        std::lock_guard<std::mutex> lk(dev.meta_mu);
        if (!dev.meta_ready) {
            init_metadata(dev, *I.session, model);
            dev.meta_ready = true;
        }
    }

    I.mem_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    I.input_names.push_back(dev.spatial_name.c_str());
    if (dev.is_katago) I.input_names.push_back(dev.global_name.c_str());
    for (auto& n : dev.output_names) I.output_names.push_back(n.c_str());

    int HW = dev.board_size * dev.board_size;
    I.stage_spatial.resize((size_t)I.max_batch * dev.input_channels * HW);
    if (dev.is_katago)
        I.stage_global.resize((size_t)I.max_batch * dev.input_global_channels);

    // Compile the tile graph at the ONE shape this handle will ever use.
    //
    // The SpacemiT EP lowers its subgraph to a tile graph on the first
    // Run and caches it against that shape — it does not recompile.  A
    // later, larger batch is then served by the batch-1 kernel (measured
    // 4x slow: 489 vs 1954 evals/s on b10c128 9x9 fp16) or, once the EP
    // hands a frozen batch-1 tensor to a CPU-EP node it did not absorb,
    // fails outright ("cannot be reshaped to the requested shape").  So
    // predict_batch always pads to exactly max_batch, and we prime the
    // cache here rather than let the first real request decide.
    if (I.ep == EP::SpaceMIT) {
        try {
            std::vector<std::vector<float>> warm(
                I.max_batch,
                std::vector<float>((size_t)dev.input_channels * HW +
                                   (dev.is_katago ? dev.input_global_channels : 0),
                                   0.0f));
            predict_batch(warm);
        } catch (const std::exception& e) {
            std::cerr << "K3: warm-up at batch " << I.max_batch
                      << " failed (" << e.what() << ")\n";
        }
    }

    std::cout << "K3 handle ready: thread=" << thread_index
              << " ep=" << (I.ep == EP::SpaceMIT ? "spacemit" : "cpu")
              << " fixed_batch=" << I.max_batch << "\n";
}

K3ComputeHandle::~K3ComputeHandle() {
    delete impl_;
}

std::unique_ptr<ComputeHandle>
K3ComputeContext::create_handle(const LoadedModel* model,
                                int gpu_id, int max_batch_size) {
    (void)gpu_id;   // single logical device (RKNN pattern)
    int tid = impl_->dev.next_thread_index.fetch_add(1);
    return std::make_unique<K3ComputeHandle>(impl_->dev, model,
                                             max_batch_size, tid);
}

// ================================================================
// Inference
// ================================================================
std::vector<K3ComputeHandle::Result>
K3ComputeHandle::predict_batch(const std::vector<std::vector<float>>& states) {
    if (states.empty()) return {};

    auto& I   = *impl_;
    auto& dev = I.dev;
    const int N  = (int)states.size();
    const int HW = dev.board_size * dev.board_size;
    const int C  = dev.input_channels;
    const int G  = dev.input_global_channels;
    const int A  = dev.action_size;
    const size_t spatial_floats = (size_t)C * HW;
    const size_t per_state_floats =
        spatial_floats + (dev.is_katago ? (size_t)G : 0);

    // ONE shape, always.  See the warm-up in the constructor: the EP
    // caches its tile graph against the first shape it compiles, so
    // every call must present that same batch.  Slots [N..M) are zero
    // and their outputs discarded.
    //
    // The waste at low occupancy is real but bounded, and it is the
    // same bargain the RKNN backend strikes with its compiled
    // model_batch.  MINIGO_K3_BUCKET=0 opts out (dynamic batch) — only
    // useful for debugging, since it forfeits the tensor-core kernel.
    const int M = dev.bucket_batches ? I.max_batch : N;
    if (N > M) {
        std::ostringstream os;
        os << "K3: got " << N << " states but the handle is compiled for batch "
           << M << ".  Lower --max-batch or raise it to cover this batch.";
        throw std::runtime_error(os.str());
    }

    if (I.stage_spatial.size() < (size_t)M * spatial_floats)
        I.stage_spatial.resize((size_t)M * spatial_floats);
    if (dev.is_katago && I.stage_global.size() < (size_t)M * G)
        I.stage_global.resize((size_t)M * G);

    std::memset(I.stage_spatial.data(), 0,
                (size_t)M * spatial_floats * sizeof(float));
    if (dev.is_katago)
        std::memset(I.stage_global.data(), 0, (size_t)M * G * sizeof(float));

    for (int n = 0; n < N; n++) {
        const auto& s = states[n];
        if (s.size() != per_state_floats) {
            std::ostringstream os;
            os << "K3: state size mismatch — got " << s.size()
               << " expected " << per_state_floats;
            throw std::runtime_error(os.str());
        }
        std::memcpy(I.stage_spatial.data() + (size_t)n * spatial_floats,
                    s.data(), spatial_floats * sizeof(float));
        if (dev.is_katago)
            std::memcpy(I.stage_global.data() + (size_t)n * G,
                        s.data() + spatial_floats, (size_t)G * sizeof(float));
    }

    // ── build input tensors ─────────────────────────────────────
    std::vector<Ort::Value> inputs;
    int64_t sdims[4] = {M, C, dev.board_size, dev.board_size};
    inputs.push_back(Ort::Value::CreateTensor<float>(
        I.mem_info, I.stage_spatial.data(), (size_t)M * spatial_floats,
        sdims, 4));
    if (dev.is_katago) {
        int64_t gdims[2] = {M, G};
        inputs.push_back(Ort::Value::CreateTensor<float>(
            I.mem_info, I.stage_global.data(), (size_t)M * G, gdims, 2));
    }

    auto outputs = I.session->Run(Ort::RunOptions{nullptr},
                                  I.input_names.data(), inputs.data(),
                                  inputs.size(),
                                  I.output_names.data(),
                                  I.output_names.size());

    // ── demux outputs ───────────────────────────────────────────
    auto stride_of = [&](int idx) -> int {
        if (idx < 0) return 0;
        auto info = outputs[idx].GetTensorTypeAndShapeInfo();
        return (int)(info.GetElementCount() / (size_t)M);
    };
    auto slot_ptr = [&](int idx, int slot) -> const float* {
        return outputs[idx].GetTensorData<float>() + (size_t)slot * stride_of(idx);
    };

    const int pol_stride = stride_of(dev.policy_idx);
    const int own_stride = stride_of(dev.ownership_idx);

    std::vector<Result> results(N);
    for (int n = 0; n < N; n++) {
        Result& r = results[n];

        if (dev.policy_idx >= 0 && pol_stride > 0) {
            const float* p = slot_ptr(dev.policy_idx, n);
            int count = std::min(A, pol_stride);
            r.policy.assign(p, p + count);
            if (count < A) r.policy.resize(A, 0.0f);
        } else {
            r.policy.assign(A, 0.0f);
        }

        r.value    = dev.value_idx    >= 0 ? slot_ptr(dev.value_idx,    n)[0] : 0.0f;
        r.score    = dev.score_idx    >= 0 ? slot_ptr(dev.score_idx,    n)[0] : 0.0f;
        r.score_sd = dev.score_sd_idx >= 0 ? slot_ptr(dev.score_sd_idx, n)[0] : 0.0f;

        if (dev.ownership_idx >= 0 && own_stride > 0) {
            const float* p = slot_ptr(dev.ownership_idx, n);
            int count = std::min(HW, own_stride);
            r.ownership.assign(p, p + count);
            if (count < HW) r.ownership.resize(HW, 0.0f);
        } else {
            r.ownership.assign(HW, 0.0f);
        }
    }

    return results;
}

}  // namespace minigo
#endif  // MINIGO_HAS_K3
