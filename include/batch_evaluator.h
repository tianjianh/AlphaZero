#pragma once

#include "compute_context.h"
#include "game.h"
#include <condition_variable>
#include <mutex>
#include <vector>

namespace minigo {

// ================================================================
// NNResultBuf — per-search-thread evaluation request (KataGo pattern)
//
// Each search thread owns one of these (stack-allocated, reused).
// The thread fills `state`, pushes `this` into the NNEvaluator queue,
// then blocks on `cv`.  The server fills `policy`/`value`, sets `done`,
// and signals `cv`.  Zero heap allocation per evaluation.
//
// The per-buf mutex+cv is the OUTBOUND half of the unified server
// design, not a per-leaf leftover: the shared NNRequestQueue funnels
// requests in with one condvar for all servers, and this cv routes
// each result back to exactly the one blocked owner thread.  A single
// shared results-cv would have to notify_all every search thread on
// every batch (thundering herd) — same reason KataGo pairs its
// queryQueue with a per-NNResultBuf clientWaitingForResult.
// ================================================================
struct NNResultBuf {
    // Input (filled by search thread before submitting)
    const float* state_data = nullptr;
    int          state_size = 0;

    // Synchronization (one mutex+condvar per search thread)
    std::mutex              mu;
    std::condition_variable cv;
    bool                    done = false;

    // Output (filled by server thread)
    std::vector<float> policy;
    float              value    = 0.0f;
    float              score    = 0.0f;
    float              score_sd = 0.0f;
    std::vector<float> ownership;
};

// ================================================================
// BatchEvaluator — abstract NN evaluation interface
// ================================================================
class BatchEvaluator {
public:
    virtual ~BatchEvaluator() = default;

    using Result = NNOutput;

    // Direct batch submission (blocks until every result is ready).
    // Production callers: the benchmark's batch-throughput sweep.  Also
    // the one pure-virtual, so minimal stub evaluators (tests) only
    // implement this and inherit the two fallthroughs below.
    virtual std::vector<Result>
    evaluate(const std::vector<std::vector<float>>& states) = 0;

    // Per-leaf blocking evaluation using a pre-allocated NNResultBuf.
    // The caller (search thread) owns the buf and reuses it for ALL
    // evaluations during the thread's lifetime — matching KataGo's pattern
    // of one NNResultBuf per SearchThread.  Zero mutex/condvar creation
    // per evaluation call.  This is THE hot path: every playout's leaf
    // eval goes through here.
    //
    // Default falls through to evaluate({state}) so stub evaluators
    // work; NNEvaluator overrides with the queue + per-buf-cv handoff.
    virtual Result evaluate_with_buf(NNResultBuf& buf, const std::vector<float>& state) {
        (void)buf;  // fallthrough path doesn't need the buf
        auto results = evaluate({state});
        return std::move(results[0]);
    }

    // Single-state convenience (temporary stack buf — fine for
    // once-per-move callers like the MCTS root eval, NOT for the
    // per-playout loop, which must reuse its thread's buf).
    virtual Result evaluate_single(const std::vector<float>& state) {
        auto results = evaluate({state});
        return std::move(results[0]);
    }

    // Encode a game position into the format this evaluator's model
    // expects. Default body is MiniGo's 17-channel encoder; the
    // NNEvaluator override dispatches on model format and routes
    // KataGo-format models to the V7 encoder. Routing the dispatch
    // through the evaluator (rather than every call site) keeps MCTS
    // format-agnostic.
    virtual void encode_state(const GoGame& game, std::vector<float>& out) const {
        game.encode(out);
    }
};

}  // namespace minigo
