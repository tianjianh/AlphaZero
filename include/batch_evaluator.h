#pragma once

#include <condition_variable>
#include <mutex>
#include <vector>
#include <utility>

namespace minigo {

// ================================================================
// NNResultBuf — per-search-thread evaluation request (KataGo pattern)
//
// Each search thread owns one of these (stack-allocated, reused).
// The thread fills `state`, pushes `this` into the NNEvaluator queue,
// then blocks on `cv`.  The server fills `policy`/`value`, sets `done`,
// and signals `cv`.  Zero heap allocation per evaluation.
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
    float              value = 0.0f;

    void reset() { done = false; }
};

// ================================================================
// BatchEvaluator — abstract NN evaluation interface
// ================================================================
class BatchEvaluator {
public:
    virtual ~BatchEvaluator() = default;

    using Result = std::pair<std::vector<float>, float>;

    // Batch evaluation (blocks until ready). Used by single-threaded search.
    virtual std::vector<Result>
    evaluate(const std::vector<std::vector<float>>& states) = 0;

    // Per-leaf blocking evaluation using a pre-allocated NNResultBuf.
    // The caller (search thread) owns the buf and reuses it for ALL
    // evaluations during the thread's lifetime — matching KataGo's pattern
    // of one NNResultBuf per SearchThread.  Zero mutex/condvar creation
    // per evaluation call.
    //
    // Default: falls through to evaluate({state}) for DirectEvaluator.
    // NNEvaluator overrides with the queue+condvar pattern.
    virtual Result evaluate_with_buf(NNResultBuf& buf, const std::vector<float>& state) {
        (void)buf;  // DirectEvaluator doesn't use the buf
        auto results = evaluate({state});
        return std::move(results[0]);
    }

    // Convenience wrapper (creates a temporary buf — use only for infrequent
    // calls like root evaluation, NOT in the hot search loop).
    virtual Result evaluate_single(const std::vector<float>& state) {
        auto results = evaluate({state});
        return std::move(results[0]);
    }
};

}  // namespace minigo
