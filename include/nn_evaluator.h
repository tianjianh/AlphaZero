#pragma once

#include "batch_evaluator.h"
#include "inference_engine.h"
#include <thread>
#include <mutex>
#include <condition_variable>
#include <vector>
#include <memory>

namespace minigo {

// ----------------------------------------------------------------
// NNEvaluator — KataGo-style batching server
//
// One server thread pops up to max_batch_size NNResultBuf pointers
// from a shared queue, runs ONE predict_batch() call, then signals
// each client's condvar.
//
// No timeout.  No minimum batch threshold.  The server simply takes
// whatever is in the queue when it wakes up.  Batch size is naturally
// determined by how many search threads submit leaves during the
// previous GPU call — adaptive without explicit tuning.
//
// With 128 search threads and a ~2ms GPU call, ~128 leaves accumulate
// during each GPU cycle → batch-128 → 55K states/s at full GPU util.
// ----------------------------------------------------------------
class NNEvaluator : public BatchEvaluator {
public:
    NNEvaluator(InferenceEngine* engine, int max_batch_size);
    ~NNEvaluator() override;

    // Batch interface (for compatibility / single-threaded search)
    std::vector<Result>
    evaluate(const std::vector<std::vector<float>>& states) override;

    // Per-leaf blocking evaluation with pre-allocated buf (KataGo pattern).
    // Search threads call this — the buf is created once per thread and reused.
    Result evaluate_with_buf(NNResultBuf& buf, const std::vector<float>& state) override;

    // Convenience: creates a temporary buf. Used for root eval (once per move).
    Result evaluate_single(const std::vector<float>& state) override;

private:
    void server_loop();

    InferenceEngine* engine_;
    int              max_batch_size_;

    // Shared queue: search threads push NNResultBuf*, server pops
    std::mutex                    queue_mutex_;
    std::condition_variable       queue_cv_;
    std::vector<NNResultBuf*>     queue_;   // use vector as queue (clear after drain)
    bool                          stop_ = false;

    std::thread server_thread_;
};

}  // namespace minigo
