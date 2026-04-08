#pragma once

#include "batch_evaluator.h"
#include "loaded_model.h"
#include "compute_context.h"
#include <thread>
#include <mutex>
#include <condition_variable>
#include <vector>
#include <memory>

namespace minigo {

// ----------------------------------------------------------------
// NNEvaluator — KataGo-style multi-server batching (N server threads)
//
// Each server thread:
//   1. Creates its own ComputeHandle ON that thread (KataGo pattern)
//   2. Drains the shared queue (competing consumers)
//   3. Runs predict_batch() on its GPU
//   4. Signals waiting search threads
//
// With N=1: single server thread (default, same as before)
// With N=2, devices=[0,1]: two GPUs, self-balancing batches
// With N=2, devices=[0,0]: two threads on same GPU (CPU pipelining)
// ----------------------------------------------------------------
class NNEvaluator : public BatchEvaluator {
public:
    // model:   shared CPU weights (loaded once on main thread)
    // context: shared device contexts (created once on main thread)
    // gpu_ids: one GPU index per server thread (length = number of server threads)
    // max_batch_size: cap on batch size per predict_batch call
    NNEvaluator(std::shared_ptr<LoadedModel> model,
                std::shared_ptr<ComputeContext> context,
                const std::vector<int>& gpu_ids,
                int max_batch_size);
    ~NNEvaluator() override;

    // Batch interface
    std::vector<Result>
    evaluate(const std::vector<std::vector<float>>& states) override;

    // Per-leaf blocking evaluation with pre-allocated buf (KataGo pattern)
    Result evaluate_with_buf(NNResultBuf& buf, const std::vector<float>& state) override;

    // Convenience: creates a temporary buf (for root eval, once per move)
    Result evaluate_single(const std::vector<float>& state) override;

    // Block until all server threads have created their ComputeHandles.
    // Call this after construction to ensure GPU resources are fully
    // initialized before starting another NNEvaluator on the same GPUs.
    void wait_ready();

private:
    void server_loop(int thread_id, int gpu_id);

    std::shared_ptr<LoadedModel>    model_;
    std::shared_ptr<ComputeContext> context_;
    int                             max_batch_size_;

    // Shared queue: search threads push NNResultBuf*, server threads pop
    std::mutex                    queue_mutex_;
    std::condition_variable       queue_cv_;
    std::vector<NNResultBuf*>     queue_;
    bool                          stop_ = false;

    // N server threads (one per GPU assignment)
    std::vector<std::thread>      server_threads_;

    // Ready synchronization — server threads signal when handle is created
    std::mutex                    ready_mutex_;
    std::condition_variable       ready_cv_;
    int                           handles_ready_ = 0;
    int                           num_threads_   = 0;
};

}  // namespace minigo
