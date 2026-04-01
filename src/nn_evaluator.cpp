#include "nn_evaluator.h"
#include <stdexcept>

namespace minigo {

NNEvaluator::NNEvaluator(InferenceEngine* engine, int max_batch_size)
    : engine_(engine), max_batch_size_(max_batch_size) {
    queue_.reserve(max_batch_size);
    server_thread_ = std::thread(&NNEvaluator::server_loop, this);
}

NNEvaluator::~NNEvaluator() {
    {
        std::lock_guard<std::mutex> lock(queue_mutex_);
        stop_ = true;
    }
    queue_cv_.notify_all();
    server_thread_.join();
}

// ── Per-leaf blocking evaluation with pre-allocated buf (KataGo pattern) ──
//
// The search thread owns the NNResultBuf and reuses it for ALL evaluations.
// The mutex + condvar are created once per thread, not per evaluation call.
// This matches KataGo's pattern: one NNResultBuf per SearchThread.
NNEvaluator::Result NNEvaluator::evaluate_with_buf(
        NNResultBuf& buf, const std::vector<float>& state) {
    buf.done = false;
    buf.state_data = state.data();
    buf.state_size = (int)state.size();

    {
        std::lock_guard<std::mutex> lock(queue_mutex_);
        queue_.push_back(&buf);
    }
    queue_cv_.notify_all();

    {
        std::unique_lock<std::mutex> lock(buf.mu);
        buf.cv.wait(lock, [&] { return buf.done; });
    }

    return { std::move(buf.policy), buf.value };
}

// Convenience wrapper — creates a temporary buf per call.
// Used only for root evaluation (once per move) and evaluate() batch path.
NNEvaluator::Result NNEvaluator::evaluate_single(const std::vector<float>& state) {
    NNResultBuf buf;
    return evaluate_with_buf(buf, state);
}

// ── Batch interface (for DirectEvaluator compatibility / VLP path) ────────
std::vector<NNEvaluator::Result>
NNEvaluator::evaluate(const std::vector<std::vector<float>>& states) {
    if (states.empty()) return {};

    int n = (int)states.size();
    std::vector<std::unique_ptr<NNResultBuf>> bufs;
    bufs.reserve(n);
    for (int i = 0; i < n; i++)
        bufs.push_back(std::make_unique<NNResultBuf>());

    {
        std::lock_guard<std::mutex> lock(queue_mutex_);
        for (int i = 0; i < n; i++) {
            bufs[i]->state_data = states[i].data();
            bufs[i]->state_size = (int)states[i].size();
            queue_.push_back(bufs[i].get());
        }
    }
    queue_cv_.notify_all();

    std::vector<Result> results;
    results.reserve(n);
    for (int i = 0; i < n; i++) {
        std::unique_lock<std::mutex> lock(bufs[i]->mu);
        bufs[i]->cv.wait(lock, [&, i] { return bufs[i]->done; });
        results.push_back({ std::move(bufs[i]->policy), bufs[i]->value });
    }
    return results;
}

// ── Server loop (KataGo's exact pattern) ──────────────────────────────────
//
// waitPopUpToN: block until ≥1 item, pop up to max_batch_size, fire.
// No timeout, no minimum threshold, no accumulation window.
//
// Batch size adapts naturally: while GPU processes batch N, search threads
// descend and submit leaves for batch N+1.  Steady-state batch ≈ num_threads.
void NNEvaluator::server_loop() {
    std::vector<NNResultBuf*> batch;
    batch.reserve(max_batch_size_);

    while (true) {
        {
            std::unique_lock<std::mutex> lock(queue_mutex_);
            queue_cv_.wait(lock, [this] { return !queue_.empty() || stop_; });
            if (stop_ && queue_.empty()) break;

            int n = std::min((int)queue_.size(), max_batch_size_);
            batch.assign(queue_.begin(), queue_.begin() + n);
            queue_.erase(queue_.begin(), queue_.begin() + n);
        }

        if (batch.empty()) continue;

        int n = (int)batch.size();
        std::vector<std::vector<float>> all_states;
        all_states.reserve(n);
        for (auto* buf : batch)
            all_states.emplace_back(buf->state_data, buf->state_data + buf->state_size);

        std::vector<Result> all_results;
        try {
            all_results = engine_->predict_batch(all_states);
        } catch (...) {
            for (auto* buf : batch) {
                std::lock_guard<std::mutex> lock(buf->mu);
                buf->done = true;
                buf->cv.notify_one();
            }
            batch.clear();
            continue;
        }

        for (int i = 0; i < n; i++) {
            NNResultBuf* buf = batch[i];
            {
                std::lock_guard<std::mutex> lock(buf->mu);
                buf->policy = std::move(all_results[i].first);
                buf->value  = all_results[i].second;
                buf->done   = true;
            }
            buf->cv.notify_one();
        }

        batch.clear();
    }
}

}  // namespace minigo
