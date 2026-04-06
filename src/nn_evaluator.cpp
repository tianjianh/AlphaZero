#include "nn_evaluator.h"
#include <iostream>
#include <stdexcept>

namespace minigo {

NNEvaluator::NNEvaluator(std::shared_ptr<LoadedModel> model,
                         std::shared_ptr<ComputeContext> context,
                         const std::vector<int>& gpu_ids,
                         int max_batch_size)
    : model_(std::move(model)),
      context_(std::move(context)),
      max_batch_size_(max_batch_size) {
    queue_.reserve(max_batch_size);

    int num_threads = (int)gpu_ids.size();
    std::cout << "NNEvaluator: " << num_threads << " server thread(s), devices=[";
    for (int i = 0; i < num_threads; i++) {
        if (i > 0) std::cout << ",";
        std::cout << gpu_ids[i];
    }
    std::cout << "], backend=" << context_->backend_name() << "\n";

    // Spawn N server threads — each creates its own ComputeHandle
    for (int i = 0; i < num_threads; i++)
        server_threads_.emplace_back(&NNEvaluator::server_loop, this, i, gpu_ids[i]);
}

NNEvaluator::~NNEvaluator() {
    {
        std::lock_guard<std::mutex> lock(queue_mutex_);
        stop_ = true;
    }
    queue_cv_.notify_all();  // wake ALL server threads
    for (auto& t : server_threads_)
        t.join();
}

// ── Per-leaf blocking evaluation with pre-allocated buf (KataGo pattern) ──
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

    return { std::move(buf.policy), buf.value, buf.score };
}

// Convenience wrapper — creates a temporary buf per call.
NNEvaluator::Result NNEvaluator::evaluate_single(const std::vector<float>& state) {
    NNResultBuf buf;
    return evaluate_with_buf(buf, state);
}

// ── Batch interface ──────────────────────────────────────────────────────
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
        results.push_back({ std::move(bufs[i]->policy), bufs[i]->value, bufs[i]->score });
    }
    return results;
}

// ── Server loop (one per server thread) ──────────────────────────────────
//
// Each thread creates its own ComputeHandle ON this thread (KataGo pattern).
// All threads drain from the same shared queue (competing consumers).
// Whichever GPU finishes first picks up the next batch — self-balancing.
void NNEvaluator::server_loop(int thread_id, int gpu_id) {
    // Create ComputeHandle ON this thread — uploads weights to GPU
    auto handle = context_->create_handle(model_.get(), gpu_id, max_batch_size_);

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

        // Flatten states
        int n = (int)batch.size();
        std::vector<std::vector<float>> all_states;
        all_states.reserve(n);
        for (auto* buf : batch)
            all_states.emplace_back(buf->state_data, buf->state_data + buf->state_size);

        // GPU call — uses THIS thread's ComputeHandle
        std::vector<ComputeHandle::Result> all_results;
        try {
            all_results = handle->predict_batch(all_states);
        } catch (...) {
            for (auto* buf : batch) {
                std::lock_guard<std::mutex> lock(buf->mu);
                buf->done = true;
                buf->cv.notify_one();
            }
            batch.clear();
            continue;
        }

        // Deliver results
        for (int i = 0; i < n; i++) {
            NNResultBuf* buf = batch[i];
            {
                std::lock_guard<std::mutex> lock(buf->mu);
                buf->policy = std::move(all_results[i].policy);
                buf->value  = all_results[i].value;
                buf->score  = all_results[i].score;
                buf->done   = true;
            }
            buf->cv.notify_one();
        }

        batch.clear();
    }
}

}  // namespace minigo
