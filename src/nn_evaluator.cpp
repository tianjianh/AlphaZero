#include "nn_evaluator.h"
#include <algorithm>
#include <iostream>
#include <stdexcept>

namespace minigo {

NNEvaluator::NNEvaluator(std::shared_ptr<LoadedModel> model,
                         std::shared_ptr<ComputeContext> context,
                         const std::vector<int>& gpu_ids,
                         int max_batch_size)
    : model_(std::move(model)),
      context_(std::move(context)),
      max_batch_size_(max_batch_size),
      // Ring-buffer capacity: smallest power-of-two ≥ KataGo's
      // max_batch_size * 4 * num_server_threads heuristic (see
      // nneval.cpp:156).  Power-of-two size lets the ring use
      // bitwise AND for modulo on push/pop.  Hard-capped: exceeding
      // it would indicate the producer upper-bound assumption is
      // wrong (expected: num_search_threads × num_games ≪ capacity).
      queue_(next_pow2_ge(
          (size_t)std::max(1, max_batch_size) * 4 *
          std::max<size_t>(1, gpu_ids.size()))) {
    int num_threads = (int)gpu_ids.size();

    std::cout << "NNEvaluator: " << num_threads << " server thread(s), devices=[";
    for (int i = 0; i < num_threads; i++) {
        if (i > 0) std::cout << ",";
        std::cout << gpu_ids[i];
    }
    std::cout << "], backend=" << context_->backend_name() << "\n";

    // Spawn N server threads — each creates its own ComputeHandle.
    num_threads_ = num_threads;
    for (int i = 0; i < num_threads; i++)
        server_threads_.emplace_back(&NNEvaluator::server_loop, this, i, gpu_ids[i]);
}

void NNEvaluator::wait_ready() {
    std::unique_lock<std::mutex> lock(ready_mutex_);
    ready_cv_.wait(lock, [this] { return handles_ready_ >= num_threads_; });
}

NNEvaluator::~NNEvaluator() {
    queue_.close();            // wakes all server threads to observe closed state
    for (auto& t : server_threads_)
        t.join();
}

// ── Per-leaf blocking evaluation with pre-allocated buf (KataGo pattern) ──
NNEvaluator::Result NNEvaluator::evaluate_with_buf(
        NNResultBuf& buf, const std::vector<float>& state) {
    buf.done = false;
    buf.state_data = state.data();
    buf.state_size = (int)state.size();

    queue_.push(&buf);

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

    // Collect raw pointers to hand to the queue under one lock.
    std::vector<NNResultBuf*> ptrs;
    ptrs.reserve(n);
    for (int i = 0; i < n; i++) {
        bufs[i]->state_data = states[i].data();
        bufs[i]->state_size = (int)states[i].size();
        ptrs.push_back(bufs[i].get());
    }
    queue_.push_batch(ptrs);

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
    // Create ComputeHandle ON this thread — uploads weights to GPU.
    std::unique_ptr<ComputeHandle> handle;
    try {
        handle = context_->create_handle(model_.get(), gpu_id, max_batch_size_);
    } catch (const std::exception& e) {
        std::cerr << "NNEvaluator thread " << thread_id
                  << " (gpu " << gpu_id << "): " << e.what() << "\n";
    }

    // Signal that this thread's handle is ready (even on failure, so
    // wait_ready() doesn't block forever).
    {
        std::lock_guard<std::mutex> lock(ready_mutex_);
        handles_ready_++;
    }
    ready_cv_.notify_all();

    if (!handle) return;  // failed — exit thread

    std::vector<NNResultBuf*> batch;
    batch.reserve(max_batch_size_);

    while (true) {
        batch.clear();
        if (!queue_.wait_drain_up_to(batch, (size_t)max_batch_size_))
            break;   // queue closed and empty → exit
        if (batch.empty()) continue;   // spurious; queue is non-empty but nothing drained

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
        } catch (const std::exception& e) {
            std::cerr << "NNEvaluator: predict_batch failed: " << e.what() << "\n";
            // Return zero-logit policy so mask_policy's softmax produces
            // uniform priors over legal moves instead of crashing on an
            // empty vector.
            int action_size = model_->action_size;
            for (auto* buf : batch) {
                std::lock_guard<std::mutex> lock(buf->mu);
                buf->policy.assign(action_size, 0.0f);
                buf->value = 0.0f;
                buf->score = 0.0f;
                buf->done = true;
                buf->cv.notify_one();
            }
            continue;
        }

        // Deliver results — per-buf buf->cv is always notify_one: exactly
        // one search thread owns this buf and is blocked on it.
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
    }
}

}  // namespace minigo
