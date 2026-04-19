#pragma once

#include "batch_evaluator.h"   // for NNResultBuf

#include <algorithm>
#include <condition_variable>
#include <deque>
#include <mutex>
#include <vector>

namespace minigo {

// ----------------------------------------------------------------
// NNRequestQueue — thread-safe FIFO for NN evaluation requests.
//
// Role mirrors KataGo's `ThreadSafeQueue<NNResultBuf*> queryQueue`
// inside its NNEvaluator: encapsulate the mutex, condition variable,
// notification rules, and shutdown semantics so NNEvaluator never
// manipulates those primitives directly.
//
// Producers (MCTS search threads) push non-blocking.  Consumers
// (NN server threads) block on wait_drain_up_to() until items are
// available or the queue is closed.
//
// Notification rule (KataGo pattern, see cpp/core/threadsafequeue.h):
//     only call cv_.notify_all() on the empty → non-empty transition.
//
// Reasoning: std::condition_variable notifications are memoryless —
// if no consumer is currently in cv_.wait() at the moment of notify,
// the signal is dropped.  A consumer that is already draining the
// queue will re-check !items_.empty() on its next loop iteration
// without needing a fresh wake.  So the only case where notify
// matters is "queue went from empty to non-empty", i.e., some
// consumer might be asleep and needs waking.  Notifying on every
// push is safe but wasteful (a futex-wake syscall per call).
// ----------------------------------------------------------------
class NNRequestQueue {
public:
    NNRequestQueue() = default;
    ~NNRequestQueue() = default;

    NNRequestQueue(const NNRequestQueue&) = delete;
    NNRequestQueue& operator=(const NNRequestQueue&) = delete;

    // Non-blocking push of a single item.
    // Notify_all on the empty→1 transition only.
    void push(NNResultBuf* buf) {
        bool was_empty;
        {
            std::lock_guard<std::mutex> lock(mu_);
            was_empty = items_.empty();
            items_.push_back(buf);
        }
        if (was_empty) cv_.notify_all();
    }

    // Non-blocking push of N items under a single lock acquisition.
    // Notify_all on the empty → (≥1) transition only.
    void push_batch(const std::vector<NNResultBuf*>& bufs) {
        if (bufs.empty()) return;
        bool was_empty;
        {
            std::lock_guard<std::mutex> lock(mu_);
            was_empty = items_.empty();
            for (NNResultBuf* b : bufs) items_.push_back(b);
        }
        if (was_empty) cv_.notify_all();
    }

    // Block until the queue is non-empty or is closed.  Drain up to
    // `max_n` items by appending them to `out`.
    //
    //   returns true  — items were drained (even if the queue is now
    //                   closed — consumers must still process any
    //                   remaining items before exiting)
    //   returns false — the queue is closed AND empty; the consumer
    //                   should exit its loop
    bool wait_drain_up_to(std::vector<NNResultBuf*>& out, size_t max_n) {
        std::unique_lock<std::mutex> lock(mu_);
        cv_.wait(lock, [this] { return !items_.empty() || closed_; });
        if (items_.empty()) return false;   // closed && empty
        size_t n = std::min(max_n, items_.size());
        out.reserve(out.size() + n);
        for (size_t i = 0; i < n; i++) {
            out.push_back(items_.front());
            items_.pop_front();
        }
        return true;
    }

    // Signal shutdown.  Wakes every waiter so they can observe the
    // closed state: consumers still drain any remaining items, then
    // see items_.empty() && closed_ on the next wait and exit.
    //
    // Post-close pushes still succeed (matches the old NNEvaluator
    // behaviour of letting in-flight submissions finish).  Items
    // pushed after close() will be drained by whichever consumer
    // is still alive; if all consumers have already exited, those
    // items are effectively leaked — that is the caller's problem
    // to avoid (stop submitting before closing).
    void close() {
        {
            std::lock_guard<std::mutex> lock(mu_);
            closed_ = true;
        }
        cv_.notify_all();
    }

private:
    std::mutex                   mu_;
    std::condition_variable      cv_;
    std::deque<NNResultBuf*>     items_;
    bool                         closed_ = false;
};

}  // namespace minigo
