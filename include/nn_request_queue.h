#pragma once

#include "batch_evaluator.h"   // for NNResultBuf

#include <algorithm>
#include <condition_variable>
#include <cstddef>
#include <mutex>
#include <stdexcept>
#include <vector>

namespace minigo {

// ----------------------------------------------------------------
// NNRequestQueue — fixed-capacity ring buffer for NN evaluation
// requests.
//
// Role mirrors KataGo's ThreadSafeQueue<NNResultBuf*> inside its
// NNEvaluator: encapsulate the mutex, condition variable, notification
// rules, and shutdown semantics so NNEvaluator does not manipulate
// those primitives directly.
//
// Layout:
//   buf_: std::vector<NNResultBuf*> of power-of-two capacity
//   head_, tail_: wrap modulo capacity via bitwise AND (free modulo)
//   size_: count of in-flight items (used for empty/full check; head_
//          == tail_ alone cannot distinguish the two)
//
// Push is O(1) — one slot write + index increment.
// Drain is O(n) — n slot reads + index increment.  No memmove ever.
//
// Notification rule (KataGo pattern, see cpp/core/threadsafequeue.h):
//   cv_.notify_all() only on the empty → non-empty transition.
//
// Capacity:
//   - Construction takes any positive desired_capacity and rounds it
//     up to the next power of two internally (so the bitwise-AND
//     modulo is always valid).  Callers do not need to think about
//     power-of-two sizing.
//   - The ring is bounded.  Push throws std::runtime_error on full —
//     with KataGo's max_batch × 4 × num_server_threads heuristic that
//     should never happen in normal operation; hitting it means the
//     assumed producer upper bound was wrong.
// ----------------------------------------------------------------
class NNRequestQueue {
public:
    explicit NNRequestQueue(size_t desired_capacity)
        : buf_(round_up_pow2(std::max<size_t>(1, desired_capacity))),
          mask_(buf_.size() - 1) {}

    ~NNRequestQueue() = default;

    NNRequestQueue(const NNRequestQueue&) = delete;
    NNRequestQueue& operator=(const NNRequestQueue&) = delete;

    // Final capacity (rounded up to the next power of two at construction).
    size_t capacity() const { return buf_.size(); }

    // Non-blocking push of a single item.  Notify_all on the empty→1
    // transition only.  Throws on capacity overflow.
    void push(NNResultBuf* buf) {
        bool was_empty;
        {
            std::lock_guard<std::mutex> lock(mu_);
            if (size_ == buf_.size())
                throw std::runtime_error(
                    "NNRequestQueue: capacity exceeded — raise "
                    "max_batch_size * 4 * num_server_threads");
            was_empty = (size_ == 0);
            buf_[tail_] = buf;
            tail_ = (tail_ + 1) & mask_;
            ++size_;
        }
        if (was_empty) cv_.notify_all();
    }

    // Non-blocking push of N items under a single lock acquisition.
    // Notify_all on the empty → (≥1) edge only.  Throws on overflow.
    void push_batch(const std::vector<NNResultBuf*>& bufs) {
        if (bufs.empty()) return;
        bool was_empty;
        {
            std::lock_guard<std::mutex> lock(mu_);
            if (size_ + bufs.size() > buf_.size())
                throw std::runtime_error(
                    "NNRequestQueue: capacity exceeded on batch push");
            was_empty = (size_ == 0);
            for (NNResultBuf* b : bufs) {
                buf_[tail_] = b;
                tail_ = (tail_ + 1) & mask_;
            }
            size_ += bufs.size();
        }
        if (was_empty) cv_.notify_all();
    }

    // Block until the queue is non-empty or is closed.  Drain up to
    // `max_n` items by appending them to `out`.
    //
    //   returns true  — items were drained
    //   returns false — the queue is closed AND empty; the consumer
    //                   should exit its loop
    bool wait_drain_up_to(std::vector<NNResultBuf*>& out, size_t max_n) {
        std::unique_lock<std::mutex> lock(mu_);
        cv_.wait(lock, [this] { return size_ != 0 || closed_; });
        if (size_ == 0) return false;   // closed && empty
        size_t n = std::min(max_n, size_);
        out.reserve(out.size() + n);
        for (size_t i = 0; i < n; ++i) {
            out.push_back(buf_[head_]);
            head_ = (head_ + 1) & mask_;
        }
        size_ -= n;
        return true;
    }

    // Signal shutdown.  Wakes every waiter so they can observe the
    // closed state: consumers still drain any remaining items, then
    // see size_ == 0 && closed_ on the next wait and exit.
    //
    // Post-close pushes still succeed if there is capacity, matching
    // the old NNEvaluator behaviour of letting in-flight submissions
    // finish.  Items pushed after every consumer has exited will never
    // be drained — caller must stop submitting before close().
    void close() {
        {
            std::lock_guard<std::mutex> lock(mu_);
            closed_ = true;
        }
        cv_.notify_all();
    }

private:
    // Smallest power of two >= x.  Assumes x >= 1.
    static size_t round_up_pow2(size_t x) {
        size_t p = 1;
        while (p < x) p <<= 1;
        return p;
    }

    std::mutex                   mu_;
    std::condition_variable      cv_;
    std::vector<NNResultBuf*>    buf_;
    size_t                       mask_;      // capacity - 1
    size_t                       head_ = 0;
    size_t                       tail_ = 0;
    size_t                       size_ = 0;
    bool                         closed_ = false;
};

}  // namespace minigo
