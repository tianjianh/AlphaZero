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
// NNEvaluator: encapsulate the mutex, the two condition variables
// (not_empty_ for consumers, not_full_ for the rare blocked producer),
// the notification rules, and shutdown semantics so NNEvaluator does
// not manipulate those primitives directly.
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
//   not_empty_.notify_all() only on the empty → non-empty transition;
//   not_full_.notify_all() only when a drain frees space on a full ring.
//
// Capacity (FIXED — the ring never reallocates):
//   - Construction takes any positive desired_capacity and rounds it
//     up to the next power of two internally (so the bitwise-AND
//     modulo is always valid).
//   - The occupancy bound is structural: every search thread owns ONE
//     NNResultBuf, pushes it, and BLOCKS on its condvar until the
//     server delivers the result — so a client thread never has a
//     second request in flight.  NNEvaluator therefore sizes the ring
//     to cover the declared client-thread count, and in normal
//     operation a push always finds a free slot.
//   - If a caller nevertheless exceeds capacity (e.g. a direct
//     evaluate() with a batch larger than the ring), push BLOCKS until
//     the servers drain space — the semantics of KataGo's bounded
//     ThreadSafeQueue::waitPush (cpp/core/threadsafequeue.h).  This is
//     pure backpressure: a full ring means the GPU is already
//     saturated, and it can never deadlock while at least one server
//     thread is draining (servers never wait on producers).  Close()
//     also wakes blocked pushers so shutdown is never stuck.
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

    // Push a single item.  Non-blocking in normal operation (the ring
    // is sized for the client-thread count); blocks on a full ring
    // until a server drains space.  Notify_all on the empty→1 edge.
    //
    // Post-close: inserts if there is space (a server that hasn't yet
    // observed closed && empty will still drain and deliver it), but
    // NEVER inserts into a closed full ring — nothing is guaranteed to
    // drain it, and writing anyway would overwrite an undelivered
    // entry.  Either way the documented contract stands: stop
    // submitting before close(), or results may never arrive.
    void push(NNResultBuf* buf) {
        bool was_empty;
        {
            std::unique_lock<std::mutex> lock(mu_);
            not_full_.wait(lock, [this] {
                return size_ < buf_.size() || closed_;
            });
            if (size_ == buf_.size()) return;   // closed && full: drop
            was_empty = (size_ == 0);
            buf_[tail_] = buf;
            tail_ = (tail_ + 1) & mask_;
            ++size_;
        }
        if (was_empty) not_empty_.notify_all();
    }

    // Push N items.  Inserts as much as fits under one lock
    // acquisition, blocking for space as needed (so a batch larger
    // than the whole ring still goes through in chunks while servers
    // drain the earlier entries).  Notify_all on each empty→(≥1) edge.
    // Post-close semantics per chunk are identical to push().
    void push_batch(const std::vector<NNResultBuf*>& bufs) {
        size_t i = 0;
        while (i < bufs.size()) {
            bool was_empty;
            {
                std::unique_lock<std::mutex> lock(mu_);
                not_full_.wait(lock, [this] {
                    return size_ < buf_.size() || closed_;
                });
                if (size_ == buf_.size()) return;   // closed && full: drop rest
                was_empty = (size_ == 0);
                while (i < bufs.size() && size_ < buf_.size()) {
                    buf_[tail_] = bufs[i++];
                    tail_ = (tail_ + 1) & mask_;
                    ++size_;
                }
            }
            if (was_empty) not_empty_.notify_all();
        }
    }

    // Block until the queue is non-empty or is closed.  Drain up to
    // `max_n` items by appending them to `out`.
    //
    //   returns true  — items were drained
    //   returns false — the queue is closed AND empty; the consumer
    //                   should exit its loop
    bool wait_drain_up_to(std::vector<NNResultBuf*>& out, size_t max_n) {
        bool freed_space;
        {
            std::unique_lock<std::mutex> lock(mu_);
            not_empty_.wait(lock, [this] { return size_ != 0 || closed_; });
            if (size_ == 0) return false;   // closed && empty
            bool was_full = (size_ == buf_.size());
            size_t n = std::min(max_n, size_);
            out.reserve(out.size() + n);
            for (size_t i = 0; i < n; ++i) {
                out.push_back(buf_[head_]);
                head_ = (head_ + 1) & mask_;
            }
            size_ -= n;
            freed_space = was_full;   // full → not-full edge only
        }
        if (freed_space) not_full_.notify_all();
        return true;
    }

    // Signal shutdown.  Wakes every waiter so they can observe the
    // closed state: consumers still drain any remaining items, then
    // see size_ == 0 && closed_ on the next wait and exit.
    //
    // A push racing with close() may still be served — servers exit
    // only once the queue is closed AND empty — which is why push()
    // still inserts post-close when there is space.  But an item
    // pushed after every consumer has exited is never drained, so the
    // contract remains: stop submitting before close().  (All binaries
    // join their search threads before destroying the NNEvaluator.)
    void close() {
        {
            std::lock_guard<std::mutex> lock(mu_);
            closed_ = true;
        }
        not_empty_.notify_all();
        not_full_.notify_all();   // wake any pusher blocked on a full ring
    }

private:
    // Smallest power of two >= x.  Assumes x >= 1.
    static size_t round_up_pow2(size_t x) {
        size_t p = 1;
        while (p < x) p <<= 1;
        return p;
    }

    std::mutex                   mu_;
    std::condition_variable      not_empty_;  // consumers wait here
    std::condition_variable      not_full_;   // producers wait here (rare)
    std::vector<NNResultBuf*>    buf_;
    size_t                       mask_;      // capacity - 1
    size_t                       head_ = 0;
    size_t                       tail_ = 0;
    size_t                       size_ = 0;
    bool                         closed_ = false;
};

}  // namespace minigo
