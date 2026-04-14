#pragma once

#include "config.h"
#include "game.h"
#include "batch_evaluator.h"
#include <atomic>
#include <cstring>
#include <memory>
#include <mutex>
#include <random>
#include <vector>

namespace minigo {

// ================================================================
// MCTSNode — thread-safe for multi-threaded search
//
// visit_count, virtual_loss_count: std::atomic<int>
// total_value: atomic float via int32_t CAS (portable C++17)
// is_expanded: three-state (UNEVALUATED → EXPANDING → EXPANDED)
//   ensures only one thread calls expand(), others backprop and retry.
// ================================================================

enum NodeState : int { NODE_UNEVALUATED = 0, NODE_EXPANDING = 1, NODE_EXPANDED = 2 };

struct MCTSNode {
    MCTSNode* parent = nullptr;
    int action = -1;
    float prior = 0.0f;
    float nn_score = 0.0f;   // NN score output at this position (set once
                             // during expand, then read by get_analysis;
                             // benign data race — 32-bit aligned float
                             // writes are atomic on target platforms)
    float nn_score_sd = 0.0f;  // NN score stdev (uncertainty)

    std::atomic<int> visit_count{0};
    std::atomic<int> virtual_loss_count{0};
    std::atomic<int> state{NODE_UNEVALUATED};

    // Flat vector indexed by action (set during expand)
    std::vector<std::unique_ptr<MCTSNode>> children;

    // ── Atomic float total_value via int32 CAS ───────────────────
    std::atomic<int32_t> value_bits_{0};

    void add_value(float v) {
        int32_t old_bits = value_bits_.load(std::memory_order_relaxed);
        int32_t new_bits;
        float old_f, new_f;
        do {
            std::memcpy(&old_f, &old_bits, sizeof(float));
            new_f = old_f + v;
            std::memcpy(&new_bits, &new_f, sizeof(float));
        } while (!value_bits_.compare_exchange_weak(old_bits, new_bits,
                 std::memory_order_relaxed));
    }

    float total_value() const {
        int32_t bits = value_bits_.load(std::memory_order_relaxed);
        float f;
        std::memcpy(&f, &bits, sizeof(float));
        return f;
    }

    // ── Derived values ───────────────────────────────────────────
    // Q from PARENT's perspective (for UCB selection).
    // Backprop stores total_value from this node's (child's) perspective,
    // so we negate.  Virtual loss: each pending thread counts as a loss
    // for the parent (−1), i.e. +1 from child's perspective, hence +vlc.
    float q_value() const {
        int vc  = visit_count.load(std::memory_order_relaxed);
        int vlc = virtual_loss_count.load(std::memory_order_relaxed);
        int total = vc + vlc;
        if (total == 0) return 0.0f;
        return -(total_value() + (float)vlc) / (float)total;
    }

    float ucb_score(float c_puct) const {
        int parent_total = parent->visit_count.load(std::memory_order_relaxed)
                         + parent->virtual_loss_count.load(std::memory_order_relaxed);
        int self_total   = visit_count.load(std::memory_order_relaxed)
                         + virtual_loss_count.load(std::memory_order_relaxed);
        float u = c_puct * prior * std::sqrt((float)parent_total)
                  / (1.0f + self_total);
        return q_value() + u;
    }

    bool is_leaf() const {
        return state.load(std::memory_order_acquire) != NODE_EXPANDED;
    }

    MCTSNode* select_child(float c_puct);
};

// ================================================================
// MCTS — multi-threaded search (KataGo-style)
// ================================================================
class MCTS {
public:
    MCTS(BatchEvaluator* evaluator, const Config& config);

    // ── Analysis info — poll the live MCTS tree ─────────────
    // Safe to call during search (reads atomics) or after search.
    // Returns top moves sorted by visit count.
    struct MoveInfo {
        int   action = -1;
        int   visits = 0;
        float prior  = 0.0f;    // policy prior from NN
        float utility = 0.0f;   // mean Q (blended value+score)
    };
    struct AnalysisInfo {
        std::vector<MoveInfo> moves;   // top moves, sorted by visits desc
        int   total_visits = 0;
        float root_utility = 0.0f;    // mean Q at root
        float root_score   = 0.0f;    // NN raw score estimate (points)
        float root_score_sd = 0.0f;   // NN score stdev (uncertainty)
        std::vector<float> root_ownership;  // [board²] NN ownership at root
    };

    AnalysisInfo get_analysis(int max_moves = 5) const;

    // ── Search ──────────────────────────────────────────────
    // search() builds a fresh root unless reuse_tree=true AND root_ is
    // already expanded (e.g. after a previous search + make_move()).
    void search(GoGame& game, std::vector<float>& visits,
                int num_simulations = -1, bool add_noise = true,
                bool reuse_tree = false);

    int get_action(GoGame& game, std::vector<float>& policy,
                   float temperature = 1.0f, int num_simulations = -1,
                   bool add_noise = true, bool reuse_tree = false);

    // ── Tree reuse (KataGo pattern) ─────────────────────────
    // Re-root the tree to the child at `action` after that move was played.
    // The chosen subtree is preserved; siblings are discarded outside the
    // tree_mutex_ lock.  If the child doesn't exist (unexplored branch),
    // the tree is cleared and the next search() will build a fresh root.
    void make_move(int action);

    // Clear the tree completely (e.g. new game, SGF load).
    void reset_tree();

    // ── Lifecycle (KataGo pattern) ──────────────────────────
    // Request early stop of a running search. Safe to call from any thread.
    // The search threads will exit at their next check point.
    void request_stop() { should_stop_.store(true, std::memory_order_relaxed); }

    // Clear the stop flag.  Call from the controller thread BEFORE a
    // new search, under whatever synchronization the controller uses
    // to serialize stop/start.  search() itself does NOT clear the
    // flag — doing so inside search() would create a window where a
    // prior stop signal could be lost (see AsyncBot::worker_loop).
    void reset_stop_flag() { should_stop_.store(false, std::memory_order_relaxed); }

private:
    BatchEvaluator* evaluator_;
    Config          config_;
    std::mt19937    rng_;

    // Live tree state (persists between search() and get_analysis()).
    // tree_mutex_ protects concurrent access from get_analysis() while
    // search() is rebuilding the root at the start of a new search.
    mutable std::mutex        tree_mutex_;
    std::unique_ptr<MCTSNode> root_;
    int   action_size_      = 0;
    bool  root_noise_added_ = false;  // set by search() when Dirichlet noise
                                      // is injected; reset by make_move() and
                                      // when a fresh root is built

    // Stop flag — checked by search threads, set by request_stop()
    std::atomic<bool> should_stop_{false};

    void mask_policy(std::vector<float>& policy,
                     const std::vector<float>& legal, int action_size);

    void expand(MCTSNode* node, const std::vector<float>& policy,
                const std::vector<float>& legal);
    void add_dirichlet_noise(MCTSNode* node, int action_size);
    std::vector<float> dirichlet(int n, float alpha);

    // ── Multi-threaded search internals ──────────────────────────
    void search_thread_loop(MCTSNode* root, const GoGame& game,
                            int action_size,
                            std::atomic<int>& sims_done,
                            int num_simulations);

    // Full backprop: undo vloss + add visit + update value
    void backprop(const std::vector<MCTSNode*>& path, float value);

    // Revert vloss ONLY (for abandoned/collision playouts — KataGo pattern)
    void revert_virtual_losses(const std::vector<MCTSNode*>& path);

    // ── Single-threaded search (fallback for DirectEvaluator) ────
    void search_single_threaded(MCTSNode* root, const GoGame& game,
                                int action_size, int num_simulations);
};

// ================================================================
// Training data
// ================================================================
struct TrainingRecord {
    std::vector<float> state;
    std::vector<float> policy;
    float value;
    float score;                       // points, current player's perspective
    std::vector<float> ownership;      // [board²] — 1.0 = current player owns
    int   opponent_action;             // opponent's next move (-1 if last move)
};

std::vector<TrainingRecord> self_play_game(
    BatchEvaluator* evaluator, const Config& config);

}  // namespace minigo
