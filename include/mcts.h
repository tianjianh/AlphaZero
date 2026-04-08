#pragma once

#include "config.h"
#include "game.h"
#include "batch_evaluator.h"
#include <atomic>
#include <cstring>
#include <memory>
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

    // Stats from the MCTS root after search (no extra compute)
    struct SearchInfo {
        float root_utility = 0.0f; // mean blended utility (value + score) from MCTS
        float root_score = 0.0f;   // NN's raw score estimate in points (+ = current player leads)
        int   total_visits = 0;
        int   best_visits = 0;     // visits on the selected move
    };

    void search(GoGame& game, std::vector<float>& visits,
                int num_simulations = -1, bool add_noise = true,
                SearchInfo* info = nullptr);

    int get_action(GoGame& game, std::vector<float>& policy,
                   float temperature = 1.0f, int num_simulations = -1,
                   bool add_noise = true, SearchInfo* info = nullptr);

private:
    BatchEvaluator* evaluator_;
    Config          config_;
    std::mt19937    rng_;

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
    float score;  // normalized score from current player's perspective [-1, 1]
};

std::vector<TrainingRecord> self_play_game(
    BatchEvaluator* evaluator, const Config& config);

}  // namespace minigo
