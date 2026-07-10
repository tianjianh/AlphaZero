#pragma once

#include "config.h"
#include "game.h"
#include "batch_evaluator.h"
#include <atomic>
#include <cmath>
#include <cstdint>
#include <memory>
#include <mutex>
#include <random>
#include <vector>

namespace minigo {

// ================================================================
// MCTSNode — thread-safe for multi-threaded search
//
// Concurrency primitives are deliberately restricted to single-AMO
// atomics (fetch_add / fetch_or) and plain load/store — NO
// compare_exchange anywhere on the search path.  Rationale: CAS
// requires LR/SC (Zalrsc) or AMOCAS (Zacas); on RISC-V profiles that
// ship only Zaamo, every compare_exchange lowers to a libatomic
// call (a hidden shared lock) — ruinous once per node per playout.
// fetch_add / fetch_or map to amoadd / amoor on Zaamo, LOCK XADD /
// LOCK OR on x86, and ldadd / ldset on ARMv8.1 — cheap everywhere.
//
// visit_count, virtual_loss_count: atomic<int>, fetch_add/sub only
// total_value: fixed-point atomic<int64_t>, fetch_add only
// state: three-state lifecycle, claimed via fetch_or (see NodeState)
// ================================================================

// Node lifecycle encoding.  Bit 0 is the "claimed" bit, set in BOTH
// EXPANDING and EXPANDED.  This lets the one-time UNEVALUATED →
// EXPANDING claim be a single fetch_or(NODE_EXPANDING) instead of a
// compare_exchange: the returned previous value distinguishes all
// three cases, and OR-ing bit 0 into an already-EXPANDED node (3|1=3)
// changes nothing, so a lost race does no damage.
enum NodeState : int {
    NODE_UNEVALUATED = 0,
    NODE_EXPANDING   = 1,   // claimed bit
    NODE_EXPANDED    = 3,   // claimed bit | published bit
};

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

    // ── Atomic value accumulator: fixed-point int64, fetch_add ──
    // Replaces the old float-bits compare_exchange_weak retry loop.
    // Single amoadd.d per backprop step, and integer addition is
    // associative, so the accumulated total is exact and identical
    // regardless of thread interleaving (the float CAS-add was
    // run-order-dependent).
    //
    // Scale 2^20: quantization 1e-6 per sample, far below NN noise.
    // Headroom: |utility| ≤ ~4 in practice (win/loss ± weight plus
    // compressed score term); even at |v| = 64 a node absorbs 2^37
    // visits before int64 overflow — beyond any conceivable search.
    static constexpr float VALUE_FP_SCALE = 1048576.0f;   // 2^20
    std::atomic<int64_t> value_fp_{0};
    static_assert(std::atomic<int64_t>::is_always_lock_free,
                  "64-bit AMO required (RV64 amoadd.d / x86-64 LOCK XADD); "
                  "a mutex-backed fallback would poison the search hot path");

    void add_value(float v) {
        value_fp_.fetch_add((int64_t)llroundf(v * VALUE_FP_SCALE),
                            std::memory_order_relaxed);
    }

    float total_value() const {
        return (float)value_fp_.load(std::memory_order_relaxed)
               * (1.0f / VALUE_FP_SCALE);
    }

    // UCB selection lives in select_child() (mcts.cpp), which computes
    // Q from the PARENT's perspective: backprop stores total_value from
    // this node's (child's) perspective, so it negates.  Virtual loss:
    // each pending thread counts as a loss for the parent (−1), i.e.
    // +1 from the child's perspective, hence the +vlc term there.
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
    bool  root_noise_added_ = false;  // set by search() when Dirichlet noise
                                      // is injected; reset by make_move() and
                                      // when a fresh root is built
    // Root NN ownership for AnalysisInfo. Stored only at the root because
    // only the root drives the analysis HUD; per-node ownership would
    // bloat MCTSNode without a current consumer.
    std::vector<float> root_nn_ownership_;

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
