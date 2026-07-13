#include "mcts.h"
#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <iostream>
#include <numeric>
#include <thread>
#include <memory>

namespace minigo {

MCTSNode* MCTSNode::select_child(float c_puct) {
    // PUCT selection.  Parent terms (visit_count + virtual_loss + sqrt)
    // are hoisted out of the per-child loop — they don't change while we
    // iterate; per-child that saves 2 atomic loads + a sqrt (×82 children
    // at a 9x9 root, per descent step).
    int parent_total =
        visit_count.load(std::memory_order_relaxed) +
        virtual_loss_count.load(std::memory_order_relaxed);
    float sqrt_pt = std::sqrt((float)parent_total);

    MCTSNode* best = nullptr;
    float best_score = -1e9f;
    for (auto& child_ptr : children) {
        MCTSNode* child = child_ptr.get();
        if (!child) continue;

        int vc  = child->visit_count.load(std::memory_order_relaxed);
        int vlc = child->virtual_loss_count.load(std::memory_order_relaxed);
        int self_total = vc + vlc;

        float q = 0.0f;
        if (self_total != 0)
            q = -(child->total_value() + (float)vlc) / (float)self_total;

        float u = c_puct * child->prior * sqrt_pt / (1.0f + (float)self_total);
        float score = q + u;
        if (score > best_score) {
            best_score = score;
            best = child;
        }
    }
    return best;
}

MCTS::MCTS(BatchEvaluator* evaluator, const Config& config)
    : evaluator_(evaluator), config_(config),
      rng_(std::random_device{}()) {}

std::vector<float> MCTS::dirichlet(int n, float alpha) {
    std::gamma_distribution<float> gamma(alpha, 1.0f);
    std::vector<float> samples(n);
    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        samples[i] = gamma(rng_);
        sum += samples[i];
    }
    if (sum > 0.0f)
        for (auto& s : samples) s /= sum;
    return samples;
}

void MCTS::expand(MCTSNode* node, const std::vector<float>& policy,
                  const std::vector<float>& legal) {
    int action_size = (int)policy.size();
    node->children.resize(action_size);
    for (int a = 0; a < action_size; a++) {
        if (legal[a] > 0.0f) {
            auto child = std::make_unique<MCTSNode>();
            child->parent = node;
            child->action = a;
            child->prior  = policy[a];
            node->children[a] = std::move(child);
        }
    }
}

void MCTS::mask_policy(std::vector<float>& policy,
                       const std::vector<float>& legal, int action_size) {
    // Softmax over legal moves only (raw logits → proper [0,1] priors).
    // Without this, raw logits normalized by sum can produce negative priors,
    // concentrating search on a handful of children and making NaN-poisoning
    // of ALL children far more likely → livelock.
    float max_logit = -1e30f;
    for (int a = 0; a < action_size; a++)
        if (legal[a] > 0.0f && std::isfinite(policy[a]))
            max_logit = std::max(max_logit, policy[a]);

    float exp_sum = 0.0f;
    for (int a = 0; a < action_size; a++) {
        if (legal[a] > 0.0f && std::isfinite(policy[a])) {
            policy[a] = std::exp(policy[a] - max_logit);
            exp_sum += policy[a];
        } else {
            policy[a] = 0.0f;
        }
    }
    if (exp_sum > 0.0f) {
        for (auto& p : policy) p /= exp_sum;
    } else {
        // All logits were NaN/Inf or no legal moves — fall back to uniform
        float ls = std::accumulate(legal.begin(), legal.end(), 0.0f);
        for (int a = 0; a < action_size; a++)
            policy[a] = legal[a] / ls;
    }
}

void MCTS::add_dirichlet_noise(MCTSNode* node, int action_size) {
    auto noise = dirichlet(action_size, config_.dirichlet_alpha);
    float eps  = config_.dirichlet_epsilon;
    for (int a = 0; a < (int)node->children.size(); a++) {
        if (node->children[a])
            node->children[a]->prior = (1.0f - eps) * node->children[a]->prior
                                     + eps * noise[a];
    }
}

// ================================================================
// Backpropagation — two variants (matching KataGo)
// ================================================================

// Full backprop: undo virtual loss + add real visit + update value.
// Called after a successful NN evaluation.
void MCTS::backprop(const std::vector<MCTSNode*>& path, float value) {
    float v = value;
    for (int i = (int)path.size() - 1; i >= 0; i--) {
        MCTSNode* node = path[i];
        node->virtual_loss_count.fetch_sub(1, std::memory_order_relaxed);
        node->visit_count.fetch_add(1, std::memory_order_relaxed);
        node->add_value(v);
        v = -v;
    }
}

// Revert virtual losses ONLY — no visit count, no value update.
// Called on collision (abandoned playout).  Matches KataGo's
// revertVirtualLosses: cleans up the vloss applied during descent
// so the abandoned playout leaves no trace except the yield.
void MCTS::revert_virtual_losses(const std::vector<MCTSNode*>& path) {
    for (auto* node : path)
        node->virtual_loss_count.fetch_sub(1, std::memory_order_relaxed);
}

// ================================================================
// Multi-threaded search: KataGo's exact pattern
//
// Each thread does ONE playout at a time:
//   descend (with vloss) → evaluate_single (block) → expand → backprop
//
// On collision (node in EXPANDING state):
//   revert virtual losses, yield, retry from root.
//   The playout does NOT count.  This matches KataGo's
//   shouldCountPlayout=false + revertVirtualLosses.
//
// Batch size = however many threads submit leaves during one GPU call.
// With N search threads across all games, steady-state batch ≈ N.
// Speed scales linearly with search threads (more threads → bigger batch).
// ================================================================

void MCTS::search_thread_loop(MCTSNode* root, const GoGame& game,
                               int action_size,
                               std::atomic<int>& sims_done,
                               int num_simulations) {
    // Pre-allocate ONE NNResultBuf for this thread's entire lifetime.
    NNResultBuf result_buf;
    // Pre-allocate GoGame on heap — reused across playouts (avoids ~4KB on stack)
    auto game_copy_ptr = std::make_unique<GoGame>(game.copy());

    while (true) {
        if (sims_done.load(std::memory_order_relaxed) >= num_simulations
            || should_stop_.load(std::memory_order_relaxed))
            break;

        // ── Descend with virtual loss ────────────────────────────
        MCTSNode* node = root;
        *game_copy_ptr = game.copy();
        std::vector<MCTSNode*> path;
        path.push_back(node);
        node->virtual_loss_count.fetch_add(1, std::memory_order_relaxed);

        bool collision = false;

        while (true) {
            int st = node->state.load(std::memory_order_acquire);

            if (st == NODE_EXPANDED) {
                MCTSNode* child = node->select_child(config_.c_puct);
                if (!child) break;
                child->virtual_loss_count.fetch_add(1, std::memory_order_relaxed);
                path.push_back(child);
                int action = child->action;
                if (action == action_size - 1)
                    game_copy_ptr->play(PASS_MOVE);
                else
                    game_copy_ptr->play(action);
                node = child;

            } else if (st == NODE_EXPANDING) {
                collision = true;
                break;

            } else {
                break;
            }
        }

        if (collision) {
            revert_virtual_losses(path);
            std::this_thread::yield();
            continue;
        }

        // ── Expanded node with no selectable child ───────────────
        // select_child returned nullptr on an expanded node (all children
        // have NaN UCB scores).  Without this guard the thread retries
        // forever: the expansion claim always loses (node is EXPANDED,
        // not UNEVALUATED), sims_done never increments → livelock,
        // 3000% CPU, 0% GPU.
        if (node->state.load(std::memory_order_acquire) == NODE_EXPANDED) {
            backprop(path, 0.0f);
            sims_done.fetch_add(1, std::memory_order_relaxed);
            continue;
        }

        // ── Terminal node ────────────────────────────────────────
        if (game_copy_ptr->game_over) {
            float leaf_value;
            if      (game_copy_ptr->winner == EMPTY)                            leaf_value =  0.0f;
            else if (game_copy_ptr->winner == game_copy_ptr->current_player)    leaf_value =  1.0f;
            else                                                                 leaf_value = -1.0f;
            backprop(path, leaf_value);
            sims_done.fetch_add(1, std::memory_order_relaxed);
            continue;
        }

        // ── Claim expansion (single AMO — see NodeState encoding) ──
        // fetch_or of the claimed bit replaces the old
        // compare_exchange_strong(UNEVALUATED → EXPANDING):
        //   prev == UNEVALUATED  → we won; proceed to evaluate+expand.
        //   prev == EXPANDING    → another thread is expanding: same
        //                          collision path as before.
        //   prev == EXPANDED     → raced past a completed expansion
        //                          (3|1 == 3, so the OR left it
        //                          untouched): also retry from root.
        // (Bit-test form: lets x86 compile the returned-value fetch_or
        // as LOCK BTS; RISC-V amoor returns the old word natively.)
        int prev = node->state.fetch_or(NODE_EXPANDING,
                                        std::memory_order_acq_rel);
        if (prev & NODE_EXPANDING) {
            revert_virtual_losses(path);
            std::this_thread::yield();
            continue;
        }

        // ── Evaluate (blocks until server processes batch) ──────
        std::vector<float> state;
        evaluator_->encode_state(*game_copy_ptr, state);

        auto result = evaluator_->evaluate_with_buf(result_buf, state);

        // ── Expand + backprop ────────────────────────────────────
        std::vector<float> legal;
        game_copy_ptr->get_legal_moves(legal);
        mask_policy(result.policy, legal, action_size);
        expand(node, result.policy, legal);
        node->nn_score    = result.score;
        node->nn_score_sd = result.score_sd;
        node->state.store(NODE_EXPANDED, std::memory_order_release);

        // Blend value + score for utility (KataGo-style atan compression)
        float utility = config_.win_loss_weight * result.value;
        if (config_.score_weight != 0.0f) {
            float score_utility = atanf(result.score / config_.score_scale) / (float)(M_PI / 2.0);
            utility += config_.score_weight * score_utility;
        }
        if (!std::isfinite(utility)) utility = 0.0f;
        backprop(path, utility);
        sims_done.fetch_add(1, std::memory_order_relaxed);
    }
}

// ================================================================
// search() — dispatch
// ================================================================

void MCTS::search(GoGame& game, std::vector<float>& visits,
                  int num_simulations, bool add_noise, bool reuse_tree) {
    if (num_simulations < 0) num_simulations = config_.num_simulations;
    // NOTE: should_stop_ is NOT cleared here.  Clearing it inside
    // search() creates a race window where a stop signal raised after
    // the controller transitioned state but before search() was entered
    // would be lost.  The caller is responsible for clearing the flag
    // (e.g., AsyncBot::worker_loop does it under control_mutex_,
    // atomically with the mode transition).  Non-AsyncBot callers
    // (selfplay, eval, benchmark) never call request_stop, so the
    // flag is default-false and stays false for them.

    int action_size = config_.action_size();

    // ── Decide whether to reuse the existing root ───────────────
    bool can_reuse = false;
    if (reuse_tree) {
        std::lock_guard<std::mutex> lock(tree_mutex_);
        can_reuse = (root_ && root_->state.load(std::memory_order_acquire) == NODE_EXPANDED);
    }

    // ── Build new root tree in a local (outside any lock) ───────
    // GPU evaluation takes ms, so we do it without holding tree_mutex_,
    // then atomically swap into root_ once the new tree is ready.
    std::unique_ptr<MCTSNode> new_root;
    NNOutput root_nn_output;

    if (!can_reuse) {
        new_root = std::make_unique<MCTSNode>();

        std::vector<float> state_enc;
        evaluator_->encode_state(game, state_enc);
        // Single-state path (stack NNResultBuf, one queue push) — the
        // batch evaluate() wrapper here was a leftover of the deleted
        // VLP per-leaf search: it heap-allocated a buf and went through
        // push_batch for one state.
        root_nn_output = evaluator_->evaluate_single(state_enc);

        std::vector<float> legal;
        game.get_legal_moves(legal);
        mask_policy(root_nn_output.policy, legal, action_size);
        expand(new_root.get(), root_nn_output.policy, legal);
        new_root->nn_score    = root_nn_output.score;
        new_root->nn_score_sd = root_nn_output.score_sd;
        new_root->state.store(NODE_EXPANDED, std::memory_order_release);
        new_root->visit_count.store(1, std::memory_order_relaxed);
        float root_utility = config_.win_loss_weight * root_nn_output.value;
        if (config_.score_weight != 0.0f) {
            float root_score_utility = atanf(root_nn_output.score / config_.score_scale) / (float)(M_PI / 2.0);
            root_utility += config_.score_weight * root_score_utility;
        }
        if (!std::isfinite(root_utility)) root_utility = 0.0f;
        new_root->add_value(root_utility);
    }

    // ── Swap in new root / add noise (brief critical section) ────
    std::unique_ptr<MCTSNode> old_root;
    {
        std::lock_guard<std::mutex> lock(tree_mutex_);
        if (!can_reuse) {
            old_root = std::move(root_);
            root_ = std::move(new_root);
            root_noise_added_ = false;
            // Snapshot ownership for AnalysisInfo INSIDE the lock:
            // get_analysis() copies this vector under tree_mutex_ from
            // the AsyncBot callback thread (live-analysis HUD), which
            // can fire concurrently with this root rebuild.  Assigning
            // it outside the lock was a data race on the vector.
            root_nn_ownership_ = std::move(root_nn_output.ownership);
        }
        // Add Dirichlet noise at the root once per position.  After
        // make_move() the flag is cleared, so the next search that
        // requests noise will refresh it on the promoted subtree.
        if (add_noise && !root_noise_added_) {
            add_dirichlet_noise(root_.get(), action_size);
            root_noise_added_ = true;
        }
    }
    // old_root destroyed here, outside the lock

    // ── Run search (KataGo pattern: N threads, each VLP=1) ─────
    int nthreads = std::max(1, config_.num_search_threads);
    std::atomic<int> sims_done{0};

    // Spawn search threads — GoGame is heap-allocated inside each thread,
    // so default stack size is fine (no large stack objects).
    std::vector<std::thread> search_threads;
    search_threads.reserve(nthreads - 1);
    for (int t = 0; t < nthreads - 1; t++)
        search_threads.emplace_back(&MCTS::search_thread_loop, this,
                                     root_.get(), std::cref(game), action_size,
                                     std::ref(sims_done), num_simulations);

    search_thread_loop(root_.get(), game, action_size, sims_done, num_simulations);

    for (auto& t : search_threads)
        t.join();

    // ── Extract visit counts ─────────────────────────────────────
    visits.assign(action_size, 0.0f);
    for (int a = 0; a < (int)root_->children.size(); a++)
        if (root_->children[a])
            visits[a] = (float)root_->children[a]->visit_count.load(std::memory_order_relaxed);
}

// ================================================================
// Tree reuse: re-root to the child for `action`, discard siblings.
// Caller must hold no MCTS locks.  Safe against concurrent
// get_analysis() via tree_mutex_.  The old tree is destroyed
// outside the lock so deep tree deallocation does not block readers.
// ================================================================
void MCTS::make_move(int action) {
    std::unique_ptr<MCTSNode> old_root;
    {
        std::lock_guard<std::mutex> lock(tree_mutex_);
        if (!root_) return;

        if (action < 0 || action >= (int)root_->children.size()
            || !root_->children[action]) {
            // Unexplored branch — drop the whole tree.
            old_root = std::move(root_);
            root_noise_added_ = false;
            return;  // old_root destroyed after lock release
        }

        auto new_root = std::move(root_->children[action]);
        new_root->parent = nullptr;
        // Any virtual loss left over from an interrupted search is stale.
        new_root->virtual_loss_count.store(0, std::memory_order_relaxed);

        old_root = std::move(root_);        // release old root
        root_    = std::move(new_root);     // install promoted subtree
        root_noise_added_ = false;          // noise must be re-added on next call
        // The promoted node carries its own nn_score, so no member
        // variable needs updating — get_analysis() reads root_->nn_score.
    }
    // old_root destroyed here, outside the lock
}

void MCTS::reset_tree() {
    std::unique_ptr<MCTSNode> old_root;
    {
        std::lock_guard<std::mutex> lock(tree_mutex_);
        old_root = std::move(root_);
        root_noise_added_ = false;
    }
}

int MCTS::get_action(GoGame& game, std::vector<float>& policy,
                     float temperature, int num_simulations,
                     bool add_noise, bool reuse_tree) {
    int action_size = config_.action_size();

    std::vector<float> visits;
    search(game, visits, num_simulations, add_noise, reuse_tree);

    if (temperature == 0.0f) {
        int best = (int)(std::max_element(visits.begin(), visits.end())
                         - visits.begin());
        policy.assign(action_size, 0.0f);
        policy[best] = 1.0f;
        return best;
    }

    policy.resize(action_size);
    float sum = 0.0f;
    for (int a = 0; a < action_size; a++) {
        policy[a] = std::pow(visits[a], 1.0f / temperature);
        sum += policy[a];
    }
    if (sum > 0.0f) {
        for (auto& p : policy) p /= sum;
    } else {
        std::vector<float> legal;
        game.get_legal_moves(legal);
        float ls = std::accumulate(legal.begin(), legal.end(), 0.0f);
        for (int a = 0; a < action_size; a++) policy[a] = legal[a] / ls;
    }

    std::discrete_distribution<int> dist(policy.begin(), policy.end());
    return dist(rng_);
}

// ================================================================
// get_analysis — poll the live MCTS tree (thread-safe via atomics)
// ================================================================

MCTS::AnalysisInfo MCTS::get_analysis(int max_moves) const {
    // Hold tree_mutex_ so that search() cannot reassign root_ (which
    // would destroy the tree we're walking).  Search threads still
    // mutate visit_count/value/prior concurrently — reads use atomics
    // or tolerate racy scalar reads.
    std::lock_guard<std::mutex> lock(tree_mutex_);

    AnalysisInfo info;
    if (!root_) {
        info.root_score = 0.0f;
        return info;
    }
    // Read the promoted node's own NN outputs — survives make_move() so
    // the analysis HUD always shows data for the current position.
    info.root_score    = root_->nn_score;
    info.root_score_sd = root_->nn_score_sd;
    info.root_ownership = root_nn_ownership_;

    int vc = root_->visit_count.load(std::memory_order_relaxed);
    info.total_visits = vc;
    info.root_utility = (vc > 0) ? root_->total_value() / (float)vc : 0.0f;

    // Collect child moves
    for (int a = 0; a < (int)root_->children.size(); a++) {
        auto& child = root_->children[a];
        if (!child) continue;
        int cv = child->visit_count.load(std::memory_order_relaxed);
        if (cv == 0) continue;
        MoveInfo mi;
        mi.action  = a;
        mi.visits  = cv;
        mi.prior   = child->prior;
        mi.utility = -child->total_value() / (float)cv;  // negate: child stores from child's perspective
        info.moves.push_back(mi);
    }

    // Sort by visits descending
    std::sort(info.moves.begin(), info.moves.end(),
              [](const MoveInfo& a, const MoveInfo& b) { return a.visits > b.visits; });

    if (max_moves > 0 && (int)info.moves.size() > max_moves)
        info.moves.resize(max_moves);

    return info;
}

// ================================================================
// Self-play game → V3 GameRecord (moves + policies + outcome only;
// no encoded states — the trainer replays and encodes per-arch, and
// dihedral augmentation happens at sample time in the loader)
// ================================================================
GameRecord self_play_game(BatchEvaluator* evaluator, const Config& config) {
    MCTS mcts(evaluator, config);
    GoGame game(config.board_size, config.komi);

    GameRecord rec;
    rec.board_size = config.board_size;
    rec.komi       = config.komi;

    int action_size = config.action_size();

    while (!game.game_over && game.move_count < config.max_moves_per_game) {
        float temp = (game.move_count < config.temperature_threshold)
                     ? 1.0f : 0.0f;

        // reuse_tree=true — previous iteration called make_move() so the
        // root is already positioned at the current game state.  Dirichlet
        // noise is refreshed each move because make_move() clears the flag.
        std::vector<float> pi;
        int action = mcts.get_action(game, pi, temp, -1, /*add_noise=*/true,
                                      /*reuse_tree=*/true);

        rec.actions.push_back((int16_t)action);
        rec.policies.push_back(std::move(pi));

        if (action == action_size - 1)
            game.play(PASS_MOVE);
        else
            game.play(action);

        mcts.make_move(action);
    }

    while (!game.game_over) game.play(PASS_MOVE);

    auto [bs, ws] = game.score();
    rec.black_score = bs - ws;          // komi included (in ws)
    rec.winner      = game.winner;

    // Final ternary ownership: {0 empty/dame, 1 black, 2 white}
    std::vector<float> own_b, own_w;
    game.get_ownership(BLACK, own_b);
    game.get_ownership(WHITE, own_w);
    int hw = config.board_size * config.board_size;
    rec.owners.resize(hw);
    for (int i = 0; i < hw; i++)
        rec.owners[i] = own_b[i] > 0.5f ? (int8_t)1
                      : own_w[i] > 0.5f ? (int8_t)2 : (int8_t)0;

    return rec;
}

}  // namespace minigo
