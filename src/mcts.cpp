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
    // Hoist the parent terms (visit_count + virtual_loss + sqrt) out of
    // the per-child loop — they don't change while we iterate, and
    // ucb_score() would re-read them per child via 2 atomic loads + a
    // sqrt otherwise.  The math here mirrors ucb_score() exactly.
    //
    // children is a dense list of legal children (no null slots), so
    // this loop runs exactly `children.size()` iterations.  For Xiangqi
    // that's ~30-50 per node instead of 8100 with the old sparse layout.
    int parent_total =
        visit_count.load(std::memory_order_relaxed) +
        virtual_loss_count.load(std::memory_order_relaxed);
    float sqrt_pt = std::sqrt((float)parent_total);

    MCTSNode* best = nullptr;
    float best_score = -1e9f;
    for (auto& child_ptr : children) {
        MCTSNode* child = child_ptr.get();

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
    // Build a dense list of just the legal children.  The parent's
    // `children` vector used to be sized to action_size (8100 for
    // Xiangqi) with ~8050 nullptr slots; that cost 64 KB per node and
    // made every descent step scan all 8100 pointers.  Keeping only
    // the ~30-50 legal children drops descent per-node from 8100
    // pointer loads to ~40 and allocates ~400 B per node.
    int action_size = (int)policy.size();
    node->children.clear();
    node->children.reserve(64);   // typical Xiangqi branching factor
    for (int a = 0; a < action_size; a++) {
        if (legal[a] > 0.0f) {
            auto child = std::make_unique<MCTSNode>();
            child->parent = node;
            child->action = a;
            child->prior  = policy[a];
            node->children.push_back(std::move(child));
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

void MCTS::add_dirichlet_noise(MCTSNode* node, int /*action_size*/) {
    // Generate Dirichlet over the LEGAL children count so the noise sums
    // to 1 over the legal subset.  The previous implementation generated
    // Dir(alpha, action_size) — 8100 samples summing to 1 over ALL
    // actions — then applied only the ~50 entries that landed on legal
    // action ids.  The resulting noise mass across legal children summed
    // to ~50/8100 ≈ 0.6% instead of 1, so with eps=0.25 the blended prior
    // received ~0.0015 worth of exploration noise, not 0.25.
    if (node->children.empty()) return;
    auto noise = dirichlet((int)node->children.size(), config_.dirichlet_alpha);
    float eps  = config_.dirichlet_epsilon;
    for (size_t i = 0; i < node->children.size(); i++) {
        node->children[i]->prior =
            (1.0f - eps) * node->children[i]->prior + eps * noise[i];
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

void MCTS::search_thread_loop(MCTSNode* root, const XiangqiGame& game,
                               int action_size,
                               std::atomic<int>& sims_done,
                               int num_simulations) {
    // Pre-allocate ONE NNResultBuf for this thread's entire lifetime.
    NNResultBuf result_buf;
    // Pre-allocate XiangqiGame on heap — reused across playouts (avoids ~4KB on stack)
    auto game_copy_ptr = std::make_unique<XiangqiGame>(game.copy());

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
        // forever: CAS fails (node is EXPANDED, not UNEVALUATED),
        // sims_done never increments → livelock, 3000% CPU, 0% GPU.
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

        // ── CAS to claim expansion ──────────────────────────────
        int expected = NODE_UNEVALUATED;
        if (!node->state.compare_exchange_strong(expected, NODE_EXPANDING,
                std::memory_order_acq_rel)) {
            revert_virtual_losses(path);
            std::this_thread::yield();
            continue;
        }

        // ── Evaluate (blocks until server processes batch) ──────
        std::vector<float> state;
        game_copy_ptr->encode(state);

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
// Single-threaded VLP search (for DirectEvaluator / Eigen)
//
// Uses VLP batching: collect vloss_parallel leaves, batch evaluate,
// expand, backprop.  Kept for the Eigen backend where there's no
// server thread and batch submission is direct.
// ================================================================

void MCTS::search_single_threaded(MCTSNode* root, const XiangqiGame& game,
                                   int action_size, int num_simulations) {
    int sims_done      = 0;
    int vloss_parallel = config_.virtual_loss_parallel;

    while (sims_done < num_simulations) {
        int batch_count = std::min(vloss_parallel, num_simulations - sims_done);

        struct PendingLeaf {
            std::vector<MCTSNode*> path;
            MCTSNode*              leaf;
            std::vector<float>     state;
            std::vector<float>     legal;
        };

        std::vector<PendingLeaf> pending;
        pending.reserve(batch_count);

        // Pre-allocate XiangqiGame on heap — reused across batch
        auto game_copy_ptr = std::make_unique<XiangqiGame>(game.copy());

        for (int i = 0; i < batch_count; i++) {
            PendingLeaf leaf;

            MCTSNode* node = root;
            *game_copy_ptr = game.copy();
            leaf.path.push_back(node);
            node->virtual_loss_count.fetch_add(1, std::memory_order_relaxed);

            while (!node->children.empty()) {
                node = node->select_child(config_.c_puct);
                if (!node) break;
                node->virtual_loss_count.fetch_add(1, std::memory_order_relaxed);
                leaf.path.push_back(node);
                int action = node->action;
                game_copy_ptr->play(action);
            }

            if (!node) {
                for (auto* n : leaf.path)
                    n->virtual_loss_count.fetch_sub(1, std::memory_order_relaxed);
                continue;
            }

            leaf.leaf = node;

            if (game_copy_ptr->game_over) {
                float v;
                if      (game_copy_ptr->winner == EMPTY)                            v =  0.0f;
                else if (game_copy_ptr->winner == game_copy_ptr->current_player)    v =  1.0f;
                else                                                                 v = -1.0f;
                backprop(leaf.path, v);
                sims_done++;
                continue;
            }

            game_copy_ptr->encode(leaf.state);
            game_copy_ptr->get_legal_moves(leaf.legal);
            pending.push_back(std::move(leaf));
        }

        if (pending.empty()) continue;

        std::vector<std::vector<float>> states;
        states.reserve(pending.size());
        for (auto& p : pending)
            states.push_back(std::move(p.state));

        auto results = evaluator_->evaluate(states);

        for (size_t i = 0; i < pending.size(); i++) {
            auto& res = results[i];
            auto& leaf = pending[i];

            mask_policy(res.policy, leaf.legal, action_size);
            if (leaf.leaf->children.empty())
                expand(leaf.leaf, res.policy, leaf.legal);
            leaf.leaf->nn_score    = res.score;
            leaf.leaf->nn_score_sd = res.score_sd;
            leaf.leaf->state.store(NODE_EXPANDED, std::memory_order_release);

            float utility = config_.win_loss_weight * res.value;
            if (config_.score_weight != 0.0f) {
                float score_utility = atanf(res.score / config_.score_scale) / (float)(M_PI / 2.0);
                utility += config_.score_weight * score_utility;
            }
            if (!std::isfinite(utility)) utility = 0.0f;
            backprop(leaf.path, utility);
            sims_done++;
        }
    }
}

// ================================================================
// search() — dispatch
// ================================================================

void MCTS::search(XiangqiGame& game, std::vector<float>& visits,
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

    action_size_ = config_.action_size();
    int action_size = action_size_;

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
        game.encode(state_enc);
        auto root_results = evaluator_->evaluate({ state_enc });
        root_nn_output = std::move(root_results[0]);

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

    // Spawn search threads — XiangqiGame is heap-allocated inside each thread,
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
    // The output `visits` is still an 8100-wide action-indexed vector
    // (contract with self_play / training records).  Iterating the
    // dense children list writes only the ~50 non-zero entries.
    visits.assign(action_size, 0.0f);
    for (auto& child : root_->children)
        visits[child->action] = (float)child->visit_count.load(std::memory_order_relaxed);
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

        // Linear scan for the child whose `action` matches.  Children
        // are a dense list (~30-50 entries), so this is fast; called
        // once per actual game ply, not in any hot loop.
        MCTSNode* match = nullptr;
        size_t match_idx = 0;
        for (size_t i = 0; i < root_->children.size(); i++) {
            if (root_->children[i] && root_->children[i]->action == action) {
                match = root_->children[i].get();
                match_idx = i;
                break;
            }
        }

        if (!match) {
            // Unexplored branch — drop the whole tree.
            old_root = std::move(root_);
            root_noise_added_ = false;
            return;  // old_root destroyed after lock release
        }

        auto new_root = std::move(root_->children[match_idx]);
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

int MCTS::get_action(XiangqiGame& game, std::vector<float>& policy,
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

    int vc = root_->visit_count.load(std::memory_order_relaxed);
    info.total_visits = vc;
    info.root_utility = (vc > 0) ? root_->total_value() / (float)vc : 0.0f;

    // Collect child moves
    for (auto& child : root_->children) {
        int cv = child->visit_count.load(std::memory_order_relaxed);
        if (cv == 0) continue;
        MoveInfo mi;
        mi.action  = child->action;
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
// Dihedral augmentation (8-fold symmetry)
// ================================================================
static void augment_sample(const std::vector<float>& state,
                           const std::vector<float>& policy,
                           float value, float score,
                           const std::vector<float>& ownership,
                           int opponent_action,
                           int board_rows, int board_cols,
                           int input_channels,
                           std::vector<TrainingRecord>& out) {
    int area = board_rows * board_cols;
    int action_size = area * area;

    auto mirror_sq = [&](int sq) {
        int r = sq / board_cols;
        int c = sq % board_cols;
        return r * board_cols + (board_cols - 1 - c);
    };

    auto mirror_action = [&](int action) {
        if (action < 0 || action >= action_size) return action;
        int src = action / area;
        int dst = action % area;
        return mirror_sq(src) * area + mirror_sq(dst);
    };

    for (int flip = 0; flip < 2; ++flip) {
        TrainingRecord rec;
        rec.state.resize((size_t)input_channels * area);
        rec.policy.resize(action_size);
        rec.ownership.resize(area);
        rec.value = value;
        rec.score = score;
        rec.opponent_action = flip ? mirror_action(opponent_action) : opponent_action;

        if (!flip) {
            rec.state = state;
            rec.policy = policy;
            rec.ownership = ownership;
            out.push_back(std::move(rec));
            continue;
        }

        for (int ch = 0; ch < input_channels; ++ch) {
            for (int r = 0; r < board_rows; ++r) {
                for (int c = 0; c < board_cols; ++c) {
                    int src_idx = ch * area + r * board_cols + c;
                    int dst_idx = ch * area + r * board_cols + (board_cols - 1 - c);
                    rec.state[dst_idx] = state[src_idx];
                }
            }
        }

        for (int sq = 0; sq < area; ++sq) {
            rec.ownership[mirror_sq(sq)] = ownership[sq];
        }
        for (int action = 0; action < action_size; ++action) {
            rec.policy[mirror_action(action)] = policy[action];
        }
        out.push_back(std::move(rec));
    }
}

// ================================================================
// Self-play game
// ================================================================
static std::vector<TrainingRecord> self_play_game_impl(
        MCTS& mcts, const Config& config) {
    XiangqiGame game(config.history_length);

    struct Step {
        std::vector<float> state;
        std::vector<float> policy;
        Stone player;
        int action;  // the action taken at this step
    };
    std::vector<Step> trajectory;

    while (!game.game_over && game.move_count < config.max_moves_per_game) {
        float temp = (game.move_count < config.temperature_threshold)
                     ? 1.0f : 0.0f;

        // reuse_tree=true — previous iteration called make_move() so the
        // root is already positioned at the current game state.  Dirichlet
        // noise is refreshed each move because make_move() clears the flag.
        std::vector<float> pi;
        int action = mcts.get_action(game, pi, temp, -1, /*add_noise=*/true,
                                      /*reuse_tree=*/true);

        Step step;
        game.encode(step.state);
        step.policy = pi;
        step.player = game.current_player;
        step.action = action;
        trajectory.push_back(std::move(step));

        game.play(action);

        mcts.make_move(action);
    }

    if (!game.game_over) game.force_draw();

    auto [red_score, black_score_total] = game.score();
    float black_score = black_score_total - red_score;

    std::vector<float> black_ownership, red_ownership;
    game.get_ownership(BLACK, black_ownership);
    game.get_ownership(RED, red_ownership);

    std::vector<TrainingRecord> records;
    records.reserve(trajectory.size() * 2);

    for (size_t i = 0; i < trajectory.size(); i++) {
        auto& step = trajectory[i];
        float value;
        if      (game.winner == EMPTY)        value =  0.0f;
        else if (game.winner == step.player)  value =  1.0f;
        else                                  value = -1.0f;

        // Score from current player's perspective (raw points)
        float score = (step.player == BLACK) ? black_score : -black_score;

        // Ownership from current player's perspective
        const auto& ownership = (step.player == BLACK) ? black_ownership : red_ownership;

        // Opponent's next action (look-ahead by one step)
        int opponent_action = -1;
        if (i + 1 < trajectory.size()) {
            opponent_action = trajectory[i + 1].action;
        }

        augment_sample(step.state, step.policy, value, score,
                       ownership, opponent_action,
                       config.board_rows, config.board_cols,
                       config.input_channels, records);
    }

    return records;
}

std::vector<TrainingRecord> self_play_game(
        BatchEvaluator* evaluator, const Config& config) {
    MCTS mcts(evaluator, config);
    return self_play_game_impl(mcts, config);
}

}  // namespace minigo
