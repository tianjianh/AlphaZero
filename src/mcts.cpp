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
    MCTSNode* best = nullptr;
    float best_score = -1e9f;
    for (auto& child : children) {
        if (!child) continue;
        float score = child->ucb_score(c_puct);
        if (score > best_score) {
            best_score = score;
            best = child.get();
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
        node->state.store(NODE_EXPANDED, std::memory_order_release);

        // Blend value + score for utility (KataGo-style atan compression)
        float utility = result.value;
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

void MCTS::search_single_threaded(MCTSNode* root, const GoGame& game,
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

        // Pre-allocate GoGame on heap — reused across batch
        auto game_copy_ptr = std::make_unique<GoGame>(game.copy());

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
                if (action == action_size - 1)
                    game_copy_ptr->play(PASS_MOVE);
                else
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
            leaf.leaf->state.store(NODE_EXPANDED, std::memory_order_release);

            float utility = res.value;
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

void MCTS::search(GoGame& game, std::vector<float>& visits,
                  int num_simulations, bool add_noise) {
    if (num_simulations < 0) num_simulations = config_.num_simulations;
    should_stop_.store(false, std::memory_order_relaxed);

    action_size_ = config_.action_size();
    int action_size = action_size_;

    // ── Build new root tree in a local (outside any lock) ───────
    // GPU evaluation takes ms, so we do it without holding tree_mutex_,
    // then atomically swap into root_ once the new tree is ready.
    auto new_root = std::make_unique<MCTSNode>();

    std::vector<float> state_enc;
    game.encode(state_enc);
    auto root_results = evaluator_->evaluate({ state_enc });
    auto& root_out = root_results[0];
    float new_root_nn_score = root_out.score;

    std::vector<float> legal;
    game.get_legal_moves(legal);
    mask_policy(root_out.policy, legal, action_size);
    expand(new_root.get(), root_out.policy, legal);
    new_root->state.store(NODE_EXPANDED, std::memory_order_release);
    new_root->visit_count.store(1, std::memory_order_relaxed);
    float root_utility = root_out.value;
    if (config_.score_weight != 0.0f) {
        float root_score_utility = atanf(root_out.score / config_.score_scale) / (float)(M_PI / 2.0);
        root_utility += config_.score_weight * root_score_utility;
    }
    if (!std::isfinite(root_utility)) root_utility = 0.0f;
    new_root->add_value(root_utility);
    if (add_noise) add_dirichlet_noise(new_root.get(), action_size);

    // ── Swap in the new tree (old tree destroyed after lock release) ──
    std::unique_ptr<MCTSNode> old_root;
    {
        std::lock_guard<std::mutex> lock(tree_mutex_);
        old_root = std::move(root_);
        root_ = std::move(new_root);
        root_nn_score_ = new_root_nn_score;
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

int MCTS::get_action(GoGame& game, std::vector<float>& policy,
                     float temperature, int num_simulations,
                     bool add_noise) {
    int action_size = config_.action_size();

    std::vector<float> visits;
    search(game, visits, num_simulations, add_noise);

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
    info.root_score = root_nn_score_;

    if (!root_) return info;

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
// Dihedral augmentation (8-fold symmetry)
// ================================================================
static void augment_sample(const std::vector<float>& state,
                           const std::vector<float>& policy,
                           float value, float score,
                           int board_size, int input_channels,
                           std::vector<TrainingRecord>& out) {
    int n  = board_size;
    int hw = n * n;
    int action_size = hw + 1;
    float pass_prob = policy[hw];

    for (int rot = 0; rot < 4; rot++) {
        for (int flip = 0; flip < 2; flip++) {
            TrainingRecord rec;
            rec.state.resize((size_t)input_channels * hw);
            rec.policy.resize(action_size);
            rec.value = value;
            rec.score = score;

            auto transform = [&](int r, int c) -> std::pair<int,int> {
                int tr = r, tc = c;
                for (int k = 0; k < rot; k++) {
                    int tmp = tr; tr = tc; tc = n - 1 - tmp;
                }
                if (flip) tc = n - 1 - tc;
                return {tr, tc};
            };

            for (int ch = 0; ch < input_channels; ch++) {
                for (int r = 0; r < n; r++) {
                    for (int c = 0; c < n; c++) {
                        auto [tr, tc] = transform(r, c);
                        rec.state[ch * hw + tr * n + tc] =
                            state[ch * hw + r * n + c];
                    }
                }
            }
            for (int r = 0; r < n; r++) {
                for (int c = 0; c < n; c++) {
                    auto [tr, tc] = transform(r, c);
                    rec.policy[tr * n + tc] = policy[r * n + c];
                }
            }
            rec.policy[hw] = pass_prob;
            out.push_back(std::move(rec));
        }
    }
}

// ================================================================
// Self-play game
// ================================================================
static std::vector<TrainingRecord> self_play_game_impl(
        MCTS& mcts, const Config& config) {
    GoGame game(config.board_size, config.komi);

    struct Step {
        std::vector<float> state;
        std::vector<float> policy;
        Stone player;
    };
    std::vector<Step> trajectory;

    int action_size = config.action_size();

    while (!game.game_over && game.move_count < config.max_moves_per_game) {
        float temp = (game.move_count < config.temperature_threshold)
                     ? 1.0f : 0.0f;

        std::vector<float> pi;
        int action = mcts.get_action(game, pi, temp, -1, true);

        Step step;
        game.encode(step.state);
        step.policy = pi;
        step.player = game.current_player;
        trajectory.push_back(std::move(step));

        if (action == action_size - 1)
            game.play(PASS_MOVE);
        else
            game.play(action);
    }

    while (!game.game_over) game.play(PASS_MOVE);

    // Compute score target: raw point difference from BLACK's perspective
    auto [bs, ws] = game.score();
    float black_score = bs - ws;  // raw points, e.g. +12.5

    std::vector<TrainingRecord> records;
    records.reserve(trajectory.size() * 8);

    for (auto& step : trajectory) {
        float value;
        if      (game.winner == EMPTY)        value =  0.0f;
        else if (game.winner == step.player)  value =  1.0f;
        else                                  value = -1.0f;

        // Score from current player's perspective (raw points)
        float score = (step.player == BLACK) ? black_score : -black_score;

        augment_sample(step.state, step.policy, value, score,
                       config.board_size, config.input_channels, records);
    }

    return records;
}

std::vector<TrainingRecord> self_play_game(
        BatchEvaluator* evaluator, const Config& config) {
    MCTS mcts(evaluator, config);
    return self_play_game_impl(mcts, config);
}

}  // namespace minigo
