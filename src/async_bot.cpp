#include "async_bot.h"

#include <algorithm>
#include <chrono>

namespace minigo {

AsyncBot::AsyncBot(BatchEvaluator* evaluator, const Config& config)
    : evaluator_(evaluator),
      config_(config),
      game_(config.board_size, config.komi),
      mcts_(std::make_unique<MCTS>(evaluator, config)) {}

AsyncBot::~AsyncBot() {
    stop_analyze_internal();
}

// ── Game state ──────────────────────────────────────────────

void AsyncBot::reset(const GoGame& initial_game) {
    stop_analyze_internal();
    {
        std::lock_guard<std::mutex> lock(game_mutex_);
        game_ = initial_game;
    }
    mcts_->reset_tree();
}

void AsyncBot::reset() {
    reset(GoGame(config_.board_size, config_.komi));
}

GoGame AsyncBot::game() const {
    std::lock_guard<std::mutex> lock(game_mutex_);
    return game_.copy();
}

// ── Synchronous gen_move ────────────────────────────────────

int AsyncBot::gen_move(Stone color, int num_simulations,
                       float temperature, bool add_noise) {
    // KataGo pattern: stopAndWait before touching the tree.  Caller is
    // responsible for restarting pondering/analyze after gen_move returns.
    stop_analyze_internal();

    // Copy the game so MCTS can safely reference it from search threads.
    GoGame game_copy;
    {
        std::lock_guard<std::mutex> lock(game_mutex_);
        game_copy = game_.copy();
        (void)color;  // Caller asserts the color matches the current player.
    }

    std::vector<float> policy;
    int action = mcts_->get_action(game_copy, policy, temperature,
                                    num_simulations, add_noise,
                                    /*reuse_tree=*/true);

    // Commit the move to the authoritative game and re-root the tree.
    {
        std::lock_guard<std::mutex> lock(game_mutex_);
        if (action == config_.action_size() - 1)
            game_.play(PASS_MOVE);
        else
            game_.play(action);
    }
    mcts_->make_move(action);

    return action;
}

// ── External move (human, opponent) ─────────────────────────

void AsyncBot::play_move(Stone color, int action) {
    // KataGo pattern: stopAndWait before touching the tree.  Caller is
    // responsible for restarting pondering/analyze after play_move returns.
    stop_analyze_internal();

    {
        std::lock_guard<std::mutex> lock(game_mutex_);
        (void)color;  // caller asserts correct color
        if (action == config_.action_size() - 1 || action == PASS_MOVE)
            game_.play(PASS_MOVE);
        else
            game_.play(action);
    }
    // Tree reuse: promote the played child.  If the child doesn't
    // exist (tree wasn't deep enough) make_move drops the tree and
    // the next search rebuilds from scratch.
    int tree_action = (action == PASS_MOVE) ? (config_.action_size() - 1) : action;
    mcts_->make_move(tree_action);
}

// ── Async analyze ───────────────────────────────────────────

void AsyncBot::start_analyze(AnalysisCallback callback, int interval_ms,
                             int max_pv_moves) {
    if (analyzing_.load(std::memory_order_acquire)) return;
    if (!callback) return;

    callback_           = std::move(callback);
    callback_interval_ms_ = std::max(50, interval_ms);
    callback_pv_moves_  = max_pv_moves;
    analyzing_.store(true, std::memory_order_release);

    search_thread_   = std::thread(&AsyncBot::search_loop_worker,  this);
    callback_thread_ = std::thread(&AsyncBot::callback_loop_worker, this);
}

void AsyncBot::stop_analyze() {
    stop_analyze_internal();
}

void AsyncBot::stop_analyze_internal() {
    if (!analyzing_.load(std::memory_order_acquire)) return;

    analyzing_.store(false, std::memory_order_release);
    mcts_->request_stop();  // wake the search threads

    {
        std::lock_guard<std::mutex> lock(callback_mutex_);
    }
    callback_cv_.notify_all();

    if (search_thread_.joinable())   search_thread_.join();
    if (callback_thread_.joinable()) callback_thread_.join();
}

// ── One-shot snapshot ───────────────────────────────────────

MCTS::AnalysisInfo AsyncBot::get_analysis(int max_moves) const {
    return mcts_->get_analysis(max_moves);
}

// ── Background workers ──────────────────────────────────────

void AsyncBot::search_loop_worker() {
    // Snapshot the game state so the search can reference it safely.
    // If play_move() commits a move later, stop_analyze_internal
    // signals request_stop() and this search returns promptly.
    GoGame game_copy;
    {
        std::lock_guard<std::mutex> lock(game_mutex_);
        game_copy = game_.copy();
    }

    // Huge sim budget → effectively "run until request_stop()".
    const int kBigSims = 1'000'000'000;
    std::vector<float> visits;
    try {
        mcts_->search(game_copy, visits, kBigSims,
                       /*add_noise=*/false, /*reuse_tree=*/true);
    } catch (...) {
        // Swallow — background thread must not crash the host.
    }
}

void AsyncBot::callback_loop_worker() {
    std::unique_lock<std::mutex> lock(callback_mutex_);
    while (analyzing_.load(std::memory_order_acquire)) {
        callback_cv_.wait_for(
            lock,
            std::chrono::milliseconds(callback_interval_ms_),
            [&] { return !analyzing_.load(std::memory_order_acquire); });

        if (!analyzing_.load(std::memory_order_acquire)) break;

        // Release the callback_mutex_ while invoking the user callback —
        // the callback may be slow and must not block stop_analyze.
        lock.unlock();
        try {
            auto info = mcts_->get_analysis(callback_pv_moves_);
            if (callback_) callback_(info);
        } catch (...) {
            // Swallow — a buggy callback must not crash the bot.
        }
        lock.lock();
    }
}

}  // namespace minigo
