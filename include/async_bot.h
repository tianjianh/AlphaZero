#pragma once

#include "config.h"
#include "game.h"
#include "mcts.h"
#include "batch_evaluator.h"

#include <atomic>
#include <condition_variable>
#include <functional>
#include <memory>
#include <mutex>
#include <thread>

namespace minigo {

// ================================================================
// AsyncBot — KataGo-style wrapper around MCTS + GoGame
//
// Owns:
//   - one MCTS instance (persistent tree across moves)
//   - one GoGame instance (authoritative game state)
//   - optional background search thread (pondering / analyze)
//   - optional callback thread (periodic AnalysisInfo snapshots)
//
// Lifecycle patterns:
//
//   Synchronous gen_move (eval, selfplay, AI-move in play):
//     int action = bot.gen_move(color, sims, temperature, add_noise);
//     // tree is re-rooted to `action` automatically; game is advanced.
//
//   Playing a move from an external source (human, opponent):
//     bot.play_move(color, action);
//     // stops any background search, advances game + tree, optionally
//     // resumes analyze mode if it was active.
//
//   Live analysis with periodic callbacks (play UI, web backend):
//     bot.start_analyze([](const MCTS::AnalysisInfo& info) {
//         // called on a background thread every interval_ms
//     }, /*interval_ms=*/500);
//     // ... elsewhere ...
//     bot.stop_analyze();
//
// Thread safety: AsyncBot is a single-writer design.  State-mutating
// methods (gen_move, play_move, start_analyze, stop_analyze, reset)
// must be called from a single "control" thread.  The background
// search and callback threads are owned by AsyncBot and never call
// public methods on it.  get_analysis() and game() are read-only and
// safe to call from any thread.
// ================================================================
class AsyncBot {
public:
    AsyncBot(BatchEvaluator* evaluator, const Config& config);
    ~AsyncBot();

    AsyncBot(const AsyncBot&)            = delete;
    AsyncBot& operator=(const AsyncBot&) = delete;

    // ── Game state ──────────────────────────────────────────
    // Start a new game from the given position (or default-constructed).
    // Stops any background search and clears the tree.
    void reset(const GoGame& initial_game);
    void reset();  // board_size/komi from config

    // Snapshot of the current game state (thread-safe copy).
    GoGame game() const;

    // ── Synchronous move selection (AI move) ────────────────
    // Stops any background analyze, runs a blocking MCTS search, picks
    // an action, advances game + tree.  Returns the played action.
    // Caller is responsible for restarting analyze afterwards if wanted.
    int gen_move(Stone color, int num_simulations = -1,
                 float temperature = 0.0f, bool add_noise = false);

    // ── Play an externally-chosen move (human / opponent) ───
    // Stops any background analyze, advances game + tree.  Caller is
    // responsible for restarting analyze afterwards if wanted.
    void play_move(Stone color, int action);

    // ── Async analysis ──────────────────────────────────────
    using AnalysisCallback = std::function<void(const MCTS::AnalysisInfo&)>;

    // Start a background search + a periodic callback that fires
    // every interval_ms with the current live analysis.  Returns
    // immediately.  No-op if already analyzing.
    void start_analyze(AnalysisCallback callback, int interval_ms = 500,
                       int max_pv_moves = 10);

    // Stop background search and callback thread.  Blocks until both
    // threads have joined.  No-op if not analyzing.
    void stop_analyze();

    bool is_analyzing() const { return analyzing_.load(std::memory_order_acquire); }

    // ── One-shot analysis snapshot ──────────────────────────
    // Returns the current tree state (locks tree_mutex_ briefly).
    MCTS::AnalysisInfo get_analysis(int max_moves = 5) const;

    // Direct access to the underlying MCTS (for niche needs).
    MCTS* mcts() { return mcts_.get(); }

private:
    // Run search on the game copy.  Returns when search completes
    // naturally (sim count reached) or should_stop_ is set.
    void search_loop_worker();

    // Wakes every callback_interval_ms_, calls callback_ with a
    // fresh analysis snapshot, loops until analyzing_ becomes false.
    void callback_loop_worker();

    // Internal stop: joins both threads without taking game_mutex_.
    // Caller must NOT hold game_mutex_.
    void stop_analyze_internal();

    BatchEvaluator* evaluator_;
    Config          config_;

    mutable std::mutex       game_mutex_;  // protects game_
    GoGame                   game_;

    std::unique_ptr<MCTS>    mcts_;

    // ── Background search + callback threads ───────────────
    std::thread              search_thread_;
    std::thread              callback_thread_;
    std::atomic<bool>        analyzing_{false};

    // Callback wakeup coordination
    std::mutex               callback_mutex_;
    std::condition_variable  callback_cv_;
    AnalysisCallback         callback_;
    int                      callback_interval_ms_ = 500;
    int                      callback_pv_moves_    = 10;
};

}  // namespace minigo
