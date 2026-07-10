#pragma once

#include "config.h"
#include "game.h"
#include "mcts.h"
#include "batch_evaluator.h"

#include <condition_variable>
#include <functional>
#include <memory>
#include <mutex>
#include <thread>

namespace minigo {

// ================================================================
// AsyncBot — KataGo-style wrapper around MCTS + GoGame
//
// Architecture: one persistent worker thread runs `worker_loop()` for
// the bot's lifetime, idle-waiting on a condvar between searches.  All
// search modes (GENMOVE, PONDER) go through the same worker.  When a
// callback is configured, the worker spawns a transient callback
// thread per search that polls `get_analysis()` at a fixed interval
// and invokes the user callback; the callback thread is joined at
// the end of each search.
//
// API separates three orthogonal concerns:
//
//   1. Callback reporting (persistent, mode-independent):
//      set_callback(cb, interval_ms, max_pv)
//      clear_callback()
//
//   2. Synchronous operations (caller blocks until done):
//      gen_move(color, ...)    — AI picks and plays a move
//      play_move(color, act)   — commit a human/opponent move
//
//   3. Asynchronous operations (caller returns immediately):
//      start_ponder()                  — background search, no sim cap
//      start_analyze(cb, ...)          — set_callback + start_ponder
//      stop()                          — stop async search, wait idle
//
// Typical usage for "game with live analysis" (AI vs human):
//
//   bot->set_callback(on_analysis_update, 500, 10);
//   while (game in progress) {
//       if (analysis_wanted && !bot->is_searching())
//           bot->start_ponder();        // callback fires during ponder
//
//       if (human_turn) {
//           bot->play_move(color, action);  // stops ponder, advances
//       } else {
//           bot->gen_move(color);            // stops ponder, runs
//                                            // GENMOVE search (callback
//                                            // still fires during it),
//                                            // advances game + tree
//       }
//   }
//
// Thread safety: single-writer contract.  gen_move / play_move /
// start_ponder / start_analyze / stop / set_callback / clear_callback /
// reset must all be called from one "control" thread.  get_analysis()
// and game() are read-only and safe from any thread.
// ================================================================
class AsyncBot {
public:
    AsyncBot(BatchEvaluator* evaluator, const Config& config);
    ~AsyncBot();

    AsyncBot(const AsyncBot&)            = delete;
    AsyncBot& operator=(const AsyncBot&) = delete;

    using AnalysisCallback = std::function<void(const MCTS::AnalysisInfo&)>;

    // ── Callback configuration ──────────────────────────────
    // The callback is persistent and applies to every subsequent
    // search (GENMOVE or PONDER).  Setting while a search is
    // running is safe; the current search keeps its snapshotted
    // callback, the next search picks up the new one.
    void set_callback(AnalysisCallback cb, int interval_ms = 500,
                      int max_pv_moves = 10);
    void clear_callback();

    // ── Game state ──────────────────────────────────────────
    // Start a new game from the given position (default: empty board
    // with configured komi).  Stops any running search and clears tree.
    void reset(const GoGame& initial_game);
    void reset();

    // Thread-safe snapshot of the authoritative game state.
    GoGame game() const;

    // ── Synchronous gen_move (AI move) ──────────────────────
    // Stops any running async search, runs a blocking GENMOVE search
    // through the worker thread, picks an action, advances game + tree.
    // Returns the played action.  Throws if color doesn't match the
    // current player (pass EMPTY to skip the check).
    int gen_move(Stone color, int num_simulations = -1,
                 float temperature = 0.0f, bool add_noise = false);

    // ── External move (human / opponent) ────────────────────
    // Stops any running async search.  Validates color + legality;
    // returns false if either check fails (game + tree unchanged).
    // Returns true on success.  Pass color=EMPTY to skip color check.
    bool play_move(Stone color, int action);

    // ── Async background search ─────────────────────────────
    // Submits a PONDER request to the worker and returns immediately.
    // If a callback is configured, it fires every interval_ms until
    // stop() is called.  No sim cap — runs until stop().
    void start_ponder();

    // Convenience: set_callback(cb, ...) + start_ponder().
    void start_analyze(AnalysisCallback cb, int interval_ms = 500,
                       int max_pv_moves = 10);

    // Stop any async search (ponder/analyze).  Blocks until the
    // worker has returned to idle state.  Safe no-op if idle.
    void stop();

    // ── Queries ─────────────────────────────────────────────
    MCTS::AnalysisInfo get_analysis(int max_moves = 5) const;
    bool is_searching() const;

private:
    enum class Mode { IDLE, GENMOVE, PONDER, SHUTDOWN };

    void worker_loop();
    void stop_locked(std::unique_lock<std::mutex>& lock);  // control_mutex_ must be held

    Config          config_;

    mutable std::mutex       game_mutex_;  // protects game_
    GoGame                   game_;

    std::unique_ptr<MCTS>    mcts_;

    // ── Worker coordination (all protected by control_mutex_) ──
    mutable std::mutex       control_mutex_;
    std::condition_variable  worker_cv_;   // worker waits on this
    std::condition_variable  done_cv_;     // callers wait on this
    Mode                     pending_mode_ = Mode::IDLE;
    Mode                     current_mode_ = Mode::IDLE;

    // Request parameters
    int   gen_move_sims_  = -1;
    float gen_move_temp_  = 0.0f;
    bool  gen_move_noise_ = false;
    int   gen_move_result_ = -1;

    AnalysisCallback callback_;
    int              callback_interval_ms_ = 500;
    int              callback_pv_moves_    = 10;

    // Persistent worker (lifetime = lifetime of AsyncBot)
    std::thread      worker_thread_;
};

}  // namespace minigo
