#include "async_bot.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <random>
#include <stdexcept>

namespace minigo {

// ================================================================
// Helper: pick an action from visit counts
// ================================================================
static int pick_action(const std::vector<float>& visits, float temperature) {
    int action_size = (int)visits.size();
    if (action_size == 0) return 0;

    if (temperature <= 0.0f) {
        // Greedy
        return (int)(std::max_element(visits.begin(), visits.end()) - visits.begin());
    }

    // Stochastic sampling with temperature
    std::vector<float> weights(action_size);
    float sum = 0.0f;
    for (int a = 0; a < action_size; a++) {
        weights[a] = std::pow(visits[a], 1.0f / temperature);
        sum += weights[a];
    }

    if (sum <= 0.0f) {
        // Fallback: greedy
        return (int)(std::max_element(visits.begin(), visits.end()) - visits.begin());
    }

    static thread_local std::mt19937 rng{std::random_device{}()};
    std::discrete_distribution<int> dist(weights.begin(), weights.end());
    return dist(rng);
}

// ================================================================
// AsyncBot
// ================================================================

AsyncBot::AsyncBot(BatchEvaluator* evaluator, const Config& config)
    : evaluator_(evaluator),
      config_(config),
      game_(config.history_length),
      mcts_(std::make_unique<MCTS>(evaluator, config)) {
    worker_thread_ = std::thread(&AsyncBot::worker_loop, this);
}

AsyncBot::~AsyncBot() {
    {
        std::unique_lock<std::mutex> lock(control_mutex_);
        stop_locked(lock);                // wait for any running search
        pending_mode_ = Mode::SHUTDOWN;
    }
    worker_cv_.notify_one();
    if (worker_thread_.joinable()) worker_thread_.join();
}

// ── Callback configuration ──────────────────────────────────

void AsyncBot::set_callback(AnalysisCallback cb, int interval_ms,
                            int max_pv_moves) {
    std::lock_guard<std::mutex> lock(control_mutex_);
    callback_             = std::move(cb);
    callback_interval_ms_ = std::max(50, interval_ms);
    callback_pv_moves_    = max_pv_moves;
}

void AsyncBot::clear_callback() {
    std::lock_guard<std::mutex> lock(control_mutex_);
    callback_ = nullptr;
}

// ── Game state ──────────────────────────────────────────────

void AsyncBot::reset(const XiangqiGame& initial_game) {
    {
        std::unique_lock<std::mutex> lock(control_mutex_);
        stop_locked(lock);
    }
    {
        std::lock_guard<std::mutex> lock(game_mutex_);
        game_ = initial_game;
    }
    mcts_->reset_tree();
}

void AsyncBot::reset() {
    reset(XiangqiGame(config_.history_length));
}

XiangqiGame AsyncBot::game() const {
    std::lock_guard<std::mutex> lock(game_mutex_);
    return game_.copy();
}

// ── Synchronous gen_move ────────────────────────────────────

int AsyncBot::gen_move(Stone color, int num_simulations,
                       float temperature, bool add_noise) {
    // Validate color (outside control_mutex_ because it needs game_mutex_)
    {
        std::lock_guard<std::mutex> gl(game_mutex_);
        if (color != EMPTY && color != game_.current_player) {
            throw std::runtime_error(
                "AsyncBot::gen_move: color does not match current player");
        }
    }

    std::unique_lock<std::mutex> lock(control_mutex_);

    // Stop any running async search, wait for worker to be idle.
    stop_locked(lock);

    // Submit GENMOVE request.
    gen_move_sims_    = num_simulations;
    gen_move_temp_    = temperature;
    gen_move_noise_   = add_noise;
    pending_mode_     = Mode::GENMOVE;

    worker_cv_.notify_one();

    // Wait for the worker to pick up the request, run it, and return to IDLE.
    done_cv_.wait(lock, [this] {
        return pending_mode_ == Mode::IDLE && current_mode_ == Mode::IDLE;
    });

    return gen_move_result_;
}

// ── External move ───────────────────────────────────────────

bool AsyncBot::play_move(Stone color, int action) {
    {
        std::unique_lock<std::mutex> lock(control_mutex_);
        stop_locked(lock);
    }

    {
        std::lock_guard<std::mutex> gl(game_mutex_);
        if (color != EMPTY && color != game_.current_player) return false;
        if (!game_.is_legal(action))                         return false;
        game_.play(action);
    }
    mcts_->make_move(action);
    return true;
}

// ── Async ponder / analyze ──────────────────────────────────

void AsyncBot::start_ponder() {
    std::unique_lock<std::mutex> lock(control_mutex_);
    stop_locked(lock);  // cancel any previous search

    pending_mode_ = Mode::PONDER;
    worker_cv_.notify_one();
    // Do NOT wait for completion — ponder runs in the background.
}

void AsyncBot::start_analyze(AnalysisCallback cb, int interval_ms,
                             int max_pv_moves) {
    {
        std::unique_lock<std::mutex> lock(control_mutex_);
        stop_locked(lock);
        callback_             = std::move(cb);
        callback_interval_ms_ = std::max(50, interval_ms);
        callback_pv_moves_    = max_pv_moves;
        pending_mode_         = Mode::PONDER;
    }
    worker_cv_.notify_one();
}

void AsyncBot::stop() {
    std::unique_lock<std::mutex> lock(control_mutex_);
    stop_locked(lock);
}

// Called with control_mutex_ held.  Cancels any pending/current
// search and waits for the worker to become idle.
void AsyncBot::stop_locked(std::unique_lock<std::mutex>& lock) {
    if (pending_mode_ == Mode::IDLE && current_mode_ == Mode::IDLE) return;

    // Cancel any pending request that hasn't been picked up yet.
    if (pending_mode_ != Mode::SHUTDOWN) {
        pending_mode_ = Mode::IDLE;
    }

    // Signal any running search to exit.  request_stop is safe to call
    // even if no search is running (it just sets an atomic flag that
    // the next search() will reset).
    lock.unlock();
    mcts_->request_stop();
    lock.lock();

    // Wait for the worker's current iteration to finish and set IDLE.
    done_cv_.wait(lock, [this] {
        return current_mode_ == Mode::IDLE
            && (pending_mode_ == Mode::IDLE || pending_mode_ == Mode::SHUTDOWN);
    });
}

// ── Queries ────────────────────────────────────────────────

MCTS::AnalysisInfo AsyncBot::get_analysis(int max_moves) const {
    return mcts_->get_analysis(max_moves);
}

bool AsyncBot::is_searching() const {
    std::lock_guard<std::mutex> lock(control_mutex_);
    return current_mode_ != Mode::IDLE || pending_mode_ != Mode::IDLE;
}

// ================================================================
// Worker thread — persistent, handles all search modes
// ================================================================

void AsyncBot::worker_loop() {
    while (true) {
        // ── Wait for a work request ──
        Mode              mode;
        int               req_sims;
        float             req_temp;
        bool              req_noise;
        AnalysisCallback  req_callback;
        int               req_interval;
        int               req_pv;
        {
            std::unique_lock<std::mutex> lock(control_mutex_);
            worker_cv_.wait(lock, [this] {
                return pending_mode_ != Mode::IDLE;
            });

            mode          = pending_mode_;
            current_mode_ = (mode == Mode::SHUTDOWN) ? Mode::IDLE : mode;
            pending_mode_ = Mode::IDLE;

            if (mode == Mode::SHUTDOWN) break;

            req_sims     = gen_move_sims_;
            req_temp     = gen_move_temp_;
            req_noise    = gen_move_noise_;
            req_callback = callback_;
            req_interval = callback_interval_ms_;
            req_pv       = callback_pv_moves_;

            // Clear the stop flag atomically with the mode transition.
            // This eliminates the race where a stop_locked() call could
            // set should_stop_ AFTER we release control_mutex_ but BEFORE
            // we entered MCTS::search() and had search() clear it.
            // Any stop() issued after we release the lock below will
            // re-set the flag cleanly, and search() will see it.
            mcts_->reset_stop_flag();
        }

        // ── Snapshot game state for the search ──
        XiangqiGame game_copy;
        {
            std::lock_guard<std::mutex> gl(game_mutex_);
            game_copy = game_.copy();
        }

        // ── Spawn callback thread if configured ──
        std::mutex               cb_mutex;
        std::condition_variable  cb_cv;
        bool                     cb_stop = false;
        std::thread              cb_thread;
        if (req_callback) {
            cb_thread = std::thread([this, req_callback, req_interval, req_pv,
                                     &cb_mutex, &cb_cv, &cb_stop]() {
                std::unique_lock<std::mutex> lock(cb_mutex);
                while (!cb_stop) {
                    cb_cv.wait_for(
                        lock,
                        std::chrono::milliseconds(req_interval),
                        [&] { return cb_stop; });
                    if (cb_stop) break;
                    lock.unlock();
                    try {
                        auto info = mcts_->get_analysis(req_pv);
                        req_callback(info);
                    } catch (...) {
                        // Swallow — buggy callback must not crash the bot.
                    }
                    lock.lock();
                }
            });
        }

        // ── Run the search ──
        int chosen_action = -1;
        bool search_ok = false;
        try {
            int search_sims;
            bool search_noise;
            if (mode == Mode::GENMOVE) {
                search_sims  = (req_sims < 0) ? config_.num_simulations : req_sims;
                search_noise = req_noise;
            } else {
                // PONDER: effectively unlimited sims; stopped by request_stop.
                search_sims  = 1'000'000'000;
                search_noise = false;
            }

            std::vector<float> visits;
            mcts_->search(game_copy, visits, search_sims,
                          search_noise, /*reuse_tree=*/true);
            search_ok = true;

            if (mode == Mode::GENMOVE) {
                chosen_action = pick_action(visits, req_temp);
            }
        } catch (const std::exception& e) {
            std::cerr << "AsyncBot: search failed: " << e.what() << "\n";
        } catch (...) {
            std::cerr << "AsyncBot: search failed (unknown exception)\n";
        }

        // ── Stop and join callback thread ──
        if (cb_thread.joinable()) {
            {
                std::lock_guard<std::mutex> lock(cb_mutex);
                cb_stop = true;
            }
            cb_cv.notify_all();
            cb_thread.join();
        }

        // ── Commit GENMOVE result (after search fully returned) ──
        if (mode == Mode::GENMOVE && search_ok && chosen_action >= 0) {
            {
                std::lock_guard<std::mutex> gl(game_mutex_);
                if (game_.is_legal(chosen_action)) {
                    game_.play(chosen_action);
                } else {
                    chosen_action = -1;  // caller will see -1 as "failed"
                }
            }
            if (chosen_action >= 0) {
                mcts_->make_move(chosen_action);
            }
        }

        // ── Mark idle, store result, notify waiters ──
        {
            std::lock_guard<std::mutex> lock(control_mutex_);
            if (mode == Mode::GENMOVE) {
                gen_move_result_ = chosen_action;
            }
            current_mode_ = Mode::IDLE;
        }
        done_cv_.notify_all();
    }

    // Shutdown path: ensure current_mode_ is IDLE so anyone still waiting
    // on done_cv_ can exit.
    {
        std::lock_guard<std::mutex> lock(control_mutex_);
        current_mode_ = Mode::IDLE;
    }
    done_cv_.notify_all();
}

}  // namespace minigo
