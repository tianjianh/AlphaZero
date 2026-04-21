// XQWL-vs-XQWL bootstrap data generator.
//
// Emits V4 self-play records from XQWL06 alpha-beta play.  The bootstrap
// output trains v0000 on a pile of decisive, positionally-reasonable games
// so that the NN-based self-play loop doesn't start from a completely
// uniform policy that cannot survive long enough to produce a decisive
// outcome.
//
// Records match the live self-play format: policy is one-hot on the move
// the XQWL engine chose (no MCTS distribution is available from a pure
// alpha-beta engine); value is the terminal outcome from each record's own
// player perspective; ownership is the terminal board from that POV;
// opponent_action is the next move in the trajectory, -1 at terminal.

#include "game.h"
#include "mcts.h"
#include "training_io.h"
#include "xqwl_engine.h"

#include <atomic>
#include <chrono>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_set>
#include <vector>

using namespace minigo;

namespace {

struct BootstrapConfig {
    int games = 100;
    int depth = 6;
    int time_ms = 0;        // 0 = rely on depth only
    int threads = 1;
    int max_plies = 300;
    std::string output_dir = "training/bootstrap";
};

void print_usage() {
    std::cout
        << "Usage: bootstrap [options]\n"
        << "  --games N          Number of games to generate (default: 100)\n"
        << "  --depth D          XQWL max search depth (default: 6)\n"
        << "  --time-ms T        XQWL per-move time budget in ms (default: 0 = off)\n"
        << "  --threads T        Worker threads (default: 1)\n"
        << "  --max-plies P      Ply cap per game (default: 300, matches self-play)\n"
        << "  --output DIR       Output directory (default: training/bootstrap)\n";
}

bool parse_args(int argc, char** argv, BootstrapConfig& cfg) {
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto need = [&](const char* flag) {
            if (i + 1 >= argc) {
                std::cerr << "missing value for " << flag << "\n";
                std::exit(1);
            }
            return std::string(argv[++i]);
        };
        if      (a == "--games")     cfg.games     = std::stoi(need("--games"));
        else if (a == "--depth")     cfg.depth     = std::stoi(need("--depth"));
        else if (a == "--time-ms")   cfg.time_ms   = std::stoi(need("--time-ms"));
        else if (a == "--threads")   cfg.threads   = std::stoi(need("--threads"));
        else if (a == "--max-plies") cfg.max_plies = std::stoi(need("--max-plies"));
        else if (a == "--output")    cfg.output_dir = need("--output");
        else if (a == "--help" || a == "-h") { print_usage(); return false; }
        else { std::cerr << "unknown flag: " << a << "\n"; print_usage(); std::exit(1); }
    }
    return true;
}

struct Step {
    std::vector<float> state;
    int action;
    Stone player;
};

// One XQWL-vs-XQWL game.  Returns the record trajectory for the entire
// game, already augmented (original + mirror per ply).
std::vector<TrainingRecord> play_one_game(xqwl::XqwlEngine& engine,
                                          const BootstrapConfig& cfg) {
    engine.reset();
    XiangqiGame game(/*history_length=*/4);

    std::vector<Step> trajectory;
    trajectory.reserve(cfg.max_plies);

    int action_size = XiangqiGame::action_size();
    Stone result_winner = EMPTY;

    while ((int)trajectory.size() < cfg.max_plies) {
        auto st = engine.status();
        if (st == xqwl::Status::CHECKMATE) {
            // Side to move (our current player) is mated → opponent wins.
            result_winner = opponent(game.current_player);
            break;
        }
        if (st == xqwl::Status::STALEMATE) {
            // In Xiangqi stalemate is a loss for the side unable to move.
            result_winner = opponent(game.current_player);
            break;
        }
        if (st == xqwl::Status::REPETITION_DRAW) {
            result_winner = EMPTY;
            break;
        }
        if (st == xqwl::Status::SELF_PERPETUAL) {
            // side to move caused the perpetual → side to move loses
            result_winner = opponent(game.current_player);
            break;
        }
        if (st == xqwl::Status::OPP_PERPETUAL) {
            result_winner = game.current_player;
            break;
        }

        int mv = engine.think();
        if (mv == 0) {
            // No legal move and not detected above — treat as current-side loss.
            result_winner = opponent(game.current_player);
            break;
        }

        int action = xqwl::xqwl_move_to_action(mv);
        if (action < 0 || action >= action_size || !game.is_legal(action)) {
            // XQWL and our move-gen disagreed — abort this game (shouldn't happen).
            result_winner = EMPTY;
            break;
        }

        Step step;
        game.encode(step.state);
        step.action = action;
        step.player = game.current_player;
        trajectory.push_back(std::move(step));

        // Apply to both engines in lockstep.
        engine.play(mv);
        game.play(action);

        if (game.game_over) {
            result_winner = game.winner;
            break;
        }
    }

    if (!game.game_over && result_winner == EMPTY && (int)trajectory.size() == cfg.max_plies) {
        // Ran out of plies without a decisive result.  Force a draw so the
        // record write-out sees a consistent terminal board.
        game.force_draw();
        result_winner = game.winner;
    } else if (!game.game_over) {
        // Engine-side termination (checkmate/repetition/stalemate) that
        // XiangqiGame didn't mark as over.  Apply force_draw so we get a
        // valid terminal state, then overwrite winner below.
        game.force_draw();
    }

    std::vector<float> red_final, black_final;
    game.get_final_ownership_from(RED, red_final);
    game.get_final_ownership_from(BLACK, black_final);

    std::vector<TrainingRecord> records;
    records.reserve(trajectory.size() * 2);

    int area = game.board_rows * game.board_cols;
    int input_channels = (game.history_length) * 14 + 1;

    for (size_t i = 0; i < trajectory.size(); ++i) {
        auto& step = trajectory[i];
        float value;
        if      (result_winner == EMPTY)        value =  0.0f;
        else if (result_winner == step.player)  value =  1.0f;
        else                                    value = -1.0f;

        std::vector<float> policy(action_size, 0.0f);
        policy[step.action] = 1.0f;

        const auto& ownership = (step.player == BLACK) ? black_final : red_final;

        int opp_action = -1;
        if (i + 1 < trajectory.size()) opp_action = trajectory[i + 1].action;

        augment_sample(step.state, policy, value, ownership, opp_action,
                       game.board_rows, game.board_cols,
                       input_channels, records);
        (void)area;
    }
    return records;
}

// Scan the output directory for already-finished games.  Supports both
// interruption-mid-run resume (some game_N.bin files were written before
// the crash) and additive runs (existing game_0..game_M-1.bin[.zst] + a
// larger --games count → only the new IDs are generated).  Recognizes both
// the raw .bin and the post-compression .bin.zst suffix so a second pass
// through compress_selfplay stays resume-safe.
std::unordered_set<int> scan_existing_game_ids(const std::string& dir) {
    std::unordered_set<int> ids;
    std::error_code ec;
    if (!std::filesystem::exists(dir, ec)) return ids;
    for (auto& entry : std::filesystem::directory_iterator(dir, ec)) {
        if (ec) break;
        if (!entry.is_regular_file()) continue;
        std::string name = entry.path().filename().string();
        if (name.rfind("game_", 0) != 0) continue;
        // Strip .bin.zst or .bin suffix.
        std::string base = name.substr(5);
        auto dot = base.find('.');
        if (dot == std::string::npos) continue;
        std::string suffix = base.substr(dot);
        if (suffix != ".bin" && suffix != ".bin.zst") continue;
        try {
            int id = std::stoi(base.substr(0, dot));
            if (entry.file_size(ec) > 0 && !ec) ids.insert(id);
        } catch (...) { /* skip non-numeric */ }
    }
    return ids;
}

}  // namespace

int main(int argc, char** argv) {
    BootstrapConfig cfg;
    if (!parse_args(argc, argv, cfg)) return 0;

    std::filesystem::create_directories(cfg.output_dir);

    auto done_ids = scan_existing_game_ids(cfg.output_dir);
    int already_done = 0;
    for (int id : done_ids) if (id < cfg.games) ++already_done;
    int to_generate = std::max(0, cfg.games - already_done);

    std::cout << "XQWL bootstrap generator\n"
              << "  Games:       " << cfg.games << "\n"
              << "  Depth:       " << cfg.depth << "\n"
              << "  Time-ms:     " << cfg.time_ms << "\n"
              << "  Threads:     " << cfg.threads << "\n"
              << "  Max plies:   " << cfg.max_plies << "\n"
              << "  Output:      " << cfg.output_dir << "\n";
    if (!done_ids.empty()) {
        std::cout << "  Existing:    " << already_done
                  << " of " << cfg.games << " (resume)\n"
                  << "  To generate: " << to_generate << "\n";
    }
    std::cout << "\n";

    if (to_generate == 0) {
        std::cout << "All " << cfg.games << " games already present — nothing to do.\n";
        return 0;
    }

    std::atomic<int> next_game{0};
    std::atomic<int> total_records{0};
    std::atomic<int> generated{0};
    std::mutex io_mutex;
    auto t_start = std::chrono::steady_clock::now();

    auto worker = [&]() {
        // 16 MB transposition table + per-game state — stays on the heap to
        // keep the worker-thread stack small and avoid stack overflows.
        auto engine = std::make_unique<xqwl::XqwlEngine>();
        engine->set_max_depth(cfg.depth);
        engine->set_time_budget_ms(cfg.time_ms);

        while (true) {
            int id = next_game.fetch_add(1);
            if (id >= cfg.games) break;
            if (done_ids.count(id)) continue;  // already on disk, skip

            auto tg0 = std::chrono::steady_clock::now();
            auto records = play_one_game(*engine, cfg);
            auto tg1 = std::chrono::steady_clock::now();

            std::string path = cfg.output_dir + "/game_" + std::to_string(id) + ".bin";
            write_records(path, records, BOARD_ROWS, BOARD_COLS);
            total_records.fetch_add(static_cast<int>(records.size()));
            int done_now = generated.fetch_add(1) + 1;

            double secs = std::chrono::duration<double>(tg1 - tg0).count();
            int moves = static_cast<int>(records.size()) / 2;

            std::lock_guard<std::mutex> lk(io_mutex);
            std::cout << "  Game " << (id + 1) << "/" << cfg.games
                      << " [" << done_now << "/" << to_generate << " new]"
                      << " — " << moves << " moves, "
                      << records.size() << " samples, "
                      << std::fixed << std::setprecision(2) << secs << "s\n";
        }
    };

    std::vector<std::thread> workers;
    workers.reserve(cfg.threads);
    for (int t = 0; t < cfg.threads; ++t) workers.emplace_back(worker);
    for (auto& t : workers) t.join();

    double total_secs = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - t_start).count();
    int gen = generated.load();
    std::cout << "\nDone! " << gen << " new games, "
              << total_records.load() << " new records in "
              << std::fixed << std::setprecision(2) << total_secs << "s "
              << "(" << (gen > 0 ? total_secs / gen : 0.0) << "s/game)\n";
    return 0;
}
