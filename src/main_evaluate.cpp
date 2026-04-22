#include "config.h"
#include "game.h"
#include "mcts.h"
#include "loaded_model.h"
#include "compute_context.h"
#include "nn_evaluator.h"

#include <atomic>
#include <algorithm>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

using namespace minigo;

static std::vector<int> parse_device_ids(const std::string& str) {
    std::vector<int> ids;
    std::istringstream ss(str);
    std::string token;
    while (std::getline(ss, token, ','))
        ids.push_back(std::stoi(token));
    return ids;
}

struct GameRecord {
    int board_rows = BOARD_ROWS;
    int board_cols = BOARD_COLS;
    bool model1_is_black = false;
    std::vector<int> moves;
    int result = 0;
};

static GameRecord play_one_game(BatchEvaluator* eval1, BatchEvaluator* eval2,
                                const Config& config, bool eval1_is_black) {
    XiangqiGame game(config.history_length);
    MCTS mcts1(eval1, config);
    MCTS mcts2(eval2, config);

    GameRecord rec;
    rec.board_rows = config.board_rows;
    rec.board_cols = config.board_cols;
    rec.model1_is_black = eval1_is_black;

    while (!game.game_over && game.move_count < config.max_moves_per_game) {
        bool current_is_black = (game.current_player == BLACK);
        MCTS& mcts = (current_is_black == eval1_is_black) ? mcts1 : mcts2;
        std::vector<float> policy;
        int action = mcts.get_action(game, policy, 0.0f, -1, false, true);
        rec.moves.push_back(action);
        game.play(action);
        mcts1.make_move(action);
        mcts2.make_move(action);
    }

    if (!game.game_over) game.force_draw();

    if (game.winner == EMPTY) {
        rec.result = 0;
    } else {
        Stone eval1_color = eval1_is_black ? BLACK : RED;
        rec.result = (game.winner == eval1_color) ? 1 : -1;
    }
    return rec;
}

static void write_game_record(const std::string& path, const GameRecord& rec,
                              int game_id, const std::string& m1_name,
                              const std::string& m2_name) {
    std::ofstream out(path);
    XiangqiGame formatter;
    out << "game " << game_id << "\n";
    out << "board " << rec.board_rows << "x" << rec.board_cols << "\n";
    out << "black " << (rec.model1_is_black ? m1_name : m2_name) << "\n";
    out << "red " << (rec.model1_is_black ? m2_name : m1_name) << "\n";
    out << "result " << rec.result << "\n";
    out << "moves";
    for (int action : rec.moves) out << ' ' << formatter.action_to_str(action);
    out << "\n";
}

int main(int argc, char* argv[]) {
    std::string model1_path, model2_path;
    std::string output_dir;
    int num_games = 100;
    int num_threads = 1;
    int search_threads = 16;
    int nn_server_threads = 1;
    std::string nn_device_ids_str = "0";
    int sims = -1;
    float threshold = 0.55f;
    int max_batch_size = 256;
    float c_puct = -1.0f;
    float win_loss_weight = -1.0f;
    float score_weight = -1.0f;
    float score_scale = -1.0f;

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if      (arg == "--model1"            && i + 1 < argc) model1_path = argv[++i];
        else if (arg == "--model2"            && i + 1 < argc) model2_path = argv[++i];
        else if (arg == "--games"             && i + 1 < argc) num_games = std::stoi(argv[++i]);
        else if (arg == "--threads"           && i + 1 < argc) num_threads = std::stoi(argv[++i]);
        else if (arg == "--search-threads"    && i + 1 < argc) search_threads = std::stoi(argv[++i]);
        else if (arg == "--sims"              && i + 1 < argc) sims = std::stoi(argv[++i]);
        else if (arg == "--nn-server-threads" && i + 1 < argc) nn_server_threads = std::stoi(argv[++i]);
        else if (arg == "--nn-device-ids"     && i + 1 < argc) nn_device_ids_str = argv[++i];
        else if (arg == "--threshold"         && i + 1 < argc) threshold = std::stof(argv[++i]);
        else if (arg == "--max-batch"         && i + 1 < argc) max_batch_size = std::stoi(argv[++i]);
        else if (arg == "--output"            && i + 1 < argc) output_dir = argv[++i];
        else if (arg == "--c-puct"            && i + 1 < argc) c_puct = std::stof(argv[++i]);
        else if (arg == "--win-loss-weight"   && i + 1 < argc) win_loss_weight = std::stof(argv[++i]);
        else if (arg == "--score-weight"      && i + 1 < argc) score_weight = std::stof(argv[++i]);
        else if (arg == "--score-scale"       && i + 1 < argc) score_scale = std::stof(argv[++i]);
        else if (arg == "--help" || arg == "-h") {
            std::cout
                << "Usage: evaluate [options]\n"
                << "  --model1 PATH           Candidate model\n"
                << "  --model2 PATH           Baseline model\n"
                << "  --games N               Games to play (default: 100)\n"
                << "  --threads N             Parallel workers (default: 1)\n"
                << "  --search-threads N      MCTS threads per move (default: 16)\n"
                << "  --sims N                MCTS simulations per move (default: 800)\n"
                << "  --max-batch N           Max GPU batch size (default: 256)\n"
                << "  --threshold FLOAT       Score threshold — (wins + 0.5*draws)/N (default: 0.55)\n"
                << "  --c-puct F              UCB exploration constant (default: 1.5)\n"
                << "  --win-loss-weight F     Win/loss utility weight (default: 1.0)\n"
                << "  --score-weight F        Score utility weight (default: 0.0)\n"
                << "  --score-scale F         Score utility scale (default: 1000.0)\n"
                << "  --output DIR            Save game records as text files\n"
                << "  --nn-server-threads N   NN server threads per model (default: 1)\n"
                << "  --nn-device-ids IDS     Comma-separated GPU indices (default: \"0\")\n";
            return 0;
        } else {
            std::cerr << "Error: unrecognized option '" << arg << "'\n";
            return 1;
        }
    }

    if (model1_path.empty() || model2_path.empty()) {
        std::cerr << "Error: both --model1 and --model2 are required\n";
        return 2;
    }

    auto device_ids = parse_device_ids(nn_device_ids_str);
    if ((int)device_ids.size() != nn_server_threads) {
        std::cerr << "ERROR: --nn-device-ids has " << device_ids.size()
                  << " entries but --nn-server-threads is " << nn_server_threads
                  << ". Must match exactly.\n";
        return 2;
    }

    auto model1 = LoadedModel::load(model1_path);
    auto model2 = LoadedModel::load(model2_path);

    if (model1->board_rows != model2->board_rows ||
        model1->board_cols != model2->board_cols ||
        model1->action_size != model2->action_size) {
        std::cerr << "ERROR: models have different board/action layouts\n";
        return 2;
    }

    Config config;
    config.model_type = model1->model_type;
    config.board_rows = model1->board_rows;
    config.board_cols = model1->board_cols;
    config.history_length = std::max(1, (model1->input_channels - 1) / 14);
    config.input_channels = model1->input_channels;
    config.num_filters = model1->num_filters;
    config.num_res_blocks = model1->num_res_blocks;
    config.num_search_threads = search_threads;
    config.max_batch_size = max_batch_size;
    if (sims > 0) config.num_simulations = sims;
    if (c_puct > 0) config.c_puct = c_puct;
    if (win_loss_weight >= 0) config.win_loss_weight = win_loss_weight;
    if (score_weight >= 0) config.score_weight = score_weight;
    if (score_scale >= 0) config.score_scale = score_scale;

    auto ctx1 = std::shared_ptr<ComputeContext>(create_compute_context(device_ids));
    auto ctx2 = std::shared_ptr<ComputeContext>(create_compute_context(device_ids));

    auto eval1 = std::make_shared<NNEvaluator>(model1, ctx1, device_ids, max_batch_size);
    eval1->wait_ready();
    auto eval2 = std::make_shared<NNEvaluator>(model2, ctx2, device_ids, max_batch_size);
    eval2->wait_ready();

    std::cout << "Xiangqi Evaluation Match\n"
              << "  Model 1: " << model1_path << "\n"
              << "  Model 2: " << model2_path << "\n"
              << "  Board:   " << config.board_rows << "x" << config.board_cols << "\n"
              << "  Sims:    " << config.num_simulations << "\n"
              << "  Threads: " << num_threads << "\n"
              << "  Backend: " << ctx1->backend_name() << "\n\n";

    if (!output_dir.empty())
        system(("mkdir -p " + output_dir).c_str());

    std::mutex print_mutex;
    std::atomic<int> games_done{0};
    std::atomic<int> m1_wins{0}, m2_wins{0}, draws{0};
    auto start = std::chrono::steady_clock::now();

    auto worker = [&]() {
        while (true) {
            int gid = games_done.fetch_add(1);
            if (gid >= num_games) break;

            bool m1_black = (gid % 2 == 0);
            auto t0 = std::chrono::steady_clock::now();
            auto rec = play_one_game(eval1.get(), eval2.get(), config, m1_black);
            auto t1 = std::chrono::steady_clock::now();
            double secs = std::chrono::duration<double>(t1 - t0).count();

            if      (rec.result > 0) m1_wins.fetch_add(1);
            else if (rec.result < 0) m2_wins.fetch_add(1);
            else                     draws.fetch_add(1);

            if (!output_dir.empty()) {
                std::string path = output_dir + "/game_" + std::to_string(gid) + ".txt";
                write_game_record(path, rec, gid, model1_path, model2_path);
            }

            std::lock_guard<std::mutex> lock(print_mutex);
            const char* winner = rec.result > 0 ? "M1" : rec.result < 0 ? "M2" : "Draw";
            std::cout << "  Game " << (gid + 1) << "/" << num_games
                      << "  M1=" << (m1_black ? "B" : "R")
                      << "  " << winner
                      << "  " << std::fixed << std::setprecision(1)
                      << secs << "s"
                      << "  [" << m1_wins.load() << "-" << m2_wins.load()
                      << "-" << draws.load() << "]\n";
        }
    };

    std::vector<std::thread> threads;
    threads.reserve(num_threads);
    for (int t = 0; t < num_threads; t++)
        threads.emplace_back(worker);
    for (auto& t : threads)
        t.join();

    double total = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - start).count();
    int w1 = m1_wins.load(), w2 = m2_wins.load(), d = draws.load();
    // Chess-style scoring: win=1, draw=0.5, loss=0.  Chinese chess has a
    // lot of draws, so counting them as 0 against Model 1 understates
    // progress and keeps candidates stuck below the promotion threshold
    // even when they're clearly stronger.
    float score = (static_cast<float>(w1) + 0.5f * static_cast<float>(d))
                  / static_cast<float>(num_games);
    float win_only = static_cast<float>(w1) / static_cast<float>(num_games);

    std::cout << "\n============================================\n"
              << "  Model 1 wins: " << w1 << "\n"
              << "  Model 2 wins: " << w2 << "\n"
              << "  Draws:        " << d << "\n"
              << "  Model 1 score:    " << std::fixed << std::setprecision(1)
              << (score * 100.0f) << "%   (W + 0.5*D / N)\n"
              << "  Model 1 win rate: " << std::setprecision(1)
              << (win_only * 100.0f) << "%   (W / N, no draw credit)\n"
              << "  Time: " << std::setprecision(1) << total << "s"
              << " (" << std::setprecision(2) << (total / num_games) << "s/game)\n"
              << "  RESULT: " << (score >= threshold ? "PASS" : "FAIL") << "\n"
              << "============================================\n";

    return (score >= threshold) ? 0 : 1;
}
