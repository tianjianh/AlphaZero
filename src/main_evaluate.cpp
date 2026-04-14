#include "config.h"
#include "game.h"
#include "mcts.h"
#include "loaded_model.h"
#include "compute_context.h"
#include "nn_evaluator.h"
#include <atomic>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <thread>
#include <mutex>

using namespace minigo;

static std::vector<int> parse_device_ids(const std::string& str) {
    std::vector<int> ids;
    std::istringstream ss(str);
    std::string token;
    while (std::getline(ss, token, ','))
        ids.push_back(std::stoi(token));
    return ids;
}

// Game record: sequence of moves + result
struct GameRecord {
    int board_size;
    float komi;
    bool model1_is_black;
    std::vector<int> moves;   // action indices (-1 = pass)
    int result;               // +1 = model1 wins, -1 = model2 wins, 0 = draw
    float black_score;        // black - white (komi included)
};

// Play one game between two evaluators.
// Returns: +1 if eval1 wins, -1 if eval2 wins, 0 if draw.
static GameRecord play_one_game(BatchEvaluator* eval1, BatchEvaluator* eval2,
                                const Config& config, bool eval1_is_black) {
    GoGame game(config.board_size, config.komi);
    MCTS mcts1(eval1, config);
    MCTS mcts2(eval2, config);
    int action_size = config.action_size();

    GameRecord rec;
    rec.board_size = config.board_size;
    rec.komi = config.komi;
    rec.model1_is_black = eval1_is_black;

    while (!game.game_over && game.move_count < config.max_moves_per_game) {
        bool current_is_black = (game.current_player == BLACK);
        MCTS& mcts = (current_is_black == eval1_is_black) ? mcts1 : mcts2;

        // Evaluation: temperature 0, no Dirichlet noise.
        // reuse_tree=true — the active MCTS's tree was advanced via
        // make_move() in the previous iteration.
        std::vector<float> policy;
        int action = mcts.get_action(game, policy, 0.0f, -1, false,
                                      /*reuse_tree=*/true);

        rec.moves.push_back(action == action_size - 1 ? PASS_MOVE : action);

        if (action == action_size - 1)
            game.play(PASS_MOVE);
        else
            game.play(action);

        // Advance BOTH MCTS trees — each player's tree needs to track
        // the opponent's move too so its next search can reuse the subtree.
        mcts1.make_move(action);
        mcts2.make_move(action);
    }

    while (!game.game_over) game.play(PASS_MOVE);

    rec.black_score = game.final_black_score;

    if (game.winner == EMPTY) rec.result = 0;
    else {
        Stone eval1_color = eval1_is_black ? BLACK : WHITE;
        rec.result = (game.winner == eval1_color) ? 1 : -1;
    }
    return rec;
}

// Write a game record as SGF
static void write_sgf(const std::string& path, const GameRecord& rec,
                      int game_id, const std::string& m1_name,
                      const std::string& m2_name) {
    std::ofstream out(path);
    int n = rec.board_size;
    std::string black_name = rec.model1_is_black ? m1_name : m2_name;
    std::string white_name = rec.model1_is_black ? m2_name : m1_name;
    std::string result_str;
    if (rec.result == 0) {
        result_str = "0";
    } else {
        bool m1_won = rec.result > 0;
        bool black_won = (m1_won == rec.model1_is_black);
        float margin = std::abs(rec.black_score);
        char buf[32];
        snprintf(buf, sizeof(buf), "%s+%.1f", black_won ? "B" : "W", margin);
        result_str = buf;
    }

    out << "(;GM[1]FF[4]SZ[" << n << "]KM[" << rec.komi << "]"
        << "PB[" << black_name << "]PW[" << white_name << "]"
        << "RE[" << result_str << "]"
        << "GN[eval_game_" << game_id << "]\n";

    for (int i = 0; i < (int)rec.moves.size(); i++) {
        const char* color = (i % 2 == 0) ? "B" : "W";
        if (rec.moves[i] == PASS_MOVE) {
            out << ";" << color << "[]";
        } else {
            int r = rec.moves[i] / n;
            int c = rec.moves[i] % n;
            char col = 'a' + c;
            char row = 'a' + r;
            out << ";" << color << "[" << col << row << "]";
        }
    }
    out << ")\n";
}

int main(int argc, char* argv[]) {
    std::string model1_path, model2_path;
    std::string output_dir;  // empty = don't save games
    int num_games          = 100;
    int num_threads        = 1;
    int search_threads     = 16;
    int nn_server_threads  = 1;
    std::string nn_device_ids_str = "0";
    int sims               = -1;
    float threshold        = 0.55f;
    int max_batch_size     = 256;
    float c_puct           = -1.0f;  // -1 = use default
    float komi             = -1.0f;  // -1 = use default
    float win_loss_weight  = -1.0f;  // -1 = use default
    float score_weight     = -1.0f;  // -1 = use default
    float score_scale      = -1.0f;  // -1 = use default

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if      (arg == "--model1"            && i+1<argc) model1_path       = argv[++i];
        else if (arg == "--model2"            && i+1<argc) model2_path       = argv[++i];
        else if (arg == "--games"             && i+1<argc) num_games         = std::stoi(argv[++i]);
        else if (arg == "--threads"           && i+1<argc) num_threads       = std::stoi(argv[++i]);
        else if (arg == "--search-threads"    && i+1<argc) search_threads    = std::stoi(argv[++i]);
        else if (arg == "--sims"              && i+1<argc) sims              = std::stoi(argv[++i]);
        else if (arg == "--nn-server-threads" && i+1<argc) nn_server_threads = std::stoi(argv[++i]);
        else if (arg == "--nn-device-ids"     && i+1<argc) nn_device_ids_str = argv[++i];
        else if (arg == "--threshold"         && i+1<argc) threshold         = std::stof(argv[++i]);
        else if (arg == "--max-batch"         && i+1<argc) max_batch_size    = std::stoi(argv[++i]);
        else if (arg == "--output"            && i+1<argc) output_dir        = argv[++i];
        else if (arg == "--c-puct"            && i+1<argc) c_puct            = std::stof(argv[++i]);
        else if (arg == "--komi"              && i+1<argc) komi              = std::stof(argv[++i]);
        else if (arg == "--win-loss-weight"   && i+1<argc) win_loss_weight   = std::stof(argv[++i]);
        else if (arg == "--score-weight"      && i+1<argc) score_weight      = std::stof(argv[++i]);
        else if (arg == "--score-scale"       && i+1<argc) score_scale        = std::stof(argv[++i]);
        else if (arg == "--help" || arg == "-h") {
            std::cout
                << "Usage: evaluate [options]\n"
                << "\n"
                << "Play games between two models to determine which is stronger.\n"
                << "Exit code 0 if model1 win rate >= threshold (model1 wins).\n"
                << "Exit code 1 if model1 win rate < threshold (model2 wins).\n"
                << "\n"
                << "Options:\n"
                << "  --model1 PATH           Candidate model (required)\n"
                << "  --model2 PATH           Baseline model (required)\n"
                << "  --games N               Games to play (default: 100)\n"
                << "  --threads N             Parallel game workers (default: 1)\n"
                << "  --search-threads N      MCTS threads per move (default: 16)\n"
                << "  --sims N                MCTS simulations per move (default: 800)\n"
                << "  --max-batch N           Max GPU batch size (default: 256)\n"
                << "  --threshold FLOAT       Win rate to pass (default: 0.55)\n"
                << "  --c-puct F              UCB exploration constant (default: 1.5)\n"
                << "  --komi F                Komi value (default: 6.5)\n"
                << "  --win-loss-weight F     Win/loss utility weight (default: 1.0)\n"
                << "  --score-weight F        Score utility weight (default: 0.0)\n"
                << "  --score-scale F         Score atan compression scale (default: 10.0)\n"
                << "  --output DIR            Save game records as SGF files\n"
                << "  --nn-server-threads N   NN server threads per model (default: 1)\n"
                << "  --nn-device-ids IDS     Comma-separated GPU indices (default: \"0\")\n";
            return 0;
        }
        else {
            std::cerr << "Error: unrecognized option '" << arg << "'\n"
                      << "Try 'evaluate --help' for usage.\n";
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

    // Load both models
    auto model1 = LoadedModel::load(model1_path);
    auto model2 = LoadedModel::load(model2_path);

    if (model1->board_size != model2->board_size) {
        std::cerr << "ERROR: models have different board sizes ("
                  << model1->board_size << " vs " << model2->board_size << ")\n";
        return 2;
    }

    // Config — MCTS params are architecture-independent
    Config config;
    config.model_type         = model1->model_type;
    config.board_size         = model1->board_size;
    config.input_channels     = model1->input_channels;
    config.num_filters        = model1->num_filters;
    config.num_res_blocks     = model1->num_res_blocks;
    config.vit_depth          = model1->vit_depth;
    config.vit_heads          = model1->vit_heads;
    config.vit_kv_groups      = model1->vit_kv_groups;
    config.max_moves_per_game = config.board_size * config.board_size * 2;
    config.num_search_threads = search_threads;
    config.max_batch_size     = max_batch_size;
    if (sims > 0) config.num_simulations = sims;
    if (c_puct > 0) config.c_puct = c_puct;
    if (komi >= 0) config.komi = komi;
    if (win_loss_weight >= 0) config.win_loss_weight = win_loss_weight;
    if (score_weight >= 0) config.score_weight = score_weight;
    if (score_scale >= 0) config.score_scale = score_scale;

    // Separate compute contexts for each model
    auto ctx1 = std::shared_ptr<ComputeContext>(
        create_compute_context(device_ids));
    auto ctx2 = std::shared_ptr<ComputeContext>(
        create_compute_context(device_ids));

    // Serialize handle creation — TensorRT engine deserialization on the
    // same GPU from two different runtimes can race at the CUDA driver
    // level.  Wait for eval1's server threads to finish creating their
    // handles (engine loaded, buffers allocated) before starting eval2.
    auto eval1 = std::make_shared<NNEvaluator>(
        model1, ctx1, device_ids, max_batch_size);
    eval1->wait_ready();
    auto eval2 = std::make_shared<NNEvaluator>(
        model2, ctx2, device_ids, max_batch_size);
    eval2->wait_ready();

    auto model_desc = [](const LoadedModel* m) -> std::string {
        if (m->model_type == "vit")
            return "d" + std::to_string(m->num_filters) + "/L" +
                   std::to_string(m->vit_depth) + "/h" +
                   std::to_string(m->vit_heads) + " vit";
        return std::to_string(m->num_filters) + "f" +
               std::to_string(m->num_res_blocks) + "b";
    };
    std::cout << "MiniGo Evaluation Match\n"
              << "  Model 1 (candidate): " << model1_path
              << " (" << model_desc(model1.get()) << ")\n"
              << "  Model 2 (baseline):  " << model2_path
              << " (" << model_desc(model2.get()) << ")\n"
              << "  Board:      " << config.board_size
              << "x" << config.board_size << "\n"
              << "  Komi:       " << config.komi << "\n"
              << "  Games:      " << num_games << "\n"
              << "  Sims:       " << config.num_simulations << "\n"
              << "  c_puct:     " << config.c_puct << "\n"
              << "  WinLoss wt: " << config.win_loss_weight << "\n"
              << "  Score wt:   " << config.score_weight << "\n"
              << "  Score sc:   " << config.score_scale << "\n"
              << "  Threads:    " << num_threads << "\n"
              << "  Threshold:  " << std::fixed << std::setprecision(1)
              << (threshold * 100.0f) << "%\n"
              << "  Backend:    " << ctx1->backend_name() << "\n\n";

    // Create output dir for SGF if requested
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

            bool m1_black = (gid % 2 == 0);   // alternate colors
            auto t0 = std::chrono::steady_clock::now();
            auto rec = play_one_game(
                eval1.get(), eval2.get(), config, m1_black);
            auto t1 = std::chrono::steady_clock::now();
            double secs = std::chrono::duration<double>(t1 - t0).count();

            int result = rec.result;
            if      (result > 0) m1_wins.fetch_add(1);
            else if (result < 0) m2_wins.fetch_add(1);
            else                 draws.fetch_add(1);

            // Save SGF
            if (!output_dir.empty()) {
                std::string sgf_path = output_dir + "/game_" +
                                       std::to_string(gid) + ".sgf";
                write_sgf(sgf_path, rec, gid, model1_path, model2_path);
            }

            {
                std::lock_guard<std::mutex> lock(print_mutex);
                const char* w = result > 0 ? "M1"
                              : result < 0 ? "M2" : "Draw";
                std::cout << "  Game " << (gid + 1) << "/" << num_games
                          << "  M1=" << (m1_black ? "B" : "W")
                          << "  " << w
                          << "  " << std::fixed << std::setprecision(1)
                          << secs << "s"
                          << "  [" << m1_wins.load()
                          << "-" << m2_wins.load()
                          << "-" << draws.load() << "]\n";
            }
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
    float wr = (float)w1 / (float)num_games;

    std::cout << "\n============================================\n"
              << "  Model 1 wins: " << w1 << "\n"
              << "  Model 2 wins: " << w2 << "\n"
              << "  Draws:        " << d << "\n"
              << "  Model 1 win rate: " << std::fixed << std::setprecision(1)
              << (wr * 100.0f) << "%\n"
              << "  Time: " << std::setprecision(1) << total << "s"
              << " (" << std::setprecision(2)
              << (total / num_games) << "s/game)\n"
              << "  RESULT: " << (wr >= threshold ? "PASS" : "FAIL")
              << " (" << std::setprecision(1) << (wr * 100.0f) << "% vs "
              << (threshold * 100.0f) << "% threshold)\n"
              << "============================================\n";

    return (wr >= threshold) ? 0 : 1;
}
