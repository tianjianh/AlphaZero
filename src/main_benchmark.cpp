#include "config.h"
#include "game.h"
#include "mcts.h"
#include "loaded_model.h"
#include "compute_context.h"
#include "nn_evaluator.h"
#include <chrono>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <thread>
#include <atomic>
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

int main(int argc, char* argv[]) {
    Config config;
    std::string model_path  = "models/best.onnx";
    int  num_games          = 5;
    int  nn_iters           = 1000;
    int  board_override     = -1;
    int  num_threads        = 1;
    int  search_threads     = 16;
    int  nn_server_threads  = 1;
    std::string nn_device_ids_str = "0";

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if      (arg == "--model"             && i+1<argc) model_path      = argv[++i];
        else if (arg == "--board"             && i+1<argc) board_override  = std::stoi(argv[++i]);
        else if (arg == "--sims"              && i+1<argc) config.num_simulations = std::stoi(argv[++i]);
        else if (arg == "--nn-iters"          && i+1<argc) nn_iters        = std::stoi(argv[++i]);
        else if (arg == "--games"             && i+1<argc) num_games       = std::stoi(argv[++i]);
        else if (arg == "--threads"           && i+1<argc) num_threads     = std::stoi(argv[++i]);
        else if (arg == "--search-threads"    && i+1<argc) search_threads  = std::stoi(argv[++i]);
        else if (arg == "--max-batch"         && i+1<argc) config.max_batch_size = std::stoi(argv[++i]);
        else if (arg == "--komi"              && i+1<argc) config.komi = std::stof(argv[++i]);
        else if (arg == "--score-weight"      && i+1<argc) config.score_weight = std::stof(argv[++i]);
        else if (arg == "--nn-server-threads" && i+1<argc) nn_server_threads = std::stoi(argv[++i]);
        else if (arg == "--nn-device-ids"     && i+1<argc) nn_device_ids_str = argv[++i];
        else if (arg == "--help") {
            std::cout << "Usage: benchmark [options]\n"
                      << "  --model PATH            Model file (default: models/best.onnx)\n"
                      << "  --board N               Board size override\n"
                      << "  --sims N                MCTS simulations\n"
                      << "  --nn-iters N            NN inference iterations (default: 1000)\n"
                      << "  --games N               Self-play games (default: 5)\n"
                      << "  --threads N             Self-play worker threads (default: 1)\n"
                      << "  --search-threads N      MCTS search threads per move (default: 16)\n"
                      << "  --max-batch N           Max GPU batch size (default: 256)\n"
                      << "  --komi F                Komi value (default: 6.5)\n"
                      << "  --score-weight F        Score utility weight (default: 0.0)\n"
                      << "  --nn-server-threads N   NN server threads (default: 1)\n"
                      << "  --nn-device-ids IDS     Comma-separated device indices (default: \"0\")\n";
            return 0;
        }
    }

    // Parse device IDs
    std::vector<int> device_ids = parse_device_ids(nn_device_ids_str);
    if ((int)device_ids.size() != nn_server_threads) {
        std::cerr << "ERROR: --nn-device-ids has " << device_ids.size()
                  << " entries but --nn-server-threads is " << nn_server_threads
                  << ". Must match exactly.\n";
        return 1;
    }

    // Load model
    std::shared_ptr<LoadedModel> model;
    std::shared_ptr<ComputeContext> context;
    bool has_model = false;
    try {
        model = LoadedModel::load(model_path);
        context = std::shared_ptr<ComputeContext>(create_compute_context(device_ids));
        has_model = true;
        config.board_size      = model->board_size;
        config.input_channels  = model->input_channels;
        config.num_filters     = model->num_filters;
        config.num_res_blocks  = model->num_res_blocks;
    } catch (...) {
        std::cout << "No model loaded — NN/MCTS benchmarks will be skipped.\n";
    }

    if (board_override > 0) config.board_size = board_override;
    config.max_moves_per_game = config.board_size * config.board_size * 2;

    std::cout << "MiniGo C++ Benchmark\n"
              << "  Board: " << config.board_size << "x" << config.board_size
              << "  Komi: " << config.komi
              << "  Score wt: " << config.score_weight;
    if (has_model)
        std::cout << "  Backend: " << context->backend_name();
    std::cout << "\n\n";

    // ── 1. Game engine speed ──────────────────────────────────────
    {
        std::cout << "1. Game engine (random playouts)...\n";
        int total_moves = 0;
        auto t0 = std::chrono::steady_clock::now();

        for (int g = 0; g < 10000; g++) {
            GoGame game(config.board_size, config.komi);
            while (!game.game_over && game.move_count < config.max_moves_per_game) {
                std::vector<float> legal;
                game.get_legal_moves(legal);
                for (int a = 0; a < (int)legal.size(); a++) {
                    if (legal[a] > 0.0f) {
                        if (a == config.action_size() - 1) game.play(PASS_MOVE);
                        else game.play(a);
                        total_moves++;
                        break;
                    }
                }
            }
        }

        double secs = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - t0).count();
        std::cout << "   10,000 games, " << total_moves << " moves in "
                  << std::fixed << std::setprecision(2) << secs
                  << "s (" << (int)(total_moves / secs) << " moves/s)\n\n";
    }

    if (!has_model) {
        std::cout << "2-5. NN/MCTS/Self-play benchmarks skipped (no model)\n";
        return 0;
    }

    // Create a single-server evaluator for tests 2-4 (direct inference)
    auto eval_single = std::make_shared<NNEvaluator>(
        model, context, std::vector<int>{device_ids[0]}, config.max_batch_size);

    // ── 2. Single-thread NN inference ────────────────────────────
    {
        std::cout << "2. NN inference, single-thread (" << context->backend_name() << ")...\n";

        GoGame game(config.board_size, config.komi);
        std::vector<float> state;
        game.encode(state);

        // Warmup
        for (int i = 0; i < 10; i++) eval_single->evaluate_single(state);

        auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < nn_iters; i++) eval_single->evaluate_single(state);
        double secs = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - t0).count();

        double us_per = secs / nn_iters * 1e6;
        std::cout << "   " << nn_iters << " iters in "
                  << std::fixed << std::setprecision(3) << secs
                  << "s  (" << (int)us_per << " µs/call, "
                  << (int)(nn_iters / secs) << " inf/s)\n\n";
    }

    // ── 3. Batch inference throughput ────────────────────────────
    {
        std::cout << "3. Batch NN inference throughput...\n";

        GoGame game(config.board_size, config.komi);
        std::vector<float> state;
        game.encode(state);

        for (int batch : {1, 8, 32, 64, 128}) {
            std::vector<std::vector<float>> batch_states(batch, state);

            // Warmup
            for (int i = 0; i < 5; i++) eval_single->evaluate(batch_states);

            int iters = std::max(10, 500 / batch);
            auto t0 = std::chrono::steady_clock::now();
            for (int i = 0; i < iters; i++) eval_single->evaluate(batch_states);
            double secs = std::chrono::duration<double>(
                std::chrono::steady_clock::now() - t0).count();

            double ms_per_batch = secs / iters * 1e3;
            double states_per_s = (double)batch * iters / secs;
            std::cout << "   batch=" << std::setw(4) << batch
                      << "  " << std::fixed << std::setprecision(2)
                      << ms_per_batch << " ms/batch  "
                      << (int)states_per_s << " states/s\n";
        }
        std::cout << "\n";
    }

    // ── 4. MCTS (1 search thread, via NNEvaluator) ──────────────
    {
        std::cout << "4. MCTS (" << config.num_simulations << " sims, 1 thread)...\n";

        config.num_search_threads = 1;
        MCTS mcts(eval_single.get(), config);
        GoGame game(config.board_size, config.komi);

        auto t0 = std::chrono::steady_clock::now();
        std::vector<float> pi;
        int action = mcts.get_action(game, pi, 0.0f, -1, false);
        double secs = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - t0).count();

        std::cout << "   First move: " << game.action_to_str(action)
                  << "  in " << std::fixed << std::setprecision(3) << secs
                  << "s  (" << (int)(config.num_simulations / secs) << " sims/s)\n\n";
    }

    // Destroy single-server evaluator before creating multi-server one
    eval_single.reset();

    // ── 5. Multi-threaded self-play ───────────────────────────────
    {
        std::cout << "5. Self-play (" << num_games << " games, "
                  << num_threads << " threads, "
                  << nn_server_threads << " server(s))...\n";

        config.num_search_threads = search_threads;

        auto nn_evaluator = std::make_shared<NNEvaluator>(
            model, context, device_ids, config.max_batch_size);

        std::atomic<int> games_done{0};
        std::atomic<int> total_records{0};
        std::mutex print_mutex;
        auto wall_t0 = std::chrono::steady_clock::now();

        auto worker = [&]() {
            while (true) {
                int gid = games_done.fetch_add(1);
                if (gid >= num_games) break;

                auto t0 = std::chrono::steady_clock::now();
                auto records = self_play_game(nn_evaluator.get(), config);
                double secs = std::chrono::duration<double>(
                    std::chrono::steady_clock::now() - t0).count();

                total_records += (int)records.size();
                std::lock_guard<std::mutex> lk(print_mutex);
                std::cout << "   Game " << (gid + 1)
                          << ": " << (records.size() / 8) << " moves, "
                          << std::fixed << std::setprecision(2) << secs << "s\n";
            }
        };

        std::vector<std::thread> threads;
        threads.reserve(num_threads);
        for (int t = 0; t < num_threads; t++)
            threads.emplace_back(worker);
        for (auto& t : threads)
            t.join();

        double total_secs = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - wall_t0).count();
        std::cout << "   Total: " << total_records.load() << " samples, "
                  << std::fixed << std::setprecision(2) << total_secs
                  << "s wall  (" << (total_secs / num_games) << "s/game)\n";
    }

    return 0;
}
