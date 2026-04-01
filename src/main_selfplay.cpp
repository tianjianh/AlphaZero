#include "config.h"
#include "game.h"
#include "mcts.h"
#include "inference_engine.h"
#include "nn_evaluator.h"
#include <atomic>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <thread>
#include <mutex>

using namespace minigo;

static void write_records(const std::string& path,
                           const std::vector<TrainingRecord>& records) {
    std::ofstream out(path, std::ios::binary);
    int32_t n = (int32_t)records.size();
    out.write(reinterpret_cast<const char*>(&n), 4);

    for (auto& rec : records) {
        int32_t ss = (int32_t)rec.state.size();
        int32_t ps = (int32_t)rec.policy.size();
        out.write(reinterpret_cast<const char*>(&ss), 4);
        out.write(reinterpret_cast<const char*>(rec.state.data()), ss * sizeof(float));
        out.write(reinterpret_cast<const char*>(&ps), 4);
        out.write(reinterpret_cast<const char*>(rec.policy.data()), ps * sizeof(float));
        out.write(reinterpret_cast<const char*>(&rec.value), sizeof(float));
    }
}

int main(int argc, char* argv[]) {
    Config config;
    std::string model_path  = "model.onnx";
    std::string output_dir  = "selfplay_data";
    int  num_games          = 100;
    int  num_threads        = 1;
    int  search_threads     = 16;

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if      (arg == "--model"          && i+1<argc) model_path                   = argv[++i];
        else if (arg == "--games"          && i+1<argc) num_games                    = std::stoi(argv[++i]);
        else if (arg == "--threads"        && i+1<argc) num_threads                  = std::stoi(argv[++i]);
        else if (arg == "--search-threads" && i+1<argc) search_threads               = std::stoi(argv[++i]);
        else if (arg == "--max-batch"      && i+1<argc) config.max_batch_size        = std::stoi(argv[++i]);
        else if (arg == "--output"         && i+1<argc) output_dir                   = argv[++i];
        else if (arg == "--sims"           && i+1<argc) config.num_simulations       = std::stoi(argv[++i]);
        else if (arg == "--help") {
            std::cout
                << "Usage: selfplay [options]\n"
                << "  --model PATH          Model file (default: model.onnx)\n"
                << "  --games N             Number of games (default: 100)\n"
                << "  --threads N           Parallel self-play workers (default: 1)\n"
                << "  --search-threads N    MCTS search threads per move (default: 16)\n"
                << "  --max-batch N         Max GPU batch size (default: 256)\n"
                << "  --output DIR          Output directory (default: selfplay_data)\n"
                << "  --sims N              MCTS simulations per move (default: 800)\n";
            return 0;
        }
    }

    system(("mkdir -p " + output_dir).c_str());

    // Create engine (backend selected at compile time)
    auto engine = std::shared_ptr<InferenceEngine>(create_engine(model_path));

    config.board_size       = engine->board_size;
    config.input_channels   = engine->input_channels;
    config.num_filters      = engine->num_filters;
    config.num_res_blocks   = engine->num_res_blocks;
    config.max_moves_per_game = config.board_size * config.board_size * 2;
    config.num_search_threads = search_threads;

    // All backends use NNEvaluator (unified architecture)
    auto nn_evaluator = std::make_shared<NNEvaluator>(
        engine.get(), config.max_batch_size);

    std::cout << "MiniGo C++ Self-Play\n"
              << "  Board:            " << config.board_size << "x" << config.board_size << "\n"
              << "  Filters:          " << config.num_filters
              << "  Blocks: "           << config.num_res_blocks << "\n"
              << "  Simulations:      " << config.num_simulations << "\n"
              << "  Search threads:   " << config.num_search_threads << "\n"
              << "  Games:            " << num_games << "\n"
              << "  Threads:          " << num_threads << "\n"
              << "  Backend:          " << engine->backend_name() << "\n"
              << "  Model:            " << model_path << "\n"
              << "  Output:           " << output_dir << "\n\n";

    std::mutex     print_mutex;
    std::atomic<int> games_done{0};
    auto start = std::chrono::steady_clock::now();

    auto worker = [&]() {
        while (true) {
            int game_id = games_done.fetch_add(1);
            if (game_id >= num_games) break;

            auto t0 = std::chrono::steady_clock::now();
            auto records = self_play_game(nn_evaluator.get(), config);
            auto t1 = std::chrono::steady_clock::now();

            double secs = std::chrono::duration<double>(t1 - t0).count();
            int moves   = (int)records.size() / 8;

            std::string filename = output_dir + "/game_" +
                                   std::to_string(game_id) + ".bin";
            write_records(filename, records);

            {
                std::lock_guard<std::mutex> lock(print_mutex);
                std::cout << "  Game " << (game_id + 1) << "/" << num_games
                          << " — " << moves << " moves, "
                          << records.size() << " samples, "
                          << std::fixed << std::setprecision(2)
                          << secs << "s\n";
            }
        }
    };

    std::vector<std::thread> threads;
    threads.reserve(num_threads);
    for (int t = 0; t < num_threads; t++)
        threads.emplace_back(worker);
    for (auto& t : threads)
        t.join();

    double total_secs = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - start).count();
    std::cout << "\nDone! " << num_games << " games in " << total_secs
              << "s (" << (total_secs / num_games) << "s/game)\n";

    return 0;
}
