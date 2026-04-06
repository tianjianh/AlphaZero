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
        out.write(reinterpret_cast<const char*>(&rec.score), sizeof(float));
    }
}

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
    std::string output_dir  = "training/selfplay";
    int  num_games          = 100;
    int  num_threads        = 1;
    int  search_threads     = 16;
    int  nn_server_threads  = 1;
    std::string nn_device_ids_str = "0";

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if      (arg == "--model"             && i+1<argc) model_path         = argv[++i];
        else if (arg == "--games"             && i+1<argc) num_games          = std::stoi(argv[++i]);
        else if (arg == "--threads"           && i+1<argc) num_threads        = std::stoi(argv[++i]);
        else if (arg == "--search-threads"    && i+1<argc) search_threads     = std::stoi(argv[++i]);
        else if (arg == "--max-batch"         && i+1<argc) config.max_batch_size = std::stoi(argv[++i]);
        else if (arg == "--output"            && i+1<argc) output_dir         = argv[++i];
        else if (arg == "--sims"              && i+1<argc) config.num_simulations = std::stoi(argv[++i]);
        else if (arg == "--c-puct"            && i+1<argc) config.c_puct = std::stof(argv[++i]);
        else if (arg == "--dirichlet-alpha"   && i+1<argc) config.dirichlet_alpha = std::stof(argv[++i]);
        else if (arg == "--dirichlet-epsilon" && i+1<argc) config.dirichlet_epsilon = std::stof(argv[++i]);
        else if (arg == "--temp-threshold"    && i+1<argc) config.temperature_threshold = std::stoi(argv[++i]);
        else if (arg == "--komi"             && i+1<argc) config.komi = std::stof(argv[++i]);
        else if (arg == "--score-weight"     && i+1<argc) config.score_weight = std::stof(argv[++i]);
        else if (arg == "--score-scale"      && i+1<argc) config.score_scale = std::stof(argv[++i]);
        else if (arg == "--nn-server-threads" && i+1<argc) nn_server_threads  = std::stoi(argv[++i]);
        else if (arg == "--nn-device-ids"     && i+1<argc) nn_device_ids_str  = argv[++i];
        else if (arg == "--help") {
            std::cout
                << "Usage: selfplay [options]\n"
                << "  --model PATH            Model file (default: models/best.onnx)\n"
                << "  --games N               Number of games (default: 100)\n"
                << "  --threads N             Parallel self-play workers (default: 1)\n"
                << "  --search-threads N      MCTS search threads per move (default: 16)\n"
                << "  --max-batch N           Max GPU batch size (default: 256)\n"
                << "  --output DIR            Output directory (default: training/selfplay)\n"
                << "  --sims N                MCTS simulations per move (default: 800)\n"
                << "  --c-puct F              UCB exploration constant (default: 1.5)\n"
                << "  --dirichlet-alpha F     Root noise concentration (default: 0.15 for 9x9)\n"
                << "  --dirichlet-epsilon F   Root noise weight (default: 0.25)\n"
                << "  --temp-threshold N      Moves of stochastic play (default: 15)\n"
                << "  --komi F                Komi value (default: 6.5)\n"
                << "  --score-weight F        Score utility weight (default: 0.0)\n"
                << "  --score-scale F         Score atan compression scale (default: 10.0)\n"
                << "  --nn-server-threads N   NN server threads (default: 1)\n"
                << "  --nn-device-ids IDS     Comma-separated device indices (default: \"0\")\n";
            return 0;
        }
    }

    system(("mkdir -p " + output_dir).c_str());

    // Parse device IDs
    std::vector<int> device_ids = parse_device_ids(nn_device_ids_str);
    if ((int)device_ids.size() != nn_server_threads) {
        std::cerr << "ERROR: --nn-device-ids has " << device_ids.size()
                  << " entries but --nn-server-threads is " << nn_server_threads
                  << ". Must match exactly.\n";
        return 1;
    }

    // Load model once (shared CPU weights — KataGo pattern)
    auto model = LoadedModel::load(model_path);

    config.model_type         = model->model_type;
    config.board_size         = model->board_size;
    config.input_channels     = model->input_channels;
    config.num_filters        = model->num_filters;
    config.num_res_blocks     = model->num_res_blocks;
    config.vit_depth          = model->vit_depth;
    config.vit_heads          = model->vit_heads;
    config.vit_kv_groups      = model->vit_kv_groups;
    config.max_moves_per_game = config.board_size * config.board_size * 2;
    config.num_search_threads = search_threads;

    // Create compute context (device init — shared across server threads)
    auto context = std::shared_ptr<ComputeContext>(create_compute_context(device_ids));

    // Create NNEvaluator with N server threads
    auto nn_evaluator = std::make_shared<NNEvaluator>(
        model, context, device_ids, config.max_batch_size);

    std::cout << "MiniGo C++ Self-Play\n"
              << "  Board:            " << config.board_size << "x" << config.board_size << "\n"
              << "  Arch:             " << config.model_type << "\n"
              << "  Komi:             " << config.komi << "\n";
    if (config.model_type == "vit")
        std::cout << "  d_model:          " << config.num_filters
                  << "  depth=" << config.vit_depth
                  << "  heads=" << config.vit_heads
                  << "  kv=" << config.vit_kv_groups << "\n";
    else
        std::cout << "  Filters:          " << config.num_filters
                  << "  Blocks: " << config.num_res_blocks << "\n";
    std::cout
              << "  Simulations:      " << config.num_simulations << "\n"
              << "  Search threads:   " << config.num_search_threads << "\n"
              << "  c_puct:           " << config.c_puct << "\n"
              << "  Score weight:     " << config.score_weight << "\n"
              << "  Score scale:      " << config.score_scale << "\n"
              << "  Dirichlet:        alpha=" << config.dirichlet_alpha
              << "  eps=" << config.dirichlet_epsilon << "\n"
              << "  Temp threshold:   " << config.temperature_threshold << "\n"
              << "  Games:            " << num_games << "\n"
              << "  Threads:          " << num_threads << "\n"
              << "  Backend:          " << context->backend_name() << "\n"
              << "  NN servers:       " << nn_server_threads << "\n"
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
