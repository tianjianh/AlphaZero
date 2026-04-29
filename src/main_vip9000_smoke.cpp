// Standalone smoke test for the VIP9000 backend.  Loads either the bs=1
// or bs=4 NBG, encodes `N` real Go positions, calls predict_batch a few
// hundred times, and reports timings.  Bypasses MCTS / NNEvaluator's
// batch-of-many-batches path so we can drive batch sizes that match the
// compiled NBG exactly (N = K), avoiding queue-capacity edge cases.

#include "compute_context.h"
#include "loaded_model.h"
#include "game.h"
#include "katago_inputs.h"

#include <chrono>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

using namespace minigo;

int main(int argc, char** argv) {
    std::string model = "models/kata1-b10c128.a733.bs1.unshared.onnx";
    int batch        = 1;
    int iters        = 100;
    int n_threads    = 1;
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        if      (a == "--model"   && i + 1 < argc) model     = argv[++i];
        else if (a == "--batch"   && i + 1 < argc) batch     = std::stoi(argv[++i]);
        else if (a == "--iters"   && i + 1 < argc) iters     = std::stoi(argv[++i]);
        else if (a == "--threads" && i + 1 < argc) n_threads = std::stoi(argv[++i]);
    }

    auto loaded = LoadedModel::load(model);
    std::vector<int> dev_ids(n_threads, 0);
    auto ctx = std::shared_ptr<ComputeContext>(create_compute_context(dev_ids));

    GoGame game(loaded->board_size, 6.5f);
    std::vector<float> state;
    encode_for_katago(game, loaded.get(), state);
    std::vector<std::vector<float>> states(batch, state);

    auto run_one = [&](int tid) {
        auto h = ctx->create_handle(loaded.get(), 0, batch);
        for (int i = 0; i < 5; i++) h->predict_batch(states);   // warmup
        if (tid == 0) {
            auto first = h->predict_batch(states);
            std::ostringstream os;
            os << "first predict: N=" << first.size()
               << " policy[0..2]=" << first[0].policy[0]
               << "," << first[0].policy[1]
               << "," << first[0].policy[2]
               << " value=" << first[0].value
               << " score=" << first[0].score
               << " score_sd=" << first[0].score_sd
               << " own[0..1]=" << first[0].ownership[0]
               << "," << first[0].ownership[1] << "\n";
            std::cout << os.str();
        }
        auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < iters; i++) h->predict_batch(states);
        double secs = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - t0).count();
        std::ostringstream os;
        os << "  thread " << tid
           << " batch=" << batch << " iters=" << iters
           << " elapsed=" << std::fixed << std::setprecision(3) << secs << "s"
           << " ms/call=" << std::setprecision(2) << (secs / iters * 1e3)
           << " states/s=" << (int)((double)batch * iters / secs)
           << "\n";
        std::cout << os.str();
    };

    auto wall_t0 = std::chrono::steady_clock::now();
    if (n_threads == 1) {
        run_one(0);
    } else {
        std::vector<std::thread> ts;
        for (int t = 0; t < n_threads; t++) ts.emplace_back(run_one, t);
        for (auto& t : ts) t.join();
    }
    double wall = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - wall_t0).count();
    std::cout << "aggregate: threads=" << n_threads
              << " batch=" << batch << " iters/thread=" << iters
              << " wall=" << std::fixed << std::setprecision(3) << wall << "s"
              << " total_states=" << (n_threads * batch * iters)
              << " agg_states/s="
              << (int)((double)n_threads * batch * iters / wall) << "\n";
    return 0;
}
