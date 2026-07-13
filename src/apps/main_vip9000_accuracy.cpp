// vip9000_accuracy — compare int8 vs fp16 NBG outputs on the same A733 NPU.
//
// Generates P random Go positions (mid-game by playing 0..N legal moves
// from an empty board), runs each through the int8 and fp16 NBGs, then
// reports:
//   * top-1 / top-3 / top-5 policy agreement (treating fp16 as reference,
//     since fp16 on VIP9000 is essentially lossless vs the source ONNX —
//     see docs/A733_CONVERSION.md §6.1 host-side parity tables);
//   * value / score / score_sd: mean and max absolute error;
//   * ownership: mean and max absolute error per board cell.
//
// Both contexts live concurrently so we don't pay vip_init/destroy churn
// between calls; precision is selected per-context via the
// VIP9000_FORCE_PRECISION env var that resolve_nbg_path() honours.

#include "nn/compute_context.h"
#include "model/loaded_model.h"
#include "engine/game.h"
#include "engine/katago_inputs.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <string>
#include <unistd.h>
#include <vector>

using namespace minigo;

namespace {

// Generate P pseudo-random Go positions by playing 0..max_moves legal
// moves from an empty board.  Deterministic given the seed.
static std::vector<std::vector<float>>
generate_positions(const LoadedModel& model, int P, int max_moves,
                   uint32_t seed,
                   std::vector<int>& legal_action_count_out) {
    std::vector<std::vector<float>> states;
    states.reserve(P);
    legal_action_count_out.reserve(P);

    std::mt19937 rng(seed);
    const int B = model.board_size;
    const int action_size = B * B + 1;

    for (int i = 0; i < P; ++i) {
        GoGame game(B, 6.5f);
        // Decide depth: half empty/early, the rest spread up to max_moves.
        int depth;
        if (i < P / 4) {
            depth = (int)(rng() % 5);   // 0..4 — opening
        } else {
            depth = (int)(rng() % (max_moves + 1));
        }

        for (int m = 0; m < depth && !game.game_over; ++m) {
            std::vector<float> legal;
            game.get_legal_moves(legal);
            // Pick a uniform random legal move.
            std::vector<int> legal_idx;
            legal_idx.reserve(action_size);
            for (int a = 0; a < action_size; ++a)
                if (legal[a] > 0.0f) legal_idx.push_back(a);
            if (legal_idx.empty()) break;
            int a = legal_idx[rng() % legal_idx.size()];
            if (a == action_size - 1) game.play(PASS_MOVE);
            else                      game.play(a);
        }

        std::vector<float> legal;
        game.get_legal_moves(legal);
        int n_legal = 0;
        for (int a = 0; a < action_size; ++a) if (legal[a] > 0.0f) ++n_legal;
        legal_action_count_out.push_back(n_legal);

        std::vector<float> state;
        encode_for_katago(game, &model, state);
        states.push_back(std::move(state));
    }
    return states;
}

// Top-K argmax — returns indices of the K largest entries, descending.
static std::vector<int> topk(const std::vector<float>& v, int k) {
    std::vector<int> idx(v.size());
    std::iota(idx.begin(), idx.end(), 0);
    if (k > (int)v.size()) k = (int)v.size();
    std::partial_sort(idx.begin(), idx.begin() + k, idx.end(),
                      [&](int a, int b) { return v[a] > v[b]; });
    idx.resize(k);
    return idx;
}

struct Stats {
    double sum = 0.0, sumsq = 0.0, max_abs = 0.0;
    int n = 0;
    void add(double err) {
        double a = std::fabs(err);
        sum   += a;
        sumsq += err * err;
        if (a > max_abs) max_abs = a;
        ++n;
    }
    double mean()    const { return n ? sum / n : 0.0; }
    double rmse()    const { return n ? std::sqrt(sumsq / n) : 0.0; }
};

}  // namespace

int main(int argc, char** argv) {
    std::string model_path = "models/kata1-b10c128.a733.bs1.unshared.onnx";
    int batch         = 1;
    int n_positions   = 100;
    int max_depth     = 60;
    uint32_t seed     = 42;

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if      (a == "--model"     && i + 1 < argc) model_path  = argv[++i];
        else if (a == "--batch"     && i + 1 < argc) batch       = std::atoi(argv[++i]);
        else if (a == "--positions" && i + 1 < argc) n_positions = std::atoi(argv[++i]);
        else if (a == "--max-depth" && i + 1 < argc) max_depth   = std::atoi(argv[++i]);
        else if (a == "--seed"      && i + 1 < argc) seed        = (uint32_t)std::atoi(argv[++i]);
        else if (a == "-h" || a == "--help") {
            std::cout
              << "vip9000_accuracy — compare int8 vs fp16 NBG predictions\n"
              << "  --model PATH      ONNX path; backend resolves to bs<K>.{int8,fp16}/.nb\n"
              << "  --batch K         compiled batch (1 or 4; default 1)\n"
              << "  --positions N     number of test positions (default 100)\n"
              << "  --max-depth N     max random plies before testing (default 60)\n"
              << "  --seed N          rng seed (default 42)\n";
            return 0;
        }
    }

    // Round n_positions up to a multiple of batch so we never partial-fill.
    int n_padded = ((n_positions + batch - 1) / batch) * batch;
    if (n_padded != n_positions) {
        std::cerr << "  (rounding --positions " << n_positions
                  << " up to " << n_padded
                  << " — multiple of batch=" << batch << ")\n";
        n_positions = n_padded;
    }

    // ── 1. Load the source ONNX (architecture metadata only) ─────
    auto loaded = LoadedModel::load(model_path);

    // ── 2. Generate positions ────────────────────────────────────
    std::cerr << "Generating " << n_positions
              << " random positions (seed=" << seed
              << ", up to " << max_depth << " plies)...\n";
    std::vector<int> legal_counts;
    auto positions = generate_positions(*loaded, n_positions, max_depth, seed, legal_counts);

    // ── 3. Run inference with both precisions, in turn ──────────
    auto run = [&](const char* prec) {
        std::cerr << "Loading " << prec << " NBG...\n";
        setenv("VIP9000_FORCE_PRECISION", prec, /*overwrite=*/1);
        auto ctx = std::shared_ptr<ComputeContext>(create_compute_context({0}));
        auto h   = ctx->create_handle(loaded.get(), 0, batch);

        // Warmup — one call discarded.
        std::vector<std::vector<float>> warm(batch, positions[0]);
        h->predict_batch(warm);

        std::cerr << "Running " << n_positions << " inferences ("
                  << prec << ", batch=" << batch << ")...\n";
        std::vector<NNOutput> out;
        out.reserve(n_positions);
        for (int i = 0; i < n_positions; i += batch) {
            std::vector<std::vector<float>> b(positions.begin() + i,
                                              positions.begin() + i + batch);
            auto r = h->predict_batch(b);
            for (auto& x : r) out.push_back(std::move(x));
        }
        return out;
    };

    auto fp16_out = run("fp16");
    auto int8_out = run("int8");

    if ((int)fp16_out.size() != n_positions || (int)int8_out.size() != n_positions) {
        std::cerr << "Internal error: output count mismatch\n";
        return 1;
    }

    // ── 4. Score ────────────────────────────────────────────────
    int top1 = 0, top3 = 0, top5 = 0;
    Stats value_err, score_err, sd_err, own_err_mean, policy_err_mean;
    int n_top_buckets[5] = {0,0,0,0,0};   // by-position counts in legal-move buckets

    for (int i = 0; i < n_positions; ++i) {
        const auto& fp = fp16_out[i];
        const auto& q8 = int8_out[i];

        // Policy top-K agreement (legal moves only — we mask illegal -inf
        // here by leaving the raw logits, since both backends produce raw
        // logits and the relative ordering is what argmax needs).
        auto fp_top1 = topk(fp.policy, 1)[0];
        auto q8_topk5 = topk(q8.policy, 5);
        if (q8_topk5[0] == fp_top1) ++top1;
        if (std::find(q8_topk5.begin(), q8_topk5.begin() + std::min(3, (int)q8_topk5.size()), fp_top1) != q8_topk5.begin() + std::min(3, (int)q8_topk5.size())) ++top3;
        if (std::find(q8_topk5.begin(), q8_topk5.end(), fp_top1) != q8_topk5.end()) ++top5;

        // Per-element MAE on policy logits (raw — gives a scale-aware feel).
        for (size_t k = 0; k < fp.policy.size(); ++k)
            policy_err_mean.add(q8.policy[k] - fp.policy[k]);

        value_err.add(q8.value    - fp.value);
        score_err.add(q8.score    - fp.score);
        sd_err   .add(q8.score_sd - fp.score_sd);

        for (size_t k = 0; k < fp.ownership.size(); ++k)
            own_err_mean.add(q8.ownership[k] - fp.ownership[k]);

        // Bucket by legal-move count.
        int b = legal_counts[i] / 20;   // 0..4 buckets at 9x9 (max 82 legal)
        if (b < 0) b = 0; if (b > 4) b = 4;
        ++n_top_buckets[b];
    }

    // ── 5. Report ───────────────────────────────────────────────
    auto pct = [&](int hit) {
        return 100.0 * hit / n_positions;
    };
    std::cout << "\n"
              << "================================================================\n"
              << "  VIP9000 int8 vs fp16 accuracy on " << n_positions
              << " random positions\n"
              << "================================================================\n"
              << "  Model:       " << model_path << "\n"
              << "  Batch:       " << batch << "\n"
              << "  Reference:   fp16 NBG (treated as ground truth)\n"
              << "  Compared:    int8 NBG\n"
              << "----------------------------------------------------------------\n";

    std::cout << std::fixed << std::setprecision(2);
    std::cout << "Policy argmax agreement:\n"
              << "  top-1 = " << pct(top1) << " %  (" << top1 << " / " << n_positions << ")\n"
              << "  top-3 = " << pct(top3) << " %  (" << top3 << " / " << n_positions << ")\n"
              << "  top-5 = " << pct(top5) << " %  (" << top5 << " / " << n_positions << ")\n";

    std::cout << std::fixed << std::setprecision(4);
    std::cout << "\nScalar heads (int8 - fp16):\n"
              << "                    MAE        RMSE       MaxAbs\n"
              << "  value          " << std::setw(8) << value_err.mean()
                                     << "   " << std::setw(8) << value_err.rmse()
                                     << "   " << std::setw(8) << value_err.max_abs << "\n"
              << "  score (points) " << std::setw(8) << score_err.mean()
                                     << "   " << std::setw(8) << score_err.rmse()
                                     << "   " << std::setw(8) << score_err.max_abs << "\n"
              << "  score_sd       " << std::setw(8) << sd_err.mean()
                                     << "   " << std::setw(8) << sd_err.rmse()
                                     << "   " << std::setw(8) << sd_err.max_abs << "\n";

    std::cout << "\nDense heads (per-element abs error):\n"
              << "                    MAE        RMSE       MaxAbs\n"
              << "  policy logits  " << std::setw(8) << policy_err_mean.mean()
                                     << "   " << std::setw(8) << policy_err_mean.rmse()
                                     << "   " << std::setw(8) << policy_err_mean.max_abs << "\n"
              << "  ownership      " << std::setw(8) << own_err_mean.mean()
                                     << "   " << std::setw(8) << own_err_mean.rmse()
                                     << "   " << std::setw(8) << own_err_mean.max_abs << "\n";

    std::cout << "\nLegal-moves distribution of test positions (sanity check):\n";
    const char* labels[] = {"  1-20", " 21-40", " 41-60", " 61-80", "81-100"};
    for (int b = 0; b < 5; ++b)
        std::cout << "  " << labels[b] << " legal: " << n_top_buckets[b] << "\n";

    std::cout << "================================================================\n";
    return 0;
}
