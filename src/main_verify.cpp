// ================================================================
// Backend numerical verification + micro-benchmark.
//
//   verify --model m.onnx --vectors m.vec [tolerances] [--bench]
//
// Reads reference vectors produced by scripts/make_test_vectors.py,
// runs the compiled backend's predict_batch over all states in one
// call, and reports per-field max abs differences vs PyTorch.
// ================================================================
#include "compute_context.h"
#include "loaded_model.h"

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

using namespace minigo;

struct Vectors {
    int board = 0, state_len = 0, n = 0;
    std::vector<std::vector<float>> states;
    std::vector<std::vector<float>> refs;    // [policy A | value | score | sd | own hw]
};

static Vectors load_vectors(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot open " + path);
    uint32_t magic, board, state_len, n;
    f.read((char*)&magic, 4); f.read((char*)&board, 4);
    f.read((char*)&state_len, 4); f.read((char*)&n, 4);
    if (magic != 0x4D475456u)
        throw std::runtime_error("bad magic in " + path);
    Vectors v;
    v.board = (int)board; v.state_len = (int)state_len; v.n = (int)n;
    int hw = v.board * v.board;
    int ref_len = 2 * hw + 4;
    v.states.resize(n); v.refs.resize(n);
    for (uint32_t i = 0; i < n; i++) {
        v.states[i].resize(state_len);
        f.read((char*)v.states[i].data(), (std::streamsize)state_len * 4);
    }
    for (uint32_t i = 0; i < n; i++) {
        v.refs[i].resize(ref_len);
        f.read((char*)v.refs[i].data(), (std::streamsize)ref_len * 4);
    }
    if (!f) throw std::runtime_error("truncated vector file " + path);
    return v;
}

int main(int argc, char* argv[]) {
    std::string model_path, vec_path;
    double tol_pol = 3e-3, tol_val = 2e-3, tol_score = 8e-3, tol_own = 3e-3;
    bool bench = false;
    int max_batch = 256;

    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        if      (a == "--model"   && i + 1 < argc) model_path = argv[++i];
        else if (a == "--vectors" && i + 1 < argc) vec_path = argv[++i];
        else if (a == "--tol-pol" && i + 1 < argc) tol_pol = std::stod(argv[++i]);
        else if (a == "--tol-val" && i + 1 < argc) tol_val = std::stod(argv[++i]);
        else if (a == "--tol-score" && i + 1 < argc) tol_score = std::stod(argv[++i]);
        else if (a == "--tol-own" && i + 1 < argc) tol_own = std::stod(argv[++i]);
        else if (a == "--max-batch" && i + 1 < argc) max_batch = std::stoi(argv[++i]);
        else if (a == "--bench") bench = true;
        else { std::cerr << "unknown arg " << a << "\n"; return 2; }
    }
    if (model_path.empty() || (vec_path.empty() && !bench)) {
        std::cerr << "usage: verify --model m.onnx --vectors m.vec "
                     "[--tol-pol X ...] [--bench]\n"
                     "       verify --model m.onnx --bench   (perf only)\n";
        return 2;
    }

    auto model = LoadedModel::load(model_path);
    auto ctx = create_compute_context({0});
    auto handle = ctx->create_handle(model.get(), 0, max_batch);

    Vectors vec;
    bool ok = true;
    if (!vec_path.empty()) {
        vec = load_vectors(vec_path);
        if (model->board_size != vec.board) {
            std::cerr << "board mismatch: model " << model->board_size
                      << " vs vectors " << vec.board << "\n";
            return 2;
        }
        const int hw = vec.board * vec.board, A = hw + 1;
        auto results = handle->predict_batch(vec.states);

        double d_pol = 0, d_val = 0, d_score = 0, d_sd = 0, d_own = 0;
        for (int i = 0; i < vec.n; i++) {
            const auto& r = results[i];
            const auto& e = vec.refs[i];
            for (int k = 0; k < A; k++)
                d_pol = std::max(d_pol, (double)std::fabs(r.policy[k] - e[k]));
            d_val   = std::max(d_val, (double)std::fabs(r.value - e[A]));
            d_score = std::max(d_score, (double)std::fabs(r.score - e[A + 1]));
            d_sd    = std::max(d_sd, (double)std::fabs(r.score_sd - e[A + 2]));
            for (int k = 0; k < hw; k++)
                d_own = std::max(d_own, (double)std::fabs(r.ownership[k] - e[A + 3 + k]));
        }

        ok = d_pol <= tol_pol && d_val <= tol_val &&
             d_score <= tol_score && d_sd <= tol_score && d_own <= tol_own;
        std::cout << std::scientific << std::setprecision(3)
                  << "max|diff|  policy=" << d_pol
                  << "  value=" << d_val
                  << "  score=" << d_score
                  << "  score_sd=" << d_sd
                  << "  ownership=" << d_own << "\n"
                  << (ok ? "PASS" : "FAIL")
                  << "  (tol: pol " << tol_pol << ", val " << tol_val
                  << ", score " << tol_score << ", own " << tol_own << ")\n";
    } else {
        // Bench without vectors: synthesize one random state.
        int hw = model->board_size * model->board_size;
        int state_len = (model->format == ModelFormat::KataGo)
            ? model->input_channels * hw + model->input_global_channels
            : model->input_channels * hw;
        vec.board = model->board_size;
        vec.state_len = state_len;
        vec.n = 1;
        vec.states.assign(1, std::vector<float>(state_len));
        unsigned seed = 12345;
        for (auto& v : vec.states[0]) {
            seed = seed * 1664525u + 1013904223u;
            v = (float)(seed >> 8) / (float)(1u << 24) - 0.5f;
        }
    }

    if (bench) {
        std::cout << std::fixed << std::setprecision(1);
        for (int B : {1, 16, 64, 128, max_batch}) {
            if (B > max_batch) continue;
            std::vector<std::vector<float>> batch(B, vec.states[0]);
            handle->predict_batch(batch);   // warm-up
            handle->predict_batch(batch);
            const int iters = std::max(3, 2000 / B);
            auto t0 = std::chrono::steady_clock::now();
            for (int it = 0; it < iters; it++)
                handle->predict_batch(batch);
            auto t1 = std::chrono::steady_clock::now();
            double s = std::chrono::duration<double>(t1 - t0).count();
            std::cout << "bench B=" << std::setw(3) << B
                      << "  " << std::setw(9) << (double)B * iters / s
                      << " evals/s   " << std::setw(8) << s / iters * 1e3
                      << " ms/batch\n";
        }
    }
    return ok ? 0 : 1;
}
