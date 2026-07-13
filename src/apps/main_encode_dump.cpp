// ================================================================
// Encoder parity dump — replay an UNCOMPRESSED V3 record and write
// the KataGo V7 encoding (22*n*n spatial + 19 global float32) of the
// position BEFORE each move.  tools/encoder_parity_test.py compares
// the output against scripts/gamedata.py's encode_katago, which must
// stay byte-identical to the C++ encoder.
//
//   encode_dump --record game.bin --output planes.bin [--time]
//
// Output layout (little-endian):
//   u32 magic 0x4B454E43 ('CNEK') | u32 board_size | u32 n_moves
//   then n_moves * (22*n*n + 19) float32
// ================================================================
#include "engine/game.h"
#include "engine/katago_inputs.h"
#include "model/loaded_model.h"

#include <chrono>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

using namespace minigo;

struct V3Record {
    int board_size = 0;
    float komi = 7.5f;
    std::vector<int16_t> actions;
};

static V3Record read_v3(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot open " + path);
    uint16_t magic = 0, version = 0;
    int32_t bs = 0, n_moves = 0;
    float komi = 0;
    int8_t winner = 0;
    float black_score = 0;
    f.read((char*)&magic, 2);
    f.read((char*)&version, 2);
    f.read((char*)&bs, 4);
    f.read((char*)&komi, 4);
    f.read((char*)&n_moves, 4);
    f.read((char*)&winner, 1);
    f.read((char*)&black_score, 4);
    if (magic != 0x4D47 || version != 3)
        throw std::runtime_error("not a V3 record: " + path);

    V3Record rec;
    rec.board_size = bs;
    rec.komi = komi;
    int hw = bs * bs;
    rec.actions.resize(n_moves);
    std::vector<float> policy(hw + 1);
    for (int m = 0; m < n_moves; m++) {
        f.read((char*)&rec.actions[m], 2);
        f.read((char*)policy.data(), (std::streamsize)(hw + 1) * 4);
    }
    if (!f) throw std::runtime_error("truncated V3 record: " + path);
    return rec;
}

int main(int argc, char* argv[]) {
    std::string record_path, output_path;
    bool time_it = false;
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        if      (a == "--record" && i + 1 < argc) record_path = argv[++i];
        else if (a == "--output" && i + 1 < argc) output_path = argv[++i];
        else if (a == "--time") time_it = true;
        else { std::cerr << "unknown arg " << a << "\n"; return 2; }
    }
    if (record_path.empty() || output_path.empty()) {
        std::cerr << "usage: encode_dump --record g.bin --output planes.bin [--time]\n";
        return 2;
    }

    V3Record rec = read_v3(record_path);
    const int n = rec.board_size;
    const int n_moves = (int)rec.actions.size();

    // Metadata-only LoadedModel: encode_for_katago just checks channels.
    LoadedModel model;
    model.format = ModelFormat::KataGo;
    model.board_size = n;
    model.input_channels = 22;
    model.input_global_channels = 19;

    std::ofstream out(output_path, std::ios::binary);
    uint32_t magic = 0x4B454E43u, bs32 = (uint32_t)n, nm32 = (uint32_t)n_moves;
    out.write((char*)&magic, 4);
    out.write((char*)&bs32, 4);
    out.write((char*)&nm32, 4);

    GoGame game(n, rec.komi);
    std::vector<float> enc;
    double total_us = 0.0;
    for (int m = 0; m < n_moves; m++) {
        auto t0 = std::chrono::steady_clock::now();
        encode_for_katago(game, &model, enc);
        auto t1 = std::chrono::steady_clock::now();
        total_us += std::chrono::duration<double, std::micro>(t1 - t0).count();
        out.write((const char*)enc.data(), (std::streamsize)enc.size() * 4);
        game.play((int)rec.actions[m]);
    }

    if (time_it)
        std::cout << "encoded " << n_moves << " positions, "
                  << total_us / std::max(1, n_moves) << " us/encode\n";
    return 0;
}
