#pragma once

#include "game.h"
#include <string>

namespace minigo {

struct Config {
    int board_rows = BOARD_ROWS;
    int board_cols = BOARD_COLS;
    int history_length = 4;

    std::string model_type = "xiangqi-resnet";
    int num_res_blocks = 10;
    int num_filters = 128;
    int input_channels = history_length * 14 + 1;
    int vit_depth = 0;
    int vit_heads = 0;
    int vit_kv_groups = 0;

    int num_simulations = 800;
    float c_puct = 1.5f;
    float dirichlet_alpha = 0.30f;
    float dirichlet_epsilon = 0.25f;
    int temperature_threshold = 18;
    float win_loss_weight = 1.0f;
    float score_weight = 0.0f;
    float score_scale = 1000.0f;

    int max_moves_per_game = 300;
    int num_search_threads = 1;
    int virtual_loss_parallel = 32;
    int max_batch_size = 256;

    int board_area() const { return board_rows * board_cols; }
    int action_size() const { return board_area() * board_area(); }
};

}  // namespace minigo
