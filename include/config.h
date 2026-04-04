#pragma once

namespace minigo {

struct Config {
    // Board
    int board_size = 9;
    float komi = 7.5f;

    // Neural network
    int num_res_blocks = 5;
    int num_filters = 64;
    int input_channels = 17;  // 8 history * 2 + 1 color

    // MCTS
    int num_simulations = 800;
    float c_puct = 1.5f;
    float dirichlet_alpha = 0.15f;   // ~10/avg_legal_moves (0.15 for 9x9, 0.03 for 19x19)
    float dirichlet_epsilon = 0.25f; // blend: 75% network prior + 25% noise
    int temperature_threshold = 15;  // moves of stochastic play (rest is greedy)

    // Self-play
    int max_moves_per_game = 162;  // board_size^2 * 2

    // Multi-threaded MCTS (KataGo pattern)
    //
    // Each MCTS::search() spawns num_search_threads internal threads.
    // Each thread: descend → evaluate_single(block) → expand → backprop → repeat.
    // Collisions (node being evaluated by another thread): revert vloss, yield, retry.
    //
    // Batch size adapts to total concurrent search threads across all games:
    //   total_threads = min(games, selfplay_threads) × num_search_threads
    //
    // Tuning: increase num_search_threads until GPU utilization plateaus.
    // KataGo recommends 8-32 per position for strong GPUs.
    int num_search_threads = 1;     // search threads per MCTS::search() call
    int virtual_loss_parallel = 32; // VLP for single-threaded Eigen fallback

    // NN server batch inference (independent of search thread count)
    // Server takes min(queue_size, max_batch_size) — just a cap.
    int max_batch_size = 256;       // max states in one GPU call

    // Derived
    int action_size() const { return board_size * board_size + 1; }
    int pass_action() const { return board_size * board_size; }
};

}  // namespace minigo
