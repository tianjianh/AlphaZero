#include "config.h"
#include "game.h"
#include "mcts.h"
#include "loaded_model.h"
#include "compute_context.h"
#include "nn_evaluator.h"
#include <algorithm>
#include <iostream>
#include <sstream>
#include <random>
#include <string>

using namespace minigo;

static int random_legal_move(GoGame& game) {
    std::vector<float> legal;
    game.get_legal_moves(legal);
    std::vector<int> actions;
    for (int a = 0; a < (int)legal.size(); a++)
        if (legal[a] > 0.0f) actions.push_back(a);
    static std::mt19937 rng(42);
    return actions[rng() % actions.size()];
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
    bool use_random         = false;
    int  board_override     = -1;
    int  search_threads     = 16;
    int  nn_server_threads  = 1;
    std::string nn_device_ids_str = "0";

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if      (arg == "--model"             && i+1<argc) model_path      = argv[++i];
        else if (arg == "--board"             && i+1<argc) board_override  = std::stoi(argv[++i]);
        else if (arg == "--sims"              && i+1<argc) config.num_simulations = std::stoi(argv[++i]);
        else if (arg == "--search-threads"    && i+1<argc) search_threads  = std::stoi(argv[++i]);
        else if (arg == "--komi"              && i+1<argc) config.komi     = std::stof(argv[++i]);
        else if (arg == "--max-batch"         && i+1<argc) config.max_batch_size = std::stoi(argv[++i]);
        else if (arg == "--c-puct"            && i+1<argc) config.c_puct   = std::stof(argv[++i]);
        else if (arg == "--nn-server-threads" && i+1<argc) nn_server_threads = std::stoi(argv[++i]);
        else if (arg == "--nn-device-ids"     && i+1<argc) nn_device_ids_str = argv[++i];
        else if (arg == "--random") use_random = true;
        else if (arg == "--help") {
            std::cout << "Usage: play [options]\n"
                      << "  --model PATH            Model file (default: models/best.onnx)\n"
                      << "  --board N               Board size (for --random mode)\n"
                      << "  --sims N                MCTS simulations (default: 800)\n"
                      << "  --search-threads N      MCTS search threads (default: 16)\n"
                      << "  --max-batch N           Max GPU batch size (default: 256)\n"
                      << "  --c-puct F              UCB exploration constant (default: 1.5)\n"
                      << "  --komi F                Komi value (default: 7.5)\n"
                      << "  --nn-server-threads N   NN server threads (default: 1)\n"
                      << "  --nn-device-ids IDS     Comma-separated device indices (default: \"0\")\n"
                      << "  --random                Use random bot (no model needed)\n";
            return 0;
        }
    }

    std::vector<int> device_ids = parse_device_ids(nn_device_ids_str);
    if ((int)device_ids.size() != nn_server_threads) {
        std::cerr << "ERROR: --nn-device-ids has " << device_ids.size()
                  << " entries but --nn-server-threads is " << nn_server_threads
                  << ". Must match exactly.\n";
        return 1;
    }

    std::shared_ptr<LoadedModel> model;
    std::shared_ptr<ComputeContext> context;
    std::shared_ptr<NNEvaluator> evaluator;
    std::unique_ptr<MCTS> mcts;

    if (!use_random) {
        try {
            model   = LoadedModel::load(model_path);
            context = std::shared_ptr<ComputeContext>(create_compute_context(device_ids));

            config.board_size         = model->board_size;
            config.input_channels     = model->input_channels;
            config.num_filters        = model->num_filters;
            config.num_res_blocks     = model->num_res_blocks;
            config.max_moves_per_game = config.board_size * config.board_size * 2;
            config.num_search_threads = search_threads;

            evaluator = std::make_shared<NNEvaluator>(
                model, context, device_ids, config.max_batch_size);
            mcts = std::make_unique<MCTS>(evaluator.get(), config);
        } catch (const std::exception& e) {
            std::cerr << "Error: " << e.what() << "\n";
            std::cout << "Use --random for random bot, or train first.\n";
            return 1;
        }
    } else {
        if (board_override > 0) config.board_size = board_override;
        config.max_moves_per_game = config.board_size * config.board_size * 2;
    }

    bool keep_playing = true;
    while (keep_playing) {
        GoGame game(config.board_size, config.komi);

        std::cout << "\n========================================\n"
                  << "  MINIGO — Human vs AI (C++)\n"
                  << "========================================\n";

        std::cout << "Play as (B)lack or (W)hite? [B]: ";
        std::string choice;
        std::getline(std::cin, choice);

        Stone human_color = BLACK;
        if (!choice.empty() && (choice[0] == 'W' || choice[0] == 'w'))
            human_color = WHITE;
        Stone ai_color = opponent(human_color);

        std::cout << "\nYou are " << (human_color == BLACK ? "Black (X)" : "White (O)")
                  << "\nBoard: " << config.board_size << "x" << config.board_size
                  << "  Komi: " << config.komi
                  << "  Sims: " << config.num_simulations;
        if (context) std::cout << "  Backend: " << context->backend_name();
        std::cout << "\nMoves: A1-"
                  << (char)('A' + (config.board_size > 8 ? config.board_size
                                                         : config.board_size - 1))
                  << config.board_size
                  << " (columns skip I), PASS, Q to quit\n\n";

        while (!game.game_over) {
            std::cout << game.display() << "\n\n";

            if (game.current_player == human_color) {
                while (true) {
                    std::cout << "Your move: ";
                    std::string input;
                    std::getline(std::cin, input);
                    if (input.empty()) continue;
                    std::transform(input.begin(), input.end(), input.begin(), ::toupper);

                    if (input == "Q") { std::cout << "Goodbye!\n"; return 0; }

                    int action;
                    if (input == "PASS") {
                        action = config.action_size() - 1;
                    } else {
                        try { action = game.str_to_action(input); }
                        catch (...) { std::cout << "  Invalid input.\n"; continue; }
                    }

                    if (action == config.action_size() - 1) {
                        game.play(PASS_MOVE); break;
                    } else if (game.is_legal(action)) {
                        game.play(action); break;
                    } else {
                        std::cout << "  Illegal move!\n";
                    }
                }
            } else {
                std::cout << "AI thinking (" << config.num_simulations << " sims)...\n";

                int action;
                if (use_random) {
                    action = random_legal_move(game);
                } else {
                    std::vector<float> pi;
                    action = mcts->get_action(game, pi, 0.0f, -1, false);
                }

                std::string move_str;
                if (action == config.action_size() - 1) {
                    move_str = "PASS"; game.play(PASS_MOVE);
                } else {
                    move_str = game.action_to_str(action); game.play(action);
                }
                std::cout << "AI plays: " << move_str << "\n\n";
            }
        }

        std::cout << "\n========================================\n"
                  << "  GAME OVER\n"
                  << "========================================\n"
                  << game.display() << "\n";

        auto [bs, ws] = game.score();
        std::cout << "\nBlack: " << bs << "  |  White: " << ws
                  << " (incl. " << config.komi << " komi)\n";

        if      (game.winner == human_color) std::cout << "You win!\n";
        else if (game.winner == ai_color)    std::cout << "AI wins!\n";
        else                                 std::cout << "Draw!\n";

        std::cout << "\nPlay again? [Y/n]: ";
        std::string again;
        std::getline(std::cin, again);
        keep_playing = again.empty() || again[0] == 'Y' || again[0] == 'y';
    }

    std::cout << "Thanks for playing!\n";
    return 0;
}
