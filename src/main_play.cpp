#include "async_bot.h"
#include "compute_context.h"
#include "config.h"
#include "game.h"
#include "loaded_model.h"
#include "mcts.h"
#include "nn_evaluator.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <future>
#include <locale.h>
#include <mutex>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include <curses.h>

using namespace minigo;

enum {
    CP_GRID = 1,
    CP_STATUS,
    CP_ACCENT,
    CP_RED,
    CP_LABEL,
    CP_RED_PIECE,
    CP_BLACK_PIECE,
    CP_CURSOR,
    CP_AI_INFO,
};

static void init_colors() {
    start_color();
    use_default_colors();
    init_pair(CP_GRID, 244, -1);
    init_pair(CP_STATUS, 252, -1);
    init_pair(CP_ACCENT, 214, -1);
    init_pair(CP_RED, 196, -1);
    init_pair(CP_LABEL, 245, -1);
    init_pair(CP_RED_PIECE, 203, -1);
    init_pair(CP_BLACK_PIECE, 81, -1);
    init_pair(CP_CURSOR, 46, -1);
    init_pair(CP_AI_INFO, 117, -1);
}

static std::vector<int> parse_device_ids(const std::string& str) {
    std::vector<int> ids;
    std::istringstream ss(str);
    std::string token;
    while (std::getline(ss, token, ','))
        ids.push_back(std::stoi(token));
    return ids;
}

static int random_legal_move(GoGame& game) {
    std::vector<float> legal;
    game.get_legal_moves(legal);
    std::vector<int> moves;
    for (int a = 0; a < (int)legal.size(); ++a)
        if (legal[a] > 0.0f) moves.push_back(a);
    static std::mt19937 rng{42};
    return moves.empty() ? 0 : moves[rng() % moves.size()];
}

static char piece_char(int8_t piece) {
    switch (piece) {
        case RED_KING: return 'K';
        case RED_ADVISOR: return 'A';
        case RED_BISHOP: return 'B';
        case RED_KNIGHT: return 'N';
        case RED_ROOK: return 'R';
        case RED_CANNON: return 'C';
        case RED_PAWN: return 'P';
        case BLACK_KING: return 'k';
        case BLACK_ADVISOR: return 'a';
        case BLACK_BISHOP: return 'b';
        case BLACK_KNIGHT: return 'n';
        case BLACK_ROOK: return 'r';
        case BLACK_CANNON: return 'c';
        case BLACK_PAWN: return 'p';
        default: return '.';
    }
}

static bool is_red_piece(int8_t piece) {
    return piece >= RED_KING && piece <= RED_PAWN;
}

static Stone piece_side(int8_t piece) {
    if (piece >= RED_KING && piece <= RED_PAWN) return RED;
    if (piece >= BLACK_KING && piece <= BLACK_PAWN) return BLACK;
    return EMPTY;
}

static bool try_parse_iccs(const std::string& input, GoGame& game, int& action) {
    try {
        action = game.str_to_action(input);
        return true;
    } catch (...) {
        return false;
    }
}

static void draw_board(WINDOW* win, const GoGame& game,
                       int cursor_r, int cursor_c, bool cursor_active,
                       int selected_sq, const std::string& status_msg,
                       const std::string& ai_info,
                       const std::string& input_buf,
                       Stone human_color) {
    werase(win);
    int h, w;
    getmaxyx(win, h, w);

    const char* title = "MiniXiangqi";
    wattron(win, COLOR_PAIR(CP_ACCENT) | A_BOLD);
    mvwaddstr(win, 0, std::max(0, (w - (int)strlen(title)) / 2), title);
    wattroff(win, COLOR_PAIR(CP_ACCENT) | A_BOLD);

    std::ostringstream status;
    status << "You: " << (human_color == RED ? "Red" : human_color == BLACK ? "Black" : "Human")
           << "  Turn: " << (game.current_player == RED ? "Red" : "Black")
           << "  Move: " << (game.move_count + 1);
    wattron(win, COLOR_PAIR(CP_STATUS));
    mvwaddnstr(win, 1, std::max(0, (w - (int)status.str().size()) / 2),
               status.str().c_str(), w - 2);
    wattroff(win, COLOR_PAIR(CP_STATUS));

    int ox = 4;
    int oy = 3;
    int panel_x = 32;
    const char* files = "abcdefghi";

    wattron(win, COLOR_PAIR(CP_LABEL));
    for (int c = 0; c < BOARD_COLS; ++c) {
        mvwaddch(win, oy - 1, ox + 2 + c * 3, files[c]);
    }
    wattroff(win, COLOR_PAIR(CP_LABEL));

    for (int r = 0; r < BOARD_ROWS; ++r) {
        int y = oy + r + (r >= 5 ? 1 : 0);
        char rank_label[4];
        snprintf(rank_label, sizeof(rank_label), "%d", BOARD_ROWS - 1 - r);
        wattron(win, COLOR_PAIR(CP_LABEL));
        mvwaddstr(win, y, ox, rank_label);
        wattroff(win, COLOR_PAIR(CP_LABEL));

        for (int c = 0; c < BOARD_COLS; ++c) {
            int x = ox + 2 + c * 3;
            int sq = r * BOARD_COLS + c;
            bool is_cursor = cursor_active && r == cursor_r && c == cursor_c;
            bool is_selected = selected_sq == sq;
            char ch = piece_char(game.board[r][c]);

            if (is_selected) wattron(win, COLOR_PAIR(CP_CURSOR) | A_BOLD);
            else if (is_red_piece(game.board[r][c])) wattron(win, COLOR_PAIR(CP_RED_PIECE) | A_BOLD);
            else if (game.board[r][c] != NO_PIECE) wattron(win, COLOR_PAIR(CP_BLACK_PIECE) | A_BOLD);
            else wattron(win, COLOR_PAIR(CP_GRID));

            mvwaddch(win, y, x, ch);

            if (is_selected) wattroff(win, COLOR_PAIR(CP_CURSOR) | A_BOLD);
            else if (is_red_piece(game.board[r][c])) wattroff(win, COLOR_PAIR(CP_RED_PIECE) | A_BOLD);
            else if (game.board[r][c] != NO_PIECE) wattroff(win, COLOR_PAIR(CP_BLACK_PIECE) | A_BOLD);
            else wattroff(win, COLOR_PAIR(CP_GRID));

            if (c < BOARD_COLS - 1) {
                wattron(win, COLOR_PAIR(CP_GRID));
                mvwaddstr(win, y, x + 1, "--");
                wattroff(win, COLOR_PAIR(CP_GRID));
            }
            if (r < BOARD_ROWS - 1 && !(r == 4)) {
                wattron(win, COLOR_PAIR(CP_GRID));
                mvwaddch(win, y + 1, x, '|');
                wattroff(win, COLOR_PAIR(CP_GRID));
            }
        }

        wattron(win, COLOR_PAIR(CP_LABEL));
        mvwaddstr(win, y, ox + 30, rank_label);
        wattroff(win, COLOR_PAIR(CP_LABEL));
    }

    wattron(win, COLOR_PAIR(CP_LABEL));
    mvwaddstr(win, oy + 5, ox + 6, "Chu He");
    mvwaddstr(win, oy + 5, ox + 17, "Han Jie");
    wattroff(win, COLOR_PAIR(CP_LABEL));

    if (cursor_active) {
        int y = oy + cursor_r + (cursor_r >= 5 ? 1 : 0);
        int x = ox + 2 + cursor_c * 3;
        wattron(win, COLOR_PAIR(CP_CURSOR));
        mvwaddch(win, y, x - 1, '[');
        mvwaddch(win, y, x + 1, ']');
        wattroff(win, COLOR_PAIR(CP_CURSOR));
    }

    if (!status_msg.empty()) {
        wattron(win, COLOR_PAIR(CP_STATUS) | A_BOLD);
        mvwaddnstr(win, oy, panel_x, status_msg.c_str(), w - panel_x - 2);
        wattroff(win, COLOR_PAIR(CP_STATUS) | A_BOLD);
    }

    if (!ai_info.empty()) {
        wattron(win, COLOR_PAIR(CP_AI_INFO));
        int y = oy + 2;
        size_t pos = 0;
        while (pos < ai_info.size() && y < h - 4) {
            size_t sep = ai_info.find('|', pos);
            std::string line = ai_info.substr(pos, sep == std::string::npos ? std::string::npos : sep - pos);
            mvwaddnstr(win, y++, panel_x, line.c_str(), w - panel_x - 2);
            pos = (sep == std::string::npos) ? ai_info.size() : sep + 1;
        }
        wattroff(win, COLOR_PAIR(CP_AI_INFO));
    }

    if (!input_buf.empty() && !game.game_over) {
        std::string prompt = "Move: " + input_buf + "_";
        wattron(win, COLOR_PAIR(CP_STATUS));
        mvwaddnstr(win, h - 3, panel_x, prompt.c_str(), w - panel_x - 2);
        wattroff(win, COLOR_PAIR(CP_STATUS));
    }

    const char* help = "arrows:move  enter:select/move  backspace:cancel  a:analyze  p:ponder  q:quit";
    wattron(win, COLOR_PAIR(CP_LABEL));
    mvwaddnstr(win, h - 1, 2, help, w - 4);
    wattroff(win, COLOR_PAIR(CP_LABEL));

    wrefresh(win);
}

int main(int argc, char* argv[]) {
    Config config;
    std::string model_path = "models/best.onnx";
    bool use_random = false;
    int search_threads = 16;
    int nn_server_threads = 1;
    int pvs = 5;
    std::string nn_device_ids_str = "0";

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if      (arg == "--model"             && i + 1 < argc) model_path = argv[++i];
        else if (arg == "--sims"              && i + 1 < argc) config.num_simulations = std::stoi(argv[++i]);
        else if (arg == "--search-threads"    && i + 1 < argc) search_threads = std::stoi(argv[++i]);
        else if (arg == "--win-loss-weight"   && i + 1 < argc) config.win_loss_weight = std::stof(argv[++i]);
        else if (arg == "--score-weight"      && i + 1 < argc) config.score_weight = std::stof(argv[++i]);
        else if (arg == "--score-scale"       && i + 1 < argc) config.score_scale = std::stof(argv[++i]);
        else if (arg == "--max-batch"         && i + 1 < argc) config.max_batch_size = std::stoi(argv[++i]);
        else if (arg == "--c-puct"            && i + 1 < argc) config.c_puct = std::stof(argv[++i]);
        else if (arg == "--nn-server-threads" && i + 1 < argc) nn_server_threads = std::stoi(argv[++i]);
        else if (arg == "--nn-device-ids"     && i + 1 < argc) nn_device_ids_str = argv[++i];
        else if (arg == "--pvs"               && i + 1 < argc) pvs = std::stoi(argv[++i]);
        else if (arg == "--random") use_random = true;
        else if (arg == "--help" || arg == "-h") {
            printf("Usage: play [options]\n"
                   "  --model PATH           Model file (default: models/best.onnx)\n"
                   "  --sims N               MCTS simulations (default: 800)\n"
                   "  --search-threads N     MCTS search threads (default: 16)\n"
                   "  --max-batch N          Max GPU batch size (default: 256)\n"
                   "  --win-loss-weight F    Win/loss utility weight (default: 1.0)\n"
                   "  --score-weight F       Score utility weight (default: 0.0)\n"
                   "  --score-scale F        Score utility scale (default: 1000.0)\n"
                   "  --c-puct F             UCB exploration constant (default: 1.5)\n"
                   "  --nn-server-threads N  NN server threads (default: 1)\n"
                   "  --nn-device-ids IDS    Comma-separated GPU indices (default: \"0\")\n"
                   "  --pvs N                Top K moves to show in analysis (default: 5)\n"
                   "  --random               Random bot (no model needed)\n");
            return 0;
        } else {
            fprintf(stderr, "Error: unrecognized option '%s'\n", argv[i]);
            return 1;
        }
    }

    std::vector<int> device_ids = parse_device_ids(nn_device_ids_str);
    std::shared_ptr<LoadedModel> model;
    std::shared_ptr<ComputeContext> context;
    std::shared_ptr<NNEvaluator> evaluator;
    std::unique_ptr<AsyncBot> bot;

    if (!use_random) {
        try {
            model = LoadedModel::load(model_path);
            context = std::shared_ptr<ComputeContext>(create_compute_context(device_ids));
            config.model_type = model->model_type;
            config.board_rows = model->board_rows;
            config.board_cols = model->board_cols;
            config.history_length = std::max(1, (model->input_channels - 1) / 14);
            config.input_channels = model->input_channels;
            config.num_filters = model->num_filters;
            config.num_res_blocks = model->num_res_blocks;
            config.num_search_threads = search_threads;
            evaluator = std::make_shared<NNEvaluator>(
                model, context, device_ids, config.max_batch_size);
            printf("Waiting for Metal inference handles...\n");
            fflush(stdout);
            evaluator->wait_ready();
            bot = std::make_unique<AsyncBot>(evaluator.get(), config);
        } catch (const std::exception& e) {
            fprintf(stderr, "Error: %s\nUse --random for a rule-only opponent.\n", e.what());
            return 1;
        }
    }

    setlocale(LC_ALL, "");
    initscr();
    init_colors();
    cbreak();
    noecho();
    keypad(stdscr, TRUE);
    curs_set(0);

    bool keep_playing = true;
    while (keep_playing) {
        werase(stdscr);
        attron(COLOR_PAIR(CP_ACCENT) | A_BOLD);
        mvaddstr(2, 4, "MiniXiangqi");
        attroff(COLOR_PAIR(CP_ACCENT) | A_BOLD);
        mvaddstr(4, 4, "Play as (R)ed, (B)lack, or (H)uman vs Human? [R]: ");
        refresh();
        curs_set(1);
        echo();
        char choice_buf[8] = {};
        getnstr(choice_buf, 4);
        noecho();
        curs_set(0);

        bool human_vs_human = (choice_buf[0] == 'H' || choice_buf[0] == 'h');
        Stone human_color = RED;
        if (choice_buf[0] == 'B' || choice_buf[0] == 'b') human_color = BLACK;

        GoGame game(config.history_length);
        if (bot) bot->reset(game);

        int cursor_r = 9;
        int cursor_c = 4;
        bool cursor_active = true;
        int selected_sq = -1;
        std::string input_buf;
        std::string status_msg;
        std::string ai_info;

        bool pondering_wanted = false;
        bool analysis_wanted = false;

        std::mutex ai_info_mutex;
        MCTS::AnalysisInfo latest_info;
        bool have_info = false;

        auto analyze_callback = [&](const MCTS::AnalysisInfo& info) {
            std::lock_guard<std::mutex> lock(ai_info_mutex);
            latest_info = info;
            have_info = (info.total_visits > 0);
        };

        auto apply_bot_state = [&]() {
            if (!bot) return;
            bot->stop();
            if (analysis_wanted) {
                bot->set_callback(analyze_callback, 100, pvs);
            } else {
                bot->clear_callback();
                std::lock_guard<std::mutex> lock(ai_info_mutex);
                have_info = false;
                ai_info.clear();
            }
            if (pondering_wanted) bot->start_ponder();
        };

        auto is_human_turn = [&]() {
            return human_vs_human || game.current_player == human_color;
        };

        auto refresh_ai_info = [&]() {
            if (!analysis_wanted) {
                ai_info = pondering_wanted ? "[pondering in background]" : "";
                return;
            }

            MCTS::AnalysisInfo info;
            {
                std::lock_guard<std::mutex> lock(ai_info_mutex);
                if (!have_info) {
                    ai_info = "[analyzing - waiting for first batch]";
                    return;
                }
                info = latest_info;
            }

            char buf[256];
            float wr = (info.root_utility + 1.0f) * 50.0f;
            snprintf(buf, sizeof(buf), "Eval %.1f%%  Score %+.1f  N=%d",
                     wr, info.root_score, info.total_visits);
            ai_info = buf;
            for (const auto& m : info.moves) {
                snprintf(buf, sizeof(buf), "| %-5s %5.1f%% n=%-5d",
                         game.action_to_str(m.action).c_str(),
                         (m.utility + 1.0f) * 50.0f, m.visits);
                ai_info += buf;
            }
        };

        while (true) {
            if (bot && is_human_turn() && pondering_wanted && !bot->is_searching())
                bot->start_ponder();

            timeout(analysis_wanted ? 100 : -1);
            refresh_ai_info();
            draw_board(stdscr, game, cursor_r, cursor_c, cursor_active,
                       selected_sq, status_msg, ai_info, input_buf, human_color);

            if (game.game_over) {
                if (bot) {
                    pondering_wanted = false;
                    analysis_wanted = false;
                    apply_bot_state();
                }
                std::string end_msg;
                if (game.winner == EMPTY) end_msg = "Draw. [r]estart or [q]uit";
                else end_msg = std::string(game.winner == RED ? "Red" : "Black") + " wins. [r]estart or [q]uit";
                status_msg = end_msg;
                draw_board(stdscr, game, cursor_r, cursor_c, cursor_active,
                           selected_sq, status_msg, ai_info, input_buf, human_color);
                int key = getch();
                if (key == 'q' || key == 'Q') { keep_playing = false; break; }
                if (key == 'r' || key == 'R') break;
                continue;
            }

            if (!is_human_turn()) {
                status_msg = "AI thinking...";
                int action = 0;
                if (use_random) {
                    action = random_legal_move(game);
                    game.play(action);
                } else {
                    auto future = std::async(std::launch::async, [&]() {
                        return bot->gen_move(game.current_player, -1, 0.0f, false);
                    });
                    while (future.wait_for(std::chrono::milliseconds(100)) != std::future_status::ready) {
                        refresh_ai_info();
                        draw_board(stdscr, game, cursor_r, cursor_c, cursor_active,
                                   selected_sq, status_msg, ai_info, input_buf, human_color);
                    }
                    action = future.get();
                    game = bot->game();
                }
                status_msg = "AI plays: " + game.action_to_str(action);
                selected_sq = -1;
                continue;
            }

            int key = getch();
            if (key == ERR) continue;
            status_msg.clear();

            if (key == 'q' || key == 'Q') {
                if (bot) {
                    pondering_wanted = false;
                    analysis_wanted = false;
                    apply_bot_state();
                }
                keep_playing = false;
                break;
            }
            if (key == 'r' || key == 12) {
                clearok(stdscr, TRUE);
                continue;
            }
            if (key == 'a') {
                analysis_wanted = !analysis_wanted;
                if (analysis_wanted) pondering_wanted = true;
                apply_bot_state();
                continue;
            }
            if (key == 'p' || key == 'P') {
                pondering_wanted = !pondering_wanted;
                if (!pondering_wanted) analysis_wanted = false;
                apply_bot_state();
                continue;
            }

            auto commit_action = [&](int action) {
                if (bot) {
                    if (!bot->play_move(game.current_player, action)) {
                        status_msg = "Illegal move";
                        return false;
                    }
                    game = bot->game();
                } else {
                    if (!game.is_legal(action)) {
                        status_msg = "Illegal move";
                        return false;
                    }
                    game.play(action);
                }
                selected_sq = -1;
                input_buf.clear();
                return true;
            };

            if (key == KEY_UP || key == 'w' || key == 'W') cursor_r = std::max(0, cursor_r - 1);
            else if (key == KEY_DOWN || key == 's' || key == 'S') cursor_r = std::min(BOARD_ROWS - 1, cursor_r + 1);
            else if (key == KEY_LEFT) cursor_c = std::max(0, cursor_c - 1);
            else if (key == KEY_RIGHT || key == 'd' || key == 'D') cursor_c = std::min(BOARD_COLS - 1, cursor_c + 1);
            else if (key == KEY_BACKSPACE || key == 127 || key == 8) {
                if (!input_buf.empty()) input_buf.pop_back();
                else selected_sq = -1;
            } else if (key == 10 || key == 13 || key == ' ') {
                int sq = cursor_r * BOARD_COLS + cursor_c;
                if (selected_sq < 0) {
                    int8_t piece = game.board[cursor_r][cursor_c];
                    if (piece == NO_PIECE || piece_side(piece) != game.current_player) {
                        status_msg = "Select one of your pieces";
                    } else {
                        selected_sq = sq;
                    }
                } else if (selected_sq == sq) {
                    selected_sq = -1;
                } else {
                    commit_action(selected_sq * BOARD_AREA + sq);
                }
            } else if (std::isalnum(key)) {
                if (input_buf.size() < 4) input_buf.push_back((char)key);
                int action = 0;
                if (input_buf.size() >= 4 && try_parse_iccs(input_buf, game, action)) {
                    commit_action(action);
                }
            }
        }
    }

    endwin();
    return 0;
}
