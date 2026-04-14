#include "config.h"
#include "game.h"
#include "mcts.h"
#include "async_bot.h"
#include "loaded_model.h"
#include "compute_context.h"
#include "nn_evaluator.h"
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <mutex>
#include <random>
#include <sstream>
#include <string>
#include <thread>
#include <future>
#include <locale.h>
#include <curses.h>

using namespace minigo;

static const char* GO_COLS = "ABCDEFGHJKLMNOPQRSTUVWXYZ";

// ================================================================
// Color pairs
// ================================================================
enum {
    CP_GRID = 1,    // grid lines
    CP_STATUS,      // status text
    CP_ACCENT,      // gold accent / title
    CP_RED,         // last move / winner
    CP_LABEL,       // dim labels
    CP_BLACK_STONE, // black stone
    CP_WHITE_STONE, // white stone
    CP_CURSOR,      // cursor highlight
    CP_AI_INFO,     // AI stats
};

static void init_colors() {
    start_color();
    use_default_colors();
    init_pair(CP_GRID,        237,  -1);   // dark gray grid
    init_pair(CP_STATUS,      252,  -1);   // status text
    init_pair(CP_ACCENT,      214,  -1);   // gold title
    init_pair(CP_RED,         196,  -1);   // red highlights
    init_pair(CP_LABEL,       245,  -1);   // dim labels
    init_pair(CP_BLACK_STONE, 255,  -1);   // black stone (bright for visibility)
    init_pair(CP_WHITE_STONE, 252,  -1);   // white stone
    init_pair(CP_CURSOR,       46,  -1);   // green cursor
    init_pair(CP_AI_INFO,      81,  -1);   // cyan AI info
}

// ================================================================
// Star points for standard board sizes
// ================================================================
static bool is_star_point(int r, int c, int n) {
    if (n == 9)  return (r==2||r==6)&&(c==2||c==6) || (r==4&&c==4);  // 4 corners + tengen
    if (n == 13) return (r==3||r==9)&&(c==3||c==9) || (r==6&&c==6);
    if (n == 19) return (r==3||r==15)&&(c==3||c==15) || (r==9&&c==9)
                     || (r==3||r==15)&&c==9 || r==9&&(c==3||c==15);  // 9 points but keep standard
    return false;
}

// ================================================================
// Board drawing (ncurses)
// ================================================================
static void draw_board(WINDOW* win, const GoGame& game, const Config& config,
                        int cursor_r, int cursor_c, bool cursor_active,
                        int last_r, int last_c,
                        Stone human_color, bool use_random,
                        const std::string& status_msg,
                        const std::string& ai_info,
                        const std::string& input_buf) {
    werase(win);
    int h, w;
    getmaxyx(win, h, w);
    int n = game.board_size;

    // Title
    const char* title = "MiniGo";
    wattron(win, COLOR_PAIR(CP_ACCENT) | A_BOLD);
    mvwaddstr(win, 0, std::max(0, (w - (int)strlen(title)) / 2), title);
    wattroff(win, COLOR_PAIR(CP_ACCENT) | A_BOLD);

    // Status line
    {
        char buf[128];
        Stone ai_color = (human_color == BLACK) ? WHITE : BLACK;
        snprintf(buf, sizeof(buf), "You: %s  AI: %s  Move: %d",
                 human_color == BLACK ? "Black" : "White",
                 ai_color == BLACK ? "Black" : "White",
                 game.move_count + 1);
        wattron(win, COLOR_PAIR(CP_STATUS));
        mvwaddstr(win, 1, std::max(0, (w - (int)strlen(buf)) / 2), buf);
        wattroff(win, COLOR_PAIR(CP_STATUS));
    }

    // Board left-aligned; right side reserved for info
    int ox = 5;   // left margin for row labels
    int oy = 3;
    int cell_w = 2;  // chars per cell (intersection + 1 hline)

    // Column labels
    wattron(win, COLOR_PAIR(CP_LABEL));
    for (int c = 0; c < n; c++)
        mvwaddch(win, oy - 1, ox + c * cell_w, GO_COLS[c]);
    wattroff(win, COLOR_PAIR(CP_LABEL));

    // Board: single-height rows, 2 chars per cell
    for (int r = 0; r < n; r++) {
        int y = oy + r;

        // Row label
        char rl[4];
        snprintf(rl, sizeof(rl), "%2d", n - r);
        wattron(win, COLOR_PAIR(CP_LABEL));
        mvwaddstr(win, y, ox - 3, rl);
        wattroff(win, COLOR_PAIR(CP_LABEL));

        for (int c = 0; c < n; c++) {
            int x = ox + c * cell_w;

            // Grid character
            chtype grid_ch;
            if (r == 0) {
                if (c == 0) grid_ch = ACS_ULCORNER;
                else if (c == n-1) grid_ch = ACS_URCORNER;
                else grid_ch = ACS_TTEE;
            } else if (r == n-1) {
                if (c == 0) grid_ch = ACS_LLCORNER;
                else if (c == n-1) grid_ch = ACS_LRCORNER;
                else grid_ch = ACS_BTEE;
            } else {
                if (c == 0) grid_ch = ACS_LTEE;
                else if (c == n-1) grid_ch = ACS_RTEE;
                else grid_ch = ACS_PLUS;
            }

            bool is_cursor = cursor_active && (r == cursor_r && c == cursor_c);
            bool is_last   = (r == last_r && c == last_c);
            int cell = game.board[r][c];

            if (cell == BLACK) {
                wattron(win, COLOR_PAIR(CP_BLACK_STONE) | A_BOLD);
                mvwaddch(win, y, x, 'X');
                wattroff(win, COLOR_PAIR(CP_BLACK_STONE) | A_BOLD);
            } else if (cell == WHITE) {
                wattron(win, COLOR_PAIR(CP_WHITE_STONE) | A_BOLD);
                mvwaddch(win, y, x, 'O');
                wattroff(win, COLOR_PAIR(CP_WHITE_STONE) | A_BOLD);
            } else if (is_cursor && !game.game_over) {
                char ghost = (game.current_player == BLACK) ? 'X' : 'O';
                wattron(win, A_BOLD);
                mvwaddch(win, y, x, ghost);
                wattroff(win, A_BOLD);
            } else if (is_star_point(r, c, n)) {
                wattron(win, COLOR_PAIR(CP_GRID));
                mvwaddch(win, y, x, '*');
                wattroff(win, COLOR_PAIR(CP_GRID));
            } else {
                wattron(win, COLOR_PAIR(CP_GRID));
                mvwaddch(win, y, x, grid_ch);
                wattroff(win, COLOR_PAIR(CP_GRID));
            }

            // Horizontal connector
            if (c < n - 1) {
                wattron(win, COLOR_PAIR(CP_GRID));
                mvwaddch(win, y, x + 1, ACS_HLINE);
                wattroff(win, COLOR_PAIR(CP_GRID));
            }
        }

        // Row label right
        wattron(win, COLOR_PAIR(CP_LABEL));
        char rr[4];
        snprintf(rr, sizeof(rr), "%d", n - r);
        mvwaddstr(win, y, ox + (n - 1) * cell_w + 2, rr);
        wattroff(win, COLOR_PAIR(CP_LABEL));
    }

    // Column labels bottom
    wattron(win, COLOR_PAIR(CP_LABEL));
    for (int c = 0; c < n; c++)
        mvwaddch(win, oy + n, ox + c * cell_w, GO_COLS[c]);
    wattroff(win, COLOR_PAIR(CP_LABEL));

    // Cursor brackets — only when arrow keys active
    if (!game.game_over && cursor_r >= 0 && cursor_c >= 0 && cursor_active) {
        int cx = ox + cursor_c * cell_w;
        int cy = oy + cursor_r;
        mvwaddch(win, cy, cx - 1, '[');
        mvwaddch(win, cy, cx + 1, ']');
    }

    // ── Right-side info panel ──
    int panel_x = ox + n * cell_w + 4;  // right of board
    int panel_y = oy;
    int panel_w = w - panel_x - 1;
    if (panel_w < 10) panel_w = 10;

    // Status message
    if (!status_msg.empty()) {
        int cp = CP_STATUS;
        if (status_msg.find("win") != std::string::npos || status_msg.find("Win") != std::string::npos)
            cp = CP_RED;
        wattron(win, COLOR_PAIR(cp) | A_BOLD);
        mvwaddnstr(win, panel_y, panel_x, status_msg.c_str(), panel_w);
        wattroff(win, COLOR_PAIR(cp) | A_BOLD);
    }
    panel_y += 2;

    // AI info
    if (!ai_info.empty()) {
        // Split ai_info by | for multi-line display
        wattron(win, COLOR_PAIR(CP_AI_INFO));
        size_t pos = 0;
        while (pos < ai_info.size() && panel_y < h - 3) {
            size_t sep = ai_info.find('|', pos);
            std::string line = ai_info.substr(pos, sep == std::string::npos ? std::string::npos : sep - pos);
            mvwaddnstr(win, panel_y++, panel_x, line.c_str(), panel_w);
            pos = (sep == std::string::npos) ? ai_info.size() : sep + 1;
        }
        wattroff(win, COLOR_PAIR(CP_AI_INFO));
    }
    panel_y++;

    // Input display
    if (game.current_player == human_color && !game.game_over) {
        std::string prompt = "Move: " + input_buf + "_";
        wattron(win, COLOR_PAIR(CP_STATUS));
        mvwaddstr(win, panel_y, panel_x, prompt.c_str());
        wattroff(win, COLOR_PAIR(CP_STATUS));
    }

    // Help at bottom
    int help_y = std::max(oy + n + 1, (int)(panel_y + 2));
    const char* help = "arrows:move  enter:place  p:pass  a:analyze  o:ownership  P:ponder  r:refresh  q:quit";
    wattron(win, COLOR_PAIR(CP_LABEL));
    mvwaddnstr(win, std::min(help_y, h - 1), 2, help, w - 4);
    wattroff(win, COLOR_PAIR(CP_LABEL));

    wrefresh(win);
}

// ================================================================
// Helpers
// ================================================================
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

static bool try_parse_coord(const std::string& s, int n, int& out_r, int& out_c) {
    if (s.size() < 2) return false;
    char col_ch = toupper(s[0]);
    const char* p = strchr(GO_COLS, col_ch);
    if (!p) return false;
    int c = (int)(p - GO_COLS);
    if (c >= n) return false;
    int row_num = 0;
    for (size_t i = 1; i < s.size(); i++) {
        if (!isdigit(s[i])) return false;
        row_num = row_num * 10 + (s[i] - '0');
    }
    if (row_num < 1 || row_num > n) return false;
    out_r = n - row_num;
    out_c = c;
    return true;
}

// ================================================================
// Main
// ================================================================
int main(int argc, char* argv[]) {
    Config config;
    std::string model_path  = "models/best.onnx";
    bool use_random         = false;
    int  board_override     = -1;
    int  search_threads     = 16;
    int  nn_server_threads  = 1;
    int  pvs                = 5;
    std::string nn_device_ids_str = "0";

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if      (arg == "--model"             && i+1<argc) model_path      = argv[++i];
        else if (arg == "--board"             && i+1<argc) board_override  = std::stoi(argv[++i]);
        else if (arg == "--sims"              && i+1<argc) config.num_simulations = std::stoi(argv[++i]);
        else if (arg == "--search-threads"    && i+1<argc) search_threads  = std::stoi(argv[++i]);
        else if (arg == "--komi"              && i+1<argc) config.komi     = std::stof(argv[++i]);
        else if (arg == "--win-loss-weight"    && i+1<argc) config.win_loss_weight = std::stof(argv[++i]);
        else if (arg == "--score-weight"      && i+1<argc) config.score_weight = std::stof(argv[++i]);
        else if (arg == "--score-scale"       && i+1<argc) config.score_scale = std::stof(argv[++i]);
        else if (arg == "--max-batch"         && i+1<argc) config.max_batch_size = std::stoi(argv[++i]);
        else if (arg == "--c-puct"            && i+1<argc) config.c_puct   = std::stof(argv[++i]);
        else if (arg == "--nn-server-threads" && i+1<argc) nn_server_threads = std::stoi(argv[++i]);
        else if (arg == "--nn-device-ids"     && i+1<argc) nn_device_ids_str = argv[++i];
        else if (arg == "--pvs"              && i+1<argc) pvs             = std::stoi(argv[++i]);
        else if (arg == "--random") use_random = true;
        else if (arg == "--help" || arg == "-h") {
            printf("Usage: play [options]\n"
                   "  --model PATH           Model file (default: models/best.onnx)\n"
                   "  --board N              Board size (--random mode)\n"
                   "  --sims N               MCTS simulations (default: 800)\n"
                   "  --search-threads N     MCTS search threads (default: 16)\n"
                   "  --max-batch N          Max GPU batch size (default: 256)\n"
                   "  --komi F               Komi value (default: 6.5)\n"
                   "  --win-loss-weight F    Win/loss utility weight (default: 1.0)\n"
                   "  --score-weight F       Score utility weight (default: 0.0)\n"
                   "  --score-scale F        Score atan compression scale (default: 10.0)\n"
                   "  --c-puct F             UCB exploration constant (default: 1.5)\n"
                   "  --nn-server-threads N  NN server threads (default: 1)\n"
                   "  --nn-device-ids IDS    Comma-separated GPU indices (default: \"0\")\n"
                   "  --pvs N                Top K moves to show in analysis (default: 5)\n"
                   "  --random               Random bot (no model needed)\n"
                   "  --help                 This help\n");
            return 0;
        }
        else {
            fprintf(stderr, "Error: unrecognized option '%s'\n"
                            "Try 'play --help' for usage.\n", argv[i]);
            return 1;
        }
    }

    // Load model (before ncurses init so errors print normally)
    std::vector<int> device_ids = parse_device_ids(nn_device_ids_str);
    std::shared_ptr<LoadedModel> model;
    std::shared_ptr<ComputeContext> context;
    std::shared_ptr<NNEvaluator> evaluator;
    std::unique_ptr<AsyncBot> bot;

    if (!use_random) {
        try {
            model   = LoadedModel::load(model_path);
            context = std::shared_ptr<ComputeContext>(create_compute_context(device_ids));
            config.model_type = model->model_type;
            config.board_size = model->board_size;
            config.input_channels = model->input_channels;
            config.num_filters = model->num_filters;
            config.num_res_blocks = model->num_res_blocks;
            config.vit_depth = model->vit_depth;
            config.vit_heads = model->vit_heads;
            config.vit_kv_groups = model->vit_kv_groups;
            config.max_moves_per_game = config.board_size * config.board_size * 2;
            config.num_search_threads = search_threads;
            evaluator = std::make_shared<NNEvaluator>(
                model, context, device_ids, config.max_batch_size);
            printf("Waiting for GPU engines ...\n");
            fflush(stdout);
            evaluator->wait_ready();
            bot = std::make_unique<AsyncBot>(evaluator.get(), config);
        } catch (const std::exception& e) {
            fprintf(stderr, "Error: %s\nUse --random for random bot.\n", e.what());
            return 1;
        }
    } else {
        if (board_override > 0) config.board_size = board_override;
        config.max_moves_per_game = config.board_size * config.board_size * 2;
    }

    int n = config.board_size;

    // ── ncurses init ────────────────────────────────────────
    // No stdout redirect — background threads (TRT engine build)
    // may print to the terminal.  Press 'r' or Ctrl-L to redraw.
    setlocale(LC_ALL, "");
    initscr();
    init_colors();
    cbreak();
    noecho();
    keypad(stdscr, TRUE);
    curs_set(0);

    bool keep_playing = true;

    while (keep_playing) {
        // Choose mode
        werase(stdscr);
        attron(COLOR_PAIR(CP_ACCENT) | A_BOLD);
        mvaddstr(2, 4, "MiniGo");
        attroff(COLOR_PAIR(CP_ACCENT) | A_BOLD);
        mvaddstr(4, 4, "Play as (B)lack, (W)hite, or (H)uman vs Human? [B]: ");
        refresh();
        curs_set(1);
        echo();
        char choice_buf[8] = {};
        getnstr(choice_buf, 4);
        noecho();
        curs_set(0);

        bool human_vs_human = (choice_buf[0] == 'H' || choice_buf[0] == 'h');
        Stone human_color = BLACK;
        if (choice_buf[0] == 'W' || choice_buf[0] == 'w')
            human_color = WHITE;

        GoGame game(config.board_size, config.komi);
        if (bot) bot->reset(game);

        int cursor_r = n / 2, cursor_c = n / 2;
        bool cursor_active = false;
        int last_r = -1, last_c = -1;
        std::string input_buf;
        std::string status_msg;
        std::string ai_info;

        // Two independent user preferences with a coupling rule:
        //   P (pondering_wanted): background MCTS search runs during idle time
        //   A (analysis_wanted):  callback fires + HUD displays live eval
        //
        // Coupling: A requires P.  Turning A on auto-turns P on; turning P
        // off auto-turns A off.  Turning A off leaves P as-is (silent ponder
        // remains a valid state).
        //
        // Reachable states: {off, ponder, analyze}.  Hotkeys 'a' and 'P'
        // (uppercase) cycle between them — see toggle_analysis / toggle_ponder
        // below.
        bool pondering_wanted = false;
        bool analysis_wanted  = false;
        bool show_ownership   = false;

        // Shared state between the bot's callback thread and the UI thread.
        std::mutex ai_info_mutex;
        MCTS::AnalysisInfo latest_info;
        bool have_info = false;

        auto analyze_callback = [&](const MCTS::AnalysisInfo& info) {
            std::lock_guard<std::mutex> lock(ai_info_mutex);
            latest_info = info;
            have_info = (info.total_visits > 0);
        };

        // Apply the two flags to the bot.  Stops any running search,
        // reconfigures the callback, then restarts ponder if wanted.
        // Called whenever the user toggles a flag.
        auto apply_bot_state = [&]() {
            if (!bot) return;

            // Stop the current search so the new callback config takes
            // effect on the next search (the worker snapshots callback_
            // when picking up a request).
            bot->stop();

            if (analysis_wanted) {
                bot->set_callback(analyze_callback, /*interval_ms=*/100, pvs);
            } else {
                bot->clear_callback();
                std::lock_guard<std::mutex> lock(ai_info_mutex);
                have_info = false;
                ai_info.clear();
            }

            if (pondering_wanted) {
                bot->start_ponder();
            }
        };

        // Toggle ponder: off ↔ on.  If turning off and analysis was on,
        // also turn off analysis (coupling: A requires P).
        auto toggle_ponder = [&]() {
            if (pondering_wanted) {
                pondering_wanted = false;
                analysis_wanted  = false;   // A requires P — no orphaned HUD
            } else {
                pondering_wanted = true;
            }
            apply_bot_state();
        };

        // Toggle analysis: off ↔ on.  If turning on and ponder was off,
        // also turn on ponder (coupling: A requires P).
        auto toggle_analysis = [&]() {
            if (analysis_wanted) {
                analysis_wanted = false;
                // pondering_wanted stays as-is — user may want silent ponder
            } else {
                analysis_wanted  = true;
                pondering_wanted = true;    // A requires P — auto-enable search
            }
            apply_bot_state();
        };

        // Disable both (used at game-over / quit cleanup).
        auto disable_all = [&]() {
            pondering_wanted = false;
            analysis_wanted  = false;
            apply_bot_state();
        };

        // Helper: check if current player is human
        auto is_human_turn = [&]() {
            return human_vs_human || game.current_player == human_color;
        };

        // Ensure the bot's running state matches the user flags.  Called at
        // the top of the game loop on each iteration.  After gen_move or
        // play_move, the bot is idle; if the user wants ponder, restart it.
        // Only starts ponder during human idle time — on AI turn, gen_move
        // handles its own search through the same worker.
        auto maintain_bot_state = [&]() {
            if (!bot) return;
            if (!is_human_turn()) return;   // AI turn: gen_move drives search
            if (pondering_wanted && !bot->is_searching()) {
                bot->start_ponder();
            }
        };

        // Refresh ai_info for display.  Shows:
        //   analyze mode (P+A): live HUD from latest callback snapshot
        //   ponder mode (P only): "[pondering]" placeholder
        //   off: cleared
        auto refresh_ai_info = [&]() {
            if (!analysis_wanted) {
                ai_info = pondering_wanted ? "[pondering — press a for HUD]" : "";
                return;
            }
            MCTS::AnalysisInfo info;
            {
                std::lock_guard<std::mutex> lock(ai_info_mutex);
                if (!have_info) {
                    ai_info = "[analyzing — waiting for first result]";
                    return;
                }
                info = latest_info;
            }
            char buf[256];
            float wr = (info.root_utility + 1.0f) / 2.0f * 100.0f;
            snprintf(buf, sizeof(buf), "WR %.1f%%  Score %+.1f \xc2\xb1 %.1f  N=%d",
                     wr, info.root_score, info.root_score_sd, info.total_visits);
            ai_info = buf;
            for (auto& m : info.moves) {
                std::string ms = (m.action == config.action_size() - 1)
                    ? "PASS" : game.action_to_str(m.action);
                float mwr = (m.utility + 1.0f) / 2.0f * 100.0f;
                snprintf(buf, sizeof(buf), "| %-3s %5.1f%% n=%-5d", ms.c_str(), mwr, m.visits);
                ai_info += buf;
            }
        };

        while (true) {
            // Ensure ponder is running if analysis is toggled on.  The bot
            // may be idle because gen_move/play_move just finished or
            // because analysis was just toggled on — restart it here so
            // the callback keeps firing.
            maintain_bot_state();

            // Set getch timeout: 500ms while HUD is active (so we can poll
            // the latest callback snapshot and redraw), blocking otherwise.
            timeout(analysis_wanted ? 100 : -1);

            refresh_ai_info();

            draw_board(stdscr, game, config, cursor_r, cursor_c, cursor_active,
                       last_r, last_c, human_color, use_random,
                       status_msg, ai_info, input_buf);

            // Ownership overlay: draw +/- at empty intersections
            if (show_ownership) {
                MCTS::AnalysisInfo info;
                {
                    std::lock_guard<std::mutex> lock(ai_info_mutex);
                    info = latest_info;
                }
                int bsz = game.board_size;
                if ((int)info.root_ownership.size() == bsz * bsz) {
                    int ox = 5, oy = 3, cell_w = 2;
                    for (int r = 0; r < bsz; r++) {
                        for (int c = 0; c < bsz; c++) {
                            if (game.board[r][c] != EMPTY) continue;
                            float own = info.root_ownership[r * bsz + c];
                            if (own > 0.6f) {
                                wattron(stdscr, COLOR_PAIR(CP_BLACK_STONE));
                                mvwaddch(stdscr, oy + r, ox + c * cell_w, '+');
                                wattroff(stdscr, COLOR_PAIR(CP_BLACK_STONE));
                            } else if (own < 0.4f) {
                                wattron(stdscr, COLOR_PAIR(CP_WHITE_STONE));
                                mvwaddch(stdscr, oy + r, ox + c * cell_w, '-');
                                wattroff(stdscr, COLOR_PAIR(CP_WHITE_STONE));
                            }
                        }
                    }
                    wrefresh(stdscr);
                }
            }

            if (game.game_over) {
                disable_all();
                auto [bs, ws] = game.score();
                char buf[128];
                snprintf(buf, sizeof(buf), "Game over  B:%.1f  W:%.1f  [r]restart [q]quit", bs, ws);
                status_msg = buf;
                draw_board(stdscr, game, config, -1, -1, false,
                           last_r, last_c, human_color, use_random,
                           status_msg, ai_info, "");
                timeout(-1);
                int key = getch();
                if (key == 'q' || key == 'Q') { keep_playing = false; break; }
                if (key == 'r' || key == 'R') break;
                continue;
            }

            if (!is_human_turn()) {
                // AI turn.
                status_msg = "AI thinking...";

                int action;
                if (use_random) {
                    draw_board(stdscr, game, config, -1, -1, false,
                               last_r, last_c, human_color, use_random,
                               status_msg, ai_info, "");
                    action = random_legal_move(game);
                    if (action == config.action_size() - 1)
                        game.play(PASS_MOVE);
                    else
                        game.play(action);
                } else {
                    // Run gen_move async so the main thread can redraw
                    // live search stats from the analysis callback.
                    auto future = std::async(std::launch::async, [&]() {
                        return bot->gen_move(game.current_player, -1, 0.0f, false);
                    });
                    while (future.wait_for(std::chrono::milliseconds(100))
                           != std::future_status::ready) {
                        refresh_ai_info();
                        draw_board(stdscr, game, config, -1, -1, false,
                                   last_r, last_c, human_color, use_random,
                                   status_msg, ai_info, "");
                    }
                    action = future.get();
                    game = bot->game();
                }

                if (action == config.action_size() - 1) {
                    last_r = last_c = -1;
                    status_msg = "AI plays: PASS";
                } else {
                    int r = action / n, c = action % n;
                    last_r = r; last_c = c;
                    char buf[32];
                    snprintf(buf, sizeof(buf), "AI plays: %s", game.action_to_str(action).c_str());
                    status_msg = buf;
                }
                continue;
            }

            // ── Human turn — read input ─────────────────────
            int key = getch();
            if (key == ERR) continue;  // timeout during analysis polling
            status_msg.clear();

            if (key == 'q' || key == 'Q') { disable_all(); keep_playing = false; break; }

            // Refresh screen (r or Ctrl-L)
            if (key == 'r' || key == 12) {
                clearok(stdscr, TRUE);
                continue;
            }

            // Toggle analysis HUD ('a').  A requires P — turning A on also
            // turns on pondering; turning A off leaves pondering as-is.
            if (key == 'a') {
                toggle_analysis();
                continue;
            }

            // Toggle ownership overlay ('o').
            if (key == 'o') {
                show_ownership = !show_ownership;
                continue;
            }

            // Toggle pondering (p/P).
            // Turning P off auto-turns A off (no orphaned HUD).
            if (key == 'p' || key == 'P') {
                toggle_ponder();
                continue;
            }

            // Helper: play a human move through the bot (advances game+tree).
            // bot->play_move stops ponder internally (if running); the top
            // of the next loop iteration auto-restarts it via maintain_bot_state.
            auto play_human_move = [&](int action) -> bool {
                if (bot) {
                    if (!bot->play_move(game.current_player, action))
                        return false;
                    game = bot->game();
                } else {
                    game.play(action == (config.action_size() - 1) ? PASS_MOVE : action);
                }
                return true;
            };

            // Movement (arrows + WASD, but not 'a' which is analysis toggle)
            if (key == KEY_UP    || key == 'w' || key == 'W') { cursor_r = std::max(0, cursor_r - 1); input_buf.clear(); cursor_active = true; }
            else if (key == KEY_DOWN  || key == 's' || key == 'S') { cursor_r = std::min(n-1, cursor_r + 1); input_buf.clear(); cursor_active = true; }
            else if (key == KEY_LEFT) { cursor_c = std::max(0, cursor_c - 1); input_buf.clear(); cursor_active = true; }
            else if (key == KEY_RIGHT || key == 'd' || key == 'D') { cursor_c = std::min(n-1, cursor_c + 1); input_buf.clear(); cursor_active = true; }

            // Place stone
            else if (key == 10 || key == 13 || key == ' ') {
                int r = -1, c = -1;
                if (!input_buf.empty() && try_parse_coord(input_buf, n, r, c)) {
                    /* typed */
                } else if (cursor_active) {
                    r = cursor_r; c = cursor_c;
                }
                if (r >= 0) {
                    int action = r * n + c;
                    if (game.is_legal(action)) {
                        play_human_move(action);
                        last_r = r; last_c = c;
                        input_buf.clear();
                        cursor_active = false;
                    } else {
                        status_msg = "Illegal move!";
                    }
                }
            }

            // Pass (Ctrl-P)
            else if (key == 16) {
                play_human_move(config.action_size() - 1);
                last_r = last_c = -1;
                input_buf.clear();
            }

            // Typed coordinate
            else if (isalpha(key) || isdigit(key)) {
                cursor_active = false;
                if (input_buf.size() < 3)
                    input_buf += (char)toupper(key);
                int r, c;
                if (try_parse_coord(input_buf, n, r, c)) {
                    int action = r * n + c;
                    if (game.is_legal(action)) {
                        cursor_r = r; cursor_c = c; cursor_active = true;
                        draw_board(stdscr, game, config, cursor_r, cursor_c, true,
                                   last_r, last_c, human_color, use_random,
                                   status_msg, ai_info, input_buf);
                        napms(120);
                        play_human_move(action);
                        last_r = r; last_c = c;
                        input_buf.clear();
                        cursor_active = false;
                    } else {
                        status_msg = "Illegal move!";
                        input_buf.clear();
                    }
                }
            }

            // Backspace
            else if (key == KEY_BACKSPACE || key == 127 || key == 8) {
                if (!input_buf.empty()) input_buf.pop_back();
            }
        }

        disable_all();
    }

    endwin();
    printf("Thanks for playing!\n");
    return 0;
}
