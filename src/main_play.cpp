#include "config.h"
#include "game.h"
#include "mcts.h"
#include "loaded_model.h"
#include "compute_context.h"
#include "nn_evaluator.h"
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
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
    if (n == 9)  return (r==2||r==4||r==6) && (c==2||c==4||c==6);
    if (n == 13) return (r==3||r==6||r==9) && (c==3||c==6||c==9);
    if (n == 19) return (r==3||r==9||r==15) && (c==3||c==9||c==15);
    return false;
}

// ================================================================
// Board drawing (ncurses)
// ================================================================
static void draw_board(WINDOW* win, const GoGame& game, const Config& config,
                        int cursor_r, int cursor_c,
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
    // Each cell = 4 chars wide (intersection + 3 hlines), 2 rows tall (grid + vline)
    int ox = 5;   // left margin for row labels
    int oy = 3;
    int cell_w = 4;  // chars per cell horizontally

    // Column labels (spaced by cell_w)
    wattron(win, COLOR_PAIR(CP_LABEL));
    for (int c = 0; c < n; c++)
        mvwaddch(win, oy - 1, ox + c * cell_w, GO_COLS[c]);
    wattroff(win, COLOR_PAIR(CP_LABEL));

    // Board grid with vertical connectors
    // Each board row r occupies 2 screen rows: grid row + vline row
    // except the last row has no vline row below
    int board_h = n * 2 - 1;  // total screen rows for board

    for (int sr = 0; sr < board_h; sr++) {
        bool is_grid_row = (sr % 2 == 0);
        int r = sr / 2;  // board row
        int y = oy + sr;

        if (is_grid_row) {
            // Row label
            char rl[4];
            snprintf(rl, sizeof(rl), "%2d", n - r);
            wattron(win, COLOR_PAIR(CP_LABEL));
            mvwaddstr(win, y, ox - 3, rl);
            wattroff(win, COLOR_PAIR(CP_LABEL));
        }

        for (int c = 0; c < n; c++) {
            int x = ox + c * cell_w;

            if (is_grid_row) {
                // ── Grid row: intersections + horizontal lines ──
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

                bool is_cursor = (r == cursor_r && c == cursor_c);
                bool is_last   = (r == last_r && c == last_c);
                int cell = game.board[r][c];

                if (cell == BLACK) {
                    int attr = COLOR_PAIR(is_last ? CP_RED : CP_BLACK_STONE) | A_BOLD;
                    wattron(win, attr);
                    mvwprintw(win, y, x, "\xe2\x9a\xab");  // ⚫ U+26AB (2-wide)
                    wattroff(win, attr);
                } else if (cell == WHITE) {
                    int attr = COLOR_PAIR(is_last ? CP_RED : CP_WHITE_STONE) | A_BOLD;
                    wattron(win, attr);
                    mvwprintw(win, y, x, "\xe2\x9a\xaa");  // ⚪ U+26AA (2-wide)
                    wattroff(win, attr);
                } else if (is_cursor && !game.game_over) {
                    wattron(win, COLOR_PAIR(CP_CURSOR) | A_BOLD);
                    if (game.current_player == BLACK)
                        mvwprintw(win, y, x, "\xe2\x9a\xab");  // ⚫
                    else
                        mvwprintw(win, y, x, "\xe2\x9a\xaa");  // ⚪
                    wattroff(win, COLOR_PAIR(CP_CURSOR) | A_BOLD);
                } else if (is_star_point(r, c, n)) {
                    wattron(win, COLOR_PAIR(CP_ACCENT));
                    mvwaddch(win, y, x, '*');
                    wattroff(win, COLOR_PAIR(CP_ACCENT));
                } else {
                    wattron(win, COLOR_PAIR(CP_GRID));
                    mvwaddch(win, y, x, grid_ch);
                    wattroff(win, COLOR_PAIR(CP_GRID));
                }

                // Horizontal connector to the right
                if (c < n - 1) {
                    bool wide = (cell != EMPTY || (is_cursor && !game.game_over));
                    int hx = wide ? x + 2 : x + 1;  // wide chars occupy 2 columns
                    wattron(win, COLOR_PAIR(CP_GRID));
                    for (int k = hx; k < x + cell_w; k++)
                        mvwaddch(win, y, k, ACS_HLINE);
                    wattroff(win, COLOR_PAIR(CP_GRID));
                }
            } else {
                // ── Vertical connector row ──
                if (r < n - 1) {
                    wattron(win, COLOR_PAIR(CP_GRID));
                    mvwaddch(win, y, x, ACS_VLINE);
                    wattroff(win, COLOR_PAIR(CP_GRID));
                }
            }
        }

        // Row label on right (grid rows only)
        if (is_grid_row) {
            wattron(win, COLOR_PAIR(CP_LABEL));
            char rl[4];
            snprintf(rl, sizeof(rl), "%d", n - r);
            mvwaddstr(win, y, ox + (n - 1) * cell_w + 3, rl);
            wattroff(win, COLOR_PAIR(CP_LABEL));
        }
    }

    // Column labels bottom
    int bot_y = oy + board_h;
    wattron(win, COLOR_PAIR(CP_LABEL));
    for (int c = 0; c < n; c++)
        mvwaddch(win, bot_y, ox + c * cell_w, GO_COLS[c]);
    wattroff(win, COLOR_PAIR(CP_LABEL));

    // ── Right-side info panel ──
    int panel_x = ox + n * cell_w + 5;  // right of board
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
    int help_y = std::max(oy + board_h + 1, (int)(panel_y + 2));
    const char* help = "Arrows/WASD: move  Enter: place  P: pass  Q: quit";
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
    std::string nn_device_ids_str = "0";

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if      (arg == "--model"             && i+1<argc) model_path      = argv[++i];
        else if (arg == "--board"             && i+1<argc) board_override  = std::stoi(argv[++i]);
        else if (arg == "--sims"              && i+1<argc) config.num_simulations = std::stoi(argv[++i]);
        else if (arg == "--search-threads"    && i+1<argc) search_threads  = std::stoi(argv[++i]);
        else if (arg == "--komi"              && i+1<argc) config.komi     = std::stof(argv[++i]);
        else if (arg == "--score-weight"      && i+1<argc) config.score_weight = std::stof(argv[++i]);
        else if (arg == "--score-scale"       && i+1<argc) config.score_scale = std::stof(argv[++i]);
        else if (arg == "--max-batch"         && i+1<argc) config.max_batch_size = std::stoi(argv[++i]);
        else if (arg == "--c-puct"            && i+1<argc) config.c_puct   = std::stof(argv[++i]);
        else if (arg == "--nn-server-threads" && i+1<argc) nn_server_threads = std::stoi(argv[++i]);
        else if (arg == "--nn-device-ids"     && i+1<argc) nn_device_ids_str = argv[++i];
        else if (arg == "--random") use_random = true;
        else if (arg == "--help") {
            printf("Usage: play [options]\n"
                   "  --model PATH      Model file\n"
                   "  --board N         Board size (--random mode)\n"
                   "  --sims N          MCTS simulations\n"
                   "  --random          Random bot\n"
                   "  --help            This help\n");
            return 0;
        }
    }

    // Load model (before ncurses init so errors print normally)
    std::vector<int> device_ids = parse_device_ids(nn_device_ids_str);
    std::shared_ptr<LoadedModel> model;
    std::shared_ptr<ComputeContext> context;
    std::shared_ptr<NNEvaluator> evaluator;
    std::unique_ptr<MCTS> mcts;

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
            mcts = std::make_unique<MCTS>(evaluator.get(), config);
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
    setlocale(LC_ALL, "");  // enable UTF-8 for wide chars (must be before initscr)
    initscr();
    init_colors();
    cbreak();
    noecho();
    keypad(stdscr, TRUE);
    curs_set(0);

    bool keep_playing = true;

    while (keep_playing) {
        // Choose color
        werase(stdscr);
        attron(COLOR_PAIR(CP_ACCENT) | A_BOLD);
        mvaddstr(2, 4, "MiniGo — Human vs AI");
        attroff(COLOR_PAIR(CP_ACCENT) | A_BOLD);
        mvaddstr(4, 4, "Play as (B)lack or (W)hite? [B]: ");
        refresh();
        curs_set(1);
        echo();
        char choice_buf[8] = {};
        getnstr(choice_buf, 4);
        noecho();
        curs_set(0);

        Stone human_color = BLACK;
        if (choice_buf[0] == 'W' || choice_buf[0] == 'w')
            human_color = WHITE;

        GoGame game(config.board_size, config.komi);
        int cursor_r = n / 2, cursor_c = n / 2;
        int last_r = -1, last_c = -1;
        std::string input_buf;
        std::string status_msg;
        std::string ai_info;
        std::vector<std::pair<int,int>> history;

        while (true) {
            draw_board(stdscr, game, config, cursor_r, cursor_c,
                       last_r, last_c, human_color, use_random,
                       status_msg, ai_info, input_buf);

            if (game.game_over) {
                auto [bs, ws] = game.score();
                char buf[128];
                if (game.winner == human_color)
                    snprintf(buf, sizeof(buf), "You win! B:%.1f W:%.1f  [r]restart [q]quit", bs, ws);
                else if (game.winner == EMPTY)
                    snprintf(buf, sizeof(buf), "Draw! B:%.1f W:%.1f  [r]restart [q]quit", bs, ws);
                else
                    snprintf(buf, sizeof(buf), "AI wins! B:%.1f W:%.1f  [r]restart [q]quit", bs, ws);
                status_msg = buf;
                draw_board(stdscr, game, config, -1, -1,
                           last_r, last_c, human_color, use_random,
                           status_msg, ai_info, "");

                int key = getch();
                if (key == 'q' || key == 'Q') { keep_playing = false; break; }
                if (key == 'r' || key == 'R') break;
                continue;
            }

            if (game.current_player != human_color) {
                // AI turn
                status_msg = "AI thinking...";
                draw_board(stdscr, game, config, -1, -1,
                           last_r, last_c, human_color, use_random,
                           status_msg, ai_info, "");

                int action;
                if (use_random) {
                    action = random_legal_move(game);
                    ai_info.clear();
                } else {
                    std::vector<float> pi;
                    action = mcts->get_action(game, pi, 0.0f, -1, false);
                    auto info = mcts->get_analysis(3);
                    float wr = (info.root_utility + 1.0f) / 2.0f * 100.0f;
                    char buf[256];
                    snprintf(buf, sizeof(buf), "WR=%.1f%%  score=%+.1f  visits=%d",
                             wr, info.root_score, info.total_visits);
                    ai_info = buf;
                    for (int i = 1; i < (int)info.moves.size(); i++) {
                        auto& m = info.moves[i];
                        std::string ms = (m.action == config.action_size() - 1)
                            ? "PASS" : game.action_to_str(m.action);
                        float mwr = (m.utility + 1.0f) / 2.0f * 100.0f;
                        snprintf(buf, sizeof(buf), "  | %s %.0f%% n=%d",
                                 ms.c_str(), mwr, m.visits);
                        ai_info += buf;
                    }
                }

                if (action == config.action_size() - 1) {
                    game.play(PASS_MOVE);
                    last_r = last_c = -1;
                    status_msg = "AI plays: PASS";
                } else {
                    int r = action / n, c = action % n;
                    history.push_back({action, game.current_player});
                    game.play(action);
                    last_r = r; last_c = c;
                    char buf[32];
                    snprintf(buf, sizeof(buf), "AI plays: %s", game.action_to_str(action).c_str());
                    status_msg = buf;
                }
                continue;
            }

            // Human turn — read input
            int key = getch();
            status_msg.clear();

            if (key == 'q' || key == 'Q') { keep_playing = false; break; }

            // Movement
            if (key == KEY_UP    || key == 'w' || key == 'W') { cursor_r = std::max(0, cursor_r - 1); input_buf.clear(); }
            else if (key == KEY_DOWN  || key == 's' || key == 'S') { cursor_r = std::min(n-1, cursor_r + 1); input_buf.clear(); }
            else if (key == KEY_LEFT  || key == 'a' || key == 'A') { cursor_c = std::max(0, cursor_c - 1); input_buf.clear(); }
            else if (key == KEY_RIGHT || key == 'd' || key == 'D') { cursor_c = std::min(n-1, cursor_c + 1); input_buf.clear(); }

            // Place stone (Enter/Space)
            else if (key == 10 || key == 13 || key == ' ') {
                int r = -1, c = -1;
                if (!input_buf.empty() && try_parse_coord(input_buf, n, r, c)) {
                    // typed coordinate
                } else {
                    r = cursor_r; c = cursor_c;
                }
                int action = r * n + c;
                if (r >= 0 && game.is_legal(action)) {
                    history.push_back({action, game.current_player});
                    game.play(action);
                    last_r = r; last_c = c;
                    input_buf.clear();
                    ai_info.clear();
                } else {
                    status_msg = "Illegal move!";
                }
            }

            // Pass
            else if (key == 'p' || key == 'P') {
                game.play(PASS_MOVE);
                last_r = last_c = -1;
                input_buf.clear();
                ai_info.clear();
            }

            // Undo
            else if (key == 'u' || key == 'U') {
                // Undo requires game reset + replay — simplified: just note
                status_msg = "Undo not supported yet";
            }

            // Typed coordinate input
            else if (isalpha(key) || isdigit(key)) {
                if (input_buf.size() < 3)
                    input_buf += (char)toupper(key);
                int r, c;
                if (try_parse_coord(input_buf, n, r, c)) {
                    int action = r * n + c;
                    if (game.is_legal(action)) {
                        cursor_r = r; cursor_c = c;
                        draw_board(stdscr, game, config, cursor_r, cursor_c,
                                   last_r, last_c, human_color, use_random,
                                   status_msg, ai_info, input_buf);
                        napms(120);
                        history.push_back({action, game.current_player});
                        game.play(action);
                        last_r = r; last_c = c;
                        input_buf.clear();
                        ai_info.clear();
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
    }

    endwin();
    printf("Thanks for playing!\n");
    return 0;
}
