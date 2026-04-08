#include "config.h"
#include "game.h"
#include "mcts.h"
#include "loaded_model.h"
#include "compute_context.h"
#include "nn_evaluator.h"
#include <algorithm>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <random>
#include <string>
#include <cstdio>

#ifdef _WIN32
#include <conio.h>
#else
#include <termios.h>
#include <unistd.h>
#include <sys/select.h>
#endif

using namespace minigo;

// ================================================================
// Terminal utilities — raw mode, ANSI escape codes
// ================================================================

#ifndef _WIN32
static termios g_orig_termios;
static bool g_raw_mode = false;

static void enable_raw_mode() {
    if (g_raw_mode) return;
    tcgetattr(STDIN_FILENO, &g_orig_termios);
    termios raw = g_orig_termios;
    raw.c_lflag &= ~(ECHO | ICANON);
    raw.c_cc[VMIN] = 0;
    raw.c_cc[VTIME] = 1;  // 100ms timeout for reads
    tcsetattr(STDIN_FILENO, TCSAFLUSH, &raw);
    g_raw_mode = true;
}

static void disable_raw_mode() {
    if (!g_raw_mode) return;
    tcsetattr(STDIN_FILENO, TCSAFLUSH, &g_orig_termios);
    g_raw_mode = false;
}
#endif

// ANSI escape helpers
static void clear_screen() { printf("\033[2J\033[H"); }
static void move_cursor(int row, int col) { printf("\033[%d;%dH", row, col); }
static void hide_cursor() { printf("\033[?25l"); }
static void show_cursor() { printf("\033[?25h"); }

// Colors
#define C_RESET   "\033[0m"
#define C_BOLD    "\033[1m"
#define C_DIM     "\033[2m"
#define C_BLACK   "\033[30m"
#define C_RED     "\033[31m"
#define C_GREEN   "\033[32m"
#define C_YELLOW  "\033[33m"
#define C_BLUE    "\033[34m"
#define C_CYAN    "\033[36m"
#define C_WHITE   "\033[37m"
#define C_BG_BOARD "\033[48;5;180m"  // warm tan/wood color
#define C_BG_RESET "\033[49m"

// Key codes
enum { KEY_NONE = 0, KEY_UP = 256, KEY_DOWN, KEY_LEFT, KEY_RIGHT, KEY_ENTER = '\r', KEY_ESC = 27 };

static int read_key() {
#ifdef _WIN32
    if (!_kbhit()) return KEY_NONE;
    int c = _getch();
    if (c == 0 || c == 224) { c = _getch(); switch(c) { case 72: return KEY_UP; case 80: return KEY_DOWN; case 75: return KEY_LEFT; case 77: return KEY_RIGHT; } }
    return c;
#else
    char c;
    if (read(STDIN_FILENO, &c, 1) <= 0) return KEY_NONE;
    if (c == '\033') {
        char seq[2];
        if (read(STDIN_FILENO, &seq[0], 1) <= 0) return KEY_ESC;
        if (read(STDIN_FILENO, &seq[1], 1) <= 0) return KEY_ESC;
        if (seq[0] == '[') {
            switch (seq[1]) {
                case 'A': return KEY_UP;
                case 'B': return KEY_DOWN;
                case 'C': return KEY_RIGHT;
                case 'D': return KEY_LEFT;
            }
        }
        return KEY_ESC;
    }
    if (c == '\n' || c == '\r') return KEY_ENTER;
    return (unsigned char)c;
#endif
}

static const char* COLS = "ABCDEFGHJKLMNOPQRSTUVWXYZ";

// ================================================================
// Board rendering — Unicode with ANSI colors
// ================================================================

static bool is_star_point(int r, int c, int n) {
    if (n == 9) {
        return (r == 2 || r == 4 || r == 6) && (c == 2 || c == 4 || c == 6);
    } else if (n == 19) {
        return (r == 3 || r == 9 || r == 15) && (c == 3 || c == 9 || c == 15);
    } else if (n == 13) {
        return (r == 3 || r == 6 || r == 9) && (c == 3 || c == 6 || c == 9);
    }
    return false;
}

// Render the full board to a string buffer (ANSI colored)
static std::string render_board(const GoGame& game, int cursor_r = -1, int cursor_c = -1,
                                 int last_r = -1, int last_c = -1,
                                 const std::string& input_buf = "") {
    int n = game.board_size;
    std::string out;
    auto add = [&](const char* s) { out += s; };

    // Column header
    add("    ");
    for (int c = 0; c < n; c++) { char buf[8]; snprintf(buf, sizeof(buf), " %c", COLS[c]); out += buf; }
    add("\n");

    for (int r = 0; r < n; r++) {
        int row_num = n - r;
        char rowbuf[8];
        snprintf(rowbuf, sizeof(rowbuf), " %2d ", row_num);
        out += rowbuf;

        for (int c = 0; c < n; c++) {
            bool is_cursor = (r == cursor_r && c == cursor_c);

            if (game.board[r][c] == BLACK) {
                if (r == last_r && c == last_c)
                    add(C_RED C_BOLD);  // last move highlight
                else
                    add(C_BOLD);
                add(is_cursor ? "[#]" : " #");
                add(C_RESET);
            } else if (game.board[r][c] == WHITE) {
                if (r == last_r && c == last_c)
                    add(C_RED C_BOLD);
                else
                    add(C_BOLD C_WHITE);
                add(is_cursor ? "[O]" : " O");
                add(C_RESET);
            } else {
                // Empty intersection
                if (is_cursor) {
                    // Show candidate stone
                    const char* sym = (game.current_player == BLACK) ? "#" : "O";
                    add(C_DIM);
                    out += "["; out += sym; out += "]";
                    add(C_RESET);
                } else if (is_star_point(r, c, n)) {
                    add(C_YELLOW " +" C_RESET);
                } else {
                    add(C_DIM " ." C_RESET);
                }
            }
        }

        snprintf(rowbuf, sizeof(rowbuf), " %d", row_num);
        out += rowbuf;
        out += "\n";
    }

    // Column footer
    add("    ");
    for (int c = 0; c < n; c++) { char buf[8]; snprintf(buf, sizeof(buf), " %c", COLS[c]); out += buf; }
    add("\n");

    return out;
}

// ================================================================
// Full screen draw
// ================================================================

static void draw_screen(const GoGame& game, const Config& config,
                         Stone human_color, bool use_random,
                         int cursor_r, int cursor_c,
                         int last_r, int last_c,
                         const std::string& input_buf,
                         const std::string& status_line,
                         const std::string& ai_info) {
    move_cursor(1, 1);

    Stone ai_color = (human_color == BLACK) ? WHITE : BLACK;
    int n = game.board_size;

    printf(C_BOLD C_CYAN "  MINIGO" C_RESET " — %dx%d  Komi: %.1f  Sims: %d\n",
           n, n, config.komi, config.num_simulations);
    printf("  You: %s    AI: %s    Move: %d\n\n",
           human_color == BLACK ? C_BOLD "#" C_RESET " Black" : C_BOLD C_WHITE "O" C_RESET " White",
           ai_color == BLACK ? C_BOLD "#" C_RESET " Black" : C_BOLD C_WHITE "O" C_RESET " White",
           game.move_count + 1);

    std::string board = render_board(game, cursor_r, cursor_c, last_r, last_c, input_buf);
    printf("%s\n", board.c_str());

    if (!ai_info.empty())
        printf("  %s\n", ai_info.c_str());

    // Input area
    if (!status_line.empty()) {
        printf("  %s\n", status_line.c_str());
    }

    if (game.current_player == human_color && !game.game_over) {
        printf("  Move: " C_BOLD "%s" C_RESET "█", input_buf.c_str());
        printf("   " C_DIM "(arrows/type A1-J9, P=pass, Q=quit)" C_RESET);
    }

    // Clear remaining lines
    printf("\033[J");
    fflush(stdout);
}

// ================================================================
// Parse partial coordinate input
// ================================================================

static bool try_parse_coord(const std::string& s, int board_size, int& out_r, int& out_c) {
    if (s.size() < 2) return false;
    char col_ch = toupper(s[0]);
    const char* p = strchr(COLS, col_ch);
    if (!p) return false;
    int c = (int)(p - COLS);
    if (c >= board_size) return false;
    int row_num = 0;
    for (size_t i = 1; i < s.size(); i++) {
        if (!isdigit(s[i])) return false;
        row_num = row_num * 10 + (s[i] - '0');
    }
    if (row_num < 1 || row_num > board_size) return false;
    out_r = board_size - row_num;
    out_c = c;
    return true;
}

// ================================================================
// Main
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
            std::cout << "Usage: play [options]\n"
                      << "  --model PATH            Model file (default: models/best.onnx)\n"
                      << "  --board N               Board size (for --random mode)\n"
                      << "  --sims N                MCTS simulations (default: 800)\n"
                      << "  --search-threads N      MCTS search threads (default: 16)\n"
                      << "  --max-batch N           Max GPU batch size (default: 256)\n"
                      << "  --c-puct F              UCB exploration constant (default: 1.5)\n"
                      << "  --komi F                Komi value (default: 6.5)\n"
                      << "  --score-weight F        Score utility weight (default: 0.0)\n"
                      << "  --score-scale F         Score atan compression scale (default: 10.0)\n"
                      << "  --nn-server-threads N   NN server threads (default: 1)\n"
                      << "  --nn-device-ids IDS     Comma-separated device indices (default: \"0\")\n"
                      << "  --random                Use random bot (no model needed)\n";
            return 0;
        }
    }

    std::vector<int> device_ids = parse_device_ids(nn_device_ids_str);
    if ((int)device_ids.size() != nn_server_threads) {
        std::cerr << "ERROR: --nn-device-ids count must match --nn-server-threads\n";
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
            config.model_type         = model->model_type;
            config.board_size         = model->board_size;
            config.input_channels     = model->input_channels;
            config.num_filters        = model->num_filters;
            config.num_res_blocks     = model->num_res_blocks;
            config.vit_depth          = model->vit_depth;
            config.vit_heads          = model->vit_heads;
            config.vit_kv_groups      = model->vit_kv_groups;
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

    // ── Game loop ────────────────────────────────────────────
    bool keep_playing = true;
    while (keep_playing) {
        // Choose color (line-buffered for this prompt)
        clear_screen();
        printf(C_BOLD C_CYAN "\n  MINIGO" C_RESET " — Human vs AI\n\n");
        printf("  Play as (B)lack or (W)hite? [B]: ");
        fflush(stdout);

        std::string choice;
        std::getline(std::cin, choice);
        Stone human_color = BLACK;
        if (!choice.empty() && (choice[0] == 'W' || choice[0] == 'w'))
            human_color = WHITE;
        Stone ai_color = (human_color == BLACK) ? WHITE : BLACK;

        GoGame game(config.board_size, config.komi);
        int n = config.board_size;
        int cursor_r = n / 2, cursor_c = n / 2;  // start cursor at center
        int last_r = -1, last_c = -1;  // last move position
        std::string input_buf;
        std::string status_line;
        std::string ai_info;
        bool cursor_active = false;

        // Enter raw mode for game
        clear_screen();
#ifndef _WIN32
        enable_raw_mode();
#endif
        hide_cursor();

        while (!game.game_over) {
            // Redraw
            int show_cr = cursor_active ? cursor_r : -1;
            int show_cc = cursor_active ? cursor_c : -1;
            draw_screen(game, config, human_color, use_random,
                        show_cr, show_cc, last_r, last_c,
                        input_buf, status_line, ai_info);

            if (game.current_player == human_color) {
                // ── Human turn: read input ──────────────────
                int key = read_key();
                if (key == KEY_NONE) continue;

                status_line.clear();

                if (key == 'q' || key == 'Q') {
                    show_cursor();
                    disable_raw_mode();
                    clear_screen();
                    printf("Goodbye!\n");
                    return 0;
                }

                if (key == 'p' || key == 'P') {
                    game.play(PASS_MOVE);
                    last_r = last_c = -1;
                    input_buf.clear();
                    cursor_active = false;
                    ai_info.clear();
                    continue;
                }

                // Arrow keys — activate and move cursor
                if (key == KEY_UP || key == KEY_DOWN || key == KEY_LEFT || key == KEY_RIGHT) {
                    cursor_active = true;
                    input_buf.clear();
                    if (key == KEY_UP && cursor_r > 0) cursor_r--;
                    if (key == KEY_DOWN && cursor_r < n - 1) cursor_r++;
                    if (key == KEY_LEFT && cursor_c > 0) cursor_c--;
                    if (key == KEY_RIGHT && cursor_c < n - 1) cursor_c++;
                    continue;
                }

                // Enter — confirm cursor position or typed input
                if (key == KEY_ENTER) {
                    int r = -1, c = -1;
                    if (!input_buf.empty() && try_parse_coord(input_buf, n, r, c)) {
                        // Use typed coordinate
                    } else if (cursor_active) {
                        r = cursor_r; c = cursor_c;
                    } else {
                        continue;  // nothing to confirm
                    }
                    int action = r * n + c;
                    if (game.is_legal(action)) {
                        game.play(action);
                        last_r = r; last_c = c;
                        input_buf.clear();
                        cursor_active = false;
                        ai_info.clear();
                    } else {
                        status_line = C_RED "Illegal move!" C_RESET;
                    }
                    continue;
                }

                // Backspace
                if (key == 127 || key == 8) {
                    if (!input_buf.empty()) input_buf.pop_back();
                    cursor_active = false;
                    continue;
                }

                // Letter/digit — build coordinate string
                if (isalpha(key) || isdigit(key)) {
                    cursor_active = false;
                    if (input_buf.size() < 3)
                        input_buf += (char)toupper(key);

                    // Check if input is a complete legal move
                    int r, c;
                    if (try_parse_coord(input_buf, n, r, c)) {
                        int action = r * n + c;
                        if (game.is_legal(action)) {
                            // Show candidate briefly, auto-confirm
                            cursor_r = r; cursor_c = c;
                            cursor_active = true;
                            draw_screen(game, config, human_color, use_random,
                                        cursor_r, cursor_c, last_r, last_c,
                                        input_buf, status_line, ai_info);
                            // Small delay to show candidate
                            usleep(150000);

                            game.play(action);
                            last_r = r; last_c = c;
                            input_buf.clear();
                            cursor_active = false;
                            ai_info.clear();
                        } else {
                            status_line = C_RED "Illegal move!" C_RESET;
                            input_buf.clear();
                        }
                    }
                    continue;
                }

            } else {
                // ── AI turn ─────────────────────────────────
                status_line = C_YELLOW "AI thinking..." C_RESET;
                draw_screen(game, config, human_color, use_random,
                            -1, -1, last_r, last_c,
                            input_buf, status_line, ai_info);

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
                    snprintf(buf, sizeof(buf),
                             C_CYAN "WR=%.1f%%  score=%+.1fpts  visits=%d" C_RESET,
                             wr, info.root_score, info.total_visits);
                    ai_info = buf;
                    // Add alternatives
                    for (int i = 1; i < (int)info.moves.size(); i++) {
                        auto& m = info.moves[i];
                        std::string ms = (m.action == config.action_size() - 1)
                            ? "PASS" : game.action_to_str(m.action);
                        float mwr = (m.utility + 1.0f) / 2.0f * 100.0f;
                        snprintf(buf, sizeof(buf), "  alt: %s WR=%.1f%% n=%d",
                                 ms.c_str(), mwr, m.visits);
                        ai_info += buf;
                    }
                }

                if (action == config.action_size() - 1) {
                    game.play(PASS_MOVE);
                    last_r = last_c = -1;
                    status_line = C_GREEN "AI plays: PASS" C_RESET;
                } else {
                    int r = action / n, c = action % n;
                    game.play(action);
                    last_r = r; last_c = c;
                    char buf[64];
                    snprintf(buf, sizeof(buf), C_GREEN "AI plays: %s" C_RESET,
                             game.action_to_str(action).c_str());
                    status_line = buf;
                }
            }
        }

        // ── Game over ────────────────────────────────────────
        draw_screen(game, config, human_color, use_random,
                    -1, -1, last_r, last_c, "", "", ai_info);

        auto [bs, ws] = game.score();
        char result_buf[128];
        const char* winner_str;
        if      (game.winner == human_color) winner_str = C_GREEN C_BOLD "You win!" C_RESET;
        else if (game.winner == EMPTY)       winner_str = C_YELLOW "Draw!" C_RESET;
        else                                 winner_str = C_RED C_BOLD "AI wins!" C_RESET;

        snprintf(result_buf, sizeof(result_buf),
                 "\n  " C_BOLD "GAME OVER" C_RESET "  Black: %.1f  White: %.1f (incl. %.1f komi)\n  %s\n",
                 bs, ws, config.komi, winner_str);
        printf("%s", result_buf);

        // Restore terminal for the prompt
        show_cursor();
#ifndef _WIN32
        disable_raw_mode();
#endif
        printf("\n  Play again? [Y/n]: ");
        fflush(stdout);
        std::string again;
        std::getline(std::cin, again);
        keep_playing = again.empty() || again[0] == 'Y' || again[0] == 'y';
    }

    clear_screen();
    printf("Thanks for playing!\n");
    return 0;
}
