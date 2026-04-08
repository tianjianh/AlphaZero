#include "game.h"
#include <algorithm>
#include <cassert>
#include <cstring>
#include <sstream>
#include <cctype>

namespace minigo {

GoGame::GoGame(int board_size, float komi, int history_length)
    : board_size(board_size), komi(komi), history_length(history_length) {
    reset();
}

void GoGame::reset() {
    std::memset(board, 0, sizeof(board));
    std::memset(prev_board, 0, sizeof(prev_board));
    has_prev_board = false;
    current_player = BLACK;
    move_count = 0;
    last_move = -2;  // sentinel: no move yet
    consecutive_passes = 0;
    game_over = false;
    winner = EMPTY;
    final_black_score = 0.0f;
    ring_head_ = 0;
    ring_size_ = 0;
    update_history();
}

GoGame GoGame::copy() const {
    GoGame g(board_size, komi, history_length);
    std::memcpy(g.board,      board,      sizeof(board));
    std::memcpy(g.prev_board, prev_board, sizeof(prev_board));
    g.has_prev_board  = has_prev_board;
    g.current_player  = current_player;
    g.move_count      = move_count;
    g.last_move       = last_move;
    g.consecutive_passes = consecutive_passes;
    g.game_over       = game_over;
    g.winner          = winner;
    // Ring buffer copy: all inline storage — no heap allocation
    g.ring_buf_  = ring_buf_;
    g.ring_head_ = ring_head_;
    g.ring_size_ = ring_size_;
    return g;
}

void GoGame::neighbors(int r, int c, Pos* nbrs, int& count) const {
    count = 0;
    if (r > 0) nbrs[count++] = {r - 1, c};
    if (r < board_size - 1) nbrs[count++] = {r + 1, c};
    if (c > 0) nbrs[count++] = {r, c - 1};
    if (c < board_size - 1) nbrs[count++] = {r, c + 1};
}

void GoGame::get_group(int r, int c, std::vector<Pos>& group, int& liberties) const {
    get_group_on(board, r, c, group, liberties);
}

void GoGame::get_group_on(const Stone brd[][MAX_BOARD], int r, int c,
                           std::vector<Pos>& group, int& liberties) const {
    Stone color = brd[r][c];
    if (color == EMPTY) { group.clear(); liberties = 0; return; }

    group.clear();
    liberties = 0;

    // BFS with visited array
    bool visited[MAX_BOARD][MAX_BOARD] = {};
    bool liberty_visited[MAX_BOARD][MAX_BOARD] = {};
    std::vector<Pos> stack;
    stack.push_back({r, c});
    visited[r][c] = true;

    while (!stack.empty()) {
        Pos p = stack.back(); stack.pop_back();
        group.push_back(p);

        Pos nbrs[4]; int cnt;
        neighbors(p.r, p.c, nbrs, cnt);
        for (int i = 0; i < cnt; i++) {
            int nr = nbrs[i].r, nc = nbrs[i].c;
            if (brd[nr][nc] == EMPTY && !liberty_visited[nr][nc]) {
                liberty_visited[nr][nc] = true;
                liberties++;
            } else if (brd[nr][nc] == color && !visited[nr][nc]) {
                visited[nr][nc] = true;
                stack.push_back({nr, nc});
            }
        }
    }
}

void GoGame::remove_group(const std::vector<Pos>& group) {
    for (auto& p : group) board[p.r][p.c] = EMPTY;
}

bool GoGame::is_legal(int action) const {
    if (game_over) return false;
    int n = board_size;

    if (action == PASS_MOVE || action == n * n) return true;

    int r = action / n, c = action % n;
    if (r < 0 || r >= n || c < 0 || c >= n) return false;
    if (board[r][c] != EMPTY) return false;

    // ── Fast path: any empty neighbour → definitely legal ────────────
    //
    // • Suicide is impossible: our stone inherits the empty neighbour
    //   as a liberty, so the group can never have 0 liberties.
    //
    // • Ko is impossible: after we play, (r,c) has our stone and the
    //   adjacent empty cell still empty.  For board == prev_board we
    //   would need our stone there in prev_board — but our stone was
    //   captured in the previous move, which requires all liberties
    //   filled (no empty neighbours at capture time).  Contradiction.
    //
    // Applies to the vast majority of moves: any interior/border position
    // with at least one open neighbour.  Eliminates the board-clone + BFS
    // for ~80–90% of is_legal() calls.
    Pos nbrs[4]; int cnt;
    neighbors(r, c, nbrs, cnt);
    for (int i = 0; i < cnt; i++) {
        if (board[nbrs[i].r][nbrs[i].c] == EMPTY)
            return true;   // legal — no suicide, no ko possible
    }

    // ── Slow path: all neighbours are occupied ────────────────────────
    // (happens only in dense/endgame positions; also covers rare eye-fills
    //  and snapbacks that need the full capture+ko simulation)
    Stone test[MAX_BOARD][MAX_BOARD];
    std::memcpy(test, board, sizeof(board));
    test[r][c] = current_player;

    // Simulate captures of opponent groups
    for (int i = 0; i < cnt; i++) {
        int nr = nbrs[i].r, nc = nbrs[i].c;
        if (test[nr][nc] == opponent(current_player)) {
            std::vector<Pos> grp; int libs;
            get_group_on(test, nr, nc, grp, libs);
            if (libs == 0)
                for (auto& p : grp) test[p.r][p.c] = EMPTY;
        }
    }

    // Check suicide
    std::vector<Pos> own_grp; int own_libs;
    get_group_on(test, r, c, own_grp, own_libs);
    if (own_libs == 0) return false;

    // Check ko
    if (has_prev_board) {
        bool same = true;
        for (int rr = 0; rr < n && same; rr++)
            for (int cc = 0; cc < n && same; cc++)
                if (test[rr][cc] != prev_board[rr][cc]) same = false;
        if (same) return false;
    }

    return true;
}

void GoGame::get_legal_moves(std::vector<float>& legal) const {
    int n = board_size;
    int action_size = n * n + 1;
    legal.assign(action_size, 0.0f);
    for (int i = 0; i < n * n; i++) {
        if (is_legal(i)) legal[i] = 1.0f;
    }
    legal[n * n] = 1.0f;  // pass always legal
}

void GoGame::play(int action) {
    int n = board_size;

    // Normalize pass
    if (action == n * n) action = PASS_MOVE;

    std::memcpy(prev_board, board, sizeof(board));
    has_prev_board = true;

    if (action == PASS_MOVE) {
        consecutive_passes++;
        if (consecutive_passes >= 2) {
            game_over = true;
            score_game();
        }
    } else {
        consecutive_passes = 0;
        int r = action / n, c = action % n;
        board[r][c] = current_player;

        // Remove captured groups
        Pos nbrs[4]; int cnt;
        neighbors(r, c, nbrs, cnt);
        for (int i = 0; i < cnt; i++) {
            int nr = nbrs[i].r, nc = nbrs[i].c;
            if (board[nr][nc] == opponent(current_player)) {
                std::vector<Pos> grp; int libs;
                get_group(nr, nc, grp, libs);
                if (libs == 0) remove_group(grp);
            }
        }
    }

    last_move = action;
    move_count++;
    current_player = opponent(current_player);
    update_history();
}

void GoGame::update_history() {
    // Write into the next ring slot (overwrites oldest if full)
    int write_idx = (ring_head_ + ring_size_) % RING_CAP;
    auto& snap = ring_buf_[write_idx];
    snap.fill(0);
    for (int r = 0; r < board_size; r++)
        for (int c = 0; c < board_size; c++)
            snap[r * board_size + c] = static_cast<int8_t>(board[r][c]);

    if (ring_size_ < history_length) {
        ring_size_++;                             // ring not full yet
    } else {
        ring_head_ = (ring_head_ + 1) % RING_CAP; // evict oldest: O(1)
    }
}

void GoGame::encode(std::vector<float>& out) const {
    int n = board_size;
    int nn = n * n;
    int planes = history_length * 2 + 1;
    out.assign(planes * nn, 0.0f);

    // History planes: most-recent first (i=0 → most recent snapshot)
    // ring_buf_[(ring_head_ + ring_size_ - 1 - i) % RING_CAP] gives snapshot i steps back
    for (int i = 0; i < history_length && i < ring_size_; i++) {
        int slot = (ring_head_ + ring_size_ - 1 - i + RING_CAP) % RING_CAP;
        const auto& snap = ring_buf_[slot];
        int cur_plane = i;
        int opp_plane = history_length + i;
        for (int pos = 0; pos < nn; pos++) {
            Stone s = static_cast<Stone>(snap[pos]);
            if (s == current_player)
                out[cur_plane * nn + pos] = 1.0f;
            else if (s == opponent(current_player))
                out[opp_plane * nn + pos] = 1.0f;
        }
    }

    // Color plane (all-ones for BLACK, all-zeros for WHITE)
    if (current_player == BLACK) {
        float* color_plane = out.data() + history_length * 2 * nn;
        std::fill(color_plane, color_plane + nn, 1.0f);
    }
}

std::pair<float, float> GoGame::score() const {
    int n = board_size;
    float black_area = 0, white_area = 0;

    // Count stones
    for (int r = 0; r < n; r++)
        for (int c = 0; c < n; c++) {
            if (board[r][c] == BLACK) black_area++;
            if (board[r][c] == WHITE) white_area++;
        }

    // Flood-fill empty regions
    bool visited[MAX_BOARD][MAX_BOARD] = {};
    for (int r = 0; r < n; r++) {
        for (int c = 0; c < n; c++) {
            if (board[r][c] != EMPTY || visited[r][c]) continue;

            std::vector<Pos> region;
            bool touches_black = false, touches_white = false;
            std::vector<Pos> stack;
            stack.push_back({r, c});
            visited[r][c] = true;

            while (!stack.empty()) {
                Pos p = stack.back(); stack.pop_back();
                region.push_back(p);

                Pos nbrs[4]; int cnt;
                neighbors(p.r, p.c, nbrs, cnt);
                for (int i = 0; i < cnt; i++) {
                    int nr = nbrs[i].r, nc = nbrs[i].c;
                    if (board[nr][nc] == EMPTY && !visited[nr][nc]) {
                        visited[nr][nc] = true;
                        stack.push_back({nr, nc});
                    } else if (board[nr][nc] == BLACK) {
                        touches_black = true;
                    } else if (board[nr][nc] == WHITE) {
                        touches_white = true;
                    }
                }
            }

            if (touches_black && !touches_white)
                black_area += region.size();
            else if (touches_white && !touches_black)
                white_area += region.size();
        }
    }

    return {black_area, white_area + komi};
}

void GoGame::score_game() {
    auto [b, w] = score();
    final_black_score = b - w;
    if (b > w) winner = BLACK;
    else if (w > b) winner = WHITE;
    else winner = EMPTY;
}

std::string GoGame::action_to_str(int action) const {
    int n = board_size;
    if (action == PASS_MOVE || action == n * n) return "PASS";
    int r = action / n, c = action % n;
    const char* cols = "ABCDEFGHJKLMNOPQRSTUVWXYZ";
    return std::string(1, cols[c]) + std::to_string(n - r);
}

int GoGame::str_to_action(const std::string& s) const {
    if (s == "PASS" || s == "pass") return board_size * board_size;
    const char* cols = "ABCDEFGHJKLMNOPQRSTUVWXYZ";
    char col_char = std::toupper(s[0]);
    int c = 0;
    while (cols[c] && cols[c] != col_char) c++;
    int row_num = std::stoi(s.substr(1));
    int r = board_size - row_num;
    return r * board_size + c;
}

std::string GoGame::display() const {
    const char* cols = "ABCDEFGHJKLMNOPQRSTUVWXYZ";
    const char symbols[] = {'.', 'X', 'O'};
    std::ostringstream ss;

    ss << "   ";
    for (int c = 0; c < board_size; c++) ss << cols[c] << ' ';
    ss << '\n';

    for (int r = 0; r < board_size; r++) {
        int row_num = board_size - r;
        if (row_num < 10) ss << ' ';
        ss << row_num << ' ';
        for (int c = 0; c < board_size; c++) {
            ss << symbols[board[r][c]];
            if (c < board_size - 1) ss << ' ';
        }
        ss << ' ';
        if (row_num < 10) ss << ' ';
        ss << row_num << '\n';
    }

    ss << "   ";
    for (int c = 0; c < board_size; c++) ss << cols[c] << ' ';
    ss << '\n';

    ss << "Move " << move_count << " | Turn: "
       << (current_player == BLACK ? "Black(X)" : "White(O)");
    return ss.str();
}

}  // namespace minigo
