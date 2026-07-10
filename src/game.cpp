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
    recent_actions_.fill(-2);
    recent_actions_count_ = 0;
    update_history();
}

int GoGame::recent_action(int steps_back) const {
    if (steps_back < 0 || steps_back >= RECENT_ACTIONS_CAP) return -2;
    if (steps_back >= recent_actions_count_) return -2;
    return recent_actions_[steps_back];
}

bool GoGame::is_ko_ban(int action) const {
    int n = board_size;
    if (!has_prev_board) return false;
    if (action < 0 || action == PASS_MOVE || action >= n * n) return false;
    int r = action / n, c = action % n;
    if (board[r][c] != EMPTY) return false;
    // Simulate playing here on a copy of the current board (without
    // ko constraints) and compare to prev_board. If they match, this
    // is the simple-ko location. We reuse is_legal_at_slow's structure
    // but exclude the suicide/ko outcome — call play() on a copy and
    // compare.
    Stone test[MAX_BOARD][MAX_BOARD];
    std::memcpy(test, board, sizeof(board));
    test[r][c] = current_player;
    Stone opp = opponent(current_player);
    auto try_capture = [&](int nr, int nc) {
        if (test[nr][nc] != opp) return;
        Pos grp[MAX_BOARD * MAX_BOARD]; int libs;
        int gsize = get_group_on(test, nr, nc, grp, libs);
        if (libs == 0)
            for (int i = 0; i < gsize; i++) test[grp[i].r][grp[i].c] = EMPTY;
    };
    if (r > 0)     try_capture(r - 1, c);
    if (r < n - 1) try_capture(r + 1, c);
    if (c > 0)     try_capture(r, c - 1);
    if (c < n - 1) try_capture(r, c + 1);

    // Suicide check
    Pos own_grp[MAX_BOARD * MAX_BOARD]; int own_libs;
    get_group_on(test, r, c, own_grp, own_libs);
    if (own_libs == 0) return false;  // suicide, not ko

    return std::memcmp(test, prev_board, sizeof(board)) == 0;
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
    g.final_black_score = final_black_score;
    // Ring buffer copy: all inline storage — no heap allocation
    g.ring_buf_  = ring_buf_;
    g.ring_head_ = ring_head_;
    g.ring_size_ = ring_size_;
    g.recent_actions_       = recent_actions_;
    g.recent_actions_count_ = recent_actions_count_;
    return g;
}

void GoGame::neighbors(int r, int c, Pos* nbrs, int& count) const {
    count = 0;
    if (r > 0) nbrs[count++] = {r - 1, c};
    if (r < board_size - 1) nbrs[count++] = {r + 1, c};
    if (c > 0) nbrs[count++] = {r, c - 1};
    if (c < board_size - 1) nbrs[count++] = {r, c + 1};
}

int GoGame::get_group(int r, int c, Pos* out_group, int& liberties) const {
    return get_group_on(board, r, c, out_group, liberties);
}

int GoGame::get_group_on(const Stone brd[][MAX_BOARD], int r, int c,
                          Pos* out_group, int& liberties) const {
    Stone color = brd[r][c];
    if (color == EMPTY) { liberties = 0; return 0; }

    int n = board_size;
    int group_size = 0;
    liberties = 0;

    // Stack-based BFS with fixed-size scratch — no heap allocation in
    // the hot path.  visited[]/lib_seen[] are 361-byte stack arrays
    // that the compiler zero-inits as a single short memset.
    bool visited[MAX_BOARD * MAX_BOARD] = {};
    bool lib_seen[MAX_BOARD * MAX_BOARD] = {};
    Pos  stack_buf[MAX_BOARD * MAX_BOARD];
    int  top = 0;

    stack_buf[top++] = {r, c};
    visited[r * MAX_BOARD + c] = true;

    auto visit = [&](int nr, int nc) {
        int idx = nr * MAX_BOARD + nc;
        Stone s = brd[nr][nc];
        if (s == EMPTY) {
            if (!lib_seen[idx]) { lib_seen[idx] = true; liberties++; }
        } else if (s == color && !visited[idx]) {
            visited[idx] = true;
            stack_buf[top++] = {nr, nc};
        }
    };

    while (top > 0) {
        Pos p = stack_buf[--top];
        out_group[group_size++] = p;
        int pr = p.r, pc = p.c;
        if (pr > 0)         visit(pr - 1, pc);
        if (pr < n - 1)     visit(pr + 1, pc);
        if (pc > 0)         visit(pr,     pc - 1);
        if (pc < n - 1)     visit(pr,     pc + 1);
    }
    return group_size;
}

void GoGame::remove_group(const Pos* group, int n) {
    for (int i = 0; i < n; i++) board[group[i].r][group[i].c] = EMPTY;
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
    // Applies to the vast majority of moves; eliminates board-clone +
    // BFS for ~80-90% of is_legal() calls.  Inlined as four explicit
    // boundary-checked loads — no Pos struct, no function call.
    if ((r > 0     && board[r - 1][c] == EMPTY) ||
        (r < n - 1 && board[r + 1][c] == EMPTY) ||
        (c > 0     && board[r][c - 1] == EMPTY) ||
        (c < n - 1 && board[r][c + 1] == EMPTY))
        return true;

    return is_legal_at_slow(r, c);
}

bool GoGame::is_legal_at_slow(int r, int c) const {
    // All four neighbours of (r,c) are non-empty (caller guarantees the
    // fast-path miss).  This handles eye-fills, snapbacks, and ko — the
    // rare case that needs a full capture+ko simulation.
    int n = board_size;
    Stone test[MAX_BOARD][MAX_BOARD];
    std::memcpy(test, board, sizeof(board));
    test[r][c] = current_player;
    Stone opp = opponent(current_player);

    auto try_capture = [&](int nr, int nc) {
        if (test[nr][nc] != opp) return;
        Pos grp[MAX_BOARD * MAX_BOARD]; int libs;
        int gsize = get_group_on(test, nr, nc, grp, libs);
        if (libs == 0)
            for (int i = 0; i < gsize; i++) test[grp[i].r][grp[i].c] = EMPTY;
    };

    if (r > 0)     try_capture(r - 1, c);
    if (r < n - 1) try_capture(r + 1, c);
    if (c > 0)     try_capture(r, c - 1);
    if (c < n - 1) try_capture(r, c + 1);

    // Suicide check
    Pos own_grp[MAX_BOARD * MAX_BOARD]; int own_libs;
    get_group_on(test, r, c, own_grp, own_libs);
    if (own_libs == 0) return false;

    // Ko check via single memcmp instead of nested r,c loop.
    if (has_prev_board && std::memcmp(test, prev_board, sizeof(board)) == 0)
        return false;

    return true;
}

void GoGame::get_legal_moves(std::vector<float>& legal) const {
    int n = board_size;
    int action_size = n * n + 1;
    legal.assign(action_size, 0.0f);
    legal[n * n] = 1.0f;  // pass always legal
    if (game_over) return;

    // Inline the fast path here — avoids 81 function calls per move and
    // the per-call game_over / pass / bounds rechecks.  is_legal() above
    // remains the canonical entry point for external callers.
    for (int r = 0; r < n; r++) {
        for (int c = 0; c < n; c++) {
            if (board[r][c] != EMPTY) continue;
            bool fast =
                (r > 0     && board[r - 1][c] == EMPTY) ||
                (r < n - 1 && board[r + 1][c] == EMPTY) ||
                (c > 0     && board[r][c - 1] == EMPTY) ||
                (c < n - 1 && board[r][c + 1] == EMPTY);
            if (fast || is_legal_at_slow(r, c))
                legal[r * n + c] = 1.0f;
        }
    }
}

void GoGame::play(int action) {
    int n = board_size;

    // Normalize pass
    if (action == n * n) action = PASS_MOVE;

    // Record the (normalized) action so the KataGo V7 history planes can
    // identify pass plies as PASS_MOVE rather than the pre-normalized
    // n*n slot. Most-recent at index 0; older entries shift toward the tail.
    for (int i = RECENT_ACTIONS_CAP - 1; i > 0; --i)
        recent_actions_[i] = recent_actions_[i - 1];
    recent_actions_[0] = action;
    if (recent_actions_count_ < RECENT_ACTIONS_CAP) ++recent_actions_count_;

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
        Stone opp = opponent(current_player);

        // Remove captured groups — inline neighbor enumeration with
        // stack-array group buffer (no heap allocation in the hot path).
        auto try_capture = [&](int nr, int nc) {
            if (board[nr][nc] != opp) return;
            Pos grp[MAX_BOARD * MAX_BOARD]; int libs;
            int gsize = get_group(nr, nc, grp, libs);
            if (libs == 0) remove_group(grp, gsize);
        };
        if (r > 0)     try_capture(r - 1, c);
        if (r < n - 1) try_capture(r + 1, c);
        if (c > 0)     try_capture(r, c - 1);
        if (c < n - 1) try_capture(r, c + 1);
    }

    last_move = action;
    move_count++;
    current_player = opponent(current_player);
    update_history();
}

void GoGame::update_history() {
    // Write into the next ring slot (overwrites oldest if full).  Drop
    // the prior snap.fill(0) — every used position is overwritten just
    // below, and snap[n*n .. MAX_BOARD*MAX_BOARD) is never read.  Each
    // row is a 1-byte-stride memcpy of `n` bytes; the compiler folds
    // small-n memcpys to register loads/stores.
    int write_idx = (ring_head_ + ring_size_) % RING_CAP;
    auto& snap = ring_buf_[write_idx];
    int n = board_size;
    for (int r = 0; r < n; r++)
        std::memcpy(&snap[r * n], &board[r][0], (size_t)n);

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

void GoGame::get_ownership(Stone player, std::vector<float>& out) const {
    int n = board_size;
    out.assign(n * n, 0.0f);

    // Reuse the same flood-fill logic as score()
    bool visited[MAX_BOARD][MAX_BOARD] = {};
    for (int r = 0; r < n; r++) {
        for (int c = 0; c < n; c++) {
            // Stones: owner is the stone's color
            if (board[r][c] == player) {
                out[r * n + c] = 1.0f;
            } else if (board[r][c] != EMPTY) {
                out[r * n + c] = 0.0f;
            } else if (!visited[r][c]) {
                // Flood-fill empty region
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

                // Assign territory: only if surrounded by one color
                bool owned_by_player = false;
                if (player == BLACK && touches_black && !touches_white)
                    owned_by_player = true;
                if (player == WHITE && touches_white && !touches_black)
                    owned_by_player = true;

                if (owned_by_player) {
                    for (auto& p : region)
                        out[p.r * n + p.c] = 1.0f;
                }
            }
        }
    }
}

void GoGame::score_game() {
    // Tromp-Taylor AREA SCORING of the final position as-is after two
    // passes.  No heuristic dead-stone removal — under Chinese rules the
    // game is played to completion, so any stones still on the board are
    // alive.  For training this provides the correct signal: the network
    // learns to capture dead stones before passing rather than relying on
    // a post-game cleanup (which was also order-dependent and broke semeai
    // / seki positions).
    //
    // NOTE: the KO RULE is simple ko (one-position memory via prev_board),
    // not Tromp-Taylor's positional superko — long cycles (triple ko etc.)
    // are bounded only by max_moves_per_game.  KataGo defaults to
    // positional superko; adding it here would need a position-hash set
    // in play().  See COMPARISON_WITH_KATAGO.md.
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

}  // namespace minigo
