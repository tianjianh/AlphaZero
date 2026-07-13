// KataGo V7 input encoder.
//
// Port of cpp/neuralnet/nninputs.cpp::fillRowV7 from KataGo (commit @
// the time of this writing; lightvector/KataGo). Reads a MiniGo `GoGame`
// position and produces the 22-spatial + 19-global float vector that
// kata1's network was trained against.
//
// Coverage notes (deliberate simplifications relative to upstream):
//   * Plane 8 is unused in V7 (always 0).
//   * Planes 14-17 (ladder features) are computed with a faithful port
//     of upstream's bounded ladder search (see the "Ladder solver"
//     section below).  The only deviation is deterministic row-major
//     chain traversal, which can reorder move lists — observable solely
//     when the 25k-node budget truncates a search.
//   * Planes 20-21 (second-encore start colors) are zeroed. We don't
//     support the encore.
//   * Globals 12-14 (encore phase / pass-would-end) are zeroed except
//     pass-would-end which is approximated from consecutive_passes.
//   * Globals 16-18 are zeroed (unused in V7 per upstream).
//
// Default rules applied (the only mode supported here):
//   - area scoring                     → global[9]  = 0
//   - simple ko                        → global[6,7] = 0, 0
//   - multi-stone suicide disallowed   → global[8]  = 0
//   - tax: TAX_NONE                    → global[10,11] = 0, 0
//   - no encore                        → global[12,13] = 0, 0

#include "katago_inputs.h"
#include "ladder.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>

namespace minigo {

namespace {

constexpr int NUM_SPATIAL = 22;
constexpr int NUM_GLOBAL  = 19;

// Liberty count for the group containing (r, c). Returns 0 if (r, c) is
// empty. Uses a stack-allocated visited buffer (no heap allocation in
// the hot path).
int group_liberty_count(const GoGame& game, int r, int c) {
    int n = game.board_size;
    Stone color = game.board[r][c];
    if (color == EMPTY) return 0;

    bool visited_stone[MAX_BOARD * MAX_BOARD] = {};
    bool visited_lib  [MAX_BOARD * MAX_BOARD] = {};
    int stack_r[MAX_BOARD * MAX_BOARD];
    int stack_c[MAX_BOARD * MAX_BOARD];
    int top = 0;
    stack_r[top] = r; stack_c[top] = c; top++;
    visited_stone[r * MAX_BOARD + c] = true;
    int libs = 0;

    while (top > 0) {
        --top;
        int pr = stack_r[top], pc = stack_c[top];
        const int dr[4] = {-1, 1, 0, 0};
        const int dc[4] = { 0, 0,-1, 1};
        for (int k = 0; k < 4; ++k) {
            int nr = pr + dr[k], nc = pc + dc[k];
            if (nr < 0 || nr >= n || nc < 0 || nc >= n) continue;
            int idx = nr * MAX_BOARD + nc;
            Stone s = game.board[nr][nc];
            if (s == EMPTY) {
                if (!visited_lib[idx]) { visited_lib[idx] = true; ++libs; }
            } else if (s == color && !visited_stone[idx]) {
                visited_stone[idx] = true;
                stack_r[top] = nr; stack_c[top] = nc; ++top;
            }
        }
    }
    return libs;
}

// Tromp-Taylor area: each cell maps to BLACK / WHITE / EMPTY. Stones
// keep their color; empty regions go to the surrounding color if
// surrounded by exactly one color, else stay EMPTY (dame).
//
// Approximation note: upstream uses Board::calculateArea which respects
// pass-alive groups and "safe big territories" — slightly different
// from plain Tromp-Taylor on disputed positions. For the typical
// in-game inference (most positions are clear), the discrepancy is
// small.
void compute_area(const GoGame& game, std::array<Stone, MAX_BOARD * MAX_BOARD>& area) {
    int n = game.board_size;
    for (int r = 0; r < n; ++r)
        for (int c = 0; c < n; ++c)
            area[r * MAX_BOARD + c] = game.board[r][c];

    bool visited[MAX_BOARD * MAX_BOARD] = {};
    int stack_r[MAX_BOARD * MAX_BOARD];
    int stack_c[MAX_BOARD * MAX_BOARD];

    for (int r0 = 0; r0 < n; ++r0) {
        for (int c0 = 0; c0 < n; ++c0) {
            if (game.board[r0][c0] != EMPTY) continue;
            if (visited[r0 * MAX_BOARD + c0]) continue;

            int top = 0;
            stack_r[top] = r0; stack_c[top] = c0; ++top;
            visited[r0 * MAX_BOARD + c0] = true;

            int region_size = 0;
            int region_r[MAX_BOARD * MAX_BOARD];
            int region_c[MAX_BOARD * MAX_BOARD];
            bool touches_black = false, touches_white = false;

            while (top > 0) {
                --top;
                int pr = stack_r[top], pc = stack_c[top];
                region_r[region_size] = pr;
                region_c[region_size] = pc;
                ++region_size;

                const int dr[4] = {-1, 1, 0, 0};
                const int dc[4] = { 0, 0,-1, 1};
                for (int k = 0; k < 4; ++k) {
                    int nr = pr + dr[k], nc = pc + dc[k];
                    if (nr < 0 || nr >= n || nc < 0 || nc >= n) continue;
                    Stone s = game.board[nr][nc];
                    if (s == EMPTY) {
                        int idx = nr * MAX_BOARD + nc;
                        if (!visited[idx]) {
                            visited[idx] = true;
                            stack_r[top] = nr; stack_c[top] = nc; ++top;
                        }
                    } else if (s == BLACK) touches_black = true;
                    else if (s == WHITE) touches_white = true;
                }
            }

            Stone owner = EMPTY;
            if (touches_black && !touches_white) owner = BLACK;
            else if (touches_white && !touches_black) owner = WHITE;
            for (int i = 0; i < region_size; ++i)
                area[region_r[i] * MAX_BOARD + region_c[i]] = owner;
        }
    }
}


// Index in the flat output for a given spatial plane and (r, c).
inline int sp_idx(int plane, int r, int c, int H, int W) {
    return plane * (H * W) + r * W + c;
}

}  // namespace

// ────────────────────────────────────────────────────────────
//  Main encoder
// ────────────────────────────────────────────────────────────

void encode_for_katago(const GoGame& game,
                       const LoadedModel* model,
                       std::vector<float>& out) {
    if (model->input_channels != NUM_SPATIAL ||
        model->input_global_channels != NUM_GLOBAL) {
        throw std::runtime_error(
            "encode_for_katago: model expected 22 spatial + 19 global "
            "channels (got " + std::to_string(model->input_channels) + " + "
            + std::to_string(model->input_global_channels) + ")");
    }

    const int H = game.board_size;
    const int W = H;
    const int HW = H * W;
    const Stone pla = game.current_player;
    const Stone opp = opponent(pla);

    out.assign((size_t)NUM_SPATIAL * HW + NUM_GLOBAL, 0.0f);
    float* sp = out.data();
    float* gl = out.data() + (size_t)NUM_SPATIAL * HW;

    // ── Plane 0: on-board (always 1 within H×W) ──────────────
    for (int pos = 0; pos < HW; ++pos)
        sp[sp_idx(0, 0, 0, H, W) + pos] = 1.0f;

    // ── Planes 1, 2, 3, 4, 5: stone presence + liberty count ─
    for (int r = 0; r < H; ++r) {
        for (int c = 0; c < W; ++c) {
            Stone s = game.board[r][c];
            if (s == EMPTY) continue;
            if (s == pla) sp[sp_idx(1, r, c, H, W)] = 1.0f;
            else          sp[sp_idx(2, r, c, H, W)] = 1.0f;
            int libs = group_liberty_count(game, r, c);
            if      (libs == 1) sp[sp_idx(3, r, c, H, W)] = 1.0f;
            else if (libs == 2) sp[sp_idx(4, r, c, H, W)] = 1.0f;
            else if (libs == 3) sp[sp_idx(5, r, c, H, W)] = 1.0f;
        }
    }

    // ── Plane 6: ko-banned point (simple ko) ─────────────────
    // After a single capture (one stone), the recapture point is the
    // captured stone's old location. is_ko_ban encapsulates the test.
    for (int r = 0; r < H; ++r) {
        for (int c = 0; c < W; ++c) {
            if (game.board[r][c] != EMPTY) continue;
            int action = r * W + c;
            if (game.is_ko_ban(action))
                sp[sp_idx(6, r, c, H, W)] = 1.0f;
        }
    }

    // Plane 7: ko-recap-blocked (encore-only) — always 0.
    // Plane 8: unused — always 0.

    // ── Planes 9-13: one-hot location of past N moves ──────
    // Upstream maps:
    //   plane 9  = opponent's previous move    (1 ply ago)
    //   plane 10 = player's move 2 plies ago
    //   plane 11 = opponent's move 3 plies ago
    //   plane 12 = player's move 4 plies ago
    //   plane 13 = opponent's move 5 plies ago
    // For pass moves at that ply, set rowGlobal[i] = 1.0 (i = plies-1).
    for (int i = 0; i < 5; ++i) {
        int act = game.recent_action(i);  // i=0 is the most recent
        if (act == -2) continue;          // no move that far back
        int plane = 9 + i;
        if (act == PASS_MOVE) {
            gl[i] = 1.0f;
        } else {
            int r = act / W, c = act % W;
            if (r >= 0 && r < H && c >= 0 && c < W)
                sp[sp_idx(plane, r, c, H, W)] = 1.0f;
        }
    }

    // ── Planes 14-17: ladder features ────────────────────────
    // Plane 14: stones of chains laddered on the current board.
    // Plane 17: attacker's working first moves against laddered 2-lib
    //           opponent chains.
    // Planes 15/16: laddered stones on the boards 1 / 2 plies ago
    //           (with those positions' own simple-ko points).
    {
        LadderBoard lb;
        lb.init(game.recent_board(0), H, game.recent_ko_point(0));
        fill_ladder_planes(lb, sp, H, W, 14, 17, (int8_t)opp);
        lb.init(game.recent_board(1), H, game.recent_ko_point(1));
        fill_ladder_planes(lb, sp, H, W, 15, -1, EMPTY);
        lb.init(game.recent_board(2), H, game.recent_ko_point(2));
        fill_ladder_planes(lb, sp, H, W, 16, -1, EMPTY);
    }

    // ── Planes 18-19: Tromp-Taylor area scoring ────────────
    {
        std::array<Stone, MAX_BOARD * MAX_BOARD> area;
        compute_area(game, area);
        for (int r = 0; r < H; ++r) {
            for (int c = 0; c < W; ++c) {
                Stone owner = area[r * MAX_BOARD + c];
                if (owner == pla) sp[sp_idx(18, r, c, H, W)] = 1.0f;
                else if (owner == opp) sp[sp_idx(19, r, c, H, W)] = 1.0f;
            }
        }
    }

    // Planes 20-21: second-encore start colors — zero (no encore).

    // ── Globals ──────────────────────────────────────────────
    // gl[0..4]: pass flags for the last 5 moves. Set above for any
    // recent_action == PASS_MOVE.

    // gl[5]: komi normalized by 20, from the perspective of the side
    // to move. Black plays first in 9×9 with komi for white, so when
    // current_player == BLACK the "self komi" is -komi (white has the
    // komi advantage); when current_player == WHITE it's +komi.
    {
        float self_komi = (pla == WHITE) ? game.komi : -game.komi;
        // Clamp to ±(boardArea + 1) per upstream guard.
        float bound = (float)(HW + 1);
        if (self_komi >  bound) self_komi =  bound;
        if (self_komi < -bound) self_komi = -bound;
        gl[5] = self_komi / 20.0f;
    }

    // gl[6,7]: ko rule. Tromp-Taylor uses simple ko → both 0.
    // gl[8]:  multi-stone suicide allowed → 0 (TT disallows).
    // gl[9]:  scoring rule. 0 = area, 1 = territory → area = 0.
    // gl[10,11]: tax rule TAX_NONE → 0, 0.
    // gl[12,13]: encore phase 0 → 0, 0.

    // gl[14]: pass-would-end-phase. In normal play (no encore), a pass
    // ends the game when the previous move was also a pass.
    if (game.consecutive_passes >= 1) gl[14] = 1.0f;

    // gl[15,16]: playoutDoublingAdvantage (unused → 0,0).
    // gl[17]:    button rule (unused → 0).

    // gl[18]: komi parity wave — triangular wave with period 2 in
    // selfKomi units.  Port of upstream fillRowV7 (nninputs.cpp:2681-
    // 2713): the komi floor is anchored on the parity of komi values
    // that can produce draws, which is the parity of the BOARD AREA
    // (drawableKomisAreEven = (xSize*ySize) % 2 == 0) — NOT the parity
    // of floor(selfKomi).  On 9x9 (area 81, odd) with komi 6.5 the
    // correct wave is +0.5 for black / -0.5 for white; anchoring on
    // floor(selfKomi) parity produced the sign-flipped values.
    {
        float self_komi = (pla == WHITE) ? game.komi : -game.komi;
        bool board_area_is_even = (HW % 2) == 0;
        bool drawable_komis_are_even = board_area_is_even;

        float komi_floor;
        if (drawable_komis_are_even)
            komi_floor = std::floor(self_komi / 2.0f) * 2.0f;
        else
            komi_floor = std::floor((self_komi - 1.0f) / 2.0f) * 2.0f + 1.0f;

        float delta = self_komi - komi_floor;   // in [0, 2)
        if (delta < 0.0f) delta = 0.0f;
        if (delta > 2.0f) delta = 2.0f;

        float wave;
        if      (delta < 0.5f) wave = delta;
        else if (delta < 1.5f) wave = 1.0f - delta;
        else                   wave = delta - 2.0f;
        gl[18] = wave;
    }
}

}  // namespace minigo
