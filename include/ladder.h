#pragma once

// KataGo V7 ladder solver — see src/ladder.cpp for the algorithm notes
// (faithful port of upstream KataGo's bounded ladder search).  Split
// into its own translation unit so it can double as the standalone
// shared library `libminigo_ladder.so`, which scripts/gamedata.py
// ctypes-loads to avoid a slow pure-Python mirror: ONE implementation
// serves both the C++ encoder and the trainer.

#include "game.h"

#include <cstdint>
#include <cstring>

namespace minigo {

constexpr int LADDER_NODE_BUDGET = 25000;   // upstream MAX_LADDER_SEARCH_NODE_BUDGET

struct LadderBoard {
    int n = 0, nn = 0;
    int8_t color[MAX_BOARD * MAX_BOARD];
    int ko_loc = -1;                        // flat index or -1

    // Generation-stamped scratch: no clearing between flood fills.
    uint32_t gen = 0;
    uint32_t stamp[MAX_BOARD * MAX_BOARD] = {};

    void init(const int8_t* src, int board_n, int ko) {
        n = board_n;
        nn = n * n;
        std::memcpy(color, src, (size_t)nn);
        ko_loc = ko;
    }
};

// For every chain with 1-2 liberties, decide once whether it is
// laddered; mark all its stones on plane `plane_stones` of `sp`
// (planes are HxW float maps at sp[plane * H * W]).  When
// `plane_working` >= 0 (current-board pass), additionally mark the
// attacker's working first moves for laddered 2-lib chains of color
// `opp_color` (upstream feature 17).
void fill_ladder_planes(LadderBoard& b, float* sp, int H, int W,
                        int plane_stones, int plane_working, int8_t opp_color);

}  // namespace minigo

// ── C ABI for ctypes (scripts/gamedata.py) ─────────────────────
// Fills `laddered_out[n*n]` (0/1) and, when `working_out` is non-null,
// `working_out[n*n]` with the attacker's working first moves against
// laddered 2-liberty chains of color `opp_color` (1=BLACK, 2=WHITE;
// pass 0 with working_out=null for the prev-board passes).
extern "C" void minigo_ladder_fill(const int8_t* board, int32_t n,
                                   int32_t ko_loc, int32_t opp_color,
                                   uint8_t* laddered_out,
                                   uint8_t* working_out);
