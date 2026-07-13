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


// ════════════════════════════════════════════════════════════
//  Ladder solver (planes 14-17)
//
//  Port of upstream KataGo's exact bounded ladder search:
//    Board::searchIsLadderCaptured / searchIsLadderCapturedAttackerFirst2Libs
//    (cpp/game/board.cpp) and iterLadders (cpp/neuralnet/nninputs.cpp),
//  including the 25 000-node budget, the defender/attacker base cases,
//  the double-ko-death rule, the non-adjacent-liberty early-outs and
//  the attack-ordering heuristic.
//
//  Deliberate deviations (documented, shared with scripts/gamedata.py):
//    * Chains are traversed in canonical row-major order (upstream
//      iterates its next_in_chain insertion order).  Move-list ORDER can
//      therefore differ from upstream, which only matters if the node
//      budget truncates a search.  The mirrored Python implementation
//      uses the same canonical order, so the C++/Python encoder pair
//      stays bit-identical — the invariant this repo actually relies on.
//    * No incremental chain bookkeeping: liberties are recomputed by
//      flood fill with a generation-stamped visited array (ladder
//      searches on Go boards touch tiny subgraphs; this is microseconds).
// ════════════════════════════════════════════════════════════

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

// Neighbor offsets in upstream's adj order: up, left, right, down.
inline int ladder_neighbors(const LadderBoard& b, int p, int out[4]) {
    int cnt = 0;
    int r = p / b.n, c = p % b.n;
    if (r > 0)         out[cnt++] = p - b.n;
    if (c > 0)         out[cnt++] = p - 1;
    if (c < b.n - 1)   out[cnt++] = p + 1;
    if (r < b.n - 1)   out[cnt++] = p + b.n;
    return cnt;
}

// Flood the chain containing p (a stone).  Writes the stones into
// `stones` SORTED ascending (canonical order) and returns the count.
inline int ladder_chain(LadderBoard& b, int p, int* stones) {
    const int8_t col = b.color[p];
    const uint32_t g = ++b.gen;
    int cnt = 0;
    stones[cnt++] = p;
    b.stamp[p] = g;
    int head = 0;
    int nbs[4];
    while (head < cnt) {
        int q = stones[head++];
        int m = ladder_neighbors(b, q, nbs);
        for (int i = 0; i < m; i++) {
            int a = nbs[i];
            if (b.color[a] == col && b.stamp[a] != g) {
                b.stamp[a] = g;
                stones[cnt++] = a;
            }
        }
    }
    std::sort(stones, stones + cnt);
    return cnt;
}

// Liberty count of the chain containing p (a stone).  Order-free, so
// it skips ladder_chain's canonical sort (hot path: called from every
// suicide/legality/bound check inside the search).
inline int ladder_chain_libs(LadderBoard& b, int p) {
    const int8_t col = b.color[p];
    const uint32_t gs = ++b.gen;          // stone stamp
    int stones[MAX_BOARD * MAX_BOARD];
    int cnt = 0;
    stones[cnt++] = p;
    b.stamp[p] = gs;
    int head = 0;
    int libs = 0;
    int nbs[4];
    // Liberties stamped with a SECOND generation so the two marks don't
    // collide (stones use gs, liberties use gl).
    const uint32_t glib = ++b.gen;
    while (head < cnt) {
        int q = stones[head++];
        int m = ladder_neighbors(b, q, nbs);
        for (int j = 0; j < m; j++) {
            int a = nbs[j];
            if (b.color[a] == col) {
                if (b.stamp[a] != gs) {
                    b.stamp[a] = gs;
                    stones[cnt++] = a;
                }
            } else if (b.color[a] == EMPTY && b.stamp[a] != glib) {
                b.stamp[a] = glib;
                libs++;
            }
        }
    }
    return libs;
}

// Upstream findLiberties: append the chain's liberties (dedup within
// buf[bufStart..)) to buf at bufIdx; returns how many were appended.
inline int ladder_find_liberties(LadderBoard& b, int p, std::vector<int>& buf,
                                 int bufStart, int bufIdx) {
    int stones[MAX_BOARD * MAX_BOARD];
    int cnt = ladder_chain(b, p, stones);
    int numFound = 0;
    int nbs[4];
    for (int i = 0; i < cnt; i++) {
        int m = ladder_neighbors(b, stones[i], nbs);
        for (int j = 0; j < m; j++) {
            int lib = nbs[j];
            if (b.color[lib] != EMPTY) continue;
            bool dup = false;
            for (int k = bufStart; k < bufIdx + numFound; k++)
                if (buf[k] == lib) { dup = true; break; }
            if (!dup) {
                if ((int)buf.size() <= bufIdx + numFound)
                    buf.resize(buf.size() * 3 / 2 + 64);
                buf[bufIdx + numFound] = lib;
                numFound++;
            }
        }
    }
    return numFound;
}

// Upstream findLibertyGainingCaptures: liberties of adjacent 1-lib
// opponent chains (deduped by chain), appended to buf.
inline int ladder_find_liberty_gaining_captures(LadderBoard& b, int p,
                                                std::vector<int>& buf,
                                                int bufStart, int bufIdx) {
    const int8_t opp = (b.color[p] == BLACK) ? WHITE : BLACK;
    int stones[MAX_BOARD * MAX_BOARD];
    int cnt = ladder_chain(b, p, stones);

    int heads_checked[MAX_BOARD * MAX_BOARD];
    int num_heads = 0;
    int numFound = 0;
    int nbs[4];
    int chain2[MAX_BOARD * MAX_BOARD];
    for (int i = 0; i < cnt; i++) {
        int m = ladder_neighbors(b, stones[i], nbs);
        for (int j = 0; j < m; j++) {
            int a = nbs[j];
            if (b.color[a] != opp) continue;
            int ccnt = ladder_chain(b, a, chain2);
            int head = chain2[0];                 // canonical head: min index
            bool seen = false;
            for (int k = 0; k < num_heads; k++)
                if (heads_checked[k] == head) { seen = true; break; }
            if (seen) continue;
            heads_checked[num_heads++] = head;
            if (ladder_chain_libs(b, a) == 1)
                numFound += ladder_find_liberties(b, a, buf, bufStart,
                                                  bufIdx + numFound);
            (void)ccnt;
        }
    }
    return numFound;
}

// Upstream hasLibertyGainingCaptures.
inline bool ladder_has_liberty_gaining_captures(LadderBoard& b, int p) {
    const int8_t opp = (b.color[p] == BLACK) ? WHITE : BLACK;
    int stones[MAX_BOARD * MAX_BOARD];
    int cnt = ladder_chain(b, p, stones);
    int nbs[4];
    for (int i = 0; i < cnt; i++) {
        int m = ladder_neighbors(b, stones[i], nbs);
        for (int j = 0; j < m; j++)
            if (b.color[nbs[j]] == opp && ladder_chain_libs(b, nbs[j]) == 1)
                return true;
    }
    return false;
}

inline int ladder_immediate_libs(const LadderBoard& b, int p) {
    int nbs[4];
    int m = ladder_neighbors(b, p, nbs);
    int libs = 0;
    for (int i = 0; i < m; i++)
        if (b.color[nbs[i]] == EMPTY) libs++;
    return libs;
}

// Upstream countHeuristicConnectionLibertiesX2 (NB: per adjacent stone,
// no chain dedup — faithfully copied).
inline int ladder_conn_libs_x2(LadderBoard& b, int p, int8_t pla) {
    int nbs[4];
    int m = ladder_neighbors(b, p, nbs);
    int x2 = 0;
    for (int i = 0; i < m; i++) {
        if (b.color[nbs[i]] != pla) continue;
        int libs = ladder_chain_libs(b, nbs[i]);
        if (libs > 1) x2 += libs * 2 - 3;
    }
    return x2;
}

// Upstream getBoundNumLibertiesAfterPlay.
inline void ladder_bound_libs_after_play(LadderBoard& b, int p, int8_t pla,
                                         int& lower, int& upper) {
    const int8_t opp = (pla == BLACK) ? WHITE : BLACK;
    int numImmediate = 0, numCaps = 0, potentialFromCaps = 0;
    int numConn = 0, maxConn = 0;
    int nbs[4];
    int m = ladder_neighbors(b, p, nbs);
    int chain2[MAX_BOARD * MAX_BOARD];
    for (int i = 0; i < m; i++) {
        int a = nbs[i];
        if (b.color[a] == EMPTY) {
            numImmediate++;
        } else if (b.color[a] == opp) {
            if (ladder_chain_libs(b, a) == 1) {
                numCaps++;
                potentialFromCaps += ladder_chain(b, a, chain2);
            }
        } else if (b.color[a] == pla) {
            int connLibs = ladder_chain_libs(b, a) - 1;
            numConn += connLibs;
            if (connLibs > maxConn) maxConn = connLibs;
        }
    }
    lower = numCaps + (maxConn > numImmediate ? maxConn : numImmediate);
    upper = numImmediate + potentialFromCaps + numConn;
}

// Upstream getNumLibertiesAfterPlay (capped at `max`).
inline int ladder_libs_after_play(LadderBoard& b, int p, int8_t pla, int maxv) {
    const int8_t opp = (pla == BLACK) ? WHITE : BLACK;
    int libs[MAX_BOARD * MAX_BOARD];
    int numLibs = 0;
    int capturedHeads[4];
    int numCaptured = 0;
    int nbs[4];
    int m = ladder_neighbors(b, p, nbs);
    int chain2[MAX_BOARD * MAX_BOARD];

    for (int i = 0; i < m; i++) {
        int a = nbs[i];
        if (b.color[a] == EMPTY) {
            libs[numLibs++] = a;
            if (numLibs >= maxv) return maxv;
        } else if (b.color[a] == opp && ladder_chain_libs(b, a) == 1) {
            libs[numLibs++] = a;
            if (numLibs >= maxv) return maxv;
            int ccnt = ladder_chain(b, a, chain2);
            (void)ccnt;
            int head = chain2[0];
            bool found = false;
            for (int j = 0; j < numCaptured; j++)
                if (capturedHeads[j] == head) { found = true; break; }
            if (!found && numCaptured < 4) capturedHeads[numCaptured++] = head;
        }
    }

    auto wouldBeEmpty = [&](int q) {
        if (b.color[q] == EMPTY) return true;
        if (b.color[q] == opp) {
            int cc = ladder_chain(b, q, chain2);
            (void)cc;
            int head = chain2[0];
            for (int j = 0; j < numCaptured; j++)
                if (capturedHeads[j] == head) return true;
        }
        return false;
    };

    int connHeads[4];
    int numConn = 0;
    int chain3[MAX_BOARD * MAX_BOARD];
    for (int i = 0; i < m; i++) {
        int a = nbs[i];
        if (b.color[a] != pla) continue;
        int ccnt = ladder_chain(b, a, chain3);
        int head = chain3[0];
        bool found = false;
        for (int j = 0; j < numConn; j++)
            if (connHeads[j] == head) { found = true; break; }
        if (found) continue;
        if (numConn < 4) connHeads[numConn++] = head;
        int nbs2[4];
        for (int s = 0; s < ccnt; s++) {
            int m2 = ladder_neighbors(b, chain3[s], nbs2);
            for (int k = 0; k < m2; k++) {
                int q = nbs2[k];
                if (q == p || !wouldBeEmpty(q)) continue;
                bool counted = false;
                for (int l = 0; l < numLibs; l++)
                    if (libs[l] == q) { counted = true; break; }
                if (!counted) {
                    libs[numLibs++] = q;
                    if (numLibs >= maxv) return maxv;
                }
            }
        }
    }
    return numLibs;
}

// Upstream wouldBeKoCapture.
inline bool ladder_would_be_ko_capture(LadderBoard& b, int p, int8_t pla) {
    if (b.color[p] != EMPTY) return false;
    const int8_t opp = (pla == BLACK) ? WHITE : BLACK;
    int nbs[4];
    int m = ladder_neighbors(b, p, nbs);
    int capturable = -1;
    // Off-board neighbors count as walls (allowed); on-board must be opp.
    for (int i = 0; i < m; i++) {
        int a = nbs[i];
        if (b.color[a] != opp) return false;
        if (ladder_chain_libs(b, a) == 1) {
            if (capturable != -1) return false;
            capturable = a;
        }
    }
    if (capturable == -1) return false;
    int chain2[MAX_BOARD * MAX_BOARD];
    return ladder_chain(b, capturable, chain2) == 1;
}

// Upstream isSuicide (multi-stone suicide illegal → isIllegalSuicide same).
inline bool ladder_is_suicide(LadderBoard& b, int p, int8_t pla) {
    const int8_t opp = (pla == BLACK) ? WHITE : BLACK;
    int nbs[4];
    int m = ladder_neighbors(b, p, nbs);
    for (int i = 0; i < m; i++) {
        int a = nbs[i];
        if (b.color[a] == EMPTY) return false;
        if (b.color[a] == pla) {
            if (ladder_chain_libs(b, a) > 1) return false;
        } else {
            if (ladder_chain_libs(b, a) == 1) return false;
        }
    }
    return true;
}

inline bool ladder_is_legal(LadderBoard& b, int p, int8_t pla) {
    return b.color[p] == EMPTY && p != b.ko_loc && !ladder_is_suicide(b, p, pla);
}

struct LadderMoveRecord {
    int loc = -1;
    int8_t pla = 0;
    int prev_ko = -1;
    int cap_start = 0, cap_count = 0;
};

// Play a (legal) move; captures recorded in cap_buf for undo.  Ko rule
// mirrors upstream playMoveAssumeLegal: exactly one stone captured AND
// the played stone is a lone stone with exactly one liberty.
inline void ladder_play(LadderBoard& b, int p, int8_t pla,
                        std::vector<int>& cap_buf, LadderMoveRecord& rec) {
    rec.loc = p;
    rec.pla = pla;
    rec.prev_ko = b.ko_loc;
    rec.cap_start = (int)cap_buf.size();

    const int8_t opp = (pla == BLACK) ? WHITE : BLACK;
    b.color[p] = pla;
    int nbs[4];
    int m = ladder_neighbors(b, p, nbs);
    int chain2[MAX_BOARD * MAX_BOARD];
    for (int i = 0; i < m; i++) {
        int a = nbs[i];
        if (b.color[a] != opp) continue;          // already-captured see EMPTY
        if (ladder_chain_libs(b, a) == 0) {
            int cnt = ladder_chain(b, a, chain2);
            for (int j = 0; j < cnt; j++) {
                cap_buf.push_back(chain2[j]);
                b.color[chain2[j]] = EMPTY;
            }
        }
    }
    rec.cap_count = (int)cap_buf.size() - rec.cap_start;

    b.ko_loc = -1;
    if (rec.cap_count == 1) {
        int own[MAX_BOARD * MAX_BOARD];
        if (ladder_chain(b, p, own) == 1 && ladder_chain_libs(b, p) == 1)
            b.ko_loc = cap_buf[rec.cap_start];
    }
}

inline void ladder_undo(LadderBoard& b, const LadderMoveRecord& rec,
                        std::vector<int>& cap_buf) {
    const int8_t opp = (rec.pla == BLACK) ? WHITE : BLACK;
    for (int i = rec.cap_start; i < rec.cap_start + rec.cap_count; i++)
        b.color[cap_buf[i]] = opp;
    cap_buf.resize(rec.cap_start);
    b.color[rec.loc] = EMPTY;
    b.ko_loc = rec.prev_ko;
}

// Per-thread search scratch, reused across every solve in an encode
// (the per-solve vector allocations dominated the encode profile).
struct LadderSearchScratch {
    static constexpr int STACK_CAP = MAX_BOARD * MAX_BOARD * 3 / 2 + 1;
    int moveListStarts[STACK_CAP];
    int moveListLens[STACK_CAP];
    int moveListCur[STACK_CAP];
    LadderMoveRecord records[STACK_CAP];
};

// Upstream Board::searchIsLadderCaptured — iterative alternating search.
// `loc` must be a stone of the defender group.
bool ladder_search_is_captured(LadderBoard& b, int loc, bool defenderFirst,
                               std::vector<int>& buf, std::vector<int>& cap_buf,
                               LadderSearchScratch& ws) {
    if (b.color[loc] != BLACK && b.color[loc] != WHITE) return false;
    {
        int libs = ladder_chain_libs(b, loc);
        if (libs > 2 || (defenderFirst && libs > 1)) return false;
    }

    const int8_t pla = b.color[loc];              // defender
    const int8_t opp = (pla == BLACK) ? WHITE : BLACK;

    // Root: assume all kos work for the defender.
    const int ko_saved = b.ko_loc;
    if (defenderFirst) b.ko_loc = -1;

    const int stackSize = b.n * b.n * 3 / 2 + 1;
    int* moveListStarts = ws.moveListStarts;
    int* moveListLens = ws.moveListLens;
    int* moveListCur = ws.moveListCur;
    LadderMoveRecord* records = ws.records;

    int stackIdx = 0;
    int searchNodeCount = 0;
    moveListCur[0] = -1;
    moveListStarts[0] = 0;
    moveListLens[0] = 0;
    bool returnValue = false;
    bool returnedFromDeeper = false;

    while (true) {
        if (stackIdx <= -1) {
            b.ko_loc = ko_saved;
            return returnValue;
        }
        // Stack limit: consider it captured (upstream behavior).
        if (stackIdx >= stackSize - 1) {
            returnValue = true; returnedFromDeeper = true; stackIdx--;
            continue;
        }
        // Node budget: assume it does not work; undo everything.
        if (searchNodeCount >= LADDER_NODE_BUDGET) {
            stackIdx -= 1;
            while (stackIdx >= 0) {
                ladder_undo(b, records[stackIdx], cap_buf);
                stackIdx -= 1;
            }
            b.ko_loc = ko_saved;
            return false;
        }

        const bool isDefender = (defenderFirst && (stackIdx % 2) == 0) ||
                                (!defenderFirst && (stackIdx % 2) == 1);

        if (moveListCur[stackIdx] == -1) {
            int libs = ladder_chain_libs(b, loc);

            if (!isDefender && libs <= 1) { returnValue = true;  returnedFromDeeper = true; stackIdx--; continue; }
            if (!isDefender && libs >= 3) { returnValue = false; returnedFromDeeper = true; stackIdx--; continue; }
            if (isDefender && libs >= 2)  { returnValue = false; returnedFromDeeper = true; stackIdx--; continue; }
            // Ladders that depend on ko: assume the defender escapes.
            if (isDefender && b.ko_loc != -1) { returnValue = false; returnedFromDeeper = true; stackIdx--; continue; }

            int start = moveListStarts[stackIdx];
            int moveListLen = 0;
            if (isDefender) {
                moveListLen = ladder_find_liberty_gaining_captures(b, loc, buf, start, start);
                moveListLen += ladder_find_liberties(b, loc, buf, start, start + moveListLen);

                int lowerBound, upperBound;
                ladder_bound_libs_after_play(b, buf[start + moveListLen - 1], pla,
                                             lowerBound, upperBound);
                if (lowerBound >= 3)
                { returnValue = false; returnedFromDeeper = true; stackIdx--; continue; }
                if (moveListLen == 1 && upperBound <= 1)
                { returnValue = true;  returnedFromDeeper = true; stackIdx--; continue; }
            } else {
                moveListLen += ladder_find_liberties(b, loc, buf, start, start);
                // Attacker to move with the defender at exactly 2 libs.
                int libs0 = ladder_immediate_libs(b, buf[start]);
                int libs1 = ladder_immediate_libs(b, buf[start + 1]);

                // Double-ko death.
                if (libs0 == 0 && libs1 == 0 &&
                    ladder_would_be_ko_capture(b, buf[start], opp) &&
                    ladder_would_be_ko_capture(b, buf[start + 1], opp)) {
                    if (ladder_libs_after_play(b, buf[start], pla, 3) <= 2 &&
                        ladder_libs_after_play(b, buf[start + 1], pla, 3) <= 2) {
                        if (!ladder_has_liberty_gaining_captures(b, loc))
                        { returnValue = true; returnedFromDeeper = true; stackIdx--; continue; }
                    }
                }

                // Early quitouts when the liberties are not adjacent.
                int r0 = buf[start] / b.n, c0 = buf[start] % b.n;
                int r1 = buf[start + 1] / b.n, c1 = buf[start + 1] % b.n;
                bool adjacent = (std::abs(r0 - r1) + std::abs(c0 - c1)) == 1;
                if (!adjacent) {
                    if (libs0 >= 3 && libs1 >= 3)
                    { returnValue = false; returnedFromDeeper = true; stackIdx--; continue; }
                    else if (libs0 >= 3)
                    { moveListLen = 1; }
                    else if (libs1 >= 3)
                    { buf[start] = buf[start + 1]; moveListLen = 1; }
                }
                // Attack the escape-richer liberty first.
                if (moveListLen > 1) {
                    libs0 = libs0 * 2 + ladder_conn_libs_x2(b, buf[start], pla);
                    libs1 = libs1 * 2 + ladder_conn_libs_x2(b, buf[start + 1], pla);
                    if (libs1 > libs0) {
                        int tmp = buf[start];
                        buf[start] = buf[start + 1];
                        buf[start + 1] = tmp;
                    }
                }
            }
            moveListLens[stackIdx] = moveListLen;
            moveListCur[stackIdx] = 0;
        } else {
            if (returnedFromDeeper)
                ladder_undo(b, records[stackIdx], cap_buf);

            if (isDefender && !returnValue) { returnedFromDeeper = true; stackIdx--; continue; }
            if (!isDefender && returnValue) { returnedFromDeeper = true; stackIdx--; continue; }
            moveListCur[stackIdx]++;
        }

        if (moveListCur[stackIdx] >= moveListLens[stackIdx]) {
            returnValue = isDefender;      // defender out of moves = captured
            returnedFromDeeper = true;
            stackIdx--;
            continue;
        }

        int move = buf[moveListStarts[stackIdx] + moveListCur[stackIdx]];
        int8_t p = isDefender ? pla : opp;

        if (!ladder_is_legal(b, move, p)) {
            returnValue = isDefender;
            returnedFromDeeper = false;
            continue;
        }

        ladder_play(b, move, p, cap_buf, records[stackIdx]);
        searchNodeCount++;

        stackIdx++;
        moveListCur[stackIdx] = -1;
        moveListStarts[stackIdx] = moveListStarts[stackIdx - 1] + moveListLens[stackIdx - 1];
        moveListLens[stackIdx] = 0;
    }
}

// Upstream Board::searchIsLadderCapturedAttackerFirst2Libs.
bool ladder_search_attacker_first_2libs(LadderBoard& b, int loc,
                                        std::vector<int>& buf,
                                        std::vector<int>& cap_buf,
                                        LadderSearchScratch& ws,
                                        int working_moves[2], int& num_working) {
    num_working = 0;
    if (b.color[loc] != BLACK && b.color[loc] != WHITE) return false;
    if (ladder_chain_libs(b, loc) != 2) return false;

    const int8_t pla = b.color[loc];
    const int8_t opp = (pla == BLACK) ? WHITE : BLACK;

    if ((int)buf.size() < 2) buf.resize(64);
    int numLibs = ladder_find_liberties(b, loc, buf, 0, 0);
    (void)numLibs;
    int move0 = buf[0];
    int move1 = buf[1];
    bool move0Works = false, move1Works = false;

    LadderMoveRecord rec;
    if (ladder_is_legal(b, move0, opp)) {
        ladder_play(b, move0, opp, cap_buf, rec);
        move0Works = ladder_search_is_captured(b, loc, true, buf, cap_buf, ws);
        ladder_undo(b, rec, cap_buf);
    }
    if (ladder_is_legal(b, move1, opp)) {
        ladder_play(b, move1, opp, cap_buf, rec);
        move1Works = ladder_search_is_captured(b, loc, true, buf, cap_buf, ws);
        ladder_undo(b, rec, cap_buf);
    }

    if (move0Works || move1Works) {
        if (move0Works) working_moves[num_working++] = move0;
        if (move1Works) working_moves[num_working++] = move1;
        return true;
    }
    return false;
}

// Upstream iterLadders: for every chain with 1-2 liberties, decide
// once whether it is laddered; mark all its stones on `plane_stones`.
// When `plane_working` >= 0 (current-board pass), additionally mark the
// attacker's working first moves for laddered 2-lib chains of color
// `opp_color` (upstream feature 17: colors[loc]==opp && libs > 1).
void fill_ladder_planes(LadderBoard& b, float* sp, int H, int W,
                        int plane_stones, int plane_working, int8_t opp_color) {
    std::vector<int> buf(256), cap_buf;
    cap_buf.reserve(64);
    static thread_local LadderSearchScratch ws;

    // Every chain is flooded exactly ONCE: `scanned` marks all visited
    // stones (any liberty count), so multi-stone chains don't re-flood
    // per stone — the dominant cost on endgame boards.
    std::vector<uint8_t> scanned(b.nn, 0);
    int stones[MAX_BOARD * MAX_BOARD];

    for (int p = 0; p < b.nn; p++) {
        if (b.color[p] != BLACK && b.color[p] != WHITE) continue;
        if (scanned[p]) continue;

        int cnt = ladder_chain(b, p, stones);
        for (int i = 0; i < cnt; i++) scanned[stones[i]] = 1;
        int libs = ladder_chain_libs(b, p);
        if (libs != 1 && libs != 2) continue;

        bool laddered;
        int working[2];
        int num_working = 0;
        if (libs == 1) {
            laddered = ladder_search_is_captured(b, p, true, buf, cap_buf, ws);
        } else {
            laddered = ladder_search_attacker_first_2libs(b, p, buf, cap_buf, ws,
                                                          working, num_working);
        }
        if (!laddered) continue;

        for (int i = 0; i < cnt; i++)
            sp[plane_stones * (H * W) + stones[i]] = 1.0f;
        if (plane_working >= 0 && b.color[p] == opp_color && libs > 1)
            for (int i = 0; i < num_working; i++)
                sp[plane_working * (H * W) + working[i]] = 1.0f;
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
