// XQWL06 engine port.  See include/xqwl_engine.h for API shape.

#include "xqwl_engine.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <random>

namespace xqwl {

namespace {

using Clock = std::chrono::steady_clock;

constexpr int MATE_VALUE     = 10000;
constexpr int BAN_VALUE      = MATE_VALUE - 100;
constexpr int WIN_VALUE      = MATE_VALUE - 200;
constexpr int DRAW_VALUE     = 20;
constexpr int ADVANCED_VALUE = 3;
constexpr int RANDOM_MASK    = 7;
constexpr int NULL_MARGIN    = 400;
constexpr int NULL_DEPTH     = 2;

constexpr int HASH_ALPHA = 1;
constexpr int HASH_BETA  = 2;
constexpr int HASH_PV    = 3;

constexpr int PIECE_KING    = 0;
constexpr int PIECE_ADVISOR = 1;
constexpr int PIECE_BISHOP  = 2;
constexpr int PIECE_KNIGHT  = 3;
constexpr int PIECE_ROOK    = 4;
constexpr int PIECE_CANNON  = 5;
constexpr int PIECE_PAWN    = 6;

constexpr int PHASE_HASH     = 0;
constexpr int PHASE_KILLER_1 = 1;
constexpr int PHASE_KILLER_2 = 2;
constexpr int PHASE_GEN_MOVES= 3;
constexpr int PHASE_REST     = 4;

// Squares that belong to the 9x10 playable board (padded 16x16).
const char ccInBoard[256] = {
  0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
  0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
  0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
  0,0,0,1,1,1,1,1,1,1,1,1,0,0,0,0,
  0,0,0,1,1,1,1,1,1,1,1,1,0,0,0,0,
  0,0,0,1,1,1,1,1,1,1,1,1,0,0,0,0,
  0,0,0,1,1,1,1,1,1,1,1,1,0,0,0,0,
  0,0,0,1,1,1,1,1,1,1,1,1,0,0,0,0,
  0,0,0,1,1,1,1,1,1,1,1,1,0,0,0,0,
  0,0,0,1,1,1,1,1,1,1,1,1,0,0,0,0,
  0,0,0,1,1,1,1,1,1,1,1,1,0,0,0,0,
  0,0,0,1,1,1,1,1,1,1,1,1,0,0,0,0,
  0,0,0,1,1,1,1,1,1,1,1,1,0,0,0,0,
  0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
  0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
  0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
};

const char ccInFort[256] = {
  0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
  0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
  0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
  0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,0,
  0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,0,
  0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,0,
  0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
  0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
  0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
  0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
  0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,0,
  0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,0,
  0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,0,
  0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
  0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
  0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
};

// Piece-movement spans.  Indexed by (dst - src) + 256.
// Nonzero entries (value: piece span kind):
//   offset ±1, ±16         : 1 (king step)
//   offset ±15, ±17        : 2 (advisor diagonal step)
//   offset ±30, ±34         : 3 (bishop 2-diagonal step)
constexpr auto make_ccLegalSpan() {
    std::array<signed char, 512> a{};
    auto set = [&](int off, signed char v) { a[off + 256] = v; };
    set(-1, 1); set(+1, 1); set(-16, 1); set(+16, 1);
    set(-15, 2); set(+15, 2); set(-17, 2); set(+17, 2);
    set(-30, 3); set(+30, 3); set(-34, 3); set(+34, 3);
    return a;
}
constexpr auto ccLegalSpan = make_ccLegalSpan();

// Knight "leg" (hobble) offsets.  Indexed by (dst - src) + 256.
// For each of the 8 knight moves, gives the relative offset of the square
// that blocks the knight if occupied.
//   move ±33, ±31  : hobble ±16 (one square vertically)
//   move ±18, ±14  : hobble ±1  (one square horizontally)
constexpr auto make_ccKnightPin() {
    std::array<signed char, 512> a{};
    auto set = [&](int off, signed char v) { a[off + 256] = v; };
    set(-33, -16); set(-31, -16);
    set(+31, +16); set(+33, +16);
    set(-18, -1);  set(+18, +1);
    set(-14, +1);  set(+14, -1);
    return a;
}
constexpr auto ccKnightPin = make_ccKnightPin();

const char ccKingDelta[4]         = {-16, -1, 1, 16};
const char ccAdvisorDelta[4]      = {-17, -15, 15, 17};
const char ccKnightDelta[4][2]    = {{-33, -31}, {-18, 14}, {-14, 18}, {31, 33}};
const char ccKnightCheckDelta[4][2] = {{-33, -18}, {-31, -14}, {14, 31}, {18, 33}};

// Starting position (pieces 8..14 = RED on rows 10..12; 16..22 = BLACK on 3..5).
const std::uint8_t cucpcStartup[256] = {
   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
   0, 0, 0,20,19,18,17,16,17,18,19,20, 0, 0, 0, 0,
   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
   0, 0, 0, 0,21, 0, 0, 0, 0, 0,21, 0, 0, 0, 0, 0,
   0, 0, 0,22, 0,22, 0,22, 0,22, 0,22, 0, 0, 0, 0,
   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
   0, 0, 0,14, 0,14, 0,14, 0,14, 0,14, 0, 0, 0, 0,
   0, 0, 0, 0,13, 0, 0, 0, 0, 0,13, 0, 0, 0, 0, 0,
   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
   0, 0, 0,12,11,10, 9, 8, 9,10,11,12, 0, 0, 0, 0,
   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
};

// Piece-square tables (KING, ADVISOR, BISHOP, KNIGHT, ROOK, CANNON, PAWN).
// Scored from RED's perspective on rows 10..12; BLACK uses SQUARE_FLIP.
const std::uint8_t cucvlPiecePos[7][256] = {
  { // KING
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,0,
    0,0,0,0,0,0,2,2,2,0,0,0,0,0,0,0,
    0,0,0,0,0,0,11,15,11,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
  },
  { // ADVISOR
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,20,0,20,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,23,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,20,0,20,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
  },
  { // BISHOP
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,20,0,0,0,20,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    0,0,0,18,0,0,0,23,0,0,0,18,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,20,0,0,0,20,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
  },
  { // KNIGHT
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    0,0,0,90,90,90,96,90,96,90,90,90,0,0,0,0,
    0,0,0,90,96,103,97,94,97,103,96,90,0,0,0,0,
    0,0,0,92,98,99,103,99,103,99,98,92,0,0,0,0,
    0,0,0,93,108,100,107,100,107,100,108,93,0,0,0,0,
    0,0,0,90,100,99,103,104,103,99,100,90,0,0,0,0,
    0,0,0,90,98,101,102,103,102,101,98,90,0,0,0,0,
    0,0,0,92,94,98,95,98,95,98,94,92,0,0,0,0,
    0,0,0,93,92,94,95,92,95,94,92,93,0,0,0,0,
    0,0,0,85,90,92,93,78,93,92,90,85,0,0,0,0,
    0,0,0,88,85,90,88,90,88,90,85,88,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
  },
  { // ROOK
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    0,0,0,206,208,207,213,214,213,207,208,206,0,0,0,0,
    0,0,0,206,212,209,216,233,216,209,212,206,0,0,0,0,
    0,0,0,206,208,207,214,216,214,207,208,206,0,0,0,0,
    0,0,0,206,213,213,216,216,216,213,213,206,0,0,0,0,
    0,0,0,208,211,211,214,215,214,211,211,208,0,0,0,0,
    0,0,0,208,212,212,214,215,214,212,212,208,0,0,0,0,
    0,0,0,204,209,204,212,214,212,204,209,204,0,0,0,0,
    0,0,0,198,208,204,212,212,212,204,208,198,0,0,0,0,
    0,0,0,200,208,206,212,200,212,206,208,200,0,0,0,0,
    0,0,0,194,206,204,212,200,212,204,206,194,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
  },
  { // CANNON
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    0,0,0,100,100,96,91,90,91,96,100,100,0,0,0,0,
    0,0,0,98,98,96,92,89,92,96,98,98,0,0,0,0,
    0,0,0,97,97,96,91,92,91,96,97,97,0,0,0,0,
    0,0,0,96,99,99,98,100,98,99,99,96,0,0,0,0,
    0,0,0,96,96,96,96,100,96,96,96,96,0,0,0,0,
    0,0,0,95,96,99,96,100,96,99,96,95,0,0,0,0,
    0,0,0,96,96,96,96,96,96,96,96,96,0,0,0,0,
    0,0,0,97,96,100,99,101,99,100,96,97,0,0,0,0,
    0,0,0,96,97,98,98,98,98,98,97,96,0,0,0,0,
    0,0,0,96,96,97,99,99,99,97,96,96,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
  },
  { // PAWN
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    0,0,0,9,9,9,11,13,11,9,9,9,0,0,0,0,
    0,0,0,19,24,34,42,44,42,34,24,19,0,0,0,0,
    0,0,0,19,24,32,37,37,37,32,24,19,0,0,0,0,
    0,0,0,19,23,27,29,30,29,27,23,19,0,0,0,0,
    0,0,0,14,18,20,27,29,27,20,18,14,0,0,0,0,
    0,0,0,7,0,13,0,16,0,13,0,7,0,0,0,0,
    0,0,0,7,0,7,0,15,0,7,0,7,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
  },
};

constexpr std::uint8_t cucMvvLva[24] = {
  0,0,0,0,0,0,0,0,
  5,1,1,3,4,3,2,0,
  5,1,1,3,4,3,2,0,
};

// Inline helpers — direct translations of the original macros.
inline bool IN_BOARD(int sq)   { return ccInBoard[sq] != 0; }
inline bool IN_FORT (int sq)   { return ccInFort[sq]  != 0; }
inline int  RANK_Y  (int sq)   { return sq >> 4; }
inline int  FILE_X  (int sq)   { return sq & 15; }
inline int  SQUARE_FLIP(int sq){ return 254 - sq; }
inline int  SQUARE_FORWARD(int sq, int sd) { return sq - 16 + (sd << 5); }
inline bool KING_SPAN(int src, int dst)    { return ccLegalSpan[dst - src + 256] == 1; }
inline bool ADVISOR_SPAN(int src, int dst) { return ccLegalSpan[dst - src + 256] == 2; }
inline bool BISHOP_SPAN(int src, int dst)  { return ccLegalSpan[dst - src + 256] == 3; }
inline int  BISHOP_PIN(int src, int dst)   { return (src + dst) >> 1; }
inline int  KNIGHT_PIN(int src, int dst)   { return src + ccKnightPin[dst - src + 256]; }
inline bool HOME_HALF(int sq, int sd)      { return (sq & 0x80) != (sd << 7); }
inline bool AWAY_HALF(int sq, int sd)      { return (sq & 0x80) == (sd << 7); }
inline bool SAME_HALF(int a, int b)        { return ((a ^ b) & 0x80) == 0; }
inline bool SAME_RANK(int a, int b)        { return ((a ^ b) & 0xf0) == 0; }
inline bool SAME_FILE(int a, int b)        { return ((a ^ b) & 0x0f) == 0; }
inline int  SIDE_TAG(int sd)               { return 8 + (sd << 3); }
inline int  OPP_SIDE_TAG(int sd)           { return 16 - (sd << 3); }
inline int  SRC(int mv)                    { return mv & 255; }
inline int  DST(int mv)                    { return mv >> 8; }
inline int  MOVE(int src, int dst)         { return src + (dst << 8); }

// Zobrist table — const after init, shared across instances.
struct ZobristTable {
    ZobristStruct Player;
    ZobristStruct Table[14][256];
};

struct RC4 {
    std::uint8_t s[256];
    int x, y;
    void init_zero() {
        x = y = 0;
        int j = 0;
        for (int i = 0; i < 256; ++i) s[i] = i;
        for (int i = 0; i < 256; ++i) {
            j = (j + s[i]) & 255;
            std::swap(s[i], s[j]);
        }
    }
    std::uint8_t next_byte() {
        x = (x + 1) & 255;
        y = (y + s[x]) & 255;
        std::swap(s[x], s[y]);
        return s[(s[x] + s[y]) & 255];
    }
    std::uint32_t next_long() {
        std::uint32_t a = next_byte(), b = next_byte(),
                      c = next_byte(), d = next_byte();
        return a | (b << 8) | (c << 16) | (d << 24);
    }
};

const ZobristTable& zobrist_table() {
    static const ZobristTable table = []{
        ZobristTable z{};
        RC4 rc4; rc4.init_zero();
        z.Player.dwKey   = rc4.next_long();
        z.Player.dwLock0 = rc4.next_long();
        z.Player.dwLock1 = rc4.next_long();
        for (int i = 0; i < 14; ++i)
            for (int j = 0; j < 256; ++j) {
                z.Table[i][j].dwKey   = rc4.next_long();
                z.Table[i][j].dwLock0 = rc4.next_long();
                z.Table[i][j].dwLock1 = rc4.next_long();
            }
        return z;
    }();
    return table;
}

inline void zobr_xor(ZobristStruct& a, const ZobristStruct& b) {
    a.dwKey   ^= b.dwKey;
    a.dwLock0 ^= b.dwLock0;
    a.dwLock1 ^= b.dwLock1;
}

// Thread-local RNG used by root randomization (substitutes for global rand()).
int tls_rand() {
    static thread_local std::mt19937 rng{
        static_cast<unsigned>(std::chrono::steady_clock::now().time_since_epoch().count())};
    return static_cast<int>(rng()) & 0x7fffffff;
}

// Search deadline, per-thread.  Consulted by the full/quiesce search via
// tls_deadline.  SearchMain sets it before the iterative-deepening loop.
thread_local Clock::time_point tls_deadline = Clock::time_point::max();
thread_local bool tls_time_up = false;

inline void check_time() {
    if (Clock::now() >= tls_deadline) tls_time_up = true;
}

}  // namespace

// ─────────────────────────────────────────────────────────────────────
// Engine implementation.
// ─────────────────────────────────────────────────────────────────────

XqwlEngine::XqwlEngine() {
    zobrist_table();  // force initialization
    reset();
}

void XqwlEngine::reset() {
    std::memset(HashTable, 0, sizeof(HashTable));
    std::memset(nHistoryTable, 0, sizeof(nHistoryTable));
    std::memset(mvKillers, 0, sizeof(mvKillers));
    mvResult = 0;
    Startup();
}

void XqwlEngine::ClearBoard() {
    sdPlayer = vlWhite = vlBlack = nDistance = 0;
    std::memset(ucpcSquares, 0, 256);
    zobr = ZobristStruct{};
}

void XqwlEngine::SetIrrev() {
    mvsList[0] = MoveStruct{0, 0, static_cast<std::uint8_t>(Checked() ? 1 : 0), zobr.dwKey};
    nMoveNum = 1;
}

void XqwlEngine::Startup() {
    ClearBoard();
    for (int sq = 0; sq < 256; ++sq) {
        int pc = cucpcStartup[sq];
        if (pc != 0) AddPiece(sq, pc);
    }
    SetIrrev();
}

void XqwlEngine::ChangeSide() {
    sdPlayer = 1 - sdPlayer;
    zobr_xor(zobr, zobrist_table().Player);
}

void XqwlEngine::AddPiece(int sq, int pc) {
    ucpcSquares[sq] = static_cast<std::uint8_t>(pc);
    if (pc < 16) {
        vlWhite += cucvlPiecePos[pc - 8][sq];
        zobr_xor(zobr, zobrist_table().Table[pc - 8][sq]);
    } else {
        vlBlack += cucvlPiecePos[pc - 16][SQUARE_FLIP(sq)];
        zobr_xor(zobr, zobrist_table().Table[pc - 9][sq]);
    }
}

void XqwlEngine::DelPiece(int sq, int pc) {
    ucpcSquares[sq] = 0;
    if (pc < 16) {
        vlWhite -= cucvlPiecePos[pc - 8][sq];
        zobr_xor(zobr, zobrist_table().Table[pc - 8][sq]);
    } else {
        vlBlack -= cucvlPiecePos[pc - 16][SQUARE_FLIP(sq)];
        zobr_xor(zobr, zobrist_table().Table[pc - 9][sq]);
    }
}

int XqwlEngine::Evaluate() const {
    return (sdPlayer == 0 ? vlWhite - vlBlack : vlBlack - vlWhite) + ADVANCED_VALUE;
}

bool XqwlEngine::InCheck() const    { return mvsList[nMoveNum - 1].ucbCheck != 0; }
bool XqwlEngine::Captured() const   { return mvsList[nMoveNum - 1].ucpcCaptured != 0; }
int  XqwlEngine::DrawValue() const  { return (nDistance & 1) == 0 ? -DRAW_VALUE : DRAW_VALUE; }

int XqwlEngine::RepValue(int nRepStatus) const {
    int v = ((nRepStatus & 2) == 0 ? 0 : nDistance - BAN_VALUE) +
            ((nRepStatus & 4) == 0 ? 0 : BAN_VALUE - nDistance);
    return v == 0 ? DrawValue() : v;
}

bool XqwlEngine::NullOkay() const {
    return (sdPlayer == 0 ? vlWhite : vlBlack) > NULL_MARGIN;
}

int XqwlEngine::MovePiece(int mv) {
    int sqSrc = SRC(mv), sqDst = DST(mv);
    int pcCaptured = ucpcSquares[sqDst];
    if (pcCaptured != 0) DelPiece(sqDst, pcCaptured);
    int pc = ucpcSquares[sqSrc];
    DelPiece(sqSrc, pc);
    AddPiece(sqDst, pc);
    return pcCaptured;
}

void XqwlEngine::UndoMovePiece(int mv, int pcCaptured) {
    int sqSrc = SRC(mv), sqDst = DST(mv);
    int pc = ucpcSquares[sqDst];
    DelPiece(sqDst, pc);
    AddPiece(sqSrc, pc);
    if (pcCaptured != 0) AddPiece(sqDst, pcCaptured);
}

bool XqwlEngine::MakeMove(int mv) {
    std::uint32_t dwKey = zobr.dwKey;
    int pcCaptured = MovePiece(mv);
    if (Checked()) {
        UndoMovePiece(mv, pcCaptured);
        return false;
    }
    ChangeSide();
    mvsList[nMoveNum] = MoveStruct{
        static_cast<std::uint16_t>(mv),
        static_cast<std::uint8_t>(pcCaptured),
        static_cast<std::uint8_t>(Checked() ? 1 : 0),
        dwKey};
    nMoveNum++;
    nDistance++;
    return true;
}

void XqwlEngine::UndoMakeMove() {
    nDistance--;
    nMoveNum--;
    ChangeSide();
    UndoMovePiece(mvsList[nMoveNum].wmv, mvsList[nMoveNum].ucpcCaptured);
}

void XqwlEngine::NullMove() {
    std::uint32_t dwKey = zobr.dwKey;
    ChangeSide();
    mvsList[nMoveNum] = MoveStruct{0, 0, 0, dwKey};
    nMoveNum++;
    nDistance++;
}

void XqwlEngine::UndoNullMove() {
    nDistance--;
    nMoveNum--;
    ChangeSide();
}

int XqwlEngine::GenerateMoves(int *mvs, bool bCapture) const {
    int nGen = 0;
    int pcSelf = SIDE_TAG(sdPlayer);
    int pcOpp  = OPP_SIDE_TAG(sdPlayer);
    for (int sqSrc = 0; sqSrc < 256; ++sqSrc) {
        int pcSrc = ucpcSquares[sqSrc];
        if ((pcSrc & pcSelf) == 0) continue;

        switch (pcSrc - pcSelf) {
        case PIECE_KING:
            for (int i = 0; i < 4; ++i) {
                int sqDst = sqSrc + ccKingDelta[i];
                if (!IN_FORT(sqDst)) continue;
                int pcDst = ucpcSquares[sqDst];
                if (bCapture ? (pcDst & pcOpp) != 0 : (pcDst & pcSelf) == 0)
                    mvs[nGen++] = MOVE(sqSrc, sqDst);
            }
            break;
        case PIECE_ADVISOR:
            for (int i = 0; i < 4; ++i) {
                int sqDst = sqSrc + ccAdvisorDelta[i];
                if (!IN_FORT(sqDst)) continue;
                int pcDst = ucpcSquares[sqDst];
                if (bCapture ? (pcDst & pcOpp) != 0 : (pcDst & pcSelf) == 0)
                    mvs[nGen++] = MOVE(sqSrc, sqDst);
            }
            break;
        case PIECE_BISHOP:
            for (int i = 0; i < 4; ++i) {
                int sqDst = sqSrc + ccAdvisorDelta[i];
                if (!(IN_BOARD(sqDst) && HOME_HALF(sqDst, sdPlayer)
                        && ucpcSquares[sqDst] == 0)) continue;
                sqDst += ccAdvisorDelta[i];
                int pcDst = ucpcSquares[sqDst];
                if (bCapture ? (pcDst & pcOpp) != 0 : (pcDst & pcSelf) == 0)
                    mvs[nGen++] = MOVE(sqSrc, sqDst);
            }
            break;
        case PIECE_KNIGHT:
            for (int i = 0; i < 4; ++i) {
                int sqDst = sqSrc + ccKingDelta[i];
                if (ucpcSquares[sqDst] != 0) continue;
                for (int j = 0; j < 2; ++j) {
                    sqDst = sqSrc + ccKnightDelta[i][j];
                    if (!IN_BOARD(sqDst)) continue;
                    int pcDst = ucpcSquares[sqDst];
                    if (bCapture ? (pcDst & pcOpp) != 0 : (pcDst & pcSelf) == 0)
                        mvs[nGen++] = MOVE(sqSrc, sqDst);
                }
            }
            break;
        case PIECE_ROOK:
            for (int i = 0; i < 4; ++i) {
                int nDelta = ccKingDelta[i];
                int sqDst = sqSrc + nDelta;
                while (IN_BOARD(sqDst)) {
                    int pcDst = ucpcSquares[sqDst];
                    if (pcDst == 0) {
                        if (!bCapture) mvs[nGen++] = MOVE(sqSrc, sqDst);
                    } else {
                        if ((pcDst & pcOpp) != 0) mvs[nGen++] = MOVE(sqSrc, sqDst);
                        break;
                    }
                    sqDst += nDelta;
                }
            }
            break;
        case PIECE_CANNON:
            for (int i = 0; i < 4; ++i) {
                int nDelta = ccKingDelta[i];
                int sqDst = sqSrc + nDelta;
                while (IN_BOARD(sqDst)) {
                    int pcDst = ucpcSquares[sqDst];
                    if (pcDst == 0) {
                        if (!bCapture) mvs[nGen++] = MOVE(sqSrc, sqDst);
                    } else {
                        break;
                    }
                    sqDst += nDelta;
                }
                sqDst += nDelta;
                while (IN_BOARD(sqDst)) {
                    int pcDst = ucpcSquares[sqDst];
                    if (pcDst != 0) {
                        if ((pcDst & pcOpp) != 0) mvs[nGen++] = MOVE(sqSrc, sqDst);
                        break;
                    }
                    sqDst += nDelta;
                }
            }
            break;
        case PIECE_PAWN: {
            int sqDst = SQUARE_FORWARD(sqSrc, sdPlayer);
            if (IN_BOARD(sqDst)) {
                int pcDst = ucpcSquares[sqDst];
                if (bCapture ? (pcDst & pcOpp) != 0 : (pcDst & pcSelf) == 0)
                    mvs[nGen++] = MOVE(sqSrc, sqDst);
            }
            if (AWAY_HALF(sqSrc, sdPlayer)) {
                for (int nDelta = -1; nDelta <= 1; nDelta += 2) {
                    sqDst = sqSrc + nDelta;
                    if (IN_BOARD(sqDst)) {
                        int pcDst = ucpcSquares[sqDst];
                        if (bCapture ? (pcDst & pcOpp) != 0 : (pcDst & pcSelf) == 0)
                            mvs[nGen++] = MOVE(sqSrc, sqDst);
                    }
                }
            }
            break;
        }
        }
    }
    return nGen;
}

bool XqwlEngine::LegalMove(int mv) const {
    int sqSrc = SRC(mv), sqDst = DST(mv);
    int pcSrc = ucpcSquares[sqSrc];
    int pcSelf = SIDE_TAG(sdPlayer);
    if ((pcSrc & pcSelf) == 0) return false;
    int pcDst = ucpcSquares[sqDst];
    if ((pcDst & pcSelf) != 0) return false;

    switch (pcSrc - pcSelf) {
    case PIECE_KING:    return IN_FORT(sqDst) && KING_SPAN(sqSrc, sqDst);
    case PIECE_ADVISOR: return IN_FORT(sqDst) && ADVISOR_SPAN(sqSrc, sqDst);
    case PIECE_BISHOP:
        return SAME_HALF(sqSrc, sqDst) && BISHOP_SPAN(sqSrc, sqDst)
            && ucpcSquares[BISHOP_PIN(sqSrc, sqDst)] == 0;
    case PIECE_KNIGHT: {
        int sqPin = KNIGHT_PIN(sqSrc, sqDst);
        return sqPin != sqSrc && ucpcSquares[sqPin] == 0;
    }
    case PIECE_ROOK:
    case PIECE_CANNON: {
        int nDelta;
        if (SAME_RANK(sqSrc, sqDst))      nDelta = (sqDst < sqSrc ? -1 : 1);
        else if (SAME_FILE(sqSrc, sqDst)) nDelta = (sqDst < sqSrc ? -16 : 16);
        else return false;
        int sqPin = sqSrc + nDelta;
        while (sqPin != sqDst && ucpcSquares[sqPin] == 0) sqPin += nDelta;
        if (sqPin == sqDst)
            return pcDst == 0 || pcSrc - pcSelf == PIECE_ROOK;
        if (pcDst != 0 && pcSrc - pcSelf == PIECE_CANNON) {
            sqPin += nDelta;
            while (sqPin != sqDst && ucpcSquares[sqPin] == 0) sqPin += nDelta;
            return sqPin == sqDst;
        }
        return false;
    }
    case PIECE_PAWN:
        if (AWAY_HALF(sqDst, sdPlayer)
                && (sqDst == sqSrc - 1 || sqDst == sqSrc + 1)) return true;
        return sqDst == SQUARE_FORWARD(sqSrc, sdPlayer);
    default:
        return false;
    }
}

bool XqwlEngine::Checked() const {
    int pcSelf = SIDE_TAG(sdPlayer);
    int pcOpp  = OPP_SIDE_TAG(sdPlayer);

    for (int sqSrc = 0; sqSrc < 256; ++sqSrc) {
        if (ucpcSquares[sqSrc] != pcSelf + PIECE_KING) continue;

        if (ucpcSquares[SQUARE_FORWARD(sqSrc, sdPlayer)] == pcOpp + PIECE_PAWN) return true;
        for (int d = -1; d <= 1; d += 2)
            if (ucpcSquares[sqSrc + d] == pcOpp + PIECE_PAWN) return true;

        for (int i = 0; i < 4; ++i) {
            if (ucpcSquares[sqSrc + ccAdvisorDelta[i]] != 0) continue;
            for (int j = 0; j < 2; ++j) {
                int pcDst = ucpcSquares[sqSrc + ccKnightCheckDelta[i][j]];
                if (pcDst == pcOpp + PIECE_KNIGHT) return true;
            }
        }

        for (int i = 0; i < 4; ++i) {
            int nDelta = ccKingDelta[i];
            int sqDst = sqSrc + nDelta;
            while (IN_BOARD(sqDst)) {
                int pcDst = ucpcSquares[sqDst];
                if (pcDst != 0) {
                    if (pcDst == pcOpp + PIECE_ROOK || pcDst == pcOpp + PIECE_KING)
                        return true;
                    break;
                }
                sqDst += nDelta;
            }
            sqDst += nDelta;
            while (IN_BOARD(sqDst)) {
                int pcDst = ucpcSquares[sqDst];
                if (pcDst != 0) {
                    if (pcDst == pcOpp + PIECE_CANNON) return true;
                    break;
                }
                sqDst += nDelta;
            }
        }
        return false;
    }
    return false;
}

int XqwlEngine::RepStatus(int nRecur) const {
    bool bSelfSide = false;
    bool bPerpCheck = true, bOppPerpCheck = true;
    const MoveStruct* lpmvs = &mvsList[nMoveNum - 1];
    while (lpmvs->wmv != 0 && lpmvs->ucpcCaptured == 0) {
        if (bSelfSide) {
            bPerpCheck = bPerpCheck && lpmvs->ucbCheck;
            if (lpmvs->dwKey == zobr.dwKey) {
                nRecur--;
                if (nRecur == 0)
                    return 1 + (bPerpCheck ? 2 : 0) + (bOppPerpCheck ? 4 : 0);
            }
        } else {
            bOppPerpCheck = bOppPerpCheck && lpmvs->ucbCheck;
        }
        bSelfSide = !bSelfSide;
        lpmvs--;
    }
    return 0;
}

int XqwlEngine::ProbeHash(int vlAlpha, int vlBeta, int nDepth, int& mv) {
    HashItem hsh = HashTable[zobr.dwKey & (HASH_SIZE - 1)];
    if (hsh.dwLock0 != zobr.dwLock0 || hsh.dwLock1 != zobr.dwLock1) {
        mv = 0;
        return -MATE_VALUE;
    }
    mv = hsh.wmv;
    bool bMate = false;
    if (hsh.svl > WIN_VALUE) {
        if (hsh.svl < BAN_VALUE) return -MATE_VALUE;
        hsh.svl = static_cast<std::int16_t>(hsh.svl - nDistance);
        bMate = true;
    } else if (hsh.svl < -WIN_VALUE) {
        if (hsh.svl > -BAN_VALUE) return -MATE_VALUE;
        hsh.svl = static_cast<std::int16_t>(hsh.svl + nDistance);
        bMate = true;
    }
    if (hsh.ucDepth >= nDepth || bMate) {
        if (hsh.ucFlag == HASH_BETA)  return hsh.svl >= vlBeta  ? hsh.svl : -MATE_VALUE;
        if (hsh.ucFlag == HASH_ALPHA) return hsh.svl <= vlAlpha ? hsh.svl : -MATE_VALUE;
        return hsh.svl;
    }
    return -MATE_VALUE;
}

void XqwlEngine::RecordHash(int nFlag, int vl, int nDepth, int mv) {
    HashItem& hsh = HashTable[zobr.dwKey & (HASH_SIZE - 1)];
    if (hsh.ucDepth > nDepth) return;
    hsh.ucFlag = static_cast<std::uint8_t>(nFlag);
    hsh.ucDepth = static_cast<std::uint8_t>(nDepth);
    if (vl > WIN_VALUE) {
        if (mv == 0 && vl <= BAN_VALUE) return;
        hsh.svl = static_cast<std::int16_t>(vl + nDistance);
    } else if (vl < -WIN_VALUE) {
        if (mv == 0 && vl >= -BAN_VALUE) return;
        hsh.svl = static_cast<std::int16_t>(vl - nDistance);
    } else {
        hsh.svl = static_cast<std::int16_t>(vl);
    }
    hsh.wmv = static_cast<std::uint16_t>(mv);
    hsh.dwLock0 = zobr.dwLock0;
    hsh.dwLock1 = zobr.dwLock1;
}

int XqwlEngine::MvvLva(int mv) const {
    return (cucMvvLva[ucpcSquares[DST(mv)]] << 3) - cucMvvLva[ucpcSquares[SRC(mv)]];
}

void XqwlEngine::SetBestMove(int mv, int nDepth) {
    nHistoryTable[mv] += nDepth * nDepth;
    int* lp = mvKillers[nDistance];
    if (lp[0] != mv) { lp[1] = lp[0]; lp[0] = mv; }
}

void XqwlEngine::SortStruct::Init(XqwlEngine* e, int mvHash_) {
    eng = e;
    mvHash = mvHash_;
    mvKiller1 = e->mvKillers[e->nDistance][0];
    mvKiller2 = e->mvKillers[e->nDistance][1];
    nPhase = PHASE_HASH;
}

int XqwlEngine::SortStruct::Next() {
    int mv;
    switch (nPhase) {
    case PHASE_HASH:
        nPhase = PHASE_KILLER_1;
        if (mvHash != 0) return mvHash;
        [[fallthrough]];
    case PHASE_KILLER_1:
        nPhase = PHASE_KILLER_2;
        if (mvKiller1 != mvHash && mvKiller1 != 0 && eng->LegalMove(mvKiller1))
            return mvKiller1;
        [[fallthrough]];
    case PHASE_KILLER_2:
        nPhase = PHASE_GEN_MOVES;
        if (mvKiller2 != mvHash && mvKiller2 != 0 && eng->LegalMove(mvKiller2))
            return mvKiller2;
        [[fallthrough]];
    case PHASE_GEN_MOVES: {
        nPhase = PHASE_REST;
        nGenMoves = eng->GenerateMoves(mvs);
        const int* hist = eng->nHistoryTable;
        std::sort(mvs, mvs + nGenMoves, [hist](int a, int b) {
            return hist[b] < hist[a];
        });
        nIndex = 0;
        [[fallthrough]];
    }
    case PHASE_REST:
        while (nIndex < nGenMoves) {
            mv = mvs[nIndex++];
            if (mv != mvHash && mv != mvKiller1 && mv != mvKiller2) return mv;
        }
        [[fallthrough]];
    default:
        return 0;
    }
}

int XqwlEngine::SearchQuiesc(int vlAlpha, int vlBeta) {
    if (tls_time_up) return vlAlpha;
    int vl = RepStatus();
    if (vl != 0) return RepValue(vl);
    if (nDistance == LIMIT_DEPTH) return Evaluate();

    int vlBest = -MATE_VALUE;
    int mvs[MAX_GEN_MOVES];
    int nGen;

    if (InCheck()) {
        nGen = GenerateMoves(mvs);
        const int* hist = nHistoryTable;
        std::sort(mvs, mvs + nGen, [hist](int a, int b) {
            return hist[b] < hist[a];
        });
    } else {
        vl = Evaluate();
        if (vl > vlBest) {
            vlBest = vl;
            if (vl >= vlBeta) return vl;
            if (vl > vlAlpha) vlAlpha = vl;
        }
        nGen = GenerateMoves(mvs, /*bCapture=*/true);
        std::sort(mvs, mvs + nGen, [this](int a, int b) {
            return MvvLva(b) < MvvLva(a);
        });
    }

    for (int i = 0; i < nGen; ++i) {
        if (MakeMove(mvs[i])) {
            vl = -SearchQuiesc(-vlBeta, -vlAlpha);
            UndoMakeMove();
            if (vl > vlBest) {
                vlBest = vl;
                if (vl >= vlBeta) return vl;
                if (vl > vlAlpha) vlAlpha = vl;
            }
        }
    }
    return vlBest == -MATE_VALUE ? nDistance - MATE_VALUE : vlBest;
}

int XqwlEngine::SearchFull(int vlAlpha, int vlBeta, int nDepth, bool bNoNull) {
    if (nDepth <= 0) return SearchQuiesc(vlAlpha, vlBeta);
    check_time();
    if (tls_time_up) return vlAlpha;

    int vl = RepStatus();
    if (vl != 0) return RepValue(vl);
    if (nDistance == LIMIT_DEPTH) return Evaluate();

    int mvHash;
    vl = ProbeHash(vlAlpha, vlBeta, nDepth, mvHash);
    if (vl > -MATE_VALUE) return vl;

    if (!bNoNull && !InCheck() && NullOkay()) {
        NullMove();
        vl = -SearchFull(-vlBeta, 1 - vlBeta, nDepth - NULL_DEPTH - 1, /*bNoNull=*/true);
        UndoNullMove();
        if (vl >= vlBeta) return vl;
    }

    int nHashFlag = HASH_ALPHA;
    int vlBest = -MATE_VALUE;
    int mvBest = 0;

    SortStruct sort;
    sort.Init(this, mvHash);

    int mv;
    while ((mv = sort.Next()) != 0) {
        if (MakeMove(mv)) {
            int nNewDepth = InCheck() ? nDepth : nDepth - 1;
            if (vlBest == -MATE_VALUE) {
                vl = -SearchFull(-vlBeta, -vlAlpha, nNewDepth);
            } else {
                vl = -SearchFull(-vlAlpha - 1, -vlAlpha, nNewDepth);
                if (vl > vlAlpha && vl < vlBeta)
                    vl = -SearchFull(-vlBeta, -vlAlpha, nNewDepth);
            }
            UndoMakeMove();
            if (vl > vlBest) {
                vlBest = vl;
                if (vl >= vlBeta) { nHashFlag = HASH_BETA; mvBest = mv; break; }
                if (vl > vlAlpha) { nHashFlag = HASH_PV;   mvBest = mv; vlAlpha = vl; }
            }
        }
    }

    if (vlBest == -MATE_VALUE) return nDistance - MATE_VALUE;
    RecordHash(nHashFlag, vlBest, nDepth, mvBest);
    if (mvBest != 0) SetBestMove(mvBest, nDepth);
    return vlBest;
}

int XqwlEngine::SearchRoot(int nDepth) {
    int vlBest = -MATE_VALUE;
    SortStruct sort;
    sort.Init(this, mvResult);
    int mv;
    while ((mv = sort.Next()) != 0) {
        if (MakeMove(mv)) {
            int nNewDepth = InCheck() ? nDepth : nDepth - 1;
            int vl;
            if (vlBest == -MATE_VALUE) {
                vl = -SearchFull(-MATE_VALUE, MATE_VALUE, nNewDepth, /*bNoNull=*/true);
            } else {
                vl = -SearchFull(-vlBest - 1, -vlBest, nNewDepth);
                if (vl > vlBest)
                    vl = -SearchFull(-MATE_VALUE, -vlBest, nNewDepth, /*bNoNull=*/true);
            }
            UndoMakeMove();
            if (vl > vlBest) {
                vlBest = vl;
                mvResult = mv;
                if (vlBest > -WIN_VALUE && vlBest < WIN_VALUE)
                    vlBest += (tls_rand() & RANDOM_MASK) - (tls_rand() & RANDOM_MASK);
            }
        }
    }
    RecordHash(HASH_PV, vlBest, nDepth, mvResult);
    if (mvResult != 0) SetBestMove(mvResult, nDepth);
    return vlBest;
}

int XqwlEngine::SearchMain() {
    std::memset(nHistoryTable, 0, sizeof(nHistoryTable));
    std::memset(mvKillers, 0, sizeof(mvKillers));
    std::memset(HashTable, 0, sizeof(HashTable));
    nDistance = 0;
    mvResult = 0;

    // Check whether there is a unique legal move.
    int mvs[MAX_GEN_MOVES];
    int nGen = GenerateMoves(mvs);
    int nLegal = 0;
    for (int i = 0; i < nGen; ++i) {
        if (MakeMove(mvs[i])) {
            UndoMakeMove();
            mvResult = mvs[i];
            nLegal++;
        }
    }
    if (nLegal == 0) return 0;
    if (nLegal == 1) return 1;

    tls_time_up = false;
    tls_deadline = (time_budget_ms_ > 0)
        ? Clock::now() + std::chrono::milliseconds(time_budget_ms_)
        : Clock::time_point::max();

    int last_depth = 0;
    for (int i = 1; i <= max_depth_; ++i) {
        int vl = SearchRoot(i);
        last_depth = i;
        if (tls_time_up) break;
        if (vl > WIN_VALUE || vl < -WIN_VALUE) break;
        check_time();
        if (tls_time_up) break;
    }
    return last_depth;
}

int XqwlEngine::think() {
    int plies = SearchMain();
    (void)plies;
    return mvResult;
}

bool XqwlEngine::play(int mv) {
    if (!LegalMove(mv)) return false;
    return MakeMove(mv);
}

int XqwlEngine::legal_move_count() {
    int mvs[MAX_GEN_MOVES];
    int nGen = GenerateMoves(mvs);
    int nLegal = 0;
    for (int i = 0; i < nGen; ++i) {
        if (MakeMove(mvs[i])) {
            UndoMakeMove();
            nLegal++;
        }
    }
    return nLegal;
}

Status XqwlEngine::status() {
    // Chinese-chess repetition rule:
    //   3 occurrences of the same position (局面循环三次) → draw, unless
    //     one side is continuously attacking (长打), in which case that
    //     side is required to vary and the game continues.
    //   4 occurrences (第四次重复局面) → the perpetually-attacking side
    //     loses; otherwise draw.
    //
    // RepStatus(n) triggers when n prior matches are found walking back
    // through the reversible-move history, i.e. n+1 total occurrences.
    // Perpetual-attack classification here only covers 长将 (perpetual
    // check) via the ucbCheck flag on each move record; 长捉 / 长杀 are
    // not distinguished from a normal repetition and will draw at three.
    int rep4 = RepStatus(3);
    if (rep4 != 0) {
        bool self_perp = (rep4 & 2) != 0;
        bool opp_perp  = (rep4 & 4) != 0;
        if (self_perp && !opp_perp) return Status::SELF_PERPETUAL;
        if (opp_perp && !self_perp) return Status::OPP_PERPETUAL;
        return Status::REPETITION_DRAW;
    }
    int rep3 = RepStatus(2);
    if (rep3 != 0) {
        bool self_perp = (rep3 & 2) != 0;
        bool opp_perp  = (rep3 & 4) != 0;
        if (!self_perp && !opp_perp) return Status::REPETITION_DRAW;
        // Attacker has one more cycle to vary before the 4-rep loss above.
    }
    if (legal_move_count() == 0) {
        return InCheck() ? Status::CHECKMATE : Status::STALEMATE;
    }
    return Status::ONGOING;
}

// ─────────────────────────────────────────────────────────────────────
// Move encoding bridge.
// ─────────────────────────────────────────────────────────────────────

static constexpr int kOurRows = 10;
static constexpr int kOurCols = 9;
static constexpr int kOurArea = kOurRows * kOurCols;

int xqwl_sq_to_xiangqi(int sq) {
    int row = (sq >> 4) - 3;  // [3..12] -> [0..9]
    int col = (sq & 15) - 3;  // [3..11] -> [0..8]
    return row * kOurCols + col;
}

int xiangqi_sq_to_xqwl(int sq) {
    int row = sq / kOurCols;
    int col = sq % kOurCols;
    return ((row + 3) << 4) | (col + 3);
}

int xqwl_move_to_action(int mv) {
    int src = xqwl_sq_to_xiangqi(mv & 255);
    int dst = xqwl_sq_to_xiangqi(mv >> 8);
    return src * kOurArea + dst;
}

int action_to_xqwl_move(int action) {
    int src = action / kOurArea;
    int dst = action % kOurArea;
    return xiangqi_sq_to_xqwl(src) | (xiangqi_sq_to_xqwl(dst) << 8);
}

}  // namespace xqwl
