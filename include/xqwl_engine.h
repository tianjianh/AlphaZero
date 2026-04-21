#pragma once

// XQWL06 (XiangQi Wizard Light 0.6) engine port.
// Alpha-beta search with PVS, quiescence, null-move, transposition table,
// killers, and history heuristic.  Ported from the Morning Yellow 2008 release;
// Win32 UI and opening-book resource are dropped, engine internals are kept
// intact and wrapped in a per-instance class so each bootstrap worker can
// own its own transposition table.

#include <array>
#include <cstdint>

namespace xqwl {

constexpr int MAX_GEN_MOVES = 128;
constexpr int MAX_MOVES     = 256;
constexpr int LIMIT_DEPTH   = 64;
constexpr int HASH_SIZE     = 1 << 20;

enum class Status {
    ONGOING,
    CHECKMATE,           // side to move has no legal move and is in check
    STALEMATE,           // side to move has no legal move and is not in check
    REPETITION_DRAW,
    SELF_PERPETUAL,      // side-to-move loses (it caused the perpetual)
    OPP_PERPETUAL,       // side-to-move wins
};

struct MoveStruct {
    std::uint16_t wmv = 0;
    std::uint8_t  ucpcCaptured = 0;
    std::uint8_t  ucbCheck = 0;
    std::uint32_t dwKey = 0;
};

struct ZobristStruct {
    std::uint32_t dwKey = 0, dwLock0 = 0, dwLock1 = 0;
};

struct HashItem {
    std::uint8_t  ucDepth;
    std::uint8_t  ucFlag;
    std::int16_t  svl;
    std::uint16_t wmv;
    std::uint16_t wReserved;
    std::uint32_t dwLock0;
    std::uint32_t dwLock1;
};

class XqwlEngine {
public:
    XqwlEngine();
    // Reset to the start position.  Also clears the transposition table and
    // move history counters so repetition detection stays scoped per game.
    void reset();

    // Iterative-deepening search.  Returns an xqwl-encoded best move
    // (src|(dst<<8)) or 0 when no legal move exists.  Bound by max_depth
    // and/or time_budget_ms, whichever trips first.
    int think();

    // Apply an xqwl-encoded move.  Returns false if illegal.
    bool play(int mv);

    // Current side to move (0 = RED, 1 = BLACK).
    int side_to_move() const { return sdPlayer; }

    // Legal-move count at the current position.
    int legal_move_count();

    // High-level status of the current position (checkmate / stalemate /
    // repetition of various flavors).
    Status status();

    // Read-only board (256-entry 16x16 XQWL layout).
    const std::uint8_t* board() const { return ucpcSquares; }

    void set_max_depth(int d) { max_depth_ = d; }
    void set_time_budget_ms(int ms) { time_budget_ms_ = ms; }

private:
    // ── Position state (was global `pos`) ─────────────────────────────
    int sdPlayer = 0;
    std::uint8_t ucpcSquares[256] = {};
    int vlWhite = 0, vlBlack = 0;
    int nDistance = 0, nMoveNum = 0;
    MoveStruct   mvsList[MAX_MOVES];
    ZobristStruct zobr;

    // ── Search state (was global `Search`) ────────────────────────────
    int mvResult = 0;
    int nHistoryTable[65536] = {};
    int mvKillers[LIMIT_DEPTH][2] = {};
    HashItem HashTable[HASH_SIZE] = {};

    // ── Configuration ─────────────────────────────────────────────────
    int max_depth_ = LIMIT_DEPTH;
    int time_budget_ms_ = 500;

    // ── Position methods ──────────────────────────────────────────────
    void ClearBoard();
    void SetIrrev();
    void Startup();
    void ChangeSide();
    void AddPiece(int sq, int pc);
    void DelPiece(int sq, int pc);
    int  Evaluate() const;
    bool InCheck() const;
    bool Captured() const;
    int  MovePiece(int mv);
    void UndoMovePiece(int mv, int pcCaptured);
    bool MakeMove(int mv);
    void UndoMakeMove();
    void NullMove();
    void UndoNullMove();
    int  GenerateMoves(int *mvs, bool bCapture = false) const;
    bool LegalMove(int mv) const;
    bool Checked() const;
    int  DrawValue() const;
    int  RepStatus(int nRecur = 1) const;
    int  RepValue(int nRepStatus) const;
    bool NullOkay() const;

    // ── Search ────────────────────────────────────────────────────────
    int  ProbeHash(int vlAlpha, int vlBeta, int nDepth, int& mv);
    void RecordHash(int nFlag, int vl, int nDepth, int mv);
    void SetBestMove(int mv, int nDepth);
    int  MvvLva(int mv) const;

    int  SearchQuiesc(int vlAlpha, int vlBeta);
    int  SearchFull(int vlAlpha, int vlBeta, int nDepth, bool bNoNull = false);
    int  SearchRoot(int nDepth);
    // Returns number of plies actually searched.
    int  SearchMain();

    // Move-sorting helper bound to this engine.
    struct SortStruct {
        XqwlEngine* eng;
        int mvHash, mvKiller1, mvKiller2;
        int nPhase, nIndex, nGenMoves;
        int mvs[MAX_GEN_MOVES];
        void Init(XqwlEngine* e, int mvHash_);
        int  Next();
    };
};

// ── Move encoding bridge with the project's XiangqiGame ────────────────
// XQWL:     sq = row<<4 | col,  row ∈ [3,12], col ∈ [3,11].  move = src | (dst<<8)
// Xiangqi:  sq = row*9 + col,   row ∈ [0,9],  col ∈ [0,8].   action = src*90 + dst
//
// Both conventions put RED on rows {7..9} / {10..12 in 16x16}; no vertical flip
// is needed, only a +3 offset on each axis.
int xqwl_sq_to_xiangqi(int sq);
int xiangqi_sq_to_xqwl(int sq);
int xqwl_move_to_action(int mv);
int action_to_xqwl_move(int action);

}  // namespace xqwl
