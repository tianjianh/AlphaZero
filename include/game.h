#pragma once

#include <array>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace minigo {

enum Stone : int8_t { EMPTY = 0, RED = 1, BLACK = 2, WHITE = BLACK };

inline Stone opponent(Stone s) { return s == RED ? BLACK : RED; }

constexpr int BOARD_ROWS = 10;
constexpr int BOARD_COLS = 9;
constexpr int BOARD_AREA = BOARD_ROWS * BOARD_COLS;
constexpr int MAX_GAME_MOVES = 512;

enum Piece : int8_t {
    NO_PIECE = 0,
    RED_KING,
    RED_ADVISOR,
    RED_BISHOP,
    RED_KNIGHT,
    RED_ROOK,
    RED_CANNON,
    RED_PAWN,
    BLACK_KING,
    BLACK_ADVISOR,
    BLACK_BISHOP,
    BLACK_KNIGHT,
    BLACK_ROOK,
    BLACK_CANNON,
    BLACK_PAWN,
};

class XiangqiGame {
public:
    explicit XiangqiGame(int history_length = 4);

    void reset();
    XiangqiGame copy() const;
    bool is_legal(int action) const;
    void play(int action);
    void force_draw();

    void get_legal_moves(std::vector<float>& legal) const;
    // Signed ownership from `side`'s perspective: +1 for own piece,
    // -1 for opponent piece, 0 for empty.  Intended to be called once at
    // game end: every training record in the game receives the same terminal
    // board (sign-flipped across the two sides).
    void get_final_ownership_from(Stone side, std::vector<float>& out) const;
    void encode(std::vector<float>& out) const;

    std::string display() const;
    std::string action_to_str(int action) const;
    int str_to_action(const std::string& s) const;

    static constexpr int rows() { return BOARD_ROWS; }
    static constexpr int cols() { return BOARD_COLS; }
    static constexpr int area() { return BOARD_AREA; }
    static constexpr int action_size() { return BOARD_AREA * BOARD_AREA; }

    int board_rows = BOARD_ROWS;
    int board_cols = BOARD_COLS;
    int history_length = 4;
    Stone current_player = RED;
    int move_count = 0;
    int last_move = -1;
    bool game_over = false;
    Stone winner = EMPTY;

    int8_t board[BOARD_ROWS][BOARD_COLS] = {};

private:
    struct MoveRecord {
        int action = -1;
        int8_t captured = NO_PIECE;
        bool gave_check = false;
        uint64_t pre_move_hash = 0;
    };

    static constexpr int PIECE_PLANES = 14;
    static constexpr int RING_CAP = 16;
    static constexpr int REPETITION_NONE = 0;
    static constexpr int REPETITION_DRAW = 1;
    static constexpr int REPETITION_SELF_PERPETUAL_CHECK = 2;
    static constexpr int REPETITION_OPP_PERPETUAL_CHECK = 4;

    std::array<std::array<int8_t, BOARD_AREA>, RING_CAP> ring_buf_{};
    int ring_head_ = 0;
    int ring_size_ = 0;
    std::array<MoveRecord, MAX_GAME_MOVES + 1> move_history_{};
    int move_history_count_ = 0;
    uint64_t current_hash_ = 0;

    static bool in_bounds(int r, int c);
    static bool in_red_palace(int r, int c);
    static bool in_black_palace(int r, int c);
    static bool in_palace(Stone side, int r, int c);
    static bool crossed_river(Stone side, int r);
    static int piece_type(int8_t piece);
    static Stone piece_color(int8_t piece);
    static bool is_side_piece(int8_t piece, Stone side);
    static char piece_to_char(int8_t piece);
    static int encode_action(int src, int dst) { return src * BOARD_AREA + dst; }
    static int sq_index(int r, int c) { return r * BOARD_COLS + c; }
    static int action_src(int action) { return action / BOARD_AREA; }
    static int action_dst(int action) { return action % BOARD_AREA; }

    bool find_king(Stone side, int& r, int& c) const;
    bool is_pseudo_legal(int sr, int sc, int dr, int dc) const;
    // Enumerate pseudo-legal destinations for the piece at (sr, sc), without
    // king-safety filtering.  Writes sq_index(dr, dc) values into out[] and
    // returns the count.  Worst case per piece is 17 (rook / cannon rays);
    // 32 is a safe upper bound.  Avoids the old "try all 90 destinations
    // then reject via is_pseudo_legal" scan in the hot path.
    int collect_pseudo_legal_dsts(int sr, int sc, int out[32]) const;
    void apply_move_unchecked(int sr, int sc, int dr, int dc, int8_t& captured);
    void undo_move_unchecked(int sr, int sc, int dr, int dc, int8_t captured);
    bool is_in_check(Stone side) const;
    bool is_square_attacked(int r, int c, Stone by) const;
    bool has_any_legal_move(Stone side) const;
    int repetition_status(int n_recur = 1) const;
    uint64_t compute_hash() const;
    void update_history();
};

}  // namespace minigo
