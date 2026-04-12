#pragma once

#include <array>
#include <cstdint>
#include <string>
#include <vector>

namespace minigo {

enum Stone : int8_t { EMPTY = 0, BLACK = 1, WHITE = 2 };

inline Stone opponent(Stone s) { return s == BLACK ? WHITE : BLACK; }

constexpr int MAX_BOARD = 19;
constexpr int PASS_MOVE = -1;

class GoGame {
public:
    GoGame(int board_size = 9, float komi = 7.5f, int history_length = 8);

    // Core interface
    void reset();
    GoGame copy() const;
    bool is_legal(int action) const;
    void play(int action);

    // Info
    void get_legal_moves(std::vector<float>& legal) const;
    std::pair<float, float> score() const;

    // Neural network encoding: (input_channels, H, W) flattened row-major
    void encode(std::vector<float>& out) const;

    // Display
    std::string display() const;
    std::string action_to_str(int action) const;
    int str_to_action(const std::string& s) const;

    // State
    int board_size;
    float komi;
    int history_length;
    Stone current_player;
    int move_count;
    int last_move;
    int consecutive_passes;
    bool game_over;
    Stone winner;
    float final_black_score;   // bs - ws (komi included), set by score_game()

    Stone board[MAX_BOARD][MAX_BOARD];

private:
    Stone prev_board[MAX_BOARD][MAX_BOARD];
    bool has_prev_board;

    // ── Ring buffer for history boards ────────────────────────────────
    // Replaces std::vector + erase(begin()) [O(n) memmove every play()].
    // All slots are inline (no heap allocation) → copy() is a plain memcpy.
    static constexpr int RING_CAP = 9;  // enough for history_length ≤ 8
    std::array<std::array<int8_t, MAX_BOARD * MAX_BOARD>, RING_CAP> ring_buf_;
    int ring_head_ = 0;   // index of oldest valid slot
    int ring_size_ = 0;   // number of valid entries (0 … history_length)

    struct Pos { int r, c; };
    // Stack-based group helpers — out_group is a caller-supplied buffer
    // sized for the worst case (`MAX_BOARD * MAX_BOARD`).  Avoids the
    // per-call heap allocation that std::vector<Pos> incurred in the
    // hot is_legal/play/score loops.
    int  get_group(int r, int c, Pos* out_group, int& liberties) const;
    int  get_group_on(const Stone brd[][MAX_BOARD], int r, int c,
                      Pos* out_group, int& liberties) const;
    void remove_group(const Pos* group, int n);
    void neighbors(int r, int c, Pos* nbrs, int& count) const;
    bool is_legal_at_slow(int r, int c) const;
    void update_history();
    void score_game();
};

}  // namespace minigo
