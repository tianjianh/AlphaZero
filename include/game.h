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

    // Per-intersection ownership from current player's perspective.
    // out[r*board_size+c] = 1.0 if owned by `player`, 0.0 otherwise.
    // Neutral intersections (dame) = 0.0.  Uses Tromp-Taylor flood-fill.
    void get_ownership(Stone player, std::vector<float>& out) const;

    // Neural network encoding: (input_channels, H, W) flattened row-major
    void encode(std::vector<float>& out) const;

    // Most-recent action locations, used by the KataGo V7 history planes
    // (which encode WHERE moves were played, not the resulting boards).
    // steps_back=0 is the most recent action, 1 is the move before, etc.
    // Values: a board action in [0, n*n), PASS_MOVE (-1) for a pass, or
    // -2 sentinel meaning "no move that far back yet". MiniGo's ring_buf_
    // (board snapshots) is used for the MiniGo encoder; this auxiliary
    // array is purely additive and only consulted by katago_inputs.cpp.
    int recent_action(int steps_back) const;

    // True if `action` is a simple-ko-banned move (would replay the
    // previous board). Empty-cell + non-suicide moves can be flagged.
    // Used by the KataGo V7 encoder for plane 6.
    bool is_ko_ban(int action) const;

    // Display
    std::string action_to_str(int action) const;

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

    // Recent move locations for KataGo V7 history planes. recent_actions_[0]
    // is the most recent action (1 ply ago). KataGo V7 reads up to 5 plies
    // of history. Only katago_inputs.cpp reads this; the MiniGo encoder
    // ignores it.
    static constexpr int RECENT_ACTIONS_CAP = 5;
    std::array<int, RECENT_ACTIONS_CAP> recent_actions_;
    int recent_actions_count_ = 0;

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
