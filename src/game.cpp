#include "game.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <sstream>
#include <stdexcept>

namespace minigo {

namespace {

constexpr std::array<int, 7> kPieceValues = {
    10000, 110, 110, 300, 600, 350, 70
};

constexpr int kKnightOffsets[8][4] = {
    {-2, -1, -1,  0},
    {-2,  1, -1,  0},
    {-1, -2,  0, -1},
    {-1,  2,  0,  1},
    { 1, -2,  0, -1},
    { 1,  2,  0,  1},
    { 2, -1,  1,  0},
    { 2,  1,  1,  0},
};

constexpr int kBishopOffsets[4][4] = {
    {-2, -2, -1, -1},
    {-2,  2, -1,  1},
    { 2, -2,  1, -1},
    { 2,  2,  1,  1},
};

constexpr int kKingDirs[4][2] = {
    {-1, 0}, {1, 0}, {0, -1}, {0, 1}
};

constexpr int kAdvisorDirs[4][2] = {
    {-1, -1}, {-1, 1}, {1, -1}, {1, 1}
};

int normalized_rank(int r) {
    return BOARD_ROWS - 1 - r;
}

bool parse_square(const std::string& s, size_t pos, int& r, int& c) {
    if (pos + 1 >= s.size()) return false;
    char file = static_cast<char>(std::tolower(static_cast<unsigned char>(s[pos])));
    char rank = s[pos + 1];
    if (file < 'a' || file >= 'a' + BOARD_COLS) return false;
    if (rank < '0' || rank > '9') return false;
    c = file - 'a';
    r = BOARD_ROWS - 1 - (rank - '0');
    return XiangqiGame::rows() > r && r >= 0;
}

}  // namespace

XiangqiGame::XiangqiGame(int history_length_)
    : history_length(std::max(1, std::min(history_length_, RING_CAP))) {
    reset();
}

void XiangqiGame::reset() {
    for (auto& row : board) {
        std::fill(std::begin(row), std::end(row), NO_PIECE);
    }

    const int8_t startup[BOARD_ROWS][BOARD_COLS] = {
        {BLACK_ROOK, BLACK_KNIGHT, BLACK_BISHOP, BLACK_ADVISOR, BLACK_KING, BLACK_ADVISOR, BLACK_BISHOP, BLACK_KNIGHT, BLACK_ROOK},
        {NO_PIECE,   NO_PIECE,     NO_PIECE,     NO_PIECE,      NO_PIECE,    NO_PIECE,      NO_PIECE,     NO_PIECE,   NO_PIECE},
        {NO_PIECE,   BLACK_CANNON, NO_PIECE,     NO_PIECE,      NO_PIECE,    NO_PIECE,      NO_PIECE,     BLACK_CANNON, NO_PIECE},
        {BLACK_PAWN, NO_PIECE,     BLACK_PAWN,   NO_PIECE,      BLACK_PAWN,  NO_PIECE,      BLACK_PAWN,   NO_PIECE,   BLACK_PAWN},
        {NO_PIECE,   NO_PIECE,     NO_PIECE,     NO_PIECE,      NO_PIECE,    NO_PIECE,      NO_PIECE,     NO_PIECE,   NO_PIECE},
        {NO_PIECE,   NO_PIECE,     NO_PIECE,     NO_PIECE,      NO_PIECE,    NO_PIECE,      NO_PIECE,     NO_PIECE,   NO_PIECE},
        {RED_PAWN,   NO_PIECE,     RED_PAWN,     NO_PIECE,      RED_PAWN,    NO_PIECE,      RED_PAWN,     NO_PIECE,   RED_PAWN},
        {NO_PIECE,   RED_CANNON,   NO_PIECE,     NO_PIECE,      NO_PIECE,    NO_PIECE,      NO_PIECE,     RED_CANNON, NO_PIECE},
        {NO_PIECE,   NO_PIECE,     NO_PIECE,     NO_PIECE,      NO_PIECE,    NO_PIECE,      NO_PIECE,     NO_PIECE,   NO_PIECE},
        {RED_ROOK,   RED_KNIGHT,   RED_BISHOP,   RED_ADVISOR,   RED_KING,    RED_ADVISOR,   RED_BISHOP,   RED_KNIGHT, RED_ROOK},
    };

    for (int r = 0; r < BOARD_ROWS; ++r) {
        for (int c = 0; c < BOARD_COLS; ++c) {
            board[r][c] = startup[r][c];
        }
    }

    current_player = RED;
    move_count = 0;
    last_move = -1;
    game_over = false;
    winner = EMPTY;
    final_black_score = 0.0f;
    ring_head_ = 0;
    ring_size_ = 0;
    position_hash_count_ = 0;
    update_history();
    position_hashes_[position_hash_count_++] = compute_hash();
}

XiangqiGame XiangqiGame::copy() const {
    return *this;
}

bool XiangqiGame::in_bounds(int r, int c) {
    return r >= 0 && r < BOARD_ROWS && c >= 0 && c < BOARD_COLS;
}

bool XiangqiGame::in_red_palace(int r, int c) {
    return r >= 7 && r <= 9 && c >= 3 && c <= 5;
}

bool XiangqiGame::in_black_palace(int r, int c) {
    return r >= 0 && r <= 2 && c >= 3 && c <= 5;
}

bool XiangqiGame::in_palace(Stone side, int r, int c) {
    return side == RED ? in_red_palace(r, c) : in_black_palace(r, c);
}

bool XiangqiGame::crossed_river(Stone side, int r) {
    return side == RED ? r <= 4 : r >= 5;
}

int XiangqiGame::piece_type(int8_t piece) {
    if (piece == NO_PIECE) return -1;
    if (piece <= RED_PAWN) return piece - RED_KING;
    return piece - BLACK_KING;
}

Stone XiangqiGame::piece_color(int8_t piece) {
    if (piece >= RED_KING && piece <= RED_PAWN) return RED;
    if (piece >= BLACK_KING && piece <= BLACK_PAWN) return BLACK;
    return EMPTY;
}

bool XiangqiGame::is_side_piece(int8_t piece, Stone side) {
    return piece_color(piece) == side;
}

char XiangqiGame::piece_to_char(int8_t piece) {
    switch (piece) {
        case RED_KING: return 'K';
        case RED_ADVISOR: return 'A';
        case RED_BISHOP: return 'B';
        case RED_KNIGHT: return 'N';
        case RED_ROOK: return 'R';
        case RED_CANNON: return 'C';
        case RED_PAWN: return 'P';
        case BLACK_KING: return 'k';
        case BLACK_ADVISOR: return 'a';
        case BLACK_BISHOP: return 'b';
        case BLACK_KNIGHT: return 'n';
        case BLACK_ROOK: return 'r';
        case BLACK_CANNON: return 'c';
        case BLACK_PAWN: return 'p';
        default: return '.';
    }
}

void XiangqiGame::apply_move_unchecked(int sr, int sc, int dr, int dc, int8_t& captured) {
    captured = board[dr][dc];
    board[dr][dc] = board[sr][sc];
    board[sr][sc] = NO_PIECE;
}

void XiangqiGame::undo_move_unchecked(int sr, int sc, int dr, int dc, int8_t captured) {
    board[sr][sc] = board[dr][dc];
    board[dr][dc] = captured;
}

bool XiangqiGame::is_square_attacked(int r, int c, Stone by) const {
    const int pawn_dir = (by == RED) ? -1 : 1;
    const int pawn_src_r = r - pawn_dir;
    if (in_bounds(pawn_src_r, c) && board[pawn_src_r][c] == (by == RED ? RED_PAWN : BLACK_PAWN))
        return true;
    for (int dc : {-1, 1}) {
        int pc = c + dc;
        if (!in_bounds(r, pc)) continue;
        int8_t piece = board[r][pc];
        if (piece != (by == RED ? RED_PAWN : BLACK_PAWN)) continue;
        if (crossed_river(by, r)) return true;
    }

    for (const auto& off : kKnightOffsets) {
        int sr = r + off[0];
        int sc = c + off[1];
        int leg_r = r + off[2];
        int leg_c = c + off[3];
        if (!in_bounds(sr, sc) || !in_bounds(leg_r, leg_c)) continue;
        if (board[leg_r][leg_c] != NO_PIECE) continue;
        int8_t piece = board[sr][sc];
        if (piece == (by == RED ? RED_KNIGHT : BLACK_KNIGHT)) return true;
    }

    for (const auto& dir : kKingDirs) {
        int rr = r + dir[0];
        int cc = c + dir[1];
        bool seen_screen = false;
        while (in_bounds(rr, cc)) {
            int8_t piece = board[rr][cc];
            if (piece != NO_PIECE) {
                if (!seen_screen) {
                    if (piece == (by == RED ? RED_ROOK : BLACK_ROOK)) return true;
                    if (piece == (by == RED ? RED_KING : BLACK_KING)) return true;
                    seen_screen = true;
                } else {
                    if (piece == (by == RED ? RED_CANNON : BLACK_CANNON)) return true;
                    break;
                }
            }
            rr += dir[0];
            cc += dir[1];
        }
    }

    return false;
}

bool XiangqiGame::is_pseudo_legal(int sr, int sc, int dr, int dc) const {
    if (!in_bounds(sr, sc) || !in_bounds(dr, dc)) return false;
    if (sr == dr && sc == dc) return false;
    int8_t piece = board[sr][sc];
    if (!is_side_piece(piece, current_player)) return false;
    if (is_side_piece(board[dr][dc], current_player)) return false;

    Stone side = current_player;
    int type = piece_type(piece);
    int row_delta = dr - sr;
    int col_delta = dc - sc;
    int abs_row = std::abs(row_delta);
    int abs_col = std::abs(col_delta);

    switch (type) {
        case 0:
            if (board[dr][dc] == (side == RED ? BLACK_KING : RED_KING) && sc == dc) {
                int step = (dr > sr) ? 1 : -1;
                for (int r = sr + step; r != dr; r += step) {
                    if (board[r][sc] != NO_PIECE) return false;
                }
                return true;
            }
            return in_palace(side, dr, dc) && abs_row + abs_col == 1;
        case 1:
            return in_palace(side, dr, dc) && abs_row == 1 && abs_col == 1;
        case 2: {
            if (abs_row != 2 || abs_col != 2) return false;
            if (side == RED && dr < 5) return false;
            if (side == BLACK && dr > 4) return false;
            return board[(sr + dr) / 2][(sc + dc) / 2] == NO_PIECE;
        }
        case 3:
            if (!((abs_row == 2 && abs_col == 1) || (abs_row == 1 && abs_col == 2))) return false;
            if (abs_row == 2) return board[sr + row_delta / 2][sc] == NO_PIECE;
            return board[sr][sc + col_delta / 2] == NO_PIECE;
        case 4:
        case 5: {
            if (sr != dr && sc != dc) return false;
            int step_r = (dr == sr) ? 0 : (dr > sr ? 1 : -1);
            int step_c = (dc == sc) ? 0 : (dc > sc ? 1 : -1);
            int blockers = 0;
            for (int rr = sr + step_r, cc = sc + step_c; rr != dr || cc != dc; rr += step_r, cc += step_c) {
                if (board[rr][cc] != NO_PIECE) blockers++;
            }
            if (type == 4) return blockers == 0;
            if (board[dr][dc] == NO_PIECE) return blockers == 0;
            return blockers == 1;
        }
        case 6: {
            int forward = (side == RED) ? -1 : 1;
            if (row_delta == forward && col_delta == 0) return true;
            if (!crossed_river(side, sr)) return false;
            return row_delta == 0 && abs_col == 1;
        }
        default:
            return false;
    }
}

bool XiangqiGame::is_legal(int action) const {
    if (game_over) return false;
    if (action < 0 || action >= action_size()) return false;
    int src = action_src(action);
    int dst = action_dst(action);
    int sr = src / BOARD_COLS;
    int sc = src % BOARD_COLS;
    int dr = dst / BOARD_COLS;
    int dc = dst % BOARD_COLS;
    if (!is_pseudo_legal(sr, sc, dr, dc)) return false;

    XiangqiGame tmp = *this;
    int8_t captured = NO_PIECE;
    tmp.apply_move_unchecked(sr, sc, dr, dc, captured);
    int king_r = -1, king_c = -1;
    int8_t king_piece = (current_player == RED) ? RED_KING : BLACK_KING;
    for (int r = 0; r < BOARD_ROWS; ++r) {
        for (int c = 0; c < BOARD_COLS; ++c) {
            if (tmp.board[r][c] == king_piece) {
                king_r = r;
                king_c = c;
                break;
            }
        }
        if (king_r >= 0) break;
    }
    if (king_r < 0) return false;
    return !tmp.is_square_attacked(king_r, king_c, opponent(current_player));
}

bool XiangqiGame::has_any_legal_move(Stone side) const {
    XiangqiGame tmp = *this;
    tmp.current_player = side;
    for (int sr = 0; sr < BOARD_ROWS; ++sr) {
        for (int sc = 0; sc < BOARD_COLS; ++sc) {
            if (!is_side_piece(tmp.board[sr][sc], side)) continue;
            int src = sq_index(sr, sc);
            for (int dr = 0; dr < BOARD_ROWS; ++dr) {
                for (int dc = 0; dc < BOARD_COLS; ++dc) {
                    int dst = sq_index(dr, dc);
                    if (tmp.is_legal(encode_action(src, dst))) return true;
                }
            }
        }
    }
    return false;
}

uint64_t XiangqiGame::compute_hash() const {
    uint64_t h = 1469598103934665603ULL;
    for (int r = 0; r < BOARD_ROWS; ++r) {
        for (int c = 0; c < BOARD_COLS; ++c) {
            h ^= static_cast<uint64_t>(board[r][c] + 17 * (r * BOARD_COLS + c + 1));
            h *= 1099511628211ULL;
        }
    }
    h ^= static_cast<uint64_t>(current_player);
    h *= 1099511628211ULL;
    return h;
}

bool XiangqiGame::is_threefold_repetition() const {
    if (position_hash_count_ < 3) return false;
    uint64_t current = position_hashes_[position_hash_count_ - 1];
    int count = 0;
    for (int i = 0; i < position_hash_count_; ++i) {
        if (position_hashes_[i] == current) count++;
    }
    return count >= 3;
}

void XiangqiGame::get_legal_moves(std::vector<float>& legal) const {
    legal.assign(action_size(), 0.0f);
    if (game_over) return;
    for (int sr = 0; sr < BOARD_ROWS; ++sr) {
        for (int sc = 0; sc < BOARD_COLS; ++sc) {
            if (!is_side_piece(board[sr][sc], current_player)) continue;
            int src = sq_index(sr, sc);
            for (int dr = 0; dr < BOARD_ROWS; ++dr) {
                for (int dc = 0; dc < BOARD_COLS; ++dc) {
                    int action = encode_action(src, sq_index(dr, dc));
                    if (is_legal(action)) legal[action] = 1.0f;
                }
            }
        }
    }
}

void XiangqiGame::play(int action) {
    if (!is_legal(action)) {
        throw std::runtime_error("XiangqiGame::play called with illegal action");
    }

    int src = action_src(action);
    int dst = action_dst(action);
    int sr = src / BOARD_COLS;
    int sc = src % BOARD_COLS;
    int dr = dst / BOARD_COLS;
    int dc = dst % BOARD_COLS;

    int8_t captured = NO_PIECE;
    apply_move_unchecked(sr, sc, dr, dc, captured);

    last_move = action;
    move_count++;
    current_player = opponent(current_player);
    update_history();
    if (position_hash_count_ < (int)position_hashes_.size())
        position_hashes_[position_hash_count_++] = compute_hash();

    if (is_threefold_repetition()) {
        game_over = true;
        winner = EMPTY;
        score_game();
        return;
    }

    if (!has_any_legal_move(current_player)) {
        game_over = true;
        winner = opponent(current_player);
        score_game();
        return;
    }

    score_game();
}

void XiangqiGame::force_draw() {
    game_over = true;
    winner = EMPTY;
    score_game();
}

std::pair<float, float> XiangqiGame::score() const {
    float red_score = 0.0f;
    float black_score = 0.0f;
    for (int r = 0; r < BOARD_ROWS; ++r) {
        for (int c = 0; c < BOARD_COLS; ++c) {
            int8_t piece = board[r][c];
            if (piece == NO_PIECE) continue;
            int value = kPieceValues[piece_type(piece)];
            if (piece_color(piece) == RED) red_score += value;
            else black_score += value;
        }
    }
    return {red_score, black_score};
}

void XiangqiGame::get_ownership(Stone player, std::vector<float>& out) const {
    out.assign(BOARD_AREA, 0.0f);
    for (int r = 0; r < BOARD_ROWS; ++r) {
        for (int c = 0; c < BOARD_COLS; ++c) {
            if (piece_color(board[r][c]) == player) {
                out[sq_index(r, c)] = 1.0f;
            }
        }
    }
}

void XiangqiGame::update_history() {
    int write_idx = (ring_head_ + ring_size_) % RING_CAP;
    auto& snap = ring_buf_[write_idx];
    for (int r = 0; r < BOARD_ROWS; ++r) {
        for (int c = 0; c < BOARD_COLS; ++c) {
            snap[sq_index(r, c)] = board[r][c];
        }
    }
    if (ring_size_ < history_length) {
        ring_size_++;
    } else {
        ring_head_ = (ring_head_ + 1) % RING_CAP;
    }
}

void XiangqiGame::encode(std::vector<float>& out) const {
    int planes = history_length * PIECE_PLANES + 1;
    out.assign((size_t)planes * BOARD_AREA, 0.0f);
    for (int i = 0; i < history_length && i < ring_size_; ++i) {
        int slot = (ring_head_ + ring_size_ - 1 - i + RING_CAP) % RING_CAP;
        const auto& snap = ring_buf_[slot];
        int plane_base = i * PIECE_PLANES;
        for (int idx = 0; idx < BOARD_AREA; ++idx) {
            int8_t piece = snap[idx];
            if (piece == NO_PIECE) continue;
            Stone side = piece_color(piece);
            int rel = piece_type(piece) + (side == current_player ? 0 : 7);
            out[(plane_base + rel) * BOARD_AREA + idx] = 1.0f;
        }
    }
    if (current_player == RED) {
        float* color_plane = out.data() + (planes - 1) * BOARD_AREA;
        std::fill(color_plane, color_plane + BOARD_AREA, 1.0f);
    }
}

void XiangqiGame::score_game() {
    auto [red_score, black_score] = score();
    final_black_score = black_score - red_score;
}

std::string XiangqiGame::action_to_str(int action) const {
    if (action < 0 || action >= action_size()) return "????";
    int src = action_src(action);
    int dst = action_dst(action);
    int sr = src / BOARD_COLS;
    int sc = src % BOARD_COLS;
    int dr = dst / BOARD_COLS;
    int dc = dst % BOARD_COLS;
    std::string s;
    s += static_cast<char>('a' + sc);
    s += static_cast<char>('0' + normalized_rank(sr));
    s += static_cast<char>('a' + dc);
    s += static_cast<char>('0' + normalized_rank(dr));
    return s;
}

int XiangqiGame::str_to_action(const std::string& s) const {
    std::string compact;
    compact.reserve(s.size());
    for (char ch : s) {
        if (!std::isspace(static_cast<unsigned char>(ch)) && ch != '-') compact.push_back(ch);
    }
    int sr, sc, dr, dc;
    if (!parse_square(compact, 0, sr, sc) || !parse_square(compact, 2, dr, dc))
        throw std::runtime_error("invalid move string: " + s);
    return encode_action(sq_index(sr, sc), sq_index(dr, dc));
}

std::string XiangqiGame::display() const {
    std::ostringstream ss;
    ss << "    a b c d e f g h i\n";
    for (int r = 0; r < BOARD_ROWS; ++r) {
        ss << ' ' << normalized_rank(r) << "  ";
        for (int c = 0; c < BOARD_COLS; ++c) {
            ss << piece_to_char(board[r][c]);
            if (c + 1 < BOARD_COLS) ss << ' ';
        }
        ss << "  " << normalized_rank(r) << '\n';
        if (r == 4) ss << "    -----------------\n";
    }
    ss << "    a b c d e f g h i\n";
    ss << "Move " << move_count + 1 << " | Turn: "
       << (current_player == RED ? "Red" : "Black");
    if (game_over) {
        if (winner == EMPTY) ss << " | Result: draw";
        else ss << " | Winner: " << (winner == RED ? "Red" : "Black");
    }
    return ss.str();
}

}  // namespace minigo
