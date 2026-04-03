#!/usr/bin/env python3
"""
MiniGo — Selfplay Game Visualizer

Reads .bin / .bin.zst / .bin.gz selfplay data files or .sgf evaluation
game files and displays the games move by move.

Usage:
  python visualize.py training/selfplay/iter_0001/game_0.bin.zst
  python visualize.py training/eval/iter_0006/game_0.sgf
  python visualize.py training/selfplay/iter_0001/game_0.bin --game 3
  python visualize.py training/eval/iter_0006/  # all SGFs in a dir
"""

import argparse
import glob
import os
import struct
import sys


# ═══════════════════════════════════════════════════════════
# Go board display
# ═══════════════════════════════════════════════════════════

EMPTY, BLACK, WHITE = 0, 1, 2
COLS = "ABCDEFGHJKLMNOPQRSTUVWXYZ"


def make_board(size):
    return [[EMPTY] * size for _ in range(size)]


def display_board(board, size):
    symbols = {EMPTY: '.', BLACK: 'X', WHITE: 'O'}
    header = "   " + " ".join(COLS[c] for c in range(size))
    lines = [header]
    for r in range(size):
        row_num = size - r
        row = " ".join(symbols[board[r][c]] for c in range(size))
        lines.append(f"{row_num:2d} {row}  {row_num}")
    lines.append(header)
    return "\n".join(lines)


def coord_to_str(r, c, size):
    return f"{COLS[c]}{size - r}"


def find_captures(board, size, r, c):
    """Find and remove captured groups adjacent to (r,c)."""
    color = board[r][c]
    opp = BLACK if color == WHITE else WHITE
    captured = []
    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        nr, nc = r + dr, c + dc
        if 0 <= nr < size and 0 <= nc < size and board[nr][nc] == opp:
            group, liberties = flood_group(board, size, nr, nc)
            if liberties == 0:
                captured.extend(group)
                for gr, gc in group:
                    board[gr][gc] = EMPTY
    return captured


def flood_group(board, size, r, c):
    """Flood fill to find a group and count its liberties."""
    color = board[r][c]
    visited = set()
    stack = [(r, c)]
    group = []
    liberties = 0
    while stack:
        cr, cc = stack.pop()
        if (cr, cc) in visited:
            continue
        visited.add((cr, cc))
        group.append((cr, cc))
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = cr + dr, cc + dc
            if 0 <= nr < size and 0 <= nc < size:
                if board[nr][nc] == color and (nr, nc) not in visited:
                    stack.append((nr, nc))
                elif board[nr][nc] == EMPTY:
                    liberties += 1
    return group, liberties


# ═══════════════════════════════════════════════════════════
# Binary selfplay data reader
# ═══════════════════════════════════════════════════════════

def decompress(filepath):
    if filepath.endswith(".zst"):
        import zstandard as zstd
        with open(filepath, "rb") as f:
            return zstd.ZstdDecompressor().decompress(f.read())
    elif filepath.endswith(".gz"):
        import gzip
        with gzip.open(filepath, "rb") as f:
            return f.read()
    else:
        with open(filepath, "rb") as f:
            return f.read()


def read_bin_file(filepath):
    """Read selfplay .bin file, return list of (state, policy, value) records."""
    data = decompress(filepath)
    pos = 0
    n = struct.unpack_from("i", data, pos)[0]; pos += 4
    records = []
    for _ in range(n):
        ss = struct.unpack_from("i", data, pos)[0]; pos += 4
        state = struct.unpack_from(f"{ss}f", data, pos); pos += ss * 4
        ps = struct.unpack_from("i", data, pos)[0]; pos += 4
        policy = struct.unpack_from(f"{ps}f", data, pos); pos += ps * 4
        value = struct.unpack_from("f", data, pos)[0]; pos += 4
        records.append((state, policy, value))
    return records


def extract_games_from_bin(records, board_size):
    """Group augmented records into games (8 augmentations per position).

    Returns list of games. Each game is a list of (board_state, policy, value).
    The first augmentation (identity) is used for display.
    """
    # Records come in groups of 8 (dihedral augmentation)
    moves_per_game = len(records) // 8
    if moves_per_game == 0:
        return []

    game = []
    for i in range(0, len(records), 8):
        state, policy, value = records[i]  # identity augmentation
        game.append((state, policy, value))

    return [game]  # one game per .bin file


def replay_game_from_bin(game, board_size):
    """Extract the move sequence from binary state data by diffing boards."""
    input_ch = 17
    hw = board_size * board_size
    moves = []

    for i, (state, policy, value) in enumerate(game):
        # State layout: [ch0..ch16] where ch0 = current player's stones,
        # ch1 = opponent's stones (current frame)
        # The color plane (ch16) tells us who is playing: 1.0 = black
        color_plane = state[16 * hw]
        is_black = (color_plane > 0.5)
        current = BLACK if is_black else WHITE

        # Find the action from policy (highest probability)
        action_size = board_size * board_size + 1
        best_action = max(range(action_size), key=lambda a: policy[a])

        if best_action == board_size * board_size:
            moves.append((current, None, None, policy, value))  # pass
        else:
            r, c = best_action // board_size, best_action % board_size
            moves.append((current, r, c, policy, value))

    return moves


# ═══════════════════════════════════════════════════════════
# SGF reader (for evaluation games)
# ═══════════════════════════════════════════════════════════

def read_sgf(filepath):
    """Parse a simple SGF file. Returns (board_size, komi, moves, result, props)."""
    with open(filepath) as f:
        content = f.read()

    # Extract properties
    import re
    props = {}
    for m in re.finditer(r'(\w+)\[([^\]]*)\]', content):
        key, val = m.group(1), m.group(2)
        props[key] = val

    board_size = int(props.get("SZ", "9"))
    komi = float(props.get("KM", "7.5"))
    result = props.get("RE", "?")
    black_name = props.get("PB", "?")
    white_name = props.get("PW", "?")

    # Extract moves
    moves = []
    for m in re.finditer(r';([BW])\[([^\]]*)\]', content):
        color = BLACK if m.group(1) == "B" else WHITE
        coord = m.group(2)
        if coord == "" or coord == "tt":  # pass
            moves.append((color, None, None))
        else:
            c = ord(coord[0]) - ord('a')
            r = ord(coord[1]) - ord('a')
            moves.append((color, r, c))

    return board_size, komi, moves, result, black_name, white_name


# ═══════════════════════════════════════════════════════════
# Interactive viewer
# ═══════════════════════════════════════════════════════════

def _get_move_info(moves_data, idx, is_sgf):
    """Extract (color, r, c, extra_str) from a move."""
    if is_sgf:
        color, r, c = moves_data[idx][:3]
        return color, r, c, ""
    else:
        m = moves_data[idx]
        val_str = f"  value={m[4]:.2f}" if len(m) > 4 else ""
        return m[0], m[1], m[2], val_str


def _replay_to(moves_data, board_size, target, is_sgf):
    """Replay the game from move 0 to target, return board."""
    board = make_board(board_size)
    for j in range(min(target, len(moves_data))):
        color, r, c, _ = _get_move_info(moves_data, j, is_sgf)
        if r is not None:
            board[r][c] = color
            find_captures(board, board_size, r, c)
    return board


def view_game_interactive(board_size, moves_data, title="", is_sgf=False):
    """Step through a game move by move with forward/backward navigation."""
    print(f"\n{'=' * 50}")
    print(f"  {title}")
    print(f"  Board: {board_size}x{board_size}  Moves: {len(moves_data)}")
    print(f"{'=' * 50}")

    board = make_board(board_size)
    print(display_board(board, board_size))
    print("\n[Enter]=next  [b]=back  [s]=skip to end  [number]=jump  [q]=quit")

    move_idx = 0
    while True:
        if move_idx >= len(moves_data):
            prompt = "Game over"
        else:
            prompt = f"Move {move_idx + 1}/{len(moves_data)}"
        try:
            cmd = input(f"{prompt}: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return

        if cmd == 'q':
            return

        elif cmd == 'b' or cmd == 'p':
            # Go back one move
            if move_idx > 0:
                move_idx -= 1
            board = _replay_to(moves_data, board_size, move_idx, is_sgf)
            if move_idx > 0:
                color, r, c, extra = _get_move_info(moves_data, move_idx - 1, is_sgf)
                color_name = "Black X" if color == BLACK else "White O"
                if r is None:
                    print(f"  {color_name} plays PASS{extra}")
                else:
                    print(f"  {color_name} plays {coord_to_str(r, c, board_size)}{extra}")
            print(display_board(board, board_size))

        elif cmd == 's':
            move_idx = len(moves_data)
            board = _replay_to(moves_data, board_size, move_idx, is_sgf)
            print(display_board(board, board_size))

        elif cmd.isdigit():
            move_idx = max(0, min(int(cmd), len(moves_data)))
            board = _replay_to(moves_data, board_size, move_idx, is_sgf)
            if move_idx > 0:
                color, r, c, extra = _get_move_info(moves_data, move_idx - 1, is_sgf)
                color_name = "Black X" if color == BLACK else "White O"
                if r is None:
                    print(f"  {color_name} plays PASS{extra}")
                else:
                    print(f"  {color_name} plays {coord_to_str(r, c, board_size)}{extra}")
            print(display_board(board, board_size))

        else:
            # Default: next move
            if move_idx >= len(moves_data):
                print("Game over. [b]=back  [number]=jump  [q]=quit")
                continue

            color, r, c, extra = _get_move_info(moves_data, move_idx, is_sgf)
            color_name = "Black X" if color == BLACK else "White O"

            if r is None:
                print(f"  {color_name} plays PASS{extra}")
            else:
                board[r][c] = color
                captures = find_captures(board, board_size, r, c)
                cap_str = f"  captures {len(captures)}" if captures else ""
                print(f"  {color_name} plays {coord_to_str(r, c, board_size)}{extra}{cap_str}")

            move_idx += 1
            print(display_board(board, board_size))


# ═══════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Visualize selfplay (.bin/.bin.zst) or evaluation (.sgf) games",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  python visualize.py training/selfplay/iter_0001/game_0.bin.zst
  python visualize.py training/eval/iter_0006/game_0.sgf
  python visualize.py training/eval/iter_0006/          # all SGFs in dir
  python visualize.py game.bin --board 9 --game 0
""")
    parser.add_argument("path", help="File or directory to visualize")
    parser.add_argument("--board", type=int, default=9,
                        help="Board size (for .bin files; default: 9)")
    parser.add_argument("--game", type=int, default=0,
                        help="Game index within .bin file (default: 0)")
    args = parser.parse_args()

    path = args.path

    # If directory, find all SGF/bin files
    if os.path.isdir(path):
        files = sorted(glob.glob(os.path.join(path, "*.sgf")))
        if not files:
            files = sorted(glob.glob(os.path.join(path, "*.bin*")))
        if not files:
            print(f"No game files found in {path}")
            return
        print(f"Found {len(files)} game files in {path}")
        for f in files:
            view_single_file(f, args.board, args.game)
            try:
                cmd = input("\n[Enter]=next file  [q]=quit: ").strip()
                if cmd == 'q':
                    return
            except (EOFError, KeyboardInterrupt):
                return
        return

    view_single_file(path, args.board, args.game)


def view_single_file(path, board_size, game_idx):
    if path.endswith(".sgf"):
        bs, komi, moves, result, pb, pw = read_sgf(path)
        title = f"{os.path.basename(path)}  B={pb} W={pw}  Result={result}"
        view_game_interactive(bs, moves, title=title, is_sgf=True)
    elif ".bin" in path:
        records = read_bin_file(path)
        game = extract_games_from_bin(records, board_size)
        if not game:
            print(f"No games in {path}")
            return
        g = game[0]
        moves = replay_game_from_bin(g, board_size)
        result_val = g[0][2]  # value from first position
        title = (f"{os.path.basename(path)}  "
                 f"{len(moves)} moves  value={result_val:+.2f}")
        view_game_interactive(board_size, moves, title=title, is_sgf=False)
    else:
        print(f"Unknown file format: {path}")


if __name__ == "__main__":
    main()
