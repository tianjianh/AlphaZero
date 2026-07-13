#!/usr/bin/env python3
"""
MiniGo — Selfplay Game Visualizer

Reads .bin / .bin.zst selfplay data files or .sgf evaluation
game files and displays the games move by move.

Usage:
  python visualize.py training/selfplay/iter_0001/game_0.bin.zst
  python visualize.py training/eval/iter_0006/game_0.sgf
  python visualize.py training/selfplay/iter_0001/game_0.bin --game 3
  python visualize.py training/eval/iter_0006/  # all SGFs in a dir
"""

import argparse
import curses
import glob
import os
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
# V3 selfplay game reader (moves stored directly — no state-diff
# reconstruction; see scripts/gamedata.py for the format)
# ═══════════════════════════════════════════════════════════

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gamedata import parse_v3  # noqa: E402


def read_v3_game(filepath):
    """Read a V3 game record (.bin or .bin.zst) → gamedata.GameV3."""
    if filepath.endswith(".zst"):
        import zstandard as zstd
        with open(filepath, "rb") as f:
            with zstd.ZstdDecompressor().stream_reader(f) as reader:
                data = reader.read()
    else:
        with open(filepath, "rb") as f:
            data = f.read()
    return parse_v3(data)


def moves_from_v3(game):
    """Convert a GameV3 into the viewer's move list:
    (color, r_or_None, c_or_None, policy, value, score, ownership).

    value/score are the per-move training targets derived exactly the
    way the trainer derives them (winner/black_score from the mover's
    perspective); ownership is the final ternary map from the mover's
    perspective (1.0 = mover owns).
    """
    n = game.board_size
    hw = n * n
    moves = []
    for m in range(game.n_moves):
        player = BLACK if (m % 2 == 0) else WHITE
        if game.winner == 0:
            value = 0.0
        else:
            value = 1.0 if game.winner == player else -1.0
        score = game.black_score if player == BLACK else -game.black_score
        ownership = tuple(1.0 if o == player else 0.0 for o in game.owners)
        a = int(game.actions[m])
        if a < 0 or a >= hw:
            moves.append((player, None, None, tuple(game.policies[m]),
                          value, score, ownership))
        else:
            moves.append((player, a // n, a % n, tuple(game.policies[m]),
                          value, score, ownership))
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
        parts = []
        if len(m) > 4: parts.append(f"V={m[4]:+.2f}")
        if len(m) > 5: parts.append(f"S={m[5]:+.1f}")
        # ownership info is at index 6 if present
        extra = "  " + " ".join(parts) if parts else ""
        return m[0], m[1], m[2], extra


def _replay_to(moves_data, board_size, target, is_sgf):
    """Replay the game from move 0 to target, return board."""
    board = make_board(board_size)
    for j in range(min(target, len(moves_data))):
        color, r, c, _ = _get_move_info(moves_data, j, is_sgf)
        if r is not None:
            board[r][c] = color
            find_captures(board, board_size, r, c)
    return board


def view_game_curses(board_size, moves_data, title="", is_sgf=False):
    """Curses-based game viewer. Left/Right arrows, Q to quit."""

    def is_star(r, c, n):
        if n == 9:  return (r==2 or r==6) and (c==2 or c==6) or (r==4 and c==4)
        if n == 13: return (r==3 or r==9) and (c==3 or c==9) or (r==6 and c==6)
        if n == 19: return (r==3 or r==15) and (c==3 or c==15) or (r==9 and c==9)
        return False

    # ACS char map (set after initscr)
    ACS = {}

    def draw(stdscr, board, move_idx, last_r, last_c):
        stdscr.erase()
        h, w = stdscr.getmaxyx()
        n = board_size

        # Title
        stdscr.attron(curses.color_pair(3) | curses.A_BOLD)
        stdscr.addnstr(0, 2, title.split('\n')[0], w - 4)
        stdscr.attroff(curses.color_pair(3) | curses.A_BOLD)

        ox, oy = 5, 2
        cell_w = 2

        # Column labels
        stdscr.attron(curses.color_pair(5))
        for c in range(n):
            stdscr.addch(oy - 1, ox + c * cell_w, COLS[c])
        stdscr.attroff(curses.color_pair(5))

        # Grid + stones
        for r in range(n):
            y = oy + r
            # Row label
            stdscr.attron(curses.color_pair(5))
            stdscr.addstr(y, ox - 3, f"{n-r:2d}")
            stdscr.attroff(curses.color_pair(5))

            for c in range(n):
                x = ox + c * cell_w
                cell = board[r][c]

                # Grid char
                if r == 0:
                    gc = ACS['ul'] if c == 0 else ACS['ur'] if c == n-1 else ACS['tt']
                elif r == n-1:
                    gc = ACS['ll'] if c == 0 else ACS['lr'] if c == n-1 else ACS['bt']
                else:
                    gc = ACS['lt'] if c == 0 else ACS['rt'] if c == n-1 else ACS['pl']

                if cell == BLACK:
                    stdscr.attron(curses.color_pair(6) | curses.A_BOLD)
                    stdscr.addch(y, x, ord('X'))
                    stdscr.attroff(curses.color_pair(6) | curses.A_BOLD)
                elif cell == WHITE:
                    stdscr.attron(curses.color_pair(7) | curses.A_BOLD)
                    stdscr.addch(y, x, ord('O'))
                    stdscr.attroff(curses.color_pair(7) | curses.A_BOLD)
                elif is_star(r, c, n):
                    stdscr.attron(curses.color_pair(1))
                    stdscr.addch(y, x, ord('*'))
                    stdscr.attroff(curses.color_pair(1))
                else:
                    stdscr.attron(curses.color_pair(1))
                    stdscr.addch(y, x, gc)
                    stdscr.attroff(curses.color_pair(1))

                # Horizontal connector
                if c < n - 1:
                    stdscr.attron(curses.color_pair(1))
                    stdscr.addch(y, x + 1, ACS['hl'])
                    stdscr.attroff(curses.color_pair(1))

            # Row label right
            stdscr.attron(curses.color_pair(5))
            try:
                stdscr.addstr(y, ox + (n-1) * cell_w + 2, str(n - r))
            except curses.error:
                pass
            stdscr.attroff(curses.color_pair(5))

        # ── Right-side info panel ──
        px = ox + n * cell_w + 4
        py = oy

        # Move counter
        total = len(moves_data)
        stdscr.attron(curses.color_pair(2))
        stdscr.addnstr(py, px, f"Move {move_idx}/{total}", w - px - 1)
        stdscr.attroff(curses.color_pair(2))
        py += 2

        # Last move info
        if move_idx > 0:
            color, r, c, extra = _get_move_info(moves_data, move_idx - 1, is_sgf)
            name = "Black" if color == BLACK else "White"
            coord = "PASS" if r is None else coord_to_str(r, c, board_size)
            stdscr.attron(curses.color_pair(2))
            stdscr.addnstr(py, px, f"{name} plays {coord}", w - px - 1)
            stdscr.attroff(curses.color_pair(2))
            py += 1
            if extra.strip():
                stdscr.attron(curses.color_pair(8))
                stdscr.addnstr(py, px, extra.strip(), w - px - 1)
                stdscr.attroff(curses.color_pair(8))
                py += 1
        py += 1

        # Title info (remaining lines)
        for line in title.split('\n')[1:]:
            line = line.strip()
            if line and py < h - 2:
                stdscr.attron(curses.color_pair(5))
                stdscr.addnstr(py, px, line, w - px - 1)
                stdscr.attroff(curses.color_pair(5))
                py += 1

        # Help
        help_y = max(oy + n, py + 1)
        stdscr.attron(curses.color_pair(5))
        try:
            stdscr.addnstr(min(help_y, h-1), 2,
                           "Left/Right: step  Home/End: jump  Q: quit", w - 4)
        except curses.error:
            pass
        stdscr.attroff(curses.color_pair(5))

        stdscr.refresh()

    def run(stdscr):
        curses.curs_set(0)
        curses.use_default_colors()
        curses.init_pair(1, 237, -1)   # grid
        curses.init_pair(2, 252, -1)   # status
        curses.init_pair(3, 214, -1)   # accent
        curses.init_pair(4, 252, -1)   # last move brackets
        curses.init_pair(5, 245, -1)   # labels
        curses.init_pair(6, 255, -1)   # black stone
        curses.init_pair(7, 252, -1)   # white stone
        curses.init_pair(8, 81,  -1)   # info

        ACS['ul'] = curses.ACS_ULCORNER
        ACS['ur'] = curses.ACS_URCORNER
        ACS['ll'] = curses.ACS_LLCORNER
        ACS['lr'] = curses.ACS_LRCORNER
        ACS['tt'] = curses.ACS_TTEE
        ACS['bt'] = curses.ACS_BTEE
        ACS['lt'] = curses.ACS_LTEE
        ACS['rt'] = curses.ACS_RTEE
        ACS['pl'] = curses.ACS_PLUS
        ACS['hl'] = curses.ACS_HLINE

        move_idx = 0
        board = make_board(board_size)
        last_r, last_c = -1, -1

        while True:
            draw(stdscr, board, move_idx, last_r, last_c)
            key = stdscr.getch()

            if key in (ord('q'), ord('Q')):
                break

            elif key == curses.KEY_RIGHT or key == ord(' ') or key == 10:
                if move_idx < len(moves_data):
                    color, r, c, _ = _get_move_info(moves_data, move_idx, is_sgf)
                    if r is not None:
                        board[r][c] = color
                        find_captures(board, board_size, r, c)
                        last_r, last_c = r, c
                    else:
                        last_r, last_c = -1, -1
                    move_idx += 1

            elif key == curses.KEY_LEFT:
                if move_idx > 0:
                    move_idx -= 1
                    board = _replay_to(moves_data, board_size, move_idx, is_sgf)
                    if move_idx > 0:
                        _, lr, lc, _ = _get_move_info(moves_data, move_idx - 1, is_sgf)
                        last_r, last_c = lr if lr is not None else -1, lc if lc is not None else -1
                    else:
                        last_r, last_c = -1, -1

            elif key == curses.KEY_HOME:
                move_idx = 0
                board = make_board(board_size)
                last_r, last_c = -1, -1

            elif key == curses.KEY_END:
                move_idx = len(moves_data)
                board = _replay_to(moves_data, board_size, move_idx, is_sgf)
                if move_idx > 0:
                    _, lr, lc, _ = _get_move_info(moves_data, move_idx - 1, is_sgf)
                    last_r, last_c = lr if lr is not None else -1, lc if lc is not None else -1

    curses.wrapper(run)


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
        return

    view_single_file(path, args.board, args.game)


def view_single_file(path, board_size, game_idx):
    if path.endswith(".sgf"):
        bs, komi, moves, result, pb, pw = read_sgf(path)
        title = (f"{os.path.basename(path)}\n"
                 f"  Black: {pb}  vs  White: {pw}\n"
                 f"  Komi: {komi}  Result: {result}")
        view_game_curses(bs, moves, title=title, is_sgf=True)
    elif ".bin" in path:
        game = read_v3_game(path)
        moves = moves_from_v3(game)
        outcome = ("Black wins" if game.winner == 1 else
                   "White wins" if game.winner == 2 else "Draw")
        title = (f"{os.path.basename(path)}\n"
                 f"  {len(moves)} moves  {outcome}  komi {game.komi:g}\n"
                 f"  Score (black): {game.black_score:+.1f} pts")
        view_game_curses(game.board_size, moves, title=title, is_sgf=False)
    else:
        print(f"Unknown file format: {path}")


if __name__ == "__main__":
    main()
