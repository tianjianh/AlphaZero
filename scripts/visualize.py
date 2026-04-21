#!/usr/bin/env python3
"""
Visualize Xiangqi self-play and evaluation games.

Supported inputs:
- self-play records: .bin / .bin.gz / .bin.zst (V4 Xiangqi format)
- evaluation records: .txt files produced by ./build/evaluate --output
"""

from __future__ import annotations

import argparse
import curses
import glob
import os
import struct
from dataclasses import dataclass


BOARD_ROWS = 10
BOARD_COLS = 9
BOARD_AREA = BOARD_ROWS * BOARD_COLS
PIECE_TYPES = "KABNRCP"
AUGMENTATIONS_PER_POSITION = 2


@dataclass
class TrainingRecord:
    state: tuple[float, ...]
    policy: tuple[float, ...]
    value: float
    ownership: tuple[float, ...]
    opponent_action: int


@dataclass
class GameView:
    rows: int
    cols: int
    snapshots: list[list[str]]
    moves: list[dict]
    title_lines: list[str]


def decompress(filepath: str) -> bytes:
    if filepath.endswith(".zst"):
        import zstandard as zstd

        with open(filepath, "rb") as f:
            return zstd.ZstdDecompressor().decompress(f.read())
    if filepath.endswith(".gz"):
        import gzip

        with gzip.open(filepath, "rb") as f:
            return f.read()
    with open(filepath, "rb") as f:
        return f.read()


def startup_board(rows: int, cols: int) -> list[str]:
    if rows != BOARD_ROWS or cols != BOARD_COLS:
        raise ValueError(f"Unsupported Xiangqi board size {rows}x{cols}")
    grid = [
        list("rnbackbnr"),
        list("........."),
        list(".c.....c."),
        list("p.p.p.p.p"),
        list("........."),
        list("........."),
        list("P.P.P.P.P"),
        list(".C.....C."),
        list("........."),
        list("RNBACKBNR"),
    ]
    return [cell for row in grid for cell in row]


def player_label(player: str) -> str:
    return "Red" if player == "red" else "Black"


def opponent(player: str) -> str:
    return "black" if player == "red" else "red"


def piece_player(piece: str) -> str | None:
    if piece == ".":
        return None
    return "red" if piece.isupper() else "black"


def action_to_iccs(src: int, dst: int, rows: int, cols: int) -> str:
    sr, sc = divmod(src, cols)
    dr, dc = divmod(dst, cols)
    return f"{chr(ord('a') + sc)}{rows - 1 - sr}{chr(ord('a') + dc)}{rows - 1 - dr}"


def parse_square(token: str, offset: int, rows: int, cols: int) -> tuple[int, int]:
    file_ch = token[offset].lower()
    rank_ch = token[offset + 1]
    if not ("a" <= file_ch < chr(ord("a") + cols)):
        raise ValueError(f"invalid file in move: {token}")
    if not rank_ch.isdigit():
        raise ValueError(f"invalid rank in move: {token}")
    col = ord(file_ch) - ord("a")
    row = rows - 1 - int(rank_ch)
    if row < 0 or row >= rows:
        raise ValueError(f"invalid rank in move: {token}")
    return row, col


def parse_iccs_move(token: str, rows: int, cols: int) -> tuple[int, int]:
    compact = "".join(ch for ch in token if not ch.isspace() and ch != "-")
    if len(compact) != 4:
        raise ValueError(f"invalid ICCS move: {token}")
    sr, sc = parse_square(compact, 0, rows, cols)
    dr, dc = parse_square(compact, 2, rows, cols)
    return sr * cols + sc, dr * cols + dc


def read_selfplay_file(filepath: str) -> tuple[int, int, list[TrainingRecord]]:
    data = decompress(filepath)
    if len(data) < 16:
        raise ValueError("file is too short to be a Xiangqi V4 self-play file")

    magic, version, count, rows, cols = struct.unpack_from("<HHiii", data, 0)
    if magic != 0x4D47 or version != 4:
        raise ValueError("unsupported self-play format; expected Xiangqi V4 records")

    pos = 16
    records = []
    for _ in range(count):
        state_size = struct.unpack_from("<i", data, pos)[0]
        pos += 4
        state = struct.unpack_from(f"<{state_size}f", data, pos)
        pos += state_size * 4

        policy_size = struct.unpack_from("<i", data, pos)[0]
        pos += 4
        policy = struct.unpack_from(f"<{policy_size}f", data, pos)
        pos += policy_size * 4

        value = struct.unpack_from("<f", data, pos)[0]
        pos += 4

        ownership_size = rows * cols
        ownership = struct.unpack_from(f"<{ownership_size}f", data, pos)
        pos += ownership_size * 4

        opponent_action = struct.unpack_from("<i", data, pos)[0]
        pos += 4

        records.append(
            TrainingRecord(
                state=state,
                policy=policy,
                value=value,
                ownership=ownership,
                opponent_action=opponent_action,
            )
        )

    return rows, cols, records


def extract_identity_records(records: list[TrainingRecord]) -> list[TrainingRecord]:
    if len(records) >= AUGMENTATIONS_PER_POSITION and len(records) % AUGMENTATIONS_PER_POSITION == 0:
        return records[::AUGMENTATIONS_PER_POSITION]
    return records


def decode_board_from_state(state: tuple[float, ...], rows: int, cols: int) -> tuple[list[str], str]:
    area = rows * cols
    channels = len(state) // area
    if channels < 15:
        raise ValueError("state tensor does not look like a Xiangqi history encoding")

    current_is_red = state[(channels - 1) * area] > 0.5
    current_pieces = PIECE_TYPES if current_is_red else PIECE_TYPES.lower()
    opponent_pieces = PIECE_TYPES.lower() if current_is_red else PIECE_TYPES

    board = ["."] * area
    for sq in range(area):
        for idx, piece in enumerate(current_pieces):
            if state[idx * area + sq] > 0.5:
                board[sq] = piece
                break
        if board[sq] != ".":
            continue
        for idx, piece in enumerate(opponent_pieces):
            if state[(7 + idx) * area + sq] > 0.5:
                board[sq] = piece
                break

    return board, ("red" if current_is_red else "black")


def infer_action_from_states(before: list[str], after: list[str], player: str) -> tuple[int, int] | None:
    player_squares = {piece for piece in before if piece_player(piece) == player}
    if not player_squares:
        return None

    src_candidates = [
        sq for sq, piece in enumerate(before)
        if piece_player(piece) == player and after[sq] == "."
    ]
    dst_candidates = [
        sq for sq, piece in enumerate(after)
        if piece_player(piece) == player and before[sq] != piece
    ]

    if len(src_candidates) == 1 and len(dst_candidates) == 1:
        return src_candidates[0], dst_candidates[0]

    changed_from = [
        sq for sq, piece in enumerate(before)
        if piece_player(piece) == player and before[sq] != after[sq]
    ]
    changed_to = [
        sq for sq, piece in enumerate(after)
        if piece_player(piece) == player and before[sq] != after[sq]
    ]
    for src in changed_from:
        piece = before[src]
        for dst in changed_to:
            if src != dst and after[dst] == piece:
                return src, dst
    return None


def action_from_policy(policy: tuple[float, ...], rows: int, cols: int) -> tuple[int, int]:
    area = rows * cols
    best_action = max(range(area * area), key=lambda action: policy[action])
    return best_action // area, best_action % area


def apply_action(board: list[str], src: int, dst: int) -> tuple[list[str], str, str]:
    next_board = board.copy()
    piece = next_board[src]
    captured = next_board[dst]
    next_board[dst] = piece
    next_board[src] = "."
    return next_board, piece, captured


def load_selfplay_view(path: str) -> GameView:
    rows, cols, all_records = read_selfplay_file(path)
    records = extract_identity_records(all_records)
    if not records:
        raise ValueError("self-play file contains no records")

    decoded = [decode_board_from_state(record.state, rows, cols) for record in records]
    board = decoded[0][0].copy()
    snapshots = [board.copy()]
    moves = []

    for idx, record in enumerate(records):
        player = decoded[idx][1]
        action = None
        if idx + 1 < len(decoded):
            action = infer_action_from_states(decoded[idx][0], decoded[idx + 1][0], player)
        if action is None:
            action = action_from_policy(record.policy, rows, cols)

        src, dst = action
        board, piece, captured = apply_action(board, src, dst)
        snapshots.append(board.copy())
        moves.append(
            {
                "player": player,
                "src": src,
                "dst": dst,
                "move": action_to_iccs(src, dst, rows, cols),
                "piece": piece,
                "captured": captured,
                "value": record.value,
            }
        )

    first_value = records[0].value
    if first_value > 0:
        outcome = "Red wins"
    elif first_value < 0:
        outcome = "Black wins"
    else:
        outcome = "Draw"

    title_lines = [
        os.path.basename(path),
        f"Self-play  {rows}x{cols}  {len(moves)} moves  {outcome}",
        f"Records: {len(all_records)} total, {len(records)} identity samples",
    ]
    return GameView(rows=rows, cols=cols, snapshots=snapshots, moves=moves, title_lines=title_lines)


def load_eval_view(path: str) -> GameView:
    fields: dict[str, str] = {}
    with open(path, encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue
            key, _, value = line.partition(" ")
            fields[key] = value

    rows, cols = map(int, fields["board"].split("x"))
    board = startup_board(rows, cols)
    snapshots = [board.copy()]
    moves = []
    player = "red"

    for move_text in fields.get("moves", "").split():
        src, dst = parse_iccs_move(move_text, rows, cols)
        board, piece, captured = apply_action(board, src, dst)
        snapshots.append(board.copy())
        moves.append(
            {
                "player": player,
                "src": src,
                "dst": dst,
                "move": move_text,
                "piece": piece,
                "captured": captured,
                "value": None,
            }
        )
        player = opponent(player)

    result = fields.get("result", "?")
    if result == "0":
        result_text = "Draw"
    elif result == "1":
        result_text = "Model 1 wins"
    elif result == "-1":
        result_text = "Model 2 wins"
    else:
        result_text = f"Result {result}"

    title_lines = [
        os.path.basename(path),
        f"Black: {fields.get('black', '?')}  Red: {fields.get('red', '?')}",
        result_text,
    ]
    return GameView(rows=rows, cols=cols, snapshots=snapshots, moves=moves, title_lines=title_lines)


def load_view(path: str) -> GameView:
    if path.endswith(".txt"):
        return load_eval_view(path)
    if ".bin" in path:
        return load_selfplay_view(path)
    raise ValueError(f"unsupported file type: {path}")


def draw_game(stdscr, view: GameView) -> None:
    rows = view.rows
    cols = view.cols
    files = "abcdefghi"

    curses.curs_set(0)
    curses.use_default_colors()
    curses.init_pair(1, 244, -1)   # grid
    curses.init_pair(2, 252, -1)   # status
    curses.init_pair(3, 214, -1)   # accent
    curses.init_pair(4, 203, -1)   # red pieces
    curses.init_pair(5, 81, -1)    # black pieces
    curses.init_pair(6, 46, -1)    # highlight
    curses.init_pair(7, 245, -1)   # labels
    curses.init_pair(8, 117, -1)   # extra info

    move_idx = 0

    while True:
        stdscr.erase()
        h, w = stdscr.getmaxyx()
        board = view.snapshots[move_idx]
        last_move = view.moves[move_idx - 1] if move_idx > 0 else None

        title = view.title_lines[0]
        stdscr.attron(curses.color_pair(3) | curses.A_BOLD)
        stdscr.addnstr(0, max(0, (w - len(title)) // 2), title, max(0, w - 2))
        stdscr.attroff(curses.color_pair(3) | curses.A_BOLD)

        ox = 4
        oy = 3
        panel_x = 34

        status = f"Move {move_idx}/{len(view.moves)}"
        stdscr.attron(curses.color_pair(2))
        stdscr.addnstr(1, max(0, (w - len(status)) // 2), status, max(0, w - 2))
        stdscr.attroff(curses.color_pair(2))

        stdscr.attron(curses.color_pair(7))
        for c in range(cols):
            stdscr.addch(oy - 1, ox + 2 + c * 3, files[c])
        stdscr.attroff(curses.color_pair(7))

        for r in range(rows):
            y = oy + r + (1 if r >= 5 else 0)
            rank = str(rows - 1 - r)
            stdscr.attron(curses.color_pair(7))
            stdscr.addnstr(y, ox, rank, 2)
            stdscr.addnstr(y, ox + 30, rank, 2)
            stdscr.attroff(curses.color_pair(7))

            for c in range(cols):
                x = ox + 2 + c * 3
                sq = r * cols + c
                piece = board[sq]
                highlight = last_move is not None and sq == last_move["dst"]

                if piece == ".":
                    stdscr.attron(curses.color_pair(1))
                    stdscr.addch(y, x, ord("."))
                    stdscr.attroff(curses.color_pair(1))
                else:
                    color_pair = 4 if piece.isupper() else 5
                    attrs = curses.color_pair(color_pair) | curses.A_BOLD
                    if highlight:
                        attrs |= curses.A_REVERSE
                    stdscr.attron(attrs)
                    stdscr.addch(y, x, ord(piece))
                    stdscr.attroff(attrs)

                if c < cols - 1:
                    stdscr.attron(curses.color_pair(1))
                    stdscr.addstr(y, x + 1, "--")
                    stdscr.attroff(curses.color_pair(1))
                if r < rows - 1 and r != 4:
                    stdscr.attron(curses.color_pair(1))
                    stdscr.addch(y + 1, x, ord("|"))
                    stdscr.attroff(curses.color_pair(1))

        stdscr.attron(curses.color_pair(7))
        stdscr.addstr(oy + 5, ox + 6, "Chu He")
        stdscr.addstr(oy + 5, ox + 17, "Han Jie")
        stdscr.attroff(curses.color_pair(7))

        info_y = oy
        for line in view.title_lines[1:]:
            if info_y < h - 1:
                stdscr.attron(curses.color_pair(7))
                stdscr.addnstr(info_y, panel_x, line, max(0, w - panel_x - 2))
                stdscr.attroff(curses.color_pair(7))
                info_y += 1

        info_y += 1
        if last_move is not None and info_y < h - 1:
            move_line = (
                f"{player_label(last_move['player'])}: {last_move['move']}  "
                f"{last_move['piece']}"
            )
            if last_move["captured"] != ".":
                move_line += f" x {last_move['captured']}"
            stdscr.attron(curses.color_pair(2) | curses.A_BOLD)
            stdscr.addnstr(info_y, panel_x, move_line, max(0, w - panel_x - 2))
            stdscr.attroff(curses.color_pair(2) | curses.A_BOLD)
            info_y += 1

            if last_move["value"] is not None and info_y < h - 1:
                stdscr.attron(curses.color_pair(8))
                stdscr.addnstr(info_y, panel_x, f"V={last_move['value']:+.2f}",
                               max(0, w - panel_x - 2))
                stdscr.attroff(curses.color_pair(8))
                info_y += 1

        help_line = "Left/Right: step  Home/End: jump  Q: quit"
        stdscr.attron(curses.color_pair(7))
        stdscr.addnstr(h - 1, 2, help_line, max(0, w - 4))
        stdscr.attroff(curses.color_pair(7))

        stdscr.refresh()
        key = stdscr.getch()

        if key in (ord("q"), ord("Q")):
            return
        if key == curses.KEY_RIGHT or key in (ord(" "), 10, 13):
            move_idx = min(len(view.moves), move_idx + 1)
        elif key == curses.KEY_LEFT:
            move_idx = max(0, move_idx - 1)
        elif key == curses.KEY_HOME:
            move_idx = 0
        elif key == curses.KEY_END:
            move_idx = len(view.moves)


def view_file(path: str) -> None:
    view = load_view(path)
    curses.wrapper(lambda stdscr: draw_game(stdscr, view))


def gather_paths(path: str) -> list[str]:
    if not os.path.isdir(path):
        return [path]

    txt_files = sorted(glob.glob(os.path.join(path, "*.txt")))
    if txt_files:
        return txt_files

    bin_files = sorted(glob.glob(os.path.join(path, "*.bin*")))
    if bin_files:
        return bin_files

    raise ValueError(f"no supported game files found in {path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize Xiangqi self-play (.bin/.bin.zst/.bin.gz) or evaluation (.txt) games",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  python3 scripts/visualize.py training/selfplay/iter_0001/game_0.bin.zst
  python3 scripts/visualize.py training/eval/iter_0001/game_0.txt
  python3 scripts/visualize.py training/selfplay/iter_0001
""",
    )
    parser.add_argument("path", help="Game file or directory")
    args = parser.parse_args()

    for path in gather_paths(args.path):
        view_file(path)


if __name__ == "__main__":
    main()
