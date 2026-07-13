"""
MiniGo AlphaZero — unified game-data library (V3 records).

One place for everything that touches the V3 game-record format and
board-state encodings, shared by the trainer (train_continuous.py) and
the viewer (visualize.py):

  parse_v3(data)          — bytes → GameV3 (moves, policies, outcome)
  Replay                  — minimal Go engine mirroring src/game.cpp's
                            board evolution, history ring, and recent-
                            action tracking (replays LEGAL recorded
                            moves; it does not re-check legality)
  encode_minigo(replay)   — 17-plane MiniGo encoding (game.cpp encode())
  encode_katago(replay)   — KataGo V7: 22 spatial planes + 19 globals
                            (port of src/katago_inputs.cpp)
  sample_positions(...)   — replay a game once, emit training samples
                            for the requested move indices in either
                            encoding, each under a random dihedral
                            transform (augmentation happens HERE, at
                            load time — V3 files store none)

V3 layout (little-endian, packed; written by src/main_selfplay.cpp):
  u16 magic 'MG' (0x4D47) | u16 version=3 | i32 board_size | f32 komi
  i32 n_moves | i8 winner (0 draw, 1 black, 2 white) | f32 black_score
  per move: i16 action (hw = pass) | f32 policy[hw+1]
  footer:   i8 owner[hw]  (0 empty/dame, 1 black, 2 white)

Per-position training targets are derived, never stored:
  player(m)     = BLACK if m even else WHITE (games strictly alternate)
  value(m)      = +1 win / −1 loss / 0 draw from player(m)'s view
  score(m)      = black_score from player(m)'s view
  ownership(m)  = 1.0 where owner == player(m), else 0.0
  opp_action(m) = actions[m+1], or −1 at the final move
"""

import os
import struct
import sys
from dataclasses import dataclass

import numpy as np

V3_MAGIC = 0x4D47   # 'MG'
V3_VERSION = 3
V3_HEADER_LEN = 21  # u16+u16+i32+f32+i32+i8+f32

EMPTY, BLACK, WHITE = 0, 1, 2

# Encodings (must match the C++ encoders exactly)
MINIGO_HISTORY = 8
MINIGO_CHANNELS = MINIGO_HISTORY * 2 + 1          # 17
KATAGO_SPATIAL = 22
KATAGO_GLOBAL = 19
KATAGO_RECENT = 5


# ════════════════════════════════════════════════════════════
#  V3 parsing
# ════════════════════════════════════════════════════════════

@dataclass
class GameV3:
    board_size: int
    komi: float
    winner: int            # 0 draw, 1 black, 2 white
    black_score: float     # black − white, komi included
    actions: np.ndarray    # [n_moves] int16, hw = pass
    policies: np.ndarray   # [n_moves, hw+1] float32
    owners: np.ndarray     # [hw] int8: 0 empty/dame, 1 black, 2 white

    @property
    def n_moves(self):
        return len(self.actions)


def peek_v3_moves(header_bytes):
    """Return n_moves from a V3 header (>= V3_HEADER_LEN bytes), or 0 if
    the bytes are not a V3 record."""
    if len(header_bytes) < V3_HEADER_LEN:
        return 0
    magic, version, _bs, _komi, n_moves = struct.unpack_from(
        "<HHifi", header_bytes, 0)
    if magic != V3_MAGIC or version != V3_VERSION:
        return 0
    return n_moves


def parse_v3(data):
    """Parse a full V3 record payload.  Raises ValueError on mismatch."""
    if len(data) < V3_HEADER_LEN:
        raise ValueError("V3 record truncated (header)")
    magic, version, board_size, komi, n_moves = struct.unpack_from(
        "<HHifi", data, 0)
    if magic != V3_MAGIC:
        raise ValueError(f"Not a MiniGo record (magic=0x{magic:04X})")
    if version != V3_VERSION:
        raise ValueError(f"Unsupported record version {version} (want 3)")
    winner, black_score = struct.unpack_from("<bf", data, 16)

    hw = board_size * board_size
    per_move = 2 + 4 * (hw + 1)
    body = np.frombuffer(data, dtype=np.uint8, offset=V3_HEADER_LEN,
                         count=n_moves * per_move).reshape(n_moves, per_move)
    actions = body[:, 0:2].copy().view(np.int16).reshape(n_moves)
    policies = body[:, 2:].copy().view(np.float32).reshape(n_moves, hw + 1)

    own_off = V3_HEADER_LEN + n_moves * per_move
    owners = np.frombuffer(data, dtype=np.int8, offset=own_off, count=hw).copy()

    return GameV3(board_size=board_size, komi=komi, winner=winner,
                  black_score=black_score, actions=actions,
                  policies=policies, owners=owners)


# ════════════════════════════════════════════════════════════
#  Replay engine — mirrors src/game.cpp board evolution
# ════════════════════════════════════════════════════════════

def _neighbors(r, c, n):
    if r > 0:
        yield r - 1, c
    if r < n - 1:
        yield r + 1, c
    if c > 0:
        yield r, c - 1
    if c < n - 1:
        yield r, c + 1


def _flood_group(board, r, c):
    """Group cells + liberty set for the stone at (r, c)."""
    n = board.shape[0]
    color = board[r, c]
    group, libs = [], set()
    seen = {(r, c)}
    stack = [(r, c)]
    while stack:
        cr, cc = stack.pop()
        group.append((cr, cc))
        for nr, nc in _neighbors(cr, cc, n):
            v = board[nr, nc]
            if v == EMPTY:
                libs.add((nr, nc))
            elif v == color and (nr, nc) not in seen:
                seen.add((nr, nc))
                stack.append((nr, nc))
    return group, libs


class Replay:
    """Replays a recorded (legal) move sequence, maintaining exactly the
    state the two encoders need: the board, the last-8 post-move board
    snapshots (seeded with the empty board, like GoGame's constructor),
    the last-5 actions, consecutive passes, and the simple-ko point."""

    def __init__(self, board_size):
        self.n = board_size
        self.hw = board_size * board_size
        self.board = np.zeros((board_size, board_size), dtype=np.int8)
        self.current_player = BLACK
        self.consecutive_passes = 0
        self.ko_point = None                      # (r, c) banned by simple ko
        self.history = [self.board.copy()]        # post-move snapshots, newest last
        self.recent_actions = []                  # newest first; -1 = pass
        self.ko_history = [None]                  # ko point per snapshot, newest last

    def play(self, action):
        """action in [0, hw) = board point, hw or -1 = pass."""
        n = self.n
        is_pass = action < 0 or action >= self.hw

        self.recent_actions.insert(0, -1 if is_pass else int(action))
        del self.recent_actions[KATAGO_RECENT:]

        if is_pass:
            self.consecutive_passes += 1
            self.ko_point = None
        else:
            self.consecutive_passes = 0
            r, c = divmod(int(action), n)
            me, opp = self.current_player, (BLACK + WHITE) - self.current_player
            self.board[r, c] = me

            captured = []
            for nr, nc in _neighbors(r, c, n):
                if self.board[nr, nc] == opp:
                    group, libs = _flood_group(self.board, nr, nc)
                    if not libs:
                        captured.extend(group)
            for cr, cc in captured:
                self.board[cr, cc] = EMPTY

            # Simple-ko point: a single stone captured a single stone and
            # the capturing stone sits in atari on exactly that square —
            # equivalent to game.cpp's "would recreate the previous board".
            self.ko_point = None
            if len(captured) == 1:
                group, libs = _flood_group(self.board, r, c)
                if len(group) == 1 and libs == {captured[0]}:
                    self.ko_point = captured[0]

        self.current_player = (BLACK + WHITE) - self.current_player
        self.history.append(self.board.copy())
        del self.history[:-MINIGO_HISTORY]
        self.ko_history.append(self.ko_point)
        del self.ko_history[:-3]                  # ladder planes need t-0/1/2


# ════════════════════════════════════════════════════════════
#  Encoders (must byte-match the C++ implementations)
# ════════════════════════════════════════════════════════════

def encode_minigo(rep):
    """MiniGo 17-plane encoding (port of game.cpp encode()):
    planes 0-7  = current player's stones, history snapshots newest first
    planes 8-15 = opponent's stones, same snapshots
    plane 16    = all-ones iff current player is BLACK
    Returns float32 [17, n, n]."""
    n = rep.n
    out = np.zeros((MINIGO_CHANNELS, n, n), dtype=np.float32)
    me = rep.current_player
    opp = (BLACK + WHITE) - me
    snaps = rep.history[::-1][:MINIGO_HISTORY]    # newest first
    for i, snap in enumerate(snaps):
        out[i][snap == me] = 1.0
        out[MINIGO_HISTORY + i][snap == opp] = 1.0
    if me == BLACK:
        out[2 * MINIGO_HISTORY] = 1.0
    return out


def _tt_area(board):
    """Tromp-Taylor area map: 0 dame, 1 black, 2 white (stones + territory)."""
    n = board.shape[0]
    area = board.astype(np.int8).copy()
    seen = np.zeros((n, n), dtype=bool)
    for r in range(n):
        for c in range(n):
            if board[r, c] != EMPTY or seen[r, c]:
                continue
            region = [(r, c)]
            seen[r, c] = True
            stack = [(r, c)]
            borders = set()
            while stack:
                cr, cc = stack.pop()
                for nr, nc in _neighbors(cr, cc, n):
                    v = board[nr, nc]
                    if v == EMPTY and not seen[nr, nc]:
                        seen[nr, nc] = True
                        region.append((nr, nc))
                        stack.append((nr, nc))
                    elif v != EMPTY:
                        borders.add(int(v))
            owner = borders.pop() if len(borders) == 1 else EMPTY
            if owner != EMPTY:
                for cr, cc in region:
                    area[cr, cc] = owner
    return area


# ════════════════════════════════════════════════════════════
#  Ladder solver (KataGo V7 planes 14-17)
#
#  Byte-identical mirror of the C++ port in src/katago_inputs.cpp
#  (itself a faithful port of upstream KataGo's bounded ladder search:
#  Board::searchIsLadderCaptured / ...AttackerFirst2Libs + iterLadders).
#  Same node budget (25 000), same base cases, same double-ko-death
#  rule, same early-outs and move-ordering heuristic, and — critically
#  for C++/Python parity — the same canonical row-major chain traversal
#  and the same neighbor order (up, left, right, down).
# ════════════════════════════════════════════════════════════

_LADDER_NODE_BUDGET = 25000

_ladder_nbr_cache = {}


def _ladder_nbrs(n):
    """Neighbor table in the C++ solver's order: up, left, right, down."""
    tbl = _ladder_nbr_cache.get(n)
    if tbl is None:
        tbl = []
        for p in range(n * n):
            r, c = divmod(p, n)
            lst = []
            if r > 0:
                lst.append(p - n)
            if c > 0:
                lst.append(p - 1)
            if c < n - 1:
                lst.append(p + 1)
            if r < n - 1:
                lst.append(p + n)
            tbl.append(tuple(lst))
        _ladder_nbr_cache[n] = tbl
    return tbl


class _LadderBoard:
    __slots__ = ("n", "nn", "color", "ko", "nbrs")

    def __init__(self, flat_board, n, ko_flat):
        self.n = n
        self.nn = n * n
        self.color = bytearray(flat_board)         # EMPTY/BLACK/WHITE
        self.ko = ko_flat                          # flat index or -1
        self.nbrs = _ladder_nbrs(n)

    def chain(self, p):
        """Chain stones, sorted ascending (canonical order)."""
        color = self.color[p]
        col = self.color
        nbrs = self.nbrs
        stones = [p]
        seen = {p}
        head = 0
        while head < len(stones):
            q = stones[head]
            head += 1
            for a in nbrs[q]:
                if col[a] == color and a not in seen:
                    seen.add(a)
                    stones.append(a)
        stones.sort()
        return stones

    def chain_libs(self, p):
        col = self.color
        nbrs = self.nbrs
        libs = set()
        for q in self.chain(p):
            for a in nbrs[q]:
                if col[a] == EMPTY:
                    libs.add(a)
        return len(libs)


def _l_find_liberties(b, p, buf, buf_start, buf_idx):
    col = b.color
    nbrs = b.nbrs
    num = 0
    for q in b.chain(p):
        for lib in nbrs[q]:
            if col[lib] != EMPTY:
                continue
            dup = False
            for k in range(buf_start, buf_idx + num):
                if buf[k] == lib:
                    dup = True
                    break
            if not dup:
                while len(buf) <= buf_idx + num:
                    buf.append(-1)
                buf[buf_idx + num] = lib
                num += 1
    return num


def _l_find_liberty_gaining_captures(b, p, buf, buf_start, buf_idx):
    opp = (BLACK + WHITE) - b.color[p]
    col = b.color
    nbrs = b.nbrs
    heads = set()
    num = 0
    for q in b.chain(p):
        for a in nbrs[q]:
            if col[a] != opp:
                continue
            head = b.chain(a)[0]
            if head in heads:
                continue
            heads.add(head)
            if b.chain_libs(a) == 1:
                num += _l_find_liberties(b, a, buf, buf_start, buf_idx + num)
    return num


def _l_has_liberty_gaining_captures(b, p):
    opp = (BLACK + WHITE) - b.color[p]
    col = b.color
    nbrs = b.nbrs
    for q in b.chain(p):
        for a in nbrs[q]:
            if col[a] == opp and b.chain_libs(a) == 1:
                return True
    return False


def _l_immediate_libs(b, p):
    col = b.color
    return sum(1 for a in b.nbrs[p] if col[a] == EMPTY)


def _l_conn_libs_x2(b, p, pla):
    # per adjacent STONE, no chain dedup — faithful to upstream
    col = b.color
    x2 = 0
    for a in b.nbrs[p]:
        if col[a] == pla:
            libs = b.chain_libs(a)
            if libs > 1:
                x2 += libs * 2 - 3
    return x2


def _l_bound_libs_after_play(b, p, pla):
    opp = (BLACK + WHITE) - pla
    col = b.color
    n_imm = n_caps = pot_caps = n_conn = max_conn = 0
    for a in b.nbrs[p]:
        v = col[a]
        if v == EMPTY:
            n_imm += 1
        elif v == opp:
            if b.chain_libs(a) == 1:
                n_caps += 1
                pot_caps += len(b.chain(a))
        else:
            conn = b.chain_libs(a) - 1
            n_conn += conn
            if conn > max_conn:
                max_conn = conn
    lower = n_caps + (max_conn if max_conn > n_imm else n_imm)
    upper = n_imm + pot_caps + n_conn
    return lower, upper


def _l_libs_after_play(b, p, pla, maxv):
    opp = (BLACK + WHITE) - pla
    col = b.color
    libs = []
    captured_heads = []
    for a in b.nbrs[p]:
        v = col[a]
        if v == EMPTY:
            libs.append(a)
            if len(libs) >= maxv:
                return maxv
        elif v == opp and b.chain_libs(a) == 1:
            libs.append(a)
            if len(libs) >= maxv:
                return maxv
            head = b.chain(a)[0]
            if head not in captured_heads:
                captured_heads.append(head)

    def would_be_empty(q):
        if col[q] == EMPTY:
            return True
        if col[q] == opp:
            return b.chain(q)[0] in captured_heads
        return False

    conn_heads = []
    for a in b.nbrs[p]:
        if col[a] != pla:
            continue
        ch = b.chain(a)
        head = ch[0]
        if head in conn_heads:
            continue
        conn_heads.append(head)
        for s2 in ch:
            for q in b.nbrs[s2]:
                if q == p or not would_be_empty(q):
                    continue
                if q not in libs:
                    libs.append(q)
                    if len(libs) >= maxv:
                        return maxv
    return len(libs)


def _l_would_be_ko_capture(b, p, pla):
    if b.color[p] != EMPTY:
        return False
    opp = (BLACK + WHITE) - pla
    col = b.color
    capturable = -1
    for a in b.nbrs[p]:                 # off-board neighbors = walls, allowed
        if col[a] != opp:
            return False
        if b.chain_libs(a) == 1:
            if capturable != -1:
                return False
            capturable = a
    if capturable == -1:
        return False
    return len(b.chain(capturable)) == 1


def _l_is_suicide(b, p, pla):
    opp = (BLACK + WHITE) - pla
    col = b.color
    for a in b.nbrs[p]:
        v = col[a]
        if v == EMPTY:
            return False
        if v == pla:
            if b.chain_libs(a) > 1:
                return False
        else:
            if b.chain_libs(a) == 1:
                return False
    return True


def _l_is_legal(b, p, pla):
    return b.color[p] == EMPTY and p != b.ko and not _l_is_suicide(b, p, pla)


def _l_play(b, p, pla, cap_buf):
    """Returns an undo record (loc, pla, prev_ko, cap_start, cap_count).
    Ko rule mirrors upstream playMoveAssumeLegal: exactly one stone
    captured AND the played stone is a lone stone with one liberty."""
    prev_ko = b.ko
    cap_start = len(cap_buf)
    opp = (BLACK + WHITE) - pla
    col = b.color
    col[p] = pla
    for a in b.nbrs[p]:
        if col[a] != opp:
            continue
        if b.chain_libs(a) == 0:
            for q in b.chain(a):
                cap_buf.append(q)
                col[q] = EMPTY
    cap_count = len(cap_buf) - cap_start

    b.ko = -1
    if cap_count == 1:
        if len(b.chain(p)) == 1 and b.chain_libs(p) == 1:
            b.ko = cap_buf[cap_start]
    return (p, pla, prev_ko, cap_start, cap_count)


def _l_undo(b, rec, cap_buf):
    loc, pla, prev_ko, cap_start, cap_count = rec
    opp = (BLACK + WHITE) - pla
    col = b.color
    for i in range(cap_start, cap_start + cap_count):
        col[cap_buf[i]] = opp
    del cap_buf[cap_start:]
    col[loc] = EMPTY
    b.ko = prev_ko


def _l_search_is_captured(b, loc, defender_first, buf, cap_buf):
    col = b.color
    if col[loc] != BLACK and col[loc] != WHITE:
        return False
    libs = b.chain_libs(loc)
    if libs > 2 or (defender_first and libs > 1):
        return False

    pla = col[loc]
    opp = (BLACK + WHITE) - pla

    ko_saved = b.ko
    if defender_first:
        b.ko = -1                       # assume all kos work for the defender

    stack_size = b.n * b.n * 3 // 2 + 1
    ml_starts = [0] * stack_size
    ml_lens = [0] * stack_size
    ml_cur = [0] * stack_size
    records = [None] * stack_size

    stack_idx = 0
    node_count = 0
    ml_cur[0] = -1
    return_value = False
    returned_from_deeper = False

    while True:
        if stack_idx <= -1:
            b.ko = ko_saved
            return return_value
        # Stack limit: consider it captured (upstream behavior).
        if stack_idx >= stack_size - 1:
            return_value = True
            returned_from_deeper = True
            stack_idx -= 1
            continue
        # Node budget: assume it does not work; undo everything.
        if node_count >= _LADDER_NODE_BUDGET:
            stack_idx -= 1
            while stack_idx >= 0:
                _l_undo(b, records[stack_idx], cap_buf)
                stack_idx -= 1
            b.ko = ko_saved
            return False

        is_defender = ((defender_first and (stack_idx % 2) == 0) or
                       (not defender_first and (stack_idx % 2) == 1))

        if ml_cur[stack_idx] == -1:
            libs = b.chain_libs(loc)

            if not is_defender and libs <= 1:
                return_value = True
                returned_from_deeper = True
                stack_idx -= 1
                continue
            if not is_defender and libs >= 3:
                return_value = False
                returned_from_deeper = True
                stack_idx -= 1
                continue
            if is_defender and libs >= 2:
                return_value = False
                returned_from_deeper = True
                stack_idx -= 1
                continue
            # Ladders that depend on ko: assume the defender escapes.
            if is_defender and b.ko != -1:
                return_value = False
                returned_from_deeper = True
                stack_idx -= 1
                continue

            start = ml_starts[stack_idx]
            if is_defender:
                ml_len = _l_find_liberty_gaining_captures(b, loc, buf, start, start)
                ml_len += _l_find_liberties(b, loc, buf, start, start + ml_len)

                lower, upper = _l_bound_libs_after_play(
                    b, buf[start + ml_len - 1], pla)
                if lower >= 3:
                    return_value = False
                    returned_from_deeper = True
                    stack_idx -= 1
                    continue
                if ml_len == 1 and upper <= 1:
                    return_value = True
                    returned_from_deeper = True
                    stack_idx -= 1
                    continue
            else:
                ml_len = _l_find_liberties(b, loc, buf, start, start)
                # Attacker to move with the defender at exactly 2 libs.
                libs0 = _l_immediate_libs(b, buf[start])
                libs1 = _l_immediate_libs(b, buf[start + 1])

                # Double-ko death.
                if (libs0 == 0 and libs1 == 0 and
                        _l_would_be_ko_capture(b, buf[start], opp) and
                        _l_would_be_ko_capture(b, buf[start + 1], opp)):
                    if (_l_libs_after_play(b, buf[start], pla, 3) <= 2 and
                            _l_libs_after_play(b, buf[start + 1], pla, 3) <= 2):
                        if not _l_has_liberty_gaining_captures(b, loc):
                            return_value = True
                            returned_from_deeper = True
                            stack_idx -= 1
                            continue

                # Early quitouts when the liberties are not adjacent.
                r0, c0 = divmod(buf[start], b.n)
                r1, c1 = divmod(buf[start + 1], b.n)
                if abs(r0 - r1) + abs(c0 - c1) != 1:
                    if libs0 >= 3 and libs1 >= 3:
                        return_value = False
                        returned_from_deeper = True
                        stack_idx -= 1
                        continue
                    elif libs0 >= 3:
                        ml_len = 1
                    elif libs1 >= 3:
                        buf[start] = buf[start + 1]
                        ml_len = 1
                # Attack the escape-richer liberty first.
                if ml_len > 1:
                    libs0 = libs0 * 2 + _l_conn_libs_x2(b, buf[start], pla)
                    libs1 = libs1 * 2 + _l_conn_libs_x2(b, buf[start + 1], pla)
                    if libs1 > libs0:
                        buf[start], buf[start + 1] = buf[start + 1], buf[start]

            ml_lens[stack_idx] = ml_len
            ml_cur[stack_idx] = 0
        else:
            if returned_from_deeper:
                _l_undo(b, records[stack_idx], cap_buf)

            if is_defender and not return_value:
                returned_from_deeper = True
                stack_idx -= 1
                continue
            if not is_defender and return_value:
                returned_from_deeper = True
                stack_idx -= 1
                continue
            ml_cur[stack_idx] += 1

        if ml_cur[stack_idx] >= ml_lens[stack_idx]:
            return_value = is_defender  # defender out of moves = captured
            returned_from_deeper = True
            stack_idx -= 1
            continue

        move = buf[ml_starts[stack_idx] + ml_cur[stack_idx]]
        p = pla if is_defender else opp

        if not _l_is_legal(b, move, p):
            return_value = is_defender
            returned_from_deeper = False
            continue

        records[stack_idx] = _l_play(b, move, p, cap_buf)
        node_count += 1

        stack_idx += 1
        ml_cur[stack_idx] = -1
        ml_starts[stack_idx] = ml_starts[stack_idx - 1] + ml_lens[stack_idx - 1]
        ml_lens[stack_idx] = 0


def _l_search_attacker_first_2libs(b, loc, buf, cap_buf):
    """Returns (laddered, working_moves)."""
    col = b.color
    if col[loc] != BLACK and col[loc] != WHITE:
        return False, []
    if b.chain_libs(loc) != 2:
        return False, []

    opp = (BLACK + WHITE) - col[loc]

    _l_find_liberties(b, loc, buf, 0, 0)
    move0, move1 = buf[0], buf[1]
    move0_works = move1_works = False

    if _l_is_legal(b, move0, opp):
        rec = _l_play(b, move0, opp, cap_buf)
        move0_works = _l_search_is_captured(b, loc, True, buf, cap_buf)
        _l_undo(b, rec, cap_buf)
    if _l_is_legal(b, move1, opp):
        rec = _l_play(b, move1, opp, cap_buf)
        move1_works = _l_search_is_captured(b, loc, True, buf, cap_buf)
        _l_undo(b, rec, cap_buf)

    working = []
    if move0_works:
        working.append(move0)
    if move1_works:
        working.append(move1)
    return bool(working), working



# ── Native ladder solver (libminigo_ladder.so via ctypes) ────
#
# The C++ solver is the single source of truth; this loader lets the
# trainer use it directly (ctypes releases the GIL, so the ring's
# thread pool parallelizes it).  The pure-Python mirror below stays as
# a fallback — correct but ~20x slower — and prints a one-time warning.
# Force the fallback with MINIGO_LADDER_FORCE_PY=1 (parity testing).

_ladder_fn = None
_ladder_checked = False


def _load_ladder_lib():
    global _ladder_fn, _ladder_checked
    if _ladder_checked:
        return _ladder_fn
    _ladder_checked = True
    if os.environ.get("MINIGO_LADDER_FORCE_PY"):
        return None
    import ctypes
    here = os.path.dirname(os.path.abspath(__file__))
    candidates = []
    if os.environ.get("MINIGO_LADDER_LIB"):
        candidates.append(os.environ["MINIGO_LADDER_LIB"])
    candidates += [
        os.path.join(here, "..", "build", "libminigo_ladder.so"),
        "libminigo_ladder.so",
    ]
    for cand in candidates:
        try:
            lib = ctypes.CDLL(cand)
            fn = lib.minigo_ladder_fill
            fn.argtypes = [ctypes.c_char_p, ctypes.c_int32, ctypes.c_int32,
                           ctypes.c_int32,
                           ctypes.POINTER(ctypes.c_uint8),
                           ctypes.POINTER(ctypes.c_uint8)]
            fn.restype = None
            _ladder_fn = fn
            return fn
        except OSError:
            continue
    print("[gamedata] WARNING: libminigo_ladder.so not found — using the "
          "pure-Python ladder solver for KataGo V7 planes 14-17 (~20x "
          "slower encoding).  Build it with: make -C build minigo_ladder",
          file=sys.stderr, flush=True)
    return None


def ladder_native_available():
    """True when the native solver library is loadable (the trainer uses
    this to pick its sampling-pool type)."""
    return _load_ladder_lib() is not None


def _fill_ladder_planes(sp, flat_board, n, ko_flat,
                        plane_stones, plane_working, opp_color):
    """Native-preferred dispatcher; falls back to the Python mirror."""
    fn = _load_ladder_lib()
    if fn is None:
        return _fill_ladder_planes_py(sp, flat_board, n, ko_flat,
                                      plane_stones, plane_working, opp_color)
    import ctypes
    nn = n * n
    lad = np.zeros(nn, dtype=np.uint8)
    lad_p = lad.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))
    if plane_working >= 0:
        work = np.zeros(nn, dtype=np.uint8)
        work_p = work.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))
        fn(flat_board, n, ko_flat, int(opp_color), lad_p, work_p)
        sp[plane_working].reshape(-1)[work != 0] = 1.0
    else:
        fn(flat_board, n, ko_flat, 0, lad_p, None)
    sp[plane_stones].reshape(-1)[lad != 0] = 1.0


def _fill_ladder_planes_py(sp, flat_board, n, ko_flat,
                           plane_stones, plane_working, opp_color):
    """Mirror of C++ fill_ladder_planes: decide once per 1-2-lib chain,
    mark its stones on plane_stones; for laddered 2-lib chains of
    opp_color additionally mark the attacker's working first moves on
    plane_working (upstream feature 17)."""
    b = _LadderBoard(flat_board, n, ko_flat)
    buf = []
    cap_buf = []
    solved = bytearray(b.nn)

    for p in range(b.nn):
        if b.color[p] == EMPTY or solved[p]:
            continue
        libs = b.chain_libs(p)
        if libs != 1 and libs != 2:
            continue
        stones = b.chain(p)
        for s2 in stones:
            solved[s2] = 1

        working = []
        if libs == 1:
            laddered = _l_search_is_captured(b, p, True, buf, cap_buf)
        else:
            laddered, working = _l_search_attacker_first_2libs(b, p, buf, cap_buf)
        if not laddered:
            continue

        for s2 in stones:
            sp[plane_stones, s2 // n, s2 % n] = 1.0
        if plane_working >= 0 and b.color[p] == opp_color and libs > 1:
            for w in working:
                sp[plane_working, w // n, w % n] = 1.0


def encode_katago(rep, komi):
    """KataGo V7 input encoding (port of src/katago_inputs.cpp).
    Returns (spatial float32 [22, n, n], global float32 [19]).

    Same fidelity caveats as the C++ encoder: encore/button/PDA
    features and non-default rules bits are zero; ladder planes 14-17
    are computed with the ported upstream bounded ladder search."""
    n, hw = rep.n, rep.hw
    sp = np.zeros((KATAGO_SPATIAL, n, n), dtype=np.float32)
    gl = np.zeros(KATAGO_GLOBAL, dtype=np.float32)
    board = rep.board
    me = rep.current_player
    opp = (BLACK + WHITE) - me

    sp[0] = 1.0                                   # on-board mask
    sp[1][board == me] = 1.0
    sp[2][board == opp] = 1.0

    # planes 3-5: stone in a group with exactly 1/2/3 liberties
    counted = np.zeros((n, n), dtype=bool)
    for r in range(n):
        for c in range(n):
            if board[r, c] == EMPTY or counted[r, c]:
                continue
            group, libs = _flood_group(board, r, c)
            nl = len(libs)
            for gr, gc in group:
                counted[gr, gc] = True
                if 1 <= nl <= 3:
                    sp[2 + nl, gr, gc] = 1.0

    # plane 6: simple-ko banned point
    if rep.ko_point is not None:
        sp[6, rep.ko_point[0], rep.ko_point[1]] = 1.0
    # planes 7, 8: encore-only / unused → 0

    # planes 9-13: move locations 1..5 plies back (one-hot); a pass at
    # ply i sets global feature i instead.
    for i in range(min(KATAGO_RECENT, len(rep.recent_actions))):
        a = rep.recent_actions[i]
        if a < 0:
            gl[i] = 1.0
        else:
            sp[9 + i, a // n, a % n] = 1.0

    # planes 14-17: ladder features (mirror of src/katago_inputs.cpp).
    # 14/17 use the current board + current ko; 15/16 use the boards
    # 1 / 2 plies ago with THEIR simple-ko points (clamped at game start).
    def _hist_at(back, seq):
        return seq[-1 - back] if back < len(seq) else seq[0]

    def _ko_flat(k):
        return -1 if k is None else k[0] * n + k[1]

    _fill_ladder_planes(sp, bytes(board.reshape(-1)), n,
                        _ko_flat(rep.ko_point), 14, 17, opp)
    _fill_ladder_planes(sp, bytes(_hist_at(1, rep.history).reshape(-1)), n,
                        _ko_flat(_hist_at(1, rep.ko_history)), 15, -1, EMPTY)
    _fill_ladder_planes(sp, bytes(_hist_at(2, rep.history).reshape(-1)), n,
                        _ko_flat(_hist_at(2, rep.ko_history)), 16, -1, EMPTY)

    # planes 18, 19: Tromp-Taylor area for me / opp
    area = _tt_area(board)
    sp[18][area == me] = 1.0
    sp[19][area == opp] = 1.0
    # planes 20, 21: encore-only → 0

    # gl[5]: self-komi / 20 (white holds the komi), clamped
    self_komi = komi if me == WHITE else -komi
    bound = float(hw + 1)
    gl[5] = max(-bound, min(bound, self_komi)) / 20.0
    # gl[6-13]: rules bits (simple ko, no suicide, area scoring, no tax,
    #           no encore) → 0
    # gl[14]: pass-would-end-phase
    if rep.consecutive_passes >= 1:
        gl[14] = 1.0
    # gl[15-17]: playoutDoublingAdvantage / button → 0
    # gl[18]: komi parity wave (board-area parity anchored, upstream
    #         fillRowV7 nninputs.cpp:2681-2713)
    drawable_even = (hw % 2) == 0
    if drawable_even:
        komi_floor = np.floor(self_komi / 2.0) * 2.0
    else:
        komi_floor = np.floor((self_komi - 1.0) / 2.0) * 2.0 + 1.0
    delta = min(2.0, max(0.0, self_komi - komi_floor))
    if delta < 0.5:
        wave = delta
    elif delta < 1.5:
        wave = 1.0 - delta
    else:
        wave = delta - 2.0
    gl[18] = wave

    return sp, gl


# ════════════════════════════════════════════════════════════
#  Dihedral transforms (augmentation at load time)
# ════════════════════════════════════════════════════════════

def _transform_planes(planes, rot, flip):
    """Apply d8 transform to [C, n, n] planes (matches the point map of
    the old C++ augment: rot k = (r,c) → (c, n−1−r) applied k times,
    then optional column flip)."""
    out = np.rot90(planes, k=-rot, axes=(1, 2)) if rot else planes
    if flip:
        out = out[:, :, ::-1]
    return np.ascontiguousarray(out)


def _transform_point(idx, n, rot, flip):
    """Same transform for a flat board index; pass/-1 unchanged."""
    if idx < 0 or idx >= n * n:
        return idx
    r, c = divmod(int(idx), n)
    for _ in range(rot):
        r, c = c, n - 1 - r
    if flip:
        c = n - 1 - c
    return r * n + c


def _transform_policy(policy, n, rot, flip):
    """d8-transform a [hw+1] policy vector (pass entry preserved)."""
    board = policy[:-1].reshape(1, n, n)
    out = np.empty_like(policy)
    out[:-1] = _transform_planes(board, rot, flip).reshape(-1)
    out[-1] = policy[-1]
    return out


# ════════════════════════════════════════════════════════════
#  Training-sample extraction
# ════════════════════════════════════════════════════════════

def sample_job(blob, want, seed, encoder):
    """Decompress one .zst game blob, replay it, and emit `want` samples.

    Module-level and torch-free on purpose: the trainer runs this in a
    ProcessPoolExecutor (spawn) for the katago encoder — the ladder
    solver is pure Python and GIL-bound, so thread pools cannot
    parallelize it.  Returns sample_positions()'s dict, or None on any
    parse/decompress failure (caller backfills).
    """
    if blob is None:
        return None
    try:
        import io as _io
        import zstandard as _zstd
        with _zstd.ZstdDecompressor().stream_reader(_io.BytesIO(blob)) as r:
            data = r.read()
        game = parse_v3(data)
        if game.n_moves == 0:
            return None
        rng = np.random.default_rng(seed)
        idx = rng.integers(0, game.n_moves, size=want)
        return sample_positions(game, idx, encoder, rng)
    except (ValueError, Exception) as e:  # zstd errors subclass Exception
        if isinstance(e, (KeyboardInterrupt, SystemExit)):
            raise
        return None


def sample_positions(game, move_indices, encoder, rng):
    """Replay `game` once and produce training samples at `move_indices`.

    encoder: "minigo" (17-plane states) or "katago" (V7 spatial+global).
    Each sample gets an independent random dihedral transform.

    Returns a dict of numpy arrays:
      minigo: states [K,17,n,n]
      katago: spatial [K,22,n,n], globals [K,19]
      both:   policies [K,hw+1], values [K], scores [K],
              owns [K,hw], opps [K] (int64; −1 = none/pass-terminal)
    """
    n = game.board_size
    hw = n * n
    idx_sorted = sorted(int(m) for m in move_indices)
    k = len(idx_sorted)

    policies = np.empty((k, hw + 1), dtype=np.float32)
    values = np.empty(k, dtype=np.float32)
    scores = np.empty(k, dtype=np.float32)
    owns = np.empty((k, hw), dtype=np.float32)
    opps = np.empty(k, dtype=np.int64)
    if encoder == "minigo":
        states = np.empty((k, MINIGO_CHANNELS, n, n), dtype=np.float32)
    elif encoder == "katago":
        spatial = np.empty((k, KATAGO_SPATIAL, n, n), dtype=np.float32)
        globals_ = np.empty((k, KATAGO_GLOBAL), dtype=np.float32)
    else:
        raise ValueError(f"unknown encoder {encoder!r}")

    owners = game.owners.astype(np.int8)
    rep = Replay(n)
    next_move = 0
    out_i = 0
    for m in idx_sorted:
        while next_move < m:                       # advance replay to move m
            rep.play(int(game.actions[next_move]))
            next_move += 1

        player = BLACK if (m % 2 == 0) else WHITE
        rot = int(rng.integers(0, 4))
        flip = bool(rng.integers(0, 2))

        if encoder == "minigo":
            states[out_i] = _transform_planes(encode_minigo(rep), rot, flip)
        else:
            sp, gl = encode_katago(rep, game.komi)
            spatial[out_i] = _transform_planes(sp, rot, flip)
            globals_[out_i] = gl                   # globals are d8-invariant

        policies[out_i] = _transform_policy(game.policies[m], n, rot, flip)

        if game.winner == EMPTY:
            values[out_i] = 0.0
        else:
            values[out_i] = 1.0 if game.winner == player else -1.0
        scores[out_i] = game.black_score if player == BLACK else -game.black_score

        own = (owners == player).astype(np.float32).reshape(1, n, n)
        owns[out_i] = _transform_planes(own, rot, flip).reshape(-1)

        opp = int(game.actions[m + 1]) if m + 1 < game.n_moves else -1
        if 0 <= opp < hw:
            opp = _transform_point(opp, n, rot, flip)
        elif opp == hw:
            pass                                    # pass action index kept
        opps[out_i] = opp
        out_i += 1

    out = {"policies": policies, "values": values, "scores": scores,
           "owns": owns, "opps": opps}
    if encoder == "minigo":
        out["states"] = states
    else:
        out["spatial"] = spatial
        out["globals"] = globals_
    return out
