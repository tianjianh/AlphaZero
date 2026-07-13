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

import struct
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


def encode_katago(rep, komi):
    """KataGo V7 input encoding (port of src/katago_inputs.cpp).
    Returns (spatial float32 [22, n, n], global float32 [19]).

    Same fidelity caveats as the C++ encoder: ladder planes 14-17,
    encore/button/PDA features, and non-default rules bits are zero."""
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

    # planes 14-17: ladder features → 0 (same TODO as the C++ encoder)

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
