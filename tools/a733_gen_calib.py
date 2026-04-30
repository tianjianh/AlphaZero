#!/usr/bin/env python3
"""Generate Acuity-style calibration data for kata1 → A733 quantization.

Acuity's `pegasus.py quantize` runs the network forward on a representative
sample set to compute per-tensor scales.  Kata1 takes
`state_spatial[K, 22, 9, 9]` and `state_global[K, 19]` fp32.

The naïve random-policy version of this script produced an int8
calibration that got 3% top-1 on the board.  The fix is to drive the
calibration games with the kata1 ONNX model itself (via onnxruntime),
so the resulting positions are *in-distribution* for kata1's
activations.  Random-only random play causes kata1 to read every
position as "losing by 50+ points" — the score-head activation range
the calibrator sees is then totally different from inference time.

This script:
  1. Plays N 9×9 Go games. Two policies are mixed:
       * `--strong-frac 0.5` (default): half the moves come from
         sampling kata1's softmax(policy_logits / temp).
       * The rest come from an eye-aware random policy — keeps weird
         tactical positions in the calibration set.
       * Optionally, a fraction of *whole games* are played all-random
         (`--random-game-frac`) for unfamiliar-position coverage.
  2. Snapshots positions densely across the game (every K moves, with
     denser sampling early/late).
  3. Encodes each snapshot via a faithful port of
     `src/katago_inputs.cpp`'s KataGo V7 encoder.
  4. Writes one ASCII text file per sample per input, plus the
     `dataset0_spatial.txt` / `dataset1_global.txt` index files
     Acuity's TEXT-mode loader points at.

Usage:
    python tools/a733_gen_calib.py \\
        --output-dir build/a733_calib \\
        --num-games 80 \\
        --onnx-model models/kata1-b10c128.a733.bs1.unshared.onnx \\
        --strong-frac 0.7 \\
        --random-game-frac 0.2 \\
        --board-size 9 \\
        --seed 1234

Output layout (relative to --output-dir):
    state_spatial/0000.tensor  ...  (np.loadtxt-readable text, flat 22*9*9
                                     = 1782 floats, one per line)
    state_global/0000.tensor   ...  (19 floats, one per line)
    dataset0_spatial.txt       list of paths, one per line
    dataset1_global.txt        list of paths, one per line

Acuity v6.30.22's TEXT-mode loader (`acuitylib/dataset/file_path_dataset.py`)
calls `np.loadtxt(path)` on each line of the dataset txt and then
reshapes to the inputmeta-declared `shape:`.  So each .tensor file is
a plain ASCII text file with one float per line (or whitespace-
separated — np.loadtxt is tolerant).  NPY mode and raw-binary both
fail in this Acuity version.

The Acuity inputmeta then points its `databases[0].path` at
`dataset0_spatial.txt` and `databases[1].path` at `dataset1_global.txt`.
"""

from __future__ import annotations

import argparse
import math
import os
import random
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import numpy as np


# ─────────────────────────────────────────────────────────────────
#  Minimal Go board + ko detection
# ─────────────────────────────────────────────────────────────────

EMPTY, BLACK, WHITE = 0, 1, 2
PASS = -1


def opponent(c: int) -> int:
    return WHITE if c == BLACK else BLACK


@dataclass
class GoBoard:
    n: int = 9
    komi: float = 7.5
    board: np.ndarray = field(default=None)  # [n,n] int8
    current: int = BLACK
    move_count: int = 0
    consecutive_passes: int = 0
    # Ring buffer of past positions for simple-ko detection (KO rule:
    # disallow re-creating the previous position).
    prev_board: Optional[np.ndarray] = None
    # Recent action history for the KataGo encoder's planes 9-13.
    # recent[0] is most recent.
    recent: List[int] = field(default_factory=list)

    def __post_init__(self):
        if self.board is None:
            self.board = np.zeros((self.n, self.n), dtype=np.int8)

    def copy(self) -> "GoBoard":
        b = GoBoard(self.n, self.komi)
        b.board = self.board.copy()
        b.current = self.current
        b.move_count = self.move_count
        b.consecutive_passes = self.consecutive_passes
        b.prev_board = None if self.prev_board is None else self.prev_board.copy()
        b.recent = list(self.recent)
        return b

    # ── Group / liberty utilities ─────────────────────────────────
    def _flood(self, r: int, c: int, color: int):
        """Return (stones_set, libs_count) for the group containing (r,c)."""
        stones = set()
        libs = set()
        stack = [(r, c)]
        while stack:
            pr, pc = stack.pop()
            if (pr, pc) in stones:
                continue
            stones.add((pr, pc))
            for nr, nc in self._neighbours(pr, pc):
                s = self.board[nr, nc]
                if s == EMPTY:
                    libs.add((nr, nc))
                elif s == color and (nr, nc) not in stones:
                    stack.append((nr, nc))
        return stones, len(libs)

    def _neighbours(self, r: int, c: int):
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < self.n and 0 <= nc < self.n:
                yield nr, nc

    def liberties_at(self, r: int, c: int) -> int:
        if self.board[r, c] == EMPTY:
            return 0
        _, libs = self._flood(r, c, int(self.board[r, c]))
        return libs

    # ── Move legality + execution ─────────────────────────────────
    def _trial_play(self, r: int, c: int, color: int):
        """Return the resulting board after playing (r,c) for `color`,
        capturing as needed.  Returns None if the move would be suicide
        (no caller flag here — purely a trial)."""
        if self.board[r, c] != EMPTY:
            return None
        b = self.board.copy()
        b[r, c] = color
        opp = opponent(color)

        # Capture opponent groups with no liberties
        captured_any = False
        for nr, nc in self._neighbours(r, c):
            if b[nr, nc] != opp:
                continue
            stones, libs = _flood_static(b, nr, nc, opp, self.n)
            if libs == 0:
                for sr, sc in stones:
                    b[sr, sc] = EMPTY
                captured_any = True

        # Suicide check on our own group
        own_stones, own_libs = _flood_static(b, r, c, color, self.n)
        if own_libs == 0:
            return None  # suicide

        return b, captured_any

    def is_legal(self, action: int) -> bool:
        if action == PASS:
            return True
        r, c = divmod(action, self.n)
        result = self._trial_play(r, c, self.current)
        if result is None:
            return False
        new_b, _ = result
        # Simple-ko rule: disallow re-creating immediate previous board
        if self.prev_board is not None and np.array_equal(new_b, self.prev_board):
            return False
        return True

    def is_ko_ban(self, action: int) -> bool:
        """The action would produce the previous position (simple ko)."""
        if action == PASS:
            return False
        r, c = divmod(action, self.n)
        if self.board[r, c] != EMPTY:
            return False
        result = self._trial_play(r, c, self.current)
        if result is None:
            return False
        new_b, _ = result
        return self.prev_board is not None and np.array_equal(new_b, self.prev_board)

    def play(self, action: int):
        if action == PASS:
            self.consecutive_passes += 1
        else:
            r, c = divmod(action, self.n)
            result = self._trial_play(r, c, self.current)
            if result is None:
                raise ValueError(f"illegal move {action}")
            new_b, _ = result
            self.prev_board = self.board.copy()
            self.board = new_b
            self.consecutive_passes = 0
        self.recent.insert(0, action)
        if len(self.recent) > 5:
            self.recent.pop()
        self.current = opponent(self.current)
        self.move_count += 1

    def legal_actions(self, exclude_eyes: bool = True) -> List[int]:
        out = []
        for r in range(self.n):
            for c in range(self.n):
                a = r * self.n + c
                if not self.is_legal(a):
                    continue
                if exclude_eyes and self._is_eye(r, c, self.current):
                    continue
                out.append(a)
        out.append(PASS)
        return out

    def _is_eye(self, r: int, c: int, color: int) -> bool:
        """Heuristic: don't fill a one-point eye surrounded by friendly
        stones with at most one diagonal enemy/edge."""
        if self.board[r, c] != EMPTY:
            return False
        # All orthogonal neighbours must be friendly stones
        n_count = 0
        for nr, nc in self._neighbours(r, c):
            n_count += 1
            if self.board[nr, nc] != color:
                return False
        # Diagonals: count opp + off-board.  Eye if (interior: ≤1, edge: 0).
        diag_bad = 0
        diag_count = 0
        for dr, dc in ((-1, -1), (-1, 1), (1, -1), (1, 1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < self.n and 0 <= nc < self.n:
                diag_count += 1
                if self.board[nr, nc] == opponent(color):
                    diag_bad += 1
            else:
                diag_bad += 1  # off-board counts as bad
        if diag_count == 4:
            return diag_bad <= 1
        return diag_bad == 0  # edge / corner: no bad diagonals

    def recent_action(self, steps_back: int) -> int:
        """KataGo V7 plane 9..13 helper.  steps_back=0 is most recent.
        Returns -2 sentinel if no move that far back yet."""
        if steps_back >= len(self.recent):
            return -2
        return self.recent[steps_back]

    # ── Tromp-Taylor area for ownership planes ────────────────────
    def compute_area(self) -> np.ndarray:
        """Returns [n,n] int8 with each cell mapped to BLACK/WHITE/EMPTY
        per Tromp-Taylor area scoring."""
        area = self.board.copy()
        visited = np.zeros((self.n, self.n), dtype=bool)
        for r0 in range(self.n):
            for c0 in range(self.n):
                if self.board[r0, c0] != EMPTY or visited[r0, c0]:
                    continue
                # BFS the empty region, track which colors it touches
                region = []
                touches_b = touches_w = False
                stack = [(r0, c0)]
                while stack:
                    pr, pc = stack.pop()
                    if visited[pr, pc]:
                        continue
                    visited[pr, pc] = True
                    region.append((pr, pc))
                    for nr, nc in self._neighbours(pr, pc):
                        s = self.board[nr, nc]
                        if s == EMPTY and not visited[nr, nc]:
                            stack.append((nr, nc))
                        elif s == BLACK:
                            touches_b = True
                        elif s == WHITE:
                            touches_w = True
                owner = EMPTY
                if touches_b and not touches_w:
                    owner = BLACK
                elif touches_w and not touches_b:
                    owner = WHITE
                for r, c in region:
                    area[r, c] = owner
        return area


# Static helper used by _trial_play (operates on a passed-in board copy
# that's not yet committed to the GoBoard instance).
def _flood_static(b: np.ndarray, r: int, c: int, color: int, n: int):
    stones = set()
    libs = set()
    stack = [(r, c)]
    while stack:
        pr, pc = stack.pop()
        if (pr, pc) in stones:
            continue
        stones.add((pr, pc))
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = pr + dr, pc + dc
            if not (0 <= nr < n and 0 <= nc < n):
                continue
            s = b[nr, nc]
            if s == EMPTY:
                libs.add((nr, nc))
            elif s == color and (nr, nc) not in stones:
                stack.append((nr, nc))
    return stones, len(libs)


# ─────────────────────────────────────────────────────────────────
#  KataGo V7 input encoder — Python port of src/katago_inputs.cpp
# ─────────────────────────────────────────────────────────────────

NUM_SPATIAL = 22
NUM_GLOBAL = 19


def encode_katago(g: GoBoard) -> tuple:
    """Returns (state_spatial[22,n,n] fp32, state_global[19] fp32)."""
    n = g.n
    pla = g.current
    opp = opponent(pla)
    sp = np.zeros((NUM_SPATIAL, n, n), dtype=np.float32)
    gl = np.zeros((NUM_GLOBAL,), dtype=np.float32)

    # Plane 0: on-board (always 1 within H×W)
    sp[0, :, :] = 1.0

    # Planes 1-5: stones + liberties
    for r in range(n):
        for c in range(n):
            s = int(g.board[r, c])
            if s == EMPTY:
                continue
            if s == pla:
                sp[1, r, c] = 1.0
            else:
                sp[2, r, c] = 1.0
            libs = g.liberties_at(r, c)
            if libs == 1:
                sp[3, r, c] = 1.0
            elif libs == 2:
                sp[4, r, c] = 1.0
            elif libs == 3:
                sp[5, r, c] = 1.0

    # Plane 6: ko-banned point (simple ko)
    for r in range(n):
        for c in range(n):
            if g.board[r, c] != EMPTY:
                continue
            if g.is_ko_ban(r * n + c):
                sp[6, r, c] = 1.0

    # Plane 7, 8: zero (encore-only / unused)

    # Planes 9-13: location of past 5 moves (i=0 most recent → plane 9)
    # Pass moves at ply i set gl[i]=1.
    for i in range(5):
        act = g.recent_action(i)
        if act == -2:
            continue
        plane = 9 + i
        if act == PASS:
            gl[i] = 1.0
        else:
            r, c = divmod(act, n)
            if 0 <= r < n and 0 <= c < n:
                sp[plane, r, c] = 1.0

    # Planes 14-17: ladders — zeroed (matches C++ port).
    # Planes 18-19: Tromp-Taylor area scoring.
    area = g.compute_area()
    for r in range(n):
        for c in range(n):
            owner = int(area[r, c])
            if owner == pla:
                sp[18, r, c] = 1.0
            elif owner == opp:
                sp[19, r, c] = 1.0

    # Planes 20-21: zero (no encore).

    # Globals -----------------------------------------------------------
    # gl[0..4]: pass flags set above.

    # gl[5]: self-komi / 20
    self_komi = g.komi if pla == WHITE else -g.komi
    bound = float(n * n + 1)
    self_komi = max(-bound, min(bound, self_komi))
    gl[5] = self_komi / 20.0

    # gl[6,7]: ko rule (simple ko) → 0,0
    # gl[8]:   suicide allowed → 0
    # gl[9]:   scoring (area) → 0
    # gl[10,11]: tax → 0,0
    # gl[12,13]: encore → 0,0

    # gl[14]: pass-would-end-phase
    if g.consecutive_passes >= 1:
        gl[14] = 1.0

    # gl[15]: triangular wave on self_komi
    floor_k = math.floor(self_komi)
    delta = self_komi - floor_k
    if int(floor_k) & 1:
        delta += 1.0
    if delta < 0.5:
        wave = delta
    elif delta < 1.5:
        wave = 1.0 - delta
    else:
        wave = delta - 2.0
    gl[15] = wave

    # gl[16,17,18]: unused
    return sp, gl


# ─────────────────────────────────────────────────────────────────
#  Game playing
# ─────────────────────────────────────────────────────────────────


def _softmax(x: np.ndarray, t: float) -> np.ndarray:
    z = x / max(t, 1e-6)
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


def _kata1_pick_move(sess, board: "GoBoard", rng: random.Random,
                     temperature: float) -> int:
    """Run kata1 ONNX on the current position; sample a move from
    softmax(policy_logits / T) restricted to legal moves.  Returns a
    flat action id (board action 0..n*n-1, or PASS=-1)."""
    sp, gl = encode_katago(board)
    sp4 = sp[np.newaxis].astype(np.float32)
    gl2 = gl[np.newaxis].astype(np.float32)
    out = sess.run(["policy_logits"], {
        "state_spatial": sp4,
        "state_global":  gl2,
    })
    pl = out[0][0]   # [n*n + 1] — last entry is PASS
    n = board.n
    # Build a mask over [0..n*n] U {pass-as-index n*n} matching pl's layout.
    mask = np.full(pl.shape, -1e9, dtype=np.float32)
    for r in range(n):
        for c in range(n):
            a = r * n + c
            if board.is_legal(a) and not board._is_eye(r, c, board.current):
                mask[a] = 0.0
    mask[n * n] = 0.0       # PASS is always legal
    probs = _softmax(pl + mask, temperature)
    if not np.isfinite(probs.sum()) or probs.sum() < 1e-9:
        return PASS
    idx = int(np.random.default_rng(rng.randint(0, 2**31 - 1)).choice(len(probs), p=probs))
    return PASS if idx == n * n else idx


def play_calibration_game(rng: random.Random,
                          board_size: int,
                          komi: float,
                          max_moves: int,
                          snapshot_at: List[int],
                          ort_session=None,
                          strong_frac: float = 0.0,
                          temperature: float = 1.0) -> List[GoBoard]:
    """Play one game.  Each move:
      - With probability `strong_frac` (and only when ort_session is
        available), sample from kata1's softmax(policy / T).
      - Otherwise pick uniformly at random from legal-non-eye moves.
    Returns the snapshots taken at moves in `snapshot_at`.
    """
    g = GoBoard(n=board_size, komi=komi)
    snapshot_set = set(snapshot_at)
    snapshots: List[GoBoard] = []

    while g.move_count < max_moves and g.consecutive_passes < 2:
        if g.move_count in snapshot_set:
            snapshots.append(g.copy())

        action: int
        use_strong = (ort_session is not None) and (rng.random() < strong_frac)
        if use_strong:
            try:
                action = _kata1_pick_move(ort_session, g, rng, temperature)
            except Exception:
                action = PASS
        else:
            legal = g.legal_actions(exclude_eyes=True)
            if not legal:
                action = PASS
            else:
                non_pass = [a for a in legal if a != PASS]
                # Small pass probability after move 30 to add pass-history.
                if not non_pass or (rng.random() < 0.02 and g.move_count > 30):
                    action = PASS
                else:
                    action = rng.choice(non_pass)
        try:
            g.play(action)
        except ValueError:
            g.play(PASS)

    if g.move_count in snapshot_set:
        snapshots.append(g.copy())
    return snapshots


# ─────────────────────────────────────────────────────────────────
#  Driver
# ─────────────────────────────────────────────────────────────────


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--output-dir", required=True,
                    help="Where to drop the calibration tensors and dataset txt files.")
    ap.add_argument("--num-games", type=int, default=80,
                    help="Self-play games (default 80). Each yields up to 12 snapshots.")
    ap.add_argument("--board-size", type=int, default=9)
    ap.add_argument("--komi", type=float, default=7.5)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--snapshots-per-game", type=str,
                    default="2,5,8,12,16,21,28,36,45,55,65,75",
                    help="Comma-separated move counts at which to snapshot. "
                         "Default mixes opening (2-16), midgame (21-45), "
                         "endgame (55-75).")
    ap.add_argument("--max-moves", type=int, default=90,
                    help="Hard cap on game length (default 90 plies).")
    ap.add_argument("--onnx-model", type=str, default=None,
                    help="kata1 ONNX (e.g. models/kata1-b10c128.a733.bs1.unshared.onnx). "
                         "Required for --strong-frac > 0.")
    ap.add_argument("--strong-frac", type=float, default=0.0,
                    help="Fraction of moves picked from kata1's policy "
                         "(softmax/temperature sampling).  0.0 = pure random. "
                         "0.7 = 70%% kata1, 30%% random per move.")
    ap.add_argument("--random-game-frac", type=float, default=0.2,
                    help="Fraction of *whole games* played all-random regardless "
                         "of --strong-frac; gives the calibration set unfamiliar "
                         "tactical positions kata1 wouldn't reach on its own.")
    ap.add_argument("--temperature", type=float, default=1.0,
                    help="Softmax temperature for kata1 policy sampling.")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    snap_at = [int(x) for x in args.snapshots_per_game.split(",")]

    sess = None
    if args.strong_frac > 0:
        if not args.onnx_model:
            raise SystemExit("--strong-frac > 0 requires --onnx-model")
        import onnxruntime as ort  # lazy import — random-only path doesn't need it
        sess = ort.InferenceSession(args.onnx_model,
                                    providers=["CPUExecutionProvider"])
        print(f"loaded ORT session from {args.onnx_model}", file=sys.stderr)

    out_root = Path(args.output_dir)
    sp_dir = out_root / "state_spatial"
    gl_dir = out_root / "state_global"
    sp_dir.mkdir(parents=True, exist_ok=True)
    gl_dir.mkdir(parents=True, exist_ok=True)

    sp_paths, gl_paths = [], []
    sample_idx = 0
    phase_counts = {"opening": 0, "midgame": 0, "endgame": 0}

    for game_i in range(args.num_games):
        # `random-game-frac` of games go pure-random (strong_frac=0) for
        # tactical diversity even when kata1 is being used overall.
        is_random_game = rng.random() < args.random_game_frac
        eff_strong = 0.0 if is_random_game else args.strong_frac
        snapshots = play_calibration_game(rng,
                                          board_size=args.board_size,
                                          komi=args.komi,
                                          max_moves=args.max_moves,
                                          snapshot_at=snap_at,
                                          ort_session=sess,
                                          strong_frac=eff_strong,
                                          temperature=args.temperature)
        for snap in snapshots:
            sp, gl = encode_katago(snap)
            # Acuity TEXT-mode loader calls np.loadtxt on each line of
            # the dataset txt.  Each .tensor file is plain ASCII text:
            # whitespace-separated floats in row-major (flat) order.
            # The inputmeta `shape:` then dictates the reshape (NCHW
            # for spatial, flat for global).
            sp_path = sp_dir / f"{sample_idx:04d}.tensor"
            gl_path = gl_dir / f"{sample_idx:04d}.tensor"
            np.savetxt(sp_path, sp.astype(np.float32).flatten(), fmt="%.6g")
            np.savetxt(gl_path, gl.astype(np.float32).flatten(), fmt="%.6g")
            sp_paths.append(sp_path)
            gl_paths.append(gl_path)

            if snap.move_count <= 15:
                phase_counts["opening"] += 1
            elif snap.move_count <= 45:
                phase_counts["midgame"] += 1
            else:
                phase_counts["endgame"] += 1

            sample_idx += 1
        if (game_i + 1) % 10 == 0:
            print(f"  played {game_i + 1}/{args.num_games} games → {sample_idx} samples", file=sys.stderr)

    # Acuity TEXT mode: each line is the path to one sample.
    # Paths are interpreted relative to the inputmeta location
    # (i.e. wherever pegasus.py is run from), so write absolute paths.
    sp_list = out_root / "dataset0_spatial.txt"
    gl_list = out_root / "dataset1_global.txt"
    with open(sp_list, "w") as f:
        f.writelines(f"{p.resolve()}\n" for p in sp_paths)
    with open(gl_list, "w") as f:
        f.writelines(f"{p.resolve()}\n" for p in gl_paths)

    print(f"\nGenerated {sample_idx} calibration samples")
    print(f"  opening: {phase_counts['opening']}")
    print(f"  midgame: {phase_counts['midgame']}")
    print(f"  endgame: {phase_counts['endgame']}")
    print(f"  state_spatial → {sp_list}")
    print(f"  state_global  → {gl_list}")


if __name__ == "__main__":
    main()
