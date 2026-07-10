#!/usr/bin/env python3
"""Generate calibration data for RKNN int8 / hybrid quantisation.

Quantisation needs a representative sample of input activations; the toolkit
computes per-tensor scales from the values it sees while replaying calibration
inputs. For Go networks the input distribution is dominated by:

  * sparse one-hot stone planes (mostly 0, occasionally 1)
  * liberty / area planes that vary with game phase (opening → endgame)
  * board-history planes that encode the last N moves

Random-noise calibration would over-estimate plane variance and under-estimate
the prior of "empty intersection" — we'd quantise the trunk against a
distribution the network never actually sees. Instead we **drive self-play
with the model itself** and dump real positions across a spread of game
phases.

The encoder mirrors `src/katago_inputs.cpp` (V7, 22 spatial + 19 global) for
KataGo-format ONNX files and `src/game.cpp::encode` (17 spatial planes for
history_length=8) for MiniGo-format files. ONNX format is auto-detected from
the model's graph inputs.

Usage
-----
    # KataGo dual-input model, fp16 self-play
    python tools/rknn_calibration.py \
        --onnx models/kata1-b10c128.onnx \
        --output calib/kata1 \
        --num-positions 200

    # MiniGo single-input model
    python tools/rknn_calibration.py \
        --onnx models/best.onnx \
        --output calib/minigo \
        --num-positions 200

The output directory contains one .npy per sample for each input tensor and a
`dataset.txt` manifest in the format `rknn-toolkit2` expects:

    calib/kata1/state_spatial_0000.npy calib/kata1/state_global_0000.npy
    calib/kata1/state_spatial_0001.npy calib/kata1/state_global_0001.npy
    ...

(For single-input MiniGo models, each line lists just one .npy.)
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

# ────────────────────────────────────────────────────────────────────────
#  Minimal Go board (port of src/game.cpp; correctness > speed)
# ────────────────────────────────────────────────────────────────────────

EMPTY, BLACK, WHITE = 0, 1, 2
PASS = -1


def _opp(s: int) -> int:
    return WHITE if s == BLACK else BLACK


@dataclass
class GoGame:
    """Minimal Go board with simple-ko, captures, history.

    Mirrors the data needed by both the MiniGo and KataGo V7 encoders:
      - board[r][c] in {EMPTY, BLACK, WHITE}
      - prev_board for ko detection
      - last 5 actions for V7 history planes
      - last `history_length` board snapshots for the MiniGo encoder
    """
    n: int = 9
    komi: float = 7.5
    history_length: int = 8

    board: np.ndarray = field(default=None)
    prev_board: Optional[np.ndarray] = None
    current_player: int = BLACK
    move_count: int = 0
    consecutive_passes: int = 0
    game_over: bool = False
    snapshots: List[np.ndarray] = field(default_factory=list)
    recent_actions: List[int] = field(default_factory=list)  # most-recent at index 0

    def __post_init__(self):
        self.board = np.zeros((self.n, self.n), dtype=np.int8)

    # ── Group / liberty helpers ─────────────────────────────────────
    def _group(self, brd: np.ndarray, r: int, c: int) -> Tuple[List[Tuple[int, int]], int]:
        """Return (stones, liberties) of the group at (r,c) on `brd`."""
        color = int(brd[r, c])
        if color == EMPTY:
            return [], 0
        n = self.n
        stack = [(r, c)]
        seen = {(r, c)}
        libs = set()
        stones = []
        while stack:
            pr, pc = stack.pop()
            stones.append((pr, pc))
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nr, nc = pr + dr, pc + dc
                if not (0 <= nr < n and 0 <= nc < n):
                    continue
                s = int(brd[nr, nc])
                if s == EMPTY:
                    libs.add((nr, nc))
                elif s == color and (nr, nc) not in seen:
                    seen.add((nr, nc))
                    stack.append((nr, nc))
        return stones, len(libs)

    def _try_play(self, action: int) -> Optional[np.ndarray]:
        """Return the board that *would* result from playing `action`, or None
        if illegal (suicide / simple-ko). Captures are applied first."""
        if action == PASS:
            return self.board.copy()
        n = self.n
        if not (0 <= action < n * n):
            return None
        r, c = divmod(action, n)
        if self.board[r, c] != EMPTY:
            return None
        test = self.board.copy()
        test[r, c] = self.current_player
        opp = _opp(self.current_player)
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r + dr, c + dc
            if not (0 <= nr < n and 0 <= nc < n):
                continue
            if test[nr, nc] != opp:
                continue
            stones, libs = self._group(test, nr, nc)
            if libs == 0:
                for sr, sc in stones:
                    test[sr, sc] = EMPTY
        # Suicide
        own, own_libs = self._group(test, r, c)
        if own_libs == 0:
            return None
        # Simple ko
        if self.prev_board is not None and np.array_equal(test, self.prev_board):
            return None
        return test

    def is_legal(self, action: int) -> bool:
        if self.game_over:
            return False
        if action == PASS:
            return True
        return self._try_play(action) is not None

    def is_ko_ban(self, action: int) -> bool:
        """V7 plane 6: empty-cell move that would be banned by simple ko."""
        if action == PASS or self.prev_board is None:
            return False
        n = self.n
        if not (0 <= action < n * n):
            return False
        r, c = divmod(action, n)
        if self.board[r, c] != EMPTY:
            return False
        test = self.board.copy()
        test[r, c] = self.current_player
        opp = _opp(self.current_player)
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r + dr, c + dc
            if not (0 <= nr < n and 0 <= nc < n):
                continue
            if test[nr, nc] != opp:
                continue
            stones, libs = self._group(test, nr, nc)
            if libs == 0:
                for sr, sc in stones:
                    test[sr, sc] = EMPTY
        own, own_libs = self._group(test, r, c)
        if own_libs == 0:
            return False
        return np.array_equal(test, self.prev_board)

    def play(self, action: int) -> None:
        next_board = self._try_play(action)
        if next_board is None:
            raise RuntimeError(f"illegal move {action}")
        self.recent_actions.insert(0, action)
        del self.recent_actions[5:]
        self.prev_board = self.board.copy()
        self.board = next_board
        if action == PASS:
            self.consecutive_passes += 1
            if self.consecutive_passes >= 2:
                self.game_over = True
        else:
            self.consecutive_passes = 0
        self.current_player = _opp(self.current_player)
        self.move_count += 1
        # Snapshots: most-recent appended to tail
        self.snapshots.append(self.board.copy())
        if len(self.snapshots) > self.history_length:
            self.snapshots = self.snapshots[-self.history_length :]


# ────────────────────────────────────────────────────────────────────────
#  Encoders (mirror the C++ ones)
# ────────────────────────────────────────────────────────────────────────

KATAGO_NUM_SPATIAL = 22
KATAGO_NUM_GLOBAL = 19


def _compute_area(brd: np.ndarray) -> np.ndarray:
    """Tromp-Taylor area: empty regions surrounded by exactly one color
    are assigned to that color; mixed regions stay EMPTY (dame)."""
    n = brd.shape[0]
    area = brd.copy()
    visited = np.zeros((n, n), dtype=bool)
    for r0 in range(n):
        for c0 in range(n):
            if brd[r0, c0] != EMPTY or visited[r0, c0]:
                continue
            stack = [(r0, c0)]
            visited[r0, c0] = True
            region = []
            touch_b = touch_w = False
            while stack:
                r, c = stack.pop()
                region.append((r, c))
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr, nc = r + dr, c + dc
                    if not (0 <= nr < n and 0 <= nc < n):
                        continue
                    s = int(brd[nr, nc])
                    if s == EMPTY:
                        if not visited[nr, nc]:
                            visited[nr, nc] = True
                            stack.append((nr, nc))
                    elif s == BLACK:
                        touch_b = True
                    elif s == WHITE:
                        touch_w = True
            if touch_b and not touch_w:
                owner = BLACK
            elif touch_w and not touch_b:
                owner = WHITE
            else:
                owner = EMPTY
            for r, c in region:
                area[r, c] = owner
    return area


def encode_katago_v7(game: GoGame) -> Tuple[np.ndarray, np.ndarray]:
    """Mirror src/katago_inputs.cpp::encode_for_katago.

    Returns (state_spatial[22, H, W], state_global[19]) as float32.
    Deliberate simplifications match the C++ port:
      - Plane 7 (encore ko-recap): zero
      - Plane 8 (unused): zero
      - Planes 14-17 (ladder features): zero (TODO in C++)
      - Planes 20-21 (encore start colors): zero
      - Globals 12-13 (encore phase): zero
      - Globals 16-18 (unused in V7): zero
    """
    n = game.n
    pla = game.current_player
    opp = _opp(pla)
    sp = np.zeros((KATAGO_NUM_SPATIAL, n, n), dtype=np.float32)
    gl = np.zeros((KATAGO_NUM_GLOBAL,), dtype=np.float32)

    # Plane 0: on-board (always 1)
    sp[0, :, :] = 1.0

    # Planes 1-5: stones + liberty count (1, 2, 3 libs)
    for r in range(n):
        for c in range(n):
            s = int(game.board[r, c])
            if s == EMPTY:
                continue
            if s == pla:
                sp[1, r, c] = 1.0
            else:
                sp[2, r, c] = 1.0
            _, libs = game._group(game.board, r, c)
            if libs == 1:
                sp[3, r, c] = 1.0
            elif libs == 2:
                sp[4, r, c] = 1.0
            elif libs == 3:
                sp[5, r, c] = 1.0

    # Plane 6: simple-ko-banned point
    for r in range(n):
        for c in range(n):
            if game.board[r, c] != EMPTY:
                continue
            if game.is_ko_ban(r * n + c):
                sp[6, r, c] = 1.0

    # Planes 9-13: history of last 5 actions (1 ply..5 ply ago).
    # If a recent action was a pass, set globals[i] = 1 instead.
    for i in range(min(5, len(game.recent_actions))):
        act = game.recent_actions[i]
        plane = 9 + i
        if act == PASS:
            gl[i] = 1.0
        else:
            r, c = divmod(act, n)
            sp[plane, r, c] = 1.0

    # Planes 18-19: Tromp-Taylor area (player owned / opponent owned)
    area = _compute_area(game.board)
    for r in range(n):
        for c in range(n):
            owner = int(area[r, c])
            if owner == pla:
                sp[18, r, c] = 1.0
            elif owner == opp:
                sp[19, r, c] = 1.0

    # Globals
    # gl[0..4]: pass flags (set above)
    # gl[5]: self-komi / 20 (clamped to ±(HW+1))
    self_komi = game.komi if pla == WHITE else -game.komi
    bound = float(n * n + 1)
    self_komi = max(-bound, min(bound, self_komi))
    gl[5] = self_komi / 20.0
    # gl[6..13]: rule flags + encore phase — all zero (Tromp-Taylor area, no encore)
    # gl[14]: pass-would-end-phase
    if game.consecutive_passes >= 1:
        gl[14] = 1.0
    # gl[15]: komi parity wave
    floor_k = float(np.floor(self_komi))
    delta = self_komi - floor_k
    if int(floor_k) & 1:
        delta += 1.0
    if delta < 0.5:
        wave = delta
    elif delta < 1.5:
        wave = 1.0 - delta
    else:
        wave = delta - 2.0
    gl[15] = float(wave)
    # gl[16..18]: unused

    return sp, gl


def encode_minigo(game: GoGame, history_length: int = 8) -> np.ndarray:
    """Mirror src/game.cpp::GoGame::encode.

    Returns state[2*history_length + 1, H, W]:
      planes [0 .. H-1]:   most-recent first, current_player presence
      planes [H .. 2H-1]:  most-recent first, opponent presence
      plane  [2H]:         color (all-ones if BLACK to play, else zero)
    """
    n = game.n
    H = history_length
    planes = 2 * H + 1
    out = np.zeros((planes, n, n), dtype=np.float32)
    pla = game.current_player
    opp = _opp(pla)
    snaps = game.snapshots[-H:]
    # snapshots[-1] is most-recent
    snaps_recent_first = list(reversed(snaps))
    for i, snap in enumerate(snaps_recent_first):
        for r in range(n):
            for c in range(n):
                s = int(snap[r, c])
                if s == pla:
                    out[i, r, c] = 1.0
                elif s == opp:
                    out[H + i, r, c] = 1.0
    if pla == BLACK:
        out[2 * H, :, :] = 1.0
    return out


# ────────────────────────────────────────────────────────────────────────
#  ONNX driver — runs the model to drive realistic self-play
# ────────────────────────────────────────────────────────────────────────


def _detect_format(onnx_path: str) -> dict:
    """Inspect graph inputs to figure out what to feed the model.

    Returns a dict with:
      format: 'katago' or 'minigo'
      board_size: int
      input_channels: int (spatial channel count)
      input_global_channels: int (KataGo only, else 0)
      input_names: list[str]
      output_names: list[str]
      history_length: int (MiniGo only — derived from spatial channel count)
    """
    import onnx

    m = onnx.load(onnx_path)
    inputs = {}
    for inp in m.graph.input:
        dims = []
        for d in inp.type.tensor_type.shape.dim:
            if d.dim_param:
                dims.append(d.dim_param)
            else:
                dims.append(int(d.dim_value))
        inputs[inp.name] = dims
    outputs = [o.name for o in m.graph.output]

    if "state_spatial" in inputs and "state_global" in inputs:
        sp_dims = inputs["state_spatial"]   # [batch, C, H, W]
        gl_dims = inputs["state_global"]    # [batch, G]
        return {
            "format": "katago",
            "board_size": int(sp_dims[2]),
            "input_channels": int(sp_dims[1]),
            "input_global_channels": int(gl_dims[1]),
            "input_names": ["state_spatial", "state_global"],
            "output_names": outputs,
            "history_length": 0,
        }
    if "state" in inputs:
        sp_dims = inputs["state"]           # [batch, C, H, W]
        c = int(sp_dims[1])
        # MiniGo encoder: planes = 2*history_length + 1
        history_length = (c - 1) // 2
        return {
            "format": "minigo",
            "board_size": int(sp_dims[2]),
            "input_channels": c,
            "input_global_channels": 0,
            "input_names": ["state"],
            "output_names": outputs,
            "history_length": history_length,
        }
    raise RuntimeError(f"Unknown ONNX input layout: {list(inputs)}")


def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


def _select_action(
    policy_logits: np.ndarray,
    game: GoGame,
    temperature: float,
    rng: np.random.Generator,
) -> int:
    """Sample a move from the policy logits (with simple-ko / suicide filtering).

    Falls back to PASS if no spatial move is legal. Handles models that include
    or omit the pass action (`policy_logits.shape[-1]` is `n*n+1` or `n*n`).
    """
    n = game.n
    has_pass = int(policy_logits.shape[-1]) == n * n + 1
    p = _softmax(policy_logits.astype(np.float64) / max(temperature, 1e-3))
    legal = np.zeros_like(p)
    if has_pass:
        legal[n * n] = 1.0  # pass always legal
    for action in range(n * n):
        if game.is_legal(action):
            legal[action] = 1.0
    p = p * legal
    s = p.sum()
    if s <= 0:
        return PASS
    p = p / s
    return int(rng.choice(len(p), p=p))


def _action_to_index(act: int, n: int) -> int:
    return n * n if act == PASS else act


def _index_to_action(idx: int, n: int) -> int:
    return PASS if idx == n * n else idx


def collect_calibration(
    onnx_path: str,
    output_dir: str,
    num_positions: int,
    games: int,
    seed: int,
    temperature: float,
    skip_first_n: int,
    every_n: int,
    providers: Optional[List[str]] = None,
) -> Tuple[dict, int]:
    """Drive self-play with the ONNX model and dump positions.

    Returns (info, num_dumped).
    """
    import onnxruntime as ort

    info = _detect_format(onnx_path)
    n = info["board_size"]

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    sess_opts = ort.SessionOptions()
    sess_opts.log_severity_level = 3  # warn+
    sess = ort.InferenceSession(
        onnx_path,
        sess_options=sess_opts,
        providers=providers or ["CPUExecutionProvider"],
    )
    rng = np.random.default_rng(seed)
    fmt = info["format"]

    manifest_lines: List[str] = []
    dumped = 0
    game_idx = 0
    while dumped < num_positions and game_idx < games:
        game = GoGame(n=n, komi=7.5, history_length=max(info.get("history_length", 8), 1))
        max_moves = n * n * 2
        ply = 0
        while not game.game_over and ply < max_moves:
            # Encode current state
            if fmt == "katago":
                sp, gl = encode_katago_v7(game)
                feeds = {
                    "state_spatial": sp[None, :, :, :].astype(np.float32),
                    "state_global": gl[None, :].astype(np.float32),
                }
            else:
                state = encode_minigo(game, info["history_length"])
                feeds = {"state": state[None, :, :, :].astype(np.float32)}

            # Decide whether to dump this position. Skip the empty-board prefix
            # so we get a spread of game phases, then dump every N plies.
            should_dump = ply >= skip_first_n and ((ply - skip_first_n) % every_n == 0)
            if should_dump and dumped < num_positions:
                if fmt == "katago":
                    sp_path = out / f"state_spatial_{dumped:04d}.npy"
                    gl_path = out / f"state_global_{dumped:04d}.npy"
                    np.save(sp_path, feeds["state_spatial"][0])
                    np.save(gl_path, feeds["state_global"][0])
                    manifest_lines.append(
                        f"{sp_path.name} {gl_path.name}"
                    )
                else:
                    sp_path = out / f"state_{dumped:04d}.npy"
                    np.save(sp_path, feeds["state"][0])
                    manifest_lines.append(sp_path.name)
                dumped += 1

            # Run the network and pick a move
            raw_outputs = sess.run(None, feeds)
            # Output 0 should be policy_logits in both formats
            policy_logits = np.asarray(raw_outputs[0])[0]
            # Use a small amount of randomness for variety; temp=1 in opening,
            # cooling toward 0 (greedy) as the game progresses.
            t = temperature * max(0.05, 1.0 - ply / float(max_moves))
            idx = _select_action(policy_logits, game, t, rng)
            game.play(_index_to_action(idx, n))
            ply += 1

            if dumped >= num_positions:
                break
        game_idx += 1

    manifest = out / "dataset.txt"
    with manifest.open("w") as f:
        for line in manifest_lines:
            f.write(line + "\n")
    return info, dumped


# ────────────────────────────────────────────────────────────────────────
#  CLI
# ────────────────────────────────────────────────────────────────────────


def main():
    ap = argparse.ArgumentParser(
        description="Generate calibration data for RKNN int8 / hybrid quantisation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("--onnx", required=True, help="Source .onnx model")
    ap.add_argument("--output", required=True, help="Output directory for .npy files + dataset.txt")
    ap.add_argument("--num-positions", type=int, default=200,
                    help="How many positions to dump (default 200; rknn-toolkit2 typically wants ≥100)")
    ap.add_argument("--games", type=int, default=64,
                    help="Maximum games to play if num-positions hasn't been reached (default 64)")
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--temperature", type=float, default=1.0,
                    help="Softmax temperature for move sampling (default 1.0). "
                         "Higher = more diverse positions.")
    ap.add_argument("--skip-first", type=int, default=2,
                    help="Plies to play before starting to dump (default 2). "
                         "Skipping the very first plies avoids over-sampling the opening prior.")
    ap.add_argument("--every", type=int, default=2,
                    help="Dump every N plies after --skip-first (default 2)")
    ap.add_argument("--providers", nargs="+",
                    help="ORT provider list (default CPUExecutionProvider). "
                         "Set to e.g. CUDAExecutionProvider CPUExecutionProvider to use a GPU.")
    args = ap.parse_args()

    info, dumped = collect_calibration(
        onnx_path=args.onnx,
        output_dir=args.output,
        num_positions=args.num_positions,
        games=args.games,
        seed=args.seed,
        temperature=args.temperature,
        skip_first_n=args.skip_first,
        every_n=args.every,
        providers=args.providers,
    )
    print(f"format         : {info['format']}")
    print(f"board_size     : {info['board_size']}")
    print(f"input_channels : {info['input_channels']}"
          + (f" + {info['input_global_channels']} global"
             if info["format"] == "katago" else ""))
    print(f"positions      : {dumped}")
    print(f"output         : {args.output}/")
    print(f"manifest       : {args.output}/dataset.txt")
    if dumped < args.num_positions:
        print(f"WARN: requested {args.num_positions} positions but only "
              f"collected {dumped} (raise --games or --every).")


if __name__ == "__main__":
    main()
