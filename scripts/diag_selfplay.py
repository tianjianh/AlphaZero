#!/usr/bin/env python3
"""Self-play diagnostic — decisive-rate, game length, policy concentration,
opening diversity, across one or more training/selfplay/iter_NNNN dirs.

Usage:
    /venv/alphazero/bin/python3 scripts/diag_selfplay.py training/selfplay/iter_0010
    /venv/alphazero/bin/python3 scripts/diag_selfplay.py training/selfplay/iter_0001 \
                                                         training/selfplay/iter_0005 \
                                                         training/selfplay/iter_0010

Nothing to train, nothing to load — reads V3 .bin/.bin.zst records only.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import math
import os
import sys
from collections import Counter

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from train import _decompress, _parse_file_bulk

BOARD_ROWS = 10
BOARD_COLS = 9
HISTORY = 4
INPUT_CHANNELS = HISTORY * 14 + 1   # 57
MAX_MOVES_PER_GAME = 300            # matches config.h


def entropy_nats(p: np.ndarray) -> float:
    p = p[p > 1e-12]
    if p.size == 0:
        return 0.0
    return float(-(p * np.log(p)).sum())


def top_k_mass(p: np.ndarray, k: int) -> float:
    if p.size == 0:
        return 0.0
    idx = np.argpartition(p, -k)[-k:] if p.size > k else np.arange(p.size)
    return float(p[idx].sum())


def state_hash(state: np.ndarray) -> str:
    return hashlib.sha1(state.tobytes()).hexdigest()[:16]


def summarize_iter(iter_dir: str) -> dict:
    files = sorted(
        glob.glob(os.path.join(iter_dir, "**", "*.bin.zst"), recursive=True) +
        glob.glob(os.path.join(iter_dir, "**", "*.bin"), recursive=True))
    if not files:
        return {"iter_dir": iter_dir, "games": 0}

    # Per-game scalars
    game_outcomes = []       # +1 (decisive), 0 (draw)
    game_win_signs = []      # +1 red-wins, -1 red-loses, 0 draw (wrt first record)
    game_moves = []          # number of real moves (records / 2)
    game_hit_limit = []      # bool: records == 2 * MAX_MOVES_PER_GAME

    # Policy stats across first record of every game
    first_entropies = []
    first_top1 = []
    first_top5_mass = []

    # Opening diversity — hash the state the player chose from at moves
    # 1, 5, 10.  Record index for move M is 2*(M-1) because every move
    # writes 2 records (original + horizontal mirror).
    open_hashes = {1: set(), 5: set(), 10: set()}

    for fpath in files:
        try:
            data = _decompress(fpath)
            parsed = _parse_file_bulk(data, BOARD_ROWS, BOARD_COLS, INPUT_CHANNELS)
        except Exception as exc:
            print(f"  [skip] {fpath}: {exc}", file=sys.stderr)
            continue
        if parsed is None:
            continue
        states, policies, values, _ownerships, _opp_actions = parsed
        n_records = len(values)
        if n_records == 0:
            continue

        moves_played = n_records // 2   # two records (orig + mirror) per move
        game_moves.append(moves_played)
        game_hit_limit.append(moves_played >= MAX_MOVES_PER_GAME)

        # Outcome from the first record's viewpoint (red at move 1).
        v0 = float(values[0])
        game_outcomes.append(1 if abs(v0) > 0.5 else 0)
        game_win_signs.append(int(np.sign(v0)) if abs(v0) > 0.5 else 0)

        # Policy shape at move 1 (first record).
        p = policies[0]
        first_entropies.append(entropy_nats(p))
        first_top1.append(float(p.max()) if p.size else 0.0)
        first_top5_mass.append(top_k_mass(p, 5))

        # Opening diversity — record i corresponds to move (i // 2) + 1.
        # For move M, the record index is 2*(M-1).  We want the STATE the
        # player is choosing from, so record 0 = move 1's state.
        for M in (1, 5, 10):
            idx = 2 * (M - 1)
            if idx < n_records:
                open_hashes[M].add(state_hash(states[idx]))

    games = len(game_moves)
    if games == 0:
        return {"iter_dir": iter_dir, "games": 0}

    moves = np.array(game_moves)
    hit_limit = np.array(game_hit_limit)
    decisive = np.array(game_outcomes)
    win_sign = np.array(game_win_signs)
    ent = np.array(first_entropies)
    top1 = np.array(first_top1)
    top5 = np.array(first_top5_mass)

    n_limit = int(hit_limit.sum())
    n_decisive = int(decisive.sum())
    n_draw = games - n_decisive

    return {
        "iter_dir": iter_dir,
        "games": games,
        "moves_mean": float(moves.mean()),
        "moves_median": float(np.median(moves)),
        "moves_p95": float(np.percentile(moves, 95)),
        "moves_max": int(moves.max()),
        "decisive_pct": 100.0 * n_decisive / games,
        "draw_pct": 100.0 * n_draw / games,
        "red_win_pct": 100.0 * int((win_sign > 0).sum()) / games,
        "red_loss_pct": 100.0 * int((win_sign < 0).sum()) / games,
        "limit_hit_pct": 100.0 * n_limit / games,
        "natural_draw_pct": 100.0 * (n_draw - n_limit) / games,
        "policy_entropy_mean": float(ent.mean()),
        "policy_top1_mean": float(top1.mean()),
        "policy_top5_mean": float(top5.mean()),
        "openings_move1": len(open_hashes[1]),
        "openings_move5": len(open_hashes[5]),
        "openings_move10": len(open_hashes[10]),
    }


def print_summary(s: dict) -> None:
    print(f"\n=== {s['iter_dir']} ===")
    if s["games"] == 0:
        print("  (no games)")
        return

    # Group 1: outcomes
    print(f"  Games:                 {s['games']}")
    print(f"  Decisive rate:         {s['decisive_pct']:5.1f}%   "
          f"(red wins {s['red_win_pct']:.1f}%, red loses {s['red_loss_pct']:.1f}%)")
    print(f"  Draw rate:             {s['draw_pct']:5.1f}%")
    print(f"    hit move-limit:      {s['limit_hit_pct']:5.1f}%   "
          f"← shuffle-draws; value target is 0")
    print(f"    natural draw:        {s['natural_draw_pct']:5.1f}%   "
          f"← repetition / stalemate")

    # Group 2: game length
    print(f"  Moves / game:")
    print(f"    mean:                {s['moves_mean']:.1f}")
    print(f"    median:              {s['moves_median']:.1f}")
    print(f"    95th pct:            {s['moves_p95']:.1f}")
    print(f"    max:                 {s['moves_max']}  "
          f"(cap = {MAX_MOVES_PER_GAME})")

    # Group 3: policy concentration (from MCTS visit distribution)
    # Reference entropies:
    #   uniform over 8100 actions:  log(8100) ≈ 8.999 nats
    #   uniform over 50 legal:      log(50)   ≈ 3.912 nats
    #   concentrated on 1 move:     0.000 nats
    print(f"  MCTS policy (move 1):")
    print(f"    mean entropy:        {s['policy_entropy_mean']:.2f} nats  "
          f"(uniform-over-50 ≈ 3.91)")
    print(f"    mean top-1 mass:     {s['policy_top1_mean']*100:.1f}%")
    print(f"    mean top-5 mass:     {s['policy_top5_mean']*100:.1f}%")

    # Group 4: opening diversity
    g = s["games"]
    print(f"  Opening diversity (unique state hashes of {g} games):")
    print(f"    at move 1:           {s['openings_move1']}")
    print(f"    at move 5:           {s['openings_move5']}")
    print(f"    at move 10:          {s['openings_move10']}")


def verdict(s: dict) -> None:
    if s["games"] == 0:
        return
    issues = []
    if s["decisive_pct"] < 5:
        issues.append(
            f"Decisive-game rate is only {s['decisive_pct']:.1f}%. "
            "Value head has no signal — targets are ~0 for ~everyone.")
    if s["limit_hit_pct"] > 80:
        issues.append(
            f"{s['limit_hit_pct']:.0f}% of games hit the {MAX_MOVES_PER_GAME}-move "
            "limit.  These are shuffle-draws; neither side is making threats.")
    if s["policy_entropy_mean"] < 1.0:
        issues.append(
            f"MCTS policy at move 1 has entropy {s['policy_entropy_mean']:.2f} — "
            "search is almost deterministic, not exploring.")
    if s["openings_move5"] < 0.05 * s["games"]:
        issues.append(
            f"Only {s['openings_move5']} unique positions at move 5 across "
            f"{s['games']} games — Dirichlet noise isn't producing variety.")
    if abs(s["red_win_pct"] - s["red_loss_pct"]) > 10 and s["decisive_pct"] > 5:
        issues.append(
            f"Large red/black asymmetry ({s['red_win_pct']:.1f}% vs "
            f"{s['red_loss_pct']:.1f}%) — maybe over-fitting to one side.")

    if issues:
        print("\nDiagnosis:")
        for i in issues:
            print(f"  • {i}")
    else:
        print("\nDiagnosis: looks healthy (for a self-play run).")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("dirs", nargs="+",
                    help="one or more training/selfplay/iter_NNNN directories")
    args = ap.parse_args()

    summaries = [summarize_iter(d) for d in args.dirs]
    for s in summaries:
        print_summary(s)
        verdict(s)

    # Cross-iteration trend
    if len(summaries) > 1:
        print("\n=== trend ===")
        hdr = (f"  {'iter':24s}  {'games':>6s}  {'dec%':>6s}  "
               f"{'draw%':>6s}  {'limit%':>7s}  {'avgM':>6s}  "
               f"{'entropy':>8s}  {'open5':>6s}")
        print(hdr)
        for s in summaries:
            if s["games"] == 0:
                continue
            print(f"  {os.path.basename(s['iter_dir']):24s}  "
                  f"{s['games']:6d}  "
                  f"{s['decisive_pct']:6.1f}  "
                  f"{s['draw_pct']:6.1f}  "
                  f"{s['limit_hit_pct']:7.1f}  "
                  f"{s['moves_mean']:6.1f}  "
                  f"{s['policy_entropy_mean']:8.2f}  "
                  f"{s['openings_move5']:6d}")


if __name__ == "__main__":
    main()
