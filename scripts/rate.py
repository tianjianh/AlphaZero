#!/usr/bin/env python3
"""
MiniGo AlphaZero — Continuous Rating (optional, monitoring-only)

Periodic round-robin tournament over the k most-recent accepted models.
Does NOT gate anything; purely produces ratings/elo.csv for monitoring.

Elo is computed from accumulated pairwise game counts via a simple
Bradley-Terry MLE (coordinate-ascent fixed-point).  We anchor the oldest
model in the pool at 0 Elo for stability; new models are added to the
rating state as they appear.
"""

import argparse
import glob
import json
import math
import os
import re
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

_STEP_RE = re.compile(r"v(\d+)\.onnx$")


def _step_from(path):
    m = _STEP_RE.search(os.path.basename(path))
    return int(m.group(1)) if m else -1


# ═══════════════════════════════════════════════════════════
#  BT rating state
# ═══════════════════════════════════════════════════════════

class RatingState:
    """Tracks pairwise (wins_a_vs_b) and produces Bradley-Terry Elo ratings."""

    def __init__(self, path):
        self.path = path
        # wins: {(a,b): int}.  a,b are model basenames (strings).
        self.wins = {}
        if os.path.isfile(path):
            try:
                with open(path) as f:
                    blob = json.load(f)
                self.wins = {tuple(k.split("|")): int(v)
                             for k, v in blob.get("wins", {}).items()}
            except (OSError, json.JSONDecodeError, ValueError):
                self.wins = {}

    def save(self):
        blob = {"wins": {"|".join(k): v for k, v in self.wins.items()}}
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        tmp = self.path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(blob, f)
        os.replace(tmp, self.path)

    def add_result(self, a, b, a_wins, b_wins):
        self.wins[(a, b)] = self.wins.get((a, b), 0) + int(a_wins)
        self.wins[(b, a)] = self.wins.get((b, a), 0) + int(b_wins)

    def bradley_terry(self, anchor, max_iter=500, tol=1e-7):
        """Fixed-point MLE for BT strengths.  Returns {model: Elo}."""
        models = set()
        for a, b in self.wins.keys():
            models.add(a); models.add(b)
        if anchor not in models and models:
            anchor = min(models, key=lambda m: _step_from(m))
        if not models:
            return {}

        # Initialize at 1.0 — positive values required for BT
        s = {m: 1.0 for m in models}
        for _ in range(max_iter):
            max_delta = 0.0
            for m in models:
                num = 0.0
                den = 0.0
                for a, b in list(self.wins.keys()):
                    if a == m:
                        n_ab = self.wins.get((a, b), 0)
                        n_ba = self.wins.get((b, a), 0)
                        total = n_ab + n_ba
                        if total == 0:
                            continue
                        num += n_ab
                        den += total / (s[m] + s[b])
                if den <= 0.0:
                    continue
                new_s = num / den
                # Avoid runaway
                new_s = max(new_s, 1e-9)
                max_delta = max(max_delta, abs(math.log(new_s) - math.log(s[m])))
                s[m] = new_s
            if max_delta < tol:
                break

        # Normalize so anchor = 0 Elo
        anchor_s = s[anchor]
        elo = {m: 400.0 * math.log10(s[m] / anchor_s) for m in models}
        return elo


# ═══════════════════════════════════════════════════════════
#  Evaluate output parsing
# ═══════════════════════════════════════════════════════════

_W1_RE = re.compile(r"Model 1 wins:\s*(\d+)")
_W2_RE = re.compile(r"Model 2 wins:\s*(\d+)")
_DR_RE = re.compile(r"Draws:\s*(\d+)")


def parse_eval_output(text):
    w1 = _W1_RE.search(text)
    w2 = _W2_RE.search(text)
    d = _DR_RE.search(text)
    return (int(w1.group(1)) if w1 else 0,
            int(w2.group(1)) if w2 else 0,
            int(d.group(1)) if d else 0)


# ═══════════════════════════════════════════════════════════
#  Logging
# ═══════════════════════════════════════════════════════════

def make_log(log_path):
    os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
    fh = open(log_path, "a", buffering=1)
    def log(tag, **kv):
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        kvs = " ".join(f"{k}={v}" for k, v in kv.items())
        line = f"[{ts}] {tag}" + (f" {kvs}" if kvs else "")
        fh.write(line + "\n")
        print(line, flush=True)
    return log


# ═══════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(description="Continuous rating loop")
    ap.add_argument("--accepted-dir", default="models/accepted")
    ap.add_argument("--ratings-dir", default="ratings")
    ap.add_argument("--log-dir", default="logs/current")
    ap.add_argument("--build-dir", default="build")
    ap.add_argument("--pool-size", type=int, default=5,
                    help="Rate k most-recent accepted models")
    ap.add_argument("--games-per-pair", type=int, default=80)
    ap.add_argument("--sims", type=int, default=200)
    ap.add_argument("--interval", type=float, default=7200.0)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--search-threads", type=int, default=16)
    ap.add_argument("--max-batch", type=int, default=256)
    ap.add_argument("--nn-server-threads", type=int, default=1)
    ap.add_argument("--nn-device-ids", default="0")
    ap.add_argument("--c-puct", type=float, default=1.25)
    ap.add_argument("--komi", type=float, default=7.5)
    ap.add_argument("--win-loss-weight", type=float, default=1.0)
    ap.add_argument("--score-weight", type=float, default=0.0)
    ap.add_argument("--score-scale", type=float, default=18.0)
    args = ap.parse_args()

    os.makedirs(args.ratings_dir, exist_ok=True)
    os.environ.setdefault(
        "MINIGO_TRT_CACHE",
        str((PROJECT_ROOT / "models" / "trt_cache").resolve()))

    log = make_log(os.path.join(args.log_dir, "rate.log"))
    elo_csv = os.path.join(args.ratings_dir, "elo.csv")
    if not os.path.exists(elo_csv):
        with open(elo_csv, "w") as f:
            f.write("wall_time,round,model,step,elo,games\n")

    state = RatingState(os.path.join(args.ratings_dir, "rating_state.json"))
    threads = args.threads if args.threads > 0 else (os.cpu_count() or 4)

    stop = {"flag": False}
    stop_event = threading.Event()
    current_child = {"proc": None}
    def _handle(sig, _f):
        if stop["flag"]:
            p = current_child["proc"]
            if p is not None and p.poll() is None:
                try:
                    p.kill()
                except OSError:
                    pass
            return
        stop["flag"] = True
        stop_event.set()
        log("SIGNAL", sig=sig, note="finishing current pair, then exiting")
    signal.signal(signal.SIGINT, _handle)
    signal.signal(signal.SIGTERM, _handle)

    evaluate_bin = os.path.join(args.build_dir, "evaluate")
    if not os.path.isfile(evaluate_bin):
        log("ERROR", msg=f"evaluate binary not found at {evaluate_bin}")
        sys.exit(2)

    round_no = 0
    while not stop["flag"]:
        accepted = sorted(
            glob.glob(os.path.join(args.accepted_dir, "v*.onnx")))
        if len(accepted) < 2:
            log("RATE_SKIP", reason="not enough accepted models",
                n=len(accepted))
            stop_event.wait(args.interval)
            continue

        pool = accepted[-args.pool_size:]
        round_no += 1
        log("RATE_ROUND_START", n=len(pool),
            newest=os.path.basename(pool[-1]))

        round_aborted = False
        for i in range(len(pool)):
            if round_aborted:
                break
            for j in range(i + 1, len(pool)):
                if stop["flag"]:
                    round_aborted = True
                    break
                m1 = pool[i]
                m2 = pool[j]
                cmd = [
                    evaluate_bin,
                    "--model1", m1, "--model2", m2,
                    "--games", str(args.games_per_pair),
                    "--threads", str(threads),
                    "--search-threads", str(args.search_threads),
                    "--sims", str(args.sims),
                    "--max-batch", str(args.max_batch),
                    "--nn-server-threads", str(args.nn_server_threads),
                    "--nn-device-ids", args.nn_device_ids,
                    "--threshold", "0.5",
                    "--c-puct", str(args.c_puct),
                    "--komi", str(args.komi),
                    "--win-loss-weight", str(args.win_loss_weight),
                    "--score-weight", str(args.score_weight),
                    "--score-scale", str(args.score_scale),
                ]
                try:
                    proc = subprocess.Popen(cmd, cwd=str(PROJECT_ROOT),
                                            stdout=subprocess.PIPE,
                                            stderr=subprocess.PIPE, text=True)
                    current_child["proc"] = proc
                    out, _err = proc.communicate()
                except Exception as e:
                    current_child["proc"] = None
                    log("RATE_SPAWN_ERROR", err=str(e))
                    continue
                current_child["proc"] = None
                w1, w2, draws = parse_eval_output(out or "")
                # In rating we split draws evenly — BT supports fractional
                # wins via rounding; we count half-draws toward each side
                # (integer-rounded by adding to whichever is smaller first).
                a = os.path.basename(m1)
                b = os.path.basename(m2)
                half1 = draws // 2
                half2 = draws - half1
                state.add_result(a, b, w1 + half1, w2 + half2)
                log("RATE_PAIR", a=a, b=b, w1=w1, w2=w2, draws=draws)

        state.save()
        elo = state.bradley_terry(anchor=os.path.basename(pool[0]))

        # Append one row per model in the current pool
        now = time.time()
        for m in pool:
            bn = os.path.basename(m)
            if bn not in elo:
                continue
            games = sum(v for (a, b), v in state.wins.items()
                        if a == bn or b == bn)
            with open(elo_csv, "a") as f:
                f.write(f"{now:.3f},{round_no},{bn},"
                        f"{_step_from(m)},{elo[bn]:.2f},{games}\n")
        log("RATE_ROUND_DONE", round=round_no,
            n=len(pool),
            sleep=f"{args.interval:.0f}s")

        # Interruptible wait — signal wakes us immediately.
        stop_event.wait(args.interval)

    log("SHUTDOWN")


if __name__ == "__main__":
    main()
