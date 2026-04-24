#!/usr/bin/env python3
"""
MiniGo AlphaZero — Continuous Gatekeeper

Polls models/candidates/ every GATE_POLL_INTERVAL seconds.  Whenever there
are one or more candidates:

  1. Newest (by filename — v{step:09d}.onnx lex-sorts as step-sorts) is
     evaluated against models/accepted/latest.
  2. All older candidates in the queue are moved to models/rejected/ as
     stale-drops without playing.
  3. Match: GATE_GAMES games at GATE_SIMS visits.  Go scoring — draws
     count as losses for the candidate (half-integer komi makes draws
     essentially impossible anyway).
  4. score = wins / total.  Accept iff score > GATE_THRESHOLD.
  5. On accept: atomic rename candidate → accepted/, atomic symlink swap
     on accepted/latest.
  6. On reject: atomic rename candidate → rejected/.

Uses build/evaluate.  MAX_BATCH_SIZE matches selfplay so the TRT engine
cache is shared between the two.
"""

import argparse
import glob
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


# ═══════════════════════════════════════════════════════════
#  Filesystem helpers
# ═══════════════════════════════════════════════════════════

def atomic_symlink_swap(target, link_path):
    """Make link_path a symlink to target, atomically.
    os.symlink + os.rename is atomic on POSIX when both live on the same fs."""
    d = os.path.dirname(link_path) or "."
    os.makedirs(d, exist_ok=True)
    # Use a unique tmp name so concurrent runs don't collide
    fd, tmp = tempfile.mkstemp(prefix=".latest.", dir=d)
    os.close(fd)
    os.remove(tmp)  # symlink() needs dest not to exist
    # Store target as relative path if it's in the same dir
    link_dir = os.path.dirname(os.path.abspath(link_path))
    t_abs = os.path.abspath(target)
    if os.path.dirname(t_abs) == link_dir:
        t_store = os.path.basename(t_abs)
    else:
        t_store = os.path.relpath(t_abs, link_dir)
    os.symlink(t_store, tmp)
    os.replace(tmp, link_path)


def atomic_move(src, dst):
    """os.rename — atomic on same-fs moves."""
    os.makedirs(os.path.dirname(dst) or ".", exist_ok=True)
    os.replace(src, dst)


def resolve_accepted_latest(accepted_dir):
    latest = os.path.join(accepted_dir, "latest")
    if os.path.islink(latest):
        target = os.readlink(latest)
        if not os.path.isabs(target):
            target = os.path.normpath(os.path.join(accepted_dir, target))
        return target
    if os.path.isfile(latest):
        return latest
    cands = sorted(glob.glob(os.path.join(accepted_dir, "v*.onnx")))
    return cands[-1] if cands else None


# ═══════════════════════════════════════════════════════════
#  Evaluate output parsing
# ═══════════════════════════════════════════════════════════

_W1_RE = re.compile(r"Model 1 wins:\s*(\d+)")
_W2_RE = re.compile(r"Model 2 wins:\s*(\d+)")
_D_RE = re.compile(r"Draws:\s*(\d+)")


def parse_eval_output(text):
    w1 = _W1_RE.search(text)
    w2 = _W2_RE.search(text)
    d = _D_RE.search(text)
    return (int(w1.group(1)) if w1 else 0,
            int(w2.group(1)) if w2 else 0,
            int(d.group(1)) if d else 0)


# ═══════════════════════════════════════════════════════════
#  Logging
# ═══════════════════════════════════════════════════════════

class CsvLogger:
    def __init__(self, path, columns):
        self.path = path
        self.columns = list(columns)
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        if not os.path.exists(path) or os.path.getsize(path) == 0:
            with open(path, "w") as f:
                f.write(",".join(self.columns) + "\n")

    def write(self, row):
        parts = []
        for c in self.columns:
            v = row.get(c, "")
            parts.append(f"{v:.6g}" if isinstance(v, float) else str(v))
        with open(self.path, "a") as f:
            f.write(",".join(parts) + "\n")


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


_STEP_RE = re.compile(r"v(\d+)\.onnx$")


def _step_from(path):
    m = _STEP_RE.search(os.path.basename(path))
    return int(m.group(1)) if m else -1


# ═══════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(description="Continuous gatekeeper")
    ap.add_argument("--candidates-dir", default="models/candidates")
    ap.add_argument("--accepted-dir", default="models/accepted")
    ap.add_argument("--rejected-dir", default="models/rejected")
    ap.add_argument("--log-dir", default="logs/current")
    ap.add_argument("--build-dir", default="build")
    ap.add_argument("--games", type=int, default=200)
    ap.add_argument("--sims", type=int, default=150)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--poll-interval", type=float, default=30.0)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--search-threads", type=int, default=16)
    ap.add_argument("--max-batch", type=int, default=256)
    ap.add_argument("--nn-server-threads", type=int, default=1)
    ap.add_argument("--nn-device-ids", default="0")
    ap.add_argument("--c-puct", type=float, default=1.25)
    ap.add_argument("--komi", type=float, default=7.5)
    ap.add_argument("--win-loss-weight", type=float, default=1.0)
    ap.add_argument("--score-weight", type=float, default=0.0,
                    help="MCTS score weight during gatekeeper match (KataGo: 0)")
    ap.add_argument("--score-scale", type=float, default=18.0)
    args = ap.parse_args()

    os.makedirs(args.candidates_dir, exist_ok=True)
    os.makedirs(args.accepted_dir, exist_ok=True)
    os.makedirs(args.rejected_dir, exist_ok=True)

    os.environ.setdefault(
        "MINIGO_TRT_CACHE",
        str((PROJECT_ROOT / "models" / "trt_cache").resolve()))

    log = make_log(os.path.join(args.log_dir, "gatekeeper.log"))
    decisions = CsvLogger(
        os.path.join(args.log_dir, "gate_decisions.csv"),
        ["wall_time", "cand_step", "baseline_step", "games", "sims",
         "wins", "draws", "losses", "score", "threshold", "verdict",
         "match_duration_s"])

    threads = args.threads if args.threads > 0 else (os.cpu_count() or 4)

    # Graceful shutdown: first signal sets flag and wakes any interruptible
    # sleep (stop_event.wait) so we exit the poll loop promptly; second
    # signal kills the in-flight evaluate child.
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
        log("SIGNAL", sig=sig, note="finishing current match, then exiting")
    signal.signal(signal.SIGINT, _handle)
    signal.signal(signal.SIGTERM, _handle)

    evaluate_bin = os.path.join(args.build_dir, "evaluate")
    if not os.path.isfile(evaluate_bin):
        log("ERROR", msg=f"evaluate binary not found at {evaluate_bin}")
        sys.exit(2)

    log("GATE_START", games=args.games, sims=args.sims,
        threshold=args.threshold, cands_dir=args.candidates_dir)

    while not stop["flag"]:
        cands = sorted(
            glob.glob(os.path.join(args.candidates_dir, "v*.onnx")))
        if not cands:
            # Interruptible wait — stop_event.set() from the signal
            # handler wakes us immediately instead of stalling up to
            # poll_interval seconds after Ctrl-C.
            stop_event.wait(args.poll_interval)
            continue

        newest = cands[-1]
        stale = cands[:-1]
        for s in stale:
            target = os.path.join(args.rejected_dir, os.path.basename(s))
            try:
                atomic_move(s, target)
                log("GATE_STALE_DROP", cand=os.path.basename(s))
            except OSError as e:
                log("GATE_STALE_DROP_FAIL", cand=s, err=str(e))

        baseline = resolve_accepted_latest(args.accepted_dir)
        if baseline is None:
            # First-run race: no accepted model yet.  run_continuous.py init
            # should seed one before starting the gatekeeper, but if it
            # didn't, promote the first candidate unconditionally.
            log("GATE_FIRST_SEED", cand=os.path.basename(newest))
            promoted = os.path.join(args.accepted_dir, os.path.basename(newest))
            atomic_move(newest, promoted)
            atomic_symlink_swap(promoted,
                                os.path.join(args.accepted_dir, "latest"))
            log("GATE_ACCEPT", cand=os.path.basename(promoted), score=1.0,
                baseline="<bootstrap>")
            continue

        cand_step = _step_from(newest)
        base_step = _step_from(baseline)

        log("GATE_MATCH_START", cand=os.path.basename(newest),
            baseline=os.path.basename(baseline),
            games=args.games, sims=args.sims)

        cmd = [
            evaluate_bin,
            "--model1", newest,
            "--model2", baseline,
            "--games", str(args.games),
            "--threads", str(threads),
            "--search-threads", str(args.search_threads),
            "--sims", str(args.sims),
            "--max-batch", str(args.max_batch),
            "--nn-server-threads", str(args.nn_server_threads),
            "--nn-device-ids", args.nn_device_ids,
            "--threshold", str(args.threshold),
            "--c-puct", str(args.c_puct),
            "--komi", str(args.komi),
            "--win-loss-weight", str(args.win_loss_weight),
            "--score-weight", str(args.score_weight),
            "--score-scale", str(args.score_scale),
        ]

        t0 = time.time()
        try:
            proc = subprocess.Popen(cmd, cwd=str(PROJECT_ROOT),
                                    stdout=subprocess.PIPE,
                                    stderr=subprocess.PIPE, text=True)
            current_child["proc"] = proc
            out, err = proc.communicate()
        except Exception as e:
            current_child["proc"] = None
            log("GATE_SPAWN_ERROR", err=str(e))
            time.sleep(2.0)
            continue
        current_child["proc"] = None
        match_dt = time.time() - t0

        out = out or ""
        err = err or ""
        w1, w2, draws = parse_eval_output(out)
        total = w1 + w2 + draws

        if total == 0:
            log("GATE_NO_RESULT", rc=proc.returncode,
                cand=os.path.basename(newest), err_head=err[:200])
            # Treat an empty-result match as a reject, but don't spin-retry —
            # move the candidate to rejected so we don't evaluate it again.
            try:
                atomic_move(newest,
                            os.path.join(args.rejected_dir,
                                         os.path.basename(newest)))
            except OSError:
                pass
            continue

        # Go scoring: draws count as losses for the candidate.
        score = w1 / float(total)
        verdict = "ACCEPT" if score > args.threshold else "REJECT"
        now = time.time()

        decisions.write({
            "wall_time": f"{now:.3f}",
            "cand_step": cand_step, "baseline_step": base_step,
            "games": total, "sims": args.sims,
            "wins": w1, "draws": draws, "losses": w2,
            "score": f"{score:.4f}", "threshold": f"{args.threshold:.4f}",
            "verdict": verdict,
            "match_duration_s": f"{match_dt:.3f}",
        })

        if verdict == "ACCEPT":
            promoted = os.path.join(args.accepted_dir, os.path.basename(newest))
            atomic_move(newest, promoted)
            atomic_symlink_swap(promoted,
                                os.path.join(args.accepted_dir, "latest"))
            log("GATE_ACCEPT", cand=os.path.basename(promoted),
                score=f"{score:.4f}", wins=w1, losses=w2, draws=draws,
                games=total, duration=f"{match_dt:.1f}")
        else:
            rejected = os.path.join(args.rejected_dir,
                                    os.path.basename(newest))
            atomic_move(newest, rejected)
            log("GATE_REJECT", cand=os.path.basename(rejected),
                score=f"{score:.4f}", wins=w1, losses=w2, draws=draws,
                games=total, duration=f"{match_dt:.1f}")

    log("SHUTDOWN")


if __name__ == "__main__":
    main()
