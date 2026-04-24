#!/usr/bin/env python3
"""
MiniGo AlphaZero — Continuous Selfplay Driver

Thin Python supervisor around build/selfplay.  Each iteration:
  1. Resolve models/accepted/latest → concrete version path.
  2. Read training/status.json → mcts_score_weight for this batch.
  3. Spawn build/selfplay into a fresh staging dir (N games).
  4. Compress game_*.bin → g_<id>.bin.zst with monotonic IDs, atomic rename.
  5. Prune pool to WINDOW_GAMES (oldest filenames first).

Crash-safety of monotonic IDs: next_id = max(existing IDs on disk) + 1 at
the moment of publishing each batch.  No separate counter file.  Orphaned
.tmp files and leftover staging dirs are cleaned up on startup.

This is the SOLE writer of training/selfplay/.
"""

import argparse
import glob
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

import zstandard as zstd

PROJECT_ROOT = Path(__file__).resolve().parent.parent

_ID_RE = re.compile(r"g_(\d+)\.bin\.zst$")


# ═══════════════════════════════════════════════════════════
#  ID helpers (crash-safe monotonic IDs from disk state)
# ═══════════════════════════════════════════════════════════

def _parse_id(path):
    m = _ID_RE.search(os.path.basename(path))
    return int(m.group(1)) if m else -1


def next_id(pool_dir):
    existing = glob.glob(os.path.join(pool_dir, "g_*.bin.zst"))
    if not existing:
        return 1
    return max(_parse_id(p) for p in existing) + 1


# ═══════════════════════════════════════════════════════════
#  Status read (non-blocking; defaults if missing)
# ═══════════════════════════════════════════════════════════

def read_status(path):
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return {"score_ramp": 0.0}


# ═══════════════════════════════════════════════════════════
#  Startup cleanup
# ═══════════════════════════════════════════════════════════

def cleanup_orphans(pool_dir, log):
    """Delete half-written .tmp files and in-flight staging dirs.
    Does NOT touch `.failed` staging dirs — those are kept for
    postmortem inspection after a PUBLISH_ERROR."""
    tmps = glob.glob(os.path.join(pool_dir, "g_*.bin.zst.tmp"))
    for t in tmps:
        try:
            os.remove(t)
            log("CLEANUP_TMP", path=os.path.basename(t))
        except OSError:
            pass
    staging_root = os.path.join(pool_dir, "staging")
    if os.path.isdir(staging_root):
        for d in glob.glob(os.path.join(staging_root, "batch_*")):
            # Preserve `.failed` sibling dirs — they contain games whose
            # publish step failed and the operator may want to inspect
            # or retry them.  Name pattern is `batch_<id>.failed-<ts>`,
            # so check for the `.failed` substring rather than an exact
            # suffix match.
            if ".failed" in os.path.basename(d):
                log("CLEANUP_STAGING_SKIP_FAILED", path=os.path.basename(d))
                continue
            try:
                shutil.rmtree(d)
                log("CLEANUP_STAGING", path=os.path.basename(d))
            except OSError:
                pass


# ═══════════════════════════════════════════════════════════
#  Compression + publish
# ═══════════════════════════════════════════════════════════

_ZSTD_LEVEL = 3  # fast; selfplay rate is bounded, not IO-bound


def _compress_and_publish(bin_path, final_path, zstd_level=_ZSTD_LEVEL):
    """Compress bin_path → final_path + '.tmp', then atomic rename to final_path.

    Uses the in-memory one-shot `compress(bytes)` path (not copy_stream)
    so the zstd frame header includes the content size.  Decompressors
    using `decompress(data)` need that field — without it, the reader
    throws `could not determine content size in frame header` and every
    game silently fails to ingest.  (Game files are small — ~20 KB
    uncompressed — so loading fully into memory is fine.)
    """
    tmp = final_path + ".tmp"
    cctx = zstd.ZstdCompressor(level=zstd_level, write_content_size=True)
    with open(bin_path, "rb") as fin:
        payload = fin.read()
    with open(tmp, "wb") as fout:
        fout.write(cctx.compress(payload))
    os.replace(tmp, final_path)


def _count_rows(zst_path):
    """Peek into a .bin.zst file and read the V2 header row count."""
    import struct
    try:
        dctx = zstd.ZstdDecompressor()
        with open(zst_path, "rb") as f:
            reader = dctx.stream_reader(f)
            hdr = reader.read(12)
        if len(hdr) < 12:
            return 0
        magic, _, count = struct.unpack_from("<HHi", hdr, 0)
        return count if magic == 0x4D47 else 0
    except (OSError, struct.error, zstd.ZstdError):
        return 0


# ═══════════════════════════════════════════════════════════
#  Accepted model resolution
# ═══════════════════════════════════════════════════════════

def resolve_accepted_latest(accepted_dir):
    """Return the concrete path pointed to by accepted/latest.
    Dereferenced once so the C++ binary sees a stable filename — this
    lets the TRT cache key (basename of model_path) stay consistent
    across iterations and hit the gatekeeper-warmed cache."""
    latest = os.path.join(accepted_dir, "latest")
    if os.path.islink(latest):
        target = os.readlink(latest)
        if not os.path.isabs(target):
            target = os.path.normpath(os.path.join(accepted_dir, target))
        return target
    if os.path.isfile(latest):
        return latest
    # No latest symlink yet — fall back to most recent v*.onnx in accepted/
    cands = sorted(glob.glob(os.path.join(accepted_dir, "v*.onnx")))
    if cands:
        return cands[-1]
    return None


# ═══════════════════════════════════════════════════════════
#  CSV logger
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
    log._fh = fh
    return log


# ═══════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(description="Continuous selfplay driver")
    ap.add_argument("--pool-dir", default="training/selfplay")
    ap.add_argument("--accepted-dir", default="models/accepted")
    ap.add_argument("--status-file", default="training/status.json")
    ap.add_argument("--log-dir", default="logs/current")
    ap.add_argument("--build-dir", default="build")
    ap.add_argument("--games-per-batch", type=int, default=300)
    ap.add_argument("--window-games", type=int, default=80000)
    # Selfplay binary knobs (mirror run_loop.py defaults where sensible)
    ap.add_argument("--sims", type=int, default=500)
    ap.add_argument("--threads", type=int, default=0,
                    help="CPU worker threads; 0 = os.cpu_count()")
    ap.add_argument("--search-threads", type=int, default=16)
    ap.add_argument("--max-batch", type=int, default=256)
    ap.add_argument("--nn-server-threads", type=int, default=2)
    ap.add_argument("--nn-device-ids", default="0,0")
    # MCTS / game params (fixed across the run)
    ap.add_argument("--c-puct", type=float, default=1.25)
    ap.add_argument("--dirichlet-alpha", type=float, default=0.15)
    ap.add_argument("--dirichlet-epsilon", type=float, default=0.22)
    ap.add_argument("--temp-threshold", type=int, default=12)
    ap.add_argument("--komi", type=float, default=7.5)
    ap.add_argument("--win-loss-weight", type=float, default=1.0)
    ap.add_argument("--score-weight-max", type=float, default=0.06,
                    help="Full value of MCTS score weight at end of ramp")
    ap.add_argument("--score-scale", type=float, default=18.0)
    # Misc
    ap.add_argument("--model-poll-interval", type=float, default=5.0,
                    help="Seconds to wait when no accepted model exists yet")
    ap.add_argument("--zstd-level", type=int, default=3)
    args = ap.parse_args()

    os.makedirs(args.pool_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.status_file) or ".", exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    # Point the downstream build/selfplay binary at the shared TRT cache
    # dir so its engine plans are reused by gatekeeper + selfplay across
    # launches regardless of cwd.
    os.environ.setdefault(
        "MINIGO_TRT_CACHE",
        str((PROJECT_ROOT / "models" / "trt_cache").resolve()))

    log = make_log(os.path.join(args.log_dir, "selfplay.log"))
    batches_csv = CsvLogger(os.path.join(args.log_dir, "selfplay_batches.csv"), [
        "batch_id", "wall_time_start", "wall_time_end", "model_in_use",
        "games_played", "positions_written", "duration_s", "selfplay_duration_s",
        "score_weight", "pool_size", "publish_failures",
    ])

    cleanup_orphans(args.pool_dir, log)

    threads = args.threads if args.threads > 0 else (os.cpu_count() or 4)

    # Graceful shutdown: first signal sets a flag and lets the current
    # batch run to completion (the C++ child is NOT signalled — the
    # supervisor's graceful-timeout + group SIGKILL escalation is the
    # hard-kill path).  Second signal force-kills the child so we exit
    # promptly if something is hung.
    stop = {"flag": False}
    stop_event = threading.Event()
    current_child = {"proc": None}
    def _handle(sig, _f):
        if stop["flag"]:
            # Second signal — escalate: kill the in-flight C++ child
            # and let subprocess.run return so we exit the loop.
            p = current_child["proc"]
            if p is not None and p.poll() is None:
                try:
                    p.kill()
                except OSError:
                    pass
            return
        stop["flag"] = True
        stop_event.set()
        log("SIGNAL", sig=sig, note="finishing current batch, then exiting")
    signal.signal(signal.SIGINT, _handle)
    signal.signal(signal.SIGTERM, _handle)

    selfplay_bin = os.path.join(args.build_dir, "selfplay")
    if not os.path.isfile(selfplay_bin):
        log("ERROR", msg=f"selfplay binary not found at {selfplay_bin}")
        sys.exit(2)

    last_model = None
    batch_id = 0

    while not stop["flag"]:
        model_path = resolve_accepted_latest(args.accepted_dir)
        if model_path is None or not os.path.isfile(model_path):
            log("NO_MODEL", accepted_dir=args.accepted_dir)
            stop_event.wait(args.model_poll_interval)
            continue
        if model_path != last_model:
            log("MODEL_SWAP", old=last_model or "-", new=model_path)
            last_model = model_path

        status = read_status(args.status_file)
        score_ramp = float(status.get("score_ramp", 0.0))
        score_w = args.score_weight_max * max(0.0, min(1.0, score_ramp))

        batch_id += 1
        staging = os.path.join(args.pool_dir, "staging", f"batch_{batch_id:09d}")
        os.makedirs(staging, exist_ok=True)

        cmd = [
            selfplay_bin,
            "--model", model_path,
            "--games", str(args.games_per_batch),
            "--threads", str(threads),
            "--search-threads", str(args.search_threads),
            "--max-batch", str(args.max_batch),
            "--nn-server-threads", str(args.nn_server_threads),
            "--nn-device-ids", args.nn_device_ids,
            "--sims", str(args.sims),
            "--c-puct", str(args.c_puct),
            "--dirichlet-alpha", str(args.dirichlet_alpha),
            "--dirichlet-epsilon", str(args.dirichlet_epsilon),
            "--temp-threshold", str(args.temp_threshold),
            "--komi", str(args.komi),
            "--win-loss-weight", str(args.win_loss_weight),
            "--score-weight", str(score_w),
            "--score-scale", str(args.score_scale),
            "--output", staging,
        ]

        wall_start = time.time()
        log("BATCH_START", batch_id=batch_id, model=os.path.basename(model_path),
            games=args.games_per_batch, sims=args.sims, score_w=f"{score_w:.4f}")

        t0 = time.time()
        try:
            proc = subprocess.Popen(cmd, cwd=str(PROJECT_ROOT))
            current_child["proc"] = proc
            rc = proc.wait()
        except Exception as e:
            current_child["proc"] = None
            log("SELFPLAY_SPAWN_ERROR", err=str(e))
            shutil.rmtree(staging, ignore_errors=True)
            time.sleep(2.0)
            continue
        current_child["proc"] = None
        selfplay_dt = time.time() - t0

        if rc != 0:
            log("SELFPLAY_EXIT_NONZERO", rc=rc, batch=batch_id)
            # Keep whatever partial output landed — publish it — but don't
            # attempt to recount as a full batch.
        bin_files = sorted(glob.glob(os.path.join(staging, "game_*.bin")))
        if not bin_files:
            log("BATCH_EMPTY", batch=batch_id)
            shutil.rmtree(staging, ignore_errors=True)
            # Throttle if selfplay is crashing immediately
            time.sleep(2.0)
            continue

        start_id = next_id(args.pool_dir)
        positions = 0
        publish_failures = 0
        for i, bp in enumerate(bin_files):
            gid = start_id + i
            final = os.path.join(args.pool_dir, f"g_{gid:017d}.bin.zst")
            try:
                _compress_and_publish(bp, final, zstd_level=args.zstd_level)
                positions += _count_rows(final)
            except Exception as e:
                log("PUBLISH_ERROR", path=os.path.basename(bp), err=str(e))
                publish_failures += 1
                # Half-written compressed tmp may remain — remove it so
                # the scanner doesn't trip over it next tick.  Leave the
                # source .bin in staging for retry/postmortem.
                try:
                    os.remove(final + ".tmp")
                except OSError:
                    pass
                continue
            # Remove the source only after a successful publish.  On
            # any failure the .bin stays in staging; the staging dir
            # itself is renamed to `.failed` below so that cleanup_orphans
            # on next startup doesn't wipe it.
            try:
                os.remove(bp)
            except OSError:
                pass

        # Pruning: keep only the latest WINDOW_GAMES files by filename
        pool = sorted(glob.glob(os.path.join(args.pool_dir, "g_*.bin.zst")))
        removed = 0
        if len(pool) > args.window_games:
            for old in pool[:-args.window_games]:
                try:
                    os.remove(old)
                    removed += 1
                except OSError:
                    pass

        if publish_failures > 0:
            # Preserve the staging dir for operator inspection.  Name
            # collisions with earlier `.failed` dirs (e.g., same batch_id
            # after restart) are prevented by the monotonic batch_id and
            # a timestamp suffix.
            failed_dir = f"{staging}.failed-{int(time.time())}"
            try:
                os.rename(staging, failed_dir)
                log("PUBLISH_STAGING_RETAINED",
                    dir=os.path.basename(failed_dir),
                    failures=publish_failures,
                    total_files=len(bin_files))
            except OSError as e:
                log("PUBLISH_STAGING_RENAME_FAIL", err=str(e))
        else:
            shutil.rmtree(staging, ignore_errors=True)

        wall_end = time.time()
        pool_size = min(len(pool), args.window_games)

        log("BATCH_DONE", batch_id=batch_id,
            games=len(bin_files), positions=positions,
            duration=f"{wall_end - wall_start:.1f}",
            selfplay=f"{selfplay_dt:.1f}",
            pool=pool_size, pruned=removed,
            publish_failures=publish_failures)

        batches_csv.write({
            "batch_id": batch_id,
            "wall_time_start": f"{wall_start:.3f}",
            "wall_time_end": f"{wall_end:.3f}",
            "model_in_use": os.path.basename(model_path),
            "games_played": len(bin_files),
            "positions_written": positions,
            "duration_s": f"{wall_end - wall_start:.3f}",
            "selfplay_duration_s": f"{selfplay_dt:.3f}",
            "score_weight": f"{score_w:.6f}",
            "pool_size": pool_size,
            "publish_failures": publish_failures,
        })

    log("SHUTDOWN", batches=batch_id)


if __name__ == "__main__":
    main()
