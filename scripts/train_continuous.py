#!/usr/bin/env python3
"""
MiniGo AlphaZero — Continuous Training

KataGo-style continuous training loop.  No epochs, no phased iterations:
one long step loop gated by a replay-ratio train bucket that fills as new
selfplay games arrive and drains as training consumes samples.

Architecture:
  - Rank 0 owns: the Bucket, the Scanner thread, the filename watermark,
    checkpoint persistence, status.json publishing, ONNX export.
  - All ranks own: a local WindowRingBuffer with its own ingest thread,
    decompressing newly-arrived games and pushing rows into a ring.
  - Every step is a collective decision: rank 0 computes go/no-go on the
    bucket + cold-start gate, broadcasts the bool, all ranks act together.

See cont_train.todo for the complete design rationale.
"""

import argparse
import glob
import io
import json
import signal
import os
import random
import re
import struct
import sys
import tempfile
import threading
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import zstandard as zstd

sys.path.insert(0, os.path.dirname(__file__))
from model import create_model


# ═══════════════════════════════════════════════════════════
#  V2 binary format (matches train.py / main_selfplay.cpp)
# ═══════════════════════════════════════════════════════════

_V2_MAGIC = 0x4D47  # 'MG'


def _decompress_bytes(filepath):
    """Decompress a selfplay record file to raw V2-format bytes.

    For .zst we use a streaming reader (not dctx.decompress(bytes))
    because the one-shot API requires the frame header to carry a
    content-size field, which isn't always present: any zstd stream
    written without a known total size up front (e.g. historical
    copy_stream-based writes) omits it.  stream_reader works in both
    cases, so this is robust to mixed-provenance pool files.
    """
    if filepath.endswith(".zst"):
        dctx = zstd.ZstdDecompressor()
        with open(filepath, "rb") as f:
            with dctx.stream_reader(f) as reader:
                return reader.read()
    if filepath.endswith(".gz"):
        import gzip
        with open(filepath, "rb") as f:
            return gzip.decompress(f.read())
    with open(filepath, "rb") as f:
        return f.read()


def _decompress_blob(blob, kind):
    """Decompress an in-memory file blob.  `kind` is 'zst', 'gz', or '' (raw).

    Used by WindowRingBuffer to lazily decompress game bytes at sample
    time — the same stream-reader logic as _decompress_bytes, but
    operating on bytes already loaded into RAM rather than hitting disk.
    """
    if kind == "zst":
        dctx = zstd.ZstdDecompressor()
        with dctx.stream_reader(io.BytesIO(blob)) as reader:
            return reader.read()
    if kind == "gz":
        import gzip
        return gzip.decompress(blob)
    return blob


def _read_blob_with_count(filepath):
    """Read a g_*.bin.zst file as raw compressed bytes and peek the V2
    row count from the decompressed header.

    Returns (blob, kind, n_rows) on success, or (None, '', 0) on any error
    or non-V2 content.  The blob is the on-disk bytes verbatim — keep it
    compressed in memory, decompress lazily later.

    Path A storage path: caller stores `blob` in the ring slot; sampler
    decompresses via _decompress_blob(blob, kind) only when rows from
    that slot are needed by a batch.
    """
    try:
        with open(filepath, "rb") as f:
            blob = f.read()
    except OSError:
        return None, "", 0

    if filepath.endswith(".zst"):
        kind = "zst"
        try:
            dctx = zstd.ZstdDecompressor()
            with dctx.stream_reader(io.BytesIO(blob)) as reader:
                hdr = reader.read(12)
        except zstd.ZstdError:
            return None, "", 0
    elif filepath.endswith(".gz"):
        kind = "gz"
        try:
            import gzip
            hdr = gzip.decompress(blob)[:12]
        except OSError:
            return None, "", 0
    else:
        kind = ""
        hdr = blob[:12]

    if len(hdr) < 12:
        return None, "", 0
    try:
        magic, _, count = struct.unpack_from("<HHi", hdr, 0)
    except struct.error:
        return None, "", 0
    if magic != _V2_MAGIC:
        return None, "", 0
    return blob, kind, int(count)


def _peek_row_count(filepath):
    """Read only the V2 header to learn the row count.  Cheap."""
    try:
        if filepath.endswith(".zst"):
            dctx = zstd.ZstdDecompressor()
            with open(filepath, "rb") as f:
                reader = dctx.stream_reader(f)
                hdr = reader.read(12)
        else:
            with open(filepath, "rb") as f:
                hdr = f.read(12)
        if len(hdr) < 12:
            return 0
        magic, _, count = struct.unpack_from("<HHi", hdr, 0)
        return count if magic == _V2_MAGIC else 0
    except (OSError, struct.error, zstd.ZstdError):
        return 0


def _parse_records(data, board_size, input_channels=17):
    """Parse a full V2 file payload into per-field numpy arrays."""
    if len(data) < 12:
        return None
    magic, _, n, _file_board = struct.unpack_from("<HHii", data, 0)
    if magic != _V2_MAGIC:
        raise ValueError(f"Not a V2 data file (magic=0x{magic:04X})")
    if n == 0:
        return None

    state_floats = input_channels * board_size * board_size
    policy_floats = board_size * board_size + 1
    own_floats = board_size * board_size
    record_bytes = (4 + state_floats * 4 + 4 + policy_floats * 4
                    + 4 + 4 + own_floats * 4 + 4)

    header_bytes = 12
    payload = np.frombuffer(data, dtype=np.uint8,
                            offset=header_bytes, count=n * record_bytes)
    records = payload.reshape(n, record_bytes)

    s_off = 4
    s_end = s_off + state_floats * 4
    p_off = s_end + 4
    p_end = p_off + policy_floats * 4
    v_off = p_end
    v_end = v_off + 4
    sc_off = v_end
    sc_end = sc_off + 4
    own_off = sc_end
    own_end = own_off + own_floats * 4
    opp_off = own_end

    states = np.frombuffer(records[:, s_off:s_end].tobytes(), dtype=np.float32
                           ).reshape(n, input_channels, board_size, board_size).copy()
    policies = np.frombuffer(records[:, p_off:p_end].tobytes(), dtype=np.float32
                             ).reshape(n, policy_floats).copy()
    values = np.frombuffer(records[:, v_off:v_end].tobytes(), dtype=np.float32).copy()
    scores = np.frombuffer(records[:, sc_off:sc_end].tobytes(), dtype=np.float32).copy()
    owns = np.frombuffer(records[:, own_off:own_end].tobytes(), dtype=np.float32
                         ).reshape(n, own_floats).copy()
    opps = np.frombuffer(records[:, opp_off:opp_off+4].tobytes(), dtype=np.int32
                         ).astype(np.int64).copy()

    return states, policies, values, scores, owns, opps


# ═══════════════════════════════════════════════════════════
#  Filename ID helpers
# ═══════════════════════════════════════════════════════════

_ID_RE = re.compile(r"g_(\d+)\.bin\.zst$")


def _parse_id(path):
    """Parse the monotonic ID from a g_<id>.bin.zst filename.  -1 if none."""
    m = _ID_RE.search(os.path.basename(path))
    return int(m.group(1)) if m else -1


def _scandir_ids(pool_dir):
    """Return list[(gid, path)] for every g_*.bin.zst entry in pool_dir.

    Unsorted — callers that only need a filtered subset sort that subset
    afterward, which is much cheaper than sorting the full pool at
    80k-file retention.  os.scandir avoids the sorted(glob(...))
    behavior in _list_pool.
    """
    out = []
    try:
        with os.scandir(pool_dir) as it:
            for entry in it:
                m = _ID_RE.search(entry.name)
                if m:
                    out.append((int(m.group(1)), entry.path))
    except FileNotFoundError:
        return []
    return out


def _warmup_watermark(pool_dir, ring_games, overshoot=1.5):
    """On resume, start the per-rank ring's ingest from the newest-K files
    instead of id=0.  Without this, ingest rehydrates the ring from the
    oldest retained pool upward, and the cold-start gate passes as soon
    as min_ring_rows worth of STALE old data has landed — briefly
    training on data several hours out of date.

    Returns a watermark value such that the ingest loop's `gid >
    watermark` filter captures at most overshoot * ring_games files
    (the newest ones).  On a fresh run or a barely-populated pool,
    returns 0 so nothing is skipped.
    """
    ids = [gid for gid, _ in _scandir_ids(pool_dir)]
    target = int(ring_games * overshoot)
    if len(ids) <= target:
        return 0
    ids.sort()
    return ids[-target] - 1


# ═══════════════════════════════════════════════════════════
#  Bucket (rank 0 only)
# ═══════════════════════════════════════════════════════════

class Bucket:
    """KataGo-style train bucket.  Credits in sample units (rows)."""

    def __init__(self, max_samples, initial=0):
        self.max = int(max_samples)
        self._level = int(initial)
        self._lock = threading.Lock()

    def available(self):
        with self._lock:
            return self._level

    def fill_ratio(self):
        with self._lock:
            return self._level / max(1, self.max)

    def credit(self, samples):
        with self._lock:
            self._level = min(self.max, self._level + int(samples))

    def drain(self, samples):
        with self._lock:
            self._level -= int(samples)
            if self._level < 0:
                self._level = 0

    def set(self, samples):
        with self._lock:
            self._level = min(self.max, int(samples))


# ═══════════════════════════════════════════════════════════
#  Scanner thread (rank 0 only)
# ═══════════════════════════════════════════════════════════

class WindowScanner(threading.Thread):
    """Background thread tailing the selfplay pool and crediting the bucket.

    Iterates every POLL_INTERVAL seconds.  Pool files are immutable once
    renamed into the pool (the selfplay driver writes via `.tmp` +
    rename), so a file's row count never changes — we cache it in
    `_row_counts` and only read the V2 header for files we haven't seen
    before.  Pruned files are detected by set-diff against the cache and
    evicted.

    The first scan after a cold start reads headers for every current
    file (O(pool_size)); subsequent scans only read headers for newly
    added files (O(new_files_per_tick), typically tiny).  `window_games`
    updates immediately from the scandir listing so the cold-start gate
    doesn't block on the one-time header sweep.
    """

    POLL_INTERVAL = 5.0

    def __init__(self, bucket, pool_dir, replay_target, n_augmentations,
                 initial_watermark=0):
        super().__init__(name="WindowScanner", daemon=True)
        self.bucket = bucket
        self.pool_dir = pool_dir
        self.replay_target = replay_target
        self.n_aug = n_augmentations
        self._watermark = int(initial_watermark)
        self._row_counts = {}      # gid -> row count (persisted across ticks)
        self._window_games = 0
        self._window_rows = 0
        self._stop = threading.Event()
        # Guards the (bucket.credit, watermark) pair so checkpoint_snapshot
        # cannot observe a torn write where the watermark moved forward
        # but the credit for the newly watermarked files hasn't yet
        # landed in the bucket (or vice versa).  Every write to
        # _watermark and every bucket.credit() in _tick() is done under
        # this lock; the public snapshot method takes the same lock.
        self._credit_lock = threading.Lock()

    def watermark(self):
        return self._watermark

    def window_games(self):
        return self._window_games

    def window_rows(self):
        return self._window_rows

    def checkpoint_snapshot(self):
        """Return (bucket_level, watermark_id) captured atomically with
        respect to the scanner's credit+advance-watermark step.  Without
        this lock, a checkpoint taken between scanner._tick's
        `bucket.credit(credit)` and `_watermark = new_max_id` would
        persist a newer watermark with an older bucket level — on
        resume, files at or below the saved watermark are never re-
        credited, so that credit is lost permanently."""
        with self._credit_lock:
            return self.bucket.available(), self._watermark

    def stop(self):
        self._stop.set()

    def _tick(self):
        """One scan pass.  Exposed for tests; `run()` calls this in a loop."""
        current = _scandir_ids(self.pool_dir)
        current_ids = {gid for gid, _ in current}
        path_by_id = dict(current)

        # Evict row counts for pruned files (gone from disk).
        for g in [g for g in self._row_counts if g not in current_ids]:
            del self._row_counts[g]

        # Read headers only for files we haven't cached yet.
        unknown = sorted(g for g in current_ids
                         if g not in self._row_counts)

        # Publish games count right away so the cold-start gate can
        # proceed even while the one-shot header sweep is still running
        # on large pools.
        self._window_games = len(current_ids)

        new_rows = 0
        new_max_id = self._watermark
        for g in unknown:
            rows = _peek_row_count(path_by_id[g])
            self._row_counts[g] = rows
            if g > self._watermark:
                new_rows += rows
                if g > new_max_id:
                    new_max_id = g

        if new_rows > 0:
            # replay_target is per unique position; each disk row is
            # one of N_AUGMENTATIONS views of a position, so credit
            # replay_target rows of budget per N_AUGMENTATIONS rows seen.
            # Hold the credit lock so checkpoint_snapshot sees these two
            # writes atomically.
            credit = (self.replay_target * new_rows) // self.n_aug
            with self._credit_lock:
                self.bucket.credit(credit)
                self._watermark = new_max_id

        self._window_rows = sum(self._row_counts.values())

    def run(self):
        while not self._stop.is_set():
            try:
                self._tick()
            except Exception as e:
                print(f"[scanner] error: {e}", file=sys.stderr, flush=True)
            self._stop.wait(self.POLL_INTERVAL)


# ═══════════════════════════════════════════════════════════
#  Ring buffer + ingest (per rank)
# ═══════════════════════════════════════════════════════════

class WindowRingBuffer:
    """Per-rank in-RAM ring of compressed game blobs.

    PATH A (compressed-in-memory) design: each ring slot holds the raw
    ``g_*.bin.zst`` bytes of one selfplay game (~25 KB) plus its row count.
    Decompression+parse happens lazily per batch, only for the games a
    sample touches.  Trades ~50 ms of CPU per batch for a ~20× reduction
    in host RAM vs. storing decompressed numpy rows — the decisive win
    because it lets the trainer's effective sampling window approach
    KataGo-scale on commodity hosts.

    Sampling pattern (important)
    ----------------------------
    We do **game-granular sampling**: pick ``K = batch_size //
    samples_per_game`` games uniformly with replacement, then sample
    ``samples_per_game`` rows from within each chosen game.  This is
    the same trick KataGo's npz shuffler uses to keep per-batch CPU
    work bounded.  Strict row-uniform sampling over a large compressed
    ring is intractable: a 256-row batch from a 2000-game ring would
    hit ~240 distinct games per batch (coupon-collector), forcing 240
    decompressions per step — ~1.2 s on a single thread.  Game-granular
    sampling at ``samples_per_game=8`` instead hits ~32 games per batch,
    parallelizes 4× across a small thread pool, lands in <100 ms.

    Trade-off: rows from the same game in one batch share value,
    score, and outcome targets.  At ``samples_per_game=8``, that's ~3%
    of the batch correlated with each game.  Across batches every
    sample is independent.  KataGo accepts the same correlation;
    raising K (lowering samples_per_game) reduces it at the cost of
    more decompressions per batch.

    Sizing
    ------
    The constructor takes ``ring_rows`` (= ring_games × ~moves × n_aug)
    for API compatibility with the legacy row-ring caller.  Internally
    we convert this to a slot count (one slot per game) by dividing by
    a per-game row estimate.  ``ring_rows_current()`` returns the
    *actual* row total summed across filled slots, so the cold-start
    gate continues to work.

    Thread safety
    -------------
    All reads/writes of slot bytes, row counts, head, and filled-count
    happen under self._lock.  Decompression and tensor packing happen
    OUTSIDE the lock — Python refcounts the snapshotted bytes objects,
    so a concurrent ingest overwriting a slot can't invalidate an
    in-flight sample.
    """

    INGEST_POLL_MIN_S = 0.5
    INGEST_POLL_MAX_S = 3.0
    MAX_QUEUED_IDS = 256

    # 9x9 games average ~80–120 moves × 8 augmentations ≈ ~800 rows.  Used
    # only to translate the row-budget API into a slot count.
    _APPROX_ROWS_PER_GAME = 800

    def __init__(self, pool_dir, ring_rows, board_size, rng_seed,
                 initial_watermark=0, samples_per_game=8,
                 decompress_workers=4):
        self.pool_dir = pool_dir
        self.ring_rows = int(ring_rows)
        self.board_size = board_size
        self._rng = np.random.default_rng(rng_seed)
        self.samples_per_game = max(1, int(samples_per_game))

        # Number of game slots in the ring.  At least 1.
        self._n_slots = max(1, self.ring_rows // self._APPROX_ROWS_PER_GAME)

        # Shared state (lock-protected)
        self._lock = threading.Lock()
        # Each slot holds raw compressed bytes (or None if unused) plus
        # the format kind ("zst" / "gz" / "") needed to decompress later.
        self._game_bytes = [None] * self._n_slots
        self._game_kind = [""] * self._n_slots
        self._game_n_rows = np.zeros(self._n_slots, dtype=np.int32)
        self._head_slot = 0       # next slot to overwrite
        self._n_filled = 0        # number of slots holding valid bytes (<= n_slots)

        self._watermark = int(initial_watermark)
        self._stop = threading.Event()
        self._ingest_thread = None
        self._ingest_poll = self.INGEST_POLL_MIN_S

        # Persistent thread pool for parallel zstd decompression.  zstd's
        # C library releases the GIL during decompress; numpy parse holds
        # it.  Empirically ~2× speedup at 4 workers, diminishing returns
        # past 8.  Start small and let the user tune via the
        # decompress_workers ctor arg.
        from concurrent.futures import ThreadPoolExecutor
        self._decompress_workers = max(1, int(decompress_workers))
        self._pool = ThreadPoolExecutor(
            max_workers=self._decompress_workers, thread_name_prefix="ring-dec")

    # ---- public API ------------------------------------------------

    def ring_rows_current(self):
        """Total rows currently stored across all filled slots."""
        with self._lock:
            if self._n_filled == 0:
                return 0
            return int(self._game_n_rows[:self._n_filled].sum())

    def start_ingest(self):
        if self._ingest_thread is not None:
            return
        self._ingest_thread = threading.Thread(
            target=self._ingest_loop, name="RingIngest", daemon=True)
        self._ingest_thread.start()

    def stop_ingest(self):
        self._stop.set()

    def sample_batch(self, batch_size, device, timeout_s=30.0):
        """Block until ring has ≥ batch_size rows; sample game-granularly.

        Pick ``K = ceil(batch_size / samples_per_game)`` games uniformly
        with replacement, decompress them in parallel, take
        ``samples_per_game`` rows from each (random within-game).
        Concatenate, truncate to exactly ``batch_size``.

        Phase 1 (under lock): snapshot bytes refs for the chosen games.
        Phase 2 (lock released): parallel decompress + parse + gather.
        """
        deadline = time.time() + timeout_s
        poll = 0.5
        spg = self.samples_per_game
        K = max(1, (batch_size + spg - 1) // spg)

        while True:
            with self._lock:
                n = self._n_filled
                if n > 0:
                    total_rows = int(self._game_n_rows[:n].sum())
                else:
                    total_rows = 0

                if total_rows >= batch_size and n > 0:
                    # Pick K random slots (with replacement).  No
                    # row-weighted sampling: 9x9 games are uniform
                    # enough in row count (~640–960) that uniform-over-
                    # slots is a close approximation to uniform-over-
                    # rows for our purposes.
                    chosen = self._rng.integers(0, n, size=K)
                    # Snapshot unique slots (pinning bytes refs)
                    unique = np.unique(chosen)
                    snapshot = {}
                    for s in unique.tolist():
                        snapshot[s] = (
                            self._game_bytes[s],
                            self._game_kind[s],
                            int(self._game_n_rows[s]),
                        )
                    break

            if time.time() > deadline:
                raise TimeoutError(
                    f"sample_batch timed out waiting for ring "
                    f"(rows={total_rows}, slots_filled={n}, need={batch_size})")
            time.sleep(poll)

        # Phase 2: parallel decompress + parse for unique slots.
        def _work(item):
            slot, blob, kind = item
            if blob is None:
                return slot, None
            try:
                raw = _decompress_blob(blob, kind)
                parsed = _parse_records(raw, self.board_size)
                return slot, parsed
            except (ValueError, zstd.ZstdError):
                return slot, None

        items = [(s, b, k) for s, (b, k, _nr) in snapshot.items()]
        slot_parsed = {}
        for slot, parsed in self._pool.map(_work, items):
            slot_parsed[int(slot)] = parsed

        # Gather samples_per_game rows per chosen slot.
        states_b, policies_b, values_b, scores_b, owns_b, opps_b = \
            [], [], [], [], [], []
        rng = self._rng  # serialized through self._lock when sampling indices
        for slot in chosen.tolist():
            parsed = slot_parsed.get(int(slot))
            if parsed is None:
                # Find any working parse in this batch as a fallback so
                # we always return batch_size rows even if a couple of
                # blobs were corrupted.
                for cand in slot_parsed.values():
                    if cand is not None:
                        parsed = cand
                        break
                if parsed is None:
                    raise RuntimeError(
                        "sample_batch: every blob in this batch failed "
                        "to decompress; ring is corrupt")
            s_arr, p_arr, v_arr, sc_arr, o_arr, op_arr = parsed
            n_rows = s_arr.shape[0]
            if n_rows == 0:
                continue
            # Sample spg rows from within this game.  With-replacement
            # is fine: in 9x9 most games have hundreds of rows.
            idx = rng.integers(0, n_rows, size=spg)
            states_b.append(s_arr[idx])
            policies_b.append(p_arr[idx])
            values_b.append(v_arr[idx])
            scores_b.append(sc_arr[idx])
            owns_b.append(o_arr[idx])
            opps_b.append(op_arr[idx])

        # Concatenate and trim to exactly batch_size.
        states = np.concatenate(states_b)[:batch_size]
        policies = np.concatenate(policies_b)[:batch_size]
        values = np.concatenate(values_b)[:batch_size]
        scores = np.concatenate(scores_b)[:batch_size]
        owns = np.concatenate(owns_b)[:batch_size]
        opps = np.concatenate(opps_b)[:batch_size]

        states = np.ascontiguousarray(states)
        policies = np.ascontiguousarray(policies)
        values = np.ascontiguousarray(values)
        scores = np.ascontiguousarray(scores)
        owns = np.ascontiguousarray(owns)
        opps = np.ascontiguousarray(opps)

        return (torch.from_numpy(states).to(device, non_blocking=True),
                torch.from_numpy(policies).to(device, non_blocking=True),
                torch.from_numpy(values).to(device, non_blocking=True),
                torch.from_numpy(scores).to(device, non_blocking=True),
                torch.from_numpy(owns).to(device, non_blocking=True),
                torch.from_numpy(opps).to(device, non_blocking=True))

    def state(self):
        """Persistable state — only watermark; ring itself rehydrates from disk."""
        return {"watermark_id": self._watermark}

    # ---- internals -------------------------------------------------

    def _append_blob(self, blob, kind, n_rows):
        """Circular append at game granularity (O(1))."""
        if blob is None or n_rows <= 0:
            return
        with self._lock:
            slot = self._head_slot
            self._game_bytes[slot] = blob
            self._game_kind[slot] = kind
            self._game_n_rows[slot] = int(n_rows)
            self._head_slot = (slot + 1) % self._n_slots
            if self._n_filled < self._n_slots:
                self._n_filled += 1

    def _ingest_loop(self):
        while not self._stop.is_set():
            try:
                # Filter BEFORE sorting — at 80k retention, sorting only
                # the small "new" subset is far cheaper than sorting all
                # 80k paths each tick.
                current = _scandir_ids(self.pool_dir)
                new_files = [(g, p) for g, p in current if g > self._watermark]
                new_files.sort()
                new_files = new_files[:self.MAX_QUEUED_IDS]

                if not new_files:
                    # Exponential back-off when caught up; resets to min
                    # as soon as new files arrive.
                    self._ingest_poll = min(
                        self._ingest_poll * 1.5, self.INGEST_POLL_MAX_S)
                    self._stop.wait(self._ingest_poll)
                    continue

                self._ingest_poll = self.INGEST_POLL_MIN_S
                for gid, p in new_files:
                    if self._stop.is_set():
                        break
                    try:
                        blob, kind, n_rows = _read_blob_with_count(p)
                        if blob is not None and n_rows > 0:
                            self._append_blob(blob, kind, n_rows)
                        self._watermark = gid
                    except (OSError, ValueError, zstd.ZstdError) as e:
                        # File may have been pruned mid-read — skip it and
                        # advance watermark so we don't retry forever.
                        self._watermark = gid
                        print(f"[ingest] skip {os.path.basename(p)}: {e}",
                              file=sys.stderr, flush=True)
            except Exception as e:
                print(f"[ingest] error: {e}", file=sys.stderr, flush=True)
                self._stop.wait(self.INGEST_POLL_MIN_S)


# ═══════════════════════════════════════════════════════════
#  LR schedule
# ═══════════════════════════════════════════════════════════

class FixedStepLR:
    """Linear warmup → base_lr → halve at each milestone."""

    def __init__(self, warmup_steps, base_lr, milestones, gamma):
        self.warmup = int(warmup_steps)
        self.base = float(base_lr)
        self.milestones = sorted(int(m) for m in milestones)
        self.gamma = float(gamma)
        self._last_milestones_hit = 0

    def lr(self, step):
        if step < self.warmup and self.warmup > 0:
            return self.base * (step + 1) / self.warmup
        lr = self.base
        hit = 0
        for m in self.milestones:
            if step >= m:
                lr *= self.gamma
                hit += 1
        return lr

    def milestones_hit(self, step):
        return sum(1 for m in self.milestones if step >= m)


def linear_ramp(step, end_step, start=0.0, end=1.0):
    if end_step <= 0:
        return end
    t = min(max(step / end_step, 0.0), 1.0)
    return start + (end - start) * t


# ═══════════════════════════════════════════════════════════
#  Loss computation (shared with train.py semantics)
# ═══════════════════════════════════════════════════════════

def _value_target(values):
    return torch.where(values > 0, 0, torch.where(values < 0, 1, 2)).long()


def compute_losses(model, batch, bin_centers, belief_sigma, weights,
                   amp_ctx):
    """Run one forward pass, return (total_loss, dict of per-head losses)."""
    states, policies, values, scores, owns, opps = batch
    vt = _value_target(values)
    with amp_ctx:
        (p_pol, p_val, p_smn, p_ssd, p_own, p_bel, p_opp) = model(states)

        policy_loss = -torch.sum(
            policies * torch.log_softmax(p_pol, dim=1)) / states.size(0)
        value_loss = F.cross_entropy(p_val, vt)
        score_mean_loss = F.huber_loss(p_smn.squeeze(1), scores, delta=12.0)
        with torch.no_grad():
            stdev_target = (scores - p_smn.squeeze(1).detach()).abs()
        score_stdev_loss = F.huber_loss(p_ssd.squeeze(1), stdev_target, delta=10.0)
        ownership_loss = F.binary_cross_entropy_with_logits(p_own, owns)
        score_exp = scores.unsqueeze(1)
        belief_logits = -0.5 * ((bin_centers.unsqueeze(0) - score_exp) / belief_sigma) ** 2
        soft_target = F.softmax(belief_logits, dim=1)
        score_belief_loss = -(soft_target * F.log_softmax(p_bel, dim=1)).sum(dim=1).mean()

        opp_mask = opps >= 0
        if opp_mask.any():
            opp_policy_loss = F.cross_entropy(p_opp[opp_mask], opps[opp_mask])
        else:
            opp_policy_loss = torch.zeros((), device=states.device)

        total = (weights["policy"] * policy_loss
                 + weights["value"] * value_loss
                 + weights["score_mean"] * score_mean_loss
                 + weights["score_stdev"] * score_stdev_loss
                 + weights["ownership"] * ownership_loss
                 + weights["score_belief"] * score_belief_loss
                 + weights["opp_policy"] * opp_policy_loss)

    parts = {
        "policy": policy_loss.item(),
        "value": value_loss.item(),
        "score_mean": score_mean_loss.item(),
        "score_stdev": score_stdev_loss.item(),
        "ownership": ownership_loss.item(),
        "score_belief": score_belief_loss.item(),
        "opp_policy": opp_policy_loss.item(),
    }
    return total, parts


# ═══════════════════════════════════════════════════════════
#  Atomic writes
# ═══════════════════════════════════════════════════════════

def _atomic_write_json(path, obj):
    d = os.path.dirname(path) or "."
    os.makedirs(d, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".tmp.", dir=d)
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(obj, f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except Exception:
        try:
            os.remove(tmp)
        except OSError:
            pass
        raise


def _atomic_save_torch(path, obj):
    d = os.path.dirname(path) or "."
    os.makedirs(d, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".tmp.", dir=d, suffix=".pt")
    os.close(fd)
    torch.save(obj, tmp)
    os.replace(tmp, path)


# ═══════════════════════════════════════════════════════════
#  CSV logger (rank 0 only)
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
            if isinstance(v, float):
                parts.append(f"{v:.6g}")
            else:
                parts.append(str(v))
        with open(self.path, "a") as f:
            f.write(",".join(parts) + "\n")


# ═══════════════════════════════════════════════════════════
#  Event logger (text log, rank 0 only)
# ═══════════════════════════════════════════════════════════

class EventLogger:
    def __init__(self, path):
        self.path = path
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self.fh = open(path, "a", buffering=1)

    def log(self, tag, **kv):
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        kvs = " ".join(f"{k}={v}" for k, v in kv.items())
        line = f"[{ts}] {tag}" + (f" {kvs}" if kvs else "")
        self.fh.write(line + "\n")
        print(line, flush=True)

    def close(self):
        try:
            self.fh.close()
        except OSError:
            pass


# ═══════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(description="Continuous training for MiniGo AlphaZero")
    # Data + model
    ap.add_argument("--pool-dir", default="training/selfplay")
    ap.add_argument("--candidates-dir", default="models/candidates")
    ap.add_argument("--checkpoint", default="training/checkpoints/training.pt")
    ap.add_argument("--status-file", default="training/status.json")
    ap.add_argument("--log-dir", default="logs/current",
                    help="Directory for train.log and train_metrics.csv")
    ap.add_argument("--board", type=int, default=9)
    ap.add_argument("--arch", default="resnet", choices=["resnet", "vit"])
    ap.add_argument("--filters", type=int, default=64)
    ap.add_argument("--blocks", type=int, default=5)
    ap.add_argument("--d-model", type=int, default=192)
    ap.add_argument("--depth", type=int, default=8)
    ap.add_argument("--heads", type=int, default=6)
    ap.add_argument("--kv-groups", type=int, default=2)
    ap.add_argument("--mlp-ratio", type=int, default=4)
    # Training
    ap.add_argument("--batch-size", type=int, default=1024,
                    help="Per-rank batch size")
    ap.add_argument("--base-lr", type=float, default=3e-4)
    ap.add_argument("--warmup-steps", type=int, default=2000)
    ap.add_argument("--lr-milestones", default="100000,400000,1500000")
    ap.add_argument("--lr-gamma", type=float, default=0.5)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--fp8", action="store_true")
    # Window / ring / bucket
    ap.add_argument("--replay-target", type=float, default=4.0)
    ap.add_argument("--n-augmentations", type=int, default=8)
    ap.add_argument("--ring-games", type=int, default=2000,
                    help="Games in per-rank ring; rows = ring_games * ~800")
    ap.add_argument("--samples-per-game", type=int, default=8,
                    help="Rows per game pulled into each batch (game-"
                         "granular sampling).  K = batch_size / "
                         "samples_per_game decompressions per batch.  "
                         "Higher = fewer decompressions, more within-"
                         "batch correlation; lower = more decompressions, "
                         "more diverse batch.")
    ap.add_argument("--ring-decompress-workers", type=int, default=4,
                    help="Thread pool size for parallel zstd decompress "
                         "during sample_batch.  zstd C lib releases the "
                         "GIL during decode; ~2× speedup at 4 workers.")
    ap.add_argument("--bucket-cap-mult", type=int, default=64)
    ap.add_argument("--min-window-games", type=int, default=2000,
                    help="Cold-start gate: min files on disk before training")
    ap.add_argument("--min-ring-rows", type=int, default=10240,
                    help="Cold-start gate: min rows in each rank's ring")
    ap.add_argument("--sample-batch-timeout-s", type=float, default=30.0)
    # Export / status
    ap.add_argument("--export-every", type=int, default=5000)
    ap.add_argument("--status-publish-every", type=int, default=100)
    ap.add_argument("--log-every", type=int, default=100)
    # Ramps
    ap.add_argument("--value-ramp-steps", type=int, default=30000)
    ap.add_argument("--score-ramp-steps", type=int, default=50000)
    # Fixed head weights
    ap.add_argument("--policy-weight", type=float, default=1.0)
    ap.add_argument("--value-weight-start", type=float, default=1.0)
    ap.add_argument("--value-weight-end", type=float, default=2.0)
    ap.add_argument("--score-mean-weight-start", type=float, default=0.004)
    ap.add_argument("--score-mean-weight-end", type=float, default=0.010)
    ap.add_argument("--score-stdev-weight", type=float, default=0.006)
    ap.add_argument("--score-belief-weight", type=float, default=0.035)
    ap.add_argument("--ownership-weight", type=float, default=0.85)
    ap.add_argument("--opp-policy-weight", type=float, default=0.1)
    # Misc
    ap.add_argument("--base-seed", type=int, default=0xC047)
    ap.add_argument("--max-steps", type=int, default=0,
                    help="If >0, stop after this many total steps (test hook)")

    args = ap.parse_args()

    # Share the TRT engine cache dir with the C++ binaries our siblings
    # launch.  Supervisor-launched runs already set this; setdefault makes
    # standalone debug runs of the trainer hit the same cache.  (The
    # trainer itself never builds TRT engines, but we keep the contract
    # uniform across all workers so the env is consistent wherever C++
    # is invoked downstream.)
    os.environ.setdefault(
        "MINIGO_TRT_CACHE",
        str(Path(args.pool_dir).resolve().parent.parent / "models" / "trt_cache"))

    # ── Graceful-shutdown signalling ───────────────────────
    # Each rank installs its own SIGTERM/SIGINT handler that just flips
    # a threading.Event.  The main loop polls rank 0's flag and
    # broadcasts a shutdown decision so every rank exits the loop
    # together.  Without this, Ctrl-C / supervisor SIGTERM would kill
    # the process between exports and drop up to (export_every - 1)
    # steps of optimizer + bucket + watermark state.
    shutdown_event = threading.Event()
    def _handle_shutdown(sig, _frame):
        shutdown_event.set()
    signal.signal(signal.SIGTERM, _handle_shutdown)
    try:
        signal.signal(signal.SIGINT, _handle_shutdown)
    except ValueError:
        # Not in main thread (shouldn't happen since main() runs in main
        # thread, but be defensive).
        pass

    # ── DDP setup ──────────────────────────────────────────
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    use_ddp = local_rank >= 0 and torch.cuda.is_available()
    if use_ddp:
        dist.init_process_group("nccl")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        rank = 0
        world_size = 1
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")

    is_main = (rank == 0)

    # ── Logging (rank 0 only) ──────────────────────────────
    events = None
    metrics = None
    if is_main:
        os.makedirs(args.log_dir, exist_ok=True)
        events = EventLogger(os.path.join(args.log_dir, "train.log"))
        metrics = CsvLogger(os.path.join(args.log_dir, "train_metrics.csv"), [
            "step", "wall_time", "samples_seen", "lr", "value_weight",
            "score_weight_mcts", "score_mean_weight",
            "loss_total", "loss_policy", "loss_value", "loss_score_mean",
            "loss_score_stdev", "loss_ownership", "loss_score_belief",
            "loss_opp_policy",
            "window_games", "window_rows", "bucket_samples", "bucket_fill_ratio",
            "ring_rows", "steps_per_sec", "gpu_mem_mb",
        ])
        events.log("TRAIN_START", rank=rank, world_size=world_size,
                   device=str(device), arch=args.arch, board=args.board,
                   batch_size=args.batch_size,
                   global_batch=args.batch_size * world_size)

    # ── Model + optimizer ──────────────────────────────────
    model = create_model(
        arch=args.arch, board_size=args.board, input_channels=17,
        num_filters=args.filters, num_res_blocks=args.blocks,
        d_model=args.d_model, depth=args.depth, heads=args.heads,
        kv_groups=args.kv_groups, mlp_ratio=args.mlp_ratio,
        use_fp8=args.fp8,
    ).to(device)

    if args.arch == "vit":
        optimizer = optim.AdamW(model.parameters(), lr=args.base_lr,
                                weight_decay=args.weight_decay)
    else:
        optimizer = optim.Adam(model.parameters(), lr=args.base_lr,
                               weight_decay=args.weight_decay)

    # ── Resume (rank 0 loads; broadcasts meta to others) ───
    resume_meta = None
    if is_main and os.path.exists(args.checkpoint):
        ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            resume_meta = {
                "step": int(ckpt.get("step", 0)),
                "bucket_level": int(ckpt.get("bucket_level", 0)),
                "watermark_id": int(ckpt.get("watermark_id", 0)),
            }
            events.log("RESUME", step=resume_meta["step"],
                       bucket=resume_meta["bucket_level"],
                       watermark=resume_meta["watermark_id"])

    if use_ddp:
        obj_list = [resume_meta]
        dist.broadcast_object_list(obj_list, src=0)
        resume_meta = obj_list[0]

    step = resume_meta["step"] if resume_meta else 0

    # Load weights + optimizer — every rank reads the same checkpoint file.
    # (Cheap — training.pt is << 1 GB.)
    if os.path.exists(args.checkpoint):
        ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
        if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            model.load_state_dict(ckpt["model_state_dict"], strict=False)
            if "optimizer_state_dict" in ckpt:
                try:
                    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
                except (ValueError, KeyError) as e:
                    if is_main:
                        events.log("OPTIM_RESET", reason=str(e))
        else:
            model.load_state_dict(ckpt, strict=False)

    # ── AMP ────────────────────────────────────────────────
    use_amp = device.type == "cuda"
    amp_dtype = torch.float32
    scaler = None
    use_te_fp8 = False
    if use_amp:
        if args.fp8:
            try:
                import transformer_engine.pytorch as te  # noqa: F401
                use_te_fp8 = True
                amp_dtype = torch.bfloat16
            except ImportError:
                args.fp8 = False
        if not use_te_fp8:
            if torch.cuda.is_bf16_supported():
                amp_dtype = torch.bfloat16
            else:
                amp_dtype = torch.float16
                scaler = torch.amp.GradScaler()

    # Build AMP context each step
    def amp_context():
        if use_te_fp8:
            import transformer_engine.pytorch as te
            return te.fp8_autocast(enabled=True)
        if use_amp:
            return torch.amp.autocast("cuda", dtype=amp_dtype, enabled=True)
        import contextlib
        return contextlib.nullcontext()

    # DDP wrap
    if use_ddp:
        model = DDP(model, device_ids=[local_rank])

    # ── Bucket + scanner (rank 0) ──────────────────────────
    global_batch = args.batch_size * world_size
    bucket = None
    scanner = None
    if is_main:
        bucket = Bucket(
            max_samples=args.bucket_cap_mult * global_batch,
            initial=resume_meta["bucket_level"] if resume_meta else 0,
        )
        scanner = WindowScanner(
            bucket=bucket,
            pool_dir=args.pool_dir,
            replay_target=args.replay_target,
            n_augmentations=args.n_augmentations,
            initial_watermark=resume_meta["watermark_id"] if resume_meta else 0,
        )
        scanner.start()

    # ── Per-rank ring ──────────────────────────────────────
    # One game ≈ 100 moves × n_augmentations rows on disk (src/mcts.cpp:725);
    # 9x9 games average ~100 moves, so ring_games × 100 × n_aug is a decent
    # upper bound for 9x9.  For other board sizes, scale by board area/81.
    avg_moves_per_game = max(50, args.board * args.board * 100 // 81)
    ring_rows = int(args.ring_games * avg_moves_per_game * args.n_augmentations)

    # On resume, start the ring's ingest from the newest ~ring_games
    # files instead of id=0, so it rehydrates with the freshest window
    # instead of replaying the oldest retained pool.  Without this, the
    # cold-start gate passes as soon as min_ring_rows worth of stale old
    # data has landed, and the trainer briefly trains on data that is
    # several hours out of date relative to the current selfplay model.
    # Fresh runs (empty pool) get watermark=0 and ingest everything.
    ring_warmup_id = _warmup_watermark(args.pool_dir, args.ring_games)
    if is_main:
        events.log("RING_WARMUP", initial_watermark=ring_warmup_id,
                   ring_games=args.ring_games)

    window = WindowRingBuffer(
        pool_dir=args.pool_dir,
        ring_rows=ring_rows,
        board_size=args.board,
        rng_seed=args.base_seed + rank + step,  # different per-rank, per-resume
        initial_watermark=ring_warmup_id,
        samples_per_game=args.samples_per_game,
        decompress_workers=args.ring_decompress_workers,
    )
    window.start_ingest()

    # ── Belief bin centers ─────────────────────────────────
    hw = args.board * args.board
    num_bins = hw * 2 + 1
    belief_sigma = 3.0
    bin_centers = (torch.arange(num_bins, dtype=torch.float32, device=device)
                   - hw)

    # ── LR schedule ────────────────────────────────────────
    lr_sched = FixedStepLR(
        warmup_steps=args.warmup_steps, base_lr=args.base_lr,
        milestones=[int(m) for m in args.lr_milestones.split(",") if m.strip()],
        gamma=args.lr_gamma,
    )

    def set_lr(lr):
        for g in optimizer.param_groups:
            g["lr"] = lr

    # ── Training loop ──────────────────────────────────────
    os.makedirs(args.candidates_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.checkpoint) or ".", exist_ok=True)

    # Clean up half-written checkpoint tmpfiles from an interrupted save.
    # (`.tmp.*` files come from tempfile.mkstemp + torch.save interrupted
    # before the atomic os.replace could run.)
    ckpt_dir = os.path.dirname(args.checkpoint) or "."
    for f in glob.glob(os.path.join(ckpt_dir, ".tmp.*")):
        try:
            os.remove(f)
            if is_main:
                events.log("CLEANUP_CKPT_TMP", path=os.path.basename(f))
        except OSError:
            pass

    last_steps_sec_t = time.time()
    last_steps_sec_count = 0
    loss_ema = None
    last_milestones_hit = lr_sched.milestones_hit(step)

    # Rolling means of per-head losses over LOG_EVERY-step windows
    roll = {k: 0.0 for k in
            ["total", "policy", "value", "score_mean", "score_stdev",
             "ownership", "score_belief", "opp_policy"]}
    roll_n = 0

    # Time-based rate limit for wait-state logs.  Step doesn't advance
    # during COLD_START / BUDGET_SLEEP, so a step%N check either spams
    # every 5 s (step==0) or never fires at all (step misaligned with N).
    WAIT_LOG_INTERVAL_S = 30.0
    last_cold_log_t = 0.0
    last_budget_log_t = 0.0

    def shutdown_requested():
        """True when rank 0 has received SIGTERM/SIGINT (broadcast-safe
        across ranks so they exit the loop together)."""
        if is_main:
            sd = 1 if shutdown_event.is_set() else 0
        else:
            sd = 0
        sd_t = torch.tensor([sd], dtype=torch.long, device=device)
        if use_ddp:
            dist.broadcast(sd_t, src=0)
        return bool(sd_t.item())

    shutdown_reason = None
    while True:
        if shutdown_requested():
            shutdown_reason = "signal"
            break

        # ── Cold-start gate (both conditions must hold) ───
        if is_main:
            disk_games = scanner.window_games()
            disk_ok = 1 if disk_games >= args.min_window_games else 0
        else:
            disk_ok = 1
        disk_t = torch.tensor([disk_ok], dtype=torch.long, device=device)
        if use_ddp:
            dist.broadcast(disk_t, src=0)

        local_rows = window.ring_rows_current()
        local_ready = 1 if local_rows >= args.min_ring_rows else 0
        local_t = torch.tensor([local_ready], dtype=torch.long, device=device)
        if use_ddp:
            dist.all_reduce(local_t, op=dist.ReduceOp.MIN)

        if disk_t.item() == 0 or local_t.item() == 0:
            now = time.time()
            if is_main and (now - last_cold_log_t) >= WAIT_LOG_INTERVAL_S:
                events.log("COLD_START_WAIT",
                           window_games=scanner.window_games(),
                           ring_rows_rank0=local_rows,
                           min_window=args.min_window_games,
                           min_ring=args.min_ring_rows)
                last_cold_log_t = now
            # Wait on the event so the signal interrupts the sleep.
            shutdown_event.wait(5.0)
            continue

        # ── Bucket gate (rank 0 authority) ────────────────
        if is_main:
            go = 1 if bucket.available() >= global_batch else 0
        else:
            go = 0
        go_t = torch.tensor([go], dtype=torch.long, device=device)
        if use_ddp:
            dist.broadcast(go_t, src=0)
        if go_t.item() == 0:
            now = time.time()
            if is_main and (now - last_budget_log_t) >= WAIT_LOG_INTERVAL_S:
                events.log("BUDGET_SLEEP", bucket=bucket.available(),
                           cap=bucket.max)
                last_budget_log_t = now
            shutdown_event.wait(5.0)
            continue

        # ── LR + ramps ────────────────────────────────────
        lr = lr_sched.lr(step)
        set_lr(lr)
        value_w = linear_ramp(step, args.value_ramp_steps,
                              start=args.value_weight_start,
                              end=args.value_weight_end)
        score_ramp = linear_ramp(step, args.score_ramp_steps, 0.0, 1.0)
        score_mean_w = (args.score_mean_weight_start
                        + (args.score_mean_weight_end - args.score_mean_weight_start) * score_ramp)

        weights = {
            "policy": args.policy_weight,
            "value": value_w,
            "score_mean": score_mean_w,
            "score_stdev": args.score_stdev_weight,
            "score_belief": args.score_belief_weight,
            "ownership": args.ownership_weight,
            "opp_policy": args.opp_policy_weight,
        }

        # ── Sample + step ─────────────────────────────────
        try:
            batch = window.sample_batch(args.batch_size, device,
                                        timeout_s=args.sample_batch_timeout_s)
        except TimeoutError as e:
            if is_main:
                events.log("SAMPLE_TIMEOUT", err=str(e))
            raise

        model.train()
        optimizer.zero_grad(set_to_none=True)

        total, parts = compute_losses(model, batch, bin_centers, belief_sigma,
                                      weights, amp_context())

        if scaler is not None:
            scaler.scale(total).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            total.backward()
            optimizer.step()

        if is_main:
            bucket.drain(global_batch)
        step += 1

        # ── Rolling metrics ───────────────────────────────
        if is_main:
            roll_n += 1
            roll["total"] += total.item()
            for k in parts:
                roll[k] += parts[k]
            loss_ema = (0.98 * loss_ema + 0.02 * total.item()
                        if loss_ema is not None else total.item())

        # ── Status publish ───────────────────────────────
        if is_main and step % args.status_publish_every == 0:
            _atomic_write_json(args.status_file, {
                "step": step,
                "score_ramp": score_ramp,
                "value_ramp": (value_w - args.value_weight_start)
                              / max(1e-9, args.value_weight_end - args.value_weight_start),
                "value_weight": value_w,
                "score_mean_weight": score_mean_w,
                "mcts_score_weight": 0.06 * score_ramp,
                "lr": lr,
                "wall_time": time.time(),
            })

        # ── Metric rows ──────────────────────────────────
        if is_main and step % args.log_every == 0:
            now = time.time()
            dt = now - last_steps_sec_t
            rate = (step - last_steps_sec_count) / max(dt, 1e-6)
            last_steps_sec_t = now
            last_steps_sec_count = step
            gpu_mem = (torch.cuda.memory_allocated(device) // (1024 * 1024)
                       if device.type == "cuda" else 0)
            avg = {k: roll[k] / max(1, roll_n) for k in roll}
            metrics.write({
                "step": step, "wall_time": f"{now:.3f}",
                "samples_seen": step * global_batch,
                "lr": lr, "value_weight": value_w,
                "score_weight_mcts": 0.06 * score_ramp,
                "score_mean_weight": score_mean_w,
                "loss_total": avg["total"], "loss_policy": avg["policy"],
                "loss_value": avg["value"], "loss_score_mean": avg["score_mean"],
                "loss_score_stdev": avg["score_stdev"],
                "loss_ownership": avg["ownership"],
                "loss_score_belief": avg["score_belief"],
                "loss_opp_policy": avg["opp_policy"],
                "window_games": scanner.window_games(),
                "window_rows": scanner.window_rows(),
                "bucket_samples": bucket.available(),
                "bucket_fill_ratio": f"{bucket.fill_ratio():.4f}",
                "ring_rows": window.ring_rows_current(),
                "steps_per_sec": f"{rate:.4f}",
                "gpu_mem_mb": gpu_mem,
            })
            roll = {k: 0.0 for k in roll}
            roll_n = 0

        # ── Export + checkpoint ──────────────────────────
        if step % args.export_every == 0:
            if use_ddp:
                dist.barrier()
            if is_main:
                out_path = os.path.join(args.candidates_dir,
                                        f"v{step:09d}.onnx")
                tmp_path = out_path + ".tmp"
                from export_onnx import export_to_onnx
                base_model = model.module if use_ddp else model
                # Clone weights to CPU copy for export so training state is
                # untouched (the export wrapper does BN folding + runs a dry
                # fwd; we don't want it to touch live GPU tensors).
                cpu_model = create_model(
                    arch=args.arch, board_size=args.board, input_channels=17,
                    num_filters=args.filters, num_res_blocks=args.blocks,
                    d_model=args.d_model, depth=args.depth, heads=args.heads,
                    kv_groups=args.kv_groups, mlp_ratio=args.mlp_ratio,
                    use_fp8=False,
                )
                cpu_model.load_state_dict(base_model.state_dict(), strict=False)
                cpu_model.eval()
                export_to_onnx(cpu_model, tmp_path,
                               board_size=args.board, arch=args.arch)
                os.replace(tmp_path, out_path)

                # Atomic (bucket_level, watermark_id) snapshot — see
                # WindowScanner.checkpoint_snapshot for why this matters.
                bucket_level, watermark_id = scanner.checkpoint_snapshot()
                ckpt = {
                    "model_state_dict": base_model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "step": step,
                    "bucket_level": bucket_level,
                    "watermark_id": watermark_id,
                    "arch": args.arch,
                    "board_size": args.board,
                    "num_filters": args.filters,
                    "num_res_blocks": args.blocks,
                    "d_model": args.d_model,
                    "depth": args.depth,
                    "heads": args.heads,
                    "kv_groups": args.kv_groups,
                    "mlp_ratio": args.mlp_ratio,
                }
                _atomic_save_torch(args.checkpoint, ckpt)
                events.log("EXPORT", step=step, path=out_path,
                           loss_ema=f"{loss_ema:.4f}" if loss_ema is not None else "-",
                           bucket=bucket_level,
                           window_games=scanner.window_games())
            if use_ddp:
                dist.barrier()

        # ── LR drop events ───────────────────────────────
        if is_main:
            hit = lr_sched.milestones_hit(step)
            if hit > last_milestones_hit:
                events.log("LR_DROP", step=step, new_lr=lr,
                           milestones_hit=hit)
                last_milestones_hit = hit

        if args.max_steps > 0 and step >= args.max_steps:
            shutdown_reason = "max_steps"
            break

    # ── Final shutdown: checkpoint before exit ───────────────
    # Saves step, optimizer, bucket, and watermark so resume picks up
    # exactly where we stopped.  Without this, anything since the last
    # EXPORT (up to export_every - 1 steps of optimizer progress and
    # replay-ratio accounting) is discarded on Ctrl-C / supervised stop.
    if use_ddp:
        dist.barrier()
    if is_main:
        try:
            base_model = model.module if use_ddp else model
            # Atomic snapshot — see WindowScanner.checkpoint_snapshot.
            if scanner is not None:
                bucket_level, watermark_id = scanner.checkpoint_snapshot()
            else:
                bucket_level = bucket.available() if bucket else 0
                watermark_id = 0
            final_ckpt = {
                "model_state_dict": base_model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "step": step,
                "bucket_level": bucket_level,
                "watermark_id": watermark_id,
                "arch": args.arch,
                "board_size": args.board,
                "num_filters": args.filters,
                "num_res_blocks": args.blocks,
                "d_model": args.d_model,
                "depth": args.depth,
                "heads": args.heads,
                "kv_groups": args.kv_groups,
                "mlp_ratio": args.mlp_ratio,
            }
            _atomic_save_torch(args.checkpoint, final_ckpt)
            events.log("FINAL_CHECKPOINT", step=step,
                       reason=shutdown_reason or "loop_exit",
                       bucket=bucket_level,
                       watermark=watermark_id)
        except Exception as e:
            events.log("FINAL_CHECKPOINT_FAIL", step=step, err=str(e))
    if use_ddp:
        # Ensure every rank has finished the save barrier before we tear
        # down the process group.  Without this, a rank that exits first
        # and calls destroy_process_group() can race with another rank
        # still inside a collective op.
        dist.barrier()

    # Stop background threads (daemons, so already doomed on process
    # exit — but asking them to stop cleanly avoids stderr spam).
    if scanner is not None:
        scanner.stop()
    window.stop_ingest()
    if is_main:
        events.log("TRAIN_STOP", step=step, reason=shutdown_reason or "loop_exit")
        events.close()
    if use_ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
