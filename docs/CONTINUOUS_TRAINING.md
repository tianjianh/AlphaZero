# Continuous Training

KataGo-style continuous pipeline. Four independent workers — **selfplay**,
**train**, **gatekeeper**, and optionally **rate** — run concurrently and
coordinate only through the filesystem. This is the ONLY training
pipeline in the repo (the old phased iterate/train/eval loop has been
removed).

This document is both the design rationale and the **operator's
reference**: how to run it, what each knob does, where logs land, and
how to diagnose common problems.

---

## Table of contents

1. [Quick start](#quick-start)
2. [Architecture](#architecture)
3. [Key concepts](#key-concepts)
4. [CLI reference](#cli-reference)
5. [File layout](#file-layout)
6. [Logging and diagnostics](#logging-and-diagnostics)
7. [Sanity checkpoints](#sanity-checkpoints)
8. [Tuning knobs](#tuning-knobs)
9. [Operations](#operations)

---

## Quick start

Two commands: `init` (once, seeds bootstrap model) and `run` (launches
the four workers).

```bash
# Python deps for the pipeline (torch, numpy, zstandard, onnx):
pip install -r scripts/requirements.txt

# One-time: archive any previous run, build C++ binaries, seed
# models/accepted/v000000000.onnx + accepted/latest symlink.
# (Backend auto-detected by CMake: tensorrt > opencl > eigen on Linux;
#  force one with MINIGO_BACKEND=opencl python scripts/run_continuous.py init)
python scripts/run_continuous.py init -y

# Launch the four workers.  Single-GPU (everything on GPU 0) is the default:
python scripts/run_continuous.py run

# 2x4090 recommended split (fully explicit — all tunable knobs shown):
python scripts/run_continuous.py run \
  \
  `# GPU assignment (CUDA_VISIBLE_DEVICES per worker)` \
  --train-gpus 0,1 \
  --selfplay-gpus 0,1 \
  --gate-gpus 0 \
  --rate-gpus 1 \
  \
  `# NN device IDs (0-indexed against the VISIBLE set above)` \
  --selfplay-nn-device-ids 0,0,1,1 \
  --gate-nn-device-ids 0 \
  --rate-nn-device-ids 0 \
  \
  `# NN server thread count (must equal len of --<proc>-nn-device-ids)` \
  --selfplay-nn-server-threads 4 \
  --gate-nn-server-threads 1 \
  --rate-nn-server-threads 1 \
  \
  `# CPU worker threads (0 = os.cpu_count())` \
  --selfplay-threads 0 \
  --gate-threads 0 \
  --rate-threads 0 \
  \
  `# MCTS search threads per move, per worker` \
  --selfplay-search-threads 16 \
  --gate-search-threads 16 \
  --rate-search-threads 16 \
  \
  `# Batch size — shared across selfplay + evaluate for TRT cache reuse` \
  --max-batch 256 \
  \
  `# Training-side DDP batch (per rank; global batch = batch-size × world_size)` \
  --batch-size 512 \
  \
  --rating
```

Ctrl-C once shuts down gracefully (workers get SIGTERM, flush logs,
exit). Ctrl-C twice force-kills.

Resumable: re-run `python scripts/run_continuous.py run ...`. Training
restores weights, optimizer state, step counter, bucket level, and the
file watermark from `training/checkpoints/training.pt`. Selfplay's
monotonic filenames pick up from `max(existing IDs) + 1`. Accepted
models and `logs/<ts>/` directories from prior runs are preserved —
each new run creates a fresh `logs/<new-ts>/`.

---

## Architecture

```
selfplay ──(game records)──> training/selfplay/   <──(read)── train
                                                                │
                                                                │ (export every N steps)
                                                                ▼
                                                         models/candidates/
                                                                │
                                                                ▼
                                                          gatekeeper
                                                        (200 games, 150 sims)
                                                         accept/reject
                                                        ╱            ╲
                                                       ▼              ▼
                                           models/accepted/     models/rejected/
                                                       │
                                                       │ (atomic symlink swap)
                                                       ▼
                                            selfplay polls this dir
                                                       │
                                                       ▼
                                           rating (optional, monitoring)
                                           ──> ratings/elo.csv
```

All four workers run in parallel. GPU assignment per-process via
`CUDA_VISIBLE_DEVICES` set by the supervisor. No IPC — only files.

### Workers

- **selfplay** (`scripts/selfplay_driver.py`) — Python supervisor around
  the existing `build/selfplay` binary. Each iteration:
  1. Resolves `models/accepted/latest` → concrete version path.
  2. Checks `training/status.json` `bucket_fill` → pauses while the
     trainer's bucket is saturated (THROTTLE, hysteresis 0.9/0.5).
  3. Reads `training/status.json` → current `score_ramp`.
  4. Spawns `build/selfplay` into a fresh staging dir (N games).
  5. Compresses `game_*.bin` → `g_<id>.bin.zst` with monotonic IDs,
     atomic rename.
  6. Prunes the pool to `window_games` files (oldest first).

  Sole writer of `training/selfplay/`. Crash-safe: `next_id =
  max(existing IDs on disk) + 1`, recomputed per batch. Orphan `.tmp`
  files and leftover staging dirs are cleaned on startup.

- **train** (`scripts/train_continuous.py`) — Single `torchrun` DDP
  process. One long step loop gated by a KataGo-style train bucket that
  fills as new selfplay rows arrive and drains as training consumes
  samples. Exports ONNX at fixed step intervals into
  `models/candidates/`. Persists Adam state across restarts.

  Rank 0 owns the Bucket, Scanner thread, file watermark, and all
  persistence. Every rank runs its own `WindowRingBuffer` ingest
  thread. Every step is a collective decision: rank 0 checks the
  bucket, broadcasts go/no-go, all ranks sleep or step together.

- **gatekeeper** (`scripts/gatekeeper.py`) — Polls
  `models/candidates/` every `gate_poll_interval` seconds. Picks the
  newest candidate (lex-sort = step-sort), moves older candidates
  to `rejected/` as GATE_STALE_DROP, plays `gate_games` games at
  `gate_sims` visits vs `accepted/latest`. Accepts iff `wins/total
  > gate_threshold` (Go scoring: draws count as losses). On accept:
  atomic rename into `accepted/`, atomic symlink swap on
  `accepted/latest`.

- **rate** *(optional, `scripts/rate.py`)* — Round-robin tournament
  over the k most-recent accepted models. Bradley-Terry MLE → Elo,
  appended to `ratings/elo.csv`. Pure monitoring; does not gate
  anything.

### File-based coordination

- Selfplay → train: game records in `training/selfplay/g_*.bin.zst`
  (monotonic IDs).
- Train → gatekeeper: ONNX dumps in `models/candidates/v*.onnx`.
- Gatekeeper → selfplay: `models/accepted/latest` symlink swap.
- Train → selfplay: `training/status.json` — carries BOTH the ramp
  state (`score_ramp`, read every batch to scale the MCTS score
  weight) and the backpressure signal (`bucket_fill`, read to decide
  whether to pause selfplay — see "Selfplay throttle" below).
- Init → run: `training/run_config.json` — architecture + komi chosen
  at `init`; `run` loads it so nothing architecture-defining is ever
  re-specified, and contradicting CLI flags are a hard error.

All writes are atomic (`*.tmp` → `os.rename`).

---

## Key concepts

### Train bucket (replay-ratio enforcement)

The trainer consumes samples from the rolling window, but only as
fast as selfplay produces them. KataGo-style credit:

```
bucket          = current credit in samples (rows)
max_samples     = bucket_cap_mult * batch_size * world_size   (default 512 * global_batch)
watermark_id    = max filename ID already credited

Scanner (rank 0, every ~5s):
  new_rows = sum of row counts of files with id > watermark_id
  bucket += replay_target * new_rows                (clamped at max_samples)
  watermark_id = max id seen this pass

Trainer:
  if bucket < batch_size * world_size: sleep (BUDGET_SLEEP); continue
  train_step(...)
  bucket -= batch_size * world_size
```

`replay_target = 4` means each unique position is visited ~4 times
(KataGo's semantics — their `-max-train-bucket-per-new-data`).  With
V3 records one disk row IS one unique position — dihedral augmentation
happens at sample time in the loader (each visit draws a fresh random
transform), so there is no augmentation divisor anywhere.

**Cap sizing**: one selfplay publish (~300 games ≈ 240k rows) credits
~120k samples in one scanner tick.  The cap must dwarf that chunk or
credit silently clips on every publish, driving the effective replay
ratio below `replay_target`.  KataGo's default is an unbounded bucket
(`python/train.py`: `max_train_bucket_size = None → 1e30`);
`bucket_cap_mult = 512` (≈ 4-5 publish chunks at global batch 1024)
is the bounded approximation.  Runaway data lead is prevented by the
selfplay throttle, not by the cap.

### Selfplay throttle (rollout↔train backpressure)

The bucket paces the TRAINER against data.  The reverse — selfplay
outpacing training — needs its own valve on a single host (KataGo's
async setup never needed one: their trainer can't be outpaced by
distributed contributors it doesn't control, and their single-host
recipe `synchronous_loop.sh` strictly alternates instead).

The trainer publishes `bucket_fill` (level / cap) in `status.json`,
including during its wait states.  The selfplay driver checks it
between batches:

```
fill >= throttle_high (0.9)  → THROTTLE_PAUSE: stop launching batches
fill <  throttle_low  (0.5)  → THROTTLE_RESUME: continue
step == 0 or no status.json  → no throttle (cold-start fills freely)
```

Hysteresis prevents flapping; the pause means selfplay GPU time is
never spent generating positions whose training credit would be
discarded at the bucket cap.  A dead trainer (stale status, full
bucket) keeps selfplay paused — that is the correct behavior, since
data generated while nothing trains is pure waste; the supervisor's
restart of the trainer releases the valve.  Disable with
`--throttle-high 0`.

**Why filename-ID-based, not file-index-based**: selfplay prunes old
files once the pool exceeds `window_games`. An index into
`sorted(pool)` shifts when files are pruned; a monotonic filename ID
survives any pruning pattern.

### Per-rank ring buffer (compressed-in-memory, game-granular)

`window_games` governs **disk retention** (enforced by selfplay's
pruner). The trainer's **effective sampling window** is a smaller
in-RAM ring — default `ring_games = 2000` game slots per rank.

Each slot holds one game's raw **compressed** `g_*.bin.zst` bytes
(~25 KB) plus its row count; the whole ring is a few hundred MB
instead of the ~10 GB a decompressed row-ring would need.
Decompression + parse happens lazily per batch, only for the games a
batch touches.

Sampling is **game-granular** (the same trick as KataGo's npz
shuffler): pick `K = batch_size / samples_per_game` games uniformly
with replacement, decompress them in parallel (small thread pool;
zstd releases the GIL), take `samples_per_game` rows from each.
Strict row-uniform sampling over a compressed ring would touch ~240
distinct games per 256-row batch (coupon collector) and decompress
for ~1.2 s per step; game-granular at `samples_per_game = 8` touches
~32 and lands under 100 ms.  Trade-off: rows sharing a game share
value/score targets (~3% of the batch per game) — KataGo accepts the
same correlation.

Each rank ingests independently. On resume, the ring's ingest starts
from roughly the newest `1.5 × ring_games` files (not id=0), so it
rehydrates with the freshest window instead of replaying the oldest
retained pool — which would briefly have the trainer learning from
hours-stale data before the cold-start gate notices.

### Cold-start gate

Two conditions must **both** hold before any rank runs `train_step`:

1. Disk readiness: rank 0's scanner sees ≥ `min_window_games`
   files (default 2000).
2. Ring readiness: every rank reports `ring_rows >=
   min_ring_rows` (default 10240). Reduced across ranks with
   `dist.all_reduce(MIN)`.

A rank-0-only gate is not sufficient — rank 0's scanner can see
plenty of disk files while a slower rank's ingest is still warming
up. Proceeding into `train_step` in that state either blocks inside
`sample_batch` or desyncs the DDP allreduce.

### Pool enumeration cost

Both the scanner (rank 0) and the per-rank ingest threads enumerate
`training/selfplay/` on a timer. To avoid the retention-sized scan
becoming a measurable tax on long runs:

- Enumeration uses `os.scandir` — no sorted-glob of the full pool.
  Each poll filters to `gid > watermark` *before* sorting, so only
  the tiny "new files" subset is sorted.
- The scanner caches per-file row counts keyed by filename ID. Pool
  files are immutable after publish (selfplay writes `.tmp` + rename),
  so a V2 header read happens exactly once per file per run. Pruned
  files are evicted from the cache via set-diff each tick.
- The ingest thread polls aggressively (0.5s) while behind and backs
  off exponentially (up to 3s) once caught up.

### Ramps (2 knobs, both with phased-pipeline precedent)

Fresh-start safety. Both are **step-driven**, not signal-driven.

- **Value-head weight** ramps linearly from 1.0 at step 0 to 2.0 at
  step `value_ramp_steps` (default 30k). Raw value CE drops from ~0.8
  (random init) to ~0.08 (converged); a fixed weight of 2.0 from step
  0 would make value loss dwarf policy loss and waste early capacity
  on a noise signal.

- **Score ramp** is a single 0 → 1 factor over `score_ramp_steps`
  (default 50k) that drives two weights, each owned by the component
  that uses it:
  - trainer: `head_score_mean_weight(step)` ramps
    `--score-mean-weight-start 0.004` → `--score-mean-weight-end 0.008`
  - selfplay driver: `mcts score_weight = --score-weight-max (0.04) ×
    score_ramp` — computed BY THE DRIVER from the published ramp;
    the trainer does not know (or publish) the selfplay maximum.

  The score head is untrustworthy on a random-init network; its
  predictions actively distort MCTS search by biasing Q values toward
  meaningless scores. Only these two ramp —
  `head_score_stdev_weight` (0.006), `head_score_belief_weight`
  (0.035), and `head_ownership_weight` (0.85) stay fixed from step 0.

Rank 0 publishes `score_ramp` to `training/status.json` every
`status_publish_every` steps (default 100), and at least every ~30 s
during cold-start / budget-sleep waits (which doubles as the
liveness signal for the throttle). Selfplay re-reads the file at the
start of each batch. This gives selfplay a smooth ramp signal
(minutes-scale updates) rather than multi-hour jumps tied to
candidate exports.

### LR schedule

Fixed step schedule with linear warmup + milestone halvings:

- Warmup: linear 0 → `base_lr` over first `warmup_steps`.
- Hold at `base_lr` until first milestone.
- At each milestone: LR *= `lr_gamma` (default 0.5).
- After the final milestone: LR stays at the floor.

Defaults (`base_lr=3e-4`, milestones `100000, 400000, 1500000`) are
sized so the first halving lands mid-run under conservative throughput
(~180k steps/week → first drop at day 4). No auto-plateau detection —
early training loss EMA is too noisy; signal-driven halving fires
prematurely.

### CPU-side parallelism

Each C++ worker binary (`build/selfplay`, `build/evaluate`) has three
layers of concurrency, all exposed as explicit per-worker flags on
the supervisor (the `proc` below is one of `selfplay`, `gate`, `rate`):

- **`--<proc>-threads`** — parallel game / match / pair workers.
  One worker runs one simulation at a time on the CPU side and
  dispatches NN requests to the batching layer. Default `0` =
  `os.cpu_count()`. The three concrete flags: `--selfplay-threads`,
  `--gate-threads`, `--rate-threads`.

- **`--<proc>-search-threads`** — MCTS search threads per worker,
  per move. These split the per-move simulation budget (e.g.
  500 sims) inside one position. Default `16`. The three concrete
  flags: `--selfplay-search-threads`, `--gate-search-threads`,
  `--rate-search-threads`.

- **`--<proc>-nn-server-threads`** — NN batching threads that collect
  requests from every search thread and ship them to the GPU. Must
  equal `len(--<proc>-nn-device-ids)`. Defaults: 2 for selfplay
  (matches `--selfplay-nn-device-ids 0,0`), 1 for gate and rate.
  The three concrete flags: `--selfplay-nn-server-threads`,
  `--gate-nn-server-threads`, `--rate-nn-server-threads`.

Rule of thumb for a 2×4090 selfplay run: 64 game workers × 16 search
threads × 4 NN server threads (`--selfplay-nn-device-ids 0,0,1,1`).
One game worker stays busy waiting for NN replies while others queue
up, keeping the batcher near its `--max-batch` size.

**`--max-batch`** (default `256`) is intentionally a single shared
flag — not per-worker — because the TRT engine cache key includes it.
Selfplay and gate must use the same value for the cache to be
shared between them.

### TRT engine cache sharing

The TRT cache key is basename-based (see `src/backends/tensorrt_compute.cpp`).
Candidates are named `v{step:09d}.onnx` and preserved across the
`candidates/ → accepted/` rename. Result: the gatekeeper's match warms
the TRT cache; selfplay's next batch loads the plan from disk in <1s.
No 30-120s rebuild tax per promotion.

The cache directory is resolved from the `MINIGO_TRT_CACHE` environment
variable, which the supervisor sets to an absolute
`<project_root>/models/trt_cache` path and propagates to every worker's
env. Standalone wrapper runs (`python scripts/selfplay_driver.py ...`)
set the same default via `os.environ.setdefault`, so gatekeeper-warmed
plans are reused regardless of the launch cwd. If `MINIGO_TRT_CACHE`
is unset, the C++ binaries fall back to `trt_cache/` relative to cwd
(legacy behavior; matches the phased pipeline).

Requirement: all workers must pass the same `--max-batch`. The
supervisor forwards a single shared value.

---

## CLI reference

### `scripts/run_continuous.py init`

One-time bootstrap. Archives any existing `training/`, `models/`,
`ratings/`, `logs/` into `archive/<ts>/` (no deletion), builds the
C++ binaries if needed, seeds `models/accepted/v000000000.onnx` with
a random-init ONNX, and writes **`training/run_config.json`** — the
single source of truth for architecture + komi that `run` reads back.

```
--arch {resnet,vit,katago}  # default: resnet
--board N                # default: 9
--filters N              # ResNet filters (default: 128)
--blocks N               # ResNet blocks (default: 10)
--d-model N              # ViT (default: 192)
--depth N                # ViT (default: 8)
--heads N                # ViT (default: 6)
--kv-groups N            # ViT GQA (default: 2)
--mlp-ratio N            # ViT (default: 4)
--komi F                 # default: 7.5 (recorded for selfplay/gate/rate)
-y, --yes                # skip confirmation
```

### `scripts/run_continuous.py run`

Launches all workers. All CLI flags have sensible defaults; override
only what you need.  **Architecture + komi come from
`training/run_config.json`** — passing a flag that contradicts it is a
hard error (change architecture only via a fresh `init`).

**Per-worker GPU + NN assignment** (naming convention: every flag
tied to one worker is prefixed with that worker's name, so `--help
| grep ^--selfplay-` enumerates every selfplay knob):
```
# CUDA_VISIBLE_DEVICES for each worker
--selfplay-gpus 0,1
--train-gpus 0,1              # torchrun --nproc_per_node = list length
--gate-gpus 0
--rate-gpus 1

# NN device ids passed to the C++ binary; 0-indexed against the visible set
--selfplay-nn-device-ids 0,0,1,1
--gate-nn-device-ids 0
--rate-nn-device-ids 0

# NN server thread count; must equal len of --<proc>-nn-device-ids
# (defaults to that length if unset)
--selfplay-nn-server-threads 4
--gate-nn-server-threads 1
--rate-nn-server-threads 1

# CPU-side parallelism (see "CPU-side parallelism" concept section)
--selfplay-threads 0          # 0 = os.cpu_count()
--gate-threads 0
--rate-threads 0
--selfplay-search-threads 16
--gate-search-threads 16
--rate-search-threads 16
```

**Model:**
```
--arch --board --filters --blocks
--d-model --depth --heads --kv-groups --mlp-ratio
--fp8                   # NVIDIA Transformer Engine FP8 (Blackwell+)
```

**Training:**
```
--batch-size 256           # per DDP rank (global = 256 × world_size)
--base-lr 3e-4
--warmup-steps 2000
--lr-milestones 100000,400000,1500000
--lr-gamma 0.5
--weight-decay 1e-4
--replay-target 4.0
--ring-games 2000
--bucket-cap-mult 512
--min-window-games 2000
--min-ring-rows 10240
--sample-batch-timeout-s 30
--export-every 5000
--status-publish-every 100
--log-every 100
--value-ramp-steps 30000
--score-ramp-steps 50000
--value-weight-start 1.0     # head-weight ramp endpoints
--value-weight-end   2.0
--score-mean-weight-start 0.004
--score-mean-weight-end   0.008
--policy-weight 1.0          # fixed head weights (no ramp, but overridable)
--score-stdev-weight 0.006
--score-belief-weight 0.035
--ownership-weight 0.85
--opp-policy-weight 0.1
```

(Per-worker thread / GPU / NN-server knobs are in the "Per-worker
GPU + NN assignment" block above. These per-section blocks are just
the workload-shape knobs.)

**Selfplay workload:**
```
--selfplay-batch-games 300    # games per build/selfplay invocation
--selfplay-sims 600           # MCTS simulations per move
--window-games 100000         # disk retention cap
--score-weight-max 0.04       # MCTS score weight at full ramp
--throttle-high 0.9           # pause selfplay at this bucket fill (0 = off)
--throttle-low 0.5            # resume below this fill
```

**Gatekeeper workload:**
```
--gate-games 200              # games per match
--gate-sims 150               # MCTS simulations per move
--gate-threshold 0.5          # accept if wins/total > this
--gate-poll-interval 30       # seconds between candidate polls
```

**Rating workload (optional):**
```
--rating                      # off by default
--rating-games 80             # games per pair
--rating-sims 200
--rating-pool-size 5          # rate the k most-recent accepted models
--rating-interval 7200        # seconds between rounds
```

**MCTS / game (shared across all workers):**
```
--c-puct 1.25
--dirichlet-alpha 0.15
--dirichlet-epsilon 0.22
--temp-threshold 12
--komi 7.5
--win-loss-weight 1.0
--score-scale 18.0
--max-batch 256               # SHARED, not per-worker — part of the TRT
                              # engine cache key, so selfplay + gate must
                              # use the same value to share plans
```

**Shutdown:**
```
--graceful-timeout 180        # seconds; SIGKILL'd after this on shutdown
```

**Heartbeat:**
```
--heartbeat-interval 60       # seconds between supervisor HEARTBEAT lines
                              # (pool / candidates / accepted / step / lr).
                              # 0 disables.
```

Sample heartbeat line:
```
[2026-04-24 10:00:00] HEARTBEAT step=12500 lr=3.00e-04 score_ramp=0.25
   pool=43207 cand=2 accepted=4 rejected=1 latest=v000012500.onnx
   alive=train,selfplay,gatekeeper,rate dead=-
```

### `scripts/run_continuous.py status`

Prints accepted/candidate/rejected model counts, pool size, current
step + LR + ramp state from `training/status.json`, and the
`logs/current` symlink target.

### Lower-level scripts

Each worker has its own `--help` and can be run stand-alone for
debugging:

```
python scripts/train_continuous.py --help
python scripts/selfplay_driver.py --help
python scripts/gatekeeper.py --help
python scripts/rate.py --help
```

---

## File layout

```
training/
├── selfplay/                         # flat pool; no iteration subdirs
│   ├── g_00000000000000001.bin.zst   # monotonic, sortable-by-name
│   ├── g_00000000000000002.bin.zst
│   └── ...
├── checkpoints/
│   └── training.pt                   # weights + optimizer + step +
│                                     #   bucket_level + watermark_id
├── run_config.json                   # arch + komi (written by init,
│                                     #   loaded + enforced by run)
└── status.json                       # rank 0 publishes here every
                                      #   STATUS_PUBLISH_EVERY steps and
                                      #   during wait states:
                                      #   {state, step, samples_seen,
                                      #    global_batch, score_ramp,
                                      #    value_weight, score_mean_weight,
                                      #    lr, bucket_level, bucket_cap,
                                      #    bucket_fill, window_games,
                                      #    wall_time}

models/
├── candidates/                       # trainer writes here
│   └── v000005000.onnx
├── accepted/                         # gatekeeper promotes here
│   ├── v000000000.onnx               # bootstrap
│   ├── v000005000.onnx
│   └── latest -> v000005000.onnx     # atomic symlink, updated on promotion
├── rejected/
│   └── v000002500.onnx
└── trt_cache/                        # shared TRT engine cache
                                      # (MINIGO_TRT_CACHE set by supervisor)

ratings/
├── rating_state.json                 # accumulated pairwise game counts
└── elo.csv                           # step, model, elo, games

logs/
├── <ts>/                             # new dir per run
│   ├── train.log                     # train events (EXPORT, LR_DROP, ...)
│   ├── train_metrics.csv             # one row per LOG_EVERY steps
│   ├── train.stdio.log               # raw stdout/stderr of torchrun
│   ├── selfplay.log
│   ├── selfplay_batches.csv          # one row per batch
│   ├── selfplay.stdio.log
│   ├── gatekeeper.log
│   ├── gate_decisions.csv            # one row per candidate evaluated
│   ├── gatekeeper.stdio.log
│   ├── rate.log                      # only if --rating
│   ├── rate.stdio.log
│   └── supervisor.log                # SPAWN / CRASH / RESTART events
└── current -> <ts>                   # always points at the live run

archive/<ts>/                         # previous run artifacts after init
```

---

## Logging and diagnostics

### Structured CSVs

**`train_metrics.csv`** (aggregated per 100 steps by default):
```
step, wall_time, samples_seen, lr, value_weight, score_ramp,
score_mean_weight, loss_total, loss_policy, loss_value, loss_score_mean,
loss_score_stdev, loss_ownership, loss_score_belief, loss_opp_policy,
window_games, window_rows, bucket_samples, bucket_fill_ratio, ring_rows,
steps_per_sec, gpu_mem_mb
```

**`selfplay_batches.csv`**:
```
batch_id, wall_time_start, wall_time_end, model_in_use, games_played,
positions_written, duration_s, selfplay_duration_s, score_weight,
pool_size, publish_failures, throttle_wait_s
```

The `publish_failures` column counts games in a batch whose zstd
compression / atomic-rename step failed; source `.bin` files for those
games are preserved in `training/selfplay/staging/batch_<id>.failed-<ts>/`
for postmortem. Nonzero values on a long unattended run usually point
at disk-full conditions or a flaky filesystem.

**`gate_decisions.csv`**:
```
wall_time, cand_step, baseline_step, games, sims, wins, draws, losses,
score, threshold, verdict, match_duration_s
```

**`ratings/elo.csv`**:
```
wall_time, round, model, step, elo, games
```

### Event tags (text logs)

```
EXPORT           step=5000 path=... loss_ema=2.3 bucket=41024
MODEL_SWAP       old=v000001000.onnx new=v000002000.onnx
GATE_MATCH_START cand=... baseline=...
GATE_ACCEPT      cand=... score=0.54 games=200
GATE_REJECT      cand=... score=0.41 games=200
GATE_STALE_DROP  cand=...
LR_DROP          step=100000 new_lr=1.5e-4 milestones_hit=1
BUDGET_SLEEP     bucket=0 cap=131072
COLD_START_WAIT  window_games=412 ring_rows_rank0=6000
THROTTLE_PAUSE   bucket_fill=0.917 high=0.9 low=0.5   (selfplay.log)
THROTTLE_RESUME  bucket_fill=0.492 waited_s=45        (selfplay.log)
HEARTBEAT        step=... state=... bucket=... pool=...  (supervisor.log)
SPAWN            proc=train pid=12345 gpus=0,1
CRASH            proc=... rc=... — followed by RESTART_SCHEDULED,
                 RESTART or DISABLED (backoff no longer blocks the
                 supervise loop)
```

### Diagnostic playbook

| Symptom                               | Where to look                                      | Likely knob                               |
| ------------------------------------- | -------------------------------------------------- | ----------------------------------------- |
| Trainer sleeping often                | `train.log` BUDGET_SLEEP frequency                 | `replay_target` too low / selfplay slow   |
| Trainer bucket saturated              | selfplay.log THROTTLE_PAUSE (expected behavior)    | training is the bottleneck: more train GPU, or accept the pacing |
| Gatekeeper queue growing              | `gate_decisions.csv` arrival vs verdict rate       | `export_every` too low / `gate_games` high|
| Many GATE_STALE_DROP                  | `gatekeeper.log`                                   | `export_every` too low vs gate rate       |
| All candidates ACCEPT                 | `gate_decisions.csv` verdict column                | `gate_threshold` too loose                |
| All candidates REJECT                 | same                                               | `gate_threshold` too tight / LR too high  |
| Selfplay using stale model            | `selfplay.log` MODEL_SWAP vs GATE_ACCEPT times     | `selfplay_batch_games` too large          |
| LR never drops                        | `train.log` LR_DROP vs milestones                  | `lr_milestones` beyond run length         |
| Elo chart flat despite promotions     | `ratings/elo.csv` deltas small                     | `gate_threshold` too loose (false accepts)|
| Elo chart jittery                     | `ratings/elo.csv`                                  | `rating_games` too small for signal       |
| TRT rebuild on every batch            | `selfplay.log` slow batch starts                   | cache key bug or `max-batch` mismatch     |

Logs are small compared to `.bin.zst` files. Keep everything for the
duration of a run. On `init`, old logs are archived (not deleted).

---

## Sanity checkpoints

A 1-week run is too long to discover the pipeline is broken on day 6.
Check these signals early; abort and debug if any are missing.

**6 hours in — basic pipeline health:**
- `selfplay_batches.csv` shows batches completing at
  `games_played == selfplay_batch_games` (no mid-batch crashes).
- `train_metrics.csv` has ≥ 5000 steps logged; `loss_total` and
  `loss_policy` are trending down.
- At least one `EXPORT` event in `train.log`.
- `bucket_fill_ratio` fluctuates in 0.05–0.9 (neither pinned empty
  nor pinned full).
- No repeating `CRASH` / `RESTART` on any worker.

**24 hours in — early strength signal:**
- ≥ 5 `GATE_ACCEPT` events. 0% accept rate → model isn't improving
  fast enough; check loss curves and LR. 100% accept rate →
  `gate_threshold` too loose.
- If rating enabled: `ratings/elo.csv` shows ≥ 100 Elo between
  `v000000000` (bootstrap) and the newest accepted. Flat curve →
  promotions aren't translating to real strength; inspect
  `gate_decisions.csv` score distribution.
- `selfplay.log` MODEL_SWAP within minutes of GATE_ACCEPT (not
  hours — that means `selfplay_batch_games` is too large).

**72 hours in — steady-state:**
- Step count roughly matches conservative projection
  (~78k steps at 0.3 steps/s). Far below → check
  `bucket_fill_ratio` for BUDGET_SLEEP pinning.
- Elo curve still rising (not plateaued ~200-300 Elo above
  bootstrap — that's the first-LR-milestone plateau).
- If LR milestone hit: `lr` column in `train_metrics.csv` reflects
  the drop.

### Scale-up decision (after a proven 1-week run)

- Elo still rising at week's end: extend runtime, don't change knobs.
- Trainer BUDGET_SLEEPing > 50%: selfplay-bound. Add GPUs to
  selfplay, reduce gatekeeper/rating GPU share, or lower
  `replay_target` to 3.
- Bucket saturated > 80%: train-bound. Add GPUs to training (wider
  DDP), raise `replay_target` to 5-6, or raise `batch_size` to 2048.
- Elo plateaus with no imbalance: model-capacity-bound. Scale
  architecture (64f/5b → 128f/10b) before scaling runtime.
- Throughput held and the run produced a usable curve: pipeline is
  proven. Next run can push `window_games`, `export_every`, and
  `lr_milestones` higher with confidence.

---

## Tuning knobs

Most-likely tuning targets based on real run behavior:

| Knob                  | Default | When to change                                                                 |
| --------------------- | ------- | ------------------------------------------------------------------------------ |
| `replay_target`       | 4       | BUDGET_SLEEPing often → try 3. Bucket always saturated → try 5-6. Cap at 8.    |
| `export_every`        | 5000    | Lower for snappier feedback; but gate matches cost ~60-120s, so <2000 queues.  |
| `gate_threshold`      | 0.5     | Many accepts with no Elo gain → bump to 0.52-0.55.                             |
| `value_ramp_steps`    | 30000   | Value loss plateaus before ramp finishes → shorten. Still noisy at 30k → 50k+. |
| `score_ramp_steps`    | 50000   | Flat Elo stretch right around step 50k → shorten. Score head still wrong past 50k → lengthen. |
| `bucket_cap_mult`     | 512     | Cap = mult × global batch. Must dwarf one publish chunk (~120k samples) or credit clips; runaway lead is prevented by the selfplay throttle, not the cap. |
| `window_games`        | 80000   | Shorter (30k) for fresher data / less disk. Governs retention, not sampling.   |
| `ring_games`          | 2000    | Raise (4-5k → 20-25 GB/rank) for more diversity if RAM allows. Don't go below 1000 (intra-batch correlation). |
| `lr_milestones[0]`    | 100000  | Throughput much lower than conservative and first drop never fires → lower to 80k, or lower `base_lr` (cleaner). |
| `batch_size`          | 1024    | Per-rank. Raise to 2048 once the pipeline is proven and memory allows.         |

---

## Operations

### Resuming a run

Just re-run `python scripts/run_continuous.py run ...`. Training
picks up from the most recent `training/checkpoints/training.pt`
(weights, optimizer, step, bucket level, watermark). A fresh
`logs/<ts>/` is created for the new session — the `logs/current`
symlink points at it.

### Retention during a trainer outage

Selfplay keeps writing + pruning while the trainer is down. If the
outage exceeds the retention window
(`window_games / selfplay_rate` ≈ 15h at defaults), the oldest files
that aged out during the outage are gone before resume; their rows
never contribute bucket credit. A day-long outage with defaults loses
bucket credit for roughly the first 9 hours of the outage — the
remaining 15h of files still get credited on first scan.

For supervisor-driven restarts (<1 min) nothing is lost.

### GPU over-subscription

Multiple workers sharing a GPU is allowed — the driver time-slices.
Typical 2×4090 config stacks selfplay + training across both GPUs,
with gatekeeper sharing GPU 0 (brief, infrequent matches) and rate
sharing GPU 1.

### Crash handling

The supervisor restarts crashed workers with exponential backoff
(1s, 2s, 4s, ..., capped at 5 min). If a worker crashes more than 3
times within a 10 min window, it is disabled and `supervisor.log`
records a `DISABLED` event. Other workers keep running. Inspect the
worker's `*.stdio.log` for the traceback.

### Shutdown

Two-stage graceful shutdown:

- **Ctrl-C once**: supervisor sends SIGTERM to each worker's Python
  wrapper PID only (not the process group). The wrapper flips a stop
  flag and lets its in-flight C++ child (selfplay batch, gatekeeper
  match, rate round) run to completion. The trainer broadcasts the
  shutdown decision across DDP ranks, then rank 0 saves a final
  checkpoint (step + optimizer + bucket + watermark) so resume picks
  up exactly where we stopped. Supervisor waits up to
  `--graceful-timeout` seconds (default 180s).
- **Ctrl-C twice, or graceful timeout exceeded**: supervisor sends
  SIGKILL to each worker's entire process group (wrapper + any C++
  child). Guarantees no orphans on supervisor exit.

Each worker is launched with `start_new_session=True` so the supervisor
can reach the whole tree via `os.killpg`; standalone wrapper runs
behave the same.

No special teardown needed — all state is on-disk and
restart-friendly. A SIGKILL between the trainer's FINAL_CHECKPOINT save
and the next EXPORT still loses at most `status_publish_every` steps,
because the previous export (or the last FINAL_CHECKPOINT on the
previous clean stop) is the fallback resume point.

### Running a single worker standalone

Useful for debugging. Each worker is fully self-contained:

```bash
# Just the trainer (useful to iterate on a checkpoint)
CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 \
  scripts/train_continuous.py \
  --pool-dir training/selfplay \
  --candidates-dir models/candidates \
  --checkpoint training/checkpoints/training.pt

# Just selfplay against the current accepted/latest
python scripts/selfplay_driver.py --games-per-batch 50

# Just the gatekeeper
python scripts/gatekeeper.py
```

Each script writes to `logs/current/<worker>.log` when that symlink
exists, or you can pass `--log-dir <path>` explicitly.

---

## See also

- `COMPARISON_WITH_KATAGO.md` — how this pipeline mirrors (and
  deviates from) KataGo's training code.
- `README.md` — project overview, build, architecture.
- `TRAINING_STRATEGY.md` — postmortems and tuning rationale from
  earlier runs.
