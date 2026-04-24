# Continuous Training

KataGo-style continuous pipeline. Four independent workers — **selfplay**,
**train**, **gatekeeper**, and optionally **rate** — run concurrently and
coordinate only through the filesystem. Replaces the old phased
iterate/train/eval loop (preserved as `run_loop_phased.py`).

Design rationale and trade-offs are tracked in `cont_train.todo`. This
document is the **operator's reference**: how to run it, what each knob
does, where logs land, and how to diagnose common problems.

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
# One-time: archive any previous run, build C++ binaries, seed
# models/accepted/v000000000.onnx + accepted/latest symlink.
python scripts/run_continuous.py init -y

# Launch the four workers.  Single-GPU (everything on GPU 0) is the default:
python scripts/run_continuous.py run

# 2x4090 recommended split:
python scripts/run_continuous.py run \
  --train-gpus 0,1 \
  --selfplay-gpus 0,1 \
  --gate-gpus 0 \
  --rate-gpus 1 \
  --nn-device-ids-selfplay 0,0,1,1 \
  --nn-device-ids-gate 0 \
  --nn-device-ids-rate 0 \
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
  2. Reads `training/status.json` → current `score_ramp`.
  3. Spawns `build/selfplay` into a fresh staging dir (N games).
  4. Compresses `game_*.bin` → `g_<id>.bin.zst` with monotonic IDs,
     atomic rename.
  5. Prunes the pool to `window_games` files (oldest first).

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
- Train → selfplay: `training/status.json` (ramp state).

All writes are atomic (`*.tmp` → `os.rename`).

---

## Key concepts

### Train bucket (replay-ratio enforcement)

The trainer consumes samples from the rolling window, but only as
fast as selfplay produces them. KataGo-style credit:

```
bucket          = current credit in samples (rows)
max_samples     = bucket_cap_mult * batch_size * world_size   (default 64 * 1024 * W)
watermark_id    = max filename ID already credited

Scanner (rank 0, every ~5s):
  new_rows = sum of row counts of files with id > watermark_id
  bucket += (replay_target * new_rows) // N_AUGMENTATIONS    (clamped at max_samples)
  watermark_id = max id seen this pass

Trainer:
  if bucket < batch_size * world_size: sleep; continue
  train_step(...)
  bucket -= batch_size * world_size
```

`replay_target = 4` means each unique position is visited ~4 times
across augmented views (KataGo's semantics). The `//N_AUGMENTATIONS`
divisor accounts for the C++ writer emitting all 8 dihedral
augmentations per position.

**Why filename-ID-based, not file-index-based**: selfplay prunes old
files once the pool exceeds `window_games`. An index into
`sorted(pool)` shifts when files are pruned; a monotonic filename ID
survives any pruning pattern.

### Per-rank ring buffer

`window_games` governs **disk retention** (enforced by selfplay's
pruner). The trainer's **effective sampling window** is a smaller
in-RAM ring buffer — default `ring_games = 2000` games × ~800 aug_rows
= ~1.6M rows × ~6 KB ≈ **10 GB per rank**. Each rank ingests
independently; when the ring is full, adding a new game's rows evicts
the oldest equal count of rows (FIFO at row granularity).

Freshness: at 1.5 games/s sustained, a row's ring lifetime is ~1.6M /
1200 rows/s ≈ 22 min. Tighter than KataGo's hour-scale shuffle buffer,
but adequate — model strength changes slowly in continuous training.

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

### Ramps (2 knobs, both with phased-pipeline precedent)

Fresh-start safety. Both are **step-driven**, not signal-driven.

- **Value-head weight** ramps linearly from 1.0 at step 0 to 2.0 at
  step `value_ramp_steps` (default 30k). Raw value CE drops from ~0.8
  (random init) to ~0.08 (converged); a fixed weight of 2.0 from step
  0 would make value loss dwarf policy loss and waste early capacity
  on a noise signal.

- **Score ramp** drives two weights from 0 → 1 over
  `score_ramp_steps` (default 50k):
  - `mcts_score_weight(step) = 0.00 + 0.06 * ramp`   (0 → 0.06)
  - `head_score_mean_weight(step) = 0.004 + 0.006 * ramp`  (0.004 → 0.010)

  The score head is untrustworthy on a random-init network; its
  predictions actively distort MCTS search by biasing Q values toward
  meaningless scores. Only `mcts_score_weight` and
  `head_score_mean_weight` ramp — `head_score_stdev_weight` (0.006),
  `head_score_belief_weight` (0.035), and `head_ownership_weight`
  (0.85) stay fixed from step 0 (matching phased precedent).

Rank 0 publishes both factors to `training/status.json` every
`status_publish_every` steps (default 100). Selfplay re-reads this
file at the start of each batch and passes `--score-weight 0.06 *
score_ramp` to `build/selfplay`. This gives selfplay a smooth ramp
signal (minutes-scale updates) rather than multi-hour jumps tied to
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

### TRT engine cache sharing

The TRT cache key is basename-based (see `src/tensorrt_compute.cpp`).
Candidates are named `v{step:09d}.onnx` and preserved across the
`candidates/ → accepted/` rename. Result: the gatekeeper's match warms
the TRT cache; selfplay's next batch loads the plan from disk in <1s.
No 30-120s rebuild tax per promotion.

Requirement: all C++ binaries run from the same CWD (project root)
and use the same `--max-batch`. Both are enforced by the supervisor.

---

## CLI reference

### `scripts/run_continuous.py init`

One-time bootstrap. Archives any existing `training/`, `models/`,
`ratings/`, `logs/` into `*.archive-<ts>/` siblings (no deletion),
builds the C++ binaries if needed, and seeds
`models/accepted/v000000000.onnx` with a random-init ONNX.

```
--arch {resnet,vit}      # default: resnet
--board N                # default: 9
--filters N              # ResNet filters (default: 64)
--blocks N               # ResNet blocks (default: 5)
--d-model N              # ViT (default: 192)
--depth N                # ViT (default: 8)
--heads N                # ViT (default: 6)
--kv-groups N            # ViT GQA (default: 2)
--mlp-ratio N            # ViT (default: 4)
-y, --yes                # skip confirmation
```

### `scripts/run_continuous.py run`

Launches all workers. All CLI flags have sensible defaults; override
only what you need.

**GPU assignment:**
```
--selfplay-gpus 0,1           # CUDA_VISIBLE_DEVICES for selfplay
--train-gpus 0,1              # torchrun --nproc_per_node = list length
--gate-gpus 0
--rate-gpus 1
--nn-device-ids-selfplay 0,0,1,1   # passed to build/selfplay; 0-indexed
                                   #   against the visible set
--nn-device-ids-gate 0
--nn-device-ids-rate 0
```

**Model:**
```
--arch --board --filters --blocks
--d-model --depth --heads --kv-groups --mlp-ratio
--fp8                   # NVIDIA Transformer Engine FP8 (Blackwell+)
```

**Training:**
```
--batch-size 1024          # per DDP rank
--base-lr 3e-4
--warmup-steps 2000
--lr-milestones 100000,400000,1500000
--lr-gamma 0.5
--replay-target 4.0
--n-augmentations 8
--ring-games 2000
--bucket-cap-mult 64
--min-window-games 2000
--min-ring-rows 10240
--export-every 5000
--status-publish-every 100
--log-every 100
--value-ramp-steps 30000
--score-ramp-steps 50000
```

**Selfplay:**
```
--selfplay-batch-games 300    # games per build/selfplay invocation
--selfplay-sims 500
--window-games 80000          # disk retention cap
```

**Gatekeeper:**
```
--gate-games 200
--gate-sims 150
--gate-threshold 0.5
--gate-poll-interval 30       # seconds
```

**Rating (optional):**
```
--rating                      # off by default
--rating-games 80             # games per pair
--rating-sims 200
--rating-pool-size 5          # most-recent accepted
--rating-interval 7200        # seconds between rounds
```

**MCTS / game (shared):**
```
--c-puct 1.25
--dirichlet-alpha 0.15
--dirichlet-epsilon 0.22
--temp-threshold 12
--komi 7.5
--score-scale 18.0
--max-batch 256
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
└── status.json                       # rank 0 publishes ramp state here
                                      #   every STATUS_PUBLISH_EVERY steps

models/
├── candidates/                       # trainer writes here
│   └── v000005000.onnx
├── accepted/                         # gatekeeper promotes here
│   ├── v000000000.onnx               # bootstrap
│   ├── v000005000.onnx
│   └── latest -> v000005000.onnx     # atomic symlink, updated on promotion
├── rejected/
│   └── v000002500.onnx
└── trt_cache/                        # shared between selfplay + evaluate

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

*.archive-<ts>/                       # previous run artifacts after init
```

---

## Logging and diagnostics

### Structured CSVs

**`train_metrics.csv`** (aggregated per 100 steps by default):
```
step, wall_time, samples_seen, lr, value_weight, score_weight_mcts,
score_mean_weight, loss_total, loss_policy, loss_value, loss_score_mean,
loss_score_stdev, loss_ownership, loss_score_belief, loss_opp_policy,
window_games, window_rows, bucket_samples, bucket_fill_ratio, ring_rows,
steps_per_sec, gpu_mem_mb
```

**`selfplay_batches.csv`**:
```
batch_id, wall_time_start, wall_time_end, model_in_use, games_played,
positions_written, duration_s, selfplay_duration_s, score_weight,
pool_size
```

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
SPAWN            proc=train pid=12345 gpus=0,1
CRASH            proc=... rc=... — followed by RESTART or DISABLED
```

### Diagnostic playbook

| Symptom                               | Where to look                                      | Likely knob                               |
| ------------------------------------- | -------------------------------------------------- | ----------------------------------------- |
| Trainer sleeping often                | `train.log` BUDGET_SLEEP frequency                 | `replay_target` too low / selfplay slow   |
| Trainer bucket saturated              | `train_metrics.csv` `bucket_fill_ratio` near 1     | `bucket_cap_mult` too high; trainer slow  |
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
| `bucket_cap_mult`     | 64      | Don't raise to hide TRT rebuild stalls — fix the cache first.                  |
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

- **Ctrl-C once**: SIGTERM to all workers; they flush logs and exit.
  Supervisor waits up to 10s per worker before SIGKILL.
- **Ctrl-C twice**: immediate SIGKILL to everything.

No special teardown needed — all state is on-disk and
restart-friendly.

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

- `cont_train.todo` — full design rationale and trade-offs.
- `COMPARISON_WITH_KATAGO.md` — how this pipeline mirrors (and
  deviates from) KataGo's training code.
- `README.md` — project overview.
- `DEVELOPMENT.md` — build, test, repo layout.
- `run_loop_phased.py` — the previous phased pipeline, preserved for
  reference.
