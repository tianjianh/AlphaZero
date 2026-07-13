# Training Strategy — Observations, Mistakes, Forward Plan

This document captures what we learned from the first two continuous-training
runs (Apr 24 → May 4, 2026) so the next run avoids repeating the same
mistakes. It is *not* a tuning guide; it is a postmortem and a watchlist.

For canonical knob descriptions and design rationale see `CONTINUOUS_TRAINING.md`. For
the upstream reference see
[KataGo `SelfplayTraining.md`](https://github.com/lightvector/KataGo/blob/master/SelfplayTraining.md).

---

## What we ran

| | Run 1 (abandoned) | Run 2 (warm-started) |
|---|---|---|
| Hardware | 2× RTX 4090 | 2× RTX 4090 |
| Architecture | resnet 64f / 5b (~0.92 M params) | resnet **128f / 10b** (~6.5 M params) |
| Bootstrap | random init | KataGo `kata1-b10c128` distillation |
| Started | 2026-04-24 17:45 | 2026-04-26 10:24 |
| Ended | 2026-04-26 ~11:20 (≈42 h) | 2026-05-04 (still running at writing) |
| Steps reached | ~175 k | ~524 k |
| Latest accepted | v145000 | v500000 |
| Loss EMA floor | 1.83 | **1.357** at step 255 k |
| Tournament peak gain | ~+250 Elo | ~+1500 Elo (ongoing) |

Run 1 was abandoned because architecture defaults didn't match what `init`
seeded (`init` was given 128f/10b but `run` defaulted to 64f/5b — a real bug
in the script described below).

---

## The core observation in run 2: a fake plateau driven by one head

Loss components across the run, sampled at EXPORT events:

| step | LR | loss_total | loss_policy | loss_value | loss_score_mean |
|---:|---:|---:|---:|---:|---:|
| 100 000 | end of 3e-4 | 1.498 | 0.646 | 0.146 | **9.15** |
| 255 000 (run ATL) | mid 1.5e-4 | **1.357** | 0.582 | 0.118 | 10.30 |
| 400 000 | end of 1.5e-4 | 1.527 | 0.582 | 0.144 | **16.83** |
| 475 000 (post-LR2 floor) | early 7.5e-5 | 1.441 | 0.537 | 0.119 | 17.67 |
| 524 000 (live) | mid 7.5e-5 | 1.493 | **0.546** | 0.131 | **19.26** |

`loss_total` looks like a plateau (1.36 → 1.49). But the components disagree:

- **Policy loss is at an all-time low (0.546).** Still descending — model
  is still learning.
- **Value loss is essentially at its floor (0.12–0.13).** Saturated, normal.
- **`loss_score_mean` *doubled* across the run (9.15 → 19.26).** The score
  head is *getting worse*.

The core position-evaluation heads are healthy. The flat `loss_total` is
**entirely** the score head dragging the sum upward. If capacity were the
bottleneck, *all* component losses would plateau together. They don't.

This is the **score-head feedback loop**, not network exhaustion:

1. `mcts_score_weight=0.06` lets score predictions perturb MCTS Q values.
2. As the network strengthens, selfplay games on 9×9 produce increasingly
   extreme score margins (winners blow out losers).
3. The `score_mean` regression target grows in magnitude.
4. The score head can't track this drifting target → its loss climbs.
5. The (now-noisier) score head bleeds into MCTS values via
   `mcts_score_weight`.
6. Selfplay games get more chaotic → score targets shift more.
7. Repeat.

KataGo's `SelfplayTraining.md` and tuning notes warn about this and
mitigate it via lower auxiliary weights and `head-lr-factor=0.5` (heads
train at half the body LR). We don't have that LR-factor mechanism —
adding it is a **forward-plan code change** (see "Future work" below).

---

## Tournament evidence: gate is letting through marginal candidates

The Bradley-Terry rate tournament (more games per pair than the gate)
disagrees with the gate signal in the post-LR-drop1 regime. Examples:

- **Round 27 (anchored v85k=0):** v110k +157 Elo, v115k +235 Elo. Gate
  said v110k clearly beat v105k and v115k clearly beat v110k. Tournament
  agreed roughly.
- **Round 33 (anchored v110k=0):** v115k +63, v145k +35, v155k +157,
  v165k +93. Tournament says v145k is *weaker* than v115k and v165k is
  *weaker* than v155k — even though both passed the gate.
- **Round 89 (anchored v475k=0):** v480k +79, v485k −1, v490k +14,
  v500k +14. The five most-recent accepted models are clustered within
  ~80 Elo. Each accept buys 5–15 Elo, not 100+.

The 200-game gate at `threshold=0.5` has a standard error of ~3.5%, so
candidates that are genuinely 1–2% stronger reject as often as they
accept. With a halved LR, real strength deltas between consecutive 5k-step
exports shrink, and the gate's signal-to-noise ratio collapses.

KataGo's `SelfplayTraining.md` notes the gatekeeper is *optional* and may
be turned off for production runs (the recommendation is to use Elo
tracking instead). We keep it for debugging, but the threshold is too
tight for the late-LR regime.

---

## Issues identified, ranked by impact

### 1. Score-head feedback loop (high confidence)

Evidence: loss_score_mean doubled while policy/value descended. Tournament
shows the model genuinely improved across the run despite total loss
staying flat.

Mitigations applied to defaults for the next run:
- `--score-weight-max` 0.06 → **0.04** (less score bleed into MCTS)
- `--score-mean-weight-end` 0.010 → **0.008** (less loss saturation)

Future work: implement KataGo's `head-lr-factor=0.5` (heads train at half
the body LR). This is an architectural change in `train_continuous.py`,
not a config tweak.

### 2. Ring buffer too tight for late-stage training (moderate confidence)

The trainer's effective sampling window is `ring_games=2000` × ~800 rows
= 1.6 M rows ≈ **22 minutes of selfplay** at our throughput. KataGo's
shuffle buffer is hour-scale — typically 5–10× larger at our generation
count.

When the model is still improving, ring=2000 is fine because the buffer
naturally contains a strength gradient (oldest games in ring are slightly
weaker than newest). At convergence, *all* games in the buffer are at
"current strength," so the trainer is effectively learning to reproduce
itself — gradient signal vanishes. This explains why diminishing returns
arrived around step 290 k–500 k while components still moved.

`cont_train.todo` lines 1218–1224 explicitly call this out as a designed
compromise: *"adequate because the model changes slowly enough that 22
min of current-strength games is representative."* That assumption stops
being true at convergence.

Mitigation: **host-RAM-bound.** On a 64 GB host, 4 DDP ranks × ~10 GB
ring each is ~40 GB and that's near the ceiling. Increasing `ring_games`
beyond 2000 requires either fewer ranks or more host RAM. Default stays
at 2000; **raise to 4000–6000 if you have ≥128 GB RAM** (the doc's tuning
table — and CONTINUOUS_TRAINING.md line 727 — already recommends this).

### 3. Selfplay sims=500 too low for late stage (moderate confidence)

KataGo's b10c128 async configs use 600–1500 visits per move. We were at
500. At low sims, the MCTS visit distribution is genuinely noisier than
the network's policy output, so the policy training target adds noise
rather than clarifying signal — exactly when the network most needs
clean signal (late training).

Mitigation: `--selfplay-sims` 500 → **600**. Cost: ~20% slower selfplay.
On 4×4070ti the budget should absorb this.

### 4. Init/run architecture divergence (real bug, addressed)

Run 1 was abandoned because:
- `init` was invoked with `--filters 128 --blocks 10` (correct).
- `run` was invoked **without** arch flags, defaulting to `--filters 64
  --blocks 5`. The trainer instantiated a 64f/5b net from random init
  and quietly threw away the 128f/10b bootstrap weights (different
  shapes — not loadable).
- Selfplay first played with the 128f/10b bootstrap as opponent, then
  after the very first GATE_ACCEPT, `accepted/latest` flipped to 64f/5b
  and stayed there.

**Mitigation in this commit:** `add_run_args` and `add_init_args` now
share defaults of `filters=128, blocks=10`. Running either with no arch
flags produces a consistent setup.

**Future work:** stronger fix is a manifest. `init` writes
`models/arch_manifest.json` capturing `{arch, board, filters, blocks,
...}`. `run` reads it and refuses to deviate (or auto-fills missing arch
flags from the manifest). Eliminates the silent-divergence foot-gun
entirely. Tracked separately, not in this PR.

### 5. Bootstrap ONNX is not actually loaded as warm-start (cosmetic but misleading)

`bootstrap_model()` in `run_continuous.py` writes a random-init ONNX to
`models/accepted/v000000000.onnx`. The trainer does not load this file —
it builds a fresh `torch.nn.Module` from `--filters/--blocks` flags and
trains from random init. The ONNX is only consumed by selfplay
(populates `accepted/latest` so the file-based pipeline has *something*
to point at before the first EXPORT).

The function name `bootstrap_model` and its log line "(seeded ...)"
suggest it warm-starts training, which it does not.

For run 2 we worked around this by manually placing a KataGo-distilled
ONNX at v000000000 *and* loading the matching torch state-dict into the
trainer's checkpoint at startup. That's why "warm-started" applies to
run 2 — but the script itself doesn't do this; it was a manual step.

**Future work:** if we want true warm-start support, save a torch
state-dict alongside the ONNX in `init` and have the trainer load it on
first run. Not in this PR.

### 6. Gate threshold too tight for late-LR regime (low priority)

200 games × `threshold=0.5` has ~3.5% std error. At LR=7.5e-5, real
strength deltas between consecutive 5k-step exports are often <3% — so
genuine improvements reject as often as they accept. KataGo's gatekeeper
defaults are 200 games at 0.5 too, but they typically run with
`USEGATING=0` for production.

Defaults unchanged. If you see ≥4 consecutive rejects with no clear cause,
consider `--gate-games 400` (halves std error to ~2.5%). Cost: 2× longer
matches; matches are ~2 min so still trivial.

---

## Mistakes I (the model assistant) made during run 2

So that the next run is monitored more carefully:

1. **Pattern-matched "looks healthy" instead of investigating components.**
   At step 200k I saw `loss_score_mean` bump 12 → 13.8 and called it
   "post-LR-drop adaptation" without checking whether it was settling or
   diverging. **It was diverging the entire time.** Should have caught it
   that day.

2. **Treated the v220–230k reject cluster as "small-LR noise"** when it
   was actually the first visible symptom of the score-head feedback
   loop. Tournament rounds at the time already showed v215k as a peak;
   that data was on disk; I didn't pull it.

3. **Reassured at every check-in instead of flagging concerns.** Six
   "healthy / let it ride" assessments in a row, then a sudden "actually
   it's plateaued" at step 524k. Diagnostic should have escalated as soon
   as `loss_score_mean` hit 17, or when round 48 showed v245k anomalously
   high (sign of MCTS-budget mismatch and gate noise).

4. **Didn't read `cont_train.todo` carefully enough.** The "22-minute
   ring window" caveat is in the design doc — I should have known that
   was a known-tight knob and watched for its symptoms (data
   monotonicity / self-distillation collapse), not waited for the user
   to ask.

5. **Treated KataGo as a vague reference instead of pulling actual
   numbers.** When questioned, the right response was "let me read
   `synchronous_loop.sh` and `train.py` for canonical defaults," not "I
   think KataGo uses about 1500 sims." Concrete references at every
   recommendation: `python/selfplay/synchronous_loop.sh`,
   `python/train.py`, `cpp/configs/training/selfplay8.cfg`.

The pattern: **I optimized for tone over signal**. The training was
*mechanically* clean (zero crashes), and I let that dominate updates
that should have been about *whether the data was getting better*.

---

## Watchlist for the next run

These are the early signals to watch. If any fire, escalate
immediately — don't write it off as noise.

| Signal | When to escalate |
|---|---|
| `loss_score_mean` rising >20% over a single 25k-step window | Score head losing its grip on the moving target. Drop `--score-weight-max` further (0.04 → 0.03) and/or `--score-mean-weight-end` (0.008 → 0.006). |
| `loss_policy` and `loss_value` both flat for ≥30k steps while `loss_total` rising | True plateau / score-head feedback. Same fix as above. |
| Tournament rate anchored on the most-recent 5 models showing a non-monotonic strength order across ≥2 rounds | MCTS-budget mismatch between gate (150) and rate (200). Consider raising `--gate-sims` to 200 to match. |
| ≥4 consecutive gate rejects with all scores in 0.40–0.49 | Gate is in the noise band — model improving but invisibly. Consider `--gate-games 400`. |
| Per-batch selfplay duration drifting up >20% | Games are getting longer (often a sign of both nets in the gate match playing more carefully because they're closer in strength). Not a problem in itself, but sometimes precedes a plateau. |
| `loss_value` below 0.10 for >50k steps with no improvement in policy | Value head saturated. Normal. Don't act. |
| `bucket_fill_ratio` pinned >0.9 for >2 hours | Trainer falling behind selfplay. Bump `--bucket-cap-mult` or accept lower replay. |
| `bucket_fill_ratio` pinned ~0 for >2 hours | Selfplay falling behind trainer. Either lower `--replay-target` to 3 or shift more GPUs to selfplay. |
| Any `CRASH`/`RESTART` event | Inspect `*.stdio.log`. If recurring (>3 in 10 min) the worker is `DISABLED` and the pipeline is silently broken. |

---

## Parameter changes in this commit

Only **training-related** defaults changed. Per-worker GPU/thread
allocation is unchanged — you'll set those on the command line for the
4×4070Ti box, where the recommended default split is described in
`CONTINUOUS_TRAINING.md`.

| Parameter | Run 2 default | New default | Why |
|---|---|---|---|
| `--filters` | 64 | **128** | Match KataGo b10c128 warm-start. 64 was too small. |
| `--blocks` | 5 | **10** | Match KataGo b10c128. |
| `--batch-size` | 1024 | **256** | Per-rank. With 4 DDP ranks → global = 1024 (matches doc target). |
| `--score-weight-max` | 0.06 | **0.04** | Cut score-head bleed into MCTS Q-values. |
| `--score-mean-weight-end` | 0.010 | **0.008** | Cut score-head loss saturation. |
| `--selfplay-sims` | 500 | **600** | Cleaner policy targets in late-stage training. |
| `--window-games` | 80 000 | **100 000** | Modest disk-pool expansion. Cheap; ~25 GB pool steady-state. |
| `init` arch defaults | 64/5 | **128/10** | Mirrors run defaults — eliminates init/run divergence. |

Unchanged (verified against KataGo canonical configs):

- `--replay-target=4` → KataGo `MAX_TRAIN_PER_DATA=4` in async production
- `--gate-games=200`, `--gate-sims=150`, `--gate-threshold=0.5` →
  KataGo gatekeeper defaults
- `--lr-milestones=100000,400000,1500000`, `--lr-gamma=0.5` → schedule
  worked correctly in run 2 (both drops fired and produced their
  expected step-count behavior)
- `--ring-games=2000` → host-RAM-bound on 64 GB box; bump to 4000+ if
  RAM allows
- `--dirichlet-alpha=0.15`, `--dirichlet-epsilon=0.22`,
  `--score-scale=18.0` → KataGo formulas (10.83/N_legal, 2·√81)

---

## How to launch the next run on 4×4070Ti

Per the user's note: hardware split (selfplay vs train GPUs) is
adjustable later. A reasonable starting split (mirrors the
2×4090 example in `CONTINUOUS_TRAINING.md` doubled out to 4 GPUs):

```bash
python scripts/run_continuous.py init -y    # uses 128/10 defaults now

python scripts/run_continuous.py run \
  --train-gpus 0,1,2,3 \
  --selfplay-gpus 0,1,2,3 \
  --gate-gpus 0 \
  --rate-gpus 1 \
  --selfplay-nn-device-ids 0,0,1,1,2,2,3,3 \
  --gate-nn-device-ids 0 \
  --rate-nn-device-ids 0 \
  --selfplay-nn-server-threads 8 \
  --gate-nn-server-threads 1 \
  --rate-nn-server-threads 1 \
  --max-batch 512 \
  --rating
```

`--batch-size`, `--filters`, `--blocks`, `--ring-games`, `--score-weight-max`,
`--score-mean-weight-end`, `--selfplay-sims`, `--window-games` all use the
new defaults; no need to specify on the CLI unless you want to override.

After the run is up:

1. Verify `loss_score_mean` is *flat or descending* by step 50k. If it's
   rising, the score-head feedback loop isn't fully tamed and you
   should drop `--score-weight-max` to 0.03.
2. Check tournament round 1 (after 5 models accepted) for monotonic
   ascending Elo. If not, the gate-vs-rate sims mismatch is biting and
   you should raise `--gate-sims` to 200.
3. Watch host RAM via `free -h`. If the trainer's per-rank ring is
   stable below ~12 GB and total host usage stays under 80% of physical
   RAM, you have room to bump `--ring-games` to 3000–4000 mid-run (will
   require restart since it's a startup-only knob).

---

## Path A: compressed-in-memory ring (NEW, implemented in this PR)

The single biggest design issue from run 2 — the small (2,000-game)
in-RAM ring forcing self-distillation collapse at convergence — is
addressed by storing the on-disk `.bin.zst` bytes verbatim in the ring
and decompressing lazily at sample time. Implemented in
`scripts/train_continuous.py:WindowRingBuffer`.

**Measured on real selfplay data (300 games sampled):**
- ~23× compression: 200-game ring is **38 MB** in RAM (was ~900 MB raw).
- Allows `--ring-games` at 4–10× larger values within the same RAM
  budget — directly addresses the late-stage convergence collapse.

### Sampling trade-off

Strict row-uniform sampling over a compressed ring is intractable: a
256-row batch from a 2000-game ring would touch ~240 distinct games
per batch (coupon-collector), forcing 240 decompressions per step
(~1.2 s on a single thread, ~750 ms with 8 threads). The ring becomes
the trainer's bottleneck.

Solution: **game-granular sampling**. Pick `K = batch_size /
samples_per_game` games per batch, take `samples_per_game` rows from
each. Same trick KataGo uses inside its npz shuffle blocks.

| `samples_per_game` | Unique games / batch | Time/batch (4 workers) | Within-batch correlation |
|---:|---:|---:|---:|
| 4 | 64 | 175 ms | ~1.5% per game |
| **8 (default)** | **32** | **110 ms** | ~3% per game |
| 16 | 16 | 63 ms | ~6% per game |
| 32 | 8 | 36 ms | ~12% per game |

`samples_per_game=8, ring_decompress_workers=4` is the default. GPU
step on 128f/10b is ~50–200 ms, so per-batch sampling fits inside
forward/backward without slowing training. Tune via
`--samples-per-game` and `--ring-decompress-workers` if needed.

**Future improvement (not in this PR)**: a proper "shuffler" that
maintains a small pool of pre-decompressed rows in RAM, refilled in
the background. Would give true row-uniform sampling at full speed.
See KataGo's `shuffle.py`.

---

## Other future work (not in this PR)

- **Arch manifest** (`models/arch_manifest.json`) written by `init`,
  read by `run`. Eliminates init/run arch divergence permanently.
- **Real warm-start.** `init` saves a torch state_dict alongside the
  ONNX. Trainer loads it on first run. Cleans up the misleading
  "bootstrap" terminology and supports KataGo distillation as a
  first-class path.
- **`head-lr-factor`.** Apply 0.5× LR to head parameters in
  `train_continuous.py`. Matches KataGo and would directly mitigate the
  score-head feedback loop at the optimizer level.
- **Visualizer multi-game support** (cosmetic). The `--game N` flag is
  currently dead code on `.bin` files because each `g_*.bin.zst` holds
  a single game.
- **Gate baseline rolling fallback.** After K consecutive rejects,
  match against `accepted[-2]` instead of `accepted[-1]`.
- **Background shuffler** (improves on Path A): pre-decompresses a
  rolling pool of rows so sample_batch becomes a numpy fancy-index
  on a small in-RAM buffer. Restores strict row-uniform sampling
  without the per-batch decompression cost.
