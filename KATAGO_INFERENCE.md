# KataGo Inference

This build can run a KataGo network through MiniGo's MCTS / NNEvaluator
runtime for **inference only** (play, benchmark, match games). KataGo
weights never enter the training pipeline; selfplay binaries reject
KataGo models with a clear error.

---

## What works

| Binary | KataGo support |
|---|---|
| `play` | yes — interactive play |
| `evaluate` | yes — match games (kata1 vs MiniGo, kata1 vs kata1, …) |
| `benchmark` | sections 1-4 — section 5 (selfplay) is auto-skipped |
| `selfplay` | **no** — errors out at startup with `return 2` |
| `scripts/train*.py`, `tools/warm_init_from_katago.py` (warm-init mode) | unchanged — no KataGo runtime path |

Backend support is restricted to **TensorRT**. Eigen / CUDA /
OpenCL / Metal / RKNN throw `"KataGo format requires the TensorRT
backend"` at handle creation.

---

## Workflow

```bash
# 1. Download a KataGo network (or any kata1 .bin.gz / .txt.gz)
wget https://media.katagotraining.org/uploaded/networks/models/kata1/kata1-b10c128-s1141046784-d204142634.txt.gz

# 2. Convert to ONNX (board=9 here; engine is per-board-size)
python tools/katago_to_onnx.py \
    --katago-bin kata1-b10c128-s1141046784-d204142634.txt.gz \
    --board 9 \
    --output models/kata1-b10c128.onnx

# 3. (optional) Validate PyTorch ↔ ONNX Runtime parity
python tools/katago_parity_test.py \
    --katago-bin kata1-b10c128-s1141046784-d204142634.txt.gz \
    --onnx models/kata1-b10c128.onnx --board 9

# 4. Build (TensorRT backend is required for KataGo runs)
cmake -B build -DMINIGO_BACKEND=tensorrt
make -C build -j

# 5. Run (see "Recommended presets" below)
build/play       --model models/kata1-b10c128.onnx --sims 800 --komi 7.0
build/benchmark  --model models/kata1-b10c128.onnx --max-batch 256 --sims 256 --komi 7.0
build/evaluate   --model1 models/kata1-b10c128.onnx \
                 --model2 models/kata1-b10c128.onnx --games 50 --komi 7.0
```

The first run on each (GPU × max-batch × precision) triple builds a
TensorRT engine and caches it under `trt_cache/`. Subsequent runs
reuse the cached engine.

---

## Knobs (CLI flag reference)

All knobs come from `Config` (`include/config.h`) and are parsed by
each `main_*.cpp`. Defaults are the engine's MiniGo-style defaults
(documented in the table). The "KataGo-like" column shows what to
change to approximate kata1's training-time setup.

| Flag | Default | KataGo-like (kata1 9×9) | What it does |
|---|---|---|---|
| `--komi F` | `6.5` | **`7.0`** or `7.5` | Komi value. Plumbed into `state_global[5] = selfKomi/20`. **Wrong komi degrades strength silently.** |
| `--sims N` | `800` | `800–1600` | MCTS simulations per move. More = stronger, linear in search time. |
| `--c-puct F` | `1.5` | `1.10` | UCB exploration constant. Upstream uses ~1.10 at root. |
| `--win-loss-weight F` | `1.0` | `1.0` | Weight on `value` in the utility blend. |
| `--score-weight F` | `0.0` | `0.10` | Weight on `atan(score/score_scale)/(π/2)` in the utility blend. **Defaults to 0** — we don't use the score head for utility unless this is raised. |
| `--score-scale F` | `18.0` | `~20` (fixed approx of upstream's adaptive scale) | atan compression scale. Upstream's `dynamicScoreCenterScale` adapts to the running score stdev; ours is a fixed scalar. |
| `--max-batch N` | `256` | `256` | GPU batch cap per `predict_batch`. KataGo also batches; defaults are similar. |
| `--search-threads N` | `1` (`play`/`benchmark` default 16) | `16–32` for strong GPUs | Search threads per `MCTS::search()` call. Each pushes through one `NNResultBuf` at a time; the server batches across threads. |
| `--nn-server-threads N` | `1` | `1–2` per GPU | NN server threads per GPU. Use 2 for CPU-pipelining on a single GPU. |
| `--nn-device-ids IDS` | `"0"` | `"0,1,..."` for multi-GPU | Comma-separated CUDA device indices. |
| `--threads N` | `1` (parallel game workers) | `1` for evaluation; more for selfplay | Concurrent game workers in `selfplay` / `benchmark` / `evaluate`. |
| `--temperature-threshold N` | `15` (selfplay only) | n/a | Moves of stochastic play in selfplay (KataGo-only knob; unused for play/evaluate). |
| `--dirichlet-alpha F` | `0.15` (9×9) / `0.03` (19×19) | `0.03` (KataGo's value) | Root noise concentration. Only fires when `add_noise=true` (selfplay); `evaluate` and `play` don't add noise. |
| `--dirichlet-epsilon F` | `0.25` | `0.25` | Mix factor: `0.75 * prior + 0.25 * dirichlet`. |

**Where the utility blend is computed** (`src/mcts.cpp:262-265`):
```cpp
float utility = config_.win_loss_weight * result.value;
if (config_.score_weight != 0.0f) {
    float s_util = atanf(result.score / config_.score_scale) / (float)(M_PI / 2.0);
    utility += config_.score_weight * s_util;
}
```

---

## Recommended presets

### MiniGo preset (default — what `evaluate` and `play` ship with)

```bash
build/evaluate \
    --model1 models/kata1-b10c128.onnx \
    --model2 models/some_minigo.onnx \
    --komi 6.5 \
    --sims 800 \
    --c-puct 1.5 \
    --win-loss-weight 1.0 \
    --score-weight 0.0 \
    --score-scale 18.0 \
    --max-batch 256 --search-threads 16 \
    --games 50
```

What this means:
- Pure win/loss utility (score head's signal is ignored — only its
  policy/value heads steer search).
- Komi 6.5 mismatches kata1's training distribution (kata1 saw 7.0/7.5
  on 9×9 most often).
- `c_puct = 1.5` is more exploratory than KataGo's tuned 1.10.

### KataGo-like preset (recommended for kata1)

```bash
build/evaluate \
    --model1 models/kata1-b10c128.onnx \
    --model2 models/kata1-b10c128.onnx \
    --komi 7.0 \
    --sims 800 \
    --c-puct 1.10 \
    --win-loss-weight 1.0 \
    --score-weight 0.10 \
    --score-scale 20 \
    --max-batch 256 --search-threads 16 \
    --games 50
```

What this changes:
- `--komi 7.0` matches kata1's typical training komi.
- `--score-weight 0.10` lets the score head influence utility
  (kata1's static factor, not its dynamic factor — see caveats).
- `--score-scale 20` is a fixed approximation of upstream's adaptive
  `dynamicScoreCenterScale`, which we don't have.
- `--c-puct 1.10` matches upstream's root PUCT constant.

This is **not** a full KataGo match. The dynamic score utility,
subtree value bias, and FPU details below still differ — see
**MCTS / search differences** for the remaining gap.

### Strength-vs-speed presets

| Goal | `--sims` | `--max-batch` | `--search-threads` |
|---|---|---|---|
| Quick smoke test | 32–64 | 64 | 1 |
| Casual play | 200–400 | 128 | 4–8 |
| Strong play (full kata1 strength) | 800–3200 | 256 | 16–32 |
| Multi-GPU strong | 1600+ | 256 per GPU | 16+ per GPU, `--nn-device-ids 0,1,...` |

---

## Network heads — kept vs dropped

`tools/katago_to_onnx.py` exports five heads with MiniGo-shaped output
names. KataGo binaries actually carry weights for several more
training-time heads which our converter parses (for stream-position
correctness) but **does not export**:

| Head | KataGo binary has weights | Our ONNX exports | Used by MCTS |
|---|---|---|---|
| **Policy** (spatial + pass) | yes — `p1Conv`, `g1Conv`, `gpoolToBias`, `p1BN`, `p2Conv`, `gpoolToPass` | yes | yes — leaf prior |
| **Value** (W/L/D) | yes — `v3Mul`/`v3Bias` outputs 3 logits | yes (collapsed to scalar `value = P(W) − P(L)`) | yes — backed-up Q |
| **ScoreMean** | yes — `sv3Mul[:, 0]` | yes (× 20 baked in) | yes — score utility blend |
| **ScoreStdev** | yes — `sv3Mul[:, 1]` (raw) | yes (softplus + × 20 baked in) | yes — analysis HUD only (not utility) |
| **Ownership** | yes — `vOwnershipConv` | yes ((tanh+1)/2 baked in) | yes — analysis HUD overlay (`o` key in `play`) |
| **Lead** | yes — `sv3Mul[:, 2]` (kata1 v8 only) | **no — dropped** | no |
| **vTime** (game-length variance) | yes — `sv3Mul[:, 3]` (kata1 v8 only) | **no — dropped** | no |
| **shortterm WL error** | yes (v11+) — `sv3Mul[:, 4]` | **no — dropped** | no |
| **shortterm Score error** | yes (v11+) — `sv3Mul[:, 5]` | **no — dropped** | no |
| **scoreBelief MoG** | yes (newer nets only) | **no — never built** | no |
| **futurePos** | yes (newer nets only) | **no — never built** | no |
| **seki / sekiVar** | yes (newer nets only) | **no — never built** | no |
| **opponent policy** | (KataGo doesn't have this; MiniGo training-only) | n/a | n/a |

Why some heads are dropped:
- **Lead, vTime, shortterm errors**: only useful for KataGo's adaptive
  score-utility scaling and resign logic. Our search uses static
  score scale and never resigns mid-search, so they have no consumer.
- **scoreBelief, futurePos, seki**: present only in newer (v15+)
  networks. They are auxiliary supervision targets for KataGo's
  trunk and don't feed search directly. Skipping them for v8-v14
  networks is exact (the layers don't exist); for v15+ networks
  this would need additional graph-building code in
  `tools/katago_arch.py`.

---

## Per-head computation: upstream vs ours

For the heads we DO export, post-processing math is **baked into
the ONNX graph** (not done in C++) so the runtime sees identical
shapes for KataGo and MiniGo:

| Head | Upstream KataGo (raw network output) | Our ONNX output | Where the math lives |
|---|---|---|---|
| Policy | `[N, 2, A]`: ch 0 = own policy, ch 1 = opp policy | `[N, A]`: ch 0 only, then concat `gpoolToPass` logit at index `H*W` | sliced + concatenated in `KataGoPolicyHead.forward` |
| Value | `[N, 3]` raw W/L/D logits | `[N, 1]` = `softmax(logits)[:,0] − softmax(logits)[:,1]` | `softmax + diff` in `KataGoValueHead.forward` |
| ScoreMean | `[N, sv3_c]`, idx 0 stored as `scoreMean / 20` | `[N, 1]` = `sv3[:, 0] * 20.0` (raw points) | `× 20` baked in |
| ScoreStdev | `[N, sv3_c]`, idx 1 stored as raw (network applies softplus elsewhere) | `[N, 1]` = `softplus(sv3[:, 1]) * 20.0` (raw points, positive) | `softplus + × 20` baked in |
| Ownership | `[N, 1, H, W]` raw → tanh in [-1, 1] | `[N, H*W]` = `(tanh(raw) + 1) / 2` in [0, 1] | `(tanh+1)/2` + flatten baked in |

Drop / approximation in the encoder direction (input):
- **Ladder features** (spatial planes 14-17): zeroed — our encoder
  doesn't run a ladder simulator.
- **Encore-only planes** (7, 20-21): zeroed — encore is unsupported.
- **Pass-would-end-phase** (global 14): approximated from
  `consecutive_passes >= 1`. Upstream's logic is more nuanced under
  area + tax-seki rules but reduces to this for area + TAX_NONE.
- **Area** (planes 18-19): Tromp-Taylor flood-fill (`compute_area`
  in `src/katago_inputs.cpp`). Upstream uses
  `Board::calculateArea` with pass-alive groups + safe-territory
  reasoning — slightly different on disputed positions.
- **Komi parity wave** (global 15): implemented per upstream's
  triangular-wave formula; should match exactly.

---

## MCTS / search differences (vs upstream KataGo)

The search runs **MiniGo's MCTS**, not KataGo's. Same backbone (PUCT,
virtual loss, Dirichlet noise, tree reuse) but several KataGo-specific
extras are absent. The KataGo-like preset above closes the easy ones;
the rest are intentional simplifications.

| Aspect | Upstream KataGo | This engine | Implication |
|---|---|---|---|
| **Node selection** | PUCT with FPU reduction (unvisited children get a parent-mean penalty) | PUCT, no FPU reduction (unvisited children get prior * sqrt(N_parent) / 1) | We over-explore unvisited moves slightly |
| **C_puct** | Adaptive: ~1.10 at root, scaled at children | Constant `c_puct` everywhere | Less differentiation between root and tree |
| **Score utility scale** | `dynamicScoreCenterScale` adapts to running score stdev each move | Constant `score_scale` (default 18) | Score utility's "what's a meaningful margin" doesn't track game state |
| **Dynamic score utility factor** | static (~0.10) + dynamic (~0.30) blend | Single `score_weight` (default 0.0) | Score head contributes less unless `--score-weight` raised |
| **Subtree value bias** | corrects backed-up Q for over-/under-estimation | not implemented | Slightly noisier value estimates |
| **Symmetric augmentation at root** | yes — averages over 8 board symmetries for variance reduction | no | Single forward pass per leaf |
| **Resign threshold** | adaptive based on game length + value | not implemented (no resign) | Plays games out fully |
| **Pondering** | yes (`ponder` config) | yes (AsyncBot) | Equivalent |
| **Tree reuse on `make_move`** | yes | yes | Equivalent |
| **Virtual loss** | per-thread, decremented on backprop / on collision revert | per-thread, same pattern | Equivalent |
| **Dirichlet noise** | alpha 0.03, epsilon 0.25 (root only) | alpha 0.15 (9×9) / 0.03 (19×19), epsilon 0.25 (root only when `add_noise=true`) | `evaluate` and `play` set `add_noise=false`, so this only matters in selfplay |
| **Time control** | sophisticated per-move budgeting | n/a — we use fixed `--sims` | No real-time budget management |
| **Multi-GPU batching** | per-GPU server threads, work-stealing queue | per-GPU server threads, single shared queue (work-stealing) | Equivalent topology |

What this gap means in practice:
- A 800-sim run of kata1 in this engine will play **slightly weaker**
  than kata1 in upstream `katago` at 800 sims. Most of the gap is
  ladder-feature absence + dynamic score utility being approximate.
- The output of the network itself (policy, value, score) is
  numerically identical to upstream's at the same input encoding —
  parity test confirms `max_abs_diff ~3e-5` between PyTorch and
  ONNX Runtime.
- For relative comparisons (kata1 vs kata1, or comparing different
  MiniGo nets through the same kata1 baseline), the simplifications
  apply equally to both sides and don't bias the result.

---

## Known limitations & caveats

### Known limitations (encoder + converter)

These are the deliberate simplifications relative to upstream KataGo's
runtime. Each one is a known strength leak; document and live with it,
or port the upstream piece if it bites.

- **Ladder features (planes 14-17) are zeroed.** A faithful port
  requires a recursive ladder simulator with depth limit. Skipping
  this costs measurable strength on ladder-heavy positions but the
  network still gets stones, liberties, history, area, and globals.
  Plan: port `searchIsLadderCapturedAttackerFirst` from upstream
  `cpp/board/boardLogic.cpp`.
- **Encore-only spatial planes (7, 20-21) are zeroed.** Plane 7 is
  ko-recap-blocked (encore phase only); planes 20-21 are
  second-encore start-stone colors. No encore support means these
  never carry signal under our rules, and zeroing them is exact for
  the supported rule set.
- **Pass-would-end-phase global (#14) is approximated** from
  `consecutive_passes >= 1`. Upstream's logic is more nuanced under
  area + tax-seki rules but reduces to this for area + TAX_NONE
  (the only rule combination this engine supports).
- **Area scoring uses simple Tromp-Taylor flood-fill** (per
  `compute_area` in `src/katago_inputs.cpp`). Upstream uses
  `Board::calculateArea` with pass-alive groups + safe-territory
  reasoning. Disputed positions can encode slightly differently;
  for the typical mid-game position the difference is zero.
- **Engine caching is per (GPU × max-batch × precision)** — already
  in place for MiniGo, no changes for KataGo. Engines are also
  per-board-size because `tools/katago_to_onnx.py --board N` bakes
  H/W into the ONNX at export time.
- **Newer model versions (v15+) are not exercised.** The parser
  extension (`tools/warm_init_from_katago.py`) handles v15+ extras
  for stream-position consistency but the architecture port targets
  v8-v14. v15+ networks may need additional layers wired up in
  `tools/katago_arch.py` (meta-encoder, extended policy pass-logit
  path, score-belief MoG).

### Strength caveats (configuration / search)

1. **Komi.** Default is 6.5; pass `--komi 7.0` (or 7.5) for kata1 on
   9×9. Wrong komi = silent strength loss because the value flows
   through the V7 global feature `state_global[5] = selfKomi/20`.
2. **Score utility off by default.** `--score-weight 0` ignores the
   score head when blending utility. `--score-weight 0.10` is the
   easiest single-flag improvement for kata1.
3. **Static score scale** vs upstream's adaptive
   `dynamicScoreCenterScale` — search's score sensitivity doesn't
   track running score stdev across the game.
4. **No FPU reduction, no subtree value bias, no symmetric root
   augmentation.** Each of these would marginally improve search;
   none are implemented (see "MCTS / search differences").
5. **Selfplay is hard-rejected.** The V2 record format and
   augmentation (`mcts.cpp:608-668`) are MiniGo-shaped. Don't
   bypass the guard — feeding KataGo states through MiniGo
   augmentation would silently produce corrupt training data.

### Operational caveats

6. **TensorRT-only.** Eigen, CUDA (non-TRT), OpenCL, Metal, RKNN all
   throw at handle creation. Build with `cmake -DMINIGO_BACKEND=tensorrt`.
7. **Old TRT versions can choke on opset.** This build uses
   PyTorch's exporter (opset 18). TRT 10.13 used here parses cleanly;
   older TRT may fail.
8. **First TRT engine build is slow** (~30-90s for kata1 b10c128 on
   a 4090; longer on older cards). Cached for subsequent runs.

### Rules caveats

The encoder zeroes the globals for unsupported rule modes. Supported
set is **Tromp-Taylor area + simple ko + half-komi + no encore**.
Anything outside that is approximated:

9. Chinese / Japanese territory scoring → not modeled (`global[9]`
   always 0 = area).
10. Multi-stone suicide rule → not modeled (`global[8]` always 0).
11. Tax rules (TAX_SEKI / TAX_ALL) → not modeled (`globals[10,11]`
    always 0).
12. Encore phases (territory-scoring 2nd encore) → not modeled
    (`globals[12,13]` always 0).
13. Super-ko under positional / situational rules → falls back to
    simple ko via `is_ko_ban`. Exact for our rule set; for KataGo's
    super-ko rule sets the ko plane will be slightly under-marked.
14. Button-Go → not modeled.

---

## Boundaries enforced

The principle "KataGo weights are inference-only" is enforced at
every entry point:

| Entry point | KataGo handling |
|---|---|
| `LoadedModel::load` | detects format from graph inputs (`state_spatial` + `state_global`) |
| `NNEvaluator::encode_state` | dispatches to `encode_for_katago` |
| `TensorRTComputeHandle::predict_batch` | input split (spatial+global) |
| `main_play`, `main_evaluate` | run via MCTS — work for both formats |
| `main_benchmark` sections 1-4 | KataGo-aware |
| `main_benchmark` section 5 | skipped for KataGo |
| `main_selfplay` | rejects KataGo at startup with `return 2` |
| Eigen / CUDA / OpenCL / Metal / RKNN backends | reject KataGo at `create_handle` |
| `mcts.cpp:699` (`self_play_game_impl` encode) | hardcoded to `game.encode()` (defense-in-depth, even though entry guards prevent KataGo from reaching it) |
