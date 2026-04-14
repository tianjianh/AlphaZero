# KataGo-Style Multi-Head Upgrade Plan

## Context

The current network has 3 heads (policy, value as tanh regression, score
as 163-bin hard-target classification).  Training stalls in the late stages
because:

1. **Score head uses hard one-hot bin targets** — noisy gradients, loss
   plateaus at ~1.6 nats, the model can't learn accurate score predictions.
2. **No per-intersection ownership signal** — the trunk only gets a single
   scalar per game ("you won by 5 points") with no spatial feedback on
   *which regions* contributed.  The model never learns endgame territory
   concepts.
3. **Value head is a single tanh** — conflates win/loss/draw into one float,
   giving weaker gradient signal than a 3-class distribution.

This upgrade replaces the 3-head design with KataGo's 7-head architecture.
Five heads are exported to ONNX for inference (policy, value, scoreMean,
scoreStdev, ownership); two are training-only auxiliaries that shape the
trunk's internal representations.

## Head Design (7 heads)

### Heads used by MCTS at inference time (exported to ONNX)

**1. Policy** — `[B, action_size]`
- **Prediction**: move probability distribution
- **Loss**: cross-entropy against the MCTS visit distribution (soft targets,
  already implemented)
- **Loss weight**: 1.0
- **MCTS role**: prior probabilities for UCB child selection
- **Change from current**: none

**2. Value** — `[B, 3]` logits → softmax → P(win), P(loss), P(draw)
- **Prediction**: game outcome distribution over 3 classes
- **Loss**: cross-entropy against one-hot {win=0, loss=1, draw=2}
- **Loss weight**: 1.5
- **MCTS role**: `value = P(win) - P(loss)` → [-1, +1]
- **ONNX post-processing**: append `softmax → slice[0] - slice[1]` so C++
  receives `value [B, 1]` as before
- **Change from current**: replace tanh regression `[B, 1]` with 3-class
  softmax.  The C++ NNOutput struct stays the same (still receives a float).

**3. ScoreMean** — `[B, 1]` raw float (no activation)
- **Prediction**: expected final score from current player's perspective
- **Loss**: MSE against actual game score
- **Loss weight**: 0.5
- **MCTS role**: `score_utility = atan(scoreMean / score_scale) / (π/2)`,
  blended into utility as `utility = value + score_weight × score_utility`
- **ONNX post-processing**: none (raw float passed through)
- **Change from current**: replaces the 163-bin classification.  Simpler,
  smoother gradients, no bin discretization noise.

**4. ScoreStdev** — `[B, 1]` raw float (softplus activation for positivity)
- **Prediction**: predicted uncertainty (standard deviation) of the score
- **Loss**: MSE against `|actual_score - predicted_scoreMean|` (running
  target from the model's own scoreMean output)
- **Loss weight**: 0.5 (default)
- **MCTS role**: optionally scales exploration — `effective_cPUCT =
  cPUCT × max(1.0, sqrt(score_stdev))`.  Can be disabled initially
  (just store in NNOutput for display).
- **ONNX post-processing**: none
- **Change from current**: new head, new NNOutput field

**5. Ownership** — `[B, board²]` per-intersection sigmoid
- **Prediction**: probability that each intersection is owned by the
  current player at game end
- **Loss**: binary cross-entropy per intersection
- **Loss weight**: 1.5 / board² (default; ~0.002 per intersection for 9×9;
  scales with board size so the total ownership gradient is ~constant)
- **MCTS role**: territory/ownership overlay in play UI
- **ONNX post-processing**: none (raw sigmoid output passed through)
- **Purpose**: gives the trunk per-intersection territory supervision.
  Without it the model only learns "you won by 5 points" with no spatial
  signal about WHERE territory is.  This is the single biggest missing
  signal in the current design.
- **Target source**: compute from game-end board state in selfplay.
  `GoGame::score()` already computes area ownership during scoring — expose
  it as a per-intersection vector.
- **Change from current**: new head, new NNOutput field.  Exported to ONNX
  so the play UI can show a live territory overlay.

### Heads used for training only (NOT exported to ONNX)

These heads exist only in the PyTorch model (.pt checkpoint).  They provide
auxiliary gradients that force the trunk to learn richer features.  They are
stripped during ONNX export to save model size and inference compute.

**6. Score Belief** — `[B, num_bins]` logits → softmax
- **Prediction**: full distribution over possible final scores
- **Loss**: cross-entropy against a **soft Gaussian target** centered on
  the actual game score, σ ≈ 3 points.  NOT a hard one-hot bin.
- **Loss weight**: 0.15 (default)
- **MCTS role**: none (training-only)
- **Purpose**: forces the trunk to represent score *uncertainty* — wide
  distribution in sharp positions, narrow in settled endgames.  The same
  features make scoreMean and value more accurate.
- **Bins**: keep `num_bins = board² × 2 + 1 = 163` for 9×9.
- **No record storage needed**: the soft Gaussian target is computed
  on-the-fly during training from the `score` field already in the
  TrainingRecord: `softmax(-0.5 * ((bin_centers - score) / σ)²)`.

**7. Opponent Policy** — `[B, action_size]`
- **Prediction**: opponent's next move
- **Loss**: cross-entropy against the actual opponent move (from trajectory)
- **Loss weight**: 0.15 (default)
- **MCTS role**: none (training-only, lowest-impact auxiliary)
- **Purpose**: teaches the trunk threat-awareness features.  Opponent's move
  is available in the selfplay trajectory — just store the next move alongside
  each training sample.

## MCTS Utility Formula

```
utility = winLossWeight × value
        + scoreWeight   × atan(scoreMean / scoreScale) / (π/2)
```

Where:
- `value = P(win) - P(loss)` from the 3-class value head
- `scoreMean` is the direct regression output
- `winLossWeight` (default 1.0), `scoreWeight`, `scoreScale` are config params

Optional (phase 2): scale exploration constant by score uncertainty:
```
effective_cPUCT = cPUCT × max(1.0, sqrt(scoreStdev / baseline))
```

## Loss Function (total)

```
L = policy_weight      × policy_CE(soft MCTS targets)
  + value_weight       × value_CE({win, loss, draw})
  + score_mean_weight  × scoreMean_MSE
  + score_stdev_weight × scoreStdev_MSE
  + ownership_weight   × ownership_BCE        (per-intersection, NOT /board²)
  + score_belief_weight × scoreBelief_CE(soft Gaussian targets)
  + opp_policy_weight  × opponentPolicy_CE
```

**All 7 weights are individually configurable** via CLI arguments in
`train.py`, plan.json `training` defaults, and per-stage overrides in
`run_loop.py`.  Defaults match KataGo's ratios:

| Weight | CLI flag | Default | plan.json key |
|--------|----------|---------|---------------|
| `policy_weight` | `--policy-weight` | 1.0 | `policy_weight` |
| `value_weight` | `--value-weight` | 1.5 | `value_weight` |
| `score_mean_weight` | `--score-mean-weight` | 0.5 | `score_mean_weight` |
| `score_stdev_weight` | `--score-stdev-weight` | 0.5 | `score_stdev_weight` |
| `ownership_weight` | `--ownership-weight` | 0.02 | `ownership_weight` |
| `score_belief_weight` | `--score-belief-weight` | 0.15 | `score_belief_weight` |
| `opp_policy_weight` | `--opp-policy-weight` | 0.15 | `opp_policy_weight` |

Note: `ownership_weight` default 0.02 ≈ 1.5/81 for 9×9.  Unlike the
other weights this one is applied as-is (not divided by board² at runtime),
so adjust it when changing board size.

**MCTS utility weights** are also individually configurable via CLI in all
C++ binaries (`selfplay`, `evaluate`, `play`, `benchmark`) and in
plan.json `mcts` defaults / per-stage overrides:

| Weight | CLI flag | Default | plan.json key |
|--------|----------|---------|---------------|
| `win_loss_weight` | `--win-loss-weight` | 1.0 | `win_loss_weight` |
| `score_weight` | `--score-weight` | 0.0 | `score_weight` |
| `score_scale` | `--score-scale` | 10.0 | `score_scale` |

---

## File-by-File Change Plan

### Python (model + training + export)

#### `scripts/model.py`

**Both AlphaZeroNet and GoViT:**

- Remove the old score bin head (GPoolHead with `out_features=num_bins`).
- Add 6 heads (policy already exists):
  - `value_head`: GPoolHead → `[B, 3]` logits (W/L/D)
  - `score_mean_head`: GPoolHead → `[B, 1]`
  - `score_stdev_head`: GPoolHead → `[B, 1]` + softplus
  - `ownership_head`: `Conv2d(trunk, 1, 1)` → per-intersection sigmoid
  - `score_belief_head`: GPoolHead → `[B, num_bins]` logits
  - `opponent_policy_head`: same arch as policy (Conv2d → BN → FC)

- `forward()` returns 7 outputs: `(policy, value, score_mean, score_stdev,
  ownership, score_belief, opponent_policy)`
- `forward_inference()` (new): returns the 5 inference heads — called by
  `predict()` and used during ONNX export.  `(policy, value, score_mean,
  score_stdev, ownership)`.  Avoids computing the 2 training-only heads
  (score_belief, opponent_policy) at inference time.

#### `scripts/export_onnx.py`

- Call `model.forward_inference()` during torch.onnx.export (5 outputs)
- ONNX outputs:
  - `policy_logits [B, action_size]` — unchanged
  - `value [B, 1]` — post-processed: `softmax(logits)[win] - softmax(logits)[loss]`
  - `score_mean [B, 1]` — raw regression output, no post-processing
  - `score_stdev [B, 1]` — raw output (post-softplus), no post-processing
  - `ownership [B, board²]` — per-intersection sigmoid, no post-processing
- Remove the old score-bin post-processing (softmax → matmul → expected value)
- Remove `_embed_state_dict` (no longer needed — TensorRT-only, Eigen disabled)
  or keep it but it's unused

#### `scripts/train.py`

- Update DataLoader to read V2 binary format directly (see below).
  **No V1 fallback** — the dataloader only accepts V2 files with the
  `0x4D47` magic header.  Old V1 data must be regenerated.
- Add CLI arguments for all 7 loss weights (see Loss Function table above)
- New loss computation:
  ```python
  # Value: 3-class CE
  value_target = ...  # 0=win, 1=loss, 2=draw
  value_loss = F.cross_entropy(pred_value_logits, value_target)

  # ScoreMean: MSE regression
  score_mean_loss = F.mse_loss(pred_score_mean, target_score)

  # ScoreStdev: MSE against |actual - predicted_mean|
  with torch.no_grad():
      stdev_target = (target_score - pred_score_mean.detach()).abs()
  score_stdev_loss = F.mse_loss(pred_score_stdev, stdev_target)

  # Ownership: per-intersection BCE
  ownership_loss = F.binary_cross_entropy_with_logits(
      pred_ownership, target_ownership)

  # Score belief: CE with soft Gaussian target (computed from score,
  # not stored in records)
  bin_centers = torch.arange(num_bins) - board_area  # [-81..+81]
  soft_target = softmax(-0.5 * ((bin_centers - target_score) / sigma)**2)
  score_belief_loss = -(soft_target * F.log_softmax(pred_belief, dim=1)).sum(dim=1).mean()

  # Opponent policy: CE against next move
  opp_policy_loss = F.cross_entropy(pred_opp_policy, target_opp_action)

  # Total (all weights from CLI args)
  loss = (args.policy_weight      * policy_loss
        + args.value_weight       * value_loss
        + args.score_mean_weight  * score_mean_loss
        + args.score_stdev_weight * score_stdev_loss
        + args.ownership_weight   * ownership_loss
        + args.score_belief_weight * score_belief_loss
        + args.opp_policy_weight  * opp_policy_loss)
  ```
- Log all 7 component losses per epoch

#### `scripts/visualize.py`

- Update to read new binary format (version 2)
- Display ownership per move (show +/- per intersection on the board)
- Display value as W/L/D percentages
- Display score with ± stdev: `Score +5.3 ± 2.1`

### C++ (selfplay, inference, MCTS, display)

#### `include/mcts.h` — TrainingRecord

Expand the struct:
```cpp
struct TrainingRecord {
    std::vector<float> state;          // [input_channels * board²]
    std::vector<float> policy;         // [action_size]
    float value;                       // {-1, 0, 1}
    float score;                       // points, current player's perspective
    std::vector<float> ownership;      // [board²] — 1.0 = current player owns
    int   opponent_action;             // opponent's next move (-1 if last move)
};
```

#### `src/mcts.cpp` — self_play_game_impl

- Compute ownership targets from `GoGame::score()` result.  Need a new
  helper `GoGame::get_ownership(Stone player, float* out)` that fills a
  board²-sized float array with 1.0 for player-owned intersections, 0.0
  otherwise (using the existing flood-fill scoring logic).
- Store opponent's next move from the trajectory (look-ahead by one step).
- Last move of the game has `opponent_action = -1` (no next move).

#### `src/main_selfplay.cpp` — write_records

- Bump binary format version.  New header: `[magic: 0x4D47 (MG)] [version: 2]`
  followed by `[count: i32]`.
- Per record: append `[ownership: f32 × board²] [opponent_action: i32]`
  after the existing `[value] [score]`.

#### `src/game.h / game.cpp` — ownership helper

Add public method:
```cpp
void get_ownership(Stone player, std::vector<float>& out) const;
```
Reuses the existing `score()` flood-fill logic but outputs a per-intersection
float vector instead of aggregate counts.

#### `include/compute_context.h` — NNOutput

Add score_stdev and ownership fields:
```cpp
struct NNOutput {
    std::vector<float> policy;
    float value    = 0.0f;              // P(win) - P(loss), already post-processed
    float score    = 0.0f;              // scoreMean (points)
    float score_sd = 0.0f;             // scoreStdev (uncertainty in points)
    std::vector<float> ownership;       // [board²] per-intersection ownership
};
```

#### `src/tensorrt_compute.cpp`

- Detect 5 output tensors instead of 3 (add score_stdev, ownership)
- Tensor identification by ONNX name: `policy_logits`, `value`,
  `score_mean`, `score_stdev`, `ownership`.
- Unpack `score_sd` per batch element alongside `value` and `score`.
- Unpack `ownership [board²]` per batch element into `NNOutput::ownership`.

#### Other backends (`cuda_compute.cu`, `opencl_compute.cpp`, `eigen_compute.cpp`, `metal_compute.mm`)

- Already throw at handle creation (disabled since the SE/GPool upgrade).
- Update the TODO comment to mention the new 5-output ONNX format.

#### `src/mcts.cpp` — utility computation

Update the expand + backprop path:
```cpp
float utility = config_.win_loss_weight * result.value;  // P(win)-P(loss)
if (config_.score_weight != 0.0f) {
    float score_utility = atanf(result.score / config_.score_scale) / (M_PI_2);
    utility += config_.score_weight * score_utility;
}
```

Add `win_loss_weight` to Config (default 1.0) for completeness.

Store `result.score_sd` in the node for analysis display:
```cpp
node->nn_score    = result.score;
node->nn_score_sd = result.score_sd;
```

#### `include/mcts.h` — MCTSNode

Add `float nn_score_sd = 0.0f;` field.

Update `AnalysisInfo`:
```cpp
struct AnalysisInfo {
    std::vector<MoveInfo> moves;
    int   total_visits  = 0;
    float root_utility  = 0.0f;
    float root_score    = 0.0f;
    float root_score_sd = 0.0f;   // new
};
```

#### `include/config.h`

Add:
```cpp
float win_loss_weight = 1.0f;
```

All C++ binaries (`main_selfplay`, `main_evaluate`, `main_play`,
`main_benchmark`) must accept `--win-loss-weight` as a CLI argument,
alongside the existing `--score-weight` and `--score-scale`.

#### `src/main_play.cpp` — display

Update the HUD to show score ± stdev:
```
WR 65.0%  Score +5.3 ± 2.1  N=1234
| D5   65.0% n=523
| E3   22.1% n=234
```

#### `src/main_play.cpp` — ownership overlay

Overlay ownership from the NN's live inference output (not training data).
Show +/- markers at each intersection, toggled by a new hotkey `o`.
The ownership data comes from `NNOutput::ownership` (the 5th ONNX output),
available whenever the NN evaluates a position (during ponder/analysis).

#### `src/main_benchmark.cpp`, `src/main_evaluate.cpp`

- Update to handle the 5-output NNOutput struct (score_sd, ownership).
- Both binaries accept `--win-loss-weight` CLI argument.
- Evaluate binary: no functional change (uses value/policy for move selection).
- Benchmark binary: add score_sd throughput measurement if desired.

### Training pipeline (`run_loop.py`)

- **plan.json `training` defaults**: add all 7 loss weight keys
  (`policy_weight`, `value_weight`, `score_mean_weight`, `score_stdev_weight`,
  `ownership_weight`, `score_belief_weight`, `opp_policy_weight`).
  Replace the old `score_weight_loss` with the individual weights.
- **plan.json `mcts` defaults**: add `win_loss_weight` (default 1.0)
  alongside existing `score_weight` and `score_scale`.
- **Per-stage overrides**: any of the 7 loss weights or 3 MCTS utility
  weights can be overridden per stage in the stages array.
  `get_stage_config()` merges them as before.
- **Pass to train.py**: all 7 `--*-weight` flags from merged training config.
- **Pass to C++ binaries**: `--win-loss-weight`, `--score-weight`,
  `--score-scale` from merged MCTS config to selfplay, evaluate, play.
- **Exploration schedules** (`_EXPLORE`, `_EXPLORE_LARGE`, `_EXPLORE_XLARGE`):
  update tuples to include all weight progressions per stage.

---

## Model Size Impact

For large ResNet (128f/10b, 9×9):

| Head | Params (approx) | In ONNX? |
|------|-----------------|----------|
| Policy | ~14K | yes |
| Value (3-class) | ~19K | yes |
| ScoreMean | ~19K | yes |
| ScoreStdev | ~19K | yes |
| Ownership | ~130 (1×1 conv) | yes |
| Score Belief | ~40K | **no** |
| Opponent Policy | ~14K | **no** |

Total .pt checkpoint: ~3.37M (up from 3.28M, +2.7%)
ONNX for inference: ~3.33M (5 inference heads, slightly larger than current
3.28M due to value 1→3 outputs, scoreStdev, and ownership 1×1 conv)

The 2 training-only heads add ~54K params to .pt but are stripped from .onnx.
Inference cost is nearly unchanged — ownership is a single 1×1 conv (trivial),
scoreMean is cheaper than the old 163-bin softmax+matmul, and scoreStdev
is a tiny MLP.

---

## Binary Data Format

### Version 1 (current)
```
[count: i32]
for each record:
    [state_size: i32] [state: f32×S]
    [policy_size: i32] [policy: f32×P]
    [value: f32]
    [score: f32]
```

### Version 2 (new)
```
[magic: u16 = 0x4D47 ("MG")]
[version: u16 = 2]
[count: i32]
[board_size: i32]
for each record:
    [state_size: i32] [state: f32×S]
    [policy_size: i32] [policy: f32×P]
    [value: f32]
    [score: f32]
    [ownership: f32 × board²]
    [opponent_action: i32]
```

**train.py DataLoader**: reads V2 directly.  No V1 fallback — expects the
`0x4D47` magic header.  Old V1 selfplay data must be regenerated.

**visualize.py**: keeps V1 fallback (checks first 2 bytes for `0x4D47`;
if absent, parses as V1) so old game files remain viewable.

---

## Testing Plan

1. **Python model**: create both ResNet and ViT, verify forward() returns 7
   heads with correct shapes.  Verify forward_inference() returns 5 heads.
   Run fwd+bwd, check all 7 losses compute without error.

2. **ONNX export**: export, verify 5 outputs (policy, value, score_mean,
   score_stdev, ownership).  Load in onnxruntime and check output shapes.

3. **C++ build**: `make -j` in build/ (TensorRT).  Verify compiles clean.

4. **Benchmark**: run `./build/benchmark --model new.onnx` — game engine
   speed (unchanged), NN inference (5 outputs), MCTS (new utility formula).

5. **Selfplay**: run `./build/selfplay --games 4 --sims 200` — check .bin
   files contain ownership + opponent_action.  Read back with visualizer.

6. **Training**: `python train.py --data ... --epochs 1` — verify all 7
   losses log correctly, gradient flows, model improves.

7. **Play UI**: `./build/play --model new.onnx` — verify HUD shows
   `Score +X.X ± Y.Y`, ownership overlay (`o` key) works, game plays
   to completion.

8. **End-to-end**: `python run_loop.py init large -y && python run_loop.py
   train --iterations 3` — full pipeline with new format.
