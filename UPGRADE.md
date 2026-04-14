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
Four heads drive MCTS inference; three are training-only auxiliaries that
shape the trunk's internal representations.

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
- **Loss weight**: 0.5
- **MCTS role**: optionally scales exploration — `effective_cPUCT =
  cPUCT × max(1.0, sqrt(score_stdev))`.  Can be disabled initially
  (just store in NNOutput for display).
- **ONNX post-processing**: none
- **Change from current**: new head, new NNOutput field

### Heads used for training only (NOT exported to ONNX)

These heads exist only in the PyTorch model (.pt checkpoint).  They provide
auxiliary gradients that force the trunk to learn richer features.  They are
stripped during ONNX export to save model size and inference compute.

**5. Ownership** — `[B, board²]` per-intersection sigmoid
- **Prediction**: probability that each intersection is owned by the
  current player at game end
- **Loss**: binary cross-entropy per intersection
- **Loss weight**: 1.5 / board² (~0.002 per intersection for 9×9; scales
  with board size so the total ownership gradient is ~constant)
- **MCTS role**: none (training-only)
- **Purpose**: gives the trunk per-intersection territory supervision.
  Without it the model only learns "you won by 5 points" with no spatial
  signal about WHERE territory is.  This is the single biggest missing
  signal in the current design.
- **Target source**: compute from game-end board state in selfplay.
  `GoGame::score()` already computes area ownership during scoring — expose
  it as a per-intersection vector.

**6. Score Belief** — `[B, num_bins]` logits → softmax
- **Prediction**: full distribution over possible final scores
- **Loss**: cross-entropy against a **soft Gaussian target** centered on
  the actual game score, σ ≈ 3 points.  NOT a hard one-hot bin.
- **Loss weight**: 0.15
- **MCTS role**: none (training-only)
- **Purpose**: forces the trunk to represent score *uncertainty* — wide
  distribution in sharp positions, narrow in settled endgames.  The same
  features make scoreMean and value more accurate.
- **Bins**: keep `num_bins = board² × 2 + 1 = 163` for 9×9.

**7. Opponent Policy** — `[B, action_size]`
- **Prediction**: opponent's next move
- **Loss**: cross-entropy against the actual opponent move (from trajectory)
- **Loss weight**: 0.15
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
L = 1.0  × policy_CE(soft MCTS targets)
  + 1.5  × value_CE({win, loss, draw})
  + 0.5  × scoreMean_MSE
  + 0.5  × scoreStdev_MSE
  + (1.5 / board²) × ownership_BCE
  + 0.15 × scoreBelief_CE(soft Gaussian targets)
  + 0.15 × opponentPolicy_CE
```

All weights are fixed constants, matching KataGo.  Per-stage overrides
from the training plan apply to `score_weight` (MCTS utility blending)
and `score_weight_loss` (overall score loss scale, multiplied onto the
scoreMean + scoreStdev + scoreBelief terms), NOT to individual head weights.

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
- `forward_inference()` (new): returns only the 4 MCTS heads — called by
  `predict()` and used during ONNX export.  Avoids computing the 3
  training-only heads at inference time.

#### `scripts/export_onnx.py`

- Call `model.forward_inference()` during torch.onnx.export (4 outputs only)
- ONNX outputs:
  - `policy_logits [B, action_size]` — unchanged
  - `value [B, 1]` — post-processed: `softmax(logits)[win] - softmax(logits)[loss]`
  - `score_mean [B, 1]` — raw regression output, no post-processing
  - `score_stdev [B, 1]` — raw output (post-softplus), no post-processing
- Remove the old score-bin post-processing (softmax → matmul → expected value)
- Remove `_embed_state_dict` (no longer needed — TensorRT-only, Eigen disabled)
  or keep it but it's unused

#### `scripts/train.py`

- Update DataLoader to load expanded TrainingRecord (see below)
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

  # Score belief: CE with soft Gaussian target
  bin_centers = torch.arange(num_bins) - board_area  # [-81..+81]
  soft_target = softmax(-0.5 * ((bin_centers - target_score) / sigma)**2)
  score_belief_loss = -(soft_target * F.log_softmax(pred_belief, dim=1)).sum(dim=1).mean()

  # Opponent policy: CE against next move
  opp_policy_loss = F.cross_entropy(pred_opp_policy, target_opp_action)

  # Total
  loss = (1.0 * policy_loss
        + 1.5 * value_loss
        + 0.5 * score_mean_loss
        + 0.5 * score_stdev_loss
        + (1.5 / board_area) * ownership_loss
        + 0.15 * score_belief_loss
        + 0.15 * opp_policy_loss)
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

Add score_stdev field:
```cpp
struct NNOutput {
    std::vector<float> policy;
    float value    = 0.0f;   // P(win) - P(loss), already post-processed
    float score    = 0.0f;   // scoreMean (points)
    float score_sd = 0.0f;   // scoreStdev (uncertainty in points)
};
```

#### `src/tensorrt_compute.cpp`

- Detect 4 output tensors instead of 3 (add score_stdev)
- Tensor identification: score_stdev is the second `[B, 1]` output after
  score_mean.  Use explicit naming in ONNX (`score_mean`, `score_stdev`)
  to avoid ambiguity.
- Unpack `score_sd` per batch element alongside `value` and `score`.

#### Other backends (`cuda_compute.cu`, `opencl_compute.cpp`, `eigen_compute.cpp`, `metal_compute.mm`)

- Already throw at handle creation (disabled since the SE/GPool upgrade).
- Update the TODO comment to mention the new 4-output ONNX format.

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

#### `src/main_play.cpp` — display

Update the HUD to show score ± stdev:
```
WR 65.0%  Score +5.3 ± 2.1  N=1234
| D5   65.0% n=523
| E3   22.1% n=234
```

#### `src/main_play.cpp` — ownership overlay (optional, phase 2)

After displaying the board, optionally overlay ownership with +/- markers
at each intersection (toggled by a new hotkey, e.g. `o`).

#### `src/main_benchmark.cpp`, `src/main_evaluate.cpp`

- Update to handle the 4-output NNOutput struct (score_sd field).
- Evaluate binary: no functional change (uses value/policy for move selection).
- Benchmark binary: add score_sd throughput measurement if desired.

### Training pipeline (`run_loop.py`)

- Pass new loss-related args to `train.py` (ownership weight, belief sigma, etc.)
- No structural change to the plan system — the per-stage `score_weight` and
  `score_weight_loss` overrides continue to work as before.
- `score_weight_loss` now scales the ScoreMean + ScoreStdev + ScoreBelief terms
  collectively (all score-related losses).

---

## Model Size Impact

For large ResNet (128f/10b, 9×9):

| Head | Params (approx) | In ONNX? |
|------|-----------------|----------|
| Policy | ~14K | yes |
| Value (3-class) | ~19K | yes |
| ScoreMean | ~19K | yes |
| ScoreStdev | ~19K | yes |
| Ownership | ~130 (1×1 conv) | **no** |
| Score Belief | ~40K | **no** |
| Opponent Policy | ~14K | **no** |

Total .pt checkpoint: ~3.37M (up from 3.28M, +2.7%)
ONNX for inference: ~3.33M (only 4 MCTS heads, slightly larger than current
3.28M due to value going from 1→3 outputs and adding scoreStdev)

The 3 training-only heads add ~55K params to .pt but are stripped from .onnx.
Inference cost is nearly unchanged — scoreMean is cheaper than the old 163-bin
softmax+matmul, and scoreStdev is a tiny MLP.

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

Detection: reader checks if first 2 bytes == 0x4D47.  If not, fall back to
version 1 parsing (for visualizing old data).

---

## Testing Plan

1. **Python model**: create both ResNet and ViT, verify forward() returns 7
   heads with correct shapes.  Verify forward_inference() returns 4 heads.
   Run fwd+bwd, check all 7 losses compute without error.

2. **ONNX export**: export, verify 4 outputs only (policy, value, score_mean,
   score_stdev).  Load in onnxruntime and check output shapes.

3. **C++ build**: `make -j` in build/ (TensorRT).  Verify compiles clean.

4. **Benchmark**: run `./build/benchmark --model new.onnx` — game engine
   speed (unchanged), NN inference (4 outputs), MCTS (new utility formula).

5. **Selfplay**: run `./build/selfplay --games 4 --sims 200` — check .bin
   files contain ownership + opponent_action.  Read back with visualizer.

6. **Training**: `python train.py --data ... --epochs 1` — verify all 7
   losses log correctly, gradient flows, model improves.

7. **Play UI**: `./build/play --model new.onnx` — verify HUD shows
   `Score +X.X ± Y.Y`, game plays to completion.

8. **End-to-end**: `python run_loop.py init large -y && python run_loop.py
   train --iterations 3` — full pipeline with new format.
