# Neural Network & Model Contract

The 7-headed network (three trainable architectures), loss functions
and weights, the MCTS utility formula, and the ONNX model contract.
Format support per backend is tracked in [FORMATS.md](FORMATS.md).

### Neural network architecture

KataGo-style 7-headed design.  Two architectures (ResNet, ViT) share the
same head structure.  5 heads are exported to ONNX for inference; 2 are
training-only auxiliaries that shape the trunk's internal representations.

**Inference heads (in ONNX):**

| Head | Shape | Activation | Loss | Target |
|------|-------|------------|------|--------|
| Policy | `[B, action_size]` | (logits) | soft CE | MCTS visit distribution |
| Value | `[B, 3]` → `[B, 1]` | softmax → P(W)-P(L) | CE | one-hot {win=0, loss=1, draw=2} |
| ScoreMean | `[B, 1]` | none | MSE in points² | actual game score (signed, current player) |
| ScoreStdev | `[B, 1]` | softplus | MSE in points² | `\|actual − pred_mean.detach()\|` |
| Ownership | `[B, board²]` | sigmoid | mean BCE | per-intersection {0, 1} from Tromp-Taylor scoring |

**Training-only heads (in .pt checkpoint, stripped from ONNX):**

| Head | Shape | Activation | Loss | Target |
|------|-------|------------|------|--------|
| Score Belief | `[B, num_bins]` | softmax | soft CE | Gaussian centered on score, σ=3 (computed from score, not stored) |
| Opponent Policy | `[B, action_size]` | (logits) | CE | next move from trajectory (masked if `-1`) |

**Loss weights and target contributions**

Weights are calibrated toward KataGo's proportions: policy dominant,
value strong secondary, ownership moderate, score small.  In the
continuous pipeline two weights ramp **by training step** (not by
phase): `value_weight` ramps `1.0 → 2.0` over `--value-ramp-steps`
(default 30k) because raw value CE shrinks as the model converges, and
`score_mean_weight` ramps `0.004 → 0.008` over `--score-ramp-steps`
(default 50k) so the noisy early score signal doesn't distort the trunk.

| Head | Weight (continuous defaults) | Loss function |
|------|------------------------------|---------------|
| Policy | 1.0 | soft CE |
| Value | 1.0 → 2.0 (step ramp) | CE (W/L/D) |
| Ownership | 0.85 | BCE mean |
| Opponent Policy | 0.1 | CE |
| Score Belief | 0.035 | soft CE (163 bins) |
| ScoreMean | 0.004 → 0.008 (step ramp) | **Huber(δ=12)** |
| ScoreStdev | 0.006 | **Huber(δ=10)** |

**KataGo comparison (loss functions).**  Score losses use the same
Huber formulation as KataGo.  Weights differ because we lack their
additional score-related heads (TD score ×3, lead, scoring — ~5 extra
heads that contribute score gradient through the shared trunk):

| | KataGo | MiniGo |
|---|---|---|
| Score mean | Huber(δ=12) | **Huber(δ=12)** (same) |
| Score stdev | Huber(δ=10) | **Huber(δ=10)** (same) |
| Score belief | CDF MSE + PDF CE, weight 0.04 total | soft CE, weight **0.035** |
| scoreMean weight | `0.0015` | `0.004 → 0.008` (higher to compensate for lacking TD/lead) |
| scoreStdev weight | `0.001` | `0.006` |
| Value weight | `1.20` | `1.0 → 2.0` (step-ramped) |
| Ownership weight | `1.5` | `0.85` (BCE averaged over 81 points is already dense signal) |

**MCTS utility formula:**
```
utility = win_loss_weight × (P(win) - P(loss))
        + score_weight × atan(scoreMean / score_scale) / (π/2)
```

| Param | Default | KataGo | Meaning |
|-------|---------|--------|---------|
| `win_loss_weight` | 1.0 | 1.0 | multiplier on P(win)-P(loss) term |
| `score_weight` | 0 → `--score-weight-max` (0.04), ramped by trainer's `score_ramp` | 0.30 (fixed) | how much MCTS values score predictions |
| `score_scale` | 18.0 | `2×√boardArea` = 18 for 9×9 | atan compression: 10pt lead → `atan(10/18)/(π/2) ≈ 0.32` |

The selfplay driver reads `score_ramp` from `training/status.json` each
batch and passes `score_weight = score_weight_max × score_ramp` to the
C++ engine, so search only starts caring about score once the score head
has had gradient steps to become meaningful.  KataGo additionally
integrates score utility over the score distribution
`(scoreMean, scoreStdev)` so uncertain scores are dampened.  We use a
point estimate on `scoreMean` — a reasonable approximation once the model
is trained and stdev is small.  Stdev integration is a future improvement.

**Configurability**

All 7 loss weights and the MCTS weights are flags on
`run_continuous.py run` (forwarded to the trainer / selfplay driver):
`--policy-weight`, `--value-weight-start/-end`,
`--score-mean-weight-start/-end`, `--score-stdev-weight`,
`--ownership-weight`, `--score-belief-weight`, `--opp-policy-weight`,
`--score-weight-max`, `--score-scale`, plus `--value-ramp-steps` /
`--score-ramp-steps` for the ramp lengths.

### Selfplay data format (V3 — engine-neutral game records)

One file per game.  Records store the GAME (moves + per-move MCTS
policy + outcome + final ownership), **not encoded states and no
pre-baked augmentation** — the trainer replays the moves and encodes
positions on the fly for whichever architecture it trains (MiniGo
17-plane or KataGo V7), applying a random dihedral transform per
sample.  That makes one record pool serve all three architectures,
lets ANY model format generate selfplay data (including converted
kata1 nets), and shrinks records ~60×.

```
Header: [magic:u16 'MG'] [version:u16 = 3] [board_size:i32] [komi:f32]
        [n_moves:i32] [winner:i8 0/1/2] [black_score:f32]
Per move: [action:i16  (hw = pass)] [policy: f32×(hw+1)]
Footer:   [owner: i8×hw  (0 empty/dame, 1 black, 2 white)]
```

Parser, replay engine, both encoders, and augmentation live in
`scripts/gamedata.py`; per-position targets (value/score/ownership/
opponent action) are derived at load time.  See `FORMATS.md`.

## Neural Network

Two architectures (`scripts/model.py`), selected with `--arch`:

**ResNet** (default): KataGo-style trunk with alternating residual blocks —
block 0 is an SE-attention block (3x3 → 3x3 → squeeze-excitation → residual),
block 1 is a global-pooling block (parallel 3x3 main + 3x3 pool branches,
where the pool branch is globally mean+max pooled and projected to per-channel
biases that are added into the main branch before the second 3x3), and so on.
Value and score heads use global-pooled heads (1x1 conv → small channel count
→ mean+max pool → MLP) instead of the classic AlphaZero 1x1-to-1-channel +
flattened-FC design, which collapses all channel information before the FC.
**ViT**: Vision Transformer with one token per intersection, GQA, and
directional positional encoding (factorized row/col embedding + signed
relative bias for full spatial and directional awareness).

**Backend support for the new ResNet:** TensorRT and OpenCL implement it
fully (OpenCL also runs the ViT and both KataGo-V7 namings — see the
"OpenCL GPU Backend" section); Eigen runs it on CPU.  CUDA/Metal still
throw at handle creation until their kernels are ported (see `ROADMAP.md`).

Both architectures share 7 output heads (KataGo-style).  4 drive MCTS at
inference time and are exported to ONNX; 3 are training-only auxiliaries
that shape the trunk's internal representations but are never evaluated
during inference (stripped from the `.onnx` file to save model size and
compute).

### Heads used by MCTS (exported to ONNX)

| Head | Model output | ONNX output | Training loss |
|------|-------------|-------------|---------------|
| **Policy** | `[B, 82]` logits | `policy_logits [B, 82]` | CE with soft MCTS visit distribution |
| **Value** | `[B, 3]` logits (W/L/D) | `value [B, 1]` = P(win)−P(loss) | CE over {win, loss, draw}, weight 1.5 |
| **ScoreMean** | `[B, 1]` raw float | `score_mean [B, 1]` (points) | MSE vs actual game score, weight 0.5 |
| **ScoreStdev** | `[B, 1]` softplus | `score_stdev [B, 1]` (points) | MSE vs |actual−predicted|, weight 0.5 |

The value head predicts a 3-class distribution: P(win), P(loss), P(draw).
The ONNX export appends `softmax → P(win) − P(loss)` post-processing so
C++ receives `value [B, 1]` in [-1, +1] as before.  ScoreMean is a direct
regression of the final score margin (in points, from the current player's
perspective) — replaces the old 163-bin classification, which suffered from
hard one-hot targets and stagnant training loss.  ScoreStdev captures the
model's uncertainty about its score estimate.

### Training-only auxiliary heads (NOT in ONNX)

| Head | Model output | Training loss | Weight |
|------|-------------|---------------|--------|
| **Ownership** | `[B, board²]` sigmoid | BCE per intersection | 1.5 / board² |
| **Score Belief** | `[B, num_bins]` logits | CE with soft Gaussian target (σ≈3) | 0.15 |
| **Opponent Policy** | `[B, action_size]` logits | CE vs opponent's next move | 0.15 |

**Ownership** is the most impactful auxiliary: it gives every trunk block
per-intersection territory supervision ("you own D4 but lost E7"), which
is far richer than the single-scalar game-outcome signal the value head
provides.  Without it the model struggles to learn endgame territory
concepts.  The ownership target comes from the game-end board state
(reusing `GoGame::score()`'s existing flood-fill logic).

**Score Belief** forces the trunk to represent score *uncertainty* — wide
distributions in sharp tactical positions, narrow peaks in settled endgames.
The target is a Gaussian centered on the actual game score, not a hard
one-hot bin.  The resulting features help ScoreMean and Value be more
accurate even though MCTS never reads the belief distribution itself.

**Opponent Policy** teaches threat-awareness by predicting what the
opponent plays next (available from the selfplay trajectory).  Lowest-
impact auxiliary (~10-20 Elo in KataGo ablations) but nearly free.

### MCTS Utility

Each leaf evaluation produces a blended utility that MCTS backpropagates:

```
utility = winLossWeight × value
        + scoreWeight   × atan(scoreMean / scoreScale) / (π/2)
```

- `value = P(win) − P(loss)` from the 3-class value head
- `scoreMean` is the regression output (points, current player's perspective)
- `atan` compression maps ±∞ → [-1, +1], preventing extreme scores from
  dominating the utility
- `winLossWeight` (default 1.0), `scoreWeight` (default 0.02, ramped up in
  later training stages), `scoreScale` (default 10.0) are configurable

The score signal teaches MCTS to prefer moves that **win by more points**,
which prevents aimless play in won positions and teaches the model to close
out games decisively.  Without it, MCTS treats "win by 1" and "win by 30"
identically.

### .pt vs .onnx

The PyTorch checkpoint (`.pt`) contains all 7 heads — needed for training
continuation.  The ONNX file (`.onnx`) contains only the 4 MCTS heads —
this keeps inference fast and the model file small.  The 3 training-only
heads add ~55K params to `.pt` but are stripped from `.onnx`.

### Precision

Auto-detected per GPU — best available precision used for both training
and inference:

| GPU | Training | Inference (TensorRT) |
|---|---|---|
| **Blackwell** (SM 10.0+) | FP8 via Transformer Engine (`te.Linear`, E4M3 fwd / E5M2 bwd) | FP8 (`kFP8` builder flag) |
| **Ampere/Ada** (SM 8.0+) | BF16 (`torch.amp.autocast`) | FP16 |
| **Turing** (SM 7.5) | FP16 + GradScaler | FP16 |
| **Pascal** (SM 6.x) | FP16 + GradScaler | FP16 |
| **CPU / older** | FP32 | N/A |

FP8 training requires `pip install transformer_engine` and `"fp8": true` in
the training plan.  Without it, Blackwell falls back to BF16.  FP8 inference
via TensorRT is automatic (no extra config needed).

**ViT positional encoding** — directional (D4 symmetry via data augmentation):
- *Factorized position*: `row_embed[r] + col_embed[c]` — 9+9=18 learned embeddings, full spatial resolution
- *Directional relative bias*: signed `(dx, dy)` offsets — 289 buckets for 9×9. Each direction is unique
  (north ≠ south ≠ east ≠ west), enabling the model to learn directional attention for captures,
  ladders, and edge awareness
- *GQA*: 6 query heads, 2 KV groups (3 queries share each K/V group)

See the **MCTS Utility** section above for how the score head drives search.
`score_weight` (default 0.02) and `score_scale` (default 10.0) are
configurable via CLI flags.  Set `score_weight` to 0 to disable score utility.

**Komi** (compensation for white) defaults to **6.5** for 9×9 and is configurable
via `--komi` in all executables and `PLAN_KOMI` in the training plan.

## Model Format

All backends use **ONNX** (`.onnx`) as the universal model format.
The project includes a built-in minimal protobuf parser (`onnx_loader.cpp`)
— **no external protobuf library required**.

- `LoadedModel::load()` parses the ONNX protobuf once and pre-fuses BN parameters
- Each `ComputeHandle` uploads these CPU weights to its own GPU buffers

The model has dynamic batch dimensions, so the same `.onnx` file works for
batch sizes 1 through N.  GPU workspace buffers auto-grow to fit the batch.

Generate standalone models of different sizes (for benchmarking):
```bash
cd scripts
python3 export_onnx.py --init --board 9 --filters 64  --blocks 5  --output ../models/small.onnx
python3 export_onnx.py --init --board 9 --filters 128 --blocks 10 --output ../models/large.onnx
python3 export_onnx.py --init --board 9 --filters 256 --blocks 20 --output ../models/xlarge.onnx
```

