# Format Support Matrix

Post-cleanup state: **three first-class model architectures, one
engine-neutral game-record format, two board-state encodings.**  Every
architecture supports both running (selfplay / play / evaluate /
benchmark) and training; the record pool is shared by all of them.
Legacy compatibility (V1/V2 records, `.bin.gz`, KataGo-inference-only
restrictions) has been removed.

## 1. Model architectures (all trainable, all runnable)

| Arch | Input | PyTorch class | ONNX contract | Sized by |
|------|-------|---------------|---------------|----------|
| `resnet` | MiniGo 17-plane `state [B,17,H,W]` | `AlphaZeroNet` | single input, 5 outputs | `--filters/--blocks` |
| `vit` | MiniGo 17-plane `state [B,17,H,W]` | `GoViT` | single input, 5 outputs | `--d-model/--depth/…` |
| `katago` | KataGo V7 `state_spatial [B,22,H,W]` + `state_global [B,19]` | `KataGoNet` | dual input, 5 outputs | `--filters/--blocks` (= channels/blocks) |

- All three share the **same 7-head training contract**
  (policy, value-WDL, scoreMean, scoreStdev, ownership + training-only
  scoreBelief, oppPolicy) and the same 5-output inference ONNX
  (`policy_logits, value, score_mean, score_stdev, ownership`).
- `model.input_kind` (`"single"`/`"dual"`) is the dispatch point for
  encoding and forward-call arity everywhere (trainer, exporter).
- The C++ loader auto-detects the format from the ONNX input names
  (`state_spatial` present → `ModelFormat::KataGo`); a **converted
  stock kata1 network** (`tools/katago_to_onnx.py`) satisfies the same
  dual-input contract and is a drop-in for every binary — including
  selfplay, since V3 records are encoding-free.  Stock kata1
  *checkpoints* are not resumable by the trainer (their heads differ);
  they run as-is or warm-init a fresh `KataGoNet`.

## 2. Model file formats

| # | Format | Produced by | Consumed by |
|---|--------|-------------|-------------|
| M1 | ONNX, single-input (resnet/vit) | `scripts/export_onnx.py` | all backends*, all binaries, whole pipeline |
| M2 | ONNX, dual-input (katago arch or converted kata1) | `export_onnx.py --arch katago` / `tools/katago_to_onnx.py` | same as M1 — no binary treats it specially anymore |
| M3 | PyTorch checkpoint `training.pt` (weights+optimizer+step+bucket+watermark+arch) | trainer | trainer resume, `export_onnx.py --checkpoint` |
| K1 | KataGo native weights `.txt.gz`/`.bin.gz` | katagotraining.org | conversion/warm-init tools only (never loaded by C++) |
| M4 | TRT `.engine` cache | TRT backend (auto) | TRT backend (safe to delete) |
| M5/M6 | `.rknn` / VIP9000 `.nb` | offline converters | NPU backends (compiled siblings of M1/M2) |

\* Backend support level (verified on hardware where possible):
**TensorRT implements both input kinds fully** (parses the ONNX
natively).  **OpenCL implements all three architectures fully** —
resnet, vit, and BOTH dual-input namings (converted kata1 with
mish/relu autodetection AND the trainable KataGoNet) — with three
precision tiers: fp32, portable fp16 (half storage + fp32 math, any
CL 1.2 device), and NVIDIA tensor-core fp16 via inline-PTX mma.sync;
verified against PyTorch reference vectors on all five format
variants (`scripts/make_test_vectors.py` + `build/verify`).
**Eigen** runs the MiniGo resnet on CPU; its dual-input path reads
the CONVERTED-kata1 tensor naming only — the trainable KataGoNet's
state_dict names are not mapped yet (clean "missing tensor" error;
ROADMAP.md deferred item 6) — and is debugging-grade speed at b10c128.  ViT on
Eigen is a TODO placeholder.  **CUDA / Metal** accept every format at
the interface but their kernels are TODO placeholders — `create_handle`
succeeds structurally and the handle constructor throws a uniform
"placeholder (TODO)" error.  **RKNN / VIP9000** run whatever was
compiled into their `.rknn`/`.nb` artifacts (both encodings supported
by their converters).

## 3. Game records — V3, the only training format

```
u16 magic 'MG' | u16 version=3 | i32 board_size | f32 komi |
i32 n_moves | i8 winner | f32 black_score
per move: i16 action (hw = pass) | f32 policy[hw+1]
footer:   i8 owner[hw]   (0 empty/dame, 1 black, 2 white)
```

- Written by `build/selfplay` (one file per game, ~30 KB uncompressed —
  ~60× smaller than the old pre-encoded V2), compressed to
  `g_<id>.bin.zst` by the selfplay driver.
- **Engine-neutral**: no encoded states, no pre-baked augmentation.
  Any model format can *generate* records and any architecture can
  *train* from them.  The trainer (via `scripts/gamedata.py`) replays
  the moves, derives all targets (value/score/ownership/opp-action per
  player), encodes positions for the active architecture, and applies
  a fresh random dihedral transform per sample.
- One disk row = one unique position; the replay bucket credits
  `replay_target × rows` directly (no augmentation divisor).
- Readers: trainer ring/scanner, `selfplay_driver._count_rows`,
  `scripts/visualize.py`.  Parser/replayer/encoders live in ONE place:
  `scripts/gamedata.py` (validated move-for-move against the C++
  engine: replayed boards reproduce the recorded owners/score/winner).
- **Removed**: V1 and V2 record support, the `.bin.gz` read fallback
  (`.zst` is the only pool compression).  Old pools cannot be read —
  archive them and regenerate (they were on-policy data for long-gone
  models anyway).

## 4. Board-state encodings

Both encodings exist twice, and the pairs must stay byte-identical:

| Encoding | C++ (selfplay/inference) | Python (training/replay) |
|----------|--------------------------|--------------------------|
| MiniGo 17-plane (8×2 history + color) | `game.cpp encode()` | `gamedata.encode_minigo` |
| KataGo V7 (22 spatial + 19 global) | `src/katago_inputs.cpp` | `gamedata.encode_katago` |

Shared V7 fidelity caveats (both sides): encore/button/PDA features
and non-default rules bits are zero; ko plane uses simple ko.  Ladder
planes 14-17 ARE computed — both sides carry a faithful port of
upstream's bounded ladder search, proven bit-identical across C++ and
Python by `tools/encoder_parity_test.py` (synthetic ladder scenarios +
real selfplay records; run it after touching either encoder).  The
komi parity wave sits at `gl[18]` with board-area parity anchoring
(upstream `fillRowV7`).

## 5. Other formats

| Format | Role |
|--------|------|
| SGF | evaluate `--output` match records; read by `visualize.py`.  Review only. |
| `training/status.json` | trainer → selfplay/supervisor contract (ramp, bucket fill, liveness) |
| `training/run_config.json` | architecture + komi single source of truth (`init` writes, `run` enforces) |
