# Format Support Matrix

Inventory of every model / data / record format the repo touches, which
component supports it, and to what degree.  Written as the ground truth
for the planned format consolidation: today **inference (play /
evaluate / benchmark) accepts both MiniGo and KataGo model formats, but
training is MiniGo-only end-to-end** — KataGo weights can seed a run
(warm-init) but KataGo-encoded data can never enter the training loop.

## 1. Model formats

| # | Format | Produced by | Consumed by | Degree |
|---|--------|-------------|-------------|--------|
| M1 | **MiniGo ONNX** — single input `state [B,17,H,W]`, outputs `policy_logits, value, score_mean, score_stdev, ownership` (post-processing baked in) | `scripts/export_onnx.py` (from random init or `.pt`) | every backend (TensorRT/CUDA*/Eigen/OpenCL*/Metal*/RKNN†/VIP9000†); all 4 binaries; whole training pipeline | **Native.** Full support everywhere. |
| M2 | **KataGo-converted ONNX** — dual input `state_spatial [B,22,H,W]` + `state_global [B,19]`, same 5 outputs | `tools/katago_to_onnx.py` (from K1) | TensorRT backend (full); RKNN/VIP9000 via pre-compiled sibling artifacts (M5/M6); `play`/`evaluate`/`benchmark` | **Inference-only.** `selfplay` refuses (`return 2`); Eigen/CUDA/OpenCL/Metal throw at handle creation; never enters training. Ladder + encore-only input features are zeroed (see `KATAGO_INFERENCE.md`). |
| K1 | **KataGo native weights** `.txt.gz` / `.bin.gz` | katagotraining.org | `tools/katago_to_onnx.py`, `tools/warm_init_from_katago.py`, `tools/katago_parity_test.py` — Python tools only | **Conversion source only.** The C++ engine never loads these directly (KataGo's own `desc.cpp` parser is ported inside the tools). |
| M3 | **PyTorch checkpoint** `training.pt` — weights + optimizer + step + bucket + watermark + arch metadata | `train_continuous.py` (every export + shutdown); `warm_init_from_katago.py --checkpoint` | `train_continuous.py` (resume), `export_onnx.py --checkpoint` | **Native training state.** Not an inference format. |
| M4 | **TensorRT engine cache** `.engine` under `MINIGO_TRT_CACHE` | TensorRT backend (lazy, per GPU × max-batch × precision × TRT version) | TensorRT backend | Derived artifact, auto-managed; safe to delete. |
| M5 | **RKNN compiled model** `.rknn` (sits next to the `.onnx`) | `tools/onnx_to_rknn.py` on x86_64 (rknn-toolkit2) | RKNN backend (aarch64) — ONNX still parsed for metadata, weights come from `.rknn` | Offline-compiled sibling; works for M1 and M2 sources. |
| M6 | **VIP9000 NBG** `network_binary.nb` in `.a733.bs<K>.{int8,fp16}/` sibling dirs | `tools/onnx_to_a733*.sh` (Acuity Docker, x86_64) | VIP9000 backend (aarch64) | Offline-compiled sibling; works for M1 and M2 sources. |

\* CUDA / OpenCL / Metal currently throw a placeholder at handle
creation for the KataGo-style ResNet (SE + GPool blocks) — intentional
TODO state; TensorRT is the production GPU backend.
† Via M5/M6 sibling artifacts, not the ONNX weights themselves.

## 2. Training-data formats

| # | Format | Produced by | Consumed by | Degree |
|---|--------|-------------|-------------|--------|
| D1 | **V2 selfplay records** `.bin` — header `[magic 0x4D47][version 2][count][board]`, per-record `[state 17ch][policy][value][score][ownership][opponent_action]`, 8-fold dihedral pre-augmented | `build/selfplay` (MiniGo models ONLY) | `selfplay_driver.py` (compresses), `train_continuous.py` (`_parse_records`), `scripts/visualize.py` | **The only training format.** States are MiniGo 17-plane; a KataGo-encoded state cannot be represented. |
| D2 | **Compressed pool files** `g_<id>.bin.zst` (zstd, content-size in frame header) | `selfplay_driver.py` publish step | trainer scanner + ring (streaming reader), `visualize.py` | Native pool format; IDs are the coordination watermark. |
| D3 | `.bin.gz` (gzip V2) | nothing anymore | readers keep a gzip fallback (`train_continuous._decompress_*`, `visualize.py`) | **Legacy read-only.** Candidate for removal in the format cleanup. |
| D4 | **KataGo training rows** (`.npz` shuffle output) | — | — | **Not supported anywhere.** KataGo's trainer format; would only matter if we ever trained on KataGo selfplay data. |

## 3. Game-record / misc formats

| # | Format | Produced by | Consumed by | Degree |
|---|--------|-------------|-------------|--------|
| G1 | **SGF** | `build/evaluate --output` | `scripts/visualize.py` | Review-only; not parsed back into training. |
| S1 | **`training/status.json`** | trainer rank 0 | selfplay driver (score ramp + throttle), supervisor heartbeat, `status` command | Pipeline contract (schema documented in `CONTINUOUS_TRAINING.md`). |
| S2 | **`training/run_config.json`** | `run_continuous.py init` | `run_continuous.py run` (defaults + conflict detection) | Architecture/komi single source of truth. |

## 4. Board-state encodings (runtime, not serialized)

| Encoding | Where | Used for |
|----------|-------|----------|
| MiniGo 17-plane (8×2 history snapshots + side-to-move) | `game.cpp encode()` | M1 models — selfplay, training data, inference |
| KataGo V7: 22 spatial planes + 19 globals | `katago_inputs.cpp` (port of upstream `fillRowV7`) | M2 models — inference only; never written to disk |

## 5. Asymmetry summary (for the next-stage cleanup)

- **Eval side**: `evaluate`/`play`/`benchmark` are format-agnostic —
  any mix of M1 and M2 models works on TensorRT (e.g. gating a MiniGo
  net against kata1).
- **Training side**: strictly M1/D1/D2 — `selfplay` hard-rejects M2
  models because D1 records can only hold 17-plane states, and the
  trainer's parser assumes them.
- **Bridge**: the only sanctioned KataGo→training path is
  `warm_init_from_katago.py`, which maps K1 *weights* into the MiniGo
  architecture (M3 + M1) — data formats never cross.
- Cleanup candidates: drop D3 (gzip fallback) once old pools are gone;
  D4 stays unsupported unless KataGo-data training becomes a goal;
  decide whether M2 support should extend beyond TensorRT or be
  documented as TRT-only permanently.
