# tools/ — conversion & validation toolbox

Everything here is offline tooling; nothing in the training pipeline
or the C++ engine imports from this directory (except that
`scripts/make_test_vectors.py` reuses `katago_arch.py` to synthesize
converted-kata1 test models).

## Stock-KataGo network conversion

| Tool | Purpose |
|---|---|
| `parse_katago.py` | Parser for kata1 `.bin.gz` / `.txt.gz` (model_version 8-15) → numpy descriptors.  Library, no CLI. |
| `katago_arch.py` | Inference-only PyTorch mirror of the kata1 graph, loaded from parsed descriptors.  Library. |
| `katago_to_onnx.py` | kata1 → dual-input ONNX with MiniGo-shape outputs (the artifact every backend runs). |
| `katago_parity_test.py` | PyTorch ↔ ONNX Runtime parity for a converted network. |

## Encoder validation

| Tool | Purpose |
|---|---|
| `encoder_parity_test.py` | Proves the C++ and Python KataGo-V7 encoders bit-identical (ladder solver included); `--force-py` tests the pure-Python fallback.  Needs `make -C build encode_dump minigo_ladder`. |

## Rockchip NPU (RKNN) chain — see docs/RKNN_CONVERSION.md

| Tool | Purpose |
|---|---|
| `kata_export_for_rknn.py` | kata1 → RKNN-friendly ONNX (4-D shapes, no dynamic batch). |
| `onnx_rknn_mitigations.py` | Graph surgery both NPU chains need (initializer unsharing, op rewrites) + parity check. |
| `onnx_to_rknn.py` | ONNX → `.rknn` compile (x86_64 host, rknn-toolkit2), fp16 or int8/hybrid. |
| `rknn_calibration.py` | Builds int8 calibration fixtures from selfplay encodings. |
| `rknn_accuracy_analysis.py` | Per-layer quantisation error analysis. |
| `rknn_onboard_test.py` | On-board smoke + accuracy harness. |

## Allwinner A733 / VIP9000 NPU chain — see docs/A733_CONVERSION.md

| Tool | Purpose |
|---|---|
| `onnx_to_a733_docker.sh` | ONNX → `.nb` (fp16) via Allwinner's Acuity Docker image — the only working path. |
| `onnx_to_a733_quantize_docker.sh` | Same, int8 + hybrid quantisation against a calibration fixture. |
| `a733_gen_calib.py` | Calibration fixture generation for the int8 path. |

Removed (git history keeps them): `warm_init_from_katago.py` (warm-
start never produced a net that beat random init under the current
7-head trainer), `onnx_to_a733.py` + `a733_verify.py` (pip-acuitylite
dead end — wrong chip table), `migrate_qkv.py` (one-time ViT
checkpoint migration, long completed).
