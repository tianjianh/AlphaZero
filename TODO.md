# TODO

## FP8 Training on Blackwell (te.Linear wgrad alignment)

**Status**: Blocked — needs verification on Blackwell machine

### Problem

Transformer Engine's `te.Linear` fails during backward on Blackwell (SM 12.0):
```
RuntimeError: cublaslt_gemm.cu:157 CanonicalizeGemmInput:
Assertion failed: ret.lda % 16 == 0. Leading dimension requirement on A for FP8 GEMM.
```

FP8 cuBLASLt requires all GEMM leading dimensions divisible by 16. For our ViT,
the wgrad backward GEMM's leading dimension depends on `seq_len = board_size^2`.
9x9 board = 81 tokens, 81 % 16 = 1 — not aligned.

All feature dimensions are already aligned:
- d_model = 192 (192/16 = 12)
- mlp_hidden = 768 (768/16 = 48)
- heads × head_dim = 6 × 32 = 192
- kv_proj out = 2 × 2 × 32 = 128 (128/16 = 8)

### Fix: Pad token sequence to next multiple of 16

Pad 81 → 96 tokens before transformer blocks, slice back to 81 after.
The padding is zero-valued (no position embedding), transparent to output shapes,
and checkpoint-compatible (rel_indices buffer stays [81,81], padded dynamically).

Implementation (was in commit bca5e9e, reverted in cb24be1):

1. `model.py` GoViT.__init__:
   ```python
   self.seq_pad = (16 - hw % 16) % 16 if use_fp8 else 0
   ```

2. `model.py` GoViT.forward, after position embedding:
   ```python
   if self.seq_pad > 0:
       x = F.pad(x, (0, 0, 0, self.seq_pad))       # [B, 96, d_model]
       padded_len = hw + self.seq_pad
       ri = x.new_zeros(padded_len, padded_len, dtype=torch.long)
       ri[:hw, :hw] = self.rel_indices
   else:
       ri = self.rel_indices
   ```

3. After final_norm, before heads:
   ```python
   if self.seq_pad > 0:
       x = x[:, :hw]
   ```

4. Restore `use_fp8` parameter in TransformerBlock and GQAAttention constructors,
   use `_linear(..., use_fp8=use_fp8)` for all linear layers in those blocks.

### Why previous attempt failed

The padding was implemented and tested locally but still crashed on the remote
Blackwell machine. Most likely cause: `model.py` wasn't deployed (the error's
`train_continuous.py` line numbers shifted correctly from the autocast changes, but model.py
padding changes may not have been pulled).

### Verification steps

1. Add debug print in GoViT.__init__ to confirm padding is active:
   ```python
   if self.seq_pad > 0:
       print(f"GoViT: FP8 sequence padding {hw} → {hw + self.seq_pad}")
   ```

2. Deploy BOTH `scripts/model.py` and `scripts/train_continuous.py` to the Blackwell machine.

3. Delete old TensorRT engine cache on the Blackwell machine:
   ```bash
   rm -rf trt_cache/*.engine
   ```
   (Cache filenames now include precision tag `_bf16`/`_fp16` since commit f574c3a,
   so old `_fp16` engines won't collide, but clean slate is safest.)

4. Run training with --fp8 and verify:
   - The "GoViT: FP8 sequence padding 81 → 96" message appears
   - Training completes without the lda assertion error

5. If it STILL fails after confirming padding is active, the issue is TE's internal
   GEMM layout on SM 12.0. Next steps would be:
   - Check TE version (`python -c "import transformer_engine; print(transformer_engine.__version__)"`)
   - Try TE nightly or file a bug at github.com/NVIDIA/TransformerEngine
   - As a workaround, adjust board representation to 10x10 = 100 tokens
     (pad board with a border row/col), since 96 < 100 and rounding to 112
     is the next multiple of 16. Or use d_model=256 which makes all dims
     powers of 2.

### Current workaround

Transformer blocks use `nn.Linear` (BF16 via `torch.amp.autocast`).
Only post-pooling heads (value_fc1, score_fc1) use `te.Linear` (FP8).
This works but wastes FP8 potential — the transformer blocks are >99% of compute.

### Relevant commits

- `f574c3a` — MCTS livelock fix + BF16 for Ampere+ + precision in cache filename
- `bca5e9e` — FP8 padding implementation + nested BF16 autocast (padding reverted in cb24be1)
- `cb24be1` — Reverted padding, restricted te.Linear to post-pooling only
- `32bc9fa` — Removed FP8 from TensorRT auto-detection (BF16 default on SM 8+)


## Non-TensorRT backends for the new KataGo-style ResNet

**Status**: OpenCL DONE (2026-07); CUDA / Metal still stubbed.

The ResNet architecture was replaced with a KataGo-style design (alternating
SE + GPool residual blocks, global-pool value/score heads; see
`scripts/model.py` for the PyTorch definition).  TensorRT parses the new
ONNX graph directly and works out of the box.

**OpenCL now implements all three architectures** (resnet, vit, katago-V7
in both namings) at three precision tiers (fp32 / portable fp16 / NVIDIA
tensor-core fp16 via inline-PTX mma.sync), verified against PyTorch
reference vectors — see the "OpenCL GPU Backend" section in README.md.
The weight plumbing it needed now lives in `LoadedModel::weights()`
(lazy, shared, all four format variants pre-fused) so re-enabling CUDA /
Metal is only a matter of porting the kernel sequences in
`src/opencl_compute.cpp::forward_*` to those APIs.

Deferred OpenCL performance ideas (measured unnecessary for now):
- register double-buffering in the MMA gemm — wins ~20% at batch ≤ 16
  but costs ~20% at batch ≥ 128 (occupancy cliff from +16..32 staging
  registers); selfplay lives at large batch, so single-stage was kept.
- a skinny-M gemm variant for the 1-2 channel head convs (launch-bound).
- ViT attention at hw=81 is latency-bound on the serial per-query
  softmax chain (~50% of ViT time); a warp-cooperative (i,j)-parallel
  scheme would need cross-lane reductions.

The CUDA / Metal backends still contain their old AlphaZero-ResNet forward
passes but throw at `ComputeHandle` construction with a "TODO: add
kernels" message.

To re-enable a backend, add forward-pass support for:

- **`SEModule`** — global avg pool over spatial dims → FC(C→C/r) → ReLU →
  FC(C/r→C) → sigmoid → per-channel broadcast multiply.
- **`GPoolResBlock`** — two parallel 3x3 convs from the same input: a "main"
  conv (C→C) and a "pool" conv (C→Cp).  The pool branch is globally mean+max
  pooled to [B, 2*Cp], projected by FC(2Cp→C), and the resulting [B, C]
  vector is added as a per-channel bias to the main branch before the
  second 3x3 conv.  Then residual add + ReLU as usual.
- **`GPoolHead`** — 1x1 conv (C→head_ch) → BN → ReLU → global mean+max pool →
  FC(2*head_ch→mlp_hidden) → ReLU → FC(mlp_hidden→out_features).

Blocks alternate in the trunk: block 0 SE, block 1 GPool, block 2 SE, ...

The weight plumbing already exists: `LoadedModel::weights()` returns a
lazily-built, pre-fused `ModelWeights` bundle covering all four format
variants (resnet / vit / KataGoNet / converted kata1) — the OpenCL
backend consumes it and is the reference for the kernel sequences.

See the remaining stubs in:

- `src/cuda_compute.cu` (`CUDAComputeHandle` constructor)
- `src/metal_compute.mm` (`MetalComputeHandle` constructor)

## Deferred items from the 2026-07 codebase audit

Verified against upstream KataGo source but deliberately NOT changed in
the audit pass (each is a behavior change or larger refactor):

1. **Positional superko**: the ko rule is simple ko (one-position
   memory via `prev_board`); Tromp-Taylor mandates positional superko.
   Long cycles (triple ko, sending-two-returning-one) are bounded only
   by `max_moves_per_game`.  Adding PSK needs a position-hash set
   maintained in `GoGame::play()`.  Documented in `game.cpp
   score_game()` and `COMPARISON_WITH_KATAGO.md`.
2. **`score_scale` not board-adaptive**: `config.h` hard-codes 18.0
   (KataGo's `2*sqrt(area)` evaluated for 9x9).  A 19x19 run should
   derive ~38 from the model's board size at load time.
3. **Eigen backend re-parses the ONNX per handle**: with N server
   threads that is N× parse cost and N× weight RAM.  Immutable weights
   belong in `EigenComputeContext`, shared across handles.  Harmless
   at N=1 (the common Eigen case).
4. **TensorRT >= 11 runs FP32/TF32 only**: TRT 11 removed weakly-typed
   precision flags; reduced precision now requires exporting FP16/BF16
   ONNX graphs.  The build supports TRT 11 (guarded), but for FP16/BF16
   engines install TRT 10.x until the exporters emit half-precision
   graphs.
5. ~~**KataGo-model banner prints `filters=0 blocks=0`**~~ FIXED with
   the OpenCL backend work: LoadedModel now reads trunk channel/block
   counts from the embedded state_dict for KataGo-format ONNX.
6. **Per-eval blocking handoff still uses mutex+condvar** (the only
   sync heavier than a single AMO left on the search path): each NN
   evaluation does one queue push (mutex) and one wait on the buf's
   mutex+condvar; the server takes the buf mutex once to deliver.
   This is KataGo's own design and is amortized against ~0.4-3 ms of
   GPU inference per batch, so it is NOT worth churn today.  If
   mutex cost ever matters more (e.g. Zaamo-only RISC-V with slow
   futex paths), the C++20 upgrade path is `std::atomic<int>::wait/
   notify_one` on `NNResultBuf::done` — removes the per-buf mutex and
   condvar entirely (plain load + futex, no LR/SC needed).  Requires
   bumping the project to -std=c++20.

7. **Eigen dual-input loader doesn't know the trainable KataGoNet's
   tensor names**: it resolves the converted-kata1 naming
   (`stem.initial_conv.weight`, from tools/katago_arch.py) and fails
   with a clean "missing tensor" error on `--arch katago` exports
   (`stem_conv.weight`, `blocks.N.conv_regular.weight`, ...).  Fix is
   a name map (or a shared naming scheme between scripts/model.py
   KataGoNet and tools/katago_arch.py) in eigen_compute's tensor
   lookup — or porting Eigen to `LoadedModel::weights()`, which
   already resolves both namings (the OpenCL backend runs both).
   TensorRT is unaffected (it reads the graph, not the embedded
   state_dict).
