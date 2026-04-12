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
`train.py` line numbers shifted correctly from the autocast changes, but model.py
padding changes may not have been pulled).

### Verification steps

1. Add debug print in GoViT.__init__ to confirm padding is active:
   ```python
   if self.seq_pad > 0:
       print(f"GoViT: FP8 sequence padding {hw} → {hw + self.seq_pad}")
   ```

2. Deploy BOTH `scripts/model.py` and `scripts/train.py` to the Blackwell machine.

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

**Status**: Stubbed — throws at handle creation.

The ResNet architecture was replaced with a KataGo-style design (alternating
SE + GPool residual blocks, global-pool value/score heads; see
`scripts/model.py` for the PyTorch definition).  TensorRT parses the new
ONNX graph directly and works out of the box.  The Eigen/CUDA/OpenCL/Metal
backends still contain their old AlphaZero-ResNet forward passes but
throw at `ComputeHandle` construction with a "TODO: add kernels" message.

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

And teach `loaded_model.cpp` to populate per-block weight slots for the
new layout.  The old per-op `load_conv` / `load_bn` / `load_fc` helpers are
kept in place (currently `(void)`-cast to silence unused warnings) for
exactly this re-enable path.

See the stubs in:

- `src/eigen_compute.cpp` (`EigenComputeHandle` constructor)
- `src/cuda_compute.cu` (`CUDAComputeHandle` constructor)
- `src/opencl_compute.cpp` (`OpenCLComputeHandle` constructor)
- `src/metal_compute.mm` (`MetalComputeHandle` constructor)
