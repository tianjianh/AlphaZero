# Roadmap & Known Limitations

Open work items.  Completed items are removed (git history has them);
everything here is still open as of 2026-07.

## CUDA / Metal backend ports (stubbed)

The trunk architectures were replaced with the KataGo-style design
(SE + GPool residual blocks, global-pool heads; plus ViT and the
dual-input KataGo-V7 arch).  TensorRT parses the ONNX directly;
**OpenCL implements all three architectures natively** (fp32 /
portable-fp16 / NVIDIA tensor-core tiers — see README "OpenCL GPU
Backend").  The CUDA and Metal backends still contain the old
AlphaZero-ResNet forward pass and throw at `ComputeHandle` creation.

Re-enabling them is now only a kernel port: all weight plumbing lives
in `LoadedModel::weights()` (lazy, shared, pre-fused, covers all four
format variants), and `src/opencl_compute.cpp::forward_*` is the
reference for the exact op sequences.  See the stubs in:

- `src/cuda_compute.cu` (`CUDAComputeHandle` constructor)
- `src/metal_compute.mm` (`MetalComputeHandle` constructor)

Because `auto` backend selection never picks a stub, this is a
performance feature, not a correctness gap.

## OpenCL performance ideas (measured, deferred)

- Register double-buffering in the tensor-core GEMM: wins ~20% at
  batch ≤ 16 but loses ~20% at batch ≥ 128 (occupancy cliff from the
  +16..32 staging registers); selfplay lives at large batch, so the
  single-stage loop was kept.
- A skinny-M GEMM variant for the 1-2-channel head convs (launch-bound).
- ViT attention at hw=81 is latency-bound on the serial per-query
  softmax chain (~50% of ViT time); a warp-cooperative (i,j)-parallel
  scheme needs cross-lane reductions.

## KataGo-arch trainer sampling prefetch

`--arch katago` batch sampling costs ~135 ms/batch with the native
ladder library (`libminigo_ladder.so`, ctypes-loaded, thread-pooled;
the pure-Python fallback needs a spawn process pool and ~230 ms).
The minigo path is ~50-80 ms.  The step loop samples synchronously, so
a one-batch prefetch (sample batch N+1 while N trains) would hide the
remaining cost entirely.

## FP8 training on Blackwell (te.Linear wgrad alignment)

**Status**: blocked — needs verification on a Blackwell machine.

Transformer Engine's `te.Linear` fails during backward on SM 12.0:
FP8 cuBLASLt requires all GEMM leading dimensions divisible by 16, and
the ViT wgrad GEMM's leading dimension is `seq_len = board²` (81 for
9x9; 81 % 16 = 1).  All feature dims are already aligned (d_model 192,
mlp 768, heads 6×32, kv 128).

Fix: pad the token sequence 81 → 96 before the transformer blocks and
slice back after (zero-valued padding, no position embedding,
checkpoint-compatible).  This was implemented in `bca5e9e` and reverted
in `cb24be1` after it still crashed on the remote machine — most likely
because `model.py` wasn't deployed there.  To retry:

1. Re-apply the padding (see `bca5e9e`: `seq_pad` in `GoViT.__init__`,
   pad after position embedding, slice before heads, restore `use_fp8`
   in TransformerBlock/GQAAttention).
2. Deploy BOTH `scripts/model.py` and `scripts/train_continuous.py`;
   add a startup print to confirm padding is active.
3. Clear `trt_cache/` on the target machine.
4. If it still fails with padding confirmed active, check the TE
   version / file an upstream bug; workarounds: 10x10 padded board
   (100 tokens) or d_model 256.

Current workaround: transformer blocks use `nn.Linear` under BF16
autocast; only the post-pooling head FCs use `te.Linear` (FP8) — works
but leaves most FP8 potential unused.

## Deferred engine items (verified against upstream KataGo, not changed)

1. **Positional superko**: the ko rule is simple ko (one-position
   memory via `prev_board`); Tromp-Taylor mandates positional superko.
   Long cycles are bounded only by `max_moves_per_game`.  Adding PSK
   needs a position-hash set maintained in `GoGame::play()`.
2. **`score_scale` not board-adaptive**: `config.h` hard-codes 18.0
   (KataGo's `2*sqrt(area)` for 9x9).  A 19x19 run should derive ~38
   from the model's board size at load time (pass `--score-scale 38`
   meanwhile).
3. **Eigen backend re-parses the ONNX per handle**: N server threads
   cost N× parse time and N× weight RAM.  Porting Eigen to the shared
   `LoadedModel::weights()` bundle fixes this (harmless at N=1).
4. **TensorRT >= 11 runs FP32/TF32 only**: TRT 11 removed weakly-typed
   precision flags; for FP16/BF16 engines stay on TRT 10.x until the
   exporters emit half-precision graphs.
5. **Per-eval blocking handoff uses mutex+condvar**: KataGo's own
   design, amortized against GPU batch time; the C++20 upgrade path is
   `std::atomic<int>::wait/notify_one` on `NNResultBuf::done` if futex
   cost ever matters.
6. **Eigen dual-input loader doesn't know the trainable KataGoNet's
   tensor names** (only the converted-kata1 naming).  Fix = port Eigen
   to `LoadedModel::weights()`, which resolves both (the OpenCL
   backend runs both today).
