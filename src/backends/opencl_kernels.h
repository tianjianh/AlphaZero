#pragma once
// ================================================================
// OpenCL kernel sources for the MiniGo backend.
//
// Two programs are built from PORTABLE_SRC:
//   * fp32:  -DSTORE_HALF=0 — plain float buffers, any OpenCL 1.2 device
//   * fp16:  -DSTORE_HALF=1 — half STORAGE via vload_half/vstore_half
//            (core since OpenCL 1.0 — no cl_khr_fp16 required), float
//            ARITHMETIC.  Halves memory traffic on every device while
//            keeping fp32 accumulation everywhere.
//
// MMA_SRC is a third program compiled only on NVIDIA devices (probed at
// context init): the GEMM/conv workhorse re-implemented with inline-PTX
// mma.sync tensor-core instructions (m16n8k16, f16 inputs, f32
// accumulate).  It is argument-compatible with `gemm` so the host picks
// per-op between them.
//
// Layout convention (all architectures):
//   activations are channel-major matrices  [C, N*HW]  with a padded
//   leading dimension ld (rows 16-element aligned).  A convolution is
//   then one implicit GEMM  weight[Co, Ci*K*K] x im2col(input) — the
//   im2col matrix is never materialized; the B-tile loader gathers it
//   on the fly.  KS==1 degenerates to a dense GEMM, which is also how
//   every fully-connected layer runs (B = [K, N] with HW=1).
// ================================================================

namespace minigo {

static const char* OPENCL_PORTABLE_SRC = R"CL(
// ---------------------------------------------------------------
// Storage-type abstraction: XT is the element type of activation /
// weight buffers.  All arithmetic is float regardless.
// ---------------------------------------------------------------
#if STORE_HALF
typedef half XT;
#define LOADX(p, i)     vload_half((size_t)(i), (p))
#define STOREX(v, p, i) vstore_half_rte((float)(v), (size_t)(i), (p))
#else
typedef float XT;
#define LOADX(p, i)     ((p)[(size_t)(i)])
#define STOREX(v, p, i) ((p)[(size_t)(i)] = (v))
#endif

// Epilogue flag bits (shared with gemm_mma and the host)
#define EF_ROWBIAS  1   // y += p1[m]                  (FC bias)
#define EF_BN       2   // y  = y * p0[m] + p1[m]      (pre-fused BN)
#define EF_CB_PRE   4   // y += cb[m, n/HW]  BEFORE BN (kata gpool / stem)
#define EF_CB_POST  8   // y += cb[m, n/HW]  AFTER  BN (minigo gpool)
#define EF_RES     16   // y += res[m, n]              (residual add)

// Activation codes
#define ACT_NONE    0
#define ACT_RELU    1
#define ACT_MISH    2
#define ACT_GELU    3
#define ACT_SIGMOID 4

inline float apply_act(float x, const int act) {
    if (act == ACT_RELU) return fmax(x, 0.0f);
    if (act == ACT_MISH) {
        // x * tanh(softplus(x)); overflow-safe (matches the Eigen backend)
        float sp = (x > 20.0f) ? x : log1p(exp(fmax(x, -30.0f)));
        return x * tanh(sp);
    }
    if (act == ACT_GELU)    return 0.5f * x * (1.0f + erf(x * 0.70710678118654752f));
    if (act == ACT_SIGMOID) return 1.0f / (1.0f + exp(-x));
    return x;
}

// ---------------------------------------------------------------
// Input unpack: fp32 staging [N, row_len] -> XT channel-major [C, *].
//   spatial planes: src_off = 0,        HW = H*W
//   global vector:  src_off = C_sp*HW,  HW = 1   (gives [C_gl, N])
// ---------------------------------------------------------------
__kernel void xpose_in(
    __global const float* src, __global XT* dst,
    const int row_len, const int src_off,
    const int C, const int HW, const int N, const int ldD)
{
    const int gid = get_global_id(0);
    const int total = C * N * HW;
    if (gid >= total) return;
    const int c   = gid / (N * HW);
    const int r   = gid - c * N * HW;
    const int img = r / HW;
    const int hw  = r - img * HW;
    STOREX(src[(size_t)img * row_len + src_off + c * HW + hw],
           dst, (size_t)c * ldD + (size_t)img * HW + hw);
}

// ---------------------------------------------------------------
// The GEMM / implicit-conv workhorse (portable version).
//
//   C[M,N] = A[M,K] x B(K,N)  (+ epilogue)
//
//   KS==1: B is a dense [K, ldB] matrix.
//   KS>=3: B is the im2col of `Bsrc` = input [C_in, ldB] over an HxW
//          board, gathered on the fly (stride 1, pad KS/2).
//
// Tiling: 16x16 threads, 4x4 outputs each -> 64x64 tile, K-tile 16.
// SMEM tiles are stored as float (converted at load) so the inner
// product runs at full fp32 regardless of storage type.
// ---------------------------------------------------------------
#define TS   16
#define WPTM 4
#define WPTN 4
#define BM   (TS * WPTM)
#define BN   (TS * WPTN)
#define BKK  16

__kernel __attribute__((reqd_work_group_size(TS, TS, 1)))
void gemm(
    __global const XT* A, __global const XT* Bsrc, __global XT* Cdst,
    __global const float* p0, __global const float* p1,
    __global const XT* cb, __global const XT* res,
    const int M, const int N, const int K,
    const int ldB, const int ldC, const int ldcb,
    const int KS, const int PAD, const int H, const int W, const int HW,
    const int flags, const int act)
{
    __local float As[BM][BKK + 1];
    __local float Bs[BKK][BN + 1];
    // K-decomposition LUT for the implicit-conv gather: one entry per
    // K-tile row.  Integer division is ~25 emulated instructions on
    // GPUs; doing it per gathered element dominates the whole kernel,
    // so it is done once per tile by BKK threads instead.
    __local int Lic[BKK], Lkh[BKK], Lkw[BKK];

    const int tm  = get_local_id(0);
    const int tn  = get_local_id(1);
    const int lid = tn * TS + tm;
    const int m0  = get_group_id(0) * BM;
    const int n0  = get_group_id(1) * BN;

    float acc[WPTM][WPTN];
    for (int i = 0; i < WPTM; i++)
        for (int j = 0; j < WPTN; j++) acc[i][j] = 0.0f;

    const int KK = KS * KS;

    // The B-tile column this thread gathers is fixed for the whole
    // K-loop (idx % BN == lid % BN) — decompose it once.
    const int b_gn = n0 + (lid & (BN - 1));
    int b_img = 0, b_oh = 0, b_ow = 0;
    if (KS > 1 && b_gn < N) {
        b_img = b_gn / HW;
        const int bhw = b_gn - b_img * HW;
        b_oh = bhw / W;
        b_ow = bhw - b_oh * W;
    }

    const int numTiles = (K + BKK - 1) / BKK;
    for (int t = 0; t < numTiles; t++) {
        const int k0 = t * BKK;

        if (KS > 1 && lid < BKK) {
            const int gk = k0 + lid;
            const int ic = gk / KK, rr = gk - ic * KK;
            const int kh = rr / KS;
            Lic[lid] = ic;
            Lkh[lid] = kh - PAD;
            Lkw[lid] = rr - kh * KS - PAD;
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        // Load A tile: BM x BKK elements, 4 per thread, row segments
        // (coalesced along K).
        for (int i = 0; i < (BM * BKK) / (TS * TS); i++) {
            const int idx = i * TS * TS + lid;
            const int r = idx / BKK, c = idx - r * BKK;
            const int gm = m0 + r, gk = k0 + c;
            As[r][c] = (gm < M && gk < K) ? (float)LOADX(A, (size_t)gm * K + gk) : 0.0f;
        }
        // Load B tile: BKK x BN, 4 per thread, coalesced along N.
        for (int i = 0; i < (BKK * BN) / (TS * TS); i++) {
            const int idx = i * TS * TS + lid;
            const int r = idx >> 6;             // idx / BN
            const int gk = k0 + r;
            float v = 0.0f;
            if (gk < K && b_gn < N) {
                if (KS == 1) {
                    v = LOADX(Bsrc, (size_t)gk * ldB + b_gn);
                } else {
                    const int ih = b_oh + Lkh[r], iw = b_ow + Lkw[r];
                    if ((uint)ih < (uint)H && (uint)iw < (uint)W)
                        v = LOADX(Bsrc, (size_t)Lic[r] * ldB + (size_t)b_img * HW + ih * W + iw);
                }
            }
            Bs[r][lid & (BN - 1)] = v;
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        for (int k = 0; k < BKK; k++) {
            float a[WPTM], b[WPTN];
            for (int i = 0; i < WPTM; i++) a[i] = As[tm + i * TS][k];
            for (int j = 0; j < WPTN; j++) b[j] = Bs[k][tn + j * TS];
            for (int i = 0; i < WPTM; i++)
                for (int j = 0; j < WPTN; j++)
                    acc[i][j] = mad(a[i], b[j], acc[i][j]);
        }
        // No trailing barrier: the LUT barrier at the top of the next
        // iteration already separates this compute from the next loads.
    }

    // Epilogue
    for (int i = 0; i < WPTM; i++) {
        const int m = m0 + tm + i * TS;
        if (m >= M) continue;
        const float s0 = (flags & EF_BN) ? p0[m] : 1.0f;
        const float s1 = (flags & (EF_BN | EF_ROWBIAS)) ? p1[m] : 0.0f;
        for (int j = 0; j < WPTN; j++) {
            const int n = n0 + tn + j * TS;
            if (n >= N) continue;
            float y = acc[i][j];
            if (flags & EF_CB_PRE)  y += (float)LOADX(cb, (size_t)m * ldcb + n / HW);
            y = y * s0 + s1;
            if (flags & EF_CB_POST) y += (float)LOADX(cb, (size_t)m * ldcb + n / HW);
            if (flags & EF_RES)     y += (float)LOADX(res, (size_t)m * ldC + n);
            STOREX(apply_act(y, act), Cdst, (size_t)m * ldC + n);
        }
    }
}

// ---------------------------------------------------------------
// Elementwise: dst = act(src * scale[c] + bias[c])
// (pre-activation BN for KataGo blocks, trunk tip, etc.)
// ---------------------------------------------------------------
__kernel void bn_act_ew(
    __global const XT* src, __global XT* dst,
    __global const float* scale, __global const float* bias,
    const int C, const int T, const int ld, const int act)
{
    const int gid = get_global_id(0);
    if (gid >= C * T) return;
    const int c = gid / T, n = gid - c * T;
    float y = (float)LOADX(src, (size_t)c * ld + n) * scale[c] + bias[c];
    STOREX(apply_act(y, act), dst, (size_t)c * ld + n);
}

// ---------------------------------------------------------------
// Elementwise: dst = act(a * (mul? mul[c, n/HW] : 1) + b)
// SE final rescale (+residual+relu) and generic residual joins.
// ---------------------------------------------------------------
__kernel void resadd_act(
    __global const XT* a, __global const XT* b, __global const XT* mul,
    __global XT* dst,
    const int C, const int T, const int ld, const int ldm, const int HW,
    const int has_mul, const int act)
{
    const int gid = get_global_id(0);
    if (gid >= C * T) return;
    const int c = gid / T, n = gid - c * T;
    float y = (float)LOADX(a, (size_t)c * ld + n);
    if (has_mul) y *= (float)LOADX(mul, (size_t)c * ldm + n / HW);
    y += (float)LOADX(b, (size_t)c * ld + n);
    STOREX(apply_act(y, act), dst, (size_t)c * ld + n);
}

// ---------------------------------------------------------------
// Global-pooling statistics.  One work-group per (c, img) pair
// reduces that image's HW values; fp32 accumulation.
//
// variant 0 MEAN:          out[c]                      (S=1, SE module)
// variant 1 MEAN_MAX:      out[c | C+c]                (S=2, minigo gpool)
// variant 2 MEAN_MAX_STD:  out[c | C+c | 2C+c]         (S=3, gpool heads,
//                          std is the unbiased PyTorch default)
// variant 3 KATA:          [mean | mean*c1 | max]      (S=3)
// variant 4 KATA_VH:       [mean | mean*c1 | mean*c2]  (S=3)
// ---------------------------------------------------------------
#define STATS_WG 64
__kernel __attribute__((reqd_work_group_size(STATS_WG, 1, 1)))
void gstats(
    __global const XT* src, __global XT* dst,
    const int C, const int N, const int HW, const int ld, const int ldo,
    const int variant, const float c1, const float c2)
{
    __local float lsum[STATS_WG], lmax[STATS_WG], lsq[STATS_WG];
    const int grp = get_group_id(0);
    const int c = grp / N, img = grp - c * N;
    const int lid = get_local_id(0);

    float s = 0.0f, mx = -3.4e38f, sq = 0.0f;
    for (int i = lid; i < HW; i += STATS_WG) {
        const float v = (float)LOADX(src, (size_t)c * ld + (size_t)img * HW + i);
        s += v; sq += v * v; mx = fmax(mx, v);
    }
    lsum[lid] = s; lmax[lid] = mx; lsq[lid] = sq;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int off = STATS_WG / 2; off > 0; off >>= 1) {
        if (lid < off) {
            lsum[lid] += lsum[lid + off];
            lsq[lid]  += lsq[lid + off];
            lmax[lid]  = fmax(lmax[lid], lmax[lid + off]);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (lid != 0) return;

    const float mean = lsum[0] / (float)HW;
    if (variant == 0) {
        STOREX(mean, dst, (size_t)c * ldo + img);
    } else if (variant == 1) {
        STOREX(mean,    dst, (size_t)c * ldo + img);
        STOREX(lmax[0], dst, (size_t)(C + c) * ldo + img);
    } else if (variant == 2) {
        const float var = fmax(lsq[0] - (float)HW * mean * mean, 0.0f)
                          / (float)max(HW - 1, 1);
        STOREX(mean,       dst, (size_t)c * ldo + img);
        STOREX(lmax[0],    dst, (size_t)(C + c) * ldo + img);
        STOREX(sqrt(var),  dst, (size_t)(2 * C + c) * ldo + img);
    } else if (variant == 3) {
        STOREX(mean,      dst, (size_t)c * ldo + img);
        STOREX(mean * c1, dst, (size_t)(C + c) * ldo + img);
        STOREX(lmax[0],   dst, (size_t)(2 * C + c) * ldo + img);
    } else {  // KATA_VH
        STOREX(mean,      dst, (size_t)c * ldo + img);
        STOREX(mean * c1, dst, (size_t)(C + c) * ldo + img);
        STOREX(mean * c2, dst, (size_t)(2 * C + c) * ldo + img);
    }
}

// ---------------------------------------------------------------
// LayerNorm over the feature dimension (rows).  One work-group per
// token column; fp32 two-pass (sum / sumsq) in local memory.
// ---------------------------------------------------------------
#define LN_WG 128
__kernel __attribute__((reqd_work_group_size(LN_WG, 1, 1)))
void layernorm(
    __global const XT* src, __global XT* dst,
    __global const float* gamma, __global const float* beta,
    const int D, const int ld, const float eps)
{
    __local float lsum[LN_WG], lsq[LN_WG];
    const int t = get_group_id(0);
    const int lid = get_local_id(0);

    float s = 0.0f, sq = 0.0f;
    for (int d = lid; d < D; d += LN_WG) {
        const float v = (float)LOADX(src, (size_t)d * ld + t);
        s += v; sq += v * v;
    }
    lsum[lid] = s; lsq[lid] = sq;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int off = LN_WG / 2; off > 0; off >>= 1) {
        if (lid < off) { lsum[lid] += lsum[lid + off]; lsq[lid] += lsq[lid + off]; }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    const float mean = lsum[0] / (float)D;
    const float var  = fmax(lsq[0] / (float)D - mean * mean, 0.0f);
    const float rstd = rsqrt(var + eps);
    for (int d = lid; d < D; d += LN_WG) {
        const float v = (float)LOADX(src, (size_t)d * ld + t);
        STOREX((v - mean) * rstd * gamma[d] + beta[d], dst, (size_t)d * ld + t);
    }
}

// ---------------------------------------------------------------
// Factorized position embedding: X[d, t] += row_e[r, d] + col_e[c, d]
// ---------------------------------------------------------------
__kernel void pos_embed_add(
    __global XT* x, __global const float* row_e, __global const float* col_e,
    const int D, const int T, const int ld, const int W, const int HW)
{
    const int gid = get_global_id(0);
    if (gid >= D * T) return;
    const int d = gid / T, t = gid - d * T;
    const int hw = t % HW;
    const int r = hw / W, c = hw - r * W;
    float v = (float)LOADX(x, (size_t)d * ld + t)
            + row_e[(size_t)r * D + d] + col_e[(size_t)c * D + d];
    STOREX(v, x, (size_t)d * ld + t);
}

// ---------------------------------------------------------------
// Fused GQA attention with directional relative bias — one
// work-group per (image, query-head), flash-attention style online
// softmax over K/V tiles staged in local memory.  The relative
// bucket index is recomputed from token coordinates on the fly so no
// [hw, hw] bias matrix is ever materialized.
//
// qkv rows: [0, Hq*dh) = Q,  [q_off_k, +G*dh) = K,  [q_off_v, ...) = V
// out rows: [Hq*dh]
// Softmax and accumulation are always fp32.
// ---------------------------------------------------------------
#define ATT_TJ 64
#define ATT_MAX_DH 64
__kernel void attention(
    __global const XT* qkv, __global XT* outb, __global const float* rel,
    const int Hq, const int G, const int dh, const int hw, const int W,
    const int ldq, const int ldo,
    const int k_off, const int v_off,
    const float scale, const int rel_off, const int span)
{
    __local float Ks[ATT_MAX_DH][ATT_TJ];
    __local float Vs[ATT_MAX_DH][ATT_TJ + 1];

    const int grp = get_group_id(0);
    const int img = grp / Hq, h = grp - img * Hq;
    const int g   = h / (Hq / G);
    const int lid = get_local_id(0);
    const int LT  = get_local_size(0);
    const int i   = lid;                    // query token (may be >= hw: pad)
    const int base = img * hw;

    float q[ATT_MAX_DH];
    if (i < hw)
        for (int d = 0; d < dh; d++)
            q[d] = (float)LOADX(qkv, (size_t)(h * dh + d) * ldq + base + i);

    const int ri = (i < hw) ? i / W : 0;
    const int ci = (i < hw) ? i - ri * W : 0;
    __global const float* relh = rel + rel_off + (size_t)h * span * span;

    float m = -3.4e38f, l = 0.0f;
    float acc[ATT_MAX_DH];
    for (int d = 0; d < dh; d++) acc[d] = 0.0f;

    for (int j0 = 0; j0 < hw; j0 += ATT_TJ) {
        const int tj = min(ATT_TJ, hw - j0);
        // Cooperative K/V tile load (coalesced along tokens)
        for (int idx = lid; idx < dh * tj; idx += LT) {
            const int d = idx / tj, jj = idx - d * tj;
            Ks[d][jj] = (float)LOADX(qkv, (size_t)(k_off + g * dh + d) * ldq + base + j0 + jj);
            Vs[d][jj] = (float)LOADX(qkv, (size_t)(v_off + g * dh + d) * ldq + base + j0 + jj);
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        if (i < hw) {
            for (int jj = 0; jj < tj; jj++) {
                const int j = j0 + jj;
                float s = 0.0f;
                for (int d = 0; d < dh; d++) s = mad(q[d], Ks[d][jj], s);
                const int rj = j / W, cj = j - rj * W;
                const int bucket = (rj - ri + W - 1) * span + (cj - ci + W - 1);
                s = s * scale + relh[bucket];

                // Branchless online softmax: a "skip rescale when the max
                // is unchanged" branch was measured slower here — each
                // thread is a different query token, so the branch
                // diverges within the warp and both paths execute anyway.
                const float m_new = fmax(m, s);
                const float corr  = exp(m - m_new);
                const float p     = exp(s - m_new);
                l = l * corr + p;
                for (int d = 0; d < dh; d++)
                    acc[d] = mad(acc[d], corr, p * Vs[d][jj]);
                m = m_new;
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (i < hw) {
        const float inv_l = 1.0f / l;
        for (int d = 0; d < dh; d++)
            STOREX(acc[d] * inv_l, outb, (size_t)(h * dh + d) * ldo + base + i);
    }
}

// ---------------------------------------------------------------
// pooled[d, img] = mean over the image's hw tokens
// ---------------------------------------------------------------
__kernel void mean_tokens(
    __global const XT* src, __global XT* dst,
    const int D, const int N, const int HW, const int ld, const int ldo)
{
    const int gid = get_global_id(0);
    if (gid >= D * N) return;
    const int d = gid / N, img = gid - d * N;
    float s = 0.0f;
    for (int i = 0; i < HW; i++)
        s += (float)LOADX(src, (size_t)d * ld + (size_t)img * HW + i);
    STOREX(s / (float)HW, dst, (size_t)d * ldo + img);
}

// ---------------------------------------------------------------
// Policy-feature flatten: [C, N*HW] -> [C*HW, N]
// (so policy_fc becomes one dense GEMM over the batch)
// ---------------------------------------------------------------
__kernel void flatten_pol(
    __global const XT* src, __global XT* dst,
    const int C, const int N, const int HW, const int ld, const int ldo)
{
    const int gid = get_global_id(0);
    if (gid >= C * N * HW) return;
    const int c = gid / (N * HW);
    const int r = gid - c * N * HW;
    const int img = r / HW, j = r - img * HW;
    STOREX((float)LOADX(src, (size_t)c * ld + (size_t)img * HW + j),
           dst, (size_t)(c * HW + j) * ldo + img);
}

// ---------------------------------------------------------------
// Output packing -> one contiguous fp32 buffer, one D2H read.
// Per image: [ policy(A) | value_logits(3) | score | score_sd | ownership(hw) ]
// All values are raw logits; the host applies softmax / softplus /
// sigmoid / tanh per the model contract.
// ---------------------------------------------------------------

// Variant A: policy is a matrix [A, ldp] (minigo-style heads)
__kernel void pack_mg(
    __global const XT* pol, __global const XT* v3,
    __global const XT* sm, __global const XT* sd, __global const XT* own,
    __global float* outb,
    const int A, const int hw,
    const int ldp, const int ldv, const int ldt, const int N)
{
    const int n = get_global_id(1);
    const int k = get_global_id(0);
    const int stride = A + 5 + hw;
    if (n >= N || k >= stride) return;
    float v;
    if (k < A)              v = (float)LOADX(pol, (size_t)k * ldp + n);
    else if (k < A + 3)     v = (float)LOADX(v3, (size_t)(k - A) * ldv + n);
    else if (k == A + 3)    v = (float)LOADX(sm, (size_t)n);
    else if (k == A + 4)    v = (float)LOADX(sd, (size_t)n);
    else                    v = (float)LOADX(own, (size_t)n * hw + (k - A - 5));
    outb[(size_t)n * stride + k] = v;
}

// Variant B: policy is spatial row `p_row` of [*, ldt] + a pass logit
// taken from row `pass_row` of pass_buf (kata1) or the constant
// pass_const (ViT).  Score rows are selectable (kata1 sv3 indexing).
__kernel void pack_sp(
    __global const XT* psrc, __global const XT* pass_buf,
    __global const XT* v3, __global const XT* ssrc, __global const XT* dsrc,
    __global const XT* own,
    __global float* outb,
    const int hw, const int p_row, const int use_pass_buf,
    const float pass_const, const int pass_row,
    const int s_row, const int d_row,
    const int ldt, const int ldv, const int N)
{
    const int n = get_global_id(1);
    const int k = get_global_id(0);
    const int A = hw + 1;
    const int stride = A + 5 + hw;
    if (n >= N || k >= stride) return;
    float v;
    if (k < hw)             v = (float)LOADX(psrc, (size_t)p_row * ldt + (size_t)n * hw + k);
    else if (k == hw)       v = use_pass_buf
                                ? (float)LOADX(pass_buf, (size_t)pass_row * ldv + n)
                                : pass_const;
    else if (k < A + 3)     v = (float)LOADX(v3, (size_t)(k - A) * ldv + n);
    else if (k == A + 3)    v = (float)LOADX(ssrc, (size_t)s_row * ldv + n);
    else if (k == A + 4)    v = (float)LOADX(dsrc, (size_t)d_row * ldv + n);
    else                    v = (float)LOADX(own, (size_t)n * hw + (k - A - 5));
    outb[(size_t)n * stride + k] = v;
}

// Ownership rows live in a [1, N*HW] spatial buffer; pack kernels index
// it as own[n*hw + i], so copy it out to a tight [N*hw] view first when
// the spatial buffer is padded.  (Cheaper: the host passes the padded
// buffer directly since row 0 is tight up to N*HW <= ld.)
)CL";

// ================================================================
// NVIDIA tensor-core GEMM (inline PTX, fp16 storage, fp32 accumulate).
// Argument-compatible with `gemm` above.  Compiled as its own program
// on NVIDIA devices only; availability is probed at context init.
//
// Tiling: 128 threads = 4 warps in a 2x2 grid; block tile 64x64,
// K-tile 32; each warp owns a 32x32 sub-tile = 2x4 mma.m16n8k16 per
// 16-wide K-slice.  SMEM tiles are packed-half `uint` arrays padded
// to a 20-uint row stride (bank-conflict-free fragment reads:
// lane -> bank map covers all 32 banks exactly).
// ================================================================
static const char* OPENCL_MMA_SRC = R"CL(
#define EF_ROWBIAS  1
#define EF_BN       2
#define EF_CB_PRE   4
#define EF_CB_POST  8
#define EF_RES     16

#define ACT_NONE    0
#define ACT_RELU    1
#define ACT_MISH    2
#define ACT_GELU    3
#define ACT_SIGMOID 4

inline float apply_act(float x, const int act) {
    if (act == ACT_RELU) return fmax(x, 0.0f);
    if (act == ACT_MISH) {
        float sp = (x > 20.0f) ? x : log1p(exp(fmax(x, -30.0f)));
        return x * tanh(sp);
    }
    if (act == ACT_GELU)    return 0.5f * x * (1.0f + erf(x * 0.70710678118654752f));
    if (act == ACT_SIGMOID) return 1.0f / (1.0f + exp(-x));
    return x;
}

#define MBM 64
#define MBN 64
#define MBK 64
#define LDS 36   // uint row stride: (MBK/2 = 32) + 4 pad — the (4g+t)%32
                 // lane->bank map covers all 32 banks (conflict-free)

__kernel __attribute__((reqd_work_group_size(128, 1, 1)))
void gemm(
    __global const ushort* A, __global const ushort* Bsrc, __global half* Cdst,
    __global const float* p0, __global const float* p1,
    __global const half* cb, __global const half* res,
    const int M, const int N, const int K,
    const int ldB, const int ldC, const int ldcb,
    const int KS, const int PAD, const int H, const int W, const int HW,
    const int flags, const int act)
{
    __local uint Asu[MBM * LDS];   // [row][kp]  row-major halves of A
    __local uint Bsu[MBN * LDS];   // [n][kp]    n-major halves of B (col operand)
    // Double-buffered K-decomposition LUT for the implicit-conv gather
    // (see the portable gemm): kills the per-element integer divisions,
    // and the parity slot lets tile t+1's LUT build while t's is in use.
    __local int Lic[2][MBK], Lkh[2][MBK], Lkw[2][MBK];

    const int lid  = get_local_id(0);
    const int lane = lid & 31;
    const int wid  = lid >> 5;
    const int wm   = wid >> 1;          // warp row   (2 warps)
    const int wn   = wid & 1;           // warp col   (2 warps)
    const int g    = lane >> 2;         // mma group id
    const int t    = lane & 3;          // thread-in-group
    const int m0   = get_group_id(0) * MBM;
    const int n0   = get_group_id(1) * MBN;

    const int KK = KS * KS;

    // This thread's B-gather column is fixed across the whole K-loop.
    const int b_gn = n0 + (lid & 63);
    int b_img = 0, b_oh = 0, b_ow = 0;
    if (KS > 1 && b_gn < N) {
        b_img = b_gn / HW;
        const int bhw = b_gn - b_img * HW;
        b_oh = bhw / W;
        b_ow = bhw - b_oh * W;
    }

    float d[2][4][4];
    for (int mt = 0; mt < 2; mt++)
        for (int nt = 0; nt < 4; nt++)
            for (int r = 0; r < 4; r++) d[mt][nt][r] = 0.0f;

// Single-stage pipeline.  Register double-buffering was measured on
// A40: it wins ~20% at batch<=16 (latency-bound) but costs ~20% at
// batch>=128 (the +16..32 staging registers cross an occupancy cliff),
// and selfplay throughput lives at large batch — so the simple
// single-stage loop is kept.
    const int nT = (K + MBK - 1) / MBK;
    for (int ti = 0; ti < nT; ti++) {
        const int k0 = ti * MBK;
        if (KS > 1 && lid < MBK) {
            const int gk = k0 + lid;
            const int ic = gk / KK, rr = gk - ic * KK;
            const int kh = rr / KS;
            Lic[0][lid] = ic;
            Lkh[0][lid] = kh - PAD;
            Lkw[0][lid] = rr - kh * KS - PAD;
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        // ---- Stage A tile: 64 rows x 32 k-pairs; 16 pairs per thread.
        for (int it = 0; it < 16; it++) {
            const int idx = it * 128 + lid;
            const int row = idx >> 5, kp = idx & 31;
            const int gm = m0 + row;
            const int gk = k0 + kp * 2;
            uint lo = 0, hi = 0;
            if (gm < M) {
                if (gk < K)     lo = A[(size_t)gm * K + gk];
                if (gk + 1 < K) hi = A[(size_t)gm * K + gk + 1];
            }
            Asu[row * LDS + kp] = lo | (hi << 16);
        }
        // ---- Stage B tile: 64 cols x 32 k-pairs; coalesced along n.
        for (int it = 0; it < 16; it++) {
            const int kp = (it * 128 + lid) >> 6;
            uint packed = 0;
            if (b_gn < N) {
                for (int h2 = 0; h2 < 2; h2++) {
                    const int kl = kp * 2 + h2;      // K index within tile
                    const int gk = k0 + kl;
                    uint hv = 0;
                    if (gk < K) {
                        if (KS == 1) {
                            hv = Bsrc[(size_t)gk * ldB + b_gn];
                        } else {
                            const int ih = b_oh + Lkh[0][kl];
                            const int iw = b_ow + Lkw[0][kl];
                            if ((uint)ih < (uint)H && (uint)iw < (uint)W)
                                hv = Bsrc[(size_t)Lic[0][kl] * ldB +
                                          (size_t)b_img * HW + ih * W + iw];
                        }
                    }
                    packed |= hv << (16 * h2);
                }
            }
            Bsu[(lid & 63) * LDS + kp] = packed;
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        // ---- Four 16-deep mma slices per K-tile.
        for (int kk = 0; kk < 4; kk++) {
            const int kb = kk * 8 + t;
            uint a[2][4], b[4][2];
            for (int mt = 0; mt < 2; mt++) {
                const int rm = wm * 32 + mt * 16;
                a[mt][0] = Asu[(rm + g) * LDS + kb];
                a[mt][1] = Asu[(rm + g + 8) * LDS + kb];
                a[mt][2] = Asu[(rm + g) * LDS + kb + 4];
                a[mt][3] = Asu[(rm + g + 8) * LDS + kb + 4];
            }
            for (int nt = 0; nt < 4; nt++) {
                const int cn = wn * 32 + nt * 8 + g;
                b[nt][0] = Bsu[cn * LDS + kb];
                b[nt][1] = Bsu[cn * LDS + kb + 4];
            }
            for (int mt = 0; mt < 2; mt++)
                for (int nt = 0; nt < 4; nt++)
                    asm volatile(
                        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                        : "+f"(d[mt][nt][0]), "+f"(d[mt][nt][1]),
                          "+f"(d[mt][nt][2]), "+f"(d[mt][nt][3])
                        : "r"(a[mt][0]), "r"(a[mt][1]), "r"(a[mt][2]), "r"(a[mt][3]),
                          "r"(b[nt][0]), "r"(b[nt][1]));
        }
        barrier(CLK_LOCAL_MEM_FENCE);   // mma reads done — safe to restage
    }

    // ---- Epilogue (fp32 math, half store).
    for (int mt = 0; mt < 2; mt++) {
        for (int r = 0; r < 4; r++) {
            const int m = m0 + wm * 32 + mt * 16 + g + ((r >> 1) << 3);
            if (m >= M) continue;
            const float s0 = (flags & EF_BN) ? p0[m] : 1.0f;
            const float s1 = (flags & (EF_BN | EF_ROWBIAS)) ? p1[m] : 0.0f;
            for (int nt = 0; nt < 4; nt++) {
                const int n = n0 + wn * 32 + nt * 8 + t * 2 + (r & 1);
                if (n >= N) continue;
                float y = d[mt][nt][r];
                if (flags & EF_CB_PRE)  y += vload_half((size_t)m * ldcb + n / HW, cb);
                y = y * s0 + s1;
                if (flags & EF_CB_POST) y += vload_half((size_t)m * ldcb + n / HW, cb);
                if (flags & EF_RES)     y += vload_half((size_t)m * ldC + n, res);
                vstore_half_rte(apply_act(y, act), (size_t)m * ldC + n, Cdst);
            }
        }
    }
}
)CL";

}  // namespace minigo
