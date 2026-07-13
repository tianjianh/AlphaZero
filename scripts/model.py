"""
MiniGo AlphaZero — Neural Network (PyTorch)

Two architectures:
  - AlphaZeroNet: KataGo-style ResNet (alternating SE + GPool residual blocks,
                  global-pooled value/score heads)
  - GoViT: Vision Transformer with GQA and directional positional encoding

Seven-headed architecture (KataGo-style):
  Inference (exported to ONNX):
    1. Policy         — move probabilities [B, action_size]
    2. Value          — W/L/D distribution [B, 3] logits
    3. ScoreMean      — expected score [B, 1]
    4. ScoreStdev     — score uncertainty [B, 1] (softplus)
    5. Ownership      — per-intersection territory [B, board²] (sigmoid)
  Training-only (NOT exported):
    6. Score Belief   — score distribution [B, num_bins] logits
    7. Opponent Policy — opponent's next move [B, action_size]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# Optional: NVIDIA Transformer Engine for FP8 training on Blackwell+
_te = None
try:
    import transformer_engine.pytorch as _te
except ImportError:
    pass


def _linear(in_f, out_f, bias=True, use_fp8=False):
    """Create nn.Linear or te.Linear based on fp8 flag.
    FP8 cuBLASLt requires both feature dims AND the batch leading-dimension
    divisible by 16.  For 3D sequence inputs (transformer blocks), the wgrad
    backward GEMM can use strides based on seq_len which may not be aligned.
    Callers that process sequences should pass use_fp8=False."""
    if use_fp8 and _te is not None and in_f % 16 == 0 and out_f % 16 == 0:
        return _te.Linear(in_f, out_f, bias=bias)
    return nn.Linear(in_f, out_f, bias=bias)


class SEModule(nn.Module):
    """Squeeze-and-Excitation: global-avg-pool → MLP → sigmoid → per-channel rescale."""

    def __init__(self, channels, reduction=16):
        super().__init__()
        hidden = max(8, channels // reduction)
        self.fc1 = nn.Linear(channels, hidden)
        self.fc2 = nn.Linear(hidden, channels)
        # Zero-init fc2 so the gate starts at sigmoid(0) = 0.5 everywhere
        # (constant, not random-per-channel).  Standard SE-ResNet trick: the
        # block behaves as a stable scaled residual at init, then learns its
        # own per-channel gates from there.
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x):
        s = x.mean(dim=[2, 3])
        s = F.relu(self.fc1(s))
        s = torch.sigmoid(self.fc2(s))
        return x * s.unsqueeze(-1).unsqueeze(-1)


class ResBlockSE(nn.Module):
    """3x3 residual block with SE channel attention at the end."""

    def __init__(self, channels, se_reduction=16):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.se = SEModule(channels, reduction=se_reduction)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        return F.relu(out + residual)


class GPoolResBlock(nn.Module):
    """KataGo-style global pooling residual block.

    A parallel 'pool' branch alongside the main 3x3 conv is globally pooled
    (mean+max) and projected by a small FC to per-channel additive biases on
    the main branch.  Gives the block a direct path for global board state
    into local features — stacked 3x3 convs alone have only a limited
    *effective* receptive field even when the *theoretical* one is huge.
    """

    def __init__(self, channels, pool_channels=None):
        super().__init__()
        if pool_channels is None:
            pool_channels = max(16, channels // 4)
        self.pool_channels = pool_channels
        self.conv_main = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn_main = nn.BatchNorm2d(channels)
        self.conv_pool = nn.Conv2d(channels, pool_channels, 3, padding=1, bias=False)
        self.bn_pool = nn.BatchNorm2d(pool_channels)
        self.pool_fc = nn.Linear(2 * pool_channels, channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        # Zero-init the injection so the block starts as a plain residual
        # (pool_fc output ≡ 0 → no bias added).  The network can then learn
        # to use global info without destabilising early training.
        nn.init.zeros_(self.pool_fc.weight)
        nn.init.zeros_(self.pool_fc.bias)

    def forward(self, x):
        residual = x
        main = self.bn_main(self.conv_main(x))
        pool = F.relu(self.bn_pool(self.conv_pool(x)))
        pool_vec = torch.cat([pool.mean(dim=[2, 3]), pool.amax(dim=[2, 3])], dim=1)
        bias = self.pool_fc(pool_vec)
        main = F.relu(main + bias.unsqueeze(-1).unsqueeze(-1))
        out = self.bn2(self.conv2(main))
        return F.relu(out + residual)


class GPoolHead(nn.Module):
    """1x1 conv → global-pool (mean+max+std) → MLP → out.

    KataGo-style value/score head.  Replaces AlphaZero's 1x1-to-1-channel +
    flattened-FC design, which collapses all channel information into a
    single feature map before the FC.  Keeping multiple channels and pooling
    them spatially preserves much richer features for the final MLP; adding
    std on top of mean+max captures per-channel spatial variance, which
    helps value/score calibration.
    """

    def __init__(self, trunk_channels, head_channels, mlp_hidden, out_features, use_fp8=False):
        super().__init__()
        self.conv = nn.Conv2d(trunk_channels, head_channels, 1, bias=False)
        self.bn = nn.BatchNorm2d(head_channels)
        self.fc1 = _linear(3 * head_channels, mlp_hidden, use_fp8=use_fp8)
        self.fc2 = _linear(mlp_hidden, out_features, use_fp8=use_fp8)

    def forward(self, x):
        h = F.relu(self.bn(self.conv(x)))
        pooled = torch.cat([
            h.mean(dim=[2, 3]),
            h.amax(dim=[2, 3]),
            h.std(dim=[2, 3]),
        ], dim=1)
        return self.fc2(F.relu(self.fc1(pooled)))


class AlphaZeroNet(nn.Module):
    input_kind = "single"   # MiniGo 17-plane tensor
    """KataGo-style ResNet: alternating SE and GPool residual blocks.

    Block 0 is SE, block 1 is GPool, and so on — with N blocks, ceil(N/2)
    are SE and floor(N/2) are GPool.  They alternate rather than stacking
    SE-on-top-of-GPool per block because both mechanisms provide
    channel-wise global conditioning; doubling up per block is redundant,
    while alternating gives every block some form of global awareness.

    Heads: policy uses the classic 1x1 → 2ch → FC + pass design; value,
    scoreMean, scoreStdev, scoreBelief use GPoolHead; ownership is a 1×1
    conv; opponent policy mirrors the policy head architecture.
    """

    def __init__(self, board_size=9, input_channels=17, num_filters=64, num_res_blocks=5,
                 use_fp8=False):
        super().__init__()
        self.board_size = board_size
        action_size = board_size * board_size + 1
        hw = board_size * board_size

        self.input_conv = nn.Conv2d(input_channels, num_filters, 3, padding=1, bias=False)
        self.input_bn = nn.BatchNorm2d(num_filters)

        self.trunk = nn.ModuleList()
        for i in range(num_res_blocks):
            if i % 2 == 0:
                self.trunk.append(ResBlockSE(num_filters))
            else:
                self.trunk.append(GPoolResBlock(num_filters))

        # --- Head 1: Policy ---
        self.policy_conv = nn.Conv2d(num_filters, 2, 1, bias=False)
        self.policy_bn = nn.BatchNorm2d(2)
        self.policy_fc = _linear(2 * hw, action_size, use_fp8=use_fp8)

        head_ch = 32
        mlp_hidden = max(64, num_filters)

        # --- Head 2: Value (3-class: win/loss/draw) ---
        self.value_head = GPoolHead(num_filters, head_ch, mlp_hidden,
                                    out_features=3, use_fp8=use_fp8)

        # --- Head 3: ScoreMean (regression) ---
        self.score_mean_head = GPoolHead(num_filters, head_ch, mlp_hidden,
                                         out_features=1, use_fp8=use_fp8)

        # --- Head 4: ScoreStdev (positive via softplus) ---
        self.score_stdev_head = GPoolHead(num_filters, head_ch, mlp_hidden,
                                          out_features=1, use_fp8=use_fp8)

        # --- Head 5: Ownership (per-intersection sigmoid) ---
        self.ownership_conv = nn.Conv2d(num_filters, 1, 1)

        # --- Head 6: Score Belief (training-only, bin classification) ---
        num_bins = hw * 2 + 1
        self.score_belief_head = GPoolHead(num_filters, head_ch, mlp_hidden,
                                           out_features=num_bins, use_fp8=use_fp8)

        # --- Head 7: Opponent Policy (training-only) ---
        self.opp_policy_conv = nn.Conv2d(num_filters, 2, 1, bias=False)
        self.opp_policy_bn = nn.BatchNorm2d(2)
        self.opp_policy_fc = _linear(2 * hw, action_size, use_fp8=use_fp8)

    def _trunk(self, x):
        out = F.relu(self.input_bn(self.input_conv(x)))
        for block in self.trunk:
            out = block(out)
        return out

    def _policy(self, trunk):
        p = F.relu(self.policy_bn(self.policy_conv(trunk)))
        p = p.view(p.size(0), -1)
        return self.policy_fc(p)

    def forward(self, x):
        """All 7 heads — used during training."""
        trunk = self._trunk(x)

        policy = self._policy(trunk)
        value = self.value_head(trunk)                         # [B, 3] logits
        score_mean = self.score_mean_head(trunk)               # [B, 1]
        score_stdev = F.softplus(self.score_stdev_head(trunk)) # [B, 1]
        ownership = self.ownership_conv(trunk).view(x.size(0), -1)  # [B, board²]
        score_belief = self.score_belief_head(trunk)           # [B, num_bins]

        opp = F.relu(self.opp_policy_bn(self.opp_policy_conv(trunk)))
        opp = opp.view(opp.size(0), -1)
        opp_policy = self.opp_policy_fc(opp)                   # [B, action_size]

        return policy, value, score_mean, score_stdev, ownership, score_belief, opp_policy

    def forward_inference(self, x):
        """5 inference heads only — used by predict() and ONNX export."""
        trunk = self._trunk(x)

        policy = self._policy(trunk)
        value = self.value_head(trunk)                         # [B, 3] logits
        score_mean = self.score_mean_head(trunk)               # [B, 1]
        score_stdev = F.softplus(self.score_stdev_head(trunk)) # [B, 1]
        ownership = torch.sigmoid(self.ownership_conv(trunk).view(x.size(0), -1))

        return policy, value, score_mean, score_stdev, ownership

    def predict(self, state_tensor, device="cpu"):
        self.eval()
        with torch.no_grad():
            x = torch.from_numpy(state_tensor).float().unsqueeze(0).to(device)
            policy, value_logits, score_mean, score_stdev, ownership = self.forward_inference(x)
            probs = F.softmax(policy, dim=1).squeeze(0).cpu().numpy()
            value = (F.softmax(value_logits, dim=1)[0, 0] - F.softmax(value_logits, dim=1)[0, 1]).item()
            score = score_mean.item()
        return probs, value, score


# ══════════════════════════════════════════════════════════
#  GoViT — Vision Transformer for Go
# ══════════════════════════════════════════════════════════

def _build_directional_rel_indices(n):
    """Signed (dx, dy) relative bias for all (i, j) token pairs.

    Each unique (dx, dy) offset gets its own bucket. For 9x9: (2*9-1)^2 = 289 buckets.
    NOT D4-invariant — the model can distinguish all 4 directions.
    D4 data augmentation handles symmetry instead.
    """
    hw = n * n
    span = 2 * n - 1
    indices = torch.zeros(hw, hw, dtype=torch.long)
    for i in range(hw):
        r1, c1 = i // n, i % n
        for j in range(hw):
            r2, c2 = j // n, j % n
            dx = r2 - r1 + (n - 1)  # shift to [0, 2n-2]
            dy = c2 - c1 + (n - 1)
            indices[i, j] = dx * span + dy
    return indices, span * span


class GQAAttention(nn.Module):
    """Grouped Query Attention with directional relative positional bias.

    Packed QKV projection (one GEMM) and fused scaled_dot_product_attention.
    """

    def __init__(self, d_model, num_heads, kv_groups, num_rel_buckets, head_dim=32,
                 attn_dropout=0.0):
        super().__init__()
        assert num_heads % kv_groups == 0
        self.num_heads = num_heads
        self.kv_groups = kv_groups
        self.head_dim = head_dim
        self.attn_dropout = attn_dropout

        q_dim = num_heads * head_dim
        kv_dim = kv_groups * head_dim

        # Packed [Q | K | V] — one read of x, one GEMM
        self.qkv_proj = nn.Linear(d_model, q_dim + 2 * kv_dim)
        self.out_proj = nn.Linear(q_dim, d_model)

        # Per-head relative positional bias: [num_heads, num_rel_buckets]
        self.rel_bias = nn.Parameter(torch.zeros(num_heads, num_rel_buckets))

    def forward(self, x, rel_indices):
        B, N, _ = x.shape
        H, G, d = self.num_heads, self.kv_groups, self.head_dim

        q_dim = H * d
        kv_dim = G * d

        qkv = self.qkv_proj(x)                                     # [B, N, Q+K+V]
        q, k, v = torch.split(qkv, [q_dim, kv_dim, kv_dim], dim=-1)

        q = q.view(B, N, H, d).transpose(1, 2)                     # [B, H, N, d]
        k = k.view(B, N, G, d).transpose(1, 2)                     # [B, G, N, d]
        v = v.view(B, N, G, d).transpose(1, 2)                     # [B, G, N, d]

        # Expand KV groups: [B, G, N, d] → [B, H, N, d]
        k = k.repeat_interleave(H // G, dim=1)
        v = v.repeat_interleave(H // G, dim=1)

        # Additive relative positional bias [1, H, N, N]
        attn_bias = self.rel_bias[:, rel_indices].unsqueeze(0)

        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_bias,
            dropout_p=self.attn_dropout if self.training else 0.0,
            is_causal=False,
        )                                                           # [B, H, N, d]

        out = out.transpose(1, 2).contiguous().view(B, N, H * d)
        return self.out_proj(out)


class TransformerBlock(nn.Module):
    """Pre-norm transformer block with GQA."""

    def __init__(self, d_model, num_heads, kv_groups, mlp_ratio, num_rel_buckets, head_dim=32,
                 attn_dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = GQAAttention(d_model, num_heads, kv_groups, num_rel_buckets, head_dim,
                                 attn_dropout)
        self.norm2 = nn.LayerNorm(d_model)
        mlp_hidden = d_model * mlp_ratio
        # nn.Linear for sequence layers — runs in BF16 via torch autocast.
        # te.Linear fails on 3D sequence input (FP8 wgrad stride alignment).
        self.mlp = nn.Sequential(
            nn.Linear(d_model, mlp_hidden),
            nn.GELU(),
            nn.Linear(mlp_hidden, d_model),
        )

    def forward(self, x, rel_indices):
        x = x + self.attn(self.norm1(x), rel_indices)
        x = x + self.mlp(self.norm2(x))
        return x


class GoViT(nn.Module):
    """Vision Transformer for Go with directional positional encoding.

    Positional encoding:
      - Absolute: factorized row + col embeddings (9+9 params for 9x9)
      - Relative: per-head bias indexed by signed (dx, dy) — 289 buckets for 9x9

    D4 symmetry handled by data augmentation, not architecture.
    Attention: Grouped Query Attention (6 Q heads, 2 KV groups by default)
    """

    def __init__(self, board_size=9, input_channels=17,
                 d_model=192, depth=8, num_heads=6, kv_groups=2,
                 mlp_ratio=4, head_dim=32, use_fp8=False):
        super().__init__()
        self.board_size = board_size

        # Token embedding: per-intersection linear projection
        self.token_proj = _linear(input_channels, d_model, use_fp8=use_fp8)

        # Factorized 2D position embedding: row + col
        self.row_embed = nn.Embedding(board_size, d_model)
        self.col_embed = nn.Embedding(board_size, d_model)
        row_ids = torch.arange(board_size).unsqueeze(1).expand(board_size, board_size).reshape(-1)
        col_ids = torch.arange(board_size).unsqueeze(0).expand(board_size, board_size).reshape(-1)
        self.register_buffer("row_ids", row_ids)
        self.register_buffer("col_ids", col_ids)

        # Directional relative bias: signed (dx, dy), 289 buckets for 9x9
        rel_indices, num_rel_buckets = _build_directional_rel_indices(board_size)
        self.register_buffer("rel_indices", rel_indices)  # always [hw, hw]

        # Transformer blocks (nn.Linear only — BF16 via torch autocast)
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, num_heads, kv_groups, mlp_ratio,
                             num_rel_buckets, head_dim, attn_dropout=0.0)
            for _ in range(depth)
        ])
        self.final_norm = nn.LayerNorm(d_model)

        # --- Head 1: Policy ---
        self.policy_proj = _linear(d_model, 1, use_fp8=use_fp8)
        self.pass_logit = nn.Parameter(torch.zeros(1))

        # --- Head 2: Value (3-class: win/loss/draw) ---
        self.value_fc1 = _linear(d_model, d_model, use_fp8=use_fp8)
        self.value_fc2 = _linear(d_model, 3, use_fp8=use_fp8)

        # --- Head 3: ScoreMean (regression) ---
        self.score_mean_fc1 = _linear(d_model, d_model, use_fp8=use_fp8)
        self.score_mean_fc2 = _linear(d_model, 1, use_fp8=use_fp8)

        # --- Head 4: ScoreStdev (positive via softplus) ---
        self.score_stdev_fc1 = _linear(d_model, d_model, use_fp8=use_fp8)
        self.score_stdev_fc2 = _linear(d_model, 1, use_fp8=use_fp8)

        # --- Head 5: Ownership (per-intersection sigmoid) ---
        self.ownership_proj = _linear(d_model, 1, use_fp8=use_fp8)

        # --- Head 6: Score Belief (training-only) ---
        num_bins = board_size * board_size * 2 + 1
        self.score_belief_fc1 = _linear(d_model, d_model, use_fp8=use_fp8)
        self.score_belief_fc2 = _linear(d_model, num_bins, use_fp8=use_fp8)

        # --- Head 7: Opponent Policy (training-only) ---
        self.opp_policy_proj = _linear(d_model, 1, use_fp8=use_fp8)
        self.opp_pass_logit = nn.Parameter(torch.zeros(1))

    def _trunk(self, x):
        x = x.flatten(2).transpose(1, 2)                           # [B, hw, C_in]
        x = self.token_proj(x)                                      # [B, hw, d_model]
        x = x + self.row_embed(self.row_ids) + self.col_embed(self.col_ids)
        for block in self.blocks:
            x = block(x, self.rel_indices)
        return self.final_norm(x)                                    # [B, hw, d_model]

    def _policy(self, trunk):
        B = trunk.size(0)
        p_board = self.policy_proj(trunk).squeeze(-1)               # [B, hw]
        p_pass = self.pass_logit.expand(B, 1)                       # [B, 1]
        return torch.cat([p_board, p_pass], dim=1)                   # [B, hw+1]

    def forward(self, x):
        """All 7 heads — used during training."""
        B = x.size(0)
        trunk = self._trunk(x)
        pooled = trunk.mean(dim=1)                                   # [B, d_model]

        policy = self._policy(trunk)
        value = self.value_fc2(F.gelu(self.value_fc1(pooled)))       # [B, 3] logits
        score_mean = self.score_mean_fc2(F.gelu(self.score_mean_fc1(pooled)))  # [B, 1]
        score_stdev = F.softplus(self.score_stdev_fc2(F.gelu(self.score_stdev_fc1(pooled))))
        ownership = self.ownership_proj(trunk).squeeze(-1)           # [B, hw]
        score_belief = self.score_belief_fc2(F.gelu(self.score_belief_fc1(pooled)))

        opp_board = self.opp_policy_proj(trunk).squeeze(-1)          # [B, hw]
        opp_pass = self.opp_pass_logit.expand(B, 1)
        opp_policy = torch.cat([opp_board, opp_pass], dim=1)        # [B, hw+1]

        return policy, value, score_mean, score_stdev, ownership, score_belief, opp_policy

    def forward_inference(self, x):
        """5 inference heads only — used by predict() and ONNX export."""
        trunk = self._trunk(x)
        pooled = trunk.mean(dim=1)

        policy = self._policy(trunk)
        value = self.value_fc2(F.gelu(self.value_fc1(pooled)))
        score_mean = self.score_mean_fc2(F.gelu(self.score_mean_fc1(pooled)))
        score_stdev = F.softplus(self.score_stdev_fc2(F.gelu(self.score_stdev_fc1(pooled))))
        ownership = torch.sigmoid(self.ownership_proj(trunk).squeeze(-1))

        return policy, value, score_mean, score_stdev, ownership

    def predict(self, state_tensor, device="cpu"):
        self.eval()
        with torch.no_grad():
            x = torch.from_numpy(state_tensor).float().unsqueeze(0).to(device)
            policy, value_logits, score_mean, score_stdev, ownership = self.forward_inference(x)
            probs = F.softmax(policy, dim=1).squeeze(0).cpu().numpy()
            value = (F.softmax(value_logits, dim=1)[0, 0] - F.softmax(value_logits, dim=1)[0, 1]).item()
            score = score_mean.item()
        return probs, value, score


# ══════════════════════════════════════════════════════════
#  KataGoNet — trainable KataGo-V7 architecture
#
#  KataGo-style trunk (pre-activation residual blocks with global-
#  pooling bias injection, dual input: 22 spatial planes + 19 global
#  features) attached to the SAME 7-head set as AlphaZeroNet/GoViT,
#  so all three architectures train under one loss and export the
#  same 5-output inference contract.  The ONNX it exports is a
#  "KataGo V7 format" model exactly like a converted kata1 net
#  (inputs state_spatial [B,22,H,W] + state_global [B,19]).
#
#  Trunk fidelity notes (vs upstream b-series / tools/katago_arch.py):
#  - stem: conv3x3(spatial) + linear(global) broadcast-added.
#  - blocks alternate ordinary / gpool, pre-activation
#    (BN→ReLU→conv), matching upstream block structure.
#  - gpool stats use KataGo's (mean, mean·(√HW−14)·0.1, max) triple.
#  Heads are OURS (not kata1's): stock kata1 checkpoints therefore
#  can't be resumed directly — convert them for inference with
#  tools/katago_to_onnx.py (warm-init from kata1 was removed).
# ══════════════════════════════════════════════════════════

def _kata_gpool_stats(x):
    """KataGo gpool triple: [mean, mean·(√HW−14)·0.1, max] → [B, 3C]."""
    hw = x.size(2) * x.size(3)
    scale = (float(hw) ** 0.5 - 14.0) * 0.1
    mean = x.mean(dim=[2, 3])
    return torch.cat([mean, mean * scale, x.amax(dim=[2, 3])], dim=1)


class KataGoPreActBlock(nn.Module):
    """Pre-activation ordinary residual block (BN→ReLU→3x3, twice)."""

    def __init__(self, channels):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = self.conv2(F.relu(self.bn2(out)))
        return x + out


class KataGoGPoolBlock(nn.Module):
    """Pre-activation residual block with a global-pooling side branch.

    The side branch is globally pooled with KataGo's stats triple and
    projected to per-channel biases added into the regular branch
    before the second conv (upstream GlobalPoolingResidualBlock).
    """

    def __init__(self, channels, pool_channels=None):
        super().__init__()
        if pool_channels is None:
            pool_channels = max(16, channels // 4)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv_regular = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.conv_gpool = nn.Conv2d(channels, pool_channels, 3, padding=1, bias=False)
        self.gpool_bn = nn.BatchNorm2d(pool_channels)
        self.gpool_to_bias = nn.Linear(3 * pool_channels, channels)
        self.bn2 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        # Zero-init the injection so the block starts as a plain residual.
        nn.init.zeros_(self.gpool_to_bias.weight)
        nn.init.zeros_(self.gpool_to_bias.bias)

    def forward(self, x):
        pre = F.relu(self.bn1(x))
        regular = self.conv_regular(pre)
        gp = F.relu(self.gpool_bn(self.conv_gpool(pre)))
        bias = self.gpool_to_bias(_kata_gpool_stats(gp))
        out = regular + bias.unsqueeze(-1).unsqueeze(-1)
        out = self.conv2(F.relu(self.bn2(out)))
        return x + out


class KataGoNet(nn.Module):
    """Trainable KataGo-V7 architecture with MiniGo's 7-head set.

    forward(spatial [B,22,H,W], global [B,19]) — same 7-tuple contract
    as AlphaZeroNet/GoViT, so train_continuous is architecture-uniform.
    """

    # Input contract (matches src/engine/katago_inputs.cpp / scripts/gamedata.py)
    SPATIAL_CHANNELS = 22
    GLOBAL_CHANNELS = 19
    input_kind = "dual"   # export/training dispatch (others default "single")

    def __init__(self, board_size=9, channels=128, num_blocks=10, use_fp8=False):
        super().__init__()
        self.board_size = board_size
        action_size = board_size * board_size + 1
        hw = board_size * board_size

        # Stem: conv(spatial) + linear(global) broadcast-added (upstream stem)
        self.stem_conv = nn.Conv2d(self.SPATIAL_CHANNELS, channels, 3,
                                   padding=1, bias=False)
        self.stem_global = nn.Linear(self.GLOBAL_CHANNELS, channels)

        self.blocks = nn.ModuleList(
            KataGoGPoolBlock(channels) if i % 2 == 1 else KataGoPreActBlock(channels)
            for i in range(num_blocks))
        self.tip_bn = nn.BatchNorm2d(channels)

        # ── Heads: identical structure to AlphaZeroNet ──
        self.policy_conv = nn.Conv2d(channels, 2, 1, bias=False)
        self.policy_bn = nn.BatchNorm2d(2)
        self.policy_fc = _linear(2 * hw, action_size, use_fp8=use_fp8)

        head_ch = 32
        mlp_hidden = max(64, channels)
        self.value_head = GPoolHead(channels, head_ch, mlp_hidden,
                                    out_features=3, use_fp8=use_fp8)
        self.score_mean_head = GPoolHead(channels, head_ch, mlp_hidden,
                                         out_features=1, use_fp8=use_fp8)
        self.score_stdev_head = GPoolHead(channels, head_ch, mlp_hidden,
                                          out_features=1, use_fp8=use_fp8)
        self.ownership_conv = nn.Conv2d(channels, 1, 1)
        num_bins = hw * 2 + 1
        self.score_belief_head = GPoolHead(channels, head_ch, mlp_hidden,
                                           out_features=num_bins, use_fp8=use_fp8)
        self.opp_policy_conv = nn.Conv2d(channels, 2, 1, bias=False)
        self.opp_policy_bn = nn.BatchNorm2d(2)
        self.opp_policy_fc = _linear(2 * hw, action_size, use_fp8=use_fp8)

    def _trunk(self, spatial, global_features):
        out = self.stem_conv(spatial) \
            + self.stem_global(global_features).unsqueeze(-1).unsqueeze(-1)
        for block in self.blocks:
            out = block(out)
        return F.relu(self.tip_bn(out))

    def _policy(self, trunk):
        p = F.relu(self.policy_bn(self.policy_conv(trunk)))
        return self.policy_fc(p.view(p.size(0), -1))

    def forward(self, spatial, global_features):
        trunk = self._trunk(spatial, global_features)

        policy = self._policy(trunk)
        value = self.value_head(trunk)                          # [B, 3] logits
        score_mean = self.score_mean_head(trunk)                # [B, 1]
        score_stdev = F.softplus(self.score_stdev_head(trunk))  # [B, 1]
        ownership = self.ownership_conv(trunk).view(spatial.size(0), -1)
        score_belief = self.score_belief_head(trunk)            # [B, num_bins]

        opp = F.relu(self.opp_policy_bn(self.opp_policy_conv(trunk)))
        opp_policy = self.opp_policy_fc(opp.view(opp.size(0), -1))

        return policy, value, score_mean, score_stdev, ownership, score_belief, opp_policy

    def forward_inference(self, spatial, global_features):
        trunk = self._trunk(spatial, global_features)

        policy = self._policy(trunk)
        value = self.value_head(trunk)
        score_mean = self.score_mean_head(trunk)
        score_stdev = F.softplus(self.score_stdev_head(trunk))
        ownership = torch.sigmoid(
            self.ownership_conv(trunk).view(spatial.size(0), -1))

        return policy, value, score_mean, score_stdev, ownership


def convert_te_to_nn(model):
    """Replace te.Linear with nn.Linear for ONNX export compatibility."""
    if _te is None:
        return
    for name, module in list(model.named_modules()):
        if isinstance(module, _te.Linear):
            replacement = nn.Linear(module.in_features, module.out_features,
                                     bias=module.bias is not None)
            replacement.weight.data.copy_(module.weight.data)
            if module.bias is not None:
                replacement.bias.data.copy_(module.bias.data)
            # Navigate to parent and replace
            parts = name.split('.')
            parent = model
            for p in parts[:-1]:
                if p.isdigit():
                    parent = parent[int(p)]
                else:
                    parent = getattr(parent, p)
            setattr(parent, parts[-1], replacement)


ARCHITECTURES = ("resnet", "vit", "katago")


def create_model(arch="resnet", board_size=9, input_channels=17, **kwargs):
    """Factory: create model by architecture name.

    resnet / vit take the MiniGo 17-plane input (single tensor);
    katago takes the KataGo-V7 dual input (22 spatial + 19 global) and
    ignores input_channels.  Check `model.input_kind` ("single"/"dual")
    to dispatch encoding and forward-call shape.
    """
    use_fp8 = kwargs.get("use_fp8", False)
    if arch == "resnet":
        return AlphaZeroNet(
            board_size=board_size, input_channels=input_channels,
            num_filters=kwargs.get("num_filters", 64),
            num_res_blocks=kwargs.get("num_res_blocks", 5),
            use_fp8=use_fp8,
        )
    elif arch == "vit":
        return GoViT(
            board_size=board_size, input_channels=input_channels,
            d_model=kwargs.get("d_model", 192),
            depth=kwargs.get("depth", 8),
            num_heads=kwargs.get("heads", 6),
            kv_groups=kwargs.get("kv_groups", 2),
            mlp_ratio=kwargs.get("mlp_ratio", 4),
            head_dim=kwargs.get("head_dim", 32),
            use_fp8=use_fp8,
        )
    elif arch == "katago":
        return KataGoNet(
            board_size=board_size,
            channels=kwargs.get("num_filters", 128),
            num_blocks=kwargs.get("num_res_blocks", 10),
            use_fp8=use_fp8,
        )
    else:
        raise ValueError(f"Unknown architecture: {arch} "
                         f"(supported: {ARCHITECTURES})")
