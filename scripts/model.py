"""
MiniGo AlphaZero — Neural Network (PyTorch)

Two architectures:
  - AlphaZeroNet: KataGo-style ResNet (alternating SE + GPool residual blocks,
                  global-pooled value/score heads)
  - GoViT: Vision Transformer with GQA and directional positional encoding

Both are triple-headed (policy + value + score).  Shared between training
and weight export.
"""

import math
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
    """KataGo-style ResNet: alternating SE and GPool residual blocks.

    Block 0 is SE, block 1 is GPool, and so on — with N blocks, ceil(N/2)
    are SE and floor(N/2) are GPool.  They alternate rather than stacking
    SE-on-top-of-GPool per block because both mechanisms provide
    channel-wise global conditioning; doubling up per block is redundant,
    while alternating gives every block some form of global awareness.

    Heads: policy uses the classic 1x1 → 2ch → FC + pass design; value
    and score use GPoolHead for global-aware output.
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

        self.policy_conv = nn.Conv2d(num_filters, 2, 1, bias=False)
        self.policy_bn = nn.BatchNorm2d(2)
        self.policy_fc = _linear(2 * hw, action_size, use_fp8=use_fp8)

        head_ch = 32
        mlp_hidden = max(64, num_filters)
        self.value_head = GPoolHead(num_filters, head_ch, mlp_hidden,
                                    out_features=1, use_fp8=use_fp8)
        num_bins = hw * 2 + 1
        self.score_head = GPoolHead(num_filters, head_ch, mlp_hidden,
                                    out_features=num_bins, use_fp8=use_fp8)

    def forward(self, x):
        out = F.relu(self.input_bn(self.input_conv(x)))
        for block in self.trunk:
            out = block(out)

        p = F.relu(self.policy_bn(self.policy_conv(out)))
        p = p.view(p.size(0), -1)
        p = self.policy_fc(p)

        v = torch.tanh(self.value_head(out))
        s = self.score_head(out)
        return p, v, s

    def predict(self, state_tensor, device="cpu"):
        self.eval()
        with torch.no_grad():
            x = torch.from_numpy(state_tensor).float().unsqueeze(0).to(device)
            logits, value, score_logits = self(x)
            probs = F.softmax(logits, dim=1).squeeze(0).cpu().numpy()
            board_area = self.board_size * self.board_size
            bins = torch.arange(score_logits.size(1), device=x.device).float() - board_area
            score = (F.softmax(score_logits, dim=1) * bins).sum(dim=1).item()
        return probs, value.item(), score


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
        hw = board_size * board_size
        action_size = hw + 1

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

        # Policy head: per-token logit + learnable pass logit
        self.policy_proj = _linear(d_model, 1, use_fp8=use_fp8)
        self.pass_logit = nn.Parameter(torch.zeros(1))

        # Value head: mean pool → MLP → tanh
        self.value_fc1 = _linear(d_model, d_model, use_fp8=use_fp8)
        self.value_fc2 = _linear(d_model, 1, use_fp8=use_fp8)

        # Score head: mean pool → MLP → bin classification
        num_bins = board_size * board_size * 2 + 1
        self.score_fc1 = _linear(d_model, d_model, use_fp8=use_fp8)
        self.score_fc2 = _linear(d_model, num_bins, use_fp8=use_fp8)

    def forward(self, x):
        B = x.size(0)
        n = self.board_size
        hw = n * n

        # Reshape [B, C, H, W] → [B, H*W, C] and project
        x = x.flatten(2).transpose(1, 2)                           # [B, hw, C_in]
        x = self.token_proj(x)                                      # [B, hw, d_model]

        # Add factorized position embedding
        x = x + self.row_embed(self.row_ids) + self.col_embed(self.col_ids)

        # Transformer blocks
        for block in self.blocks:
            x = block(x, self.rel_indices)

        x = self.final_norm(x)                                      # [B, hw, d_model]

        # Policy: per-token logit + pass
        p_board = self.policy_proj(x).squeeze(-1)                   # [B, hw]
        p_pass = self.pass_logit.expand(B, 1)                       # [B, 1]
        p = torch.cat([p_board, p_pass], dim=1)                     # [B, hw+1]

        # Value: mean pool → MLP → tanh
        pooled = x.mean(dim=1)                                      # [B, d_model]
        v = torch.tanh(self.value_fc2(F.gelu(self.value_fc1(pooled))))

        # Score: mean pool → MLP → bin logits
        s = self.score_fc2(F.gelu(self.score_fc1(pooled)))

        return p, v, s

    def predict(self, state_tensor, device="cpu"):
        self.eval()
        with torch.no_grad():
            x = torch.from_numpy(state_tensor).float().unsqueeze(0).to(device)
            logits, value, score_logits = self(x)
            probs = F.softmax(logits, dim=1).squeeze(0).cpu().numpy()
            board_area = self.board_size * self.board_size
            bins = torch.arange(score_logits.size(1), device=x.device).float() - board_area
            score = (F.softmax(score_logits, dim=1) * bins).sum(dim=1).item()
        return probs, value.item(), score


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


def create_model(arch="resnet", board_size=9, input_channels=17, **kwargs):
    """Factory: create model by architecture name."""
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
    else:
        raise ValueError(f"Unknown architecture: {arch}")
