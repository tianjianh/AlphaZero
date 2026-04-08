"""
MiniGo AlphaZero — Neural Network (PyTorch)

Two architectures:
  - AlphaZeroNet: triple-headed ResNet (policy + value + score)
  - GoViT: triple-headed Vision Transformer with D4-invariant positional encoding

Shared between training and weight export.
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
    """Create nn.Linear or te.Linear based on fp8 flag."""
    if use_fp8 and _te is not None:
        return _te.Linear(in_f, out_f, bias=bias)
    return nn.Linear(in_f, out_f, bias=bias)


class ResBlock(nn.Module):
    def __init__(self, num_filters):
        super().__init__()
        self.conv1 = nn.Conv2d(num_filters, num_filters, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(num_filters)
        self.conv2 = nn.Conv2d(num_filters, num_filters, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(num_filters)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + residual)


class AlphaZeroNet(nn.Module):
    def __init__(self, board_size=9, input_channels=17, num_filters=64, num_res_blocks=5,
                 use_fp8=False):
        super().__init__()
        self.board_size = board_size
        action_size = board_size * board_size + 1

        self.input_conv = nn.Conv2d(input_channels, num_filters, 3, padding=1, bias=False)
        self.input_bn = nn.BatchNorm2d(num_filters)
        self.res_blocks = nn.ModuleList(
            [ResBlock(num_filters) for _ in range(num_res_blocks)]
        )

        self.policy_conv = nn.Conv2d(num_filters, 2, 1, bias=False)
        self.policy_bn = nn.BatchNorm2d(2)
        self.policy_fc = _linear(2 * board_size * board_size, action_size, use_fp8=use_fp8)

        self.value_conv = nn.Conv2d(num_filters, 1, 1, bias=False)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = _linear(board_size * board_size, 64, use_fp8=use_fp8)
        self.value_fc2 = _linear(64, 1, use_fp8=use_fp8)

        num_bins = board_size * board_size * 2 + 1
        self.score_conv = nn.Conv2d(num_filters, 1, 1, bias=False)
        self.score_bn = nn.BatchNorm2d(1)
        self.score_fc1 = _linear(board_size * board_size, 64, use_fp8=use_fp8)
        self.score_fc2 = _linear(64, num_bins, use_fp8=use_fp8)

    def forward(self, x):
        out = F.relu(self.input_bn(self.input_conv(x)))
        for block in self.res_blocks:
            out = block(out)

        p = F.relu(self.policy_bn(self.policy_conv(out)))
        p = p.view(p.size(0), -1)
        p = self.policy_fc(p)

        v = F.relu(self.value_bn(self.value_conv(out)))
        v = v.view(v.size(0), -1)
        v = F.relu(self.value_fc1(v))
        v = torch.tanh(self.value_fc2(v))

        s = F.relu(self.score_bn(self.score_conv(out)))
        s = s.view(s.size(0), -1)
        s = F.relu(self.score_fc1(s))
        s = self.score_fc2(s)

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
    """Grouped Query Attention with D4-invariant relative positional bias.

    Fused Q + fused KV projections for efficiency.
    """

    def __init__(self, d_model, num_heads, kv_groups, num_rel_buckets, head_dim=32,
                 use_fp8=False):
        super().__init__()
        assert num_heads % kv_groups == 0
        self.num_heads = num_heads
        self.kv_groups = kv_groups
        self.head_dim = head_dim
        self.group_size = num_heads // kv_groups
        self.scale = math.sqrt(head_dim)

        self.q_proj = _linear(d_model, num_heads * head_dim, use_fp8=use_fp8)
        self.kv_proj = _linear(d_model, 2 * kv_groups * head_dim, use_fp8=use_fp8)
        self.out_proj = _linear(num_heads * head_dim, d_model, use_fp8=use_fp8)

        # Per-head relative positional bias: [num_heads, num_rel_buckets]
        self.rel_bias = nn.Parameter(torch.zeros(num_heads, num_rel_buckets))

    def forward(self, x, rel_indices):
        B, N, _ = x.shape
        H, G, d = self.num_heads, self.kv_groups, self.head_dim

        q = self.q_proj(x).view(B, N, H, d).transpose(1, 2)       # [B, H, N, d]

        kv = self.kv_proj(x).view(B, N, 2, G, d).permute(2, 0, 3, 1, 4)  # [2, B, G, N, d]
        k, v = kv[0], kv[1]                                        # [B, G, N, d] each

        # Expand KV groups: [B, G, N, d] → [B, H, N, d]
        k = k.repeat_interleave(self.group_size, dim=1)            # [B, H, N, d]
        v = v.repeat_interleave(self.group_size, dim=1)            # [B, H, N, d]

        # Attention scores
        attn = (q @ k.transpose(-2, -1)) / self.scale              # [B, H, N, N]

        # Add D4-invariant relative positional bias
        bias = self.rel_bias[:, rel_indices]                        # [H, N, N]
        attn = attn + bias.unsqueeze(0)

        attn = F.softmax(attn, dim=-1)
        out = (attn @ v).transpose(1, 2).contiguous().view(B, N, H * d)
        return self.out_proj(out)


class TransformerBlock(nn.Module):
    """Pre-norm transformer block with GQA."""

    def __init__(self, d_model, num_heads, kv_groups, mlp_ratio, num_rel_buckets, head_dim=32,
                 use_fp8=False):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = GQAAttention(d_model, num_heads, kv_groups, num_rel_buckets, head_dim,
                                  use_fp8=use_fp8)
        self.norm2 = nn.LayerNorm(d_model)
        mlp_hidden = d_model * mlp_ratio
        self.mlp = nn.Sequential(
            _linear(d_model, mlp_hidden, use_fp8=use_fp8),
            nn.GELU(),
            _linear(mlp_hidden, d_model, use_fp8=use_fp8),
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
        self.register_buffer("rel_indices", rel_indices)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, num_heads, kv_groups, mlp_ratio,
                             num_rel_buckets, head_dim, use_fp8=use_fp8)
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
        x = x.view(B, x.size(1), hw).permute(0, 2, 1)             # [B, hw, C_in]
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
