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
    def __init__(self, board_size=9, input_channels=17, num_filters=64, num_res_blocks=5):
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
        self.policy_fc = nn.Linear(2 * board_size * board_size, action_size)

        self.value_conv = nn.Conv2d(num_filters, 1, 1, bias=False)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(board_size * board_size, 64)
        self.value_fc2 = nn.Linear(64, 1)

        self.score_conv = nn.Conv2d(num_filters, 1, 1, bias=False)
        self.score_bn = nn.BatchNorm2d(1)
        self.score_fc1 = nn.Linear(board_size * board_size, 64)
        self.score_fc2 = nn.Linear(64, 1)

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
        s = torch.tanh(self.score_fc2(s))

        return p, v, s

    def predict(self, state_tensor, device="cpu"):
        self.eval()
        with torch.no_grad():
            x = torch.from_numpy(state_tensor).float().unsqueeze(0).to(device)
            logits, value, score = self(x)
            probs = F.softmax(logits, dim=1).squeeze(0).cpu().numpy()
        return probs, value.item(), score.item()


# ══════════════════════════════════════════════════════════
#  GoViT — Vision Transformer for Go
# ══════════════════════════════════════════════════════════

def _build_orbit_ids(n):
    """Compute D4 orbit id for each board position. Returns [n*n] int tensor.

    orbit = index of sorted(min(r, n-1-r), min(c, n-1-c)) in the canonical list.
    For 9x9: 15 unique orbits.
    """
    half = (n - 1) / 2.0
    orbits = {}
    idx = 0
    ids = []
    for r in range(n):
        for c in range(n):
            a, b = min(r, n - 1 - r), min(c, n - 1 - c)
            key = (min(a, b), max(a, b))
            if key not in orbits:
                orbits[key] = idx
                idx += 1
            ids.append(orbits[key])
    return torch.tensor(ids, dtype=torch.long)


def _build_rel_bias_indices(n):
    """Compute D4-invariant relative bias index for all (i, j) token pairs.

    bias_bucket = index of sorted(|dx|, |dy|) in canonical list.
    For 9x9: 45 unique buckets. Returns [n*n, n*n] int tensor.
    """
    coords = []
    for r in range(n):
        for c in range(n):
            coords.append((r, c))

    buckets = {}
    idx = 0
    hw = n * n
    indices = torch.zeros(hw, hw, dtype=torch.long)

    for i, (r1, c1) in enumerate(coords):
        for j, (r2, c2) in enumerate(coords):
            adx, ady = abs(r1 - r2), abs(c1 - c2)
            key = (min(adx, ady), max(adx, ady))
            if key not in buckets:
                buckets[key] = idx
                idx += 1
            indices[i, j] = buckets[key]

    return indices, idx  # indices [hw, hw], num_buckets


class GQAAttention(nn.Module):
    """Grouped Query Attention with D4-invariant relative positional bias.

    Fused Q + fused KV projections for efficiency.
    """

    def __init__(self, d_model, num_heads, kv_groups, num_rel_buckets, head_dim=32):
        super().__init__()
        assert num_heads % kv_groups == 0
        self.num_heads = num_heads
        self.kv_groups = kv_groups
        self.head_dim = head_dim
        self.group_size = num_heads // kv_groups
        self.scale = math.sqrt(head_dim)

        self.q_proj = nn.Linear(d_model, num_heads * head_dim)
        self.kv_proj = nn.Linear(d_model, 2 * kv_groups * head_dim)
        self.out_proj = nn.Linear(num_heads * head_dim, d_model)

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

    def __init__(self, d_model, num_heads, kv_groups, mlp_ratio, num_rel_buckets, head_dim=32):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = GQAAttention(d_model, num_heads, kv_groups, num_rel_buckets, head_dim)
        self.norm2 = nn.LayerNorm(d_model)
        mlp_hidden = d_model * mlp_ratio
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
    """Vision Transformer for Go with D4-invariant positional encoding.

    Positional encoding:
      - Absolute: orbit embedding (15 classes for 9x9) — D4-invariant
      - Relative: per-head bias indexed by sorted(|dx|, |dy|) — 45 buckets for 9x9

    Attention: Grouped Query Attention (6 Q heads, 2 KV groups by default)
    """

    def __init__(self, board_size=9, input_channels=17,
                 d_model=192, depth=8, num_heads=6, kv_groups=2,
                 mlp_ratio=4, head_dim=32):
        super().__init__()
        self.board_size = board_size
        hw = board_size * board_size
        action_size = hw + 1

        # Token embedding: per-intersection linear projection
        self.token_proj = nn.Linear(input_channels, d_model)

        # Orbit embedding: D4-invariant absolute position (15 classes for 9x9)
        orbit_ids = _build_orbit_ids(board_size)
        self.register_buffer("orbit_ids", orbit_ids)
        num_orbits = orbit_ids.max().item() + 1
        self.orbit_embed = nn.Embedding(num_orbits, d_model)

        # Relative bias indices: D4-invariant (45 buckets for 9x9)
        rel_indices, num_rel_buckets = _build_rel_bias_indices(board_size)
        self.register_buffer("rel_indices", rel_indices)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, num_heads, kv_groups, mlp_ratio,
                             num_rel_buckets, head_dim)
            for _ in range(depth)
        ])
        self.final_norm = nn.LayerNorm(d_model)

        # Policy head: per-token logit + learnable pass logit
        self.policy_proj = nn.Linear(d_model, 1)
        self.pass_logit = nn.Parameter(torch.zeros(1))

        # Value head: mean pool → MLP → tanh
        self.value_fc1 = nn.Linear(d_model, d_model)
        self.value_fc2 = nn.Linear(d_model, 1)

        # Score head: mean pool → MLP → tanh
        self.score_fc1 = nn.Linear(d_model, d_model)
        self.score_fc2 = nn.Linear(d_model, 1)

    def forward(self, x):
        B = x.size(0)
        n = self.board_size
        hw = n * n

        # Reshape [B, C, H, W] → [B, H*W, C] and project
        x = x.view(B, x.size(1), hw).permute(0, 2, 1)             # [B, hw, C_in]
        x = self.token_proj(x)                                      # [B, hw, d_model]

        # Add orbit embedding
        x = x + self.orbit_embed(self.orbit_ids)                    # broadcast over B

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

        # Score: mean pool → MLP → tanh
        s = torch.tanh(self.score_fc2(F.gelu(self.score_fc1(pooled))))

        return p, v, s

    def predict(self, state_tensor, device="cpu"):
        self.eval()
        with torch.no_grad():
            x = torch.from_numpy(state_tensor).float().unsqueeze(0).to(device)
            logits, value, score = self(x)
            probs = F.softmax(logits, dim=1).squeeze(0).cpu().numpy()
        return probs, value.item(), score.item()


def create_model(arch="resnet", board_size=9, input_channels=17, **kwargs):
    """Factory: create model by architecture name."""
    if arch == "resnet":
        return AlphaZeroNet(
            board_size=board_size, input_channels=input_channels,
            num_filters=kwargs.get("num_filters", 64),
            num_res_blocks=kwargs.get("num_res_blocks", 5),
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
        )
    else:
        raise ValueError(f"Unknown architecture: {arch}")
