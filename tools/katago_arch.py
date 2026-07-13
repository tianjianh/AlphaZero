"""KataGo inference-only architecture in PyTorch (for ONNX export).

Mirrors the C++ runtime in cpp/neuralnet/desc.cpp / openclbackend.cpp /
cudabackend.cpp. The PyTorch model is built once, weights are copied
from the parsed KataGo binary (parse_katago.parse_katago_model),
then exported as ONNX via katago_to_onnx.py.

Key design choices:
  - Pre-activation BN: BN -> activation -> conv (KataGo's order).
  - Global pool: 3 stats per channel.
      gpool (used by GPoolBlock + PolicyHead's g1):
          stat0 = mean
          stat1 = mean * (sqrt(N) - 14) * 0.1
          stat2 = max
      vhpool (used by ValueHead's v1):
          stat0 = mean
          stat1 = mean * (sqrt(N) - 14) * 0.1
          stat2 = mean * ((sqrt(N) - 14)^2 * 0.01 - 0.1)
  - Heads bake post-processing into the graph (so the C++ side reads
    the same shapes/scales as MiniGo's ONNX):
      - policy_logits = concat(spatial_p2_logits[:, 0, :, :].flatten(),
                               pass_logit_from_gpool)        -> [N, H*W+1]
      - value         = softmax(v3_logits)[:,0] - softmax(v3_logits)[:,1]  -> [N, 1]
      - score_mean    = sv3_logits[:, 0] * 20.0              -> [N, 1]
      - score_stdev   = softplus(sv3_logits[:, 1]) * 20.0    -> [N, 1]
      - ownership     = (tanh(v_ownership_conv(v1_act)) + 1) * 0.5  -> [N, H*W]
"""

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ────────────────────────────────────────────────────────────
#  Helpers
# ────────────────────────────────────────────────────────────


def _act(name: str) -> nn.Module:
    """Map KataGo's activation token to a PyTorch nn.Module."""
    if name == "relu":
        return nn.ReLU(inplace=False)
    if name == "mish":
        return nn.Mish(inplace=False)
    if name == "identity":
        return nn.Identity()
    raise ValueError(f"Unknown activation: {name}")


def _load_bn(bn: nn.BatchNorm2d, kbn) -> None:
    """Load KataGo BN parameters into a PyTorch BatchNorm2d.

    KataGo's BN is `(x - mean) * scale / sqrt(var + eps) + bias`. For
    inference we copy mean / var / scale / bias directly and put the
    layer in eval mode so PyTorch uses running stats.
    """
    n = kbn.num_channels
    bn.running_mean.copy_(torch.from_numpy(kbn.mean.astype(np.float32)))
    bn.running_var.copy_(torch.from_numpy(kbn.variance.astype(np.float32)))
    bn.weight.data.copy_(torch.from_numpy(kbn.scale.astype(np.float32)))
    bn.bias.data.copy_(torch.from_numpy(kbn.bias.astype(np.float32)))
    bn.eps = float(kbn.epsilon)
    bn.num_batches_tracked.fill_(0)
    assert bn.num_features == n, f"BN channel mismatch: {bn.num_features} vs {n}"


def _load_conv(conv: nn.Conv2d, kconv) -> None:
    """Load KataGo conv weights into a PyTorch Conv2d.

    KataGo stores weights as [out_ch, in_ch, y, x] — same as PyTorch.
    """
    w = torch.from_numpy(kconv.weights.astype(np.float32))
    assert w.shape == conv.weight.shape, (
        f"Conv shape mismatch: pytorch={conv.weight.shape} kg={w.shape}"
    )
    conv.weight.data.copy_(w)


def _load_matmul(linear: nn.Linear, kmm) -> None:
    """Load KataGo MatMul as a PyTorch Linear (no bias)."""
    w = torch.from_numpy(kmm.weights.astype(np.float32))
    assert w.shape == linear.weight.shape, (
        f"Linear shape mismatch: pytorch={linear.weight.shape} kg={w.shape}"
    )
    linear.weight.data.copy_(w)


def _load_matbias(bias: nn.Parameter, kmb) -> None:
    """Load KataGo MatBias into a 1-D nn.Parameter."""
    b = torch.from_numpy(kmb.weights.astype(np.float32))
    assert b.shape == bias.shape, f"Bias shape mismatch: {bias.shape} vs {b.shape}"
    bias.data.copy_(b)


def _gpool_stats(x: torch.Tensor) -> torch.Tensor:
    """Global pool used by GPoolBlock + PolicyHead's g1 features.

    x: [N, C, H, W]
    out: [N, 3*C] = [mean, mean*(sqrt(N)-14)*0.1, max]
    """
    n, c, h, w = x.shape
    sqrt_div = math.sqrt(float(h * w))
    scale = (sqrt_div - 14.0) * 0.1
    mean = x.mean(dim=[2, 3])                    # [N, C]
    mx = x.amax(dim=[2, 3])                      # [N, C]
    return torch.cat([mean, mean * scale, mx], dim=1)


def _vhpool_stats(x: torch.Tensor) -> torch.Tensor:
    """Pool used by ValueHead's v1 features.

    x: [N, C, H, W]
    out: [N, 3*C] = [mean, mean*(sqrt(N)-14)*0.1, mean*((sqrt(N)-14)^2*0.01-0.1)]
    """
    n, c, h, w = x.shape
    sqrt_div = math.sqrt(float(h * w))
    a = (sqrt_div - 14.0) * 0.1
    b = (sqrt_div - 14.0) ** 2 * 0.01 - 0.1
    mean = x.mean(dim=[2, 3])                    # [N, C]
    return torch.cat([mean, mean * a, mean * b], dim=1)


# ────────────────────────────────────────────────────────────
#  Blocks
# ────────────────────────────────────────────────────────────


class KataGoOrdinaryBlock(nn.Module):
    """Pre-activation residual block.

    pre_bn -> pre_act -> regular_conv -> mid_bn -> mid_act -> final_conv -> + skip
    """

    def __init__(self, kblock):
        super().__init__()
        self.pre_bn = nn.BatchNorm2d(kblock.regular_conv.in_ch)
        self.pre_act = _act(kblock.pre_act.activation)
        rc = kblock.regular_conv
        self.regular_conv = nn.Conv2d(rc.in_ch, rc.out_ch, kernel_size=rc.y,
                                      padding=rc.y // 2, bias=False)
        self.mid_bn = nn.BatchNorm2d(kblock.mid_bn.num_channels)
        self.mid_act = _act(kblock.mid_act.activation)
        fc = kblock.final_conv
        self.final_conv = nn.Conv2d(fc.in_ch, fc.out_ch, kernel_size=fc.y,
                                    padding=fc.y // 2, bias=False)
        # Load weights
        _load_bn(self.pre_bn, kblock.pre_bn)
        _load_conv(self.regular_conv, kblock.regular_conv)
        _load_bn(self.mid_bn, kblock.mid_bn)
        _load_conv(self.final_conv, kblock.final_conv)

    def forward(self, x):
        skip = x
        h = self.pre_act(self.pre_bn(x))
        h = self.regular_conv(h)
        h = self.mid_act(self.mid_bn(h))
        h = self.final_conv(h)
        return skip + h


class KataGoGPoolBlock(nn.Module):
    """Pre-activation residual block with a global-pool side branch.

    pre_bn -> pre_act -> { regular_conv -> +bias  AND  gpool_conv -> gpool_bn -> gpool_act -> gpool_stats -> gpool_to_bias }
                                              ↓
                                      mid_bn -> mid_act -> final_conv -> + skip
    """

    def __init__(self, kblock):
        super().__init__()
        self.pre_bn = nn.BatchNorm2d(kblock.regular_conv.in_ch)
        self.pre_act = _act(kblock.pre_act.activation)
        rc = kblock.regular_conv
        gc = kblock.gpool_conv
        self.regular_conv = nn.Conv2d(rc.in_ch, rc.out_ch, kernel_size=rc.y,
                                      padding=rc.y // 2, bias=False)
        self.gpool_conv = nn.Conv2d(gc.in_ch, gc.out_ch, kernel_size=gc.y,
                                    padding=gc.y // 2, bias=False)
        self.gpool_bn = nn.BatchNorm2d(kblock.gpool_bn.num_channels)
        self.gpool_act = _act(kblock.gpool_act.activation)
        gpb = kblock.gpool_to_bias_mul
        self.gpool_to_bias = nn.Linear(gpb.in_ch, gpb.out_ch, bias=False)
        self.mid_bn = nn.BatchNorm2d(kblock.mid_bn.num_channels)
        self.mid_act = _act(kblock.mid_act.activation)
        fc = kblock.final_conv
        self.final_conv = nn.Conv2d(fc.in_ch, fc.out_ch, kernel_size=fc.y,
                                    padding=fc.y // 2, bias=False)
        # Load weights
        _load_bn(self.pre_bn, kblock.pre_bn)
        _load_conv(self.regular_conv, kblock.regular_conv)
        _load_conv(self.gpool_conv, kblock.gpool_conv)
        _load_bn(self.gpool_bn, kblock.gpool_bn)
        _load_matmul(self.gpool_to_bias, kblock.gpool_to_bias_mul)
        _load_bn(self.mid_bn, kblock.mid_bn)
        _load_conv(self.final_conv, kblock.final_conv)

    def forward(self, x):
        skip = x
        h = self.pre_act(self.pre_bn(x))
        regular = self.regular_conv(h)                      # [N, regular_c, H, W]
        gpool = self.gpool_act(self.gpool_bn(self.gpool_conv(h)))  # [N, gpool_c, H, W]
        gstats = _gpool_stats(gpool)                        # [N, 3*gpool_c]
        bias = self.gpool_to_bias(gstats)                   # [N, regular_c]
        regular = regular + bias.unsqueeze(-1).unsqueeze(-1)  # broadcast bias
        regular = self.mid_act(self.mid_bn(regular))
        out = self.final_conv(regular)                      # [N, trunk_c, H, W]
        return skip + out


# ────────────────────────────────────────────────────────────
#  Stem + heads
# ────────────────────────────────────────────────────────────


class KataGoStem(nn.Module):
    """Spatial conv + global features projected as channel-wise bias."""

    def __init__(self, ktrunk):
        super().__init__()
        ic = ktrunk.initial_conv
        self.initial_conv = nn.Conv2d(ic.in_ch, ic.out_ch, kernel_size=ic.y,
                                      padding=ic.y // 2, bias=False)
        im = ktrunk.initial_matmul
        self.initial_matmul = nn.Linear(im.in_ch, im.out_ch, bias=False)
        _load_conv(self.initial_conv, ic)
        _load_matmul(self.initial_matmul, im)

    def forward(self, spatial, global_features):
        # spatial: [N, in_ch, H, W]    global_features: [N, num_global]
        x = self.initial_conv(spatial)
        bias = self.initial_matmul(global_features)         # [N, trunk_c]
        return x + bias.unsqueeze(-1).unsqueeze(-1)


class KataGoPolicyHead(nn.Module):
    """Spatial logits + pass logit, concatenated as [N, H*W+1]."""

    def __init__(self, kpolicy):
        super().__init__()
        p1c = kpolicy.p1_conv
        g1c = kpolicy.g1_conv
        self.p1_conv = nn.Conv2d(p1c.in_ch, p1c.out_ch, kernel_size=p1c.y,
                                 padding=p1c.y // 2, bias=False)
        self.g1_conv = nn.Conv2d(g1c.in_ch, g1c.out_ch, kernel_size=g1c.y,
                                 padding=g1c.y // 2, bias=False)
        self.g1_bn = nn.BatchNorm2d(kpolicy.g1_bn.num_channels)
        self.g1_act = _act(kpolicy.g1_act.activation)
        gpb = kpolicy.gpool_to_bias
        self.gpool_to_bias = nn.Linear(gpb.in_ch, gpb.out_ch, bias=False)
        self.p1_bn = nn.BatchNorm2d(kpolicy.p1_bn.num_channels)
        self.p1_act = _act(kpolicy.p1_act.activation)
        p2c = kpolicy.p2_conv
        self.p2_conv = nn.Conv2d(p2c.in_ch, p2c.out_ch, kernel_size=p2c.y,
                                 padding=p2c.y // 2, bias=False)
        self.p2_out_channels = p2c.out_ch
        gpp = kpolicy.gpool_to_pass
        self.gpool_to_pass = nn.Linear(gpp.in_ch, gpp.out_ch, bias=False)
        # v15+ extras would go here; we don't need them for v8-v14.

        # Load weights
        _load_conv(self.p1_conv, p1c)
        _load_conv(self.g1_conv, g1c)
        _load_bn(self.g1_bn, kpolicy.g1_bn)
        _load_matmul(self.gpool_to_bias, gpb)
        _load_bn(self.p1_bn, kpolicy.p1_bn)
        _load_conv(self.p2_conv, p2c)
        _load_matmul(self.gpool_to_pass, gpp)

    def forward(self, trunk_out):
        # trunk_out: [N, trunk_c, H, W]  (already through trunk-tip BN+act)
        n, _, h, w = trunk_out.shape
        p1 = self.p1_conv(trunk_out)                        # [N, p1_c, H, W]
        g1 = self.g1_act(self.g1_bn(self.g1_conv(trunk_out)))  # [N, g1_c, H, W]
        gstats = _gpool_stats(g1)                           # [N, 3*g1_c]
        bias = self.gpool_to_bias(gstats)                   # [N, p1_c]
        p1 = p1 + bias.unsqueeze(-1).unsqueeze(-1)
        p1 = self.p1_act(self.p1_bn(p1))
        spatial = self.p2_conv(p1)                          # [N, out_ch, H, W]
        # KataGo's p2_conv outputs 1 channel for v8-v11, 2 for v12+. We
        # always take channel 0 (the policy for the side to move).
        spatial0 = spatial[:, 0, :, :].reshape(n, h * w)    # [N, H*W]
        pass_logit = self.gpool_to_pass(gstats)             # [N, 1]
        return torch.cat([spatial0, pass_logit], dim=1)     # [N, H*W+1]


class KataGoValueHead(nn.Module):
    """Outputs value, score_mean, score_stdev, ownership."""

    def __init__(self, kvalue, score_mean_idx=0, score_stdev_idx=1):
        super().__init__()
        v1c = kvalue.v1_conv
        self.v1_conv = nn.Conv2d(v1c.in_ch, v1c.out_ch, kernel_size=v1c.y,
                                 padding=v1c.y // 2, bias=False)
        self.v1_bn = nn.BatchNorm2d(kvalue.v1_bn.num_channels)
        self.v1_act = _act(kvalue.v1_act.activation)
        v2m = kvalue.v2_mul
        self.v2_mul = nn.Linear(v2m.in_ch, v2m.out_ch, bias=False)
        self.v2_bias = nn.Parameter(torch.zeros(kvalue.v2_bias.num_channels))
        self.v2_act = _act(kvalue.v2_act.activation)
        v3m = kvalue.v3_mul
        self.v3_mul = nn.Linear(v3m.in_ch, v3m.out_ch, bias=False)
        self.v3_bias = nn.Parameter(torch.zeros(kvalue.v3_bias.num_channels))
        sv3m = kvalue.sv3_mul
        self.sv3_mul = nn.Linear(sv3m.in_ch, sv3m.out_ch, bias=False)
        self.sv3_bias = nn.Parameter(torch.zeros(kvalue.sv3_bias.num_channels))
        voc = kvalue.v_ownership_conv
        self.v_ownership_conv = nn.Conv2d(voc.in_ch, voc.out_ch, kernel_size=voc.y,
                                          padding=voc.y // 2, bias=False)
        self.score_mean_idx = score_mean_idx
        self.score_stdev_idx = score_stdev_idx

        _load_conv(self.v1_conv, v1c)
        _load_bn(self.v1_bn, kvalue.v1_bn)
        _load_matmul(self.v2_mul, v2m)
        _load_matbias(self.v2_bias, kvalue.v2_bias)
        _load_matmul(self.v3_mul, v3m)
        _load_matbias(self.v3_bias, kvalue.v3_bias)
        _load_matmul(self.sv3_mul, sv3m)
        _load_matbias(self.sv3_bias, kvalue.sv3_bias)
        _load_conv(self.v_ownership_conv, voc)

    def forward(self, trunk_out):
        # trunk_out: [N, trunk_c, H, W]
        n, _, h, w = trunk_out.shape
        v1 = self.v1_act(self.v1_bn(self.v1_conv(trunk_out)))  # [N, v1_c, H, W]
        pooled = _vhpool_stats(v1)                           # [N, 3*v1_c]
        v2 = self.v2_act(self.v2_mul(pooled) + self.v2_bias)  # [N, v2_c]
        v3_logits = self.v3_mul(v2) + self.v3_bias            # [N, 3]  (W/L/D)
        sv3 = self.sv3_mul(v2) + self.sv3_bias                # [N, sv3_c]

        # Bake post-processing into the graph so the C++ side sees
        # MiniGo-shape outputs.
        wld = F.softmax(v3_logits, dim=1)                     # [N, 3]
        value = (wld[:, 0:1] - wld[:, 1:2])                   # [N, 1]  P(W)-P(L)

        score_mean = sv3[:, self.score_mean_idx:self.score_mean_idx + 1] * 20.0  # [N,1]
        score_stdev_raw = sv3[:, self.score_stdev_idx:self.score_stdev_idx + 1]
        score_stdev = F.softplus(score_stdev_raw) * 20.0      # [N, 1]

        own_raw = self.v_ownership_conv(v1)                   # [N, 1, H, W]
        ownership = (torch.tanh(own_raw) + 1.0) * 0.5         # in [0, 1]
        ownership = ownership.reshape(n, h * w)               # [N, H*W]

        return value, score_mean, score_stdev, ownership


# ────────────────────────────────────────────────────────────
#  Top-level
# ────────────────────────────────────────────────────────────


class KataGoNet(nn.Module):
    """Full KataGo inference graph with MiniGo-shape outputs."""

    def __init__(self, kmodel):
        super().__init__()
        self.input_channels = kmodel.num_input_channels
        self.input_global_channels = kmodel.num_input_global_channels
        self.trunk_channels = kmodel.trunk.trunk_num_channels

        # Stem
        self.stem = KataGoStem(kmodel.trunk)

        # Trunk blocks
        blocks = []
        for kind, kblk in zip(kmodel.trunk.block_kinds, kmodel.trunk.blocks):
            if kind == "regular":
                blocks.append(KataGoOrdinaryBlock(kblk))
            elif kind == "gpool":
                blocks.append(KataGoGPoolBlock(kblk))
            else:
                raise ValueError(f"Unknown block kind: {kind}")
        self.blocks = nn.ModuleList(blocks)

        # Trunk tip
        self.trunk_tip_bn = nn.BatchNorm2d(kmodel.trunk.trunk_tip_bn.num_channels)
        self.trunk_tip_act = _act(kmodel.trunk.trunk_tip_act.activation)
        _load_bn(self.trunk_tip_bn, kmodel.trunk.trunk_tip_bn)

        # Heads
        self.policy_head = KataGoPolicyHead(kmodel.policy_head)
        self.value_head = KataGoValueHead(kmodel.value_head)

        # Inference-only
        self.eval()

    def forward(self, state_spatial, state_global):
        # state_spatial: [N, in_ch, H, W]   state_global: [N, num_global]
        x = self.stem(state_spatial, state_global)
        for block in self.blocks:
            x = block(x)
        x = self.trunk_tip_act(self.trunk_tip_bn(x))
        policy_logits = self.policy_head(x)               # [N, H*W+1]
        value, score_mean, score_stdev, ownership = self.value_head(x)
        return policy_logits, value, score_mean, score_stdev, ownership
