"""
Xiangqi residual network (KataGo-style).

Trunk alternates two residual block kinds (matches the multi-gpu Go branch):
  even index  -> ResBlockSE     (squeeze-and-excitation channel attention)
  odd index   -> GPoolResBlock  (parallel global-pool branch injecting biases)

The value head emits 3-class WLD logits (win / draw / loss).  For inference /
ONNX export these are collapsed to a single P(win) - P(loss) scalar so the
C++ NNOutput.value keeps its existing interpretation.

Weight names mirror the C++ loader expectations:
    input_conv / input_bn
    res_blocks.N.conv1 / bn1 / conv2 / bn2
    res_blocks.N.se.fc1 / fc2                    (SE blocks only)
    res_blocks.N.pool_conv / pool_bn / pool_fc   (GPool blocks only)
    policy_conv / policy_bn / policy_fc
    value_conv / value_bn / value_fc1 / value_fc2    (fc2: 256 -> 3 WLD)
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class SEModule(nn.Module):
    """Squeeze-and-excitation channel gate."""

    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        hidden = max(4, channels // reduction)
        self.fc1 = nn.Linear(channels, hidden)
        self.fc2 = nn.Linear(hidden, channels)
        # Zero-init the gate so the block behaves like identity at step 0.
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, H, W]
        s = x.mean(dim=(2, 3))              # [B, C]
        s = F.relu(self.fc1(s))
        gate = torch.sigmoid(self.fc2(s))   # [B, C]
        return x * gate.unsqueeze(-1).unsqueeze(-1)


class ResBlockSE(nn.Module):
    """Residual block with squeeze-and-excitation."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.se = SEModule(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.relu(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        y = self.se(y)
        return F.relu(x + y)


class GPoolResBlock(nn.Module):
    """Residual block with a parallel global-pool branch injecting biases."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)

        self.pool_conv = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.pool_bn = nn.BatchNorm2d(channels)
        self.pool_fc = nn.Linear(2 * channels, channels)
        # Zero-init so the pool injection is identity at step 0.
        nn.init.zeros_(self.pool_fc.weight)
        nn.init.zeros_(self.pool_fc.bias)

        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.relu(self.bn1(self.conv1(x)))

        p = F.relu(self.pool_bn(self.pool_conv(x)))
        stats = torch.cat([p.mean(dim=(2, 3)), p.amax(dim=(2, 3))], dim=1)  # [B, 2C]
        bias = self.pool_fc(stats).unsqueeze(-1).unsqueeze(-1)              # [B, C, 1, 1]

        y = self.bn2(self.conv2(y + bias))
        return F.relu(x + y)


class XiangqiNet(nn.Module):
    def __init__(
        self,
        board_rows: int = 10,
        board_cols: int = 9,
        input_channels: int = 57,
        num_filters: int = 128,
        num_res_blocks: int = 10,
    ):
        super().__init__()
        self.board_rows = board_rows
        self.board_cols = board_cols
        self.board_area = board_rows * board_cols
        self.action_size = self.board_area * self.board_area

        self.input_conv = nn.Conv2d(input_channels, num_filters, 3, padding=1, bias=False)
        self.input_bn = nn.BatchNorm2d(num_filters)

        # Alternate SE (even) and GPool (odd) blocks, matching the Go branch.
        blocks = []
        for i in range(num_res_blocks):
            if i % 2 == 0:
                blocks.append(ResBlockSE(num_filters))
            else:
                blocks.append(GPoolResBlock(num_filters))
        self.res_blocks = nn.ModuleList(blocks)

        self.policy_conv = nn.Conv2d(num_filters, 4, 1, bias=False)
        self.policy_bn = nn.BatchNorm2d(4)
        self.policy_fc = nn.Linear(4 * self.board_area, self.action_size)

        self.value_conv = nn.Conv2d(num_filters, 2, 1, bias=False)
        self.value_bn = nn.BatchNorm2d(2)
        self.value_fc1 = nn.Linear(2 * self.board_area, 256)
        # 3-class WLD logits (index 0 = win, 1 = draw, 2 = loss).
        self.value_fc2 = nn.Linear(256, 3)

    def trunk(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.input_bn(self.input_conv(x)))
        for block in self.res_blocks:
            out = block(out)
        return out

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Training forward — returns (policy_logits, wdl_logits)."""
        trunk = self.trunk(x)

        policy = F.relu(self.policy_bn(self.policy_conv(trunk)))
        policy = policy.view(policy.size(0), -1)
        policy = self.policy_fc(policy)

        value = F.relu(self.value_bn(self.value_conv(trunk)))
        value = value.view(value.size(0), -1)
        value = F.relu(self.value_fc1(value))
        wdl_logits = self.value_fc2(value)
        return policy, wdl_logits

    def forward_inference(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Inference forward — returns (policy_logits, scalar_value).

        The scalar value is P(win) - P(loss) after softmax over the WDL head.
        This matches the single-float value the C++ NNOutput expects.
        """
        policy_logits, wdl_logits = self.forward(x)
        wdl = F.softmax(wdl_logits, dim=-1)
        value = (wdl[..., 0] - wdl[..., 2]).unsqueeze(-1)
        return policy_logits, value

    def predict(self, state_tensor: np.ndarray, device: str = "cpu"):
        self.eval()
        with torch.no_grad():
            x = torch.from_numpy(state_tensor).float().unsqueeze(0).to(device)
            policy_logits, value = self.forward_inference(x)
            probs = F.softmax(policy_logits, dim=1).squeeze(0).cpu().numpy()
            return probs, float(value.item())


def create_model(
    arch: str = "resnet",
    board_rows: int = 10,
    board_cols: int = 9,
    input_channels: int = 57,
    num_filters: int = 128,
    num_res_blocks: int = 10,
    **_: object,
) -> XiangqiNet:
    if arch != "resnet":
        raise ValueError("Only the Xiangqi residual network is supported in this port")
    return XiangqiNet(
        board_rows=board_rows,
        board_cols=board_cols,
        input_channels=input_channels,
        num_filters=num_filters,
        num_res_blocks=num_res_blocks,
    )
