"""
Xiangqi residual network used by the C++ MCTS engine.

The module names intentionally match the legacy C++ weight loader:
  - input_conv / input_bn
  - res_blocks.N.conv{1,2} / bn{1,2}
  - policy_conv / policy_bn / policy_fc
  - value_conv / value_bn / value_fc{1,2}
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + residual)


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

        self.res_blocks = nn.ModuleList(
            [ResidualBlock(num_filters) for _ in range(num_res_blocks)]
        )

        self.policy_conv = nn.Conv2d(num_filters, 4, 1, bias=False)
        self.policy_bn = nn.BatchNorm2d(4)
        self.policy_fc = nn.Linear(4 * self.board_area, self.action_size)

        self.value_conv = nn.Conv2d(num_filters, 2, 1, bias=False)
        self.value_bn = nn.BatchNorm2d(2)
        self.value_fc1 = nn.Linear(2 * self.board_area, 256)
        self.value_fc2 = nn.Linear(256, 1)

    def trunk(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.input_bn(self.input_conv(x)))
        for block in self.res_blocks:
            out = block(out)
        return out

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        trunk = self.trunk(x)

        policy = F.relu(self.policy_bn(self.policy_conv(trunk)))
        policy = policy.view(policy.size(0), -1)
        policy = self.policy_fc(policy)

        value = F.relu(self.value_bn(self.value_conv(trunk)))
        value = value.view(value.size(0), -1)
        value = F.relu(self.value_fc1(value))
        value = torch.tanh(self.value_fc2(value))
        return policy, value

    def forward_inference(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.forward(x)

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
