"""
MiniGo AlphaZero — Neural Network (PyTorch)
Triple-headed ResNet: policy head + value head + score head.
Shared between training and weight export.
"""

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
