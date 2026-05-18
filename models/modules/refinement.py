"""Lightweight residual refinement head for RAP-ERRNet."""

from __future__ import annotations

import torch
from torch import nn


class _ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.body(x)


class LightweightRefinement(nn.Module):
    """Predict a local RGB correction from ``concat([I, T0, R0, P])``.

    The expected input has 10 channels: image, coarse transmission, residual,
    and one-channel reflection prior. The output is an unconstrained RGB delta;
    the caller should apply the tanh/prior/residual-scale constraint.
    """

    def __init__(self, in_channels: int = 10, hidden_channels: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            _ResidualBlock(hidden_channels),
            _ResidualBlock(hidden_channels),
            nn.Conv2d(hidden_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 3, kernel_size=3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4 or x.shape[1] != 10:
            raise ValueError(f"LightweightRefinement expects [B,10,H,W], got {tuple(x.shape)}.")
        return self.net(x)

