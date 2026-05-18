"""Reflection-aware residual blocks for RAP-ERRNet."""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class _ResidualBranch(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x)


class ReflectionGatedResBlock(nn.Module):
    """Residual block modulated by a reflection prior map.

    Computes ``F' = F + gamma * resize(P) * ResBlock(F)`` when ``prior`` is
    provided. If ``prior`` is ``None``, it falls back to a standard residual
    block ``F' = F + ResBlock(F)`` for easy ablations and compatibility.
    """

    def __init__(self, channels: int):
        super().__init__()
        self.block = _ResidualBranch(channels)
        self.gamma = nn.Parameter(torch.tensor(0.0))

    def forward(self, x: torch.Tensor, prior: torch.Tensor | None = None) -> torch.Tensor:
        residual = self.block(x)
        if prior is None:
            return x + residual

        if prior.ndim == 3:
            prior = prior.unsqueeze(1)
        if prior.ndim != 4:
            raise ValueError(f"Expected prior shape [B,1,H,W], got {tuple(prior.shape)}.")
        if prior.shape[1] != 1:
            prior = prior.mean(dim=1, keepdim=True)
        prior_resized = F.interpolate(prior, size=x.shape[-2:], mode="bilinear", align_corners=False)
        return x + self.gamma * prior_resized * residual

