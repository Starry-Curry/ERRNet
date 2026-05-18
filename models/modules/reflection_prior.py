"""Reflection prior prediction head for RAP-ERRNet."""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


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


class _ConvRelu(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, *, stride: int = 1):
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
            nn.ReLU(inplace=True),
        )


class ReflectionPriorHead(nn.Module):
    """Lightweight U-Net that predicts a reflection probability map.

    Input is an RGB tensor ``I`` with shape ``[B, 3, H, W]`` and range [0, 1].
    Output is ``P`` with shape ``[B, 1, H, W]`` and range [0, 1].
    """

    def __init__(self, in_channels: int = 3, base_channels: int = 32):
        super().__init__()
        c1 = base_channels
        c2 = base_channels * 2
        c3 = base_channels * 4

        self.enc1 = _ConvRelu(in_channels, c1)
        self.enc2 = _ConvRelu(c1, c2, stride=2)
        self.enc3 = _ConvRelu(c2, c3, stride=2)
        self.bottleneck = nn.Sequential(_ResidualBlock(c3), _ResidualBlock(c3))

        self.up1 = _ConvRelu(c3, c2)
        self.merge1 = _ConvRelu(c2 + c2, c2)
        self.up2 = _ConvRelu(c2, c1)
        self.merge2 = _ConvRelu(c1 + c1, c1)
        self.out = nn.Sequential(nn.Conv2d(c1, 1, kernel_size=3, padding=1), nn.Sigmoid())

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        if image.ndim != 4 or image.shape[1] != 3:
            raise ValueError(f"ReflectionPriorHead expects [B,3,H,W], got {tuple(image.shape)}.")

        skip1 = self.enc1(image)
        skip2 = self.enc2(skip1)
        x = self.enc3(skip2)
        x = self.bottleneck(x)

        x = F.interpolate(x, size=skip2.shape[-2:], mode="bilinear", align_corners=False)
        x = self.up1(x)
        x = self.merge1(torch.cat([x, skip2], dim=1))

        x = F.interpolate(x, size=skip1.shape[-2:], mode="bilinear", align_corners=False)
        x = self.up2(x)
        x = self.merge2(torch.cat([x, skip1], dim=1))
        return self.out(x)

