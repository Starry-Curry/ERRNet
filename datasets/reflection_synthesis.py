"""Physics-guided reflection synthesis for RAP-ERRNet.

This module does not read or download any dataset. It only combines two input
images/tensors into one synthetic reflection-removal training sample.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


@dataclass
class SynthesisConfig:
    beta_min: float = 0.15
    beta_max: float = 0.75
    kappa_min: float = 0.0
    kappa_max: float = 0.25
    sigma1_min: float = 1.0
    sigma1_max: float = 5.0
    sigma2_min: float = 5.0
    sigma2_max: float = 15.0
    sigma_m_min: float = 8.0
    sigma_m_max: float = 24.0
    alpha2_min: float = 0.0
    alpha2_max: float = 0.35
    shift_min: float = -8.0
    shift_max: float = 8.0
    mask_floor_min: float = 0.0
    mask_floor_max: float = 0.0
    mask_gamma_min: float = 1.0
    mask_gamma_max: float = 1.0
    noise_std: float = 0.005
    strong_prob: float = 0.0
    strong_beta_min: float = 0.55
    strong_beta_max: float = 0.95
    strong_kappa_min: float = 0.10
    strong_kappa_max: float = 0.45
    strong_sigma1_min: float = 0.3
    strong_sigma1_max: float = 2.5
    strong_sigma2_min: float = 3.0
    strong_sigma2_max: float = 12.0
    strong_sigma_m_min: float = 18.0
    strong_sigma_m_max: float = 48.0
    strong_alpha2_min: float = 0.15
    strong_alpha2_max: float = 0.70
    strong_shift_min: float = -24.0
    strong_shift_max: float = 24.0
    strong_mask_floor_min: float = 0.12
    strong_mask_floor_max: float = 0.35
    strong_mask_gamma_min: float = 0.35
    strong_mask_gamma_max: float = 0.75

    @classmethod
    def from_mapping(cls, config: Optional[Mapping[str, Any]]) -> "SynthesisConfig":
        if config is None:
            return cls()
        valid = {field.name for field in cls.__dataclass_fields__.values()}
        kwargs = {key: value for key, value in dict(config).items() if key in valid}
        return cls(**kwargs)


def _image_to_tensor(image) -> torch.Tensor:
    if isinstance(image, Image.Image):
        arr = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
        return torch.from_numpy(arr).permute(2, 0, 1).contiguous()
    if isinstance(image, torch.Tensor):
        tensor = image.detach().float().cpu()
        if tensor.ndim == 4:
            if tensor.shape[0] != 1:
                raise ValueError(f"Expected a single image tensor, got shape {tuple(tensor.shape)}.")
            tensor = tensor[0]
        if tensor.ndim != 3:
            raise ValueError(f"Expected tensor shape [3,H,W] or [1,3,H,W], got {tuple(tensor.shape)}.")
        if tensor.shape[0] != 3 and tensor.shape[-1] == 3:
            tensor = tensor.permute(2, 0, 1)
        if tensor.shape[0] != 3:
            raise ValueError(f"Expected RGB tensor with 3 channels, got {tuple(tensor.shape)}.")
        return tensor.clamp(0.0, 1.0)
    raise TypeError(f"Unsupported image type: {type(image)!r}. Use PIL.Image or torch.Tensor.")


def _gaussian_kernel1d(sigma: float, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    sigma = max(float(sigma), 1e-3)
    radius = max(1, int(math.ceil(3.0 * sigma)))
    coords = torch.arange(-radius, radius + 1, dtype=dtype, device=device)
    kernel = torch.exp(-(coords**2) / (2 * sigma * sigma))
    return kernel / kernel.sum()


def _gaussian_blur(image: torch.Tensor, sigma: float) -> torch.Tensor:
    """Blur a CHW tensor using depth-wise separable Gaussian convolution."""

    if sigma <= 0:
        return image
    c, h, w = image.shape
    kernel = _gaussian_kernel1d(sigma, image.dtype, image.device)
    radius = kernel.numel() // 2
    x = image.unsqueeze(0)
    pad_mode = "replicate"

    kernel_x = kernel.view(1, 1, 1, -1).repeat(c, 1, 1, 1)
    kernel_y = kernel.view(1, 1, -1, 1).repeat(c, 1, 1, 1)
    x = F.pad(x, (radius, radius, 0, 0), mode=pad_mode)
    x = F.conv2d(x, kernel_x, groups=c)
    x = F.pad(x, (0, 0, radius, radius), mode=pad_mode)
    x = F.conv2d(x, kernel_y, groups=c)
    return x.squeeze(0)


def _translate(image: torch.Tensor, dx: float, dy: float) -> torch.Tensor:
    """Translate a CHW tensor by dx/dy pixels using bilinear sampling."""

    c, h, w = image.shape
    theta = torch.tensor(
        [[[1.0, 0.0, -2.0 * dx / max(w - 1, 1)], [0.0, 1.0, -2.0 * dy / max(h - 1, 1)]]],
        dtype=image.dtype,
        device=image.device,
    )
    grid = F.affine_grid(theta, size=(1, c, h, w), align_corners=False)
    return F.grid_sample(image.unsqueeze(0), grid, mode="bilinear", padding_mode="border", align_corners=False).squeeze(0)


def _normalize_mask(mask: torch.Tensor) -> torch.Tensor:
    mask_min = mask.amin(dim=(-2, -1), keepdim=True)
    mask_max = mask.amax(dim=(-2, -1), keepdim=True)
    denom = (mask_max - mask_min).clamp_min(1e-6)
    return ((mask - mask_min) / denom).clamp(0.0, 1.0)


def _uniform(generator: torch.Generator, low: float, high: float) -> float:
    if float(high) <= float(low):
        return float(low)
    return float(torch.empty(()).uniform_(float(low), float(high), generator=generator).item())


def _range_value(cfg: SynthesisConfig, generator: torch.Generator, name: str, *, strong: bool) -> float:
    prefix = "strong_" if strong else ""
    low = getattr(cfg, f"{prefix}{name}_min")
    high = getattr(cfg, f"{prefix}{name}_max")
    return _uniform(generator, low, high)


def synthesize_reflection_pair(
    transmission,
    reflection,
    seed: Optional[int] = None,
    config: Optional[Mapping[str, Any]] = None,
) -> Dict[str, torch.Tensor]:
    """Create a synthetic reflection pair.

    Args:
        transmission: PIL image or tensor [3,H,W] in [0, 1].
        reflection: PIL image or tensor [3,H,W] in [0, 1].
        seed: Optional deterministic seed for all sampled parameters.
        config: Optional parameter overrides matching :class:`SynthesisConfig`.

    Returns:
        Dict with ``input``, ``target``, ``reflection`` and ``mask`` tensors.
    """

    cfg = SynthesisConfig.from_mapping(config)
    generator = torch.Generator(device="cpu")
    if seed is not None:
        generator.manual_seed(int(seed))
    else:
        generator.seed()

    t = _image_to_tensor(transmission)
    r = _image_to_tensor(reflection)
    _, h, w = t.shape
    if r.shape[-2:] != (h, w):
        r = F.interpolate(r.unsqueeze(0), size=(h, w), mode="bilinear", align_corners=False).squeeze(0)

    strong = _uniform(generator, 0.0, 1.0) < float(cfg.strong_prob)
    beta = _range_value(cfg, generator, "beta", strong=strong)
    kappa = _range_value(cfg, generator, "kappa", strong=strong)
    sigma1 = _range_value(cfg, generator, "sigma1", strong=strong)
    sigma2 = _range_value(cfg, generator, "sigma2", strong=strong)
    sigma_m = _range_value(cfg, generator, "sigma_m", strong=strong)
    alpha2 = _range_value(cfg, generator, "alpha2", strong=strong)
    shift_min = cfg.strong_shift_min if strong else cfg.shift_min
    shift_max = cfg.strong_shift_max if strong else cfg.shift_max
    dx = _uniform(generator, shift_min, shift_max)
    dy = _uniform(generator, shift_min, shift_max)
    mask_floor = _range_value(cfg, generator, "mask_floor", strong=strong)
    mask_gamma = _range_value(cfg, generator, "mask_gamma", strong=strong)

    r_blur_1 = _gaussian_blur(r, sigma1)
    r_blur_2 = _gaussian_blur(r, sigma2)
    r_ghost = _translate(r_blur_2, dx=dx, dy=dy)
    r_mixed = (r_blur_1 + alpha2 * r_ghost).clamp(0.0, 1.0)

    noise_map = torch.rand((1, h, w), generator=generator, dtype=t.dtype)
    mask = _normalize_mask(_gaussian_blur(noise_map, sigma_m))
    mask = mask.clamp(0.0, 1.0).pow(max(float(mask_gamma), 1e-3))
    if mask_floor > 0:
        mask = (float(mask_floor) + (1.0 - float(mask_floor)) * mask).clamp(0.0, 1.0)
    noise = torch.randn(t.shape, generator=generator, dtype=t.dtype) * float(cfg.noise_std)

    blended = ((1.0 - kappa * mask) * t + beta * mask * r_mixed + noise).clamp(0.0, 1.0)

    return {
        "input": blended.contiguous(),
        "target": t.contiguous(),
        "reflection": r_mixed.contiguous(),
        "mask": mask.contiguous(),
        "hard_synth": torch.tensor([1.0 if strong else 0.0], dtype=t.dtype),
    }
