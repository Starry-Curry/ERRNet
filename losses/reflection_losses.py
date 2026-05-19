"""Loss functions for RAP-ERRNet training."""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import torch
from torch import nn
import torch.nn.functional as F


def _get_config_value(config: Mapping[str, Any], key: str, default: Any) -> Any:
    if key in config:
        return config[key]
    loss_cfg = config.get("loss", {}) if isinstance(config, Mapping) else {}
    if isinstance(loss_cfg, Mapping) and key in loss_cfg:
        return loss_cfg[key]
    return default


def _get_batch_tensor(batch: Mapping[str, Any], *names: str) -> Optional[torch.Tensor]:
    for name in names:
        value = batch.get(name)
        if isinstance(value, torch.Tensor):
            return value
    return None


def normalize_mask(mask: torch.Tensor) -> torch.Tensor:
    if mask.ndim == 3:
        mask = mask.unsqueeze(1)
    if mask.shape[1] != 1:
        mask = mask.max(dim=1, keepdim=True).values
    min_val = mask.amin(dim=(-2, -1), keepdim=True)
    max_val = mask.amax(dim=(-2, -1), keepdim=True)
    return ((mask - min_val) / (max_val - min_val).clamp_min(1e-6)).clamp(0.0, 1.0)


def pseudo_mask_from_pair(input_image: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return normalize_mask((input_image - target).abs().max(dim=1, keepdim=True).values)


def reflection_weighted_l1(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    lambda_reflection_weight: float = 2.0,
) -> torch.Tensor:
    mask = normalize_mask(mask).to(device=pred.device, dtype=pred.dtype)
    return torch.mean(torch.abs((1.0 + lambda_reflection_weight * mask) * (pred - target)))


def gradient_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred_dx = pred[..., 1:, :] - pred[..., :-1, :]
    pred_dy = pred[..., 1:] - pred[..., :-1]
    target_dx = target[..., 1:, :] - target[..., :-1, :]
    target_dy = target[..., 1:] - target[..., :-1]
    return F.l1_loss(pred_dx, target_dx) + F.l1_loss(pred_dy, target_dy)


def ssim_loss(pred: torch.Tensor, target: torch.Tensor, window_size: int = 11, data_range: float = 1.0) -> torch.Tensor:
    """Differentiable SSIM loss using local average windows."""

    padding = window_size // 2
    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2

    mu_x = F.avg_pool2d(pred, window_size, stride=1, padding=padding)
    mu_y = F.avg_pool2d(target, window_size, stride=1, padding=padding)
    sigma_x = F.avg_pool2d(pred * pred, window_size, stride=1, padding=padding) - mu_x * mu_x
    sigma_y = F.avg_pool2d(target * target, window_size, stride=1, padding=padding) - mu_y * mu_y
    sigma_xy = F.avg_pool2d(pred * target, window_size, stride=1, padding=padding) - mu_x * mu_y

    numerator = (2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)
    denominator = (mu_x.pow(2) + mu_y.pow(2) + c1) * (sigma_x + sigma_y + c2)
    ssim_map = numerator / denominator.clamp_min(1e-6)
    return 1.0 - ssim_map.clamp(0.0, 1.0).mean()


def mask_loss(prior: torch.Tensor, mask: torch.Tensor, sample_weight: Optional[torch.Tensor] = None) -> torch.Tensor:
    target = normalize_mask(mask).to(device=prior.device, dtype=prior.dtype)
    pred = prior.clamp(1e-6, 1.0 - 1e-6)
    bce = F.binary_cross_entropy(pred, target, reduction="none").mean(dim=(1, 2, 3))
    dice = (2.0 * (pred * target).sum(dim=(1, 2, 3)) + 1e-6) / (
        pred.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3)) + 1e-6
    )
    loss = bce + 0.5 * (1.0 - dice)
    if sample_weight is not None:
        weight = sample_weight.to(device=prior.device, dtype=prior.dtype).reshape(-1)
        if weight.numel() != loss.numel():
            raise ValueError(f"mask sample_weight has {weight.numel()} values, expected {loss.numel()}.")
        return (loss * weight).sum() / weight.sum().clamp_min(1e-6)
    return loss.mean()


def clean_consistency_loss(input_image: torch.Tensor, pred: torch.Tensor, prior: torch.Tensor) -> torch.Tensor:
    prior = prior.detach().to(device=pred.device, dtype=pred.dtype)
    return torch.mean(torch.abs((1.0 - prior) * (input_image - pred)))


def baseline_anchor_loss(pred: torch.Tensor, anchor: torch.Tensor, prior: torch.Tensor) -> torch.Tensor:
    """Keep low-prior regions close to the pretrained backbone output."""

    prior = prior.detach().to(device=pred.device, dtype=pred.dtype)
    anchor = anchor.detach().to(device=pred.device, dtype=pred.dtype).clamp(0.0, 1.0)
    if anchor.shape[-2:] != pred.shape[-2:]:
        anchor = F.interpolate(anchor, size=pred.shape[-2:], mode="bilinear", align_corners=False)
    return torch.mean(torch.abs((1.0 - prior) * (pred - anchor)))


def residual_magnitude_loss(pred: torch.Tensor, anchor: torch.Tensor) -> torch.Tensor:
    """Discourage unnecessary global changes relative to the backbone output."""

    anchor = anchor.detach().to(device=pred.device, dtype=pred.dtype).clamp(0.0, 1.0)
    if anchor.shape[-2:] != pred.shape[-2:]:
        anchor = F.interpolate(anchor, size=pred.shape[-2:], mode="bilinear", align_corners=False)
    return torch.mean(torch.abs(pred - anchor))


def exclusion_loss(input_image: torch.Tensor, pred: torch.Tensor, level: float = 1.0) -> torch.Tensor:
    reflection = input_image - pred
    pred_dx = pred[..., 1:, :] - pred[..., :-1, :]
    pred_dy = pred[..., 1:] - pred[..., :-1]
    refl_dx = reflection[..., 1:, :] - reflection[..., :-1, :]
    refl_dy = reflection[..., 1:] - reflection[..., :-1]
    loss_x = torch.abs(torch.tanh(level * pred_dx.abs()) * torch.tanh(level * refl_dx.abs())).mean()
    loss_y = torch.abs(torch.tanh(level * pred_dy.abs()) * torch.tanh(level * refl_dy.abs())).mean()
    return loss_x + loss_y


class OptionalPerceptualLoss(nn.Module):
    """VGG perceptual loss that never downloads weights automatically."""

    def __init__(self, weight: float = 0.0, weights_path: Optional[str | Path] = None):
        super().__init__()
        self.enabled = False
        self.features = None
        self.criterion = nn.L1Loss()
        self.slices = (4, 9, 18, 27)

        if weight <= 0:
            return
        if weights_path is None:
            warnings.warn(
                "lambda_perc > 0 but no local VGG weights path was provided; perceptual loss is disabled "
                "to avoid automatic downloads.",
                RuntimeWarning,
            )
            return

        try:
            from torchvision import models
        except Exception as exc:
            warnings.warn(f"torchvision is unavailable; perceptual loss is disabled: {exc}", RuntimeWarning)
            return

        path = Path(weights_path)
        if not path.exists():
            warnings.warn(f"Local VGG weights not found at {path}; perceptual loss is disabled.", RuntimeWarning)
            return

        vgg = models.vgg19(weights=None).features
        state = torch.load(path, map_location="cpu")
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        try:
            vgg.load_state_dict(state, strict=False)
        except Exception as exc:
            warnings.warn(f"Failed to load VGG weights from {path}; perceptual loss is disabled: {exc}", RuntimeWarning)
            return
        for param in vgg.parameters():
            param.requires_grad = False
        self.features = vgg.eval()
        self.enabled = True

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if not self.enabled or self.features is None:
            return pred.new_tensor(0.0)
        self.features = self.features.to(pred.device)
        loss = pred.new_tensor(0.0)
        x = pred
        y = target
        for i, layer in enumerate(self.features):
            x = layer(x)
            y = layer(y)
            if i in self.slices:
                loss = loss + self.criterion(x, y.detach())
            if i >= self.slices[-1]:
                break
        return loss


class ReflectionRemovalLoss(nn.Module):
    """Weighted loss bundle for RAP-ERRNet.

    ``outputs`` must contain at least ``output`` and usually ``prior``.
    ``batch`` should contain ``target`` or ``target_t``; ``input`` and ``mask``
    are used for mask and clean-region terms.
    """

    def __init__(self, config: Mapping[str, Any]):
        super().__init__()
        self.lambda_pix = float(_get_config_value(config, "lambda_pix", 1.0))
        self.lambda_perc = float(_get_config_value(config, "lambda_perc", 0.0))
        self.lambda_grad = float(_get_config_value(config, "lambda_grad", 0.1))
        self.lambda_ssim = float(_get_config_value(config, "lambda_ssim", 0.2))
        self.lambda_mask = float(_get_config_value(config, "lambda_mask", 0.05))
        self.lambda_mask_pseudo_scale = float(_get_config_value(config, "lambda_mask_pseudo_scale", 1.0))
        self.lambda_clean = float(_get_config_value(config, "lambda_clean", 0.05))
        self.lambda_anchor = float(_get_config_value(config, "lambda_anchor", 0.0))
        self.lambda_delta = float(_get_config_value(config, "lambda_delta", 0.0))
        self.lambda_excl = float(_get_config_value(config, "lambda_excl", 0.0))
        self.lambda_reflection_weight = float(_get_config_value(config, "lambda_reflection_weight", 2.0))
        self.perceptual = OptionalPerceptualLoss(
            weight=self.lambda_perc,
            weights_path=_get_config_value(config, "vgg_weights_path", None),
        )

    def forward(self, outputs: Mapping[str, torch.Tensor], batch: Mapping[str, Any]) -> Dict[str, torch.Tensor]:
        pred = outputs["output"]
        target = _get_batch_tensor(batch, "target", "target_t")
        if target is None:
            raise KeyError("ReflectionRemovalLoss requires batch['target'] or batch['target_t'].")
        target = target.to(device=pred.device, dtype=pred.dtype).clamp(0.0, 1.0)

        input_image = _get_batch_tensor(batch, "input")
        if input_image is not None:
            input_image = input_image.to(device=pred.device, dtype=pred.dtype).clamp(0.0, 1.0)
        else:
            input_image = target

        prior = outputs.get("prior")
        if prior is None:
            prior = torch.ones((pred.shape[0], 1, pred.shape[2], pred.shape[3]), device=pred.device, dtype=pred.dtype)
        prior = prior.to(device=pred.device, dtype=pred.dtype).clamp(0.0, 1.0)

        mask = _get_batch_tensor(batch, "mask", "prior_mask")
        if mask is None:
            mask = pseudo_mask_from_pair(input_image, target)
        mask = mask.to(device=pred.device, dtype=pred.dtype)

        pix = reflection_weighted_l1(pred, target, mask, self.lambda_reflection_weight)
        perc = self.perceptual(pred, target)
        grad = gradient_loss(pred, target)
        ssim = ssim_loss(pred, target)
        mask_reliable = _get_batch_tensor(batch, "mask_reliable")
        mask_weight = None
        if mask_reliable is not None:
            mask_reliable = mask_reliable.to(device=pred.device, dtype=pred.dtype)
            mask_weight = mask_reliable + (1.0 - mask_reliable) * self.lambda_mask_pseudo_scale
        msk = mask_loss(prior, mask, sample_weight=mask_weight)
        clean = clean_consistency_loss(input_image, pred, prior)
        anchor_base = outputs.get("backbone_output", outputs.get("coarse", pred))
        anchor = baseline_anchor_loss(pred, anchor_base, prior) if self.lambda_anchor > 0 else pred.new_tensor(0.0)
        delta = residual_magnitude_loss(pred, anchor_base) if self.lambda_delta > 0 else pred.new_tensor(0.0)
        excl = exclusion_loss(input_image, pred) if self.lambda_excl > 0 else pred.new_tensor(0.0)

        total = (
            self.lambda_pix * pix
            + self.lambda_perc * perc
            + self.lambda_grad * grad
            + self.lambda_ssim * ssim
            + self.lambda_mask * msk
            + self.lambda_clean * clean
            + self.lambda_anchor * anchor
            + self.lambda_delta * delta
            + self.lambda_excl * excl
        )
        return {
            "total": total,
            "pix": pix,
            "perc": perc,
            "grad": grad,
            "ssim": ssim,
            "mask": msk,
            "clean": clean,
            "anchor": anchor,
            "delta": delta,
            "excl": excl,
        }
