"""RAP-ERRNet model wrapper.

The wrapper keeps the original ERRNet baseline untouched. It reuses
``models.arch.errnet`` as the coarse transmission backbone, then optionally
adds reflection prior prediction, gated residual adaptation, and lightweight
residual refinement.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from torch import nn
import torch.nn.functional as F

from models import arch
from models.modules.gated_blocks import ReflectionGatedResBlock
from models.modules.refinement import LightweightRefinement
from models.modules.reflection_prior import ReflectionPriorHead
from models.vgg import Vgg19


def _load_torch_checkpoint(path: Path, map_location: str | torch.device = "cpu") -> Dict[str, Any]:
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


class _GatedAdapter(nn.Module):
    """Small prior-gated adapter used when DRNet internals are left untouched."""

    def __init__(self, channels: int = 32, residual_scale: float = 0.05):
        super().__init__()
        self.residual_scale = residual_scale
        self.in_proj = nn.Sequential(nn.Conv2d(3, channels, kernel_size=3, padding=1), nn.ReLU(inplace=True))
        self.blocks = nn.ModuleList([ReflectionGatedResBlock(channels), ReflectionGatedResBlock(channels)])
        self.out_proj = nn.Conv2d(channels, 3, kernel_size=3, padding=1)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, coarse: torch.Tensor, prior: torch.Tensor) -> torch.Tensor:
        x = self.in_proj(coarse)
        for block in self.blocks:
            x = block(x, prior)
        return torch.clamp(coarse + self.residual_scale * torch.tanh(self.out_proj(x)), 0.0, 1.0)


class RAPERRNet(nn.Module):
    """Reflection-Aware Physics-guided ERRNet.

    Args:
        use_prior_head: Predict a one-channel reflection prior map. If disabled,
            a neutral all-ones prior is used so refinement can still run.
        use_gated_blocks: Add a lightweight prior-gated adapter after the
            coarse ERRNet output. The original ERRNet residual blocks are not
            modified in this compatibility implementation.
        use_refinement: Enable final residual refinement.
        use_hypercolumn_backbone: Use the original ERRNet ``--hyper`` input
            path: RGB plus VGG19 hypercolumn features, 1475 channels total.
            This is the recommended setting when loading the course pretrained
            ``--hyper`` ERRNet checkpoint.
        freeze_backbone: Freeze the reused ERRNet coarse backbone.
        pretrained_errnet_path: Optional checkpoint containing ``icnn`` weights.
        residual_scale: Scale for the final tanh-bounded residual.
    """

    def __init__(
        self,
        *,
        use_prior_head: bool = True,
        use_gated_blocks: bool = True,
        use_refinement: bool = True,
        use_hypercolumn_backbone: bool = False,
        freeze_backbone: bool = False,
        pretrained_errnet_path: Optional[str | Path] = None,
        residual_scale: float = 0.1,
    ):
        super().__init__()
        self.use_prior_head = bool(use_prior_head)
        self.use_gated_blocks = bool(use_gated_blocks)
        self.use_refinement = bool(use_refinement)
        self.use_hypercolumn_backbone = bool(use_hypercolumn_backbone)
        self.residual_scale = float(residual_scale)

        self.vgg = Vgg19(requires_grad=False) if self.use_hypercolumn_backbone else None
        backbone_in_channels = 1475 if self.use_hypercolumn_backbone else 3
        self.backbone = arch.errnet(backbone_in_channels, 3)
        if pretrained_errnet_path is not None:
            self.load_pretrained_errnet(pretrained_errnet_path)
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        self.prior_head = ReflectionPriorHead() if self.use_prior_head else None
        self.gated_adapter = _GatedAdapter() if self.use_gated_blocks else None
        self.refinement = LightweightRefinement() if self.use_refinement else None

    def load_pretrained_errnet(self, checkpoint_path: str | Path) -> None:
        path = Path(checkpoint_path)
        if not path.exists():
            raise FileNotFoundError(f"ERRNet checkpoint not found: {path}")
        checkpoint = _load_torch_checkpoint(path, map_location="cpu")
        state = checkpoint.get("icnn", checkpoint.get("state_dict", checkpoint))
        if not isinstance(state, dict):
            raise ValueError(f"Unsupported checkpoint format in {path}")

        current = self.backbone.state_dict()
        compatible = {k: v for k, v in state.items() if k in current and tuple(v.shape) == tuple(current[k].shape)}
        skipped = sorted(set(state.keys()) - set(compatible.keys()))
        self.backbone.load_state_dict(compatible, strict=False)
        if skipped:
            warnings.warn(
                f"Loaded {len(compatible)} ERRNet backbone tensors from {path}; "
                f"skipped {len(skipped)} incompatible tensors.",
                RuntimeWarning,
            )

    def _build_backbone_input(self, image: torch.Tensor) -> torch.Tensor:
        if self.vgg is None:
            return image
        self.vgg = self.vgg.to(image.device)
        self.vgg.eval()
        with torch.no_grad():
            hypercolumn = self.vgg(image)
            hypercolumn = [
                F.interpolate(feature.detach(), size=image.shape[-2:], mode="bilinear", align_corners=False)
                for feature in hypercolumn
            ]
        return torch.cat([image, *hypercolumn], dim=1)

    def forward(self, image: torch.Tensor) -> Dict[str, torch.Tensor]:
        if image.ndim != 4 or image.shape[1] != 3:
            raise ValueError(f"RAPERRNet expects input shape [B,3,H,W], got {tuple(image.shape)}.")

        image = image.clamp(0.0, 1.0)
        if self.prior_head is not None:
            prior = self.prior_head(image)
        else:
            prior = torch.ones((image.shape[0], 1, image.shape[2], image.shape[3]), dtype=image.dtype, device=image.device)

        backbone_input = self._build_backbone_input(image)
        coarse = self.backbone(backbone_input)
        if coarse.shape[-2:] != image.shape[-2:]:
            coarse = F.interpolate(coarse, size=image.shape[-2:], mode="bilinear", align_corners=False)
        coarse = coarse.clamp(0.0, 1.0)

        if self.gated_adapter is not None:
            coarse = self.gated_adapter(coarse, prior)

        residual = torch.zeros_like(coarse)
        if self.refinement is not None:
            reflection_residual = image - coarse
            refine_in = torch.cat([image, coarse, reflection_residual, prior], dim=1)
            residual = self.residual_scale * torch.tanh(self.refinement(refine_in)) * prior

        output = torch.clamp(coarse + residual, 0.0, 1.0)
        return {
            "output": output,
            "coarse": coarse,
            "prior": prior,
            "residual": residual,
        }
