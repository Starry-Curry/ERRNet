"""Check that BP-RAP zero-residual initialization preserves ERRNet output.

Run this on the training server after pulling the code and preparing the course
checkpoint. The script uses a random tensor only; it does not read datasets.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sanity check BP-RAP zero-residual initialization.")
    parser.add_argument("--config", default="configs/bp_rap_hyper_zerores_staged.yaml")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--height", type=int, default=224)
    parser.add_argument("--width", type=int, default=224)
    parser.add_argument("--tolerance", type=float, default=1.0e-6)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    import torch

    from models.rap_errnet import RAPERRNet
    from train_rap_errnet import choose_device, load_config

    cfg = load_config(Path(args.config))
    model_cfg = dict(cfg.get("model", {}))
    device = choose_device(args.device)
    model = RAPERRNet(
        use_prior_head=bool(model_cfg.get("use_prior_head", True)),
        use_gated_blocks=bool(model_cfg.get("use_gated_blocks", True)),
        use_refinement=bool(model_cfg.get("use_refinement", True)),
        use_hypercolumn_backbone=bool(model_cfg.get("use_hypercolumn_backbone", False)),
        freeze_backbone=bool(model_cfg.get("freeze_backbone", False)),
        pretrained_errnet_path=model_cfg.get("pretrained_errnet_path"),
        residual_scale=float(model_cfg.get("residual_scale", 0.1)),
    ).to(device)
    model.eval()

    x = torch.rand(1, 3, args.height, args.width, device=device)
    with torch.no_grad():
        outputs = model(x)
    output = outputs["output"]
    backbone_output = outputs["backbone_output"]
    residual = outputs["residual"]
    max_output_delta = torch.max(torch.abs(output - backbone_output)).item()
    max_residual = torch.max(torch.abs(residual)).item()

    print(f"max_abs(output - backbone_output): {max_output_delta:.8g}")
    print(f"max_abs(residual): {max_residual:.8g}")
    if max_output_delta > args.tolerance or max_residual > args.tolerance:
        raise SystemExit(
            "Zero-residual sanity check failed. The RAP branches are not preserving the pretrained backbone output."
        )
    print("Zero-residual sanity check passed.")


if __name__ == "__main__":
    main()
