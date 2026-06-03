"""Analyze paired reflection strength and frequency type.

This is a lightweight data-analysis utility for BP-RAP v1.4/RAFA. It does not
train a model. For each paired sample it estimates:

- strength = mean(abs(input - target))
- hf_ratio = mean(abs(highpass(input - target))) / (strength + eps)

Rows are binned by dataset-level medians into weak/strong and veil/ghost groups.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any, Dict, List


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze reflection strength bins for paired datasets.")
    parser.add_argument("--data_root", default="./data")
    parser.add_argument("--datasets", default="openrr_train,zhang_train")
    parser.add_argument("--out", default="results/reflection_strength_bins.csv")
    parser.add_argument("--max_pairs", type=int, default=None)
    parser.add_argument("--crop_size", type=int, default=None)
    parser.add_argument("--max_long_edge", type=int, default=512)
    parser.add_argument("--sigma", type=float, default=3.0)
    return parser.parse_args()


def _gaussian_kernel1d(sigma: float, dtype, device):
    import torch

    sigma = max(float(sigma), 1e-3)
    radius = max(1, int(round(3.0 * sigma)))
    coords = torch.arange(-radius, radius + 1, dtype=dtype, device=device)
    kernel = torch.exp(-(coords**2) / (2.0 * sigma * sigma))
    return kernel / kernel.sum().clamp_min(1e-12)


def _gaussian_blur(image, sigma: float):
    import torch.nn.functional as F

    if image.ndim != 4:
        raise ValueError(f"expected BCHW tensor, got {tuple(image.shape)}")
    channels = image.shape[1]
    kernel = _gaussian_kernel1d(sigma, image.dtype, image.device)
    radius = kernel.numel() // 2
    kernel_x = kernel.view(1, 1, 1, -1).repeat(channels, 1, 1, 1)
    kernel_y = kernel.view(1, 1, -1, 1).repeat(channels, 1, 1, 1)
    x = F.pad(image, (radius, radius, 0, 0), mode="replicate")
    x = F.conv2d(x, kernel_x, groups=channels)
    x = F.pad(x, (0, 0, radius, radius), mode="replicate")
    return F.conv2d(x, kernel_y, groups=channels)


def _median(values: List[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return 0.5 * (ordered[mid - 1] + ordered[mid])


def _assign_bin(strength: float, hf_ratio: float, strength_mid: float, hf_mid: float) -> str:
    strength_tag = "strong" if strength >= strength_mid else "weak"
    freq_tag = "ghost" if hf_ratio >= hf_mid else "veil"
    return f"{strength_tag}_{freq_tag}"


def analyze_dataset(name: str, args: argparse.Namespace) -> List[Dict[str, Any]]:
    import torch

    from datasets.unified_reflection_dataset import UnifiedReflectionDataset

    dataset = UnifiedReflectionDataset(
        args.data_root,
        name,
        crop_size=args.crop_size,
        max_long_edge=args.max_long_edge,
        max_pairs=args.max_pairs,
    )
    rows: List[Dict[str, Any]] = []
    for index in range(len(dataset)):
        sample = dataset[index]
        diff = (sample["input"] - sample["target"]).abs().float().unsqueeze(0)
        strength = float(diff.mean())
        high = diff - _gaussian_blur(diff, args.sigma)
        hf_ratio = float(high.abs().mean() / diff.mean().clamp_min(1e-6))
        rows.append(
            {
                "dataset": name,
                "name": str(sample["name"]),
                "strength": strength,
                "hf_ratio": hf_ratio,
            }
        )
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    strength_mid = _median([float(row["strength"]) for row in rows])
    hf_mid = _median([float(row["hf_ratio"]) for row in rows])
    for row in rows:
        row["bin"] = _assign_bin(float(row["strength"]), float(row["hf_ratio"]), strength_mid, hf_mid)
        row["strength_median"] = strength_mid
        row["hf_ratio_median"] = hf_mid
    return rows


def main() -> None:
    args = parse_args()
    all_rows: List[Dict[str, Any]] = []
    for name in [item.strip() for item in args.datasets.split(",") if item.strip()]:
        all_rows.extend(analyze_dataset(name, args))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["dataset", "name", "strength", "hf_ratio", "bin", "strength_median", "hf_ratio_median"]
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)
    print(out_path)


if __name__ == "__main__":
    main()
