"""Evaluate ERRNet/RAP adaptive or constant inference fusion.

The script runs both ERRNet and a RAP/RAFA checkpoint, then evaluates

    output = alpha * RAP + (1 - alpha) * ERRNet

for a list of constant alphas. It also records per-sample features that are
useful for designing an input-driven selector without using dataset names.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


DEFAULT_DATASETS = "ceilnet,zhang20,sir2_objects,sir2_postcard,sir2_wild,openrr_val"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate constant/adaptive ERRNet-RAP fusion.")
    parser.add_argument("--data_root", default="./data")
    parser.add_argument("--datasets", default=DEFAULT_DATASETS)
    parser.add_argument("--save_dir", required=True)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--max_long_edge", type=int, default=None)
    parser.add_argument("--errnet_ckpt", required=True)
    parser.add_argument("--errnet_hyper", action="store_true")
    parser.add_argument("--rafa_ckpt", required=True)
    parser.add_argument("--rafa_config", required=True)
    parser.add_argument("--alphas", default="0,0.25,0.5,0.75,1.0")
    parser.add_argument("--include_adaptive", action="store_true")
    parser.add_argument("--adaptive_base", type=float, default=0.85)
    parser.add_argument("--adaptive_min", type=float, default=0.20)
    parser.add_argument("--adaptive_max", type=float, default=1.00)
    parser.add_argument("--prior_area_threshold", type=float, default=0.55)
    parser.add_argument("--prior_area_weight", type=float, default=0.45)
    parser.add_argument("--lowdiff_threshold", type=float, default=0.08)
    parser.add_argument("--lowdiff_scale", type=float, default=0.08)
    parser.add_argument("--lowdiff_weight", type=float, default=0.25)
    parser.add_argument("--blur_sigma", type=float, default=9.0)
    return parser.parse_args()


def _parse_alphas(text: str) -> List[float]:
    values = []
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        value = float(item)
        if value < 0 or value > 1:
            raise ValueError(f"alpha must be in [0,1], got {value}")
        values.append(value)
    if not values:
        raise ValueError("No alpha values were provided.")
    return values


def _alpha_tag(alpha: float) -> str:
    return f"alpha_{alpha:.2f}".replace(".", "p")


def _tensor_scalar(value) -> float:
    return float(value.detach().cpu().item())


def _center_crop_tensor(tensor, size: tuple[int, int]):
    h, w = size
    top = max((tensor.shape[-2] - h) // 2, 0)
    left = max((tensor.shape[-1] - w) // 2, 0)
    return tensor[..., top : top + h, left : left + w]


def _common_spatial_size(*tensors) -> tuple[int, int]:
    heights = [int(tensor.shape[-2]) for tensor in tensors if tensor is not None]
    widths = [int(tensor.shape[-1]) for tensor in tensors if tensor is not None]
    if not heights or not widths:
        raise ValueError("No tensors were provided for spatial alignment.")
    return min(heights), min(widths)


def _align_tensors(*tensors):
    size = _common_spatial_size(*tensors)
    return tuple(_center_crop_tensor(tensor, size) if tensor is not None else None for tensor in tensors)


def _stats(prior, err_out, rafa_out, input_tensor, blur_sigma: float) -> Dict[str, float]:
    import torch

    from eval_all import _gaussian_blur_batch

    input_tensor, err_out, rafa_out, prior = _align_tensors(input_tensor, err_out, rafa_out, prior)
    prior = prior.detach().clamp(0.0, 1.0)
    if prior.shape[1] != 1:
        prior = prior.mean(dim=1, keepdim=True)
    err_out = err_out.detach().clamp(0.0, 1.0)
    rafa_out = rafa_out.detach().clamp(0.0, 1.0)
    input_tensor = input_tensor.detach().clamp(0.0, 1.0)
    low_err = _gaussian_blur_batch(err_out, blur_sigma)
    low_rafa = _gaussian_blur_batch(rafa_out, blur_sigma)
    flat_prior = prior.reshape(-1)
    kth = max(1, min(flat_prior.numel(), int(round(0.90 * flat_prior.numel()))))
    return {
        "prior_mean": _tensor_scalar(prior.mean()),
        "prior_p90": _tensor_scalar(flat_prior.kthvalue(kth).values),
        "prior_area05": _tensor_scalar((prior > 0.5).float().mean()),
        "err_change": _tensor_scalar((input_tensor - err_out).abs().mean()),
        "rafa_change": _tensor_scalar((input_tensor - rafa_out).abs().mean()),
        "err_rafa_l1": _tensor_scalar((err_out - rafa_out).abs().mean()),
        "err_rafa_low_l1": _tensor_scalar((low_err - low_rafa).abs().mean()),
    }


def _adaptive_alpha(stats: Mapping[str, float], args: argparse.Namespace) -> float:
    area_den = max(1.0 - float(args.prior_area_threshold), 1e-6)
    area_excess = max(0.0, float(stats["prior_area05"]) - float(args.prior_area_threshold)) / area_den
    lowdiff_excess = max(0.0, float(stats["err_rafa_low_l1"]) - float(args.lowdiff_threshold)) / max(
        float(args.lowdiff_scale), 1e-6
    )
    alpha = (
        float(args.adaptive_base)
        - float(args.prior_area_weight) * area_excess
        - float(args.lowdiff_weight) * lowdiff_excess
    )
    return min(float(args.adaptive_max), max(float(args.adaptive_min), alpha))


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: List[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _fmt(value: float) -> str:
    return "nan" if math.isnan(value) else f"{value:.6f}"


def evaluate_dataset(args: argparse.Namespace, dataset_name: str, errnet, rafa, labels: Sequence[str], alphas: Sequence[float], device):
    import torch
    from torch.utils.data import DataLoader

    from datasets.unified_reflection_dataset import UnifiedReflectionDataset
    from metrics.reflection_metrics import average_metrics, compute_metrics, save_metrics_csv

    try:
        dataset = UnifiedReflectionDataset(
            Path(args.data_root),
            dataset_name,
            crop_size=None,
            image_size=None,
            max_long_edge=args.max_long_edge,
        )
    except FileNotFoundError as exc:
        print(f"[w] skipping {dataset_name}: {exc}")
        return {}, []

    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    rows_by_label: Dict[str, List[Dict[str, Any]]] = {label: [] for label in labels}
    feature_rows: List[Dict[str, Any]] = []
    for batch in loader:
        input_tensor = batch["input"].to(device)
        target_tensor = batch["target"].to(device)
        name = batch["name"][0] if isinstance(batch["name"], (list, tuple)) else str(batch["name"])
        with torch.no_grad():
            err_outputs = errnet(input_tensor)
            rafa_outputs = rafa(input_tensor)
        err_out = err_outputs["output"].clamp(0.0, 1.0)
        rafa_out = rafa_outputs["output"].clamp(0.0, 1.0)
        prior = rafa_outputs.get("prior")
        if prior is None:
            prior = torch.ones((1, 1, input_tensor.shape[-2], input_tensor.shape[-1]), dtype=input_tensor.dtype, device=device)
        input_aligned, target_aligned, err_out, rafa_out, prior = _align_tensors(
            input_tensor,
            target_tensor,
            err_out,
            rafa_out,
            prior,
        )
        stats = _stats(prior, err_out, rafa_out, input_aligned, float(args.blur_sigma))

        err_metrics = compute_metrics(err_out[0], target_aligned[0])
        rafa_metrics = compute_metrics(rafa_out[0], target_aligned[0])
        feature_row: Dict[str, Any] = {
            "dataset": dataset_name,
            "name": name,
            **stats,
            "ERR_PSNR": err_metrics["PSNR"],
            "RAFA_PSNR": rafa_metrics["PSNR"],
            "RAFA_minus_ERR_PSNR": rafa_metrics["PSNR"] - err_metrics["PSNR"],
        }

        for label, alpha in zip([_alpha_tag(a) for a in alphas], alphas):
            fused = (float(alpha) * rafa_out + (1.0 - float(alpha)) * err_out).clamp(0.0, 1.0)
            metrics = compute_metrics(fused[0], target_aligned[0])
            rows_by_label[label].append({"dataset": dataset_name, "name": name, **metrics})

        if args.include_adaptive:
            alpha = _adaptive_alpha(stats, args)
            fused = (float(alpha) * rafa_out + (1.0 - float(alpha)) * err_out).clamp(0.0, 1.0)
            metrics = compute_metrics(fused[0], target_aligned[0])
            rows_by_label["adaptive"].append({"dataset": dataset_name, "name": name, **metrics})
            feature_row["adaptive_alpha"] = alpha

        feature_rows.append(feature_row)

    summary = {}
    save_dir = Path(args.save_dir)
    for label, rows in rows_by_label.items():
        avg = save_metrics_csv(rows, save_dir / label / f"metrics_{dataset_name}.csv")
        summary[label] = avg
    _write_csv(save_dir / "features" / f"features_{dataset_name}.csv", feature_rows)
    return summary, feature_rows


def main() -> None:
    from eval_all import choose_device, load_model
    from metrics.reflection_metrics import save_metrics_csv

    args = parse_args()
    device = choose_device(args.device)
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    alphas = _parse_alphas(args.alphas)
    labels = [_alpha_tag(alpha) for alpha in alphas]
    if args.include_adaptive:
        labels.append("adaptive")

    errnet = load_model("errnet", Path(args.errnet_ckpt), device, errnet_hyper=args.errnet_hyper)
    rafa = load_model("rap_errnet", Path(args.rafa_ckpt), device, config_path=args.rafa_config)

    all_rows_by_label: Dict[str, List[Dict[str, Any]]] = {label: [] for label in labels}
    summary_rows: List[Dict[str, Any]] = []
    datasets = [item.strip() for item in args.datasets.split(",") if item.strip()]
    for dataset_name in datasets:
        print(f"[i] evaluating {dataset_name}")
        summary, _features = evaluate_dataset(args, dataset_name, errnet, rafa, labels, alphas, device)
        for label in labels:
            csv_path = save_dir / label / f"metrics_{dataset_name}.csv"
            if csv_path.exists():
                with csv_path.open(newline="", encoding="utf-8") as f:
                    rows = list(csv.DictReader(f))
                all_rows_by_label[label].extend(row for row in rows if row.get("dataset") != "average")
            avg = summary.get(label)
            if avg is not None:
                summary_rows.append(
                    {
                        "label": label,
                        "dataset": dataset_name,
                        "PSNR": _fmt(float(avg["PSNR"])),
                        "SSIM": _fmt(float(avg["SSIM"])),
                        "NCC": _fmt(float(avg["NCC"])),
                        "LMSE": _fmt(float(avg["LMSE"])),
                    }
                )

    for label, rows in all_rows_by_label.items():
        if rows:
            avg = save_metrics_csv(rows, save_dir / label / "metrics_all.csv")
            summary_rows.append(
                {
                    "label": label,
                    "dataset": "all",
                    "PSNR": _fmt(float(avg["PSNR"])),
                    "SSIM": _fmt(float(avg["SSIM"])),
                    "NCC": _fmt(float(avg["NCC"])),
                    "LMSE": _fmt(float(avg["LMSE"])),
                }
            )
    _write_csv(save_dir / "summary.csv", summary_rows)
    print(save_dir / "summary.csv")


if __name__ == "__main__":
    main()
