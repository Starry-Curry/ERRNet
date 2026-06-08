"""Training entry point for RAP-ERRNet.

This script is intentionally independent from the original ERRNet ``Engine`` so
that baseline files remain usable. It performs no work unless explicitly run.
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
import time
import warnings
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional

try:
    import yaml
except Exception as exc:  # pragma: no cover
    yaml = None
    YAML_IMPORT_ERROR = exc
else:
    YAML_IMPORT_ERROR = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train RAP-ERRNet.")
    parser.add_argument("--config", default="configs/rap_errnet.yaml", help="YAML config path")
    parser.add_argument("--name", default="rap_errnet_main", help="experiment name")
    parser.add_argument("--data_root", default=None, help="dataset root, overrides config data.data_root")
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--resume", default=None, help="checkpoint path to resume")
    parser.add_argument("--resume_model_only", action="store_true", help="load checkpoint weights but reset epoch and optimizer state")
    parser.add_argument("--use_physics_synthesis", action="store_true", help="enable VOC physics-guided synthesis")
    parser.add_argument("--use_prior_head", action="store_true", default=None, help="enable reflection prior head")
    parser.add_argument("--no_prior_head", action="store_true", help="disable reflection prior head")
    parser.add_argument("--use_gated_blocks", action="store_true", default=None, help="enable prior-gated adapter")
    parser.add_argument("--no_gated_blocks", action="store_true", help="disable prior-gated adapter")
    parser.add_argument("--use_refinement", action="store_true", default=None, help="enable residual refinement")
    parser.add_argument("--no_refinement", action="store_true", help="disable residual refinement")
    parser.add_argument("--use_hypercolumn_backbone", action="store_true", default=None, help="use ERRNet --hyper VGG feature backbone")
    parser.add_argument("--no_hypercolumn_backbone", action="store_true", help="disable ERRNet --hyper VGG feature backbone")
    parser.add_argument("--use_openrr", action="store_true", help="include OpenRR train pairs if available")
    parser.add_argument("--max_openrr_pairs", type=int, default=None, help="limit OpenRR training pairs")
    parser.add_argument("--use_extra_train", action="store_true", help="include generic data/extra_train paired data if available")
    parser.add_argument("--max_extra_pairs", type=int, default=None, help="limit generic extra_train pairs")
    parser.add_argument("--no_zhang_train", action="store_true", help="do not include Zhang real89 training pairs")
    parser.add_argument("--use_course_replay", action="store_true", help="enable course-distribution replay sampling/distillation")
    parser.add_argument("--openrr_ratio", type=float, default=None, help="target OpenRR sampling mass for weighted replay")
    parser.add_argument("--course_replay_ratio", type=float, default=None, help="target VOC/Zhang sampling mass for weighted replay")
    parser.add_argument("--samples_per_epoch", type=int, default=None, help="weighted-sampler samples per epoch")
    parser.add_argument("--balance_openrr_bins", action="store_true", help="balance OpenRR replay by weak/strong and veil/ghost bins")
    parser.add_argument("--reflection_bins_csv", default=None, help="optional CSV from tools/analyze_reflection_strength.py")
    parser.add_argument("--reflection_bin_sigma", type=float, default=None, help="Gaussian sigma for on-the-fly reflection binning")
    parser.add_argument("--teacher_ckpt", default=None, help="teacher checkpoint for replay-anchored old-model distillation")
    parser.add_argument("--lambda_old", type=float, default=None, help="weight for replay teacher distillation")
    parser.add_argument("--old_loss_prob", type=float, default=None, help="probability of applying teacher distillation per batch")
    parser.add_argument("--backbone_lr", type=float, default=None, help="learning rate for ERRNet backbone parameter group")
    parser.add_argument("--new_lr", type=float, default=None, help="learning rate for RAP prior/gating/refinement parameter group")
    parser.add_argument("--use_ric", action="store_true", help="enable reflection-invariant consistency loss")
    parser.add_argument("--use_freq_loss", action="store_true", help="enable frequency-selective supervision loss")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"], help="training device")
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--log_interval", type=int, default=50, help="batches between training progress prints")
    parser.add_argument("--progress_bar", action="store_true", help="show an in-place dynamic epoch progress bar")
    parser.add_argument("--progress_interval", type=float, default=1.0, help="seconds between progress bar refreshes")
    parser.add_argument("--debug", action="store_true", help="run only a few iterations per epoch")
    return parser.parse_args()


def load_config(path: str | Path) -> Dict[str, Any]:
    if yaml is None:
        raise RuntimeError(f"PyYAML is required to read configs: {YAML_IMPORT_ERROR}")
    cfg_path = Path(path)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config file not found: {cfg_path}")
    with cfg_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def configure_optional_losses(cfg: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    """Keep v1.3 losses opt-in even when a config contains their weights."""

    cfg = dict(cfg)
    loss_cfg = dict(cfg.get("loss", {}))
    if args.use_ric:
        if float(loss_cfg.get("lambda_ric", 0.0)) <= 0:
            loss_cfg["lambda_ric"] = 0.03
        loss_cfg.setdefault("ric_prob", 0.5)
        loss_cfg.setdefault("ric_prior_weight", 1.0)
    else:
        loss_cfg["lambda_ric"] = 0.0

    if args.use_freq_loss:
        if float(loss_cfg.get("lambda_freq", 0.0)) <= 0:
            loss_cfg["lambda_freq"] = 0.03
        loss_cfg.setdefault("lambda_freq_low", 1.0)
        loss_cfg.setdefault("lambda_freq_high", 0.5)
        loss_cfg.setdefault("freq_sigma", 3.0)
        loss_cfg.setdefault("freq_prior_weight", 1.0)
    else:
        loss_cfg["lambda_freq"] = 0.0
    cfg["loss"] = loss_cfg
    return cfg


def choose_device(name: str) -> torch.device:
    import torch

    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is False.")
    return torch.device(name)


def set_seed(seed: int) -> None:
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _resolve_bool(args: argparse.Namespace, cfg: Mapping[str, Any], key: str) -> bool:
    if getattr(args, f"no_{key.replace('use_', '')}", False):
        return False
    value = getattr(args, key)
    if value is not None:
        return bool(value)
    return bool(cfg.get(key, True))


def build_model(args: argparse.Namespace, cfg: Mapping[str, Any]):
    from models.rap_errnet import RAPERRNet

    model_cfg = dict(cfg.get("model", {}))
    return RAPERRNet(
        use_prior_head=_resolve_bool(args, model_cfg, "use_prior_head"),
        use_gated_blocks=_resolve_bool(args, model_cfg, "use_gated_blocks"),
        use_refinement=_resolve_bool(args, model_cfg, "use_refinement"),
        use_hypercolumn_backbone=_resolve_bool(args, model_cfg, "use_hypercolumn_backbone"),
        freeze_backbone=bool(model_cfg.get("freeze_backbone", False)),
        pretrained_errnet_path=model_cfg.get("pretrained_errnet_path"),
        residual_scale=float(model_cfg.get("residual_scale", 0.1)),
    )


def build_train_dataset(args: argparse.Namespace, cfg: Mapping[str, Any]):
    from torch.utils.data import ConcatDataset

    from datasets.unified_reflection_dataset import UnifiedReflectionDataset, list_available_datasets

    data_cfg = dict(cfg.get("data", {}))
    data_root = Path(args.data_root or data_cfg.get("data_root", "./data"))
    crop_size = data_cfg.get("crop_size", 224)
    image_size = data_cfg.get("image_size", None)
    synthesis_cfg = cfg.get("synthesis", {})

    if not data_root.exists():
        raise FileNotFoundError(
            f"data_root does not exist: {data_root}. Prepare datasets on the server first; this script will not download them."
        )

    available = list_available_datasets(data_root)
    datasets = []
    missing_reasons: List[str] = []
    if args.use_physics_synthesis:
        if available.get("voc", False):
            datasets.append(
                UnifiedReflectionDataset(
                    data_root,
                    "voc",
                    crop_size=crop_size,
                    image_size=image_size,
                    use_physics_synthesis=True,
                    synthesis_config=synthesis_cfg,
                )
            )
        else:
            missing_reasons.append("VOC synthesis requested but data/VOC2012 is unavailable")

    use_zhang_train = bool(data_cfg.get("use_zhang_train", True)) and not args.no_zhang_train
    if use_zhang_train and available.get("zhang_train", False):
        datasets.append(
            UnifiedReflectionDataset(data_root, "zhang_train", crop_size=crop_size, image_size=image_size)
        )
    elif use_zhang_train:
        missing_reasons.append("Zhang train pairs unavailable")

    if args.use_openrr:
        if available.get("openrr_train", False):
            datasets.append(
                UnifiedReflectionDataset(
                    data_root,
                    "openrr_train",
                    crop_size=crop_size,
                    image_size=image_size,
                    max_pairs=args.max_openrr_pairs
                    if args.max_openrr_pairs is not None
                    else data_cfg.get("max_openrr_pairs"),
                )
            )
        else:
            missing_reasons.append("OpenRR train requested but unavailable")

    if args.use_extra_train or bool(data_cfg.get("use_extra_train", False)):
        if available.get("extra_train", False):
            datasets.append(
                UnifiedReflectionDataset(
                    data_root,
                    "extra_train",
                    crop_size=crop_size,
                    image_size=image_size,
                    max_pairs=args.max_extra_pairs
                    if args.max_extra_pairs is not None
                    else data_cfg.get("max_extra_pairs"),
                )
            )
        else:
            missing_reasons.append("extra_train requested but unavailable")

    if not datasets:
        detail = "; ".join(missing_reasons) if missing_reasons else "no supported training datasets found"
        raise FileNotFoundError(f"No training data was found under {data_root}: {detail}.")

    return datasets[0] if len(datasets) == 1 else ConcatDataset(datasets)


def move_batch_to_device(batch: Mapping[str, Any], device) -> Dict[str, Any]:
    import torch

    moved = {}
    for key, value in batch.items():
        moved[key] = value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value
    return moved


LOSS_KEYS = ["total", "pix", "perc", "grad", "ssim", "mask", "clean", "anchor", "delta", "freq", "ric", "old", "excl"]
COURSE_REPLAY_DATASETS = {"voc", "zhang_train"}


def _component_dataset_name(dataset) -> str:
    return str(getattr(dataset, "dataset", dataset.__class__.__name__))


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


def _assign_reflection_bin(strength: float, hf_ratio: float, strength_mid: float, hf_mid: float) -> str:
    strength_tag = "strong" if strength >= strength_mid else "weak"
    freq_tag = "ghost" if hf_ratio >= hf_mid else "veil"
    return f"{strength_tag}_{freq_tag}"


def _pair_sample_name(dataset, index: int) -> str:
    pairs = getattr(dataset, "pairs", None)
    if pairs and 0 <= index < len(pairs):
        return Path(pairs[index][0]).stem
    return str(index)


def _load_reflection_bins_csv(path: Optional[str | Path]) -> Dict[str, str]:
    if not path:
        return {}
    csv_path = Path(path)
    if not csv_path.exists():
        warnings.warn(f"reflection_bins_csv was provided but does not exist: {csv_path}", RuntimeWarning)
        return {}

    bins: Dict[str, str] = {}
    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("dataset") != "openrr_train":
                continue
            name = str(row.get("name", "")).strip()
            bin_name = str(row.get("bin", "")).strip()
            if name and bin_name:
                bins[name] = bin_name
    return bins


def _compute_reflection_bins(dataset, sigma: float) -> List[str]:
    import torch

    print("[i] computing OpenRR reflection bins for balanced sampler; this may take a few minutes")
    rows: List[Dict[str, float]] = []
    with torch.no_grad():
        for index in range(len(dataset)):
            sample = dataset[index]
            diff = (sample["input"] - sample["target"]).abs().float().unsqueeze(0)
            strength = float(diff.mean())
            high = diff - _gaussian_blur(diff, sigma)
            hf_ratio = float(high.abs().mean() / diff.mean().clamp_min(1e-6))
            rows.append({"strength": strength, "hf_ratio": hf_ratio})

    strength_mid = _median([row["strength"] for row in rows])
    hf_mid = _median([row["hf_ratio"] for row in rows])
    return [_assign_reflection_bin(row["strength"], row["hf_ratio"], strength_mid, hf_mid) for row in rows]


def _openrr_bin_assignments(dataset, csv_path: Optional[str | Path], sigma: float) -> List[str]:
    csv_bins = _load_reflection_bins_csv(csv_path)
    if csv_bins:
        names = [_pair_sample_name(dataset, index) for index in range(len(dataset))]
        if all(name in csv_bins for name in names):
            return [csv_bins[name] for name in names]
        warnings.warn(
            "reflection_bins_csv does not cover all OpenRR training pairs; computing bins on the fly.",
            RuntimeWarning,
        )
    return _compute_reflection_bins(dataset, sigma)


def _build_weighted_sampler(dataset, args: argparse.Namespace, cfg: Mapping[str, Any]):
    """Optional RAFA sampler that balances OpenRR and course replay samples."""

    from collections import Counter

    import torch
    from torch.utils.data import WeightedRandomSampler

    data_cfg = dict(cfg.get("data", {}))
    train_cfg = dict(cfg.get("train", {}))
    use_replay = bool(args.use_course_replay or data_cfg.get("use_course_replay", False))
    balance_openrr_bins = bool(args.balance_openrr_bins or data_cfg.get("balance_openrr_bins", False))
    openrr_ratio = args.openrr_ratio if args.openrr_ratio is not None else data_cfg.get("openrr_ratio")
    course_ratio = (
        args.course_replay_ratio
        if args.course_replay_ratio is not None
        else data_cfg.get("course_replay_ratio")
    )
    if not use_replay and not balance_openrr_bins and openrr_ratio is None and course_ratio is None:
        return None

    children = getattr(dataset, "datasets", None)
    cumulative = getattr(dataset, "cumulative_sizes", None)
    if not children or not cumulative:
        warnings.warn("Replay sampling requested but training dataset is not a ConcatDataset; using normal shuffle.", RuntimeWarning)
        return None

    groups: Dict[str, List[range]] = {"openrr": [], "course": [], "other": []}
    components: List[Dict[str, Any]] = []
    start = 0
    for child, end in zip(children, cumulative):
        name = _component_dataset_name(child)
        group = "openrr" if name == "openrr_train" else "course" if name in COURSE_REPLAY_DATASETS else "other"
        end = int(end)
        groups[group].append(range(start, end))
        components.append({"dataset": child, "name": name, "group": group, "start": start, "end": end})
        start = int(end)

    group_counts = {key: sum(len(item) for item in ranges) for key, ranges in groups.items()}
    masses: Dict[str, float] = {}
    if openrr_ratio is not None and group_counts["openrr"] > 0:
        masses["openrr"] = float(openrr_ratio)
    if (course_ratio is not None or use_replay) and group_counts["course"] > 0:
        if course_ratio is None:
            course_ratio = max(0.0, 1.0 - float(openrr_ratio or 0.0))
        masses["course"] = float(course_ratio)
    used_mass = sum(max(value, 0.0) for value in masses.values())
    if group_counts["other"] > 0:
        masses["other"] = max(0.0, 1.0 - used_mass)
    norm = sum(value for key, value in masses.items() if group_counts.get(key, 0) > 0 and value > 0)
    if norm <= 0:
        warnings.warn("Replay sampler had no positive sampling masses; using normal shuffle.", RuntimeWarning)
        return None

    weights = torch.zeros(len(dataset), dtype=torch.double)
    summary = []
    for group, ranges in groups.items():
        count = group_counts[group]
        if count <= 0:
            continue
        mass = max(0.0, masses.get(group, 0.0)) / norm
        per_sample = mass / count if mass > 0 else 0.0
        for index_range in ranges:
            weights[list(index_range)] = per_sample
        summary.append(f"{group}={mass:.3f}({count})")

    if balance_openrr_bins:
        openrr_mass = max(0.0, masses.get("openrr", 0.0)) / norm if group_counts["openrr"] > 0 else 0.0
        if openrr_mass <= 0:
            warnings.warn("balance_openrr_bins requested but OpenRR sampling mass is zero.", RuntimeWarning)
        else:
            csv_path = args.reflection_bins_csv or data_cfg.get("reflection_bins_csv")
            sigma = float(
                args.reflection_bin_sigma
                if args.reflection_bin_sigma is not None
                else data_cfg.get("reflection_bin_sigma", 3.0)
            )
            openrr_items: List[tuple[int, str]] = []
            for component in components:
                if component["group"] != "openrr":
                    continue
                child = component["dataset"]
                bins = _openrr_bin_assignments(child, csv_path, sigma)
                expected = int(component["end"]) - int(component["start"])
                if len(bins) != expected:
                    raise RuntimeError(f"OpenRR bin count mismatch: expected {expected}, got {len(bins)}")
                for local_index, bin_name in enumerate(bins):
                    openrr_items.append((int(component["start"]) + local_index, bin_name))

            bin_counts = Counter(bin_name for _, bin_name in openrr_items)
            active_bins = sorted(bin_name for bin_name, count in bin_counts.items() if count > 0)
            if active_bins:
                per_bin_mass = openrr_mass / len(active_bins)
                for global_index, bin_name in openrr_items:
                    weights[global_index] = per_bin_mass / bin_counts[bin_name]
                summary.append(
                    "openrr_bins="
                    + ",".join(f"{bin_name}:{bin_counts[bin_name]}" for bin_name in active_bins)
                )

    samples_per_epoch = (
        args.samples_per_epoch
        if args.samples_per_epoch is not None
        else data_cfg.get("samples_per_epoch", train_cfg.get("samples_per_epoch", len(dataset)))
    )
    samples_per_epoch = int(samples_per_epoch)
    if samples_per_epoch <= 0:
        raise ValueError(f"samples_per_epoch must be positive, got {samples_per_epoch}.")
    print(f"[i] replay sampler enabled: samples_per_epoch={samples_per_epoch} " + " ".join(summary))
    return WeightedRandomSampler(weights, num_samples=samples_per_epoch, replacement=True)


def save_checkpoint(path: Path, model, optimizer, epoch: int, best_loss: float, stage: Mapping[str, Any] | None = None) -> None:
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "best_loss": best_loss,
            "stage": dict(stage or {}),
        },
        path,
    )


def append_log(path: Path, row: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "epoch",
                "stage",
                "total",
                "pix",
                "perc",
                "grad",
                "ssim",
                "mask",
                "clean",
                "anchor",
                "delta",
                "freq",
                "ric",
                "old",
                "excl",
            ],
        )
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def _build_training_stages(train_cfg: Mapping[str, Any], args: argparse.Namespace, total_epochs: int) -> List[Dict[str, Any]]:
    stages_cfg = train_cfg.get("stages", [])
    if not stages_cfg:
        lr = float(args.lr or train_cfg.get("lr", 1.0e-4))
        return [
            {
                "name": "main",
                "start": 0,
                "end": total_epochs,
                "freeze_backbone": bool(train_cfg.get("freeze_backbone", False)),
                "lr": lr,
                "backbone_lr": float(args.backbone_lr if args.backbone_lr is not None else train_cfg.get("backbone_lr", lr)),
                "new_lr": float(args.new_lr if args.new_lr is not None else train_cfg.get("new_lr", lr)),
            }
        ]

    stages: List[Dict[str, Any]] = []
    start = 0
    for index, stage_cfg in enumerate(stages_cfg):
        stage = dict(stage_cfg)
        duration = int(stage.get("epochs", 0))
        if duration <= 0:
            raise ValueError(f"train.stages[{index}].epochs must be positive, got {duration}.")
        end = min(total_epochs, start + duration)
        lr = float(args.lr or stage.get("lr", train_cfg.get("lr", 1.0e-4)))
        stage.update(
            {
                "name": str(stage.get("name", f"stage{index + 1}")),
                "start": start,
                "end": end,
                "freeze_backbone": bool(stage.get("freeze_backbone", False)),
                "lr": lr,
                "backbone_lr": float(
                    args.backbone_lr
                    if args.backbone_lr is not None
                    else stage.get("backbone_lr", train_cfg.get("backbone_lr", lr))
                ),
                "new_lr": float(
                    args.new_lr
                    if args.new_lr is not None
                    else stage.get("new_lr", train_cfg.get("new_lr", lr))
                ),
            }
        )
        stages.append(stage)
        start = end
        if start >= total_epochs:
            break

    if not stages:
        raise ValueError("No active training stages were configured.")
    if stages[-1]["end"] < total_epochs:
        tail = dict(stages[-1])
        tail["start"] = stages[-1]["end"]
        tail["end"] = total_epochs
        tail["name"] = f"{tail['name']}_extended"
        stages.append(tail)
    return stages


def _stage_for_epoch(epoch: int, stages: List[Mapping[str, Any]]) -> tuple[int, Mapping[str, Any]]:
    for index, stage in enumerate(stages):
        if int(stage["start"]) <= epoch < int(stage["end"]):
            return index, stage
    return len(stages) - 1, stages[-1]


def _set_requires_grad(module, enabled: bool) -> None:
    if module is None:
        return
    for param in module.parameters():
        param.requires_grad = enabled


def _apply_training_stage(model, stage: Mapping[str, Any]) -> None:
    freeze_backbone = bool(stage.get("freeze_backbone", False))
    _set_requires_grad(getattr(model, "backbone", None), not freeze_backbone)
    _set_requires_grad(getattr(model, "prior_head", None), True)
    _set_requires_grad(getattr(model, "gated_adapter", None), True)
    _set_requires_grad(getattr(model, "refinement", None), True)
    _set_requires_grad(getattr(model, "vgg", None), False)


def _build_optimizer(model, stage: Mapping[str, Any], train_cfg: Mapping[str, Any]):
    import torch

    weight_decay = float(train_cfg.get("weight_decay", train_cfg.get("wd", 0.0)))
    param_groups = []
    backbone_params = [p for p in getattr(model, "backbone").parameters() if p.requires_grad]
    if backbone_params:
        param_groups.append({"params": backbone_params, "lr": float(stage.get("backbone_lr", stage.get("lr", 1.0e-4)))})

    new_params = []
    for module_name in ("prior_head", "gated_adapter", "refinement"):
        module = getattr(model, module_name, None)
        if module is not None:
            new_params.extend(p for p in module.parameters() if p.requires_grad)
    if new_params:
        param_groups.append({"params": new_params, "lr": float(stage.get("new_lr", stage.get("lr", 1.0e-4)))})

    if not param_groups:
        raise RuntimeError("No trainable parameters are active for the current training stage.")
    optimizer_name = str(train_cfg.get("optimizer", "adam")).lower()
    if optimizer_name != "adam":
        warnings.warn(f"Unsupported optimizer '{optimizer_name}', falling back to Adam.", RuntimeWarning)
    return torch.optim.Adam(param_groups, weight_decay=weight_decay)


def _batch_dataset_names(batch: Mapping[str, Any], batch_size: int) -> List[str]:
    names = batch.get("dataset", "")
    if isinstance(names, str):
        return [names] * batch_size
    if isinstance(names, (list, tuple)):
        return [str(name) for name in names]
    return [str(names)] * batch_size


def _build_ric_inputs(batch: Mapping[str, Any], device, synthesis_config: Mapping[str, Any]) -> tuple[Any, Any]:
    import torch

    from datasets.reflection_synthesis import synthesize_reflection_pair

    target = batch.get("target")
    if not isinstance(target, torch.Tensor):
        return None, None
    names = _batch_dataset_names(batch, int(target.shape[0]))
    voc_indices = [index for index, name in enumerate(names) if name == "voc"]
    if not voc_indices:
        return None, None

    selected = torch.tensor(voc_indices, dtype=torch.long, device=device)
    target_cpu = target.detach().cpu()
    if len(voc_indices) > 1:
        reflection_pool = target_cpu[voc_indices].roll(shifts=1, dims=0)
    else:
        reflection = batch.get("reflection")
        if isinstance(reflection, torch.Tensor):
            reflection_pool = reflection.detach().cpu()[voc_indices]
        else:
            reflection_pool = target_cpu[voc_indices]

    ric_inputs = []
    for local_index, batch_index in enumerate(voc_indices):
        seed = random.randint(0, 2**31 - 1)
        sample = synthesize_reflection_pair(
            target_cpu[batch_index],
            reflection_pool[local_index],
            seed=seed,
            config=synthesis_config,
        )
        ric_inputs.append(sample["input"])
    return selected, torch.stack(ric_inputs, dim=0).to(device, non_blocking=True)


def reflection_invariant_consistency_loss(
    outputs_a: Mapping[str, Any],
    outputs_b: Mapping[str, Any],
    indices,
    prior_weight: float,
) -> Any:
    import torch

    pred_a = outputs_a["output"].index_select(0, indices)
    pred_b = outputs_b["output"]
    diff = torch.abs(pred_a - pred_b)
    if prior_weight > 0:
        prior_a = outputs_a["prior"].index_select(0, indices)
        prior_b = outputs_b["prior"]
        prior = 0.5 * (prior_a + prior_b)
        diff = (1.0 + float(prior_weight) * prior) * diff
    return diff.mean()


def _build_teacher_model(args: argparse.Namespace, cfg: Mapping[str, Any], ckpt_path: str | Path, device):
    import torch

    teacher = build_model(args, cfg).to(device)
    checkpoint = torch.load(ckpt_path, map_location=device)
    state = checkpoint.get("model", checkpoint.get("state_dict", checkpoint))
    teacher.load_state_dict(state, strict=False)
    teacher.eval()
    for param in teacher.parameters():
        param.requires_grad = False
    return teacher


def _course_replay_indices(batch: Mapping[str, Any], device) -> Any:
    import torch

    input_tensor = batch.get("input")
    if not isinstance(input_tensor, torch.Tensor):
        return None
    names = _batch_dataset_names(batch, int(input_tensor.shape[0]))
    indices = [index for index, name in enumerate(names) if name in COURSE_REPLAY_DATASETS]
    if not indices:
        return None
    return torch.tensor(indices, dtype=torch.long, device=device)


def old_model_distillation_loss(outputs, teacher_outputs, indices) -> Any:
    import torch

    student = outputs["output"].index_select(0, indices)
    teacher = teacher_outputs["output"].detach()
    return torch.mean(torch.abs(student - teacher))


def _format_seconds(seconds: float) -> str:
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours:d}h{minutes:02d}m"
    if minutes:
        return f"{minutes:d}m{seconds:02d}s"
    return f"{seconds:d}s"


def _progress_line(
    *,
    epoch: int,
    epochs: int,
    batch_idx: int,
    total_batches: int,
    running: Mapping[str, float],
    steps: int,
    current_total: float,
    epoch_start_time: float,
    width: int = 20,
) -> str:
    progress = batch_idx / max(total_batches, 1)
    filled = min(width, int(round(width * progress)))
    bar = "#" * filled + "." * (width - filled)
    elapsed = time.time() - epoch_start_time
    rate = batch_idx / elapsed if elapsed > 0 else 0.0
    eta = (total_batches - batch_idx) / rate if rate > 0 else 0.0
    avg_total = running["total"] / max(steps, 1)
    avg_pix = running["pix"] / max(steps, 1)
    avg_grad = running["grad"] / max(steps, 1)
    return (
        f"ep {epoch + 1}/{epochs} [{bar}] {batch_idx}/{total_batches} "
        f"{progress * 100:4.1f}% eta { _format_seconds(eta)} "
        f"loss {avg_total:.4f} pix {avg_pix:.4f} grad {avg_grad:.4f} cur {current_total:.4f}"
    )


def main() -> None:
    args = parse_args()

    import torch
    from torch.utils.data import DataLoader

    from losses.reflection_losses import ReflectionRemovalLoss

    cfg = configure_optional_losses(load_config(args.config), args)
    train_cfg = dict(cfg.get("train", {}))
    data_cfg = dict(cfg.get("data", {}))
    loss_cfg = dict(cfg.get("loss", {}))
    lambda_ric = float(loss_cfg.get("lambda_ric", 0.0))
    ric_prob = float(loss_cfg.get("ric_prob", 0.0))
    ric_prior_weight = float(loss_cfg.get("ric_prior_weight", 0.0))
    lambda_old = float(args.lambda_old if args.lambda_old is not None else loss_cfg.get("lambda_old", 0.0))
    old_loss_prob = float(args.old_loss_prob if args.old_loss_prob is not None else loss_cfg.get("old_loss_prob", 1.0))
    teacher_ckpt = args.teacher_ckpt or train_cfg.get("teacher_ckpt") or loss_cfg.get("teacher_ckpt")
    if lambda_ric > 0 and not args.use_physics_synthesis:
        warnings.warn(
            "RIC is enabled but --use_physics_synthesis is off; RIC will be skipped because it only applies to VOC synthetic samples.",
            RuntimeWarning,
        )
    if lambda_old > 0 and not teacher_ckpt:
        warnings.warn("lambda_old > 0 but no --teacher_ckpt/train.teacher_ckpt was provided; old distillation is disabled.", RuntimeWarning)
        lambda_old = 0.0

    seed = int(train_cfg.get("seed", 42))
    set_seed(seed)
    device = choose_device(args.device)

    dataset = build_train_dataset(args, cfg)
    batch_size = int(args.batch_size or train_cfg.get("batch_size", 8))
    num_workers = int(args.num_workers if args.num_workers is not None else data_cfg.get("num_workers", 0))
    sampler = _build_weighted_sampler(dataset, args, cfg)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )
    print(
        f"[i] training samples={len(dataset)} batches_per_epoch={len(loader)} "
        f"batch_size={batch_size} device={device}"
    )

    model = build_model(args, cfg).to(device)
    teacher_model = None
    if lambda_old > 0:
        teacher_model = _build_teacher_model(args, cfg, teacher_ckpt, device)
        print(f"[i] loaded replay teacher from {teacher_ckpt}; lambda_old={lambda_old:.4f} old_loss_prob={old_loss_prob:.2f}")
    criterion = ReflectionRemovalLoss(cfg)
    epochs = int(args.epochs or train_cfg.get("epochs", 100))
    stages = _build_training_stages(train_cfg, args, epochs)
    optimizer = None
    active_stage_index = None
    optimizer_restored = False
    pending_optimizer_state = None

    start_epoch = 0
    best_loss = float("inf")
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint["model"], strict=False)
        if args.resume_model_only:
            print(f"[i] loaded model weights from {args.resume}; epoch and optimizer were reset")
        else:
            pending_optimizer_state = checkpoint.get("optimizer")
            start_epoch = int(checkpoint.get("epoch", 0)) + 1
            best_loss = float(checkpoint.get("best_loss", best_loss))

    save_dir = Path("checkpoints") / args.name
    log_path = save_dir / "train_log.csv"
    for stage in stages:
        print(
            f"[i] stage {stage['name']}: epochs {int(stage['start']) + 1}-{int(stage['end'])}, "
            f"freeze_backbone={bool(stage.get('freeze_backbone', False))} "
            f"backbone_lr={float(stage.get('backbone_lr', stage.get('lr', 1.0e-4))):.2e} "
            f"new_lr={float(stage.get('new_lr', stage.get('lr', 1.0e-4))):.2e}"
        )

    for epoch in range(start_epoch, epochs):
        stage_index, stage = _stage_for_epoch(epoch, stages)
        if stage_index != active_stage_index:
            _apply_training_stage(model, stage)
            optimizer = _build_optimizer(model, stage, train_cfg)
            active_stage_index = stage_index
            print(f"[i] entering stage {stage['name']} at epoch {epoch + 1}")
            if pending_optimizer_state is not None and not optimizer_restored:
                try:
                    optimizer.load_state_dict(pending_optimizer_state)
                    optimizer_restored = True
                    print("[i] optimizer state restored from checkpoint")
                except ValueError as exc:
                    warnings.warn(
                        f"Could not restore optimizer state after stage setup; continuing with a fresh optimizer: {exc}",
                        RuntimeWarning,
                    )
                    optimizer_restored = True

        model.train()
        running = {key: 0.0 for key in LOSS_KEYS}
        steps = 0
        epoch_start_time = time.time()
        last_progress_time = 0.0
        print(f"[i] epoch {epoch + 1}/{epochs} started ({stage['name']})")
        for batch_idx, batch in enumerate(loader, start=1):
            batch = move_batch_to_device(batch, device)
            outputs = model(batch["input"])
            losses = criterion(outputs, batch)
            if lambda_ric > 0 and ric_prob > 0 and random.random() < ric_prob:
                ric_indices, ric_input = _build_ric_inputs(batch, device, cfg.get("synthesis", {}))
                if ric_indices is not None and ric_input is not None:
                    ric_outputs = model(ric_input)
                    ric = reflection_invariant_consistency_loss(outputs, ric_outputs, ric_indices, ric_prior_weight)
                    losses["ric"] = ric
                    losses["total"] = losses["total"] + lambda_ric * ric
            if teacher_model is not None and lambda_old > 0 and old_loss_prob > 0 and random.random() < old_loss_prob:
                old_indices = _course_replay_indices(batch, device)
                if old_indices is not None:
                    old_input = batch["input"].index_select(0, old_indices)
                    with torch.no_grad():
                        teacher_outputs = teacher_model(old_input)
                    old = old_model_distillation_loss(outputs, teacher_outputs, old_indices)
                    losses["old"] = old
                    losses["total"] = losses["total"] + lambda_old * old
            if optimizer is None:
                raise RuntimeError("Optimizer was not initialized.")
            optimizer.zero_grad(set_to_none=True)
            losses["total"].backward()
            optimizer.step()

            steps += 1
            current_total = float(losses["total"].detach().cpu())
            for key in running:
                value = losses.get(key)
                if value is not None:
                    running[key] += float(value.detach().cpu())
            should_finish_progress = batch_idx == len(loader) or (args.debug and steps >= 3)
            if args.progress_bar:
                if time.time() - last_progress_time >= args.progress_interval or should_finish_progress:
                    line = _progress_line(
                        epoch=epoch,
                        epochs=epochs,
                        batch_idx=batch_idx,
                        total_batches=len(loader),
                        running=running,
                        steps=steps,
                        current_total=current_total,
                        epoch_start_time=epoch_start_time,
                    )
                    sys.stdout.write("\r" + line + " " * 12)
                    sys.stdout.flush()
                    last_progress_time = time.time()
            else:
                if args.log_interval > 0 and (batch_idx % args.log_interval == 0 or batch_idx == len(loader)):
                    avg_total = running["total"] / steps
                    avg_pix = running["pix"] / steps
                    avg_grad = running["grad"] / steps
                    print(
                        f"epoch={epoch + 1}/{epochs} batch={batch_idx}/{len(loader)} "
                        f"avg_total={avg_total:.6f} avg_pix={avg_pix:.6f} "
                        f"avg_grad={avg_grad:.6f} current_total={current_total:.6f}",
                        flush=True,
                    )
            if args.debug and steps >= 3:
                break
        if args.progress_bar:
            sys.stdout.write("\n")
            sys.stdout.flush()

        if steps == 0:
            raise RuntimeError("Training dataloader produced zero batches.")
        avg = {key: value / steps for key, value in running.items()}
        append_log(log_path, {"epoch": epoch, "stage": stage["name"], **avg})
        save_checkpoint(save_dir / "latest.pt", model, optimizer, epoch, best_loss, stage=stage)
        if avg["total"] < best_loss:
            best_loss = avg["total"]
            save_checkpoint(save_dir / "best.pt", model, optimizer, epoch, best_loss, stage=stage)
        print(
            f"epoch={epoch} stage={stage['name']} total={avg['total']:.6f} "
            f"pix={avg['pix']:.6f} grad={avg['grad']:.6f} "
            f"anchor={avg['anchor']:.6f} delta={avg['delta']:.6f} "
            f"freq={avg['freq']:.6f} ric={avg['ric']:.6f} old={avg['old']:.6f}"
        )


if __name__ == "__main__":
    main()
