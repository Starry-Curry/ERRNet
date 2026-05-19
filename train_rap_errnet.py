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
from typing import Any, Dict, Iterable, List, Mapping

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
    parser.add_argument("--backbone_lr", type=float, default=None, help="learning rate for ERRNet backbone parameter group")
    parser.add_argument("--new_lr", type=float, default=None, help="learning rate for RAP prior/gating/refinement parameter group")
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

    if available.get("zhang_train", False):
        datasets.append(
            UnifiedReflectionDataset(data_root, "zhang_train", crop_size=crop_size, image_size=image_size)
        )
    else:
        missing_reasons.append("Zhang train pairs unavailable")

    if args.use_openrr:
        if available.get("openrr_train", False):
            datasets.append(
                UnifiedReflectionDataset(
                    data_root,
                    "openrr_train",
                    crop_size=crop_size,
                    image_size=image_size,
                    max_pairs=args.max_openrr_pairs,
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
                    max_pairs=args.max_extra_pairs,
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


LOSS_KEYS = ["total", "pix", "perc", "grad", "ssim", "mask", "clean", "anchor", "delta", "excl"]


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

    cfg = load_config(args.config)
    train_cfg = dict(cfg.get("train", {}))
    data_cfg = dict(cfg.get("data", {}))

    seed = int(train_cfg.get("seed", 42))
    set_seed(seed)
    device = choose_device(args.device)

    dataset = build_train_dataset(args, cfg)
    batch_size = int(args.batch_size or train_cfg.get("batch_size", 8))
    num_workers = int(args.num_workers if args.num_workers is not None else data_cfg.get("num_workers", 0))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=device.type == "cuda")
    print(
        f"[i] training samples={len(dataset)} batches_per_epoch={len(loader)} "
        f"batch_size={batch_size} device={device}"
    )

    model = build_model(args, cfg).to(device)
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
            f"anchor={avg['anchor']:.6f} delta={avg['delta']:.6f}"
        )


if __name__ == "__main__":
    main()
