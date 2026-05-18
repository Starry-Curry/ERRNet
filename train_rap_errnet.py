"""Training entry point for RAP-ERRNet.

This script is intentionally independent from the original ERRNet ``Engine`` so
that baseline files remain usable. It performs no work unless explicitly run.
"""

from __future__ import annotations

import argparse
import csv
import random
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
    parser.add_argument("--use_physics_synthesis", action="store_true", help="enable VOC physics-guided synthesis")
    parser.add_argument("--use_prior_head", action="store_true", default=None, help="enable reflection prior head")
    parser.add_argument("--no_prior_head", action="store_true", help="disable reflection prior head")
    parser.add_argument("--use_gated_blocks", action="store_true", default=None, help="enable prior-gated adapter")
    parser.add_argument("--no_gated_blocks", action="store_true", help="disable prior-gated adapter")
    parser.add_argument("--use_refinement", action="store_true", default=None, help="enable residual refinement")
    parser.add_argument("--no_refinement", action="store_true", help="disable residual refinement")
    parser.add_argument("--use_openrr", action="store_true", help="include OpenRR train pairs if available")
    parser.add_argument("--max_openrr_pairs", type=int, default=None, help="limit OpenRR training pairs")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"], help="training device")
    parser.add_argument("--num_workers", type=int, default=None)
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


def save_checkpoint(path: Path, model, optimizer, epoch: int, best_loss: float) -> None:
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "best_loss": best_loss,
        },
        path,
    )


def append_log(path: Path, row: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "total", "pix", "perc", "grad", "ssim", "mask", "clean", "excl"])
        if not exists:
            writer.writeheader()
        writer.writerow(row)


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

    model = build_model(args, cfg).to(device)
    criterion = ReflectionRemovalLoss(cfg)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(args.lr or train_cfg.get("lr", 1.0e-4)))

    start_epoch = 0
    best_loss = float("inf")
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint["model"], strict=False)
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = int(checkpoint.get("epoch", 0)) + 1
        best_loss = float(checkpoint.get("best_loss", best_loss))

    save_dir = Path("checkpoints") / args.name
    log_path = save_dir / "train_log.csv"
    epochs = int(args.epochs or train_cfg.get("epochs", 100))

    for epoch in range(start_epoch, epochs):
        model.train()
        running = {"total": 0.0, "pix": 0.0, "perc": 0.0, "grad": 0.0, "ssim": 0.0, "mask": 0.0, "clean": 0.0, "excl": 0.0}
        steps = 0
        for batch in loader:
            batch = move_batch_to_device(batch, device)
            outputs = model(batch["input"])
            losses = criterion(outputs, batch)
            optimizer.zero_grad(set_to_none=True)
            losses["total"].backward()
            optimizer.step()

            steps += 1
            for key in running:
                running[key] += float(losses[key].detach().cpu())
            if args.debug and steps >= 3:
                break

        if steps == 0:
            raise RuntimeError("Training dataloader produced zero batches.")
        avg = {key: value / steps for key, value in running.items()}
        append_log(log_path, {"epoch": epoch, **avg})
        save_checkpoint(save_dir / "latest.pt", model, optimizer, epoch, best_loss)
        if avg["total"] < best_loss:
            best_loss = avg["total"]
            save_checkpoint(save_dir / "best.pt", model, optimizer, epoch, best_loss)
        print(f"epoch={epoch} total={avg['total']:.6f} pix={avg['pix']:.6f} grad={avg['grad']:.6f}")


if __name__ == "__main__":
    main()
