"""Evaluate ERRNet/RAP-ERRNet on all configured reflection datasets."""

from __future__ import annotations

import argparse
import csv
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


DEFAULT_DATASETS = "ceilnet,zhang20,sir2_objects,sir2_postcard,sir2_wild,self,openrr_val"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate ERRNet or RAP-ERRNet on multiple datasets.")
    parser.add_argument("--model", choices=["errnet", "rap_errnet"], default="rap_errnet")
    parser.add_argument("--ckpt", required=True, help="checkpoint path")
    parser.add_argument("--config", default=None, help="RAP-ERRNet YAML config used to build the checkpointed model")
    parser.add_argument("--baseline_ckpt", default=None, help="optional ERRNet checkpoint to include in saved visualizations")
    parser.add_argument("--baseline_hyper", action="store_true", help="build the optional baseline as ERRNet --hyper")
    parser.add_argument("--data_root", default="./data")
    parser.add_argument("--save_dir", default="results/rap_errnet_eval")
    parser.add_argument("--datasets", default=DEFAULT_DATASETS, help="comma-separated dataset names")
    parser.add_argument("--save_images", action="store_true")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    return parser.parse_args()


def load_config(path: Optional[str | Path]) -> Dict[str, Any]:
    if path is None:
        return {}
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
        raise RuntimeError("CUDA was requested but is not available.")
    return torch.device(name)


def _load_checkpoint(path: Path, map_location) -> Dict[str, Any]:
    import torch

    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def _filter_compatible(state: Mapping[str, Any], model) -> Dict[str, Any]:
    current = model.state_dict()
    return {
        key: value
        for key, value in state.items()
        if key in current and hasattr(value, "shape") and tuple(value.shape) == tuple(current[key].shape)
    }


def _build_rap_model_from_config(config_path: Optional[str | Path]):
    from models.rap_errnet import RAPERRNet

    cfg = load_config(config_path)
    model_cfg = dict(cfg.get("model", {}))
    return RAPERRNet(
        use_prior_head=bool(model_cfg.get("use_prior_head", True)),
        use_gated_blocks=bool(model_cfg.get("use_gated_blocks", True)),
        use_refinement=bool(model_cfg.get("use_refinement", True)),
        use_hypercolumn_backbone=bool(model_cfg.get("use_hypercolumn_backbone", False)),
        freeze_backbone=False,
        pretrained_errnet_path=None,
        residual_scale=float(model_cfg.get("residual_scale", 0.1)),
    )


def _raise_if_rap_backbone_mismatch(state: Mapping[str, Any], model, ckpt_path: Path, config_path: Optional[str | Path]) -> None:
    first_key = "backbone.conv1.conv2d.weight"
    if first_key not in state or first_key not in model.state_dict():
        return
    if not hasattr(state[first_key], "shape"):
        return
    checkpoint_shape = tuple(state[first_key].shape)
    model_shape = tuple(model.state_dict()[first_key].shape)
    if checkpoint_shape == model_shape:
        return
    hint = (
        "This usually means the checkpoint was trained with the hypercolumn backbone "
        "(1475 input channels) but eval_all.py built the default 3-channel RAP model."
    )
    config_hint = (
        " Pass the matching config, for example "
        "`--config configs/rap_errnet_hyper_pretrained.yaml`."
        if config_path is None
        else f" Check that {config_path} has the same model.use_hypercolumn_backbone setting used for training."
    )
    raise ValueError(
        f"RAP-ERRNet checkpoint/model backbone mismatch for {ckpt_path}: "
        f"checkpoint {first_key} has shape {checkpoint_shape}, model expects {model_shape}. "
        f"{hint}{config_hint}"
    )


def load_model(
    model_name: str,
    ckpt_path: Path,
    device,
    config_path: Optional[str | Path] = None,
    *,
    errnet_hyper: bool = False,
):
    import torch
    from torch import nn
    import torch.nn.functional as F

    from models import arch
    from models.vgg import Vgg19

    class ERRNetEvalWrapper(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = arch.errnet(3, 3)

        def forward(self, x):
            output = self.net(x).clamp(0.0, 1.0)
            return {
                "output": output,
                "coarse": output,
                "prior": torch.zeros((x.shape[0], 1, x.shape[2], x.shape[3]), dtype=x.dtype, device=x.device),
                "residual": torch.zeros_like(output),
            }

    class ERRNetHyperEvalWrapper(nn.Module):
        def __init__(self):
            super().__init__()
            self.vgg = Vgg19(requires_grad=False)
            self.net = arch.errnet(1475, 3)

        def forward(self, x):
            self.vgg = self.vgg.to(x.device)
            self.vgg.eval()
            with torch.no_grad():
                hypercolumn = self.vgg(x)
                hypercolumn = [
                    F.interpolate(feature.detach(), size=x.shape[-2:], mode="bilinear", align_corners=False)
                    for feature in hypercolumn
                ]
            output = self.net(torch.cat([x, *hypercolumn], dim=1)).clamp(0.0, 1.0)
            return {
                "output": output,
                "coarse": output,
                "prior": torch.zeros((x.shape[0], 1, x.shape[2], x.shape[3]), dtype=x.dtype, device=x.device),
                "residual": torch.zeros_like(output),
            }

    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    checkpoint = _load_checkpoint(ckpt_path, map_location=device)

    if model_name == "rap_errnet":
        model: nn.Module = _build_rap_model_from_config(config_path)
        state = checkpoint.get("model", checkpoint.get("state_dict", checkpoint))
        _raise_if_rap_backbone_mismatch(state, model, ckpt_path, config_path)
        compatible = _filter_compatible(state, model)
        skipped = len(state) - len(compatible)
        if skipped:
            warnings.warn(f"Skipped {skipped} incompatible RAP-ERRNet checkpoint tensors.", RuntimeWarning)
        if not compatible:
            raise ValueError(f"No compatible RAP-ERRNet tensors were found in {ckpt_path}.")
        model.load_state_dict(compatible, strict=False)
    else:
        model = ERRNetHyperEvalWrapper() if errnet_hyper else ERRNetEvalWrapper()
        state = checkpoint.get("icnn", checkpoint.get("model", checkpoint.get("state_dict", checkpoint)))
        compatible = _filter_compatible(state, model.net)
        skipped = len(state) - len(compatible)
        if skipped:
            warnings.warn(
                f"Skipped {skipped} incompatible ERRNet tensors. Hypercolumn ERRNet checkpoints should still be "
                "evaluated with the original test_errnet.py unless local VGG feature support is added.",
                RuntimeWarning,
            )
        if not compatible:
            raise ValueError(f"No compatible ERRNet tensors were found in {ckpt_path}.")
        model.net.load_state_dict(compatible, strict=False)
    model.to(device)
    model.eval()
    return model


def _tensor_to_rgb(tensor):
    import numpy as np

    arr = tensor.detach().cpu().float().clamp(0.0, 1.0)
    if arr.ndim == 4:
        arr = arr[0]
    if arr.shape[0] == 1:
        arr = arr.repeat(3, 1, 1)
    arr = arr.permute(1, 2, 0).numpy()
    return (arr * 255.0).round().astype(np.uint8)


def _center_crop_array(arr, size: tuple[int, int]):
    h, w = size
    top = max((arr.shape[0] - h) // 2, 0)
    left = max((arr.shape[1] - w) // 2, 0)
    return arr[top : top + h, left : left + w]


def save_visualization(
    save_path: Path,
    input_tensor,
    output_tensor,
    target_tensor,
    prior_tensor: Optional[Any],
    baseline_tensor: Optional[Any] = None,
) -> None:
    import numpy as np
    from PIL import Image

    input_img = _tensor_to_rgb(input_tensor)
    baseline_img = _tensor_to_rgb(baseline_tensor) if baseline_tensor is not None else None
    output_img = _tensor_to_rgb(output_tensor)
    target_img = _tensor_to_rgb(target_tensor)
    base_images = [input_img, output_img, target_img]
    if baseline_img is not None:
        base_images.append(baseline_img)
    h = min(image.shape[0] for image in base_images)
    w = min(image.shape[1] for image in base_images)
    input_img = _center_crop_array(input_img, (h, w))
    if baseline_img is not None:
        baseline_img = _center_crop_array(baseline_img, (h, w))
    output_img = _center_crop_array(output_img, (h, w))
    target_img = _center_crop_array(target_img, (h, w))
    error = np.abs(output_img.astype(np.float32) - target_img.astype(np.float32)).mean(axis=2)
    error = np.clip(error / max(error.max(), 1.0), 0.0, 1.0)
    error_rgb = np.stack([error, np.zeros_like(error), 1.0 - error], axis=2)
    error_rgb = (error_rgb * 255.0).astype(np.uint8)
    if prior_tensor is None:
        prior_rgb = np.zeros_like(input_img)
    else:
        prior = _tensor_to_rgb(prior_tensor)
        prior_rgb = _center_crop_array(prior, (h, w))
    columns = [input_img]
    if baseline_img is not None:
        columns.append(baseline_img)
    columns.extend([output_img, target_img, error_rgb, prior_rgb])
    canvas = np.concatenate(columns, axis=1)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(canvas).save(save_path)


def _save_output_image(save_path: Path, output_tensor) -> None:
    from PIL import Image

    save_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(_tensor_to_rgb(output_tensor)).save(save_path)


def evaluate_dataset(
    model,
    dataset_name: str,
    data_root: Path,
    save_dir: Path,
    device,
    save_images: bool,
    baseline_model=None,
) -> List[Dict[str, Any]]:
    import torch
    from torch.utils.data import DataLoader

    from datasets.unified_reflection_dataset import UnifiedReflectionDataset
    from metrics.reflection_metrics import compute_metrics, save_metrics_csv

    try:
        dataset = UnifiedReflectionDataset(data_root, dataset_name, crop_size=None, image_size=None)
    except FileNotFoundError as exc:
        warnings.warn(f"Skipping {dataset_name}: {exc}", RuntimeWarning)
        return []

    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    rows: List[Dict[str, Any]] = []
    for batch in loader:
        input_tensor = batch["input"].to(device)
        target_tensor = batch["target"].to(device)
        name = batch["name"][0] if isinstance(batch["name"], (list, tuple)) else str(batch["name"])
        with torch.no_grad():
            outputs = model(input_tensor)
        output = outputs["output"].clamp(0.0, 1.0)
        baseline_output = None
        if baseline_model is not None:
            with torch.no_grad():
                baseline_outputs = baseline_model(input_tensor)
            baseline_output = baseline_outputs["output"].clamp(0.0, 1.0)
        metric_values = compute_metrics(output[0], target_tensor[0])
        row = {"dataset": dataset_name, "name": name, **metric_values}
        rows.append(row)

        if save_images:
            _save_output_image(save_dir / "outputs" / dataset_name / f"{name}.png", output[0])
            if baseline_output is not None:
                _save_output_image(save_dir / "baseline_outputs" / dataset_name / f"{name}.png", baseline_output[0])
            save_visualization(
                save_dir / "visualizations" / dataset_name / f"{name}.png",
                input_tensor[0],
                output[0],
                target_tensor[0],
                outputs.get("prior", None)[0] if outputs.get("prior", None) is not None else None,
                baseline_output[0] if baseline_output is not None else None,
            )

    save_metrics_csv(rows, save_dir / f"metrics_{dataset_name}.csv")
    return rows


def main() -> None:
    args = parse_args()
    from metrics.reflection_metrics import save_metrics_csv

    device = choose_device(args.device)
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    model = load_model(args.model, Path(args.ckpt), device, config_path=args.config)
    baseline_model = None
    if args.baseline_ckpt:
        if not args.save_images:
            warnings.warn("--baseline_ckpt only affects saved visualizations; use --save_images to write comparison images.", RuntimeWarning)
        baseline_model = load_model("errnet", Path(args.baseline_ckpt), device, errnet_hyper=args.baseline_hyper)

    all_rows: List[Dict[str, Any]] = []
    dataset_names = [item.strip() for item in args.datasets.split(",") if item.strip()]
    for dataset_name in dataset_names:
        rows = evaluate_dataset(
            model,
            dataset_name,
            Path(args.data_root),
            save_dir,
            device,
            args.save_images,
            baseline_model=baseline_model,
        )
        all_rows.extend(rows)

    if all_rows:
        save_metrics_csv(all_rows, save_dir / "metrics_all.csv")
    else:
        warnings.warn("No datasets were evaluated. Check --data_root and --datasets.", RuntimeWarning)


if __name__ == "__main__":
    main()
