"""Create final multi-method visual comparison sheets.

The script is intended for report figures after metric selection. It compares:

Input | ERRNet | BP-RAP RIC | RAFA | GT | Error | Prior
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

import numpy as np
from PIL import Image, ImageDraw


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


DEFAULT_DATASETS = "ceilnet,zhang20,sir2_objects,sir2_postcard,sir2_wild,openrr_val"
GROUP_ORDER = ("better", "similar", "worse")
GROUP_TITLES = {
    "better": "Better than ERRNet",
    "similar": "Similar to ERRNet",
    "worse": "Worse than ERRNet",
    "even": "Evenly sampled examples",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build final ERRNet/BP-RAP/RAFA visual comparison sheets.")
    parser.add_argument("--data_root", default="./data")
    parser.add_argument("--datasets", default=DEFAULT_DATASETS)
    parser.add_argument("--out_dir", default="results/final_visuals")
    parser.add_argument("--max_images", type=int, default=4, help="number of samples per dataset")
    parser.add_argument("--max_long_edge", type=int, default=512)
    parser.add_argument("--row_width", type=int, default=2450, help="resize each comparison row to this width")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument(
        "--selection_mode",
        choices=["even", "delta_groups"],
        default="even",
        help="even selects evenly spaced samples; delta_groups builds one per-dataset figure with better/similar/worse sections",
    )
    parser.add_argument("--group_size", type=int, default=2, help="samples per better/similar/worse group")
    parser.add_argument("--similar_abs_delta", type=float, default=0.2, help="PSNR delta threshold for similar group")

    parser.add_argument("--errnet_ckpt", default="checkpoints/errnet/errnet_060_00463920.pt")
    parser.add_argument("--errnet_hyper", action="store_true", help="load ERRNet baseline with hypercolumn input")
    parser.add_argument("--bprap_ckpt", default="checkpoints/bp_rap_hyper_ric_ft_from_hyper_ppu_bs24/best.pt")
    parser.add_argument("--bprap_config", default="configs/bp_rap_hyper_zerores_staged_ric.yaml")
    parser.add_argument("--rafa_ckpt", default="checkpoints/bp_rap_rafa_openrr3k_balanced_e10/best.pt")
    parser.add_argument("--rafa_config", default="configs/bp_rap_rafa_openrr3k_balanced.yaml")
    parser.add_argument("--rafa_label", default="RAFA-Balanced")
    return parser.parse_args()


def _select_indices(length: int, count: int) -> List[int]:
    if length <= 0 or count <= 0:
        return []
    if count >= length:
        return list(range(length))
    if count == 1:
        return [0]
    return sorted({int(round(i * (length - 1) / float(count - 1))) for i in range(count)})


def _tensor_to_rgb(tensor: torch.Tensor) -> np.ndarray:
    array = tensor.detach().float().cpu().clamp(0.0, 1.0)
    if array.ndim != 3:
        raise ValueError(f"expected CHW tensor, got {tuple(array.shape)}")
    if array.shape[0] == 1:
        array = array.repeat(3, 1, 1)
    return (array.permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)


def _resize_chw(tensor: torch.Tensor, size: Sequence[int]) -> torch.Tensor:
    import torch.nn.functional as F

    if tuple(tensor.shape[-2:]) == tuple(size):
        return tensor
    return F.interpolate(tensor.unsqueeze(0), size=tuple(size), mode="bilinear", align_corners=False)[0]


def _center_crop(image: np.ndarray, height: int, width: int) -> np.ndarray:
    top = max((image.shape[0] - height) // 2, 0)
    left = max((image.shape[1] - width) // 2, 0)
    return image[top : top + height, left : left + width]


def _error_map(output: torch.Tensor, target: torch.Tensor) -> np.ndarray:
    error = (output.detach().float().cpu() - target.detach().float().cpu()).abs().mean(dim=0).numpy()
    error = np.clip(error / max(float(error.max()), 1e-6), 0.0, 1.0)
    rgb = np.stack([error, np.zeros_like(error), 1.0 - error], axis=2)
    return (rgb * 255.0).round().astype(np.uint8)


def _resize_width(image: Image.Image, width: int) -> Image.Image:
    if width <= 0 or image.width == width:
        return image
    height = max(1, int(round(image.height * width / float(image.width))))
    try:
        resample = Image.Resampling.BICUBIC
    except AttributeError:  # Pillow < 9
        resample = Image.BICUBIC
    return image.resize((width, height), resample)


def _make_row(columns: Sequence[np.ndarray], labels: Sequence[str], row_width: int, title: str | None = None) -> Image.Image:
    height = min(column.shape[0] for column in columns)
    width = min(column.shape[1] for column in columns)
    cropped = [_center_crop(column, height, width) for column in columns]
    header_h = 52 if title else 28
    canvas = Image.new("RGB", (width * len(cropped), height + header_h), "white")
    draw = ImageDraw.Draw(canvas)
    label_y = 31 if title else 7
    if title:
        draw.text((8, 7), title, fill=(20, 20, 20))
    for idx, (column, label) in enumerate(zip(cropped, labels)):
        x = idx * width
        draw.text((x + 8, label_y), label, fill=(20, 20, 20))
        canvas.paste(Image.fromarray(column), (x, header_h))
    return _resize_width(canvas, row_width)


def _stack_rows(rows: Sequence[Image.Image], output_path: Path) -> None:
    if not rows:
        return
    width = max(row.width for row in rows)
    height = sum(row.height for row in rows)
    canvas = Image.new("RGB", (width, height), "white")
    y = 0
    for row in rows:
        canvas.paste(row, (0, y))
        y += row.height
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def _section_header(width: int, text: str) -> Image.Image:
    height = 40
    canvas = Image.new("RGB", (width, height), (245, 245, 245))
    draw = ImageDraw.Draw(canvas)
    draw.text((12, 12), text, fill=(20, 20, 20))
    return canvas


def _stack_grouped_rows(group_rows: Mapping[str, Sequence[Image.Image]], output_path: Path) -> None:
    ordered_rows: List[Image.Image] = []
    width = max((row.width for rows in group_rows.values() for row in rows), default=0)
    if width <= 0:
        return
    for group_name in [*GROUP_ORDER, "even"]:
        rows = list(group_rows.get(group_name, []))
        if not rows:
            continue
        ordered_rows.append(_section_header(width, GROUP_TITLES.get(group_name, group_name)))
        ordered_rows.extend(rows)
    _stack_rows(ordered_rows, output_path)


def _forward(model, input_tensor: torch.Tensor) -> Dict[str, torch.Tensor]:
    import torch

    with torch.no_grad():
        outputs = model(input_tensor)
    return outputs


def _score_dataset(
    dataset,
    errnet,
    rafa,
    device,
) -> List[Dict[str, Any]]:
    from metrics.reflection_metrics import compute_metrics

    records: List[Dict[str, Any]] = []
    for index in range(len(dataset)):
        sample = dataset[index]
        input_tensor = sample["input"].unsqueeze(0).to(device)
        target = sample["target"]
        errnet_out = _resize_chw(_forward(errnet, input_tensor)["output"][0].clamp(0.0, 1.0), target.shape[-2:])
        rafa_out = _resize_chw(_forward(rafa, input_tensor)["output"][0].clamp(0.0, 1.0), target.shape[-2:])
        errnet_metrics = compute_metrics(errnet_out, target)
        rafa_metrics = compute_metrics(rafa_out, target)
        records.append(
            {
                "index": index,
                "name": str(sample["name"]),
                "errnet_psnr": float(errnet_metrics["PSNR"]),
                "rafa_psnr": float(rafa_metrics["PSNR"]),
                "delta_psnr": float(rafa_metrics["PSNR"]) - float(errnet_metrics["PSNR"]),
                "errnet_ssim": float(errnet_metrics["SSIM"]),
                "rafa_ssim": float(rafa_metrics["SSIM"]),
                "delta_ssim": float(rafa_metrics["SSIM"]) - float(errnet_metrics["SSIM"]),
            }
        )
    return records


def _take_unique(records: Sequence[Mapping[str, Any]], count: int) -> List[Mapping[str, Any]]:
    selected: List[Mapping[str, Any]] = []
    seen = set()
    for record in records:
        index = int(record["index"])
        if index in seen:
            continue
        selected.append(record)
        seen.add(index)
        if len(selected) >= count:
            break
    return selected


def _select_delta_groups(
    records: Sequence[Mapping[str, Any]],
    group_size: int,
    similar_abs_delta: float,
) -> Dict[str, List[Mapping[str, Any]]]:
    better = [row for row in records if float(row["delta_psnr"]) >= similar_abs_delta]
    worse = [row for row in records if float(row["delta_psnr"]) <= -similar_abs_delta]
    if len(better) < group_size:
        better = [row for row in records if float(row["delta_psnr"]) > 0]
    if len(worse) < group_size:
        worse = [row for row in records if float(row["delta_psnr"]) < 0]
    return {
        "better": _take_unique(sorted(better, key=lambda row: float(row["delta_psnr"]), reverse=True), group_size),
        "similar": _take_unique(sorted(records, key=lambda row: abs(float(row["delta_psnr"]))), group_size),
        "worse": _take_unique(sorted(worse, key=lambda row: float(row["delta_psnr"])), group_size),
    }


def _write_selection_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "dataset",
        "group",
        "index",
        "name",
        "errnet_psnr",
        "rafa_psnr",
        "delta_psnr",
        "errnet_ssim",
        "rafa_ssim",
        "delta_ssim",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_visuals(args: argparse.Namespace) -> None:
    import torch

    from datasets.unified_reflection_dataset import UnifiedReflectionDataset
    from eval_all import choose_device, load_model

    device = choose_device(args.device)
    data_root = Path(args.data_root)
    out_dir = Path(args.out_dir)

    errnet = load_model("errnet", Path(args.errnet_ckpt), device, errnet_hyper=args.errnet_hyper)
    bprap = load_model("rap_errnet", Path(args.bprap_ckpt), device, config_path=args.bprap_config)
    rafa = load_model("rap_errnet", Path(args.rafa_ckpt), device, config_path=args.rafa_config)

    labels = ["Input", "ERRNet", "BP-RAP RIC", args.rafa_label, "GT", "Error", "Prior"]
    overview_rows: List[Image.Image] = []
    selection_rows: List[Dict[str, Any]] = []
    for dataset_name in [item.strip() for item in args.datasets.split(",") if item.strip()]:
        dataset = UnifiedReflectionDataset(
            data_root,
            dataset_name,
            crop_size=None,
            image_size=None,
            max_long_edge=args.max_long_edge,
        )
        dataset_rows: List[Image.Image] = []
        selected: List[Dict[str, Any]] = []
        if args.selection_mode == "delta_groups":
            records = _score_dataset(dataset, errnet, rafa, device)
            groups = _select_delta_groups(records, int(args.group_size), float(args.similar_abs_delta))
            for group_name in GROUP_ORDER:
                for record in groups[group_name]:
                    row = {"dataset": dataset_name, "group": group_name, **dict(record)}
                    selected.append(row)
                    selection_rows.append(row)
        else:
            selected = [
                {"dataset": dataset_name, "group": "even", "index": index, "name": ""}
                for index in _select_indices(len(dataset), args.max_images)
            ]

        group_rows: Dict[str, List[Image.Image]] = {}
        for selected_record in selected:
            index = int(selected_record["index"])
            sample = dataset[index]
            name = str(sample["name"])
            input_tensor = sample["input"].unsqueeze(0).to(device)
            target = sample["target"]

            errnet_out = _forward(errnet, input_tensor)["output"][0].clamp(0.0, 1.0)
            bprap_out = _forward(bprap, input_tensor)["output"][0].clamp(0.0, 1.0)
            rafa_outputs = _forward(rafa, input_tensor)
            rafa_out = rafa_outputs["output"][0].clamp(0.0, 1.0)
            prior = rafa_outputs.get("prior")
            prior_tensor = prior[0].detach().float().cpu().clamp(0.0, 1.0) if prior is not None else torch.zeros_like(target[:1])
            target_size = target.shape[-2:]
            errnet_out = _resize_chw(errnet_out, target_size)
            bprap_out = _resize_chw(bprap_out, target_size)
            rafa_out = _resize_chw(rafa_out, target_size)
            prior_tensor = _resize_chw(prior_tensor, target_size)

            columns = [
                _tensor_to_rgb(sample["input"]),
                _tensor_to_rgb(errnet_out),
                _tensor_to_rgb(bprap_out),
                _tensor_to_rgb(rafa_out),
                _tensor_to_rgb(target),
                _error_map(rafa_out, target),
                _tensor_to_rgb(prior_tensor),
            ]
            group_name = str(selected_record.get("group", "even"))
            delta = selected_record.get("delta_psnr")
            title = f"{dataset_name}/{name} [{group_name}]"
            if delta is not None:
                title += (
                    f"  PSNR: ERRNet {float(selected_record['errnet_psnr']):.2f}, "
                    f"{args.rafa_label} {float(selected_record['rafa_psnr']):.2f}, "
                    f"delta {float(delta):+.2f} dB"
                )
            row = _make_row(columns, labels, args.row_width, title=title)
            row_path = out_dir / "rows" / dataset_name / group_name / f"{name}.png"
            row_path.parent.mkdir(parents=True, exist_ok=True)
            row.save(row_path)
            dataset_rows.append(row)
            group_rows.setdefault(group_name, []).append(row)
            overview_rows.append(row)

        if args.selection_mode == "delta_groups":
            _stack_grouped_rows(group_rows, out_dir / "contact_sheets" / f"{dataset_name}.png")
        else:
            _stack_rows(dataset_rows, out_dir / "contact_sheets" / f"{dataset_name}.png")

    _stack_rows(overview_rows, out_dir / "contact_sheets" / "overview.png")
    if selection_rows:
        _write_selection_csv(out_dir / "selection_summary.csv", selection_rows)
    print(out_dir)


def main() -> None:
    build_visuals(parse_args())


if __name__ == "__main__":
    main()
