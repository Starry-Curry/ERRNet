"""Build detailed CEILNet/Zhang20 qualitative diagnostic sheets.

The script compares ERRNet, BP-RAP RIC, and the selected RAFA/final model. It
selects examples by PSNR delta against ERRNet and writes both contact sheets and
a CSV with per-sample metrics.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import numpy as np
from PIL import Image, ImageDraw


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create CEILNet/Zhang20 failure analysis sheets.")
    parser.add_argument("--data_root", default="./data")
    parser.add_argument("--datasets", default="ceilnet,zhang20")
    parser.add_argument("--out_dir", default="results/ceilnet_zhang_diagnostics")
    parser.add_argument("--figures_dir", default="paper/figures")
    parser.add_argument("--max_long_edge", type=int, default=512)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--worst_k", type=int, default=8)
    parser.add_argument("--similar_k", type=int, default=4)
    parser.add_argument("--better_k", type=int, default=4)
    parser.add_argument("--cell_width", type=int, default=190)
    parser.add_argument("--errnet_ckpt", default="checkpoints/errnet/errnet_060_00463920.pt")
    parser.add_argument("--errnet_hyper", action="store_true")
    parser.add_argument("--bprap_ckpt", default="checkpoints/bp_rap_hyper_ric_ft_from_hyper_ppu_bs24/best.pt")
    parser.add_argument("--bprap_config", default="configs/bp_rap_hyper_zerores_staged_ric.yaml")
    parser.add_argument("--rafa_ckpt", default="checkpoints/bp_rap_rafa_openrr3k_no_old_e10/best.pt")
    parser.add_argument("--rafa_config", default="configs/bp_rap_rafa_openrr3k_no_old.yaml")
    parser.add_argument("--rafa_label", default="RAFA3k-no-old")
    return parser.parse_args()


def _to_rgb(tensor) -> np.ndarray:
    array = tensor.detach().float().cpu().clamp(0.0, 1.0)
    if array.ndim != 3:
        raise ValueError(f"expected CHW tensor, got {tuple(array.shape)}")
    if array.shape[0] == 1:
        array = array.repeat(3, 1, 1)
    return (array.permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)


def _resize_chw(tensor, size):
    import torch.nn.functional as F

    if tuple(tensor.shape[-2:]) == tuple(size):
        return tensor
    return F.interpolate(tensor.unsqueeze(0), size=tuple(size), mode="bilinear", align_corners=False)[0]


def _forward(model, input_tensor):
    import torch

    with torch.no_grad():
        return model(input_tensor)


def _center_crop(array: np.ndarray, height: int, width: int) -> np.ndarray:
    top = max((array.shape[0] - height) // 2, 0)
    left = max((array.shape[1] - width) // 2, 0)
    return array[top : top + height, left : left + width]


def _fit_cell(array: np.ndarray, width: int) -> Image.Image:
    image = Image.fromarray(array)
    if image.width == width:
        return image
    height = max(1, int(round(image.height * width / float(image.width))))
    return image.resize((width, height), Image.Resampling.BICUBIC)


def _error_abs(output, target) -> np.ndarray:
    return (output.detach().float().cpu() - target.detach().float().cpu()).abs().mean(dim=0).numpy()


def _heat_abs(diff: np.ndarray, scale: float) -> np.ndarray:
    value = np.clip(diff / max(float(scale), 1e-6), 0.0, 1.0)
    rgb = np.stack([value, np.zeros_like(value), 1.0 - value], axis=2)
    return (rgb * 255.0).round().astype(np.uint8)


def _heat_prior(prior: np.ndarray) -> np.ndarray:
    value = np.clip(prior, 0.0, 1.0)
    red = value
    green = 0.25 + 0.55 * (1.0 - np.abs(value - 0.5) * 2.0)
    blue = 1.0 - value
    return (np.stack([red, green, blue], axis=2) * 255.0).round().astype(np.uint8)


def _draw_grid(rows: Sequence[Sequence[np.ndarray]], labels: Sequence[str], titles: Sequence[str], cell_width: int) -> Image.Image:
    fitted_rows: List[List[Image.Image]] = []
    for row in rows:
        min_h = min(item.shape[0] for item in row)
        min_w = min(item.shape[1] for item in row)
        fitted_rows.append([_fit_cell(_center_crop(item, min_h, min_w), cell_width) for item in row])
    cell_heights = [max(cell.height for cell in row) for row in fitted_rows]
    label_h = 28
    title_h = 30
    width = cell_width * len(labels)
    height = sum(title_h + label_h + h for h in cell_heights)
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    y = 0
    for row_index, row in enumerate(fitted_rows):
        draw.text((8, y + 8), titles[row_index], fill=(20, 20, 20))
        y += title_h
        for index, label in enumerate(labels):
            draw.text((index * cell_width + 8, y + 7), label, fill=(20, 20, 20))
        y += label_h
        for index, cell in enumerate(row):
            canvas.paste(cell, (index * cell_width, y))
        y += cell_heights[row_index]
    return canvas


def _save(image: Image.Image, path_base: Path) -> None:
    path_base.parent.mkdir(parents=True, exist_ok=True)
    image.save(path_base.with_suffix(".png"))
    image.save(path_base.with_suffix(".pdf"))
    print(path_base.with_suffix(".png"))
    print(path_base.with_suffix(".pdf"))


def _load_dataset(data_root: Path, name: str, max_long_edge: int):
    from datasets.unified_reflection_dataset import UnifiedReflectionDataset

    resize = max_long_edge if name == "zhang20" else None
    return UnifiedReflectionDataset(data_root, name, crop_size=None, image_size=None, max_long_edge=resize)


def _unique(records: Iterable[Mapping[str, Any]]) -> List[Mapping[str, Any]]:
    seen = set()
    out = []
    for record in records:
        key = (record["dataset"], int(record["index"]))
        if key in seen:
            continue
        seen.add(key)
        out.append(record)
    return out


def _select_records(records: Sequence[Mapping[str, Any]], worst_k: int, similar_k: int, better_k: int) -> List[Mapping[str, Any]]:
    ordered = sorted(records, key=lambda row: float(row["delta_rafa_vs_errnet"]))
    worst = [{"group": "worst", **dict(row)} for row in ordered[:worst_k]]
    similar = [{"group": "similar", **dict(row)} for row in sorted(records, key=lambda row: abs(float(row["delta_rafa_vs_errnet"])))[:similar_k]]
    positive = [row for row in ordered if float(row["delta_rafa_vs_errnet"]) > 0]
    better = [{"group": "better", **dict(row)} for row in positive[-better_k:][::-1]]
    return _unique([*worst, *similar, *better])


def _score_dataset(dataset_name: str, dataset, errnet, bprap, rafa, device) -> List[Dict[str, Any]]:
    from metrics.reflection_metrics import compute_metrics

    records = []
    for index in range(len(dataset)):
        sample = dataset[index]
        input_tensor = sample["input"].unsqueeze(0).to(device)
        target = sample["target"]
        target_size = target.shape[-2:]
        err = _resize_chw(_forward(errnet, input_tensor)["output"][0].clamp(0.0, 1.0), target_size)
        bp = _resize_chw(_forward(bprap, input_tensor)["output"][0].clamp(0.0, 1.0), target_size)
        rf = _resize_chw(_forward(rafa, input_tensor)["output"][0].clamp(0.0, 1.0), target_size)
        em = compute_metrics(err, target)
        bm = compute_metrics(bp, target)
        rm = compute_metrics(rf, target)
        records.append(
            {
                "dataset": dataset_name,
                "index": index,
                "name": str(sample["name"]),
                "errnet_psnr": float(em["PSNR"]),
                "bprap_psnr": float(bm["PSNR"]),
                "rafa_psnr": float(rm["PSNR"]),
                "delta_bprap_vs_errnet": float(bm["PSNR"]) - float(em["PSNR"]),
                "delta_rafa_vs_errnet": float(rm["PSNR"]) - float(em["PSNR"]),
                "errnet_ssim": float(em["SSIM"]),
                "bprap_ssim": float(bm["SSIM"]),
                "rafa_ssim": float(rm["SSIM"]),
                "errnet_lmse": float(em["LMSE"]),
                "bprap_lmse": float(bm["LMSE"]),
                "rafa_lmse": float(rm["LMSE"]),
            }
        )
    return records


def _render_record(record: Mapping[str, Any], dataset, errnet, bprap, rafa, device):
    sample = dataset[int(record["index"])]
    input_tensor = sample["input"].unsqueeze(0).to(device)
    target = sample["target"]
    target_size = target.shape[-2:]
    err = _resize_chw(_forward(errnet, input_tensor)["output"][0].clamp(0.0, 1.0), target_size)
    bp = _resize_chw(_forward(bprap, input_tensor)["output"][0].clamp(0.0, 1.0), target_size)
    rafa_outputs = _forward(rafa, input_tensor)
    rf = _resize_chw(rafa_outputs["output"][0].clamp(0.0, 1.0), target_size)
    prior = rafa_outputs.get("prior")
    if prior is not None:
        prior_tensor = _resize_chw(prior[0].detach().float().cpu().clamp(0.0, 1.0), target_size)
    else:
        import torch

        prior_tensor = torch.zeros_like(target[:1])

    err_e = _error_abs(err, target)
    bp_e = _error_abs(bp, target)
    rf_e = _error_abs(rf, target)
    scale = max(float(err_e.max()), float(bp_e.max()), float(rf_e.max()), 1e-6)
    prior_np = prior_tensor.detach().float().cpu().squeeze(0).numpy()
    return [
        _to_rgb(sample["input"]),
        _to_rgb(target),
        _to_rgb(err),
        _to_rgb(bp),
        _to_rgb(rf),
        _heat_abs(err_e, scale),
        _heat_abs(bp_e, scale),
        _heat_abs(rf_e, scale),
        _heat_prior(prior_np),
    ]


def _write_csv(path: Path, records: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "dataset",
        "group",
        "index",
        "name",
        "errnet_psnr",
        "bprap_psnr",
        "rafa_psnr",
        "delta_bprap_vs_errnet",
        "delta_rafa_vs_errnet",
        "errnet_ssim",
        "bprap_ssim",
        "rafa_ssim",
        "errnet_lmse",
        "bprap_lmse",
        "rafa_lmse",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)
    print(path)


def main() -> None:
    args = parse_args()
    from eval_all import choose_device, load_model

    device = choose_device(args.device)
    data_root = Path(args.data_root)
    out_dir = Path(args.out_dir)
    figures_dir = Path(args.figures_dir)
    errnet = load_model("errnet", Path(args.errnet_ckpt), device, errnet_hyper=args.errnet_hyper)
    bprap = load_model("rap_errnet", Path(args.bprap_ckpt), device, config_path=args.bprap_config)
    rafa = load_model("rap_errnet", Path(args.rafa_ckpt), device, config_path=args.rafa_config)

    labels = [
        "Input",
        "GT",
        "ERRNet",
        "BP-RAP",
        args.rafa_label,
        "ERR Err",
        "BP Err",
        "RAFA Err",
        "Prior",
    ]
    all_selected: List[Mapping[str, Any]] = []
    for dataset_name in [item.strip() for item in args.datasets.split(",") if item.strip()]:
        dataset = _load_dataset(data_root, dataset_name, int(args.max_long_edge))
        records = _score_dataset(dataset_name, dataset, errnet, bprap, rafa, device)
        selected = _select_records(records, int(args.worst_k), int(args.similar_k), int(args.better_k))
        all_selected.extend(selected)

        for group_name in ("worst", "similar", "better"):
            group_records = [row for row in selected if row.get("group") == group_name]
            if not group_records:
                continue
            rows = []
            titles = []
            for record in group_records:
                rows.append(_render_record(record, dataset, errnet, bprap, rafa, device))
                titles.append(
                    f"{dataset_name}/{record['name']} [{group_name}] "
                    f"ERR {float(record['errnet_psnr']):.2f}, BP {float(record['bprap_psnr']):.2f}, "
                    f"RAFA {float(record['rafa_psnr']):.2f}, dRAFA {float(record['delta_rafa_vs_errnet']):+.2f} dB"
                )
            sheet = _draw_grid(rows, labels, titles, int(args.cell_width))
            _save(sheet, out_dir / "contact_sheets" / f"{dataset_name}_{group_name}")
            _save(sheet, figures_dir / f"{dataset_name}_{group_name}_diagnostics")

        mixed = selected
        rows = [_render_record(record, dataset, errnet, bprap, rafa, device) for record in mixed]
        titles = [
            f"{dataset_name}/{record['name']} [{record['group']}] "
            f"ERR {float(record['errnet_psnr']):.2f}, BP {float(record['bprap_psnr']):.2f}, "
            f"RAFA {float(record['rafa_psnr']):.2f}, dRAFA {float(record['delta_rafa_vs_errnet']):+.2f} dB"
            for record in mixed
        ]
        mixed_sheet = _draw_grid(rows, labels, titles, int(args.cell_width))
        _save(mixed_sheet, out_dir / "contact_sheets" / f"{dataset_name}_mixed")
        _save(mixed_sheet, figures_dir / f"{dataset_name}_mixed_diagnostics")

    _write_csv(out_dir / "selection_summary.csv", all_selected)


if __name__ == "__main__":
    main()
