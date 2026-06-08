"""Create final qualitative paper figures with formal method comparisons."""

from __future__ import annotations

import argparse
import csv
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import numpy as np
from PIL import Image, ImageDraw


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build paper qualitative figures 4-6.")
    parser.add_argument("--data_root", default="./data")
    parser.add_argument("--figures_dir", default="paper/figures")
    parser.add_argument("--out_dir", default="results/paper_qualitative")
    parser.add_argument("--max_long_edge", type=int, default=512)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--errnet_ckpt", default="checkpoints/errnet/errnet_060_00463920.pt")
    parser.add_argument("--errnet_hyper", action="store_true")
    parser.add_argument("--bprap_ckpt", default="checkpoints/bp_rap_hyper_ric_ft_from_hyper_ppu_bs24/best.pt")
    parser.add_argument("--bprap_config", default="configs/bp_rap_hyper_zerores_staged_ric.yaml")
    parser.add_argument("--rafa_ckpt", default="checkpoints/bp_rap_rafa_openrr3k_balanced_e10/best.pt")
    parser.add_argument("--rafa_config", default="configs/bp_rap_rafa_openrr3k_balanced.yaml")
    parser.add_argument("--rafa_label", default="Balanced RAFA3k")
    parser.add_argument("--cell_width", type=int, default=260)
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


def _heat_abs(diff: np.ndarray, scale: float) -> np.ndarray:
    value = np.clip(diff / max(float(scale), 1e-6), 0.0, 1.0)
    return np.stack([value, np.zeros_like(value), 1.0 - value], axis=2).round().astype(np.float32) * 255.0


def _heat_prior(prior: np.ndarray) -> np.ndarray:
    value = np.clip(prior, 0.0, 1.0)
    red = value
    green = 0.25 + 0.55 * (1.0 - np.abs(value - 0.5) * 2.0)
    blue = 1.0 - value
    return (np.stack([red, green, blue], axis=2) * 255.0).round().astype(np.uint8)


def _error_abs(output, target) -> np.ndarray:
    return (output.detach().float().cpu() - target.detach().float().cpu()).abs().mean(dim=0).numpy()


def _draw_grid(rows: Sequence[Sequence[np.ndarray]], labels: Sequence[str], titles: Sequence[str], cell_width: int) -> Image.Image:
    fitted_rows: List[List[Image.Image]] = []
    for row in rows:
        min_h = min(item.shape[0] for item in row)
        min_w = min(item.shape[1] for item in row)
        fitted_rows.append([_fit_cell(_center_crop(item, min_h, min_w), cell_width) for item in row])
    cell_heights = [max(cell.height for cell in row) for row in fitted_rows]
    label_h = 28
    title_h = 28
    width = cell_width * len(labels)
    height = sum(title_h + label_h + h for h in cell_heights)
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    y = 0
    for row_index, row in enumerate(fitted_rows):
        draw.text((8, y + 7), titles[row_index], fill=(20, 20, 20))
        y += title_h
        for index, label in enumerate(labels):
            draw.text((index * cell_width + 8, y + 7), label, fill=(20, 20, 20))
        y += label_h
        for index, cell in enumerate(row):
            canvas.paste(cell, (index * cell_width, y))
        y += cell_heights[row_index]
    return canvas


def _save(image: Image.Image, figures_dir: Path, name: str) -> None:
    figures_dir.mkdir(parents=True, exist_ok=True)
    png = figures_dir / f"{name}.png"
    pdf = figures_dir / f"{name}.pdf"
    image.save(png)
    image.save(pdf)
    print(png)
    print(pdf)


def _load_dataset(data_root: Path, name: str, max_long_edge: int):
    from datasets.unified_reflection_dataset import UnifiedReflectionDataset

    return UnifiedReflectionDataset(data_root, name, crop_size=None, image_size=None, max_long_edge=max_long_edge)


def _score_dataset(dataset_name: str, dataset, errnet, rafa, device) -> List[Dict[str, Any]]:
    from metrics.reflection_metrics import compute_metrics

    records: List[Dict[str, Any]] = []
    for index in range(len(dataset)):
        sample = dataset[index]
        input_tensor = sample["input"].unsqueeze(0).to(device)
        target = sample["target"]
        err_out = _resize_chw(_forward(errnet, input_tensor)["output"][0].clamp(0.0, 1.0), target.shape[-2:])
        rafa_out = _resize_chw(_forward(rafa, input_tensor)["output"][0].clamp(0.0, 1.0), target.shape[-2:])
        err_metrics = compute_metrics(err_out, target)
        rafa_metrics = compute_metrics(rafa_out, target)
        records.append(
            {
                "dataset": dataset_name,
                "index": index,
                "name": str(sample["name"]),
                "errnet_psnr": float(err_metrics["PSNR"]),
                "rafa_psnr": float(rafa_metrics["PSNR"]),
                "delta_psnr": float(rafa_metrics["PSNR"]) - float(err_metrics["PSNR"]),
            }
        )
    return records


def _select(records: Sequence[Mapping[str, Any]], mode: str) -> Optional[Mapping[str, Any]]:
    if not records:
        return None
    ordered = sorted(records, key=lambda row: float(row["delta_psnr"]))
    if mode == "worst":
        return ordered[0]
    improved = [row for row in ordered if float(row["delta_psnr"]) > 0]
    pool = improved if improved else ordered
    if mode == "median":
        return pool[len(pool) // 2]
    return pool[-1]


def _render_outputs(record: Mapping[str, Any], datasets: Mapping[str, Any], errnet, bprap, rafa, device):
    dataset = datasets[str(record["dataset"])]
    sample = dataset[int(record["index"])]
    input_tensor = sample["input"].unsqueeze(0).to(device)
    target = sample["target"]
    err_out = _resize_chw(_forward(errnet, input_tensor)["output"][0].clamp(0.0, 1.0), target.shape[-2:])
    bp_out = _resize_chw(_forward(bprap, input_tensor)["output"][0].clamp(0.0, 1.0), target.shape[-2:])
    rafa_outputs = _forward(rafa, input_tensor)
    rafa_out = _resize_chw(rafa_outputs["output"][0].clamp(0.0, 1.0), target.shape[-2:])
    prior = rafa_outputs.get("prior")
    if prior is not None:
        prior_tensor = _resize_chw(prior[0].detach().float().cpu().clamp(0.0, 1.0), target.shape[-2:])
    else:
        import torch

        prior_tensor = torch.zeros_like(target[:1])
    return {
        "input": sample["input"],
        "target": target,
        "errnet": err_out,
        "bprap": bp_out,
        "rafa": rafa_out,
        "prior": prior_tensor,
        "title": f"{record['dataset']}/{record['name']}  delta={float(record['delta_psnr']):+.2f} dB",
    }


def _placeholder_row(cols: int, height: int = 180, text: str = "missing self-collected data") -> List[np.ndarray]:
    row = []
    for _ in range(cols):
        image = Image.new("RGB", (260, height), (245, 245, 245))
        draw = ImageDraw.Draw(image)
        draw.text((18, height // 2 - 8), text, fill=(80, 80, 80))
        row.append(np.asarray(image))
    return row


def main() -> None:
    args = parse_args()
    from eval_all import choose_device, load_model

    device = choose_device(args.device)
    data_root = Path(args.data_root)
    out_dir = Path(args.out_dir)
    figures_dir = Path(args.figures_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    errnet = load_model("errnet", Path(args.errnet_ckpt), device, errnet_hyper=args.errnet_hyper)
    bprap = load_model("rap_errnet", Path(args.bprap_ckpt), device, config_path=args.bprap_config)
    rafa = load_model("rap_errnet", Path(args.rafa_ckpt), device, config_path=args.rafa_config)

    dataset_names = ["sir2_wild", "sir2_objects", "sir2_postcard", "openrr_val", "self", "ceilnet", "zhang20"]
    datasets: Dict[str, Any] = {}
    scores: Dict[str, List[Dict[str, Any]]] = {}
    for name in dataset_names:
        try:
            datasets[name] = _load_dataset(data_root, name, int(args.max_long_edge))
            scores[name] = _score_dataset(name, datasets[name], errnet, rafa, device)
        except FileNotFoundError as exc:
            warnings.warn(str(exc), RuntimeWarning)
            scores[name] = []

    selections: List[Mapping[str, Any]] = []
    success_specs = [
        ("sir2_wild", "top", "SIR2 Wild success"),
        ("sir2_objects,sir2_postcard", "median", "SIR2 Objects/Postcard median success"),
        ("openrr_val", "top", "OpenRR success"),
        ("self", "top", "Self-collected success"),
    ]
    q_rows: List[List[np.ndarray]] = []
    q_titles: List[str] = []
    for dataset_group, mode, label in success_specs:
        records: List[Mapping[str, Any]] = []
        for name in dataset_group.split(","):
            records.extend(scores.get(name, []))
        record = _select(records, mode)
        if record is None:
            q_rows.append(_placeholder_row(5))
            q_titles.append(label)
            continue
        selections.append({"figure": "qualitative_main", "slot": label, **dict(record)})
        outputs = _render_outputs(record, datasets, errnet, bprap, rafa, device)
        q_rows.append([_to_rgb(outputs[key]) for key in ["input", "errnet", "bprap", "rafa", "target"]])
        q_titles.append(label + "  " + str(outputs["title"]))
    _save(_draw_grid(q_rows, ["Input", "ERRNet", "BP-RAP RIC", args.rafa_label, "GT"], q_titles, int(args.cell_width)), figures_dir, "qualitative_main")

    analysis_records: List[Mapping[str, Any]] = []
    for name, mode in [("sir2_wild", "top"), ("openrr_val", "top"), ("self", "top")]:
        record = _select(scores.get(name, []), mode)
        if record is not None:
            analysis_records.append(record)
    e_rows: List[List[np.ndarray]] = []
    e_titles: List[str] = []
    for record in analysis_records:
        selections.append({"figure": "prior_error_analysis", "slot": str(record["dataset"]), **dict(record)})
        outputs = _render_outputs(record, datasets, errnet, bprap, rafa, device)
        err_e = _error_abs(outputs["errnet"], outputs["target"])
        bp_e = _error_abs(outputs["bprap"], outputs["target"])
        rafa_e = _error_abs(outputs["rafa"], outputs["target"])
        scale = max(float(err_e.max()), float(bp_e.max()), float(rafa_e.max()), 1e-6)
        prior = outputs["prior"].detach().float().cpu().squeeze(0).numpy()
        e_rows.append([
            _to_rgb(outputs["input"]),
            _to_rgb(outputs["target"]),
            _heat_abs(err_e, scale).astype(np.uint8),
            _heat_abs(bp_e, scale).astype(np.uint8),
            _heat_abs(rafa_e, scale).astype(np.uint8),
            _heat_prior(prior),
        ])
        e_titles.append(str(outputs["title"]))
    if e_rows:
        _save(_draw_grid(e_rows, ["Input", "GT", "ERRNet Error", "BP-RAP Error", "RAFA Error", "Prior Map"], e_titles, int(args.cell_width)), figures_dir, "prior_error_analysis")

    f_rows: List[List[np.ndarray]] = []
    f_titles: List[str] = []
    for name, label in [("ceilnet", "CEILNet failure"), ("zhang20", "Zhang20 failure")]:
        record = _select(scores.get(name, []), "worst")
        if record is None:
            f_rows.append(_placeholder_row(5))
            f_titles.append(label)
            continue
        selections.append({"figure": "failure_cases", "slot": label, **dict(record)})
        outputs = _render_outputs(record, datasets, errnet, bprap, rafa, device)
        f_rows.append([_to_rgb(outputs[key]) for key in ["input", "errnet", "bprap", "rafa", "target"]])
        f_titles.append(label + "  " + str(outputs["title"]))
    _save(_draw_grid(f_rows, ["Input", "ERRNet", "BP-RAP RIC", args.rafa_label, "GT"], f_titles, int(args.cell_width)), figures_dir, "failure_cases")

    with (out_dir / "selection_summary.csv").open("w", newline="", encoding="utf-8") as f:
        fieldnames = ["figure", "slot", "dataset", "index", "name", "errnet_psnr", "rafa_psnr", "delta_psnr"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(selections)
    print(out_dir / "selection_summary.csv")


if __name__ == "__main__":
    main()
