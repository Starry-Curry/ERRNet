"""Create qualitative figures for ERRNet-RAFA inference fusion."""

from __future__ import annotations

import argparse
import csv
import math
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
    parser = argparse.ArgumentParser(description="Build qualitative figures for ERRNet-RAFA fusion.")
    parser.add_argument("--data_root", default="./data")
    parser.add_argument("--datasets", default="ceilnet,zhang20,sir2_objects,sir2_postcard,sir2_wild,openrr_val,self")
    parser.add_argument("--figures_dir", default="paper/figures")
    parser.add_argument("--out_dir", default="results/fusion_qualitative")
    parser.add_argument("--max_long_edge", type=int, default=512)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--errnet_ckpt", default="checkpoints/errnet/errnet_060_00463920.pt")
    parser.add_argument("--errnet_hyper", action="store_true")
    parser.add_argument("--rafa_ckpt", default="checkpoints/bp_rap_rafa_openrr3k_no_old_e10/best.pt")
    parser.add_argument("--rafa_config", default="configs/bp_rap_rafa_openrr3k_no_old.yaml")
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--cell_width", type=int, default=245)
    return parser.parse_args()


def _center_crop_tensor(tensor, size: tuple[int, int]):
    h, w = size
    top = max((tensor.shape[-2] - h) // 2, 0)
    left = max((tensor.shape[-1] - w) // 2, 0)
    return tensor[..., top : top + h, left : left + w]


def _common_size(*tensors) -> tuple[int, int]:
    heights = [int(tensor.shape[-2]) for tensor in tensors if tensor is not None]
    widths = [int(tensor.shape[-1]) for tensor in tensors if tensor is not None]
    return min(heights), min(widths)


def _align_tensors(*tensors):
    size = _common_size(*tensors)
    return tuple(_center_crop_tensor(tensor, size) if tensor is not None else None for tensor in tensors)


def _to_rgb(tensor) -> np.ndarray:
    arr = tensor.detach().float().cpu().clamp(0.0, 1.0)
    if arr.ndim != 3:
        raise ValueError(f"expected CHW tensor, got {tuple(arr.shape)}")
    if arr.shape[0] == 1:
        arr = arr.repeat(3, 1, 1)
    return (arr.permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)


def _center_crop_array(arr: np.ndarray, height: int, width: int) -> np.ndarray:
    top = max((arr.shape[0] - height) // 2, 0)
    left = max((arr.shape[1] - width) // 2, 0)
    return arr[top : top + height, left : left + width]


def _fit_cell(arr: np.ndarray, width: int) -> Image.Image:
    image = Image.fromarray(arr)
    if image.width == width:
        return image
    height = max(1, int(round(image.height * width / float(image.width))))
    return image.resize((width, height), Image.Resampling.BICUBIC)


def _draw_grid(rows: Sequence[Sequence[np.ndarray]], labels: Sequence[str], titles: Sequence[str], cell_width: int) -> Image.Image:
    fitted_rows: List[List[Image.Image]] = []
    for row in rows:
        min_h = min(item.shape[0] for item in row)
        min_w = min(item.shape[1] for item in row)
        fitted_rows.append([_fit_cell(_center_crop_array(item, min_h, min_w), cell_width) for item in row])
    label_h = 28
    title_h = 30
    cell_heights = [max(cell.height for cell in row) for row in fitted_rows]
    canvas = Image.new("RGB", (cell_width * len(labels), sum(title_h + label_h + h for h in cell_heights)), "white")
    draw = ImageDraw.Draw(canvas)
    y = 0
    for row_index, row in enumerate(fitted_rows):
        draw.text((8, y + 8), titles[row_index], fill=(20, 20, 20))
        y += title_h
        for col, label in enumerate(labels):
            draw.text((col * cell_width + 8, y + 7), label, fill=(20, 20, 20))
        y += label_h
        for col, cell in enumerate(row):
            canvas.paste(cell, (col * cell_width, y))
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


def _forward(model, input_tensor):
    import torch

    with torch.no_grad():
        return model(input_tensor)


def _predict(sample: Mapping[str, Any], errnet, rafa, device, alpha: float) -> Dict[str, Any]:
    input_tensor = sample["input"].unsqueeze(0).to(device)
    target = sample["target"].to(device)
    err = _forward(errnet, input_tensor)["output"].clamp(0.0, 1.0)
    rafa_outputs = _forward(rafa, input_tensor)
    rafa_out = rafa_outputs["output"].clamp(0.0, 1.0)
    input_aligned, target_aligned, err, rafa_out = _align_tensors(input_tensor, target.unsqueeze(0), err, rafa_out)
    fusion = (float(alpha) * rafa_out + (1.0 - float(alpha)) * err).clamp(0.0, 1.0)
    return {
        "input": input_aligned[0].detach().cpu(),
        "target": target_aligned[0].detach().cpu(),
        "errnet": err[0].detach().cpu(),
        "rafa": rafa_out[0].detach().cpu(),
        "fusion": fusion[0].detach().cpu(),
    }


def _score_dataset(dataset_name: str, dataset, errnet, rafa, device, alpha: float) -> List[Dict[str, Any]]:
    from metrics.reflection_metrics import compute_metrics

    rows: List[Dict[str, Any]] = []
    for index in range(len(dataset)):
        sample = dataset[index]
        pred = _predict(sample, errnet, rafa, device, alpha)
        err_m = compute_metrics(pred["errnet"], pred["target"])
        rafa_m = compute_metrics(pred["rafa"], pred["target"])
        fusion_m = compute_metrics(pred["fusion"], pred["target"])
        rows.append(
            {
                "dataset": dataset_name,
                "index": index,
                "name": str(sample["name"]),
                "ERRNet_PSNR": float(err_m["PSNR"]),
                "RAFA_PSNR": float(rafa_m["PSNR"]),
                "Fusion_PSNR": float(fusion_m["PSNR"]),
                "Fusion_minus_ERRNet": float(fusion_m["PSNR"]) - float(err_m["PSNR"]),
                "Fusion_minus_RAFA": float(fusion_m["PSNR"]) - float(rafa_m["PSNR"]),
                "RAFA_minus_ERRNet": float(rafa_m["PSNR"]) - float(err_m["PSNR"]),
            }
        )
    return rows


def _select_top(records: Iterable[Mapping[str, Any]], key: str, *, positive: bool = False) -> Optional[Mapping[str, Any]]:
    pool = [row for row in records if math.isfinite(float(row[key]))]
    if positive:
        pool = [row for row in pool if float(row[key]) > 0]
    if not pool:
        return None
    return max(pool, key=lambda row: float(row[key]))


def _select_median_positive(records: Iterable[Mapping[str, Any]], key: str) -> Optional[Mapping[str, Any]]:
    pool = sorted((row for row in records if float(row[key]) > 0), key=lambda row: float(row[key]))
    if not pool:
        pool = sorted(records, key=lambda row: float(row[key]))
    return pool[len(pool) // 2] if pool else None


def _select_top_beats_both(records: Iterable[Mapping[str, Any]]) -> Optional[Mapping[str, Any]]:
    pool = [
        row
        for row in records
        if float(row["Fusion_minus_ERRNet"]) > 0 and float(row["Fusion_minus_RAFA"]) > 0
    ]
    if not pool:
        return _select_top(records, "Fusion_minus_ERRNet", positive=True)
    return max(pool, key=lambda row: min(float(row["Fusion_minus_ERRNet"]), float(row["Fusion_minus_RAFA"])))


def _select_median_beats_both(records: Iterable[Mapping[str, Any]]) -> Optional[Mapping[str, Any]]:
    pool = sorted(
        (
            row
            for row in records
            if float(row["Fusion_minus_ERRNet"]) > 0 and float(row["Fusion_minus_RAFA"]) > 0
        ),
        key=lambda row: min(float(row["Fusion_minus_ERRNet"]), float(row["Fusion_minus_RAFA"])),
    )
    if not pool:
        return _select_median_positive(records, "Fusion_minus_ERRNet")
    return pool[len(pool) // 2]


def _select_failure(records: Iterable[Mapping[str, Any]]) -> Optional[Mapping[str, Any]]:
    pool = [row for row in records if math.isfinite(float(row["Fusion_minus_ERRNet"]))]
    if not pool:
        return None
    return min(pool, key=lambda row: float(row["Fusion_minus_ERRNet"]))


def _row_for_record(record: Mapping[str, Any], datasets: Mapping[str, Any], errnet, rafa, device, alpha: float) -> tuple[List[np.ndarray], str]:
    sample = datasets[str(record["dataset"])][int(record["index"])]
    pred = _predict(sample, errnet, rafa, device, alpha)
    row = [_to_rgb(pred[key]) for key in ("input", "errnet", "rafa", "fusion", "target")]
    title = (
        f"{record['dataset']}/{record['name']}  "
        f"Fusion-ERRNet={float(record['Fusion_minus_ERRNet']):+.2f} dB, "
        f"Fusion-RAFA={float(record['Fusion_minus_RAFA']):+.2f} dB"
    )
    return row, title


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


def main() -> None:
    from eval_all import choose_device, load_model

    args = parse_args()
    if args.alpha < 0 or args.alpha > 1:
        raise ValueError(f"--alpha must be in [0,1], got {args.alpha}")
    device = choose_device(args.device)
    data_root = Path(args.data_root)
    figures_dir = Path(args.figures_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    errnet = load_model("errnet", Path(args.errnet_ckpt), device, errnet_hyper=args.errnet_hyper)
    rafa = load_model("rap_errnet", Path(args.rafa_ckpt), device, config_path=args.rafa_config)

    datasets: Dict[str, Any] = {}
    scores: Dict[str, List[Dict[str, Any]]] = {}
    for name in [item.strip() for item in args.datasets.split(",") if item.strip()]:
        try:
            datasets[name] = _load_dataset(data_root, name, int(args.max_long_edge))
            scores[name] = _score_dataset(name, datasets[name], errnet, rafa, device, float(args.alpha))
        except FileNotFoundError as exc:
            warnings.warn(str(exc), RuntimeWarning)
            scores[name] = []

    all_scores = [row for rows in scores.values() for row in rows]
    _write_csv(out_dir / "fusion_selection_scores.csv", all_scores)

    rows: List[List[np.ndarray]] = []
    titles: List[str] = []
    selected: List[Mapping[str, Any]] = []
    selection_specs = [
        ("sir2_wild", "both", "top_both", "SIR2 Wild gain"),
        ("sir2_objects,sir2_postcard", "both", "median_both", "SIR2 median gain"),
        ("openrr_val", "both", "top_both", "OpenRR gain"),
    ]
    if scores.get("self"):
        selection_specs.append(("self", "both", "top_both", "Self-collected gain"))

    for dataset_group, key, mode, title_prefix in selection_specs:
        candidates: List[Mapping[str, Any]] = []
        for name in dataset_group.split(","):
            candidates.extend(scores.get(name, []))
        if mode == "top_both":
            record = _select_top_beats_both(candidates)
        elif mode == "median_both":
            record = _select_median_beats_both(candidates)
        else:
            record = _select_top(candidates, key, positive=True) if mode == "top" else _select_median_positive(candidates, key)
        if record is None:
            continue
        row, title = _row_for_record(record, datasets, errnet, rafa, device, float(args.alpha))
        rows.append(row)
        titles.append(f"{title_prefix}: {title}")
        selected.append(record)

    if rows:
        _save(
            _draw_grid(rows, ["Input", "ERRNet", "RAFA", f"Fusion a={args.alpha:.2f}", "GT"], titles, int(args.cell_width)),
            figures_dir,
            "fusion_qualitative_main",
        )

    protect_records = []
    for name in ("ceilnet", "zhang20"):
        record = _select_top(scores.get(name, []), "Fusion_minus_RAFA", positive=True)
        if record is not None:
            protect_records.append(record)
    protect_rows: List[List[np.ndarray]] = []
    protect_titles: List[str] = []
    for record in protect_records:
        row, title = _row_for_record(record, datasets, errnet, rafa, device, float(args.alpha))
        protect_rows.append(row)
        protect_titles.append(f"Protection: {title}")
        selected.append(record)
    if protect_rows:
        _save(
            _draw_grid(
                protect_rows,
                ["Input", "ERRNet", "RAFA", f"Fusion a={args.alpha:.2f}", "GT"],
                protect_titles,
                int(args.cell_width),
            ),
            figures_dir,
            "fusion_protection_cases",
        )

    fail_records = []
    for name in ("ceilnet", "zhang20"):
        record = _select_failure(scores.get(name, []))
        if record is not None:
            fail_records.append(record)
    fail_rows: List[List[np.ndarray]] = []
    fail_titles: List[str] = []
    for record in fail_records:
        row, title = _row_for_record(record, datasets, errnet, rafa, device, float(args.alpha))
        fail_rows.append(row)
        fail_titles.append(f"Failure: {title}")
        selected.append(record)
    if fail_rows:
        _save(
            _draw_grid(fail_rows, ["Input", "ERRNet", "RAFA", f"Fusion a={args.alpha:.2f}", "GT"], fail_titles, int(args.cell_width)),
            figures_dir,
            "fusion_failure_cases",
        )

    _write_csv(out_dir / "fusion_selected_examples.csv", selected)
    print(out_dir / "fusion_selection_scores.csv")
    print(out_dir / "fusion_selected_examples.csv")


if __name__ == "__main__":
    main()
