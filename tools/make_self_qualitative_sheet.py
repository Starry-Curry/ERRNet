"""Create all-scene qualitative sheets for the self-collected set.

The script evaluates ERRNet, RAFA, and their constant-alpha fusion on every
scene in ``data/self_collected/test``. It saves one contact sheet, one image per
scene, raw column images, and a per-scene metrics CSV.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

import numpy as np
from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build self-collected qualitative comparison sheets.")
    parser.add_argument("--data_root", default="./data")
    parser.add_argument("--out_dir", default="results/self_collected_qualitative")
    parser.add_argument("--figures_dir", default="paper/figures")
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


def _align_tensors(*tensors):
    heights = [int(tensor.shape[-2]) for tensor in tensors if tensor is not None]
    widths = [int(tensor.shape[-1]) for tensor in tensors if tensor is not None]
    size = (min(heights), min(widths))
    return tuple(_center_crop_tensor(tensor, size) if tensor is not None else None for tensor in tensors)


def _to_rgb(tensor) -> np.ndarray:
    arr = tensor.detach().float().cpu().clamp(0.0, 1.0)
    if arr.ndim == 4:
        arr = arr[0]
    if arr.shape[0] == 1:
        arr = arr.repeat(3, 1, 1)
    return (arr.permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)


def _fit_cell(arr: np.ndarray, width: int) -> Image.Image:
    image = Image.fromarray(arr)
    if image.width == width:
        return image
    height = max(1, int(round(image.height * width / float(image.width))))
    return image.resize((width, height), Image.Resampling.BICUBIC)


def _draw_grid(rows: Sequence[Sequence[np.ndarray]], labels: Sequence[str], titles: Sequence[str], cell_width: int) -> Image.Image:
    fitted_rows: List[List[Image.Image]] = [[_fit_cell(item, cell_width) for item in row] for row in rows]
    title_h = 32
    label_h = 28
    row_heights = [max(cell.height for cell in row) for row in fitted_rows]
    canvas = Image.new("RGB", (cell_width * len(labels), sum(title_h + label_h + h for h in row_heights)), "white")
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
        y += row_heights[row_index]
    return canvas


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


def _scene_name(dataset, index: int) -> str:
    pairs = getattr(dataset, "pairs", [])
    if index < len(pairs):
        input_path = Path(pairs[index][0])
        parent = input_path.parent.name
        if parent:
            return parent
    return f"scene_{index + 1:03d}"


def _metrics_with_prefix(prefix: str, metrics: Mapping[str, float]) -> Dict[str, float]:
    return {f"{prefix}_{key}": float(value) for key, value in metrics.items()}


def _save_image(path: Path, arr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr).save(path)


def _write_markdown(path: Path, rows: Sequence[Mapping[str, Any]], alpha: float) -> None:
    lines = [
        "# Self-Collected Qualitative Review",
        "",
        f"Fusion alpha: {alpha:.2f}",
        "",
        "| Scene | Best PSNR | ERRNet | RAFA | Fusion | Fusion-ERRNet | Fusion-RAFA | Reading |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        err = float(row["ERRNet_PSNR"])
        rafa = float(row["RAFA_PSNR"])
        fusion = float(row["Fusion_PSNR"])
        best = max((err, "ERRNet"), (rafa, "RAFA"), (fusion, "Fusion"))[1]
        d_err = fusion - err
        d_rafa = fusion - rafa
        if best == "Fusion":
            reading = "fusion gives the best compromise on this pair"
        elif best == "ERRNet":
            reading = "baseline is still strongest; inspect for over-removal or alignment effects"
        else:
            reading = "RAFA is strongest; fusion is conservative"
        lines.append(
            f"| {row['scene']} | {best} | {err:.4f} | {rafa:.4f} | {fusion:.4f} | "
            f"{d_err:+.4f} | {d_rafa:+.4f} | {reading} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    import torch

    from datasets.unified_reflection_dataset import UnifiedReflectionDataset
    from eval_all import choose_device, load_model
    from metrics.reflection_metrics import compute_metrics

    args = parse_args()
    if args.alpha < 0 or args.alpha > 1:
        raise ValueError(f"--alpha must be in [0,1], got {args.alpha}")

    device = choose_device(args.device)
    dataset = UnifiedReflectionDataset(Path(args.data_root), "self", crop_size=None, image_size=None, max_long_edge=None)
    errnet = load_model("errnet", Path(args.errnet_ckpt), device, errnet_hyper=args.errnet_hyper)
    rafa = load_model("rap_errnet", Path(args.rafa_ckpt), device, config_path=args.rafa_config)

    out_dir = Path(args.out_dir)
    figures_dir = Path(args.figures_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "scenes").mkdir(parents=True, exist_ok=True)
    (out_dir / "columns").mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    labels = ["Input", "ERRNet", "RAFA", f"Fusion a={args.alpha:.2f}", "GT"]
    rows_for_sheet: List[List[np.ndarray]] = []
    row_titles: List[str] = []
    metric_rows: List[Dict[str, Any]] = []

    for index in range(len(dataset)):
        sample = dataset[index]
        scene = _scene_name(dataset, index)
        input_tensor = sample["input"].unsqueeze(0).to(device)
        target_tensor = sample["target"].unsqueeze(0).to(device)
        with torch.no_grad():
            err = errnet(input_tensor)["output"].clamp(0.0, 1.0)
            rafa_out = rafa(input_tensor)["output"].clamp(0.0, 1.0)
        input_aligned, target_aligned, err, rafa_out = _align_tensors(input_tensor, target_tensor, err, rafa_out)
        fusion = (float(args.alpha) * rafa_out + (1.0 - float(args.alpha)) * err).clamp(0.0, 1.0)

        err_metrics = compute_metrics(err[0], target_aligned[0])
        rafa_metrics = compute_metrics(rafa_out[0], target_aligned[0])
        fusion_metrics = compute_metrics(fusion[0], target_aligned[0])
        metric_row: Dict[str, Any] = {
            "scene": scene,
            "index": index,
            **_metrics_with_prefix("ERRNet", err_metrics),
            **_metrics_with_prefix("RAFA", rafa_metrics),
            **_metrics_with_prefix("Fusion", fusion_metrics),
            "Fusion_minus_ERRNet_PSNR": float(fusion_metrics["PSNR"]) - float(err_metrics["PSNR"]),
            "Fusion_minus_RAFA_PSNR": float(fusion_metrics["PSNR"]) - float(rafa_metrics["PSNR"]),
            "RAFA_minus_ERRNet_PSNR": float(rafa_metrics["PSNR"]) - float(err_metrics["PSNR"]),
        }
        metric_rows.append(metric_row)

        arrays = {
            "input": _to_rgb(input_aligned[0]),
            "errnet": _to_rgb(err[0]),
            "rafa": _to_rgb(rafa_out[0]),
            "fusion": _to_rgb(fusion[0]),
            "gt": _to_rgb(target_aligned[0]),
        }
        for key, arr in arrays.items():
            _save_image(out_dir / "columns" / scene / f"{key}.png", arr)

        row = [arrays[key] for key in ("input", "errnet", "rafa", "fusion", "gt")]
        rows_for_sheet.append(row)
        row_titles.append(
            f"{scene}: ERRNet {err_metrics['PSNR']:.2f} dB, RAFA {rafa_metrics['PSNR']:.2f} dB, "
            f"Fusion {fusion_metrics['PSNR']:.2f} dB "
            f"(F-E {metric_row['Fusion_minus_ERRNet_PSNR']:+.2f}, F-R {metric_row['Fusion_minus_RAFA_PSNR']:+.2f})"
        )
        scene_sheet = _draw_grid([row], labels, [row_titles[-1]], int(args.cell_width))
        scene_sheet.save(out_dir / "scenes" / f"{scene}_all_methods.png")

    contact = _draw_grid(rows_for_sheet, labels, row_titles, int(args.cell_width))
    contact_png = out_dir / "self_collected_all_methods.png"
    contact_pdf = out_dir / "self_collected_all_methods.pdf"
    contact.save(contact_png)
    contact.save(contact_pdf)
    contact.save(figures_dir / "self_collected_all_methods.png")
    contact.save(figures_dir / "self_collected_all_methods.pdf")

    _write_csv(out_dir / "self_collected_per_scene_metrics.csv", metric_rows)
    _write_markdown(out_dir / "SELF_COLLECTED_QUALITATIVE_REVIEW.md", metric_rows, float(args.alpha))
    print(contact_png)
    print(out_dir / "self_collected_per_scene_metrics.csv")
    print(out_dir / "SELF_COLLECTED_QUALITATIVE_REVIEW.md")


if __name__ == "__main__":
    main()
