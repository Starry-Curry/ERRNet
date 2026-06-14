"""Build a two-example cover image for the project presentation.

The cover image is intentionally simpler than the paper qualitative figures:
two rows, each with Input | Fusion | GT.  The default selector uses one OpenRR
success case and one SIR2 Objects/Postcard success case so the cover shows
real-domain gains without looking like an experiment table.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import numpy as np
from PIL import Image, ImageDraw


REPO_ROOT = Path(__file__).resolve().parents[1]
TOOLS_DIR = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from make_fusion_qualitative_figures import (  # noqa: E402
    _load_dataset,
    _predict,
    _score_dataset,
    _select_median_beats_both,
    _select_top_beats_both,
    _to_rgb,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create PPT cover two-case qualitative image.")
    parser.add_argument("--data_root", default="./data")
    parser.add_argument("--out_dir", default="results/ppt_cover_cases")
    parser.add_argument("--figures_dir", default="PPT/assets")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--errnet_ckpt", default="checkpoints/errnet/errnet_060_00463920.pt")
    parser.add_argument("--errnet_hyper", action="store_true")
    parser.add_argument("--rafa_ckpt", default="checkpoints/bp_rap_rafa_openrr3k_no_old_e10/best.pt")
    parser.add_argument("--rafa_config", default="configs/bp_rap_rafa_openrr3k_no_old.yaml")
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--max_long_edge", type=int, default=512)
    parser.add_argument("--cell_width", type=int, default=320)
    parser.add_argument("--first_group", default="openrr_val")
    parser.add_argument("--second_group", default="sir2_objects,sir2_postcard")
    parser.add_argument("--first_mode", default="top_both", choices=["top_both", "median_both"])
    parser.add_argument("--second_mode", default="median_both", choices=["top_both", "median_both"])
    parser.add_argument(
        "--manual",
        default="",
        help="Optional comma-separated selections like openrr_val:207,sir2_objects:9. "
        "Index is matched first, then exact sample name.",
    )
    return parser.parse_args()


def _fit_cell(arr: np.ndarray, width: int, height: int) -> Image.Image:
    image = Image.fromarray(arr)
    scale = max(width / float(image.width), height / float(image.height))
    new_size = (max(1, int(round(image.width * scale))), max(1, int(round(image.height * scale))))
    image = image.resize(new_size, Image.Resampling.BICUBIC)
    left = max((image.width - width) // 2, 0)
    top = max((image.height - height) // 2, 0)
    return image.crop((left, top, left + width, top + height))


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


def _select_from_group(records: Iterable[Mapping[str, Any]], mode: str) -> Optional[Mapping[str, Any]]:
    if mode == "top_both":
        return _select_top_beats_both(records)
    if mode == "median_both":
        return _select_median_beats_both(records)
    raise ValueError(f"unknown mode: {mode}")


def _group_records(group: str, scores: Mapping[str, Sequence[Mapping[str, Any]]]) -> List[Mapping[str, Any]]:
    rows: List[Mapping[str, Any]] = []
    for name in [item.strip() for item in group.split(",") if item.strip()]:
        rows.extend(scores.get(name, []))
    return rows


def _manual_records(manual: str, scores: Mapping[str, Sequence[Mapping[str, Any]]]) -> List[Mapping[str, Any]]:
    selected: List[Mapping[str, Any]] = []
    for item in [part.strip() for part in manual.split(",") if part.strip()]:
        if ":" not in item:
            raise ValueError(f"manual selection must be dataset:index_or_name, got {item!r}")
        dataset, key = item.split(":", 1)
        dataset = dataset.strip()
        key = key.strip()
        pool = list(scores.get(dataset, []))
        record = None
        if key.isdigit():
            idx = int(key)
            record = next((row for row in pool if int(row["index"]) == idx), None)
        if record is None:
            record = next((row for row in pool if str(row["name"]) == key), None)
        if record is None:
            available = ", ".join(str(row["name"]) for row in pool[:8])
            raise ValueError(f"could not find {item!r}; first available names: {available}")
        selected.append(record)
    return selected


def _row_arrays(record: Mapping[str, Any], datasets: Mapping[str, Any], errnet, rafa, device, alpha: float) -> List[np.ndarray]:
    sample = datasets[str(record["dataset"])][int(record["index"])]
    pred = _predict(sample, errnet, rafa, device, alpha)
    return [_to_rgb(pred[key]) for key in ("input", "fusion", "target")]


def _row_title(record: Mapping[str, Any], label: str) -> str:
    d_err = float(record["Fusion_minus_ERRNet"])
    d_rafa = float(record["Fusion_minus_RAFA"])
    return f"{label}: {record['dataset']} / {record['name']}   Fusion-ERRNet {d_err:+.2f} dB, Fusion-RAFA {d_rafa:+.2f} dB"


def _draw_cover(rows: Sequence[Sequence[np.ndarray]], titles: Sequence[str], alpha: float, cell_width: int) -> Image.Image:
    labels = ["Input", f"Fusion a={alpha:.2f}", "GT"]
    cell_height = int(round(cell_width * 0.72))
    title_h = 30
    label_h = 24
    gap = 14
    pad = 12
    width = pad * 2 + cell_width * len(labels)
    height = pad * 2 + len(rows) * (title_h + label_h + cell_height) + (len(rows) - 1) * gap
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)

    y = pad
    for row, title in zip(rows, titles):
        draw.text((pad, y + 7), title, fill=(32, 48, 64))
        y += title_h
        for col, label in enumerate(labels):
            x = pad + col * cell_width
            draw.text((x + 8, y + 5), label, fill=(32, 48, 64))
        y += label_h
        for col, arr in enumerate(row):
            x = pad + col * cell_width
            image = _fit_cell(arr, cell_width, cell_height)
            canvas.paste(image, (x, y))
        y += cell_height + gap
    return canvas


def main() -> None:
    from eval_all import choose_device, load_model

    args = parse_args()
    if not 0.0 <= float(args.alpha) <= 1.0:
        raise ValueError(f"--alpha must be in [0,1], got {args.alpha}")

    device = choose_device(args.device)
    errnet = load_model("errnet", Path(args.errnet_ckpt), device, errnet_hyper=args.errnet_hyper)
    rafa = load_model("rap_errnet", Path(args.rafa_ckpt), device, config_path=args.rafa_config)

    dataset_names = {
        item.strip()
        for group in (args.first_group, args.second_group)
        for item in group.split(",")
        if item.strip()
    }
    if args.manual:
        dataset_names.update(item.split(":", 1)[0].strip() for item in args.manual.split(",") if item.strip())

    datasets: Dict[str, Any] = {}
    scores: Dict[str, List[Dict[str, Any]]] = {}
    for name in sorted(dataset_names):
        datasets[name] = _load_dataset(Path(args.data_root), name, int(args.max_long_edge))
        scores[name] = _score_dataset(name, datasets[name], errnet, rafa, device, float(args.alpha))

    if args.manual:
        selected = _manual_records(args.manual, scores)
        titles = [_row_title(record, f"Case {idx + 1}") for idx, record in enumerate(selected)]
    else:
        first = _select_from_group(_group_records(args.first_group, scores), args.first_mode)
        second = _select_from_group(_group_records(args.second_group, scores), args.second_mode)
        selected = [record for record in (first, second) if record is not None]
        titles = []
        if first is not None:
            titles.append(_row_title(first, "OpenRR real-pair success"))
        if second is not None:
            titles.append(_row_title(second, "SIR2 real-scene success"))

    if len(selected) != 2:
        raise RuntimeError(f"expected exactly 2 selected cases, got {len(selected)}")
    if any(not math.isfinite(float(row["Fusion_minus_ERRNet"])) for row in selected):
        raise RuntimeError("selected records contain non-finite metrics")

    rows = [_row_arrays(record, datasets, errnet, rafa, device, float(args.alpha)) for record in selected]
    image = _draw_cover(rows, titles, float(args.alpha), int(args.cell_width))

    out_dir = Path(args.out_dir)
    figures_dir = Path(args.figures_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    image.save(out_dir / "cover_two_cases.png")
    image.save(out_dir / "cover_two_cases.pdf")
    image.save(figures_dir / "cover_two_cases.png")
    image.save(figures_dir / "cover_two_cases.pdf")
    _write_csv(out_dir / "cover_two_cases_selected.csv", selected)
    print(out_dir / "cover_two_cases.png")
    print(out_dir / "cover_two_cases_selected.csv")
    print(figures_dir / "cover_two_cases.png")


if __name__ == "__main__":
    main()
