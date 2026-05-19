"""Build contact-sheet visualizations from eval_all.py outputs."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create contact sheets from eval_all.py visualizations.")
    parser.add_argument("--input_dir", default="results/rap_errnet_eval", help="eval_all.py save_dir")
    parser.add_argument("--output_dir", default=None, help="directory for generated contact sheets")
    parser.add_argument("--max_images", type=int, default=20)
    parser.add_argument("--thumb_width", type=int, default=1200)
    return parser.parse_args()


def _find_visuals(input_dir: Path) -> List[Path]:
    vis_root = input_dir / "visualizations"
    if not vis_root.exists():
        return []
    return sorted(p for p in vis_root.rglob("*.png") if p.is_file())


def _resize_width(image: Image.Image, width: int) -> Image.Image:
    if image.width == width:
        return image
    height = max(1, int(round(image.height * width / float(image.width))))
    try:
        resample = Image.Resampling.BICUBIC
    except AttributeError:  # Pillow < 9
        resample = Image.BICUBIC
    return image.resize((width, height), resample)


def build_contact_sheet(images: List[Path], output_path: Path, thumb_width: int) -> None:
    loaded = [_resize_width(Image.open(path).convert("RGB"), thumb_width) for path in images]
    if not loaded:
        raise FileNotFoundError("No visualization images were found.")
    total_height = sum(image.height for image in loaded)
    canvas = Image.new("RGB", (thumb_width, total_height), "white")
    y = 0
    for image in loaded:
        canvas.paste(image, (0, y))
        y += image.height
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir) if args.output_dir else input_dir / "contact_sheets"
    visuals = _find_visuals(input_dir)[: args.max_images]
    if not visuals:
        raise FileNotFoundError(f"No eval visualizations found under {input_dir / 'visualizations'}.")
    build_contact_sheet(visuals, output_dir / "overview.png", args.thumb_width)
    by_dataset = {}
    for path in visuals:
        dataset = path.parent.name
        by_dataset.setdefault(dataset, []).append(path)
    for dataset, paths in by_dataset.items():
        build_contact_sheet(paths, output_dir / f"{dataset}.png", args.thumb_width)


if __name__ == "__main__":
    main()
