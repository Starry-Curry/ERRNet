"""Prepare 224x224 VOC crops for RAP-ERRNet synthesis.

This script assumes VOC has already been downloaded and extracted. It does not
download any data.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image


IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Center-crop VOC images.")
    parser.add_argument("--input_dir", required=True, help="VOC2012/JPEGImages directory")
    parser.add_argument("--output_dir", required=True, help="output directory for cropped PNG files")
    parser.add_argument("--list_file", default=None, help="optional VOC image-id or filename list")
    parser.add_argument("--crop_size", type=int, default=224)
    return parser.parse_args()


def read_names(list_file: Path | None, input_dir: Path):
    if list_file is None:
        return sorted(path.name for path in input_dir.iterdir() if path.suffix.lower() in IMG_EXTENSIONS)
    with list_file.open("r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def resolve_image(input_dir: Path, name: str) -> Path | None:
    candidate = input_dir / name
    if candidate.exists():
        return candidate
    stem = Path(name).stem
    for ext in IMG_EXTENSIONS:
        candidate = input_dir / f"{stem}{ext}"
        if candidate.exists():
            return candidate
    return None


def center_crop(image: Image.Image, crop_size: int) -> Image.Image:
    w, h = image.size
    if w < crop_size or h < crop_size:
        scale = crop_size / float(min(w, h))
        image = image.resize((int(round(w * scale)), int(round(h * scale))), Image.BICUBIC)
        w, h = image.size
    left = (w - crop_size) // 2
    top = (h - crop_size) // 2
    return image.crop((left, top, left + crop_size, top + crop_size))


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    list_file = Path(args.list_file) if args.list_file else None
    if not input_dir.is_dir():
        raise FileNotFoundError(f"input_dir not found: {input_dir}")
    if list_file is not None and not list_file.exists():
        raise FileNotFoundError(f"list_file not found: {list_file}")
    output_dir.mkdir(parents=True, exist_ok=True)

    count = 0
    for name in read_names(list_file, input_dir):
        src = resolve_image(input_dir, name)
        if src is None:
            print(f"[warn] missing image for {name}")
            continue
        image = Image.open(src).convert("RGB")
        crop = center_crop(image, args.crop_size)
        crop.save(output_dir / f"{src.stem}.png")
        count += 1
    print(f"wrote {count} crops to {output_dir}")


if __name__ == "__main__":
    main()

