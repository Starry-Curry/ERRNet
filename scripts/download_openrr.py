"""Download and arrange OpenRR-5k for RAP/BP-RAP experiments.

The script downloads files from the public Hugging Face dataset page and
rearranges them into the layout expected by ``UnifiedReflectionDataset``:

data/openrr5k/
  train/blended/
  train/transmission/
  val/blended/
  val/transmission/
  test/blended/

It never runs during training; call it explicitly on the server.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path
from typing import Iterable, Optional


REPO_ID = "qiuzhangTiTi/OpenRR-5k"
HF_BASE_URL = "https://huggingface.co/datasets/{repo}/resolve/main/{filename}"
FILES = {
    # The upstream Hugging Face repo currently uses the misspelled archive name
    # ``trian_5k.zip`` for the 5k training split.
    "train": ["trian_5k.zip"],
    "val": ["val_300_blended.zip", "val_300_transmission.zip"],
    "test": ["test_100_blended.zip"],
}
IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
BLENDED_HINTS = ("blended", "blend", "input", "inputs", "mixed", "reflection")
TARGET_HINTS = ("transmission", "gt", "ground", "target", "clean", "background")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download and arrange OpenRR-5k from Hugging Face.")
    parser.add_argument("--data_root", default="./data", help="project data root")
    parser.add_argument("--cache_dir", default=None, help="download/extract cache; default: data_root/openrr5k_raw")
    parser.add_argument("--repo", default=REPO_ID, help="Hugging Face dataset repo id")
    parser.add_argument("--splits", default="val", help="comma-separated train,val,test,all")
    parser.add_argument("--token", default=None, help="optional Hugging Face token; env HF_TOKEN is also used")
    parser.add_argument("--force", action="store_true", help="redownload and re-extract existing files")
    parser.add_argument("--no_extract", action="store_true", help="only download zip files")
    parser.add_argument("--copy", action="store_true", help="copy files instead of symlinking from extracted cache")
    return parser.parse_args()


def _selected_splits(value: str) -> list[str]:
    parts = [item.strip().lower() for item in value.split(",") if item.strip()]
    if "all" in parts:
        return ["train", "val", "test"]
    unknown = sorted(set(parts) - set(FILES))
    if unknown:
        raise ValueError(f"Unknown split(s): {unknown}. Expected train,val,test,all.")
    return parts or ["val"]


def _download_with_hf_hub(repo: str, filename: str, out_path: Path, token: Optional[str]) -> bool:
    try:
        from huggingface_hub import hf_hub_download
    except Exception:
        return False
    downloaded = hf_hub_download(
        repo_id=repo,
        repo_type="dataset",
        filename=filename,
        token=token,
        local_dir=str(out_path.parent),
        local_dir_use_symlinks=False,
    )
    downloaded_path = Path(downloaded)
    if downloaded_path.resolve() != out_path.resolve():
        out_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(downloaded_path, out_path)
    return True


def _download_with_urllib(repo: str, filename: str, out_path: Path, token: Optional[str]) -> None:
    url = HF_BASE_URL.format(repo=repo, filename=filename)
    request = urllib.request.Request(url)
    if token:
        request.add_header("Authorization", f"Bearer {token}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(request) as response, out_path.open("wb") as f:
        shutil.copyfileobj(response, f)


def download_file(repo: str, filename: str, cache_dir: Path, token: Optional[str], force: bool) -> Path:
    out_path = cache_dir / filename
    if out_path.exists() and out_path.stat().st_size > 0 and not force:
        print(f"[i] exists: {out_path}")
        return out_path
    print(f"[i] downloading {repo}/{filename}")
    if not _download_with_hf_hub(repo, filename, out_path, token):
        _download_with_urllib(repo, filename, out_path, token)
    print(f"[i] saved: {out_path} ({out_path.stat().st_size / (1024 ** 3):.2f} GiB)")
    return out_path


def extract_zip(zip_path: Path, extract_dir: Path, force: bool) -> Path:
    marker = extract_dir / ".extracted"
    if marker.exists() and not force:
        print(f"[i] extracted: {extract_dir}")
        return extract_dir
    if extract_dir.exists() and force:
        shutil.rmtree(extract_dir)
    extract_dir.mkdir(parents=True, exist_ok=True)
    print(f"[i] extracting {zip_path.name} -> {extract_dir}")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(extract_dir)
    marker.write_text(zip_path.name, encoding="utf-8")
    return extract_dir


def _is_image(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMG_EXTENSIONS and not path.name.startswith("._")


def _iter_images(path: Path) -> list[Path]:
    return sorted(p for p in path.rglob("*") if _is_image(p))


def _score_path(path: Path, hints: Iterable[str]) -> int:
    text = str(path).lower()
    return sum(1 for hint in hints if hint in text)


def _find_best_dir(root: Path, hints: Iterable[str]) -> Optional[Path]:
    candidates = []
    for directory in [root, *[p for p in root.rglob("*") if p.is_dir()]]:
        images = _iter_images(directory)
        if not images:
            continue
        score = _score_path(directory, hints)
        if score > 0:
            candidates.append((score, len(images), directory))
    if not candidates:
        return None
    candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return candidates[0][2]


def _clear_dir(path: Path) -> None:
    if path.exists():
        for item in path.iterdir():
            if item.is_dir() and not item.is_symlink():
                shutil.rmtree(item)
            else:
                item.unlink()
    path.mkdir(parents=True, exist_ok=True)


def _link_or_copy(src: Path, dst: Path, copy: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if copy:
        shutil.copy2(src, dst)
    else:
        try:
            os.symlink(src.resolve(), dst)
        except OSError:
            shutil.copy2(src, dst)


def _arrange_images(src_dir: Path, dst_dir: Path, copy: bool) -> int:
    images = _iter_images(src_dir)
    _clear_dir(dst_dir)
    for src in images:
        _link_or_copy(src, dst_dir / src.name, copy=copy)
    print(f"[i] arranged {len(images)} images: {dst_dir}")
    return len(images)


def arrange_split(split: str, extracted: dict[str, Path], data_root: Path, copy: bool) -> None:
    openrr_root = data_root / "openrr5k"
    if split == "val":
        _arrange_images(extracted["val_300_blended.zip"], openrr_root / "val" / "blended", copy)
        _arrange_images(extracted["val_300_transmission.zip"], openrr_root / "val" / "transmission", copy)
        return
    if split == "test":
        _arrange_images(extracted["test_100_blended.zip"], openrr_root / "test" / "blended", copy)
        return
    if split == "train":
        train_root = extracted.get("trian_5k.zip") or extracted.get("train_5000.zip")
        if train_root is None:
            raise FileNotFoundError("OpenRR train archive was not extracted.")
        blended_dir = _find_best_dir(train_root, BLENDED_HINTS)
        target_dir = _find_best_dir(train_root, TARGET_HINTS)
        if blended_dir is None or target_dir is None:
            raise FileNotFoundError(
                "Could not auto-detect train blended/transmission directories inside the OpenRR train archive. "
                f"Inspect extracted files under {train_root} and arrange them manually."
            )
        print(f"[i] train blended source: {blended_dir}")
        print(f"[i] train transmission source: {target_dir}")
        _arrange_images(blended_dir, openrr_root / "train" / "blended", copy)
        _arrange_images(target_dir, openrr_root / "train" / "transmission", copy)


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root)
    cache_dir = Path(args.cache_dir) if args.cache_dir else data_root / "openrr5k_raw"
    token = args.token or os.environ.get("HF_TOKEN")
    splits = _selected_splits(args.splits)
    cache_dir.mkdir(parents=True, exist_ok=True)

    downloaded: dict[str, Path] = {}
    extracted: dict[str, Path] = {}
    for split in splits:
        for filename in FILES[split]:
            zip_path = download_file(args.repo, filename, cache_dir, token, args.force)
            downloaded[filename] = zip_path
            if not args.no_extract:
                extracted[filename] = extract_zip(zip_path, cache_dir / filename.replace(".zip", ""), args.force)

    if args.no_extract:
        print("[i] download-only mode completed.")
        return

    for split in splits:
        arrange_split(split, extracted, data_root, args.copy)

    print("[i] done. Check availability with:")
    print("python - <<'PY'")
    print("from datasets.unified_reflection_dataset import list_available_datasets")
    print(f"print(list_available_datasets({str(data_root)!r}))")
    print("PY")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
