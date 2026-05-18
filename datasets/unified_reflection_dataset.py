"""Unified dataset adapter for reflection-removal experiments.

The adapter supports the directory layouts documented for RAP-ERRNet, but it
does not download or prepare any data. Missing datasets raise clear
``FileNotFoundError`` messages that include the expected paths.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from datasets.reflection_synthesis import synthesize_reflection_pair


IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".ppm"}


DATASET_PATHS = {
    "voc": Path("VOC2012"),
    "zhang_train": Path("zhang2018") / "train",
    "zhang20": Path("zhang2018") / "test",
    "ceilnet": Path("ceilnet") / "testdata_reflection_synthetic_table2",
    "sir2_objects": Path("sir2") / "Objects",
    "sir2_postcard": Path("sir2") / "Postcard",
    "sir2_wild": Path("sir2") / "Wild",
    "openrr_train": Path("openrr5k") / "train",
    "openrr_val": Path("openrr5k") / "val",
    "self": Path("self_collected") / "test",
}

INPUT_DIR_CANDIDATES = (
    "blended",
    "input",
    "inputs",
    "mixed",
    "reflection_blended",
    "reflection",
)
TARGET_DIR_CANDIDATES = (
    "transmission_layer",
    "transmission",
    "background",
    "target",
    "gt",
    "clean",
)
MASK_DIR_CANDIDATES = ("mask", "masks", "reflection_mask", "prior")


def _is_image(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMG_EXTENSIONS


def _list_images(path: Path) -> List[Path]:
    if not path.exists():
        return []
    return sorted(p for p in path.rglob("*") if _is_image(p))


def _find_first_existing_dir(base: Path, names: Sequence[str]) -> Optional[Path]:
    for name in names:
        path = base / name
        if path.is_dir():
            return path
    return None


def _read_list_file(path: Path) -> List[str]:
    with path.open("r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def _to_tensor(image: Image.Image) -> torch.Tensor:
    arr = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous()


def _to_mask_tensor(image: Image.Image) -> torch.Tensor:
    arr = np.asarray(image.convert("L"), dtype=np.float32) / 255.0
    return torch.from_numpy(arr).unsqueeze(0).contiguous()


def _parse_size(size) -> Optional[Tuple[int, int]]:
    if size is None:
        return None
    if isinstance(size, int):
        return (size, size)
    if isinstance(size, (tuple, list)) and len(size) == 2:
        return (int(size[0]), int(size[1]))
    raise ValueError(f"Invalid size value: {size!r}")


def _center_crop(image: Image.Image, crop_size: Optional[int]) -> Image.Image:
    if crop_size is None:
        return image
    w, h = image.size
    side = int(crop_size)
    if w < side or h < side:
        scale = side / float(min(w, h))
        new_size = (max(side, int(round(w * scale))), max(side, int(round(h * scale))))
        image = image.resize(new_size, Image.BICUBIC)
        w, h = image.size
    left = max((w - side) // 2, 0)
    top = max((h - side) // 2, 0)
    return image.crop((left, top, left + side, top + side))


def _load_rgb(path: Path, image_size=None, crop_size: Optional[int] = None) -> Image.Image:
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")
    image = Image.open(path).convert("RGB")
    size = _parse_size(image_size)
    if size is not None:
        image = image.resize(size, Image.BICUBIC)
    image = _center_crop(image, crop_size)
    return image


def _pseudo_mask(input_tensor: torch.Tensor, target_tensor: torch.Tensor) -> torch.Tensor:
    diff = (input_tensor - target_tensor).abs().max(dim=0, keepdim=True).values
    min_val = diff.amin(dim=(-2, -1), keepdim=True)
    max_val = diff.amax(dim=(-2, -1), keepdim=True)
    return ((diff - min_val) / (max_val - min_val).clamp_min(1e-6)).clamp(0.0, 1.0)


def _match_pairs(input_dir: Path, target_dir: Path, mask_dir: Optional[Path] = None) -> List[Tuple[Path, Path, Optional[Path]]]:
    inputs = _list_images(input_dir)
    targets_by_name = {p.name: p for p in _list_images(target_dir)}
    masks_by_name = {p.name: p for p in _list_images(mask_dir)} if mask_dir is not None else {}
    pairs = []
    for input_path in inputs:
        target_path = targets_by_name.get(input_path.name)
        if target_path is None:
            stem_matches = [p for p in targets_by_name.values() if p.stem == input_path.stem]
            target_path = stem_matches[0] if stem_matches else None
        if target_path is not None:
            mask_path = masks_by_name.get(input_path.name)
            pairs.append((input_path, target_path, mask_path))
    return pairs


def _discover_paired_dataset(base: Path) -> List[Tuple[Path, Path, Optional[Path]]]:
    input_dir = _find_first_existing_dir(base, INPUT_DIR_CANDIDATES)
    target_dir = _find_first_existing_dir(base, TARGET_DIR_CANDIDATES)
    mask_dir = _find_first_existing_dir(base, MASK_DIR_CANDIDATES)
    if input_dir is None or target_dir is None:
        expected = ", ".join(str(base / d) for d in INPUT_DIR_CANDIDATES[:3])
        expected_t = ", ".join(str(base / d) for d in TARGET_DIR_CANDIDATES[:3])
        raise FileNotFoundError(
            f"Could not find paired data under {base}. Expected an input directory like {expected} "
            f"and a target directory like {expected_t}."
        )
    pairs = _match_pairs(input_dir, target_dir, mask_dir)
    if not pairs:
        raise FileNotFoundError(f"No matched input/target image pairs found in {input_dir} and {target_dir}.")
    return pairs


def _discover_self_collected(base: Path) -> List[Tuple[Path, Path, Optional[Path]]]:
    if not base.exists():
        raise FileNotFoundError(f"Self-collected data directory not found: {base}")
    pairs = []
    for scene_dir in sorted(p for p in base.iterdir() if p.is_dir()):
        input_path = scene_dir / "blended.png"
        target_path = scene_dir / "transmission.png"
        mask_path = scene_dir / "mask.png"
        if input_path.exists() and target_path.exists():
            pairs.append((input_path, target_path, mask_path if mask_path.exists() else None))
    if not pairs:
        raise FileNotFoundError(
            f"No self-collected scenes found under {base}. Expected scene_xxx/blended.png and "
            "scene_xxx/transmission.png."
        )
    return pairs


def _discover_voc_images(data_root: Path) -> List[Path]:
    base = data_root / DATASET_PATHS["voc"]
    crop_dir = base / "cropped_224" / "train"
    if crop_dir.is_dir():
        paths = _list_images(crop_dir)
        if paths:
            return paths

    jpeg_dir = base / "JPEGImages"
    list_candidates = [base / "VOC2012_224_train_png.txt", data_root / "VOC2012_224_train_png.txt"]
    if jpeg_dir.is_dir():
        for list_file in list_candidates:
            if list_file.exists():
                names = _read_list_file(list_file)
                paths = []
                for name in names:
                    candidate = jpeg_dir / name
                    if candidate.suffix == "":
                        matches = [jpeg_dir / f"{name}{ext}" for ext in IMG_EXTENSIONS]
                        candidate = next((p for p in matches if p.exists()), candidate)
                    if candidate.exists():
                        paths.append(candidate)
                if paths:
                    return paths
        paths = _list_images(jpeg_dir)
        if paths:
            return paths

    raise FileNotFoundError(
        f"VOC images not found. Expected {crop_dir} or {jpeg_dir} with optional VOC2012_224_train_png.txt."
    )


def list_available_datasets(data_root: str | Path) -> Dict[str, bool]:
    """Return a quick availability map for known RAP-ERRNet datasets."""

    root = Path(data_root)
    availability: Dict[str, bool] = {}
    for name, rel in DATASET_PATHS.items():
        try:
            if name == "voc":
                availability[name] = bool(_discover_voc_images(root))
            elif name == "self":
                availability[name] = bool(_discover_self_collected(root / rel))
            else:
                availability[name] = bool(_discover_paired_dataset(root / rel))
        except FileNotFoundError:
            availability[name] = False
    return availability


class UnifiedReflectionDataset(Dataset):
    """Unified paired/synthetic reflection dataset.

    Returned samples follow the common format:
    ``input [3,H,W]``, ``target [3,H,W]``, ``mask [1,H,W]``, ``name`` and
    ``dataset``.
    """

    def __init__(
        self,
        data_root: str | Path,
        dataset: str,
        *,
        crop_size: Optional[int] = None,
        image_size=None,
        use_physics_synthesis: bool = False,
        max_pairs: Optional[int] = None,
        synthesis_config: Optional[dict] = None,
    ):
        self.data_root = Path(data_root)
        self.dataset = dataset
        self.crop_size = crop_size
        self.image_size = image_size
        self.use_physics_synthesis = use_physics_synthesis
        self.synthesis_config = synthesis_config

        if dataset not in DATASET_PATHS:
            raise ValueError(f"Unknown dataset '{dataset}'. Known datasets: {sorted(DATASET_PATHS)}")
        if not self.data_root.exists():
            raise FileNotFoundError(f"data_root does not exist: {self.data_root}")

        self.voc_images: List[Path] = []
        self.pairs: List[Tuple[Path, Path, Optional[Path]]] = []
        if dataset == "voc":
            self.voc_images = _discover_voc_images(self.data_root)
            if len(self.voc_images) < 2 and use_physics_synthesis:
                raise FileNotFoundError("VOC synthesis requires at least two source images.")
        elif dataset == "self":
            self.pairs = _discover_self_collected(self.data_root / DATASET_PATHS[dataset])
        else:
            self.pairs = _discover_paired_dataset(self.data_root / DATASET_PATHS[dataset])

        if max_pairs is not None:
            if self.voc_images:
                self.voc_images = self.voc_images[: int(max_pairs)]
            if self.pairs:
                self.pairs = self.pairs[: int(max_pairs)]

    def __len__(self) -> int:
        return len(self.voc_images) if self.dataset == "voc" else len(self.pairs)

    def _load_voc_sample(self, index: int) -> Dict[str, torch.Tensor | str]:
        t_path = self.voc_images[index % len(self.voc_images)]
        r_path = self.voc_images[(index + max(1, len(self.voc_images) // 2)) % len(self.voc_images)]
        t_img = _load_rgb(t_path, image_size=self.image_size, crop_size=self.crop_size)
        r_img = _load_rgb(r_path, image_size=self.image_size, crop_size=self.crop_size)
        if self.use_physics_synthesis:
            sample = synthesize_reflection_pair(t_img, r_img, seed=index, config=self.synthesis_config)
        else:
            target = _to_tensor(t_img)
            sample = {
                "input": target.clone(),
                "target": target,
                "reflection": torch.zeros_like(target),
                "mask": torch.zeros((1, target.shape[1], target.shape[2]), dtype=target.dtype),
            }
        sample["name"] = t_path.stem
        sample["dataset"] = self.dataset
        return sample

    def _load_pair_sample(self, index: int) -> Dict[str, torch.Tensor | str]:
        input_path, target_path, mask_path = self.pairs[index]
        input_img = _load_rgb(input_path, image_size=self.image_size, crop_size=self.crop_size)
        target_img = _load_rgb(target_path, image_size=self.image_size, crop_size=self.crop_size)
        input_tensor = _to_tensor(input_img)
        target_tensor = _to_tensor(target_img)
        if mask_path is not None and mask_path.exists():
            mask_img = Image.open(mask_path)
            size = _parse_size(self.image_size)
            if size is not None:
                mask_img = mask_img.resize(size, Image.BICUBIC)
            mask_img = _center_crop(mask_img, self.crop_size)
            mask = _to_mask_tensor(mask_img)
        else:
            warnings.warn(
                f"No mask found for {input_path.name}; using abs(input-target) pseudo mask.",
                RuntimeWarning,
            )
            mask = _pseudo_mask(input_tensor, target_tensor)
        return {
            "input": input_tensor,
            "target": target_tensor,
            "mask": mask,
            "name": input_path.stem,
            "dataset": self.dataset,
        }

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor | str]:
        if self.dataset == "voc":
            return self._load_voc_sample(index)
        return self._load_pair_sample(index)

