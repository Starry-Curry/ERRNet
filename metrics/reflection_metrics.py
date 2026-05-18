"""Image quality metrics used by RAP-ERRNet evaluation.

The functions accept PyTorch tensors or NumPy arrays in RGB range [0, 1].
They crop mismatched inputs to their common center region and use data_range=1.0.
"""

from __future__ import annotations

import csv
import math
import warnings
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, MutableMapping, Tuple

import numpy as np

try:
    import torch
except Exception:  # pragma: no cover - metrics can still operate on numpy arrays
    torch = None

try:
    from skimage.metrics import structural_similarity as skimage_ssim
except Exception:  # pragma: no cover
    skimage_ssim = None


MetricDict = Dict[str, float]


def _to_numpy_rgb(image) -> np.ndarray:
    """Convert tensor/array image to HWC float32 RGB in [0, 1]."""

    if torch is not None and isinstance(image, torch.Tensor):
        image = image.detach().cpu().float().numpy()
    arr = np.asarray(image, dtype=np.float32)

    if arr.ndim == 4:
        if arr.shape[0] != 1:
            raise ValueError(f"Expected a single image or batch size 1, got shape {arr.shape}.")
        arr = arr[0]

    if arr.ndim == 2:
        arr = arr[..., None]
    elif arr.ndim == 3 and arr.shape[0] in (1, 3) and arr.shape[-1] not in (1, 3):
        arr = np.transpose(arr, (1, 2, 0))

    if arr.ndim != 3:
        raise ValueError(f"Expected image with 2 or 3 dimensions, got shape {arr.shape}.")

    if arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)
    if arr.shape[-1] != 3:
        raise ValueError(f"Expected RGB image with 3 channels, got shape {arr.shape}.")

    return np.clip(arr, 0.0, 1.0)


def _center_crop_common(pred: np.ndarray, target: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    h = min(pred.shape[0], target.shape[0])
    w = min(pred.shape[1], target.shape[1])
    if pred.shape[:2] != target.shape[:2]:
        warnings.warn(
            f"Metric inputs have different sizes {pred.shape[:2]} and {target.shape[:2]}; "
            f"center-cropping both to {(h, w)}.",
            RuntimeWarning,
        )

    def crop(x: np.ndarray) -> np.ndarray:
        top = max((x.shape[0] - h) // 2, 0)
        left = max((x.shape[1] - w) // 2, 0)
        return x[top : top + h, left : left + w]

    return crop(pred), crop(target)


def prepare_pair(pred, target) -> Tuple[np.ndarray, np.ndarray]:
    """Return center-aligned RGB arrays in [0, 1]."""

    pred_np = _to_numpy_rgb(pred)
    target_np = _to_numpy_rgb(target)
    return _center_crop_common(pred_np, target_np)


def psnr(pred, target, data_range: float = 1.0) -> float:
    pred_np, target_np = prepare_pair(pred, target)
    mse = float(np.mean((pred_np - target_np) ** 2))
    if mse <= 1e-12:
        return float("inf")
    return 10.0 * math.log10((data_range * data_range) / mse)


def ssim(pred, target, data_range: float = 1.0) -> float:
    pred_np, target_np = prepare_pair(pred, target)
    if skimage_ssim is not None:
        min_side = min(pred_np.shape[0], pred_np.shape[1])
        win_size = min(7, min_side if min_side % 2 == 1 else min_side - 1)
        if win_size < 3:
            return 1.0 if np.allclose(pred_np, target_np) else 0.0
        try:
            return float(
                skimage_ssim(
                    target_np,
                    pred_np,
                    data_range=data_range,
                    channel_axis=-1,
                    win_size=win_size,
                )
            )
        except TypeError:
            return float(
                skimage_ssim(
                    target_np,
                    pred_np,
                    data_range=data_range,
                    multichannel=True,
                    win_size=win_size,
                )
            )

    # Lightweight fallback: global SSIM approximation over all RGB pixels.
    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    x = pred_np.reshape(-1, 3)
    y = target_np.reshape(-1, 3)
    mux, muy = x.mean(axis=0), y.mean(axis=0)
    vx, vy = x.var(axis=0), y.var(axis=0)
    cov = ((x - mux) * (y - muy)).mean(axis=0)
    score = ((2 * mux * muy + c1) * (2 * cov + c2)) / ((mux**2 + muy**2 + c1) * (vx + vy + c2))
    return float(np.mean(score))


def ncc(pred, target) -> float:
    pred_np, target_np = prepare_pair(pred, target)
    x = pred_np.reshape(-1).astype(np.float64)
    y = target_np.reshape(-1).astype(np.float64)
    x = x - x.mean()
    y = y - y.mean()
    denom = math.sqrt(float(np.sum(x * x) * np.sum(y * y)))
    if denom <= 1e-12:
        return 1.0 if np.allclose(pred_np, target_np) else 0.0
    return float(np.sum(x * y) / denom)


def _scaled_ssq_error(correct: np.ndarray, estimate: np.ndarray) -> float:
    denom = float(np.sum(estimate**2))
    alpha = float(np.sum(correct * estimate) / denom) if denom > 1e-12 else 0.0
    return float(np.sum((correct - alpha * estimate) ** 2))


def lmse(pred, target, window_size: int = 20, stride: int = 10) -> float:
    pred_np, target_np = prepare_pair(pred, target)
    h, w, c = target_np.shape
    win_h = min(window_size, h)
    win_w = min(window_size, w)
    if win_h <= 0 or win_w <= 0:
        raise ValueError("Cannot compute LMSE for empty images.")

    row_starts = list(range(0, max(h - win_h + 1, 1), stride)) or [0]
    col_starts = list(range(0, max(w - win_w + 1, 1), stride)) or [0]
    if row_starts[-1] != h - win_h:
        row_starts.append(h - win_h)
    if col_starts[-1] != w - win_w:
        col_starts.append(w - win_w)

    ssq = 0.0
    total = 0.0
    for ch in range(c):
        for i in row_starts:
            for j in col_starts:
                correct = target_np[i : i + win_h, j : j + win_w, ch]
                estimate = pred_np[i : i + win_h, j : j + win_w, ch]
                ssq += _scaled_ssq_error(correct, estimate)
                total += float(np.sum(correct**2))

    if total <= 1e-12:
        return 0.0 if ssq <= 1e-12 else float("inf")
    return float(ssq / total)


def compute_metrics(
    pred,
    target,
    *,
    data_range: float = 1.0,
    lmse_window: int = 20,
    lmse_stride: int = 10,
) -> MetricDict:
    """Compute PSNR, SSIM, NCC, and LMSE for one prediction/target pair."""

    return {
        "PSNR": psnr(pred, target, data_range=data_range),
        "SSIM": ssim(pred, target, data_range=data_range),
        "NCC": ncc(pred, target),
        "LMSE": lmse(pred, target, window_size=lmse_window, stride=lmse_stride),
    }


def average_metrics(rows: Iterable[Mapping[str, object]]) -> MetricDict:
    rows = list(rows)
    metrics = ("PSNR", "SSIM", "NCC", "LMSE")
    avg: MetricDict = {}
    for key in metrics:
        values = [float(row[key]) for row in rows if key in row and not math.isnan(float(row[key]))]
        finite = [v for v in values if math.isfinite(v)]
        if values and len(finite) != len(values) and key == "PSNR":
            avg[key] = float("inf")
        else:
            avg[key] = float(np.mean(finite)) if finite else float("nan")
    return avg


def save_metrics_csv(rows: List[Mapping[str, object]], csv_path: Path | str) -> MetricDict:
    """Save per-image metrics and an average row to CSV."""

    path = Path(csv_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    avg = average_metrics(rows)
    fieldnames = ["dataset", "name", "PSNR", "SSIM", "NCC", "LMSE"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})
        avg_row: MutableMapping[str, object] = {"dataset": "average", "name": "average"}
        avg_row.update(avg)
        writer.writerow(avg_row)
    return avg

