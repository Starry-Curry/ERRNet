"""Analyze ERRNet-RAFA fusion sweep outputs.

This script reads outputs from ``tools/eval_fusion_sweep.py`` and writes a
markdown report with per-dataset gains and adaptive-alpha distribution.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence


COURSE_DATASETS = ["ceilnet", "zhang20", "sir2_objects", "sir2_postcard", "sir2_wild"]
SIR2_DATASETS = ["sir2_objects", "sir2_postcard", "sir2_wild"]
ALL_DATASETS = COURSE_DATASETS + ["openrr_val"]
DEFAULT_LABELS = ["alpha_0p00", "alpha_0p25", "alpha_0p50", "alpha_0p75", "alpha_1p00", "adaptive"]
LABEL_NAMES = {
    "alpha_0p00": "ERRNet",
    "alpha_0p25": "Fusion a=0.25",
    "alpha_0p50": "Fusion a=0.50",
    "alpha_0p75": "Fusion a=0.75",
    "alpha_1p00": "RAFA3k no-old",
    "adaptive": "Adaptive fusion",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze fusion sweep metrics and adaptive-alpha routing.")
    parser.add_argument("--standard_dir", required=True, help="fusion sweep directory for native CEILNet/SIR2/OpenRR")
    parser.add_argument("--zhang_dir", required=True, help="fusion sweep directory for Zhang20 512")
    parser.add_argument("--out", default="results/FUSION_ROUTING_ANALYSIS.md")
    parser.add_argument("--labels", default=",".join(DEFAULT_LABELS))
    parser.add_argument("--errnet_threshold", type=float, default=0.33, help="adaptive alpha <= threshold is ERRNet-leaning")
    parser.add_argument("--rafa_threshold", type=float, default=0.67, help="adaptive alpha >= threshold is RAFA-leaning")
    return parser.parse_args()


def _labels(text: str) -> List[str]:
    return [item.strip() for item in text.split(",") if item.strip()]


def _read_rows(path: Path) -> List[Mapping[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _metric_dir(standard_dir: Path, zhang_dir: Path, dataset: str) -> Path:
    return zhang_dir if dataset == "zhang20" else standard_dir


def _read_metric_rows(standard_dir: Path, zhang_dir: Path, label: str, dataset: str) -> List[Mapping[str, str]]:
    root = _metric_dir(standard_dir, zhang_dir, dataset)
    rows = _read_rows(root / label / f"metrics_{dataset}.csv")
    return [row for row in rows if row.get("dataset") != "average"]


def _mean(values: Iterable[float]) -> float:
    values = [value for value in values if math.isfinite(value)]
    return sum(values) / len(values) if values else math.nan


def _fmt(value: float) -> str:
    return "missing" if math.isnan(value) else f"{value:.4f}"


def _pct(count: int, total: int) -> str:
    return "0.0%" if total <= 0 else f"{100.0 * count / total:.1f}%"


def _rows_by_name(rows: Sequence[Mapping[str, str]]) -> Dict[str, Mapping[str, str]]:
    return {str(row.get("name", "")): row for row in rows if row.get("name")}


def _psnr(row: Optional[Mapping[str, str]]) -> float:
    if row is None or not row.get("PSNR"):
        return math.nan
    return float(row["PSNR"])


def _dataset_stats(standard_dir: Path, zhang_dir: Path, labels: Sequence[str], dataset: str) -> Dict[str, Dict[str, float]]:
    err_rows = _rows_by_name(_read_metric_rows(standard_dir, zhang_dir, "alpha_0p00", dataset))
    rafa_rows = _rows_by_name(_read_metric_rows(standard_dir, zhang_dir, "alpha_1p00", dataset))
    stats: Dict[str, Dict[str, float]] = {}
    for label in labels:
        rows = _read_metric_rows(standard_dir, zhang_dir, label, dataset)
        values = []
        delta_err = []
        delta_rafa = []
        win_err = 0
        win_rafa = 0
        valid = 0
        for row in rows:
            name = str(row.get("name", ""))
            value = _psnr(row)
            err = _psnr(err_rows.get(name))
            rafa = _psnr(rafa_rows.get(name))
            if math.isnan(value):
                continue
            values.append(value)
            if math.isfinite(err):
                delta_err.append(value - err)
                win_err += int(value > err)
            if math.isfinite(rafa):
                delta_rafa.append(value - rafa)
                win_rafa += int(value > rafa)
            valid += 1
        stats[label] = {
            "count": float(valid),
            "psnr": _mean(values),
            "delta_err": _mean(delta_err),
            "delta_rafa": _mean(delta_rafa),
            "win_err": float(win_err),
            "win_rafa": float(win_rafa),
        }
    return stats


def _feature_rows(standard_dir: Path, zhang_dir: Path, dataset: str) -> List[Mapping[str, str]]:
    root = _metric_dir(standard_dir, zhang_dir, dataset)
    return _read_rows(root / "features" / f"features_{dataset}.csv")


def _adaptive_alpha_stats(rows: Sequence[Mapping[str, str]], err_thr: float, rafa_thr: float) -> Mapping[str, float]:
    values = [float(row["adaptive_alpha"]) for row in rows if row.get("adaptive_alpha")]
    if not values:
        return {
            "count": 0,
            "mean": math.nan,
            "min": math.nan,
            "max": math.nan,
            "errnet_lean": 0,
            "mixed": 0,
            "rafa_lean": 0,
            "rafa_oracle_wins": 0,
        }
    errnet_lean = sum(1 for value in values if value <= err_thr)
    rafa_lean = sum(1 for value in values if value >= rafa_thr)
    mixed = len(values) - errnet_lean - rafa_lean
    rafa_oracle_wins = sum(1 for row in rows if row.get("RAFA_minus_ERR_PSNR") and float(row["RAFA_minus_ERR_PSNR"]) > 0)
    return {
        "count": len(values),
        "mean": _mean(values),
        "min": min(values),
        "max": max(values),
        "errnet_lean": errnet_lean,
        "mixed": mixed,
        "rafa_lean": rafa_lean,
        "rafa_oracle_wins": rafa_oracle_wins,
    }


def _overall_mean(dataset_stats: Mapping[str, Dict[str, Dict[str, float]]], label: str, datasets: Sequence[str], key: str) -> float:
    return _mean(dataset_stats[dataset][label][key] for dataset in datasets if dataset in dataset_stats)


def build_report(args: argparse.Namespace) -> List[str]:
    standard_dir = Path(args.standard_dir)
    zhang_dir = Path(args.zhang_dir)
    labels = _labels(args.labels)
    dataset_stats = {
        dataset: _dataset_stats(standard_dir, zhang_dir, labels, dataset)
        for dataset in ALL_DATASETS
    }

    lines = [
        "# Fusion Routing And Gain Analysis",
        "",
        "Generated from `tools/eval_fusion_sweep.py` outputs. Constant-alpha fusion is not a hard route: every image uses both ERRNet and RAFA with the same alpha.",
        "",
        "## Mean PSNR And Gains",
        "",
        "| Method | Course PSNR | d vs ERRNet | SIR2 PSNR | d vs ERRNet | OpenRR PSNR | d vs ERRNet | Six-set PSNR |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    six = COURSE_DATASETS + ["openrr_val"]
    for label in labels:
        name = LABEL_NAMES.get(label, label)
        course = _overall_mean(dataset_stats, label, COURSE_DATASETS, "psnr")
        course_d = _overall_mean(dataset_stats, label, COURSE_DATASETS, "delta_err")
        sir2 = _overall_mean(dataset_stats, label, SIR2_DATASETS, "psnr")
        sir2_d = _overall_mean(dataset_stats, label, SIR2_DATASETS, "delta_err")
        openrr = dataset_stats["openrr_val"][label]["psnr"]
        openrr_d = dataset_stats["openrr_val"][label]["delta_err"]
        six_mean = _overall_mean(dataset_stats, label, six, "psnr")
        lines.append(
            f"| {name} | {_fmt(course)} | {_fmt(course_d)} | {_fmt(sir2)} | {_fmt(sir2_d)} | "
            f"{_fmt(openrr)} | {_fmt(openrr_d)} | {_fmt(six_mean)} |"
        )

    lines.extend([
        "",
        "## Per-Dataset PSNR Delta vs ERRNet",
        "",
        "| Method | CEILNet | Zhang20 | Objects | Postcard | Wild | OpenRR |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ])
    for label in labels:
        name = LABEL_NAMES.get(label, label)
        cells = [f"| {name}"]
        for dataset in ALL_DATASETS:
            cells.append(_fmt(dataset_stats[dataset][label]["delta_err"]))
        lines.append(" | ".join(cells) + " |")

    lines.extend([
        "",
        "## Per-Dataset Win Counts",
        "",
        "Rows report how many samples a method beats ERRNet / RAFA3k-no-old by PSNR.",
        "",
        "| Dataset | Method | Count | Beat ERRNet | Beat RAFA | Mean d vs ERRNet | Mean d vs RAFA |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ])
    for dataset in ALL_DATASETS:
        for label in labels:
            if label in {"alpha_0p00", "alpha_1p00"}:
                continue
            stat = dataset_stats[dataset][label]
            count = int(stat["count"])
            lines.append(
                f"| {dataset} | {LABEL_NAMES.get(label, label)} | {count} | "
                f"{int(stat['win_err'])} ({_pct(int(stat['win_err']), count)}) | "
                f"{int(stat['win_rafa'])} ({_pct(int(stat['win_rafa']), count)}) | "
                f"{_fmt(stat['delta_err'])} | {_fmt(stat['delta_rafa'])} |"
            )

    lines.extend([
        "",
        "## Adaptive Alpha Distribution",
        "",
        f"`alpha <= {args.errnet_threshold:.2f}` is ERRNet-leaning, `alpha >= {args.rafa_threshold:.2f}` is RAFA-leaning, and the middle interval is mixed.",
        "",
        "| Dataset | Count | Mean alpha | Min | Max | ERRNet-leaning | Mixed | RAFA-leaning | Oracle RAFA wins vs ERRNet |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ])
    for dataset in ALL_DATASETS:
        rows = _feature_rows(standard_dir, zhang_dir, dataset)
        stat = _adaptive_alpha_stats(rows, float(args.errnet_threshold), float(args.rafa_threshold))
        count = int(stat["count"])
        lines.append(
            f"| {dataset} | {count} | {_fmt(float(stat['mean']))} | {_fmt(float(stat['min']))} | "
            f"{_fmt(float(stat['max']))} | {int(stat['errnet_lean'])} ({_pct(int(stat['errnet_lean']), count)}) | "
            f"{int(stat['mixed'])} ({_pct(int(stat['mixed']), count)}) | "
            f"{int(stat['rafa_lean'])} ({_pct(int(stat['rafa_lean']), count)}) | "
            f"{int(stat['rafa_oracle_wins'])} ({_pct(int(stat['rafa_oracle_wins']), count)}) |"
        )

    lines.extend([
        "",
        "## Recommended Reading",
        "",
        "- Use `Fusion a=0.50` as the robust final inference strategy if it keeps CEILNet/Zhang close enough to ERRNet while retaining strong SIR2/OpenRR gains.",
        "- Use `Fusion a=0.25` as a conservative fallback if the report prioritizes course-test stability over OpenRR gain.",
        "- Adaptive fusion should only be claimed as the final method if its selector distribution and per-dataset gains are clearly better than constant-alpha fusion.",
    ])
    return lines


def main() -> None:
    args = parse_args()
    lines = build_report(args)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
