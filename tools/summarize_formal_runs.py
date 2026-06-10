"""Summarize formal reflection-removal eval CSVs.

Each run is passed as ``LABEL=STANDARD_DIR,ZHANG20_512_DIR``. The standard
directory should contain CEILNet, SIR2, and OpenRR native-resolution metrics;
the Zhang directory should contain Zhang20 evaluated with ``--max_long_edge
512``.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple


COURSE_DATASETS = ["ceilnet", "zhang20", "sir2_objects", "sir2_postcard", "sir2_wild"]
SIR2_DATASETS = ["sir2_objects", "sir2_postcard", "sir2_wild"]
STANDARD_DATASETS = ["ceilnet", "sir2_objects", "sir2_postcard", "sir2_wild", "openrr_val"]
METRICS = ["PSNR", "SSIM", "NCC", "LMSE"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize formal eval_all.py outputs.")
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        metavar="LABEL=STANDARD_DIR,ZHANG20_512_DIR",
        help="Method label and two formal result directories. Repeat for multiple methods.",
    )
    parser.add_argument("--baseline", default=None, help="Label used for delta columns. Defaults to the first --run label.")
    parser.add_argument("--out", default=None, help="Optional markdown output path.")
    return parser.parse_args()


def _parse_run_spec(spec: str) -> Tuple[str, Path, Path]:
    if "=" not in spec:
        raise ValueError(f"Invalid --run spec, expected LABEL=STD,ZHANG: {spec}")
    label, dirs = spec.split("=", 1)
    parts = [part.strip() for part in dirs.split(",") if part.strip()]
    if len(parts) != 2:
        raise ValueError(f"Invalid --run dirs, expected STANDARD_DIR,ZHANG20_512_DIR: {spec}")
    return label.strip(), Path(parts[0]), Path(parts[1])


def _read_last_row(path: Path) -> Mapping[str, str] | None:
    if not path.exists():
        return None
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return rows[-1] if rows else None


def _read_run(standard_dir: Path, zhang_dir: Path) -> Dict[str, Mapping[str, str]]:
    rows: Dict[str, Mapping[str, str]] = {}
    for dataset in STANDARD_DATASETS:
        row = _read_last_row(standard_dir / f"metrics_{dataset}.csv")
        if row is not None:
            rows[dataset] = row
    row = _read_last_row(zhang_dir / "metrics_zhang20.csv")
    if row is not None:
        rows["zhang20"] = row
    return rows


def _mean(rows: Mapping[str, Mapping[str, str]], datasets: Sequence[str], metric: str) -> float:
    values = [float(rows[name][metric]) for name in datasets if name in rows and rows[name].get(metric)]
    return sum(values) / len(values) if values else math.nan


def _metric(rows: Mapping[str, Mapping[str, str]], dataset: str, metric: str) -> float:
    if dataset not in rows or not rows[dataset].get(metric):
        return math.nan
    return float(rows[dataset][metric])


def _fmt(value: float) -> str:
    return "missing" if math.isnan(value) else f"{value:.4f}"


def _delta(value: float, baseline: float) -> float:
    if math.isnan(value) or math.isnan(baseline):
        return math.nan
    return value - baseline


def _build_summary(run_specs: Iterable[Tuple[str, Path, Path]], baseline_label: str | None) -> List[str]:
    runs = [(label, _read_run(std, zhang)) for label, std, zhang in run_specs]
    if not runs:
        raise ValueError("No runs were provided.")
    if baseline_label is None:
        baseline_label = runs[0][0]
    baseline_rows = next((rows for label, rows in runs if label == baseline_label), None)
    if baseline_rows is None:
        raise ValueError(f"Baseline label not found in --run specs: {baseline_label}")

    lines = [
        "# Formal Evaluation Summary",
        "",
        "Protocol: CEILNet/SIR2/OpenRR use native resolution; Zhang20 uses `--max_long_edge 512`.",
        "",
        "## Mean Metrics",
        "",
        "| Method | Course PSNR | dPSNR | Course SSIM | SIR2 PSNR | dPSNR | OpenRR PSNR | dPSNR | Six-set PSNR | dPSNR |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    six = COURSE_DATASETS + ["openrr_val"]
    base_course = _mean(baseline_rows, COURSE_DATASETS, "PSNR")
    base_sir2 = _mean(baseline_rows, SIR2_DATASETS, "PSNR")
    base_openrr = _metric(baseline_rows, "openrr_val", "PSNR")
    base_six = _mean(baseline_rows, six, "PSNR")
    for label, rows in runs:
        course = _mean(rows, COURSE_DATASETS, "PSNR")
        sir2 = _mean(rows, SIR2_DATASETS, "PSNR")
        openrr = _metric(rows, "openrr_val", "PSNR")
        six_mean = _mean(rows, six, "PSNR")
        lines.append(
            f"| {label} | {_fmt(course)} | {_fmt(_delta(course, base_course))} | "
            f"{_fmt(_mean(rows, COURSE_DATASETS, 'SSIM'))} | {_fmt(sir2)} | {_fmt(_delta(sir2, base_sir2))} | "
            f"{_fmt(openrr)} | {_fmt(_delta(openrr, base_openrr))} | {_fmt(six_mean)} | {_fmt(_delta(six_mean, base_six))} |"
        )

    lines.extend(
        [
            "",
            "## Per-Dataset PSNR",
            "",
            "| Method | CEILNet | d | Zhang20 | d | Objects | d | Postcard | d | Wild | d | OpenRR | d |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    datasets = COURSE_DATASETS + ["openrr_val"]
    for label, rows in runs:
        cells = [f"| {label}"]
        for dataset in datasets:
            value = _metric(rows, dataset, "PSNR")
            base = _metric(baseline_rows, dataset, "PSNR")
            cells.append(f"{_fmt(value)} | {_fmt(_delta(value, base))}")
        lines.append(" | ".join(cells) + " |")

    lines.extend(
        [
            "",
            "## Reading Guide",
            "",
            "- Compare against ERRNet for CEILNet/Zhang20 protection.",
            "- Compare against RAFA3k w/o old for OpenRR/SIR2 regression.",
            "- A useful hard-synthesis run should recover CEILNet/Zhang20 PSNR while losing little SIR2/OpenRR PSNR.",
        ]
    )
    return lines


def main() -> None:
    args = parse_args()
    run_specs = [_parse_run_spec(spec) for spec in args.run]
    lines = _build_summary(run_specs, args.baseline)
    text = "\n".join(lines) + "\n"
    if args.out:
        path = Path(args.out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        print(path)
    else:
        print(text)


if __name__ == "__main__":
    main()
