"""Collect eval_all.py metric CSVs into a markdown summary."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, Iterable, List, Mapping


DEFAULT_DATASETS = "ceilnet,zhang20,sir2_objects,sir2_postcard,sir2_wild,openrr_val,self"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize eval_all.py result directories.")
    parser.add_argument(
        "--method",
        action="append",
        default=[],
        help="method spec in the form Name=dir1,dir2. Can be repeated.",
    )
    parser.add_argument("--datasets", default=DEFAULT_DATASETS, help="comma-separated dataset order")
    parser.add_argument("--out", default="results/NEXT_STAGE_SUMMARY.md")
    return parser.parse_args()


def _parse_method_specs(specs: Iterable[str]) -> Dict[str, List[Path]]:
    methods: Dict[str, List[Path]] = {}
    for spec in specs:
        if "=" not in spec:
            raise ValueError(f"Invalid --method spec {spec!r}; expected Name=dir1,dir2.")
        name, dirs = spec.split("=", 1)
        paths = [Path(item.strip()) for item in dirs.split(",") if item.strip()]
        if not paths:
            raise ValueError(f"Method {name!r} has no result directories.")
        methods[name.strip()] = paths
    return methods


def _read_average(path: Path) -> Mapping[str, str] | None:
    if not path.exists():
        return None
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return rows[-1] if rows else None


def _find_metric_row(paths: Iterable[Path], dataset: str) -> Mapping[str, str] | None:
    for root in paths:
        row = _read_average(root / f"metrics_{dataset}.csv")
        if row is not None:
            return row
    return None


def _fmt(value: float) -> str:
    return f"{value:.4f}"


def _mean(rows: List[Mapping[str, str]], key: str) -> float:
    values = [float(row[key]) for row in rows if key in row and row[key] not in ("", "nan")]
    return sum(values) / len(values) if values else float("nan")


def build_markdown(methods: Mapping[str, List[Path]], datasets: List[str]) -> str:
    lines: List[str] = []
    lines.append("# Next Stage Result Summary")
    lines.append("")
    lines.append("Generated from `eval_all.py` metric CSV files.")
    lines.append("")
    lines.append("## Per-Dataset Metrics")
    lines.append("")
    lines.append("| Method | Dataset | PSNR | SSIM | NCC | LMSE |")
    lines.append("| --- | --- | ---: | ---: | ---: | ---: |")

    method_rows: Dict[str, List[Mapping[str, str]]] = {}
    for method, paths in methods.items():
        method_rows[method] = []
        for dataset in datasets:
            row = _find_metric_row(paths, dataset)
            if row is None:
                continue
            method_rows[method].append(row)
            lines.append(
                "| "
                + " | ".join(
                    [
                        method,
                        dataset,
                        _fmt(float(row["PSNR"])),
                        _fmt(float(row["SSIM"])),
                        _fmt(float(row["NCC"])),
                        _fmt(float(row["LMSE"])),
                    ]
                )
                + " |"
            )

    lines.append("")
    lines.append("## Means Over Available Datasets")
    lines.append("")
    lines.append("| Method | Count | PSNR | SSIM | NCC | LMSE |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: |")
    for method, rows in method_rows.items():
        if not rows:
            continue
        lines.append(
            "| "
            + " | ".join(
                [
                    method,
                    str(len(rows)),
                    _fmt(_mean(rows, "PSNR")),
                    _fmt(_mean(rows, "SSIM")),
                    _fmt(_mean(rows, "NCC")),
                    _fmt(_mean(rows, "LMSE")),
                ]
            )
            + " |"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    methods = _parse_method_specs(args.method)
    datasets = [item.strip() for item in args.datasets.split(",") if item.strip()]
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(build_markdown(methods, datasets), encoding="utf-8")
    print(out_path)


if __name__ == "__main__":
    main()
