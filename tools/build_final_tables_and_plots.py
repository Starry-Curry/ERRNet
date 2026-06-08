"""Build final markdown tables and numeric paper figures from formal eval CSVs."""

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence


COURSE_DATASETS = ["ceilnet", "zhang20", "sir2_objects", "sir2_postcard", "sir2_wild"]
SIR2_DATASETS = ["sir2_objects", "sir2_postcard", "sir2_wild"]
ALL_DATASETS = COURSE_DATASETS + ["openrr_val", "self"]
METRICS = ["PSNR", "SSIM", "NCC", "LMSE"]


@dataclass(frozen=True)
class MethodSpec:
    name: str
    dirs: Sequence[str]
    prior: str = "-"
    gate: str = "-"
    refine: str = "-"
    ric: str = "-"
    fss: str = "-"
    uses_openrr: str = "-"
    openrr_pairs: str = "-"
    course_replay: str = "-"
    lambda_old: str = "-"
    balanced: str = "-"
    role: str = ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create final paper tables and numeric figures.")
    parser.add_argument("--results_root", default="results")
    parser.add_argument("--figures_dir", default="paper/figures")
    parser.add_argument("--dpi", type=int, default=220)
    return parser.parse_args()


def _result_dirs(root: Path, tag: str) -> List[Path]:
    return [
        root / f"{tag}_standard",
        root / f"{tag}_zhang20_512",
        root / f"{tag}_openrr",
        root / f"{tag}_self",
    ]


def _structure_methods(root: Path) -> List[MethodSpec]:
    return [
        MethodSpec("BP-RAP RIC", _result_dirs(root, "final_eval_bprap_ric"), "Y", "Y", "Y", "Y", "N", role="full course-fair model"),
        MethodSpec("w/o prior", _result_dirs(root, "final_eval_ablate_no_prior"), "N", "Y", "Y", "Y", "N", role="constant prior P=1"),
        MethodSpec("w/o gate", _result_dirs(root, "final_eval_ablate_no_gate"), "Y", "N", "Y", "Y", "N", role="remove prior-gated adapter"),
        MethodSpec("w/o refinement", _result_dirs(root, "final_eval_ablate_no_refine"), "Y", "Y", "N", "Y", "N", role="Delta=0, output=Tg"),
        MethodSpec("w/o RIC", _result_dirs(root, "final_eval_ablate_no_ric"), "Y", "Y", "Y", "N", "N", role="same schedule, lambda_ric=0"),
        MethodSpec("RIC+FSS", _result_dirs(root, "final_eval_bprap_ric_fss"), "Y", "Y", "Y", "Y", "Y", role="frequency supervision ablation"),
    ]


def _rafa_methods(root: Path) -> List[MethodSpec]:
    return [
        MethodSpec("ERRNet baseline", _result_dirs(root, "final_eval_errnet"), uses_openrr="N", openrr_pairs="0", course_replay="N", lambda_old="0", balanced="N", role="course baseline"),
        MethodSpec("BP-RAP RIC", _result_dirs(root, "final_eval_bprap_ric"), uses_openrr="N", openrr_pairs="0", course_replay="N", lambda_old="0", balanced="N", role="course-fair main method"),
        MethodSpec("OpenRR-FT 1k", _result_dirs(root, "final_eval_openrr_ft_1k"), uses_openrr="Y", openrr_pairs="1000", course_replay="N", lambda_old="0", balanced="N", role="real-only target adaptation"),
        MethodSpec("Soup a0.25", _result_dirs(root, "final_eval_soup_a0p25"), uses_openrr="indirect", openrr_pairs="1000", course_replay="N/A", lambda_old="N/A", balanced="N", role="checkpoint soup fallback"),
        MethodSpec("RAFA1k", _result_dirs(root, "final_eval_rafa1k"), uses_openrr="Y", openrr_pairs="1000", course_replay="Y", lambda_old="0.2", balanced="N", role="replay anchored adaptation"),
        MethodSpec("RAFA3k", _result_dirs(root, "final_eval_rafa3k"), uses_openrr="Y", openrr_pairs="3000", course_replay="Y", lambda_old="0.2", balanced="N", role="main extra-data algorithm"),
        MethodSpec("Balanced RAFA3k", _result_dirs(root, "final_eval_rafa3k_balanced"), uses_openrr="Y", openrr_pairs="3000", course_replay="Y", lambda_old="0.2", balanced="Y", role="metric-best checkpoint"),
        MethodSpec("Old04", _result_dirs(root, "final_eval_rafa3k_old04"), uses_openrr="Y", openrr_pairs="3000", course_replay="Y", lambda_old="0.4", balanced="N", role="stronger distillation diagnostic"),
        MethodSpec("RAFA3k w/o old", _result_dirs(root, "final_eval_rafa3k_no_old"), uses_openrr="Y", openrr_pairs="3000", course_replay="Y", lambda_old="0.0", balanced="N", role="old-loss ablation"),
        MethodSpec("OpenRR-only 3k", _result_dirs(root, "final_eval_rafa3k_noreplay"), uses_openrr="Y", openrr_pairs="3000", course_replay="N", lambda_old="0.0", balanced="N", role="w/o course replay; OpenRR-only"),
    ]


def _read_average(path: Path) -> Optional[Mapping[str, str]]:
    if not path.exists():
        return None
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return rows[-1] if rows else None


def _find_row(dirs: Iterable[Path], dataset: str) -> Optional[Mapping[str, str]]:
    for root in dirs:
        row = _read_average(root / f"metrics_{dataset}.csv")
        if row is not None:
            return row
    return None


def _method_rows(method: MethodSpec, datasets: Sequence[str]) -> Dict[str, Mapping[str, str]]:
    return {dataset: row for dataset in datasets if (row := _find_row(method.dirs, dataset)) is not None}


def _mean(rows: Mapping[str, Mapping[str, str]], datasets: Sequence[str], metric: str) -> float:
    values = [float(rows[name][metric]) for name in datasets if name in rows and rows[name].get(metric)]
    return sum(values) / len(values) if values else math.nan


def _fmt(value: float) -> str:
    return "missing" if math.isnan(value) else f"{value:.4f}"


def _metric(rows: Mapping[str, Mapping[str, str]], dataset: str, metric: str = "PSNR") -> float:
    if dataset not in rows or not rows[dataset].get(metric):
        return math.nan
    return float(rows[dataset][metric])


def _write(path: Path, lines: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(path)


def _build_structure_summary(methods: Sequence[MethodSpec]) -> List[str]:
    lines = [
        "# Structure Ablation Summary",
        "",
        "Generated from formal evaluation CSV files. Course means use CEILNet native, Zhang20 512, and SIR2 native. OpenRR/self are reported separately.",
        "",
        "| Method | Prior | Gate | Refine | RIC | FSS | Course 5-set PSNR | SIR2 PSNR | OpenRR val PSNR | Self PSNR | Main reading |",
        "| --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for method in methods:
        rows = _method_rows(method, ALL_DATASETS)
        lines.append(
            f"| {method.name} | {method.prior} | {method.gate} | {method.refine} | {method.ric} | {method.fss} | "
            f"{_fmt(_mean(rows, COURSE_DATASETS, 'PSNR'))} | {_fmt(_mean(rows, SIR2_DATASETS, 'PSNR'))} | "
            f"{_fmt(_metric(rows, 'openrr_val'))} | {_fmt(_metric(rows, 'self'))} | {method.role} |"
        )
    return lines


def _build_rafa_summary(methods: Sequence[MethodSpec]) -> List[str]:
    lines = [
        "# RAFA / OpenRR Adaptation Summary",
        "",
        "Generated from formal evaluation CSV files. OpenRR-train methods are extra-data setting and must not be mixed into the course-fair table.",
        "",
        "| Method | Uses OpenRR train? | OpenRR pairs | Course replay? | lambda_old | Balanced sampling? | Course 5-set PSNR | SIR2 PSNR | OpenRR val PSNR | Six-set PSNR | Self PSNR | Role |",
        "| --- | --- | ---: | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    six = COURSE_DATASETS + ["openrr_val"]
    for method in methods:
        rows = _method_rows(method, ALL_DATASETS)
        lines.append(
            f"| {method.name} | {method.uses_openrr} | {method.openrr_pairs} | {method.course_replay} | {method.lambda_old} | {method.balanced} | "
            f"{_fmt(_mean(rows, COURSE_DATASETS, 'PSNR'))} | {_fmt(_mean(rows, SIR2_DATASETS, 'PSNR'))} | "
            f"{_fmt(_metric(rows, 'openrr_val'))} | {_fmt(_mean(rows, six, 'PSNR'))} | {_fmt(_metric(rows, 'self'))} | {method.role} |"
        )
    return lines


def _build_self_summary(methods: Sequence[MethodSpec]) -> List[str]:
    lines = [
        "# Self-Collected Summary",
        "",
        "Requires `data/self_collected/test/scene_xxx/blended.png` and `transmission.png`, evaluated with `--datasets self`.",
        "",
        "## Average Metrics",
        "",
        "| Method | PSNR | SSIM | NCC | LMSE |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for method in methods:
        row = _find_row(method.dirs, "self")
        if row is None:
            lines.append(f"| {method.name} | missing | missing | missing | missing |")
        else:
            lines.append(f"| {method.name} | {float(row['PSNR']):.4f} | {float(row['SSIM']):.4f} | {float(row['NCC']):.4f} | {float(row['LMSE']):.4f} |")
    lines.extend([
        "",
        "## Per-Scene Metrics",
        "",
        "The per-scene rows are stored in each method directory as `metrics_self.csv`.",
        "",
        "## Qualitative Paths",
        "",
        "- `paper/figures/qualitative_main.png` / `.pdf`",
        "- `paper/figures/prior_error_analysis.png` / `.pdf`",
    ])
    return lines


def _build_final_tables(course_methods: Sequence[MethodSpec], structure_methods: Sequence[MethodSpec], rafa_methods: Sequence[MethodSpec]) -> List[str]:
    lines = [
        "# Final Tables For Paper",
        "",
        "All tables below are generated from formal evaluation CSV files. Missing values mean the corresponding formal eval has not been run yet.",
        "",
        "## Table 1. Dataset And Protocol",
        "",
        "| Split | Dataset | Count | Protocol |",
        "| --- | --- | ---: | --- |",
        "| Test | CEILNet Table2 | 100 | native |",
        "| Test | Zhang real20 | 20 | max long edge 512 |",
        "| Test | SIR2 Objects | 200 | native |",
        "| Test | SIR2 Postcard | 179 | native |",
        "| Test | SIR2 Wild | 101 | native |",
        "| External | OpenRR val | 300 | native |",
        "| Self | self-collected | >=5 | native unless memory requires documented resize |",
        "",
        "## Table 2. Course-Fair Results",
        "",
        "| Method | Uses OpenRR train? | Course 5-set PSNR | SSIM | NCC | LMSE | SIR2 PSNR | SIR2 SSIM | SIR2 NCC | SIR2 LMSE |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for method in course_methods:
        rows = _method_rows(method, ALL_DATASETS)
        lines.append(
            f"| {method.name} | N | {_fmt(_mean(rows, COURSE_DATASETS, 'PSNR'))} | {_fmt(_mean(rows, COURSE_DATASETS, 'SSIM'))} | "
            f"{_fmt(_mean(rows, COURSE_DATASETS, 'NCC'))} | {_fmt(_mean(rows, COURSE_DATASETS, 'LMSE'))} | "
            f"{_fmt(_mean(rows, SIR2_DATASETS, 'PSNR'))} | {_fmt(_mean(rows, SIR2_DATASETS, 'SSIM'))} | "
            f"{_fmt(_mean(rows, SIR2_DATASETS, 'NCC'))} | {_fmt(_mean(rows, SIR2_DATASETS, 'LMSE'))} |"
        )
    lines.extend(["", "## Table 3. Structure Ablation", ""])
    lines.extend(_build_structure_summary(structure_methods)[4:])
    lines.extend(["", "## Table 4. OpenRR / RAFA Adaptation", ""])
    lines.extend(_build_rafa_summary(rafa_methods)[4:])
    self_methods = [m for m in rafa_methods if m.name in {"ERRNet baseline", "BP-RAP RIC", "Soup a0.25", "RAFA3k", "Balanced RAFA3k"}]
    lines.extend(["", "## Table 5. Self-Collected Results", ""])
    lines.extend(_build_self_summary(self_methods)[5:9])
    lines.extend([
        "",
        "## Figure Caption Drafts",
        "",
        "**Figure 1. Method overview.** BP-RAP starts from ERRNet-Hyper output, predicts a reflection prior, applies prior-gated adaptation and residual refinement, and uses RIC during training. RAFA is an extra-data training branch using OpenRR supervision, course replay, and teacher distillation.",
        "",
        "**Figure 2. Course PSNR delta.** Per-dataset PSNR difference between BP-RAP RIC and ERRNet. Negative values on CEILNet/Zhang reflect pixel-aligned fidelity loss; positive values on SIR2 show stronger real-scene performance.",
        "",
        "**Figure 3. OpenRR adaptation.** OpenRR val PSNR and course/SIR2 stability across ERRNet, BP-RAP RIC, OpenRR-FT, soup, RAFA3k, and Balanced RAFA3k. OpenRR-trained methods are reported as extra-data results.",
        "",
        "**Figure 4. Qualitative main examples.** Input, ERRNet, BP-RAP RIC, Balanced RAFA3k, and ground truth for selected success cases.",
        "",
        "**Figure 5. Prior and error analysis.** Error maps use a shared per-row color scale, and prior maps show where the final model applies reflection-aware correction.",
        "",
        "**Figure 6. Failure cases.** CEILNet and Zhang20 examples where ERRNet outperforms BP-RAP/RAFA, illustrating pixel-aligned fidelity and color/texture drift limitations.",
    ])
    return lines


def _save_plot(fig, figures_dir: Path, name: str, dpi: int) -> None:
    figures_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(figures_dir / f"{name}.png", dpi=dpi, bbox_inches="tight")
    fig.savefig(figures_dir / f"{name}.pdf", bbox_inches="tight")
    print(figures_dir / f"{name}.png")
    print(figures_dir / f"{name}.pdf")


def _plot_method_overview(figures_dir: Path, dpi: int) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

    fig, ax = plt.subplots(figsize=(11.5, 4.8))
    ax.axis("off")
    boxes = {
        "Input": (0.04, 0.55, 0.12, 0.16),
        "ERRNet-Hyper\nT0": (0.22, 0.55, 0.16, 0.16),
        "Prior Head\nP": (0.22, 0.23, 0.16, 0.16),
        "Prior-gated\nAdapter": (0.45, 0.55, 0.16, 0.16),
        "Residual\nRefinement": (0.66, 0.55, 0.16, 0.16),
        "Output": (0.88, 0.55, 0.09, 0.16),
        "RIC branch\ntraining only": (0.45, 0.18, 0.16, 0.14),
        "RAFA branch\nOpenRR + replay": (0.66, 0.18, 0.20, 0.14),
    }
    for label, (x, y, w, h) in boxes.items():
        dashed = "training only" in label or "OpenRR" in label
        box = FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.012,rounding_size=0.025",
            linewidth=1.4,
            edgecolor="#263238",
            facecolor="#f7f9fb" if not dashed else "#fff8e1",
            linestyle="--" if dashed else "-",
        )
        ax.add_patch(box)
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=10)

    def arrow(a: str, b: str, dashed: bool = False) -> None:
        ax1, ay1, aw, ah = boxes[a]
        bx, by, bw, bh = boxes[b]
        start = (ax1 + aw, ay1 + ah / 2)
        end = (bx, by + bh / 2)
        ax.add_patch(
            FancyArrowPatch(
                start,
                end,
                arrowstyle="-|>",
                mutation_scale=12,
                linewidth=1.3,
                color="#263238",
                linestyle="--" if dashed else "-",
            )
        )

    arrow("Input", "ERRNet-Hyper\nT0")
    arrow("ERRNet-Hyper\nT0", "Prior-gated\nAdapter")
    arrow("Prior-gated\nAdapter", "Residual\nRefinement")
    arrow("Residual\nRefinement", "Output")
    ax.add_patch(FancyArrowPatch((0.28, 0.55), (0.28, 0.39), arrowstyle="-|>", mutation_scale=12, linewidth=1.2, color="#263238"))
    ax.add_patch(FancyArrowPatch((0.38, 0.31), (0.45, 0.59), arrowstyle="-|>", mutation_scale=12, linewidth=1.2, color="#263238"))
    ax.add_patch(FancyArrowPatch((0.53, 0.55), (0.53, 0.32), arrowstyle="-|>", mutation_scale=12, linewidth=1.2, color="#263238", linestyle="--"))
    ax.add_patch(FancyArrowPatch((0.74, 0.55), (0.76, 0.32), arrowstyle="-|>", mutation_scale=12, linewidth=1.2, color="#263238", linestyle="--"))
    ax.text(0.5, 0.93, "BP-RAP / RIC / RAFA Overview", ha="center", va="center", fontsize=14, fontweight="bold")
    ax.text(0.53, 0.08, "Dashed components are training-only; inference is a single forward pass.", ha="center", fontsize=9, color="#455a64")
    _save_plot(fig, figures_dir, "method_overview", dpi)
    plt.close(fig)


def _plot_course_delta(root: Path, figures_dir: Path, dpi: int) -> None:
    import matplotlib.pyplot as plt

    err = _method_rows(MethodSpec("ERRNet", _result_dirs(root, "final_eval_errnet")), COURSE_DATASETS)
    bp = _method_rows(MethodSpec("BP-RAP RIC", _result_dirs(root, "final_eval_bprap_ric")), COURSE_DATASETS)
    labels = ["CEILNet", "Zhang20", "Objects", "Postcard", "Wild"]
    deltas = [_metric(bp, ds) - _metric(err, ds) for ds in COURSE_DATASETS]
    fig, ax = plt.subplots(figsize=(8, 3.6))
    colors = ["#c0392b" if v < 0 else "#2e7d32" for v in deltas]
    ax.bar(labels, deltas, color=colors)
    ax.axhline(0, color="#263238", linewidth=1)
    ax.set_ylabel("Delta PSNR vs ERRNet (dB)")
    ax.set_title("Course-Fair BP-RAP RIC Delta")
    ax.grid(axis="y", alpha=0.25)
    for index, value in enumerate(deltas):
        if not math.isnan(value):
            ax.text(index, value + (0.08 if value >= 0 else -0.18), f"{value:+.2f}", ha="center", va="bottom" if value >= 0 else "top", fontsize=9)
    _save_plot(fig, figures_dir, "course_delta_psnr", dpi)
    plt.close(fig)


def _plot_openrr_summary(root: Path, figures_dir: Path, dpi: int) -> None:
    import matplotlib.pyplot as plt

    methods = [
        MethodSpec("ERRNet", _result_dirs(root, "final_eval_errnet")),
        MethodSpec("BP-RAP RIC", _result_dirs(root, "final_eval_bprap_ric")),
        MethodSpec("OpenRR-FT", _result_dirs(root, "final_eval_openrr_ft_1k")),
        MethodSpec("Soup", _result_dirs(root, "final_eval_soup_a0p25")),
        MethodSpec("RAFA3k", _result_dirs(root, "final_eval_rafa3k")),
        MethodSpec("Balanced", _result_dirs(root, "final_eval_rafa3k_balanced")),
    ]
    labels = [m.name for m in methods]
    openrr = []
    course = []
    for method in methods:
        rows = _method_rows(method, COURSE_DATASETS + ["openrr_val"])
        openrr.append(_metric(rows, "openrr_val"))
        course.append(_mean(rows, COURSE_DATASETS, "PSNR"))
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.8))
    axes[0].bar(labels, openrr, color="#1565c0")
    axes[0].set_title("OpenRR val PSNR")
    axes[0].set_ylabel("PSNR")
    axes[1].bar(labels, course, color="#6a1b9a")
    axes[1].set_title("Course 5-set mean PSNR")
    for ax in axes:
        ax.tick_params(axis="x", rotation=30)
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle("OpenRR Adaptation Summary (extra-data methods are marked by OpenRR train usage in the table)")
    _save_plot(fig, figures_dir, "openrr_adaptation_summary", dpi)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    root = Path(args.results_root)
    figures_dir = Path(args.figures_dir)
    structure_methods = _structure_methods(root)
    rafa_methods = _rafa_methods(root)
    course_methods = [
        MethodSpec("ERRNet baseline", _result_dirs(root, "final_eval_errnet")),
        MethodSpec("BP-RAP RIC", _result_dirs(root, "final_eval_bprap_ric")),
        MethodSpec("RIC+FSS", _result_dirs(root, "final_eval_bprap_ric_fss")),
    ]
    self_methods = [m for m in rafa_methods if m.name in {"ERRNet baseline", "BP-RAP RIC", "Soup a0.25", "RAFA3k", "Balanced RAFA3k"}]

    _write(root / "ABLATION_STRUCTURE_SUMMARY.md", _build_structure_summary(structure_methods))
    _write(root / "ABLATION_RAFA_SUMMARY.md", _build_rafa_summary(rafa_methods))
    _write(root / "SELF_COLLECTED_SUMMARY.md", _build_self_summary(self_methods))
    _write(root / "FINAL_TABLES_FOR_PAPER.md", _build_final_tables(course_methods, structure_methods, rafa_methods))

    _plot_method_overview(figures_dir, args.dpi)
    _plot_course_delta(root, figures_dir, args.dpi)
    _plot_openrr_summary(root, figures_dir, args.dpi)


if __name__ == "__main__":
    main()
