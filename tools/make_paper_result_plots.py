"""Generate compact result plots for the course report.

The values are taken from the confirmed formal evaluations in
EXPERIMENT_LOG_RAP_ERRNET.md.  This script only creates paper figures; it does
not evaluate models.
"""

from __future__ import annotations

import argparse
from pathlib import Path


COURSE_DATASETS = ["CEILNet", "Zhang20", "SIR2 Obj.", "SIR2 Post.", "SIR2 Wild"]
ERRNET_PSNR = [27.6414, 23.4367, 24.6983, 21.8856, 24.8763]
BPRAP_RIC_PSNR = [23.9710, 21.0237, 25.8905, 22.4479, 26.1264]

EXTRA_METHODS = ["ERRNet", "BP-RAP RIC", "RAFA3k", "Balanced RAFA"]
OPENRR_PSNR = [25.4874, 26.8597, 27.8483, 27.8349]

SIX_SET_METHODS = ["OpenRR-Soup", "RAFA1k", "RAFA3k", "Balanced RAFA"]
SIX_SET_MEAN_PSNR = [24.5308, 24.5779, 24.6336, 24.6390]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create paper result summary plots.")
    parser.add_argument("--out_dir", default="paper/figures")
    parser.add_argument("--dpi", type=int, default=220)
    return parser.parse_args()


def _annotate_bars(ax, bars, fmt="{:.2f}", dy=0.04):
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + dy,
            fmt.format(height),
            ha="center",
            va="bottom",
            fontsize=7,
        )


def main() -> None:
    import numpy as np
    import matplotlib.pyplot as plt

    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update(
        {
            "font.size": 8,
            "axes.titlesize": 9,
            "axes.labelsize": 8,
            "legend.fontsize": 7,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
        }
    )

    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.0), constrained_layout=True)

    x = np.arange(len(COURSE_DATASETS))
    width = 0.38
    bars_a = axes[0].bar(x - width / 2, ERRNET_PSNR, width, label="ERRNet", color="#6a8caf")
    bars_b = axes[0].bar(x + width / 2, BPRAP_RIC_PSNR, width, label="BP-RAP RIC", color="#d9824b")
    axes[0].set_title("Course benchmark PSNR")
    axes[0].set_ylabel("PSNR (dB)")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(COURSE_DATASETS, rotation=25, ha="right")
    axes[0].legend(frameon=False)
    axes[0].grid(axis="y", alpha=0.25)

    delta = np.array(BPRAP_RIC_PSNR) - np.array(ERRNET_PSNR)
    colors = ["#3a9d6f" if value >= 0 else "#c44e52" for value in delta]
    bars = axes[1].bar(x, delta, color=colors)
    axes[1].axhline(0, color="black", linewidth=0.8)
    axes[1].set_title("BP-RAP RIC vs ERRNet")
    axes[1].set_ylabel("PSNR delta (dB)")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(COURSE_DATASETS, rotation=25, ha="right")
    axes[1].grid(axis="y", alpha=0.25)
    for bar, value in zip(bars, delta):
        va = "bottom" if value >= 0 else "top"
        offset = 0.05 if value >= 0 else -0.08
        axes[1].text(bar.get_x() + bar.get_width() / 2.0, value + offset, f"{value:+.2f}", ha="center", va=va, fontsize=7)

    x2 = np.arange(len(EXTRA_METHODS))
    bars = axes[2].bar(x2 - 0.18, OPENRR_PSNR, 0.36, label="OpenRR val", color="#4c78a8")
    _annotate_bars(axes[2], bars, dy=0.03)
    ax2 = axes[2].twinx()
    x3 = np.arange(len(SIX_SET_METHODS))
    line = ax2.plot(
        x3 + 0.18,
        SIX_SET_MEAN_PSNR,
        marker="o",
        color="#f58518",
        label="Six-set mean",
        linewidth=1.8,
    )
    axes[2].set_title("Extra-data adaptation")
    axes[2].set_ylabel("OpenRR PSNR (dB)")
    ax2.set_ylabel("Six-set mean PSNR (dB)")
    axes[2].set_xticks(x2)
    axes[2].set_xticklabels(EXTRA_METHODS, rotation=25, ha="right")
    axes[2].grid(axis="y", alpha=0.25)
    axes[2].legend([bars], ["OpenRR val"], frameon=False, loc="upper left")
    ax2.legend(line, ["Six-set mean"], frameon=False, loc="lower right")

    fig.savefig(out_dir / "result_summary.png", dpi=args.dpi, bbox_inches="tight")
    print(out_dir / "result_summary.png")


if __name__ == "__main__":
    main()
