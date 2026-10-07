"""
Box plot of per-case metrics from inference CSVs.
"""

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


COLORS = ["#2176AE", "#E84855", "#57A773", "#F4A261", "#8B5CF6"]


def setup_style():
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 11,
            "axes.linewidth": 0.8,
            "axes.grid": True,
            "grid.alpha": 0.3,
            "grid.linewidth": 0.5,
            "figure.dpi": 600,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.05,
        }
    )


def main():
    parser = argparse.ArgumentParser(description="Box plot of per-case metrics")
    parser.add_argument("--csvs", nargs="+", required=True, help="Pairs: <per_case.csv> <label>")
    parser.add_argument("--out", default="figures/boxplot.png")
    parser.add_argument("--title", default="")
    parser.add_argument("--metric", default="dice_mean", help="Column name from per_case.csv")
    parser.add_argument("--ymin", type=float, default=None, help="Y-axis minimum")
    parser.add_argument("--ymax", type=float, default=None, help="Y-axis maximum")
    parser.add_argument("--box_width", type=float, default=0.5, help="Box width")
    args = parser.parse_args()

    if len(args.csvs) % 2 != 0:
        parser.error("--csvs requires pairs of <path> <label>")
    pairs = [(args.csvs[idx], args.csvs[idx + 1]) for idx in range(0, len(args.csvs), 2)]

    data = []
    labels = []
    for csv_path, label in pairs:
        with open(csv_path, "r", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            data.append([float(row[args.metric]) for row in reader])
        labels.append(label)

    setup_style()
    fig, ax = plt.subplots(figsize=(max(4, len(pairs) * 1.5), 5))
    bp = ax.boxplot(
        data,
        tick_labels=labels,
        patch_artist=True,
        widths=args.box_width,
        showmeans=True,
        meanprops=dict(marker="D", markerfacecolor="white", markeredgecolor="black", markersize=5),
        medianprops=dict(color="black", linewidth=1.5),
        flierprops=dict(marker="o", markersize=4, alpha=0.5),
    )

    for patch, color in zip(bp["boxes"], COLORS[: len(data)]):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    for idx, values in enumerate(data):
        jitter = np.random.normal(0, 0.04, len(values))
        ax.scatter(
            np.full(len(values), idx + 1) + jitter, values, alpha=0.4, s=15, color=COLORS[idx % len(COLORS)], zorder=3
        )

    ax.set_ylabel("Dice" if args.metric == "dice_mean" else args.metric)
    if args.title:
        ax.set_title(args.title)
    if args.ymin is not None or args.ymax is not None:
        ax.set_ylim(bottom=args.ymin, top=args.ymax)

    y_lo, y_hi = ax.get_ylim()
    text_y = y_lo + (y_hi - y_lo) * 0.02
    for idx, values in enumerate(data):
        ax.text(
            idx + 1,
            text_y,
            f"\u03bc={np.mean(values):.4f}\n\u03c3={np.std(values):.4f}",
            ha="center",
            va="bottom",
            fontsize=8,
            style="italic",
        )

    fig.tight_layout()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
    print(f"[OK] {out}")


if __name__ == "__main__":
    main()
