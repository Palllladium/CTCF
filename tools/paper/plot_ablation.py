"""
Ablation study visualization for the OASIS paper figure.

Reads ablation_N_results.txt summaries and per-experiment logfiles to build
a multi-panel ablation figure.
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

from tools.paper.log_parser import parse_ablation_summary, parse_log


LABEL_MAP = {
    "ABL_01_BASELINE": "R1: Baseline\n(default CTCF)",
    "ABL_03_ICONL2": "R1: ICON L2",
    "ABL2_02_L3_NCC_CH64_TS6": "R2: L3=64,\nNCC, TS6",
    "ABL2_04_DROP01_QKVT": "R2: Swin tuning\nonly",
    "ABL3_01_L1CH32_L3CH64_TS6": "R3: L1=32,\nL3=64, TS6",
    "ABL3_02_L1CH64_L3CH64_TS6": "R3: L1=64,\nL3=64, TS6",
    "ABL4_01_L2_ONLY": "R4: L2 only",
    "ABL4_02_L1_L2": "R4: L1+L2",
    "ABL4_03_L2_L3": "R4: L2+L3",
}

PAPER_RUNS = [
    "ABL_01_BASELINE",
    "ABL_03_ICONL2",
    "ABL2_02_L3_NCC_CH64_TS6",
    "ABL2_04_DROP01_QKVT",
    "ABL3_01_L1CH32_L3CH64_TS6",
    "ABL3_02_L1CH64_L3CH64_TS6",
    "ABL4_01_L2_ONLY",
    "ABL4_02_L1_L2",
    "ABL4_03_L2_L3",
]

ROUND_COLORS = {
    "R1": "#A8DADC",
    "R2": "#457B9D",
    "R3": "#1D3557",
    "R4": "#9C7A5B",
}


def setup_style():
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 10,
            "axes.linewidth": 0.8,
            "axes.grid": True,
            "grid.alpha": 0.3,
            "grid.linewidth": 0.5,
            "figure.dpi": 600,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.05,
        }
    )


def get_extra_metrics(log_dir: Path, name: str):
    logfile = log_dir / name / "logfile.log"
    if not logfile.exists():
        return None, None
    log = parse_log(logfile)
    if not log.epochs:
        return None, None
    last = log.epochs[-1]
    return last.sdlogj, last.fold_pct


def pick_round_color(name: str) -> str:
    if name.startswith("ABL4_"):
        return ROUND_COLORS["R4"]
    if name.startswith("ABL3_"):
        return ROUND_COLORS["R3"]
    if name.startswith("ABL2_"):
        return ROUND_COLORS["R2"]
    if name.startswith("ABL_"):
        return ROUND_COLORS["R1"]
    return "#888888"


def main():
    parser = argparse.ArgumentParser(description="Ablation study visualization")
    parser.add_argument("--summaries", nargs="+", required=True, help="ablation_N_results.txt files")
    parser.add_argument("--log_dir", default="logs", help="Directory containing per-experiment log subdirs")
    parser.add_argument("--out", default="paper/figures/ablation_visual.png")
    parser.add_argument("--paper", action="store_true", help="Show only the subset used in the paper")
    parser.add_argument("--oasis_only", action="store_true", help="Exclude IXI experiments")
    parser.add_argument("--dice_ymin", type=float, default=0.79, help="Y-axis minimum for Dice")
    parser.add_argument("--dice_ymax", type=float, default=0.83, help="Y-axis maximum for Dice")
    parser.add_argument("--sdlogj_ymin", type=float, default=0.05, help="Y-axis minimum for SDlogJ")
    parser.add_argument("--sdlogj_ymax", type=float, default=0.088, help="Y-axis maximum for SDlogJ")
    parser.add_argument("--fold_ymax", type=float, default=0.40, help="Y-axis maximum for Fold %")
    args = parser.parse_args()

    log_dir = Path(args.log_dir)
    all_runs = []
    for summary_file in args.summaries:
        all_runs.extend(parse_ablation_summary(summary_file))

    if args.oasis_only:
        all_runs = [r for r in all_runs if "IXI" not in r["name"].upper()]

    if args.paper:
        order = {name: idx for idx, name in enumerate(PAPER_RUNS)}
        all_runs = [r for r in all_runs if r["name"] in order]
        all_runs.sort(key=lambda r: order[r["name"]])

    all_runs = [r for r in all_runs if r["best_dice"] > 0.01]

    for run in all_runs:
        sdlogj, fold_pct = get_extra_metrics(log_dir, run["name"])
        run["sdlogj"] = sdlogj
        run["fold_pct"] = fold_pct

    names = [LABEL_MAP.get(r["name"], r["name"]) for r in all_runs]
    best_dice = [r["best_dice"] for r in all_runs]
    sdlogj_vals = [0.0 if r["sdlogj"] is None else r["sdlogj"] for r in all_runs]
    fold_vals = [0.0 if r["fold_pct"] is None else r["fold_pct"] for r in all_runs]
    bar_colors = [pick_round_color(r["name"]) for r in all_runs]

    setup_style()
    fig, axes = plt.subplots(3, 1, figsize=(max(10, len(all_runs) * 0.95), 7), sharex=True)
    x = np.arange(len(all_runs))

    ax = axes[0]
    bars = ax.bar(x, best_dice, color=bar_colors, edgecolor="white", linewidth=0.5)
    ax.set_ylabel("Best Dice")
    ax.set_title("Ablation Study: Best Validation Dice (100 epochs)")
    ax.set_ylim(bottom=args.dice_ymin, top=args.dice_ymax)
    for bar, value in zip(bars, best_dice):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.0003,
            f"{value:.4f}",
            ha="center",
            va="bottom",
            fontsize=7,
            rotation=0,
        )

    best_idx = int(np.argmax(best_dice))
    bars[best_idx].set_edgecolor("#E84855")
    bars[best_idx].set_linewidth(2)

    baseline_dice = next((r["best_dice"] for r in all_runs if "BASELINE" in r["name"]), None)
    if baseline_dice is not None:
        ax.axhline(y=baseline_dice, color="#E84855", linestyle="--", linewidth=0.8, alpha=0.7)
        y_lo, y_hi = ax.get_ylim()
        ax.text(
            -0.3,
            baseline_dice - max((y_hi - y_lo) * 0.012, 0.0006),
            "baseline",
            fontsize=8,
            color="black",
            fontweight="bold",
            ha="left",
            va="top",
        )

    ax.legend(handles=[Patch(facecolor=color, label=label) for label, color in ROUND_COLORS.items()], loc="lower right")

    ax2 = axes[1]
    ax2.bar(x, sdlogj_vals, color=bar_colors, edgecolor="white", linewidth=0.5)
    ax2.set_ylabel("SDlogJ")
    ax2.set_title("Deformation Regularity (lower = smoother)")
    ax2.set_ylim(bottom=args.sdlogj_ymin, top=args.sdlogj_ymax)
    for idx, value in enumerate(sdlogj_vals):
        if value > 0:
            ax2.text(idx, value + 0.001, f"{value:.4f}", ha="center", va="bottom", fontsize=7, rotation=0)

    ax3 = axes[2]
    ax3.bar(x, fold_vals, color=bar_colors, edgecolor="white", linewidth=0.5)
    ax3.set_ylabel("Fold %")
    ax3.set_title("Folding Percentage (lower = better topology)")
    ax3.set_ylim(top=args.fold_ymax)
    for idx, value in enumerate(fold_vals):
        if value > 0:
            ax3.text(idx, value + 0.002, f"{value:.2f}", ha="center", va="bottom", fontsize=7, rotation=0)

    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(names, rotation=0, ha="center", fontsize=7)

    fig.tight_layout()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
    print(f"[OK] {out}")


if __name__ == "__main__":
    main()
