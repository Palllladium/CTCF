"""
Round 5 resolution-scaling ablation chart.
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


R5_RUNS = [
    ("ABL3_01_L1CH32_L3CH64_TS6", "R3: Reference", "#8E8E8E"),
    ("GEN2_01_ITER_L3_N2", "R5: L3 iter x2", "#6E8B8B"),
    ("GEN2_03_LEARNED_UP", "R5: Learned upsample", "#6B7A8F"),
    ("GEN2_04_L2_L3_SKIP", "R5: L2->L3 skip", "#B5A277"),
    ("GEN2_06_L3_ZONE", "R5: L3 zone combined", "#A98574"),
]


def parse_best_epoch_metrics(logfile: Path):
    best_dice = 0.0
    best_sdlogj = None
    best_fold = None
    import re

    with logfile.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if "[epoch" not in line:
                continue
            m_dice = re.search(r"val_dice=([\d.]+)", line)
            m_best = re.search(r"best=([\d.]+)", line)
            m_sdlogj = re.search(r"sdlogj=([\d.]+)", line, re.IGNORECASE)
            m_fold = re.search(r"j<=0%=([\d.]+)", line)
            if not (m_dice and m_best):
                continue
            val_dice = float(m_dice.group(1))
            best_value = float(m_best.group(1))
            if best_value > best_dice or (val_dice == best_value and best_value >= best_dice):
                best_dice = best_value
                best_sdlogj = None if m_sdlogj is None else float(m_sdlogj.group(1))
                best_fold = None if m_fold is None else float(m_fold.group(1))
    return best_dice, best_sdlogj, best_fold


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_dir", default="logs")
    parser.add_argument("--out", default="paper/figures/ablation_r5.png")
    parser.add_argument("--dice_ymin", type=float, default=0.810)
    parser.add_argument("--dice_ymax", type=float, default=0.840)
    parser.add_argument("--sdlogj_ymin", type=float, default=0.0)
    parser.add_argument("--sdlogj_ymax", type=float, default=0.1)
    parser.add_argument("--fold_ymin", type=float, default=0.0)
    parser.add_argument("--fold_ymax", type=float, default=0.55)
    args = parser.parse_args()

    log_dir = Path(args.log_dir)
    labels, dices, sdlogjs, folds, colors = [], [], [], [], []
    for dirname, label, color in R5_RUNS:
        logfile = log_dir / dirname / "logfile.log"
        if not logfile.exists():
            print(f"[WARN] {logfile} not found, skipping")
            continue
        dice, sdlogj, fold = parse_best_epoch_metrics(logfile)
        labels.append(label)
        dices.append(dice)
        sdlogjs.append(0.0 if sdlogj is None else sdlogj)
        folds.append(0.0 if fold is None else fold)
        colors.append(color)

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 9,
            "axes.linewidth": 0.8,
            "axes.grid": True,
            "grid.alpha": 0.3,
            "grid.linewidth": 0.5,
            "figure.dpi": 600,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.05,
        }
    )

    x = np.arange(len(labels))
    fig, axes = plt.subplots(3, 1, figsize=(7, 6), sharex=True)

    ax = axes[0]
    bars = ax.bar(x, dices, color=colors, edgecolor="white", linewidth=0.5)
    ax.set_ylabel("Best Dice")
    ax.set_title("Round 5: Resolution Scaling (100 epochs, OASIS)")
    ax.set_ylim(bottom=args.dice_ymin, top=args.dice_ymax)
    for bar, value in zip(bars, dices):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.0003, f"{value:.4f}", ha="center", va="bottom", fontsize=8)
    if dices:
        ax.axhline(y=dices[0], color="#E84855", linestyle="--", linewidth=0.8, alpha=0.75)
        ax.text(
            len(labels) - 0.8, dices[0] + 0.0003, "R3 ref", fontsize=7, color="black", fontweight="bold", ha="right"
        )
    best_idx = int(np.argmax(dices))
    bars[best_idx].set_edgecolor("#E84855")
    bars[best_idx].set_linewidth(2)

    ax2 = axes[1]
    ax2.bar(x, sdlogjs, color=colors, edgecolor="white", linewidth=0.5)
    ax2.set_ylabel("SDlogJ")
    ax2.set_title("Deformation Regularity (lower = smoother)")
    ax2.set_ylim(bottom=args.sdlogj_ymin, top=args.sdlogj_ymax)
    for idx, value in enumerate(sdlogjs):
        if value > 0:
            ax2.text(idx, value + 0.001, f"{value:.4f}", ha="center", va="bottom", fontsize=8)

    ax3 = axes[2]
    ax3.bar(x, folds, color=colors, edgecolor="white", linewidth=0.5)
    ax3.set_ylabel("Fold %")
    ax3.set_title("Folding Percentage (lower = better topology)")
    ax3.set_ylim(bottom=args.fold_ymin, top=args.fold_ymax)
    for idx, value in enumerate(folds):
        if value > 0:
            ax3.text(idx, value + 0.005, f"{value:.2f}%", ha="center", va="bottom", fontsize=8)

    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(labels, rotation=0, ha="center", fontsize=7)

    fig.tight_layout()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
    print(f"[OK] {out}")


if __name__ == "__main__":
    main()
