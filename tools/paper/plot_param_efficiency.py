"""
Parameter efficiency scatter plot: Dice vs #params, with inference time as marker size.

Usage:
    python -m tools.paper.plot_param_efficiency \
        --models "CTCF" 289.0 0.8162 1.50 \
                 "TM-DCA" 288.0 0.8145 0.82 \
                 "UTSRMorph" 172.0 0.8172 1.28 \
        --out figures/param_efficiency.pdf

Each model: <label> <params_M> <dice> <time_sec>
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


COLORS = ["#2176AE", "#E84855", "#57A773", "#F4A261", "#8B5CF6"]
MARKERS = ["o", "s", "D", "^", "v"]


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
    parser = argparse.ArgumentParser(description="Parameter efficiency scatter plot")
    parser.add_argument("--models", nargs="+", required=True, help="Groups of 4: <label> <params_M> <dice> <time_sec>")
    parser.add_argument("--out", default="figures/param_efficiency.png")
    args = parser.parse_args()

    if len(args.models) % 4 != 0:
        parser.error("--models requires groups of 4: <label> <params_M> <dice> <time_sec>")

    models = []
    for i in range(0, len(args.models), 4):
        models.append(
            {
                "label": args.models[i],
                "params_m": float(args.models[i + 1]),
                "dice": float(args.models[i + 2]),
                "time_sec": float(args.models[i + 3]),
            }
        )

    setup_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

    # ── Left: Dice vs Params ──
    for i, m in enumerate(models):
        ax1.scatter(
            m["params_m"],
            m["dice"],
            s=200,
            c=COLORS[i % len(COLORS)],
            marker=MARKERS[i % len(MARKERS)],
            label=m["label"],
            zorder=5,
            edgecolors="black",
            linewidth=0.5,
        )
        ax1.annotate(m["label"], (m["params_m"], m["dice"]), textcoords="offset points", xytext=(8, 8), fontsize=9)

    ax1.set_xlabel("Parameters (M)")
    ax1.set_ylabel("Dice")
    ax1.set_title("Accuracy vs Model Size")

    # ── Right: Dice vs Inference Time ──
    for i, m in enumerate(models):
        ax2.scatter(
            m["time_sec"],
            m["dice"],
            s=m["params_m"] * 1.5,  # size proportional to params
            c=COLORS[i % len(COLORS)],
            marker=MARKERS[i % len(MARKERS)],
            label=m["label"],
            zorder=5,
            edgecolors="black",
            linewidth=0.5,
            alpha=0.8,
        )
        ax2.annotate(
            f"{m['label']}\n({m['params_m']:.0f}M)",
            (m["time_sec"], m["dice"]),
            textcoords="offset points",
            xytext=(8, 8),
            fontsize=8,
        )

    ax2.set_xlabel("Inference Time (s)")
    ax2.set_ylabel("Dice")
    ax2.set_title("Accuracy vs Speed (marker size ∝ params)")

    fig.tight_layout()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
    print(f"[OK] {out}")


if __name__ == "__main__":
    main()
