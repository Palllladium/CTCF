"""
Per-region Dice comparison across models.

Produces: grouped bar chart showing Dice per anatomical region,
plus a diff chart showing per-region advantage of the first model.

Usage:
    python -m tools.paper.plot_per_region_dice \
        --csvs results/infer/OASIS/ctcf/best/per_case.csv "CTCF" \
               results/infer/OASIS/tm-dca/best.pth/per_case.csv "TM-DCA" \
               results/infer/OASIS/utsrmorph/best.pth/per_case.csv "UTSRMorph" \
        --out figures/per_region_oasis.png
"""

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


COLORS = ["#2176AE", "#E84855", "#57A773", "#F4A261", "#8B5CF6"]

# FreeSurfer subcortical label names used across OASIS (35 regions) and IXI (30 regions).
# IXI uses a non-contiguous subset (e.g. 1,2,3,5,...,34,36) of the same atlas.
REGION_NAMES = {
    1: "L-Cerebral-WM",
    2: "L-Cerebral-Cortex",
    3: "L-Lateral-Ventricle",
    4: "L-Inf-Lat-Vent",
    5: "L-Cerebellum-WM",
    6: "L-Cerebellum-Cortex",
    7: "L-Thalamus",
    8: "L-Caudate",
    9: "L-Putamen",
    10: "L-Pallidum",
    11: "3rd-Ventricle",
    12: "4th-Ventricle",
    13: "Brain-Stem",
    14: "L-Hippocampus",
    15: "L-Amygdala",
    16: "L-Accumbens",
    17: "L-VentralDC",
    18: "L-vessel",
    19: "L-choroid-plexus",
    20: "R-Cerebral-WM",
    21: "R-Cerebral-Cortex",
    22: "R-Lateral-Ventricle",
    23: "R-Inf-Lat-Vent",
    24: "R-Cerebellum-WM",
    25: "R-Cerebellum-Cortex",
    26: "R-Thalamus",
    27: "R-Caudate",
    28: "R-Putamen",
    29: "R-Pallidum",
    30: "R-Hippocampus",
    31: "R-Amygdala",
    32: "R-Accumbens",
    33: "R-VentralDC",
    34: "R-vessel",
    35: "R-choroid-plexus",
    36: "R-WM-hypointensities",
}

# Keep old name as alias for backward compatibility
OASIS_REGION_NAMES = REGION_NAMES
_BASE_FONTSIZE = 16


def setup_style():
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": _BASE_FONTSIZE,
            "axes.linewidth": 0.8,
            "axes.grid": True,
            "grid.alpha": 0.3,
            "grid.linewidth": 0.5,
            "figure.dpi": 600,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.05,
        }
    )


def load_per_region(csv_path: str):
    """Load per-region Dice means from per_case.csv.

    Returns: dict {region_idx: mean_dice}
    """
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    # find dice_lbl_* columns
    lbl_cols = sorted([c for c in rows[0].keys() if c.startswith("dice_lbl_")])
    region_means = {}
    for col in lbl_cols:
        idx = int(col.replace("dice_lbl_", ""))
        vals = [float(row[col]) for row in rows]
        region_means[idx] = np.mean(vals)
    return region_means


def main():
    parser = argparse.ArgumentParser(description="Per-region Dice comparison")
    parser.add_argument("--csvs", nargs="+", required=True, help="Pairs: <per_case.csv> <label>")
    parser.add_argument("--out", default="figures/per_region_dice.png")
    parser.add_argument("--top_n", type=int, default=0, help="Show only top N regions with largest variance (0=all)")
    parser.add_argument("--sort_by", default="variance", choices=["variance", "mean", "index"], help="Sort regions by")
    parser.add_argument("--ymin", type=float, default=None, help="Y-axis minimum")
    args = parser.parse_args()

    if len(args.csvs) % 2 != 0:
        parser.error("--csvs requires pairs")
    pairs = [(args.csvs[i], args.csvs[i + 1]) for i in range(0, len(args.csvs), 2)]

    # load per-region data
    model_regions = {}
    for csv_path, label in pairs:
        model_regions[label] = load_per_region(csv_path)

    # common regions
    all_regions = set()
    for regions in model_regions.values():
        all_regions.update(regions.keys())
    common_regions = sorted(all_regions)

    # compute inter-model variance per region for sorting
    region_variance = {}
    region_mean = {}
    for r in common_regions:
        vals = [model_regions[label].get(r, 0) for label in model_regions]
        region_variance[r] = np.var(vals)
        region_mean[r] = np.mean(vals)

    # sort
    if args.sort_by == "variance":
        common_regions.sort(key=lambda r: region_variance[r], reverse=True)
    elif args.sort_by == "mean":
        common_regions.sort(key=lambda r: region_mean[r])
    # else keep index order

    if args.top_n > 0:
        common_regions = common_regions[: args.top_n]

    # ── grouped bar chart ──
    setup_style()
    n_regions = len(common_regions)
    n_models = len(pairs)

    fig, ax = plt.subplots(figsize=(max(10, n_regions * 0.5), 5))

    x = np.arange(n_regions)
    width = 0.8 / n_models

    for j, (_, label) in enumerate(pairs):
        vals = [model_regions[label].get(r, 0) for r in common_regions]
        offset = (j - n_models / 2 + 0.5) * width
        ax.bar(x + offset, vals, width, label=label, color=COLORS[j % len(COLORS)], alpha=0.85)

    # region names on x-axis — use actual label numbers from CSV headers
    region_labels = []
    for r in common_regions:
        name = REGION_NAMES.get(r, f"Region {r}")
        # shorten long names for readability
        name = name.replace("Cerebral-", "Cer-").replace("Cerebellum-", "Cbl-")
        name = name.replace("Lateral-", "Lat-").replace("Ventricle", "Vent")
        name = name.replace("hypointensities", "hypo")
        region_labels.append(name)

    tick_fontsize = int(_BASE_FONTSIZE * 0.75)
    ax.set_xticks(x)
    ax.set_xticklabels(region_labels, rotation=90, ha="center", fontsize=tick_fontsize)
    ax.set_ylabel("Mean Dice")
    ax.legend(loc="lower left")
    if args.ymin is not None:
        ax.set_ylim(bottom=args.ymin)

    fig.tight_layout()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
    print(f"[OK] {out}")

    # ── Diff plot: side-by-side grouped bars ──
    if n_models >= 2:
        n_series = n_models - 1
        fig2, ax2 = plt.subplots(figsize=(max(10, n_regions * 0.5), 4))
        first_label = pairs[0][1]
        diff_width = 0.8 / n_series

        for j in range(1, n_models):
            other_label = pairs[j][1]
            diff = []
            for r in common_regions:
                d = model_regions[first_label].get(r, 0) - model_regions[other_label].get(r, 0)
                diff.append(d)
            offset = (j - 1 - n_series / 2 + 0.5) * diff_width
            bars = ax2.bar(
                x + offset,
                diff,
                diff_width,
                label=f"{first_label} \u2212 {other_label}",
                color=COLORS[j % len(COLORS)],
                alpha=0.85,
            )

        ax2.axhline(y=0, color="black", linewidth=0.5)
        ax2.set_xticks(x)
        ax2.set_xticklabels(region_labels, rotation=90, ha="center", fontsize=tick_fontsize)
        ax2.set_ylabel("Dice Difference")
        ax2.legend(loc="best")
        fig2.tight_layout()

        out2 = out.with_stem(out.stem + "_diff")
        fig2.savefig(out2)
        plt.close(fig2)
        print(f"[OK] {out2}")


if __name__ == "__main__":
    main()
