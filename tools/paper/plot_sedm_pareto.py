"""
SEDM Pareto-style scatter plot.

The figure summarizes cascade configurations without qualitative examples:
  x-axis       number of trainable parameters, M (lower is better)
  y-axis       mean Dice (higher is better)
  color        base registration network
  fill style   L3-SVF ON/OFF, unless hidden for presentation figures
  bubble size  peak GPU memory from logs/<exp_name>/logfile.log

Usage:
    python -m tools.paper.plot_sedm_pareto \
        --summary-csv results/SEDM/summary/aggregated.csv \
        --logs-dir logs \
        --out results/SEDM/figures/sedm_pareto.png \
        --pdf

Presentation variant without explicit SVF mention:
    python -m tools.paper.plot_sedm_pareto \
        --svf-mode on \
        --hide-svf-legend \
        --params-mode cascade-total \
        --x-min 5 \
        --x-max 55 \
        --out results/SEDM/figures/sedm_pareto_presentation.png \
        --pdf
"""

from __future__ import annotations

import argparse
import csv
import math
import re
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


MODEL_COLORS = {
    "VoxelMorph": "#4C78A8",
    "LKU-8": "#F58518",
    "LKU-32": "#E45756",
    "MambaMorph": "#54A24B",
    "VMambaMorph": "#B279A2",
}

CASCADE_TOTAL_PARAMS_M = {
    "VoxelMorph": 9.240905,
    "LKU-8": 6.691596,
    "LKU-32": 37.952796,
    "MambaMorph": 11.910713,
    "VMambaMorph": 13.957625,
}

DISPLAY_LABELS = {
    "VoxelMorph": "VoxelMorph",
    "LKU-8": "LKU-8",
    "LKU-32": "LKU-32",
    "MambaMorph": "MambaMorph",
    "VMambaMorph": "VMambaMorph",
}


def setup_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 8.5,
            "axes.linewidth": 0.8,
            "axes.grid": True,
            "grid.alpha": 0.22,
            "grid.linewidth": 0.5,
            "figure.dpi": 450,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.04,
        }
    )


def read_rows(
    path: Path,
    logs_dir: Path,
    *,
    svf_mode: str,
    params_mode: str,
) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = [row for row in csv.DictReader(handle) if row.get("group") == "cascade"]

    if svf_mode != "all":
        rows = [row for row in rows if row.get("svf", "").lower() == svf_mode]

    for row in rows:
        row["params_m_float"] = parse_float(row["params_m"])
        if params_mode == "cascade-total":
            row["params_m_float"] = CASCADE_TOTAL_PARAMS_M[row["backbone"]]
        row["dice_mean_float"] = parse_float(row["dice_mean"])
        row["dice_std_float"] = parse_float(row["dice_std"])
        row["peak_vram_gb"] = peak_vram_from_log(logs_dir / row["exp_name"] / "logfile.log")

    return rows


def parse_float(value: str) -> float:
    if value in {"", "-", "—", None}:
        return float("nan")
    return float(value)


def peak_vram_from_log(log_path: Path) -> float:
    if not log_path.exists():
        return float("nan")
    text = log_path.read_text(encoding="utf-8", errors="ignore")
    peaks = [float(match.group(1)) for match in re.finditer(r"peak=([0-9.]+)GB", text)]
    return max(peaks) if peaks else float("nan")


def point_size(vram_gb: float, min_vram: float, max_vram: float) -> float:
    if math.isnan(vram_gb):
        return 90.0
    if max_vram <= min_vram:
        return 120.0
    t = (vram_gb - min_vram) / (max_vram - min_vram)
    return 60.0 + 190.0 * math.sqrt(max(0.0, min(1.0, t)))


def label_for(row: dict) -> str:
    label = DISPLAY_LABELS.get(row["backbone"], row["backbone"])
    if row["backbone"] == "LKU-8" and "FIXSCHED" in row["exp_name"]:
        label = "LKU-8*"
    if row.get("svf") == "OFF" and row["backbone"] in {"LKU-8", "MambaMorph"}:
        label += " noSVF"
    return label


def jitter_x(rows: list[dict]) -> dict[str, float]:
    """Small deterministic x-jitter for points with identical x within a panel."""
    groups: dict[tuple[str, float], list[dict]] = defaultdict(list)
    for row in rows:
        groups[(row["ds"], row["params_m_float"])].append(row)

    out = {}
    for _, group in groups.items():
        group = sorted(group, key=lambda r: (r["backbone"], r["svf"], r["exp_name"]))
        n = len(group)
        for idx, row in enumerate(group):
            offset = idx - (n - 1) / 2
            out[row["exp_name"]] = row["params_m_float"] * (1.0 + 0.10 * offset)
    return out


def plot_dataset(
    ax,
    rows: list[dict],
    ds: str,
    x_positions: dict[str, float],
    min_vram: float,
    max_vram: float,
    *,
    annotate: bool,
    error_bars: bool,
    x_min: float,
    x_max: float,
) -> None:
    ds_rows = [row for row in rows if row["ds"] == ds]

    for row in ds_rows:
        x = x_positions[row["exp_name"]]
        y = row["dice_mean_float"]
        color = MODEL_COLORS.get(row["backbone"], "#666666")
        filled = row.get("svf") == "ON"

        if error_bars and not math.isnan(row["dice_std_float"]):
            ax.errorbar(
                x,
                y,
                yerr=row["dice_std_float"],
                color=color,
                alpha=0.25,
                linewidth=0.8,
                capsize=1.5,
                zorder=1,
            )

        ax.scatter(
            x,
            y,
            s=point_size(row["peak_vram_gb"], min_vram, max_vram),
            marker="o",
            facecolors=color if filled else "white",
            edgecolors=color,
            linewidth=1.4 if not filled else 0.9,
            alpha=0.82,
            zorder=3,
        )

        if annotate:
            ax.annotate(
                label_for(row),
                (x, y),
                xytext=(4, 4),
                textcoords="offset points",
                fontsize=6.5,
                color="#222222",
                clip_on=True,
            )

    y_values = [row["dice_mean_float"] for row in ds_rows]
    y_min, y_max = min(y_values), max(y_values)
    y_pad = max(0.0015, (y_max - y_min) * 0.18)

    ax.set_xscale("log")
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min - y_pad, y_max + y_pad)
    ax.set_title(ds)
    ax.set_xlabel("Parameters, M (log scale)")
    ax.set_ylabel("Dice")


def add_legend(fig, min_vram: float, max_vram: float, *, include_svf: bool) -> None:
    model_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=color,
            markeredgecolor=color,
            markersize=6.5,
            label=model,
        )
        for model, color in MODEL_COLORS.items()
    ]

    svf_handles = (
        [
            Line2D(
                [0],
                [0],
                marker="o",
                color="none",
                markerfacecolor="#777777",
                markeredgecolor="#777777",
                markersize=6.5,
                label="L3-SVF ON",
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                color="none",
                markerfacecolor="white",
                markeredgecolor="#777777",
                markeredgewidth=1.4,
                markersize=6.5,
                label="L3-SVF OFF",
            ),
        ]
        if include_svf
        else []
    )

    size_values = nice_size_values(min_vram, max_vram)
    size_handles = [
        plt.scatter(
            [],
            [],
            s=point_size(v, min_vram, max_vram),
            facecolors="#BDBDBD",
            edgecolors="#777777",
            linewidth=0.8,
            alpha=0.82,
            label=f"{v:g} GB",
        )
        for v in size_values
    ]

    handles = model_handles + svf_handles + size_handles
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=6,
        bbox_to_anchor=(0.5, -0.055),
        frameon=False,
        fontsize=6.8,
        handletextpad=0.45,
        columnspacing=0.95,
    )


def nice_size_values(min_vram: float, max_vram: float) -> list[float]:
    candidates = [10, 20, 30, 40]
    values = [v for v in candidates if min_vram <= v <= max_vram]
    if len(values) >= 2:
        return values
    return [round(min_vram, 1), round((min_vram + max_vram) / 2, 1), round(max_vram, 1)]


def write_vram_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["exp_name", "dataset", "backbone", "svf", "params_m", "dice_mean", "peak_vram_gb"])
        for row in sorted(rows, key=lambda r: (r["ds"], r["params_m_float"], r["backbone"], r["svf"])):
            writer.writerow(
                [
                    row["exp_name"],
                    row["ds"],
                    row["backbone"],
                    row["svf"],
                    row["params_m"],
                    row["dice_mean"],
                    f"{row['peak_vram_gb']:.2f}" if not math.isnan(row["peak_vram_gb"]) else "",
                ]
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Create SEDM Pareto-style scatter plot")
    parser.add_argument("--summary-csv", default="results/SEDM/summary/aggregated.csv")
    parser.add_argument("--logs-dir", default="logs")
    parser.add_argument("--out", default="results/SEDM/figures/sedm_pareto.png")
    parser.add_argument("--pdf", action="store_true", help="Also write a PDF next to the output file.")
    parser.add_argument("--annotate", action="store_true", help="Annotate points with compact model labels.")
    parser.add_argument(
        "--error-bars", action="store_true", help="Draw Dice std bars. Off by default to avoid clutter."
    )
    parser.add_argument(
        "--svf-mode", choices=["all", "on", "off"], default="all", help="Filter cascade rows by SVF state."
    )
    parser.add_argument("--hide-svf-legend", action="store_true", help="Hide filled/hollow SVF legend entries.")
    parser.add_argument(
        "--params-mode",
        choices=["backbone", "cascade-total"],
        default="backbone",
        help="Use backbone params from the summary CSV or exact full cascade params.",
    )
    parser.add_argument("--x-min", type=float, default=0.23)
    parser.add_argument("--x-max", type=float, default=52.0)
    args = parser.parse_args()

    rows = read_rows(
        Path(args.summary_csv),
        Path(args.logs_dir),
        svf_mode=args.svf_mode,
        params_mode=args.params_mode,
    )
    if not rows:
        raise SystemExit(f"No cascade rows found in {args.summary_csv}")

    vrams = [row["peak_vram_gb"] for row in rows if not math.isnan(row["peak_vram_gb"])]
    min_vram, max_vram = min(vrams), max(vrams)
    x_positions = jitter_x(rows)

    setup_style()
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2), sharex=True)
    plot_dataset(
        axes[0],
        rows,
        "OASIS",
        x_positions,
        min_vram,
        max_vram,
        annotate=args.annotate,
        error_bars=args.error_bars,
        x_min=args.x_min,
        x_max=args.x_max,
    )
    plot_dataset(
        axes[1],
        rows,
        "IXI",
        x_positions,
        min_vram,
        max_vram,
        annotate=args.annotate,
        error_bars=args.error_bars,
        x_min=args.x_min,
        x_max=args.x_max,
    )
    add_legend(fig, min_vram, max_vram, include_svf=not args.hide_svf_legend)
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.27, wspace=0.28)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    print(f"[OK] {out}")

    if args.pdf:
        pdf_out = out.with_suffix(".pdf")
        fig.savefig(pdf_out)
        print(f"[OK] {pdf_out}")

    vram_csv = out.with_name(f"{out.stem}_vram.csv")
    write_vram_csv(rows, vram_csv)
    print(f"[OK] {vram_csv}")

    plt.close(fig)


if __name__ == "__main__":
    main()
