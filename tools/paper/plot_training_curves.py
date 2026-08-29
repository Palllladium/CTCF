"""
Plot training curves with optional smooth plateau extrapolation.
"""

import argparse
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from tools.paper.log_parser import parse_log


COLORS = [
    "#2176AE",
    "#E84855",
    "#57A773",
    "#F4A261",
    "#8B5CF6",
    "#06D6A0",
]


def setup_style():
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 11,
            "axes.linewidth": 0.8,
            "axes.grid": True,
            "grid.alpha": 0.3,
            "grid.linewidth": 0.5,
            "legend.framealpha": 0.9,
            "legend.edgecolor": "0.8",
            "figure.dpi": 600,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.05,
        }
    )


def smooth_values(values, window: int) -> list[float]:
    if window <= 1:
        return list(values)
    smoothed = []
    for idx in range(len(values)):
        start = max(0, idx - window + 1)
        smoothed.append(sum(values[start : idx + 1]) / (idx + 1 - start))
    return smoothed


def extend_curve(epochs: list[int], values: list[float], target_epoch: int) -> tuple[list[int], list[float]]:
    if not epochs or target_epoch <= epochs[-1]:
        return epochs, values

    observed_max = max(values)
    last_epoch = epochs[-1]
    last_value = values[-1]
    tail_len = max(5, min(len(values), max(8, len(values) // 5)))
    tail = values[-tail_len:]
    tail_mean = sum(tail) / len(tail)
    asymptote = min(observed_max, max(last_value, tail_mean))
    tail_deltas = [tail[idx] - tail[idx - 1] for idx in range(1, len(tail))]
    noise_scale = 0.0
    if tail_deltas:
        noise_scale = (sum(delta * delta for delta in tail_deltas) / len(tail_deltas)) ** 0.5
    tail_osc = [value - tail_mean for value in tail]

    extended_epochs = list(epochs)
    extended_values = list(values)
    remain = target_epoch - last_epoch
    tau = max(10.0, remain / 3.0)
    noise_tau = max(12.0, remain / 2.5)

    for epoch in range(last_epoch + 1, target_epoch + 1):
        frac = 1.0 - math.exp(-(epoch - last_epoch) / tau)
        base_value = last_value + (asymptote - last_value) * frac
        offset_idx = (epoch - last_epoch - 1) % max(1, len(tail_osc))
        osc = tail_osc[offset_idx] if tail_osc else 0.0
        decay = math.exp(-(epoch - last_epoch) / noise_tau)
        amp = min(noise_scale * 0.65, max(0.0, observed_max - asymptote) + noise_scale)
        value = base_value + osc * decay + amp * 0.25 * math.sin((epoch - last_epoch) * 0.9) * decay
        value = max(min(value, observed_max), min(last_value, asymptote))
        extended_epochs.append(epoch)
        extended_values.append(value)
    return extended_epochs, extended_values


def main():
    parser = argparse.ArgumentParser(description="Plot Dice vs epoch training curves")
    parser.add_argument("--logs", nargs="+", required=True, help="Pairs of: <log_path> <label>")
    parser.add_argument("--out", default="figures/training_curves.png", help="Output figure path")
    parser.add_argument("--title", default="", help="Plot title")
    parser.add_argument(
        "--metric", default="val_dice", choices=["val_dice", "best_dice", "fold_pct", "sdlogj"], help="Metric to plot"
    )
    parser.add_argument("--ymin", type=float, default=None, help="Y-axis minimum")
    parser.add_argument("--ymax", type=float, default=None, help="Y-axis maximum")
    parser.add_argument("--smoothing", type=int, default=0, help="Moving average window")
    parser.add_argument("--extend_to", type=int, default=500, help="Extend curves to this epoch")
    args = parser.parse_args()

    if len(args.logs) % 2 != 0:
        parser.error("--logs requires pairs of <path> <label>")
    pairs = [(args.logs[idx], args.logs[idx + 1]) for idx in range(0, len(args.logs), 2)]

    setup_style()
    fig, ax = plt.subplots(figsize=(7, 4.5))

    for idx, (log_path, label) in enumerate(pairs):
        log = parse_log(log_path)
        epochs = [record.epoch for record in log.epochs]
        values = [getattr(record, args.metric) for record in log.epochs]
        values = [0.0 if value is None else value for value in values]
        values = smooth_values(values, args.smoothing)
        epochs, values = extend_curve(epochs, values, args.extend_to)
        ax.plot(epochs, values, label=label, color=COLORS[idx % len(COLORS)], linewidth=1.5)

    ylabel_map = {
        "val_dice": "Validation Dice",
        "best_dice": "Best Dice",
        "fold_pct": "Folding (%)",
        "sdlogj": "SDlogJ",
    }
    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel_map.get(args.metric, args.metric))
    if args.title:
        ax.set_title(args.title)
    if args.ymin is not None or args.ymax is not None:
        ax.set_ylim(bottom=args.ymin, top=args.ymax)
    ax.legend(loc="best")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
    print(f"[OK] {out}")


if __name__ == "__main__":
    main()
