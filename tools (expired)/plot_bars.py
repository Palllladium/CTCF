import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def load_summary(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def collect_runs(base_dir: Path):
    out = {}
    for p in sorted(base_dir.glob("*/summary.json")):
        out[p.parent.name] = p
    if not out:
        raise RuntimeError(f"No summary.json found under {base_dir}/*/summary.json")
    return out


def metric_values(runs, metric):
    means, stds, names = [], [], []
    for name, path in runs.items():
        s = load_summary(path)
        if metric not in s.get("metrics", {}):
            continue
        m = s["metrics"][metric]
        means.append(float(m.get("mean", 0.0)))
        stds.append(float(m.get("std", 0.0)))
        names.append(name)
    if not names:
        raise RuntimeError(f"Metric '{metric}' not found in any summary.json")
    return names, means, stds


def draw(runs, base_dir: Path, metric: str, ylabel: str, out_name: str, with_err: bool):
    names, means, stds = metric_values(runs, metric)
    plt.figure()
    if with_err:
        plt.bar(names, means, yerr=stds)
    else:
        plt.bar(names, means)
    plt.ylabel(ylabel)
    plt.title(metric)
    plt.tight_layout()
    out = base_dir / out_name
    plt.savefig(out, dpi=250)
    plt.close()
    print(f"[OK] {out}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base_dir", required=True, help="Directory containing run folders with summary.json")
    p.add_argument("--with_err", type=int, default=1, help="1 to draw std bars, 0 otherwise")
    args = p.parse_args()

    base_dir = Path(args.base_dir)
    runs = collect_runs(base_dir)
    with_err = bool(int(args.with_err))
    draw(runs, base_dir, "dice_mean", "Dice", "bar_dice.png", with_err)
    draw(runs, base_dir, "fold_percent", "Fold (%)", "bar_fold.png", with_err)
    try:
        draw(runs, base_dir, "hd95_mean", "HD95", "bar_hd95.png", with_err)
    except RuntimeError:
        print("[WARN] hd95_mean not found; skip bar_hd95.png")


if __name__ == "__main__":
    main()