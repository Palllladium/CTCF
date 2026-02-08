import json
import os
import matplotlib.pyplot as plt

BASE = r"C:\Users\user\Documents\Education\MasterWork\results\infer"

runs = {
    "CTCF":     os.path.join(BASE, "CTCF", "summary.json"),
    "UTSRMorph":os.path.join(BASE, "UTSRMorph", "summary.json"),
    "TM-DCA":   os.path.join(BASE, "TM-DCA", "summary.json"),
}

def load_metric(path, key):
    with open(path, "r", encoding="utf-8") as f:
        s = json.load(f)
    if key not in s["metrics"]:
        raise KeyError(f"Metric '{key}' not found in {path}. Did you run inference with the needed flags?")
    return s["metrics"][key]["mean"], s["metrics"][key]["std"]

methods = list(runs.keys())

def bar(metric_key, ylabel, title, out_png, use_errorbars=True):
    means = []
    stds = []
    for m in methods:
        mean, std = load_metric(runs[m], metric_key)
        means.append(mean)
        stds.append(std)

    plt.figure()
    if use_errorbars:
        plt.bar(methods, means, yerr=stds)
    else:
        plt.bar(methods, means)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(os.path.join(BASE, out_png), dpi=250)
    plt.close()

if __name__ == "__main__":
    # Dice
    bar("dice_mean", "Dice", "Dice on OASIS Test (mean ± std)", "bar_dice.png", use_errorbars=True)

    # Fold%
    bar("fold_percent", "Fold (%)", "Folding Percentage on OASIS Test (mean ± std)", "bar_fold.png", use_errorbars=True)

    # HD95 (только если ты запускал инференс с --hd95)
    try:
        bar("hd95_mean", "HD95", "HD95 on OASIS Test (mean ± std)", "bar_hd95.png", use_errorbars=True)
    except KeyError:
        print("[WARN] hd95_mean not found in summaries. Run inference with --hd95 if you need HD95 bar chart.")

    print("[OK] Saved charts into:", BASE)
