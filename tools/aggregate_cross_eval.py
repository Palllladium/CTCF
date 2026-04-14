import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.stats import norm, wilcoxon


METRICS = ["dice_mean", "hd95_mean", "sdlogj", "j_leq0_percent", "ndv_percent", "time_sec"]

# Which metric is available per dataset (OASIS has sdlogj, IXI has j_leq0_percent/ndv_percent)
PAIRWISE_METRICS_BY_DS = {
    "OASIS": ["dice_mean", "hd95_mean", "sdlogj"],
    "IXI":   ["dice_mean", "hd95_mean", "j_leq0_percent"],
}


def hodges_lehmann_paired(x, y, alpha=0.05):
    """Walsh-average HL estimator + Wilcoxon p-value + CI via normal approx."""
    d = np.asarray(x, dtype=np.float64) - np.asarray(y, dtype=np.float64)
    n = len(d)
    if n < 2:
        return None

    walsh = []
    for i in range(n):
        for j in range(i, n):
            walsh.append((d[i] + d[j]) / 2.0)
    walsh = np.sort(walsh)
    hl = float(np.median(walsh))

    try:
        method = "exact" if n <= 25 else "approx"
        stat, pval = wilcoxon(d, alternative="two-sided", method=method)
    except Exception:
        stat, pval, method = np.nan, np.nan, "n/a"

    N_walsh = len(walsh)
    z = norm.ppf(1 - alpha / 2)
    mu_T = n * (n + 1) / 4
    sigma_T = np.sqrt(n * (n + 1) * (2 * n + 1) / 24)
    K = int(np.round(mu_T - z * sigma_T))
    K = max(1, min(K, N_walsh))

    return {
        "hl": hl,
        "ci_lo": float(walsh[K - 1]),
        "ci_hi": float(walsh[N_walsh - K]),
        "p_wilcoxon": float(pval) if pval == pval else float("nan"),
        "n": n,
        "method": method,
    }


def load_summary(path: Path) -> dict | None:
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return data.get("metrics", {})


def load_per_case(path: Path) -> dict | None:
    """Read per_case.csv into {case_id: {metric: value}}."""
    if not path.exists():
        return None
    rows = {}
    with open(path, encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            cid = row.get("case_id", "")
            rows[cid] = row
    return rows


def parse_manifest(manifest_path: Path) -> list[tuple[str, str, str, str, str]]:
    rows = []
    for raw in manifest_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line:
            continue
        parts = line.split("|")
        if len(parts) < 5:
            continue
        model, ckpt_path, ckpt_ds, eval_ds, config_key = parts[:5]
        rows.append((model, ckpt_path, ckpt_ds, eval_ds, config_key))
    return rows


def ckpt_tag_from_path(ckpt_path: str) -> str:
    p = Path(ckpt_path)
    parent = p.parent
    return parent.parent.name if parent.name == "ckpt" else parent.name


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True, help="Pipe-separated experiment manifest (same as runner).")
    ap.add_argument("--results-root", default="results/infer", help="Root where per-run summary.json lives.")
    ap.add_argument("--out", required=True, help="Output CSV path (wide summary).")
    ap.add_argument("--pairwise-out", default=None, help="Optional CSV for pairwise Wilcoxon+HL stats (default: alongside --out).")
    args = ap.parse_args()

    experiments = parse_manifest(Path(args.manifest))
    results_root = Path(args.results_root)

    # ── Summary CSV (means + stds) ─────────────────────────────────────
    header = (
        ["model", "ckpt_ds", "eval_ds", "direction", "config"]
        + [f"{m}_mean" for m in METRICS]
        + [f"{m}_std" for m in METRICS]
        + ["n_cases", "summary_path"]
    )
    rows_out: list[list] = []

    # Keep per-case data for pairwise stats
    per_case_index: dict[tuple[str, str, str], dict] = {}  # (model, ckpt_ds, eval_ds) -> rows

    for model, ckpt_path, ckpt_ds, eval_ds, config_key in experiments:
        ckpt_tag = ckpt_tag_from_path(ckpt_path)
        run_dir = results_root / eval_ds / model / ckpt_tag
        summary_path = run_dir / "summary.json"
        per_case_path = run_dir / "per_case.csv"
        metrics = load_summary(summary_path)

        direction = "native" if ckpt_ds == eval_ds else "cross"
        row = [model, ckpt_ds, eval_ds, direction, config_key]

        if metrics is None:
            row += ["" for _ in METRICS] + ["" for _ in METRICS] + ["", str(summary_path)]
        else:
            means, stds = [], []
            for m in METRICS:
                entry = metrics.get(m)
                if entry is None:
                    means.append(""); stds.append("")
                else:
                    means.append(f"{float(entry['mean']):.6f}")
                    stds.append(f"{float(entry['std']):.6f}")
            with open(summary_path, encoding="utf-8") as f:
                n_cases = str(json.load(f).get("n_cases", ""))
            row += means + stds + [n_cases, str(summary_path)]

        rows_out.append(row)

        pc = load_per_case(per_case_path)
        if pc is not None:
            per_case_index[(model, ckpt_ds, eval_ds)] = pc

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows_out)
    print(f"[SAVED] {out_path}")

    # Pretty short table for console
    print()
    print(f"{'model':<10} {'ckpt':<6} {'eval':<6} {'dir':<6} {'dice':<8} {'hd95':<8} {'sdlogj':<8} {'j<=0%':<8}")
    print("-" * 68)
    for row in rows_out:
        model, ckpt_ds, eval_ds, direction = row[:4]
        dice, hd95, sdlogj, jleq0 = row[5], row[6], row[7], row[8]
        def fmt(x):
            try:
                return f"{float(x):.4f}"
            except (ValueError, TypeError):
                return "-"
        print(f"{model:<10} {ckpt_ds:<6} {eval_ds:<6} {direction:<6} {fmt(dice):<8} {fmt(hd95):<8} {fmt(sdlogj):<8} {fmt(jleq0):<8}")

    # ── Pairwise Wilcoxon + HL within each (ckpt_ds, eval_ds) cell ─────
    pairwise_path = Path(args.pairwise_out) if args.pairwise_out else out_path.with_name(out_path.stem + "_pairwise.csv")
    pairwise_header = ["ckpt_ds", "eval_ds", "direction", "metric", "model_a", "model_b", "n", "hl", "ci_lo", "ci_hi", "p_wilcoxon", "method"]
    pairwise_rows = []

    # Collect all (ckpt_ds, eval_ds) cells
    cells = sorted({(ckpt_ds, eval_ds) for (_, ckpt_ds, eval_ds) in per_case_index.keys()})
    print()
    print("Pairwise comparisons (CTCF vs baselines):")
    for ckpt_ds, eval_ds in cells:
        ctcf_rows = per_case_index.get(("ctcf", ckpt_ds, eval_ds))
        if ctcf_rows is None:
            continue
        direction = "native" if ckpt_ds == eval_ds else "cross"
        metrics_here = PAIRWISE_METRICS_BY_DS.get(eval_ds, ["dice_mean", "hd95_mean"])
        case_ids = sorted(ctcf_rows.keys())

        for baseline in ("tm-dca", "utsrmorph"):
            base_rows = per_case_index.get((baseline, ckpt_ds, eval_ds))
            if base_rows is None:
                continue
            shared = [cid for cid in case_ids if cid in base_rows]
            if len(shared) < 2:
                continue

            for metric in metrics_here:
                try:
                    xs = [float(ctcf_rows[cid][metric]) for cid in shared]
                    ys = [float(base_rows[cid][metric]) for cid in shared]
                except (KeyError, ValueError):
                    continue
                res = hodges_lehmann_paired(xs, ys)
                if res is None:
                    continue
                pairwise_rows.append([
                    ckpt_ds, eval_ds, direction, metric, "ctcf", baseline,
                    res["n"], f"{res['hl']:+.6f}", f"{res['ci_lo']:+.6f}", f"{res['ci_hi']:+.6f}",
                    f"{res['p_wilcoxon']:.3e}", res["method"],
                ])
                print(f"  [{ckpt_ds}->{eval_ds}] {metric:<14} ctcf vs {baseline:<10} HL={res['hl']:+.5f}  CI=[{res['ci_lo']:+.5f}, {res['ci_hi']:+.5f}]  p={res['p_wilcoxon']:.3e}  n={res['n']}")

    if pairwise_rows:
        with open(pairwise_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(pairwise_header)
            w.writerows(pairwise_rows)
        print(f"\n[SAVED] {pairwise_path}")
    else:
        print("\n[WARN] No pairwise stats produced (per_case.csv missing or case_id mismatch).")


if __name__ == "__main__":
    main()
