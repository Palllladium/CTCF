"""
Unified statistics tool: paired Wilcoxon + Hodges-Lehmann estimator + 95% CI.

Replaces three previous scripts:
  - compute_stats.py              → mode 'paper1'
  - compute_stats_cross.py        → mode 'cross'
  - compute_stats_sedm_vs_paper1.py → mode 'sedm_vs_paper1'

Usage:
    # Paper 1 baselines (CTCF Swin-DCA vs TM-DCA vs UTSRMorph)
    python tools/analysis/compute_stats.py paper1 [--infer-root results/infer]

    # Cross-dataset evaluation (manifest-driven)
    python tools/analysis/compute_stats.py cross \\
        --manifest path/to/manifest.txt \\
        --results-root results/infer \\
        --out results/cross_summary.csv

    # SEDM cascades vs Paper 1 CTCF Swin-DCA cascade
    python tools/analysis/compute_stats.py sedm_vs_paper1 \\
        [--infer-root results/infer] \\
        [--sedm-root results/SEDM/inference] \\
        [--out results/SEDM/summary/stat_tests_vs_paper1.txt]
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm, wilcoxon


# ---------------------------------------------------------------------------
# Shared utility — Hodges-Lehmann + Wilcoxon
# ---------------------------------------------------------------------------

def hodges_lehmann_paired(x, y, alpha=0.05):
    """Paired HL estimator with Walsh-CI and Wilcoxon p-value.

    Returns dict with hl, ci_lo, ci_hi, p_wilcoxon, wilcoxon_stat, n, method, K, N_walsh.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    d = x - y
    n = len(d)
    if n < 2:
        return None

    walsh = np.array([(d[i] + d[j]) / 2.0 for i in range(n) for j in range(i, n)])
    walsh.sort()
    hl = float(np.median(walsh))
    K_walsh = len(walsh)

    method = "exact" if n <= 25 else "approx"
    try:
        stat, pval = wilcoxon(d, alternative="two-sided", method=method)
    except Exception:
        stat, pval, method = float("nan"), float("nan"), "n/a"

    # Normal-approx CI for rank
    z = norm.ppf(1 - alpha / 2)
    mu_T = n * (n + 1) / 4
    sigma_T = np.sqrt(n * (n + 1) * (2 * n + 1) / 24)
    K = int(np.round(mu_T - z * sigma_T))
    K = max(1, min(K, K_walsh))

    return {
        "hl": hl,
        "ci_lo": float(walsh[K - 1]),
        "ci_hi": float(walsh[K_walsh - K]),
        "p_wilcoxon": float(pval) if pval == pval else float("nan"),
        "wilcoxon_stat": float(stat) if stat == stat else float("nan"),
        "n": n,
        "method": method,
        "K": K,
        "N_walsh": K_walsh,
    }


def _try_per_case(infer_root: Path, dataset: str, model: str) -> pd.DataFrame:
    for subdir in ("best", "best.pth"):
        p = infer_root / dataset / model / subdir / "per_case.csv"
        if p.exists():
            return pd.read_csv(p)
    raise FileNotFoundError(f"No per_case.csv at {infer_root}/{dataset}/{model}/{{best,best.pth}}/")


def _print_pair_result(metric: str, res: dict):
    direction = "higher is better" if metric == "dice_mean" else "lower is better"
    print(f"    {metric:20s} ({direction}):")
    print(f"      HL = {res['hl']:+.6f}  95% CI = [{res['ci_lo']:+.6f}, {res['ci_hi']:+.6f}]")
    print(f"      Wilcoxon p = {res['p_wilcoxon']:.4e}  (N={res['n']}, method={res['method']})")


# ---------------------------------------------------------------------------
# Mode 1: Paper 1 baselines (CTCF Swin-DCA vs TM-DCA vs UTSRMorph)
# ---------------------------------------------------------------------------

def cmd_paper1(args):
    infer_root = Path(args.infer_root)

    print("=" * 80)
    print("Paper 1 baseline comparisons (CTCF Swin-DCA vs TM-DCA / UTSRMorph)")
    print("=" * 80)

    for ds, metrics in (
        ("OASIS", ["dice_mean", "hd95_mean", "sdlogj"]),
        ("IXI",   ["dice_mean", "hd95_mean", "j_leq0_percent"]),
    ):
        print(f"\n### {ds}")
        ctcf = _try_per_case(infer_root, ds, "ctcf")
        tmdca = _try_per_case(infer_root, ds, "tm-dca")
        utsr = _try_per_case(infer_root, ds, "utsrmorph")

        # Verify pairing
        if list(ctcf["case_id"]) != list(tmdca["case_id"]):
            print(f"  ⚠ {ds} case_id mismatch CTCF vs TM-DCA")
        if list(ctcf["case_id"]) != list(utsr["case_id"]):
            print(f"  ⚠ {ds} case_id mismatch CTCF vs UTSRMorph")

        for comp_label, df_b in (("CTCF vs TM-DCA", tmdca), ("CTCF vs UTSRMorph", utsr)):
            print(f"\n  {comp_label}:")
            for metric in metrics:
                res = hodges_lehmann_paired(ctcf[metric].values, df_b[metric].values)
                _print_pair_result(metric, res)


# ---------------------------------------------------------------------------
# Mode 2: Cross-dataset (manifest-driven)
# ---------------------------------------------------------------------------

METRICS_CROSS = ["dice_mean", "hd95_mean", "sdlogj", "j_leq0_percent", "ndv_percent", "time_sec"]
PAIRWISE_METRICS_BY_DS = {
    "OASIS": ["dice_mean", "hd95_mean", "sdlogj"],
    "IXI":   ["dice_mean", "hd95_mean", "j_leq0_percent"],
}


def _load_summary(p: Path):
    if not p.exists():
        return None
    with open(p, encoding="utf-8") as f:
        return json.load(f).get("metrics", {})


def _load_per_case_dict(p: Path):
    if not p.exists():
        return None
    out = {}
    with open(p, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            out[row.get("case_id", "")] = row
    return out


def _parse_manifest(p: Path):
    rows = []
    for raw in p.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line:
            continue
        parts = line.split("|")
        if len(parts) < 5:
            continue
        rows.append(tuple(parts[:5]))
    return rows


def _ckpt_tag(ckpt_path: str) -> str:
    p = Path(ckpt_path)
    parent = p.parent
    return parent.parent.name if parent.name == "ckpt" else parent.name


def cmd_cross(args):
    experiments = _parse_manifest(Path(args.manifest))
    results_root = Path(args.results_root)

    header = (
        ["model", "ckpt_ds", "eval_ds", "direction", "config"]
        + [f"{m}_mean" for m in METRICS_CROSS]
        + [f"{m}_std" for m in METRICS_CROSS]
        + ["n_cases", "summary_path"]
    )
    rows_out = []
    per_case_index = {}

    for model, ckpt_path, ckpt_ds, eval_ds, config_key in experiments:
        tag = _ckpt_tag(ckpt_path)
        run_dir = results_root / eval_ds / model / tag
        summary_path = run_dir / "summary.json"
        per_case_path = run_dir / "per_case.csv"

        metrics = _load_summary(summary_path)
        direction = "native" if ckpt_ds == eval_ds else "cross"
        row = [model, ckpt_ds, eval_ds, direction, config_key]

        if metrics is None:
            row += [""] * len(METRICS_CROSS) * 2 + ["", str(summary_path)]
        else:
            means, stds = [], []
            for m in METRICS_CROSS:
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

        pc = _load_per_case_dict(per_case_path)
        if pc is not None:
            per_case_index[(model, ckpt_ds, eval_ds)] = pc

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(header); w.writerows(rows_out)
    print(f"[SAVED] {out_path}")

    # Short console table
    print(f"\n{'model':<10} {'ckpt':<6} {'eval':<6} {'dir':<6} {'dice':<8} {'hd95':<8} {'sdlogj':<8} {'j<=0%':<8}")
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

    # Pairwise within each (ckpt_ds, eval_ds)
    pairwise_path = Path(args.pairwise_out) if args.pairwise_out else out_path.with_name(out_path.stem + "_pairwise.csv")
    pairwise_header = ["ckpt_ds", "eval_ds", "direction", "metric", "model_a", "model_b", "n", "hl", "ci_lo", "ci_hi", "p_wilcoxon", "method"]
    pairwise_rows = []

    cells = sorted({(ckpt_ds, eval_ds) for (_, ckpt_ds, eval_ds) in per_case_index.keys()})
    print("\nPairwise comparisons (CTCF vs baselines):")
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
            w = csv.writer(f); w.writerow(pairwise_header); w.writerows(pairwise_rows)
        print(f"\n[SAVED] {pairwise_path}")
    else:
        print("\n[WARN] No pairwise stats produced (per_case.csv missing or case_id mismatch).")


# ---------------------------------------------------------------------------
# Mode 3: SEDM cascades vs Paper 1 CTCF Swin-DCA cascade
# ---------------------------------------------------------------------------

SEDM_OASIS_CONFIGS = [
    ("P9_CASC_VXM_SVF_OASIS",          "VxM SVF"),
    ("SEDM_CASC_VXM_NOSVF_OASIS",      "VxM NoSVF"),
    ("P8_CASC_LKU8_FIXSCHED_OASIS",    "LKU-8 NoSVF (fixsched)"),
    ("P9_CASC_LKU8_SVF_OASIS",         "LKU-8 SVF"),
    ("P8_CASC_LKU32_SVF_OASIS",        "LKU-32 SVF"),
    ("P7_CASC_MAMBA_SVF_OASIS",        "Mamba SVF"),
    ("SEDM_CASC_MAMBA_NOSVF_OASIS",    "Mamba NoSVF"),
    ("P7_CASC_VMAMBA_SVF_OASIS",       "VMamba SVF"),
]
SEDM_IXI_CONFIGS = [
    ("P9_CASC_VXM_SVF_IXI",            "VxM SVF"),
    ("SEDM_CASC_VXM_NOSVF_IXI",        "VxM NoSVF"),
    ("P9_CASC_LKU8_SVF_IXI",           "LKU-8 SVF"),
    ("SEDM_CASC_LKU8_NOSVF_IXI",       "LKU-8 NoSVF"),
    ("P8_CASC_LKU32_SVF_IXI",          "LKU-32 SVF"),
    ("P8_CASC_MAMBA_SVF_IXI",          "Mamba SVF"),
    ("SEDM_CASC_MAMBA_NOSVF_IXI",      "Mamba NoSVF"),
    ("P9_CASC_VMAMBA_SVF_IXI",         "VMamba SVF"),
]


def _compare_sedm(df_a: pd.DataFrame, df_b: pd.DataFrame, metrics, label_a: str, label_b: str):
    print(f"\n  {label_a} vs {label_b}:")
    if list(df_a["case_id"]) != list(df_b["case_id"]):
        print(f"    ⚠ case_id mismatch: {label_a} != {label_b}")
        common = set(df_a["case_id"]) & set(df_b["case_id"])
        df_a = df_a[df_a["case_id"].isin(common)].sort_values("case_id").reset_index(drop=True)
        df_b = df_b[df_b["case_id"].isin(common)].sort_values("case_id").reset_index(drop=True)
        print(f"    aligned to {len(df_a)} common cases")
    for m in metrics:
        if m not in df_a.columns or m not in df_b.columns:
            print(f"    {m}: not available")
            continue
        res = hodges_lehmann_paired(df_a[m].values, df_b[m].values)
        _print_pair_result(m, res)


def cmd_sedm_vs_paper1(args):
    infer_root = Path(args.infer_root)
    sedm_root = Path(args.sedm_root)

    print("=" * 80)
    print("OASIS — SEDM cascades vs Paper 1 CTCF Swin-DCA cascade")
    print("=" * 80)
    p1_oasis = _try_per_case(infer_root, "OASIS", "ctcf")
    print(f"Paper 1 CTCF OASIS: {len(p1_oasis)} cases, mean Dice = {p1_oasis['dice_mean'].mean():.4f}")
    metrics_oasis = ["dice_mean", "hd95_mean", "sdlogj"]
    for exp, label in SEDM_OASIS_CONFIGS:
        sedm_path = sedm_root / exp / "per_case.csv"
        if not sedm_path.exists():
            print(f"\n  SKIP {label}: {sedm_path} missing")
            continue
        df = pd.read_csv(sedm_path)
        _compare_sedm(df, p1_oasis, metrics_oasis, label, "Paper 1 CTCF Swin-DCA")

    print("\n" + "=" * 80)
    print("IXI — SEDM cascades vs Paper 1 CTCF Swin-DCA cascade")
    print("=" * 80)
    p1_ixi = _try_per_case(infer_root, "IXI", "ctcf")
    print(f"Paper 1 CTCF IXI: {len(p1_ixi)} cases, mean Dice = {p1_ixi['dice_mean'].mean():.4f}")
    metrics_ixi = ["dice_mean", "hd95_mean", "j_leq0_percent"]
    for exp, label in SEDM_IXI_CONFIGS:
        sedm_path = sedm_root / exp / "per_case.csv"
        if not sedm_path.exists():
            print(f"\n  SKIP {label}: {sedm_path} missing")
            continue
        df = pd.read_csv(sedm_path)
        _compare_sedm(df, p1_ixi, metrics_ixi, label, "Paper 1 CTCF Swin-DCA")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1].strip())
    sub = p.add_subparsers(dest="mode", required=True)

    p1 = sub.add_parser("paper1", help="Paper 1 baselines: CTCF Swin-DCA vs TM-DCA / UTSRMorph")
    p1.add_argument("--infer-root", default="results/infer")

    p2 = sub.add_parser("cross", help="Cross-dataset evaluation via manifest")
    p2.add_argument("--manifest", required=True)
    p2.add_argument("--results-root", default="results/infer")
    p2.add_argument("--out", required=True)
    p2.add_argument("--pairwise-out", default=None)

    p3 = sub.add_parser("sedm_vs_paper1", help="SEDM cascade configs vs Paper 1 CTCF Swin-DCA cascade")
    p3.add_argument("--infer-root", default="results/infer")
    p3.add_argument("--sedm-root", default="results/SEDM/inference")

    args = p.parse_args()
    if args.mode == "paper1":
        cmd_paper1(args)
    elif args.mode == "cross":
        cmd_cross(args)
    elif args.mode == "sedm_vs_paper1":
        cmd_sedm_vs_paper1(args)


if __name__ == "__main__":
    main()
