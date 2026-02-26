"""
aggregate_results.py

Aggregates multiple summary.json files (from experiments/inference.py) into a single CSV table
suitable for MSIT paper tables.

Usage:
  python aggregate_results.py --base_dir /path/to/results/infer
Outputs:
  <base_dir>/paper/tab/aggregate_summary.csv
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
from typing import Dict, List, Tuple


def ensure_dir(p: str) -> str:
    os.makedirs(p, exist_ok=True)
    return p


def load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_dir", required=True, help="Directory containing run folders with summary.json")
    ap.add_argument("--out_csv", default="", help="Override output CSV path")
    args = ap.parse_args()

    base_dir = args.base_dir
    out_tab = ensure_dir(os.path.join(base_dir, "paper", "tab"))
    out_csv = args.out_csv or os.path.join(out_tab, "aggregate_summary.csv")

    paths = sorted(glob.glob(os.path.join(base_dir, "*", "summary.json")))
    if not paths:
        raise RuntimeError(f"No summary.json found under: {base_dir}/*/summary.json")

    rows: List[Dict[str, str]] = []
    all_metrics = set()

    for p in paths:
        s = load_json(p)
        name = str(s.get("model") or os.path.basename(os.path.dirname(p)))
        metrics: Dict[str, dict] = s.get("metrics", {})
        for k in metrics.keys():
            all_metrics.add(k)

        row = {"model": name}
        for k, v in metrics.items():
            row[f"{k}_mean"] = f"{float(v.get('mean', 0.0)):.6f}"
            row[f"{k}_std"]  = f"{float(v.get('std', 0.0)):.6f}"
        rows.append(row)

    metric_list = sorted(all_metrics)
    header = ["model"]
    for k in metric_list:
        header += [f"{k}_mean", f"{k}_std"]

    # Fill missing keys with empty
    for r in rows:
        for h in header:
            r.setdefault(h, "")

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    print("[OK] Wrote:", out_csv)


if __name__ == "__main__":
    main()