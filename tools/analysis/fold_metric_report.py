"""How much does the choice of finite-difference scheme change a published fold count?

Every per_case.csv now carries three counts of the same folds on the same field:
  j_leq0_central_percent   central differences  — the community default (VoxelMorph/TransMorph lineage)
  j_leq0_corners_percent   8 one-sided schemes
  j_leq0_percent           all 10 determinants  — the full digital criterion (Liu et al., IJCV 2024)

This reports their disagreement across every scored configuration, and runs paired
Wilcoxon + Hodges-Lehmann of each fold count against a baseline checkpoint.

    python tools/analysis/fold_metric_report.py \
        --inference-dir results/SEDM/inference \
        --baseline results/tto2/RS_SWIN_OASIS__none \
        --out results/SEDM/summary/fold_metric_report.md
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools.analysis.compute_stats import benjamini_hochberg, hodges_lehmann_paired

CENTRAL, CORNERS, DIGITAL, NDV = (
    "j_leq0_central_percent",
    "j_leq0_corners_percent",
    "j_leq0_percent",
    "ndv_percent",
)
# A count below this prints as "0.00" at the two decimals papers report.
PRINTS_AS_ZERO = 0.005


def _load(d: Path) -> pd.DataFrame | None:
    p = d / "per_case.csv"
    if not p.exists():
        return None
    df = pd.read_csv(p)
    return df if CENTRAL in df.columns else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inference-dir", default="results/SEDM/inference")
    ap.add_argument("--baseline", default="results/tto2/RS_SWIN_OASIS__none")
    ap.add_argument("--out", default="results/SEDM/summary/fold_metric_report.md")
    a = ap.parse_args()

    rows = []
    for d in sorted(Path(a.inference_dir).iterdir()):
        df = _load(d)
        if df is None:
            continue
        rows.append(
            dict(
                exp=d.name,
                n=len(df),
                dice=df.dice_mean.mean(),
                central=df[CENTRAL].mean(),
                corners=df[CORNERS].mean(),
                digital=df[DIGITAL].mean(),
                ndv=df[NDV].mean(),
            )
        )
    t = pd.DataFrame(rows)
    if t.empty:
        print(f"No per_case.csv with fold columns under {a.inference_dir}")
        return
    t["ratio"] = t.digital / t.central.replace(0, np.nan)

    md = [
        "# Fold count vs finite-difference scheme\n",
        f"Source: `{a.inference_dir}` — {len(t)} configurations.\n",
        "All three columns count folds **in the same deformation field**; they differ only in which",
        "finite differences build the Jacobian.\n",
        "| config | N | Dice | central | corners-8 | digital-10 | digital/central | NDV |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for _, r in t.iterrows():
        ratio = f"{r.ratio:.0f}x" if np.isfinite(r.ratio) else ("∞" if r.digital > 0 else "—")
        md.append(
            f"| {r.exp} | {r.n} | {r.dice:.4f} | {r.central:.4f} | {r.corners:.4f} | "
            f"{r.digital:.4f} | {ratio} | {r.ndv:.4f} |"
        )

    fin = t[np.isfinite(t.ratio)]
    zeros = t[(t.central < PRINTS_AS_ZERO) & (t.digital > 0)]
    md += [
        "\n## Disagreement\n",
        f"- digital-10 / central across {len(fin)} configs with a non-zero central count: "
        f"**median {fin.ratio.median():.0f}x**, range {fin.ratio.min():.0f}x-{fin.ratio.max():.0f}x.",
        f"- digital-10 / corners-8: median {(t.digital / t.corners.replace(0, np.nan)).median():.3f}x.",
        f"- **{len(zeros)} of {len(t)}** configurations would be published as `0.00 %` under central "
        f"differences while the full criterion finds up to {zeros.digital.max():.4f} % folded voxels.\n",
        "### The error is largest where the field looks best\n",
        "| central count | n | median digital/central |",
        "|---|---|---|",
    ]
    for lo, hi, lbl in (
        (0, 0.01, "< 0.01 (looks perfect)"),
        (0.01, 0.5, "0.01 - 0.5"),
        (0.5, 1e9, "> 0.5 (visibly broken)"),
    ):
        s = t[(t.central > 0) & (t.central >= lo) & (t.central < hi)]
        if len(s):
            md.append(f"| {lbl} | {len(s)} | {(s.digital / s.central).median():.0f}x |")
    md.append(
        "\nA permissive scheme flatters a good field far more than a broken one, so the metric is "
        "least trustworthy exactly in the range where published methods operate.\n"
    )

    base = _load(Path(a.baseline))
    if base is not None:
        md += [
            f"## Paired tests vs `{Path(a.baseline).name}`\n",
            f"Baseline: N={len(base)}, Dice {base.dice_mean.mean():.4f}, "
            f"central {base[CENTRAL].mean():.4f} %, digital-10 {base[DIGITAL].mean():.4f} %.\n",
            "Hodges-Lehmann shift (negative = fewer folds than baseline), paired Wilcoxon, "
            "Benjamini-Hochberg across configs per metric.\n",
            "| config | metric | HL | 95% CI | p | q |",
            "|---|---|---|---|---|---|",
        ]
        for metric, label in ((CENTRAL, "central"), (DIGITAL, "digital-10")):
            res, labels = [], []
            for _, r in t.iterrows():
                df = _load(Path(a.inference_dir) / r.exp)
                if df is None or metric not in df.columns:
                    continue
                # Pair on case_id, never on row order: a reordered export would silently
                # compare different scans and every p-value below would be meaningless.
                m = df[["case_id", metric]].merge(base[["case_id", metric]], on="case_id", suffixes=("_a", "_b"))
                if len(m) != len(base):
                    continue
                out = hodges_lehmann_paired(m[f"{metric}_a"].to_numpy(), m[f"{metric}_b"].to_numpy())
                if out is None:
                    continue
                res.append(out)
                labels.append(r.exp)
            if not res:
                continue
            q, _ = benjamini_hochberg([x["p_wilcoxon"] for x in res])
            for lb, x, qq in zip(labels, res, q, strict=False):
                md.append(
                    f"| {lb} | {label} | {x['hl']:+.4f} | "
                    f"[{x['ci_lo']:+.4f}, {x['ci_hi']:+.4f}] | {x['p_wilcoxon']:.2e} | {qq:.2e} |"
                )
    else:
        md.append(f"\n_Baseline `{a.baseline}` has no per_case.csv with fold columns — paired tests skipped._\n")

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md[len(t) + 6 :]))
    print(f"\nWritten: {out}")


if __name__ == "__main__":
    main()
