"""
Per-region Dice aggregation for Paper 2 §7 (region x backbone analysis).

Reads the per-label Dice already stored in each run's per_case.csv (`dice_lbl_<N>` columns,
where <N> is the true VOI label value — OASIS uses 1..35, IXI uses 1..36 minus the 6 absent
labels), and produces format-agnostic CSV summaries plus an optional heatmap PNG:

  per_region_<DS>_long.csv     — one row per (config, label): dice_mean, dice_std, n
  per_region_<DS>_matrix.csv   — labels x backbones grid of mean Dice
  per_region_<DS>_winners.csv  — per label: best backbone, margin over runner-up
  per_region_<DS>_heatmap.png  — labels x backbones heatmap (skip with --no-plot)
  per_region_<DS>_by_class.csv — mean Dice per anatomical class x backbone (only with --label-map)

Anatomical region NAMES and cortical/subcortical CLASSES are not encoded anywhere in this
codebase, so by default the analysis is by integer label only. Pass --label-map CSV with columns
`label,name,class` to add names and the cortical/subcortical breakdown — this tool will NOT
invent that mapping.

Usage:
    python tools/analysis/per_region.py --ds OASIS
    python tools/analysis/per_region.py --ds both --label-map docs/oasis_labels.csv
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import numpy as np
import pandas as pd

MANIFEST_PATH = Path(__file__).parent / "manifests" / "experiments.csv"
PAPER1_LABEL = "CTCF Swin-DCA (P1)"
_DICE_LBL_RE = re.compile(r"^dice_lbl_(\d+)$")


def load_manifest(path: Path) -> dict[str, dict]:
    with open(path, encoding="utf-8") as f:
        return {row["exp_name"]: row for row in csv.DictReader(f)}


def backbone_label(entry: dict) -> str:
    """Human-readable backbone tag from a manifest row, e.g. 'MambaMorph SVF'."""
    svf = "SVF" if entry.get("svf") == "ON" else "NoSVF"
    return f"{entry['backbone']} {svf}"


def per_label_stats(per_case_csv: Path) -> dict[int, tuple[float, float, int]]:
    """Return {label: (dice_mean, dice_std, n)} from a per_case.csv's dice_lbl_<N> columns."""
    df = pd.read_csv(per_case_csv)
    out: dict[int, tuple[float, float, int]] = {}
    for col in df.columns:
        m = _DICE_LBL_RE.match(col)
        if not m:
            continue
        vals = df[col].to_numpy(dtype=np.float64)
        vals = vals[~np.isnan(vals)]
        if vals.size == 0:
            continue
        std = float(vals.std(ddof=1)) if vals.size > 1 else 0.0
        out[int(m.group(1))] = (float(vals.mean()), std, int(vals.size))
    return out


def default_configs(manifest: dict, ds: str) -> list[tuple[str, str, Path]]:
    """Headline §7 set: the 500ep P10 longruns for this dataset. Returns (exp, label, per_case)."""
    out = []
    for exp, entry in manifest.items():
        if entry["ds"] == ds and entry["group"] == "cascade" and exp.startswith("P10_LONGRUN"):
            out.append((exp, backbone_label(entry)))
    out.sort(key=lambda t: manifest[t[0]]["params_m"])
    return out


def collect(ds: str, args, manifest: dict) -> pd.DataFrame:
    """Build the long-format per-region table for one dataset."""
    inference_dir = Path(args.inference_dir)
    sources: list[tuple[str, Path]] = []  # (label, per_case_path)

    # Paper-1 reference first, then the cascade configs.
    p1 = Path(args.paper1_root) / ds / "ctcf" / "best" / "per_case.csv"
    if p1.exists():
        sources.append((PAPER1_LABEL, p1))

    configs = args.configs or [exp for exp, _ in default_configs(manifest, ds)]
    for exp in configs:
        label = backbone_label(manifest[exp]) if exp in manifest else exp
        path = inference_dir / exp / "per_case.csv"
        if path.exists():
            sources.append((label, path))
        else:
            print(f"[SKIP] {ds}: {path} missing")

    rows = []
    for label, path in sources:
        for lbl, (mean, std, n) in sorted(per_label_stats(path).items()):
            rows.append({"ds": ds, "backbone": label, "label": lbl, "dice_mean": mean, "dice_std": std, "n": n})
    return pd.DataFrame(rows)


def attach_label_map(df: pd.DataFrame, label_map_csv: Path) -> pd.DataFrame:
    lm = pd.read_csv(label_map_csv)  # expects columns: label, name, class
    return df.merge(lm[["label", "name", "class"]], on="label", how="left")


def write_outputs(df: pd.DataFrame, ds: str, out_dir: Path, plot: bool) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / f"per_region_{ds}_long.csv", index=False)

    matrix = df.pivot(index="label", columns="backbone", values="dice_mean")
    # Preserve the collection order of backbones (pivot sorts columns alphabetically otherwise).
    matrix = matrix[list(dict.fromkeys(df["backbone"]))]
    matrix.to_csv(out_dir / f"per_region_{ds}_matrix.csv")

    winners = _winners_table(matrix)
    winners.to_csv(out_dir / f"per_region_{ds}_winners.csv", index=False)

    if "class" in df.columns:
        by_class = df.groupby(["class", "backbone"], dropna=True)["dice_mean"].mean().unstack("backbone")
        by_class = by_class[[c for c in matrix.columns if c in by_class.columns]]
        by_class.to_csv(out_dir / f"per_region_{ds}_by_class.csv")
        print(f"  by-class summary written ({by_class.shape[0]} classes)")

    if plot:
        _heatmap(matrix, ds, out_dir / f"per_region_{ds}_heatmap.png")

    print(f"  {ds}: {matrix.shape[0]} labels x {matrix.shape[1]} configs -> {out_dir}")


def _winners_table(matrix: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for lbl, series in matrix.iterrows():
        ordered = series.sort_values(ascending=False)
        best, best_val = ordered.index[0], ordered.iloc[0]
        runner_val = ordered.iloc[1] if len(ordered) > 1 else float("nan")
        rows.append(
            {
                "label": lbl,
                "best_backbone": best,
                "best_dice": round(float(best_val), 5),
                "margin_over_runner_up": round(float(best_val - runner_val), 5),
            }
        )
    return pd.DataFrame(rows)


def _heatmap(matrix: pd.DataFrame, ds: str, out_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(1.6 * matrix.shape[1] + 2, 0.28 * matrix.shape[0] + 2))
    im = ax.imshow(matrix.to_numpy(), aspect="auto", cmap="viridis")
    ax.set_xticks(range(matrix.shape[1]), matrix.columns, rotation=30, ha="right")
    ax.set_yticks(range(matrix.shape[0]), [f"lbl {i}" for i in matrix.index], fontsize=7)
    ax.set_title(f"{ds} — per-region Dice (label x backbone)")
    fig.colorbar(im, ax=ax, label="Dice")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1].strip())
    p.add_argument("--ds", choices=["OASIS", "IXI", "both"], default="both")
    p.add_argument("--inference-dir", default="results/SEDM/inference")
    p.add_argument("--paper1-root", default="results/infer")
    p.add_argument("--manifest", default=str(MANIFEST_PATH))
    p.add_argument("--out-dir", default="results/SEDM/summary")
    p.add_argument("--configs", nargs="*", default=None, help="Explicit exp_name list (default: P10 longruns).")
    p.add_argument("--label-map", default=None, help="CSV with columns label,name,class for the anatomical breakdown.")
    p.add_argument("--no-plot", action="store_true")
    args = p.parse_args()

    manifest = load_manifest(Path(args.manifest))
    datasets = ["OASIS", "IXI"] if args.ds == "both" else [args.ds]
    for ds in datasets:
        df = collect(ds, args, manifest)
        if df.empty:
            print(f"[WARN] {ds}: no per_case data found, skipping")
            continue
        if args.label_map:
            df = attach_label_map(df, Path(args.label_map))
        write_outputs(df, ds, Path(args.out_dir), plot=not args.no_plot)


if __name__ == "__main__":
    main()
