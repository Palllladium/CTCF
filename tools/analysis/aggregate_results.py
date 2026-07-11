"""
Aggregate per_case.csv outputs from inference.py into paper-ready summary tables.

Reads from:  <inference-dir>/<exp_name>/per_case.csv
Writes:
  <output-dir>/main_oasis.md        — Markdown table of OASIS cascade configs
  <output-dir>/main_oasis.tex       — LaTeX tabular of the same
  <output-dir>/main_ixi.md          — Markdown table of IXI cascade configs
  <output-dir>/main_ixi.tex         — LaTeX tabular of the same
  <output-dir>/cascade_delta.md     — L2-only vs cascade Δ comparison (OASIS + IXI)
  <output-dir>/cascade_delta.tex
  <output-dir>/stat_tests.md        — paired Wilcoxon: Mamba NoSVF vs SVF (SVF redundancy claim)
  <output-dir>/aggregated.csv       — flat CSV with all (exp_name, metric, mean, std)

Usage:
    python tools/analysis/aggregate_results.py \
        --inference-dir results/SEDM/inference \
        --output-dir results/SEDM/summary
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

try:
    from scipy.stats import wilcoxon
except ImportError:
    wilcoxon = None


MANIFEST_PATH = Path(__file__).parent / "manifests" / "experiments.csv"


def load_manifest(path: Path) -> dict[str, dict]:
    """Load the experiment registry from the CSV manifest, keyed by exp_name."""
    registry: dict[str, dict] = {}
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            entry = dict(
                family=row["family"],
                backbone=row["backbone"],
                params_m=float(row["params_m"]),
                ds=row["ds"],
                group=row["group"],
            )
            if row.get("svf"):
                entry["svf"] = row["svf"]
            if row.get("note"):
                entry["note"] = row["note"]
            registry[row["exp_name"]] = entry
    return registry


# Final experiment registry (corrected after the Level-3 SVF audit). Historical
# pre-correction values live in git history; this is the single source of truth.
CONFIGS = load_manifest(MANIFEST_PATH)


def load_per_case(csv_path: Path) -> list[dict]:
    with open(csv_path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def mean_std(values: list[float]) -> tuple[float, float] | tuple[None, None]:
    values = [v for v in values if v is not None]
    if not values:
        return (None, None)
    n = len(values)
    m = sum(values) / n
    if n < 2:
        return (m, 0.0)
    var = sum((v - m) ** 2 for v in values) / (n - 1)
    return (m, var**0.5)


def safe_float(s):
    try:
        return float(s)
    except (TypeError, ValueError):
        return None


def aggregate_one(csv_path: Path, ds: str) -> dict:
    rows = load_per_case(csv_path)
    dice = [safe_float(r.get("dice_mean")) for r in rows]
    hd95 = [safe_float(r.get("hd95_mean")) for r in rows if r.get("hd95_mean")]
    if ds == "OASIS":
        jac = [safe_float(r.get("sdlogj")) for r in rows]
        jac_name = "SDlogJ"
    else:
        jac = [safe_float(r.get("j_leq0_percent")) for r in rows]
        jac_name = "Fold,%"
    return dict(
        n=len(rows),
        dice=mean_std([v for v in dice if v is not None]),
        hd95=mean_std(hd95),
        jac=mean_std([v for v in jac if v is not None]),
        jac_name=jac_name,
        dice_arr=[v for v in dice if v is not None],
    )


def fmt(pair, decimals=4):
    m, s = pair
    if m is None:
        return "—"
    if s is None or s == 0.0:
        return f"{m:.{decimals}f}"
    return f"{m:.{decimals}f} ± {s:.{decimals}f}"


def fmt_tex(pair, decimals=4):
    m, s = pair
    if m is None:
        return "—"
    if s is None or s == 0.0:
        return f"${m:.{decimals}f}$"
    return f"${m:.{decimals}f} \\pm {s:.{decimals}f}$"


def write_main_table(stats_by_exp, output_dir: Path, ds: str):
    """OASIS or IXI main cascade table."""
    suffix = ds.lower()
    rows = [
        (name, s)
        for name, s in stats_by_exp.items()
        if CONFIGS[name]["ds"] == ds and CONFIGS[name]["group"] == "cascade"
    ]
    if not rows:
        return

    jac_name = rows[0][1]["jac_name"]

    # Markdown
    md = [f"# {ds} — каскадные конфигурации (100 эпох)\n"]
    md.append(f"| Семейство | Backbone | SVF L3 | Параметры, млн | Dice ↑ | HD95 ↓ | {jac_name} ↓ | N |")
    md.append("|---|---|---|---|---|---|---|---|")
    for exp_name, s in rows:
        cfg = CONFIGS[exp_name]
        note = f" *({cfg['note']})*" if cfg.get("note") else ""
        md.append(
            f"| {cfg['family']} | {cfg['backbone']}{note} | {cfg.get('svf', '-')} | "
            f"{cfg['params_m']} | {fmt(s['dice'])} | {fmt(s['hd95'], decimals=3)} | "
            f"{fmt(s['jac'])} | {s['n']} |"
        )
    (output_dir / f"main_{suffix}.md").write_text("\n".join(md), encoding="utf-8")

    # LaTeX
    jac_header = jac_name.replace("%", r"\%")
    tex = [
        r"\begin{table}[h]",
        r"\centering",
        f"\\caption{{Сравнение каскадных конфигураций на {ds}, 100 эпох.}}",
        f"\\label{{tab:main_{suffix}}}",
        r"\begin{tabular}{lllrccccr}",
        r"\hline",
        f"Семейство & Backbone & SVF & Параметры, млн & Dice $\\uparrow$ & "
        f"HD95 $\\downarrow$ & {jac_header} $\\downarrow$ & N \\\\",
        r"\hline",
    ]
    for exp_name, s in rows:
        cfg = CONFIGS[exp_name]
        bk = cfg["backbone"]
        if cfg.get("note"):
            bk = f"{bk}\\textsuperscript{{*}}"
        tex.append(
            f"{cfg['family']} & {bk} & {cfg.get('svf', '-')} & {cfg['params_m']} & "
            f"{fmt_tex(s['dice'])} & {fmt_tex(s['hd95'], decimals=3)} & {fmt_tex(s['jac'])} & {s['n']} \\\\"
        )
    tex += [r"\hline", r"\end{tabular}", r"\end{table}"]
    (output_dir / f"main_{suffix}.tex").write_text("\n".join(tex), encoding="utf-8")


def write_cascade_delta_table(stats_by_exp, output_dir: Path):
    """L2-only vs Cascade Δ Dice table. Pairs L2-only and the corresponding cascade
    for each backbone × dataset."""
    pairs = {}
    for exp_name, s in stats_by_exp.items():
        cfg = CONFIGS[exp_name]
        key = (cfg["backbone"], cfg["ds"])
        slot = pairs.setdefault(key, {})
        if cfg["group"] == "L2-only":
            slot["L2"] = s
        # Prefer the SVF-canonical cascade; else keep the first cascade seen.
        elif cfg["group"] == "cascade" and (cfg.get("svf", "OFF") == "ON" or "cascade" not in slot):
            slot["cascade"] = s

    md = [
        "# L2-only vs Cascade — Δ Dice\n",
        "| Dataset | Backbone | L2-only Dice | Cascade Dice | Δ |",
        "|---|---|---|---|---|",
    ]
    tex = [
        r"\begin{table}[h]",
        r"\centering",
        r"\caption{Δ Dice каскадной интеграции относительно L2-only "
        r"под унифицированным протоколом, 100 эпох.}",
        r"\label{tab:cascade_delta}",
        r"\begin{tabular}{llccc}",
        r"\hline",
        r"Dataset & Backbone & L2-only Dice & Cascade Dice & $\Delta$ Dice \\",
        r"\hline",
    ]

    for (backbone, ds), slot in pairs.items():
        l2 = slot.get("L2")
        cs = slot.get("cascade")
        if not l2 or not cs:
            continue
        if l2["dice"][0] is None or cs["dice"][0] is None:
            continue
        delta = cs["dice"][0] - l2["dice"][0]
        md.append(
            f"| {ds} | {backbone} | {fmt(l2['dice'])} | {fmt(cs['dice'])} | {'+' if delta >= 0 else ''}{delta:.4f} |"
        )
        sign = "+" if delta >= 0 else ""
        tex.append(f"{ds} & {backbone} & {fmt_tex(l2['dice'])} & {fmt_tex(cs['dice'])} & ${sign}{delta:.4f}$ \\\\")

    tex += [r"\hline", r"\end{tabular}", r"\end{table}"]
    (output_dir / "cascade_delta.md").write_text("\n".join(md), encoding="utf-8")
    (output_dir / "cascade_delta.tex").write_text("\n".join(tex), encoding="utf-8")


def write_stat_tests(stats_by_exp, output_dir: Path):
    """Paired Wilcoxon for available NoSVF vs SVF comparisons."""
    md = [
        "# Парные статистические тесты (Wilcoxon signed-rank)\n",
        "Доступные парные сравнения L3-SVF OFF против L3-SVF ON:\n",
    ]
    if wilcoxon is None:
        md.append("> SciPy не установлен, тесты пропущены.")
        (output_dir / "stat_tests.md").write_text("\n".join(md), encoding="utf-8")
        return

    comparisons = [
        ("SEDM_CASC_VXM_NOSVF_OASIS", "P9_CASC_VXM_SVF_OASIS", "VoxelMorph NoSVF vs SVF (OASIS)"),
        ("SEDM_CASC_VXM_NOSVF_IXI", "P9_CASC_VXM_SVF_IXI", "VoxelMorph NoSVF vs SVF (IXI)"),
        ("P8_CASC_LKU8_FIXSCHED_OASIS", "P9_CASC_LKU8_SVF_OASIS", "LKU-8 NoSVF vs SVF (OASIS)"),
        ("SEDM_CASC_LKU8_NOSVF_IXI", "P9_CASC_LKU8_SVF_IXI", "LKU-8 NoSVF vs SVF (IXI)"),
        ("SEDM_CASC_MAMBA_NOSVF_OASIS", "P7_CASC_MAMBA_SVF_OASIS", "Mamba NoSVF vs SVF (OASIS)"),
        ("SEDM_CASC_MAMBA_NOSVF_IXI", "P8_CASC_MAMBA_SVF_IXI", "Mamba NoSVF vs SVF (IXI)"),
    ]
    md.append("| Сравнение | N | mean Δ Dice | Wilcoxon p |")
    md.append("|---|---|---|---|")
    for a, b, label in comparisons:
        sa, sb = stats_by_exp.get(a), stats_by_exp.get(b)
        if not sa or not sb:
            md.append(f"| {label} | — | — | данных нет |")
            continue
        n = min(len(sa["dice_arr"]), len(sb["dice_arr"]))
        if n < 2:
            md.append(f"| {label} | {n} | — | мало точек |")
            continue
        diffs = [sa["dice_arr"][i] - sb["dice_arr"][i] for i in range(n)]
        mean_diff = sum(diffs) / n
        try:
            if n <= 25:
                _stat, p = wilcoxon(diffs, alternative="two-sided", method="exact")
            else:
                _stat, p = wilcoxon(diffs, alternative="two-sided", method="approx")
            md.append(f"| {label} | {n} | {mean_diff:+.4f} | {p:.4f} |")
        except Exception as e:
            md.append(f"| {label} | {n} | {mean_diff:+.4f} | ошибка: {e} |")

    (output_dir / "stat_tests.md").write_text("\n".join(md), encoding="utf-8")


def write_aggregated_csv(stats_by_exp, output_dir: Path):
    with open(output_dir / "aggregated.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "exp_name",
                "ds",
                "group",
                "family",
                "backbone",
                "svf",
                "params_m",
                "n",
                "dice_mean",
                "dice_std",
                "hd95_mean",
                "hd95_std",
                "jac_metric",
                "jac_mean",
                "jac_std",
            ]
        )
        for exp_name, s in stats_by_exp.items():
            cfg = CONFIGS[exp_name]
            row = [
                exp_name,
                cfg["ds"],
                cfg["group"],
                cfg["family"],
                cfg["backbone"],
                cfg.get("svf", "-"),
                cfg.get("params_m", "-"),
                s["n"],
                f"{s['dice'][0]:.4f}" if s["dice"][0] is not None else "—",
                f"{s['dice'][1]:.4f}" if s["dice"][1] is not None else "—",
                f"{s['hd95'][0]:.3f}" if s["hd95"][0] is not None else "—",
                f"{s['hd95'][1]:.3f}" if s["hd95"][1] is not None else "—",
                s["jac_name"],
                f"{s['jac'][0]:.4f}" if s["jac"][0] is not None else "—",
                f"{s['jac'][1]:.4f}" if s["jac"][1] is not None else "—",
            ]
            w.writerow(row)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--inference-dir", required=True)
    p.add_argument("--output-dir", required=True)
    args = p.parse_args()

    inference_dir = Path(args.inference_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    stats_by_exp = {}
    for exp_name, cfg in CONFIGS.items():
        # inference.py with --out_dir writes per_case.csv directly under that dir
        # (no extra <ckpt_stem> subdir when --out_dir is explicitly provided).
        csv_path = inference_dir / exp_name / "per_case.csv"
        if not csv_path.exists():
            continue
        stats_by_exp[exp_name] = aggregate_one(csv_path, cfg["ds"])

    if not stats_by_exp:
        print("No per_case.csv found in", inference_dir, file=sys.stderr)
        sys.exit(1)

    print(f"Aggregating {len(stats_by_exp)} experiments...")
    write_main_table(stats_by_exp, output_dir, "OASIS")
    write_main_table(stats_by_exp, output_dir, "IXI")
    write_cascade_delta_table(stats_by_exp, output_dir)
    write_stat_tests(stats_by_exp, output_dir)
    write_aggregated_csv(stats_by_exp, output_dir)

    print(f"Done. Output in {output_dir}/")
    for f in sorted(output_dir.iterdir()):
        print(f"  {f.name}")


if __name__ == "__main__":
    main()
