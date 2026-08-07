"""
Generate publication-ready LaTeX tables from inference summary.json files.

Usage:
    python -m tools.paper.make_paper_tables \
        --jsons results/infer/OASIS/ctcf/best/summary.json "CTCF" \
                results/infer/OASIS/tm-dca/best.pth/summary.json "TM-DCA" \
                results/infer/OASIS/utsrmorph/best.pth/summary.json "UTSRMorph" \
        --out figures/table_oasis.tex
"""

import argparse
import json
from pathlib import Path


def load_summary(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def fmt_metric(summary: dict, metric: str, fmt: str = ".4f", show_std: bool = True) -> str:
    """Format metric as 'mean ± std' or just 'mean'."""
    metrics = summary.get("metrics", {})
    if metric not in metrics:
        return "---"
    m = metrics[metric]
    mean = m["mean"]
    std = m.get("std", 0)
    if show_std:
        return f"{mean:{fmt}} $\\pm$ {std:{fmt}}"
    return f"{mean:{fmt}}"


def generate_main_table(pairs: list, dataset: str) -> str:
    """Generate the main comparison table."""
    lines = []
    lines.append("\\begin{table}[htbp]")
    lines.append(
        f"\\caption{{Quantitative comparison on {dataset} test set. "
        "Bold indicates best, underline indicates second best.}}"
    )
    lines.append(f"\\label{{tab:results_{dataset.lower()}}}")
    lines.append("\\centering")
    lines.append("\\begin{tabular}{lcccc}")
    lines.append("\\toprule")
    lines.append("Method & Dice $\\uparrow$ & HD95 $\\downarrow$ & Fold \\% $\\downarrow$ & SDlogJ $\\downarrow$ \\\\")
    lines.append("\\midrule")

    # collect values for bolding
    all_data = []
    for json_path, label in pairs:
        s = load_summary(json_path)
        dice = s["metrics"].get("dice_mean", {}).get("mean")
        hd95 = s["metrics"].get("hd95_mean", {}).get("mean")
        fold = s["metrics"].get("j_leq0_percent", s["metrics"].get("fold_percent", {})).get("mean")
        sdlogj = s["metrics"].get("sdlogj", {}).get("mean")
        all_data.append(
            {
                "label": label,
                "summary": s,
                "dice": dice,
                "hd95": hd95,
                "fold": fold,
                "sdlogj": sdlogj,
            }
        )

    for entry in all_data:
        s = entry["summary"]
        dice_str = fmt_metric(s, "dice_mean", ".4f")
        hd95_str = fmt_metric(s, "hd95_mean", ".2f", show_std=False)
        sdlogj_str = fmt_metric(s, "sdlogj", ".4f", show_std=False)

        # fold% — try j_leq0_percent first, then fold_percent
        fold_str = "—"
        for key in ["j_leq0_percent", "fold_percent"]:
            if key in s.get("metrics", {}):
                fold_str = fmt_metric(s, key, ".3f", show_std=False)
                break

        lines.append(f"{entry['label']} & {dice_str} & {hd95_str} & {fold_str} & {sdlogj_str} \\\\")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")
    return "\n".join(lines)


def generate_ablation_table(pairs: list) -> str:
    """Generate ablation study table from multiple summaries."""
    lines = []
    lines.append("\\begin{table}[htbp]")
    lines.append("\\caption{Ablation study results (100 epochs, OASIS).}")
    lines.append("\\label{tab:ablation}")
    lines.append("\\centering")
    lines.append("\\begin{tabular}{lccc}")
    lines.append("\\toprule")
    lines.append("Configuration & Dice $\\uparrow$ & SDlogJ $\\downarrow$ & Fold \\% $\\downarrow$ \\\\")
    lines.append("\\midrule")

    for json_path, label in pairs:
        s = load_summary(json_path)
        dice_str = fmt_metric(s, "dice_mean", ".4f", show_std=False)
        sdlogj_str = fmt_metric(s, "sdlogj", ".4f", show_std=False)
        fold_str = "—"
        for key in ["j_leq0_percent", "fold_percent"]:
            if key in s.get("metrics", {}):
                fold_str = fmt_metric(s, key, ".3f", show_std=False)
                break
        lines.append(f"{label} & {dice_str} & {sdlogj_str} & {fold_str} \\\\")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Generate LaTeX tables from summary.json")
    parser.add_argument("--jsons", nargs="+", required=True, help="Pairs: <summary.json> <label>")
    parser.add_argument("--out", default="figures/table_main.tex")
    parser.add_argument("--dataset", default="OASIS", help="Dataset name for caption")
    parser.add_argument(
        "--mode", default="main", choices=["main", "ablation"], help="Table type: 'main' comparison or 'ablation' study"
    )
    args = parser.parse_args()

    if len(args.jsons) % 2 != 0:
        parser.error("--jsons requires pairs")
    pairs = [(args.jsons[i], args.jsons[i + 1]) for i in range(0, len(args.jsons), 2)]

    if args.mode == "main":
        tex = generate_main_table(pairs, args.dataset)
    else:
        tex = generate_ablation_table(pairs)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(tex, encoding="utf-8")
    print(tex)
    print(f"\n[OK] Saved to {out}")


if __name__ == "__main__":
    main()
