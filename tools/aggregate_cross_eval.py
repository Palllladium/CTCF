import argparse
import csv
import json
from pathlib import Path


METRICS = ["dice_mean", "hd95_mean", "sdlogj", "j_leq0_percent", "ndv_percent", "time_sec"]


def load_summary(path: Path) -> dict | None:
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return data.get("metrics", {})


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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True, help="Pipe-separated experiment manifest (same as runner).")
    ap.add_argument("--results-root", default="results/infer", help="Root where per-run summary.json lives.")
    ap.add_argument("--out", required=True, help="Output CSV path.")
    args = ap.parse_args()

    experiments = parse_manifest(Path(args.manifest))
    results_root = Path(args.results_root)

    header = ["model", "ckpt_ds", "eval_ds", "direction", "config"] + [f"{m}_mean" for m in METRICS] + [f"{m}_std" for m in METRICS] + ["n_cases", "summary_path"]
    rows_out: list[list] = []

    for model, ckpt_path, ckpt_ds, eval_ds, config_key in experiments:
        p = Path(ckpt_path)
        # Match the bash-side CKPT_TAG rule: use the parent folder name, or its
        # grandparent if the file lives in a `ckpt/` subdirectory.
        parent = p.parent
        ckpt_tag = parent.parent.name if parent.name == "ckpt" else parent.name
        summary_path = results_root / eval_ds / model / ckpt_tag / "summary.json"
        metrics = load_summary(summary_path)

        direction = "native" if ckpt_ds == eval_ds else "cross"
        row = [model, ckpt_ds, eval_ds, direction, config_key]

        if metrics is None:
            row += ["" for _ in METRICS] + ["" for _ in METRICS] + ["", str(summary_path)]
        else:
            means = []
            stds = []
            for m in METRICS:
                entry = metrics.get(m)
                if entry is None:
                    means.append("")
                    stds.append("")
                else:
                    means.append(f"{float(entry['mean']):.6f}")
                    stds.append(f"{float(entry['std']):.6f}")
            n_cases = ""
            with open(summary_path, encoding="utf-8") as f:
                n_cases = str(json.load(f).get("n_cases", ""))
            row += means + stds + [n_cases, str(summary_path)]

        rows_out.append(row)

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
        dice = row[5] or "-"
        hd95 = row[6] or "-"
        sdlogj = row[7] or "-"
        jleq0 = row[8] or "-"
        def fmt(x):
            try:
                return f"{float(x):.4f}"
            except (ValueError, TypeError):
                return "-"
        print(f"{model:<10} {ckpt_ds:<6} {eval_ds:<6} {direction:<6} {fmt(dice):<8} {fmt(hd95):<8} {fmt(sdlogj):<8} {fmt(jleq0):<8}")


if __name__ == "__main__":
    main()
