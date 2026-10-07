"""Dice over all 35 OASIS labels, the 29 regular ones and the 6 small low-contrast ones.

The six are the inferior lateral ventricles (4, 23), vessels (18, 34) and choroid plexus (19, 35):
they hold about half of the gain labels bring and nearly all of the Dice the certified repair
costs, so every pilot reports them apart. Group means are equal-weight over labels, then cases.

    python -m tools.analysis.dice_groups results/pilots/*/per_case.csv
"""

from __future__ import annotations

import argparse
import csv
import statistics

SMALL_LABELS = (4, 18, 19, 23, 34, 35)
REGULAR_LABELS = tuple(label for label in range(1, 36) if label not in SMALL_LABELS)


def group_means(per_case_csv: str) -> dict[str, float]:
    with open(per_case_csv, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"{per_case_csv}: no cases")

    def mean_over(labels: tuple[int, ...]) -> float:
        return statistics.mean(statistics.mean(float(row[f"dice_lbl_{label}"]) for label in labels) for row in rows)

    summary = {
        "cases": float(len(rows)),
        "dice35": mean_over(REGULAR_LABELS + SMALL_LABELS),
        "dice29": mean_over(REGULAR_LABELS),
        "dice6": mean_over(SMALL_LABELS),
    }
    if "tri_cert_bound" in rows[0]:
        summary["min_tri_cert_bound"] = min(float(row["tri_cert_bound"]) for row in rows)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("per_case", nargs="+", help="per_case.csv files written by experiments.inference")
    args = parser.parse_args()
    print(f"{'cases':>5} {'dice35':>8} {'dice29':>8} {'dice6':>8} {'min_cert':>10}  file")
    for path in args.per_case:
        s = group_means(path)
        cert = f"{s['min_tri_cert_bound']:10.4f}" if "min_tri_cert_bound" in s else f"{'-':>10}"
        print(f"{int(s['cases']):5d} {s['dice35']:8.4f} {s['dice29']:8.4f} {s['dice6']:8.4f} {cert}  {path}")


if __name__ == "__main__":
    main()
