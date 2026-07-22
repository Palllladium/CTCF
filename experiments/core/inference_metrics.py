from __future__ import annotations

import csv
import json
import math
import os
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import torch

from utils import IXI_VOI_LABELS, OASIS_VOI_LABELS, digital_jacobian_metrics, logdet_std_from_flow
from utils.field import digital_fold_percent, jacobian_nonpositive_percent


def _fold_counts(flow: torch.Tensor, x_seg: torch.Tensor) -> dict[str, float]:
    """The three fold counts under one roof: they differ by orders of magnitude on one field,
    so a run must emit all of them or the numbers cannot be compared across papers.
    `central` is the community default (VoxelMorph/TransMorph lineage), `corners` the 8 one-sided
    schemes, `j_leq0` the full 10-determinant digital criterion (Liu et al., IJCV 2024).
    """
    j_leq0, ndv = digital_jacobian_metrics(flow, mask=x_seg)
    return {
        "j_leq0_percent": float(j_leq0),
        "j_leq0_corners_percent": float(digital_fold_percent(flow, corners_only=True).item()),
        "j_leq0_central_percent": float(jacobian_nonpositive_percent(flow, crop=1)),
        "ndv_percent": float(ndv),
    }


def _ixi_jac_metrics(flow: torch.Tensor, x_seg: torch.Tensor) -> dict[str, float]:
    return _fold_counts(flow, x_seg)


def _ixi_jac_log(row: dict) -> str:
    return f" j<=0%={row['j_leq0_percent']:.4f} ndv%={row['ndv_percent']:.4f}"


def _oasis_jac_metrics(flow: torch.Tensor, x_seg: torch.Tensor) -> dict[str, float]:
    # sdlogj stays first and unchanged: every published OASIS table is keyed on it.
    return {"sdlogj": float(logdet_std_from_flow(flow)), **_fold_counts(flow, x_seg)}


def _oasis_jac_log(row: dict) -> str:
    return f" sdlogj={row['sdlogj']:.4f} j<=0%={row['j_leq0_percent']:.4f}"


@dataclass
class MetricProfile:
    """Per-dataset evaluation protocol: VOI labels plus Jacobian metric/log functions."""

    labels: tuple[int, ...]
    jac_metrics: Callable[[torch.Tensor, torch.Tensor], dict[str, float]]
    jac_log: Callable[[dict], str]


_PROFILES = {
    "IXI": MetricProfile(IXI_VOI_LABELS, _ixi_jac_metrics, _ixi_jac_log),
    "OASIS": MetricProfile(OASIS_VOI_LABELS, _oasis_jac_metrics, _oasis_jac_log),
}


def metric_profile_for(ds_key: str) -> MetricProfile:
    """Return the evaluation protocol for a dataset key."""
    if ds_key not in _PROFILES:
        raise ValueError(f"No metric profile for dataset '{ds_key}'.")
    return _PROFILES[ds_key]


def write_trace(rows: list[dict], out_dir: str) -> None:
    """Persist per-case metrics at intermediate TTO steps, plus the per-step mean curve."""
    if not rows:
        return

    trace_path = os.path.join(out_dir, "tto_trace.csv")
    with open(trace_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    steps = sorted({int(r["tto_step"]) for r in rows})
    metric_keys = [k for k in rows[0] if k not in ("case_id", "tto_step")]
    curve_path = os.path.join(out_dir, "tto_curve.csv")
    with open(curve_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["tto_step", "n_cases", *[f"{k}_mean" for k in metric_keys]])
        for s in steps:
            at_step = [r for r in rows if int(r["tto_step"]) == s]
            means = [float(np.mean([r[k] for r in at_step])) for k in metric_keys]
            writer.writerow([s, len(at_step), *[f"{m:.6f}" for m in means]])

    print(f"[SAVED] tto trace: {trace_path}")
    print(f"[SAVED] tto curve: {curve_path}")


def write_results(
    rows: list[dict],
    out_dir: str,
    model_name: str,
    ckpt_path: str,
    test_dir: str,
) -> None:
    """Persist per-case rows and aggregated mean/std/sem/ci95 summaries to CSV/JSON."""
    if not rows:
        raise RuntimeError("No inference rows produced; check dataset/checkpoint setup.")

    header = list(rows[0].keys())
    per_case_path = os.path.join(out_dir, "per_case.csv")
    with open(per_case_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        writer.writerows(rows)

    n = len(rows)
    metric_keys = [k for k in header if k != "case_id" and not k.startswith("dice_lbl_")]
    metrics = {}
    for k in metric_keys:
        arr = np.array([r[k] for r in rows], dtype=np.float64)
        mean = float(arr.mean())
        std = float(arr.std(ddof=1)) if n > 1 else 0.0
        sem = std / math.sqrt(n) if n > 1 else 0.0
        metrics[k] = {"mean": mean, "std": std, "sem": sem, "ci95": 1.96 * sem}

    summary = {
        "model": model_name,
        "ckpt_path": ckpt_path,
        "test_dir": test_dir,
        "n_cases": n,
        "metrics": metrics,
    }
    summary_json = os.path.join(out_dir, "summary.json")
    summary_csv = os.path.join(out_dir, "summary.csv")
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    with open(summary_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "mean", "std", "sem", "ci95"])
        for k, v in metrics.items():
            writer.writerow([k, f"{v['mean']:.6f}", f"{v['std']:.6f}", f"{v['sem']:.6f}", f"{v['ci95']:.6f}"])

    print(f"\n[SAVED] per-case: {per_case_path}")
    print(f"[SAVED] summary: {summary_json}")
    print(f"[SAVED] summary: {summary_csv}")
