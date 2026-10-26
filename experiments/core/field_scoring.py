"""Per-case FF and REP rows, with the columns of experiments.inference, for fields that do not come from InferRunner.

A legacy field is a stored CTCF flow sampled by SpatialTransformer (N - 1 normalization, align_corners=False); a
canonical field is sampled exactly at x + u(x). Certificates concern the stored array's trilinear interpolant.
"""

from __future__ import annotations

import time

import numpy as np
import torch

from experiments.core.inference_metrics import (
    digital_projection_columns,
    geometry_columns,
    metric_profile_for,
    trilinear_projection_columns,
)
from utils.dice import dice_per_label
from utils.field import _warp, digital_project, trilinear_cert_bound, trilinear_project
from utils.spatial import SpatialTransformer

CONVENTIONS = ("legacy", "canonical")
REPAIR = {"project_eps": 0.0, "tri_eps": 0.001, "damp": 0.6, "iters": 80}
PROFILE = metric_profile_for("OASIS")
_SAMPLERS: dict[tuple, SpatialTransformer] = {}


def repair_flags() -> str:
    """The experiments.inference flags of the same repair chain."""
    return (
        f"--tto_project 1 --tto_project_eps {REPAIR['project_eps']} --tto_project_damp {REPAIR['damp']} "
        f"--tto_project_iters {REPAIR['iters']} --tto_tri_project 1 --tto_tri_project_eps {REPAIR['tri_eps']} "
        f"--tto_tri_project_damp {REPAIR['damp']} --tto_tri_project_iters {REPAIR['iters']}"
    )


def warp_labels(seg: torch.Tensor, flow: torch.Tensor, convention: str) -> torch.Tensor:
    if convention == "canonical":
        return _warp(seg.float(), flow, mode="nearest")
    if convention != "legacy":
        raise ValueError(f"convention must be one of {CONVENTIONS}, got {convention!r}")
    key = (tuple(flow.shape[2:]), flow.device)
    if key not in _SAMPLERS:
        _SAMPLERS[key] = SpatialTransformer(key[0], mode="nearest").to(flow.device)
    return _SAMPLERS[key](seg.float(), flow.float())


def score(flow, case_id, seconds, x_seg, y_seg, eps, convention, extra=None) -> dict:
    """One row for moving labels x_seg and fixed labels y_seg; ``extra`` columns precede the per-label Dice."""
    with torch.no_grad():
        warped = warp_labels(x_seg, flow, convention).long()
        dice = dice_per_label(warped, y_seg.long(), labels=PROFILE.labels)
    row = {
        "case_id": case_id,
        "time_sec": seconds,
        **geometry_columns(flow, trilinear_cert_bound(flow, 0, eps), eps),
        "dice_mean": float(np.mean(dice)),
        **PROFILE.jac_metrics(flow, x_seg),
        **(extra or {}),
    }
    row.update({f"dice_lbl_{label}": float(v) for label, v in zip(PROFILE.labels, dice, strict=True)})
    return row


def repair(flow, fixed_mask=None, fixed_values=None) -> tuple[torch.Tensor, dict]:
    repaired, folds, iters = digital_project(
        flow,
        eps=REPAIR["project_eps"],
        damp=REPAIR["damp"],
        max_iters=REPAIR["iters"],
        fixed_mask=fixed_mask,
        fixed_values=fixed_values,
    )
    repaired, report = trilinear_project(
        repaired,
        eps=REPAIR["tri_eps"],
        damp=REPAIR["damp"],
        max_iters=REPAIR["iters"],
        subdiv_depth=0,
        fixed_mask=fixed_mask,
        fixed_values=fixed_values,
    )
    return repaired, {**digital_projection_columns(folds, iters), **trilinear_projection_columns(report)}


def score_ff_rep(flow, case_id, seconds, x_seg, y_seg, convention) -> tuple[dict, dict]:
    """FF and REP rows of one field; the REP time includes the forward time ``seconds``."""
    ff = score(flow, case_id, seconds, x_seg, y_seg, 0.0, convention)
    t0 = time.perf_counter()
    repaired, columns = repair(flow)
    elapsed = seconds + time.perf_counter() - t0
    return ff, score(repaired, case_id, elapsed, x_seg, y_seg, REPAIR["tri_eps"], convention, columns)
