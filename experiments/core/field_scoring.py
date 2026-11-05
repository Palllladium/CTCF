"""Per-case FF and REP rows, with the columns of experiments.inference, for fields that do not come from InferRunner.

A legacy field is a stored CTCF flow sampled by SpatialTransformer (N - 1 normalization, align_corners=False); a
canonical field is sampled exactly at x + u(x). Certificates concern the stored array's trilinear interpolant.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
from scipy.stats import t as student_t

from experiments.core.inference_metrics import (
    digital_projection_columns,
    geometry_columns,
    metric_profile_for,
    trilinear_projection_columns,
)
from tools.analysis.run_artifacts import atomic_write_json, sha256_file
from utils.cert_boundary import boundary_report, global_verdict
from utils.cert_exact import certify_flow_exact
from utils.dice import dice_per_label
from utils.field import (
    _warp,
    boundary_vertex_mask,
    digital_project,
    enforce_identity_boundary,
    identity_collar,
    logdet_std_from_flow,
    trilinear_cert_bound,
    trilinear_project,
)
from utils.field_io import load_flow_npz, save_flow_npz_atomic
from utils.spatial import SpatialTransformer

CONVENTIONS = ("legacy", "canonical")
REPAIR = {"project_eps": 0.0, "tri_eps": 0.001, "damp": 0.6, "iters": 80}
# The repair works above the exact claim, as in tools/runners/topology/collar.sh.
IDENTITY_BOUNDARY = {"collar": 4, "work_eps": 0.0011, "claim_eps": "0.001"}
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


def score(flow, case_id, seconds, x_seg, y_seg, eps, convention, extra=None, *, profile=PROFILE) -> dict:
    """One row for moving labels x_seg and fixed labels y_seg; ``extra`` columns precede the per-label Dice."""
    with torch.no_grad():
        warped = warp_labels(x_seg, flow, convention).long()
        dice = dice_per_label(warped, y_seg.long(), labels=profile.labels)
    row = {
        "case_id": case_id,
        "time_sec": seconds,
        **geometry_columns(flow, trilinear_cert_bound(flow, 0, eps), eps),
        "dice_mean": float(np.mean(dice)),
        **profile.jac_metrics(flow, x_seg),
        **(extra or {}),
    }
    row.update({f"dice_lbl_{label}": float(v) for label, v in zip(profile.labels, dice, strict=True)})
    if "sdlogj" not in row:
        row["sdlogj_plus3"] = logdet_std_from_flow(flow)
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


def repair_to_identity_boundary(flow) -> tuple[torch.Tensor, dict]:
    """Taper a free-boundary canonical field to an exact identity boundary and repair with that boundary fixed."""
    collared = identity_collar(flow.float(), width=IDENTITY_BOUNDARY["collar"])
    mask, zeros = boundary_vertex_mask(collared), torch.zeros_like(collared)
    repaired, folds, iters = digital_project(
        collared,
        eps=REPAIR["project_eps"],
        damp=REPAIR["damp"],
        max_iters=REPAIR["iters"],
        fixed_mask=mask,
        fixed_values=zeros,
    )
    repaired, report = trilinear_project(
        repaired,
        eps=IDENTITY_BOUNDARY["work_eps"],
        damp=REPAIR["damp"],
        max_iters=REPAIR["iters"],
        subdiv_depth=0,
        fixed_mask=mask,
        fixed_values=zeros,
    )
    return enforce_identity_boundary(repaired), {
        **digital_projection_columns(folds, iters),
        **trilinear_projection_columns(report),
    }


def score_ff_rep(flow, case_id, seconds, x_seg, y_seg, convention, *, profile=PROFILE) -> tuple[dict, dict]:
    """FF and REP rows of one field; the REP time includes the forward time ``seconds``."""
    ff = score(flow, case_id, seconds, x_seg, y_seg, 0.0, convention, profile=profile)
    t0 = time.perf_counter()
    repaired, columns = repair(flow)
    elapsed = seconds + time.perf_counter() - t0
    return ff, score(repaired, case_id, elapsed, x_seg, y_seg, REPAIR["tri_eps"], convention, columns, profile=profile)


def score_saved_field(
    flow,
    case_id,
    seconds,
    x_seg,
    y_seg,
    directory: Path,
    *,
    profile=PROFILE,
    repair_fn=None,
    repair_output=True,
    repair_eps=0.0011,
    certify_modes=("FF", "REP"),
    cache=True,
    repair_columns=True,
):
    """Persist canonical raw scores first, then repair and certify the exact saved arrays independently."""
    if cache and (directory / "report.json").exists():
        rows, report = load_saved_scores(directory)
        if report["certify_modes"] != list(certify_modes):
            raise ValueError("Cached pair uses another certification policy; use a new output directory")
        return rows, report
    directory.mkdir(parents=True, exist_ok=True)
    rows, report = (
        {},
        {
            "errors": [],
            "certify_modes": list(certify_modes),
            "certification": "SEE_PER_FIELD_REPORTS" if certify_modes else "NOT_RUN",
        },
    )

    def store(mode, field, elapsed, eps, extra=None):
        save_flow_npz_atomic(directory / f"{mode}.npz", field)
        rows[mode] = score(field, case_id, elapsed, x_seg, y_seg, eps, "canonical", extra, profile=profile)
        atomic_write_json(directory / "metrics.json", rows, allow_nan=False)

    store("FF", flow, seconds, 0.0)
    if repair_output:
        try:
            started = time.perf_counter()
            repaired, columns = (repair_fn or repair_to_identity_boundary)(flow)
            if flow.is_cuda:
                torch.cuda.synchronize(flow.device)
            report["repair_seconds"], report["repair"] = time.perf_counter() - started, columns
            store("REP", repaired, seconds + report["repair_seconds"], repair_eps, columns if repair_columns else None)
        except (RuntimeError, ValueError, OSError, FloatingPointError) as error:
            report["repair_error"] = repr(error)
            report["errors"].append(f"repair: {error!r}")
    for mode in (mode for mode in rows if mode in certify_modes):
        try:
            started = time.perf_counter()
            stored = load_flow_npz(directory / f"{mode}.npz")
            cell = certify_flow_exact(stored, eps=IDENTITY_BOUNDARY["claim_eps"])
            boundary = boundary_report(stored)
            verdict = global_verdict(cell, boundary)
            cell["global_homeomorphism_hypotheses_met"] = verdict["global_homeomorphism"]
            report[f"certificate_{mode}"], report[f"boundary_{mode}"], report[f"global_{mode}"] = (
                cell,
                boundary,
                verdict,
            )
            report[f"certificate_{mode}_seconds"] = time.perf_counter() - started
        except (RuntimeError, ValueError, OSError, FloatingPointError) as error:
            report[f"error_certificate_{mode}"] = repr(error)
            report["errors"].append(f"certificate_{mode}: {error!r}")
    report["global_homeomorphism"] = bool(report.get("global_REP", {}).get("global_homeomorphism", False))
    report["field_sha256"] = {mode: sha256_file(directory / f"{mode}.npz") for mode in rows}
    atomic_write_json(directory / "report.json", report)
    return rows, report


def load_saved_scores(directory):
    report = json.loads((directory / "report.json").read_text())
    for mode, digest in report["field_sha256"].items():
        if sha256_file(directory / f"{mode}.npz") != digest:
            raise ValueError(f"Cached {mode} field changed at {directory}")
    return json.loads((directory / "metrics.json").read_text()), report


def paired_dice_delta(rows, control):
    """Average reverse directions before comparing disjoint subject pairs; intervals are descriptive."""
    tables = [{r["case_id"]: float(r["dice_mean"]) for r in values} for values in (rows, control)]
    if (
        any(len(table) != len(values) for table, values in zip(tables, (rows, control), strict=True))
        or tables[0].keys() != tables[1].keys()
    ):
        raise ValueError("Paired comparison needs the same unique directed case IDs")
    pairs = {}
    for case, value in tables[0].items():
        subjects = case.split("__")
        if len(subjects) != 2 or subjects[0] == subjects[1]:
            raise ValueError("Expected moving__fixed subject case IDs")
        pairs.setdefault(tuple(sorted(subjects)), []).append(value - tables[1][case])
    if not pairs or any(len(values) != 2 for values in pairs.values()):
        raise ValueError("Paired comparison needs both directions of every pair")
    values = np.asarray([np.mean(v) for v in pairs.values()])
    mean = float(values.mean())
    half = (
        float(student_t.ppf(0.975, len(values) - 1) * values.std(ddof=1) / len(values) ** 0.5) if len(values) > 1 else 0
    )
    return {
        "pairs": len(values),
        "mean": mean,
        "median": float(np.median(values)),
        "wins": int((values > 0).sum()),
        "ci95": [mean - half, mean + half],
        "resource_signal": mean >= 0.005,
        "scope": "one seed; descriptive paired interval",
    }
