"""Post-training working margin for the actual stored effective map."""

from __future__ import annotations

import math
from typing import Any

import torch

from utils.cert_exact import certify_flow_exact
from utils.field import trilinear_cert_bound

WORK_MARGIN_POLICY = {
    "schema": "ctcf-stage5-source-work-margin-v1",
    "claim_epsilon": "0.001",
    "nominal_work_epsilon": 0.0011,
    "selection": "min_nominal_fast_bound_exact_lower",
    "acceptance": "exact_certificate_of_saved_fp32_at_claim_or_byte_identical_source",
}


def margin_from_bounds(fast_bound: float, exact_lower: float | None = None) -> float:
    """Keep the nominal margin when the source affords it, else use the source's verified bound.

    The local clip keeps every Bernstein coefficient >= work_eps in float64, so the full distance
    between work_eps and the claim absorbs float32 rounding of the output. Lowering work_eps only as
    far as the source requires keeps that distance as large as the source permits.
    """
    claim = float(WORK_MARGIN_POLICY["claim_epsilon"])
    nominal = float(WORK_MARGIN_POLICY["nominal_work_epsilon"])
    if not math.isfinite(fast_bound):
        raise RuntimeError("Stage5 source working margin has a non-finite bound")
    if fast_bound >= nominal:
        return nominal
    if exact_lower is None or not math.isfinite(exact_lower):
        raise RuntimeError("Stage5 source working margin requires a finite exact lower bound")
    epsilon = min(fast_bound, exact_lower)
    if not epsilon > claim:
        raise RuntimeError("Stage5 certified source has no verified working margin above the claim")
    return epsilon


def select_work_margin(source: torch.Tensor) -> dict[str, Any]:
    nominal = float(WORK_MARGIN_POLICY["nominal_work_epsilon"])
    fast = trilinear_cert_bound(source)
    exact_lower = None
    if math.isfinite(fast) and fast < nominal:
        exact = certify_flow_exact(source, eps=WORK_MARGIN_POLICY["claim_epsilon"])
        if exact.get("status") != "CERTIFIED" or exact.get("certified") is not True:
            raise RuntimeError("Stage5 source failed exact certification before selecting its working margin")
        exact_lower = float(exact["interval_lo_min"])
    return {
        "policy": WORK_MARGIN_POLICY["schema"],
        "nominal_work_eps": nominal,
        "source_fast_bound": fast,
        "source_exact_lower": exact_lower,
        "selected_work_eps": margin_from_bounds(fast, exact_lower),
    }
