from __future__ import annotations

import math
import time
from dataclasses import dataclass

import torch

from tools.analysis.search.cost_volume import masked_vector_rms
from tools.analysis.search.multiscale import (
    CenteredCostVolume,
    DecodedProposal,
    PosteriorVolume,
    PostprocessedProposal,
    build_raw_intensity_cost_volume,
    centered_standardize,
    decode_posterior_mean,
    offsets_for_stride,
    posterior_from_standardized_costs_with_prior,
    postprocess_and_match_rms,
)
from tools.analysis.search.transaction import certified_local_clip_candidate


@dataclass(frozen=True, slots=True)
class IntensityReachBank:
    reach_id: str
    cost_id: str
    stride_voxels: int
    volume: CenteredCostVolume
    elapsed_sec: float
    generation_count: int
    raw_all_candidates_valid_count: int
    standardized_informative_count: int


@dataclass(frozen=True, slots=True)
class IntensityDirection:
    direction_id: str
    reach_id: str
    cost_id: str
    stride_voxels: int
    centre_beta: float
    posterior_temperature: float
    posterior: PosteriorVolume
    decoded: DecodedProposal


@dataclass(frozen=True, slots=True)
class IntensityCandidate:
    candidate_id: str
    direction_id: str
    candidate: torch.Tensor
    requested_displacement: torch.Tensor
    postprocessed: PostprocessedProposal
    post_rms_amplitude: float
    requested_rms: float
    realized_rms: float
    clip_rms_retention: float
    clip_cosine: float
    work_eps: float
    sweeps: int
    operator: dict[str, float | int | str]


def _require_id(value: str, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    return value


def _require_finite_positive(value: float, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{label} must be a real scalar")
    observed = float(value)
    if not math.isfinite(observed) or observed <= 0.0:
        raise ValueError(f"{label} must be finite and positive")
    return observed


def build_intensity_reach_bank(
    fixed_normalized: torch.Tensor,
    moving_normalized: torch.Tensor,
    initial: torch.Tensor,
    generation_mask: torch.Tensor,
    *,
    reach_id: str,
    cost_id: str,
    stride_voxels: int,
    standardization_floor: float,
    require_all_candidates_valid: bool = True,
) -> IntensityReachBank:
    """Build and center-standardize one sparse intensity cost bank."""

    frozen_reach_id = _require_id(reach_id, "reach_id")
    frozen_cost_id = _require_id(cost_id, "cost_id")
    if isinstance(stride_voxels, bool) or not isinstance(stride_voxels, int):
        raise TypeError("stride_voxels must be an integer")
    if not isinstance(require_all_candidates_valid, bool):
        raise TypeError("require_all_candidates_valid must be a bool")
    floor = _require_finite_positive(standardization_floor, "standardization_floor")

    started = time.perf_counter()
    raw = build_raw_intensity_cost_volume(
        fixed_normalized,
        moving_normalized,
        initial,
        generation_mask,
        offsets=offsets_for_stride(stride_voxels),
        cost_id=frozen_cost_id,
    )
    generation_count = int(generation_mask.sum(dtype=torch.int64).item())
    if generation_count <= 0:
        raise ValueError(f"{frozen_reach_id} generation support is empty")
    raw_all_valid = generation_mask & raw.valid.all(dim=1, keepdim=True)
    raw_all_valid_count = int(raw_all_valid.sum(dtype=torch.int64).item())
    if require_all_candidates_valid and raw_all_valid_count != generation_count:
        raise RuntimeError(
            f"{frozen_reach_id} is not fully valid on its generation support: {raw_all_valid_count}/{generation_count}"
        )

    volume = centered_standardize(raw, standardization_floor=floor)
    informative = generation_mask & volume.valid.all(dim=1, keepdim=True)
    informative_count = int(informative.sum(dtype=torch.int64).item())
    if not 0 < informative_count <= generation_count:
        raise RuntimeError(f"{frozen_reach_id} standardized informative support is empty or invalid")
    return IntensityReachBank(
        reach_id=frozen_reach_id,
        cost_id=frozen_cost_id,
        stride_voxels=stride_voxels,
        volume=volume,
        elapsed_sec=time.perf_counter() - started,
        generation_count=generation_count,
        raw_all_candidates_valid_count=raw_all_valid_count,
        standardized_informative_count=informative_count,
    )


def decode_intensity_direction(
    bank: IntensityReachBank,
    *,
    direction_id: str,
    centre_beta: float,
    posterior_temperature: float,
) -> IntensityDirection:
    """Decode one posterior-mean direction from an intensity bank."""

    if not isinstance(bank, IntensityReachBank):
        raise TypeError("bank must be an IntensityReachBank")
    frozen_direction_id = _require_id(direction_id, "direction_id")
    if isinstance(centre_beta, bool) or not isinstance(centre_beta, (int, float)):
        raise TypeError("centre_beta must be a real scalar")
    beta = float(centre_beta)
    if not math.isfinite(beta) or beta < 0.0:
        raise ValueError("centre_beta must be finite and non-negative")
    temperature = _require_finite_positive(posterior_temperature, "posterior_temperature")
    posterior = posterior_from_standardized_costs_with_prior(
        bank.volume,
        beta=beta,
        temperature=temperature,
    )
    decoded = decode_posterior_mean(posterior)
    return IntensityDirection(
        direction_id=frozen_direction_id,
        reach_id=bank.reach_id,
        cost_id=bank.cost_id,
        stride_voxels=bank.stride_voxels,
        centre_beta=beta,
        posterior_temperature=temperature,
        posterior=posterior,
        decoded=decoded,
    )


def _global_cosine(requested: torch.Tensor, realized: torch.Tensor, mask: torch.Tensor) -> float:
    expanded = mask.expand_as(requested)
    left = requested.masked_select(expanded).double()
    right = realized.masked_select(expanded).double()
    denominator = left.square().sum().sqrt() * right.square().sum().sqrt()
    if float(denominator.item()) == 0.0:
        return 0.0
    value = float((left * right).sum().div(denominator).item())
    if not math.isfinite(value):
        raise RuntimeError("intensity clipping cosine is non-finite")
    return min(1.0, max(-1.0, value))


def materialize_intensity_candidate(
    direction: IntensityDirection,
    initial: torch.Tensor,
    rms_reference: torch.Tensor,
    mask: torch.Tensor,
    *,
    candidate_id: str,
    pre_rms_multiplier: float,
    post_rms_amplitude: float,
    smoothing_passes: int,
    collar_width: int,
    work_eps: float,
    sweeps: int,
) -> IntensityCandidate:
    """Scale one fixed pre-clip direction and apply one certified clip call."""

    if not isinstance(direction, IntensityDirection):
        raise TypeError("direction must be an IntensityDirection")
    frozen_candidate_id = _require_id(candidate_id, "candidate_id")
    multiplier = _require_finite_positive(pre_rms_multiplier, "pre_rms_multiplier")
    amplitude = _require_finite_positive(post_rms_amplitude, "post_rms_amplitude")
    epsilon = _require_finite_positive(work_eps, "work_eps")
    for value, label in ((smoothing_passes, "smoothing_passes"), (collar_width, "collar_width")):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{label} must be a non-negative integer")
    if isinstance(sweeps, bool) or not isinstance(sweeps, int) or sweeps < 1:
        raise ValueError("sweeps must be a positive integer")

    postprocessed = postprocess_intensity_direction(
        direction,
        rms_reference,
        mask,
        pre_rms_multiplier=multiplier,
        smoothing_passes=smoothing_passes,
        collar_width=collar_width,
    )
    return materialize_postprocessed_intensity_candidate(
        direction,
        postprocessed,
        initial,
        mask,
        candidate_id=frozen_candidate_id,
        post_rms_amplitude=amplitude,
        work_eps=epsilon,
        sweeps=sweeps,
    )


def postprocess_intensity_direction(
    direction: IntensityDirection,
    rms_reference: torch.Tensor,
    mask: torch.Tensor,
    *,
    pre_rms_multiplier: float,
    smoothing_passes: int,
    collar_width: int,
) -> PostprocessedProposal:
    """Postprocess and RMS-match one direction exactly once for an amplitude family."""

    if not isinstance(direction, IntensityDirection):
        raise TypeError("direction must be an IntensityDirection")
    multiplier = _require_finite_positive(pre_rms_multiplier, "pre_rms_multiplier")
    for value, label in ((smoothing_passes, "smoothing_passes"), (collar_width, "collar_width")):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{label} must be a non-negative integer")
    return postprocess_and_match_rms(
        direction.decoded,
        mask,
        proposal_multiplier=multiplier,
        smoothing_passes=smoothing_passes,
        collar_width=collar_width,
        rms_reference=rms_reference,
    )


def materialize_postprocessed_intensity_candidate(
    direction: IntensityDirection,
    postprocessed: PostprocessedProposal,
    initial: torch.Tensor,
    mask: torch.Tensor,
    *,
    candidate_id: str,
    post_rms_amplitude: float,
    work_eps: float,
    sweeps: int,
) -> IntensityCandidate:
    """Apply amplitude and one certified clip call to a frozen postprocessed direction."""

    if not isinstance(direction, IntensityDirection):
        raise TypeError("direction must be an IntensityDirection")
    if not isinstance(postprocessed, PostprocessedProposal):
        raise TypeError("postprocessed must be a PostprocessedProposal")
    frozen_candidate_id = _require_id(candidate_id, "candidate_id")
    amplitude = _require_finite_positive(post_rms_amplitude, "post_rms_amplitude")
    epsilon = _require_finite_positive(work_eps, "work_eps")
    if isinstance(sweeps, bool) or not isinstance(sweeps, int) or sweeps < 1:
        raise ValueError("sweeps must be a positive integer")
    if postprocessed.target_rms is None:
        raise RuntimeError("intensity candidate requires an RMS reference")
    requested = (postprocessed.displacement * amplitude).float()
    requested_rms = masked_vector_rms(requested, mask)
    expected_rms = float(postprocessed.target_rms) * amplitude
    if (
        not math.isfinite(requested_rms)
        or requested_rms <= 0.0
        or not math.isfinite(expected_rms)
        or expected_rms <= 0.0
        or not math.isclose(requested_rms, expected_rms, rel_tol=1e-7, abs_tol=1e-8)
    ):
        raise RuntimeError(f"post-RMS amplitude stage changed: {frozen_candidate_id}")

    candidate, operator = certified_local_clip_candidate(
        initial,
        requested,
        mask,
        work_eps=epsilon,
        sweeps=sweeps,
    )
    realized = (candidate - initial).float()
    realized_rms = masked_vector_rms(realized, mask)
    raw_retention = realized_rms / requested_rms
    if not math.isfinite(raw_retention) or raw_retention < 0.0 or raw_retention > 1.0 + 1e-6:
        raise RuntimeError(f"clipping RMS retention is invalid: {frozen_candidate_id}")
    return IntensityCandidate(
        candidate_id=frozen_candidate_id,
        direction_id=direction.direction_id,
        candidate=candidate,
        requested_displacement=requested,
        postprocessed=postprocessed,
        post_rms_amplitude=amplitude,
        requested_rms=requested_rms,
        realized_rms=realized_rms,
        clip_rms_retention=min(1.0, raw_retention),
        clip_cosine=_global_cosine(requested, realized, mask),
        work_eps=epsilon,
        sweeps=sweeps,
        operator=dict(operator),
    )


__all__ = [
    "IntensityCandidate",
    "IntensityDirection",
    "IntensityReachBank",
    "build_intensity_reach_bank",
    "decode_intensity_direction",
    "materialize_intensity_candidate",
    "materialize_postprocessed_intensity_candidate",
    "postprocess_intensity_direction",
]
