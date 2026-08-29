from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from functools import reduce
from operator import and_

import torch

from tools.analysis.search.cost_volume import (
    STANDARDIZATION_FLOOR,
    match_postprocessed_rms,
    postprocess_residual,
    standardize_candidate_costs,
)
from tools.analysis.search.transaction import mind_ssc, sample_at_psi, voxel_grid_like

Offset = tuple[int, int, int]

MIND_RADIUS = 1
MIND_DILATIONS = (1, 2, 4)
C4_COMMON_COLLAR_WIDTH = 7

# These tables are protocol data. Their order is lexicographic z, then y, then x.
OFFSETS_STRIDE1: tuple[Offset, ...] = (
    (-1, -1, -1),
    (-1, -1, 0),
    (-1, -1, 1),
    (-1, 0, -1),
    (-1, 0, 0),
    (-1, 0, 1),
    (-1, 1, -1),
    (-1, 1, 0),
    (-1, 1, 1),
    (0, -1, -1),
    (0, -1, 0),
    (0, -1, 1),
    (0, 0, -1),
    (0, 0, 0),
    (0, 0, 1),
    (0, 1, -1),
    (0, 1, 0),
    (0, 1, 1),
    (1, -1, -1),
    (1, -1, 0),
    (1, -1, 1),
    (1, 0, -1),
    (1, 0, 0),
    (1, 0, 1),
    (1, 1, -1),
    (1, 1, 0),
    (1, 1, 1),
)

OFFSETS_STRIDE2: tuple[Offset, ...] = (
    (-2, -2, -2),
    (-2, -2, 0),
    (-2, -2, 2),
    (-2, 0, -2),
    (-2, 0, 0),
    (-2, 0, 2),
    (-2, 2, -2),
    (-2, 2, 0),
    (-2, 2, 2),
    (0, -2, -2),
    (0, -2, 0),
    (0, -2, 2),
    (0, 0, -2),
    (0, 0, 0),
    (0, 0, 2),
    (0, 2, -2),
    (0, 2, 0),
    (0, 2, 2),
    (2, -2, -2),
    (2, -2, 0),
    (2, -2, 2),
    (2, 0, -2),
    (2, 0, 0),
    (2, 0, 2),
    (2, 2, -2),
    (2, 2, 0),
    (2, 2, 2),
)

OFFSETS_STRIDE3: tuple[Offset, ...] = tuple((3 * z, 3 * y, 3 * x) for z, y, x in OFFSETS_STRIDE1)
OFFSETS_STRIDE4: tuple[Offset, ...] = tuple((4 * z, 4 * y, 4 * x) for z, y, x in OFFSETS_STRIDE1)

MAIN_ARM_IDS = (
    "mind_d1_s1",
    "mind_d2_s1",
    "mind_d4_s1",
    "mind_f124_s1",
    "mind_d1_s2",
    "mind_d2_s2",
    "mind_d4_s2",
    "mind_f124_s2",
)
DIAGNOSTIC_ARM_IDS = (
    "legacy_mind_d2_s1_collar4",
    "mind_f222_s1",
    "intensity_s1",
    "intensity_s2",
)


@dataclass(frozen=True, slots=True)
class WorkEstimate:
    descriptor_evaluations: int
    candidate_samples: int
    feature_channel_comparisons: int

    def __post_init__(self) -> None:
        if min(self.descriptor_evaluations, self.candidate_samples, self.feature_channel_comparisons) < 0:
            raise ValueError("work counters must be non-negative")

    def __add__(self, other: WorkEstimate) -> WorkEstimate:
        if not isinstance(other, WorkEstimate):
            return NotImplemented
        return WorkEstimate(
            descriptor_evaluations=self.descriptor_evaluations + other.descriptor_evaluations,
            candidate_samples=self.candidate_samples + other.candidate_samples,
            feature_channel_comparisons=self.feature_channel_comparisons + other.feature_channel_comparisons,
        )


@dataclass(frozen=True, slots=True)
class TimedWork:
    stage_id: str
    elapsed_seconds: float
    work: WorkEstimate

    def __post_init__(self) -> None:
        if not self.stage_id:
            raise ValueError("stage_id must be non-empty")
        if not math.isfinite(self.elapsed_seconds) or self.elapsed_seconds < 0.0:
            raise ValueError("elapsed_seconds must be finite and non-negative")


@dataclass(frozen=True, slots=True)
class MindFeaturePair:
    feature_id: str
    radius: int
    dilation: int
    fixed: torch.Tensor
    moving: torch.Tensor
    work: WorkEstimate


@dataclass(frozen=True, slots=True)
class RawCostVolume:
    cost_id: str
    costs: torch.Tensor
    valid: torch.Tensor
    offsets: tuple[Offset, ...]
    work: WorkEstimate


@dataclass(frozen=True, slots=True)
class CenteredCostVolume:
    cost_id: str
    standardized_costs: torch.Tensor
    valid: torch.Tensor
    offsets: tuple[Offset, ...]
    cost_mean: torch.Tensor
    cost_std: torch.Tensor
    floor_hit: torch.Tensor
    source_ids: tuple[str, ...]
    work: WorkEstimate


@dataclass(frozen=True, slots=True)
class PosteriorVolume:
    cost_id: str
    probabilities: torch.Tensor
    entropy: torch.Tensor
    confidence: torch.Tensor
    valid: torch.Tensor
    offsets: tuple[Offset, ...]
    temperature: float


@dataclass(frozen=True, slots=True)
class DecodedProposal:
    cost_id: str
    displacement: torch.Tensor
    offsets: tuple[Offset, ...]


@dataclass(frozen=True, slots=True)
class PostprocessedProposal:
    displacement: torch.Tensor
    proposal_multiplier: float
    smoothing_passes: int
    collar_width: int
    rms_scale_factor: float
    source_rms: float
    target_rms: float | None
    output_rms: float


@dataclass(frozen=True, slots=True)
class C4CommonSupport:
    geometry_mask: torch.Tensor
    fixed_descriptor_support: torch.Tensor
    sampled_descriptor_support: torch.Tensor
    common_mask: torch.Tensor
    geometry_count: int
    common_count: int
    retention: float


@dataclass(frozen=True, slots=True)
class DuplicateFusionDiagnostic:
    source_id: str
    fusion_id: str
    active_voxel_count: int
    max_abs_standardized_difference: float
    argmin_agreement: float


@dataclass(frozen=True, slots=True)
class ScaleAgreementDiagnostics:
    left_id: str
    right_id: str
    active_voxel_count: int
    argmin_agreement: float
    left_entropy_mean: float
    right_entropy_mean: float
    left_top1_top2_gap_mean: float
    right_top1_top2_gap_mean: float
    posterior_js_divergence_mean: float
    posterior_cosine_mean: float
    residual_cosine_mean: float | None
    residual_cosine_voxel_count: int


def offsets_for_stride(stride: int) -> tuple[Offset, ...]:
    if stride == 1:
        return OFFSETS_STRIDE1
    if stride == 2:
        return OFFSETS_STRIDE2
    if stride == 3:
        return OFFSETS_STRIDE3
    if stride == 4:
        return OFFSETS_STRIDE4
    raise ValueError("candidate stride must be one of 1, 2, 3 or 4")


def quadratic_center_log_prior(
    offsets: Sequence[Offset],
    *,
    beta: float,
) -> tuple[float, ...]:
    """Return ``-beta * ||offset||^2 / stride^2`` in candidate order."""

    frozen = _validate_offsets(offsets)
    if isinstance(beta, bool) or not isinstance(beta, (int, float)):
        raise TypeError("centre-prior beta must be a real scalar")
    beta_value = float(beta)
    if not math.isfinite(beta_value) or beta_value < 0.0:
        raise ValueError("centre-prior beta must be finite and non-negative")
    stride = max(abs(value) for offset in frozen for value in offset)
    if stride < 1:
        raise ValueError("centre-prior offsets must include a non-zero candidate")
    denominator = float(stride * stride)
    return tuple(-beta_value * sum(value * value for value in offset) / denominator for offset in frozen)


def descriptor_support_margin(*, radius: int, dilation: int) -> int:
    if isinstance(radius, bool) or not isinstance(radius, int) or radius < 0:
        raise ValueError("descriptor radius must be a non-negative integer")
    if isinstance(dilation, bool) or not isinstance(dilation, int) or dilation < 1:
        raise ValueError("descriptor dilation must be a positive integer")
    return radius + dilation


def _require_field(field: torch.Tensor, name: str = "field") -> None:
    if field.ndim != 5 or field.shape[0] != 1 or field.shape[1] != 3:
        raise ValueError(f"{name} must have shape [1,3,D,H,W], got {tuple(field.shape)}")
    if not field.is_floating_point():
        raise TypeError(f"{name} must use a floating dtype")


def _require_scalar_pair(fixed: torch.Tensor, moving: torch.Tensor) -> None:
    if fixed.ndim != 5 or fixed.shape[0] != 1 or fixed.shape[1] != 1:
        raise ValueError(f"fixed must have shape [1,1,D,H,W], got {tuple(fixed.shape)}")
    if moving.shape != fixed.shape:
        raise ValueError("moving must match fixed shape")
    if fixed.device != moving.device or fixed.dtype != moving.dtype or not fixed.is_floating_point():
        raise ValueError("fixed and moving must share a floating dtype and device")


def _validate_offsets(offsets: Sequence[Offset]) -> tuple[Offset, ...]:
    frozen = tuple(tuple(item) for item in offsets)
    if len(frozen) != 27 or len(set(frozen)) != 27:
        raise ValueError("an explicit cost volume must contain 27 unique offsets")
    if any(
        len(offset) != 3 or any(isinstance(value, bool) or not isinstance(value, int) for value in offset)
        for offset in frozen
    ):
        raise TypeError("offsets must be integer z-y-x triples")
    if (0, 0, 0) not in frozen:
        raise ValueError("offsets must include the zero displacement")
    return frozen


def _require_mask(mask: torch.Tensor, spatial: tuple[int, int, int], device: torch.device) -> None:
    if mask.shape != (1, 1, *spatial) or mask.dtype != torch.bool or mask.device != device:
        raise ValueError("mask must be boolean [1,1,D,H,W] on the data device")


def collar_mask_like(field: torch.Tensor, *, width: int = C4_COMMON_COLLAR_WIDTH) -> torch.Tensor:
    _require_field(field)
    spatial = tuple(field.shape[-3:])
    if isinstance(width, bool) or not isinstance(width, int) or width < 1 or min(spatial) <= 2 * width:
        raise ValueError(f"invalid collar width {width} for spatial shape {spatial}")
    mask = torch.zeros((1, 1, *spatial), dtype=torch.bool, device=field.device)
    mask[:, :, width:-width, width:-width, width:-width] = True
    return mask


def fixed_descriptor_support_mask(field: torch.Tensor, *, margin: int) -> torch.Tensor:
    _require_field(field)
    spatial = tuple(field.shape[-3:])
    if isinstance(margin, bool) or not isinstance(margin, int) or margin < 0 or min(spatial) <= 2 * margin:
        raise ValueError(f"invalid descriptor margin {margin} for spatial shape {spatial}")
    mask = torch.zeros((1, 1, *spatial), dtype=torch.bool, device=field.device)
    if margin == 0:
        mask.fill_(True)
    else:
        mask[:, :, margin:-margin, margin:-margin, margin:-margin] = True
    return mask


def sampled_descriptor_support_mask(
    psi_displacement: torch.Tensor,
    offset: Offset,
    *,
    margin: int,
) -> torch.Tensor:
    _require_field(psi_displacement, "psi_displacement")
    if margin < 0:
        raise ValueError("descriptor margin must be non-negative")
    coordinates = voxel_grid_like(psi_displacement) + psi_displacement
    coordinates = coordinates + psi_displacement.new_tensor(offset).view(1, 3, 1, 1, 1)
    upper = coordinates.new_tensor(psi_displacement.shape[-3:]).view(1, 3, 1, 1, 1) - 1.0 - margin
    return ((coordinates >= float(margin)) & (coordinates <= upper)).all(dim=1, keepdim=True)


def build_c4_common_support(
    psi_displacement: torch.Tensor,
    geometry_mask: torch.Tensor | None = None,
    *,
    collar_width: int = C4_COMMON_COLLAR_WIDTH,
    radius: int = MIND_RADIUS,
    max_dilation: int = max(MIND_DILATIONS),
    offsets: Sequence[Offset] = OFFSETS_STRIDE2,
) -> C4CommonSupport:
    """Freeze fair support for all C4 descriptor scales and displacement strides."""

    _require_field(psi_displacement, "psi_displacement")
    frozen_offsets = _validate_offsets(offsets)
    spatial = tuple(psi_displacement.shape[-3:])
    collar = collar_mask_like(psi_displacement, width=collar_width)
    if geometry_mask is None:
        geometry = collar
    else:
        _require_mask(geometry_mask, spatial, psi_displacement.device)
        geometry = geometry_mask & collar
    margin = descriptor_support_margin(radius=radius, dilation=max_dilation)
    fixed_support = fixed_descriptor_support_mask(psi_displacement, margin=margin)
    sampled_per_offset = tuple(
        sampled_descriptor_support_mask(psi_displacement, offset, margin=margin) for offset in frozen_offsets
    )
    sampled_support = reduce(and_, sampled_per_offset)
    common = geometry & fixed_support & sampled_support
    geometry_count = int(geometry.sum(dtype=torch.int64).item())
    common_count = int(common.sum(dtype=torch.int64).item())
    if geometry_count == 0:
        raise ValueError("C4 geometry mask is empty")
    if common_count == 0:
        raise ValueError("C4 common descriptor support is empty")
    return C4CommonSupport(
        geometry_mask=geometry,
        fixed_descriptor_support=fixed_support,
        sampled_descriptor_support=sampled_support,
        common_mask=common,
        geometry_count=geometry_count,
        common_count=common_count,
        retention=common_count / geometry_count,
    )


def build_raw_feature_cost_volume(
    fixed_feature: torch.Tensor,
    moving_feature: torch.Tensor,
    psi_displacement: torch.Tensor,
    geometry_mask: torch.Tensor,
    *,
    offsets: Sequence[Offset],
    support_margin: int,
    cost_id: str,
    descriptor_evaluations: int = 0,
) -> RawCostVolume:
    """Build one explicit-offset cost bank from precomputed feature tensors.

    ``descriptor_evaluations`` records work performed by the caller specifically for
    this bank. C4 normally computes each MIND feature pair once and reuses it for both
    search strides, so shared descriptor work is reported separately and this value is
    zero.
    """

    frozen_offsets = _validate_offsets(offsets)
    _require_field(psi_displacement, "psi_displacement")
    spatial = tuple(psi_displacement.shape[-3:])
    if fixed_feature.ndim != 5 or fixed_feature.shape[0] != 1 or tuple(fixed_feature.shape[-3:]) != spatial:
        raise ValueError("fixed feature must have shape [1,C,D,H,W] on the Psi grid")
    if moving_feature.shape != fixed_feature.shape:
        raise ValueError("moving feature must match fixed feature")
    if (
        fixed_feature.device != psi_displacement.device
        or fixed_feature.dtype != psi_displacement.dtype
        or moving_feature.device != fixed_feature.device
        or moving_feature.dtype != fixed_feature.dtype
    ):
        raise ValueError("features and Psi must share dtype and device")
    if not fixed_feature.is_floating_point():
        raise TypeError("features must use a floating dtype")
    _require_mask(geometry_mask, spatial, fixed_feature.device)
    if not bool(torch.isfinite(fixed_feature).all()) or not bool(torch.isfinite(moving_feature).all()):
        raise ValueError("feature tensors must be finite")

    fixed_support = fixed_descriptor_support_mask(psi_displacement, margin=support_margin)
    costs = fixed_feature.new_zeros((1, 27, *spatial))
    valid = torch.zeros_like(costs, dtype=torch.bool)
    with torch.inference_mode():
        for index, offset in enumerate(frozen_offsets):
            candidate_valid = geometry_mask & fixed_support
            candidate_valid &= sampled_descriptor_support_mask(psi_displacement, offset, margin=support_margin)
            sampled = sample_at_psi(moving_feature, psi_displacement, offset)
            candidate_cost = (fixed_feature - sampled).square().mean(dim=1, keepdim=True)
            candidate_cost = torch.where(candidate_valid, candidate_cost, torch.zeros_like(candidate_cost))
            costs[:, index : index + 1].copy_(candidate_cost)
            valid[:, index : index + 1].copy_(candidate_valid)
    if not bool(valid.any()):
        raise ValueError(f"{cost_id} has no valid descriptor comparison")
    if not bool(torch.isfinite(costs.masked_select(valid)).all()):
        raise ValueError(f"{cost_id} produced a non-finite valid cost")
    candidate_samples = int(valid.sum(dtype=torch.int64).item())
    work = WorkEstimate(
        descriptor_evaluations=descriptor_evaluations,
        candidate_samples=candidate_samples,
        feature_channel_comparisons=candidate_samples * fixed_feature.shape[1],
    )
    return RawCostVolume(cost_id=cost_id, costs=costs, valid=valid, offsets=frozen_offsets, work=work)


def build_mind_feature_pair(
    fixed: torch.Tensor,
    moving: torch.Tensor,
    *,
    dilation: int,
    radius: int = MIND_RADIUS,
    feature_id: str | None = None,
) -> MindFeaturePair:
    """Compute one MIND pair for reuse across multiple explicit search reaches."""

    _require_scalar_pair(fixed, moving)
    descriptor_support_margin(radius=radius, dilation=dilation)
    with torch.inference_mode():
        fixed_mind = mind_ssc(fixed, radius=radius, dilation=dilation)
        moving_mind = mind_ssc(moving, radius=radius, dilation=dilation)
    identifier = feature_id or f"mind_r{radius}_d{dilation}"
    return MindFeaturePair(
        feature_id=identifier,
        radius=radius,
        dilation=dilation,
        fixed=fixed_mind,
        moving=moving_mind,
        work=WorkEstimate(descriptor_evaluations=2, candidate_samples=0, feature_channel_comparisons=0),
    )


def build_raw_mind_cost_volume_from_features(
    features: MindFeaturePair,
    psi_displacement: torch.Tensor,
    geometry_mask: torch.Tensor,
    *,
    offsets: Sequence[Offset],
    cost_id: str | None = None,
) -> RawCostVolume:
    """Build one MIND cost bank while preserving shared-feature provenance."""

    if not isinstance(features, MindFeaturePair):
        raise TypeError("features must be a MindFeaturePair")
    stride = max(max(abs(value) for value in offset) for offset in offsets)
    identifier = cost_id or f"{features.feature_id}_s{stride}"
    return build_raw_feature_cost_volume(
        features.fixed,
        features.moving,
        psi_displacement,
        geometry_mask,
        offsets=offsets,
        support_margin=descriptor_support_margin(radius=features.radius, dilation=features.dilation),
        cost_id=identifier,
    )


def build_raw_mind_cost_volume(
    fixed: torch.Tensor,
    moving: torch.Tensor,
    psi_displacement: torch.Tensor,
    geometry_mask: torch.Tensor,
    *,
    dilation: int,
    offsets: Sequence[Offset],
    radius: int = MIND_RADIUS,
    cost_id: str | None = None,
) -> RawCostVolume:
    """Compute one target-centred MIND bank without retaining sampled features."""

    _require_scalar_pair(fixed, moving)
    _require_field(psi_displacement, "psi_displacement")
    if fixed.shape[-3:] != psi_displacement.shape[-3:]:
        raise ValueError("images and Psi must share a spatial grid")
    features = build_mind_feature_pair(fixed, moving, radius=radius, dilation=dilation)
    stride = max(max(abs(value) for value in offset) for offset in offsets)
    identifier = cost_id or f"mind_r{radius}_d{dilation}_s{stride}"
    raw = build_raw_mind_cost_volume_from_features(
        features,
        psi_displacement,
        geometry_mask,
        offsets=offsets,
        cost_id=identifier,
    )
    return RawCostVolume(
        cost_id=raw.cost_id,
        costs=raw.costs,
        valid=raw.valid,
        offsets=raw.offsets,
        work=raw.work + features.work,
    )


def build_raw_intensity_cost_volume(
    fixed_normalized: torch.Tensor,
    moving_normalized: torch.Tensor,
    psi_displacement: torch.Tensor,
    geometry_mask: torch.Tensor,
    *,
    offsets: Sequence[Offset],
    cost_id: str,
) -> RawCostVolume:
    """Build scalar SSD costs from separately normalized intensity volumes."""

    _require_scalar_pair(fixed_normalized, moving_normalized)
    return build_raw_feature_cost_volume(
        fixed_normalized,
        moving_normalized,
        psi_displacement,
        geometry_mask,
        offsets=offsets,
        support_margin=0,
        cost_id=cost_id,
        descriptor_evaluations=0,
    )


def centered_standardize(
    raw: RawCostVolume,
    *,
    standardization_floor: float = STANDARDIZATION_FLOOR,
) -> CenteredCostVolume:
    """Two-pass FP32 standardization; locally flat candidate sets leave support."""

    if not isinstance(raw, RawCostVolume):
        raise TypeError("raw must be a RawCostVolume")
    result = standardize_candidate_costs(
        raw.costs,
        raw.valid,
        mode="centered_two_pass_fp32",
        standardization_floor=standardization_floor,
    )
    informative = (result.valid_count >= 2) & ~result.floor_hit
    valid = raw.valid & informative
    if not bool(valid.any()):
        raise FloatingPointError(f"{raw.cost_id} is flat or unsupported at every voxel")
    standardized = torch.where(valid, result.standardized_costs, torch.zeros_like(result.standardized_costs))
    if not bool(torch.isfinite(standardized).all()):
        raise FloatingPointError(f"{raw.cost_id} standardized costs are non-finite")
    return CenteredCostVolume(
        cost_id=raw.cost_id,
        standardized_costs=standardized,
        valid=valid,
        offsets=raw.offsets,
        cost_mean=result.cost_mean,
        cost_std=result.cost_std,
        floor_hit=result.floor_hit,
        source_ids=(raw.cost_id,),
        work=raw.work,
    )


def fuse_standardized_costs(
    banks: Sequence[CenteredCostVolume],
    *,
    cost_id: str,
    standardization_floor: float = STANDARDIZATION_FLOOR,
) -> CenteredCostVolume:
    """Mean separately standardized banks, then center-standardize the fusion."""

    frozen = tuple(banks)
    if not frozen:
        raise ValueError("fusion requires at least one standardized bank")
    reference = frozen[0]
    for bank in frozen[1:]:
        if (
            bank.offsets != reference.offsets
            or bank.standardized_costs.shape != reference.standardized_costs.shape
            or bank.standardized_costs.device != reference.standardized_costs.device
            or bank.standardized_costs.dtype != reference.standardized_costs.dtype
        ):
            raise ValueError("fusion banks must share offsets, shape, dtype and device")
    valid = reduce(and_, (bank.valid for bank in frozen))
    mean = torch.stack([bank.standardized_costs for bank in frozen], dim=0).mean(dim=0)
    mean = torch.where(valid, mean, torch.zeros_like(mean))
    work = sum((bank.work for bank in frozen), WorkEstimate(0, 0, 0))
    raw = RawCostVolume(cost_id=cost_id, costs=mean, valid=valid, offsets=reference.offsets, work=work)
    fused = centered_standardize(raw, standardization_floor=standardization_floor)
    return CenteredCostVolume(
        cost_id=fused.cost_id,
        standardized_costs=fused.standardized_costs,
        valid=fused.valid,
        offsets=fused.offsets,
        cost_mean=fused.cost_mean,
        cost_std=fused.cost_std,
        floor_hit=fused.floor_hit,
        source_ids=tuple(bank.cost_id for bank in frozen),
        work=fused.work,
    )


def posterior_from_standardized_costs(
    volume: CenteredCostVolume,
    *,
    temperature: float = 1.0,
) -> PosteriorVolume:
    if not math.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("posterior temperature must be finite and positive")
    costs, valid = volume.standardized_costs, volume.valid
    if not bool(valid.any()):
        raise ValueError("posterior input has no valid candidate")
    if not bool(torch.isfinite(costs.masked_select(valid)).all()):
        raise ValueError("valid standardized costs must be finite")
    logits = -costs / float(temperature)
    negative_inf = torch.full_like(logits, -torch.inf)
    maximum = torch.where(valid, logits, negative_inf).amax(dim=1, keepdim=True)
    active = valid.any(dim=1, keepdim=True)
    maximum = torch.where(active, maximum, torch.zeros_like(maximum))
    weights = torch.where(valid, torch.exp(logits - maximum), torch.zeros_like(logits))
    normalizer = weights.sum(dim=1, keepdim=True)
    probabilities = torch.where(active, weights / normalizer.clamp_min(torch.finfo(costs.dtype).tiny), 0.0)
    log_probabilities = torch.where(
        probabilities > 0,
        probabilities.clamp_min(torch.finfo(costs.dtype).tiny).log(),
        torch.zeros_like(probabilities),
    )
    entropy = -(probabilities * log_probabilities).sum(dim=1, keepdim=True)
    count = valid.sum(dim=1, keepdim=True)
    confidence = torch.where(count > 1, 1.0 - entropy / count.to(costs.dtype).log(), active.to(costs.dtype))
    confidence = confidence.clamp(0.0, 1.0)
    for name, tensor in (("probabilities", probabilities), ("entropy", entropy), ("confidence", confidence)):
        if not bool(torch.isfinite(tensor).all()):
            raise FloatingPointError(f"posterior {name} became non-finite")
    return PosteriorVolume(
        cost_id=volume.cost_id,
        probabilities=probabilities,
        entropy=entropy,
        confidence=confidence,
        valid=valid,
        offsets=volume.offsets,
        temperature=float(temperature),
    )


def posterior_from_standardized_costs_with_prior(
    volume: CenteredCostVolume,
    *,
    beta: float,
    temperature: float = 1.0,
) -> PosteriorVolume:
    """Apply a stride-normalized quadratic centre prior to the C4 posterior."""

    log_prior_values = quadratic_center_log_prior(volume.offsets, beta=beta)
    if float(beta) == 0.0:
        return posterior_from_standardized_costs(volume, temperature=temperature)
    if not math.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("posterior temperature must be finite and positive")
    costs, valid = volume.standardized_costs, volume.valid
    if not bool(valid.any()):
        raise ValueError("posterior input has no valid candidate")
    if not bool(torch.isfinite(costs.masked_select(valid)).all()):
        raise ValueError("valid standardized costs must be finite")
    log_prior = costs.new_tensor(log_prior_values).view(1, -1, 1, 1, 1)
    logits = -costs / float(temperature) + log_prior
    negative_inf = torch.full_like(logits, -torch.inf)
    maximum = torch.where(valid, logits, negative_inf).amax(dim=1, keepdim=True)
    active = valid.any(dim=1, keepdim=True)
    maximum = torch.where(active, maximum, torch.zeros_like(maximum))
    weights = torch.where(valid, torch.exp(logits - maximum), torch.zeros_like(logits))
    normalizer = weights.sum(dim=1, keepdim=True)
    probabilities = torch.where(active, weights / normalizer.clamp_min(torch.finfo(costs.dtype).tiny), 0.0)
    log_probabilities = torch.where(
        probabilities > 0,
        probabilities.clamp_min(torch.finfo(costs.dtype).tiny).log(),
        torch.zeros_like(probabilities),
    )
    entropy = -(probabilities * log_probabilities).sum(dim=1, keepdim=True)
    count = valid.sum(dim=1, keepdim=True)
    confidence = torch.where(count > 1, 1.0 - entropy / count.to(costs.dtype).log(), active.to(costs.dtype))
    confidence = confidence.clamp(0.0, 1.0)
    for name, tensor in (("probabilities", probabilities), ("entropy", entropy), ("confidence", confidence)):
        if not bool(torch.isfinite(tensor).all()):
            raise FloatingPointError(f"posterior {name} became non-finite")
    return PosteriorVolume(
        cost_id=volume.cost_id,
        probabilities=probabilities,
        entropy=entropy,
        confidence=confidence,
        valid=valid,
        offsets=volume.offsets,
        temperature=float(temperature),
    )


def decode_posterior_mean(posterior: PosteriorVolume) -> DecodedProposal:
    offsets = posterior.probabilities.new_tensor(posterior.offsets)
    displacement = torch.einsum("bkdwh,kc->bcdwh", posterior.probabilities, offsets)
    if not bool(torch.isfinite(displacement).all()):
        raise FloatingPointError("decoded posterior mean is non-finite")
    return DecodedProposal(cost_id=posterior.cost_id, displacement=displacement, offsets=posterior.offsets)


def postprocess_and_match_rms(
    proposal: DecodedProposal | torch.Tensor,
    geometry_mask: torch.Tensor,
    *,
    proposal_multiplier: float,
    smoothing_passes: int,
    collar_width: int,
    rms_reference: torch.Tensor | None = None,
) -> PostprocessedProposal:
    displacement = proposal.displacement if isinstance(proposal, DecodedProposal) else proposal
    processed = postprocess_residual(
        displacement,
        scale=proposal_multiplier,
        post_smoothing_passes=smoothing_passes,
        collar_width=collar_width,
    )
    if rms_reference is None:
        from tools.analysis.search.cost_volume import masked_vector_rms

        output_rms = masked_vector_rms(processed, geometry_mask)
        return PostprocessedProposal(
            displacement=processed,
            proposal_multiplier=float(proposal_multiplier),
            smoothing_passes=smoothing_passes,
            collar_width=collar_width,
            rms_scale_factor=1.0,
            source_rms=output_rms,
            target_rms=None,
            output_rms=output_rms,
        )
    match = match_postprocessed_rms(processed, rms_reference, geometry_mask)
    return PostprocessedProposal(
        displacement=match.displacement,
        proposal_multiplier=float(proposal_multiplier),
        smoothing_passes=smoothing_passes,
        collar_width=collar_width,
        rms_scale_factor=match.scale_factor,
        source_rms=match.source_rms,
        target_rms=match.target_rms,
        output_rms=match.matched_rms,
    )


def _masked_gap(volume: CenteredCostVolume, mask: torch.Tensor) -> torch.Tensor:
    logits = torch.where(volume.valid, -volume.standardized_costs, -torch.inf)
    top2 = logits.topk(k=2, dim=1).values
    return (top2[:, 0:1] - top2[:, 1:2]).masked_select(mask)


def duplicate_fusion_diagnostic(
    source: CenteredCostVolume,
    fusion: CenteredCostVolume,
    geometry_mask: torch.Tensor,
) -> DuplicateFusionDiagnostic:
    if source.offsets != fusion.offsets or source.standardized_costs.shape != fusion.standardized_costs.shape:
        raise ValueError("source and fusion must share offsets and shape")
    _require_mask(geometry_mask, tuple(source.standardized_costs.shape[-3:]), source.standardized_costs.device)
    common = geometry_mask & source.valid.all(dim=1, keepdim=True) & fusion.valid.all(dim=1, keepdim=True)
    if not bool(common.any()):
        raise ValueError("duplicate-fusion diagnostic has empty common support")
    expanded = common.expand_as(source.standardized_costs)
    difference = (source.standardized_costs - fusion.standardized_costs).abs().masked_select(expanded)
    source_argmin = source.standardized_costs.masked_fill(~source.valid, torch.inf).argmin(dim=1, keepdim=True)
    fusion_argmin = fusion.standardized_costs.masked_fill(~fusion.valid, torch.inf).argmin(dim=1, keepdim=True)
    agreement = (source_argmin == fusion_argmin).masked_select(common).double().mean()
    return DuplicateFusionDiagnostic(
        source_id=source.cost_id,
        fusion_id=fusion.cost_id,
        active_voxel_count=int(common.sum(dtype=torch.int64).item()),
        max_abs_standardized_difference=float(difference.double().max().item()),
        argmin_agreement=float(agreement.item()),
    )


def scale_agreement_diagnostics(
    left: CenteredCostVolume,
    right: CenteredCostVolume,
    left_posterior: PosteriorVolume,
    right_posterior: PosteriorVolume,
    left_residual: torch.Tensor,
    right_residual: torch.Tensor,
    geometry_mask: torch.Tensor,
) -> ScaleAgreementDiagnostics:
    if left.offsets != right.offsets or left.standardized_costs.shape != right.standardized_costs.shape:
        raise ValueError("scale diagnostics require matching candidate tables and shapes")
    spatial = tuple(left.standardized_costs.shape[-3:])
    _require_mask(geometry_mask, spatial, left.standardized_costs.device)
    _require_field(left_residual, "left_residual")
    _require_field(right_residual, "right_residual")
    if left_residual.shape != right_residual.shape or tuple(left_residual.shape[-3:]) != spatial:
        raise ValueError("residuals must share the cost-volume grid")
    common_valid = left.valid & right.valid & left_posterior.valid & right_posterior.valid
    common = geometry_mask & common_valid.all(dim=1, keepdim=True)
    if not bool(common.any()):
        raise ValueError("scale-agreement support is empty")

    left_argmin = left.standardized_costs.masked_fill(~left.valid, torch.inf).argmin(dim=1, keepdim=True)
    right_argmin = right.standardized_costs.masked_fill(~right.valid, torch.inf).argmin(dim=1, keepdim=True)
    argmin_agreement = (left_argmin == right_argmin).masked_select(common).double().mean()
    left_entropy = left_posterior.entropy.masked_select(common).double().mean()
    right_entropy = right_posterior.entropy.masked_select(common).double().mean()

    p = left_posterior.probabilities.clamp_min(0.0)
    q = right_posterior.probabilities.clamp_min(0.0)
    midpoint = 0.5 * (p + q)
    tiny = torch.finfo(p.dtype).tiny
    kl_p = torch.where(p > 0, p * (p.clamp_min(tiny).log() - midpoint.clamp_min(tiny).log()), 0.0).sum(
        dim=1, keepdim=True
    )
    kl_q = torch.where(q > 0, q * (q.clamp_min(tiny).log() - midpoint.clamp_min(tiny).log()), 0.0).sum(
        dim=1, keepdim=True
    )
    js = 0.5 * (kl_p + kl_q)
    posterior_cosine = (p * q).sum(dim=1, keepdim=True) / (
        p.square().sum(dim=1, keepdim=True).sqrt() * q.square().sum(dim=1, keepdim=True).sqrt()
    ).clamp_min(tiny)

    left_norm = left_residual.square().sum(dim=1, keepdim=True).sqrt()
    right_norm = right_residual.square().sum(dim=1, keepdim=True).sqrt()
    nonzero = common & (left_norm > 0.0) & (right_norm > 0.0)
    residual_count = int(nonzero.sum(dtype=torch.int64).item())
    residual_cosine_mean: float | None = None
    if residual_count:
        cosine = (left_residual * right_residual).sum(dim=1, keepdim=True) / (left_norm * right_norm)
        residual_cosine_mean = float(cosine.masked_select(nonzero).double().mean().item())

    values = (
        argmin_agreement,
        left_entropy,
        right_entropy,
        _masked_gap(left, common).double().mean(),
        _masked_gap(right, common).double().mean(),
        js.masked_select(common).double().mean(),
        posterior_cosine.masked_select(common).double().mean(),
    )
    if not all(bool(torch.isfinite(value)) for value in values):
        raise FloatingPointError("scale-agreement diagnostics became non-finite")
    if residual_cosine_mean is not None and not math.isfinite(residual_cosine_mean):
        raise FloatingPointError("residual cosine became non-finite")
    return ScaleAgreementDiagnostics(
        left_id=left.cost_id,
        right_id=right.cost_id,
        active_voxel_count=int(common.sum(dtype=torch.int64).item()),
        argmin_agreement=float(values[0].item()),
        left_entropy_mean=float(values[1].item()),
        right_entropy_mean=float(values[2].item()),
        left_top1_top2_gap_mean=float(values[3].item()),
        right_top1_top2_gap_mean=float(values[4].item()),
        posterior_js_divergence_mean=float(values[5].item()),
        posterior_cosine_mean=float(values[6].item()),
        residual_cosine_mean=residual_cosine_mean,
        residual_cosine_voxel_count=residual_count,
    )
