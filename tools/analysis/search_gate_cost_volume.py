from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn.functional as F

from tools.analysis.transactional_search import OFFSETS, sample_at_psi, smooth_proposal, valid_sample_mask
from utils.field import identity_collar

MessagePassingMode = Literal["none", "isotropic", "adaptive"]
DecoderMode = Literal["confidence", "posterior_mean"]

MIND_CHANNELS = 12
STANDARDIZATION_FLOOR = 1e-6
LOGIT_MESSAGE_AXIS_KERNEL = (0.25, 0.5, 0.25)
POSTERIOR_DIAGNOSTICS_ID = "MASKED_27_OFFSET_LOGIT_POSTERIOR_GEOMETRY_V1"
RMS_FIRST_DIFFERENCE_ROUGHNESS_ID = "RMS_VECTOR_FIRST_DIFFERENCE_GEOMETRY_PAIRS_V1"


@dataclass(frozen=True)
class StandardizedCostVolume:
    """One scalar standardized MIND cost per candidate offset and target voxel."""

    standardized_costs: torch.Tensor
    valid: torch.Tensor
    valid_count: torch.Tensor
    cost_mean: torch.Tensor
    cost_std: torch.Tensor


@dataclass(frozen=True)
class PosteriorResult:
    probabilities: torch.Tensor
    entropy: torch.Tensor
    normalized_entropy: torch.Tensor
    confidence: torch.Tensor
    valid_count: torch.Tensor


@dataclass(frozen=True)
class MessagePassingResult:
    logits: torch.Tensor
    lambda_map: torch.Tensor
    lambda_mean: float
    mode: MessagePassingMode


@dataclass(frozen=True)
class DecoderResult:
    displacement: torch.Tensor
    posterior_mean: torch.Tensor
    entropy: torch.Tensor
    normalized_entropy: torch.Tensor
    confidence: torch.Tensor
    mode: DecoderMode


@dataclass(frozen=True)
class RMSMatchResult:
    displacement: torch.Tensor
    scale_factor: float
    source_rms: float
    target_rms: float
    matched_rms: float


@dataclass(frozen=True)
class PosteriorDiagnostics:
    """Case-level label-free summaries over the explicitly supplied geometry mask."""

    diagnostic_id: str
    active_voxel_count: int
    candidate_count: int
    top1_top2_valid_logit_gap_mean: float
    posterior_peak_probability_mean: float
    entropy_nats_mean: float
    invalid_offset_fraction: float
    posterior_mean_l2_norm_mean: float
    confidence_weighted_mean_l2_norm_mean: float
    confidence_to_mean_l2_norm_ratio: float | None


@dataclass(frozen=True)
class RMSFirstDifferenceRoughness:
    """RMS vector first difference over positive-axis adjacent pairs inside geometry."""

    metric_id: str
    rms_vector_first_difference: float
    pair_count: int
    axis_pair_counts_zyx: tuple[int, int, int]
    axis_rms_vector_first_difference_zyx: tuple[float | None, float | None, float | None]


def _require_feature_pair(fixed: torch.Tensor, moving: torch.Tensor) -> None:
    expected = "[1,12,D,H,W]"
    if fixed.dim() != 5 or fixed.shape[0] != 1 or fixed.shape[1] != MIND_CHANNELS:
        raise ValueError(f"fixed MIND features must have shape {expected}, got {tuple(fixed.shape)}")
    if moving.shape != fixed.shape:
        raise ValueError(f"moving MIND features must match fixed features, got {tuple(moving.shape)}")
    if fixed.device != moving.device or fixed.dtype != moving.dtype:
        raise ValueError("fixed and moving MIND features must share device and dtype")
    if not fixed.is_floating_point():
        raise TypeError("MIND features must use a floating-point dtype")


def _require_field(field: torch.Tensor, spatial: tuple[int, int, int] | None = None) -> None:
    if field.dim() != 5 or field.shape[0] != 1 or field.shape[1] != 3:
        raise ValueError(f"Psi/residual must have shape [1,3,D,H,W], got {tuple(field.shape)}")
    if spatial is not None and tuple(field.shape[-3:]) != spatial:
        raise ValueError(f"Psi/residual spatial shape {tuple(field.shape[-3:])} does not match {spatial}")
    if not field.is_floating_point():
        raise TypeError("Psi/residual must use a floating-point dtype")


def _require_geometry_mask(mask: torch.Tensor, spatial: tuple[int, int, int], device: torch.device) -> None:
    if mask.shape != (1, 1, *spatial):
        raise ValueError(f"geometry mask must have shape {(1, 1, *spatial)}, got {tuple(mask.shape)}")
    if mask.dtype != torch.bool:
        raise TypeError("geometry mask must be boolean")
    if mask.device != device:
        raise ValueError("geometry mask must be on the same device as the search tensors")
    if not bool(mask.any()):
        raise ValueError("geometry mask must contain at least one active voxel")


def _require_candidate_tensor(tensor: torch.Tensor, name: str) -> None:
    if tensor.dim() != 5 or tensor.shape[0] != 1 or tensor.shape[1] != len(OFFSETS):
        raise ValueError(f"{name} must have shape [1,{len(OFFSETS)},D,H,W], got {tuple(tensor.shape)}")


def _standardize_candidate_costs(
    costs: torch.Tensor,
    valid: torch.Tensor,
    *,
    standardization_floor: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Standardize candidates with the frozen legacy accumulation order.

    C1/C2 accumulated the 27 first and second moments one offset at a time.
    A vectorized reduction is mathematically equivalent but not numerically
    equivalent in float32 when the candidate costs are nearly flat; cancellation
    can then change the posterior materially on CUDA.  Keep the historical order
    so the raw C3 controls remain genuine C1/C2 controls.
    """

    _require_candidate_tensor(costs, "candidate costs")
    _require_candidate_tensor(valid, "candidate validity mask")
    if valid.dtype != torch.bool or valid.shape != costs.shape or valid.device != costs.device:
        raise ValueError("valid must be a boolean tensor sharing candidate-cost shape and device")
    if not costs.is_floating_point():
        raise TypeError("candidate costs must use a floating-point dtype")
    if not math.isfinite(standardization_floor) or standardization_floor <= 0.0:
        raise ValueError("standardization_floor must be finite and positive")
    if not bool(torch.isfinite(costs.masked_select(valid)).all()):
        raise ValueError("valid candidate costs must be finite")

    map_shape = (1, 1, *costs.shape[-3:])
    sum_cost = costs.new_zeros(map_shape)
    sum_square = costs.new_zeros(map_shape)
    for index in range(len(OFFSETS)):
        candidate = costs[:, index : index + 1]
        candidate_valid = valid[:, index : index + 1]
        safe = torch.where(candidate_valid, candidate, torch.zeros_like(candidate))
        sum_cost += safe
        sum_square += safe.square()

    valid_count = valid.sum(dim=1, keepdim=True)
    count = valid_count.to(costs.dtype).clamp_min(1.0)
    mean = sum_cost / count
    variance = (sum_square / count - mean.square()).clamp_min(0.0)
    std = variance.sqrt().clamp_min(float(standardization_floor))
    standardized = torch.where(valid, (costs - mean) / std, torch.zeros_like(costs))
    return standardized, valid_count, mean, std


def build_standardized_mind_cost_volume(
    fixed_mind: torch.Tensor,
    moving_mind: torch.Tensor,
    psi_displacement: torch.Tensor,
    geometry_mask: torch.Tensor,
    *,
    standardization_floor: float = STANDARDIZATION_FLOOR,
) -> StandardizedCostVolume:
    """Build 27 target-centred scalar MIND costs and standardize over valid offsets.

    Feature sampling remains streaming: only the scalar cost of each offset is retained,
    never a ``[27,12,D,H,W]`` sampled-feature tensor. Invalid candidates have a stored
    standardized cost of zero and are excluded by the accompanying boolean mask.
    """

    _require_feature_pair(fixed_mind, moving_mind)
    spatial = tuple(fixed_mind.shape[-3:])
    _require_field(psi_displacement, spatial)
    _require_geometry_mask(geometry_mask, spatial, fixed_mind.device)
    if psi_displacement.device != fixed_mind.device or psi_displacement.dtype != fixed_mind.dtype:
        raise ValueError("Psi and MIND features must share device and dtype")
    if not math.isfinite(standardization_floor) or standardization_floor <= 0.0:
        raise ValueError("standardization_floor must be finite and positive")

    shape = (1, len(OFFSETS), *spatial)
    costs = torch.empty(shape, dtype=fixed_mind.dtype, device=fixed_mind.device)
    valid = torch.empty(shape, dtype=torch.bool, device=fixed_mind.device)

    with torch.inference_mode():
        for index, offset in enumerate(OFFSETS):
            sampled = sample_at_psi(moving_mind, psi_displacement, offset)
            cost = (fixed_mind - sampled).square().mean(dim=1)
            candidate_valid = valid_sample_mask(psi_displacement, offset)[:, 0] & geometry_mask[:, 0]
            costs[:, index].copy_(cost)
            valid[:, index].copy_(candidate_valid)

        standardized, valid_count, mean, std = _standardize_candidate_costs(
            costs,
            valid,
            standardization_floor=standardization_floor,
        )
        if bool((geometry_mask & (valid_count == 0)).any()):
            raise RuntimeError("MIND cost volume has an active voxel without a valid candidate")

    return StandardizedCostVolume(
        standardized_costs=standardized,
        valid=valid,
        valid_count=valid_count,
        cost_mean=mean,
        cost_std=std,
    )


def posterior_from_logits(
    logits: torch.Tensor,
    valid: torch.Tensor,
    *,
    temperature: float = 1.0,
) -> PosteriorResult:
    """Masked 27-way posterior with explicit entropy and normalized entropy ``h``."""

    _require_candidate_tensor(logits, "logits")
    _require_candidate_tensor(valid, "valid mask")
    if valid.dtype != torch.bool:
        raise TypeError("candidate validity mask must be boolean")
    if logits.shape != valid.shape or logits.device != valid.device:
        raise ValueError("logits and candidate validity must share shape and device")
    if not logits.is_floating_point():
        raise TypeError("logits must use a floating-point dtype")
    if not math.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("posterior temperature must be finite and positive")
    if not bool(torch.isfinite(logits.masked_select(valid)).all()):
        raise ValueError("valid candidate logits must be finite")

    with torch.inference_mode():
        valid_count = valid.sum(dim=1, keepdim=True)
        active = valid_count > 0
        scaled_logits = logits / float(temperature)
        scaled_maximum = torch.full(
            (1, 1, *logits.shape[-3:]),
            -torch.inf,
            dtype=logits.dtype,
            device=logits.device,
        )
        for index in range(len(OFFSETS)):
            candidate = scaled_logits[:, index : index + 1]
            candidate_valid = valid[:, index : index + 1]
            scaled_maximum = torch.where(
                candidate_valid & (candidate > scaled_maximum),
                candidate,
                scaled_maximum,
            )
        scaled_maximum = torch.where(active, scaled_maximum, torch.zeros_like(scaled_maximum))
        weights = torch.zeros_like(logits)
        normalizer = torch.zeros_like(scaled_maximum)
        weighted_shift_sum = torch.zeros_like(scaled_maximum)
        for index in range(len(OFFSETS)):
            candidate_valid = valid[:, index : index + 1]
            shifted = scaled_logits[:, index : index + 1] - scaled_maximum
            safe_shifted = torch.where(candidate_valid, shifted, torch.zeros_like(shifted))
            weight = torch.exp(safe_shifted) * candidate_valid.to(logits.dtype)
            weights[:, index : index + 1].copy_(weight)
            normalizer += weight
            weighted_shift_sum += weight * safe_shifted
        probabilities = torch.where(active, weights / normalizer.clamp_min(torch.finfo(logits.dtype).tiny), 0.0)
        entropy = torch.where(
            active,
            torch.log(normalizer.clamp_min(torch.finfo(logits.dtype).tiny))
            - weighted_shift_sum / normalizer.clamp_min(torch.finfo(logits.dtype).tiny),
            torch.zeros_like(normalizer),
        )
        log_k = torch.log(valid_count.to(logits.dtype).clamp_min(1.0))
        normalized_entropy = torch.where(valid_count > 1, entropy / log_k, torch.zeros_like(entropy))
        normalized_entropy = normalized_entropy.clamp(0.0, 1.0)
        confidence = torch.where(
            valid_count > 1,
            (1.0 - normalized_entropy).clamp(0.0, 1.0),
            active.to(logits.dtype),
        )

    return PosteriorResult(
        probabilities=probabilities,
        entropy=entropy,
        normalized_entropy=normalized_entropy,
        confidence=confidence,
        valid_count=valid_count,
    )


def raw_posterior(cost_volume: StandardizedCostVolume, *, temperature: float = 1.0) -> PosteriorResult:
    """Posterior of the unregularized standardized costs: ``softmax(-z)``."""

    return posterior_from_logits(-cost_volume.standardized_costs, cost_volume.valid, temperature=temperature)


def masked_separable_smooth_logits(logits: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """One masked separable ``[1,2,1]/4`` pass over each candidate logit map.

    The numerator and denominator are each filtered along all three axes before the
    single division. Invalid candidates at the receiving voxel remain invalid and
    are represented by zero in the returned tensor.
    """

    _require_candidate_tensor(logits, "logits")
    _require_candidate_tensor(valid, "valid mask")
    if valid.dtype != torch.bool or valid.shape != logits.shape or valid.device != logits.device:
        raise ValueError("valid must be a boolean tensor sharing logits shape and device")
    if not bool(torch.isfinite(logits.masked_select(valid)).all()):
        raise ValueError("valid candidate logits must be finite")

    channels = logits.shape[1]
    kernel = logits.new_tensor(LOGIT_MESSAGE_AXIS_KERNEL)
    validity = valid.to(logits.dtype)
    numerator = torch.where(valid, logits, torch.zeros_like(logits))
    denominator = validity
    with torch.inference_mode():
        for axis in range(3):
            kernel_shape = [1, 1, 1]
            kernel_shape[axis] = 3
            weight = kernel.view(1, 1, *kernel_shape).expand(channels, 1, *kernel_shape)
            padding = [0, 0, 0]
            padding[axis] = 1
            numerator = F.conv3d(numerator, weight, padding=tuple(padding), groups=channels)
            denominator = F.conv3d(denominator, weight, padding=tuple(padding), groups=channels)
    return torch.where(valid & (denominator > 0.0), numerator / denominator.clamp_min(1e-12), 0.0)


def apply_message_passing(
    logits: torch.Tensor,
    valid: torch.Tensor,
    raw_normalized_entropy: torch.Tensor,
    geometry_mask: torch.Tensor,
    *,
    mode: MessagePassingMode,
) -> MessagePassingResult:
    """Apply the frozen C3a one-pass message rule without restandardization.

    ``adaptive`` uses ``lambda(p)=h(p)``. ``isotropic`` uses the scalar
    ``lambda=mean_G h``. ``none`` is an out-of-place exact copy of the raw logits.
    """

    _require_candidate_tensor(logits, "logits")
    _require_candidate_tensor(valid, "valid mask")
    spatial = tuple(logits.shape[-3:])
    _require_geometry_mask(geometry_mask, spatial, logits.device)
    if raw_normalized_entropy.shape != (1, 1, *spatial):
        raise ValueError("raw normalized entropy must have shape [1,1,D,H,W]")
    if raw_normalized_entropy.device != logits.device or not raw_normalized_entropy.is_floating_point():
        raise ValueError("raw normalized entropy must share the logits device and use a floating dtype")
    if valid.dtype != torch.bool or valid.shape != logits.shape or valid.device != logits.device:
        raise ValueError("valid must be a boolean tensor sharing logits shape and device")
    if mode not in ("none", "isotropic", "adaptive"):
        raise ValueError(f"unsupported message-passing mode: {mode!r}")
    active = geometry_mask & (valid.sum(dim=1, keepdim=True) > 0)
    if not bool(active.any()):
        raise ValueError("message-passing geometry has no voxel with a valid candidate")
    h_values = raw_normalized_entropy.masked_select(active)
    if not bool(torch.isfinite(h_values).all()) or bool(((h_values < 0.0) | (h_values > 1.0)).any()):
        raise ValueError("raw normalized entropy must be finite and lie in [0,1] on G")

    if mode == "none":
        return MessagePassingResult(
            logits=logits.clone(),
            lambda_map=torch.zeros_like(raw_normalized_entropy),
            lambda_mean=0.0,
            mode=mode,
        )

    smoothed = masked_separable_smooth_logits(logits, valid)
    if mode == "isotropic":
        scalar = h_values.double().mean().to(dtype=logits.dtype)
        lambda_map = torch.where(active, scalar, torch.zeros_like(raw_normalized_entropy))
    else:
        lambda_map = torch.where(active, raw_normalized_entropy, torch.zeros_like(raw_normalized_entropy))
    mixed = (1.0 - lambda_map) * logits + lambda_map * smoothed
    mixed = torch.where(valid, mixed, torch.zeros_like(mixed))
    return MessagePassingResult(
        logits=mixed,
        lambda_map=lambda_map,
        lambda_mean=float(lambda_map.masked_select(active).double().mean().item()),
        mode=mode,
    )


def decode_posterior(posterior: PosteriorResult, *, mode: DecoderMode) -> DecoderResult:
    """Decode an offset posterior as its mean, optionally attenuated by confidence."""

    probabilities = posterior.probabilities
    _require_candidate_tensor(probabilities, "posterior probabilities")
    if mode not in ("confidence", "posterior_mean"):
        raise ValueError(f"unsupported decoder mode: {mode!r}")
    if not bool(torch.isfinite(probabilities).all()):
        raise ValueError("posterior probabilities must be finite")
    posterior_mean = probabilities.new_zeros((1, 3, *probabilities.shape[-3:]))
    for index, offset in enumerate(OFFSETS):
        posterior_mean += probabilities[:, index : index + 1] * probabilities.new_tensor(offset).view(1, 3, 1, 1, 1)
    displacement = posterior_mean * posterior.confidence if mode == "confidence" else posterior_mean.clone()
    return DecoderResult(
        displacement=displacement,
        posterior_mean=posterior_mean,
        entropy=posterior.entropy,
        normalized_entropy=posterior.normalized_entropy,
        confidence=posterior.confidence,
        mode=mode,
    )


def posterior_diagnostics(
    logits: torch.Tensor,
    valid: torch.Tensor,
    posterior: PosteriorResult,
    geometry_mask: torch.Tensor,
    *,
    temperature: float = 1.0,
) -> PosteriorDiagnostics:
    """Summarize a masked offset posterior without labels or deformation metrics.

    All scalar means use target voxels in ``geometry_mask``. The logit gap is
    ``largest(valid logits) - second_largest(valid logits)``; for logits ``-z``
    it is equivalently the second-smallest minus smallest standardized cost.
    Invalid-offset fraction uses ``|G| * 27`` as its denominator. Vector norms
    are voxelwise Euclidean norms averaged over ``G``. Their ratio is the ratio
    of those two means and is ``None`` when the unweighted mean norm is exactly
    zero. At least two valid candidates are required at every voxel in ``G``.

    The function fails closed if the supplied ``PosteriorResult`` is not the
    normalized masked posterior described by ``valid``. Invalid logits may be
    arbitrary because they are not evidence; valid logits must be finite.
    """

    _require_candidate_tensor(logits, "logits")
    _require_candidate_tensor(valid, "valid mask")
    probabilities = posterior.probabilities
    _require_candidate_tensor(probabilities, "posterior probabilities")
    spatial = tuple(logits.shape[-3:])
    _require_geometry_mask(geometry_mask, spatial, logits.device)
    if valid.dtype != torch.bool or valid.shape != logits.shape or valid.device != logits.device:
        raise ValueError("valid must be a boolean tensor sharing logits shape and device")
    if probabilities.shape != logits.shape or probabilities.device != logits.device:
        raise ValueError("posterior probabilities must share logits shape and device")
    if not logits.is_floating_point() or probabilities.dtype != logits.dtype:
        raise TypeError("logits and posterior probabilities must share a floating-point dtype")
    if not bool(torch.isfinite(logits.masked_select(valid)).all()):
        raise ValueError("valid candidate logits must be finite")

    map_shape = (1, 1, *spatial)
    for name, value in (
        ("entropy", posterior.entropy),
        ("normalized entropy", posterior.normalized_entropy),
        ("confidence", posterior.confidence),
    ):
        if value.shape != map_shape or value.device != logits.device or value.dtype != logits.dtype:
            raise ValueError(f"posterior {name} must share logits dtype/device and have shape {map_shape}")
    if posterior.valid_count.shape != map_shape or posterior.valid_count.device != logits.device:
        raise ValueError(f"posterior valid_count must have shape {map_shape} on the logits device")

    tolerance = max(1e-6, 16.0 * torch.finfo(logits.dtype).eps)
    expected_count = valid.sum(dim=1, keepdim=True)
    if not torch.equal(posterior.valid_count, expected_count):
        raise ValueError("posterior valid_count does not match the candidate mask")
    if not bool(torch.isfinite(probabilities).all()):
        raise ValueError("posterior probabilities must be finite, including invalid entries")
    if bool(((probabilities < -tolerance) | (probabilities > 1.0 + tolerance)).any()):
        raise ValueError("posterior probabilities must lie in [0,1]")
    if bool((probabilities.masked_select(~valid) != 0.0).any()):
        raise ValueError("invalid candidates must have exactly zero posterior probability")

    active = expected_count > 0
    probability_sum = probabilities.sum(dim=1, keepdim=True)
    expected_sum = active.to(logits.dtype)
    if not torch.allclose(probability_sum, expected_sum, atol=tolerance, rtol=tolerance):
        raise ValueError("posterior probabilities do not have the required masked normalization")
    reference = posterior_from_logits(logits, valid, temperature=temperature)
    if not torch.allclose(probabilities, reference.probabilities, atol=tolerance, rtol=tolerance):
        raise ValueError("posterior probabilities are inconsistent with logits, mask, and temperature")
    tiny = torch.finfo(logits.dtype).tiny
    log_probabilities = torch.where(
        probabilities > 0.0,
        torch.log(probabilities.clamp_min(tiny)),
        torch.zeros_like(probabilities),
    )
    expected_entropy = -(probabilities * log_probabilities).sum(dim=1, keepdim=True)
    log_k = torch.log(expected_count.to(logits.dtype).clamp_min(1.0))
    expected_h = torch.where(expected_count > 1, expected_entropy / log_k, torch.zeros_like(expected_entropy))
    expected_h = expected_h.clamp(0.0, 1.0)
    expected_confidence = torch.where(
        expected_count > 1,
        (1.0 - expected_h).clamp(0.0, 1.0),
        active.to(logits.dtype),
    )
    for name, observed, expected in (
        ("entropy", posterior.entropy, expected_entropy),
        ("normalized entropy", posterior.normalized_entropy, expected_h),
        ("confidence", posterior.confidence, expected_confidence),
    ):
        if not torch.allclose(observed, expected, atol=tolerance, rtol=tolerance):
            raise ValueError(f"posterior {name} is inconsistent with its probabilities")

    geometry_count = expected_count.masked_select(geometry_mask)
    if bool((geometry_count < 2).any()):
        raise ValueError("top1-top2 gap requires at least two valid candidates at every geometry voxel")
    masked_logits = torch.where(valid, logits, torch.full_like(logits, -torch.inf))
    top_two = masked_logits.topk(k=2, dim=1).values
    gap = top_two[:, :1] - top_two[:, 1:2]
    peak = probabilities.amax(dim=1, keepdim=True)

    posterior_mean = decode_posterior(posterior, mode="posterior_mean").posterior_mean
    confidence_weighted = posterior_mean * posterior.confidence
    posterior_mean_norm = posterior_mean.double().square().sum(dim=1, keepdim=True).sqrt()
    confidence_norm = confidence_weighted.double().square().sum(dim=1, keepdim=True).sqrt()
    mean_norm = float(posterior_mean_norm.masked_select(geometry_mask).mean().item())
    weighted_mean_norm = float(confidence_norm.masked_select(geometry_mask).mean().item())
    if mean_norm == 0.0:
        if weighted_mean_norm != 0.0:
            raise RuntimeError("confidence weighting created norm from an exactly zero posterior mean")
        norm_ratio = None
    else:
        norm_ratio = weighted_mean_norm / mean_norm
        if not math.isfinite(norm_ratio) or not 0.0 <= norm_ratio <= 1.0 + tolerance:
            raise RuntimeError("confidence-weighted posterior norm ratio is outside [0,1]")

    expanded_geometry = geometry_mask.expand_as(valid)
    active_voxels = int(geometry_mask.sum(dtype=torch.int64).item())
    invalid_offsets = int((expanded_geometry & ~valid).sum(dtype=torch.int64).item())
    candidate_denominator = active_voxels * len(OFFSETS)
    result = PosteriorDiagnostics(
        diagnostic_id=POSTERIOR_DIAGNOSTICS_ID,
        active_voxel_count=active_voxels,
        candidate_count=len(OFFSETS),
        top1_top2_valid_logit_gap_mean=float(gap.masked_select(geometry_mask).double().mean().item()),
        posterior_peak_probability_mean=float(peak.masked_select(geometry_mask).double().mean().item()),
        entropy_nats_mean=float(posterior.entropy.masked_select(geometry_mask).double().mean().item()),
        invalid_offset_fraction=invalid_offsets / candidate_denominator,
        posterior_mean_l2_norm_mean=mean_norm,
        confidence_weighted_mean_l2_norm_mean=weighted_mean_norm,
        confidence_to_mean_l2_norm_ratio=norm_ratio,
    )
    numeric = (
        result.top1_top2_valid_logit_gap_mean,
        result.posterior_peak_probability_mean,
        result.entropy_nats_mean,
        result.invalid_offset_fraction,
        result.posterior_mean_l2_norm_mean,
        result.confidence_weighted_mean_l2_norm_mean,
    )
    if not all(math.isfinite(value) for value in numeric):
        raise RuntimeError("posterior diagnostics produced a non-finite scalar")
    return result


def postprocess_residual(
    residual: torch.Tensor,
    *,
    scale: float,
    post_smoothing_passes: int,
    collar_width: int,
) -> torch.Tensor:
    """Apply scale, then proposal smoothing, then the exact identity collar."""

    _require_field(residual)
    if not math.isfinite(scale):
        raise ValueError("scale must be finite")
    if isinstance(post_smoothing_passes, bool) or not isinstance(post_smoothing_passes, int):
        raise TypeError("post_smoothing_passes must be an integer")
    if post_smoothing_passes < 0:
        raise ValueError("post_smoothing_passes must be non-negative")
    if isinstance(collar_width, bool) or not isinstance(collar_width, int) or collar_width < 1:
        raise ValueError("collar_width must be a positive integer")
    out = residual * float(scale)
    if post_smoothing_passes:
        out = smooth_proposal(out, passes=post_smoothing_passes)
    return identity_collar(out, width=collar_width)


def masked_rms_first_difference_roughness(
    residual: torch.Tensor,
    geometry_mask: torch.Tensor,
) -> RMSFirstDifferenceRoughness:
    """RMS of vector first differences over geometry-internal adjacent pairs.

    The input contract is the already postprocessed residual (scale, requested
    post-smoothing, then identity collar); this pure helper does not transform it.
    Every positive z/y/x edge whose two endpoint voxels are both in
    ``geometry_mask`` contributes once. The scalar is
    ``sqrt(sum_pairs ||u(q)-u(p)||_2^2 / number_of_pairs)``; it is not a
    componentwise RMS. Axis-specific values use the same definition. At least
    one valid adjacent pair is required. Values outside the geometry mask are
    ignored, while any non-finite value inside it is an integrity error.
    """

    _require_field(residual)
    spatial = tuple(residual.shape[-3:])
    _require_geometry_mask(geometry_mask, spatial, residual.device)
    expanded_geometry = geometry_mask.expand_as(residual)
    if not bool(torch.isfinite(residual.masked_select(expanded_geometry)).all()):
        raise ValueError("residual must be finite at every geometry voxel")

    residual64 = residual.double()
    pair_counts: list[int] = []
    axis_rms: list[float | None] = []
    total_energy = 0.0
    for axis in range(3):
        leading = [slice(None)] * 5
        trailing = [slice(None)] * 5
        dimension = axis + 2
        leading[dimension] = slice(1, None)
        trailing[dimension] = slice(None, -1)
        pair_mask = geometry_mask[tuple(leading)] & geometry_mask[tuple(trailing)]
        pair_count = int(pair_mask.sum(dtype=torch.int64).item())
        pair_counts.append(pair_count)
        if pair_count == 0:
            axis_rms.append(None)
            continue
        difference = residual64[tuple(leading)] - residual64[tuple(trailing)]
        squared_vector_norm = difference.square().sum(dim=1, keepdim=True)
        energy = float(squared_vector_norm.masked_select(pair_mask).sum().item())
        if not math.isfinite(energy):
            raise ValueError("residual first-difference energy is non-finite")
        total_energy += energy
        axis_rms.append(math.sqrt(energy / pair_count))

    total_pairs = sum(pair_counts)
    if total_pairs == 0:
        raise ValueError("RMS first-difference roughness requires at least one geometry-internal adjacent pair")
    roughness = math.sqrt(total_energy / total_pairs)
    if not math.isfinite(roughness):
        raise RuntimeError("RMS first-difference roughness is non-finite")
    return RMSFirstDifferenceRoughness(
        metric_id=RMS_FIRST_DIFFERENCE_ROUGHNESS_ID,
        rms_vector_first_difference=roughness,
        pair_count=total_pairs,
        axis_pair_counts_zyx=tuple(pair_counts),
        axis_rms_vector_first_difference_zyx=tuple(axis_rms),
    )


def masked_vector_rms(residual: torch.Tensor, geometry_mask: torch.Tensor) -> float:
    """RMS of the residual vector norm over the active geometry voxels."""

    _require_field(residual)
    spatial = tuple(residual.shape[-3:])
    _require_geometry_mask(geometry_mask, spatial, residual.device)
    squared_norm = residual.double().square().sum(dim=1, keepdim=True)
    return float(squared_norm.masked_select(geometry_mask).mean().sqrt().item())


def match_postprocessed_rms(
    residual: torch.Tensor,
    reference: torch.Tensor,
    geometry_mask: torch.Tensor,
) -> RMSMatchResult:
    """Match RMS after both inputs have already undergone scale/smooth/collar.

    This helper is pure: neither input nor the mask is modified. A zero source cannot
    be matched to a non-zero target and is rejected instead of inventing a direction.
    """

    _require_field(residual)
    _require_field(reference, tuple(residual.shape[-3:]))
    if residual.device != reference.device or residual.dtype != reference.dtype:
        raise ValueError("residual and RMS reference must share device and dtype")
    source_rms = masked_vector_rms(residual, geometry_mask)
    target_rms = masked_vector_rms(reference, geometry_mask)
    if source_rms == 0.0:
        if target_rms != 0.0:
            raise ValueError("a zero residual cannot be RMS-matched to a non-zero reference")
        factor = 1.0
    else:
        factor = target_rms / source_rms
    matched = residual * factor
    return RMSMatchResult(
        displacement=matched,
        scale_factor=float(factor),
        source_rms=source_rms,
        target_rms=target_rms,
        matched_rms=masked_vector_rms(matched, geometry_mask),
    )
