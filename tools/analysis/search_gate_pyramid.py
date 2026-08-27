from __future__ import annotations

import hashlib
import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Literal

import numpy as np
import torch
import torch.nn.functional as F

from tools.analysis.search_gate_cost_volume import masked_vector_rms
from tools.analysis.search_gate_intensity_runtime import build_intensity_reach_bank, decode_intensity_direction
from tools.analysis.search_gate_multiscale import postprocess_and_match_rms
from tools.analysis.transactional_search import (
    certified_local_clip_candidate,
    geometry_mask,
    masked_zscore,
    valid_sample_mask,
)

PyramidFamily = Literal["full_resolution", "blurred_full_resolution", "true_pyramid"]
BINOMIAL5 = (1.0, 4.0, 6.0, 4.0, 1.0)


@dataclass(frozen=True, slots=True)
class PyramidStage:
    factor: int
    stride_voxels: int
    level_shape: tuple[int, int, int]
    level_collar: int
    generation_count: int
    informative_count: int
    entropy_mean: float
    confidence_mean: float
    decoded_rms_full_grid: float
    requested_stage_rms: float
    realized_stage_rms: float
    clip_retention: float


@dataclass(frozen=True, slots=True)
class PyramidDirection:
    family: PyramidFamily
    factors: tuple[int, ...]
    rewarp_between_levels: bool
    displacement: torch.Tensor
    reference_rms: float
    pre_normalization_rms: float
    normalized_rms: float
    stages: tuple[PyramidStage, ...]


def array_sha256(tensor: torch.Tensor) -> str:
    array = np.ascontiguousarray(tensor.detach().cpu().numpy())
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def _require_volume(tensor: torch.Tensor, channels: int, label: str) -> tuple[int, int, int]:
    if tensor.ndim != 5 or tensor.shape[0] != 1 or tensor.shape[1] != channels:
        raise ValueError(f"{label} must have shape [1,{channels},D,H,W]")
    if not tensor.is_floating_point() or not bool(torch.isfinite(tensor).all()):
        raise ValueError(f"{label} must be a finite floating-point tensor")
    return tuple(int(value) for value in tensor.shape[-3:])


def _require_factor(factor: int) -> int:
    if isinstance(factor, bool) or not isinstance(factor, int) or factor not in (1, 2, 4):
        raise ValueError("pyramid factor must be one of 1, 2, 4")
    return factor


def _binomial_blur_once(tensor: torch.Tensor, dilation: int) -> torch.Tensor:
    if isinstance(dilation, bool) or not isinstance(dilation, int) or dilation < 1:
        raise ValueError("blur dilation must be a positive integer")
    result = tensor
    kernel = tensor.new_tensor(BINOMIAL5).div_(16.0)
    channels = tensor.shape[1]
    for axis in range(3):
        shape = [1, 1, 1]
        shape[axis] = 5
        weight = kernel.view(1, 1, *shape).expand(channels, 1, *shape)
        padding = [0, 0, 0, 0, 0, 0]
        # F.pad uses W-left/right, H-left/right, D-left/right.
        pad_axis = 2 - axis
        padding[2 * pad_axis] = 2 * dilation
        padding[2 * pad_axis + 1] = 2 * dilation
        result = F.conv3d(
            F.pad(result, tuple(padding), mode="replicate"),
            weight,
            groups=channels,
            dilation=tuple(dilation if index == axis else 1 for index in range(3)),
        )
    return result


def binomial_blur3d(tensor: torch.Tensor, passes: int = 1) -> torch.Tensor:
    """Repeated unit-grid separable [1,4,6,4,1]/16 filtering."""

    if tensor.ndim != 5 or not tensor.is_floating_point():
        raise ValueError("binomial blur expects a floating [B,C,D,H,W] tensor")
    if isinstance(passes, bool) or not isinstance(passes, int) or passes < 0:
        raise ValueError("blur passes must be a non-negative integer")
    result = tensor
    for _ in range(passes):
        result = _binomial_blur_once(result, 1)
    return result


def downsample_image(image: torch.Tensor, factor: int) -> torch.Tensor:
    spatial = _require_volume(image, 1, "image")
    factor = _require_factor(factor)
    if any(size % factor for size in spatial):
        raise ValueError(f"image shape {spatial} is not divisible by factor {factor}")
    if factor == 1:
        return image.clone()
    if factor == 4:
        return downsample_image(downsample_image(image, 2), 2)
    filtered = _binomial_blur_once(image, 1)
    size = tuple(value // 2 for value in spatial)
    return F.interpolate(filtered, size=size, mode="trilinear", align_corners=False)


def blurred_full_resolution_image(image: torch.Tensor, factor: int) -> torch.Tensor:
    _require_volume(image, 1, "image")
    factor = _require_factor(factor)
    if factor == 1:
        return image.clone()
    result = _binomial_blur_once(image, 1)
    return _binomial_blur_once(result, 2) if factor == 4 else result


def project_psi_to_level(field: torch.Tensor, factor: int) -> torch.Tensor:
    """Project a full-grid voxel displacement to an align_corners=False coarse grid."""

    spatial = _require_volume(field, 3, "Psi field")
    factor = _require_factor(factor)
    if any(size % factor for size in spatial):
        raise ValueError(f"field shape {spatial} is not divisible by factor {factor}")
    if factor == 1:
        return field.clone()
    if factor == 4:
        return project_psi_to_level(project_psi_to_level(field, 2), 2)
    filtered = _binomial_blur_once(field, 1)
    coarse = F.interpolate(
        filtered,
        size=tuple(value // 2 for value in spatial),
        mode="trilinear",
        align_corners=False,
    )
    return coarse / 2.0


def lift_level_vector(vector: torch.Tensor, full_shape: Sequence[int], factor: int) -> torch.Tensor:
    level_shape = _require_volume(vector, 3, "level vector")
    factor = _require_factor(factor)
    target = tuple(int(value) for value in full_shape)
    if len(target) != 3 or any(value <= 0 for value in target):
        raise ValueError("full_shape must contain three positive integers")
    if tuple(value * factor for value in level_shape) != target:
        raise ValueError(f"level shape {level_shape} and factor {factor} do not reconstruct {target}")
    if factor == 1:
        return vector.clone()
    return F.interpolate(vector, size=target, mode="trilinear", align_corners=False) * float(factor)


def level_collar(full_collar: int, factor: int) -> int:
    factor = _require_factor(factor)
    if isinstance(full_collar, bool) or not isinstance(full_collar, int) or full_collar < 1:
        raise ValueError("full collar must be a positive integer")
    return math.ceil(full_collar / factor)


def all_offsets_valid_mask(field: torch.Tensor, collar: int, stride: int) -> torch.Tensor:
    spatial = _require_volume(field, 3, "Psi field")
    if stride < 1:
        raise ValueError("stride must be positive")
    mask = geometry_mask(spatial, collar, field.device)
    for dz in (-stride, 0, stride):
        for dy in (-stride, 0, stride):
            for dx in (-stride, 0, stride):
                mask = mask & valid_sample_mask(field, (dz, dy, dx))
    if not bool(mask.any()):
        raise RuntimeError("pyramid stage has empty all-offset support")
    return mask


def _mean_on(value: torch.Tensor, mask: torch.Tensor) -> float:
    selected = value.masked_select(mask)
    if selected.numel() == 0 or not bool(torch.isfinite(selected).all()):
        raise RuntimeError("pyramid diagnostic support is empty or non-finite")
    return float(selected.double().mean().item())


def _match_rms(vector: torch.Tensor, mask: torch.Tensor, target: float) -> torch.Tensor:
    observed = masked_vector_rms(vector, mask)
    if not math.isfinite(target) or target <= 0.0 or observed <= 0.0:
        raise RuntimeError(f"invalid RMS match: observed={observed}, target={target}")
    return vector * float(target / observed)


def build_pyramid_direction(
    fixed: torch.Tensor,
    moving: torch.Tensor,
    initial: torch.Tensor,
    rms_reference: torch.Tensor,
    *,
    family: PyramidFamily,
    factors: Sequence[int],
    rewarp_between_levels: bool,
    full_collar: int,
    work_eps: float,
    standardization_floor: float,
    image_std_floor: float,
    proposal_multiplier: float,
    post_smoothing_passes: int,
    posterior_temperature: float,
    centre_beta: float,
    require_all_candidates_valid: bool,
    stage_clip_sweeps: int,
) -> PyramidDirection:
    """Build one matched-budget coarse-to-fine direction without label access.

    Every numerical setting is supplied by the caller's frozen contract; this module owns no default.
    """

    spatial = _require_volume(initial, 3, "initial Psi")
    if _require_volume(rms_reference, 3, "RMS reference") != spatial:
        raise ValueError("initial and RMS reference shapes differ")
    if _require_volume(fixed, 1, "fixed image") != spatial or _require_volume(moving, 1, "moving image") != spatial:
        raise ValueError("images and fields must share the full-resolution grid")
    frozen_factors = tuple(_require_factor(int(value)) for value in factors)
    if not frozen_factors or tuple(sorted(frozen_factors, reverse=True)) != frozen_factors or frozen_factors[-1] != 1:
        raise ValueError("factors must be a non-empty coarse-to-fine sequence ending at 1")
    if len(set(frozen_factors)) != len(frozen_factors):
        raise ValueError("pyramid factors must be unique")
    if family not in ("full_resolution", "blurred_full_resolution", "true_pyramid"):
        raise ValueError(f"unknown pyramid family: {family}")
    if not isinstance(rewarp_between_levels, bool):
        raise TypeError("rewarp_between_levels must be bool")

    full_mask = geometry_mask(spatial, full_collar, initial.device)
    reference_rms = masked_vector_rms(rms_reference, full_mask)
    if reference_rms <= 0.0:
        raise RuntimeError("historical RMS reference is zero")
    stage_target = reference_rms / len(frozen_factors)
    current = initial.clone()
    accumulated = torch.zeros_like(initial)
    stages: list[PyramidStage] = []

    for factor in frozen_factors:
        base_current = current if rewarp_between_levels else initial
        if family == "true_pyramid":
            level_fixed = downsample_image(fixed, factor)
            level_moving = downsample_image(moving, factor)
            level_current = project_psi_to_level(base_current, factor)
            stride = 1
            collar = level_collar(full_collar, factor)
        else:
            level_fixed = (
                blurred_full_resolution_image(fixed, factor) if family == "blurred_full_resolution" else fixed.clone()
            )
            level_moving = (
                blurred_full_resolution_image(moving, factor) if family == "blurred_full_resolution" else moving.clone()
            )
            level_current = base_current
            stride = factor
            collar = full_collar

        level_mask = all_offsets_valid_mask(level_current, collar, stride)
        fixed_norm = masked_zscore(level_fixed, level_mask, std_floor=image_std_floor)
        moving_norm = masked_zscore(level_moving, level_mask, std_floor=image_std_floor)
        bank = build_intensity_reach_bank(
            fixed_norm,
            moving_norm,
            level_current,
            level_mask,
            reach_id=f"{family}_f{factor}",
            cost_id=f"intensity_{family}_f{factor}",
            stride_voxels=stride,
            standardization_floor=standardization_floor,
            require_all_candidates_valid=require_all_candidates_valid,
        )
        decoded = decode_intensity_direction(
            bank,
            direction_id=f"{family}_f{factor}_b0",
            centre_beta=centre_beta,
            posterior_temperature=posterior_temperature,
        )
        full_direction = (
            lift_level_vector(decoded.decoded.displacement, spatial, factor)
            if family == "true_pyramid"
            else decoded.decoded.displacement
        )
        processed = postprocess_and_match_rms(
            full_direction,
            full_mask,
            proposal_multiplier=proposal_multiplier,
            smoothing_passes=post_smoothing_passes,
            collar_width=full_collar,
            rms_reference=rms_reference / float(len(frozen_factors)),
        )
        requested = processed.displacement
        if rewarp_between_levels:
            updated, operator = certified_local_clip_candidate(
                current,
                requested,
                full_mask,
                work_eps=work_eps,
                sweeps=stage_clip_sweeps,
            )
            realized = updated - current
            current = updated
        else:
            realized = requested
            accumulated = accumulated + requested
            operator = {"retained_norm_ratio": 1.0}
        stages.append(
            PyramidStage(
                factor=factor,
                stride_voxels=stride,
                level_shape=tuple(level_current.shape[-3:]),
                level_collar=collar,
                generation_count=bank.generation_count,
                informative_count=bank.standardized_informative_count,
                entropy_mean=_mean_on(decoded.posterior.entropy, level_mask),
                confidence_mean=_mean_on(decoded.posterior.confidence, level_mask),
                decoded_rms_full_grid=masked_vector_rms(full_direction, full_mask),
                requested_stage_rms=stage_target,
                realized_stage_rms=masked_vector_rms(realized, full_mask),
                clip_retention=float(operator["retained_norm_ratio"]),
            )
        )

    net = current - initial if rewarp_between_levels else accumulated
    pre_rms = masked_vector_rms(net, full_mask)
    normalized = _match_rms(net, full_mask, reference_rms)
    normalized_rms = masked_vector_rms(normalized, full_mask)
    return PyramidDirection(
        family=family,
        factors=frozen_factors,
        rewarp_between_levels=rewarp_between_levels,
        displacement=normalized,
        reference_rms=reference_rms,
        pre_normalization_rms=pre_rms,
        normalized_rms=normalized_rms,
        stages=tuple(stages),
    )


def direction_record(direction: PyramidDirection) -> dict[str, object]:
    return {
        "family": direction.family,
        "factors": list(direction.factors),
        "rewarp_between_levels": direction.rewarp_between_levels,
        "reference_rms": direction.reference_rms,
        "pre_normalization_rms": direction.pre_normalization_rms,
        "normalized_rms": direction.normalized_rms,
        "array_sha256": array_sha256(direction.displacement),
        "stages": [asdict(stage) for stage in direction.stages],
    }
