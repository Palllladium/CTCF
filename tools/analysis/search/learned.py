from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn

from models.CorrMLP.wrapper import CorrMLPSolo
from tools.analysis.search.cost_volume import (
    MomentReductionMode,
    StandardizedCostVolume,
    standardize_candidate_costs,
)
from tools.analysis.search.transaction import OFFSETS, sample_at_psi, voxel_grid_like

CORRMLP_IXI_LAST_CHECKPOINT_SHA256 = "9cafbf426bd8a86cf9bc7e2981fcf7399101af6177292c3e726fc5b56eefa170"
CORRMLP_IXI_LAST_EPOCH = 99
CORRMLP_FULL_STATE_KEY_COUNT = 386
CORRMLP_X1_CHANNELS = 8
CORRMLP_X1_CONV_PADDING_MARGIN = 2
CORRMLP_X1_COST_ID = "CORRMLP_P11_IXI_LAST_E99_X1_NEGATIVE_MEAN_PRODUCT_V1"
EQUAL_INTENSITY_HYBRID_COST_ID = "CORRMLP_X1_INTENSITY_EQUAL_STANDARDIZED_FUSION_V1"
DEFAULT_MOMENT_REDUCTION: MomentReductionMode = "centered_two_pass_fp32"
DEFAULT_STANDARDIZATION_FLOOR = 1e-6


@dataclass(frozen=True)
class FrozenCorrMLPDescriptor:
    model: CorrMLPSolo
    checkpoint_path: str
    checkpoint_sha256: str
    epoch: int
    state_key_count: int


@dataclass(frozen=True)
class RawCandidateCostVolume:
    cost_id: str
    costs: torch.Tensor
    valid: torch.Tensor
    valid_count: torch.Tensor
    offsets: tuple[tuple[int, int, int], ...]


@dataclass(frozen=True)
class EqualHybridCostVolume:
    cost_id: str
    standardized_costs: torch.Tensor
    valid: torch.Tensor
    valid_count: torch.Tensor
    offsets: tuple[tuple[int, int, int], ...]
    learned: StandardizedCostVolume
    intensity: StandardizedCostVolume
    fusion: StandardizedCostVolume


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_sha256(value: str, label: str) -> str:
    normalized = value.lower()
    if len(normalized) != 64 or any(character not in "0123456789abcdef" for character in normalized):
        raise ValueError(f"{label} must be a lowercase-compatible SHA-256 hex digest")
    return normalized


def _checkpoint_state(checkpoint: object) -> tuple[int, Mapping[str, torch.Tensor]]:
    if not isinstance(checkpoint, dict):
        raise TypeError("CorrMLP checkpoint must be a dictionary")
    epoch = checkpoint.get("epoch")
    if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch != CORRMLP_IXI_LAST_EPOCH:
        raise RuntimeError(f"CorrMLP checkpoint epoch mismatch: expected {CORRMLP_IXI_LAST_EPOCH}, got {epoch!r}")
    state = checkpoint.get("state_dict")
    if not isinstance(state, Mapping) or not state:
        raise RuntimeError("CorrMLP checkpoint must contain a non-empty state_dict mapping")
    if not all(isinstance(key, str) and isinstance(value, torch.Tensor) for key, value in state.items()):
        raise TypeError("CorrMLP state_dict must map string keys to tensors")
    return epoch, state


def load_frozen_corrmlp_x1(
    checkpoint_path: str | Path,
    *,
    expected_sha256: str = CORRMLP_IXI_LAST_CHECKPOINT_SHA256,
) -> FrozenCorrMLPDescriptor:
    path = Path(checkpoint_path).resolve()
    expected = _require_sha256(expected_sha256, "expected checkpoint SHA-256")
    if not path.is_file():
        raise FileNotFoundError(f"CorrMLP checkpoint not found: {path}")
    if path.stat().st_size <= 0:
        raise RuntimeError(f"CorrMLP checkpoint is empty: {path}")
    actual = _sha256_file(path)
    if actual != expected:
        raise RuntimeError(f"CorrMLP checkpoint SHA-256 mismatch: expected {expected}, got {actual}")

    checkpoint = torch.load(path, map_location="cpu")
    epoch, state = _checkpoint_state(checkpoint)
    if len(state) != CORRMLP_FULL_STATE_KEY_COUNT:
        raise RuntimeError(
            f"CorrMLP state key count mismatch: expected {CORRMLP_FULL_STATE_KEY_COUNT}, got {len(state)}"
        )
    if any(not key.startswith("net.") for key in state):
        raise RuntimeError("CorrMLP checkpoint contains a state key without the required 'net.' prefix")

    model = CorrMLPSolo(enc_channels=8, dec_channels=16, use_checkpoint=False)
    expected_keys = set(model.state_dict())
    observed_keys = set(state)
    if observed_keys != expected_keys:
        missing = sorted(expected_keys - observed_keys)
        unexpected = sorted(observed_keys - expected_keys)
        raise RuntimeError(f"CorrMLP full-wrapper state mismatch: missing={missing[:5]}, unexpected={unexpected[:5]}")
    if tuple(model.named_buffers()):
        raise RuntimeError("Frozen CorrMLP x1 contract unexpectedly acquired persistent buffers")
    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError as error:
        raise RuntimeError(f"Strict CorrMLP full-wrapper load failed: {error}") from error
    model.requires_grad_(False)
    model.eval()
    if model.training or any(parameter.requires_grad for parameter in model.parameters()):
        raise RuntimeError("CorrMLP descriptor could not be frozen")
    return FrozenCorrMLPDescriptor(
        model=model,
        checkpoint_path=str(path),
        checkpoint_sha256=actual,
        epoch=epoch,
        state_key_count=len(state),
    )


def _model_device(model: nn.Module) -> torch.device:
    devices = {parameter.device for parameter in model.parameters()}
    if len(devices) != 1:
        raise RuntimeError(f"CorrMLP parameters must occupy exactly one device, got {sorted(map(str, devices))}")
    return next(iter(devices))


def extract_corrmlp_x1(model: CorrMLPSolo, image: torch.Tensor) -> torch.Tensor:
    if not isinstance(model, CorrMLPSolo):
        raise TypeError("model must be the strict-loaded CorrMLPSolo wrapper")
    if model.training or any(parameter.requires_grad for parameter in model.parameters()):
        raise RuntimeError("CorrMLP descriptor model must be frozen and in eval mode")
    if image.dim() != 5 or image.shape[0] != 1 or image.shape[1] != 1:
        raise ValueError(f"image must have shape [1,1,D,H,W], got {tuple(image.shape)}")
    if image.dtype != torch.float32:
        raise TypeError(f"CorrMLP descriptor input must be float32, got {image.dtype}")
    if image.device != _model_device(model):
        raise ValueError("CorrMLP descriptor model and image must occupy the same device")
    if not bool(torch.isfinite(image).all()):
        raise ValueError("CorrMLP descriptor input contains NaN or Inf")

    with torch.inference_mode(), torch.autocast(device_type=image.device.type, enabled=False):
        pyramid = model.net.Encoder(image)
    if not isinstance(pyramid, (list, tuple)) or len(pyramid) != 4:
        raise RuntimeError("CorrMLP shared encoder must return exactly four unary feature levels")
    x1 = pyramid[0]
    expected_shape = (1, CORRMLP_X1_CHANNELS, *image.shape[-3:])
    if tuple(x1.shape) != expected_shape:
        raise RuntimeError(f"CorrMLP x1 shape mismatch: expected {expected_shape}, got {tuple(x1.shape)}")
    if x1.dtype != torch.float32 or x1.device != image.device:
        raise RuntimeError("CorrMLP x1 must preserve FP32 dtype and input device")
    if x1.requires_grad or not bool(torch.isfinite(x1).all()):
        raise FloatingPointError("CorrMLP x1 is not a finite detached feature tensor")
    return x1


def _require_feature_pair(fixed: torch.Tensor, moving: torch.Tensor) -> tuple[int, int, int]:
    if fixed.dim() != 5 or tuple(fixed.shape[:2]) != (1, CORRMLP_X1_CHANNELS):
        raise ValueError(f"fixed CorrMLP x1 must have shape [1,{CORRMLP_X1_CHANNELS},D,H,W], got {tuple(fixed.shape)}")
    if moving.shape != fixed.shape:
        raise ValueError("moving CorrMLP x1 must match fixed CorrMLP x1")
    if fixed.dtype != torch.float32 or moving.dtype != torch.float32:
        raise TypeError("CorrMLP x1 cost construction is frozen to float32")
    if fixed.device != moving.device:
        raise ValueError("fixed and moving CorrMLP x1 must occupy the same device")
    if not bool(torch.isfinite(fixed).all()) or not bool(torch.isfinite(moving).all()):
        raise ValueError("CorrMLP x1 features contain NaN or Inf")
    return tuple(int(value) for value in fixed.shape[-3:])


def _require_field_and_mask(
    field: torch.Tensor,
    geometry_mask: torch.Tensor,
    spatial: tuple[int, int, int],
    device: torch.device,
) -> None:
    if field.shape != (1, 3, *spatial) or field.dtype != torch.float32 or field.device != device:
        raise ValueError("Psi must be a float32 [1,3,D,H,W] field on the feature device")
    if not bool(torch.isfinite(field).all()):
        raise ValueError("Psi contains NaN or Inf")
    if geometry_mask.shape != (1, 1, *spatial) or geometry_mask.dtype != torch.bool:
        raise ValueError("geometry_mask must be boolean with shape [1,1,D,H,W]")
    if geometry_mask.device != device or not bool(geometry_mask.any()):
        raise ValueError("geometry_mask must be non-empty and occupy the feature device")


def corrmlp_x1_offsets(stride_voxels: int) -> tuple[tuple[int, int, int], ...]:
    if isinstance(stride_voxels, bool) or not isinstance(stride_voxels, int) or stride_voxels < 1:
        raise ValueError("stride_voxels must be a positive integer")
    return tuple(tuple(component * stride_voxels for component in offset) for offset in OFFSETS)


def valid_corrmlp_x1_sample_mask(
    psi_displacement: torch.Tensor,
    offset: tuple[int, int, int],
    *,
    stride_voxels: int = 1,
    padding_margin: int = CORRMLP_X1_CONV_PADDING_MARGIN,
) -> torch.Tensor:
    offsets = corrmlp_x1_offsets(stride_voxels)
    if offset not in offsets:
        raise ValueError(f"offset must be one of the 27 stride-{stride_voxels} offsets, got {offset!r}")
    if isinstance(padding_margin, bool) or not isinstance(padding_margin, int) or padding_margin < 0:
        raise ValueError("padding_margin must be a non-negative integer")
    spatial = tuple(int(value) for value in psi_displacement.shape[-3:])
    if psi_displacement.shape != (1, 3, *spatial) or not psi_displacement.is_floating_point():
        raise ValueError("Psi must have shape [1,3,D,H,W] and floating-point dtype")
    if min(spatial) <= 2 * padding_margin:
        raise ValueError(f"padding margin {padding_margin} leaves no feature interior for shape {spatial}")
    shift = psi_displacement.new_tensor(offset).view(1, 3, 1, 1, 1)
    coordinates = voxel_grid_like(psi_displacement) + psi_displacement + shift
    lower = float(padding_margin)
    upper = coordinates.new_tensor(spatial).view(1, 3, 1, 1, 1) - 1.0 - lower
    return ((coordinates >= lower) & (coordinates <= upper)).all(dim=1, keepdim=True)


def build_raw_corrmlp_x1_cost_volume(
    fixed_x1: torch.Tensor,
    moving_x1: torch.Tensor,
    psi_displacement: torch.Tensor,
    geometry_mask: torch.Tensor,
    *,
    stride_voxels: int,
    padding_margin: int = CORRMLP_X1_CONV_PADDING_MARGIN,
    require_all_candidates_valid: bool = True,
) -> RawCandidateCostVolume:
    spatial = _require_feature_pair(fixed_x1, moving_x1)
    _require_field_and_mask(psi_displacement, geometry_mask, spatial, fixed_x1.device)
    offsets = corrmlp_x1_offsets(stride_voxels)
    fixed_support = valid_corrmlp_x1_sample_mask(
        torch.zeros_like(psi_displacement),
        (0, 0, 0),
        stride_voxels=stride_voxels,
        padding_margin=padding_margin,
    )
    if bool((geometry_mask & ~fixed_support).any()):
        raise ValueError("geometry_mask includes fixed x1 voxels contaminated by encoder padding")

    shape = (1, len(OFFSETS), *spatial)
    costs = torch.empty(shape, dtype=torch.float32, device=fixed_x1.device)
    valid = torch.empty(shape, dtype=torch.bool, device=fixed_x1.device)
    with torch.inference_mode():
        for index, offset in enumerate(offsets):
            sampled = sample_at_psi(moving_x1, psi_displacement, offset)
            negative_mean_product = -(fixed_x1 * sampled).mean(dim=1)
            candidate_valid = (
                geometry_mask[:, 0]
                & valid_corrmlp_x1_sample_mask(
                    psi_displacement,
                    offset,
                    stride_voxels=stride_voxels,
                    padding_margin=padding_margin,
                )[:, 0]
            )
            costs[:, index].copy_(negative_mean_product)
            valid[:, index].copy_(candidate_valid)
    valid_count = valid.sum(dim=1, keepdim=True)
    active_count = valid_count.masked_select(geometry_mask)
    if bool((active_count < 2).any()):
        raise RuntimeError("CorrMLP x1 cost volume has fewer than two candidates on active support")
    if require_all_candidates_valid and bool((active_count != len(OFFSETS)).any()):
        raise RuntimeError("CorrMLP x1 cost volume lost a candidate on the frozen common support")
    if not bool(torch.isfinite(costs.masked_select(valid)).all()):
        raise FloatingPointError("CorrMLP x1 cost volume contains non-finite valid costs")
    return RawCandidateCostVolume(
        cost_id=f"{CORRMLP_X1_COST_ID}_STRIDE{stride_voxels}",
        costs=costs,
        valid=valid,
        valid_count=valid_count,
        offsets=offsets,
    )


def raw_candidate_cost_volume(
    cost_id: str,
    costs: torch.Tensor,
    valid: torch.Tensor,
    *,
    offsets: tuple[tuple[int, int, int], ...],
) -> RawCandidateCostVolume:
    if not cost_id:
        raise ValueError("cost_id must be non-empty")
    if costs.dim() != 5 or costs.shape[0] != 1 or costs.shape[1] != len(OFFSETS):
        raise ValueError(f"candidate costs must have shape [1,{len(OFFSETS)},D,H,W]")
    if not costs.is_floating_point() or valid.dtype != torch.bool or valid.shape != costs.shape:
        raise ValueError("candidate costs and validity mask have incompatible dtype or shape")
    if valid.device != costs.device or not bool(valid.any()):
        raise ValueError("candidate validity mask must be non-empty and occupy the cost device")
    if not bool(torch.isfinite(costs.masked_select(valid)).all()):
        raise ValueError("valid candidate costs must be finite")
    if len(offsets) != len(OFFSETS) or len(set(offsets)) != len(OFFSETS):
        raise ValueError("offsets must contain exactly 27 unique physical displacements")
    if offsets[len(offsets) // 2] != (0, 0, 0):
        raise ValueError("offsets must preserve the frozen zero-offset centre ordering")
    return RawCandidateCostVolume(
        cost_id=cost_id,
        costs=costs,
        valid=valid,
        valid_count=valid.sum(dim=1, keepdim=True),
        offsets=offsets,
    )


def standardize_raw_candidate_costs(
    volume: RawCandidateCostVolume,
    *,
    mode: MomentReductionMode = DEFAULT_MOMENT_REDUCTION,
    standardization_floor: float = DEFAULT_STANDARDIZATION_FLOOR,
) -> StandardizedCostVolume:
    if not math.isfinite(standardization_floor) or standardization_floor <= 0.0:
        raise ValueError("standardization_floor must be finite and positive")
    observed_count = volume.valid.sum(dim=1, keepdim=True)
    if not torch.equal(volume.valid_count, observed_count):
        raise RuntimeError("raw cost volume valid_count does not match its validity mask")
    return standardize_candidate_costs(
        volume.costs,
        volume.valid,
        mode=mode,
        standardization_floor=standardization_floor,
    )


def equal_standardized_intensity_hybrid(
    learned: RawCandidateCostVolume,
    intensity: RawCandidateCostVolume,
    *,
    mode: MomentReductionMode = DEFAULT_MOMENT_REDUCTION,
    standardization_floor: float = DEFAULT_STANDARDIZATION_FLOOR,
) -> EqualHybridCostVolume:
    if learned.costs.shape != intensity.costs.shape:
        raise ValueError("learned and intensity candidate costs must have identical shape")
    if learned.costs.device != intensity.costs.device or learned.costs.dtype != intensity.costs.dtype:
        raise ValueError("learned and intensity candidate costs must share device and dtype")
    if not torch.equal(learned.valid, intensity.valid):
        raise RuntimeError("learned and intensity costs must use the exact same candidate-support mask")
    if learned.offsets != intensity.offsets:
        raise RuntimeError("learned and intensity costs must use the exact same physical offsets")
    learned_z = standardize_raw_candidate_costs(
        learned,
        mode=mode,
        standardization_floor=standardization_floor,
    )
    intensity_z = standardize_raw_candidate_costs(
        intensity,
        mode=mode,
        standardization_floor=standardization_floor,
    )
    component_informative = (
        (learned_z.valid_count >= 2) & ~learned_z.floor_hit & (intensity_z.valid_count >= 2) & ~intensity_z.floor_hit
    )
    component_valid = learned.valid & component_informative
    if not bool(component_valid.any()):
        raise FloatingPointError("equal learned/intensity fusion has no jointly informative component support")
    equal_mean = 0.5 * learned_z.standardized_costs + 0.5 * intensity_z.standardized_costs
    equal_mean = torch.where(component_valid, equal_mean, torch.zeros_like(equal_mean))
    if not bool(torch.isfinite(equal_mean).all()):
        raise FloatingPointError("equal standardized learned/intensity fusion became non-finite")
    fusion = standardize_candidate_costs(
        equal_mean,
        component_valid,
        mode=mode,
        standardization_floor=standardization_floor,
    )
    fusion_valid = component_valid & ~fusion.floor_hit
    if not bool(fusion_valid.any()):
        raise FloatingPointError("equal learned/intensity fusion is flat on all jointly informative support")
    standardized = torch.where(fusion_valid, fusion.standardized_costs, torch.zeros_like(fusion.standardized_costs))
    return EqualHybridCostVolume(
        cost_id=EQUAL_INTENSITY_HYBRID_COST_ID,
        standardized_costs=standardized,
        valid=fusion_valid,
        valid_count=fusion_valid.sum(dim=1, keepdim=True),
        offsets=learned.offsets,
        learned=learned_z,
        intensity=intensity_z,
        fusion=fusion,
    )
