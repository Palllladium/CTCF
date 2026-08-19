from __future__ import annotations

import math
import os
import shutil
import tempfile
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Literal

import numpy as np
import torch
import torch.nn.functional as F

from tools.analysis.run_artifacts import sha256_file
from utils.cert_exact import certify_flow_exact
from utils.field import identity_collar

Offset = tuple[int, int, int]
FeatureKind = Literal["mind", "intensity"]
DecoderKind = Literal["soft", "hard"]
Orientation = Literal["target_centered", "reversed"]

OFFSETS: tuple[Offset, ...] = tuple((dz, dy, dx) for dz in (-1, 0, 1) for dy in (-1, 0, 1) for dx in (-1, 0, 1))
ZERO_OFFSET_INDEX = OFFSETS.index((0, 0, 0))
STEP_COEFFICIENTS: tuple[float, ...] = tuple(2.0**-index for index in range(13))


@dataclass(frozen=True)
class ProposalResult:
    displacement: torch.Tensor
    hard_displacement: torch.Tensor
    entropy: torch.Tensor
    confidence: torch.Tensor
    valid_candidates: torch.Tensor
    feature: FeatureKind
    orientation: Orientation


@dataclass(frozen=True)
class CandidateScreen:
    coefficient: float
    utility: float | None
    improvement: float | None
    tolerance: float
    cert_bound: float | None
    utility_passed: bool
    fast_certificate_passed: bool


@dataclass(frozen=True)
class TransactionOutcome:
    status: Literal["ACCEPTED", "ROLLED_BACK"]
    selected: CandidateScreen | None
    exact_report: dict[str, object]
    output_path: Path
    output_sha256: str
    rollback_byte_identical: bool


def _require_scalar_volume(image: torch.Tensor, name: str) -> None:
    if image.dim() != 5 or image.shape[0] != 1 or image.shape[1] != 1:
        raise ValueError(f"{name} must have shape [1,1,D,H,W], got {tuple(image.shape)}")


def _require_field(field: torch.Tensor, name: str) -> None:
    if field.dim() != 5 or field.shape[0] != 1 or field.shape[1] != 3:
        raise ValueError(f"{name} must have shape [1,3,D,H,W], got {tuple(field.shape)}")


@lru_cache(maxsize=8)
def _cached_voxel_grid(
    d: int,
    h: int,
    w: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    zz, yy, xx = torch.meshgrid(
        torch.arange(d, device=device, dtype=dtype),
        torch.arange(h, device=device, dtype=dtype),
        torch.arange(w, device=device, dtype=dtype),
        indexing="ij",
    )
    return torch.stack((zz, yy, xx), dim=0).unsqueeze(0)


def voxel_grid_like(field: torch.Tensor) -> torch.Tensor:
    """Return a cached immutable [1,3,D,H,W] z-y-x identity grid."""
    _require_field(field, "field")
    d, h, w = field.shape[-3:]
    return _cached_voxel_grid(d, h, w, field.device, field.dtype)


def phi_to_psi_displacement(phi_displacement: torch.Tensor) -> torch.Tensor:
    """Materialise the effective source-index map used by CTCF's legacy sampler."""
    grid = voxel_grid_like(phi_displacement)
    d, h, w = phi_displacement.shape[-3:]
    scale = phi_displacement.new_tensor((d / (d - 1), h / (h - 1), w / (w - 1))).view(1, 3, 1, 1, 1)
    return scale * (grid + phi_displacement) - 0.5 - grid


def psi_to_phi_displacement(psi_displacement: torch.Tensor) -> torch.Tensor:
    """Inverse of `phi_to_psi_displacement` in voxel coordinates."""
    grid = voxel_grid_like(psi_displacement)
    d, h, w = psi_displacement.shape[-3:]
    inverse_scale = psi_displacement.new_tensor(((d - 1) / d, (h - 1) / h, (w - 1) / w)).view(1, 3, 1, 1, 1)
    return inverse_scale * (grid + psi_displacement + 0.5) - grid


def geometry_mask(shape: tuple[int, int, int], collar_width: int, device: torch.device) -> torch.Tensor:
    """Fixed mask that excludes the geometry collar used by every C0 utility comparison."""
    if collar_width < 1 or min(shape) <= 2 * collar_width:
        raise ValueError(f"Invalid collar_width={collar_width} for shape={shape}")
    mask = torch.zeros((1, 1, *shape), dtype=torch.bool, device=device)
    mask[
        :,
        :,
        collar_width:-collar_width,
        collar_width:-collar_width,
        collar_width:-collar_width,
    ] = True
    return mask


def masked_zscore(image: torch.Tensor, mask: torch.Tensor, std_floor: float = 1e-6) -> torch.Tensor:
    """Independently standardise one image inside the fixed geometric mask, without clipping."""
    _require_scalar_volume(image, "image")
    values = image.masked_select(mask)
    if values.numel() == 0:
        raise ValueError("normalisation mask is empty")
    mean = values.double().mean().to(image.dtype)
    std = values.double().std(unbiased=False).clamp_min(std_floor).to(image.dtype)
    return (image - mean) / std


def mind_ssc(image: torch.Tensor, radius: int = 1, dilation: int = 2) -> torch.Tensor:
    """Independent MIND-SSC implementation matching ConvexAdam b229e52 channel order.

    The production implementation intentionally has no import-time or run-time dependency on
    ConvexAdam. C0.1 compares it against a pinned reference implementation.
    """
    _require_scalar_volume(image, "image")
    if radius < 0 or dilation < 1:
        raise ValueError("radius must be >= 0 and dilation must be >= 1")

    six = torch.tensor(
        [[0, 1, 1], [1, 1, 0], [1, 0, 1], [1, 1, 2], [2, 1, 1], [1, 2, 1]],
        device=image.device,
        dtype=torch.long,
    )
    delta = six[:, None, :] - six[None, :, :]
    dist = (delta * delta).sum(dim=-1)
    row, col = torch.meshgrid(
        torch.arange(6, device=image.device),
        torch.arange(6, device=image.device),
        indexing="ij",
    )
    pair_mask = (row > col) & (dist == 2)
    shift1 = six[:, None, :].expand(6, 6, 3)[pair_mask]
    shift2 = six[None, :, :].expand(6, 6, 3)[pair_mask]

    kernels1 = image.new_zeros((12, 1, 3, 3, 3))
    kernels2 = image.new_zeros((12, 1, 3, 3, 3))
    channels = torch.arange(12, device=image.device)
    kernels1[channels, 0, shift1[:, 0], shift1[:, 1], shift1[:, 2]] = 1.0
    kernels2[channels, 0, shift2[:, 0], shift2[:, 1], shift2[:, 2]] = 1.0

    padded = F.pad(image, (dilation,) * 6, mode="replicate")
    differences = F.conv3d(padded, kernels1, dilation=dilation) - F.conv3d(padded, kernels2, dilation=dilation)
    kernel_size = 2 * radius + 1
    ssd = F.avg_pool3d(
        F.pad(differences.square(), (radius,) * 6, mode="replicate"),
        kernel_size=kernel_size,
        stride=1,
    )
    descriptor = ssd - ssd.amin(dim=1, keepdim=True)
    variance = descriptor.mean(dim=1, keepdim=True)
    global_variance = variance.mean()
    if not bool(torch.isfinite(global_variance)):
        raise RuntimeError("MIND-SSC variance is not finite")
    if float(global_variance.item()) == 0.0:
        return torch.ones_like(descriptor)
    variance = variance.clamp(global_variance * 0.001, global_variance * 1000.0)
    descriptor = torch.exp(-descriptor / variance.clamp_min(torch.finfo(image.dtype).tiny))
    order = torch.tensor([6, 8, 1, 11, 2, 10, 0, 7, 9, 4, 5, 3], device=image.device)
    return descriptor.index_select(1, order)


def _psi_coordinates(psi_displacement: torch.Tensor, offset: Offset = (0, 0, 0)) -> torch.Tensor:
    grid = voxel_grid_like(psi_displacement)
    shift = psi_displacement.new_tensor(offset).view(1, 3, 1, 1, 1)
    return grid + psi_displacement + shift


def _coordinates_to_align_false_grid(coordinates: torch.Tensor) -> torch.Tensor:
    """Convert z-y-x source indices to grid_sample's x-y-z grid for align_corners=False."""
    _require_field(coordinates, "coordinates")
    d, h, w = coordinates.shape[-3:]
    size = coordinates.new_tensor((d, h, w)).view(1, 3, 1, 1, 1)
    normalised = 2.0 * (coordinates + 0.5) / size - 1.0
    return normalised[:, (2, 1, 0)].permute(0, 2, 3, 4, 1)


def sample_at_psi(
    source: torch.Tensor,
    psi_displacement: torch.Tensor,
    offset: Offset = (0, 0, 0),
    mode: Literal["bilinear", "nearest"] = "bilinear",
) -> torch.Tensor:
    """Pull source features at Psi(p)+offset using the deployed align_corners=False convention."""
    _require_field(psi_displacement, "psi_displacement")
    if source.dim() != 5 or source.shape[0] != 1 or source.shape[-3:] != psi_displacement.shape[-3:]:
        raise ValueError("source and psi_displacement must be single-volume tensors on the same grid")
    grid = _coordinates_to_align_false_grid(_psi_coordinates(psi_displacement, offset))
    return F.grid_sample(source, grid, mode=mode, padding_mode="zeros", align_corners=False)


def valid_sample_mask(psi_displacement: torch.Tensor, offset: Offset = (0, 0, 0)) -> torch.Tensor:
    """Mask locations whose interpolation centre remains inside the source voxel-centre domain."""
    coordinates = _psi_coordinates(psi_displacement, offset)
    limits = coordinates.new_tensor(psi_displacement.shape[-3:]).view(1, 3, 1, 1, 1) - 1.0
    valid = ((coordinates >= 0.0) & (coordinates <= limits)).all(dim=1, keepdim=True)
    return valid


def _box_sum_3d(tensor: torch.Tensor, win: int) -> torch.Tensor:
    """Separable zero-padded box sum, mathematically equal to a win^3 all-ones convolution."""
    kernel = tensor.new_ones(win)
    out = tensor
    for shape, padding in (
        ((win, 1, 1), (win // 2, 0, 0)),
        ((1, win, 1), (0, win // 2, 0)),
        ((1, 1, win), (0, 0, win // 2)),
    ):
        out = F.conv3d(out, kernel.view(1, 1, *shape), padding=padding)
    return out


def _local_ncc_map(fixed: torch.Tensor, moving: torch.Tensor, win: int = 9, eps: float = 1e-5) -> torch.Tensor:
    """Voxelwise NCCVxm loss map; its masked FP64 mean is the registered C0 utility."""
    if fixed.shape != moving.shape or fixed.shape[1] != 1:
        raise ValueError("local NCC expects equal [1,1,D,H,W] tensors")
    count = float(win**3)

    fixed_sum = _box_sum_3d(fixed, win)
    moving_sum = _box_sum_3d(moving, win)
    fixed2_sum = _box_sum_3d(fixed.square(), win)
    moving2_sum = _box_sum_3d(moving.square(), win)
    product_sum = _box_sum_3d(fixed * moving, win)
    fixed_mean = fixed_sum / count
    moving_mean = moving_sum / count
    cross = product_sum - moving_mean * fixed_sum - fixed_mean * moving_sum + fixed_mean * moving_mean * count
    fixed_var = (fixed2_sum - 2.0 * fixed_mean * fixed_sum + fixed_mean.square() * count).clamp_min(eps)
    moving_var = (moving2_sum - 2.0 * moving_mean * moving_sum + moving_mean.square() * count).clamp_min(eps)
    return -(cross.square() / (fixed_var * moving_var))


def utility_loss(
    fixed: torch.Tensor,
    moving: torch.Tensor,
    psi_displacement: torch.Tensor,
    mask: torch.Tensor,
    win: int = 9,
    eps: float = 1e-5,
) -> float:
    fixed_norm = masked_zscore(fixed, mask)
    moving_norm = masked_zscore(moving, mask)
    warped = sample_at_psi(moving_norm, psi_displacement)
    valid = mask & valid_sample_mask(psi_displacement)
    values = _local_ncc_map(fixed_norm, warped, win=win, eps=eps).masked_select(valid)
    if values.numel() == 0 or not bool(torch.isfinite(values).all()):
        return float("nan")
    return float(values.double().mean().item())


def mind_distance(
    fixed: torch.Tensor,
    moving: torch.Tensor,
    psi_displacement: torch.Tensor,
    mask: torch.Tensor,
    radius: int = 1,
    dilation: int = 2,
) -> float:
    fixed_feature = mind_ssc(masked_zscore(fixed, mask), radius=radius, dilation=dilation)
    moving_feature = mind_ssc(masked_zscore(moving, mask), radius=radius, dilation=dilation)
    warped = sample_at_psi(moving_feature, psi_displacement)
    valid = mask & valid_sample_mask(psi_displacement)
    values = (fixed_feature - warped).square().mean(dim=1, keepdim=True).masked_select(valid)
    if values.numel() == 0 or not bool(torch.isfinite(values).all()):
        return float("nan")
    return float(values.double().mean().item())


def mind_distance_from_features(
    fixed_feature: torch.Tensor,
    moving_feature: torch.Tensor,
    psi_displacement: torch.Tensor,
    mask: torch.Tensor,
) -> float:
    if fixed_feature.shape != moving_feature.shape or fixed_feature.shape[1] != 12:
        raise ValueError("MIND feature tensors must have equal [1,12,D,H,W] shapes")
    warped = sample_at_psi(moving_feature, psi_displacement)
    valid = mask & valid_sample_mask(psi_displacement)
    values = (fixed_feature - warped).square().mean(dim=1, keepdim=True).masked_select(valid)
    if values.numel() == 0 or not bool(torch.isfinite(values).all()):
        return float("nan")
    return float(values.double().mean().item())


def _candidate_cost(
    fixed_feature: torch.Tensor,
    moving_feature: torch.Tensor,
    psi_displacement: torch.Tensor,
    offset: Offset,
    feature: FeatureKind,
    orientation: Orientation,
    reversed_current_moving: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if orientation == "target_centered":
        sampled = sample_at_psi(moving_feature, psi_displacement, offset)
        valid = valid_sample_mask(psi_displacement, offset)
        target = fixed_feature
    elif orientation == "reversed":
        if reversed_current_moving is None:
            raise ValueError("reversed orientation requires a precomputed current moving feature")
        zero_psi = torch.zeros_like(psi_displacement)
        sampled = sample_at_psi(fixed_feature, zero_psi, offset)
        valid = valid_sample_mask(zero_psi, offset) & valid_sample_mask(psi_displacement)
        target = reversed_current_moving
    else:
        raise ValueError(f"Unknown orientation: {orientation}")

    if feature == "mind":
        cost = (target - sampled).square().mean(dim=1, keepdim=True)
    elif feature == "intensity":
        cost = _local_ncc_map(target, sampled)
    else:
        raise ValueError(f"Unknown feature: {feature}")
    return cost, valid


def _smooth_proposal(proposal: torch.Tensor) -> torch.Tensor:
    _require_field(proposal, "proposal")
    kernel = proposal.new_tensor([1.0, 2.0, 1.0]) / 4.0
    out = proposal
    for axis in range(3):
        shape = [1, 1, 1]
        shape[axis] = 3
        weight = kernel.view(1, 1, *shape).expand(3, 1, *shape)
        padding = [0, 0, 0]
        padding[axis] = 1
        out = F.conv3d(out, weight, padding=tuple(padding), groups=3)
    return out


def build_proposal(
    fixed: torch.Tensor,
    moving: torch.Tensor,
    psi_displacement: torch.Tensor,
    mask: torch.Tensor,
    feature: FeatureKind = "mind",
    orientation: Orientation = "target_centered",
    collar_width: int = 4,
    mind_radius: int = 1,
    mind_dilation: int = 2,
    fixed_feature_override: torch.Tensor | None = None,
    moving_feature_override: torch.Tensor | None = None,
) -> ProposalResult:
    """Build soft and hard residual proposals without materialising [27,12,D,H,W]."""
    _require_scalar_volume(fixed, "fixed")
    _require_scalar_volume(moving, "moving")
    _require_field(psi_displacement, "psi_displacement")
    if fixed.shape[-3:] != moving.shape[-3:] or fixed.shape[-3:] != psi_displacement.shape[-3:]:
        raise ValueError("fixed, moving and Psi must use the same spatial grid")

    if (fixed_feature_override is None) != (moving_feature_override is None):
        raise ValueError("both feature overrides must be supplied together")
    if fixed_feature_override is not None and moving_feature_override is not None:
        fixed_feature, moving_feature = fixed_feature_override, moving_feature_override
    else:
        fixed_norm = masked_zscore(fixed, mask)
        moving_norm = masked_zscore(moving, mask)
        if feature == "mind":
            fixed_feature = mind_ssc(fixed_norm, radius=mind_radius, dilation=mind_dilation)
            moving_feature = mind_ssc(moving_norm, radius=mind_radius, dilation=mind_dilation)
        else:
            fixed_feature, moving_feature = fixed_norm, moving_norm
    expected_channels = 12 if feature == "mind" else 1
    if fixed_feature.shape != moving_feature.shape or fixed_feature.shape[1] != expected_channels:
        raise ValueError(f"feature overrides must have equal [1,{expected_channels},D,H,W] shapes")

    shape = (1, 1, *fixed.shape[-3:])
    sum_cost = fixed.new_zeros(shape)
    sum_square = fixed.new_zeros(shape)
    count = torch.zeros(shape, device=fixed.device, dtype=torch.int16)
    min_cost = torch.full(shape, float("inf"), device=fixed.device, dtype=fixed.dtype)
    argmin = torch.full(shape, ZERO_OFFSET_INDEX, device=fixed.device, dtype=torch.int16)
    reversed_current_moving = sample_at_psi(moving_feature, psi_displacement) if orientation == "reversed" else None

    with torch.inference_mode():
        for index, offset in enumerate(OFFSETS):
            cost, valid = _candidate_cost(
                fixed_feature,
                moving_feature,
                psi_displacement,
                offset,
                feature,
                orientation,
                reversed_current_moving,
            )
            valid = valid & mask
            safe_cost = torch.where(valid, cost, torch.zeros_like(cost))
            sum_cost += safe_cost
            sum_square += safe_cost.square()
            count += valid.to(count.dtype)
            better = valid & (cost < min_cost)
            min_cost = torch.where(better, cost, min_cost)
            argmin = torch.where(better, torch.full_like(argmin, index), argmin)

        if bool(((count == 0) & mask).any()):
            raise RuntimeError("candidate search produced an in-mask voxel with no valid candidate")
        count_f = count.to(fixed.dtype)
        safe_count = count_f.clamp_min(1.0)
        mean = sum_cost / safe_count
        variance = (sum_square / safe_count - mean.square()).clamp_min(0.0)
        std = variance.sqrt().clamp_min(1e-6)
        z_min = (min_cost - mean) / std

        sum_weight = fixed.new_zeros(shape)
        sum_weight_log = fixed.new_zeros(shape)
        expected = fixed.new_zeros((1, 3, *fixed.shape[-3:]))
        for offset in OFFSETS:
            cost, valid = _candidate_cost(
                fixed_feature,
                moving_feature,
                psi_displacement,
                offset,
                feature,
                orientation,
                reversed_current_moving,
            )
            valid = valid & mask
            log_weight = -((cost - mean) / std - z_min)
            safe_log_weight = torch.where(valid, log_weight, torch.zeros_like(log_weight))
            weight = torch.exp(safe_log_weight) * valid.to(cost.dtype)
            sum_weight += weight
            sum_weight_log += weight * safe_log_weight
            expected += weight * fixed.new_tensor(offset).view(1, 3, 1, 1, 1)

        expected /= sum_weight.clamp_min(torch.finfo(fixed.dtype).tiny)
        entropy = torch.log(sum_weight.clamp_min(torch.finfo(fixed.dtype).tiny)) - (
            sum_weight_log / sum_weight.clamp_min(torch.finfo(fixed.dtype).tiny)
        )
        entropy = torch.where(mask, entropy, torch.zeros_like(entropy))
        log_k = torch.log(count_f.clamp_min(1.0))
        confidence = torch.where(count > 1, (1.0 - entropy / log_k).clamp(0.0, 1.0), torch.ones_like(entropy))
        soft = expected * confidence

        offset_table = fixed.new_tensor(OFFSETS)
        hard = offset_table[argmin.long().squeeze(1)].permute(0, 4, 1, 2, 3)
        if orientation == "reversed":
            # Reversed costs index fixed(p+d); d lives in target coordinates, so -d is the
            # additive source-coordinate correction required by the ProposalResult contract.
            soft = -soft
            hard = -hard
        soft = identity_collar(_smooth_proposal(soft), width=collar_width)
        hard = identity_collar(_smooth_proposal(hard), width=collar_width)

    return ProposalResult(
        displacement=soft,
        hard_displacement=hard,
        entropy=entropy,
        confidence=confidence,
        valid_candidates=count,
        feature=feature,
        orientation=orientation,
    )


def screen_candidates(
    initial_psi: torch.Tensor,
    proposal: torch.Tensor,
    fixed: torch.Tensor,
    moving: torch.Tensor,
    mask: torch.Tensor,
    fast_certificate: Callable[[torch.Tensor], float],
    eps: float = 0.001,
    utility_relative_tolerance: float = 1e-6,
    coefficients: tuple[float, ...] = STEP_COEFFICIENTS,
) -> tuple[list[CandidateScreen], list[CandidateScreen]]:
    """Evaluate the preregistered ladder; exact certification remains the caller's responsibility."""
    _require_field(initial_psi, "initial_psi")
    _require_field(proposal, "proposal")
    baseline = utility_loss(fixed, moving, initial_psi, mask)
    if not math.isfinite(baseline):
        raise RuntimeError("baseline utility is not finite")
    tolerance = utility_relative_tolerance * max(1.0, abs(baseline))
    records: list[CandidateScreen] = []
    eligible: list[CandidateScreen] = []
    with torch.inference_mode():
        for coefficient in coefficients:
            candidate = (initial_psi + float(coefficient) * proposal).float()
            utility = utility_loss(fixed, moving, candidate, mask)
            finite_utility = math.isfinite(utility)
            improvement = baseline - utility if finite_utility else None
            utility_passed = improvement is not None and improvement >= tolerance
            raw_cert_bound = fast_certificate(candidate) if utility_passed else None
            cert_bound = raw_cert_bound if raw_cert_bound is not None and math.isfinite(raw_cert_bound) else None
            cert_passed = cert_bound is not None and cert_bound >= eps
            record = CandidateScreen(
                coefficient=float(coefficient),
                utility=utility if finite_utility else None,
                improvement=improvement,
                tolerance=tolerance,
                cert_bound=cert_bound,
                utility_passed=utility_passed,
                fast_certificate_passed=cert_passed,
            )
            records.append(record)
            if cert_passed:
                eligible.append(record)
    return records, eligible


def save_flow_npz_atomic(path: Path, flow: torch.Tensor) -> None:
    """Atomically persist the exact float32 flow object consumed by the verifier."""
    _require_field(flow, "flow")
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".npz", dir=path.parent)
    os.close(fd)
    try:
        array = flow.detach().cpu().contiguous().numpy().astype(np.float32, copy=False)
        np.savez_compressed(temporary_name, flow=array)
        os.replace(temporary_name, path)
    except BaseException:
        with suppress(FileNotFoundError):
            os.unlink(temporary_name)
        raise


def load_flow_npz(path: Path) -> torch.Tensor:
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != {"flow"}:
            raise ValueError(f"{path}: expected the sole key 'flow', got {archive.files}")
        array = archive["flow"]
    if array.dtype != np.float32:
        raise TypeError(f"{path}: stored flow must be float32, got {array.dtype}")
    tensor = torch.from_numpy(np.ascontiguousarray(array))
    if tensor.dim() == 4:
        tensor = tensor.unsqueeze(0)
    _require_field(tensor, str(path))
    return tensor


def commit_exact_candidate(
    initial_path: Path,
    output_path: Path,
    eligible: list[CandidateScreen],
    initial_psi: torch.Tensor | None = None,
    proposal: torch.Tensor | None = None,
    eps: str = "0.001",
    max_exact: int = 100_000,
    tile_shape: tuple[int, int, int] = (8, 64, 64),
) -> TransactionOutcome:
    """Commit the first exactly certified saved candidate, otherwise copy the initial bytes back.

    `eligible` must already follow the preregistered descending coefficient order. Every candidate is
    saved and reloaded before the exact check. A failed transaction never serialises the initial tensor:
    it restores the original NPZ bytes with `copyfile` and verifies their SHA-256 identity.
    """
    initial_path = initial_path.resolve()
    output_path = output_path.resolve()
    if not initial_path.is_file():
        raise FileNotFoundError(initial_path)
    if initial_path == output_path:
        raise ValueError("initial_path and output_path must differ so rollback can be byte-preserving")
    initial_sha = sha256_file(initial_path)
    exact_report: dict[str, object] = {
        "status": "NOT_ATTEMPTED",
        "certified": False,
        "complete": True,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if eligible and (initial_psi is None or proposal is None):
        raise ValueError("initial_psi and proposal are required when eligible candidates are present")
    for screen in eligible:
        assert initial_psi is not None and proposal is not None
        candidate = (initial_psi + screen.coefficient * proposal).float()
        fd, candidate_name = tempfile.mkstemp(
            prefix=f".{output_path.name}.candidate.", suffix=".npz", dir=output_path.parent
        )
        os.close(fd)
        candidate_path = Path(candidate_name)
        try:
            save_flow_npz_atomic(candidate_path, candidate.float())
            stored = load_flow_npz(candidate_path)
            exact_report = certify_flow_exact(stored, eps=eps, max_exact=max_exact, tile_shape=tile_shape)
            if exact_report.get("status") == "CERTIFIED" and exact_report.get("certified") is True:
                os.replace(candidate_path, output_path)
                return TransactionOutcome(
                    status="ACCEPTED",
                    selected=screen,
                    exact_report=exact_report,
                    output_path=output_path,
                    output_sha256=sha256_file(output_path),
                    rollback_byte_identical=False,
                )
        finally:
            with suppress(FileNotFoundError):
                candidate_path.unlink()

    fd, rollback_name = tempfile.mkstemp(prefix=f".{output_path.name}.rollback.", dir=output_path.parent)
    os.close(fd)
    try:
        shutil.copyfile(initial_path, rollback_name)
        os.replace(rollback_name, output_path)
    finally:
        with suppress(FileNotFoundError):
            os.unlink(rollback_name)
    output_sha = sha256_file(output_path)
    identical = output_sha == initial_sha
    if not identical:
        raise RuntimeError("rollback output is not byte-identical to the initial artifact")
    rollback_report = certify_flow_exact(
        load_flow_npz(output_path), eps=eps, max_exact=max_exact, tile_shape=tile_shape
    )
    if rollback_report.get("status") != "CERTIFIED" or rollback_report.get("certified") is not True:
        raise RuntimeError("byte-exact rollback target is not exactly certified")
    return TransactionOutcome(
        status="ROLLED_BACK",
        selected=None,
        exact_report=rollback_report,
        output_path=output_path,
        output_sha256=output_sha,
        rollback_byte_identical=True,
    )
