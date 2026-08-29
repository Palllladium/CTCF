from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from experiments.stage5.safety import COLLAR_WIDTH
from models.CTCF.controller import STAGE5_CHANNEL_GROUPS, STAGE5_INPUT_CHANNEL_COUNT
from tools.analysis.search.intensity import build_intensity_reach_bank, decode_intensity_direction
from tools.analysis.search.pyramid import all_offsets_valid_mask
from tools.analysis.search.transaction import masked_zscore, sample_at_psi, smooth_proposal
from utils.field import identity_collar

IMAGE_STD_FLOOR = 1e-6
COST_STD_FLOOR = 1e-6
POSTERIOR_TEMPERATURE = 1.0
CENTRE_BETA = 0.0
FLOW_CONTEXT_SCALE_VOXELS = 16.0
SEARCH_STRIDES = (2, 4)


@dataclass(frozen=True, slots=True)
class ReachFeatures:
    stride: int
    vector_input: torch.Tensor
    posterior: torch.Tensor
    statistics: torch.Tensor
    proposal: torch.Tensor


@dataclass(frozen=True, slots=True)
class Stage5FeatureBundle:
    controller_input: torch.Tensor
    common_support: torch.Tensor
    fixed_normalized: torch.Tensor
    moving_normalized: torch.Tensor
    s2: ReachFeatures
    s4: ReachFeatures


def _require_scalar_pair(fixed: torch.Tensor, moving: torch.Tensor, psi0: torch.Tensor) -> None:
    if fixed.dim() != 5 or fixed.shape[0] != 1 or fixed.shape[1] != 1:
        raise ValueError("fixed must have shape [1,1,D,H,W]")
    if moving.shape != fixed.shape:
        raise ValueError("moving and fixed must share shape")
    if psi0.shape != (1, 3, *fixed.shape[-3:]):
        raise ValueError("psi0 must have shape [1,3,D,H,W] on the image grid")
    tensors = (fixed, moving, psi0)
    if any(not tensor.is_floating_point() for tensor in tensors):
        raise TypeError("images and psi0 must be floating-point tensors")
    if any(tensor.device != fixed.device for tensor in tensors):
        raise ValueError("images and psi0 must share a device")
    if any(not bool(torch.isfinite(tensor).all()) for tensor in tensors):
        raise FloatingPointError("images and psi0 must be finite")


def _reach_features(
    fixed_normalized: torch.Tensor,
    moving_normalized: torch.Tensor,
    psi0: torch.Tensor,
    common_support: torch.Tensor,
    *,
    stride: int,
) -> ReachFeatures:
    bank = build_intensity_reach_bank(
        fixed_normalized,
        moving_normalized,
        psi0,
        common_support,
        reach_id=f"stage5_intensity_s{stride}",
        cost_id=f"stage5_intensity_cost_s{stride}",
        stride_voxels=stride,
        standardization_floor=COST_STD_FLOOR,
        require_all_candidates_valid=True,
    )
    direction = decode_intensity_direction(
        bank,
        direction_id=f"stage5_intensity_direction_s{stride}",
        centre_beta=CENTRE_BETA,
        posterior_temperature=POSTERIOR_TEMPERATURE,
    )
    posterior = direction.posterior.probabilities.float()
    if posterior.shape[1] != 27:
        raise RuntimeError(f"S{stride} posterior must contain 27 candidates")
    top2 = posterior.topk(k=2, dim=1).values
    gap = top2[:, :1] - top2[:, 1:2]
    normalized_entropy = direction.posterior.entropy.float() / math.log(27.0)
    support = direction.posterior.valid.all(dim=1, keepdim=True).to(posterior.dtype)
    statistics = torch.cat((normalized_entropy, gap, support), dim=1)
    mask = common_support.to(posterior.dtype)
    vector = direction.decoded.displacement.float() * mask
    proposal = identity_collar(smooth_proposal(vector, passes=1), width=COLLAR_WIDTH) * mask
    return ReachFeatures(
        stride=stride,
        vector_input=(vector / float(stride)).contiguous(),
        posterior=(posterior * mask).contiguous(),
        statistics=(statistics * mask).contiguous(),
        proposal=proposal.contiguous(),
    )


def assemble_controller_input(
    fixed_normalized: torch.Tensor,
    warped_moving_normalized: torch.Tensor,
    psi0: torch.Tensor,
    s2: ReachFeatures,
    s4: ReachFeatures,
) -> torch.Tensor:
    spatial = fixed_normalized.shape[-3:]
    expected_scalar = (1, 1, *spatial)
    if fixed_normalized.shape != expected_scalar or warped_moving_normalized.shape != expected_scalar:
        raise ValueError("normalized context images must have shape [1,1,D,H,W]")
    if psi0.shape != (1, 3, *spatial):
        raise ValueError("psi0 context has the wrong shape")
    parts = (
        fixed_normalized.float(),
        warped_moving_normalized.float(),
        psi0.float() / FLOW_CONTEXT_SCALE_VOXELS,
        s2.vector_input.float(),
        s2.posterior.float(),
        s2.statistics.float(),
        s4.vector_input.float(),
        s4.posterior.float(),
        s4.statistics.float(),
    )
    result = torch.cat(parts, dim=1).contiguous()
    if result.shape != (1, STAGE5_INPUT_CHANNEL_COUNT, *spatial):
        raise RuntimeError(f"assembled Stage5 input has unexpected shape {tuple(result.shape)}")
    # Finiteness is checked once, at the controller's own input boundary
    # (``_validate_feature_tensor``). Repeating the scan here would sweep the full
    # 71-channel volume a second time on every direction of every training pair.
    for name, channel_slice in STAGE5_CHANNEL_GROUPS.items():
        if channel_slice.stop > result.shape[1]:
            raise RuntimeError(f"controller channel slice is out of range: {name}")
    return result


@torch.inference_mode()
def build_stage5_features(fixed: torch.Tensor, moving: torch.Tensor, psi0: torch.Tensor) -> Stage5FeatureBundle:
    _require_scalar_pair(fixed, moving, psi0)
    # Search inputs are a frozen numerical contract shared by training and
    # deployment.  They must remain FP32 even when a caller wraps controller
    # inference in an AMP context.
    with torch.autocast(device_type=fixed.device.type, enabled=False):
        common_support = all_offsets_valid_mask(psi0.float(), COLLAR_WIDTH, max(SEARCH_STRIDES))
        fixed_normalized = masked_zscore(fixed.float(), common_support, std_floor=IMAGE_STD_FLOOR)
        moving_normalized = masked_zscore(moving.float(), common_support, std_floor=IMAGE_STD_FLOOR)
        warped = sample_at_psi(moving_normalized, psi0.float())
        s2 = _reach_features(
            fixed_normalized,
            moving_normalized,
            psi0.float(),
            common_support,
            stride=SEARCH_STRIDES[0],
        )
        s4 = _reach_features(
            fixed_normalized,
            moving_normalized,
            psi0.float(),
            common_support,
            stride=SEARCH_STRIDES[1],
        )
        controller_input = assemble_controller_input(fixed_normalized, warped, psi0, s2, s4)
    return Stage5FeatureBundle(
        controller_input=controller_input,
        common_support=common_support,
        fixed_normalized=fixed_normalized,
        moving_normalized=moving_normalized,
        s2=s2,
        s4=s4,
    )


__all__ = [
    "CENTRE_BETA",
    "COLLAR_WIDTH",
    "COST_STD_FLOOR",
    "FLOW_CONTEXT_SCALE_VOXELS",
    "IMAGE_STD_FLOOR",
    "POSTERIOR_TEMPERATURE",
    "SEARCH_STRIDES",
    "ReachFeatures",
    "Stage5FeatureBundle",
    "assemble_controller_input",
    "build_stage5_features",
]
