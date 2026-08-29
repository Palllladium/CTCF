"""Learned spatial controller used by the Stage-5 causal experiment."""

from __future__ import annotations

import math
from dataclasses import dataclass
from types import MappingProxyType
from typing import Final

import torch
import torch.nn.functional as F
from torch import nn

from utils.field import identity_collar

STAGE5_VARIANTS: Final = (
    "F0",
    "F2V",
    "F2S",
    "F2P",
    "F4P",
    "F24P",
    "A2P",
    "A24P",
)

# Both posterior slots use the frozen lexicographic z-y-x candidate order. Search
# support is represented explicitly and every unavailable group is still written as
# literal zeros. A supported posterior has unit mass, while ``valid_support`` records
# whether the frozen search primitive produced that posterior at the voxel.
STAGE5_INPUT_CHANNELS: Final = (
    "fixed_image",
    "warped_moving_image",
    "certified_source_displacement_z",
    "certified_source_displacement_y",
    "certified_source_displacement_x",
    "s2_vector_z",
    "s2_vector_y",
    "s2_vector_x",
    *(f"s2_posterior_{index:02d}" for index in range(27)),
    "s2_entropy",
    "s2_top2_gap",
    "s2_valid_support",
    "s4_vector_z",
    "s4_vector_y",
    "s4_vector_x",
    *(f"s4_posterior_{index:02d}" for index in range(27)),
    "s4_entropy",
    "s4_top2_gap",
    "s4_valid_support",
)
STAGE5_INPUT_CHANNEL_COUNT: Final = 71
STAGE5_FREE_RESIDUAL_HEAD: Final = slice(0, 3)
STAGE5_S2_ATTENUATION_HEAD: Final = 3
STAGE5_S4_ATTENUATION_HEAD: Final = 4
STAGE5_RESERVED_HEAD: Final = 5

STAGE5_CHANNEL_GROUPS: Final = MappingProxyType(
    {
        "context": slice(0, 5),
        "s2_vector": slice(5, 8),
        "s2_posterior": slice(8, 35),
        "s2_stats": slice(35, 38),
        "s4_vector": slice(38, 41),
        "s4_posterior": slice(41, 68),
        "s4_stats": slice(68, 71),
    }
)

STAGE5_VARIANT_GROUPS: Final = MappingProxyType(
    {
        "F0": ("context",),
        "F2V": ("context", "s2_vector"),
        "F2S": ("context", "s2_vector", "s2_stats"),
        "F2P": ("context", "s2_vector", "s2_posterior"),
        "F4P": ("context", "s4_vector", "s4_posterior"),
        "F24P": ("context", "s2_vector", "s2_posterior", "s4_vector", "s4_posterior"),
        "A2P": ("context", "s2_vector", "s2_posterior"),
        "A24P": ("context", "s2_vector", "s2_posterior", "s4_vector", "s4_posterior"),
    }
)


def _validate_variant(variant: str) -> str:
    if variant not in STAGE5_VARIANT_GROUPS:
        raise ValueError(f"unknown Stage-5 controller variant {variant!r}; expected one of {STAGE5_VARIANTS}")
    return variant


def stage5_variant_mask(variant: str, *, device: torch.device | None = None) -> torch.Tensor:
    """Return the immutable logical channel mask as ``[1,71,1,1,1]``."""

    frozen_variant = _validate_variant(variant)
    mask = torch.zeros((1, STAGE5_INPUT_CHANNEL_COUNT, 1, 1, 1), dtype=torch.bool, device=device)
    for group in STAGE5_VARIANT_GROUPS[frozen_variant]:
        mask[:, STAGE5_CHANNEL_GROUPS[group]] = True
    return mask


def _validate_feature_tensor(features: torch.Tensor) -> None:
    if features.ndim != 5 or features.shape[1] != STAGE5_INPUT_CHANNEL_COUNT:
        raise ValueError(
            "Stage-5 controller input must have shape "
            f"[B,{STAGE5_INPUT_CHANNEL_COUNT},D,H,W], got {tuple(features.shape)}"
        )
    if features.shape[0] < 1 or min(features.shape[-3:]) < 2:
        raise ValueError("Stage-5 controller input must contain a non-empty 3-D volume with dimensions >= 2")
    if not features.is_floating_point():
        raise TypeError("Stage-5 controller input must use a floating dtype")
    if not bool(torch.isfinite(features).all()):
        raise ValueError("Stage-5 controller input must be finite before variant masking")


def mask_stage5_features(features: torch.Tensor, variant: str) -> torch.Tensor:
    """Zero every unavailable group without preserving negative-zero payload bits."""

    _validate_feature_tensor(features)
    mask = stage5_variant_mask(variant, device=features.device)
    return torch.where(mask, features, torch.zeros_like(features))


def _validate_proposal(proposal: torch.Tensor | None, reference: torch.Tensor, label: str) -> torch.Tensor:
    if proposal is None:
        raise ValueError(f"{label} is required by the selected attenuation policy")
    expected = (reference.shape[0], 3, *reference.shape[-3:])
    if proposal.shape != expected:
        raise ValueError(f"{label} must have shape {expected}, got {tuple(proposal.shape)}")
    if not proposal.is_floating_point() or proposal.device != reference.device or proposal.dtype != reference.dtype:
        raise ValueError(f"{label} must share the controller output dtype and device")
    if not bool(torch.isfinite(proposal).all()):
        raise ValueError(f"{label} must be finite")
    return proposal


def _ste_unit_interval(raw: torch.Tensor) -> torch.Tensor:
    """Clamp to [0,1] in the forward pass while retaining identity gradients.

    A differentiable non-negative gate cannot be both exactly zero and have a non-zero
    derivative at its zero initial state. The straight-through contract makes that
    optimisation choice explicit rather than relying on a framework-specific boundary
    derivative of ``clamp`` or ``relu``.
    """

    bounded = raw.clamp(0.0, 1.0)
    return raw + (bounded - raw).detach()


def _normalise_two_attenuations(first: torch.Tensor, second: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    total = first + second
    denominator = torch.maximum(torch.ones_like(total), total)
    return first / denominator, second / denominator


class _ControllerBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv3d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, channels),
            nn.SiLU(inplace=True),
            nn.Conv3d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, channels),
        )
        self.activation = nn.SiLU(inplace=True)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.activation(value + self.body(value))


@dataclass(frozen=True, slots=True)
class Stage5ControllerOutput:
    variant: str
    requested_delta: torch.Tensor
    raw_head: torch.Tensor
    alpha_s2: torch.Tensor | None
    alpha_s4: torch.Tensor | None


class Stage5SpatialController(nn.Module):
    """Small equal-capacity spatial controller for the frozen Stage-5 matrix.

    Every variant executes this exact module and owns the same parameter shapes. The
    variant changes only a literal-zero input mask and the fixed interpretation of the
    common six-channel head. Search-derived values never set the free-residual scale.
    """

    def __init__(
        self,
        *,
        width: int = 16,
        free_residual_limit_voxels: float = 2.0,
        collar_width: int = 7,
    ) -> None:
        super().__init__()
        if isinstance(width, bool) or not isinstance(width, int) or width < 4:
            raise ValueError("width must be an integer >= 4")
        if (
            isinstance(free_residual_limit_voxels, bool)
            or not isinstance(free_residual_limit_voxels, (int, float))
            or not math.isfinite(float(free_residual_limit_voxels))
            or float(free_residual_limit_voxels) <= 0.0
        ):
            raise ValueError("free_residual_limit_voxels must be finite and positive")
        if isinstance(collar_width, bool) or not isinstance(collar_width, int) or collar_width < 1:
            raise ValueError("collar_width must be a positive integer")

        deep_width = width * 2
        self.free_residual_limit_voxels = float(free_residual_limit_voxels)
        self.collar_width = collar_width

        self.stem = nn.Sequential(
            nn.Conv3d(STAGE5_INPUT_CHANNEL_COUNT, width, kernel_size=1, bias=True),
            nn.GroupNorm(1, width),
            nn.SiLU(inplace=True),
        )
        self.encoder = _ControllerBlock(width)
        self.down = nn.Sequential(
            nn.Conv3d(width, deep_width, kernel_size=3, stride=2, padding=1, bias=True),
            nn.GroupNorm(1, deep_width),
            nn.SiLU(inplace=True),
        )
        self.bottleneck = _ControllerBlock(deep_width)
        self.up_projection = nn.Conv3d(deep_width, width, kernel_size=1, bias=True)
        self.fusion = nn.Sequential(
            nn.Conv3d(width * 2, width, kernel_size=3, padding=1, bias=True),
            nn.GroupNorm(1, width),
            nn.SiLU(inplace=True),
            _ControllerBlock(width),
        )
        self.head = nn.Conv3d(width, 6, kernel_size=3, padding=1, bias=True)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def _raw_head(self, masked_features: torch.Tensor) -> torch.Tensor:
        skip = self.encoder(self.stem(masked_features))
        deep = self.bottleneck(self.down(skip))
        up = F.interpolate(self.up_projection(deep), size=skip.shape[-3:], mode="trilinear", align_corners=False)
        # Controller convolution may run under AMP, but the requested physical field
        # is always materialised in FP32 before attenuation and boundary tapering.
        learned = self.head(self.fusion(torch.cat((skip, up), dim=1))).float()
        raw = torch.cat(
            (
                learned[:, :STAGE5_RESERVED_HEAD],
                torch.zeros_like(learned[:, STAGE5_RESERVED_HEAD : STAGE5_RESERVED_HEAD + 1]),
            ),
            dim=1,
        )
        if not bool(torch.isfinite(raw).all()):
            raise FloatingPointError("Stage-5 controller produced a non-finite head")
        return raw

    def _apply_policy(
        self,
        raw: torch.Tensor,
        variant: str,
        *,
        s2_proposal: torch.Tensor | None,
        s4_proposal: torch.Tensor | None,
    ) -> Stage5ControllerOutput:
        alpha_s2 = None
        alpha_s4 = None
        if variant.startswith("F"):
            delta = torch.tanh(raw[:, STAGE5_FREE_RESIDUAL_HEAD]) * self.free_residual_limit_voxels
        elif variant == "A2P":
            proposal_s2 = _validate_proposal(s2_proposal, raw, "s2_proposal")
            alpha_s2 = _ste_unit_interval(raw[:, STAGE5_S2_ATTENUATION_HEAD : STAGE5_S2_ATTENUATION_HEAD + 1])
            delta = alpha_s2 * proposal_s2
        elif variant == "A24P":
            proposal_s2 = _validate_proposal(s2_proposal, raw, "s2_proposal")
            proposal_s4 = _validate_proposal(s4_proposal, raw, "s4_proposal")
            raw_s2 = _ste_unit_interval(raw[:, STAGE5_S2_ATTENUATION_HEAD : STAGE5_S2_ATTENUATION_HEAD + 1])
            raw_s4 = _ste_unit_interval(raw[:, STAGE5_S4_ATTENUATION_HEAD : STAGE5_S4_ATTENUATION_HEAD + 1])
            alpha_s2, alpha_s4 = _normalise_two_attenuations(raw_s2, raw_s4)
            delta = alpha_s2 * proposal_s2 + alpha_s4 * proposal_s4
        else:  # guarded by _validate_variant; retained to keep this method fail-closed.
            raise RuntimeError(f"unsupported Stage-5 output policy: {variant}")

        tapered = identity_collar(delta, width=self.collar_width)
        if not bool(torch.isfinite(tapered).all()):
            raise FloatingPointError("Stage-5 output policy produced a non-finite residual")
        return Stage5ControllerOutput(
            variant=variant,
            requested_delta=tapered,
            raw_head=raw,
            alpha_s2=alpha_s2,
            alpha_s4=alpha_s4,
        )

    def forward(
        self,
        features: torch.Tensor,
        variant: str,
        *,
        s2_proposal: torch.Tensor | None = None,
        s4_proposal: torch.Tensor | None = None,
    ) -> Stage5ControllerOutput:
        """Request a residual under the selected frozen output policy.

        The S2/S4 vector *input channels* are divided by their stride by the feature
        builder. ``s2_proposal`` and ``s4_proposal`` are separate physical vectors in
        full-resolution voxel units. The attenuation policies consume only those
        physical proposal arguments, never reconstruct their amplitude from the
        stride-normalised controller channels.
        """

        frozen_variant = _validate_variant(variant)
        masked = mask_stage5_features(features, frozen_variant)
        raw = self._raw_head(masked)
        return self._apply_policy(
            raw,
            frozen_variant,
            s2_proposal=s2_proposal,
            s4_proposal=s4_proposal,
        )


__all__ = [
    "STAGE5_CHANNEL_GROUPS",
    "STAGE5_FREE_RESIDUAL_HEAD",
    "STAGE5_INPUT_CHANNELS",
    "STAGE5_INPUT_CHANNEL_COUNT",
    "STAGE5_RESERVED_HEAD",
    "STAGE5_S2_ATTENUATION_HEAD",
    "STAGE5_S4_ATTENUATION_HEAD",
    "STAGE5_VARIANTS",
    "STAGE5_VARIANT_GROUPS",
    "Stage5ControllerOutput",
    "Stage5SpatialController",
    "mask_stage5_features",
    "stage5_variant_mask",
]
