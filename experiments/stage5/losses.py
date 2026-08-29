from __future__ import annotations

from dataclasses import dataclass

import torch

from tools.analysis.search.transaction import sample_at_psi
from tools.analysis.stage5.primitives import require_finite, require_int
from utils import NCCVxm


@dataclass(frozen=True, slots=True)
class ControllerLossConfig:
    ncc_weight: float = 1.0
    diffusion_weight: float = 0.05
    inverse_consistency_weight: float = 0.1
    magnitude_weight: float = 0.01
    ncc_window: int = 7

    def __post_init__(self) -> None:
        # NaN survives ``value < 0.0`` and ``sum(...) <= 0.0``, so the weights are screened
        # for finiteness first; this class is the only owner of that contract.
        weights = [
            require_finite(getattr(self, name), f"Stage5 loss {name}", minimum=0.0, error=ValueError)
            for name in ("ncc_weight", "diffusion_weight", "inverse_consistency_weight", "magnitude_weight")
        ]
        if sum(weights) <= 0.0:
            raise ValueError("Stage5 loss weights must be non-negative and not all zero")
        if require_int(self.ncc_window, "Stage5 NCC window", minimum=1, error=ValueError) % 2 != 1:
            raise ValueError("Stage5 NCC window must be a positive odd integer")


def _diffusion(value: torch.Tensor) -> torch.Tensor:
    terms = []
    for dim in (2, 3, 4):
        upper = value.narrow(dim, 1, value.shape[dim] - 1)
        lower = value.narrow(dim, 0, value.shape[dim] - 1)
        terms.append((upper - lower).square().mean())
    return torch.stack(terms).mean()


def _composition_residual(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
    """Compose the actual Psi maps used by the align-corners-false deployment warp.

    Do not convert these fields back to the model's historical Phi convention: the
    Stage-5 objective measures the transformation that is really sampled at runtime.
    """

    return first + sample_at_psi(second, first)


def _require_finite(name: str, value: torch.Tensor) -> torch.Tensor:
    if value.numel() != 1 or not bool(torch.isfinite(value).all()):
        raise FloatingPointError(f"Stage5 controller objective term {name} became non-finite")
    return value


def controller_objective(
    fixed_forward: torch.Tensor,
    moving_forward: torch.Tensor,
    fixed_reverse: torch.Tensor,
    moving_reverse: torch.Tensor,
    psi_forward: torch.Tensor,
    psi_reverse: torch.Tensor,
    delta_forward: torch.Tensor,
    delta_reverse: torch.Tensor,
    *,
    config: ControllerLossConfig,
) -> tuple[torch.Tensor, dict[str, float]]:
    tensors = (
        fixed_forward,
        moving_forward,
        fixed_reverse,
        moving_reverse,
        psi_forward,
        psi_reverse,
        delta_forward,
        delta_reverse,
    )
    if any(not bool(torch.isfinite(value).all()) for value in tensors):
        raise FloatingPointError("Stage5 objective received a non-finite tensor")
    # The controller itself may run under CUDA FP16 autocast, but the objective must
    # not. NCCVxm squares local window sums; for the frozen 7^3 window even ordinary
    # normalized inputs can produce intermediates above the FP16 maximum (65504).
    # Casting here keeps one numerical contract for smoke and full training while
    # preserving gradients from the FP32 objective back through the AMP controller.
    with torch.autocast(device_type=fixed_forward.device.type, enabled=False):
        (
            fixed_forward,
            moving_forward,
            fixed_reverse,
            moving_reverse,
            psi_forward,
            psi_reverse,
            delta_forward,
            delta_reverse,
        ) = (value.float() for value in tensors)
        requested_forward = psi_forward + delta_forward
        requested_reverse = psi_reverse + delta_reverse
        warped_forward = sample_at_psi(moving_forward, requested_forward)
        warped_reverse = sample_at_psi(moving_reverse, requested_reverse)
        ncc = NCCVxm(win=(config.ncc_window,) * 3)
        ncc_forward = ncc(warped_forward, fixed_forward)
        ncc_reverse = ncc(warped_reverse, fixed_reverse)
        similarity = _require_finite("ncc", 0.5 * (ncc_forward + ncc_reverse))
        diffusion = _require_finite("diffusion", 0.5 * (_diffusion(delta_forward) + _diffusion(delta_reverse)))
        forward_composition_err = _composition_residual(requested_forward, requested_reverse).square().mean()
        reverse_composition_err = _composition_residual(requested_reverse, requested_forward).square().mean()
        inverse = _require_finite("inverse_consistency", 0.5 * (forward_composition_err + reverse_composition_err))
        magnitude = _require_finite("magnitude", 0.5 * (delta_forward.square().mean() + delta_reverse.square().mean()))
        loss = _require_finite(
            "loss",
            config.ncc_weight * similarity
            + config.diffusion_weight * diffusion
            + config.inverse_consistency_weight * inverse
            + config.magnitude_weight * magnitude,
        )
    logs = {
        "loss": float(loss.detach().item()),
        "ncc": float(similarity.detach().item()),
        "diffusion": float(diffusion.detach().item()),
        "inverse_consistency": float(inverse.detach().item()),
        "magnitude": float(magnitude.detach().item()),
    }
    return loss, logs


__all__ = ["ControllerLossConfig", "controller_objective"]
