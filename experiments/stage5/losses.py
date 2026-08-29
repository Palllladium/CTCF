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
    requested_forward = psi_forward + delta_forward
    requested_reverse = psi_reverse + delta_reverse
    warped_forward = sample_at_psi(moving_forward, requested_forward)
    warped_reverse = sample_at_psi(moving_reverse, requested_reverse)
    ncc = NCCVxm(win=(config.ncc_window,) * 3)
    similarity = 0.5 * (ncc(warped_forward, fixed_forward) + ncc(warped_reverse, fixed_reverse))
    diffusion = 0.5 * (_diffusion(delta_forward) + _diffusion(delta_reverse))
    inverse = 0.5 * (
        _composition_residual(requested_forward, requested_reverse).square().mean()
        + _composition_residual(requested_reverse, requested_forward).square().mean()
    )
    magnitude = 0.5 * (delta_forward.square().mean() + delta_reverse.square().mean())
    loss = (
        config.ncc_weight * similarity
        + config.diffusion_weight * diffusion
        + config.inverse_consistency_weight * inverse
        + config.magnitude_weight * magnitude
    )
    if not bool(torch.isfinite(loss)):
        raise FloatingPointError("Stage5 controller loss became non-finite")
    logs = {
        "loss": float(loss.detach().item()),
        "ncc": float(similarity.detach().item()),
        "diffusion": float(diffusion.detach().item()),
        "inverse_consistency": float(inverse.detach().item()),
        "magnitude": float(magnitude.detach().item()),
    }
    return loss, logs


__all__ = ["ControllerLossConfig", "controller_objective"]
