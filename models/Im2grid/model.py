"""Thin im2grid-Lite adapter; upstream code is downloaded separately under GPL-3.0."""

from __future__ import annotations

import torch
from torch import nn

from .fetch import PORT_VERSION, REVISION

METHOD = {
    "name": "im2grid_lite",
    "revision": REVISION,
    "port_version": PORT_VERSION,
    "training_padding": "border",
    "evaluation_padding": "zeros",
}


def to_canonical(grid: torch.Tensor) -> torch.Tensor:
    """Convert an upstream absolute voxel grid (z,y,x) to a canonical displacement."""
    axes = [torch.arange(n, device=grid.device, dtype=torch.float32) for n in grid.shape[2:]]
    identity = torch.stack(torch.meshgrid(*axes, indexing="ij"))[None]
    return grid.float() - identity


def local_smoothness(grids: list[torch.Tensor]) -> torch.Tensor:
    """Equation (5): sum native-level diffusion penalties of normalized local displacements."""
    terms = []
    for grid in grids:
        axes = [torch.arange(n, device=grid.device, dtype=torch.float32) / (n - 1) * 2 - 1 for n in grid.shape[2:]]
        displacement = grid.float() - torch.stack(torch.meshgrid(*axes, indexing="ij"))[None]
        terms.append(sum(displacement.diff(dim=axis).square().mean() for axis in (2, 3, 4)))
    return torch.stack(terms).sum()


class Im2gridLite(nn.Module):
    def __init__(self):
        super().__init__()
        try:
            from .vfa_port.model import Im2gridDecoder, Im2gridEncoder
        except ModuleNotFoundError as exc:
            raise RuntimeError("im2grid source is missing: python -m models.Im2grid.fetch") from exc
        self.encoder = Im2gridEncoder(in_channels=1, downsamples=4, start_channels=8, max_channels=64)
        self.decoder = Im2gridDecoder(downsamples=4, matching_channels=8, start_channels=8, max_channels=64, skip={0})

    def forward(self, moving: torch.Tensor, fixed: torch.Tensor):
        if moving.shape != fixed.shape or any(n < 32 or n % 16 for n in moving.shape[2:]):
            raise ValueError("im2grid needs equal image shapes with axes divisible by 16 and at least 32")
        composed, local, _ = self.decoder(self.encoder(fixed), self.encoder(moving), None)
        return composed[-1].float(), local

    def losses(self, moving: torch.Tensor, fixed: torch.Tensor):
        from .vfa_port.sampling import grid_sampler

        grid, local = self(moving, fixed)
        with torch.autocast(device_type=moving.device.type, enabled=False):
            warped = grid_sampler(moving.float(), grid)
            similarity = (warped - fixed.float()).square().mean()
            smoothness = local_smoothness(local[:-1])  # The skipped finest grid is constant identity.
        return similarity, smoothness
