from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class SpatialTransformer(nn.Module):
    """Voxel-unit displacement warp via grid_sample with align_corners=False."""

    def __init__(self, size, mode: str = "bilinear"):
        super().__init__()
        self.mode = mode
        vectors = [torch.arange(start=0, end=s) for s in size]
        grid = torch.stack(torch.meshgrid(vectors, indexing="ij")).unsqueeze(0).float()
        self.register_buffer(name="grid", tensor=grid, persistent=False)

    def forward(self, src: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
        new_locs = self.grid + flow
        shape = flow.shape[2:]
        for i in range(len(shape)):
            new_locs[:, i, ...] = 2.0 * (new_locs[:, i, ...] / (shape[i] - 1) - 0.5)
        if len(shape) == 2:
            new_locs = new_locs.permute(0, 2, 3, 1)[..., [1, 0]]
        elif len(shape) == 3:
            new_locs = new_locs.permute(0, 2, 3, 4, 1)[..., [2, 1, 0]]
        # Project-wide convention: align_corners=False with (shape-1) normalization. Identity
        # flow lands half a voxel off; training absorbs it and all checkpoints depend on it.
        return F.grid_sample(src, new_locs, align_corners=False, mode=self.mode)


class RegisterModel(nn.Module):
    """Wrap SpatialTransformer to accept (src, flow) as a single tuple argument."""

    def __init__(self, img_size=(64, 256, 256), mode: str = "bilinear"):
        super().__init__()
        self.spatial_trans = SpatialTransformer(size=img_size, mode=mode)

    def forward(self, x):
        return self.spatial_trans(x[0], x[1])
