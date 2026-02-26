import torch
import torch.nn.functional as F
from torch import nn


class SpatialTransformer(nn.Module):
    def __init__(self, size, mode: str = "bilinear"):
        super().__init__()
        self.mode = mode
        vectors = [torch.arange(0, s) for s in size]
        grid = torch.stack(torch.meshgrid(vectors, indexing="ij")).unsqueeze(0).float()
        self.register_buffer("grid", grid, persistent=False)


    def forward(self, src: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
        new_locs = self.grid + flow
        shape = flow.shape[2:]
        for i in range(len(shape)):
            new_locs[:, i, ...] = 2.0 * (new_locs[:, i, ...] / (shape[i] - 1) - 0.5)
        if len(shape) == 2:
            new_locs = new_locs.permute(0, 2, 3, 1)[..., [1, 0]]
        elif len(shape) == 3:
            new_locs = new_locs.permute(0, 2, 3, 4, 1)[..., [2, 1, 0]]
        return F.grid_sample(src, new_locs, align_corners=False, mode=self.mode)


class RegisterModel(nn.Module):
    def __init__(self, img_size=(64, 256, 256), mode="bilinear"):
        super().__init__()
        self.spatial_trans = SpatialTransformer(img_size, mode)


    def forward(self, x):
        return self.spatial_trans(x[0], x[1])
