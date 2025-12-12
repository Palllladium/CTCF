import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
import pystrum.pynd.ndutils as nd


class SpatialTransformer(nn.Module):
    """
    Double of ST from TM-DCA model.
    """
    def __init__(self, size, mode: str = "bilinear"):
        super().__init__()
        self.mode = mode

        vectors = [torch.arange(0, s) for s in size]
        grids = torch.meshgrid(vectors, indexing="ij")
        grid = torch.stack(grids)
        grid = torch.unsqueeze(grid, 0)
        grid = grid.type(torch.FloatTensor)
        self.register_buffer("grid", grid, persistent=False)

    def forward(self, src: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
        new_locs = self.grid + flow
        shape = flow.shape[2:]

        # normalize to [-1, 1]
        for i in range(len(shape)):
            new_locs[:, i, ...] = 2.0 * (new_locs[:, i, ...] / (shape[i] - 1) - 0.5)

        if len(shape) == 2:
            new_locs = new_locs.permute(0, 2, 3, 1)
            new_locs = new_locs[..., [1, 0]]
        elif len(shape) == 3:
            new_locs = new_locs.permute(0, 2, 3, 4, 1)
            new_locs = new_locs[..., [2, 1, 0]]

        return F.grid_sample(src, new_locs, align_corners=False, mode=self.mode)


class register_model(nn.Module):
    """
    Wrapper used in old trainers: forward([img, flow]) -> warped_img.
    """
    def __init__(self, img_size=(64, 256, 256), mode="bilinear"):
        super().__init__()
        self.spatial_trans = SpatialTransformer(img_size, mode)

    def forward(self, x):
        img, flow = x[0], x[1]
        return self.spatial_trans(img, flow)


def jacobian_determinant_vxm(disp: np.ndarray) -> np.ndarray:
    """
    Jacobian determinant of a displacement field (numpy version, VXM-style).
    disp: [*vol_shape, nb_dims] OR with transpose needed.
    """
    disp = disp.transpose(1, 2, 3, 0)
    volshape = disp.shape[:-1]
    nb_dims = len(volshape)
    assert nb_dims in (2, 3), "flow has to be 2D or 3D"

    grid_lst = nd.volsize2ndgrid(volshape)
    grid = np.stack(grid_lst, len(volshape))
    J = np.gradient(disp + grid)

    if nb_dims == 3:
        dx, dy, dz = J[0], J[1], J[2]
        Jdet0 = dx[..., 0] * (dy[..., 1] * dz[..., 2] - dy[..., 2] * dz[..., 1])
        Jdet1 = dx[..., 1] * (dy[..., 0] * dz[..., 2] - dy[..., 2] * dz[..., 0])
        Jdet2 = dx[..., 2] * (dy[..., 0] * dz[..., 1] - dy[..., 1] * dz[..., 0])
        return Jdet0 - Jdet1 + Jdet2
    else:
        dfdx, dfdy = J[0], J[1]
        return dfdx[..., 0] * dfdy[..., 1] - dfdy[..., 0] * dfdx[..., 1]