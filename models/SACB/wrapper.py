from __future__ import annotations

import torch
from torch import nn

from models.SACB.model import SACB_Net


class SACBSolo(nn.Module):
    """SACB-Net (Cheng et al., CVPR 2025) as a standalone baseline in the CTCF harness.

    SACB_Net.forward(moving, fixed) -> (warped, flow) already matches the CTCF convention:
    its flow is voxel-units, channel-first, full-res (the model warps with the TransMorph-style
    `utils.SpatialTransformer`, grid = arange voxel coords), so it drops straight into our
    Runner / validation exactly like VoxelMorph and CorrMLP — no flow conversion needed.

    Requires CUDA at forward time (SACB1 hardcodes `.cuda()` and uses kmeans_gpu); cannot be
    smoke-tested on CPU. Verify on the first GPU run that validation Dice is sane (>0.7).
    """

    def __init__(self, img_size: tuple[int, int, int], num_k: int = 7, ch_scale: int = 4):
        super().__init__()
        self.net = SACB_Net(inshape=tuple(img_size), num_k=num_k, ch_scale=ch_scale)

    def forward(self, mov: torch.Tensor, fix: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        warped, flow = self.net(mov, fix)
        return warped, flow
