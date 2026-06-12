from __future__ import annotations

import torch
from torch import nn

from models.CorrMLP.networks import CorrMLP  # verbatim upstream (GPL-3.0), not our code


class CorrMLPSolo(nn.Module):
    """CorrMLP (Meng et al., CVPR 2024) as a standalone baseline in the CTCF harness.

    Adapts the upstream signature `CorrMLP.forward(fixed, moving) -> (warped, flow)` to the
    CTCF convention `forward(moving, fixed) -> (warped, flow)` (voxel-units, channel-first,
    full-res). The network and its loss (NCC win=9 + diffusion) match our OASIS/IXI protocol,
    so it trains under the same Runner as VoxelMorph for an apples-to-apples gap check.
    """

    def __init__(
        self,
        enc_channels: int = 8,
        dec_channels: int = 16,
        use_checkpoint: bool = True,
    ):
        super().__init__()
        self.net = CorrMLP(
            in_channels=1,
            enc_channels=enc_channels,
            dec_channels=dec_channels,
            use_checkpoint=use_checkpoint,
        )

    def forward(self, mov: torch.Tensor, fix: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        warped, flow = self.net(fix, mov)  # upstream order is (fixed, moving)
        return warped, flow
