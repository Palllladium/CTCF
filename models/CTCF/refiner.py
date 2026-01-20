# models/CTCF/refiner.py

from __future__ import annotations

import torch
import torch.nn as nn

from models.TransMorph_DCA.model import Conv3dReLU


class FlowRefiner3D(nn.Module):
    """
    Refinement на 1/2 с error-map.
    Вход: mov_w_half, fix_half, err_half -> concat(B, 3, D,H,W)
    Выход: delta_flow_half (B,3,D,H,W)
    """
    def __init__(self, base_ch: int = 16, in_ch: int = 3):
        super().__init__()
        self.net = nn.Sequential(
            Conv3dReLU(in_ch, base_ch, kernel_size=3, padding=1),
            Conv3dReLU(base_ch, base_ch, kernel_size=3, padding=1),
            Conv3dReLU(base_ch, base_ch * 2, kernel_size=3, stride=2, padding=1),
            Conv3dReLU(base_ch * 2, base_ch * 2, kernel_size=3, padding=1),
            nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False),
            Conv3dReLU(base_ch * 2, base_ch, kernel_size=3, padding=1),
        )
        self.out = nn.Conv3d(base_ch, 3, kernel_size=3, padding=1, bias=True)

        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, mov_w: torch.Tensor, fix: torch.Tensor, err: torch.Tensor) -> torch.Tensor:
        x = torch.cat([mov_w, fix, err], dim=1)  # (B,3,D,H,W)
        f = self.net(x)
        dflow = self.out(f)
        return dflow