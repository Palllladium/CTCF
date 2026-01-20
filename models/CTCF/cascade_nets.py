# models/CTCF/cascade_nets.py

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.InstanceNorm3d(out_ch, affine=True),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv3d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.InstanceNorm3d(out_ch, affine=True),
            nn.LeakyReLU(0.1, inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CoarseFlowNetQuarter(nn.Module):
    """
    Level-1: quarter-res coarse flow predictor (conv-only).
    Input: mov_q, fix_q -> concat (B,2,D,H,W)
    Output: flow_q (B,3,D,H,W)
    """
    def __init__(self, base_ch: int = 16):
        super().__init__()
        c = int(base_ch)

        self.enc1 = ConvBlock(2, c)
        self.pool1 = nn.AvgPool3d(2)

        self.enc2 = ConvBlock(c, c * 2)
        self.pool2 = nn.AvgPool3d(2)

        self.bot = ConvBlock(c * 2, c * 4)

        self.up2 = nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False)
        self.dec2 = ConvBlock(c * 4 + c * 2, c * 2)

        self.up1 = nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False)
        self.dec1 = ConvBlock(c * 2 + c, c)

        self.out = nn.Conv3d(c, 3, kernel_size=3, padding=1, bias=True)

        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, mov: torch.Tensor, fix: torch.Tensor) -> torch.Tensor:
        x = torch.cat([mov, fix], dim=1)

        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        b = self.bot(self.pool2(e2))

        d2 = self.up2(b)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))

        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        return self.out(d1)


class FlowRefiner3D(nn.Module):
    """
    Level-3: half-res refinement using error-map.
    Inputs:
      - mov_warp_half: (B,1,D,H,W)
      - fix_half:      (B,1,D,H,W)
      - flow_half:     (B,3,D,H,W)  current flow (context)
    Output:
      - delta_flow_half: (B,3,D,H,W)
    """
    def __init__(self, base_ch: int = 16, error_mode: str = "absdiff"):
        super().__init__()
        self.error_mode = str(error_mode)
        c = int(base_ch)

        # channels: mov(1) + fix(1) + err(1) + flow(3) = 6
        self.enc1 = ConvBlock(6, c)
        self.pool1 = nn.AvgPool3d(2)

        self.enc2 = ConvBlock(c, c * 2)
        self.pool2 = nn.AvgPool3d(2)

        self.bot = ConvBlock(c * 2, c * 4)

        self.up2 = nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False)
        self.dec2 = ConvBlock(c * 4 + c * 2, c * 2)

        self.up1 = nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False)
        self.dec1 = ConvBlock(c * 2 + c, c)

        self.out = nn.Conv3d(c, 3, kernel_size=3, padding=1, bias=True)

        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def _error_map(self, mov_w: torch.Tensor, fix: torch.Tensor) -> torch.Tensor:
        if self.error_mode == "absdiff":
            return (mov_w - fix).abs()
        raise ValueError(f"Unsupported error_mode: {self.error_mode}")

    def forward(self, mov_warp: torch.Tensor, fix: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
        err = self._error_map(mov_warp, fix)
        x = torch.cat([mov_warp, fix, err, flow], dim=1)  # (B,6,...)

        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        b = self.bot(self.pool2(e2))

        d2 = self.up2(b)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))

        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        return self.out(d1)