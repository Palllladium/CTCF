from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class CA(nn.Module):
    """Channel Attention (RCAN-style) for 3D tensors [B,C,D,H,W]."""
    def __init__(self, num_feat: int, squeeze_factor: int = 16):
        super().__init__()
        hidden = max(1, num_feat // squeeze_factor)
        self.net = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Conv3d(num_feat, hidden, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv3d(hidden, num_feat, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.net(x)


class CAB(nn.Module):
    """Conv3d -> GELU -> Conv3d -> CA."""
    def __init__(self, num_feat: int, compress_ratio: int = 3, squeeze_factor: int = 30):
        super().__init__()
        hidden = max(1, num_feat // compress_ratio)
        self.body = nn.Sequential(
            nn.Conv3d(num_feat, hidden, kernel_size=3, padding=1, bias=True),
            nn.GELU(),
            nn.Conv3d(hidden, num_feat, kernel_size=3, padding=1, bias=True),
            CA(num_feat, squeeze_factor=squeeze_factor),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x)


class Conv3dAct(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, k: int = 3, act: str = "gelu"):
        super().__init__()
        self.conv = nn.Conv3d(in_ch, out_ch, kernel_size=k, padding=k // 2, bias=True)
        if act == "gelu":
            self.act = nn.GELU()
        elif act == "lrelu":
            self.act = nn.LeakyReLU(0.1, inplace=True)
        else:
            raise ValueError(f"Unknown act: {act}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.conv(x))


class ResidualContext3D(nn.Module):
    """Light residual context block for coarse flow backbone."""
    def __init__(self, channels: int, dilation: int = 1, scale: float = 0.1):
        super().__init__()
        self.scale = float(scale)
        self.conv1 = nn.Conv3d(channels, channels, kernel_size=3, padding=int(dilation), dilation=int(dilation), bias=False)
        self.in1 = nn.InstanceNorm3d(channels, affine=True)
        self.act1 = nn.LeakyReLU(0.1, inplace=True)
        self.conv2 = nn.Conv3d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.in2 = nn.InstanceNorm3d(channels, affine=True)
        self.act2 = nn.LeakyReLU(0.1, inplace=True)
        nn.init.zeros_(self.conv2.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv1(x)
        y = self.in1(y)
        y = self.act1(y)
        y = self.conv2(y)
        y = self.in2(y)
        y = self.act2(y)
        return x + self.scale * y


class RefineGate3D(nn.Module):
    """Spatial gate for level-3 residual flow refinement."""
    def __init__(self, in_ch: int = 4, base_ch: int = 8, init_bias: float = -1.2):
        super().__init__()
        c = int(base_ch)
        self.body = nn.Sequential(
            nn.Conv3d(int(in_ch), c, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(c, affine=True),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv3d(c, c, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(c, affine=True),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv3d(c, 1, kernel_size=1, bias=True),
        )
        nn.init.zeros_(self.body[-1].weight)
        nn.init.constant_(self.body[-1].bias, float(init_bias))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.body(x))


def _match_size_3d(x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    if x.dim() != 5 or ref.dim() != 5:
        raise ValueError(f"_match_size_3d expects 5D tensors, got x={x.shape}, ref={ref.shape}")

    xd, xh, xw = x.shape[-3:]
    rd, rh, rw = ref.shape[-3:]

    pd = max(0, rd - xd)
    ph = max(0, rh - xh)
    pw = max(0, rw - xw)
    if pd or ph or pw:
        pad = (pw // 2, pw - pw // 2, ph // 2, ph - ph // 2, pd // 2, pd - pd // 2)
        x = F.pad(x, pad)

    xd, xh, xw = x.shape[-3:]
    sd = (xd - rd) // 2 if xd > rd else 0
    sh = (xh - rh) // 2 if xh > rh else 0
    sw = (xw - rw) // 2 if xw > rw else 0
    return x[..., sd:sd + rd, sh:sh + rh, sw:sw + rw]


class SRUpBlock3D(nn.Module):
    """Safe SR-style x2 upsampling block with optional skip connection."""
    def __init__(self, in_channels: int, out_channels: int, skip_channels: int = 0):
        super().__init__()
        self.skip_channels = int(skip_channels)
        self.up = nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False)
        self.conv1 = Conv3dAct(in_channels + self.skip_channels, out_channels, k=3, act="gelu")
        self.conv2 = Conv3dAct(out_channels, out_channels, k=3, act="gelu")

    def forward(self, x: torch.Tensor, skip: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.up(x)
        if self.skip_channels > 0:
            if skip is None:
                raise ValueError("SRUpBlock3D expects skip tensor but got None.")
            x = _match_size_3d(x, skip)
            x = torch.cat([x, skip], dim=1)
        return self.conv2(self.conv1(x))


def upsample_flow(flow: torch.Tensor, scale_factor: float = 2.0) -> torch.Tensor:
    if scale_factor == 1:
        return flow
    flow_up = F.interpolate(flow, scale_factor=scale_factor, mode="trilinear", align_corners=False)
    return flow_up * float(scale_factor)


class LearnedFlowUpsample3D(nn.Module):
    """Convex upsampling for 3D flow fields (RAFT-style).

    For each low-res voxel, predicts convex combination weights over a
    3x3x3 neighborhood for each of its 2x2x2 = 8 high-res sub-voxels.
    This learns sub-voxel interpolation kernels sharper than trilinear.
    """
    def __init__(self, scale: int = 2, hidden_ch: int = 64):
        super().__init__()
        self.scale = int(scale)
        self.k = 3
        self.k3 = self.k ** 3   # 27 neighbors
        s3 = self.scale ** 3    # 8 sub-voxels
        self.mask_net = nn.Sequential(
            nn.Conv3d(3, hidden_ch, 3, padding=1, bias=False),
            nn.InstanceNorm3d(hidden_ch, affine=True),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv3d(hidden_ch, s3 * self.k3, 1, bias=True),
        )

    def forward(self, flow: torch.Tensor) -> torch.Tensor:
        B, C, D, H, W = flow.shape
        s = self.scale
        k = self.k

        # Predict per-sub-voxel weights: (B, s^3*k^3, D, H, W)
        mask = self.mask_net(flow)
        mask = mask.view(B, s ** 3, self.k3, D, H, W)
        mask = torch.softmax(mask, dim=2)  # convex combination over neighbors

        # Extract 3x3x3 neighborhoods: (B, C, k^3, D, H, W)
        p = k // 2
        flow_pad = F.pad(flow, (p, p, p, p, p, p), mode="replicate")
        neighbors = []
        for dz in range(k):
            for dy in range(k):
                for dx in range(k):
                    neighbors.append(flow_pad[:, :, dz:dz + D, dy:dy + H, dx:dx + W])
        neighbors = torch.stack(neighbors, dim=2)
        neighbors = neighbors * float(s)  # scale flow values for upsampling

        # Weighted sum per sub-voxel (loop over 8 for memory efficiency)
        parts = []
        for i in range(s ** 3):
            w = mask[:, i: i + 1]           # (B, 1, k^3, D, H, W)
            parts.append((w * neighbors).sum(dim=2))  # (B, C, D, H, W)
        flow_up = torch.stack(parts, dim=2)  # (B, C, s^3, D, H, W)

        # Reshape s^3 sub-voxels back to spatial: (B, C, s*D, s*H, s*W)
        flow_up = flow_up.view(B, C, s, s, s, D, H, W)
        flow_up = flow_up.permute(0, 1, 5, 2, 6, 3, 7, 4).contiguous()
        flow_up = flow_up.view(B, C, s * D, s * H, s * W)

        return flow_up
