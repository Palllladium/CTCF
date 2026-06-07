from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class CA(nn.Module):
    """Channel attention block for 3D feature maps."""

    def __init__(
        self,
        num_feat: int,
        squeeze_factor: int = 16,
    ):
        super().__init__()
        hidden = max(1, num_feat // squeeze_factor)
        self.net = nn.Sequential(
            nn.AdaptiveAvgPool3d(output_size=1),
            nn.Conv3d(
                in_channels=num_feat,
                out_channels=hidden,
                kernel_size=1,
                bias=True,
            ),
            nn.ReLU(inplace=True),
            nn.Conv3d(
                in_channels=hidden,
                out_channels=num_feat,
                kernel_size=1,
                bias=True,
            ),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.net(x)


class CAB(nn.Module):
    """Channel attention block with residual fusion: Conv3d -> GELU -> Conv3d -> CA."""

    def __init__(
        self,
        num_feat: int,
        compress_ratio: int = 3,
        squeeze_factor: int = 30,
    ):
        super().__init__()
        hidden = max(1, num_feat // compress_ratio)
        self.body = nn.Sequential(
            nn.Conv3d(
                in_channels=num_feat,
                out_channels=hidden,
                kernel_size=3,
                padding=1,
                bias=True,
            ),
            nn.GELU(),
            nn.Conv3d(
                in_channels=hidden,
                out_channels=num_feat,
                kernel_size=3,
                padding=1,
                bias=True,
            ),
            CA(num_feat=num_feat, squeeze_factor=squeeze_factor),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x)


class Conv3dAct(nn.Module):
    """Conv3d followed by an activation (GELU or LeakyReLU)."""

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        k: int = 3,
        act: str = "gelu",
    ):
        super().__init__()
        self.conv = nn.Conv3d(
            in_channels=in_ch,
            out_channels=out_ch,
            kernel_size=k,
            padding=k // 2,
            bias=True,
        )

        if act == "gelu":
            self.act = nn.GELU()
        elif act == "lrelu":
            self.act = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        else:
            raise ValueError(f"Unknown act: {act}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.conv(x))


class ResidualContext3D(nn.Module):
    """Residual block with a (possibly dilated) first conv. Second conv is zero-initialised."""

    def __init__(
        self,
        channels: int,
        dilation: int = 1,
        scale: float = 0.1,
    ):
        super().__init__()
        self.scale = scale

        self.conv1 = nn.Conv3d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=3,
            padding=dilation,
            dilation=dilation,
            bias=False,
        )
        self.in1 = nn.InstanceNorm3d(num_features=channels, affine=True)
        self.act1 = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        self.conv2 = nn.Conv3d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=3,
            padding=1,
            bias=False,
        )
        self.in2 = nn.InstanceNorm3d(num_features=channels, affine=True)
        self.act2 = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        nn.init.zeros_(self.conv2.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv1(x)
        y = self.in1(y)
        y = self.act1(y)

        y = self.conv2(y)
        y = self.in2(y)
        y = self.act2(y)

        return x + self.scale * y


def _match_size_3d(x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    """Center-pad or center-crop the last three dims of `x` to match `ref`."""
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
    return x[..., sd : sd + rd, sh : sh + rh, sw : sw + rw]


class SRUpBlock3D(nn.Module):
    """Trilinear x2 upsample, optional skip concat, two Conv3dAct refinement layers."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        skip_channels: int = 0,
    ):
        super().__init__()
        self.skip_channels = skip_channels
        self.up = nn.Upsample(
            scale_factor=2,
            mode="trilinear",
            align_corners=False,
        )
        self.conv1 = Conv3dAct(
            in_ch=in_channels + self.skip_channels,
            out_ch=out_channels,
            k=3,
            act="gelu",
        )
        self.conv2 = Conv3dAct(
            in_ch=out_channels,
            out_ch=out_channels,
            k=3,
            act="gelu",
        )

    def forward(
        self,
        x: torch.Tensor,
        skip: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = self.up(x)

        if self.skip_channels > 0:
            if skip is None:
                raise ValueError("SRUpBlock3D expects skip tensor but got None.")
            x = _match_size_3d(x, skip)
            x = torch.cat([x, skip], dim=1)

        return self.conv2(self.conv1(x))


def resize_3d(x: torch.Tensor, scale_factor: float) -> torch.Tensor:
    """Trilinear (re)sampling of a 5D tensor by `scale_factor`."""
    return F.interpolate(x, scale_factor=scale_factor, mode="trilinear", align_corners=False)


def upsample_flow(flow: torch.Tensor, scale_factor: float = 2.0) -> torch.Tensor:
    """Trilinear upsample of a displacement field with magnitude rescaling.

    Flow values are in voxel units, so enlarging the grid by `scale_factor` requires
    multiplying every displacement vector by the same factor.
    """
    if scale_factor == 1:
        return flow
    return resize_3d(flow, scale_factor) * scale_factor
