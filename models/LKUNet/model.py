"""
LKU-Net: U-Net with Large-Kernel encoder blocks (Jia et al., WBIR 2022 / arxiv 2208.04939).

Original source: https://github.com/xi-jia/LKU-Net (LKU-Net_3D_OASIS/LKU-Net-Full-Resolution).
This port keeps the architecture intact, removes hardcoded .cuda() calls so the
module works on any device, and registers SpatialTransform's grids as buffers
to avoid re-creating them on every forward pass.

Key differences from VoxelMorph:
- LK_encoder block: parallel 3x3x3 + 5x5x5 + 1x1x1 conv summed with input residual
- 4-level encoder/decoder (uses all skip connections in full-res variant)
- Output flow uses Softsign activation (bounded to [-1, 1])
- Flow is in normalized grid coordinates (compatible with grid_sample directly),
  NOT voxel units - this differs from VoxelMorph/CTCF convention.

Model output convention: flow tensor (B, 3, D, H, W), channel-first.
SpatialTransform expects flow permuted to (B, D, H, W, 3) before warping.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class LK_encoder(nn.Module):
    """Large-kernel encoder block: parallel 3x3x3 + KxKxK + 1x1x1 + residual."""
    def __init__(self, in_channels, out_channels, kernel_size=5, stride=1, padding=2, bias=False):
        super().__init__()
        self.layer_regularKernel = nn.Conv3d(in_channels, out_channels, 3, stride=1, padding=1, bias=bias)
        self.layer_largeKernel = nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias)
        self.layer_oneKernel = nn.Conv3d(in_channels, out_channels, 1, stride=1, padding=0, bias=bias)
        self.layer_nonlinearity = nn.PReLU()

    def forward(self, inputs):
        outputs = (
            self.layer_regularKernel(inputs)
            + self.layer_largeKernel(inputs)
            + self.layer_oneKernel(inputs)
            + inputs
        )
        return self.layer_nonlinearity(outputs)


class UNet(nn.Module):
    """LKU-Net: 4-level U-Net with LK_encoder blocks at every encoder stage.

    Args:
        in_channel: input channels (2 for concat[mov, fix]).
        n_classes: output channels (3 for displacement field).
        start_channel: base channel count. Paper uses 32 for full-res OASIS (~2M params).
    """
    def __init__(self, in_channel=2, n_classes=3, start_channel=32):
        super().__init__()
        self.in_channel = in_channel
        self.n_classes = n_classes
        self.start_channel = start_channel
        bias_opt = True
        sc = start_channel

        self.eninput = self._encoder(in_channel, sc, bias=bias_opt)
        self.ec1 = self._encoder(sc, sc, bias=bias_opt)
        self.ec2 = self._encoder(sc, sc * 2, stride=2, bias=bias_opt)
        self.ec3 = LK_encoder(sc * 2, sc * 2, kernel_size=5, stride=1, padding=2, bias=bias_opt)
        self.ec4 = self._encoder(sc * 2, sc * 4, stride=2, bias=bias_opt)
        self.ec5 = LK_encoder(sc * 4, sc * 4, kernel_size=5, stride=1, padding=2, bias=bias_opt)
        self.ec6 = self._encoder(sc * 4, sc * 8, stride=2, bias=bias_opt)
        self.ec7 = LK_encoder(sc * 8, sc * 8, kernel_size=5, stride=1, padding=2, bias=bias_opt)
        self.ec8 = self._encoder(sc * 8, sc * 8, stride=2, bias=bias_opt)
        self.ec9 = LK_encoder(sc * 8, sc * 8, kernel_size=5, stride=1, padding=2, bias=bias_opt)

        self.dc1 = self._encoder(sc * 8 + sc * 8, sc * 8, kernel_size=3, stride=1, bias=bias_opt)
        self.dc2 = self._encoder(sc * 8, sc * 4, kernel_size=3, stride=1, bias=bias_opt)
        self.dc3 = self._encoder(sc * 4 + sc * 4, sc * 4, kernel_size=3, stride=1, bias=bias_opt)
        self.dc4 = self._encoder(sc * 4, sc * 2, kernel_size=3, stride=1, bias=bias_opt)
        self.dc5 = self._encoder(sc * 2 + sc * 2, sc * 4, kernel_size=3, stride=1, bias=bias_opt)
        self.dc6 = self._encoder(sc * 4, sc * 2, kernel_size=3, stride=1, bias=bias_opt)
        self.dc7 = self._encoder(sc * 2 + sc, sc * 2, kernel_size=3, stride=1, bias=bias_opt)
        self.dc8 = self._encoder(sc * 2, sc * 2, kernel_size=3, stride=1, bias=bias_opt)
        self.dc9 = self._outputs(sc * 2, n_classes, kernel_size=3, stride=1, padding=1, bias=False)

        self.up1 = self._decoder(sc * 8, sc * 8)
        self.up2 = self._decoder(sc * 4, sc * 4)
        self.up3 = self._decoder(sc * 2, sc * 2)
        self.up4 = self._decoder(sc * 2, sc * 2)

    @staticmethod
    def _encoder(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False):
        return nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias),
            nn.PReLU(),
        )

    @staticmethod
    def _decoder(in_channels, out_channels, kernel_size=2, stride=2, padding=0, output_padding=0, bias=True):
        return nn.Sequential(
            nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride,
                               padding=padding, output_padding=output_padding, bias=bias),
            nn.PReLU(),
        )

    @staticmethod
    def _outputs(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False):
        # Softsign activation bounds the flow to [-1, 1] (normalized grid units).
        return nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias),
            nn.Softsign(),
        )

    def forward(self, x, y):
        x_in = torch.cat((x, y), 1)
        e0 = self.eninput(x_in)
        e0 = self.ec1(e0)

        e1 = self.ec2(e0)
        e1 = self.ec3(e1)

        e2 = self.ec4(e1)
        e2 = self.ec5(e2)

        e3 = self.ec6(e2)
        e3 = self.ec7(e3)

        e4 = self.ec8(e3)
        e4 = self.ec9(e4)

        d0 = torch.cat((self.up1(e4), e3), 1)
        d0 = self.dc1(d0)
        d0 = self.dc2(d0)

        d1 = torch.cat((self.up2(d0), e2), 1)
        d1 = self.dc3(d1)
        d1 = self.dc4(d1)

        d2 = torch.cat((self.up3(d1), e1), 1)
        d2 = self.dc5(d2)
        d2 = self.dc6(d2)

        d3 = torch.cat((self.up4(d2), e0), 1)
        d3 = self.dc7(d3)
        d3 = self.dc8(d3)

        return self.dc9(d3)


class SpatialTransform(nn.Module):
    """LKU-Net spatial transform expecting flow in NORMALIZED grid units, channel-LAST.

    Input flow: (B, D, H, W, 3) with values typically in [-1, 1] (Softsign output).
    The grid sampling positions are constructed as `grid + flow` directly,
    where grid is built from `linspace(-1, 1, ...)`.
    """
    def __init__(self, vol_size):
        super().__init__()
        d, h, w = vol_size
        grid_d, grid_h, grid_w = torch.meshgrid(
            torch.linspace(-1, 1, d), torch.linspace(-1, 1, h), torch.linspace(-1, 1, w),
            indexing="ij",
        )
        # Stored as buffers so they follow .to(device) and don't allocate per forward.
        self.register_buffer("grid_d", grid_d.float(), persistent=False)
        self.register_buffer("grid_h", grid_h.float(), persistent=False)
        self.register_buffer("grid_w", grid_w.float(), persistent=False)

    def forward(self, mov_image, flow, mode: str = "bilinear"):
        flow_d = flow[..., 0]
        flow_h = flow[..., 1]
        flow_w = flow[..., 2]
        disp_d = self.grid_d + flow_d
        disp_h = self.grid_h + flow_h
        disp_w = self.grid_w + flow_w
        sample_grid = torch.stack((disp_w, disp_h, disp_d), dim=4)
        return F.grid_sample(mov_image, sample_grid, mode=mode, align_corners=True)


def smoothloss(y_pred):
    """Axis-scaled smoothness regularizer (LKU-Net's specific form).

    Note: the per-axis multiplication by spatial dimension (* d2 / 2 etc.) is
    LKU-Net's particular choice; it differs from VoxelMorph's plain L2 gradient
    regularizer (utils.NCCVxm Grad3d) which does not scale by spatial size.
    """
    d2, h2, w2 = y_pred.shape[-3:]
    dy = torch.abs(y_pred[:, :, 1:, :, :] - y_pred[:, :, :-1, :, :]) / 2 * d2
    dx = torch.abs(y_pred[:, :, :, 1:, :] - y_pred[:, :, :, :-1, :]) / 2 * h2
    dz = torch.abs(y_pred[:, :, :, :, 1:] - y_pred[:, :, :, :, :-1]) / 2 * w2
    return (torch.mean(dx * dx) + torch.mean(dy * dy) + torch.mean(dz * dz)) / 3.0


def mse_loss(y_true, y_pred):
    return torch.mean((y_true - y_pred) ** 2)


def sad_loss(y_true, y_pred):
    return torch.mean(torch.abs(y_true - y_pred))
