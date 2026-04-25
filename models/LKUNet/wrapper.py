"""
LKU-Net solo wrapper for standalone training.

Mirrors the VxmDense wrapper pattern: instantiate UNet + SpatialTransform,
expose forward(mov, fix) -> (warped, flow_chw) where flow_chw is the
channel-first displacement tensor (B, 3, D, H, W) in NORMALIZED grid units.
"""
import torch.nn as nn

from models.LKUNet.model import UNet, SpatialTransform


class LkuNetSolo(nn.Module):
    """LKU-Net standalone registration network (UNet + SpatialTransform)."""
    def __init__(self, vol_size, in_channel=2, n_classes=3, start_channel=32):
        super().__init__()
        self.unet = UNet(in_channel=in_channel, n_classes=n_classes, start_channel=start_channel)
        self.transform = SpatialTransform(vol_size)

    def forward(self, mov, fix):
        # Channel-first flow (B, 3, D, H, W) with values in [-1, 1] (Softsign output).
        flow = self.unet(mov, fix)
        # SpatialTransform expects channel-last (B, D, H, W, 3).
        warped = self.transform(mov, flow.permute(0, 2, 3, 4, 1).contiguous())
        return warped, flow
