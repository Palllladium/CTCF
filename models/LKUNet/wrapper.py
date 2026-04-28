"""LKU-Net wrapper with forward(mov, fix) -> (warped, normalized_flow)."""
import torch.nn as nn

from models.LKUNet.model import UNet, SpatialTransform


class LkuNetSolo(nn.Module):
    """LKU-Net standalone registration network (UNet + SpatialTransform)."""
    def __init__(self, vol_size, in_channel=2, n_classes=3, start_channel=32):
        super().__init__()
        self.unet = UNet(in_channel=in_channel, n_classes=n_classes, start_channel=start_channel)
        self.transform = SpatialTransform(vol_size)


    def forward(self, mov, fix):
        flow = self.unet(mov, fix)
        warped = self.transform(mov, flow.permute(0, 2, 3, 4, 1).contiguous())
        return warped, flow
