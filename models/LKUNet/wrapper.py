from __future__ import annotations

from copy import deepcopy

import torch
from torch import nn

from models.LKUNet.configs import CONFIGS as _CFG_REGISTRY
from models.LKUNet.model import SpatialTransform, UNet
from utils.field import compose_flows
from utils.spatial import SpatialTransformer


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


class LkuNetCascadeL2(nn.Module):
    """LKU-Net as a full-resolution CTCF Level-2 backbone.

    LKU-Net predicts normalized grid flow. This wrapper converts it to voxel
    units, composes it with optional L1 init flow, and returns `(warped, flow)`.
    """

    def __init__(self, config):
        super().__init__()
        config_key = getattr(config, "lku_config", "LKU-8")
        if config_key not in _CFG_REGISTRY:
            raise KeyError(f"Unknown LKU-Net config '{config_key}'. Available: {list(_CFG_REGISTRY.keys())}")

        cfg = deepcopy(_CFG_REGISTRY[config_key])
        self.config_key = config_key
        self.vol_size = tuple(getattr(config, "img_size", cfg.img_size))
        self.unet = UNet(
            in_channel=cfg.in_channel,
            n_classes=cfg.n_classes,
            start_channel=cfg.start_channel,
        )
        self.spatial_transform = SpatialTransformer(self.vol_size)

        d, h, w = self.vol_size
        scale = torch.tensor(
            [(d - 1) / 2.0, (h - 1) / 2.0, (w - 1) / 2.0],
            dtype=torch.float32,
        ).view(1, 3, 1, 1, 1)
        self.register_buffer(name="flow_scale", tensor=scale, persistent=False)

    def forward(
        self,
        mov: torch.Tensor,
        fix: torch.Tensor,
        init_flow=None,
        return_all_flows: bool = False,
    ):
        mov_warped = mov
        if init_flow is not None:
            mov_warped = self.spatial_transform(mov, init_flow)

        flow_norm = self.unet(mov_warped, fix)
        flow_pred = flow_norm * self.flow_scale.to(device=flow_norm.device, dtype=flow_norm.dtype)

        flow_total = flow_pred
        if init_flow is not None:
            flow_total = compose_flows(flow_pred, init_flow)
        warped = self.spatial_transform(mov, flow_total)
        return warped, flow_total
