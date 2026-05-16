"""EfficientMorph wrappers for standalone and CTCF-cascade use."""
from copy import deepcopy

import torch
import torch.nn as nn

from models.EfficientMorph.configs import CONFIGS as _CFG_REGISTRY
from models.EfficientMorph.model import EfficientMorph as _EfficientMorphCore
from utils.field import compose_flows
from utils.spatial import SpatialTransformer


class EfficientMorphSolo(nn.Module):
    """EfficientMorph standalone registration network."""
    def __init__(self, config_key: str, img_size=None):
        super().__init__()
        if config_key not in _CFG_REGISTRY:
            raise KeyError(f"Unknown EfficientMorph config '{config_key}'. "
                           f"Available: {list(_CFG_REGISTRY.keys())}")
        self.cfg = deepcopy(_CFG_REGISTRY[config_key])
        if img_size is not None:
            self.cfg.img_size = tuple(int(v) for v in img_size)
        self.cfg_key = config_key
        self.model = _EfficientMorphCore(self.cfg)


    def forward(self, mov, fix):
        x_in = torch.cat([mov, fix], dim=1)
        warped, flow = self.model(x_in)
        return warped, flow


    @property
    def spatial_trans(self):
        return self.model.spatial_trans


class EffMorphCascadeL2(nn.Module):
    """EfficientMorph as a full-resolution CTCF Level-2 backbone.

    Mirrors LkuNetCascadeL2 / MambaMorphCascadeL2 contract:
    forward(mov, fix, init_flow=None) -> (warped, flow_total) in voxel-units,
    channel-first (B, 3, D, H, W) at backbone's native (full) resolution.
    """

    def __init__(self, config):
        super().__init__()
        config_key = str(getattr(config, "effm_config", "EfficientMorph_2x3_2_hires"))
        if config_key not in _CFG_REGISTRY:
            raise KeyError(f"Unknown EfficientMorph config '{config_key}'. "
                           f"Available: {list(_CFG_REGISTRY.keys())}")

        cfg = deepcopy(_CFG_REGISTRY[config_key])
        cfg.img_size = tuple(int(v) for v in getattr(config, "img_size", cfg.img_size))
        self.config_key = config_key
        self.model = _EfficientMorphCore(cfg)
        # External spatial transformer (utils/spatial) for cascade composition
        self.vol_size = tuple(int(v) for v in cfg.img_size)
        self.spatial_transform = SpatialTransformer(self.vol_size)


    def forward(
        self,
        mov: torch.Tensor,
        fix: torch.Tensor,
        init_flow=None,
        return_all_flows: bool = False,
    ):
        # Pre-warp moving image by L1 init_flow if provided
        mov_warped = self.spatial_transform(mov, init_flow) if init_flow is not None else mov

        # EM core expects concatenated [mov, fix] along channel dim
        x_in = torch.cat([mov_warped, fix], dim=1)
        _, flow_pred = self.model(x_in)

        if init_flow is not None:
            flow_total = compose_flows(flow_pred, init_flow)
            warped = self.spatial_transform(mov, flow_total)
        else:
            flow_total = flow_pred
            warped = self.spatial_transform(mov, flow_total)

        return warped, flow_total
