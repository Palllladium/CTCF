from __future__ import annotations

from copy import deepcopy

import torch
from torch import nn

from models.VMambaMorph.configs import CONFIGS as _CFG_REGISTRY
from models.VMambaMorph.model import VMambaMorph
from utils.field import compose_flows


class VMambaMorphSolo(nn.Module):
    """VMambaMorph standalone registration network."""

    def __init__(self, config_key: str = "VMambaMorph", img_size=None):
        super().__init__()
        if config_key not in _CFG_REGISTRY:
            raise KeyError(f"Unknown VMambaMorph config '{config_key}'. Available: {list(_CFG_REGISTRY.keys())}")
        self.cfg = deepcopy(_CFG_REGISTRY[config_key])
        if img_size is not None:
            self.cfg.img_size = tuple(img_size)
        self.cfg_key = config_key
        self.model = VMambaMorph(self.cfg)

    def forward(self, mov, fix):
        out = self.model(mov, fix)
        return out["moved_vol"], out["pos_flow"]

    @property
    def spatial_trans(self):
        return self.model.spatial_trans


class VMambaMorphCascadeL2(nn.Module):
    """Diffeomorphic VMambaMorph as a full-resolution CTCF Level-2 backbone."""

    def __init__(self, config):
        super().__init__()
        config_key = getattr(config, "vmamba_config", "VMambaMorph")
        if config_key not in _CFG_REGISTRY:
            raise KeyError(f"Unknown VMambaMorph config '{config_key}'. Available: {list(_CFG_REGISTRY.keys())}")

        cfg = deepcopy(_CFG_REGISTRY[config_key])
        cfg.img_size = tuple(getattr(config, "img_size", cfg.img_size))
        self.config_key = config_key
        self.model = VMambaMorph(cfg)
        self.spatial_transform = self.model.spatial_trans

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
        out = self.model(mov_warped, fix)
        flow_pred = out["pos_flow"]

        if init_flow is not None:
            flow_total = compose_flows(flow_pred, init_flow)
            warped = self.spatial_transform(mov, flow_total)
        else:
            flow_total = flow_pred
            warped = out["moved_vol"]

        return warped, flow_total
