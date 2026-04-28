"""MambaMorph wrappers for standalone and CTCF-cascade use."""
from copy import deepcopy

import torch
import torch.nn as nn

from models.MambaMorph.configs import CONFIGS as _CFG_REGISTRY
from models.MambaMorph.model import MambaMorph, MambaMorphOri
from utils.field import compose_flows


class MambaMorphSolo(nn.Module):
    """MambaMorph standalone registration network."""
    def __init__(self, config_key: str = "MambaMorph", diffeomorphic: bool = True, img_size=None):
        super().__init__()
        if config_key not in _CFG_REGISTRY:
            raise KeyError(f"Unknown MambaMorph config '{config_key}'. "
                           f"Available: {list(_CFG_REGISTRY.keys())}")
        self.cfg = deepcopy(_CFG_REGISTRY[config_key])
        if img_size is not None:
            self.cfg.img_size = tuple(int(v) for v in img_size)
        self.cfg_key = config_key
        self.diffeomorphic = bool(diffeomorphic)
        cls = MambaMorph if self.diffeomorphic else MambaMorphOri
        self.model = cls(self.cfg)


    def forward(self, mov, fix):
        out = self.model(mov, fix)
        warped = out["moved_vol"]
        flow = out["pos_flow"] if self.diffeomorphic else out["preint_flow"]
        return warped, flow


    @property
    def spatial_trans(self):
        return self.model.spatial_trans


    @property
    def preint_flow_fn(self):
        """Return the pre-integration field used by SVF regularization."""
        return lambda mov, fix: self.model(mov, fix)["preint_flow"]


class MambaMorphCascadeL2(nn.Module):
    """Diffeomorphic MambaMorph as a full-resolution CTCF Level-2 backbone."""

    def __init__(self, config):
        super().__init__()
        config_key = str(getattr(config, "mamba_config", "MambaMorph"))
        if config_key not in _CFG_REGISTRY:
            raise KeyError(f"Unknown MambaMorph config '{config_key}'. "
                           f"Available: {list(_CFG_REGISTRY.keys())}")

        cfg = deepcopy(_CFG_REGISTRY[config_key])
        cfg.img_size = tuple(int(v) for v in getattr(config, "img_size", cfg.img_size))
        self.config_key = config_key
        self.model = MambaMorph(cfg)
        self.spatial_transform = self.model.spatial_trans


    def forward(
        self,
        mov: torch.Tensor,
        fix: torch.Tensor,
        init_flow=None,
        return_all_flows: bool = False,
    ):
        mov_warped = self.spatial_transform(mov, init_flow) if init_flow is not None else mov
        out = self.model(mov_warped, fix)
        flow_pred = out["pos_flow"]

        if init_flow is not None:
            flow_total = compose_flows(flow_pred, init_flow)
            warped = self.spatial_transform(mov, flow_total)
        else:
            flow_total = flow_pred
            warped = out["moved_vol"]

        return warped, flow_total
