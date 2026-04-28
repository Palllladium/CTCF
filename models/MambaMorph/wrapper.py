"""MambaMorph wrapper with forward(mov, fix) -> (warped, flow)."""
from copy import deepcopy

import torch.nn as nn

from models.MambaMorph.configs import CONFIGS as _CFG_REGISTRY
from models.MambaMorph.model import MambaMorph, MambaMorphOri


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
