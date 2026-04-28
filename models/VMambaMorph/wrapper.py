"""VMambaMorph wrapper with forward(mov, fix) -> (warped, integrated_flow)."""
from copy import deepcopy

import torch.nn as nn

from models.VMambaMorph.configs import CONFIGS as _CFG_REGISTRY
from models.VMambaMorph.model import VMambaMorph


class VMambaMorphSolo(nn.Module):
    def __init__(self, config_key: str = "VMambaMorph", img_size=None):
        super().__init__()
        if config_key not in _CFG_REGISTRY:
            raise KeyError(f"Unknown VMambaMorph config '{config_key}'. "
                           f"Available: {list(_CFG_REGISTRY.keys())}")
        self.cfg = deepcopy(_CFG_REGISTRY[config_key])
        if img_size is not None:
            self.cfg.img_size = tuple(int(v) for v in img_size)
        self.cfg_key = config_key
        self.model = VMambaMorph(self.cfg)


    def forward(self, mov, fix):
        out = self.model(mov, fix)
        return out["moved_vol"], out["pos_flow"]


    @property
    def spatial_trans(self):
        return self.model.spatial_trans
