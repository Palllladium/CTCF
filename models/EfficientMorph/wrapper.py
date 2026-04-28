"""EfficientMorph wrapper with forward(mov, fix) -> (warped, voxel_flow)."""
from copy import deepcopy

import torch
import torch.nn as nn

from models.EfficientMorph.configs import CONFIGS as _CFG_REGISTRY
from models.EfficientMorph.model import EfficientMorph as _EfficientMorphCore


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
