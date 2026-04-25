"""
VMambaMorph solo wrapper for standalone training.

Mirrors the MambaMorphSolo pattern. Always diffeomorphic (the upstream
VMambaMorph class is the integrated SVF variant; no `Ori` companion is
provided by the authors).
"""
import torch.nn as nn

from models.VMambaMorph.configs import CONFIGS as _CFG_REGISTRY
from models.VMambaMorph.model import VMambaMorph as _VMambaMorph


class VMambaMorphSolo(nn.Module):
    def __init__(self, config_key: str = "VMambaMorph"):
        super().__init__()
        if config_key not in _CFG_REGISTRY:
            raise KeyError(f"Unknown VMambaMorph config '{config_key}'. "
                           f"Available: {list(_CFG_REGISTRY.keys())}")
        self.cfg = _CFG_REGISTRY[config_key]
        self.cfg_key = config_key
        self.model = _VMambaMorph(self.cfg)

    def forward(self, mov, fix):
        out = self.model(mov, fix)
        return out["moved_vol"], out["pos_flow"]

    @property
    def spatial_trans(self):
        return self.model.spatial_trans
