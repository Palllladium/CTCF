"""
MambaMorph solo wrapper for standalone training.

Exposes a uniform `forward(mov, fix) -> (warped, flow_voxel)` interface
matching the rest of this repo. The diffeomorphic variant (`MambaMorph`,
with VecInt integration of the predicted stationary velocity field) is the
default; the non-diffeomorphic variant (`MambaMorphOri`, direct displacement)
is available via `--diffeo 0`.
"""
import torch.nn as nn

from models.MambaMorph.configs import CONFIGS as _CFG_REGISTRY
from models.MambaMorph.model import MambaMorph as _MambaMorph
from models.MambaMorph.model import MambaMorphOri as _MambaMorphOri


class MambaMorphSolo(nn.Module):
    """MambaMorph standalone registration network."""
    def __init__(self, config_key: str = "MambaMorph", diffeomorphic: bool = True):
        super().__init__()
        if config_key not in _CFG_REGISTRY:
            raise KeyError(f"Unknown MambaMorph config '{config_key}'. "
                           f"Available: {list(_CFG_REGISTRY.keys())}")
        self.cfg = _CFG_REGISTRY[config_key]
        self.cfg_key = config_key
        self.diffeomorphic = bool(diffeomorphic)
        cls = _MambaMorph if self.diffeomorphic else _MambaMorphOri
        self.model = cls(self.cfg)

    def forward(self, mov, fix):
        out = self.model(mov, fix)
        warped = out["moved_vol"]
        # `pos_flow` is the integrated displacement field (diffeo variant);
        # `preint_flow` is the raw network output (used for regularization).
        flow = out["pos_flow"] if self.diffeomorphic else out["preint_flow"]
        return warped, flow

    @property
    def spatial_trans(self):
        return self.model.spatial_trans

    @property
    def preint_flow_fn(self):
        """Helper to access the pre-integration velocity for diffeomorphic
        regularization (smoothness on the velocity field, not the displacement)."""
        return lambda mov, fix: self.model(mov, fix)["preint_flow"]
