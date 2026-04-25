"""
EfficientMorph solo wrapper for standalone training.

EfficientMorph (Bin Aziz et al., WACV 2025) accepts a 2-channel concatenated
input (mov, fix) and returns (warped, flow). This wrapper exposes the more
conventional forward(mov, fix) signature used elsewhere in this repo.

Original source: https://github.com/MedVICLab/Efficient_Morph_Registration

Note: model spatial transform produces flow in VOXEL units (not normalized);
no scale conversion is required for downstream metrics, unlike LKU-Net.
"""
import torch
import torch.nn as nn

from models.EfficientMorph.model import EfficientMorph as _EfficientMorphCore
from models.EfficientMorph.model import CONFIGS as _MODEL_CONFIGS


class EfficientMorphSolo(nn.Module):
    """EfficientMorph standalone registration network."""
    def __init__(self, config_key: str):
        super().__init__()
        if config_key not in _MODEL_CONFIGS:
            raise KeyError(f"Unknown EfficientMorph config '{config_key}'. "
                           f"Available: {list(_MODEL_CONFIGS.keys())}")
        self.cfg = _MODEL_CONFIGS[config_key]
        self.cfg_key = config_key
        self.model = _EfficientMorphCore(self.cfg)

    def forward(self, mov, fix):
        x_in = torch.cat([mov, fix], dim=1)
        warped, flow = self.model(x_in)
        return warped, flow

    @property
    def spatial_trans(self):
        """Expose internal SpatialTransformer for re-warping segmentations etc."""
        return self.model.spatial_trans
