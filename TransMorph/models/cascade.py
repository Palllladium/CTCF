
import torch
import torch.nn as nn
from TransMorph.models.utils_field import warp, compose_flows

class CascadeCTCF(nn.Module):
    """
    Generic 2-level cascade wrapper around a base registration model.
    The base model must implement forward(F, M) -> (warped_M, flow_vox)
    where flow_vox is [B,3,D,H,W] in voxel units (z,y,x components).
    """
    def __init__(self, base_model_fn, base_cfg, levels=2):
        super().__init__()
        assert levels >= 1
        self.levels = levels
        self.stages = nn.ModuleList([base_model_fn(base_cfg) for _ in range(levels)])

    def forward(self, F_img, M_img):
        flow_total = None
        M_warp = M_img
        for i, stage in enumerate(self.stages):
            out, flow = stage(F_img, M_warp)
            if flow_total is None:
                flow_total = flow
            else:
                flow_total = compose_flows(flow_total, flow)  # phi = phi + warp(flow, phi)
            M_warp = warp(M_img, flow_total)  # always warp original M by total flow
        return M_warp, flow_total
