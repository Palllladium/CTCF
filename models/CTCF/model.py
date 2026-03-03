from typing import Tuple

import torch
import torch.nn as nn

from models.TransMorph_DCA.model import SpatialTransformer
from models.CTCF.blocks import upsample_flow
from models.CTCF.stages import CTCF_DCA_CoreHalf, CoarseFlowNetQuarter, FlowRefiner3D
from models.CTCF.configs import CONFIGS


class CTCF_CascadeA(nn.Module):
    """
    Variant A:
      L1: CoarseFlowNetQuarter (1/4)
      L2: CTCF_DCA_CoreHalf    (1/2) with init_flow from L1
      L3: FlowRefiner3D        (1/2) with residual refinement
    Output is produced on FULL-res by upsampling final half-res flow by x2.
    """
    def __init__(self, config):
        super().__init__()

        self.img_size_full: Tuple[int, int, int] = tuple(config.img_size)
        self.use_level1 = bool(config.use_level1)
        self.use_level2 = bool(config.use_level2)
        self.use_level3 = bool(config.use_level3)

        self.level1 = CoarseFlowNetQuarter(base_ch=config.level1_base_ch) if self.use_level1 else None
        self.level2 = CTCF_DCA_CoreHalf(config, time_steps=config.time_steps) if self.use_level2 else None
        self.level3 = FlowRefiner3D(base_ch=config.level3_base_ch) if self.use_level3 else None

        self.st_full = SpatialTransformer(self.img_size_full)

    def forward(
        self,
        mov_full: torch.Tensor,
        fix_full: torch.Tensor,
        *,
        return_all: bool = False,
        alpha_l1: float = 1.0,
    ):
        mov_half = nn.functional.interpolate(mov_full, scale_factor=0.5, mode="trilinear", align_corners=False)
        fix_half = nn.functional.interpolate(fix_full, scale_factor=0.5, mode="trilinear", align_corners=False)

        aux = {} if return_all else None
        flow_half_init = None

        if self.level1 is not None and alpha_l1 > 0.0:
            mov_quarter = nn.functional.interpolate(mov_full, scale_factor=0.25, mode="trilinear", align_corners=False)
            fix_quarter = nn.functional.interpolate(fix_full, scale_factor=0.25, mode="trilinear", align_corners=False)
            flow_quarter = self.level1(mov_quarter, fix_quarter)
            flow_half_init = upsample_flow(flow_quarter, scale_factor=2) * float(alpha_l1)
            if aux is not None:
                aux["flow_quarter"] = flow_quarter
                aux["flow_half_init"] = flow_half_init

        if self.level2 is None:
            raise RuntimeError("CTCF_CascadeA requires level2 enabled (use_level2=True).")

        def_half_l2, flow_half_l2 = self.level2(mov_half, fix_half, init_flow_half=flow_half_init, return_all_flows=False)
        if aux is not None:
            aux["def_half_l2"] = def_half_l2
            aux["flow_half_l2"] = flow_half_l2

        if self.level3 is not None:
            flow_half_ref = self.level3(def_half_l2, fix_half, flow_half_l2)
            flow_half = flow_half_l2 + flow_half_ref
            if aux is not None:
                aux["flow_half_ref"] = flow_half_ref
        else:
            flow_half = flow_half_l2

        flow_full = upsample_flow(flow_half, scale_factor=2)
        def_full = self.st_full(mov_full, flow_full)

        if aux is not None:
            aux["flow_half_final"] = flow_half
            aux["flow_full"] = flow_full

        if return_all:
            return def_full, flow_full, aux
        return def_full, flow_full
