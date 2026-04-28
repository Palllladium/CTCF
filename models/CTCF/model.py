from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.TransMorph_DCA.model import SpatialTransformer
from models.CTCF.blocks import upsample_flow
from models.CTCF.stages import CTCF_DCA_CoreHalf, CoarseFlowNetQuarter, FlowRefiner3D
from models.CTCF.configs import CONFIGS
from utils.field import compose_flows


def _build_level2(config, backbone: str, l2_full_res: bool = False):
    """Build L2 backbone module based on config.backbone."""
    if backbone == "swin-dca":
        return CTCF_DCA_CoreHalf(config, time_steps=config.time_steps)
    elif backbone == "vxm":
        from models.VoxelMorph.wrapper import VxmDenseHalf
        vxm = getattr(config, "vxm", None)
        vol = tuple(config.img_size) if l2_full_res else tuple(s // 2 for s in config.img_size)
        return VxmDenseHalf(
            vol_size=vol,
            enc_nf=list(vxm.enc_nf) if vxm else [16, 32, 32, 32],
            dec_nf=list(vxm.dec_nf) if vxm else [32, 32, 32, 32, 32, 16, 16],
            int_steps=int(vxm.int_steps) if vxm else 7,
        )
    else:
        raise ValueError(f"Unknown backbone: {backbone}")


class CTCF_CascadeA(nn.Module):
    """
    Variant A:
      L1: CoarseFlowNetQuarter (1/4 or 1/2)
      L2: CTCF_DCA_CoreHalf    (1/2) with init_flow from L1
      L3: FlowRefiner3D        (1/2 or full) with residual refinement
    Output is produced on FULL-res by upsampling final half-res flow by x2.
    """
    def __init__(self, config):
        super().__init__()

        self.img_size_full: Tuple[int, int, int] = tuple(config.img_size)
        self.use_level1 = bool(config.use_level1)
        self.use_level2 = bool(config.use_level2)
        self.use_level3 = bool(config.use_level3)
        self.backbone = str(getattr(config, 'backbone', 'swin-dca'))

        self.l3_iters = int(getattr(config, 'l3_iters', 1))
        self.l3_svf = bool(getattr(config, 'l3_svf', False))
        self.l1_half_res = bool(getattr(config, 'l1_half_res', False))
        self.l2_full_res = bool(getattr(config, 'l2_full_res', False))
        self.l3_full_res = bool(getattr(config, 'l3_full_res', False))
        self.l3_unshared = bool(getattr(config, 'l3_unshared', False)) and self.l3_iters > 1

        l3_kwargs = dict(
            base_ch=config.level3_base_ch,
            error_mode=config.level3_error_mode,
        )

        self.level1 = CoarseFlowNetQuarter(base_ch=config.level1_base_ch) if self.use_level1 else None
        self.level2 = _build_level2(config, self.backbone, self.l2_full_res) if self.use_level2 else None
        self.level3 = FlowRefiner3D(**l3_kwargs) if self.use_level3 else None

        # Unshared L3: separate weights for iterations 1..n-1
        if self.l3_unshared and self.use_level3:
            self.level3_extra = nn.ModuleList([
                FlowRefiner3D(**l3_kwargs) for _ in range(self.l3_iters - 1)
            ])
        else:
            self.level3_extra = None

        self.st_full = SpatialTransformer(self.img_size_full)
        img_size_half = tuple(s // 2 for s in self.img_size_full)
        self.st_half = SpatialTransformer(img_size_half)


    def _get_l3(self, it: int) -> FlowRefiner3D:
        """Return L3 module for iteration `it` (unshared: separate weights per iter)."""
        if it == 0 or self.level3_extra is None:
            return self.level3
        return self.level3_extra[it - 1]


    def _upsample_to_full(self, flow_half: torch.Tensor) -> torch.Tensor:
        """Upsample half-res flow to full-res via trilinear + magnitude scaling."""
        return upsample_flow(flow_half, scale_factor=2)


    def _integrate_svf(self, vel: torch.Tensor, st: SpatialTransformer, steps: int = 7) -> torch.Tensor:
        """Integrate stationary velocity field via scaling and squaring."""
        disp = vel * (1.0 / (2 ** steps))
        for _ in range(steps):
            disp = disp + st(disp, disp)
        return disp


    def _apply_l3_delta(self, delta: torch.Tensor, flow_cur: torch.Tensor,
                        st: SpatialTransformer) -> torch.Tensor:
        """Apply L3 delta to current flow (SVF integration + composition, or simple add)."""
        if self.l3_svf:
            delta = self._integrate_svf(delta, st)
            return compose_flows(delta, flow_cur)
        return flow_cur + delta


    def forward(
        self,
        mov_full: torch.Tensor,
        fix_full: torch.Tensor,
        *,
        alpha_l1: float = 1.0,
        alpha_l3: float = 1.0,
    ):
        mov_half = F.interpolate(mov_full, scale_factor=0.5, mode="trilinear", align_corners=False)
        fix_half = F.interpolate(fix_full, scale_factor=0.5, mode="trilinear", align_corners=False)

        flow_half_init = None

        # Level 1
        if self.level1 is not None and alpha_l1 > 0.0:
            if self.l1_half_res:
                flow_half_l1 = self.level1(mov_half, fix_half)
                flow_half_init = flow_half_l1 * float(alpha_l1)
            else:
                mov_quarter = F.interpolate(mov_full, scale_factor=0.25, mode="trilinear", align_corners=False)
                fix_quarter = F.interpolate(fix_full, scale_factor=0.25, mode="trilinear", align_corners=False)
                flow_quarter = self.level1(mov_quarter, fix_quarter)
                flow_half_init = upsample_flow(flow_quarter, scale_factor=2) * float(alpha_l1)

        if self.l2_full_res:
            flow_l2_init = self._upsample_to_full(flow_half_init) if flow_half_init is not None else None
            l2_mov, l2_fix = mov_full, fix_full
        else:
            flow_l2_init = flow_half_init
            l2_mov, l2_fix = mov_half, fix_half

        # Level 2
        if self.level2 is None:
            raise RuntimeError("CTCF_CascadeA requires level2 enabled (use_level2=True).")

        def_half_l2, flow_half_l2 = self.level2(
            l2_mov, l2_fix,
            init_flow_half=flow_l2_init,
            return_all_flows=False,
        )

        # Level 3
        flow_half = flow_half_l2
        if self.level3 is not None and alpha_l3 > 0.0:
            if self.l3_full_res or self.l2_full_res:
                if self.l2_full_res:
                    flow_full_cur = flow_half_l2   # already full-res
                    def_full_cur = def_half_l2
                else:
                    flow_full_cur = self._upsample_to_full(flow_half_l2)
                    def_full_cur = self.st_full(mov_full, flow_full_cur)

                for it in range(self.l3_iters):
                    l3_mod = self._get_l3(it)
                    delta_full = l3_mod(def_full_cur, fix_full, flow_full_cur)
                    delta_full = delta_full * float(alpha_l3)
                    flow_full_cur = self._apply_l3_delta(delta_full, flow_full_cur, self.st_full)
                    if it < self.l3_iters - 1:
                        def_full_cur = self.st_full(mov_full, flow_full_cur)

                flow_full = flow_full_cur
                def_full = self.st_full(mov_full, flow_full)
                return def_full, flow_full
            else:
                def_cur = def_half_l2
                flow_cur = flow_half_l2

                for it in range(self.l3_iters):
                    l3_mod = self._get_l3(it)
                    delta = l3_mod(def_cur, fix_half, flow_cur)
                    delta = delta * float(alpha_l3)
                    flow_cur = self._apply_l3_delta(delta, flow_cur, self.st_half)
                    if it < self.l3_iters - 1:
                        def_cur = self.st_half(mov_half, flow_cur)

                flow_half = flow_cur

        if self.l2_full_res: flow_full = flow_half
        else: flow_full = self._upsample_to_full(flow_half)
        def_full = self.st_full(mov_full, flow_full)

        return def_full, flow_full
