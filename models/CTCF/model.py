# models/CTCF/model.py

from __future__ import annotations

from typing import Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.TransMorph_DCA.model import (
    SwinTransformer,
    Conv3dReLU,
    RegistrationHead,
    SpatialTransformer,
)

import models.CTCF.configs as configs
from models.CTCF.ut_blocks import SRUpBlock3D, CAB, upsample_flow
from models.CTCF.cascade_nets import CoarseFlowNetQuarter, FlowRefiner3D


class CTCF_DCA_CoreHalf(nn.Module):
    """
    Level-2: TM-DCA Swin encoder + SR-style decoder blocks + time integration.
    Operates on HALF-res grid (config.img_size).
    Expected tensors: [B, 1, D, H, W]
    """
    def __init__(self, config, time_steps: int):
        super().__init__()

        # ---- contract fields (NO getattr) ----
        self.if_convskip = bool(config.if_convskip)
        self.if_transskip = bool(config.if_transskip)

        self.time_steps = int(time_steps)
        self.img_size = tuple(config.img_size)  # (D,H,W)

        embed_dim = int(config.embed_dim)

        self.transformer = SwinTransformer(
            patch_size=config.patch_size,
            in_chans=config.in_chans,
            embed_dim=config.embed_dim,
            depths=config.depths,
            num_heads=config.num_heads,
            window_size=config.window_size,
            mlp_ratio=config.mlp_ratio,
            qkv_bias=config.qkv_bias,
            drop_rate=config.drop_rate,
            drop_path_rate=config.drop_path_rate,
            ape=config.ape,
            spe=config.spe,
            rpe=config.rpe,
            patch_norm=config.patch_norm,
            use_checkpoint=config.use_checkpoint,
            out_indices=config.out_indices,
            pat_merg_rf=config.pat_merg_rf,
            img_size=config.img_size,
            dwin_size=config.dwin_kernel_size,
        )

        self.use_cab = True
        self.cab0 = CAB(embed_dim * 4, compress_ratio=3, squeeze_factor=30)
        self.cab1 = CAB(embed_dim * 2, compress_ratio=3, squeeze_factor=30)
        self.cab2 = CAB(embed_dim, compress_ratio=3, squeeze_factor=30)

        self.up0 = SRUpBlock3D(
            in_channels=embed_dim * 4,
            out_channels=embed_dim * 2,
            skip_channels=(embed_dim * 2 if self.if_transskip else 0),
        )
        self.up1 = SRUpBlock3D(
            in_channels=embed_dim * 2,
            out_channels=embed_dim,
            skip_channels=(embed_dim if self.if_transskip else 0),
        )
        self.up2 = SRUpBlock3D(
            in_channels=embed_dim,
            out_channels=embed_dim // 2,
            skip_channels=(embed_dim // 2 if self.if_transskip else 0),
        )

        self.avg_pool = nn.AvgPool3d(3, stride=2, padding=1)
        self.c1 = Conv3dReLU(2, embed_dim // 2, kernel_size=3, stride=1, use_batchnorm=False)

        self.cs = nn.ModuleList()
        self.up3s = nn.ModuleList()
        self.reg_heads = nn.ModuleList()

        for _ in range(self.time_steps):
            self.cs.append(Conv3dReLU(2, embed_dim // 2, kernel_size=3, stride=1, use_batchnorm=False))
            self.up3s.append(
                SRUpBlock3D(
                    in_channels=embed_dim // 2,
                    out_channels=embed_dim // 2,
                    skip_channels=(embed_dim // 2 if self.if_convskip else 0),
                )
            )
            self.reg_heads.append(
                RegistrationHead(
                    in_channels=embed_dim // 2,
                    out_channels=3,
                    kernel_size=3,
                )
            )

        self.spatial_trans = SpatialTransformer(self.img_size)

    def forward(
        self,
        mov_half: torch.Tensor,
        fix_half: torch.Tensor,
        *,
        init_flow_half: Optional[torch.Tensor] = None,
        return_all_flows: bool = False,
    ):
        """
        Returns:
          def_x_half, flow_half
          optionally also per-step flows list
        """
        if init_flow_half is None:
            flow_prev = torch.zeros(
                (mov_half.shape[0], 3, *self.img_size),
                device=mov_half.device,
                dtype=mov_half.dtype,
            )
            def_x = mov_half
        else:
            flow_prev = init_flow_half
            def_x = self.spatial_trans(mov_half, flow_prev)

        f3 = self.c1(torch.cat((mov_half, fix_half), dim=1))
        f3 = self.avg_pool(f3).to(mov_half.dtype) if self.if_convskip else None

        out_feats = self.transformer((mov_half, fix_half))

        if self.if_transskip:
            mov_f1, fix_f1 = out_feats[-2]
            f1 = self.cab1(mov_f1 + fix_f1)

            mov_f2, fix_f2 = out_feats[-3]
            f2 = self.cab2(mov_f2 + fix_f2)
        else:
            f1 = None
            f2 = None

        mov_f0, fix_f0 = out_feats[-1]
        f0 = self.cab0(mov_f0 + fix_f0)

        x = self.up0(f0, f1)
        x = self.up1(x, f2)
        xx = self.up2(x, f3)

        flows: List[torch.Tensor] = []

        for t in range(self.time_steps):
            f_out = self.cs[t](torch.cat((def_x, fix_half), dim=1))
            x_t = self.up3s[t](xx, f_out if self.if_convskip else None)
            flow_step = self.reg_heads[t](x_t)
            flows.append(flow_step)

            flow_new = flow_prev + self.spatial_trans(flow_step, flow_prev)
            def_x = self.spatial_trans(mov_half, flow_new)
            flow_prev = flow_new

        if return_all_flows:
            return def_x, flow_prev, flows
        return def_x, flow_prev


class CTCF_CascadeA(nn.Module):
    """
    Variant A:
      L1: CoarseFlowNetQuarter (1/4)
      L2: CTCF_DCA_CoreHalf    (1/2) with init_flow from L1
      L3: FlowRefiner3D        (1/2) with error-map
    Output is produced on FULL-res by upsampling final half-res flow by x2 and warping full mov.
    """
    def __init__(self, config):
        super().__init__()
        self.cfg = config

        self.img_size_half: Tuple[int, int, int] = tuple(config.img_size)
        self.use_level1 = bool(config.use_level1)
        self.use_level2 = bool(config.use_level2)
        self.use_level3 = bool(config.use_level3)

        self.st_half = SpatialTransformer(self.img_size_half)

        if self.use_level1:
            self.level1 = CoarseFlowNetQuarter(base_ch=config.level1_base_ch)
        else:
            self.level1 = None

        if self.use_level2:
            self.level2 = CTCF_DCA_CoreHalf(config, time_steps=config.time_steps)
        else:
            self.level2 = None

        if self.use_level3:
            self.level3 = FlowRefiner3D(
                base_ch=config.level3_base_ch,
                error_mode=config.level3_error_mode,
            )
        else:
            self.level3 = None

    def forward(
        self,
        mov: torch.Tensor,
        fix: torch.Tensor,
        *,
        return_all: bool = False,
    ):
        """
        Inputs: mov, fix on FULL-res [B,1,2D,2H,2W]
        Internals:
          - half: avg_pool3d(2)
          - quarter: avg_pool3d(4)
        Returns:
          def_full, flow_full
          if return_all: also dict with intermediate flows/maps
        """
        mov_half = F.avg_pool3d(mov, kernel_size=2, stride=2)
        fix_half = F.avg_pool3d(fix, kernel_size=2, stride=2)

        flow_half = torch.zeros(
            (mov.shape[0], 3, *self.img_size_half),
            device=mov.device,
            dtype=mov.dtype,
        )

        aux = {}

        if self.use_level1:
            mov_q = F.avg_pool3d(mov, kernel_size=4, stride=4)
            fix_q = F.avg_pool3d(fix, kernel_size=4, stride=4)

            flow_q = self.level1(mov_q, fix_q)              # (B,3,D/4,H/4,W/4)
            flow_half_l1 = upsample_flow(flow_q, scale=2)   # -> half with correct scaling

            flow_half = flow_half + flow_half_l1
            aux["flow_q"] = flow_q
            aux["flow_half_l1"] = flow_half_l1

        mov_w_half = self.st_half(mov_half, flow_half)

        if self.use_level2:
            def_half_l2, flow_half_l2 = self.level2(
                mov_half,
                fix_half,
                init_flow_half=flow_half,
                return_all_flows=False,
            )
            flow_half = flow_half_l2
            mov_w_half = def_half_l2
            aux["flow_half_l2"] = flow_half_l2

        if self.use_level3:
            delta_half = self.level3(mov_w_half, fix_half, flow_half)
            flow_half = flow_half + delta_half
            mov_w_half = self.st_half(mov_half, flow_half)
            aux["delta_flow_half_l3"] = delta_half
            aux["flow_half_l3"] = flow_half

        flow_full = upsample_flow(flow_half, scale=2)
        full_size = tuple(mov.shape[2:])
        st_full = SpatialTransformer(full_size).to(mov.device)
        def_full = st_full(mov, flow_full)

        if return_all:
            aux["flow_half_final"] = flow_half
            aux["mov_half"] = mov_half
            aux["fix_half"] = fix_half
            aux["mov_w_half_final"] = mov_w_half
            return def_full, flow_full, aux

        return def_full, flow_full


CONFIGS = {
    "CTCF-CascadeA": configs.get_CTCF_config(),
    "CTCF-CascadeA-Debug": configs.get_CTCF_debug_config(),
}