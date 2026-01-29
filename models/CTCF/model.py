from typing import Optional, Tuple, List

import torch
import torch.nn as nn

from models.TransMorph_DCA.model import (
    SwinTransformer,
    Conv3dReLU,
    RegistrationHead,
    SpatialTransformer,
)

from models.CTCF.ut_blocks import SRUpBlock3D, CAB, upsample_flow
from models.CTCF.cascade_nets import CoarseFlowNetQuarter
from models.CTCF.refiner import FlowRefiner3D
from models.CTCF.configs import CONFIGS


class CTCF_DCA_CoreHalf(nn.Module):
    """
    Level-2: TM-DCA Swin encoder + SR-style decoder blocks + time integration.
    Operates on HALF-res grid (config.img_size).
    Inputs: [B, 1, D, H, W]
    Outputs: def_x_half [B,1,D,H,W], flow_half [B,3,D,H,W]
    """
    def __init__(self, config, time_steps: int):
        super().__init__()

        self.if_convskip = bool(config.if_convskip)
        self.if_transskip = bool(config.if_transskip)
        self.time_steps = int(time_steps)
        self.img_size = tuple(config.img_size)  # (D,H,W)

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
            dwin_size=config.dwin_size,
        )

        feats = list(self.transformer.num_features)
        assert len(feats) >= 3, "CTCF expects >=3 Swin stages (depths length >= 3)."
        c0, c1, c2 = int(feats[0]), int(feats[1]), int(feats[2])

        self.cab0 = CAB(c2, compress_ratio=3, squeeze_factor=30)
        self.cab1 = CAB(c1, compress_ratio=3, squeeze_factor=30)
        self.cab2 = CAB(c0, compress_ratio=3, squeeze_factor=30)

        self.up0 = SRUpBlock3D(in_channels=c2, out_channels=c1, skip_channels=(c1 if self.if_transskip else 0))
        self.up1 = SRUpBlock3D(in_channels=c1, out_channels=c0, skip_channels=(c0 if self.if_transskip else 0))

        self.avg_pool = nn.AvgPool3d(3, stride=2, padding=1)
        self.c1 = Conv3dReLU(2, max(1, c0 // 2), kernel_size=3, stride=1, use_batchnorm=False)

        self.up2 = SRUpBlock3D(
            in_channels=c0,
            out_channels=max(1, c0 // 2),
            skip_channels=(max(1, c0 // 2) if self.if_convskip else 0),
        )

        self.cs = nn.ModuleList()
        self.up3s = nn.ModuleList()
        self.reg_heads = nn.ModuleList()
        reg_ch = int(config.reg_head_chan)

        self.bridge = nn.Conv3d(
            max(1, c0 // 2),
            reg_ch,
            kernel_size=1,
            bias=True,
        )

        for _ in range(self.time_steps):
            self.cs.append(
                Conv3dReLU(2, reg_ch, kernel_size=3, stride=1, use_batchnorm=False)
            )

            self.up3s.append(
                SRUpBlock3D(
                    in_channels=reg_ch,
                    out_channels=reg_ch,
                    skip_channels=(reg_ch if self.if_convskip else 0),
                )
            )

            self.reg_heads.append(
                RegistrationHead(
                    in_channels=reg_ch,
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
        xx = self.bridge(self.up2(x, f3))

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
        self.st_full = SpatialTransformer(
            tuple(config.img_size[i] * 2 for i in range(3))
        )

        self.use_level1 = bool(config.use_level1)
        self.use_level2 = bool(config.use_level2)
        self.use_level3 = bool(config.use_level3)

        self.st_half = SpatialTransformer(self.img_size_half)

        self.level1 = CoarseFlowNetQuarter(base_ch=config.level1_base_ch) if self.use_level1 else None
        self.level2 = CTCF_DCA_CoreHalf(config, time_steps=config.time_steps) if self.use_level2 else None
        self.level3 = FlowRefiner3D(
            base_ch=config.level3_base_ch,
        ) if self.use_level3 else None

    def forward(
        self,
        mov_full: torch.Tensor,
        fix_full: torch.Tensor,
        *,
        return_all: bool = False,
        alpha_l1: float = 1.0,
        alpha_l3: float = 1.0,
    ):
        mov_half = nn.functional.interpolate(mov_full, scale_factor=0.5, mode="trilinear", align_corners=True)
        fix_half = nn.functional.interpolate(fix_full, scale_factor=0.5, mode="trilinear", align_corners=True)

        flow_half_init = None
        aux: dict = {}

        if self.level1 is not None and alpha_l1 > 0.0:
            mov_quarter = nn.functional.interpolate(mov_full, scale_factor=0.25, mode="trilinear", align_corners=True)
            fix_quarter = nn.functional.interpolate(fix_full, scale_factor=0.25, mode="trilinear", align_corners=True)
            flow_quarter = self.level1(mov_quarter, fix_quarter)
            aux["flow_quarter"] = flow_quarter
            flow_half_init = upsample_flow(flow_quarter, scale_factor=2.0) * float(alpha_l1)

        if self.level2 is None:
            raise RuntimeError("CTCF_CascadeA requires level2 enabled (use_level2=True).")

        out_l2 = self.level2(
            mov_half,
            fix_half,
            init_flow_half=flow_half_init,
            return_all_flows=return_all,
        )

        if return_all:
            def_half_l2, flow_half_l2, flows_l2 = out_l2
            aux["flows_l2"] = flows_l2
        else:
            def_half_l2, flow_half_l2 = out_l2

        if self.level3 is not None and alpha_l3 > 0.0:
            flow_half_ref = self.level3(def_half_l2, fix_half, flow_half_l2) * float(alpha_l3)
            flow_half = flow_half_l2 + flow_half_ref
            aux["flow_half_ref"] = flow_half_ref
        else:
            flow_half = flow_half_l2

        flow_full = upsample_flow(flow_half, scale_factor=2.0) * 2.0
        def_full = self.st_full(mov_full, flow_full)

        aux["mov_half"] = mov_half
        aux["fix_half"] = fix_half

        aux["mov_w_half_final"] = self.st_half(mov_half, flow_half)
        aux["flow_half_final"] = flow_half

        if return_all:
            aux["flow_half_init"] = flow_half_init
            aux["flow_half_l2"] = flow_half_l2
            aux["flow_half"] = flow_half
            aux["flow_full"] = flow_full
            return def_full, flow_full, aux
        return def_full, flow_full