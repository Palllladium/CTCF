from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.TransMorph_DCA.model import SwinTransformer, Conv3dReLU, RegistrationHead, SpatialTransformer
from models.CTCF.blocks import SRUpBlock3D, CAB, ResidualContext3D


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.InstanceNorm3d(out_ch, affine=True),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv3d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.InstanceNorm3d(out_ch, affine=True),
            nn.LeakyReLU(0.1, inplace=True),
        )


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CoarseFlowNetQuarter(nn.Module):
    """
    Level-1: quarter-res coarse flow predictor (conv-only).
    Input: mov_q, fix_q -> concat (B,2,D,H,W)
    Output: flow_q (B,3,D,H,W)
    """
    def __init__(self, base_ch: int = 16, use_cab: bool = False):
        super().__init__()
        c = int(base_ch)
        self.enc1 = ConvBlock(2, c)
        self.pool1 = nn.AvgPool3d(2)
        self.enc2 = ConvBlock(c, c * 2)
        self.pool2 = nn.AvgPool3d(2)
        self.bot = ConvBlock(c * 2, c * 4)
        self.ctx1 = ResidualContext3D(c * 4, dilation=1, scale=0.1)
        self.ctx2 = ResidualContext3D(c * 4, dilation=2, scale=0.1)
        self.up2 = nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False)
        self.dec2 = ConvBlock(c * 4 + c * 2, c * 2)
        self.up1 = nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False)
        self.dec1 = ConvBlock(c * 2 + c, c)
        self.cab = CAB(c, compress_ratio=3, squeeze_factor=30) if use_cab else None
        self.out = nn.Conv3d(c, 3, kernel_size=3, padding=1, bias=True)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)


    def forward(self, mov: torch.Tensor, fix: torch.Tensor, *, return_features: bool = False):
        x = torch.cat([mov, fix], dim=1)
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        b = self.bot(self.pool2(e2))
        b = self.ctx1(b)
        b = self.ctx2(b)
        d2 = self.up2(b)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))
        if self.cab is not None:
            d1 = d1 + self.cab(d1)
        flow = self.out(d1)
        if return_features:
            return flow, d1  # d1: (B, base_ch, D, H, W) at L1 output resolution
        return flow


class CTCF_DCA_CoreHalf(nn.Module):
    """
    Level-2: TM-DCA Swin encoder + SR-style decoder blocks + time integration.
    Operates on HALF-res grid derived from config.img_size.
    """
    def __init__(self, config, time_steps: int):
        super().__init__()
        self.if_convskip = bool(config.if_convskip)
        self.if_transskip = bool(config.if_transskip)
        self.prealign_encoder = bool(getattr(config, 'prealign_encoder', False))
        self.l1_l2_skip = bool(getattr(config, 'l1_l2_skip', False))
        self.time_steps = int(time_steps)
        self.img_size_full = tuple(config.img_size)
        self.img_size = tuple(s // 2 for s in self.img_size_full)

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
            img_size=self.img_size,
            dwin_size=config.dwin_size,
        )

        c0, c1, c2 = (int(v) for v in self.transformer.num_features[:3])
        self.c_mid = max(1, c0 // 2)
        self.cab0 = CAB(c2, compress_ratio=3, squeeze_factor=30)
        self.cab1 = CAB(c1, compress_ratio=3, squeeze_factor=30)
        self.cab2 = CAB(c0, compress_ratio=3, squeeze_factor=30)
        self.up0 = SRUpBlock3D(in_channels=c2, out_channels=c1, skip_channels=(c1 if self.if_transskip else 0))
        self.up1 = SRUpBlock3D(in_channels=c1, out_channels=c0, skip_channels=(c0 if self.if_transskip else 0))
        self.avg_pool = nn.AvgPool3d(3, stride=2, padding=1)

        # Conv-skip path: 2ch (mov+fix) + optional L1 features
        l1_skip_ch = int(getattr(config, 'level1_base_ch', 32)) if self.l1_l2_skip else 0
        self.c1 = Conv3dReLU(2 + l1_skip_ch, self.c_mid, kernel_size=3, stride=1, use_batchnorm=False)
        self.up2 = SRUpBlock3D(in_channels=c0, out_channels=self.c_mid, skip_channels=(self.c_mid if self.if_convskip else 0))

        reg_ch = int(config.reg_head_chan)
        self.cs = nn.ModuleList()
        self.up3s = nn.ModuleList()
        self.reg_heads = nn.ModuleList()
        for _ in range(self.time_steps):
            self.cs.append(Conv3dReLU(2, self.c_mid, kernel_size=3, stride=1, use_batchnorm=False))
            self.up3s.append(SRUpBlock3D(in_channels=self.c_mid, out_channels=reg_ch, skip_channels=(self.c_mid if self.if_convskip else 0)))
            self.reg_heads.append(RegistrationHead(in_channels=reg_ch, out_channels=3, kernel_size=3))
        self.spatial_trans = SpatialTransformer(self.img_size)


    def forward(
        self,
        mov_half: torch.Tensor,
        fix_half: torch.Tensor,
        init_flow_half: Optional[torch.Tensor] = None,
        return_all_flows: bool = False,
        l1_feat: Optional[torch.Tensor] = None,
        return_features: bool = False,
    ):
        if init_flow_half is None:
            flow_prev = torch.zeros((mov_half.shape[0], 3, *self.img_size), device=mov_half.device, dtype=mov_half.dtype)
            def_x = mov_half
        else:
            flow_prev = init_flow_half
            def_x = self.spatial_trans(mov_half, flow_prev)

        enc_mov = def_x if self.prealign_encoder and init_flow_half is not None else mov_half
        x_cat = torch.cat((enc_mov, fix_half), dim=1)

        if self.if_convskip:
            pooled = self.avg_pool(x_cat)
            if self.l1_l2_skip and l1_feat is not None:
                # L1 features are at L1 output res; resize to match pooled (quarter of half = 1/4 full)
                l1_feat_resized = F.interpolate(l1_feat, size=pooled.shape[2:], mode="trilinear", align_corners=False)
                pooled = torch.cat([pooled, l1_feat_resized], dim=1)
            f3 = self.c1(pooled).to(mov_half.dtype)
        else:
            f3 = None

        out_feats = self.transformer((enc_mov, fix_half))

        if self.if_transskip:
            mov_f1, fix_f1 = out_feats[-2]
            mov_f2, fix_f2 = out_feats[-3]
            f1 = self.cab1(mov_f1 + fix_f1)
            f2 = self.cab2(mov_f2 + fix_f2)
        else:
            f1 = None
            f2 = None

        mov_f0, fix_f0 = out_feats[-1]
        x = self.up0(self.cab0(mov_f0 + fix_f0), f1)
        x = self.up1(x, f2)
        xx = self.up2(x, f3)

        flows = [] if return_all_flows else None
        for t in range(self.time_steps):
            f_out = self.cs[t](torch.cat((def_x, fix_half), dim=1))
            x_t = self.up3s[t](xx, f_out if self.if_convskip else None)
            flow_step = self.reg_heads[t](x_t)
            if flows is not None:
                flows.append(flow_step)
            flow_prev = flow_prev + self.spatial_trans(flow_step, flow_prev)
            def_x = self.spatial_trans(mov_half, flow_prev)

        out = (def_x, flow_prev)
        if return_all_flows:
            out = (def_x, flow_prev, flows)
        if return_features:
            # xx: (B, c_mid, D_half, H_half, W_half) — decoder features for L2→L3 skip
            return out + (xx,)
        return out


class FlowRefiner3D(nn.Module):
    """
    Level-3: refinement using error-map.
    Inputs:
      - mov_warp: (B,1,D,H,W) warped moving image
      - fix:      (B,1,D,H,W) fixed image
      - flow:     (B,3,D,H,W) current flow (context)
      - l2_feat:  (B,C,D,H,W) optional L2 decoder features (skip connection)
    Output:
      - delta_flow: (B,3,D,H,W)
    """
    def __init__(self, base_ch: int = 16, error_mode: str = "absdiff", l2_skip_ch: int = 0,
                 use_cab: bool = False, num_context_blocks: int = 0):
        super().__init__()
        self.error_mode = str(error_mode)
        self.l2_skip_ch = int(l2_skip_ch)
        c = int(base_ch)
        in_ch = 6 + self.l2_skip_ch  # mov_warp(1) + fix(1) + err(1) + flow(3) + optional l2_feat
        self.enc1 = ConvBlock(in_ch, c)
        self.pool1 = nn.AvgPool3d(2)
        self.enc2 = ConvBlock(c, c * 2)
        self.pool2 = nn.AvgPool3d(2)
        self.bot = ConvBlock(c * 2, c * 4)
        # Optional context blocks in bottleneck (like L1)
        if num_context_blocks > 0:
            self.context = nn.Sequential(*[
                ResidualContext3D(c * 4, dilation=1 + (i % 2), scale=0.1)
                for i in range(int(num_context_blocks))
            ])
        else:
            self.context = None
        self.up2 = nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False)
        self.dec2 = ConvBlock(c * 4 + c * 2, c * 2)
        self.up1 = nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False)
        self.dec1 = ConvBlock(c * 2 + c, c)
        self.cab = CAB(c, compress_ratio=3, squeeze_factor=30) if use_cab else None
        self.out = nn.Conv3d(c, 3, kernel_size=3, padding=1, bias=True)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)


    @staticmethod
    def _grad_mag(x: torch.Tensor) -> torch.Tensor:
        dz = x[:, :, 1:, :, :] - x[:, :, :-1, :, :]
        dy = x[:, :, :, 1:, :] - x[:, :, :, :-1, :]
        dx = x[:, :, :, :, 1:] - x[:, :, :, :, :-1]
        dz = F.pad(dz, (0, 0, 0, 0, 0, 1))
        dy = F.pad(dy, (0, 0, 0, 1, 0, 0))
        dx = F.pad(dx, (0, 1, 0, 0, 0, 0))
        return torch.sqrt(dx * dx + dy * dy + dz * dz + 1e-6)


    @staticmethod
    @torch.amp.custom_fwd(device_type="cuda", cast_inputs=torch.float32)
    def _local_ncc_map(a: torch.Tensor, b: torch.Tensor, win: int = 9) -> torch.Tensor:
        """Return per-voxel 1 - NCC² map (0 = perfect match, 1 = no correlation).

        Forces float32 to avoid catastrophic cancellation in cross-correlation
        sums when running under float16 autocast (9³=729 taps).
        """
        pad = win // 2
        filt = torch.ones((1, 1, win, win, win), device=a.device, dtype=torch.float32)
        kw = dict(stride=(1, 1, 1), padding=(pad, pad, pad))
        a_sum = F.conv3d(a, filt, **kw)
        b_sum = F.conv3d(b, filt, **kw)
        a2_sum = F.conv3d(a * a, filt, **kw)
        b2_sum = F.conv3d(b * b, filt, **kw)
        ab_sum = F.conv3d(a * b, filt, **kw)
        n = float(win ** 3)
        ua, ub = a_sum / n, b_sum / n
        cross = ab_sum - ub * a_sum - ua * b_sum + ua * ub * n
        a_var = (a2_sum - 2 * ua * a_sum + ua * ua * n).clamp(min=1e-5)
        b_var = (b2_sum - 2 * ub * b_sum + ub * ub * n).clamp(min=1e-5)
        ncc2 = ((cross * cross) / (a_var * b_var)).clamp(0.0, 1.0)
        return 1.0 - ncc2


    def _error_map(self, mov_w: torch.Tensor, fix: torch.Tensor) -> torch.Tensor:
        if self.error_mode == "absdiff":
            return (mov_w - fix).abs()
        if self.error_mode == "gradmag":
            return (self._grad_mag(mov_w) - self._grad_mag(fix)).abs()
        if self.error_mode == "ncc":
            with torch.no_grad():
                return self._local_ncc_map(mov_w, fix)
        raise ValueError(f"Unsupported error_mode: {self.error_mode}")


    def forward(self, mov_warp: torch.Tensor, fix: torch.Tensor, flow: torch.Tensor,
                l2_feat: Optional[torch.Tensor] = None) -> torch.Tensor:
        err = self._error_map(mov_warp, fix)
        parts = [mov_warp, fix, err, flow]
        if self.l2_skip_ch > 0 and l2_feat is not None:
            # Resize l2_feat to match spatial dims if needed
            if l2_feat.shape[2:] != mov_warp.shape[2:]:
                l2_feat = F.interpolate(l2_feat, size=mov_warp.shape[2:], mode="trilinear", align_corners=False)
            parts.append(l2_feat)
        x = torch.cat(parts, dim=1)
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        b = self.bot(self.pool2(e2))
        if self.context is not None:
            b = self.context(b)
        d2 = self.up2(b)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))
        if self.cab is not None:
            d1 = d1 + self.cab(d1)
        return self.out(d1)
