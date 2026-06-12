from __future__ import annotations

from typing import ClassVar

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.CTCF.blocks import CAB, CostVolume3D, ResidualContext3D, SRUpBlock3D
from models.TransMorph_DCA.model import (
    Conv3dReLU,
    RegistrationHead,
    SpatialTransformer,
    SwinTransformer,
)


class ConvBlock(nn.Module):
    """Two stacked Conv3d -> InstanceNorm3d -> LeakyReLU layers."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(
                in_channels=in_ch,
                out_channels=out_ch,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.InstanceNorm3d(num_features=out_ch, affine=True),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv3d(
                in_channels=out_ch,
                out_channels=out_ch,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.InstanceNorm3d(num_features=out_ch, affine=True),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CoarseFlowNetQuarter(nn.Module):
    """L1: zero-initialised U-Net producing a coarse flow at 1/4 resolution."""

    def __init__(self, base_ch: int = 16):
        super().__init__()
        c = base_ch
        in_ch = 2  # mov + fix concat
        out_ch = 3  # x, y, z displacement components

        self.enc1 = ConvBlock(in_ch=in_ch, out_ch=c)
        self.pool1 = nn.AvgPool3d(kernel_size=2)
        self.enc2 = ConvBlock(in_ch=c, out_ch=c * 2)
        self.pool2 = nn.AvgPool3d(kernel_size=2)

        self.bot = ConvBlock(in_ch=c * 2, out_ch=c * 4)
        self.ctx1 = ResidualContext3D(channels=c * 4, dilation=1, scale=0.1)
        self.ctx2 = ResidualContext3D(channels=c * 4, dilation=2, scale=0.1)

        self.up2 = nn.Upsample(
            scale_factor=2,
            mode="trilinear",
            align_corners=False,
        )
        self.dec2 = ConvBlock(in_ch=c * 4 + c * 2, out_ch=c * 2)
        self.up1 = nn.Upsample(
            scale_factor=2,
            mode="trilinear",
            align_corners=False,
        )
        self.dec1 = ConvBlock(in_ch=c * 2 + c, out_ch=c)

        self.out = nn.Conv3d(
            in_channels=c,
            out_channels=out_ch,
            kernel_size=3,
            padding=1,
            bias=True,
        )
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, mov: torch.Tensor, fix: torch.Tensor) -> torch.Tensor:
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

        return self.out(d1)


class CTCFDCACoreHalf(nn.Module):
    """L2 Swin-DCA backbone: Swin encoder + SR-style decoder + time-step velocity integration."""

    _CKPT_RENAMES: ClassVar[dict[str, str]] = {
        "cs.": "time_skip_convs.",
        "up3s.": "time_up_blocks.",
        "c1.": "conv_skip_proj.",
    }

    def __init__(self, config, time_steps: int):
        super().__init__()

        self.if_convskip = config.if_convskip
        self.if_transskip = config.if_transskip
        self.time_steps = time_steps
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

        c0, c1, c2 = self.transformer.num_features[:3]
        self.c_mid = max(1, c0 // 2)

        skip_ch_2 = 2  # mov + fix concat for conv-skip projection

        self.cab0 = CAB(num_feat=c2, compress_ratio=3, squeeze_factor=30)
        self.cab1 = CAB(num_feat=c1, compress_ratio=3, squeeze_factor=30)
        self.cab2 = CAB(num_feat=c0, compress_ratio=3, squeeze_factor=30)

        self.up0 = SRUpBlock3D(
            in_channels=c2,
            out_channels=c1,
            skip_channels=(c1 if self.if_transskip else 0),
        )
        self.up1 = SRUpBlock3D(
            in_channels=c1,
            out_channels=c0,
            skip_channels=(c0 if self.if_transskip else 0),
        )
        self.avg_pool = nn.AvgPool3d(
            kernel_size=3,
            stride=2,
            padding=1,
        )

        self.conv_skip_proj = Conv3dReLU(
            in_channels=skip_ch_2,
            out_channels=self.c_mid,
            kernel_size=3,
            stride=1,
            use_batchnorm=False,
        )
        self.up2 = SRUpBlock3D(
            in_channels=c0,
            out_channels=self.c_mid,
            skip_channels=(self.c_mid if self.if_convskip else 0),
        )

        reg_ch = config.reg_head_chan
        flow_ch = 3  # x, y, z displacement components

        self.time_skip_convs = nn.ModuleList()
        self.time_up_blocks = nn.ModuleList()
        self.reg_heads = nn.ModuleList()
        for _ in range(self.time_steps):
            self.time_skip_convs.append(
                Conv3dReLU(
                    in_channels=skip_ch_2,
                    out_channels=self.c_mid,
                    kernel_size=3,
                    stride=1,
                    use_batchnorm=False,
                ),
            )
            self.time_up_blocks.append(
                SRUpBlock3D(
                    in_channels=self.c_mid,
                    out_channels=reg_ch,
                    skip_channels=(self.c_mid if self.if_convskip else 0),
                ),
            )
            self.reg_heads.append(
                RegistrationHead(
                    in_channels=reg_ch,
                    out_channels=flow_ch,
                    kernel_size=3,
                ),
            )

        self.spatial_trans = SpatialTransformer(self.img_size)

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        *args,
        **kwargs,
    ):
        """Rewrite legacy parameter keys before delegating to the parent loader."""
        for old, new in self._CKPT_RENAMES.items():
            old_full = prefix + old
            new_full = prefix + new
            for key in list(state_dict.keys()):
                if key.startswith(old_full):
                    state_dict[new_full + key[len(old_full) :]] = state_dict.pop(key)
        super()._load_from_state_dict(
            state_dict,
            prefix,
            *args,
            **kwargs,
        )

    def forward(
        self,
        mov_half: torch.Tensor,
        fix_half: torch.Tensor,
        init_flow: torch.Tensor | None = None,
        return_all_flows: bool = False,
    ):
        if init_flow is None:
            flow_prev = torch.zeros(
                (mov_half.shape[0], 3, *self.img_size),
                device=mov_half.device,
                dtype=mov_half.dtype,
            )
            def_x = mov_half
        else:
            flow_prev = init_flow
            def_x = self.spatial_trans(mov_half, flow_prev)

        x_cat = torch.cat((mov_half, fix_half), dim=1)

        if self.if_convskip:
            f_conv_skip = self.conv_skip_proj(self.avg_pool(x_cat)).to(mov_half.dtype)
        else:
            f_conv_skip = None

        out_feats = self.transformer((mov_half, fix_half))

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
        x = self.up2(x, f_conv_skip)

        if return_all_flows:
            flows: list[torch.Tensor] | None = []
        else:
            flows = None

        for t in range(self.time_steps):
            skip_t = None
            if self.if_convskip:
                skip_t = self.time_skip_convs[t](torch.cat((def_x, fix_half), dim=1))

            x_t = self.time_up_blocks[t](x, skip_t)
            flow_step = self.reg_heads[t](x_t)

            if flows is not None:
                flows.append(flow_step)

            flow_prev = flow_prev + self.spatial_trans(flow_step, flow_prev)
            def_x = self.spatial_trans(mov_half, flow_prev)

        if return_all_flows:
            return def_x, flow_prev, flows
        return def_x, flow_prev


class FlowRefiner3D(nn.Module):
    """L3: NCC-error-driven residual U-Net producing a delta-flow update.

    When num_heads > 1, the single output conv is replaced by K parallel flow heads with
    per-voxel softmax routing. Heads and routing logits are zero-initialised.
    """

    def __init__(
        self,
        base_ch: int = 16,
        error_mode: str = "absdiff",
        num_heads: int = 1,
        corr_mode: str = "none",
    ):
        super().__init__()

        self.error_mode = error_mode
        self.num_heads = max(1, num_heads)
        c = base_ch
        flow_ch = 3  # x, y, z displacement components

        self.cost_volume = None
        corr_ch = 0
        if corr_mode != "none":
            self.cost_volume = CostVolume3D(corr_mode)
            corr_ch = self.cost_volume.extra_channels

        in_ch = 6 + corr_ch  # mov_warp(1) + fix(1) + err(1) + flow(3) + cost-volume(corr_ch)

        self.enc1 = ConvBlock(in_ch=in_ch, out_ch=c)
        self.pool1 = nn.AvgPool3d(kernel_size=2)
        self.enc2 = ConvBlock(in_ch=c, out_ch=c * 2)
        self.pool2 = nn.AvgPool3d(kernel_size=2)

        self.bot = ConvBlock(in_ch=c * 2, out_ch=c * 4)

        self.up2 = nn.Upsample(
            scale_factor=2,
            mode="trilinear",
            align_corners=False,
        )
        self.dec2 = ConvBlock(in_ch=c * 4 + c * 2, out_ch=c * 2)
        self.up1 = nn.Upsample(
            scale_factor=2,
            mode="trilinear",
            align_corners=False,
        )
        self.dec1 = ConvBlock(in_ch=c * 2 + c, out_ch=c)

        if self.num_heads == 1:
            self.out = nn.Conv3d(
                in_channels=c,
                out_channels=flow_ch,
                kernel_size=3,
                padding=1,
                bias=True,
            )
            nn.init.zeros_(self.out.weight)
            nn.init.zeros_(self.out.bias)
        else:
            heads = [
                nn.Conv3d(
                    in_channels=c,
                    out_channels=flow_ch,
                    kernel_size=3,
                    padding=1,
                    bias=True,
                )
                for _ in range(self.num_heads)
            ]
            self.flow_heads = nn.ModuleList(heads)
            for head in self.flow_heads:
                nn.init.zeros_(head.weight)
                nn.init.zeros_(head.bias)

            r_mid = max(1, c // 2)
            self.routing = nn.Sequential(
                nn.Conv3d(
                    in_channels=c,
                    out_channels=r_mid,
                    kernel_size=3,
                    padding=1,
                    bias=True,
                ),
                nn.LeakyReLU(negative_slope=0.1, inplace=True),
                nn.Conv3d(
                    in_channels=r_mid,
                    out_channels=self.num_heads,
                    kernel_size=1,
                    bias=True,
                ),
            )
            nn.init.zeros_(self.routing[-1].weight)
            nn.init.zeros_(self.routing[-1].bias)

    @staticmethod
    def _grad_mag(x: torch.Tensor) -> torch.Tensor:
        """Per-voxel gradient magnitude of a 1-channel volume via forward differences."""
        dz = x[:, :, 1:, :, :] - x[:, :, :-1, :, :]
        dy = x[:, :, :, 1:, :] - x[:, :, :, :-1, :]
        dx = x[:, :, :, :, 1:] - x[:, :, :, :, :-1]

        dz = F.pad(dz, (0, 0, 0, 0, 0, 1))
        dy = F.pad(dy, (0, 0, 0, 1, 0, 0))
        dx = F.pad(dx, (0, 1, 0, 0, 0, 0))

        return torch.sqrt(dx * dx + dy * dy + dz * dz + 1e-6)

    @staticmethod
    @torch.amp.custom_fwd(device_type="cuda", cast_inputs=torch.float32)
    def _local_ncc_map(
        a: torch.Tensor,
        b: torch.Tensor,
        win: int = 9,
    ) -> torch.Tensor:
        """Per-voxel local NCC error map (1 - NCC^2). Forced to float32 for numerical stability."""
        pad = win // 2
        filt = torch.ones(
            (1, 1, win, win, win),
            device=a.device,
            dtype=torch.float32,
        )
        conv_kwargs = {"stride": (1, 1, 1), "padding": (pad, pad, pad)}

        a_sum = F.conv3d(a, filt, **conv_kwargs)
        b_sum = F.conv3d(b, filt, **conv_kwargs)
        a2_sum = F.conv3d(a * a, filt, **conv_kwargs)
        b2_sum = F.conv3d(b * b, filt, **conv_kwargs)
        ab_sum = F.conv3d(a * b, filt, **conv_kwargs)

        n = float(win**3)
        ua, ub = a_sum / n, b_sum / n
        cross = ab_sum - ub * a_sum - ua * b_sum + ua * ub * n
        a_var = (a2_sum - 2 * ua * a_sum + ua * ua * n).clamp(min=1e-5)
        b_var = (b2_sum - 2 * ub * b_sum + ub * ub * n).clamp(min=1e-5)

        ncc2 = ((cross * cross) / (a_var * b_var)).clamp(0.0, 1.0)
        return 1.0 - ncc2

    def _error_map(self, mov_warp: torch.Tensor, fix: torch.Tensor) -> torch.Tensor:
        if self.error_mode == "absdiff":
            return (mov_warp - fix).abs()
        if self.error_mode == "gradmag":
            return (self._grad_mag(mov_warp) - self._grad_mag(fix)).abs()
        if self.error_mode == "ncc":
            with torch.no_grad():
                return self._local_ncc_map(mov_warp, fix)
        raise ValueError(f"Unsupported error_mode: {self.error_mode}")

    def forward(
        self,
        mov_warp: torch.Tensor,
        fix: torch.Tensor,
        flow: torch.Tensor,
    ) -> torch.Tensor:
        err = self._error_map(mov_warp, fix)
        parts = [mov_warp, fix, err, flow]
        if self.cost_volume is not None:
            parts.append(self.cost_volume(mov_warp, fix))
        x = torch.cat(parts, dim=1)

        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        b = self.bot(self.pool2(e2))

        d2 = self.up2(b)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
        d1 = self.up1(d2)
        feats = self.dec1(torch.cat([d1, e1], dim=1))

        if self.num_heads == 1:
            return self.out(feats)

        flows = torch.stack([h(feats) for h in self.flow_heads], dim=1)  # (B, K, 3, D, H, W)
        weights = F.softmax(self.routing(feats), dim=1).unsqueeze(2)  # (B, K, 1, D, H, W)
        return (flows * weights).sum(dim=1)
