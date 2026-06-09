"""VMambaMorph: a Visual-Mamba registration framework that replaces MambaMorph's
single-scan layer with a 4-direction cross-scan (SS2D) over non-causal 3D data.

Source:  https://github.com/ziyangwang007/VMambaMorph
Authors: Ziyang Wang, Jianqing Zheng, Chao Ma, Tao Guo
Paper:   Wang et al., "VMambaMorph: a Visual Mamba-based Framework with Cross-Scan Module
         for Deformable 3D Image Registration", 2024 (arXiv:2404.05105)
License: Apache-2.0

Self-contained port of the relevant classes (mambamorph/torch/{vmamba,mamba,TransMorph}):
  - the 4-direction `SS2D` selective-scan block, `VSSBlock`, `VMambaLayer`, `VMambaBlock`
    encoder, and the `VMambaMorph` model are defined here;
  - common decoder / spatial-transform / VecInt blocks are imported from
    `models.MambaMorph.model` to avoid duplication.

Protocol note: configs set img_size = (160, 192, 224) (our protocol) instead of the
upstream (176, 208, 192); with patch_size=4 and 3 encoder stages the spatial sizes
40x48x56 -> 20x24x28 -> 10x12x14 stay cleanly divisible by 2.
"""

import math
from functools import partial
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as nnf
from einops import repeat
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
from timm.models.layers import DropPath, to_3tuple, trunc_normal_

# Reuse common blocks from the MambaMorph port
from models.MambaMorph.model import (
    Conv3dReLU,
    DecoderBlock,
    PatchEmbed,
    PatchMerging,
    RegistrationHead,
    SinPositionalEncoding3D,
    SpatialTransformer,
    VecInt,
)


class SS2D(nn.Module):
    """3D selective scan with 4-direction cross-scan.

    Despite the name (kept for fidelity with the upstream code) this operates
    on 5D tensors of shape (B, H, W, D, C) reshaped from sequence form.
    """

    def __init__(
        self,
        d_model,
        d_state=16,
        d_conv=3,
        expand=2,
        dt_rank="auto",
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        dropout=0.0,
        conv_bias=True,
        bias=False,
        device=None,
        dtype=None,
        **kwargs,
    ):
        factory = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = to_3tuple(d_conv)
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory)
        self.conv3d = nn.Conv3d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory,
        )
        self.act = nn.SiLU()

        # 4-direction projections: stacked weights (K=4)
        x_proj = tuple(
            nn.Linear(self.d_inner, self.dt_rank + self.d_state * 2, bias=False, **factory) for _ in range(4)
        )
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in x_proj], dim=0))

        dt_projs = tuple(
            self._dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory)
            for _ in range(4)
        )
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in dt_projs], dim=0))
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in dt_projs], dim=0))

        self.A_logs = self._A_log_init(self.d_state, self.d_inner, copies=4, merge=True)
        self.Ds = self._D_init(self.d_inner, copies=4, merge=True)

        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory)
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else None

    @staticmethod
    def _dt_init(
        dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4, **factory
    ):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory)
        dt_init_std = dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError
        dt = torch.exp(torch.rand(d_inner, **factory) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)).clamp(
            min=dt_init_floor
        )
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        dt_proj.bias._no_reinit = True
        return dt_proj

    @staticmethod
    def _A_log_init(d_state, d_inner, copies=1, device=None, merge=True):
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)
        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def _D_init(d_inner, copies=1, device=None, merge=True):
        D = torch.ones(d_inner, device=device)
        if copies > 1:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)
        D._no_weight_decay = True
        return D

    def _forward_core(self, x):
        B, C, H, W, D = x.shape
        L = H * W * D
        K = 4
        x_hwwh = torch.stack(
            [x.view(B, -1, L), torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, -1, L)],
            dim=1,
        ).view(B, 2, -1, L)
        xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1)

        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight)

        xs = xs.float().view(B, -1, L)
        dts = dts.contiguous().float().view(B, -1, L)
        Bs = Bs.float().view(B, K, -1, L)
        Cs = Cs.float().view(B, K, -1, L)
        Ds = self.Ds.float().view(-1)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1)

        out_y = selective_scan_fn(
            xs,
            dts,
            As,
            Bs,
            Cs,
            Ds,
            z=None,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
            return_last_state=False,
        ).view(B, K, -1, L)
        assert out_y.dtype == torch.float

        inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)
        wh_y = torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        invwh_y = torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        y = out_y[:, 0] + inv_y[:, 0] + wh_y + invwh_y

        y = torch.transpose(y, dim0=1, dim1=2).contiguous().view(B, H, W, D, -1)
        y = self.out_norm(y).to(x.dtype)
        return y

    def forward(self, x, H, W, T):
        # input: (B, L, C). The upstream code assumed L is a perfect cube
        # (root**3 == L), which breaks for typical brain MRI tensor shapes
        # (e.g. 40x48x56 after patch embedding). We instead receive the actual
        # (H, W, T) from the caller and reshape exactly.
        B, L, C = x.shape
        assert L == H * W * T, f"SS2D shape mismatch: L={L}, H*W*T={H * W * T}"
        x = x.view(B, H, W, T, C)
        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)
        x = x.permute(0, 4, 1, 2, 3).contiguous()
        x = self.act(self.conv3d(x))
        y = self._forward_core(x)
        y = y * nnf.silu(z)
        out = self.out_proj(y)
        if self.dropout is not None:
            out = self.dropout(out)
        return out


class VSSBlock(nn.Module):
    def __init__(
        self,
        hidden_dim,
        drop_path=0.0,
        norm_layer: Callable[..., nn.Module] = partial(nn.LayerNorm, eps=1e-6),
        attn_drop_rate=0.0,
        d_state=16,
        **kwargs,
    ):
        super().__init__()
        self.ln_1 = norm_layer(hidden_dim)
        self.self_attention = SS2D(d_model=hidden_dim, dropout=attn_drop_rate, d_state=d_state, **kwargs)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x, H, W, T):
        a = self.drop_path(self.self_attention(self.ln_1(x), H, W, T))
        # SS2D returns (B, H, W, T, C); flatten back to sequence form to match
        # the residual pathway and to feed the next block uniformly.
        B = a.shape[0]
        C = a.shape[-1]
        a = a.view(B, H * W * T, C)
        return x + a


class VMambaLayer(nn.Module):
    def __init__(self, dim, depths, d_state=16, d_conv=4, expand=2, downsample=None):
        super().__init__()
        self.dim = dim
        self.norm = nn.LayerNorm(dim)
        self.mamba = nn.ModuleList(
            [
                VSSBlock(hidden_dim=dim, drop_path=0.0, norm_layer=nn.LayerNorm, attn_drop_rate=0.0, d_state=d_state)
                for _ in range(depths)
            ]
        )
        self.downsample = (
            downsample(dim=dim, norm_layer=nn.LayerNorm, reduce_factor=4) if downsample is not None else None
        )

    def forward(self, x, H, W, T):
        assert x.shape[-1] == self.dim
        x_norm = self.norm(x)
        if x_norm.dtype == torch.float16:
            x_norm = x_norm.float()
        for blk in self.mamba:
            x_norm = blk(x_norm, H, W, T)
        x = x_norm.to(x.dtype)
        if self.downsample is not None:
            x_down = self.downsample(x, H, W, T)
            return x, H, W, T, x_down, (H + 1) // 2, (W + 1) // 2, (T + 1) // 2
        return x, H, W, T, x, H, W, T


class VMambaBlock(nn.Module):
    def __init__(
        self,
        patch_size=4,
        in_chans=2,
        embed_dim=96,
        depths=(2, 2, 2),
        drop_rate=0.0,
        norm_layer=nn.LayerNorm,
        ape=False,
        spe=False,
        rpe=True,
        patch_norm=True,
        out_indices=(0, 1, 2),
        frozen_stages=-1,
        d_state=16,
        d_conv=4,
        expand=2,
        pretrain_img_size=224,
    ):
        super().__init__()
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.spe = spe
        self.out_indices = out_indices

        self.patch_embed = PatchEmbed(
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            norm_layer=norm_layer if patch_norm else None,
        )
        if ape:
            ps3 = to_3tuple(patch_size)
            ts3 = to_3tuple(pretrain_img_size)
            res = [ts3[i] // ps3[i] for i in range(3)]
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, embed_dim, *res))
            trunc_normal_(self.absolute_pos_embed, std=0.02)
        elif spe:
            self.pos_embd = SinPositionalEncoding3D(embed_dim)
        self.pos_drop = nn.Dropout(p=drop_rate)

        self.layers = nn.ModuleList()
        for i in range(self.num_layers):
            self.layers.append(
                VMambaLayer(
                    dim=int(embed_dim * 2**i),
                    depths=depths[i],
                    d_state=d_state,
                    d_conv=d_conv,
                    expand=expand,
                    downsample=PatchMerging if (i < self.num_layers - 1) else None,
                )
            )

        self.num_features = [int(embed_dim * 2**i) for i in range(self.num_layers)]
        for i in out_indices:
            self.add_module(f"norm{i}", norm_layer(self.num_features[i]))

    def forward(self, x):
        x = self.patch_embed(x)
        Wh, Ww, Wt = x.size(2), x.size(3), x.size(4)
        if self.ape:
            absolute_pos_embed = nnf.interpolate(
                self.absolute_pos_embed,
                size=(Wh, Ww, Wt),
                mode="trilinear",
            )
            x = (x + absolute_pos_embed).flatten(2).transpose(1, 2)
        elif self.spe:
            x = (x + self.pos_embd(x)).flatten(2).transpose(1, 2)
        else:
            x = x.flatten(2).transpose(1, 2)
        x = self.pos_drop(x)

        outs = []
        for i in range(self.num_layers):
            x_out, H, W, T, x, Wh, Ww, Wt = self.layers[i](x, Wh, Ww, Wt)
            if i in self.out_indices:
                norm_layer = getattr(self, f"norm{i}")
                x_out = norm_layer(x_out)
                out = x_out.view(-1, H, W, T, self.num_features[i]).permute(0, 4, 1, 2, 3).contiguous()
                outs.append(out)
        return outs


class VMambaMorph(nn.Module):
    """Diffeomorphic VMambaMorph: VSS-based encoder + decoder + VecInt SVF integration."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.if_convskip = config.if_convskip
        self.if_transskip = config.if_transskip
        embed_dim = config.embed_dim

        self.transformer = VMambaBlock(
            patch_size=config.patch_size,
            in_chans=config.in_chans,
            embed_dim=embed_dim,
            depths=config.depths,
            drop_rate=config.drop_rate,
            ape=config.ape,
            spe=config.spe,
            rpe=config.rpe,
            patch_norm=config.patch_norm,
            out_indices=config.out_indices,
            d_state=config.d_state,
            d_conv=config.d_conv,
            expand=config.expand,
        )
        self.up1 = DecoderBlock(
            embed_dim * 4, embed_dim * 2, skip_channels=embed_dim * 2 if self.if_transskip else 0, use_batchnorm=False
        )
        self.up2 = DecoderBlock(
            embed_dim * 2, embed_dim, skip_channels=embed_dim if self.if_transskip else 0, use_batchnorm=False
        )
        self.up3 = DecoderBlock(
            embed_dim, embed_dim // 2, skip_channels=embed_dim // 2 if self.if_convskip else 0, use_batchnorm=False
        )
        self.up4 = DecoderBlock(
            embed_dim // 2,
            config.reg_head_chan,
            skip_channels=config.reg_head_chan if self.if_convskip else 0,
            use_batchnorm=False,
        )
        self.c1 = Conv3dReLU(2, embed_dim // 2, 3, 1, use_batchnorm=False)
        self.c2 = Conv3dReLU(2, config.reg_head_chan, 3, 1, use_batchnorm=False)
        self.reg_head = RegistrationHead(in_channels=config.reg_head_chan, out_channels=3, kernel_size=3)
        self.spatial_trans = SpatialTransformer(config.img_size)
        self.avg_pool = nn.AvgPool3d(3, stride=2, padding=1)
        self.integrate = VecInt(config.img_size, nsteps=7)

    def forward(self, source, target, return_pos_flow=True):
        x = torch.cat([source, target], dim=1)
        if self.if_convskip:
            f4 = self.c1(self.avg_pool(x))
            f5 = self.c2(x)
        else:
            f4 = f5 = None
        feats = self.transformer(x)
        f1 = feats[-2] if self.if_transskip else None
        f2 = feats[-3] if self.if_transskip else None
        x = self.up1(feats[-1], f1)
        x = self.up2(x, f2)
        x = self.up3(x, f4)
        x = self.up4(x, f5)
        flow = self.reg_head(x)
        pos_flow = self.integrate(flow)
        moved = self.spatial_trans(source, pos_flow)
        ret = {"moved_vol": moved, "preint_flow": flow}
        if return_pos_flow:
            ret["pos_flow"] = pos_flow
        return ret
