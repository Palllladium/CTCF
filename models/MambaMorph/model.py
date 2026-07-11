"""MambaMorph: a Mamba-based U-Net for deformable registration; a Mamba selective-scan
block replaces transformer self-attention in the encoder.

Source:  https://github.com/Guo-Stone/MambaMorph
Authors: Yinuo Wang, Tao Guo, Weimin Yuan, Shihao Shu, Cai Meng, Xiangzhi Bai
Paper:   Wang et al., "Mamba-based deformable medical image registration with an
         annotated brain MR-CT dataset", 2025 (arXiv:2401.13934)
License: MIT (Copyright (c) 2024 Guo-Stone)

Self-contained port of mambamorph/torch/, with the dependency graph collapsed and:
  - removed the hardcoded sys.path append to the author's local voxelmorph-dev clone;
  - removed `.cuda()` hardcoding in `SinPositionalEncoding3D` and `MambaMorph.__init__`;
  - registered `inv_freq` as a buffer so positional encoding follows .to(device);
  - dropped the MR-CT-specific feature-extractor variants; kept only `MambaMorph`
    (VecInt diffeomorphic integration) and `MambaMorphOri` (no integration).

Protocol note: configs set img_size = (160, 192, 224) (our protocol) instead of the
upstream (176, 208, 192); with patch_size=4 and 3 encoder stages this divides cleanly
(160/4=40, 192/4=48, 224/4=56).

Requires `mamba_ssm` in the active environment (see tools/install_mamba.sh).
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as nnf
from mamba_ssm import Mamba
from timm.models.layers import to_3tuple, trunc_normal_
from torch.distributions.normal import Normal


class PatchEmbed(nn.Module):
    def __init__(self, patch_size=4, in_chans=2, embed_dim=96, norm_layer=None):
        super().__init__()
        patch_size = to_3tuple(patch_size)
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer is not None else None

    def forward(self, x):
        _, _, H, W, T = x.size()
        if T % self.patch_size[2] != 0:
            x = nnf.pad(x, (0, self.patch_size[2] - T % self.patch_size[2]))
        if W % self.patch_size[1] != 0:
            x = nnf.pad(x, (0, 0, 0, self.patch_size[1] - W % self.patch_size[1]))
        if H % self.patch_size[0] != 0:
            x = nnf.pad(x, (0, 0, 0, 0, 0, self.patch_size[0] - H % self.patch_size[0]))
        x = self.proj(x)
        if self.norm is not None:
            Wh, Ww, Wt = x.size(2), x.size(3), x.size(4)
            x = x.flatten(2).transpose(1, 2)
            x = self.norm(x)
            x = x.transpose(1, 2).view(-1, self.embed_dim, Wh, Ww, Wt)
        return x


class SinPositionalEncoding3D(nn.Module):
    def __init__(self, channels):
        super().__init__()
        channels = int(np.ceil(channels / 6) * 2)
        if channels % 2:
            channels += 1
        self.channels = channels
        inv_freq = 1.0 / (10000 ** (torch.arange(0, channels, 2).float() / channels))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, tensor):
        tensor = tensor.permute(0, 2, 3, 4, 1)
        if len(tensor.shape) != 5:
            raise RuntimeError("The input tensor has to be 5d!")
        bs, x, y, z, orig_ch = tensor.shape
        device = tensor.device
        pos_x = torch.arange(x, device=device).to(self.inv_freq.dtype)
        pos_y = torch.arange(y, device=device).to(self.inv_freq.dtype)
        pos_z = torch.arange(z, device=device).to(self.inv_freq.dtype)
        sin_x = torch.einsum("i,j->ij", pos_x, self.inv_freq)
        sin_y = torch.einsum("i,j->ij", pos_y, self.inv_freq)
        sin_z = torch.einsum("i,j->ij", pos_z, self.inv_freq)
        emb_x = torch.cat((sin_x.sin(), sin_x.cos()), dim=-1).unsqueeze(1).unsqueeze(1)
        emb_y = torch.cat((sin_y.sin(), sin_y.cos()), dim=-1).unsqueeze(1)
        emb_z = torch.cat((sin_z.sin(), sin_z.cos()), dim=-1)
        emb = torch.zeros((x, y, z, self.channels * 3), device=device, dtype=tensor.dtype)
        emb[..., : self.channels] = emb_x
        emb[..., self.channels : 2 * self.channels] = emb_y
        emb[..., 2 * self.channels :] = emb_z
        emb = emb[None, ..., :orig_ch].repeat(bs, 1, 1, 1, 1)
        return emb.permute(0, 4, 1, 2, 3)


class PatchMerging(nn.Module):
    def __init__(self, dim, norm_layer=nn.LayerNorm, reduce_factor=2):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(8 * dim, (8 // reduce_factor) * dim, bias=False)
        self.norm = norm_layer(8 * dim)

    def forward(self, x, H, W, T):
        B, L, C = x.shape
        assert L == H * W * T, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0 and T % 2 == 0, f"x size ({H}*{W}) are not even."
        x = x.view(B, H, W, T, C)
        if (H % 2 == 1) or (W % 2 == 1) or (T % 2 == 1):
            x = nnf.pad(x, (0, 0, 0, T % 2, 0, W % 2, 0, H % 2))
        x0 = x[:, 0::2, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, 0::2, :]
        x3 = x[:, 0::2, 0::2, 1::2, :]
        x4 = x[:, 1::2, 1::2, 0::2, :]
        x5 = x[:, 0::2, 1::2, 1::2, :]
        x6 = x[:, 1::2, 0::2, 1::2, :]
        x7 = x[:, 1::2, 1::2, 1::2, :]
        x = torch.cat([x0, x1, x2, x3, x4, x5, x6, x7], -1)
        x = x.view(B, -1, 8 * C)
        x = self.norm(x)
        x = self.reduction(x)
        return x


class MambaLayer(nn.Module):
    def __init__(self, dim, d_state=16, d_conv=4, expand=2, downsample=None):
        super().__init__()
        self.dim = dim
        self.norm = nn.LayerNorm(dim)
        self.mamba = Mamba(d_model=dim, d_state=d_state, d_conv=d_conv, expand=expand)
        self.downsample = (
            downsample(dim=dim, norm_layer=nn.LayerNorm, reduce_factor=4) if downsample is not None else None
        )

    def forward(self, x, H, W, T):
        assert x.shape[-1] == self.dim
        x_norm = self.norm(x)
        if x_norm.dtype == torch.float16:
            x_norm = x_norm.float()
        x_mamba = self.mamba(x_norm)
        x = x_mamba.to(x.dtype)
        if self.downsample is not None:
            x_down = self.downsample(x, H, W, T)
            return x, H, W, T, x_down, (H + 1) // 2, (W + 1) // 2, (T + 1) // 2
        return x, H, W, T, x, H, W, T


class MambaBlock(nn.Module):
    def __init__(
        self,
        patch_size=4,
        in_chans=2,
        embed_dim=96,
        depths=(2, 2, 4),
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
        self.rpe = rpe
        self.patch_norm = patch_norm
        self.out_indices = out_indices
        self.frozen_stages = frozen_stages

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
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, embed_dim, res[0], res[1], res[2]))
            trunc_normal_(self.absolute_pos_embed, std=0.02)
        elif spe:
            self.pos_embd = SinPositionalEncoding3D(embed_dim)
        self.pos_drop = nn.Dropout(p=drop_rate)

        self.layers = nn.ModuleList()
        for i in range(self.num_layers):
            self.layers.append(
                MambaLayer(
                    dim=int(embed_dim * 2**i),
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
            absolute_pos_embed = nnf.interpolate(self.absolute_pos_embed, size=(Wh, Ww, Wt), mode="trilinear")
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


class Conv3dReLU(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size, padding=0, stride=1, use_batchnorm=True):
        conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=False)
        nm = nn.BatchNorm3d(out_channels) if use_batchnorm else nn.InstanceNorm3d(out_channels)
        super().__init__(conv, nm, nn.LeakyReLU(inplace=True))


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, skip_channels=0, use_batchnorm=True):
        super().__init__()
        self.conv1 = Conv3dReLU(
            in_channels + skip_channels, out_channels, kernel_size=3, padding=1, use_batchnorm=use_batchnorm
        )
        self.conv2 = Conv3dReLU(out_channels, out_channels, kernel_size=3, padding=1, use_batchnorm=use_batchnorm)
        self.up = nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False)

    def forward(self, x, skip=None):
        x = self.up(x)
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class RegistrationHead(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        conv3d = nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        conv3d.weight = nn.Parameter(Normal(0, 1e-5).sample(conv3d.weight.shape))
        conv3d.bias = nn.Parameter(torch.zeros(conv3d.bias.shape))
        super().__init__(conv3d)


class SpatialTransformer(nn.Module):
    """N-D Spatial Transformer (voxel-unit flow), buffer-backed grid."""

    def __init__(self, size, mode="bilinear"):
        super().__init__()
        self.mode = mode
        vectors = [torch.arange(0, s) for s in size]
        grid = torch.stack(torch.meshgrid(vectors, indexing="ij")).unsqueeze(0).float()
        self.register_buffer("grid", grid, persistent=False)

    def forward(self, src, flow):
        new_locs = self.grid + flow
        shape = flow.shape[2:]
        for i in range(len(shape)):
            new_locs[:, i, ...] = 2 * (new_locs[:, i, ...] / (shape[i] - 1) - 0.5)
        if len(shape) == 3:
            new_locs = new_locs.permute(0, 2, 3, 4, 1)[..., [2, 1, 0]]
        elif len(shape) == 2:
            new_locs = new_locs.permute(0, 2, 3, 1)[..., [1, 0]]
        return nnf.grid_sample(src, new_locs, align_corners=True, mode=self.mode)


class VecInt(nn.Module):
    """Scaling-and-squaring velocity field integration."""

    def __init__(self, inshape, nsteps=7):
        super().__init__()
        assert nsteps >= 0
        self.nsteps = nsteps
        self.scale = 1.0 / (2**nsteps)
        self.transformer = SpatialTransformer(inshape)

    def forward(self, vec):
        vec = vec * self.scale
        for _ in range(self.nsteps):
            vec = vec + self.transformer(vec, vec)
        return vec


class MambaMorph(nn.Module):
    """Diffeomorphic MambaMorph (with VecInt integration of the predicted velocity)."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.if_convskip = config.if_convskip
        self.if_transskip = config.if_transskip
        embed_dim = config.embed_dim

        self.transformer = MambaBlock(
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


class MambaMorphOri(nn.Module):
    """Non-diffeomorphic MambaMorph (direct displacement, no VecInt)."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.if_convskip = config.if_convskip
        self.if_transskip = config.if_transskip
        embed_dim = config.embed_dim

        self.transformer = MambaBlock(
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

    def forward(self, source, target):
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
        moved = self.spatial_trans(source, flow)
        return {"moved_vol": moved, "preint_flow": flow}
