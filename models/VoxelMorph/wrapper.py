"""VoxelMorph variants built from the legacy model.py components."""

import torch
import torch.nn as nn
from torch.distributions.normal import Normal

from models.VoxelMorph.model import unet_core, SpatialTransformer


class VecInt(nn.Module):
    """Scaling-and-squaring integration for a stationary velocity field."""
    def __init__(self, vol_size, int_steps=7):
        super().__init__()
        self.int_steps = int_steps
        self.scale = 1.0 / (2 ** self.int_steps)
        self.transformer = SpatialTransformer(vol_size)


    def forward(self, vel):
        disp = vel * self.scale
        for _ in range(self.int_steps):
            disp = disp + self.transformer(disp, disp)
        return disp


class VxmDense(nn.Module):
    """VoxelMorph dense diffeomorphic network."""
    def __init__(self, vol_size, enc_nf, dec_nf, int_steps=7, bidir=False):
        super().__init__()

        dim = len(vol_size)
        self.bidir = bidir

        self.unet_model = unet_core(dim, enc_nf, dec_nf, full_size=True)
        conv_fn = getattr(nn, 'Conv%dd' % dim)
        self.flow = conv_fn(dec_nf[-1], dim, kernel_size=3, padding=1)

        nd = Normal(0, 1e-5)
        self.flow.weight = nn.Parameter(nd.sample(self.flow.weight.shape))
        self.flow.bias = nn.Parameter(torch.zeros(self.flow.bias.shape))

        self.integrate = VecInt(vol_size, int_steps)
        self.spatial_transform = SpatialTransformer(vol_size)


    def forward(self, src, tgt):
        x = torch.cat([src, tgt], dim=1)
        x = self.unet_model(x)
        vel = self.flow(x)

        disp = self.integrate(vel)

        y = self.spatial_transform(src, disp)

        if self.bidir:
            disp_neg = self.integrate(-vel)
            y_tgt = self.spatial_transform(tgt, disp_neg)
            return y, disp, y_tgt, disp_neg

        return y, disp


class VxmDenseHalf(nn.Module):
    """Cascade-compatible VoxelMorph L2 backbone."""
    def __init__(self, vol_size, enc_nf, dec_nf, int_steps=7):
        super().__init__()
        self.vol_size = tuple(vol_size)
        self.vxm = VxmDense(vol_size, enc_nf, dec_nf, int_steps=int_steps, bidir=False)
        self.spatial_transform = self.vxm.spatial_transform


    def forward(
        self,
        mov_half: torch.Tensor,
        fix_half: torch.Tensor,
        init_flow=None,
        return_all_flows: bool = False,
        l1_feat=None,
        return_features: bool = False,
    ):
        if init_flow is not None: mov_warped = self.spatial_transform(mov_half, init_flow)
        else: mov_warped = mov_half

        _, disp = self.vxm(mov_warped, fix_half)

        # Compose flows: total(x) = disp(x) + init(x + disp(x))
        if init_flow is not None: flow_total = disp + self.spatial_transform(init_flow, disp)
        else: flow_total = disp

        def_half = self.spatial_transform(mov_half, flow_total)
        out = (def_half, flow_total)
        if return_features:
            # VxmDense has no decoder features to share with L3.
            feat = torch.zeros(
                mov_half.shape[0], 16, *self.vol_size,
                device=mov_half.device, dtype=mov_half.dtype,
            )
            return out + (feat,)
        return out


class VxmCascadeL2(VxmDenseHalf):
    """VoxelMorph as a CTCF Level-2 backbone, built directly from CTCF config."""

    def __init__(self, config):
        vxm = getattr(config, "vxm", None)
        l2_full_res = bool(getattr(config, "l2_full_res", False))
        vol = tuple(config.img_size) if l2_full_res else tuple(s // 2 for s in config.img_size)
        super().__init__(
            vol_size=vol,
            enc_nf=list(vxm.enc_nf) if vxm else [16, 32, 32, 32],
            dec_nf=list(vxm.dec_nf) if vxm else [32, 32, 32, 32, 32, 16, 16],
            int_steps=int(vxm.int_steps) if vxm else 7,
        )
