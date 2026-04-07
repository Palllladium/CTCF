"""
VoxelMorph diffeomorphic (TMI 2019) assembled from original components.

Uses:
  - unet_core, SpatialTransformer, conv_block from model.py (legacy branch, UNTOUCHED)
  - VecInt: standard scaling-and-squaring velocity integration
    (identical logic to voxelmorph dev branch IntegrateVelocityField)

Standard VoxelMorph-2 configuration:
  enc_nf = [16, 32, 32, 32]
  dec_nf = [32, 32, 32, 32, 32, 16, 16]
  int_steps = 7
"""

import torch
import torch.nn as nn
from torch.distributions.normal import Normal

from models.VoxelMorph.model import unet_core, SpatialTransformer


class VecInt(nn.Module):
    """
    Vector integration via scaling and squaring.

    Given a stationary velocity field v, computes the displacement field
    phi = exp(v) by:
      1. Scaling: v <- v / 2^steps
      2. Squaring: for i in range(steps): v <- v + ST(v, v)

    This is the standard method from:
      Arsigny et al., "A Log-Euclidean Framework for Statistics on
      Diffeomorphisms", MICCAI 2006.

    Used in VoxelMorph (Dalca et al., TMI 2019) for diffeomorphic registration.
    Logic identical to voxelmorph.nn.modules.IntegrateVelocityField (dev branch).
    """
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
    """
    VoxelMorph dense diffeomorphic network (Dalca et al., TMI 2019).

    Architecture: U-Net -> velocity field -> scaling & squaring -> displacement -> warp.
    Uses the original unet_core and SpatialTransformer from the VoxelMorph legacy branch.
    """
    def __init__(self, vol_size, enc_nf, dec_nf, int_steps=7, bidir=False):
        super().__init__()

        dim = len(vol_size)
        self.bidir = bidir

        # Original VoxelMorph U-Net (from legacy/pytorch/model.py)
        self.unet_model = unet_core(dim, enc_nf, dec_nf, full_size=True)

        # Flow convolution: predict velocity field
        conv_fn = getattr(nn, 'Conv%dd' % dim)
        self.flow = conv_fn(dec_nf[-1], dim, kernel_size=3, padding=1)

        # Initialize flow weights small (same as original cvpr2018_net)
        nd = Normal(0, 1e-5)
        self.flow.weight = nn.Parameter(nd.sample(self.flow.weight.shape))
        self.flow.bias = nn.Parameter(torch.zeros(self.flow.bias.shape))

        # Scaling and squaring integration
        self.integrate = VecInt(vol_size, int_steps)

        # Spatial transformer for warping
        self.spatial_transform = SpatialTransformer(vol_size)


    def forward(self, src, tgt):
        x = torch.cat([src, tgt], dim=1)
        x = self.unet_model(x)
        vel = self.flow(x)

        # Integrate velocity field to get displacement
        disp = self.integrate(vel)

        y = self.spatial_transform(src, disp)

        if self.bidir:
            disp_neg = self.integrate(-vel)
            y_tgt = self.spatial_transform(tgt, disp_neg)
            return y, disp, y_tgt, disp_neg

        return y, disp


class VxmDenseHalf(nn.Module):
    """
    VoxelMorph wrapped as cascade-compatible L2 backbone.

    Matches CTCF_DCA_CoreHalf forward interface so it can be plugged
    directly into CTCF_CascadeA as a lightweight L2 replacement.

    Flow composition when init_flow is provided:
      mov_warped = ST(mov, init_flow)
      _, disp    = VxmDense(mov_warped, fix)     # residual disp
      total_flow = disp + ST(init_flow, disp)     # proper composition
    """
    def __init__(self, vol_size, enc_nf, dec_nf, int_steps=7):
        super().__init__()
        self.vol_size = tuple(vol_size)
        self.vxm = VxmDense(vol_size, enc_nf, dec_nf, int_steps=int_steps, bidir=False)
        # Reuse VxmDense's ST for flow composition (same grid)
        self.spatial_transform = self.vxm.spatial_transform


    def forward(
        self,
        mov_half: torch.Tensor,
        fix_half: torch.Tensor,
        init_flow_half=None,
        return_all_flows: bool = False,
        l1_feat=None,
        return_features: bool = False,
    ):
        # Warp mov with L1 coarse flow if provided
        if init_flow_half is not None: mov_warped = self.spatial_transform(mov_half, init_flow_half)
        else: mov_warped = mov_half

        # VxmDense predicts residual displacement (mov_warped -> fix)
        _, disp = self.vxm(mov_warped, fix_half)

        # Compose flows: total(x) = disp(x) + init(x + disp(x))
        if init_flow_half is not None: flow_total = disp + self.spatial_transform(init_flow_half, disp)
        else: flow_total = disp

        def_half = self.spatial_transform(mov_half, flow_total)
        out = (def_half, flow_total)
        if return_features:
            # VxmDense has no decoder features to share with L3;
            # return zeros with 16 channels (matches reg_head_chan convention)
            feat = torch.zeros(
                mov_half.shape[0], 16, *self.vol_size,
                device=mov_half.device, dtype=mov_half.dtype,
            )
            return out + (feat,)
        return out
