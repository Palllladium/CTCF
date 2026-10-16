"""Vendored from multimodallearning/reg-cyclical-self-train @ d17802279177d7d689e52b7e625dd8689e973ab2 (MIT, see LICENSE
in this folder): the feature network, the correlation layer, the coupled convex optimiser with inverse consistency and
the Adam instance optimisation of Bigalke et al., "Unsupervised 3D registration through optimization-guided cyclical
self-training", MICCAI 2023.

Changes: tensors follow the input device instead of .cuda(); align_corners=False is spelled out where the original
relied on that default; formatting. One correction: offsets are normalized by n/2, the scale of the align_corners=False
warps that apply them. The original converts voxels with (n - 1)/2 in the solver, its inverse consistency and the Adam
refinement, which inflates displacements by n/(n - 1): 4-5% on the 1/8 grid, about 1% at half resolution.
Algorithms and constants are otherwise unchanged; displacements are normalized grid_sample offsets in (x, y, z)
channel order, sampled with align_corners=False. The training/pseudo-label stream that used to call these lives in models.CST.adaptation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def make_feature_net():
    """The CST feature extractor: strides 2-2-2, 16 output channels on the 1/8 grid (train.py)."""
    return nn.Sequential(
        nn.Conv3d(1, 32, 3, padding=1, stride=2),
        nn.BatchNorm3d(32),
        nn.ReLU(),
        nn.Conv3d(32, 64, 3, padding=1),
        nn.BatchNorm3d(64),
        nn.ReLU(),
        nn.Conv3d(64, 128, 3, padding=1, stride=2),
        nn.BatchNorm3d(128),
        nn.ReLU(),
        nn.Conv3d(128, 128, 3, padding=1),
        nn.BatchNorm3d(128),
        nn.ReLU(),
        nn.Conv3d(128, 128, 3, padding=1, stride=2),
        nn.BatchNorm3d(128),
        nn.ReLU(),
        nn.Conv3d(128, 16, 1),
    )


# correlation layer in 3D similar to PWC-Net
def correlate(mind_fix, mind_mov, disp_hw):
    B, C_mind, H, W, D = mind_fix.shape
    ssd = torch.zeros(B, (disp_hw * 2 + 1) ** 3, H, W, D, dtype=mind_fix.dtype, device=mind_fix.device)
    for b in range(B):
        mind_unfold = F.unfold(
            F.pad(mind_mov[b : b + 1].squeeze(0), (disp_hw, disp_hw, disp_hw, disp_hw, disp_hw, disp_hw)).squeeze(0),
            disp_hw * 2 + 1,
        )
        mind_unfold = mind_unfold.view(C_mind, -1, (disp_hw * 2 + 1) ** 2, W, D)

        for i in range(disp_hw * 2 + 1):
            mind_sum = (
                (mind_fix[b : b + 1].permute(1, 2, 0, 3, 4) - mind_unfold[:, i : i + H]).abs().sum(0, keepdim=True)
            )

            ssd[b, i :: (disp_hw * 2 + 1)] = F.avg_pool3d(mind_sum.transpose(2, 1), 3, stride=1, padding=1).squeeze(1)
    ssd = (
        ssd.view(B, disp_hw * 2 + 1, disp_hw * 2 + 1, disp_hw * 2 + 1, H, W, D)
        .transpose(2, 1)
        .reshape(B, (disp_hw * 2 + 1) ** 3, H, W, D)
    )

    return ssd


# solve two coupled convex optimisation problems for efficient global regularisation
def coupled_convex_optim(ssd, disp_mesh_t, delta=20):
    B, C, H_, W_, D_ = ssd.shape  # [-3:]
    disp_soft = torch.zeros(B, 3, H_, W_, D_).to(ssd.device)
    for i in range(H_):
        output = (disp_mesh_t.view(1, 3, C, 1) * torch.softmax(-delta * ssd[:, :, i].view(B, 1, C, -1), 2)).sum(2)

        disp_soft[:, :, i] = output.view(B, 3, W_, D_)

    coeffs = torch.tensor([0.003, 0.03, 0.3])
    for j in range(len(coeffs)):
        for i in range(H_):
            coupled = ssd[:, :, i, :, :] + coeffs[j] * (
                disp_mesh_t.unsqueeze(0) - disp_soft[:, :, i].view(B, 3, 1, -1)
            ).pow(2).sum(1).view(B, -1, W_, D_)
            output = (disp_mesh_t.view(1, 3, C, 1) * torch.softmax(-delta * coupled.view(B, 1, C, -1), 2)).sum(2)

            disp_soft[:, :, i] = output.view(B, 3, W_, D_)
        disp_soft = F.avg_pool3d(disp_soft.reshape(B, 3, H_, W_, D_), 3, padding=1, stride=1)
    return disp_soft


# enforce inverse consistency of forward and backward transform
def inverse_consistency(disp_field1s, disp_field2s, iter=20):
    B, C, H, W, D = disp_field1s.size()
    with torch.no_grad():
        disp_field1i = disp_field1s.clone()
        disp_field2i = disp_field2s.clone()

        identity = (
            F.affine_grid(torch.eye(3, 4).unsqueeze(0), (1, 1, H, W, D), align_corners=False)
            .permute(0, 4, 1, 2, 3)
            .to(disp_field1s.device)
            .to(disp_field1s.dtype)
        )
        for i in range(iter):
            disp_field1s = disp_field1i.clone()
            disp_field2s = disp_field2i.clone()

            disp_field1i = 0.5 * (
                disp_field1s
                - F.grid_sample(disp_field2s, (identity + disp_field1s).permute(0, 2, 3, 4, 1), align_corners=False)
            )
            disp_field2i = 0.5 * (
                disp_field2s
                - F.grid_sample(disp_field1s, (identity + disp_field2s).permute(0, 2, 3, 4, 1), align_corners=False)
            )

    return disp_field1i, disp_field2i


# the optimizer h in our framework
def coupled_convex(feat_fix, feat_mov, use_ice, img_shape):
    device = feat_fix.device
    disp_hw = 2
    ssd = correlate(5 * (feat_fix), 5 * (feat_mov), disp_hw)
    disp_mesh_t = (
        F.affine_grid(
            disp_hw * torch.eye(3, 4, device=device).unsqueeze(0),
            (1, 1, disp_hw * 2 + 1, disp_hw * 2 + 1, disp_hw * 2 + 1),
            align_corners=True,
        )
        .permute(0, 4, 1, 2, 3)
        .reshape(3, -1, 1)
    )
    H2, W2, D2 = ssd.shape[-3:]
    disp_soft0 = coupled_convex_optim(ssd, disp_mesh_t)

    if use_ice:
        ssd_ = correlate(5 * (feat_mov), 5 * (feat_fix), disp_hw)
        disp_soft0_ = coupled_convex_optim(ssd_, disp_mesh_t)
        scale = torch.tensor([H2, W2, D2], device=device).view(1, 3, 1, 1, 1) / 2
        disp_soft0 = (
            inverse_consistency((disp_soft0 / scale).flip(1), (disp_soft0_ / scale).flip(1), iter=15)[0].flip(1) * scale
        )

    scale = torch.tensor([H2, W2, D2], device=device).view(1, 3, 1, 1, 1) / 2
    disp_torch = F.avg_pool3d(
        F.avg_pool3d(
            F.interpolate((disp_soft0 / scale).flip(1), scale_factor=2, mode="trilinear"), 3, padding=1, stride=1
        ),
        3,
        padding=1,
        stride=1,
    )

    disp_soft1 = F.interpolate(
        disp_torch, scale_factor=2, mode="trilinear", align_corners=False, recompute_scale_factor=False
    )
    disp_soft2 = F.avg_pool3d(F.avg_pool3d(disp_soft1, 3, stride=1, padding=1), 3, stride=1, padding=1)
    disp_soft3 = F.interpolate(disp_soft2, size=img_shape, mode="trilinear")

    return disp_soft3


# Instance optimization with Adam: minimize joint cost function of feature similarity and regularization
def AdamReg(mind_fix, mind_mov, dense_flow, reg_fac=1.0):
    device = mind_fix.device
    if dense_flow.shape[-1] == 3:
        dense_flow = dense_flow.permute(0, 4, 1, 2, 3)

    H, W, D = dense_flow[0, 0].shape

    disp_hr = dense_flow.to(device).flip(1) * torch.tensor([H, W, D], device=device).view(1, 3, 1, 1, 1) / 2
    with torch.enable_grad():
        grid_sp = 2

        disp_lr = F.interpolate(
            disp_hr, size=(H // grid_sp, W // grid_sp, D // grid_sp), mode="trilinear", align_corners=False
        )
        net = nn.Sequential(nn.Conv3d(3, 1, (H // grid_sp, W // grid_sp, D // grid_sp), bias=False))
        net[0].weight.data[:] = disp_lr.float().cpu().data / grid_sp
        net.to(device)
        optimizer = torch.optim.Adam(net.parameters(), lr=1)
        grid0 = F.affine_grid(
            torch.eye(3, 4, device=device).unsqueeze(0),
            (1, 1, H // grid_sp, W // grid_sp, D // grid_sp),
            align_corners=False,
        )
        # run Adam optimisation with diffusion regularisation and B-spline smoothing
        lambda_weight = 0.65
        for iter in range(50):
            optimizer.zero_grad()
            disp_sample = F.avg_pool3d(
                F.avg_pool3d(F.avg_pool3d(net[0].weight, 3, stride=1, padding=1), 3, stride=1, padding=1),
                3,
                stride=1,
                padding=1,
            ).permute(0, 2, 3, 4, 1)
            reg_loss = (
                lambda_weight * ((disp_sample[0, :, 1:, :] - disp_sample[0, :, :-1, :]) ** 2).mean()
                + lambda_weight * ((disp_sample[0, 1:, :, :] - disp_sample[0, :-1, :, :]) ** 2).mean()
                + lambda_weight * ((disp_sample[0, :, :, 1:] - disp_sample[0, :, :, :-1]) ** 2).mean()
            )
            scale = torch.tensor([(H // grid_sp) / 2, (W // grid_sp) / 2, (D // grid_sp) / 2], device=device).unsqueeze(
                0
            )
            grid_disp = grid0.view(-1, 3).float() + ((disp_sample.view(-1, 3)) / scale).flip(1).float()
            patch_mov_sampled = F.grid_sample(
                mind_mov.float(),
                grid_disp.view(1, H // grid_sp, W // grid_sp, D // grid_sp, 3),
                align_corners=False,
                mode="bilinear",
            )
            sampled_cost = (patch_mov_sampled - mind_fix.float()).pow(2).mean(1) * 12
            loss = sampled_cost.mean()

            (loss + reg_fac * reg_loss).backward()
            optimizer.step()
        fitted_grid = disp_sample.permute(0, 4, 1, 2, 3).detach()
        disp_hr = F.interpolate(fitted_grid * grid_sp, size=(H, W, D), mode="trilinear", align_corners=False)

    disp_smooth = F.avg_pool3d(
        F.avg_pool3d(F.avg_pool3d(disp_hr, 3, padding=1, stride=1), 3, padding=1, stride=1), 3, padding=1, stride=1
    )

    disp_hr = torch.flip(disp_smooth / torch.tensor([H, W, D], device=device).view(1, 3, 1, 1, 1) * 2, [1])
    return disp_hr
