from __future__ import annotations

import numpy as np
import scipy.linalg
import torch
import torch.nn.functional as F

from experiments.eval_cst import calibrate_weight
from models.CST.adaptation import (
    adjust_target,
    compose,
    equivariance_nce,
    equivariance_points,
    identity_grid,
    make_pairs,
    random_affine_pair,
    teacher_fields,
    to_canonical,
    voxel_error,
    warp,
    warp_affine,
)
from models.CST.core import coupled_convex, make_feature_net
from utils.field import _warp


def _smooth(shape, amplitude, seed, channels=3):
    low = torch.randn((1, channels, 4, 4, 4), generator=torch.Generator().manual_seed(seed), dtype=torch.float64)
    return amplitude * F.interpolate(low, size=shape, mode="trilinear", align_corners=True) / low.abs().max()


def _inner(t, margin=4):
    return t[(slice(None), slice(None), *(slice(margin, -margin),) * 3)]


def test_compose_follows_sequential_warping_and_the_cst_order_does_not():
    shape = (20, 24, 28)
    u0, u1 = _smooth(shape, 0.08, 0), _smooth(shape, 0.08, 1)
    coords = identity_grid(shape, u0).permute(0, 4, 1, 2, 3)
    sequential = warp(warp(coords, u0), u1)
    assert torch.allclose(_inner(warp(coords, compose(u0, u1))), _inner(sequential), atol=1e-6)
    cst_order = u0 + warp(u1, u0)
    assert (_inner(warp(coords, cst_order)) - _inner(sequential)).abs().max() > 1e-3


def test_canonical_flow_reproduces_the_cst_warp():
    shape = (20, 24, 28)
    volume = torch.rand((1, 2, *shape), generator=torch.Generator().manual_seed(3), dtype=torch.float64)
    offsets = _smooth(shape, 0.1, 2)
    assert torch.allclose(_warp(volume, to_canonical(offsets)), warp(volume, offsets), atol=1e-8)


def test_solver_recovers_a_one_cell_shift_along_the_first_axis():
    torch.manual_seed(0)
    features = F.interpolate(torch.randn((1, 16, 8, 8, 8)), size=(16, 16, 16), mode="trilinear", align_corners=True)
    moving = torch.roll(features, shifts=1, dims=2)  # moving(h) = fixed(h - 1): fixed(h) = moving(h + 1)
    for ice in (False, True):
        flow = to_canonical(coupled_convex(features, moving, use_ice=ice, img_shape=(128, 128, 128)))
        centre = flow[0, :, 48:80, 48:80, 48:80].mean(dim=(1, 2, 3))
        # one cell of the 1/8 grid is 8 voxels; the original (n - 1)/2 normalisation returned 8 * 16/15
        assert abs(centre[0] - 8.0) < 0.05 and centre[1:].abs().max() < 0.05


def test_adjusted_target_registers_the_augmented_pair_and_the_cst_formula_does_not():
    shape = (24, 28, 32)
    x, y, z = identity_grid(shape, torch.zeros((), dtype=torch.float64)).unbind(-1)
    moving = (torch.sin(1.3 * x) + torch.cos(1.1 * y) + torch.sin(0.9 * z + 0.5)).unsqueeze(1)  # smooth: exact-ish lerp
    offsets = _smooth(shape, 0.08, 5)
    fixed = warp(moving, offsets)  # fixed(x) = moving(x + u(x))
    a1, a2 = (a.double() for a in random_affine_pair(0.05, torch.Generator().manual_seed(6)))
    fixed_aug, moving_aug = warp_affine(fixed, a1), warp_affine(moving, a2)
    ours = adjust_target(offsets, a1, a2)
    a12 = torch.from_numpy(scipy.linalg.expm(scipy.linalg.logm(a1.numpy()) - scipy.linalg.logm(a2.numpy())).real)
    grid12 = F.affine_grid(a12[:3].unsqueeze(0), (1, 1, *shape), align_corners=False)
    theirs = F.grid_sample(offsets, grid12, align_corners=False) + (grid12 - identity_grid(shape, offsets)).permute(
        0, 4, 1, 2, 3
    )
    error = {
        name: (_inner(warp(moving_aug, t), 6) - _inner(fixed_aug, 6)).abs().max()
        for name, t in (("ours", ours), ("theirs", theirs))
    }
    assert error["ours"] < 5e-3 and error["theirs"] > 10 * error["ours"]


def test_equivariance_points_are_distinct_inside_the_mask_and_off_the_border():
    mask = torch.zeros((40, 48, 56), dtype=torch.bool)
    mask[8:32, 10:38, 12:44] = True
    points = equivariance_points(mask, 1000, torch.Generator().manual_seed(0))
    assert len(torch.unique(points, dim=0)) == 1000
    assert mask[points[:, 0], points[:, 1], points[:, 2]].all() and (points >= 5).all()


def test_equivariance_nce_rewards_matching_views():
    anchor = torch.randn((64, 16), generator=torch.Generator().manual_seed(0))
    matched = equivariance_nce(anchor, anchor + 0.01 * torch.randn_like(anchor), 0.1)
    shuffled = equivariance_nce(anchor, anchor.roll(1, dims=0), 0.1)
    assert matched < 0.1 < shuffled


def test_voxel_error_scales_each_axis_by_its_full_resolution_size():
    full = (48, 56, 64)
    a = torch.zeros((1, 3, 24, 28, 32))
    b = a.clone()
    b[:, 0] = 0.1  # x offset: the last axis, 64 voxels
    error = voxel_error(a, b, full)
    assert error.numel() == 8 * 12 * 16 and torch.allclose(error, torch.full_like(error, 0.1 * 64 / 2 * 1.5))


def test_voxel_error_has_a_finite_gradient_at_a_zero_residual():
    a = torch.zeros((1, 3, 24, 28, 32), requires_grad=True)
    voxel_error(a, torch.zeros((1, 3, 24, 28, 32)), (48, 56, 64)).mean().backward()
    assert torch.isfinite(a.grad).all()


def test_make_pairs_balances_subjects_without_self_or_repeated_pairs():
    pairs = make_pairs(11, 3, seed=0)
    fixed, moving = np.bincount([p[0] for p in pairs]), np.bincount([p[1] for p in pairs])
    assert len(pairs) == 33 == len(set(pairs)) and (fixed == 3).all() and (moving == 3).all()
    assert all(i != j for i, j in pairs)


def test_teacher_runs_on_cpu():
    torch.manual_seed(0)
    net = make_feature_net()
    fixed, moving = torch.rand((1, 1, 32, 32, 32)), torch.rand((1, 1, 32, 32, 32))
    solver, refined = teacher_fields(net, fixed, moving, reg_fac=1.0, proj_seed=0, pad=0.0)
    assert solver.shape == refined.shape == (1, 3, 32, 32, 32)
    assert torch.isfinite(solver).all() and torch.isfinite(refined).all()


def test_calibration_reports_whether_the_target_sdlogj_was_reached():
    def falling(log_reg):
        return 0.2 / (1.0 + 10.0**log_reg)

    reached = calibrate_weight(falling, 0.05, 0.01, 1000.0, steps=14)
    assert reached["status"] == "reached" and abs(reached["achieved"] - 0.05) <= 0.03 * 0.05
    assert calibrate_weight(lambda log_reg: 1.0, 0.1, 0.01, 1000.0, steps=10)["status"] == "target_below_range"
    assert calibrate_weight(lambda log_reg: 0.01, 0.1, 0.01, 1000.0, steps=10)["status"] == "target_above_range"
