from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def _crop_spatial(t: torch.Tensor, n: int) -> torch.Tensor:
    """Crop n voxels from each spatial side for [B,C,D,H,W] tensors."""
    if n <= 0:
        return t
    if min(t.shape[-3:]) <= 2 * n:
        return t
    return t[..., n:-n, n:-n, n:-n]


def _warp(tensor: torch.Tensor, flow: torch.Tensor, mode: str = "bilinear") -> torch.Tensor:
    """Warp tensor by dense voxel-unit flow on a unit-spaced grid."""
    _, _, d, h, w = tensor.shape
    device = tensor.device
    zz = torch.arange(end=d, device=device)
    yy = torch.arange(end=h, device=device)
    xx = torch.arange(end=w, device=device)
    grid = torch.stack(torch.meshgrid(zz, yy, xx, indexing="ij"), dim=0).float().unsqueeze(0)
    new_locs = grid + flow
    new_locs[:, 0] = 2.0 * (new_locs[:, 0] / (d - 1) - 0.5)
    new_locs[:, 1] = 2.0 * (new_locs[:, 1] / (h - 1) - 0.5)
    new_locs[:, 2] = 2.0 * (new_locs[:, 2] / (w - 1) - 0.5)
    grid_sample_grid = new_locs.permute(0, 2, 3, 4, 1)[..., [2, 1, 0]]
    # Cascade-composition convention: align_corners=True here, deliberately unlike
    # SpatialTransformer's align_corners=False. Trained checkpoints depend on both — do not unify.
    return F.grid_sample(tensor, grid_sample_grid, mode=mode, align_corners=True)


def compose_flows(flow_ab: torch.Tensor, flow_bc: torch.Tensor, mode: str = "bilinear") -> torch.Tensor:
    """Compose flows A->B and B->C into A->C in voxel units."""
    return flow_ab + _warp(flow_bc, flow_ab, mode=mode)


def integrate_svf(vel: torch.Tensor, st, steps: int = 7) -> torch.Tensor:
    """Integrate a stationary velocity field into a displacement via scaling-and-squaring."""
    disp = vel * (1.0 / (2**steps))
    for _ in range(steps):
        disp = disp + st(disp, disp)
    return disp


def jacobian_det(flow: torch.Tensor) -> torch.Tensor:
    """Jacobian determinant map for a 3D displacement field [B,3,D,H,W]."""
    dz = flow[:, :, 2:, :, :] - flow[:, :, :-2, :, :]
    dy = flow[:, :, :, 2:, :] - flow[:, :, :, :-2, :]
    dx = flow[:, :, :, :, 2:] - flow[:, :, :, :, :-2]
    dz = F.pad(dz, pad=(0, 0, 0, 0, 1, 1)) * 0.5
    dy = F.pad(dy, pad=(0, 0, 1, 1, 0, 0)) * 0.5
    dx = F.pad(dx, pad=(1, 1, 0, 0, 0, 0)) * 0.5

    fz_z, fz_y, fz_x = dz[:, 0], dy[:, 0], dx[:, 0]
    fy_z, fy_y, fy_x = dz[:, 1], dy[:, 1], dx[:, 1]
    fx_z, fx_y, fx_x = dz[:, 2], dy[:, 2], dx[:, 2]

    j00 = 1.0 + fz_z
    j01 = fz_y
    j02 = fz_x
    j10 = fy_z
    j11 = 1.0 + fy_y
    j12 = fy_x
    j20 = fx_z
    j21 = fx_y
    j22 = 1.0 + fx_x

    det = j00 * (j11 * j22 - j12 * j21) - j01 * (j10 * j22 - j12 * j20) + j02 * (j10 * j21 - j11 * j20)
    return det.unsqueeze(1)


def _neg_jac_penalty_from_det(
    det: torch.Tensor,
    mask: torch.Tensor | None = None,
    crop: int = 1,
    eps: float = 0.0,
) -> torch.Tensor:
    pen = torch.relu(-_crop_spatial(det, crop) + eps)

    if mask is None:
        return pen.mean()

    if mask.dim() == 4:
        mask = mask.unsqueeze(1)
    m = (mask > 0).to(pen.dtype)
    m = _crop_spatial(m, crop)
    denom = torch.clamp(m.sum(), min=1.0)
    return (pen * m).sum() / denom


def neg_jacobian_penalty(
    flow: torch.Tensor,
    mask: torch.Tensor | None = None,
    crop: int = 1,
    eps: float = 0.0,
) -> torch.Tensor:
    """Mean penalty over non-positive Jacobian determinant voxels.
    `eps` > 0 widens the band to detJ < eps; 0.0 is the form every trained checkpoint depends on.
    """
    return _neg_jac_penalty_from_det(jacobian_det(flow), mask, crop, eps)


_AXIS_MODES = (
    ("+", "+", "+"), ("+", "+", "-"), ("+", "-", "+"), ("+", "-", "-"),
    ("-", "+", "+"), ("-", "+", "-"), ("-", "-", "+"), ("-", "-", "-"),
)  # fmt: skip


def _one_sided_diff(t: torch.Tensor, axis: int, mode: str) -> torch.Tensor:
    """Forward ('+') or backward ('-') difference of [3,D,H,W] along a spatial axis, edge-clamped."""
    dim = axis + 1
    n = t.shape[dim]
    idx = torch.arange(n, device=t.device)
    if mode == "+":
        return t.index_select(dim, torch.clamp(idx + 1, max=n - 1)) - t
    return t - t.index_select(dim, torch.clamp(idx - 1, min=0))


def digital_fold_percent(flow: torch.Tensor) -> torch.Tensor:
    """Percent of voxels where any of the 8 one-sided Jacobian determinants is non-positive.
    Stricter than the central-difference `jacobian_det`; the two are not interchangeable.
    """
    disp = flow[0]
    d, h, w = disp.shape[1:]
    zz, yy, xx = torch.meshgrid(
        torch.arange(d, device=flow.device),
        torch.arange(h, device=flow.device),
        torch.arange(w, device=flow.device),
        indexing="ij",
    )
    trans = disp + torch.stack([zz, yy, xx], dim=0).to(disp.dtype)

    all_pos = None
    for mx, my, mz in _AXIS_MODES:
        gx = _one_sided_diff(trans, 0, mx)
        gy = _one_sided_diff(trans, 1, my)
        gz = _one_sided_diff(trans, 2, mz)
        det = (
            gx[0] * (gy[1] * gz[2] - gy[2] * gz[1])
            - gx[1] * (gy[0] * gz[2] - gy[2] * gz[0])
            + gx[2] * (gy[0] * gz[1] - gy[1] * gz[0])
        )[1:-1, 1:-1, 1:-1]
        pos = det > 0.0
        all_pos = pos if all_pos is None else (all_pos & pos)

    return (~all_pos).to(flow.dtype).mean() * 100.0


def jacobian_penalty_and_folds(
    flow: torch.Tensor,
    mask: torch.Tensor | None = None,
    crop: int = 1,
    eps: float = 0.0,
    strict: bool = True,
) -> tuple[torch.Tensor, float]:
    """Fold penalty on the central-difference detJ, plus a fold percentage, from one pass.
    `strict` counts the percentage by `digital_fold_percent` instead of that same detJ.
    """
    det = jacobian_det(flow)
    pen = _neg_jac_penalty_from_det(det, mask, crop, eps)
    with torch.no_grad():
        if strict:
            folds = float(digital_fold_percent(flow).item())
        else:
            folds = float((_crop_spatial(det, crop) <= 0.0).to(det.dtype).mean().item() * 100.0)
    return pen, folds


def jacobian_nonpositive_percent(
    flow: torch.Tensor,
    mask: torch.Tensor | None = None,
    crop: int = 0,
) -> float:
    """Non-positive Jacobian ratio (detJ <= 0) in percent."""
    det = _crop_spatial(jacobian_det(flow.float()), crop)
    neg = (det <= 0.0).float()

    if mask is None:
        return float(neg.mean().item() * 100.0)

    if mask.dim() == 4:
        mask = mask.unsqueeze(1)
    m = (mask > 0).to(neg.dtype)
    m = _crop_spatial(m, crop)
    denom = float(torch.clamp(m.sum(), min=1.0).item())
    num = float((neg * m).sum().item())
    return num / denom * 100.0


def logdet_std_from_flow(flow: torch.Tensor, eps: float = 1e-9) -> float:
    """std(log(detJ + 3)) — matches the UTSRMorph reporting convention."""
    det = torch.clamp(jacobian_det(flow.float()) + 3.0, min=eps, max=1e9)
    return float(torch.std(torch.log(det)).item())


def digital_jacobian_metrics(flow: torch.Tensor, mask: torch.Tensor) -> tuple[float, float]:
    """Digital Jacobian metrics (%|J|<=0 and %NDV) from displacement and brain mask."""
    if flow.dim() != 5 or flow.shape[0] != 1 or flow.shape[1] != 3:
        raise ValueError(f"Expected flow shape [1,3,D,H,W], got {tuple(flow.shape)}.")
    if mask is None:
        raise ValueError("digital_jacobian_metrics requires x_seg mask.")

    if mask.dim() == 5:
        mask_np = mask.detach().cpu().numpy()[0, 0]
    elif mask.dim() == 4:
        mask_np = mask.detach().cpu().numpy()[0]
    else:
        raise ValueError(f"Expected mask shape [1,1,D,H,W] or [1,D,H,W], got {tuple(mask.shape)}.")

    disp = flow.detach().float().cpu().numpy()[0]
    d, h, w = disp.shape[1:]
    zz, yy, xx = np.meshgrid(np.arange(d), np.arange(h), np.arange(w), indexing="ij")
    trans = disp + np.stack([zz, yy, xx], axis=0).astype(np.float32)

    def _det_from_axis_modes(mx: str, my: str, mz: str) -> np.ndarray:
        def fd(arr: np.ndarray, axis: int, mode: str) -> np.ndarray:
            n = arr.shape[axis]
            idx = np.arange(n)
            if mode == "+":
                return np.take(arr, np.clip(idx + 1, 0, n - 1), axis=axis) - arr
            if mode == "-":
                return arr - np.take(arr, np.clip(idx - 1, 0, n - 1), axis=axis)
            return 0.5 * (
                np.take(arr, np.clip(idx + 1, 0, n - 1), axis=axis)
                - np.take(arr, np.clip(idx - 1, 0, n - 1), axis=axis)
            )

        dx0, dx1, dx2 = fd(trans[0], 0, mx), fd(trans[1], 0, mx), fd(trans[2], 0, mx)
        dy0, dy1, dy2 = fd(trans[0], 1, my), fd(trans[1], 1, my), fd(trans[2], 1, my)
        dz0, dz1, dz2 = fd(trans[0], 2, mz), fd(trans[1], 2, mz), fd(trans[2], 2, mz)
        det = dx0 * (dy1 * dz2 - dy2 * dz1) - dx1 * (dy0 * dz2 - dy2 * dz0) + dx2 * (dy0 * dz1 - dy1 * dz0)
        return det[1:-1, 1:-1, 1:-1]

    def _corr3d_nearest(arr: np.ndarray, kernel: np.ndarray) -> np.ndarray:
        kz, ky, kx = kernel.shape
        pz, py, px = kz // 2, ky // 2, kx // 2
        pad = np.pad(arr, ((pz, pz), (py, py), (px, px)), mode="edge")
        out = np.zeros_like(arr, dtype=np.float32)
        d0, h0, w0 = arr.shape
        for iz in range(kz):
            for iy in range(ky):
                for ix in range(kx):
                    w = float(kernel[iz, iy, ix])
                    if w == 0.0:
                        continue
                    out += w * pad[iz : iz + d0, iy : iy + h0, ix : ix + w0]
        return out

    def _det_from_kernels(kx: np.ndarray, ky: np.ndarray, kz: np.ndarray) -> np.ndarray:
        gradx = np.stack([_corr3d_nearest(trans[c], kx) for c in range(3)], axis=0)
        grady = np.stack([_corr3d_nearest(trans[c], ky) for c in range(3)], axis=0)
        gradz = np.stack([_corr3d_nearest(trans[c], kz) for c in range(3)], axis=0)
        det = (
            gradx[0] * (grady[1] * gradz[2] - grady[2] * gradz[1])
            - gradx[1] * (grady[0] * gradz[2] - grady[2] * gradz[0])
            + gradx[2] * (grady[0] * gradz[1] - grady[1] * gradz[0])
        )
        return det[1:-1, 1:-1, 1:-1]

    det_pm = []
    for mx, my, mz in (
        ("+", "+", "+"),
        ("+", "+", "-"),
        ("+", "-", "+"),
        ("+", "-", "-"),
        ("-", "+", "+"),
        ("-", "+", "-"),
        ("-", "-", "+"),
        ("-", "-", "-"),
    ):
        det_pm.append(_det_from_axis_modes(mx, my, mz))

    all_pos = np.ones_like(det_pm[0], dtype=np.bool_)
    for det in det_pm:
        all_pos &= det > 0.0
    j_leq0_percent = float((~all_pos).sum() / all_pos.size * 100.0)

    k1 = np.array([[1, 0, 0], [0, -1, 0], [0, 0, 0]], dtype=np.float32)
    k2 = np.array([[0, 0, 0], [0, -1, 0], [0, 0, 1]], dtype=np.float32)
    jstar1 = _det_from_kernels(k1.reshape(3, 3, 1), k1.reshape(3, 1, 3), k1.reshape(1, 3, 3))
    jstar2 = _det_from_kernels(k2.reshape(3, 3, 1), k2.reshape(1, 3, 3), k2.reshape(3, 1, 3))

    brain = (mask_np[1:-1, 1:-1, 1:-1] > 0).astype(np.float32)
    denom = float(brain.sum())
    if denom <= 0.0:
        return j_leq0_percent, 0.0

    ndv = 0.0
    for det in (*det_pm, jstar1, jstar2):
        ndv += float((-0.5 * np.minimum(det, 0.0) * brain / 6.0).sum())
    ndv_percent = ndv / denom * 100.0
    return j_leq0_percent, float(ndv_percent)
