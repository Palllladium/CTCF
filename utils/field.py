import numpy as np
import torch
import torch.nn.functional as F


def _crop_spatial(t: torch.Tensor, n: int) -> torch.Tensor:
    """Crop n voxels from each spatial side for [B,C,D,H,W] tensors."""
    n = int(n)
    if n <= 0:
        return t
    if min(t.shape[-3:]) <= 2 * n:
        return t
    return t[..., n:-n, n:-n, n:-n]


def _warp(tensor: torch.Tensor, flow: torch.Tensor, mode: str = "bilinear") -> torch.Tensor:
    """Warp tensor by dense flow on voxel grid."""
    _, _, d, h, w = tensor.shape
    device = tensor.device
    zz = torch.arange(d, device=device)
    yy = torch.arange(h, device=device)
    xx = torch.arange(w, device=device)
    grid = torch.stack(torch.meshgrid(zz, yy, xx, indexing="ij"), dim=0).float().unsqueeze(0)
    new_locs = grid + flow
    new_locs[:, 0] = 2.0 * (new_locs[:, 0] / (d - 1) - 0.5)
    new_locs[:, 1] = 2.0 * (new_locs[:, 1] / (h - 1) - 0.5)
    new_locs[:, 2] = 2.0 * (new_locs[:, 2] / (w - 1) - 0.5)
    grid_sample_grid = new_locs.permute(0, 2, 3, 4, 1)[..., [2, 1, 0]]
    return F.grid_sample(tensor, grid_sample_grid, mode=mode, align_corners=False)


def compose_flows(flow_ab: torch.Tensor, flow_bc: torch.Tensor, mode: str = "bilinear") -> torch.Tensor:
    """Compose two flows: A->B and B->C into A->C."""
    return flow_ab + _warp(flow_bc, flow_ab, mode=mode)


def jacobian_det(flow: torch.Tensor) -> torch.Tensor:
    """Compute Jacobian determinant map for 3D displacement field [B,3,D,H,W]."""
    dz = flow[:, :, 2:, :, :] - flow[:, :, :-2, :, :]
    dy = flow[:, :, :, 2:, :] - flow[:, :, :, :-2, :]
    dx = flow[:, :, :, :, 2:] - flow[:, :, :, :, :-2]
    dz = F.pad(dz, (0, 0, 0, 0, 1, 1)) * 0.5
    dy = F.pad(dy, (0, 0, 1, 1, 0, 0)) * 0.5
    dx = F.pad(dx, (1, 1, 0, 0, 0, 0)) * 0.5
    fz_z, fz_y, fz_x = dz[:, 0], dy[:, 0], dx[:, 0]
    fy_z, fy_y, fy_x = dz[:, 1], dy[:, 1], dx[:, 1]
    fx_z, fx_y, fx_x = dz[:, 2], dy[:, 2], dx[:, 2]
    J00 = 1.0 + fz_z
    J01 = fz_y
    J02 = fz_x
    J10 = fy_z
    J11 = 1.0 + fy_y
    J12 = fy_x
    J20 = fx_z
    J21 = fx_y
    J22 = 1.0 + fx_x
    det = (
        J00 * (J11 * J22 - J12 * J21)
        - J01 * (J10 * J22 - J12 * J20)
        + J02 * (J10 * J21 - J11 * J20)
    )
    return det.unsqueeze(1)


def neg_jacobian_penalty(flow: torch.Tensor, mask: torch.Tensor = None, crop: int = 1) -> torch.Tensor:
    """Mean penalty over non-positive Jacobian determinant voxels."""
    pen = torch.relu(-_crop_spatial(jacobian_det(flow), int(crop)))

    if mask is None:
        return pen.mean()

    if mask.dim() == 4:
        mask = mask.unsqueeze(1)
    m = (mask > 0).to(pen.dtype)
    m = _crop_spatial(m, int(crop))
    denom = torch.clamp(m.sum(), min=1.0)
    return (pen * m).sum() / denom


def jacobian_nonpositive_percent(flow: torch.Tensor, mask: torch.Tensor = None, crop: int = 0) -> float:
    """Compute non-positive Jacobian ratio in percent (detJ <= 0)."""
    det = _crop_spatial(jacobian_det(flow.float()), int(crop))
    neg = (det <= 0.0).float()

    if mask is None:
        return float(neg.mean().item() * 100.0)

    if mask.dim() == 4:
        mask = mask.unsqueeze(1)
    m = (mask > 0).to(neg.dtype)
    m = _crop_spatial(m, int(crop))
    denom = float(torch.clamp(m.sum(), min=1.0).item())
    num = float((neg * m).sum().item())
    return num / denom * 100.0


def logdet_std_from_flow(flow: torch.Tensor, eps: float = 1e-9) -> float:
    """Compute std(log(detJ)) with safe clamping."""
    det = torch.clamp(jacobian_det(flow.float()), min=float(eps), max=1e9)
    return float(torch.std(torch.log(det)).item())


def digital_jacobian_metrics(flow: torch.Tensor, mask: torch.Tensor) -> tuple[float, float]:
    """Compute digital Jacobian metrics: %|J|<=0 and %NDV from displacement + brain mask."""
    if flow.dim() != 5 or int(flow.shape[0]) != 1 or int(flow.shape[1]) != 3:
        raise ValueError(f"Expected flow shape [1,3,D,H,W], got {tuple(flow.shape)}.")
    if mask is None: raise ValueError("digital_jacobian_metrics requires x_seg mask.")

    if mask.dim() == 5: mask_np = mask.detach().cpu().numpy()[0, 0]
    elif mask.dim() == 4: mask_np = mask.detach().cpu().numpy()[0]
    else: raise ValueError(f"Expected mask shape [1,1,D,H,W] or [1,D,H,W], got {tuple(mask.shape)}.")

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
                    out += w * pad[iz:iz + d0, iy:iy + h0, ix:ix + w0]
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
        ("+", "+", "+"), ("+", "+", "-"), ("+", "-", "+"), ("+", "-", "-"),
        ("-", "+", "+"), ("-", "+", "-"), ("-", "-", "+"), ("-", "-", "-"),
    ):
        det_pm.append(_det_from_axis_modes(mx, my, mz))

    all_pos = np.ones_like(det_pm[0], dtype=np.bool_)
    for det in det_pm:
        all_pos &= (det > 0.0)
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
