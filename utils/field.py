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


def fold_percent_from_flow(flow: torch.Tensor, mask: torch.Tensor = None, crop: int = 0) -> float:
    """Compute folding ratio in percent (detJ <= 0)."""
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
