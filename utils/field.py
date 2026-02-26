import torch
import torch.nn.functional as F


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


def neg_jacobian_penalty(flow: torch.Tensor) -> torch.Tensor:
    """Mean penalty over non-positive Jacobian determinant voxels."""
    return torch.relu(-jacobian_det(flow)).mean()


def fold_percent_from_flow(flow: torch.Tensor) -> float:
    """Compute folding ratio in percent (detJ <= 0)."""
    det = jacobian_det(flow.float())
    return float((det <= 0.0).float().mean().item() * 100.0)


def logdet_std_from_flow(flow: torch.Tensor, eps: float = 1e-9) -> float:
    """Compute std(log(detJ)) with safe clamping."""
    det = torch.clamp(jacobian_det(flow.float()), min=float(eps), max=1e9)
    return float(torch.std(torch.log(det)).item())
