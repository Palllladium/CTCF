import torch
import torch.nn as nn
import torch.nn.functional as F

from . import field


class Grad3d(nn.Module):
    def __init__(self, penalty="l1", loss_mult=None):
        super().__init__()
        self.penalty = penalty
        self.loss_mult = loss_mult


    def forward(self, y_pred, _=None):
        dy = torch.abs(y_pred[:, :, 1:, :, :] - y_pred[:, :, :-1, :, :])
        dx = torch.abs(y_pred[:, :, :, 1:, :] - y_pred[:, :, :, :-1, :])
        dz = torch.abs(y_pred[:, :, :, :, 1:] - y_pred[:, :, :, :, :-1])
        if self.penalty == "l2":
            dy, dx, dz = dy * dy, dx * dx, dz * dz
        out = (torch.mean(dx) + torch.mean(dy) + torch.mean(dz)) / 3.0
        return out * self.loss_mult if self.loss_mult is not None else out


class NCCVxm(nn.Module):
    def __init__(self, win=None, eps: float = 1e-5):
        super().__init__()
        self.win = win
        self.eps = float(eps)


    def forward(self, y_true, y_pred):
        ii, ji = y_true, y_pred
        ndims = len(ii.shape) - 2
        assert ndims in (1, 2, 3)
        win = [9] * ndims if self.win is None else list(self.win)
        pad = win[0] // 2
        if ndims == 1:
            stride, padding = (1,), (pad,)
        elif ndims == 2:
            stride, padding = (1, 1), (pad, pad)
        else:
            stride, padding = (1, 1, 1), (pad, pad, pad)
        conv = getattr(F, f"conv{ndims}d")
        filt = torch.ones((1, 1, *win), device=ii.device, dtype=ii.dtype)
        i2, j2, ij = ii * ii, ji * ji, ii * ji
        i_sum = conv(ii, filt, stride=stride, padding=padding)
        j_sum = conv(ji, filt, stride=stride, padding=padding)
        i2_sum = conv(i2, filt, stride=stride, padding=padding)
        j2_sum = conv(j2, filt, stride=stride, padding=padding)
        ij_sum = conv(ij, filt, stride=stride, padding=padding)
        win_size = float(torch.tensor(win).prod().item())
        ui, uj = i_sum / win_size, j_sum / win_size
        cross = ij_sum - uj * i_sum - ui * j_sum + ui * uj * win_size
        i_var = torch.clamp(i2_sum - 2 * ui * i_sum + ui * ui * win_size, min=self.eps)
        j_var = torch.clamp(j2_sum - 2 * uj * j_sum + uj * uj * win_size, min=self.eps)
        return -torch.mean((cross * cross) / (i_var * j_var))


class DareDiffusion(nn.Module):
    """
    DARE-minimal: spatially-adaptive diffusion regularization.

    alpha(x) = 1 + beta * exp(-||grad(flow(x))||)
    Smooth regions (low gradient) get MORE regularization,
    boundaries (high gradient) get LESS.

    Reference: DARE (arXiv 2510.19353, Oct 2025).
    """
    def __init__(self, beta: float = 1.0, penalty: str = "l2"):
        super().__init__()
        self.beta = float(beta)
        self.penalty = penalty

    def forward(self, flow: torch.Tensor) -> torch.Tensor:
        dy = flow[:, :, 1:, :, :] - flow[:, :, :-1, :, :]
        dx = flow[:, :, :, 1:, :] - flow[:, :, :, :-1, :]
        dz = flow[:, :, :, :, 1:] - flow[:, :, :, :, :-1]

        if self.penalty == "l2":
            dy2, dx2, dz2 = dy * dy, dx * dx, dz * dz
        else:
            dy2, dx2, dz2 = dy.abs(), dx.abs(), dz.abs()

        # Gradient magnitude at each voxel (approximate, average over axes)
        # Use central part where all three diffs are defined
        d, h, w = flow.shape[2] - 1, flow.shape[3] - 1, flow.shape[4] - 1
        grad_mag = (
            dy2[:, :, :, :h, :w].sum(dim=1, keepdim=True) +
            dx2[:, :, :d, :, :w].sum(dim=1, keepdim=True) +
            dz2[:, :, :d, :h, :].sum(dim=1, keepdim=True)
        ) / 3.0  # (B, 1, d, h, w)

        alpha = 1.0 + self.beta * torch.exp(-grad_mag)

        # Weighted per-voxel diffusion (central region)
        diff = (dy2[:, :, :, :h, :w] + dx2[:, :, :d, :, :w] + dz2[:, :, :d, :h, :]) / 3.0
        return (alpha * diff).mean()


def elastic_loss(flow: torch.Tensor, mu: float = 1.0, lam: float = 1.0) -> torch.Tensor:
    """
    Navier-Cauchy equilibrium residual (ElasticMorph, CMPB 2025).

    R = mu * laplacian(u) + (mu + lam) * grad(div(u))
    Loss = mean(||R||^2)

    Principled elastic regularizer with theoretical upper bound on folds.
    """
    # Central differences for first derivatives
    # du_i / dx_j using central difference (pad to keep size)
    def cd(t, dim):
        # Central difference along spatial dim (0=D, 1=H, 2=W) for [B,C,D,H,W]
        sd = dim + 2  # shift to spatial axis
        slc_p = [slice(None)] * 5
        slc_m = [slice(None)] * 5
        slc_p[sd] = slice(2, None)
        slc_m[sd] = slice(None, -2)
        interior = 0.5 * (t[tuple(slc_p)] - t[tuple(slc_m)])
        # Pad back to original size
        pad_spec = [0] * 6
        pad_spec[2 * (4 - sd)] = 1  # before
        pad_spec[2 * (4 - sd) + 1] = 1  # after
        return F.pad(interior, pad_spec, mode="replicate")

    # Divergence: div(u) = du0/dx0 + du1/dx1 + du2/dx2
    du0_d0 = cd(flow[:, 0:1], 0)
    du1_d1 = cd(flow[:, 1:2], 1)
    du2_d2 = cd(flow[:, 2:3], 2)
    div_u = du0_d0 + du1_d1 + du2_d2  # (B, 1, D, H, W)

    # grad(div(u))
    grad_div = torch.cat([cd(div_u, 0), cd(div_u, 1), cd(div_u, 2)], dim=1)  # (B, 3, D, H, W)

    # Laplacian per component: lap(u_i) = d²u_i/dx0² + d²u_i/dx1² + d²u_i/dx2²
    def lap_component(u_i):
        # Second derivative via central difference: f''(x) ≈ f(x+1) - 2f(x) + f(x-1)
        lap = torch.zeros_like(u_i)
        for sd in (2, 3, 4):
            slc_p = [slice(None)] * 5
            slc_m = [slice(None)] * 5
            slc_c = [slice(None)] * 5
            slc_p[sd] = slice(2, None)
            slc_m[sd] = slice(None, -2)
            slc_c[sd] = slice(1, -1)
            interior = u_i[tuple(slc_p)] + u_i[tuple(slc_m)] - 2.0 * u_i[tuple(slc_c)]
            pad_spec = [0] * 6
            pad_spec[2 * (4 - sd)] = 1
            pad_spec[2 * (4 - sd) + 1] = 1
            lap = lap + F.pad(interior, pad_spec, mode="replicate")
        return lap

    lap_u = torch.cat([lap_component(flow[:, i:i+1]) for i in range(3)], dim=1)  # (B, 3, D, H, W)

    # R = mu * lap(u) + (mu + lam) * grad(div(u))
    R = mu * lap_u + (mu + lam) * grad_div
    return (R * R).mean()


def icon_loss(flow_ab: torch.Tensor, flow_ba: torch.Tensor, mode: str = "l1") -> torch.Tensor:
    phi_ab_ba = field.compose_flows(flow_ab, flow_ba, mode="bilinear")
    phi_ba_ab = field.compose_flows(flow_ba, flow_ab, mode="bilinear")
    if mode == "l2":
        return (phi_ab_ba ** 2).mean() + (phi_ba_ab ** 2).mean()
    return phi_ab_ba.abs().mean() + phi_ba_ab.abs().mean()
