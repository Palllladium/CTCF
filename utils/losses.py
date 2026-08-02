from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from . import field


class Grad3d(nn.Module):
    """Spatial gradient regulariser (diffusion prior) on a 3D displacement field."""

    def __init__(self, penalty: str = "l1", loss_mult: float | None = None):
        super().__init__()
        self.penalty = penalty
        self.loss_mult = loss_mult

    def forward(self, y_pred: torch.Tensor, _: torch.Tensor | None = None) -> torch.Tensor:
        dy = torch.abs(y_pred[:, :, 1:, :, :] - y_pred[:, :, :-1, :, :])
        dx = torch.abs(y_pred[:, :, :, 1:, :] - y_pred[:, :, :, :-1, :])
        dz = torch.abs(y_pred[:, :, :, :, 1:] - y_pred[:, :, :, :, :-1])
        if self.penalty == "l2":
            dy, dx, dz = dy * dy, dx * dx, dz * dz
        out = (torch.mean(dx) + torch.mean(dy) + torch.mean(dz)) / 3.0
        if self.loss_mult is not None:
            return out * self.loss_mult
        return out


class NCCVxm(nn.Module):
    """Local normalised cross-correlation loss; squared correlation, sign-flipped."""

    def __init__(self, win=None, eps: float = 1e-5):
        super().__init__()
        self.win = win
        self.eps = eps

    def forward(
        self,
        y_true: torch.Tensor,
        y_pred: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """`mask` restricts the average to the given voxels (e.g. the brain).

        Left at None the whole volume is averaged, which is what every trained checkpoint used --
        do not switch it on in training without re-tuning w_reg, since masking rescales the
        similarity term relative to the regulariser.
        """
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
        cc = (cross * cross) / (i_var * j_var)

        if mask is None:
            return -torch.mean(cc)

        m = (mask > 0).to(cc.dtype)
        if m.dim() == cc.dim() - 1:
            m = m.unsqueeze(1)
        return -(cc * m).sum() / torch.clamp(m.sum(), min=1.0)


_MIND_OFFSETS = ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1))


def _mind_shift(x: torch.Tensor, dz: int, dy: int, dx: int) -> torch.Tensor:
    """Edge-clamped (replicate) shift x[i] -> x[clamp(i+offset)] — NON-circular, unlike torch.roll, so
    opposite faces of the volume are never treated as neighbours. Same convention as field.py."""
    out = x
    for dim, off in ((2, dz), (3, dy), (4, dx)):
        if off == 0:
            continue
        n = out.shape[dim]
        idx = torch.clamp(torch.arange(n, device=x.device) + off, 0, n - 1)
        out = out.index_select(dim, idx)
    return out


def mind_descriptor(img: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """MIND-6 descriptor (Heinrich et al., MedIA 2012, six-neighbour form): for each 6-connected
    neighbour, the 3x3x3 box patch-SSD to it, normalised by the local variance and exp'd, max-channel
    normalised — a MODALITY/intensity-INVARIANT local signature, [B,6,D,H,W]. Box patch (not the paper's
    Gaussian) and edge-clamped boundary; NOT bit-equivalent to MIND-SSC (12-edge self-similarity, deeds/
    ConvexAdam). Aligning MIND (not raw intensity) is why it can survive contrast shifts NCC drifts on."""
    dists = [
        F.avg_pool3d(
            F.pad((img - _mind_shift(img, dz, dy, dx)).pow(2), (1, 1, 1, 1, 1, 1), mode="replicate"),
            kernel_size=3,
            stride=1,
        )
        for dz, dy, dx in _MIND_OFFSETS
    ]
    dp = torch.cat(dists, dim=1)
    var = dp.mean(dim=1, keepdim=True).clamp_min(eps)
    mind = torch.exp(-dp / var)
    return mind / mind.amax(dim=1, keepdim=True).clamp_min(eps)


def mind_loss(fixed: torch.Tensor, moving: torch.Tensor) -> torch.Tensor:
    """Mean-absolute MIND distance between two volumes; minimised when their anatomy aligns, regardless of
    intensity/contrast differences (the leak-free cross-domain objective, as opposed to test labels)."""
    return (mind_descriptor(fixed) - mind_descriptor(moving)).abs().mean()


class DareDiffusion(nn.Module):
    """Spatially-adaptive diffusion regularisation: alpha(x) = 1 + beta * exp(-|grad flow|)."""

    def __init__(self, beta: float = 1.0, penalty: str = "l2"):
        super().__init__()
        self.beta = beta
        self.penalty = penalty

    def forward(self, flow: torch.Tensor) -> torch.Tensor:
        dy = flow[:, :, 1:, :, :] - flow[:, :, :-1, :, :]
        dx = flow[:, :, :, 1:, :] - flow[:, :, :, :-1, :]
        dz = flow[:, :, :, :, 1:] - flow[:, :, :, :, :-1]

        if self.penalty == "l2":
            dy2, dx2, dz2 = dy * dy, dx * dx, dz * dz
        else:
            dy2, dx2, dz2 = dy.abs(), dx.abs(), dz.abs()

        d, h, w = flow.shape[2] - 1, flow.shape[3] - 1, flow.shape[4] - 1
        grad_mag = (
            dy2[:, :, :, :h, :w].sum(dim=1, keepdim=True)
            + dx2[:, :, :d, :, :w].sum(dim=1, keepdim=True)
            + dz2[:, :, :d, :h, :].sum(dim=1, keepdim=True)
        ) / 3.0

        alpha = 1.0 + self.beta * torch.exp(-grad_mag)
        diff = (dy2[:, :, :, :h, :w] + dx2[:, :, :d, :, :w] + dz2[:, :, :d, :h, :]) / 3.0
        return (alpha * diff).mean()


def elastic_loss(flow: torch.Tensor, mu: float = 1.0, lam: float = 1.0) -> torch.Tensor:
    """Navier-Cauchy equilibrium residual; ||mu * lap(u) + (mu + lam) * grad(div(u))||^2."""

    def cd(t: torch.Tensor, dim: int) -> torch.Tensor:
        sd = dim + 2
        slc_p = [slice(None)] * 5
        slc_m = [slice(None)] * 5
        slc_p[sd] = slice(2, None)
        slc_m[sd] = slice(None, -2)
        interior = 0.5 * (t[tuple(slc_p)] - t[tuple(slc_m)])
        pad_spec = [0] * 6
        pad_spec[2 * (4 - sd)] = 1
        pad_spec[2 * (4 - sd) + 1] = 1
        return F.pad(interior, pad_spec, mode="replicate")

    du0_d0 = cd(flow[:, 0:1], 0)
    du1_d1 = cd(flow[:, 1:2], 1)
    du2_d2 = cd(flow[:, 2:3], 2)
    div_u = du0_d0 + du1_d1 + du2_d2

    grad_div = torch.cat([cd(div_u, 0), cd(div_u, 1), cd(div_u, 2)], dim=1)

    def lap_component(u_i: torch.Tensor) -> torch.Tensor:
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

    lap_u = torch.cat([lap_component(flow[:, i : i + 1]) for i in range(3)], dim=1)

    r = mu * lap_u + (mu + lam) * grad_div
    return (r * r).mean()


def icon_loss(flow_ab: torch.Tensor, flow_ba: torch.Tensor, mode: str = "l1") -> torch.Tensor:
    """Inverse-consistency loss on composed forward/backward flows."""
    phi_ab_ba = field.compose_flows(flow_ab, flow_ba, mode="bilinear")
    phi_ba_ab = field.compose_flows(flow_ba, flow_ab, mode="bilinear")
    if mode == "l2":
        return (phi_ab_ba**2).mean() + (phi_ba_ab**2).mean()
    return phi_ab_ba.abs().mean() + phi_ba_ab.abs().mean()
