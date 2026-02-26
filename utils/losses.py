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


def icon_loss(flow_ab: torch.Tensor, flow_ba: torch.Tensor) -> torch.Tensor:
    phi_ab_ba = field.compose_flows(flow_ab, flow_ba, mode="bilinear")
    phi_ba_ab = field.compose_flows(flow_ba, flow_ab, mode="bilinear")
    return phi_ab_ba.abs().mean() + phi_ba_ab.abs().mean()
