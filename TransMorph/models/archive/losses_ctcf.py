
import torch
import torch.nn as nn
import torch.nn.functional as F
from TransMorph.models.utils_field import neg_jacobian_penalty

class LNCCLoss(nn.Module):
    """
    Local (windowed) normalized cross-correlation in 3D.
    Inputs: x,y in [B,1,D,H,W]; window size ws: tuple or int.
    """
    def __init__(self, win=(9,9,9), eps=1e-5):
        super().__init__()
        if isinstance(win, int):
            win = (win,)*3
        self.win = win
        self.eps = eps

    def forward(self, x, y):
        B, C, D, H, W = x.shape
        pad = tuple([w//2 for w in self.win])[::-1] * 2  # (W,H,D)*2 for F.pad
        ones = torch.ones_like(x)

        def convn(z):
            k = torch.ones(1,1,*self.win, device=z.device, dtype=z.dtype)
            return F.conv3d(z, k, padding=0)

        x_pad = F.pad(x, pad, mode='replicate')
        y_pad = F.pad(y, pad, mode='replicate')
        ones_pad = F.pad(ones, pad, mode='replicate')

        N = convn(ones_pad)
        xm = convn(x_pad) / N
        ym = convn(y_pad) / N
        x2 = convn(x_pad * x_pad) / N
        y2 = convn(y_pad * y_pad) / N
        xy = convn(x_pad * y_pad) / N

        cross = xy - xm * ym
        var_x = x2 - xm * xm
        var_y = y2 - ym * ym
        ncc = (cross * cross) / (var_x * var_y + self.eps)
        return -ncc.mean()

class Grad3dLoss(nn.Module):
    """First-order gradient regularization for flow [B,3,D,H,W]."""
    def __init__(self, penalty='l2'):
        super().__init__()
        self.penalty = penalty

    def forward(self, flow):
        dx = flow[:,:,:,:,1:] - flow[:,:,:,:,:-1]
        dy = flow[:,:,:,1:,:] - flow[:,:,:,:-1,:]
        dz = flow[:,:,1:,:,:] - flow[:,:,:-1,:,:]
        if self.penalty == 'l2':
            return (dx*dx).mean() + (dy*dy).mean() + (dz*dz).mean()
        else:
            return (dx.abs().mean() + dy.abs().mean() + dz.abs().mean())

def icon_flow_loss(phi_ab, phi_ba, compose_fn):
    """
    Inverse-consistency on flows: phi_ab ∘ phi_ba ≈ 0 (identity in flow space).
    Returns L1 norm of composed flow.
    """
    comp = compose_fn(phi_ab, phi_ba)  # [B,3,D,H,W]
    return comp.abs().mean()

def cycle_image_loss(A, B, warp_fn, phi_ab, phi_ba):
    """
    Optional: cycle-consistency on images: A -> B -> A ≈ A.
    """
    A2 = warp_fn(A, phi_ab)
    A3 = warp_fn(A2, phi_ba)
    return (A3 - A).abs().mean()

def neg_jac_loss(flow, eps=0.0):
    """Wrapper for Jacobian negativity penalty."""
    return neg_jacobian_penalty(flow, eps=eps)
