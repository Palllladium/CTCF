import torch
from utils import field


def icon_loss(flow_ab: torch.Tensor, flow_ba: torch.Tensor) -> torch.Tensor:
    """
    ICON / inverse-consistency.
    Penalize composition phi_ab ∘ phi_ba deviating from identity.
    """
    phi_ab_ba = field.compose_flows(flow_ab, flow_ba, mode='bilinear')
    phi_ba_ab = field.compose_flows(flow_ba, flow_ab, mode='bilinear')
    return phi_ab_ba.abs().mean() + phi_ba_ab.abs().mean()


def cycle_image_loss(model,
                     x: torch.Tensor, y: torch.Tensor,
                     x_warp: torch.Tensor, y_warp: torch.Tensor,
                     flow_xy: torch.Tensor, flow_yx: torch.Tensor) -> torch.Tensor:
    """
    Cycle-consistency in image space:
    x -> y -> x and y -> x -> y.
    """
    # Prefer spatial_trans_full if exists, else spatial_trans
    if hasattr(model, 'spatial_trans_down'):
        warp_fn = model.spatial_trans_down
    elif hasattr(model, 'spatial_trans_full'):
        warp_fn = model.spatial_trans_full
    else:
        warp_fn = model.spatial_trans

    x_cycle = warp_fn(x_warp, flow_yx)
    y_cycle = warp_fn(y_warp, flow_xy)
    return (x_cycle - x).abs().mean() + (y_cycle - y).abs().mean()


def percent_nonpositive_jacobian(flow: torch.Tensor) -> torch.Tensor:
    """
    Metric: percentage of voxels with detJ <= 0 (foldings).
    Returns a scalar tensor in [0,100].
    """
    detJ = field.jacobian_det(flow)  # [B,1,D,H,W]
    return (detJ <= 0.0).float().mean() * 100.0