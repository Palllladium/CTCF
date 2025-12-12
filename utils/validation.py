from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Dict, Optional, Any
import torch

from utils import AverageMeter, jacobian_det


@dataclass
class ValResult:
    dsc: float
    fold_percent: float
    last_vis: Dict[str, Any]


@torch.no_grad()
def validate_oasis(
    *,
    model: torch.nn.Module,
    val_loader,
    device: torch.device,
    forward_flow_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    dice_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    register_model_cls,
    mk_grid_img_fn: Optional[Callable[..., torch.Tensor]] = None,
    grid_step: int = 8,
    line_thickness: int = 1,
) -> ValResult:
    """
    Universal OASIS validation:
      - computes DSC on VOI labels using warped segmentation (nearest)
      - computes folding percent (detJ <= 0)
      - optionally returns a deformed grid for visualization

    Contract:
      forward_flow_fn(x, y) -> flow_full of shape [B,3,D,H,W] on `device`
    """
    model.eval()

    reg_nearest = None
    reg_bilin = None

    dsc_meter = AverageMeter()
    fold_meter = AverageMeter()

    last_vis: Dict[str, Any] = {}

    for batch in val_loader:
        # expected: x, y, x_seg, y_seg
        x, y, x_seg, y_seg = [t.to(device, non_blocking=True) for t in batch]
        vol_shape = tuple(x.shape[2:])  # (D,H,W) from tensor

        if reg_nearest is None:
            reg_nearest = register_model_cls(vol_shape, mode="nearest").to(device)
            reg_bilin = register_model_cls(vol_shape, mode="bilinear").to(device)

        flow = forward_flow_fn(x, y)  # [B,3,D,H,W]

        # warp seg (nearest)
        def_seg = reg_nearest([x_seg.float(), flow.float()])
        dsc = dice_fn(def_seg.long(), y_seg.long())
        dsc_meter.update(float(dsc), x.size(0))

        # fold %
        detJ = jacobian_det(flow.float())  # [B,1,D,H,W]
        fold = (detJ <= 0.0).float().mean() * 100.0
        fold_meter.update(float(fold), x.size(0))

        # optional grid for visuals (only keep last batch)
        def_grid = None
        if mk_grid_img_fn is not None:
            grid_img = mk_grid_img_fn(grid_step, line_thickness, vol_shape, device=device)
            def_grid = reg_bilin([grid_img.float(), flow.float()])

        last_vis = {
            "x_seg": x_seg,
            "y_seg": y_seg,
            "def_seg": def_seg,
            "def_grid": def_grid,
            "flow": flow,
        }

    return ValResult(
        dsc=dsc_meter.avg,
        fold_percent=fold_meter.avg,
        last_vis=last_vis,
    )