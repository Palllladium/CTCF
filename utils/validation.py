from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional

import torch

from .common import AverageMeter
from .field import fold_percent_from_flow


@dataclass
class ValResult:
    """Validation outputs for one epoch."""
    dsc: float
    fold_percent: float
    last_vis: Dict[str, Any]


@torch.no_grad()
def validate(
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
    max_batches: Optional[int] = None,
) -> ValResult:
    """Run validation loop and return aggregated Dice/Fold with last visualization tensors."""
    model.eval()
    reg_nearest = None
    reg_bilin = None
    dsc_meter = AverageMeter()
    fold_meter = AverageMeter()
    last_vis: Dict[str, Any] = {}
    max_batches = None if max_batches is None else int(max_batches)
    expected_batches = None
    
    if mk_grid_img_fn is not None:
        try:
            expected_batches = len(val_loader)
            if max_batches is not None and max_batches > 0:
                expected_batches = min(expected_batches, max_batches)
        except TypeError:
            expected_batches = None

    for bidx, batch in enumerate(val_loader):
        if max_batches is not None and max_batches > 0 and bidx >= max_batches:
            break
        
        x, y, x_seg, y_seg = [t.to(device, non_blocking=True) for t in batch]
        vol_shape = tuple(x.shape[2:])
        
        if reg_nearest is None:
            reg_nearest = register_model_cls(vol_shape, mode="nearest").to(device)
            reg_bilin = register_model_cls(vol_shape, mode="bilinear").to(device)
        
        flow = forward_flow_fn(x, y)
        def_seg = reg_nearest((x_seg.float(), flow.float()))
        dsc = dice_fn(def_seg.long(), y_seg.long())
        dsc_meter.update(float(dsc), x.size(0))
        fold = fold_percent_from_flow(flow)
        fold_meter.update(float(fold), x.size(0))
        
        def_grid = None
        make_grid = mk_grid_img_fn is not None and (expected_batches is None or bidx == expected_batches - 1)
        
        if make_grid:
            grid_img = mk_grid_img_fn(flow, grid_step=grid_step, line_thickness=line_thickness)
            def_grid = reg_bilin((grid_img.float(), flow.float()))
        
        last_vis = {
            "x_seg": x_seg,
            "y_seg": y_seg,
            "def_seg": def_seg,
            "def_grid": def_grid,
            "flow": flow,
        }
    return ValResult(dsc=dsc_meter.avg, fold_percent=fold_meter.avg, last_vis=last_vis)
