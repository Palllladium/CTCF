from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional

import torch

from .common import AverageMeter
from .field import jacobian_nonpositive_percent, digital_jacobian_metrics, logdet_std_from_flow


@dataclass
class ValResult:
    """Validation outputs for one epoch."""
    dsc: float
    jac_nonpos_percent: float
    ndv_percent: Optional[float]
    sdlogj: Optional[float]
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
    ds: str = "OASIS",
) -> ValResult:
    """Run validation loop and return protocol metrics plus last visualization tensors."""
    model.eval()
    reg_nearest = None
    reg_bilin = None
    dsc_meter = AverageMeter()
    jac_meter = AverageMeter()

    ds_key = str(ds).upper()
    is_ixi = ds_key == "IXI"
    ndv_meter = AverageMeter() if is_ixi else None
    sdlogj_meter = None if is_ixi else AverageMeter()



    if is_ixi:
        def calc_jac(flow, x_seg):
            jac, ndv = digital_jacobian_metrics(flow, mask=x_seg)
            return float(jac), float(ndv), None
    else:
        def calc_jac(flow, x_seg):
            del x_seg
            jac = jacobian_nonpositive_percent(flow, crop=1)
            sdlogj = logdet_std_from_flow(flow)
            return float(jac), None, float(sdlogj)

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
        jac, ndv, sdlogj = calc_jac(flow, x_seg)
        jac_meter.update(float(jac), x.size(0))

        if ndv_meter is not None and ndv is not None: ndv_meter.update(float(ndv), x.size(0))
        if sdlogj_meter is not None and sdlogj is not None: sdlogj_meter.update(float(sdlogj), x.size(0))
        
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

    ndv_avg = ndv_meter.avg if ndv_meter is not None and ndv_meter.count > 0 else None
    sdlogj_avg = sdlogj_meter.avg if sdlogj_meter is not None and sdlogj_meter.count > 0 else None
    return ValResult(dsc=dsc_meter.avg, jac_nonpos_percent=jac_meter.avg, ndv_percent=ndv_avg, sdlogj=sdlogj_avg, last_vis=last_vis)
