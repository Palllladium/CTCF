"""Stage-5 controller NCC with FP64 local moments and a checked value range.

This is intentionally separate from the historical NCC used to train U0 and
from search/L3 metrics. The accumulation and adjoint match the H100 diagnostic
reference; epsilon, window padding and full-volume averaging are unchanged.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

NCC_VARIANCE_FLOOR = 1e-5
NCC_RANGE_TOLERANCE = 1e-6


def controller_ncc_contract() -> dict:
    return {
        "ncc_operator_id": "CTCF_STAGE5_NCC_FP64_SEPARABLE_CHECKED_V1",
        "ncc_input_dtype": "float32",
        "ncc_moments_dtype": "float64",
        "ncc_box_sum": "separable_axis_order_DHW_offset_order_ascending",
        "ncc_padding": "zero_full_window_divisor",
        "ncc_variance_floor": NCC_VARIANCE_FLOOR,
        "ncc_reduction": "negative_full_volume_mean_squared_correlation_float64_then_float32",
        "ncc_range_tolerance": NCC_RANGE_TOLERANCE,
        "ncc_out_of_range_policy": "RAISE_WITHOUT_CLIPPING",
        "remaining_objective_dtype": "float32",
    }


def _separable_box_sum(value: torch.Tensor, window: tuple[int, ...]) -> torch.Tensor:
    for dim, width in enumerate(window, start=2):
        padding = [0] * 6
        offset = 2 * (4 - dim)
        padding[offset : offset + 2] = [width // 2] * 2
        padded = F.pad(value, padding)
        result = torch.zeros_like(value)
        for shift in range(width):
            result.add_(padded.narrow(dim, shift, value.shape[dim]))
        value = result
    return value


class _BoxSum(torch.autograd.Function):
    """The symmetric, zero-padded box operator is its own adjoint.

    No image or unfolded window is retained for backward. This avoids a large
    full-volume FP64 convolution workspace on the H100 training grid.
    """

    @staticmethod
    def forward(ctx, value, window):
        ctx.window = window
        return _separable_box_sum(value, window)

    @staticmethod
    def backward(ctx, gradient):
        return _BoxSum.apply(gradient, ctx.window), None


def _squared_correlation(first: torch.Tensor, second: torch.Tensor, window: tuple[int, ...]) -> torch.Tensor:
    def box(value):
        return _BoxSum.apply(value, window)

    first_sum, second_sum = box(first), box(second)
    first_squared = box(first * first)
    second_squared = box(second * second)
    product = box(first * second)
    size = float(window[0] * window[1] * window[2])
    mean_first, mean_second = first_sum / size, second_sum / size
    cross = product - mean_second * first_sum - mean_first * second_sum + mean_first * mean_second * size
    var_first = first_squared - 2 * mean_first * first_sum + mean_first * mean_first * size
    var_second = second_squared - 2 * mean_second * second_sum + mean_second * mean_second * size
    denominator = var_first.clamp_min(NCC_VARIANCE_FLOOR) * var_second.clamp_min(NCC_VARIANCE_FLOOR)
    return cross.square() / denominator


class ControllerNCC(torch.nn.Module):
    def __init__(self, win: tuple[int, int, int]):
        super().__init__()
        self.window = tuple(win)
        if len(self.window) != 3 or any(type(width) is not int or width <= 0 or width % 2 != 1 for width in win):
            raise ValueError("Stage5 NCC needs three positive odd integer window widths")

    def forward(self, first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
        if first.ndim != 5 or first.shape[1] != 1 or first.shape != second.shape:
            raise ValueError("Stage5 NCC needs matching scalar 3D images")
        if first.dtype != torch.float32 or second.dtype != torch.float32:
            raise ValueError("Stage5 NCC requires the objective's FP32 image tensors")
        with torch.autocast(device_type=first.device.type, enabled=False):
            cc = _squared_correlation(first.double(), second.double(), self.window)
            # One check over all windows: a finite mean can conceal bad windows.
            valid = torch.isfinite(cc) & (cc >= 0) & (cc <= 1.0 + NCC_RANGE_TOLERANCE)
            if not bool(valid.all()):
                raise FloatingPointError("Stage5 NCC local squared correlation is non-finite or outside [0, 1]")
            return -cc.mean().float()
