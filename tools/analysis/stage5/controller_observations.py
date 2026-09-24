"""Read-only forward observations for the frozen attenuation controllers.

These summaries describe the endpoint policy, not the cause of training gradients.
No gradients, losses, labels, or alternate controller forwards are evaluated here.
"""

from __future__ import annotations

import math
import time
from typing import Any

import torch

from models.CTCF.controller import (
    STAGE5_S2_ATTENUATION_HEAD,
    STAGE5_S4_ATTENUATION_HEAD,
    Stage5ControllerOutput,
)

_CHUNK_VALUES = 262_144


def _summary(value: torch.Tensor, *, unit_interval: bool, add: torch.Tensor | None = None) -> dict[str, Any]:
    """Reduce detached CPU slabs without allocating a full-volume GPU temporary."""
    count = value.numel()
    finite_count = 0
    total = squared_total = 0.0
    minimum, maximum = math.inf, -math.inf
    bins = dict.fromkeys(("below_zero", "at_zero", "between_zero_and_one", "at_one", "above_one"), 0)
    # Transfer before flattening: an interior crop is not necessarily contiguous.
    depth_step = max(1, _CHUNK_VALUES // (value.shape[-2] * value.shape[-1]))
    for batch in range(value.shape[0]):
        for channel in range(value.shape[1]):
            for start in range(0, value.shape[2], depth_step):
                index = (batch, channel, slice(start, start + depth_step))
                chunk = value.detach()[index].to(device="cpu").to(dtype=torch.float64).reshape(-1)
                if add is not None:
                    # Match the actual FP32 alpha sum before the CPU reduction.
                    other = add.detach()[index].to(device="cpu", dtype=value.dtype).reshape(-1)
                    chunk = (chunk.to(dtype=value.dtype) + other).to(dtype=torch.float64)
                finite = chunk[torch.isfinite(chunk)]
                finite_count += finite.numel()
                if finite.numel() == 0:
                    continue
                minimum = min(minimum, float(finite.min()))
                maximum = max(maximum, float(finite.max()))
                total += float(finite.sum())
                squared_total += float(finite.square().sum())
                if unit_interval:
                    bins["below_zero"] += int((finite < 0).sum())
                    bins["at_zero"] += int((finite == 0).sum())
                    bins["between_zero_and_one"] += int(((finite > 0) & (finite < 1)).sum())
                    bins["at_one"] += int((finite == 1).sum())
                    bins["above_one"] += int((finite > 1).sum())
    result = {
        "count": count,
        "finite_count": finite_count,
        "nonfinite_count": count - finite_count,
        "min": minimum if finite_count else None,
        "max": maximum if finite_count else None,
        "abs_max": max(abs(minimum), abs(maximum)) if finite_count else None,
        "mean": total / finite_count if finite_count else None,
        "rms": math.sqrt(squared_total / finite_count) if finite_count else None,
    }
    if unit_interval:
        result["unit_interval_counts"] = bins
    return result


def _regions(
    value: torch.Tensor, *, collar_width: int, unit_interval: bool, add: torch.Tensor | None = None
) -> dict[str, Any]:
    full = _summary(value, unit_interval=unit_interval, add=add)
    interior = None
    if min(value.shape[-3:]) > 2 * collar_width:
        index = (..., *(slice(collar_width, -collar_width) for _ in range(3)))
        interior = _summary(value[index], unit_interval=unit_interval, add=None if add is None else add[index])
    return {"full_volume": full, "interior_without_collar": interior}


def observe_controller_output(output: Stage5ControllerOutput, *, collar_width: int) -> dict[str, Any] | None:
    """Summarize only A policies; observations never alter the requested field."""
    if output.variant not in ("A2P", "A24P"):
        return None
    if not isinstance(collar_width, int) or isinstance(collar_width, bool) or collar_width < 1:
        raise ValueError("observation collar width must be a positive integer")
    # Work queued by the real forward belongs to inference, not this timer.
    peak_before_observation = 0
    if output.raw_head.device.type == "cuda":
        torch.cuda.synchronize(output.raw_head.device)
        peak_before_observation = int(torch.cuda.max_memory_allocated(output.raw_head.device))
    started = time.perf_counter()
    active = [("s2", STAGE5_S2_ATTENUATION_HEAD, output.alpha_s2)]
    if output.variant == "A24P":
        active.append(("s4", STAGE5_S4_ATTENUATION_HEAD, output.alpha_s4))
    raw_head = {}
    alpha = {}
    for name, channel, gate in active:
        if gate is None:
            raise ValueError(f"{output.variant} observation requires alpha_{name}")
        raw_head[name] = {
            "channel_index": channel,
            **_regions(output.raw_head[:, channel : channel + 1], collar_width=collar_width, unit_interval=True),
        }
        alpha[name] = _regions(gate, collar_width=collar_width, unit_interval=True)
    alpha_sum = None
    if output.variant == "A24P":
        alpha_sum = _regions(output.alpha_s2, collar_width=collar_width, unit_interval=True, add=output.alpha_s4)
    return {
        "schema": "ctcf-stage5-controller-forward-observations-v1",
        "variant": output.variant,
        "interpretation": "descriptive endpoint forward values; not a causal gradient diagnosis",
        "raw_head_scope": "active attenuation channels before straight-through clamp",
        "alpha_scope": "effective attenuation after the existing controller policy, before collar taper",
        "requested_delta_units": "full-resolution voxel displacement components after collar taper",
        "collar_width": collar_width,
        "finite_statistics_denominator": "finite_count; nonfinite values are counted separately",
        "raw_head": raw_head,
        "alpha": alpha,
        "alpha_sum": alpha_sum,
        "requested_delta": _regions(output.requested_delta, collar_width=collar_width, unit_interval=False),
        "observation_seconds": time.perf_counter() - started,
        "timing_scope": "excluded from decision runtime; CPU reductions and device transfers after CUDA synchronization",
        "peak_memory_bytes_before_observation": peak_before_observation,
        "memory_scope": "decision peak measured before diagnostic transfers and their bounded staging buffers",
    }
