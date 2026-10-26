"""Orthonormal velocity coordinates and canonical subject-to-atlas maps."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from utils.field import _warp, compose_flows, integrate_svf


@dataclass(frozen=True)
class HaarLevel:
    shape: tuple[int, int, int]
    details: tuple[torch.Tensor, ...]


@dataclass(frozen=True)
class HaarPyramid:
    coarse: torch.Tensor
    levels: tuple[HaarLevel, ...]


def _check_values(values: torch.Tensor) -> None:
    if values.ndim < 3 or min(values.shape[-3:]) < 1:
        raise ValueError("Expected three nonempty spatial dimensions")
    if values.dtype not in (torch.float32, torch.float64):
        raise ValueError("Geometry requires float32 or float64 tensors")


def _split_axis(values: torch.Tensor, axis: int) -> tuple[torch.Tensor, torch.Tensor]:
    values = values.movedim(axis, -1)
    paired = values.shape[-1] // 2 * 2
    even, odd = values[..., :paired:2], values[..., 1:paired:2]
    low, high = (even + odd) * 2**-0.5, (even - odd) * 2**-0.5
    if values.shape[-1] % 2:
        low = torch.cat((low, values[..., -1:]), dim=-1)
    return low.movedim(-1, axis), high.movedim(-1, axis)


def _merge_axis(low: torch.Tensor, high: torch.Tensor, axis: int, size: int) -> torch.Tensor:
    low, high = low.movedim(axis, -1), high.movedim(axis, -1)
    if low.shape[-1] != (size + 1) // 2 or high.shape[-1] != size // 2:
        raise ValueError("Haar bands do not match the recorded spatial shape")
    paired_low = low[..., : size // 2]
    even, odd = (paired_low + high) * 2**-0.5, (paired_low - high) * 2**-0.5
    values = torch.stack((even, odd), dim=-1).flatten(-2)
    if size % 2:
        values = torch.cat((values, low[..., -1:]), dim=-1)
    return values.movedim(-1, axis)


def haar_forward(values: torch.Tensor, levels: int) -> HaarPyramid:
    """Transform only the low band recursively; carry odd endpoints without scaling."""
    _check_values(values)
    if type(levels) is not int or levels < 0:
        raise ValueError("levels must be a nonnegative integer")
    records = []
    coarse = values
    for _ in range(levels):
        shape = tuple(coarse.shape[-3:])
        if shape == (1, 1, 1):
            raise ValueError("Requested levels exceed the spatial resolution")
        bands = [coarse]
        for axis in (-3, -2, -1):
            bands = [part for band in bands for part in _split_axis(band, axis)]
        coarse = bands[0]
        records.append(HaarLevel(shape, tuple(bands[1:])))
    return HaarPyramid(coarse, tuple(records))


def haar_inverse(pyramid: HaarPyramid) -> torch.Tensor:
    """Invert a pyramid with detail bands in lexicographic low/high axis order."""
    _check_values(pyramid.coarse)
    coarse = pyramid.coarse
    for level in reversed(pyramid.levels):
        if len(level.details) != 7:
            raise ValueError("Each Haar level must contain seven detail bands")
        bands = [coarse, *level.details]
        for axis in (-1, -2, -3):
            bands = [
                _merge_axis(bands[index], bands[index + 1], axis, level.shape[axis])
                for index in range(0, len(bands), 2)
            ]
        coarse = bands[0]
    return coarse


def coefficient_bands(pyramid: HaarPyramid) -> tuple[torch.Tensor, ...]:
    """Return the coarse tensor followed by detail bands from coarse to fine."""
    return (pyramid.coarse, *(band for level in reversed(pyramid.levels) for band in level.details))


def _check_velocity(velocity: torch.Tensor) -> None:
    _check_values(velocity)
    if velocity.ndim != 5 or velocity.shape[1] != 3 or velocity.shape[0] < 1:
        raise ValueError("Expected velocity shape (B, 3, D, H, W)")


def pad_velocity(interior: torch.Tensor, collar: int) -> torch.Tensor:
    """Add a zero collar to each spatial side of an interior velocity field."""
    _check_velocity(interior)
    if type(collar) is not int or collar < 1:
        raise ValueError("collar must be a positive integer")
    return F.pad(interior, (collar,) * 6)


def canonical_flow(velocity: torch.Tensor, steps: int = 7) -> torch.Tensor:
    """Integrate a voxel-unit SVF using the canonical align_corners=True warp."""
    _check_velocity(velocity)
    if min(velocity.shape[-3:]) < 2:
        raise ValueError("Canonical warping requires spatial dimensions of at least two")
    if type(steps) is not int or steps < 0:
        raise ValueError("steps must be a nonnegative integer")
    with torch.autocast(device_type=velocity.device.type, enabled=False):
        return integrate_svf(velocity, _warp, steps=steps)


def subject_pair_flow(fixed_velocity: torch.Tensor, moving_velocity: torch.Tensor, steps: int = 7) -> torch.Tensor:
    """Materialize Exp(-v_moving) composed with Exp(v_fixed), fixed to moving."""
    _check_velocity(fixed_velocity)
    _check_velocity(moving_velocity)
    if (
        fixed_velocity.shape != moving_velocity.shape
        or fixed_velocity.dtype != moving_velocity.dtype
        or fixed_velocity.device != moving_velocity.device
    ):
        raise ValueError("Subject velocities must have matching shapes, dtypes and devices")
    with torch.autocast(device_type=fixed_velocity.device.type, enabled=False):
        fixed_flow = canonical_flow(fixed_velocity, steps=steps)
        inverse_moving_flow = canonical_flow(-moving_velocity, steps=steps)
        return compose_flows(fixed_flow, inverse_moving_flow)
