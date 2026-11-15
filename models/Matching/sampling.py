"""Canonical voxel-coordinate sampling with differentiable backward operators."""

from __future__ import annotations

from itertools import product

import torch
import torch.nn.functional as F


def voxel_grid(shape, *, batch=1, device=None, dtype=torch.float32):
    if len(shape) != 3 or min(shape) < 1:
        raise ValueError("A positive three-dimensional shape is required")
    axes = [torch.arange(n, device=device, dtype=dtype) for n in shape]
    return torch.stack(torch.meshgrid(*axes, indexing="ij"), dim=0).unsqueeze(0).expand(batch, -1, -1, -1, -1)


def sample_voxels(image, coordinates, padding="border"):
    """Sample B,C,D,H,W at B,3,... voxel coordinates ordered z,y,x."""
    if image.ndim != 5 or coordinates.ndim < 3 or coordinates.shape[:2] != (image.shape[0], 3):
        raise ValueError("Expected a B,C,D,H,W image and B,3,... coordinates")
    if image.dtype not in (torch.float32, torch.float64) or coordinates.dtype != image.dtype:
        raise ValueError("Sampling geometry requires matching float32 or float64 tensors")
    if coordinates.device != image.device or padding not in ("zeros", "border"):
        raise ValueError("Sampling requires one device and zeros or border padding")
    if not bool(torch.isfinite(coordinates).all()):
        raise ValueError("Sampling coordinates must be finite")
    values = list(coordinates.unbind(1))
    shape = image.shape[2:]
    if padding == "border":
        values = [
            torch.where((value > 0) & (value < n - 1), value, value.clamp(0, n - 1).detach())
            for value, n in zip(values, shape, strict=True)
        ]
    lower = [value.floor().long() for value in values]
    fractions = [value - index for value, index in zip(values, lower, strict=True)]
    output = image.new_zeros((image.shape[0], image.shape[1], *coordinates.shape[2:]))
    flat = image.flatten(2)
    for offsets in product((0, 1), repeat=3):
        indices = [value + offset for value, offset in zip(lower, offsets, strict=True)]
        weight = torch.ones_like(values[0])
        valid = torch.ones_like(weight, dtype=torch.bool)
        for index, fraction, n, offset in zip(indices, fractions, shape, offsets, strict=True):
            weight = weight * (fraction if offset else 1 - fraction)
            valid = valid & (index >= 0) & (index < n)
        z, y, x = [index.clamp(0, n - 1) for index, n in zip(indices, shape, strict=True)]
        linear = (z * shape[1] * shape[2] + y * shape[2] + x).flatten(1).unsqueeze(1)
        sampled = flat.gather(2, linear.expand(-1, image.shape[1], -1)).reshape_as(output)
        output = output + sampled * (weight * valid).unsqueeze(1)
    return output


def resize_flow(flow, size):
    """Resample endpoint-aligned fields and convert each component's voxel units."""
    if flow.ndim != 5 or flow.shape[1] != 3 or len(size) != 3 or min(*flow.shape[2:], *size) < 2:
        raise ValueError("Flow resizing requires B,3,D,H,W with all input/output dimensions >= 2")
    scale = flow.new_tensor([(new - 1) / (old - 1) for old, new in zip(flow.shape[2:], size, strict=True)])
    return F.interpolate(flow, size=tuple(size), mode="trilinear", align_corners=True) * scale.view(1, 3, 1, 1, 1)


def flow_inside_fraction(flow):
    positions = voxel_grid(flow.shape[2:], batch=flow.shape[0], device=flow.device, dtype=flow.dtype) + flow
    valid = torch.ones_like(flow[:, :1], dtype=torch.bool)
    for axis, n in enumerate(flow.shape[2:]):
        valid = valid & (positions[:, axis : axis + 1] >= 0) & (positions[:, axis : axis + 1] <= n - 1)
    return valid.to(flow.dtype).mean()
