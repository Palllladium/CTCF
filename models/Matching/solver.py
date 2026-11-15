"""Continuous matching costs and differentiable refinement in voxel units."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from .sampling import sample_voxels, voxel_grid


def _squared_cost(fixed, moving):
    return (
        (F.normalize(fixed, dim=1, eps=1e-6) - F.normalize(moving, dim=1, eps=1e-6)).square().sum(dim=1, keepdim=True)
    )


def _check_features(fixed, moving, flow):
    if fixed.shape != moving.shape or fixed.ndim != 5 or flow.ndim != 5 or flow.shape[:2] != (fixed.shape[0], 3):
        raise ValueError("Matching requires equal B,C,D,H,W features and a B,3,D,H,W field")
    if min(*fixed.shape[2:], *flow.shape[2:]) < 2:
        raise ValueError("Feature and field grids require spatial dimensions >= 2")


def feature_points(points, field_shape, feature_shape):
    scale = points.new_tensor([(b - 1) / (a - 1) for a, b in zip(field_shape, feature_shape, strict=True)])
    return points * scale.view(1, 3, *((1,) * (points.ndim - 2)))


def matching_cost(fixed, moving, flow):
    _check_features(fixed, moving, flow)
    points = voxel_grid(flow.shape[2:], batch=flow.shape[0], device=flow.device, dtype=flow.dtype)
    query = (
        fixed
        if fixed.shape[2:] == flow.shape[2:]
        else sample_voxels(fixed, feature_points(points, flow.shape[2:], fixed.shape[2:]))
    )
    return _squared_cost(query, sample_voxels(moving, feature_points(points + flow, flow.shape[2:], moving.shape[2:])))


def _masked_mean(values, mask):
    if mask is None:
        return values.mean()
    if mask.shape != values.shape or not bool(torch.isfinite(mask).all()) or bool((mask < 0).any()):
        raise ValueError("Matching support must be finite, nonnegative and have shape B,1,D,H,W")
    count = mask.sum()
    if not bool(count > 0):
        raise ValueError("Matching support is empty")
    return (values * mask).sum() / count


def matching_energy(fixed, moving, flow, *, reg_weight=0.01, mask=None, outside_weight=1.0):
    if not all(math.isfinite(x) and x >= 0 for x in (reg_weight, outside_weight)):
        raise ValueError("Energy weights must be finite and nonnegative")
    cost = matching_cost(fixed, moving, flow)
    value = _masked_mean(cost, mask)
    if reg_weight:
        penalty = torch.stack([flow.diff(dim=axis).square().mean() for axis in (2, 3, 4)]).mean()
        value = value + reg_weight * penalty
    if outside_weight:
        positions = voxel_grid(flow.shape[2:], batch=flow.shape[0], device=flow.device, dtype=flow.dtype) + flow
        upper = flow.new_tensor([n - 1 for n in flow.shape[2:]]).view(1, 3, 1, 1, 1)
        distance = (F.relu(-positions).square() + F.relu(positions - upper).square()).sum(dim=1, keepdim=True)
        value = value + outside_weight * _masked_mean(distance, mask)
    return value


def candidate_logits(fixed, moving, flow, offsets, *, temperature=0.2, points=None):
    """Evaluate the continuous matching cost on dense or sparse candidate coordinates."""
    _check_features(fixed, moving, flow)
    offsets = torch.as_tensor(offsets, device=flow.device, dtype=flow.dtype)
    if offsets.ndim != 2 or offsets.shape[1] != 3 or not len(offsets):
        raise ValueError("Candidate offsets must have shape K,3")
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("Candidate temperature must be finite and positive")
    if points is None:
        points = voxel_grid(flow.shape[2:], batch=flow.shape[0], device=flow.device, dtype=flow.dtype)
        centers = points + flow
    else:
        centers = points + sample_voxels(flow, points)
    fixed_query = sample_voxels(fixed, feature_points(points, flow.shape[2:], fixed.shape[2:]))
    view = (1, 3, *((1,) * (points.ndim - 2)))
    scores = [
        _squared_cost(
            fixed_query,
            sample_voxels(moving, feature_points(centers + offset.view(view), flow.shape[2:], moving.shape[2:])),
        )
        for offset in offsets
    ]
    return -torch.cat(scores, dim=1) / temperature


def refine(
    fixed,
    moving,
    flow,
    *,
    steps=3,
    step_size=0.25,
    reg_weight=0.01,
    mask=None,
    outside_weight=1.0,
    max_update=0.5,
    create_graph=True,
):
    """Refine field-grid voxel displacements; max_update bounds each component per step."""
    if isinstance(steps, bool) or not isinstance(steps, int) or steps < 0:
        raise ValueError("Solver steps must be a nonnegative integer")
    if not all(math.isfinite(x) and x > 0 for x in (step_size, max_update)):
        raise ValueError("Solver step size and update bound must be finite and positive")
    _check_features(fixed, moving, flow)
    if not steps:
        return flow if create_graph else flow.detach()
    with torch.enable_grad():
        current = flow if flow.requires_grad else flow.detach().requires_grad_(True)
        pixels = flow.shape[0] * math.prod(flow.shape[2:])
        for _ in range(steps):
            energy = matching_energy(
                fixed, moving, current, reg_weight=reg_weight, mask=mask, outside_weight=outside_weight
            )
            gradient = torch.autograd.grad(energy, current, create_graph=create_graph)[0]
            update = max_update * torch.tanh(step_size * pixels * gradient / max_update)
            current = current - update
            if not create_graph:
                current = current.detach().requires_grad_(True)
        return current if create_graph else current.detach()
