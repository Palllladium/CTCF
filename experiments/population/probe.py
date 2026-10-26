"""Fit a label-driven SVF for a diagnostic; this is supervised geometry, not ground truth."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from experiments.population.geometry import canonical_flow, coefficient_bands, haar_forward, pad_velocity
from utils import pkload
from utils.dice import dice_per_label
from utils.field import _warp
from utils.field_io import save_flow_npz_atomic
from utils.losses import Grad3d


@dataclass
class LabelFit:
    velocity: torch.Tensor
    flow: torch.Tensor
    history: list[dict[str, float | int]]


def _smooth_labels(values: torch.Tensor, sigma: float) -> torch.Tensor:
    if sigma == 0:
        return values
    radius = math.ceil(3 * sigma)
    coordinates = torch.arange(-radius, radius + 1, device=values.device, dtype=values.dtype)
    kernel = torch.exp(-0.5 * (coordinates / sigma).square())
    kernel = kernel / kernel.sum()
    channels = values.shape[1]
    for axis in range(3):
        shape, padding = [1, 1, 1], [0, 0, 0]
        shape[axis], padding[axis] = len(kernel), radius
        weights = kernel.reshape(1, 1, *shape).expand(channels, 1, *shape)
        values = F.conv3d(values, weights, padding=padding, groups=channels)
    return values


def fit_label_velocity(
    moving: torch.Tensor,
    fixed: torch.Tensor,
    *,
    steps: int,
    lr: float,
    regularization: float,
    collar: int,
    integration_steps: int = 7,
    log_every: int = 10,
    smoothing_sigmas: tuple[float, ...] = (0.0,),
) -> LabelFit:
    """Fit fixed->moving, with ``steps`` updates at each explicitly chosen smoothing scale."""
    if moving.shape != fixed.shape or moving.ndim != 5 or moving.shape[:2] != (1, 1):
        raise ValueError("labels must have matching [1,1,D,H,W] shapes")
    if moving.device != fixed.device or moving.is_floating_point() or fixed.is_floating_point():
        raise ValueError("labels must be integer tensors on the same device")
    if steps < 0 or not math.isfinite(lr) or lr <= 0 or log_every < 1:
        raise ValueError("steps must be nonnegative; lr and log_every must be positive")
    if regularization < 0 or not math.isfinite(regularization):
        raise ValueError("regularization must be finite and nonnegative")
    if (
        not smoothing_sigmas
        or any(not math.isfinite(sigma) or sigma < 0 for sigma in smoothing_sigmas)
        or smoothing_sigmas[-1] != 0
        or any(a <= b for a, b in pairwise(smoothing_sigmas))
    ):
        raise ValueError("smoothing_sigmas must decrease strictly to zero")
    if collar < 1 or any(n - 2 * collar < 2 for n in moving.shape[2:]):
        raise ValueError("collar must leave at least two interior voxels along every axis")
    if moving.min() < 0 or fixed.min() < 0:
        raise ValueError("label IDs must be nonnegative")
    label_ids = torch.unique(torch.cat((moving.flatten(), fixed.flatten())))
    label_ids = label_ids[label_ids > 0]
    if not len(label_ids):
        raise ValueError("at least one foreground label is required")
    source = torch.cat([(moving == label).float() for label in label_ids], dim=1)
    target = torch.cat([(fixed == label).float() for label in label_ids], dim=1)
    interior = torch.nn.Parameter(
        torch.zeros(
            (1, 3, *(n - 2 * collar for n in moving.shape[2:])),
            device=moving.device,
            dtype=torch.float32,
        )
    )
    diffusion = Grad3d(penalty="l2")
    history = []
    for scale, sigma in enumerate(smoothing_sigmas):
        source_scale, target_scale = _smooth_labels(source, sigma), _smooth_labels(target, sigma)
        optimizer = torch.optim.Adam([interior], lr=lr)
        for step in range(steps + 1):
            optimizer.zero_grad(set_to_none=True)
            velocity = pad_velocity(interior, collar)
            flow = canonical_flow(velocity, integration_steps)
            # A per-voxel label discrepancy; its scale does not shrink with the number of labels.
            data = 0.5 * (_warp(source_scale, flow) - target_scale).square().sum(dim=1).mean()
            smooth = diffusion(velocity)
            loss = data + regularization * smooth
            if not torch.isfinite(loss):
                raise FloatingPointError(f"nonfinite label fit at scale {scale}, step {step}")
            if step % log_every == 0 or step == steps:
                history.append(
                    {
                        "step": scale * steps + step,
                        "smoothing_sigma": sigma,
                        "loss": float(loss.detach()),
                        "label_ssd": float(data.detach()),
                        "diffusion": float(smooth.detach()),
                    }
                )
            if step == steps:
                break
            loss.backward()
            if interior.grad is None or not torch.isfinite(interior.grad).all():
                raise FloatingPointError(f"nonfinite label-fit gradient at scale {scale}, step {step}")
            optimizer.step()
    return LabelFit(velocity.detach(), flow.detach(), history)


def _load_labels(path: Path, device: str) -> torch.Tensor:
    payload = pkload(path)
    if not isinstance(payload, (tuple, list)) or len(payload) != 2:
        raise ValueError(f"{path}: expected a single-subject (image, segmentation) pickle")
    labels = np.asarray(payload[1])
    if labels.ndim != 3 or not np.issubdtype(labels.dtype, np.integer):
        raise ValueError(f"{path}: segmentation must be a 3D integer array")
    return torch.from_numpy(np.ascontiguousarray(labels.astype(np.int64)))[None, None].to(device)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--moving", type=Path, required=True, help="template label subject, in processed pickle format")
    parser.add_argument("--fixed", type=Path, required=True, help="subject to map into the template")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, required=True, help="updates at each smoothing scale")
    parser.add_argument("--lr", type=float, required=True)
    parser.add_argument("--regularization", type=float, required=True)
    parser.add_argument("--collar", type=int, required=True)
    parser.add_argument("--levels", type=int, required=True)
    parser.add_argument("--integration-steps", type=int, default=7)
    parser.add_argument(
        "--smoothing-sigmas", type=float, nargs="+", default=[0.0], help="voxel sigmas, decreasing to 0"
    )
    parser.add_argument("--device", default="cpu")
    run_probe(parser.parse_args())


def run_probe(args) -> dict:
    moving, fixed = _load_labels(args.moving, args.device), _load_labels(args.fixed, args.device)
    if args.output.exists():
        raise FileExistsError(f"use a fresh output directory: {args.output}")
    interior_shape = tuple(n - 2 * args.collar for n in fixed.shape[2:])
    haar_forward(torch.zeros((1, 3, *interior_shape)), args.levels)
    fit = fit_label_velocity(
        moving,
        fixed,
        steps=args.steps,
        lr=args.lr,
        regularization=args.regularization,
        collar=args.collar,
        integration_steps=args.integration_steps,
        smoothing_sigmas=tuple(args.smoothing_sigmas),
    )
    interior = fit.velocity[:, :, args.collar : -args.collar, args.collar : -args.collar, args.collar : -args.collar]
    bands = coefficient_bands(haar_forward(interior, args.levels))
    label_ids = sorted(set(moving.unique().tolist()) | set(fixed.unique().tolist()))
    label_ids = [label for label in label_ids if label > 0]
    warped = _warp(moving.float(), fit.flow, mode="nearest").long()
    dice = dice_per_label(warped, fixed, label_ids)
    initial_dice = dice_per_label(moving, fixed, label_ids)
    foreground = (warped > 0) | (fixed > 0)
    foreground_voxels = int(foreground.sum())
    args.output.mkdir(parents=True)
    save_flow_npz_atomic(args.output / "flow.npz", fit.flow)
    np.save(args.output / "velocity.npy", fit.velocity.cpu().numpy())
    report = {
        "kind": "SUPERVISED_DIAGNOSTIC",
        "certification": "NOT_RUN",
        "direction": "fixed_to_moving",
        "channels": "zyx",
        "units": "voxels",
        "align_corners": True,
        "arguments": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "history": fit.history,
        "label_ids": label_ids,
        "initial_dice": initial_dice.tolist(),
        "dice": dice.tolist(),
        "foreground_union_voxels": foreground_voxels,
        "foreground_mismatch_fraction": float(((warped != fixed) & foreground).sum()) / max(foreground_voxels, 1),
        "dice_scope": "foreground_union_present_not_benchmark_Dice35",
        "band_shapes": [list(band.shape) for band in bands],
        "band_energy": [float(band.double().square().sum()) for band in bands],
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "dice_mean": float(dice.mean()), "certification": "NOT_RUN"}))
    return report


if __name__ == "__main__":
    main()
