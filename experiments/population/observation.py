"""Inspect the Gaussian appearance reference at a fixed canonical map; no model selection or registration."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

from experiments.population.appearance import appearance_marginal
from utils import pkload
from utils.field import _warp
from utils.field_io import load_flow_npz


def residual_summary(residual: torch.Tensor) -> dict:
    """Descriptive voxel residual statistics; spatial correlation is not a fitted noise covariance."""
    values = residual.detach().double()
    correlations = []
    for axis in range(2, 5):
        left = values.narrow(axis, 0, values.shape[axis] - 1).flatten()
        right = values.narrow(axis, 1, values.shape[axis] - 1).flatten()
        left, right = left - left.mean(), right - right.mean()
        norm = left.norm() * right.norm()
        correlations.append(float(torch.dot(left, right) / norm) if norm > 0 else None)
    return {
        "mean": float(values.mean()),
        "rms": float(values.square().mean().sqrt()),
        "adjacent_correlation_zyx": correlations,
    }


def load_image(path: Path, scale: float, device: str | torch.device) -> torch.Tensor:
    payload = pkload(path)
    if not isinstance(payload, (tuple, list)) or len(payload) != 2:
        raise ValueError(f"{path}: expected a single-subject (image, segmentation) pickle")
    image = np.asarray(payload[0], dtype=np.float32)
    if image.ndim != 3 or min(image.shape) < 2 or not np.isfinite(image).all():
        raise ValueError(f"{path}: image must be finite and 3D with at least two voxels per axis")
    scaled = torch.from_numpy(np.ascontiguousarray(image))[None, None].to(device) / scale
    if not torch.isfinite(scaled).all():
        raise ValueError(f"{path}: image scale produces nonfinite observations")
    return scaled


def add_appearance_args(parser: argparse.ArgumentParser) -> None:
    for name in ("image-scale", "noise-std", "gain-mean", "gain-std", "offset-std", "bias-std", "gain-min"):
        parser.add_argument(f"--{name}", type=float, required=True)
    parser.add_argument("--offset-mean", type=float, default=0.0)


def appearance_prior(args, device: str | torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    for name in ("image_scale", "noise_std", "gain_std", "offset_std", "bias_std", "gain_min"):
        value = getattr(args, name)
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive")
    if not math.isfinite(args.gain_mean) or not math.isfinite(args.offset_mean):
        raise ValueError("gain and offset prior means must be finite")
    mean = torch.tensor([args.gain_mean, args.offset_mean, *([0.0] * 26)], device=device, dtype=torch.float64)
    std = torch.tensor([args.gain_std, args.offset_std, *([args.bias_std] * 26)], device=device, dtype=torch.float64)
    return mean, torch.diag(std.square())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--subject", type=Path, required=True)
    parser.add_argument(
        "--flow", type=Path, help="canonical subject->template pull displacement, NOT a legacy CTCF field"
    )
    parser.add_argument("--output", type=Path, required=True)
    add_appearance_args(parser)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    mean, covariance = appearance_prior(args, args.device)
    if args.output.exists():
        raise FileExistsError(f"use a fresh output directory: {args.output}")
    observed = load_image(args.subject, args.image_scale, args.device)
    template = load_image(args.template, 1.0, args.device)
    if observed.shape != template.shape:
        raise ValueError("template and subject must share the processed voxel grid")
    template_mean = template.double().mean()
    template_rms = (template.double() - template_mean).square().mean().sqrt()
    if template_rms <= 0:
        raise ValueError("template must have nonzero contrast")
    template = ((template.double() - template_mean) / template_rms).float()
    if args.flow is not None:
        flow = load_flow_npz(args.flow).to(args.device)
        if flow.shape[2:] != observed.shape[2:] or not torch.isfinite(flow).all():
            raise ValueError("flow must be finite and share the observed voxel grid")
        template = _warp(template, flow)
    with torch.no_grad():
        marginal = appearance_marginal(observed, template, args.noise_std**2, mean, covariance, args.gain_min)
    report = {
        "kind": "EXPLORATORY_APPEARANCE_REFERENCE",
        "selection": "NOT_RUN",
        "arguments": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "shape": list(observed.shape),
        "template_normalization": {"mean": float(template_mean), "rms": float(template_rms)},
        "nll": float(-marginal.log_prob[0]),
        "nll_per_voxel": float(-marginal.log_prob[0] / observed.numel()),
        "base_posterior_mean": marginal.posterior_mean[0].cpu().tolist(),
        "base_posterior_std": marginal.posterior_covariance[0].diagonal().sqrt().cpu().tolist(),
        "residual": residual_summary(marginal.residual),
        "residual_scope": "untruncated Gaussian-base nuisance fit, not a truncated-posterior predictive mean",
        "density_scope": "scaled subject voxels, conditional on the fixed template/map and supplied nuisance priors",
    }
    args.output.mkdir(parents=True)
    np.save(args.output / "residual.npy", marginal.residual[0, 0].float().cpu().numpy())
    (args.output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "nll_per_voxel": report["nll_per_voxel"], "selection": "NOT_RUN"}))


if __name__ == "__main__":
    main()
