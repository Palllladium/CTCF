"""Label-free anatomical targets: soft tissue classes of a 1-D intensity Gaussian mixture.

The mixture is fitted once on foreground voxels of training images (background is exactly 0
in the Learn2Reg OASIS volumes) and its components are ordered by mean intensity, so in T1
they read CSF < GM < WM. No segmentation enters the fit or the posteriors.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
import torch


@dataclass(frozen=True)
class TissueGMM:
    means: np.ndarray
    variances: np.ndarray
    weights: np.ndarray


def fit_tissue_gmm(
    volumes: Iterable[np.ndarray],
    k: int = 3,
    iters: int = 100,
    samples_per_volume: int = 200_000,
    seed: int = 0,
) -> TissueGMM:
    """EM fit of a k-component 1-D mixture to a fixed random subsample of foreground voxels."""
    rng = np.random.default_rng(seed)
    samples = []
    for volume in volumes:
        foreground = np.asarray(volume, dtype=np.float64)[np.asarray(volume) > 0]
        take = min(samples_per_volume, foreground.size)
        samples.append(rng.choice(foreground, size=take, replace=False))
    x = np.concatenate(samples)[:, None]
    if x.size < k:
        raise ValueError("Too few foreground voxels to fit the tissue mixture.")

    means = np.quantile(x, np.linspace(0.15, 0.85, k))
    variances = np.full(k, x.var() / k)
    weights = np.full(k, 1.0 / k)
    for _ in range(iters):
        log_p = -0.5 * ((x - means) ** 2 / variances + np.log(2.0 * math.pi * variances)) + np.log(weights)
        log_p -= log_p.max(axis=1, keepdims=True)
        resp = np.exp(log_p)
        resp /= resp.sum(axis=1, keepdims=True)
        mass = resp.sum(axis=0)
        weights = mass / x.shape[0]
        means = (resp * x).sum(axis=0) / mass
        variances = (resp * (x - means) ** 2).sum(axis=0) / mass + 1e-8

    order = np.argsort(means)
    return TissueGMM(means=means[order], variances=variances[order], weights=weights[order])


def tissue_posteriors(image: torch.Tensor, gmm: TissueGMM) -> torch.Tensor:
    """[B,1,D,H,W] image -> [B,1+k,D,H,W]: channel 0 is background, then soft tissue classes."""
    shape = (1, -1, 1, 1, 1)
    means, variances, weights = (
        torch.as_tensor(values, dtype=image.dtype, device=image.device).view(shape)
        for values in (gmm.means, gmm.variances, gmm.weights)
    )
    log_p = -0.5 * ((image - means) ** 2 / variances + torch.log(2.0 * math.pi * variances)) + torch.log(weights)
    foreground = (image > 0).to(image.dtype)
    return torch.cat([1.0 - foreground, torch.softmax(log_p, dim=1) * foreground], dim=1)
