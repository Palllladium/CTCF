"""A Gaussian reference observation model with exactly marginalized appearance coefficients."""

from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import product

import torch


@dataclass
class AppearanceMarginal:
    log_prob: torch.Tensor
    posterior_mean: torch.Tensor
    posterior_covariance: torch.Tensor
    prediction: torch.Tensor
    residual: torch.Tensor


def _dct_axes(shape: tuple[int, ...], order: int, reference: torch.Tensor) -> tuple[torch.Tensor, ...]:
    axes = []
    for size in shape:
        coordinates = torch.arange(size, device=reference.device, dtype=torch.float64) + 0.5
        frequencies = torch.arange(order + 1, device=reference.device, dtype=torch.float64)
        axis = torch.cos(math.pi / size * frequencies[:, None] * coordinates)
        scaling = torch.full_like(frequencies, math.sqrt(2))
        scaling[0] = 1
        axis = axis * scaling[:, None]
        axes.append(axis)
    return tuple(axes)


def _design_chunk(template: torch.Tensor, start: int, end: int, axes: tuple[torch.Tensor, ...]) -> torch.Tensor:
    positions = torch.arange(start, end, device=template.device)
    _, height, width = (axis.shape[1] for axis in axes)
    zz, yy, xx = positions // (height * width), positions // width % height, positions % width
    modes = [
        axes[0][kz, zz] * axes[1][ky, yy] * axes[2][kx, xx]
        for kz, ky, kx in product(range(axes[0].shape[0]), repeat=3)
        if (kz, ky, kx) != (0, 0, 0)
    ]
    constant = torch.ones((end - start, 1), device=template.device, dtype=torch.float64)
    bias = torch.stack(modes, dim=-1) if modes else constant[:, :0]
    nuisance = torch.cat((constant, bias), dim=-1).expand(template.shape[0], -1, -1)
    return torch.cat((template[:, start:end, None].double(), nuisance), dim=-1)


def appearance_marginal(
    observed: torch.Tensor,
    warped_template: torch.Tensor,
    noise_variance: torch.Tensor | float,
    prior_mean: torch.Tensor,
    prior_covariance: torch.Tensor,
    gain_min: float,
    *,
    bias_order: int = 2,
    chunk_size: int = 65536,
    observation_mask: torch.Tensor | None = None,
) -> AppearanceMarginal:
    """Evaluate p(y|template), with gain constrained above ``gain_min``.

    DCT bias modes have unit voxel-RMS and no DC component. Coefficients are gain,
    offset, then lexicographic DCT modes. Posterior moments and the nuisance-fit
    prediction/residual refer to the *untruncated Gaussian base*, not its predictive mean.
    All matrix operations and reductions use float64; chunking avoids an N-by-28 allocation.
    """
    if observed.shape != warped_template.shape or observed.ndim != 5 or observed.shape[1] != 1:
        raise ValueError("observed and warped_template must have matching [B,1,D,H,W] shapes")
    if (
        observed.device != warped_template.device
        or observed.dtype not in (torch.float32, torch.float64)
        or warped_template.dtype not in (torch.float32, torch.float64)
    ):
        raise ValueError("observations must be float32/float64 on the template device")
    shape = tuple(observed.shape[2:])
    if type(bias_order) is not int or not 0 <= bias_order < min(shape) or chunk_size < 1:
        raise ValueError("bias_order must fit each spatial axis and chunk_size must be positive")
    if not math.isfinite(gain_min) or gain_min <= 0:
        raise ValueError("gain_min must be finite and positive")
    batch, count = observed.shape[0], math.prod(shape)
    if observation_mask is not None:
        if observation_mask.shape != observed.shape or observation_mask.dtype != torch.bool:
            raise ValueError("observation_mask must be boolean and have the observation shape")
        mask = observation_mask.to(observed.device).flatten(1)
        observed_count = mask.sum(1, dtype=torch.float64)
        if bool((observed_count == 0).any()):
            raise ValueError("each observation mask must retain at least one voxel")
    else:
        mask, observed_count = None, count
    coefficients = (bias_order + 1) ** 3 + 1
    mean = prior_mean.to(device=observed.device, dtype=torch.float64)
    covariance = prior_covariance.to(device=observed.device, dtype=torch.float64)
    if mean.shape == (coefficients,):
        mean = mean.expand(batch, -1)
    if mean.shape != (batch, coefficients) or covariance.shape != (coefficients, coefficients):
        raise ValueError("prior dimensions must match gain, offset and the chosen DCT modes")
    if not torch.allclose(covariance, covariance.T, rtol=1e-12, atol=1e-12):
        raise ValueError("prior_covariance must be symmetric")
    variance = torch.as_tensor(noise_variance, device=observed.device, dtype=torch.float64)
    if variance.ndim != 0 or not torch.isfinite(variance) or variance <= 0:
        raise ValueError("noise_variance must be one finite positive scalar")
    axes = _dct_axes(shape, bias_order, observed)
    template, values = warped_template.flatten(1), observed.flatten(1).double()
    gram = torch.zeros(batch, coefficients, coefficients, device=observed.device, dtype=torch.float64)
    projection = torch.zeros(batch, coefficients, device=observed.device, dtype=torch.float64)
    for start in range(0, count, chunk_size):
        end = min(start + chunk_size, count)
        design = _design_chunk(template, start, end, axes)
        weights = mask[:, start:end, None] if mask is not None else 1
        gram = gram + design.transpose(1, 2) @ (design * weights)
        projection = projection + (design.transpose(1, 2) @ (values[:, start:end, None] * weights)).squeeze(-1)
    prior_cholesky = torch.linalg.cholesky(covariance)
    precision = torch.cholesky_inverse(prior_cholesky)
    posterior_cholesky = torch.linalg.cholesky(precision[None] + gram / variance)
    rhs = mean @ precision.T + projection / variance
    posterior_mean = torch.cholesky_solve(rhs[..., None], posterior_cholesky).squeeze(-1)
    posterior_covariance = torch.cholesky_inverse(posterior_cholesky)
    predictions = []
    residual_square = torch.zeros(batch, device=observed.device, dtype=torch.float64)
    for start in range(0, count, chunk_size):
        end = min(start + chunk_size, count)
        design = _design_chunk(template, start, end, axes)
        fitted = (design @ posterior_mean[..., None]).squeeze(-1)
        predictions.append(fitted)
        errors = (values[:, start:end] - fitted).square()
        if mask is not None:
            errors = errors * mask[:, start:end]
        residual_square = residual_square + errors.sum(dim=-1)
    displacement = posterior_mean - mean
    quadratic = residual_square / variance + (displacement @ precision * displacement).sum(dim=-1)
    log_determinant = (
        observed_count * variance.log()
        + 2 * prior_cholesky.diagonal().log().sum()
        + 2 * posterior_cholesky.diagonal(dim1=-2, dim2=-1).log().sum(dim=-1)
    )
    prior_z = (mean[:, 0] - gain_min) / covariance[0, 0].sqrt()
    posterior_z = (posterior_mean[:, 0] - gain_min) / posterior_covariance[:, 0, 0].sqrt()
    log_prob = (
        -0.5 * (observed_count * math.log(2 * math.pi) + log_determinant + quadratic)
        + torch.special.log_ndtr(posterior_z)
        - torch.special.log_ndtr(prior_z)
    )
    prediction = torch.cat(predictions, dim=-1).reshape_as(observed)
    return AppearanceMarginal(
        log_prob, posterior_mean, posterior_covariance, prediction, observed.double() - prediction
    )
