import math
from itertools import product

import numpy as np
import pytest
import torch
from scipy.integrate import quad
from scipy.stats import multivariate_normal, norm

from experiments.population.appearance import appearance_marginal


def _dense_design(template, order):
    shape = template.shape[-3:]
    axes = torch.meshgrid(*(torch.arange(n, dtype=torch.float64) for n in shape), indexing="ij")
    basis = []
    for frequencies in product(range(order + 1), repeat=3):
        if frequencies == (0, 0, 0):
            continue
        values = torch.ones_like(axes[0])
        for frequency, coordinates, size in zip(frequencies, axes, shape, strict=True):
            values = values * torch.cos(math.pi * frequency * (coordinates + 0.5) / size)
            if frequency:
                values = values * math.sqrt(2)
        basis.append(values.flatten())
    nuisance = torch.stack((torch.ones(math.prod(shape), dtype=torch.float64), *basis), dim=-1)
    return torch.cat((template.flatten(1)[..., None], nuisance.expand(template.shape[0], -1, -1)), dim=-1)


def test_marginal_matches_dense_covariance_and_its_gradients():
    generator = torch.Generator().manual_seed(7)
    template = torch.randn(2, 1, 3, 4, 5, generator=generator, dtype=torch.float64, requires_grad=True)
    observed = torch.randn(2, 1, 3, 4, 5, generator=generator, dtype=torch.float64, requires_grad=True)
    variance = torch.tensor(0.3, dtype=torch.float64, requires_grad=True)
    matrix = torch.randn(28, 28, generator=generator, dtype=torch.float64) * 0.02
    covariance = matrix @ matrix.T + torch.eye(28, dtype=torch.float64) * 0.1
    mean = torch.zeros(28, dtype=torch.float64)
    mean[0] = 0.8
    fitted = appearance_marginal(observed, template, variance, mean, covariance, 0.1, chunk_size=7)
    design = _dense_design(template, order=2)
    dense_covariance = variance * torch.eye(60, dtype=torch.float64) + design @ covariance @ design.transpose(1, 2)
    gaussian = torch.distributions.MultivariateNormal(design @ mean, covariance_matrix=dense_covariance)
    posterior_covariance = torch.linalg.inv(torch.linalg.inv(covariance) + design.transpose(1, 2) @ design / variance)
    posterior_mean = (
        posterior_covariance
        @ (
            torch.linalg.solve(covariance, mean)[None]
            + (design.transpose(1, 2) @ observed.flatten(1)[..., None]).squeeze(-1) / variance
        )[..., None]
    ).squeeze(-1)
    expected = gaussian.log_prob(observed.flatten(1))
    expected = expected + torch.special.log_ndtr((posterior_mean[:, 0] - 0.1) / posterior_covariance[:, 0, 0].sqrt())
    expected = expected - torch.special.log_ndtr((mean[0] - 0.1) / covariance[0, 0].sqrt())
    torch.testing.assert_close(fitted.log_prob, expected, atol=1e-9, rtol=1e-10)
    torch.testing.assert_close(fitted.posterior_mean, posterior_mean, atol=1e-10, rtol=1e-10)
    actual_grad = torch.autograd.grad(fitted.log_prob.sum(), (template, observed, variance), retain_graph=True)
    expected_grad = torch.autograd.grad(expected.sum(), (template, observed, variance))
    for actual, reference in zip(actual_grad, expected_grad, strict=True):
        torch.testing.assert_close(actual, reference, atol=1e-8, rtol=1e-8)


def test_positive_gain_normalizer_matches_direct_numerical_integration():
    template = torch.linspace(-1, 1, 8, dtype=torch.float64).reshape(1, 1, 2, 2, 2)
    observed = template * 0.5 + 0.2
    mean = torch.tensor([1.0, 0.1], dtype=torch.float64)
    covariance = torch.diag(torch.tensor([0.36, 0.04], dtype=torch.float64))
    fitted = appearance_marginal(observed, template, 0.3, mean, covariance, 0.2, bias_order=0)
    conditional_covariance = np.eye(8) * 0.3 + np.ones((8, 8)) * 0.04

    def integrand(gain):
        conditional_mean = template.numpy().flatten() * gain + 0.1
        return (
            multivariate_normal.pdf(observed.numpy().flatten(), conditional_mean, conditional_covariance)
            * norm.pdf(gain, 1.0, 0.6)
            / norm.sf(0.2, 1.0, 0.6)
        )

    density, _ = quad(integrand, 0.2, np.inf, epsabs=1e-12)
    assert fitted.log_prob.item() == pytest.approx(math.log(density), abs=1e-9)
