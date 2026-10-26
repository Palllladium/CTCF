import math

import pytest
import torch

from experiments.population.geometry import (
    coefficient_bands,
    haar_forward,
    haar_inverse,
    pad_velocity,
    subject_pair_flow,
)


@pytest.mark.parametrize("shape", [(8, 10, 12), (9, 11, 13), (1, 7, 9), (7, 1, 5), (3, 5, 1)])
def test_haar_preserves_values_energy_and_gradient_for_non_dyadic_shapes(shape):
    x = torch.randn((2, 3, *shape), generator=torch.Generator().manual_seed(19), dtype=torch.float64)
    x.requires_grad_()
    pyramid = haar_forward(x, levels=2)
    reconstructed = haar_inverse(pyramid)
    torch.testing.assert_close(reconstructed, x, rtol=1e-12, atol=1e-12)
    energy = sum(band.square().sum() for band in coefficient_bands(pyramid))
    torch.testing.assert_close(energy, x.square().sum(), rtol=1e-12, atol=1e-12)
    gradient = torch.autograd.grad(energy, x)[0]
    torch.testing.assert_close(gradient, 2 * x, rtol=1e-12, atol=1e-12)


def test_haar_uses_orthonormal_block_scaling():
    pyramid = haar_forward(torch.ones(1, 1, 2, 2, 2, dtype=torch.float64), levels=1)
    assert pyramid.coarse.item() == pytest.approx(math.sqrt(8), abs=1e-14)
    for detail in coefficient_bands(pyramid)[1:]:
        assert torch.count_nonzero(detail) == 0


def test_velocity_padding_fixes_the_collar_and_preserves_interior_gradients():
    interior = torch.ones(1, 3, 7, 9, 11, requires_grad=True)
    velocity = pad_velocity(interior, collar=2)
    assert velocity.shape == (1, 3, 11, 13, 15)
    torch.testing.assert_close(velocity[:, :, 2:-2, 2:-2, 2:-2], interior)
    for axis in range(2, 5):
        assert torch.count_nonzero(velocity.narrow(axis, 0, 2)) == 0
        assert torch.count_nonzero(velocity.narrow(axis, velocity.shape[axis] - 2, 2)) == 0
    velocity.sum().backward()
    torch.testing.assert_close(interior.grad, torch.ones_like(interior))


def test_zero_subject_maps_produce_exact_identity():
    velocity = torch.zeros(1, 3, 13, 15, 17)
    assert torch.equal(subject_pair_flow(velocity, velocity), velocity)


def test_same_smooth_subject_maps_cancel_to_interpolation_accuracy_inside():
    axes = [torch.linspace(0, math.pi, n) for n in (17, 19, 21)]
    zz, yy, xx = torch.meshgrid(*axes, indexing="ij")
    envelope = zz.sin() * yy.sin() * xx.sin()
    velocity = torch.stack((envelope, envelope * xx.cos(), envelope * yy.cos()))[None] * 0.05
    residual = subject_pair_flow(velocity, velocity, steps=7)[:, :, 3:-3, 3:-3, 3:-3]
    assert residual.abs().max() < 3e-4


def test_pair_direction_is_fixed_to_canonical_then_canonical_to_moving():
    fixed = torch.zeros(1, 3, 15, 17, 19)
    moving = torch.zeros_like(fixed)
    fixed[:, 2] = 0.2
    moving[:, 2] = -0.1
    flow = subject_pair_flow(fixed, moving, steps=7)[:, :, 3:-3, 3:-3, 3:-3]
    expected = torch.zeros_like(flow)
    expected[:, 2] = 0.3
    torch.testing.assert_close(flow, expected, rtol=1e-5, atol=1e-6)
