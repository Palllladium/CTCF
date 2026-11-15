from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F
from torch.autograd import gradcheck, gradgradcheck

from models.Matching import (
    FeatureEncoder,
    candidate_logits,
    flow_inside_fraction,
    matching_cost,
    matching_energy,
    refine,
    resize_flow,
    sample_voxels,
    voxel_grid,
)


@pytest.fixture(autouse=True)
def small_cpu_pool():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


def native_sample(image, coordinates, padding):
    normalized = torch.stack([2 * coordinates[:, axis] / (image.shape[axis + 2] - 1) - 1 for axis in (2, 1, 0)], dim=-1)
    grid = normalized.reshape(image.shape[0], -1, 1, 1, 3)
    values = F.grid_sample(image, grid, padding_mode=padding, align_corners=True)
    return values.reshape(image.shape[0], image.shape[1], *coordinates.shape[2:])


@pytest.mark.parametrize("padding", ["zeros", "border"])
def test_sampler_values_first_and_mixed_second_derivatives(padding):
    torch.manual_seed(11)
    image = torch.randn(1, 2, 4, 5, 6, dtype=torch.double, requires_grad=True)
    points = torch.tensor([[[0.23, -0.27, 2.31], [1.17, 2.29, 3.37], [2.43, 4.11, 1.53]]], dtype=torch.double)
    points.requires_grad_()
    actual = sample_voxels(image, points, padding)
    expected = native_sample(image, points, padding)
    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
    upstream = torch.randn_like(actual)
    first = torch.autograd.grad(actual, (image, points), upstream, retain_graph=True)
    reference = torch.autograd.grad(expected, (image, points), upstream, retain_graph=True)
    for got, wanted in zip(first, reference, strict=True):
        torch.testing.assert_close(got, wanted, rtol=1e-11, atol=1e-11)
    function = lambda x, p: sample_voxels(x, p, padding)  # noqa: E731
    assert gradcheck(function, (image, points), fast_mode=True)
    assert gradgradcheck(function, (image, points), fast_mode=True)


@pytest.mark.parametrize("padding", ["zeros", "border"])
def test_sampler_grid_boundary_convention_matches_native(padding):
    image = torch.arange(60, dtype=torch.double).reshape(1, 1, 3, 4, 5).requires_grad_()
    points = torch.tensor([[[0.0, 2.0, -0.2, 2.3], [1.2, 2.3, 1.2, 1.2], [2.2, 3.3, 2.2, 2.2]]], dtype=torch.double)
    points.requires_grad_()
    actual, expected = sample_voxels(image, points, padding), native_sample(image, points, padding)
    torch.testing.assert_close(actual, expected)
    actual_gradient = torch.autograd.grad(actual.sum(), points)[0]
    expected_gradient = torch.autograd.grad(expected.sum(), points)[0]
    torch.testing.assert_close(actual_gradient, expected_gradient, atol=1e-12, rtol=1e-12)


def test_flow_resizing_preserves_anisotropic_affine_maps():
    grid = voxel_grid((5, 7, 9), dtype=torch.double)
    factors = torch.tensor([0.13, -0.07, 0.19], dtype=torch.double).view(1, 3, 1, 1, 1)
    original = grid * factors + 0.31
    resized = resize_flow(original, (4, 6, 8))
    restored = resize_flow(resized, original.shape[2:])
    torch.testing.assert_close(restored, original, atol=1e-14, rtol=1e-14)
    translated = resize_flow(torch.ones_like(original), (4, 6, 8))
    expected = original.new_tensor([3 / 4, 5 / 6, 7 / 8]).view(1, 3, 1, 1, 1).expand_as(translated)
    torch.testing.assert_close(translated, expected)


def test_encoder_uses_endpoint_aligned_input_geometry_and_normalized_features():
    torch.manual_seed(7)
    encoder = FeatureEncoder(2, width=3, channels=5, stride=2)
    image = torch.rand(1, 2, 17, 19, 21)
    seen = []
    handle = encoder.stem[0].register_forward_pre_hook(lambda _, args: seen.append(args[0]))
    features = encoder(image)
    handle.remove()
    torch.testing.assert_close(seen[0], image)
    assert features.shape == (1, 5, 8, 9, 10)
    torch.testing.assert_close(features.square().sum(1), torch.ones_like(features[:, 0]), atol=2e-6, rtol=2e-6)


def test_descriptor_resolution_preserves_native_voxel_cost_and_field_gradient():
    grid = voxel_grid((5, 7, 9), dtype=torch.double)
    features = torch.cat((grid / 9, torch.ones_like(grid[:, :1])), dim=1)
    coarse = F.interpolate(features, (3, 4, 5), mode="trilinear", align_corners=True)
    field = torch.full_like(grid, 0.17).requires_grad_()
    native = matching_cost(features, features, field)
    half = matching_cost(coarse, coarse, field)
    torch.testing.assert_close(native, half, atol=1e-14, rtol=1e-12)
    first = torch.autograd.grad(native.sum(), field)[0]
    second = torch.autograd.grad(half.sum(), field)[0]
    torch.testing.assert_close(first, second, atol=1e-13, rtol=1e-11)


def test_checkpointed_multiscale_encoder_preserves_the_solver_hypergradient():
    torch.manual_seed(93)
    encoder = FeatureEncoder(1, width=2, channels=4, stride=1, precision="fp32").double()
    image, change = [torch.randn(1, 1, 16, 16, 16, dtype=torch.double) for _ in range(2)]
    fixed = encoder(torch.rand_like(image)).detach()
    field = torch.full((1, 3, 16, 16, 16), 0.17, dtype=torch.double)

    def objective(theta):
        moving = encoder(image + theta * change)
        refined = refine(fixed, moving, field, steps=2, step_size=0.01, max_update=0.1)
        return refined.square().mean()

    theta = torch.tensor(0.13, dtype=torch.double, requires_grad=True)
    actual = torch.autograd.grad(objective(theta), theta)[0]
    step = 1e-5
    reference = (objective(theta.detach() + step) - objective(theta.detach() - step)) / (2 * step)
    torch.testing.assert_close(actual, reference, atol=1e-8, rtol=5e-3)
    assert actual.abs() > 1e-8
    encoder.checkpoint_blocks = False
    without = torch.autograd.grad(objective(theta), theta)[0]
    torch.testing.assert_close(actual, without, atol=1e-12, rtol=1e-9)


def test_candidate_distribution_uses_the_solver_cost():
    torch.manual_seed(4)
    fixed, moving = [F.normalize(torch.randn(1, 4, 3, 4, 5, dtype=torch.double), dim=1) for _ in range(2)]
    flow = torch.full((1, 3, 3, 4, 5), 0.17, dtype=torch.double)
    logits = candidate_logits(fixed, moving, flow, [[0, 0, 0], [0.2, -0.1, 0.3]], temperature=0.3)
    torch.testing.assert_close(logits[:, :1], -matching_cost(fixed, moving, flow) / 0.3)
    points = voxel_grid((3, 4, 5), dtype=torch.double).flatten(2)[:, :, [7, 16, 43]]
    sparse = candidate_logits(fixed, moving, flow, [[0, 0, 0]], temperature=0.3, points=points)
    torch.testing.assert_close(sparse, logits[:, :1].flatten(2)[:, :, [7, 16, 43]])


def test_exact_solver_hypergradient_matches_finite_difference():
    torch.manual_seed(9)
    shape = (3, 4, 5)
    fixed, moving, change = [torch.randn(1, 3, *shape, dtype=torch.double) for _ in range(3)]
    initial = torch.full((1, 3, *shape), 0.21, dtype=torch.double)
    target = torch.full_like(initial, -0.12)
    mask = torch.zeros((1, 1, *shape), dtype=torch.double)
    mask[:, :, 1:-1, 1:-1, 1:-1] = 1

    def objective(theta):
        features = F.normalize(moving + theta * change, dim=1)
        output = refine(
            F.normalize(fixed, dim=1), features, initial, steps=3, step_size=0.01, max_update=0.2, mask=mask
        )
        return (output - target).square().mean()

    theta = torch.tensor(0.13, dtype=torch.double, requires_grad=True)
    actual = torch.autograd.grad(objective(theta), theta)[0]
    epsilon = 1e-5
    expected = (objective(theta.detach() + epsilon) - objective(theta.detach() - epsilon)) / (2 * epsilon)
    assert actual.abs() > 1e-7
    torch.testing.assert_close(actual, expected, atol=1e-8, rtol=2e-4)


def test_solver_update_is_independent_of_repeated_batch_size():
    torch.manual_seed(2)
    fixed, moving = [F.normalize(torch.randn(1, 4, 3, 4, 5), dim=1) for _ in range(2)]
    initial = torch.full((1, 3, 3, 4, 5), 0.17)
    one = refine(fixed, moving, initial, steps=2, create_graph=False)
    two = refine(fixed.repeat(2, 1, 1, 1, 1), moving.repeat(2, 1, 1, 1, 1), initial.repeat(2, 1, 1, 1, 1), steps=2)
    torch.testing.assert_close(two, one.repeat(2, 1, 1, 1, 1))


def test_outside_penalty_recovers_from_flat_border_cost():
    features = F.normalize(torch.ones(1, 2, 3, 4, 5), dim=1)
    initial = torch.full((1, 3, 3, 4, 5), 7.0)
    no_guard = refine(features, features, initial, steps=1, reg_weight=0, outside_weight=0, create_graph=False)
    guarded = refine(features, features, initial, steps=1, reg_weight=0, outside_weight=1, create_graph=False)
    torch.testing.assert_close(no_guard, initial)
    assert guarded.abs().mean() < initial.abs().mean()
    assert flow_inside_fraction(initial) == 0


def test_solver_reduces_energy_for_a_known_small_translation():
    grid = voxel_grid((3, 4, 9), dtype=torch.double)
    moving = F.normalize(torch.cat((grid[:, 2:3] / 8, torch.ones_like(grid[:, :1])), dim=1), dim=1)
    target = torch.zeros_like(grid)
    target[:, 2] = 0.4
    fixed = F.normalize(sample_voxels(moving, grid + target), dim=1)
    initial = torch.zeros_like(target)
    mask = torch.zeros_like(fixed[:, :1])
    mask[:, :, 1:-1, 1:-1, 1:-1] = 1
    start = matching_energy(fixed, moving, initial, reg_weight=0, mask=mask, outside_weight=0)
    output = refine(
        fixed,
        moving,
        initial,
        steps=40,
        step_size=1,
        reg_weight=0,
        outside_weight=0,
        max_update=0.1,
        mask=mask,
        create_graph=False,
    )
    end = matching_energy(fixed, moving, output, reg_weight=0, mask=mask, outside_weight=0)
    assert end < start / 2
    assert output[:, 2:3][mask.bool()].mean() > 0.2


def test_invalid_geometry_and_empty_support_fail_explicitly():
    image = torch.ones(1, 1, 3, 4, 5)
    points = voxel_grid((3, 4, 5))
    points[:, 0, 0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        sample_voxels(image, points)
    flow = torch.zeros(1, 3, 3, 4, 5)
    with pytest.raises(ValueError, match="empty"):
        matching_energy(image, image, flow, mask=torch.zeros_like(image))
    with pytest.raises(ValueError, match="dimensions"):
        resize_flow(flow, (1, 2, 3))
