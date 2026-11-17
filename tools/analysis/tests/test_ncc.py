import pytest
import torch
from torch.autograd import gradcheck

from utils.losses import NCCVxm, local_ncc_map, stable_ncc_map


@pytest.fixture(autouse=True)
def cpu_pool():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


@pytest.mark.parametrize("value", [0.0, 0.9, 1.0])
def test_flat_windows_have_bounded_correlation_and_finite_gradients(value):
    image = torch.full((1, 1, 19, 19, 19), value, requires_grad=True)
    result = stable_ncc_map(image, image)
    assert result.min() >= 0 and result.max() <= 1
    assert result[0, 0, 9, 9, 9] == 0
    result.mean().backward()
    assert torch.isfinite(image.grad).all()


def test_stable_values_and_gradients_match_full_binary64_reference():
    torch.manual_seed(7)
    a, b = [torch.rand(1, 1, 7, 8, 9, dtype=torch.double, requires_grad=True) for _ in range(2)]
    # The legacy routine supports a cubic window; use one for the independent direct-convolution reference.
    actual = stable_ncc_map(a, b, win=(3, 3, 3))
    reference = local_ncc_map(a, b, win=(3, 3, 3))
    torch.testing.assert_close(actual, reference, atol=1e-13, rtol=1e-11)
    for got, expected in zip(
        torch.autograd.grad(actual.mean(), (a, b)), torch.autograd.grad(reference.mean(), (a, b)), strict=True
    ):
        torch.testing.assert_close(got, expected, atol=1e-13, rtol=1e-10)
    assert gradcheck(lambda x, y: stable_ncc_map(x, y, win=(3, 3, 3)), (a, b), fast_mode=True)


def test_legacy_mode_is_unchanged_and_stable_mode_is_explicit():
    torch.manual_seed(4)
    a, b = [torch.rand(1, 1, 9, 9, 9) for _ in range(2)]
    torch.testing.assert_close(NCCVxm()(a, b), -local_ncc_map(a, b).mean(), atol=0, rtol=0)
    torch.testing.assert_close(NCCVxm(mode="stable")(a, b), -stable_ncc_map(a, b).mean(), atol=0, rtol=0)
    with pytest.raises(ValueError):
        stable_ncc_map(a, b, win=(2, 3, 3))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_stable_ncc_survives_cuda_bf16_autocast_without_changing_precision():
    a = torch.full((1, 1, 19, 19, 19), 0.9, device="cuda", requires_grad=True)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        result = NCCVxm(mode="stable")(a, a)
    result.backward()
    assert -1 <= result <= 0 and result.dtype == torch.float32
    assert torch.isfinite(a.grad).all()


def test_stable_ncc_through_the_actual_training_sampler_preserves_legacy_gradient_units():
    from models.Matching import sample_voxels, voxel_grid
    from utils.field import legacy_sampling_displacement
    from utils.spatial import RegisterModel

    torch.manual_seed(31)
    shape = (9, 10, 11)
    moving, fixed = [torch.rand(1, 1, *shape, dtype=torch.double) + 0.1 for _ in range(2)]
    raw = (torch.rand(1, 3, *shape, dtype=torch.double) * 0.2 + 0.07).requires_grad_(True)
    sampler = RegisterModel(shape)
    warped = sampler((moving, raw))
    psi = legacy_sampling_displacement(raw).detach().requires_grad_(True)
    canonical = sample_voxels(moving, voxel_grid(shape, dtype=torch.double) + psi, padding="zeros")
    torch.testing.assert_close(warped, canonical, atol=1e-14, rtol=1e-12)
    stable = NCCVxm(win=(3, 3, 3), mode="stable")
    actual, other = stable(warped, fixed), stable(canonical, fixed)
    reference = -local_ncc_map(warped, fixed, win=(3, 3, 3)).mean()
    torch.testing.assert_close(actual, reference, atol=1e-13, rtol=1e-10)
    legacy_gradient = torch.autograd.grad(actual, raw)[0]
    psi_gradient = torch.autograd.grad(other, psi)[0]
    scale = raw.new_tensor([n / (n - 1) for n in shape]).view(1, 3, 1, 1, 1)
    torch.testing.assert_close(legacy_gradient, psi_gradient * scale, atol=1e-12, rtol=1e-9)
    ones, zeros = torch.ones_like(moving), torch.zeros_like(raw)
    assert sampler((ones, zeros))[0, 0, 0, 0, 0] == 0.125  # Zeros padding, not canonical border padding.
    variable = raw.detach().float().requires_grad_(True)
    loss = stable(sampler((moving.float(), variable)), fixed.float())
    assert torch.isfinite(torch.autograd.grad(loss, variable)[0]).all()
