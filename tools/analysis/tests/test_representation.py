import copy

import numpy as np
import pytest
import torch
from scipy.ndimage import convolve
from torch import nn

from experiments.core.inference_runtime import load_checkpoint_state
from models.CTCF import model as cascade
from models.CTCF.configs import CONFIGS
from tools.analysis.tests.test_contrasts import TinyL2
from utils.image_representation import Mind6


@pytest.fixture(autouse=True)
def small_pool():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


def test_mind_matches_literal_gaussian_equation_and_replicated_boundary():
    array = np.random.default_rng(17).random((7, 8, 9), dtype=np.float32)
    padded = np.pad(array, 1, mode="edge")
    k = np.exp(-((np.arange(-2, 3) / 0.5) ** 2))
    kernel = np.einsum("i,j,k->ijk", k, k, k)
    kernel /= kernel.sum()
    distances = np.stack(
        [
            convolve((array - padded[1 + z : 8 + z, 1 + y : 9 + y, 1 + x : 10 + x]) ** 2, kernel, mode="nearest")
            for z, y, x in ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1))
        ]
    )
    expected = np.exp(-distances / distances.mean(0, keepdims=True).clip(1e-6))
    actual = Mind6()(torch.from_numpy(array)[None, None])
    np.testing.assert_allclose(actual[0].numpy(), expected, atol=2e-7, rtol=3e-6)


def test_mind_finite_on_constant_and_affine_intensity_invariant_away_from_floor():
    transform = Mind6()
    torch.testing.assert_close(transform(torch.zeros(2, 1, 7, 8, 9)), torch.ones(2, 6, 7, 8, 9))
    image = torch.rand(1, 1, 7, 8, 9, requires_grad=True)
    original = transform(image)
    torch.testing.assert_close(transform(2 * image + 3), original, atol=2e-6, rtol=1e-5)
    torch.testing.assert_close(transform(1 - image), original, atol=2e-6, rtol=1e-5)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        torch.testing.assert_close(transform(image), original, atol=0, rtol=0)
    original.square().mean().backward()
    assert torch.isfinite(image.grad).all() and image.grad.abs().sum() > 0


def test_cascade_uses_six_features_but_returns_raw_scalar_warp_and_binds_checkpoint(monkeypatch, tmp_path):
    monkeypatch.setattr(cascade, "_build_level2", lambda config, _: TinyL2(config))
    config = copy.deepcopy(CONFIGS["CTCF-CascadeA-Mamba"])
    config.img_size, config.level1_base_ch, config.level3_base_ch = (32, 32, 32), 2, 2
    config.l3_svf, config.input_representation = False, "mind6"
    model = cascade.CTCFCascadeA(config).eval()
    nn.init.normal_(model.level3.out.weight, std=0.002)
    x, y = torch.rand(1, 1, 32, 32, 32), torch.rand(1, 1, 32, 32, 32)
    output, field = model(x, y)
    assert output.shape == x.shape and model.level1.enc1.net[0].in_channels == 12
    torch.testing.assert_close(output, model.st_full(x, field), atol=0, rtol=0)
    (output - y).square().mean().backward()
    assert model.level3.out.weight.grad.abs().sum() > 0
    path = tmp_path / "mind.pth"
    torch.save({"state_dict": model.state_dict(), "input_representation": "mind6", "precision": "bf16"}, path)
    restored = cascade.CTCFCascadeA(config).eval()
    load_checkpoint_state(restored, str(path), True)
    torch.testing.assert_close(restored(x, y)[1], field, atol=0, rtol=0)
    config.input_representation = "raw"
    with pytest.raises(ValueError, match="representation"):
        load_checkpoint_state(cascade.CTCFCascadeA(config), str(path), True)
    config.input_representation, config.image_channels = "mind6", 2
    with pytest.raises(ValueError, match="scalar"):
        cascade.CTCFCascadeA(config)
