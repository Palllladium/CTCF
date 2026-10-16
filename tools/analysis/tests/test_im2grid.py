from __future__ import annotations

import math
import pickle
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from experiments.core import matcher_training
from experiments.core.matcher_training import TrainingPairs, parameter_gradient_norm
from models.Im2grid import Im2gridLite, local_smoothness, to_canonical
from models.Im2grid.fetch import PORT, prepare_port
from models.MultiMatch.model import (
    MultiMatch,
    identity_coords,
    level_mismatch,
    level_smoothness,
    sample_at,
    window_offsets,
)
from utils.field import _warp

needs_port = pytest.mark.skipif(not (PORT / "model.py").is_file(), reason="python -m models.Im2grid.fetch first")


def _normalized_identity(shape):
    axes = [torch.arange(n).float() / (n - 1) * 2 - 1 for n in shape]
    return torch.stack(torch.meshgrid(*axes, indexing="ij"))[None]


def _voxel_identity(shape):
    axes = [torch.arange(n, dtype=torch.float32) for n in shape]
    return torch.stack(torch.meshgrid(*axes, indexing="ij"))[None]


def test_smoothness_uses_native_displacements_and_is_translation_invariant():
    grids = [_normalized_identity(shape) for shape in ((4, 5, 6), (8, 10, 12))]
    assert local_smoothness(grids).item() == 0.0
    assert local_smoothness([g + 0.1 for g in grids]).item() < 1e-12
    ramp = grids[0].clone()
    ramp[:, 0] += torch.arange(4)[:, None, None] * 0.03
    assert abs(local_smoothness([ramp]).item() - 0.03**2 / 3) < 1e-8


def test_port_preparation_fails_when_an_edit_no_longer_applies():
    with pytest.raises(RuntimeError, match="no longer contains"):
        prepare_port("class Im2gridDecoder:\n    pass\n", "def grid_sampler():\n    pass\n")


@needs_port
def test_canonical_export_reproduces_upstream_sampling_including_non_cubic_axes():
    from models.Im2grid.vfa_port.sampling import grid_sampler

    torch.manual_seed(2)
    image = torch.rand((1, 2, 8, 10, 12))
    grid = _normalized_identity(image.shape[2:]) * 0.8 + 0.01 * torch.randn((1, 3, 8, 10, 12))
    absolute = (grid + 1) * 0.5 * (torch.tensor(image.shape[2:]).view(1, 3, 1, 1, 1) - 1)
    assert torch.allclose(_warp(image, to_canonical(absolute)), grid_sampler(image, absolute), atol=2e-6)


@needs_port
def test_both_losses_reach_parameters_while_the_skipped_level_is_inactive():
    torch.manual_seed(0)
    threads = torch.get_num_threads()
    torch.set_num_threads(2)
    try:
        model = Im2gridLite()
        mse, regularity = model.losses(torch.rand((1, 1, 32, 32, 32)), torch.rand((1, 1, 32, 32, 32)))
        parameters = list(model.parameters())
        assert parameter_gradient_norm(mse, parameters) > 0 and parameter_gradient_norm(regularity, parameters) > 0
        unused = torch.autograd.grad(regularity, model.decoder.positional_embedding[0].parameters(), allow_unused=True)
        assert all(g is None for g in unused)
    finally:
        torch.set_num_threads(threads)


@needs_port
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA autocast check")
def test_autocast_keeps_local_and_composed_coordinates_in_float32():
    model = Im2gridLite().cuda()
    image = torch.rand((1, 1, 32, 32, 32), device="cuda")
    with torch.autocast("cuda", dtype=torch.float16):
        grid, local = model(image, image)
    assert grid.dtype == torch.float32 and all(g.dtype == torch.float32 for g in local)
    assert [tuple(g.shape[2:]) for g in local] == [(2, 2, 2), (4, 4, 4), (8, 8, 8), (16, 16, 16), (32, 32, 32)]


def _pickled_images(tmp_path, count, shape, make):
    files = []
    for index in range(count):
        path = tmp_path / f"{index}.pkl"
        with path.open("wb") as stream:
            pickle.dump((make(index, shape),), stream)
        files.append(path)
    return files


def test_pair_stream_and_common_flips_resume_at_the_same_attempt(tmp_path):
    files = _pickled_images(
        tmp_path, 3, (8, 10, 12), lambda i, s: np.arange(np.prod(s), dtype=np.float32).reshape(s) + i * 1000
    )
    original, resumed = TrainingPairs(files, seed=7, start=0, length=8), TrainingPairs(files, seed=7, start=5, length=3)
    for index in range(3):
        assert all(torch.equal(a, b) for a, b in zip(original[index + 5], resumed[index], strict=True))


def test_shared_trainer_counts_successful_updates_and_resumes_the_same_stream(tmp_path, monkeypatch):
    class Scaler:  # skips the first update, as an AMP overflow would
        def __init__(self, *args, **kwargs):
            self.calls, self.value = 0, 4.0

        def is_enabled(self):
            return True

        def get_scale(self):
            return self.value

        def scale(self, loss):
            return loss

        def step(self, optimizer):
            self.calls += 1
            if self.calls > 1:
                optimizer.step()

        def update(self):
            if self.calls == 1:
                self.value = 2.0

        def state_dict(self):
            return vars(self).copy()

        def load_state_dict(self, state):
            self.__dict__.update(state)

    files = _pickled_images(tmp_path, 3, (4, 4, 4), lambda i, s: np.random.default_rng(i).random(s, dtype=np.float32))
    monkeypatch.setattr(matcher_training, "training_files", lambda args: files)
    monkeypatch.setattr(torch.amp, "GradScaler", Scaler)
    args = SimpleNamespace(seed=0, lr=3e-4, weight_decay=1e-9, crop=None, num_workers=0)

    def build(args):
        return torch.nn.Conv3d(1, 1, 1)

    def losses(model, moving, fixed):
        return (model(moving) - fixed).square().mean(), model.weight.square().mean()

    directory = tmp_path / "ckpt"
    directory.mkdir()
    device, method = torch.device("cpu"), {"name": "stub"}
    _, counts = matcher_training.train(args, device, 0.1, 2, build, losses, method, directory)
    assert counts["updates"] == 2 and counts["attempts"] == 3 and counts["skipped"] == 1
    resumed, _ = matcher_training.train(args, device, 0.1, 4, build, losses, method, directory)
    reference, _ = matcher_training.train(args, device, 0.1, 4, build, losses, method)
    assert all(torch.equal(a, b) for a, b in zip(resumed.parameters(), reference.parameters(), strict=True))


def test_both_modes_issue_the_same_number_of_distinct_queries():
    small, large = window_offsets("27"), window_offsets("81")
    assert len(large) == 3 * len(small) == 81 and len({tuple(o.tolist()) for o in large}) == 81
    assert (small.abs() <= 1).all()


def test_sampling_keeps_fractional_coordinates_under_autocast():
    image = torch.arange(5.0).view(1, 1, 1, 1, 5).expand(1, 1, 5, 5, 5)
    coords = torch.tensor([[[2.0, 2.0, 2.12345]]])
    expected = sample_at(image, coords)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        assert torch.equal(sample_at(image, coords), expected)


def _probe(peaks):
    """Level-1 matching of one voxel whose window logits are given (z, y, x offsets -> logit); identity projections,
    no compatibility pass (the flat neighbours would reweight the masses)."""
    model = MultiMatch("D3", levels=1, start=8, max_channels=16, matching_channels=16, compat_weight=0.0).eval()
    with torch.no_grad():
        embedding = model.decoder.positional_embedding[1]
        embedding.tensor_proj.weight.copy_(torch.eye(16).view(16, 16, 1, 1, 1))
        embedding.tensor_proj.bias.zero_()
        embedding.grid_scale.zero_()
    spatial = (5, 5, 5)
    fixed, moving = torch.zeros(1, 16, *spatial), torch.zeros(1, 16, *spatial)
    fixed[0, 0, 2, 2, 2] = 1.0
    for offset, value in peaks.items():
        moving[(0, 0, *(2 + o for o in offset))] = value
    coords = _voxel_identity(spatial).permute(0, 2, 3, 4, 1)[:, None].expand(1, 3, *spatial, 3).clone()
    logmass = torch.full((1, 3, *spatial), -1e9)
    logmass[:, 0] = 0.0
    with torch.no_grad():
        coords, logmass, _ = model.decoder._level(1, fixed, moving, coords, logmass)
    alive = logmass[0, :, 2, 2, 2] > -1e8
    return coords[0, :, 2, 2, 2][alive], torch.softmax(logmass[0, :, 2, 2, 2][alive], 0)


@needs_port
def test_hypotheses_are_peaks_only():
    coords, _ = _probe({(0, 0, 1): 9.0})
    assert len(coords) == 1 and (coords[0] - torch.tensor([2.0, 2.0, 3.0])).abs().max() < 0.05
    coords, mass = _probe({(0, 0, 1): 9.0, (0, 0, -1): 8.0})
    assert len(coords) == 2 and mass[0] > mass[1]
    assert (coords[0] - torch.tensor([2.0, 2.0, 3.0])).abs().max() < 0.05
    assert (coords[1] - torch.tensor([2.0, 2.0, 1.0])).abs().max() < 0.05
    coords, _ = _probe({})  # a flat window: one mode, no invented ones
    assert len(coords) == 1


@needs_port
def test_gradients_reach_masses_and_coordinates():
    torch.manual_seed(0)
    model = MultiMatch("D3", levels=2, start=4, max_channels=8, matching_channels=8)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    fixed, moving = torch.rand(1, 1, 16, 16, 16), torch.rand(1, 1, 16, 16, 16)
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        flow, states = model(moving, fixed)
        assert flow.shape == (1, 3, 16, 16, 16)
        (level_mismatch(states, fixed, moving) + level_smoothness(states)).backward()
        assert model.decoder.positional_embedding[1].tensor_proj.weight.grad.abs().sum() > 0
        optimizer.step()
    assert model.encoder.in_conv.weight.grad.abs().sum() > 0
    assert model.decoder.positional_embedding[2].grid_scale.grad.abs().sum() > 0
    coords = _voxel_identity((8, 8, 8)).permute(0, 2, 3, 4, 1)[:, None].repeat(1, 2, 1, 1, 1, 1)
    coords[:, 1, ..., 0] += 0.3
    coords.requires_grad_()
    masses = torch.zeros((1, 2, 8, 8, 8), requires_grad=True)
    level_mismatch([(1, coords, masses)], fixed, moving).backward()
    assert coords.grad.abs().sum() > 0 and masses.grad.abs().sum() > 0


@needs_port
def test_h1_starts_from_the_same_upstream_encoder_and_positional_parameters():
    from models.Im2grid.vfa_port.model import Im2gridDecoder, Im2gridEncoder

    torch.manual_seed(17)
    native = Im2gridLite().state_dict()
    for mode in ("E3", "D3"):
        torch.manual_seed(17)
        model = MultiMatch(mode)
        assert isinstance(model.encoder, Im2gridEncoder) and isinstance(model.decoder, Im2gridDecoder)
        assert native.keys() == model.state_dict().keys()
        assert all(torch.equal(value, model.state_dict()[key]) for key, value in native.items())


@needs_port
def test_one_mode_matches_upstream_at_the_identity_and_after_a_nonlinear_parent():
    from models.Im2grid.vfa_port.model import Im2gridDecoder
    from models.Im2grid.vfa_port.sampling import grid_sampler
    from models.MultiMatch.model import MultiModeDecoder

    torch.manual_seed(19)
    source = Im2gridDecoder(0, 8, 8, 8, skip=set())
    with torch.no_grad():
        source.positional_embedding[0].tensor_proj.weight.normal_(std=0.001)
    model = MultiModeDecoder("D3", 0, 8, 8, 8, 1, 1.0, 0.0)
    model.load_state_dict(source.state_dict())
    fixed, moving = torch.rand((1, 8, 6, 7, 8)), torch.rand((1, 8, 6, 7, 8))
    logmass = torch.zeros((1, 1, *moving.shape[2:]))
    identity = _voxel_identity(moving.shape[2:])
    parent = identity + 0.2 * identity.sin()
    sizes = parent.new_tensor(parent.shape[2:]).view(1, 3, 1, 1, 1)
    with torch.no_grad():
        expected = source([fixed], [moving], None)[0][-1]
        actual = model._level(0, fixed, moving, identity.permute(0, 2, 3, 4, 1)[:, None], logmass)[0]
        assert torch.allclose(actual[:, 0].permute(0, 4, 1, 2, 3), expected, atol=2e-6)
        warped = grid_sampler(moving, parent / (sizes - 1) * 2 - 1, normalize=False)
        expected = grid_sampler(parent, source([fixed], [warped], None)[1][-1], normalize=False)
        actual = model._level(0, fixed, moving, parent.permute(0, 2, 3, 4, 1)[:, None], logmass)[0]
        assert torch.allclose(actual[:, 0].permute(0, 4, 1, 2, 3), expected, atol=2e-6)


@needs_port
def test_compatibility_does_not_connect_opposite_faces():
    model = MultiMatch("D3", levels=1)
    coords = identity_coords((4, 4, 4), "cpu")[None, None].repeat(1, 3, 1, 1, 1, 1)
    mass = torch.zeros(1, 3, 4, 4, 4)
    before = model.decoder._compatibility(coords, mass, passes=1)
    coords[:, 1, -1, ..., 0] += 100
    assert torch.equal(before[:, :, 0], model.decoder._compatibility(coords, mass, passes=1)[:, :, 0])


@needs_port
def test_lineage_diagnostics_compare_actual_coarse_origins_within_shared_cells():
    model = MultiMatch("D3", levels=1, start=4, max_channels=8)
    origins = torch.full((1, 2, 2, 2), 2)
    origins[0, 0, 0, 0] = 1
    model.decoder._lineage_diagnostics(
        torch.zeros((1, 3, 2, 2, 2)), origins, torch.full_like(origins, 2), torch.zeros_like(origins)
    )
    assert model.diagnostics["coarse_origin_switch_fraction"] == 1 / 8
    assert model.diagnostics["mode_switch_fraction_within_coarse_cell"] == 3 / 12
    assert model.diagnostics["mode_switch_eligible_edges"] == 12
    assert model.diagnostics["displacement_jump_max_voxels"] == 0
    model.eval()
    with torch.no_grad():
        model(torch.rand((1, 1, 8, 8, 8)), torch.rand((1, 1, 8, 8, 8)))
    assert all(math.isfinite(value) for value in model.diagnostics.values())
