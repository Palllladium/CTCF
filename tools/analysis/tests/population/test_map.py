import math

import pytest
import torch

from experiments.population.evaluate import fit_subject, velocity_energy, velocity_rms_by_scale
from experiments.population.geometry import coefficient_bands, haar_forward
from experiments.population.model import PopulationAtlas
from experiments.population.train import observation_nll


@pytest.fixture(autouse=True)
def _one_cpu_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.mark.parametrize("shape", [(8, 10, 12), (9, 11, 13)])
def test_native_velocity_energy_and_gradients_match_the_coefficient_objective(shape):
    torch.manual_seed(13)
    template = torch.randn(1, 1, *shape)
    observed = torch.randn_like(template)
    model = PopulationAtlas(
        template,
        levels=2,
        collar=2,
        width=2,
        arm="A1",
        scale_min=0.001,
        scale_max=1.0,
        prior_std=0.2,
        posterior_std=0.1,
        integration_steps=2,
    )
    model.requires_grad_(False)
    interior = (torch.randn(1, 3, *(size - 4 for size in shape)) * 0.03).requires_grad_()
    mean = torch.zeros(28, dtype=torch.float64)
    mean[0] = 0.8
    covariance = torch.eye(28, dtype=torch.float64) * 0.1
    coefficients = haar_forward(interior, model.levels)
    old_data = observation_nll(model, observed, coefficients, 0.3, mean, covariance, 0.01).sum()
    old_prior = -model.log_prior(coefficients).sum()
    data, prior = velocity_energy(model, observed, interior, 0.3, mean, covariance, 0.01)
    torch.testing.assert_close(data + prior, old_data + old_prior, rtol=2e-6, atol=1e-5)
    old_gradient = torch.autograd.grad(old_data + old_prior, interior, retain_graph=True)[0]
    new_gradient = torch.autograd.grad(data + prior, interior)[0]
    torch.testing.assert_close(new_gradient, old_gradient, rtol=3e-5, atol=3e-5)


@pytest.mark.parametrize("shape", [(8, 8, 8), (9, 7, 5)])
def test_velocity_map_recovers_coarse_and_detail_motion_on_even_and_odd_grids(shape, monkeypatch):
    class ToyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.marker = torch.nn.Parameter(torch.tensor(1.0))
            self.collar, self.levels, self.integration_steps = 1, 2, 1

        def posterior_mean(self, observed):
            return haar_forward(observed.new_zeros(1, 3, *shape), self.levels)

    model = ToyModel()
    _zz, yy, xx = torch.meshgrid(*(torch.arange(size) for size in shape), indexing="ij")
    target = (1 + 0.4 * torch.sin(2 * math.pi * xx / 4) + 0.2 * torch.cos(math.pi * yy))[None, None].expand(
        1, 3, *shape
    )
    optimum = target / 1.1

    def quadratic(model, observed, interior, *unused):
        data = 0.5 * (interior - target).square().sum()
        prior = 0.05 * sum(band.square().sum() for band in coefficient_bands(haar_forward(interior, 2)))
        return data, prior

    monkeypatch.setattr("experiments.population.evaluate.velocity_energy", quadratic)
    observed = torch.zeros(1, 1, *(size + 2 for size in shape))
    velocity, _, report = fit_subject(model, observed, None, None, None, None, steps=180, lr=0.1, log_every=30)
    error = velocity[:, :, 1:-1, 1:-1, 1:-1] - optimum
    assert max(velocity_rms_by_scale(error, 2)) < 2e-4
    assert report["inference_mode"] == "velocity_map"
    assert report["best_velocity_change_rms_by_scale"][0] > 0.5
    assert report["best_velocity_change_rms_by_scale"][-1] > 0.05
    assert report["trace"][-1]["previous_update_rms"] > 0
    assert model.marker.requires_grad is True
