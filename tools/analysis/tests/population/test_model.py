import math

import pytest
import torch

from experiments.population.geometry import coefficient_bands, haar_forward
from experiments.population.model import PopulationAtlas, _normal_kl, _normal_log_prob


@pytest.fixture(autouse=True)
def _one_cpu_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def _model(arm="A1"):
    torch.manual_seed(17)
    template = torch.randn(1, 1, 9, 11, 13)
    return PopulationAtlas(
        template,
        levels=2,
        collar=2,
        width=3,
        arm=arm,
        scale_min=0.01,
        scale_max=2.0,
        prior_std=0.15,
        posterior_std=0.15,
        integration_steps=2,
    )


def test_gaussian_normalizers_and_kl_match_distributions():
    value = torch.linspace(-1, 1, 18).reshape(2, 3, 1, 1, 3)
    q_mean, p_mean = value * 0.2, value * -0.1
    q_std = torch.full((2, 3, 1, 1, 1), 0.3)
    p_std = torch.full((1, 3, 1, 1, 1), 0.7)
    q = torch.distributions.Normal(q_mean.double(), q_std.double())
    p = torch.distributions.Normal(p_mean.double(), p_std.double())
    expected_kl = torch.distributions.kl_divergence(q, p).flatten(1).sum(1, dtype=torch.float64)
    torch.testing.assert_close(_normal_kl(q_mean, q_std, p_mean, p_std), expected_kl)
    torch.testing.assert_close(_normal_log_prob(value, p_mean, p_std), p.log_prob(value.double()).flatten(1).sum(1))
    assert _normal_log_prob(
        torch.zeros(1, 3, 1, 1, 1), torch.zeros(1, 3, 1, 1, 1), torch.ones(1, 3, 1, 1, 1)
    ).item() == pytest.approx(-1.5 * math.log(2 * math.pi))


def test_complete_odd_shape_hierarchy_has_normalized_density_and_gradients():
    model = _model()
    observed = torch.randn(2, 1, 9, 11, 13)
    state = model.sample(observed)
    bands = coefficient_bands(state.coefficients)
    assert sum(band.numel() for band in bands) == observed.shape[0] * 3 * 5 * 7 * 9
    assert len(bands) == 15
    assert bands[-1].numel() > 0
    torch.testing.assert_close(model.log_prior(state.coefficients), state.log_prior)
    torch.testing.assert_close(state.kl, torch.zeros_like(state.kl), atol=1e-4, rtol=0)
    assert state.flow.shape == (2, 3, 9, 11, 13)
    assert torch.count_nonzero(state.velocity[:, :, :2]) == 0
    assert torch.count_nonzero(state.velocity[:, :, :, -2:]) == 0
    loss = state.flow.square().mean() + state.kl.mean() / observed[0].numel() + 1e-5 * model.weight_penalty()
    loss.backward()
    for parameter in model.parameters():
        if parameter.grad is not None:
            assert torch.isfinite(parameter.grad).all()
    assert model.posterior_heads[0].mean.weight.grad.abs().sum() > 0
    torch.testing.assert_close(model.atlas().mean(), torch.tensor(0.0), atol=1e-7, rtol=0)
    torch.testing.assert_close(model.atlas().square().mean(), torch.tensor(1.0), atol=1e-6, rtol=0)


def test_nested_arms_share_initialization_posterior_and_unconditional_mean():
    control, linear, nonlinear = (_model(arm) for arm in ("A0", "A_lin", "A1"))
    for key, value in control.state_dict().items():
        torch.testing.assert_close(value, linear.state_dict()[key], atol=0, rtol=0)
        torch.testing.assert_close(value, nonlinear.state_dict()[key], atol=0, rtol=0)
    observed = torch.randn(1, 1, 9, 11, 13)
    for left, right in zip(
        coefficient_bands(nonlinear.posterior_mean(observed)),
        coefficient_bands(control.posterior_mean(observed)),
        strict=True,
    ):
        torch.testing.assert_close(left, right)
    zero = control._empty_bands(1)
    shifted = control._empty_bands(1)
    shifted[0] = torch.randn_like(shifted[0])
    for bands in (zero, shifted):
        base, _ = control._prior_step(1, bands)
        for model in (linear, nonlinear):
            means, _ = model._prior_step(1, bands)
            for actual, expected in zip(means, base, strict=True):
                torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    with torch.no_grad():
        for head in linear.prior_heads:
            head.output.weight.normal_(std=0.1)
        for head in linear.linear_heads:
            head.weight.normal_(std=0.1)
        control.load_state_dict(linear.state_dict())
        nonlinear.load_state_dict(linear.state_dict())
    cond_a, _ = linear._prior_step(1, zero)
    cond_b, _ = linear._prior_step(1, shifted)
    base_a, _ = control._prior_step(1, zero)
    base_b, _ = control._prior_step(1, shifted)
    assert any(not torch.allclose(a, b) for a, b in zip(cond_a, cond_b, strict=True))
    for a, b in zip(base_a, base_b, strict=True):
        torch.testing.assert_close(a, b)
    for bands in (zero, shifted):
        lin_means, _ = linear._prior_step(1, bands)
        nested_means, _ = nonlinear._prior_step(1, bands)
        for actual, expected in zip(nested_means, lin_means, strict=True):
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    with torch.no_grad():
        for head in nonlinear.residual_heads:
            head.output.weight.normal_(std=0.1)
    nested_zero, _ = nonlinear._prior_step(1, zero)
    for actual, expected in zip(nested_zero, base_a, strict=True):
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    assert all(head.bias is None for head in linear.linear_heads)
    assert all(head.kernel_size == (5, 5, 5) for head in linear.linear_heads)


def test_linear_conditioning_is_linear_at_fixed_atlas_and_inactive_heads_are_excluded():
    model = _model("A_lin")
    with torch.no_grad():
        model.prior_heads[1].output.weight.normal_(std=0.1)
        model.linear_heads[1].weight.normal_(std=0.1)
    zero = model._empty_bands(1)
    first, second = model._empty_bands(1), model._empty_bands(1)
    for index in range(8):
        first[index], second[index] = torch.randn_like(first[index]), torch.randn_like(second[index])
    combined = [left + right for left, right in zip(first, second, strict=True)]
    base, _ = model._prior_step(2, zero)
    left, _ = model._prior_step(2, first)
    right, _ = model._prior_step(2, second)
    both, _ = model._prior_step(2, combined)
    for a, b, c, d in zip(base, left, right, both, strict=True):
        torch.testing.assert_close(d - a, (b - a) + (c - a), atol=2e-6, rtol=2e-6)
    before = model.weight_penalty().detach().clone()
    with torch.no_grad():
        model.residual_heads[0].hidden[0].weight.add_(10)
    torch.testing.assert_close(model.weight_penalty(), before, atol=0, rtol=0)
    active_ids = {id(parameter) for parameter in model.prior_parameters()}
    assert all(id(parameter) in active_ids for parameter in model.linear_heads.parameters())
    assert all(id(parameter) not in active_ids for parameter in model.residual_heads.parameters())
    assert all(not parameter.requires_grad for parameter in model.residual_heads.parameters())


def test_fine_posterior_reads_coarse_state_and_prior_hyperprior_excludes_q():
    model = _model()
    with torch.no_grad():
        model.posterior_heads[1].mean.weight.normal_(std=0.1)
    features = torch.randn(1, 3, *model.interior_shape)
    zero, shifted = model._empty_bands(1), model._empty_bands(1)
    shifted[0] = torch.randn_like(shifted[0])
    first, _ = model._posterior_step(1, features, zero)
    second, _ = model._posterior_step(1, features, shifted)
    assert any(not torch.allclose(a, b) for a, b in zip(first, second, strict=True))
    before = model.weight_penalty().detach().clone()
    with torch.no_grad():
        model.encoder[0].weight.add_(3)
    torch.testing.assert_close(model.weight_penalty(), before)
    parameters = list(model.prior_parameters())
    assert parameters[0] is model.prior_scale_logits


def test_stage_scales_and_nearly_equal_gaussian_kl():
    model = PopulationAtlas(
        torch.randn(1, 1, 9, 11, 13),
        levels=2,
        collar=2,
        width=2,
        conditional=True,
        scale_min=0.001,
        scale_max=2.0,
        prior_std=(0.7, 0.2, 0.04),
        posterior_std=(0.5, 0.1, 0.02),
    )
    scales = model._scale(model.prior_scale_logits)[:, 0]
    torch.testing.assert_close(scales, torch.tensor([0.7, *([0.2] * 7), *([0.04] * 7)]))
    means = torch.zeros(1, 3, 4, 4, 4)
    p_std, q_std = torch.tensor(0.1), torch.tensor(0.1 + 1e-8)
    actual = _normal_kl(means, q_std, means, p_std)
    assert actual.item() > 0
    reference = torch.distributions.kl_divergence(
        torch.distributions.Normal(means.double(), q_std.double()),
        torch.distributions.Normal(means.double(), p_std.double()),
    ).sum()
    torch.testing.assert_close(actual[0], reference, atol=1e-15, rtol=1e-3)
    options = dict(
        levels=2,
        collar=2,
        width=2,
        conditional=True,
        scale_min=0.001,
        scale_max=2.0,
        prior_std=[0.2],
        posterior_std=[0.1],
    )
    repeated = PopulationAtlas(torch.randn(1, 1, 9, 11, 13), **options)
    torch.testing.assert_close(repeated._scale(repeated.prior_scale_logits), torch.full((15, 3), 0.2))


def test_conditional_prior_learns_an_identifiable_observed_coarse_fine_relation():
    """Fully observed synthetic coefficients test prior learning, not identifiability from MRI."""
    torch.manual_seed(42)
    template = torch.randn(1, 1, 8, 8, 8)
    options = dict(
        levels=1,
        collar=2,
        width=4,
        scale_min=0.001,
        scale_max=2.0,
        prior_std=0.1,
        posterior_std=0.1,
    )
    conditional = PopulationAtlas(template, arm="A1", **options)
    control = PopulationAtlas(template, arm="A0", **options)
    control.load_state_dict(conditional.state_dict())
    for model in (conditional, control):
        model.raw_atlas.requires_grad_(False)

    def observed_coefficients(values):
        bands = conditional._empty_bands(len(values))
        bands[0][:, 0] = values[:, None, None, None]
        context = conditional._context(bands, conditional.interior_shape)
        shifted = torch.nn.functional.pad(context[:, :1, :, :, 1:], (0, 1, 0, 0, 0, 0))
        target = haar_forward(torch.cat((shifted, torch.zeros_like(context[:, 1:])), dim=1), 1).levels[0].details
        return bands, tuple(value.detach() for value in target)

    train, target = observed_coefficients(torch.linspace(-1, 1, 8))
    heldout, truth = observed_coefficients(torch.tensor([-0.75, -0.25, 0.25, 0.75]))

    def prediction_error(model, bands, targets):
        means, _ = model._prior_step(1, bands)
        return sum((mean - target).square().sum() for mean, target in zip(means, targets, strict=True)) / sum(
            value.numel() for value in targets
        )

    for model in (conditional, control):
        optimizer = torch.optim.Adam(model.prior_parameters(), lr=0.03)
        for _ in range(100):
            optimizer.zero_grad(set_to_none=True)
            loss = prediction_error(model, train, target)
            loss.backward()
            optimizer.step()
    assert prediction_error(conditional, heldout, truth) < 0.1 * prediction_error(control, heldout, truth)
