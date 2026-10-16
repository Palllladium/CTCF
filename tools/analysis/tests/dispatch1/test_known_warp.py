from __future__ import annotations

import pickle

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from experiments.dispatch1.known_warp import (
    MARGIN,
    Affine,
    AnchorTargets,
    KnownWarpSource,
    Radial,
    Warp,
    anchor_loss,
    effective_points,
    sample_points,
    structure_weights,
)


def _grid(shape):
    axes = [torch.arange(n, dtype=torch.float32) for n in shape]
    return torch.stack(torch.meshgrid(*axes, indexing="ij"))


def _brain(shape=(40, 44, 48), seed=0):
    """A smooth synthetic head: an ellipsoid brain with blobs of texture and a dark central ventricle."""
    grid = _grid(shape)
    center = torch.tensor([(n - 1) / 2 for n in shape]).view(3, 1, 1, 1)
    radius = torch.tensor([n * 0.42 for n in shape]).view(3, 1, 1, 1)
    noise = torch.rand((1, 1, 6, 6, 6), generator=torch.Generator().manual_seed(seed))
    texture = F.interpolate(noise, size=shape, mode="trilinear", align_corners=True)[0, 0]
    image = torch.where((((grid - center) / radius) ** 2).sum(0) <= 1, 0.5 + 0.5 * texture, torch.zeros(shape))
    image = torch.where((((grid - center) / (radius / 4)) ** 2).sum(0) <= 1, torch.full(shape, 0.15), image)
    return image.numpy().astype(np.float32)


def _write_subject(path, image):
    with open(path, "wb") as h:
        pickle.dump((image, (image > 0).astype(np.int16)), h)
    return str(path)


@pytest.fixture
def clean_source(tmp_path, monkeypatch):
    """A source on the synthetic head whose views keep their clean intensities."""
    monkeypatch.setattr(KnownWarpSource, "_appearance", lambda self, clean: clean)
    files = [_write_subject(tmp_path / "s0.pkl", _brain())]
    return KnownWarpSource(
        files, "cpu", seed=0, anchors=256, target="set", stats={"tau_small": 0.05, "tau_large": 0.05}
    )


def test_warp_inverse_is_accurate_at_the_extreme_parameters():
    gen = torch.Generator().manual_seed(0)
    center = torch.tensor([20.0, 22.0, 24.0])
    matrix = torch.eye(3) + 0.03 * (2 * torch.rand((3, 3), generator=gen) - 1)
    radials = [Radial(center + 3, 8.0, 1.5), Radial(center - 4, 8.0, -0.6), Radial(center, 14.0, 1.0)]
    warp = Warp(Affine(matrix, center, torch.tensor([2.0, -2, 1])), radials)
    points = 48 * torch.rand((2000, 3), generator=gen)
    assert (warp.inverse(warp.forward(points)) - points).abs().max() < 1e-4


def test_anchors_are_observable_and_their_targets_match_intensities(clean_source):
    pair = clean_source.sample()
    shape = torch.tensor(pair.fixed.shape[2:])
    for targets, fixed, moving in ((pair.fm, pair.fixed, pair.moving), (pair.mf, pair.moving, pair.fixed)):
        assert len(targets.anchors) > 0
        assert ((targets.anchors >= MARGIN) & (targets.anchors <= shape - 1 - MARGIN)).all()
        assert ((targets.points >= MARGIN) & (targets.points <= shape - 1 - MARGIN)).all()
        at_target = sample_points(moving, targets.points)[..., 0]
        at_anchor = fixed[0, 0][tuple(targets.anchors.T)]
        assert (at_target > 0).all() and (at_anchor > 0).all()
        unmoved = sample_points(moving, targets.anchors.float())[..., 0]
        # F(x) = I(b(x)) and M(T(x)) = I(a(T(x))) = I(b(x)): equal up to the two interpolations
        assert (at_target - at_anchor).abs().mean() < 0.25 * (unmoved - at_anchor).abs().mean()


def test_point_and_set_losses_vanish_at_the_exact_field(clean_source):
    targets = clean_source.sample().fm
    flow = torch.zeros(1, 3, 40, 44, 48)
    scale = torch.tensor([n / (n - 1) for n in flow.shape[2:]])
    exact = (targets.points + 0.5) / scale - targets.anchors.float()
    flow[0, :, targets.anchors[:, 0], targets.anchors[:, 1], targets.anchors[:, 2]] = exact.T
    assert torch.allclose(effective_points(flow, targets.anchors), targets.points, atol=1e-4)
    assert anchor_loss(flow, targets, "point") < 1e-8
    assert anchor_loss(flow, targets, "set") < 1e-8
    assert anchor_loss(torch.zeros_like(flow), targets, "point") > 0.1


def test_samples_ignore_the_callers_autocast(tmp_path):
    files = [_write_subject(tmp_path / "s0.pkl", _brain())]
    plain = KnownWarpSource(files, "cpu", seed=3, anchors=64, target="structure").sample()
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):  # training steps run under autocast
        mixed = KnownWarpSource(files, "cpu", seed=3, anchors=64, target="structure").sample()
    assert torch.equal(plain.fm.points, mixed.fm.points) and torch.equal(plain.mf.points, mixed.mf.points)
    assert torch.equal(plain.fm.normal_weight, mixed.fm.normal_weight) and torch.equal(plain.moving, mixed.moving)


def test_targets_share_the_views_and_anchors(tmp_path):
    files = [_write_subject(tmp_path / "s0.pkl", _brain())]
    stats = {"tau_small": 0.3, "tau_large": 0.2}
    targets = ("point", "set", "structure", "deadzone")
    sources = [KnownWarpSource(files, "cpu", seed=3, anchors=16, target=t, stats=stats) for t in targets]
    for _ in range(2):
        reference, *others = (source.sample() for source in sources)
        for pair in others:
            for name in ("fixed", "moving", "clean_fixed", "clean_moving"):
                assert torch.equal(getattr(pair, name), getattr(reference, name))
            for direction in ("fm", "mf"):
                assert torch.equal(getattr(pair, direction).anchors, getattr(reference, direction).anchors)
                assert torch.equal(getattr(pair, direction).points, getattr(reference, direction).points)


def _members(source, image, point):
    members, mask = source.set_members(image[None, None], torch.tensor([point], dtype=torch.float32), tau=(1e-4, 1e-4))
    offsets = (members[0] - torch.tensor(point, dtype=torch.float32)).round().long()
    return {tuple(o.tolist()): bool(m) for o, m in zip(offsets, mask[0], strict=True)}


def test_set_members_on_analytic_images(clean_source):
    z, y, x = _grid((32, 32, 32))
    point = (16, 16, 16)
    assert all(_members(clean_source, torch.ones(32, 32, 32), point).values())
    step = _members(clean_source, (z >= 16).float(), point)
    assert step[(0, 3, -2)] and step[(0, 0, 4)] and not step[(1, 0, 0)] and not step[(-1, 2, 0)]
    corner = _members(clean_source, ((z >= 16) & (y >= 16)).float(), point)
    assert corner[(0, 0, 3)] and not corner[(0, 1, 0)] and not corner[(1, 0, 1)]
    cells = sum(torch.div(axis, 2, rounding_mode="floor") for axis in (z, y, x))
    checker = _members(clean_source, (cells % 2).float(), point)
    assert checker[(4, 0, 0)] and checker[(2, 2, 0)] and not checker[(2, 0, 0)] and not checker[(1, 0, 0)]


def test_structure_weight_ignores_shifts_along_an_edge():
    z = _grid((24, 24, 24))[0]
    image = torch.sigmoid(2 * (z - 11.5))[None, None]
    points = torch.tensor([[11.5, float(y), float(x)] for y in range(8, 16) for x in range(8, 16)])
    weight = structure_weights(image, points)
    along, across = torch.tensor([0.0, 1.0, 0.0]), torch.tensor([1.0, 0.0, 0.0])
    assert (along @ weight @ along).abs().max() < 1e-3
    assert (across @ weight @ across).min() > 0.9


def test_structure_loss_keeps_float32_values_and_gradients_under_autocast():
    generator = torch.Generator().manual_seed(7)
    flow = torch.randn((1, 3, 6, 6, 6), generator=generator, requires_grad=True)
    matrix = torch.randn((3, 3, 3), generator=generator)
    targets = AnchorTargets(
        anchors=torch.tensor([[1, 2, 3], [3, 2, 1], [2, 3, 2]]),
        points=torch.rand((3, 3), generator=generator) + 2,
        normal_weight=matrix @ matrix.transpose(-1, -2),
    )
    expected = anchor_loss(flow, targets, "structure")
    expected_gradient = torch.autograd.grad(expected, flow)[0]
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        actual = anchor_loss(flow, targets, "structure")
    assert actual.dtype == torch.float32 and torch.equal(actual, expected)
    assert torch.equal(torch.autograd.grad(actual, flow)[0], expected_gradient)
