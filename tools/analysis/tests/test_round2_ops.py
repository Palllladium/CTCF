from __future__ import annotations

from types import SimpleNamespace

import torch
import torch.nn.functional as F

from experiments.train_features import HARD_DEFAULTS, info_nce, mine_ncc_hard, patch_ncc, patches, warp_index
from utils.feature_similarity import (
    ContrastiveFeatureNet,
    FeatureAutoencoder,
    LearnedSimilarity,
    feature_dissimilarity,
    level_points,
    load_feature_net,
    sample_at,
    save_feature_net,
    warped_level_maps,
)
from utils.losses import local_ncc_map
from utils.spatial import SpatialTransformer


class _IdentityFeatures(torch.nn.Module):
    """The image itself as a single full-resolution feature level: an exactly equivariant 'feature net'."""

    def feature_maps(self, image):
        return [image]


def _smooth_flow(shape, amplitude, seed=0):
    noise = torch.randn((1, 3, *(max(2, n // 8) for n in shape)), generator=torch.Generator().manual_seed(seed))
    flow = F.interpolate(noise, size=shape, mode="trilinear", align_corners=True)
    return amplitude * flow / flow.abs().amax()


def test_warp_index_shows_point_x_plus_d():
    image = torch.zeros((1, 1, 24, 24, 24))
    image[0, 0, 10, 12, 14] = 1.0
    d = torch.tensor([2.0, -1.0, 3.0]).view(1, 3, 1, 1, 1).expand(1, 3, 24, 24, 24)
    warped = warp_index(image, d)
    assert torch.equal(warped[0, 0].nonzero()[0], torch.tensor([8, 13, 11]))
    assert torch.isclose(warped.max(), torch.tensor(1.0))


def test_sample_at_integer_points_reads_voxels():
    volume = torch.randn((1, 2, 6, 7, 8))
    points = torch.tensor([[0.0, 0.0, 0.0], [5.0, 6.0, 7.0], [2.0, 3.0, 4.0]])
    expected = volume[0, :, points[:, 0].long(), points[:, 1].long(), points[:, 2].long()].T
    assert torch.allclose(sample_at(volume, points), expected, atol=1e-6)


def test_level_points_match_trilinear_upsampling():
    coarse = torch.randn((1, 3, 8, 10, 12))
    shape = (16, 20, 24)
    full = F.interpolate(coarse, size=shape, mode="trilinear", align_corners=False)
    points = torch.tensor([[3.0, 4.0, 5.0], [7.0, 11.0, 13.0], [12.0, 15.0, 18.0]])
    direct = full[0, :, points[:, 0].long(), points[:, 1].long(), points[:, 2].long()].T
    assert torch.allclose(sample_at(coarse, level_points(points, shape, coarse.shape[2:])), direct, atol=1e-5)


def test_patch_ncc_is_the_loss_map_value():
    a, b = torch.rand((1, 1, 24, 24, 24)), torch.rand((1, 1, 24, 24, 24))
    b = 0.6 * a + 0.4 * b
    point = torch.tensor([[11.0, 12.0, 13.0]])
    expected = local_ncc_map(a, b)[0, 0, 11, 12, 13]
    assert torch.isclose(patch_ncc(patches(a, point), patches(b, point))[0], expected, rtol=1e-4)
    assert torch.isclose(patch_ncc(patches(a, point), patches(-a, point))[0], torch.tensor(1.0), atol=1e-5)


def test_ncc_hard_negatives_found_in_periodic_texture_and_absent_in_flat():
    n = 64
    x = torch.arange(n, dtype=torch.float32)
    texture = (0.5 + 0.4 * torch.sin(2 * torch.pi * x / 8)).view(1, 1, 1, 1, n).expand(1, 1, n, n, n).contiguous()
    flat = torch.full((1, 1, n, n, n), 0.5)
    anchors = torch.tensor([[32.0, 32.0, 32.0], [30.0, 33.0, 31.0], [34.0, 31.0, 33.0]])
    settings = SimpleNamespace(**HARD_DEFAULTS)
    generator = torch.Generator().manual_seed(0)
    _, valid, stats = mine_ncc_hard(texture, texture, anchors, anchors, settings, generator)
    assert stats["coverage"] == 1.0 and valid.any(dim=1).all()
    _, valid, stats = mine_ncc_hard(flat, flat, anchors, anchors, settings, generator)
    assert stats["coverage"] == 0.0 and not valid.any()


def test_info_nce_rewards_the_true_positive():
    anchors = torch.eye(6)
    random_negatives = F.normalize(torch.randn((10, 6)), dim=-1)
    matched = info_nce(anchors, anchors, random_negatives, None, None, temperature=0.1)
    shuffled = info_nce(anchors, anchors.roll(1, dims=0), random_negatives, None, None, temperature=0.1)
    hard = torch.eye(6)[:, None, :].expand(6, 2, 6)
    with_hard = info_nce(anchors, anchors.roll(1, dims=0), random_negatives, hard, torch.ones((6, 2), dtype=bool), 0.1)
    assert matched < shuffled < with_hard


def test_warped_level_maps_at_full_resolution_is_the_spatial_transformer():
    shape = (20, 24, 28)
    image, flow = torch.rand((1, 2, *shape)), _smooth_flow(shape, 3.0)
    expected = SpatialTransformer(shape)(image, flow)
    assert torch.allclose(warped_level_maps([image], flow)[0], expected, atol=1e-5)


def test_modes_differ_only_in_order_for_an_equivariant_feature():
    shape = (20, 24, 28)
    st = SpatialTransformer(shape)
    moving, fixed = torch.rand((1, 1, *shape)), torch.rand((1, 1, *shape))
    values, grads = [], []
    for mode in ("warped_image", "warped_features"):
        flow = _smooth_flow(shape, 2.0).requires_grad_(True)
        value = LearnedSimilarity(_IdentityFeatures(), mode)(moving, st(moving, flow), fixed, flow)
        values.append(value)
        grads.append(torch.autograd.grad(value, flow)[0])
    assert torch.isclose(values[0], values[1], atol=1e-6)
    assert torch.allclose(grads[0], grads[1], atol=1e-6)


def test_feature_dissimilarity_is_zero_for_identical_inputs():
    net = ContrastiveFeatureNet().eval()
    image = torch.rand((1, 1, 32, 32, 32))
    maps = net.feature_maps(image)
    assert feature_dissimilarity(maps, maps, image > 0).abs() < 1e-6


def test_learned_similarity_gradients_reach_the_field_only():
    shape = (32, 32, 32)
    st = SpatialTransformer(shape)
    moving, fixed = torch.rand((1, 1, *shape)), torch.rand((1, 1, *shape))
    for net in (
        ContrastiveFeatureNet().eval().requires_grad_(False),
        FeatureAutoencoder().eval().requires_grad_(False),
    ):
        for mode in ("warped_image", "warped_features"):
            flow = (0.5 * torch.randn((1, 3, *shape))).requires_grad_(True)
            warped = st(moving, flow) if mode == "warped_image" else None
            (grad,) = torch.autograd.grad(LearnedSimilarity(net, mode)(moving, warped, fixed, flow), flow)
            assert torch.isfinite(grad).all() and grad.abs().sum() > 0
            assert all(p.grad is None for p in net.parameters())


def test_feature_net_checkpoint_round_trip(tmp_path):
    net = FeatureAutoencoder().eval()
    path = str(tmp_path / "net.pth")
    save_feature_net(path, net, "autoencoder", {}, {"note": "test"})
    loaded = load_feature_net(path, torch.device("cpu"))
    image = torch.rand((1, 1, 32, 32, 32))
    assert not any(p.requires_grad for p in loaded.parameters()) and not loaded.training
    for a, b in zip(net.feature_maps(image), loaded.feature_maps(image), strict=True):
        assert torch.equal(a, b)
