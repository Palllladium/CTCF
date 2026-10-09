"""Operators added for the round-1 diagnostics and the per-structure NCC training probe."""

from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage

from tools.analysis.diag_round1 import (
    double_index_field,
    fixed_id,
    halve_index_field,
    halve_labels,
    index_displacement,
    subjects,
    tissue_regions,
    warp_labels_index,
)
from tools.analysis.field_audit import _materialize_ctcf_legacy_sampler
from utils.anatomy import TissueGMM
from utils.field import certified_local_clip, differentiable_local_clip
from utils.losses import NCCVxm, local_ncc_map, region_mean
from utils.spatial import SpatialTransformer


def _nccvxm_before_refactor(y_true, y_pred, win=(9, 9, 9), eps=1e-5):
    """NCCVxm.forward as it stood before local_ncc_map was split out (3-D, no mask), frozen for comparison."""
    ii, ji = y_true, y_pred
    filt = torch.ones((1, 1, *win), device=ii.device, dtype=ii.dtype)
    i2, j2, ij = ii * ii, ji * ji, ii * ji
    i_sum, j_sum = F.conv3d(ii, filt, padding=4), F.conv3d(ji, filt, padding=4)
    i2_sum, j2_sum, ij_sum = F.conv3d(i2, filt, padding=4), F.conv3d(j2, filt, padding=4), F.conv3d(ij, filt, padding=4)
    win_size = float(torch.tensor(win).prod().item())
    ui, uj = i_sum / win_size, j_sum / win_size
    cross = ij_sum - uj * i_sum - ui * j_sum + ui * uj * win_size
    i_var = torch.clamp(i2_sum - 2 * ui * i_sum + ui * ui * win_size, min=eps)
    j_var = torch.clamp(j2_sum - 2 * uj * j_sum + uj * uj * win_size, min=eps)
    return -torch.mean((cross * cross) / (i_var * j_var))


def test_nccvxm_is_bitwise_unchanged_and_shares_its_map():
    g = torch.Generator().manual_seed(0)
    fixed = torch.rand((1, 1, 20, 22, 24), generator=g)
    moving = torch.rand((1, 1, 20, 22, 24), generator=g)
    loss = NCCVxm(win=[9, 9, 9])(moving, fixed)
    assert torch.equal(loss, _nccvxm_before_refactor(moving, fixed))
    # the per-structure training arm averages exactly this map, so control and arm share the operator
    assert torch.equal(-local_ncc_map(moving, fixed, [9, 9, 9], 1e-5).mean(), loss)


def test_region_mean_weights_each_region_once():
    values = torch.tensor([10.0, 1.0, 3.0, 4.0, 4.0, 4.0, 7.0])
    regions = torch.tensor([0, 1, 1, 2, 2, 2, 3])
    # region 0 is ignored; regions 1, 2, 3 have means 2, 4, 7 regardless of their sizes
    assert torch.isclose(region_mean(values, regions), torch.tensor(13.0 / 3.0))


def test_clip_from_a_given_current_matches_the_certified_clip():
    g = torch.Generator().manual_seed(1)
    current = torch.zeros((1, 3, 12, 12, 12), dtype=torch.float64)
    proposal = 2.0 * torch.randn((1, 3, 12, 12, 12), generator=g, dtype=torch.float64)
    start = certified_local_clip(current, 0.5 * proposal, eps=1e-3, sweeps=1).double()
    expected = certified_local_clip(start, proposal, eps=1e-3, sweeps=2).double()
    got = differentiable_local_clip(proposal, eps=1e-3, sweeps=2, flow_current=start)
    assert torch.allclose(got, expected, atol=1e-6)
    from_identity = differentiable_local_clip(proposal, eps=1e-3, sweeps=2)
    assert torch.allclose(
        from_identity, certified_local_clip(current, proposal, eps=1e-3, sweeps=2).double(), atol=1e-6
    )


def test_halved_affine_field_doubles_back_exactly():
    d, h, w = 16, 18, 20
    zz, yy, xx = torch.meshgrid(*(torch.arange(n, dtype=torch.float64) for n in (d, h, w)), indexing="ij")
    field = torch.stack([0.1 * zz + 1.0, -0.05 * yy, 0.02 * xx + 0.3])[None]
    half = halve_index_field(field)
    assert half.shape[2:] == (8, 9, 10)
    # half voxel i sits at full position i * (N - 1) / (M - 1): there the field, in half voxels, is d / scale
    assert abs(float(half[0, 0, 7, 0, 0]) - (0.1 * 15 + 1.0) / (15 / 7)) < 1e-12
    assert torch.allclose(double_index_field(half, (d, h, w)), field, atol=1e-9)


def test_halved_labels_match_scipy_zoom_order_0():
    g = torch.Generator().manual_seed(4)
    labels = F.max_pool3d(torch.randint(0, 6, (1, 1, 20, 22, 24), generator=g).double(), 3, 1, 1).long()
    reference = ndimage.zoom(labels[0, 0].numpy(), (10 / 20, 11 / 22, 12 / 24), order=0)
    assert (halve_labels(labels)[0, 0].numpy() != reference).mean() < 1e-2  # rounding ties at half voxels


def _smooth_flow_and_labels(seed, shape=(18, 20, 22)):
    g = torch.Generator().manual_seed(seed)
    flow = F.avg_pool3d(3.0 * torch.randn((1, 3, *shape), generator=g, dtype=torch.float64), 5, 1, 2)
    labels = torch.randint(0, 6, (1, 1, *shape), generator=g)
    labels = F.max_pool3d(labels.double(), 3, 1, 1).long()  # blocky labels: fewer ties at label borders
    return flow, labels


def test_index_displacement_samples_what_our_transformer_samples():
    flow, labels = _smooth_flow_and_labels(2)
    ours = SpatialTransformer(labels.shape[2:], mode="nearest").double()(labels.double(), flow).round().long()
    official = warp_labels_index(labels, index_displacement(flow))
    # same sampling points; only exact rounding ties at a half-voxel may differ
    assert (ours != official).float().mean() < 1e-3


def test_index_displacement_is_the_audit_conversion():
    flow, _ = _smooth_flow_and_labels(5)
    assert torch.allclose(index_displacement(flow), _materialize_ctcf_legacy_sampler(flow), atol=1e-12)


def test_zero_canonical_field_is_the_identity():
    _, labels = _smooth_flow_and_labels(6)
    zero = torch.zeros((1, 3, *labels.shape[2:]), dtype=torch.float64)
    assert torch.equal(warp_labels_index(labels, zero), labels)


def test_index_warp_matches_scipy_map_coordinates():
    flow, labels = _smooth_flow_and_labels(3)
    d = index_displacement(flow)
    grid = np.stack(np.meshgrid(*(np.arange(n) for n in labels.shape[2:]), indexing="ij")) + d[0].numpy()
    reference = ndimage.map_coordinates(labels[0, 0].numpy(), grid, order=0, mode="constant", cval=0)
    ours = warp_labels_index(labels, d)[0, 0].numpy()
    inside = (grid >= 0.5).all(axis=0) & (grid <= np.array(labels.shape[2:])[:, None, None, None] - 1.5).all(axis=0)
    assert (ours[inside] != reference[inside]).mean() < 1e-3  # boundary conventions differ, the interior must not


def test_pair_files_store_the_moving_subject_as_x():
    # p_0438_0439 holds x = 0439 (moving) and y = 0438 (fixed); p_0439_0440 holds x = 0440, y = 0439
    pairs = [
        ("0438_0439", "img0439", "img0438", "seg0439", "seg0438"),
        ("0439_0440", "img0440", "img0439", "seg0440", "seg0439"),
    ]
    found = subjects(SimpleNamespace(pairs=pairs))
    assert found == {"0438": ("img0438", "seg0438"), "0439": ("img0439", "seg0439"), "0440": ("img0440", "seg0440")}
    assert fixed_id("0438_0439") == "0438"


def test_tissue_regions_separates_components():
    image = torch.zeros((1, 1, 12, 12, 12))
    image[0, 0, 1:5, 1:5, 1:5] = 0.2  # one dark blob
    image[0, 0, 7:11, 7:11, 7:11] = 0.2  # a second, disconnected dark blob
    image[0, 0, 1:3, 8:11, 1:3] = 0.8  # one bright blob
    gmm = TissueGMM(means=np.array([0.2, 0.5, 0.8]), variances=np.full(3, 0.001), weights=np.full(3, 1 / 3))
    ctx = SimpleNamespace(gmm=lambda: gmm, device=torch.device("cpu"))
    regions = tissue_regions(ctx, image)
    assert int(regions.max()) == 3
    assert int((regions == 0).sum()) == image.numel() - int((image > 0).sum())
