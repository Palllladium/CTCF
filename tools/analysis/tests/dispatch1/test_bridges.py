from __future__ import annotations

import argparse
import json
import math
import pickle
import random
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from datasets.OASIS import OASISBrainDataset
from experiments.core.cli_inference import add_tto_args
from experiments.core.field_scoring import REPAIR, repair_flags, score, score_ff_rep, warp_labels
from experiments.core.inference_metrics import digital_projection_columns, trilinear_projection_columns
from experiments.dispatch1.bridges import EnergyTable, compose_bridge
from experiments.dispatch1.distill_teacher import DistillTeacher
from experiments.dispatch1.energy_table import merge
from experiments.train_CTCF import Runner, _gradient_ratio
from utils import NumpyType
from utils.field import _warp, digital_project, legacy_sampling_displacement, trilinear_project
from utils.spatial import RegisterModel, SpatialTransformer

SHAPE = (20, 22, 24)


def _smooth(shape, amplitude, seed):
    low = torch.randn((1, 3, 4, 4, 4), generator=torch.Generator().manual_seed(seed))
    return amplitude * F.interpolate(low, size=shape, mode="trilinear", align_corners=True) / low.abs().max()


def _inner(t, margin=4):
    return t[(slice(None), slice(None), *(slice(margin, -margin),) * 3)]


def _grid(shape):
    axes = [torch.arange(n, dtype=torch.float32) for n in shape]
    return torch.stack(torch.meshgrid(*axes, indexing="ij"))[None]


def _scale(shape):
    return torch.tensor([n / (n - 1) for n in shape]).view(1, 3, 1, 1, 1)


def _effective(flow):
    """The map the legacy sampler reads for a stored flow, s * (x + u) - 1/2."""
    return _scale(flow.shape[2:]) * (_grid(flow.shape[2:]) + flow) - 0.5


def _sine(points, amplitude, phase):
    """A smooth field whose component a varies along axis a + 1, at voxel coordinates (1, 3, ...)."""
    waves = [torch.sin(2 * math.pi * points[:, (a + 1) % 3 : (a + 1) % 3 + 1] / 24 + phase + a) for a in range(3)]
    return amplitude * torch.cat(waves, dim=1)


def _write_subject(path, image, seg=None):
    with open(path, "wb") as h:
        pickle.dump((image, seg if seg is not None else (image > 0).astype(np.int16)), h)
    return str(path)


def test_legacy_displacement_equals_the_previous_materializer_and_reproduces_the_legacy_sampler():
    flow = _smooth(SHAPE, 2.0, 0)
    previous = _scale(SHAPE) * (_grid(SHAPE) + flow) - 0.5 - _grid(SHAPE)  # tools/analysis/materialize_psi before
    assert torch.allclose(legacy_sampling_displacement(flow), previous, atol=1e-6)
    image = _smooth(SHAPE, 1.0, 1)[:, :1]
    legacy = SpatialTransformer(SHAPE)(image, flow)
    assert torch.allclose(_inner(_warp(image, legacy_sampling_displacement(flow))), _inner(legacy), atol=1e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_legacy_displacement_runs_on_the_device_of_the_flow():
    flow = _smooth(SHAPE, 2.0, 0)
    on_device = legacy_sampling_displacement(flow.cuda())
    assert on_device.is_cuda and torch.allclose(on_device.cpu(), legacy_sampling_displacement(flow), atol=1e-5)


def test_label_warps_follow_the_inference_runner_and_the_canonical_sampler():
    seg = torch.randint(0, 36, (1, 1, *SHAPE)).float()
    flow = _smooth(SHAPE, 2.0, 2)
    assert torch.equal(warp_labels(seg, flow, "legacy"), RegisterModel(SHAPE, mode="nearest")((seg, flow)))
    assert torch.equal(warp_labels(seg, flow, "canonical"), _warp(seg, flow, mode="nearest"))
    with pytest.raises(ValueError):
        warp_labels(seg, flow, "other")


def test_rep_is_the_inference_chain_with_its_column_order():
    seg = torch.randint(0, 36, (1, 1, *SHAPE))
    flow = _smooth(SHAPE, 3.0, 4)
    _, rep = score_ff_rep(flow, "case", 0.0, seg, seg, "canonical")
    repaired, folds, iters = digital_project(flow, eps=0.0, damp=0.6, max_iters=80)
    repaired, report = trilinear_project(repaired, eps=0.001, damp=0.6, max_iters=80, subdiv_depth=0)
    projection = {**digital_projection_columns(folds, iters), **trilinear_projection_columns(report)}
    expected = score(repaired, "case", 0.0, seg, seg, 0.001, "canonical", projection)
    assert list(rep) == list(expected) and all(rep[k] == expected[k] for k in rep if k != "time_sec")
    labels = [k for k in rep if k.startswith("dice_lbl_")]
    assert list(rep).index("tri_proj_iters") < list(rep).index(labels[0])  # as experiments.inference writes them


def test_repair_flags_give_experiments_inference_the_same_chain():
    parser = argparse.ArgumentParser()
    add_tto_args(parser)
    args = parser.parse_args(repair_flags().split())
    assert args.tto_project and args.tto_tri_project and args.tto_tri_subdiv_depth == 0
    assert (args.tto_project_eps, args.tto_tri_project_eps) == (REPAIR["project_eps"], REPAIR["tri_eps"])
    assert args.tto_project_damp == args.tto_tri_project_damp == REPAIR["damp"]
    assert args.tto_project_iters == args.tto_tri_project_iters == REPAIR["iters"]


def test_identity_scores_full_dice_and_repair_keeps_it():
    seg = torch.randint(0, 36, (1, 1, *SHAPE))
    ff, rep = score_ff_rep(torch.zeros(1, 3, *SHAPE), "case", 0.5, seg, seg, "canonical")
    assert ff["dice_mean"] == pytest.approx(1.0) and rep["dice_mean"] == pytest.approx(1.0)
    assert rep["proj_folds_end"] == 0 and rep["time_sec"] >= ff["time_sec"] == 0.5


def test_bridge_reads_the_composed_legacy_maps_for_smooth_fields():
    flow_fs, flow_sm = _sine(_grid(SHAPE), 1.5, 0.3), _sine(_grid(SHAPE), 1.0, 1.1)
    first = _effective(flow_fs)  # F -> S
    analytic = _scale(SHAPE) * (first + _sine(first, 1.0, 1.1)) - 0.5  # S -> M at the continuous point
    error = (_inner(_effective(compose_bridge(flow_fs, flow_sm))) - _inner(analytic)).abs().max()
    assert error < 0.05  # trilinear interpolation of a 24-voxel sine of amplitude 1


def test_bridge_of_constant_shifts_is_exact_inside():
    flow_fs = torch.tensor([0.5, -1.25, 2.0]).view(1, 3, 1, 1, 1).expand(1, 3, *SHAPE).contiguous()
    flow_sm = torch.tensor([-0.75, 0.5, 1.0]).view(1, 3, 1, 1, 1).expand(1, 3, *SHAPE).contiguous()
    analytic = _scale(SHAPE) * (_effective(flow_fs) + flow_sm) - 0.5
    assert torch.allclose(_inner(_effective(compose_bridge(flow_fs, flow_sm))), _inner(analytic), atol=1e-4)


class _ShiftNet(torch.nn.Module):
    """A toy parent: subject k is the constant image k; F <- M is the constant flow (k_F - k_M) * v."""

    v = torch.tensor([0.5, -0.25, 0.75]).view(1, 3, 1, 1, 1)

    def forward(self, moving, fixed, alpha_l1=1.0, alpha_l3=1.0):
        centre = tuple(n // 2 for n in moving.shape[2:])
        code = fixed[(0, 0, *centre)] - moving[(0, 0, *centre)]
        return moving, (code * self.v).expand(1, 3, *moving.shape[2:]).contiguous()


def _subjects(tmp_path, codes):
    return [_write_subject(tmp_path / f"{s}.pkl", np.full(SHAPE, k, np.float32)) for s, k in codes.items()]


def _image(code):
    return torch.full((1, 1, *SHAPE), float(code))


def _through(first, second):
    """The legacy map of the constant stored flow ``first`` followed by the constant ``second``."""
    return _scale(SHAPE) * (_scale(SHAPE) * (_grid(SHAPE) + first) - 0.5 + second) - 0.5


def test_bridge_teacher_composes_the_parent_legs(tmp_path):
    files = _subjects(tmp_path, {"s1": 1.0, "s2": 2.0, "s3": 3.0})
    field = DistillTeacher(_ShiftNet(), "random_bridge", "all", "cpu", files).target(_image(3), _image(1), "s3", "s1")
    v = _ShiftNet.v  # the only bridge is s2: F <- S is (1 - 2) v, S <- M is (2 - 3) v
    assert torch.allclose(_inner(_effective(field)), _inner(_through(-v, -v)), atol=1e-4)
    direct = DistillTeacher(_ShiftNet(), "direct", "all", "cpu", files).target(_image(3), _image(1), "s3", "s1")
    assert torch.allclose(direct, -2 * v.expand_as(direct))


def test_second_pass_teacher_composes_both_passes(tmp_path):
    files = _subjects(tmp_path, {"s1": 1.0, "s3": 3.0})
    field = DistillTeacher(_ShiftNet(), "recursion", "all", "cpu", files).target(_image(3), _image(1), "s3", "s1")
    v = _ShiftNet.v  # the warped constant image keeps its code, so both passes are (1 - 3) v
    assert torch.allclose(_inner(_effective(field)), _inner(_through(-2 * v, -2 * v)), atol=1e-4)


def test_proxy_teacher_bridges_through_the_subject_nearest_the_pair(tmp_path):
    codes = {"s1": 1.0, "s2": 2.0, "s3": 3.0, "s4": 4.0}
    proxy = tmp_path / "proxy.json"
    proxy.write_text(json.dumps({"values": {"s1": 100, "s2": 190, "s3": 700, "s4": 400}}))
    teacher = DistillTeacher(_ShiftNet(), "bridge_proxy", "all", "cpu", _subjects(tmp_path, codes), proxy=str(proxy))
    field = teacher.target(_image(4), _image(1), "s4", "s1")  # geometric mean 200: s2
    v = _ShiftNet.v
    assert torch.allclose(_inner(_effective(field)), _inner(_through(-v, -2 * v)), atol=1e-4)


def test_eligible_policy_teaches_only_pairs_a_bridge_eases(tmp_path):
    codes = {"s1": 1.0, "s2": 2.0, "s3": 3.0, "s4": 4.0}
    norm, bank, subjects = float(torch.linalg.vector_norm(_ShiftNet.v)), ["s2", "s3"], sorted(codes)
    np.savez(  # the toy parent's energies: |k_F - k_M| |v|
        tmp_path / "table.npz",
        subjects=np.array(subjects),
        bank=np.array(bank),
        to_bank=np.array([[abs(codes[f] - codes[b]) * norm for b in bank] for f in subjects]),
        from_bank=np.array([[abs(codes[b] - codes[m]) * norm for m in subjects] for b in bank]),
        ckpt=np.array("parent.pth"),
        config=np.array("cfg"),
    )
    teacher = DistillTeacher(
        _ShiftNet(), "bridge_minimax", "eligible", "cpu", _subjects(tmp_path, codes), table=str(tmp_path / "table.npz")
    )
    far = teacher.target(_image(4), _image(1), "s4", "s1")  # direct 3|v|; through s2 the harder leg is 2|v|
    v = _ShiftNet.v
    assert torch.allclose(_inner(_effective(far)), _inner(_through(-v, -2 * v)), atol=1e-4)
    assert teacher.target(_image(2), _image(1), "s2", "s1") is None  # direct |v|; through s3 the harder leg is 2|v|


def test_bridge_images_stay_on_the_host_with_their_intensities(tmp_path):
    image = np.linspace(0.11, 0.97, 1000, dtype=np.float32).reshape(10, 10, 10)
    teacher = DistillTeacher(
        torch.nn.Linear(1, 1), "random_bridge", "all", "cpu", [_write_subject(tmp_path / "s.pkl", image)]
    )
    assert torch.equal(teacher._image("s")[0, 0], torch.from_numpy(image))
    assert teacher._images["s"].device.type == "cpu"


def _table(path, subjects, bank, seed):
    gen = np.random.default_rng(seed)
    to_bank, from_bank = gen.random((len(subjects), len(bank))), gen.random((len(bank), len(subjects)))
    arrays = {"subjects": np.array(subjects), "bank": np.array(bank), "to_bank": to_bank, "from_bank": from_bank}
    np.savez(path, **arrays, ckpt=np.array("parent.pth"), config=np.array("cfg"))
    return str(path)


def test_minimax_lookup_equals_brute_force_and_shards_merge(tmp_path):
    subjects, bank = [f"s{k}" for k in range(6)], ["s1", "s3", "s4", "s5"]
    table = EnergyTable(_table(tmp_path / "t.npz", subjects, bank, 0))
    for fixed in subjects:
        for moving in (s for s in subjects if s != fixed):
            scores = {
                s: max(float(table.to_bank[table.row[fixed], k]), float(table.from_bank[k, table.row[moving]]))
                for k, s in enumerate(bank)
                if s not in (fixed, moving)
            }
            best = min(scores, key=scores.get)
            assert table.best(fixed, moving) == (best, pytest.approx(scores[best]))
    whole = dict(np.load(tmp_path / "t.npz"))
    for name, rows in (("a", [3, 4, 5]), ("b", [0, 1, 2])):
        part = {**whole, "subjects": whole["subjects"][rows], "to_bank": whole["to_bank"][rows]}
        np.savez(tmp_path / f"{name}.npz", **{**part, "from_bank": whole["from_bank"][:, rows]})
    merge(str(tmp_path / "m.npz"), [str(tmp_path / "a.npz"), str(tmp_path / "b.npz")])
    merged = np.load(tmp_path / "m.npz")
    assert all(np.array_equal(merged[key], whole[key]) for key in ("subjects", "to_bank", "from_bank"))


def test_energy_merge_rejects_duplicate_rows_and_mixed_parents(tmp_path):
    def part(name, ckpt):
        path = tmp_path / name
        np.savez(path, subjects=["a"], bank=["b"], to_bank=[[1.0]], from_bank=[[1.0]], ckpt=ckpt, config="cfg")
        return str(path)

    first = part("a.npz", "parent1")
    with pytest.raises(ValueError, match="different ckpt"):
        merge(str(tmp_path / "out.npz"), [first, part("b.npz", "parent2")])
    with pytest.raises(ValueError, match="duplicate"):
        merge(str(tmp_path / "out.npz"), [first, first])


def test_subject_ids_do_not_change_the_dataset_output(tmp_path):
    shape = (4, 5, 6)
    files = [
        _write_subject(tmp_path / f"subj{k}.pkl", np.full(shape, float(k), np.float32), np.full(shape, k, np.int16))
        for k in range(4)
    ]
    transforms = NumpyType((np.float32, np.int16))
    random.seed(7)
    plain = [OASISBrainDataset(files, transforms)[i] for i in range(4)]
    random.seed(7)
    named = [OASISBrainDataset(files, transforms, return_ids=True)[i] for i in range(4)]
    for a, b in zip(plain, named, strict=True):
        assert len(b) == 6 and all(torch.equal(u, v) for u, v in zip(a, b[:4], strict=True))
        assert b[4] == f"subj{int(b[0].flatten()[0])}" and b[5] == f"subj{int(b[1].flatten()[0])}"


def test_zero_beta_trains_like_no_teacher_without_building_one():
    results = []
    for teacher in ("none", "direct"):
        runner = Runner.__new__(Runner)
        runner.args = SimpleNamespace(
            sim="ncc",
            ncc_regions="none",
            known_warp_every=0,
            distill_teacher=teacher,
            distill_beta=0.0,
            aux_calibrate="",
        )
        parameter = torch.tensor(0.3, requires_grad=True)
        runner._ema_update = lambda: None
        runner._unpack = lambda batch: (None, None, None, None, None)
        runner._pair_loss = lambda *args, parameter=parameter: ((parameter - 2).square(), {}, parameter, parameter)
        with patch("experiments.train_CTCF.DistillTeacher", side_effect=AssertionError("teacher must stay disabled")):
            runner._init_auxiliary_objectives(runner.args, torch.device("cpu"))
            loss, _ = runner.train_step(None, 0)
        loss.backward()
        results.append((loss.detach(), parameter.grad))
    assert all(torch.equal(a, b) for a, b in zip(*results, strict=True))


def test_gradient_ratio_counts_unused_tensors_and_rejects_nonfinite_norms():
    active, unused = torch.tensor(2.0, requires_grad=True), torch.tensor(0.0, requires_grad=True)
    assert _gradient_ratio(active.square(), 2 * active.square(), [active, unused]) == 0.5
    with pytest.raises(RuntimeError, match="finite nonzero"):
        _gradient_ratio(active.square(), active * float("inf"), [active, unused])
