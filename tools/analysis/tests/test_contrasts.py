from __future__ import annotations

import copy
import json
import pickle
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import SimpleITK as sitk  # noqa: N813
import torch
from scipy import ndimage
from torch import nn

from datasets.IXI import IXIPairedDataset, IXIPairSampler, read_paired_manifest
from experiments.contrasts import RECIPE, evaluate, training_args
from experiments.core import field_scoring
from experiments.core.data_loaders import ixi_paired_loaders
from experiments.core.inference_metrics import metric_profile_for
from experiments.core.inference_runtime import load_checkpoint_state
from experiments.dispatch1.eval_consensus import bridge, certificate_modes
from experiments.train_CTCF import Runner, parse_args
from models.CTCF import model as cascade
from models.CTCF.configs import CONFIGS
from models.CTCF.stages import FlowRefiner3D
from models.TransMorph_DCA.model import SpatialTransformer as ModelWarp
from tools.analysis.prepare_ixi import (
    calibrate,
    finalise,
    laterality_report,
    prepare,
    prepare_arrays,
    processed_image,
    register,
    rigid_motion,
)
from tools.analysis.run_artifacts import sha256_file
from utils import NCCVxm, RegisterModel
from utils.field import _warp
from utils.spatial import SpatialTransformer


@pytest.fixture(autouse=True)
def small_cpu_pool():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


class TinyL2(nn.Module):
    """Exercise all three Mamba input-kernel names without its CUDA-only selective scan."""

    def __init__(self, config):
        super().__init__()
        self.model = nn.Module()
        self.model.transformer = nn.Module()
        self.model.transformer.patch_embed = nn.Module()
        channels = 2 * config.image_channels
        self.model.transformer.patch_embed.proj = nn.Conv3d(channels, 3, 1)
        self.model.c1 = nn.Sequential(nn.Conv3d(channels, 3, 1))
        self.model.c2 = nn.Sequential(nn.Conv3d(channels, 3, 1))
        self.st = SpatialTransformer(config.img_size)

    def forward(self, mov, fix, init_flow=None, return_all_flows=False):
        inputs = torch.cat((mov, fix), 1)
        flow = 0.01 * (self.model.transformer.patch_embed.proj(inputs) + self.model.c1(inputs) + self.model.c2(inputs))
        if init_flow is not None:
            flow = flow + init_flow
        return self.st(mov, flow), flow


def test_expansion_preserves_nontrivial_parent_and_new_inputs_receive_gradients(monkeypatch, tmp_path):
    monkeypatch.setattr(cascade, "_build_level2", lambda config, _: TinyL2(config))
    config = copy.deepcopy(CONFIGS["CTCF-CascadeA-Mamba"])
    config.img_size, config.level1_base_ch, config.level3_base_ch = (32, 32, 32), 2, 2
    config.l3_svf = False
    scalar = cascade.CTCFCascadeA(config).eval()
    nn.init.normal_(scalar.level1.out.weight, std=0.002)
    nn.init.normal_(scalar.level3.out.weight, std=0.002)
    path = tmp_path / "parent.pth"
    torch.save({"state_dict": scalar.state_dict(), "precision": "bf16"}, path)
    config.image_channels = 2
    expanded = cascade.CTCFCascadeA(config).eval()
    report = load_checkpoint_state(expanded, str(path), True, expand_image_channels=True)
    assert len(report["expanded_input_keys"]) == 5
    moving, fixed = torch.rand(1, 1, 32, 32, 32), torch.rand(1, 1, 32, 32, 32)
    reference = scalar(moving, fixed)[1].detach()
    assert float(reference.abs().max()) > 0.01
    for t2 in (moving, torch.rand_like(moving)):
        output = expanded(torch.cat((moving, t2), 1), torch.cat((fixed, 1 - t2), 1))[1]
        torch.testing.assert_close(output, reference, rtol=2e-4, atol=2e-5)
    output.square().mean().backward()
    for key in report["expanded_input_keys"]:
        gradient = dict(expanded.named_parameters())[key].grad[:, 1]
        assert torch.isfinite(gradient).all() and float(gradient.abs().sum()) > 0
    state, keys = expanded.expand_scalar_inputs(expanded.state_dict())
    assert not keys
    malformed = dict(scalar.state_dict())
    malformed[report["expanded_input_keys"][0]] = state[report["expanded_input_keys"][0]]
    with pytest.raises(ValueError, match="mixes"):
        expanded.expand_scalar_inputs(malformed)


def test_two_channel_ncc_error_maps_do_not_mix_contrasts():
    a, b = torch.rand(1, 2, 12, 13, 14), torch.rand(1, 2, 12, 13, 14)
    together = FlowRefiner3D._local_ncc_map(a, b, win=3)
    separate = torch.cat([FlowRefiner3D._local_ncc_map(a[:, i : i + 1], b[:, i : i + 1], win=3) for i in range(2)], 1)
    torch.testing.assert_close(together, separate)


@pytest.fixture
def manifest_file(tmp_path):
    groups = []
    shape = (12, 13, 14)
    for split, count in (("Train", 4), ("Val", 2)):
        rows = []
        for i in range(count):
            name = f"{split}_{i}"
            t1 = np.random.default_rng(i).random(shape, dtype=np.float32)
            with (tmp_path / f"{name}.pkl").open("wb") as stream:
                pickle.dump((t1, np.full(shape, 41, dtype=np.int16)), stream)
            np.save(tmp_path / f"{name}.npy", 1 - t1)
            rows.append(
                {
                    "id": name,
                    "ixi_id": name,
                    "split": split,
                    "t1": f"{name}.pkl",
                    "t2": f"{name}.npy",
                    "t1_sha256": sha256_file(tmp_path / f"{name}.pkl"),
                    "t2_sha256": sha256_file(tmp_path / f"{name}.npy"),
                }
            )
        groups.append(rows)
    path = tmp_path / "manifest.json"
    path.write_text(
        json.dumps(
            {
                "reviewed": True,
                "shape": shape,
                "training": groups[0],
                "validation": groups[1],
                "expected_counts": {"training": 4, "validation": 2},
                "validation_pairs": [["Val_0", "Val_1"], ["Val_1", "Val_0"]],
            }
        )
    )
    return path


def test_paired_data_separates_input_observation_and_remaps_validation_labels(manifest_file):
    manifest = read_paired_manifest(manifest_file)
    b0, b1, b2 = [IXIPairedDataset(manifest, arm) for arm in ("b0", "b1", "b2")]
    x0, y0, _, _ = b0[(0, 1)]
    x1, y1, tx, _ = b1[(0, 1)]
    x2, _, _, _ = b2[(0, 1)]
    assert torch.equal(x0, x1) and torch.equal(y0, y1) and torch.equal(x0[0], x0[1])
    assert torch.equal(x2[1:], tx) and not torch.equal(x2, x0)
    Path(b0.rows[0]["t2"]).unlink()
    assert torch.equal(b0[(0, 1)][0], x0)
    np.save(b1.rows[0]["t2"], np.zeros(manifest["shape"], np.float32))
    assert torch.equal(b1[(0, 1)][0], x1)
    labels = IXIPairedDataset(manifest, "b0", validation=True)[0][2]
    assert labels.unique().tolist() == [21]  # FreeSurfer 41 -> the established dense protocol index.
    for row in manifest["validation"]:
        Path(row["t2"]).unlink()
    t1_only = read_paired_manifest(manifest_file, require_t2=False)
    for arm in ("b0", "b1"):
        assert IXIPairedDataset(t1_only, arm, validation=True)[0][0].shape[0] == 2
    with pytest.raises(FileNotFoundError):
        IXIPairedDataset(t1_only, "b2", validation=True)[0]
    sampler = IXIPairSampler(4, 0, "frozen")
    first = list(sampler)
    assert first == list(sampler) and sorted(a for a, _ in first) == list(range(4))
    assert all(a != b for a, b in first)
    sampler.set_epoch(1)
    assert list(sampler) != first


@pytest.mark.parametrize("defect", ("review", "leakage", "reverse"))
def test_manifest_rejects_unreviewed_or_invalid_split_and_pairs(manifest_file, defect):
    manifest = json.loads(manifest_file.read_text())
    if defect == "review":
        manifest["reviewed"] = False
    elif defect == "leakage":
        manifest["validation"][0]["ixi_id"] = manifest["training"][0]["ixi_id"]
    else:
        manifest["validation_pairs"][1] = manifest["validation_pairs"][0]
    manifest_file.write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        read_paired_manifest(manifest_file)


def test_readiness_rechecks_the_prepared_image_bytes(manifest_file):
    manifest = read_paired_manifest(manifest_file, verify_bytes=True)
    np.save(manifest["training"][0]["t2"], np.zeros(manifest["shape"], np.float32))
    with pytest.raises(ValueError, match="bytes changed"):
        read_paired_manifest(manifest_file, verify_bytes=True)


def test_paired_loader_resume_binding_and_common_pair_stream(manifest_file):
    args = parse_args(["--ds", "IXI", "--ixi_pairs", str(manifest_file), "--image_channels", "2", "--num_workers", "0"])
    first, _ = ixi_paired_loaders(args)
    args.exp, args.resume = "other_runtime_name", "last.pth"
    resumed, _ = ixi_paired_loaders(args)
    assert first.sampler.fingerprint == resumed.sampler.fingerprint
    args.ixi_arm = "b1"
    auxiliary, _ = ixi_paired_loaders(args)
    assert first.sampler.fingerprint != auxiliary.sampler.fingerprint
    assert list(first.sampler) == list(auxiliary.sampler)
    args.w_reg = 4
    changed, _ = ixi_paired_loaders(args)
    assert changed.sampler.fingerprint != auxiliary.sampler.fingerprint


def test_finalisation_requires_review_and_preserves_the_frozen_cohort(manifest_file, tmp_path):
    manifest = json.loads(manifest_file.read_text())
    out = tmp_path / "prepared"
    rows, reviewed = [], []
    geometry = {"direction": list(np.eye(3).ravel()), "orientation_qc_passed": True}
    mapping = tmp_path / "mapping.csv"
    mapping.write_text("synthetic crosswalk")
    for group in ("training", "validation"):
        for row in manifest[group]:
            record = row | {
                "t1": str(tmp_path / row["t1"]),
                "t2": str(tmp_path / row["t2"]),
                "shape": manifest["shape"],
                "flags": [],
                "qc": {"laterality": {"passed": True}} if row["split"] == "Train" else {},
                "match": "demographics",
                "binding": {"geometry": geometry, "sources": {str(tmp_path / row["t1"]): row["t1_sha256"]}},
            }
            folder = out / row["split"] / row["id"]
            folder.mkdir(parents=True)
            (folder / "qc.json").write_text(json.dumps(record))
            rows.append({"split": row["split"], "processed_file": row["id"] + ".pkl", "t1_file": "IXI-Guys-T1.nii.gz"})
            reviewed.append(row["id"])
    (out / "identity_controls.json").write_text(
        json.dumps({"geometry": geometry, "mapping_sha256": sha256_file(mapping), "controls": []})
    )
    args = SimpleNamespace(out=out, mapping=mapping, expected_train=4, expected_val=2, reviewed=None, review_extra=None)
    with pytest.raises(ValueError, match="reviewed"):
        finalise(args, rows)
    args.reviewed = tmp_path / "reviewed.txt"
    args.reviewed.write_text("\n".join(reviewed))
    finalise(args, rows)
    prepared = read_paired_manifest(out / "manifest.json", verify_bytes=True)
    assert prepared["validation_pairs"] == [["Val_0", "Val_1"], ["Val_1", "Val_0"]]
    report = out / "Train" / manifest["training"][0]["id"] / "qc.json"
    record = json.loads(report.read_text())
    report.write_text(json.dumps(record | {"qc": {}}))
    with pytest.raises(ValueError, match="Every Train"):
        finalise(args, rows)
    report.write_text(json.dumps(record))
    (tmp_path / manifest["training"][0]["t1"]).write_bytes(b"changed T1 source")
    with pytest.raises(ValueError, match="source changed"):
        finalise(args, rows)


def test_duplicate_t2_has_identical_loss_and_field_gradients():
    runner = Runner.__new__(Runner)
    runner.args = SimpleNamespace(ixi_arm="b1")
    runner.ctx = SimpleNamespace(ncc=NCCVxm(win=(3, 3, 3)), reg_model=RegisterModel((12, 13, 14)))
    x, y = torch.rand(1, 2, 12, 13, 14), torch.rand(1, 2, 12, 13, 14)
    fields = [torch.full((1, 3, 12, 13, 14), 0.13, requires_grad=True) for _ in range(2)]
    native_warp = ModelWarp((12, 13, 14))
    warped = [native_warp(a, f) for a, f in zip((x, y), fields, strict=True)]
    d1, d2 = runner.contrast_terms(x, y, *warped, *fields, (x[:, :1], y[:, :1]))
    torch.testing.assert_close(d1, d2)
    g1 = torch.autograd.grad(d1, fields, retain_graph=True)
    g2 = torch.autograd.grad((d1 + d2) / 2, fields)
    for a, b in zip(g1, g2, strict=True):
        torch.testing.assert_close(a, b)
    runner.args.ixi_arm = "b0"
    assert runner.contrast_terms(x, y, *warped, *fields, None)[1] == 0


@pytest.mark.parametrize("rotated", (False, True))
def test_t2_resampling_uses_composed_pull_in_physical_coordinates_once(rotated):
    z, y, x = np.mgrid[:14, :15, :16]
    t1 = (1 + x + 2 * y + z).astype(np.float32)
    t1[[0, -1]] = 0
    raw1 = sitk.GetImageFromArray(t1)
    raw2 = sitk.GetImageFromArray((x * x + y + 4 * z).astype(np.float32))
    geometry = {"spacing": (1, 1, 1), "direction": tuple(np.eye(3).ravel()), "origin": (0, 0, 0)}
    if rotated:
        geometry.update(direction=(0, -1, 0, 1, 0, 0, 0, 0, 1), origin=(14, 0, 0))
    first, second = sitk.AffineTransform(3), sitk.Euler3DTransform()
    first.SetMatrix((1, 0.04, 0, 0, 1, 0, 0, 0, 1))
    first.SetTranslation((0.2, 0.1, 0))
    second.SetRotation(0, 0, 0.01)
    second.SetTranslation((-0.1, 0, 0.2))
    output, aligned, qc, _, _ = prepare_arrays(t1, raw1, raw2, geometry, processed_to_t1=first, t1_to_t2=second)
    transform = sitk.CompositeTransform([second, first])
    physical = np.asarray(geometry["direction"]).reshape(3, 3) @ np.stack((x, y, z)).reshape(3, -1)
    physical += np.asarray(geometry["origin"])[:, None]
    first_points = np.asarray(first.GetMatrix()).reshape(3, 3) @ physical + np.asarray(first.GetTranslation())[:, None]
    second_points = (
        np.asarray(second.GetMatrix()).reshape(3, 3) @ first_points + np.asarray(second.GetTranslation())[:, None]
    )
    expected = ndimage.map_coordinates(sitk.GetArrayFromImage(raw2), second_points[::-1], order=1).reshape(t1.shape)
    inside = np.all((second_points >= 0) & (second_points < np.array(raw2.GetSize())[:, None] - 1), axis=0).reshape(
        t1.shape
    )
    lo, hi = qc["t2_percentiles"]
    expected = np.where(t1 > 0, np.clip((expected - lo) / (hi - lo), 0, 1), 0)
    np.testing.assert_allclose(output[inside], expected[inside], atol=1e-6)
    assert not np.array_equal(aligned, t1) and output.dtype == np.float32
    reverse = sitk.CompositeTransform([first, second])
    assert transform.TransformPoint((4, 5, 6)) != reverse.TransformPoint((4, 5, 6))
    assert np.count_nonzero(output[t1 == 0]) == 0
    first.SetMatrix((-1, 0, 0, 0, 1, 0, 0, 0, 1))
    with pytest.raises(ValueError, match="reflects"):
        prepare_arrays(t1, raw1, raw2, geometry, processed_to_t1=first, t1_to_t2=second)


def test_canonical_consensus_composes_both_legacy_legs():
    # Affine coordinate ramps avoid the resampling blur of a generic image under two warps.
    shape = (17, 19, 21)
    grid = torch.stack(torch.meshgrid(*(torch.arange(n) for n in shape), indexing="ij"))[None].float()
    first, second = torch.eye(3), torch.eye(3)
    first[0, 1], second[1, 0] = 0.06, 0.08
    canonical = [torch.einsum("ij,bjdhw->bidhw", matrix, grid) - grid for matrix in (first, second)]
    scale = torch.tensor([(n - 1) / n for n in shape]).reshape(1, 3, 1, 1, 1)
    fields = [(grid + field + 0.5) * scale - grid for field in canonical]
    tags = {"moving": 1, "fixed": 2, "intermediate": 3}
    legs = {(3, 2): fields[0], (1, 3): fields[1]}
    parent = SimpleNamespace(flow=lambda moving, fixed: legs[(int(moving[0, 0, 0, 0, 0]), int(fixed[0, 0, 0, 0, 0]))])
    from unittest.mock import patch

    with patch(
        "experiments.dispatch1.eval_consensus.image",
        side_effect=lambda name: torch.full((1, 1, *shape), float(tags[name])),
    ):
        actual = bridge(parent, "moving", "fixed", "intermediate", torch.device("cpu"))
    inside = (slice(None), slice(None), slice(4, -4), slice(4, -4), slice(4, -4))
    expected = torch.einsum("ij,bjdhw->bidhw", second @ first, grid)
    sequential = _warp(_warp(grid, canonical[1]), canonical[0])
    torch.testing.assert_close((grid + actual)[inside], expected[inside], atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(_warp(grid, actual)[inside], sequential[inside], atol=1e-5, rtol=1e-5)
    wrong = _warp(_warp(grid, canonical[0]), canonical[1])
    assert float((sequential[inside] - wrong[inside]).abs().max()) > 0.01


def test_saved_byte_certificate_and_raw_scores_survive_repair_failure(tmp_path, monkeypatch):
    labels = torch.ones((1, 1, 12, 13, 14), dtype=torch.long)
    flow = torch.zeros((1, 3, 12, 13, 14))
    rows, report = field_scoring.score_saved_field(
        flow, "a__b", 0, labels, labels, tmp_path / "ok", profile=metric_profile_for("IXI")
    )
    assert report["global_homeomorphism"] and len([key for key in rows["FF"] if key.startswith("dice_lbl_")]) == 30

    def fail(_):
        raise FloatingPointError("synthetic repair failure")

    monkeypatch.setattr(field_scoring, "repair_to_identity_boundary", fail)
    rows, report = field_scoring.score_saved_field(flow, "a__b", 0, labels, labels, tmp_path / "failed")
    assert "FF" in rows and "REP" not in rows and not report["global_homeomorphism"]
    assert (tmp_path / "failed" / "metrics.json").is_file() and report["errors"]
    assert field_scoring.load_saved_scores(tmp_path / "ok")[1]["global_homeomorphism"]
    (tmp_path / "ok" / "REP.npz").write_bytes(b"changed field")
    with pytest.raises(ValueError, match="changed"):
        field_scoring.load_saved_scores(tmp_path / "ok")


def test_paired_statistics_average_directions_before_counting_samples():
    cases = ("a__b", "b__a", "c__d", "d__c")
    control = [{"case_id": case, "dice_mean": 0.7} for case in cases]
    rows = [{"case_id": case, "dice_mean": value} for case, value in zip(cases, (0.8, 0.6, 0.72, 0.72), strict=True)]
    summary = field_scoring.paired_dice_delta(rows, control)
    assert summary["pairs"] == 2 and summary["mean"] == pytest.approx(0.01)
    with pytest.raises(ValueError):
        field_scoring.paired_dice_delta(rows[:-1], control[:-1])


def test_training_cli_keeps_scalar_default_and_accepts_explicit_paired_recipe():
    assert parse_args([]).image_channels is None
    args = parse_args(["--ds", "IXI", "--ixi_pairs", "manifest.json", "--image_channels", "2", "--ixi_arm", "b1"])
    assert args.ixi_arm == "b1" and args.image_channels == 2


def test_train_readiness_and_evaluation_share_the_recipe(tmp_path, monkeypatch):
    args = SimpleNamespace(
        manifest=tmp_path / "manifest.json",
        checkpoint=tmp_path / "last.pth",
        precision="bf16",
        device=torch.device("cpu"),
        arm="b1",
    )
    configured = training_args(args, "b1")
    assert (
        configured.lr == 2e-5
        and configured.w_reg == 4
        and all(getattr(configured, key) == value for key, value in RECIPE.items())
    )
    trained = vars(configured) | {"ixi_manifest_sha256": "fixed"}
    for key, wrong in (("lr", 1e-4), ("w_reg", 1)):
        torch.save({"arguments": trained | {key: wrong}}, args.checkpoint)
        with pytest.raises(ValueError, match="differs"):
            evaluate(args, {"fingerprint": "fixed"})


def test_consensus_certification_is_explicit_and_diagnostics_are_not_certified(tmp_path):
    arms = (
        "direct",
        "smooth",
        "bank0_k1",
        "bank0_k4",
        "bank0_k16",
        "bank1_k1",
        "bank1_k4",
        "bank1_k16",
        "dev_transductive_k16",
    )
    assert sum(len(certificate_modes(arm)) for arm in arms) == 5
    flow, labels = torch.zeros(1, 3, 12, 12, 12), torch.ones(1, 1, 12, 12, 12).long()
    rows, report = field_scoring.score_saved_field(
        flow, "a__b", 0, labels, labels, tmp_path / "diagnostic", certify_modes=()
    )
    assert set(rows) == {"FF", "REP"} and report["certification"] == "NOT_RUN" and not report["global_homeomorphism"]
    assert not any(key.startswith("certificate_") for key in report)
    _, certified = field_scoring.score_saved_field(
        flow, "a__b", 0, labels, labels, tmp_path / "primary", certify_modes=("REP",)
    )
    assert certified["global_homeomorphism"] and "certificate_FF" not in certified
    assert "sdlogj_plus3" not in rows["FF"] and "sdlogj" in rows["FF"]


def test_laterality_uses_relative_lps_positions_after_the_full_transform():
    labels = np.zeros((12, 14, 16), np.int16)
    for left, right, y in ((2, 41, 3), (3, 42, 9)):
        labels[8:10, y : y + 2, 4:8], labels[2:4, y : y + 2, 4:8] = left, right
    geometry = {"direction": (0, 0, 1, 1, 0, 0, 0, 1, 0), "spacing": (1, 1, 1), "origin": (100, -200, 30)}
    reference = processed_image(np.ones(labels.shape), geometry)
    fitted = sitk.Euler3DTransform()
    fitted.SetTranslation((50, 0, 0))
    assert laterality_report(labels, reference, fitted)["passed"]
    fitted.SetRotation(0, 0, np.pi)
    assert not laterality_report(labels, reference, fitted)["passed"]
    geometry["direction"] = (0, 0, -1, 1, 0, 0, 0, 1, 0)
    assert not laterality_report(labels, processed_image(np.ones(labels.shape), geometry), sitk.Euler3DTransform())[
        "passed"
    ]


def test_rigid_registration_starts_from_headers_and_reports_center_motion(monkeypatch):
    from unittest.mock import MagicMock

    fixed, moving = sitk.Image([12, 14, 16], sitk.sitkFloat32), sitk.Image([22, 24, 26], sitk.sitkFloat32)
    fixed.SetOrigin((-20, 30, 40))
    moving.SetOrigin((100, -300, 200))
    method = MagicMock()
    method.Execute.side_effect = lambda *_: method.SetInitialTransform.call_args.args[0]
    monkeypatch.setattr("tools.analysis.prepare_ixi.sitk.ImageRegistrationMethod", lambda: method)
    result, report = register(fixed, moving, affine=False)
    assert report["initialisation"] == "header_identity"
    assert result.TransformPoint((1, 2, 3)) == (1, 2, 3)
    result.SetTranslation((3, 4, 0))
    result.SetRotation(0, 0, np.pi / 6)
    motion = rigid_motion(result, fixed)
    assert motion["rigid_shift_mm"] == pytest.approx(5) and motion["rigid_angle_deg"] == pytest.approx(30)


@pytest.mark.parametrize("arm", ("b0", "b1", "b2"))
def test_native_trainer_executes_the_paired_step_with_a_fresh_optimizer(arm, monkeypatch):
    monkeypatch.setattr(cascade, "_build_level2", lambda config, _: TinyL2(config))
    config = copy.deepcopy(CONFIGS["CTCF-CascadeA-Mamba"])
    config.img_size, config.level1_base_ch, config.level3_base_ch = (32, 32, 32), 2, 2
    monkeypatch.setitem(CONFIGS, "CTCF-CascadeA-Mamba", config)
    args = parse_args(
        [
            "--ds",
            "IXI",
            "--ixi_pairs",
            "unused.json",
            "--ixi_arm",
            arm,
            "--config",
            "CTCF-CascadeA-Mamba",
            "--image_channels",
            "2",
            "--finetune",
            "1",
            "--l3_svf",
            "0",
            "--w_reg",
            "4",
            "--precision",
            "fp32",
        ]
    )
    runner = Runner(args, torch.device("cpu"))
    inputs = [torch.rand(1, 2, 32, 32, 32) for _ in range(2)]
    batch = inputs + [x[:, :1].clone() for x in inputs]
    loss, log = runner.train_step(batch, 0)
    assert torch.isfinite(loss) and "ncc_t1" in log and "ncc_t2" in log
    loss.backward()
    runner.optimizer.step()
    assert runner.optimizer.state and all(torch.isfinite(p).all() for p in runner.model.parameters())


def test_preparation_checks_all_train_laterality_and_refreshes_old_cache(tmp_path, monkeypatch):
    processed, raw, out = [tmp_path / name for name in ("processed", "raw", "out")]
    raw.mkdir()
    out.mkdir()
    geometry = {
        "direction": list(np.eye(3).ravel()),
        "spacing": [1, 1, 1],
        "origin": [0, 0, 0],
        "orientation_qc_passed": True,
        "calibration_subjects": ["IXI000"],
        "coverage_min": 0.99,
        "correlation_min": 0.8,
        "rigid_shift_mm_max": 20,
        "rigid_angle_deg_max": 20,
    }
    geometry_path = out / "geometry.json"
    geometry_path.write_text(json.dumps(geometry))
    t1 = np.ones((10, 12, 14), np.float32)
    labels = np.zeros(t1.shape, np.int16)
    labels[2:4, 2:4, 9:11], labels[2:4, 2:4, 2:4] = 2, 41
    labels[6:8, 8:10, 9:11], labels[6:8, 8:10, 2:4] = 3, 42
    rows, calls = [], []
    for index, split in enumerate(("Train", "Train", "Val")):
        folder = processed / split
        folder.mkdir(exist_ok=True, parents=True)
        with (folder / f"subject_{index}.pkl").open("wb") as stream:
            pickle.dump((t1, labels), stream)
        sitk.WriteImage(sitk.GetImageFromArray(t1), str(raw / f"{index}.nii.gz"))
        rows.append(
            dict(
                split=split,
                processed_file=f"subject_{index}.pkl",
                ixi_id=f"IXI{index:03}",
                t1_file=f"{index}.nii.gz",
                t2_file=f"{index}.nii.gz",
                match="demographics",
            )
        )

    def fit(*args, **kwargs):
        calls.append(1)
        return (
            t1,
            t1,
            dict(brain_coverage=1, t1_correlation=1, rigid_shift_mm=0, rigid_angle_deg=0),
            sitk.AffineTransform(3),
            sitk.Euler3DTransform(),
        )

    monkeypatch.setattr("tools.analysis.prepare_ixi.prepare_arrays", fit)
    monkeypatch.setattr("tools.analysis.prepare_ixi.preview", lambda *args: None)
    args = SimpleNamespace(
        geometry=geometry_path, limit=0, shards=1, shard=0, out=out, processed=processed, raw=raw, transforms=None
    )
    prepare(args, rows)
    assert json.loads((out / "Train/subject_1/qc.json").read_text())["qc"]["laterality"]["passed"]
    assert "laterality" not in json.loads((out / "Val/subject_2/qc.json").read_text())["qc"]
    report = out / "Train/subject_1/qc.json"
    record = json.loads(report.read_text())
    del record["qc"]["laterality"]
    report.write_text(json.dumps(record))
    prepare(args, rows)
    assert len(calls) == 4 and json.loads(report.read_text())["qc"]["laterality"]["passed"]


def test_calibration_fits_all_orientations_before_shortlisting(tmp_path, monkeypatch):
    processed, raw, out = [tmp_path / name for name in ("processed", "raw", "out")]
    (processed / "Train").mkdir(parents=True)
    raw.mkdir()
    out.mkdir()
    z, y, x = np.mgrid[:16, :16, :16]
    image = (1 + x * x + 3 * y + z).astype(np.float32)
    labels = np.zeros_like(image, dtype=np.int16)
    labels[3:6, 3:6, 10:13], labels[3:6, 3:6, 2:5] = 2, 41
    labels[9:12, 9:12, 10:13], labels[9:12, 9:12, 2:5] = 3, 42
    rows = []
    for index in range(2):
        row = {
            "split": "Train",
            "processed_file": f"subject_{index}.pkl",
            "ixi_id": f"IXI{index:03}",
            "t1_file": f"IXI{index:03}-Guys-T1.nii.gz",
        }
        with (processed / "Train" / row["processed_file"]).open("wb") as stream:
            pickle.dump((image, labels), stream)
        sitk.WriteImage(sitk.GetImageFromArray(image), str(raw / row["t1_file"]))
        rows.append(row)
    fitted_directions = []

    def fit(fixed, moving, *, affine, coarse):
        assert coarse and fixed.GetSize() == moving.GetSize() == (4, 4, 4)
        direction = np.asarray(fixed.GetDirection()).reshape(3, 3)
        fitted_directions.append(tuple(direction.ravel()))
        matrix = direction.T
        transform = sitk.AffineTransform(3)
        if np.linalg.det(direction) < 0:
            matrix = np.diag([-1, 1, 1]) @ matrix
            transform.SetTranslation((15, 0, 0))
        assert np.linalg.det(matrix) > 0
        transform.SetMatrix(tuple(matrix.ravel()))
        return transform, {}

    monkeypatch.setattr("tools.analysis.prepare_ixi.register", fit)
    calibrate(SimpleNamespace(processed=processed, raw=raw, out=out), rows)
    report = json.loads((out / "orientation_scores.json").read_text())
    assert len(set(fitted_directions)) == len(report["screening_candidates"]) == 48
    assert any(not row["train_check"]["passed"] for row in report["screening_candidates"])
    selected = json.loads((out / "geometry_0.json").read_text())
    assert selected["orientation_qc_passed"] and selected["fitted_correlation"] == pytest.approx(1)


def test_coarse_affine_preserves_overlap_with_distant_physical_origins():
    z, y, x = np.mgrid[:32, :36, :40]
    values = np.exp(-((x - 18) ** 2 / 45 + (y - 17) ** 2 / 35 + (z - 15) ** 2 / 30))
    values += 0.8 * np.exp(-((x - 25) ** 2 / 20 + (y - 12) ** 2 / 25 + (z - 21) ** 2 / 16))
    values[values < 0.01] = 0
    fixed = sitk.GetImageFromArray(values.astype(np.float32))
    moving = sitk.GetImageFromArray((3 * values).astype(np.float32))
    fixed.SetOrigin((-200, 300, 50))
    moving.SetOrigin((120, -80, -200))
    transform, report = register(fixed, moving, affine=True, coarse=True)
    center = fixed.TransformContinuousIndexToPhysicalPoint((19.5, 17.5, 15.5))
    expected = moving.TransformContinuousIndexToPhysicalPoint((19.5, 17.5, 15.5))
    np.testing.assert_allclose(transform.TransformPoint(center), expected, atol=0.25)
    aligned = sitk.GetArrayFromImage(sitk.Resample(moving, fixed, transform, sitk.sitkLinear, 0))
    assert np.corrcoef(values.ravel(), aligned.ravel())[0, 1] > 0.99
    assert report["coarse"] and report["initialisation"] == "geometry"
