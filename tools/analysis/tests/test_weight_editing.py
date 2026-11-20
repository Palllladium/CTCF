import copy
import json
import os
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from datasets.IXI import IXIPairedDataset, read_ixi_manifest
from experiments import (
    contrast_audit,
    contrast_supervision as supervision,
    train_CTCF as trainer,
    weight_interventions as intervention,
)
from experiments.core.data_loaders import ixi_paired_loaders
from experiments.core.distributed import restored_state_digest
from experiments.core.inference_metrics import metric_profile_for
from experiments.core.weight_editing import (
    error_counts,
    feature_moments,
    fit_direction,
    group_mean,
    joint_criterion,
    matched_features,
    project_weight,
    projection_report,
    protected_weight,
)
from experiments.eval_matching import cases
from experiments.train_CTCF import Runner, paired_ixi_dice, parse_args
from models.CTCF import model as cascade
from models.CTCF.configs import CONFIGS
from tools.analysis.run_artifacts import sha256_file
from tools.analysis.tests.test_contrasts import TinyL2, manifest_file as manifest_file
from utils import RegisterModel
from utils.field import _warp


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


@pytest.mark.parametrize("shape,padding", (((6, 7, 8), 1), ((5, 6, 7), 0), ((5, 6, 7), 2)))
def test_compiled_projection_matches_activation_erasure_including_padding(shape, padding):
    generator = torch.Generator().manual_seed(41)
    x = torch.randn(2, 5, *shape, dtype=torch.double, generator=generator)
    weight = torch.randn(3, 5, 3, 3, 3, dtype=torch.double, generator=generator)
    bias = torch.randn(3, dtype=torch.double, generator=generator)
    v = torch.randn(5, dtype=torch.double, generator=generator)
    v /= v.norm()
    projected = x - 0.5 * torch.einsum("bczyx,c->bzyx", x, v)[:, None] * v[None, :, None, None, None]
    expected = F.conv3d(projected, weight, bias, padding=padding)
    actual = F.conv3d(x, project_weight(weight, v, 0.5), bias, padding=padding)
    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)


def test_random_controls_match_weight_change_not_just_the_vector_length():
    weight = torch.randn(3, 8, 3, 3, 3)
    v, r = torch.randn(8), torch.randn(8)
    norm = (project_weight(weight, v, 0.5) - weight).norm()
    scale = norm / (project_weight(weight, r, 1) - weight).norm()
    assert (project_weight(weight, r, scale) - weight).norm() == pytest.approx(float(norm), rel=2e-6)


def test_two_sided_protection_covers_cortex_crossed_only_between_the_endpoints():
    shape = (15, 16, 17)
    moving = torch.zeros(1, 1, *shape, dtype=torch.long)
    moving[:, :, 7] = 2
    fixed = torch.zeros_like(moving)
    first, second = [torch.zeros(1, 3, *shape) for _ in range(2)]
    first[:, 0] = -2
    second[:, 0] = 2
    assert _warp(moving.float(), first, "nearest")[0, 0, 7, 8, 8] == 0
    assert _warp(moving.float(), second, "nearest")[0, 0, 7, 8, 8] == 0
    assert _warp(moving.float(), (first + second) / 2, "nearest")[0, 0, 7, 8, 8] == 2
    weight = protected_weight(moving, fixed, first, second)
    assert weight[0, 0, 7, 8, 8] == 0
    old = _warp(moving.float(), first, "nearest") == 2
    for alpha in np.linspace(0, 1, 21):
        candidate = first + alpha * weight * (second - first)
        assert torch.equal(_warp(moving.float(), candidate, "nearest") == 2, old)


def test_protection_handles_empty_cortex_and_near_boundary_samples():
    labels = torch.zeros(1, 1, 8, 9, 10, dtype=torch.long)
    first = torch.zeros(1, 3, 8, 9, 10)
    second = first + 2
    assert torch.equal(protected_weight(labels, labels, first, second), torch.ones_like(labels).float())
    labels[:, :, 0, 2:4, 2:4] = 22
    first[:, 0] = -0.7
    weight = protected_weight(labels, labels, first, second)
    assert not weight[labels == 22].any()


def test_context_matching_and_pair_weighted_direction_use_only_supplied_features():
    generator = torch.Generator().manual_seed(3)
    features = torch.randn(1, 4, 8, 8, 8, generator=generator)
    error = torch.zeros(1, 1, 8, 8, 8, dtype=torch.bool)
    error[:, :, :4] = True
    correct = ~error
    features[:, 0] += error[0, 0] * 3
    selected = matched_features(
        features, error, correct, torch.ones_like(error).float() * 0.5, torch.zeros_like(error).long(), seed=42
    )
    assert selected is not None and selected[0].shape == selected[1].shape
    moment = feature_moments(*selected)
    direction = fit_direction([moment, moment])
    assert direction is not None and direction.norm() == pytest.approx(1)
    assert direction[0] > 0.9
    assert fit_direction([]) is None
    assert matched_features(features, torch.zeros_like(error), correct, error.float(), error.long(), seed=42) is None


def test_partition_excludes_known_probes_from_sealed_and_keeps_pairs_disjoint():
    names = [f"subject_{i}" for i in range(397)]
    known = set(names[:100])
    groups = intervention.split_subjects(names, known)
    assert {k: len(v) for k, v in groups.items()} == {"discovery": 237, "selection": 80, "sealed": 80}
    assert not known & set(groups["sealed"])
    assert len(set().union(*map(set, groups.values()))) == 397
    for group in ("selection", "sealed"):
        pairs = intervention.reciprocal_pairs(groups[group])
        assert len(pairs) == 80
        assert sorted(a for a, _ in pairs) == sorted(groups[group])
        assert all([b, a] in pairs for a, b in pairs)
    assert groups == intervention.split_subjects(names[::-1], known)


def test_label_permissions_do_not_change_native_pairs_or_the_four_original_tensors(manifest_file, tmp_path):
    manifest = read_ixi_manifest(manifest_file)
    original = IXIPairedDataset(manifest, "b1")
    allowed = {row["id"] for row in manifest["training"][:2]}
    dataset = IXIPairedDataset(manifest, "b1", label_ids=allowed)
    for pair, eligible in (((0, 1), True), ((0, 2), False)):
        current = dataset[pair]
        assert len(current) == 7 and current[-1] is eligible
        for a, b in zip(current[:4], original[pair], strict=True):
            assert torch.equal(a, b)
    assert dataset[0, 1][4].unique().tolist() == [21]
    assert not dataset[0, 2][5].any()
    path = tmp_path / "labels.txt"
    path.write_text("\n".join(sorted(allowed)) + "\n")
    args = parse_args(["--ds", "IXI", "--ixi_pairs", str(manifest_file), "--num_workers", "0"])
    first, _ = ixi_paired_loaders(args)
    args.w_dice, args.ixi_label_ids = 1, str(path)
    second, _ = ixi_paired_loaders(args)
    assert list(first.sampler) == list(second.sampler)
    assert first.sampler.fingerprint != second.sampler.fingerprint
    path.write_text("\n".join(row["id"] for row in manifest["training"][1:3]) + "\n")
    third, _ = ixi_paired_loaders(args)
    assert third.sampler.fingerprint != second.sampler.fingerprint
    args.ixi_label_ids = ""
    with pytest.raises(ValueError, match="requires both"):
        ixi_paired_loaders(args)


def test_paired_dice_has_real_flow_gradients_and_no_gradient_from_forbidden_labels():
    shape = (10, 11, 12)
    labels = torch.zeros(2, 1, *shape, dtype=torch.long)
    labels[:, :, 3:7, 3:7, 3:8] = 2
    fixed = labels.roll(1, dims=2)
    xy = torch.zeros(2, 3, *shape, requires_grad=True)
    yx = torch.zeros_like(xy, requires_grad=True)
    sampler = RegisterModel(shape)
    loss = paired_ixi_dice(xy, yx, labels, fixed, torch.tensor([True, False]), sampler)
    gradient = torch.autograd.grad(loss, (xy, yx))
    assert all(torch.isfinite(g).all() and g[0].abs().sum() > 0 and not g[1].any() for g in gradient)
    other = fixed.clone()
    other[1] = 36
    assert paired_ixi_dice(xy, yx, labels, other, torch.tensor([True, False]), sampler) == loss
    zero = paired_ixi_dice(xy, yx, labels, fixed, torch.tensor([False, False]), sampler)
    assert zero == 0 and not torch.autograd.grad(zero, xy)[0].any()


def test_real_trainer_accepts_separate_ixi_labels_and_preserves_zero_weight_path(monkeypatch):
    monkeypatch.setattr(cascade, "_build_level2", lambda config, _: TinyL2(config))
    config = copy.deepcopy(CONFIGS["CTCF-CascadeA-Mamba"])
    config.img_size = (32, 32, 32)
    monkeypatch.setitem(CONFIGS, "CTCF-CascadeA-Mamba", config)
    args = parse_args(
        [
            "--ds",
            "IXI",
            "--config",
            "CTCF-CascadeA-Mamba",
            "--img_size",
            "32",
            "32",
            "32",
            "--image_channels",
            "2",
            "--l1_base_ch",
            "2",
            "--l3_base_ch",
            "2",
            "--l3_svf",
            "0",
            "--finetune",
            "1",
            "--ncc_mode",
            "stable",
            "--precision",
            "fp32",
        ]
    )
    args.ixi_pairs = "paired.json"
    args.w_dice, args.w_reg, args.ixi_arm = 1, 1, "b1"
    runner = Runner(args, torch.device("cpu"))
    x, y = [torch.rand(1, 2, 32, 32, 32) for _ in range(2)]
    labels = torch.zeros(1, 1, 32, 32, 32, dtype=torch.long)
    labels[:, :, 8:24, 8:24, 8:24] = 2
    batch = [x, y, x[:, :1], y[:, :1], labels, labels.roll(2, dims=2), torch.tensor([True])]
    loss, logs = runner.train_step(batch, 0)
    loss.backward()
    assert logs["dice"] > 0 and logs["dice_labelled_pairs"] == 1
    assert runner.model.level3.out.weight.grad.abs().sum() > 0
    runner.optimizer.zero_grad(set_to_none=True)
    runner.args.w_dice = 0
    reference, _ = runner.train_step(batch[:4], 0)
    current, _ = runner.train_step(batch, 0)
    torch.testing.assert_close(current, reference, rtol=0, atol=0)
    runner.args.w_dice = 1
    with pytest.raises(RuntimeError, match="separate labels"):
        runner.train_step(batch[:4], 0)


def test_aggregation_and_joint_gate_do_not_confuse_macro_gain_with_cortex_recovery():
    def table(macro, cortex, deep):
        rows = [
            {
                "case_id": case,
                "dice_mean": macro,
                **{f"dice_lbl_{i}": cortex if i in (2, 22) else deep for i in (2, 22, 7, 8, 9, 10, 27, 28, 29, 30)},
            }
            for case in ("a__b", "b__a", "c__d", "d__c")
        ]
        return group_mean(rows)

    first, second = table(0.78, 0.7, 0.8), table(0.784, 0.69, 0.82)
    good = joint_criterion(table(0.784, 0.7, 0.817), first, second)
    assert good["passed"] and first["unordered_pairs"] == 2
    assert not joint_criterion(second, first, second)["passed"]
    assert not joint_criterion(table(0.79, 0.7, 0.81), first, second)["passed"]


def test_selection_is_frozen_before_sealed_and_empty_summary_is_not_success(tmp_path, monkeypatch):
    args = SimpleNamespace(out=tmp_path)
    (tmp_path / "edits.json").write_text(json.dumps({"rows": [{"name": "l2_50_target"}, {"name": "l2_50_random0"}]}))
    (tmp_path / "splits.json").write_text("{}")
    monkeypatch.setattr(intervention, "verify", lambda _: {})

    def values(_args, _split, name, _mode):
        value = 0.8 if name.startswith("b0") else 0.81
        return {"means": {"dice_mean": value, "cortex": value, "deep_gray": value}}

    monkeypatch.setattr(intervention, "readout", values)
    intervention.select(args)
    assert (tmp_path / "selection.json").is_file()
    before = (tmp_path / "selection.json").read_bytes()
    intervention.select(args)
    assert (tmp_path / "selection.json").read_bytes() == before
    with pytest.raises((RuntimeError, ValueError)):
        intervention.summary(args)


def test_prepare_cli_binds_real_inputs_and_rejects_changed_source_weights(manifest_file, tmp_path):
    original = read_ixi_manifest(manifest_file)
    shape = original["shape"]
    manifest = {key: original[key] for key in ("reviewed", "shape")}
    groups = {}
    for split, key, count in (("Train", "training", 397), ("Val", "validation", 58)):
        groups[key] = [dict(original[key][0], id=f"{split}_{i}", ixi_id=f"{split}_{i}") for i in range(count)]
    manifest.update(
        groups,
        expected_counts={k: len(v) for k, v in groups.items()},
        validation_pairs=intervention.reciprocal_pairs([r["id"] for r in groups["validation"]]),
    )
    manifest_file.write_text(json.dumps(manifest))
    parent = tmp_path / "parent.pth"
    parent.write_bytes(b"parent")
    source, history = tmp_path / "source", tmp_path / "history"
    (source / "teacher_audit").mkdir(parents=True)
    (source / "teacher_audit/report.json").write_text(json.dumps({"rows": [{"case_id": "Train_2__Train_3"}]}))
    for family in ("finetune", "scratch"):
        (source / "regional" / family).mkdir(parents=True)
        (source / "regional" / family / "report.json").write_text('{"rows": []}')
    history.mkdir()
    (history / "probe_inputs.json").write_text(json.dumps({"train_pairs": [["Train_0", "Train_1"]]}))
    paths = []
    for seed in (0, 1):
        for arm in ("b0", "b1"):
            name = f"{arm}_s{seed}"
            folder = source / "registration" / name
            folder.mkdir(parents=True)
            weight = folder / "last.pth"
            weight.write_bytes(name.encode())
            paths.append(weight)
            recipe = {
                "parent_sha256": sha256_file(parent),
                "manifest_sha256": sha256_file(manifest_file),
                "initial_model_sha256": f"initial{seed}",
            }
            (folder / "recipe.json").write_text(json.dumps(recipe))
            settings = dict(
                seed=seed,
                ixi_arm=arm,
                ncc_mode="stable",
                precision="bf16",
                w_reg=1,
                w_dice=0,
                image_channels=2,
                finetune=1,
            )
            (folder / "report.json").write_text(
                json.dumps(
                    {
                        "arguments": settings,
                        "updates": 3990,
                        "checkpoint": str(weight),
                        "checkpoint_sha256": sha256_file(weight),
                    }
                )
            )
            endpoint = source / "stable/IXI" / name
            endpoint.mkdir(parents=True)
            (endpoint / "report.json").write_text(
                json.dumps(
                    {
                        "cases": 58,
                        "complete_certified_endpoint": True,
                        "binding": {"base_sha256": sha256_file(weight), "manifest_sha256": sha256_file(manifest_file)},
                    }
                )
            )
    for name in ("scratch_b0", "scratch_b1"):
        folder = source / "IXI" / name
        folder.mkdir(parents=True)
        (folder / "report.json").write_text(json.dumps({"cases": 58, "complete_certified_endpoint": True}))
    np.savez(tmp_path / "lpba.npz", image=np.zeros(shape, np.float32), labels=np.zeros(shape, np.int16))
    external = tmp_path / "external.json"
    external.write_text(
        json.dumps(
            {
                "shape": shape,
                "label_ids": list(range(1, 57)),
                "subjects": [
                    {"id": f"s{i}", "file": "lpba.npz", "sha256": sha256_file(tmp_path / "lpba.npz")} for i in range(40)
                ],
                "pairs": [[f"s{i}", f"s{(i + j) % 40}"] for i in range(40) for j in (1, 2, 3)],
            }
        )
    )
    out = tmp_path / "run"
    command = [
        sys.executable,
        "-X",
        "utf8",
        "-m",
        "experiments.weight_interventions",
        "prepare",
        "--out",
        str(out),
        "--manifest",
        str(manifest_file),
        "--external_manifest",
        str(external),
        "--checkpoint",
        str(parent),
        "--source",
        str(source),
        "--history_run",
        str(history),
    ]
    environment = os.environ | {"CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "1"}
    result = subprocess.run(command, capture_output=True, text=True, env=environment, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
    groups = intervention.read(out / "splits.json")["subjects"]
    assert len(groups["discovery"]) == 237 and not {"Train_0", "Train_1", "Train_2", "Train_3"} & set(groups["sealed"])
    weights = paths[0].read_bytes()
    paths[0].write_bytes(b"changed")
    result = subprocess.run(command, capture_output=True, text=True, env=environment, timeout=60)
    assert result.returncode != 0 and "checkpoint changed" in result.stderr
    paths[0].write_bytes(weights)
    ids = (out / "discovery_ids.txt").read_bytes()
    assert len(ids.splitlines()) == 237


def test_profile_really_exercises_only_allowed_subjects_without_changing_fit_cohort(manifest_file, tmp_path):
    args = parse_args(["--ds", "IXI", "--ixi_pairs", str(manifest_file), "--num_workers", "0"])
    path = tmp_path / "labels.txt"
    path.write_text("Train_0\nTrain_1\n")
    args.ixi_label_ids, args.w_dice = str(path), 1
    train, _ = supervision.profile_loaders(args)
    batches = list(train)
    assert len(batches) == 20 and all(batch[6].all() for batch in batches)
    assert len(ixi_paired_loaders(args)[0].dataset) == 4


def test_external_labels_are_not_reported_as_ixi_cortex():
    rows = [{"case_id": "a__b", "dice_mean": 0.8, "dice_lbl_2": 0.9, "dice_lbl_22": 0.7}]
    assert "cortex" in group_mean(rows)["means"]
    assert "cortex" not in group_mean(rows, regions={})["means"]


def test_unlabelled_fp16_pair_does_not_overflow_while_forming_a_zero_loss():
    flow = torch.full((1, 3, 32, 32, 32), 1e4, dtype=torch.float16, requires_grad=True)
    loss = paired_ixi_dice(flow, flow, None, None, torch.tensor([False]), None)
    assert torch.isfinite(loss) and loss == 0
    assert not torch.autograd.grad(loss, flow)[0].any()


def test_edit_checkpoints_preserve_architecture_and_bias_and_are_resumable(tmp_path, monkeypatch):
    sources = {}
    state = {}
    for path in intervention.HEADS.values():
        state[f"{path}.weight"] = torch.randn(3, 4, 3, 3, 3)
        state[f"{path}.bias"] = torch.randn(3)
    state["unchanged.weight"] = torch.randn(4, 4)
    for seed in (0, 1):
        path = tmp_path / f"source{seed}.pth"
        torch.save(
            {
                "state_dict": state,
                "precision": "bf16",
                "arguments": {"image_channels": 2},
                "optimizer": {"irrelevant": True},
            },
            path,
        )
        sources[f"b1_s{seed}"] = str(path)
        (tmp_path / f"directions_s{seed}.json").write_text(
            json.dumps({"directions": {head: [1, 2, 3, 4] for head in intervention.HEADS}})
        )
    monkeypatch.setattr(intervention, "verify", lambda _: {"weights": sources})
    args = SimpleNamespace(out=tmp_path)
    intervention.edits(args)
    edits = intervention.read(tmp_path / "edits.json")
    assert len(edits["rows"]) == 32
    for row in edits["rows"]:
        saved = torch.load(row["checkpoint"], weights_only=False)
        assert saved["state_dict"].keys() == state.keys() and "optimizer" not in saved
        assert saved["intervention"]["label_guidance"] is True
        changed = [key for key in state if not torch.equal(state[key], saved["state_dict"][key])]
        assert changed == [f"{intervention.HEADS[row['head']]}.weight"]
        assert row["edit_norm"] == pytest.approx(row["target_edit_norm"], rel=2e-6)
    intervention.edits(args)
    assert intervention.read(tmp_path / "edits.json") == edits


def test_final_readout_uses_real_saved_byte_certificates_and_keeps_raw_after_refusal(tmp_path, monkeypatch):
    checkpoint = tmp_path / "weights.pth"
    checkpoint.write_bytes(b"test base")
    (tmp_path / "inputs.json").write_text(json.dumps({"weights": {"b1_s0": str(checkpoint)}}))
    (tmp_path / "splits.json").write_text(json.dumps({"pairs": {"sealed": [["a", "b"], ["b", "a"]]}}))
    shape = (12, 13, 14)
    x, y = torch.rand(1, *shape), torch.rand(1, *shape)
    labels = torch.ones(1, *shape, dtype=torch.long)
    rows = [(x, y, labels, labels, "a__b"), (y, x, labels, labels, "b__a")]
    from utils.field_io import save_flow_npz_atomic

    for case in ("a__b", "b__a"):
        folder = tmp_path / "readouts/sealed/b1_s0/pairs" / case
        save_flow_npz_atomic(folder / "FF.npz", torch.zeros(1, 3, *shape))
        (folder / "readout.json").write_text("{}")
    args = SimpleNamespace(out=tmp_path, source=tmp_path / "unused", device=torch.device("cpu"))
    monkeypatch.setattr(intervention, "data", lambda _args, _split, **_kwargs: (iter(rows), metric_profile_for("IXI")))
    monkeypatch.setattr(intervention, "load_base", lambda *_: object())
    monkeypatch.setattr(intervention, "base_flow", lambda *_: torch.zeros(1, 3, *shape))
    intervention.evaluate_model(args, "l3_50_target_s0", checkpoint, "sealed", exact=True, controls=True)
    report = intervention.read(tmp_path / "readouts/sealed/l3_50_target_s0/pairs/a__b/readout.json")
    assert report["complete"] and report["certificate"]["global_homeomorphism"]
    assert report["rows"]["FF"]["delta_b1_rms_voxels"] == 0
    assert report["controls"]["amplitude_certified"] and report["controls"]["gaussian_certified"]
    assert intervention.readout(args, "sealed", "l3_50_target_s0", "REP")["unordered_pairs"] == 1
    original = intervention.certified_scores

    def refused(flow, case, seconds, xs, ys, directory, profile, dataset):
        values, proof = original(flow, case, seconds, xs, ys, directory, profile, dataset)
        if case == "a__b":
            proof = dict(proof, global_homeomorphism=False, errors=["injected test refusal"])
        return values, proof

    monkeypatch.setattr(intervention, "certified_scores", refused)
    with pytest.raises(RuntimeError, match="all raw Dice retained"):
        intervention.evaluate_model(args, "l3_100_target_s0", checkpoint, "sealed", exact=True)
    for case in ("a__b", "b__a"):
        saved = intervention.read(tmp_path / "readouts/sealed/l3_100_target_s0/pairs" / case / "readout.json")
        assert "FF" in saved["rows"]
    assert (tmp_path / "readouts/sealed/l3_100_target_s0/pairs/b__a/readout.json").is_file()
    assert intervention.readout(args, "sealed", "l3_100_target_s0", "FF")["directions"] == 2
    with pytest.raises(ValueError, match="Incomplete"):
        intervention.readout(args, "sealed", "l3_100_target_s0", "REP")


def test_failed_final_model_does_not_hold_other_frozen_models(tmp_path, monkeypatch):
    (tmp_path / "edits.json").write_text('{"rows": []}')
    (tmp_path / "splits.json").write_text("{}")
    (tmp_path / "selection.json").write_text(
        json.dumps(
            {
                "targets": [],
                "random_controls": [],
                "edits_sha256": sha256_file(tmp_path / "edits.json"),
                "splits_sha256": sha256_file(tmp_path / "splits.json"),
            }
        )
    )
    weights = {f"{arm}_s{seed}": "unused" for seed in (0, 1) for arm in ("b0", "b1")}
    monkeypatch.setattr(intervention, "verify", lambda _: {"weights": weights})
    calls = []

    def evaluate(_args, name, _path, _split, **_kwargs):
        calls.append(name)
        if name == "b0_s0":
            raise RuntimeError("Synthetic certificate failure")

    monkeypatch.setattr(intervention, "evaluate_model", evaluate)
    args = SimpleNamespace(out=tmp_path, supervised_checkpoint=None, split="sealed", lane=0, lanes=1)
    with pytest.raises(RuntimeError, match="other candidate"):
        intervention.final(args)
    assert calls == list(weights)
    assert len(intervention.read(tmp_path / "final_sealed_0.json")["failures"]) == 1


def test_training_entrypoint_accepts_only_explicit_paired_label_permissions(manifest_file, tmp_path, monkeypatch):
    path = tmp_path / "labels.txt"
    path.write_text("Train_0\nTrain_1\n")
    args = parse_args(
        [
            "--ds",
            "IXI",
            "--ixi_pairs",
            str(manifest_file),
            "--image_channels",
            "2",
            "--precision",
            "fp32",
            "--w_dice",
            "1",
            "--ixi_label_ids",
            str(path),
            "--num_workers",
            "0",
        ]
    )
    monkeypatch.setattr(trainer, "setup_device", lambda **_: torch.device("cpu"))
    monkeypatch.setattr(trainer, "Runner", lambda args, device: object())
    seen = []

    def execute(args, runner, build_loaders):
        loader, _ = build_loaders(args)
        batch = next(iter(loader))
        seen.append(len(batch))

    monkeypatch.setattr(trainer, "run_train", execute)
    trainer.main(args)
    assert seen == [7]
    args.ixi_label_ids = ""
    with pytest.raises(ValueError, match="requires both"):
        trainer.main(args)
    args.ixi_label_ids, args.ixi_pairs = str(path), ""
    with pytest.raises(ValueError, match="requires --ixi_pairs"):
        trainer.main(args)


def test_real_head_compilation_report_includes_padding_and_precision_scope():
    head = torch.nn.Conv3d(4, 3, 3, padding=1)
    result = projection_report(head, torch.randn(1, 4, 8, 9, 10), torch.randn(4))
    assert len(result["rows"]) == 2
    assert max(row["fp32_max_error"] for row in result["rows"]) < 1e-5
    assert "BF16 is not bitwise" in result["scope"]


def test_selection_carries_all_matched_random_directions_into_the_sealed_readout(tmp_path, monkeypatch):
    names = [
        f"{head}_{strength}_{kind}"
        for head in ("l2", "l3")
        for strength in (50, 100)
        for kind in ("target", "random0", "random1", "random2")
    ]
    (tmp_path / "edits.json").write_text(json.dumps({"rows": [{"name": name} for name in names]}))
    (tmp_path / "splits.json").write_text("{}")
    monkeypatch.setattr(intervention, "verify", lambda _: {})

    def values(_args, _split, name, _mode):
        first = name.startswith("b0_")
        return {"means": {"dice_mean": 0.78 if first else 0.79, "cortex": 0.7, "deep_gray": 0.8 if first else 0.82}}

    monkeypatch.setattr(intervention, "readout", values)
    intervention.select(SimpleNamespace(out=tmp_path))
    chosen = intervention.read(tmp_path / "selection.json")
    assert len(chosen["targets"]) == 2 and len(chosen["random_controls"]) == 6
    for name in chosen["targets"]:
        assert all(name.removesuffix("target") + f"random{i}" in chosen["random_controls"] for i in range(3))


def test_supervised_calibration_uses_real_dice_gradients_without_updating_weights(manifest_file, tmp_path, monkeypatch):
    manifest = read_ixi_manifest(manifest_file)
    ids = [row["id"] for row in manifest["training"]]
    pairs = intervention.reciprocal_pairs(ids) * 2
    (tmp_path / "splits.json").write_text(json.dumps({"subjects": {"discovery": ids}, "pairs": {"discovery": pairs}}))
    parent = tmp_path / "parent.pth"
    parent.write_bytes(b"parent")

    class CalibrationRunner:
        def __init__(self, args, device):
            self.args = args
            self.model = torch.nn.Module()
            self.model.register_parameter("p", torch.nn.Parameter(torch.tensor(0.1)))
            self.optimizer = torch.optim.Adam(self.model.parameters())
            self.reg_bilin = RegisterModel(manifest["shape"])

        def _pair_loss(self, x, y, xs, ys, epoch, t2):
            assert self.args.w_dice == 0 and t2 is not None and xs is ys is None
            xy = self.model.p.expand(1, 3, *manifest["shape"]).clone()
            yx = -xy
            return xy.square().mean() + yx.square().mean(), {}, xy, yx

    configured = SimpleNamespace(w_dice=1)
    reference = CalibrationRunner(configured, torch.device("cpu"))
    digest = restored_state_digest(reference.model, reference.optimizer, torch.device("cpu"))["model"]
    monkeypatch.setattr(supervision, "Runner", CalibrationRunner)
    monkeypatch.setattr(supervision, "setup_device", lambda **_: torch.device("cpu"))
    monkeypatch.setattr(supervision, "verify", lambda _: {"initial": {"b1_s0": digest}})
    args = SimpleNamespace(
        out=tmp_path, checkpoint=parent, manifest=manifest_file, exp="unit", device=torch.device("cpu")
    )
    supervision.calibrate(args)
    result = intervention.read(tmp_path / "supervision/calibration.json")
    assert len(result["ratios"]) == 8 and result["coefficient"] > 0
    assert result["initial_model_sha256"] == digest


def test_supervised_early_snapshots_keep_their_own_epoch_and_weights(tmp_path, monkeypatch):
    runner = supervision.SupervisionRunner.__new__(supervision.SupervisionRunner)
    runner.model = torch.nn.Conv3d(1, 3, 1)
    runner.model.input_representation = "raw"
    runner.args = SimpleNamespace(max_epoch=10, precision="bf16", ncc_mode="stable")
    runner.observation_manifest = runner.discovery_manifest = {}
    runner.observation_out = tmp_path

    def observe(_runner, _manifest, directory):
        directory.mkdir(parents=True)

    monkeypatch.setattr(supervision, "observe", observe)
    runner.on_val_end(0, 0, 0)
    record = intervention.read(tmp_path / "epoch1/checkpoint.json")
    assert record["epoch"] == 1 and record["sha256"] == sha256_file(tmp_path / "weights/epoch1.pth")
    saved = torch.load(record["checkpoint"], weights_only=False)
    torch.testing.assert_close(saved["state_dict"]["weight"], runner.model.weight)
    runner.on_val_end(9, 0, 0)
    assert not (tmp_path / "weights/epoch10.pth").exists()


def test_case_shards_partition_before_loading_without_changing_old_namespace_semantics(manifest_file):
    args = SimpleNamespace(dataset="IXI", manifest=manifest_file, limit=0, lane=1, lanes=2)
    original, _, _, count = cases(args)
    assert count == 2
    names = [row[-1] for row in original]
    shards = []
    for lane in (0, 1):
        values, _, _, local_count = cases(args, lane=lane, lanes=2)
        assert local_count == 1
        shards.extend(row[-1] for row in values)
    assert sorted(shards) == sorted(names) and len(set(shards)) == 2


def test_oracle_repair_preserves_a_valid_identity_boundary_field_without_double_taper(tmp_path):
    shape = (12, 13, 14)
    flow = torch.zeros(1, 3, *shape)
    flow[:, 0, 1:-1, 1:-1, 1:-1] = 0.2
    result, detail = intervention.boundary_repair(flow)
    torch.testing.assert_close(result, flow, rtol=0, atol=0)
    assert detail["tri_proj_iters"] == 0
    tapered, _ = intervention.repair_to_identity_boundary(flow)
    assert not torch.equal(tapered, flow)
    labels = torch.ones(1, 1, *shape, dtype=torch.long)
    _, proof = intervention.certified_scores(
        flow,
        "a__b",
        0,
        labels,
        labels,
        tmp_path,
        metric_profile_for("IXI"),
        "IXI",
        repair_fn=intervention.boundary_repair,
    )
    assert proof["global_homeomorphism"] and "provided_boundary" in proof["repair_policy"]
    stored = intervention.load_flow_npz(tmp_path / "REP.npz")
    torch.testing.assert_close(stored, flow, rtol=0, atol=0)


def test_cortex_error_counts_keep_left_and_right_separate():
    a = torch.tensor([2, 22]).reshape(1, 1, 1, 1, 2)
    counts = error_counts(a, a.flip(-1))
    assert counts["cortex_tp"] == 2 and counts["cortex_fp"] == 0
    assert counts["cortex_2_tp"] == 0 and counts["cortex_2_fp"] == counts["cortex_2_fn"] == 1
    assert counts["cortex_22_tp"] == 0


def test_cache_only_recovery_cannot_silently_use_another_checkpoint(tmp_path, monkeypatch):
    labels = torch.ones(1, 4, 4, 4, dtype=torch.long)
    item = (labels.float(), labels.float(), labels, labels, "subject_557__subject_85")
    monkeypatch.setattr(contrast_audit, "cases", lambda _: ([item], None, None, 1))

    def forbidden(*_args):
        raise AssertionError("A cache-only recovery must not run another model")

    monkeypatch.setattr(contrast_audit, "load_base", forbidden)
    args = SimpleNamespace(
        manifest=tmp_path / "manifest",
        source=tmp_path / "source",
        out=tmp_path / "out",
        base_checkpoint=None,
        device=torch.device("cpu"),
    )
    with pytest.raises(FileNotFoundError, match="Required saved"):
        contrast_audit.recovery(args)


def test_frozen_calibration_is_reused_without_new_gradient_fitting(tmp_path, monkeypatch):
    (tmp_path / "supervision").mkdir()
    (tmp_path / "splits.json").write_text("{}")
    checkpoint = tmp_path / "parent.pth"
    checkpoint.write_bytes(b"parent")
    saved = {
        "coefficient": 2,
        "splits_sha256": sha256_file(tmp_path / "splits.json"),
        "parent_sha256": sha256_file(checkpoint),
        "initial_model_sha256": "initial",
    }
    (tmp_path / "supervision/calibration.json").write_text(json.dumps(saved))
    monkeypatch.setattr(supervision, "verify", lambda _: {"initial": {"b1_s0": "initial"}})

    def forbidden(*_args):
        raise AssertionError("Frozen calibration must not instantiate another model")

    monkeypatch.setattr(supervision, "Runner", forbidden)
    supervision.calibrate(SimpleNamespace(out=tmp_path, checkpoint=checkpoint))
    assert intervention.read(tmp_path / "supervision/calibration.json") == saved


def test_partial_oracle_cohort_cannot_pass_the_full_cohort_criterion(tmp_path, monkeypatch):
    monkeypatch.setattr(intervention, "load_saved_scores", lambda _: ({}, {}))
    (tmp_path / "selection.json").write_text('{"targets": [], "random_controls": []}')
    (tmp_path / "readouts").mkdir()
    (tmp_path / "supervision").mkdir()
    (tmp_path / "supervision/report.json").write_text("{}")
    monkeypatch.setattr(intervention, "case_names", lambda _args, _split: ["a__b", "b__a"])
    monkeypatch.setattr(intervention, "readout", lambda *_: {"directions": 2, "means": {}})
    for family in ("stable0", "stable1", "scratch0"):
        folder = tmp_path / "fields" / family / "a__b"
        folder.mkdir(parents=True)
        rows = []
        for name in ("b0", "b1", "half", "fixed_only", "two_sided"):
            deep = 0.8 if name == "b0" else 0.82
            rows.append(
                {
                    "variant": name,
                    "mode": "REP",
                    "case_id": "a__b",
                    "dice_mean": 0.8,
                    **{f"dice_lbl_{i}": 0.7 if i in (2, 22) else deep for i in (2, 22, 7, 8, 9, 10, 27, 28, 29, 30)},
                }
            )
        (folder / "readout.json").write_text(json.dumps({"complete": True, "rows": rows}))
    with pytest.raises(RuntimeError, match="incomplete"):
        intervention.summary(SimpleNamespace(out=tmp_path))
    result = intervention.read(tmp_path / "report.json")["field_oracle"]["stable0"]["criteria"]["two_sided"]
    assert not result["passed"] and not result["complete_cohort"]


def test_production_recovery_stops_extra_diagnostics_without_changing_selected_bytes(tmp_path):
    from experiments.repair_diagnostics import repair_one

    shape = (12, 13, 14)
    field = torch.zeros(1, 3, *shape)
    labels = [torch.ones(1, 1, *shape, dtype=torch.long)] * 2
    case = {"dataset": "IXI", "case_id": "a__b", "old_tri_iters": 80}
    complete = repair_one(field, field, case, labels, tmp_path / "complete")
    compact = repair_one(field, field, case | {"diagnostics": False}, labels, tmp_path / "compact")
    assert complete["selected"] == compact["selected"] and compact["global_REP"]["global_homeomorphism"]
    assert len(complete["candidates"]) > len(compact["candidates"]) == 1
    first = intervention.load_flow_npz(tmp_path / "complete" / complete["selected"] / "REP.npz")
    second = intervention.load_flow_npz(tmp_path / "compact" / compact["selected"] / "REP.npz")
    torch.testing.assert_close(first, second, rtol=0, atol=0)


def test_review_cached_oracle_requires_saved_arrays(tmp_path, monkeypatch):
    folder = tmp_path / "fields/stable0/a__b"
    folder.mkdir(parents=True)
    (folder / "readout.json").write_text('{"complete": true, "rows": []}')
    monkeypatch.setattr(intervention, "verify", lambda _: {})
    monkeypatch.setattr(intervention, "data", lambda *a, **k: ([(None, None, None, None, "a__b")], None))
    with pytest.raises(FileNotFoundError):
        intervention.fields(SimpleNamespace(out=tmp_path, lane=0, lanes=1))


def test_review_measure_reuses_existing_geometry_row(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Existing score must not be calculated again")

    monkeypatch.setattr(intervention, "score", forbidden)
    flow = torch.zeros(1, 3, 12, 13, 14)
    image = torch.rand(1, 1, 12, 13, 14)
    labels = torch.ones_like(image).long()
    original = {"case_id": "a__b", "dice_mean": 0.75, "time_sec": 12}
    result = intervention.measure(flow, "a__b", image, image, labels, labels, metric_profile_for("IXI"), row=original)
    assert result["dice_mean"] == 0.75 and result["time_sec"] == 12 and "ncc" in result
    assert "ncc" not in original
