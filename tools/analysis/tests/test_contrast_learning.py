from __future__ import annotations

import json
import pickle
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import SimpleITK as sitk  # noqa: N813
import torch
from scipy import ndimage

from datasets.IXI import IXIPairedDataset, IXIPairSampler, read_ixi_manifest
from experiments import contrast_learning
from experiments.contrast_learning import derangement, options_for, prepare, resample_t1_control
from tools.analysis.run_artifacts import atomic_write_json, sha256_file
from utils import pkload


def image(values, *, spacing=(1.0, 1.0, 1.0), origin=(0.0, 0.0, 0.0)):
    result = sitk.GetImageFromArray(values.astype(np.float32))
    result.SetSpacing(spacing)
    result.SetOrigin(origin)
    return result


def test_shuffle_is_a_frozen_bijection_without_correct_subjects():
    first = derangement(397)
    assert sorted(first) == list(range(397))
    assert all(i != j for i, j in enumerate(first))
    assert first == derangement(397) and first != derangement(397, seed=1)
    with pytest.raises(ValueError, match="at least two"):
        derangement(1)


def test_sampling_control_never_uses_native_t2_intensities():
    t1 = np.indices((9, 10, 11)).sum(0).astype(np.float32) + 1
    geometry = dict(direction=np.eye(3).ravel().tolist(), spacing=[1.0] * 3, origin=[0.0] * 3)
    identity = sitk.AffineTransform(3)
    first, report = resample_t1_control(t1, image(t1), geometry, identity, sitk.Euler3DTransform())
    second, _ = resample_t1_control(t1, image(np.zeros_like(t1)), geometry, identity, sitk.Euler3DTransform())
    lo, hi = np.percentile(t1, [1, 99])
    np.testing.assert_array_equal(first, second)
    np.testing.assert_allclose(first, np.clip((t1 - lo) / (hi - lo), 0, 1), atol=1e-7)
    assert report["coverage"] == 1


def test_normalisation_only_matches_the_roundtrip_coverage_and_never_resamples_t1():
    shape = (9, 10, 11)
    t1 = np.indices(shape).sum(0).astype(np.float32) + 1
    native = image(np.zeros((8, 9, 10), dtype=np.float32))
    geometry = dict(direction=np.eye(3).ravel().tolist(), spacing=[1.0] * 3, origin=[0.0] * 3)
    values, report = resample_t1_control(
        t1, native, geometry, sitk.AffineTransform(3), sitk.Euler3DTransform(), normalisation_only=True
    )
    lo, hi = np.percentile(t1[:8, :9, :10], [1, 99])
    np.testing.assert_allclose(values, np.clip((t1 - lo) / (hi - lo), 0, 1), atol=1e-7)
    assert report["percentiles"] == [lo, hi]
    assert report["coverage"] == pytest.approx(8 * 9 * 10 / (9 * 10 * 11))
    # The control retains original T1 values beyond native coverage; the roundtrip cannot.
    roundtrip, other = resample_t1_control(t1, native, geometry, sitk.AffineTransform(3), sitk.Euler3DTransform())
    assert report == other and values[-1, -1, -1] == 1 and roundtrip[-1, -1, -1] == 0


def test_noncommuting_physical_roundtrip_matches_independent_scipy_reference():
    shape = (13, 14, 15)
    z, y, x = np.indices(shape)
    t1 = np.exp(-((x - 7) ** 2 / 12 + (y - 6) ** 2 / 8 + (z - 5) ** 2 / 16)).astype(np.float32)
    geometry = dict(direction=np.eye(3).ravel().tolist(), spacing=[1.1, 0.9, 1.2], origin=[-1.3, 0.7, -2.1])
    native = image(np.ones((18, 19, 20)), spacing=(0.95, 1.05, 1.15), origin=(-3.2, -1.4, -4.6))
    affine = sitk.AffineTransform(3)
    a = np.array([[1.02, 0.025, 0.0], [0.0, 0.99, -0.015], [0.01, 0.0, 1.01]])
    affine.SetMatrix(tuple(a.ravel()))
    affine.SetTranslation((0.31, -0.23, 0.17))
    rigid = sitk.Euler3DTransform()
    rigid.SetRotation(0.012, -0.018, 0.026)
    rigid.SetTranslation((-0.37, 0.29, 0.21))
    actual, _ = resample_t1_control(t1, native, geometry, affine, rigid)
    b = np.array(rigid.GetMatrix()).reshape(3, 3)
    matrix, shift = b @ a, b @ np.array(affine.GetTranslation()) + rigid.GetTranslation()

    def physical(shape, spacing, origin):
        return np.moveaxis(np.indices(shape)[::-1], 0, -1) * spacing + origin

    def sample(values, indices):
        coordinates = np.moveaxis(indices[..., ::-1], -1, 0)
        inside = np.all((indices >= -0.5) & (indices < np.array(values.shape[::-1]) - 0.5), axis=-1)
        return np.where(inside, ndimage.map_coordinates(values, coordinates, order=1, mode="nearest"), 0)

    world = physical(tuple(reversed(native.GetSize())), native.GetSpacing(), native.GetOrigin())
    processed_world = (world - shift) @ np.linalg.inv(matrix).T
    processed_index = (processed_world - geometry["origin"]) / geometry["spacing"]
    sampled_native = sample(t1, processed_index).astype(np.float32)
    world = physical(shape, geometry["spacing"], geometry["origin"])
    native_index = (world @ matrix.T + shift - native.GetOrigin()) / native.GetSpacing()
    restored = sample(sampled_native, native_index)
    covered = np.all((native_index >= -0.5) & (native_index < np.array(native.GetSize()) - 0.5), axis=-1)
    lo, hi = np.percentile(restored[covered], [1, 99])
    expected = np.clip((restored - lo) / (hi - lo), 0, 1)
    np.testing.assert_allclose(actual, expected, atol=2e-6)


@pytest.fixture
def cohort(tmp_path):
    shape = (8, 9, 10)
    geometry = dict(direction=np.eye(3).ravel().tolist(), spacing=[1.0] * 3, origin=[0.0] * 3)
    raw = tmp_path / "raw"
    raw.mkdir()
    groups = {}
    for split, count in (("training", 4), ("validation", 2)):
        rows = []
        for index in range(count):
            name = f"{split}_{index}"
            folder = tmp_path / name
            folder.mkdir()
            t1 = (np.indices(shape).sum(0) + index + 1).astype(np.float32)
            path = folder / "t1.pkl"
            with path.open("wb") as stream:
                pickle.dump((t1, np.ones(shape, dtype=np.int16) * 2), stream)
            t2 = folder / "t2.npy"
            np.save(t2, np.flip(t1, 0).copy())
            row = dict(
                id=name,
                ixi_id=f"IXI_{name}",
                split="Train" if split == "training" else "Val",
                t1=str(path),
                t2=str(t2),
                t1_sha256=sha256_file(path),
                t2_sha256=sha256_file(t2),
            )
            rows.append(row)
            native = raw / f"{name}-T2.nii.gz"
            sitk.WriteImage(image(t1), str(native))
            for transform in ("processed_to_t1", "t1_to_t2"):
                sitk.WriteTransform(sitk.AffineTransform(3), str(folder / f"{transform}.tfm"))
            atomic_write_json(
                folder / "qc.json",
                dict(
                    id=name,
                    t1_sha256=row["t1_sha256"],
                    t2_sha256=row["t2_sha256"],
                    binding=dict(geometry=geometry, sources={str(native): sha256_file(native)}),
                ),
            )
        groups[split] = rows
    ids = [row["id"] for row in groups["validation"]]
    manifest = tmp_path / "manifest.json"
    atomic_write_json(
        manifest,
        dict(
            reviewed=True,
            shape=list(shape),
            **groups,
            expected_counts=dict(training=4, validation=2),
            validation_pairs=[ids, ids[::-1]],
        ),
    )
    parent = tmp_path / "parent.pth"
    parent.write_bytes(b"fixed parent")
    source = tmp_path / "source"
    atomic_write_json(
        source / "inputs.json", dict(manifest_sha256=sha256_file(manifest), parent_sha256=sha256_file(parent))
    )
    for arm in ("b0", "b1"):
        directory = source / f"{arm}_w1_s0"
        atomic_write_json(
            directory / "report.json",
            dict(arm=arm, seed=0, w_reg=1, updates=1197, manifest_sha256=sha256_file(manifest)),
        )
        (directory / "epoch3").mkdir()
        cases = ("__".join(ids), "__".join(ids[::-1]))
        (directory / "epoch3/per_case.csv").write_text(
            "case_id,dice_mean\n" + "".join(f"{case},0.7\n" for case in cases)
        )
        atomic_write_json(
            directory / "repair_summary.json",
            dict(certified=True, directions=2, fallbacks=0, rows=[dict(case_id=case, dice_mean=0.7) for case in cases]),
        )
    return SimpleNamespace(
        manifest=manifest, checkpoint=parent, source=source, out=tmp_path / "controls", raw_root=raw, threads=1
    )


def test_preparation_changes_only_training_auxiliary_targets_and_detects_tampering(cohort):
    original_bytes = cohort.manifest.read_bytes()
    source = read_ixi_manifest(cohort.manifest, verify_bytes=True)
    prepare(cohort)
    for kind in ("resampled_t1", "shuffled_t2"):
        path = cohort.out / "targets" / kind / "manifest.json"
        changed = read_ixi_manifest(path, verify_bytes=True)
        assert changed["validation"] == source["validation"]
        assert changed["validation_pairs"] == source["validation_pairs"]
        assert changed["source_manifest_sha256"] == source["fingerprint"]
        assert all(a["t1"] == b["t1"] for a, b in zip(changed["training"], source["training"], strict=True))
        if kind == "shuffled_t2":
            assert all(a["t2"] != b["t2"] for a, b in zip(changed["training"], source["training"], strict=True))
        dataset = IXIPairedDataset(changed, "b1")
        moving, _, auxiliary, _ = dataset[(0, 1)]
        np.testing.assert_array_equal(moving[0], moving[1])
        np.testing.assert_array_equal(moving[0], pkload(source["training"][0]["t1"])[0])
        np.testing.assert_array_equal(auxiliary[0], np.load(changed["training"][0]["t2"]))
        assert list(IXIPairSampler(4, 0, changed["fingerprint"], samples=399)) == list(
            IXIPairSampler(4, 0, source["fingerprint"], samples=399)
        )
    prepare(cohort)
    assert cohort.manifest.read_bytes() == original_bytes
    target = cohort.out / "targets/resampled_t1/Train/training_0/aux.npy"
    np.save(target, np.zeros(source["shape"], dtype=np.float32))
    with pytest.raises(ValueError, match="auxiliary target changed"):
        prepare(cohort)


def test_scratch_recipe_shares_start_and_controls_reuse_the_original_trainer(tmp_path):
    base = dict(
        manifest=tmp_path / "manifest.json",
        prepared=tmp_path / "controls",
        checkpoint=tmp_path / "parent.pth",
        exp="unit",
        precision="bf16",
        workers=0,
    )
    first, second = [options_for(SimpleNamespace(**base, kind=k)) for k in ("scratch_b0", "scratch_b1")]
    assert first.ixi_arm == "b0" and second.ixi_arm == "b1"
    assert vars(first) | {"ixi_arm": "b1"} == vars(second)
    assert first.init_ckpt == "" and first.finetune == 0 and first.lr == 1e-4
    assert first.max_epoch == first.schedule_max_epoch == 100 and first.max_train_iters == 399
    for kind in ("resampled_t1", "shuffled_t2"):
        control = options_for(SimpleNamespace(**base, kind=kind))
        assert control.ixi_arm == "b1" and control.finetune == 1 and control.lr == 2e-5
        assert control.w_reg == 1 and control.max_epoch == 3 and control.seed == 0
        assert control.ixi_pairs == str(base["prepared"] / "targets" / kind / "manifest.json")


def test_preflight_binds_real_legacy_report_schema_and_rejects_changed_endpoint_bytes(cohort):
    cohort.bm_source = cohort.out / "BM"
    for arm, field in (("b0", "b0_checkpoint"), ("b1", "base_checkpoint")):
        checkpoint = cohort.raw_root / f"{arm}.pth"
        checkpoint.write_bytes(arm.encode())
        setattr(cohort, field, checkpoint)
        atomic_write_json(
            cohort.bm_source / f"{arm}_s1/report.json",
            dict(
                arm=arm,
                epochs=10,
                seed=1,
                w_reg=1,
                manifest_sha256=sha256_file(cohort.manifest),
                checkpoint_sha256=sha256_file(checkpoint),
            ),
        )
    external = cohort.out / "external"
    external.mkdir(parents=True)
    subjects = []
    for index in range(40):
        file = external / f"S{index:02}.npz"
        file.write_bytes(f"verified external payload {index}".encode())
        subjects.append(dict(id=file.stem, file=file.name, sha256=sha256_file(file)))
    cohort.external_manifest = external / "manifest.json"
    pairs = [[f"S{i:02}", f"S{(i + offset) % 40:02}"] for i in range(40) for offset in (1, 2, 3)]
    atomic_write_json(
        cohort.external_manifest,
        dict(
            dataset="LPBA40",
            subjects=subjects,
            pairs=pairs,
            shape=[8, 9, 10],
            label_ids=list(range(1, 57)),
            tensor_axes="LIA",
        ),
    )
    contrast_learning.preflight(cohort)
    original = (cohort.out / "preflight.json").read_bytes()
    assert json.loads(original)["reference_controls"]["b0"]["REP_summary_sha256"] == sha256_file(
        cohort.source / "b0_w1_s0/repair_summary.json"
    )
    contrast_learning.preflight(cohort)
    assert (cohort.out / "preflight.json").read_bytes() == original
    cohort.base_checkpoint.write_bytes(b"different weights under same filename")
    with pytest.raises(ValueError, match="recorded SHA256"):
        contrast_learning.preflight(cohort)


def test_cost_describes_update_counts_without_a_gpu_hour_gate(tmp_path):
    (tmp_path / "stride.txt").write_text("1\n")
    for path, seconds in (
        ("ctcf_profile/report.json", 1.0),
        ("profile_student_1/profile.json", 2.0),
        ("profile_teacher/profile.json", 3.0),
    ):
        atomic_write_json(tmp_path / path, dict(update_seconds=seconds, passed=True))
    args = SimpleNamespace(out=tmp_path)
    contrast_learning.cost(args)
    saved = json.loads((tmp_path / "cost.json").read_text())
    expected = (2 * 39900 + 2 * 1197 + 2 * 4000 * 3 + 5 * 4000 * 2) / 3600 + 12
    assert saved["estimated_gpu_hours"] == pytest.approx(expected)
    assert saved["stride"] == 1 and "passed" not in saved and "budget_gpu_hours" not in saved
    assert saved["profiles"]["student"]["file_sha256"] == sha256_file(tmp_path / "profile_student_1/profile.json")


@pytest.mark.parametrize("mode,epochs", (("stable", (1, 3, 8, 9, 10)), ("legacy", (1, 3, 10))))
def test_stable_endpoint_has_terminal_regional_readouts_without_changing_legacy(tmp_path, monkeypatch, mode, epochs):
    runner = object.__new__(contrast_learning.ReadoutRunner)
    runner.args = SimpleNamespace(finetune=1, max_epoch=10, ncc_mode=mode)
    runner.observation_manifest, runner.observation_out = {}, tmp_path
    observed = []
    monkeypatch.setattr(
        contrast_learning,
        "observe",
        lambda _runner, _manifest, path: observed.append(int(path.name.removeprefix("epoch"))),
    )
    for epoch in range(10):
        runner.on_val_end(epoch, 0, 0)
    assert tuple(observed) == epochs


def test_report_reads_shared_evaluations_and_requires_identical_scratch_initializations(tmp_path):
    evaluations, source = tmp_path / "shared_scores", tmp_path / "legacy"
    for kind in contrast_learning.KINDS[:2]:
        atomic_write_json(tmp_path / "registration" / kind / "report.json", dict(initial_model_sha256="same"))
    cases = ("A__B", "B__A")
    for kind in contrast_learning.KINDS:
        value = 0.72 if kind == "scratch_b1" else 0.71
        for mode in ("FF", "REP"):
            directory = evaluations / kind / mode
            directory.mkdir(parents=True)
            (directory / "per_case.csv").write_text(
                "case_id,dice_mean\n" + "".join(f"{case},{value}\n" for case in cases)
            )
    for arm in ("b0", "b1"):
        directory = source / f"{arm}_w1_s0/epoch3"
        directory.mkdir(parents=True)
        (directory / "per_case.csv").write_text("case_id,dice_mean\n" + "".join(f"{case},0.70\n" for case in cases))
        atomic_write_json(
            directory.parent / "repair_summary.json",
            dict(certified=True, directions=2, fallbacks=0, rows=[dict(case_id=case, dice_mean=0.7) for case in cases]),
        )
    args = SimpleNamespace(out=tmp_path, source=source, evaluations_root=evaluations)
    contrast_learning.report(args)
    result = json.loads((tmp_path / "learning_summary.json").read_text())
    assert result["same_scratch_initialization"] and not result["missing"]
    assert result["comparisons"]["REP_scratch_b1-scratch_b0"]["mean"] == pytest.approx(0.01)
    atomic_write_json(tmp_path / "registration/scratch_b1/report.json", dict(initial_model_sha256="changed"))
    with pytest.raises(RuntimeError, match="incomplete"):
        contrast_learning.report(args)
    assert not json.loads((tmp_path / "learning_summary.json").read_text())["same_scratch_initialization"]


def test_historical_repair_rejects_duplicate_or_uncertified_primary_rows(cohort):
    directory = cohort.source / "b0_w1_s0"
    path = directory / "repair_summary.json"
    raw = contrast_learning.read_csv(directory / "epoch3/per_case.csv")
    saved = json.loads(path.read_text())
    assert contrast_learning.historical_repair(directory, raw) == saved["rows"]
    duplicated = dict(saved, rows=[saved["rows"][0]] * 2)
    for changed in (duplicated, dict(saved, certified=False), dict(saved, fallbacks=1)):
        atomic_write_json(path, changed)
        with pytest.raises(ValueError, match="incomplete, uncertified or uses fallback"):
            contrast_learning.frozen_references(cohort, sha256_file(cohort.manifest))


def test_resume_keeps_original_recipe_and_both_scratch_models_share_fresh_state(cohort, monkeypatch):
    monkeypatch.chdir(cohort.out.parent)
    cohort.kind, cohort.exp, cohort.precision, cohort.workers = "scratch_b0", "unit_b0", "bf16", 0
    cohort.prepared = None
    calls = []

    def cpu_device(**kwargs):
        torch.manual_seed(kwargs["seed"])
        return torch.device("cpu")

    class TinyRunner:
        def __init__(self, options, device, *_):
            self.model = torch.nn.Linear(2, 1)
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=options.lr)

    def complete(options, runner, **_):
        calls.append((options.resume, options.init_ckpt))
        path = Path("results") / options.exp / "ckpt/last.pth"
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(dict(epoch=options.max_epoch - 1, arguments=vars(options)), path)

    monkeypatch.setattr(contrast_learning, "setup_device", cpu_device)
    monkeypatch.setattr(contrast_learning, "configure_precision", lambda *_: None)
    monkeypatch.setattr(contrast_learning, "ReadoutRunner", TinyRunner)
    monkeypatch.setattr(contrast_learning, "run_train", complete)
    contrast_learning.train(cohort)
    recipe = (cohort.out / "recipe.json").read_bytes()
    first = json.loads(recipe)["initial_model_sha256"]
    contrast_learning.train(cohort)
    assert (cohort.out / "recipe.json").read_bytes() == recipe
    assert calls[0] == ("", "") and Path(calls[1][0]).parts[-3:] == ("unit_b0", "ckpt", "last.pth")
    assert calls[1][1] == ""
    cohort.kind, cohort.exp, cohort.out = "scratch_b1", "unit_b1", cohort.out / "other"
    contrast_learning.train(cohort)
    assert json.loads((cohort.out / "recipe.json").read_text())["initial_model_sha256"] == first
