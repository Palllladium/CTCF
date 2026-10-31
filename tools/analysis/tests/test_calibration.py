import json
import os
import pickle
import shutil
import time
import zipfile
from pathlib import Path

import numpy as np
import pytest
import torch

from datasets.OASIS import OASISBrainDataset, OASISBrainInferDataset
from experiments.core.data_loaders import oasis_subject_paths, oasis_validation_pairs
from experiments.population import calibrate, evaluate
from experiments.population.evaluate import BOUNDARY_ROUNDING_LIMIT, score_pair_field, spatial_holdout
from tools.analysis.check_lumir_training import (
    checkpoint_comparison,
    export_reproduction,
    restored_state_check,
    trajectory_spread,
)
from tools.analysis.run_artifacts import package_run
from tools.analysis.tests.test_lumir_runner import dispatch as dispatch
from utils import NumpyType


def test_subject_manifest_filters_both_training_sides_and_dev_pairs(tmp_path):
    image = np.arange(27, dtype=np.float32).reshape(3, 3, 3)
    for i in range(1, 5):
        (tmp_path / f"p_{i:04d}.pkl").write_bytes(pickle.dumps((image + i, np.ones_like(image, dtype=np.int16))))
    split = {
        "training": [{"subject_id": f"OASIS_ALL_P{i:04d}"} for i in (1, 2)],
        "development": [{"subject_id": f"OASIS_ALL_P{i:04d}"} for i in (3, 4)],
    }
    path = tmp_path / "split.json"
    path.write_text(json.dumps(split))
    (tmp_path / "pair_manifest.json").write_text(
        json.dumps(
            {
                "cases": [
                    {"moving_subject_id": "OASIS_ALL_P0003", "fixed_subject_id": "OASIS_ALL_P0004"},
                    {"moving_subject_id": "OASIS_ALL_P0004", "fixed_subject_id": "OASIS_ALL_P0003"},
                ]
            }
        )
    )
    training, development = oasis_subject_paths(tmp_path, path)
    dataset = OASISBrainDataset(training, NumpyType((np.float32, np.int16)), return_ids=True)
    assert {value for i in range(len(dataset)) for value in dataset[i][-2:]} == {"p_0001", "p_0002"}
    dev = OASISBrainInferDataset([(development[0], development[1])], NumpyType((np.float32, np.int16)))
    x, y, _, _ = dev[0]
    torch.testing.assert_close(y - x, torch.ones_like(x))
    split["development"][0] = split["training"][0]
    path.write_text(json.dumps(split))
    with pytest.raises(ValueError, match="disjoint"):
        oasis_subject_paths(tmp_path, path)


def test_expanded_cohorts_share_heldout_and_freeze_input_bytes(tmp_path, monkeypatch):
    image = np.linspace(0, 1, 9**3, dtype=np.float32).reshape(9, 9, 9)
    root = tmp_path / "subjects"
    root.mkdir()
    for i in range(394):
        (root / f"p_{i:04d}.pkl").write_bytes(pickle.dumps((image + i / 10000, np.zeros_like(image, dtype=np.int16))))
    split = {
        "training": [{"subject_id": f"OASIS_ALL_P{i:04d}"} for i in range(294)],
        "development": [{"subject_id": f"OASIS_ALL_P{i:04d}"} for i in range(294, 394)],
    }
    path = tmp_path / "split.json"
    path.write_text(json.dumps(split))
    ids = [row["subject_id"] for row in split["development"]]
    (tmp_path / "pair_manifest.json").write_text(
        json.dumps(
            {
                "cases": [
                    {"moving_subject_id": a, "fixed_subject_id": b}
                    for i in range(0, len(ids), 2)
                    for a, b in ((ids[i], ids[i + 1]), (ids[i + 1], ids[i]))
                ]
            }
        )
    )
    monkeypatch.setattr(calibrate, "SPLIT", path)
    run = tmp_path / "run"
    config = calibrate.prepare(run, root)
    assert len(config["small"]) == 64 and len(config["large"]) == 278 and len(config["heldout"]) == 16
    assert set(config["small"]).issubset(config["large"])
    assert not set(config["large"]) & set(config["heldout"])
    assert not set(config["training"]) & set(config["development"])
    assert config["reference"] in config["small"]
    assert calibrate.prepare(run, root) == config
    heldout = calibrate.Path(config["heldout"][0])
    heldout.write_bytes(b"changed payload")
    with pytest.raises(ValueError, match="bytes changed"):
        calibrate.prepare(run, root)


def test_spatial_holdout_has_a_halo_and_targets_do_not_enter_visible_context():
    visible, target = spatial_holdout((40, 40, 40), "cpu", block_size=8, gap=2)
    assert visible.any() and target.any()
    assert not (visible & target).any()
    expanded = torch.nn.functional.max_pool3d(target.float(), 5, stride=1, padding=2).bool()
    torch.testing.assert_close(visible, ~expanded)


def test_real_dev_pairs_match_frozen_manifest_including_direction():
    root = Path(__file__).resolve().parents[3] / "datasets/splits/oasis_subject_disjoint_v2"
    split = json.loads((root / "split_manifest.json").read_text())
    paths = [f"All/p_{row['subject_id'].rsplit('_P', 1)[1]}.pkl" for row in split["development"]]
    records = json.loads((root / "pair_manifest.json").read_text())["cases"]
    actual = oasis_validation_pairs(paths, root / "pair_manifest.json")
    assert actual == [
        (
            f"All/p_{r['moving_subject_id'].rsplit('_P', 1)[1]}.pkl",
            f"All/p_{r['fixed_subject_id'].rsplit('_P', 1)[1]}.pkl",
        )
        for r in records
    ]
    assert len(actual) == 100
    with pytest.raises(ValueError, match="Dev manifest"):
        oasis_validation_pairs(paths[2:], root / "pair_manifest.json")


def test_trajectory_spread_is_descriptive_and_restoration_is_checked_exactly():
    zero = np.zeros((2, 3, 4, 3), dtype=np.float32)
    spread = trajectory_spread(zero, zero + 0.1, zero + 2.0)
    assert spread["status"] == "DESCRIPTIVE" and spread["errors_voxels"]["resumed"]["maximum"] == 2
    digest = {"model": "a", "optimizer": "b", "rng": ["c", "d"]}
    log = f"noise\n>>> restored_state {json.dumps(digest)}\nmore\n"
    assert restored_state_check(log, digest) == digest
    with pytest.raises(ValueError, match="differs"):
        restored_state_check(log, digest | {"optimizer": "other"})
    with pytest.raises(ValueError, match="did not report"):
        restored_state_check("noise", digest)


def test_export_reproduction_compares_a_fresh_forward_pass_with_the_written_field(tmp_path, monkeypatch):
    import types

    import tools.analysis.check_lumir_training as check

    class Model(torch.nn.Module):
        def forward(self, x, _y, **_kwargs):
            return x, torch.full((1, 3, 2, 3, 4), 0.25)

    monkeypatch.setattr(check, "volume_window_size", lambda shape: shape)
    monkeypatch.setattr(check, "get_model_adapter", lambda _: types.SimpleNamespace(build=lambda **_: Model()))
    monkeypatch.setattr(check, "load_checkpoint_state", lambda *a, **k: {"strict": True})
    monkeypatch.setattr(check, "_load_pair", lambda *_: (torch.zeros(1, 1, 2, 3, 4), torch.zeros(1, 1, 2, 3, 4)))
    manifest = {"pairs_json": "pairs.json", "files": [{"fixed": "f", "moving": "m"}]}
    field = check.evaluator_displacement(torch.full((1, 3, 2, 3, 4), 0.25))
    device = torch.device("cpu")
    assert export_reproduction(tmp_path / "last.pth", manifest, field, device) == 0
    assert export_reproduction(tmp_path / "last.pth", manifest, field + 0.5, device) == pytest.approx(0.5)


def test_parent_reference_scores_the_same_pairs_in_sampler_map_coordinates(tmp_path, monkeypatch):
    import types

    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:

        class Parent(torch.nn.Module):
            def forward(self, x, y, **_kwargs):
                return x, torch.zeros_like(x).expand(-1, 3, -1, -1, -1)

        monkeypatch.setattr(
            calibrate, "get_model_adapter", lambda _: types.SimpleNamespace(build=lambda **kwargs: Parent())
        )
        monkeypatch.setattr(calibrate, "load_checkpoint_state", lambda *a, **k: {"strict": True})
        heldout = []
        for i in range(2):
            path = tmp_path / f"subject_{i}.pkl"
            image = np.linspace(0, 1, 32**3, dtype=np.float32).reshape(32, 32, 32)
            path.write_bytes(pickle.dumps((image, (image > 0.5).astype(np.int16))))
            heldout.append(str(path))
        checkpoint = tmp_path / "parent.pth"
        checkpoint.write_bytes(b"synthetic checkpoint")
        run = tmp_path / "run"
        calibrate.parent_reference(run, {"heldout": heldout}, checkpoint, "cpu")
        report = json.loads((run / "parent/report.json").read_text())
        assert report["pairs"] == 2 and report["heldout"] == heldout
        assert report["precision"] == "tf32" and report["certification"].startswith("NOT_RUN")
        assert (run / "parent/REP/summary.json").is_file()
    finally:
        torch.set_num_threads(previous)


def test_a_calibration_runs_training_map_milestones_repair_and_saved_byte_certificate(tmp_path, monkeypatch):
    from tools.analysis.tests.population.test_runtime import _args, _inputs

    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        paths, images = _inputs(tmp_path)
        heldout = []
        for index, image in enumerate(images):
            path = tmp_path / f"heldout_{index}.pkl"
            path.write_bytes(pickle.dumps((image, (image > 0.5).astype(np.int16))))
            heldout.append(str(path))
        template = tmp_path / "initial.npy"
        np.save(template, np.stack(images).mean(0)[None, None])
        options = vars(_args(paths, tmp_path / "unused", 1))
        config = dict(
            training_options=options,
            small=list(map(str, paths)),
            heldout=heldout,
            template=str(template),
            input_sha256={},
            map_steps=1,
            map_lr=0.01,
            map_readouts=[0, 1],
        )
        monkeypatch.setattr(calibrate, "ARMS", {"unit": ("A_lin", 64, 2, 0, 1)})
        monkeypatch.setattr(calibrate, "EXTENDED_EPOCHS", 2)
        monkeypatch.setattr(calibrate, "EXTENDED_SNAPSHOT_EPOCHS", [2])
        run = tmp_path / "run"
        calibrate.fit_arm(run, config, "unit", "cpu")
        for path in heldout:
            fitted = run / "labels" / Path(path).stem
            fitted.mkdir(parents=True)
            np.save(fitted / "velocity.npy", np.zeros((1, 3, 9, 9, 9), dtype=np.float32))
            (fitted / "report.json").write_text(json.dumps({"directory": str(fitted)}))
        options = dict(cohort="heldout")
        check = run / "arms/unit/last/pair_check.json"

        def broken(*_args, **_kwargs):
            raise RuntimeError("exploded")

        with monkeypatch.context() as patched:
            patched.setattr(evaluate, "repair", broken)
            with pytest.raises(RuntimeError, match="raw scores are written"):
                calibrate.assess_arm(run, config, "unit", "last", "cpu", limit_pairs=1, **options)
            assert json.loads(check.read_text())["status"] == "RAW_ONLY"
        with monkeypatch.context() as patched:
            patched.setattr(calibrate, "score_pair_field", broken)
            with pytest.raises(RuntimeError, match="exploded"):
                calibrate.assess_arm(run, config, "unit", "last", "cpu", limit_pairs=1, **options)
            assert not check.exists()
        calibrate.assess_arm(run, config, "unit", "last", "cpu", limit_pairs=1, **options)
        assert json.loads(check.read_text())["status"] == "PASS"
        assert not (run / "arms/unit/last/report.json").exists()
        calibrate.assess_arm(run, config, "unit", "last", "cpu", **options)
        case = next((run / "arms/unit/last/pairs").iterdir())
        report = json.loads((case / "report.json").read_text())
        assert report["certificate_REP"]["status"] == "CERTIFIED"
        assert report["global_REP"]["global_homeomorphism"] is True
        assert set(json.loads((case / "map_readouts.json").read_text())["steps"]) == {"0", "1"}
        assert set(report["rows"]) == {"FF", "REP"}
        assert not list((run / "arms/unit/last/subjects").glob("*/step_*.npy"))
        assert list((run / "arms/unit/last/subjects").glob("*/velocity.npy"))
        assert (run / "arms/unit/last/atlas_preview.png").stat().st_size > 0
        terms = json.loads((run / "arms/unit/last/diagnostics.json").read_text())
        assert terms["subjects"][Path(heldout[0]).stem]["encoder"][0]["linear_rms"] >= 0
        distance = terms["label_map_distance_voxels"][case.name]
        assert distance["foreground_label_map_norm"] == 0 and distance["foreground_distance"] >= 0
        checkpoint = calibrate.arm_point(run, "unit", "last")[0]
        evaluated = checkpoint.read_bytes()
        calibrate.extend_arm(run, config, "unit", "cpu")
        assert checkpoint.read_bytes() == evaluated
        assert torch.load(run / "arms/unit/ext/last.pth", weights_only=False)["completed_epochs"] == 2
        assert (run / "arms/unit/ext/epoch_2.pth").is_file()
        calibrate.assess_arm(run, config, "unit", "ext/last", "cpu", certify_output=False, **options)
        assert set(
            json.loads(next((run / "arms/unit/ext_last/pairs").iterdir()).joinpath("report.json").read_text())["rows"]
        ) == {"FF", "REP"}
        calibrate.assess_arm(run, config, "unit", "last", "cpu", **options)
        (case / "FF.npz").write_bytes(b"changed cached field")
        with pytest.raises(ValueError, match="pair-field"):
            calibrate.assess_arm(run, config, "unit", "last", "cpu", **options)
    finally:
        torch.set_num_threads(previous_threads)


def _pair_inputs():
    velocity = torch.zeros(1, 3, 12, 12, 12)
    velocity[:, :, 4:8, 4:8, 4:8] = 0.2
    labels = (torch.arange(12**3).reshape(1, 1, 12, 12, 12) % 3).long()
    return velocity, labels


def test_boundary_rounding_is_removed_and_reported_before_the_bytes_are_written(tmp_path):
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        flow, labels = _pair_inputs()
        flow[:, 0, -1, 5, 5] = 5e-9
        rows, report = score_pair_field(
            flow, labels, labels, "case", tmp_path, 1.0, repair_output=True, certify_output=True
        )
        assert set(rows) == {"FF", "REP"} and report["boundary_rounding_removed_voxels"] == pytest.approx(5e-9)
        assert report["global_FF"]["boundary_status"] == "PASSED"
        assert report["global_REP"]["global_homeomorphism"] is True
    finally:
        torch.set_num_threads(previous)


def test_raw_dice_survives_a_failed_repair_and_real_boundary_motion_refuses_the_claim(tmp_path, monkeypatch):
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        flow, labels = _pair_inputs()

        def broken(*_args, **_kwargs):
            raise RuntimeError("repair exploded")

        monkeypatch.setattr(evaluate, "repair", broken)
        rows, report = score_pair_field(
            flow, labels, labels, "case", tmp_path / "broken", 1.0, repair_output=True, certify_output=True
        )
        assert set(rows) == {"FF"} and "exploded" in report["repair_error"]
        assert evaluate.pair_errors(report) == ["repair_error"]
        assert json.loads((tmp_path / "broken/metrics.json").read_text())["FF"]["dice_mean"] == rows["FF"]["dice_mean"]
        moved = flow.clone()
        moved[:, 0, -1, 5, 5] = 10 * BOUNDARY_ROUNDING_LIMIT
        rows, report = score_pair_field(
            moved, labels, labels, "case", tmp_path / "moved", 1.0, repair_output=True, certify_output=False
        )
        assert set(rows) == {"FF"} and "not rounding" in report["repair_error"]
    finally:
        torch.set_num_threads(previous)


def test_two_packages_of_one_run_keep_their_own_manifests(tmp_path):
    run = tmp_path / "run"
    (run / "exit_codes").mkdir(parents=True)
    (run / "exit_codes/first.txt").write_text("0")
    (run / "preview.png").write_bytes(b"png")
    package_run(run, tmp_path / "early", 0, ["first"], "head", "EARLY")
    package_run(run, tmp_path / "late", 0, ["first", "other"], "head", "LATE")
    with zipfile.ZipFile(tmp_path / "early__COMPLETE.zip") as handle:
        assert json.loads(handle.read("early__COMPLETE/package_manifest.json"))["label"] == "EARLY"
        assert "early__COMPLETE/preview.png" in handle.namelist()
    assert json.loads((run / "package_manifest.json").read_text())["label"] == "LATE"


def test_combined_package_includes_external_logs_and_excludes_heavy_payload(tmp_path):
    run, external = tmp_path / "run", tmp_path / "outside"
    (run / "exit_codes").mkdir(parents=True)
    external.mkdir()
    (run / "exit_codes/work.txt").write_text("0")
    (external / "metrics.csv").write_text("epoch,dice\n0,0.5\n")
    (external / "last.pth").write_bytes(b"weights")
    assert package_run(run, tmp_path / "export", 0, ["work"], "head", "TEST", [external]) == "COMPLETE"
    with zipfile.ZipFile(tmp_path / "export__COMPLETE.zip") as handle:
        names = handle.namelist()
        assert any(name.endswith("external/0_outside/metrics.csv") for name in names)
        assert not any(name.endswith("last.pth") for name in names)


def test_readiness_rejects_missing_optimizer_progress_and_nonfinite_state(tmp_path):
    state = {
        "epoch": 3,
        "precision": "bf16",
        "training": {"global_batch_size": 4, "world_size": 4, "rng": [None] * 4, "sampler_fingerprint": "same"},
        "state_dict": {"weight": torch.ones(2)},
        "optimizer": {"state": {0: {"step": torch.tensor(40.0), "exp_avg": torch.ones(2)}}},
    }
    a, b = tmp_path / "a.pth", tmp_path / "b.pth"
    torch.save(state, a)
    torch.save(state, b)
    assert checkpoint_comparison(a, b)["model_relative_l2_difference"] == 0
    state["optimizer"]["state"][0]["step"] = torch.tensor(39.0)
    torch.save(state, b)
    with pytest.raises(ValueError, match="step counters"):
        checkpoint_comparison(a, b)
    state["state_dict"]["weight"][0] = float("nan")
    torch.save(state, b)
    with pytest.raises(FloatingPointError, match="nonfinite"):
        checkpoint_comparison(a, b)


def test_official_export_inventory_includes_its_required_top_level_folder(tmp_path, monkeypatch):
    import nibabel as nib

    import tools.analysis.check_lumir_training as check
    from tools.analysis.run_artifacts import sha256_file

    monkeypatch.setattr(check, "LUMIR_SHAPE", (2, 3, 4))
    monkeypatch.setattr(check, "read_lumir_json", lambda _: ([], [("fixed", "moving")]))
    monkeypatch.setattr(check, "lumir_disp_name", lambda *_: "disp")
    folder = tmp_path / "predictions"
    folder.mkdir()
    field = folder / "disp.nii.gz"
    nib.save(nib.Nifti1Image(np.zeros((2, 3, 4, 3), np.float32), np.eye(4)), field)
    archive = tmp_path / "predictions.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        handle.write(field, "predictions/disp.nii.gz")
    manifest = {
        "pairs_json": "pairs.json",
        "archive": archive.name,
        "archive_sha256": sha256_file(archive),
        "files": [{"file": field.name, "sha256": sha256_file(field)}],
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    assert check.validate_export(tmp_path) == manifest
    with zipfile.ZipFile(archive, "w") as handle:
        handle.write(field, "disp.nii.gz")
    with pytest.raises(ValueError, match="ZIP"):
        check.validate_export(tmp_path)


@pytest.fixture
def calibration_dispatch(dispatch):
    root, run = dispatch
    script = Path(__file__).resolve().parents[3] / "tools/runners/train/calibration.sh"
    destination = root / "tools/runners/train"
    destination.mkdir(parents=True)
    shutil.copyfile(script.with_name("lumir.sh"), destination / "lumir.sh")
    (root / "bin/setsid").write_text('#!/usr/bin/env bash\nexec "$@"\n', newline="\n")
    (root / "bin/setsid").chmod(0o755)
    if os.name == "nt":
        (root / "bin/flock").write_text('#!/usr/bin/env bash\n[ "${FLOCK_BUSY:-0}" != 1 ]\n', newline="\n")
        (root / "bin/flock").chmod(0o755)
    parent = root / "parent.pth"
    parent.write_bytes(b"synthetic parent checkpoint")
    population = root / "experiments/population"
    population.mkdir()
    (population / "__init__.py").touch()
    (population / "calibrate.py").write_text(
        "import sys,json,os\nfrom pathlib import Path\na=sys.argv; task=a[a.index('--task')+1]; root=Path(a[a.index('--run')+1]); root.mkdir(parents=True,exist_ok=True)\n"
        "value=lambda flag,default=None: a[a.index(flag)+1] if flag in a else default\n"
        "name=value('--arm'); gpu=os.environ.get('CUDA_VISIBLE_DEVICES')\n"
        "journal=Path(os.environ['TEST_ROOT'])/'jobs.txt'\n"
        "with journal.open('a') as handle: handle.write(' '.join([str(gpu),task,str(name),value('--point','-'),value('--cohort','-'),value('--stage','-')])+'\\n')\n"
        "pair_check='--limit-pairs' in a and os.environ.get('PAIR_CHECK')\n"
        "if pair_check=='raw': raise SystemExit(1)\n"
        "marker=Path(os.environ['TEST_ROOT'])/'killed_once'\n"
        "if name and os.environ.get('KILL_ONCE')==name and not marker.exists(): marker.touch(); raise SystemExit(137)\n"
        "if name and os.environ.get('FAIL_ARM')==name: raise SystemExit(7)\n"
        "if task=='prepare': target=root/'config.json'; (root/'split.json').write_text('{}')\n"
        "elif task=='labels':\n"
        "    if os.environ.get('FAIL_LABEL') and a[a.index('--shard')+1]=='0': raise SystemExit(8)\n"
        "    target=root/('labels_'+a[a.index('--shards')+1]+'_'+a[a.index('--shard')+1]+'.done')\n"
        "elif task in ('fit','local'): target=root/('arms' if task=='fit' else 'local')/name/'report.json'\n"
        "elif task=='extend': target=root/'arms'/name/'ext/report.json'\n"
        "elif task=='assess':\n"
        "    label=value('--point','last').replace('/','_'); cohort=value('--cohort','heldout'); stage=value('--stage','all')\n"
        "    folder=root/'arms'/name/(label if cohort=='heldout' else label+'_'+cohort)\n"
        "    target=folder/'pair_check.json' if '--limit-pairs' in a else folder/(stage+'_'+value('--shard','0')+'_'+value('--shards','1')+'.done') if stage in ('fit','score') else folder/'report.json'\n"
        "elif task=='oasis19': target=root/'arms'/name/(value('--point')+'_oasis19')/'REP/summary.json'\n"
        "elif task=='reference': target=root/'references'/value('--label')/value('--cohort')/'report.json'\n"
        "elif task=='ridge': target=root/'ridge/n_277.json'\n"
        "elif task=='parent': target=root/'parent/report.json'\n"
        "else: target=root/'summary.json'\n"
        "target.parent.mkdir(parents=True,exist_ok=True); target.write_text('{}')\n"
        "raise SystemExit(1 if pair_check=='repair' else 0)\n"
    )
    exporter = root / "experiments/export_lumir.py"
    exporter.write_text(
        exporter.read_text().replace(
            "out.mkdir(parents=True)",
            "out.mkdir(parents=True,exist_ok=True)\n"
            "if os.environ.get('FAIL_CERT_CHECK') and '--limit_pairs' in a: raise SystemExit(9)\n"
            "(out/'manifest.json').write_text('{}')",
        )
    )
    trainer = root / "experiments/train_CTCF.py"
    source = trainer.read_text().replace(
        "if a[a.index('--save_ckpt')+1]=='1':",
        "if os.environ.get('FAIL_BEFORE_FIRST') and exp.endswith('_lumir_control'): raise SystemExit(8)\nif a[a.index('--save_ckpt')+1]=='1':",
    )
    # The original stub exits explicitly; delay before that exit instead.
    source = source.replace(
        "raise SystemExit(7 if",
        "if os.environ.get('SLOW_MAMBA') and exp.endswith('_mamba294'): time.sleep(18)\nraise SystemExit(7 if",
    )
    trainer.write_text(source)
    (root / "tools/analysis/check_lumir_training.py").write_text(
        "import sys,os\nfrom pathlib import Path\n"
        "if os.environ.get('FAIL_READY'): raise SystemExit(8)\n"
        "target=Path(sys.argv[sys.argv.index('--run')+1])/'report.json'; target.parent.mkdir(parents=True,exist_ok=True); target.write_text('{\"experiments\":[\"unit_ready\"]}')\n"
    )

    def launch(**options):
        return run(
            **(
                dict(
                    TEST_SCRIPT=script.as_posix(),
                    RUN_ID="unit_cal",
                    A_GPUS="1 2",
                    MAMBA_GPU="3",
                    LUMIR_GPUS="4 5 6 7",
                    MIN_FREE_GIB="0",
                    PARENT_CHECKPOINT=parent.as_posix(),
                    A_SLOTS="8",
                    QUEUE_POLL_SECONDS="0.1",
                    KILL_RETRY_SECONDS="0",
                )
                | options
            ),
            timeout=120,
        )

    return root, launch


def test_calibration_automatically_starts_authorized_control_and_can_resume(calibration_dispatch):
    root, run = calibration_dispatch
    result = run()
    assert result.returncode == 0, result.stderr + result.stdout
    assert list((root / "results/exports").glob("unit_cal*__COMPLETE.zip"))
    assert (root / "results/unit_cal_lumir_control/ckpt/last.pth").exists()
    again = run()
    assert again.returncode == 0, again.stderr + again.stdout
    assert "SKIP lumir_control" in again.stdout and "SKIP mamba294" in again.stdout
    assert "SKIP fit_a1_278" in again.stdout and "SKIP assess_a1_278_last" in again.stdout
    args = json.loads((root / "logs/unit_cal_mamba294/args.json").read_text())
    assert args[args.index("--max_epoch") + 1] == "250" and args[args.index("--schedule_max_epoch") + 1] == "500"
    jobs = (root / "jobs.txt").read_text().splitlines()
    position = {line.split(" ", 1)[1]: index for index, line in reversed(list(enumerate(jobs)))}
    assert position["fit lin64_s0 - - -"] < position["assess lin64_s0 last - -"] < position["labels None - - -"]
    assert position["assess lin64_s0 last - -"] < position["assess a1_64_s0 last - -"] < position["fit a1_278 - - -"]
    assert position["fit a1_278 - - -"] < position["assess a1_278 last - -"]
    assert (
        position["assess a1_278 last dev fit"]
        < position["assess a1_278 last dev score"]
        < position["assess a1_278 last dev summary"]
    )
    assert (root / "results/lumir/unit_cal_lumir_control/certified_lumir25/manifest.json").is_file()
    check = root / "results/calibration/unit_cal/readiness/certified_check"
    arguments = json.loads((check / "args.json").read_text())
    assert arguments[arguments.index("--ckpt") + 1] == "results/unit_ready/ckpt/last.pth"
    assert arguments[arguments.index("--limit_pairs") + 1] == "1"
    assert json.loads((check / "env.json").read_text())["gpus"] == "4"
    assert any("__LUMIR__COMPLETE" in p.name for p in (root / "results/exports").glob("*.zip"))
    assert any("__A64__COMPLETE" in p.name for p in (root / "results/exports").glob("*.zip"))


def test_sigkilled_job_restarts_a_bounded_number_of_times(calibration_dispatch):
    root, run = calibration_dispatch
    result = run(KILL_ONCE="lin64_s0")
    assert result.returncode == 0, result.stderr + result.stdout
    assert "RETRY after SIGKILL (1/2)" in result.stdout
    assert (root / "results/calibration/unit_cal/exit_codes/fit_lin64_s0.txt").read_text().strip() == "0"


def test_freed_gpus_join_the_a_queue(calibration_dispatch):
    root, run = calibration_dispatch
    result = run(A_GPUS="1", A_SLOTS="4")
    assert result.returncode == 0, result.stderr + result.stdout
    assert "GPUs joining the A queue: 5 6 7" in result.stdout and "GPUs joining the A queue: 3" in result.stdout
    late = [line.split()[0] for line in (root / "jobs.txt").read_text().splitlines() if " assess " in f" {line}"]
    assert "1" in late and set(late) & {"3", "4", "5", "6", "7"}


def test_new_commit_continues_only_when_explicitly_allowed(calibration_dispatch):
    root, run = calibration_dispatch
    first = run(FAIL_ARM="a1_64_s0")
    assert first.returncode != 0
    (root / "results/calibration/unit_cal/head.txt").write_text("older\n")
    refused = run()
    assert refused.returncode != 0 and "another HEAD" in refused.stdout
    resumed = run(ALLOW_NEW_HEAD="1")
    assert resumed.returncode == 0, resumed.stderr + resumed.stdout
    assert (root / "results/calibration/unit_cal/head.txt").read_text().strip() == "older"
    assert "reviewed" in (root / "results/calibration/unit_cal/heads.txt").read_text().splitlines()[-1]


def test_broken_raw_chain_holds_evaluation_but_not_training(calibration_dispatch):
    root, run = calibration_dispatch
    result = run(PAIR_CHECK="raw")
    assert result.returncode != 0, result.stderr + result.stdout
    directory = root / "results/calibration/unit_cal"
    jobs = [line.split()[1] for line in (root / "jobs.txt").read_text().splitlines()]
    assert jobs.count("assess") == 1 and "oasis19" not in jobs
    assert (directory / "a/arms/a1_278/report.json").is_file() and (
        directory / "a/arms/a1_64_s0/ext/report.json"
    ).is_file()
    assert (directory / "exit_codes/assess_a1_64_s0_last.txt").read_text().strip() == "125"
    assert (root / "results/unit_cal_lumir_control/ckpt/last.pth").is_file()


def test_failed_repair_in_the_pair_check_leaves_raw_evaluation_running(calibration_dispatch):
    root, run = calibration_dispatch
    result = run(PAIR_CHECK="repair")
    assert result.returncode != 0, result.stderr + result.stdout
    directory = root / "results/calibration/unit_cal"
    assert (directory / "exit_codes/pair_check.txt").read_text().strip() == "1"
    assert (directory / "a/arms/a1_64_s0/last/report.json").is_file()
    assert (directory / "a/arms/a1_278/last_dev/report.json").is_file()


def test_failed_certified_check_holds_only_the_certified_export(calibration_dispatch):
    root, run = calibration_dispatch
    result = run(FAIL_CERT_CHECK="1")
    assert result.returncode != 0, result.stderr + result.stdout
    assert "HELD lumir_certified" in result.stdout
    assert (root / "results/unit_cal_lumir_control/ckpt/last.pth").is_file()
    assert not (root / "results/lumir/unit_cal_lumir_control/certified_lumir25").exists()
    assert (root / "results/calibration/unit_cal/exit_codes/lumir_certified.txt").read_text().strip() == "125"


def test_failed_labels_do_not_cancel_image_training(calibration_dispatch):
    root, run = calibration_dispatch
    result = run(FAIL_LABEL="1")
    assert result.returncode != 0
    for name in calibrate.ARMS:
        assert (root / f"results/calibration/unit_cal/a/arms/{name}/report.json").is_file()
    assert not (root / "results/calibration/unit_cal/a/local").exists()
    assert (root / "results/unit_cal_lumir_control/ckpt/last.pth").is_file()


def test_low_disk_stops_before_gpu_tasks(calibration_dispatch):
    root, run = calibration_dispatch
    result = run(MIN_FREE_GIB="1000000")
    assert result.returncode != 0
    assert "required=1e+06" in result.stdout
    assert not (root / "logs").exists()


def test_missing_parent_is_detected_before_any_training(calibration_dispatch):
    root, run = calibration_dispatch
    (root / "parent.pth").unlink()
    result = run()
    assert result.returncode != 0
    assert "missing P10 checkpoint" in result.stdout
    assert not (root / "results/calibration").exists()


@pytest.mark.skipif(os.name != "nt", reason="Windows-only flock control-flow stub")
def test_duplicate_worker_exits_before_touching_run_outputs(calibration_dispatch):
    root, run = calibration_dispatch
    result = run(FLOCK_BUSY="1")
    assert result.returncode != 0 and "already active" in result.stdout
    assert not (root / "results/calibration/unit_cal/logs").exists()
    assert not (root / "results/exports").exists()


@pytest.mark.skipif(os.name != "posix", reason="real flock semantics are checked on Linux")
def test_live_posix_lock_rejects_a_duplicate_and_releases_after_owner_exit(calibration_dispatch):
    import fcntl

    root, run = calibration_dispatch
    directory = root / "results/calibration/unit_cal"
    directory.mkdir(parents=True)
    with (directory / ".lock").open("w") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        rejected = run()
        assert rejected.returncode != 0 and "already active" in rejected.stdout
        assert not (directory / "logs").exists()
    completed = run()
    assert completed.returncode == 0, completed.stdout + completed.stderr


def test_bad_resume_returns_without_exiting_the_calling_shell(calibration_dispatch):
    root, run = calibration_dispatch
    wrapper = root / "resume_check.sh"
    wrapper.write_text(
        'bash "$CALIBRATION_SCRIPT" resume unknown_run\ncode=$?\necho "CALLING_SHELL_ALIVE=$code"\nexit "$code"\n',
        newline="\n",
    )
    script = Path(__file__).resolve().parents[3] / "tools/runners/train/calibration.sh"
    result = run(TEST_SCRIPT=wrapper.as_posix(), CALIBRATION_SCRIPT=script.as_posix())
    assert result.returncode != 0
    assert "CALLING_SHELL_ALIVE=1" in result.stdout


def test_start_command_backgrounds_the_same_worker(calibration_dispatch):
    root, run = calibration_dispatch
    wrapper = root / "start_check.sh"
    wrapper.write_text('exec bash "$CALIBRATION_SCRIPT" start\n', newline="\n")
    script = Path(__file__).resolve().parents[3] / "tools/runners/train/calibration.sh"
    result = run(TEST_SCRIPT=wrapper.as_posix(), CALIBRATION_SCRIPT=script.as_posix())
    assert result.returncode == 0, result.stderr + result.stdout
    assert "PID=" in result.stdout and "RUN_ID=unit_cal LOG=results/calibration/unit_cal_" in result.stdout
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        archives = list((root / "results/exports").glob("unit_cal*__COMPLETE.zip"))
        if any("__A_LUMIR" not in path.name for path in archives):
            break
        time.sleep(0.2)
    else:
        pytest.fail("background worker did not produce its final archive")


def test_early_archive_does_not_wait_for_mamba(calibration_dispatch):
    root, run = calibration_dispatch
    result = run(SLOW_MAMBA="1")
    assert result.returncode == 0, result.stderr + result.stdout
    early = next((root / "results/exports").glob("*__A_LUMIR__COMPLETE.zip"))
    with zipfile.ZipFile(early) as handle:
        manifest = json.loads(handle.read(f"{early.stem}/package_manifest.json"))
    assert "mamba294" not in manifest["required_steps"]
    assert any("__A_LUMIR" not in p.name for p in (root / "results/exports").glob("*__COMPLETE.zip"))


def test_lumir_retry_before_first_checkpoint_preserves_old_attempt(calibration_dispatch):
    root, run = calibration_dispatch
    failed = run(FAIL_BEFORE_FIRST="1", KILL_RETRIES="0")
    assert failed.returncode != 0
    original = root / "results/lumir/unit_cal_lumir_control"
    assert original.is_dir()
    result = run()
    assert result.returncode == 0, result.stderr + result.stdout
    experiment = (root / "results/calibration/unit_cal/lumir_exp.txt").read_text().strip()
    assert experiment != "unit_cal_lumir_control" and original.is_dir()
    assert (root / f"results/{experiment}/ckpt/last.pth").is_file()


@pytest.mark.parametrize("failure", ({"FAIL_READY": "1"}, {"FAIL_ARM": "a1_64_s0"}))
def test_readiness_failure_blocks_only_long_lumir_and_a_failure_does_not(calibration_dispatch, failure):
    root, run = calibration_dispatch
    result = run(**failure)
    assert result.returncode != 0, result.stderr + result.stdout
    assert (root / "results/unit_cal_mamba294/ckpt/last.pth").exists()
    assert (root / "results/calibration/unit_cal/a/arms/lin278/report.json").exists()
    assert (root / "results/unit_cal_lumir_control/ckpt/last.pth").exists() is ("FAIL_READY" not in failure)
