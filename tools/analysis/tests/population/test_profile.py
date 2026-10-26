import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from experiments.population import profile


def test_residual_spectrum_preserves_both_sides_of_a_travelling_wave():
    z, _y, x = torch.meshgrid(*(torch.arange(8, dtype=torch.float64) for _ in range(3)), indexing="ij")
    residual = torch.cos(2 * torch.pi * (z + x) / 8)
    spectra = profile.axis_frequency_energy(residual)
    expected = float(residual.square().sum())
    for values in spectra:
        assert sum(values) == pytest.approx(expected)
    for axis in (0, 2):
        assert spectra[axis][1] == pytest.approx(expected / 2)
        assert spectra[axis][-1] == pytest.approx(expected / 2)


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    data = tmp_path / "All"
    data.mkdir()
    coordinates = np.stack(np.meshgrid(*([np.arange(9)] * 3), indexing="ij"))
    for index in range(1, 8):
        center = np.array([4, 4, 3 + index % 3])[:, None, None, None]
        image = np.exp(-((coordinates - center) ** 2).sum(0) / 5).astype(np.float32)
        labels = (image > 0.5).astype(np.int16)
        (data / f"p_{index:04d}.pkl").write_bytes(pickle.dumps((image, labels)))
    split = tmp_path / "split.json"
    split.write_text(
        json.dumps(
            {
                "training": [{"subject_id": f"OASIS_ALL_P{i:04d}"} for i in range(1, 7)],
                "development": [{"subject_id": "OASIS_ALL_P0007"}],
            }
        )
    )
    monkeypatch.setattr(profile, "SPLIT", split)
    for name, value in {
        "PROFILE_WIDTH": 2,
        "PROFILE_LEVELS": 1,
        "PROFILE_EPOCHS": 2,
        "PROBE_STEPS": 1,
        "MAP_STEPS": 1,
    }.items():
        monkeypatch.setenv(name, str(value))
    args = argparse.Namespace(
        task="prepare", run=tmp_path / "run", data_root=data, parent_checkpoint=tmp_path / "parent.pth", device="cpu"
    )
    config = profile.prepare(args)
    yield args, config
    torch.set_num_threads(previous)


def test_prepare_freezes_train_selection_and_detects_changed_source(prepared):
    args, config = prepared
    assert "OASIS_ALL_P0007" not in config["subject_ids"]
    assert len(config["inner_train_indices"]) == 4
    assert profile.prepare(args) == config
    template = Path(config["template"])
    original = template.read_bytes()
    template.write_bytes(original + b"changed")
    with pytest.raises(ValueError, match="Train-mean bytes changed"):
        profile.prepare(args)
    template.write_bytes(original)
    path = Path(config["subjects"][0])
    path.write_bytes(path.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="bytes changed"):
        profile.prepare(args)


def test_profile_paths_and_summary_do_not_claim_a_main_experiment(prepared, monkeypatch):
    args, config = prepared
    args.task = "linear"
    output = profile._attempt(args.run, args.task)
    report = profile.profile_arm(args, config, output)
    assert report["arm"] == "A_lin" and report["train_updates"] == 8
    assert len(report["cases"]) == 2
    assert "seconds_compute" in report["epochs"][0]
    profile.write_json(args.run / args.task / "report.json", report)
    (args.run / f"{args.task}.done").touch()
    for task in profile.PROBES:
        args.task = task
        report = profile.profile_probe(args, config, profile._attempt(args.run, task))
        profile.write_json(args.run / task / "report.json", report)
        (args.run / f"{task}.done").touch()

    from experiments.dispatch1.bridges import Parent

    class FakeParent:
        def __init__(self):
            self.load_report = {"kind": "TEST_STUB"}

        def model(self, moving, fixed, **kwargs):
            return moving, torch.zeros(1, 3, *moving.shape[2:])

    args.parent_checkpoint.write_bytes(b"test stub, not a real model")
    monkeypatch.setattr(Parent, "load", lambda *args: FakeParent())
    args.task = "k3"
    k3 = profile.profile_residuals(args, config, profile._attempt(args.run, "k3"))
    assert k3["parent_load"] == {"kind": "TEST_STUB"}
    assert len(k3["cases"]) == 4
    assert all(np.isfinite(row["nll_per_voxel"]) for row in k3["cases"])
    summary = profile.summarize(args, config)
    assert summary["scope"] == "PRELIMINARY_DIAGNOSTICS_NOT_MAIN_TRAINING_OR_MODEL_SELECTION"
    assert "a1" in summary["missing_tasks"]
    assert summary["k2"]["status"] in ("DESCRIPTIVE_ONLY", "UNINFORMATIVE")
    args.parent_checkpoint.write_bytes(b"changed parent")
    with pytest.raises(ValueError, match="parent checkpoint bytes changed"):
        profile.prepare(args)


def test_summary_cli_reports_incomplete_tasks_as_failure(prepared, monkeypatch):
    args, _config = prepared
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "profile",
            "--task",
            "summary",
            "--run",
            str(args.run),
            "--data-root",
            str(args.data_root),
            "--parent-checkpoint",
            str(args.parent_checkpoint),
        ],
    )
    with pytest.raises(SystemExit) as failure:
        profile.main()
    assert failure.value.code == 1
    summary = json.loads((args.run / "summary.json").read_text())
    assert len(summary["missing_tasks"]) == 10
