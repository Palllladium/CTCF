from __future__ import annotations

import json
import shutil
from types import SimpleNamespace

import pytest
import torch

from datasets.IXI import read_ixi_manifest
from experiments import matching_diagnostics as diagnostics, train_matching as training
from tools.analysis.run_artifacts import sha256_file
from tools.analysis.tests.test_matching_train import (
    prepared as prepared,
    small_cpu_pool as small_cpu_pool,
)


def test_common_solver_calibration_uses_only_train_images_and_native_units(prepared, monkeypatch):
    root, path, _, _ = prepared
    args = SimpleNamespace(out=root / "calibration", stride=1, seed=19, pairs=1, points=8)
    args.out.mkdir()
    monkeypatch.setattr(diagnostics, "SegNorm", lambda: pytest.fail("Calibration accessed anatomical labels"))
    manifest = read_ixi_manifest(path, require_t2=False)
    result = diagnostics.calibrate(args, manifest, torch.device("cpu"))
    assert result["binding"]["manifest_sha256"] == manifest["fingerprint"]
    assert len(result["trials"]) == 12
    winner = min(result["trials"], key=lambda row: (row["mean"], row["step_size"], -row["reg_weight"]))
    assert result["selected"] == {key: winner[key] for key in ("step_size", "reg_weight")}
    ncc_winner = min(result["ncc_trials"], key=lambda row: (row["mean"], row["step_size"], -row["reg_weight"]))
    assert result["ncc_selected"] == {key: ncc_winner[key] for key in ("step_size", "reg_weight")}
    assert all(torch.isfinite(torch.tensor(row["endpoint_errors"])).all() for row in result["trials"])
    args.seed += 1
    with pytest.raises(ValueError, match="Diagnostic inputs changed"):
        diagnostics.calibrate(args, manifest, torch.device("cpu"))


def test_teacher_monitoring_excludes_fit_and_val_and_probes_do_not_change_weights(prepared, monkeypatch):
    root, manifest_path, base, arguments = prepared
    document = json.loads(manifest_path.read_text())
    source = document["training"][0]
    for index in range(3):
        t1, t2 = root / f"extra{index}.pkl", root / f"extra{index}.npy"
        shutil.copyfile(source["t1"], t1)
        shutil.copyfile(source["t2"], t2)
        document["training"].append(
            source | {"id": f"extra{index}", "ixi_id": f"IXIextra{index}", "t1": str(t1), "t2": str(t2)}
        )
    document["expected_counts"]["training"] = 6
    manifest_path.write_text(json.dumps(document))
    training.run(arguments("teacher", "teacher_probe", "--updates", "1", "--source", "t1"))
    checkpoint = root / "teacher_probe/ckpt/last.pth"
    saved = torch.load(checkpoint, weights_only=False)
    assert not set(saved["teacher_fit_ids"]) & set(saved["teacher_heldout_ids"])
    assert not set(saved["teacher_heldout_ids"]) & {row["id"] for row in document["validation"]}
    assert len(saved["teacher_fit_ids"]) == 2 and len(saved["teacher_heldout_ids"]) == 4
    before = sha256_file(checkpoint)
    args = SimpleNamespace(teacher=checkpoint, base_checkpoint=base, out=root / "probes", pairs=2, points=8, seed=17)
    args.out.mkdir()
    seen = []
    original = diagnostics.pkload

    def load(path):
        seen.append(str(path))
        return original(path)

    monkeypatch.setattr(diagnostics, "pkload", load)
    monkeypatch.setattr(diagnostics, "load_base", training.load_base)
    torch.manual_seed(args.seed)
    result = diagnostics.teacher_probes(args, read_ixi_manifest(manifest_path), torch.device("cpu"))
    assert sha256_file(checkpoint) == before
    assert not set(seen) & {row["t1"] for row in document["validation"]}
    intersubject = [row for row in result["rows"] if row.get("readout")]
    assert len(intersubject) == 8
    assert all(row["same_class_fraction"] == 1 for row in intersubject)
    synthetic = {row["variant"]: row for row in result["rows"] if row["pair"] == 0 and "known_local_top1" in row}
    assert {
        "true",
        "copy_t1",
        "zero_t2",
        "foreign_t2",
        "blank",
        "spatial_shuffle",
        "nonmatching_image",
    } <= synthetic.keys()
    for variant in ("copy_t1", "zero_t2", "foreign_t2"):
        assert synthetic[variant]["known_local_top1"] == synthetic["true"]["known_local_top1"]
        assert synthetic[variant]["positive_margin"] == synthetic["true"]["positive_margin"]
