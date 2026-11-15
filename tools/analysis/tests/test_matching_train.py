from __future__ import annotations

import json
import pickle
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from datasets.IXI import IXIPairedDataset, read_ixi_manifest
from experiments import train_matching as training
from models.Matching import FeatureEncoder
from tools.analysis.run_artifacts import sha256_file

LOAD_BASE = training.load_base


@pytest.fixture(autouse=True)
def small_cpu_pool():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    rows = []
    coordinates = np.indices((32, 32, 32), dtype=np.float32)
    for index in range(5):
        image = (
            0.3 + coordinates[0] / 90 + np.sin(coordinates[1] / (3 + index)) / 12 + np.cos(coordinates[2] / 5) / 10
        ).astype(np.float32)
        path = tmp_path / f"t1_{index}.pkl"
        with path.open("wb") as stream:
            pickle.dump((image, np.zeros_like(image, dtype=np.int16)), stream)
        side = tmp_path / f"t2_{index}.npy"
        np.save(side, np.sqrt(image))
        rows.append(
            {
                "id": f"subject_{index}",
                "ixi_id": f"IXI{index:03d}",
                "split": "Train" if index < 3 else "Val",
                "t1": str(path),
                "t2": str(side),
                "t1_sha256": sha256_file(path),
                "t2_sha256": sha256_file(side),
            }
        )
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "reviewed": True,
                "training": rows[:3],
                "validation": rows[3:],
                "expected_counts": {"training": 3, "validation": 2},
                "shape": [32, 32, 32],
                "validation_pairs": [["subject_3", "subject_4"], ["subject_4", "subject_3"]],
            }
        )
    )

    class Base(nn.Module):
        cfg = SimpleNamespace(image_channels=2)

        def forward(self, moving, fixed, **kwargs):
            return moving, moving.new_zeros((moving.shape[0], 3, *moving.shape[2:]))

    monkeypatch.setattr(training, "load_base", lambda _path, _shape, _device: Base())
    base = tmp_path / "base.pth"
    torch.save({"arguments": {"image_channels": 2}}, base)

    def arguments(stage, name, *extra):
        return training.parse_args(
            [
                stage,
                "--manifest",
                str(manifest),
                "--out",
                str(tmp_path / name),
                "--device",
                "cpu",
                "--points",
                "8",
                "--width",
                "2",
                "--teacher_width",
                "2",
                "--channels",
                "4",
                "--stride",
                "1",
                "--updates",
                "4",
                "--log_every",
                "1",
                "--save_every",
                "2",
                *extra,
            ]
        )

    return tmp_path, manifest, base, arguments


def test_teacher_resume_restores_optimizer_rng_and_history(prepared, monkeypatch):
    root, _, _, arguments = prepared
    full = arguments("teacher", "full")
    training.run(full)
    interrupted = arguments("teacher", "interrupted")
    save = training.save_ckpt

    def stop_after_checkpoint(state, directory, is_best):
        save(state, directory, is_best)
        if state["step"] == 2:
            raise RuntimeError("test interruption")

    monkeypatch.setattr(training, "save_ckpt", stop_after_checkpoint)
    with pytest.raises(RuntimeError, match="test interruption"):
        training.run(interrupted)
    monkeypatch.setattr(training, "save_ckpt", save)
    report = training.run(interrupted)
    assert report["resumed_update"] == 2
    expected = torch.load(root / "full/ckpt/last.pth", weights_only=False)
    actual = torch.load(root / "interrupted/ckpt/last.pth", weights_only=False)
    for name in expected["state_dict"]:
        torch.testing.assert_close(actual["state_dict"][name], expected["state_dict"][name], atol=0, rtol=0)
    assert actual["history"] == expected["history"]
    assert len((root / "interrupted/metrics.csv").read_text().splitlines()) == 5
    changed = arguments("teacher", "interrupted", "--lr", "0.002")
    with pytest.raises(ValueError, match="recipe changed"):
        training.run(changed)


def test_t1_teacher_never_opens_t2_arrays(prepared):
    root, manifest, _, arguments = prepared
    content = json.loads(manifest.read_text())
    for row in content["training"] + content["validation"]:
        row["t2"] = str(root / "missing_t2.npy")
    manifest.write_text(json.dumps(content))
    report = training.run(arguments("teacher", "t1", "--source", "t1", "--updates", "1"))
    assert report["completed_updates"] == 1


def test_student_solver_backpropagates_through_frozen_teacher_and_reloads(prepared):
    root, _, base, arguments = prepared
    training.run(arguments("teacher", "teacher", "--updates", "1"))
    path = root / "teacher/ckpt/last.pth"
    before = sha256_file(path)
    args = arguments(
        "student",
        "student",
        "--teacher",
        str(path),
        "--base_checkpoint",
        str(base),
        "--updates",
        "1",
        "--mode",
        "solver",
    )
    report = training.run(args)
    model, saved = training.load_encoder(root / "student/ckpt/last.pth", "cpu")
    assert report["completed_updates"] == saved["step"] == 1
    assert sha256_file(path) == before
    assert report["teacher_sha256"] == before
    assert saved["history"][0]["gradient_norm"] > 0
    assert saved["history"][0]["outer"] > 0
    assert model.stem[0].in_channels == 1
    assert training.refine_config(saved)["steps"] == 3


def test_profile_uses_real_solver_gradient_and_explicit_random_teacher(prepared):
    root, _, base, arguments = prepared
    args = arguments("profile", "profile", "--base_checkpoint", str(base), "--mode", "solver")
    report = training.run(args)
    assert report["passed"]
    assert report["measured_updates"] == 2
    assert report["update_seconds"] > 0
    assert report["random_teacher_profile"]
    assert (root / "profile/profile.json").is_file()
    assert not (root / "profile/ckpt/last.pth").exists()


def test_profile_fallback_is_limited_to_explicit_memory_failures(prepared, monkeypatch):
    _, manifest, _, _ = prepared
    args = ["profile", "--manifest", str(manifest), "--out", str(manifest.parent / "memory")]
    monkeypatch.setattr(training, "run", lambda _: {"passed": False, "reason": "MEMORY_LIMIT"})
    assert training.main(args) == 86

    def broken_math(_):
        raise FloatingPointError("bad gradient")

    monkeypatch.setattr(training, "run", broken_math)
    with pytest.raises(FloatingPointError, match="bad gradient"):
        training.main(args)


def test_student_rejects_incomplete_teacher_budget(prepared):
    root, _, base, arguments = prepared
    training.run(arguments("teacher", "teacher_budget", "--updates", "1"))
    path = root / "teacher_budget/ckpt/last.pth"
    saved = torch.load(path, weights_only=False)
    saved["recipe"]["updates"] = 4000
    torch.save(saved, path)
    args = arguments("student", "student_budget", "--teacher", str(path), "--base_checkpoint", str(base))
    with pytest.raises(ValueError, match="fixed training budget"):
        training.run(args)


def test_main_students_share_initialization_with_differently_sized_frozen_teachers(prepared):
    root, _, base, arguments = prepared
    fingerprints = []
    for source in ("t1", "paired"):
        name = f"teacher_wide_{source}"
        training.run(arguments("teacher", name, "--source", source, "--teacher_width", "3", "--updates", "1"))
        teacher = root / name / "ckpt/last.pth"
        for mode in ("value", "solver"):
            report = training.run(
                arguments(
                    "student",
                    f"student_shared_{source}_{mode}",
                    "--source",
                    source,
                    "--mode",
                    mode,
                    "--teacher",
                    str(teacher),
                    "--base_checkpoint",
                    str(base),
                    "--updates",
                    "1",
                )
            )
            fingerprints.append(report["initial_model_sha256"])
            assert report["encoder_options"]["width"] == 2
    assert len(set(fingerprints)) == 1


def test_solver_student_resume_restores_the_calibrated_run_exactly(prepared, monkeypatch):
    root, manifest, base, arguments = prepared
    training.run(arguments("teacher", "resume_teacher", "--updates", "1"))
    teacher = root / "resume_teacher/ckpt/last.pth"
    calibration = root / "calibration.json"
    calibration.write_text(
        json.dumps(
            {
                "binding": {"manifest_sha256": sha256_file(manifest)},
                "selected": {"step_size": 0.1, "reg_weight": 0.03},
                "ncc_selected": {"step_size": 0.05, "reg_weight": 0.01},
            }
        )
    )
    options = [
        "--mode",
        "solver",
        "--teacher",
        str(teacher),
        "--base_checkpoint",
        str(base),
        "--calibration",
        str(calibration),
    ]
    training.run(arguments("student", "resume_full", *options))
    save = training.save_ckpt

    def interrupt(state, directory, is_best):
        save(state, directory, is_best)
        if state["step"] == 2:
            raise RuntimeError("student interruption")

    monkeypatch.setattr(training, "save_ckpt", interrupt)
    with pytest.raises(RuntimeError, match="student interruption"):
        training.run(arguments("student", "resume_split", *options))
    monkeypatch.setattr(training, "save_ckpt", save)
    result = training.run(arguments("student", "resume_split", *options))
    assert result["resumed_update"] == 2
    full = torch.load(root / "resume_full/ckpt/last.pth", weights_only=False)
    split = torch.load(root / "resume_split/ckpt/last.pth", weights_only=False)
    assert full["history"] == split["history"]
    assert split["recipe"]["step_size"] == 0.1 and split["recipe"]["reg_weight"] == 0.03
    assert split["ncc_config"] == {"step_size": 0.05, "reg_weight": 0.01}
    for key in full["state_dict"]:
        torch.testing.assert_close(full["state_dict"][key], split["state_dict"][key], rtol=0, atol=0)


@pytest.mark.parametrize(
    "device",
    ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable"))],
)
def test_solver_outer_loss_has_its_own_nonzero_encoder_hypergradient(prepared, device):
    _, manifest_path, base_path, arguments = prepared
    device = torch.device(device)
    args = arguments("student", "outer_gradient", "--base_checkpoint", str(base_path))
    args.kind = "student"
    dataset = IXIPairedDataset(read_ixi_manifest(manifest_path), "b1")
    batch = training.training_pair(dataset, 0, args.seed, device)
    base = training.load_base(base_path, (32, 32, 32), device).to(device)
    torch.manual_seed(19)
    encoder = FeatureEncoder(1, width=2, channels=4, stride=1).to(device)
    teacher = FeatureEncoder(2, width=3, channels=4, stride=1).to(device).eval().requires_grad_(False)
    cpu_rng = torch.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state(device) if device.type == "cuda" else None
    args.mode = "value"
    value_loss, _ = training.matching_step(encoder, teacher, base, batch, args)
    torch.set_rng_state(cpu_rng)
    if cuda_rng is not None:
        torch.cuda.set_rng_state(cuda_rng, device)
    args.mode = "solver"
    solver_loss, _ = training.matching_step(encoder, teacher, base, batch, args)
    gradients = torch.autograd.grad(solver_loss - value_loss, tuple(encoder.parameters()))
    assert all(torch.isfinite(gradient).all() for gradient in gradients)
    assert sum(float(gradient.abs().sum()) for gradient in gradients) > 1e-7
    assert all(parameter.grad is None for parameter in teacher.parameters())


@pytest.mark.parametrize("recorded,expected", [(None, 1), (2, 2)])
def test_base_loading_preserves_scalar_or_paired_input_channels(prepared, monkeypatch, recorded, expected):
    root, _, _, _ = prepared
    captured = {}

    class Base(nn.Module):
        def forward(self, moving, fixed, **kwargs):
            assert moving.shape[1] == fixed.shape[1] == expected
            return moving, moving.new_zeros((1, 3, *moving.shape[2:]))

    class Adapter:
        def build(self, **kwargs):
            captured.update(kwargs)
            model = Base()
            model.cfg = SimpleNamespace(image_channels=kwargs["image_channels"])
            return model

    payload = {} if recorded is None else {"arguments": {"image_channels": recorded}}
    path = root / "input_channels.pth"
    torch.save(payload, path)
    monkeypatch.setattr(training, "get_model_adapter", lambda _: Adapter())
    monkeypatch.setattr(training, "load_checkpoint_state", lambda *args, **kwargs: {})
    model = LOAD_BASE(path, (32, 32, 32), "cpu")
    result = training.base_flow(model, torch.zeros(1, 1, 32, 32, 32), torch.zeros(1, 1, 32, 32, 32), "fp32")
    assert captured["image_channels"] == expected
    assert result.shape == (1, 3, 32, 32, 32)
