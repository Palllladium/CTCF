import json
import os
import shutil
from pathlib import Path
from textwrap import dedent

import pytest

from tools.analysis.tests.test_lumir_runner import dispatch as dispatch

SCRIPT = Path(__file__).resolve().parents[3] / "tools/runners/train/contrasts.sh"


@pytest.fixture
def learning_dispatch(dispatch):
    root, run = dispatch
    (root / "results/contrasts/unit").mkdir(parents=True)
    destination = root / "tools/runners/train"
    destination.mkdir(parents=True)
    shutil.copyfile(SCRIPT, destination / SCRIPT.name)
    launcher = root / "launch.sh"
    launcher.write_text('exec bash "$TEST_ROOT/tools/runners/train/contrasts.sh" _work unit\n', newline="\n")
    for name in ("parent.pth", "base.pth", "b0.pth", "manifest.json", "external.json"):
        (root / name).write_text("{}")
    (root / "oas").mkdir()
    oasis = root / "datasets/splits/oasis_subject_disjoint_v2"
    oasis.mkdir(parents=True)
    (oasis / "pair_manifest.json").write_text("{}")
    stub = dedent("""\
        import json, os, sys, time
        from pathlib import Path

        args = sys.argv[1:]
        stage = args[0]
        module = Path(__file__).stem
        def option(name, default=""):
            return args[args.index(name) + 1] if name in args else default
        out = Path(option("--out"))
        out.mkdir(parents=True, exist_ok=True)
        calls = Path(os.environ["TEST_ROOT"]) / "learning_calls"
        calls.mkdir(exist_ok=True)
        row = dict(module=module, stage=stage, args=args, gpu=os.environ.get("CUDA_VISIBLE_DEVICES"),
                   source=option("--source"), kind=option("--kind"), mode=option("--mode"),
                   stride=option("--stride"), width=option("--width", "16"), out=str(out))
        (calls / f"{time.monotonic_ns()}_{os.getpid()}.json").write_text(json.dumps(row))
        if module == "train_matching":
            if stage == "profile" and option("--kind") == "student" and option("--stride") == "1":
                raise_code = int(os.environ.get("LEARNING_PROFILE1_EXIT", "0"))
                if raise_code:
                    raise SystemExit(raise_code)
            if stage == "student":
                teacher = Path(option("--teacher"))
                assert teacher.is_file(), teacher
                assert teacher.read_text() == option("--source"), teacher
                assert Path(option("--base_checkpoint")).is_file()
            if stage in {"teacher", "student"}:
                ckpt = out / "ckpt"
                ckpt.mkdir(exist_ok=True)
                (ckpt / "last.pth").write_text(option("--source"))
        if module == "matching_diagnostics":
            if stage == "calibrate":
                (out / "calibration.json").write_text("{}")
            elif os.environ.get("LEARNING_FAIL_DIAGNOSTIC") == "1":
                raise SystemExit(8)
        if module == "contrast_audit":
            if stage == "targets":
                (out / "ready.json").write_text("{}")
            if stage == os.environ.get("CLOSURE_FAIL_STAGE"):
                raise SystemExit(11)
        if module == "contrast_learning":
            if stage == "prepare":
                if os.environ.get("LEARNING_FAIL_PREPARE") == "1":
                    raise SystemExit(9)
                (out / "ready.json").write_text("{}")
            if stage == "train":
                if not option("--kind").startswith("scratch_"):
                    assert (Path(option("--prepared")) / "ready.json").is_file()
                ckpt = Path("results") / option("--exp") / "ckpt"
                ckpt.mkdir(parents=True, exist_ok=True)
                (ckpt / "last.pth").write_text("weights")
        if module == "eval_matching" and stage == "evaluate":
            assert Path(option("--manifest")).is_file()
            for key in ("--base_checkpoint", "--metric_checkpoint", "--endpoint"):
                if option(key):
                    assert Path(option(key)).is_file(), (key, option(key))
        (out / "report.json").write_text("{}")
    """)
    for name in (
        "contrast_learning",
        "contrast_diagnostics",
        "train_matching",
        "eval_matching",
        "matching_diagnostics",
        "contrast_audit",
    ):
        (root / f"experiments/{name}.py").write_text(stub, newline="\n")
    if os.name == "nt":
        for name in ("flock", "ps"):
            path = root / "bin" / name
            path.write_text("#!/usr/bin/env bash\nexit 0\n", newline="\n")
            path.chmod(0o755)

    def launch(**options):
        defaults = dict(
            TEST_SCRIPT=launcher.as_posix(),
            EXPECTED_GIT_HEAD="reviewed",
            WORKFLOW="learning",
            PARENT_CHECKPOINT=(root / "parent.pth").as_posix(),
            BASE_B1_CHECKPOINT=(root / "base.pth").as_posix(),
            BASE_B0_CHECKPOINT=(root / "b0.pth").as_posix(),
            IXI_MANIFEST=(root / "manifest.json").as_posix(),
            LPBA_MANIFEST=(root / "external.json").as_posix(),
            OASIS_ALL_ROOT=(root / "oas").as_posix(),
            SOURCE_RUN="source",
            BM_SOURCE_RUN="bm",
            IXI_RAW_ROOT="raw",
            GPUS="2 3 4 5",
            MIN_FREE_GIB="0",
            PRETESTS="1",
            PREP_THREADS="32",
            OMP_NUM_THREADS="1",
            OPENBLAS_NUM_THREADS="1",
            MKL_NUM_THREADS="1",
            LEARNING_PROFILE1_EXIT="0",
            LEARNING_FAIL_DIAGNOSTIC="0",
            LEARNING_FAIL_PREPARE="0",
            FAIL_STAGE="",
        )
        return run(timeout=90, **(defaults | options))

    return root, launch


def calls(root):
    return [json.loads(path.read_text()) for path in sorted((root / "learning_calls").glob("*.json"))]


@pytest.mark.parametrize("gpus", ("2 3 4 5", "0 1 2 3 4 5 6 7"))
def test_learning_shares_teachers_and_runs_the_full_matrix_on_selected_gpus(learning_dispatch, gpus):
    root, run = learning_dispatch
    result = run(GPUS=gpus)
    assert result.returncode == 0, result.stdout + result.stderr
    rows = calls(root)
    selected = gpus.split()
    trained = [r for r in rows if r["stage"] == "train"]
    assert {r["kind"]: r["gpu"] for r in trained} == dict(
        zip(("scratch_b0", "scratch_b1", "resampled_t1", "shuffled_t2"), selected[:4], strict=True)
    )
    teachers = [
        r
        for r in rows
        if r["module"] == "train_matching" and r["stage"] == "teacher" and "/smoke/" not in r["out"].replace("\\", "/")
    ]
    students = [r for r in rows if r["stage"] == "student" and "/smoke/" not in r["out"].replace("\\", "/")]
    assert sorted(r["source"] for r in teachers) == ["paired", "t1"]
    assert sorted((r["source"], r["mode"]) for r in students if r["width"] == "16") == [
        ("paired", "solver"),
        ("paired", "value"),
        ("t1", "solver"),
        ("t1", "value"),
    ]
    for row in students:
        family = row["source"]
        expected_slot = (4 if family == "t1" else 6) if len(selected) == 8 else (2 if family == "t1" else 3)
        expected_slot += int(len(selected) == 8 and row["mode"] == "solver" and row["width"] == "16")
        assert row["gpu"] == selected[expected_slot]
        assert (
            row["args"][row["args"].index("--teacher") + 1] == f"results/contrasts/unit/teachers/{family}/ckpt/last.pth"
        )
    evaluations = [r for r in rows if r["module"] == "eval_matching" and r["stage"] == "evaluate"]
    assert len([r for r in evaluations if "/smoke/" not in r["out"].replace("\\", "/")]) == 39
    assert {Path(r["out"]).name for r in evaluations if "/smoke/" in r["out"].replace("\\", "/")} == {
        "IXI",
        "LPBA",
        "OASIS",
        "half",
    }
    profiles = [r for r in rows if r["module"] == "train_matching" and r["stage"] == "profile"]
    assert profiles[0]["kind"] == "student" and profiles[0]["mode"] == "solver"
    assert {r["stride"] for r in rows if r["module"] == "train_matching"} == {"1"}
    assert len(students) == 5 and len([r for r in students if r["width"] == "8"]) == 1
    assert len([r for r in rows if r["module"] == "matching_diagnostics" and r["stage"] == "teacher"]) == 2
    assert list((root / "results/exports").glob("*__COMPLETE.zip"))


def test_learning_resume_rechecks_inputs_without_retraining_and_accepts_new_gpu_numbers(learning_dispatch):
    root, run = learning_dispatch
    result = run()
    assert result.returncode == 0, result.stdout + result.stderr
    before = calls(root)
    resumed = run(GPUS="4 5 6 7")
    assert resumed.returncode == 0, resumed.stdout + resumed.stderr
    new = calls(root)[len(before) :]
    assert [(r["module"], r["stage"]) for r in new] == [
        ("contrast_learning", "preflight"),
        ("contrast_learning", "prepare"),
        ("eval_matching", "report"),
    ]
    assert "SKIP scratch_b0 (completed)" in resumed.stdout
    assert "SKIP paired_solver (completed)" in resumed.stdout


def test_learning_memory_failure_holds_the_dispatch_without_changing_resolution(learning_dispatch):
    root, run = learning_dispatch
    result = run(LEARNING_PROFILE1_EXIT="86")
    assert result.returncode != 0
    rows = calls(root)
    profiles = [r for r in rows if r["stage"] == "profile" and r["module"] == "train_matching"]
    assert [(r["kind"], r["stride"]) for r in profiles] == [("student", "1")]
    assert not [r for r in rows if r["stage"] in ("teacher", "student", "train")]
    assert not (root / "results/contrasts/unit/stride.txt").exists()
    assert list((root / "results/exports").glob("*__FAILED.zip"))


def test_learning_real_profile_error_holds_the_dispatch_instead_of_reducing_resolution(learning_dispatch):
    root, run = learning_dispatch
    result = run(LEARNING_PROFILE1_EXIT="1")
    assert result.returncode != 0
    rows = calls(root)
    assert [(r["module"], r["stage"]) for r in rows] == [
        ("contrast_learning", "preflight"),
        ("train_matching", "profile"),
    ]
    assert rows[-1]["stride"] == "1"
    assert not (root / "results/contrasts/unit/stride.txt").exists()
    assert list((root / "results/exports").glob("*__FAILED.zip"))


def test_teacher_diagnostic_failure_is_reported_without_discarding_student_training(learning_dispatch):
    root, run = learning_dispatch
    result = run(LEARNING_FAIL_DIAGNOSTIC="1")
    assert result.returncode != 0
    students = [r for r in calls(root) if r["stage"] == "student" and "/smoke/" not in r["out"].replace("\\", "/")]
    assert len(students) == 5
    assert (root / "results/contrasts/unit/exit_codes/diagnostics.txt").read_text().strip() == "1"
    assert (root / "results/contrasts/unit/exit_codes/students.txt").read_text().strip() == "0"


@pytest.mark.parametrize("gpus", ("2 3 4 5", "0 1 2 3 4 5 6 7"))
def test_learning_auxiliary_preparation_failure_preserves_scratch_and_matching(learning_dispatch, gpus):
    root, run = learning_dispatch
    result = run(GPUS=gpus, LEARNING_FAIL_PREPARE="1")
    assert result.returncode != 0
    rows = calls(root)
    assert {r["kind"] for r in rows if r["stage"] == "train"} == {"scratch_b0", "scratch_b1"}
    students = [r for r in rows if r["stage"] == "student" and "/smoke/" not in r["out"].replace("\\", "/")]
    assert len(students) == 5
    for name in ("resampled_t1", "shuffled_t2"):
        assert (root / f"results/contrasts/unit/exit_codes/{name}.txt").read_text().strip() == "125"
    assert (root / "results/contrasts/unit/exit_codes/registrations.txt").read_text().strip() == "1"
    assert (root / "results/contrasts/unit/exit_codes/students.txt").read_text().strip() == "0"
    assert list((root / "results/exports").glob("*__FAILED.zip"))
