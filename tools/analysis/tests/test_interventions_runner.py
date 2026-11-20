import shlex
import shutil
import sys
from pathlib import Path
from textwrap import dedent
from types import SimpleNamespace

import pytest

from tools.analysis.tests.test_learning_runner import calls, learning_dispatch as learning_dispatch
from tools.analysis.tests.test_lumir_runner import dispatch as dispatch


def test_review_disk_reserve_credits_existing_run_bytes(tmp_path, monkeypatch):
    script = Path("tools/runners/train/contrasts.sh").read_text()
    line = next(line for line in script.splitlines() if "shutil.disk_usage" in line)
    code = shlex.split(line)[2]
    monkeypatch.setattr(shutil, "disk_usage", lambda _: SimpleNamespace(free=1024**3 - 8))
    run = tmp_path / "run"
    run.mkdir()
    (run / "field.npz").write_bytes(b"12345678")
    for credit_root, expected in (("", True), (str(run), False)):
        monkeypatch.setattr(sys, "argv", ["-c", "1", credit_root])
        with pytest.raises(SystemExit) as stopped:
            exec(code, {})
        assert stopped.value.code is expected


@pytest.fixture
def interventions_dispatch(learning_dispatch):
    root, run = learning_dispatch
    stub = dedent("""\
        import json, os, sys, time
        from pathlib import Path
        args = sys.argv[1:]
        stage, module = args[0], Path(__file__).stem
        def option(name, default=""):
            return args[args.index(name)+1] if name in args else default
        root = Path(os.environ["TEST_ROOT"])
        out = Path(option("--out"))
        out.mkdir(parents=True, exist_ok=True)
        calls = root / "learning_calls"
        calls.mkdir(exist_ok=True)
        row = dict(module=module, stage=stage, args=args, gpu=os.environ.get("CUDA_VISIBLE_DEVICES"))
        (calls / f"{time.monotonic_ns()}_{os.getpid()}.json").write_text(json.dumps(row))
        if stage == os.environ.get("INTERVENTION_FAIL"):
            raise SystemExit(9)
        if stage == "prepare":
            (out / "inputs.json").write_text("{}")
        if stage == "select":
            (out / "selection.json").write_text("{}")
        if stage == "final":
            assert (out / "selection.json").is_file()
            if option("--supervised_checkpoint"):
                assert Path(option("--supervised_checkpoint")).is_file()
        if module == "contrast_supervision" and stage == "train":
            folder = Path("results") / option("--exp") / "ckpt"
            folder.mkdir(parents=True, exist_ok=True)
            (folder / "last.pth").write_text("weights")
        """)
    for name in ("weight_interventions", "contrast_supervision"):
        (root / f"experiments/{name}.py").write_text(stub, newline="\n")
    return root, lambda **kwargs: run(WORKFLOW="interventions", **kwargs)


def test_dispatch_has_one_training_and_separates_selection_from_final_readout(interventions_dispatch):
    root, run = interventions_dispatch
    result = run()
    assert result.returncode == 0, result.stdout + result.stderr
    rows = calls(root)
    assert [(r["module"], r["stage"]) for r in rows if r["stage"] == "train"] == [("contrast_supervision", "train")]
    assert {r["gpu"] for r in rows if r["stage"] == "collect"} == {"2", "3"}
    assert {r["gpu"] for r in rows if r["module"] == "contrast_supervision"} == {"5"}
    assert len([r for r in rows if r["stage"] == "final"]) == 12
    select_index = next(i for i, r in enumerate(rows) if r["stage"] == "select")
    assert all(i > select_index for i, r in enumerate(rows) if r["stage"] == "final")
    assert not any("--budget_hours" in r["args"] for r in rows)
    assert list((root / "results/exports").glob("*__COMPLETE.zip"))
    before = len(rows)
    resumed = run(GPUS="4 5 6 7")
    assert resumed.returncode == 0, resumed.stdout + resumed.stderr
    assert [(r["module"], r["stage"]) for r in calls(root)[before:]] == [
        ("weight_interventions", "prepare"),
        ("weight_interventions", "summary"),
    ]


def test_failed_supervised_profile_keeps_weight_and_field_diagnostics(interventions_dispatch):
    root, run = interventions_dispatch
    result = run(INTERVENTION_FAIL="profile")
    assert result.returncode != 0
    rows = calls(root)
    assert not [r for r in rows if r["stage"] == "train"]
    assert [r for r in rows if r["stage"] == "fields"]
    finals = [r for r in rows if r["stage"] == "final"]
    assert len(finals) == 12 and all("--supervised_checkpoint" not in r["args"] for r in finals)
    assert (root / "results/contrasts/unit/exit_codes/supervision.txt").read_text().strip() == "125"
    assert list((root / "results/exports").glob("*__FAILED.zip"))


def test_failed_selection_never_opens_the_sealed_readout(interventions_dispatch):
    root, run = interventions_dispatch
    result = run(INTERVENTION_FAIL="selection")
    assert result.returncode != 0
    assert not [r for r in calls(root) if r["stage"] == "final"]
    assert [r for r in calls(root) if r["stage"] == "fields"]
    assert (root / "results/contrasts/unit/exit_codes/final.txt").read_text().strip() == "125"


def test_resume_adds_supervised_final_readout_after_earlier_supervision_failure(interventions_dispatch):
    root, run = interventions_dispatch
    assert run(INTERVENTION_FAIL="profile").returncode != 0
    before = len(calls(root))
    result = run(GPUS="4 5 6 7")
    assert result.returncode == 0, result.stdout + result.stderr
    finals = [r for r in calls(root)[before:] if r["stage"] == "final"]
    assert len(finals) == 12 and all("--supervised_checkpoint" in r["args"] for r in finals)
