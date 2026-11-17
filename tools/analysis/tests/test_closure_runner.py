from pathlib import Path

import pytest

from tools.analysis.tests.test_learning_runner import calls, learning_dispatch as learning_dispatch
from tools.analysis.tests.test_lumir_runner import dispatch as dispatch


@pytest.mark.parametrize("gpus", ("0 1 2 3", "0 1 2 3 4 5 6 7"))
def test_closure_reuses_old_weights_runs_seven_fixed_controls_and_is_resumable(learning_dispatch, gpus):
    root, run = learning_dispatch
    result = run(WORKFLOW="closure", GPUS=gpus)
    assert result.returncode == 0, result.stdout + result.stderr
    rows = calls(root)
    trained = [r for r in rows if r["module"] == "contrast_learning" and r["stage"] == "train"]
    assert sorted((r["kind"], r["args"][r["args"].index("--seed") + 1]) for r in trained) == [
        ("b0", "0"),
        ("b0", "1"),
        ("b1", "0"),
        ("b1", "1"),
        ("normalised_t1", "0"),
        ("resampled_t1", "0"),
        ("shuffled_t2", "0"),
    ]
    for row in trained:
        assert row["args"][row["args"].index("--epochs") + 1] == "10"
        assert row["args"][row["args"].index("--ncc_mode") + 1] == "stable"
        assert row["gpu"] in gpus.split()
    assert not [r for r in rows if r["stage"] in ("teacher", "student")]
    evaluations = [r for r in rows if r["module"] == "eval_matching" and r["stage"] == "evaluate"]
    assert len(evaluations) == 22
    assert len([r for r in evaluations if "--raw_from" in r["args"]]) == 7
    assert all("--budget_hours" not in r["args"] for r in rows)
    before = len(rows)
    resumed = run(WORKFLOW="closure", GPUS="4 5 6 7")
    assert resumed.returncode == 0, resumed.stdout + resumed.stderr
    new = calls(root)[before:]
    assert not [r for r in new if r["stage"] in ("train", "teachers", "regional", "solvers")]
    # Changing 8 -> 4 GPUs changes the completion partition; case caches handle reuse.
    assert len([r for r in new if r["stage"] == "complete"]) == (2 if len(gpus.split()) == 8 else 0)
    assert list((root / "results/exports").glob("*__COMPLETE.zip"))


def test_numerical_failure_holds_only_new_training_not_the_frozen_readout(learning_dispatch):
    root, run = learning_dispatch
    result = run(WORKFLOW="closure", CLOSURE_FAIL_STAGE="inputs")
    assert result.returncode != 0
    rows = calls(root)
    assert not [r for r in rows if r["stage"] == "train"]
    assert [r for r in rows if r["stage"] == "complete"]
    assert [r for r in rows if r["stage"] == "teachers"]
    assert [r for r in rows if r["stage"] == "certificates"]
    assert Path(root / "results/contrasts/unit/exit_codes/stable.txt").read_text().strip() == "125"
    assert list((root / "results/exports").glob("*__FAILED.zip"))


def test_certificate_comparison_failure_keeps_completed_training_and_raw_evaluation(learning_dispatch):
    root, run = learning_dispatch
    result = run(WORKFLOW="closure", CLOSURE_FAIL_STAGE="certificates")
    assert result.returncode != 0
    rows = calls(root)
    assert len([r for r in rows if r["module"] == "contrast_learning" and r["stage"] == "train"]) == 7
    assert [r for r in rows if r["stage"] == "summary"]
    assert Path(root / "results/contrasts/unit/exit_codes/stable.txt").read_text().strip() == "0"
