import json
import os
import shutil
import time
from pathlib import Path

import pytest

from tools.analysis.tests.test_lumir_runner import dispatch as dispatch

SCRIPT = Path(__file__).resolve().parents[3] / "tools/runners/train/contrasts.sh"


@pytest.fixture
def contrast_dispatch(dispatch):
    root, run = dispatch
    (root / "results/contrasts/unit").mkdir(parents=True)
    destination = root / "tools/runners/train"
    destination.mkdir(parents=True)
    shutil.copyfile(SCRIPT, destination / SCRIPT.name)
    launcher = root / "launch.sh"
    launcher.write_text(
        'exec bash "$TEST_ROOT/tools/runners/train/contrasts.sh" "${TEST_MODE:-_work}" unit\n', newline="\n"
    )
    parent, manifest, data = root / "parent.pth", root / "manifest.json", root / "oas"
    parent.write_bytes(b"parent")
    manifest.write_text("{}")
    data.mkdir()
    (root / "experiments/dispatch1").mkdir()
    (root / "experiments/dispatch1/__init__.py").touch()
    (root / "experiments/dispatch1/eval_consensus.py").write_text(
        "import sys,os,time\nfrom pathlib import Path\na=sys.argv; out=Path(a[a.index('--out')+1]); out.mkdir(parents=True,exist_ok=True); (out/'report.json').write_text('{}')\n"
        "if os.environ.get('CONSENSUS_WAIT_PARENT'):\n"
        "    deadline=time.monotonic()+10\n"
        "    while not (out.parent/'parent/evaluate.json').exists():\n"
        "        if time.monotonic()>deadline: raise SystemExit('parent evaluation waited for consensus')\n"
        "        time.sleep(0.05)\n"
        "if os.environ.get('SLOW_CONSENSUS'): time.sleep(20)\n"
    )
    (root / "experiments/contrasts.py").write_text(
        "import sys,os\nfrom pathlib import Path\na=sys.argv; stage=a[1]; out=Path(a[a.index('--out')+1]); out.mkdir(parents=True,exist_ok=True)\n"
        "with (Path(os.environ['TEST_ROOT'])/'calls.txt').open('a') as f: f.write(stage+' '+(a[a.index('--arm')+1] if '--arm' in a else 'parent')+' '+str(os.environ.get('CUDA_VISIBLE_DEVICES'))+'\\n')\n"
        "if stage=='train':\n"
        "    exp=a[a.index('--exp')+1]; folder=Path('logs')/exp; folder.mkdir(parents=True,exist_ok=True); (folder/'args.json').write_text(__import__('json').dumps(a))\n"
        "    weights=Path('results')/exp/'ckpt'; weights.mkdir(parents=True,exist_ok=True); (weights/'last.pth').write_text('weights')\n"
        "(out/(stage+'.json')).write_text('{}')\n"
        "raise SystemExit(8 if stage=='check' and os.environ.get('FAIL_CHECK') else 0)\n"
    )
    if os.name == "nt":
        for name, content in (("flock", "exit 0"), ("ps", "exit 0")):
            path = root / "bin" / name
            path.write_text(f"#!/usr/bin/env bash\n{content}\n", newline="\n")
            path.chmod(0o755)

    def launch(**options):
        return run(
            **(
                dict(
                    TEST_SCRIPT=launcher.as_posix(),
                    PARENT_CHECKPOINT=parent.as_posix(),
                    IXI_MANIFEST=manifest.as_posix(),
                    OASIS_ALL_ROOT=data.as_posix(),
                    GPUS="2 3 4 5",
                    MIN_FREE_GIB="0",
                    RUN_B="1",
                    WORKFLOW="main",
                )
                | options
            )
        )

    return root, launch


def test_dispatch_queues_matched_arms_and_can_resume_without_retraining(contrast_dispatch):
    root, run = contrast_dispatch
    result = run(CONSENSUS_WAIT_PARENT="1")
    assert result.returncode == 0, result.stdout + result.stderr
    args = [json.loads((root / f"logs/unit_{arm}/args.json").read_text()) for arm in ("b0", "b1", "b2")]
    for arm, values in zip(("b0", "b1", "b2"), args, strict=True):
        assert values[1] == "train" and values[values.index("--arm") + 1] == arm
        assert "--checkpoint" in values
    calls = (root / "calls.txt").read_text().splitlines()
    assert max(i for i, row in enumerate(calls) if row.startswith("train ")) < next(
        i for i, row in enumerate(calls) if row.startswith("evaluate parent ")
    )
    assert len(list((root / "results/exports").glob("*COMPLETE.zip"))) == 1
    resumed = run(GPUS="6")
    assert resumed.returncode == 0 and "SKIP b0" in resumed.stdout, resumed.stdout + resumed.stderr


@pytest.mark.parametrize("failure", ("missing_data", "readiness"))
def test_b_hold_does_not_cancel_consensus(contrast_dispatch, failure):
    root, run = contrast_dispatch
    options = {"IXI_MANIFEST": "missing.json"} if failure == "missing_data" else {"FAIL_CHECK": "1"}
    result = run(**options)
    assert result.returncode != 0
    assert (root / "results/contrasts/unit/consensus/report.json").is_file()
    assert not (root / "logs/unit_b0").exists()
    assert list((root / "results/exports").glob("*FAILED.zip"))


def test_changed_parent_and_duplicate_gpu_are_rejected(contrast_dispatch):
    root, run = contrast_dispatch
    assert run(GPUS="2 2").returncode != 0
    assert run().returncode == 0
    (root / "parent.pth").write_bytes(b"another parent")
    changed = run()
    assert changed.returncode != 0 and "Changed parent" in changed.stdout


@pytest.fixture
def mechanism_dispatch(contrast_dispatch):
    root, run = contrast_dispatch
    raw = root / "raw.pth"
    raw.write_text("raw reference")
    shutil.copyfile(SCRIPT.with_name("lumir.sh"), root / "tools/runners/train/lumir.sh")
    (root / "experiments/contrast_probes.py").write_text(
        "import sys,json,os\nfrom pathlib import Path\na=sys.argv;out=Path(a[a.index('--out')+1]);out.mkdir(parents=True,exist_ok=True)\n"
        "with Path('probe_calls.txt').open('a') as f:f.write(a[1]+' '+os.environ.get('CUDA_VISIBLE_DEVICES','')+'\\n')\n"
        "(out/(a[1]+'.json')).write_text('{}')\n"
    )
    (root / "experiments/contrast_diagnostics.py").write_text(
        "import sys,json\nfrom pathlib import Path\na=sys.argv;exp=a[a.index('--exp')+1]\n"
        "out=Path(a[a.index('--out')+1]);out.mkdir(parents=True,exist_ok=True);(out/'args.json').write_text(json.dumps(a))\n"
        "folder=Path('results')/exp/'ckpt';folder.mkdir(parents=True,exist_ok=True);(folder/'last.pth').write_text('weights')\n"
    )
    (root / "experiments/repair_diagnostics.py").write_text(
        "import sys,json\nfrom pathlib import Path\na=sys.argv;out=Path(a[a.index('--arm-dir')+1]);e=a[a.index('--epoch')+1]\n"
        "(out/('repair_epoch'+e+'.json')).write_text('{}')\n"
    )
    (root / "experiments/lumir_probes.py").write_text(
        "import sys,os\nfrom pathlib import Path\na=sys.argv;out=Path(a[a.index('--out')+1]);out.mkdir(parents=True,exist_ok=True)\n"
        "(out/'report.json').write_text('{}')\n"
        "raise SystemExit(1 if a[1]=='check' and os.environ.get('FAIL_MIND_CHECK') else 0)\n"
    )

    def launch(**kwargs):
        return run(
            **(
                dict(
                    WORKFLOW="mechanism",
                    GPUS="0 1 2 3",
                    SOURCE_RUN="source",
                    ORIGINAL_RUN="original",
                    LUMIR_RAW_CHECKPOINT=raw.as_posix(),
                )
                | kwargs
            )
        )

    return root, launch


def test_mechanism_dispatch_has_four_fixed_trainings_and_resumes_on_other_gpus(mechanism_dispatch):
    root, run = mechanism_dispatch
    result = run()
    assert result.returncode == 0, result.stdout + result.stderr
    run_root = root / "results/contrasts/unit"
    for name, epochs, weights in (
        ("half_t1", "3", ("0.5", "0")),
        ("full_t1_t2", "3", ("1", "0.5")),
        ("b0_s1", "10", ("1", "0")),
        ("b1_s1", "10", ("0.5", "0.5")),
    ):
        args = json.loads((run_root / name / "args.json").read_text())
        assert args[args.index("--epochs") + 1] == epochs
        assert tuple(args[args.index(flag) + 1] for flag in ("--t1_weight", "--t2_weight")) == weights
        assert (run_root / name / "repair_epoch3.json").is_file()
        assert (run_root / name / "repair_epoch10.json").is_file() == (epochs == "10")
    train = json.loads((root / "logs/unit_mind/args.json").read_text())
    assert train[train.index("--max_epoch") + 1] == "40" and train[train.index("--global_batch_size") + 1] == "4"
    assert train[train.index("--input_representation") + 1] == "mind6" and "--init_ckpt" not in train
    assert (root / "results/lumir/unit_mind/certified_lumir25/args.json").is_file()
    resumed = run(GPUS="4 5 6 7")
    assert resumed.returncode == 0 and "SKIP b1_s1" in resumed.stdout and "SKIP lumir_train" in resumed.stdout


def test_failed_mind_check_keeps_b_results_and_holds_only_lumir(mechanism_dispatch):
    root, run = mechanism_dispatch
    result = run(FAIL_MIND_CHECK="1")
    assert result.returncode != 0
    assert (root / "results/contrasts/unit/b1_s1/repair_epoch10.json").is_file()
    assert (root / "results/contrasts/unit/summary.json").is_file()
    assert not (root / "results/unit_mind").exists()
    assert list((root / "results/exports").glob("*FAILED.zip"))


def test_eight_gpu_dispatch_keeps_lumir_on_its_own_four_devices(mechanism_dispatch):
    root, run = mechanism_dispatch
    result = run(GPUS="0 1 2 3 4 5 6 7")
    assert result.returncode == 0, result.stdout + result.stderr
    export = next((root / "results/lumir/unit_mind/attempts").glob("*/export_lumir25/env.json"))
    assert json.loads(export.read_text())["gpus"] == "4"
    args = (root / "torchrun_args.txt").read_text()
    assert "--nproc-per-node=4" in args


def test_mechanism_resume_refreshes_summary_after_a_failed_dependency_recovers(mechanism_dispatch):
    root, run = mechanism_dispatch
    (root / "experiments/contrast_probes.py").write_text(
        "import sys,json,os\nfrom pathlib import Path\na=sys.argv;out=Path(a[a.index('--out')+1]);out.mkdir(parents=True,exist_ok=True)\n"
        "if a[1]=='smooth' and os.environ.get('FAIL_MECHANISM_SMOOTH'):raise SystemExit(9)\n"
        "payload={'smooth_available':(out/'smooth.json').exists()} if a[1]=='summary' else {}\n"
        "(out/(a[1]+'.json')).write_text(json.dumps(payload))\n"
    )
    first = run(FAIL_MECHANISM_SMOOTH="1")
    report = root / "results/contrasts/unit/summary.json"
    assert first.returncode != 0 and json.loads(report.read_text())["smooth_available"] is False
    resumed = run()
    assert resumed.returncode == 0, resumed.stdout + resumed.stderr
    assert json.loads(report.read_text())["smooth_available"] is True
    assert "SKIP b0_s1" in resumed.stdout and "SKIP summary" not in resumed.stdout


@pytest.fixture
def followup_dispatch(contrast_dispatch):
    root, run = contrast_dispatch
    source = root / "source"
    (source / "parent").mkdir(parents=True)
    (source / "head.txt").write_text("source")
    (source / "parent/recipe.json").write_text("{}")
    (root / "experiments/contrast_diagnostics.py").write_text(
        "import sys,os,json\nfrom pathlib import Path\n"
        "a=sys.argv; stage=a[1]; out=Path(a[a.index('--out')+1]); out.mkdir(parents=True,exist_ok=True)\n"
        "arm=a[a.index('--arm')+1] if '--arm' in a else ''; reg=a[a.index('--w_reg')+1] if '--w_reg' in a else ''\n"
        "with (Path(os.environ['TEST_ROOT'])/'followup_calls.txt').open('a') as f: f.write(stage+' '+arm+' '+reg+' '+os.environ.get('CUDA_VISIBLE_DEVICES','')+'\\n')\n"
        "if stage=='train':\n"
        "    exp=a[a.index('--exp')+1]; folder=Path('logs')/exp;folder.mkdir(parents=True,exist_ok=True);(folder/'metrics.csv').write_text('data')\n"
        "if stage=='anchor': (out/'anchor.json').write_text(json.dumps({'add_b1_w4':bool(os.environ.get('DRIFT_ANCHOR'))}))\n"
        "if stage=='decide': (out/'decision.json').write_text(json.dumps({'repeat_arm':'b1' if os.environ.get('SIGNAL') else None}))\n"
    )
    (root / "experiments/repair_diagnostics.py").write_text(
        "import sys,os\nfrom pathlib import Path\na=sys.argv;stage=a[1]\n"
        "with (Path(os.environ['TEST_ROOT'])/'followup_calls.txt').open('a') as f:f.write('repair_'+stage+' '+os.environ.get('CUDA_VISIBLE_DEVICES','')+'\\n')\n"
        "raise SystemExit(9 if stage=='saved' and os.environ.get('FAIL_REPAIR') else 0)\n"
    )

    def launch(**options):
        return run(WORKFLOW="followup", SOURCE_RUN=source.as_posix(), **options)

    return root, launch


def test_followup_runs_four_controls_and_reuses_completed_jobs(followup_dispatch):
    root, run = followup_dispatch
    result = run()
    assert result.returncode == 0, result.stdout + result.stderr
    calls = (root / "followup_calls.txt").read_text().splitlines()
    trains = [line for line in calls if line.startswith("train ")]
    assert sorted(line.split()[1:3] for line in trains) == [["b0", "1"], ["b0", "4"], ["b1", "1"], ["b2", "1"]]
    assert all(line.split()[-1] != "5" for line in trains)
    assert "repair_saved 5" in calls
    count = len(calls)
    resumed = run(GPUS="6 7")
    assert resumed.returncode == 0 and "SKIP b0_w4_s0" in resumed.stdout
    after = (root / "followup_calls.txt").read_text().splitlines()
    assert len(after) == count + 1 and after[-1].startswith("prepare ")


@pytest.mark.parametrize("changed", ("manifest", "source"))
def test_followup_rejects_changed_inputs_even_when_all_jobs_are_cached(followup_dispatch, changed):
    root, run = followup_dispatch
    assert run().returncode == 0
    calls = (root / "followup_calls.txt").read_text()
    path = root / ("manifest.json" if changed == "manifest" else "source/parent/recipe.json")
    path.write_text("changed")
    refused = run()
    assert refused.returncode != 0 and f"Changed {changed}" in refused.stdout
    assert (root / "followup_calls.txt").read_text() == calls


@pytest.mark.parametrize("condition", ("DRIFT_ANCHOR", "SIGNAL", "FAIL_REPAIR"))
def test_followup_conditions_are_independent_and_visible(followup_dispatch, condition):
    root, run = followup_dispatch
    result = run(**{condition: "1"})
    calls = (root / "followup_calls.txt").read_text().splitlines()
    trains = [line.split()[1:3] for line in calls if line.startswith("train ")]
    if condition == "DRIFT_ANCHOR":
        assert result.returncode == 0 and ["b1", "4"] in trains and len(trains) == 5
    elif condition == "SIGNAL":
        assert result.returncode == 0 and len(trains) == 6
    else:
        assert result.returncode != 0 and len(trains) == 4
        assert list((root / "results/exports").glob("*FAILED.zip"))


@pytest.mark.skipif(os.name == "nt", reason="real Linux flock, setsid and /proc process-group ownership")
def test_linux_launcher_lock_and_stop_own_their_process_group(contrast_dispatch):
    root, run = contrast_dispatch
    run_root = root / "results/contrasts/unit"
    (run_root / "head.txt").write_text("reviewed\n")
    try:
        started = run(TEST_MODE="resume", RUN_B="0", SLOW_CONSENSUS="1")
        assert started.returncode == 0, started.stdout + started.stderr
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            if (run_root / "status.txt").exists() and "START consensus" in (run_root / "status.txt").read_text():
                break
            time.sleep(0.1)
        pid = (run_root / "runner.pid").read_text()
        assert run(TEST_MODE="resume", RUN_B="0").returncode == 0
        time.sleep(0.2)
        assert (run_root / "runner.pid").read_text() == pid
        assert run(TEST_MODE="stop").returncode == 0
        deadline = time.monotonic() + 10
        while (run_root / "runner.pid").exists() and time.monotonic() < deadline:
            time.sleep(0.1)
        assert not (run_root / "runner.pid").exists()
        assert list((root / "results/exports").glob("*PARTIAL.zip"))
    finally:
        if (run_root / "runner.pid").exists():
            run(TEST_MODE="stop")
