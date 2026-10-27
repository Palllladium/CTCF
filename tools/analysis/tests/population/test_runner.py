import json
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[4] / "tools/runners/train/population.sh"
BASH = shutil.which("bash") if os.name != "nt" else r"C:\Program Files\Git\bin\bash.exe"


@pytest.fixture
def runner(tmp_path):
    if not BASH or not Path(BASH).is_file():
        pytest.skip("bash is unavailable")
    data = tmp_path / "data"
    data.mkdir()
    package = tmp_path / "experiments/population"
    package.mkdir(parents=True)
    (package.parent / "__init__.py").touch()
    (package / "__init__.py").touch()
    certificate = tmp_path / "tools/analysis"
    certificate.mkdir(parents=True)
    (certificate.parent / "__init__.py").touch()
    (certificate / "__init__.py").touch()
    shutil.copyfile(SCRIPT.parents[2] / "analysis/run_artifacts.py", certificate / "run_artifacts.py")
    (certificate / "certify_psi.py").write_text(
        "import argparse, os\nfrom pathlib import Path\n"
        "p=argparse.ArgumentParser(); p.add_argument('--root'); p.add_argument('--out'); a=p.parse_args()\n"
        "assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''\n"
        "out=Path(a.out); out.mkdir(parents=True,exist_ok=True); (out/'report.json').write_text('{}')\n"
        "raise SystemExit(1 if os.environ.get('FAIL_TASK') == 'certificate' else 0)\n"
    )
    (package / "profile.py").write_text(
        """import argparse, json, os, subprocess, time
from pathlib import Path
p = argparse.ArgumentParser()
for key in ('task', 'run', 'data-root', 'device', 'parent-checkpoint'): p.add_argument('--'+key)
a = p.parse_args(); root = Path(a.run); root.mkdir(parents=True, exist_ok=True)
if os.environ.get('STOP_TASK') == a.task:
    subprocess.run([os.environ['RUNNER_TEST_BASH'], '-c', 'kill -TERM $(cat "$RUNNER_TEST_ROOT/runner.pid")'], check=True)
    time.sleep(15)
if a.task == 'prepare':
    config = {'data': a.data_root, 'width': os.environ.get('PROFILE_WIDTH', '4')}
    path = root/'config.json'
    if path.exists() and json.loads(path.read_text()) != config: raise SystemExit(3)
    path.write_text(json.dumps(config))
elif a.task == 'summary':
    (root/'summary.json').write_text('{}')
else:
    if a.device != 'cuda:0': raise SystemExit(4)
    out = root/a.task; out.mkdir(exist_ok=True)
    started = time.time(); time.sleep(.03)
    record = {'gpu': os.environ['CUDA_VISIBLE_DEVICES'], 'start': started, 'end': time.time()}
    (out/('call_'+str(time.time_ns())+'.json')).write_text(json.dumps(record))
    (out/'heavy.npy').write_bytes(b'payload')
    if os.environ.get('FAIL_TASK') == a.task: raise SystemExit(2)
    (out/'report.json').write_text(json.dumps(record))
""",
        encoding="utf-8",
    )
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    git = bin_dir / "git"
    git.write_text(
        '#!/usr/bin/env bash\ncase "$*" in\n'
        '"rev-parse --show-toplevel") echo "$RUNNER_TEST_ROOT" ;;\n'
        '"rev-parse HEAD") echo 0123456789abcdef ;;\n'
        '"diff --quiet HEAD --") exit 0 ;;\n*) exit 5 ;;\nesac\n',
        encoding="utf-8",
        newline="\n",
    )
    git.chmod(0o755)
    parent = tmp_path / "parent.pth"
    parent.write_bytes(b"parent")
    environment = os.environ | {
        "PARENT_CHECKPOINT": parent.as_posix(),
        "PRETESTS": "0",
        "PATH": str(bin_dir) + os.pathsep + os.environ["PATH"],
        "PYTHONPATH": str(tmp_path),
        "RUNNER_TEST_ROOT": tmp_path.as_posix(),
        "PYBIN": Path(sys.executable).as_posix(),
        "EXPECTED_GIT_HEAD": "0123456789abcdef",
        "OASIS_ALL_ROOT": data.as_posix(),
        "RUN_ID": "test",
        "GPUS": "2 3",
        "RUNNER_TEST_BIN": bin_dir.as_posix(),
        "RUNNER_TEST_SCRIPT": SCRIPT.as_posix(),
        "RUNNER_TEST_BASH": Path(BASH).as_posix(),
    }

    def run(**updates):
        prefix = '$(cygpath -u "$RUNNER_TEST_BIN")' if os.name == "nt" else "$RUNNER_TEST_BIN"
        command = (
            f'export PATH="{prefix}:$PATH"; echo $$ > "$RUNNER_TEST_ROOT/runner.pid"; exec bash "$RUNNER_TEST_SCRIPT"'
        )
        return subprocess.run(
            [BASH, "-c", command],
            cwd=tmp_path,
            env=environment | updates,
            capture_output=True,
            text=True,
            timeout=60,
        )

    return tmp_path, run


def test_runner_queues_without_sharing_gpu_and_resumes_on_new_ids(runner):
    root, run = runner
    first = run()
    assert first.returncode == 0, first.stdout + first.stderr
    assert "RUN_ID=test HEAD=0123456789abcdef GPUS=2 3" in first.stdout
    calls = list((root / "results/population/test").glob("*/call_*.json"))
    assert len(calls) == 10
    records = [json.loads(path.read_text()) for path in calls]
    assert {record["gpu"] for record in records} == {"2", "3"}
    for left in records:
        for right in records:
            if left is not right and left["gpu"] == right["gpu"]:
                assert left["end"] <= right["start"] or right["end"] <= left["start"]
    missing = root / "results/population/test/probe_train1/report.json"
    missing.unlink()
    assert (missing.parent.parent / "probe_train1.done").is_file()
    resumed = run(GPUS="0 1 2 3")
    assert resumed.returncode == 0, resumed.stdout + resumed.stderr
    resumed_calls = list((root / "results/population/test").glob("*/call_*.json"))
    assert len(resumed_calls) == 11
    assert sum(path.parent.name == "probe_train1" for path in resumed_calls) == 2
    with zipfile.ZipFile(sorted((root / "results/exports").glob("*COMPLETE.zip"))[-1]) as archive:
        assert any(name.endswith("/SHA256SUMS") for name in archive.namelist())
        assert not any(name.endswith(".npy") for name in archive.namelist())
        assert len({name.split("/")[0] for name in archive.namelist()}) == 1


def test_runner_packages_failure_and_retries_only_failed_task(runner):
    root, run = runner
    failed = run(FAIL_TASK="linear")
    assert failed.returncode != 0
    assert list((root / "results/exports").glob("*FAILED.zip"))
    assert len(list((root / "results/population/test").glob("*/call_*.json"))) == 10
    retried = run(GPUS="0 1 2 3")
    assert retried.returncode == 0, retried.stdout + retried.stderr
    calls = list((root / "results/population/test").glob("*/call_*.json"))
    assert len(calls) == 11
    assert sum(path.parent.name == "linear" for path in calls) == 2
    linear = [json.loads(path.read_text()) for path in calls if path.parent.name == "linear"]
    assert any(record["gpu"] in ("0", "1") for record in linear)


def test_certificate_failure_is_packaged_without_cancelling_gpu_tasks(runner):
    root, run = runner
    failed = run(FAIL_TASK="certificate")
    assert failed.returncode != 0
    assert len(list((root / "results/population/test").glob("*/call_*.json"))) == 10
    assert list((root / "results/exports").glob("*FAILED.zip"))
    resumed = run()
    assert resumed.returncode == 0
    assert "SKIP a0" in resumed.stdout and "PASS certificate" in resumed.stdout


@pytest.mark.parametrize("task", ("prepare", "a0"))
def test_stop_during_cpu_or_gpu_work_keeps_partial_delivery(runner, task):
    root, run = runner
    stopped = run(STOP_TASK=task)
    assert stopped.returncode == 143, stopped.stdout + stopped.stderr
    archive = next((root / "results/exports").glob("*__PARTIAL.zip"))
    with zipfile.ZipFile(archive) as handle:
        manifest = json.loads(handle.read(archive.stem + "/package_manifest.json"))
        assert manifest["status"] == "PARTIAL" and manifest["runner_exit_code"] == 143
    assert not (root / "results/population/test" / f"{task}.done").exists()
