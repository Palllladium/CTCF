import hashlib
import json
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[3] / "tools/runners/train/lumir.sh"
BASH = r"C:\Program Files\Git\bin\bash.exe" if os.name == "nt" else shutil.which("bash")


@pytest.fixture
def dispatch(tmp_path):
    if not BASH or not Path(BASH).is_file():
        pytest.skip("bash unavailable")
    package = tmp_path / "experiments"
    package.mkdir()
    (package / "__init__.py").touch()
    artifacts = tmp_path / "tools/analysis"
    artifacts.mkdir(parents=True)
    (artifacts.parent / "__init__.py").touch()
    (artifacts / "__init__.py").touch()
    shutil.copyfile(SCRIPT.parents[2] / "analysis/run_artifacts.py", artifacts / "run_artifacts.py")
    (artifacts / "check_lumir.py").write_text(
        "import json,sys,os\nfrom pathlib import Path\na=sys.argv[1:]\n"
        "Path(a[a.index('--out')+1]).write_text(json.dumps({'args':a,'gpus':os.environ.get('CUDA_VISIBLE_DEVICES')}))\n"
        "raise SystemExit(6 if os.environ.get('FAIL_STAGE') == 'preflight' else 0)\n"
    )
    (package / "train_CTCF.py").write_text(
        "import sys,json,os,subprocess,time\nfrom pathlib import Path\na=sys.argv[1:]; exp=a[a.index('--exp')+1]\n"
        "logs=Path('logs')/exp; logs.mkdir(parents=True,exist_ok=True); (logs/'args.json').write_text(json.dumps(a))\n"
        "if a[a.index('--save_ckpt')+1]=='1':\n"
        "    ckpt=Path('results')/exp/'ckpt'; ckpt.mkdir(parents=True,exist_ok=True); (ckpt/'last.pth').write_text('weights')\n"
        "print('[perf 000] example')\n"
        "print('[throughput] pairs_per_second=2.5')\n"
        "if os.environ.get('FAIL_STAGE') == 'interrupt':\n"
        "    subprocess.run([os.environ['TEST_BASH'], '-c', 'kill -TERM $(cat \"$TEST_ROOT/runner.pid\")'], check=True)\n"
        "    time.sleep(15)\n"
        "raise SystemExit(7 if os.environ.get('FAIL_STAGE') == 'train' else 0)\n"
    )
    (package / "export_lumir.py").write_text(
        "import sys,json,os\nfrom pathlib import Path\na=sys.argv[1:]; out=Path(a[a.index('--out')+1]); out.mkdir(parents=True)\n"
        "(out/'args.json').write_text(json.dumps(a))\n"
        "(out/'env.json').write_text(json.dumps({'gpus':os.environ.get('CUDA_VISIBLE_DEVICES')}))\n"
        "(out/'heavy.nii.gz').write_bytes(b'field'); (out/'submission.zip').write_bytes(b'payload')\n"
        "raise SystemExit(9 if os.environ.get('FAIL_STAGE') == 'export' else 0)\n"
    )
    binary = tmp_path / "bin"
    binary.mkdir()
    (binary / "git").write_text(
        '#!/usr/bin/env bash\ncase "$*" in\n'
        '"rev-parse --show-toplevel") echo "$TEST_ROOT";;\n'
        '"rev-parse HEAD") echo reviewed;;\n"diff --quiet HEAD --") exit 0;;\n*) exit 9;;\nesac\n',
        newline="\n",
    )
    (binary / "python_stub").write_text(
        '#!/usr/bin/env bash\nif [[ "$*" == "-m pytest "* ]]; then\n'
        '[ "${FAIL_STAGE:-}" != pretests ] || exit 5\nexit 0\nfi\n'
        'if [[ "$*" == "-m torch.distributed.run "* ]]; then\n'
        'printf "%s\\n" "$@" > "$TEST_ROOT/torchrun_args.txt"\nshift 6\nexec "$REALPY" -m "$@"\nfi\n'
        'exec "$REALPY" "$@"\n',
        newline="\n",
    )
    for path in binary.iterdir():
        path.chmod(0o755)
    env = os.environ | dict(
        TEST_ROOT=tmp_path.as_posix(),
        TEST_BIN=binary.as_posix(),
        TEST_SCRIPT=SCRIPT.as_posix(),
        TEST_BASH=Path(BASH).as_posix(),
        REALPY=Path(sys.executable).as_posix(),
        PYBIN=(binary / "python_stub").as_posix(),
        PYTHONPATH=tmp_path.as_posix(),
        EXPECTED_GIT_HEAD="reviewed",
        PRECISION="bf16",
        GLOBAL_BATCH="1",
        ITERS="3",
        EPOCHS="2",
        PAIRS="lumir25",
        EXP="test",
    )

    def run(**updates):
        prefix = '$(cygpath -u "$TEST_BIN")' if os.name == "nt" else "$TEST_BIN"
        return subprocess.run(
            [BASH, "-c", f'export PATH="{prefix}:$PATH"; echo $$ > "$TEST_ROOT/runner.pid"; exec bash "$TEST_SCRIPT"'],
            cwd=tmp_path,
            env=env | updates,
            capture_output=True,
            text=True,
            timeout=35,
        )

    return tmp_path, run


@pytest.mark.parametrize("config", ("CTCF-CascadeA-Mamba", "CTCF-CascadeA", "CTCF-VM-solo"))
def test_profile_activates_selected_configured_stages_and_preserves_precision(dispatch, config):
    root, run = dispatch
    result = run(PROFILE="1", CONFIG=config, MODEL="", GPU="3")
    assert result.returncode == 0, result.stderr + result.stdout
    args = json.loads(next((root / "logs").glob("*/args.json")).read_text())
    assert args[args.index("--config") + 1] == config
    assert args[args.index("--precision") + 1] == "bf16"
    assert args[args.index("--finetune") + 1] == "1"
    assert args[args.index("--max_train_iters") + 1] == "205"
    assert args[args.index("--perf_warmup_updates") + 1] == "5"
    assert args[args.index("--save_ckpt") + 1] == "0"
    assert not list((root / "results").rglob("*.pth"))
    assert "[throughput] pairs_per_second=2.5" in result.stdout
    assert "--l3_svf" not in args


def test_training_and_export_use_the_same_explicit_numerical_and_model_recipe(dispatch):
    root, run = dispatch
    result = run(PRECISION="tf32", MODEL="--l3_svf 0\n--disable_l1 1", STAGES="ramp", TIME_STEPS="4")
    assert result.returncode == 0, result.stderr + result.stdout
    train = json.loads((root / "logs/test/args.json").read_text())
    export = json.loads(next((root / "results/lumir").rglob("export_lumir25/args.json")).read_text())
    for args in (train, export):
        assert args[args.index("--precision") + 1] == "tf32"
        assert args[args.index("--time_steps") + 1] == "4"
        assert args[args.index("--l3_svf") + 1] == "0"
        assert args[args.index("--disable_l1") + 1] == "1"
    assert train[train.index("--finetune") + 1] == "0"


def test_resume_allows_new_gpu_but_rejects_a_changed_numerical_recipe(dispatch):
    _root, run = dispatch
    assert run().returncode == 0
    assert run(RESUME="1", GPU="2").returncode == 0
    changed = run(RESUME="1", PRECISION="fp32")
    assert changed.returncode != 0 and "resume recipe differs" in changed.stdout


def test_training_only_architecture_override_is_rejected_before_work_starts(dispatch):
    root, run = dispatch
    result = run(TRAIN="--l3_iters 2")
    assert result.returncode != 0 and "architecture overrides in MODEL" in result.stdout
    assert not (root / "logs").exists()


@pytest.mark.parametrize(
    "stage,code", (("pretests", 5), ("preflight", 6), ("train", 7), ("export", 9), ("interrupt", 143), ("", 0))
)
def test_delivery_on_success_failure_and_stop(dispatch, stage, code):
    root, run = dispatch
    result = run(FAIL_STAGE=stage)
    assert result.returncode == code, result.stderr + result.stdout
    status = "COMPLETE" if code == 0 else "PARTIAL" if code == 143 else "FAILED"
    archive = next((root / "results/exports").glob(f"*__{status}.zip"))
    checksum = Path(str(archive) + ".sha256").read_text().split()[0]
    assert hashlib.sha256(archive.read_bytes()).hexdigest() == checksum
    with zipfile.ZipFile(archive) as handle:
        prefix = archive.stem + "/"
        assert all(name.startswith(prefix) for name in handle.namelist())
        manifest = json.loads(handle.read(prefix + "package_manifest.json"))
        assert manifest["status"] == status and manifest["runner_exit_code"] == code
        assert not any(name.endswith((".pth", ".nii.gz", ".zip")) for name in handle.namelist())
        for line in handle.read(prefix + "SHA256SUMS").decode().splitlines():
            digest, name = line.split("  ", 1)
            assert hashlib.sha256(handle.read(prefix + name)).hexdigest() == digest


def test_ddp_uses_selected_devices_and_exports_on_only_one(dispatch):
    root, run = dispatch
    result = run(GPUS="2,3", GLOBAL_BATCH="4")
    assert result.returncode == 0, result.stdout + result.stderr
    assert "--nproc-per-node=2" in (root / "torchrun_args.txt").read_text()
    args = json.loads((root / "logs/test/args.json").read_text())
    assert args[args.index("--global_batch_size") + 1] == "4"
    preflight = json.loads(next((root / "results/lumir").rglob("preflight.json")).read_text())
    assert preflight["gpus"] == "2,3"
    exported = json.loads(next((root / "results/lumir").rglob("export_lumir25/env.json")).read_text())
    assert exported["gpus"] == "2"


def test_invalid_global_batch_is_rejected_before_work(dispatch):
    root, run = dispatch
    result = run(GPUS="0 1 2", GLOBAL_BATCH="4")
    assert result.returncode != 0 and "divisible" in result.stdout
    assert not (root / "logs").exists()


def test_full_training_requires_an_explicit_update_budget(dispatch):
    root, run = dispatch
    result = run(ITERS="")
    assert result.returncode != 0 and "Set ITERS" in result.stderr
    assert not (root / "logs").exists()


def test_skipped_pretests_are_not_reported_as_passed(dispatch):
    root, run = dispatch
    result = run(PRETESTS="0", FAIL_STAGE="pretests")
    assert result.returncode == 0, result.stdout + result.stderr
    manifest = json.loads(next((root / "results/lumir").rglob("package_manifest.json")).read_text())
    assert manifest["status"] == "COMPLETE" and "pretests" not in manifest["exit_codes"]
    assert "SKIPPED pretests" in result.stdout
    assert "SKIPPED" in next((root / "results/lumir").rglob("pretests.log")).read_text()
