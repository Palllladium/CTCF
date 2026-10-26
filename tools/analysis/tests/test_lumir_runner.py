import json
import os
import shutil
import subprocess
import sys
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
    (package / "train_CTCF.py").write_text(
        "import sys,json\nfrom pathlib import Path\na=sys.argv[1:]; exp=a[a.index('--exp')+1]\n"
        "logs=Path('logs')/exp; logs.mkdir(parents=True,exist_ok=True); (logs/'args.json').write_text(json.dumps(a))\n"
        "ckpt=Path('results')/exp/'ckpt'; ckpt.mkdir(parents=True,exist_ok=True); (ckpt/'last.pth').write_text('weights')\n"
        "print('[perf 000] example')\n"
    )
    (package / "export_lumir.py").write_text(
        "import sys,json\nfrom pathlib import Path\na=sys.argv[1:]; out=Path(a[a.index('--out')+1]); out.mkdir(parents=True)\n"
        "(out/'args.json').write_text(json.dumps(a))\n"
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
        '#!/usr/bin/env bash\nif [[ "$*" == "-m pytest "* ]]; then exit 0; fi\nexec "$REALPY" "$@"\n', newline="\n"
    )
    for path in binary.iterdir():
        path.chmod(0o755)
    env = os.environ | dict(
        TEST_ROOT=tmp_path.as_posix(),
        TEST_BIN=binary.as_posix(),
        TEST_SCRIPT=SCRIPT.as_posix(),
        REALPY=Path(sys.executable).as_posix(),
        PYBIN=(binary / "python_stub").as_posix(),
        PYTHONPATH=tmp_path.as_posix(),
        EXPECTED_GIT_HEAD="reviewed",
        PRECISION="bf16",
        PAIRS="lumir25",
        EXP="test",
    )

    def run(**updates):
        prefix = '$(cygpath -u "$TEST_BIN")' if os.name == "nt" else "$TEST_BIN"
        return subprocess.run(
            [BASH, "-c", f'export PATH="{prefix}:$PATH"; exec bash "$TEST_SCRIPT"'],
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
    assert "--l3_svf" not in args


def test_training_and_export_use_the_same_explicit_numerical_and_model_recipe(dispatch):
    root, run = dispatch
    result = run(PRECISION="tf32", MODEL="--l3_svf 0 --disable_l1 1", STAGES="ramp", TIME_STEPS="4")
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
