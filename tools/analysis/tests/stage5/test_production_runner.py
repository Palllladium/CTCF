"""Execute the real shell scheduling functions with cheap command fixtures."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

GIT_BASH = Path("C:/Program Files/Git/bin/bash.exe")
BASH = str(GIT_BASH) if os.name == "nt" and GIT_BASH.is_file() else shutil.which("bash")
SOURCE = Path("tools/runners/train/stage5.sh").read_text(encoding="utf-8")
HEAD = "a" * 40


def shell_function(name):
    return name + "() {\n" + SOURCE.split(name + "() {\n", 1)[1].split("\n}\n", 1)[0] + "\n}\n"


@unittest.skipUnless(BASH, "Bash is required for the production scheduler test")
class ProductionRunnerTest(unittest.TestCase):
    def execute(self, gpu_list, *, source_run="", fail_variant="", phase="all", training_head="", fail_prepare=False):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            header = (
                "set -Eeuo pipefail\n"
                f"GPU_LIST='{gpu_list}'\nEXPECTED_GIT_HEAD='{HEAD}'\n"
                f"RUN_ID='S5_DEVELOPMENT_20260911T000000Z_{(training_head or HEAD)[:12]}'\n"
                f"IMPORT_U0_RUN_ID='{source_run}'\nPHASE='{phase}'\nTRAINING_GIT_HEAD='{training_head}'\n"
            )
            validation = SOURCE.split("readonly -a SEEDS=", 1)[1].split('\nmkdir -p "$LOG_ROOT"', 1)[0]
            setup = """
LOG_ROOT=logs
STATUS_ROOT=status
HEAVY_ROOT=heavy
CHECKPOINT_ROOT=heavy/checkpoints
SOURCE_ROOT=heavy/source_fields
DECISION_ROOT=heavy/decisions
TRAINING_BARRIER=training.json
DECISION_BARRIER=decision.json
EVALUATION_BARRIER=evaluation.json
EVALUATION_ROOT=evaluation
DATA_CONTRACT=data_contract.json
OASIS_ALL_ROOT=fixture
PROTOCOL_ARGS=()
PROTOCOL=protocol.json
CONTINUATION=continuation.json
DATA_ARGS=()
GIT_ARGS=()
mkdir -p "$LOG_ROOT" "$STATUS_ROOT"
run_cli() {
  printf '%s|%s\n' "${CUDA_VISIBLE_DEVICES:-none}" "$*" >> calls.txt
  if [[ "$1" == prepare-continuation && "$FAIL_PREPARE" == 1 ]]; then return 8; fi
  if [[ "$1" == train-controller && "$*" == *"--variant $FAIL_VARIANT "* ]]; then
    return 7
  fi
}
python_for_evaluation() { printf 'fixture_decision_hash'; }
PYBIN=python_for_evaluation
# Keep mocked jobs fast while exercising the same completion polling loop.
sleep() { command sleep 0.01; }
"""
            names = (
                "run_logged",
                "terminate_active_children",
                "wait_for_batch",
                "train_u0_phase",
                "materialize_source_phase",
                "train_controller_wave",
                "train_controller_phase",
                "decision_worker",
                "decide_phase",
                "evaluate_phase",
                "continue_evaluation_phase",
            )
            script = header + "readonly -a SEEDS=" + validation + "\n" + setup
            script += f"FAIL_VARIANT='{fail_variant}'\nFAIL_PREPARE={int(fail_prepare)}\n"
            script += "\n".join(shell_function(name) for name in names)
            if phase == "continue-evaluation":
                script += '\nPROTOCOL_ARGS=(--continuation "$CONTINUATION")\ncontinue_evaluation_phase\n'
            else:
                script += (
                    "\ntrain_u0_phase\ntrain_controller_phase\nmaterialize_source_phase\ndecide_phase\nevaluate_phase\n"
                )
            result = subprocess.run([BASH, "-s"], input=script, text=True, cwd=root, capture_output=True, timeout=30)
            calls = (root / "calls.txt").read_text().splitlines() if (root / "calls.txt").exists() else []
            return result, calls

    def test_one_and_two_arbitrary_gpus_execute_every_seed_and_controller(self):
        for gpu_list in ("2,3", "9", "7,4,11,1"):
            with self.subTest(gpu_list=gpu_list):
                result, calls = self.execute(gpu_list)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                commands = [line.split("|", 1) for line in calls]
                for device, command in commands:
                    if device != "none":
                        self.assertIn(device, gpu_list.split(","))
                        self.assertIn("--device cuda:0", command)
                u0 = [command for _, command in commands if command.startswith("train-u0 ")]
                self.assertEqual(sorted(re.search(r"--seed (\d+)", command)[1] for command in u0), ["0", "1", "2"])
                controllers = [command for _, command in commands if command.startswith("train-controller ")]
                matrix = {
                    (int(re.search(r"--seed (\d+)", command)[1]), re.search(r"--variant (\w+)", command)[1])
                    for command in controllers
                }
                self.assertEqual(len(controllers), 24)
                self.assertEqual(
                    matrix,
                    {
                        (seed, variant)
                        for seed in range(3)
                        for variant in ("F0", "F2V", "F2S", "F2P", "F4P", "F24P", "A2P", "A24P")
                    },
                )
                sources = [command for _, command in commands if command.startswith("materialize-source ")]
                self.assertEqual({re.search(r"--seed (\d+)", command)[1] for command in sources}, {"0", "1", "2"})
                for seed in ("0", "1", "2"):
                    records = [command for command in sources if f"--seed {seed} " in command]
                    count = int(re.search(r"--num-shards (\d+)", records[0])[1])
                    self.assertEqual(
                        sorted(int(re.search(r"--shard-index (\d+)", command)[1]) for command in records),
                        list(range(count)),
                    )
                self.assertEqual(sum(command.startswith("decide ") for _, command in commands), 27)
                self.assertEqual(
                    sum(command.startswith("evaluate ") for _, command in commands), len(gpu_list.split(","))
                )
                self.assertEqual(sum(command.startswith("aggregate ") for _, command in commands), 1)

    def test_failed_controller_stops_later_training_and_evaluation(self):
        result, calls = self.execute("2,3", fail_variant="F0")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn(sum("|train-controller " in line for line in calls), (1, 2))
        self.assertFalse(
            any("|freeze-training " in line or "|decide " in line or "|evaluate " in line for line in calls)
        )

    def test_continuation_only_runs_post_training_on_arbitrary_gpus(self):
        for gpu_list in ("3", "2,3", "0,1,2,3"):
            with self.subTest(gpu_list=gpu_list):
                result, calls = self.execute(gpu_list, phase="continue-evaluation", training_head="b" * 40)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                commands = [line.split("|", 1)[1] for line in calls]
                self.assertTrue(commands[0].startswith("prepare-continuation "))
                self.assertFalse(
                    any(command.startswith(("train-", "materialize-source", "prepare-data")) for command in commands)
                )
                self.assertEqual(sum(command.startswith("decide ") for command in commands), 27)
                self.assertEqual(sum(command.startswith("evaluate ") for command in commands), len(gpu_list.split(",")))
                self.assertEqual(sum(command.startswith("aggregate ") for command in commands), 1)
                for command in commands[1:]:
                    if not command.startswith("disk-preflight"):
                        self.assertIn("--continuation continuation.json", command)

    def test_failed_continuation_verification_starts_no_gpu_work(self):
        result, calls = self.execute("2,3", phase="continue-evaluation", training_head="b" * 40, fail_prepare=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(len(calls), 1)

    def test_training_head_exception_is_restricted_to_explicit_continuation(self):
        for phase, training_head in (("all", "b" * 40), ("continue-evaluation", "")):
            result, calls = self.execute("2,3", phase=phase, training_head=training_head)
            self.assertEqual(result.returncode, 2)
            self.assertFalse(calls)

    def test_invalid_gpu_lists_fail_before_any_worker(self):
        for gpu_list in ("", "2,2", "2,", "2,,3", "02,2", "-1,3", "a,3"):
            with self.subTest(gpu_list=gpu_list):
                result, calls = self.execute(gpu_list)
                self.assertEqual(result.returncode, 2)
                self.assertFalse(calls)

    def test_both_supported_u0_source_revisions_pass_shell_validation(self):
        for suffix in ("68df5b241042", "ffd3090f6129"):
            result, _ = self.execute("2,3", source_run=f"S5_DEVELOPMENT_20260907T202438Z_{suffix}")
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_failure_manifest_is_copied_but_tensor_capture_stays_heavy(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            failure = root / "heavy/checkpoints/controllers/seed_0/F0/failures/step_1"
            failure.mkdir(parents=True)
            (failure / "failure.json").write_text('{"status":"FAILED"}\n', encoding="utf-8")
            (failure / "capture.pth").write_bytes(b"heavy fixture")
            script = (
                """
set -Eeuo pipefail
COMPACT_ROOT=compact
MANIFEST_ROOT=manifests
CHECKPOINT_ROOT=heavy/checkpoints
SOURCE_ROOT=heavy/source_fields
DECISION_ROOT=heavy/decisions
"""
                + shell_function("copy_compact_attestations")
                + "\ncopy_compact_attestations\n"
            )
            result = subprocess.run([BASH, "-s"], input=script, text=True, cwd=root, capture_output=True, timeout=10)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            compact_files = list((root / "compact").rglob("*.*"))
            self.assertEqual(len(compact_files), 1)
            self.assertEqual(compact_files[0].name, "failure.json")
            self.assertEqual(compact_files[0].read_bytes(), (failure / "failure.json").read_bytes())
            self.assertTrue((failure / "capture.pth").is_file())

    def test_quick_failure_stops_slow_sibling_before_failed_packaging(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "slow.sh").write_text(
                "trap 'sleep 0.2; echo stopped >> events; exit 143' TERM\n"
                "echo $BASHPID > slow_child.pid\n"
                "while true; do sleep 0.05; done\n",
                encoding="utf-8",
            )
            (root / "fail.sh").write_text(
                "while [[ ! -f slow_child.pid ]]; do sleep 0.05; done\necho capture_complete >> events\nexit 7\n",
                encoding="utf-8",
            )
            script = "set -Eeuo pipefail\nACTIVE_PIDS=()\nFINALIZED=0\n"
            script += "\n".join(
                shell_function(name)
                for name in ("run_logged", "terminate_active_children", "wait_for_batch", "on_exit")
            )
            # Git Bash has no pkill. This fixture signals only the declared direct
            # child of our slow wrapper; POSIX exercises the production pkill.
            if os.name == "nt":
                script += """
pkill() {
  [[ "$1" == -TERM && "$2" == -P && "$3" == "$slow_pid" ]] || return 1
  kill -TERM "$(cat slow_child.pid)"
}
"""
            script += """
package_attempt() {
  [[ "$1" == FAILED && "$2" -ne 0 ]] || return 90
  grep -q capture_complete events || return 91
  grep -q stopped events || return 92
  if kill -0 "$slow_pid" 2>/dev/null || kill -0 "$failed_pid" 2>/dev/null; then return 93; fi
  if kill -0 "$(cat slow_child.pid)" 2>/dev/null; then return 94; fi
  echo packaged >> events
}
trap on_exit EXIT
run_logged slow.log bash slow.sh &
slow_pid=$!
echo "$slow_pid" > slow_wrapper.pid
run_logged failed.log bash fail.sh &
failed_pid=$!
echo "$failed_pid" > failed_wrapper.pid
wait_for_batch "$slow_pid" "$failed_pid"
echo unexpected_success >> events
"""
            process = subprocess.Popen(
                [BASH, "-s"], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, cwd=root
            )
            try:
                stdout, stderr = process.communicate(script, timeout=12)
                self.assertNotEqual(process.returncode, 0, stdout + stderr)
                self.assertEqual(
                    (root / "events").read_text().splitlines(), ["capture_complete", "stopped", "packaged"]
                )
            finally:
                if process.poll() is None:
                    for name in ("slow_child.pid", "slow_wrapper.pid", "failed_wrapper.pid"):
                        path = root / name
                        if path.exists():
                            pid = int(path.read_text())
                            subprocess.run(
                                [BASH, "-c", f"kill -KILL {pid} 2>/dev/null || true"], timeout=5, check=False
                            )
                    process.kill()
                    process.communicate(timeout=5)


if __name__ == "__main__":
    unittest.main()
