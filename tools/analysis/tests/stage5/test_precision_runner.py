"""Exercise the real Bash queue with lightweight worker processes, without CUDA."""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from tools.analysis.stage5.precision_contract import JOBS, SOURCE_RUN

BASH = "C:/Program Files/Git/bin/bash.exe" if os.name == "nt" else shutil.which("bash")
HEAD = "a" * 40


@unittest.skipUnless(BASH and Path(BASH).is_file(), "Bash unavailable")
class PrecisionRunnerQueueTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        runner = self.root / "tools/runners/train/stage5_precision_diagnostic.sh"
        runner.parent.mkdir(parents=True)
        repo = Path(__file__).resolve().parents[4]
        shutil.copyfile(repo / "tools/runners/train/stage5_precision_diagnostic.sh", runner)
        source = self.root / "results/stage5" / SOURCE_RUN
        source.mkdir(parents=True)
        (source / "stage5.lock").touch()
        binary = self.root / "bin"
        binary.mkdir()
        contract_output = "\\n".join((SOURCE_RUN, *JOBS))
        scripts = {
            "git": f'#!/usr/bin/env bash\nif [[ "$1" == rev-parse ]]; then echo {HEAD}; fi\n',
            "flock": "#!/usr/bin/env bash\nexit 0\n",
            "nvidia-smi": "#!/usr/bin/env bash\necho fixture\n",
            "python-wrapper": (
                '#!/usr/bin/env bash\nset -eu\ncase "$*" in\n'
                f'*precision_contract*) printf "{contract_output}\\n" ;;\n'
                "*diagnose_stage5_precision*)\n"
                '  while [[ "$1" != --job ]]; do shift; done\n'
                '  job="$2"; gpu="$CUDA_VISIBLE_DEVICES"\n'
                '  mkdir "gpu_$gpu.lock" || exit 99\n'
                '  printf "START %s %s\\n" "$job" "$gpu" >> events.txt\n'
                '  if [[ "$job" == F0 && "${SLOW_FIRST:-0}" == 1 ]]; then sleep 3; else sleep 0.05; fi\n'
                '  printf "END %s %s\\n" "$job" "$gpu" >> events.txt\n'
                '  rmdir "gpu_$gpu.lock"\n'
                '  if [[ "$job" == "${FAIL_JOB:-}" ]]; then exit 7; fi\n'
                "  ;;\n"
                "*precision_report*) touch finalized; ;;\n"
                "*) exit 88 ;;\nesac\n"
            ),
        }
        for name, content in scripts.items():
            path = binary / name
            path.write_text(content, encoding="utf-8", newline="\n")
            path.chmod(0o755)

    def run_queue(self, gpus, *, failure="", slow_first=False):
        env = dict(os.environ)
        env.update(GPU_LIST=gpus, EXPECTED_GIT_HEAD=HEAD, FAIL_JOB=failure, SLOW_FIRST=str(int(slow_first)))
        return subprocess.run(
            [
                BASH,
                "-c",
                'export PATH="$PWD/bin:$PATH"; export PYBIN="$PWD/bin/python-wrapper"; '
                "exec bash tools/runners/train/stage5_precision_diagnostic.sh",
            ],
            cwd=self.root,
            env=env,
            capture_output=True,
            text=True,
            timeout=25,
        )

    def assert_complete_queue(self, result, allowed_gpus, *, exit_code=0):
        self.assertEqual(result.returncode, exit_code, result.stdout + result.stderr)
        self.assertTrue((self.root / "finalized").is_file())
        events = [line.split() for line in (self.root / "events.txt").read_text().splitlines()]
        active, started, ended = {}, [], []
        for event, job, gpu in events:
            self.assertIn(gpu, allowed_gpus)
            if event == "START":
                self.assertNotIn(gpu, active, "Two live workers shared a GPU")
                active[gpu] = job
                started.append(job)
            else:
                self.assertEqual(active.pop(gpu), job)
                ended.append(job)
        self.assertFalse(active)
        self.assertCountEqual(started, JOBS)
        self.assertCountEqual(ended, JOBS)
        codes = next((self.root / "results/stage5_diagnostics").glob("*/job_exit_codes.tsv")).read_text()
        self.assertEqual(len(codes.splitlines()), len(JOBS) + 1)
        return events, codes

    def test_nonzero_gpu_indices_refill_as_soon_as_a_worker_finishes(self):
        result = self.run_queue("2,3", slow_first=True)
        events, _ = self.assert_complete_queue(result, {"2", "3"})
        self.assertLess(events.index(["START", "F2P", "3"]), events.index(["END", "F0", "2"]))

    def test_one_gpu_runs_the_entire_queue(self):
        self.assert_complete_queue(self.run_queue("3"), {"3"})

    def test_four_noncontiguous_gpu_indices_are_supported(self):
        self.assert_complete_queue(self.run_queue("2,3,5,7"), {"2", "3", "5", "7"})

    def test_worker_failure_does_not_discard_waiting_jobs(self):
        _, codes = self.assert_complete_queue(self.run_queue("2,3", failure="F0"), {"2", "3"}, exit_code=1)
        self.assertTrue(any(line.startswith("F0\t") and line.endswith("\t7") for line in codes.splitlines()))

    def test_invalid_gpu_lists_fail_before_launching_workers(self):
        for gpus in ("2,2", "2,,3", "2,3,", "-1,3"):
            with self.subTest(gpus=gpus):
                result = self.run_queue(gpus)
                self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
                self.assertFalse((self.root / "events.txt").exists())


if __name__ == "__main__":
    unittest.main()
