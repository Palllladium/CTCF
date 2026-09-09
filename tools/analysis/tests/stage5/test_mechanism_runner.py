from __future__ import annotations

import shutil
import unittest
from pathlib import Path

from tools.analysis.stage5.mechanism_contract import JOBS, PRECISION_RUN
from tools.analysis.stage5.precision_contract import SOURCE_RUN
from tools.analysis.tests.stage5 import test_precision_runner as fixture_module
from tools.analysis.tests.stage5.test_precision_runner import BASH


@unittest.skipUnless(BASH and Path(BASH).is_file(), "Bash unavailable")
class MechanismRunnerTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixture_module.PrecisionRunnerQueueTest()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.root = self.fixture.root
        # Reuse executable process fixtures while exercising the new runner bytes.
        repo = Path(__file__).resolve().parents[4]
        shutil.copyfile(
            repo / "tools/runners/train/stage5_mechanism_diagnostic.sh",
            self.root / "tools/runners/train/stage5_precision_diagnostic.sh",
        )
        wrapper = self.root / "bin/python-wrapper"
        text = wrapper.read_text()
        old_values = "\\n".join((SOURCE_RUN, *fixture_module.JOBS))
        new_values = "\\n".join((SOURCE_RUN, PRECISION_RUN, *JOBS))
        assert old_values in text
        text = text.replace(old_values, new_values).replace("precision_contract", "mechanism_contract")
        text = text.replace("diagnose_stage5_precision", "diagnose_stage5_mechanism").replace(
            "precision_report", "mechanism_report"
        )
        wrapper.write_text(text, encoding="utf-8", newline="\n")

    def check_events(self, gpus, *, failure=""):
        result = self.fixture.run_queue(gpus, failure=failure)
        self.assertEqual(result.returncode, 1 if failure else 0, result.stdout + result.stderr)
        events = [row.split() for row in (self.root / "events.txt").read_text().splitlines()]
        active, started = {}, []
        for event, job, gpu in events:
            self.assertIn(gpu, gpus.split(","))
            if event == "START":
                self.assertNotIn(gpu, active)
                active[gpu] = job
                started.append(job)
            else:
                self.assertEqual(active.pop(gpu), job)
        self.assertFalse(active)
        self.assertCountEqual(started, JOBS)
        self.assertTrue((self.root / "finalized").is_file())

    def test_two_physical_gpus_continue_after_failed_saved_state_job(self):
        self.check_events("2,3", failure="F0")

    def test_one_gpu_runs_all_three_jobs(self):
        self.check_events("3")


if __name__ == "__main__":
    unittest.main()
