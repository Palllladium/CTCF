"""Exercise final report files, archive integrity, and the actual runner contract."""

from __future__ import annotations

import contextlib
import hashlib
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest import mock

from tools.analysis.stage5 import mechanism_contract as contract, mechanism_report as report
from tools.analysis.tests.stage5 import test_mechanism_runner as runner_fixture
from tools.analysis.tests.stage5.test_mechanism_report import complete_audits

REPO = Path(__file__).resolve().parents[4]
HEAD = "a" * 40


def _worker(job):
    audits = complete_audits()
    worker = {
        "schema": contract.SCHEMA,
        "job": job,
        "diagnostic_git_head": HEAD,
        "workload_contract": contract.workload_contract(),
        "precision_package_verified": True,
        "mechanism_sources_unchanged": True,
        "production_checkpoint_written": False,
        "labels_accessed": False,
        "status": "DIAGNOSTIC_COMPLETE",
        "environment": {"backend_flags": audits["bias_audit"]["original_backend_flags"]},
    }
    if job != "F2P":
        worker.update(audits)
        return worker
    worker["f2p_replay"] = [
        {
            "attempt": index,
            "status": "NOT_REPRODUCED",
            "expected_pairs": contract.F2P_UPDATES,
            "completed_updates": contract.F2P_UPDATES,
            "initial_model_sha256": "a" * 64,
            "pair_schedule_sha256": "b" * 64,
            "pairs": [{"pair_index": pair} for pair in range(1, contract.F2P_UPDATES + 1)],
        }
        for index in range(1, contract.F2P_ATTEMPTS + 1)
    ]
    return worker


class MechanismFinalizationTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name) / "S5_MECHANISMDIAG_fixture"
        self.root.mkdir()
        self.exports = Path(temp.name) / "exports"

    def write_workers(self, *, omit=None):
        for job in contract.JOBS:
            if job != omit:
                (self.root / f"{job}.json").write_text(json.dumps(_worker(job)), encoding="utf-8")
        logs = self.root / "logs"
        logs.mkdir()
        (logs / "F0.log").write_text("captured observations\n", encoding="utf-8")

    def finalize(self, exit_code=0):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            result = report.main(
                [
                    "--output-root",
                    str(self.root),
                    "--export-root",
                    str(self.exports),
                    "--expected-git-head",
                    HEAD,
                    "--runner-exit-code",
                    str(exit_code),
                ]
            )
        return result, output.getvalue()

    def check_package(self, summary):
        archive = self.exports / f"{self.root.name}.zip"
        sidecar = archive.with_suffix(".zip.sha256")
        digest = hashlib.sha256(archive.read_bytes()).hexdigest()
        self.assertEqual(sidecar.read_text(encoding="utf-8"), f"{digest}  {archive.name}\n")
        with zipfile.ZipFile(archive) as package:
            self.assertIsNone(package.testzip())
            names = package.namelist()
            self.assertEqual(len(names), len(set(names)))
            prefix = self.root.name + "/"
            self.assertTrue(all(name.startswith(prefix) for name in names))
            relative = {name.removeprefix(prefix) for name in names}
            self.assertEqual(
                relative,
                {path.relative_to(self.root).as_posix() for path in self.root.rglob("*") if path.is_file()},
            )
            self.assertIn("logs/F0.log", relative)
            self.assertEqual(json.loads(package.read(prefix + "summary.json")), summary)
            sums = package.read(prefix + "SHA256SUMS").decode("utf-8")
            self.assertEqual(sums, (self.root / "SHA256SUMS").read_text(encoding="utf-8"))
            entries = [line.split("  ", 1) for line in sums.splitlines()]
            self.assertEqual({name for _, name in entries}, relative - {"SHA256SUMS"})
            for expected, name in entries:
                self.assertEqual(hashlib.sha256(package.read(prefix + name)).hexdigest(), expected, name)
        return archive, digest

    def test_complete_main_writes_review_summary_and_verified_single_root_zip(self):
        self.write_workers()
        result, output = self.finalize()
        self.assertEqual(result, 0)
        summary = json.loads((self.root / "summary.json").read_text(encoding="utf-8"))
        self.assertEqual(summary["status"], "OBSERVATIONS_COMPLETE_REVIEW_REQUIRED")
        self.assertEqual(summary["errors"], [])
        self.assertFalse(summary["production_training_validated"])
        self.assertFalse(summary["production_restart_authorized"])
        self.assertEqual(summary["jobs"]["F2P"]["reproduction"], "NOT_REPRODUCED")
        for job in contract.JOBS:
            self.assertEqual(
                summary["worker_report_sha256"][job],
                hashlib.sha256((self.root / f"{job}.json").read_bytes()).hexdigest(),
            )
        markdown = (self.root / "SUMMARY.md").read_text(encoding="utf-8")
        self.assertIn(summary["status"], markdown)
        self.assertIn("F2P reproduction: NOT_REPRODUCED", markdown)
        archive, digest = self.check_package(summary)
        self.assertIn(f"[MECHANISM DIAG STATUS] {summary['status']}", output)
        self.assertIn(f"[MECHANISM DIAG PACKAGE] {archive}", output)
        self.assertIn(f"[MECHANISM DIAG PACKAGE SIDECAR] {archive}.sha256", output)
        self.assertIn(f"{digest}  {archive.name}", output)

    def test_incomplete_main_still_packages_available_evidence_and_returns_failure(self):
        self.write_workers(omit="F2V")
        result, output = self.finalize(exit_code=7)
        self.assertEqual(result, 1)
        summary = json.loads((self.root / "summary.json").read_text(encoding="utf-8"))
        self.assertEqual(summary["status"], "INCOMPLETE")
        self.assertTrue(any(error.startswith("F2V: missing or unreadable worker") for error in summary["errors"]))
        self.assertIn("Runner exit code 7", summary["errors"])
        self.assertEqual(set(summary["jobs"]), {"F0", "F2P"})
        self.assertIn("[MECHANISM DIAG STATUS] INCOMPLETE", output)
        self.assertIn("Runner exit code 7", (self.root / "SUMMARY.md").read_text(encoding="utf-8"))
        self.check_package(summary)

    def test_existing_summary_is_never_overwritten_or_partially_repackaged(self):
        self.write_workers()
        self.finalize()
        files = [path for directory in (self.root, self.exports) for path in directory.rglob("*") if path.is_file()]
        original = {path: path.read_bytes() for path in files}
        with self.assertRaises(FileExistsError):
            self.finalize(exit_code=9)
        self.assertEqual({path: path.read_bytes() for path in files}, original)


class MechanismContractEntrypointTests(unittest.TestCase):
    def test_actual_contract_cli_has_source_precision_and_jobs_in_runner_order(self):
        result = subprocess.run(
            [sys.executable, "-m", "tools.analysis.stage5.mechanism_contract", "--runner-values"],
            cwd=REPO,
            capture_output=True,
            text=True,
            timeout=20,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(result.stdout.splitlines(), [contract.SOURCE_RUN, contract.PRECISION_RUN, *contract.JOBS])

    @unittest.skipUnless(runner_fixture.BASH and Path(runner_fixture.BASH).is_file(), "Bash unavailable")
    def test_shell_parser_uses_real_contract_for_worker_paths_and_job_queue(self):
        fixture = runner_fixture.MechanismRunnerTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        wrapper = fixture.root / "bin/python-wrapper"
        source = wrapper.read_text(encoding="utf-8")
        expected = "\\n".join((contract.SOURCE_RUN, contract.PRECISION_RUN, *contract.JOBS))
        # Windows Python writes CRLF to pipes. Normalize that transport detail so
        # this Linux runner test executes the real module without a fake response.
        source = source.replace(
            f'*mechanism_contract*) printf "{expected}\\n" ;;',
            '*mechanism_contract*) (cd "$TEST_SOURCE_REPO"; "$TEST_REAL_PYTHON" "$@") | tr -d \'\\r\' ;;',
        )
        self.assertNotIn(f'printf "{expected}\\n"', source)
        source = source.replace("*diagnose_stage5_mechanism*)\n", '*diagnose_stage5_mechanism*)\n  all_args=("$@")\n')
        source = source.replace(
            '  job="$2"; gpu="$CUDA_VISIBLE_DEVICES"\n',
            '  job="$2"; gpu="$CUDA_VISIBLE_DEVICES"\n  printf "%s\\n" "${all_args[@]}" > "args_$job.txt"\n',
        )
        wrapper.write_text(source, encoding="utf-8", newline="\n")
        with mock.patch.dict(
            os.environ, {"TEST_SOURCE_REPO": REPO.as_posix(), "TEST_REAL_PYTHON": Path(sys.executable).as_posix()}
        ):
            fixture.check_events("2,3")
        for job in contract.JOBS:
            arguments = (fixture.root / f"args_{job}.txt").read_text(encoding="utf-8").splitlines()
            for flag, expected in (
                ("--job", job),
                ("--source-root", f"results/stage5/{contract.SOURCE_RUN}"),
                ("--checkpoint-root", f"results/stage5_heavy/{contract.SOURCE_RUN}/checkpoints"),
                ("--precision-root", f"results/stage5_diagnostics/{contract.PRECISION_RUN}"),
            ):
                self.assertEqual(arguments[arguments.index(flag) + 1], expected)


if __name__ == "__main__":
    unittest.main()
