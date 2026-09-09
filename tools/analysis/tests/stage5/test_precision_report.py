from __future__ import annotations

import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import unittest
import zipfile
from pathlib import Path

from tools.analysis.stage5 import precision_report as report

HEAD = "a" * 40


def complete_worker(job: str) -> dict:
    value = {
        "schema": report.WORKER_SCHEMA,
        "job": job,
        "source_run": report.SOURCE_RUN,
        "source_git_head": report.SOURCE_HEAD,
        "diagnostic_git_head": HEAD,
        "diagnostic_only": True,
        "labels_accessed": False,
        "production_checkpoint_written": False,
        "source_bytes_unchanged": True,
        "status": "DIAGNOSTIC_COMPLETE",
    }
    if job == "coverage":
        value["coverage"] = {
            "expected_cases": 24,
            "cases": [
                {"seed": seed, "variant": variant, "status": "COMPLETE", "expected_updates": 8, "completed_updates": 8}
                for seed in range(3)
                for variant in report.VARIANTS
            ],
        }
        return value
    value.update(
        failure_reproduced=True,
        replay={"status": "FAILURE_REPRODUCED"},
        comparison={
            "status": "COMPLETE",
            "ncc_audit": {"status": "PASS"},
            "component_gradient_audit": {"status": "PASS"},
            "probes": [
                {"mode": "fp32_strict", "scale": 1, "status": "FINITE"},
                *[
                    {
                        "mode": "fp16",
                        "scale": scale,
                        "status": "FINITE",
                        "parameter_gradient_comparison": {"status": "FINITE"},
                        "requested_delta_gradient_comparison": {"status": "FINITE"},
                    }
                    for scale in report.PROBE_SCALES
                ],
            ],
        },
        fp32_trajectory={"status": "COMPLETE", "expected_updates": 294, "completed_updates": 294},
        benchmark={
            "status": "COMPLETE",
            "paired_inputs": True,
            "fp16": {
                "completed_steps": 3,
                "full_step_seconds": [8, 10, 9],
                "peak_allocated_bytes": 100,
                "gradient_status": ["FINITE"] * 3,
            },
            "fp32_strict": {
                "completed_steps": 3,
                "full_step_seconds": [11, 12, 10],
                "peak_allocated_bytes": 130,
                "gradient_status": ["FINITE"] * 3,
            },
        },
    )
    return value


class PrecisionReportTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / "S5_PRECISIONDIAG_test"
        self.root.mkdir()
        self.workers = {job: complete_worker(job) for job in report.JOBS}

    def aggregate(self, *, exit_code=0):
        for job, worker in self.workers.items():
            (self.root / f"{job}.json").write_text(json.dumps(worker), encoding="utf-8")
        return report.aggregate_reports(
            self.root, expected_git_head=HEAD, source_run=report.SOURCE_RUN, runner_exit_code=exit_code
        )

    def test_complete_evidence_requires_cost_review_without_training_claim(self):
        result = self.aggregate()
        self.assertEqual(result["branch"], "FP32_CANDIDATE_COST_REVIEW")
        self.assertFalse(result["production_training_validated"])
        self.assertFalse(result["development_evaluation_authorized"])
        self.assertEqual(result["cost_acceptable"], "USER_DECISION_REQUIRED")
        self.assertAlmostEqual(result["benchmarks"]["F0"]["fp32_to_fp16_time_ratio"], 11 / 9)

    def test_missing_worker_never_yields_candidate(self):
        del self.workers["F2P"]
        result = self.aggregate()
        self.assertEqual(result["branch"], "INCOMPLETE")
        self.assertTrue(any("F2P: missing" in error for error in result["errors"]))

    def test_wrong_provenance_blocks_diagnostic_conclusion(self):
        self.workers["F0"]["diagnostic_git_head"] = "b" * 40
        self.workers["F0"]["comparison"]["ncc_audit"]["status"] = "FAIL"
        self.assertEqual(self.aggregate()["branch"], "INCOMPLETE")

    def test_source_mutation_is_separate_integrity_failure(self):
        self.workers["coverage"]["source_bytes_unchanged"] = False
        self.assertEqual(self.aggregate()["branch"], "SOURCE_INTEGRITY_FAILURE")

    def test_audit_failure_is_not_a_candidate_even_with_finite_reference(self):
        self.workers["F2V"]["comparison"]["ncc_audit"]["status"] = "FAIL"
        self.assertEqual(self.aggregate()["branch"], "FIX_NUMERICS_BEFORE_TRAINING")

    def test_explicit_fp32_failure_selects_negative_branch_despite_incomplete_trajectory(self):
        trajectory = self.workers["F0"]["fp32_trajectory"]
        trajectory.update(status="FAILED", completed_updates=2, failure={"kind": "NUMERICAL"})
        result = self.aggregate()
        self.assertEqual(result["branch"], "FIX_NUMERICS_BEFORE_TRAINING")
        self.assertEqual(result["status"], "INCOMPLETE")

    def test_oom_is_not_called_mathematical_failure(self):
        trajectory = self.workers["F0"]["fp32_trajectory"]
        trajectory.update(status="FAILED", completed_updates=0, failure={"kind": "OOM"})
        self.assertEqual(self.aggregate()["branch"], "RESOURCE_LIMIT_FP32")

    def test_reference_oom_and_unavailable_audits_are_resource_limit(self):
        comparison = self.workers["F0"]["comparison"]
        comparison["status"] = "INCOMPLETE"
        comparison["probes"][0]["status"] = "OOM"
        comparison["ncc_audit"] = {"status": "ERROR", "error_kind": "OOM"}
        comparison["component_gradient_audit"] = {"status": "ERROR", "error_kind": "OOM"}
        result = self.aggregate()
        self.assertEqual(result["branch"], "RESOURCE_LIMIT_FP32")
        self.assertFalse(any("audit failed" in finding for finding in result["observed_findings"]))

    def test_observed_math_mismatch_remains_actionable_in_incomplete_probe(self):
        comparison = self.workers["F0"]["comparison"]
        comparison["status"] = "INCOMPLETE"
        comparison["ncc_audit"]["status"] = "FAIL"
        self.assertEqual(self.aggregate()["branch"], "FIX_NUMERICS_BEFORE_TRAINING")

    def test_fp32_coverage_failure_blocks_candidate(self):
        case = self.workers["coverage"]["coverage"]["cases"][-1]
        case.update(status="FAILED", completed_updates=2, failure={"kind": "NUMERICAL"})
        self.assertEqual(self.aggregate()["branch"], "FIX_NUMERICS_BEFORE_TRAINING")

    def test_fp32_benchmark_failure_is_not_only_timing_evidence(self):
        self.workers["F0"]["benchmark"]["fp32_strict"]["gradient_status"][0] = "NONFINITE_GRADIENT_OR_STATE"
        self.assertEqual(self.aggregate()["branch"], "FIX_NUMERICS_BEFORE_TRAINING")

    def test_unverified_source_after_interruption_is_not_called_mutated(self):
        self.workers["F0"]["source_bytes_unchanged"] = None
        self.workers["F0"]["status"] = "RUNNING"
        self.assertEqual(self.aggregate(exit_code=143)["branch"], "INCOMPLETE")

    def test_non_reproduction_cannot_validate_original_failures(self):
        self.workers["F2P"]["failure_reproduced"] = False
        self.assertEqual(self.aggregate()["branch"], "FAILURE_NOT_REPRODUCED")

    def test_duplicate_coverage_case_does_not_cover_missing_variant(self):
        cases = self.workers["coverage"]["coverage"]["cases"]
        cases[-1] = dict(cases[0])
        self.assertEqual(self.aggregate()["branch"], "INCOMPLETE")

    def test_wrong_update_count_and_missing_gradient_audit_are_incomplete(self):
        self.workers["F0"]["fp32_trajectory"]["completed_updates"] = 293
        self.workers["F0"]["comparison"].pop("component_gradient_audit")
        self.assertEqual(self.aggregate()["branch"], "INCOMPLETE")

    def test_nonfinite_timing_is_not_used_as_cost_estimate(self):
        self.workers["F0"]["benchmark"]["fp32_strict"]["full_step_seconds"][0] = float("nan")
        result = self.aggregate()
        self.assertEqual(result["branch"], "INCOMPLETE")
        self.assertNotIn("F0", result["benchmarks"])

    def test_signal_exit_packages_partial_logs_and_marks_incomplete(self):
        (self.root / "F0.log").write_bytes(b"interrupted during backward\n")
        exports = Path(self.temp.name) / "exports"
        code = report.main(
            [
                "--output-root",
                str(self.root),
                "--export-root",
                str(exports),
                "--expected-git-head",
                HEAD,
                "--source-run",
                report.SOURCE_RUN,
                "--runner-exit-code",
                "143",
            ]
        )
        self.assertEqual(code, 1)
        summary = json.loads((self.root / "summary.json").read_text())
        self.assertEqual(summary["branch"], "INCOMPLETE")
        self.assertEqual(summary["runner_exit_code"], 143)
        archives = list(exports.glob("*.zip"))
        self.assertEqual(len(archives), 1)
        with zipfile.ZipFile(archives[0]) as package:
            self.assertEqual(package.read(f"{self.root.name}/F0.log"), b"interrupted during backward\n")

    def test_zip_has_one_root_and_correct_internal_and_external_hashes(self):
        (self.root / "F0.log").write_bytes(b"failure\n")
        (self.root / "nested").mkdir()
        (self.root / "nested" / "report.json").write_bytes(b"{}")
        archive, digest = report.package_compact_zip(self.root, Path(self.temp.name) / "exports")
        self.assertEqual(hashlib.sha256(archive.read_bytes()).hexdigest(), digest)
        self.assertEqual(archive.with_suffix(".zip.sha256").read_text(), f"{digest}  {archive.name}\n")
        with zipfile.ZipFile(archive) as package:
            self.assertIsNone(package.testzip())
            self.assertEqual({name.split("/")[0] for name in package.namelist()}, {self.root.name})
            lines = package.read(f"{self.root.name}/SHA256SUMS").decode().splitlines()
            self.assertEqual(len(lines), 2)
            for line in lines:
                expected, relative = line.split("  ", 1)
                data = package.read(f"{self.root.name}/{relative}")
                self.assertEqual(hashlib.sha256(data).hexdigest(), expected)

    def test_zip_refuses_overwrite_or_heavy_files(self):
        (self.root / "checkpoint.pt").write_bytes(b"tensor")
        with self.assertRaisesRegex(ValueError, "Non-compact"):
            report.package_compact_zip(self.root, Path(self.temp.name) / "exports")
        self.assertFalse((self.root / "SHA256SUMS").exists())

    def test_zip_refuses_existing_archive_before_writing_hashes(self):
        exports = Path(self.temp.name) / "exports"
        exports.mkdir()
        archive = exports / f"{self.root.name}.zip"
        archive.write_bytes(b"do not overwrite")
        with self.assertRaises(FileExistsError):
            report.package_compact_zip(self.root, exports)
        self.assertEqual(archive.read_bytes(), b"do not overwrite")
        self.assertFalse((self.root / "SHA256SUMS").exists())

    def test_zip_refuses_symlink(self):
        target = Path(self.temp.name) / "outside.json"
        target.write_text("{}")
        try:
            (self.root / "linked.json").symlink_to(target)
        except OSError:
            self.skipTest("Symlinks unavailable on this platform")
        with self.assertRaisesRegex(ValueError, "Symlinks"):
            report.package_compact_zip(self.root, Path(self.temp.name) / "exports")


@unittest.skipUnless(os.name == "posix" and shutil.which("bash"), "POSIX process signal test")
class PrecisionRunnerSignalTest(unittest.TestCase):
    def test_sigterm_waits_for_children_and_packages_logs(self):
        real_repo = Path(__file__).resolve().parents[4]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            runner = root / "tools/runners/train/stage5_precision_diagnostic.sh"
            runner.parent.mkdir(parents=True)
            shutil.copyfile(real_repo / "tools/runners/train/stage5_precision_diagnostic.sh", runner)
            source = root / "results/stage5" / report.SOURCE_RUN
            source.mkdir(parents=True)
            (source / "stage5.lock").write_text("")
            binary = root / "bin"
            binary.mkdir()
            scripts = {
                "git": f'#!/usr/bin/env bash\nif [[ "$1" == rev-parse ]]; then echo {HEAD}; fi\n',
                "flock": "#!/usr/bin/env bash\nexit 0\n",
                "nvidia-smi": "#!/usr/bin/env bash\necho fixture\n",
                "python-wrapper": (
                    '#!/usr/bin/env bash\nif [[ "$*" == *tools.analysis.diagnose_stage5_precision* ]]; then\n'
                    '  exec "$REAL_PYTHON" "$STUB_WORKER" "$@"\n'
                    'else\n  exec "$REAL_PYTHON" "$@"\nfi\n'
                ),
            }
            for name, content in scripts.items():
                path = binary / name
                path.write_text(content)
                path.chmod(0o755)
            worker = root / "worker.py"
            worker.write_text(
                "import signal, sys, time\n"
                "def stop(*args):\n    print('worker stopped', flush=True)\n    raise SystemExit(143)\n"
                "signal.signal(signal.SIGTERM, stop)\nprint('worker ready', flush=True)\n"
                "while True: time.sleep(0.1)\n"
            )
            env = dict(os.environ)
            env.update(
                PATH=str(binary) + os.pathsep + env["PATH"],
                PYBIN=str(binary / "python-wrapper"),
                EXPECTED_GIT_HEAD=HEAD,
                REAL_PYTHON=sys.executable,
                STUB_WORKER=str(worker),
                PYTHONPATH=str(real_repo),
            )
            process = subprocess.Popen(["bash", str(runner)], env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            try:
                deadline = time.monotonic() + 15
                ready = False
                while time.monotonic() < deadline and process.poll() is None:
                    logs = list((root / "results/stage5_diagnostics").glob("*/*.log"))
                    ready = len(logs) == 4 and all("worker ready" in path.read_text() for path in logs)
                    if ready:
                        break
                    time.sleep(0.05)
                self.assertTrue(ready, "Runner did not start all four worker fixtures")
                process.send_signal(signal.SIGTERM)
                output, _ = process.communicate(timeout=20)
                self.assertEqual(process.returncode, 143, output.decode())
                archive = next((root / "results/exports").glob("*.zip"))
                with zipfile.ZipFile(archive) as package:
                    logs = [name for name in package.namelist() if name.endswith(".log")]
                    self.assertEqual(len(logs), 4)
                    self.assertTrue(all(b"worker stopped" in package.read(name) for name in logs))
            finally:
                if process.poll() is None:
                    process.kill()
                    process.communicate(timeout=10)


if __name__ == "__main__":
    unittest.main()
