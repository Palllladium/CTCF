"""Exercise subprocess isolation, fair scheduling, and restart provenance cheaply."""

from __future__ import annotations

import copy
import csv
import io
import subprocess
import sys
import tempfile
import unittest
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from tools.analysis.run_artifacts import sha256_file
from tools.analysis.stage5.comparison_execution import (
    ComparisonExecutor,
    ComparisonJob,
    _device_interval_metrics,
    build_job_plan,
    execute_comparison,
    parse_gpu_list,
    read_monitor_samples,
    throughput_summary,
    write_json,
)
from tools.analysis.tests.stage5.test_production_runner import BASH, SOURCE, shell_function

# Verbatim rows from S5_COMPARISON_20260912T061525Z_632e518a15c9,
# comparison/attempts/20260912T061712_4a6d4760/gpu.csv. Keep the units:
# the original regression used bare numbers, unlike the recorded --format=csv.
H100_CSV = """timestamp, index, uuid, name, utilization.gpu [%], utilization.memory [%], memory.used [MiB], memory.total [MiB], temperature.gpu, power.draw [W]
2026/09/12 06:17:26.305, 2, GPU-107f0d10-2c02-7c73-7b9f-680acd29b484, NVIDIA H100 80GB HBM3, 100 %, 1 %, 29089 MiB, 81559 MiB, 35, 154.88 W
2026/09/12 06:17:26.305, 3, GPU-8c958eaa-b244-64cb-6283-d37792aac963, NVIDIA H100 80GB HBM3, 100 %, 3 %, 29089 MiB, 81559 MiB, 29, 158.86 W
2026/09/12 06:17:28.306, 3, GPU-8c958eaa-b244-64cb-6283-d37792aac963, NVIDIA H100 80GB HBM3, 100 %, 0 %, 36227 MiB, 81559 MiB, 29, 162.00 W
2026/09/12 06:17:29.306, 2, GPU-107f0d10-2c02-7c73-7b9f-680acd29b484, NVIDIA H100 80GB HBM3, 100 %, 0 %, 36227 MiB, 81559 MiB, 36, 150.74 W
"""

FIXTURE = r"""
import argparse, hashlib, json, os, time
from datetime import datetime, timezone
from pathlib import Path
p = argparse.ArgumentParser()
for name in ("stage", "variant", "seed", "mode", "replicate", "protocol", "data-contract", "image-root", "checkpoint-root", "run-id", "expected-git-head", "repo-root", "precision-source-root", "capture-root", "output-root", "heavy-root", "ready-file", "start-gate", "gate-timeout-seconds"):
    p.add_argument("--" + name)
a = p.parse_args()
output = Path(a.output_root)
if a.ready_file:
    Path(a.ready_file).write_text("{}")
    start = time.monotonic()
    while not Path(a.start_gate).is_file():
        if time.monotonic() - start > 5: raise RuntimeError("gate timeout")
        time.sleep(.01)
started = datetime.now(timezone.utc).isoformat()
time.sleep(.15)
if a.variant == "INFRA": raise RuntimeError("fixture infrastructure failure")
expected = 4 if a.stage == "paired" else 24 if a.stage == "coverage" else 294
status = "CANDIDATE_FAILURE" if a.variant == "F2V" else "COMPLETE"
record = dict(schema="ctcf-stage5-precision-comparison-worker-v1", status=status, stage=a.stage, variant=a.variant, seed=int(a.seed), mode=a.mode, replicate=int(a.replicate), expected_updates=expected, successful_updates=expected if status == "COMPLETE" else 0, gpu=os.environ["CUDA_VISIBLE_DEVICES"], training_started_at_utc=started, training_finished_at_utc=datetime.now(timezone.utc).isoformat())
if Path(a.protocol).exists():
    protocol = json.loads(Path(a.protocol).read_text())
    digest = hashlib.sha256((json.dumps(protocol, sort_keys=True, separators=(",", ":")) + "\n").encode()).hexdigest()
    record.update(git_head=a.expected_git_head, run_id=a.run_id, protocol_sha256=digest, sources_unchanged=True, labels_accessed=False, production_checkpoint_written=False, automatic_mode_selection=False)
    record["source_hashes"] = {str(Path(a.protocol).resolve()): hashlib.sha256(Path(a.protocol).read_bytes()).hexdigest()}
(output / "result.json").write_text(json.dumps(record))
"""


class ComparisonSchedulerTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.worker = self.root / "worker.py"
        self.worker.write_text(FIXTURE, encoding="utf-8")
        self.args = SimpleNamespace(
            output_root=self.root / "compact",
            heavy_root=self.root / "heavy",
            repo_root=self.root,
            protocol=self.root / "protocol.json",
            data_contract=self.root / "data.json",
            image_root=self.root / "images",
            checkpoint_root=self.root / "checkpoints",
            precision_source_root=self.root / "historical",
            capture_root=self.root / "captures",
            run_id="S5_COMPARE_20260912T000000Z_aaaaaaaaaaaa",
            expected_git_head="a" * 40,
            gpu_list="2,3",
        )

    def executor(self, **context):
        return ComparisonExecutor(
            self.args, context={"head": "a" * 40, **context}, worker_prefix=[sys.executable, str(self.worker)]
        )

    def test_complete_plan_and_physical_gpu_pinning(self):
        for value in ("2,3", "9", "4,8,3,1"):
            with self.subTest(gpus=value):
                jobs = build_job_plan(parse_gpu_list(value))
                self.assertEqual(len(jobs), 51)
                self.assertEqual(len({job.job_id for job in jobs}), len(jobs))
                self.assertEqual(sum(job.stage == "coverage" for job in jobs), 24)
                self.assertEqual(sum(job.stage == "trajectory" for job in jobs), 12)
                for variant in ("F0", "F2V", "F2P"):
                    devices = {
                        job.gpu for job in jobs if job.variant == variant and job.stage in ("paired", "trajectory")
                    }
                    self.assertEqual(len(devices), 1)
                self.assertTrue(all(job.gpu in value.split(",") for job in jobs))
        for value in ("", "2,2", "2,", "02,2", "-1", "2,,3"):
            with self.assertRaises(ValueError):
                parse_gpu_list(value)

    def test_worker_failures_are_observations_and_do_not_stop_other_jobs(self):
        executor = self.executor()
        jobs = [
            ComparisonJob(f"j{i}", "paired", variant, 0, "fp32_strict", 0, str(i % 2 + 2))
            for i, variant in enumerate(("F2V", "INFRA", "F0"))
        ]
        executor.run_jobs(jobs)
        records = {record["spec"]["job"]["variant"]: record for record in executor.results}
        self.assertEqual(records["F2V"]["status"], "CANDIDATE_FAILURE")
        self.assertEqual(records["INFRA"]["status"], "INCOMPLETE")
        self.assertEqual(records["F0"]["status"], "COMPLETE")
        self.assertEqual(records["F0"]["result"]["gpu"], "2")
        self.assertFalse(executor.active)

    def test_reuse_requires_identical_spec_and_authenticated_complete_result(self):
        job = ComparisonJob("j", "paired", "F0", 0, "fp32_strict", 0, "3")
        first = self.executor()
        first.run_jobs([job])
        second = self.executor()
        second.run_jobs([job])
        self.assertTrue(second.results[0]["reused"])
        self.assertIsNone(second.reusable(replace(job, gpu="7")))
        self.assertIsNone(self.executor(head="b" * 40).reusable(job))
        path = Path(first.results[0]["output_root"]) / "result.json"
        path.write_text(path.read_text() + " ")
        self.assertIsNone(second.reusable(job))
        third = self.executor()
        third.run_jobs([job])
        self.assertFalse(third.results[0]["reused"])
        self.assertTrue(path.is_file())

    def test_dual_throughput_uses_shared_gate_and_cannot_reuse_half_a_pair(self):
        jobs = [ComparisonJob(f"dual_r{i}", "throughput", "F0", 0, "fp32_strict", i, "7", 2) for i in (0, 1)]
        first = self.executor()
        first.run_jobs(jobs, shared_gate=True)
        self.assertEqual([record["status"] for record in first.results], ["COMPLETE", "COMPLETE"])
        self.assertTrue(all(record["result"]["gpu"] == "7" for record in first.results))
        starts = [record["result"]["training_started_at_utc"] for record in first.results]
        ends = [record["result"]["training_finished_at_utc"] for record in first.results]
        self.assertLess(max(starts), min(ends))
        path = Path(first.results[0]["output_root"]) / "result.json"
        path.write_text("invalid")
        second = self.executor()
        second.run_jobs(jobs, shared_gate=True)
        self.assertTrue(all(not record["reused"] for record in second.results))
        self.assertEqual(len(list(path.parent.parent.glob("*/execution.json"))), 2)

    def test_incomplete_update_count_is_not_accepted(self):
        job = ComparisonJob("j", "trajectory", "F0", 0, "fp32_strict", 0, "3")
        result = {
            "schema": "ctcf-stage5-precision-comparison-worker-v1",
            "status": "COMPLETE",
            "stage": "trajectory",
            "variant": "F0",
            "seed": 0,
            "mode": "fp32_strict",
            "replicate": 0,
            "expected_updates": 1,
            "successful_updates": 1,
        }
        with self.assertRaisesRegex(ValueError, "required updates"):
            ComparisonExecutor._validate_result(job, result)

    def test_launch_failure_does_not_overwrite_or_claim_completion(self):
        executor = self.executor()
        job = ComparisonJob("j", "paired", "F0", 0, "fp32_strict", 0, "3")
        with patch("subprocess.Popen", side_effect=OSError("fixture launch failure")):
            executor.run_jobs([job])
        self.assertEqual(executor.results[0]["status"], "INCOMPLETE")
        self.assertFalse(executor.active)

    def test_stop_terminates_only_owned_worker_processes(self):
        unrelated = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
        executor = self.executor()
        try:
            job = ComparisonJob("owned", "paired", "F0", 0, "fp32_strict", 0, "3")
            executor._launch(job, None)
            owned = [state["process"] for state in executor.active.values()]
            executor.stop()
            self.assertTrue(all(process.poll() is not None for process in owned))
            self.assertIsNone(unrelated.poll())
            self.assertFalse(executor.active)
            self.assertEqual(executor.results[0]["status"], "INCOMPLETE")
        finally:
            unrelated.terminate()
            unrelated.wait(timeout=10)

    def test_measured_throughput_requires_both_successful_members(self):
        executor = self.executor()
        plan = [job for job in build_job_plan(("3",)) if job.stage == "throughput" and job.mode == "fp32_strict"]
        for job in plan[:2]:
            executor.run_jobs([job], shared_gate=True)
        executor.run_jobs(plan[2:], shared_gate=True)
        summary = throughput_summary(executor.results)[0]
        self.assertEqual(summary["concurrency_1"]["successful_updates"], 588)
        self.assertEqual(summary["concurrency_2"]["successful_updates"], 588)
        self.assertGreater(summary["two_process_throughput_ratio"], 1)

    def test_full_orchestration_writes_complete_observation_and_preserves_attempts(self):
        self.args.protocol.write_text("{}")
        self.args.data_contract.write_text("{}")
        self.args.precision_source_root.mkdir()
        (self.args.precision_source_root / "SHA256SUMS").write_text("fixture inventory")
        worker = self.worker

        class FixtureExecutor(ComparisonExecutor):
            def __init__(self, args, *, context):
                super().__init__(args, context=context, worker_prefix=[sys.executable, str(worker)])

        def fake_monitor_stop(monitor):
            records = []
            for name in ("gpu", "processes"):
                path = monitor.root / f"{name}.csv"
                path.write_text("fixture monitoring")
                records.append(
                    {
                        "name": name,
                        "status": "RECORDED",
                        "path": str(path),
                        "bytes": path.stat().st_size,
                        "sha256": sha256_file(path),
                        "sample_status": "RECORDED",
                        "gpu_summary": {gpu: {"samples": 1} for gpu in monitor.gpus},
                    }
                )
            write_json(monitor.root / "monitor.json", records)
            return records

        with (
            patch("tools.analysis.stage5.comparison_execution.ComparisonExecutor", FixtureExecutor),
            patch("tools.analysis.stage5.comparison_execution.GpuMonitor.start"),
            patch("tools.analysis.stage5.comparison_execution.GpuMonitor.stop", fake_monitor_stop),
        ):
            first = execute_comparison(self.args, protocol={})
            second = execute_comparison(self.args, protocol={})
        self.assertEqual(first["status"], "DIAGNOSTIC_COMPLETE")
        self.assertEqual(first["recorded_jobs"], 51)
        self.assertFalse(first["automatic_mode_selection"])
        self.assertFalse(first["production_checkpoint_promotion"])
        self.assertTrue(all(record["reused"] for record in second["results"]))
        self.assertTrue(all("result" not in record for record in first["results"]))
        self.assertEqual(len(list((self.args.output_root / "attempts").glob("*/summary.json"))), 2)
        self.assertTrue((self.args.output_root / "report.md").is_file())
        old_monitor = Path(first["results"][0]["monitor_root"]) / "gpu.csv"
        old_monitor.write_text("corrupted old monitoring")
        probe = FixtureExecutor(self.args, context=first["context"])
        self.assertIsNone(probe.reusable(build_job_plan(("2", "3"))[0]))

    def test_monitor_requires_real_gpu_samples_and_summarizes_measured_peaks(self):
        path = self.root / "gpu.csv"
        header = "timestamp, index, uuid, memory.used [MiB], utilization.gpu [%], temperature.gpu, power.draw [W]\n"
        path.write_text(header)
        with self.assertRaisesRegex(ValueError, "no data samples"):
            read_monitor_samples(path, "gpu", ("3",))
        path.write_text(
            header
            + "2026/09/12 01:00:00.000, 3, GPU-fixture, 30000, 98, 70, 200\n2026/09/12 01:00:01.000, 3, GPU-fixture, 40000, 95, 72, 220\n"
        )
        record = read_monitor_samples(path, "gpu", ("3",))
        self.assertEqual(record["gpu_summary"]["3"]["peak_memory_used_mib"], 40000)
        self.assertEqual(record["gpu_summary"]["3"]["peak_utilization_percent"], 98)
        self.assertEqual(record["gpu_summary"]["3"]["samples"], 2)
        with self.assertRaisesRegex(ValueError, "no data samples"):
            read_monitor_samples(path, "gpu", ("2", "3"))

    def monitored_interval(self, text, *, gpu="2"):
        path = self.root / "gpu.csv"
        path.write_text(text, encoding="utf-8")
        manifest = self.root / "monitor.json"
        write_json(
            manifest,
            [
                {
                    "name": "gpu",
                    "status": "RECORDED",
                    "path": str(path),
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                    "timestamp_utc_offset_seconds": 0,
                }
            ],
        )
        record = {
            "spec": {"job": {"gpu": gpu}},
            "monitor_root": str(self.root),
            "monitor_ref": {"path": str(manifest), "sha256": sha256_file(manifest)},
        }
        start = datetime(2026, 9, 12, tzinfo=timezone.utc).timestamp()
        return _device_interval_metrics([record], [(start, start + 86400)], {})

    def test_real_h100_units_in_both_monitor_aggregation_paths(self):
        path = self.root / "gpu.csv"
        path.write_text(H100_CSV, encoding="utf-8")
        summary = read_monitor_samples(path, "gpu", ("2", "3"), timestamp_utc_offset_seconds=0)
        self.assertEqual(summary["sample_rows"], 4)
        for gpu, power, temperature in (("2", 154.88, 36), ("3", 162.0, 29)):
            with self.subTest(gpu=gpu):
                overall = summary["gpu_summary"][gpu]
                interval = self.monitored_interval(H100_CSV, gpu=gpu)
                self.assertEqual(interval["status"], "RECORDED")
                for record in (overall, interval):
                    self.assertEqual(record["samples"], 2)
                    self.assertEqual(record["peak_memory_used_mib"], 36227)
                    self.assertEqual(record["peak_utilization_percent"], 100)
                    self.assertEqual(record["peak_power_w"], power)
                    self.assertEqual(record["peak_temperature_c"], temperature)
                    self.assertTrue(
                        all(field["status"] == "RECORDED" for field in record["measurement_coverage"].values())
                    )

    def test_invalid_units_and_numbers_are_rejected_even_for_optional_sensors(self):
        original = next(csv.DictReader(io.StringIO(H100_CSV), skipinitialspace=True))
        fields = {"memory.used [MiB]": "MiB", "utilization.gpu [%]": "%", "temperature.gpu": "", "power.draw [W]": "W"}
        for field, unit in fields.items():
            for invalid in ("1 bananas", "1 GiB", "nan", "inf", "-1", "1e999", f"-1 {unit}".strip()):
                with self.subTest(field=field, invalid=invalid):
                    stream = io.StringIO()
                    writer = csv.DictWriter(stream, fieldnames=original)
                    writer.writeheader()
                    writer.writerow({**original, field: invalid})
                    text = stream.getvalue()
                    interval = self.monitored_interval(text)
                    self.assertEqual(interval["status"], "UNAVAILABLE")
                    self.assertNotIn("peak_memory_used_mib", interval)
                    with self.assertRaises(ValueError):
                        read_monitor_samples(self.root / "gpu.csv", "gpu", ("2",), timestamp_utc_offset_seconds=0)

    def test_optional_unavailable_sensors_have_explicit_coverage(self):
        header = "timestamp,index,uuid,memory.used [MiB],utilization.gpu [%],temperature.gpu,power.draw [W]\n"
        text = (
            header
            + "2026/09/12 06:17:26.305,2,GPU-fixture,1 MiB,0 %,N/A,[Not Supported]\n"
            + "2026/09/12 06:17:27.305,2,GPU-fixture,2,1,35,[N/A]\n"
        )
        interval = self.monitored_interval(text)
        overall = read_monitor_samples(self.root / "gpu.csv", "gpu", ("2",), timestamp_utc_offset_seconds=0)[
            "gpu_summary"
        ]["2"]
        for result in (overall, interval):
            self.assertEqual(result["peak_temperature_c"], 35)
            self.assertIsNone(result["peak_power_w"])
            self.assertEqual(
                result["measurement_coverage"]["peak_temperature_c"],
                {"status": "PARTIAL", "valid_samples": 1, "unavailable_samples": 1, "unavailable_reasons": {"n/a": 1}},
            )
            self.assertEqual(result["measurement_coverage"]["peak_power_w"]["status"], "UNAVAILABLE")
            self.assertEqual(result["measurement_coverage"]["peak_power_w"]["unavailable_samples"], 2)
        self.assertEqual(interval["status"], "RECORDED")
        # Missing optional columns are explicitly unavailable, not fabricated zeroes.
        text = "timestamp,index,uuid,memory.used [MiB],utilization.gpu [%]\n2026/09/12 06:17:26.305,2,GPU-fixture,1 MiB,0 %\n"
        result = self.monitored_interval(text)
        self.assertIsNone(result["peak_power_w"])
        self.assertEqual(result["measurement_coverage"]["peak_power_w"]["unavailable_reasons"], {"MISSING": 1})
        for missing in ("N/A", "[Not Supported]", ""):
            interval = self.monitored_interval(text.replace("1 MiB", missing))
            self.assertEqual(interval["status"], "UNAVAILABLE")
            with self.assertRaisesRegex(ValueError, "no finite"):
                read_monitor_samples(self.root / "gpu.csv", "gpu", ("2",), timestamp_utc_offset_seconds=0)

    def test_authenticated_failed_monitor_still_prevents_reuse(self):
        executor = self.executor()
        monitors = []
        for name in ("gpu", "processes"):
            path = executor.attempt_root / f"{name}.csv"
            path.write_text("fixture")
            monitors.append(
                {
                    "name": name,
                    "status": "FAILED",
                    "path": str(path),
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
        manifest = executor.attempt_root / "monitor.json"
        write_json(manifest, monitors)
        record = {
            "monitor_root": str(executor.attempt_root),
            "monitor_ref": {"path": str(manifest), "bytes": manifest.stat().st_size, "sha256": sha256_file(manifest)},
        }
        with self.assertRaisesRegex(ValueError, "failed or external monitoring"):
            executor._validate_monitor(record)

    def throughput_records(self):
        start = datetime(2026, 9, 12, tzinfo=timezone.utc)
        records = []
        for concurrency, replicate, life, training in (
            (1, 0, (0, 10), (2, 8)),
            (1, 1, (10, 20), (12, 18)),
            (2, 0, (20, 32), (22, 28)),
            (2, 1, (20, 32), (23, 29)),
        ):
            record = {
                "spec": {
                    "job": {
                        "stage": "throughput",
                        "mode": "fp32_strict",
                        "concurrency": concurrency,
                        "replicate": replicate,
                        "gpu": "3",
                    }
                },
                "status": "COMPLETE",
                "result": {"successful_updates": 294},
            }
            for boundary, seconds in zip(("started", "finished"), life, strict=True):
                record[f"{boundary}_at_utc"] = (start + timedelta(seconds=seconds)).isoformat()
            for boundary, seconds in zip(("started", "finished"), training, strict=True):
                record["result"][f"training_{boundary}_at_utc"] = (start + timedelta(seconds=seconds)).isoformat()
            records.append(record)
        return records

    def test_throughput_rejects_missing_naive_reversed_and_nonoverlapping_intervals(self):
        for problem in ("missing", "naive", "reversed", "nonoverlap", "outside_lifecycle"):
            with self.subTest(problem=problem):
                records = self.throughput_records()
                if problem == "missing":
                    records[0]["result"].pop("training_started_at_utc")
                elif problem == "naive":
                    records[0]["result"]["training_started_at_utc"] = "2026-09-12T00:00:02"
                elif problem == "reversed":
                    # The other positive interval must not hide this negative one.
                    records[0]["result"]["training_finished_at_utc"] = "2026-09-12T00:00:01+00:00"
                elif problem == "nonoverlap":
                    records[3]["result"]["training_started_at_utc"] = "2026-09-12T00:00:28+00:00"
                else:
                    records[0]["result"]["training_started_at_utc"] = "2026-09-11T23:59:59+00:00"
                summary = throughput_summary(records)[0]
                branch = summary["concurrency_2" if problem == "nonoverlap" else "concurrency_1"]
                self.assertEqual(
                    branch["status"], "INCOMPLETE_CONCURRENCY" if problem == "nonoverlap" else "INCOMPLETE_TIMING"
                )
                self.assertNotIn("two_process_throughput_ratio", summary)
                self.assertNotIn("two_process_training_throughput_ratio", summary)
                self.assertNotIn("updates_per_second", branch)

    def test_device_peaks_use_original_monitor_and_explicit_timezone_within_each_window(self):
        records = self.throughput_records()
        monitor_root = self.root / "original_attempt"
        monitor_root.mkdir()
        csv_path = monitor_root / "gpu.csv"
        header = "timestamp, index, uuid, memory.used [MiB], utilization.gpu [%], temperature.gpu, power.draw [W]\n"
        csv_path.write_text(
            header
            + "2026/09/12 02:59:59.000, 3, GPU-fixture, 99999, 100, 99, 999\n"
            + "2026/09/12 03:00:01.000, 3, GPU-fixture, 400, 95, 72, 250\n"
            + "2026/09/12 03:00:03.000, 3, GPU-fixture, 100, 80, 60, 180\n"
            + "2026/09/12 03:00:13.000, 3, GPU-fixture, 150, 85, 65, 190\n"
            + "2026/09/12 03:00:21.000, 3, GPU-fixture, 500, 99, 76, 290\n"
            + "2026/09/12 03:00:24.000, 3, GPU-fixture, 300, 98, 74, 280\n"
            + "2026/09/12 03:00:24.000, 2, GPU-other, 77777, 100, 99, 999\n"
        )
        stream = {
            "name": "gpu",
            "status": "RECORDED",
            "path": str(csv_path),
            "bytes": csv_path.stat().st_size,
            "sha256": sha256_file(csv_path),
            "timestamp_utc_offset_seconds": 10800,
        }
        manifest = monitor_root / "monitor.json"
        write_json(manifest, [stream])
        for record in records:
            record.update(
                reused=True,
                monitor_root=str(monitor_root),
                monitor_ref={"path": str(manifest), "sha256": sha256_file(manifest), "bytes": manifest.stat().st_size},
            )
        summary = throughput_summary(records)[0]
        first, second = summary["concurrency_1"], summary["concurrency_2"]
        self.assertEqual(first["device_training_metrics"]["peak_memory_used_mib"], 150)
        self.assertEqual(first["device_lifecycle_metrics"]["peak_memory_used_mib"], 400)
        self.assertEqual(second["device_training_metrics"]["peak_memory_used_mib"], 300)
        self.assertEqual(second["device_lifecycle_metrics"]["peak_memory_used_mib"], 500)
        self.assertEqual(second["device_training_metrics"]["samples"], 1)
        self.assertEqual(first["device_training_metrics"]["first_sample_utc"], "2026-09-12T00:00:03+00:00")
        unavailable = copy.deepcopy(records)
        for record in unavailable:
            end = datetime.fromisoformat(record["result"]["training_finished_at_utc"])
            record["result"]["training_started_at_utc"] = (end - timedelta(seconds=1)).isoformat()
        first = throughput_summary(unavailable)[0]["concurrency_1"]
        self.assertEqual(first["device_training_metrics"]["status"], "UNAVAILABLE")
        self.assertEqual(first["device_training_metrics"]["samples"], 0)
        self.assertNotIn("peak_memory_used_mib", first["device_training_metrics"])


@unittest.skipUnless(BASH, "Bash is required")
class ComparisonShellTest(unittest.TestCase):
    def test_u0_adoption_uses_comparison_disk_budget(self):
        setup = """
set -Eeuo pipefail
PHASE=compare-precision
SEEDS=(0 1 2)
GPUS=(2 3)
ACTIVE_PIDS=()
GIT_ARGS=()
PROTOCOL_ARGS=()
DATA_ARGS=()
HEAVY_ROOT=heavy
CHECKPOINT_ROOT=heavy/checkpoints
LOG_ROOT=logs
run_cli() { printf '%s\n' "$*" >> calls; }
sleep() { command sleep .01; }
"""
        script = setup + "\n".join(
            shell_function(name)
            for name in ("run_logged", "wait_for_batch", "terminate_active_children", "train_u0_phase")
        )
        script += "\ntrain_u0_phase\n"
        with tempfile.TemporaryDirectory() as directory:
            result = subprocess.run(
                [BASH, "-s"], input=script, text=True, cwd=directory, capture_output=True, timeout=15
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            calls = (Path(directory) / "calls").read_text().splitlines()
            self.assertTrue(any("--phase comparison" in line for line in calls))
            self.assertFalse(any("--phase source" in line for line in calls))

    def test_comparison_phase_never_enters_production_training_or_evaluation(self):
        with tempfile.TemporaryDirectory() as directory:
            setup = """
set -Eeuo pipefail
SEEDS=(0 1 2)
ACTIVE_PIDS=()
PROTOCOL_ARGS=()
DATA_ARGS=()
CHECKPOINT_ROOT=checkpoints
COMPACT_ROOT=compact
HEAVY_ROOT=heavy
LOG_ROOT=logs
RUN_ID=fixture
GPU_LIST=2,3
PRECISION_SOURCE_ROOT=historical
PRECISION_CAPTURE_ROOT=captures
run_cli() { printf '%s\n' "$*" >> calls; }
prepare_phase() { echo prepare >> calls; }
import_u0_phase() { echo import >> calls; }
train_u0_phase() { echo adopt >> calls; }
sleep() { command sleep .01; }
"""
            script = setup + "\n".join(
                shell_function(name)
                for name in ("run_logged", "wait_for_batch", "terminate_active_children", "compare_precision_phase")
            )
            script += "\ncompare_precision_phase\n"
            result = subprocess.run(
                [BASH, "-s"], input=script, text=True, cwd=directory, capture_output=True, timeout=15
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            calls = (Path(directory) / "calls").read_text().splitlines()
            self.assertEqual(calls[:3], ["prepare", "import", "adopt"])
            self.assertEqual(sum(value.startswith("init-controller") for value in calls), 3)
            comparison = [value for value in calls if value.startswith("compare-precision")]
            self.assertEqual(len(comparison), 1)
            self.assertIn("--gpu-list 2,3", comparison[0])
            self.assertFalse(
                any(value.startswith(("train-controller", "decide", "evaluate", "freeze-training")) for value in calls)
            )
            self.assertIn('elif [[ "$PHASE" == "import-u0" || "$PHASE" == "compare-precision" ]]', SOURCE)


if __name__ == "__main__":
    unittest.main()
