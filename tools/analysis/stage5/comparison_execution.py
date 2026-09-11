"""Bounded precision experiments, with isolated workers and resumable evidence.

This scheduler never promotes a diagnostic checkpoint or enters Stage5 evaluation.
GPU numbers are execution resources, not seed identities. Comparisons for one
variant remain on one physical GPU within an attempt.
"""

from __future__ import annotations

import csv
import json
import math
import os
import re
import signal
import subprocess
import sys
import time
import uuid
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from tools.analysis.run_artifacts import atomic_write_text, sha256_file
from tools.analysis.stage5.comparison_contract import (
    MODES,
    PAIRED_MODES,
    SCHEMA as WORKER_SCHEMA,
    SEEDS,
    TRAJECTORY_VARIANTS,
    VARIANTS,
    expected_updates,
    workload_contract,
)
from tools.analysis.stage5.primitives import canonical_sha256, write_immutable_json

OBSERVED_STATUSES = frozenset(("COMPLETE", "CANDIDATE_FAILURE", "UNSUPPORTED"))
GATE_TIMEOUT_SECONDS = 900


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, value: Any) -> None:
    atomic_write_text(path, json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def parse_gpu_list(value: str) -> tuple[str, ...]:
    if re.fullmatch(r"(?:0|[1-9][0-9]*)(?:,(?:0|[1-9][0-9]*))*", value) is None:
        raise ValueError("GPU list must contain distinct non-negative integer indices")
    result = tuple(value.split(","))
    if len(set(result)) != len(result):
        raise ValueError("GPU list contains duplicate indices")
    return result


@dataclass(frozen=True)
class ComparisonJob:
    job_id: str
    stage: str
    variant: str
    seed: int
    mode: str
    replicate: int
    gpu: str
    concurrency: int = 1


def build_job_plan(gpus: tuple[str, ...]) -> list[ComparisonJob]:
    jobs = []
    for index, variant in enumerate(TRAJECTORY_VARIANTS):
        jobs.append(ComparisonJob(f"paired_{variant}", "paired", variant, 0, "fp32_strict", 0, gpus[index % len(gpus)]))
    for seed in SEEDS:
        for index, variant in enumerate(VARIANTS):
            jobs.append(
                ComparisonJob(
                    f"coverage_s{seed}_{variant}", "coverage", variant, seed, "fp32_strict", 0, gpus[index % len(gpus)]
                )
            )
    for index, variant in enumerate(TRAJECTORY_VARIANTS):
        for mode, replicate in PAIRED_MODES:
            jobs.append(
                ComparisonJob(
                    f"trajectory_{variant}_{mode}_r{replicate}",
                    "trajectory",
                    variant,
                    0,
                    mode,
                    replicate,
                    gpus[index % len(gpus)],
                )
            )
    for mode in MODES:
        for concurrency in (1, 2):
            for replicate in (0, 1):
                jobs.append(
                    ComparisonJob(
                        f"throughput_{mode}_c{concurrency}_r{replicate}",
                        "throughput",
                        "F0",
                        0,
                        mode,
                        replicate,
                        gpus[0],
                        concurrency,
                    )
                )
    return jobs


def _terminate(process: subprocess.Popen, *, force: bool = False) -> None:
    if process.poll() is not None:
        return
    try:
        if os.name == "posix":
            os.killpg(process.pid, signal.SIGKILL if force else signal.SIGTERM)
        elif force:
            process.kill()
        else:
            process.terminate()
    except ProcessLookupError:
        pass


class GpuMonitor:
    """Raw GPU and process CSV streams, separate from worker progress logs."""

    def __init__(self, root: Path, gpus: tuple[str, ...]):
        self.root, self.gpus = root, gpus
        self.processes = []
        self.streams = []
        self.records = []

    def start(self) -> None:
        queries = {
            "gpu": "--query-gpu=timestamp,index,uuid,name,utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu,power.draw",
            "processes": "--query-compute-apps=timestamp,gpu_uuid,pid,process_name,used_gpu_memory",
        }
        for name, query in queries.items():
            argv = ["nvidia-smi", f"--id={','.join(self.gpus)}", query, "--format=csv", "--loop=1"]
            # nvidia-smi prints local timestamps. Pin its process timezone so
            # recorded CSV times remain unambiguous across hosts and DST changes.
            env = {**os.environ, "TZ": "UTC"}
            record = {
                "name": name,
                "argv": argv,
                "path": str(self.root / f"{name}.csv"),
                "environment": {"TZ": "UTC"},
                "timestamp_timezone": "UTC",
                "timestamp_utc_offset_seconds": 0,
            }
            self.records.append(record)
            stream = (self.root / f"{name}.csv").open("w", encoding="utf-8")
            self.streams.append(stream)
            try:
                process = subprocess.Popen(
                    argv, stdout=stream, stderr=subprocess.STDOUT, env=env, start_new_session=os.name == "posix"
                )
            except OSError as exc:
                record.update(status="UNAVAILABLE", error=str(exc))
                continue
            self.processes.append((process, record))

    def stop(self) -> list[dict]:
        for process, record in self.processes:
            alive = process.poll() is None
            _terminate(process)
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                _terminate(process, force=True)
                process.wait()
            record.update(status="RECORDED" if alive else "FAILED", exit_code=process.returncode)
        for stream in self.streams:
            stream.close()
        for record in self.records:
            path = Path(record["path"])
            record.update(bytes=path.stat().st_size, sha256=sha256_file(path))
            try:
                record.update(read_monitor_samples(path, record["name"], self.gpus, timestamp_utc_offset_seconds=0))
            except (OSError, ValueError, KeyError) as exc:
                record.update(status="FAILED", sample_error=str(exc))
        write_json(self.root / "monitor.json", self.records)
        return self.records


def read_monitor_samples(
    path: Path, name: str, gpus: tuple[str, ...], *, timestamp_utc_offset_seconds: int | None = None
) -> dict:
    if timestamp_utc_offset_seconds is None:
        timestamp_utc_offset_seconds = int(datetime.now().astimezone().utcoffset().total_seconds())
    with path.open(encoding="utf-8") as stream:
        reader = csv.DictReader(stream, skipinitialspace=True)
        fields = reader.fieldnames or []
        required = (
            {"timestamp", "index", "uuid", "memory.used [MiB]", "utilization.gpu [%]"}
            if name == "gpu"
            else {"timestamp", "gpu_uuid", "pid"}
        )
        if not required.issubset(fields):
            raise ValueError("Monitoring CSV lacks required columns")
        rows = [row for row in reader if row.get("timestamp") != "timestamp"]
    if name == "processes":
        # Header-only is valid when no compute process is present at sample time.
        return {"sample_rows": len(rows), "sample_status": "RECORDED" if rows else "NO_COMPUTE_PROCESS_SAMPLED"}
    summaries = {}
    for row in rows:
        gpu = row["index"]
        if gpu not in gpus:
            raise ValueError("Monitoring CSV contains an undeclared GPU")
        datetime.strptime(row["timestamp"], "%Y/%m/%d %H:%M:%S.%f")
        if not str(row["uuid"]).startswith("GPU-"):
            raise ValueError("Monitoring CSV has no GPU identity")
        summary = summaries.setdefault(gpu, {"samples": 0, "uuid": row["uuid"], "first_sample_local": row["timestamp"]})
        summary["samples"] += 1
        summary["last_sample_local"] = row["timestamp"]
        for field, key, mandatory in (
            ("memory.used [MiB]", "peak_memory_used_mib", True),
            ("utilization.gpu [%]", "peak_utilization_percent", True),
            ("temperature.gpu", "peak_temperature_c", False),
            ("power.draw [W]", "peak_power_w", False),
        ):
            try:
                value = float(row[field])
                if not math.isfinite(value) or value < 0:
                    raise ValueError("Non-finite or negative GPU measurement")
            except (KeyError, TypeError, ValueError):
                if mandatory:
                    raise ValueError(f"Monitoring CSV has no finite {field}") from None
                summary.setdefault(key, None)
                continue
            summary[key] = max(value, summary.get(key) or 0)
    if set(summaries) != set(gpus):
        raise ValueError("Monitoring CSV has no data samples for every selected GPU")
    return {
        "sample_rows": len(rows),
        "sample_status": "RECORDED",
        "gpu_summary": summaries,
        "timestamp_timezone": str(timezone(timedelta(seconds=timestamp_utc_offset_seconds))),
        "timestamp_utc_offset_seconds": timestamp_utc_offset_seconds,
    }


class ComparisonExecutor:
    def __init__(self, args, *, context: dict, worker_prefix: list[str] | None = None):
        self.args = args
        self.context = context
        self.worker_prefix = worker_prefix or [
            sys.executable,
            "-m",
            "tools.analysis.compare_stage5_precision",
            "worker",
        ]
        self.active: dict[int, dict] = {}
        self.results: list[dict] = []
        self.verified_source_hashes: dict[str, str] = {}
        self.verified_monitors: set[str] = set()
        self.attempt_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "_" + uuid.uuid4().hex[:8]
        self.attempt_root = args.output_root / "attempts" / self.attempt_id
        self.attempt_root.mkdir(parents=True, exist_ok=False)
        self.interrupted = False

    def _spec(self, job: ComparisonJob) -> dict:
        return {"schema": "ctcf-stage5-comparison-execution-v1", "job": asdict(job), "context": self.context}

    def reusable(self, job: ComparisonJob) -> dict | None:
        spec = self._spec(job)
        root = self.args.output_root / "jobs" / job.job_id
        for path in sorted(root.glob("*/execution.json"), reverse=True):
            try:
                record = json.loads(path.read_text(encoding="utf-8"))
                result_path = path.with_name("result.json")
                if (
                    record.get("spec") != spec
                    or record.get("exit_code") != 0
                    or record.get("status") not in OBSERVED_STATUSES
                ):
                    continue
                if record.get("result_sha256") != sha256_file(result_path):
                    continue
                artifacts = record.get("artifacts")
                if not isinstance(artifacts, list) or not artifacts:
                    continue
                for artifact in artifacts:
                    member = path.parent / artifact["path"]
                    if not member.resolve().is_relative_to(path.parent.resolve()):
                        raise ValueError("Comparison artifact escapes its worker output")
                    if member.stat().st_size != artifact["bytes"] or sha256_file(member) != artifact["sha256"]:
                        raise ValueError("Comparison artifact changed after worker completion")
                result = json.loads(result_path.read_text(encoding="utf-8"))
                self._validate_result(job, result)
                self._validate_context(result)
                if "git_head" in self.context:
                    self._validate_monitor(record)
                for source, digest in result.get("source_hashes", {}).items():
                    if source not in self.verified_source_hashes:
                        self.verified_source_hashes[source] = sha256_file(Path(source))
                    if self.verified_source_hashes[source] != digest:
                        raise ValueError("Comparison source changed since the reusable job")
            except (OSError, ValueError, KeyError, TypeError):
                continue
            return {**record, "result": result, "reused": True}
        return None

    def _validate_monitor(self, record: dict) -> None:
        reference = record["monitor_ref"]
        root = Path(record["monitor_root"])
        path = Path(reference["path"])
        if path != root / "monitor.json" or not root.resolve().is_relative_to(self.args.output_root.resolve()):
            raise ValueError("Reused monitoring is outside its comparison attempt")
        key = str(path.resolve()) + ":" + reference["sha256"]
        if key in self.verified_monitors:
            return
        if path.stat().st_size != reference["bytes"] or sha256_file(path) != reference["sha256"]:
            raise ValueError("Reused monitoring attestation changed")
        monitors = json.loads(path.read_text(encoding="utf-8"))
        if {monitor["name"] for monitor in monitors} != {"gpu", "processes"}:
            raise ValueError("Reused job has incomplete monitoring streams")
        for monitor in monitors:
            member = Path(monitor["path"])
            if monitor["status"] != "RECORDED" or not member.resolve().is_relative_to(root.resolve()):
                raise ValueError("Reused job has failed or external monitoring")
            if monitor["name"] == "gpu" and (
                monitor.get("sample_status") != "RECORDED"
                or record["spec"]["job"]["gpu"] not in monitor.get("gpu_summary", {})
            ):
                raise ValueError("Reused job has no verified GPU samples")
            if member.stat().st_size != monitor["bytes"] or sha256_file(member) != monitor["sha256"]:
                raise ValueError("Reused monitoring stream changed")
        self.verified_monitors.add(key)

    def attach_monitoring(self) -> None:
        path = self.attempt_root / "monitor.json"
        reference = {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}
        for record in self.results:
            if record.get("reused"):
                continue
            record["monitor_ref"] = reference
            write_json(
                Path(record["output_root"]) / "execution.json",
                {key: value for key, value in record.items() if key != "result"},
            )

    def _validate_context(self, result: dict) -> None:
        for key in ("git_head", "run_id", "protocol_sha256"):
            if key in self.context and result.get(key) != self.context[key]:
                raise ValueError(f"Worker result context mismatch: {key}")
        if "git_head" in self.context and result.get("sources_unchanged") is not True:
            raise ValueError("Worker did not verify unchanged source artifacts")
        if "git_head" in self.context:
            if not isinstance(result.get("source_hashes"), dict) or not result["source_hashes"]:
                raise ValueError("Worker has no authenticated source inventory")
            for key in ("labels_accessed", "production_checkpoint_written", "automatic_mode_selection"):
                if result.get(key) is not False:
                    raise ValueError(f"Worker violated the comparison scope: {key}")

    @staticmethod
    def _validate_result(job: ComparisonJob, result: dict) -> None:
        if (
            not isinstance(result, dict)
            or result.get("schema") != WORKER_SCHEMA
            or result.get("status") not in OBSERVED_STATUSES
        ):
            raise ValueError("Worker result is incomplete or has an unknown schema/status")
        identity = asdict(job)
        for key in ("stage", "variant", "seed", "mode", "replicate"):
            if result.get(key) != identity[key]:
                raise ValueError(f"Worker result identity mismatch: {key}")
        if result.get("status") == "COMPLETE":
            expected, completed = result.get("expected_updates"), result.get("successful_updates")
            if type(expected) is not int or expected != expected_updates(job.stage) or completed != expected:
                raise ValueError("Worker reported COMPLETE without all required updates")

    def _launch(self, job: ComparisonJob, gate: Path | None) -> None:
        output = self.args.output_root / "jobs" / job.job_id / self.attempt_id
        heavy = self.args.heavy_root / "jobs" / job.job_id / self.attempt_id
        output.mkdir(parents=True, exist_ok=False)
        heavy.mkdir(parents=True, exist_ok=False)
        argv = list(self.worker_prefix)
        for key in ("stage", "variant", "seed", "mode", "replicate"):
            argv.extend(("--" + key, str(getattr(job, key))))
        for key in (
            "protocol",
            "data_contract",
            "image_root",
            "checkpoint_root",
            "run_id",
            "expected_git_head",
            "repo_root",
            "precision_source_root",
            "capture_root",
        ):
            argv.extend(("--" + key.replace("_", "-"), str(getattr(self.args, key))))
        argv.extend(("--output-root", str(output), "--heavy-root", str(heavy)))
        ready = output / "ready.json"
        if gate is not None:
            argv.extend(
                (
                    "--ready-file",
                    str(ready),
                    "--start-gate",
                    str(gate),
                    "--gate-timeout-seconds",
                    str(GATE_TIMEOUT_SECONDS),
                )
            )
        env = os.environ.copy()
        env.update(CUDA_VISIBLE_DEVICES=job.gpu, CUDA_DEVICE_ORDER="PCI_BUS_ID", PYTHONUNBUFFERED="1")
        record = {
            "spec": self._spec(job),
            "spec_sha256": canonical_sha256(self._spec(job)),
            "argv": argv,
            "environment": {
                "CUDA_VISIBLE_DEVICES": job.gpu,
                "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
                "PYTHONUNBUFFERED": "1",
            },
            "output_root": str(output),
            "heavy_root": str(heavy),
            "monitor_root": str(self.attempt_root),
            "started_at_utc": utc_now(),
            "status": "RUNNING",
            "reused": False,
        }
        write_json(output / "execution.json", record)
        log = (output / "worker.log").open("w", encoding="utf-8")
        try:
            process = subprocess.Popen(
                argv,
                cwd=self.args.repo_root,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=os.name == "posix",
            )
        except OSError as exc:
            log.close()
            record.update(status="INCOMPLETE", exit_code=None, error=str(exc), finished_at_utc=utc_now())
            write_json(output / "execution.json", record)
            self.results.append(record)
            return
        self.active[process.pid] = {
            "process": process,
            "job": job,
            "record": record,
            "log": log,
            "ready": ready,
            "started": time.monotonic(),
        }
        print(f"[COMPARISON START] {job.job_id} gpu={job.gpu} pid={process.pid}", flush=True)

    def _collect(self, pid: int) -> None:
        state = self.active.pop(pid)
        process, job, record = state["process"], state["job"], state["record"]
        process.wait()
        state["log"].close()
        output = Path(record["output_root"])
        record.update(
            exit_code=process.returncode,
            finished_at_utc=utc_now(),
            lifecycle_seconds=time.monotonic() - state["started"],
        )
        try:
            result = json.loads((output / "result.json").read_text(encoding="utf-8"))
            self._validate_result(job, result)
            self._validate_context(result)
            if process.returncode != 0:
                raise ValueError(f"Worker exited with status {process.returncode}")
            record.update(status=result["status"], result_sha256=sha256_file(output / "result.json"))
            record["artifacts"] = [
                {"path": str(path.relative_to(output)), "bytes": path.stat().st_size, "sha256": sha256_file(path)}
                for path in sorted(output.rglob("*"))
                if path.is_file() and path.name != "execution.json"
            ]
            record["result"] = result
        except (OSError, ValueError, TypeError) as exc:
            record.update(status="INCOMPLETE", error=str(exc))
        write_json(output / "execution.json", {key: value for key, value in record.items() if key != "result"})
        self.results.append(record)
        print(f"[COMPARISON RESULT] {job.job_id} {record['status']}", flush=True)

    def stop(self) -> None:
        for state in self.active.values():
            _terminate(state["process"])
        deadline = time.monotonic() + 20
        while any(state["process"].poll() is None for state in self.active.values()) and time.monotonic() < deadline:
            time.sleep(0.05)
        for state in self.active.values():
            _terminate(state["process"], force=True)
        for pid in list(self.active):
            self._collect(pid)

    def run_jobs(self, jobs: list[ComparisonJob], *, shared_gate: bool = False) -> None:
        cached = [self.reusable(job) for job in jobs]
        # A two-process throughput observation is indivisible: never combine a
        # reused member with a newly executed member and call it concurrent.
        if shared_gate:
            same_attempt = len({record["monitor_root"] for record in cached if record is not None}) == 1
            if not all(cached) or not same_attempt:
                cached = [None] * len(jobs)
        pending = []
        for job, record in zip(jobs, cached, strict=True):
            if record is None:
                pending.append(job)
            else:
                self.results.append(record)
                print(f"[COMPARISON REUSE] {job.job_id}", flush=True)
        gate = self.attempt_root / (jobs[0].job_id + "_start.json") if shared_gate and pending else None
        gate_open = False
        gate_started = time.monotonic()
        try:
            while pending or self.active:
                occupied = Counter(state["job"].gpu for state in self.active.values())
                for job in list(pending):
                    if occupied[job.gpu] >= (2 if shared_gate and len(jobs) == 2 else 1):
                        continue
                    self._launch(job, gate)
                    pending.remove(job)
                    occupied[job.gpu] += 1
                for pid, state in list(self.active.items()):
                    if state["process"].poll() is not None:
                        self._collect(pid)
                if gate is not None and not gate_open and not pending:
                    if all(state["ready"].is_file() for state in self.active.values()):
                        write_json(gate, {"started_at_utc": utc_now()})
                        gate_open = True
                    elif time.monotonic() - gate_started > GATE_TIMEOUT_SECONDS:
                        raise RuntimeError("Throughput workers did not become ready before the gate timeout")
                if self.active:
                    time.sleep(0.1)
        except BaseException:
            self.stop()
            raise


def _measurement_interval(record: dict, *, training: bool) -> tuple[float, float]:
    payload = record["result"] if training else record
    prefix = "training_" if training else ""
    times = [datetime.fromisoformat(payload[f"{prefix}{boundary}_at_utc"]) for boundary in ("started", "finished")]
    if any(value.tzinfo is None or value.utcoffset() is None for value in times):
        raise ValueError("Measurement timestamps require an explicit timezone")
    start, end = (value.timestamp() for value in times)
    if not math.isfinite(start) or not math.isfinite(end) or end <= start:
        raise ValueError("Every measured interval must have a finite end later than its start")
    return start, end


def _device_interval_metrics(records: list[dict], intervals: list[tuple[float, float]], cache: dict) -> dict:
    """Measure device peaks in these workers' original, authenticated time windows."""
    result = {"status": "UNAVAILABLE", "samples": 0, "sources": []}
    gpus = {record["spec"]["job"]["gpu"] for record in records}
    if len(gpus) != 1:
        return {**result, "reason": "Workers do not share one physical GPU"}
    gpu = next(iter(gpus))
    result["gpu"] = gpu
    selected = {}
    try:
        for record, (start, end) in zip(records, intervals, strict=True):
            reference = record["monitor_ref"]
            root = Path(record["monitor_root"])
            manifest = root / "monitor.json"
            if Path(reference["path"]) != manifest or sha256_file(manifest) != reference["sha256"]:
                raise ValueError("Original monitor attestation is missing or changed")
            key = (str(manifest.resolve()), reference["sha256"])
            if key not in cache:
                metadata = json.loads(manifest.read_text(encoding="utf-8"))
                stream = next(item for item in metadata if item["name"] == "gpu")
                path = Path(stream["path"])
                offset = stream["timestamp_utc_offset_seconds"]
                if type(offset) is not int or not -86400 < offset < 86400:
                    raise ValueError("Original GPU monitoring has no valid UTC offset")
                if stream["status"] != "RECORDED" or not path.resolve().is_relative_to(root.resolve()):
                    raise ValueError("Original GPU monitoring was not recorded")
                if path.stat().st_size != stream["bytes"] or sha256_file(path) != stream["sha256"]:
                    raise ValueError("Original GPU CSV changed")
                zone = timezone(timedelta(seconds=offset))
                samples = []
                with path.open(encoding="utf-8") as handle:
                    for index, row in enumerate(csv.DictReader(handle, skipinitialspace=True)):
                        if row.get("timestamp") == "timestamp":
                            continue
                        stamp = (
                            datetime.strptime(row["timestamp"], "%Y/%m/%d %H:%M:%S.%f").replace(tzinfo=zone).timestamp()
                        )
                        samples.append((index, stamp, row))
                cache[key] = (path, samples)
            path, samples = cache[key]
            if str(path) not in result["sources"]:
                result["sources"].append(str(path))
            for index, stamp, row in samples:
                if row["index"] == gpu and start <= stamp <= end:
                    selected[(str(path), index)] = (stamp, row)
        if not selected:
            return {**result, "reason": "No GPU samples fall within the measured worker windows"}
        times = []
        for stamp, row in selected.values():
            times.append(stamp)
            for field, key, mandatory in (
                ("memory.used [MiB]", "peak_memory_used_mib", True),
                ("utilization.gpu [%]", "peak_utilization_percent", True),
                ("temperature.gpu", "peak_temperature_c", False),
                ("power.draw [W]", "peak_power_w", False),
            ):
                try:
                    value = float(row[field])
                    if not math.isfinite(value) or value < 0:
                        raise ValueError("Invalid GPU measurement")
                except (KeyError, TypeError, ValueError):
                    if mandatory:
                        raise ValueError(f"No finite GPU measurement: {field}") from None
                    result.setdefault(key, None)
                    continue
                result[key] = max(value, result.get(key) or 0)
        result.update(
            status="RECORDED",
            samples=len(selected),
            first_sample_utc=datetime.fromtimestamp(min(times), timezone.utc).isoformat(),
            last_sample_utc=datetime.fromtimestamp(max(times), timezone.utc).isoformat(),
        )
        return result
    except (OSError, KeyError, TypeError, ValueError, StopIteration) as exc:
        return {"status": "UNAVAILABLE", "samples": 0, "gpu": gpu, "sources": result["sources"], "reason": str(exc)}


def throughput_summary(records: list[dict]) -> list[dict]:
    summary = []
    monitor_cache = {}
    for mode in MODES:
        row = {"mode": mode, "timing_kind": "two_matched_workers_including_startup_and_gate_wait"}
        for concurrency in (1, 2):
            selected = [
                record
                for record in records
                if record["spec"]["job"]["stage"] == "throughput"
                and record["spec"]["job"]["mode"] == mode
                and record["spec"]["job"]["concurrency"] == concurrency
            ]
            if len(selected) != 2 or any(record["status"] != "COMPLETE" for record in selected):
                row[f"concurrency_{concurrency}"] = {"status": "INCOMPLETE_OR_CANDIDATE_FAILURE"}
                continue
            measured = {"status": "INCOMPLETE_TIMING"}
            row[f"concurrency_{concurrency}"] = measured
            try:
                lifecycle = [_measurement_interval(record, training=False) for record in selected]
                training = [_measurement_interval(record, training=True) for record in selected]
                starts, ends = zip(*training, strict=True)
                measured["device_lifecycle_metrics"] = _device_interval_metrics(selected, lifecycle, monitor_cache)
                measured["device_training_metrics"] = _device_interval_metrics(selected, training, monitor_cache)
                if concurrency == 2 and max(starts) >= min(ends):
                    measured.update(status="INCOMPLETE_CONCURRENCY", concurrency_overlap_observed=False)
                    continue
                if any(
                    train_start < life_start or train_end > life_end
                    for (train_start, train_end), (life_start, life_end) in zip(training, lifecycle, strict=True)
                ):
                    raise ValueError("Training interval falls outside its worker lifecycle")
                seconds = (
                    sum(end - start for start, end in lifecycle)
                    if concurrency == 1
                    else max(end for _, end in lifecycle) - min(start for start, _ in lifecycle)
                )
                measured_seconds = (
                    sum(end - start for start, end in training) if concurrency == 1 else max(ends) - min(starts)
                )
                updates = sum(record["result"]["successful_updates"] for record in selected)
                measured.update(
                    status="COMPLETE",
                    lifecycle_seconds=seconds,
                    successful_updates=updates,
                    updates_per_second=updates / seconds,
                    training_interval_seconds=measured_seconds,
                    training_updates_per_second=updates / measured_seconds,
                )
                if concurrency == 2:
                    measured["concurrency_overlap_observed"] = True
            except (KeyError, TypeError, ValueError) as exc:
                measured.update(status="INCOMPLETE_TIMING", reason=str(exc))
        first, second = row.get("concurrency_1", {}), row.get("concurrency_2", {})
        if first.get("status") == second.get("status") == "COMPLETE":
            row["two_process_throughput_ratio"] = second["updates_per_second"] / first["updates_per_second"]
            if "training_updates_per_second" in first and "training_updates_per_second" in second:
                row["two_process_training_throughput_ratio"] = (
                    second["training_updates_per_second"] / first["training_updates_per_second"]
                )
        summary.append(row)
    return summary


def execute_comparison(args, *, protocol: dict) -> dict:
    gpus = parse_gpu_list(args.gpu_list)
    jobs = build_job_plan(gpus)
    context = {
        "git_head": args.expected_git_head,
        "run_id": args.run_id,
        "protocol_sha256": canonical_sha256(protocol),
        "protocol_file_sha256": sha256_file(args.protocol),
        "data_contract_file_sha256": sha256_file(args.data_contract),
        "precision_source_inventory_sha256": sha256_file(args.precision_source_root / "SHA256SUMS"),
        "workload": workload_contract(),
        "paths": {
            key: str(getattr(args, key).resolve())
            for key in ("image_root", "checkpoint_root", "precision_source_root", "capture_root")
        },
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    args.heavy_root.mkdir(parents=True, exist_ok=True)
    write_immutable_json(args.output_root / "comparison_contract.json", context)
    executor = ComparisonExecutor(args, context=context)
    write_json(executor.attempt_root / "job_plan.json", [asdict(job) for job in jobs])
    monitor = GpuMonitor(executor.attempt_root, gpus)
    previous_handlers = {}
    error = None

    def interrupted(signum, frame):
        if executor.interrupted:
            return
        executor.interrupted = True
        raise KeyboardInterrupt(f"Signal {signum}")

    try:
        for sig in (signal.SIGTERM, signal.SIGINT):
            previous_handlers[sig] = signal.signal(sig, interrupted)
        monitor.start()
        for stage in ("paired", "coverage", "trajectory"):
            executor.run_jobs([job for job in jobs if job.stage == stage])
        for mode in MODES:
            for concurrency in (1, 2):
                selected = [
                    job
                    for job in jobs
                    if job.stage == "throughput" and job.mode == mode and job.concurrency == concurrency
                ]
                if concurrency == 1:
                    for job in selected:
                        executor.run_jobs([job], shared_gate=True)
                else:
                    executor.run_jobs(selected, shared_gate=True)
    except BaseException as exc:
        error = {"type": type(exc).__name__, "message": str(exc)}
    finally:
        executor.stop()
        monitor_records = monitor.stop()
        executor.attach_monitoring()
        for sig, handler in previous_handlers.items():
            signal.signal(sig, handler)
    complete = len(executor.results) == len(jobs) and all(
        record["status"] in OBSERVED_STATUSES for record in executor.results
    )
    from tools.analysis.stage5.comparison_summary import render_comparison_report, summarize_comparison_results

    try:
        comparisons = summarize_comparison_results(executor.results)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        comparisons = None
        error = {"type": type(exc).__name__, "message": str(exc), "stage": "aggregation"}
    monitoring_complete = all(record.get("status") == "RECORDED" for record in monitor_records)
    throughput = throughput_summary(executor.results)
    if any(
        row.get(f"concurrency_{count}", {}).get("status") in {"INCOMPLETE_CONCURRENCY", "INCOMPLETE_TIMING"}
        for row in throughput
        for count in (1, 2)
    ):
        error = error or {
            "type": "IncompleteThroughputTiming",
            "message": "Worker timing or concurrency evidence is incomplete",
        }
    summary = {
        "schema": "ctcf-stage5-precision-comparison-summary-v1",
        "status": "DIAGNOSTIC_COMPLETE" if complete and error is None and monitoring_complete else "INCOMPLETE",
        "context": context,
        "attempt_id": executor.attempt_id,
        "error": error,
        "expected_jobs": len(jobs),
        "recorded_jobs": len(executor.results),
        "status_counts": dict(Counter(record["status"] for record in executor.results)),
        "automatic_mode_selection": False,
        "production_checkpoint_promotion": False,
        "registration_quality_claim": False,
        "labels_accessed": False,
        "monitoring": monitor_records,
        "throughput": throughput,
        "comparisons": comparisons,
        "results": [{key: value for key, value in record.items() if key != "result"} for record in executor.results],
    }
    write_json(executor.attempt_root / "summary.json", summary)
    write_json(args.output_root / "summary.json", summary)
    atomic_write_text(args.output_root / "report.md", render_comparison_report(summary))
    print(f"[PRECISION COMPARISON] {summary['status']} {args.output_root / 'summary.json'}", flush=True)
    return summary
