"""Conservative diagnostic aggregation and single-layer compact ZIP export.

This module uses only the standard library so failed CUDA workers cannot prevent
collection of their logs. A completed diagnostic never validates full training.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import zipfile
from pathlib import Path
from typing import Any

from tools.analysis.stage5 import precision_contract as contract
from tools.analysis.stage5.precision_contract import JOBS, PROBE_SCALES, SOURCE_HEAD, SOURCE_RUN, WORKER_SCHEMA

COMPACT_EXTENSIONS = frozenset({".json", ".jsonl", ".log", ".txt", ".tsv", ".csv", ".md", ".sh"})
MAX_COMPACT_FILE_BYTES = 64 * 1024 * 1024


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _positive_number(value: Any) -> bool:
    return type(value) in (int, float) and math.isfinite(value) and value > 0


def _completed_updates(record: dict, expected: int) -> bool:
    return (
        record.get("status") == "COMPLETE"
        and type(record.get("expected_updates")) is int
        and record["expected_updates"] == expected
        and type(record.get("completed_updates")) is int
        and record["completed_updates"] == expected
    )


def _benchmark_summary(record: dict) -> dict | None:
    if record.get("status") != "COMPLETE" or record.get("paired_inputs") is not True:
        return None
    modes = {}
    for mode in ("fp16", "fp32_strict"):
        values = record.get(mode, {})
        if not isinstance(values, dict):
            return None
        timings = values.get("full_step_seconds", [])
        count = values.get("completed_steps")
        memory = values.get("peak_allocated_bytes")
        gradient_status = values.get("gradient_status", [])
        if (
            not isinstance(timings, list)
            or type(count) is not int
            or count < contract.BENCHMARK_REPEATS
            or len(timings) != count
            or not all(_positive_number(value) for value in timings)
            or not _positive_number(memory)
            or not isinstance(gradient_status, list)
            or len(gradient_status) != count
            or not all(isinstance(status, str) for status in gradient_status)
        ):
            return None
        modes[mode] = {
            "measured_steps": count,
            "median_full_step_seconds": statistics.median(timings),
            "peak_allocated_bytes": memory,
            "gradient_status": gradient_status,
        }
        compute = values.get("compute_seconds")
        if compute is not None:
            if not isinstance(compute, list) or len(compute) != count or not all(_positive_number(v) for v in compute):
                return None
            modes[mode]["median_compute_seconds"] = statistics.median(compute)
        if record.get("memory_kind") is not None:
            for key in ("mode_peak_allocated_bytes", "mode_start_allocated_bytes", "mode_incremental_peak_bytes"):
                series = values.get(key)
                if (
                    not isinstance(series, list)
                    or len(series) != count
                    or any(type(v) is not int or v < 0 for v in series)
                ):
                    return None
                modes[mode][key] = max(series)
    modes["fp32_to_fp16_full_step_time_ratio"] = (
        modes["fp32_strict"]["median_full_step_seconds"] / modes["fp16"]["median_full_step_seconds"]
    )
    modes["fp32_to_fp16_full_step_peak_ratio"] = (
        modes["fp32_strict"]["peak_allocated_bytes"] / modes["fp16"]["peak_allocated_bytes"]
    )
    modes["timing_kind"] = record.get("timing_kind", "SHARED_PREPARATION_PLUS_MEASURED_MODE_STEP")
    modes["memory_kind"] = record.get("memory_kind", "LEGACY_FULL_STEP_MAX_INCLUDING_SHARED_PREPARATION")
    modes["memory_breakdown_status"] = "RECORDED" if record.get("memory_kind") else "NOT_RECORDED"
    for key, ratio_key in (
        ("median_compute_seconds", "fp32_to_fp16_compute_time_ratio"),
        ("mode_peak_allocated_bytes", "fp32_to_fp16_mode_peak_ratio"),
        ("mode_incremental_peak_bytes", "fp32_to_fp16_incremental_peak_ratio"),
    ):
        first, second = modes["fp16"].get(key), modes["fp32_strict"].get(key)
        modes[ratio_key] = second / first if first is not None and first > 0 and second is not None else None
    if record.get("memory_kind"):
        count = record["fp16"]["completed_steps"]
        preparation = record.get("preparation_seconds", [])
        peak = record.get("preparation_peak_allocated_bytes", [])
        if (
            not isinstance(preparation, list)
            or not isinstance(peak, list)
            or len(preparation) != count
            or len(peak) != count
            or not all(_positive_number(v) for v in [*preparation, *peak])
        ):
            return None
        modes["preparation"] = {"median_seconds": statistics.median(preparation), "peak_allocated_bytes": max(peak)}
    modes["interpretation"] = (
        "Full-step cost includes shared U0/feature preparation. A full-step memory ratio of 1 does not imply equal controller memory. "
        "Mode peaks include live inputs/U0; incremental peaks subtract allocations live at mode entry. Allocated bytes are not reserved/device-wide memory."
    )
    return modes


def _coverage_errors(report: dict) -> list[str]:
    coverage = report.get("coverage", {})
    if not isinstance(coverage, dict):
        return ["coverage: invalid coverage object"]
    cases = coverage.get("cases", [])
    expected = contract.coverage_cases()
    if coverage.get("expected_cases") != len(expected) or not isinstance(cases, list):
        return ["coverage: invalid expected case matrix"]
    observed = []
    errors = []
    for case in cases:
        if not isinstance(case, dict):
            errors.append("coverage: invalid case record")
            continue
        seed, variant = case.get("seed"), case.get("variant")
        if type(seed) is not int or not isinstance(variant, str):
            errors.append("coverage: invalid seed or variant")
            continue
        observed.append((seed, variant))
        if not _completed_updates(case, contract.COVERAGE_UPDATES):
            errors.append(f"coverage: {seed}/{variant} did not complete {contract.COVERAGE_UPDATES} FP32 updates")
    if len(observed) != len(expected) or set(observed) != expected:
        errors.append("coverage: missing, duplicate, or unexpected seed/variant cases")
    return errors


def aggregate_reports(output_root: Path, *, expected_git_head: str, source_run: str, runner_exit_code: int) -> dict:
    """Classify only evidence present in all four validated worker reports."""
    reports = {}
    errors = []
    findings = []
    if source_run != SOURCE_RUN:
        errors.append("Unsupported source run")
    if len(expected_git_head) != 40 or any(character not in "0123456789abcdef" for character in expected_git_head):
        errors.append("Invalid diagnostic Git SHA")
    for job in JOBS:
        path = output_root / f"{job}.json"
        try:
            report = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(report, dict):
                raise ValueError("Worker report must be an object")
        except (OSError, ValueError) as exc:
            errors.append(f"{job}: missing or unreadable report ({exc})")
            continue
        reports[job] = report
        expected = {
            "schema": WORKER_SCHEMA,
            "job": job,
            "source_run": source_run,
            "source_git_head": SOURCE_HEAD,
            "diagnostic_git_head": expected_git_head,
            "diagnostic_only": True,
            "labels_accessed": False,
            "production_checkpoint_written": False,
            "source_bytes_unchanged": True,
        }
        for key, value in expected.items():
            if report.get(key) != value or type(report.get(key)) is not type(value):
                errors.append(f"{job}: invalid {key}")
        if "workload_contract" in report and report["workload_contract"] != contract.workload_contract():
            errors.append(f"{job}: workload contract differs from this diagnostic revision")
    invalid_provenance = bool(errors)
    for job, report in reports.items():
        if report.get("status") != "DIAGNOSTIC_COMPLETE":
            errors.append(f"{job}: worker did not complete")
    source_changed = any(report.get("source_bytes_unchanged") is False for report in reports.values())
    math_failure = False
    resource_limit = False
    failed_reproduction = []
    benchmarks = {}
    mixed_observations = {}
    for job in JOBS[:3]:
        report = reports.get(job, {})
        comparison = report.get("comparison", {})
        if not isinstance(comparison, dict):
            errors.append(f"{job}: invalid comparison object")
            continue
        if comparison.get("status") != "COMPLETE":
            errors.append(f"{job}: comparison incomplete")
        audits = ("ncc_audit", "component_gradient_audit")
        for audit in audits:
            audit_record = comparison.get(audit, {})
            status = audit_record.get("status") if isinstance(audit_record, dict) else None
            if status == "FAIL":
                math_failure = True
                findings.append(f"{job}: {audit} failed")
            elif status != "PASS":
                errors.append(f"{job}: {audit} incomplete")
        probes = comparison.get("probes", [])
        if not isinstance(probes, list):
            probes = []
        reference = [
            probe
            for probe in probes
            if isinstance(probe, dict) and probe.get("mode") == "fp32_strict" and probe.get("scale") == 1
        ]
        if len(reference) != 1:
            errors.append(f"{job}: missing or ambiguous strict FP32 reference")
        elif reference[0].get("status") == "OOM":
            resource_limit = True
            findings.append(f"{job}: strict FP32 reference exceeded GPU memory")
        elif reference[0].get("status") in {"NONFINITE_GRADIENT", "MISSING_GRADIENT", "MATH_ERROR"}:
            math_failure = True
            findings.append(f"{job}: strict FP32 reference failed numerically")
        elif reference[0].get("status") != "FINITE":
            errors.append(f"{job}: strict FP32 reference incomplete")
        mixed = [probe for probe in probes if isinstance(probe, dict) and probe.get("mode") == "fp16"]
        scales = [probe.get("scale") for probe in mixed]
        if len(scales) != len(PROBE_SCALES) or any(scales.count(scale) != 1 for scale in PROBE_SCALES):
            errors.append(f"{job}: missing or duplicate FP16 scale probes")
        mixed_observations[job] = [
            {
                "scale": probe.get("scale"),
                "status": probe.get("status"),
                "parameter_gradient_comparison": probe.get("parameter_gradient_comparison"),
                "requested_delta_gradient_comparison": probe.get("requested_delta_gradient_comparison"),
            }
            for probe in mixed
        ]
        if any(
            probe.get("status") not in {"FINITE", "NONFINITE_GRADIENT", "MISSING_GRADIENT", "MATH_ERROR", "OOM"}
            for probe in mixed
        ):
            errors.append(f"{job}: FP16 scale probe incomplete")
        for probe in mixed:
            if probe.get("status") != "FINITE":
                continue
            for name in ("parameter_gradient_comparison", "requested_delta_gradient_comparison"):
                difference = probe.get(name, {})
                if not isinstance(difference, dict) or difference.get("status") != "FINITE":
                    errors.append(f"{job}: finite FP16 probe lacks {name}")
        if report.get("failure_reproduced") is not True:
            failed_reproduction.append(job)
        replay = report.get("replay", {})
        if not isinstance(replay, dict) or replay.get("status") not in {"FAILURE_REPRODUCED", "NOT_REPRODUCED"}:
            errors.append(f"{job}: failure replay incomplete")
        elif report.get("failure_reproduced") is True and replay["status"] != "FAILURE_REPRODUCED":
            errors.append(f"{job}: contradictory replay verdict")
        trajectory = report.get("fp32_trajectory", {})
        if not isinstance(trajectory, dict) or not _completed_updates(trajectory, contract.fp32_updates()):
            errors.append(f"{job}: FP32 trajectory did not complete {contract.fp32_updates()} updates")
        if isinstance(trajectory, dict):
            failure = trajectory.get("failure", {})
            if not isinstance(failure, dict):
                failure = {}
            if failure.get("kind") == "NUMERICAL":
                math_failure = True
                findings.append(f"{job}: FP32 trajectory failed numerically")
            elif failure.get("kind") == "OOM":
                resource_limit = True
                findings.append(f"{job}: FP32 trajectory exceeded GPU memory")
        benchmark = report.get("benchmark", {})
        if isinstance(benchmark, dict):
            fp32_benchmark = benchmark.get("fp32_strict", {})
            if isinstance(fp32_benchmark, dict) and any(
                status != "FINITE" for status in fp32_benchmark.get("gradient_status", [])
            ):
                math_failure = True
                findings.append(f"{job}: strict FP32 benchmark update failed numerically")
        summary = _benchmark_summary(benchmark) if isinstance(benchmark, dict) else None
        if summary is None:
            errors.append(f"{job}: paired full-step benchmark incomplete")
        else:
            benchmarks[job] = summary
    errors.extend(_coverage_errors(reports.get("coverage", {})))
    coverage = reports.get("coverage", {}).get("coverage", {})
    cases = coverage.get("cases", []) if isinstance(coverage, dict) else []
    if isinstance(cases, list):
        for case in cases:
            failure = case.get("failure", {}) if isinstance(case, dict) else {}
            kind = failure.get("kind") if isinstance(failure, dict) else None
            if kind == "NUMERICAL":
                math_failure = True
                findings.append(f"coverage: seed={case.get('seed')} variant={case.get('variant')} failed numerically")
            elif kind == "OOM":
                resource_limit = True
                findings.append(f"coverage: seed={case.get('seed')} variant={case.get('variant')} exceeded GPU memory")
    if runner_exit_code != 0:
        errors.append(f"Runner exited with code {runner_exit_code}")

    if source_changed:
        branch = "SOURCE_INTEGRITY_FAILURE"
    elif invalid_provenance:
        branch = "INCOMPLETE"
    elif math_failure:
        branch = "FIX_NUMERICS_BEFORE_TRAINING"
    elif resource_limit:
        branch = "RESOURCE_LIMIT_FP32"
    elif errors:
        branch = "INCOMPLETE"
    elif failed_reproduction:
        branch = "FAILURE_NOT_REPRODUCED"
    else:
        branch = "FP32_CANDIDATE_COST_REVIEW"
    return {
        "schema": "ctcf-stage5-precision-summary-v1",
        "branch": branch,
        "status": "INCOMPLETE" if errors else "DIAGNOSTIC_COMPLETE",
        "diagnostic_git_head": expected_git_head,
        "source_run": source_run,
        "source_git_head": SOURCE_HEAD,
        "workload_contract": contract.workload_contract(),
        "runner_exit_code": runner_exit_code,
        "errors": errors,
        "observed_findings": findings,
        "failures_not_reproduced": failed_reproduction,
        "benchmarks": benchmarks,
        "mixed_precision_observations": mixed_observations,
        "cost_acceptable": "USER_DECISION_REQUIRED",
        "production_training_validated": False,
        "production_restart_authorized": False,
        "development_evaluation_authorized": False,
        "limitations": [
            "Numerical probes and short trajectories do not prove full-training stability or registration quality.",
            "Incomplete reports never authorize a production restart, even if a subset passed.",
            "CUDA replay is not guaranteed bitwise identical to the historical failed trajectory.",
            "Timing ratios describe this paired diagnostic workload and GPU contention only.",
            "A precise cause requires reviewing tensor traces, NCC audits, and gradient comparisons in worker reports.",
            "The component audit checks finite derivatives and weighted gradient sums; it is not an independent oracle for every operator.",
        ],
        "worker_report_sha256": {job: sha256_file(output_root / f"{job}.json") for job in reports},
    }


def _compact_files(root: Path) -> list[Path]:
    if root.is_symlink() or not root.is_dir():
        raise ValueError("Compact root must be a real directory")
    files = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"Symlinks are forbidden in compact output: {path}")
        if path.is_dir():
            continue
        relative = path.relative_to(root).as_posix()
        if "\n" in relative or "\r" in relative or "\\" in relative:
            raise ValueError(f"Ambiguous archive path: {relative!r}")
        if path.name == "SHA256SUMS":
            raise FileExistsError("SHA256SUMS already exists; refusing to overwrite a packaged attempt")
        if not path.is_file() or path.suffix.lower() not in COMPACT_EXTENSIONS:
            raise ValueError(f"Non-compact file is forbidden in ZIP: {relative}")
        if path.stat().st_size > MAX_COMPACT_FILE_BYTES:
            raise ValueError(f"Compact file exceeds size limit: {relative}")
        files.append(path)
    return files


def package_compact_zip(output_root: Path, export_root: Path) -> tuple[Path, str]:
    """Create one root folder, internal hashes, and an external ZIP checksum.

    Refuse overwrites and non-compact/symlinked content. Diagnostic tensor
    captures must remain in a separate stage5_heavy directory.
    """
    files = [path.resolve() for path in _compact_files(output_root)]
    output_root = output_root.resolve()
    export_root = export_root.resolve()
    if export_root.is_relative_to(output_root):
        raise ValueError("Export directory must be outside compact output")
    export_root.mkdir(parents=True, exist_ok=True)
    archive = export_root / f"{output_root.name}.zip"
    sidecar = archive.with_suffix(".zip.sha256")
    if archive.exists() or sidecar.exists():
        raise FileExistsError("Diagnostic archive or checksum already exists")
    hashes = "".join(f"{sha256_file(path)}  {path.relative_to(output_root).as_posix()}\n" for path in files)
    hash_path = output_root / "SHA256SUMS"
    with hash_path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(hashes)
    with zipfile.ZipFile(archive, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as package:
        for path in [*files, hash_path]:
            member = f"{output_root.name}/{path.relative_to(output_root).as_posix()}"
            package.write(path, member)
    with zipfile.ZipFile(archive) as package:
        bad_member = package.testzip()
        if bad_member is not None:
            raise ValueError(f"ZIP verification failed: {bad_member}")
    digest = sha256_file(archive)
    with sidecar.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(f"{digest}  {archive.name}\n")
    return archive, digest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--export-root", required=True, type=Path)
    parser.add_argument("--expected-git-head", required=True)
    parser.add_argument("--source-run", required=True)
    parser.add_argument("--runner-exit-code", required=True, type=int)
    args = parser.parse_args(argv)
    summary = aggregate_reports(
        args.output_root,
        expected_git_head=args.expected_git_head,
        source_run=args.source_run,
        runner_exit_code=args.runner_exit_code,
    )
    with (args.output_root / "summary.json").open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(summary, handle, indent=2, allow_nan=False)
        handle.write("\n")
    report = [f"Diagnostic branch: {summary['branch']}", "", "Production training has not been validated.", ""]
    report.extend(f"- {item}" for item in summary["observed_findings"] + summary["errors"])
    report.extend(["", "Limitations:", *[f"- {item}" for item in summary["limitations"]]])
    with (args.output_root / "SUMMARY.md").open("x", encoding="utf-8", newline="\n") as handle:
        handle.write("\n".join(report) + "\n")
    archive, digest = package_compact_zip(args.output_root, args.export_root)
    print(f"[PRECISION DIAG BRANCH] {summary['branch']}")
    print(f"[PRECISION DIAG PACKAGE] {archive}")
    print(f"[PRECISION DIAG PACKAGE SIDECAR] {archive}.sha256")
    print(f"{digest}  {archive.name}")
    # A demonstrated numerical failure or resource limit is a diagnostic result.
    # Keep incomplete subchecks visible without calling successful packaging a failure.
    return int(summary["branch"] in {"INCOMPLETE", "SOURCE_INTEGRITY_FAILURE"})


if __name__ == "__main__":
    raise SystemExit(main())
