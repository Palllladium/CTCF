"""Describe comparison observations without selecting a training precision."""

from __future__ import annotations

import json
import statistics
from pathlib import Path


def _mean(values):
    return statistics.mean(values) if values else None


def _ratio(actual, reference):
    return actual / reference if actual is not None and reference is not None and reference > 0 else None


def _step_metrics(record):
    result = record.get("result", {})
    name = result.get("step_records")
    if not name:
        return []
    root = Path(record["output_root"])
    path = root / name
    if not path.resolve().is_relative_to(root.resolve()):
        raise ValueError("Step record path escapes worker output")
    rows = []
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            rows.append(json.loads(line))
    if len(rows) != result["successful_updates"]:
        raise ValueError("Step record count differs from worker result")
    return rows


def _telemetry_summary(rows):
    values = {}
    for row in rows:
        for group, measurements in row.get("telemetry", {}).get("groups", {}).items():
            for name, value in measurements.items():
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    values.setdefault(f"{group}.{name}", []).append((value, row["epoch"], row["pair"]))
    result = {}
    for name, observed in values.items():
        maximum = max(observed, key=lambda item: item[0])
        result[name] = {
            "mean": _mean([item[0] for item in observed]),
            "max": maximum[0],
            "max_at": {"epoch": maximum[1], "pair": maximum[2]},
        }
    return result


def summarize_comparison_results(records):
    """Records come from authenticated scheduler outputs, not arbitrary globbing."""
    paired, coverage, trajectories = [], [], []
    reference = {}
    for record in records:
        result = record.get("result") or {}
        job = record["spec"]["job"]
        stage = job["stage"]
        if stage == "paired":
            paired.append(
                {
                    "variant": job["variant"],
                    "status": result.get("status", "INCOMPLETE"),
                    "cases": result.get("cases", []),
                    "benchmark": result.get("benchmark"),
                }
            )
        elif stage == "coverage":
            coverage.append(
                {
                    "variant": job["variant"],
                    "seed": job["seed"],
                    "cases": [
                        {
                            key: case.get(key)
                            for key in (
                                "mode",
                                "status",
                                "successful_updates",
                                "expected_updates",
                                "failure",
                                "observed_arithmetic",
                            )
                        }
                        for case in result.get("cases", [])
                    ],
                }
            )
        elif stage == "trajectory":
            row = {
                "variant": job["variant"],
                "mode": job["mode"],
                "replicate": job["replicate"],
                "gpu": job["gpu"],
                "status": result.get("status", "INCOMPLETE"),
                "successful_updates": result.get("successful_updates", 0),
                "mean_full_step_seconds": _mean(result.get("full_step_seconds", [])),
                "mean_compute_seconds": _mean(result.get("compute_seconds", [])),
                "mean_preparation_seconds": _mean(result.get("preparation_seconds", [])),
                "mean_telemetry_seconds": _mean(result.get("telemetry_seconds", [])),
                "peak_allocated_bytes": result.get("peak_allocated_bytes"),
                "peak_reserved_bytes": result.get("peak_reserved_bytes"),
                "epochs": result.get("epochs", []),
                "initial_model_sha256": result.get("initial_model_sha256"),
                "final_model_sha256": result.get("final_model_sha256"),
            }
            trajectories.append((record, row))
            if job["mode"] == "fp32_strict" and job["replicate"] == 0:
                reference[job["variant"]] = (record, row)
    for record, row in trajectories:
        baseline = reference.get(row["variant"])
        if baseline is None:
            continue
        reference_record, reference_row = baseline
        if row["status"] != "COMPLETE" or reference_row["status"] != "COMPLETE":
            row["comparison_status"] = "INCOMPLETE_TRAJECTORY"
            continue
        if row["initial_model_sha256"] != reference_row["initial_model_sha256"] or row["gpu"] != reference_row["gpu"]:
            raise ValueError("Trajectory comparison does not share initialization and physical GPU")
        row["full_step_time_ratio_vs_strict"] = _ratio(
            row["mean_full_step_seconds"], reference_row["mean_full_step_seconds"]
        )
        row["compute_time_ratio_vs_strict"] = _ratio(row["mean_compute_seconds"], reference_row["mean_compute_seconds"])
        actual, expected = _step_metrics(record), _step_metrics(reference_record)
        row["telemetry"] = _telemetry_summary(actual)
        if len(actual) != len(expected):
            raise ValueError("Completed trajectory lengths differ")
        differences = {}
        for first, second in zip(actual, expected, strict=True):
            if first["pair"] != second["pair"] or first["epoch"] != second["epoch"]:
                raise ValueError("Trajectory pair schedules differ")
            if set(first["metrics"]) != set(second["metrics"]):
                raise ValueError("Trajectory metric inventories differ")
            for key in first["metrics"]:
                differences.setdefault(key, []).append(first["metrics"][key] - second["metrics"][key])
        row["trajectory_metric_difference_vs_strict"] = {
            key: {
                "mean_signed": _mean(values),
                "mean_absolute": _mean([abs(x) for x in values]),
                "max_absolute": max(abs(x) for x in values),
            }
            for key, values in differences.items()
        }
        row["comparison_status"] = "OBSERVED"
    return {
        "paired": paired,
        "coverage": coverage,
        "trajectories": [row for _, row in trajectories],
        "automatic_mode_selection": False,
        "training_quality_assessed": False,
        "limitations": [
            "FP32 is a numerical reference, not an exact mathematical oracle.",
            "A repeated FP32 trajectory measures observed repeatability, not a statistical noise bound.",
            "Changing trajectories changes evaluated states; paired probes isolate arithmetic on identical states.",
            "Two epochs cannot establish long-run stability, convergence, registration quality, or unseen memory peaks.",
            "TF32 settings grant kernel permission; they do not prove a particular kernel used Tensor Cores.",
        ],
    }


def render_comparison_report(summary):
    def number(value):
        return f"{value:.6g}" if isinstance(value, (int, float)) else "N/A"

    comparisons = summary.get("comparisons") or {}
    lines = [
        "# Stage5 precision comparison",
        "",
        f"Status: {summary['status']}",
        f"Jobs: {summary['recorded_jobs']}/{summary['expected_jobs']}",
        "",
        "Numerical and resource observations only. No mode is automatically selected and no diagnostic checkpoint is promoted.",
        "",
        "## Two-epoch trajectories",
        "",
        "Full step includes input preparation, controller update, capture-state copies and scalar telemetry. Report I/O is excluded; full training intervals include it. The first step also observes convolution dtypes.",
        "",
        "| Variant | Mode / repeat | Status | Full step s | Controller s | Preparation s | Peak allocated GiB | Peak reserved GiB | Time / strict |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in comparisons.get("trajectories", []):
        allocated, reserved = row.get("peak_allocated_bytes"), row.get("peak_reserved_bytes")
        lines.append(
            f"| {row['variant']} | {row['mode']} / {row['replicate']} | {row['status']} | {number(row.get('mean_full_step_seconds'))} | {number(row.get('mean_compute_seconds'))} | {number(row.get('mean_preparation_seconds'))} | {number(allocated / 2**30 if allocated is not None else None)} | {number(reserved / 2**30 if reserved is not None else None)} | {number(row.get('full_step_time_ratio_vs_strict'))} |"
        )
    lines.extend(
        [
            "",
            "## Identical-state numerical probes",
            "",
            "Relative differences are observations, not training-quality acceptance thresholds. Zero-reference comparisons have no relative ratio.",
            "",
            "| Variant | Mode / repeat | Status | Field relative L2 forward / reverse | Field-gradient relative L2 forward / reverse | Parameter-gradient relative L2 | AdamW-update relative L2 |",
            "| --- | --- | --- | --- | --- | ---: | ---: |",
        ]
    )
    for job in comparisons.get("paired", []):
        for case in job["cases"]:
            differences = case.get("difference_vs_strict", {})
            fields, gradients = differences.get("fields", {}), differences.get("field_gradients", {})
            field_text = " / ".join(
                number(fields.get(direction, {}).get("relative_l2")) for direction in ("forward", "reverse")
            )
            gradient_text = " / ".join(
                number(gradients.get(direction, {}).get("relative_l2")) for direction in ("forward", "reverse")
            )
            parameters = differences.get("parameter_gradients", {}).get("global", {}).get("relative_l2")
            updates = differences.get("parameter_updates", {}).get("global", {}).get("relative_l2")
            lines.append(
                f"| {job['variant']} | {case['mode']} / {case['replicate']} | {case['status']} | {field_text} | {gradient_text} | {number(parameters)} | {number(updates)} |"
            )
    lines.extend(
        [
            "",
            "## Warm identical-state controller benchmark",
            "",
            "Input preparation and numerical-observation hooks are excluded. The update retains production finite checks and pre-update state copies. Incremental allocated memory is relative to live prepared inputs, not total device memory.",
            "",
            "| Variant | Mode | Status | Mean controller step s | Incremental peak GiB | Controller time / strict |",
            "| --- | --- | --- | ---: | ---: | ---: |",
        ]
    )
    for job in comparisons.get("paired", []):
        cases = (job.get("benchmark") or {}).get("cases", {})
        reference = cases.get("fp32_strict", {})
        reference_seconds = _mean(reference.get("compute_seconds", []))
        for mode, case in cases.items():
            seconds = _mean(case.get("compute_seconds", []))
            peaks = case.get("incremental_peak_bytes", [])
            peak = max(peaks) / 2**30 if peaks else None
            ratio = (
                _ratio(seconds, reference_seconds)
                if case.get("status") == reference.get("status") == "COMPLETE"
                else None
            )
            lines.append(
                f"| {job['variant']} | {mode} | {case['status']} | {number(seconds)} | {number(peak)} | {number(ratio)} |"
            )
    lines.extend(
        [
            "",
            "## One versus two processes on one GPU",
            "",
            "Compare total successful updates per second for matched workloads. Lifecycle includes startup; training intervals begin after readiness gates. GPU/device-wide memory, utilization, temperature and power samples are in attempts/*/gpu.csv.",
            "",
            "| Mode | One process updates/s | Two processes updates/s | Training throughput ratio |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for row in summary.get("throughput", []):
        first, second = row.get("concurrency_1", {}), row.get("concurrency_2", {})
        lines.append(
            f"| {row['mode']} | {number(first.get('training_updates_per_second'))} | {number(second.get('training_updates_per_second'))} | {number(row.get('two_process_training_throughput_ratio'))} |"
        )
    lines.extend(
        [
            "",
            "Device-wide sampled peaks include other activity on the selected GPU. They are measured over each workload's own training or lifecycle intervals, including the original attempt for reused jobs. Sampling can miss brief peaks; CUDA allocator peaks are recorded separately in worker results.",
            "",
            "| Mode | Processes | Interval | Samples status | Peak used GiB | Peak GPU utilization % | Peak temperature C | Peak power W |",
            "| --- | ---: | --- | --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in summary.get("throughput", []):
        for concurrency in (1, 2):
            workload = row.get(f"concurrency_{concurrency}", {})
            for interval in ("training", "lifecycle"):
                metrics = workload.get(f"device_{interval}_metrics", {})
                memory = metrics.get("peak_memory_used_mib")
                lines.append(
                    f"| {row['mode']} | {concurrency} | {interval} | {metrics.get('status', 'UNAVAILABLE')} | {number(memory / 1024 if memory is not None else None)} | {number(metrics.get('peak_utilization_percent'))} | {number(metrics.get('peak_temperature_c'))} | {number(metrics.get('peak_power_w'))} |"
                )
    lines.extend(["", "## Scope", ""])
    lines.extend("- " + text for text in comparisons.get("limitations", []))
    lines.extend(
        [
            "- Per-step metrics and parameter telemetry are kept in each worker's separate steps_*.jsonl file.",
            "- Sources and compact worker artifacts are authenticated by execution manifests; numerical failures retain mode-specific heavy captures on the server.",
            "",
            "## Jobs",
            "",
            "| Job | GPU | Status |",
            "| --- | --- | --- |",
        ]
    )
    for record in summary.get("results", []):
        job = record["spec"]["job"]
        lines.append(f"| {job['job_id']} | {job['gpu']} | {record['status']} |")
    return "\n".join(lines) + "\n"
