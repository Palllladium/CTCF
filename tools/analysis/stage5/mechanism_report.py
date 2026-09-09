"""Package mechanism observations without converting incomplete evidence into a fix."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from tools.analysis.stage5 import mechanism_contract as contract
from tools.analysis.stage5.precision_report import package_compact_zip, sha256_file


def _mapping(value):
    return value if isinstance(value, dict) else {}


def _finite_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _stats_valid(value):
    value = _mapping(value)
    count, bad = value.get("elements"), value.get("nonfinite")
    return (
        type(count) is int
        and count > 0
        and type(bad) is int
        and 0 <= bad <= count
        and isinstance(value.get("dtype"), str)
        and isinstance(value.get("shape"), list)
    )


def _comparison_valid(value):
    value = _mapping(value)
    if not isinstance(value.get("within_heuristic_tolerance"), bool) or not _mapping(value.get("tolerances")):
        return False
    if value.get("status") == "NONFINITE":
        return value["within_heuristic_tolerance"] is False
    return value.get("status") == "FINITE" and all(
        _finite_number(value.get(key)) for key in ("reference_l2", "actual_l2", "absolute_l2", "max_abs")
    )


def _direction_comparisons_valid(value):
    value = _mapping(value)
    return set(value) == {"forward", "reverse"} and all(_comparison_valid(item) for item in value.values())


def _state_errors(audit, key, expected):
    values = _mapping(audit.get(key))
    if not expected.issubset(values) or any(value is not True for value in values.values()):
        return [f"{key}: missing or failed preservation checks"]
    return []


def field_audit_errors(audit):
    """Check prescribed observations; finite disagreement remains valid evidence."""
    audit = _mapping(audit)
    errors = _state_errors(
        audit,
        "state_preserved",
        {
            "model_during_audit",
            "training_flags_during_audit",
            "input_versions",
            "parameter_gradients",
            "backend_flags",
            "rng_restored",
            "training_flags_restored",
        },
    )
    modes = {name: (autocast, tf32) for name, autocast, tf32 in contract.FIELD_MODES}
    outputs, points = _mapping(audit.get("controller_outputs")), _mapping(audit.get("field_points"))
    if set(outputs) != set(modes) or set(points) != set(modes):
        errors.append("field matrix: missing or unexpected controller mode/field point")
    baseline_name = contract.FIELD_MODES[0][0]
    common_status = _mapping(_mapping(_mapping(points.get(baseline_name)).get("evaluations")).get(baseline_name)).get(
        "status"
    )
    for name, (autocast, tf32) in modes.items():
        output = _mapping(outputs.get(name))
        flags = _mapping(output.get("backend_flags"))
        if (
            output.get("autocast") is not autocast
            or flags.get("cudnn_allow_tf32") is not tf32
            or flags.get("matmul_allow_tf32") is not False
            or flags.get("float32_matmul_precision") != "highest"
        ):
            errors.append(f"{name}: controller arithmetic flags missing or incorrect")
        stats = _mapping(output.get("field_stats"))
        if (
            output.get("status") not in {"FINITE", "NONFINITE"}
            or set(stats) != {"forward", "reverse"}
            or not all(_stats_valid(v) for v in stats.values())
        ):
            errors.append(f"{name}: missing controller field observations")
        point = _mapping(points.get(name))
        evaluations = _mapping(point.get("evaluations"))
        if set(evaluations) != set(modes):
            errors.append(f"{name}: objective arithmetic matrix incomplete")
        if not _direction_comparisons_valid(point.get("field_shift_vs_strict_controller")):
            errors.append(f"{name}: field-shift comparison missing")
        reference_status = _mapping(evaluations.get(baseline_name)).get("status")
        if (
            reference_status == "FINITE"
            and common_status == "FINITE"
            and not _direction_comparisons_valid(point.get("evaluation_point_effect"))
        ):
            errors.append(f"{name}: evaluation-point comparison missing")
        observations = [
            *((mode, _mapping(evaluations.get(mode)), *settings) for mode, settings in modes.items()),
            ("repeat", _mapping(point.get("repeat")), False, False),
        ]
        for mode, observation, outer_autocast, allow_tf32 in observations:
            label = f"{name}/{mode}"
            status = observation.get("status")
            if status == "MATH_ERROR":
                if not isinstance(observation.get("error"), str) or not observation["error"]:
                    errors.append(f"{label}: numerical failure lacks error evidence")
                continue
            if status not in {"FINITE", "NONFINITE"}:
                errors.append(f"{label}: observation incomplete ({status})")
                continue
            stats = _mapping(observation.get("gradient_stats"))
            flags = _mapping(observation.get("backend_flags"))
            if set(stats) != {"forward", "reverse"} or not all(_stats_valid(v) for v in stats.values()):
                errors.append(f"{label}: gradient observations missing")
            elif status == "FINITE" and any(v["nonfinite"] for v in stats.values()):
                errors.append(f"{label}: FINITE contradicts gradient counts")
            if (
                observation.get("outer_autocast") is not outer_autocast
                or observation.get("leaf_dtype") != "torch.float32"
                or flags.get("cudnn_allow_tf32") is not allow_tf32
                or flags.get("matmul_allow_tf32") is not False
                or flags.get("float32_matmul_precision") != "highest"
            ):
                errors.append(f"{label}: wrong or unrecorded arithmetic flags")
            comparison = "same_point_repeat_difference" if mode == "repeat" else "same_point_arithmetic_effect"
            if (
                status == "FINITE"
                and reference_status == "FINITE"
                and not _direction_comparisons_valid(observation.get(comparison))
            ):
                errors.append(f"{label}: same-point comparison missing")
    return errors


def bias_audit_errors(audit, backend):
    audit = _mapping(audit)
    errors = _state_errors(audit, "state_restored", {"rng", "backend", "prepared_input_versions", "model"})
    backend = _mapping(backend)
    boolean_flags = (
        "cudnn_allow_tf32",
        "matmul_allow_tf32",
        "cudnn_benchmark",
        "cudnn_deterministic",
        "deterministic_algorithms",
    )
    if any(type(backend.get(key)) is not bool for key in boolean_flags) or backend.get(
        "float32_matmul_precision"
    ) not in {"highest", "high", "medium"}:
        errors.append("worker execution backend flags missing")
    if _mapping(audit.get("original_backend_flags")) != backend:
        errors.append("bias baseline differs from worker execution backend flags")
    plan = contract.bias_modes({key: backend.get(key) is True for key in ("cudnn_allow_tf32", "matmul_allow_tf32")})
    probes = _mapping(audit.get("probes"))
    if set(probes) != {mode[0] for mode in plan}:
        errors.append("bias mode matrix incomplete")
    for name, fp32, scale, disable_tf32 in plan:
        probe = _mapping(probes.get(name))
        expected_flags = dict(backend)
        if disable_tf32:
            expected_flags.update(cudnn_allow_tf32=False, matmul_allow_tf32=False, float32_matmul_precision="highest")
        if _mapping(probe.get("backend_flags")) != expected_flags:
            errors.append(f"{name}: probe arithmetic flags differ from requested mode")
        if (
            probe.get("status") != "CAPTURED"
            or probe.get("mode") != ("fp32" if fp32 else "fp16")
            or probe.get("scale") != scale
        ):
            errors.append(f"{name}: bias probe missing or mode/scale mismatch")
        parameters = _mapping(probe.get("parameters"))
        if probe.get("missing_parameters") != [] or not _stats_valid(parameters.get("stem.0.bias")):
            errors.append(f"{name}: parameter gradients missing")
        directions = probe.get("bias_directions", [])
        if (
            not isinstance(directions, list)
            or len(directions) != 2
            or {_mapping(d).get("direction") for d in directions} != {"ab", "ba"}
        ):
            errors.append(f"{name}: bias directions incomplete")
            continue
        for direction in directions:
            label = f"{name}/{direction['direction']}"
            if direction.get("status") != "CAPTURED" or direction.get("node") != "ConvolutionBackward0":
                errors.append(f"{label}: convolution bias return not captured")
            if not all(_stats_valid(direction.get(key)) for key in ("upstream", "convolution_returned_bias")):
                errors.append(f"{label}: gradient boundary statistics missing")
            numbers = [direction.get(key) for key in ("sum_fp32", "sum_fp64", "convolution_returned_bias_values")]
            if (
                not all(
                    isinstance(row, list) and row and all(v is None or _finite_number(v) for v in row)
                    for row in numbers
                )
                or len({len(row) for row in numbers if isinstance(row, list)}) != 1
            ):
                errors.append(f"{label}: channel reductions/bias values missing")
            elif all(v is not None for v in numbers[1]) and not _comparison_valid(
                direction.get("returned_bias_vs_fp64_sum")
            ):
                errors.append(f"{label}: returned bias comparison missing")
        if not isinstance(probe.get("layer_gradients"), list) or not probe["layer_gradients"]:
            errors.append(f"{name}: layer gradient observations missing")
    comparisons = _mapping(audit.get("comparisons"))
    for actual, reference in contract.BIAS_COMPARISONS:
        if not {actual, reference}.issubset({mode[0] for mode in plan}):
            continue
        name = f"{actual}_vs_{reference}"
        comparison = _mapping(comparisons.get(name))
        for category in ("parameters_all_elements", "layer_samples"):
            values = _mapping(comparison.get(category))
            if not values or not all(_comparison_valid(value) for value in values.values()):
                errors.append(f"{name}: {category} comparisons incomplete")
        if comparison.get("missing_layer_samples") != []:
            errors.append(f"{name}: layer sample coverage incomplete")
    return errors


def _f2p_attempt_errors(attempts):
    errors = []
    for index, attempt in enumerate(attempts, 1):
        count, pairs = attempt.get("completed_updates"), attempt.get("pairs")
        if (
            attempt.get("attempt") != index
            or attempt.get("expected_pairs") != contract.F2P_UPDATES
            or type(count) is not int
            or not 0 <= count <= contract.F2P_UPDATES
            or not isinstance(pairs, list)
            or len(pairs) != count
            or not all(isinstance(pair, dict) for pair in pairs)
        ):
            errors.append(f"F2P attempt {index}: invalid workload or completed update evidence")
        if (index < len(attempts) or attempt.get("status") == "NOT_REPRODUCED") and (
            attempt.get("status") != "NOT_REPRODUCED" or count != contract.F2P_UPDATES
        ):
            errors.append(f"F2P attempt {index}: incomplete nonreproduction")
    for key in ("initial_model_sha256", "pair_schedule_sha256"):
        values = [a.get(key) for a in attempts]
        valid = all(
            isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)
            for value in values
        )
        if not valid or (values and any(value != values[0] for value in values)):
            errors.append(f"F2P: missing or inconsistent {key} across attempts")
    return errors


def aggregate_reports(root, expected_head, runner_exit_code):
    errors, jobs, hashes = [], {}, {}
    for job in contract.JOBS:
        path = root / f"{job}.json"
        try:
            worker = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            errors.append(f"{job}: missing or unreadable worker ({exc})")
            continue
        hashes[job] = sha256_file(path)
        worker = _mapping(worker)
        if (
            worker.get("schema") != contract.SCHEMA
            or worker.get("job") != job
            or worker.get("diagnostic_git_head") != expected_head
            or worker.get("workload_contract") != contract.workload_contract()
            or worker.get("precision_package_verified") is not True
            or worker.get("mechanism_sources_unchanged") is not True
            or worker.get("production_checkpoint_written") is not False
            or worker.get("labels_accessed") is not False
        ):
            errors.append(f"{job}: invalid provenance, source integrity or scope")
        if worker.get("status") != "DIAGNOSTIC_COMPLETE":
            errors.append(f"{job}: worker did not complete")
        item = {"status": worker.get("status"), "audits": {}}
        if job == "F2P":
            attempts = worker.get("f2p_replay", [])
            if not isinstance(attempts, list) or not all(isinstance(a, dict) for a in attempts):
                errors.append("F2P: invalid replay records")
                attempts = []
            errors.extend(_f2p_attempt_errors(attempts))
            item["attempts"] = [{k: v for k, v in attempt.items() if k != "pairs"} for attempt in attempts]
            captured = bool(attempts) and attempts[-1].get("status") == "FAILURE_CAPTURED"
            not_reproduced = len(attempts) == contract.F2P_ATTEMPTS and all(
                a.get("status") == "NOT_REPRODUCED"
                and a.get("completed_updates") == contract.F2P_UPDATES
                and isinstance(a.get("pairs"), list)
                and len(a.get("pairs", [])) == contract.F2P_UPDATES
                for a in attempts
            )
            valid_indices = [a.get("attempt") for a in attempts] == list(range(1, len(attempts) + 1))
            if not valid_indices or not 1 <= len(attempts) <= contract.F2P_ATTEMPTS or not (captured or not_reproduced):
                errors.append("F2P: bounded replay incomplete")
            if captured and not attempts[-1].get("capture_saved"):
                errors.append("F2P: failure observed but exact capture was not saved")
            item["reproduction"] = "CAPTURED" if captured else "NOT_REPRODUCED" if not_reproduced else "INCOMPLETE"
            if not captured:
                jobs[job] = item
                continue
        for name in ("bias_audit", "field_audit"):
            audit = _mapping(worker.get(name))
            item["audits"][name] = audit
            if audit.get("status") != "COMPLETE":
                errors.append(f"{job}: {name} incomplete")
            audit_errors = (
                field_audit_errors(audit)
                if name == "field_audit"
                else bias_audit_errors(audit, _mapping(_mapping(worker.get("environment")).get("backend_flags")))
            )
            errors.extend(f"{job}/{name}: {message}" for message in audit_errors)
        jobs[job] = item
    if runner_exit_code:
        errors.append(f"Runner exit code {runner_exit_code}")
    return {
        "schema": "ctcf-stage5-mechanism-summary-v1",
        "diagnostic_git_head": expected_head,
        "workload_contract": contract.workload_contract(),
        "runner_exit_code": runner_exit_code,
        "status": "INCOMPLETE" if errors else "OBSERVATIONS_COMPLETE_REVIEW_REQUIRED",
        "errors": errors,
        "jobs": jobs,
        "worker_report_sha256": hashes,
        "production_training_validated": False,
        "production_restart_authorized": False,
        "limitations": [
            "Complete means prescribed observations were collected, not that a mathematical fix is established.",
            "Parameter/node hooks do not expose every CUDA kernel accumulator or conversion.",
            "Field shifts and same-field arithmetic comparisons are distinct evidence.",
            "F2P not reproduced remains unknown; bounded retries do not prove it fixed.",
            "F2P repeats reset the same seed and initial state; they probe CUDA nondeterminism, not independent starts or training diversity.",
            "The field matrix varies cuDNN TF32 only; matmul TF32 is disabled and its separate effect is not isolated.",
            "No production training or labelled evaluation is performed.",
        ],
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--export-root", type=Path, required=True)
    parser.add_argument("--expected-git-head", required=True)
    parser.add_argument("--runner-exit-code", type=int, required=True)
    args = parser.parse_args(argv)
    summary = aggregate_reports(args.output_root, args.expected_git_head, args.runner_exit_code)
    with (args.output_root / "summary.json").open("x", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, allow_nan=False)
        handle.write("\n")
    lines = [
        f"Mechanism diagnostic: {summary['status']}",
        "",
        "Review bias_audit and field_audit separately for each job.",
    ]
    lines.extend([f"F2P reproduction: {summary['jobs'].get('F2P', {}).get('reproduction', 'INCOMPLETE')}", ""])
    lines.extend(f"- {message}" for message in summary["errors"] + summary["limitations"])
    (args.output_root / "SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    archive, digest = package_compact_zip(args.output_root, args.export_root)
    print(f"[MECHANISM DIAG STATUS] {summary['status']}")
    print(f"[MECHANISM DIAG PACKAGE] {archive}")
    print(f"[MECHANISM DIAG PACKAGE SIDECAR] {archive}.sha256")
    print(f"{digest}  {archive.name}")
    return int(bool(summary["errors"]))


if __name__ == "__main__":
    raise SystemExit(main())
