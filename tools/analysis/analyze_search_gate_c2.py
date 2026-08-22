from __future__ import annotations

import argparse
import csv
import io
import json
import math
import os
import statistics
import sys
import tempfile
from collections import Counter
from itertools import pairwise
from pathlib import Path
from typing import Any

from tools.analysis.run_artifacts import sha256_file

ANALYSIS_SCHEMA = "ctcf-search-c2-analysis-v1"
ANALYSIS_KIND = "POST_HOC_DEVELOPMENT_DIAGNOSTIC"
SIGN_RULES = ("accept_all", "ncc7_improves", "ncc9_improves", "ncc7_and_ncc9_improve", "ncc7_or_ncc9_improve")
SMOOTHING_CONTRAST = ("mind_s2_sm2", "mind_s2_sm1")
SDLOGJ_CAP_TRAJECTORY = "mind_s2_sm2_sdlogj_cap"
CANDIDATE_OFFSETS = 27
REQUIRED_INPUTS = ("per_step.csv", "summary.json")
OPTIONAL_INPUTS = ("per_branch.csv", "trajectory_summary.csv")
C2_SUMMARY_SCHEMA = "ctcf-search-c2-summary-v1"
C2_MANIFEST_SCHEMA = "ctcf-search-c2-run-manifest-v1"
C2_PROTOCOL_ID = "CTCF-SEARCH-GATE-C2-V1"
SIGN_RULE_ESTIMAND = "ONE_STEP_COUNTERFACTUAL_FROM_THE_OBSERVED_ENTERING_STATE"
REQUIRED_STEP_COLUMNS = frozenset(
    {
        "action",
        "baseline_dice",
        "candidate_dice",
        "candidate_dice_delta",
        "candidate_exact_status",
        "candidate_ncc7",
        "candidate_ncc9",
        "candidate_sdlogj",
        "case_id",
        "current_ncc7",
        "current_ncc9",
        "current_sdlogj",
        "labels_used_for_decision",
        "mind_improved",
        "proposal_built",
        "proposal_confidence_mean",
        "proposal_entropy_mean",
        "reason",
        "returned_dice",
        "returned_dice_delta",
        "returned_j_leq0_digital10_percent",
        "returned_sdlogj",
        "scale",
        "sdlogj_cap_limit",
        "sdlogj_cap_passed",
        "sdlogj_cap_relative",
        "smoothing_passes",
        "step",
        "trajectory_id",
        "work_eps",
    }
)

WORK_MARGIN_STATEMENT = (
    "Step index and work margin move together: work_eps is reduced at every step, so the incremental "
    "gain of a later step is not a clean causal ablation of iteration count."
)
REUSE_WARNING = (
    "Post-hoc diagnostic on the already opened IXI validation-58. It cannot promote C2, choose a "
    "threshold on the same cases, or authorise the IXI test-115 split."
)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def _number(row: dict[str, str], key: str) -> float | None:
    value = row.get(key, "")
    if value in ("", "None", "nan"):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _flag(row: dict[str, str], key: str) -> bool:
    return row.get(key, "") == "True"


def _mean(values: list[float]) -> float | None:
    return statistics.fmean(values) if values else None


def _steps(rows: list[dict[str, str]]) -> list[int]:
    return sorted({int(row["step"]) for row in rows})


def _trajectories(rows: list[dict[str, str]], directory: Path) -> list[str]:
    """Frozen branch order from trajectory_summary.csv when it is present, else the CSV order."""
    summary = directory / "trajectory_summary.csv"
    if summary.is_file():
        ordered = [row["trajectory_id"] for row in _read_csv(summary)]
        if len(ordered) != len(set(ordered)) or set(ordered) != {row["trajectory_id"] for row in rows}:
            raise ValueError(f"{summary}: trajectory ids do not match per_step.csv exactly.")
        return ordered
    seen: list[str] = []
    for row in rows:
        if row["trajectory_id"] not in seen:
            seen.append(row["trajectory_id"])
    return seen


def _branch_steps(rows: list[dict[str, str]], trajectories: list[str], baseline_mean: float) -> list[dict[str, Any]]:
    table: list[dict[str, Any]] = []
    for trajectory_id in trajectories:
        previous = baseline_mean
        for step in _steps(rows):
            group = [row for row in rows if row["trajectory_id"] == trajectory_id and int(row["step"]) == step]
            if not group:
                continue
            returned = [value for row in group if (value := _number(row, "returned_dice")) is not None]
            entry = {
                "trajectory_id": trajectory_id,
                "step": step,
                "cases": len(group),
                "actions": dict(sorted(Counter(row["action"] for row in group).items())),
                "reasons": dict(sorted(Counter(row["reason"] for row in group).items())),
                "proposals_built": sum(1 for row in group if _flag(row, "proposal_built")),
                "mind_improved": sum(1 for row in group if _flag(row, "mind_improved")),
                "exact_certified": sum(1 for row in group if row["candidate_exact_status"] == "CERTIFIED"),
                "scale": sorted({_number(row, "scale") for row in group}),
                "smoothing_passes": sorted({_number(row, "smoothing_passes") for row in group}),
                "work_eps": sorted({_number(row, "work_eps") for row in group}),
                "returned_dice_mean": _mean(returned),
                "returned_dice_median": statistics.median(returned) if returned else None,
                "returned_dice_delta_mean": _mean(
                    [value for row in group if (value := _number(row, "returned_dice_delta")) is not None]
                ),
                "returned_sdlogj_mean": _mean(
                    [value for row in group if (value := _number(row, "returned_sdlogj")) is not None]
                ),
                "returned_digital10_percent_mean": _mean(
                    [value for row in group if (value := _number(row, "returned_j_leq0_digital10_percent")) is not None]
                ),
            }
            entry["incremental_gain_vs_previous_step"] = (
                None if entry["returned_dice_mean"] is None else entry["returned_dice_mean"] - previous
            )
            previous = entry["returned_dice_mean"] if entry["returned_dice_mean"] is not None else previous
            table.append(entry)
    return table


def _work_margin_confound(rows: list[dict[str, str]], trajectories: list[str]) -> dict[str, Any]:
    by_trajectory: dict[str, list[float | None]] = {}
    for trajectory_id in trajectories:
        values = []
        for step in _steps(rows):
            group = [row for row in rows if row["trajectory_id"] == trajectory_id and int(row["step"]) == step]
            unique = {_number(row, "work_eps") for row in group}
            values.append(unique.pop() if len(unique) == 1 else None)
        by_trajectory[trajectory_id] = values
    series = [values for values in by_trajectory.values() if all(value is not None for value in values)]
    return {
        "statement": WORK_MARGIN_STATEMENT,
        "work_eps_by_step": by_trajectory,
        "work_eps_changes_with_step": any(len(set(values)) > 1 for values in series),
        "monotone_decreasing": bool(series)
        and all(all(later < earlier for earlier, later in pairwise(values)) for values in series),
    }


def _entering_dice(rows: list[dict[str, str]]) -> dict[tuple[str, str, int], float | None]:
    """Dice each case carried into a step: its baseline at step 1, else the previous returned value."""
    returned = {(row["trajectory_id"], row["case_id"], int(row["step"])): _number(row, "returned_dice") for row in rows}
    entering: dict[tuple[str, str, int], float | None] = {}
    for row in rows:
        step = int(row["step"])
        key = (row["trajectory_id"], row["case_id"], step)
        entering[key] = (
            _number(row, "baseline_dice")
            if step == 1
            else returned.get((row["trajectory_id"], row["case_id"], step - 1))
        )
    return entering


def _rule_accepts(row: dict[str, str], rule: str) -> bool:
    ncc7 = _number(row, "candidate_ncc7"), _number(row, "current_ncc7")
    ncc9 = _number(row, "candidate_ncc9"), _number(row, "current_ncc9")
    better7 = None not in ncc7 and ncc7[0] < ncc7[1]
    better9 = None not in ncc9 and ncc9[0] < ncc9[1]
    if rule == "accept_all":
        return True
    if rule == "ncc7_improves":
        return better7
    if rule == "ncc9_improves":
        return better9
    if rule == "ncc7_and_ncc9_improve":
        return better7 and better9
    return better7 or better9


def _sign_rules(rows: list[dict[str, str]], trajectories: list[str]) -> list[dict[str, Any]]:
    entering = _entering_dice(rows)
    table: list[dict[str, Any]] = []
    for trajectory_id in trajectories:
        for step in _steps(rows):
            group = [row for row in rows if row["trajectory_id"] == trajectory_id and int(row["step"]) == step]
            if not group:
                continue
            for rule in SIGN_RULES:
                accepted, improved, worsened, unchanged = 0, 0, 0, 0
                returned_dice: list[float] = []
                returned_delta: list[float] = []
                for row in group:
                    candidate = _number(row, "candidate_dice")
                    baseline = _number(row, "baseline_dice")
                    start = entering[(trajectory_id, row["case_id"], step)]
                    takes = candidate is not None and _rule_accepts(row, rule)
                    if takes:
                        accepted += 1
                        delta = candidate - start if start is not None else 0.0
                        improved += delta > 0.0
                        worsened += delta < 0.0
                        unchanged += delta == 0.0
                    value = candidate if takes else start
                    if value is not None:
                        returned_dice.append(value)
                        if baseline is not None:
                            returned_delta.append(value - baseline)
                table.append(
                    {
                        "trajectory_id": trajectory_id,
                        "step": step,
                        "rule": rule,
                        "estimand": SIGN_RULE_ESTIMAND,
                        "selection_authorized": False,
                        "cases": len(group),
                        "candidates_available": sum(1 for row in group if _number(row, "candidate_dice") is not None),
                        "accepted": accepted,
                        "dice_improved": improved,
                        "dice_worsened": worsened,
                        "dice_unchanged": unchanged,
                        "mean_returned_dice": _mean(returned_dice),
                        "mean_returned_dice_delta": _mean(returned_delta),
                    }
                )
    return table


def _smoothing_contrast(rows: list[dict[str, str]]) -> dict[str, Any] | None:
    first, second = SMOOTHING_CONTRAST
    present = {row["trajectory_id"] for row in rows}
    if not {first, second} <= present:
        return None

    def indexed(trajectory_id: str, step: int, key: str) -> dict[str, float | None]:
        return {
            row["case_id"]: _number(row, key)
            for row in rows
            if row["trajectory_id"] == trajectory_id and int(row["step"]) == step
        }

    def paired(step: int, key: str) -> dict[str, float]:
        left, right = indexed(first, step, key), indexed(second, step, key)
        shared = sorted(set(left) & set(right))
        return {case: left[case] - right[case] for case in shared if None not in (left[case], right[case])}

    per_step = []
    for step in _steps(rows):
        dice = paired(step, "returned_dice")
        values = list(dice.values())
        per_step.append(
            {
                "step": step,
                "cases": len(values),
                "dice_difference_mean": _mean(values),
                "dice_difference_median": statistics.median(values) if values else None,
                "first_better_cases": sum(1 for value in values if value > 0.0),
                "first_better_sdlogj_cases": sum(
                    1 for value in paired(step, "returned_sdlogj").values() if value < 0.0
                ),
                "first_better_digital10_cases": sum(
                    1 for value in paired(step, "returned_j_leq0_digital10_percent").values() if value < 0.0
                ),
            }
        )
    final_step = _steps(rows)[-1]
    final_dice = paired(final_step, "returned_dice")
    final_values = list(final_dice.values())
    return {
        "trajectories": [first, second],
        "note": "Positive dice difference means the first trajectory is ahead; for sdlogj and digital10 the "
        "counted cases are those where the first trajectory is lower, which is the better direction for both.",
        "final": {
            "step": final_step,
            "cases": len(final_values),
            "dice_difference_mean": _mean(final_values),
            "dice_difference_median": statistics.median(final_values) if final_values else None,
            "first_better_cases": sum(1 for value in final_values if value > 0.0),
            "per_case_dice_difference": dict(sorted(final_dice.items())),
        },
        "per_step": per_step,
    }


def _proposal_information(rows: list[dict[str, str]]) -> dict[str, Any]:
    built = [row for row in rows if _flag(row, "proposal_built")]
    entropy = [value for row in built if (value := _number(row, "proposal_entropy_mean")) is not None]
    confidence = [value for row in built if (value := _number(row, "proposal_confidence_mean")) is not None]
    max_entropy = math.log(CANDIDATE_OFFSETS)
    entropy_mean = _mean(entropy)
    return {
        "candidate_offsets": CANDIDATE_OFFSETS,
        "max_entropy_nats": max_entropy,
        "proposals_built": len(built),
        "entropy_mean_nats": entropy_mean,
        "entropy_ratio": None if entropy_mean is None else entropy_mean / max_entropy,
        "confidence_mean": _mean(confidence),
        "confidence_at_uniform_posterior": 0.0,
        "pooling": "row-weighted mean over proposal executions, including repeated steps of capped trajectories",
        "note": "Confidence is 1 - entropy / log(valid candidates), so a uniform cost volume scores 0.",
    }


def _sdlogj_cap(rows: list[dict[str, str]]) -> dict[str, Any] | None:
    group = [row for row in rows if row["trajectory_id"] == SDLOGJ_CAP_TRAJECTORY and int(row["step"]) == 1]
    if not group:
        return None
    growth = [
        (candidate - current) / current
        for row in group
        if (candidate := _number(row, "candidate_sdlogj")) is not None
        and (current := _number(row, "current_sdlogj")) not in (None, 0.0)
    ]
    limits = {_number(row, "sdlogj_cap_relative") for row in group}
    return {
        "trajectory_id": SDLOGJ_CAP_TRAJECTORY,
        "cases": len(group),
        "relative_limit": limits.pop() if len(limits) == 1 else None,
        "rolled_back_cases": sum(1 for row in group if row["action"] == "ROLLBACK"),
        "cap_passed_cases": sum(1 for row in group if _flag(row, "sdlogj_cap_passed")),
        "mind_and_certificate_passed_cases": sum(
            1 for row in group if _flag(row, "mind_improved") and row["candidate_exact_status"] == "CERTIFIED"
        ),
        "relative_growth_percent_min": min(growth) * 100.0 if growth else None,
        "relative_growth_percent_max": max(growth) * 100.0 if growth else None,
        "relative_growth_percent_mean": _mean(growth) * 100.0 if growth else None,
    }


def _required_false(mapping: dict[str, Any], key: str, source: Path) -> None:
    if key not in mapping or type(mapping[key]) is not bool or mapping[key] is not False:
        raise ValueError(f"{source}: {key} must be the JSON boolean false.")


def _required_count(mapping: dict[str, Any], key: str, expected: int, source: Path) -> None:
    if key not in mapping or type(mapping[key]) is not int or mapping[key] != expected:
        raise ValueError(f"{source}: {key} must equal the observed count {expected}, got {mapping.get(key)!r}.")


def _validate_rows(rows: list[dict[str, str]], summary: dict[str, Any], source: Path) -> None:
    identities: set[tuple[str, str, int]] = set()
    baselines: dict[str, set[float]] = {}
    steps_by_pair: dict[tuple[str, str], set[int]] = {}
    for row_index, row in enumerate(rows, start=2):
        case_id, trajectory_id = row.get("case_id", ""), row.get("trajectory_id", "")
        if not case_id or not trajectory_id:
            raise ValueError(f"{source}:{row_index}: case_id and trajectory_id must be non-empty.")
        try:
            step = int(row["step"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"{source}:{row_index}: step must be an integer.") from error
        if step < 1 or str(step) != row["step"]:
            raise ValueError(f"{source}:{row_index}: step must be a canonical positive integer.")
        identity = (trajectory_id, case_id, step)
        if identity in identities:
            raise ValueError(f"{source}:{row_index}: duplicate row identity {identity}.")
        identities.add(identity)
        if row.get("labels_used_for_decision") != "False":
            raise ValueError(f"{source}:{row_index}: labels_used_for_decision must be exactly False.")
        baseline = _number(row, "baseline_dice")
        if baseline is None:
            raise ValueError(f"{source}:{row_index}: baseline_dice must be finite.")
        baselines.setdefault(case_id, set()).add(baseline)
        steps_by_pair.setdefault((trajectory_id, case_id), set()).add(step)

    if any(len(values) != 1 for values in baselines.values()):
        raise ValueError(f"{source}: baseline_dice is not constant within one or more cases.")
    case_ids = {row["case_id"] for row in rows}
    trajectory_ids = {row["trajectory_id"] for row in rows}
    observed_steps = {int(row["step"]) for row in rows}
    expected_pairs = {(trajectory, case) for trajectory in trajectory_ids for case in case_ids}
    if set(steps_by_pair) != expected_pairs or any(steps != observed_steps for steps in steps_by_pair.values()):
        raise ValueError(f"{source}: rows do not form a complete trajectory x case x step grid.")
    _required_count(summary, "n_cases", len(case_ids), source.parent / "summary.json")
    _required_count(summary, "n_trajectories", len(trajectory_ids), source.parent / "summary.json")
    _required_count(summary, "n_step_rows", len(rows), source.parent / "summary.json")


def _validate_manifest(directory: Path, summary: dict[str, Any]) -> None:
    path = directory / "c2_manifest.json"
    if not path.is_file():
        return
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("schema") != C2_MANIFEST_SCHEMA or manifest.get("protocol_id") != C2_PROTOCOL_ID:
        raise ValueError(f"{path}: unexpected C2 manifest schema or protocol.")
    if manifest.get("status") != "COMPLETE" or manifest.get("summary") != summary:
        raise ValueError(f"{path}: manifest is not COMPLETE or its embedded summary differs from summary.json.")
    files = manifest.get("files")
    if not isinstance(files, dict):
        raise ValueError(f"{path}: files must be an object.")
    hash_keys = {
        "per_step.csv": "per_step_sha256",
        "per_branch.csv": "per_branch_sha256",
        "trajectory_summary.csv": "trajectory_summary_sha256",
        "summary.json": "summary_sha256",
    }
    for name, key in hash_keys.items():
        candidate = directory / name
        if candidate.is_file() and files.get(key) != sha256_file(candidate):
            raise ValueError(f"{path}: {key} does not match {name}.")


def analyze(c2_dir: Path) -> dict[str, Any]:
    """Recompute the compact C2 development diagnostics from the stored CSVs. Nothing is written."""
    directory = Path(c2_dir).resolve()
    for name in REQUIRED_INPUTS:
        if not (directory / name).is_file():
            raise ValueError(f"{directory}: required input {name} is missing.")
    rows = _read_csv(directory / "per_step.csv")
    if not rows:
        raise ValueError(f"{directory}: per_step.csv has no rows.")
    missing = REQUIRED_STEP_COLUMNS - set(rows[0])
    if missing:
        raise ValueError(f"{directory}: per_step.csv is missing columns {sorted(missing)}.")
    summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    if not isinstance(summary, dict):
        raise ValueError(f"{directory / 'summary.json'}: top-level JSON value must be an object.")
    if summary.get("schema") != C2_SUMMARY_SCHEMA or summary.get("protocol_id") != C2_PROTOCOL_ID:
        raise ValueError(f"{directory / 'summary.json'}: unexpected C2 summary schema or protocol.")
    for key in ("test_115_authorized", "test_split_accessed", "labels_used_for_transaction_decision"):
        _required_false(summary, key, directory / "summary.json")
    _validate_rows(rows, summary, directory / "per_step.csv")
    _validate_manifest(directory, summary)

    trajectories = _trajectories(rows, directory)
    baselines = {row["case_id"]: value for row in rows if (value := _number(row, "baseline_dice")) is not None}
    baseline_values = [baselines[case_id] for case_id in sorted(baselines)]
    inputs = [
        {"name": name, "bytes": (directory / name).stat().st_size, "sha256": sha256_file(directory / name)}
        for name in (*REQUIRED_INPUTS, *OPTIONAL_INPUTS, "c2_manifest.json")
        if (directory / name).is_file()
    ]
    return {
        "schema": ANALYSIS_SCHEMA,
        "analysis_kind": ANALYSIS_KIND,
        "test_115_authorized": summary["test_115_authorized"],
        "test_split_accessed": summary["test_split_accessed"],
        "reuse_warning": REUSE_WARNING,
        "source": {
            "c2_dir": str(directory),
            "protocol_id": summary.get("protocol_id"),
            "scientific_status": summary.get("scientific_status"),
            "selected_trajectory_id": summary.get("selected_trajectory_id"),
            "labels_used_for_transaction_decision": summary["labels_used_for_transaction_decision"],
            "n_cases": summary.get("n_cases"),
            "n_step_rows": summary.get("n_step_rows"),
            "n_trajectories": summary.get("n_trajectories"),
            "trajectories": trajectories,
            "inputs": sorted(inputs, key=lambda entry: entry["name"]),
        },
        "baseline": {
            "cases": len(baseline_values),
            "dice_mean": _mean(baseline_values),
            "dice_median": statistics.median(baseline_values) if baseline_values else None,
        },
        "branch_steps": _branch_steps(rows, trajectories, _mean(baseline_values)),
        "work_margin_confound": _work_margin_confound(rows, trajectories),
        "smoothing_contrast": _smoothing_contrast(rows),
        "proposal_information": _proposal_information(rows),
        "sign_rules": _sign_rules(rows, trajectories),
        "sdlogj_cap": _sdlogj_cap(rows),
    }


def _write_atomic(path: Path, text: str) -> None:
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    os.close(descriptor)
    try:
        Path(temporary).write_text(text, encoding="utf-8", newline="\n")
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def _cell(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        return json.dumps(value, separators=(",", ":"), sort_keys=True)
    return str(value)


def _csv_text(table: list[dict[str, Any]], columns: list[str]) -> str:
    stream = io.StringIO()
    writer = csv.writer(stream, lineterminator="\n")
    writer.writerow(columns)
    writer.writerows([[_cell(row.get(column)) for column in columns] for row in table])
    return stream.getvalue()


def write_outputs(analysis: dict[str, Any], output_dir: Path) -> None:
    """Atomic JSON plus the three flat tables; an existing file is replaced, never appended to."""
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_atomic(output_dir / "analysis.json", json.dumps(analysis, indent=2, sort_keys=True) + "\n")
    _write_atomic(
        output_dir / "branch_steps.csv",
        _csv_text(
            analysis["branch_steps"],
            [
                "trajectory_id",
                "step",
                "cases",
                "actions",
                "reasons",
                "proposals_built",
                "mind_improved",
                "exact_certified",
                "work_eps",
                "returned_dice_mean",
                "returned_dice_delta_mean",
                "incremental_gain_vs_previous_step",
                "returned_sdlogj_mean",
                "returned_digital10_percent_mean",
            ],
        ),
    )
    _write_atomic(
        output_dir / "sign_rules.csv",
        _csv_text(
            analysis["sign_rules"],
            [
                "trajectory_id",
                "step",
                "rule",
                "cases",
                "candidates_available",
                "accepted",
                "dice_improved",
                "dice_worsened",
                "mean_returned_dice",
                "mean_returned_dice_delta",
            ],
        ),
    )
    contrast = analysis["smoothing_contrast"]
    _write_atomic(
        output_dir / "smoothing_contrast.csv",
        _csv_text(
            [] if contrast is None else contrast["per_step"],
            [
                "step",
                "cases",
                "dice_difference_mean",
                "dice_difference_median",
                "first_better_cases",
                "first_better_sdlogj_cases",
                "first_better_digital10_cases",
            ],
        ),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Read-only post-hoc analysis of a compact Gate C2 product.")
    parser.add_argument("--c2_dir", required=True, help="Compact C2 output directory; it is never modified.")
    parser.add_argument("--output", default=None, help="Directory for analysis.json and the CSV tables.")
    arguments = parser.parse_args(argv)

    c2_dir = Path(arguments.c2_dir).resolve()
    analysis = analyze(c2_dir)
    if arguments.output is None:
        print(json.dumps(analysis, indent=2, sort_keys=True))
        return 0
    output_dir = Path(arguments.output).resolve()
    if output_dir == c2_dir or c2_dir in output_dir.parents:
        raise ValueError(f"{output_dir}: refusing to write inside the C2 product {c2_dir}.")
    write_outputs(analysis, output_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
