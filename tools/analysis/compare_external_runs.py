"""Gate scientific stability of two fresh controlled-CUDA external-method runs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

_FOLD_METRICS = ("central_fold_pct", "digital10_fold_pct", "sampled_tri_fold_pct")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_array(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes(order="C")).hexdigest()


def _flow_files(directory: Path) -> dict[str, Path]:
    return {path.stem.removeprefix("flow_"): path for path in sorted(directory.glob("flow_*.npz"))}


def _audit_rows(path: Path) -> dict[str, dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    result = {row["field"].removeprefix("flow_"): row for row in rows}
    if len(result) != len(rows):
        raise RuntimeError(f"duplicate cases in {path}")
    return result


def _exact_rows(path: Path) -> dict[str, dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise RuntimeError(f"expected a JSON list in {path}")
    result = {Path(row["file"]).stem.removeprefix("flow_"): row for row in payload}
    if len(result) != len(payload):
        raise RuntimeError(f"duplicate cases in {path}")
    return result


def _manifest(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("cases"), dict):
        raise RuntimeError(f"invalid manifest {path}")
    return payload


def _load_flow(path: Path) -> np.ndarray:
    with np.load(path, allow_pickle=False) as data:
        if set(data.files) != {"flow"}:
            raise RuntimeError(f"{path}: expected only the 'flow' array")
        flow = np.asarray(data["flow"])
    if flow.dtype != np.float32 or flow.ndim != 5 or flow.shape[:2] != (1, 3) or not np.isfinite(flow).all():
        raise RuntimeError(f"{path}: invalid canonical float32 flow")
    return flow


def _require_same_provenance(first: dict, second: dict) -> None:
    for key in ("schema", "method", "contract", "ctcf_git_sha", "ctcf_dirty", "environment"):
        if first.get(key) != second.get(key):
            raise RuntimeError(f"repeat manifests differ in {key}")
    if first.get("ctcf_dirty") is not False:
        raise RuntimeError("repeatability evidence requires a clean CTCF commit")


def _validate_binding(
    case: str,
    flow: np.ndarray,
    flow_path: Path,
    manifest_case: dict,
    audit: dict[str, str],
    exact: dict,
) -> str:
    array_hash = _sha256_array(flow)
    expected_array_hashes = {
        manifest_case.get("row", {}).get("flow_array_sha256"),
        audit.get("source_array_sha256"),
        exact.get("sha256"),
    }
    if expected_array_hashes != {array_hash}:
        raise RuntimeError(f"{case}: flow/manifest/audit/exact SHA-256 binding failed: {expected_array_hashes}")
    if manifest_case.get("row", {}).get("flow_file_sha256") != _sha256_file(flow_path):
        raise RuntimeError(f"{case}: compressed flow file SHA-256 differs from manifest")
    if exact.get("complete") is not True or exact.get("status") == "INCONCLUSIVE_RESOURCE_LIMIT":
        raise RuntimeError(f"{case}: exact audit is incomplete")
    run_id = manifest_case.get("generation_run_id")
    if not isinstance(run_id, str) or not run_id:
        raise RuntimeError(f"{case}: manifest has no generation_run_id")
    return run_id


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--first-flows", type=Path, required=True)
    parser.add_argument("--second-flows", type=Path, required=True)
    parser.add_argument("--first-manifest", type=Path, required=True)
    parser.add_argument("--second-manifest", type=Path, required=True)
    parser.add_argument("--first-audit", type=Path, required=True)
    parser.add_argument("--second-audit", type=Path, required=True)
    parser.add_argument("--first-exact", type=Path, required=True)
    parser.add_argument("--second-exact", type=Path, required=True)
    parser.add_argument("--max-flow-max-delta", type=float, default=1.0, help="voxel")
    parser.add_argument("--max-flow-mean-delta", type=float, default=0.01, help="voxel")
    parser.add_argument("--max-dice-delta", type=float, default=0.001)
    parser.add_argument("--max-fold-pct-delta", type=float, default=0.001, help="percentage points")
    parser.add_argument("--max-bound-delta", type=float, default=0.01)
    parser.add_argument("--max-failure-relative-delta", type=float, default=0.05)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    first_manifest = _manifest(args.first_manifest)
    second_manifest = _manifest(args.second_manifest)
    _require_same_provenance(first_manifest, second_manifest)
    first_files = _flow_files(args.first_flows)
    second_files = _flow_files(args.second_flows)
    first_audit = _audit_rows(args.first_audit)
    second_audit = _audit_rows(args.second_audit)
    first_exact = _exact_rows(args.first_exact)
    second_exact = _exact_rows(args.second_exact)
    case_sets = [
        set(value)
        for value in (
            first_files,
            second_files,
            first_manifest["cases"],
            second_manifest["cases"],
            first_audit,
            second_audit,
            first_exact,
            second_exact,
        )
    ]
    if not case_sets[0] or any(cases != case_sets[0] for cases in case_sets[1:]):
        raise RuntimeError(f"repeatability case sets differ: {[sorted(cases) for cases in case_sets]}")

    passed = True
    rows = []
    for case in sorted(case_sets[0]):
        first = _load_flow(first_files[case])
        second = _load_flow(second_files[case])
        if first.shape != second.shape:
            raise RuntimeError(f"{case}: repeat flow shapes differ: {first.shape} vs {second.shape}")
        first_case = first_manifest["cases"][case]
        second_case = second_manifest["cases"][case]
        for key in ("pair_semantics", "pair_file_sha256", "input_sha256"):
            if first_case.get(key) != second_case.get(key):
                raise RuntimeError(f"{case}: repeat manifests differ in {key}")
        first_run_id = _validate_binding(
            case, first, first_files[case], first_case, first_audit[case], first_exact[case]
        )
        second_run_id = _validate_binding(
            case, second, second_files[case], second_case, second_audit[case], second_exact[case]
        )
        if first_run_id == second_run_id:
            raise RuntimeError(f"{case}: the two fields came from the same generation invocation")

        delta = np.abs(first.astype(np.float64) - second.astype(np.float64))
        flow_max_delta = float(delta.max())
        flow_mean_delta = float(delta.mean())
        dice_delta = abs(float(first_audit[case]["dice_feedfwd"]) - float(second_audit[case]["dice_feedfwd"]))
        fold_deltas = {
            key: abs(float(first_audit[case][key]) - float(second_audit[case][key])) for key in _FOLD_METRICS
        }
        bound_delta = abs(float(first_audit[case]["tri_cert_bound"]) - float(second_audit[case]["tri_cert_bound"]))
        first_failures = int(first_exact[case]["n_failures"])
        second_failures = int(second_exact[case]["n_failures"])
        failure_relative_delta = abs(first_failures - second_failures) / max(first_failures, second_failures, 1)
        audit_state_equal = first_audit[case]["audit_state"] == second_audit[case]["audit_state"]
        exact_status_equal = first_exact[case]["status"] == second_exact[case]["status"]
        case_passed = (
            audit_state_equal
            and exact_status_equal
            and flow_max_delta <= args.max_flow_max_delta
            and flow_mean_delta <= args.max_flow_mean_delta
            and dice_delta <= args.max_dice_delta
            and max(fold_deltas.values()) <= args.max_fold_pct_delta
            and bound_delta <= args.max_bound_delta
            and failure_relative_delta <= args.max_failure_relative_delta
        )
        passed &= case_passed
        rows.append(
            {
                "case": case,
                "passed": case_passed,
                "first_generation_run_id": first_run_id,
                "second_generation_run_id": second_run_id,
                "bitwise_equal": bool(np.array_equal(first, second)),
                "first_sha256": _sha256_array(first),
                "second_sha256": _sha256_array(second),
                "flow_max_abs_delta_voxel": flow_max_delta,
                "flow_mean_abs_delta_voxel": flow_mean_delta,
                "dice_abs_delta": dice_delta,
                "fold_pct_abs_deltas": fold_deltas,
                "tri_cert_bound_abs_delta": bound_delta,
                "first_audit_state": first_audit[case]["audit_state"],
                "second_audit_state": second_audit[case]["audit_state"],
                "first_exact_status": first_exact[case]["status"],
                "second_exact_status": second_exact[case]["status"],
                "first_exact_failure_count": first_failures,
                "second_exact_failure_count": second_failures,
                "exact_failure_relative_delta": failure_relative_delta,
                "first_exact_interval_lo_min": first_exact[case]["interval_lo_min"],
                "second_exact_interval_lo_min": second_exact[case]["interval_lo_min"],
            }
        )

    report = {
        "passed": passed,
        "criterion": {
            "same_float64_audit_state": True,
            "same_exact_predicate_status": True,
            "max_flow_max_abs_delta_voxel": args.max_flow_max_delta,
            "max_flow_mean_abs_delta_voxel": args.max_flow_mean_delta,
            "max_dice_abs_delta": args.max_dice_delta,
            "max_fold_abs_delta_percentage_points": args.max_fold_pct_delta,
            "max_tri_cert_bound_abs_delta": args.max_bound_delta,
            "max_exact_failure_relative_delta": args.max_failure_relative_delta,
            "bitwise_identity_required": False,
        },
        "reason_bitwise_identity_not_required": (
            "official ConvexAdam Adam refinement uses CUDA grid_sampler_3d_backward and AvgPool3d backward, "
            "for which PyTorch does not provide deterministic implementations"
        ),
        "provenance": {
            "ctcf_git_sha": first_manifest["ctcf_git_sha"],
            "method": first_manifest["method"],
            "contract": first_manifest["contract"],
            "environment": first_manifest["environment"],
            "artifacts": {
                "first_manifest": {"path": str(args.first_manifest), "sha256": _sha256_file(args.first_manifest)},
                "second_manifest": {
                    "path": str(args.second_manifest),
                    "sha256": _sha256_file(args.second_manifest),
                },
                "first_audit": {"path": str(args.first_audit), "sha256": _sha256_file(args.first_audit)},
                "second_audit": {"path": str(args.second_audit), "sha256": _sha256_file(args.second_audit)},
                "first_exact": {"path": str(args.first_exact), "sha256": _sha256_file(args.first_exact)},
                "second_exact": {"path": str(args.second_exact), "sha256": _sha256_file(args.second_exact)},
            },
        },
        "cases": rows,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    for row in rows:
        print(
            f"{row['case']}: pass={row['passed']} max|dflow|={row['flow_max_abs_delta_voxel']:.6g} "
            f"mean|dflow|={row['flow_mean_abs_delta_voxel']:.6g} dDice={row['dice_abs_delta']:.6g} "
            f"exact={row['first_exact_status']}/{row['second_exact_status']}"
        )
    print(f"repeatability verdict: {'PASS' if passed else 'FAIL'}; report={args.report}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
