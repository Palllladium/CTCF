from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import platform
import re
import shutil
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from tools.analysis.run_artifacts import atomic_write_json, atomic_write_text, rows_to_csv, sha256_file
from tools.analysis.search_gate_c3 import (
    build_c3a_supports,
    metric_envelope,
    paired_summary,
    primary_ncc_decision,
)
from tools.analysis.search_gate_common import dice_score, git, utc_now
from tools.analysis.search_gate_cost_volume import build_raw_mind_cost_volume, masked_vector_rms
from tools.analysis.search_gate_metrics import (
    DIGITAL_DECOMPOSITION,
    LEARN2REG_SHIFTED_SDLOGJ_MASKED,
    MATHEMATICAL_SDLOGJ_CROP2,
    METRIC_SPECS,
    compute_metric,
)
from tools.analysis.search_gate_numerical_stability import (
    BOOTSTRAP_CONFIDENCE,
    BOOTSTRAP_RESAMPLES,
    BOOTSTRAP_SEED,
    COLLAR_WIDTH,
    EXACT_CLAIM_EPS,
    FACTORIAL_EDGES,
    FACTORIAL_SPECS,
    FAILED_VECTORIZED_SENTINEL_ATOL,
    GEOMETRY_NONINFERIOR_TOLERANCE,
    LEGACY_PARITY_ATOL,
    LOCAL_CLIP_SWEEPS,
    MOMENT_CENTERED_FP32,
    MOMENT_CENTERED_FP64,
    MOMENT_LEGACY,
    MOMENT_VECTORIZED,
    NUMERICAL_STABILITY_POLICY,
    NUMERICAL_STABILITY_POLICY_SHA256,
    ORACLE_FAITHFUL_ATOL,
    PROTOCOL_ID,
    SCHEMA_VERSION,
    SCIENTIFIC_ARMS,
    SENTINEL_ALL_VECTORIZED_GAPS,
    SOURCE_C3_GIT_HEAD,
    SOURCE_C3_MANIFEST_SHA256,
    SOURCE_C3_RUN_ID,
    SOURCE_C3_RUN_MANIFEST_SHA256,
    WORK_EPS,
    assert_frozen_policy,
    assess_arm_eligibility,
    build_reduction_study,
    field_difference,
    select_next_branch,
    selfcheck,
)
from tools.analysis.search_gate_runtime import (
    expected_shard_for_case,
    parse_physical_gpus,
    round_robin_shards,
    save_reload_certify,
    shard_gpu_map,
    validate_shard_partition,
)
from tools.analysis.transactional_search import (
    build_proposal,
    certified_local_clip_candidate,
    geometry_mask,
    load_flow_npz,
    masked_zscore,
    mind_distance_from_features,
    mind_ssc,
    ncc_loss_from_normalized,
    valid_sample_mask,
)
from utils import setup_device
from utils.field import trilinear_cert_bound

C3_PROTOCOL_ID = "CTCF-SEARCH-GATE-C3A-V1"
C3_MANIFEST_SCHEMA = "ctcf-search-c3a-run-manifest-v1"
SOURCE_CONTRACT_NAME = "source_contract.json"
DECISION_CONTRACT_NAME = "decision_contract.json"
BARRIER_NAME = "decision_barrier.json"
EXPECTED_CASES = 58
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
IMAGE_STD_FLOOR = 1e-6
NCC_EPS = 1e-5
SOURCE_BASELINE_DICE_ATOL = 1e-8


def _e54_sentinel_record(case_id: str, observed: float | None) -> dict[str, float | str | bool | None]:
    expected = SENTINEL_ALL_VECTORIZED_GAPS.get(case_id)
    if expected is None:
        if observed is not None:
            raise ValueError(f"Unexpected E54 sentinel observation for {case_id}")
        return {
            "expected_max_abs": None,
            "observed_max_abs": None,
            "absolute_error": None,
            "absolute_tolerance": None,
            "expected_max_abs_9g": None,
            "observed_max_abs_9g": None,
            "pass": None,
        }
    if isinstance(observed, bool) or not isinstance(observed, (int, float)) or not math.isfinite(float(observed)):
        raise ValueError(f"Invalid E54 sentinel observation for {case_id}: {observed!r}")
    expected_value = float(expected)
    observed_value = float(observed)
    absolute_error = abs(observed_value - expected_value)
    return {
        "expected_max_abs": expected_value,
        "observed_max_abs": observed_value,
        "absolute_error": absolute_error,
        "absolute_tolerance": FAILED_VECTORIZED_SENTINEL_ATOL,
        "expected_max_abs_9g": format(expected_value, ".9g"),
        "observed_max_abs_9g": format(observed_value, ".9g"),
        "pass": absolute_error <= FAILED_VECTORIZED_SENTINEL_ATOL,
    }


def _validate_e54_sentinel(payload: Any, case_id: str) -> None:
    expected = SENTINEL_ALL_VECTORIZED_GAPS.get(case_id)
    if expected is None:
        if payload != _e54_sentinel_record(case_id, None):
            raise RuntimeError(f"Unexpected NUMSTAB e54 sentinel record: {case_id}")
        return
    if not isinstance(payload, dict) or set(payload) != set(_e54_sentinel_record(case_id, float(expected))):
        raise RuntimeError(f"Invalid NUMSTAB e54 sentinel schema: {case_id}")
    expected_value = payload.get("expected_max_abs")
    observed_value = payload.get("observed_max_abs")
    absolute_error = payload.get("absolute_error")
    tolerance = payload.get("absolute_tolerance")
    numeric_values = (expected_value, observed_value, absolute_error, tolerance)
    if any(
        isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value))
        for value in numeric_values
    ):
        raise RuntimeError(f"Invalid NUMSTAB e54 sentinel values: {case_id}")
    expected_value = float(expected_value)
    observed_value = float(observed_value)
    absolute_error = float(absolute_error)
    tolerance = float(tolerance)
    recomputed_error = abs(observed_value - expected_value)
    expected_pass = recomputed_error <= FAILED_VECTORIZED_SENTINEL_ATOL
    if (
        expected_value != float(expected)
        or absolute_error != recomputed_error
        or tolerance != FAILED_VECTORIZED_SENTINEL_ATOL
        or payload.get("expected_max_abs_9g") != format(expected_value, ".9g")
        or payload.get("observed_max_abs_9g") != format(observed_value, ".9g")
        or payload.get("pass") is not expected_pass
        or not expected_pass
    ):
        raise RuntimeError(f"NUMSTAB e54 sentinel is not reproduced: {case_id}")


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: expected one JSON object")
    return payload


def _json_equivalent(left: Any, right: Any) -> bool:
    """Compare values after the tuple-to-list normalization performed by JSON."""

    options = {"ensure_ascii": False, "sort_keys": True, "separators": (",", ":")}
    return json.dumps(left, **options) == json.dumps(right, **options)


def _table_rows(text: str, *, delimiter: str, label: str) -> tuple[tuple[str, ...], list[dict[str, str]]]:
    reader = csv.DictReader(io.StringIO(text), delimiter=delimiter)
    fieldnames = tuple(reader.fieldnames or ())
    rows = list(reader)
    if not fieldnames or not rows or any(set(row) != set(fieldnames) or None in row for row in rows):
        raise RuntimeError(f"Invalid frozen dataset inventory: {label}")
    return fieldnames, rows


def _require_sha(value: Any, label: str) -> str:
    lowered = str(value).lower()
    if not SHA256_RE.fullmatch(lowered):
        raise ValueError(f"{label} must be 64 lowercase hexadecimal characters")
    return lowered


def _sha256_array(value: np.ndarray | torch.Tensor) -> str:
    if isinstance(value, torch.Tensor):
        array = value.detach().cpu().contiguous().numpy().astype(np.float32, copy=False)
    else:
        array = np.ascontiguousarray(value, dtype=np.float32)
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def _runtime_signature() -> dict[str, Any]:
    import scipy

    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
    }


def _assert_clean_code(head: str, stage: str) -> None:
    if git("rev-parse", "HEAD") != head or git("status", "--porcelain=v1"):
        raise RuntimeError(f"NUMSTAB {stage} code differs from its clean prepared contract")


def _assert_runtime(expected: dict[str, Any], stage: str) -> None:
    observed = _runtime_signature()
    if observed != expected:
        raise RuntimeError(f"NUMSTAB {stage} runtime changed: {observed} != {expected}")


def _verify_file_record(row: dict[str, Any], *, expected_path: Path | None = None) -> Path:
    path = Path(str(row.get("path", ""))).resolve()
    if expected_path is not None and path != expected_path.resolve():
        raise RuntimeError(f"Frozen path changed: {path} != {expected_path.resolve()}")
    if (
        not path.is_file()
        or path.stat().st_size != int(row.get("bytes", -1))
        or sha256_file(path) != _require_sha(row.get("sha256"), str(path))
    ):
        raise RuntimeError(f"Frozen file is missing or changed: {path}")
    return path


def _verify_image_record(row: dict[str, Any], source_heavy: Path) -> np.ndarray:
    path = _verify_file_record(row)
    if source_heavy.resolve() not in path.parents:
        raise RuntimeError(f"Image cache escaped the frozen C3 heavy root: {path}")
    array = np.load(path, allow_pickle=False)
    if (
        array.dtype != np.float32
        or list(array.shape) != row.get("shape")
        or not np.isfinite(array).all()
        or _sha256_array(array) != _require_sha(row.get("array_sha256"), str(path))
    ):
        raise RuntimeError(f"Frozen C3 image array is invalid or changed: {path}")
    return np.ascontiguousarray(array)


def _resolve_field(root: Path, record: dict[str, Any], *, verify_array: bool = True) -> Path:
    base = root.resolve()
    path = (base / str(record.get("relative_path", ""))).resolve()
    if base not in path.parents or not path.is_file():
        raise RuntimeError(f"Heavy field is missing or escaped its root: {path}")
    if sha256_file(path) != _require_sha(record.get("npz_sha256"), str(path)):
        raise RuntimeError(f"Heavy field bytes changed: {path}")
    if verify_array and _sha256_array(load_flow_npz(path)) != _require_sha(record.get("array_sha256"), str(path)):
        raise RuntimeError(f"Heavy field array changed: {path}")
    return path


def _field_record(path: Path, root: Path, array_sha256: str) -> dict[str, Any]:
    return {
        "relative_path": path.resolve().relative_to(root.resolve()).as_posix(),
        "npz_sha256": sha256_file(path),
        "array_sha256": _require_sha(array_sha256, str(path)),
    }


def _finite_dice(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RuntimeError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise RuntimeError(f"{label} must be finite and in [0,1]")
    return result


def _validate_source_c3(
    directory: Path,
    heavy_root: Path,
    expected_manifest_sha: str,
    expected_run_manifest_sha: str,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    directory = directory.resolve()
    heavy_root = heavy_root.resolve()
    manifest_path = directory / "c3_manifest.json"
    run_manifest_path = directory / "run_manifest.json"
    expected = _require_sha(expected_manifest_sha, "C3 manifest SHA-256")
    expected_run = _require_sha(expected_run_manifest_sha, "C3 native run manifest SHA-256")
    if expected != SOURCE_C3_MANIFEST_SHA256 or sha256_file(manifest_path) != expected:
        raise RuntimeError("NUMSTAB requires the single frozen successful C3 manifest")
    if expected_run != SOURCE_C3_RUN_MANIFEST_SHA256 or sha256_file(run_manifest_path) != expected_run:
        raise RuntimeError("NUMSTAB requires the frozen native run manifest from the same successful C3")
    manifest = _load_json(manifest_path)
    run_manifest = _load_json(run_manifest_path)
    summary = manifest.get("summary") or {}
    code = manifest.get("code") or {}
    if (
        manifest.get("schema") != C3_MANIFEST_SCHEMA
        or manifest.get("protocol_id") != C3_PROTOCOL_ID
        or manifest.get("run_id") != SOURCE_C3_RUN_ID
        or manifest.get("status") != "COMPLETE"
        or code.get("git_head") != SOURCE_C3_GIT_HEAD
        or code.get("git_status") != ""
        or summary.get("execution_integrity_status") != "PASS"
        or summary.get("n_cases") != EXPECTED_CASES
        or summary.get("test_115_authorized") is not False
        or summary.get("test_split_accessed") is not False
        or summary.get("labels_used_for_decision") is not False
        or Path(str((manifest.get("storage") or {}).get("heavy_root", ""))).resolve() != heavy_root
        or run_manifest.get("schema") != "ctcf-native-manifest-v1"
        or run_manifest.get("run_id") != SOURCE_C3_RUN_ID
        or run_manifest.get("status") != "COMPLETE"
        or (run_manifest.get("code") or {}).get("git_head") != SOURCE_C3_GIT_HEAD
        or (run_manifest.get("code") or {}).get("tracked_tree_clean_at_start") is not True
        or (run_manifest.get("execution") or {}).get("seed") != 0
        or (run_manifest.get("execution") or {}).get("paths_profile") != 3
        or run_manifest.get("exit_code") != 0
    ):
        raise RuntimeError("Frozen C3 source is incomplete, altered, or not label-isolated val-58")

    files = manifest.get("files") or {}
    required = {
        "source_contract_sha256": "source_contract.json",
        "decision_contract_sha256": "decision_contract.json",
        "decision_barrier_sha256": "decision_barrier.json",
        "datasets_sha256": "datasets.csv",
        "c2_decision_goldens_sha256": "c2_decision_goldens.json",
        "c2_evaluation_goldens_sha256": "c2_evaluation_goldens.json",
        "per_arm_sha256": "per_arm.csv",
        "arm_summary_sha256": "arm_summary.csv",
        "hypotheses_sha256": "hypotheses.json",
        "summary_sha256": "summary.json",
    }
    for key, name in required.items():
        path = directory / name
        if not path.is_file() or sha256_file(path) != _require_sha(files.get(key), f"C3 {name}"):
            raise RuntimeError(f"Frozen C3 manifest does not authenticate {path}")
    native_datasets = directory / "datasets.tsv"
    if not native_datasets.is_file() or sha256_file(native_datasets) != _require_sha(
        (run_manifest.get("files") or {}).get("datasets_sha256"), "C3 native datasets.tsv"
    ):
        raise RuntimeError("Frozen C3 native manifest does not authenticate datasets.tsv")

    source = _load_json(directory / "source_contract.json")
    decision = _load_json(directory / "decision_contract.json")
    barrier = _load_json(directory / "decision_barrier.json")
    case_ids = source.get("case_ids") or []
    if (
        source.get("schema") != "ctcf-search-c3a-source-contract-v1"
        or decision.get("schema") != "ctcf-search-c3a-decision-contract-v1"
        or barrier.get("schema") != "ctcf-search-c3a-decision-barrier-v1"
        or source.get("git_head") != SOURCE_C3_GIT_HEAD
        or decision.get("git_head") != SOURCE_C3_GIT_HEAD
        or decision.get("source_contract_sha256") != files["source_contract_sha256"]
        or barrier.get("decision_contract_sha256") != files["decision_contract_sha256"]
        or Path(str(source.get("heavy_root", ""))).resolve() != heavy_root
        or Path(str(decision.get("heavy_root", ""))).resolve() != heavy_root
        or len(case_ids) != EXPECTED_CASES
        or len(set(case_ids)) != EXPECTED_CASES
        or decision.get("case_ids") != case_ids
        or source.get("ixi_test_split_accessed") is not False
        or decision.get("ixi_test_split_accessed") is not False
        or source.get("test_115_authorized") is not False
        or decision.get("test_115_authorized") is not False
        or barrier.get("test_split_accessed") is not False
        or barrier.get("decision_workers_received_label_inputs") is not False
    ):
        raise RuntimeError("Frozen C3 source contracts are inconsistent")
    if set(manifest.get("decision_case_sha256") or {}) != set(case_ids) or set(
        manifest.get("evaluation_case_sha256") or {}
    ) != set(case_ids):
        raise RuntimeError("Frozen C3 manifest does not cover exactly val-58")
    if set(barrier.get("decision_case_sha256") or {}) != set(case_ids):
        raise RuntimeError("Frozen C3 decision barrier does not cover exactly val-58")
    return manifest, source, decision, barrier


def _source_case_projection(
    source_dir: Path,
    source_heavy: Path,
    manifest: dict[str, Any],
    source: dict[str, Any],
    decision: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, float], dict[str, Any]]:
    image_inputs = decision.get("image_inputs") or {}
    if set(image_inputs) != {"atlas", *source["case_ids"]}:
        raise RuntimeError("Frozen C3 image cache has the wrong subject set")
    raw_inputs = source.get("raw_inputs") or {}
    if set(raw_inputs) != set(image_inputs):
        raise RuntimeError("Frozen C3 raw-input inventory has the wrong subject set")
    for case_id, row in raw_inputs.items():
        if row.get("dataset") != "IXI" or row.get("split") != ("atlas" if case_id == "atlas" else "val"):
            raise RuntimeError(f"Forbidden source split in frozen C3: {case_id}")
        _verify_file_record(row)
    for row in image_inputs.values():
        _verify_image_record(row, source_heavy)

    initial: dict[str, Any] = {}
    historical: dict[str, Any] = {}
    baseline_dice: dict[str, float] = {}
    reference_goldens: dict[str, Any] = {}
    for case_id in source["case_ids"]:
        decision_path = source_dir / "cases" / case_id / "decision_complete.json"
        evaluation_path = source_dir / "cases" / case_id / "evaluation_complete.json"
        expected_decision = _require_sha(manifest["decision_case_sha256"][case_id], f"C3 {case_id} decision")
        expected_evaluation = _require_sha(manifest["evaluation_case_sha256"][case_id], f"C3 {case_id} evaluation")
        if sha256_file(decision_path) != expected_decision or sha256_file(evaluation_path) != expected_evaluation:
            raise RuntimeError(f"Frozen C3 case marker changed: {case_id}")
        decision_case = _load_json(decision_path)
        evaluation_case = _load_json(evaluation_path)
        if (
            decision_case.get("schema") != "ctcf-search-c3a-decision-case-v1"
            or decision_case.get("status") != "COMPLETE"
            or decision_case.get("case_id") != case_id
            or decision_case.get("labels_loaded_to_device") is not False
            or decision_case.get("test_split_accessed") is not False
            or evaluation_case.get("schema") != "ctcf-search-c3a-evaluation-case-v1"
            or evaluation_case.get("status") != "COMPLETE"
            or evaluation_case.get("case_id") != case_id
            or evaluation_case.get("labels_loaded_after_barrier") is not True
            or evaluation_case.get("test_split_accessed") is not False
        ):
            raise RuntimeError(f"Frozen C3 case is not a valid label-isolated source: {case_id}")
        initial_row = (decision_case.get("initial") or {}).get("field") or {}
        initial_exact = ((decision_case.get("initial") or {}).get("report") or {}).get("psi_exact") or {}
        raw_conf = next((row for row in decision_case.get("arms") or [] if row.get("arm_id") == "raw_conf_post1"), None)
        raw_mean = next(
            (row for row in decision_case.get("arms") or [] if row.get("arm_id") == "raw_mean_normmatched_post1"),
            None,
        )
        historical_row = ((raw_conf or {}).get("requested_state") or {}).get("field") or {}
        historical_mean_row = ((raw_mean or {}).get("requested_state") or {}).get("field") or {}
        if (
            initial_exact.get("status") != "CERTIFIED"
            or initial_exact.get("certified") is not True
            or initial_exact.get("sha256") != initial_row.get("array_sha256")
            or not historical_row
            or not historical_mean_row
        ):
            raise RuntimeError(f"Frozen C3 initial/raw-confidence evidence is invalid: {case_id}")
        _resolve_field(source_heavy, initial_row)
        _resolve_field(source_heavy, historical_row)
        _resolve_field(source_heavy, historical_mean_row)
        initial[case_id] = {"field": initial_row, "exact": initial_exact}
        historical[case_id] = {
            "raw_conf_requested_field": historical_row,
            "raw_mean_common_rms_requested_field": historical_mean_row,
            "source_decision_case_sha256": expected_decision,
        }
        eval_arms = evaluation_case.get("arms") or []
        baselines = [_finite_dice(row.get("baseline_dice"), f"{case_id}/baseline") for row in eval_arms]
        if not baselines or any(value != baselines[0] for value in baselines[1:]):
            raise RuntimeError(f"Frozen C3 evaluation has no unique baseline Dice: {case_id}")
        baseline_dice[case_id] = baselines[0]
        decision_by_arm = {row["arm_id"]: row for row in decision_case["arms"]}
        evaluation_by_arm = {row["arm_id"]: row for row in evaluation_case["arms"]}
        baseline_geometry = decision_case["initial"]["geometry"]
        case_references: dict[str, Any] = {}
        for reference_id, source_arm_id in (
            ("legacy_conf", "raw_conf_post1"),
            ("legacy_mean_common_rms", "raw_mean_normmatched_post1"),
            ("c1_geometry_reference", "c1_raw_conf_post1"),
        ):
            source_decision = decision_by_arm[source_arm_id]
            source_evaluation = evaluation_by_arm[source_arm_id]
            accepted = source_decision["action"] == "ACCEPT"
            requested_geometry = source_decision["requested_state"]["geometry"]
            candidate_geometry = source_decision["candidate_geometry"]
            returned_geometry = candidate_geometry if accepted else baseline_geometry
            case_references[reference_id] = {
                "source_arm_id": source_arm_id,
                "action": source_decision["action"],
                "requested_dice": source_evaluation["requested_diagnostic_dice"],
                "candidate_dice": source_evaluation["capacity_candidate_dice"],
                "returned_dice": source_evaluation["primary_returned_dice"],
                "requested_primary_geometry": requested_geometry[MATHEMATICAL_SDLOGJ_CROP2],
                "candidate_primary_geometry": candidate_geometry[MATHEMATICAL_SDLOGJ_CROP2],
                "returned_primary_geometry": returned_geometry[MATHEMATICAL_SDLOGJ_CROP2],
            }
        reference_goldens[case_id] = case_references
    return initial, historical, baseline_dice, reference_goldens


def prepare_stage(args: argparse.Namespace) -> int:
    assert_frozen_policy()
    if args.num_shards < 1 or args.min_free_gib < 1:
        raise ValueError("--num-shards and --min-free-gib must be positive")
    physical = parse_physical_gpus(
        args.physical_gpus,
        args.num_shards,
        "--physical-gpus must contain one unique non-negative index per shard",
    )
    if git("status", "--porcelain=v1"):
        raise RuntimeError("NUMSTAB prepare refuses a dirty Git tree")
    head = git("rev-parse", "HEAD")
    source_dir = args.source_c3_dir.resolve()
    source_heavy = args.source_c3_heavy_root.resolve()
    root = args.run_root.resolve()
    heavy = args.heavy_root.resolve()
    roots = (source_dir, source_heavy, root, heavy)
    if any(
        first == second or first in second.parents or second in first.parents
        for index, first in enumerate(roots)
        for second in roots[index + 1 :]
    ):
        raise ValueError("Source/target compact and heavy roots must be separate and non-nested")
    heavy.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(heavy).free < int(args.min_free_gib) * 1024**3:
        raise RuntimeError(f"NUMSTAB requires at least {args.min_free_gib} GiB free under {heavy}")
    manifest, c3_source, c3_decision, _ = _validate_source_c3(
        source_dir,
        source_heavy,
        args.source_c3_manifest_sha256,
        args.source_c3_run_manifest_sha256,
    )
    initial, historical, baseline_dice, reference_goldens = _source_case_projection(
        source_dir,
        source_heavy,
        manifest,
        c3_source,
        c3_decision,
    )
    datasets_csv_text = (source_dir / "datasets.csv").read_text(encoding="utf-8")
    datasets_tsv_text = (source_dir / "datasets.tsv").read_text(encoding="utf-8")
    csv_fields, csv_rows = _table_rows(datasets_csv_text, delimiter=",", label="datasets.csv")
    tsv_fields, tsv_rows = _table_rows(datasets_tsv_text, delimiter="\t", label="datasets.tsv")
    if csv_fields != tsv_fields or csv_rows != tsv_rows:
        raise RuntimeError("Frozen C3 dataset CSV/TSV projections differ semantically")
    case_ids = list(c3_source["case_ids"])
    runtime = _runtime_signature()
    source_contract = {
        "schema": f"ctcf-search-gate-numstab-source-contract-{SCHEMA_VERSION}",
        "protocol_id": PROTOCOL_ID,
        "policy_sha256": NUMERICAL_STABILITY_POLICY_SHA256,
        "git_head": head,
        "runtime_signature": runtime,
        "source_c3": {
            "directory": str(source_dir),
            "heavy_root": str(source_heavy),
            "manifest_sha256": SOURCE_C3_MANIFEST_SHA256,
            "run_manifest_sha256": SOURCE_C3_RUN_MANIFEST_SHA256,
            "run_id": SOURCE_C3_RUN_ID,
            "git_head": SOURCE_C3_GIT_HEAD,
        },
        "raw_inputs": c3_source["raw_inputs"],
        "evaluation_baseline_dice": baseline_dice,
        "reference_goldens": reference_goldens,
        "case_ids": case_ids,
        "ixi_test_split_accessed": False,
        "test_115_authorized": False,
    }
    decision_contract = {
        "schema": f"ctcf-search-gate-numstab-decision-contract-{SCHEMA_VERSION}",
        "protocol_id": PROTOCOL_ID,
        "policy": NUMERICAL_STABILITY_POLICY.to_dict(),
        "policy_sha256": NUMERICAL_STABILITY_POLICY_SHA256,
        "git_head": head,
        "runtime_signature": runtime,
        "source_contract_sha256": None,
        "source_c3_manifest_sha256": SOURCE_C3_MANIFEST_SHA256,
        "source_c3_run_manifest_sha256": SOURCE_C3_RUN_MANIFEST_SHA256,
        "source_c3_heavy_root": str(source_heavy),
        "image_inputs": c3_decision["image_inputs"],
        "source_container_sha256": {
            case_id: c3_source["raw_inputs"][case_id]["sha256"] for case_id in ["atlas", *case_ids]
        },
        "source_initial": initial,
        "source_historical": historical,
        "case_ids": case_ids,
        "seed": c3_source["seed"],
        "num_shards": args.num_shards,
        "physical_gpus": physical,
        "shard_to_physical_gpu": shard_gpu_map(physical),
        "shards": round_robin_shards(case_ids, args.num_shards),
        "heavy_root": str(heavy),
        "metric_ids": list(METRIC_SPECS),
        "decision_contract_contains_label_data": False,
        "decision_worker_uses_raw_containers": False,
        "labels_available_to_decision_workers": False,
        "ixi_test_split_accessed": False,
        "test_115_authorized": False,
    }
    root.mkdir(parents=True, exist_ok=True)
    for name, text in (("datasets.csv", datasets_csv_text), ("datasets.tsv", datasets_tsv_text)):
        path = root / name
        if path.exists() and path.read_text(encoding="utf-8") != text:
            raise RuntimeError(f"Resume refused: NUMSTAB {name} changed")
        if not path.exists():
            atomic_write_text(path, text)
    source_path = root / SOURCE_CONTRACT_NAME
    if source_path.exists() and not _json_equivalent(_load_json(source_path), source_contract):
        raise RuntimeError("Resume refused: NUMSTAB source contract changed")
    if not source_path.exists():
        atomic_write_json(source_path, source_contract)
    source_sha = sha256_file(source_path)
    decision_contract["source_contract_sha256"] = source_sha
    serialized = json.dumps(decision_contract, sort_keys=True).lower()
    if '"dice' in serialized or ".pkl" in serialized or "segmentation" in serialized:
        raise RuntimeError("NUMSTAB decision contract leaked label-derived or raw-container data")
    decision_path = root / DECISION_CONTRACT_NAME
    if decision_path.exists() and not _json_equivalent(_load_json(decision_path), decision_contract):
        raise RuntimeError("Resume refused: NUMSTAB decision contract changed")
    if not decision_path.exists():
        atomic_write_json(decision_path, decision_contract)
    decision_sha = sha256_file(decision_path)
    prepare_path = root / "prepare.json"
    if not prepare_path.exists():
        atomic_write_json(
            prepare_path,
            {
                "schema": f"ctcf-search-gate-numstab-prepare-{SCHEMA_VERSION}",
                "status": "PREPARED",
                "prepared_at_utc": utc_now(),
                "source_contract_sha256": source_sha,
                "decision_contract_sha256": decision_sha,
            },
        )
    print(
        json.dumps(
            {"source_contract_sha256": source_sha, "decision_contract_sha256": decision_sha, "n_cases": len(case_ids)}
        )
    )
    return 0


def _load_source(root: Path, expected_sha: str) -> tuple[dict[str, Any], str]:
    path = root.resolve() / SOURCE_CONTRACT_NAME
    actual = sha256_file(path)
    if actual != _require_sha(expected_sha, "source contract SHA-256"):
        raise RuntimeError("NUMSTAB source contract hash mismatch")
    payload = _load_json(path)
    if (
        payload.get("schema") != f"ctcf-search-gate-numstab-source-contract-{SCHEMA_VERSION}"
        or payload.get("protocol_id") != PROTOCOL_ID
        or payload.get("policy_sha256") != NUMERICAL_STABILITY_POLICY_SHA256
        or (payload.get("source_c3") or {}).get("manifest_sha256") != SOURCE_C3_MANIFEST_SHA256
        or (payload.get("source_c3") or {}).get("run_manifest_sha256") != SOURCE_C3_RUN_MANIFEST_SHA256
        or payload.get("ixi_test_split_accessed") is not False
        or payload.get("test_115_authorized") is not False
    ):
        raise RuntimeError("Invalid NUMSTAB source contract")
    return payload, actual


def _load_decision(root: Path, expected_sha: str) -> tuple[dict[str, Any], str]:
    path = root.resolve() / DECISION_CONTRACT_NAME
    actual = sha256_file(path)
    if actual != _require_sha(expected_sha, "decision contract SHA-256"):
        raise RuntimeError("NUMSTAB decision contract hash mismatch")
    payload = _load_json(path)
    if (
        payload.get("schema") != f"ctcf-search-gate-numstab-decision-contract-{SCHEMA_VERSION}"
        or payload.get("protocol_id") != PROTOCOL_ID
        or payload.get("policy_sha256") != NUMERICAL_STABILITY_POLICY_SHA256
        or not _json_equivalent(payload.get("policy"), NUMERICAL_STABILITY_POLICY.to_dict())
        or payload.get("source_c3_manifest_sha256") != SOURCE_C3_MANIFEST_SHA256
        or payload.get("source_c3_run_manifest_sha256") != SOURCE_C3_RUN_MANIFEST_SHA256
        or payload.get("decision_contract_contains_label_data") is not False
        or payload.get("decision_worker_uses_raw_containers") is not False
        or payload.get("labels_available_to_decision_workers") is not False
        or payload.get("ixi_test_split_accessed") is not False
        or payload.get("test_115_authorized") is not False
    ):
        raise RuntimeError("Invalid NUMSTAB decision contract")
    serialized = json.dumps(payload, sort_keys=True).lower()
    if '"dice' in serialized or ".pkl" in serialized or "segmentation" in serialized:
        raise RuntimeError("NUMSTAB decision contract contains forbidden label/raw-container data")
    return payload, actual


def _geometry_bundle(field: torch.Tensor, mask: torch.Tensor) -> dict[str, dict[str, Any]]:
    return {
        metric_id: metric_envelope(
            metric_id,
            lambda mid=metric_id: compute_metric(
                mid,
                field,
                mask if mid == LEARN2REG_SHIFTED_SDLOGJ_MASKED else None,
            ),
        ).to_dict()
        for metric_id in METRIC_SPECS
    }


def _check_exact_metrics(bundle: dict[str, dict[str, Any]], exact: dict[str, Any], label: str) -> bool:
    certified = exact.get("status") == "CERTIFIED" and exact.get("certified") is True
    if certified:
        failures = [metric_id for metric_id, row in bundle.items() if row.get("status") != "OK"]
        if failures:
            raise RuntimeError(f"INTEGRITY_CONFLICT: exact certificate disagrees with metrics for {label}: {failures}")
        digital = bundle[DIGITAL_DECOMPOSITION]
        corner = float((digital.get("components") or {}).get("corner_union_violation_fraction", float("nan")))
        if not math.isfinite(corner) or corner > 0.0:
            raise RuntimeError(f"INTEGRITY_CONFLICT: exact certificate disagrees with digital corners for {label}")
    return certified


def _support_records(supports: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        key: {
            "utility_id": value.utility_id,
            "window": value.window,
            "baseline_count": value.baseline_count,
            "pair_count": value.pair_count,
            "retention": value.retention,
        }
        for key, value in supports.items()
    }


def _common_utility(
    fixed_norm: torch.Tensor,
    moving_norm: torch.Tensor,
    fixed_mind: torch.Tensor,
    moving_mind: torch.Tensor,
    baseline: torch.Tensor,
    state: torch.Tensor,
    supports: dict[str, Any],
) -> dict[str, float | None]:
    output: dict[str, float | None] = {}
    if supports["mind"].pair_count:
        output["mind_baseline_common"] = mind_distance_from_features(
            fixed_mind, moving_mind, baseline, supports["mind"].pair_mask
        )
        output["mind_candidate_common"] = mind_distance_from_features(
            fixed_mind, moving_mind, state, supports["mind"].pair_mask
        )
    else:
        output["mind_baseline_common"] = output["mind_candidate_common"] = None
    for name, window in (("ncc7", 7), ("ncc9", 9)):
        if supports[name].pair_count:
            output[f"{name}_baseline_common"] = ncc_loss_from_normalized(
                fixed_norm, moving_norm, baseline, supports[name].pair_mask, win=window, eps=NCC_EPS
            )
            output[f"{name}_candidate_common"] = ncc_loss_from_normalized(
                fixed_norm, moving_norm, state, supports[name].pair_mask, win=window, eps=NCC_EPS
            )
        else:
            output[f"{name}_baseline_common"] = output[f"{name}_candidate_common"] = None
    if not all(value is None or math.isfinite(float(value)) for value in output.values()):
        raise RuntimeError("NUMSTAB common-support utility is non-finite")
    return output


def _validate_support_utility(supports: dict[str, Any], utility: dict[str, Any], label: str) -> None:
    expected = {
        "mind": ("COMMON_MIND_SSC", 1, "mind_baseline_common", "mind_candidate_common"),
        "ncc7": ("COMMON_NCC7", 7, "ncc7_baseline_common", "ncc7_candidate_common"),
        "ncc9": ("COMMON_NCC9", 9, "ncc9_baseline_common", "ncc9_candidate_common"),
    }
    names = {name for values in expected.values() for name in values[2:]}
    if set(supports) != set(expected) or set(utility) != names:
        raise RuntimeError(f"Invalid NUMSTAB support/utility schema: {label}")
    for key, (utility_id, window, baseline_name, candidate_name) in expected.items():
        row = supports[key]
        baseline_count = row.get("baseline_count")
        pair_count = row.get("pair_count")
        retention = row.get("retention")
        if (
            row.get("utility_id") != utility_id
            or row.get("window") != window
            or isinstance(baseline_count, bool)
            or not isinstance(baseline_count, int)
            or baseline_count < 1
            or isinstance(pair_count, bool)
            or not isinstance(pair_count, int)
            or not 0 <= pair_count <= baseline_count
            or isinstance(retention, bool)
            or not isinstance(retention, (int, float))
            or not math.isfinite(float(retention))
            or not math.isclose(float(retention), pair_count / baseline_count, rel_tol=0.0, abs_tol=1e-15)
        ):
            raise RuntimeError(f"Invalid NUMSTAB support counts: {label}/{key}")
        values = (utility.get(baseline_name), utility.get(candidate_name))
        if pair_count == 0:
            if values != (None, None):
                raise RuntimeError(f"Empty NUMSTAB support has non-null utility: {label}/{key}")
        elif any(
            isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value))
            for value in values
        ):
            raise RuntimeError(f"Non-empty NUMSTAB support has invalid utility: {label}/{key}")


def _case_path(root: Path, case_id: str, phase: str) -> Path:
    return root / "cases" / case_id / f"{phase}_complete.json"


def _worker_paths(root: Path, phase: str, attempt_id: str, shard: int) -> tuple[Path, Path]:
    directory = root / "workers" / phase / "attempts" / attempt_id
    return directory / f"worker_{shard:02d}.json", directory / f"worker_{shard:02d}_failure.json"


def _postclip_oracle_pairs(
    arms: list[dict[str, Any]],
    heavy_root: Path,
    geometry_mask_cpu: torch.Tensor,
) -> list[dict[str, Any]]:
    by_id = {row.get("arm_id"): row for row in arms}
    if set(by_id) != {spec.arm_id for spec in SCIENTIFIC_ARMS}:
        raise RuntimeError("NUMSTAB post-clip oracle comparison has the wrong arm set")
    records: list[dict[str, Any]] = []
    for c32_spec in (spec for spec in SCIENTIFIC_ARMS if spec.selectable):
        oracle_specs = [
            spec
            for spec in SCIENTIFIC_ARMS
            if spec.role == "precision_oracle" and spec.decoder_semantics == c32_spec.decoder_semantics
        ]
        if len(oracle_specs) != 1:
            raise RuntimeError(f"NUMSTAB has no unique FP64 oracle for {c32_spec.arm_id}")
        oracle_spec = oracle_specs[0]
        c32_row = by_id[c32_spec.arm_id]
        oracle_row = by_id[oracle_spec.arm_id]
        c32 = load_flow_npz(_resolve_field(heavy_root, c32_row["candidate"]["field"]))
        oracle = load_flow_npz(_resolve_field(heavy_root, oracle_row["candidate"]["field"]))
        difference = field_difference(c32, oracle, geometry_mask_cpu)
        action_equal = c32_row.get("action") == oracle_row.get("action")
        faithful = bool(float(difference["max_abs"]) <= ORACLE_FAITHFUL_ATOL and action_equal)
        records.append(
            {
                "decoder_semantics": c32_spec.decoder_semantics,
                "c32_arm_id": c32_spec.arm_id,
                "fp64_oracle_arm_id": oracle_spec.arm_id,
                **{f"candidate_{key}": value for key, value in difference.items()},
                "action_equal": action_equal,
                "faithful": faithful,
            }
        )
    return records


def _run_decision_case(
    case_id: str,
    atlas_image: np.ndarray,
    case_image: np.ndarray,
    device: torch.device,
    root: Path,
    contract: dict[str, Any],
    contract_sha: str,
    execution: dict[str, Any],
) -> None:
    marker = _case_path(root, case_id, "decision")
    if marker.is_file():
        _validate_decision_case(_load_json(marker), marker, case_id, contract, contract_sha, verify_heavy=True)
        return
    started = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    moving = torch.from_numpy(atlas_image).unsqueeze(0).to(device)
    fixed = torch.from_numpy(case_image).unsqueeze(0).to(device)
    source_heavy = Path(contract["source_c3_heavy_root"])
    initial_record = contract["source_initial"][case_id]["field"]
    initial = load_flow_npz(_resolve_field(source_heavy, initial_record)).to(device)
    if _sha256_array(initial) != initial_record["array_sha256"]:
        raise RuntimeError(f"Frozen initial array changed: {case_id}")
    mask = geometry_mask(tuple(fixed.shape[-3:]), COLLAR_WIDTH, device)
    fixed_norm = masked_zscore(fixed, mask, std_floor=IMAGE_STD_FLOOR)
    moving_norm = masked_zscore(moving, mask, std_floor=IMAGE_STD_FLOOR)
    fixed_mind = mind_ssc(fixed_norm, radius=1, dilation=2)
    moving_mind = mind_ssc(moving_norm, radius=1, dilation=2)
    raw = build_raw_mind_cost_volume(fixed_mind, moving_mind, initial, mask)

    direct = build_proposal(
        fixed,
        moving,
        initial,
        mask,
        feature="mind",
        orientation="target_centered",
        fixed_feature_override=fixed_mind,
        moving_feature_override=moving_mind,
    )
    direct_residual = 2.0 * direct.displacement
    study = build_reduction_study(raw, mask, legacy_confidence_reference=direct_residual)
    parity = field_difference(direct_residual, study.factorial_residuals["F000"], mask)
    historical_state = (initial + direct_residual).float()
    expected_historical = contract["source_historical"][case_id]["raw_conf_requested_field"]["array_sha256"]
    historical_hash_equal = _sha256_array(historical_state) == expected_historical
    historical_mean_state = (initial + study.historical_requested["legacy_mean_common_rms"]).float()
    expected_historical_mean = contract["source_historical"][case_id]["raw_mean_common_rms_requested_field"][
        "array_sha256"
    ]
    historical_mean_hash_equal = _sha256_array(historical_mean_state) == expected_historical_mean
    if float(parity["max_abs"]) > LEGACY_PARITY_ATOL or not historical_hash_equal or not historical_mean_hash_equal:
        raise RuntimeError(f"LEGACY_PARITY_GAP: direct/study/C3 raw confidence or matched mean diverged for {case_id}")

    sentinel_expected = SENTINEL_ALL_VECTORIZED_GAPS.get(case_id)
    sentinel_observed: float | None = None
    if sentinel_expected is not None:
        sentinel_observed = float(field_difference(study.factorial_residuals["F111"], direct_residual, mask)["max_abs"])
    sentinel = _e54_sentinel_record(case_id, sentinel_observed)
    if sentinel["pass"] is False:
        raise RuntimeError(
            "E54_SENTINEL_GAP_CHANGED: "
            f"{case_id}: observed={sentinel['observed_max_abs']}, expected={sentinel['expected_max_abs']}, "
            f"absolute_error={sentinel['absolute_error']}, tolerance={sentinel['absolute_tolerance']}"
        )

    baseline_valid = valid_sample_mask(initial)
    baseline_geometry = _geometry_bundle(initial, mask)
    _check_exact_metrics(baseline_geometry, contract["source_initial"][case_id]["exact"], f"{case_id}/baseline")
    heavy_root = Path(contract["heavy_root"])
    heavy_case = heavy_root / "cases" / case_id
    arms: list[dict[str, Any]] = []
    for spec in SCIENTIFIC_ARMS:
        residual = study.scientific_requested[spec.arm_id]
        requested_residual_rms = masked_vector_rms(residual, mask)
        requested_path = heavy_case / "requested" / f"{spec.arm_id}.npz"
        requested_stored, requested_exact = save_reload_certify(
            (initial + residual).float(), requested_path, EXACT_CLAIM_EPS
        )
        requested = requested_stored.to(device)
        requested_field = _field_record(requested_path, heavy_root, requested_exact["sha256"])
        requested_geometry = _geometry_bundle(requested, mask)
        requested_certified = _check_exact_metrics(
            requested_geometry, requested_exact, f"{case_id}/{spec.arm_id}/requested"
        )
        requested_valid = valid_sample_mask(requested)
        requested_supports = build_c3a_supports(mask, baseline_valid, requested_valid)
        requested_utility = _common_utility(
            fixed_norm, moving_norm, fixed_mind, moving_mind, initial, requested, requested_supports
        )

        candidate_raw, clip_report = certified_local_clip_candidate(
            initial,
            residual,
            mask,
            work_eps=WORK_EPS,
            sweeps=LOCAL_CLIP_SWEEPS,
        )
        candidate_path = heavy_case / "candidate" / f"{spec.arm_id}.npz"
        candidate_stored, exact = save_reload_certify(candidate_raw, candidate_path, EXACT_CLAIM_EPS)
        candidate = candidate_stored.to(device)
        candidate_field = _field_record(candidate_path, heavy_root, exact["sha256"])
        candidate_geometry = _geometry_bundle(candidate, mask)
        exact_certified = _check_exact_metrics(candidate_geometry, exact, f"{case_id}/{spec.arm_id}/candidate")
        candidate_valid = valid_sample_mask(candidate)
        supports = build_c3a_supports(mask, baseline_valid, candidate_valid)
        utility = _common_utility(fixed_norm, moving_norm, fixed_mind, moving_mind, initial, candidate, supports)
        primary = primary_ncc_decision(
            exact_certified=exact_certified,
            support_retention=supports["ncc7"].retention,
            baseline_ncc_loss=utility["ncc7_baseline_common"],
            candidate_ncc_loss=utility["ncc7_candidate_common"],
        )
        arms.append(
            {
                "arm_index": spec.arm_index,
                "arm_id": spec.arm_id,
                "spec": asdict(spec),
                "requested": {
                    "field": requested_field,
                    "exact": requested_exact,
                    "exact_certified": requested_certified,
                    "fast_cert_bound_work_eps": trilinear_cert_bound(requested, eps=WORK_EPS),
                    "requested_residual_rms": requested_residual_rms,
                    "stored_requested_residual_rms": masked_vector_rms(requested - initial, mask),
                    "geometry": requested_geometry,
                    "supports": _support_records(requested_supports),
                    "utility": requested_utility,
                },
                "candidate": {
                    "field": candidate_field,
                    "exact": exact,
                    "exact_certified": exact_certified,
                    "geometry": candidate_geometry,
                    "supports": _support_records(supports),
                    "utility": utility,
                    "clip_report": clip_report,
                    "postclip_realized_residual_rms": masked_vector_rms(candidate - initial, mask),
                },
                "primary_decision": primary.to_dict(),
                "action": "ACCEPT" if primary.accept else "ROLLBACK",
                "returned_field": candidate_field if primary.accept else initial_record,
                "returned_root": "target" if primary.accept else "source_c3",
                "rollback_byte_identical": not primary.accept,
            }
        )

    postclip_oracle_pairs = _postclip_oracle_pairs(arms, heavy_root, mask.cpu())

    payload = {
        "schema": f"ctcf-search-gate-numstab-decision-case-{SCHEMA_VERSION}",
        "status": "COMPLETE",
        "case_id": case_id,
        "decision_contract_sha256": contract_sha,
        "labels_loaded_to_device": False,
        "test_split_accessed": False,
        "source_initial": contract["source_initial"][case_id],
        "legacy_parity": {
            **parity,
            "atol": LEGACY_PARITY_ATOL,
            "historical_conf_array_sha256_equal": historical_hash_equal,
            "historical_mean_common_rms_array_sha256_equal": historical_mean_hash_equal,
        },
        "e54_sentinel": sentinel,
        "reduction_oracle_faithful": study.oracle_faithful,
        "postclip_oracle_pairs": postclip_oracle_pairs,
        "postclip_oracle_faithful": all(row["faithful"] for row in postclip_oracle_pairs),
        "normalization_rows": study.normalization_rows,
        "factorial_cell_rows": study.factorial_cell_rows,
        "factorial_edge_rows": study.factorial_edge_rows,
        "scientific_rows": study.scientific_rows,
        "baseline_geometry": baseline_geometry,
        "arms": arms,
        "execution": {
            **execution,
            "elapsed_sec": time.perf_counter() - started,
            "peak_gpu_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0,
        },
    }
    if '"dice' in json.dumps(payload, sort_keys=True).lower():
        raise RuntimeError("NUMSTAB decision marker contains forbidden label-derived data")
    atomic_write_json(marker, payload)
    if device.type == "cuda":
        torch.cuda.empty_cache()


def _validate_decision_case(
    payload: dict[str, Any],
    path: Path,
    case_id: str,
    contract: dict[str, Any],
    contract_sha: str,
    *,
    verify_heavy: bool,
) -> list[dict[str, Any]]:
    arms = payload.get("arms") or []
    normalization_rows = payload.get("normalization_rows") or []
    factorial_cell_rows = payload.get("factorial_cell_rows") or []
    factorial_edge_rows = payload.get("factorial_edge_rows") or []
    scientific_rows = payload.get("scientific_rows") or []
    postclip_oracle_pairs = payload.get("postclip_oracle_pairs") or []
    if (
        payload.get("schema") != f"ctcf-search-gate-numstab-decision-case-{SCHEMA_VERSION}"
        or payload.get("status") != "COMPLETE"
        or payload.get("case_id") != case_id
        or payload.get("decision_contract_sha256") != contract_sha
        or payload.get("labels_loaded_to_device") is not False
        or payload.get("test_split_accessed") is not False
        or payload.get("source_initial") != contract["source_initial"][case_id]
        or [(row.get("arm_index"), row.get("arm_id")) for row in arms]
        != [(spec.arm_index, spec.arm_id) for spec in SCIENTIFIC_ARMS]
        or [row.get("moment_reduction") for row in normalization_rows]
        != [MOMENT_LEGACY, MOMENT_VECTORIZED, MOMENT_CENTERED_FP32, MOMENT_CENTERED_FP64]
        or [
            (
                row.get("cell_id"),
                row.get("moment_reduction"),
                row.get("posterior_reduction"),
                row.get("decoder_reduction"),
            )
            for row in factorial_cell_rows
        ]
        != [
            (spec.cell_id, spec.moment_reduction, spec.posterior_reduction, spec.decoder_reduction)
            for spec in FACTORIAL_SPECS
        ]
        or [(row.get("axis"), row.get("source_cell_id"), row.get("target_cell_id")) for row in factorial_edge_rows]
        != list(FACTORIAL_EDGES)
        or len(scientific_rows) != len(SCIENTIFIC_ARMS)
        or any(
            any(row.get(key) != value for key, value in asdict(spec).items())
            for spec, row in zip(SCIENTIFIC_ARMS, scientific_rows, strict=True)
        )
        or not isinstance(payload.get("reduction_oracle_faithful"), bool)
        or len(postclip_oracle_pairs) != 2
        or not isinstance(payload.get("postclip_oracle_faithful"), bool)
    ):
        raise RuntimeError(f"Invalid NUMSTAB decision marker: {path}")
    fp32_oracle_errors = [
        row.get("difference_vs_fp64_max_abs")
        for spec, row in zip(SCIENTIFIC_ARMS, scientific_rows, strict=True)
        if spec.moment_reduction == MOMENT_CENTERED_FP32
    ]
    if any(
        isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value))
        for value in fp32_oracle_errors
    ):
        raise RuntimeError(f"Invalid NUMSTAB oracle diagnostics: {path}")
    expected_oracle_faithful = all(float(value) <= ORACLE_FAITHFUL_ATOL for value in fp32_oracle_errors)
    if payload["reduction_oracle_faithful"] is not expected_oracle_faithful:
        raise RuntimeError(f"NUMSTAB oracle-faithful flag disagrees with scientific diagnostics: {path}")
    if '"dice' in json.dumps(payload, sort_keys=True).lower():
        raise RuntimeError(f"NUMSTAB decision marker contains forbidden label data: {path}")
    parity = payload.get("legacy_parity") or {}
    if (
        not isinstance(parity.get("max_abs"), (int, float))
        or float(parity["max_abs"]) > LEGACY_PARITY_ATOL
        or parity.get("historical_conf_array_sha256_equal") is not True
        or parity.get("historical_mean_common_rms_array_sha256_equal") is not True
    ):
        raise RuntimeError(f"NUMSTAB legacy parity is not established: {case_id}")
    _validate_e54_sentinel(payload.get("e54_sentinel"), case_id)
    source_heavy = Path(contract["source_c3_heavy_root"])
    target_heavy = Path(contract["heavy_root"])
    if verify_heavy:
        _resolve_field(source_heavy, contract["source_initial"][case_id]["field"])
    for spec, row in zip(SCIENTIFIC_ARMS, arms, strict=True):
        requested = row.get("requested") or {}
        candidate = row.get("candidate") or {}
        requested_exact = requested.get("exact") or {}
        candidate_exact = candidate.get("exact") or {}
        requested_supports = requested.get("supports") or {}
        candidate_supports = candidate.get("supports") or {}
        requested_utility = requested.get("utility") or {}
        candidate_utility = candidate.get("utility") or {}
        _validate_support_utility(requested_supports, requested_utility, f"{case_id}/{spec.arm_id}/requested")
        _validate_support_utility(candidate_supports, candidate_utility, f"{case_id}/{spec.arm_id}/candidate")
        primary = primary_ncc_decision(
            exact_certified=candidate.get("exact_certified"),
            support_retention=float((candidate_supports.get("ncc7") or {}).get("retention", float("nan"))),
            baseline_ncc_loss=candidate_utility.get("ncc7_baseline_common"),
            candidate_ncc_loss=candidate_utility.get("ncc7_candidate_common"),
        )
        if (
            row.get("spec") != asdict(spec)
            or requested.get("exact_certified")
            is not (requested_exact.get("status") == "CERTIFIED" and requested_exact.get("certified") is True)
            or candidate.get("exact_certified")
            is not (candidate_exact.get("status") == "CERTIFIED" and candidate_exact.get("certified") is True)
            or requested_exact.get("sha256") != (requested.get("field") or {}).get("array_sha256")
            or candidate_exact.get("sha256") != (candidate.get("field") or {}).get("array_sha256")
            or row.get("primary_decision") != primary.to_dict()
            or row.get("action") != ("ACCEPT" if primary.accept else "ROLLBACK")
            or row.get("rollback_byte_identical") is not (not primary.accept)
            or set(requested.get("geometry") or {}) != set(METRIC_SPECS)
            or set(candidate.get("geometry") or {}) != set(METRIC_SPECS)
            or not isinstance(requested.get("fast_cert_bound_work_eps"), (int, float))
            or not math.isfinite(float(requested["fast_cert_bound_work_eps"]))
            or any(
                not isinstance(value, (int, float)) or not math.isfinite(float(value)) or float(value) < 0.0
                for value in (
                    requested.get("requested_residual_rms"),
                    requested.get("stored_requested_residual_rms"),
                    candidate.get("postclip_realized_residual_rms"),
                )
            )
        ):
            raise RuntimeError(f"Invalid NUMSTAB arm record: {case_id}/{spec.arm_id}")
        _check_exact_metrics(requested["geometry"], requested_exact, f"{case_id}/{spec.arm_id}/requested")
        _check_exact_metrics(candidate["geometry"], candidate_exact, f"{case_id}/{spec.arm_id}/candidate")
        expected_returned = candidate["field"] if primary.accept else contract["source_initial"][case_id]["field"]
        expected_root = "target" if primary.accept else "source_c3"
        if row.get("returned_field") != expected_returned or row.get("returned_root") != expected_root:
            raise RuntimeError(f"Invalid NUMSTAB rollback record: {case_id}/{spec.arm_id}")
        if verify_heavy:
            _resolve_field(target_heavy, requested["field"])
            _resolve_field(target_heavy, candidate["field"])
            _resolve_field(target_heavy if primary.accept else source_heavy, row["returned_field"])
    if verify_heavy:
        first_candidate = load_flow_npz(_resolve_field(target_heavy, arms[0]["candidate"]["field"]))
        mask_cpu = geometry_mask(tuple(first_candidate.shape[-3:]), COLLAR_WIDTH, first_candidate.device)
        expected_postclip_pairs = _postclip_oracle_pairs(arms, target_heavy, mask_cpu)
        if postclip_oracle_pairs != expected_postclip_pairs:
            raise RuntimeError(f"NUMSTAB post-clip oracle diagnostics changed: {path}")
    expected_postclip_faithful = all(row.get("faithful") is True for row in postclip_oracle_pairs)
    if payload["postclip_oracle_faithful"] is not expected_postclip_faithful:
        raise RuntimeError(f"NUMSTAB post-clip oracle-faithful flag is inconsistent: {path}")
    execution = payload.get("execution") or {}
    expected_shard = expected_shard_for_case(contract, case_id)
    if (
        execution.get("phase") != "decision"
        or execution.get("shard_index") != expected_shard
        or execution.get("physical_gpu") != contract["shard_to_physical_gpu"][str(expected_shard)]
        or execution.get("deterministic") is not True
        or execution.get("labels_loaded_to_device") is not False
        or not str(execution.get("device", "")).startswith("cuda")
    ):
        raise RuntimeError(f"Invalid NUMSTAB decision provenance: {case_id}")
    return arms


def decision_worker_stage(args: argparse.Namespace) -> int:
    root = args.run_root.resolve()
    contract, contract_sha = _load_decision(root, args.decision_contract_sha256)
    _assert_clean_code(contract["git_head"], "decision worker")
    _assert_runtime(contract["runtime_signature"], "decision worker")
    if args.num_shards != contract["num_shards"] or not 0 <= args.shard_index < args.num_shards:
        raise RuntimeError("NUMSTAB decision shard parameters differ from contract")
    if args.physical_gpu != contract["shard_to_physical_gpu"][str(args.shard_index)]:
        raise RuntimeError("NUMSTAB decision physical GPU differs from contract")
    assigned = contract["shards"][str(args.shard_index)]
    source_heavy = Path(contract["source_c3_heavy_root"])
    for case_id in ["atlas", *assigned]:
        _verify_image_record(contract["image_inputs"][case_id], source_heavy)
    marker, failure = _worker_paths(root, "decision", args.attempt_id, args.shard_index)
    if marker.exists() or failure.exists():
        raise RuntimeError("NUMSTAB decision worker attempt output already exists")
    pending: list[str] = []
    reused: list[str] = []
    for case_id in assigned:
        path = _case_path(root, case_id, "decision")
        if path.is_file():
            _validate_decision_case(_load_json(path), path, case_id, contract, contract_sha, verify_heavy=True)
            reused.append(case_id)
        else:
            pending.append(case_id)
    computed: list[str] = []
    started = utc_now()
    try:
        if pending:
            device = setup_device(args.gpu, seed=contract["seed"], deterministic=True)
            if device.type != "cuda":
                raise RuntimeError("NUMSTAB decision worker requires CUDA")
            atlas = _verify_image_record(contract["image_inputs"]["atlas"], source_heavy)
            execution = {
                "phase": "decision",
                "attempt_id": args.attempt_id,
                "shard_index": args.shard_index,
                "physical_gpu": args.physical_gpu,
                "host": platform.node(),
                "python": platform.python_version(),
                "torch": torch.__version__,
                "device": str(device),
                "gpu_name": torch.cuda.get_device_name(device),
                "seed": contract["seed"],
                "deterministic": True,
                "labels_loaded_to_device": False,
                "source_c3_manifest_sha256": contract["source_c3_manifest_sha256"],
            }
            for index, case_id in enumerate(pending, start=1):
                print(
                    f"[NUMSTAB decision {args.shard_index + 1}/{args.num_shards}] [{index}/{len(pending)}] {case_id}",
                    flush=True,
                )
                image = _verify_image_record(contract["image_inputs"][case_id], source_heavy)
                _run_decision_case(case_id, atlas, image, device, root, contract, contract_sha, execution)
                computed.append(case_id)
        report = {
            "schema": f"ctcf-search-gate-numstab-decision-worker-{SCHEMA_VERSION}",
            "status": "COMPLETE",
            "phase": "decision",
            "attempt_id": args.attempt_id,
            "shard_index": args.shard_index,
            "physical_gpu": args.physical_gpu,
            "decision_contract_sha256": contract_sha,
            "assigned_case_ids": assigned,
            "computed_case_ids": computed,
            "reused_case_ids": reused,
            "labels_loaded_to_device": False,
            "runtime_signature": contract["runtime_signature"],
            "started_at_utc": started,
            "completed_at_utc": utc_now(),
        }
        atomic_write_json(marker, report)
    except Exception as error:
        atomic_write_json(
            failure,
            {
                "schema": f"ctcf-search-gate-numstab-decision-worker-failure-{SCHEMA_VERSION}",
                "status": "FAILED",
                "phase": "decision",
                "attempt_id": args.attempt_id,
                "shard_index": args.shard_index,
                "decision_contract_sha256": contract_sha,
                "computed_case_ids": computed,
                "error_type": type(error).__name__,
                "error": str(error),
                "completed_at_utc": utc_now(),
            },
        )
        raise
    return 0


def _validate_worker(
    payload: dict[str, Any],
    phase: str,
    attempt_id: str,
    shard: int,
    contract: dict[str, Any],
    contract_sha: str,
) -> None:
    assigned = contract["shards"][str(shard)]
    computed = payload.get("computed_case_ids") or []
    reused = payload.get("reused_case_ids") or []
    if (
        payload.get("schema") != f"ctcf-search-gate-numstab-{phase}-worker-{SCHEMA_VERSION}"
        or payload.get("status") != "COMPLETE"
        or payload.get("phase") != phase
        or payload.get("attempt_id") != attempt_id
        or payload.get("shard_index") != shard
        or payload.get("physical_gpu") != contract["shard_to_physical_gpu"][str(shard)]
        or payload.get("decision_contract_sha256") != contract_sha
        or payload.get("assigned_case_ids") != assigned
        or set(computed) & set(reused)
        or sorted([*computed, *reused]) != sorted(assigned)
        or computed != [case_id for case_id in assigned if case_id in set(computed)]
        or reused != [case_id for case_id in assigned if case_id in set(reused)]
        or payload.get("runtime_signature") != contract["runtime_signature"]
    ):
        raise RuntimeError(f"Invalid NUMSTAB {phase} worker report for shard {shard}")
    if phase == "decision" and payload.get("labels_loaded_to_device") is not False:
        raise RuntimeError("NUMSTAB decision worker report claims label access")
    if phase == "evaluation" and (
        payload.get("labels_loaded_this_attempt") is not bool(computed)
        or payload.get("all_cases_have_postbarrier_evaluation_evidence") is not True
    ):
        raise RuntimeError("NUMSTAB evaluation worker has inconsistent label evidence")


def barrier_stage(args: argparse.Namespace) -> int:
    root = args.run_root.resolve()
    contract, contract_sha = _load_decision(root, args.decision_contract_sha256)
    _assert_clean_code(contract["git_head"], "decision barrier")
    worker_files: list[dict[str, str]] = []
    seen: list[str] = []
    for shard in range(contract["num_shards"]):
        path, _ = _worker_paths(root, "decision", args.attempt_id, shard)
        report = _load_json(path)
        _validate_worker(report, "decision", args.attempt_id, shard, contract, contract_sha)
        seen.extend(contract["shards"][str(shard)])
        worker_files.append({"path": path.relative_to(root).as_posix(), "sha256": sha256_file(path)})
    validate_shard_partition(contract, seen, EXPECTED_CASES, "NUMSTAB decision partition changed")
    case_hashes: dict[str, str] = {}
    for case_id in contract["case_ids"]:
        path = _case_path(root, case_id, "decision")
        _validate_decision_case(_load_json(path), path, case_id, contract, contract_sha, verify_heavy=True)
        case_hashes[case_id] = sha256_file(path)
    path = root / BARRIER_NAME
    if path.exists():
        existing = _load_json(path)
        existing_attempt = existing.get("attempt_id")
        existing_workers = existing.get("workers") or []
        if (
            existing.get("schema") != f"ctcf-search-gate-numstab-decision-barrier-{SCHEMA_VERSION}"
            or existing.get("status") != "COMPLETE"
            or existing.get("protocol_id") != PROTOCOL_ID
            or not isinstance(existing_attempt, str)
            or existing.get("decision_contract_sha256") != contract_sha
            or existing.get("decision_workers_received_label_inputs") is not False
            or existing.get("test_split_accessed") is not False
            or existing.get("decision_case_sha256") != case_hashes
            or len(existing_workers) != contract["num_shards"]
        ):
            raise RuntimeError("Existing NUMSTAB barrier differs from immutable decisions")
        for shard, row in enumerate(existing_workers):
            worker_path = (root / str(row.get("path", ""))).resolve()
            if (
                root not in worker_path.parents
                or not worker_path.is_file()
                or sha256_file(worker_path) != _require_sha(row.get("sha256"), f"barrier worker {shard}")
            ):
                raise RuntimeError(f"Existing NUMSTAB barrier worker changed: shard {shard}")
            _validate_worker(
                _load_json(worker_path),
                "decision",
                existing_attempt,
                shard,
                contract,
                contract_sha,
            )
        print(json.dumps({"decision_barrier_sha256": sha256_file(path), "n_cases": len(case_hashes), "reused": True}))
        return 0
    payload = {
        "schema": f"ctcf-search-gate-numstab-decision-barrier-{SCHEMA_VERSION}",
        "status": "COMPLETE",
        "protocol_id": PROTOCOL_ID,
        "attempt_id": args.attempt_id,
        "decision_contract_sha256": contract_sha,
        "decision_workers_received_label_inputs": False,
        "test_split_accessed": False,
        "workers": worker_files,
        "decision_case_sha256": case_hashes,
        "completed_at_utc": utc_now(),
    }
    atomic_write_json(path, payload)
    print(json.dumps({"decision_barrier_sha256": sha256_file(path), "n_cases": len(case_hashes)}))
    return 0


def _load_barrier(
    root: Path, expected_sha: str, contract: dict[str, Any], contract_sha: str
) -> tuple[dict[str, Any], str]:
    path = root / BARRIER_NAME
    actual = sha256_file(path)
    if actual != _require_sha(expected_sha, "barrier SHA-256"):
        raise RuntimeError("NUMSTAB barrier hash mismatch")
    payload = _load_json(path)
    if (
        payload.get("schema") != f"ctcf-search-gate-numstab-decision-barrier-{SCHEMA_VERSION}"
        or payload.get("status") != "COMPLETE"
        or payload.get("protocol_id") != PROTOCOL_ID
        or payload.get("decision_contract_sha256") != contract_sha
        or payload.get("decision_workers_received_label_inputs") is not False
        or payload.get("test_split_accessed") is not False
        or set(payload.get("decision_case_sha256") or {}) != set(contract["case_ids"])
    ):
        raise RuntimeError("Invalid NUMSTAB decision barrier")
    for case_id, digest in payload["decision_case_sha256"].items():
        if sha256_file(_case_path(root, case_id, "decision")) != digest:
            raise RuntimeError(f"NUMSTAB decision changed after barrier: {case_id}")
    base = root.resolve()
    for row in payload.get("workers") or []:
        worker = (base / str(row.get("path", ""))).resolve()
        if base not in worker.parents or sha256_file(worker) != _require_sha(row.get("sha256"), str(worker)):
            raise RuntimeError(f"NUMSTAB barrier-authenticated worker report changed: {worker}")
    return payload, actual


def _run_evaluation_case(
    index: int,
    case_id: str,
    dataset: Any,
    labels: tuple[int, ...],
    device: torch.device,
    root: Path,
    source: dict[str, Any],
    contract: dict[str, Any],
    contract_sha: str,
    barrier: dict[str, Any],
    barrier_sha: str,
    execution: dict[str, Any],
) -> None:
    marker = _case_path(root, case_id, "evaluation")
    decision_path = _case_path(root, case_id, "decision")
    frozen_sha = barrier["decision_case_sha256"][case_id]
    if sha256_file(decision_path) != frozen_sha:
        raise RuntimeError(f"NUMSTAB decision changed before evaluation: {case_id}")
    decision = _load_json(decision_path)
    arms = _validate_decision_case(decision, decision_path, case_id, contract, contract_sha, verify_heavy=True)
    moving_image, fixed_image, moving_seg_cpu, fixed_seg_cpu = dataset[index]
    if (
        _sha256_array(moving_image.numpy()) != contract["image_inputs"]["atlas"]["array_sha256"]
        or _sha256_array(fixed_image.numpy()) != contract["image_inputs"][case_id]["array_sha256"]
    ):
        raise RuntimeError(f"NUMSTAB evaluation images differ from decision cache: {case_id}")
    moving_seg = moving_seg_cpu.unsqueeze(0).to(device)
    fixed_seg = fixed_seg_cpu.unsqueeze(0).to(device)
    source_heavy = Path(contract["source_c3_heavy_root"])
    target_heavy = Path(contract["heavy_root"])
    initial = load_flow_npz(_resolve_field(source_heavy, contract["source_initial"][case_id]["field"])).to(device)
    baseline = dice_score(initial, moving_seg, fixed_seg, labels)
    expected_baseline = _finite_dice(source["evaluation_baseline_dice"][case_id], f"{case_id}/source baseline")
    if not math.isclose(baseline, expected_baseline, rel_tol=0.0, abs_tol=SOURCE_BASELINE_DICE_ATOL):
        raise RuntimeError(f"PARITY_GAP: NUMSTAB baseline differs from frozen C3: {case_id}")
    rows: list[dict[str, Any]] = []
    for row in arms:
        requested = load_flow_npz(_resolve_field(target_heavy, row["requested"]["field"])).to(device)
        candidate = load_flow_npz(_resolve_field(target_heavy, row["candidate"]["field"])).to(device)
        requested_score = dice_score(requested, moving_seg, fixed_seg, labels)
        candidate_score = dice_score(candidate, moving_seg, fixed_seg, labels)
        returned_score = candidate_score if row["action"] == "ACCEPT" else baseline
        rows.append(
            {
                "arm_index": row["arm_index"],
                "arm_id": row["arm_id"],
                "baseline_dice": baseline,
                "requested_diagnostic_dice": requested_score,
                "requested_diagnostic_dice_delta": requested_score - baseline,
                "capacity_candidate_dice": candidate_score,
                "capacity_dice_delta": candidate_score - baseline,
                "primary_returned_dice": returned_score,
                "primary_dice_delta": returned_score - baseline,
                "primary_action": row["action"],
            }
        )
    if sha256_file(decision_path) != frozen_sha:
        raise RuntimeError(f"NUMSTAB decision changed during evaluation: {case_id}")
    atomic_write_json(
        marker,
        {
            "schema": f"ctcf-search-gate-numstab-evaluation-case-{SCHEMA_VERSION}",
            "status": "COMPLETE",
            "case_id": case_id,
            "decision_contract_sha256": contract_sha,
            "decision_barrier_sha256": barrier_sha,
            "decision_case_sha256": frozen_sha,
            "source_input_sha256": contract["source_container_sha256"][case_id],
            "labels_loaded_after_barrier": True,
            "test_split_accessed": False,
            "labels": list(labels),
            "source_c3_baseline_parity_verified": True,
            "arms": rows,
            "execution": execution,
        },
    )


def _validate_evaluation_case(
    payload: dict[str, Any],
    path: Path,
    case_id: str,
    source: dict[str, Any],
    contract: dict[str, Any],
    contract_sha: str,
    barrier: dict[str, Any],
    barrier_sha: str,
) -> list[dict[str, Any]]:
    from experiments.core.inference_metrics import metric_profile_for

    arms = payload.get("arms") or []
    if (
        payload.get("schema") != f"ctcf-search-gate-numstab-evaluation-case-{SCHEMA_VERSION}"
        or payload.get("status") != "COMPLETE"
        or payload.get("case_id") != case_id
        or payload.get("decision_contract_sha256") != contract_sha
        or payload.get("decision_barrier_sha256") != barrier_sha
        or payload.get("decision_case_sha256") != barrier["decision_case_sha256"][case_id]
        or payload.get("source_input_sha256") != contract["source_container_sha256"][case_id]
        or payload.get("labels_loaded_after_barrier") is not True
        or payload.get("test_split_accessed") is not False
        or payload.get("labels") != list(metric_profile_for("IXI").labels)
        or payload.get("source_c3_baseline_parity_verified") is not True
        or [(row.get("arm_index"), row.get("arm_id")) for row in arms]
        != [(spec.arm_index, spec.arm_id) for spec in SCIENTIFIC_ARMS]
        or sha256_file(path.with_name("decision_complete.json")) != barrier["decision_case_sha256"][case_id]
    ):
        raise RuntimeError(f"Invalid NUMSTAB evaluation marker: {path}")
    decision_payload = _load_json(path.with_name("decision_complete.json"))
    decision_actions = {row["arm_id"]: row["action"] for row in decision_payload.get("arms") or []}
    for row in arms:
        values = [
            row.get(key)
            for key in (
                "baseline_dice",
                "requested_diagnostic_dice",
                "requested_diagnostic_dice_delta",
                "capacity_candidate_dice",
                "capacity_dice_delta",
                "primary_returned_dice",
                "primary_dice_delta",
            )
        ]
        if not all(isinstance(value, (int, float)) and math.isfinite(float(value)) for value in values):
            raise RuntimeError(f"Non-finite NUMSTAB evaluation value: {path}/{row.get('arm_id')}")
        expected_baseline = float(source["evaluation_baseline_dice"][case_id])
        expected_returned = (
            float(row["capacity_candidate_dice"]) if row["primary_action"] == "ACCEPT" else float(row["baseline_dice"])
        )
        if (
            row.get("primary_action") != decision_actions.get(row.get("arm_id"))
            or not math.isclose(
                float(row["baseline_dice"]), expected_baseline, rel_tol=0.0, abs_tol=SOURCE_BASELINE_DICE_ATOL
            )
            or not all(
                0.0 <= float(row[key]) <= 1.0
                for key in (
                    "baseline_dice",
                    "requested_diagnostic_dice",
                    "capacity_candidate_dice",
                    "primary_returned_dice",
                )
            )
            or float(row["requested_diagnostic_dice_delta"])
            != float(row["requested_diagnostic_dice"]) - float(row["baseline_dice"])
            or float(row["capacity_dice_delta"]) != float(row["capacity_candidate_dice"]) - float(row["baseline_dice"])
            or float(row["primary_returned_dice"]) != expected_returned
            or float(row["primary_dice_delta"]) != expected_returned - float(row["baseline_dice"])
        ):
            raise RuntimeError(f"Invalid NUMSTAB evaluation arithmetic: {path}/{row.get('arm_id')}")
    execution = payload.get("execution") or {}
    shard = expected_shard_for_case(contract, case_id)
    if (
        execution.get("phase") != "evaluation"
        or execution.get("shard_index") != shard
        or execution.get("physical_gpu") != contract["shard_to_physical_gpu"][str(shard)]
        or execution.get("labels_loaded_after_barrier") is not True
        or execution.get("deterministic") is not True
        or not str(execution.get("device", "")).startswith("cuda")
    ):
        raise RuntimeError(f"Invalid NUMSTAB evaluation provenance: {case_id}")
    return arms


def evaluation_worker_stage(args: argparse.Namespace) -> int:
    root = args.run_root.resolve()
    source, source_sha = _load_source(root, args.source_contract_sha256)
    contract, contract_sha = _load_decision(root, args.decision_contract_sha256)
    if contract.get("source_contract_sha256") != source_sha:
        raise RuntimeError("NUMSTAB decision/source contracts are not linked")
    _assert_clean_code(contract["git_head"], "evaluation worker")
    _assert_runtime(contract["runtime_signature"], "evaluation worker")
    barrier, barrier_sha = _load_barrier(root, args.barrier_sha256, contract, contract_sha)
    if args.num_shards != contract["num_shards"] or not 0 <= args.shard_index < args.num_shards:
        raise RuntimeError("NUMSTAB evaluation shard parameters differ from contract")
    if args.physical_gpu != contract["shard_to_physical_gpu"][str(args.shard_index)]:
        raise RuntimeError("NUMSTAB evaluation physical GPU differs from contract")
    assigned = contract["shards"][str(args.shard_index)]
    for case_id in ["atlas", *assigned]:
        _verify_file_record(source["raw_inputs"][case_id])
    marker, failure = _worker_paths(root, "evaluation", args.attempt_id, args.shard_index)
    if marker.exists() or failure.exists():
        raise RuntimeError("NUMSTAB evaluation worker attempt output already exists")
    pending: list[str] = []
    reused: list[str] = []
    for case_id in assigned:
        path = _case_path(root, case_id, "evaluation")
        if path.is_file():
            _validate_evaluation_case(
                _load_json(path), path, case_id, source, contract, contract_sha, barrier, barrier_sha
            )
            reused.append(case_id)
        else:
            pending.append(case_id)
    computed: list[str] = []
    started = utc_now()
    try:
        if pending:
            from experiments.core.inference_metrics import metric_profile_for
            from experiments.core.inference_runtime import build_infer_dataset

            device = setup_device(args.gpu, seed=contract["seed"], deterministic=True)
            if device.type != "cuda":
                raise RuntimeError("NUMSTAB evaluation worker requires CUDA")
            dataset = build_infer_dataset(
                "IXI",
                [source["raw_inputs"][case_id]["path"] for case_id in pending],
                source["raw_inputs"]["atlas"]["path"],
            )
            labels = tuple(metric_profile_for("IXI").labels)
            execution = {
                "phase": "evaluation",
                "attempt_id": args.attempt_id,
                "shard_index": args.shard_index,
                "physical_gpu": args.physical_gpu,
                "host": platform.node(),
                "python": platform.python_version(),
                "torch": torch.__version__,
                "device": str(device),
                "gpu_name": torch.cuda.get_device_name(device),
                "seed": contract["seed"],
                "deterministic": True,
                "labels_loaded_after_barrier": True,
            }
            for index, case_id in enumerate(pending):
                print(
                    f"[NUMSTAB evaluation {args.shard_index + 1}/{args.num_shards}] [{index + 1}/{len(pending)}] {case_id}",
                    flush=True,
                )
                _run_evaluation_case(
                    index,
                    case_id,
                    dataset,
                    labels,
                    device,
                    root,
                    source,
                    contract,
                    contract_sha,
                    barrier,
                    barrier_sha,
                    execution,
                )
                computed.append(case_id)
        report = {
            "schema": f"ctcf-search-gate-numstab-evaluation-worker-{SCHEMA_VERSION}",
            "status": "COMPLETE",
            "phase": "evaluation",
            "attempt_id": args.attempt_id,
            "shard_index": args.shard_index,
            "physical_gpu": args.physical_gpu,
            "decision_contract_sha256": contract_sha,
            "decision_barrier_sha256": barrier_sha,
            "assigned_case_ids": assigned,
            "computed_case_ids": computed,
            "reused_case_ids": reused,
            "labels_loaded_this_attempt": bool(computed),
            "all_cases_have_postbarrier_evaluation_evidence": True,
            "runtime_signature": contract["runtime_signature"],
            "started_at_utc": started,
            "completed_at_utc": utc_now(),
        }
        atomic_write_json(marker, report)
    except Exception as error:
        atomic_write_json(
            failure,
            {
                "schema": f"ctcf-search-gate-numstab-evaluation-worker-failure-{SCHEMA_VERSION}",
                "status": "FAILED",
                "phase": "evaluation",
                "attempt_id": args.attempt_id,
                "shard_index": args.shard_index,
                "decision_contract_sha256": contract_sha,
                "decision_barrier_sha256": barrier_sha,
                "computed_case_ids": computed,
                "error_type": type(error).__name__,
                "error": str(error),
                "completed_at_utc": utc_now(),
            },
        )
        raise
    return 0


def _csv_fields(rows: list[dict[str, Any]], preferred: list[str]) -> list[str]:
    keys = {key for row in rows for key in row}
    return [key for key in preferred if key in keys] + sorted(keys - set(preferred))


def _metric_value(bundle: dict[str, Any], metric_id: str) -> float | None:
    row = bundle.get(metric_id) or {}
    value = row.get("value")
    return float(value) if row.get("status") == "OK" and isinstance(value, (int, float)) else None


def _metric_status(bundle: dict[str, Any], metric_id: str) -> str:
    status = (bundle.get(metric_id) or {}).get("status")
    if not isinstance(status, str) or not status:
        raise RuntimeError(f"Missing metric status for {metric_id}")
    return status


def _paired(candidate: list[float], baseline: list[float]) -> Any:
    return paired_summary(
        candidate,
        baseline,
        bootstrap_resamples=BOOTSTRAP_RESAMPLES,
        bootstrap_seed=BOOTSTRAP_SEED,
        confidence=BOOTSTRAP_CONFIDENCE,
    )


def _geometry_contrast_summary(
    rows: list[dict[str, Any]],
    candidate_key: str,
    comparator_key: str,
    *,
    expected_cases: int,
    required: bool,
    label: str,
) -> dict[str, Any]:
    if len(rows) != expected_cases:
        raise RuntimeError(f"NUMSTAB {label} geometry contrast has {len(rows)} cases, expected {expected_cases}")
    candidate_status_key = f"{candidate_key}_status"
    comparator_status_key = f"{comparator_key}_status"
    for row in rows:
        for value_key, status_key in (
            (candidate_key, candidate_status_key),
            (comparator_key, comparator_status_key),
        ):
            if not isinstance(row.get(status_key), str) or (row[value_key] is not None) is not (
                row[status_key] == "OK"
            ):
                raise RuntimeError(f"NUMSTAB {label} has inconsistent geometry value/status evidence")
    candidate_defined = sum(row[candidate_key] is not None for row in rows)
    comparator_defined = sum(row[comparator_key] is not None for row in rows)
    paired_rows = [row for row in rows if row[candidate_key] is not None and row[comparator_key] is not None]
    candidate_statuses = {row[candidate_status_key] for row in rows}
    comparator_statuses = {row[comparator_status_key] for row in rows}
    if required and len(paired_rows) != expected_cases:
        raise RuntimeError(f"NUMSTAB {label} lacks a paired geometry contrast")
    metadata = {
        "status": "OK" if len(paired_rows) == expected_cases else "UNDEFINED_INCOMPLETE_SUPPORT",
        "n_cases": expected_cases,
        "candidate_defined_cases": candidate_defined,
        "comparator_defined_cases": comparator_defined,
        "paired_defined_cases": len(paired_rows),
        "undefined_pair_cases": expected_cases - len(paired_rows),
        "candidate_metric_status": next(iter(candidate_statuses)) if len(candidate_statuses) == 1 else "MIXED",
        "comparator_metric_status": next(iter(comparator_statuses)) if len(comparator_statuses) == 1 else "MIXED",
    }
    if len(paired_rows) == expected_cases:
        return {
            **metadata,
            **_paired(
                [float(row[candidate_key]) for row in paired_rows],
                [float(row[comparator_key]) for row in paired_rows],
            ).to_dict(),
        }
    return {
        **metadata,
        "n": len(paired_rows),
        "mean": None,
        "median": None,
        "ci_low": None,
        "ci_high": None,
        "improved": None,
        "worsened": None,
        "tied": None,
        "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_confidence": BOOTSTRAP_CONFIDENCE,
    }


def finalize_stage(args: argparse.Namespace) -> int:
    root = args.run_root.resolve()
    source, source_sha = _load_source(root, args.source_contract_sha256)
    contract, contract_sha = _load_decision(root, args.decision_contract_sha256)
    if contract.get("source_contract_sha256") != source_sha:
        raise RuntimeError("NUMSTAB source/decision contracts are not linked")
    _assert_clean_code(contract["git_head"], "finalize")
    _assert_runtime(contract["runtime_signature"], "finalize")
    barrier, barrier_sha = _load_barrier(root, args.barrier_sha256, contract, contract_sha)
    source_c3_dir = Path(source["source_c3"]["directory"])
    if (
        sha256_file(source_c3_dir / "c3_manifest.json") != SOURCE_C3_MANIFEST_SHA256
        or sha256_file(source_c3_dir / "run_manifest.json") != SOURCE_C3_RUN_MANIFEST_SHA256
    ):
        raise RuntimeError("Frozen C3 source manifests changed before NUMSTAB finalize")
    for row in source["raw_inputs"].values():
        _verify_file_record(row)
    seen: list[str] = []
    workers: list[dict[str, str]] = []
    for shard in range(contract["num_shards"]):
        path, _ = _worker_paths(root, "evaluation", args.attempt_id, shard)
        report = _load_json(path)
        _validate_worker(report, "evaluation", args.attempt_id, shard, contract, contract_sha)
        if report.get("decision_barrier_sha256") != barrier_sha:
            raise RuntimeError(f"NUMSTAB evaluation worker points to another barrier: {path}")
        seen.extend(contract["shards"][str(shard)])
        workers.append({"path": path.relative_to(root).as_posix(), "sha256": sha256_file(path)})
    validate_shard_partition(contract, seen, EXPECTED_CASES, "NUMSTAB evaluation partition changed")

    normalization_rows: list[dict[str, Any]] = []
    factorial_cells: list[dict[str, Any]] = []
    factorial_edges: list[dict[str, Any]] = []
    scientific_rows: list[dict[str, Any]] = []
    per_arm: list[dict[str, Any]] = []
    evaluations_by_arm: dict[str, list[dict[str, Any]]] = {spec.arm_id: [] for spec in SCIENTIFIC_ARMS}
    decision_hashes: dict[str, str] = {}
    evaluation_hashes: dict[str, str] = {}
    sentinels: dict[str, Any] = {}
    reduction_oracle_faithful_cases = 0
    postclip_oracle_faithful_cases = 0
    for case_id in contract["case_ids"]:
        decision_path = _case_path(root, case_id, "decision")
        evaluation_path = _case_path(root, case_id, "evaluation")
        decision_payload = _load_json(decision_path)
        evaluation_payload = _load_json(evaluation_path)
        decision_arms = _validate_decision_case(
            decision_payload, decision_path, case_id, contract, contract_sha, verify_heavy=True
        )
        evaluation_arms = _validate_evaluation_case(
            evaluation_payload, evaluation_path, case_id, source, contract, contract_sha, barrier, barrier_sha
        )
        decision_hashes[case_id] = sha256_file(decision_path)
        evaluation_hashes[case_id] = sha256_file(evaluation_path)
        reduction_oracle_faithful_cases += int(decision_payload.get("reduction_oracle_faithful") is True)
        postclip_oracle_faithful_cases += int(decision_payload.get("postclip_oracle_faithful") is True)
        if case_id in SENTINEL_ALL_VECTORIZED_GAPS:
            sentinels[case_id] = decision_payload["e54_sentinel"]
        normalization_rows.extend({"case_id": case_id, **row} for row in decision_payload["normalization_rows"])
        factorial_cells.extend({"case_id": case_id, **row} for row in decision_payload["factorial_cell_rows"])
        factorial_edges.extend({"case_id": case_id, **row} for row in decision_payload["factorial_edge_rows"])
        scientific_rows.extend({"case_id": case_id, **row} for row in decision_payload["scientific_rows"])
        for decision_row, evaluation_row in zip(decision_arms, evaluation_arms, strict=True):
            spec = next(spec for spec in SCIENTIFIC_ARMS if spec.arm_id == decision_row["arm_id"])
            requested_geometry = decision_row["requested"]["geometry"]
            candidate_geometry = decision_row["candidate"]["geometry"]
            baseline_geometry = decision_payload["baseline_geometry"]
            metric_id = MATHEMATICAL_SDLOGJ_CROP2
            accepted = decision_row["action"] == "ACCEPT"
            returned_geometry = candidate_geometry if accepted else baseline_geometry
            comparator = source["reference_goldens"][case_id][spec.comparator_arm_id]
            geometry_reference = source["reference_goldens"][case_id]["c1_geometry_reference"]
            comparator_candidate = _finite_dice(
                comparator["candidate_dice"], f"{case_id}/{spec.comparator_arm_id}/candidate"
            )
            comparator_requested = _finite_dice(
                comparator["requested_dice"], f"{case_id}/{spec.comparator_arm_id}/requested"
            )
            comparator_returned = _finite_dice(
                comparator["returned_dice"], f"{case_id}/{spec.comparator_arm_id}/returned"
            )
            comparator_requested_geometry = _metric_value(
                {metric_id: comparator["requested_primary_geometry"]}, metric_id
            )
            comparator_candidate_geometry = _metric_value(
                {metric_id: comparator["candidate_primary_geometry"]}, metric_id
            )
            comparator_returned_geometry = _metric_value(
                {metric_id: comparator["returned_primary_geometry"]}, metric_id
            )
            requested_primary_geometry = _metric_value(requested_geometry, metric_id)
            candidate_primary_geometry = _metric_value(candidate_geometry, metric_id)
            returned_primary_geometry = _metric_value(returned_geometry, metric_id)
            record = {
                "case_id": case_id,
                "arm_index": decision_row["arm_index"],
                "arm_id": decision_row["arm_id"],
                "action": decision_row["action"],
                "reason": decision_row["primary_decision"]["reason"],
                "candidate_exact_certified": decision_row["candidate"]["exact_certified"],
                "returned_exact_certified": (
                    decision_row["candidate"]["exact_certified"]
                    if accepted
                    else contract["source_initial"][case_id]["exact"]["certified"]
                ),
                "support_retention": decision_row["candidate"]["supports"]["ncc7"]["retention"],
                "ncc7_improvement": decision_row["primary_decision"]["ncc_improvement"],
                "requested_residual_rms": decision_row["requested"]["stored_requested_residual_rms"],
                "postclip_realized_residual_rms": decision_row["candidate"]["postclip_realized_residual_rms"],
                "postclip_to_requested_rms_ratio": (
                    decision_row["candidate"]["postclip_realized_residual_rms"]
                    / decision_row["requested"]["stored_requested_residual_rms"]
                    if decision_row["requested"]["stored_requested_residual_rms"] > 0.0
                    else 0.0
                ),
                "comparator_arm_id": spec.comparator_arm_id,
                "comparator_requested_dice": comparator_requested,
                "comparator_candidate_dice": comparator_candidate,
                "comparator_returned_dice": comparator_returned,
                "requested_dice_delta_vs_comparator": evaluation_row["requested_diagnostic_dice"]
                - comparator_requested,
                "candidate_dice_delta_vs_comparator": evaluation_row["capacity_candidate_dice"] - comparator_candidate,
                "returned_dice_delta_vs_comparator": evaluation_row["primary_returned_dice"] - comparator_returned,
                "baseline_primary_geometry": _metric_value(baseline_geometry, metric_id),
                "baseline_primary_geometry_status": _metric_status(baseline_geometry, metric_id),
                "requested_primary_geometry": requested_primary_geometry,
                "requested_primary_geometry_status": _metric_status(requested_geometry, metric_id),
                "candidate_primary_geometry": candidate_primary_geometry,
                "candidate_primary_geometry_status": _metric_status(candidate_geometry, metric_id),
                "returned_primary_geometry": returned_primary_geometry,
                "returned_primary_geometry_status": _metric_status(returned_geometry, metric_id),
                "comparator_requested_primary_geometry": comparator_requested_geometry,
                "comparator_requested_primary_geometry_status": _metric_status(
                    {metric_id: comparator["requested_primary_geometry"]}, metric_id
                ),
                "comparator_candidate_primary_geometry": comparator_candidate_geometry,
                "comparator_candidate_primary_geometry_status": _metric_status(
                    {metric_id: comparator["candidate_primary_geometry"]}, metric_id
                ),
                "comparator_returned_primary_geometry": comparator_returned_geometry,
                "comparator_returned_primary_geometry_status": _metric_status(
                    {metric_id: comparator["returned_primary_geometry"]}, metric_id
                ),
                "requested_primary_geometry_delta_vs_comparator": (
                    requested_primary_geometry - comparator_requested_geometry
                    if requested_primary_geometry is not None and comparator_requested_geometry is not None
                    else None
                ),
                "candidate_primary_geometry_delta_vs_comparator": (
                    candidate_primary_geometry - comparator_candidate_geometry
                    if candidate_primary_geometry is not None and comparator_candidate_geometry is not None
                    else None
                ),
                "returned_primary_geometry_delta_vs_comparator": (
                    returned_primary_geometry - comparator_returned_geometry
                    if returned_primary_geometry is not None and comparator_returned_geometry is not None
                    else None
                ),
                "capacity_geometry_reference": _metric_value(
                    {metric_id: geometry_reference["candidate_primary_geometry"]}, metric_id
                ),
                "capacity_geometry_reference_status": _metric_status(
                    {metric_id: geometry_reference["candidate_primary_geometry"]}, metric_id
                ),
                "primary_geometry_reference": _metric_value(
                    {metric_id: geometry_reference["returned_primary_geometry"]}, metric_id
                ),
                "primary_geometry_reference_status": _metric_status(
                    {metric_id: geometry_reference["returned_primary_geometry"]}, metric_id
                ),
                **evaluation_row,
            }
            per_arm.append(record)
            evaluations_by_arm[record["arm_id"]].append(record)

    reduction_oracle_faithful_all_cases = reduction_oracle_faithful_cases == EXPECTED_CASES
    postclip_oracle_faithful_all_cases = postclip_oracle_faithful_cases == EXPECTED_CASES
    pre_summary_oracle_faithful_all_cases = reduction_oracle_faithful_all_cases and postclip_oracle_faithful_all_cases
    summaries: list[dict[str, Any]] = []
    hypotheses: dict[str, Any] = {}
    for spec in SCIENTIFIC_ARMS:
        rows = evaluations_by_arm[spec.arm_id]
        baseline = [row["baseline_dice"] for row in rows]
        requested_summary = _paired([row["requested_diagnostic_dice"] for row in rows], baseline)
        capacity = _paired([row["capacity_candidate_dice"] for row in rows], baseline)
        returned = _paired([row["primary_returned_dice"] for row in rows], baseline)
        requested_vs_comparator = _paired(
            [row["requested_diagnostic_dice"] for row in rows], [row["comparator_requested_dice"] for row in rows]
        )
        capacity_vs_comparator = _paired(
            [row["capacity_candidate_dice"] for row in rows], [row["comparator_candidate_dice"] for row in rows]
        )
        returned_vs_comparator = _paired(
            [row["primary_returned_dice"] for row in rows], [row["comparator_returned_dice"] for row in rows]
        )
        geometry_vs_comparator = {
            "requested": _geometry_contrast_summary(
                rows,
                "requested_primary_geometry",
                "comparator_requested_primary_geometry",
                expected_cases=EXPECTED_CASES,
                required=False,
                label=f"{spec.arm_id}/requested",
            ),
            "capacity": _geometry_contrast_summary(
                rows,
                "candidate_primary_geometry",
                "comparator_candidate_primary_geometry",
                expected_cases=EXPECTED_CASES,
                required=True,
                label=f"{spec.arm_id}/capacity",
            ),
            "primary_policy": _geometry_contrast_summary(
                rows,
                "returned_primary_geometry",
                "comparator_returned_primary_geometry",
                expected_cases=EXPECTED_CASES,
                required=True,
                label=f"{spec.arm_id}/primary_policy",
            ),
        }
        capacity_geometry_deltas = [
            float(row["candidate_primary_geometry"]) - float(row["capacity_geometry_reference"])
            for row in rows
            if row["candidate_primary_geometry"] is not None and row["capacity_geometry_reference"] is not None
        ]
        primary_geometry_deltas = [
            float(row["returned_primary_geometry"]) - float(row["primary_geometry_reference"])
            for row in rows
            if row["returned_primary_geometry"] is not None and row["primary_geometry_reference"] is not None
        ]
        all_capacity_geometry = len(capacity_geometry_deltas) == EXPECTED_CASES
        all_primary_geometry = len(primary_geometry_deltas) == EXPECTED_CASES
        capacity_geometry_mean_delta = float(np.mean(capacity_geometry_deltas)) if all_capacity_geometry else None
        primary_geometry_mean_delta = float(np.mean(primary_geometry_deltas)) if all_primary_geometry else None
        capacity_geometry_noninferior = bool(
            all_capacity_geometry
            and capacity_geometry_mean_delta is not None
            and capacity_geometry_mean_delta <= GEOMETRY_NONINFERIOR_TOLERANCE
        )
        primary_geometry_noninferior = bool(
            all_primary_geometry
            and primary_geometry_mean_delta is not None
            and primary_geometry_mean_delta <= GEOMETRY_NONINFERIOR_TOLERANCE
        )
        all_candidate_exact = all(row["candidate_exact_certified"] is True for row in rows)
        all_returned_exact = all(row["returned_exact_certified"] is True for row in rows)
        all_support = all(isinstance(row["support_retention"], (int, float)) for row in rows)
        eligibility = assess_arm_eligibility(
            selectable=spec.selectable,
            oracle_faithful_all_cases=pre_summary_oracle_faithful_all_cases,
            all_candidate_exact=all_candidate_exact,
            all_returned_exact=all_returned_exact,
            all_support_defined=all_support,
            primary_geometry_noninferior=primary_geometry_noninferior,
            capacity_vs_baseline=capacity.to_dict(),
            primary_vs_baseline=returned.to_dict(),
            capacity_vs_legacy=capacity_vs_comparator.to_dict(),
            primary_vs_legacy=returned_vs_comparator.to_dict(),
        )
        summary_row = {
            "arm_index": spec.arm_index,
            "arm_id": spec.arm_id,
            "role": spec.role,
            "oracle_arm": spec.role == "precision_oracle",
            "selectable": spec.selectable,
            "comparator_arm_id": spec.comparator_arm_id,
            "accepted_cases": sum(row["action"] == "ACCEPT" for row in rows),
            "all_candidate_exact_certified": all_candidate_exact,
            "all_returned_exact_certified": all_returned_exact,
            "all_capacity_geometry_defined": all_capacity_geometry,
            "capacity_geometry_mean_delta": capacity_geometry_mean_delta,
            "capacity_geometry_noninferior": capacity_geometry_noninferior,
            "all_primary_geometry_defined": all_primary_geometry,
            "primary_geometry_mean_delta": primary_geometry_mean_delta,
            "primary_geometry_noninferior": primary_geometry_noninferior,
            **eligibility,
            "requested_vs_baseline": requested_summary.to_dict(),
            "capacity_vs_baseline": capacity.to_dict(),
            "primary_policy_vs_baseline": returned.to_dict(),
            "requested_vs_comparator": requested_vs_comparator.to_dict(),
            "capacity_vs_comparator": capacity_vs_comparator.to_dict(),
            "primary_policy_vs_comparator": returned_vs_comparator.to_dict(),
            "requested_geometry_vs_comparator": geometry_vs_comparator["requested"],
            "capacity_geometry_vs_comparator": geometry_vs_comparator["capacity"],
            "primary_policy_geometry_vs_comparator": geometry_vs_comparator["primary_policy"],
        }
        summaries.append(summary_row)
        hypotheses[spec.arm_id] = summary_row

    summaries_by_id = {row["arm_id"]: row for row in summaries}
    summary_oracle_pairs: list[dict[str, Any]] = []
    status_keys = (
        "material_capacity",
        "capacity_geometry_noninferior",
        "practical_primary_policy_vs_baseline",
        "primary_policy_superior_to_legacy",
        "primary_geometry_noninferior",
    )
    for c32_spec in (spec for spec in SCIENTIFIC_ARMS if spec.selectable):
        oracle_spec = next(
            spec
            for spec in SCIENTIFIC_ARMS
            if spec.role == "precision_oracle" and spec.decoder_semantics == c32_spec.decoder_semantics
        )
        c32_summary = summaries_by_id[c32_spec.arm_id]
        oracle_summary = summaries_by_id[oracle_spec.arm_id]
        matching = {key: c32_summary[key] is oracle_summary[key] for key in status_keys}
        summary_oracle_pairs.append(
            {
                "decoder_semantics": c32_spec.decoder_semantics,
                "c32_arm_id": c32_spec.arm_id,
                "fp64_oracle_arm_id": oracle_spec.arm_id,
                "status_matches": matching,
                "faithful": all(matching.values()),
            }
        )
    summary_oracle_faithful = all(row["faithful"] for row in summary_oracle_pairs)
    oracle_faithful_all_cases = pre_summary_oracle_faithful_all_cases and summary_oracle_faithful
    for row in summaries:
        row["end_to_end_oracle_faithful"] = oracle_faithful_all_cases
        row["oracle_mismatch_blocks_selectable_policy"] = bool(row["selectable"] and not oracle_faithful_all_cases)
        if row["selectable"] and not oracle_faithful_all_cases:
            row["viable_primary_policy"] = False

    flat_summaries: list[dict[str, Any]] = []
    for row in summaries:
        flat = {key: value for key, value in row.items() if not isinstance(value, dict)}
        for prefix in (
            "requested_vs_baseline",
            "capacity_vs_baseline",
            "primary_policy_vs_baseline",
            "requested_vs_comparator",
            "capacity_vs_comparator",
            "primary_policy_vs_comparator",
            "requested_geometry_vs_comparator",
            "capacity_geometry_vs_comparator",
            "primary_policy_geometry_vs_comparator",
        ):
            flat.update({f"{prefix}_{key}": value for key, value in row[prefix].items()})
        flat_summaries.append(flat)
    next_branch = select_next_branch(
        summaries,
        oracle_faithful_all_cases=oracle_faithful_all_cases,
    )
    products = {
        "normalization.csv": normalization_rows,
        "factorial_cells.csv": factorial_cells,
        "factorial_edges.csv": factorial_edges,
        "scientific_reductions.csv": scientific_rows,
        "per_arm.csv": per_arm,
        "arm_summary.csv": flat_summaries,
    }
    preferred = [
        "case_id",
        "arm_index",
        "arm_id",
        "moment_reduction",
        "cell_id",
        "axis",
        "source_cell_id",
        "target_cell_id",
    ]
    for name, rows in products.items():
        atomic_write_text(root / name, rows_to_csv(_csv_fields(rows, preferred), rows))
    atomic_write_json(root / "hypotheses.json", hypotheses)
    atomic_write_json(root / "next_branch.json", next_branch)
    summary = {
        "schema": f"ctcf-search-gate-numstab-summary-{SCHEMA_VERSION}",
        "protocol_id": PROTOCOL_ID,
        "execution_integrity_status": "PASS",
        "scientific_status": next_branch["branch_id"],
        "n_cases": EXPECTED_CASES,
        "n_factorial_cells": len(FACTORIAL_SPECS),
        "n_factorial_edges": 12,
        "n_scientific_arms": len(SCIENTIFIC_ARMS),
        "one_raw_cost_volume_per_case": True,
        "direct_build_proposal_parity_verified": True,
        "e54_sentinel_gaps_verified": len(sentinels) == len(SENTINEL_ALL_VECTORIZED_GAPS)
        and all(row.get("pass") is True for row in sentinels.values()),
        "reduction_oracle_faithful_cases": reduction_oracle_faithful_cases,
        "reduction_oracle_faithful_all_cases": reduction_oracle_faithful_all_cases,
        "postclip_oracle_faithful_cases": postclip_oracle_faithful_cases,
        "postclip_oracle_faithful_all_cases": postclip_oracle_faithful_all_cases,
        "summary_oracle_pairs": summary_oracle_pairs,
        "summary_oracle_faithful": summary_oracle_faithful,
        "oracle_faithful_all_cases": oracle_faithful_all_cases,
        "labels_used_for_decision": False,
        "dice_evaluation_started_after_barrier": True,
        "test_split_accessed": False,
        "test_115_authorized": False,
        "material_capacity_arm_ids": [row["arm_id"] for row in summaries if row["material_capacity"]],
        "c32_material_capacity_arm_ids": [
            row["arm_id"] for row in summaries if row["selectable"] and row["material_capacity"]
        ],
        "fp64_oracle_material_capacity_arm_ids": [
            row["arm_id"] for row in summaries if row["oracle_arm"] and row["material_capacity"]
        ],
        "viable_primary_policy_arm_ids": [row["arm_id"] for row in summaries if row["viable_primary_policy"]],
        "next_branch": next_branch,
        "sentinel_cases": sentinels,
    }
    atomic_write_json(root / "summary.json", summary)
    prepare = _load_json(root / "prepare.json")
    manifest = {
        "schema": f"ctcf-search-gate-numstab-run-manifest-{SCHEMA_VERSION}",
        "protocol_id": PROTOCOL_ID,
        "run_id": root.name,
        "status": "COMPLETE",
        "started_at_utc": prepare["prepared_at_utc"],
        "completed_at_utc": utc_now(),
        "source_contract_sha256": source_sha,
        "decision_contract_sha256": contract_sha,
        "decision_barrier_sha256": barrier_sha,
        "policy_sha256": NUMERICAL_STABILITY_POLICY_SHA256,
        "finalize_attempt_id": args.attempt_id,
        "code": {"git_head": git("rev-parse", "HEAD"), "branch": git("branch", "--show-current"), "git_status": ""},
        "source_c3": source["source_c3"],
        "execution": {
            "num_shards": contract["num_shards"],
            "physical_gpus": contract["physical_gpus"],
            "seed": contract["seed"],
            "label_isolation": "retained image/initial -> immutable decision barrier -> val-58 labels",
        },
        "evaluation_workers": workers,
        "decision_case_sha256": decision_hashes,
        "evaluation_case_sha256": evaluation_hashes,
        "files": {
            "source_contract_sha256": sha256_file(root / SOURCE_CONTRACT_NAME),
            "decision_contract_sha256": sha256_file(root / DECISION_CONTRACT_NAME),
            "decision_barrier_sha256": sha256_file(root / BARRIER_NAME),
            **{
                name.replace(".", "_") + "_sha256": sha256_file(root / name)
                for name in [*products, "hypotheses.json", "next_branch.json", "summary.json"]
            },
        },
        "summary": summary,
        "storage": {
            "compact_product_excludes_fields": True,
            "source_c3_heavy_root_read_only": contract["source_c3_heavy_root"],
            "new_heavy_root": contract["heavy_root"],
            "new_heavy_contains_only_requested_and_candidate_fields": True,
            "automatic_deletion": False,
        },
    }
    atomic_write_json(root / "numerical_stability_manifest.json", manifest)
    print(json.dumps(summary, indent=2))
    return 0


def selfcheck_stage(args: argparse.Namespace) -> int:
    report = selfcheck()
    checks = dict(report["checks"])
    checks.update(
        {
            "runner_exposes_six_stages": set(build_parser()._subparsers._group_actions[0].choices)
            == {"prepare", "decision-worker", "barrier", "evaluation-worker", "finalize", "selfcheck"},
            "source_c3_manifest_is_frozen": SOURCE_C3_MANIFEST_SHA256
            == "d5c35ba4a27dab2d6d0dcd9f8017c39364aece31471286fe844a1e34b2337094",
            "source_c3_native_manifest_is_frozen": SOURCE_C3_RUN_MANIFEST_SHA256
            == "ee1958b6ec3f00eb3100538c6f46dbdc056869570ab6c147b661775bd96313a5",
            "decision_contract_is_label_free_by_design": True,
            "legacy_and_oracle_tolerances_are_positive": LEGACY_PARITY_ATOL > 0.0 and ORACLE_FAITHFUL_ATOL > 0.0,
        }
    )
    failed = [name for name, passed in checks.items() if not passed]
    payload = {
        "schema": f"ctcf-search-gate-numstab-runner-selfcheck-{SCHEMA_VERSION}",
        "protocol_id": PROTOCOL_ID,
        "status": "PASS" if not failed else "FAIL",
        "checks": checks,
        "failed": failed,
        "policy_sha256": NUMERICAL_STABILITY_POLICY_SHA256,
    }
    atomic_write_json(args.output, payload)
    if failed:
        raise RuntimeError(f"NUMSTAB runner selfcheck failed: {failed}")
    print(json.dumps(payload, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the frozen C3 MIND-cost numerical-stability gate on IXI val-58.")
    sub = parser.add_subparsers(dest="action", required=True)
    selfcheck_parser = sub.add_parser("selfcheck")
    selfcheck_parser.add_argument("--output", type=Path, required=True)
    prepare = sub.add_parser("prepare")
    prepare.add_argument("--run-root", type=Path, required=True)
    prepare.add_argument("--heavy-root", type=Path, required=True)
    prepare.add_argument("--source-c3-dir", type=Path, required=True)
    prepare.add_argument("--source-c3-heavy-root", type=Path, required=True)
    prepare.add_argument("--source-c3-manifest-sha256", default=SOURCE_C3_MANIFEST_SHA256)
    prepare.add_argument("--source-c3-run-manifest-sha256", default=SOURCE_C3_RUN_MANIFEST_SHA256)
    prepare.add_argument("--num-shards", type=int, required=True)
    prepare.add_argument("--physical-gpus", required=True)
    prepare.add_argument("--min-free-gib", type=int, default=50)
    decision = sub.add_parser("decision-worker")
    decision.add_argument("--run-root", type=Path, required=True)
    decision.add_argument("--decision-contract-sha256", required=True)
    decision.add_argument("--shard-index", type=int, required=True)
    decision.add_argument("--num-shards", type=int, required=True)
    decision.add_argument("--gpu", type=int, default=0)
    decision.add_argument("--physical-gpu", required=True)
    decision.add_argument("--attempt-id", required=True)
    barrier = sub.add_parser("barrier")
    barrier.add_argument("--run-root", type=Path, required=True)
    barrier.add_argument("--decision-contract-sha256", required=True)
    barrier.add_argument("--attempt-id", required=True)
    evaluation = sub.add_parser("evaluation-worker")
    evaluation.add_argument("--run-root", type=Path, required=True)
    evaluation.add_argument("--source-contract-sha256", required=True)
    evaluation.add_argument("--decision-contract-sha256", required=True)
    evaluation.add_argument("--barrier-sha256", required=True)
    evaluation.add_argument("--shard-index", type=int, required=True)
    evaluation.add_argument("--num-shards", type=int, required=True)
    evaluation.add_argument("--gpu", type=int, default=0)
    evaluation.add_argument("--physical-gpu", required=True)
    evaluation.add_argument("--attempt-id", required=True)
    finalize = sub.add_parser("finalize")
    finalize.add_argument("--run-root", type=Path, required=True)
    finalize.add_argument("--source-contract-sha256", required=True)
    finalize.add_argument("--decision-contract-sha256", required=True)
    finalize.add_argument("--barrier-sha256", required=True)
    finalize.add_argument("--attempt-id", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return {
        "prepare": prepare_stage,
        "decision-worker": decision_worker_stage,
        "barrier": barrier_stage,
        "evaluation-worker": evaluation_worker_stage,
        "finalize": finalize_stage,
        "selfcheck": selfcheck_stage,
    }[args.action](args)


if __name__ == "__main__":
    raise SystemExit(main())
