from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from tools.analysis.run_artifacts import atomic_write_json, sha256_file
from tools.analysis.search_gate_metrics import DETJ_DIAGNOSTICS, DIGITAL_DECOMPOSITION, METRIC_SPECS

PROTOCOL_ID = "CTCF-SEARCH-GATE-C4-V1"
SCHEMA_VERSION = "v1"

POLICY_SCHEMA = "ctcf-search-c4-policy-v1"
SOURCE_SCHEMA = "ctcf-search-c4-source-contract-v1"
DECISION_SCHEMA = "ctcf-search-c4-decision-contract-v1"
BARRIER_SCHEMA = "ctcf-search-c4-decision-barrier-v1"
DECISION_CASE_SCHEMA = "ctcf-search-c4-decision-case-v1"
EVALUATION_CASE_SCHEMA = "ctcf-search-c4-evaluation-case-v1"
WORKER_SCHEMA = "ctcf-search-c4-worker-v1"
PERSISTENCE_PROTOCOL_ID = "CTCF-SAVE-RELOAD-CERTIFY-V1"

SOURCE_C3_RUN_ID = "C3_DEVELOPMENT_20260822T212632Z_5d6c762a8a6f"
SOURCE_C3_MANIFEST_SHA256 = "d5c35ba4a27dab2d6d0dcd9f8017c39364aece31471286fe844a1e34b2337094"
SOURCE_C3_RUN_MANIFEST_SHA256 = "ee1958b6ec3f00eb3100538c6f46dbdc056869570ab6c147b661775bd96313a5"
SOURCE_C3_GIT_HEAD = "5d6c762a8a6f11607fe4312f30da774b3e05dec2"

EXPECTED_CASES = 58
COMMON_EVIDENCE_COLLAR = 7
SUPPORT_RETENTION_MIN = 0.99
PRIMARY_UTILITY_ID = "COMMON_NCC7"
PRIMARY_NCC_WINDOW = 7
PRIMARY_NCC_IMPROVEMENT_MIN = 1e-6
DICE_ARITHMETIC_ATOL = 1e-12
DICE_AGGREGATE_ATOL = 1e-8
SCIENTIFIC_REFERENCE_ARM_ID = "mind_d2_s1"
SCIENTIFIC_ARM_IDS = (
    "mind_d1_s1",
    "mind_d2_s1",
    "mind_d4_s1",
    "mind_f124_s1",
    "mind_d1_s2",
    "mind_d2_s2",
    "mind_d4_s2",
    "mind_f124_s2",
)
DIAGNOSTIC_ARM_IDS = (
    "legacy_mind_d2_s1_collar4",
    "mind_f222_s1",
    "intensity_s1",
    "intensity_s2",
)
ALL_ARM_IDS = (*SCIENTIFIC_ARM_IDS, *DIAGNOSTIC_ARM_IDS)

SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
GIT_SHA_RE = re.compile(r"^[0-9a-f]{40}$")
_DECISION_FORBIDDEN_KEY_PARTS = ("dice", "segmentation")
_DECISION_ALLOWED_LABEL_KEYS = {
    "labels_available_to_decision_workers",
    "labels_loaded_to_device",
    "decision_contract_contains_label_data",
}


@dataclass(frozen=True)
class ContractBundle:
    source: dict[str, Any]
    source_sha256: str
    decision: dict[str, Any]
    decision_sha256: str


def payload_sha256(payload: Any) -> str:
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def array_sha256(array: np.ndarray) -> str:
    value = np.ascontiguousarray(array, dtype=np.float32)
    return hashlib.sha256(value.tobytes(order="C")).hexdigest()


def canonical_offset_table() -> list[dict[str, Any]]:
    return [
        {
            "search_id": f"S{stride}",
            "offset_stride_voxels": stride,
            "offsets_zyx": [
                [dz, dy, dx]
                for dz in (-stride, 0, stride)
                for dy in (-stride, 0, stride)
                for dx in (-stride, 0, stride)
            ],
        }
        for stride in (1, 2)
    ]


def validate_offset_table(offset_table: Sequence[Mapping[str, Any]]) -> None:
    if list(offset_table) != canonical_offset_table():
        raise ValueError("C4 requires explicit ordered S1/S2 3x3x3 offset tables")


def validate_arm_specs(arm_specs: Sequence[Mapping[str, Any]]) -> None:
    if len(arm_specs) != len(ALL_ARM_IDS):
        raise ValueError("C4 arm specification has the wrong cardinality")
    for index, (row, expected_id) in enumerate(zip(arm_specs, ALL_ARM_IDS, strict=True)):
        required = {
            "arm_index",
            "arm_id",
            "role",
            "selectable",
            "diagnostic_only",
            "materialize_candidate",
            "post_barrier_evaluation",
        }
        if not required.issubset(row):
            raise ValueError(f"C4 arm {expected_id} lacks explicit identity metadata")
        if expected_id in SCIENTIFIC_ARM_IDS:
            expected_role = (
                "scientific_reference" if expected_id == SCIENTIFIC_REFERENCE_ARM_ID else "scientific_candidate"
            )
            expected_flags = (True, False, True, True)
        else:
            diagnostic_layout = {
                "legacy_mind_d2_s1_collar4": ("legacy_parity_diagnostic", False, False),
                "mind_f222_s1": ("fusion_idempotence_diagnostic", False, False),
                "intensity_s1": ("descriptor_specificity_diagnostic", True, True),
                "intensity_s2": ("descriptor_specificity_diagnostic", True, True),
            }
            expected_role, materialize, evaluate = diagnostic_layout[expected_id]
            expected_flags = (False, True, materialize, evaluate)
        observed_flags = (
            row["selectable"],
            row["diagnostic_only"],
            row["materialize_candidate"],
            row["post_barrier_evaluation"],
        )
        if (
            row["arm_index"] != index
            or row["arm_id"] != expected_id
            or row["role"] != expected_role
            or observed_flags != expected_flags
            or any(not isinstance(value, bool) for value in observed_flags)
        ):
            raise ValueError(f"C4 arm identity or role changed at index {index}")


def validate_support_contract(support: Mapping[str, Any]) -> None:
    required = {
        "support_id",
        "collar_width",
        "mask_rule",
        "utility_retention_min",
        "descriptor_retention_policy",
        "utility_id",
        "window",
        "improvement_min",
    }
    if not required.issubset(support):
        raise ValueError("C4 support contract lacks explicit collar/support metadata")
    if not isinstance(support["support_id"], str) or not support["support_id"]:
        raise ValueError("C4 support_id must be non-empty")
    collar = support["collar_width"]
    if collar != COMMON_EVIDENCE_COLLAR or isinstance(collar, bool):
        raise ValueError("C4 collar_width must remain frozen at 7")
    if not isinstance(support["mask_rule"], str) or not support["mask_rule"]:
        raise ValueError("C4 mask_rule must be non-empty")
    retention = support["utility_retention_min"]
    if isinstance(retention, bool) or retention != SUPPORT_RETENTION_MIN:
        raise ValueError("C4 utility_retention_min must remain frozen at 0.99")
    if support["descriptor_retention_policy"] != "diagnostic_only_nonempty":
        raise ValueError("C4 descriptor support retention must remain diagnostic-only")
    if support["utility_id"] != PRIMARY_UTILITY_ID:
        raise ValueError("C4 utility_id must remain COMMON_NCC7")
    window = support["window"]
    if window != PRIMARY_NCC_WINDOW or isinstance(window, bool):
        raise ValueError("C4 utility window must remain 7")
    improvement = support["improvement_min"]
    if isinstance(improvement, bool) or improvement != PRIMARY_NCC_IMPROVEMENT_MIN:
        raise ValueError("C4 improvement_min must remain 1e-6")


def _require_sha(value: Any, label: str) -> str:
    if not isinstance(value, str) or SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return value


def _finite_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise RuntimeError(f"{label} must be a finite number")
    return float(value)


def _unit_interval(value: Any, label: str) -> float:
    result = _finite_number(value, label)
    if not 0.0 <= result <= 1.0:
        raise RuntimeError(f"{label} must be in [0,1]")
    return result


def _validate_geometry_bundle(bundle: Any, label: str) -> None:
    if not isinstance(bundle, Mapping) or set(bundle) != set(METRIC_SPECS):
        raise RuntimeError(f"{label} has an incomplete geometry metric bundle")
    for metric_id, row in bundle.items():
        if not isinstance(row, Mapping) or row.get("status") != "OK":
            raise RuntimeError(f"{label} has undefined geometry: {metric_id}")
        if metric_id == DETJ_DIAGNOSTICS:
            components = row.get("components") or {}
            detj_min = _finite_number(components.get("detj_min"), f"{label} detJ minimum")
            invalid = _finite_number(components.get("invalid_count"), f"{label} invalid detJ count")
            if row.get("value") is not None or detj_min <= 0.0 or invalid != 0.0:
                raise RuntimeError(f"{label} has invalid component-only detJ diagnostics")
        else:
            _finite_number(row.get("value"), f"{label} {metric_id}")
    components = bundle[DIGITAL_DECOMPOSITION].get("components") or {}
    corner = _finite_number(components.get("corner_union_violation_fraction"), f"{label} digital corner")
    if corner != 0.0:
        raise RuntimeError(f"{label} contradicts its exact no-fold certificate")


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"{path}: expected a JSON object")
    return payload


def _canonical_json_text(payload: Mapping[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def _write_immutable_json(path: Path, payload: dict[str, Any]) -> str:
    expected = _canonical_json_text(payload)
    if path.exists():
        if path.read_text(encoding="utf-8") != expected:
            raise RuntimeError(f"Immutable C4 JSON changed or is non-canonical: {path}")
    else:
        atomic_write_json(path, payload)
    return sha256_file(path)


def _assert_unique_cases(case_ids: Any, label: str) -> list[str]:
    if (
        not isinstance(case_ids, list)
        or len(case_ids) != EXPECTED_CASES
        or len(set(case_ids)) != EXPECTED_CASES
        or any(not isinstance(case_id, str) or not case_id for case_id in case_ids)
        or "subject_115" in case_ids
    ):
        raise RuntimeError(f"{label} must contain exactly {EXPECTED_CASES} unique non-test cases")
    return case_ids


def _verify_file_record(record: Mapping[str, Any], *, root: Path | None = None) -> Path:
    path = Path(str(record.get("path", ""))).resolve()
    if root is not None and path != root and root not in path.parents:
        raise RuntimeError(f"Observed file escapes its frozen root: {path}")
    if not path.is_file():
        raise RuntimeError(f"Observed file is missing: {path}")
    try:
        expected_bytes = int(record["bytes"])
    except (KeyError, TypeError, ValueError) as error:
        raise RuntimeError(f"Observed file has invalid byte metadata: {path}") from error
    if path.stat().st_size != expected_bytes or sha256_file(path) != _require_sha(record.get("sha256"), str(path)):
        raise RuntimeError(f"Observed file bytes changed: {path}")
    return path


def _resolve_heavy_field(root: Path, record: Mapping[str, Any], *, verify_array: bool) -> Path:
    relative = Path(str(record.get("relative_path", "")))
    if relative.is_absolute() or ".." in relative.parts:
        raise RuntimeError("Heavy field path must be relative and traversal-free")
    path = (root / relative).resolve()
    if root != path and root not in path.parents:
        raise RuntimeError("Heavy field escapes its declared root")
    if not path.is_file() or sha256_file(path) != _require_sha(record.get("npz_sha256"), str(path)):
        raise RuntimeError(f"Heavy field is missing or changed: {path}")
    if verify_array:
        with np.load(path, allow_pickle=False) as archive:
            if set(archive.files) != {"flow"}:
                raise RuntimeError(f"{path}: expected the sole key 'flow'")
            array = archive["flow"]
        if (
            array.dtype != np.float32
            or array.ndim != 5
            or array.shape[0:2] != (1, 3)
            or not np.isfinite(array).all()
            or array_sha256(array) != _require_sha(record.get("array_sha256"), str(path))
        ):
            raise RuntimeError(f"Heavy field array changed or is invalid: {path}")
    return path


def _assert_label_free(payload: Mapping[str, Any], label: str) -> None:
    stack: list[Any] = [payload]
    while stack:
        value = stack.pop()
        if isinstance(value, Mapping):
            for key, child in value.items():
                lowered = str(key).lower()
                if lowered == "policy":
                    continue
                if any(part in lowered for part in _DECISION_FORBIDDEN_KEY_PARTS):
                    raise RuntimeError(f"{label} contains forbidden decision key: {key}")
                if "label" in lowered and lowered not in _DECISION_ALLOWED_LABEL_KEYS:
                    raise RuntimeError(f"{label} contains forbidden decision key: {key}")
                if lowered in _DECISION_ALLOWED_LABEL_KEYS and child is not False:
                    raise RuntimeError(f"{label} isolation flag {key} must be exactly false")
                stack.append(child)
        elif isinstance(value, (list, tuple)):
            stack.extend(value)
        elif isinstance(value, str) and ("segmentation" in value.lower() or value.lower().endswith(".pkl")):
            raise RuntimeError(f"{label} contains a raw-container or segmentation reference")


def _validate_sharding(contract: Mapping[str, Any]) -> None:
    case_ids = _assert_unique_cases(contract.get("case_ids"), "C4 contract")
    num_shards = contract.get("num_shards")
    physical = contract.get("physical_gpus")
    if isinstance(num_shards, bool) or not isinstance(num_shards, int) or num_shards < 1:
        raise RuntimeError("C4 num_shards must be positive")
    if (
        not isinstance(physical, list)
        or len(physical) != num_shards
        or len(set(physical)) != num_shards
        or any(not isinstance(value, str) or not value.isdigit() for value in physical)
    ):
        raise RuntimeError("C4 requires one unique physical GPU per shard")
    expected_shards = {
        str(index): [case_id for position, case_id in enumerate(case_ids) if position % num_shards == index]
        for index in range(num_shards)
    }
    expected_map = {str(index): value for index, value in enumerate(physical)}
    if contract.get("shards") != expected_shards or contract.get("shard_to_physical_gpu") != expected_map:
        raise RuntimeError("C4 shard partition, order, or physical-GPU mapping changed")


def authenticate_frozen_c3(compact_dir: Path, heavy_root: Path) -> dict[str, Any]:
    compact = compact_dir.resolve()
    heavy = heavy_root.resolve()
    manifest_path = compact / "c3_manifest.json"
    run_manifest_path = compact / "run_manifest.json"
    if sha256_file(manifest_path) != SOURCE_C3_MANIFEST_SHA256:
        raise RuntimeError("C4 requires the exact frozen successful C3 manifest")
    if sha256_file(run_manifest_path) != SOURCE_C3_RUN_MANIFEST_SHA256:
        raise RuntimeError("C4 requires the exact frozen native C3 run manifest")
    manifest = _load_json(manifest_path)
    run_manifest = _load_json(run_manifest_path)
    summary = manifest.get("summary") or {}
    code = manifest.get("code") or {}
    if (
        manifest.get("schema") != "ctcf-search-c3a-run-manifest-v1"
        or manifest.get("protocol_id") != "CTCF-SEARCH-GATE-C3A-V1"
        or manifest.get("run_id") != SOURCE_C3_RUN_ID
        or manifest.get("status") != "COMPLETE"
        or code.get("git_head") != SOURCE_C3_GIT_HEAD
        or code.get("git_status") != ""
        or summary.get("execution_integrity_status") != "PASS"
        or summary.get("n_cases") != EXPECTED_CASES
        or summary.get("test_115_authorized") is not False
        or summary.get("test_split_accessed") is not False
        or summary.get("labels_used_for_decision") is not False
        or Path(str((manifest.get("storage") or {}).get("heavy_root", ""))).resolve() != heavy
        or run_manifest.get("schema") != "ctcf-native-manifest-v1"
        or run_manifest.get("run_id") != SOURCE_C3_RUN_ID
        or run_manifest.get("status") != "COMPLETE"
        or (run_manifest.get("code") or {}).get("git_head") != SOURCE_C3_GIT_HEAD
        or (run_manifest.get("code") or {}).get("tracked_tree_clean_at_start") is not True
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
        path = compact / name
        if not path.is_file() or sha256_file(path) != _require_sha(files.get(key), f"C3 {name}"):
            raise RuntimeError(f"Frozen C3 manifest does not authenticate {path}")
    datasets_tsv = compact / "datasets.tsv"
    if not datasets_tsv.is_file() or sha256_file(datasets_tsv) != _require_sha(
        (run_manifest.get("files") or {}).get("datasets_sha256"), "C3 datasets.tsv"
    ):
        raise RuntimeError("Frozen C3 native manifest does not authenticate datasets.tsv")

    source = _load_json(compact / "source_contract.json")
    decision = _load_json(compact / "decision_contract.json")
    barrier = _load_json(compact / "decision_barrier.json")
    case_ids = _assert_unique_cases(source.get("case_ids"), "Frozen C3")
    if (
        source.get("schema") != "ctcf-search-c3a-source-contract-v1"
        or decision.get("schema") != "ctcf-search-c3a-decision-contract-v1"
        or barrier.get("schema") != "ctcf-search-c3a-decision-barrier-v1"
        or source.get("git_head") != SOURCE_C3_GIT_HEAD
        or decision.get("git_head") != SOURCE_C3_GIT_HEAD
        or decision.get("source_contract_sha256") != files["source_contract_sha256"]
        or barrier.get("decision_contract_sha256") != files["decision_contract_sha256"]
        or Path(str(source.get("heavy_root", ""))).resolve() != heavy
        or Path(str(decision.get("heavy_root", ""))).resolve() != heavy
        or decision.get("case_ids") != case_ids
        or source.get("test_115_authorized") is not False
        or decision.get("test_115_authorized") is not False
        or source.get("ixi_test_split_accessed") is not False
        or decision.get("ixi_test_split_accessed") is not False
        or barrier.get("test_split_accessed") is not False
        or barrier.get("decision_workers_received_label_inputs") is not False
    ):
        raise RuntimeError("Frozen C3 source contracts are inconsistent")
    decision_hashes = manifest.get("decision_case_sha256") or {}
    evaluation_hashes = manifest.get("evaluation_case_sha256") or {}
    if set(decision_hashes) != set(case_ids) or set(evaluation_hashes) != set(case_ids):
        raise RuntimeError("Frozen C3 manifest does not cover exactly val-58")
    if barrier.get("decision_case_sha256") != decision_hashes:
        raise RuntimeError("Frozen C3 barrier and final manifest disagree")

    raw_inputs = source.get("raw_inputs") or {}
    image_inputs = decision.get("image_inputs") or {}
    if set(raw_inputs) != {"atlas", *case_ids} or set(image_inputs) != set(raw_inputs):
        raise RuntimeError("Frozen C3 raw/image input inventories are incomplete")
    for case_id, record in raw_inputs.items():
        expected_split = "atlas" if case_id == "atlas" else "val"
        if record.get("dataset") != "IXI" or record.get("split") != expected_split or record.get("case_id") != case_id:
            raise RuntimeError(f"Frozen C3 source has an invalid split record: {case_id}")
        _verify_file_record(record)
    for record in image_inputs.values():
        _verify_file_record(record, root=heavy)

    source_initial: dict[str, Any] = {}
    source_historical: dict[str, Any] = {}
    evaluation_baseline_dice: dict[str, float] = {}
    for case_id in case_ids:
        decision_path = compact / "cases" / case_id / "decision_complete.json"
        evaluation_path = compact / "cases" / case_id / "evaluation_complete.json"
        if sha256_file(decision_path) != _require_sha(
            decision_hashes[case_id], f"C3 {case_id} decision"
        ) or sha256_file(evaluation_path) != _require_sha(evaluation_hashes[case_id], f"C3 {case_id} evaluation"):
            raise RuntimeError(f"Frozen C3 case marker changed: {case_id}")
        decision_case = _load_json(decision_path)
        evaluation_case = _load_json(evaluation_path)
        initial = decision_case.get("initial") or {}
        field = initial.get("field") or {}
        exact = (initial.get("report") or {}).get("psi_exact") or {}
        raw_conf_rows = [row for row in decision_case.get("arms") or [] if row.get("arm_id") == "raw_conf_post1"]
        baseline_values = {
            _finite_number(row.get("baseline_dice"), f"C3 {case_id} baseline Dice")
            for row in evaluation_case.get("arms") or []
        }
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
            or exact.get("status") != "CERTIFIED"
            or exact.get("certified") is not True
            or exact.get("sha256") != field.get("array_sha256")
            or len(raw_conf_rows) != 1
            or len(baseline_values) != 1
        ):
            raise RuntimeError(f"Frozen C3 case is not a valid label-isolated source: {case_id}")
        _resolve_heavy_field(heavy, field, verify_array=False)
        raw_conf_field = (raw_conf_rows[0].get("requested_state") or {}).get("field") or {}
        _resolve_heavy_field(heavy, raw_conf_field, verify_array=False)
        source_initial[case_id] = {
            "field": field,
            "exact": exact,
            "source_decision_case_sha256": decision_hashes[case_id],
        }
        source_historical[case_id] = {
            "raw_conf_requested_field": raw_conf_field,
            "source_decision_case_sha256": decision_hashes[case_id],
        }
        evaluation_baseline_dice[case_id] = baseline_values.pop()

    return {
        "source_c3": {
            "compact_directory": str(compact),
            "heavy_root": str(heavy),
            "run_id": SOURCE_C3_RUN_ID,
            "git_head": SOURCE_C3_GIT_HEAD,
            "manifest_sha256": SOURCE_C3_MANIFEST_SHA256,
            "run_manifest_sha256": SOURCE_C3_RUN_MANIFEST_SHA256,
            "source_contract_sha256": files["source_contract_sha256"],
            "decision_contract_sha256": files["decision_contract_sha256"],
            "decision_barrier_sha256": files["decision_barrier_sha256"],
        },
        "raw_inputs": raw_inputs,
        "image_inputs": image_inputs,
        "source_initial": source_initial,
        "source_historical": source_historical,
        "evaluation_baseline_dice": evaluation_baseline_dice,
        "case_ids": case_ids,
        "seed": source["seed"],
        "runtime_signature": source["runtime_signature"],
    }


def _build_sharding(case_ids: list[str], physical_gpus: Sequence[str]) -> dict[str, Any]:
    physical = list(physical_gpus)
    if not physical or len(set(physical)) != len(physical) or any(not value.isdigit() for value in physical):
        raise ValueError("C4 physical GPUs must be unique non-negative integer strings")
    shards = {
        str(index): [case_id for position, case_id in enumerate(case_ids) if position % len(physical) == index]
        for index in range(len(physical))
    }
    return {
        "num_shards": len(physical),
        "physical_gpus": physical,
        "shard_to_physical_gpu": {str(index): value for index, value in enumerate(physical)},
        "shards": shards,
    }


def build_source_contract(
    snapshot: Mapping[str, Any],
    *,
    git_head: str,
    runtime_signature: Mapping[str, Any],
    target_heavy_root: Path,
    physical_gpus: Sequence[str],
) -> dict[str, Any]:
    if GIT_SHA_RE.fullmatch(git_head) is None:
        raise ValueError("C4 git_head must be a full lowercase Git SHA")
    case_ids = _assert_unique_cases(snapshot.get("case_ids"), "C4 source snapshot")
    payload = {
        "schema": SOURCE_SCHEMA,
        "protocol_id": PROTOCOL_ID,
        "git_head": git_head,
        "runtime_signature": dict(runtime_signature),
        "source_c3": snapshot["source_c3"],
        "raw_inputs": snapshot["raw_inputs"],
        "image_inputs": snapshot["image_inputs"],
        "source_initial": snapshot["source_initial"],
        "source_historical": snapshot["source_historical"],
        "evaluation_baseline_dice": snapshot["evaluation_baseline_dice"],
        "case_ids": case_ids,
        "seed": snapshot["seed"],
        "ixi_test_split_accessed": False,
        "test_115_authorized": False,
        "heavy_root": str(target_heavy_root.resolve()),
        **_build_sharding(case_ids, physical_gpus),
    }
    validate_source_contract(payload)
    return payload


def validate_source_contract(payload: Mapping[str, Any]) -> None:
    if (
        payload.get("schema") != SOURCE_SCHEMA
        or payload.get("protocol_id") != PROTOCOL_ID
        or GIT_SHA_RE.fullmatch(str(payload.get("git_head", ""))) is None
        or payload.get("ixi_test_split_accessed") is not False
        or payload.get("test_115_authorized") is not False
        or (payload.get("source_c3") or {}).get("run_id") != SOURCE_C3_RUN_ID
        or (payload.get("source_c3") or {}).get("git_head") != SOURCE_C3_GIT_HEAD
        or (payload.get("source_c3") or {}).get("manifest_sha256") != SOURCE_C3_MANIFEST_SHA256
        or (payload.get("source_c3") or {}).get("run_manifest_sha256") != SOURCE_C3_RUN_MANIFEST_SHA256
    ):
        raise RuntimeError("Invalid or altered C4 source contract")
    case_ids = _assert_unique_cases(payload.get("case_ids"), "C4 source contract")
    if set(payload.get("raw_inputs") or {}) != {"atlas", *case_ids}:
        raise RuntimeError("C4 source contract has the wrong raw-input inventory")
    if set(payload.get("image_inputs") or {}) != {"atlas", *case_ids}:
        raise RuntimeError("C4 source contract has the wrong image inventory")
    if set(payload.get("source_initial") or {}) != set(case_ids):
        raise RuntimeError("C4 source contract has the wrong initial-field inventory")
    if set(payload.get("source_historical") or {}) != set(case_ids):
        raise RuntimeError("C4 source contract has the wrong historical-field inventory")
    baselines = payload.get("evaluation_baseline_dice") or {}
    if set(baselines) != set(case_ids):
        raise RuntimeError("C4 source contract has the wrong evaluation-baseline inventory")
    for case_id, value in baselines.items():
        result = _finite_number(value, f"C4 {case_id} evaluation baseline Dice")
        if not 0.0 <= result <= 1.0:
            raise RuntimeError(f"C4 {case_id} evaluation baseline Dice must be in [0,1]")
    _validate_sharding(payload)


def build_decision_contract(
    source: Mapping[str, Any],
    source_sha256: str,
    *,
    policy: Mapping[str, Any],
    expected_policy_sha256: str,
    arm_specs: Sequence[Mapping[str, Any]],
    expected_arm_specs_sha256: str,
    offset_table: Sequence[Mapping[str, Any]],
    expected_offset_table_sha256: str,
    support_contract: Mapping[str, Any],
    expected_support_contract_sha256: str,
) -> dict[str, Any]:
    validate_source_contract(source)
    _require_sha(source_sha256, "C4 source contract")
    if policy.get("schema_version") != SCHEMA_VERSION or policy.get("protocol_id") != PROTOCOL_ID:
        raise ValueError("C4 policy has an unsupported schema or protocol")
    validate_arm_specs(arm_specs)
    validate_offset_table(offset_table)
    validate_support_contract(support_contract)
    checks = (
        (policy, expected_policy_sha256, "policy"),
        (list(arm_specs), expected_arm_specs_sha256, "arm specification"),
        (list(offset_table), expected_offset_table_sha256, "offset table"),
        (dict(support_contract), expected_support_contract_sha256, "support contract"),
    )
    for value, expected, label in checks:
        if payload_sha256(value) != _require_sha(expected, f"C4 {label}"):
            raise RuntimeError(f"C4 {label} SHA-256 does not match its payload")
    payload = {
        "schema": DECISION_SCHEMA,
        "protocol_id": PROTOCOL_ID,
        "git_head": source["git_head"],
        "runtime_signature": source["runtime_signature"],
        "source_contract_sha256": source_sha256,
        "source_c3_manifest_sha256": SOURCE_C3_MANIFEST_SHA256,
        "source_c3_run_manifest_sha256": SOURCE_C3_RUN_MANIFEST_SHA256,
        "source_c3_heavy_root": source["source_c3"]["heavy_root"],
        "image_inputs": source["image_inputs"],
        "source_initial": source["source_initial"],
        "source_historical": source["source_historical"],
        "case_ids": source["case_ids"],
        "seed": source["seed"],
        "policy_schema": POLICY_SCHEMA,
        "policy": dict(policy),
        "policy_sha256": expected_policy_sha256,
        "arm_specs": list(arm_specs),
        "arm_specs_sha256": expected_arm_specs_sha256,
        "offset_table": list(offset_table),
        "offset_table_sha256": expected_offset_table_sha256,
        "support_contract": dict(support_contract),
        "support_contract_sha256": expected_support_contract_sha256,
        "decision_contract_contains_label_data": False,
        "decision_worker_uses_raw_containers": False,
        "labels_available_to_decision_workers": False,
        "ixi_test_split_accessed": False,
        "test_115_authorized": False,
        "heavy_root": source["heavy_root"],
        "num_shards": source["num_shards"],
        "physical_gpus": source["physical_gpus"],
        "shard_to_physical_gpu": source["shard_to_physical_gpu"],
        "shards": source["shards"],
    }
    validate_decision_contract(
        payload,
        source=source,
        expected_source_sha256=source_sha256,
        expected_policy_sha256=expected_policy_sha256,
        expected_arm_specs_sha256=expected_arm_specs_sha256,
        expected_offset_table_sha256=expected_offset_table_sha256,
        expected_support_contract_sha256=expected_support_contract_sha256,
    )
    return payload


def validate_decision_contract(
    payload: Mapping[str, Any],
    *,
    source: Mapping[str, Any],
    expected_source_sha256: str,
    expected_policy_sha256: str,
    expected_arm_specs_sha256: str,
    expected_offset_table_sha256: str,
    expected_support_contract_sha256: str,
) -> None:
    validate_source_contract(source)
    _validate_decision_owned_invariants(
        payload,
        expected_source_sha256=expected_source_sha256,
        expected_policy_sha256=expected_policy_sha256,
        expected_arm_specs_sha256=expected_arm_specs_sha256,
        expected_offset_table_sha256=expected_offset_table_sha256,
        expected_support_contract_sha256=expected_support_contract_sha256,
    )
    if (
        payload.get("case_ids") != source.get("case_ids")
        or payload.get("image_inputs") != source.get("image_inputs")
        or payload.get("source_initial") != source.get("source_initial")
        or payload.get("source_historical") != source.get("source_historical")
        or payload.get("heavy_root") != source.get("heavy_root")
        or any(
            payload.get(key) != source.get(key)
            for key in ("num_shards", "physical_gpus", "shards", "shard_to_physical_gpu")
        )
    ):
        raise RuntimeError("C4 decision contract differs from its source projection")


def _validate_decision_owned_invariants(
    payload: Mapping[str, Any],
    *,
    expected_source_sha256: str,
    expected_policy_sha256: str,
    expected_arm_specs_sha256: str,
    expected_offset_table_sha256: str,
    expected_support_contract_sha256: str,
) -> None:
    if (
        payload.get("schema") != DECISION_SCHEMA
        or payload.get("protocol_id") != PROTOCOL_ID
        or GIT_SHA_RE.fullmatch(str(payload.get("git_head", ""))) is None
        or payload.get("source_contract_sha256") != _require_sha(expected_source_sha256, "C4 source")
        or payload.get("source_c3_manifest_sha256") != SOURCE_C3_MANIFEST_SHA256
        or payload.get("source_c3_run_manifest_sha256") != SOURCE_C3_RUN_MANIFEST_SHA256
        or payload.get("policy_schema") != POLICY_SCHEMA
        or payload.get("decision_contract_contains_label_data") is not False
        or payload.get("decision_worker_uses_raw_containers") is not False
        or payload.get("labels_available_to_decision_workers") is not False
        or payload.get("ixi_test_split_accessed") is not False
        or payload.get("test_115_authorized") is not False
        or "evaluation_baseline_dice" in payload
        or not isinstance(payload.get("source_c3_heavy_root"), str)
        or not payload.get("source_c3_heavy_root")
        or not isinstance(payload.get("heavy_root"), str)
        or not payload.get("heavy_root")
    ):
        raise RuntimeError("Invalid or altered C4 decision contract")
    case_ids = _assert_unique_cases(payload.get("case_ids"), "C4 decision contract")
    image_inputs = payload.get("image_inputs") or {}
    source_initial = payload.get("source_initial") or {}
    source_historical = payload.get("source_historical") or {}
    if set(image_inputs) != {"atlas", *case_ids}:
        raise RuntimeError("C4 decision contract has the wrong image inventory")
    if set(source_initial) != set(case_ids) or set(source_historical) != set(case_ids):
        raise RuntimeError("C4 decision contract has the wrong source-field inventory")
    for identity, record in image_inputs.items():
        if not isinstance(record, Mapping):
            raise RuntimeError(f"C4 decision image record is invalid: {identity}")
        _require_sha(record.get("array_sha256"), f"C4 {identity} image array")
    for case_id in case_ids:
        initial = source_initial[case_id]
        historical = source_historical[case_id]
        if not isinstance(initial, Mapping) or not isinstance(historical, Mapping):
            raise RuntimeError(f"C4 decision source-field record is invalid: {case_id}")
        initial_field = initial.get("field") or {}
        historical_field = historical.get("raw_conf_requested_field") or {}
        initial_sha = _require_sha(initial_field.get("array_sha256"), f"C4 {case_id} initial field")
        _require_sha(historical_field.get("array_sha256"), f"C4 {case_id} historical field")
        source_case_sha = _require_sha(initial.get("source_decision_case_sha256"), f"C4 {case_id} source case")
        if historical.get("source_decision_case_sha256") != source_case_sha:
            raise RuntimeError(f"C4 historical field is bound to a foreign source case: {case_id}")
        exact = initial.get("exact") or {}
        if (
            exact.get("certified") is not True
            or exact.get("status") != "CERTIFIED"
            or exact.get("sha256") != initial_sha
        ):
            raise RuntimeError(f"C4 initial field lacks its exact certificate: {case_id}")
    checks = (
        (payload.get("policy"), payload.get("policy_sha256"), expected_policy_sha256, "policy"),
        (payload.get("arm_specs"), payload.get("arm_specs_sha256"), expected_arm_specs_sha256, "arm specification"),
        (payload.get("offset_table"), payload.get("offset_table_sha256"), expected_offset_table_sha256, "offset table"),
        (
            payload.get("support_contract"),
            payload.get("support_contract_sha256"),
            expected_support_contract_sha256,
            "support contract",
        ),
    )
    for value, observed, expected, label in checks:
        if observed != _require_sha(expected, f"C4 {label}") or payload_sha256(value) != observed:
            raise RuntimeError(f"C4 {label} payload or SHA-256 changed")
    if (payload.get("policy") or {}).get("schema_version") != SCHEMA_VERSION or (payload.get("policy") or {}).get(
        "protocol_id"
    ) != PROTOCOL_ID:
        raise RuntimeError("C4 policy schema changed")
    validate_arm_specs(payload.get("arm_specs") or [])
    validate_offset_table(payload.get("offset_table") or [])
    validate_support_contract(payload.get("support_contract") or {})
    _validate_sharding(payload)
    _assert_label_free(payload, "C4 decision contract")


def prepare_contracts(
    *,
    run_root: Path,
    source_c3_dir: Path,
    source_c3_heavy_root: Path,
    target_heavy_root: Path,
    git_head: str,
    runtime_signature: Mapping[str, Any],
    physical_gpus: Sequence[str],
    policy: Mapping[str, Any],
    expected_policy_sha256: str,
    arm_specs: Sequence[Mapping[str, Any]],
    expected_arm_specs_sha256: str,
    offset_table: Sequence[Mapping[str, Any]],
    expected_offset_table_sha256: str,
    support_contract: Mapping[str, Any],
    expected_support_contract_sha256: str,
) -> ContractBundle:
    snapshot = authenticate_frozen_c3(source_c3_dir, source_c3_heavy_root)
    source = build_source_contract(
        snapshot,
        git_head=git_head,
        runtime_signature=runtime_signature,
        target_heavy_root=target_heavy_root,
        physical_gpus=physical_gpus,
    )
    root = run_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    source_sha = _write_immutable_json(root / "source_contract.json", source)
    decision = build_decision_contract(
        source,
        source_sha,
        policy=policy,
        expected_policy_sha256=expected_policy_sha256,
        arm_specs=arm_specs,
        expected_arm_specs_sha256=expected_arm_specs_sha256,
        offset_table=offset_table,
        expected_offset_table_sha256=expected_offset_table_sha256,
        support_contract=support_contract,
        expected_support_contract_sha256=expected_support_contract_sha256,
    )
    decision_sha = _write_immutable_json(root / "decision_contract.json", decision)
    return ContractBundle(source, source_sha, decision, decision_sha)


def load_source_contract(run_root: Path, expected_sha256: str) -> tuple[dict[str, Any], str]:
    path = run_root.resolve() / "source_contract.json"
    actual = sha256_file(path)
    if actual != _require_sha(expected_sha256, "C4 source contract"):
        raise RuntimeError("C4 source contract hash mismatch")
    payload = _load_json(path)
    validate_source_contract(payload)
    return payload, actual


def load_decision_contract(
    run_root: Path,
    expected_sha256: str,
    *,
    source: Mapping[str, Any],
    expected_source_sha256: str,
    expected_policy_sha256: str,
    expected_arm_specs_sha256: str,
    expected_offset_table_sha256: str,
    expected_support_contract_sha256: str,
) -> tuple[dict[str, Any], str]:
    path = run_root.resolve() / "decision_contract.json"
    actual = sha256_file(path)
    if actual != _require_sha(expected_sha256, "C4 decision contract"):
        raise RuntimeError("C4 decision contract hash mismatch")
    payload = _load_json(path)
    validate_decision_contract(
        payload,
        source=source,
        expected_source_sha256=expected_source_sha256,
        expected_policy_sha256=expected_policy_sha256,
        expected_arm_specs_sha256=expected_arm_specs_sha256,
        expected_offset_table_sha256=expected_offset_table_sha256,
        expected_support_contract_sha256=expected_support_contract_sha256,
    )
    return payload, actual


def load_decision_contract_isolated(
    run_root: Path,
    expected_decision_sha256: str,
    *,
    expected_source_sha256: str,
    expected_policy_sha256: str,
    expected_arm_specs_sha256: str,
    expected_offset_table_sha256: str,
    expected_support_contract_sha256: str,
) -> tuple[dict[str, Any], str]:
    root = run_root.resolve()
    decision_path = root / "decision_contract.json"
    actual_decision_sha = sha256_file(decision_path)
    if actual_decision_sha != _require_sha(expected_decision_sha256, "C4 decision contract"):
        raise RuntimeError("C4 decision contract hash mismatch")
    actual_source_sha = sha256_file(root / "source_contract.json")
    if actual_source_sha != _require_sha(expected_source_sha256, "C4 source contract"):
        raise RuntimeError("C4 source contract byte hash mismatch")
    payload = _load_json(decision_path)
    _validate_decision_owned_invariants(
        payload,
        expected_source_sha256=expected_source_sha256,
        expected_policy_sha256=expected_policy_sha256,
        expected_arm_specs_sha256=expected_arm_specs_sha256,
        expected_offset_table_sha256=expected_offset_table_sha256,
        expected_support_contract_sha256=expected_support_contract_sha256,
    )
    return payload, actual_decision_sha


def _expected_shard(contract: Mapping[str, Any], case_id: str) -> int:
    matches = [int(index) for index, case_ids in contract["shards"].items() if case_id in case_ids]
    if len(matches) != 1:
        raise RuntimeError(f"C4 case has no unique frozen shard: {case_id}")
    return matches[0]


def _validate_cuda_execution(
    payload: Mapping[str, Any],
    contract: Mapping[str, Any],
    *,
    phase: str,
    shard_index: int,
    physical_gpu: str,
    attempt_id: str | None = None,
) -> None:
    execution = payload.get("execution") or {}
    runtime = contract.get("runtime_signature") or {}
    observed_attempt = execution.get("attempt_id")
    if (
        execution.get("phase") != phase
        or execution.get("shard_index") != shard_index
        or execution.get("physical_gpu") != physical_gpu
        or execution.get("seed") != contract.get("seed")
        or execution.get("deterministic") is not True
        or not isinstance(observed_attempt, str)
        or not observed_attempt
        or (attempt_id is not None and observed_attempt != attempt_id)
        or execution.get("python") != runtime.get("python")
        or execution.get("torch") != runtime.get("torch")
        or not str(execution.get("device", "")).startswith("cuda")
        or not isinstance(execution.get("host"), str)
        or not execution.get("host")
        or not isinstance(execution.get("gpu_name"), str)
        or not execution.get("gpu_name")
    ):
        raise RuntimeError(f"Invalid C4 {phase} CUDA execution provenance")
    if phase == "decision":
        if execution.get("labels_loaded_to_device") is not False or "labels_loaded_after_barrier" in execution:
            raise RuntimeError("C4 decision execution provenance carries label access")
    elif phase == "evaluation":
        if execution.get("labels_loaded_after_barrier") is not True or "labels_loaded_to_device" in execution:
            raise RuntimeError("C4 evaluation execution provenance is not post-barrier")
    else:
        raise ValueError(f"Unsupported C4 execution phase: {phase}")


def _validate_candidate_arm(
    row: Mapping[str, Any],
    spec: Mapping[str, Any],
    contract: Mapping[str, Any],
    *,
    verify_heavy_bytes: bool,
) -> None:
    if row.get("arm_id") != spec["arm_id"] or row.get("arm_index") != spec["arm_index"]:
        raise RuntimeError("C4 decision arm order or identity changed")
    expected_actions = {"ACCEPT", "ROLLBACK"} if spec["materialize_candidate"] else {"DIAGNOSTIC_ONLY"}
    if row.get("action") not in expected_actions:
        raise RuntimeError(f"C4 arm has an unsupported action: {spec['arm_id']}")
    if row.get("support_contract_sha256") != contract["support_contract_sha256"]:
        raise RuntimeError(f"C4 arm support contract changed: {spec['arm_id']}")
    materialized_keys = {"candidate_field", "persistence", "exact"}
    if not spec["materialize_candidate"]:
        if materialized_keys & set(row):
            raise RuntimeError(f"Nonmaterialized C4 diagnostic carries a candidate: {spec['arm_id']}")
        return
    if not materialized_keys.issubset(row):
        raise RuntimeError(f"Materialized C4 arm lacks persistence evidence: {spec['arm_id']}")
    support = row.get("support") or {}
    utility = row.get("utility") or {}
    if support.get("support_id") != contract["support_contract"]["support_id"]:
        raise RuntimeError(f"C4 arm has the wrong common support: {spec['arm_id']}")
    baseline_count = support.get("baseline_count")
    pair_count = support.get("pair_count")
    if (
        isinstance(baseline_count, bool)
        or not isinstance(baseline_count, int)
        or baseline_count < 1
        or isinstance(pair_count, bool)
        or not isinstance(pair_count, int)
        or not 0 < pair_count <= baseline_count
    ):
        raise RuntimeError(f"C4 arm has invalid common-support counts: {spec['arm_id']}")
    retention = _finite_number(support.get("retention"), f"C4 {spec['arm_id']} retention")
    if not 0.0 <= retention <= 1.0 or not math.isclose(
        retention, pair_count / baseline_count, rel_tol=0.0, abs_tol=1e-15
    ):
        raise RuntimeError(f"C4 arm has inconsistent common-support retention: {spec['arm_id']}")
    if utility.get("utility_id") != contract["support_contract"]["utility_id"]:
        raise RuntimeError(f"C4 arm has the wrong label-free utility: {spec['arm_id']}")
    baseline_loss = _finite_number(utility.get("baseline_loss"), f"C4 {spec['arm_id']} baseline utility")
    candidate_loss = _finite_number(utility.get("candidate_loss"), f"C4 {spec['arm_id']} candidate utility")
    improvement = _finite_number(utility.get("improvement"), f"C4 {spec['arm_id']} utility improvement")
    if not math.isclose(improvement, baseline_loss - candidate_loss, rel_tol=0.0, abs_tol=1e-12):
        raise RuntimeError(f"C4 arm utility arithmetic is inconsistent: {spec['arm_id']}")
    should_accept = retention >= float(contract["support_contract"]["utility_retention_min"]) and improvement >= float(
        contract["support_contract"]["improvement_min"]
    )
    if row.get("action") != ("ACCEPT" if should_accept else "ROLLBACK"):
        raise RuntimeError(f"C4 arm action disagrees with the frozen transaction: {spec['arm_id']}")
    field = row.get("candidate_field") or {}
    persistence = row.get("persistence") or {}
    exact = row.get("exact") or {}
    if (
        persistence.get("protocol_id") != PERSISTENCE_PROTOCOL_ID
        or persistence.get("saved_npz_sha256") != field.get("npz_sha256")
        or persistence.get("reloaded_array_sha256") != field.get("array_sha256")
        or exact.get("sha256") != field.get("array_sha256")
        or exact.get("certified") is not True
        or exact.get("status") != "CERTIFIED"
    ):
        raise RuntimeError(f"C4 save/reload/certificate metadata is inconsistent: {spec['arm_id']}")
    if row["action"] == "ACCEPT":
        if row.get("returned_field") != field or row.get("rollback_to_source_initial") is not False:
            raise RuntimeError(f"C4 accepted arm does not return its certified field: {spec['arm_id']}")
    elif row.get("returned_field") is not None or row.get("rollback_to_source_initial") is not True:
        raise RuntimeError(f"C4 rolled-back arm does not return the exact source state: {spec['arm_id']}")
    _validate_geometry_bundle(row.get("geometry"), f"C4 {spec['arm_id']} candidate")
    _require_sha(field.get("npz_sha256"), f"C4 {spec['arm_id']} npz")
    _require_sha(field.get("array_sha256"), f"C4 {spec['arm_id']} array")
    if verify_heavy_bytes:
        _resolve_heavy_field(Path(contract["heavy_root"]).resolve(), field, verify_array=True)


def validate_decision_case_marker(
    payload: Mapping[str, Any],
    contract: Mapping[str, Any],
    decision_contract_sha256: str,
    *,
    verify_heavy_bytes: bool = True,
) -> None:
    case_id = payload.get("case_id")
    if case_id not in contract.get("case_ids", []):
        raise RuntimeError("C4 decision marker belongs to a foreign case")
    shard = _expected_shard(contract, case_id)
    if (
        payload.get("schema") != DECISION_CASE_SCHEMA
        or payload.get("protocol_id") != PROTOCOL_ID
        or payload.get("status") != "COMPLETE"
        or payload.get("decision_contract_sha256") != _require_sha(decision_contract_sha256, "C4 decision contract")
        or payload.get("shard_index") != shard
        or payload.get("physical_gpu") != contract["shard_to_physical_gpu"][str(shard)]
        or payload.get("arm_specs_sha256") != contract["arm_specs_sha256"]
        or payload.get("offset_table_sha256") != contract["offset_table_sha256"]
        or payload.get("support_contract_sha256") != contract["support_contract_sha256"]
        or payload.get("labels_loaded_to_device") is not False
        or payload.get("test_split_accessed") is not False
        or payload.get("source_image_array_sha256") != contract["image_inputs"][case_id]["array_sha256"]
        or payload.get("source_initial_array_sha256") != contract["source_initial"][case_id]["field"]["array_sha256"]
    ):
        raise RuntimeError(f"Invalid C4 decision marker: {case_id}")
    arms = payload.get("arms")
    if not isinstance(arms, list) or len(arms) != len(contract["arm_specs"]):
        raise RuntimeError(f"C4 decision marker has the wrong arm set: {case_id}")
    for row, spec in zip(arms, contract["arm_specs"], strict=True):
        _validate_candidate_arm(row, spec, contract, verify_heavy_bytes=verify_heavy_bytes)
    _validate_geometry_bundle(payload.get("baseline_geometry"), f"C4 {case_id} source baseline")
    common = payload.get("support") or {}
    geometry_count = common.get("geometry_count")
    common_count = common.get("common_count")
    if (
        isinstance(geometry_count, bool)
        or not isinstance(geometry_count, int)
        or geometry_count < 1
        or isinstance(common_count, bool)
        or not isinstance(common_count, int)
        or not 0 < common_count <= geometry_count
    ):
        raise RuntimeError(f"C4 decision marker has invalid common descriptor support: {case_id}")
    retention = _finite_number(common.get("retention"), f"C4 {case_id} common descriptor retention")
    if not math.isclose(retention, common_count / geometry_count, rel_tol=0.0, abs_tol=1e-15):
        raise RuntimeError(f"C4 decision marker has inconsistent common descriptor support: {case_id}")
    _validate_cuda_execution(
        payload,
        contract,
        phase="decision",
        shard_index=shard,
        physical_gpu=contract["shard_to_physical_gpu"][str(shard)],
    )
    _assert_label_free(payload, f"C4 decision marker {case_id}")


def validate_worker_marker(
    payload: Mapping[str, Any],
    contract: Mapping[str, Any],
    decision_contract_sha256: str,
    *,
    phase: str,
    shard_index: int,
    attempt_id: str,
    barrier_sha256: str | None = None,
) -> None:
    if phase not in {"decision", "evaluation"}:
        raise ValueError("C4 worker phase must be decision or evaluation")
    expected_cases = contract["shards"].get(str(shard_index))
    if (
        payload.get("schema") != WORKER_SCHEMA
        or payload.get("protocol_id") != PROTOCOL_ID
        or payload.get("status") != "COMPLETE"
        or payload.get("phase") != phase
        or payload.get("attempt_id") != attempt_id
        or payload.get("shard_index") != shard_index
        or payload.get("physical_gpu") != contract["shard_to_physical_gpu"].get(str(shard_index))
        or payload.get("case_ids") != expected_cases
        or payload.get("decision_contract_sha256") != decision_contract_sha256
        or payload.get("test_split_accessed") is not False
    ):
        raise RuntimeError(f"Invalid C4 {phase} worker marker for shard {shard_index}")
    if phase == "decision":
        if payload.get("labels_loaded_to_device") is not False:
            raise RuntimeError("C4 decision worker received labels")
        _validate_cuda_execution(
            payload,
            contract,
            phase="decision",
            shard_index=shard_index,
            physical_gpu=contract["shard_to_physical_gpu"][str(shard_index)],
            attempt_id=attempt_id,
        )
        _assert_label_free(payload, f"C4 decision worker {shard_index}")
    elif (
        payload.get("decision_barrier_sha256") != _require_sha(barrier_sha256, "C4 decision barrier")
        or payload.get("labels_loaded_after_barrier") is not True
    ):
        raise RuntimeError("C4 evaluation worker is not bound to the frozen barrier and label phase")
    else:
        _validate_cuda_execution(
            payload,
            contract,
            phase="evaluation",
            shard_index=shard_index,
            physical_gpu=contract["shard_to_physical_gpu"][str(shard_index)],
            attempt_id=attempt_id,
        )


def build_decision_barrier(
    run_root: Path,
    contract: Mapping[str, Any],
    decision_contract_sha256: str,
    *,
    attempt_id: str,
    worker_paths: Sequence[Path],
    verify_heavy_bytes: bool = True,
    completed_at_utc: str,
) -> dict[str, Any]:
    if len(worker_paths) != contract["num_shards"]:
        raise RuntimeError("C4 decision barrier requires exactly one worker per shard")
    workers: list[dict[str, str]] = []
    for shard, path in enumerate(worker_paths):
        payload = _load_json(path)
        validate_worker_marker(
            payload,
            contract,
            decision_contract_sha256,
            phase="decision",
            shard_index=shard,
            attempt_id=attempt_id,
        )
        workers.append({"path": path.resolve().relative_to(run_root.resolve()).as_posix(), "sha256": sha256_file(path)})
    decision_hashes: dict[str, str] = {}
    for case_id in contract["case_ids"]:
        path = run_root.resolve() / "cases" / case_id / "decision_complete.json"
        payload = _load_json(path)
        validate_decision_case_marker(
            payload,
            contract,
            decision_contract_sha256,
            verify_heavy_bytes=verify_heavy_bytes,
        )
        decision_hashes[case_id] = sha256_file(path)
    return {
        "schema": BARRIER_SCHEMA,
        "protocol_id": PROTOCOL_ID,
        "status": "COMPLETE",
        "attempt_id": attempt_id,
        "decision_contract_sha256": decision_contract_sha256,
        "decision_workers_received_label_inputs": False,
        "test_split_accessed": False,
        "workers": workers,
        "decision_case_sha256": decision_hashes,
        "completed_at_utc": completed_at_utc,
    }


def write_decision_barrier(run_root: Path, payload: dict[str, Any]) -> str:
    return _write_immutable_json(run_root.resolve() / "decision_barrier.json", payload)


def load_decision_barrier(
    run_root: Path,
    expected_sha256: str,
    *,
    decision_contract_sha256: str,
    case_ids: Sequence[str],
) -> tuple[dict[str, Any], str]:
    path = run_root.resolve() / "decision_barrier.json"
    actual = sha256_file(path)
    if actual != _require_sha(expected_sha256, "C4 decision barrier"):
        raise RuntimeError("C4 decision barrier hash mismatch")
    payload = _load_json(path)
    if (
        payload.get("schema") != BARRIER_SCHEMA
        or payload.get("protocol_id") != PROTOCOL_ID
        or payload.get("status") != "COMPLETE"
        or payload.get("decision_contract_sha256") != decision_contract_sha256
        or payload.get("decision_workers_received_label_inputs") is not False
        or payload.get("test_split_accessed") is not False
        or set(payload.get("decision_case_sha256") or {}) != set(case_ids)
    ):
        raise RuntimeError("Invalid or altered C4 decision barrier")
    for value in (payload.get("decision_case_sha256") or {}).values():
        _require_sha(value, "C4 decision case")
    return payload, actual


def validate_evaluation_case_marker(
    payload: Mapping[str, Any],
    contract: Mapping[str, Any],
    decision_contract_sha256: str,
    barrier: Mapping[str, Any],
    barrier_sha256: str,
) -> None:
    case_id = payload.get("case_id")
    if case_id not in contract.get("case_ids", []):
        raise RuntimeError("C4 evaluation marker belongs to a foreign case")
    if (
        payload.get("schema") != EVALUATION_CASE_SCHEMA
        or payload.get("protocol_id") != PROTOCOL_ID
        or payload.get("status") != "COMPLETE"
        or payload.get("decision_contract_sha256") != decision_contract_sha256
        or payload.get("decision_barrier_sha256") != _require_sha(barrier_sha256, "C4 decision barrier")
        or payload.get("decision_case_sha256") != (barrier.get("decision_case_sha256") or {}).get(case_id)
        or payload.get("labels_loaded_after_barrier") is not True
        or payload.get("test_split_accessed") is not False
    ):
        raise RuntimeError(f"Invalid C4 post-barrier evaluation marker: {case_id}")
    labels = payload.get("labels")
    if (
        payload.get("baseline_c3_parity_verified") is not True
        or not isinstance(labels, list)
        or not labels
        or any(isinstance(label, bool) or not isinstance(label, int) for label in labels)
        or len(set(labels)) != len(labels)
    ):
        raise RuntimeError(f"C4 evaluation marker has invalid frozen labels or baseline parity: {case_id}")
    arms = payload.get("arms")
    if not isinstance(arms, list) or [row.get("arm_id") for row in arms] != list(ALL_ARM_IDS):
        raise RuntimeError(f"C4 evaluation marker has the wrong arm order: {case_id}")
    if [row.get("arm_index") for row in arms] != list(range(len(ALL_ARM_IDS))):
        raise RuntimeError(f"C4 evaluation marker has the wrong arm indices: {case_id}")
    baseline_values: set[float] = set()
    for row, spec in zip(arms, contract["arm_specs"], strict=True):
        if row.get("evaluated") is not spec["post_barrier_evaluation"]:
            raise RuntimeError(f"C4 evaluation scope changed for arm: {spec['arm_id']}")
        if not spec["post_barrier_evaluation"]:
            if any("dice" in str(key).lower() for key in row) or row.get("primary_action") != "DIAGNOSTIC_ONLY":
                raise RuntimeError(f"Non-evaluated C4 diagnostic carries Dice or an action: {spec['arm_id']}")
            continue
        required = {
            "baseline_dice",
            "capacity_candidate_dice",
            "capacity_dice_delta",
            "primary_returned_dice",
            "primary_dice_delta",
            "primary_action",
            "per_label",
        }
        if not required.issubset(row) or row.get("primary_action") not in {"ACCEPT", "ROLLBACK"}:
            raise RuntimeError(f"Evaluated C4 arm lacks complete Dice evidence: {spec['arm_id']}")
        baseline = _unit_interval(row["baseline_dice"], f"C4 {spec['arm_id']} baseline Dice")
        candidate = _unit_interval(row["capacity_candidate_dice"], f"C4 {spec['arm_id']} candidate Dice")
        returned = _unit_interval(row["primary_returned_dice"], f"C4 {spec['arm_id']} returned Dice")
        capacity_delta = _finite_number(row["capacity_dice_delta"], f"C4 {spec['arm_id']} capacity delta")
        primary_delta = _finite_number(row["primary_dice_delta"], f"C4 {spec['arm_id']} returned delta")
        expected_returned = candidate if row["primary_action"] == "ACCEPT" else baseline
        if (
            not math.isclose(capacity_delta, candidate - baseline, rel_tol=0.0, abs_tol=DICE_ARITHMETIC_ATOL)
            or not math.isclose(primary_delta, returned - baseline, rel_tol=0.0, abs_tol=DICE_ARITHMETIC_ATOL)
            or not math.isclose(returned, expected_returned, rel_tol=0.0, abs_tol=DICE_ARITHMETIC_ATOL)
        ):
            raise RuntimeError(f"C4 evaluated Dice arithmetic changed: {spec['arm_id']}")
        per_label = row["per_label"]
        if not isinstance(per_label, list) or [item.get("label") for item in per_label] != labels:
            raise RuntimeError(f"C4 per-label Dice inventory changed: {spec['arm_id']}")
        per_baselines: list[float] = []
        per_candidates: list[float] = []
        per_returned_values: list[float] = []
        for item in per_label:
            per_baseline = _unit_interval(item.get("baseline_dice"), f"C4 {spec['arm_id']} label baseline")
            per_candidate = _unit_interval(item.get("candidate_dice"), f"C4 {spec['arm_id']} label candidate")
            per_returned = _unit_interval(item.get("returned_dice"), f"C4 {spec['arm_id']} label returned")
            per_baselines.append(per_baseline)
            per_candidates.append(per_candidate)
            per_returned_values.append(per_returned)
            expected_per_returned = per_candidate if row["primary_action"] == "ACCEPT" else per_baseline
            if not math.isclose(
                per_returned,
                expected_per_returned,
                rel_tol=0.0,
                abs_tol=DICE_ARITHMETIC_ATOL,
            ):
                raise RuntimeError(f"C4 per-label returned Dice disagrees with its action: {spec['arm_id']}")
        if any(
            not math.isclose(observed, sum(values) / len(values), rel_tol=0.0, abs_tol=DICE_AGGREGATE_ATOL)
            for observed, values in (
                (baseline, per_baselines),
                (candidate, per_candidates),
                (returned, per_returned_values),
            )
        ):
            raise RuntimeError(f"C4 aggregate Dice disagrees with per-label values: {spec['arm_id']}")
        baseline_values.add(baseline)
    if len(baseline_values) != 1:
        raise RuntimeError(f"C4 evaluation arms disagree on the case baseline: {case_id}")
    shard = _expected_shard(contract, str(case_id))
    _validate_cuda_execution(
        payload,
        contract,
        phase="evaluation",
        shard_index=shard,
        physical_gpu=contract["shard_to_physical_gpu"][str(shard)],
    )
