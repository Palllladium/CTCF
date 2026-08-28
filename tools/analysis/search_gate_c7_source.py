from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np

from tools.analysis.run_artifacts import atomic_write_json, sha256_file
from tools.analysis.search_gate_c7 import (
    EVALUATION_LABEL_IDS,
    EXPECTED_CASE_COUNT,
    REFERENCE_ARM_ID,
    SOURCE_C6_CONTEXT_ARM_ID,
    SOURCE_C6_POLICY_SHA256,
    SOURCE_C6_PROTOCOL_ID,
)
from tools.analysis.search_gate_pyramid import array_sha256
from tools.analysis.transactional_search import load_flow_npz

C6_MANIFEST_SHA256 = "e83e5bafeee830a3b7c96e75e57fde8f97c2590f76f80538d1a48e3ea7d3353f"
C6_MANIFEST_SCHEMA = "ctcf-search-c6-run-manifest-v2"
C6_EVALUATION_CONTRACT_SCHEMA = "ctcf-search-c6-evaluation-contract-v2"
C6_EVALUATION_BARRIER_SCHEMA = "ctcf-search-c6-evaluation-barrier-v2"
C6_EVALUATION_CASE_SCHEMA = "ctcf-search-c6-evaluation-case-v2"
SOURCE_SCHEMA = "ctcf-search-c7-source-v1"
DECISION_SCHEMA = "ctcf-search-c7-decision-v1"
DECISION_CASE_SCHEMA = "ctcf-search-c7-decision-case-v1"
WORKER_SCHEMA = "ctcf-search-c7-worker-v1"
BARRIER_SCHEMA = "ctcf-search-c7-decision-barrier-v1"
EVALUATION_CONTRACT_SCHEMA = "ctcf-search-c7-evaluation-contract-v1"
EVALUATION_CASE_SCHEMA = "ctcf-search-c7-evaluation-case-v1"
EVALUATION_BARRIER_SCHEMA = "ctcf-search-c7-evaluation-barrier-v1"


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"expected a JSON object: {path}")
    return payload


def json_equivalent(left: Any, right: Any) -> bool:
    options = {"ensure_ascii": False, "sort_keys": True, "separators": (",", ":"), "allow_nan": False}
    return json.dumps(left, **options) == json.dumps(right, **options)


def immutable_json(path: Path, payload: Mapping[str, Any]) -> str:
    frozen = dict(payload)
    if path.exists():
        if not json_equivalent(load_json(path), frozen):
            raise FileExistsError(f"refusing to replace immutable C7 artifact: {path}")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_json(path, frozen)
    return sha256_file(path)


def _relative_path(value: Any) -> PurePosixPath:
    if not isinstance(value, str) or not value or "\\" in value:
        raise RuntimeError("C7 rooted artifact path must be a POSIX relative path")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise RuntimeError("C7 rooted artifact path escapes its root")
    return path


def roots(decision: Mapping[str, Any]) -> dict[str, Path]:
    value = decision.get("roots")
    expected = {"source_c3_heavy", "source_c4_heavy", "source_c6_heavy", "target_c7_heavy"}
    if not isinstance(value, Mapping) or set(value) != expected:
        raise RuntimeError("C7 rooted-storage inventory changed")
    result = {str(key): Path(str(path)).resolve() for key, path in value.items()}
    if len(set(result.values())) != len(result):
        raise RuntimeError("C7 storage roots must be distinct")
    values = list(result.values())
    if any(
        left in right.parents or right in left.parents
        for index, left in enumerate(values)
        for right in values[index + 1 :]
    ):
        raise RuntimeError("C7 storage roots must not overlap")
    return result


def verify_record(
    decision: Mapping[str, Any],
    record: Mapping[str, Any],
    *,
    verify_array: bool = False,
) -> Path:
    owner_roots = roots(decision)
    root_id = record.get("root_id")
    if root_id not in owner_roots:
        raise RuntimeError("C7 artifact has an unknown root owner")
    root = owner_roots[str(root_id)]
    relative = _relative_path(record.get("relative_path"))
    path = root.joinpath(*relative.parts).resolve()
    if root not in path.parents:
        raise RuntimeError("C7 rooted artifact escaped its root")
    expected = record.get("npz_sha256", record.get("sha256"))
    if not path.is_file() or sha256_file(path) != expected:
        raise RuntimeError(f"C7 artifact bytes changed or are absent: {path}")
    if verify_array:
        tensor = load_flow_npz(path)
        if array_sha256(tensor) != record.get("array_sha256"):
            raise RuntimeError(f"C7 field array changed: {path}")
    return path


def load_field(decision: Mapping[str, Any], record: Mapping[str, Any]):
    return load_flow_npz(verify_record(decision, record, verify_array=True))


def load_image(decision: Mapping[str, Any], record: Mapping[str, Any]) -> np.ndarray:
    path = verify_record(decision, record)
    array = np.load(path, allow_pickle=False)
    if (
        array.dtype != np.float32
        or list(array.shape) != record.get("shape")
        or not np.isfinite(array).all()
        or array_sha256(np_to_tensor(array)) != record.get("array_sha256")
    ):
        raise RuntimeError(f"C7 cached image changed: {path}")
    return np.ascontiguousarray(array)


def np_to_tensor(array: np.ndarray):
    import torch

    return torch.from_numpy(array)


def field_record(path: Path, heavy_root: Path, digest: str) -> dict[str, Any]:
    return {
        "root_id": "target_c7_heavy",
        "relative_path": path.resolve().relative_to(heavy_root.resolve()).as_posix(),
        "npz_sha256": sha256_file(path),
        "array_sha256": digest,
    }


def assert_decision_payload_label_free(payload: Mapping[str, Any]) -> None:
    false_flags = {"labels_loaded", "labels_loaded_to_device", "test_split_accessed", "test_115_authorized"}

    def visit(value: Any, path: tuple[str, ...] = ()) -> None:
        if isinstance(value, Mapping):
            for key, child in value.items():
                token = str(key).lower()
                child_path = (*path, str(key))
                if token in false_flags:
                    if child is not False:
                        raise RuntimeError(f"C7 protected flag is not false: {'.'.join(child_path)}")
                    continue
                if (
                    "dice" in token
                    or "segmentation" in token
                    or "label" in token
                    or ("evaluation" in token and token != "evaluation_device")
                ):
                    raise RuntimeError(f"C7 decision payload contains evaluation data: {'.'.join(child_path)}")
                visit(child, child_path)
        elif isinstance(value, (list, tuple)):
            for index, child in enumerate(value):
                visit(child, (*path, str(index)))
        elif isinstance(value, str) and ("segmentation" in value.lower() or value.lower().endswith(".pkl")):
            raise RuntimeError(f"C7 decision payload contains an evaluation locator: {'.'.join(path)}")

    visit(payload)


def _require_file(root: Path, relative: str, expected: str) -> Path:
    path = root / relative
    if not path.is_file() or sha256_file(path) != expected:
        raise RuntimeError(f"authenticated C6 artifact changed or is absent: {path}")
    return path


def _retag_record(record: Mapping[str, Any], root_id: str) -> dict[str, Any]:
    result = dict(record)
    result["root_id"] = root_id
    return result


def _verify_source_record(record: Mapping[str, Any], owner_roots: Mapping[str, Path], *, array: bool) -> None:
    root_id = record.get("root_id")
    if root_id not in owner_roots:
        raise RuntimeError("C6 source record has an unknown root")
    relative = _relative_path(record.get("relative_path"))
    root = owner_roots[str(root_id)]
    path = root.joinpath(*relative.parts).resolve()
    if root not in path.parents or not path.is_file():
        raise RuntimeError(f"C6 source record is absent or escaped its root: {path}")
    if sha256_file(path) != record.get("npz_sha256", record.get("sha256")):
        raise RuntimeError(f"C6 source record SHA-256 changed: {path}")
    if array and array_sha256(load_flow_npz(path)) != record.get("array_sha256"):
        raise RuntimeError(f"C6 source field payload changed: {path}")


def _arm(marker: Mapping[str, Any], arm_id: str) -> Mapping[str, Any]:
    rows = [row for row in marker.get("arms", []) if isinstance(row, Mapping) and row.get("arm_id") == arm_id]
    if len(rows) != 1:
        raise RuntimeError(f"C6 marker does not contain exactly one arm {arm_id}")
    return rows[0]


def authenticate_c6_source(
    compact_root: Path,
    source_c3_heavy: Path,
    source_c4_heavy: Path,
    source_c6_heavy: Path,
    *,
    verify_heavy_bytes: bool = True,
) -> dict[str, Any]:
    compact_root = compact_root.resolve()
    manifest_path = _require_file(compact_root, "c6_manifest.json", C6_MANIFEST_SHA256)
    manifest = load_json(manifest_path)
    if (
        manifest.get("schema") != C6_MANIFEST_SCHEMA
        or manifest.get("protocol_id") != SOURCE_C6_PROTOCOL_ID
        or manifest.get("policy_sha256") != SOURCE_C6_POLICY_SHA256
        or manifest.get("status") != "COMPLETE"
        or manifest.get("test_115_authorized") is not False
        or manifest.get("test_split_accessed") is not False
    ):
        raise RuntimeError("C6 source manifest is not the frozen successful run")
    fixed_files = {
        "decision_contract.json": manifest["decision_contract_sha256"],
        "decision_barrier.json": manifest["decision_barrier_sha256"],
    }
    for relative, digest in fixed_files.items():
        _require_file(compact_root, relative, digest)
    post_barrier_files = {
        "metric_contract_sha256": manifest["evaluation_contract_sha256"],
        "metric_barrier_sha256": manifest["evaluation_barrier_sha256"],
    }
    decision = load_json(compact_root / "decision_contract.json")
    barrier = load_json(compact_root / "decision_barrier.json")
    case_ids = list(decision.get("case_ids") or [])
    if (
        len(case_ids) != EXPECTED_CASE_COUNT
        or len(set(case_ids)) != EXPECTED_CASE_COUNT
        or barrier.get("decision_case_sha256") != manifest.get("decision_case_sha256")
        or decision.get("policy_sha256") != SOURCE_C6_POLICY_SHA256
    ):
        raise RuntimeError("C6 source contracts or case inventory changed")
    supplied_roots = {
        "source_c3_heavy": source_c3_heavy.resolve(),
        "source_c4_heavy": source_c4_heavy.resolve(),
        "target_c6_heavy": source_c6_heavy.resolve(),
    }
    observed_roots = {key: Path(str(value)).resolve() for key, value in (decision.get("roots") or {}).items()}
    if supplied_roots != observed_roots:
        raise RuntimeError(f"C6 heavy roots differ from its frozen contract: {supplied_roots} != {observed_roots}")

    c4_anchors: dict[str, Any] = {}
    c6_context: dict[str, Any] = {}
    for case_id in case_ids:
        decision_path = _require_file(
            compact_root,
            f"cases/{case_id}/decision_complete.json",
            manifest["decision_case_sha256"][case_id],
        )
        decision_marker = load_json(decision_path)
        reference = _arm(decision_marker, REFERENCE_ARM_ID)
        context = _arm(decision_marker, SOURCE_C6_CONTEXT_ARM_ID)
        reference_record = _retag_record(reference["candidate_field"], "source_c4_heavy")
        context_record = _retag_record(context["candidate_field"], "source_c6_heavy")
        c4_anchors[case_id] = {"field": reference_record, "geometry": reference["geometry"]}
        c6_context[case_id] = {
            "field": context_record,
            "geometry": context["geometry"],
            "action": context["action"],
            "support": context["support"],
            "utility": context["utility"],
        }
        if verify_heavy_bytes:
            _verify_source_record(reference_record, supplied_roots, array=True)
            _verify_source_record(context_record, supplied_roots, array=True)

    image_inputs = decision["image_inputs"]
    source_initial = decision["source_initial"]
    source_historical = decision["source_historical"]
    if verify_heavy_bytes:
        for record in image_inputs.values():
            _verify_source_record(record, supplied_roots, array=False)
        for case_id in case_ids:
            _verify_source_record(source_initial[case_id]["field"], supplied_roots, array=True)
            _verify_source_record(source_historical[case_id]["raw_conf_requested_field"], supplied_roots, array=True)

    return {
        "schema": "ctcf-search-c7-authenticated-c6-projection-v1",
        "source_c6_manifest_sha256": C6_MANIFEST_SHA256,
        "source_c6_files": {relative: digest for relative, digest in fixed_files.items()},
        "source_c6_decision_case_sha256": manifest["decision_case_sha256"],
        "post_barrier_metric_source": {
            "compact_root": str(compact_root),
            **post_barrier_files,
            "metric_case_sha256": manifest["evaluation_case_sha256"],
        },
        "case_ids": case_ids,
        "seed": decision["seed"],
        "image_inputs": image_inputs,
        "source_initial": source_initial,
        "source_historical": source_historical,
        "baseline_geometry": decision["baseline_geometry"],
        "source_c4_anchors": c4_anchors,
        "source_c6_context": c6_context,
        "roots": {
            "source_c3_heavy": str(source_c3_heavy.resolve()),
            "source_c4_heavy": str(source_c4_heavy.resolve()),
            "source_c6_heavy": str(source_c6_heavy.resolve()),
        },
        "test_115_authorized": False,
        "test_split_accessed": False,
    }


def load_c6_metrics_after_barrier(snapshot: Mapping[str, Any]) -> dict[str, Any]:
    source = snapshot.get("post_barrier_metric_source")
    if not isinstance(source, Mapping):
        raise RuntimeError("C7 post-barrier metric source is absent")
    compact_root = Path(str(source.get("compact_root", ""))).resolve()
    contract_path = _require_file(
        compact_root,
        "evaluation_contract.json",
        str(source.get("metric_contract_sha256", "")),
    )
    barrier_path = _require_file(
        compact_root,
        "evaluation_barrier.json",
        str(source.get("metric_barrier_sha256", "")),
    )
    contract = load_json(contract_path)
    barrier = load_json(barrier_path)
    case_ids = list(snapshot.get("case_ids") or [])
    case_hashes = source.get("metric_case_sha256")
    label_ids = contract.get("evaluation_label_ids")
    if (
        len(case_ids) != EXPECTED_CASE_COUNT
        or contract.get("schema") != C6_EVALUATION_CONTRACT_SCHEMA
        or contract.get("protocol_id") != SOURCE_C6_PROTOCOL_ID
        or contract.get("case_ids") != case_ids
        or label_ids != list(EVALUATION_LABEL_IDS)
        or contract.get("test_115_authorized") is not False
        or contract.get("test_split_accessed") is not False
        or barrier.get("schema") != C6_EVALUATION_BARRIER_SCHEMA
        or barrier.get("protocol_id") != SOURCE_C6_PROTOCOL_ID
        or barrier.get("status") != "COMPLETE"
        or barrier.get("strict") is not True
        or barrier.get("evaluation_contract_sha256") != source.get("metric_contract_sha256")
        or barrier.get("evaluation_case_sha256") != case_hashes
        or not isinstance(case_hashes, Mapping)
        or set(case_hashes) != set(case_ids)
    ):
        raise RuntimeError("C7 post-barrier C6 metric contract is invalid or altered")
    baseline_dice = contract.get("evaluation_baseline_dice")
    baseline_per_label = contract.get("evaluation_baseline_per_label")
    if (
        not isinstance(baseline_dice, Mapping)
        or set(baseline_dice) != set(case_ids)
        or not isinstance(baseline_per_label, Mapping)
        or set(baseline_per_label) != set(case_ids)
    ):
        raise RuntimeError("C7 post-barrier C6 baseline inventory changed")
    source_rows: dict[str, Any] = {}
    for case_id in case_ids:
        marker_path = _require_file(
            compact_root,
            f"cases/{case_id}/evaluation_complete.json",
            str(case_hashes[case_id]),
        )
        marker = load_json(marker_path)
        baseline_rows = baseline_per_label[case_id]
        baseline_labels = [row.get("label") for row in baseline_rows]
        baseline_values = np.asarray([row.get("dice") for row in baseline_rows], dtype=np.float64)
        if (
            marker.get("schema") != C6_EVALUATION_CASE_SCHEMA
            or marker.get("protocol_id") != SOURCE_C6_PROTOCOL_ID
            or marker.get("status") != "COMPLETE"
            or marker.get("strict") is not True
            or marker.get("case_id") != case_id
            or marker.get("evaluation_contract_sha256") != source.get("metric_contract_sha256")
            or marker.get("decision_case_sha256") != snapshot["source_c6_decision_case_sha256"][case_id]
            or marker.get("test_split_accessed") is not False
            or marker.get("labels_loaded_after_barrier") is not True
            or marker.get("labels") != list(EVALUATION_LABEL_IDS)
            or baseline_labels != list(EVALUATION_LABEL_IDS)
            or baseline_values.shape != (len(EVALUATION_LABEL_IDS),)
            or not np.isfinite(baseline_values).all()
            or not np.isclose(
                float(baseline_dice[case_id]),
                float(baseline_values.mean()),
                rtol=0.0,
                atol=1e-12,
            )
        ):
            raise RuntimeError(f"C7 post-barrier C6 metric marker is invalid: {case_id}")
        reference = dict(_arm(marker, REFERENCE_ARM_ID))
        context = dict(_arm(marker, SOURCE_C6_CONTEXT_ARM_ID))
        if (
            reference.get("action") != "REFERENCE"
            or context.get("action") != snapshot["source_c6_context"][case_id]["action"]
        ):
            raise RuntimeError(f"C7 post-barrier C6 source action changed: {case_id}")
        for arm_id, arm in ((REFERENCE_ARM_ID, reference), (SOURCE_C6_CONTEXT_ARM_ID, context)):
            rows = arm.get("per_label") or []
            if [row.get("label") for row in rows] != list(EVALUATION_LABEL_IDS):
                raise RuntimeError(f"C7 post-barrier C6 arm label order changed: {case_id}/{arm_id}")
            candidates = np.asarray([row.get("candidate_dice") for row in rows], dtype=np.float64)
            returned = np.asarray([row.get("returned_dice") for row in rows], dtype=np.float64)
            arm_baseline = np.asarray([row.get("baseline_dice") for row in rows], dtype=np.float64)
            if (
                not np.isfinite(candidates).all()
                or not np.isfinite(returned).all()
                or not np.array_equal(arm_baseline, baseline_values)
                or not np.isclose(float(arm.get("candidate_dice")), float(candidates.mean()), rtol=0.0, atol=1e-12)
                or not np.isclose(float(arm.get("returned_dice")), float(returned.mean()), rtol=0.0, atol=1e-12)
                or not np.array_equal(
                    returned,
                    candidates if arm.get("action") in {"REFERENCE", "ACCEPT"} else baseline_values,
                )
            ):
                raise RuntimeError(f"C7 post-barrier C6 arm arithmetic changed: {case_id}/{arm_id}")
        source_rows[case_id] = {"reference": reference, "context": context}
    return {
        "raw_inputs": contract["raw_inputs"],
        "evaluation_label_ids": label_ids,
        "evaluation_baseline_dice": baseline_dice,
        "evaluation_baseline_per_label": baseline_per_label,
        "source_rows": source_rows,
    }


def validate_case_inventory(case_ids: Sequence[str]) -> tuple[str, ...]:
    frozen = tuple(str(value) for value in case_ids)
    if len(frozen) != EXPECTED_CASE_COUNT or len(set(frozen)) != EXPECTED_CASE_COUNT:
        raise RuntimeError("C7 requires exactly 58 unique validation cases")
    return frozen


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


__all__ = [
    "BARRIER_SCHEMA",
    "C6_MANIFEST_SHA256",
    "DECISION_CASE_SCHEMA",
    "DECISION_SCHEMA",
    "EVALUATION_BARRIER_SCHEMA",
    "EVALUATION_CASE_SCHEMA",
    "EVALUATION_CONTRACT_SCHEMA",
    "SOURCE_SCHEMA",
    "WORKER_SCHEMA",
    "assert_decision_payload_label_free",
    "authenticate_c6_source",
    "field_record",
    "immutable_json",
    "json_equivalent",
    "load_c6_metrics_after_barrier",
    "load_field",
    "load_image",
    "load_json",
    "roots",
    "validate_case_inventory",
    "verify_record",
]
