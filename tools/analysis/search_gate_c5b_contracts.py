from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from tools.analysis.run_artifacts import atomic_write_json, sha256_file
from tools.analysis.search_gate_c5b import (
    ANCHOR_ARM_IDS,
    ARM_SPECS,
    C5B_POLICY_SHA256,
    DIAGNOSTIC_ARM_ID,
    EVALUATION_LABEL_IDS,
    EXPECTED_CASE_COUNT,
    PROTOCOL_ID,
    REFERENCE_ARM_ID,
    SCHEMA_VERSION,
    SELECTABLE_ARM_IDS,
    validate_c5b_geometry_bundle,
)
from tools.analysis.search_gate_c5b_source import (
    assert_c5b_decision_projection_is_label_free,
    authenticate_c5_source,
)
from tools.analysis.search_gate_runtime import round_robin_shards, shard_gpu_map

SOURCE_SCHEMA = f"ctcf-search-c5b-source-{SCHEMA_VERSION}"
DECISION_SCHEMA = f"ctcf-search-c5b-decision-{SCHEMA_VERSION}"
DECISION_CASE_SCHEMA = f"ctcf-search-c5b-decision-case-{SCHEMA_VERSION}"
WORKER_SCHEMA = f"ctcf-search-c5b-worker-{SCHEMA_VERSION}"
BARRIER_SCHEMA = f"ctcf-search-c5b-decision-barrier-{SCHEMA_VERSION}"
EVALUATION_CONTRACT_SCHEMA = f"ctcf-search-c5b-evaluation-contract-{SCHEMA_VERSION}"
EVALUATION_CASE_SCHEMA = f"ctcf-search-c5b-evaluation-case-{SCHEMA_VERSION}"
EVALUATION_BARRIER_SCHEMA = f"ctcf-search-c5b-evaluation-barrier-{SCHEMA_VERSION}"

_SOURCE_ANCHOR_NAMES = (
    "c4_reference_s2_a10_b0",
    "c5_s4_a10_b0_sweep1",
    "c5_s4_a20_b0_sweep1",
)
_EXPECTED_ANCHOR_GEOMETRY_PREFLIGHT = {
    "validated_anchor_count": EXPECTED_CASE_COUNT * len(_SOURCE_ANCHOR_NAMES),
    "central_invalid_count": 0,
    "corner_union_violation_count": 0,
    "digital_ten_nonzero_anchor_count": EXPECTED_CASE_COUNT * len(_SOURCE_ANCHOR_NAMES),
}


def load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError(f"Cannot read C5b JSON: {path}") from error
    if not isinstance(payload, dict):
        raise RuntimeError(f"C5b JSON must contain an object: {path}")
    return payload


def _immutable_json(path: Path, payload: Mapping[str, Any]) -> str:
    candidate = dict(payload)
    if path.exists():
        if load_json(path) != candidate:
            raise FileExistsError(f"Refusing to replace immutable C5b artifact: {path}")
    else:
        atomic_write_json(path, candidate)
    return sha256_file(path)


def _require_sha(value: Any, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        raise RuntimeError(f"{label} is not a lowercase SHA-256")
    return value


def _finite(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise RuntimeError(f"{label} must be a finite real scalar")
    return float(value)


def _nonnegative_integer(value: Any, label: str) -> int:
    observed = _finite(value, label)
    if observed < 0.0 or not observed.is_integer():
        raise RuntimeError(f"{label} must be a non-negative integer")
    return int(observed)


def _array_sha256(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes(order="C")).hexdigest()


def _validate_anchor_geometry_preflight(projection: Mapping[str, Any]) -> None:
    observed = projection.get("anchor_geometry_preflight")
    if not isinstance(observed, Mapping) or dict(observed) != _EXPECTED_ANCHOR_GEOMETRY_PREFLIGHT:
        raise RuntimeError(
            f"C5b historical anchor geometry preflight changed: {observed!r} != {_EXPECTED_ANCHOR_GEOMETRY_PREFLIGHT!r}"
        )


def _assert_decision_payload_label_free(payload: Mapping[str, Any]) -> None:
    allowed_flags = {"labels_loaded", "labels_loaded_to_device"}
    allowed_diagnostic_keys = {"evaluation_device"}
    stack: list[Any] = [payload]
    while stack:
        value = stack.pop()
        if isinstance(value, Mapping):
            for key, child in value.items():
                token = str(key).lower()
                if token in allowed_flags:
                    if child is not False:
                        raise RuntimeError("C5b decision payload reports label access")
                elif token in allowed_diagnostic_keys:
                    pass
                elif any(part in token for part in ("label", "dice", "segmentation", "raw_input", "evaluation")):
                    raise RuntimeError(f"C5b decision payload leaks label-derived metadata: {key}")
                stack.append(child)
        elif isinstance(value, (list, tuple)):
            stack.extend(value)
        elif isinstance(value, str) and value.lower().endswith(".pkl"):
            raise RuntimeError("C5b decision payload leaks a raw label container")


def _safe_record(record: Mapping[str, Any], roots: Mapping[str, str], label: str) -> Path:
    root_id = record.get("root_id")
    relative = record.get("relative_path")
    if root_id not in roots or not isinstance(relative, str) or not relative:
        raise RuntimeError(f"Invalid rooted C5b record: {label}")
    root = Path(roots[str(root_id)]).resolve()
    path = (root / Path(relative)).resolve()
    try:
        path.relative_to(root)
    except ValueError as error:
        raise RuntimeError(f"C5b record escapes its root: {label}") from error
    return path


def verify_field_record(record: Mapping[str, Any], roots: Mapping[str, str], label: str) -> Path:
    path = _safe_record(record, roots, label)
    if not path.is_file() or sha256_file(path) != _require_sha(record.get("npz_sha256"), f"{label} file"):
        raise RuntimeError(f"C5b field bytes changed or are absent: {label}")
    expected_array_sha = _require_sha(record.get("array_sha256"), f"{label} array")
    try:
        with np.load(path, allow_pickle=False) as archive:
            if archive.files != ["flow"]:
                raise RuntimeError(f"C5b field archive inventory changed: {label}")
            array = np.asarray(archive["flow"])
    except (OSError, ValueError) as error:
        raise RuntimeError(f"Cannot read C5b field array: {label}") from error
    if array.dtype != np.float32 or _array_sha256(array) != expected_array_sha:
        raise RuntimeError(f"C5b field array changed: {label}")
    return path


def verify_image_record(record: Mapping[str, Any], roots: Mapping[str, str], label: str) -> Path:
    path = _safe_record(record, roots, label)
    if (
        not path.is_file()
        or path.stat().st_size != int(record.get("bytes", -1))
        or sha256_file(path) != _require_sha(record.get("sha256"), f"{label} file")
    ):
        raise RuntimeError(f"C5b image bytes changed or are absent: {label}")
    _require_sha(record.get("array_sha256"), f"{label} array")
    return path


def _extract_frozen_evaluation(compact_dir: Path, case_ids: Sequence[str]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    source_arm_map = {
        REFERENCE_ARM_ID: "int_s2_a10_b0",
        ANCHOR_ARM_IDS[1]: "int_s4_a10_b0",
        ANCHOR_ARM_IDS[2]: "int_s4_a20_b0",
    }
    for case_id in case_ids:
        payload = load_json(compact_dir / "cases" / case_id / "evaluation_complete.json")
        rows = {row.get("arm_id"): row for row in payload.get("arms", []) if isinstance(row, Mapping)}
        anchors: dict[str, Any] = {}
        for target_id, source_id in source_arm_map.items():
            row = rows.get(source_id)
            if not isinstance(row, Mapping):
                raise RuntimeError(f"Authenticated C5 evaluation lacks {case_id}/{source_id}")
            per_label = row.get("per_label")
            if (
                not isinstance(per_label, list)
                or tuple(item.get("label") for item in per_label) != EVALUATION_LABEL_IDS
            ):
                raise RuntimeError(f"Authenticated C5 label order changed: {case_id}/{source_id}")
            anchors[target_id] = {
                "candidate_dice": float(row["candidate_dice"]),
                "per_label": [
                    {"label": int(item["label"]), "candidate_dice": float(item["candidate_dice"])} for item in per_label
                ],
            }
        first = rows[source_arm_map[REFERENCE_ARM_ID]]
        output[case_id] = {
            "baseline_dice": float(first["baseline_dice"]),
            "baseline_per_label": [
                {"label": int(item["label"]), "dice": float(item["baseline_dice"])} for item in first["per_label"]
            ],
            "anchors": anchors,
        }
    return output


def prepare_contracts(
    *,
    run_root: Path,
    target_heavy_root: Path,
    source_c5_dir: Path,
    source_c5_heavy_root: Path,
    git_head: str,
    runtime_signature: Mapping[str, Any],
    physical_gpus: Sequence[str],
) -> tuple[dict[str, Any], dict[str, Any], str, str]:
    projection = authenticate_c5_source(source_c5_dir, source_c5_heavy_root, verify_heavy_bytes=True)
    _validate_anchor_geometry_preflight(projection)
    case_ids = list(projection["case_ids"])
    if len(case_ids) != EXPECTED_CASE_COUNT:
        raise RuntimeError("C5b source is not the frozen validation-58 inventory")
    target = target_heavy_root.resolve()
    source_roots = {name: Path(value).resolve() for name, value in projection["roots"].items()}
    if target in source_roots.values() or any(
        target in root.parents or root in target.parents for root in source_roots.values()
    ):
        raise RuntimeError("C5b target heavy root overlaps an authenticated source root")
    target.mkdir(parents=True, exist_ok=True)

    authenticated_source = load_json(source_c5_dir / "source_contract.json")
    raw_inputs = authenticated_source.get("raw_inputs")
    if not isinstance(raw_inputs, Mapping) or set(raw_inputs) != {"atlas", *case_ids}:
        raise RuntimeError("Authenticated C5 raw-input inventory changed")
    frozen_evaluation = _extract_frozen_evaluation(source_c5_dir, case_ids)

    roots = {**projection["roots"], "target_c5b_heavy": str(target)}
    shards = round_robin_shards(case_ids, len(physical_gpus))
    sharding = {
        "num_shards": len(physical_gpus),
        "physical_gpus": list(physical_gpus),
        "shard_to_physical_gpu": shard_gpu_map(list(physical_gpus)),
        "shards": shards,
    }
    source = {
        "schema": SOURCE_SCHEMA,
        "protocol_id": PROTOCOL_ID,
        "status": "AUTHENTICATED",
        "git_head": git_head,
        "runtime_signature": dict(runtime_signature),
        "policy_sha256": C5B_POLICY_SHA256,
        "source_projection": projection,
        "roots": roots,
        "raw_inputs": dict(raw_inputs),
        "frozen_evaluation": frozen_evaluation,
        "test_115_authorized": False,
        "test_split_accessed": False,
    }
    decision = {
        "schema": DECISION_SCHEMA,
        "protocol_id": PROTOCOL_ID,
        "status": "FROZEN_LABEL_FREE",
        "git_head": git_head,
        "runtime_signature": dict(runtime_signature),
        "policy_sha256": C5B_POLICY_SHA256,
        "source_identity": projection["source_identity"],
        "case_ids": case_ids,
        "seed": projection["seed"],
        "roots": roots,
        "image_inputs": projection["image_inputs"],
        "source_initial": projection["source_initial"],
        "source_rms": projection["source_rms"],
        "source_anchors": projection["source_anchors"],
        "arm_specs": [asdict(spec) for spec in ARM_SPECS],
        **sharding,
        "decision_input_class": "AUTHENTICATED_IMAGES_AND_FIELDS_ONLY",
        "test_115_authorized": False,
        "test_split_accessed": False,
    }
    assert_c5b_decision_projection_is_label_free(decision)
    run_root.mkdir(parents=True, exist_ok=True)
    source_sha = _immutable_json(run_root / "source_contract.json", source)
    decision["source_contract_sha256"] = source_sha
    decision_sha = _immutable_json(run_root / "decision_contract.json", decision)
    return source, decision, source_sha, decision_sha


def load_decision_contract_isolated(
    run_root: Path,
    decision_sha256: str,
) -> dict[str, Any]:
    decision_path = run_root.resolve() / "decision_contract.json"
    if sha256_file(decision_path) != _require_sha(decision_sha256, "C5b decision contract"):
        raise RuntimeError("C5b decision contract bytes changed")
    decision = load_json(decision_path)
    if (
        decision.get("schema") != DECISION_SCHEMA
        or decision.get("protocol_id") != PROTOCOL_ID
        or decision.get("status") != "FROZEN_LABEL_FREE"
        or decision.get("policy_sha256") != C5B_POLICY_SHA256
        or decision.get("test_115_authorized") is not False
        or decision.get("test_split_accessed") is not False
    ):
        raise RuntimeError("Invalid C5b decision contract")
    _require_sha(decision.get("source_contract_sha256"), "C5b embedded source contract")
    assert_c5b_decision_projection_is_label_free(decision)
    return decision


def load_contracts(run_root: Path, source_sha256: str, decision_sha256: str) -> tuple[dict[str, Any], dict[str, Any]]:
    source_path = run_root.resolve() / "source_contract.json"
    if sha256_file(source_path) != _require_sha(source_sha256, "C5b source contract"):
        raise RuntimeError("C5b source contract bytes changed")
    decision = load_decision_contract_isolated(run_root, decision_sha256)
    source = load_json(source_path)
    if (
        source.get("schema") != SOURCE_SCHEMA
        or source.get("protocol_id") != PROTOCOL_ID
        or source.get("status") != "AUTHENTICATED"
        or source.get("policy_sha256") != C5B_POLICY_SHA256
        or source.get("test_115_authorized") is not False
        or source.get("test_split_accessed") is not False
    ):
        raise RuntimeError("Invalid C5b source contract")
    if decision.get("source_contract_sha256") != source_sha256:
        raise RuntimeError("C5b decision contract points to another source contract")
    if decision.get("case_ids") != source.get("source_projection", {}).get("case_ids"):
        raise RuntimeError("C5b source/decision case inventories disagree")
    _validate_anchor_geometry_preflight(source["source_projection"])
    return source, decision


def decision_case_path(run_root: Path, case_id: str) -> Path:
    return run_root.resolve() / "cases" / case_id / "decision_complete.json"


def evaluation_case_path(run_root: Path, case_id: str) -> Path:
    return run_root.resolve() / "cases" / case_id / "evaluation_complete.json"


def worker_path(run_root: Path, phase: str, attempt_id: str, shard_index: int) -> Path:
    return run_root.resolve() / "workers" / phase / "attempts" / attempt_id / f"worker_{shard_index:02d}.json"


def _expected_shard(decision: Mapping[str, Any], case_id: str) -> int:
    matches = [int(index) for index, values in decision["shards"].items() if case_id in values]
    if len(matches) != 1:
        raise RuntimeError(f"C5b case lacks a unique shard: {case_id}")
    return matches[0]


def _validate_execution(
    execution: Any,
    *,
    phase: str,
    shard_index: int,
    physical_gpu: str,
    labels_loaded_to_device: bool,
) -> None:
    if not isinstance(execution, Mapping) or (
        execution.get("phase") != phase
        or not isinstance(execution.get("attempt_id"), str)
        or not execution.get("attempt_id")
        or execution.get("shard_index") != shard_index
        or str(execution.get("physical_gpu")) != str(physical_gpu)
        or execution.get("labels_loaded_to_device") is not labels_loaded_to_device
        or execution.get("deterministic") is not True
    ):
        raise RuntimeError(f"Invalid C5b {phase} execution provenance")


def validate_decision_case_marker(
    payload: Mapping[str, Any],
    decision: Mapping[str, Any],
    decision_sha256: str,
    *,
    verify_heavy_bytes: bool,
) -> None:
    case_id = payload.get("case_id")
    if case_id not in decision["case_ids"]:
        raise RuntimeError("Invalid C5b decision-case identity")
    shard = _expected_shard(decision, str(case_id))
    if (
        payload.get("schema") != DECISION_CASE_SCHEMA
        or payload.get("protocol_id") != PROTOCOL_ID
        or payload.get("status") != "COMPLETE"
        or payload.get("strict") is not True
        or payload.get("decision_contract_sha256") != decision_sha256
        or payload.get("shard_index") != shard
        or str(payload.get("physical_gpu")) != decision["shard_to_physical_gpu"][str(shard)]
        or payload.get("labels_loaded_to_device") is not False
        or payload.get("test_split_accessed") is not False
    ):
        raise RuntimeError(f"Invalid C5b decision marker: {case_id}")
    arms = payload.get("arms")
    if not isinstance(arms, list) or tuple(row.get("arm_id") for row in arms) != tuple(
        spec.arm_id for spec in ARM_SPECS
    ):
        raise RuntimeError(f"C5b arm inventory changed: {case_id}")
    roots = decision["roots"]
    identities: set[tuple[str, str]] = set()
    direction_sha = payload.get("s4_preclip_direction_array_sha256")
    _require_sha(direction_sha, f"{case_id} S4 direction")
    postprocessed_sha = payload.get("s4_postprocessed_array_sha256")
    _require_sha(postprocessed_sha, f"{case_id} postprocessed S4 direction")
    shared_s4_proposal: tuple[float, float, float, float] | None = None
    for spec, row in zip(ARM_SPECS, arms, strict=True):
        if (
            row.get("arm_index") != spec.arm_index
            or row.get("role") != spec.role
            or row.get("selectable") is not spec.selectable
            or not math.isclose(float(row.get("post_rms_amplitude", math.nan)), spec.post_rms_amplitude)
            or row.get("local_clip_sweeps") != spec.local_clip_sweeps
            or (row.get("exact") or {}).get("certified") is not True
            or (row.get("exact") or {}).get("status") != "CERTIFIED"
        ):
            raise RuntimeError(f"Invalid C5b arm marker: {case_id}/{spec.arm_id}")
        field = row.get("candidate_field") or {}
        expected_owner = (
            "source_c4_heavy"
            if spec.arm_id == REFERENCE_ARM_ID
            else "source_c5_heavy"
            if spec.arm_id in ANCHOR_ARM_IDS[1:]
            else "target_c5b_heavy"
        )
        if field.get("root_id") != expected_owner:
            raise RuntimeError(f"C5b field has the wrong storage owner: {case_id}/{spec.arm_id}")
        if spec.arm_id in ANCHOR_ARM_IDS:
            source_name = {
                REFERENCE_ARM_ID: "c4_reference_s2_a10_b0",
                ANCHOR_ARM_IDS[1]: "c5_s4_a10_b0_sweep1",
                ANCHOR_ARM_IDS[2]: "c5_s4_a20_b0_sweep1",
            }[spec.arm_id]
            if field != decision["source_anchors"][case_id][source_name]["field"]:
                raise RuntimeError(f"C5b anchor field differs from its authenticated source: {case_id}/{spec.arm_id}")
        elif field.get("relative_path") != f"cases/{case_id}/arms/{spec.arm_id}.npz":
            raise RuntimeError(f"C5b target field path changed: {case_id}/{spec.arm_id}")
        identity = (str(field.get("root_id")), str(field.get("relative_path")))
        if identity in identities:
            raise RuntimeError(f"Duplicate C5b field identity: {case_id}/{spec.arm_id}")
        identities.add(identity)
        if verify_heavy_bytes:
            verify_field_record(field, roots, f"{case_id}/{spec.arm_id}")
        exact = row["exact"]
        if exact.get("sha256") != field.get("array_sha256") or exact.get("epsilon_decimal") != "0.001":
            raise RuntimeError(f"C5b exact certificate differs from field bytes: {case_id}/{spec.arm_id}")
        geometry = validate_c5b_geometry_bundle(row.get("geometry"), f"{case_id}/{spec.arm_id}")
        if row.get("observed_fold_count") != geometry.corner_union_violation_count:
            raise RuntimeError(f"C5b corner-union fold witness changed: {case_id}/{spec.arm_id}")
        if spec.arm_id in ANCHOR_ARM_IDS[1:]:
            parity = row.get("source_parity") or {}
            if (
                parity.get("array_byte_identical") is not True
                or parity.get("source_array_sha256") != field.get("array_sha256")
                or parity.get("replayed_array_sha256") != field.get("array_sha256")
            ):
                raise RuntimeError(f"C5b endpoint replay differs from C5: {case_id}/{spec.arm_id}")
        elif spec.arm_id not in ANCHOR_ARM_IDS and row.get("source_parity") is not None:
            raise RuntimeError(f"C5b generated arm claims source parity: {case_id}/{spec.arm_id}")
        if spec.stride_voxels == 4:
            proposal = row.get("proposal") or {}
            if proposal.get("amplitude_stage") != "after_rms_match_before_local_clip":
                raise RuntimeError(f"C5b amplitude stage changed: {case_id}/{spec.arm_id}")
            rms_source = _finite(proposal.get("rms_source"), f"{case_id}/{spec.arm_id} source RMS")
            rms_target = _finite(proposal.get("rms_target"), f"{case_id}/{spec.arm_id} target RMS")
            rms_matched = _finite(proposal.get("rms_matched"), f"{case_id}/{spec.arm_id} matched RMS")
            rms_scale = _finite(proposal.get("rms_scale_factor"), f"{case_id}/{spec.arm_id} RMS scale")
            requested = _finite(proposal.get("rms_requested"), f"{case_id}/{spec.arm_id} requested RMS")
            realized = _finite(proposal.get("rms_realized"), f"{case_id}/{spec.arm_id} realized RMS")
            retention = _finite(proposal.get("clip_rms_retention"), f"{case_id}/{spec.arm_id} clipping retention")
            cosine = _finite(proposal.get("clip_cosine"), f"{case_id}/{spec.arm_id} clipping cosine")
            if (
                proposal.get("preclip_direction_array_sha256") != direction_sha
                or proposal.get("postprocessed_direction_array_sha256") != postprocessed_sha
                or not math.isclose(float(proposal.get("post_rms_amplitude", math.nan)), spec.post_rms_amplitude)
                or proposal.get("local_clip_sweeps") != spec.local_clip_sweeps
                or min(rms_source, rms_target, rms_matched, rms_scale, requested) <= 0.0
                or realized < 0.0
                or not math.isclose(rms_matched, rms_target, rel_tol=1e-7, abs_tol=1e-8)
                or not math.isclose(requested, rms_target * spec.post_rms_amplitude, rel_tol=1e-7, abs_tol=1e-8)
                or not math.isclose(retention, min(1.0, realized / requested), rel_tol=1e-7, abs_tol=1e-8)
                or not 0.0 <= retention <= 1.0
                or not -1.0 <= cosine <= 1.0
            ):
                raise RuntimeError(f"C5b S4 arms do not share one frozen direction: {case_id}/{spec.arm_id}")
            shared = (rms_source, rms_target, rms_matched, rms_scale)
            if shared_s4_proposal is None:
                shared_s4_proposal = shared
            elif any(
                not math.isclose(left, right, rel_tol=0.0, abs_tol=1e-12)
                for left, right in zip(shared_s4_proposal, shared, strict=True)
            ):
                raise RuntimeError(f"C5b S4 postprocessing differs across amplitudes: {case_id}/{spec.arm_id}")
            operator = proposal.get("operator") or {}
            current_bound = _finite(
                operator.get("current_fast_cert_bound"), f"{case_id}/{spec.arm_id} current certificate"
            )
            output_bound = _finite(
                operator.get("output_fast_cert_bound"), f"{case_id}/{spec.arm_id} output certificate"
            )
            if (
                operator.get("operator") != "CERTIFIED_LOCAL_CLIP"
                or operator.get("sweeps") != spec.local_clip_sweeps
                or not math.isclose(float(operator.get("work_eps", math.nan)), 0.0011)
                or current_bound < 0.0011
                or output_bound < 0.0011
            ):
                raise RuntimeError(f"C5b clip operator changed: {case_id}/{spec.arm_id}")
        if spec.arm_id in (*SELECTABLE_ARM_IDS, DIAGNOSTIC_ARM_ID):
            proposal = row.get("proposal") or {}
            for key in ("clip_rms_retention", "rms_requested", "rms_realized"):
                value = proposal.get(key)
                if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                    raise RuntimeError(f"Invalid C5b proposal scalar: {case_id}/{spec.arm_id}/{key}")
            if not 0.0 <= float(proposal["clip_rms_retention"]) <= 1.0:
                raise RuntimeError(f"Invalid C5b amplitude retention: {case_id}/{spec.arm_id}")
        if spec.arm_id == REFERENCE_ARM_ID and row.get("proposal") not in ({}, None):
            raise RuntimeError(f"C5b C4 reference unexpectedly carries an S4 proposal: {case_id}")
    _validate_execution(
        payload.get("execution"),
        phase="decision",
        shard_index=shard,
        physical_gpu=decision["shard_to_physical_gpu"][str(shard)],
        labels_loaded_to_device=False,
    )
    _assert_decision_payload_label_free(payload)


def validate_worker_marker(
    payload: Mapping[str, Any],
    decision: Mapping[str, Any],
    decision_sha256: str,
    *,
    phase: str,
    shard_index: int,
    attempt_id: str,
    barrier_sha256: str | None = None,
    evaluation_contract_sha256: str | None = None,
) -> None:
    if (
        payload.get("schema") != WORKER_SCHEMA
        or payload.get("protocol_id") != PROTOCOL_ID
        or payload.get("status") != "COMPLETE"
        or payload.get("strict") is not True
        or payload.get("phase") != phase
        or payload.get("attempt_id") != attempt_id
        or payload.get("shard_index") != shard_index
        or str(payload.get("physical_gpu")) != decision["shard_to_physical_gpu"][str(shard_index)]
        or payload.get("case_ids") != decision["shards"][str(shard_index)]
        or payload.get("decision_contract_sha256") != decision_sha256
        or payload.get("test_split_accessed") is not False
    ):
        raise RuntimeError(f"Invalid C5b {phase} worker marker: shard {shard_index}")
    case_sha256 = payload.get("case_sha256")
    if not isinstance(case_sha256, Mapping) or set(case_sha256) != set(decision["shards"][str(shard_index)]):
        raise RuntimeError(f"Invalid C5b {phase} worker case inventory: shard {shard_index}")
    for case_id, digest in case_sha256.items():
        _require_sha(digest, f"C5b {phase} worker case {case_id}")
    _validate_execution(
        payload.get("execution"),
        phase=phase,
        shard_index=shard_index,
        physical_gpu=decision["shard_to_physical_gpu"][str(shard_index)],
        labels_loaded_to_device=phase == "evaluation",
    )
    if phase == "decision":
        if payload.get("labels_loaded") is not False:
            raise RuntimeError("C5b decision worker received labels")
        _assert_decision_payload_label_free(payload)
    if phase == "evaluation" and (
        payload.get("labels_loaded") is not True
        or payload.get("decision_barrier_sha256") != barrier_sha256
        or payload.get("evaluation_contract_sha256") != evaluation_contract_sha256
    ):
        raise RuntimeError("C5b evaluation worker is not bound to the frozen barrier")


def build_decision_barrier(
    *,
    run_root: Path,
    decision: Mapping[str, Any],
    decision_sha256: str,
    attempt_id: str,
    completed_at_utc: str,
) -> tuple[dict[str, Any], str]:
    cases: dict[str, str] = {}
    for case_id in decision["case_ids"]:
        path = decision_case_path(run_root, case_id)
        payload = load_json(path)
        validate_decision_case_marker(payload, decision, decision_sha256, verify_heavy_bytes=True)
        cases[case_id] = sha256_file(path)
    workers = []
    for shard_index in range(decision["num_shards"]):
        path = worker_path(run_root, "decision", attempt_id, shard_index)
        payload = load_json(path)
        validate_worker_marker(
            payload,
            decision,
            decision_sha256,
            phase="decision",
            shard_index=shard_index,
            attempt_id=attempt_id,
        )
        if payload["case_sha256"] != {case_id: cases[case_id] for case_id in decision["shards"][str(shard_index)]}:
            raise RuntimeError(f"C5b decision worker case hashes changed: shard {shard_index}")
        workers.append({"relative_path": path.relative_to(run_root.resolve()).as_posix(), "sha256": sha256_file(path)})
    barrier = {
        "schema": BARRIER_SCHEMA,
        "protocol_id": PROTOCOL_ID,
        "status": "COMPLETE",
        "decision_contract_sha256": decision_sha256,
        "attempt_id": attempt_id,
        "case_ids": list(decision["case_ids"]),
        "decision_case_sha256": cases,
        "workers": workers,
        "decision_workers_received_label_inputs": False,
        "test_split_accessed": False,
        "completed_at_utc": completed_at_utc,
    }
    digest = _immutable_json(run_root.resolve() / "decision_barrier.json", barrier)
    return barrier, digest


def load_barrier(
    run_root: Path,
    barrier_sha256: str,
    decision: Mapping[str, Any],
    decision_sha256: str,
) -> dict[str, Any]:
    path = run_root.resolve() / "decision_barrier.json"
    if sha256_file(path) != _require_sha(barrier_sha256, "C5b barrier"):
        raise RuntimeError("C5b barrier bytes changed")
    barrier = load_json(path)
    if (
        barrier.get("schema") != BARRIER_SCHEMA
        or barrier.get("protocol_id") != PROTOCOL_ID
        or barrier.get("status") != "COMPLETE"
        or barrier.get("decision_contract_sha256") != decision_sha256
        or not isinstance(barrier.get("attempt_id"), str)
        or not barrier.get("attempt_id")
        or barrier.get("case_ids") != decision["case_ids"]
        or set(barrier.get("decision_case_sha256") or {}) != set(decision["case_ids"])
        or not isinstance(barrier.get("workers"), list)
        or len(barrier.get("workers")) != decision["num_shards"]
        or barrier.get("decision_workers_received_label_inputs") is not False
        or barrier.get("test_split_accessed") is not False
    ):
        raise RuntimeError("Invalid or altered C5b decision barrier")
    for case_id, expected in barrier["decision_case_sha256"].items():
        path = decision_case_path(run_root, case_id)
        if sha256_file(path) != expected:
            raise RuntimeError(f"C5b frozen decision case changed: {case_id}")
        validate_decision_case_marker(load_json(path), decision, decision_sha256, verify_heavy_bytes=False)
    expected_worker_rows = []
    for shard_index in range(decision["num_shards"]):
        worker = worker_path(run_root, "decision", barrier["attempt_id"], shard_index)
        payload = load_json(worker)
        validate_worker_marker(
            payload,
            decision,
            decision_sha256,
            phase="decision",
            shard_index=shard_index,
            attempt_id=barrier["attempt_id"],
        )
        expected_case_hashes = {
            case_id: barrier["decision_case_sha256"][case_id] for case_id in decision["shards"][str(shard_index)]
        }
        if payload["case_sha256"] != expected_case_hashes:
            raise RuntimeError(f"C5b frozen decision worker cases changed: shard {shard_index}")
        expected_worker_rows.append(
            {"relative_path": worker.relative_to(run_root.resolve()).as_posix(), "sha256": sha256_file(worker)}
        )
    if barrier["workers"] != expected_worker_rows:
        raise RuntimeError("C5b frozen decision worker inventory changed")
    return barrier


def freeze_evaluation_contract(
    *,
    run_root: Path,
    source: Mapping[str, Any],
    source_sha256: str,
    decision: Mapping[str, Any],
    decision_sha256: str,
    barrier: Mapping[str, Any],
    barrier_sha256: str,
) -> tuple[dict[str, Any], str]:
    contract = {
        "schema": EVALUATION_CONTRACT_SCHEMA,
        "protocol_id": PROTOCOL_ID,
        "status": "FROZEN_AFTER_DECISION_BARRIER",
        "source_contract_sha256": source_sha256,
        "decision_contract_sha256": decision_sha256,
        "decision_barrier_sha256": barrier_sha256,
        "decision_case_sha256": dict(barrier["decision_case_sha256"]),
        "git_head": decision["git_head"],
        "runtime_signature": decision["runtime_signature"],
        "case_ids": list(decision["case_ids"]),
        "raw_inputs": source["raw_inputs"],
        "frozen_evaluation": source["frozen_evaluation"],
        "evaluation_label_ids": list(EVALUATION_LABEL_IDS),
        "test_115_authorized": False,
        "test_split_accessed": False,
    }
    digest = _immutable_json(run_root.resolve() / "evaluation_contract.json", contract)
    return contract, digest


def load_evaluation_contract(
    run_root: Path,
    evaluation_sha256: str,
    source: Mapping[str, Any],
    source_sha256: str,
    decision: Mapping[str, Any],
    decision_sha256: str,
    barrier_sha256: str,
) -> dict[str, Any]:
    path = run_root.resolve() / "evaluation_contract.json"
    if sha256_file(path) != _require_sha(evaluation_sha256, "C5b evaluation contract"):
        raise RuntimeError("C5b evaluation contract bytes changed")
    payload = load_json(path)
    if (
        payload.get("schema") != EVALUATION_CONTRACT_SCHEMA
        or payload.get("protocol_id") != PROTOCOL_ID
        or payload.get("status") != "FROZEN_AFTER_DECISION_BARRIER"
        or payload.get("source_contract_sha256") != source_sha256
        or payload.get("decision_contract_sha256") != decision_sha256
        or payload.get("decision_barrier_sha256") != barrier_sha256
        or payload.get("decision_case_sha256")
        != load_barrier(run_root, barrier_sha256, decision, decision_sha256)["decision_case_sha256"]
        or payload.get("case_ids") != decision["case_ids"]
        or payload.get("git_head") != source.get("git_head")
        or payload.get("git_head") != decision.get("git_head")
        or payload.get("runtime_signature") != source.get("runtime_signature")
        or payload.get("runtime_signature") != decision.get("runtime_signature")
        or payload.get("raw_inputs") != source.get("raw_inputs")
        or payload.get("frozen_evaluation") != source.get("frozen_evaluation")
        or tuple(payload.get("evaluation_label_ids") or ()) != EVALUATION_LABEL_IDS
        or payload.get("test_115_authorized") is not False
        or payload.get("test_split_accessed") is not False
    ):
        raise RuntimeError("Invalid C5b evaluation contract")
    return payload


def validate_evaluation_case_marker(
    payload: Mapping[str, Any],
    decision: Mapping[str, Any],
    decision_sha256: str,
    barrier: Mapping[str, Any],
    barrier_sha256: str,
    evaluation: Mapping[str, Any],
    evaluation_sha256: str,
) -> None:
    case_id = payload.get("case_id")
    if case_id not in decision["case_ids"]:
        raise RuntimeError("Invalid C5b evaluation-case identity")
    if (
        payload.get("schema") != EVALUATION_CASE_SCHEMA
        or payload.get("protocol_id") != PROTOCOL_ID
        or payload.get("status") != "COMPLETE"
        or payload.get("strict") is not True
        or payload.get("decision_contract_sha256") != decision_sha256
        or payload.get("decision_barrier_sha256") != barrier_sha256
        or payload.get("evaluation_contract_sha256") != evaluation_sha256
        or payload.get("decision_case_sha256") != barrier["decision_case_sha256"][case_id]
        or payload.get("labels_loaded_after_barrier") is not True
        or payload.get("test_split_accessed") is not False
        or tuple(payload.get("labels") or ()) != EVALUATION_LABEL_IDS
    ):
        raise RuntimeError(f"Invalid C5b evaluation marker: {case_id}")
    rows = payload.get("arms")
    if not isinstance(rows, list) or tuple(row.get("arm_id") for row in rows) != tuple(
        spec.arm_id for spec in ARM_SPECS
    ):
        raise RuntimeError(f"C5b evaluation arm inventory changed: {case_id}")
    frozen_case = evaluation["frozen_evaluation"][case_id]
    shard = _expected_shard(decision, str(case_id))
    _validate_execution(
        payload.get("execution"),
        phase="evaluation",
        shard_index=shard,
        physical_gpu=decision["shard_to_physical_gpu"][str(shard)],
        labels_loaded_to_device=True,
    )
    for spec, row in zip(ARM_SPECS, rows, strict=True):
        if row.get("arm_index") != spec.arm_index:
            raise RuntimeError(f"C5b evaluation arm index changed: {case_id}/{spec.arm_id}")
        per_label = row.get("per_label")
        if not isinstance(per_label, list) or tuple(item.get("label") for item in per_label) != EVALUATION_LABEL_IDS:
            raise RuntimeError(f"C5b evaluation label inventory changed: {case_id}/{row.get('arm_id')}")
        for key in ("baseline_dice", "candidate_dice", "dice_delta"):
            value = row.get(key)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                raise RuntimeError(f"Invalid C5b evaluation scalar: {case_id}/{row.get('arm_id')}/{key}")
        baseline = float(row["baseline_dice"])
        candidate = float(row["candidate_dice"])
        delta = float(row["dice_delta"])
        if not 0.0 <= baseline <= 1.0 or not 0.0 <= candidate <= 1.0:
            raise RuntimeError(f"C5b evaluation Dice is outside [0,1]: {case_id}/{row.get('arm_id')}")
        if not math.isclose(candidate - baseline, delta, rel_tol=0.0, abs_tol=1e-12) or not math.isclose(
            baseline, float(frozen_case["baseline_dice"]), rel_tol=0.0, abs_tol=1e-8
        ):
            raise RuntimeError(f"C5b evaluation arithmetic changed: {case_id}/{row.get('arm_id')}")
        baseline_labels = []
        candidate_labels = []
        for index, item in enumerate(per_label):
            left, right, observed = (
                _finite(item.get("baseline_dice"), f"{case_id}/{row.get('arm_id')} baseline label Dice"),
                _finite(item.get("candidate_dice"), f"{case_id}/{row.get('arm_id')} candidate label Dice"),
                _finite(item.get("dice_delta"), f"{case_id}/{row.get('arm_id')} label Dice delta"),
            )
            if not 0.0 <= left <= 1.0 or not 0.0 <= right <= 1.0:
                raise RuntimeError(f"C5b per-label Dice is outside [0,1]: {case_id}/{row.get('arm_id')}")
            if not math.isclose(right - left, observed, rel_tol=0.0, abs_tol=1e-12):
                raise RuntimeError(f"C5b per-label arithmetic changed: {case_id}/{row.get('arm_id')}")
            baseline_labels.append(left)
            candidate_labels.append(right)
            if not math.isclose(
                left,
                float(frozen_case["baseline_per_label"][index]["dice"]),
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                raise RuntimeError(f"C5b baseline per-label Dice changed: {case_id}/{row.get('arm_id')}")
        if not math.isclose(
            sum(baseline_labels) / len(baseline_labels), baseline, rel_tol=0.0, abs_tol=1e-12
        ) or not math.isclose(sum(candidate_labels) / len(candidate_labels), candidate, rel_tol=0.0, abs_tol=1e-12):
            raise RuntimeError(f"C5b aggregate Dice does not match per-label rows: {case_id}/{row.get('arm_id')}")
        arm_id = row.get("arm_id")
        if arm_id in ANCHOR_ARM_IDS:
            expected = frozen_case["anchors"][arm_id]
            expected_labels = [float(item["candidate_dice"]) for item in expected["per_label"]]
            if (
                row.get("source_evaluation_parity_verified") is not True
                or not math.isclose(candidate, float(expected["candidate_dice"]), rel_tol=0.0, abs_tol=1e-12)
                or any(
                    not math.isclose(left, right, rel_tol=0.0, abs_tol=1e-12)
                    for left, right in zip(candidate_labels, expected_labels, strict=True)
                )
            ):
                raise RuntimeError(f"C5b evaluation anchor parity changed: {case_id}/{arm_id}")
        elif row.get("source_evaluation_parity_verified") is not None:
            raise RuntimeError(f"C5b non-anchor reports source evaluation parity: {case_id}/{arm_id}")


def build_evaluation_barrier(
    *,
    run_root: Path,
    decision: Mapping[str, Any],
    decision_sha256: str,
    decision_barrier: Mapping[str, Any],
    decision_barrier_sha256: str,
    evaluation: Mapping[str, Any],
    evaluation_sha256: str,
    attempt_id: str,
    completed_at_utc: str,
) -> tuple[dict[str, Any], str]:
    observed_cases: dict[str, str] = {}
    workers = []
    for shard_index in range(decision["num_shards"]):
        path = worker_path(run_root, "evaluation", attempt_id, shard_index)
        worker = load_json(path)
        validate_worker_marker(
            worker,
            decision,
            decision_sha256,
            phase="evaluation",
            shard_index=shard_index,
            attempt_id=attempt_id,
            barrier_sha256=decision_barrier_sha256,
            evaluation_contract_sha256=evaluation_sha256,
        )
        observed: dict[str, str] = {}
        for case_id in decision["shards"][str(shard_index)]:
            case_path = evaluation_case_path(run_root, case_id)
            payload = load_json(case_path)
            validate_evaluation_case_marker(
                payload,
                decision,
                decision_sha256,
                decision_barrier,
                decision_barrier_sha256,
                evaluation,
                evaluation_sha256,
            )
            observed[case_id] = sha256_file(case_path)
        if worker["case_sha256"] != observed:
            raise RuntimeError(f"C5b evaluation worker case hashes changed: shard {shard_index}")
        if set(observed_cases).intersection(observed):
            raise RuntimeError("C5b evaluation case was reported by multiple workers")
        observed_cases.update(observed)
        workers.append({"relative_path": path.relative_to(run_root.resolve()).as_posix(), "sha256": sha256_file(path)})
    if set(observed_cases) != set(decision["case_ids"]):
        raise RuntimeError("C5b evaluation barrier case inventory changed")
    cases = {case_id: observed_cases[case_id] for case_id in decision["case_ids"]}
    barrier = {
        "schema": EVALUATION_BARRIER_SCHEMA,
        "protocol_id": PROTOCOL_ID,
        "status": "COMPLETE",
        "decision_contract_sha256": decision_sha256,
        "decision_barrier_sha256": decision_barrier_sha256,
        "evaluation_contract_sha256": evaluation_sha256,
        "attempt_id": attempt_id,
        "case_ids": list(decision["case_ids"]),
        "evaluation_case_sha256": cases,
        "workers": workers,
        "test_split_accessed": False,
        "completed_at_utc": completed_at_utc,
    }
    digest = _immutable_json(run_root.resolve() / "evaluation_barrier.json", barrier)
    return barrier, digest


def load_evaluation_barrier(
    run_root: Path,
    evaluation_barrier_sha256: str,
    decision: Mapping[str, Any],
    decision_sha256: str,
    decision_barrier: Mapping[str, Any],
    decision_barrier_sha256: str,
    evaluation: Mapping[str, Any],
    evaluation_sha256: str,
) -> dict[str, Any]:
    path = run_root.resolve() / "evaluation_barrier.json"
    if sha256_file(path) != _require_sha(evaluation_barrier_sha256, "C5b evaluation barrier"):
        raise RuntimeError("C5b evaluation barrier bytes changed")
    barrier = load_json(path)
    if (
        barrier.get("schema") != EVALUATION_BARRIER_SCHEMA
        or barrier.get("protocol_id") != PROTOCOL_ID
        or barrier.get("status") != "COMPLETE"
        or barrier.get("decision_contract_sha256") != decision_sha256
        or barrier.get("decision_barrier_sha256") != decision_barrier_sha256
        or barrier.get("evaluation_contract_sha256") != evaluation_sha256
        or not isinstance(barrier.get("attempt_id"), str)
        or not barrier.get("attempt_id")
        or barrier.get("case_ids") != decision["case_ids"]
        or set(barrier.get("evaluation_case_sha256") or {}) != set(decision["case_ids"])
        or not isinstance(barrier.get("workers"), list)
        or len(barrier["workers"]) != decision["num_shards"]
        or barrier.get("test_split_accessed") is not False
    ):
        raise RuntimeError("Invalid or altered C5b evaluation barrier")
    expected_workers = []
    for shard_index in range(decision["num_shards"]):
        worker_path_value = worker_path(run_root, "evaluation", barrier["attempt_id"], shard_index)
        worker = load_json(worker_path_value)
        validate_worker_marker(
            worker,
            decision,
            decision_sha256,
            phase="evaluation",
            shard_index=shard_index,
            attempt_id=barrier["attempt_id"],
            barrier_sha256=decision_barrier_sha256,
            evaluation_contract_sha256=evaluation_sha256,
        )
        expected_case_hashes = {
            case_id: barrier["evaluation_case_sha256"][case_id] for case_id in decision["shards"][str(shard_index)]
        }
        if worker["case_sha256"] != expected_case_hashes:
            raise RuntimeError(f"C5b frozen evaluation worker cases changed: shard {shard_index}")
        expected_workers.append(
            {
                "relative_path": worker_path_value.relative_to(run_root.resolve()).as_posix(),
                "sha256": sha256_file(worker_path_value),
            }
        )
    if barrier["workers"] != expected_workers:
        raise RuntimeError("C5b frozen evaluation worker inventory changed")
    for case_id, expected_sha in barrier["evaluation_case_sha256"].items():
        case_path = evaluation_case_path(run_root, case_id)
        if sha256_file(case_path) != _require_sha(expected_sha, f"C5b evaluation case {case_id}"):
            raise RuntimeError(f"C5b frozen evaluation case changed: {case_id}")
        validate_evaluation_case_marker(
            load_json(case_path),
            decision,
            decision_sha256,
            decision_barrier,
            decision_barrier_sha256,
            evaluation,
            evaluation_sha256,
        )
    return barrier


__all__ = [
    "BARRIER_SCHEMA",
    "DECISION_CASE_SCHEMA",
    "DECISION_SCHEMA",
    "EVALUATION_BARRIER_SCHEMA",
    "EVALUATION_CASE_SCHEMA",
    "EVALUATION_CONTRACT_SCHEMA",
    "SOURCE_SCHEMA",
    "WORKER_SCHEMA",
    "build_decision_barrier",
    "build_evaluation_barrier",
    "decision_case_path",
    "evaluation_case_path",
    "freeze_evaluation_contract",
    "load_barrier",
    "load_contracts",
    "load_decision_contract_isolated",
    "load_evaluation_barrier",
    "load_evaluation_contract",
    "load_json",
    "prepare_contracts",
    "validate_decision_case_marker",
    "validate_evaluation_case_marker",
    "validate_worker_marker",
    "verify_field_record",
    "verify_image_record",
    "worker_path",
]
