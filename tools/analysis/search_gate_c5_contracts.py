from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np

from tools.analysis.run_artifacts import atomic_write_json, sha256_file
from tools.analysis.run_search_gate_c4 import (
    ARM_SPECS_SHA256 as C4_ARM_SPECS_SHA256,
    OFFSET_TABLE_SHA256 as C4_OFFSET_TABLE_SHA256,
    SUPPORT_CONTRACT_SHA256 as C4_SUPPORT_CONTRACT_SHA256,
)
from tools.analysis.search_gate_c4 import C4_POLICY_SHA256
from tools.analysis.search_gate_c4_contracts import (
    load_decision_barrier as load_c4_decision_barrier,
    load_decision_contract as load_c4_decision_contract,
    load_source_contract as load_c4_source_contract,
    payload_sha256,
    validate_decision_case_marker as validate_c4_decision_case_marker,
)
from tools.analysis.search_gate_c5 import (
    C5_DECISION_POLICY_SHA256,
    C5_POLICY_SHA256,
    HISTORICAL_ANCHOR_ARM_IDS,
    SELECTABLE_ARM_IDS,
    SELECTOR_IDS,
    CandidateSignals,
    choose_global_candidate,
    decision_policy_contract,
)
from tools.analysis.search_gate_metrics import MATHEMATICAL_SDLOGJ_CROP2

PROTOCOL_ID = "CTCF-SEARCH-GATE-C5-V1"
SCHEMA_VERSION = "v1"

SOURCE_SCHEMA = "ctcf-search-c5-source-contract-v1"
DECISION_SCHEMA = "ctcf-search-c5-decision-contract-v1"
BARRIER_SCHEMA = "ctcf-search-c5-decision-barrier-v1"
EVALUATION_CONTRACT_SCHEMA = "ctcf-search-c5-evaluation-contract-v1"
DECISION_CASE_SCHEMA = "ctcf-search-c5-decision-case-v1"
EVALUATION_CASE_SCHEMA = "ctcf-search-c5-evaluation-case-v1"
WORKER_SCHEMA = "ctcf-search-c5-worker-v1"

SOURCE_C4_RUN_ID = "C4_DEVELOPMENT_20260824T161239Z_c69d12000176"
SOURCE_C4_MANIFEST_SHA256 = "dd3c6a6eb4d0fdcb479b7dee4d722c4573cd68a59ac3849a0482610a7721f230"
SOURCE_C4_RUN_MANIFEST_SHA256 = "70d6f526501a26b442ad1d3b821028a7d5ed9ef5b6c809f7e7fdebe2cc15460e"
SOURCE_C4_GIT_HEAD = "c69d12000176c796e96dd15f4b831c8df05ef64b"
SOURCE_C4_ANCHOR_IDS = ("intensity_s1", "intensity_s2")
EVALUATION_LABEL_IDS = (
    1,
    2,
    3,
    5,
    6,
    7,
    8,
    9,
    10,
    11,
    12,
    13,
    14,
    15,
    16,
    18,
    20,
    21,
    22,
    23,
    25,
    26,
    27,
    28,
    29,
    30,
    31,
    32,
    34,
    36,
)
NGF_DIAGNOSTIC_CONTRACT = {
    "diagnostic_id": "COMMON_NGF_CENTRAL3_FLOAT64_V1",
    "dtype": "float64",
    "derivative": "central_difference_kernel_[-0.5,0,0.5]_per_zyx_axis_after_warp",
    "coordinate_units": "voxel_index",
    "support": "erode(C4_common_mask,1) ∩ erode(initial_valid,1) ∩ erode(candidate_valid,1); eta uses fixed baseline support",
    "eta_formula": "0.1*sqrt(mean((norm2(grad_fixed)+norm2(grad_baseline_warp))/2))",
    "similarity_formula": "mean(((dot(grad_fixed,grad_warped)+eta^2)^2)/((norm2(grad_fixed)+eta^2)*(norm2(grad_warped)+eta^2)))",
    "reduction": "float64_arithmetic_mean_on_candidate_pair_ngf_support",
    "improvement": "candidate_similarity_minus_baseline_similarity_higher_is_better",
    "selector_eligible": False,
}
EXPECTED_SUPPORT_CONTRACT = {
    "schema": "ctcf-search-c5-support-contract-v1",
    "geometry": {
        "generation_support_id": "C4_COMMON_DESCRIPTOR_SUPPORT_RETAINED_V1",
        "generation_mask": "frozen_C4_common_mask",
        "rms_mask": "C4_collar7_geometry_mask",
        "collar_width": 7,
        "all_reach_candidate_valid_required": True,
        "all_reach_intersection_diagnostic_only": True,
    },
    "normalization": {
        "mode": "independent_masked_zscore",
        "mask": "C4_collar7_geometry_mask",
        "std_floor": 1e-6,
    },
    "selector_support": {
        "retention": "min(ncc7.retention,mind_d2.retention)",
        "minimum": 0.99,
    },
    "utilities": {
        **{
            f"ncc{window}": {
                "utility_id": f"COMMON_NCC{window}",
                "window": window,
                "eps": 1e-5,
                "support": "pairwise_common_valid_eroded_by_window_radius",
                "reduction": "fp64_mean_local_ncc_loss",
                "improvement": "baseline_loss_minus_candidate_loss",
                "selector_eligible": window == 7,
            }
            for window in (5, 7, 9)
        },
        "mind_d2": {
            "utility_id": "COMMON_MIND_D2",
            "radius": 1,
            "dilation": 2,
            "support_window": 1,
            "support": "pairwise_common_valid",
            "reduction": "fp64_mean_channel_mean_squared_descriptor_distance",
            "improvement": "baseline_loss_minus_candidate_loss",
            "selector_eligible": True,
        },
        "ngf": NGF_DIAGNOSTIC_CONTRACT,
    },
}

EXPECTED_CASES = 58
ROOT_IDS = ("source_c3_heavy", "source_c4_heavy", "target_c5_heavy")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
GIT_SHA_RE = re.compile(r"^[0-9a-f]{40}$")
_LABEL_ALLOWED_KEYS = {
    "decision_contract_contains_label_data",
    "labels_available_to_decision_workers",
    "labels_loaded_to_device",
    "labels_accessible",
}


@dataclass(frozen=True)
class ContractBundle:
    source: dict[str, Any]
    source_sha256: str
    decision: dict[str, Any]
    decision_sha256: str


def array_sha256(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    return hashlib.sha256(contiguous.tobytes(order="C")).hexdigest()


def _require_sha(value: Any, label: str) -> str:
    text = str(value)
    if SHA256_RE.fullmatch(text) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return text


def _finite(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise ValueError(f"{label} must be finite")
    return float(value)


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return payload


def _write_immutable_json(path: Path, payload: dict[str, Any]) -> str:
    if path.exists():
        observed = _load_json(path)
        if observed != payload:
            raise FileExistsError(f"Refusing to replace immutable contract: {path}")
        return sha256_file(path)
    atomic_write_json(path, payload)
    return sha256_file(path)


def _case_ids(value: Any, label: str) -> list[str]:
    if (
        not isinstance(value, list)
        or len(value) != EXPECTED_CASES
        or any(not isinstance(item, str) or not item for item in value)
        or len(set(value)) != len(value)
    ):
        raise RuntimeError(f"{label} must contain exactly {EXPECTED_CASES} unique ordered case IDs")
    return list(value)


def _assert_label_free(value: Any, label: str, path: tuple[str, ...] = ()) -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            token = str(key).lower()
            child_path = (*path, str(key))
            if "dice" in token or "segmentation" in token:
                raise RuntimeError(f"{label} contains forbidden evaluation key: {'.'.join(child_path)}")
            if "label" in token and token not in _LABEL_ALLOWED_KEYS and not token.startswith("label_free"):
                raise RuntimeError(f"{label} contains forbidden label key: {'.'.join(child_path)}")
            _assert_label_free(child, label, child_path)
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _assert_label_free(child, label, (*path, str(index)))
    elif isinstance(value, str):
        lowered = value.lower()
        if "segmentation" in lowered or lowered.endswith(".pkl"):
            raise RuntimeError(f"{label} contains forbidden evaluation value: {'.'.join(path)}")


def _root_map(value: Any) -> dict[str, Path]:
    if not isinstance(value, Mapping) or set(value) != set(ROOT_IDS):
        raise RuntimeError("C5 contract must bind exactly the three declared storage roots")
    roots = {key: Path(str(value[key])).resolve() for key in ROOT_IDS}
    if len(set(roots.values())) != len(roots):
        raise RuntimeError("C5 storage roots must be distinct")
    if any(
        left == right or left in right.parents or right in left.parents
        for i, left in enumerate(roots.values())
        for right in list(roots.values())[i + 1 :]
    ):
        raise RuntimeError("C5 storage roots must not overlap")
    return roots


def _relative_path(value: Any, label: str) -> PurePosixPath:
    if not isinstance(value, str) or not value or "\\" in value:
        raise RuntimeError(f"{label} must be a non-empty POSIX relative path")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise RuntimeError(f"{label} escapes its declared root")
    return path


def verify_rooted_record(
    record: Mapping[str, Any],
    roots: Mapping[str, Path | str],
    *,
    verify_bytes: bool,
    verify_array: bool = False,
) -> Path:
    root_id = record.get("root_id")
    if root_id not in ROOT_IDS or root_id not in roots:
        raise RuntimeError("C5 artifact record has an unknown root_id")
    relative = _relative_path(record.get("relative_path"), "C5 artifact relative_path")
    root = Path(roots[root_id]).resolve()
    path = root.joinpath(*relative.parts).resolve()
    if root not in path.parents:
        raise RuntimeError("C5 artifact path escaped its declared root")
    file_sha = _require_sha(record.get("sha256", record.get("npz_sha256")), "C5 artifact file")
    if verify_bytes and (not path.is_file() or sha256_file(path) != file_sha):
        raise RuntimeError(f"C5 artifact bytes changed: {path}")
    if verify_array:
        expected = _require_sha(record.get("array_sha256"), "C5 artifact array")
        if not path.is_file():
            raise RuntimeError(f"C5 array artifact is absent: {path}")
        with np.load(path, allow_pickle=False) as archive:
            if archive.files != ["flow"] or array_sha256(np.asarray(archive["flow"])) != expected:
                raise RuntimeError(f"C5 artifact array changed: {path}")
    return path


def _tag_record(record: Mapping[str, Any], root_id: str, root: Path) -> dict[str, Any]:
    if root_id not in ROOT_IDS:
        raise ValueError(f"Unknown C5 root: {root_id}")
    if "relative_path" in record:
        relative = _relative_path(record["relative_path"], "source relative_path").as_posix()
    else:
        path = Path(str(record.get("path", ""))).resolve()
        try:
            relative = path.relative_to(root.resolve()).as_posix()
        except ValueError as exc:
            raise RuntimeError("Frozen source record is outside its declared heavy root") from exc
    result = {"root_id": root_id, "relative_path": relative}
    if "npz_sha256" in record:
        result["npz_sha256"] = _require_sha(record["npz_sha256"], "source NPZ")
    elif "sha256" in record:
        result["sha256"] = _require_sha(record["sha256"], "source file")
    else:
        raise RuntimeError("Frozen source record lacks a byte SHA-256")
    if "array_sha256" in record:
        result["array_sha256"] = _require_sha(record["array_sha256"], "source array")
    for key in ("dtype", "shape", "bytes"):
        if key in record:
            result[key] = record[key]
    return result


def _build_sharding(case_ids: list[str], physical_gpus: Sequence[str]) -> dict[str, Any]:
    physical = list(physical_gpus)
    if not physical or len(set(physical)) != len(physical) or any(not value.isdigit() for value in physical):
        raise ValueError("C5 physical GPUs must be unique non-negative integer strings")
    shards = {str(index): case_ids[index :: len(physical)] for index in range(len(physical))}
    return {
        "num_shards": len(physical),
        "physical_gpus": physical,
        "shard_to_physical_gpu": {str(index): value for index, value in enumerate(physical)},
        "shards": shards,
    }


def _validate_sharding(contract: Mapping[str, Any]) -> None:
    case_ids = _case_ids(contract.get("case_ids"), "C5 sharding")
    physical = contract.get("physical_gpus")
    count = contract.get("num_shards")
    if (
        not isinstance(physical, list)
        or isinstance(count, bool)
        or not isinstance(count, int)
        or count != len(physical)
    ):
        raise RuntimeError("C5 sharding has invalid GPU metadata")
    expected = _build_sharding(case_ids, physical)
    for key in ("num_shards", "physical_gpus", "shard_to_physical_gpu", "shards"):
        if contract.get(key) != expected[key]:
            raise RuntimeError("C5 shard partition, order, or physical-GPU mapping changed")


def authenticate_frozen_c4(
    compact_dir: Path,
    c4_heavy_root: Path,
    *,
    verify_anchor_bytes: bool = True,
) -> dict[str, Any]:
    compact = compact_dir.resolve()
    c4_heavy = c4_heavy_root.resolve()
    manifest_path = compact / "c4_manifest.json"
    run_manifest_path = compact / "run_manifest.json"
    if sha256_file(manifest_path) != SOURCE_C4_MANIFEST_SHA256:
        raise RuntimeError("C5 requires the exact frozen successful C4 manifest")
    if sha256_file(run_manifest_path) != SOURCE_C4_RUN_MANIFEST_SHA256:
        raise RuntimeError("C5 requires the exact frozen native C4 run manifest")
    manifest = _load_json(manifest_path)
    native = _load_json(run_manifest_path)
    code = native.get("code") or {}
    if (
        manifest.get("schema") != "ctcf-search-c4-run-manifest-v1"
        or manifest.get("protocol_id") != "CTCF-SEARCH-GATE-C4-V1"
        or manifest.get("status") != "COMPLETE"
        or manifest.get("test_115_authorized") is not False
        or manifest.get("test_split_accessed") is not False
        or native.get("schema") != "ctcf-native-manifest-v1"
        or native.get("run_id") != SOURCE_C4_RUN_ID
        or native.get("status") != "COMPLETE"
        or native.get("exit_code") != 0
        or code.get("git_head") != SOURCE_C4_GIT_HEAD
        or code.get("tracked_tree_clean_at_start") is not True
    ):
        raise RuntimeError("Frozen C4 source is incomplete, altered, or not val-58 only")

    files = manifest.get("files") or {}
    for stem, expected in files.items():
        suffix = ".json" if stem in {"hypotheses", "next_branch", "summary"} else ".csv"
        path = compact / f"{stem}{suffix}"
        if not path.is_file() or sha256_file(path) != _require_sha(expected, f"C4 {path.name}"):
            raise RuntimeError(f"Frozen C4 manifest does not authenticate {path}")
    datasets = compact / "datasets.tsv"
    native_files = native.get("files") or {}
    if not datasets.is_file() or sha256_file(datasets) != _require_sha(
        native_files.get("datasets_sha256"), "C4 datasets.tsv"
    ):
        raise RuntimeError("Frozen C4 native manifest does not authenticate datasets.tsv")

    source_sha = sha256_file(compact / "source_contract.json")
    decision_sha = _require_sha(manifest.get("decision_contract_sha256"), "C4 decision contract")
    barrier_sha = _require_sha(manifest.get("decision_barrier_sha256"), "C4 decision barrier")
    source, _ = load_c4_source_contract(compact, source_sha)
    decision, _ = load_c4_decision_contract(
        compact,
        decision_sha,
        source=source,
        expected_source_sha256=source_sha,
        expected_policy_sha256=C4_POLICY_SHA256,
        expected_arm_specs_sha256=C4_ARM_SPECS_SHA256,
        expected_offset_table_sha256=C4_OFFSET_TABLE_SHA256,
        expected_support_contract_sha256=C4_SUPPORT_CONTRACT_SHA256,
    )
    barrier, _ = load_c4_decision_barrier(
        compact,
        barrier_sha,
        decision_contract_sha256=decision_sha,
        case_ids=decision["case_ids"],
    )
    case_ids = _case_ids(decision.get("case_ids"), "Frozen C4")
    if Path(str(decision.get("heavy_root", ""))).resolve() != c4_heavy:
        raise RuntimeError("Frozen C4 decision contract points at another heavy root")
    decision_hashes = manifest.get("decision_case_sha256") or {}
    evaluation_hashes = manifest.get("evaluation_case_sha256") or {}
    if (
        set(decision_hashes) != set(case_ids)
        or set(evaluation_hashes) != set(case_ids)
        or barrier.get("decision_case_sha256") != decision_hashes
    ):
        raise RuntimeError("Frozen C4 case inventories are incomplete or inconsistent")

    c3_heavy = Path(str(decision.get("source_c3_heavy_root", ""))).resolve()
    roots = {
        "source_c3_heavy": c3_heavy,
        "source_c4_heavy": c4_heavy,
        "target_c5_heavy": c4_heavy.parent / "__c5_target_unbound__",
    }
    anchors: dict[str, dict[str, dict[str, Any]]] = {}
    baseline_geometry: dict[str, Any] = {}
    evaluation_c4_anchor_dice: dict[str, dict[str, Any]] = {}
    evaluation_baseline_per_label: dict[str, list[dict[str, Any]]] = {}
    for case_id in case_ids:
        decision_path = compact / "cases" / case_id / "decision_complete.json"
        evaluation_path = compact / "cases" / case_id / "evaluation_complete.json"
        if sha256_file(decision_path) != _require_sha(decision_hashes[case_id], f"C4 {case_id} decision"):
            raise RuntimeError(f"Frozen C4 decision marker changed: {case_id}")
        if sha256_file(evaluation_path) != _require_sha(evaluation_hashes[case_id], f"C4 {case_id} evaluation"):
            raise RuntimeError(f"Frozen C4 evaluation marker changed: {case_id}")
        marker = _load_json(decision_path)
        evaluation = _load_json(evaluation_path)
        validate_c4_decision_case_marker(marker, decision, decision_sha, verify_heavy_bytes=False)
        by_id = {row.get("arm_id"): row for row in marker.get("arms") or []}
        if set(SOURCE_C4_ANCHOR_IDS) - set(by_id):
            raise RuntimeError(f"Frozen C4 intensity anchors are absent: {case_id}")
        case_anchors: dict[str, dict[str, Any]] = {}
        for anchor_id in SOURCE_C4_ANCHOR_IDS:
            row = by_id[anchor_id]
            field = _tag_record(row.get("candidate_field") or {}, "source_c4_heavy", c4_heavy)
            exact = row.get("exact") or {}
            if (
                exact.get("status") != "CERTIFIED"
                or exact.get("certified") is not True
                or exact.get("sha256") != field.get("array_sha256")
            ):
                raise RuntimeError(f"Frozen C4 anchor lacks an exact certificate: {case_id}/{anchor_id}")
            verify_rooted_record(field, roots, verify_bytes=verify_anchor_bytes, verify_array=verify_anchor_bytes)
            case_anchors[anchor_id] = {
                "field": field,
                "source_decision_case_sha256": decision_hashes[case_id],
                "exact_array_sha256": exact["sha256"],
            }
        anchors[case_id] = case_anchors
        baseline_geometry[case_id] = marker.get("baseline_geometry")
        if tuple(evaluation.get("labels") or ()) != EVALUATION_LABEL_IDS:
            raise RuntimeError(f"Frozen C4 evaluation label order changed: {case_id}")
        evaluation_by_id = {row.get("arm_id"): row for row in evaluation.get("arms") or []}
        case_dice: dict[str, Any] = {}
        case_baseline_labels: list[dict[str, Any]] | None = None
        for anchor_id in SOURCE_C4_ANCHOR_IDS:
            row = evaluation_by_id.get(anchor_id) or {}
            per_label = row.get("per_label")
            if (
                not isinstance(per_label, list)
                or tuple(item.get("label") for item in per_label) != EVALUATION_LABEL_IDS
            ):
                raise RuntimeError(f"Frozen C4 per-label anchor Dice changed: {case_id}/{anchor_id}")
            aggregate = _finite(row.get("capacity_candidate_dice"), f"C4 {case_id}/{anchor_id} Dice")
            if not 0.0 <= aggregate <= 1.0:
                raise RuntimeError(f"Frozen C4 anchor Dice is outside [0,1]: {case_id}/{anchor_id}")
            case_dice[anchor_id] = {
                "aggregate_dice": aggregate,
                "per_label": [
                    {
                        "label": item["label"],
                        "dice": _finite(item.get("candidate_dice"), f"C4 {case_id}/{anchor_id} label Dice"),
                    }
                    for item in per_label
                ],
                "source_evaluation_case_sha256": evaluation_hashes[case_id],
            }
            observed_baselines = [
                {
                    "label": item["label"],
                    "dice": _finite(item.get("baseline_dice"), f"C4 {case_id} baseline label Dice"),
                }
                for item in per_label
            ]
            if case_baseline_labels is not None and observed_baselines != case_baseline_labels:
                raise RuntimeError(f"Frozen C4 anchor baselines disagree: {case_id}")
            case_baseline_labels = observed_baselines
        evaluation_c4_anchor_dice[case_id] = case_dice
        if case_baseline_labels is None:
            raise RuntimeError(f"Frozen C4 baseline labels are absent: {case_id}")
        evaluation_baseline_per_label[case_id] = case_baseline_labels

    image_inputs = {
        identity: _tag_record(record, "source_c3_heavy", c3_heavy)
        for identity, record in decision["image_inputs"].items()
    }
    source_initial = {
        case_id: {
            **decision["source_initial"][case_id],
            "field": _tag_record(decision["source_initial"][case_id]["field"], "source_c3_heavy", c3_heavy),
        }
        for case_id in case_ids
    }
    source_historical = {
        case_id: {
            **decision["source_historical"][case_id],
            "raw_conf_requested_field": _tag_record(
                decision["source_historical"][case_id]["raw_conf_requested_field"],
                "source_c3_heavy",
                c3_heavy,
            ),
        }
        for case_id in case_ids
    }
    return {
        "source_c4": {
            "compact_directory": str(compact),
            "heavy_root": str(c4_heavy),
            "run_id": SOURCE_C4_RUN_ID,
            "git_head": SOURCE_C4_GIT_HEAD,
            "manifest_sha256": SOURCE_C4_MANIFEST_SHA256,
            "run_manifest_sha256": SOURCE_C4_RUN_MANIFEST_SHA256,
            "source_contract_sha256": source_sha,
            "decision_contract_sha256": decision_sha,
            "decision_barrier_sha256": barrier_sha,
        },
        "source_c3_heavy_root": str(c3_heavy),
        "raw_inputs": source["raw_inputs"],
        "image_inputs": image_inputs,
        "source_initial": source_initial,
        "source_historical": source_historical,
        "source_c4_anchors": anchors,
        "baseline_geometry": baseline_geometry,
        "evaluation_baseline_dice": source["evaluation_baseline_dice"],
        "evaluation_c4_anchor_dice": evaluation_c4_anchor_dice,
        "evaluation_baseline_per_label": evaluation_baseline_per_label,
        "evaluation_label_ids": list(EVALUATION_LABEL_IDS),
        "case_ids": case_ids,
        "seed": decision["seed"],
        "runtime_signature": decision["runtime_signature"],
    }


def build_source_contract(
    snapshot: Mapping[str, Any],
    *,
    git_head: str,
    runtime_signature: Mapping[str, Any],
    target_heavy_root: Path,
    physical_gpus: Sequence[str],
    full_policy: Mapping[str, Any],
    expected_full_policy_sha256: str,
    contrast_contract: Mapping[str, Any],
    expected_contrast_contract_sha256: str,
) -> dict[str, Any]:
    if GIT_SHA_RE.fullmatch(git_head) is None:
        raise ValueError("C5 git_head must be a full lowercase Git SHA")
    case_ids = _case_ids(snapshot.get("case_ids"), "C5 source snapshot")
    source_c4 = snapshot.get("source_c4") or {}
    roots = {
        "source_c3_heavy": str(Path(str(snapshot["source_c3_heavy_root"])).resolve()),
        "source_c4_heavy": str(Path(str(source_c4["heavy_root"])).resolve()),
        "target_c5_heavy": str(target_heavy_root.resolve()),
    }
    _root_map(roots)
    full_policy_sha = _require_sha(expected_full_policy_sha256, "C5 full policy")
    contrast_sha = _require_sha(expected_contrast_contract_sha256, "C5 contrast contract")
    if full_policy_sha != C5_POLICY_SHA256 or payload_sha256(full_policy) != full_policy_sha:
        raise RuntimeError("C5 full policy differs from its frozen public owner")
    if payload_sha256(contrast_contract) != contrast_sha:
        raise RuntimeError("C5 contrast contract SHA-256 does not match its payload")
    payload = {
        "schema": SOURCE_SCHEMA,
        "protocol_id": PROTOCOL_ID,
        "git_head": git_head,
        "runtime_signature": dict(runtime_signature),
        "full_policy": dict(full_policy),
        "full_policy_sha256": full_policy_sha,
        "contrast_contract": dict(contrast_contract),
        "contrast_contract_sha256": contrast_sha,
        "source_c4": dict(source_c4),
        "roots": roots,
        "raw_inputs": snapshot["raw_inputs"],
        "image_inputs": snapshot["image_inputs"],
        "source_initial": snapshot["source_initial"],
        "source_historical": snapshot["source_historical"],
        "source_c4_anchors": snapshot["source_c4_anchors"],
        "baseline_geometry": snapshot["baseline_geometry"],
        "evaluation_baseline_dice": snapshot["evaluation_baseline_dice"],
        "evaluation_c4_anchor_dice": snapshot["evaluation_c4_anchor_dice"],
        "evaluation_baseline_per_label": snapshot["evaluation_baseline_per_label"],
        "evaluation_label_ids": snapshot["evaluation_label_ids"],
        "case_ids": case_ids,
        "seed": snapshot["seed"],
        "ixi_test_split_accessed": False,
        "test_115_authorized": False,
        **_build_sharding(case_ids, physical_gpus),
    }
    validate_source_contract(payload)
    return payload


def validate_source_contract(payload: Mapping[str, Any]) -> None:
    source = payload.get("source_c4") or {}
    if (
        payload.get("schema") != SOURCE_SCHEMA
        or payload.get("protocol_id") != PROTOCOL_ID
        or GIT_SHA_RE.fullmatch(str(payload.get("git_head", ""))) is None
        or payload.get("ixi_test_split_accessed") is not False
        or payload.get("test_115_authorized") is not False
        or source.get("run_id") != SOURCE_C4_RUN_ID
        or source.get("git_head") != SOURCE_C4_GIT_HEAD
        or source.get("manifest_sha256") != SOURCE_C4_MANIFEST_SHA256
        or source.get("run_manifest_sha256") != SOURCE_C4_RUN_MANIFEST_SHA256
    ):
        raise RuntimeError("Invalid or altered C5 source contract")
    if (
        payload.get("full_policy_sha256") != C5_POLICY_SHA256
        or payload_sha256(payload.get("full_policy")) != C5_POLICY_SHA256
        or payload_sha256(payload.get("contrast_contract"))
        != _require_sha(payload.get("contrast_contract_sha256"), "C5 contrast contract")
    ):
        raise RuntimeError("C5 source policy or contrast contract changed")
    case_ids = _case_ids(payload.get("case_ids"), "C5 source contract")
    roots = _root_map(payload.get("roots"))
    if Path(str(source.get("heavy_root", ""))).resolve() != roots["source_c4_heavy"]:
        raise RuntimeError("C5 source C4 locator and declared root disagree")
    if set(payload.get("raw_inputs") or {}) != {"atlas", *case_ids}:
        raise RuntimeError("C5 source contract has the wrong raw-input inventory")
    if set(payload.get("image_inputs") or {}) != {"atlas", *case_ids}:
        raise RuntimeError("C5 source contract has the wrong image inventory")
    if set(payload.get("source_initial") or {}) != set(case_ids):
        raise RuntimeError("C5 source contract has the wrong initial-field inventory")
    if set(payload.get("source_historical") or {}) != set(case_ids):
        raise RuntimeError("C5 source contract has the wrong historical-field inventory")
    if set(payload.get("source_c4_anchors") or {}) != set(case_ids):
        raise RuntimeError("C5 source contract has the wrong C4-anchor inventory")
    if set(payload.get("baseline_geometry") or {}) != set(case_ids):
        raise RuntimeError("C5 source contract has the wrong baseline-geometry inventory")
    baselines = payload.get("evaluation_baseline_dice") or {}
    if set(baselines) != set(case_ids):
        raise RuntimeError("C5 source contract has the wrong evaluation-baseline inventory")
    for value in baselines.values():
        number = _finite(value, "C5 baseline Dice")
        if not 0.0 <= number <= 1.0:
            raise RuntimeError("C5 baseline Dice must be in [0,1]")
    if tuple(payload.get("evaluation_label_ids") or ()) != EVALUATION_LABEL_IDS:
        raise RuntimeError("C5 source contract has the wrong IXI evaluation labels")
    anchor_dice = payload.get("evaluation_c4_anchor_dice") or {}
    if set(anchor_dice) != set(case_ids):
        raise RuntimeError("C5 source contract has the wrong C4 anchor-Dice inventory")
    baseline_labels = payload.get("evaluation_baseline_per_label") or {}
    if set(baseline_labels) != set(case_ids):
        raise RuntimeError("C5 source contract has the wrong baseline per-label inventory")
    for record in (payload.get("image_inputs") or {}).values():
        if record.get("root_id") != "source_c3_heavy":
            raise RuntimeError("C5 image input has the wrong root owner")
        verify_rooted_record(record, roots, verify_bytes=False)
    for case_id in case_ids:
        initial = (payload["source_initial"][case_id] or {}).get("field") or {}
        if initial.get("root_id") != "source_c3_heavy":
            raise RuntimeError(f"C5 initial field has the wrong root owner: {case_id}")
        verify_rooted_record(initial, roots, verify_bytes=False)
        historical = (payload["source_historical"][case_id] or {}).get("raw_conf_requested_field") or {}
        if historical.get("root_id") != "source_c3_heavy":
            raise RuntimeError(f"C5 RMS-reference field has the wrong root owner: {case_id}")
        verify_rooted_record(historical, roots, verify_bytes=False)
        anchors = payload["source_c4_anchors"][case_id]
        if set(anchors) != set(SOURCE_C4_ANCHOR_IDS):
            raise RuntimeError(f"C5 source does not bind both C4 intensity anchors: {case_id}")
        for anchor_id, anchor in anchors.items():
            field = (anchor or {}).get("field") or {}
            if field.get("root_id") != "source_c4_heavy":
                raise RuntimeError(f"C5 anchor has the wrong root owner: {case_id}/{anchor_id}")
            verify_rooted_record(field, roots, verify_bytes=False)
            if anchor.get("exact_array_sha256") != field.get("array_sha256"):
                raise RuntimeError(f"C5 anchor exact hash disagrees: {case_id}/{anchor_id}")
            dice = (anchor_dice[case_id] or {}).get(anchor_id) or {}
            aggregate = _finite(dice.get("aggregate_dice"), f"C5 {case_id}/{anchor_id} source Dice")
            labels = dice.get("per_label")
            if (
                not 0.0 <= aggregate <= 1.0
                or not isinstance(labels, list)
                or tuple(row.get("label") for row in labels) != EVALUATION_LABEL_IDS
            ):
                raise RuntimeError(f"C5 C4 anchor Dice changed: {case_id}/{anchor_id}")
            if any(not 0.0 <= _finite(row.get("dice"), "C5 source per-label Dice") <= 1.0 for row in labels):
                raise RuntimeError(f"C5 C4 per-label Dice is outside [0,1]: {case_id}/{anchor_id}")
        if tuple(row.get("label") for row in baseline_labels[case_id]) != EVALUATION_LABEL_IDS or any(
            not 0.0 <= _finite(row.get("dice"), "C5 source baseline per-label Dice") <= 1.0
            for row in baseline_labels[case_id]
        ):
            raise RuntimeError(f"C5 baseline per-label Dice changed: {case_id}")
    _validate_sharding(payload)


def _validate_hashed_payload(
    payload: Mapping[str, Any],
    *,
    value_key: str,
    sha_key: str,
    expected_sha256: str,
    label: str,
) -> None:
    expected = _require_sha(expected_sha256, f"C5 {label}")
    if payload.get(sha_key) != expected or payload_sha256(payload.get(value_key)) != expected:
        raise RuntimeError(f"C5 {label} payload or SHA-256 changed")


def _arm_ids(arm_specs: Any) -> list[str]:
    if not isinstance(arm_specs, list) or len(arm_specs) != 36:
        raise RuntimeError("C5 requires exactly 36 materialized arm specifications")
    ids = [row.get("arm_id") for row in arm_specs if isinstance(row, Mapping)]
    indices = [row.get("arm_index") for row in arm_specs if isinstance(row, Mapping)]
    if len(ids) != 36 or any(not isinstance(value, str) or not value for value in ids):
        raise RuntimeError("C5 arm identifiers are invalid")
    if len(set(ids)) != 36 or indices != list(range(36)):
        raise RuntimeError("C5 arm identifiers or order changed")
    combinations = {
        (int(row["stride_voxels"]), float(row["post_rms_amplitude"]), int(row["bias_level"])) for row in arm_specs
    }
    expected = {
        (reach, amplitude, bias) for reach in (1, 2, 3, 4) for amplitude in (0.5, 1.0, 2.0) for bias in (0, 1, 2)
    }
    if combinations != expected:
        raise RuntimeError("C5 arm grid is not the frozen 4x3x3 factorial")
    if tuple(ids) != SELECTABLE_ARM_IDS:
        raise RuntimeError("C5 arm IDs differ from the frozen policy owner")
    return ids


def _selector_ids(selector_specs: Any) -> list[str]:
    if not isinstance(selector_specs, list) or len(selector_specs) != 5:
        raise RuntimeError("C5 requires exactly five virtual selector specifications")
    ids = [row.get("selector_id") for row in selector_specs if isinstance(row, Mapping)]
    indices = [row.get("selector_index") for row in selector_specs if isinstance(row, Mapping)]
    if len(ids) != 5 or len(set(ids)) != 5 or indices != list(range(5)):
        raise RuntimeError("C5 selector identifiers or order changed")
    if tuple(ids) != SELECTOR_IDS:
        raise RuntimeError("C5 selector IDs differ from the frozen policy owner")
    return ids


def validate_support_contract(support: Mapping[str, Any]) -> None:
    if support != EXPECTED_SUPPORT_CONTRACT:
        raise RuntimeError("C5 support/utility/NGF contract changed")


def build_decision_contract(
    source: Mapping[str, Any],
    source_sha256: str,
    *,
    decision_policy: Mapping[str, Any],
    expected_decision_policy_sha256: str,
    arm_specs: Sequence[Mapping[str, Any]],
    expected_arm_specs_sha256: str,
    selector_specs: Sequence[Mapping[str, Any]],
    expected_selector_specs_sha256: str,
    offset_table: Sequence[Mapping[str, Any]],
    expected_offset_table_sha256: str,
    support_contract: Mapping[str, Any],
    expected_support_contract_sha256: str,
) -> dict[str, Any]:
    validate_source_contract(source)
    source_sha = _require_sha(source_sha256, "C5 source contract")
    if expected_decision_policy_sha256 != C5_DECISION_POLICY_SHA256 or decision_policy != decision_policy_contract():
        raise RuntimeError("C5 decision policy differs from its frozen public owner")
    values = (
        (decision_policy, expected_decision_policy_sha256, "decision policy"),
        (list(arm_specs), expected_arm_specs_sha256, "arm specifications"),
        (list(selector_specs), expected_selector_specs_sha256, "selector specifications"),
        (list(offset_table), expected_offset_table_sha256, "offset table"),
        (dict(support_contract), expected_support_contract_sha256, "support contract"),
    )
    for value, expected, label in values:
        if payload_sha256(value) != _require_sha(expected, f"C5 {label}"):
            raise RuntimeError(f"C5 {label} SHA-256 does not match its payload")
    _arm_ids(list(arm_specs))
    _selector_ids(list(selector_specs))
    validate_support_contract(support_contract)
    payload = {
        "schema": DECISION_SCHEMA,
        "protocol_id": PROTOCOL_ID,
        "git_head": source["git_head"],
        "runtime_signature": source["runtime_signature"],
        "source_contract_sha256": source_sha,
        "source_c4_manifest_sha256": SOURCE_C4_MANIFEST_SHA256,
        "source_c4_run_manifest_sha256": SOURCE_C4_RUN_MANIFEST_SHA256,
        "roots": source["roots"],
        "image_inputs": source["image_inputs"],
        "source_initial": source["source_initial"],
        "source_historical": source["source_historical"],
        "source_c4_anchors": source["source_c4_anchors"],
        "baseline_geometry": source["baseline_geometry"],
        "case_ids": source["case_ids"],
        "seed": source["seed"],
        "full_policy_sha256": source["full_policy_sha256"],
        "contrast_contract_sha256": source["contrast_contract_sha256"],
        "decision_policy": dict(decision_policy),
        "decision_policy_sha256": expected_decision_policy_sha256,
        "arm_specs": list(arm_specs),
        "arm_specs_sha256": expected_arm_specs_sha256,
        "selector_specs": list(selector_specs),
        "selector_specs_sha256": expected_selector_specs_sha256,
        "offset_table": list(offset_table),
        "offset_table_sha256": expected_offset_table_sha256,
        "support_contract": dict(support_contract),
        "support_contract_sha256": expected_support_contract_sha256,
        "decision_contract_contains_label_data": False,
        "labels_available_to_decision_workers": False,
        "decision_worker_uses_raw_containers": False,
        "ixi_test_split_accessed": False,
        "test_115_authorized": False,
        "num_shards": source["num_shards"],
        "physical_gpus": source["physical_gpus"],
        "shard_to_physical_gpu": source["shard_to_physical_gpu"],
        "shards": source["shards"],
    }
    validate_decision_contract(
        payload,
        source=source,
        expected_source_sha256=source_sha,
        expected_decision_policy_sha256=expected_decision_policy_sha256,
        expected_arm_specs_sha256=expected_arm_specs_sha256,
        expected_selector_specs_sha256=expected_selector_specs_sha256,
        expected_offset_table_sha256=expected_offset_table_sha256,
        expected_support_contract_sha256=expected_support_contract_sha256,
        expected_contrast_contract_sha256=source["contrast_contract_sha256"],
    )
    return payload


def _validate_decision_owned(
    payload: Mapping[str, Any],
    *,
    expected_source_sha256: str,
    expected_decision_policy_sha256: str,
    expected_arm_specs_sha256: str,
    expected_selector_specs_sha256: str,
    expected_offset_table_sha256: str,
    expected_support_contract_sha256: str,
    expected_contrast_contract_sha256: str,
) -> None:
    if (
        payload.get("schema") != DECISION_SCHEMA
        or payload.get("protocol_id") != PROTOCOL_ID
        or GIT_SHA_RE.fullmatch(str(payload.get("git_head", ""))) is None
        or payload.get("source_contract_sha256") != _require_sha(expected_source_sha256, "C5 source")
        or payload.get("source_c4_manifest_sha256") != SOURCE_C4_MANIFEST_SHA256
        or payload.get("source_c4_run_manifest_sha256") != SOURCE_C4_RUN_MANIFEST_SHA256
        or payload.get("full_policy_sha256") != C5_POLICY_SHA256
        or payload.get("contrast_contract_sha256")
        != _require_sha(expected_contrast_contract_sha256, "C5 contrast contract")
        or payload.get("decision_policy_sha256") != C5_DECISION_POLICY_SHA256
        or payload.get("decision_policy") != decision_policy_contract()
        or (payload.get("decision_policy") or {}).get("labels_accessible") is not False
        or payload.get("decision_contract_contains_label_data") is not False
        or payload.get("labels_available_to_decision_workers") is not False
        or payload.get("decision_worker_uses_raw_containers") is not False
        or payload.get("ixi_test_split_accessed") is not False
        or payload.get("test_115_authorized") is not False
        or "raw_inputs" in payload
        or "evaluation_baseline_dice" in payload
    ):
        raise RuntimeError("Invalid or altered C5 label-free decision contract")
    _root_map(payload.get("roots"))
    _case_ids(payload.get("case_ids"), "C5 decision contract")
    _validate_hashed_payload(
        payload,
        value_key="decision_policy",
        sha_key="decision_policy_sha256",
        expected_sha256=expected_decision_policy_sha256,
        label="decision policy",
    )
    for value_key, sha_key, expected, label in (
        ("arm_specs", "arm_specs_sha256", expected_arm_specs_sha256, "arm specifications"),
        ("selector_specs", "selector_specs_sha256", expected_selector_specs_sha256, "selector specifications"),
        ("offset_table", "offset_table_sha256", expected_offset_table_sha256, "offset table"),
        ("support_contract", "support_contract_sha256", expected_support_contract_sha256, "support contract"),
    ):
        _validate_hashed_payload(
            payload,
            value_key=value_key,
            sha_key=sha_key,
            expected_sha256=expected,
            label=label,
        )
    _arm_ids(payload.get("arm_specs"))
    _selector_ids(payload.get("selector_specs"))
    validate_support_contract(payload.get("support_contract") or {})
    _validate_sharding(payload)
    _assert_label_free(payload, "C5 decision contract")


def validate_decision_contract(
    payload: Mapping[str, Any],
    *,
    source: Mapping[str, Any],
    expected_source_sha256: str,
    expected_decision_policy_sha256: str,
    expected_arm_specs_sha256: str,
    expected_selector_specs_sha256: str,
    expected_offset_table_sha256: str,
    expected_support_contract_sha256: str,
    expected_contrast_contract_sha256: str,
) -> None:
    validate_source_contract(source)
    _validate_decision_owned(
        payload,
        expected_source_sha256=expected_source_sha256,
        expected_decision_policy_sha256=expected_decision_policy_sha256,
        expected_arm_specs_sha256=expected_arm_specs_sha256,
        expected_selector_specs_sha256=expected_selector_specs_sha256,
        expected_offset_table_sha256=expected_offset_table_sha256,
        expected_support_contract_sha256=expected_support_contract_sha256,
        expected_contrast_contract_sha256=expected_contrast_contract_sha256,
    )
    projected = (
        "git_head",
        "runtime_signature",
        "roots",
        "image_inputs",
        "source_initial",
        "source_historical",
        "source_c4_anchors",
        "baseline_geometry",
        "case_ids",
        "seed",
        "num_shards",
        "physical_gpus",
        "shard_to_physical_gpu",
        "shards",
    )
    if any(payload.get(key) != source.get(key) for key in projected):
        raise RuntimeError("C5 decision contract differs from its source projection")


def prepare_contracts(
    *,
    run_root: Path,
    source_c4_dir: Path,
    source_c4_heavy_root: Path,
    target_heavy_root: Path,
    git_head: str,
    runtime_signature: Mapping[str, Any],
    physical_gpus: Sequence[str],
    full_policy: Mapping[str, Any],
    expected_full_policy_sha256: str,
    decision_policy: Mapping[str, Any],
    expected_decision_policy_sha256: str,
    arm_specs: Sequence[Mapping[str, Any]],
    expected_arm_specs_sha256: str,
    selector_specs: Sequence[Mapping[str, Any]],
    expected_selector_specs_sha256: str,
    offset_table: Sequence[Mapping[str, Any]],
    expected_offset_table_sha256: str,
    support_contract: Mapping[str, Any],
    expected_support_contract_sha256: str,
    contrast_contract: Mapping[str, Any],
    expected_contrast_contract_sha256: str,
    verify_anchor_bytes: bool = True,
) -> ContractBundle:
    snapshot = authenticate_frozen_c4(
        source_c4_dir,
        source_c4_heavy_root,
        verify_anchor_bytes=verify_anchor_bytes,
    )
    source = build_source_contract(
        snapshot,
        git_head=git_head,
        runtime_signature=runtime_signature,
        target_heavy_root=target_heavy_root,
        physical_gpus=physical_gpus,
        full_policy=full_policy,
        expected_full_policy_sha256=expected_full_policy_sha256,
        contrast_contract=contrast_contract,
        expected_contrast_contract_sha256=expected_contrast_contract_sha256,
    )
    root = run_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    source_sha = _write_immutable_json(root / "source_contract.json", source)
    decision = build_decision_contract(
        source,
        source_sha,
        decision_policy=decision_policy,
        expected_decision_policy_sha256=expected_decision_policy_sha256,
        arm_specs=arm_specs,
        expected_arm_specs_sha256=expected_arm_specs_sha256,
        selector_specs=selector_specs,
        expected_selector_specs_sha256=expected_selector_specs_sha256,
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
    if actual != _require_sha(expected_sha256, "C5 source contract"):
        raise RuntimeError("C5 source contract hash mismatch")
    payload = _load_json(path)
    validate_source_contract(payload)
    return payload, actual


def load_decision_contract(
    run_root: Path,
    expected_sha256: str,
    *,
    source: Mapping[str, Any],
    expected_source_sha256: str,
    expected_decision_policy_sha256: str,
    expected_arm_specs_sha256: str,
    expected_selector_specs_sha256: str,
    expected_offset_table_sha256: str,
    expected_support_contract_sha256: str,
    expected_contrast_contract_sha256: str,
) -> tuple[dict[str, Any], str]:
    path = run_root.resolve() / "decision_contract.json"
    actual = sha256_file(path)
    if actual != _require_sha(expected_sha256, "C5 decision contract"):
        raise RuntimeError("C5 decision contract hash mismatch")
    payload = _load_json(path)
    validate_decision_contract(
        payload,
        source=source,
        expected_source_sha256=expected_source_sha256,
        expected_decision_policy_sha256=expected_decision_policy_sha256,
        expected_arm_specs_sha256=expected_arm_specs_sha256,
        expected_selector_specs_sha256=expected_selector_specs_sha256,
        expected_offset_table_sha256=expected_offset_table_sha256,
        expected_support_contract_sha256=expected_support_contract_sha256,
        expected_contrast_contract_sha256=expected_contrast_contract_sha256,
    )
    return payload, actual


def load_decision_contract_isolated(
    run_root: Path,
    expected_decision_sha256: str,
    *,
    expected_source_sha256: str,
    expected_decision_policy_sha256: str,
    expected_arm_specs_sha256: str,
    expected_selector_specs_sha256: str,
    expected_offset_table_sha256: str,
    expected_support_contract_sha256: str,
    expected_contrast_contract_sha256: str,
) -> tuple[dict[str, Any], str]:
    root = run_root.resolve()
    decision_path = root / "decision_contract.json"
    actual = sha256_file(decision_path)
    if actual != _require_sha(expected_decision_sha256, "C5 decision contract"):
        raise RuntimeError("C5 decision contract hash mismatch")
    if sha256_file(root / "source_contract.json") != _require_sha(expected_source_sha256, "C5 source contract"):
        raise RuntimeError("C5 source contract byte hash mismatch")
    payload = _load_json(decision_path)
    _validate_decision_owned(
        payload,
        expected_source_sha256=expected_source_sha256,
        expected_decision_policy_sha256=expected_decision_policy_sha256,
        expected_arm_specs_sha256=expected_arm_specs_sha256,
        expected_selector_specs_sha256=expected_selector_specs_sha256,
        expected_offset_table_sha256=expected_offset_table_sha256,
        expected_support_contract_sha256=expected_support_contract_sha256,
        expected_contrast_contract_sha256=expected_contrast_contract_sha256,
    )
    return payload, actual


def _expected_shard(contract: Mapping[str, Any], case_id: str) -> int:
    matches = [int(index) for index, values in contract["shards"].items() if case_id in values]
    if len(matches) != 1:
        raise RuntimeError(f"C5 case does not have exactly one shard: {case_id}")
    return matches[0]


def _metric_value(bundle: Any, metric_id: str, label: str) -> float:
    if not isinstance(bundle, Mapping):
        raise RuntimeError(f"{label} geometry bundle is missing")
    record = bundle.get(metric_id)
    if not isinstance(record, Mapping) or record.get("metric_id") != metric_id or record.get("status") != "OK":
        raise RuntimeError(f"{label} geometry metric is absent or invalid: {metric_id}")
    return _finite(record.get("value"), f"{label} {metric_id}")


def _candidate_signals(
    row: Mapping[str, Any],
    spec: Mapping[str, Any],
    baseline_geometry: Mapping[str, Any],
    roots: Mapping[str, Path],
    *,
    decision_policy: Mapping[str, Any],
    expected_offset_table_sha256: str,
    expected_support_contract_sha256: str,
    expected_informative_count: int,
    verify_heavy_bytes: bool,
    historical_anchor_field: Mapping[str, Any] | None = None,
) -> CandidateSignals:
    if row.get("arm_index") != spec["arm_index"] or row.get("arm_id") != spec["arm_id"]:
        raise RuntimeError("C5 decision arm order or identity changed")
    factor_fields = (
        "descriptor_id",
        "reach_id",
        "stride_voxels",
        "post_rms_amplitude",
        "centre_beta",
        "historical_anchor",
        "selectable",
    )
    if any(row.get(key) != spec.get(key) for key in factor_fields):
        raise RuntimeError(f"C5 arm factor identity changed: {spec['arm_id']}")
    for key, expected in (
        ("offset_table_sha256", expected_offset_table_sha256),
        ("support_contract_sha256", expected_support_contract_sha256),
    ):
        if key in row and row.get(key) != expected:
            raise RuntimeError(f"C5 arm contract provenance changed: {spec['arm_id']}/{key}")
    field = row.get("candidate_field") or {}
    expected_root = "source_c4_heavy" if bool(spec.get("historical_anchor")) else "target_c5_heavy"
    if field.get("root_id") != expected_root:
        raise RuntimeError(f"C5 candidate field has the wrong root owner: {spec['arm_id']}")
    if historical_anchor_field is not None and field != historical_anchor_field:
        raise RuntimeError(f"C5 historical anchor was duplicated or changed: {spec['arm_id']}")
    if bool(spec.get("historical_anchor")) != (historical_anchor_field is not None):
        raise RuntimeError(f"C5 historical-anchor role changed: {spec['arm_id']}")
    verify_rooted_record(field, roots, verify_bytes=verify_heavy_bytes, verify_array=verify_heavy_bytes)
    persistence = row.get("persistence") or {}
    if (
        persistence.get("owner") != field.get("root_id")
        or persistence.get("saved_npz_sha256") != field.get("npz_sha256")
        or persistence.get("reloaded_array_sha256") != field.get("array_sha256")
        or persistence.get("source_anchor_reused") is not bool(spec.get("historical_anchor"))
    ):
        raise RuntimeError(f"C5 candidate persistence provenance changed: {spec['arm_id']}")
    exact = row.get("exact") or {}
    if (
        exact.get("status") != "CERTIFIED"
        or exact.get("certified") is not True
        or exact.get("sha256") != field.get("array_sha256")
    ):
        raise RuntimeError(f"C5 candidate lacks its exact certificate: {spec['arm_id']}")

    support = row.get("support") or {}
    utilities = row.get("utilities") or {}
    expected_utilities = {"ncc5", "ncc7", "ncc9", "mind_d2", "ngf"}
    if not isinstance(utilities, Mapping) or set(utilities) != expected_utilities:
        raise RuntimeError(f"C5 utility inventory changed: {spec['arm_id']}")

    def validate_utility(
        utility: Mapping[str, Any],
        utility_id: str,
        *,
        selector_eligible: bool,
    ) -> tuple[float, dict[str, Any]]:
        baseline_count = utility.get("baseline_count")
        pair_count = utility.get("pair_count")
        if (
            isinstance(baseline_count, bool)
            or not isinstance(baseline_count, int)
            or baseline_count <= 0
            or isinstance(pair_count, bool)
            or not isinstance(pair_count, int)
            or not 0 < pair_count <= baseline_count
        ):
            raise RuntimeError(f"C5 utility support counts are invalid: {spec['arm_id']}/{utility_id}")
        retention_value = _finite(utility.get("retention"), f"C5 {spec['arm_id']}/{utility_id} support retention")
        baseline_loss = _finite(utility.get("baseline_loss"), f"C5 {spec['arm_id']}/{utility_id} baseline loss")
        candidate_loss = _finite(utility.get("candidate_loss"), f"C5 {spec['arm_id']}/{utility_id} candidate loss")
        improvement = _finite(utility.get("improvement"), f"C5 {spec['arm_id']}/{utility_id} improvement")
        if (
            utility.get("utility_id") != utility_id
            or utility.get("selector_eligible") is not selector_eligible
            or not math.isclose(retention_value, pair_count / baseline_count, rel_tol=0.0, abs_tol=1e-12)
            or not math.isclose(improvement, baseline_loss - candidate_loss, rel_tol=0.0, abs_tol=1e-12)
        ):
            raise RuntimeError(f"C5 utility arithmetic changed: {spec['arm_id']}/{utility_id}")
        counts = {
            "baseline_count": baseline_count,
            "pair_count": pair_count,
            "retention": retention_value,
        }
        return improvement, counts

    validated_utilities = {
        name: validate_utility(
            utilities[name],
            str(EXPECTED_SUPPORT_CONTRACT["utilities"][name]["utility_id"]),
            selector_eligible=bool(EXPECTED_SUPPORT_CONTRACT["utilities"][name]["selector_eligible"]),
        )
        for name in ("ncc5", "ncc7", "ncc9", "mind_d2")
    }
    ncc_improvement, ncc_counts = validated_utilities["ncc7"]
    mind_improvement, mind_counts = validated_utilities["mind_d2"]

    ngf = utilities["ngf"]
    ngf_definition = EXPECTED_SUPPORT_CONTRACT["utilities"]["ngf"]
    ngf_baseline_count = ngf.get("baseline_support_count")
    ngf_pair_count = ngf.get("pair_support_count")
    ngf_eta = _finite(ngf.get("eta"), f"C5 {spec['arm_id']}/ngf eta")
    ngf_baseline = _finite(ngf.get("baseline_similarity"), f"C5 {spec['arm_id']}/ngf baseline similarity")
    ngf_candidate = _finite(ngf.get("candidate_similarity"), f"C5 {spec['arm_id']}/ngf candidate similarity")
    ngf_improvement = _finite(ngf.get("improvement"), f"C5 {spec['arm_id']}/ngf improvement")
    if (
        ngf.get("diagnostic_id") != ngf_definition["diagnostic_id"]
        or ngf.get("selector_eligible") is not False
        or isinstance(ngf_baseline_count, bool)
        or not isinstance(ngf_baseline_count, int)
        or ngf_baseline_count <= 0
        or isinstance(ngf_pair_count, bool)
        or not isinstance(ngf_pair_count, int)
        or not 0 < ngf_pair_count <= ngf_baseline_count
        or ngf_eta <= 0.0
        or not 0.0 <= ngf_baseline <= 1.0
        or not 0.0 <= ngf_candidate <= 1.0
        or not math.isclose(ngf_improvement, ngf_candidate - ngf_baseline, rel_tol=0.0, abs_tol=1e-12)
    ):
        raise RuntimeError(f"C5 NGF diagnostic arithmetic changed: {spec['arm_id']}")
    retention = _finite(support.get("retention"), f"C5 {spec['arm_id']} support retention")
    if (
        support.get("ncc7") != ncc_counts
        or support.get("mind_d2") != mind_counts
        or not math.isclose(
            retention, min(ncc_counts["retention"], mind_counts["retention"]), rel_tol=0.0, abs_tol=1e-12
        )
    ):
        raise RuntimeError(f"C5 combined utility support changed: {spec['arm_id']}")
    baseline_sdlogj = _metric_value(baseline_geometry, MATHEMATICAL_SDLOGJ_CROP2, "C5 baseline")
    candidate_sdlogj = _metric_value(row.get("geometry"), MATHEMATICAL_SDLOGJ_CROP2, f"C5 {spec['arm_id']}")
    geometry_delta = _finite(row.get("mathematical_sdlogj_delta"), f"C5 {spec['arm_id']} mathematical SDlogJ delta")
    if not math.isclose(geometry_delta, candidate_sdlogj - baseline_sdlogj, rel_tol=0.0, abs_tol=1e-12):
        raise RuntimeError(f"C5 geometry delta arithmetic changed: {spec['arm_id']}")

    proposal = row.get("proposal") or {}
    posterior = proposal.get("posterior") or {}
    pipeline = dict(decision_policy.get("proposal_pipeline") or ())
    reaches = {item.get("reach_id"): item for item in decision_policy.get("reaches") or () if isinstance(item, Mapping)}
    reach = reaches.get(spec["reach_id"]) or {}
    if (
        proposal.get("reach_id") != spec["reach_id"]
        or proposal.get("stride_voxels") != spec["stride_voxels"]
        or proposal.get("centre_beta") != spec["centre_beta"]
        or proposal.get("pre_rms_multiplier") != reach.get("pre_rms_multiplier")
        or proposal.get("amplitude_stage") != pipeline.get("amplitude_stage")
        or proposal.get("smoothing_passes") != pipeline.get("post_smoothing_passes")
        or proposal.get("collar_width") != pipeline.get("evidence_collar")
        or proposal.get("rms_target_source_id") != pipeline.get("rms_target_source_id")
        or isinstance(posterior.get("active_voxels"), bool)
        or posterior.get("active_voxels") != expected_informative_count
    ):
        raise RuntimeError(f"C5 proposal factor or pipeline identity changed: {spec['arm_id']}")
    amplitude = _finite(proposal.get("post_rms_amplitude"), f"C5 {spec['arm_id']} amplitude")
    target = _finite(proposal.get("rms_target"), f"C5 {spec['arm_id']} RMS target")
    requested = _finite(proposal.get("rms_requested"), f"C5 {spec['arm_id']} requested RMS")
    realized = _finite(proposal.get("rms_realized"), f"C5 {spec['arm_id']} realized RMS")
    clip_retention_raw = _finite(proposal.get("clip_rms_retention_raw"), f"C5 {spec['arm_id']} raw clip retention")
    clip_retention = _finite(proposal.get("clip_rms_retention"), f"C5 {spec['arm_id']} clip retention")
    clip_cosine = _finite(proposal.get("clip_cosine"), f"C5 {spec['arm_id']} clip cosine")
    if (
        amplitude != float(spec["post_rms_amplitude"])
        or target <= 0.0
        or requested <= 0.0
        or realized < 0.0
        or not math.isclose(requested, target * amplitude, rel_tol=1e-7, abs_tol=1e-8)
        or not math.isclose(clip_retention_raw, realized / requested, rel_tol=1e-7, abs_tol=1e-8)
        or not 0.0 <= clip_retention_raw <= 1.0 + 1e-6
        or not math.isclose(clip_retention, min(1.0, clip_retention_raw), rel_tol=0.0, abs_tol=1e-12)
        or not -1.0 <= clip_cosine <= 1.0
    ):
        raise RuntimeError(f"C5 post-RMS amplitude or clipping diagnostics are inconsistent: {spec['arm_id']}")
    return CandidateSignals(
        arm_id=spec["arm_id"],
        exact_certified=True,
        support_retention=retention,
        amplitude_retention=clip_retention,
        ncc7_improvement=ncc_improvement,
        mind_d2_improvement=mind_improvement,
        mathematical_sdlogj_delta=geometry_delta,
    )


def _validate_case_provenance(
    payload: Mapping[str, Any],
    contract: Mapping[str, Any],
    case_id: str,
) -> tuple[float, dict[str, int]]:
    historical = contract["source_historical"][case_id]
    rms = payload.get("source_rms_reference") or {}
    residual_rms = _finite(rms.get("residual_rms"), f"C5 {case_id} RMS reference")
    if (
        rms.get("field") != historical.get("raw_conf_requested_field")
        or rms.get("source_decision_case_sha256") != historical.get("source_decision_case_sha256")
        or residual_rms <= 0.0
    ):
        raise RuntimeError(f"C5 RMS-reference provenance changed: {case_id}")

    if payload.get("baseline_geometry") != contract["baseline_geometry"][case_id]:
        raise RuntimeError(f"C5 baseline geometry differs from its frozen contract: {case_id}")
    if payload.get("source_image_array_sha256") != contract["image_inputs"][case_id].get("array_sha256"):
        raise RuntimeError(f"C5 decision marker is bound to another source image: {case_id}")

    support = payload.get("generation_support") or {}
    geometry_count = support.get("geometry_count")
    common_count = support.get("common_count")
    if (
        support.get("support_id") != contract["support_contract"]["geometry"]["generation_support_id"]
        or isinstance(geometry_count, bool)
        or not isinstance(geometry_count, int)
        or geometry_count <= 0
        or isinstance(common_count, bool)
        or not isinstance(common_count, int)
        or not 0 < common_count <= geometry_count
        or not math.isclose(
            _finite(support.get("retention"), f"C5 {case_id} generation retention"),
            common_count / geometry_count,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
    ):
        raise RuntimeError(f"C5 generation support changed: {case_id}")
    expected_reaches = contract["decision_policy"].get("reaches") or []
    observed_reaches = support.get("reach")
    if not isinstance(observed_reaches, list) or len(observed_reaches) != 4 or len(expected_reaches) != 4:
        raise RuntimeError(f"C5 reach-support inventory changed: {case_id}")
    informative_by_reach: dict[str, int] = {}
    for observed, expected in zip(observed_reaches, expected_reaches, strict=True):
        if not isinstance(observed, Mapping) or not isinstance(expected, Mapping):
            raise RuntimeError(f"C5 reach support changed: {case_id}")
        informative_count = observed.get("standardized_informative_count")
        informative_fraction = _finite(
            observed.get("standardized_informative_fraction"),
            f"C5 {case_id}/{expected.get('reach_id')} standardized informative fraction",
        )
        if (
            observed.get("reach_id") != expected.get("reach_id")
            or observed.get("stride_voxels") != expected.get("stride_voxels")
            or observed.get("generation_count") != common_count
            or observed.get("all_candidates_valid_count") != common_count
            or observed.get("all_candidates_valid") is not True
            or isinstance(informative_count, bool)
            or not isinstance(informative_count, int)
            or not 0 < informative_count <= common_count
            or not math.isclose(
                informative_fraction,
                informative_count / common_count,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
        ):
            raise RuntimeError(f"C5 reach support changed: {case_id}/{expected.get('reach_id')}")
        informative_by_reach[str(expected["reach_id"])] = informative_count
    return residual_rms, informative_by_reach


def validate_decision_case_marker(
    payload: Mapping[str, Any],
    contract: Mapping[str, Any],
    decision_contract_sha256: str,
    *,
    verify_heavy_bytes: bool = True,
) -> None:
    case_id = payload.get("case_id")
    if case_id not in contract.get("case_ids", []):
        raise RuntimeError("C5 decision marker belongs to a foreign case")
    shard = _expected_shard(contract, case_id)
    if (
        payload.get("schema") != DECISION_CASE_SCHEMA
        or payload.get("protocol_id") != PROTOCOL_ID
        or payload.get("status") != "COMPLETE"
        or payload.get("decision_contract_sha256") != _require_sha(decision_contract_sha256, "C5 decision contract")
        or payload.get("shard_index") != shard
        or str(payload.get("physical_gpu")) != contract["shard_to_physical_gpu"][str(shard)]
        or payload.get("labels_loaded_to_device") is not False
        or payload.get("test_split_accessed") is not False
        or payload.get("arm_specs_sha256") != contract["arm_specs_sha256"]
        or payload.get("selector_specs_sha256") != contract["selector_specs_sha256"]
    ):
        raise RuntimeError(f"Invalid C5 label-free decision marker: {case_id}")
    initial = contract["source_initial"][case_id]["field"]
    if payload.get("source_initial_array_sha256") != initial.get("array_sha256"):
        raise RuntimeError(f"C5 decision marker is bound to another initial field: {case_id}")
    residual_rms, informative_by_reach = _validate_case_provenance(payload, contract, case_id)

    parity = payload.get("historical_anchor_parity")
    anchor_map = dict(zip(HISTORICAL_ANCHOR_ARM_IDS, SOURCE_C4_ANCHOR_IDS, strict=True))
    if not isinstance(parity, list) or [row.get("arm_id") for row in parity] != list(HISTORICAL_ANCHOR_ARM_IDS):
        raise RuntimeError(f"C5 historical anchor parity inventory changed: {case_id}")
    for row in parity:
        source_id = anchor_map[row["arm_id"]]
        expected = contract["source_c4_anchors"][case_id][source_id]["field"]["array_sha256"]
        if (
            row.get("source_anchor_id") != source_id
            or row.get("source_array_sha256") != expected
            or row.get("candidate_array_sha256") != expected
            or row.get("array_byte_identical") is not True
        ):
            raise RuntimeError(f"C5 historical anchor parity failed: {case_id}/{row['arm_id']}")

    arms = payload.get("arms")
    specs = contract["arm_specs"]
    if not isinstance(arms, list) or [row.get("arm_id") for row in arms] != [row["arm_id"] for row in specs]:
        raise RuntimeError(f"C5 decision marker has the wrong arm order: {case_id}")
    roots = _root_map(contract["roots"])
    source_anchor_by_c5 = {
        c5_id: contract["source_c4_anchors"][case_id][c4_id]["field"] for c5_id, c4_id in anchor_map.items()
    }
    signals = [
        _candidate_signals(
            row,
            spec,
            contract["baseline_geometry"][case_id],
            roots,
            decision_policy=contract["decision_policy"],
            expected_offset_table_sha256=contract["offset_table_sha256"],
            expected_support_contract_sha256=contract["support_contract_sha256"],
            expected_informative_count=informative_by_reach[str(spec["reach_id"])],
            verify_heavy_bytes=verify_heavy_bytes,
            historical_anchor_field=source_anchor_by_c5.get(spec["arm_id"]),
        )
        for row, spec in zip(arms, specs, strict=True)
    ]
    for row in arms:
        proposal = row.get("proposal") or {}
        if not math.isclose(
            _finite(proposal.get("rms_target"), f"C5 {row.get('arm_id')} RMS target"),
            residual_rms,
            rel_tol=1e-7,
            abs_tol=1e-8,
        ):
            raise RuntimeError(f"C5 arm is bound to another RMS reference: {case_id}/{row.get('arm_id')}")
    by_arm = {row["arm_id"]: row for row in arms}
    selectors = payload.get("selectors")
    if not isinstance(selectors, list) or [row.get("selector_id") for row in selectors] != list(SELECTOR_IDS):
        raise RuntimeError(f"C5 decision marker has the wrong selector order: {case_id}")
    for index, row in enumerate(selectors):
        expected = choose_global_candidate(signals, row["selector_id"])
        if (
            row.get("selector_index") != index
            or row.get("action") != expected.action
            or row.get("selected_arm_id") != expected.selected_arm_id
            or tuple(row.get("eligible_arm_ids") or ()) != expected.eligible_arm_ids
        ):
            raise RuntimeError(f"C5 selector output changed: {case_id}/{row.get('selector_id')}")
        returned = row.get("returned_field")
        if expected.selected_arm_id is None:
            if returned is not None or row.get("rollback_to_source_initial") is not True:
                raise RuntimeError(f"C5 empty selector did not return baseline: {case_id}/{row['selector_id']}")
        elif (
            returned != by_arm[expected.selected_arm_id]["candidate_field"]
            or row.get("rollback_to_source_initial") is not False
        ):
            raise RuntimeError(f"C5 selector returned a foreign candidate: {case_id}/{row['selector_id']}")
    _assert_label_free(payload, "C5 decision marker")


def validate_worker_marker(
    payload: Mapping[str, Any],
    contract: Mapping[str, Any],
    decision_contract_sha256: str,
    *,
    phase: str,
    shard_index: int,
    attempt_id: str,
    barrier_sha256: str | None = None,
    evaluation_contract_sha256: str | None = None,
) -> None:
    if phase not in {"decision", "evaluation"}:
        raise ValueError(f"Unsupported C5 worker phase: {phase}")
    expected_cases = contract["shards"][str(shard_index)]
    if (
        payload.get("schema") != WORKER_SCHEMA
        or payload.get("protocol_id") != PROTOCOL_ID
        or payload.get("status") != "COMPLETE"
        or payload.get("phase") != phase
        or payload.get("attempt_id") != attempt_id
        or payload.get("shard_index") != shard_index
        or str(payload.get("physical_gpu")) != contract["shard_to_physical_gpu"][str(shard_index)]
        or payload.get("case_ids") != expected_cases
        or payload.get("decision_contract_sha256") != decision_contract_sha256
        or payload.get("test_split_accessed") is not False
    ):
        raise RuntimeError(f"Invalid C5 {phase} worker marker for shard {shard_index}")
    if phase == "decision":
        if (
            payload.get("labels_loaded") is not False
            or "decision_barrier_sha256" in payload
            or "evaluation_contract_sha256" in payload
        ):
            raise RuntimeError("C5 decision worker received labels or a premature barrier")
    elif (
        payload.get("labels_loaded") is not True
        or payload.get("decision_barrier_sha256") != _require_sha(barrier_sha256, "C5 decision barrier")
        or payload.get("evaluation_contract_sha256")
        != _require_sha(evaluation_contract_sha256, "C5 evaluation contract")
    ):
        raise RuntimeError("C5 evaluation worker is not bound to the decision barrier")


def build_decision_barrier(
    run_root: Path,
    contract: Mapping[str, Any],
    decision_contract_sha256: str,
    *,
    attempt_id: str,
    worker_paths: Sequence[Path],
    completed_at_utc: str,
    verify_heavy_bytes: bool = True,
) -> dict[str, Any]:
    if len(worker_paths) != contract["num_shards"]:
        raise RuntimeError("C5 decision barrier requires exactly one worker per shard")
    root = run_root.resolve()
    workers: list[dict[str, str]] = []
    for shard, path in enumerate(worker_paths):
        marker = _load_json(path)
        validate_worker_marker(
            marker,
            contract,
            decision_contract_sha256,
            phase="decision",
            shard_index=shard,
            attempt_id=attempt_id,
        )
        workers.append({"path": path.resolve().relative_to(root).as_posix(), "sha256": sha256_file(path)})
    case_hashes: dict[str, str] = {}
    for case_id in contract["case_ids"]:
        path = root / "cases" / case_id / "decision_complete.json"
        marker = _load_json(path)
        validate_decision_case_marker(
            marker,
            contract,
            decision_contract_sha256,
            verify_heavy_bytes=verify_heavy_bytes,
        )
        case_hashes[case_id] = sha256_file(path)
    return {
        "schema": BARRIER_SCHEMA,
        "protocol_id": PROTOCOL_ID,
        "status": "COMPLETE",
        "attempt_id": attempt_id,
        "decision_contract_sha256": decision_contract_sha256,
        "decision_workers_received_label_inputs": False,
        "test_split_accessed": False,
        "workers": workers,
        "decision_case_sha256": case_hashes,
        "completed_at_utc": completed_at_utc,
    }


def write_decision_barrier(run_root: Path, payload: dict[str, Any]) -> str:
    return _write_immutable_json(run_root.resolve() / "decision_barrier.json", payload)


def load_decision_barrier(
    run_root: Path,
    expected_sha256: str,
    *,
    contract: Mapping[str, Any],
    decision_contract_sha256: str,
    case_ids: Sequence[str],
) -> tuple[dict[str, Any], str]:
    path = run_root.resolve() / "decision_barrier.json"
    actual = sha256_file(path)
    if actual != _require_sha(expected_sha256, "C5 decision barrier"):
        raise RuntimeError("C5 decision barrier hash mismatch")
    payload = _load_json(path)
    if (
        payload.get("schema") != BARRIER_SCHEMA
        or payload.get("protocol_id") != PROTOCOL_ID
        or payload.get("status") != "COMPLETE"
        or payload.get("decision_contract_sha256") != decision_contract_sha256
        or payload.get("decision_workers_received_label_inputs") is not False
        or payload.get("test_split_accessed") is not False
        or list((payload.get("decision_case_sha256") or {}).keys()) != list(case_ids)
        or not isinstance(payload.get("completed_at_utc"), str)
        or not payload.get("completed_at_utc")
    ):
        raise RuntimeError("Invalid or altered C5 decision barrier")
    if list(case_ids) != contract.get("case_ids"):
        raise RuntimeError("C5 decision barrier case inventory differs from its contract")
    attempt_id = payload.get("attempt_id")
    workers = payload.get("workers")
    if not isinstance(attempt_id, str) or not attempt_id or not isinstance(workers, list):
        raise RuntimeError("C5 decision barrier worker provenance is missing")
    if len(workers) != contract.get("num_shards"):
        raise RuntimeError("C5 decision barrier worker inventory changed")
    root = run_root.resolve()
    for shard_index, record in enumerate(workers):
        expected_relative = f"workers/decision/attempts/{attempt_id}/worker_{shard_index:02d}.json"
        if not isinstance(record, Mapping) or record.get("path") != expected_relative:
            raise RuntimeError(f"C5 decision barrier worker path changed: shard {shard_index}")
        worker_path = root / expected_relative
        if not worker_path.is_file() or sha256_file(worker_path) != _require_sha(
            record.get("sha256"), f"C5 decision worker {shard_index}"
        ):
            raise RuntimeError(f"C5 decision barrier worker bytes changed: shard {shard_index}")
        validate_worker_marker(
            _load_json(worker_path),
            contract,
            decision_contract_sha256,
            phase="decision",
            shard_index=shard_index,
            attempt_id=attempt_id,
        )
    for case_id, value in payload["decision_case_sha256"].items():
        expected = _require_sha(value, "C5 decision case")
        case_path = root / "cases" / case_id / "decision_complete.json"
        if not case_path.is_file() or sha256_file(case_path) != expected:
            raise RuntimeError(f"C5 frozen decision case bytes changed: {case_id}")
    return payload, actual


def build_evaluation_contract(
    source: Mapping[str, Any],
    source_contract_sha256: str,
    decision_contract_sha256: str,
    barrier: Mapping[str, Any],
    barrier_sha256: str,
) -> dict[str, Any]:
    validate_source_contract(source)
    if (
        barrier.get("schema") != BARRIER_SCHEMA
        or barrier.get("status") != "COMPLETE"
        or barrier.get("decision_contract_sha256") != decision_contract_sha256
        or set(barrier.get("decision_case_sha256") or {}) != set(source["case_ids"])
    ):
        raise RuntimeError("C5 evaluation contract requires a complete matching decision barrier")
    payload = {
        "schema": EVALUATION_CONTRACT_SCHEMA,
        "protocol_id": PROTOCOL_ID,
        "status": "FROZEN_AFTER_DECISION_BARRIER",
        "source_contract_sha256": _require_sha(source_contract_sha256, "C5 source contract"),
        "decision_contract_sha256": _require_sha(decision_contract_sha256, "C5 decision contract"),
        "decision_barrier_sha256": _require_sha(barrier_sha256, "C5 decision barrier"),
        "decision_case_sha256": dict(barrier["decision_case_sha256"]),
        "case_ids": source["case_ids"],
        "full_policy": source["full_policy"],
        "full_policy_sha256": source["full_policy_sha256"],
        "contrast_contract": source["contrast_contract"],
        "contrast_contract_sha256": source["contrast_contract_sha256"],
        "roots": source["roots"],
        "raw_inputs": source["raw_inputs"],
        "evaluation_baseline_dice": source["evaluation_baseline_dice"],
        "evaluation_c4_anchor_dice": source["evaluation_c4_anchor_dice"],
        "evaluation_baseline_per_label": source["evaluation_baseline_per_label"],
        "evaluation_label_ids": source["evaluation_label_ids"],
        "source_c4_anchors": source["source_c4_anchors"],
        "labels_available_only_after_barrier": True,
        "ixi_test_split_accessed": False,
        "test_115_authorized": False,
    }
    validate_evaluation_contract(
        payload,
        source=source,
        barrier=barrier,
        expected_source_sha256=source_contract_sha256,
        expected_decision_sha256=decision_contract_sha256,
        expected_barrier_sha256=barrier_sha256,
    )
    return payload


def validate_evaluation_contract(
    payload: Mapping[str, Any],
    *,
    source: Mapping[str, Any],
    barrier: Mapping[str, Any],
    expected_source_sha256: str,
    expected_decision_sha256: str,
    expected_barrier_sha256: str,
) -> None:
    validate_source_contract(source)
    if (
        payload.get("schema") != EVALUATION_CONTRACT_SCHEMA
        or payload.get("protocol_id") != PROTOCOL_ID
        or payload.get("status") != "FROZEN_AFTER_DECISION_BARRIER"
        or payload.get("source_contract_sha256") != _require_sha(expected_source_sha256, "C5 source")
        or payload.get("decision_contract_sha256") != _require_sha(expected_decision_sha256, "C5 decision")
        or payload.get("decision_barrier_sha256") != _require_sha(expected_barrier_sha256, "C5 barrier")
        or payload.get("labels_available_only_after_barrier") is not True
        or payload.get("ixi_test_split_accessed") is not False
        or payload.get("test_115_authorized") is not False
    ):
        raise RuntimeError("Invalid or altered C5 post-barrier evaluation contract")
    projected = (
        "case_ids",
        "full_policy",
        "full_policy_sha256",
        "contrast_contract",
        "contrast_contract_sha256",
        "roots",
        "raw_inputs",
        "evaluation_baseline_dice",
        "evaluation_c4_anchor_dice",
        "evaluation_baseline_per_label",
        "evaluation_label_ids",
        "source_c4_anchors",
    )
    if any(payload.get(key) != source.get(key) for key in projected):
        raise RuntimeError("C5 evaluation contract differs from its authenticated source projection")
    if tuple(payload.get("evaluation_label_ids") or ()) != EVALUATION_LABEL_IDS:
        raise RuntimeError("C5 evaluation contract has the wrong IXI label order")
    if list((payload.get("decision_case_sha256") or {}).keys()) != source["case_ids"]:
        raise RuntimeError("C5 evaluation contract has the wrong decision-case inventory")
    if payload.get("decision_case_sha256") != barrier.get("decision_case_sha256"):
        raise RuntimeError("C5 evaluation contract differs from the frozen decision-case hashes")


def write_evaluation_contract(run_root: Path, payload: dict[str, Any]) -> str:
    return _write_immutable_json(run_root.resolve() / "evaluation_contract.json", payload)


def load_evaluation_contract(
    run_root: Path,
    expected_sha256: str,
    *,
    source: Mapping[str, Any],
    barrier: Mapping[str, Any],
    expected_source_sha256: str,
    expected_decision_sha256: str,
    expected_barrier_sha256: str,
) -> tuple[dict[str, Any], str]:
    path = run_root.resolve() / "evaluation_contract.json"
    actual = sha256_file(path)
    if actual != _require_sha(expected_sha256, "C5 evaluation contract"):
        raise RuntimeError("C5 evaluation contract hash mismatch")
    payload = _load_json(path)
    validate_evaluation_contract(
        payload,
        source=source,
        barrier=barrier,
        expected_source_sha256=expected_source_sha256,
        expected_decision_sha256=expected_decision_sha256,
        expected_barrier_sha256=expected_barrier_sha256,
    )
    return payload, actual


def _dice(value: Any, label: str) -> float:
    result = _finite(value, label)
    if not 0.0 <= result <= 1.0:
        raise RuntimeError(f"{label} must be in [0,1]")
    return result


def _validate_per_label_dice(
    rows: Any,
    *,
    baseline: Sequence[Mapping[str, Any]],
    candidate_key: str,
    label: str,
) -> dict[int, float]:
    if not isinstance(rows, list) or tuple(row.get("label") for row in rows) != EVALUATION_LABEL_IDS:
        raise RuntimeError(f"{label} has the wrong per-label order")
    values: dict[int, float] = {}
    for row, frozen in zip(rows, baseline, strict=True):
        baseline_dice = _dice(row.get("baseline_dice"), f"{label} baseline label Dice")
        candidate_dice = _dice(row.get(candidate_key), f"{label} returned label Dice")
        delta = _finite(row.get("dice_delta"), f"{label} label Dice delta")
        if not math.isclose(baseline_dice, float(frozen["dice"]), rel_tol=0.0, abs_tol=1e-12) or not math.isclose(
            delta, candidate_dice - baseline_dice, rel_tol=0.0, abs_tol=1e-12
        ):
            raise RuntimeError(f"{label} per-label Dice arithmetic or baseline changed")
        values[int(row["label"])] = candidate_dice
    return values


def validate_evaluation_case_marker(
    payload: Mapping[str, Any],
    contract: Mapping[str, Any],
    decision_contract_sha256: str,
    barrier: Mapping[str, Any],
    barrier_sha256: str,
    evaluation_contract: Mapping[str, Any],
    evaluation_contract_sha256: str,
    decision_case: Mapping[str, Any],
    decision_case_sha256: str,
) -> None:
    case_id = payload.get("case_id")
    if case_id not in contract.get("case_ids", []):
        raise RuntimeError("C5 evaluation marker belongs to a foreign case")
    if (
        payload.get("schema") != EVALUATION_CASE_SCHEMA
        or payload.get("protocol_id") != PROTOCOL_ID
        or payload.get("status") != "COMPLETE"
        or payload.get("decision_contract_sha256") != decision_contract_sha256
        or payload.get("decision_barrier_sha256") != _require_sha(barrier_sha256, "C5 decision barrier")
        or payload.get("evaluation_contract_sha256")
        != _require_sha(evaluation_contract_sha256, "C5 evaluation contract")
        or payload.get("decision_case_sha256") != (barrier.get("decision_case_sha256") or {}).get(case_id)
        or payload.get("labels_loaded_after_barrier") is not True
        or payload.get("test_split_accessed") is not False
        or tuple(payload.get("labels") or ()) != EVALUATION_LABEL_IDS
        or evaluation_contract.get("decision_barrier_sha256") != barrier_sha256
    ):
        raise RuntimeError(f"Invalid C5 post-barrier evaluation marker: {case_id}")
    if (
        decision_case.get("case_id") != case_id
        or decision_case.get("decision_contract_sha256") != decision_contract_sha256
        or _require_sha(decision_case_sha256, "C5 decision case")
        != (barrier.get("decision_case_sha256") or {}).get(case_id)
    ):
        raise RuntimeError(f"C5 evaluation is bound to a foreign decision case: {case_id}")
    baseline_expected = _dice(
        evaluation_contract["evaluation_baseline_dice"][case_id], f"C5 {case_id} frozen baseline Dice"
    )
    baseline_labels = evaluation_contract["evaluation_baseline_per_label"][case_id]
    arms = payload.get("arms")
    specs = contract["arm_specs"]
    if not isinstance(arms, list) or [row.get("arm_id") for row in arms] != [row["arm_id"] for row in specs]:
        raise RuntimeError(f"C5 evaluation marker has the wrong arm order: {case_id}")
    evaluated_by_arm: dict[str, tuple[float, dict[int, float]]] = {}
    source_anchor_map = dict(zip(HISTORICAL_ANCHOR_ARM_IDS, SOURCE_C4_ANCHOR_IDS, strict=True))
    for row, spec in zip(arms, specs, strict=True):
        if row.get("arm_index") != spec["arm_index"]:
            raise RuntimeError(f"C5 evaluation arm index changed: {case_id}/{spec['arm_id']}")
        baseline = _dice(row.get("baseline_dice"), f"C5 {case_id}/{spec['arm_id']} baseline Dice")
        candidate = _dice(row.get("candidate_dice"), f"C5 {case_id}/{spec['arm_id']} candidate Dice")
        delta = _finite(row.get("capacity_dice_delta"), f"C5 {case_id}/{spec['arm_id']} Dice delta")
        if not math.isclose(baseline, baseline_expected, rel_tol=0.0, abs_tol=1e-8) or not math.isclose(
            delta, candidate - baseline, rel_tol=0.0, abs_tol=1e-12
        ):
            raise RuntimeError(f"C5 evaluation Dice arithmetic or baseline changed: {case_id}/{spec['arm_id']}")
        per_label = _validate_per_label_dice(
            row.get("per_label"),
            baseline=baseline_labels,
            candidate_key="candidate_dice",
            label=f"C5 {case_id}/{spec['arm_id']}",
        )
        source_anchor_id = source_anchor_map.get(spec["arm_id"])
        if source_anchor_id is not None:
            frozen = evaluation_contract["evaluation_c4_anchor_dice"][case_id][source_anchor_id]
            frozen_labels = {int(item["label"]): float(item["dice"]) for item in frozen["per_label"]}
            if (
                not math.isclose(candidate, float(frozen["aggregate_dice"]), rel_tol=0.0, abs_tol=1e-12)
                or per_label != frozen_labels
                or row.get("historical_c4_dice_parity_verified") is not True
            ):
                raise RuntimeError(f"C5 historical anchor Dice changed: {case_id}/{spec['arm_id']}")
        evaluated_by_arm[spec["arm_id"]] = (candidate, per_label)

    decision_selectors = {row["selector_id"]: row for row in decision_case["selectors"]}
    selectors = payload.get("selectors")
    if not isinstance(selectors, list) or [row.get("selector_id") for row in selectors] != list(SELECTOR_IDS):
        raise RuntimeError(f"C5 evaluation marker has the wrong selector order: {case_id}")
    baseline_label_map = {int(item["label"]): float(item["dice"]) for item in baseline_labels}
    for index, row in enumerate(selectors):
        decision_row = decision_selectors[row["selector_id"]]
        if (
            row.get("selector_index") != index
            or row.get("action") != decision_row["action"]
            or row.get("selected_arm_id") != decision_row["selected_arm_id"]
        ):
            raise RuntimeError(f"C5 evaluation changed a label-free selector: {case_id}/{row['selector_id']}")
        selected = decision_row["selected_arm_id"]
        expected_dice, expected_labels = (
            (baseline_expected, baseline_label_map) if selected is None else evaluated_by_arm[selected]
        )
        returned = _dice(row.get("returned_dice"), f"C5 {case_id}/{row['selector_id']} returned Dice")
        delta = _finite(row.get("dice_delta"), f"C5 {case_id}/{row['selector_id']} returned Dice delta")
        per_label = _validate_per_label_dice(
            row.get("per_label"),
            baseline=baseline_labels,
            candidate_key="returned_dice",
            label=f"C5 {case_id}/{row['selector_id']}",
        )
        if (
            not math.isclose(returned, expected_dice, rel_tol=0.0, abs_tol=1e-12)
            or per_label != expected_labels
            or not math.isclose(delta, returned - baseline_expected, rel_tol=0.0, abs_tol=1e-12)
        ):
            raise RuntimeError(f"C5 selector Dice does not match its frozen return: {case_id}/{row['selector_id']}")
