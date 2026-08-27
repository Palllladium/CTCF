from __future__ import annotations

import csv
import hashlib
import json
import math
import re
from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np

from tools.analysis.search_gate_c5b import SCHEMA_VERSION, validate_c5b_geometry_bundle

FROZEN_C5_RUN_ID = "C5_DEVELOPMENT_20260825T175112Z_242dde3281d2"
FROZEN_C5_PROTOCOL_ID = "CTCF-SEARCH-GATE-C5-V1"
FROZEN_C5_MANIFEST_SHA256 = "2fad4990d1b557101f9eb93312c3d2705f79f81a6d46753bdcdd7da879a65027"
FROZEN_C5_RUN_MANIFEST_SHA256 = "094860ab5ff2ef4c5c411a5daa065b4aecd6a98a715d0add060be8cb4fdc8670"
FROZEN_C5_SOURCE_SHA256 = "6b3607681d39486358233ac33ee73cd50bcadc859c09d185b78c56a6d2450eea"
FROZEN_C5_DECISION_SHA256 = "9c60a2718d3af126ca050766535876e3690896545ecfe0d54ca113ef53ceff77"
FROZEN_C5_BARRIER_SHA256 = "5ecc26975df37c5c6e6bdb31919b133c2ef94dff7c5ce1d587885705d0940cc7"
FROZEN_C5_POST_BARRIER_SHA256 = "aeeb0274d4fe6f50c8cc3c021c10f5637ecb8a2977cd00f78a1870feada93f4b"
FROZEN_C5_PRODUCER_HEAD = "242dde3281d22c573a4a64bc494c8d2e5ef597b2"
FROZEN_C5_CONSUMER_HEAD = "baafc3bcfed74ed3e471222390e3a604816ae4a3"
FROZEN_C5_TRANSITION = "POST_BARRIER_RECOVERY"
FROZEN_C5_BRANCH = "OPEN_C5_CLIPPING_SATURATION"
EXPECTED_CASE_COUNT = 58

FROZEN_C5_OUTPUT_SHA256 = {
    "arm_summary.csv": "4a0cc939d54fec8b3e51d2e04242912d426620a0d834435a9da68d2cf51f166d",
    "diagnostic_utilities.csv": "98075a788847c2c5b55c6c36af85ab1ca1b83caa51ed097a33c19820cd8752e5",
    "hypotheses.json": "bd77f252a76311572e2d14f2bedfc88e227e578d8c87962d4dc3c431a2b3849c",
    "next_branch.json": "b35008c32d929117521c965981a2219e3266e7d7d9921c6017551c40e98e103d",
    "per_arm.csv": "3d77275d1dcb8f9f9726bb21996bb42d8b1ef56c6ffbd356f638cb77ca6d91a6",
    "per_arm_label_dice.csv": "dae0b60fdb6e4520d17bae035a3d7c10a29a7860da967cbb96f374eceb239a15",
    "per_selector.csv": "912e5ee7f7c076aa29094fbd3fa090342f8b278c3d2b648681cb48fc70f6fdb1",
    "per_selector_label_dice.csv": "b82505ba0209ecbd43716ad8306dd74e33adabaa8ce4104335575c13c3d69186",
    "preregistered_contrasts.csv": "a7bd96bb99aa34161441df5f4e7d34e3aaff5f18a65b4f5ca0e72aacce903660",
    "resource_summary.csv": "ac9e71048c645eb2066a5f8b5e5791274fe1aa34e48a4b9cdb0c9f8af162f8a5",
    "selector_summary.csv": "736686b4e41531d34bccfccab219b80cf04bb824499aa782e0af871b41e1bc42",
    "summary.json": "776cc59a3580c489ecca009b7d6b19d997ceba33ed6b1b726193192db6492267",
}

_MANIFEST_FILE_KEYS = {Path(name).stem: digest for name, digest in FROZEN_C5_OUTPUT_SHA256.items()}
_ROOT_IDS = ("source_c3_heavy", "source_c4_heavy", "source_c5_heavy")
_SHA256 = re.compile(r"[0-9a-f]{64}")
_ANCHORS = {
    "c4_reference_s2_a10_b0": ("int_s2_a10_b0", "source_c4_heavy", 2, 1.0),
    "c5_s4_a10_b0_sweep1": ("int_s4_a10_b0", "source_c5_heavy", 4, 1.0),
    "c5_s4_a20_b0_sweep1": ("int_s4_a20_b0", "source_c5_heavy", 4, 2.0),
}
_FORBIDDEN_PROJECTION_TOKENS = ("label", "evaluation", "dice", "segmentation", "raw_input", "compact")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_sha(value: Any, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise RuntimeError(f"{label} is not a lowercase SHA-256")
    return value


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError(f"Cannot read authenticated C5 JSON: {path}") from error
    if not isinstance(value, dict):
        raise RuntimeError(f"Authenticated C5 JSON must contain an object: {path}")
    return value


def _require_file(root: Path, relative_path: str, expected_sha256: str) -> Path:
    relative = PurePosixPath(relative_path)
    if relative.is_absolute() or not relative.parts or any(part in {"", ".", ".."} for part in relative.parts):
        raise RuntimeError(f"Unsafe C5 compact relative path: {relative_path}")
    path = root.joinpath(*relative.parts)
    try:
        path.resolve(strict=False).relative_to(root.resolve(strict=False))
    except ValueError as error:
        raise RuntimeError(f"C5 compact path escapes its root: {relative_path}") from error
    if not path.is_file() or _sha256_file(path) != _require_sha(expected_sha256, relative_path):
        raise RuntimeError(f"C5 compact bytes changed: {relative_path}")
    return path


def _canonical_root(value: str | Path) -> Path:
    return Path(value).resolve(strict=False)


def _validate_roots(
    recorded: Mapping[str, Any],
    c5_heavy_root: Path,
    *,
    require_exists: bool,
) -> dict[str, Path]:
    if set(recorded) != {"source_c3_heavy", "source_c4_heavy", "target_c5_heavy"}:
        raise RuntimeError("Frozen C5 source must bind exactly its three heavy roots")
    roots = {
        "source_c3_heavy": _canonical_root(str(recorded["source_c3_heavy"])),
        "source_c4_heavy": _canonical_root(str(recorded["source_c4_heavy"])),
        "source_c5_heavy": _canonical_root(c5_heavy_root),
    }
    if roots["source_c5_heavy"] != _canonical_root(str(recorded["target_c5_heavy"])):
        raise RuntimeError("Explicit C5 heavy root differs from the authenticated C5 root")
    if len(set(roots.values())) != len(roots):
        raise RuntimeError("C5b source heavy roots must be distinct")
    for left_name, left in roots.items():
        for right_name, right in roots.items():
            if left_name >= right_name:
                continue
            if left in right.parents or right in left.parents:
                raise RuntimeError("C5b source heavy roots must not overlap")
    if require_exists:
        missing = [name for name, path in roots.items() if not path.is_dir()]
        if missing:
            raise RuntimeError(f"Authenticated C5 heavy roots are absent: {missing}")
    return roots


def _relative_path(record: Mapping[str, Any], label: str) -> PurePosixPath:
    value = record.get("relative_path")
    if not isinstance(value, str):
        raise RuntimeError(f"{label} lacks a relative path")
    relative = PurePosixPath(value)
    if relative.is_absolute() or not relative.parts or any(part in {"", ".", ".."} for part in relative.parts):
        raise RuntimeError(f"{label} escapes its declared heavy root")
    return relative


def _retag_record(record: Mapping[str, Any], root_id: str) -> dict[str, Any]:
    allowed = {"relative_path", "sha256", "npz_sha256", "array_sha256", "bytes", "dtype", "shape"}
    result = {key: deepcopy(value) for key, value in record.items() if key in allowed}
    result["root_id"] = root_id
    _relative_path(result, "C5b projected record")
    if "sha256" in result:
        _require_sha(result["sha256"], "C5b projected file")
    if "npz_sha256" in result:
        _require_sha(result["npz_sha256"], "C5b projected NPZ")
    _require_sha(result.get("array_sha256"), "C5b projected array")
    return result


def _array_sha256(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes(order="C")).hexdigest()


def _verify_heavy_record(record: Mapping[str, Any], roots: Mapping[str, Path], label: str) -> None:
    root_id = record.get("root_id")
    if root_id not in roots:
        raise RuntimeError(f"{label} has an unknown heavy-root owner")
    relative = _relative_path(record, label)
    root = roots[str(root_id)]
    path = root.joinpath(*relative.parts)
    try:
        path.resolve(strict=False).relative_to(root)
    except ValueError as error:
        raise RuntimeError(f"{label} escapes its declared heavy root") from error
    expected_file_sha = record.get("npz_sha256", record.get("sha256"))
    if not path.is_file() or _sha256_file(path) != _require_sha(expected_file_sha, f"{label} file"):
        raise RuntimeError(f"{label} heavy bytes changed or are absent")
    if "bytes" in record and path.stat().st_size != int(record["bytes"]):
        raise RuntimeError(f"{label} byte count changed")
    try:
        if path.suffix == ".npy":
            array = np.load(path, allow_pickle=False)
        elif path.suffix == ".npz":
            with np.load(path, allow_pickle=False) as archive:
                if archive.files != ["flow"]:
                    raise RuntimeError(f"{label} NPZ inventory changed")
                array = np.asarray(archive["flow"])
        else:
            raise RuntimeError(f"{label} uses an unsupported heavy format")
    except (OSError, ValueError) as error:
        raise RuntimeError(f"Cannot load authenticated heavy array: {label}") from error
    if _array_sha256(np.asarray(array)) != _require_sha(record.get("array_sha256"), f"{label} array"):
        raise RuntimeError(f"{label} array bytes changed")


def _validate_native_inventory(root: Path, manifest: Mapping[str, Any]) -> None:
    native_names = {
        "commands_sha256": "commands.sh",
        "datasets_sha256": "datasets.tsv",
        "environment_sha256": "environment.txt",
        "git_status_sha256": "git_status.txt",
        "outputs_sha256": "outputs.tsv",
    }
    files = manifest.get("files")
    if not isinstance(files, Mapping) or set(files) != set(native_names):
        raise RuntimeError("Frozen C5 native-manifest file inventory changed")
    for key, name in native_names.items():
        _require_file(root, name, str(files[key]))

    outputs_path = root / "outputs.tsv"
    with outputs_path.open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))
    if not rows or set(rows[0]) != {"relative_path", "bytes", "sha256"}:
        raise RuntimeError("Frozen C5 outputs.tsv schema changed")
    seen: set[str] = set()
    for row in rows:
        relative = str(row["relative_path"])
        if relative in seen:
            raise RuntimeError(f"Duplicate C5 compact output: {relative}")
        seen.add(relative)
        path = _require_file(root, relative, str(row["sha256"]))
        try:
            expected_bytes = int(row["bytes"])
        except (TypeError, ValueError) as error:
            raise RuntimeError(f"Invalid C5 output byte count: {relative}") from error
        if expected_bytes < 0 or path.stat().st_size != expected_bytes:
            raise RuntimeError(f"C5 compact output byte count changed: {relative}")
    observed = {path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file()}
    if observed != seen | {"outputs.tsv", "run_manifest.json"}:
        raise RuntimeError("C5 compact directory has missing or unexpected files")


def _validate_frozen_manifests(root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    c5_path = _require_file(root, "c5_manifest.json", FROZEN_C5_MANIFEST_SHA256)
    run_path = _require_file(root, "run_manifest.json", FROZEN_C5_RUN_MANIFEST_SHA256)
    c5 = _load_json(c5_path)
    run = _load_json(run_path)
    provenance = c5.get("code_provenance") or {}
    branch = c5.get("next_branch") or {}
    if (
        c5.get("schema") != "ctcf-search-c5-run-manifest-v1"
        or c5.get("protocol_id") != FROZEN_C5_PROTOCOL_ID
        or c5.get("status") != "COMPLETE"
        or c5.get("test_115_authorized") is not False
        or c5.get("test_split_accessed") is not False
        or provenance.get("decision_git_head") != FROZEN_C5_PRODUCER_HEAD
        or provenance.get("git_head") != FROZEN_C5_CONSUMER_HEAD
        or provenance.get("transition") != FROZEN_C5_TRANSITION
        or branch.get("branch_id") != FROZEN_C5_BRANCH
        or branch.get("selected_arm_id") is not None
        or branch.get("selected_selector_id") is not None
    ):
        raise RuntimeError("Frozen successful C5 scientific manifest changed")
    manifest_files = c5.get("files")
    if not isinstance(manifest_files, Mapping) or dict(manifest_files) != _MANIFEST_FILE_KEYS:
        raise RuntimeError("Frozen C5 scientific output inventory changed")
    for name, digest in FROZEN_C5_OUTPUT_SHA256.items():
        _require_file(root, name, digest)
    if (
        run.get("schema") != "ctcf-native-manifest-v1"
        or run.get("run_id") != FROZEN_C5_RUN_ID
        or run.get("status") != "COMPLETE"
        or run.get("exit_code") != 0
        or (run.get("code") or {}).get("git_head") != FROZEN_C5_CONSUMER_HEAD
        or (run.get("code") or {}).get("tracked_tree_clean_at_start") is not True
        or (run.get("execution") or {}).get("mode") != "development"
        or (run.get("execution") or {}).get("seed") != 0
    ):
        raise RuntimeError("Frozen successful C5 native manifest changed")
    _validate_native_inventory(root, run)
    return c5, run


def _validate_frozen_contracts(root: Path, c5: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    expected = {
        "source_contract.json": FROZEN_C5_SOURCE_SHA256,
        "decision_contract.json": FROZEN_C5_DECISION_SHA256,
        "decision_barrier.json": FROZEN_C5_BARRIER_SHA256,
        "evaluation_contract.json": FROZEN_C5_POST_BARRIER_SHA256,
    }
    loaded = {name: _load_json(_require_file(root, name, digest)) for name, digest in expected.items()}
    source = loaded["source_contract.json"]
    decision = loaded["decision_contract.json"]
    barrier = loaded["decision_barrier.json"]
    post_barrier = loaded["evaluation_contract.json"]
    if (
        c5.get("source_contract_sha256") != FROZEN_C5_SOURCE_SHA256
        or c5.get("decision_contract_sha256") != FROZEN_C5_DECISION_SHA256
        or c5.get("decision_barrier_sha256") != FROZEN_C5_BARRIER_SHA256
        or c5.get("evaluation_contract_sha256") != FROZEN_C5_POST_BARRIER_SHA256
        or source.get("git_head") != FROZEN_C5_PRODUCER_HEAD
        or source.get("test_115_authorized") is not False
        or source.get("ixi_test_split_accessed") is not False
        or decision.get("git_head") != FROZEN_C5_PRODUCER_HEAD
        or decision.get("source_contract_sha256") != FROZEN_C5_SOURCE_SHA256
        or decision.get("decision_contract_contains_label_data") is not False
        or decision.get("labels_available_to_decision_workers") is not False
        or decision.get("decision_worker_uses_raw_containers") is not False
        or decision.get("test_115_authorized") is not False
        or decision.get("ixi_test_split_accessed") is not False
        or barrier.get("status") != "COMPLETE"
        or barrier.get("decision_contract_sha256") != FROZEN_C5_DECISION_SHA256
        or barrier.get("decision_workers_received_label_inputs") is not False
        or barrier.get("test_split_accessed") is not False
        or post_barrier.get("status") != "FROZEN_AFTER_DECISION_BARRIER"
        or post_barrier.get("test_115_authorized") is not False
        or post_barrier.get("ixi_test_split_accessed") is not False
        or (post_barrier.get("evaluation_code") or {}).get("decision_git_head") != FROZEN_C5_PRODUCER_HEAD
        or (post_barrier.get("evaluation_code") or {}).get("git_head") != FROZEN_C5_CONSUMER_HEAD
        or (post_barrier.get("evaluation_code") or {}).get("transition") != FROZEN_C5_TRANSITION
    ):
        raise RuntimeError("Frozen C5 source, barrier, or post-barrier contract changed")
    return source, decision


def _validate_case_markers(
    root: Path,
    c5: Mapping[str, Any],
    decision: Mapping[str, Any],
) -> tuple[list[str], dict[str, dict[str, Any]]]:
    case_ids = decision.get("case_ids")
    if (
        not isinstance(case_ids, list)
        or len(case_ids) != EXPECTED_CASE_COUNT
        or len(set(case_ids)) != EXPECTED_CASE_COUNT
        or any(not isinstance(case_id, str) or not case_id.startswith("subject_") for case_id in case_ids)
    ):
        raise RuntimeError("Frozen C5 case inventory changed")
    decision_hashes = c5.get("decision_case_sha256")
    post_hashes = c5.get("evaluation_case_sha256")
    if not isinstance(decision_hashes, Mapping) or set(decision_hashes) != set(case_ids):
        raise RuntimeError("Frozen C5 decision-marker inventory changed")
    if not isinstance(post_hashes, Mapping) or set(post_hashes) != set(case_ids):
        raise RuntimeError("Frozen C5 post-barrier marker inventory changed")
    markers: dict[str, dict[str, Any]] = {}
    for case_id in case_ids:
        decision_path = _require_file(
            root,
            f"cases/{case_id}/decision_complete.json",
            str(decision_hashes[case_id]),
        )
        post_path = _require_file(
            root,
            f"cases/{case_id}/evaluation_complete.json",
            str(post_hashes[case_id]),
        )
        marker = _load_json(decision_path)
        post = _load_json(post_path)
        if (
            marker.get("schema") != "ctcf-search-c5-decision-case-v1"
            or marker.get("protocol_id") != FROZEN_C5_PROTOCOL_ID
            or marker.get("status") != "COMPLETE"
            or marker.get("case_id") != case_id
            or marker.get("decision_contract_sha256") != FROZEN_C5_DECISION_SHA256
            or marker.get("labels_loaded_to_device") is not False
            or marker.get("test_split_accessed") is not False
        ):
            raise RuntimeError(f"Frozen C5 decision marker changed: {case_id}")
        if (
            post.get("schema") != "ctcf-search-c5-evaluation-case-v1"
            or post.get("protocol_id") != FROZEN_C5_PROTOCOL_ID
            or post.get("status") != "COMPLETE"
            or post.get("case_id") != case_id
            or post.get("decision_contract_sha256") != FROZEN_C5_DECISION_SHA256
            or post.get("decision_barrier_sha256") != FROZEN_C5_BARRIER_SHA256
            or post.get("evaluation_contract_sha256") != FROZEN_C5_POST_BARRIER_SHA256
            or post.get("decision_case_sha256") != decision_hashes[case_id]
            or post.get("labels_loaded_after_barrier") is not True
            or post.get("test_split_accessed") is not False
        ):
            raise RuntimeError(f"Frozen C5 post-barrier marker changed: {case_id}")
        markers[case_id] = marker
    return list(case_ids), markers


def _anchor_from_marker(
    marker: Mapping[str, Any],
    case_id: str,
    arm_id: str,
    expected_root: str,
    stride: int,
    amplitude: float,
) -> tuple[dict[str, Any], Any]:
    arms = marker.get("arms")
    if not isinstance(arms, list):
        raise RuntimeError("Frozen C5 decision marker lacks its arm inventory")
    matches = [row for row in arms if isinstance(row, Mapping) and row.get("arm_id") == arm_id]
    if len(matches) != 1:
        raise RuntimeError(f"Frozen C5 anchor inventory changed: {arm_id}")
    row = matches[0]
    field = row.get("candidate_field") or {}
    proposal = row.get("proposal") or {}
    operator = row.get("operator") or {}
    exact = row.get("exact") or {}
    persistence = row.get("persistence") or {}
    source_root = "target_c5_heavy" if expected_root == "source_c5_heavy" else expected_root
    if (
        field.get("root_id") != source_root
        or proposal.get("reach_id") != f"S{stride}"
        or proposal.get("stride_voxels") != stride
        or not math.isclose(float(proposal.get("post_rms_amplitude", math.nan)), amplitude)
        or not math.isclose(float(proposal.get("centre_beta", math.nan)), 0.0)
        or operator.get("operator") != "CERTIFIED_LOCAL_CLIP"
        or operator.get("sweeps") != 1
        or not math.isclose(float(operator.get("work_eps", math.nan)), 0.0011)
        or exact.get("status") != "CERTIFIED"
        or exact.get("certified") is not True
        or exact.get("sha256") != field.get("array_sha256")
        or persistence.get("owner") != source_root
        or persistence.get("saved_npz_sha256") != field.get("npz_sha256")
        or persistence.get("reloaded_array_sha256") != field.get("array_sha256")
    ):
        raise RuntimeError(f"Frozen C5 anchor provenance changed: {case_id}/{arm_id}")
    diagnostics = validate_c5b_geometry_bundle(row.get("geometry"), f"frozen C5/{case_id}/{arm_id}")
    return _retag_record(field, expected_root), diagnostics


def assert_c5b_decision_projection_is_label_free(payload: Mapping[str, Any]) -> None:
    stack: list[tuple[tuple[str, ...], Any]] = [((), payload)]
    while stack:
        path, value = stack.pop()
        if isinstance(value, Mapping):
            for key, child in value.items():
                token = str(key).lower()
                if any(forbidden in token for forbidden in _FORBIDDEN_PROJECTION_TOKENS):
                    raise RuntimeError(f"C5b decision source leaks forbidden metadata: {'.'.join((*path, token))}")
                stack.append(((*path, token), child))
        elif isinstance(value, (list, tuple)):
            stack.extend(((*path, str(index)), child) for index, child in enumerate(value))
        elif isinstance(value, str) and value.lower().endswith(".pkl"):
            raise RuntimeError(f"C5b decision source leaks a raw container: {'.'.join(path)}")


def authenticate_c5_source(
    compact_dir: str | Path,
    c5_heavy_root: str | Path,
    *,
    verify_heavy_bytes: bool = True,
) -> dict[str, Any]:
    root = Path(compact_dir).resolve(strict=False)
    if not root.is_dir():
        raise RuntimeError(f"Frozen successful C5 compact directory is absent: {root}")
    c5, _ = _validate_frozen_manifests(root)
    source, decision = _validate_frozen_contracts(root, c5)
    if source.get("case_ids") != decision.get("case_ids") or source.get("roots") != decision.get("roots"):
        raise RuntimeError("Frozen C5 source and decision projections disagree")
    roots = _validate_roots(
        source.get("roots") or {},
        Path(c5_heavy_root),
        require_exists=verify_heavy_bytes,
    )
    case_ids, markers = _validate_case_markers(root, c5, decision)
    expected_images = {"atlas", *case_ids}
    if set(decision.get("image_inputs") or {}) != expected_images:
        raise RuntimeError("Frozen C5 image-input inventory changed")
    for name in ("source_initial", "source_historical", "source_c4_anchors"):
        if set(decision.get(name) or {}) != set(case_ids):
            raise RuntimeError(f"Frozen C5 {name} inventory changed")

    image_inputs = {
        key: _retag_record(value, "source_c3_heavy") for key, value in (decision.get("image_inputs") or {}).items()
    }
    source_initial: dict[str, dict[str, Any]] = {}
    source_rms: dict[str, dict[str, Any]] = {}
    source_anchors: dict[str, dict[str, Any]] = {}
    anchor_geometry = []
    unique_fields: set[tuple[str, str]] = set()
    for case_id in case_ids:
        initial_payload = decision["source_initial"][case_id] or {}
        initial = _retag_record(initial_payload.get("field") or {}, "source_c3_heavy")
        if (initial_payload.get("exact") or {}).get("certified") is not True or (
            initial_payload.get("exact") or {}
        ).get("sha256") != initial.get("array_sha256"):
            raise RuntimeError(f"Frozen C5 initial exact certificate changed: {case_id}")
        historical = decision["source_historical"][case_id] or {}
        rms = _retag_record(historical.get("raw_conf_requested_field") or {}, "source_c3_heavy")
        source_initial[case_id] = {
            "field": initial,
            "origin_marker_sha256": _require_sha(
                initial_payload.get("source_decision_case_sha256"),
                f"{case_id} initial origin",
            ),
        }
        source_rms[case_id] = {
            "field": rms,
            "origin_marker_sha256": _require_sha(
                historical.get("source_decision_case_sha256"),
                f"{case_id} RMS origin",
            ),
        }
        anchors = {}
        for name, (arm_id, owner, stride, amplitude) in _ANCHORS.items():
            field, diagnostics = _anchor_from_marker(markers[case_id], case_id, arm_id, owner, stride, amplitude)
            anchor_geometry.append(diagnostics)
            anchors[name] = {
                "field": field,
                "parent_marker_sha256": _require_sha(
                    (c5.get("decision_case_sha256") or {}).get(case_id),
                    f"{case_id} C5 decision marker",
                ),
                "stride_voxels": stride,
                "post_rms_amplitude": amplitude,
                "centre_beta": 0.0,
                "clip_sweeps": 1,
            }
        frozen_c4 = (decision["source_c4_anchors"][case_id] or {}).get("intensity_s2") or {}
        if anchors["c4_reference_s2_a10_b0"]["field"] != _retag_record(frozen_c4.get("field") or {}, "source_c4_heavy"):
            raise RuntimeError(f"Frozen C4 reference differs inside C5: {case_id}")
        source_anchors[case_id] = anchors
        for payload in (source_initial[case_id], source_rms[case_id], *anchors.values()):
            field = payload["field"]
            identity = (str(field["root_id"]), str(field["relative_path"]))
            if identity in unique_fields and payload not in (source_initial[case_id], source_rms[case_id]):
                raise RuntimeError("Frozen C5 anchor paths are not unique")
            unique_fields.add(identity)

    projection = {
        "schema": f"ctcf-search-c5b-decision-source-{SCHEMA_VERSION}",
        "source_protocol_id": FROZEN_C5_PROTOCOL_ID,
        "source_identity": {
            "run_id": FROZEN_C5_RUN_ID,
            "scientific_manifest_sha256": FROZEN_C5_MANIFEST_SHA256,
            "native_manifest_sha256": FROZEN_C5_RUN_MANIFEST_SHA256,
            "source_contract_sha256": FROZEN_C5_SOURCE_SHA256,
            "decision_contract_sha256": FROZEN_C5_DECISION_SHA256,
            "decision_barrier_sha256": FROZEN_C5_BARRIER_SHA256,
            "producer_git_head": FROZEN_C5_PRODUCER_HEAD,
            "consumer_git_head": FROZEN_C5_CONSUMER_HEAD,
            "code_transition": FROZEN_C5_TRANSITION,
            "terminal_branch": FROZEN_C5_BRANCH,
        },
        "case_ids": case_ids,
        "seed": 0,
        "runtime_signature": deepcopy(decision.get("runtime_signature")),
        "roots": {name: str(path) for name, path in roots.items()},
        "image_inputs": image_inputs,
        "source_initial": source_initial,
        "source_rms": source_rms,
        "source_anchors": source_anchors,
        "anchor_geometry_preflight": {
            "validated_anchor_count": len(anchor_geometry),
            "central_invalid_count": sum(row.central_invalid_count for row in anchor_geometry),
            "corner_union_violation_count": sum(row.corner_union_violation_count for row in anchor_geometry),
            "digital_ten_nonzero_anchor_count": sum(
                row.digital_ten_union_violation_count > 0 for row in anchor_geometry
            ),
        },
        "test_115_authorized": False,
        "test_split_accessed": False,
    }
    assert_c5b_decision_projection_is_label_free(projection)
    if verify_heavy_bytes:
        for key, record in image_inputs.items():
            _verify_heavy_record(record, roots, f"image input {key}")
        for case_id in case_ids:
            _verify_heavy_record(source_initial[case_id]["field"], roots, f"{case_id} initial")
            _verify_heavy_record(source_rms[case_id]["field"], roots, f"{case_id} RMS reference")
            for name, anchor in source_anchors[case_id].items():
                _verify_heavy_record(anchor["field"], roots, f"{case_id} {name}")
    return projection


__all__ = [
    "FROZEN_C5_BRANCH",
    "FROZEN_C5_CONSUMER_HEAD",
    "FROZEN_C5_MANIFEST_SHA256",
    "FROZEN_C5_PRODUCER_HEAD",
    "FROZEN_C5_RUN_ID",
    "FROZEN_C5_RUN_MANIFEST_SHA256",
    "FROZEN_C5_TRANSITION",
    "assert_c5b_decision_projection_is_label_free",
    "authenticate_c5_source",
]
