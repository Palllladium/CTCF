from __future__ import annotations

import argparse
import csv
import io
import json
import math
import platform
import shutil
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np
import torch

from tools.analysis.run_artifacts import atomic_write_json, atomic_write_text, rows_to_csv, sha256_file
from tools.analysis.search_gate_c3 import (
    NCC7_WINDOW,
    NCC_DENOMINATOR_EPS,
    build_common_support,
    metric_envelope,
    primary_ncc_decision,
)
from tools.analysis.search_gate_c5_contracts import authenticate_frozen_c4
from tools.analysis.search_gate_c6 import (
    ALL_LABEL_CI_LOW_VS_C4_MIN_STRICT,
    ARM_SPECS,
    C6_POLICY_SHA256,
    COMMON_EVIDENCE_COLLAR,
    EXACT_CLAIM_EPS,
    EXPECTED_CASE_COUNT,
    FINAL_LOCAL_CLIP_SWEEPS,
    IMAGE_NORMALIZATION_STD_FLOOR,
    PROTOCOL_ID,
    REFERENCE_ARM_ID,
    RISK_LABEL_IDS,
    SCHEMA_VERSION,
    SELECTABLE_ARM_IDS,
    TEST_115_AUTHORIZED,
    WORK_EPS,
    assert_frozen_policy,
    assess_arm,
    frozen_construction_kwargs,
    matched_control_ids,
    policy_dict,
    select_branch,
    simultaneous_paired_summaries,
)
from tools.analysis.search_gate_common import git, utc_now
from tools.analysis.search_gate_metrics import (
    DIGITAL_DECOMPOSITION,
    LEARN2REG_SHIFTED_SDLOGJ_MASKED,
    MATHEMATICAL_SDLOGJ_CROP2,
    METRIC_SPECS,
    compute_metric,
)
from tools.analysis.search_gate_pyramid import array_sha256, build_pyramid_direction, direction_record
from tools.analysis.search_gate_runtime import (
    parse_physical_gpus,
    round_robin_shards,
    save_reload_certify,
    shard_gpu_map,
)
from tools.analysis.transactional_search import (
    certified_local_clip_candidate,
    geometry_mask,
    load_flow_npz,
    masked_zscore,
    ncc_loss_from_normalized,
    sample_at_psi,
    valid_sample_mask,
)
from utils import dice_per_label, setup_device

SOURCE_SCHEMA = "ctcf-search-c6-source-v1"
DECISION_SCHEMA = "ctcf-search-c6-decision-v1"
DECISION_CASE_SCHEMA = "ctcf-search-c6-decision-case-v1"
WORKER_SCHEMA = "ctcf-search-c6-worker-v1"
BARRIER_SCHEMA = "ctcf-search-c6-decision-barrier-v1"
EVALUATION_CONTRACT_SCHEMA = "ctcf-search-c6-evaluation-contract-v1"
EVALUATION_CASE_SCHEMA = "ctcf-search-c6-evaluation-case-v1"
EVALUATION_BARRIER_SCHEMA = "ctcf-search-c6-evaluation-barrier-v1"
DEFAULT_MIN_FREE_GIB = 80.0
RESUME_MIN_FREE_GIB = 5.0
DIRECTION_DIAGNOSTIC_FIELDS = (
    "pre_normalization_rms",
    "rematch_gain",
    "normalized_rms",
    "stage_clip_retention_min",
    "stage_clip_retention_mean",
    "final_clip_retained_norm_ratio",
)


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"expected a JSON object: {path}")
    return payload


def _immutable_json(path: Path, payload: dict[str, Any]) -> str:
    if path.exists():
        if _load_json(path) != payload:
            raise FileExistsError(f"refusing to replace immutable C6 artifact: {path}")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_json(path, payload)
    return sha256_file(path)


def _immutable_text(path: Path, text: str) -> None:
    if path.exists():
        if path.read_text(encoding="utf-8") != text:
            raise FileExistsError(f"refusing to replace immutable C6 artifact: {path}")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_text(path, text)


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


def _assert_clean_runtime(decision: Mapping[str, Any], stage: str) -> None:
    if git("rev-parse", "HEAD") != decision["git_head"] or git("status", "--porcelain=v1"):
        raise RuntimeError(f"C6 {stage} code differs from its clean prepared contract")
    observed = _runtime_signature()
    if observed != dict(decision["runtime_signature"]):
        raise RuntimeError(f"C6 {stage} runtime changed: {observed} != {dict(decision['runtime_signature'])}")


def _tree_bytes(root: Path) -> int:
    return sum(path.stat().st_size for path in root.rglob("*") if path.is_file()) if root.exists() else 0


def _validate_disk_budget(target: Path, minimum_free_gib: float) -> None:
    if not math.isfinite(minimum_free_gib) or minimum_free_gib < 0:
        raise ValueError("C6 disk budget must be finite and non-negative")
    target.parent.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(target.parent).free / 2**30
    retained = _tree_bytes(target) / 2**30
    if target.exists():
        if free < RESUME_MIN_FREE_GIB or free + retained < minimum_free_gib:
            raise RuntimeError(
                f"C6 resume lacks disk: free={free:.2f} GiB, retained={retained:.2f} GiB, required={minimum_free_gib:.2f} GiB"
            )
    elif free < minimum_free_gib:
        raise RuntimeError(f"C6 requires {minimum_free_gib:.2f} GiB free; found {free:.2f} GiB")


def _roots(decision: Mapping[str, Any]) -> dict[str, Path]:
    value = decision.get("roots")
    expected = {"source_c3_heavy", "source_c4_heavy", "target_c6_heavy"}
    if not isinstance(value, Mapping) or set(value) != expected:
        raise RuntimeError("C6 rooted-storage inventory changed")
    result = {str(key): Path(str(path)).resolve() for key, path in value.items()}
    if len(set(result.values())) != len(result):
        raise RuntimeError("C6 storage roots must be distinct")
    values = list(result.values())
    if any(
        left in right.parents or right in left.parents
        for index, left in enumerate(values)
        for right in values[index + 1 :]
    ):
        raise RuntimeError("C6 storage roots must not overlap")
    return result


def _relative_path(value: Any) -> PurePosixPath:
    if not isinstance(value, str) or not value or "\\" in value:
        raise RuntimeError("C6 rooted artifact path must be a POSIX relative path")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise RuntimeError("C6 rooted artifact path escapes its root")
    return path


def _verify_record(decision: Mapping[str, Any], record: Mapping[str, Any], *, verify_array: bool = False) -> Path:
    roots = _roots(decision)
    root_id = record.get("root_id")
    if root_id not in roots:
        raise RuntimeError("C6 artifact has an unknown root owner")
    root = roots[str(root_id)]
    relative = _relative_path(record.get("relative_path"))
    path = root.joinpath(*relative.parts).resolve()
    if root not in path.parents:
        raise RuntimeError("C6 rooted artifact escaped its root")
    expected = record.get("npz_sha256", record.get("sha256"))
    if not path.is_file() or sha256_file(path) != expected:
        raise RuntimeError(f"C6 artifact bytes changed or are absent: {path}")
    if verify_array:
        array = load_flow_npz(path)
        if array_sha256(array) != record.get("array_sha256"):
            raise RuntimeError(f"C6 field array changed: {path}")
    return path


def _load_field(decision: Mapping[str, Any], record: Mapping[str, Any]) -> torch.Tensor:
    return load_flow_npz(_verify_record(decision, record, verify_array=True))


def _load_image(decision: Mapping[str, Any], record: Mapping[str, Any]) -> np.ndarray:
    path = _verify_record(decision, record)
    array = np.load(path, allow_pickle=False)
    if (
        array.dtype != np.float32
        or list(array.shape) != record.get("shape")
        or not np.isfinite(array).all()
        or array_sha256(torch.from_numpy(array)) != record.get("array_sha256")
    ):
        raise RuntimeError(f"C6 cached image changed: {path}")
    return np.ascontiguousarray(array)


def _field_record(path: Path, heavy_root: Path, digest: str) -> dict[str, Any]:
    return {
        "root_id": "target_c6_heavy",
        "relative_path": path.resolve().relative_to(heavy_root.resolve()).as_posix(),
        "npz_sha256": sha256_file(path),
        "array_sha256": digest,
    }


def _dataset_tsv(raw_inputs: Mapping[str, Mapping[str, Any]]) -> str:
    fields = ("dataset", "split", "case_id", "path", "bytes", "sha256", "mtime_utc")
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields, delimiter="\t", lineterminator="\n")
    writer.writeheader()
    for record in raw_inputs.values():
        writer.writerow({field: record.get(field, "") for field in fields})
    return stream.getvalue()


def _assert_decision_label_free(payload: Mapping[str, Any]) -> None:
    allowed_flags = {"labels_loaded_to_device", "labels_loaded"}

    def visit(value: Any, path: tuple[str, ...] = ()) -> None:
        if isinstance(value, Mapping):
            for key, child in value.items():
                token = str(key).lower()
                if token in allowed_flags and child is not False:
                    raise RuntimeError(f"C6 label-free flag is not false: {'.'.join((*path, str(key)))}")
                if "dice" in token or "segmentation" in token or ("label" in token and token not in allowed_flags):
                    raise RuntimeError(f"C6 decision contract contains evaluation data: {'.'.join((*path, str(key)))}")
                visit(child, (*path, str(key)))
        elif isinstance(value, (list, tuple)):
            for index, child in enumerate(value):
                visit(child, (*path, str(index)))
        elif isinstance(value, str) and ("segmentation" in value.lower() or value.lower().endswith(".pkl")):
            raise RuntimeError(f"C6 decision contract contains an evaluation locator: {'.'.join(path)}")

    visit(payload)


def prepare_contracts(
    *,
    run_root: Path,
    heavy_root: Path,
    source_c4_dir: Path,
    source_c4_heavy_root: Path,
    physical_gpus: Sequence[str],
) -> tuple[str, str]:
    snapshot = authenticate_frozen_c4(source_c4_dir, source_c4_heavy_root, verify_anchor_bytes=True)
    if len(snapshot["case_ids"]) != EXPECTED_CASE_COUNT:
        raise RuntimeError("C6 source C4 does not contain IXI validation-58")
    source = {
        "schema": SOURCE_SCHEMA,
        "protocol_id": PROTOCOL_ID,
        "policy": policy_dict(),
        "policy_sha256": C6_POLICY_SHA256,
        "source_snapshot": snapshot,
        "test_115_authorized": False,
        "test_split_accessed": False,
    }
    roots = {
        "source_c3_heavy": str(Path(snapshot["source_c3_heavy_root"]).resolve()),
        "source_c4_heavy": str(source_c4_heavy_root.resolve()),
        "target_c6_heavy": str(heavy_root.resolve()),
    }
    case_ids = list(snapshot["case_ids"])
    shards = round_robin_shards(case_ids, len(physical_gpus))
    decision = {
        "schema": DECISION_SCHEMA,
        "protocol_id": PROTOCOL_ID,
        "git_head": git("rev-parse", "HEAD"),
        "runtime_signature": _runtime_signature(),
        "policy": policy_dict(),
        "policy_sha256": C6_POLICY_SHA256,
        "roots": roots,
        "case_ids": case_ids,
        "seed": snapshot["seed"],
        "image_inputs": snapshot["image_inputs"],
        "source_initial": snapshot["source_initial"],
        "source_historical": snapshot["source_historical"],
        "source_c4_anchors": snapshot["source_c4_anchors"],
        "baseline_geometry": snapshot["baseline_geometry"],
        "num_shards": len(physical_gpus),
        "physical_gpus": list(physical_gpus),
        "shard_to_physical_gpu": shard_gpu_map(list(physical_gpus)),
        "shards": shards,
        "labels_loaded_to_device": False,
        "test_115_authorized": False,
        "test_split_accessed": False,
    }
    _roots(decision)
    _assert_decision_label_free(decision)
    source_sha = _immutable_json(run_root / "source_contract.json", source)
    decision["source_contract_sha256"] = source_sha
    decision_sha = _immutable_json(run_root / "decision_contract.json", decision)
    _immutable_text(run_root / "datasets.tsv", _dataset_tsv(snapshot["raw_inputs"]))
    _immutable_text(
        run_root / "heavy_retention.txt",
        "".join(f"{key}={value}\n" for key, value in roots.items())
        + "retention=RETAIN_ALL_THREE_ROOTS_UNTIL_EXPLICIT_OPERATOR_DECISION\npackaged=false\n",
    )
    return source_sha, decision_sha


def _load_decision(run_root: Path, digest: str) -> dict[str, Any]:
    path = run_root / "decision_contract.json"
    if sha256_file(path) != digest:
        raise RuntimeError("C6 decision contract SHA-256 changed")
    payload = _load_json(path)
    case_ids = payload.get("case_ids") or []
    physical_gpus = payload.get("physical_gpus") or []
    if (
        payload.get("schema") != DECISION_SCHEMA
        or payload.get("protocol_id") != PROTOCOL_ID
        or payload.get("policy_sha256") != C6_POLICY_SHA256
        or payload.get("policy") != policy_dict()
        or payload.get("test_115_authorized") is not False
        or payload.get("test_split_accessed") is not False
        or len(case_ids) != EXPECTED_CASE_COUNT
        or len(set(case_ids)) != EXPECTED_CASE_COUNT
        or not physical_gpus
        or len(set(physical_gpus)) != len(physical_gpus)
        or any(not isinstance(value, str) or not value.isdigit() for value in physical_gpus)
        or payload.get("num_shards") != len(physical_gpus)
        or payload.get("shards") != round_robin_shards(case_ids, len(physical_gpus))
        or payload.get("shard_to_physical_gpu") != shard_gpu_map(physical_gpus)
        or set(payload.get("image_inputs") or {}) != {"atlas", *case_ids}
        or set(payload.get("source_initial") or {}) != set(case_ids)
        or set(payload.get("source_historical") or {}) != set(case_ids)
        or set(payload.get("source_c4_anchors") or {}) != set(case_ids)
    ):
        raise RuntimeError("invalid or altered C6 decision contract")
    _roots(payload)
    _assert_decision_label_free(payload)
    return payload


def _decision_case_path(run_root: Path, case_id: str) -> Path:
    return run_root / "cases" / case_id / "decision_complete.json"


def _evaluation_case_path(run_root: Path, case_id: str) -> Path:
    return run_root / "cases" / case_id / "evaluation_complete.json"


def _worker_path(run_root: Path, phase: str, attempt_id: str, shard_index: int) -> Path:
    return run_root / "workers" / phase / "attempts" / attempt_id / f"worker_{shard_index:02d}.json"


def _geometry_bundle(field: torch.Tensor, mask: torch.Tensor) -> dict[str, dict[str, Any]]:
    return {
        metric_id: metric_envelope(
            metric_id,
            lambda mid=metric_id: compute_metric(mid, field, mask if mid == LEARN2REG_SHIFTED_SDLOGJ_MASKED else None),
        ).to_dict()
        for metric_id in METRIC_SPECS
    }


def _require_exact_geometry(bundle: Mapping[str, Any], label: str) -> None:
    if any(not isinstance(row, Mapping) or row.get("status") != "OK" for row in bundle.values()):
        raise RuntimeError(f"C6 geometry metric failed closed: {label}")
    components = bundle[DIGITAL_DECOMPOSITION]["components"]
    if float(components["corner_union_violation_fraction"]) != 0.0:
        raise RuntimeError(f"C6 exact certificate disagrees with digital corner determinants: {label}")


def _utility_action(
    fixed_norm: torch.Tensor,
    moving_norm: torch.Tensor,
    initial: torch.Tensor,
    candidate: torch.Tensor,
    mask: torch.Tensor,
) -> tuple[dict[str, Any], dict[str, Any], str, str]:
    support = build_common_support(
        mask,
        valid_sample_mask(initial),
        valid_sample_mask(candidate),
        window=NCC7_WINDOW,
        utility_id="COMMON_NCC7",
    )
    baseline = ncc_loss_from_normalized(
        fixed_norm, moving_norm, initial, support.pair_mask, win=NCC7_WINDOW, eps=NCC_DENOMINATOR_EPS
    )
    observed = ncc_loss_from_normalized(
        fixed_norm, moving_norm, candidate, support.pair_mask, win=NCC7_WINDOW, eps=NCC_DENOMINATOR_EPS
    )
    decision = primary_ncc_decision(
        exact_certified=True,
        support_retention=support.retention,
        baseline_ncc_loss=baseline,
        candidate_ncc_loss=observed,
    ).to_dict()
    action = "ACCEPT" if decision["accept"] else "ROLLBACK"
    return (
        {
            "utility_id": "COMMON_NCC7",
            "baseline_count": support.baseline_count,
            "pair_count": support.pair_count,
            "retention": support.retention,
        },
        {"baseline_loss": baseline, "candidate_loss": observed, "improvement": baseline - observed},
        action,
        str(decision["reason"]),
    )


def _materialize(
    *,
    case_id: str,
    spec: Any,
    direction: Any,
    initial: torch.Tensor,
    mask: torch.Tensor,
    fixed_norm: torch.Tensor,
    moving_norm: torch.Tensor,
    decision: Mapping[str, Any],
) -> dict[str, Any]:
    requested = direction.displacement * float(spec.amplitude)
    candidate_raw, operator = certified_local_clip_candidate(
        initial, requested, mask, work_eps=WORK_EPS, sweeps=FINAL_LOCAL_CLIP_SWEEPS
    )
    heavy_root = _roots(decision)["target_c6_heavy"]
    path = heavy_root / "cases" / case_id / "arms" / f"{spec.arm_id}.npz"
    stored, exact = save_reload_certify(candidate_raw, path, EXACT_CLAIM_EPS)
    if exact.get("status") != "CERTIFIED" or exact.get("certified") is not True:
        raise RuntimeError(f"C6 save/reload exact certification failed: {case_id}/{spec.arm_id}")
    candidate = stored.to(initial.device)
    geometry = _geometry_bundle(candidate, mask)
    _require_exact_geometry(geometry, f"{case_id}/{spec.arm_id}")
    support, utility, action, reason = _utility_action(fixed_norm, moving_norm, initial, candidate, mask)
    return {
        "arm_index": spec.arm_index,
        "arm_id": spec.arm_id,
        "role": spec.role,
        "selectable": spec.selectable,
        "action": action,
        "reason": reason,
        "family": spec.family,
        "factors": list(spec.factors),
        "amplitude": spec.amplitude,
        "rewarp_between_levels": spec.rewarp_between_levels,
        "direction": direction_record(direction),
        "requested_array_sha256": array_sha256(requested),
        "operator": operator,
        "candidate_field": _field_record(path, heavy_root, str(exact["sha256"])),
        "exact": exact,
        "geometry": geometry,
        "support": support,
        "utility": utility,
    }


def run_decision_case(
    *,
    case_id: str,
    shard_index: int,
    physical_gpu: str,
    run_root: Path,
    decision: Mapping[str, Any],
    decision_sha256: str,
    device: torch.device,
    execution: Mapping[str, Any],
) -> Path:
    marker = _decision_case_path(run_root, case_id)
    if marker.is_file():
        payload = _load_json(marker)
        if payload.get("decision_contract_sha256") != decision_sha256 or payload.get("status") != "COMPLETE":
            raise RuntimeError(f"invalid existing C6 decision marker: {case_id}")
        return marker
    if case_id not in decision["shards"].get(str(shard_index), []):
        raise RuntimeError(f"C6 case belongs to another shard: {case_id}")
    if str(physical_gpu) != str(decision["shard_to_physical_gpu"].get(str(shard_index))):
        raise RuntimeError(f"C6 case belongs to another physical GPU: {case_id}")
    started = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    atlas = torch.from_numpy(_load_image(decision, decision["image_inputs"]["atlas"])).unsqueeze(0).to(device)
    fixed = torch.from_numpy(_load_image(decision, decision["image_inputs"][case_id])).unsqueeze(0).to(device)
    initial = _load_field(decision, decision["source_initial"][case_id]["field"]).to(device)
    historical = _load_field(decision, decision["source_historical"][case_id]["raw_conf_requested_field"]).to(device)
    rms_reference = (historical - initial).float()
    mask = geometry_mask(tuple(initial.shape[-3:]), COMMON_EVIDENCE_COLLAR, device)
    fixed_norm = masked_zscore(fixed, mask, std_floor=IMAGE_NORMALIZATION_STD_FLOOR)
    moving_norm = masked_zscore(atlas, mask, std_floor=IMAGE_NORMALIZATION_STD_FLOOR)

    anchor_record = decision["source_c4_anchors"][case_id]["intensity_s2"]["field"]
    anchor = _load_field(decision, anchor_record).to(device)
    anchor_geometry = _geometry_bundle(anchor, mask)
    _require_exact_geometry(anchor_geometry, f"{case_id}/{REFERENCE_ARM_ID}")
    rows = [
        {
            "arm_index": 0,
            "arm_id": REFERENCE_ARM_ID,
            "role": "FROZEN_REFERENCE",
            "selectable": False,
            "action": "REFERENCE",
            "source_arm_id": "intensity_s2",
            "candidate_field": anchor_record,
            "geometry": anchor_geometry,
        }
    ]
    direction_cache: dict[tuple[str, tuple[int, ...], bool], Any] = {}
    for spec in ARM_SPECS[1:]:
        key = (spec.family, spec.factors, spec.rewarp_between_levels)
        if key not in direction_cache:
            direction_cache[key] = build_pyramid_direction(
                fixed,
                atlas,
                initial,
                rms_reference,
                family=spec.family,
                factors=spec.factors,
                rewarp_between_levels=spec.rewarp_between_levels,
                **frozen_construction_kwargs(),
            )
        rows.append(
            _materialize(
                case_id=case_id,
                spec=spec,
                direction=direction_cache[key],
                initial=initial,
                mask=mask,
                fixed_norm=fixed_norm,
                moving_norm=moving_norm,
                decision=decision,
            )
        )
    if tuple(row["arm_id"] for row in rows) != tuple(spec.arm_id for spec in ARM_SPECS):
        raise RuntimeError("C6 worker arm order changed")
    payload = {
        "schema": DECISION_CASE_SCHEMA,
        "protocol_id": PROTOCOL_ID,
        "status": "COMPLETE",
        "strict": True,
        "case_id": case_id,
        "shard_index": shard_index,
        "physical_gpu": str(physical_gpu),
        "decision_contract_sha256": decision_sha256,
        "labels_loaded_to_device": False,
        "test_split_accessed": False,
        "arms": rows,
        "resource": {
            "wall_sec": time.perf_counter() - started,
            "peak_cuda_bytes": int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0,
        },
        "execution": dict(execution),
    }
    _assert_decision_label_free(payload)
    _immutable_json(marker, payload)
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return marker


def run_decision_worker(
    *,
    shard_index: int,
    physical_gpu: str,
    attempt_id: str,
    run_root: Path,
    decision: Mapping[str, Any],
    decision_sha256: str,
    device: torch.device,
    execution: Mapping[str, Any],
) -> Path:
    case_ids = decision["shards"][str(shard_index)]
    for case_id in case_ids:
        run_decision_case(
            case_id=case_id,
            shard_index=shard_index,
            physical_gpu=physical_gpu,
            run_root=run_root,
            decision=decision,
            decision_sha256=decision_sha256,
            device=device,
            execution=execution,
        )
    marker = _worker_path(run_root, "decision", attempt_id, shard_index)
    payload = {
        "schema": WORKER_SCHEMA,
        "protocol_id": PROTOCOL_ID,
        "status": "COMPLETE",
        "strict": True,
        "phase": "decision",
        "attempt_id": attempt_id,
        "shard_index": shard_index,
        "physical_gpu": str(physical_gpu),
        "case_ids": list(case_ids),
        "case_sha256": {case_id: sha256_file(_decision_case_path(run_root, case_id)) for case_id in case_ids},
        "decision_contract_sha256": decision_sha256,
        "labels_loaded": False,
        "test_split_accessed": False,
        "execution": dict(execution),
    }
    _assert_decision_label_free(payload)
    _immutable_json(marker, payload)
    return marker


def build_decision_barrier(run_root: Path, decision: Mapping[str, Any], decision_sha: str, attempt_id: str) -> str:
    barrier_path = run_root / "decision_barrier.json"
    if barrier_path.is_file():
        digest = sha256_file(barrier_path)
        _load_barrier(run_root, digest, decision_sha)
        return digest
    case_hashes = {}
    for case_id in decision["case_ids"]:
        path = _decision_case_path(run_root, case_id)
        payload = _load_json(path)
        if payload.get("status") != "COMPLETE" or payload.get("decision_contract_sha256") != decision_sha:
            raise RuntimeError(f"invalid C6 decision case at barrier: {case_id}")
        case_hashes[case_id] = sha256_file(path)
    payload = {
        "schema": BARRIER_SCHEMA,
        "protocol_id": PROTOCOL_ID,
        "status": "COMPLETE",
        "strict": True,
        "attempt_id": attempt_id,
        "decision_contract_sha256": decision_sha,
        "decision_case_sha256": case_hashes,
        "labels_loaded_before_barrier": False,
        "test_split_accessed": False,
        "completed_at_utc": utc_now(),
    }
    return _immutable_json(barrier_path, payload)


def _load_barrier(run_root: Path, digest: str, decision_sha: str) -> dict[str, Any]:
    path = run_root / "decision_barrier.json"
    if sha256_file(path) != digest:
        raise RuntimeError("C6 decision barrier SHA-256 changed")
    payload = _load_json(path)
    if (
        payload.get("schema") != BARRIER_SCHEMA
        or payload.get("status") != "COMPLETE"
        or payload.get("decision_contract_sha256") != decision_sha
        or payload.get("labels_loaded_before_barrier") is not False
        or set(payload.get("decision_case_sha256") or {})
        != set(_load_json(run_root / "decision_contract.json").get("case_ids") or [])
    ):
        raise RuntimeError("invalid or altered C6 decision barrier")
    return payload


def freeze_evaluation(
    run_root: Path,
    source_sha: str,
    decision: Mapping[str, Any],
    decision_sha: str,
    barrier_sha: str,
) -> str:
    source_path = run_root / "source_contract.json"
    if sha256_file(source_path) != source_sha:
        raise RuntimeError("C6 source contract SHA-256 changed")
    source = _load_json(source_path)
    _load_barrier(run_root, barrier_sha, decision_sha)
    snapshot = source["source_snapshot"]
    payload = {
        "schema": EVALUATION_CONTRACT_SCHEMA,
        "protocol_id": PROTOCOL_ID,
        "source_contract_sha256": source_sha,
        "decision_contract_sha256": decision_sha,
        "decision_barrier_sha256": barrier_sha,
        "raw_inputs": snapshot["raw_inputs"],
        "evaluation_baseline_dice": snapshot["evaluation_baseline_dice"],
        "evaluation_baseline_per_label": snapshot["evaluation_baseline_per_label"],
        "evaluation_c4_anchor_dice": snapshot["evaluation_c4_anchor_dice"],
        "evaluation_label_ids": snapshot["evaluation_label_ids"],
        "case_ids": snapshot["case_ids"],
        "test_115_authorized": False,
        "test_split_accessed": False,
    }
    return _immutable_json(run_root / "evaluation_contract.json", payload)


def _load_evaluation(
    run_root: Path, digest: str, source_sha: str, decision_sha: str, barrier_sha: str
) -> dict[str, Any]:
    path = run_root / "evaluation_contract.json"
    if sha256_file(path) != digest:
        raise RuntimeError("C6 evaluation contract SHA-256 changed")
    payload = _load_json(path)
    case_ids = payload.get("case_ids") or []
    if (
        payload.get("schema") != EVALUATION_CONTRACT_SCHEMA
        or payload.get("source_contract_sha256") != source_sha
        or payload.get("decision_contract_sha256") != decision_sha
        or payload.get("decision_barrier_sha256") != barrier_sha
        or payload.get("test_115_authorized") is not False
        or len(case_ids) != EXPECTED_CASE_COUNT
        or set(payload.get("raw_inputs") or {}) != {"atlas", *case_ids}
        or set(payload.get("evaluation_baseline_dice") or {}) != set(case_ids)
        or set(payload.get("evaluation_baseline_per_label") or {}) != set(case_ids)
        or set(payload.get("evaluation_c4_anchor_dice") or {}) != set(case_ids)
    ):
        raise RuntimeError("invalid or altered C6 evaluation contract")
    return payload


def _verify_raw(record: Mapping[str, Any]) -> None:
    path = Path(str(record.get("path", ""))).resolve()
    if (
        not path.is_file()
        or path.stat().st_size != int(record.get("bytes", -1))
        or sha256_file(path) != record.get("sha256")
    ):
        raise RuntimeError(f"C6 frozen raw input changed: {path}")


def run_evaluation_case(
    *,
    case_id: str,
    dataset_item: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    labels: Sequence[int],
    run_root: Path,
    decision: Mapping[str, Any],
    decision_sha: str,
    barrier: Mapping[str, Any],
    barrier_sha: str,
    evaluation: Mapping[str, Any],
    evaluation_sha: str,
    device: torch.device,
    execution: Mapping[str, Any],
) -> Path:
    marker = _evaluation_case_path(run_root, case_id)
    if marker.is_file():
        payload = _load_json(marker)
        if payload.get("evaluation_contract_sha256") != evaluation_sha or payload.get("status") != "COMPLETE":
            raise RuntimeError(f"invalid existing C6 evaluation marker: {case_id}")
        return marker
    decision_path = _decision_case_path(run_root, case_id)
    if sha256_file(decision_path) != barrier["decision_case_sha256"][case_id]:
        raise RuntimeError(f"C6 decision snapshot changed before evaluation: {case_id}")
    decision_case = _load_json(decision_path)
    moving_image, fixed_image, moving_seg, fixed_seg = dataset_item
    if array_sha256(moving_image) != decision["image_inputs"]["atlas"]["array_sha256"]:
        raise RuntimeError("C6 evaluation atlas differs from the decision cache")
    if array_sha256(fixed_image) != decision["image_inputs"][case_id]["array_sha256"]:
        raise RuntimeError(f"C6 evaluation image differs from the decision cache: {case_id}")
    labels_tuple = tuple(int(value) for value in labels)
    if labels_tuple != tuple(evaluation["evaluation_label_ids"]):
        raise RuntimeError("C6 IXI label order changed")
    moving_seg = moving_seg.unsqueeze(0).to(device)
    fixed_seg = fixed_seg.unsqueeze(0).to(device)
    initial = _load_field(decision, decision["source_initial"][case_id]["field"]).to(device)
    baseline_labels = dice_per_label(
        sample_at_psi(moving_seg.float(), initial, mode="nearest").long(), fixed_seg.long(), labels_tuple
    )
    baseline = float(baseline_labels.mean())
    if not math.isclose(baseline, float(evaluation["evaluation_baseline_dice"][case_id]), rel_tol=0.0, abs_tol=1e-8):
        raise RuntimeError(f"C6 baseline Dice differs from frozen C4: {case_id}")
    rows = []
    for arm in decision_case["arms"]:
        candidate = _load_field(decision, arm["candidate_field"]).to(device)
        candidate_labels = dice_per_label(
            sample_at_psi(moving_seg.float(), candidate, mode="nearest").long(), fixed_seg.long(), labels_tuple
        )
        candidate_dice = float(candidate_labels.mean())
        returned_labels = candidate_labels if arm["action"] in {"ACCEPT", "REFERENCE"} else baseline_labels
        returned_dice = float(returned_labels.mean())
        source_parity = None
        if arm["arm_id"] == REFERENCE_ARM_ID:
            expected = evaluation["evaluation_c4_anchor_dice"][case_id]["intensity_s2"]
            expected_labels = [float(row["dice"]) for row in expected["per_label"]]
            if not math.isclose(candidate_dice, float(expected["aggregate_dice"]), rel_tol=0.0, abs_tol=1e-12) or any(
                not math.isclose(float(left), right, rel_tol=0.0, abs_tol=1e-12)
                for left, right in zip(candidate_labels, expected_labels, strict=True)
            ):
                raise RuntimeError(f"C6 C4-reference Dice parity failed: {case_id}")
            source_parity = True
        rows.append(
            {
                "arm_index": arm["arm_index"],
                "arm_id": arm["arm_id"],
                "action": arm["action"],
                "baseline_dice": baseline,
                "candidate_dice": candidate_dice,
                "capacity_delta_vs_initial": candidate_dice - baseline,
                "returned_dice": returned_dice,
                "returned_delta_vs_initial": returned_dice - baseline,
                "source_reference_parity_verified": source_parity,
                "per_label": [
                    {
                        "label": label,
                        "baseline_dice": float(base),
                        "candidate_dice": float(candidate_value),
                        "returned_dice": float(returned_value),
                    }
                    for label, base, candidate_value, returned_value in zip(
                        labels_tuple, baseline_labels, candidate_labels, returned_labels, strict=True
                    )
                ],
            }
        )
    if sha256_file(decision_path) != barrier["decision_case_sha256"][case_id]:
        raise RuntimeError(f"C6 decision snapshot changed during evaluation: {case_id}")
    payload = {
        "schema": EVALUATION_CASE_SCHEMA,
        "protocol_id": PROTOCOL_ID,
        "status": "COMPLETE",
        "strict": True,
        "case_id": case_id,
        "decision_contract_sha256": decision_sha,
        "decision_barrier_sha256": barrier_sha,
        "evaluation_contract_sha256": evaluation_sha,
        "decision_case_sha256": barrier["decision_case_sha256"][case_id],
        "labels_loaded_after_barrier": True,
        "test_split_accessed": False,
        "labels": list(labels_tuple),
        "arms": rows,
        "execution": dict(execution),
    }
    _immutable_json(marker, payload)
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return marker


def run_evaluation_worker(
    *,
    shard_index: int,
    physical_gpu: str,
    attempt_id: str,
    run_root: Path,
    decision: Mapping[str, Any],
    decision_sha: str,
    barrier: Mapping[str, Any],
    barrier_sha: str,
    evaluation: Mapping[str, Any],
    evaluation_sha: str,
    device: torch.device,
    execution: Mapping[str, Any],
    dataset_item_for_case: Callable[[str], tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]],
    labels: Sequence[int],
) -> Path:
    case_ids = decision["shards"][str(shard_index)]
    for case_id in case_ids:
        run_evaluation_case(
            case_id=case_id,
            dataset_item=dataset_item_for_case(case_id),
            labels=labels,
            run_root=run_root,
            decision=decision,
            decision_sha=decision_sha,
            barrier=barrier,
            barrier_sha=barrier_sha,
            evaluation=evaluation,
            evaluation_sha=evaluation_sha,
            device=device,
            execution=execution,
        )
    marker = _worker_path(run_root, "evaluation", attempt_id, shard_index)
    payload = {
        "schema": WORKER_SCHEMA,
        "protocol_id": PROTOCOL_ID,
        "status": "COMPLETE",
        "strict": True,
        "phase": "evaluation",
        "attempt_id": attempt_id,
        "shard_index": shard_index,
        "physical_gpu": str(physical_gpu),
        "case_ids": list(case_ids),
        "case_sha256": {case_id: sha256_file(_evaluation_case_path(run_root, case_id)) for case_id in case_ids},
        "decision_contract_sha256": decision_sha,
        "decision_barrier_sha256": barrier_sha,
        "evaluation_contract_sha256": evaluation_sha,
        "labels_loaded": True,
        "test_split_accessed": False,
        "execution": dict(execution),
    }
    _immutable_json(marker, payload)
    return marker


def _direction_diagnostics(row: Mapping[str, Any], label: str) -> dict[str, float | None]:
    """Surface the rematch gain and the clip retention that decide how much amplitude each arm kept."""

    if row["arm_id"] == REFERENCE_ARM_ID:
        return dict.fromkeys(DIRECTION_DIAGNOSTIC_FIELDS)
    direction = row["direction"]
    reference = _finite(direction["reference_rms"], f"{label} reference_rms", positive=True)
    pre = _finite(direction["pre_normalization_rms"], f"{label} pre_normalization_rms", positive=True)
    retentions = [_finite(stage["clip_retention"], f"{label} stage clip_retention") for stage in direction["stages"]]
    if not retentions:
        raise RuntimeError(f"C6 direction has no stages: {label}")
    return {
        "pre_normalization_rms": pre,
        "rematch_gain": reference / pre,
        "normalized_rms": _finite(direction["normalized_rms"], f"{label} normalized_rms", positive=True),
        "stage_clip_retention_min": min(retentions),
        "stage_clip_retention_mean": sum(retentions) / len(retentions),
        "final_clip_retained_norm_ratio": _finite(
            row["operator"]["retained_norm_ratio"], f"{label} retained_norm_ratio"
        ),
    }


def _finite(value: Any, label: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise RuntimeError(f"C6 diagnostic is not a finite number: {label}")
    observed = float(value)
    if positive and observed <= 0.0:
        raise RuntimeError(f"C6 diagnostic must be strictly positive: {label}")
    return observed


def _metric_value(bundle: Mapping[str, Any], metric_id: str, label: str) -> float:
    row = bundle.get(metric_id)
    value = row.get("value") if isinstance(row, Mapping) else None
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(float(value)):
        raise RuntimeError(f"undefined C6 metric: {label}/{metric_id}")
    return float(value)


def _write_csv(path: Path, rows: list[dict[str, Any]], preferred: Sequence[str]) -> None:
    fields = [name for name in preferred if any(name in row for row in rows)]
    fields.extend(sorted({key for row in rows for key in row} - set(fields)))
    encoded = [
        {
            key: json.dumps(value, sort_keys=True, separators=(",", ":"))
            if isinstance(value, (dict, list, tuple))
            else value
            for key, value in row.items()
        }
        for row in rows
    ]
    atomic_write_text(path, rows_to_csv(fields, encoded))


def finalize(
    run_root: Path,
    decision: Mapping[str, Any],
    decision_sha: str,
    barrier: Mapping[str, Any],
    barrier_sha: str,
    evaluation: Mapping[str, Any],
    evaluation_sha: str,
) -> dict[str, str]:
    arm_ids = tuple(spec.arm_id for spec in ARM_SPECS)
    evaluation_hashes = {
        case_id: sha256_file(_evaluation_case_path(run_root, case_id)) for case_id in decision["case_ids"]
    }
    evaluation_barrier = {
        "schema": EVALUATION_BARRIER_SCHEMA,
        "protocol_id": PROTOCOL_ID,
        "status": "COMPLETE",
        "strict": True,
        "decision_contract_sha256": decision_sha,
        "decision_barrier_sha256": barrier_sha,
        "evaluation_contract_sha256": evaluation_sha,
        "evaluation_case_sha256": evaluation_hashes,
        "test_split_accessed": False,
    }
    evaluation_barrier_sha = _immutable_json(run_root / "evaluation_barrier.json", evaluation_barrier)
    decisions: dict[str, dict[str, Any]] = {}
    evaluations: dict[str, dict[str, Any]] = {}
    for case_id in decision["case_ids"]:
        dpath, epath = _decision_case_path(run_root, case_id), _evaluation_case_path(run_root, case_id)
        if sha256_file(dpath) != barrier["decision_case_sha256"][case_id]:
            raise RuntimeError(f"C6 decision snapshot changed before finalization: {case_id}")
        if sha256_file(epath) != evaluation_hashes[case_id]:
            raise RuntimeError(f"C6 evaluation snapshot changed before finalization: {case_id}")
        decisions[case_id], evaluations[case_id] = _load_json(dpath), _load_json(epath)
        if evaluations[case_id].get("evaluation_contract_sha256") != evaluation_sha:
            raise RuntimeError(f"invalid C6 evaluation case at finalization: {case_id}")

    dice = {arm_id: [] for arm_id in arm_ids}
    returned = {arm_id: [] for arm_id in arm_ids}
    sdlogj = {arm_id: [] for arm_id in arm_ids}
    folds = {arm_id: [] for arm_id in arm_ids}
    label_dice = {arm_id: {label: [] for label in evaluation["evaluation_label_ids"]} for arm_id in arm_ids}
    diagnostics = {arm_id: {name: [] for name in DIRECTION_DIAGNOSTIC_FIELDS} for arm_id in arm_ids}
    per_case: list[dict[str, Any]] = []
    per_label: list[dict[str, Any]] = []
    for case_id in decision["case_ids"]:
        drows = {row["arm_id"]: row for row in decisions[case_id]["arms"]}
        erows = {row["arm_id"]: row for row in evaluations[case_id]["arms"]}
        if tuple(drows) != arm_ids or tuple(erows) != arm_ids:
            raise RuntimeError(f"C6 arm inventory changed: {case_id}")
        for arm_id in arm_ids:
            drow, erow = drows[arm_id], erows[arm_id]
            sd = _metric_value(drow["geometry"], MATHEMATICAL_SDLOGJ_CROP2, f"{case_id}/{arm_id}")
            corner_fraction = float(
                drow["geometry"][DIGITAL_DECOMPOSITION]["components"]["corner_union_violation_fraction"]
            )
            direction_row = _direction_diagnostics(drow, f"{case_id}/{arm_id}")
            for name, value in direction_row.items():
                if value is not None:
                    diagnostics[arm_id][name].append(value)
            dice[arm_id].append(float(erow["candidate_dice"]))
            returned[arm_id].append(float(erow["returned_dice"]))
            sdlogj[arm_id].append(sd)
            folds[arm_id].append(corner_fraction)
            per_case.append(
                {
                    "case_id": case_id,
                    "arm_id": arm_id,
                    "action": erow["action"],
                    "baseline_dice": erow["baseline_dice"],
                    "candidate_dice": erow["candidate_dice"],
                    "capacity_delta_vs_initial": erow["capacity_delta_vs_initial"],
                    "returned_dice": erow["returned_dice"],
                    "returned_delta_vs_initial": erow["returned_delta_vs_initial"],
                    "sdlogj": sd,
                    "corner_fold_fraction": corner_fraction,
                    **direction_row,
                }
            )
            for row in erow["per_label"]:
                label = int(row["label"])
                label_dice[arm_id][label].append(float(row["candidate_dice"]))
                per_label.append({"case_id": case_id, "arm_id": arm_id, **row})

    for arm_id in arm_ids:
        expected = 0 if arm_id == REFERENCE_ARM_ID else EXPECTED_CASE_COUNT
        if any(len(diagnostics[arm_id][name]) != expected for name in DIRECTION_DIAGNOSTIC_FIELDS):
            raise RuntimeError(f"C6 direction diagnostics are incomplete: {arm_id}")

    def arrays(table: Mapping[str, Sequence[float]], key: str) -> np.ndarray:
        return np.asarray(table[key], dtype=np.float64)

    capacity_family = simultaneous_paired_summaries(
        "c6_capacity_vs_c4",
        {arm_id: arrays(dice, arm_id) - arrays(dice, REFERENCE_ARM_ID) for arm_id in SELECTABLE_ARM_IDS},
    )
    returned_family = simultaneous_paired_summaries(
        "c6_returned_vs_c4",
        {arm_id: arrays(returned, arm_id) - arrays(dice, REFERENCE_ARM_ID) for arm_id in SELECTABLE_ARM_IDS},
    )
    full_family = simultaneous_paired_summaries(
        "c6_causal_vs_full",
        {arm_id: arrays(dice, arm_id) - arrays(dice, matched_control_ids(arm_id)[0]) for arm_id in SELECTABLE_ARM_IDS},
    )
    blur_family = simultaneous_paired_summaries(
        "c6_causal_vs_blur",
        {arm_id: arrays(dice, arm_id) - arrays(dice, matched_control_ids(arm_id)[1]) for arm_id in SELECTABLE_ARM_IDS},
    )
    sd_family = simultaneous_paired_summaries(
        "c6_sdlogj_vs_c4",
        {arm_id: arrays(sdlogj, arm_id) - arrays(sdlogj, REFERENCE_ARM_ID) for arm_id in SELECTABLE_ARM_IDS},
    )
    regional_inputs = {
        f"{arm_id}::label_{label}": np.asarray(label_dice[arm_id][label])
        - np.asarray(label_dice[REFERENCE_ARM_ID][label])
        for arm_id in SELECTABLE_ARM_IDS
        for label in evaluation["evaluation_label_ids"]
    }
    regional_family = simultaneous_paired_summaries("c6_regional_vs_c4", regional_inputs)
    no_rewarp = simultaneous_paired_summaries(
        "c6_no_rewarp_diagnostic",
        {"pyr421_norewarp_a100_vs_pyr421_a100": arrays(dice, "pyr421_norewarp_a100") - arrays(dice, "pyr421_a100")},
    )["pyr421_norewarp_a100_vs_pyr421_a100"]
    assessments = []
    for arm_id in SELECTABLE_ARM_IDS:
        regional = [
            (label, regional_family[f"{arm_id}::label_{label}"]) for label in evaluation["evaluation_label_ids"]
        ]
        assessments.append(
            assess_arm(
                arm_id,
                capacity_vs_c4=capacity_family[arm_id],
                causal_vs_full=full_family[arm_id],
                causal_vs_blur=blur_family[arm_id],
                returned_vs_c4=returned_family[arm_id],
                sdlogj_vs_c4=sd_family[arm_id],
                regional_vs_c4=regional,
                folds_all_zero=all(value == 0.0 for value in folds[arm_id]),
            )
        )
    branch = select_branch(
        assessments,
        {arm_id: capacity_family[arm_id].mean for arm_id in SELECTABLE_ARM_IDS},
        no_rewarp_vs_rewarp=no_rewarp,
    )
    summary = {
        "schema": "ctcf-search-c6-summary-v1",
        "protocol_id": PROTOCOL_ID,
        "status": "COMPLETE",
        "n_cases": EXPECTED_CASE_COUNT,
        "test_115_authorized": False,
        "test_split_accessed": False,
        "absolute": {
            arm_id: {
                "dice_mean": float(arrays(dice, arm_id).mean()),
                "dice_median": float(np.median(arrays(dice, arm_id))),
                "returned_dice_mean": float(arrays(returned, arm_id).mean()),
                "sdlogj_mean": float(arrays(sdlogj, arm_id).mean()),
                "corner_folds_all_zero": all(value == 0.0 for value in folds[arm_id]),
                **{
                    f"{name}_mean": (
                        None
                        if not diagnostics[arm_id][name]
                        else float(np.asarray(diagnostics[arm_id][name], dtype=np.float64).mean())
                    )
                    for name in DIRECTION_DIAGNOSTIC_FIELDS
                },
            }
            for arm_id in arm_ids
        },
        "capacity_vs_c4": {key: asdict(value) for key, value in capacity_family.items()},
        "causal_vs_full": {key: asdict(value) for key, value in full_family.items()},
        "causal_vs_blur": {key: asdict(value) for key, value in blur_family.items()},
        "returned_vs_c4": {key: asdict(value) for key, value in returned_family.items()},
        "sdlogj_vs_c4": {key: asdict(value) for key, value in sd_family.items()},
        "regional_vs_c4": {key: asdict(value) for key, value in regional_family.items()},
        "no_rewarp_diagnostic": asdict(no_rewarp),
        "assessments": [asdict(row) for row in assessments],
        "next_branch": branch,
        "threshold_note": f"non-risk regional lower bound is strictly above {ALL_LABEL_CI_LOW_VS_C4_MIN_STRICT}; risk labels {RISK_LABEL_IDS} use their tighter frozen bound",
    }
    paths = {
        "per_case": run_root / "per_case.csv",
        "per_label": run_root / "per_label.csv",
        "summary": run_root / "summary.json",
        "next_branch": run_root / "next_branch.json",
    }
    _write_csv(
        paths["per_case"],
        per_case,
        (
            "case_id",
            "arm_id",
            "action",
            "baseline_dice",
            "candidate_dice",
            "returned_dice",
            "sdlogj",
            *DIRECTION_DIAGNOSTIC_FIELDS,
        ),
    )
    _write_csv(
        paths["per_label"],
        per_label,
        ("case_id", "arm_id", "label", "baseline_dice", "candidate_dice", "returned_dice"),
    )
    atomic_write_json(paths["summary"], summary)
    atomic_write_json(paths["next_branch"], branch)
    files = {key: sha256_file(path) for key, path in paths.items()}
    manifest = {
        "schema": "ctcf-search-c6-run-manifest-v1",
        "protocol_id": PROTOCOL_ID,
        "status": "COMPLETE",
        "policy_sha256": C6_POLICY_SHA256,
        "decision_contract_sha256": decision_sha,
        "decision_barrier_sha256": barrier_sha,
        "evaluation_contract_sha256": evaluation_sha,
        "evaluation_barrier_sha256": evaluation_barrier_sha,
        "decision_case_sha256": barrier["decision_case_sha256"],
        "evaluation_case_sha256": evaluation_hashes,
        "files": files,
        "test_115_authorized": False,
        "test_split_accessed": False,
        "completed_at_utc": utc_now(),
    }
    for case_id, digest in evaluation_hashes.items():
        if sha256_file(_evaluation_case_path(run_root, case_id)) != digest:
            raise RuntimeError(f"C6 evaluation snapshot changed during finalization: {case_id}")
    atomic_write_json(run_root / "c6_manifest.json", manifest)
    files["c6_manifest"] = sha256_file(run_root / "c6_manifest.json")
    return files


def _execution(
    decision: Mapping[str, Any], phase: str, attempt_id: str, shard_index: int, physical_gpu: str, device: torch.device
) -> dict[str, Any]:
    return {
        "phase": phase,
        "attempt_id": attempt_id,
        "shard_index": shard_index,
        "physical_gpu": physical_gpu,
        "host": platform.node(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(device),
        "seed": decision["seed"],
        "deterministic": True,
        "labels_loaded_to_device": phase == "evaluation",
    }


def selfcheck_stage(args: argparse.Namespace) -> int:
    assert_frozen_policy()
    report = {
        "schema": f"ctcf-search-c6-selfcheck-{SCHEMA_VERSION}",
        "protocol_id": PROTOCOL_ID,
        "status": "PASS",
        "policy_sha256": C6_POLICY_SHA256,
        "arm_ids": [row.arm_id for row in ARM_SPECS],
        "selectable_arm_ids": list(SELECTABLE_ARM_IDS),
        "matched_controls": {arm_id: list(matched_control_ids(arm_id)) for arm_id in SELECTABLE_ARM_IDS},
        "test_115_authorized": TEST_115_AUTHORIZED,
    }
    atomic_write_json(args.output, report)
    print(json.dumps(report, indent=2))
    return 0


def prepare_stage(args: argparse.Namespace) -> int:
    assert_frozen_policy()
    if git("status", "--porcelain=v1"):
        raise RuntimeError("C6 prepare requires a clean tracked and untracked tree")
    physical_gpus = parse_physical_gpus(
        args.physical_gpus, args.num_shards, "C6 requires one unique physical GPU per shard"
    )
    run_root, heavy_root = args.run_root.resolve(), args.heavy_root.resolve()
    if run_root == heavy_root or run_root in heavy_root.parents or heavy_root in run_root.parents:
        raise RuntimeError("C6 compact and heavy roots must not overlap")
    _validate_disk_budget(heavy_root, args.min_free_gib)
    source_sha, decision_sha = prepare_contracts(
        run_root=run_root,
        heavy_root=heavy_root,
        source_c4_dir=args.source_c4_dir,
        source_c4_heavy_root=args.source_c4_heavy_root,
        physical_gpus=physical_gpus,
    )
    print(
        json.dumps(
            {
                "source_contract_sha256": source_sha,
                "decision_contract_sha256": decision_sha,
                "n_cases": EXPECTED_CASE_COUNT,
            }
        )
    )
    return 0


def _decision_context(args: argparse.Namespace) -> tuple[dict[str, Any], str, torch.device]:
    decision_sha = str(args.decision_contract_sha256)
    decision = _load_decision(args.run_root, decision_sha)
    _assert_clean_runtime(decision, args.action)
    if args.num_shards != decision["num_shards"] or args.physical_gpu != decision["shard_to_physical_gpu"].get(
        str(args.shard_index)
    ):
        raise RuntimeError("C6 worker settings differ from the frozen contract")
    device = setup_device(args.gpu, seed=decision["seed"], deterministic=True)
    if device.type != "cuda":
        raise RuntimeError("C6 workers require CUDA")
    return decision, decision_sha, device


def decision_pilot_stage(args: argparse.Namespace) -> int:
    args.shard_index = 0
    decision, decision_sha, device = _decision_context(args)
    case_id = decision["shards"]["0"][0]
    marker = run_decision_case(
        case_id=case_id,
        shard_index=0,
        physical_gpu=args.physical_gpu,
        run_root=args.run_root,
        decision=decision,
        decision_sha256=decision_sha,
        device=device,
        execution=_execution(decision, "decision", args.attempt_id, 0, args.physical_gpu, device),
    )
    print(f"[C6 DECISION PILOT COMPLETE] case={case_id} marker={marker}")
    return 0


def decision_worker_stage(args: argparse.Namespace) -> int:
    decision, decision_sha, device = _decision_context(args)
    marker = run_decision_worker(
        shard_index=args.shard_index,
        physical_gpu=args.physical_gpu,
        attempt_id=args.attempt_id,
        run_root=args.run_root,
        decision=decision,
        decision_sha256=decision_sha,
        device=device,
        execution=_execution(decision, "decision", args.attempt_id, args.shard_index, args.physical_gpu, device),
    )
    print(f"[C6 DECISION WORKER COMPLETE] {marker}")
    return 0


def barrier_stage(args: argparse.Namespace) -> int:
    decision = _load_decision(args.run_root, args.decision_contract_sha256)
    _assert_clean_runtime(decision, "decision barrier")
    digest = build_decision_barrier(args.run_root, decision, args.decision_contract_sha256, args.attempt_id)
    print(f"[C6 DECISION BARRIER] {digest}")
    return 0


def freeze_evaluation_stage(args: argparse.Namespace) -> int:
    decision = _load_decision(args.run_root, args.decision_contract_sha256)
    _assert_clean_runtime(decision, "evaluation freeze")
    digest = freeze_evaluation(
        args.run_root,
        args.source_contract_sha256,
        decision,
        args.decision_contract_sha256,
        args.barrier_sha256,
    )
    print(f"[C6 EVALUATION CONTRACT] {digest}")
    return 0


def evaluation_worker_stage(args: argparse.Namespace) -> int:
    decision, decision_sha, device = _decision_context(args)
    barrier = _load_barrier(args.run_root, args.barrier_sha256, decision_sha)
    evaluation = _load_evaluation(
        args.run_root,
        args.evaluation_contract_sha256,
        args.source_contract_sha256,
        decision_sha,
        args.barrier_sha256,
    )
    assigned = decision["shards"][str(args.shard_index)]
    for case_id in ("atlas", *assigned):
        _verify_raw(evaluation["raw_inputs"][case_id])
    from experiments.core.inference_metrics import metric_profile_for
    from experiments.core.inference_runtime import build_infer_dataset

    dataset = build_infer_dataset(
        "IXI",
        [evaluation["raw_inputs"][case_id]["path"] for case_id in assigned],
        evaluation["raw_inputs"]["atlas"]["path"],
    )
    index_by_case = {case_id: index for index, case_id in enumerate(assigned)}
    marker = run_evaluation_worker(
        shard_index=args.shard_index,
        physical_gpu=args.physical_gpu,
        attempt_id=args.attempt_id,
        run_root=args.run_root,
        decision=decision,
        decision_sha=decision_sha,
        barrier=barrier,
        barrier_sha=args.barrier_sha256,
        evaluation=evaluation,
        evaluation_sha=args.evaluation_contract_sha256,
        device=device,
        execution=_execution(decision, "evaluation", args.attempt_id, args.shard_index, args.physical_gpu, device),
        dataset_item_for_case=lambda case_id: dataset[index_by_case[case_id]],
        labels=tuple(metric_profile_for("IXI").labels),
    )
    print(f"[C6 EVALUATION WORKER COMPLETE] {marker}")
    return 0


def finalize_stage(args: argparse.Namespace) -> int:
    decision = _load_decision(args.run_root, args.decision_contract_sha256)
    _assert_clean_runtime(decision, "finalization")
    barrier = _load_barrier(args.run_root, args.barrier_sha256, args.decision_contract_sha256)
    evaluation = _load_evaluation(
        args.run_root,
        args.evaluation_contract_sha256,
        args.source_contract_sha256,
        args.decision_contract_sha256,
        args.barrier_sha256,
    )
    artifacts = finalize(
        args.run_root,
        decision,
        args.decision_contract_sha256,
        barrier,
        args.barrier_sha256,
        evaluation,
        args.evaluation_contract_sha256,
    )
    print(json.dumps({"status": "COMPLETE", "artifacts": artifacts}, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the frozen C6 true-pyramid causal gate on IXI validation-58.")
    sub = parser.add_subparsers(dest="action", required=True)
    selfcheck = sub.add_parser("selfcheck")
    selfcheck.add_argument("--output", type=Path, required=True)
    prepare = sub.add_parser("prepare")
    prepare.add_argument("--run-root", type=Path, required=True)
    prepare.add_argument("--heavy-root", type=Path, required=True)
    prepare.add_argument("--source-c4-dir", type=Path, required=True)
    prepare.add_argument("--source-c4-heavy-root", type=Path, required=True)
    prepare.add_argument("--num-shards", type=int, required=True)
    prepare.add_argument("--physical-gpus", required=True)
    prepare.add_argument("--min-free-gib", type=float, default=DEFAULT_MIN_FREE_GIB)
    pilot = sub.add_parser("decision-pilot")
    pilot.add_argument("--run-root", type=Path, required=True)
    pilot.add_argument("--decision-contract-sha256", required=True)
    pilot.add_argument("--num-shards", type=int, required=True)
    pilot.add_argument("--gpu", type=int, default=0)
    pilot.add_argument("--physical-gpu", required=True)
    pilot.add_argument("--attempt-id", required=True)
    for action in ("decision-worker", "evaluation-worker"):
        worker = sub.add_parser(action)
        worker.add_argument("--run-root", type=Path, required=True)
        worker.add_argument("--decision-contract-sha256", required=True)
        worker.add_argument("--shard-index", type=int, required=True)
        worker.add_argument("--num-shards", type=int, required=True)
        worker.add_argument("--gpu", type=int, default=0)
        worker.add_argument("--physical-gpu", required=True)
        worker.add_argument("--attempt-id", required=True)
        if action == "evaluation-worker":
            worker.add_argument("--source-contract-sha256", required=True)
            worker.add_argument("--barrier-sha256", required=True)
            worker.add_argument("--evaluation-contract-sha256", required=True)
    barrier = sub.add_parser("decision-barrier")
    barrier.add_argument("--run-root", type=Path, required=True)
    barrier.add_argument("--decision-contract-sha256", required=True)
    barrier.add_argument("--attempt-id", required=True)
    freeze = sub.add_parser("freeze-evaluation")
    freeze.add_argument("--run-root", type=Path, required=True)
    freeze.add_argument("--source-contract-sha256", required=True)
    freeze.add_argument("--decision-contract-sha256", required=True)
    freeze.add_argument("--barrier-sha256", required=True)
    final = sub.add_parser("finalize")
    final.add_argument("--run-root", type=Path, required=True)
    final.add_argument("--source-contract-sha256", required=True)
    final.add_argument("--decision-contract-sha256", required=True)
    final.add_argument("--barrier-sha256", required=True)
    final.add_argument("--evaluation-contract-sha256", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    actions = {
        "selfcheck": selfcheck_stage,
        "prepare": prepare_stage,
        "decision-pilot": decision_pilot_stage,
        "decision-worker": decision_worker_stage,
        "decision-barrier": barrier_stage,
        "freeze-evaluation": freeze_evaluation_stage,
        "evaluation-worker": evaluation_worker_stage,
        "finalize": finalize_stage,
    }
    return actions[args.action](args)


if __name__ == "__main__":
    raise SystemExit(main())
