from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import re
import shutil
import tempfile
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from tools.analysis.run_artifacts import atomic_write_json, atomic_write_text, rows_to_csv, sha256_file
from tools.analysis.search_gate_c3 import (
    ARM_SPECS,
    ARM_SPECS_BY_ID,
    C2_DICE_PARITY_ATOL,
    C3A_POLICY,
    C3A_POLICY_SHA256,
    CANDIDATE_COUNT,
    CANDIDATE_RADIUS,
    COLLAR_WIDTH,
    CONTROL_DECODER_PARITY_ATOL,
    COST_STANDARDIZATION_FLOOR,
    EXACT_CLAIM_EPS,
    FLOAT32_PARITY_ATOL,
    FLOAT32_PARITY_RTOL,
    IMAGE_ZSCORE_STD_FLOOR,
    LOCAL_CLIP_SWEEPS,
    MESSAGE_PASSING_AXIS_KERNEL,
    MIND_DILATION,
    MIND_RADIUS,
    NCC_DENOMINATOR_EPS,
    POSTERIOR_TEMPERATURE,
    PROTOCOL_ID,
    SCHEMA_VERSION,
    SELECTABLE_ARM_IDS,
    SUPPORT_RETENTION_MIN,
    WORK_EPS,
    build_c3a_supports,
    geometry_noninferior,
    materially_strong_capacity,
    metric_envelope,
    paired_summary,
    primary_ncc_decision,
    select_winner,
    viable_primary_policy,
    wins,
)
from tools.analysis.search_gate_common import (
    CONFIG_KEY,
    DEFAULT_CHECKPOINT,
    TIME_STEPS,
    build_model,
    dice_score,
    git,
    payload_sha256,
    prepare_initial_state,
    proposal_statistics,
    utc_now,
)
from tools.analysis.search_gate_cost_volume import (
    LOGIT_MESSAGE_AXIS_KERNEL,
    POSTERIOR_DIAGNOSTICS_ID,
    RMS_FIRST_DIFFERENCE_ROUGHNESS_ID,
    STANDARDIZATION_FLOOR,
    apply_message_passing,
    build_standardized_mind_cost_volume,
    decode_posterior,
    masked_rms_first_difference_roughness,
    masked_vector_rms,
    match_postprocessed_rms,
    posterior_diagnostics,
    posterior_from_logits,
    postprocess_residual,
    raw_posterior,
)
from tools.analysis.search_gate_metrics import (
    DETJ_DIAGNOSTICS,
    DIGITAL_DECOMPOSITION,
    LEARN2REG_SHIFTED_SDLOGJ_MASKED,
    MATHEMATICAL_SDLOGJ_CROP2,
    METRIC_SPECS,
    compute_metric,
)
from tools.analysis.search_gate_runtime import (
    expected_shard_for_case,
    parse_physical_gpus,
    read_csv,
    round_robin_shards,
    save_reload_certify,
    shard_gpu_map,
    validate_shard_partition,
)
from tools.analysis.transactional_search import (
    OFFSETS,
    build_proposal,
    certified_local_clip_candidate,
    geometry_mask,
    identity_collar,
    load_flow_npz,
    masked_zscore,
    mind_distance_from_features,
    mind_ssc,
    ncc_loss_from_normalized,
    smooth_proposal,
    valid_sample_mask,
)
from utils import setup_device
from utils.field import trilinear_cert_bound

C2_PROTOCOL_ID = "CTCF-SEARCH-GATE-C2-V1"
C2_MANIFEST_SCHEMA = "ctcf-search-c2-run-manifest-v1"
C2_CONTRACT_SCHEMA = "ctcf-search-c2-contract-v1"
SOURCE_CONTRACT_NAME = "source_contract.json"
DECISION_CONTRACT_NAME = "decision_contract.json"
DECISION_BARRIER_NAME = "decision_barrier.json"
EXPECTED_CASES = 58
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
CONTROL_GOLDEN_ROWS = {
    "c1_raw_conf_post1": ("mind_s1_sm1", 1),
    "raw_conf_post1": ("mind_s2_sm1", 1),
    "raw_conf_post2": ("mind_s2_sm2", 1),
}
RESUME_MIN_FREE_GIB = 5.0


def _sha256_array(array: np.ndarray | torch.Tensor) -> str:
    if isinstance(array, torch.Tensor):
        value = array.detach().cpu().contiguous().numpy().astype(np.float32, copy=False)
    else:
        value = np.ascontiguousarray(array, dtype=np.float32)
    return hashlib.sha256(value.tobytes(order="C")).hexdigest()


def _require_sha(value: str, label: str) -> str:
    lowered = str(value).lower()
    if not SHA256_RE.fullmatch(lowered):
        raise ValueError(f"{label} must be 64 lowercase hexadecimal characters")
    return lowered


def _finite_dice(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RuntimeError(f"{label} must be a numeric Dice value")
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise RuntimeError(f"{label} must be finite and in [0, 1]")
    return result


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return payload


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


def _assert_runtime_signature(expected: dict[str, Any], stage: str) -> None:
    observed = _runtime_signature()
    if observed != expected:
        raise RuntimeError(f"C3a {stage} runtime differs from the frozen source contract: {observed} != {expected}")


def _verify_file_record(row: dict[str, Any]) -> None:
    path = Path(str(row["path"]))
    if not path.is_file() or path.stat().st_size != int(row["bytes"]) or sha256_file(path) != str(row["sha256"]):
        raise RuntimeError(f"Frozen input changed: {path}")


def _assert_clean_code(head: str, stage: str) -> None:
    if git("rev-parse", "HEAD") != head or git("status", "--porcelain=v1"):
        raise RuntimeError(f"C3a {stage} code differs from the clean prepared contract")


def _validate_disk_budget(*, free_gib: float, heavy_gib: float, min_free_gib: float, is_resume: bool) -> None:
    values = (free_gib, heavy_gib, min_free_gib)
    if any(not math.isfinite(float(value)) or float(value) < 0.0 for value in values):
        raise ValueError("C3a disk-budget values must be finite and non-negative")
    if not is_resume and free_gib < min_free_gib:
        raise RuntimeError(f"C3a requires at least {min_free_gib} GiB free; found {free_gib:.2f}")
    if is_resume and (free_gib < RESUME_MIN_FREE_GIB or free_gib + heavy_gib < min_free_gib):
        raise RuntimeError(
            "C3a resume lacks its frozen disk budget: "
            f"free={free_gib:.2f} GiB, current_run={heavy_gib:.2f} GiB, "
            f"required_total={min_free_gib} GiB, reserve={RESUME_MIN_FREE_GIB:.2f} GiB"
        )


def _validate_c2_source(directory: Path, expected_manifest_sha: str) -> tuple[dict[str, Any], dict[str, Any]]:
    directory = directory.resolve()
    manifest_path = directory / "c2_manifest.json"
    expected = _require_sha(expected_manifest_sha, "C2 manifest SHA-256")
    if not manifest_path.is_file() or sha256_file(manifest_path) != expected:
        raise RuntimeError(f"Frozen C2 manifest is missing or changed: {manifest_path}")
    manifest = _load_json(manifest_path)
    summary = manifest.get("summary") or {}
    code = manifest.get("code") or {}
    if (
        manifest.get("schema") != C2_MANIFEST_SCHEMA
        or manifest.get("protocol_id") != C2_PROTOCOL_ID
        or manifest.get("status") != "COMPLETE"
        or code.get("git_status") != ""
        or summary.get("execution_integrity_status") != "PASS"
        or summary.get("n_cases") != EXPECTED_CASES
        or summary.get("n_step_rows") != 928
        or summary.get("n_trajectories") != 4
        or summary.get("test_115_authorized") is not False
        or summary.get("test_split_accessed") is not False
        or summary.get("labels_used_for_transaction_decision") is not False
    ):
        raise RuntimeError("C3a requires one clean, COMPLETE, label-isolated C2 val-58 product")
    files = manifest.get("files") or {}
    required = {
        "contract_sha256": "c2_contract.json",
        "datasets_sha256": "datasets.csv",
        "c1_reference_sha256": "c1_reference.csv",
        "per_step_sha256": "per_step.csv",
        "per_branch_sha256": "per_branch.csv",
        "trajectory_summary_sha256": "trajectory_summary.csv",
        "summary_sha256": "summary.json",
    }
    for key, name in required.items():
        path = directory / name
        if not path.is_file() or files.get(key) != sha256_file(path):
            raise RuntimeError(f"C2 manifest does not authenticate {path}")
    contract = _load_json(directory / "c2_contract.json")
    if (
        contract.get("schema") != C2_CONTRACT_SCHEMA
        or contract.get("protocol_id") != C2_PROTOCOL_ID
        or contract.get("ixi_test_split_accessed") is not False
        or contract.get("case_ids") is None
        or len(contract["case_ids"]) != EXPECTED_CASES
        or len(set(contract["case_ids"])) != EXPECTED_CASES
        or manifest.get("contract_sha256") != files["contract_sha256"]
    ):
        raise RuntimeError("Frozen C2 contract is inconsistent")
    return manifest, contract


def _extract_c2_goldens(
    directory: Path,
    manifest: dict[str, Any],
    contract: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    marker_hashes = manifest.get("case_marker_sha256") or {}
    if set(marker_hashes) != set(contract["case_ids"]):
        raise RuntimeError("C2 manifest does not authenticate exactly 58 case markers")
    decision_cases: dict[str, Any] = {}
    dice_cases: dict[str, Any] = {}
    for case_id in contract["case_ids"]:
        marker = directory / "cases" / case_id / "case_complete.json"
        if not marker.is_file() or sha256_file(marker) != marker_hashes[case_id]:
            raise RuntimeError(f"C2 case marker is missing or changed: {marker}")
        payload = _load_json(marker)
        initial = payload.get("initial") or {}
        initial_exact = initial.get("psi_exact") or {}
        rows = payload.get("rows") or []
        if (
            payload.get("schema") != "ctcf-search-c2-case-v1"
            or payload.get("status") != "COMPLETE"
            or payload.get("case_id") != case_id
            or initial_exact.get("status") != "CERTIFIED"
            or initial_exact.get("certified") is not True
        ):
            raise RuntimeError(f"Invalid C2 golden case marker: {marker}")
        controls: dict[str, str] = {}
        control_dice: dict[str, float] = {}
        baseline_dice: float | None = None
        for arm_id, identity in CONTROL_GOLDEN_ROWS.items():
            matches = [row for row in rows if (row.get("trajectory_id"), row.get("step")) == identity]
            if (
                len(matches) != 1
                or matches[0].get("candidate_exact_status") != "CERTIFIED"
                or matches[0].get("candidate_exact_certified") is not True
            ):
                raise RuntimeError(f"C2 golden row is missing for {case_id}/{arm_id}")
            controls[arm_id] = _require_sha(matches[0]["candidate_array_sha256"], f"{case_id}/{arm_id}")
            row_baseline = _finite_dice(matches[0].get("baseline_dice"), f"{case_id}/{arm_id}/baseline")
            row_candidate = _finite_dice(matches[0].get("candidate_dice"), f"{case_id}/{arm_id}/candidate")
            if baseline_dice is not None and row_baseline != baseline_dice:
                raise RuntimeError(f"C2 controls disagree on baseline Dice for {case_id}")
            baseline_dice = row_baseline
            control_dice[arm_id] = row_candidate
        if baseline_dice is None:
            raise RuntimeError(f"C2 baseline Dice is missing for {case_id}")
        decision_cases[case_id] = {
            "c2_case_marker_sha256": marker_hashes[case_id],
            "initial_array_sha256": _require_sha(initial_exact["sha256"], f"{case_id}/initial"),
            "control_candidate_array_sha256": controls,
        }
        dice_cases[case_id] = {
            "baseline_dice": baseline_dice,
            "control_candidate_dice": control_dice,
        }
    common = {
        "source_manifest_sha256": sha256_file(directory / "c2_manifest.json"),
        "source_contract_sha256": sha256_file(directory / "c2_contract.json"),
        "case_ids": list(contract["case_ids"]),
    }
    return (
        {
            "schema": f"ctcf-search-c3a-c2-decision-goldens-{SCHEMA_VERSION}",
            **common,
            "cases": decision_cases,
        },
        {
            "schema": f"ctcf-search-c3a-c2-evaluation-goldens-{SCHEMA_VERSION}",
            **common,
            "cases": dice_cases,
        },
    )


def _atomic_save_npy(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".npy", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            np.save(stream, array, allow_pickle=False)
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def _image_record(path: Path, array: np.ndarray) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "array_sha256": _sha256_array(array),
        "shape": list(array.shape),
        "dtype": str(array.dtype),
    }


def _verify_image_record(row: dict[str, Any]) -> np.ndarray:
    _verify_file_record(row)
    array = np.load(Path(row["path"]), allow_pickle=False)
    if (
        array.dtype != np.float32
        or list(array.shape) != row.get("shape")
        or _sha256_array(array) != row.get("array_sha256")
        or not np.isfinite(array).all()
    ):
        raise RuntimeError(f"Image-only cache changed or is invalid: {row['path']}")
    return np.ascontiguousarray(array)


def prepare_stage(args: argparse.Namespace) -> int:
    if args.num_shards < 1 or args.min_free_gib < 1:
        raise ValueError("--num-shards and --min-free-gib must be positive")
    physical = parse_physical_gpus(
        args.physical_gpus,
        args.num_shards,
        "--physical-gpus must contain one unique non-negative index per shard",
    )
    if git("status", "--porcelain=v1"):
        raise RuntimeError("C3a prepare refuses a dirty Git tree")
    head = git("rev-parse", "HEAD")
    c2_dir = args.c2_dir.resolve()
    manifest, c2_contract = _validate_c2_source(c2_dir, args.c2_manifest_sha256)
    if c2_contract.get("paths_profile") != args.paths_profile or c2_contract.get("seed") != args.seed:
        raise RuntimeError("C3a paths profile or seed differs from frozen C2")
    checkpoint = args.checkpoint.resolve()
    if not checkpoint.is_file() or sha256_file(checkpoint) != c2_contract.get("checkpoint_sha256"):
        raise RuntimeError("C3a checkpoint differs from frozen C2")
    dataset_path = c2_dir / "datasets.csv"
    rows = read_csv(dataset_path)
    if len(rows) != EXPECTED_CASES + 1 or {row["case_id"] for row in rows} != {*c2_contract["case_ids"], "atlas"}:
        raise RuntimeError("C2 dataset inventory does not contain atlas plus exactly 58 cases")
    for row in rows:
        _verify_file_record(row)
    decision_goldens, evaluation_goldens = _extract_c2_goldens(c2_dir, manifest, c2_contract)

    root = args.run_root.resolve()
    heavy_root = args.heavy_root.resolve()
    if root == heavy_root or root in heavy_root.parents or heavy_root in root.parents:
        raise ValueError("Compact run root and heavy root must be separate, non-nested directories")
    heavy_root.mkdir(parents=True, exist_ok=True)
    free_gib = shutil.disk_usage(heavy_root).free / (1024**3)
    is_resume = (root / SOURCE_CONTRACT_NAME).is_file()
    heavy_gib = (
        sum(path.stat().st_size for path in heavy_root.rglob("*") if path.is_file()) / (1024**3) if is_resume else 0.0
    )
    _validate_disk_budget(
        free_gib=free_gib,
        heavy_gib=heavy_gib,
        min_free_gib=float(args.min_free_gib),
        is_resume=is_resume,
    )
    root.mkdir(parents=True, exist_ok=True)
    datasets_csv = (c2_dir / "datasets.csv").read_text(encoding="utf-8")
    datasets_tsv = (c2_dir / "datasets.tsv").read_text(encoding="utf-8")
    source_contract = {
        "schema": f"ctcf-search-c3a-source-contract-{SCHEMA_VERSION}",
        "protocol_id": PROTOCOL_ID,
        "git_head": head,
        "c2_source": {
            "directory": str(c2_dir),
            "manifest_sha256": sha256_file(c2_dir / "c2_manifest.json"),
            "contract_sha256": sha256_file(c2_dir / "c2_contract.json"),
            "run_id": manifest["run_id"],
        },
        "raw_inputs": {row["case_id"]: row for row in rows},
        "c2_decision_goldens_sha256": payload_sha256(decision_goldens),
        "c2_evaluation_goldens_sha256": payload_sha256(evaluation_goldens),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "config": CONFIG_KEY,
        "time_steps": TIME_STEPS,
        "paths_profile": args.paths_profile,
        "seed": args.seed,
        "runtime_signature": _runtime_signature(),
        "case_ids": list(c2_contract["case_ids"]),
        "ixi_test_split_accessed": False,
        "test_115_authorized": False,
        "num_shards": args.num_shards,
        "physical_gpus": physical,
        "shard_to_physical_gpu": shard_gpu_map(physical),
        "shards": round_robin_shards(list(c2_contract["case_ids"]), args.num_shards),
        "heavy_root": str(heavy_root),
        "min_free_gib": args.min_free_gib,
        "resume_min_free_gib": RESUME_MIN_FREE_GIB,
    }
    paths = {
        "source": root / SOURCE_CONTRACT_NAME,
        "decision_goldens": root / "c2_decision_goldens.json",
        "evaluation_goldens": root / "c2_evaluation_goldens.json",
        "datasets_csv": root / "datasets.csv",
        "datasets_tsv": root / "datasets.tsv",
    }
    if paths["source"].exists():
        if (
            _load_json(paths["source"]) != source_contract
            or _load_json(paths["decision_goldens"]) != decision_goldens
            or _load_json(paths["evaluation_goldens"]) != evaluation_goldens
        ):
            raise RuntimeError("Resume refused: C3a source contract changed")
    else:
        atomic_write_text(paths["datasets_csv"], datasets_csv)
        atomic_write_text(paths["datasets_tsv"], datasets_tsv)
        atomic_write_json(paths["decision_goldens"], decision_goldens)
        atomic_write_json(paths["evaluation_goldens"], evaluation_goldens)
        atomic_write_json(paths["source"], source_contract)
    source_sha = sha256_file(paths["source"])
    prepare = root / "prepare.json"
    if not prepare.exists():
        atomic_write_json(
            prepare,
            {
                "schema": f"ctcf-search-c3a-prepare-{SCHEMA_VERSION}",
                "status": "SOURCE_PREPARED",
                "prepared_at_utc": utc_now(),
                "source_contract_sha256": source_sha,
                "free_gib_observed": free_gib,
            },
        )
    print(json.dumps({"source_contract_sha256": source_sha, "n_cases": EXPECTED_CASES}))
    return 0


def _load_source_contract(root: Path, expected_sha: str) -> tuple[dict[str, Any], str]:
    path = root.resolve() / SOURCE_CONTRACT_NAME
    actual = sha256_file(path)
    if actual != _require_sha(expected_sha, "source contract SHA-256"):
        raise RuntimeError("C3a source contract hash mismatch")
    contract = _load_json(path)
    if (
        contract.get("schema") != f"ctcf-search-c3a-source-contract-{SCHEMA_VERSION}"
        or contract.get("protocol_id") != PROTOCOL_ID
        or contract.get("ixi_test_split_accessed") is not False
        or contract.get("test_115_authorized") is not False
    ):
        raise RuntimeError("Unsupported or altered C3a source contract")
    return contract, actual


def _load_c2_evaluation_goldens(root: Path, source: dict[str, Any]) -> dict[str, Any]:
    path = root.resolve() / "c2_evaluation_goldens.json"
    goldens = _load_json(path)
    if (
        payload_sha256(goldens) != source.get("c2_evaluation_goldens_sha256")
        or goldens.get("schema") != f"ctcf-search-c3a-c2-evaluation-goldens-{SCHEMA_VERSION}"
        or goldens.get("case_ids") != source.get("case_ids")
        or set(goldens.get("cases") or {}) != set(source.get("case_ids") or [])
    ):
        raise RuntimeError("C2 evaluation-golden projection is missing, changed, or malformed")
    for case_id, row in goldens["cases"].items():
        _finite_dice(row.get("baseline_dice"), f"{case_id}/baseline")
        controls = row.get("control_candidate_dice") or {}
        if set(controls) != set(CONTROL_GOLDEN_ROWS):
            raise RuntimeError(f"C2 evaluation golden has the wrong control set for {case_id}")
        for arm_id, value in controls.items():
            _finite_dice(value, f"{case_id}/{arm_id}")
    return goldens


def extract_images_stage(args: argparse.Namespace) -> int:
    root = args.run_root.resolve()
    source, source_sha = _load_source_contract(root, args.source_contract_sha256)
    _assert_clean_code(source["git_head"], "image extraction")
    _assert_runtime_signature(source["runtime_signature"], "image extraction")
    from utils import pkload

    cache_root = Path(source["heavy_root"]) / "image_cache"
    image_records: dict[str, Any] = {}
    for case_id in ["atlas", *source["case_ids"]]:
        raw = source["raw_inputs"][case_id]
        _verify_file_record(raw)
        loaded = pkload(raw["path"])
        if not isinstance(loaded, (tuple, list)) or len(loaded) != 2:
            raise RuntimeError(f"{raw['path']}: expected an (image, segmentation) container")
        image = np.ascontiguousarray(np.asarray(loaded[0])[None], dtype=np.float32)
        if image.ndim != 4 or not np.isfinite(image).all():
            raise RuntimeError(f"{raw['path']}: image must be one finite 3-D volume")
        path = cache_root / ("atlas.npy" if case_id == "atlas" else f"cases/{case_id}.npy")
        if path.is_file():
            existing = np.load(path, allow_pickle=False)
            if existing.shape != image.shape or existing.dtype != image.dtype or not np.array_equal(existing, image):
                raise RuntimeError(f"Resume refused: image cache differs at {path}")
        else:
            _atomic_save_npy(path, image)
        image_records[case_id] = _image_record(path, image)
        del loaded

    decision_goldens = _load_json(root / "c2_decision_goldens.json")
    if payload_sha256(decision_goldens) != source["c2_decision_goldens_sha256"]:
        raise RuntimeError("C2 decision-golden projection changed after source prepare")
    decision = {
        "schema": f"ctcf-search-c3a-decision-contract-{SCHEMA_VERSION}",
        "protocol_id": PROTOCOL_ID,
        "git_head": source["git_head"],
        "source_contract_sha256": source_sha,
        "checkpoint": source["checkpoint"],
        "checkpoint_sha256": source["checkpoint_sha256"],
        "config": source["config"],
        "time_steps": source["time_steps"],
        "seed": source["seed"],
        "runtime_signature": source["runtime_signature"],
        "case_ids": source["case_ids"],
        "image_inputs": image_records,
        "source_container_sha256": {
            case_id: source["raw_inputs"][case_id]["sha256"] for case_id in ["atlas", *source["case_ids"]]
        },
        "c2_decision_goldens": decision_goldens,
        "policy": C3A_POLICY.to_dict(),
        "policy_sha256": C3A_POLICY_SHA256,
        "metric_ids": list(METRIC_SPECS),
        "primary_geometry_metric_id": MATHEMATICAL_SDLOGJ_CROP2,
        "decision_contract_contains_label_data": False,
        "decision_worker_uses_raw_containers": False,
        "os_level_data_isolation_claimed": False,
        "ixi_test_split_accessed": False,
        "test_115_authorized": False,
        "num_shards": source["num_shards"],
        "physical_gpus": source["physical_gpus"],
        "shard_to_physical_gpu": source["shard_to_physical_gpu"],
        "shards": source["shards"],
        "heavy_root": source["heavy_root"],
    }
    if "dice" in json.dumps(decision_goldens, sort_keys=True).lower():
        raise RuntimeError("Decision-worker C2 projection contains label-derived Dice data")
    serialized = json.dumps(decision, sort_keys=True)
    if ".pkl" in serialized.lower() or "segmentation" in serialized.lower():
        raise RuntimeError("Decision contract leaked a raw container or segmentation reference")
    path = root / DECISION_CONTRACT_NAME
    if path.exists() and _load_json(path) != decision:
        raise RuntimeError("Resume refused: C3a decision contract changed")
    if not path.exists():
        atomic_write_json(path, decision)
    decision_sha = sha256_file(path)
    marker = {
        "schema": f"ctcf-search-c3a-image-extraction-{SCHEMA_VERSION}",
        "status": "COMPLETE",
        "source_contract_sha256": source_sha,
        "decision_contract_sha256": decision_sha,
        "segmentation_object_deserialized_but_never_inspected": True,
        "completed_at_utc": utc_now(),
    }
    atomic_write_json(root / "image_extraction.json", marker)
    print(json.dumps({"decision_contract_sha256": decision_sha, "images": len(image_records)}))
    return 0


def _load_decision_contract(root: Path, expected_sha: str) -> tuple[dict[str, Any], str]:
    path = root.resolve() / DECISION_CONTRACT_NAME
    actual = sha256_file(path)
    if actual != _require_sha(expected_sha, "decision contract SHA-256"):
        raise RuntimeError("C3a decision contract hash mismatch")
    contract = _load_json(path)
    if (
        contract.get("schema") != f"ctcf-search-c3a-decision-contract-{SCHEMA_VERSION}"
        or contract.get("protocol_id") != PROTOCOL_ID
        or contract.get("policy_sha256") != C3A_POLICY_SHA256
        or contract.get("policy") != C3A_POLICY.to_dict()
        or contract.get("decision_contract_contains_label_data") is not False
        or contract.get("decision_worker_uses_raw_containers") is not False
        or contract.get("os_level_data_isolation_claimed") is not False
        or contract.get("ixi_test_split_accessed") is not False
        or contract.get("test_115_authorized") is not False
    ):
        raise RuntimeError("Unsupported or altered C3a decision contract")
    serialized = json.dumps(contract, sort_keys=True)
    if ".pkl" in serialized.lower() or "segmentation" in serialized.lower():
        raise RuntimeError("C3a decision contract contains forbidden source data")
    if "dice" in json.dumps(contract.get("c2_decision_goldens") or {}, sort_keys=True).lower():
        raise RuntimeError("C3a decision contract contains label-derived C2 golden data")
    return contract, actual


def _geometry_bundle(field: torch.Tensor, mask: torch.Tensor) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for metric_id in METRIC_SPECS:
        metric_mask = mask if metric_id == LEARN2REG_SHIFTED_SDLOGJ_MASKED else None
        envelope = metric_envelope(metric_id, lambda mid=metric_id, mm=metric_mask: compute_metric(mid, field, mm))
        output[metric_id] = envelope.to_dict()
    return output


def _metric_ok(bundle: dict[str, dict[str, Any]], metric_id: str) -> bool:
    return (bundle.get(metric_id) or {}).get("status") == "OK"


def _metric_value(bundle: dict[str, dict[str, Any]], metric_id: str) -> float:
    row = bundle.get(metric_id) or {}
    if row.get("status") != "OK" or not isinstance(row.get("value"), (int, float)):
        raise RuntimeError(f"Required metric is undefined: {metric_id}")
    value = float(row["value"])
    if not math.isfinite(value):
        raise RuntimeError(f"Required metric is non-finite: {metric_id}")
    return value


def _check_exact_digital_consistency(bundle: dict[str, dict[str, Any]], *, exact_certified: bool, label: str) -> None:
    if exact_certified:
        non_ok = [metric_id for metric_id, metric in bundle.items() if metric.get("status") != "OK"]
        if non_ok:
            raise RuntimeError(
                f"INTEGRITY_CONFLICT: exact certificate has undefined/error geometry metrics for {label}: {non_ok}"
            )
    digital = bundle.get(DIGITAL_DECOMPOSITION) or {}
    if exact_certified and digital.get("status") == "OK":
        corner = float((digital.get("components") or {}).get("corner_union_violation_fraction", float("nan")))
        if not math.isfinite(corner) or corner > 0.0:
            raise RuntimeError(f"INTEGRITY_CONFLICT: exact certificate disagrees with corner determinants for {label}")


def _posterior_summary(posterior: Any, mask: torch.Tensor) -> dict[str, float]:
    return {
        "entropy_mean": float(posterior.entropy.masked_select(mask).double().mean().item()),
        "normalized_entropy_mean": float(posterior.normalized_entropy.masked_select(mask).double().mean().item()),
        "confidence_mean": float(posterior.confidence.masked_select(mask).double().mean().item()),
        "valid_candidates_mean": float(posterior.valid_count.masked_select(mask).double().mean().item()),
        "valid_candidates_min": float(posterior.valid_count.masked_select(mask).min().item()),
    }


def _validate_arm_construction_invariants(arms: list[dict[str, Any]], *, label: str) -> None:
    by_id = {row.get("arm_id"): row for row in arms}
    if set(by_id) != set(ARM_SPECS_BY_ID):
        raise RuntimeError(f"C3a arm invariant set is incomplete: {label}")

    def value(arm_id: str, key: str) -> float:
        raw = (by_id[arm_id].get("proposal") or {}).get(key)
        if isinstance(raw, bool) or not isinstance(raw, (int, float)) or not math.isfinite(float(raw)):
            raise RuntimeError(f"C3a arm invariant value is invalid: {label}/{arm_id}/{key}")
        return float(raw)

    references = {
        "raw_mean_normmatched_post1": "raw_conf_post1",
        "adaptive_mean_adaptref_normmatched_post1": "adaptive_mp_conf_post1",
        "adaptive_mean_rawref_normmatched_post1": "raw_conf_post1",
    }
    for arm_id, reference_arm_id in references.items():
        reference = value(arm_id, "rms_reference")
        matched = value(arm_id, "rms_matched")
        source = value(reference_arm_id, "requested_rms")
        if not (
            math.isclose(reference, source, rel_tol=FLOAT32_PARITY_RTOL, abs_tol=FLOAT32_PARITY_ATOL)
            and math.isclose(matched, reference, rel_tol=FLOAT32_PARITY_RTOL, abs_tol=FLOAT32_PARITY_ATOL)
        ):
            raise RuntimeError(f"C3a RMS-matched arm violates its frozen reference: {label}/{arm_id}")
    if not math.isclose(
        value("iso_mp_conf_post1", "lambda_mean"),
        value("adaptive_mp_conf_post1", "lambda_mean"),
        rel_tol=FLOAT32_PARITY_RTOL,
        abs_tol=FLOAT32_PARITY_ATOL,
    ):
        raise RuntimeError(f"C3a isotropic/adaptive mean message mass differs: {label}")


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


def _common_support_utility(
    *,
    fixed_norm: torch.Tensor,
    moving_norm: torch.Tensor,
    fixed_mind: torch.Tensor,
    moving_mind: torch.Tensor,
    baseline: torch.Tensor,
    state: torch.Tensor,
    supports: dict[str, Any],
) -> dict[str, float | None]:
    output: dict[str, float | None] = {}
    if supports["mind"].pair_count == 0:
        output.update(mind_baseline_common=None, mind_candidate_common=None)
    else:
        output.update(
            mind_baseline_common=mind_distance_from_features(
                fixed_mind, moving_mind, baseline, supports["mind"].pair_mask
            ),
            mind_candidate_common=mind_distance_from_features(
                fixed_mind, moving_mind, state, supports["mind"].pair_mask
            ),
        )
    for key, window in (("ncc7", 7), ("ncc9", 9)):
        names = (f"{key}_baseline_common", f"{key}_candidate_common")
        if supports[key].pair_count == 0:
            output.update({names[0]: None, names[1]: None})
        else:
            output.update(
                {
                    names[0]: ncc_loss_from_normalized(
                        fixed_norm,
                        moving_norm,
                        baseline,
                        supports[key].pair_mask,
                        win=window,
                        eps=NCC_DENOMINATOR_EPS,
                    ),
                    names[1]: ncc_loss_from_normalized(
                        fixed_norm,
                        moving_norm,
                        state,
                        supports[key].pair_mask,
                        win=window,
                        eps=NCC_DENOMINATOR_EPS,
                    ),
                }
            )
    if not all(value is None or math.isfinite(float(value)) for value in output.values()):
        raise RuntimeError("Non-finite C3a common-support utility")
    return output


def _validate_support_utility(
    supports: dict[str, Any],
    utility: dict[str, Any],
    *,
    label: str,
) -> None:
    expected = {
        "mind": ("COMMON_MIND_SSC", 1, "mind_baseline_common", "mind_candidate_common"),
        "ncc7": ("COMMON_NCC7", 7, "ncc7_baseline_common", "ncc7_candidate_common"),
        "ncc9": ("COMMON_NCC9", 9, "ncc9_baseline_common", "ncc9_candidate_common"),
    }
    if set(supports) != set(expected) or set(utility) != {name for values in expected.values() for name in values[2:]}:
        raise RuntimeError(f"Invalid C3a common-support schema: {label}")
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
            raise RuntimeError(f"Invalid C3a common-support counts: {label}/{key}")
        values = (utility[baseline_name], utility[candidate_name])
        if pair_count == 0:
            if values != (None, None):
                raise RuntimeError(f"Empty C3a support must have null utility: {label}/{key}")
        elif any(
            isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value))
            for value in values
        ):
            raise RuntimeError(f"Non-empty C3a support must have finite utility: {label}/{key}")


def _build_requested_arms(
    *,
    fixed_image: torch.Tensor,
    moving_image: torch.Tensor,
    initial: torch.Tensor,
    mask: torch.Tensor,
    fixed_mind: torch.Tensor,
    moving_mind: torch.Tensor,
) -> tuple[dict[str, torch.Tensor], dict[str, dict[str, Any]]]:
    zero = torch.zeros_like(initial)
    legacy = build_proposal(
        fixed_image,
        moving_image,
        initial,
        mask,
        feature="mind",
        orientation="target_centered",
        fixed_feature_override=fixed_mind,
        moving_feature_override=moving_mind,
    )
    raw_post1_s1 = legacy.displacement
    raw_post1_s2 = 2.0 * legacy.displacement
    raw_post2_s2 = 2.0 * identity_collar(
        smooth_proposal(legacy.displacement, passes=1),
        width=COLLAR_WIDTH,
    )

    volume = build_standardized_mind_cost_volume(
        fixed_mind,
        moving_mind,
        initial,
        mask,
        standardization_floor=COST_STANDARDIZATION_FLOOR,
    )
    raw = raw_posterior(volume, temperature=POSTERIOR_TEMPERATURE)
    raw_conf = decode_posterior(raw, mode="confidence")
    raw_mean = decode_posterior(raw, mode="posterior_mean")
    raw_conf_new = postprocess_residual(
        raw_conf.displacement,
        scale=2.0,
        post_smoothing_passes=1,
        collar_width=COLLAR_WIDTH,
    )
    raw_mean_post = postprocess_residual(
        raw_mean.displacement,
        scale=2.0,
        post_smoothing_passes=1,
        collar_width=COLLAR_WIDTH,
    )
    independent_decoder_difference = float((raw_conf_new - raw_post1_s2).abs().max().item())
    if independent_decoder_difference > CONTROL_DECODER_PARITY_ATOL:
        raise RuntimeError(
            "PARITY_GAP: independent standardized-cost decoder differs from the frozen legacy raw control "
            f"by {independent_decoder_difference:.9g} > {CONTROL_DECODER_PARITY_ATOL:.9g}"
        )

    raw_logits = -volume.standardized_costs
    isotropic_mp = apply_message_passing(
        raw_logits,
        volume.valid,
        raw.normalized_entropy,
        mask,
        mode="isotropic",
    )
    isotropic_posterior = posterior_from_logits(
        isotropic_mp.logits,
        volume.valid,
        temperature=POSTERIOR_TEMPERATURE,
    )
    isotropic_conf = postprocess_residual(
        decode_posterior(isotropic_posterior, mode="confidence").displacement,
        scale=2.0,
        post_smoothing_passes=1,
        collar_width=COLLAR_WIDTH,
    )

    adaptive_mp = apply_message_passing(
        raw_logits,
        volume.valid,
        raw.normalized_entropy,
        mask,
        mode="adaptive",
    )
    adaptive_posterior = posterior_from_logits(
        adaptive_mp.logits,
        volume.valid,
        temperature=POSTERIOR_TEMPERATURE,
    )
    adaptive_conf = postprocess_residual(
        decode_posterior(adaptive_posterior, mode="confidence").displacement,
        scale=2.0,
        post_smoothing_passes=1,
        collar_width=COLLAR_WIDTH,
    )
    adaptive_mean_raw = postprocess_residual(
        decode_posterior(adaptive_posterior, mode="posterior_mean").displacement,
        scale=2.0,
        post_smoothing_passes=1,
        collar_width=COLLAR_WIDTH,
    )
    raw_match = match_postprocessed_rms(raw_mean_post, raw_post1_s2, mask)
    adaptive_match = match_postprocessed_rms(adaptive_mean_raw, adaptive_conf, mask)
    adaptive_rawref_match = match_postprocessed_rms(adaptive_mean_raw, raw_post1_s2, mask)

    requested = {
        "zero_update": zero,
        "c1_raw_conf_post1": raw_post1_s1,
        "raw_conf_post1": raw_post1_s2,
        "raw_conf_post2": raw_post2_s2,
        "iso_mp_conf_post1": isotropic_conf,
        "adaptive_mp_conf_post1": adaptive_conf,
        "raw_mean_normmatched_post1": raw_match.displacement,
        "adaptive_mean_adaptref_normmatched_post1": adaptive_match.displacement,
        "adaptive_mean_rawref_normmatched_post1": adaptive_rawref_match.displacement,
        "adaptive_mean_raw_post1": adaptive_mean_raw,
    }
    raw_diagnostics = asdict(
        posterior_diagnostics(
            raw_logits,
            volume.valid,
            raw,
            mask,
            temperature=POSTERIOR_TEMPERATURE,
        )
    )
    isotropic_diagnostics = asdict(
        posterior_diagnostics(
            isotropic_mp.logits,
            volume.valid,
            isotropic_posterior,
            mask,
            temperature=POSTERIOR_TEMPERATURE,
        )
    )
    adaptive_diagnostics = asdict(
        posterior_diagnostics(
            adaptive_mp.logits,
            volume.valid,
            adaptive_posterior,
            mask,
            temperature=POSTERIOR_TEMPERATURE,
        )
    )
    raw_summary = _posterior_summary(raw, mask)
    metadata: dict[str, dict[str, Any]] = {
        "zero_update": {"requested_rms": 0.0},
        "c1_raw_conf_post1": {
            **proposal_statistics(legacy, raw_post1_s1, mask),
            "requested_rms": masked_vector_rms(raw_post1_s1, mask),
        },
        "raw_conf_post1": {
            **proposal_statistics(legacy, raw_post1_s2, mask),
            "requested_rms": masked_vector_rms(raw_post1_s2, mask),
            "independent_decoder_max_abs_difference": independent_decoder_difference,
            "independent_decoder_parity_atol": CONTROL_DECODER_PARITY_ATOL,
        },
        "raw_conf_post2": {
            **proposal_statistics(legacy, raw_post2_s2, mask),
            "requested_rms": masked_vector_rms(raw_post2_s2, mask),
        },
        "iso_mp_conf_post1": {
            **_posterior_summary(isotropic_posterior, mask),
            "lambda_mean": isotropic_mp.lambda_mean,
            "requested_rms": masked_vector_rms(isotropic_conf, mask),
        },
        "adaptive_mp_conf_post1": {
            **_posterior_summary(adaptive_posterior, mask),
            "lambda_mean": adaptive_mp.lambda_mean,
            "requested_rms": masked_vector_rms(adaptive_conf, mask),
        },
        "raw_mean_normmatched_post1": {
            **raw_summary,
            "rms_source": raw_match.source_rms,
            "rms_reference": raw_match.target_rms,
            "rms_matched": raw_match.matched_rms,
            "rms_scale_factor": raw_match.scale_factor,
            "requested_rms": masked_vector_rms(raw_match.displacement, mask),
        },
        "adaptive_mean_adaptref_normmatched_post1": {
            **_posterior_summary(adaptive_posterior, mask),
            "lambda_mean": adaptive_mp.lambda_mean,
            "rms_source": adaptive_match.source_rms,
            "rms_reference": adaptive_match.target_rms,
            "rms_matched": adaptive_match.matched_rms,
            "rms_scale_factor": adaptive_match.scale_factor,
            "requested_rms": masked_vector_rms(adaptive_match.displacement, mask),
        },
        "adaptive_mean_rawref_normmatched_post1": {
            **_posterior_summary(adaptive_posterior, mask),
            "lambda_mean": adaptive_mp.lambda_mean,
            "rms_source": adaptive_rawref_match.source_rms,
            "rms_reference": adaptive_rawref_match.target_rms,
            "rms_matched": adaptive_rawref_match.matched_rms,
            "rms_scale_factor": adaptive_rawref_match.scale_factor,
            "requested_rms": masked_vector_rms(adaptive_rawref_match.displacement, mask),
        },
        "adaptive_mean_raw_post1": {
            **_posterior_summary(adaptive_posterior, mask),
            "lambda_mean": adaptive_mp.lambda_mean,
            "requested_rms": masked_vector_rms(adaptive_mean_raw, mask),
        },
    }
    posterior_by_arm = {
        "zero_update": None,
        "c1_raw_conf_post1": raw_diagnostics,
        "raw_conf_post1": raw_diagnostics,
        "raw_conf_post2": raw_diagnostics,
        "iso_mp_conf_post1": isotropic_diagnostics,
        "adaptive_mp_conf_post1": adaptive_diagnostics,
        "raw_mean_normmatched_post1": raw_diagnostics,
        "adaptive_mean_adaptref_normmatched_post1": adaptive_diagnostics,
        "adaptive_mean_rawref_normmatched_post1": adaptive_diagnostics,
        "adaptive_mean_raw_post1": adaptive_diagnostics,
    }
    for arm_id, residual in requested.items():
        metadata[arm_id]["posterior_diagnostics"] = posterior_by_arm[arm_id]
        metadata[arm_id]["postprocessed_residual_roughness"] = asdict(
            masked_rms_first_difference_roughness(residual, mask)
        )
    if set(requested) != set(ARM_SPECS_BY_ID) or set(metadata) != set(ARM_SPECS_BY_ID):
        raise AssertionError("C3a arm construction does not cover the frozen arm matrix")
    return requested, metadata


def _field_record(path: Path, heavy_root: Path, array_sha256: str) -> dict[str, Any]:
    return {
        "relative_path": path.resolve().relative_to(heavy_root.resolve()).as_posix(),
        "npz_sha256": sha256_file(path),
        "array_sha256": _require_sha(array_sha256, str(path)),
    }


def _resolve_field(contract: dict[str, Any], record: dict[str, Any]) -> Path:
    heavy = Path(contract["heavy_root"]).resolve()
    path = (heavy / record["relative_path"]).resolve()
    if heavy not in path.parents or not path.is_file() or sha256_file(path) != record.get("npz_sha256"):
        raise RuntimeError(f"Heavy field is missing, escaped its root, or changed: {path}")
    field = load_flow_npz(path)
    if _sha256_array(field) != record.get("array_sha256"):
        raise RuntimeError(f"Heavy field array changed: {path}")
    return path


def _decision_case_path(root: Path, case_id: str) -> Path:
    return root / "cases" / case_id / "decision_complete.json"


def _run_decision_case(
    *,
    case_id: str,
    atlas_image: np.ndarray,
    case_image: np.ndarray,
    adapter: Any,
    model: Any,
    device: torch.device,
    root: Path,
    contract: dict[str, Any],
    contract_sha: str,
    execution: dict[str, Any],
) -> None:
    marker = _decision_case_path(root, case_id)
    if marker.is_file():
        _validate_decision_case(_load_json(marker), marker, case_id, contract, contract_sha, verify_heavy=True)
        return
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    x = torch.from_numpy(atlas_image).unsqueeze(0).to(device)
    y = torch.from_numpy(case_image).unsqueeze(0).to(device)
    with torch.inference_mode():
        flow = adapter.forward(model, x, y, amp=True)
    heavy_root = Path(contract["heavy_root"])
    heavy_case = heavy_root / "cases" / case_id
    initial, initial_path, initial_report = prepare_initial_state(flow, heavy_case / "initial")
    (heavy_case / "initial" / "initial_phi.npz").unlink(missing_ok=True)
    initial = load_flow_npz(initial_path).to(device)
    golden = contract["c2_decision_goldens"]["cases"][case_id]
    if _sha256_array(initial) != golden["initial_array_sha256"]:
        raise RuntimeError(f"PARITY_GAP: initial field differs from frozen C2 for {case_id}")

    mask = geometry_mask(tuple(x.shape[-3:]), COLLAR_WIDTH, device)
    fixed_norm = masked_zscore(y, mask, std_floor=IMAGE_ZSCORE_STD_FLOOR)
    moving_norm = masked_zscore(x, mask, std_floor=IMAGE_ZSCORE_STD_FLOOR)
    fixed_mind = mind_ssc(fixed_norm, radius=MIND_RADIUS, dilation=MIND_DILATION)
    moving_mind = mind_ssc(moving_norm, radius=MIND_RADIUS, dilation=MIND_DILATION)
    baseline_valid = valid_sample_mask(initial)
    baseline_geometry = _geometry_bundle(initial, mask)
    _check_exact_digital_consistency(baseline_geometry, exact_certified=True, label=f"{case_id}/baseline")
    requested, proposal_metadata = _build_requested_arms(
        fixed_image=y,
        moving_image=x,
        initial=initial,
        mask=mask,
        fixed_mind=fixed_mind,
        moving_mind=moving_mind,
    )

    initial_field = _field_record(initial_path, heavy_root, initial_report["psi_exact"]["sha256"])
    arms: list[dict[str, Any]] = []
    for spec in ARM_SPECS:
        arm_id = spec.arm_id
        requested_raw = (initial + requested[arm_id]).float()
        if arm_id == "zero_update":
            requested_state = initial
            requested_field = dict(initial_field)
            requested_exact = dict(initial_report["psi_exact"])
        else:
            requested_path = heavy_case / "requested" / f"{arm_id}.npz"
            requested_stored, requested_exact = save_reload_certify(requested_raw, requested_path, EXACT_CLAIM_EPS)
            requested_state = requested_stored.to(device)
            requested_field = _field_record(requested_path, heavy_root, requested_exact["sha256"])
        requested_exact_certified = (
            requested_exact.get("status") == "CERTIFIED" and requested_exact.get("certified") is True
        )
        requested_geometry = _geometry_bundle(requested_state, mask)
        _check_exact_digital_consistency(
            requested_geometry,
            exact_certified=requested_exact_certified,
            label=f"{case_id}/{arm_id}/requested",
        )
        requested_valid = valid_sample_mask(requested_state)
        requested_supports = build_c3a_supports(mask, baseline_valid, requested_valid)
        requested_utility = _common_support_utility(
            fixed_norm=fixed_norm,
            moving_norm=moving_norm,
            fixed_mind=fixed_mind,
            moving_mind=moving_mind,
            baseline=initial,
            state=requested_state,
            supports=requested_supports,
        )
        requested_fast_bound = trilinear_cert_bound(requested_state, eps=WORK_EPS)
        requested_state_residual_rms = masked_vector_rms(requested_state - initial, mask)
        if not math.isfinite(requested_fast_bound):
            raise RuntimeError(f"Non-finite C3a requested-state fast bound for {case_id}/{arm_id}")
        if arm_id == "zero_update":
            candidate = initial
            exact = dict(initial_report["psi_exact"])
            candidate_field = dict(initial_field)
            operator_report: dict[str, Any] = {
                "operator": "ZERO_UPDATE",
                "sweeps": 0,
                "work_eps": WORK_EPS,
            }
            candidate_geometry = baseline_geometry
        else:
            candidate_raw, operator_report = certified_local_clip_candidate(
                initial,
                requested[arm_id],
                mask,
                work_eps=WORK_EPS,
                sweeps=LOCAL_CLIP_SWEEPS,
            )
            candidate_path = heavy_case / "arms" / f"{arm_id}.npz"
            stored, exact = save_reload_certify(candidate_raw, candidate_path, EXACT_CLAIM_EPS)
            candidate = stored.to(device)
            candidate_field = _field_record(candidate_path, heavy_root, exact["sha256"])
            expected_control = golden["control_candidate_array_sha256"].get(arm_id)
            if expected_control is not None and candidate_field["array_sha256"] != expected_control:
                raise RuntimeError(f"PARITY_GAP: {case_id}/{arm_id} differs from frozen C2")
            candidate_geometry = _geometry_bundle(candidate, mask)
        exact_certified = exact.get("status") == "CERTIFIED" and exact.get("certified") is True
        _check_exact_digital_consistency(
            candidate_geometry,
            exact_certified=exact_certified,
            label=f"{case_id}/{arm_id}",
        )
        candidate_valid = valid_sample_mask(candidate)
        postclip_residual_rms = masked_vector_rms(candidate - initial, mask)
        supports = build_c3a_supports(mask, baseline_valid, candidate_valid)
        utility = _common_support_utility(
            fixed_norm=fixed_norm,
            moving_norm=moving_norm,
            fixed_mind=fixed_mind,
            moving_mind=moving_mind,
            baseline=initial,
            state=candidate,
            supports=supports,
        )
        primary = primary_ncc_decision(
            exact_certified=exact_certified,
            support_retention=supports["ncc7"].retention,
            baseline_ncc_loss=utility["ncc7_baseline_common"],
            candidate_ncc_loss=utility["ncc7_candidate_common"],
        )
        returned_field = candidate_field if primary.accept else initial_field
        arms.append(
            {
                "arm_index": spec.arm_index,
                "arm_id": arm_id,
                "selectable": spec.selectable,
                "stress_only": spec.stress_only,
                "proposal": proposal_metadata[arm_id],
                "requested_state": {
                    "selectable": False,
                    "field": requested_field,
                    "exact": requested_exact,
                    "exact_certified": requested_exact_certified,
                    "geometry": requested_geometry,
                    "fast_cert_bound_work_eps": requested_fast_bound,
                    "residual_rms_from_stored_state": requested_state_residual_rms,
                    "supports": _support_records(requested_supports),
                    "utility": requested_utility,
                },
                "operator": operator_report,
                "exact": exact,
                "exact_certified": exact_certified,
                "candidate_field": candidate_field,
                "candidate_geometry": candidate_geometry,
                "postclip_residual_rms": postclip_residual_rms,
                "primary_geometry_defined": _metric_ok(candidate_geometry, MATHEMATICAL_SDLOGJ_CROP2),
                "supports": _support_records(supports),
                "utility": utility,
                "primary_decision": primary.to_dict(),
                "action": "ACCEPT" if primary.accept else "ROLLBACK",
                "reason": primary.reason,
                "returned_field": returned_field,
                "returned_residual_rms": postclip_residual_rms if primary.accept else 0.0,
                "rollback_byte_identical": not primary.accept and returned_field == initial_field,
            }
        )

    payload = {
        "schema": f"ctcf-search-c3a-decision-case-{SCHEMA_VERSION}",
        "status": "COMPLETE",
        "case_id": case_id,
        "decision_contract_sha256": contract_sha,
        "image_input_sha256": contract["image_inputs"][case_id]["sha256"],
        "c2_golden": golden,
        "labels_loaded_to_device": False,
        "test_split_accessed": False,
        "initial": {
            "field": initial_field,
            "report": initial_report,
            "geometry": baseline_geometry,
            "storage": {
                "psi_retained": True,
                "phi_retained": False,
                "phi_reason": "C3a decisions and evaluation consume displacement Psi only",
            },
        },
        "arms": arms,
        "execution": execution,
        "elapsed_sec": time.perf_counter() - started,
        "peak_gpu_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0,
    }
    serialized = json.dumps(payload, sort_keys=True)
    if '"dice' in serialized.lower() or ".pkl" in serialized.lower():
        raise RuntimeError("Decision marker leaked a Dice value or raw container path")
    atomic_write_json(marker, payload)
    if device.type == "cuda":
        torch.cuda.empty_cache()


def _validate_decision_case(
    payload: dict[str, Any],
    marker: Path,
    case_id: str,
    contract: dict[str, Any],
    contract_sha: str,
    *,
    verify_heavy: bool,
) -> list[dict[str, Any]]:
    arms = payload.get("arms")
    if (
        payload.get("schema") != f"ctcf-search-c3a-decision-case-{SCHEMA_VERSION}"
        or payload.get("status") != "COMPLETE"
        or payload.get("case_id") != case_id
        or payload.get("decision_contract_sha256") != contract_sha
        or payload.get("labels_loaded_to_device") is not False
        or payload.get("test_split_accessed") is not False
        or payload.get("c2_golden") != contract["c2_decision_goldens"]["cases"][case_id]
        or not isinstance(arms, list)
        or [(row.get("arm_index"), row.get("arm_id")) for row in arms]
        != [(spec.arm_index, spec.arm_id) for spec in ARM_SPECS]
    ):
        raise RuntimeError(f"Invalid C3a decision marker: {marker}")
    if '"dice' in json.dumps(payload, sort_keys=True).lower():
        raise RuntimeError(f"C3a decision marker contains a forbidden Dice value: {marker}")
    initial_field = (payload.get("initial") or {}).get("field") or {}
    initial_exact = ((payload.get("initial") or {}).get("report") or {}).get("psi_exact") or {}
    initial_geometry = (payload.get("initial") or {}).get("geometry") or {}
    golden = contract["c2_decision_goldens"]["cases"][case_id]
    if (
        initial_field.get("array_sha256") != golden["initial_array_sha256"]
        or initial_exact.get("status") != "CERTIFIED"
        or initial_exact.get("certified") is not True
        or initial_exact.get("sha256") != initial_field.get("array_sha256")
        or set(initial_geometry) != set(METRIC_SPECS)
    ):
        raise RuntimeError(f"C3a initial parity changed: {marker}")
    _check_exact_digital_consistency(initial_geometry, exact_certified=True, label=f"{case_id}/baseline")
    if verify_heavy:
        _resolve_field(contract, initial_field)
    for spec, row in zip(ARM_SPECS, arms, strict=True):
        exact = row.get("exact") or {}
        certified = exact.get("status") == "CERTIFIED" and exact.get("certified") is True
        supports = row.get("supports") or {}
        utility = row.get("utility") or {}
        requested_state = row.get("requested_state") or {}
        requested_supports = requested_state.get("supports") or {}
        requested_utility = requested_state.get("utility") or {}
        requested_exact = requested_state.get("exact") or {}
        requested_certified = requested_exact.get("status") == "CERTIFIED" and requested_exact.get("certified") is True
        proposal = row.get("proposal") or {}
        posterior = proposal.get("posterior_diagnostics")
        roughness = proposal.get("postprocessed_residual_roughness") or {}
        if row.get("exact_certified") is not certified or set(supports) != {"mind", "ncc7", "ncc9"}:
            raise RuntimeError(f"Invalid C3a exact/support record: {marker}/{spec.arm_id}")
        _validate_support_utility(supports, utility, label=f"{marker}/{spec.arm_id}/postclip")
        _validate_support_utility(
            requested_supports,
            requested_utility,
            label=f"{marker}/{spec.arm_id}/requested",
        )
        if (
            requested_state.get("selectable") is not False
            or requested_state.get("exact_certified") is not requested_certified
            or requested_exact.get("sha256") != (requested_state.get("field") or {}).get("array_sha256")
            or set(requested_state.get("geometry") or {}) != set(METRIC_SPECS)
            or not isinstance(requested_state.get("fast_cert_bound_work_eps"), (int, float))
            or not math.isfinite(float(requested_state["fast_cert_bound_work_eps"]))
            or not isinstance(requested_state.get("residual_rms_from_stored_state"), (int, float))
            or not math.isfinite(float(requested_state["residual_rms_from_stored_state"]))
            or float(requested_state["residual_rms_from_stored_state"]) < 0.0
        ):
            raise RuntimeError(f"Invalid C3a requested-state diagnostic: {marker}/{spec.arm_id}")
        _check_exact_digital_consistency(
            requested_state["geometry"],
            exact_certified=requested_certified,
            label=f"{case_id}/{spec.arm_id}/requested",
        )
        posterior_numeric_keys = (
            "top1_top2_valid_logit_gap_mean",
            "posterior_peak_probability_mean",
            "entropy_nats_mean",
            "invalid_offset_fraction",
            "posterior_mean_l2_norm_mean",
            "confidence_weighted_mean_l2_norm_mean",
        )
        if (
            not isinstance(proposal.get("requested_rms"), (int, float))
            or not math.isfinite(float(proposal["requested_rms"]))
            or roughness.get("metric_id") != RMS_FIRST_DIFFERENCE_ROUGHNESS_ID
            or not isinstance(roughness.get("rms_vector_first_difference"), (int, float))
            or not math.isfinite(float(roughness["rms_vector_first_difference"]))
            or float(roughness["rms_vector_first_difference"]) < 0.0
            or not isinstance(roughness.get("pair_count"), int)
            or roughness["pair_count"] < 1
            or not isinstance(roughness.get("axis_pair_counts_zyx"), list)
            or len(roughness["axis_pair_counts_zyx"]) != 3
            or sum(roughness["axis_pair_counts_zyx"]) != roughness["pair_count"]
        ):
            raise RuntimeError(f"Invalid C3a residual diagnostic: {marker}/{spec.arm_id}")
        if spec.arm_id == "zero_update":
            if posterior is not None:
                raise RuntimeError(f"C3a zero arm unexpectedly has posterior diagnostics: {marker}")
        elif (
            not isinstance(posterior, dict)
            or posterior.get("diagnostic_id") != POSTERIOR_DIAGNOSTICS_ID
            or posterior.get("candidate_count") != 27
            or not isinstance(posterior.get("active_voxel_count"), int)
            or posterior["active_voxel_count"] < 1
            or not all(
                isinstance(posterior.get(key), (int, float)) and math.isfinite(float(posterior[key]))
                for key in posterior_numeric_keys
            )
            or not 0.0 <= float(posterior["posterior_peak_probability_mean"]) <= 1.0
            or not 0.0 <= float(posterior["invalid_offset_fraction"]) <= 1.0
            or (
                posterior.get("confidence_to_mean_l2_norm_ratio") is not None
                and (
                    not isinstance(posterior["confidence_to_mean_l2_norm_ratio"], (int, float))
                    or not math.isfinite(float(posterior["confidence_to_mean_l2_norm_ratio"]))
                    or not 0.0 <= float(posterior["confidence_to_mean_l2_norm_ratio"]) <= 1.0 + 1e-6
                )
            )
            or (
                posterior.get("confidence_to_mean_l2_norm_ratio") is None
                and (
                    float(posterior["posterior_mean_l2_norm_mean"]) != 0.0
                    or float(posterior["confidence_weighted_mean_l2_norm_mean"]) != 0.0
                )
            )
        ):
            raise RuntimeError(f"Invalid C3a posterior diagnostic: {marker}/{spec.arm_id}")
        if spec.arm_id == "raw_conf_post1" and (
            proposal.get("independent_decoder_parity_atol") != CONTROL_DECODER_PARITY_ATOL
            or not isinstance(proposal.get("independent_decoder_max_abs_difference"), (int, float))
            or float(proposal["independent_decoder_max_abs_difference"]) > CONTROL_DECODER_PARITY_ATOL
        ):
            raise RuntimeError(f"C3a raw decoder parity is not established: {marker}/{spec.arm_id}")
        decision = primary_ncc_decision(
            exact_certified=certified,
            support_retention=float(supports["ncc7"]["retention"]),
            baseline_ncc_loss=utility["ncc7_baseline_common"],
            candidate_ncc_loss=utility["ncc7_candidate_common"],
        )
        if (
            row.get("primary_decision") != decision.to_dict()
            or row.get("action") != ("ACCEPT" if decision.accept else "ROLLBACK")
            or row.get("reason") != decision.reason
            or row.get("selectable") is not spec.selectable
            or row.get("stress_only") is not spec.stress_only
            or set(row.get("candidate_geometry") or {}) != set(METRIC_SPECS)
            or exact.get("sha256") != (row.get("candidate_field") or {}).get("array_sha256")
            or not isinstance(row.get("postclip_residual_rms"), (int, float))
            or not math.isfinite(float(row["postclip_residual_rms"]))
            or float(row["postclip_residual_rms"]) < 0.0
            or not isinstance(row.get("returned_residual_rms"), (int, float))
            or not math.isfinite(float(row["returned_residual_rms"]))
            or not math.isclose(
                float(row["returned_residual_rms"]),
                float(row["postclip_residual_rms"]) if decision.accept else 0.0,
                rel_tol=FLOAT32_PARITY_RTOL,
                abs_tol=FLOAT32_PARITY_ATOL,
            )
        ):
            raise RuntimeError(f"Invalid C3a policy reconstruction: {marker}/{spec.arm_id}")
        _check_exact_digital_consistency(
            row["candidate_geometry"],
            exact_certified=certified,
            label=f"{case_id}/{spec.arm_id}",
        )
        expected = golden["control_candidate_array_sha256"].get(spec.arm_id)
        if expected is not None and (row.get("candidate_field") or {}).get("array_sha256") != expected:
            raise RuntimeError(f"C3a C2 control parity changed: {marker}/{spec.arm_id}")
        returned = row.get("returned_field") or {}
        expected_returned = row.get("candidate_field") if decision.accept else initial_field
        if returned != expected_returned or row.get("rollback_byte_identical") is not (not decision.accept):
            raise RuntimeError(f"Invalid C3a rollback record: {marker}/{spec.arm_id}")
        if verify_heavy:
            _resolve_field(contract, requested_state["field"])
            _resolve_field(contract, row["candidate_field"])
            _resolve_field(contract, returned)
    _validate_arm_construction_invariants(arms, label=str(marker))
    execution = payload.get("execution") or {}
    load = execution.get("checkpoint_load_report") or {}
    expected_shard = expected_shard_for_case(contract, case_id)
    if (
        not str(execution.get("device", "")).startswith("cuda")
        or execution.get("deterministic") is not True
        or execution.get("shard_index") != expected_shard
        or execution.get("physical_gpu") != contract["shard_to_physical_gpu"][str(expected_shard)]
        or execution.get("checkpoint_sha256") != contract["checkpoint_sha256"]
        or load.get("strict") is not True
        or bool(load.get("unexpected_keys"))
        or set(load.get("missing_keys") or []) != set(load.get("allowed_missing_buffers") or [])
    ):
        raise RuntimeError(f"Invalid C3a decision execution provenance: {marker}")
    return arms


def _worker_paths(root: Path, phase: str, attempt_id: str, shard: int) -> tuple[Path, Path]:
    directory = root / "workers" / phase / "attempts" / attempt_id
    return directory / f"worker_{shard:02d}.json", directory / f"worker_{shard:02d}_failure.json"


def decision_worker_stage(args: argparse.Namespace) -> int:
    root = args.run_root.resolve()
    contract, contract_sha = _load_decision_contract(root, args.decision_contract_sha256)
    _assert_clean_code(contract["git_head"], "decision worker")
    _assert_runtime_signature(contract["runtime_signature"], "decision worker")
    if args.num_shards != contract["num_shards"] or not 0 <= args.shard_index < args.num_shards:
        raise RuntimeError("Decision worker shard parameters differ from C3a contract")
    if args.physical_gpu != contract["shard_to_physical_gpu"][str(args.shard_index)]:
        raise RuntimeError("Decision worker physical GPU differs from C3a contract")
    assigned = contract["shards"][str(args.shard_index)]
    for case_id in ["atlas", *assigned]:
        _verify_image_record(contract["image_inputs"][case_id])
    checkpoint = Path(contract["checkpoint"])
    if not checkpoint.is_file() or sha256_file(checkpoint) != contract["checkpoint_sha256"]:
        raise RuntimeError("Checkpoint changed after C3a extraction")
    marker, failure = _worker_paths(root, "decision", args.attempt_id, args.shard_index)
    if marker.exists() or failure.exists():
        raise RuntimeError("Decision worker attempt output already exists")
    pending: list[str] = []
    reused: list[str] = []
    for case_id in assigned:
        case_marker = _decision_case_path(root, case_id)
        if case_marker.is_file():
            _validate_decision_case(
                _load_json(case_marker), case_marker, case_id, contract, contract_sha, verify_heavy=True
            )
            reused.append(case_id)
        else:
            pending.append(case_id)
    started = utc_now()
    computed: list[str] = []
    try:
        load_report: dict[str, Any] = {"strict": None, "missing_keys": [], "unexpected_keys": []}
        if pending:
            device = setup_device(args.gpu, seed=contract["seed"], deterministic=True)
            if device.type != "cuda":
                raise RuntimeError("C3a decision worker requires CUDA")
            adapter, model, load_report = build_model(str(checkpoint), contract["config"], device)
            atlas = _verify_image_record(contract["image_inputs"]["atlas"])
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
                "checkpoint_sha256": contract["checkpoint_sha256"],
                "checkpoint_load_report": load_report,
            }
            for index, case_id in enumerate(pending, start=1):
                print(
                    f"[decision {args.shard_index + 1}/{args.num_shards}] [{index}/{len(pending)}] IXI {case_id}",
                    flush=True,
                )
                case_image = _verify_image_record(contract["image_inputs"][case_id])
                _run_decision_case(
                    case_id=case_id,
                    atlas_image=atlas,
                    case_image=case_image,
                    adapter=adapter,
                    model=model,
                    device=device,
                    root=root,
                    contract=contract,
                    contract_sha=contract_sha,
                    execution=execution,
                )
                computed.append(case_id)
        report = {
            "schema": f"ctcf-search-c3a-decision-worker-{SCHEMA_VERSION}",
            "status": "COMPLETE",
            "phase": "decision",
            "attempt_id": args.attempt_id,
            "shard_index": args.shard_index,
            "physical_gpu": args.physical_gpu,
            "decision_contract_sha256": contract_sha,
            "assigned_case_ids": assigned,
            "computed_case_ids": computed,
            "reused_case_ids": reused,
            "checkpoint_load_report": load_report,
            "runtime_signature": contract["runtime_signature"],
            "labels_loaded_to_device": False,
            "started_at_utc": started,
            "completed_at_utc": utc_now(),
        }
        atomic_write_json(marker, report)
    except Exception as error:
        atomic_write_json(
            failure,
            {
                "schema": f"ctcf-search-c3a-decision-worker-failure-{SCHEMA_VERSION}",
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


def _validate_worker_report(
    payload: dict[str, Any],
    *,
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
        payload.get("schema") != f"ctcf-search-c3a-{phase}-worker-{SCHEMA_VERSION}"
        or payload.get("status") != "COMPLETE"
        or payload.get("phase") != phase
        or payload.get("attempt_id") != attempt_id
        or payload.get("shard_index") != shard
        or payload.get("physical_gpu") != contract["shard_to_physical_gpu"][str(shard)]
        or payload.get("decision_contract_sha256") != contract_sha
        or payload.get("assigned_case_ids") != assigned
        or len(computed) + len(reused) != len(assigned)
        or set(computed) & set(reused)
        or set([*computed, *reused]) != set(assigned)
        or payload.get("runtime_signature") != contract["runtime_signature"]
    ):
        raise RuntimeError(f"Invalid C3a {phase} worker report for shard {shard}")
    if phase == "decision" and payload.get("labels_loaded_to_device") is not False:
        raise RuntimeError("Decision worker report claims label access")
    if phase == "evaluation" and (
        payload.get("labels_loaded_this_attempt") is not bool(computed)
        or payload.get("all_cases_have_postbarrier_evaluation_evidence") is not True
    ):
        raise RuntimeError("Evaluation worker report has inconsistent post-barrier label evidence")


def decision_barrier_stage(args: argparse.Namespace) -> int:
    root = args.run_root.resolve()
    contract, contract_sha = _load_decision_contract(root, args.decision_contract_sha256)
    _assert_clean_code(contract["git_head"], "decision barrier")
    worker_files: list[dict[str, Any]] = []
    seen: list[str] = []
    for shard in range(contract["num_shards"]):
        path, _ = _worker_paths(root, "decision", args.attempt_id, shard)
        payload = _load_json(path)
        _validate_worker_report(
            payload,
            phase="decision",
            attempt_id=args.attempt_id,
            shard=shard,
            contract=contract,
            contract_sha=contract_sha,
        )
        seen.extend(contract["shards"][str(shard)])
        worker_files.append({"path": path.relative_to(root).as_posix(), "sha256": sha256_file(path)})
    validate_shard_partition(
        contract,
        seen,
        EXPECTED_CASES,
        "C3a decision worker partition has missing, duplicate, or reordered cases",
    )
    case_hashes: dict[str, str] = {}
    for case_id in contract["case_ids"]:
        path = _decision_case_path(root, case_id)
        _validate_decision_case(_load_json(path), path, case_id, contract, contract_sha, verify_heavy=True)
        case_hashes[case_id] = sha256_file(path)
    path = root / DECISION_BARRIER_NAME
    if path.exists():
        existing = _load_json(path)
        if (
            existing.get("schema") != f"ctcf-search-c3a-decision-barrier-{SCHEMA_VERSION}"
            or existing.get("status") != "COMPLETE"
            or existing.get("decision_contract_sha256") != contract_sha
            or existing.get("decision_case_sha256") != case_hashes
        ):
            raise RuntimeError("Existing decision barrier differs from the current immutable decisions")
        print(json.dumps({"decision_barrier_sha256": sha256_file(path), "n_cases": len(case_hashes)}))
        return 0
    barrier = {
        "schema": f"ctcf-search-c3a-decision-barrier-{SCHEMA_VERSION}",
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
    atomic_write_json(path, barrier)
    print(json.dumps({"decision_barrier_sha256": sha256_file(path), "n_cases": len(case_hashes)}))
    return 0


def _load_barrier(
    root: Path,
    expected_sha: str,
    contract: dict[str, Any],
    contract_sha: str,
) -> tuple[dict[str, Any], str]:
    path = root / DECISION_BARRIER_NAME
    actual = sha256_file(path)
    if actual != _require_sha(expected_sha, "decision barrier SHA-256"):
        raise RuntimeError("C3a decision barrier hash mismatch")
    barrier = _load_json(path)
    if (
        barrier.get("schema") != f"ctcf-search-c3a-decision-barrier-{SCHEMA_VERSION}"
        or barrier.get("status") != "COMPLETE"
        or barrier.get("protocol_id") != PROTOCOL_ID
        or barrier.get("decision_contract_sha256") != contract_sha
        or barrier.get("decision_workers_received_label_inputs") is not False
        or barrier.get("test_split_accessed") is not False
        or set(barrier.get("decision_case_sha256") or {}) != set(contract["case_ids"])
    ):
        raise RuntimeError("Invalid C3a decision barrier")
    for case_id, digest in barrier["decision_case_sha256"].items():
        if sha256_file(_decision_case_path(root, case_id)) != digest:
            raise RuntimeError(f"Decision snapshot changed after barrier: {case_id}")
    return barrier, actual


def _evaluation_case_path(root: Path, case_id: str) -> Path:
    return root / "cases" / case_id / "evaluation_complete.json"


def _run_evaluation_case(
    *,
    index: int,
    case_id: str,
    dataset: Any,
    labels: tuple[int, ...],
    device: torch.device,
    root: Path,
    source: dict[str, Any],
    decision: dict[str, Any],
    decision_sha: str,
    barrier: dict[str, Any],
    barrier_sha: str,
    execution: dict[str, Any],
    c2_evaluation_goldens: dict[str, Any],
) -> None:
    marker = _evaluation_case_path(root, case_id)
    if marker.is_file():
        _validate_evaluation_case(
            _load_json(marker),
            marker,
            case_id,
            decision,
            decision_sha,
            barrier,
            barrier_sha,
            c2_evaluation_goldens,
        )
        return
    decision_path = _decision_case_path(root, case_id)
    frozen_decision_sha = barrier["decision_case_sha256"][case_id]
    if sha256_file(decision_path) != frozen_decision_sha:
        raise RuntimeError(f"Decision snapshot changed before evaluation: {case_id}")
    decision_payload = _load_json(decision_path)
    decision_arms = _validate_decision_case(
        decision_payload, decision_path, case_id, decision, decision_sha, verify_heavy=True
    )
    x_cpu, y_cpu, moving_seg_cpu, fixed_seg_cpu = dataset[index]
    if (
        _sha256_array(x_cpu.numpy()) != decision["image_inputs"]["atlas"]["array_sha256"]
        or _sha256_array(y_cpu.numpy()) != decision["image_inputs"][case_id]["array_sha256"]
    ):
        raise RuntimeError(f"Evaluation image/segmentation pair does not match the decision cache for {case_id}")
    moving_seg = moving_seg_cpu.unsqueeze(0).to(device)
    fixed_seg = fixed_seg_cpu.unsqueeze(0).to(device)
    initial_record = decision_payload["initial"]["field"]
    initial = load_flow_npz(_resolve_field(decision, initial_record)).to(device)
    baseline_dice = dice_score(initial, moving_seg, fixed_seg, labels)
    c2_golden = c2_evaluation_goldens["cases"][case_id]
    if not math.isclose(
        baseline_dice,
        float(c2_golden["baseline_dice"]),
        rel_tol=0.0,
        abs_tol=C2_DICE_PARITY_ATOL,
    ):
        raise RuntimeError(f"PARITY_GAP: baseline Dice differs from frozen C2 for {case_id}")
    arms: list[dict[str, Any]] = []
    for decision_row in decision_arms:
        requested = load_flow_npz(_resolve_field(decision, decision_row["requested_state"]["field"])).to(device)
        candidate = load_flow_npz(_resolve_field(decision, decision_row["candidate_field"])).to(device)
        requested_dice = dice_score(requested, moving_seg, fixed_seg, labels)
        candidate_dice = dice_score(candidate, moving_seg, fixed_seg, labels)
        expected_c2_dice = c2_golden["control_candidate_dice"].get(decision_row["arm_id"])
        if expected_c2_dice is not None and not math.isclose(
            candidate_dice,
            float(expected_c2_dice),
            rel_tol=0.0,
            abs_tol=C2_DICE_PARITY_ATOL,
        ):
            raise RuntimeError(f"PARITY_GAP: control Dice differs from frozen C2 for {case_id}")
        returned_dice = candidate_dice if decision_row["action"] == "ACCEPT" else baseline_dice
        arms.append(
            {
                "arm_index": decision_row["arm_index"],
                "arm_id": decision_row["arm_id"],
                "baseline_dice": baseline_dice,
                "requested_diagnostic_dice": requested_dice,
                "requested_diagnostic_dice_delta": requested_dice - baseline_dice,
                "capacity_candidate_dice": candidate_dice,
                "capacity_dice_delta": candidate_dice - baseline_dice,
                "primary_returned_dice": returned_dice,
                "primary_dice_delta": returned_dice - baseline_dice,
                "primary_action": decision_row["action"],
            }
        )
    if sha256_file(decision_path) != frozen_decision_sha:
        raise RuntimeError(f"Decision snapshot changed during evaluation: {case_id}")
    payload = {
        "schema": f"ctcf-search-c3a-evaluation-case-{SCHEMA_VERSION}",
        "status": "COMPLETE",
        "case_id": case_id,
        "decision_contract_sha256": decision_sha,
        "decision_barrier_sha256": barrier_sha,
        "decision_case_sha256": frozen_decision_sha,
        "source_input_sha256": source["raw_inputs"][case_id]["sha256"],
        "labels_loaded_after_barrier": True,
        "test_split_accessed": False,
        "labels": list(labels),
        "c2_dice_parity_verified": True,
        "arms": arms,
        "execution": execution,
    }
    atomic_write_json(marker, payload)
    if device.type == "cuda":
        torch.cuda.empty_cache()


def _validate_evaluation_case(
    payload: dict[str, Any],
    marker: Path,
    case_id: str,
    contract: dict[str, Any],
    contract_sha: str,
    barrier: dict[str, Any],
    barrier_sha: str,
    c2_evaluation_goldens: dict[str, Any],
) -> list[dict[str, Any]]:
    arms = payload.get("arms")
    from experiments.core.inference_metrics import metric_profile_for

    expected_labels = list(metric_profile_for("IXI").labels)
    if (
        payload.get("schema") != f"ctcf-search-c3a-evaluation-case-{SCHEMA_VERSION}"
        or payload.get("status") != "COMPLETE"
        or payload.get("case_id") != case_id
        or payload.get("decision_contract_sha256") != contract_sha
        or payload.get("decision_barrier_sha256") != barrier_sha
        or payload.get("decision_case_sha256") != barrier["decision_case_sha256"][case_id]
        or payload.get("source_input_sha256") != contract["source_container_sha256"][case_id]
        or payload.get("labels_loaded_after_barrier") is not True
        or payload.get("test_split_accessed") is not False
        or payload.get("labels") != expected_labels
        or payload.get("c2_dice_parity_verified") is not True
        or not isinstance(arms, list)
        or [(row.get("arm_index"), row.get("arm_id")) for row in arms]
        != [(spec.arm_index, spec.arm_id) for spec in ARM_SPECS]
    ):
        raise RuntimeError(f"Invalid C3a evaluation marker: {marker}")
    frozen_decision_path = marker.with_name("decision_complete.json")
    if sha256_file(frozen_decision_path) != barrier["decision_case_sha256"][case_id]:
        raise RuntimeError(f"C3a decision changed before evaluation validation: {case_id}")
    frozen_decision_payload = _load_json(frozen_decision_path)
    decision_by_arm = {row["arm_id"]: row for row in (frozen_decision_payload.get("arms") or [])}
    baselines = [float(row["baseline_dice"]) for row in arms]
    if not baselines or any(value != baselines[0] for value in baselines[1:]):
        raise RuntimeError(f"C3a evaluation arms do not share one frozen baseline: {marker}")
    c2_golden = c2_evaluation_goldens["cases"][case_id]
    if not math.isclose(
        baselines[0],
        float(c2_golden["baseline_dice"]),
        rel_tol=0.0,
        abs_tol=C2_DICE_PARITY_ATOL,
    ):
        raise RuntimeError(f"C3a baseline Dice parity changed: {marker}")
    for row in arms:
        values = [
            row.get("baseline_dice"),
            row.get("requested_diagnostic_dice"),
            row.get("requested_diagnostic_dice_delta"),
            row.get("capacity_candidate_dice"),
            row.get("capacity_dice_delta"),
            row.get("primary_returned_dice"),
            row.get("primary_dice_delta"),
        ]
        if not all(isinstance(value, (int, float)) and math.isfinite(float(value)) for value in values):
            raise RuntimeError(f"Non-finite C3a Dice output: {marker}/{row.get('arm_id')}")
        if not all(
            0.0 <= float(row[key]) <= 1.0
            for key in (
                "baseline_dice",
                "requested_diagnostic_dice",
                "capacity_candidate_dice",
                "primary_returned_dice",
            )
        ):
            raise RuntimeError(f"Out-of-range C3a Dice output: {marker}/{row.get('arm_id')}")
        frozen_decision = decision_by_arm.get(row.get("arm_id")) or {}
        expected_c2_dice = c2_golden["control_candidate_dice"].get(row.get("arm_id"))
        if expected_c2_dice is not None and not math.isclose(
            float(row["capacity_candidate_dice"]),
            float(expected_c2_dice),
            rel_tol=0.0,
            abs_tol=C2_DICE_PARITY_ATOL,
        ):
            raise RuntimeError(f"C3a control Dice parity changed: {marker}/{row.get('arm_id')}")
        expected_action = frozen_decision.get("action")
        expected_returned = (
            float(row["capacity_candidate_dice"]) if expected_action == "ACCEPT" else float(row["baseline_dice"])
        )
        if (
            row.get("primary_action") != expected_action
            or float(row["requested_diagnostic_dice_delta"])
            != float(row["requested_diagnostic_dice"]) - float(row["baseline_dice"])
            or float(row["primary_returned_dice"]) != expected_returned
            or float(row["capacity_dice_delta"]) != float(row["capacity_candidate_dice"]) - float(row["baseline_dice"])
            or float(row["primary_dice_delta"]) != float(row["primary_returned_dice"]) - float(row["baseline_dice"])
        ):
            raise RuntimeError(f"Invalid C3a Dice arithmetic: {marker}/{row.get('arm_id')}")
    execution = payload.get("execution") or {}
    expected_shard = expected_shard_for_case(contract, case_id)
    if (
        execution.get("phase") != "evaluation"
        or execution.get("shard_index") != expected_shard
        or execution.get("physical_gpu") != contract["shard_to_physical_gpu"][str(expected_shard)]
        or execution.get("deterministic") is not True
        or execution.get("labels_loaded_after_barrier") is not True
        or not str(execution.get("device", "")).startswith("cuda")
    ):
        raise RuntimeError(f"Invalid C3a evaluation execution provenance: {marker}")
    return arms


def evaluation_worker_stage(args: argparse.Namespace) -> int:
    root = args.run_root.resolve()
    source, source_sha = _load_source_contract(root, args.source_contract_sha256)
    decision, decision_sha = _load_decision_contract(root, args.decision_contract_sha256)
    if decision.get("source_contract_sha256") != source_sha:
        raise RuntimeError("Decision contract is not linked to this source contract")
    _assert_clean_code(decision["git_head"], "evaluation worker")
    _assert_runtime_signature(decision["runtime_signature"], "evaluation worker")
    barrier, barrier_sha = _load_barrier(root, args.barrier_sha256, decision, decision_sha)
    c2_evaluation_goldens = _load_c2_evaluation_goldens(root, source)
    if args.num_shards != decision["num_shards"] or not 0 <= args.shard_index < args.num_shards:
        raise RuntimeError("Evaluation worker shard parameters differ from C3a contract")
    if args.physical_gpu != decision["shard_to_physical_gpu"][str(args.shard_index)]:
        raise RuntimeError("Evaluation worker physical GPU differs from C3a contract")
    assigned = decision["shards"][str(args.shard_index)]
    for case_id in ["atlas", *assigned]:
        _verify_file_record(source["raw_inputs"][case_id])
    marker, failure = _worker_paths(root, "evaluation", args.attempt_id, args.shard_index)
    if marker.exists() or failure.exists():
        raise RuntimeError("Evaluation worker attempt output already exists")
    pending: list[str] = []
    reused: list[str] = []
    for case_id in assigned:
        case_marker = _evaluation_case_path(root, case_id)
        if case_marker.is_file():
            _validate_evaluation_case(
                _load_json(case_marker),
                case_marker,
                case_id,
                decision,
                decision_sha,
                barrier,
                barrier_sha,
                c2_evaluation_goldens,
            )
            reused.append(case_id)
        else:
            pending.append(case_id)
    started = utc_now()
    computed: list[str] = []
    try:
        if pending:
            from experiments.core.inference_metrics import metric_profile_for
            from experiments.core.inference_runtime import build_infer_dataset

            device = setup_device(args.gpu, seed=decision["seed"], deterministic=True)
            if device.type != "cuda":
                raise RuntimeError("C3a evaluation worker requires CUDA")
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
                "seed": decision["seed"],
                "deterministic": True,
                "labels_loaded_after_barrier": True,
            }
            for index, case_id in enumerate(pending):
                print(
                    f"[evaluation {args.shard_index + 1}/{args.num_shards}] [{index + 1}/{len(pending)}] IXI {case_id}",
                    flush=True,
                )
                _run_evaluation_case(
                    index=index,
                    case_id=case_id,
                    dataset=dataset,
                    labels=labels,
                    device=device,
                    root=root,
                    source=source,
                    decision=decision,
                    decision_sha=decision_sha,
                    barrier=barrier,
                    barrier_sha=barrier_sha,
                    execution=execution,
                    c2_evaluation_goldens=c2_evaluation_goldens,
                )
                computed.append(case_id)
        report = {
            "schema": f"ctcf-search-c3a-evaluation-worker-{SCHEMA_VERSION}",
            "status": "COMPLETE",
            "phase": "evaluation",
            "attempt_id": args.attempt_id,
            "shard_index": args.shard_index,
            "physical_gpu": args.physical_gpu,
            "decision_contract_sha256": decision_sha,
            "decision_barrier_sha256": barrier_sha,
            "assigned_case_ids": assigned,
            "computed_case_ids": computed,
            "reused_case_ids": reused,
            "labels_loaded_this_attempt": bool(computed),
            "all_cases_have_postbarrier_evaluation_evidence": True,
            "runtime_signature": decision["runtime_signature"],
            "started_at_utc": started,
            "completed_at_utc": utc_now(),
        }
        atomic_write_json(marker, report)
    except Exception as error:
        atomic_write_json(
            failure,
            {
                "schema": f"ctcf-search-c3a-evaluation-worker-failure-{SCHEMA_VERSION}",
                "status": "FAILED",
                "phase": "evaluation",
                "attempt_id": args.attempt_id,
                "shard_index": args.shard_index,
                "decision_contract_sha256": decision_sha,
                "decision_barrier_sha256": barrier_sha,
                "computed_case_ids": computed,
                "error_type": type(error).__name__,
                "error": str(error),
                "completed_at_utc": utc_now(),
            },
        )
        raise
    return 0


def _paired_dict(candidate: list[float], baseline: list[float]) -> dict[str, Any]:
    return paired_summary(np.asarray(candidate), np.asarray(baseline)).to_dict()


def _contrast(rows_by_arm: dict[str, list[dict[str, Any]]], first: str, second: str) -> dict[str, Any]:
    requested = paired_summary(
        [row["requested_diagnostic_dice"] for row in rows_by_arm[first]],
        [row["requested_diagnostic_dice"] for row in rows_by_arm[second]],
    )
    capacity = paired_summary(
        [row["capacity_candidate_dice"] for row in rows_by_arm[first]],
        [row["capacity_candidate_dice"] for row in rows_by_arm[second]],
    )
    primary = paired_summary(
        [row["primary_returned_dice"] for row in rows_by_arm[first]],
        [row["primary_returned_dice"] for row in rows_by_arm[second]],
    )
    capacity_exact_eligible = all(row["exact_certified"] is True for row in rows_by_arm[first]) and all(
        row["exact_certified"] is True for row in rows_by_arm[second]
    )
    requested_exact_eligible = all(row["requested_exact_certified"] is True for row in rows_by_arm[first]) and all(
        row["requested_exact_certified"] is True for row in rows_by_arm[second]
    )
    return {
        "first_arm_id": first,
        "second_arm_id": second,
        "estimand": "paired first-minus-second on the same 58 already-open validation cases",
        "requested_preclip": requested.to_dict(),
        "requested_preclip_exact_eligible": requested_exact_eligible,
        "requested_preclip_descriptive_wins": wins(requested),
        "requested_preclip_exact_wins": requested_exact_eligible and wins(requested),
        "capacity": capacity.to_dict(),
        "capacity_estimand": "accept-all exact post-clip candidate",
        "capacity_exact_eligible": capacity_exact_eligible,
        "capacity_wins": capacity_exact_eligible and wins(capacity),
        "primary_policy": primary.to_dict(),
        "primary_policy_wins": wins(primary),
    }


def _build_summaries(
    decision_payloads: dict[str, dict[str, Any]],
    evaluation_payloads: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any], dict[str, Any], bool]:
    flat_rows: list[dict[str, Any]] = []
    rows_by_arm: dict[str, list[dict[str, Any]]] = {spec.arm_id: [] for spec in ARM_SPECS}
    baseline_geometry_defined = True
    metric_error_present = False
    for case_id in decision_payloads:
        decision = decision_payloads[case_id]
        evaluation = evaluation_payloads[case_id]
        eval_by_arm = {row["arm_id"]: row for row in evaluation["arms"]}
        baseline_bundle = decision["initial"]["geometry"]
        metric_error_present |= any(metric.get("status") == "ERROR" for metric in baseline_bundle.values())
        baseline_geometry_defined &= _metric_ok(baseline_bundle, MATHEMATICAL_SDLOGJ_CROP2)
        baseline_geometry = (
            _metric_value(baseline_bundle, MATHEMATICAL_SDLOGJ_CROP2)
            if _metric_ok(baseline_bundle, MATHEMATICAL_SDLOGJ_CROP2)
            else None
        )
        for row in decision["arms"]:
            arm_id = row["arm_id"]
            evaluated = eval_by_arm[arm_id]
            candidate_bundle = row["candidate_geometry"]
            requested_bundle = row["requested_state"]["geometry"]
            returned_bundle = candidate_bundle if row["action"] == "ACCEPT" else baseline_bundle
            candidate_geometry = (
                _metric_value(candidate_bundle, MATHEMATICAL_SDLOGJ_CROP2)
                if _metric_ok(candidate_bundle, MATHEMATICAL_SDLOGJ_CROP2)
                else None
            )
            requested_geometry = (
                _metric_value(requested_bundle, MATHEMATICAL_SDLOGJ_CROP2)
                if _metric_ok(requested_bundle, MATHEMATICAL_SDLOGJ_CROP2)
                else None
            )
            returned_geometry = (
                _metric_value(returned_bundle, MATHEMATICAL_SDLOGJ_CROP2)
                if _metric_ok(returned_bundle, MATHEMATICAL_SDLOGJ_CROP2)
                else None
            )
            digital = candidate_bundle[DIGITAL_DECOMPOSITION]
            diagnostics = candidate_bundle[DETJ_DIAGNOSTICS]
            requested_utility = row["requested_state"]["utility"]
            operator = row["operator"]
            posterior_diagnostic = row["proposal"].get("posterior_diagnostics") or {}
            roughness = row["proposal"].get("postprocessed_residual_roughness") or {}
            flat = {
                "case_id": case_id,
                "arm_index": row["arm_index"],
                "arm_id": arm_id,
                "selectable": row["selectable"],
                "stress_only": row["stress_only"],
                "action": row["action"],
                "reason": row["reason"],
                "exact_status": row["exact"].get("status"),
                "exact_certified": row["exact_certified"],
                "returned_exact_certified": row["exact_certified"] if row["action"] == "ACCEPT" else True,
                "candidate_array_sha256": row["candidate_field"]["array_sha256"],
                "requested_array_sha256": row["requested_state"]["field"]["array_sha256"],
                "requested_exact_status": row["requested_state"]["exact"].get("status"),
                "requested_exact_certified": row["requested_state"]["exact_certified"],
                "returned_array_sha256": row["returned_field"]["array_sha256"],
                "mind_support_baseline_count": row["supports"]["mind"]["baseline_count"],
                "mind_support_pair_count": row["supports"]["mind"]["pair_count"],
                "ncc7_support_baseline_count": row["supports"]["ncc7"]["baseline_count"],
                "ncc7_support_pair_count": row["supports"]["ncc7"]["pair_count"],
                "ncc7_support_retention": row["supports"]["ncc7"]["retention"],
                "ncc9_support_baseline_count": row["supports"]["ncc9"]["baseline_count"],
                "ncc9_support_pair_count": row["supports"]["ncc9"]["pair_count"],
                "ncc9_support_retention": row["supports"]["ncc9"]["retention"],
                "mind_support_retention": row["supports"]["mind"]["retention"],
                **row["utility"],
                **{f"requested_{key}": value for key, value in requested_utility.items()},
                "requested_rms": row["proposal"].get("requested_rms"),
                "rms_reference": row["proposal"].get("rms_reference"),
                "rms_matched": row["proposal"].get("rms_matched"),
                "lambda_mean": row["proposal"].get("lambda_mean"),
                "confidence_mean": row["proposal"].get("confidence_mean"),
                "posterior_diagnostic_id": posterior_diagnostic.get("diagnostic_id"),
                "top1_top2_valid_logit_gap_mean": posterior_diagnostic.get("top1_top2_valid_logit_gap_mean"),
                "posterior_peak_probability_mean": posterior_diagnostic.get("posterior_peak_probability_mean"),
                "posterior_entropy_nats_mean": posterior_diagnostic.get("entropy_nats_mean"),
                "invalid_offset_fraction": posterior_diagnostic.get("invalid_offset_fraction"),
                "posterior_mean_l2_norm_mean": posterior_diagnostic.get("posterior_mean_l2_norm_mean"),
                "confidence_weighted_mean_l2_norm_mean": posterior_diagnostic.get(
                    "confidence_weighted_mean_l2_norm_mean"
                ),
                "confidence_to_mean_l2_norm_ratio": posterior_diagnostic.get("confidence_to_mean_l2_norm_ratio"),
                "roughness_metric_id": roughness.get("metric_id"),
                "requested_residual_rms_first_difference": roughness.get("rms_vector_first_difference"),
                "requested_state_residual_rms": row["requested_state"]["residual_rms_from_stored_state"],
                "postclip_residual_rms": row["postclip_residual_rms"],
                "returned_residual_rms": row["returned_residual_rms"],
                "requested_fast_cert_bound_work_eps": row["requested_state"]["fast_cert_bound_work_eps"],
                "postclip_fast_cert_bound_work_eps": operator.get("output_fast_cert_bound"),
                "clip_retained_norm_ratio": operator.get("retained_norm_ratio"),
                "clip_alpha_min": operator.get("effective_alpha_min"),
                "clip_alpha_p50": operator.get("effective_alpha_p50"),
                "clip_alpha_p95": operator.get("effective_alpha_p95"),
                "clip_alpha_max": operator.get("effective_alpha_max"),
                "baseline_primary_geometry": baseline_geometry,
                "requested_primary_geometry": requested_geometry,
                "candidate_primary_geometry": candidate_geometry,
                "returned_primary_geometry": returned_geometry,
                "requested_geometry_status": requested_bundle[MATHEMATICAL_SDLOGJ_CROP2]["status"],
                "candidate_geometry_status": candidate_bundle[MATHEMATICAL_SDLOGJ_CROP2]["status"],
                "returned_geometry_status": returned_bundle[MATHEMATICAL_SDLOGJ_CROP2]["status"],
                "metric_error_present": any(
                    metric.get("status") == "ERROR"
                    for bundle in (requested_bundle, candidate_bundle)
                    for metric in bundle.values()
                ),
                "digital_corner_violation_fraction": (digital.get("components") or {}).get(
                    "corner_union_violation_fraction"
                ),
                "digital_jstar_violation_fraction": (digital.get("components") or {}).get(
                    "jstar_union_violation_fraction"
                ),
                "detj_nonfinite_count": (diagnostics.get("components") or {}).get("nonfinite_count"),
                "detj_nonpositive_count": (diagnostics.get("components") or {}).get("nonpositive_count"),
                **{
                    key: evaluated[key]
                    for key in (
                        "baseline_dice",
                        "requested_diagnostic_dice",
                        "requested_diagnostic_dice_delta",
                        "capacity_candidate_dice",
                        "capacity_dice_delta",
                        "primary_returned_dice",
                        "primary_dice_delta",
                    )
                },
            }
            flat_rows.append(flat)
            rows_by_arm[arm_id].append(flat)
            metric_error_present |= bool(flat["metric_error_present"])

    summaries: list[dict[str, Any]] = []
    preliminary: dict[str, dict[str, Any]] = {}
    for spec in ARM_SPECS:
        rows = rows_by_arm[spec.arm_id]
        if len(rows) != EXPECTED_CASES:
            raise RuntimeError(f"C3a arm {spec.arm_id} does not cover 58 cases")
        baseline_dice = [float(row["baseline_dice"]) for row in rows]
        capacity_dice = [float(row["capacity_candidate_dice"]) for row in rows]
        primary_dice = [float(row["primary_returned_dice"]) for row in rows]
        capacity = paired_summary(capacity_dice, baseline_dice)
        primary = paired_summary(primary_dice, baseline_dice)
        all_candidate_geometry_defined = all(row["candidate_primary_geometry"] is not None for row in rows)
        candidate_geometry_deltas = [
            float(row["candidate_primary_geometry"]) - float(row["baseline_primary_geometry"])
            for row in rows
            if row["candidate_primary_geometry"] is not None and row["baseline_primary_geometry"] is not None
        ]
        all_returned_geometry_defined = all(row["returned_primary_geometry"] is not None for row in rows)
        returned_geometry_deltas = [
            float(row["returned_primary_geometry"]) - float(row["baseline_primary_geometry"])
            for row in rows
            if row["returned_primary_geometry"] is not None and row["baseline_primary_geometry"] is not None
        ]
        all_exact = all(row["exact_certified"] is True for row in rows)
        preliminary[spec.arm_id] = {
            "spec": spec,
            "rows": rows,
            "capacity": capacity,
            "primary": primary,
            "all_candidate_geometry_defined": all_candidate_geometry_defined
            and len(candidate_geometry_deltas) == EXPECTED_CASES,
            "candidate_geometry_delta_mean": (
                float(np.mean(candidate_geometry_deltas)) if len(candidate_geometry_deltas) == EXPECTED_CASES else None
            ),
            "all_returned_geometry_defined": all_returned_geometry_defined
            and len(returned_geometry_deltas) == EXPECTED_CASES,
            "returned_geometry_delta_mean": (
                float(np.mean(returned_geometry_deltas)) if len(returned_geometry_deltas) == EXPECTED_CASES else None
            ),
            "all_exact": all_exact,
            "all_requested_exact": all(row["requested_exact_certified"] is True for row in rows),
            "support_defined": all(
                all(
                    int(row[f"{key}_support_baseline_count"]) > 0
                    and 0 <= int(row[f"{key}_support_pair_count"]) <= int(row[f"{key}_support_baseline_count"])
                    and 0.0 <= float(row[f"{key}_support_retention"]) <= 1.0
                    for key in ("mind", "ncc7", "ncc9")
                )
                for row in rows
            ),
            "all_candidate_support_eligible": all(
                int(row["ncc7_support_pair_count"]) > 0
                and float(row["ncc7_support_retention"]) >= SUPPORT_RETENTION_MIN
                for row in rows
            ),
            "all_returned_exact": all(row["returned_exact_certified"] is True for row in rows),
        }
    c1 = preliminary["c1_raw_conf_post1"]
    c1_geometry_valid = c1["all_returned_geometry_defined"] and c1["returned_geometry_delta_mean"] is not None
    frozen_controls_exact = all(
        preliminary[arm_id]["all_exact"] for arm_id in ("c1_raw_conf_post1", "raw_conf_post1", "raw_conf_post2")
    )
    viable_map: dict[str, bool] = {}
    scores: dict[str, float] = {}
    for spec in ARM_SPECS:
        item = preliminary[spec.arm_id]
        geometry_ok = (
            c1_geometry_valid
            and item["returned_geometry_delta_mean"] is not None
            and geometry_noninferior(
                item["returned_geometry_delta_mean"],
                c1["returned_geometry_delta_mean"],
                all_candidate_metrics_defined=item["all_returned_geometry_defined"],
            )
        )
        viable = bool(
            spec.selectable
            and viable_primary_policy(
                item["primary"],
                all_returned_exact_certified=item["all_returned_exact"],
                all_support_diagnostics_defined=item["support_defined"],
                geometry_is_noninferior=geometry_ok,
            )
        )
        viable_map[spec.arm_id] = viable
        scores[spec.arm_id] = float(np.mean([row["primary_returned_dice"] for row in item["rows"]]))
        summary = {
            "arm_index": spec.arm_index,
            "arm_id": spec.arm_id,
            "role": spec.role,
            "selectable": spec.selectable,
            "stress_only": spec.stress_only,
            "cases": EXPECTED_CASES,
            "accepted": sum(row["action"] == "ACCEPT" for row in item["rows"]),
            "rolled_back": sum(row["action"] == "ROLLBACK" for row in item["rows"]),
            "capacity_vs_baseline": item["capacity"].to_dict(),
            "primary_policy_vs_baseline": item["primary"].to_dict(),
            "capacity_estimand": "accept-all exact post-clip candidate",
            "capacity_exact_eligible": item["all_exact"],
            "capacity_materially_strong": item["all_exact"] and materially_strong_capacity(item["capacity"]),
            "all_candidate_exact_certified": item["all_exact"],
            "all_requested_exact_certified": item["all_requested_exact"],
            "all_returned_exact_certified": item["all_returned_exact"],
            "all_support_diagnostics_defined": item["support_defined"],
            "all_candidates_pass_support_retention_gate": item["all_candidate_support_eligible"],
            "all_candidate_geometry_defined": item["all_candidate_geometry_defined"],
            "candidate_geometry_delta_mean": item["candidate_geometry_delta_mean"],
            "all_primary_returned_geometry_defined": item["all_returned_geometry_defined"],
            "primary_returned_geometry_delta_mean": item["returned_geometry_delta_mean"],
            "geometry_noninferior_to_c1_control": geometry_ok,
            "viable_primary_policy": viable,
            "capacity_dice_mean": float(np.mean([row["capacity_candidate_dice"] for row in item["rows"]])),
            "primary_returned_dice_mean": scores[spec.arm_id],
        }
        diagnostic_keys = (
            "top1_top2_valid_logit_gap_mean",
            "posterior_peak_probability_mean",
            "posterior_entropy_nats_mean",
            "invalid_offset_fraction",
            "posterior_mean_l2_norm_mean",
            "confidence_weighted_mean_l2_norm_mean",
            "confidence_to_mean_l2_norm_ratio",
            "requested_residual_rms_first_difference",
            "clip_retained_norm_ratio",
        )
        for key in diagnostic_keys:
            values = [float(row[key]) for row in item["rows"] if row.get(key) is not None]
            summary[f"{key}_mean"] = float(np.mean(values)) if values else None
            summary[f"{key}_defined_cases"] = len(values)
        summaries.append(summary)

    winner = select_winner(scores, viable_map)
    rms_raw = all(
        math.isclose(
            float(row["rms_matched"]),
            float(row["rms_reference"]),
            rel_tol=FLOAT32_PARITY_RTOL,
            abs_tol=FLOAT32_PARITY_ATOL,
        )
        for row in rows_by_arm["raw_mean_normmatched_post1"]
    )
    rms_adaptive_reference = all(
        math.isclose(
            float(row["rms_matched"]),
            float(row["rms_reference"]),
            rel_tol=FLOAT32_PARITY_RTOL,
            abs_tol=FLOAT32_PARITY_ATOL,
        )
        for row in rows_by_arm["adaptive_mean_adaptref_normmatched_post1"]
    )
    rms_raw_reference_adaptive = all(
        math.isclose(
            float(row["rms_matched"]),
            float(row["rms_reference"]),
            rel_tol=FLOAT32_PARITY_RTOL,
            abs_tol=FLOAT32_PARITY_ATOL,
        )
        for row in rows_by_arm["adaptive_mean_rawref_normmatched_post1"]
    )
    mean_decoder_common_reference = all(
        math.isclose(
            float(raw["rms_reference"]),
            float(adaptive["rms_reference"]),
            rel_tol=FLOAT32_PARITY_RTOL,
            abs_tol=FLOAT32_PARITY_ATOL,
        )
        for raw, adaptive in zip(
            rows_by_arm["raw_mean_normmatched_post1"],
            rows_by_arm["adaptive_mean_rawref_normmatched_post1"],
            strict=True,
        )
    )
    mp_mean_mass_parity = all(
        math.isclose(
            float(isotropic["lambda_mean"]),
            float(adaptive["lambda_mean"]),
            rel_tol=FLOAT32_PARITY_RTOL,
            abs_tol=FLOAT32_PARITY_ATOL,
        )
        for isotropic, adaptive in zip(
            rows_by_arm["iso_mp_conf_post1"],
            rows_by_arm["adaptive_mp_conf_post1"],
            strict=True,
        )
    )

    def residual_rms_parity(first: str, second: str, key: str) -> bool:
        return all(
            math.isclose(
                float(left[key]),
                float(right[key]),
                rel_tol=FLOAT32_PARITY_RTOL,
                abs_tol=FLOAT32_PARITY_ATOL,
            )
            for left, right in zip(rows_by_arm[first], rows_by_arm[second], strict=True)
        )

    def interaction(first: str, second: str, third: str, fourth: str, key: str) -> dict[str, Any]:
        differences = [
            (float(a[key]) - float(b[key])) - (float(c[key]) - float(d[key]))
            for a, b, c, d in zip(
                rows_by_arm[first],
                rows_by_arm[second],
                rows_by_arm[third],
                rows_by_arm[fourth],
                strict=True,
            )
        ]
        return paired_summary(differences).to_dict()

    contrasts = {
        "iso_mp_vs_raw_post1": _contrast(rows_by_arm, "iso_mp_conf_post1", "raw_conf_post1"),
        "iso_mp_vs_raw_post2": _contrast(rows_by_arm, "iso_mp_conf_post1", "raw_conf_post2"),
        "adaptive_mp_vs_iso_mp": _contrast(rows_by_arm, "adaptive_mp_conf_post1", "iso_mp_conf_post1"),
        "adaptive_mp_vs_raw_post2": _contrast(rows_by_arm, "adaptive_mp_conf_post1", "raw_conf_post2"),
        "raw_mean_vs_raw_conf": _contrast(rows_by_arm, "raw_mean_normmatched_post1", "raw_conf_post1"),
        "adaptive_adaptref_mean_vs_adaptive_conf": _contrast(
            rows_by_arm, "adaptive_mean_adaptref_normmatched_post1", "adaptive_mp_conf_post1"
        ),
        "adaptive_rawref_mean_vs_raw_mean": _contrast(
            rows_by_arm, "adaptive_mean_rawref_normmatched_post1", "raw_mean_normmatched_post1"
        ),
        "adaptive_raw_mean_vs_adaptref_mean": _contrast(
            rows_by_arm, "adaptive_mean_raw_post1", "adaptive_mean_adaptref_normmatched_post1"
        ),
        "adaptive_raw_mean_vs_adaptive_conf_total_effect": _contrast(
            rows_by_arm, "adaptive_mean_raw_post1", "adaptive_mp_conf_post1"
        ),
    }
    hypotheses = {
        "schema": f"ctcf-search-c3a-hypotheses-{SCHEMA_VERSION}",
        "analysis_scope": "single-scale MIND-SSC radius-1, frozen linear entropy MP, IXI val-58 development",
        "contrasts": contrasts,
        "H_COST_MP_HELPS": {
            "preclip_requested_signal": contrasts["iso_mp_vs_raw_post1"]["requested_preclip_descriptive_wins"],
            "preclip_requested_all_exact": contrasts["iso_mp_vs_raw_post1"]["requested_preclip_exact_eligible"],
            "effect_survives_exact_postclip_capacity": contrasts["iso_mp_vs_raw_post1"]["capacity_wins"],
            "practically_outperforms_raw_post2_on_capacity": contrasts["iso_mp_vs_raw_post2"]["capacity_wins"],
        },
        "H_ADAPTIVITY_HELPS": {
            "mean_message_mass_parity_verified": mp_mean_mass_parity,
            "preclip_requested_signal": mp_mean_mass_parity
            and contrasts["adaptive_mp_vs_iso_mp"]["requested_preclip_descriptive_wins"],
            "preclip_requested_all_exact": contrasts["adaptive_mp_vs_iso_mp"]["requested_preclip_exact_eligible"],
            "effect_survives_exact_postclip_capacity": mp_mean_mass_parity
            and contrasts["adaptive_mp_vs_iso_mp"]["capacity_wins"],
            "practically_outperforms_raw_post2_on_capacity": contrasts["adaptive_mp_vs_raw_post2"]["capacity_wins"],
        },
        "H_CONFIDENCE_IS_HARMFUL": {
            "without_mp": {
                "requested_rms_parity_verified": rms_raw,
                "postclip_rms_parity_verified": residual_rms_parity(
                    "raw_mean_normmatched_post1", "raw_conf_post1", "postclip_residual_rms"
                ),
                "preclip_requested_signal": rms_raw
                and contrasts["raw_mean_vs_raw_conf"]["requested_preclip_descriptive_wins"],
                "preclip_requested_all_exact": contrasts["raw_mean_vs_raw_conf"]["requested_preclip_exact_eligible"],
                "effect_survives_exact_postclip_capacity": rms_raw
                and contrasts["raw_mean_vs_raw_conf"]["capacity_wins"],
            },
            "under_adaptive_mp": {
                "requested_rms_parity_verified": rms_adaptive_reference,
                "postclip_rms_parity_verified": residual_rms_parity(
                    "adaptive_mean_adaptref_normmatched_post1",
                    "adaptive_mp_conf_post1",
                    "postclip_residual_rms",
                ),
                "preclip_requested_signal": rms_adaptive_reference
                and contrasts["adaptive_adaptref_mean_vs_adaptive_conf"]["requested_preclip_descriptive_wins"],
                "preclip_requested_all_exact": contrasts["adaptive_adaptref_mean_vs_adaptive_conf"][
                    "requested_preclip_exact_eligible"
                ],
                "effect_survives_exact_postclip_capacity": rms_adaptive_reference
                and contrasts["adaptive_adaptref_mean_vs_adaptive_conf"]["capacity_wins"],
            },
            "decoder_by_mp_interaction_descriptive": {
                "requested_preclip": interaction(
                    "adaptive_mean_adaptref_normmatched_post1",
                    "adaptive_mp_conf_post1",
                    "raw_mean_normmatched_post1",
                    "raw_conf_post1",
                    "requested_diagnostic_dice",
                ),
                "postclip_capacity": interaction(
                    "adaptive_mean_adaptref_normmatched_post1",
                    "adaptive_mp_conf_post1",
                    "raw_mean_normmatched_post1",
                    "raw_conf_post1",
                    "capacity_candidate_dice",
                ),
                "all_four_capacity_exact_eligible": all(
                    row["exact_certified"] is True
                    for arm_id in (
                        "adaptive_mean_adaptref_normmatched_post1",
                        "adaptive_mp_conf_post1",
                        "raw_mean_normmatched_post1",
                        "raw_conf_post1",
                    )
                    for row in rows_by_arm[arm_id]
                ),
            },
        },
        "H_MP_HELPS_MEAN_DECODER": {
            "rms_parity_verified": rms_raw and rms_raw_reference_adaptive and mean_decoder_common_reference,
            "postclip_rms_parity_verified": residual_rms_parity(
                "adaptive_mean_rawref_normmatched_post1",
                "raw_mean_normmatched_post1",
                "postclip_residual_rms",
            ),
            "preclip_requested_signal": rms_raw
            and rms_raw_reference_adaptive
            and mean_decoder_common_reference
            and contrasts["adaptive_rawref_mean_vs_raw_mean"]["requested_preclip_descriptive_wins"],
            "preclip_requested_all_exact": contrasts["adaptive_rawref_mean_vs_raw_mean"][
                "requested_preclip_exact_eligible"
            ],
            "effect_survives_exact_postclip_capacity": rms_raw
            and rms_raw_reference_adaptive
            and mean_decoder_common_reference
            and contrasts["adaptive_rawref_mean_vs_raw_mean"]["capacity_wins"],
        },
        "H_AMPLITUDE_OR_CLIP_IS_BOTTLENECK": {
            "estimand": "adaptive raw posterior mean minus its adaptive-reference RMS-matched counterpart",
            "preclip_requested_signal": contrasts["adaptive_raw_mean_vs_adaptref_mean"][
                "requested_preclip_descriptive_wins"
            ],
            "preclip_requested_all_exact": contrasts["adaptive_raw_mean_vs_adaptref_mean"][
                "requested_preclip_exact_eligible"
            ],
            "effect_survives_exact_postclip_capacity": contrasts["adaptive_raw_mean_vs_adaptref_mean"]["capacity_wins"],
            "total_effect_vs_confidence": {
                "preclip_requested_signal": contrasts["adaptive_raw_mean_vs_adaptive_conf_total_effect"][
                    "requested_preclip_descriptive_wins"
                ],
                "preclip_requested_all_exact": contrasts["adaptive_raw_mean_vs_adaptive_conf_total_effect"][
                    "requested_preclip_exact_eligible"
                ],
                "effect_survives_exact_postclip_capacity": contrasts["adaptive_raw_mean_vs_adaptive_conf_total_effect"][
                    "capacity_wins"
                ],
            },
        },
        "stress_arm_is_never_selectable": True,
        "no_material_plus_0_002_capacity_closes_only_this_configuration": not any(
            row["selectable"] and row["capacity_materially_strong"] for row in summaries
        ),
        "test_115_authorized": False,
    }
    fatal = not baseline_geometry_defined or not c1_geometry_valid or not frozen_controls_exact or metric_error_present
    summary = {
        "schema": f"ctcf-search-c3a-summary-{SCHEMA_VERSION}",
        "protocol_id": PROTOCOL_ID,
        "execution_integrity_status": "FAIL" if fatal else "PASS",
        "scientific_status": (
            "C3A_METRIC_EXECUTION_ERROR"
            if metric_error_present
            else "C3A_FROZEN_CONTROL_EXACT_PARITY_FAILURE"
            if not frozen_controls_exact
            else "C3A_INVALID_BASELINE_OR_CONTROL_GEOMETRY"
            if not baseline_geometry_defined or not c1_geometry_valid
            else "C3A_VIABLE_WINNER"
            if winner is not None
            else "C3A_NO_VIABLE_ARM"
        ),
        "n_cases": EXPECTED_CASES,
        "n_arms": len(ARM_SPECS),
        "n_case_arm_rows": EXPECTED_CASES * len(ARM_SPECS),
        "winner_arm_id": winner,
        "selectable_arm_ids": list(SELECTABLE_ARM_IDS),
        "viable_arm_ids": [arm_id for arm_id in SELECTABLE_ARM_IDS if viable_map[arm_id]],
        "baseline_and_c1_primary_geometry_defined": baseline_geometry_defined and c1_geometry_valid,
        "frozen_controls_exact_certified": frozen_controls_exact,
        "frozen_c2_control_dice_parity_verified": True,
        "metric_execution_error_present": metric_error_present,
        "test_split_accessed": False,
        "test_115_authorized": False,
        "labels_used_for_decision": False,
        "decision_workers_received_label_inputs": False,
        "image_extractor_deserialized_segmentation_objects": True,
        "dice_evaluation_started_after_barrier": True,
    }
    return flat_rows, summaries, hypotheses, summary, fatal


def _flat_arm_summary_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for row in rows:
        flat = {key: value for key, value in row.items() if not isinstance(value, dict)}
        for prefix in ("capacity_vs_baseline", "primary_policy_vs_baseline"):
            for key, value in row[prefix].items():
                flat[f"{prefix}_{key}"] = value
        output.append(flat)
    return output


def _csv_fields(rows: list[dict[str, Any]], preferred: list[str]) -> list[str]:
    keys = {key for row in rows for key in row}
    return [key for key in preferred if key in keys] + sorted(keys - set(preferred))


def finalize_stage(args: argparse.Namespace) -> int:
    root = args.run_root.resolve()
    source, source_sha = _load_source_contract(root, args.source_contract_sha256)
    decision, decision_sha = _load_decision_contract(root, args.decision_contract_sha256)
    if decision.get("source_contract_sha256") != source_sha:
        raise RuntimeError("Decision and source contracts are not linked")
    _assert_clean_code(decision["git_head"], "finalize")
    _assert_runtime_signature(decision["runtime_signature"], "finalize")
    barrier, barrier_sha = _load_barrier(root, args.barrier_sha256, decision, decision_sha)
    c2_evaluation_goldens = _load_c2_evaluation_goldens(root, source)
    c2_source = source["c2_source"]
    _validate_c2_source(Path(c2_source["directory"]), c2_source["manifest_sha256"])
    for row in source["raw_inputs"].values():
        _verify_file_record(row)

    evaluation_workers: list[dict[str, Any]] = []
    seen: list[str] = []
    for shard in range(decision["num_shards"]):
        path, _ = _worker_paths(root, "evaluation", args.attempt_id, shard)
        payload = _load_json(path)
        _validate_worker_report(
            payload,
            phase="evaluation",
            attempt_id=args.attempt_id,
            shard=shard,
            contract=decision,
            contract_sha=decision_sha,
        )
        if payload.get("decision_barrier_sha256") != barrier_sha:
            raise RuntimeError(f"Evaluation worker is linked to another barrier: {path}")
        seen.extend(decision["shards"][str(shard)])
        evaluation_workers.append({"path": path.relative_to(root).as_posix(), "sha256": sha256_file(path)})
    validate_shard_partition(
        decision,
        seen,
        EXPECTED_CASES,
        "C3a evaluation worker partition has missing, duplicate, or reordered cases",
    )

    decisions: dict[str, dict[str, Any]] = {}
    evaluations: dict[str, dict[str, Any]] = {}
    decision_hashes: dict[str, str] = {}
    evaluation_hashes: dict[str, str] = {}
    for case_id in decision["case_ids"]:
        decision_path = _decision_case_path(root, case_id)
        evaluation_path = _evaluation_case_path(root, case_id)
        decision_payload = _load_json(decision_path)
        evaluation_payload = _load_json(evaluation_path)
        _validate_decision_case(decision_payload, decision_path, case_id, decision, decision_sha, verify_heavy=True)
        _validate_evaluation_case(
            evaluation_payload,
            evaluation_path,
            case_id,
            decision,
            decision_sha,
            barrier,
            barrier_sha,
            c2_evaluation_goldens,
        )
        decisions[case_id] = decision_payload
        evaluations[case_id] = evaluation_payload
        decision_hashes[case_id] = sha256_file(decision_path)
        evaluation_hashes[case_id] = sha256_file(evaluation_path)

    flat_rows, summary_rows, hypotheses, summary, fatal = _build_summaries(decisions, evaluations)
    flat_summary = _flat_arm_summary_rows(summary_rows)
    per_arm_path = root / "per_arm.csv"
    arm_summary_path = root / "arm_summary.csv"
    hypotheses_path = root / "hypotheses.json"
    summary_path = root / "summary.json"
    atomic_write_text(
        per_arm_path,
        rows_to_csv(_csv_fields(flat_rows, ["case_id", "arm_index", "arm_id", "action", "reason"]), flat_rows),
    )
    atomic_write_text(
        arm_summary_path,
        rows_to_csv(_csv_fields(flat_summary, ["arm_index", "arm_id", "role", "selectable"]), flat_summary),
    )
    atomic_write_json(hypotheses_path, hypotheses)
    atomic_write_json(summary_path, summary)
    prepare = _load_json(root / "prepare.json")
    manifest = {
        "schema": f"ctcf-search-c3a-run-manifest-{SCHEMA_VERSION}",
        "protocol_id": PROTOCOL_ID,
        "run_id": root.name,
        "status": "FAILED_SCIENTIFIC_INTEGRITY" if fatal else "COMPLETE",
        "started_at_utc": prepare["prepared_at_utc"],
        "completed_at_utc": utc_now(),
        "source_contract_sha256": source_sha,
        "decision_contract_sha256": decision_sha,
        "decision_barrier_sha256": barrier_sha,
        "policy_sha256": C3A_POLICY_SHA256,
        "finalize_attempt_id": args.attempt_id,
        "code": {
            "git_head": git("rev-parse", "HEAD"),
            "branch": git("branch", "--show-current"),
            "git_status": "",
        },
        "checkpoint": {
            "path": decision["checkpoint"],
            "sha256": decision["checkpoint_sha256"],
            "strict": True,
        },
        "c2_source": source["c2_source"],
        "execution": {
            "num_shards": decision["num_shards"],
            "physical_gpus": decision["physical_gpus"],
            "seed": decision["seed"],
            "paths_profile": source["paths_profile"],
            "time_steps": decision["time_steps"],
            "label_isolation": "image extraction -> decision barrier -> label evaluation",
        },
        "evaluation_workers": evaluation_workers,
        "decision_case_sha256": decision_hashes,
        "evaluation_case_sha256": evaluation_hashes,
        "files": {
            "source_contract_sha256": sha256_file(root / SOURCE_CONTRACT_NAME),
            "decision_contract_sha256": sha256_file(root / DECISION_CONTRACT_NAME),
            "decision_barrier_sha256": sha256_file(root / DECISION_BARRIER_NAME),
            "datasets_sha256": sha256_file(root / "datasets.csv"),
            "c2_decision_goldens_sha256": sha256_file(root / "c2_decision_goldens.json"),
            "c2_evaluation_goldens_sha256": sha256_file(root / "c2_evaluation_goldens.json"),
            "per_arm_sha256": sha256_file(per_arm_path),
            "arm_summary_sha256": sha256_file(arm_summary_path),
            "hypotheses_sha256": sha256_file(hypotheses_path),
            "summary_sha256": sha256_file(summary_path),
        },
        "summary": summary,
        "storage": {
            "compact_package_excludes_heavy_fields": True,
            "heavy_root": decision["heavy_root"],
            "heavy_fields_retained_for_review": True,
            "automatic_deletion": False,
        },
    }
    atomic_write_json(root / "c3_manifest.json", manifest)
    print(json.dumps(summary, indent=2))
    if fatal:
        raise RuntimeError("C3a scientific-integrity checks failed; inspect summary.json")
    return 0


def selfcheck_stage(args: argparse.Namespace) -> int:
    fixed = dict(C3A_POLICY.to_dict()["fixed_parameters"])
    checks = {
        "ten_unique_ordered_arms": [spec.arm_index for spec in ARM_SPECS] == list(range(10))
        and len({spec.arm_id for spec in ARM_SPECS}) == 10,
        "five_selectable_arms": tuple(spec.arm_id for spec in ARM_SPECS if spec.selectable) == SELECTABLE_ARM_IDS,
        "single_step_work_margin_is_frozen": WORK_EPS == 0.0011 and EXACT_CLAIM_EPS == 0.001,
        "candidate_lattice_matches_policy": len(OFFSETS) == CANDIDATE_COUNT
        and set(OFFSETS)
        == {
            (z, y, x)
            for z in range(-CANDIDATE_RADIUS, CANDIDATE_RADIUS + 1)
            for y in range(-CANDIDATE_RADIUS, CANDIDATE_RADIUS + 1)
            for x in range(-CANDIDATE_RADIUS, CANDIDATE_RADIUS + 1)
        },
        "cost_standardization_matches_policy": STANDARDIZATION_FLOOR == COST_STANDARDIZATION_FLOOR,
        "message_kernel_matches_policy": LOGIT_MESSAGE_AXIS_KERNEL == MESSAGE_PASSING_AXIS_KERNEL,
        "posterior_temperature_is_positive": POSTERIOR_TEMPERATURE > 0.0,
        "decision_worker_has_no_label_or_raw_container_inputs": fixed["decision_worker_has_label_inputs"] is False
        and fixed["decision_worker_has_raw_container_inputs"] is False,
        "os_level_isolation_is_not_overclaimed": fixed["os_level_data_isolation_claimed"] is False,
        "test_115_is_unreachable": fixed["test115_accessible"] is False,
        "metric_ids_are_explicit": "sdlogj" not in {metric_id.lower() for metric_id in METRIC_SPECS},
        "primary_geometry_is_mathematical_crop2": MATHEMATICAL_SDLOGJ_CROP2 in METRIC_SPECS,
        "policy_hash_is_canonical": payload_sha256(C3A_POLICY.to_dict()) == C3A_POLICY_SHA256,
    }
    failed = [key for key, value in checks.items() if not value]
    payload = {
        "schema": f"ctcf-search-c3a-selfcheck-{SCHEMA_VERSION}",
        "protocol_id": PROTOCOL_ID,
        "status": "PASS" if not failed else "FAIL",
        "checks": checks,
        "failed": failed,
        "policy_sha256": C3A_POLICY_SHA256,
    }
    atomic_write_json(args.output, payload)
    if failed:
        raise RuntimeError(f"C3a self-check failed: {failed}")
    print(json.dumps(payload, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run frozen label-isolated search Gate C3a on IXI val-58.")
    sub = parser.add_subparsers(dest="action", required=True)
    selfcheck = sub.add_parser("selfcheck")
    selfcheck.add_argument("--output", type=Path, required=True)
    prepare = sub.add_parser("prepare")
    prepare.add_argument("--run-root", type=Path, required=True)
    prepare.add_argument("--heavy-root", type=Path, required=True)
    prepare.add_argument("--paths-profile", type=int, default=3)
    prepare.add_argument("--checkpoint", type=Path, default=Path(DEFAULT_CHECKPOINT))
    prepare.add_argument("--seed", type=int, default=0)
    prepare.add_argument("--num-shards", type=int, required=True)
    prepare.add_argument("--physical-gpus", required=True)
    prepare.add_argument("--c2-dir", type=Path, required=True)
    prepare.add_argument("--c2-manifest-sha256", required=True)
    prepare.add_argument("--min-free-gib", type=int, default=100)
    extract = sub.add_parser("extract-images")
    extract.add_argument("--run-root", type=Path, required=True)
    extract.add_argument("--source-contract-sha256", required=True)
    decision = sub.add_parser("decision-worker")
    decision.add_argument("--run-root", type=Path, required=True)
    decision.add_argument("--decision-contract-sha256", required=True)
    decision.add_argument("--shard-index", type=int, required=True)
    decision.add_argument("--num-shards", type=int, required=True)
    decision.add_argument("--gpu", type=int, default=0)
    decision.add_argument("--physical-gpu", required=True)
    decision.add_argument("--attempt-id", required=True)
    barrier = sub.add_parser("decision-barrier")
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
    actions = {
        "selfcheck": selfcheck_stage,
        "prepare": prepare_stage,
        "extract-images": extract_images_stage,
        "decision-worker": decision_worker_stage,
        "decision-barrier": decision_barrier_stage,
        "evaluation-worker": evaluation_worker_stage,
        "finalize": finalize_stage,
    }
    return actions[args.action](args)


if __name__ == "__main__":
    raise SystemExit(main())
