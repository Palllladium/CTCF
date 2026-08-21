from __future__ import annotations

import argparse
import csv
import glob
import hashlib
import json
import math
import os
import platform
import re
import shutil
import subprocess
import tempfile
import time
from contextlib import suppress
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

from experiments.core.inference_metrics import metric_profile_for
from experiments.core.inference_runtime import build_infer_dataset
from tools.analysis.run_artifacts import (
    atomic_write_json,
    atomic_write_text,
    rows_to_csv,
    rows_to_tsv,
    sha256_file,
)
from tools.analysis.run_search_gate_c0 import (
    IXI_DEVELOPMENT_CASES,
    _build_model,
    _case_id,
    _dice,
    _prepare_initial_state,
    _proposal_statistics,
    _salted_case_hash,
)
from tools.analysis.transactional_search import (
    OFFSETS,
    ZERO_OFFSET_INDEX,
    ProposalResult,
    build_proposal,
    certified_local_clip_candidate,
    field_change_statistics,
    geometry_mask,
    load_flow_npz,
    masked_zscore,
    mind_distance_from_features,
    mind_ssc,
    ncc_loss_from_normalized,
    proposal_support_weights,
    save_flow_npz_atomic,
)
from utils import setup_device
from utils.cert_exact import certify_flow_exact
from utils.field import (
    boundary_vertex_mask,
    digital_fold_percent,
    jacobian_nonpositive_percent,
    logdet_std_from_flow,
    trilinear_cert_bound,
    trilinear_project,
)

PROTOCOL_ID = "CTCF-SEARCH-GATE-C1-V1"
SPLIT_PROTOCOL_ID = "CTCF-GATE-C0-V1-SALTED-IXI-VAL-58"
CLAIM_EPS = 0.001
WORK_EPS = 0.0011
COLLAR_WIDTH = 4
TIME_STEPS = 6
UTILITY_RELATIVE_TOLERANCE = 1e-6
GLOBAL_COEFFICIENTS = tuple(2.0**-index for index in range(17))
UTILITY_RULES = ("topology_only", "mind", "ncc9", "support_ncc9", "mind_and_ncc9")
DEFAULT_CHECKPOINT = "results/P10_LONGRUN_VXM_UNIFIED_SVF_IXI/ckpt/best.pth"
CONFIG_KEY = "CTCF-CascadeA-VM-Unified"
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")

LOCAL_OPERATOR_SPECS: tuple[dict[str, Any], ...] = (
    {
        "candidate_id": "mind_clip_s0p5_w1",
        "feature": "mind",
        "orientation": "target_centered",
        "operator": "certified_local_clip",
        "scale": 0.5,
        "sweeps": 1,
    },
    {
        "candidate_id": "mind_clip_s1_w1",
        "feature": "mind",
        "orientation": "target_centered",
        "operator": "certified_local_clip",
        "scale": 1.0,
        "sweeps": 1,
    },
    {
        "candidate_id": "mind_clip_s2_w1",
        "feature": "mind",
        "orientation": "target_centered",
        "operator": "certified_local_clip",
        "scale": 2.0,
        "sweeps": 1,
    },
    {
        "candidate_id": "mind_clip_s1_w2",
        "feature": "mind",
        "orientation": "target_centered",
        "operator": "certified_local_clip",
        "scale": 1.0,
        "sweeps": 2,
    },
    {
        "candidate_id": "mind_project_s1",
        "feature": "mind",
        "orientation": "target_centered",
        "operator": "trilinear_project",
        "scale": 1.0,
        "sweeps": None,
    },
    {
        "candidate_id": "intensity_clip_s1_w1",
        "feature": "intensity",
        "orientation": "target_centered",
        "operator": "certified_local_clip",
        "scale": 1.0,
        "sweeps": 1,
    },
    {
        "candidate_id": "mind_reversed_clip_s1_w1",
        "feature": "mind",
        "orientation": "reversed",
        "operator": "certified_local_clip",
        "scale": 1.0,
        "sweeps": 1,
    },
)
CONFIRMATION_SPEC = next(spec for spec in LOCAL_OPERATOR_SPECS if spec["candidate_id"] == "mind_clip_s1_w1")
CONFIRMATION_POLICY: dict[str, Any] = {
    "candidate_id": "mind_clip_s1_w1",
    "proposal": {
        "decoder": "soft",
        "feature": "mind",
        "orientation": "target_centered",
        "scale": 1.0,
    },
    "operator": {
        "name": "certified_local_clip",
        "work_eps": WORK_EPS,
        "sweeps": 1,
        "fixed_boundary": True,
    },
    "materialization": {"dtype": "float32", "save_reload": True},
    "topology": {"exact_eps": CLAIM_EPS, "checker": "utils.cert_exact.certify_flow_exact"},
    "acceptance": {
        "metric": "mind",
        "direction": "lower_is_better",
        "relative_tolerance": UTILITY_RELATIVE_TOLERANCE,
        "rollback": "byte_identical_initial_npz",
    },
    "labels": {"transaction_decision": False, "post_decision_evaluation": True},
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _git(*args: str) -> str:
    return subprocess.check_output(["git", *args], text=True, encoding="utf-8").strip()


def _text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _payload_sha256(payload: Any) -> str:
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return _text_sha256(encoded)


CONFIRMATION_POLICY_SHA256 = _payload_sha256(CONFIRMATION_POLICY)


def _is_finite_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float, np.integer, np.floating))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _require_finite(values: dict[str, Any], label: str) -> None:
    invalid = sorted(key for key, value in values.items() if not _is_finite_number(value))
    if invalid:
        raise RuntimeError(f"{label} contains non-finite or non-numeric values: {invalid}")


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _deformation_quality_metrics(field: torch.Tensor, *, exact_certified: bool) -> dict[str, Any]:
    """Report paper-facing geometry without conflating a sampled count with the exact certificate."""
    metrics: dict[str, Any] = {
        "sdlogj": float(logdet_std_from_flow(field)),
        "j_leq0_central_percent": float(jacobian_nonpositive_percent(field, crop=1)),
        "j_leq0_digital10_percent": float(digital_fold_percent(field).item()),
        "trilinear_fold_percent_upper_bound": 0.0 if exact_certified else None,
        "trilinear_fold_status": "ZERO_BY_EXACT_CERTIFICATE" if exact_certified else "NOT_ESTABLISHED",
    }
    _require_finite(
        {
            key: value
            for key, value in metrics.items()
            if key not in {"trilinear_fold_percent_upper_bound", "trilinear_fold_status"}
        },
        "deformation quality metrics",
    )
    if any(float(metrics[key]) < 0.0 for key in ("sdlogj", "j_leq0_central_percent", "j_leq0_digital10_percent")):
        raise RuntimeError("Deformation quality metrics must be non-negative")
    return metrics


def _select_cases(stage: str, paths_profile: int) -> tuple[list[str], str, list[str]]:
    # Import locally so preparing C1 never inspects the IXI test split by accident.
    from experiments.core.path_profiles import get_dataset_paths

    paths = get_dataset_paths(paths_profile, "IXI")
    root = paths["val_dir"]
    files = sorted(glob.glob(os.path.join(root, "*.pkl")))
    if not files:
        raise FileNotFoundError(f"No IXI/val .pkl files found under {root}")
    by_id = {_case_id(path): path for path in files}
    if len(files) != len(by_id):
        raise RuntimeError("IXI validation contains duplicate case identifiers")
    if len(by_id) != 58:
        raise RuntimeError(f"C1 requires exactly 58 IXI validation cases, found {len(by_id)}")

    ranked_ids = [case_id for _, case_id in sorted((_salted_case_hash(case_id), case_id) for case_id in by_id)]
    if tuple(ranked_ids[:19]) != IXI_DEVELOPMENT_CASES:
        raise RuntimeError("IXI validation split contract changed: the first 19 salted cases no longer match Gate C0")
    selected_ids = ranked_ids[:19] if stage == "exploration" else ranked_ids[19:]
    expected = 19 if stage == "exploration" else 39
    if len(selected_ids) != expected:
        raise RuntimeError(f"{stage} requires {expected} cases, selected {len(selected_ids)}")
    return (
        [by_id[case_id] for case_id in selected_ids],
        str(paths["atlas_path"]),
        [by_id[case_id] for case_id in ranked_ids],
    )


def _validate_exploration_manifest(path_value: str, expected_sha256: str) -> dict[str, Any]:
    expected = expected_sha256.lower()
    if not SHA256_RE.fullmatch(expected):
        raise ValueError("--explore-manifest-sha256 must contain exactly 64 lowercase hexadecimal characters")
    path = Path(path_value).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Exploration manifest not found: {path}")
    actual = sha256_file(path)
    if actual != expected:
        raise RuntimeError(f"Exploration manifest SHA-256 mismatch: expected={expected}, actual={actual}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if (
        payload.get("schema") != "ctcf-search-c1-stage-manifest-v1"
        or payload.get("protocol_id") != PROTOCOL_ID
        or payload.get("stage") != "exploration"
        or payload.get("status") != "COMPLETE"
        or (payload.get("summary") or {}).get("execution_integrity_status") != "PASS"
        or (payload.get("summary") or {}).get("scientific_status") != "EXPLORATORY_COMPLETE"
        or (payload.get("summary") or {}).get("n_cases") != 19
        or (payload.get("code") or {}).get("git_status")
        or payload.get("confirmation_policy_sha256") != CONFIRMATION_POLICY_SHA256
        or payload.get("confirmation_policy") != CONFIRMATION_POLICY
    ):
        raise RuntimeError("Confirmation requires a COMPLETE C1 exploration stage manifest")
    file_names = {
        "datasets_sha256": "datasets.csv",
        "validation_universe_sha256": "validation_universe.csv",
        "per_candidate_sha256": "per_candidate.csv",
        "per_case_sha256": "per_case.csv",
        "operator_rule_summary_sha256": "operator_rule_summary.csv",
        "summary_sha256": "summary.json",
    }
    for key, name in file_names.items():
        artifact = path.parent / name
        recorded = (payload.get("files") or {}).get(key)
        if not artifact.is_file() or not recorded or sha256_file(artifact) != recorded:
            raise RuntimeError(f"Exploration artifact is missing or differs from its manifest: {artifact}")
    contract_path = path.parent / "stage_contract.json"
    if not contract_path.is_file() or sha256_file(contract_path) != payload.get("contract_sha256"):
        raise RuntimeError("Exploration stage contract is missing or differs from its manifest")
    exploration_contract = json.loads(contract_path.read_text(encoding="utf-8"))
    if (
        exploration_contract.get("schema") != "ctcf-search-c1-stage-contract-v1"
        or exploration_contract.get("protocol_id") != PROTOCOL_ID
        or exploration_contract.get("stage") != "exploration"
        or len(exploration_contract.get("case_ids") or []) != 19
        or exploration_contract.get("confirmation_policy_sha256") != CONFIRMATION_POLICY_SHA256
        or exploration_contract.get("confirmation_policy") != CONFIRMATION_POLICY
        or (payload.get("files") or {}).get("stage_contract_sha256") != payload.get("contract_sha256")
        or (payload.get("files") or {}).get("datasets_sha256") != exploration_contract.get("datasets_csv_sha256")
        or (payload.get("files") or {}).get("validation_universe_sha256")
        != exploration_contract.get("validation_universe_sha256")
        or (payload.get("code") or {}).get("git_head") != exploration_contract.get("git_head")
        or (payload.get("checkpoint") or {}).get("sha256") != exploration_contract.get("checkpoint_sha256")
        or (payload.get("checkpoint") or {}).get("config") != exploration_contract.get("config")
        or (payload.get("execution") or {}).get("seed") != exploration_contract.get("seed")
        or (payload.get("execution") or {}).get("paths_profile") != exploration_contract.get("paths_profile")
        or (payload.get("execution") or {}).get("time_steps") != exploration_contract.get("time_steps")
    ):
        raise RuntimeError("Exploration manifest and stage contract do not form one frozen C1 protocol")
    summary_path = path.parent / "summary.json"
    if json.loads(summary_path.read_text(encoding="utf-8")) != payload.get("summary"):
        raise RuntimeError("Exploration manifest summary differs from summary.json")
    return {
        "path": str(path),
        "sha256": actual,
        "run_id": payload.get("run_id"),
        "git_head": (payload.get("code") or {}).get("git_head"),
        "contract_sha256": payload.get("contract_sha256"),
        "summary_sha256": (payload.get("files") or {}).get("summary_sha256"),
        "scientific_status": (payload.get("summary") or {}).get("scientific_status"),
        "checkpoint_sha256": (payload.get("checkpoint") or {}).get("sha256"),
        "checkpoint_config": (payload.get("checkpoint") or {}).get("config"),
        "seed": (payload.get("execution") or {}).get("seed"),
        "paths_profile": (payload.get("execution") or {}).get("paths_profile"),
        "time_steps": (payload.get("execution") or {}).get("time_steps"),
        "validation_universe_sha256": (payload.get("files") or {}).get("validation_universe_sha256"),
        "confirmation_policy_sha256": payload.get("confirmation_policy_sha256"),
    }


def _dataset_rows(files: list[str], atlas: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path_value in [*files, atlas]:
        path = Path(path_value).resolve()
        stat = path.stat()
        rows.append(
            {
                "dataset": "IXI",
                "split": "atlas" if path_value == atlas else "val",
                "case_id": "atlas" if path_value == atlas else _case_id(path_value),
                "path": str(path),
                "bytes": stat.st_size,
                "sha256": sha256_file(path),
                "mtime_utc": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat().replace("+00:00", "Z"),
            }
        )
    return rows


def _stage_dir(run_root: Path, stage: str) -> Path:
    return run_root.resolve() / stage


def _verify_contract(stage_dir: Path, expected_sha256: str) -> tuple[dict[str, Any], str]:
    contract_path = stage_dir / "stage_contract.json"
    if not contract_path.is_file():
        raise FileNotFoundError(f"Missing C1 stage contract: {contract_path}")
    actual = sha256_file(contract_path)
    if actual != expected_sha256.lower():
        raise RuntimeError(f"Stage contract SHA-256 mismatch: expected={expected_sha256}, actual={actual}")
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    if contract.get("schema") != "ctcf-search-c1-stage-contract-v1":
        raise RuntimeError("Unsupported C1 stage contract schema")
    return contract, actual


def _read_dataset_rows(stage_dir: Path, contract: dict[str, Any]) -> list[dict[str, str]]:
    path = stage_dir / "datasets.csv"
    if not path.is_file() or sha256_file(path) != contract["datasets_csv_sha256"]:
        raise RuntimeError("datasets.csv is missing or differs from the stage contract")
    tsv_path = stage_dir / "datasets.tsv"
    if not tsv_path.is_file() or sha256_file(tsv_path) != contract["datasets_tsv_sha256"]:
        raise RuntimeError("datasets.tsv is missing or differs from the stage contract")
    with path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    expected = len(contract["case_ids"]) + 1
    if len(rows) != expected or sum(row["split"] == "atlas" for row in rows) != 1:
        raise RuntimeError(f"Dataset manifest must contain {expected} rows including exactly one atlas")
    if [row["case_id"] for row in rows if row["split"] == "val"] != contract["case_ids"]:
        raise RuntimeError("Dataset case order differs from the stage contract")
    return rows


def _read_validation_universe(stage_dir: Path, contract: dict[str, Any]) -> list[dict[str, str]]:
    path = stage_dir / "validation_universe.csv"
    if not path.is_file() or sha256_file(path) != contract["validation_universe_sha256"]:
        raise RuntimeError("validation_universe.csv is missing or differs from the frozen 58-case contract")
    with path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if (
        len(rows) != 59
        or sum(row["split"] == "atlas" for row in rows) != 1
        or len({row["case_id"] for row in rows if row["split"] == "val"}) != 58
    ):
        raise RuntimeError("Validation universe must contain 58 unique IXI validation cases and one atlas")
    return rows


def _verify_observed_file(row: dict[str, str]) -> None:
    path = Path(row["path"])
    if not path.is_file():
        raise FileNotFoundError(path)
    if path.stat().st_size != int(row["bytes"]):
        raise RuntimeError(f"Dataset size changed after prepare: {path}")
    actual = sha256_file(path)
    if actual != row["sha256"]:
        raise RuntimeError(f"Dataset SHA-256 changed after prepare: {path}")


def prepare_stage(args: argparse.Namespace) -> int:
    if args.num_shards < 1:
        raise ValueError("--num-shards must be >= 1")
    physical_gpus = [value.strip() for value in args.physical_gpus.split(",")]
    if (
        len(physical_gpus) != args.num_shards
        or any(not value.isdigit() for value in physical_gpus)
        or len(set(physical_gpus)) != len(physical_gpus)
    ):
        raise ValueError("--physical-gpus must list one unique non-negative integer per shard")
    if args.stage == "exploration" and (args.explore_manifest or args.explore_manifest_sha256):
        raise ValueError("Exploration must not depend on a prior exploration manifest")
    if args.stage == "confirmation" and not (args.explore_manifest and args.explore_manifest_sha256):
        raise ValueError("Confirmation requires --explore-manifest and --explore-manifest-sha256")

    head = _git("rev-parse", "HEAD")
    git_status = _git("status", "--porcelain=v1")
    if git_status:
        raise RuntimeError("C1 prepare refuses a dirty Git tree")
    checkpoint = Path(args.checkpoint or DEFAULT_CHECKPOINT).resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    checkpoint_sha256 = sha256_file(checkpoint)
    files, atlas, universe_files = _select_cases(args.stage, args.paths_profile)
    rows = _dataset_rows(files, atlas)
    universe_rows = _dataset_rows(universe_files, atlas)
    fields = ["dataset", "split", "case_id", "path", "bytes", "sha256", "mtime_utc"]
    csv_text = rows_to_csv(fields, rows)
    tsv_text = rows_to_tsv(fields, rows)
    universe_csv_text = rows_to_csv(fields, universe_rows)
    universe_sha256 = _text_sha256(universe_csv_text)
    exploration_freeze = (
        _validate_exploration_manifest(args.explore_manifest, args.explore_manifest_sha256)
        if args.stage == "confirmation"
        else None
    )
    if exploration_freeze and exploration_freeze["git_head"] != head:
        raise RuntimeError(
            "Confirmation must use the same CTCF Git HEAD as the frozen exploration; rerun exploration after code changes"
        )
    if exploration_freeze and (
        exploration_freeze["checkpoint_sha256"] != checkpoint_sha256
        or exploration_freeze["checkpoint_config"] != CONFIG_KEY
        or exploration_freeze["seed"] != args.seed
        or exploration_freeze["paths_profile"] != args.paths_profile
        or exploration_freeze["time_steps"] != TIME_STEPS
        or exploration_freeze["validation_universe_sha256"] != universe_sha256
        or exploration_freeze["confirmation_policy_sha256"] != CONFIRMATION_POLICY_SHA256
    ):
        raise RuntimeError(
            "Confirmation checkpoint, seed, paths profile, time_steps, atlas, or validation-58 universe "
            "differs from exploration"
        )
    case_ids = [_case_id(path) for path in files]
    if args.num_shards > len(case_ids):
        raise ValueError(f"--num-shards={args.num_shards} exceeds the {len(case_ids)} selected cases")
    local_specs = [dict(spec) for spec in LOCAL_OPERATOR_SPECS]
    contract = {
        "schema": "ctcf-search-c1-stage-contract-v1",
        "protocol_id": PROTOCOL_ID,
        "split_protocol_id": SPLIT_PROTOCOL_ID,
        "stage": args.stage,
        "git_head": head,
        "dataset": "IXI",
        "split": "val",
        "case_ids": case_ids,
        "case_inputs": {
            row["case_id"]: {key: row[key] for key in ("path", "bytes", "sha256")}
            for row in rows
            if row["split"] == "val"
        },
        "atlas_input": {
            key: next(row for row in rows if row["split"] == "atlas")[key] for key in ("path", "bytes", "sha256")
        },
        "selection_hashes": {case_id: _salted_case_hash(case_id) for case_id in case_ids},
        "selection_rule": (
            "first 19 by C0 salted SHA-256 rank"
            if args.stage == "exploration"
            else "remaining 39 after the frozen C0 salted validation-19"
        ),
        "ixi_test_split_accessed": False,
        "atlas_path": str(Path(atlas).resolve()),
        "datasets_csv_sha256": _text_sha256(csv_text),
        "datasets_tsv_sha256": _text_sha256(tsv_text),
        "validation_universe_sha256": universe_sha256,
        "validation_universe_case_ids": [_case_id(path) for path in universe_files],
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_sha256,
        "config": CONFIG_KEY,
        "time_steps": TIME_STEPS,
        "ctcf_l3_svf": True,
        "claim_eps": CLAIM_EPS,
        "work_eps": WORK_EPS,
        "collar_width": COLLAR_WIDTH,
        "feature_primary": "MIND-SSC radius=1 dilation=2, target-centred, soft expectation*confidence",
        "mind_reference_commit": "b229e52e44b114e2040a503334c92269750c16b2",
        "offsets_zyx": [list(offset) for offset in OFFSETS],
        "zero_offset_index": ZERO_OFFSET_INDEX,
        "global_coefficient_panel": list(GLOBAL_COEFFICIENTS) if args.stage == "exploration" else [],
        "local_operator_panel": local_specs if args.stage == "exploration" else [dict(CONFIRMATION_SPEC)],
        "utility_metrics": ["MIND-SSC", "NCCVxm win=9", "NCCVxm win=7", "proposal-support NCCVxm win=9"],
        "utility_rules": list(UTILITY_RULES),
        "utility_relative_tolerance": UTILITY_RELATIVE_TOLERANCE,
        "utility_tolerance_formula": "tau = 1e-6 * max(abs(baseline), smallest positive normal float64)",
        "confirmation_acceptance_rule": (
            "exactly certified saved FP32 mind_clip_s1_w1 and relative MIND improvement >= 1e-6"
            if args.stage == "confirmation"
            else None
        ),
        "confirmation_nonprimary_metrics_and_rules_are_diagnostic_only": args.stage == "confirmation",
        "confirmation_policy": CONFIRMATION_POLICY,
        "confirmation_policy_sha256": CONFIRMATION_POLICY_SHA256,
        "labels_used_for_transaction_decision": False,
        "labels_used_for_exploratory_evaluation": args.stage == "exploration",
        "labels_used_for_final_gate_assessment": args.stage == "confirmation",
        "num_shards": args.num_shards,
        "physical_gpus": physical_gpus,
        "shard_to_physical_gpu": {str(index): value for index, value in enumerate(physical_gpus)},
        "shards": {
            str(index): [case_id for position, case_id in enumerate(case_ids) if position % args.num_shards == index]
            for index in range(args.num_shards)
        },
        "paths_profile": args.paths_profile,
        "seed": args.seed,
        "keep_fields": args.keep_fields,
        "exploration_freeze": exploration_freeze,
    }

    stage_dir = _stage_dir(args.run_root, args.stage)
    stage_dir.mkdir(parents=True, exist_ok=True)
    contract_path = stage_dir / "stage_contract.json"
    datasets_csv = stage_dir / "datasets.csv"
    datasets_tsv = stage_dir / "datasets.tsv"
    universe_csv = stage_dir / "validation_universe.csv"
    resuming = contract_path.exists()
    if resuming:
        existing = json.loads(contract_path.read_text(encoding="utf-8"))
        if existing != contract:
            raise RuntimeError("Resume refused: C1 stage contract differs from the existing contract")
        if not datasets_csv.is_file() or datasets_csv.read_text(encoding="utf-8") != csv_text:
            raise RuntimeError("Resume refused: datasets.csv differs from the frozen contract")
        if not datasets_tsv.is_file() or datasets_tsv.read_text(encoding="utf-8") != tsv_text:
            raise RuntimeError("Resume refused: datasets.tsv differs from the frozen contract")
        if not universe_csv.is_file() or universe_csv.read_text(encoding="utf-8") != universe_csv_text:
            raise RuntimeError("Resume refused: validation_universe.csv differs from the frozen contract")
    else:
        atomic_write_text(datasets_csv, csv_text)
        atomic_write_text(datasets_tsv, tsv_text)
        atomic_write_text(universe_csv, universe_csv_text)
        atomic_write_json(contract_path, contract)
    contract_sha = sha256_file(contract_path)
    prepare_path = stage_dir / "stage_prepare.json"
    if resuming:
        if not prepare_path.is_file():
            raise RuntimeError("Resume refused: stage_prepare.json is missing")
        prepare_payload = json.loads(prepare_path.read_text(encoding="utf-8"))
        if (
            prepare_payload.get("schema") != "ctcf-search-c1-stage-prepare-v1"
            or prepare_payload.get("status") != "PREPARED"
            or prepare_payload.get("stage") != args.stage
            or prepare_payload.get("contract_sha256") != contract_sha
        ):
            raise RuntimeError("Resume refused: stage_prepare.json differs from the frozen contract")
    else:
        atomic_write_json(
            prepare_path,
            {
                "schema": "ctcf-search-c1-stage-prepare-v1",
                "status": "PREPARED",
                "stage": args.stage,
                "prepared_at_utc": _utc_now(),
                "contract_sha256": contract_sha,
            },
        )
    print(json.dumps({"stage": args.stage, "contract_sha256": contract_sha, "n_cases": len(case_ids)}))
    return 0


def _relative_improvement(baseline: float, candidate: float) -> tuple[float | None, float, bool]:
    tolerance = UTILITY_RELATIVE_TOLERANCE * max(abs(baseline), np.finfo(np.float64).tiny)
    if not (math.isfinite(baseline) and math.isfinite(candidate)):
        return None, tolerance, False
    improvement = baseline - candidate
    return improvement, tolerance, improvement >= tolerance


def _compact_exact_report(report: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "status",
        "certified",
        "complete",
        "sha256",
        "epsilon_decimal",
        "interval_lo_min",
        "interval_hi_min",
        "exact_min_over_ambiguous",
        "n_cells",
        "n_failures",
        "n_unresolved",
        "boundary_nonzero_count",
    )
    return {key: report.get(key) for key in keys}


def _materialize_for_exact_check(
    candidate: torch.Tensor,
    path: Path,
    device: torch.device,
    retain_file: bool,
) -> tuple[torch.Tensor, dict[str, Any]]:
    report: dict[str, Any]
    try:
        save_flow_npz_atomic(path, candidate.float())
        stored = load_flow_npz(path)
        exact = certify_flow_exact(stored, eps=str(CLAIM_EPS))
        report = {
            "exact_attempted": True,
            "exact_status": exact.get("status"),
            "exact_certified": exact.get("status") == "CERTIFIED" and exact.get("certified") is True,
            "candidate_npz_sha256": sha256_file(path),
            "candidate_array_sha256": exact.get("sha256"),
            "exact_report": _compact_exact_report(exact),
            "exact_error_type": None,
            "exact_error": None,
        }
        materialized = stored.to(device)
    except Exception as exc:
        report = {
            "exact_attempted": True,
            "exact_status": "ERROR",
            "exact_certified": False,
            "candidate_npz_sha256": sha256_file(path) if path.is_file() else None,
            "candidate_array_sha256": None,
            "exact_report": None,
            "exact_error_type": type(exc).__name__,
            "exact_error": str(exc),
        }
        materialized = candidate.float()
    if not retain_file:
        with suppress(FileNotFoundError):
            path.unlink()
    return materialized, report


def _candidate_metrics(
    candidate: torch.Tensor,
    fixed_norm: torch.Tensor,
    moving_norm: torch.Tensor,
    fixed_mind: torch.Tensor,
    moving_mind: torch.Tensor,
    mask: torch.Tensor,
    support_weights: torch.Tensor,
) -> dict[str, float]:
    metrics = {
        "ncc9": ncc_loss_from_normalized(fixed_norm, moving_norm, candidate, mask, win=9),
        "ncc7": ncc_loss_from_normalized(fixed_norm, moving_norm, candidate, mask, win=7),
        "support_ncc9": ncc_loss_from_normalized(
            fixed_norm,
            moving_norm,
            candidate,
            mask,
            win=9,
            weights=support_weights,
        ),
        "mind": mind_distance_from_features(fixed_mind, moving_mind, candidate, mask),
    }
    _require_finite(metrics, "candidate utility metrics")
    return metrics


def _evaluate_candidate(
    *,
    stage: str,
    case_id: str,
    candidate_id: str,
    family: str,
    feature: str,
    orientation: str,
    operator: str,
    scale: float,
    sweeps: int | None,
    coefficient_index: int | None,
    initial_psi: torch.Tensor,
    requested_delta: torch.Tensor,
    candidate: torch.Tensor,
    mask: torch.Tensor,
    fixed_norm: torch.Tensor,
    moving_norm: torch.Tensor,
    fixed_mind: torch.Tensor,
    moving_mind: torch.Tensor,
    support_weights: torch.Tensor,
    baseline_metrics: dict[str, float],
    baseline_support_ncc9: float,
    proposal_stats: dict[str, float],
    operator_report: dict[str, Any],
    work_dir: Path,
    device: torch.device,
    retain_materialized: bool,
) -> tuple[dict[str, Any], torch.Tensor, Path | None]:
    _sync(device)
    started = time.perf_counter()
    fast_bound = trilinear_cert_bound(candidate, eps=CLAIM_EPS)
    if not math.isfinite(fast_bound):
        raise RuntimeError(f"Non-finite fast certificate for {case_id}/{candidate_id}")
    fast_passed = fast_bound >= CLAIM_EPS
    exact_path: Path | None = None
    if fast_passed:
        exact_path = work_dir / f"candidate_{candidate_id}.npz"
        evaluated, exact = _materialize_for_exact_check(candidate, exact_path, device, retain_materialized)
    else:
        evaluated = candidate.float()
        exact = {
            "exact_attempted": False,
            "exact_status": "NOT_ATTEMPTED_FAST_PREDICATE_REJECTION",
            "exact_certified": False,
            "candidate_npz_sha256": None,
            "candidate_array_sha256": None,
            "exact_report": None,
            "exact_error_type": None,
            "exact_error": None,
        }
    candidate_metrics = _candidate_metrics(
        evaluated,
        fixed_norm,
        moving_norm,
        fixed_mind,
        moving_mind,
        mask,
        support_weights,
    )
    metric_baselines = {**baseline_metrics, "support_ncc9": baseline_support_ncc9}
    _require_finite(metric_baselines, "baseline utility metrics")
    decisions: dict[str, Any] = {}
    for metric in ("mind", "ncc9", "ncc7", "support_ncc9"):
        improvement, tolerance, passed = _relative_improvement(metric_baselines[metric], candidate_metrics[metric])
        decisions[f"{metric}_improvement"] = improvement
        decisions[f"{metric}_tolerance"] = tolerance
        decisions[f"{metric}_improved"] = passed
    _require_finite(
        {key: value for key, value in decisions.items() if key.endswith(("_improvement", "_tolerance"))},
        "utility decision scalars",
    )

    topology_safe = bool(exact["exact_certified"])
    rules = {
        "topology_only": topology_safe,
        "mind": topology_safe and decisions["mind_improved"],
        "ncc9": topology_safe and decisions["ncc9_improved"],
        "support_ncc9": topology_safe and decisions["support_ncc9_improved"],
        "mind_and_ncc9": topology_safe and decisions["mind_improved"] and decisions["ncc9_improved"],
    }
    field_stats = field_change_statistics(initial_psi, requested_delta, evaluated, mask)
    _require_finite(field_stats, "field-change statistics")
    _require_finite(
        {
            key: value
            for key, value in operator_report.items()
            if isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, bool)
        },
        "operator report scalars",
    )
    _sync(device)
    elapsed = time.perf_counter() - started
    row: dict[str, Any] = {
        "stage": stage,
        "case_id": case_id,
        "candidate_id": candidate_id,
        "family": family,
        "feature": feature,
        "orientation": orientation,
        "decoder": "soft",
        "operator": operator,
        "operator_status": operator_report.get("status", "COMPLETE"),
        "operator_error_type": None,
        "operator_error": None,
        "scale": scale,
        "sweeps": sweeps,
        "coefficient_index": coefficient_index,
        "coefficient": scale if family == "global" else None,
        "fast_cert_bound": fast_bound,
        "fast_certificate_passed": fast_passed,
        "candidate_elapsed_sec": elapsed,
        **{f"proposal_{key}": value for key, value in proposal_stats.items()},
        **{f"field_{key}": value for key, value in field_stats.items()},
        **{f"operator_{key}": value for key, value in operator_report.items() if key != "status"},
        **{f"baseline_{key}": value for key, value in metric_baselines.items()},
        **{f"candidate_{key}": value for key, value in candidate_metrics.items()},
        **decisions,
        **{f"rule_{key}": value for key, value in rules.items()},
        **{key: value for key, value in exact.items() if key != "exact_report"},
        "exact_report": exact["exact_report"],
        "labels_used_for_transaction_decision": False,
        "baseline_dice": None,
        "candidate_dice": None,
        "candidate_dice_delta": None,
        "baseline_sdlogj": None,
        "candidate_sdlogj": None,
        "candidate_sdlogj_delta": None,
        "baseline_j_leq0_central_percent": None,
        "candidate_j_leq0_central_percent": None,
        "baseline_j_leq0_digital10_percent": None,
        "candidate_j_leq0_digital10_percent": None,
        "baseline_trilinear_fold_percent_upper_bound": None,
        "candidate_trilinear_fold_percent_upper_bound": None,
        "baseline_trilinear_fold_status": None,
        "candidate_trilinear_fold_status": None,
        "action": None,
        "final_dice": None,
        "accepted_dice_delta": None,
        "final_sdlogj": None,
        "accepted_sdlogj_delta": None,
        "final_j_leq0_central_percent": None,
        "final_j_leq0_digital10_percent": None,
        "final_trilinear_fold_percent_upper_bound": None,
        "final_trilinear_fold_status": None,
        "final_npz_sha256": None,
        "final_array_sha256": None,
        "final_exact_status": None,
        "rollback_byte_identical": None,
    }
    return row, evaluated, exact_path


def _operator_failure_row(
    *,
    stage: str,
    case_id: str,
    spec: dict[str, Any],
    proposal_stats: dict[str, float],
    exc: Exception,
) -> dict[str, Any]:
    return {
        "stage": stage,
        "case_id": case_id,
        "candidate_id": spec["candidate_id"],
        "family": "local",
        "feature": spec["feature"],
        "orientation": spec["orientation"],
        "decoder": "soft",
        "operator": spec["operator"],
        "operator_status": "ERROR",
        "operator_error_type": type(exc).__name__,
        "operator_error": str(exc),
        "scale": spec["scale"],
        "sweeps": spec["sweeps"],
        "coefficient_index": None,
        "coefficient": None,
        "fast_cert_bound": None,
        "fast_certificate_passed": False,
        "exact_attempted": False,
        "exact_status": "NOT_ATTEMPTED_OPERATOR_ERROR",
        "exact_certified": False,
        "candidate_npz_sha256": None,
        "candidate_array_sha256": None,
        "exact_error_type": None,
        "exact_error": None,
        "exact_report": None,
        "rule_topology_only": False,
        "rule_mind": False,
        "rule_ncc9": False,
        "rule_support_ncc9": False,
        "rule_mind_and_ncc9": False,
        "labels_used_for_transaction_decision": False,
        "baseline_dice": None,
        "candidate_dice": None,
        "candidate_dice_delta": None,
        "baseline_sdlogj": None,
        "candidate_sdlogj": None,
        "candidate_sdlogj_delta": None,
        "baseline_j_leq0_central_percent": None,
        "candidate_j_leq0_central_percent": None,
        "baseline_j_leq0_digital10_percent": None,
        "candidate_j_leq0_digital10_percent": None,
        "baseline_trilinear_fold_percent_upper_bound": None,
        "candidate_trilinear_fold_percent_upper_bound": None,
        "baseline_trilinear_fold_status": None,
        "candidate_trilinear_fold_status": None,
        "action": None,
        "final_dice": None,
        "accepted_dice_delta": None,
        "final_sdlogj": None,
        "accepted_sdlogj_delta": None,
        "final_j_leq0_central_percent": None,
        "final_j_leq0_digital10_percent": None,
        "final_trilinear_fold_percent_upper_bound": None,
        "final_trilinear_fold_status": None,
        "final_npz_sha256": None,
        "final_array_sha256": None,
        "final_exact_status": None,
        "rollback_byte_identical": None,
        **{f"proposal_{key}": value for key, value in proposal_stats.items()},
    }


def _build_local_candidate(
    spec: dict[str, Any],
    initial_psi: torch.Tensor,
    proposal: torch.Tensor,
    mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    requested = float(spec["scale"]) * proposal
    if spec["operator"] == "certified_local_clip":
        current_bound = trilinear_cert_bound(initial_psi, eps=WORK_EPS)
        if not math.isfinite(current_bound) or current_bound < WORK_EPS:
            return (
                requested,
                initial_psi,
                {
                    "status": "BASELINE_BELOW_WORK_MARGIN",
                    "operator": "CERTIFIED_LOCAL_CLIP",
                    "sweeps": int(spec["sweeps"]),
                    "work_eps": WORK_EPS,
                    "current_fast_cert_bound": current_bound if math.isfinite(current_bound) else None,
                    "current_fast_cert_finite": math.isfinite(current_bound),
                    "output_fast_cert_bound": current_bound if math.isfinite(current_bound) else None,
                    "topology_noop": True,
                },
            )
        output, report = certified_local_clip_candidate(
            initial_psi,
            requested,
            mask,
            work_eps=WORK_EPS,
            sweeps=int(spec["sweeps"]),
        )
        output_bound = float(report["output_fast_cert_bound"])
        if not math.isfinite(output_bound):
            raise RuntimeError("certified_local_clip returned a non-finite certificate bound")
        retained_work_margin = output_bound >= WORK_EPS
        retained_claim_margin = output_bound >= CLAIM_EPS
        return (
            requested,
            output,
            {
                "status": "COMPLETE" if retained_work_margin else "COMPLETE_BELOW_WORK_MARGIN",
                **report,
                "retained_work_margin_after_float32": retained_work_margin,
                "retained_claim_margin_after_float32": retained_claim_margin,
            },
        )
    if spec["operator"] == "trilinear_project":
        fixed_mask = boundary_vertex_mask(initial_psi)
        target = (initial_psi + requested).float()
        output, projection = trilinear_project(
            target,
            eps=WORK_EPS,
            fixed_mask=fixed_mask,
            fixed_values=initial_psi,
        )
        boundary = fixed_mask.expand_as(initial_psi)
        if not torch.equal(output.masked_select(boundary), initial_psi.masked_select(boundary)):
            raise RuntimeError("trilinear_project changed the protected current boundary")
        status = "COMPLETE" if projection.certified and projection.cert_bound >= WORK_EPS else "UNCERTIFIED"
        projection_payload = asdict(projection)
        projection_status = projection_payload.pop("status")
        return requested, output, {**projection_payload, "projection_status": projection_status, "status": status}
    raise AssertionError(f"Unsupported local operator: {spec['operator']}")


def _proposal_for_spec(
    spec: dict[str, Any],
    mind_target: ProposalResult,
    intensity_target: ProposalResult,
    mind_reversed: ProposalResult,
) -> ProposalResult:
    if spec["feature"] == "intensity":
        return intensity_target
    if spec["orientation"] == "reversed":
        return mind_reversed
    return mind_target


def _atomic_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{destination.name}.rollback.", dir=destination.parent)
    os.close(fd)
    try:
        shutil.copyfile(source, temporary)
        os.replace(temporary, destination)
    finally:
        with suppress(FileNotFoundError):
            os.unlink(temporary)


POST_DECISION_EVALUATION_FIELDS = {
    "baseline_dice",
    "candidate_dice",
    "candidate_dice_delta",
    "final_dice",
    "accepted_dice_delta",
    "baseline_sdlogj",
    "candidate_sdlogj",
    "candidate_sdlogj_delta",
    "final_sdlogj",
    "accepted_sdlogj_delta",
    "baseline_j_leq0_central_percent",
    "candidate_j_leq0_central_percent",
    "final_j_leq0_central_percent",
    "baseline_j_leq0_digital10_percent",
    "candidate_j_leq0_digital10_percent",
    "final_j_leq0_digital10_percent",
    "baseline_trilinear_fold_percent_upper_bound",
    "candidate_trilinear_fold_percent_upper_bound",
    "final_trilinear_fold_percent_upper_bound",
    "baseline_trilinear_fold_status",
    "candidate_trilinear_fold_status",
    "final_trilinear_fold_status",
}


def _decision_input_row(row: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in row.items() if key not in POST_DECISION_EVALUATION_FIELDS}


def _decision_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "candidate_id": row["candidate_id"],
        "fast_certificate_passed": row["fast_certificate_passed"],
        "exact_status": row["exact_status"],
        "exact_certified": row["exact_certified"],
        "rule_topology_only": row["rule_topology_only"],
        "rule_mind": row["rule_mind"],
        "rule_ncc9": row["rule_ncc9"],
        "rule_support_ncc9": row["rule_support_ncc9"],
        "rule_mind_and_ncc9": row["rule_mind_and_ncc9"],
        "action": row["action"],
    }


def _write_decision_snapshots(
    case_dir: Path,
    stage: str,
    case_id: str,
    contract_sha256: str,
    rows: list[dict[str, Any]],
    attempt_id: str,
    snapshot_dir: Path | None = None,
) -> dict[str, Any]:
    target_dir = snapshot_dir or case_dir
    inputs_path = target_dir / "decision_inputs.json"
    decisions_path = target_dir / "decisions.json"
    atomic_write_json(
        inputs_path,
        {
            "schema": "ctcf-search-c1-decision-inputs-v1",
            "stage": stage,
            "case_id": case_id,
            "attempt_id": attempt_id,
            "contract_sha256": contract_sha256,
            "confirmation_policy_sha256": CONFIRMATION_POLICY_SHA256,
            "labels_loaded_to_device": False,
            "rows": [_decision_input_row(row) for row in rows],
        },
    )
    atomic_write_json(
        decisions_path,
        {
            "schema": "ctcf-search-c1-decisions-v1",
            "stage": stage,
            "case_id": case_id,
            "attempt_id": attempt_id,
            "contract_sha256": contract_sha256,
            "confirmation_policy_sha256": CONFIRMATION_POLICY_SHA256,
            "labels_loaded_to_device": False,
            "decisions": [_decision_row(row) for row in rows],
        },
    )
    return {
        "decision_inputs_path": inputs_path.relative_to(case_dir).as_posix(),
        "decision_inputs_sha256": sha256_file(inputs_path),
        "decisions_path": decisions_path.relative_to(case_dir).as_posix(),
        "decisions_sha256": sha256_file(decisions_path),
    }


def _exact_checker_gap(row: dict[str, Any]) -> bool:
    return row.get("exact_status") not in {
        "CERTIFIED",
        "NOT_CERTIFIED_BY_PREDICATE",
        "NOT_ATTEMPTED_FAST_PREDICATE_REJECTION",
    }


def _run_case(
    *,
    stage: str,
    index: int,
    path: str,
    dataset: Any,
    adapter: Any,
    model: Any,
    device: torch.device,
    labels: tuple[int, ...],
    stage_dir: Path,
    contract_sha256: str,
    keep_fields: bool,
    execution: dict[str, Any],
    input_sha256: str,
) -> list[dict[str, Any]]:
    case_id = _case_id(path)
    case_dir = stage_dir / "cases" / case_id
    complete_marker = case_dir / "case_complete.json"
    if complete_marker.is_file():
        payload = json.loads(complete_marker.read_text(encoding="utf-8"))
        _validate_case_payload(payload, stage, case_id, contract_sha256, complete_marker)
        if not keep_fields:
            shutil.rmtree(case_dir / "work", ignore_errors=True)
        return payload["rows"]

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    _sync(device)
    started = time.perf_counter()
    x_cpu, y_cpu, x_seg_cpu, y_seg_cpu = dataset[index]
    x = x_cpu.unsqueeze(0).to(device)
    y = y_cpu.unsqueeze(0).to(device)
    with torch.inference_mode():
        flow = adapter.forward(model, x, y, amp=True)
    work_dir = case_dir / "work"
    initial_psi, initial_path, initial_report = _prepare_initial_state(flow, work_dir)
    mask = geometry_mask(tuple(x.shape[-3:]), COLLAR_WIDTH, device)
    fixed_norm = masked_zscore(y, mask)
    moving_norm = masked_zscore(x, mask)
    fixed_mind = mind_ssc(fixed_norm, radius=1, dilation=2)
    moving_mind = mind_ssc(moving_norm, radius=1, dilation=2)
    mind_target = build_proposal(
        y,
        x,
        initial_psi,
        mask,
        feature="mind",
        orientation="target_centered",
        fixed_feature_override=fixed_mind,
        moving_feature_override=moving_mind,
    )
    intensity_target: ProposalResult | None = None
    mind_reversed: ProposalResult | None = None
    proposal_results = {"mind_target": mind_target}
    if stage == "exploration":
        intensity_target = build_proposal(
            y,
            x,
            initial_psi,
            mask,
            feature="intensity",
            orientation="target_centered",
            fixed_feature_override=fixed_norm,
            moving_feature_override=moving_norm,
        )
        mind_reversed = build_proposal(
            y,
            x,
            initial_psi,
            mask,
            feature="mind",
            orientation="reversed",
            fixed_feature_override=fixed_mind,
            moving_feature_override=moving_mind,
        )
        proposal_results["intensity_target"] = intensity_target
        proposal_results["mind_reversed"] = mind_reversed
    proposal_stats = {
        key: _proposal_statistics(value, value.displacement, mask) for key, value in proposal_results.items()
    }
    for name, statistics in proposal_stats.items():
        _require_finite(statistics, f"{case_id} proposal statistics {name}")
    baseline_metrics = {
        "ncc9": ncc_loss_from_normalized(fixed_norm, moving_norm, initial_psi, mask, win=9),
        "ncc7": ncc_loss_from_normalized(fixed_norm, moving_norm, initial_psi, mask, win=7),
        "mind": mind_distance_from_features(fixed_mind, moving_mind, initial_psi, mask),
    }
    supports = {key: proposal_support_weights(value.displacement, mask) for key, value in proposal_results.items()}
    baseline_support = {
        key: ncc_loss_from_normalized(
            fixed_norm,
            moving_norm,
            initial_psi,
            mask,
            win=9,
            weights=weights,
        )
        for key, weights in supports.items()
    }
    _require_finite(baseline_metrics, f"{case_id} baseline utility metrics")
    _require_finite(baseline_support, f"{case_id} support-weighted baseline metrics")

    rows: list[dict[str, Any]] = []
    evaluated_fields: list[torch.Tensor | None] = []
    confirmation_exact_path: Path | None = None
    if stage == "exploration":
        primary_stats = proposal_stats["mind_target"]
        for coefficient_index, coefficient in enumerate(GLOBAL_COEFFICIENTS):
            requested = float(coefficient) * mind_target.displacement
            candidate = (initial_psi + requested).float()
            row, evaluated, _ = _evaluate_candidate(
                stage=stage,
                case_id=case_id,
                candidate_id=f"mind_global_k{coefficient_index:02d}",
                family="global",
                feature="mind",
                orientation="target_centered",
                operator="global_scalar",
                scale=float(coefficient),
                sweeps=None,
                coefficient_index=coefficient_index,
                initial_psi=initial_psi,
                requested_delta=requested,
                candidate=candidate,
                mask=mask,
                fixed_norm=fixed_norm,
                moving_norm=moving_norm,
                fixed_mind=fixed_mind,
                moving_mind=moving_mind,
                support_weights=supports["mind_target"],
                baseline_metrics=baseline_metrics,
                baseline_support_ncc9=baseline_support["mind_target"],
                proposal_stats=primary_stats,
                operator_report={"status": "COMPLETE"},
                work_dir=work_dir,
                device=device,
                retain_materialized=keep_fields,
            )
            rows.append(row)
            evaluated_fields.append(evaluated)

        for spec in LOCAL_OPERATOR_SPECS:
            assert intensity_target is not None and mind_reversed is not None
            proposal_result = _proposal_for_spec(spec, mind_target, intensity_target, mind_reversed)
            proposal_key = (
                "intensity_target"
                if spec["feature"] == "intensity"
                else "mind_reversed"
                if spec["orientation"] == "reversed"
                else "mind_target"
            )
            try:
                requested, candidate, operator_report = _build_local_candidate(
                    spec,
                    initial_psi,
                    proposal_result.displacement,
                    mask,
                )
                row, evaluated, _ = _evaluate_candidate(
                    stage=stage,
                    case_id=case_id,
                    candidate_id=spec["candidate_id"],
                    family="local",
                    feature=spec["feature"],
                    orientation=spec["orientation"],
                    operator=spec["operator"],
                    scale=float(spec["scale"]),
                    sweeps=spec["sweeps"],
                    coefficient_index=None,
                    initial_psi=initial_psi,
                    requested_delta=requested,
                    candidate=candidate,
                    mask=mask,
                    fixed_norm=fixed_norm,
                    moving_norm=moving_norm,
                    fixed_mind=fixed_mind,
                    moving_mind=moving_mind,
                    support_weights=supports[proposal_key],
                    baseline_metrics=baseline_metrics,
                    baseline_support_ncc9=baseline_support[proposal_key],
                    proposal_stats=proposal_stats[proposal_key],
                    operator_report=operator_report,
                    work_dir=work_dir,
                    device=device,
                    retain_materialized=keep_fields,
                )
            except Exception as exc:
                row = _operator_failure_row(
                    stage=stage,
                    case_id=case_id,
                    spec=spec,
                    proposal_stats=proposal_stats[proposal_key],
                    exc=exc,
                )
                evaluated = None
                if device.type == "cuda":
                    torch.cuda.empty_cache()
            rows.append(row)
            evaluated_fields.append(evaluated)
    else:
        spec = dict(CONFIRMATION_SPEC)
        requested, candidate, operator_report = _build_local_candidate(
            spec,
            initial_psi,
            mind_target.displacement,
            mask,
        )
        row, evaluated, confirmation_exact_path = _evaluate_candidate(
            stage=stage,
            case_id=case_id,
            candidate_id=spec["candidate_id"],
            family="local",
            feature=spec["feature"],
            orientation=spec["orientation"],
            operator=spec["operator"],
            scale=float(spec["scale"]),
            sweeps=spec["sweeps"],
            coefficient_index=None,
            initial_psi=initial_psi,
            requested_delta=requested,
            candidate=candidate,
            mask=mask,
            fixed_norm=fixed_norm,
            moving_norm=moving_norm,
            fixed_mind=fixed_mind,
            moving_mind=moving_mind,
            support_weights=supports["mind_target"],
            baseline_metrics=baseline_metrics,
            baseline_support_ncc9=baseline_support["mind_target"],
            proposal_stats=proposal_stats["mind_target"],
            operator_report=operator_report,
            work_dir=work_dir,
            device=device,
            retain_materialized=True,
        )
        rows.append(row)
        evaluated_fields.append(evaluated)

        if _exact_checker_gap(row):
            row["action"] = "INVALID_CHECKER_GAP"
            failure_dir = case_dir / "failures" / execution["attempt_id"]
            decision_snapshot = _write_decision_snapshots(
                case_dir,
                stage,
                case_id,
                contract_sha256,
                rows,
                execution["attempt_id"],
                snapshot_dir=failure_dir,
            )
            atomic_write_json(
                failure_dir / "transaction_failure.json",
                {
                    "schema": "ctcf-search-c1-case-transaction-failure-v1",
                    "status": "FAILED",
                    "stage": stage,
                    "case_id": case_id,
                    "contract_sha256": contract_sha256,
                    "reason": "EXACT_CHECKER_GAP",
                    "decision_snapshot": decision_snapshot,
                    "rows": rows,
                    "execution": execution,
                },
            )
            raise RuntimeError(f"Confirmation exact checker gap for {case_id}: status={row['exact_status']}")
        final_path = work_dir / "final_primary.npz"
        if row["rule_mind"]:
            if confirmation_exact_path is None or not confirmation_exact_path.is_file():
                raise RuntimeError("Confirmation lost the exactly certified candidate before commit")
            os.replace(confirmation_exact_path, final_path)
            row["action"] = "ACCEPT"
            row["rollback_byte_identical"] = False
        else:
            _atomic_copy(initial_path, final_path)
            if confirmation_exact_path is not None:
                with suppress(FileNotFoundError):
                    confirmation_exact_path.unlink()
            row["action"] = "ROLLBACK"
            row["rollback_byte_identical"] = sha256_file(final_path) == sha256_file(initial_path)
            if not row["rollback_byte_identical"]:
                raise RuntimeError("Confirmation rollback is not byte-identical to the initial artifact")
        final_stored = load_flow_npz(final_path)
        final_exact = certify_flow_exact(final_stored, eps=str(CLAIM_EPS))
        if final_exact.get("status") != "CERTIFIED" or final_exact.get("certified") is not True:
            raise RuntimeError("Confirmation final artifact failed exact certification")
        row["final_npz_sha256"] = sha256_file(final_path)
        row["final_array_sha256"] = final_exact.get("sha256")
        row["final_exact_status"] = final_exact.get("status")
        final_field = final_stored.to(device)

    # These immutable snapshots are written before segmentation labels reach the accelerator.
    decision_snapshot = _write_decision_snapshots(
        case_dir,
        stage,
        case_id,
        contract_sha256,
        rows,
        execution["attempt_id"],
        snapshot_dir=case_dir / "decision_attempts" / execution["attempt_id"],
    )
    moving_seg = x_seg_cpu.unsqueeze(0).to(device)
    fixed_seg = y_seg_cpu.unsqueeze(0).to(device)
    baseline_dice = _dice(initial_psi, moving_seg, fixed_seg, labels)
    if not _is_finite_number(baseline_dice):
        raise RuntimeError(f"Non-finite baseline Dice for {case_id}")
    baseline_geometry = _deformation_quality_metrics(initial_psi, exact_certified=True)
    for row, candidate_field in zip(rows, evaluated_fields, strict=True):
        row["baseline_dice"] = baseline_dice
        for key, value in baseline_geometry.items():
            row[f"baseline_{key}"] = value
        if candidate_field is not None:
            candidate_dice = _dice(candidate_field, moving_seg, fixed_seg, labels)
            if not _is_finite_number(candidate_dice):
                raise RuntimeError(f"Non-finite candidate Dice for {case_id}/{row['candidate_id']}")
            row["candidate_dice"] = candidate_dice
            row["candidate_dice_delta"] = candidate_dice - baseline_dice
            candidate_geometry = _deformation_quality_metrics(
                candidate_field,
                exact_certified=bool(row["exact_certified"]),
            )
            for key, value in candidate_geometry.items():
                row[f"candidate_{key}"] = value
            row["candidate_sdlogj_delta"] = candidate_geometry["sdlogj"] - baseline_geometry["sdlogj"]
    if stage == "confirmation":
        final_dice = _dice(final_field, moving_seg, fixed_seg, labels)
        if not _is_finite_number(final_dice):
            raise RuntimeError(f"Non-finite final Dice for {case_id}")
        rows[0]["final_dice"] = final_dice
        rows[0]["accepted_dice_delta"] = final_dice - baseline_dice
        final_geometry = _deformation_quality_metrics(final_field, exact_certified=True)
        for key, value in final_geometry.items():
            rows[0][f"final_{key}"] = value
        rows[0]["accepted_sdlogj_delta"] = final_geometry["sdlogj"] - baseline_geometry["sdlogj"]

    _sync(device)
    elapsed = time.perf_counter() - started
    peak_bytes = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
    case_report = {
        "schema": "ctcf-search-c1-case-v1",
        "status": "COMPLETE",
        "stage": stage,
        "case_id": case_id,
        "attempt_id": execution["attempt_id"],
        "contract_sha256": contract_sha256,
        "input_path": str(Path(path).resolve()),
        "input_bytes": Path(path).stat().st_size,
        "input_sha256": input_sha256,
        "selection_sha256": _salted_case_hash(case_id),
        "initial": initial_report,
        "proposal_statistics": proposal_stats,
        "decision_snapshot": decision_snapshot,
        "execution": execution,
        "elapsed_sec": elapsed,
        "peak_gpu_bytes": peak_bytes,
        "rows": rows,
    }
    atomic_write_json(complete_marker, case_report)
    if not keep_fields:
        shutil.rmtree(work_dir, ignore_errors=True)
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return rows


def _expected_candidate_ids(stage: str) -> list[str]:
    if stage == "confirmation":
        return [CONFIRMATION_SPEC["candidate_id"]]
    return [
        *[f"mind_global_k{index:02d}" for index in range(len(GLOBAL_COEFFICIENTS))],
        *[spec["candidate_id"] for spec in LOCAL_OPERATOR_SPECS],
    ]


def _validate_decision_snapshots(
    payload: dict[str, Any],
    rows: list[dict[str, Any]],
    stage: str,
    case_id: str,
    contract_sha256: str,
    marker_path: Path,
    attempt_id: str,
) -> None:
    snapshot = payload.get("decision_snapshot") or {}
    expected_files = {
        "decision_inputs_path": (
            f"decision_attempts/{attempt_id}/decision_inputs.json",
            "decision_inputs_sha256",
        ),
        "decisions_path": (f"decision_attempts/{attempt_id}/decisions.json", "decisions_sha256"),
    }
    loaded: dict[str, dict[str, Any]] = {}
    for path_key, (expected_name, hash_key) in expected_files.items():
        if snapshot.get(path_key) != expected_name or not SHA256_RE.fullmatch(str(snapshot.get(hash_key, ""))):
            raise RuntimeError(f"Invalid decision snapshot locator in {marker_path}")
        path = marker_path.parent / expected_name
        if not path.is_file() or sha256_file(path) != snapshot[hash_key]:
            raise RuntimeError(f"Decision snapshot is missing or changed: {path}")
        loaded[Path(expected_name).name] = json.loads(path.read_text(encoding="utf-8"))

    common = {
        "stage": stage,
        "case_id": case_id,
        "attempt_id": attempt_id,
        "contract_sha256": contract_sha256,
        "confirmation_policy_sha256": CONFIRMATION_POLICY_SHA256,
        "labels_loaded_to_device": False,
    }
    inputs = loaded["decision_inputs.json"]
    decisions = loaded["decisions.json"]
    if (
        inputs.get("schema") != "ctcf-search-c1-decision-inputs-v1"
        or any(inputs.get(key) != value for key, value in common.items())
        or inputs.get("rows") != [_decision_input_row(row) for row in rows]
        or decisions.get("schema") != "ctcf-search-c1-decisions-v1"
        or any(decisions.get(key) != value for key, value in common.items())
        or decisions.get("decisions") != [_decision_row(row) for row in rows]
    ):
        raise RuntimeError(f"Decision snapshot does not reconstruct from the completed rows: {marker_path}")


def _required_row_scalars(row: dict[str, Any]) -> dict[str, Any]:
    names = (
        "fast_cert_bound",
        "baseline_mind",
        "baseline_ncc9",
        "baseline_ncc7",
        "baseline_support_ncc9",
        "candidate_mind",
        "candidate_ncc9",
        "candidate_ncc7",
        "candidate_support_ncc9",
        "mind_improvement",
        "mind_tolerance",
        "ncc9_improvement",
        "ncc9_tolerance",
        "ncc7_improvement",
        "ncc7_tolerance",
        "support_ncc9_improvement",
        "support_ncc9_tolerance",
        "baseline_dice",
        "candidate_dice",
        "candidate_dice_delta",
        "baseline_sdlogj",
        "candidate_sdlogj",
        "candidate_sdlogj_delta",
        "baseline_j_leq0_central_percent",
        "candidate_j_leq0_central_percent",
        "baseline_j_leq0_digital10_percent",
        "candidate_j_leq0_digital10_percent",
    )
    return {name: row.get(name) for name in names}


def _valid_deformation_quality(row: dict[str, Any], prefix: str, *, exact_certified: bool) -> bool:
    numeric = (
        row.get(f"{prefix}_sdlogj"),
        row.get(f"{prefix}_j_leq0_central_percent"),
        row.get(f"{prefix}_j_leq0_digital10_percent"),
    )
    if any(not _is_finite_number(value) or float(value) < 0.0 for value in numeric):
        return False
    expected_status = "ZERO_BY_EXACT_CERTIFICATE" if exact_certified else "NOT_ESTABLISHED"
    expected_upper_bound = 0.0 if exact_certified else None
    return (
        row.get(f"{prefix}_trilinear_fold_status") == expected_status
        and row.get(f"{prefix}_trilinear_fold_percent_upper_bound") == expected_upper_bound
    )


def _valid_row_decisions(row: dict[str, Any]) -> bool:
    if row.get("operator_status") == "ERROR":
        return all(row.get(f"rule_{rule}") is False for rule in UTILITY_RULES)
    fast_bound = row.get("fast_cert_bound")
    if not _is_finite_number(fast_bound):
        return False
    fast_passed = float(fast_bound) >= CLAIM_EPS
    if row.get("fast_certificate_passed") is not fast_passed or row.get("exact_attempted") is not fast_passed:
        return False
    exact_status = row.get("exact_status")
    exact_certified = exact_status == "CERTIFIED"
    if row.get("exact_certified") is not exact_certified:
        return False
    if fast_passed:
        if exact_status not in {
            "CERTIFIED",
            "NOT_CERTIFIED_BY_PREDICATE",
            "INCONCLUSIVE_RESOURCE_LIMIT",
            "INVALID_INPUT",
            "ERROR",
        }:
            return False
        if exact_status != "ERROR" and (
            not SHA256_RE.fullmatch(str(row.get("candidate_npz_sha256", "")))
            or not SHA256_RE.fullmatch(str(row.get("candidate_array_sha256", "")))
        ):
            return False
    elif exact_status != "NOT_ATTEMPTED_FAST_PREDICATE_REJECTION" or row.get("exact_certified") is not False:
        return False

    improved: dict[str, bool] = {}
    for metric in ("mind", "ncc9", "ncc7", "support_ncc9"):
        expected_improvement, expected_tolerance, expected_passed = _relative_improvement(
            float(row[f"baseline_{metric}"]),
            float(row[f"candidate_{metric}"]),
        )
        if (
            row.get(f"{metric}_improvement") != expected_improvement
            or row.get(f"{metric}_tolerance") != expected_tolerance
            or row.get(f"{metric}_improved") is not expected_passed
        ):
            return False
        improved[metric] = expected_passed
    expected_rules = {
        "topology_only": exact_certified,
        "mind": exact_certified and improved["mind"],
        "ncc9": exact_certified and improved["ncc9"],
        "support_ncc9": exact_certified and improved["support_ncc9"],
        "mind_and_ncc9": exact_certified and improved["mind"] and improved["ncc9"],
    }
    return all(row.get(f"rule_{rule}") is expected for rule, expected in expected_rules.items())


def _validate_case_payload(
    payload: dict[str, Any],
    stage: str,
    case_id: str,
    contract_sha256: str,
    marker_path: Path,
    contract: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    rows = payload.get("rows")
    expected_ids = _expected_candidate_ids(stage)
    initial = payload.get("initial") or {}
    phi_exact = initial.get("phi_exact") or {}
    psi_exact = initial.get("psi_exact") or {}
    if (
        payload.get("status") != "COMPLETE"
        or payload.get("contract_sha256") != contract_sha256
        or payload.get("stage") != stage
        or payload.get("case_id") != case_id
        or not isinstance(rows, list)
        or [row.get("candidate_id") for row in rows] != expected_ids
        or any(row.get("stage") != stage or row.get("case_id") != case_id for row in rows)
        or any(row.get("labels_used_for_transaction_decision") is not False for row in rows)
        or any(not _is_finite_number(row.get("baseline_dice")) for row in rows)
        or any(not _valid_deformation_quality(row, "baseline", exact_certified=True) for row in rows)
        or any(
            row.get("operator_status") != "ERROR"
            and (
                any(not _is_finite_number(value) for value in _required_row_scalars(row).values())
                or not _valid_deformation_quality(
                    row,
                    "candidate",
                    exact_certified=row.get("exact_certified") is True,
                )
            )
            for row in rows
        )
        or any(not all(isinstance(row.get(f"rule_{rule}"), bool) for rule in UTILITY_RULES) for row in rows)
        or any(not _valid_row_decisions(row) for row in rows)
        or phi_exact.get("status") != "CERTIFIED"
        or phi_exact.get("certified") is not True
        or phi_exact.get("boundary_nonzero_count") != 0
        or psi_exact.get("status") != "CERTIFIED"
        or psi_exact.get("certified") is not True
        or not SHA256_RE.fullmatch(str(initial.get("phi_npz_sha256", "")))
        or not SHA256_RE.fullmatch(str(initial.get("psi_npz_sha256", "")))
        or not SHA256_RE.fullmatch(str(phi_exact.get("sha256", "")))
        or not SHA256_RE.fullmatch(str(psi_exact.get("sha256", "")))
        or not _is_finite_number(payload.get("elapsed_sec"))
        or not _is_finite_number(payload.get("peak_gpu_bytes"))
    ):
        raise RuntimeError(f"Invalid C1 case marker: {marker_path}")
    for name, statistics in (payload.get("proposal_statistics") or {}).items():
        if not isinstance(statistics, dict):
            raise RuntimeError(f"Invalid proposal statistics for {name}: {marker_path}")
        _require_finite(statistics, f"proposal statistics {name}")
    execution = payload.get("execution") or {}
    case_load = execution.get("checkpoint_load_report") or {}
    if (
        not re.fullmatch(r"[A-Za-z0-9_.-]+", str(execution.get("attempt_id", "")))
        or execution.get("checkpoint_sha256") is None
        or execution.get("model_load_strict") is not True
        or execution.get("contract_sha256") != contract_sha256
        or payload.get("attempt_id") != execution.get("attempt_id")
        or not all(isinstance(execution.get(key), str) and execution.get(key) for key in ("host", "python", "torch"))
        or not str(execution.get("device", "")).startswith("cuda")
        or case_load.get("strict") is not True
        or set(case_load.get("missing_keys") or []) != set(case_load.get("allowed_missing_buffers") or [])
        or bool(case_load.get("unexpected_keys"))
    ):
        raise RuntimeError(f"Invalid per-case execution provenance: {marker_path}")
    _validate_decision_snapshots(
        payload,
        rows,
        stage,
        case_id,
        contract_sha256,
        marker_path,
        execution["attempt_id"],
    )
    if contract is not None:
        expected_shard = next(
            index for index in range(contract["num_shards"]) if case_id in contract["shards"][str(index)]
        )
        if (
            execution.get("shard_index") != expected_shard
            or execution.get("physical_gpu") != contract["shard_to_physical_gpu"][str(expected_shard)]
            or execution.get("checkpoint_sha256") != contract["checkpoint_sha256"]
            or execution.get("seed") != contract["seed"]
            or execution.get("deterministic") is not True
            or payload.get("input_path") != contract["case_inputs"][case_id]["path"]
            or payload.get("input_bytes") != contract["case_inputs"][case_id]["bytes"]
            or payload.get("input_sha256") != contract["case_inputs"][case_id]["sha256"]
            or payload.get("selection_sha256") != contract["selection_hashes"][case_id]
        ):
            raise RuntimeError(f"Per-case execution provenance differs from the frozen contract: {marker_path}")
    if stage == "confirmation":
        row = rows[0]
        if (
            row.get("final_exact_status") != "CERTIFIED"
            or row.get("action") not in {"ACCEPT", "ROLLBACK"}
            or not _is_finite_number(row.get("final_dice"))
            or not _is_finite_number(row.get("accepted_dice_delta"))
            or not _is_finite_number(row.get("accepted_sdlogj_delta"))
            or not _valid_deformation_quality(row, "final", exact_certified=True)
            or (
                row.get("action") == "ACCEPT"
                and (
                    row.get("exact_certified") is not True
                    or row.get("rule_mind") is not True
                    or row.get("final_npz_sha256") != row.get("candidate_npz_sha256")
                    or row.get("final_array_sha256") != row.get("candidate_array_sha256")
                )
            )
            or (row.get("action") == "ROLLBACK" and row.get("rule_mind") is not False)
            or (row.get("action") == "ROLLBACK" and row.get("rollback_byte_identical") is not True)
            or (
                row.get("action") == "ROLLBACK"
                and (
                    row.get("final_npz_sha256") != initial.get("psi_npz_sha256")
                    or row.get("final_array_sha256") != psi_exact.get("sha256")
                )
            )
        ):
            raise RuntimeError(f"Invalid C1 confirmation transaction marker: {marker_path}")
    return rows


def _valid_worker_report(
    report: dict[str, Any],
    contract: dict[str, Any],
    contract_sha256: str,
    shard_index: int,
    attempt_id: str,
) -> bool:
    checkpoint = report.get("checkpoint") or {}
    execution = report.get("execution") or {}
    missing = set(checkpoint.get("missing_keys") or [])
    allowed_missing = set(checkpoint.get("allowed_missing_buffers") or [])
    assigned = contract["shards"][str(shard_index)]
    computed = report.get("computed_case_ids") or []
    reused = report.get("reused_case_ids") or []
    partition_ok = (
        len(computed) + len(reused) == len(assigned)
        and len(set([*computed, *reused])) == len(assigned)
        and [case_id for case_id in assigned if case_id in set(computed)] == computed
        and [case_id for case_id in assigned if case_id in set(reused)] == reused
    )
    load_ok = (
        execution.get("model_loaded") is True
        and checkpoint.get("strict") is True
        and missing == allowed_missing
        and not checkpoint.get("unexpected_keys")
        if computed
        else execution.get("model_loaded") is False
    )
    return bool(
        report.get("status") == "COMPLETE"
        and report.get("schema") == "ctcf-search-c1-worker-attempt-v1"
        and report.get("stage") == contract["stage"]
        and report.get("attempt_id") == attempt_id
        and report.get("shard_index") == shard_index
        and report.get("num_shards") == contract["num_shards"]
        and report.get("contract_sha256") == contract_sha256
        and report.get("assigned_case_ids") == assigned
        and partition_ok
        and all(isinstance(execution.get(key), str) and execution.get(key) for key in ("host", "python", "torch"))
        and checkpoint.get("sha256") == contract["checkpoint_sha256"]
        and checkpoint.get("path") == contract["checkpoint"]
        and load_ok
        and execution.get("seed") == contract["seed"]
        and execution.get("deterministic") is True
        and execution.get("physical_gpu") == contract["shard_to_physical_gpu"][str(shard_index)]
    )


def worker_stage(args: argparse.Namespace) -> int:
    stage_dir = _stage_dir(args.run_root, args.stage)
    contract, contract_sha = _verify_contract(stage_dir, args.contract_sha256)
    if contract["stage"] != args.stage or contract["protocol_id"] != PROTOCOL_ID:
        raise RuntimeError("Worker stage does not match the frozen contract")
    if args.num_shards != contract["num_shards"]:
        raise RuntimeError("Worker --num-shards differs from the frozen contract")
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError("--shard-index must satisfy 0 <= index < num_shards")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", args.attempt_id):
        raise ValueError("--attempt-id contains unsupported characters")
    if args.physical_gpu != contract["shard_to_physical_gpu"][str(args.shard_index)]:
        raise RuntimeError("Worker physical GPU differs from the frozen shard mapping")
    if _git("rev-parse", "HEAD") != contract["git_head"] or _git("status", "--porcelain=v1"):
        raise RuntimeError("Worker code provenance differs from the clean prepared contract")

    rows = _read_dataset_rows(stage_dir, contract)
    by_id = {row["case_id"]: row for row in rows}
    assigned_ids = contract["shards"][str(args.shard_index)]
    assigned_rows = [by_id[case_id] for case_id in assigned_ids]
    atlas_row = by_id["atlas"]
    for row in [*assigned_rows, atlas_row]:
        _verify_observed_file(row)
    checkpoint = Path(contract["checkpoint"])
    if sha256_file(checkpoint) != contract["checkpoint_sha256"]:
        raise RuntimeError("Checkpoint SHA-256 differs from the frozen C1 contract")

    started_at = _utc_now()
    attempt_dir = stage_dir / "workers" / "attempts" / args.attempt_id
    worker_marker = attempt_dir / f"worker_{args.shard_index:02d}.json"
    failure_marker = attempt_dir / f"worker_{args.shard_index:02d}_failure.json"
    if worker_marker.exists() or failure_marker.exists():
        raise RuntimeError(f"Attempt output already exists; use a new --attempt-id: {attempt_dir}")

    reused_ids: list[str] = []
    pending_rows: list[dict[str, str]] = []
    for row in assigned_rows:
        case_id = row["case_id"]
        case_marker = stage_dir / "cases" / case_id / "case_complete.json"
        if not case_marker.is_file():
            pending_rows.append(row)
            continue
        case_payload = json.loads(case_marker.read_text(encoding="utf-8"))
        _validate_case_payload(case_payload, args.stage, case_id, contract_sha, case_marker, contract)
        reused_ids.append(case_id)
        if not contract["keep_fields"]:
            shutil.rmtree(stage_dir / "cases" / case_id / "work", ignore_errors=True)

    computed_ids: list[str] = []
    active_case_id: str | None = None
    load_report: dict[str, Any] = {
        "strict": None,
        "missing_keys": [],
        "allowed_missing_buffers": [],
        "unexpected_keys": [],
    }
    device: torch.device | None = None
    try:
        if pending_rows:
            device = setup_device(args.gpu, seed=contract["seed"], deterministic=True)
            if device.type != "cuda":
                raise RuntimeError("C1 worker requires an explicitly assigned CUDA device")
            adapter, model, load_report = _build_model(str(checkpoint), contract["config"], device)
            files = [row["path"] for row in pending_rows]
            dataset = build_infer_dataset("IXI", files, atlas_row["path"])
            labels = tuple(metric_profile_for("IXI").labels)
            case_execution = {
                "attempt_id": args.attempt_id,
                "shard_index": args.shard_index,
                "contract_sha256": contract_sha,
                "host": platform.node(),
                "python": platform.python_version(),
                "torch": torch.__version__,
                "device": str(device),
                "physical_gpu": args.physical_gpu,
                "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
                "seed": contract["seed"],
                "deterministic": True,
                "checkpoint_sha256": contract["checkpoint_sha256"],
                "model_load_strict": load_report.get("strict") is True,
                "checkpoint_load_report": load_report,
            }
            for index, row in enumerate(pending_rows):
                active_case_id = row["case_id"]
                print(
                    f"[shard {args.shard_index + 1}/{args.num_shards}] "
                    f"[{index + 1}/{len(pending_rows)}] IXI {row['case_id']}",
                    flush=True,
                )
                _run_case(
                    stage=args.stage,
                    index=index,
                    path=row["path"],
                    dataset=dataset,
                    adapter=adapter,
                    model=model,
                    device=device,
                    labels=labels,
                    stage_dir=stage_dir,
                    contract_sha256=contract_sha,
                    keep_fields=bool(contract["keep_fields"]),
                    execution=case_execution,
                    input_sha256=row["sha256"],
                )
                computed_ids.append(row["case_id"])
                active_case_id = None
        report = {
            "schema": "ctcf-search-c1-worker-attempt-v1",
            "status": "COMPLETE",
            "stage": args.stage,
            "attempt_id": args.attempt_id,
            "shard_index": args.shard_index,
            "num_shards": args.num_shards,
            "assigned_case_ids": assigned_ids,
            "computed_case_ids": computed_ids,
            "reused_case_ids": reused_ids,
            "contract_sha256": contract_sha,
            "started_at_utc": started_at,
            "completed_at_utc": _utc_now(),
            "execution": {
                "host": platform.node(),
                "python": platform.python_version(),
                "torch": torch.__version__,
                "device": str(device) if device is not None else None,
                "physical_gpu": args.physical_gpu,
                "gpu_name": (
                    torch.cuda.get_device_name(device) if device is not None and device.type == "cuda" else None
                ),
                "seed": contract["seed"],
                "deterministic": True,
                "model_loaded": bool(pending_rows),
            },
            "checkpoint": {**load_report, "path": str(checkpoint), "sha256": contract["checkpoint_sha256"]},
        }
        atomic_write_json(worker_marker, report)
    except BaseException as exc:
        if not contract["keep_fields"]:
            for case_id in assigned_ids:
                shutil.rmtree(stage_dir / "cases" / case_id / "work", ignore_errors=True)
        atomic_write_json(
            failure_marker,
            {
                "schema": "ctcf-search-c1-worker-attempt-failure-v1",
                "status": "FAILED",
                "stage": args.stage,
                "attempt_id": args.attempt_id,
                "shard_index": args.shard_index,
                "contract_sha256": contract_sha,
                "computed_case_ids": computed_ids,
                "reused_case_ids": reused_ids,
                "active_case_id": active_case_id,
                "started_at_utc": started_at,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "completed_at_utc": _utc_now(),
                "execution": {
                    "host": platform.node(),
                    "python": platform.python_version(),
                    "torch": torch.__version__,
                    "physical_gpu": args.physical_gpu,
                    "seed": contract["seed"],
                    "deterministic": True,
                },
                "checkpoint": {"path": str(checkpoint), "sha256": contract["checkpoint_sha256"]},
            },
        )
        raise
    return 0


def _bootstrap_ci(values: np.ndarray) -> dict[str, Any]:
    if values.size == 0:
        return {"method": "not_available", "low": None, "high": None, "replicates": 0, "seed": 0}
    generator = np.random.default_rng(0)
    samples = generator.choice(values, size=(10_000, values.size), replace=True).mean(axis=1)
    low, high = np.quantile(samples, (0.025, 0.975))
    return {
        "method": "case-bootstrap percentile CI for the mean; diagnostic only",
        "low": float(low),
        "high": float(high),
        "replicates": 10_000,
        "seed": 0,
    }


def _sign_summary(values: np.ndarray) -> dict[str, Any]:
    if values.size == 0 or not np.isfinite(values).all():
        raise RuntimeError("Sign summary requires a non-empty finite vector")
    return {
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "min": float(values.min()),
        "max": float(values.max()),
        "improved": int((values > 0.0).sum()),
        "worsened": int((values < 0.0).sum()),
        "unchanged": int((values == 0.0).sum()),
        "mean_ci95": _bootstrap_ci(values),
    }


def _distribution_summary(values: np.ndarray) -> dict[str, float]:
    if values.size == 0 or not np.isfinite(values).all():
        raise RuntimeError("Distribution summary requires a non-empty finite vector")
    return {
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "min": float(values.min()),
        "max": float(values.max()),
    }


def _operator_rule_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    candidate_ids = sorted({row["candidate_id"] for row in rows})
    for candidate_id in candidate_ids:
        selected = [row for row in rows if row["candidate_id"] == candidate_id]
        for rule in UTILITY_RULES:
            accepted = [bool(row.get(f"rule_{rule}")) for row in selected]
            effective = np.array(
                [
                    float(row["candidate_dice_delta"])
                    if is_accepted and row.get("candidate_dice_delta") is not None
                    else 0.0
                    for row, is_accepted in zip(selected, accepted, strict=True)
                ],
                dtype=np.float64,
            )
            returned_dice = np.array(
                [
                    float(row["candidate_dice"]) if is_accepted else float(row["baseline_dice"])
                    for row, is_accepted in zip(selected, accepted, strict=True)
                ],
                dtype=np.float64,
            )
            returned_sdlogj = np.array(
                [
                    float(row["candidate_sdlogj"]) if is_accepted else float(row["baseline_sdlogj"])
                    for row, is_accepted in zip(selected, accepted, strict=True)
                ],
                dtype=np.float64,
            )
            returned_sdlogj_delta = np.array(
                [
                    float(row["candidate_sdlogj_delta"])
                    if is_accepted and row.get("candidate_sdlogj_delta") is not None
                    else 0.0
                    for row, is_accepted in zip(selected, accepted, strict=True)
                ],
                dtype=np.float64,
            )
            returned_central_folds = np.array(
                [
                    float(row["candidate_j_leq0_central_percent"])
                    if is_accepted
                    else float(row["baseline_j_leq0_central_percent"])
                    for row, is_accepted in zip(selected, accepted, strict=True)
                ],
                dtype=np.float64,
            )
            returned_digital_folds = np.array(
                [
                    float(row["candidate_j_leq0_digital10_percent"])
                    if is_accepted
                    else float(row["baseline_j_leq0_digital10_percent"])
                    for row, is_accepted in zip(selected, accepted, strict=True)
                ],
                dtype=np.float64,
            )
            signs = _sign_summary(effective)
            dice_absolute = _distribution_summary(returned_dice)
            sdlogj_absolute = _distribution_summary(returned_sdlogj)
            sdlogj_delta = _distribution_summary(returned_sdlogj_delta)
            result.append(
                {
                    "candidate_id": candidate_id,
                    "family": selected[0]["family"],
                    "feature": selected[0]["feature"],
                    "orientation": selected[0]["orientation"],
                    "operator": selected[0]["operator"],
                    "scale": selected[0]["scale"],
                    "sweeps": selected[0]["sweeps"],
                    "coefficient_index": selected[0]["coefficient_index"],
                    "rule": rule,
                    "n_cases": len(selected),
                    "n_operator_errors": sum(row["operator_status"] == "ERROR" for row in selected),
                    "n_accepted": sum(accepted),
                    "returned_dice_mean": dice_absolute["mean"],
                    "returned_dice_median": dice_absolute["median"],
                    "returned_dice_min": dice_absolute["min"],
                    "returned_dice_max": dice_absolute["max"],
                    "accepted_dice_delta_mean": signs["mean"],
                    "accepted_dice_delta_median": signs["median"],
                    "accepted_dice_delta_min": signs["min"],
                    "accepted_dice_delta_max": signs["max"],
                    "returned_sdlogj_mean": sdlogj_absolute["mean"],
                    "returned_sdlogj_median": sdlogj_absolute["median"],
                    "returned_sdlogj_min": sdlogj_absolute["min"],
                    "returned_sdlogj_max": sdlogj_absolute["max"],
                    "accepted_sdlogj_delta_mean": sdlogj_delta["mean"],
                    "accepted_sdlogj_delta_median": sdlogj_delta["median"],
                    "returned_j_leq0_central_percent_mean": float(returned_central_folds.mean()),
                    "returned_j_leq0_central_percent_max": float(returned_central_folds.max()),
                    "returned_j_leq0_digital10_percent_mean": float(returned_digital_folds.mean()),
                    "returned_j_leq0_digital10_percent_max": float(returned_digital_folds.max()),
                    "returned_trilinear_fold_percent_upper_bound": 0.0,
                    "returned_trilinear_fold_status": "ZERO_BY_EXACT_CERTIFICATE",
                    "improved": signs["improved"],
                    "worsened": signs["worsened"],
                    "unchanged": signs["unchanged"],
                }
            )
    return result


def _per_case_rows(rows: list[dict[str, Any]], stage: str) -> list[dict[str, Any]]:
    primary_id = CONFIRMATION_SPEC["candidate_id"]
    selected = [row for row in rows if row["candidate_id"] == primary_id]
    fields = (
        "case_id",
        "operator_status",
        "exact_status",
        "exact_certified",
        "baseline_dice",
        "candidate_dice",
        "candidate_dice_delta",
        "baseline_sdlogj",
        "candidate_sdlogj",
        "candidate_sdlogj_delta",
        "baseline_j_leq0_central_percent",
        "candidate_j_leq0_central_percent",
        "baseline_j_leq0_digital10_percent",
        "candidate_j_leq0_digital10_percent",
        "baseline_trilinear_fold_percent_upper_bound",
        "candidate_trilinear_fold_percent_upper_bound",
        "baseline_trilinear_fold_status",
        "candidate_trilinear_fold_status",
        "rule_topology_only",
        "rule_mind",
        "rule_ncc9",
        "rule_support_ncc9",
        "rule_mind_and_ncc9",
        "action",
        "final_dice",
        "accepted_dice_delta",
        "final_sdlogj",
        "accepted_sdlogj_delta",
        "final_j_leq0_central_percent",
        "final_j_leq0_digital10_percent",
        "final_trilinear_fold_percent_upper_bound",
        "final_trilinear_fold_status",
        "rollback_byte_identical",
    )
    return [
        {"stage": stage, "primary_candidate_id": primary_id, **{key: row.get(key) for key in fields}}
        for row in selected
    ]


def _summarise(rows: list[dict[str, Any]], stage: str, n_cases: int) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    expected_per_case = 24 if stage == "exploration" else 1
    structural_pass = len(rows) == n_cases * expected_per_case
    auxiliary_errors = sum(row["operator_status"] == "ERROR" for row in rows)
    exact_failures = sum(bool(row["exact_attempted"]) and not bool(row["exact_certified"]) for row in rows)
    exact_checker_errors = sum(row["exact_status"] == "ERROR" for row in rows)
    exact_predicate_rejections = sum(
        bool(row["exact_attempted"]) and row["exact_status"] == "NOT_CERTIFIED_BY_PREDICATE" for row in rows
    )
    exact_inconclusive_gaps = sum(bool(row["exact_attempted"]) and _exact_checker_gap(row) for row in rows)
    rule_rows = _operator_rule_rows(rows)
    primary = [row for row in rows if row["candidate_id"] == CONFIRMATION_SPEC["candidate_id"]]
    baseline_dice = _distribution_summary(np.array([float(row["baseline_dice"]) for row in primary], dtype=np.float64))
    baseline_sdlogj = _distribution_summary(
        np.array([float(row["baseline_sdlogj"]) for row in primary], dtype=np.float64)
    )
    if stage == "exploration":
        if not structural_pass:
            integrity_status = "FAIL"
            scientific_status = "INVALID"
        elif auxiliary_errors or exact_inconclusive_gaps:
            integrity_status = "PASS_WITH_AUXILIARY_GAPS"
            scientific_status = "EXPLORATORY_COMPLETE_WITH_GAPS"
        else:
            integrity_status = "PASS"
            scientific_status = "EXPLORATORY_COMPLETE"
        summary = {
            "execution_integrity_status": integrity_status,
            "scientific_status": scientific_status,
            "n_cases": n_cases,
            "n_candidate_rows": len(rows),
            "expected_candidate_rows": n_cases * expected_per_case,
            "auxiliary_operator_errors": auxiliary_errors,
            "saved_fp32_exact_failures": exact_failures,
            "saved_fp32_exact_predicate_rejections": exact_predicate_rejections,
            "saved_fp32_exact_inconclusive_gaps": exact_inconclusive_gaps,
            "exact_checker_errors": exact_checker_errors,
            "baseline_dice_mean": baseline_dice["mean"],
            "baseline_dice_median": baseline_dice["median"],
            "baseline_sdlogj_mean": baseline_sdlogj["mean"],
            "baseline_sdlogj_median": baseline_sdlogj["median"],
            "baseline_j_leq0_central_percent_max": max(
                float(row["baseline_j_leq0_central_percent"]) for row in primary
            ),
            "baseline_j_leq0_digital10_percent_max": max(
                float(row["baseline_j_leq0_digital10_percent"]) for row in primary
            ),
            "baseline_trilinear_fold_percent_upper_bound": 0.0,
            "baseline_trilinear_fold_status": "ZERO_BY_EXACT_CERTIFICATE",
            "labels_used_for_transaction_decision": False,
            "labels_used_for_exploratory_evaluation": True,
            "test_split_accessed": False,
            "interpretation": "Selection data only; no confirmation claim is made on the exploration-19 cases.",
        }
        return summary, rule_rows

    final_exact = all(row["final_exact_status"] == "CERTIFIED" for row in primary)
    rollback_exact = all(row["action"] != "ROLLBACK" or row["rollback_byte_identical"] is True for row in primary)
    integrity = (
        structural_pass and auxiliary_errors == 0 and exact_inconclusive_gaps == 0 and final_exact and rollback_exact
    )
    deltas = np.array([float(row["accepted_dice_delta"]) for row in primary], dtype=np.float64)
    signs = _sign_summary(deltas)
    final_dice = _distribution_summary(np.array([float(row["final_dice"]) for row in primary], dtype=np.float64))
    final_sdlogj = _distribution_summary(np.array([float(row["final_sdlogj"]) for row in primary], dtype=np.float64))
    sdlogj_deltas = _distribution_summary(
        np.array([float(row["accepted_sdlogj_delta"]) for row in primary], dtype=np.float64)
    )
    promising = integrity and signs["mean"] > 0.0 and signs["median"] > 0.0 and signs["improved"] > signs["worsened"]
    summary = {
        "execution_integrity_status": "PASS" if integrity else "FAIL",
        "scientific_status": "PROMISING" if promising else "NOT_PROMISING",
        "n_cases": n_cases,
        "n_candidate_rows": len(rows),
        "primary_candidate_id": CONFIRMATION_SPEC["candidate_id"],
        "accepted_cases": sum(row["action"] == "ACCEPT" for row in primary),
        "rolled_back_cases": sum(row["action"] == "ROLLBACK" for row in primary),
        "all_final_maps_exactly_certified": final_exact,
        "all_rollbacks_byte_identical": rollback_exact,
        "candidate_exact_predicate_rejections": exact_predicate_rejections,
        "candidate_exact_inconclusive_gaps": exact_inconclusive_gaps,
        "baseline_dice_mean": baseline_dice["mean"],
        "baseline_dice_median": baseline_dice["median"],
        "final_dice_mean": final_dice["mean"],
        "final_dice_median": final_dice["median"],
        "primary_accepted_dice_delta_mean": signs["mean"],
        "primary_accepted_dice_delta_median": signs["median"],
        "primary_accepted_dice_delta_min": signs["min"],
        "primary_accepted_dice_delta_max": signs["max"],
        "primary_improved_cases": signs["improved"],
        "primary_worsened_cases": signs["worsened"],
        "primary_unchanged_cases": signs["unchanged"],
        "primary_mean_dice_delta_ci95": signs["mean_ci95"],
        "baseline_sdlogj_mean": baseline_sdlogj["mean"],
        "baseline_sdlogj_median": baseline_sdlogj["median"],
        "final_sdlogj_mean": final_sdlogj["mean"],
        "final_sdlogj_median": final_sdlogj["median"],
        "accepted_sdlogj_delta_mean": sdlogj_deltas["mean"],
        "accepted_sdlogj_delta_median": sdlogj_deltas["median"],
        "final_j_leq0_central_percent_mean": float(
            np.mean([float(row["final_j_leq0_central_percent"]) for row in primary])
        ),
        "final_j_leq0_central_percent_max": max(float(row["final_j_leq0_central_percent"]) for row in primary),
        "final_j_leq0_digital10_percent_mean": float(
            np.mean([float(row["final_j_leq0_digital10_percent"]) for row in primary])
        ),
        "final_j_leq0_digital10_percent_max": max(float(row["final_j_leq0_digital10_percent"]) for row in primary),
        "final_trilinear_fold_percent_upper_bound": 0.0,
        "final_trilinear_fold_status": "ZERO_BY_EXACT_CERTIFICATE",
        "promising_rule": "integrity PASS and mean>0 and median>0 and improved>worsened",
        "nonprimary_metrics_and_rules_are_diagnostic_only": True,
        "labels_used_for_transaction_decision": False,
        "labels_used_for_final_gate_assessment": True,
        "test_split_accessed": False,
    }
    return summary, rule_rows


def _csv_fields(rows: list[dict[str, Any]], preferred: list[str]) -> list[str]:
    keys = {key for row in rows for key in row if key != "exact_report"}
    return [key for key in preferred if key in keys] + sorted(keys - set(preferred))


def selfcheck_stage(args: argparse.Namespace) -> int:
    candidate_ids = _expected_candidate_ids("exploration")
    checks = {
        "fixed_c0_exploration_count_is_19": len(IXI_DEVELOPMENT_CASES) == 19,
        "global_panel_is_k_0_through_16": tuple(2.0**-index for index in range(17)) == GLOBAL_COEFFICIENTS,
        "exploration_has_24_unique_candidates": len(candidate_ids) == len(set(candidate_ids)) == 24,
        "confirmation_candidate_is_in_exploration": CONFIRMATION_SPEC["candidate_id"] in candidate_ids,
        "confirmation_policy_hash_is_canonical": (_payload_sha256(CONFIRMATION_POLICY) == CONFIRMATION_POLICY_SHA256),
        "relative_tolerance_accepts_clear_improvement": _relative_improvement(2.0, 2.0 - 4e-6)[2],
        "relative_tolerance_rejects_no_change": not _relative_improvement(2.0, 2.0)[2],
        "predicate_rejection_is_not_checker_gap": not _exact_checker_gap(
            {"exact_status": "NOT_CERTIFIED_BY_PREDICATE"}
        ),
        "resource_limit_is_checker_gap": _exact_checker_gap({"exact_status": "INCONCLUSIVE_RESOURCE_LIMIT"}),
    }
    failed = sorted(name for name, passed in checks.items() if not passed)
    payload = {
        "schema": "ctcf-search-c1-selfcheck-v1",
        "protocol_id": PROTOCOL_ID,
        "status": "PASS" if not failed else "FAIL",
        "checks": checks,
        "failed": failed,
        "confirmation_policy_sha256": CONFIRMATION_POLICY_SHA256,
    }
    atomic_write_json(args.output, payload)
    if failed:
        raise RuntimeError(f"C1 self-check failed: {failed}")
    print(json.dumps(payload, indent=2))
    return 0


def finalize_stage(args: argparse.Namespace) -> int:
    stage_dir = _stage_dir(args.run_root, args.stage)
    contract, contract_sha = _verify_contract(stage_dir, args.contract_sha256)
    if contract["stage"] != args.stage or contract["protocol_id"] != PROTOCOL_ID:
        raise RuntimeError("Finalize stage does not match the frozen C1 contract")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", args.attempt_id):
        raise ValueError("--attempt-id contains unsupported characters")
    if _git("rev-parse", "HEAD") != contract["git_head"] or _git("status", "--porcelain=v1"):
        raise RuntimeError("Finalize code provenance differs from the clean prepared contract")
    manifest_path = stage_dir / "run_manifest.json"
    existing_manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.is_file() else None
    if existing_manifest and (
        existing_manifest.get("status") not in {"COMPLETE", "COMPLETE_WITH_GAPS"}
        or existing_manifest.get("contract_sha256") != contract_sha
    ):
        raise RuntimeError("Resume refused: existing C1 stage manifest does not match the frozen contract")

    dataset_rows = _read_dataset_rows(stage_dir, contract)
    for row in dataset_rows:
        _verify_observed_file(row)
    universe_rows = _read_validation_universe(stage_dir, contract)
    if [row["case_id"] for row in universe_rows if row["split"] == "val"] != contract["validation_universe_case_ids"]:
        raise RuntimeError("Validation-universe case order differs from the frozen contract")
    for row in universe_rows:
        _verify_observed_file(row)
    if sha256_file(Path(contract["checkpoint"])) != contract["checkpoint_sha256"]:
        raise RuntimeError("Checkpoint changed between C1 prepare and finalize")
    freeze = contract.get("exploration_freeze")
    if freeze:
        observed_freeze = _validate_exploration_manifest(freeze["path"], freeze["sha256"])
        if observed_freeze != freeze:
            raise RuntimeError("Frozen exploration inputs changed before confirmation finalization")

    worker_reports: list[dict[str, Any]] = []
    for shard_index in range(contract["num_shards"]):
        path = stage_dir / "workers" / "attempts" / args.attempt_id / f"worker_{shard_index:02d}.json"
        if not path.is_file():
            raise FileNotFoundError(f"Missing completed worker report: {path}")
        report = json.loads(path.read_text(encoding="utf-8"))
        if not _valid_worker_report(report, contract, contract_sha, shard_index, args.attempt_id):
            raise RuntimeError(f"Worker report does not match the contract: {path}")
        worker_reports.append({"path": path.relative_to(stage_dir).as_posix(), "sha256": sha256_file(path), **report})

    all_rows: list[dict[str, Any]] = []
    case_marker_hashes: dict[str, str] = {}
    for case_id in contract["case_ids"]:
        path = stage_dir / "cases" / case_id / "case_complete.json"
        if not path.is_file():
            raise FileNotFoundError(f"Missing completed case marker: {path}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        case_rows = _validate_case_payload(payload, args.stage, case_id, contract_sha, path, contract)
        case_marker_hashes[case_id] = sha256_file(path)
        all_rows.extend(case_rows)

    summary, rule_rows = _summarise(all_rows, args.stage, len(contract["case_ids"]))
    if summary["execution_integrity_status"] == "FAIL":
        raise RuntimeError("C1 stage execution integrity failed during finalization")
    preferred = [
        "stage",
        "case_id",
        "candidate_id",
        "family",
        "feature",
        "orientation",
        "operator",
        "operator_status",
        "scale",
        "sweeps",
        "coefficient_index",
        "coefficient",
    ]
    serializable_rows = [{key: value for key, value in row.items() if key != "exact_report"} for row in all_rows]
    per_candidate_path = stage_dir / "per_candidate.csv"
    atomic_write_text(per_candidate_path, rows_to_csv(_csv_fields(serializable_rows, preferred), serializable_rows))
    per_case = _per_case_rows(all_rows, args.stage)
    per_case_path = stage_dir / "per_case.csv"
    atomic_write_text(per_case_path, rows_to_csv(list(per_case[0].keys()), per_case))
    operator_path = stage_dir / "operator_rule_summary.csv"
    atomic_write_text(operator_path, rows_to_csv(list(rule_rows[0].keys()), rule_rows))
    summary_path = stage_dir / "summary.json"
    atomic_write_json(summary_path, summary)

    prepare_path = stage_dir / "stage_prepare.json"
    prepare_payload = json.loads(prepare_path.read_text(encoding="utf-8"))
    if (
        prepare_payload.get("schema") != "ctcf-search-c1-stage-prepare-v1"
        or prepare_payload.get("status") != "PREPARED"
        or prepare_payload.get("stage") != args.stage
        or prepare_payload.get("contract_sha256") != contract_sha
    ):
        raise RuntimeError("stage_prepare.json differs from the frozen contract")
    completed_at = (
        existing_manifest["completed_at_utc"]
        if existing_manifest and existing_manifest.get("finalize_attempt_id") == args.attempt_id
        else _utc_now()
    )
    manifest = {
        "schema": "ctcf-search-c1-stage-manifest-v1",
        "protocol_id": PROTOCOL_ID,
        "run_id": args.run_root.resolve().name,
        "status": (
            "COMPLETE_WITH_GAPS" if summary["execution_integrity_status"] == "PASS_WITH_AUXILIARY_GAPS" else "COMPLETE"
        ),
        "stage": args.stage,
        "started_at_utc": prepare_payload["prepared_at_utc"],
        "completed_at_utc": completed_at,
        "contract_sha256": contract_sha,
        "finalize_attempt_id": args.attempt_id,
        "confirmation_policy": CONFIRMATION_POLICY,
        "confirmation_policy_sha256": CONFIRMATION_POLICY_SHA256,
        "code": {
            "git_head": _git("rev-parse", "HEAD"),
            "branch": _git("branch", "--show-current"),
            "git_status": _git("status", "--porcelain=v1"),
        },
        "execution": {
            "host": platform.node(),
            "num_shards": contract["num_shards"],
            "paths_profile": contract["paths_profile"],
            "seed": contract["seed"],
            "deterministic": True,
            "time_steps": contract["time_steps"],
            "physical_gpus": contract["physical_gpus"],
        },
        "checkpoint": {
            "path": contract["checkpoint"],
            "sha256": contract["checkpoint_sha256"],
            "config": contract["config"],
            "strict_load_for_every_case_computation": True,
        },
        "workers": worker_reports,
        "case_marker_sha256": case_marker_hashes,
        "exploration_freeze": freeze,
        "files": {
            "stage_contract_sha256": sha256_file(stage_dir / "stage_contract.json"),
            "datasets_sha256": sha256_file(stage_dir / "datasets.csv"),
            "validation_universe_sha256": sha256_file(stage_dir / "validation_universe.csv"),
            "per_candidate_sha256": sha256_file(per_candidate_path),
            "per_case_sha256": sha256_file(per_case_path),
            "operator_rule_summary_sha256": sha256_file(operator_path),
            "summary_sha256": sha256_file(summary_path),
        },
        "summary": summary,
        "storage": {
            "heavy_fields_retained_in_run_directory": contract["keep_fields"],
            "heavy_fields_in_standard_package": False,
            "compact_outputs_only_by_default": True,
        },
    }
    atomic_write_json(manifest_path, manifest)
    print(json.dumps(summary, indent=2), flush=True)
    return 3 if summary["execution_integrity_status"] == "PASS_WITH_AUXILIARY_GAPS" else 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run CTCF search Gate C1 as a frozen prepare/worker/finalize multi-GPU protocol."
    )
    subparsers = parser.add_subparsers(dest="action", required=True)

    selfcheck = subparsers.add_parser("selfcheck", help="Validate pure C1 protocol invariants.")
    selfcheck.add_argument("--output", type=Path, required=True)

    prepare = subparsers.add_parser("prepare", help="Freeze a stage contract and dataset manifest.")
    prepare.add_argument("--stage", choices=["exploration", "confirmation"], required=True)
    prepare.add_argument("--run-root", type=Path, required=True)
    prepare.add_argument("--paths-profile", type=int, default=3)
    prepare.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    prepare.add_argument("--seed", type=int, default=0)
    prepare.add_argument("--num-shards", type=int, required=True)
    prepare.add_argument("--physical-gpus", required=True)
    prepare.add_argument("--keep-fields", action="store_true")
    prepare.add_argument("--explore-manifest")
    prepare.add_argument("--explore-manifest-sha256")

    worker = subparsers.add_parser("worker", help="Run one deterministic case shard with one strict model.")
    worker.add_argument("--stage", choices=["exploration", "confirmation"], required=True)
    worker.add_argument("--run-root", type=Path, required=True)
    worker.add_argument("--contract-sha256", required=True)
    worker.add_argument("--shard-index", type=int, required=True)
    worker.add_argument("--num-shards", type=int, required=True)
    worker.add_argument("--gpu", type=int, default=0)
    worker.add_argument("--physical-gpu", default="UNKNOWN")
    worker.add_argument("--attempt-id", required=True)

    finalize = subparsers.add_parser("finalize", help="Validate all shards and build deterministic compact outputs.")
    finalize.add_argument("--stage", choices=["exploration", "confirmation"], required=True)
    finalize.add_argument("--run-root", type=Path, required=True)
    finalize.add_argument("--contract-sha256", required=True)
    finalize.add_argument("--attempt-id", required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.action == "selfcheck":
        return selfcheck_stage(args)
    if args.action == "prepare":
        return prepare_stage(args)
    if args.action == "worker":
        return worker_stage(args)
    if args.action == "finalize":
        return finalize_stage(args)
    raise AssertionError(f"Unhandled action: {args.action}")


if __name__ == "__main__":
    raise SystemExit(main())
