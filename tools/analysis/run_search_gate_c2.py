from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
import platform
import re
import shutil
import time
from datetime import datetime, timezone
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np
import torch

from experiments.core.inference_metrics import metric_profile_for
from experiments.core.inference_runtime import build_infer_dataset
from tools.analysis.run_artifacts import atomic_write_json, atomic_write_text, rows_to_csv, rows_to_tsv, sha256_file
from tools.analysis.run_search_gate_c0 import _build_model, _case_id, _dice, _prepare_initial_state, _salted_case_hash
from tools.analysis.run_search_gate_c1 import (
    CLAIM_EPS,
    COLLAR_WIDTH,
    CONFIG_KEY,
    DEFAULT_CHECKPOINT,
    PROTOCOL_ID as C1_PROTOCOL_ID,
    SPLIT_PROTOCOL_ID,
    TIME_STEPS,
    _bootstrap_ci,
    _candidate_metrics,
    _deformation_quality_metrics,
    _distribution_summary,
    _git,
    _is_finite_number,
    _payload_sha256,
    _proposal_statistics,
    _relative_improvement,
    _sign_summary,
    _sync,
    _text_sha256,
)
from tools.analysis.transactional_search import (
    build_proposal,
    certified_local_clip_candidate,
    geometry_mask,
    identity_collar,
    load_flow_npz,
    masked_zscore,
    mind_ssc,
    proposal_support_weights,
    save_flow_npz_atomic,
    smooth_proposal,
)
from utils import setup_device
from utils.cert_exact import certify_flow_exact
from utils.field import trilinear_cert_bound

PROTOCOL_ID = "CTCF-SEARCH-GATE-C2-V1"
SCHEMA_VERSION = "v1"
MAX_STEPS = 4
MARGIN_SCHEDULE = (0.0011, 0.001075, 0.00105, 0.001025)
SDLOGJ_RELATIVE_CAP = 0.005
MIN_MEAN_DICE_DELTA = 0.001
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")

TRAJECTORIES: tuple[dict[str, Any], ...] = (
    {
        "trajectory_id": "mind_s1_sm1",
        "scale": 1.0,
        "smoothing_passes": 1,
        "sdlogj_relative_cap": None,
    },
    {
        "trajectory_id": "mind_s2_sm1",
        "scale": 2.0,
        "smoothing_passes": 1,
        "sdlogj_relative_cap": None,
    },
    {
        "trajectory_id": "mind_s2_sm2",
        "scale": 2.0,
        "smoothing_passes": 2,
        "sdlogj_relative_cap": None,
    },
    {
        "trajectory_id": "mind_s2_sm2_sdlogj_cap",
        "scale": 2.0,
        "smoothing_passes": 2,
        "sdlogj_relative_cap": SDLOGJ_RELATIVE_CAP,
    },
)

C2_POLICY: dict[str, Any] = {
    "scope": "all 58 already-open IXI validation cases; IXI test is unreachable",
    "max_steps": MAX_STEPS,
    "work_margin_schedule": list(MARGIN_SCHEDULE),
    "exact_claim_eps": CLAIM_EPS,
    "proposal": "target-centred MIND-SSC soft expectation*confidence, recomputed from current Psi",
    "trajectories": [dict(item) for item in TRAJECTORIES],
    "acceptance": {
        "all": "saved/reloaded FP32 field exactly certified and relative MIND decrease >= 1e-6",
        "regularized_branch": "also cumulative SDlogJ <= baseline * 1.005",
        "failure": "byte-identical rollback and stop the trajectory",
    },
    "diagnostics_only": ["NCCVxm win=9", "NCCVxm win=7", "proposal-support NCCVxm win=9"],
    "labels_used_for_transaction_decision": False,
    "selection_gate": {
        "mean_dice_delta_min": MIN_MEAN_DICE_DELTA,
        "paired_mean_vs_c1": ">0",
        "paired_median_vs_c1": ">0",
        "paired_bootstrap_ci95_low_vs_c1": ">0",
        "mean_sdlogj_delta": "<= pooled C1 reference",
        "mean_digital10_percent": "<= pooled C1 reference",
    },
}
C2_POLICY_SHA256 = _payload_sha256(C2_POLICY)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def _float(row: dict[str, str], key: str) -> float:
    value = float(row[key])
    if not math.isfinite(value):
        raise RuntimeError(f"Non-finite {key} in C1 reference row for {row.get('case_id')}")
    return value


def _validate_c1_stage(path_value: str, expected_sha256: str, stage: str) -> dict[str, Any]:
    if not SHA256_RE.fullmatch(expected_sha256.lower()):
        raise ValueError(f"C1 {stage} manifest SHA-256 must be 64 lowercase hex characters")
    path = Path(path_value).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    actual = sha256_file(path)
    if actual != expected_sha256.lower():
        raise RuntimeError(f"C1 {stage} manifest hash mismatch: expected={expected_sha256}, actual={actual}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    expected_cases = 19 if stage == "exploration" else 39
    summary = payload.get("summary") or {}
    if (
        payload.get("schema") != "ctcf-search-c1-stage-manifest-v1"
        or payload.get("protocol_id") != C1_PROTOCOL_ID
        or payload.get("stage") != stage
        or payload.get("status") != "COMPLETE"
        or summary.get("execution_integrity_status") != "PASS"
        or summary.get("n_cases") != expected_cases
        or (payload.get("code") or {}).get("git_status")
        or summary.get("test_split_accessed") is not False
    ):
        raise RuntimeError(f"C2 requires a clean COMPLETE C1 {stage} manifest")
    files = payload.get("files") or {}
    required = {
        "stage_contract_sha256": "stage_contract.json",
        "datasets_sha256": "datasets.csv",
        "validation_universe_sha256": "validation_universe.csv",
        "per_case_sha256": "per_case.csv",
        "summary_sha256": "summary.json",
    }
    for key, name in required.items():
        artifact = path.parent / name
        if not artifact.is_file() or files.get(key) != sha256_file(artifact):
            raise RuntimeError(f"C1 {stage} artifact differs from its manifest: {artifact}")
    contract = json.loads((path.parent / "stage_contract.json").read_text(encoding="utf-8"))
    if (
        contract.get("protocol_id") != C1_PROTOCOL_ID
        or contract.get("stage") != stage
        or contract.get("ixi_test_split_accessed") is not False
        or contract.get("validation_universe_sha256") != files["validation_universe_sha256"]
        or len(contract.get("case_ids") or []) != expected_cases
    ):
        raise RuntimeError(f"C1 {stage} contract is inconsistent with its manifest")
    rows = _read_csv(path.parent / "per_case.csv")
    if len(rows) != expected_cases or len({row["case_id"] for row in rows}) != expected_cases:
        raise RuntimeError(f"C1 {stage} per_case.csv has invalid case coverage")
    return {
        "path": str(path),
        "sha256": actual,
        "run_id": payload.get("run_id"),
        "stage": stage,
        "git_head": (payload.get("code") or {}).get("git_head"),
        "checkpoint_sha256": (payload.get("checkpoint") or {}).get("sha256"),
        "checkpoint_config": (payload.get("checkpoint") or {}).get("config"),
        "paths_profile": (payload.get("execution") or {}).get("paths_profile"),
        "seed": (payload.get("execution") or {}).get("seed"),
        "time_steps": (payload.get("execution") or {}).get("time_steps"),
        "validation_universe_sha256": files["validation_universe_sha256"],
        "per_case_path": str((path.parent / "per_case.csv").resolve()),
        "per_case_sha256": files["per_case_sha256"],
        "case_ids": contract["case_ids"],
    }


def _build_c1_reference(exploration: dict[str, Any], confirmation: dict[str, Any]) -> list[dict[str, Any]]:
    if (
        exploration["git_head"] != confirmation["git_head"]
        or exploration["checkpoint_sha256"] != confirmation["checkpoint_sha256"]
        or exploration["checkpoint_config"] != confirmation["checkpoint_config"]
        or exploration["paths_profile"] != confirmation["paths_profile"]
        or exploration["seed"] != confirmation["seed"]
        or exploration["time_steps"] != confirmation["time_steps"]
        or exploration["validation_universe_sha256"] != confirmation["validation_universe_sha256"]
    ):
        raise RuntimeError("C1 exploration and confirmation do not share one checkpoint/data/protocol contract")
    result: list[dict[str, Any]] = []
    for source in (exploration, confirmation):
        for row in _read_csv(Path(source["per_case_path"])):
            if row.get("primary_candidate_id") != "mind_clip_s1_w1" or row.get("exact_certified") != "True":
                raise RuntimeError(f"Unexpected C1 primary row for {row.get('case_id')}")
            final_dice_key = "candidate_dice" if source["stage"] == "exploration" else "final_dice"
            final_sdlogj_key = "candidate_sdlogj" if source["stage"] == "exploration" else "final_sdlogj"
            final_digital_key = (
                "candidate_j_leq0_digital10_percent"
                if source["stage"] == "exploration"
                else "final_j_leq0_digital10_percent"
            )
            result.append(
                {
                    "case_id": row["case_id"],
                    "c1_stage": source["stage"],
                    "c1_manifest_sha256": source["sha256"],
                    "c1_per_case_sha256": source["per_case_sha256"],
                    "baseline_dice": _float(row, "baseline_dice"),
                    "final_dice": _float(row, final_dice_key),
                    "dice_delta": _float(row, final_dice_key) - _float(row, "baseline_dice"),
                    "baseline_sdlogj": _float(row, "baseline_sdlogj"),
                    "final_sdlogj": _float(row, final_sdlogj_key),
                    "sdlogj_delta": _float(row, final_sdlogj_key) - _float(row, "baseline_sdlogj"),
                    "final_digital10_percent": _float(row, final_digital_key),
                }
            )
    if len(result) != 58 or len({row["case_id"] for row in result}) != 58:
        raise RuntimeError("Frozen C1 reference must contain exactly 58 unique validation cases")
    return result


def _select_all_validation(paths_profile: int) -> tuple[list[str], str]:
    from experiments.core.path_profiles import get_dataset_paths

    paths = get_dataset_paths(paths_profile, "IXI")
    files = sorted(glob.glob(os.path.join(paths["val_dir"], "*.pkl")))
    by_id = {_case_id(path): path for path in files}
    if len(files) != 58 or len(by_id) != 58:
        raise RuntimeError(f"C2 requires exactly 58 unique IXI validation cases, found {len(by_id)}")
    order = [case_id for _, case_id in sorted((_salted_case_hash(case_id), case_id) for case_id in by_id)]
    return [by_id[case_id] for case_id in order], str(paths["atlas_path"])


def _dataset_rows(files: list[str], atlas: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for value in [*files, atlas]:
        path = Path(value).resolve()
        stat = path.stat()
        rows.append(
            {
                "dataset": "IXI",
                "split": "atlas" if value == atlas else "val",
                "case_id": "atlas" if value == atlas else _case_id(value),
                "path": str(path),
                "bytes": stat.st_size,
                "sha256": sha256_file(path),
                "mtime_utc": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat().replace("+00:00", "Z"),
            }
        )
    return rows


def _verify_file(row: dict[str, Any]) -> None:
    path = Path(row["path"])
    if not path.is_file() or path.stat().st_size != int(row["bytes"]) or sha256_file(path) != row["sha256"]:
        raise RuntimeError(f"Frozen input changed: {path}")


def prepare_stage(args: argparse.Namespace) -> int:
    if args.num_shards < 1:
        raise ValueError("--num-shards must be positive")
    physical = [value.strip() for value in args.physical_gpus.split(",")]
    if (
        len(physical) != args.num_shards
        or len(set(physical)) != len(physical)
        or any(not x.isdigit() for x in physical)
    ):
        raise ValueError("--physical-gpus must contain one unique non-negative index per shard")
    if _git("status", "--porcelain=v1"):
        raise RuntimeError("C2 prepare refuses a dirty Git tree")
    head = _git("rev-parse", "HEAD")
    checkpoint = Path(args.checkpoint).resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    exploration = _validate_c1_stage(args.c1_exploration_manifest, args.c1_exploration_sha256, "exploration")
    confirmation = _validate_c1_stage(args.c1_confirmation_manifest, args.c1_confirmation_sha256, "confirmation")
    reference = _build_c1_reference(exploration, confirmation)
    if exploration["checkpoint_sha256"] != sha256_file(checkpoint):
        raise RuntimeError("C2 checkpoint differs from the frozen C1 checkpoint")
    if exploration["paths_profile"] != args.paths_profile or exploration["seed"] != args.seed:
        raise RuntimeError("C2 paths profile or seed differs from C1")

    files, atlas = _select_all_validation(args.paths_profile)
    rows = _dataset_rows(files, atlas)
    fields = ["dataset", "split", "case_id", "path", "bytes", "sha256", "mtime_utc"]
    datasets_csv_text = rows_to_csv(fields, rows)
    datasets_tsv_text = rows_to_tsv(fields, rows)
    c1_fields = list(reference[0].keys())
    c1_csv_text = rows_to_csv(c1_fields, reference)
    if [row["case_id"] for row in reference] != [_case_id(path) for path in files]:
        # C1 rows are stage-grouped; coverage must match, while C2 uses the canonical salted order.
        by_case = {row["case_id"]: row for row in reference}
        if set(by_case) != {_case_id(path) for path in files}:
            raise RuntimeError("C1 reference does not cover the C2 validation universe")
        reference = [by_case[_case_id(path)] for path in files]
        c1_csv_text = rows_to_csv(c1_fields, reference)

    case_ids = [_case_id(path) for path in files]
    contract = {
        "schema": f"ctcf-search-c2-contract-{SCHEMA_VERSION}",
        "protocol_id": PROTOCOL_ID,
        "split_protocol_id": SPLIT_PROTOCOL_ID,
        "git_head": head,
        "dataset": "IXI",
        "split": "val58_already_open",
        "case_ids": case_ids,
        "ixi_test_split_accessed": False,
        "case_inputs": {row["case_id"]: row for row in rows if row["split"] == "val"},
        "atlas_input": next(row for row in rows if row["split"] == "atlas"),
        "datasets_csv_sha256": _text_sha256(datasets_csv_text),
        "datasets_tsv_sha256": _text_sha256(datasets_tsv_text),
        "c1_reference_sha256": _text_sha256(c1_csv_text),
        "c1_sources": {"exploration": exploration, "confirmation": confirmation},
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "config": CONFIG_KEY,
        "time_steps": TIME_STEPS,
        "paths_profile": args.paths_profile,
        "seed": args.seed,
        "claim_eps": CLAIM_EPS,
        "policy": C2_POLICY,
        "policy_sha256": C2_POLICY_SHA256,
        "num_shards": args.num_shards,
        "physical_gpus": physical,
        "shard_to_physical_gpu": {str(i): value for i, value in enumerate(physical)},
        "shards": {
            str(i): [case_id for position, case_id in enumerate(case_ids) if position % args.num_shards == i]
            for i in range(args.num_shards)
        },
        "keep_fields": False,
    }
    root = args.run_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    paths = {
        "contract": root / "c2_contract.json",
        "datasets_csv": root / "datasets.csv",
        "datasets_tsv": root / "datasets.tsv",
        "c1_reference": root / "c1_reference.csv",
    }
    if paths["contract"].exists():
        if json.loads(paths["contract"].read_text(encoding="utf-8")) != contract:
            raise RuntimeError("Resume refused: C2 contract changed")
        expected_text = {
            "datasets_csv": datasets_csv_text,
            "datasets_tsv": datasets_tsv_text,
            "c1_reference": c1_csv_text,
        }
        for key, content in expected_text.items():
            if paths[key].read_text(encoding="utf-8") != content:
                raise RuntimeError(f"Resume refused: {paths[key]} changed")
    else:
        atomic_write_text(paths["datasets_csv"], datasets_csv_text)
        atomic_write_text(paths["datasets_tsv"], datasets_tsv_text)
        atomic_write_text(paths["c1_reference"], c1_csv_text)
        atomic_write_json(paths["contract"], contract)
    contract_sha = sha256_file(paths["contract"])
    prepare = root / "prepare.json"
    if not prepare.exists():
        atomic_write_json(
            prepare,
            {
                "schema": f"ctcf-search-c2-prepare-{SCHEMA_VERSION}",
                "status": "PREPARED",
                "prepared_at_utc": _utc_now(),
                "contract_sha256": contract_sha,
            },
        )
    print(json.dumps({"contract_sha256": contract_sha, "n_cases": 58, "policy_sha256": C2_POLICY_SHA256}))
    return 0


def _load_contract(root: Path, expected_sha: str) -> tuple[dict[str, Any], str]:
    path = root.resolve() / "c2_contract.json"
    actual = sha256_file(path)
    if actual != expected_sha.lower():
        raise RuntimeError(f"C2 contract hash mismatch: expected={expected_sha}, actual={actual}")
    contract = json.loads(path.read_text(encoding="utf-8"))
    if (
        contract.get("schema") != f"ctcf-search-c2-contract-{SCHEMA_VERSION}"
        or contract.get("protocol_id") != PROTOCOL_ID
        or contract.get("policy_sha256") != C2_POLICY_SHA256
        or contract.get("policy") != C2_POLICY
        or contract.get("ixi_test_split_accessed") is not False
    ):
        raise RuntimeError("Unsupported or altered C2 contract")
    return contract, actual


def _read_frozen_rows(root: Path, contract: dict[str, Any]) -> tuple[list[dict[str, str]], dict[str, dict[str, str]]]:
    dataset_path = root / "datasets.csv"
    reference_path = root / "c1_reference.csv"
    if sha256_file(dataset_path) != contract["datasets_csv_sha256"]:
        raise RuntimeError("datasets.csv differs from C2 contract")
    if sha256_file(reference_path) != contract["c1_reference_sha256"]:
        raise RuntimeError("c1_reference.csv differs from C2 contract")
    rows = _read_csv(dataset_path)
    refs = {row["case_id"]: row for row in _read_csv(reference_path)}
    if len(rows) != 59 or len(refs) != 58:
        raise RuntimeError("C2 frozen inputs have invalid cardinality")
    return rows, refs


def _exact_materialize(
    candidate: torch.Tensor, path: Path, device: torch.device
) -> tuple[torch.Tensor, dict[str, Any]]:
    save_flow_npz_atomic(path, candidate.float())
    stored = load_flow_npz(path)
    exact = certify_flow_exact(stored, eps=str(CLAIM_EPS))
    status = exact.get("status")
    if status in {"ERROR", "INVALID_INPUT", "INCONCLUSIVE_RESOURCE_LIMIT"}:
        raise RuntimeError(f"Exact checker integrity failure: {status}")
    return stored.to(device), {
        "status": status,
        "certified": status == "CERTIFIED" and exact.get("certified") is True,
        "npz_sha256": sha256_file(path),
        "array_sha256": exact.get("sha256"),
        "interval_lo_min": exact.get("interval_lo_min"),
        "exact_min_over_ambiguous": exact.get("exact_min_over_ambiguous"),
        "n_failures": exact.get("n_failures"),
        "n_unresolved": exact.get("n_unresolved"),
    }


def _row_without_labels(row: dict[str, Any]) -> dict[str, Any]:
    postdecision = {
        "baseline_dice",
        "candidate_dice",
        "candidate_dice_delta",
        "returned_dice",
        "returned_dice_delta",
        "returned_sdlogj",
        "returned_sdlogj_delta",
        "returned_j_leq0_central_percent",
        "returned_j_leq0_digital10_percent",
        "returned_trilinear_fold_percent_upper_bound",
    }
    return {key: value for key, value in row.items() if key not in postdecision}


def _build_branch_rows(rows: list[dict[str, Any]], c1_row: dict[str, Any]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for spec in TRAJECTORIES:
        trajectory_id = spec["trajectory_id"]
        selected = [row for row in rows if row["trajectory_id"] == trajectory_id]
        if len(selected) != MAX_STEPS:
            raise RuntimeError(f"Expected {MAX_STEPS} rows for {trajectory_id}")
        last = selected[-1]
        result.append(
            {
                "case_id": last["case_id"],
                "trajectory_id": trajectory_id,
                "accepted_steps": sum(row["action"] == "ACCEPT" for row in selected),
                "terminal_action": next(
                    (row["action"] for row in selected if row["action"] not in {"ACCEPT", "STOPPED"}),
                    "MAX_STEPS_REACHED",
                ),
                "baseline_dice": last["baseline_dice"],
                "final_dice": last["returned_dice"],
                "dice_delta": last["returned_dice"] - last["baseline_dice"],
                "c1_final_dice": float(c1_row["final_dice"]),
                "paired_dice_advantage_vs_c1": last["returned_dice"] - float(c1_row["final_dice"]),
                "baseline_sdlogj": last["returned_sdlogj"] - last["returned_sdlogj_delta"],
                "final_sdlogj": last["returned_sdlogj"],
                "sdlogj_delta": last["returned_sdlogj_delta"],
                "c1_sdlogj_delta": float(c1_row["sdlogj_delta"]),
                "final_j_leq0_central_percent": last["returned_j_leq0_central_percent"],
                "final_j_leq0_digital10_percent": last["returned_j_leq0_digital10_percent"],
                "c1_final_digital10_percent": float(c1_row["final_digital10_percent"]),
                "final_trilinear_fold_percent_upper_bound": 0.0,
                "final_exact_status": "CERTIFIED",
            }
        )
    return result


def _run_case(
    *,
    index: int,
    input_row: dict[str, str],
    c1_row: dict[str, str],
    dataset: Any,
    adapter: Any,
    model: Any,
    device: torch.device,
    labels: tuple[int, ...],
    root: Path,
    contract_sha: str,
    execution: dict[str, Any],
) -> list[dict[str, Any]]:
    case_id = input_row["case_id"]
    case_dir = root / "cases" / case_id
    marker = case_dir / "case_complete.json"
    if marker.is_file():
        payload = json.loads(marker.read_text(encoding="utf-8"))
        return _validate_case(payload, marker, case_id, contract_sha, input_row)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    x_cpu, y_cpu, moving_seg_cpu, fixed_seg_cpu = dataset[index]
    x = x_cpu.unsqueeze(0).to(device)
    y = y_cpu.unsqueeze(0).to(device)
    with torch.inference_mode():
        flow = adapter.forward(model, x, y, amp=True)
    work = case_dir / "work"
    initial, _, initial_report = _prepare_initial_state(flow, work / "initial")
    initial_exact = initial_report["psi_exact"]
    if initial_exact.get("status") != "CERTIFIED":
        raise RuntimeError(f"Initial Psi not exactly certified for {case_id}")
    mask = geometry_mask(tuple(x.shape[-3:]), COLLAR_WIDTH, device)
    fixed_norm = masked_zscore(y, mask)
    moving_norm = masked_zscore(x, mask)
    fixed_mind = mind_ssc(fixed_norm, radius=1, dilation=2)
    moving_mind = mind_ssc(moving_norm, radius=1, dilation=2)
    baseline_geometry = _deformation_quality_metrics(initial, exact_certified=True)
    baseline_mind = _candidate_metrics(
        initial,
        fixed_norm,
        moving_norm,
        fixed_mind,
        moving_mind,
        mask,
        torch.ones_like(mask, dtype=initial.dtype),
    )["mind"]

    rows: list[dict[str, Any]] = []
    returned_cpu: list[torch.Tensor] = []
    candidate_cpu: list[torch.Tensor | None] = []
    initial_cpu = initial.detach().cpu()
    for spec in TRAJECTORIES:
        trajectory_id = spec["trajectory_id"]
        current = initial.clone()
        current_cpu = initial_cpu
        current_npz_sha = initial_report["psi_npz_sha256"]
        current_array_sha = initial_exact["sha256"]
        stopped_reason: str | None = None
        for step_index, work_eps in enumerate(MARGIN_SCHEDULE, start=1):
            current_fast = float(trilinear_cert_bound(current, eps=work_eps))
            current_metrics = _candidate_metrics(
                current,
                fixed_norm,
                moving_norm,
                fixed_mind,
                moving_mind,
                mask,
                torch.ones_like(mask, dtype=current.dtype),
            )
            current_geometry = _deformation_quality_metrics(current, exact_certified=True)
            base = {
                "case_id": case_id,
                "trajectory_id": trajectory_id,
                "step": step_index,
                "scale": spec["scale"],
                "smoothing_passes": spec["smoothing_passes"],
                "work_eps": work_eps,
                "claim_eps": CLAIM_EPS,
                "current_fast_cert_bound": current_fast,
                "current_mind": current_metrics["mind"],
                "current_ncc9": current_metrics["ncc9"],
                "current_ncc7": current_metrics["ncc7"],
                "current_sdlogj": current_geometry["sdlogj"],
                "labels_used_for_decision": False,
            }
            if stopped_reason is not None:
                row = {
                    **base,
                    "action": "STOPPED",
                    "reason": stopped_reason,
                    "proposal_built": False,
                    "candidate_exact_status": "NOT_ATTEMPTED_STOPPED",
                    "candidate_exact_certified": False,
                    "mind_improved": False,
                    "sdlogj_cap_passed": None,
                    "returned_npz_sha256": current_npz_sha,
                    "returned_array_sha256": current_array_sha,
                    "returned_exact_status": "CERTIFIED",
                }
                rows.append(row)
                candidate_cpu.append(None)
                returned_cpu.append(current_cpu)
                continue
            if not math.isfinite(current_fast) or current_fast < work_eps:
                stopped_reason = "WORK_MARGIN_EXHAUSTED"
                row = {
                    **base,
                    "action": "MARGIN_EXHAUSTED",
                    "reason": stopped_reason,
                    "proposal_built": False,
                    "candidate_exact_status": "NOT_ATTEMPTED_MARGIN",
                    "candidate_exact_certified": False,
                    "mind_improved": False,
                    "sdlogj_cap_passed": None,
                    "returned_npz_sha256": current_npz_sha,
                    "returned_array_sha256": current_array_sha,
                    "returned_exact_status": "CERTIFIED",
                }
                rows.append(row)
                candidate_cpu.append(None)
                returned_cpu.append(current_cpu)
                continue

            proposal = build_proposal(
                y,
                x,
                current,
                mask,
                feature="mind",
                orientation="target_centered",
                fixed_feature_override=fixed_mind,
                moving_feature_override=moving_mind,
            )
            displacement = proposal.displacement
            if int(spec["smoothing_passes"]) > 1:
                displacement = identity_collar(
                    smooth_proposal(displacement, passes=int(spec["smoothing_passes"]) - 1),
                    width=COLLAR_WIDTH,
                )
            requested = float(spec["scale"]) * displacement
            support = proposal_support_weights(requested, mask)
            current_metrics = _candidate_metrics(
                current, fixed_norm, moving_norm, fixed_mind, moving_mind, mask, support
            )
            proposal_stats = _proposal_statistics(proposal, requested, mask)
            candidate, operator_report = certified_local_clip_candidate(
                current, requested, mask, work_eps=work_eps, sweeps=1
            )
            candidate_fast = float(trilinear_cert_bound(candidate, eps=CLAIM_EPS))
            candidate_path = work / trajectory_id / f"step_{step_index:02d}.npz"
            evaluated, exact = _exact_materialize(candidate, candidate_path, device)
            candidate_metrics = _candidate_metrics(
                evaluated, fixed_norm, moving_norm, fixed_mind, moving_mind, mask, support
            )
            candidate_geometry = _deformation_quality_metrics(evaluated, exact_certified=bool(exact["certified"]))
            candidate_snapshot = evaluated.detach().cpu()
            mind_improvement, mind_tolerance, mind_improved = _relative_improvement(
                current_metrics["mind"], candidate_metrics["mind"]
            )
            cap = spec["sdlogj_relative_cap"]
            cap_limit = baseline_geometry["sdlogj"] * (1.0 + float(cap)) if cap is not None else None
            cap_passed = candidate_geometry["sdlogj"] <= cap_limit if cap_limit is not None else True
            accepted = bool(exact["certified"] and mind_improved and cap_passed)
            if accepted:
                current = evaluated
                current_cpu = candidate_snapshot
                current_npz_sha = exact["npz_sha256"]
                current_array_sha = exact["array_sha256"]
                action = "ACCEPT"
                reason = "EXACT_MIND_AND_GEOMETRY_POLICY_PASS"
            else:
                action = "ROLLBACK"
                reason = (
                    "EXACT_PREDICATE_REJECT"
                    if not exact["certified"]
                    else "MIND_NOT_IMPROVED"
                    if not mind_improved
                    else "CUMULATIVE_SDLOGJ_CAP_EXCEEDED"
                )
                stopped_reason = reason
            row = {
                **base,
                "action": action,
                "reason": reason,
                "proposal_built": True,
                **{f"proposal_{key}": value for key, value in proposal_stats.items()},
                **{f"operator_{key}": value for key, value in operator_report.items()},
                "candidate_fast_cert_bound": candidate_fast,
                "candidate_exact_status": exact["status"],
                "candidate_exact_certified": exact["certified"],
                "candidate_npz_sha256": exact["npz_sha256"],
                "candidate_array_sha256": exact["array_sha256"],
                "candidate_mind": candidate_metrics["mind"],
                "candidate_ncc9": candidate_metrics["ncc9"],
                "candidate_ncc7": candidate_metrics["ncc7"],
                "candidate_support_ncc9": candidate_metrics["support_ncc9"],
                "mind_improvement": mind_improvement,
                "mind_tolerance": mind_tolerance,
                "mind_improved": mind_improved,
                "candidate_sdlogj": candidate_geometry["sdlogj"],
                "candidate_j_leq0_central_percent": candidate_geometry["j_leq0_central_percent"],
                "candidate_j_leq0_digital10_percent": candidate_geometry["j_leq0_digital10_percent"],
                "candidate_trilinear_fold_percent_upper_bound": candidate_geometry[
                    "trilinear_fold_percent_upper_bound"
                ],
                "sdlogj_cap_relative": cap,
                "sdlogj_cap_limit": cap_limit,
                "sdlogj_cap_passed": cap_passed,
                "cumulative_mind_improvement": baseline_mind
                - (candidate_metrics["mind"] if accepted else current_metrics["mind"]),
                "returned_npz_sha256": current_npz_sha,
                "returned_array_sha256": current_array_sha,
                "returned_exact_status": "CERTIFIED",
            }
            rows.append(row)
            candidate_cpu.append(candidate_snapshot)
            returned_cpu.append(current_cpu)
    decision_rows = [_row_without_labels(row) for row in rows]
    decision_inputs = {
        "schema": f"ctcf-search-c2-decision-inputs-{SCHEMA_VERSION}",
        "case_id": case_id,
        "contract_sha256": contract_sha,
        "policy_sha256": C2_POLICY_SHA256,
        "labels_loaded_to_device": False,
        "rows": decision_rows,
    }
    decision_dir = case_dir / "decision"
    atomic_write_json(decision_dir / "decision_inputs.json", decision_inputs)
    decision_hash = sha256_file(decision_dir / "decision_inputs.json")
    atomic_write_json(
        decision_dir / "decisions.json",
        {
            "schema": f"ctcf-search-c2-decisions-{SCHEMA_VERSION}",
            "case_id": case_id,
            "contract_sha256": contract_sha,
            "decision_inputs_sha256": decision_hash,
            "labels_loaded_to_device": False,
            "actions": [{key: row.get(key) for key in ("trajectory_id", "step", "action", "reason")} for row in rows],
        },
    )

    moving_seg = moving_seg_cpu.unsqueeze(0).to(device)
    fixed_seg = fixed_seg_cpu.unsqueeze(0).to(device)
    baseline_dice = _dice(initial, moving_seg, fixed_seg, labels)
    for row, candidate, returned in zip(rows, candidate_cpu, returned_cpu, strict=True):
        if candidate is not None:
            candidate_dice = _dice(candidate.to(device), moving_seg, fixed_seg, labels)
            row["candidate_dice"] = candidate_dice
            row["candidate_dice_delta"] = candidate_dice - baseline_dice
        else:
            row["candidate_dice"] = None
            row["candidate_dice_delta"] = None
        returned_gpu = returned.to(device)
        returned_dice = _dice(returned_gpu, moving_seg, fixed_seg, labels)
        returned_geometry = _deformation_quality_metrics(returned_gpu, exact_certified=True)
        row["baseline_dice"] = baseline_dice
        row["returned_dice"] = returned_dice
        row["returned_dice_delta"] = returned_dice - baseline_dice
        row["returned_sdlogj"] = returned_geometry["sdlogj"]
        row["returned_sdlogj_delta"] = returned_geometry["sdlogj"] - baseline_geometry["sdlogj"]
        row["returned_j_leq0_central_percent"] = returned_geometry["j_leq0_central_percent"]
        row["returned_j_leq0_digital10_percent"] = returned_geometry["j_leq0_digital10_percent"]
        row["returned_trilinear_fold_percent_upper_bound"] = returned_geometry["trilinear_fold_percent_upper_bound"]

    branch_rows = _build_branch_rows(rows, c1_row)
    _sync(device)
    payload = {
        "schema": f"ctcf-search-c2-case-{SCHEMA_VERSION}",
        "status": "COMPLETE",
        "case_id": case_id,
        "contract_sha256": contract_sha,
        "input": input_row,
        "c1_reference": c1_row,
        "initial": initial_report,
        "decision_inputs_sha256": decision_hash,
        "decisions_sha256": sha256_file(decision_dir / "decisions.json"),
        "execution": execution,
        "elapsed_sec": time.perf_counter() - started,
        "peak_gpu_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0,
        "rows": rows,
        "branch_rows": branch_rows,
    }
    atomic_write_json(marker, payload)
    shutil.rmtree(work, ignore_errors=True)
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return rows


def _validate_case(
    payload: dict[str, Any],
    marker: Path,
    case_id: str,
    contract_sha: str,
    input_row: dict[str, Any],
    contract: dict[str, Any] | None = None,
    c1_row: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    rows = payload.get("rows")
    branch_rows = payload.get("branch_rows")
    expected_pairs = [(spec["trajectory_id"], step) for spec in TRAJECTORIES for step in range(1, MAX_STEPS + 1)]
    if (
        payload.get("schema") != f"ctcf-search-c2-case-{SCHEMA_VERSION}"
        or payload.get("status") != "COMPLETE"
        or payload.get("case_id") != case_id
        or payload.get("contract_sha256") != contract_sha
        or payload.get("input") != input_row
        or not isinstance(rows, list)
        or [(row.get("trajectory_id"), row.get("step")) for row in rows] != expected_pairs
        or not isinstance(branch_rows, list)
        or [row.get("trajectory_id") for row in branch_rows] != [spec["trajectory_id"] for spec in TRAJECTORIES]
        or any(row.get("labels_used_for_decision") is not False for row in rows)
        or any(row.get("returned_exact_status") != "CERTIFIED" for row in rows)
        or any(not _is_finite_number(row.get("baseline_dice")) for row in rows)
        or any(not _is_finite_number(row.get("returned_dice")) for row in rows)
        or any(not _is_finite_number(row.get("returned_sdlogj")) for row in rows)
        or any(not _is_finite_number(row.get("returned_j_leq0_digital10_percent")) for row in rows)
        or any(row.get("returned_trilinear_fold_percent_upper_bound") != 0.0 for row in rows)
        or any(not SHA256_RE.fullmatch(str(row.get("returned_array_sha256", ""))) for row in rows)
    ):
        raise RuntimeError(f"Invalid C2 case marker: {marker}")
    initial = payload.get("initial") or {}
    previous_by_trajectory = {
        spec["trajectory_id"]: (initial.get("psi_npz_sha256"), (initial.get("psi_exact") or {}).get("sha256"))
        for spec in TRAJECTORIES
    }
    stopped = {spec["trajectory_id"]: False for spec in TRAJECTORIES}
    specs = {spec["trajectory_id"]: spec for spec in TRAJECTORIES}
    initial_sdlogj = {
        trajectory_id: float(next(row for row in rows if row["trajectory_id"] == trajectory_id)["current_sdlogj"])
        for trajectory_id in specs
    }
    for row in rows:
        trajectory_id = row["trajectory_id"]
        action = row.get("action")
        previous_npz, previous_array = previous_by_trajectory[trajectory_id]
        if row.get("proposal_built") is True:
            improvement, tolerance, improved = _relative_improvement(
                float(row["current_mind"]), float(row["candidate_mind"])
            )
            cap = specs[trajectory_id]["sdlogj_relative_cap"]
            cap_limit = initial_sdlogj[trajectory_id] * (1.0 + float(cap)) if cap is not None else None
            cap_passed = float(row["candidate_sdlogj"]) <= cap_limit if cap_limit is not None else True
            if (
                not all(
                    _is_finite_number(row.get(key))
                    for key in (
                        "current_fast_cert_bound",
                        "candidate_fast_cert_bound",
                        "current_mind",
                        "candidate_mind",
                        "candidate_sdlogj",
                        "candidate_dice",
                        "candidate_dice_delta",
                    )
                )
                or float(row["current_fast_cert_bound"]) < float(row["work_eps"])
                or row.get("mind_improvement") != improvement
                or row.get("mind_tolerance") != tolerance
                or row.get("mind_improved") is not improved
                or row.get("sdlogj_cap_limit") != cap_limit
                or row.get("sdlogj_cap_passed") is not cap_passed
                or not SHA256_RE.fullmatch(str(row.get("candidate_npz_sha256", "")))
                or not SHA256_RE.fullmatch(str(row.get("candidate_array_sha256", "")))
                or (row.get("candidate_exact_status") == "CERTIFIED")
                is not (row.get("candidate_exact_certified") is True)
            ):
                raise RuntimeError(f"Invalid C2 candidate decision scalars in {marker}: {trajectory_id}")
        if action == "ACCEPT":
            valid = (
                not stopped[trajectory_id]
                and row.get("proposal_built") is True
                and row.get("candidate_exact_certified") is True
                and row.get("candidate_exact_status") == "CERTIFIED"
                and row.get("mind_improved") is True
                and row.get("sdlogj_cap_passed") is True
                and row.get("returned_npz_sha256") == row.get("candidate_npz_sha256")
                and row.get("returned_array_sha256") == row.get("candidate_array_sha256")
            )
            previous_by_trajectory[trajectory_id] = (
                row.get("returned_npz_sha256"),
                row.get("returned_array_sha256"),
            )
        elif action in {"ROLLBACK", "MARGIN_EXHAUSTED", "STOPPED"}:
            valid = (
                (action == "STOPPED") is stopped[trajectory_id]
                and row.get("returned_npz_sha256") == previous_npz
                and row.get("returned_array_sha256") == previous_array
            )
            if action in {"ROLLBACK", "MARGIN_EXHAUSTED"}:
                stopped[trajectory_id] = True
            if action == "ROLLBACK":
                valid = (
                    valid
                    and row.get("proposal_built") is True
                    and not (
                        row.get("candidate_exact_certified") is True
                        and row.get("mind_improved") is True
                        and row.get("sdlogj_cap_passed") is True
                    )
                )
            else:
                valid = valid and row.get("proposal_built") is False
            if action == "MARGIN_EXHAUSTED" and not (
                _is_finite_number(row.get("current_fast_cert_bound"))
                and float(row["current_fast_cert_bound"]) < float(row["work_eps"])
            ):
                valid = False
        else:
            valid = False
        if (
            specs[trajectory_id]["sdlogj_relative_cap"] is None
            and row.get("proposal_built") is True
            and row.get("sdlogj_cap_passed") is not True
        ):
            valid = False
        if not valid:
            raise RuntimeError(f"Invalid C2 transaction chain in {marker}: {trajectory_id}/step={row.get('step')}")
    decision_dir = marker.parent / "decision"
    for name, key in (("decision_inputs.json", "decision_inputs_sha256"), ("decisions.json", "decisions_sha256")):
        path = decision_dir / name
        if not path.is_file() or sha256_file(path) != payload.get(key):
            raise RuntimeError(f"C2 decision snapshot changed: {path}")
    decision_inputs = json.loads((decision_dir / "decision_inputs.json").read_text(encoding="utf-8"))
    if decision_inputs.get("rows") != [_row_without_labels(row) for row in rows]:
        raise RuntimeError(f"C2 decision inputs do not reconstruct from case rows: {marker}")
    decisions = json.loads((decision_dir / "decisions.json").read_text(encoding="utf-8"))
    expected_actions = [{key: row.get(key) for key in ("trajectory_id", "step", "action", "reason")} for row in rows]
    if (
        decision_inputs.get("labels_loaded_to_device") is not False
        or decision_inputs.get("contract_sha256") != contract_sha
        or decisions.get("labels_loaded_to_device") is not False
        or decisions.get("contract_sha256") != contract_sha
        or decisions.get("decision_inputs_sha256") != payload.get("decision_inputs_sha256")
        or decisions.get("actions") != expected_actions
    ):
        raise RuntimeError(f"Invalid C2 decision snapshot semantics: {marker}")
    execution = payload.get("execution") or {}
    load = execution.get("checkpoint_load_report") or {}
    if (
        not str(execution.get("device", "")).startswith("cuda")
        or execution.get("deterministic") is not True
        or load.get("strict") is not True
        or set(load.get("missing_keys") or []) != set(load.get("allowed_missing_buffers") or [])
        or bool(load.get("unexpected_keys"))
    ):
        raise RuntimeError(f"Invalid C2 case execution provenance: {marker}")
    if contract is not None:
        expected_shard = next(
            index for index in range(contract["num_shards"]) if case_id in contract["shards"][str(index)]
        )
        if (
            execution.get("shard_index") != expected_shard
            or execution.get("physical_gpu") != contract["shard_to_physical_gpu"][str(expected_shard)]
            or execution.get("checkpoint_sha256") != contract["checkpoint_sha256"]
            or execution.get("seed") != contract["seed"]
        ):
            raise RuntimeError(f"C2 case execution differs from frozen contract: {marker}")
    if c1_row is not None and (
        payload.get("c1_reference") != c1_row or payload.get("branch_rows") != _build_branch_rows(rows, c1_row)
    ):
        raise RuntimeError(f"C2 branch summary or C1 reference changed: {marker}")
    return rows


def worker_stage(args: argparse.Namespace) -> int:
    root = args.run_root.resolve()
    contract, contract_sha = _load_contract(root, args.contract_sha256)
    if _git("rev-parse", "HEAD") != contract["git_head"] or _git("status", "--porcelain=v1"):
        raise RuntimeError("C2 worker code differs from the clean prepared contract")
    if args.num_shards != contract["num_shards"] or not 0 <= args.shard_index < args.num_shards:
        raise RuntimeError("Worker shard parameters differ from C2 contract")
    if args.physical_gpu != contract["shard_to_physical_gpu"][str(args.shard_index)]:
        raise RuntimeError("Worker physical GPU differs from C2 contract")
    rows, refs = _read_frozen_rows(root, contract)
    by_id = {row["case_id"]: row for row in rows}
    assigned = contract["shards"][str(args.shard_index)]
    for case_id in [*assigned, "atlas"]:
        _verify_file(by_id[case_id])
    checkpoint = Path(contract["checkpoint"])
    if sha256_file(checkpoint) != contract["checkpoint_sha256"]:
        raise RuntimeError("Checkpoint changed after C2 prepare")
    attempt_dir = root / "workers" / "attempts" / args.attempt_id
    marker = attempt_dir / f"worker_{args.shard_index:02d}.json"
    failure = attempt_dir / f"worker_{args.shard_index:02d}_failure.json"
    if marker.exists() or failure.exists():
        raise RuntimeError("Worker attempt output already exists")
    pending: list[dict[str, str]] = []
    reused: list[str] = []
    for case_id in assigned:
        case_marker = root / "cases" / case_id / "case_complete.json"
        if case_marker.is_file():
            _validate_case(
                json.loads(case_marker.read_text(encoding="utf-8")),
                case_marker,
                case_id,
                contract_sha,
                by_id[case_id],
                contract,
                refs[case_id],
            )
            reused.append(case_id)
        else:
            pending.append(by_id[case_id])
    started = _utc_now()
    computed: list[str] = []
    try:
        load_report: dict[str, Any] = {"strict": None, "missing_keys": [], "unexpected_keys": []}
        if pending:
            device = setup_device(args.gpu, seed=contract["seed"], deterministic=True)
            if device.type != "cuda":
                raise RuntimeError("C2 worker requires CUDA")
            adapter, model, load_report = _build_model(str(checkpoint), contract["config"], device)
            dataset = build_infer_dataset("IXI", [row["path"] for row in pending], by_id["atlas"]["path"])
            labels = tuple(metric_profile_for("IXI").labels)
            execution = {
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
                "checkpoint_sha256": contract["checkpoint_sha256"],
                "checkpoint_load_report": load_report,
            }
            for index, input_row in enumerate(pending):
                print(
                    f"[shard {args.shard_index + 1}/{args.num_shards}] "
                    f"[{index + 1}/{len(pending)}] IXI {input_row['case_id']}",
                    flush=True,
                )
                _run_case(
                    index=index,
                    input_row=input_row,
                    c1_row=refs[input_row["case_id"]],
                    dataset=dataset,
                    adapter=adapter,
                    model=model,
                    device=device,
                    labels=labels,
                    root=root,
                    contract_sha=contract_sha,
                    execution=execution,
                )
                computed.append(input_row["case_id"])
        report = {
            "schema": f"ctcf-search-c2-worker-{SCHEMA_VERSION}",
            "status": "COMPLETE",
            "attempt_id": args.attempt_id,
            "shard_index": args.shard_index,
            "contract_sha256": contract_sha,
            "assigned_case_ids": assigned,
            "computed_case_ids": computed,
            "reused_case_ids": reused,
            "checkpoint_load_report": load_report,
            "started_at_utc": started,
            "completed_at_utc": _utc_now(),
        }
        atomic_write_json(marker, report)
    except Exception as exc:
        atomic_write_json(
            failure,
            {
                "schema": f"ctcf-search-c2-worker-failure-{SCHEMA_VERSION}",
                "status": "FAILED",
                "attempt_id": args.attempt_id,
                "shard_index": args.shard_index,
                "contract_sha256": contract_sha,
                "computed_case_ids": computed,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "completed_at_utc": _utc_now(),
            },
        )
        raise
    return 0


def _summary_rows(
    branch_rows: list[dict[str, Any]], c1_rows: list[dict[str, str]]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    c1_dice_delta = np.array([float(row["dice_delta"]) for row in c1_rows], dtype=np.float64)
    c1_sdlogj_delta = np.array([float(row["sdlogj_delta"]) for row in c1_rows], dtype=np.float64)
    c1_digital = np.array([float(row["final_digital10_percent"]) for row in c1_rows], dtype=np.float64)
    summaries: list[dict[str, Any]] = []
    for spec in TRAJECTORIES:
        trajectory_id = spec["trajectory_id"]
        selected = [row for row in branch_rows if row["trajectory_id"] == trajectory_id]
        if len(selected) != 58:
            raise RuntimeError(f"Trajectory {trajectory_id} does not cover 58 cases")
        dice_delta = np.array([float(row["dice_delta"]) for row in selected], dtype=np.float64)
        paired = np.array([float(row["paired_dice_advantage_vs_c1"]) for row in selected], dtype=np.float64)
        final_dice = np.array([float(row["final_dice"]) for row in selected], dtype=np.float64)
        final_sdlogj = np.array([float(row["final_sdlogj"]) for row in selected], dtype=np.float64)
        sdlogj_delta = np.array([float(row["sdlogj_delta"]) for row in selected], dtype=np.float64)
        digital = np.array([float(row["final_j_leq0_digital10_percent"]) for row in selected], dtype=np.float64)
        ci = _bootstrap_ci(paired)
        eligible = bool(
            all(row["final_exact_status"] == "CERTIFIED" for row in selected)
            and float(dice_delta.mean()) >= MIN_MEAN_DICE_DELTA
            and float(paired.mean()) > 0.0
            and float(np.median(paired)) > 0.0
            and float(ci["low"]) > 0.0
            and float(sdlogj_delta.mean()) <= float(c1_sdlogj_delta.mean())
            and float(digital.mean()) <= float(c1_digital.mean())
        )
        summaries.append(
            {
                "trajectory_id": trajectory_id,
                "eligible_for_test_gate": eligible,
                "accepted_steps_total": sum(int(row["accepted_steps"]) for row in selected),
                "cases_with_any_step": sum(int(row["accepted_steps"]) > 0 for row in selected),
                "final_dice_mean": float(final_dice.mean()),
                "final_dice_median": float(np.median(final_dice)),
                "dice_delta_mean": float(dice_delta.mean()),
                "dice_delta_median": float(np.median(dice_delta)),
                "dice_delta_improved": int((dice_delta > 0).sum()),
                "dice_delta_worsened": int((dice_delta < 0).sum()),
                "paired_advantage_vs_c1_mean": float(paired.mean()),
                "paired_advantage_vs_c1_median": float(np.median(paired)),
                "paired_advantage_vs_c1_ci95_low": ci["low"],
                "paired_advantage_vs_c1_ci95_high": ci["high"],
                "final_sdlogj_mean": float(final_sdlogj.mean()),
                "sdlogj_delta_mean": float(sdlogj_delta.mean()),
                "final_digital10_percent_mean": float(digital.mean()),
                "final_digital10_percent_max": float(digital.max()),
                "final_trilinear_fold_percent_upper_bound": 0.0,
            }
        )
    eligible = [row for row in summaries if row["eligible_for_test_gate"]]
    selected = (
        sorted(eligible, key=lambda row: (-row["final_dice_mean"], row["trajectory_id"]))[0] if eligible else None
    )
    summary = {
        "schema": f"ctcf-search-c2-summary-{SCHEMA_VERSION}",
        "protocol_id": PROTOCOL_ID,
        "execution_integrity_status": "PASS",
        "scientific_status": "C2_PROMISING" if selected else "C2_NOT_PROMISING",
        "n_cases": 58,
        "n_trajectories": len(TRAJECTORIES),
        "n_step_rows": 58 * len(TRAJECTORIES) * MAX_STEPS,
        "selected_trajectory_id": selected["trajectory_id"] if selected else None,
        "selection_rule": C2_POLICY["selection_gate"],
        "c1_reference": {
            "dice_delta": _sign_summary(c1_dice_delta),
            "sdlogj_delta": _distribution_summary(c1_sdlogj_delta),
            "final_digital10_percent": _distribution_summary(c1_digital),
        },
        "test_split_accessed": False,
        "test_115_authorized": False,
        "labels_used_for_transaction_decision": False,
    }
    return summaries, summary


def _csv_fields(rows: list[dict[str, Any]], preferred: list[str]) -> list[str]:
    keys = {key for row in rows for key in row}
    return [key for key in preferred if key in keys] + sorted(keys - set(preferred))


def finalize_stage(args: argparse.Namespace) -> int:
    root = args.run_root.resolve()
    contract, contract_sha = _load_contract(root, args.contract_sha256)
    if _git("rev-parse", "HEAD") != contract["git_head"] or _git("status", "--porcelain=v1"):
        raise RuntimeError("C2 finalize code differs from clean prepared contract")
    dataset_rows, refs = _read_frozen_rows(root, contract)
    for row in dataset_rows:
        _verify_file(row)
    for stage, source in contract["c1_sources"].items():
        if _validate_c1_stage(source["path"], source["sha256"], stage) != source:
            raise RuntimeError(f"Frozen C1 {stage} source changed")
    if sha256_file(Path(contract["checkpoint"])) != contract["checkpoint_sha256"]:
        raise RuntimeError("Checkpoint changed before C2 finalize")
    worker_files: list[dict[str, Any]] = []
    seen: list[str] = []
    for shard in range(contract["num_shards"]):
        path = root / "workers" / "attempts" / args.attempt_id / f"worker_{shard:02d}.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        assigned = contract["shards"][str(shard)]
        computed = payload.get("computed_case_ids") or []
        reused = payload.get("reused_case_ids") or []
        if (
            payload.get("schema") != f"ctcf-search-c2-worker-{SCHEMA_VERSION}"
            or payload.get("status") != "COMPLETE"
            or payload.get("attempt_id") != args.attempt_id
            or payload.get("shard_index") != shard
            or payload.get("contract_sha256") != contract_sha
            or payload.get("assigned_case_ids") != assigned
            or len(computed) + len(reused) != len(assigned)
            or len(set([*computed, *reused])) != len(assigned)
            or sorted([*computed, *reused]) != sorted(assigned)
            or (
                computed
                and (
                    (payload.get("checkpoint_load_report") or {}).get("strict") is not True
                    or bool((payload.get("checkpoint_load_report") or {}).get("unexpected_keys"))
                )
            )
        ):
            raise RuntimeError(f"Invalid C2 worker report: {path}")
        seen.extend(assigned)
        worker_files.append({"path": path.relative_to(root).as_posix(), "sha256": sha256_file(path)})
    if set(seen) != set(contract["case_ids"]) or len(seen) != 58 or len(set(seen)) != 58:
        raise RuntimeError("C2 worker partition has missing, duplicate, or reordered cases")

    step_rows: list[dict[str, Any]] = []
    branch_rows: list[dict[str, Any]] = []
    case_hashes: dict[str, str] = {}
    by_input = {row["case_id"]: row for row in dataset_rows if row["split"] == "val"}
    for case_id in contract["case_ids"]:
        path = root / "cases" / case_id / "case_complete.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        step_rows.extend(
            _validate_case(payload, path, case_id, contract_sha, by_input[case_id], contract, refs[case_id])
        )
        branch_rows.extend(payload["branch_rows"])
        case_hashes[case_id] = sha256_file(path)
    summary_rows, summary = _summary_rows(branch_rows, [refs[case_id] for case_id in contract["case_ids"]])
    step_path = root / "per_step.csv"
    branch_path = root / "per_branch.csv"
    summary_csv_path = root / "trajectory_summary.csv"
    summary_path = root / "summary.json"
    atomic_write_text(
        step_path,
        rows_to_csv(_csv_fields(step_rows, ["case_id", "trajectory_id", "step", "action", "reason"]), step_rows),
    )
    atomic_write_text(
        branch_path,
        rows_to_csv(
            _csv_fields(branch_rows, ["case_id", "trajectory_id", "accepted_steps", "terminal_action"]),
            branch_rows,
        ),
    )
    atomic_write_text(summary_csv_path, rows_to_csv(list(summary_rows[0].keys()), summary_rows))
    atomic_write_json(summary_path, summary)
    prepare = json.loads((root / "prepare.json").read_text(encoding="utf-8"))
    manifest = {
        "schema": f"ctcf-search-c2-run-manifest-{SCHEMA_VERSION}",
        "protocol_id": PROTOCOL_ID,
        "run_id": root.name,
        "status": "COMPLETE",
        "started_at_utc": prepare["prepared_at_utc"],
        "completed_at_utc": _utc_now(),
        "contract_sha256": contract_sha,
        "policy_sha256": C2_POLICY_SHA256,
        "finalize_attempt_id": args.attempt_id,
        "code": {"git_head": _git("rev-parse", "HEAD"), "branch": _git("branch", "--show-current"), "git_status": ""},
        "checkpoint": {"path": contract["checkpoint"], "sha256": contract["checkpoint_sha256"], "strict": True},
        "execution": {
            "num_shards": contract["num_shards"],
            "physical_gpus": contract["physical_gpus"],
            "seed": contract["seed"],
            "paths_profile": contract["paths_profile"],
            "time_steps": contract["time_steps"],
        },
        "c1_sources": contract["c1_sources"],
        "workers": worker_files,
        "case_marker_sha256": case_hashes,
        "files": {
            "contract_sha256": sha256_file(root / "c2_contract.json"),
            "datasets_sha256": sha256_file(root / "datasets.csv"),
            "c1_reference_sha256": sha256_file(root / "c1_reference.csv"),
            "per_step_sha256": sha256_file(step_path),
            "per_branch_sha256": sha256_file(branch_path),
            "trajectory_summary_sha256": sha256_file(summary_csv_path),
            "summary_sha256": sha256_file(summary_path),
        },
        "summary": summary,
        "storage": {"compact_outputs_only": True, "heavy_fields_retained": False},
    }
    atomic_write_json(root / "c2_manifest.json", manifest)
    print(json.dumps(summary, indent=2))
    return 0


def selfcheck_stage(args: argparse.Namespace) -> int:
    checks = {
        "four_unique_trajectories": len(TRAJECTORIES) == len({x["trajectory_id"] for x in TRAJECTORIES}) == 4,
        "four_strictly_decreasing_work_margins": len(MARGIN_SCHEDULE) == 4
        and all(left > right > CLAIM_EPS for left, right in pairwise(MARGIN_SCHEDULE)),
        "test_split_is_unreachable": C2_POLICY["scope"].startswith("all 58 already-open IXI validation"),
        "policy_hash_is_canonical": _payload_sha256(C2_POLICY) == C2_POLICY_SHA256,
        "regularized_branch_is_unique": sum(x["sdlogj_relative_cap"] is not None for x in TRAJECTORIES) == 1,
        "minimum_effect_is_preregistered": MIN_MEAN_DICE_DELTA == 0.001,
    }
    failed = [key for key, value in checks.items() if not value]
    payload = {
        "schema": f"ctcf-search-c2-selfcheck-{SCHEMA_VERSION}",
        "protocol_id": PROTOCOL_ID,
        "status": "PASS" if not failed else "FAIL",
        "checks": checks,
        "failed": failed,
        "policy_sha256": C2_POLICY_SHA256,
    }
    atomic_write_json(args.output, payload)
    if failed:
        raise RuntimeError(f"C2 self-check failed: {failed}")
    print(json.dumps(payload, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run frozen sequential transactional search Gate C2 on IXI val-58.")
    sub = parser.add_subparsers(dest="action", required=True)
    selfcheck = sub.add_parser("selfcheck")
    selfcheck.add_argument("--output", type=Path, required=True)
    prepare = sub.add_parser("prepare")
    prepare.add_argument("--run-root", type=Path, required=True)
    prepare.add_argument("--paths-profile", type=int, default=3)
    prepare.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    prepare.add_argument("--seed", type=int, default=0)
    prepare.add_argument("--num-shards", type=int, required=True)
    prepare.add_argument("--physical-gpus", required=True)
    prepare.add_argument("--c1-exploration-manifest", required=True)
    prepare.add_argument("--c1-exploration-sha256", required=True)
    prepare.add_argument("--c1-confirmation-manifest", required=True)
    prepare.add_argument("--c1-confirmation-sha256", required=True)
    worker = sub.add_parser("worker")
    worker.add_argument("--run-root", type=Path, required=True)
    worker.add_argument("--contract-sha256", required=True)
    worker.add_argument("--shard-index", type=int, required=True)
    worker.add_argument("--num-shards", type=int, required=True)
    worker.add_argument("--gpu", type=int, default=0)
    worker.add_argument("--physical-gpu", required=True)
    worker.add_argument("--attempt-id", required=True)
    finalize = sub.add_parser("finalize")
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
    raise AssertionError(args.action)


if __name__ == "__main__":
    raise SystemExit(main())
