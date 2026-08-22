from __future__ import annotations

import argparse
import glob
import json
import os
import platform
import time
from contextlib import suppress
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from experiments.core.inference_metrics import metric_profile_for
from experiments.core.inference_runtime import build_infer_dataset
from experiments.core.path_profiles import get_dataset_paths
from tools.analysis.run_artifacts import (
    atomic_write_json,
    atomic_write_text,
    rows_to_csv,
    rows_to_tsv,
    sha256_file,
)
from tools.analysis.search_gate_common import (
    CLAIM_EPS,
    COLLAR_WIDTH,
    IXI_DEVELOPMENT_CASES,
    TIME_STEPS,
    WORK_EPS,
    build_model,
    case_id_from_path,
    dice_score,
    git,
    prepare_initial_state,
    proposal_statistics,
    salted_case_hash,
    utc_now,
)
from tools.analysis.search_gate_runtime import dataset_rows
from tools.analysis.transactional_search import (
    OFFSETS,
    ZERO_OFFSET_INDEX,
    ProposalResult,
    build_proposal,
    commit_exact_candidate,
    geometry_mask,
    load_flow_npz,
    masked_zscore,
    mind_distance_from_features,
    mind_ssc,
    screen_candidates,
    utility_loss,
)
from utils import setup_device
from utils.field import (
    trilinear_cert_bound,
)

PROTOCOL_ID = "CTCF-SEARCH-GATE-C0-V1"
BRANCH_ORDER = ("mind_soft", "mind_hard", "intensity_soft", "mind_reversed")


def _select_cases(stage: str, paths_profile: int) -> tuple[str, str, list[str], str | None]:
    dataset = "OASIS" if stage == "smoke" else "IXI"
    paths = get_dataset_paths(paths_profile, dataset)
    split = "val"
    root = paths["val_dir"]
    all_files = sorted(glob.glob(os.path.join(root, "*.pkl")))
    if not all_files:
        raise FileNotFoundError(f"No {dataset}/{split} .pkl files found under {root}")
    by_id = {case_id_from_path(path): path for path in all_files}

    if stage == "smoke":
        ranked = sorted((salted_case_hash(case), case) for case in by_id)
        expected = "0456_0457"
        if len(ranked) != 19 or ranked[0][1] != expected:
            raise RuntimeError(f"OASIS smoke selection contract changed: n={len(ranked)}, minimum={ranked[0][1]}")
        selected_ids = [expected]
    else:
        missing = [case for case in IXI_DEVELOPMENT_CASES if case not in by_id]
        ranked = [case for _, case in sorted((salted_case_hash(case), case) for case in by_id)[:19]]
        if missing or tuple(ranked) != IXI_DEVELOPMENT_CASES:
            raise RuntimeError(f"IXI development selection contract changed: missing={missing}, ranked={ranked}")
        if len(by_id) != 58:
            raise RuntimeError(f"IXI development protocol expects 58 validation cases, found {len(by_id)}")
        selected_ids = list(IXI_DEVELOPMENT_CASES)

    atlas = str(paths["atlas_path"]) if dataset == "IXI" else None
    return dataset, split, [by_id[case] for case in selected_ids], atlas


def _checkpoint_contract(stage: str, override: str | None) -> tuple[str, str]:
    if stage == "smoke":
        default = "results/P16_W1_VXM_OASIS_LBL_DIG_J15/ckpt/best.pth"
    else:
        default = "results/P10_LONGRUN_VXM_UNIFIED_SVF_IXI/ckpt/best.pth"
    checkpoint = str(Path(override or default))
    return checkpoint, "CTCF-CascadeA-VM-Unified"


def _run_branch(
    name: str,
    proposal_result: ProposalResult,
    proposal_tensor: torch.Tensor,
    initial_psi: torch.Tensor,
    initial_path: Path,
    fixed: torch.Tensor,
    moving: torch.Tensor,
    moving_seg: torch.Tensor,
    fixed_seg: torch.Tensor,
    labels: tuple[int, ...],
    mask: torch.Tensor,
    fixed_mind: torch.Tensor,
    moving_mind: torch.Tensor,
    case_dir: Path,
) -> dict[str, Any]:
    baseline_utility = utility_loss(fixed, moving, initial_psi, mask)
    baseline_mind = mind_distance_from_features(fixed_mind, moving_mind, initial_psi, mask)
    ungated = (initial_psi + proposal_tensor).float()
    ungated_utility = utility_loss(fixed, moving, ungated, mask)
    ungated_mind = mind_distance_from_features(fixed_mind, moving_mind, ungated, mask)
    ungated_cert_bound = trilinear_cert_bound(ungated, eps=CLAIM_EPS)

    screens, eligible = screen_candidates(
        initial_psi,
        proposal_tensor,
        fixed,
        moving,
        mask,
        fast_certificate=lambda candidate: trilinear_cert_bound(candidate, eps=CLAIM_EPS),
        eps=CLAIM_EPS,
    )
    output = case_dir / "work" / f"final_{name}.npz"
    outcome = commit_exact_candidate(
        initial_path,
        output,
        eligible,
        initial_psi=initial_psi,
        proposal=proposal_tensor,
        eps=str(CLAIM_EPS),
    )
    final_psi = load_flow_npz(output).to(initial_psi.device)
    final_utility = utility_loss(fixed, moving, final_psi, mask)
    final_mind = mind_distance_from_features(fixed_mind, moving_mind, final_psi, mask)
    selected = outcome.selected
    action = "ROLLBACK"
    if selected is not None:
        action = "ACCEPT" if selected.coefficient == 1.0 else "BACKTRACK"

    # Labels are deliberately touched only after the utility/topology transaction has committed or rolled back.
    baseline_dice = dice_score(initial_psi, moving_seg, fixed_seg, labels)
    ungated_dice = dice_score(ungated, moving_seg, fixed_seg, labels)
    final_dice = dice_score(final_psi, moving_seg, fixed_seg, labels)

    report = {
        "branch": name,
        "feature": proposal_result.feature,
        "orientation": proposal_result.orientation,
        "decoder": "hard" if name == "mind_hard" else "soft",
        **proposal_statistics(proposal_result, proposal_tensor, mask),
        "baseline_utility": baseline_utility,
        "baseline_dice": baseline_dice,
        "baseline_mind_distance": baseline_mind,
        "ungated_utility": ungated_utility,
        "ungated_dice": ungated_dice,
        "ungated_mind_distance": ungated_mind,
        "ungated_cert_bound": ungated_cert_bound,
        "action": action,
        "selected_coefficient": None if selected is None else selected.coefficient,
        "accepted_utility": final_utility,
        "accepted_dice": final_dice,
        "accepted_mind_distance": final_mind,
        "accepted_npz_sha256": outcome.output_sha256,
        "accepted_array_sha256": outcome.exact_report.get("sha256"),
        "exact_status": outcome.exact_report.get("status"),
        "exact_certified": outcome.exact_report.get("certified"),
        "rollback_byte_identical": outcome.rollback_byte_identical,
        "screens": [asdict(record) for record in screens],
        "exact_report": outcome.exact_report,
    }
    atomic_write_json(case_dir / f"branch_{name}.json", report)
    return report


def _run_case(
    index: int,
    path: str,
    dataset,
    adapter,
    model,
    device: torch.device,
    labels: tuple[int, ...],
    stage_dir: Path,
    keep_fields: bool,
) -> list[dict[str, Any]]:
    case_id = case_id_from_path(path)
    case_dir = stage_dir / "cases" / case_id
    complete_marker = case_dir / "case_complete.json"
    if complete_marker.is_file():
        payload = json.loads(complete_marker.read_text(encoding="utf-8"))
        if payload.get("status") != "COMPLETE":
            raise RuntimeError(f"Invalid resume marker: {complete_marker}")
        return payload["rows"]

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    x, y, x_seg, y_seg = dataset[index]
    x, y, x_seg, y_seg = [tensor.unsqueeze(0).to(device) for tensor in (x, y, x_seg, y_seg)]
    with torch.inference_mode():
        flow = adapter.forward(model, x, y, amp=True)
    initial_psi, initial_path, initial_report = prepare_initial_state(flow, case_dir / "work")
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
    branches = {
        "mind_soft": (mind_target, mind_target.displacement),
        "mind_hard": (mind_target, mind_target.hard_displacement),
        "intensity_soft": (intensity_target, intensity_target.displacement),
        "mind_reversed": (mind_reversed, mind_reversed.displacement),
    }
    rows: list[dict[str, Any]] = []
    for name in BRANCH_ORDER:
        result, proposal = branches[name]
        branch = _run_branch(
            name,
            result,
            proposal,
            initial_psi,
            initial_path,
            y,
            x,
            x_seg,
            y_seg,
            labels,
            mask,
            fixed_mind,
            moving_mind,
            case_dir,
        )
        rows.append({"case_id": case_id, **{k: v for k, v in branch.items() if k not in {"screens", "exact_report"}}})

    elapsed = time.perf_counter() - started
    peak_bytes = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
    case_report = {
        "schema": "ctcf-search-c0-case-v1",
        "status": "COMPLETE",
        "case_id": case_id,
        "input_path": str(Path(path).resolve()),
        "input_bytes": Path(path).stat().st_size,
        "selection_sha256": salted_case_hash(case_id),
        "initial": initial_report,
        "elapsed_sec": elapsed,
        "peak_gpu_bytes": peak_bytes,
        "rows": rows,
    }
    atomic_write_json(complete_marker, case_report)
    if not keep_fields:
        for field_path in (case_dir / "work").glob("*.npz"):
            field_path.unlink()
        with suppress(OSError):
            (case_dir / "work").rmdir()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return rows


def _summarise(rows: list[dict[str, Any]], stage: str) -> dict[str, Any]:
    primary = [row for row in rows if row["branch"] == "mind_soft"]
    deltas = np.array([float(row["accepted_dice"]) - float(row["baseline_dice"]) for row in primary])
    all_exact = all(row["exact_status"] == "CERTIFIED" for row in rows)
    rollback_exact = all(row["action"] != "ROLLBACK" or bool(row["rollback_byte_identical"]) for row in rows)
    accepted_nonzero = any(row["action"] != "ROLLBACK" for row in rows)
    execution_pass = all_exact and rollback_exact
    promising = bool(stage == "development" and deltas.mean() > 0.0 and np.median(deltas) > 0.0)
    return {
        "execution_integrity_status": "PASS" if execution_pass else "FAIL",
        "search_update_status": "FOUND_UPDATE" if accepted_nonzero else "NO_ACCEPTED_UPDATE",
        "scientific_status": "PROMISING" if promising else "NOT_PROMISING",
        "n_cases": len(primary),
        "n_branch_rows": len(rows),
        "all_final_maps_exactly_certified": all_exact,
        "all_rollbacks_byte_identical": rollback_exact,
        "at_least_one_nonzero_update": accepted_nonzero,
        "primary_dice_delta_mean": float(deltas.mean()),
        "primary_dice_delta_median": float(np.median(deltas)),
        "primary_dice_delta_min": float(deltas.min()),
        "primary_dice_delta_max": float(deltas.max()),
        "primary_improved_cases": int((deltas > 0).sum()),
        "primary_worsened_cases": int((deltas < 0).sum()),
        "labels_used_for_decision": False,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run preregistered CTCF search Gate C0.2 or C0.3.")
    parser.add_argument("--stage", choices=["smoke", "development"], required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--paths-profile", type=int, default=3)
    parser.add_argument("--checkpoint")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--keep-fields", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    started_at = utc_now()
    head = git("rev-parse", "HEAD")
    stage_dir = args.run_root.resolve() / args.stage
    stage_dir.mkdir(parents=True, exist_ok=True)
    contract_path = stage_dir / "stage_contract.json"

    dataset_name, split, files, atlas = _select_cases(args.stage, args.paths_profile)
    checkpoint, config = _checkpoint_contract(args.stage, args.checkpoint)
    checkpoint_path = Path(checkpoint).resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    contract = {
        "schema": "ctcf-search-c0-stage-contract-v1",
        "protocol_id": PROTOCOL_ID,
        "stage": args.stage,
        "git_head": head,
        "dataset": dataset_name,
        "split": split,
        "case_ids": [case_id_from_path(path) for path in files],
        "selection_hashes": {case_id_from_path(path): salted_case_hash(case_id_from_path(path)) for path in files},
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "config": config,
        "time_steps": TIME_STEPS,
        "ctcf_l3_svf": True,
        "claim_eps": CLAIM_EPS,
        "work_eps": WORK_EPS,
        "collar_width": COLLAR_WIDTH,
        "feature_primary": "MIND-SSC radius=1 dilation=2",
        "mind_reference_commit": "b229e52e44b114e2040a503334c92269750c16b2",
        "decoder_primary": "softmax-expectation*confidence",
        "temperature": 1.0,
        "offsets_zyx": [list(offset) for offset in OFFSETS],
        "zero_offset_index": ZERO_OFFSET_INDEX,
        "utility": "NCCVxm win=9 eps=1e-5 FP64 masked mean",
        "utility_relative_tolerance": 1e-6,
        "coefficients": [2.0**-index for index in range(13)],
        "branches": list(BRANCH_ORDER),
        "seed": args.seed,
    }
    if contract_path.is_file():
        existing = json.loads(contract_path.read_text(encoding="utf-8"))
        if existing != contract:
            raise RuntimeError("Resume refused: stage contract differs from the existing run")
    else:
        atomic_write_json(contract_path, contract)

    manifest_rows = dataset_rows(files, dataset_name, split, atlas)
    atomic_write_text(
        stage_dir / "datasets.csv",
        rows_to_csv(["dataset", "split", "case_id", "path", "bytes", "sha256", "mtime_utc"], manifest_rows),
    )
    atomic_write_text(
        stage_dir / "datasets.tsv",
        rows_to_tsv(["dataset", "split", "case_id", "path", "bytes", "sha256", "mtime_utc"], manifest_rows),
    )
    device = setup_device(args.gpu, seed=args.seed, deterministic=True)
    adapter, model, load_report = build_model(str(checkpoint_path), config, device)
    dataset = build_infer_dataset(dataset_name, files, atlas)
    labels = tuple(metric_profile_for(dataset_name).labels)
    all_rows: list[dict[str, Any]] = []
    try:
        for index, path in enumerate(files):
            print(f"[{index + 1}/{len(files)}] {dataset_name} {case_id_from_path(path)}", flush=True)
            all_rows.extend(
                _run_case(index, path, dataset, adapter, model, device, labels, stage_dir, args.keep_fields)
            )
    except BaseException as exc:
        atomic_write_json(
            stage_dir / "stage_failure.json",
            {
                "schema": "ctcf-search-c0-stage-failure-v1",
                "status": "FAILED",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "completed_at_utc": utc_now(),
            },
        )
        raise

    fields = list(all_rows[0].keys())
    atomic_write_text(stage_dir / "per_case_branch.csv", rows_to_csv(fields, all_rows))
    summary = _summarise(all_rows, args.stage)
    atomic_write_json(stage_dir / "summary.json", summary)
    failure_marker = stage_dir / "stage_failure.json"
    if failure_marker.exists():
        failure_marker.unlink()
    manifest = {
        "schema": "ctcf-native-manifest-v1",
        "protocol_id": PROTOCOL_ID,
        "status": "COMPLETE",
        "stage": args.stage,
        "started_at_utc": started_at,
        "completed_at_utc": utc_now(),
        "code": {
            "git_head": head,
            "branch": git("branch", "--show-current"),
            "git_status": git("status", "--porcelain=v1"),
        },
        "execution": {
            "host": platform.node(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "device": str(device),
            "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            "paths_profile": args.paths_profile,
            "seed": args.seed,
            "deterministic": True,
        },
        "checkpoint": {**load_report, "path": str(checkpoint_path), "sha256": sha256_file(checkpoint_path)},
        "contract_sha256": sha256_file(contract_path),
        "datasets_sha256": sha256_file(stage_dir / "datasets.csv"),
        "results_sha256": sha256_file(stage_dir / "per_case_branch.csv"),
        "summary_sha256": sha256_file(stage_dir / "summary.json"),
        "summary": summary,
        "storage": {"heavy_fields_retained": args.keep_fields, "compact_outputs_only_by_default": True},
    }
    atomic_write_json(stage_dir / "run_manifest.json", manifest)
    print(json.dumps(summary, indent=2), flush=True)
    # A scientifically negative or no-op C0 is a completed experiment, not an execution failure.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
