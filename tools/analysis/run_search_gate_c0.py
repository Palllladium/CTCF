from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import platform
import subprocess
import time
from contextlib import suppress
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

from experiments.core.inference_metrics import metric_profile_for
from experiments.core.inference_runtime import build_infer_dataset, load_checkpoint_state
from experiments.core.model_adapters import get_model_adapter
from experiments.core.path_profiles import get_dataset_paths
from tools.analysis.run_artifacts import (
    atomic_write_json,
    atomic_write_text,
    rows_to_csv,
    rows_to_tsv,
    sha256_file,
)
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
    phi_to_psi_displacement,
    sample_at_psi,
    save_flow_npz_atomic,
    screen_candidates,
    utility_loss,
)
from utils import dice_per_label, setup_device
from utils.cert_exact import certify_flow_exact
from utils.field import (
    boundary_nonzero_count,
    boundary_vertex_mask,
    digital_project,
    enforce_identity_boundary,
    identity_collar,
    trilinear_cert_bound,
    trilinear_project,
)

PROTOCOL_ID = "CTCF-SEARCH-GATE-C0-V1"
PROTOCOL_SALT = "CTCF-GATE-C0-V1|"
CLAIM_EPS = 0.001
WORK_EPS = 0.0011
COLLAR_WIDTH = 4
TIME_STEPS = 6
IXI_DEVELOPMENT_CASES = (
    "subject_344",
    "subject_136",
    "subject_165",
    "subject_475",
    "subject_131",
    "subject_389",
    "subject_485",
    "subject_153",
    "subject_252",
    "subject_509",
    "subject_126",
    "subject_459",
    "subject_222",
    "subject_474",
    "subject_144",
    "subject_85",
    "subject_248",
    "subject_151",
    "subject_295",
)
BRANCH_ORDER = ("mind_soft", "mind_hard", "intensity_soft", "mind_reversed")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _git(*args: str) -> str:
    return subprocess.check_output(["git", *args], text=True, encoding="utf-8").strip()


def _case_id(path: str) -> str:
    stem = Path(path).stem
    return stem[2:] if stem.startswith("p_") else stem


def _salted_case_hash(case_id: str) -> str:
    return hashlib.sha256((PROTOCOL_SALT + case_id).encode("utf-8")).hexdigest()


def _select_cases(stage: str, paths_profile: int) -> tuple[str, str, list[str], str | None]:
    dataset = "OASIS" if stage == "smoke" else "IXI"
    paths = get_dataset_paths(paths_profile, dataset)
    split = "val"
    root = paths["val_dir"]
    all_files = sorted(glob.glob(os.path.join(root, "*.pkl")))
    if not all_files:
        raise FileNotFoundError(f"No {dataset}/{split} .pkl files found under {root}")
    by_id = {_case_id(path): path for path in all_files}

    if stage == "smoke":
        ranked = sorted((_salted_case_hash(case), case) for case in by_id)
        expected = "0456_0457"
        if len(ranked) != 19 or ranked[0][1] != expected:
            raise RuntimeError(f"OASIS smoke selection contract changed: n={len(ranked)}, minimum={ranked[0][1]}")
        selected_ids = [expected]
    else:
        missing = [case for case in IXI_DEVELOPMENT_CASES if case not in by_id]
        ranked = [case for _, case in sorted((_salted_case_hash(case), case) for case in by_id)[:19]]
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


def _build_model(checkpoint: str, config: str, device: torch.device):
    adapter = get_model_adapter("ctcf")
    model = adapter.build(time_steps=TIME_STEPS, config_key=config, l3_svf=True).to(device)
    load_report = load_checkpoint_state(model, checkpoint, strict=True)
    model.eval()
    return adapter, model, load_report


def _dataset_manifest(files: list[str], dataset: str, split: str, atlas: str | None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path_string in files + ([atlas] if atlas else []):
        assert path_string is not None
        path = Path(path_string)
        stat = path.stat()
        rows.append(
            {
                "dataset": dataset,
                "split": "atlas" if atlas and path_string == atlas else split,
                "case_id": "atlas" if atlas and path_string == atlas else _case_id(path_string),
                "path": str(path.resolve()),
                "bytes": stat.st_size,
                "sha256": sha256_file(path),
                "mtime_utc": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat().replace("+00:00", "Z"),
            }
        )
    return rows


def _prepare_initial_state(flow: torch.Tensor, work_dir: Path) -> tuple[torch.Tensor, Path, dict[str, Any]]:
    collared = identity_collar(flow.float(), width=COLLAR_WIDTH)
    fixed_mask = boundary_vertex_mask(collared)
    fixed_values = torch.zeros_like(collared)
    digital, residual, digital_iterations = digital_project(
        collared,
        eps=0.0,
        fixed_mask=fixed_mask,
        fixed_values=fixed_values,
    )
    if residual != 0.0:
        raise RuntimeError(f"Digital preconditioner failed closed: residual={residual}")
    repaired, repair = trilinear_project(
        digital,
        eps=WORK_EPS,
        fixed_mask=fixed_mask,
        fixed_values=fixed_values,
    )
    phi = enforce_identity_boundary(repaired.float())
    if not repair.certified or repair.cert_bound < WORK_EPS or boundary_nonzero_count(phi) != 0:
        raise RuntimeError(f"Initial Phi repair failed closed: status={repair.status}, bound={repair.cert_bound}")

    work_dir.mkdir(parents=True, exist_ok=True)
    phi_path = work_dir / "initial_phi.npz"
    save_flow_npz_atomic(phi_path, phi)
    stored_phi = load_flow_npz(phi_path)
    phi_exact = certify_flow_exact(stored_phi, eps=str(CLAIM_EPS))
    if phi_exact["status"] != "CERTIFIED" or phi_exact["boundary_nonzero_count"] != 0:
        raise RuntimeError(f"Stored initial Phi failed exact certification: {phi_exact['status']}")

    psi = phi_to_psi_displacement(phi).float()
    psi_path = work_dir / "initial_psi.npz"
    save_flow_npz_atomic(psi_path, psi)
    stored_psi = load_flow_npz(psi_path)
    psi_exact = certify_flow_exact(stored_psi, eps=str(CLAIM_EPS))
    if psi_exact["status"] != "CERTIFIED":
        raise RuntimeError(f"Stored initial Psi failed exact certification: {psi_exact['status']}")
    report = {
        "digital_residual_percent": residual,
        "digital_iterations": digital_iterations,
        "trilinear_repair": asdict(repair),
        "phi_npz_sha256": sha256_file(phi_path),
        "phi_exact": phi_exact,
        "psi_npz_sha256": sha256_file(psi_path),
        "psi_exact": psi_exact,
    }
    return psi, psi_path, report


def _dice(psi: torch.Tensor, moving_seg: torch.Tensor, fixed_seg: torch.Tensor, labels: tuple[int, ...]) -> float:
    warped = sample_at_psi(moving_seg.float(), psi, mode="nearest").long()
    return float(dice_per_label(warped, fixed_seg.long(), labels=labels).mean())


def _proposal_statistics(proposal: ProposalResult, tensor: torch.Tensor, mask: torch.Tensor) -> dict[str, float]:
    magnitude = tensor.square().sum(dim=1, keepdim=True).sqrt()
    return {
        "entropy_mean": float(proposal.entropy.masked_select(mask).double().mean().item()),
        "confidence_mean": float(proposal.confidence.masked_select(mask).double().mean().item()),
        "proposal_norm_mean": float(magnitude.masked_select(mask).double().mean().item()),
        "proposal_norm_max": float(magnitude.masked_select(mask).max().item()),
    }


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
    baseline_dice = _dice(initial_psi, moving_seg, fixed_seg, labels)
    ungated_dice = _dice(ungated, moving_seg, fixed_seg, labels)
    final_dice = _dice(final_psi, moving_seg, fixed_seg, labels)

    report = {
        "branch": name,
        "feature": proposal_result.feature,
        "orientation": proposal_result.orientation,
        "decoder": "hard" if name == "mind_hard" else "soft",
        **_proposal_statistics(proposal_result, proposal_tensor, mask),
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
    case_id = _case_id(path)
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
    initial_psi, initial_path, initial_report = _prepare_initial_state(flow, case_dir / "work")
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
        "selection_sha256": _salted_case_hash(case_id),
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
    started_at = _utc_now()
    head = _git("rev-parse", "HEAD")
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
        "case_ids": [_case_id(path) for path in files],
        "selection_hashes": {_case_id(path): _salted_case_hash(_case_id(path)) for path in files},
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

    dataset_rows = _dataset_manifest(files, dataset_name, split, atlas)
    atomic_write_text(
        stage_dir / "datasets.csv",
        rows_to_csv(["dataset", "split", "case_id", "path", "bytes", "sha256", "mtime_utc"], dataset_rows),
    )
    atomic_write_text(
        stage_dir / "datasets.tsv",
        rows_to_tsv(["dataset", "split", "case_id", "path", "bytes", "sha256", "mtime_utc"], dataset_rows),
    )
    device = setup_device(args.gpu, seed=args.seed, deterministic=True)
    adapter, model, load_report = _build_model(str(checkpoint_path), config, device)
    dataset = build_infer_dataset(dataset_name, files, atlas)
    labels = tuple(metric_profile_for(dataset_name).labels)
    all_rows: list[dict[str, Any]] = []
    try:
        for index, path in enumerate(files):
            print(f"[{index + 1}/{len(files)}] {dataset_name} {_case_id(path)}", flush=True)
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
                "completed_at_utc": _utc_now(),
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
        "completed_at_utc": _utc_now(),
        "code": {
            "git_head": head,
            "branch": _git("branch", "--show-current"),
            "git_status": _git("status", "--porcelain=v1"),
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
