from __future__ import annotations

import hashlib
import json
import math
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from tools.analysis.run_artifacts import atomic_write_json, atomic_write_text, rows_to_csv, sha256_file
from tools.analysis.search_gate_c3 import metric_envelope
from tools.analysis.search_gate_c5b import (
    ANCHOR_ARM_IDS,
    ARM_SPECS,
    ARM_SPECS_BY_ID,
    CENTRE_BETA,
    COMMON_EVIDENCE_COLLAR,
    DIAGNOSTIC_ARM_ID,
    DICE_VS_REFERENCE_FAMILY_ID,
    EVALUATION_LABEL_IDS,
    EXACT_CLAIM_EPS,
    EXPECTED_CASE_COUNT,
    IMAGE_NORMALIZATION_STD_FLOOR,
    POST_SMOOTHING_PASSES,
    POSTERIOR_TEMPERATURE,
    PRE_RMS_MULTIPLIER,
    PROTOCOL_ID,
    REFERENCE_ARM_ID,
    SCHEMA_VERSION,
    SDLOGJ_VS_REFERENCE_FAMILY_ID,
    SELECTABLE_ARM_IDS,
    STANDARDIZATION_FLOOR,
    STRIDE_VOXELS,
    WORK_EPS,
    ArmEvidence,
    RegionalEvidence,
    regional_repair_family_id,
    regional_zero_family_id,
    select_next_branch,
    simultaneous_paired_summaries,
    validate_c5b_clip_operator,
    validate_c5b_geometry_bundle,
)
from tools.analysis.search_gate_c5b_contracts import (
    DECISION_CASE_SCHEMA,
    EVALUATION_CASE_SCHEMA,
    WORKER_SCHEMA,
    decision_case_path,
    evaluation_case_path,
    load_json,
    validate_decision_case_marker,
    validate_evaluation_case_marker,
    validate_worker_marker,
    verify_field_record,
    verify_image_record,
    worker_path,
)
from tools.analysis.search_gate_intensity_runtime import (
    build_intensity_reach_bank,
    decode_intensity_direction,
    materialize_postprocessed_intensity_candidate,
    postprocess_intensity_direction,
)
from tools.analysis.search_gate_metrics import (
    LEARN2REG_SHIFTED_SDLOGJ_MASKED,
    MATHEMATICAL_SDLOGJ_CROP2,
    METRIC_SPECS,
    compute_metric,
)
from tools.analysis.search_gate_multiscale import build_c4_common_support
from tools.analysis.search_gate_runtime import save_reload_certify
from tools.analysis.transactional_search import geometry_mask, load_flow_npz, masked_zscore, sample_at_psi
from utils import dice_per_label


def array_sha256(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes(order="C")).hexdigest()


def _roots(decision: Mapping[str, Any]) -> dict[str, str]:
    roots = decision.get("roots")
    expected = {"source_c3_heavy", "source_c4_heavy", "source_c5_heavy", "target_c5b_heavy"}
    if not isinstance(roots, Mapping) or set(roots) != expected:
        raise RuntimeError("C5b rooted-storage inventory changed")
    return {str(key): str(value) for key, value in roots.items()}


def _load_field(decision: Mapping[str, Any], record: Mapping[str, Any]) -> torch.Tensor:
    return load_flow_npz(verify_field_record(record, _roots(decision), "worker field"))


def _load_image(decision: Mapping[str, Any], record: Mapping[str, Any]) -> np.ndarray:
    path = verify_image_record(record, _roots(decision), "worker image")
    array = np.load(path, allow_pickle=False)
    if (
        array.dtype != np.float32
        or list(array.shape) != record.get("shape")
        or not np.isfinite(array).all()
        or array_sha256(array) != record.get("array_sha256")
    ):
        raise RuntimeError(f"C5b image array changed: {path}")
    return np.ascontiguousarray(array)


def _field_record(path: Path, root_id: str, root: Path, digest: str) -> dict[str, Any]:
    return {
        "root_id": root_id,
        "relative_path": path.resolve().relative_to(root.resolve()).as_posix(),
        "npz_sha256": sha256_file(path),
        "array_sha256": digest,
    }


def _geometry_bundle(field: torch.Tensor, mask: torch.Tensor) -> dict[str, dict[str, Any]]:
    return {
        metric_id: metric_envelope(
            metric_id,
            lambda mid=metric_id: compute_metric(
                mid,
                field,
                mask if mid == LEARN2REG_SHIFTED_SDLOGJ_MASKED else None,
            ),
        ).to_dict()
        for metric_id in METRIC_SPECS
    }


def _metric_value(bundle: Mapping[str, Any], metric_id: str, label: str) -> float:
    row = bundle.get(metric_id)
    value = row.get("value") if isinstance(row, Mapping) else None
    if (
        not isinstance(row, Mapping)
        or row.get("status") != "OK"
        or isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise RuntimeError(f"C5b metric is undefined: {label}/{metric_id}")
    return float(value)


def _publish(path: Path, payload: dict[str, Any], validator: Callable[[Mapping[str, Any]], None]) -> None:
    validator(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(path, payload)
    validator(load_json(path))


def _source_anchor_record(decision: Mapping[str, Any], case_id: str, arm_id: str) -> Mapping[str, Any]:
    name = {
        REFERENCE_ARM_ID: "c4_reference_s2_a10_b0",
        ANCHOR_ARM_IDS[1]: "c5_s4_a10_b0_sweep1",
        ANCHOR_ARM_IDS[2]: "c5_s4_a20_b0_sweep1",
    }[arm_id]
    return decision["source_anchors"][case_id][name]["field"]


def _anchor_row(
    *,
    decision: Mapping[str, Any],
    case_id: str,
    arm_id: str,
    field: torch.Tensor,
    mask: torch.Tensor,
    proposal: Mapping[str, Any] | None,
    parity: Mapping[str, Any] | None,
) -> dict[str, Any]:
    spec = ARM_SPECS_BY_ID[arm_id]
    source = dict(_source_anchor_record(decision, case_id, arm_id))
    digest = array_sha256(field.cpu().numpy())
    if digest != source["array_sha256"]:
        raise RuntimeError(f"C5b source anchor array differs: {case_id}/{arm_id}")
    geometry = _geometry_bundle(field, mask)
    diagnostics = validate_c5b_geometry_bundle(geometry, f"{case_id}/{arm_id}")
    if proposal is not None:
        validate_c5b_clip_operator(
            proposal.get("operator"),
            expected_sweeps=spec.local_clip_sweeps,
            label=f"producer replay/{case_id}/{arm_id}",
        )
    return {
        "arm_index": spec.arm_index,
        "arm_id": arm_id,
        "role": spec.role,
        "selectable": False,
        "post_rms_amplitude": spec.post_rms_amplitude,
        "local_clip_sweeps": spec.local_clip_sweeps,
        "proposal": dict(proposal or {}),
        "source_parity": dict(parity or {"array_byte_identical": True}),
        "candidate_field": source,
        "exact": {
            "status": "CERTIFIED",
            "certified": True,
            "sha256": digest,
            "epsilon_decimal": "0.001",
            "provenance": "FROZEN_SOURCE",
        },
        "observed_fold_count": diagnostics.corner_union_violation_count,
        "geometry": geometry,
    }


def _new_arm_row(
    *,
    decision: Mapping[str, Any],
    case_id: str,
    arm_id: str,
    candidate: Any,
    mask: torch.Tensor,
    direction_sha256: str,
    postprocessed_sha256: str,
) -> dict[str, Any]:
    spec = ARM_SPECS_BY_ID[arm_id]
    root = Path(decision["roots"]["target_c5b_heavy"]).resolve()
    path = root / "cases" / case_id / "arms" / f"{arm_id}.npz"
    stored, exact = save_reload_certify(candidate.candidate, path, EXACT_CLAIM_EPS)
    if exact.get("certified") is not True or exact.get("status") != "CERTIFIED":
        raise RuntimeError(f"C5b exact certification failed: {case_id}/{arm_id}")
    if not torch.equal(stored.cpu(), candidate.candidate.cpu()):
        raise RuntimeError(f"C5b save/reload changed the field: {case_id}/{arm_id}")
    digest = array_sha256(stored.cpu().numpy())
    if exact.get("sha256") != digest:
        raise RuntimeError(f"C5b exact certificate hash changed: {case_id}/{arm_id}")
    geometry = _geometry_bundle(stored.to(mask.device), mask)
    diagnostics = validate_c5b_geometry_bundle(geometry, f"{case_id}/{arm_id}")
    proposal = {
        "amplitude_stage": "after_rms_match_before_local_clip",
        "post_rms_amplitude": spec.post_rms_amplitude,
        "local_clip_sweeps": spec.local_clip_sweeps,
        "rms_source": candidate.postprocessed.source_rms,
        "rms_target": candidate.postprocessed.target_rms,
        "rms_matched": candidate.postprocessed.output_rms,
        "rms_scale_factor": candidate.postprocessed.rms_scale_factor,
        "rms_requested": candidate.requested_rms,
        "rms_realized": candidate.realized_rms,
        "clip_rms_retention": candidate.clip_rms_retention,
        "clip_cosine": candidate.clip_cosine,
        "operator": candidate.operator,
        "preclip_direction_array_sha256": direction_sha256,
        "postprocessed_direction_array_sha256": postprocessed_sha256,
    }
    validate_c5b_clip_operator(
        proposal["operator"],
        expected_sweeps=spec.local_clip_sweeps,
        label=f"producer candidate/{case_id}/{arm_id}",
    )
    return {
        "arm_index": spec.arm_index,
        "arm_id": arm_id,
        "role": spec.role,
        "selectable": spec.selectable,
        "post_rms_amplitude": spec.post_rms_amplitude,
        "local_clip_sweeps": spec.local_clip_sweeps,
        "proposal": proposal,
        "source_parity": None,
        "candidate_field": _field_record(path, "target_c5b_heavy", root, digest),
        "exact": exact,
        "observed_fold_count": diagnostics.corner_union_violation_count,
        "geometry": geometry,
    }


def _replayed_anchor_proposal(
    candidate: Any,
    *,
    direction_sha256: str,
    postprocessed_sha256: str,
) -> dict[str, Any]:
    return {
        "amplitude_stage": "after_rms_match_before_local_clip",
        "post_rms_amplitude": candidate.post_rms_amplitude,
        "local_clip_sweeps": candidate.sweeps,
        "rms_source": candidate.postprocessed.source_rms,
        "rms_target": candidate.postprocessed.target_rms,
        "rms_matched": candidate.postprocessed.output_rms,
        "rms_scale_factor": candidate.postprocessed.rms_scale_factor,
        "rms_requested": candidate.requested_rms,
        "rms_realized": candidate.realized_rms,
        "clip_rms_retention": candidate.clip_rms_retention,
        "clip_cosine": candidate.clip_cosine,
        "operator": candidate.operator,
        "preclip_direction_array_sha256": direction_sha256,
        "postprocessed_direction_array_sha256": postprocessed_sha256,
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
    marker = decision_case_path(run_root, case_id)
    if marker.is_file():
        validate_decision_case_marker(load_json(marker), decision, decision_sha256, verify_heavy_bytes=True)
        return marker
    if case_id not in decision["shards"].get(str(shard_index), []):
        raise RuntimeError(f"C5b decision case belongs to another shard: {case_id}")
    if str(physical_gpu) != str(decision["shard_to_physical_gpu"].get(str(shard_index))):
        raise RuntimeError(f"C5b decision case belongs to another physical GPU: {case_id}")
    started = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    atlas = torch.from_numpy(_load_image(decision, decision["image_inputs"]["atlas"])).unsqueeze(0).to(device)
    fixed = torch.from_numpy(_load_image(decision, decision["image_inputs"][case_id])).unsqueeze(0).to(device)
    initial = _load_field(decision, decision["source_initial"][case_id]["field"]).to(device)
    historical = _load_field(decision, decision["source_rms"][case_id]["field"]).to(device)
    rms_reference = (historical - initial).float()
    mask = geometry_mask(tuple(initial.shape[-3:]), COMMON_EVIDENCE_COLLAR, device)
    common = build_c4_common_support(initial, mask)
    fixed_normalized = masked_zscore(fixed, mask, std_floor=IMAGE_NORMALIZATION_STD_FLOOR)
    moving_normalized = masked_zscore(atlas, mask, std_floor=IMAGE_NORMALIZATION_STD_FLOOR)
    bank = build_intensity_reach_bank(
        fixed_normalized,
        moving_normalized,
        initial,
        common.common_mask,
        reach_id="S4",
        cost_id="intensity_s4",
        stride_voxels=STRIDE_VOXELS,
        standardization_floor=STANDARDIZATION_FLOOR,
    )
    direction = decode_intensity_direction(
        bank,
        direction_id="intensity_s4_b0",
        centre_beta=CENTRE_BETA,
        posterior_temperature=POSTERIOR_TEMPERATURE,
    )
    postprocessed = postprocess_intensity_direction(
        direction,
        rms_reference,
        mask,
        pre_rms_multiplier=PRE_RMS_MULTIPLIER,
        smoothing_passes=POST_SMOOTHING_PASSES,
        collar_width=COMMON_EVIDENCE_COLLAR,
    )
    candidates = {
        spec.arm_id: materialize_postprocessed_intensity_candidate(
            direction,
            postprocessed,
            initial,
            mask,
            candidate_id=spec.arm_id,
            post_rms_amplitude=spec.post_rms_amplitude,
            work_eps=WORK_EPS,
            sweeps=spec.local_clip_sweeps,
        )
        for spec in ARM_SPECS
        if spec.stride_voxels == STRIDE_VOXELS
    }
    direction_sha = array_sha256(direction.decoded.displacement.cpu().numpy())
    postprocessed_sha = array_sha256(postprocessed.displacement.cpu().numpy())

    c4 = _load_field(decision, _source_anchor_record(decision, case_id, REFERENCE_ARM_ID)).to(device)
    rows = [
        _anchor_row(
            decision=decision, case_id=case_id, arm_id=REFERENCE_ARM_ID, field=c4, mask=mask, proposal=None, parity=None
        )
    ]
    for arm_id in ANCHOR_ARM_IDS[1:]:
        observed = candidates[arm_id]
        source = _load_field(decision, _source_anchor_record(decision, case_id, arm_id)).to(device)
        observed_sha = array_sha256(observed.candidate.cpu().numpy())
        source_sha = array_sha256(source.cpu().numpy())
        if observed_sha != source_sha or not torch.equal(observed.candidate.cpu(), source.cpu()):
            raise RuntimeError(f"C5b endpoint replay is not byte-identical: {case_id}/{arm_id}")
        rows.append(
            _anchor_row(
                decision=decision,
                case_id=case_id,
                arm_id=arm_id,
                field=source,
                mask=mask,
                proposal=_replayed_anchor_proposal(
                    observed,
                    direction_sha256=direction_sha,
                    postprocessed_sha256=postprocessed_sha,
                ),
                parity={
                    "source_array_sha256": source_sha,
                    "replayed_array_sha256": observed_sha,
                    "array_byte_identical": True,
                },
            )
        )
    for arm_id in (*SELECTABLE_ARM_IDS, DIAGNOSTIC_ARM_ID):
        rows.append(
            _new_arm_row(
                decision=decision,
                case_id=case_id,
                arm_id=arm_id,
                candidate=candidates[arm_id],
                mask=mask,
                direction_sha256=direction_sha,
                postprocessed_sha256=postprocessed_sha,
            )
        )
    if tuple(row["arm_id"] for row in rows) != tuple(spec.arm_id for spec in ARM_SPECS):
        raise RuntimeError("C5b worker arm order changed")

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
        "s4_preclip_direction_array_sha256": direction_sha,
        "s4_postprocessed_array_sha256": postprocessed_sha,
        "reach_support": {
            "generation_count": bank.generation_count,
            "raw_all_candidates_valid_count": bank.raw_all_candidates_valid_count,
            "standardized_informative_count": bank.standardized_informative_count,
        },
        "arms": rows,
        "resource": {
            "wall_sec": time.perf_counter() - started,
            "bank_elapsed_sec": bank.elapsed_sec,
            "peak_cuda_bytes": int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0,
        },
        "execution": dict(execution),
    }
    _publish(
        marker,
        payload,
        lambda value: validate_decision_case_marker(value, decision, decision_sha256, verify_heavy_bytes=True),
    )
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return marker


def run_decision_worker(
    *,
    case_ids: Sequence[str],
    shard_index: int,
    physical_gpu: str,
    attempt_id: str,
    run_root: Path,
    decision: Mapping[str, Any],
    decision_sha256: str,
    device: torch.device,
    execution: Mapping[str, Any],
) -> Path:
    if list(case_ids) != decision["shards"].get(str(shard_index)):
        raise RuntimeError("C5b decision worker shard changed")
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
    marker = worker_path(run_root, "decision", attempt_id, shard_index)
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
        "case_sha256": {case_id: sha256_file(decision_case_path(run_root, case_id)) for case_id in case_ids},
        "decision_contract_sha256": decision_sha256,
        "labels_loaded": False,
        "test_split_accessed": False,
        "execution": dict(execution),
    }
    _publish(
        marker,
        payload,
        lambda value: validate_worker_marker(
            value,
            decision,
            decision_sha256,
            phase="decision",
            shard_index=shard_index,
            attempt_id=attempt_id,
        ),
    )
    return marker


def _per_label_rows(labels: Sequence[int], baseline: torch.Tensor, candidate: torch.Tensor) -> list[dict[str, Any]]:
    return [
        {
            "label": int(label),
            "baseline_dice": float(left),
            "candidate_dice": float(right),
            "dice_delta": float(right - left),
        }
        for label, left, right in zip(labels, baseline, candidate, strict=True)
    ]


def run_evaluation_case(
    *,
    case_id: str,
    dataset_item: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    labels: Sequence[int],
    run_root: Path,
    decision: Mapping[str, Any],
    decision_sha256: str,
    barrier: Mapping[str, Any],
    barrier_sha256: str,
    evaluation: Mapping[str, Any],
    evaluation_sha256: str,
    device: torch.device,
    execution: Mapping[str, Any],
) -> Path:
    marker = evaluation_case_path(run_root, case_id)
    if marker.is_file():
        validate_evaluation_case_marker(
            load_json(marker), decision, decision_sha256, barrier, barrier_sha256, evaluation, evaluation_sha256
        )
        return marker
    decision_path = decision_case_path(run_root, case_id)
    if sha256_file(decision_path) != barrier["decision_case_sha256"][case_id]:
        raise RuntimeError(f"C5b decision snapshot changed before evaluation: {case_id}")
    decision_case = load_json(decision_path)
    validate_decision_case_marker(decision_case, decision, decision_sha256, verify_heavy_bytes=True)
    labels_tuple = tuple(int(value) for value in labels)
    if labels_tuple != EVALUATION_LABEL_IDS:
        raise RuntimeError("C5b IXI label order changed")
    moving_image, fixed_image, moving_seg, fixed_seg = dataset_item
    if array_sha256(moving_image.numpy()) != decision["image_inputs"]["atlas"]["array_sha256"]:
        raise RuntimeError("C5b evaluation atlas differs from decision cache")
    if array_sha256(fixed_image.numpy()) != decision["image_inputs"][case_id]["array_sha256"]:
        raise RuntimeError(f"C5b evaluation image differs from decision cache: {case_id}")
    moving_seg = moving_seg.unsqueeze(0).to(device)
    fixed_seg = fixed_seg.unsqueeze(0).to(device)
    initial = _load_field(decision, decision["source_initial"][case_id]["field"]).to(device)
    baseline_labels = dice_per_label(
        sample_at_psi(moving_seg.float(), initial, mode="nearest").long(),
        fixed_seg.long(),
        labels_tuple,
    )
    baseline_dice = float(baseline_labels.mean())
    frozen = evaluation["frozen_evaluation"][case_id]
    if not math.isclose(baseline_dice, float(frozen["baseline_dice"]), rel_tol=0.0, abs_tol=1e-8):
        raise RuntimeError(f"C5b baseline Dice differs from C5: {case_id}")
    rows = []
    for arm in decision_case["arms"]:
        candidate = _load_field(decision, arm["candidate_field"]).to(device)
        per_label = dice_per_label(
            sample_at_psi(moving_seg.float(), candidate, mode="nearest").long(),
            fixed_seg.long(),
            labels_tuple,
        )
        candidate_dice = float(per_label.mean())
        arm_id = str(arm["arm_id"])
        source_parity = None
        if arm_id in ANCHOR_ARM_IDS:
            expected = frozen["anchors"][arm_id]
            expected_labels = [float(item["candidate_dice"]) for item in expected["per_label"]]
            if not math.isclose(candidate_dice, float(expected["candidate_dice"]), rel_tol=0.0, abs_tol=1e-12) or any(
                not math.isclose(float(left), right, rel_tol=0.0, abs_tol=1e-12)
                for left, right in zip(per_label, expected_labels, strict=True)
            ):
                raise RuntimeError(f"C5b anchor Dice differs from C5: {case_id}/{arm_id}")
            source_parity = True
        rows.append(
            {
                "arm_index": arm["arm_index"],
                "arm_id": arm_id,
                "baseline_dice": baseline_dice,
                "candidate_dice": candidate_dice,
                "dice_delta": candidate_dice - baseline_dice,
                "source_evaluation_parity_verified": source_parity,
                "per_label": _per_label_rows(labels_tuple, baseline_labels, per_label),
            }
        )
    if sha256_file(decision_path) != barrier["decision_case_sha256"][case_id]:
        raise RuntimeError(f"C5b decision snapshot changed during evaluation: {case_id}")
    payload = {
        "schema": EVALUATION_CASE_SCHEMA,
        "protocol_id": PROTOCOL_ID,
        "status": "COMPLETE",
        "strict": True,
        "case_id": case_id,
        "decision_contract_sha256": decision_sha256,
        "decision_barrier_sha256": barrier_sha256,
        "evaluation_contract_sha256": evaluation_sha256,
        "decision_case_sha256": barrier["decision_case_sha256"][case_id],
        "labels_loaded_after_barrier": True,
        "test_split_accessed": False,
        "labels": list(labels_tuple),
        "arms": rows,
        "execution": dict(execution),
    }
    _publish(
        marker,
        payload,
        lambda value: validate_evaluation_case_marker(
            value, decision, decision_sha256, barrier, barrier_sha256, evaluation, evaluation_sha256
        ),
    )
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return marker


def run_evaluation_worker(
    *,
    case_ids: Sequence[str],
    dataset_item_for_case: Callable[[str], tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]],
    labels: Sequence[int],
    shard_index: int,
    physical_gpu: str,
    attempt_id: str,
    run_root: Path,
    decision: Mapping[str, Any],
    decision_sha256: str,
    barrier: Mapping[str, Any],
    barrier_sha256: str,
    evaluation: Mapping[str, Any],
    evaluation_sha256: str,
    device: torch.device,
    execution: Mapping[str, Any],
) -> Path:
    if list(case_ids) != decision["shards"].get(str(shard_index)):
        raise RuntimeError("C5b evaluation worker shard changed")
    for case_id in case_ids:
        run_evaluation_case(
            case_id=case_id,
            dataset_item=dataset_item_for_case(case_id),
            labels=labels,
            run_root=run_root,
            decision=decision,
            decision_sha256=decision_sha256,
            barrier=barrier,
            barrier_sha256=barrier_sha256,
            evaluation=evaluation,
            evaluation_sha256=evaluation_sha256,
            device=device,
            execution=execution,
        )
    marker = worker_path(run_root, "evaluation", attempt_id, shard_index)
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
        "case_sha256": {case_id: sha256_file(evaluation_case_path(run_root, case_id)) for case_id in case_ids},
        "decision_contract_sha256": decision_sha256,
        "decision_barrier_sha256": barrier_sha256,
        "evaluation_contract_sha256": evaluation_sha256,
        "labels_loaded": True,
        "test_split_accessed": False,
        "execution": dict(execution),
    }
    _publish(
        marker,
        payload,
        lambda value: validate_worker_marker(
            value,
            decision,
            decision_sha256,
            phase="evaluation",
            shard_index=shard_index,
            attempt_id=attempt_id,
            barrier_sha256=barrier_sha256,
            evaluation_contract_sha256=evaluation_sha256,
        ),
    )
    return marker


def _csv_value(value: Any) -> Any:
    return json.dumps(value, sort_keys=True, separators=(",", ":")) if isinstance(value, (dict, list, tuple)) else value


def _write_rows(path: Path, rows: list[dict[str, Any]], preferred: Sequence[str]) -> None:
    fields = [name for name in preferred if any(name in row for row in rows)]
    fields.extend(sorted({key for row in rows for key in row} - set(fields)))
    atomic_write_text(
        path, rows_to_csv(fields, [{key: _csv_value(row.get(key, "")) for key in fields} for row in rows])
    )


def _summary_map(family_id: str, vectors: Mapping[str, np.ndarray]) -> dict[str, Any]:
    return {row.contrast_id: row for row in simultaneous_paired_summaries(family_id, vectors)}


def finalize_c5b(
    *,
    run_root: Path,
    decision: Mapping[str, Any],
    decision_sha256: str,
    barrier: Mapping[str, Any],
    barrier_sha256: str,
    evaluation: Mapping[str, Any],
    evaluation_sha256: str,
    evaluation_barrier: Mapping[str, Any],
    evaluation_barrier_sha256: str,
) -> dict[str, str]:
    decision_cases: dict[str, dict[str, Any]] = {}
    evaluation_cases: dict[str, dict[str, Any]] = {}
    for case_id in decision["case_ids"]:
        dpath, epath = decision_case_path(run_root, case_id), evaluation_case_path(run_root, case_id)
        if sha256_file(dpath) != barrier["decision_case_sha256"][case_id]:
            raise RuntimeError(f"C5b decision snapshot changed before finalization: {case_id}")
        if sha256_file(epath) != evaluation_barrier["evaluation_case_sha256"][case_id]:
            raise RuntimeError(f"C5b evaluation snapshot changed before finalization: {case_id}")
        drow, erow = load_json(dpath), load_json(epath)
        validate_decision_case_marker(drow, decision, decision_sha256, verify_heavy_bytes=True)
        validate_evaluation_case_marker(
            erow, decision, decision_sha256, barrier, barrier_sha256, evaluation, evaluation_sha256
        )
        decision_cases[case_id], evaluation_cases[case_id] = drow, erow

    arm_ids = tuple(spec.arm_id for spec in ARM_SPECS)
    per_arm: list[dict[str, Any]] = []
    per_label: list[dict[str, Any]] = []
    resource_rows: list[dict[str, Any]] = []
    dice_by_arm = {arm_id: [] for arm_id in arm_ids}
    sd_by_arm = {arm_id: [] for arm_id in arm_ids}
    retention_by_arm = {arm_id: [] for arm_id in (*ANCHOR_ARM_IDS[1:], *SELECTABLE_ARM_IDS, DIAGNOSTIC_ARM_ID)}
    labels_by_arm = {arm_id: {label: [] for label in EVALUATION_LABEL_IDS} for arm_id in arm_ids}
    baseline_labels = {label: [] for label in EVALUATION_LABEL_IDS}
    exact_by_arm = {arm_id: [] for arm_id in arm_ids}
    folds_by_arm = {arm_id: [] for arm_id in arm_ids}
    digital_ten_percent_by_arm = {arm_id: [] for arm_id in arm_ids}
    digital_ten_count_by_arm = {arm_id: [] for arm_id in arm_ids}
    digital_jstar_fraction_by_arm = {arm_id: [] for arm_id in arm_ids}
    for case_id in decision["case_ids"]:
        drows = {row["arm_id"]: row for row in decision_cases[case_id]["arms"]}
        erows = {row["arm_id"]: row for row in evaluation_cases[case_id]["arms"]}
        resource_rows.append({"case_id": case_id, **decision_cases[case_id]["resource"]})
        for arm_id in arm_ids:
            drow, erow = drows[arm_id], erows[arm_id]
            sdlogj = _metric_value(drow["geometry"], MATHEMATICAL_SDLOGJ_CROP2, f"{case_id}/{arm_id}")
            geometry = validate_c5b_geometry_bundle(drow["geometry"], f"{case_id}/{arm_id}")
            fold_count = geometry.corner_union_violation_count
            dice_by_arm[arm_id].append(float(erow["candidate_dice"]))
            sd_by_arm[arm_id].append(sdlogj)
            exact_by_arm[arm_id].append(drow["exact"].get("certified") is True)
            folds_by_arm[arm_id].append(fold_count)
            digital_ten_percent_by_arm[arm_id].append(geometry.digital_ten_union_percent)
            digital_ten_count_by_arm[arm_id].append(geometry.digital_ten_union_violation_count)
            digital_jstar_fraction_by_arm[arm_id].append(geometry.jstar_union_violation_fraction)
            if arm_id in retention_by_arm:
                retention_by_arm[arm_id].append(float(drow["proposal"]["clip_rms_retention"]))
            per_arm.append(
                {
                    "case_id": case_id,
                    "arm_id": arm_id,
                    "role": drow["role"],
                    "selectable": drow["selectable"],
                    "baseline_dice": erow["baseline_dice"],
                    "candidate_dice": erow["candidate_dice"],
                    "dice_delta_vs_zero": erow["dice_delta"],
                    "mathematical_sdlogj": sdlogj,
                    "clip_rms_retention": drow["proposal"].get("clip_rms_retention"),
                    "observed_fold_count": fold_count,
                    "digital_ten_union_violation_count": geometry.digital_ten_union_violation_count,
                    "digital_ten_union_violation_fraction": geometry.digital_ten_union_violation_fraction,
                    "digital_ten_union_percent": geometry.digital_ten_union_percent,
                    "digital_jstar_union_violation_fraction": geometry.jstar_union_violation_fraction,
                    "exact_certified": drow["exact"]["certified"],
                    "candidate_field": drow["candidate_field"],
                }
            )
            for item in erow["per_label"]:
                label = int(item["label"])
                labels_by_arm[arm_id][label].append(float(item["candidate_dice"]))
                if arm_id == REFERENCE_ARM_ID:
                    baseline_labels[label].append(float(item["baseline_dice"]))
                per_label.append({"case_id": case_id, "arm_id": arm_id, **item})

    reference_dice = np.asarray(dice_by_arm[REFERENCE_ARM_ID], dtype=np.float64)
    reference_sd = np.asarray(sd_by_arm[REFERENCE_ARM_ID], dtype=np.float64)
    dice_vectors = {
        f"dice::{arm_id}::vs_{REFERENCE_ARM_ID}": np.asarray(dice_by_arm[arm_id]) - reference_dice
        for arm_id in SELECTABLE_ARM_IDS
    }
    sd_vectors = {
        f"sdlogj::{arm_id}::vs_{REFERENCE_ARM_ID}": np.asarray(sd_by_arm[arm_id]) - reference_sd
        for arm_id in SELECTABLE_ARM_IDS
    }
    dice_summaries = _summary_map(DICE_VS_REFERENCE_FAMILY_ID, dice_vectors)
    sd_summaries = _summary_map(SDLOGJ_VS_REFERENCE_FAMILY_ID, sd_vectors)
    zero_family = regional_zero_family_id(SELECTABLE_ARM_IDS[0])
    zero_vectors = {
        f"label::{label}::{arm_id}::vs_zero": np.asarray(labels_by_arm[arm_id][label])
        - np.asarray(baseline_labels[label])
        for arm_id in SELECTABLE_ARM_IDS
        for label in EVALUATION_LABEL_IDS
    }
    repair_family = regional_repair_family_id(SELECTABLE_ARM_IDS[0])
    repair_vectors = {
        f"label::{label}::{arm_id}::vs_{REFERENCE_ARM_ID}": np.asarray(labels_by_arm[arm_id][label])
        - np.asarray(labels_by_arm[REFERENCE_ARM_ID][label])
        for arm_id in SELECTABLE_ARM_IDS
        for label in (9, 29)
    }
    all_regional_zero = _summary_map(zero_family, zero_vectors)
    all_regional_repair = _summary_map(repair_family, repair_vectors)
    regional_zero = {
        arm_id: {
            f"label::{label}::{arm_id}::vs_zero": all_regional_zero[f"label::{label}::{arm_id}::vs_zero"]
            for label in EVALUATION_LABEL_IDS
        }
        for arm_id in SELECTABLE_ARM_IDS
    }
    regional_repair = {
        arm_id: {
            f"label::{label}::{arm_id}::vs_{REFERENCE_ARM_ID}": all_regional_repair[
                f"label::{label}::{arm_id}::vs_{REFERENCE_ARM_ID}"
            ]
            for label in (9, 29)
        }
        for arm_id in SELECTABLE_ARM_IDS
    }

    evidence_rows = []
    for arm_id in SELECTABLE_ARM_IDS:
        retention = np.asarray(retention_by_arm[arm_id], dtype=np.float64)
        evidence = ArmEvidence(
            arm_id=arm_id,
            dice_vs_reference=dice_summaries[f"dice::{arm_id}::vs_{REFERENCE_ARM_ID}"],
            sdlogj_vs_reference=sd_summaries[f"sdlogj::{arm_id}::vs_{REFERENCE_ARM_ID}"],
            regional=RegionalEvidence(
                tuple(regional_zero[arm_id].values()),
                tuple(regional_repair[arm_id].values()),
            ),
            all_work_units_complete=len(dice_by_arm[arm_id]) == EXPECTED_CASE_COUNT,
            all_exact_certified=all(exact_by_arm[arm_id]),
            observed_fold_count=sum(folds_by_arm[arm_id]),
            amplitude_retention_median=float(np.median(retention)),
            amplitude_retention_cases_at_least_090=int(np.count_nonzero(retention >= 0.90)),
        )
        evidence_rows.append(evidence)
    branch = select_next_branch(tuple(evidence_rows))

    arm_summary = []
    for spec in ARM_SPECS:
        arm_id = spec.arm_id
        retention = np.asarray(retention_by_arm.get(arm_id, []), dtype=np.float64)
        assessment = next((row for row in evidence_rows if row.arm_id == arm_id), None)
        arm_summary.append(
            {
                "arm_id": arm_id,
                "role": spec.role,
                "selectable": spec.selectable,
                "post_rms_amplitude": spec.post_rms_amplitude,
                "local_clip_sweeps": spec.local_clip_sweeps,
                "candidate_dice_mean": float(np.mean(dice_by_arm[arm_id])),
                "reference_c4_dice_mean": float(np.mean(reference_dice)),
                "dice_delta_vs_c4_mean": float(np.mean(np.asarray(dice_by_arm[arm_id]) - reference_dice)),
                "mathematical_sdlogj_mean": float(np.mean(sd_by_arm[arm_id])),
                "sdlogj_delta_vs_c4_mean": float(np.mean(np.asarray(sd_by_arm[arm_id]) - reference_sd)),
                "clip_rms_retention_median": float(np.median(retention)) if retention.size else None,
                "clip_rms_retention_cases_at_least_090": int(np.count_nonzero(retention >= 0.90))
                if retention.size
                else None,
                "all_exact_certified": all(exact_by_arm[arm_id]),
                "observed_fold_count": sum(folds_by_arm[arm_id]),
                "digital_ten_union_violation_count_total": sum(digital_ten_count_by_arm[arm_id]),
                "digital_ten_union_percent_mean": float(np.mean(digital_ten_percent_by_arm[arm_id])),
                "digital_jstar_union_violation_fraction_mean": float(np.mean(digital_jstar_fraction_by_arm[arm_id])),
                "dice_inference": asdict(assessment.dice_vs_reference) if assessment else None,
                "sdlogj_inference": asdict(assessment.sdlogj_vs_reference) if assessment else None,
            }
        )
    contrast_rows = [
        asdict(row)
        for rows in (
            dice_summaries.values(),
            sd_summaries.values(),
            *(regional_zero[arm_id].values() for arm_id in SELECTABLE_ARM_IDS),
            *(regional_repair[arm_id].values() for arm_id in SELECTABLE_ARM_IDS),
        )
        for row in rows
    ]
    summary = {
        "schema": f"ctcf-search-c5b-summary-{SCHEMA_VERSION}",
        "protocol_id": PROTOCOL_ID,
        "dataset": "IXI_VALIDATION_58",
        "cases": EXPECTED_CASE_COUNT,
        "test_115_authorized": False,
        "test_split_accessed": False,
        "reference_c4": {
            "arm_id": REFERENCE_ARM_ID,
            "dice_mean": float(np.mean(reference_dice)),
            "mathematical_sdlogj_mean": float(np.mean(reference_sd)),
            "digital_ten_union_percent_mean": float(np.mean(digital_ten_percent_by_arm[REFERENCE_ARM_ID])),
        },
        "arms": arm_summary,
        "diagnostic_w2_vs_w1": {
            "dice_delta_mean": float(
                np.mean(np.asarray(dice_by_arm[DIAGNOSTIC_ARM_ID]) - np.asarray(dice_by_arm[ANCHOR_ARM_IDS[2]]))
            ),
            "sdlogj_delta_mean": float(
                np.mean(np.asarray(sd_by_arm[DIAGNOSTIC_ARM_ID]) - np.asarray(sd_by_arm[ANCHOR_ARM_IDS[2]]))
            ),
            "digital_ten_union_percent_delta_mean": float(
                np.mean(
                    np.asarray(digital_ten_percent_by_arm[DIAGNOSTIC_ARM_ID])
                    - np.asarray(digital_ten_percent_by_arm[ANCHOR_ARM_IDS[2]])
                )
            ),
        },
        "next_branch": asdict(branch),
    }
    root = run_root.resolve()
    _write_rows(root / "per_arm.csv", per_arm, ("case_id", "arm_id"))
    _write_rows(root / "per_arm_label_dice.csv", per_label, ("case_id", "arm_id", "label"))
    _write_rows(root / "arm_summary.csv", arm_summary, ("arm_id",))
    _write_rows(root / "preregistered_contrasts.csv", contrast_rows, ("family_id", "contrast_id"))
    _write_rows(root / "resource_summary.csv", resource_rows, ("case_id",))
    atomic_write_json(root / "summary.json", summary)
    atomic_write_json(root / "next_branch.json", asdict(branch))
    artifacts = {
        name: sha256_file(root / name)
        for name in (
            "per_arm.csv",
            "per_arm_label_dice.csv",
            "arm_summary.csv",
            "preregistered_contrasts.csv",
            "resource_summary.csv",
            "summary.json",
            "next_branch.json",
        )
    }
    manifest = {
        "schema": f"ctcf-search-c5b-run-manifest-{SCHEMA_VERSION}",
        "protocol_id": PROTOCOL_ID,
        "status": "COMPLETE",
        "source_contract_sha256": sha256_file(root / "source_contract.json"),
        "decision_contract_sha256": decision_sha256,
        "decision_barrier_sha256": barrier_sha256,
        "evaluation_contract_sha256": evaluation_sha256,
        "evaluation_barrier_sha256": evaluation_barrier_sha256,
        "decision_case_sha256": dict(barrier["decision_case_sha256"]),
        "evaluation_case_sha256": dict(evaluation_barrier["evaluation_case_sha256"]),
        "files": artifacts,
        "next_branch": asdict(branch),
        "test_115_authorized": False,
        "test_split_accessed": False,
    }
    atomic_write_json(root / "c5b_manifest.json", manifest)
    return {**artifacts, "c5b_manifest.json": sha256_file(root / "c5b_manifest.json")}


__all__ = [
    "array_sha256",
    "finalize_c5b",
    "run_decision_case",
    "run_decision_worker",
    "run_evaluation_case",
    "run_evaluation_worker",
]
