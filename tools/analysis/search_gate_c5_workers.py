from __future__ import annotations

import json
import math
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from tools.analysis.run_artifacts import atomic_write_json, atomic_write_text, rows_to_csv, sha256_file
from tools.analysis.search_gate_c3 import binary_erode_mask, build_common_support, metric_envelope
from tools.analysis.search_gate_c5 import (
    AMPLITUDE_STAGE,
    ARM_SPECS,
    CAPACITY_FAMILY_ID,
    COMMON_EVIDENCE_COLLAR,
    CONTRAST_IDS_BY_FAMILY,
    DECODER_MODE,
    EVALUATION_LABEL_IDS,
    EXACT_CLAIM_EPS,
    EXPECTED_CASE_COUNT,
    HISTORICAL_ANCHOR_ARM_IDS,
    INCREMENTAL_FAMILY_ID,
    INFERENCE_FAMILY_IDS,
    INTERACTION_FAMILY_ID,
    LOCAL_CLIP_SWEEPS,
    MARGINAL_FAMILY_ID,
    MIND_DILATION,
    MIND_RADIUS,
    POST_SMOOTHING_PASSES,
    POSTERIOR_TEMPERATURE,
    PRIMARY_REFERENCE_ARM_ID,
    PRIMARY_SELECTOR_ID,
    PROTOCOL_ID,
    REACH_SPECS,
    REACH_SPECS_BY_ID,
    REGIONAL_REFERENCE_FAMILY_ID,
    REGIONAL_REPAIR_LABEL_IDS,
    REGIONAL_ZERO_FAMILY_ID,
    RMS_TARGET_SOURCE_ID,
    SELECTABLE_ARM_IDS,
    SELECTOR_IDS,
    SELECTOR_REFERENCE_FAMILY_ID,
    SELECTOR_SPECS_BY_ID,
    SELECTOR_ZERO_FAMILY_ID,
    STANDARDIZATION_FLOOR,
    WORK_EPS,
    ArmEvidence,
    CandidateSignals,
    RegionalEvidence,
    SelectorEvidence,
    assess_arm,
    choose_global_candidate,
    factor_contrast_differences,
    select_next_branch,
    simultaneous_paired_summaries,
)
from tools.analysis.search_gate_c5_contracts import (
    DECISION_CASE_SCHEMA,
    EVALUATION_CASE_SCHEMA,
    EXPECTED_SUPPORT_CONTRACT,
    SOURCE_C4_ANCHOR_IDS,
    WORKER_SCHEMA,
    array_sha256,
    validate_decision_case_marker,
    validate_evaluation_case_marker,
    validate_worker_marker,
    verify_rooted_record,
)
from tools.analysis.search_gate_cost_volume import masked_vector_rms
from tools.analysis.search_gate_metrics import (
    DETJ_DIAGNOSTICS,
    DIGITAL_DECOMPOSITION,
    LEARN2REG_SHIFTED_SDLOGJ_MASKED,
    MATHEMATICAL_SDLOGJ_CROP2,
    METRIC_SPECS,
    compute_metric,
)
from tools.analysis.search_gate_multiscale import (
    CenteredCostVolume,
    DecodedProposal,
    PosteriorVolume,
    build_c4_common_support,
    build_raw_intensity_cost_volume,
    centered_standardize,
    decode_posterior_mean,
    offsets_for_stride,
    posterior_from_standardized_costs_with_prior,
    postprocess_and_match_rms,
)
from tools.analysis.search_gate_runtime import save_reload_certify
from tools.analysis.transactional_search import (
    certified_local_clip_candidate,
    geometry_mask,
    load_flow_npz,
    masked_zscore,
    mind_distance_from_features,
    mind_ssc,
    ncc_loss_from_normalized,
    sample_at_psi,
    valid_sample_mask,
)
from utils import dice_per_label

SCHEMA_VERSION = "v1"
_ANCHOR_MAP = dict(zip(HISTORICAL_ANCHOR_ARM_IDS, SOURCE_C4_ANCHOR_IDS, strict=True))


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"{path}: expected a JSON object")
    return payload


def _roots(contract: Mapping[str, Any]) -> dict[str, Path]:
    roots = contract.get("roots")
    if not isinstance(roots, Mapping):
        raise RuntimeError("C5 contract has no rooted-storage map")
    output = {str(key): Path(str(value)).resolve() for key, value in roots.items()}
    if set(output) != {"source_c3_heavy", "source_c4_heavy", "target_c5_heavy"}:
        raise RuntimeError("C5 contract has the wrong rooted-storage inventory")
    return output


def _field_record(path: Path, root_id: str, root: Path, digest: str) -> dict[str, Any]:
    return {
        "root_id": root_id,
        "relative_path": path.resolve().relative_to(root.resolve()).as_posix(),
        "npz_sha256": sha256_file(path),
        "array_sha256": digest,
    }


def _load_field(contract: Mapping[str, Any], record: Mapping[str, Any]) -> torch.Tensor:
    path = verify_rooted_record(record, _roots(contract), verify_bytes=True, verify_array=True)
    return load_flow_npz(path)


def _load_image(contract: Mapping[str, Any], record: Mapping[str, Any]) -> np.ndarray:
    path = verify_rooted_record(record, _roots(contract), verify_bytes=True)
    array = np.load(path, allow_pickle=False)
    if (
        array.dtype != np.float32
        or list(array.shape) != record.get("shape")
        or not np.isfinite(array).all()
        or array_sha256(array) != record.get("array_sha256")
    ):
        raise RuntimeError(f"C5 image cache array changed or is invalid: {path}")
    return np.ascontiguousarray(array)


def _geometry_bundle(field: torch.Tensor, mask: torch.Tensor) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for metric_id in METRIC_SPECS:
        metric_mask = mask if metric_id == LEARN2REG_SHIFTED_SDLOGJ_MASKED else None
        output[metric_id] = metric_envelope(
            metric_id,
            lambda mid=metric_id, mm=metric_mask: compute_metric(mid, field, mm),
        ).to_dict()
    return output


def _metric_value(bundle: Mapping[str, Any], metric_id: str, label: str) -> float:
    row = bundle.get(metric_id)
    value = row.get("value") if isinstance(row, Mapping) else None
    if (
        not isinstance(row, Mapping)
        or row.get("metric_id") != metric_id
        or row.get("status") != "OK"
        or isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise RuntimeError(f"C5 required geometry metric is undefined: {label}/{metric_id}")
    return float(value)


def _metric_contract_values(bundle: Mapping[str, Any], metric_id: str, label: str) -> tuple[float, ...]:
    if metric_id != DETJ_DIAGNOSTICS:
        return (_metric_value(bundle, metric_id, label),)
    row = bundle.get(metric_id)
    components = row.get("components") if isinstance(row, Mapping) else None
    detj_min = components.get("detj_min") if isinstance(components, Mapping) else None
    invalid_count = components.get("invalid_count") if isinstance(components, Mapping) else None
    values = (detj_min, invalid_count)
    if (
        not isinstance(row, Mapping)
        or row.get("metric_id") != metric_id
        or row.get("status") != "OK"
        or row.get("value") is not None
        or any(
            isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value))
            for value in values
        )
    ):
        raise RuntimeError(f"C5 component-only detJ diagnostics are invalid: {label}/{metric_id}")
    return tuple(float(value) for value in values)


def _assert_exact_geometry(bundle: Mapping[str, Mapping[str, Any]], *, label: str) -> None:
    for metric_id in METRIC_SPECS:
        values = _metric_contract_values(bundle, metric_id, label)
        if metric_id == DETJ_DIAGNOSTICS and (values[0] <= 0.0 or values[1] != 0.0):
            raise RuntimeError(f"C5 component-only detJ diagnostics are invalid: {label}/{metric_id}")
    digital = bundle[DIGITAL_DECOMPOSITION]
    corner = float((digital.get("components") or {}).get("corner_union_violation_fraction", math.nan))
    if not math.isfinite(corner) or corner > 0.0:
        raise RuntimeError(f"Exact C5 certificate disagrees with digital corner determinants: {label}")


def _mean_on(tensor: torch.Tensor, mask: torch.Tensor, label: str) -> float:
    values = tensor.masked_select(mask)
    if values.numel() == 0 or not bool(torch.isfinite(values).all()):
        raise RuntimeError(f"C5 {label} support is empty or non-finite")
    return float(values.double().mean().item())


def _posterior_record(volume: CenteredCostVolume, posterior: PosteriorVolume, mask: torch.Tensor) -> dict[str, Any]:
    support = mask & volume.valid.all(dim=1, keepdim=True)
    logits = volume.standardized_costs.masked_fill(~volume.valid, torch.inf)
    top2 = (-logits).topk(k=2, dim=1).values
    return {
        "active_voxels": int(support.sum(dtype=torch.int64).item()),
        "entropy_mean": _mean_on(posterior.entropy, support, "posterior entropy"),
        "confidence_mean": _mean_on(posterior.confidence, support, "posterior confidence"),
        "top1_top2_gap_mean": _mean_on(top2[:, 0:1] - top2[:, 1:2], support, "posterior gap"),
        "floor_hit_fraction": _mean_on(volume.floor_hit.float(), mask, "standardization floor"),
        "candidate_cost_mean": _mean_on(volume.cost_mean, mask, "candidate cost mean"),
        "candidate_cost_std_mean": _mean_on(volume.cost_std, mask, "candidate cost std"),
        "work": asdict(volume.work),
    }


def _assert_support_contract(contract: Mapping[str, Any]) -> Mapping[str, Any]:
    support = contract.get("support_contract")
    if support != EXPECTED_SUPPORT_CONTRACT:
        raise RuntimeError("C5 worker support contract differs from the frozen definition")
    return support


def _central_difference_zyx(image: torch.Tensor) -> torch.Tensor:
    if image.ndim != 5 or image.shape[0] != 1 or image.shape[1] != 1:
        raise ValueError("C5 NGF image must have shape [1,1,D,H,W]")
    values = image.to(torch.float64)
    gradient = values.new_zeros((1, 3, *values.shape[-3:]))
    gradient[:, 0, 1:-1, :, :] = 0.5 * (values[:, 0, 2:, :, :] - values[:, 0, :-2, :, :])
    gradient[:, 1, :, 1:-1, :] = 0.5 * (values[:, 0, :, 2:, :] - values[:, 0, :, :-2, :])
    gradient[:, 2, :, :, 1:-1] = 0.5 * (values[:, 0, :, :, 2:] - values[:, 0, :, :, :-2])
    if not bool(torch.isfinite(gradient).all()):
        raise FloatingPointError("C5 NGF central differences became non-finite")
    return gradient


@dataclass(frozen=True)
class NGFReference:
    fixed_gradient: torch.Tensor
    baseline_gradient: torch.Tensor
    base_support: torch.Tensor
    baseline_valid_eroded: torch.Tensor
    eta: float


def _ngf_reference(
    fixed_norm: torch.Tensor,
    moving_norm: torch.Tensor,
    initial: torch.Tensor,
    common_mask: torch.Tensor,
) -> NGFReference:
    base_support = binary_erode_mask(common_mask, 1)
    baseline_valid = binary_erode_mask(valid_sample_mask(initial), 1)
    support = base_support & baseline_valid
    if not bool(support.any()):
        raise RuntimeError("C5 NGF baseline support is empty")
    fixed_gradient = _central_difference_zyx(fixed_norm)
    baseline_gradient = _central_difference_zyx(sample_at_psi(moving_norm, initial))
    energy = 0.5 * (
        fixed_gradient.square().sum(dim=1, keepdim=True) + baseline_gradient.square().sum(dim=1, keepdim=True)
    )
    mean_energy = _mean_on(energy, support, "NGF eta")
    eta = 0.1 * math.sqrt(mean_energy)
    if not math.isfinite(eta) or eta <= 0.0:
        raise RuntimeError("C5 NGF candidate-independent eta is non-positive or non-finite")
    return NGFReference(fixed_gradient, baseline_gradient, base_support, baseline_valid, eta)


def _ngf_similarity(
    fixed_gradient: torch.Tensor,
    warped_gradient: torch.Tensor,
    support: torch.Tensor,
    eta: float,
) -> float:
    eta2 = eta * eta
    dot = (fixed_gradient * warped_gradient).sum(dim=1, keepdim=True)
    fixed_norm2 = fixed_gradient.square().sum(dim=1, keepdim=True)
    warped_norm2 = warped_gradient.square().sum(dim=1, keepdim=True)
    similarity = (dot + eta2).square() / ((fixed_norm2 + eta2) * (warped_norm2 + eta2))
    value = _mean_on(similarity, support, "NGF similarity")
    if value < -1e-12 or value > 1.0 + 1e-12:
        raise RuntimeError("C5 NGF similarity escaped its mathematical [0,1] range")
    return min(1.0, max(0.0, value))


def _ngf_diagnostic(
    reference: NGFReference,
    moving_norm: torch.Tensor,
    candidate: torch.Tensor,
) -> dict[str, Any]:
    candidate_valid = binary_erode_mask(valid_sample_mask(candidate), 1)
    support = reference.base_support & reference.baseline_valid_eroded & candidate_valid
    if not bool(support.any()):
        raise RuntimeError("C5 NGF pair support is empty")
    baseline = _ngf_similarity(
        reference.fixed_gradient,
        reference.baseline_gradient,
        support,
        reference.eta,
    )
    candidate_gradient = _central_difference_zyx(sample_at_psi(moving_norm, candidate))
    observed = _ngf_similarity(reference.fixed_gradient, candidate_gradient, support, reference.eta)
    return {
        "diagnostic_id": EXPECTED_SUPPORT_CONTRACT["utilities"]["ngf"]["diagnostic_id"],
        "eta": reference.eta,
        "baseline_support_count": int(
            (reference.base_support & reference.baseline_valid_eroded).sum(dtype=torch.int64).item()
        ),
        "pair_support_count": int(support.sum(dtype=torch.int64).item()),
        "baseline_similarity": baseline,
        "candidate_similarity": observed,
        "improvement": observed - baseline,
        "selector_eligible": False,
    }


def _loss_row(
    utility_id: str,
    baseline_loss: float,
    candidate_loss: float,
    *,
    support: Any,
    selector_eligible: bool,
) -> dict[str, Any]:
    values = (baseline_loss, candidate_loss)
    if any(not math.isfinite(value) for value in values):
        raise RuntimeError(f"C5 {utility_id} utility is non-finite")
    return {
        "utility_id": utility_id,
        "baseline_count": support.baseline_count,
        "pair_count": support.pair_count,
        "retention": support.retention,
        "baseline_loss": baseline_loss,
        "candidate_loss": candidate_loss,
        "improvement": baseline_loss - candidate_loss,
        "selector_eligible": selector_eligible,
    }


def _utility_bundle(
    fixed_norm: torch.Tensor,
    moving_norm: torch.Tensor,
    fixed_mind: torch.Tensor,
    moving_mind: torch.Tensor,
    initial: torch.Tensor,
    candidate: torch.Tensor,
    mask: torch.Tensor,
    ngf_reference: NGFReference,
    support_contract: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    if support_contract != EXPECTED_SUPPORT_CONTRACT:
        raise RuntimeError("C5 utility calculation received an unfrozen support contract")
    baseline_valid = valid_sample_mask(initial)
    candidate_valid = valid_sample_mask(candidate)
    definitions = support_contract["utilities"]
    utilities: dict[str, dict[str, Any]] = {}
    for name in ("ncc5", "ncc7", "ncc9"):
        definition = definitions[name]
        support = build_common_support(
            mask,
            baseline_valid,
            candidate_valid,
            window=int(definition["window"]),
            utility_id=str(definition["utility_id"]),
        )
        baseline_loss = ncc_loss_from_normalized(
            fixed_norm,
            moving_norm,
            initial,
            support.pair_mask,
            win=int(definition["window"]),
            eps=float(definition["eps"]),
        )
        candidate_loss = ncc_loss_from_normalized(
            fixed_norm,
            moving_norm,
            candidate,
            support.pair_mask,
            win=int(definition["window"]),
            eps=float(definition["eps"]),
        )
        utilities[name] = _loss_row(
            str(definition["utility_id"]),
            baseline_loss,
            candidate_loss,
            support=support,
            selector_eligible=bool(definition["selector_eligible"]),
        )

    mind_definition = definitions["mind_d2"]
    mind_support = build_common_support(
        mask,
        baseline_valid,
        candidate_valid,
        window=int(mind_definition["support_window"]),
        utility_id=str(mind_definition["utility_id"]),
    )
    baseline_mind = mind_distance_from_features(fixed_mind, moving_mind, initial, mind_support.pair_mask)
    candidate_mind = mind_distance_from_features(fixed_mind, moving_mind, candidate, mind_support.pair_mask)
    utilities["mind_d2"] = _loss_row(
        str(mind_definition["utility_id"]),
        baseline_mind,
        candidate_mind,
        support=mind_support,
        selector_eligible=True,
    )
    utilities["ngf"] = _ngf_diagnostic(ngf_reference, moving_norm, candidate)
    selector_retention = min(utilities["ncc7"]["retention"], utilities["mind_d2"]["retention"])

    def counts(name: str) -> dict[str, Any]:
        return {
            "baseline_count": utilities[name]["baseline_count"],
            "pair_count": utilities[name]["pair_count"],
            "retention": utilities[name]["retention"],
        }

    return ({"ncc7": counts("ncc7"), "mind_d2": counts("mind_d2"), "retention": selector_retention}, utilities)


def _build_reach_bank(
    fixed_norm: torch.Tensor,
    moving_norm: torch.Tensor,
    initial: torch.Tensor,
    generation_mask: torch.Tensor,
    stride: int,
) -> tuple[CenteredCostVolume, float]:
    started = time.perf_counter()
    raw = build_raw_intensity_cost_volume(
        fixed_norm,
        moving_norm,
        initial,
        generation_mask,
        offsets=offsets_for_stride(stride),
        cost_id=f"intensity_s{stride}",
    )
    invalid = generation_mask & ~raw.valid.all(dim=1, keepdim=True)
    if bool(invalid.any()):
        raise RuntimeError(f"C5 S{stride} is not fully valid on the frozen C4 generation support")
    return centered_standardize(raw, standardization_floor=STANDARDIZATION_FLOOR), time.perf_counter() - started


def _decode_direction(volume: CenteredCostVolume, beta: float) -> tuple[PosteriorVolume, DecodedProposal]:
    posterior = posterior_from_standardized_costs_with_prior(
        volume,
        beta=beta,
        temperature=POSTERIOR_TEMPERATURE,
    )
    if DECODER_MODE != "posterior_mean":
        raise RuntimeError("C5 worker only implements the frozen posterior-mean decoder")
    return posterior, decode_posterior_mean(posterior)


def _global_cosine(requested: torch.Tensor, realized: torch.Tensor, mask: torch.Tensor) -> float:
    expanded = mask.expand_as(requested)
    left = requested.masked_select(expanded).double()
    right = realized.masked_select(expanded).double()
    denominator = left.square().sum().sqrt() * right.square().sum().sqrt()
    if float(denominator.item()) == 0.0:
        return 0.0
    value = float((left * right).sum().div(denominator).item())
    if not math.isfinite(value):
        raise RuntimeError("C5 clipping cosine is non-finite")
    return min(1.0, max(-1.0, value))


def _apply_amplitude_and_clip(
    decoded: DecodedProposal,
    arm: Any,
    initial: torch.Tensor,
    rms_reference: torch.Tensor,
    mask: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, Any], dict[str, Any]]:
    reach = REACH_SPECS_BY_ID[arm.reach_id]
    post = postprocess_and_match_rms(
        decoded,
        mask,
        proposal_multiplier=reach.pre_rms_multiplier,
        smoothing_passes=POST_SMOOTHING_PASSES,
        collar_width=COMMON_EVIDENCE_COLLAR,
        rms_reference=rms_reference,
    )
    requested = (post.displacement * arm.post_rms_amplitude).float()
    requested_rms = masked_vector_rms(requested, mask)
    expected_rms = float(post.target_rms) * arm.post_rms_amplitude
    if not math.isclose(requested_rms, expected_rms, rel_tol=1e-7, abs_tol=1e-8):
        raise RuntimeError(f"C5 post-RMS amplitude stage changed: {arm.arm_id}")
    candidate, operator = certified_local_clip_candidate(
        initial,
        requested,
        mask,
        work_eps=WORK_EPS,
        sweeps=LOCAL_CLIP_SWEEPS,
    )
    realized = (candidate - initial).float()
    realized_rms = masked_vector_rms(realized, mask)
    raw_retention = realized_rms / requested_rms
    if not math.isfinite(raw_retention) or raw_retention < 0.0 or raw_retention > 1.0 + 1e-6:
        raise RuntimeError(f"C5 clipping RMS retention is invalid: {arm.arm_id}")
    retention = min(1.0, raw_retention)
    proposal = {
        "amplitude_stage": AMPLITUDE_STAGE,
        "pre_rms_multiplier": reach.pre_rms_multiplier,
        "post_rms_amplitude": arm.post_rms_amplitude,
        "smoothing_passes": post.smoothing_passes,
        "collar_width": post.collar_width,
        "rms_target_source_id": RMS_TARGET_SOURCE_ID,
        "rms_source": post.source_rms,
        "rms_target": post.target_rms,
        "rms_matched": post.output_rms,
        "rms_scale_factor": post.rms_scale_factor,
        "rms_requested": requested_rms,
        "rms_realized": realized_rms,
        "clip_rms_retention": retention,
        "clip_rms_retention_raw": raw_retention,
        "clip_cosine": _global_cosine(requested, realized, mask),
    }
    return candidate, proposal, operator


def _persist_candidate(
    *,
    case_id: str,
    arm: Any,
    candidate: torch.Tensor,
    decision: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any] | None]:
    digest = array_sha256(candidate.detach().cpu().numpy())
    if arm.historical_anchor:
        source_anchor_id = _ANCHOR_MAP[arm.arm_id]
        source = decision["source_c4_anchors"][case_id][source_anchor_id]
        field = source["field"]
        observed = _load_field(decision, field)
        if digest != field.get("array_sha256") or not torch.equal(candidate.cpu(), observed.cpu()):
            raise RuntimeError(f"C5 reconstruction differs from frozen C4 anchor: {case_id}/{arm.arm_id}")
        exact = {
            "status": "CERTIFIED",
            "certified": True,
            "sha256": digest,
            "provenance": "FROZEN_C4_EXACT_CERTIFICATE_AND_C5_BYTE_RECONSTRUCTION",
        }
        parity = {
            "arm_id": arm.arm_id,
            "source_anchor_id": source_anchor_id,
            "source_array_sha256": field["array_sha256"],
            "candidate_array_sha256": digest,
            "array_byte_identical": True,
        }
        return dict(field), exact, parity

    roots = _roots(decision)
    heavy_root = roots["target_c5_heavy"]
    path = heavy_root / "cases" / case_id / "arms" / f"{arm.arm_id}.npz"
    stored, exact = save_reload_certify(candidate, path, EXACT_CLAIM_EPS)
    if exact.get("status") != "CERTIFIED" or exact.get("certified") is not True:
        raise RuntimeError(f"C5 save/reload exact certificate failed: {case_id}/{arm.arm_id}")
    if not torch.equal(candidate.cpu(), stored.cpu()):
        raise RuntimeError(f"C5 saved field differs from materialized field: {case_id}/{arm.arm_id}")
    return _field_record(path, "target_c5_heavy", heavy_root, str(exact["sha256"])), exact, None


def _assert_unique_field_records(rows: Sequence[Mapping[str, Any]]) -> None:
    identities = [(row["candidate_field"].get("root_id"), row["candidate_field"].get("relative_path")) for row in rows]
    if len(identities) != len(set(identities)):
        raise RuntimeError("C5 materialized arm inventory contains duplicate field owners")


def _assert_decision_label_free(payload: Mapping[str, Any]) -> None:
    allowed = {
        "labels_loaded_to_device",
        "labels_available_to_decision_workers",
        "decision_contract_contains_label_data",
    }
    stack: list[Any] = [payload]
    while stack:
        value = stack.pop()
        if isinstance(value, Mapping):
            for key, child in value.items():
                token = str(key).lower()
                if "dice" in token or "segmentation" in token:
                    raise RuntimeError("C5 decision payload leaked label-derived data")
                if "label" in token and token not in allowed and not token.startswith("label_free"):
                    raise RuntimeError("C5 decision payload leaked label-derived data")
                if token in allowed and child is not False:
                    raise RuntimeError("C5 decision payload leaked label-derived data")
                stack.append(child)
        elif isinstance(value, (list, tuple)):
            stack.extend(value)
        elif isinstance(value, str) and ("segmentation" in value.lower() or value.lower().endswith(".pkl")):
            raise RuntimeError("C5 decision payload leaked raw-container data")


def _selector_rows(arm_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    if tuple(row.get("arm_id") for row in arm_rows) != SELECTABLE_ARM_IDS:
        raise RuntimeError("C5 selectors require every arm in frozen order")
    signals = [
        CandidateSignals(
            arm_id=str(row["arm_id"]),
            exact_certified=(row.get("exact") or {}).get("certified") is True,
            support_retention=float(row["support"]["retention"]),
            amplitude_retention=float(row["proposal"]["clip_rms_retention"]),
            ncc7_improvement=float(row["utilities"]["ncc7"]["improvement"]),
            mind_d2_improvement=float(row["utilities"]["mind_d2"]["improvement"]),
            mathematical_sdlogj_delta=float(row["mathematical_sdlogj_delta"]),
        )
        for row in arm_rows
    ]
    by_arm = {row["arm_id"]: row for row in arm_rows}
    output: list[dict[str, Any]] = []
    for selector_id in SELECTOR_IDS:
        selector = SELECTOR_SPECS_BY_ID[selector_id]
        choice = choose_global_candidate(signals, selector_id)
        field = by_arm[choice.selected_arm_id]["candidate_field"] if choice.selected_arm_id is not None else None
        output.append(
            {
                "selector_index": selector.selector_index,
                "selector_id": selector_id,
                "action": choice.action,
                "selected_arm_id": choice.selected_arm_id,
                "eligible_arm_ids": list(choice.eligible_arm_ids),
                "returned_field": field,
                "rollback_to_source_initial": choice.selected_arm_id is None,
            }
        )
    return output


def _decision_case_path(run_root: Path, case_id: str) -> Path:
    return run_root.resolve() / "cases" / case_id / "decision_complete.json"


def _worker_path(run_root: Path, phase: str, attempt_id: str, shard_index: int) -> Path:
    return run_root.resolve() / "workers" / phase / "attempts" / attempt_id / f"worker_{shard_index:02d}.json"


def _verify_baseline_geometry(
    observed: Mapping[str, Any],
    expected: Mapping[str, Any],
    *,
    case_id: str,
) -> None:
    for metric_id in METRIC_SPECS:
        left = _metric_contract_values(observed, metric_id, f"{case_id}/recomputed baseline")
        right = _metric_contract_values(expected, metric_id, f"{case_id}/frozen baseline")
        if len(left) != len(right) or any(
            not math.isclose(observed_value, expected_value, rel_tol=0.0, abs_tol=1e-12)
            for observed_value, expected_value in zip(left, right, strict=True)
        ):
            raise RuntimeError(f"C5 baseline geometry differs from frozen C4: {case_id}/{metric_id}")


def _materialize_arm(
    *,
    case_id: str,
    arm: Any,
    decoded: DecodedProposal,
    volume: CenteredCostVolume,
    posterior: PosteriorVolume,
    bank_elapsed: float,
    initial: torch.Tensor,
    rms_reference: torch.Tensor,
    mask: torch.Tensor,
    generation_mask: torch.Tensor,
    fixed_norm: torch.Tensor,
    moving_norm: torch.Tensor,
    fixed_mind: torch.Tensor,
    moving_mind: torch.Tensor,
    ngf_reference: NGFReference,
    baseline_geometry: Mapping[str, Any],
    decision: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    candidate, proposal, operator = _apply_amplitude_and_clip(decoded, arm, initial, rms_reference, mask)
    field, exact, parity = _persist_candidate(
        case_id=case_id,
        arm=arm,
        candidate=candidate,
        decision=decision,
    )
    geometry = _geometry_bundle(candidate, mask)
    _assert_exact_geometry(geometry, label=f"{case_id}/{arm.arm_id}")
    support, utilities = _utility_bundle(
        fixed_norm,
        moving_norm,
        fixed_mind,
        moving_mind,
        initial,
        candidate,
        mask,
        ngf_reference,
        _assert_support_contract(decision),
    )
    candidate_sdlogj = _metric_value(geometry, MATHEMATICAL_SDLOGJ_CROP2, arm.arm_id)
    baseline_sdlogj = _metric_value(baseline_geometry, MATHEMATICAL_SDLOGJ_CROP2, "baseline")
    proposal.update(
        {
            "reach_id": arm.reach_id,
            "stride_voxels": arm.stride_voxels,
            "centre_beta": arm.centre_beta,
            "posterior": _posterior_record(volume, posterior, generation_mask),
            "bank_elapsed_sec": bank_elapsed,
        }
    )
    row = {
        "arm_index": arm.arm_index,
        "arm_id": arm.arm_id,
        "descriptor_id": arm.descriptor_id,
        "reach_id": arm.reach_id,
        "stride_voxels": arm.stride_voxels,
        "post_rms_amplitude": arm.post_rms_amplitude,
        "centre_beta": arm.centre_beta,
        "historical_anchor": arm.historical_anchor,
        "selectable": arm.selectable,
        "proposal": proposal,
        "operator": operator,
        "candidate_field": field,
        "persistence": {
            "owner": field["root_id"],
            "saved_npz_sha256": field["npz_sha256"],
            "reloaded_array_sha256": field["array_sha256"],
            "source_anchor_reused": arm.historical_anchor,
        },
        "exact": exact,
        "geometry": geometry,
        "mathematical_sdlogj_delta": candidate_sdlogj - baseline_sdlogj,
        "support": support,
        "utilities": utilities,
    }
    return row, parity


def run_decision_case(
    *,
    case_id: str,
    shard_index: int,
    physical_gpu: str,
    run_root: Path,
    decision: Mapping[str, Any],
    decision_sha256: str,
    device: torch.device,
    execution: Mapping[str, Any] | None = None,
) -> Path:
    marker = _decision_case_path(run_root, case_id)
    if marker.is_file():
        validate_decision_case_marker(
            _load_json(marker),
            decision,
            decision_sha256,
            verify_heavy_bytes=True,
        )
        return marker
    if case_id not in decision["shards"].get(str(shard_index), []):
        raise RuntimeError(f"C5 decision case is assigned to another shard: {case_id}")
    if str(physical_gpu) != str(decision["shard_to_physical_gpu"].get(str(shard_index))):
        raise RuntimeError(f"C5 decision case is assigned to another physical GPU: {case_id}")
    _assert_support_contract(decision)
    started = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    atlas = _load_image(decision, decision["image_inputs"]["atlas"])
    case = _load_image(decision, decision["image_inputs"][case_id])
    moving = torch.from_numpy(atlas).unsqueeze(0).to(device)
    fixed = torch.from_numpy(case).unsqueeze(0).to(device)
    initial_record = decision["source_initial"][case_id]["field"]
    initial = _load_field(decision, initial_record).to(device)
    historical_record = decision["source_historical"][case_id]["raw_conf_requested_field"]
    historical_state = _load_field(decision, historical_record).to(device)
    rms_reference = (historical_state - initial).float()
    if masked_vector_rms(rms_reference, geometry_mask(tuple(initial.shape[-3:]), COMMON_EVIDENCE_COLLAR, device)) <= 0:
        raise RuntimeError(f"C5 authenticated RMS reference is zero: {case_id}")

    mask7 = geometry_mask(tuple(initial.shape[-3:]), COMMON_EVIDENCE_COLLAR, device)
    common = build_c4_common_support(initial, mask7)
    fixed_norm = masked_zscore(
        fixed,
        mask7,
        std_floor=float(decision["support_contract"]["normalization"]["std_floor"]),
    )
    moving_norm = masked_zscore(
        moving,
        mask7,
        std_floor=float(decision["support_contract"]["normalization"]["std_floor"]),
    )
    fixed_mind = mind_ssc(fixed_norm, radius=MIND_RADIUS, dilation=MIND_DILATION)
    moving_mind = mind_ssc(moving_norm, radius=MIND_RADIUS, dilation=MIND_DILATION)
    ngf_reference = _ngf_reference(fixed_norm, moving_norm, initial, common.common_mask)

    baseline_geometry = _geometry_bundle(initial, mask7)
    _assert_exact_geometry(baseline_geometry, label=f"{case_id}/source_initial")
    _verify_baseline_geometry(baseline_geometry, decision["baseline_geometry"][case_id], case_id=case_id)

    arm_rows: list[dict[str, Any]] = []
    parity_rows: list[dict[str, Any]] = []
    raw_bank_count = 0
    posterior_direction_count = 0
    reach_support: list[dict[str, Any]] = []
    for reach in REACH_SPECS:
        volume, bank_elapsed = _build_reach_bank(
            fixed_norm,
            moving_norm,
            initial,
            common.common_mask,
            reach.stride_voxels,
        )
        raw_bank_count += 1
        directions: dict[int, tuple[PosteriorVolume, DecodedProposal]] = {}
        for bias_level in range(3):
            arm = next(spec for spec in ARM_SPECS if spec.reach_id == reach.reach_id and spec.bias_level == bias_level)
            directions[bias_level] = _decode_direction(volume, arm.centre_beta)
            posterior_direction_count += 1
        reach_support.append(
            {
                "reach_id": reach.reach_id,
                "stride_voxels": reach.stride_voxels,
                "generation_count": common.common_count,
                "all_candidates_valid_count": int(
                    (common.common_mask & volume.valid.all(dim=1, keepdim=True)).sum(dtype=torch.int64).item()
                ),
                "all_candidates_valid": True,
            }
        )
        for arm in (spec for spec in ARM_SPECS if spec.reach_id == reach.reach_id):
            posterior, decoded = directions[arm.bias_level]
            row, parity = _materialize_arm(
                case_id=case_id,
                arm=arm,
                decoded=decoded,
                volume=volume,
                posterior=posterior,
                bank_elapsed=bank_elapsed,
                initial=initial,
                rms_reference=rms_reference,
                mask=mask7,
                generation_mask=common.common_mask,
                fixed_norm=fixed_norm,
                moving_norm=moving_norm,
                fixed_mind=fixed_mind,
                moving_mind=moving_mind,
                ngf_reference=ngf_reference,
                baseline_geometry=baseline_geometry,
                decision=decision,
            )
            arm_rows.append(row)
            if parity is not None:
                parity_rows.append(parity)

    arm_rows.sort(key=lambda row: int(row["arm_index"]))
    if tuple(row["arm_id"] for row in arm_rows) != SELECTABLE_ARM_IDS:
        raise RuntimeError("C5 arm materialization order is incomplete or changed")
    _assert_unique_field_records(arm_rows)
    parity_by_arm = {row["arm_id"]: row for row in parity_rows}
    parity_rows = [parity_by_arm[arm_id] for arm_id in HISTORICAL_ANCHOR_ARM_IDS]
    selectors = _selector_rows(arm_rows)
    payload = {
        "schema": DECISION_CASE_SCHEMA,
        "protocol_id": PROTOCOL_ID,
        "status": "COMPLETE",
        "case_id": case_id,
        "decision_contract_sha256": decision_sha256,
        "shard_index": shard_index,
        "physical_gpu": str(physical_gpu),
        "arm_specs_sha256": decision["arm_specs_sha256"],
        "selector_specs_sha256": decision["selector_specs_sha256"],
        "labels_loaded_to_device": False,
        "test_split_accessed": False,
        "source_image_array_sha256": decision["image_inputs"][case_id]["array_sha256"],
        "source_initial_array_sha256": initial_record["array_sha256"],
        "source_rms_reference": {
            "field": historical_record,
            "source_decision_case_sha256": decision["source_historical"][case_id]["source_decision_case_sha256"],
            "residual_rms": masked_vector_rms(rms_reference, mask7),
        },
        "generation_support": {
            "support_id": decision["support_contract"]["geometry"]["generation_support_id"],
            "geometry_count": common.geometry_count,
            "common_count": common.common_count,
            "retention": common.retention,
            "reach": reach_support,
        },
        "baseline_geometry": baseline_geometry,
        "historical_anchor_parity": parity_rows,
        "arms": arm_rows,
        "selectors": selectors,
        "resources": {
            "elapsed_sec": time.perf_counter() - started,
            "peak_gpu_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0,
            "raw_intensity_bank_count": raw_bank_count,
            "posterior_direction_count": posterior_direction_count,
            "materialized_arm_count": len(arm_rows),
            "source_anchor_reuse_count": len(parity_rows),
            "new_heavy_field_count": len(arm_rows) - len(parity_rows),
            "mind_feature_evaluations": 2,
        },
        "execution": dict(execution or {}),
    }
    if (raw_bank_count, posterior_direction_count, len(arm_rows), len(parity_rows)) != (4, 12, 36, 2):
        raise RuntimeError("C5 worker arithmetic differs from the frozen 4-bank/12-direction/36-arm design")
    _assert_decision_label_free(payload)
    marker.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(marker, payload)
    validate_decision_case_marker(payload, decision, decision_sha256, verify_heavy_bytes=True)
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
    execution: Mapping[str, Any] | None = None,
) -> Path:
    expected = decision["shards"].get(str(shard_index))
    if list(case_ids) != expected or str(physical_gpu) != str(decision["shard_to_physical_gpu"].get(str(shard_index))):
        raise RuntimeError("C5 decision worker does not match the frozen shard")
    for case_id in case_ids:
        run_decision_case(
            case_id=case_id,
            shard_index=shard_index,
            physical_gpu=str(physical_gpu),
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
        "phase": "decision",
        "attempt_id": attempt_id,
        "shard_index": shard_index,
        "physical_gpu": str(physical_gpu),
        "case_ids": list(case_ids),
        "decision_contract_sha256": decision_sha256,
        "labels_loaded": False,
        "test_split_accessed": False,
        "execution": dict(execution or {}),
    }
    _assert_decision_label_free(payload)
    marker.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(marker, payload)
    validate_worker_marker(
        payload,
        decision,
        decision_sha256,
        phase="decision",
        shard_index=shard_index,
        attempt_id=attempt_id,
    )
    return marker


def _evaluation_case_path(run_root: Path, case_id: str) -> Path:
    return run_root.resolve() / "cases" / case_id / "evaluation_complete.json"


def _per_label_rows(
    labels: Sequence[int],
    baseline: torch.Tensor,
    observed: torch.Tensor,
    *,
    value_key: str,
) -> list[dict[str, Any]]:
    return [
        {
            "label": int(label),
            "baseline_dice": float(base),
            value_key: float(value),
            "dice_delta": float(value - base),
        }
        for label, base, value in zip(labels, baseline, observed, strict=True)
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
    evaluation_contract: Mapping[str, Any],
    evaluation_contract_sha256: str,
    device: torch.device,
    execution: Mapping[str, Any] | None = None,
) -> Path:
    marker = _evaluation_case_path(run_root, case_id)
    decision_path = _decision_case_path(run_root, case_id)
    expected_decision_sha = barrier["decision_case_sha256"][case_id]
    if sha256_file(decision_path) != expected_decision_sha:
        raise RuntimeError(f"C5 decision snapshot changed before evaluation: {case_id}")
    decision_case = _load_json(decision_path)
    validate_decision_case_marker(
        decision_case,
        decision,
        decision_sha256,
        verify_heavy_bytes=True,
    )
    if marker.is_file():
        payload = _load_json(marker)
        validate_evaluation_case_marker(
            payload,
            decision,
            decision_sha256,
            barrier,
            barrier_sha256,
            evaluation_contract,
            evaluation_contract_sha256,
            decision_case,
            expected_decision_sha,
        )
        return marker

    labels_tuple = tuple(int(value) for value in labels)
    if labels_tuple != EVALUATION_LABEL_IDS or tuple(evaluation_contract["evaluation_label_ids"]) != labels_tuple:
        raise RuntimeError("C5 evaluation received the wrong IXI label order")
    moving_image, fixed_image, moving_seg, fixed_seg = dataset_item
    if array_sha256(moving_image.numpy()) != decision["image_inputs"]["atlas"]["array_sha256"]:
        raise RuntimeError("C5 evaluation atlas image differs from the decision cache")
    if array_sha256(fixed_image.numpy()) != decision["image_inputs"][case_id]["array_sha256"]:
        raise RuntimeError(f"C5 evaluation case image differs from the decision cache: {case_id}")
    moving_seg_device = moving_seg.unsqueeze(0).to(device)
    fixed_seg_device = fixed_seg.unsqueeze(0).to(device)
    initial = _load_field(decision, decision["source_initial"][case_id]["field"]).to(device)
    baseline_warped = sample_at_psi(moving_seg_device.float(), initial, mode="nearest").long()
    baseline_per_label = dice_per_label(baseline_warped, fixed_seg_device.long(), labels_tuple)
    baseline_dice = float(baseline_per_label.mean())
    frozen_baseline = float(evaluation_contract["evaluation_baseline_dice"][case_id])
    if not math.isclose(baseline_dice, frozen_baseline, rel_tol=0.0, abs_tol=1e-8):
        raise RuntimeError(f"C5 baseline Dice differs from frozen C4: {case_id}")
    frozen_per_label = evaluation_contract["evaluation_baseline_per_label"][case_id]
    if tuple(row["label"] for row in frozen_per_label) != labels_tuple:
        raise RuntimeError(f"C5 frozen baseline label order changed: {case_id}")
    for value, frozen in zip(baseline_per_label, frozen_per_label, strict=True):
        if not math.isclose(float(value), float(frozen["dice"]), rel_tol=0.0, abs_tol=1e-12):
            raise RuntimeError(f"C5 baseline per-label Dice differs from frozen C4: {case_id}")

    arm_rows: list[dict[str, Any]] = []
    evaluated: dict[str, tuple[float, torch.Tensor]] = {}
    for arm in decision_case["arms"]:
        candidate = _load_field(decision, arm["candidate_field"]).to(device)
        warped = sample_at_psi(moving_seg_device.float(), candidate, mode="nearest").long()
        per_label = dice_per_label(warped, fixed_seg_device.long(), labels_tuple)
        candidate_dice = float(per_label.mean())
        arm_id = str(arm["arm_id"])
        evaluated[arm_id] = (candidate_dice, per_label)
        arm_rows.append(
            {
                "arm_index": arm["arm_index"],
                "arm_id": arm_id,
                "baseline_dice": baseline_dice,
                "candidate_dice": candidate_dice,
                "capacity_dice_delta": candidate_dice - baseline_dice,
                "historical_c4_dice_parity_verified": arm_id in HISTORICAL_ANCHOR_ARM_IDS,
                "per_label": _per_label_rows(
                    labels_tuple,
                    baseline_per_label,
                    per_label,
                    value_key="candidate_dice",
                ),
            }
        )

    selector_rows: list[dict[str, Any]] = []
    for selector in decision_case["selectors"]:
        selected = selector["selected_arm_id"]
        returned_dice, returned_per_label = (
            (baseline_dice, baseline_per_label) if selected is None else evaluated[selected]
        )
        selector_rows.append(
            {
                "selector_index": selector["selector_index"],
                "selector_id": selector["selector_id"],
                "action": selector["action"],
                "selected_arm_id": selected,
                "baseline_dice": baseline_dice,
                "returned_dice": returned_dice,
                "dice_delta": returned_dice - baseline_dice,
                "per_label": _per_label_rows(
                    labels_tuple,
                    baseline_per_label,
                    returned_per_label,
                    value_key="returned_dice",
                ),
            }
        )

    if sha256_file(decision_path) != expected_decision_sha:
        raise RuntimeError(f"C5 decision snapshot changed during evaluation: {case_id}")
    payload = {
        "schema": EVALUATION_CASE_SCHEMA,
        "protocol_id": PROTOCOL_ID,
        "status": "COMPLETE",
        "case_id": case_id,
        "decision_contract_sha256": decision_sha256,
        "decision_barrier_sha256": barrier_sha256,
        "evaluation_contract_sha256": evaluation_contract_sha256,
        "decision_case_sha256": expected_decision_sha,
        "labels_loaded_after_barrier": True,
        "test_split_accessed": False,
        "labels": list(labels_tuple),
        "baseline_c4_parity_verified": True,
        "arms": arm_rows,
        "selectors": selector_rows,
        "execution": dict(execution or {}),
    }
    marker.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(marker, payload)
    validate_evaluation_case_marker(
        payload,
        decision,
        decision_sha256,
        barrier,
        barrier_sha256,
        evaluation_contract,
        evaluation_contract_sha256,
        decision_case,
        expected_decision_sha,
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
    evaluation_contract: Mapping[str, Any],
    evaluation_contract_sha256: str,
    device: torch.device,
    execution: Mapping[str, Any] | None = None,
) -> Path:
    expected = decision["shards"].get(str(shard_index))
    if list(case_ids) != expected or str(physical_gpu) != str(decision["shard_to_physical_gpu"].get(str(shard_index))):
        raise RuntimeError("C5 evaluation worker does not match the frozen shard")
    if evaluation_contract.get("decision_barrier_sha256") != barrier_sha256:
        raise RuntimeError("C5 evaluation worker received a foreign post-barrier contract")
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
            evaluation_contract=evaluation_contract,
            evaluation_contract_sha256=evaluation_contract_sha256,
            device=device,
            execution=execution,
        )
    marker = _worker_path(run_root, "evaluation", attempt_id, shard_index)
    payload = {
        "schema": WORKER_SCHEMA,
        "protocol_id": PROTOCOL_ID,
        "status": "COMPLETE",
        "phase": "evaluation",
        "attempt_id": attempt_id,
        "shard_index": shard_index,
        "physical_gpu": str(physical_gpu),
        "case_ids": list(case_ids),
        "decision_contract_sha256": decision_sha256,
        "decision_barrier_sha256": barrier_sha256,
        "evaluation_contract_sha256": evaluation_contract_sha256,
        "labels_loaded": True,
        "test_split_accessed": False,
        "execution": dict(execution or {}),
    }
    marker.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(marker, payload)
    validate_worker_marker(
        payload,
        decision,
        decision_sha256,
        phase="evaluation",
        shard_index=shard_index,
        attempt_id=attempt_id,
        barrier_sha256=barrier_sha256,
        evaluation_contract_sha256=evaluation_contract_sha256,
    )
    return marker


def _csv_value(value: Any) -> Any:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return value


def _write_rows(path: Path, rows: list[dict[str, Any]], preferred: Sequence[str]) -> None:
    fields = [name for name in preferred if any(name in row for row in rows)]
    fields.extend(sorted({key for row in rows for key in row} - set(fields)))
    normalized = [{key: _csv_value(row.get(key, "")) for key in fields} for row in rows]
    atomic_write_text(path, rows_to_csv(fields, normalized))


def _contrast_vectors(
    capacity_deltas: Mapping[str, Sequence[float] | np.ndarray],
    candidate_dice: Mapping[str, Sequence[float] | np.ndarray],
    selector_deltas: Mapping[str, Sequence[float] | np.ndarray],
    selector_dice: Mapping[str, Sequence[float] | np.ndarray],
    primary_label_zero: Mapping[int, Sequence[float] | np.ndarray],
    primary_label_reference: Mapping[int, Sequence[float] | np.ndarray],
) -> dict[str, dict[str, np.ndarray]]:
    if tuple(capacity_deltas) != SELECTABLE_ARM_IDS or tuple(candidate_dice) != SELECTABLE_ARM_IDS:
        raise ValueError("C5 finalizer requires all arm vectors in frozen order")
    if tuple(selector_deltas) != SELECTOR_IDS or tuple(selector_dice) != SELECTOR_IDS:
        raise ValueError("C5 finalizer requires all selector vectors in frozen order")
    if tuple(primary_label_zero) != EVALUATION_LABEL_IDS:
        raise ValueError("C5 finalizer requires all IXI label vectors in frozen order")
    if tuple(primary_label_reference) != REGIONAL_REPAIR_LABEL_IDS:
        raise ValueError("C5 finalizer requires both risk-repair label vectors in frozen order")

    capacity = {
        contrast_id: np.asarray(capacity_deltas[contrast_id.split("::")[1]], dtype=np.float64)
        for contrast_id in CONTRAST_IDS_BY_FAMILY[CAPACITY_FAMILY_ID]
    }
    reference = np.asarray(candidate_dice[PRIMARY_REFERENCE_ARM_ID], dtype=np.float64)
    incremental = {
        contrast_id: np.asarray(candidate_dice[contrast_id.split("::")[1]], dtype=np.float64) - reference
        for contrast_id in CONTRAST_IDS_BY_FAMILY[INCREMENTAL_FAMILY_ID]
    }
    factor = factor_contrast_differences(capacity_deltas)
    marginal = {key: factor[key] for key in CONTRAST_IDS_BY_FAMILY[MARGINAL_FAMILY_ID]}
    interactions = {key: factor[key] for key in CONTRAST_IDS_BY_FAMILY[INTERACTION_FAMILY_ID]}
    selector_zero = {
        contrast_id: np.asarray(selector_deltas[contrast_id.split("::")[1]], dtype=np.float64)
        for contrast_id in CONTRAST_IDS_BY_FAMILY[SELECTOR_ZERO_FAMILY_ID]
    }
    selector_reference = {
        contrast_id: np.asarray(selector_dice[contrast_id.split("::")[1]], dtype=np.float64) - reference
        for contrast_id in CONTRAST_IDS_BY_FAMILY[SELECTOR_REFERENCE_FAMILY_ID]
    }
    regional_zero = {
        contrast_id: np.asarray(primary_label_zero[int(contrast_id.split("::")[1])], dtype=np.float64)
        for contrast_id in CONTRAST_IDS_BY_FAMILY[REGIONAL_ZERO_FAMILY_ID]
    }
    regional_reference = {
        contrast_id: np.asarray(primary_label_reference[int(contrast_id.split("::")[1])], dtype=np.float64)
        for contrast_id in CONTRAST_IDS_BY_FAMILY[REGIONAL_REFERENCE_FAMILY_ID]
    }
    output = {
        CAPACITY_FAMILY_ID: capacity,
        INCREMENTAL_FAMILY_ID: incremental,
        MARGINAL_FAMILY_ID: marginal,
        INTERACTION_FAMILY_ID: interactions,
        SELECTOR_ZERO_FAMILY_ID: selector_zero,
        SELECTOR_REFERENCE_FAMILY_ID: selector_reference,
        REGIONAL_ZERO_FAMILY_ID: regional_zero,
        REGIONAL_REFERENCE_FAMILY_ID: regional_reference,
    }
    if tuple(output) != INFERENCE_FAMILY_IDS:
        raise RuntimeError("C5 finalizer inference-family order changed")
    for family_id, rows in output.items():
        if tuple(rows) != CONTRAST_IDS_BY_FAMILY[family_id]:
            raise RuntimeError(f"C5 finalizer contrast order changed: {family_id}")
        if any(np.asarray(values).shape != (EXPECTED_CASE_COUNT,) for values in rows.values()):
            raise RuntimeError(f"C5 finalizer contrast accounting is not val-58: {family_id}")
    return output


def _summary_rows(
    vectors: Mapping[str, Mapping[str, Sequence[float] | np.ndarray]],
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    by_id: dict[str, dict[str, Any]] = {}
    rows: list[dict[str, Any]] = []
    for family_id in INFERENCE_FAMILY_IDS:
        summaries = simultaneous_paired_summaries(family_id, vectors[family_id])
        for contrast_id in CONTRAST_IDS_BY_FAMILY[family_id]:
            payload = asdict(summaries[contrast_id])
            by_id[contrast_id] = payload
            rows.append({"contrast_id": contrast_id, **payload})
    return by_id, rows


def _paired_summary(by_id: Mapping[str, Mapping[str, Any]], contrast_id: str) -> Any:
    from tools.analysis.search_gate_c5 import PairedSummary

    return PairedSummary(**dict(by_id[contrast_id]))


def finalize_c5(
    *,
    run_root: Path,
    decision: Mapping[str, Any],
    decision_sha256: str,
    barrier: Mapping[str, Any],
    barrier_sha256: str,
    evaluation_contract: Mapping[str, Any],
    evaluation_contract_sha256: str,
) -> dict[str, str]:
    root = run_root.resolve()
    decision_payloads: dict[str, dict[str, Any]] = {}
    evaluation_payloads: dict[str, dict[str, Any]] = {}
    for case_id in decision["case_ids"]:
        decision_path = _decision_case_path(root, case_id)
        evaluation_path = _evaluation_case_path(root, case_id)
        expected_decision_sha = barrier["decision_case_sha256"][case_id]
        if sha256_file(decision_path) != expected_decision_sha:
            raise RuntimeError(f"C5 decision snapshot changed before finalization: {case_id}")
        decision_case = _load_json(decision_path)
        evaluation_case = _load_json(evaluation_path)
        validate_decision_case_marker(
            decision_case,
            decision,
            decision_sha256,
            verify_heavy_bytes=True,
        )
        validate_evaluation_case_marker(
            evaluation_case,
            decision,
            decision_sha256,
            barrier,
            barrier_sha256,
            evaluation_contract,
            evaluation_contract_sha256,
            decision_case,
            expected_decision_sha,
        )
        decision_payloads[case_id] = decision_case
        evaluation_payloads[case_id] = evaluation_case

    per_arm: list[dict[str, Any]] = []
    per_selector: list[dict[str, Any]] = []
    per_arm_label: list[dict[str, Any]] = []
    per_selector_label: list[dict[str, Any]] = []
    diagnostic_rows: list[dict[str, Any]] = []
    resource_rows: list[dict[str, Any]] = []
    by_arm: dict[str, list[dict[str, Any]]] = {arm_id: [] for arm_id in SELECTABLE_ARM_IDS}
    by_selector: dict[str, list[dict[str, Any]]] = {selector_id: [] for selector_id in SELECTOR_IDS}
    for case_id in decision["case_ids"]:
        dec = decision_payloads[case_id]
        evaluation = evaluation_payloads[case_id]
        evaluated_arms = {row["arm_id"]: row for row in evaluation["arms"]}
        evaluated_selectors = {row["selector_id"]: row for row in evaluation["selectors"]}
        for arm in dec["arms"]:
            observed = evaluated_arms[arm["arm_id"]]
            row = {
                "case_id": case_id,
                "arm_index": arm["arm_index"],
                "arm_id": arm["arm_id"],
                "reach_id": arm["reach_id"],
                "stride_voxels": arm["stride_voxels"],
                "post_rms_amplitude": arm["post_rms_amplitude"],
                "centre_beta": arm["centre_beta"],
                "historical_anchor": arm["historical_anchor"],
                "exact_certified": arm["exact"]["certified"],
                "candidate_field": arm["candidate_field"],
                "proposal": arm["proposal"],
                "clip_rms_retention": arm["proposal"]["clip_rms_retention"],
                "clip_cosine": arm["proposal"]["clip_cosine"],
                "mathematical_sdlogj": _metric_value(
                    arm["geometry"], MATHEMATICAL_SDLOGJ_CROP2, f"{case_id}/{arm['arm_id']}"
                ),
                "mathematical_sdlogj_delta": arm["mathematical_sdlogj_delta"],
                "support_retention": arm["support"]["retention"],
                "ncc7_improvement": arm["utilities"]["ncc7"]["improvement"],
                "mind_d2_improvement": arm["utilities"]["mind_d2"]["improvement"],
                "ncc5_improvement": arm["utilities"]["ncc5"]["improvement"],
                "ncc9_improvement": arm["utilities"]["ncc9"]["improvement"],
                "ngf_improvement": arm["utilities"]["ngf"]["improvement"],
                **{key: value for key, value in observed.items() if key not in {"arm_index", "arm_id", "per_label"}},
            }
            per_arm.append(row)
            by_arm[arm["arm_id"]].append(row)
            for label_row in observed["per_label"]:
                per_arm_label.append({"case_id": case_id, "arm_id": arm["arm_id"], **label_row})
            for utility_id, utility in arm["utilities"].items():
                diagnostic_rows.append(
                    {
                        "case_id": case_id,
                        "arm_id": arm["arm_id"],
                        "utility_key": utility_id,
                        **utility,
                    }
                )
        for selector in dec["selectors"]:
            observed = evaluated_selectors[selector["selector_id"]]
            row = {
                "case_id": case_id,
                "selector_index": selector["selector_index"],
                "selector_id": selector["selector_id"],
                "action": selector["action"],
                "selected_arm_id": selector["selected_arm_id"],
                "eligible_arm_ids": selector["eligible_arm_ids"],
                **{
                    key: value
                    for key, value in observed.items()
                    if key not in {"selector_index", "selector_id", "per_label"}
                },
            }
            per_selector.append(row)
            by_selector[selector["selector_id"]].append(row)
            for label_row in observed["per_label"]:
                per_selector_label.append({"case_id": case_id, "selector_id": selector["selector_id"], **label_row})
        resource_rows.append({"case_id": case_id, **dec["resources"]})

    if any(len(rows) != EXPECTED_CASE_COUNT for rows in (*by_arm.values(), *by_selector.values())):
        raise RuntimeError("C5 finalizer does not have complete val-58 arm/selector accounting")
    capacity_deltas = {
        arm_id: np.asarray([row["capacity_dice_delta"] for row in rows], dtype=np.float64)
        for arm_id, rows in by_arm.items()
    }
    candidate_dice = {
        arm_id: np.asarray([row["candidate_dice"] for row in rows], dtype=np.float64) for arm_id, rows in by_arm.items()
    }
    selector_deltas = {
        selector_id: np.asarray([row["dice_delta"] for row in rows], dtype=np.float64)
        for selector_id, rows in by_selector.items()
    }
    selector_dice = {
        selector_id: np.asarray([row["returned_dice"] for row in rows], dtype=np.float64)
        for selector_id, rows in by_selector.items()
    }
    primary_labels = [row for row in per_selector_label if row["selector_id"] == PRIMARY_SELECTOR_ID]
    reference_labels = [row for row in per_arm_label if row["arm_id"] == PRIMARY_REFERENCE_ARM_ID]
    primary_label_zero = {
        label_id: np.asarray(
            [row["dice_delta"] for row in primary_labels if row["label"] == label_id], dtype=np.float64
        )
        for label_id in EVALUATION_LABEL_IDS
    }
    primary_label_reference = {
        label_id: np.asarray(
            [
                primary["returned_dice"] - reference["candidate_dice"]
                for primary, reference in zip(
                    (row for row in primary_labels if row["label"] == label_id),
                    (row for row in reference_labels if row["label"] == label_id),
                    strict=True,
                )
            ],
            dtype=np.float64,
        )
        for label_id in REGIONAL_REPAIR_LABEL_IDS
    }
    vectors = _contrast_vectors(
        capacity_deltas,
        candidate_dice,
        selector_deltas,
        selector_dice,
        primary_label_zero,
        primary_label_reference,
    )
    summaries, contrast_rows = _summary_rows(vectors)

    arm_evidence: list[ArmEvidence] = []
    arm_summary: list[dict[str, Any]] = []
    for arm_id in SELECTABLE_ARM_IDS:
        rows = by_arm[arm_id]
        retention = np.asarray([row["clip_rms_retention"] for row in rows], dtype=np.float64)
        incremental_id = f"incremental::{arm_id}::vs_{PRIMARY_REFERENCE_ARM_ID}"
        evidence = ArmEvidence(
            arm_id=arm_id,
            capacity_vs_zero=_paired_summary(summaries, f"capacity::{arm_id}::vs_zero"),
            incremental_vs_reference=(
                None if arm_id == PRIMARY_REFERENCE_ARM_ID else _paired_summary(summaries, incremental_id)
            ),
            all_work_units_complete=len(rows) == EXPECTED_CASE_COUNT,
            all_exact_certified=all(row["exact_certified"] is True for row in rows),
            amplitude_retention_median=float(np.median(retention)),
            amplitude_retention_cases_at_least_090=int((retention >= 0.90).sum()),
        )
        arm_evidence.append(evidence)
        assessment = assess_arm(evidence)
        arm_summary.append(
            {
                "arm_id": arm_id,
                "candidate_dice_mean": float(np.mean([row["candidate_dice"] for row in rows])),
                "baseline_dice_mean": float(np.mean([row["baseline_dice"] for row in rows])),
                "capacity": summaries[f"capacity::{arm_id}::vs_zero"],
                "incremental_vs_reference": None if arm_id == PRIMARY_REFERENCE_ARM_ID else summaries[incremental_id],
                "mathematical_sdlogj_mean": float(np.mean([row["mathematical_sdlogj"] for row in rows])),
                "clip_rms_retention_median": float(np.median(retention)),
                "clip_rms_retention_cases_at_least_090": int((retention >= 0.90).sum()),
                **asdict(assessment),
            }
        )

    selector_evidence: list[SelectorEvidence] = []
    selector_summary: list[dict[str, Any]] = []
    for selector_id in SELECTOR_IDS:
        rows = by_selector[selector_id]
        evidence = SelectorEvidence(
            selector_id=selector_id,
            vs_zero=_paired_summary(summaries, f"selector::{selector_id}::vs_zero"),
            vs_reference=_paired_summary(
                summaries,
                f"selector::{selector_id}::vs_{PRIMARY_REFERENCE_ARM_ID}",
            ),
            all_choices_complete=len(rows) == EXPECTED_CASE_COUNT,
            all_choices_contract_valid=True,
        )
        selector_evidence.append(evidence)
        selector_summary.append(
            {
                "selector_id": selector_id,
                "returned_dice_mean": float(np.mean([row["returned_dice"] for row in rows])),
                "baseline_dice_mean": float(np.mean([row["baseline_dice"] for row in rows])),
                "accepted_cases": sum(row["selected_arm_id"] is not None for row in rows),
                "vs_zero": summaries[f"selector::{selector_id}::vs_zero"],
                "vs_reference": summaries[f"selector::{selector_id}::vs_{PRIMARY_REFERENCE_ARM_ID}"],
            }
        )

    regional = RegionalEvidence(
        vs_zero=tuple(
            _paired_summary(summaries, f"label::{label_id}::{PRIMARY_SELECTOR_ID}::vs_zero")
            for label_id in EVALUATION_LABEL_IDS
        ),
        risk_vs_reference=tuple(
            _paired_summary(
                summaries,
                f"label::{label_id}::{PRIMARY_SELECTOR_ID}::vs_{PRIMARY_REFERENCE_ARM_ID}",
            )
            for label_id in REGIONAL_REPAIR_LABEL_IDS
        ),
    )
    marginal_summaries = tuple(
        _paired_summary(summaries, contrast_id) for contrast_id in CONTRAST_IDS_BY_FAMILY[MARGINAL_FAMILY_ID]
    )
    next_branch = asdict(
        select_next_branch(
            arm_evidence,
            selector_evidence,
            marginal_summaries,
            regional,
            integrity_passed=True,
        )
    )

    hypotheses = {
        "schema": f"ctcf-search-c5-hypotheses-{SCHEMA_VERSION}",
        "protocol_id": PROTOCOL_ID,
        "post_barrier_only": True,
        "test_115_authorized": False,
        "inference_families": {
            family_id: [row for row in contrast_rows if row["family_id"] == family_id]
            for family_id in INFERENCE_FAMILY_IDS
        },
    }
    scientific_identity = {
        "source_contract_sha256": decision["source_contract_sha256"],
        "decision_contract_sha256": decision_sha256,
        "full_policy_sha256": decision["full_policy_sha256"],
        "decision_policy_sha256": decision["decision_policy_sha256"],
        "arm_specs_sha256": decision["arm_specs_sha256"],
        "selector_specs_sha256": decision["selector_specs_sha256"],
        "offset_table_sha256": decision["offset_table_sha256"],
        "support_contract_sha256": decision["support_contract_sha256"],
        "contrast_contract_sha256": decision["contrast_contract_sha256"],
        "decision_barrier_sha256": barrier_sha256,
        "evaluation_contract_sha256": evaluation_contract_sha256,
    }
    design_counts = {
        "arms": len(SELECTABLE_ARM_IDS),
        "selectors": len(SELECTOR_IDS),
        "inference_families": len(INFERENCE_FAMILY_IDS),
        "cases": EXPECTED_CASE_COUNT,
        "c4_anchor_references_reused": EXPECTED_CASE_COUNT * len(HISTORICAL_ANCHOR_ARM_IDS),
        "new_c5_heavy_fields": EXPECTED_CASE_COUNT * (len(SELECTABLE_ARM_IDS) - len(HISTORICAL_ANCHOR_ARM_IDS)),
    }
    summary = {
        "schema": f"ctcf-search-c5-summary-{SCHEMA_VERSION}",
        "protocol_id": PROTOCOL_ID,
        "status": "COMPLETE",
        "n_cases": EXPECTED_CASE_COUNT,
        "test_115_authorized": False,
        "test_split_accessed": False,
        "labels_used_for_decision": False,
        "decision_barrier_sha256": barrier_sha256,
        "evaluation_contract_sha256": evaluation_contract_sha256,
        "scientific_identity": scientific_identity,
        "design_counts": design_counts,
        "primary_reference_arm_id": PRIMARY_REFERENCE_ARM_ID,
        "primary_selector_id": PRIMARY_SELECTOR_ID,
        "next_branch": next_branch,
        "arm_summaries": arm_summary,
        "selector_summaries": selector_summary,
    }
    paths = {
        "per_arm": root / "per_arm.csv",
        "per_selector": root / "per_selector.csv",
        "per_arm_label_dice": root / "per_arm_label_dice.csv",
        "per_selector_label_dice": root / "per_selector_label_dice.csv",
        "diagnostic_utilities": root / "diagnostic_utilities.csv",
        "arm_summary": root / "arm_summary.csv",
        "selector_summary": root / "selector_summary.csv",
        "preregistered_contrasts": root / "preregistered_contrasts.csv",
        "resource_summary": root / "resource_summary.csv",
        "hypotheses": root / "hypotheses.json",
        "summary": root / "summary.json",
        "next_branch": root / "next_branch.json",
    }
    _write_rows(paths["per_arm"], per_arm, ("case_id", "arm_index", "arm_id"))
    _write_rows(paths["per_selector"], per_selector, ("case_id", "selector_index", "selector_id"))
    _write_rows(paths["per_arm_label_dice"], per_arm_label, ("case_id", "arm_id", "label"))
    _write_rows(paths["per_selector_label_dice"], per_selector_label, ("case_id", "selector_id", "label"))
    _write_rows(paths["diagnostic_utilities"], diagnostic_rows, ("case_id", "arm_id", "utility_key"))
    _write_rows(paths["arm_summary"], arm_summary, ("arm_id", "candidate_dice_mean"))
    _write_rows(paths["selector_summary"], selector_summary, ("selector_id", "returned_dice_mean"))
    _write_rows(paths["preregistered_contrasts"], contrast_rows, ("family_id", "contrast_id", "mean"))
    _write_rows(paths["resource_summary"], resource_rows, ("case_id", "elapsed_sec", "peak_gpu_bytes"))
    atomic_write_json(paths["hypotheses"], hypotheses)
    atomic_write_json(paths["summary"], summary)
    atomic_write_json(paths["next_branch"], next_branch)

    artifact_hashes = {name: sha256_file(path) for name, path in paths.items()}
    manifest = {
        "schema": f"ctcf-search-c5-run-manifest-{SCHEMA_VERSION}",
        "protocol_id": PROTOCOL_ID,
        "status": "COMPLETE",
        "source_contract_sha256": decision["source_contract_sha256"],
        "decision_contract_sha256": decision_sha256,
        "decision_barrier_sha256": barrier_sha256,
        "evaluation_contract_sha256": evaluation_contract_sha256,
        "full_policy_sha256": decision["full_policy_sha256"],
        "decision_policy_sha256": decision["decision_policy_sha256"],
        "arm_specs_sha256": decision["arm_specs_sha256"],
        "selector_specs_sha256": decision["selector_specs_sha256"],
        "offset_table_sha256": decision["offset_table_sha256"],
        "support_contract_sha256": decision["support_contract_sha256"],
        "contrast_contract_sha256": decision["contrast_contract_sha256"],
        "design_counts": design_counts,
        "test_115_authorized": False,
        "test_split_accessed": False,
        "decision_case_sha256": {case_id: barrier["decision_case_sha256"][case_id] for case_id in decision["case_ids"]},
        "evaluation_case_sha256": {
            case_id: sha256_file(_evaluation_case_path(root, case_id)) for case_id in decision["case_ids"]
        },
        "files": artifact_hashes,
        "next_branch": next_branch,
    }
    manifest_path = root / "c5_manifest.json"
    atomic_write_json(manifest_path, manifest)
    artifact_hashes["c5_manifest"] = sha256_file(manifest_path)
    return artifact_hashes
