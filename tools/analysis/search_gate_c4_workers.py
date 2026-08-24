from __future__ import annotations

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
from tools.analysis.search_gate_c3 import (
    NCC_DENOMINATOR_EPS,
    build_common_support,
    metric_envelope,
    primary_ncc_decision,
)
from tools.analysis.search_gate_c4 import (
    ARM_SPECS,
    ARM_SPECS_BY_ID,
    BASELINE_DICE_PARITY_ATOL,
    COMMON_EVIDENCE_COLLAR,
    CONTRAST_SPECS,
    EXACT_CLAIM_EPS,
    EXPECTED_CASE_COUNT,
    LOCAL_CLIP_SWEEPS,
    MAIN_ARM_SPECS,
    POST_SMOOTHING_PASSES,
    PRIMARY_NCC_IMPROVEMENT_MIN,
    PRIMARY_NCC_WINDOW,
    PRIMARY_UTILITY_ID,
    PROTOCOL_ID,
    RMS_TARGET_SOURCE_ID,
    SCIENTIFIC_REFERENCE_ARM_ID,
    SELECTABLE_ARM_IDS,
    STANDARDIZATION_FLOOR,
    SUPPORT_RETENTION_MIN,
    WORK_EPS,
    ArmEvidence,
    GeometryComparison,
    PairedSummary,
    assess_arm,
    materially_strong_policy,
    select_next_branch,
    simultaneous_paired_summaries,
)
from tools.analysis.search_gate_c4_contracts import (
    DECISION_CASE_SCHEMA,
    EVALUATION_CASE_SCHEMA,
    PERSISTENCE_PROTOCOL_ID,
    WORKER_SCHEMA,
    array_sha256,
    validate_decision_case_marker,
    validate_evaluation_case_marker,
    validate_worker_marker,
)
from tools.analysis.search_gate_metrics import (
    DIGITAL_DECOMPOSITION,
    LEARN2REG_SHIFTED_SDLOGJ_MASKED,
    MATHEMATICAL_SDLOGJ_CROP2,
    METRIC_SPECS,
    compute_metric,
)
from tools.analysis.search_gate_multiscale import (
    OFFSETS_STRIDE1,
    OFFSETS_STRIDE2,
    CenteredCostVolume,
    DecodedProposal,
    MindFeaturePair,
    PosteriorVolume,
    build_c4_common_support,
    build_mind_feature_pair,
    build_raw_intensity_cost_volume,
    build_raw_mind_cost_volume_from_features,
    centered_standardize,
    decode_posterior_mean,
    duplicate_fusion_diagnostic,
    fuse_standardized_costs,
    posterior_from_standardized_costs,
    postprocess_and_match_rms,
    scale_agreement_diagnostics,
)
from tools.analysis.search_gate_runtime import save_reload_certify
from tools.analysis.transactional_search import (
    build_proposal,
    certified_local_clip_candidate,
    geometry_mask,
    load_flow_npz,
    masked_zscore,
    mind_ssc,
    ncc_loss_from_normalized,
    sample_at_psi,
    valid_sample_mask,
)
from utils import dice_per_label

SCHEMA_VERSION = "v1"


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"{path}: expected a JSON object")
    return payload


def _field_record(path: Path, heavy_root: Path, digest: str) -> dict[str, Any]:
    return {
        "relative_path": path.resolve().relative_to(heavy_root.resolve()).as_posix(),
        "npz_sha256": sha256_file(path),
        "array_sha256": digest,
    }


def _resolve_field(root: Path, record: Mapping[str, Any], *, verify_array: bool = True) -> Path:
    relative = Path(str(record.get("relative_path", "")))
    if relative.is_absolute() or ".." in relative.parts:
        raise RuntimeError("C4 field path must be relative and traversal-free")
    resolved_root = root.resolve()
    path = (resolved_root / relative).resolve()
    if resolved_root != path and resolved_root not in path.parents:
        raise RuntimeError("C4 field escapes its declared heavy root")
    if not path.is_file() or sha256_file(path) != record.get("npz_sha256"):
        raise RuntimeError(f"C4 field bytes changed or are missing: {path}")
    if verify_array:
        array = load_flow_npz(path).cpu().numpy()
        if array_sha256(array) != record.get("array_sha256"):
            raise RuntimeError(f"C4 field array changed: {path}")
    return path


def _load_image(record: Mapping[str, Any]) -> np.ndarray:
    path = Path(str(record.get("path", ""))).resolve()
    if not path.is_file() or path.stat().st_size != int(record.get("bytes", -1)):
        raise RuntimeError(f"C4 image cache is missing or has the wrong size: {path}")
    if sha256_file(path) != record.get("sha256"):
        raise RuntimeError(f"C4 image cache bytes changed: {path}")
    array = np.load(path, allow_pickle=False)
    if (
        array.dtype != np.float32
        or list(array.shape) != record.get("shape")
        or not np.isfinite(array).all()
        or array_sha256(array) != record.get("array_sha256")
    ):
        raise RuntimeError(f"C4 image cache array changed or is invalid: {path}")
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


def _assert_exact_geometry(bundle: Mapping[str, Mapping[str, Any]], *, label: str) -> None:
    invalid = [metric_id for metric_id, row in bundle.items() if row.get("status") != "OK"]
    if invalid:
        raise RuntimeError(f"Exact C4 field has undefined geometry metrics for {label}: {invalid}")
    digital = bundle[DIGITAL_DECOMPOSITION]
    corner = float((digital.get("components") or {}).get("corner_union_violation_fraction", math.nan))
    if not math.isfinite(corner) or corner > 0.0:
        raise RuntimeError(f"Exact C4 certificate disagrees with digital corner determinants for {label}")


def _mean_on(tensor: torch.Tensor, mask: torch.Tensor) -> float:
    values = tensor.masked_select(mask)
    if values.numel() == 0 or not bool(torch.isfinite(values).all()):
        raise RuntimeError("C4 diagnostic support is empty or non-finite")
    return float(values.double().mean().item())


def _posterior_record(volume: CenteredCostVolume, posterior: PosteriorVolume, mask: torch.Tensor) -> dict[str, Any]:
    support = mask & volume.valid.all(dim=1, keepdim=True)
    logits = volume.standardized_costs.masked_fill(~volume.valid, torch.inf)
    top2 = (-logits).topk(k=2, dim=1).values
    return {
        "active_voxels": int(support.sum(dtype=torch.int64).item()),
        "entropy_mean": _mean_on(posterior.entropy, support),
        "confidence_mean": _mean_on(posterior.confidence, support),
        "top1_top2_gap_mean": _mean_on(top2[:, 0:1] - top2[:, 1:2], support),
        "floor_hit_fraction": _mean_on(volume.floor_hit.float(), mask),
        "candidate_cost_mean": _mean_on(volume.cost_mean, mask),
        "candidate_cost_std_mean": _mean_on(volume.cost_std, mask),
        "work": asdict(volume.work),
    }


def _same_index_reach_diagnostic(
    arm_id: str,
    left: tuple[CenteredCostVolume, PosteriorVolume, torch.Tensor],
    right: tuple[CenteredCostVolume, PosteriorVolume, torch.Tensor],
    mask: torch.Tensor,
) -> dict[str, Any]:
    left_volume, left_posterior, left_residual = left
    right_volume, right_posterior, right_residual = right
    common = mask & left_volume.valid.all(dim=1, keepdim=True) & right_volume.valid.all(dim=1, keepdim=True)
    if not bool(common.any()):
        raise RuntimeError(f"C4 reach diagnostic has empty support: {arm_id}")
    left_argmin = left_volume.standardized_costs.masked_fill(~left_volume.valid, torch.inf).argmin(1, keepdim=True)
    right_argmin = right_volume.standardized_costs.masked_fill(~right_volume.valid, torch.inf).argmin(1, keepdim=True)
    p, q = left_posterior.probabilities, right_posterior.probabilities
    midpoint = 0.5 * (p + q)
    tiny = torch.finfo(p.dtype).tiny
    js = 0.5 * (
        torch.where(p > 0, p * (p.clamp_min(tiny).log() - midpoint.clamp_min(tiny).log()), 0.0).sum(1, keepdim=True)
        + torch.where(q > 0, q * (q.clamp_min(tiny).log() - midpoint.clamp_min(tiny).log()), 0.0).sum(1, keepdim=True)
    )
    left_norm = left_residual.square().sum(1, keepdim=True).sqrt()
    right_norm = right_residual.square().sum(1, keepdim=True).sqrt()
    nonzero = common & (left_norm > 0) & (right_norm > 0)
    cosine: float | None = None
    if bool(nonzero.any()):
        cosine_map = (left_residual * right_residual).sum(1, keepdim=True) / (left_norm * right_norm).clamp_min(tiny)
        cosine = _mean_on(cosine_map, nonzero)
    return {
        "comparison_id": f"reach::{arm_id}::S1_vs_S2",
        "comparison_kind": "normalized_offset_index_across_physical_reach",
        "active_voxels": int(common.sum(dtype=torch.int64).item()),
        "argmin_index_agreement": _mean_on((left_argmin == right_argmin).float(), common),
        "posterior_js_divergence_mean": _mean_on(js, common),
        "residual_cosine_mean": cosine,
    }


def _source_c3_raw_reference(
    decision: Mapping[str, Any], case_id: str, initial: torch.Tensor, device: torch.device
) -> tuple[torch.Tensor, dict[str, Any]]:
    historical = decision["source_historical"][case_id]
    field_record = historical["raw_conf_requested_field"]
    path = _resolve_field(Path(decision["source_c3_heavy_root"]), field_record)
    state = load_flow_npz(path).to(device)
    residual = (state - initial).float()
    return residual, {
        "field": field_record,
        "source_decision_case_sha256": historical["source_decision_case_sha256"],
    }


def _legacy_parity(
    fixed: torch.Tensor,
    moving: torch.Tensor,
    initial: torch.Tensor,
    source_residual: torch.Tensor,
) -> dict[str, Any]:
    mask4 = geometry_mask(tuple(initial.shape[-3:]), 4, initial.device)
    fixed_norm = masked_zscore(fixed, mask4)
    moving_norm = masked_zscore(moving, mask4)
    fixed_mind = mind_ssc(fixed_norm, radius=1, dilation=2)
    moving_mind = mind_ssc(moving_norm, radius=1, dilation=2)
    proposal = build_proposal(
        fixed,
        moving,
        initial,
        mask4,
        feature="mind",
        orientation="target_centered",
        collar_width=4,
        mind_radius=1,
        mind_dilation=2,
        fixed_feature_override=fixed_mind,
        moving_feature_override=moving_mind,
    )
    observed = (initial + 2.0 * proposal.displacement).float()
    expected = (initial + source_residual).float()
    observed_sha = array_sha256(observed.detach().cpu().numpy())
    expected_sha = array_sha256(expected.detach().cpu().numpy())
    max_abs = float((observed - expected).abs().max().item())
    if observed_sha != expected_sha or max_abs != 0.0:
        raise RuntimeError(f"C4 legacy control does not reproduce frozen C3 raw_conf_post1: max_abs={max_abs}")
    return {
        "historical_protocol_id": "CTCF-SEARCH-GATE-C3A-V1",
        "observed_array_sha256": observed_sha,
        "expected_array_sha256": expected_sha,
        "max_abs_difference": max_abs,
        "byte_identical": True,
    }


def _utility_and_action(
    fixed_norm: torch.Tensor,
    moving_norm: torch.Tensor,
    initial: torch.Tensor,
    candidate: torch.Tensor,
    mask: torch.Tensor,
    *,
    exact_certified: bool,
    support_contract: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    frozen_window = support_contract.get("utility_window", support_contract.get("window"))
    if (
        support_contract.get("utility_id") != PRIMARY_UTILITY_ID
        or frozen_window != PRIMARY_NCC_WINDOW
        or float(support_contract.get("utility_retention_min", math.nan)) != SUPPORT_RETENTION_MIN
        or float(support_contract.get("improvement_min", math.nan)) != PRIMARY_NCC_IMPROVEMENT_MIN
    ):
        raise RuntimeError("C4 decision support disagrees with the frozen NCC7 policy")
    support = build_common_support(
        mask,
        valid_sample_mask(initial),
        valid_sample_mask(candidate),
        window=PRIMARY_NCC_WINDOW,
        utility_id=PRIMARY_UTILITY_ID,
    )
    baseline_loss = ncc_loss_from_normalized(
        fixed_norm,
        moving_norm,
        initial,
        support.pair_mask,
        win=PRIMARY_NCC_WINDOW,
        eps=NCC_DENOMINATOR_EPS,
    )
    candidate_loss = ncc_loss_from_normalized(
        fixed_norm,
        moving_norm,
        candidate,
        support.pair_mask,
        win=PRIMARY_NCC_WINDOW,
        eps=NCC_DENOMINATOR_EPS,
    )
    decision = primary_ncc_decision(
        exact_certified=exact_certified,
        support_retention=support.retention,
        baseline_ncc_loss=baseline_loss,
        candidate_ncc_loss=candidate_loss,
    )
    return (
        {
            "support_id": support_contract["support_id"],
            "baseline_count": support.baseline_count,
            "pair_count": support.pair_count,
            "retention": support.retention,
        },
        {
            "utility_id": support.utility_id,
            "baseline_loss": baseline_loss,
            "candidate_loss": candidate_loss,
            "improvement": baseline_loss - candidate_loss,
        },
        decision.to_dict(),
    )


def _materialize_arm(
    *,
    case_id: str,
    arm: Any,
    decoded: DecodedProposal,
    initial: torch.Tensor,
    source_rms_reference: torch.Tensor,
    mask: torch.Tensor,
    fixed_norm: torch.Tensor,
    moving_norm: torch.Tensor,
    heavy_root: Path,
    support_contract: Mapping[str, Any],
    support_contract_sha256: str,
    proposal_diagnostics: Mapping[str, Any],
) -> dict[str, Any]:
    post = postprocess_and_match_rms(
        decoded,
        mask,
        proposal_multiplier=arm.pre_rms_multiplier,
        smoothing_passes=POST_SMOOTHING_PASSES,
        collar_width=COMMON_EVIDENCE_COLLAR,
        rms_reference=source_rms_reference,
    )
    candidate_raw, operator = certified_local_clip_candidate(
        initial,
        post.displacement,
        mask,
        work_eps=WORK_EPS,
        sweeps=LOCAL_CLIP_SWEEPS,
    )
    path = heavy_root / "cases" / case_id / "arms" / f"{arm.arm_id}.npz"
    stored, exact = save_reload_certify(candidate_raw, path, EXACT_CLAIM_EPS)
    certified = exact.get("status") == "CERTIFIED" and exact.get("certified") is True
    if not certified:
        raise RuntimeError(f"C4 save/reload exact certificate failed: {case_id}/{arm.arm_id}")
    candidate = stored.to(initial.device)
    field = _field_record(path, heavy_root, str(exact["sha256"]))
    geometry = _geometry_bundle(candidate, mask)
    _assert_exact_geometry(geometry, label=f"{case_id}/{arm.arm_id}")
    support, utility, primary = _utility_and_action(
        fixed_norm,
        moving_norm,
        initial,
        candidate,
        mask,
        exact_certified=certified,
        support_contract=support_contract,
    )
    action = "ACCEPT" if primary["accept"] else "ROLLBACK"
    return {
        "arm_index": arm.arm_index,
        "arm_id": arm.arm_id,
        "role": arm.role,
        "selectable": arm.selectable,
        "diagnostic_only": arm.diagnostic_only,
        "action": action,
        "reason": primary["reason"],
        "support_contract_sha256": support_contract_sha256,
        "proposal": {
            **dict(proposal_diagnostics),
            "proposal_multiplier": post.proposal_multiplier,
            "smoothing_passes": post.smoothing_passes,
            "collar_width": post.collar_width,
            "rms_target_source_id": RMS_TARGET_SOURCE_ID,
            "rms_source": post.source_rms,
            "rms_target": post.target_rms,
            "rms_output": post.output_rms,
            "rms_scale_factor": post.rms_scale_factor,
        },
        "operator": operator,
        "candidate_field": field,
        "persistence": {
            "protocol_id": PERSISTENCE_PROTOCOL_ID,
            "saved_npz_sha256": field["npz_sha256"],
            "reloaded_array_sha256": field["array_sha256"],
        },
        "exact": exact,
        "geometry": geometry,
        "support": support,
        "utility": utility,
        "primary_decision": primary,
        "returned_field": field if action == "ACCEPT" else None,
        "rollback_to_source_initial": action == "ROLLBACK",
    }


def _bank(
    fixed_norm: torch.Tensor,
    moving_norm: torch.Tensor,
    initial: torch.Tensor,
    mask: torch.Tensor,
    *,
    dilation: int | None,
    stride: int,
    mind_features: MindFeaturePair | None = None,
) -> tuple[CenteredCostVolume, PosteriorVolume, DecodedProposal, float]:
    started = time.perf_counter()
    offsets = OFFSETS_STRIDE1 if stride == 1 else OFFSETS_STRIDE2
    if dilation is None:
        raw = build_raw_intensity_cost_volume(
            fixed_norm,
            moving_norm,
            initial,
            mask,
            offsets=offsets,
            cost_id=f"intensity_s{stride}",
        )
    else:
        if mind_features is None or mind_features.dilation != dilation:
            raise ValueError("C4 MIND bank requires its matching reusable feature pair")
        raw = build_raw_mind_cost_volume_from_features(
            mind_features,
            initial,
            mask,
            offsets=offsets,
            cost_id=f"mind_d{dilation}_s{stride}",
        )
    volume = centered_standardize(raw, standardization_floor=STANDARDIZATION_FLOOR)
    posterior = posterior_from_standardized_costs(volume)
    decoded = decode_posterior_mean(posterior)
    return volume, posterior, decoded, time.perf_counter() - started


def _decision_case_path(run_root: Path, case_id: str) -> Path:
    return run_root / "cases" / case_id / "decision_complete.json"


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
    marker = _decision_case_path(run_root.resolve(), case_id)
    if marker.is_file():
        validate_decision_case_marker(_load_json(marker), decision, decision_sha256, verify_heavy_bytes=True)
        return marker
    if case_id not in decision["shards"].get(str(shard_index), []):
        raise RuntimeError(f"C4 decision case is assigned to another shard: {case_id}")
    started = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    atlas = _load_image(decision["image_inputs"]["atlas"])
    case = _load_image(decision["image_inputs"][case_id])
    moving = torch.from_numpy(atlas).unsqueeze(0).to(device)
    fixed = torch.from_numpy(case).unsqueeze(0).to(device)
    source_heavy = Path(decision["source_c3_heavy_root"])
    initial_record = decision["source_initial"][case_id]["field"]
    initial = load_flow_npz(_resolve_field(source_heavy, initial_record)).to(device)
    mask7 = geometry_mask(tuple(initial.shape[-3:]), COMMON_EVIDENCE_COLLAR, device)
    common = build_c4_common_support(initial, mask7)
    fixed_norm = masked_zscore(fixed, mask7)
    moving_norm = masked_zscore(moving, mask7)
    source_residual, source_reference = _source_c3_raw_reference(decision, case_id, initial, device)
    legacy = _legacy_parity(fixed, moving, initial, source_residual)

    banks: dict[str, tuple[CenteredCostVolume, PosteriorVolume, DecodedProposal, float]] = {}
    mind_features: dict[int, MindFeaturePair] = {}
    mind_feature_elapsed: dict[int, float] = {}
    for dilation in (1, 2, 4):
        feature_started = time.perf_counter()
        mind_features[dilation] = build_mind_feature_pair(
            fixed_norm,
            moving_norm,
            dilation=dilation,
            feature_id=f"mind_d{dilation}",
        )
        mind_feature_elapsed[dilation] = time.perf_counter() - feature_started
    for stride in (1, 2):
        for dilation in (1, 2, 4):
            key = f"mind_d{dilation}_s{stride}"
            banks[key] = _bank(
                fixed_norm,
                moving_norm,
                initial,
                common.common_mask,
                dilation=dilation,
                stride=stride,
                mind_features=mind_features[dilation],
            )
        components = [banks[f"mind_d{dilation}_s{stride}"][0] for dilation in (1, 2, 4)]
        started_fusion = time.perf_counter()
        fused = fuse_standardized_costs(components, cost_id=f"mind_f124_s{stride}")
        fused_posterior = posterior_from_standardized_costs(fused)
        banks[f"mind_f124_s{stride}"] = (
            fused,
            fused_posterior,
            decode_posterior_mean(fused_posterior),
            time.perf_counter() - started_fusion,
        )

    intensity: dict[str, tuple[CenteredCostVolume, PosteriorVolume, DecodedProposal, float]] = {}
    for stride in (1, 2):
        intensity[f"intensity_s{stride}"] = _bank(
            fixed_norm,
            moving_norm,
            initial,
            common.common_mask,
            dilation=None,
            stride=stride,
        )

    scale_rows: list[dict[str, Any]] = []
    for stride in (1, 2):
        for left_id, right_id in (("d1", "d2"), ("d2", "d4"), ("d1", "d4"), ("f124", "d2")):
            left = banks[f"mind_{left_id}_s{stride}"]
            right = banks[f"mind_{right_id}_s{stride}"]
            row = asdict(
                scale_agreement_diagnostics(
                    left[0],
                    right[0],
                    left[1],
                    right[1],
                    left[2].displacement,
                    right[2].displacement,
                    common.common_mask,
                )
            )
            row.update(
                comparison_id=f"descriptor::{left_id}_vs_{right_id}::s{stride}",
                comparison_kind="descriptor_context_at_fixed_physical_reach",
            )
            scale_rows.append(row)
    for descriptor in ("d1", "d2", "d4", "f124"):
        scale_rows.append(
            _same_index_reach_diagnostic(
                descriptor,
                (
                    banks[f"mind_{descriptor}_s1"][0],
                    banks[f"mind_{descriptor}_s1"][1],
                    banks[f"mind_{descriptor}_s1"][2].displacement,
                ),
                (
                    banks[f"mind_{descriptor}_s2"][0],
                    banks[f"mind_{descriptor}_s2"][1],
                    banks[f"mind_{descriptor}_s2"][2].displacement,
                ),
                common.common_mask,
            )
        )

    heavy_root = Path(decision["heavy_root"]).resolve()
    arm_rows: list[dict[str, Any]] = []
    for arm in MAIN_ARM_SPECS:
        volume, posterior, decoded, elapsed = banks[arm.arm_id]
        diagnostics = _posterior_record(volume, posterior, common.common_mask)
        diagnostics["elapsed_sec"] = elapsed
        arm_rows.append(
            _materialize_arm(
                case_id=case_id,
                arm=arm,
                decoded=decoded,
                initial=initial,
                source_rms_reference=source_residual,
                mask=mask7,
                fixed_norm=fixed_norm,
                moving_norm=moving_norm,
                heavy_root=heavy_root,
                support_contract=decision["support_contract"],
                support_contract_sha256=decision["support_contract_sha256"],
                proposal_diagnostics=diagnostics,
            )
        )

    legacy_spec = ARM_SPECS_BY_ID["legacy_mind_d2_s1_collar4"]
    arm_rows.append(
        {
            "arm_index": legacy_spec.arm_index,
            "arm_id": legacy_spec.arm_id,
            "role": legacy_spec.role,
            "selectable": False,
            "diagnostic_only": True,
            "action": "DIAGNOSTIC_ONLY",
            "reason": "HISTORICAL_C3_PARITY_ONLY",
            "support_contract_sha256": decision["support_contract_sha256"],
            "diagnostics": legacy,
        }
    )
    d2 = banks["mind_d2_s1"][0]
    f222 = fuse_standardized_costs((d2, d2, d2), cost_id="mind_f222_s1")
    duplicate = duplicate_fusion_diagnostic(d2, f222, common.common_mask)
    f222_spec = ARM_SPECS_BY_ID["mind_f222_s1"]
    arm_rows.append(
        {
            "arm_index": f222_spec.arm_index,
            "arm_id": f222_spec.arm_id,
            "role": f222_spec.role,
            "selectable": False,
            "diagnostic_only": True,
            "action": "DIAGNOSTIC_ONLY",
            "reason": "DUPLICATE_FUSION_IDEMPOTENCE_ONLY",
            "support_contract_sha256": decision["support_contract_sha256"],
            "diagnostics": asdict(duplicate),
        }
    )
    for arm_id in ("intensity_s1", "intensity_s2"):
        arm = ARM_SPECS_BY_ID[arm_id]
        volume, posterior, decoded, elapsed = intensity[arm_id]
        diagnostics = _posterior_record(volume, posterior, common.common_mask)
        diagnostics["elapsed_sec"] = elapsed
        arm_rows.append(
            _materialize_arm(
                case_id=case_id,
                arm=arm,
                decoded=decoded,
                initial=initial,
                source_rms_reference=source_residual,
                mask=mask7,
                fixed_norm=fixed_norm,
                moving_norm=moving_norm,
                heavy_root=heavy_root,
                support_contract=decision["support_contract"],
                support_contract_sha256=decision["support_contract_sha256"],
                proposal_diagnostics=diagnostics,
            )
        )
    arm_rows.sort(key=lambda row: int(row["arm_index"]))

    baseline_geometry = _geometry_bundle(initial, mask7)
    _assert_exact_geometry(baseline_geometry, label=f"{case_id}/source_initial")
    payload = {
        "schema": DECISION_CASE_SCHEMA,
        "protocol_id": PROTOCOL_ID,
        "status": "COMPLETE",
        "case_id": case_id,
        "decision_contract_sha256": decision_sha256,
        "shard_index": shard_index,
        "physical_gpu": physical_gpu,
        "arm_specs_sha256": decision["arm_specs_sha256"],
        "offset_table_sha256": decision["offset_table_sha256"],
        "support_contract_sha256": decision["support_contract_sha256"],
        "labels_loaded_to_device": False,
        "test_split_accessed": False,
        "source_image_array_sha256": decision["image_inputs"][case_id]["array_sha256"],
        "source_initial_array_sha256": initial_record["array_sha256"],
        "source_reference": source_reference,
        "support": {
            "geometry_count": common.geometry_count,
            "common_count": common.common_count,
            "retention": common.retention,
        },
        "baseline_geometry": baseline_geometry,
        "arms": arm_rows,
        "scale_agreement": scale_rows,
        "resources": {
            "elapsed_sec": time.perf_counter() - started,
            "peak_gpu_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0,
            "mind_feature_elapsed_sec": {str(key): value for key, value in mind_feature_elapsed.items()},
            "mind_descriptor_evaluations": sum(value.work.descriptor_evaluations for value in mind_features.values()),
        },
        "execution": dict(execution or {}),
    }
    _assert_decision_label_free(payload)
    marker.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(marker, payload)
    validate_decision_case_marker(payload, decision, decision_sha256, verify_heavy_bytes=True)
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return marker


def _assert_decision_label_free(payload: Mapping[str, Any]) -> None:
    allowed_label_flags = {
        "labels_available_to_decision_workers",
        "labels_loaded_to_device",
        "decision_contract_contains_label_data",
    }
    stack: list[Any] = [payload]
    while stack:
        value = stack.pop()
        if isinstance(value, Mapping):
            for key, child in value.items():
                lowered = str(key).lower()
                if "dice" in lowered or "segmentation" in lowered:
                    raise RuntimeError("C4 decision payload leaked label-derived or raw-container data")
                if "label" in lowered and lowered not in allowed_label_flags:
                    raise RuntimeError("C4 decision payload leaked label-derived or raw-container data")
                if lowered in allowed_label_flags and child is not False:
                    raise RuntimeError("C4 decision payload leaked label-derived or raw-container data")
                stack.append(child)
        elif isinstance(value, (list, tuple)):
            stack.extend(value)
        elif isinstance(value, str) and ("segmentation" in value.lower() or value.lower().endswith(".pkl")):
            raise RuntimeError("C4 decision payload leaked label-derived or raw-container data")


def _worker_path(run_root: Path, phase: str, attempt_id: str, shard_index: int) -> Path:
    return run_root / "workers" / phase / "attempts" / attempt_id / f"worker_{shard_index:02d}.json"


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
    if list(case_ids) != expected or physical_gpu != decision["shard_to_physical_gpu"].get(str(shard_index)):
        raise RuntimeError("C4 decision worker does not match the frozen shard")
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
    marker = _worker_path(run_root.resolve(), "decision", attempt_id, shard_index)
    payload = {
        "schema": WORKER_SCHEMA,
        "protocol_id": PROTOCOL_ID,
        "status": "COMPLETE",
        "phase": "decision",
        "attempt_id": attempt_id,
        "shard_index": shard_index,
        "physical_gpu": physical_gpu,
        "case_ids": list(case_ids),
        "decision_contract_sha256": decision_sha256,
        "labels_loaded_to_device": False,
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
    return run_root / "cases" / case_id / "evaluation_complete.json"


def _source_c3_baseline(source: Mapping[str, Any], case_id: str) -> float:
    value = (source.get("evaluation_baseline_dice") or {}).get(case_id)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise RuntimeError(f"Authenticated C3 baseline is missing or invalid: {case_id}")
    return float(value)


def run_evaluation_case(
    *,
    case_id: str,
    dataset_item: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    labels: Sequence[int],
    run_root: Path,
    source: Mapping[str, Any],
    decision: Mapping[str, Any],
    decision_sha256: str,
    barrier: Mapping[str, Any],
    barrier_sha256: str,
    device: torch.device,
    execution: Mapping[str, Any] | None = None,
) -> Path:
    marker = _evaluation_case_path(run_root.resolve(), case_id)
    if marker.is_file():
        payload = _load_json(marker)
        validate_evaluation_case_marker(payload, decision, decision_sha256, barrier, barrier_sha256)
        return marker
    decision_path = _decision_case_path(run_root.resolve(), case_id)
    expected_decision_sha = barrier["decision_case_sha256"][case_id]
    if sha256_file(decision_path) != expected_decision_sha:
        raise RuntimeError(f"C4 decision snapshot changed before evaluation: {case_id}")
    decision_payload = _load_json(decision_path)
    validate_decision_case_marker(decision_payload, decision, decision_sha256, verify_heavy_bytes=True)

    moving_image, fixed_image, moving_seg, fixed_seg = dataset_item
    if array_sha256(moving_image.numpy()) != decision["image_inputs"]["atlas"]["array_sha256"]:
        raise RuntimeError("C4 evaluation atlas image differs from the decision cache")
    if array_sha256(fixed_image.numpy()) != decision["image_inputs"][case_id]["array_sha256"]:
        raise RuntimeError(f"C4 evaluation case image differs from the decision cache: {case_id}")
    moving_seg = moving_seg.unsqueeze(0).to(device)
    fixed_seg = fixed_seg.unsqueeze(0).to(device)
    initial_record = decision["source_initial"][case_id]["field"]
    initial = load_flow_npz(_resolve_field(Path(decision["source_c3_heavy_root"]), initial_record)).to(device)
    labels_tuple = tuple(int(value) for value in labels)
    baseline_warped = sample_at_psi(moving_seg.float(), initial, mode="nearest").long()
    baseline_per_label = dice_per_label(baseline_warped, fixed_seg.long(), labels_tuple)
    baseline = float(baseline_per_label.mean())
    historical = _source_c3_baseline(source, case_id)
    if not math.isclose(baseline, historical, rel_tol=0.0, abs_tol=BASELINE_DICE_PARITY_ATOL):
        raise RuntimeError(f"C4 baseline Dice differs from frozen C3: {case_id}")

    rows: list[dict[str, Any]] = []
    for arm in decision_payload["arms"]:
        if arm.get("candidate_field") is None:
            rows.append(
                {
                    "arm_index": arm["arm_index"],
                    "arm_id": arm["arm_id"],
                    "evaluated": False,
                    "primary_action": "DIAGNOSTIC_ONLY",
                }
            )
            continue
        candidate = load_flow_npz(_resolve_field(Path(decision["heavy_root"]), arm["candidate_field"])).to(device)
        candidate_warped = sample_at_psi(moving_seg.float(), candidate, mode="nearest").long()
        candidate_per_label = dice_per_label(candidate_warped, fixed_seg.long(), labels_tuple)
        candidate_mean = float(candidate_per_label.mean())
        accept = arm["action"] == "ACCEPT"
        returned_per_label = candidate_per_label if accept else baseline_per_label
        returned_mean = candidate_mean if accept else baseline
        rows.append(
            {
                "arm_index": arm["arm_index"],
                "arm_id": arm["arm_id"],
                "evaluated": True,
                "baseline_dice": baseline,
                "capacity_candidate_dice": candidate_mean,
                "capacity_dice_delta": candidate_mean - baseline,
                "primary_returned_dice": returned_mean,
                "primary_dice_delta": returned_mean - baseline,
                "primary_action": arm["action"],
                "per_label": [
                    {
                        "label": label,
                        "baseline_dice": float(base),
                        "candidate_dice": float(candidate_value),
                        "returned_dice": float(returned),
                    }
                    for label, base, candidate_value, returned in zip(
                        labels_tuple,
                        baseline_per_label,
                        candidate_per_label,
                        returned_per_label,
                        strict=True,
                    )
                ],
            }
        )
    if sha256_file(decision_path) != expected_decision_sha:
        raise RuntimeError(f"C4 decision snapshot changed during evaluation: {case_id}")
    payload = {
        "schema": EVALUATION_CASE_SCHEMA,
        "protocol_id": PROTOCOL_ID,
        "status": "COMPLETE",
        "case_id": case_id,
        "decision_contract_sha256": decision_sha256,
        "decision_barrier_sha256": barrier_sha256,
        "decision_case_sha256": expected_decision_sha,
        "labels_loaded_after_barrier": True,
        "test_split_accessed": False,
        "labels": list(labels_tuple),
        "baseline_c3_parity_verified": True,
        "arms": rows,
        "execution": dict(execution or {}),
    }
    marker.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(marker, payload)
    validate_evaluation_case_marker(payload, decision, decision_sha256, barrier, barrier_sha256)
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
    source: Mapping[str, Any],
    decision: Mapping[str, Any],
    decision_sha256: str,
    barrier: Mapping[str, Any],
    barrier_sha256: str,
    device: torch.device,
    execution: Mapping[str, Any] | None = None,
) -> Path:
    expected = decision["shards"].get(str(shard_index))
    if list(case_ids) != expected or physical_gpu != decision["shard_to_physical_gpu"].get(str(shard_index)):
        raise RuntimeError("C4 evaluation worker does not match the frozen shard")
    for case_id in case_ids:
        run_evaluation_case(
            case_id=case_id,
            dataset_item=dataset_item_for_case(case_id),
            labels=labels,
            run_root=run_root,
            source=source,
            decision=decision,
            decision_sha256=decision_sha256,
            barrier=barrier,
            barrier_sha256=barrier_sha256,
            device=device,
            execution=execution,
        )
    marker = _worker_path(run_root.resolve(), "evaluation", attempt_id, shard_index)
    payload = {
        "schema": WORKER_SCHEMA,
        "protocol_id": PROTOCOL_ID,
        "status": "COMPLETE",
        "phase": "evaluation",
        "attempt_id": attempt_id,
        "shard_index": shard_index,
        "physical_gpu": physical_gpu,
        "case_ids": list(case_ids),
        "decision_contract_sha256": decision_sha256,
        "decision_barrier_sha256": barrier_sha256,
        "labels_loaded_after_barrier": True,
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
    )
    return marker


def classify_next_branch(
    evidence: Sequence[ArmEvidence],
    returned_policy: Mapping[str, PairedSummary],
) -> dict[str, Any]:
    if set(returned_policy) != set(SELECTABLE_ARM_IDS):
        raise ValueError("C4 returned-policy evidence is incomplete")
    for row in evidence:
        if row.policy_vs_baseline != returned_policy.get(row.arm_id):
            raise ValueError("C4 ArmEvidence and returned-policy map disagree")
    return asdict(select_next_branch(evidence))


def _csv_value(value: Any) -> Any:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return value


def _write_rows(path: Path, rows: list[dict[str, Any]], preferred: Sequence[str]) -> None:
    fields = [name for name in preferred if any(name in row for row in rows)]
    fields.extend(sorted({key for row in rows for key in row} - set(fields)))
    normalized = [{key: _csv_value(row.get(key, "")) for key in fields} for row in rows]
    atomic_write_text(path, rows_to_csv(fields, normalized))


def _metric_value(bundle: Mapping[str, Any], metric_id: str) -> float:
    row = bundle.get(metric_id) or {}
    value = row.get("value")
    if row.get("status") != "OK" or isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RuntimeError(f"C4 required geometry metric is undefined: {metric_id}")
    return float(value)


def _scientific_reference_geometry_mean(rows_by_arm: Mapping[str, Sequence[Mapping[str, Any]]]) -> float:
    rows = rows_by_arm.get(SCIENTIFIC_REFERENCE_ARM_ID, ())
    if len(rows) != EXPECTED_CASE_COUNT:
        raise RuntimeError("C4 scientific geometry reference does not cover val-58")
    values = np.asarray([row.get("candidate_geometry") for row in rows], dtype=np.float64)
    if not np.isfinite(values).all():
        raise RuntimeError("C4 scientific geometry reference is undefined")
    return float(values.mean())


def finalize_c4(
    *,
    run_root: Path,
    source: Mapping[str, Any],
    decision: Mapping[str, Any],
    decision_sha256: str,
    barrier: Mapping[str, Any],
    barrier_sha256: str,
) -> dict[str, str]:
    root = run_root.resolve()
    decision_payloads: dict[str, dict[str, Any]] = {}
    evaluation_payloads: dict[str, dict[str, Any]] = {}
    for case_id in decision["case_ids"]:
        decision_path = _decision_case_path(root, case_id)
        evaluation_path = _evaluation_case_path(root, case_id)
        if sha256_file(decision_path) != barrier["decision_case_sha256"][case_id]:
            raise RuntimeError(f"C4 decision snapshot changed before finalization: {case_id}")
        decision_payloads[case_id] = _load_json(decision_path)
        evaluation_payloads[case_id] = _load_json(evaluation_path)
        validate_decision_case_marker(decision_payloads[case_id], decision, decision_sha256, verify_heavy_bytes=True)
        validate_evaluation_case_marker(
            evaluation_payloads[case_id], decision, decision_sha256, barrier, barrier_sha256
        )

    per_arm: list[dict[str, Any]] = []
    per_label: list[dict[str, Any]] = []
    scale_rows: list[dict[str, Any]] = []
    resource_rows: list[dict[str, Any]] = []
    rows_by_arm: dict[str, list[dict[str, Any]]] = {arm.arm_id: [] for arm in ARM_SPECS}
    for case_id in decision["case_ids"]:
        dec = decision_payloads[case_id]
        evaluated = {row["arm_id"]: row for row in evaluation_payloads[case_id]["arms"]}
        observed_baselines = {float(row["baseline_dice"]) for row in evaluated.values() if row.get("evaluated") is True}
        historical_baseline = float(source["evaluation_baseline_dice"][case_id])
        if len(observed_baselines) != 1 or not math.isclose(
            observed_baselines.pop(),
            historical_baseline,
            rel_tol=0.0,
            abs_tol=BASELINE_DICE_PARITY_ATOL,
        ):
            raise RuntimeError(f"C4 evaluation baseline no longer matches authenticated C3: {case_id}")
        baseline_geometry = _metric_value(dec["baseline_geometry"], MATHEMATICAL_SDLOGJ_CROP2)
        for arm in dec["arms"]:
            evaluation = evaluated[arm["arm_id"]]
            if evaluation["primary_action"] != arm["action"]:
                raise RuntimeError(
                    f"C4 evaluation action changed after the decision barrier: {case_id}/{arm['arm_id']}"
                )
            row = {
                "case_id": case_id,
                "arm_index": arm["arm_index"],
                "arm_id": arm["arm_id"],
                "role": arm["role"],
                "selectable": arm["selectable"],
                "diagnostic_only": arm["diagnostic_only"],
                "action": arm["action"],
                "reason": arm["reason"],
                "exact_certified": (arm.get("exact") or {}).get("certified"),
                "baseline_geometry": baseline_geometry,
                "candidate_geometry": (
                    _metric_value(arm["geometry"], MATHEMATICAL_SDLOGJ_CROP2) if arm.get("geometry") else None
                ),
                "ncc7_improvement": (arm.get("utility") or {}).get("improvement"),
                "support_retention": (arm.get("support") or {}).get("retention"),
                "proposal": arm.get("proposal"),
                "diagnostics": arm.get("diagnostics"),
                **{key: value for key, value in evaluation.items() if key not in {"per_label", "arm_index", "arm_id"}},
            }
            per_arm.append(row)
            rows_by_arm[arm["arm_id"]].append(row)
            for label_row in evaluation.get("per_label", []):
                per_label.append({"case_id": case_id, "arm_id": arm["arm_id"], **label_row})
        scale_rows.extend({"case_id": case_id, **row} for row in dec["scale_agreement"])
        resource_rows.append({"case_id": case_id, **dec["resources"]})

    contrast_vectors: dict[str, np.ndarray] = {}
    for contrast in CONTRAST_SPECS:
        candidate_key = (
            "primary_returned_dice" if contrast.family == "returned_policy_vs_baseline" else "capacity_candidate_dice"
        )
        candidate = np.asarray([row[candidate_key] for row in rows_by_arm[contrast.candidate_arm_id]])
        if contrast.reference_id == "zero_update_baseline":
            reference = np.asarray([row["baseline_dice"] for row in rows_by_arm[contrast.candidate_arm_id]])
        else:
            reference = np.asarray([row["capacity_candidate_dice"] for row in rows_by_arm[contrast.reference_id]])
        contrast_vectors[contrast.contrast_id] = candidate - reference
    simultaneous = simultaneous_paired_summaries(contrast_vectors)

    arm_summaries: list[dict[str, Any]] = []
    evidence: list[ArmEvidence] = []
    returned_policy: dict[str, PairedSummary] = {}
    reference_geometry_mean = _scientific_reference_geometry_mean(rows_by_arm)
    for arm_id in SELECTABLE_ARM_IDS:
        rows = rows_by_arm[arm_id]
        capacity = simultaneous[f"capacity::{arm_id}::vs_zero_update_baseline"]
        incremental = None
        if arm_id != SCIENTIFIC_REFERENCE_ARM_ID:
            incremental = simultaneous[f"incremental::{arm_id}::vs_{SCIENTIFIC_REFERENCE_ARM_ID}"]
        policy = simultaneous[f"returned_policy::{arm_id}::vs_zero_update_baseline"]
        returned_policy[arm_id] = policy
        candidate_geometry = float(np.mean([row["candidate_geometry"] for row in rows]))
        baseline_geometry = float(np.mean([row["baseline_geometry"] for row in rows]))
        item = ArmEvidence(
            arm_id=arm_id,
            capacity_vs_baseline=capacity,
            incremental_vs_reference=incremental,
            policy_vs_baseline=policy,
            geometry=(
                GeometryComparison(
                    MATHEMATICAL_SDLOGJ_CROP2,
                    candidate_geometry,
                    reference_geometry_mean,
                    True,
                ),
            ),
            all_work_units_complete=len(rows) == EXPECTED_CASE_COUNT,
            all_exact_certified=all(row["exact_certified"] is True for row in rows),
        )
        evidence.append(item)
        eligibility = assess_arm(item)
        arm_summaries.append(
            {
                "arm_id": arm_id,
                "capacity": asdict(capacity),
                "incremental_vs_reference": asdict(incremental) if incremental else None,
                "returned_policy": asdict(policy),
                "candidate_dice_mean": float(np.mean([row["capacity_candidate_dice"] for row in rows])),
                "baseline_dice_mean": float(np.mean([row["baseline_dice"] for row in rows])),
                "returned_dice_mean": float(np.mean([row["primary_returned_dice"] for row in rows])),
                "candidate_geometry_mean": candidate_geometry,
                "baseline_geometry_mean": baseline_geometry,
                "scientific_reference_geometry_mean": reference_geometry_mean,
                "geometry_delta_vs_baseline_mean": candidate_geometry - baseline_geometry,
                "geometry_delta_vs_scientific_reference_mean": candidate_geometry - reference_geometry_mean,
                "accepted_cases": sum(row["action"] == "ACCEPT" for row in rows),
                **asdict(eligibility),
                "returned_policy_material": materially_strong_policy(policy),
            }
        )

    diagnostic_specificity: list[dict[str, Any]] = []
    for arm_id, reference_id in (("intensity_s1", "mind_d2_s1"), ("intensity_s2", "mind_d2_s2")):
        rows = rows_by_arm[arm_id]
        reference_rows = rows_by_arm[reference_id]
        capacity_delta = np.asarray(
            [row["capacity_candidate_dice"] - row["baseline_dice"] for row in rows], dtype=np.float64
        )
        vs_mind = np.asarray(
            [
                row["capacity_candidate_dice"] - reference["capacity_candidate_dice"]
                for row, reference in zip(rows, reference_rows, strict=True)
            ],
            dtype=np.float64,
        )
        diagnostic = {
            "arm_id": arm_id,
            "diagnostic_only": True,
            "selectable": False,
            "evaluated": True,
            "reference_arm_id": reference_id,
            "capacity_dice_delta_mean": float(capacity_delta.mean()),
            "capacity_dice_delta_median": float(np.median(capacity_delta)),
            "capacity_improved_cases": int((capacity_delta > 0).sum()),
            "capacity_worsened_cases": int((capacity_delta < 0).sum()),
            "capacity_delta_vs_mind_mean": float(vs_mind.mean()),
            "candidate_dice_mean": float(np.mean([row["capacity_candidate_dice"] for row in rows])),
            "returned_dice_mean": float(np.mean([row["primary_returned_dice"] for row in rows])),
            "accepted_cases": sum(row["action"] == "ACCEPT" for row in rows),
            "promotion_eligible": False,
            "inference_scope": "POST_BARRIER_DESCRIPTOR_SPECIFICITY_DIAGNOSTIC",
        }
        diagnostic_specificity.append(diagnostic)
        arm_summaries.append(diagnostic)
    for arm_id in ("legacy_mind_d2_s1_collar4", "mind_f222_s1"):
        arm_summaries.append(
            {
                "arm_id": arm_id,
                "diagnostic_only": True,
                "selectable": False,
                "evaluated": False,
                "promotion_eligible": False,
                "inference_scope": "LABEL_FREE_INTEGRITY_DIAGNOSTIC",
            }
        )

    next_branch = classify_next_branch(evidence, returned_policy)
    contrast_rows = [
        {
            "contrast_id": contrast_id,
            "simultaneous_family_size": len(simultaneous),
            "method": "paired_case_bootstrap_max_absolute_centered_mean_deviation",
            **asdict(summary),
        }
        for contrast_id, summary in simultaneous.items()
    ]
    hypotheses = {
        "schema": f"ctcf-search-c4-hypotheses-{SCHEMA_VERSION}",
        "protocol_id": PROTOCOL_ID,
        "post_barrier_only": True,
        "test_115_authorized": False,
        "families": {
            "descriptor_context": [row for row in contrast_rows if row["contrast_id"].startswith("descriptor::")],
            "physical_reach": [row for row in contrast_rows if row["contrast_id"].startswith("reach::")],
            "capacity": [row for row in contrast_rows if row["contrast_id"].startswith("capacity::")],
            "returned_policy": [row for row in contrast_rows if row["contrast_id"].startswith("returned_policy::")],
            "descriptor_specificity_diagnostic": diagnostic_specificity,
        },
    }
    summary = {
        "schema": f"ctcf-search-c4-summary-{SCHEMA_VERSION}",
        "protocol_id": PROTOCOL_ID,
        "status": "COMPLETE",
        "n_cases": len(decision_payloads),
        "test_115_authorized": False,
        "test_split_accessed": False,
        "labels_used_for_decision": False,
        "decision_barrier_sha256": barrier_sha256,
        "scientific_reference_arm_id": SCIENTIFIC_REFERENCE_ARM_ID,
        "next_branch": next_branch,
        "arm_summaries": arm_summaries,
    }

    paths = {
        "per_arm": root / "per_arm.csv",
        "per_label_dice": root / "per_label_dice.csv",
        "scale_agreement": root / "scale_agreement.csv",
        "arm_summary": root / "arm_summary.csv",
        "preregistered_contrasts": root / "preregistered_contrasts.csv",
        "resource_summary": root / "resource_summary.csv",
        "hypotheses": root / "hypotheses.json",
        "summary": root / "summary.json",
        "next_branch": root / "next_branch.json",
    }
    _write_rows(paths["per_arm"], per_arm, ("case_id", "arm_index", "arm_id", "action"))
    _write_rows(paths["per_label_dice"], per_label, ("case_id", "arm_id", "label"))
    _write_rows(paths["scale_agreement"], scale_rows, ("case_id", "comparison_id", "comparison_kind"))
    _write_rows(paths["arm_summary"], arm_summaries, ("arm_id", "candidate_dice_mean", "returned_dice_mean"))
    _write_rows(paths["preregistered_contrasts"], contrast_rows, ("contrast_id", "mean", "ci_low", "ci_high"))
    _write_rows(paths["resource_summary"], resource_rows, ("case_id", "elapsed_sec", "peak_gpu_bytes"))
    atomic_write_json(paths["hypotheses"], hypotheses)
    atomic_write_json(paths["summary"], summary)
    atomic_write_json(paths["next_branch"], next_branch)

    artifact_hashes = {name: sha256_file(path) for name, path in paths.items()}
    manifest = {
        "schema": f"ctcf-search-c4-run-manifest-{SCHEMA_VERSION}",
        "protocol_id": PROTOCOL_ID,
        "status": "COMPLETE",
        "decision_contract_sha256": decision_sha256,
        "decision_barrier_sha256": barrier_sha256,
        "source_c3_run_id": source["source_c3"]["run_id"],
        "test_115_authorized": False,
        "test_split_accessed": False,
        "decision_case_sha256": {case_id: barrier["decision_case_sha256"][case_id] for case_id in decision["case_ids"]},
        "evaluation_case_sha256": {
            case_id: sha256_file(_evaluation_case_path(root, case_id)) for case_id in decision["case_ids"]
        },
        "files": artifact_hashes,
        "next_branch": next_branch,
    }
    manifest_path = root / "c4_manifest.json"
    atomic_write_json(manifest_path, manifest)
    artifact_hashes["c4_manifest"] = sha256_file(manifest_path)
    return artifact_hashes
