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
from pathlib import Path
from typing import Any

import numpy as np
import torch

from models.CorrMLP.networks import Correlation
from tools.analysis.run_artifacts import atomic_write_json, atomic_write_text, sha256_file
from tools.analysis.run_search_gate_c6 import (
    _geometry_bundle,
    _metric_value,
    _require_exact_geometry,
    _utility_action,
    _write_csv,
)
from tools.analysis.search_gate_c3 import primary_ncc_decision
from tools.analysis.search_gate_c7 import (
    ALL_LABEL_CI_LOW_VS_C4_MIN_STRICT,
    ARM_SPECS,
    C7_POLICY_SHA256,
    CLIP_RETENTION_GE_090_MIN_COUNT,
    CLIP_RETENTION_MEDIAN_MIN,
    CLIP_RETENTION_PER_CASE_MIN,
    COMMON_EVIDENCE_COLLAR,
    DESCRIPTOR_CHECKPOINT_BYTES,
    DESCRIPTOR_CHECKPOINT_SHA256,
    EVALUATION_LABEL_IDS,
    EXACT_CLAIM_EPS,
    EXPECTED_CASE_COUNT,
    FACTORS,
    FINAL_AMPLITUDE,
    FINAL_LOCAL_CLIP_SWEEPS,
    IMAGE_NORMALIZATION_STD_FLOOR,
    MATCHED_CONTROL_ARM_ID,
    POST_SMOOTHING_PASSES,
    POSTERIOR_TEMPERATURE,
    PROPOSAL_MULTIPLIER,
    PROTOCOL_ID,
    REFERENCE_ARM_ID,
    RISK_LABEL_CI_LOW_VS_C4_MIN_STRICT,
    RISK_LABEL_IDS,
    SCHEMA_VERSION,
    SDLOGJ_CI_HIGH_VS_C4_MAX,
    SELECTABLE_ARM_IDS,
    SOURCE_CONTEXT_ARM_ID,
    STAGE_LOCAL_CLIP_SWEEPS,
    STAGE_WORK_EPS_DECREMENT,
    STANDARDIZATION_FLOOR,
    TEST_115_AUTHORIZED,
    WORK_EPS,
    ZERO_FIELD_LOCAL_COST_PARITY_ATOL,
    ZERO_FIELD_LOCAL_COST_PARITY_RTOL,
    assert_frozen_policy,
    assess_arm,
    policy_dict,
    select_branch,
    simultaneous_paired_summaries,
)
from tools.analysis.search_gate_c7_source import (
    BARRIER_SCHEMA,
    DECISION_CASE_SCHEMA,
    DECISION_SCHEMA,
    EVALUATION_BARRIER_SCHEMA,
    EVALUATION_CASE_SCHEMA,
    EVALUATION_CONTRACT_SCHEMA,
    SOURCE_SCHEMA,
    WORKER_SCHEMA,
    assert_decision_payload_label_free,
    authenticate_c6_source,
    field_record,
    immutable_json,
    json_equivalent,
    load_c6_metrics_after_barrier,
    load_field,
    load_image,
    load_json,
    roots,
    verify_record,
)
from tools.analysis.search_gate_common import git, utc_now
from tools.analysis.search_gate_cost_volume import masked_vector_rms
from tools.analysis.search_gate_learned import (
    CORRMLP_FULL_STATE_KEY_COUNT,
    CORRMLP_IXI_LAST_EPOCH,
    CORRMLP_X1_CONV_PADDING_MARGIN,
    DEFAULT_MOMENT_REDUCTION,
    EqualHybridCostVolume,
    FrozenCorrMLPDescriptor,
    RawCandidateCostVolume,
    build_raw_corrmlp_x1_cost_volume,
    corrmlp_x1_offsets,
    equal_standardized_intensity_hybrid,
    extract_corrmlp_x1,
    load_frozen_corrmlp_x1,
    raw_candidate_cost_volume,
    standardize_raw_candidate_costs,
    valid_corrmlp_x1_sample_mask,
)
from tools.analysis.search_gate_metrics import DIGITAL_DECOMPOSITION, MATHEMATICAL_SDLOGJ_CROP2
from tools.analysis.search_gate_multiscale import (
    CenteredCostVolume,
    DecodedProposal,
    WorkEstimate,
    decode_posterior_mean,
    posterior_from_standardized_costs,
    postprocess_and_match_rms,
)
from tools.analysis.search_gate_pyramid import PyramidDirection, PyramidStage, array_sha256, direction_record
from tools.analysis.search_gate_runtime import (
    parse_physical_gpus,
    round_robin_shards,
    save_reload_certify,
    shard_gpu_map,
)
from tools.analysis.transactional_search import (
    certified_local_clip_candidate,
    geometry_mask,
    masked_zscore,
    sample_at_psi,
)
from utils import dice_per_label, setup_device

DEFAULT_MIN_FREE_GIB = 45.0
RESUME_MIN_FREE_GIB = 5.0
PILOT_CASE_ID = "subject_344"
COMPUTED_ARM_IDS = (MATCHED_CONTROL_ARM_ID, *SELECTABLE_ARM_IDS)
DIRECTION_DIAGNOSTIC_FIELDS = (
    "pre_normalization_rms",
    "rematch_gain",
    "normalized_rms",
    "stage_clip_retention_min",
    "stage_clip_retention_mean",
    "final_clip_retained_norm_ratio",
)


def _configure_strict_fp32_backend() -> None:
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    if (
        torch.backends.cuda.matmul.allow_tf32
        or torch.backends.cudnn.allow_tf32
        or torch.get_float32_matmul_precision() != "highest"
    ):
        raise RuntimeError("C7 could not freeze the no-TF32 FP32 backend contract")


def _runtime_signature() -> dict[str, Any]:
    import scipy

    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
    }


def _assert_clean_runtime(decision: Mapping[str, Any], stage: str) -> None:
    if git("rev-parse", "HEAD") != decision["git_head"] or git("status", "--porcelain=v1"):
        raise RuntimeError(f"C7 {stage} code differs from its clean prepared contract")
    observed = _runtime_signature()
    if observed != dict(decision["runtime_signature"]):
        raise RuntimeError(f"C7 {stage} runtime changed: {observed} != {dict(decision['runtime_signature'])}")


def _tree_bytes(root: Path) -> int:
    return sum(path.stat().st_size for path in root.rglob("*") if path.is_file()) if root.exists() else 0


def _validate_disk_budget(target: Path, minimum_free_gib: float) -> None:
    if not math.isfinite(minimum_free_gib) or minimum_free_gib < 0:
        raise ValueError("C7 disk budget must be finite and non-negative")
    target.parent.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(target.parent).free / 2**30
    retained = _tree_bytes(target) / 2**30
    if target.exists():
        if free < RESUME_MIN_FREE_GIB or free + retained < minimum_free_gib:
            raise RuntimeError(
                f"C7 resume lacks disk: free={free:.2f} GiB, retained={retained:.2f} GiB, "
                f"required={minimum_free_gib:.2f} GiB"
            )
    elif free < minimum_free_gib:
        raise RuntimeError(f"C7 requires {minimum_free_gib:.2f} GiB free; found {free:.2f} GiB")


def _dataset_tsv(raw_inputs: Mapping[str, Mapping[str, Any]]) -> str:
    fields = ("dataset", "split", "case_id", "path", "bytes", "sha256", "mtime_utc")
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields, delimiter="\t", lineterminator="\n")
    writer.writeheader()
    for record in raw_inputs.values():
        writer.writerow({field: record.get(field, "") for field in fields})
    return stream.getvalue()


def _immutable_text(path: Path, value: str) -> None:
    if path.exists():
        if path.read_text(encoding="utf-8") != value:
            raise FileExistsError(f"refusing to replace immutable C7 artifact: {path}")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_text(path, value)


def _decision_policy() -> dict[str, Any]:
    full = policy_dict()
    descriptor = dict(full["descriptor"])
    descriptor["checkpoint_selection"] = "fixed_epoch_99_last_endpoint_not_metric_selected_best"
    return {
        "protocol_id": full["protocol_id"],
        "schema_version": full["schema_version"],
        "dataset": full["dataset"],
        "test_115_authorized": False,
        "source": full["source"],
        "descriptor": descriptor,
        "arms": full["arms"],
        "construction": full["construction"],
    }


def _checkpoint_record(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"C7 descriptor checkpoint is absent: {resolved}")
    if resolved.stat().st_size != DESCRIPTOR_CHECKPOINT_BYTES:
        raise RuntimeError(
            f"C7 descriptor checkpoint size mismatch: {resolved.stat().st_size} != {DESCRIPTOR_CHECKPOINT_BYTES}"
        )
    observed = sha256_file(resolved)
    if observed != DESCRIPTOR_CHECKPOINT_SHA256:
        raise RuntimeError(f"C7 descriptor checkpoint SHA-256 mismatch: {observed}")
    return {
        "path": str(resolved),
        "bytes": resolved.stat().st_size,
        "sha256": observed,
        "epoch": CORRMLP_IXI_LAST_EPOCH,
        "state_key_count": CORRMLP_FULL_STATE_KEY_COUNT,
        "selection": "fixed_epoch_99_last_endpoint",
        "best_checkpoint_forbidden": True,
    }


def prepare_contracts(
    *,
    run_root: Path,
    heavy_root: Path,
    source_c6_dir: Path,
    source_c3_heavy_root: Path,
    source_c4_heavy_root: Path,
    source_c6_heavy_root: Path,
    descriptor_checkpoint: Path,
    physical_gpus: Sequence[str],
) -> tuple[str, str]:
    snapshot = authenticate_c6_source(
        source_c6_dir,
        source_c3_heavy_root,
        source_c4_heavy_root,
        source_c6_heavy_root,
        verify_heavy_bytes=True,
    )
    case_ids = list(snapshot["case_ids"])
    if PILOT_CASE_ID not in case_ids:
        raise RuntimeError(f"C7 frozen pilot is absent: {PILOT_CASE_ID}")
    checkpoint = _checkpoint_record(descriptor_checkpoint)
    source = {
        "schema": SOURCE_SCHEMA,
        "protocol_id": PROTOCOL_ID,
        "decision_policy": _decision_policy(),
        "policy_sha256": C7_POLICY_SHA256,
        "authenticated_c6": snapshot,
        "descriptor_checkpoint": checkpoint,
        "test_115_authorized": False,
        "test_split_accessed": False,
    }
    shards = round_robin_shards(case_ids, len(physical_gpus))
    decision = {
        "schema": DECISION_SCHEMA,
        "protocol_id": PROTOCOL_ID,
        "git_head": git("rev-parse", "HEAD"),
        "runtime_signature": _runtime_signature(),
        "decision_policy": _decision_policy(),
        "policy_sha256": C7_POLICY_SHA256,
        "roots": {
            **snapshot["roots"],
            "target_c7_heavy": str(heavy_root.resolve()),
        },
        "source_c6_manifest_sha256": snapshot["source_c6_manifest_sha256"],
        "source_c6_decision_case_sha256": snapshot["source_c6_decision_case_sha256"],
        "case_ids": case_ids,
        "seed": snapshot["seed"],
        "image_inputs": snapshot["image_inputs"],
        "source_initial": snapshot["source_initial"],
        "source_historical": snapshot["source_historical"],
        "source_c4_anchors": snapshot["source_c4_anchors"],
        "source_c6_context": snapshot["source_c6_context"],
        "baseline_geometry": snapshot["baseline_geometry"],
        "descriptor_checkpoint": checkpoint,
        "pilot_case_id": PILOT_CASE_ID,
        "num_shards": len(physical_gpus),
        "physical_gpus": list(physical_gpus),
        "shard_to_physical_gpu": shard_gpu_map(list(physical_gpus)),
        "shards": shards,
        "labels_loaded_to_device": False,
        "test_115_authorized": False,
        "test_split_accessed": False,
    }
    roots(decision)
    assert_decision_payload_label_free(source)
    assert_decision_payload_label_free(decision)
    source_sha = immutable_json(run_root / "source_contract.json", source)
    decision["source_contract_sha256"] = source_sha
    decision_sha = immutable_json(run_root / "decision_contract.json", decision)
    _immutable_text(
        run_root / "heavy_retention.txt",
        "".join(f"{key}={value}\n" for key, value in decision["roots"].items())
        + "retention=RETAIN_SOURCE_AND_TARGET_HEAVY_ROOTS_UNTIL_EXPLICIT_OPERATOR_DECISION\n"
        + "packaged=false\n",
    )
    return source_sha, decision_sha


def _load_decision(run_root: Path, digest: str) -> dict[str, Any]:
    path = run_root / "decision_contract.json"
    if not path.is_file() or sha256_file(path) != digest:
        raise RuntimeError("C7 decision contract SHA-256 changed")
    payload = load_json(path)
    case_ids = payload.get("case_ids") or []
    physical_gpus = payload.get("physical_gpus") or []
    if (
        payload.get("schema") != DECISION_SCHEMA
        or payload.get("protocol_id") != PROTOCOL_ID
        or payload.get("policy_sha256") != C7_POLICY_SHA256
        or not json_equivalent(payload.get("decision_policy"), _decision_policy())
        or payload.get("test_115_authorized") is not False
        or payload.get("test_split_accessed") is not False
        or len(case_ids) != EXPECTED_CASE_COUNT
        or len(set(case_ids)) != EXPECTED_CASE_COUNT
        or PILOT_CASE_ID not in case_ids
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
        or set(payload.get("source_c6_context") or {}) != set(case_ids)
    ):
        raise RuntimeError("invalid or altered C7 decision contract")
    roots(payload)
    if _checkpoint_record(Path(payload["descriptor_checkpoint"]["path"])) != payload["descriptor_checkpoint"]:
        raise RuntimeError("C7 descriptor checkpoint record changed")
    assert_decision_payload_label_free(payload)
    return payload


def _decision_case_path(run_root: Path, case_id: str) -> Path:
    return run_root / "cases" / case_id / "decision_complete.json"


def _evaluation_case_path(run_root: Path, case_id: str) -> Path:
    return run_root / "cases" / case_id / "evaluation_complete.json"


def _worker_path(run_root: Path, phase: str, attempt_id: str, shard_index: int) -> Path:
    return run_root / "workers" / phase / "attempts" / attempt_id / f"worker_{shard_index:02d}.json"


def _mean_on(value: torch.Tensor, mask: torch.Tensor) -> float:
    selected = value.masked_select(mask)
    if selected.numel() == 0 or not bool(torch.isfinite(selected).all()):
        raise RuntimeError("C7 diagnostic support is empty or non-finite")
    return float(selected.double().mean())


def _padding_support_mask(
    current: torch.Tensor,
    full_mask: torch.Tensor,
    stride: int,
) -> torch.Tensor:
    common = full_mask.clone()
    zero = torch.zeros_like(current)
    common &= valid_corrmlp_x1_sample_mask(
        zero,
        (0, 0, 0),
        stride_voxels=stride,
        padding_margin=CORRMLP_X1_CONV_PADDING_MARGIN,
    )
    for offset in corrmlp_x1_offsets(stride):
        common &= valid_corrmlp_x1_sample_mask(
            current,
            offset,
            stride_voxels=stride,
            padding_margin=CORRMLP_X1_CONV_PADDING_MARGIN,
        )
    if not bool(common.any()):
        raise RuntimeError(f"C7 stride {stride} common convolution-padding support is empty")
    return common


def _common_padding_support(
    learned_fixed: torch.Tensor,
    learned_moving: torch.Tensor,
    current: torch.Tensor,
    full_mask: torch.Tensor,
    stride: int,
) -> RawCandidateCostVolume:
    common = _padding_support_mask(current, full_mask, stride)
    return build_raw_corrmlp_x1_cost_volume(
        learned_fixed,
        learned_moving,
        current,
        common,
        stride_voxels=stride,
        padding_margin=CORRMLP_X1_CONV_PADDING_MARGIN,
        require_all_candidates_valid=True,
    )


def _intensity_raw_on_support(
    fixed: torch.Tensor,
    moving: torch.Tensor,
    current: torch.Tensor,
    support: torch.Tensor,
    stride: int,
) -> RawCandidateCostVolume:
    if not bool(support.any()):
        raise RuntimeError(f"C7 stride {stride} learned support is empty")
    fixed_norm = masked_zscore(fixed, support, std_floor=IMAGE_NORMALIZATION_STD_FLOOR)
    moving_norm = masked_zscore(moving, support, std_floor=IMAGE_NORMALIZATION_STD_FLOOR)
    offsets = corrmlp_x1_offsets(stride)
    valid = support.expand(-1, len(offsets), -1, -1, -1)
    costs = torch.empty(valid.shape, dtype=torch.float32, device=fixed.device)
    with torch.inference_mode():
        for index, offset in enumerate(offsets):
            sampled = sample_at_psi(moving_norm, current, offset)
            costs[:, index].copy_(((fixed_norm - sampled) ** 2).mean(dim=1))
    return raw_candidate_cost_volume(
        f"C7_MATCHED_INTENSITY_SSD_STRIDE{stride}",
        costs,
        valid,
        offsets=offsets,
    )


def _centered_for_posterior(
    raw: RawCandidateCostVolume,
    *,
    hybrid: EqualHybridCostVolume | None = None,
    source_ids: tuple[str, ...] | None = None,
) -> CenteredCostVolume:
    standardized = (
        hybrid.fusion
        if hybrid is not None
        else standardize_raw_candidate_costs(
            raw,
            mode=DEFAULT_MOMENT_REDUCTION,
            standardization_floor=STANDARDIZATION_FLOOR,
        )
    )
    valid = raw.valid & ~standardized.floor_hit
    if not bool(valid.any()):
        raise RuntimeError(f"C7 cost volume is flat on all support: {raw.cost_id}")
    costs = torch.where(valid, standardized.standardized_costs, torch.zeros_like(standardized.standardized_costs))
    return CenteredCostVolume(
        cost_id=hybrid.cost_id if hybrid is not None else raw.cost_id,
        standardized_costs=costs,
        valid=valid,
        offsets=raw.offsets,
        cost_mean=standardized.cost_mean,
        cost_std=standardized.cost_std,
        floor_hit=standardized.floor_hit,
        source_ids=(raw.cost_id,) if source_ids is None else source_ids,
        work=WorkEstimate(0, 27, 0),
    )


def _build_direction(
    *,
    arm_id: str,
    fixed: torch.Tensor,
    moving: torch.Tensor,
    learned_fixed: torch.Tensor,
    learned_moving: torch.Tensor,
    initial: torch.Tensor,
    rms_reference: torch.Tensor,
) -> PyramidDirection:
    if arm_id not in COMPUTED_ARM_IDS:
        raise ValueError(f"C7 cannot construct source arm: {arm_id}")
    spatial = tuple(int(value) for value in initial.shape[-3:])
    full_mask = geometry_mask(spatial, COMMON_EVIDENCE_COLLAR, initial.device)
    reference_rms = masked_vector_rms(rms_reference, full_mask)
    if reference_rms <= 0.0:
        raise RuntimeError("C7 historical RMS reference is zero")
    stage_target = reference_rms / len(FACTORS)
    current = initial.clone()
    stages: list[PyramidStage] = []
    for stage_index, factor in enumerate(FACTORS):
        common_support = _padding_support_mask(current, full_mask, factor)
        hybrid = None
        hybrid_raw = None
        if arm_id == MATCHED_CONTROL_ARM_ID:
            learned_raw = None
            intensity_raw = _intensity_raw_on_support(fixed, moving, current, common_support, factor)
            centered = _centered_for_posterior(intensity_raw)
        elif arm_id == SELECTABLE_ARM_IDS[0]:
            learned_raw = build_raw_corrmlp_x1_cost_volume(
                learned_fixed,
                learned_moving,
                current,
                common_support,
                stride_voxels=factor,
                padding_margin=CORRMLP_X1_CONV_PADDING_MARGIN,
                require_all_candidates_valid=True,
            )
            intensity_raw = None
            centered = _centered_for_posterior(learned_raw)
        else:
            learned_raw = build_raw_corrmlp_x1_cost_volume(
                learned_fixed,
                learned_moving,
                current,
                common_support,
                stride_voxels=factor,
                padding_margin=CORRMLP_X1_CONV_PADDING_MARGIN,
                require_all_candidates_valid=True,
            )
            intensity_raw = _intensity_raw_on_support(fixed, moving, current, common_support, factor)
            hybrid = equal_standardized_intensity_hybrid(
                learned_raw,
                intensity_raw,
                mode=DEFAULT_MOMENT_REDUCTION,
                standardization_floor=STANDARDIZATION_FLOOR,
            )
            hybrid_raw = RawCandidateCostVolume(
                cost_id=hybrid.cost_id,
                costs=hybrid.standardized_costs,
                valid=hybrid.valid,
                valid_count=hybrid.valid_count,
                offsets=hybrid.offsets,
            )
            centered = _centered_for_posterior(
                hybrid_raw,
                hybrid=hybrid,
                source_ids=(learned_raw.cost_id, intensity_raw.cost_id),
            )
        posterior = posterior_from_standardized_costs(centered, temperature=POSTERIOR_TEMPERATURE)
        decoded = decode_posterior_mean(posterior)
        processed = postprocess_and_match_rms(
            DecodedProposal(centered.cost_id, decoded.displacement, centered.offsets),
            full_mask,
            proposal_multiplier=PROPOSAL_MULTIPLIER,
            smoothing_passes=POST_SMOOTHING_PASSES,
            collar_width=COMMON_EVIDENCE_COLLAR,
            rms_reference=rms_reference / float(len(FACTORS)),
        )
        clip_eps = round(WORK_EPS - stage_index * STAGE_WORK_EPS_DECREMENT, 9)
        continuation_eps = (
            round(WORK_EPS - (stage_index + 1) * STAGE_WORK_EPS_DECREMENT, 9)
            if stage_index + 1 < len(FACTORS)
            else EXACT_CLAIM_EPS
        )
        updated, operator = certified_local_clip_candidate(
            current,
            processed.displacement,
            full_mask,
            work_eps=clip_eps,
            sweeps=STAGE_LOCAL_CLIP_SWEEPS,
        )
        bound = float(operator["output_fast_cert_bound"])
        if not math.isfinite(bound) or bound < continuation_eps:
            raise RuntimeError(f"C7 {arm_id} stage={stage_index} cannot continue safely: {bound} < {continuation_eps}")
        realized = updated - current
        current = updated
        active = centered.valid.any(dim=1, keepdim=True) & full_mask
        stages.append(
            PyramidStage(
                factor=factor,
                stride_voxels=factor,
                level_shape=spatial,
                level_collar=COMMON_EVIDENCE_COLLAR,
                generation_count=int(common_support.sum().item()),
                informative_count=int(active.sum().item()),
                entropy_mean=_mean_on(posterior.entropy, active),
                confidence_mean=_mean_on(posterior.confidence, active),
                decoded_rms_full_grid=masked_vector_rms(decoded.displacement, full_mask),
                requested_stage_rms=stage_target,
                realized_stage_rms=masked_vector_rms(realized, full_mask),
                clip_retention=float(operator["retained_norm_ratio"]),
                clip_work_eps=clip_eps,
                continuation_eps=continuation_eps,
                output_fast_cert_bound=bound,
            )
        )
        del learned_raw, intensity_raw, hybrid, hybrid_raw, centered, posterior, decoded, processed
    net = current - initial
    pre_rms = masked_vector_rms(net, full_mask)
    if pre_rms <= 0.0:
        raise RuntimeError(f"C7 {arm_id} produced a zero direction")
    normalized = net * float(reference_rms / pre_rms)
    return PyramidDirection(
        family="full_resolution",
        factors=FACTORS,
        rewarp_between_levels=True,
        displacement=normalized,
        reference_rms=reference_rms,
        pre_normalization_rms=pre_rms,
        normalized_rms=masked_vector_rms(normalized, full_mask),
        stages=tuple(stages),
    )


def _materialize(
    *,
    case_id: str,
    spec: Any,
    direction: PyramidDirection,
    initial: torch.Tensor,
    mask: torch.Tensor,
    fixed_norm: torch.Tensor,
    moving_norm: torch.Tensor,
    decision: Mapping[str, Any],
) -> dict[str, Any]:
    requested = direction.displacement * FINAL_AMPLITUDE
    candidate_raw, operator = certified_local_clip_candidate(
        initial,
        requested,
        mask,
        work_eps=WORK_EPS,
        sweeps=FINAL_LOCAL_CLIP_SWEEPS,
    )
    heavy_root = roots(decision)["target_c7_heavy"]
    path = heavy_root / "cases" / case_id / "arms" / f"{spec.arm_id}.npz"
    stored, exact = save_reload_certify(candidate_raw, path, EXACT_CLAIM_EPS)
    if exact.get("status") != "CERTIFIED" or exact.get("certified") is not True:
        raise RuntimeError(f"C7 save/reload exact certification failed: {case_id}/{spec.arm_id}")
    candidate = stored.to(initial.device)
    geometry = _geometry_bundle(candidate, mask)
    _require_exact_geometry(geometry, f"{case_id}/{spec.arm_id}")
    support, utility, action, reason = _utility_action(fixed_norm, moving_norm, initial, candidate, mask)
    return {
        "arm_index": spec.arm_index,
        "arm_id": spec.arm_id,
        "role": spec.role,
        "descriptor": spec.descriptor,
        "selectable": spec.selectable,
        "action": action,
        "reason": reason,
        "factors": list(FACTORS),
        "amplitude": FINAL_AMPLITUDE,
        "rewarp_between_levels": True,
        "direction": direction_record(direction),
        "requested_array_sha256": array_sha256(requested),
        "operator": operator,
        "candidate_field": field_record(path, heavy_root, str(exact["sha256"])),
        "exact": exact,
        "geometry": geometry,
        "support": support,
        "utility": utility,
    }


def _source_arm_row(
    *,
    spec: Any,
    record: Mapping[str, Any],
    geometry: Mapping[str, Any],
    action: str,
    source_case_sha256: str,
    support: Mapping[str, Any] | None = None,
    utility: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "arm_index": spec.arm_index,
        "arm_id": spec.arm_id,
        "role": spec.role,
        "descriptor": spec.descriptor,
        "selectable": False,
        "action": action,
        "factors": list(spec.factors),
        "amplitude": spec.amplitude,
        "rewarp_between_levels": spec.rewarp_between_stages,
        "source_arm_id": spec.source_arm_id,
        "source_c6_decision_case_sha256": source_case_sha256,
        "candidate_field": dict(record),
        "geometry": dict(geometry),
        "support": None if support is None else dict(support),
        "utility": None if utility is None else dict(utility),
    }


def _finite_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise RuntimeError(f"C7 value is not finite: {label}")
    return float(value)


def _validate_case_execution(
    payload: Mapping[str, Any],
    decision: Mapping[str, Any],
    case_id: str,
    *,
    phase: str,
) -> dict[str, Any]:
    expected_shards = [index for index, cases in decision["shards"].items() if case_id in cases]
    if len(expected_shards) != 1:
        raise RuntimeError(f"C7 case has an invalid frozen shard assignment: {case_id}")
    shard = int(expected_shards[0])
    physical_gpu = str(decision["shard_to_physical_gpu"][str(shard)])
    execution = payload.get("execution")
    runtime = decision["runtime_signature"]
    if (
        not isinstance(execution, Mapping)
        or payload.get("shard_index") != shard
        or str(payload.get("physical_gpu")) != physical_gpu
        or execution.get("phase") != phase
        or execution.get("shard_index") != shard
        or str(execution.get("physical_gpu")) != physical_gpu
        or execution.get("seed") != decision["seed"]
        or execution.get("deterministic") is not True
        or execution.get("labels_loaded_to_device") is not (phase == "evaluation")
        or execution.get("python") != runtime["python"]
        or execution.get("torch") != runtime["torch"]
        or execution.get("cuda_matmul_allow_tf32") != runtime["cuda_matmul_allow_tf32"]
        or execution.get("cudnn_allow_tf32") != runtime["cudnn_allow_tf32"]
        or execution.get("float32_matmul_precision") != runtime["float32_matmul_precision"]
        or not isinstance(execution.get("attempt_id"), str)
        or not execution["attempt_id"]
        or not isinstance(execution.get("host"), str)
        or not execution["host"]
        or not isinstance(execution.get("gpu_name"), str)
        or not execution["gpu_name"]
        or not isinstance(execution.get("device"), str)
        or not str(execution["device"]).startswith("cuda")
    ):
        raise RuntimeError(f"C7 {phase} execution provenance changed: {case_id}")
    return dict(execution)


def _validate_utility_decision(row: Mapping[str, Any], label: str) -> None:
    support = row.get("support")
    utility = row.get("utility")
    if not isinstance(support, Mapping) or not isinstance(utility, Mapping):
        raise RuntimeError(f"C7 utility evidence is absent: {label}")
    baseline_count = support.get("baseline_count")
    pair_count = support.get("pair_count")
    if (
        support.get("utility_id") != "COMMON_NCC7"
        or isinstance(baseline_count, bool)
        or not isinstance(baseline_count, int)
        or baseline_count <= 0
        or isinstance(pair_count, bool)
        or not isinstance(pair_count, int)
        or not 0 <= pair_count <= baseline_count
    ):
        raise RuntimeError(f"C7 utility support changed: {label}")
    retention = _finite_number(support.get("retention"), f"{label}/support_retention")
    baseline_loss = _finite_number(utility.get("baseline_loss"), f"{label}/baseline_loss")
    candidate_loss = _finite_number(utility.get("candidate_loss"), f"{label}/candidate_loss")
    improvement = _finite_number(utility.get("improvement"), f"{label}/improvement")
    if not math.isclose(retention, pair_count / baseline_count, rel_tol=0.0, abs_tol=1e-15) or not math.isclose(
        improvement,
        baseline_loss - candidate_loss,
        rel_tol=0.0,
        abs_tol=1e-15,
    ):
        raise RuntimeError(f"C7 utility arithmetic changed: {label}")
    inferred = primary_ncc_decision(
        exact_certified=row.get("exact", {}).get("certified") is True,
        support_retention=retention,
        baseline_ncc_loss=baseline_loss,
        candidate_ncc_loss=candidate_loss,
    ).to_dict()
    expected_action = "ACCEPT" if inferred["accept"] else "ROLLBACK"
    if row.get("action") != expected_action or row.get("reason") != inferred["reason"]:
        raise RuntimeError(f"C7 action is inconsistent with frozen utility evidence: {label}")


def _validate_decision_case_marker(
    path: Path,
    decision: Mapping[str, Any],
    decision_sha: str,
    case_id: str,
) -> dict[str, Any]:
    payload = load_json(path)
    rows = payload.get("arms") or []
    if (
        payload.get("schema") != DECISION_CASE_SCHEMA
        or payload.get("protocol_id") != PROTOCOL_ID
        or payload.get("status") != "COMPLETE"
        or payload.get("strict") is not True
        or payload.get("case_id") != case_id
        or payload.get("decision_contract_sha256") != decision_sha
        or payload.get("descriptor_checkpoint_sha256") != DESCRIPTOR_CHECKPOINT_SHA256
        or payload.get("descriptor_epoch") != CORRMLP_IXI_LAST_EPOCH
        or payload.get("descriptor_state_key_count") != CORRMLP_FULL_STATE_KEY_COUNT
        or payload.get("labels_loaded_to_device") is not False
        or payload.get("test_115_authorized") is not False
        or payload.get("test_split_accessed") is not False
        or tuple(row.get("arm_id") for row in rows) != tuple(spec.arm_id for spec in ARM_SPECS)
    ):
        raise RuntimeError(f"invalid or altered C7 decision marker: {case_id}")
    execution = _validate_case_execution(payload, decision, case_id, phase="decision")
    resource = payload.get("resource")
    if not isinstance(resource, Mapping):
        raise RuntimeError(f"C7 decision resource record is absent: {case_id}")
    _finite_number(resource.get("wall_sec"), f"{case_id}/wall_sec")
    peak = resource.get("peak_cuda_bytes")
    if isinstance(peak, bool) or not isinstance(peak, int) or peak < 0:
        raise RuntimeError(f"C7 decision peak CUDA record changed: {case_id}")
    source_sha = decision["source_c6_decision_case_sha256"][case_id]
    for row, spec in zip(rows, ARM_SPECS, strict=True):
        label = f"{case_id}/{spec.arm_id}"
        if (
            row.get("arm_index") != spec.arm_index
            or row.get("role") != spec.role
            or row.get("descriptor") != spec.descriptor
            or row.get("selectable") is not spec.selectable
            or row.get("factors") != list(spec.factors)
            or row.get("amplitude") != spec.amplitude
            or row.get("rewarp_between_levels") is not spec.rewarp_between_stages
            or row.get("source_arm_id") != spec.source_arm_id
        ):
            raise RuntimeError(f"C7 arm identity changed: {case_id}/{spec.arm_id}")
        verify_record(decision, row["candidate_field"], verify_array=True)
        _require_exact_geometry(row["geometry"], f"{case_id}/{spec.arm_id}")
        if spec.arm_id == REFERENCE_ARM_ID:
            expected = decision["source_c4_anchors"][case_id]
            if (
                row.get("action") != "REFERENCE"
                or row["candidate_field"].get("root_id") != "source_c4_heavy"
                or row.get("source_c6_decision_case_sha256") != source_sha
                or row.get("support") is not None
                or row.get("utility") is not None
                or not json_equivalent(row["candidate_field"], expected["field"])
                or not json_equivalent(row["geometry"], expected["geometry"])
            ):
                raise RuntimeError(f"C7 frozen reference record changed: {label}")
        elif spec.arm_id == SOURCE_CONTEXT_ARM_ID:
            expected = decision["source_c6_context"][case_id]
            if (
                row.get("action") != expected["action"]
                or row["candidate_field"].get("root_id") != "source_c6_heavy"
                or row.get("source_c6_decision_case_sha256") != source_sha
                or not json_equivalent(row["candidate_field"], expected["field"])
                or not json_equivalent(row["geometry"], expected["geometry"])
                or not json_equivalent(row.get("support"), expected["support"])
                or not json_equivalent(row.get("utility"), expected["utility"])
            ):
                raise RuntimeError(f"C7 frozen C6 context record changed: {label}")
        else:
            exact = row.get("exact")
            direction = row.get("direction")
            if (
                not isinstance(exact, Mapping)
                or exact.get("certified") is not True
                or exact.get("status") != "CERTIFIED"
                or exact.get("sha256") != row["candidate_field"].get("array_sha256")
                or row["candidate_field"].get("root_id") != "target_c7_heavy"
                or row.get("action") not in {"ACCEPT", "ROLLBACK"}
                or not isinstance(direction, Mapping)
                or direction.get("family") != "full_resolution"
                or direction.get("factors") != list(spec.factors)
                or direction.get("rewarp_between_levels") is not spec.rewarp_between_stages
            ):
                raise RuntimeError(f"C7 computed arm is incomplete: {label}")
            _validate_utility_decision(row, label)
    assert_decision_payload_label_free(payload)
    payload["execution"] = execution
    return payload


def _load_descriptor(decision: Mapping[str, Any], device: torch.device) -> FrozenCorrMLPDescriptor:
    record = decision["descriptor_checkpoint"]
    if _checkpoint_record(Path(record["path"])) != record:
        raise RuntimeError("C7 descriptor checkpoint record changed before strict load")
    descriptor = load_frozen_corrmlp_x1(record["path"], expected_sha256=DESCRIPTOR_CHECKPOINT_SHA256)
    descriptor.model.to(device)
    descriptor.model.eval()
    if any(parameter.requires_grad for parameter in descriptor.model.parameters()):
        raise RuntimeError("C7 descriptor parameters became trainable")
    return descriptor


def run_decision_case(
    *,
    case_id: str,
    shard_index: int,
    physical_gpu: str,
    run_root: Path,
    decision: Mapping[str, Any],
    decision_sha: str,
    device: torch.device,
    descriptor: FrozenCorrMLPDescriptor,
    execution: Mapping[str, Any],
) -> Path:
    marker = _decision_case_path(run_root, case_id)
    if marker.is_file():
        _validate_decision_case_marker(marker, decision, decision_sha, case_id)
        return marker
    if case_id not in decision["shards"].get(str(shard_index), []):
        raise RuntimeError(f"C7 case belongs to another shard: {case_id}")
    if str(physical_gpu) != str(decision["shard_to_physical_gpu"].get(str(shard_index))):
        raise RuntimeError(f"C7 case belongs to another physical GPU: {case_id}")
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats(device)
    atlas = torch.from_numpy(load_image(decision, decision["image_inputs"]["atlas"])).unsqueeze(0).to(device)
    fixed = torch.from_numpy(load_image(decision, decision["image_inputs"][case_id])).unsqueeze(0).to(device)
    initial = load_field(decision, decision["source_initial"][case_id]["field"]).to(device)
    historical = load_field(decision, decision["source_historical"][case_id]["raw_conf_requested_field"]).to(device)
    rms_reference = (historical - initial).float()
    mask = geometry_mask(tuple(initial.shape[-3:]), COMMON_EVIDENCE_COLLAR, device)
    fixed_norm = masked_zscore(fixed, mask, std_floor=IMAGE_NORMALIZATION_STD_FLOOR)
    moving_norm = masked_zscore(atlas, mask, std_floor=IMAGE_NORMALIZATION_STD_FLOOR)
    with torch.inference_mode():
        learned_fixed = extract_corrmlp_x1(descriptor.model, fixed.float())
        learned_moving = extract_corrmlp_x1(descriptor.model, atlas.float())

    c4 = decision["source_c4_anchors"][case_id]
    c6 = decision["source_c6_context"][case_id]
    c4_field = load_field(decision, c4["field"]).to(device)
    c6_field = load_field(decision, c6["field"]).to(device)
    c4_geometry = _geometry_bundle(c4_field, mask)
    c6_geometry = _geometry_bundle(c6_field, mask)
    _require_exact_geometry(c4_geometry, f"{case_id}/{REFERENCE_ARM_ID}")
    _require_exact_geometry(c6_geometry, f"{case_id}/{SOURCE_CONTEXT_ARM_ID}")
    if not json_equivalent(c4_geometry, c4["geometry"]) or not json_equivalent(c6_geometry, c6["geometry"]):
        raise RuntimeError(f"C7 source geometry parity failed: {case_id}")
    source_sha = decision["source_c6_decision_case_sha256"][case_id]
    rows = [
        _source_arm_row(
            spec=ARM_SPECS[0],
            record=c4["field"],
            geometry=c4_geometry,
            action="REFERENCE",
            source_case_sha256=source_sha,
        ),
        _source_arm_row(
            spec=ARM_SPECS[1],
            record=c6["field"],
            geometry=c6_geometry,
            action=str(c6["action"]),
            source_case_sha256=source_sha,
            support=c6["support"],
            utility=c6["utility"],
        ),
    ]
    for spec in ARM_SPECS[2:]:
        direction = _build_direction(
            arm_id=spec.arm_id,
            fixed=fixed,
            moving=atlas,
            learned_fixed=learned_fixed,
            learned_moving=learned_moving,
            initial=initial,
            rms_reference=rms_reference,
        )
        rows.append(
            _materialize(
                case_id=case_id,
                spec=spec,
                direction=direction,
                initial=initial,
                mask=mask,
                fixed_norm=fixed_norm,
                moving_norm=moving_norm,
                decision=decision,
            )
        )
        del direction
        torch.cuda.empty_cache()
    if tuple(row["arm_id"] for row in rows) != tuple(spec.arm_id for spec in ARM_SPECS):
        raise RuntimeError("C7 worker arm order changed")
    payload = {
        "schema": DECISION_CASE_SCHEMA,
        "protocol_id": PROTOCOL_ID,
        "status": "COMPLETE",
        "strict": True,
        "case_id": case_id,
        "shard_index": shard_index,
        "physical_gpu": str(physical_gpu),
        "decision_contract_sha256": decision_sha,
        "descriptor_checkpoint_sha256": descriptor.checkpoint_sha256,
        "descriptor_epoch": descriptor.epoch,
        "descriptor_state_key_count": descriptor.state_key_count,
        "labels_loaded_to_device": False,
        "test_115_authorized": False,
        "test_split_accessed": False,
        "arms": rows,
        "resource": {
            "wall_sec": time.perf_counter() - started,
            "peak_cuda_bytes": int(torch.cuda.max_memory_allocated(device)),
        },
        "execution": dict(execution),
    }
    assert_decision_payload_label_free(payload)
    immutable_json(marker, payload)
    _validate_decision_case_marker(marker, decision, decision_sha, case_id)
    torch.cuda.empty_cache()
    return marker


def run_decision_worker(
    *,
    shard_index: int,
    physical_gpu: str,
    attempt_id: str,
    run_root: Path,
    decision: Mapping[str, Any],
    decision_sha: str,
    device: torch.device,
    execution: Mapping[str, Any],
) -> Path:
    descriptor = _load_descriptor(decision, device)
    case_ids = decision["shards"][str(shard_index)]
    for case_id in case_ids:
        run_decision_case(
            case_id=case_id,
            shard_index=shard_index,
            physical_gpu=physical_gpu,
            run_root=run_root,
            decision=decision,
            decision_sha=decision_sha,
            device=device,
            descriptor=descriptor,
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
        "decision_contract_sha256": decision_sha,
        "descriptor_checkpoint_sha256": descriptor.checkpoint_sha256,
        "labels_loaded": False,
        "test_115_authorized": False,
        "test_split_accessed": False,
        "execution": dict(execution),
    }
    assert_decision_payload_label_free(payload)
    immutable_json(marker, payload)
    return marker


def _load_descriptor_pilot(run_root: Path, decision_sha: str, expected_sha: str | None = None) -> dict[str, Any]:
    path = run_root / "descriptor_pilot.json"
    if not path.is_file() or (expected_sha is not None and sha256_file(path) != expected_sha):
        raise RuntimeError("C7 descriptor pilot is absent or changed")
    payload = load_json(path)
    decision_path = run_root / "decision_contract.json"
    if not decision_path.is_file() or sha256_file(decision_path) != decision_sha:
        raise RuntimeError("C7 descriptor pilot decision contract is absent or changed")
    decision = load_json(decision_path)
    maximum = payload.get("zero_field_local_cost_parity_max_abs")
    support_count = payload.get("zero_field_local_cost_parity_support_count")
    shape = payload.get("feature_shape")
    expected_spatial = list(decision.get("image_inputs", {}).get(PILOT_CASE_ID, {}).get("shape", []))[-3:]
    expected_keys = {
        "schema",
        "protocol_id",
        "status",
        "strict",
        "case_id",
        "decision_contract_sha256",
        "checkpoint_sha256",
        "checkpoint_epoch",
        "state_key_count",
        "feature_shape",
        "feature_dtype",
        "feature_array_sha256",
        "feature_nonconstant",
        "feature_deterministic",
        "feature_requires_grad",
        "zero_field_local_cost_parity_max_abs",
        "zero_field_local_cost_parity_atol",
        "zero_field_local_cost_parity_rtol",
        "zero_field_local_cost_parity_basis",
        "zero_field_local_cost_parity_support_count",
        "zero_field_local_cost_sign_mapping",
        "zero_field_local_cost_offset_order_mapping",
        "sampling_scope",
        "full_native_decoder_sampling_parity_claimed",
        "target_centred_nonconstant_field_semantics",
        "labels_loaded_to_device",
        "test_115_authorized",
        "test_split_accessed",
    }
    if (
        set(payload) != expected_keys
        or payload.get("schema") != "ctcf-search-c7-descriptor-pilot-v2"
        or payload.get("protocol_id") != PROTOCOL_ID
        or payload.get("status") != "PASS"
        or payload.get("strict") is not True
        or payload.get("case_id") != PILOT_CASE_ID
        or payload.get("decision_contract_sha256") != decision_sha
        or payload.get("checkpoint_sha256") != DESCRIPTOR_CHECKPOINT_SHA256
        or payload.get("checkpoint_epoch") != CORRMLP_IXI_LAST_EPOCH
        or payload.get("state_key_count") != CORRMLP_FULL_STATE_KEY_COUNT
        or payload.get("feature_nonconstant") is not True
        or payload.get("feature_deterministic") is not True
        or payload.get("feature_requires_grad") is not False
        or not isinstance(shape, list)
        or len(shape) != 5
        or shape[0] != 1
        or shape[1] != 8
        or shape[-3:] != expected_spatial
        or any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in shape)
        or payload.get("feature_dtype") != "torch.float32"
        or not isinstance(payload.get("feature_array_sha256"), str)
        or len(payload["feature_array_sha256"]) != 64
        or any(character not in "0123456789abcdef" for character in payload["feature_array_sha256"])
        or payload.get("sampling_scope") != "zero_field_local_cost_only"
        or payload.get("full_native_decoder_sampling_parity_claimed") is not False
        or payload.get("target_centred_nonconstant_field_semantics")
        != "intentionally_not_native_neighbor_warp_semantics"
        or payload.get("zero_field_local_cost_sign_mapping")
        != "negative_runner_cost_equals_positive_upstream_correlation"
        or payload.get("zero_field_local_cost_offset_order_mapping") != "identity_lexicographic_zyx_stride1"
        or isinstance(support_count, bool)
        or not isinstance(support_count, int)
        or support_count <= 0
        or isinstance(maximum, bool)
        or not isinstance(maximum, (int, float))
        or not math.isfinite(float(maximum))
        or payload.get("zero_field_local_cost_parity_atol") != ZERO_FIELD_LOCAL_COST_PARITY_ATOL
        or payload.get("zero_field_local_cost_parity_rtol") != ZERO_FIELD_LOCAL_COST_PARITY_RTOL
        or payload.get("zero_field_local_cost_parity_basis") != "ctcf_align_false_grid_sample_vs_corrmlp_integer_slices"
        or float(maximum) > ZERO_FIELD_LOCAL_COST_PARITY_ATOL
        or payload.get("labels_loaded_to_device") is not False
        or payload.get("test_115_authorized") is not False
        or payload.get("test_split_accessed") is not False
    ):
        raise RuntimeError("invalid or altered C7 descriptor pilot")
    assert_decision_payload_label_free(payload)
    return payload


def _write_descriptor_preflight(
    run_root: Path,
    decision: Mapping[str, Any],
    pilot: Mapping[str, Any],
) -> None:
    checkpoint = decision["descriptor_checkpoint"]
    immutable_json(
        run_root / "preflight" / "corrmlp_x1.json",
        {
            "schema": "ctcf-checkpoint-preflight-v1",
            "status": "PASS",
            "checkpoint": checkpoint["path"],
            "sha256": checkpoint["sha256"],
            "ctcf_config": None,
            "time_steps": None,
            "ctcf_l3_svf": None,
            "load": {
                "strict": True,
                "missing_keys": [],
                "allowed_missing_buffers": [],
                "unexpected_keys": [],
                "state_key_count": pilot["state_key_count"],
                "epoch": pilot["checkpoint_epoch"],
            },
        },
    )


def _validate_zero_field_local_cost_parity(
    raw: RawCandidateCostVolume,
    upstream: torch.Tensor,
) -> tuple[float, int]:
    if upstream.shape != raw.costs.shape or upstream.dtype != raw.costs.dtype or upstream.device != raw.costs.device:
        raise RuntimeError("C7 CorrMLP upstream Correlation parity tensors are incompatible")
    runner_values = -raw.costs.masked_select(raw.valid)
    upstream_values = upstream.masked_select(raw.valid)
    comparison = (runner_values - upstream_values).abs()
    if comparison.numel() == 0 or not bool(torch.isfinite(comparison).all()):
        raise RuntimeError("C7 CorrMLP upstream parity support is empty or non-finite")
    maximum = float(comparison.max())
    if not torch.allclose(
        runner_values,
        upstream_values,
        atol=ZERO_FIELD_LOCAL_COST_PARITY_ATOL,
        rtol=ZERO_FIELD_LOCAL_COST_PARITY_RTOL,
    ):
        raise RuntimeError(
            "C7 CorrMLP upstream Correlation parity failed: "
            f"max_abs={maximum}, atol={ZERO_FIELD_LOCAL_COST_PARITY_ATOL}, "
            f"rtol={ZERO_FIELD_LOCAL_COST_PARITY_RTOL}"
        )
    return maximum, int(comparison.numel())


def build_decision_barrier(run_root: Path, decision: Mapping[str, Any], decision_sha: str, attempt_id: str) -> str:
    path = run_root / "decision_barrier.json"
    if path.is_file():
        digest = sha256_file(path)
        _load_barrier(run_root, digest, decision_sha)
        return digest
    hashes: dict[str, str] = {}
    execution_map: dict[str, Any] = {}
    _load_descriptor_pilot(run_root, decision_sha)
    pilot_sha = sha256_file(run_root / "descriptor_pilot.json")
    for case_id in decision["case_ids"]:
        case_path = _decision_case_path(run_root, case_id)
        marker = _validate_decision_case_marker(case_path, decision, decision_sha, case_id)
        hashes[case_id] = sha256_file(case_path)
        execution = marker["execution"]
        execution_map[case_id] = {
            key: execution[key] for key in ("attempt_id", "shard_index", "physical_gpu", "host", "device", "gpu_name")
        }
    payload = {
        "schema": BARRIER_SCHEMA,
        "protocol_id": PROTOCOL_ID,
        "status": "COMPLETE",
        "strict": True,
        "attempt_id": attempt_id,
        "decision_contract_sha256": decision_sha,
        "decision_case_sha256": hashes,
        "decision_execution_by_case": execution_map,
        "descriptor_checkpoint_sha256": DESCRIPTOR_CHECKPOINT_SHA256,
        "descriptor_pilot_sha256": pilot_sha,
        "labels_loaded_to_device": False,
        "test_115_authorized": False,
        "test_split_accessed": False,
        "completed_at_utc": utc_now(),
    }
    assert_decision_payload_label_free(payload)
    return immutable_json(path, payload)


def _load_barrier(run_root: Path, digest: str, decision_sha: str) -> dict[str, Any]:
    path = run_root / "decision_barrier.json"
    if not path.is_file() or sha256_file(path) != digest:
        raise RuntimeError("C7 decision barrier SHA-256 changed")
    payload = load_json(path)
    decision_path = run_root / "decision_contract.json"
    if not decision_path.is_file() or sha256_file(decision_path) != decision_sha:
        raise RuntimeError("C7 decision contract changed after barrier")
    decision = load_json(decision_path)
    hashes = payload.get("decision_case_sha256") or {}
    execution_map = payload.get("decision_execution_by_case") or {}
    if (
        payload.get("schema") != BARRIER_SCHEMA
        or payload.get("protocol_id") != PROTOCOL_ID
        or payload.get("status") != "COMPLETE"
        or payload.get("strict") is not True
        or payload.get("decision_contract_sha256") != decision_sha
        or payload.get("descriptor_checkpoint_sha256") != DESCRIPTOR_CHECKPOINT_SHA256
        or not isinstance(payload.get("descriptor_pilot_sha256"), str)
        or payload.get("labels_loaded_to_device") is not False
        or payload.get("test_115_authorized") is not False
        or payload.get("test_split_accessed") is not False
        or set(hashes) != set(decision.get("case_ids") or [])
        or set(execution_map) != set(decision.get("case_ids") or [])
    ):
        raise RuntimeError("invalid or altered C7 decision barrier")
    _load_descriptor_pilot(run_root, decision_sha, str(payload["descriptor_pilot_sha256"]))
    for case_id, case_sha in hashes.items():
        case_path = _decision_case_path(run_root, case_id)
        if not case_path.is_file() or sha256_file(case_path) != case_sha:
            raise RuntimeError(f"C7 decision snapshot changed after barrier: {case_id}")
        execution = load_json(case_path).get("execution") or {}
        expected = {
            key: execution.get(key)
            for key in ("attempt_id", "shard_index", "physical_gpu", "host", "device", "gpu_name")
        }
        if not json_equivalent(execution_map[case_id], expected):
            raise RuntimeError(f"C7 decision execution map changed after barrier: {case_id}")
    assert_decision_payload_label_free(payload)
    return payload


def _validate_metric_projection(evaluation: Mapping[str, Any], case_ids: Sequence[str]) -> None:
    if evaluation.get("evaluation_label_ids") != list(EVALUATION_LABEL_IDS):
        raise RuntimeError("C7 post-barrier IXI label inventory changed")
    baseline_dice = evaluation.get("evaluation_baseline_dice")
    baseline_per_label = evaluation.get("evaluation_baseline_per_label")
    if (
        not isinstance(baseline_dice, Mapping)
        or set(baseline_dice) != set(case_ids)
        or not isinstance(baseline_per_label, Mapping)
        or set(baseline_per_label) != set(case_ids)
    ):
        raise RuntimeError("C7 post-barrier baseline inventory changed")
    for case_id in case_ids:
        rows = baseline_per_label[case_id]
        if not isinstance(rows, list) or [row.get("label") for row in rows] != list(EVALUATION_LABEL_IDS):
            raise RuntimeError(f"C7 frozen baseline label order changed: {case_id}")
        values = np.asarray([row.get("dice") for row in rows], dtype=np.float64)
        if (
            values.shape != (len(EVALUATION_LABEL_IDS),)
            or not np.isfinite(values).all()
            or not math.isclose(float(baseline_dice[case_id]), float(values.mean()), rel_tol=0.0, abs_tol=1e-12)
        ):
            raise RuntimeError(f"C7 frozen baseline arithmetic changed: {case_id}")


def freeze_evaluation(
    run_root: Path,
    source_sha: str,
    decision: Mapping[str, Any],
    decision_sha: str,
    barrier_sha: str,
) -> str:
    source_path = run_root / "source_contract.json"
    if not source_path.is_file() or sha256_file(source_path) != source_sha:
        raise RuntimeError("C7 source contract SHA-256 changed")
    _load_barrier(run_root, barrier_sha, decision_sha)
    source = load_json(source_path)
    snapshot = source["authenticated_c6"]
    assert_decision_payload_label_free(source)
    evaluation = load_c6_metrics_after_barrier(snapshot)
    _validate_metric_projection(evaluation, snapshot["case_ids"])
    payload = {
        "schema": EVALUATION_CONTRACT_SCHEMA,
        "protocol_id": PROTOCOL_ID,
        "source_contract_sha256": source_sha,
        "decision_contract_sha256": decision_sha,
        "decision_barrier_sha256": barrier_sha,
        "raw_inputs": evaluation["raw_inputs"],
        "evaluation_label_ids": evaluation["evaluation_label_ids"],
        "evaluation_baseline_dice": evaluation["evaluation_baseline_dice"],
        "evaluation_baseline_per_label": evaluation["evaluation_baseline_per_label"],
        "source_c6_evaluation_rows": evaluation["source_rows"],
        "case_ids": snapshot["case_ids"],
        "test_115_authorized": False,
        "test_split_accessed": False,
    }
    digest = immutable_json(run_root / "evaluation_contract.json", payload)
    _immutable_text(run_root / "datasets.tsv", _dataset_tsv(evaluation["raw_inputs"]))
    return digest


def _load_evaluation(
    run_root: Path,
    digest: str,
    source_sha: str,
    decision_sha: str,
    barrier_sha: str,
) -> dict[str, Any]:
    path = run_root / "evaluation_contract.json"
    if not path.is_file() or sha256_file(path) != digest:
        raise RuntimeError("C7 evaluation contract SHA-256 changed")
    payload = load_json(path)
    case_ids = payload.get("case_ids") or []
    if (
        payload.get("schema") != EVALUATION_CONTRACT_SCHEMA
        or payload.get("protocol_id") != PROTOCOL_ID
        or payload.get("source_contract_sha256") != source_sha
        or payload.get("decision_contract_sha256") != decision_sha
        or payload.get("decision_barrier_sha256") != barrier_sha
        or payload.get("test_115_authorized") is not False
        or payload.get("test_split_accessed") is not False
        or len(case_ids) != EXPECTED_CASE_COUNT
        or payload.get("evaluation_label_ids") != list(EVALUATION_LABEL_IDS)
        or set(payload.get("raw_inputs") or {}) != {"atlas", *case_ids}
        or set(payload.get("evaluation_baseline_dice") or {}) != set(case_ids)
        or set(payload.get("evaluation_baseline_per_label") or {}) != set(case_ids)
        or set(payload.get("source_c6_evaluation_rows") or {}) != set(case_ids)
    ):
        raise RuntimeError("invalid or altered C7 evaluation contract")
    _validate_metric_projection(payload, case_ids)
    return payload


def _verify_raw(record: Mapping[str, Any]) -> None:
    path = Path(str(record.get("path", ""))).resolve()
    if (
        not path.is_file()
        or path.stat().st_size != int(record.get("bytes", -1))
        or sha256_file(path) != record.get("sha256")
    ):
        raise RuntimeError(f"C7 frozen raw input changed: {path}")


def _source_eval_parity(
    source_row: Mapping[str, Any],
    baseline_dice: float,
    baseline_labels: torch.Tensor,
    candidate_dice: float,
    candidate_labels: torch.Tensor,
    returned_dice: float,
    returned_labels: torch.Tensor,
    case_id: str,
    arm_id: str,
    labels: Sequence[int],
) -> None:
    observed_labels = [int(row["label"]) for row in source_row["per_label"]]
    if observed_labels != list(labels):
        raise RuntimeError(f"C7 frozen C6 label order changed: {case_id}/{arm_id}")
    expected_baseline = [float(row["baseline_dice"]) for row in source_row["per_label"]]
    expected_candidate = [float(row["candidate_dice"]) for row in source_row["per_label"]]
    expected_returned = [float(row["returned_dice"]) for row in source_row["per_label"]]
    checks = (
        math.isclose(baseline_dice, float(source_row["baseline_dice"]), rel_tol=0.0, abs_tol=1e-12),
        math.isclose(baseline_dice, float(np.mean(expected_baseline)), rel_tol=0.0, abs_tol=1e-12),
        math.isclose(candidate_dice, float(source_row["candidate_dice"]), rel_tol=0.0, abs_tol=1e-12),
        math.isclose(returned_dice, float(source_row["returned_dice"]), rel_tol=0.0, abs_tol=1e-12),
        all(
            math.isclose(float(left), right, rel_tol=0.0, abs_tol=1e-12)
            for left, right in zip(baseline_labels, expected_baseline, strict=True)
        ),
        all(
            math.isclose(float(left), right, rel_tol=0.0, abs_tol=1e-12)
            for left, right in zip(candidate_labels, expected_candidate, strict=True)
        ),
        all(
            math.isclose(float(left), right, rel_tol=0.0, abs_tol=1e-12)
            for left, right in zip(returned_labels, expected_returned, strict=True)
        ),
    )
    if not all(checks):
        raise RuntimeError(f"C7 frozen C6 evaluation parity failed: {case_id}/{arm_id}")


def _validate_evaluation_case_marker(
    path: Path,
    *,
    decision: Mapping[str, Any],
    evaluation: Mapping[str, Any],
    case_id: str,
    decision_sha: str,
    barrier_sha: str,
    evaluation_sha: str,
    decision_case_sha: str,
    labels: Sequence[int],
) -> dict[str, Any]:
    payload = load_json(path)
    rows = payload.get("arms") or []
    decision_path = path.with_name("decision_complete.json")
    if not decision_path.is_file() or sha256_file(decision_path) != decision_case_sha:
        raise RuntimeError(f"C7 frozen decision marker is absent or changed: {case_id}")
    decision_rows = {row["arm_id"]: row for row in load_json(decision_path).get("arms", [])}
    if (
        payload.get("schema") != EVALUATION_CASE_SCHEMA
        or payload.get("protocol_id") != PROTOCOL_ID
        or payload.get("status") != "COMPLETE"
        or payload.get("strict") is not True
        or payload.get("case_id") != case_id
        or payload.get("decision_contract_sha256") != decision_sha
        or payload.get("decision_barrier_sha256") != barrier_sha
        or payload.get("evaluation_contract_sha256") != evaluation_sha
        or payload.get("decision_case_sha256") != decision_case_sha
        or payload.get("labels_loaded_after_barrier") is not True
        or payload.get("test_115_authorized") is not False
        or payload.get("test_split_accessed") is not False
        or payload.get("labels") != list(labels)
        or tuple(row.get("arm_id") for row in rows) != tuple(spec.arm_id for spec in ARM_SPECS)
    ):
        raise RuntimeError(f"invalid or altered C7 evaluation marker: {case_id}")
    _validate_case_execution(payload, decision, case_id, phase="evaluation")
    for row, spec in zip(rows, ARM_SPECS, strict=True):
        if (
            row.get("arm_index") != spec.arm_index
            or row.get("action") not in {"REFERENCE", "ACCEPT", "ROLLBACK"}
            or row.get("action") != decision_rows.get(spec.arm_id, {}).get("action")
            or row.get("source_parity_verified")
            is not (True if spec.arm_id in {REFERENCE_ARM_ID, SOURCE_CONTEXT_ARM_ID} else None)
        ):
            raise RuntimeError(f"C7 evaluation arm identity changed: {case_id}/{spec.arm_id}")
        for key in (
            "baseline_dice",
            "candidate_dice",
            "capacity_delta_vs_initial",
            "returned_dice",
            "returned_delta_vs_initial",
        ):
            value = row.get(key)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                raise RuntimeError(f"C7 evaluation value is non-finite: {case_id}/{spec.arm_id}/{key}")
        per_label = row.get("per_label") or []
        if [item.get("label") for item in per_label] != list(labels):
            raise RuntimeError(f"C7 per-label order changed: {case_id}/{spec.arm_id}")
        for item in per_label:
            for key in ("baseline_dice", "candidate_dice", "returned_dice"):
                value = item.get(key)
                if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                    raise RuntimeError(f"C7 per-label value is non-finite: {case_id}/{spec.arm_id}/{key}")
        baseline_values = np.asarray([float(item["baseline_dice"]) for item in per_label], dtype=np.float64)
        candidate_values = np.asarray([float(item["candidate_dice"]) for item in per_label], dtype=np.float64)
        returned_values = np.asarray([float(item["returned_dice"]) for item in per_label], dtype=np.float64)
        expected_returned = candidate_values if row["action"] in {"REFERENCE", "ACCEPT"} else baseline_values
        checks = (
            math.isclose(float(row["baseline_dice"]), float(baseline_values.mean()), rel_tol=0.0, abs_tol=1e-12),
            math.isclose(float(row["candidate_dice"]), float(candidate_values.mean()), rel_tol=0.0, abs_tol=1e-12),
            math.isclose(float(row["returned_dice"]), float(returned_values.mean()), rel_tol=0.0, abs_tol=1e-12),
            np.array_equal(returned_values, expected_returned),
            math.isclose(
                float(row["capacity_delta_vs_initial"]),
                float(row["candidate_dice"]) - float(row["baseline_dice"]),
                rel_tol=0.0,
                abs_tol=1e-15,
            ),
            math.isclose(
                float(row["returned_delta_vs_initial"]),
                float(row["returned_dice"]) - float(row["baseline_dice"]),
                rel_tol=0.0,
                abs_tol=1e-15,
            ),
        )
        if not all(checks):
            raise RuntimeError(f"C7 evaluation arithmetic or action semantics changed: {case_id}/{spec.arm_id}")
        frozen_baseline_rows = evaluation["evaluation_baseline_per_label"][case_id]
        if [item.get("label") for item in frozen_baseline_rows] != list(labels) or any(
            not math.isclose(float(observed), float(expected["dice"]), rel_tol=0.0, abs_tol=1e-12)
            for observed, expected in zip(baseline_values, frozen_baseline_rows, strict=True)
        ):
            raise RuntimeError(f"C7 frozen baseline parity changed: {case_id}/{spec.arm_id}")
        if spec.arm_id in {REFERENCE_ARM_ID, SOURCE_CONTEXT_ARM_ID}:
            source_key = "reference" if spec.arm_id == REFERENCE_ARM_ID else "context"
            source = evaluation["source_c6_evaluation_rows"][case_id][source_key]
            _source_eval_parity(
                source,
                float(row["baseline_dice"]),
                baseline_values,
                float(row["candidate_dice"]),
                candidate_values,
                float(row["returned_dice"]),
                returned_values,
                case_id,
                spec.arm_id,
                labels,
            )
    return payload


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
        _validate_evaluation_case_marker(
            marker,
            decision=decision,
            evaluation=evaluation,
            case_id=case_id,
            decision_sha=decision_sha,
            barrier_sha=barrier_sha,
            evaluation_sha=evaluation_sha,
            decision_case_sha=barrier["decision_case_sha256"][case_id],
            labels=labels,
        )
        return marker
    decision_path = _decision_case_path(run_root, case_id)
    if sha256_file(decision_path) != barrier["decision_case_sha256"][case_id]:
        raise RuntimeError(f"C7 decision snapshot changed before evaluation: {case_id}")
    decision_case = _validate_decision_case_marker(decision_path, decision, decision_sha, case_id)
    moving_image, fixed_image, moving_seg, fixed_seg = dataset_item
    if array_sha256(moving_image) != decision["image_inputs"]["atlas"]["array_sha256"]:
        raise RuntimeError("C7 evaluation atlas differs from the decision cache")
    if array_sha256(fixed_image) != decision["image_inputs"][case_id]["array_sha256"]:
        raise RuntimeError(f"C7 evaluation image differs from the decision cache: {case_id}")
    labels_tuple = tuple(int(value) for value in labels)
    if labels_tuple != EVALUATION_LABEL_IDS or labels_tuple != tuple(evaluation["evaluation_label_ids"]):
        raise RuntimeError("C7 IXI label order changed")
    moving_seg = moving_seg.unsqueeze(0).to(device)
    fixed_seg = fixed_seg.unsqueeze(0).to(device)
    initial = load_field(decision, decision["source_initial"][case_id]["field"]).to(device)
    baseline_labels = dice_per_label(
        sample_at_psi(moving_seg.float(), initial, mode="nearest").long(), fixed_seg.long(), labels_tuple
    )
    baseline = float(baseline_labels.mean())
    if not math.isclose(baseline, float(evaluation["evaluation_baseline_dice"][case_id]), rel_tol=0.0, abs_tol=1e-8):
        raise RuntimeError(f"C7 baseline Dice differs from frozen C6: {case_id}")
    frozen_baseline_rows = evaluation["evaluation_baseline_per_label"][case_id]
    if [row.get("label") for row in frozen_baseline_rows] != list(EVALUATION_LABEL_IDS) or any(
        not math.isclose(float(observed), float(expected["dice"]), rel_tol=0.0, abs_tol=1e-8)
        for observed, expected in zip(baseline_labels, frozen_baseline_rows, strict=True)
    ):
        raise RuntimeError(f"C7 baseline per-label Dice differs from frozen C6: {case_id}")
    rows = []
    for arm in decision_case["arms"]:
        candidate = load_field(decision, arm["candidate_field"]).to(device)
        candidate_labels = dice_per_label(
            sample_at_psi(moving_seg.float(), candidate, mode="nearest").long(), fixed_seg.long(), labels_tuple
        )
        candidate_dice = float(candidate_labels.mean())
        returned_labels = candidate_labels if arm["action"] in {"ACCEPT", "REFERENCE"} else baseline_labels
        returned_dice = float(returned_labels.mean())
        source_parity = None
        if arm["arm_id"] == REFERENCE_ARM_ID:
            source = evaluation["source_c6_evaluation_rows"][case_id]["reference"]
            _source_eval_parity(
                source,
                baseline,
                baseline_labels,
                candidate_dice,
                candidate_labels,
                returned_dice,
                returned_labels,
                case_id,
                arm["arm_id"],
                labels_tuple,
            )
            source_parity = True
        elif arm["arm_id"] == SOURCE_CONTEXT_ARM_ID:
            source = evaluation["source_c6_evaluation_rows"][case_id]["context"]
            _source_eval_parity(
                source,
                baseline,
                baseline_labels,
                candidate_dice,
                candidate_labels,
                returned_dice,
                returned_labels,
                case_id,
                arm["arm_id"],
                labels_tuple,
            )
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
                "source_parity_verified": source_parity,
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
        raise RuntimeError(f"C7 decision snapshot changed during evaluation: {case_id}")
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
        "shard_index": int(execution["shard_index"]),
        "physical_gpu": str(execution["physical_gpu"]),
        "labels_loaded_after_barrier": True,
        "test_115_authorized": False,
        "test_split_accessed": False,
        "labels": list(labels_tuple),
        "arms": rows,
        "execution": dict(execution),
    }
    immutable_json(marker, payload)
    _validate_evaluation_case_marker(
        marker,
        decision=decision,
        evaluation=evaluation,
        case_id=case_id,
        decision_sha=decision_sha,
        barrier_sha=barrier_sha,
        evaluation_sha=evaluation_sha,
        decision_case_sha=barrier["decision_case_sha256"][case_id],
        labels=labels_tuple,
    )
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
        "test_115_authorized": False,
        "test_split_accessed": False,
        "execution": dict(execution),
    }
    immutable_json(marker, payload)
    return marker


def _build_evaluation_barrier(
    run_root: Path,
    decision: Mapping[str, Any],
    decision_sha: str,
    barrier: Mapping[str, Any],
    barrier_sha: str,
    evaluation: Mapping[str, Any],
    evaluation_sha: str,
    labels: Sequence[int],
) -> tuple[str, dict[str, str], dict[str, Any]]:
    hashes: dict[str, str] = {}
    execution_map: dict[str, Any] = {}
    for case_id in decision["case_ids"]:
        path = _evaluation_case_path(run_root, case_id)
        marker = _validate_evaluation_case_marker(
            path,
            decision=decision,
            evaluation=evaluation,
            case_id=case_id,
            decision_sha=decision_sha,
            barrier_sha=barrier_sha,
            evaluation_sha=evaluation_sha,
            decision_case_sha=barrier["decision_case_sha256"][case_id],
            labels=labels,
        )
        hashes[case_id] = sha256_file(path)
        execution = marker["execution"]
        execution_map[case_id] = {
            key: execution[key] for key in ("attempt_id", "shard_index", "physical_gpu", "host", "device", "gpu_name")
        }
    payload = {
        "schema": EVALUATION_BARRIER_SCHEMA,
        "protocol_id": PROTOCOL_ID,
        "status": "COMPLETE",
        "strict": True,
        "decision_contract_sha256": decision_sha,
        "decision_barrier_sha256": barrier_sha,
        "evaluation_contract_sha256": evaluation_sha,
        "evaluation_case_sha256": hashes,
        "evaluation_execution_by_case": execution_map,
        "test_115_authorized": False,
        "test_split_accessed": False,
    }
    digest = immutable_json(run_root / "evaluation_barrier.json", payload)
    return digest, hashes, execution_map


def _direction_diagnostics(row: Mapping[str, Any], label: str) -> dict[str, float | None]:
    if row["arm_id"] not in COMPUTED_ARM_IDS:
        return dict.fromkeys(DIRECTION_DIAGNOSTIC_FIELDS)
    direction = row["direction"]
    reference = float(direction["reference_rms"])
    pre = float(direction["pre_normalization_rms"])
    retentions = [float(stage["clip_retention"]) for stage in direction["stages"]]
    if not (math.isfinite(reference) and reference > 0 and math.isfinite(pre) and pre > 0 and retentions):
        raise RuntimeError(f"invalid C7 direction diagnostics: {label}")
    return {
        "pre_normalization_rms": pre,
        "rematch_gain": reference / pre,
        "normalized_rms": float(direction["normalized_rms"]),
        "stage_clip_retention_min": min(retentions),
        "stage_clip_retention_mean": sum(retentions) / len(retentions),
        "final_clip_retained_norm_ratio": float(row["operator"]["retained_norm_ratio"]),
    }


def _paired_maps(prefix: str, values: Mapping[str, np.ndarray]) -> dict[str, Any]:
    return simultaneous_paired_summaries(prefix, values)


def finalize(
    run_root: Path,
    decision: Mapping[str, Any],
    decision_sha: str,
    barrier: Mapping[str, Any],
    barrier_sha: str,
    evaluation: Mapping[str, Any],
    evaluation_sha: str,
) -> dict[str, str]:
    pilot = _load_descriptor_pilot(
        run_root,
        decision_sha,
        str(barrier["descriptor_pilot_sha256"]),
    )
    descriptor_valid = (
        pilot.get("status") == "PASS"
        and pilot.get("feature_nonconstant") is True
        and pilot.get("feature_deterministic") is True
        and pilot.get("feature_requires_grad") is False
        and pilot.get("full_native_decoder_sampling_parity_claimed") is False
    )
    labels = tuple(int(value) for value in evaluation["evaluation_label_ids"])
    if labels != EVALUATION_LABEL_IDS:
        raise RuntimeError("C7 finalizer IXI label inventory changed")
    evaluation_barrier_sha, evaluation_hashes, evaluation_execution_map = _build_evaluation_barrier(
        run_root,
        decision,
        decision_sha,
        barrier,
        barrier_sha,
        evaluation,
        evaluation_sha,
        labels,
    )
    arm_ids = tuple(spec.arm_id for spec in ARM_SPECS)
    dice = {arm_id: [] for arm_id in arm_ids}
    returned = {arm_id: [] for arm_id in arm_ids}
    sdlogj = {arm_id: [] for arm_id in arm_ids}
    returned_sdlogj = {arm_id: [] for arm_id in arm_ids}
    folds = {arm_id: [] for arm_id in arm_ids}
    label_candidate = {arm_id: {label: [] for label in labels} for arm_id in arm_ids}
    label_returned = {arm_id: {label: [] for label in labels} for arm_id in arm_ids}
    label_initial = {label: [] for label in labels}
    initial_dice: list[float] = []
    diagnostics = {arm_id: {name: [] for name in DIRECTION_DIAGNOSTIC_FIELDS} for arm_id in arm_ids}
    per_case: list[dict[str, Any]] = []
    per_label: list[dict[str, Any]] = []
    for case_id in decision["case_ids"]:
        dpath = _decision_case_path(run_root, case_id)
        epath = _evaluation_case_path(run_root, case_id)
        if sha256_file(dpath) != barrier["decision_case_sha256"][case_id]:
            raise RuntimeError(f"C7 decision changed before finalization: {case_id}")
        if sha256_file(epath) != evaluation_hashes[case_id]:
            raise RuntimeError(f"C7 evaluation changed before finalization: {case_id}")
        drows = {row["arm_id"]: row for row in load_json(dpath)["arms"]}
        erows = {row["arm_id"]: row for row in load_json(epath)["arms"]}
        baseline_sdlogj = _metric_value(
            decision["baseline_geometry"][case_id],
            MATHEMATICAL_SDLOGJ_CROP2,
            f"{case_id}/initial",
        )
        if set(drows) != set(arm_ids) or set(erows) != set(arm_ids):
            raise RuntimeError(f"C7 final arm inventory changed: {case_id}")
        for arm_id in arm_ids:
            drow, erow = drows[arm_id], erows[arm_id]
            sd = _metric_value(drow["geometry"], MATHEMATICAL_SDLOGJ_CROP2, f"{case_id}/{arm_id}")
            corner = float(drow["geometry"][DIGITAL_DECOMPOSITION]["components"]["corner_union_violation_fraction"])
            diag = _direction_diagnostics(drow, f"{case_id}/{arm_id}")
            for name, value in diag.items():
                if value is not None:
                    diagnostics[arm_id][name].append(value)
            dice[arm_id].append(float(erow["candidate_dice"]))
            returned[arm_id].append(float(erow["returned_dice"]))
            returned_sd = baseline_sdlogj if erow["action"] == "ROLLBACK" else sd
            returned_sdlogj[arm_id].append(returned_sd)
            if arm_id == REFERENCE_ARM_ID:
                initial_dice.append(float(erow["baseline_dice"]))
            sdlogj[arm_id].append(sd)
            folds[arm_id].append(corner)
            minimum_retention = (
                None
                if arm_id not in COMPUTED_ARM_IDS
                else min(
                    float(diag["stage_clip_retention_min"]),
                    float(diag["final_clip_retained_norm_ratio"]),
                )
            )
            per_case.append(
                {
                    "case_id": case_id,
                    "arm_id": arm_id,
                    "action": erow["action"],
                    "baseline_dice": erow["baseline_dice"],
                    "candidate_dice": erow["candidate_dice"],
                    "returned_dice": erow["returned_dice"],
                    "sdlogj": sd,
                    "returned_sdlogj": returned_sd,
                    "corner_fold_fraction": corner,
                    "minimum_clip_retention": minimum_retention,
                    **diag,
                }
            )
            for row in erow["per_label"]:
                label = int(row["label"])
                label_initial[label].append(float(row["baseline_dice"])) if arm_id == REFERENCE_ARM_ID else None
                label_candidate[arm_id][label].append(float(row["candidate_dice"]))
                label_returned[arm_id][label].append(float(row["returned_dice"]))
                per_label.append({"case_id": case_id, "arm_id": arm_id, **row})

    def vector(table: Mapping[str, Sequence[float]], arm_id: str) -> np.ndarray:
        value = np.asarray(table[arm_id], dtype=np.float64)
        if value.shape != (EXPECTED_CASE_COUNT,) or not np.isfinite(value).all():
            raise RuntimeError(f"C7 final vector is incomplete: {arm_id}")
        return value

    capacity = _paired_maps(
        "c7_capacity_vs_c4",
        {arm: vector(dice, arm) - vector(dice, REFERENCE_ARM_ID) for arm in SELECTABLE_ARM_IDS},
    )
    causal = _paired_maps(
        "c7_causal_vs_matched_intensity",
        {arm: vector(dice, arm) - vector(dice, MATCHED_CONTROL_ARM_ID) for arm in SELECTABLE_ARM_IDS},
    )
    returned_family = _paired_maps(
        "c7_returned_vs_c4",
        {arm: vector(returned, arm) - vector(dice, REFERENCE_ARM_ID) for arm in SELECTABLE_ARM_IDS},
    )
    safety_arm_ids = (MATCHED_CONTROL_ARM_ID, *SELECTABLE_ARM_IDS)
    initial_vector = np.asarray(initial_dice, dtype=np.float64)
    if initial_vector.shape != (EXPECTED_CASE_COUNT,) or not np.isfinite(initial_vector).all():
        raise RuntimeError("C7 initial Dice vector is incomplete")
    computed_vs_initial = _paired_maps(
        "c7_computed_vs_initial",
        {
            **{f"{arm}::candidate": vector(dice, arm) - initial_vector for arm in COMPUTED_ARM_IDS},
            **{f"{arm}::returned": vector(returned, arm) - initial_vector for arm in COMPUTED_ARM_IDS},
        },
    )
    hybrid_vs_learned = _paired_maps(
        "c7_hybrid_vs_learned",
        {
            "candidate": vector(dice, SELECTABLE_ARM_IDS[1]) - vector(dice, SELECTABLE_ARM_IDS[0]),
            "returned": vector(returned, SELECTABLE_ARM_IDS[1]) - vector(returned, SELECTABLE_ARM_IDS[0]),
        },
    )
    matched_vs_c6_context = _paired_maps(
        "c7_matched_intensity_vs_c6_context",
        {
            "candidate": vector(dice, MATCHED_CONTROL_ARM_ID) - vector(dice, SOURCE_CONTEXT_ARM_ID),
            "returned": vector(returned, MATCHED_CONTROL_ARM_ID) - vector(returned, SOURCE_CONTEXT_ARM_ID),
        },
    )
    sd_family = _paired_maps(
        "c7_sdlogj_vs_c4",
        {arm: vector(sdlogj, arm) - vector(sdlogj, REFERENCE_ARM_ID) for arm in safety_arm_ids},
    )

    def regional(family: str, source: Mapping[str, Mapping[int, list[float]]], reference: str) -> dict[str, Any]:
        values = {}
        for arm in safety_arm_ids:
            for label in labels:
                left = np.asarray(source[arm][label], dtype=np.float64)
                if reference == "initial":
                    right = np.asarray(label_initial[label], dtype=np.float64)
                else:
                    right = np.asarray(label_candidate[REFERENCE_ARM_ID][label], dtype=np.float64)
                values[f"{arm}::label_{label}"] = left - right
        return _paired_maps(family, values)

    candidate_initial = regional("c7_candidate_regional_vs_initial", label_candidate, "initial")
    candidate_c4 = regional("c7_candidate_regional_vs_c4", label_candidate, "c4")
    returned_initial = regional("c7_returned_regional_vs_initial", label_returned, "initial")
    returned_c4 = regional("c7_returned_regional_vs_c4", label_returned, "c4")

    def regional_rows(table: Mapping[str, Any], arm: str) -> list[tuple[int, Any]]:
        return [(label, table[f"{arm}::label_{label}"]) for label in labels]

    def regional_pass(table: Mapping[str, Any], arm: str) -> bool:
        return all(
            table[f"{arm}::label_{label}"].ci_low
            > (RISK_LABEL_CI_LOW_VS_C4_MIN_STRICT if label in RISK_LABEL_IDS else ALL_LABEL_CI_LOW_VS_C4_MIN_STRICT)
            for label in labels
        )

    matched_retentions = [
        min(stage, final)
        for stage, final in zip(
            diagnostics[MATCHED_CONTROL_ARM_ID]["stage_clip_retention_min"],
            diagnostics[MATCHED_CONTROL_ARM_ID]["final_clip_retained_norm_ratio"],
            strict=True,
        )
    ]
    matched_safety = (
        all(value == 0.0 for value in folds[MATCHED_CONTROL_ARM_ID])
        and sd_family[MATCHED_CONTROL_ARM_ID].ci_high <= SDLOGJ_CI_HIGH_VS_C4_MAX
        and regional_pass(candidate_initial, MATCHED_CONTROL_ARM_ID)
        and regional_pass(candidate_c4, MATCHED_CONTROL_ARM_ID)
        and regional_pass(returned_initial, MATCHED_CONTROL_ARM_ID)
        and regional_pass(returned_c4, MATCHED_CONTROL_ARM_ID)
        and len(matched_retentions) == EXPECTED_CASE_COUNT
        and float(np.median(matched_retentions)) >= CLIP_RETENTION_MEDIAN_MIN
        and sum(value >= CLIP_RETENTION_PER_CASE_MIN for value in matched_retentions) >= CLIP_RETENTION_GE_090_MIN_COUNT
    )
    assessments = []
    for arm in SELECTABLE_ARM_IDS:
        retentions = [
            min(stage, final)
            for stage, final in zip(
                diagnostics[arm]["stage_clip_retention_min"],
                diagnostics[arm]["final_clip_retained_norm_ratio"],
                strict=True,
            )
        ]
        if len(retentions) != EXPECTED_CASE_COUNT:
            raise RuntimeError(f"C7 clip retention is incomplete: {arm}")
        assessments.append(
            assess_arm(
                arm,
                descriptor_valid=descriptor_valid,
                capacity_vs_c4=capacity[arm],
                causal_vs_intensity=causal[arm],
                returned_vs_c4=returned_family[arm],
                sdlogj_vs_c4=sd_family[arm],
                candidate_regional_vs_initial=regional_rows(candidate_initial, arm),
                candidate_regional_vs_c4=regional_rows(candidate_c4, arm),
                returned_regional_vs_initial=regional_rows(returned_initial, arm),
                returned_regional_vs_c4=regional_rows(returned_c4, arm),
                folds_all_zero=all(value == 0.0 for value in folds[arm]),
                clip_retention_median=float(np.median(retentions)),
                clip_retention_ge_090_count=sum(value >= CLIP_RETENTION_PER_CASE_MIN for value in retentions),
                matched_control_safety_passed=matched_safety,
            )
        )
    branch = select_branch(
        assessments,
        {arm: float(vector(returned, arm).mean()) for arm in SELECTABLE_ARM_IDS},
        {arm: float(vector(returned_sdlogj, arm).mean()) for arm in SELECTABLE_ARM_IDS},
        integrity_passed=True,
    )
    summary = {
        "schema": "ctcf-search-c7-summary-v1",
        "protocol_id": PROTOCOL_ID,
        "status": "COMPLETE",
        "n_cases": EXPECTED_CASE_COUNT,
        "policy_sha256": C7_POLICY_SHA256,
        "descriptor_checkpoint_sha256": DESCRIPTOR_CHECKPOINT_SHA256,
        "descriptor_pilot_sha256": barrier["descriptor_pilot_sha256"],
        "descriptor_valid": descriptor_valid,
        "test_115_authorized": False,
        "test_split_accessed": False,
        "initial": {
            "dice_mean": float(initial_vector.mean()),
            "dice_median": float(np.median(initial_vector)),
        },
        "absolute": {
            arm: {
                "dice_mean": float(vector(dice, arm).mean()),
                "dice_median": float(np.median(vector(dice, arm))),
                "returned_dice_mean": float(vector(returned, arm).mean()),
                "sdlogj_mean": float(vector(sdlogj, arm).mean()),
                "returned_sdlogj_mean": float(vector(returned_sdlogj, arm).mean()),
                "corner_folds_all_zero": all(value == 0.0 for value in folds[arm]),
                "minimum_clip_retention_median": (
                    None
                    if arm not in COMPUTED_ARM_IDS
                    else float(
                        np.median(
                            np.minimum(
                                diagnostics[arm]["stage_clip_retention_min"],
                                diagnostics[arm]["final_clip_retained_norm_ratio"],
                            )
                        )
                    )
                ),
                "minimum_clip_retention_ge_090_count": (
                    None
                    if arm not in COMPUTED_ARM_IDS
                    else int(
                        np.count_nonzero(
                            np.minimum(
                                diagnostics[arm]["stage_clip_retention_min"],
                                diagnostics[arm]["final_clip_retained_norm_ratio"],
                            )
                            >= CLIP_RETENTION_PER_CASE_MIN
                        )
                    )
                ),
                **{
                    f"{name}_mean": (
                        None
                        if not diagnostics[arm][name]
                        else float(np.asarray(diagnostics[arm][name], dtype=np.float64).mean())
                    )
                    for name in DIRECTION_DIAGNOSTIC_FIELDS
                },
            }
            for arm in arm_ids
        },
        "capacity_vs_c4": {key: asdict(value) for key, value in capacity.items()},
        "causal_vs_matched_intensity": {key: asdict(value) for key, value in causal.items()},
        "returned_vs_c4": {key: asdict(value) for key, value in returned_family.items()},
        "computed_vs_initial": {key: asdict(value) for key, value in computed_vs_initial.items()},
        "hybrid_vs_learned": {key: asdict(value) for key, value in hybrid_vs_learned.items()},
        "matched_intensity_vs_c6_context": {key: asdict(value) for key, value in matched_vs_c6_context.items()},
        "sdlogj_vs_c4": {key: asdict(value) for key, value in sd_family.items()},
        "candidate_regional_vs_initial": {key: asdict(value) for key, value in candidate_initial.items()},
        "candidate_regional_vs_c4": {key: asdict(value) for key, value in candidate_c4.items()},
        "returned_regional_vs_initial": {key: asdict(value) for key, value in returned_initial.items()},
        "returned_regional_vs_c4": {key: asdict(value) for key, value in returned_c4.items()},
        "matched_intensity_safety_passed": matched_safety,
        "assessments": [asdict(row) for row in assessments],
        "next_branch": branch,
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
            "returned_sdlogj",
            "corner_fold_fraction",
            "minimum_clip_retention",
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
        "schema": "ctcf-search-c7-run-manifest-v1",
        "protocol_id": PROTOCOL_ID,
        "status": "COMPLETE",
        "policy_sha256": C7_POLICY_SHA256,
        "descriptor_checkpoint_sha256": DESCRIPTOR_CHECKPOINT_SHA256,
        "descriptor_pilot_sha256": barrier["descriptor_pilot_sha256"],
        "decision_contract_sha256": decision_sha,
        "decision_barrier_sha256": barrier_sha,
        "evaluation_contract_sha256": evaluation_sha,
        "evaluation_barrier_sha256": evaluation_barrier_sha,
        "decision_case_sha256": barrier["decision_case_sha256"],
        "evaluation_case_sha256": evaluation_hashes,
        "decision_execution_by_case": barrier["decision_execution_by_case"],
        "evaluation_execution_by_case": evaluation_execution_map,
        "files": files,
        "test_115_authorized": False,
        "test_split_accessed": False,
        "completed_at_utc": utc_now(),
    }
    atomic_write_json(run_root / "c7_manifest.json", manifest)
    files["c7_manifest"] = sha256_file(run_root / "c7_manifest.json")
    return files


def _descriptor_pilot_checks(
    *,
    run_root: Path,
    decision: Mapping[str, Any],
    decision_sha: str,
    descriptor: FrozenCorrMLPDescriptor,
    device: torch.device,
) -> str:
    path = run_root / "descriptor_pilot.json"
    if path.is_file():
        payload = _load_descriptor_pilot(run_root, decision_sha)
        _write_descriptor_preflight(run_root, decision, payload)
        return sha256_file(path)
    case_id = str(decision["pilot_case_id"])
    atlas = torch.from_numpy(load_image(decision, decision["image_inputs"]["atlas"])).unsqueeze(0).to(device)
    fixed = torch.from_numpy(load_image(decision, decision["image_inputs"][case_id])).unsqueeze(0).to(device)
    with torch.inference_mode():
        fixed_first = extract_corrmlp_x1(descriptor.model, fixed.float())
        fixed_second = extract_corrmlp_x1(descriptor.model, fixed.float())
        moving = extract_corrmlp_x1(descriptor.model, atlas.float())
    if not torch.equal(fixed_first, fixed_second):
        raise RuntimeError("C7 CorrMLP x1 extraction is not deterministic")
    if fixed_first.requires_grad or moving.requires_grad:
        raise RuntimeError("C7 CorrMLP pilot features require gradients")
    if float(fixed_first.float().std(unbiased=False)) <= 0.0 or float(moving.float().std(unbiased=False)) <= 0.0:
        raise RuntimeError("C7 CorrMLP pilot features are constant")
    zero = torch.zeros((1, 3, *fixed.shape[-3:]), dtype=torch.float32, device=device)
    mask = geometry_mask(tuple(zero.shape[-3:]), COMMON_EVIDENCE_COLLAR, device)
    raw = _common_padding_support(fixed_first, moving, zero, mask, 1)
    upstream = Correlation(max_disp=1, use_checkpoint=False).to(device).eval()(fixed_first, moving)
    maximum, support_count = _validate_zero_field_local_cost_parity(raw, upstream)
    payload = {
        "schema": "ctcf-search-c7-descriptor-pilot-v2",
        "protocol_id": PROTOCOL_ID,
        "status": "PASS",
        "strict": True,
        "case_id": case_id,
        "decision_contract_sha256": decision_sha,
        "checkpoint_sha256": descriptor.checkpoint_sha256,
        "checkpoint_epoch": descriptor.epoch,
        "state_key_count": descriptor.state_key_count,
        "feature_shape": list(fixed_first.shape),
        "feature_dtype": str(fixed_first.dtype),
        "feature_array_sha256": array_sha256(fixed_first),
        "feature_nonconstant": True,
        "feature_deterministic": True,
        "feature_requires_grad": False,
        "zero_field_local_cost_parity_max_abs": maximum,
        "zero_field_local_cost_parity_atol": ZERO_FIELD_LOCAL_COST_PARITY_ATOL,
        "zero_field_local_cost_parity_rtol": ZERO_FIELD_LOCAL_COST_PARITY_RTOL,
        "zero_field_local_cost_parity_basis": "ctcf_align_false_grid_sample_vs_corrmlp_integer_slices",
        "zero_field_local_cost_parity_support_count": support_count,
        "zero_field_local_cost_sign_mapping": "negative_runner_cost_equals_positive_upstream_correlation",
        "zero_field_local_cost_offset_order_mapping": "identity_lexicographic_zyx_stride1",
        "sampling_scope": "zero_field_local_cost_only",
        "full_native_decoder_sampling_parity_claimed": False,
        "target_centred_nonconstant_field_semantics": "intentionally_not_native_neighbor_warp_semantics",
        "labels_loaded_to_device": False,
        "test_115_authorized": False,
        "test_split_accessed": False,
    }
    assert_decision_payload_label_free(payload)
    digest = immutable_json(path, payload)
    _load_descriptor_pilot(run_root, decision_sha, digest)
    _write_descriptor_preflight(run_root, decision, payload)
    return digest


def _execution(
    decision: Mapping[str, Any],
    phase: str,
    attempt_id: str,
    shard_index: int,
    physical_gpu: str,
    device: torch.device,
) -> dict[str, Any]:
    return {
        "phase": phase,
        "attempt_id": attempt_id,
        "shard_index": shard_index,
        "physical_gpu": physical_gpu,
        "host": platform.node(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(device),
        "seed": decision["seed"],
        "deterministic": True,
        "labels_loaded_to_device": phase == "evaluation",
    }


def selfcheck_stage(args: argparse.Namespace) -> int:
    assert_frozen_policy()
    if DEFAULT_MOMENT_REDUCTION != "centered_two_pass_fp32":
        raise RuntimeError("C7 runner requires centered two-pass FP32 candidate moments")
    report = {
        "schema": f"ctcf-search-c7-selfcheck-{SCHEMA_VERSION}",
        "protocol_id": PROTOCOL_ID,
        "status": "PASS",
        "policy_sha256": C7_POLICY_SHA256,
        "arm_ids": [row.arm_id for row in ARM_SPECS],
        "computed_arm_ids": list(COMPUTED_ARM_IDS),
        "selectable_arm_ids": list(SELECTABLE_ARM_IDS),
        "descriptor_checkpoint_sha256": DESCRIPTOR_CHECKPOINT_SHA256,
        "descriptor_epoch": CORRMLP_IXI_LAST_EPOCH,
        "descriptor_state_key_count": CORRMLP_FULL_STATE_KEY_COUNT,
        "candidate_moment_reduction": DEFAULT_MOMENT_REDUCTION,
        "strict_fp32_backend": {
            "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
            "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
            "float32_matmul_precision": torch.get_float32_matmul_precision(),
        },
        "convolutional_padding_margin": CORRMLP_X1_CONV_PADDING_MARGIN,
        "instance_normalization_scope": "global_not_a_finite_total_receptive_field",
        "test_115_authorized": TEST_115_AUTHORIZED,
    }
    atomic_write_json(args.output, report)
    print(json.dumps(report, indent=2))
    return 0


def prepare_stage(args: argparse.Namespace) -> int:
    assert_frozen_policy()
    if git("status", "--porcelain=v1"):
        raise RuntimeError("C7 prepare requires a clean tracked and untracked tree")
    physical_gpus = parse_physical_gpus(
        args.physical_gpus,
        args.num_shards,
        "C7 requires one unique physical GPU per shard",
    )
    run_root, heavy_root = args.run_root.resolve(), args.heavy_root.resolve()
    if run_root == heavy_root or run_root in heavy_root.parents or heavy_root in run_root.parents:
        raise RuntimeError("C7 compact and heavy roots must not overlap")
    _validate_disk_budget(heavy_root, args.min_free_gib)
    source_sha, decision_sha = prepare_contracts(
        run_root=run_root,
        heavy_root=heavy_root,
        source_c6_dir=args.source_c6_dir,
        source_c3_heavy_root=args.source_c3_heavy_root,
        source_c4_heavy_root=args.source_c4_heavy_root,
        source_c6_heavy_root=args.source_c6_heavy_root,
        descriptor_checkpoint=args.descriptor_checkpoint,
        physical_gpus=physical_gpus,
    )
    print(
        json.dumps(
            {
                "source_contract_sha256": source_sha,
                "decision_contract_sha256": decision_sha,
                "descriptor_checkpoint_sha256": DESCRIPTOR_CHECKPOINT_SHA256,
                "n_cases": EXPECTED_CASE_COUNT,
            }
        )
    )
    return 0


def _worker_context(args: argparse.Namespace) -> tuple[dict[str, Any], str, torch.device]:
    decision_sha = str(args.decision_contract_sha256)
    decision = _load_decision(args.run_root, decision_sha)
    _assert_clean_runtime(decision, args.action)
    if args.num_shards != decision["num_shards"] or args.physical_gpu != decision["shard_to_physical_gpu"].get(
        str(args.shard_index)
    ):
        raise RuntimeError("C7 worker settings differ from the frozen contract")
    device = setup_device(args.gpu, seed=decision["seed"], deterministic=True)
    if device.type != "cuda":
        raise RuntimeError("C7 workers require CUDA")
    return decision, decision_sha, device


def decision_pilot_stage(args: argparse.Namespace) -> int:
    decision = _load_decision(args.run_root, args.decision_contract_sha256)
    pilot_case = decision["pilot_case_id"]
    shard_index = next(index for index in range(decision["num_shards"]) if pilot_case in decision["shards"][str(index)])
    args.shard_index = shard_index
    if args.physical_gpu != decision["shard_to_physical_gpu"][str(shard_index)]:
        raise RuntimeError("C7 pilot physical GPU differs from its frozen shard")
    decision, decision_sha, device = _worker_context(args)
    descriptor = _load_descriptor(decision, device)
    pilot_sha = _descriptor_pilot_checks(
        run_root=args.run_root,
        decision=decision,
        decision_sha=decision_sha,
        descriptor=descriptor,
        device=device,
    )
    marker = run_decision_case(
        case_id=pilot_case,
        shard_index=shard_index,
        physical_gpu=args.physical_gpu,
        run_root=args.run_root,
        decision=decision,
        decision_sha=decision_sha,
        device=device,
        descriptor=descriptor,
        execution=_execution(decision, "decision", args.attempt_id, shard_index, args.physical_gpu, device),
    )
    print(f"[C7 DECISION PILOT COMPLETE] case={pilot_case} descriptor_pilot_sha={pilot_sha} marker={marker}")
    return 0


def decision_worker_stage(args: argparse.Namespace) -> int:
    decision, decision_sha, device = _worker_context(args)
    marker = run_decision_worker(
        shard_index=args.shard_index,
        physical_gpu=args.physical_gpu,
        attempt_id=args.attempt_id,
        run_root=args.run_root,
        decision=decision,
        decision_sha=decision_sha,
        device=device,
        execution=_execution(
            decision,
            "decision",
            args.attempt_id,
            args.shard_index,
            args.physical_gpu,
            device,
        ),
    )
    print(f"[C7 DECISION WORKER COMPLETE] {marker}")
    return 0


def barrier_stage(args: argparse.Namespace) -> int:
    decision = _load_decision(args.run_root, args.decision_contract_sha256)
    _assert_clean_runtime(decision, "decision barrier")
    _load_descriptor_pilot(args.run_root, args.decision_contract_sha256)
    digest = build_decision_barrier(args.run_root, decision, args.decision_contract_sha256, args.attempt_id)
    print(f"[C7 DECISION BARRIER] {digest}")
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
    print(f"[C7 EVALUATION CONTRACT] {digest}")
    return 0


def evaluation_worker_stage(args: argparse.Namespace) -> int:
    decision, decision_sha, device = _worker_context(args)
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
    evaluation_labels = tuple(metric_profile_for("IXI").labels)
    if evaluation_labels != EVALUATION_LABEL_IDS or evaluation.get("evaluation_label_ids") != list(
        EVALUATION_LABEL_IDS
    ):
        raise RuntimeError("C7 runtime IXI label inventory differs from the frozen protocol")
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
        execution=_execution(
            decision,
            "evaluation",
            args.attempt_id,
            args.shard_index,
            args.physical_gpu,
            device,
        ),
        dataset_item_for_case=lambda case_id: dataset[index_by_case[case_id]],
        labels=evaluation_labels,
    )
    print(f"[C7 EVALUATION WORKER COMPLETE] {marker}")
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
    parser = argparse.ArgumentParser(description="Run the frozen C7 CorrMLP-x1 descriptor causal gate.")
    sub = parser.add_subparsers(dest="action", required=True)
    selfcheck = sub.add_parser("selfcheck")
    selfcheck.add_argument("--output", type=Path, required=True)
    prepare = sub.add_parser("prepare")
    prepare.add_argument("--run-root", type=Path, required=True)
    prepare.add_argument("--heavy-root", type=Path, required=True)
    prepare.add_argument("--source-c6-dir", type=Path, required=True)
    prepare.add_argument("--source-c3-heavy-root", type=Path, required=True)
    prepare.add_argument("--source-c4-heavy-root", type=Path, required=True)
    prepare.add_argument("--source-c6-heavy-root", type=Path, required=True)
    prepare.add_argument("--descriptor-checkpoint", type=Path, required=True)
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
    _configure_strict_fp32_backend()
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
