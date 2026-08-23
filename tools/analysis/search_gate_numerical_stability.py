from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from types import MappingProxyType
from typing import Any

import torch

from tools.analysis.search_gate_cost_volume import (
    DecoderReductionMode,
    MomentReductionMode,
    PosteriorReductionMode,
    RawMindCostVolume,
    StandardizedCostVolume,
    decode_posterior,
    masked_vector_rms,
    match_postprocessed_rms,
    posterior_from_logits,
    postprocess_residual,
    standardize_candidate_costs,
)
from tools.analysis.search_gate_metrics import MATHEMATICAL_SDLOGJ_CROP2, METRIC_SPECS

PROTOCOL_ID = "CTCF-SEARCH-GATE-NUMSTAB-V1"
SCHEMA_VERSION = "v1"
SOURCE_C3_RUN_ID = "C3_DEVELOPMENT_20260822T212632Z_5d6c762a8a6f"
SOURCE_C3_MANIFEST_SHA256 = "d5c35ba4a27dab2d6d0dcd9f8017c39364aece31471286fe844a1e34b2337094"
SOURCE_C3_RUN_MANIFEST_SHA256 = "ee1958b6ec3f00eb3100538c6f46dbdc056869570ab6c147b661775bd96313a5"
SOURCE_C3_GIT_HEAD = "5d6c762a8a6f11607fe4312f30da774b3e05dec2"

STANDARDIZATION_FLOOR = 1e-6
POSTERIOR_TEMPERATURE = 1.0
RESIDUAL_SCALE = 2.0
POST_SMOOTHING_PASSES = 1
COLLAR_WIDTH = 4
WORK_EPS = 0.0011
EXACT_CLAIM_EPS = 0.001
LOCAL_CLIP_SWEEPS = 1
LEGACY_PARITY_ATOL = 2e-6
ORACLE_FAITHFUL_ATOL = 2e-6
CAPACITY_MEAN_DICE_DELTA_MIN = 0.002
CAPACITY_CI_LOW_DICE_DELTA_MIN = 0.001
POLICY_MEAN_DICE_DELTA_MIN = 0.001
GEOMETRY_NONINFERIOR_TOLERANCE = 1e-6
SUPPORT_RETENTION_MIN = 0.99
PRIMARY_NCC_IMPROVEMENT_MIN = 1e-6
NCC7_WINDOW = 7
BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 0
BOOTSTRAP_CONFIDENCE = 0.95
WINNER_MEAN_TIE_TOLERANCE = 1e-6

BRANCH_ADVANCE_C32 = "ADVANCE_C32_WITHOUT_C3B"
BRANCH_OPEN_C3B = "OPEN_C3B_GEOMETRY_OR_POLICY"
BRANCH_OPEN_FP64 = "OPEN_MIXED_OR_FP64_NORMALIZATION_STUDY"
BRANCH_CLOSE_SINGLE_SCALE = "CLOSE_SINGLE_SCALE_RADIUS1_MIND"

MOMENT_LEGACY: MomentReductionMode = "legacy_sequential_fp32"
MOMENT_VECTORIZED: MomentReductionMode = "vectorized_second_moment_fp32"
MOMENT_CENTERED_FP32: MomentReductionMode = "centered_two_pass_fp32"
MOMENT_CENTERED_FP64: MomentReductionMode = "centered_two_pass_fp64"
POSTERIOR_LEGACY: PosteriorReductionMode = "legacy_sequential"
POSTERIOR_VECTORIZED: PosteriorReductionMode = "vectorized_e54"
DECODER_LEGACY: DecoderReductionMode = "legacy_sequential"
DECODER_VECTORIZED: DecoderReductionMode = "einsum_e54"

SENTINEL_ALL_VECTORIZED_GAPS = MappingProxyType(
    {
        "subject_344": 0.473039687,
        "subject_136": 0.460612535,
        "subject_165": 0.500411987,
        "subject_475": 0.530803859,
        "subject_131": 0.438622355,
    }
)


@dataclass(frozen=True, slots=True)
class FactorialSpec:
    cell_id: str
    moment_reduction: MomentReductionMode
    posterior_reduction: PosteriorReductionMode
    decoder_reduction: DecoderReductionMode


FACTORIAL_SPECS = tuple(
    FactorialSpec(
        f"F{moment_bit}{posterior_bit}{decoder_bit}",
        MOMENT_VECTORIZED if moment_bit else MOMENT_LEGACY,
        POSTERIOR_VECTORIZED if posterior_bit else POSTERIOR_LEGACY,
        DECODER_VECTORIZED if decoder_bit else DECODER_LEGACY,
    )
    for moment_bit in (0, 1)
    for posterior_bit in (0, 1)
    for decoder_bit in (0, 1)
)
FACTORIAL_BY_ID = MappingProxyType({spec.cell_id: spec for spec in FACTORIAL_SPECS})


def _factorial_edges() -> tuple[tuple[str, str, str], ...]:
    edges: list[tuple[str, str, str]] = []
    for axis, bit_index in (("moments", 1), ("posterior", 2), ("decoder", 3)):
        for cell_id in sorted(FACTORIAL_BY_ID):
            if cell_id[bit_index] != "0":
                continue
            changed = list(cell_id)
            changed[bit_index] = "1"
            edges.append((axis, cell_id, "".join(changed)))
    return tuple(edges)


FACTORIAL_EDGES = _factorial_edges()


@dataclass(frozen=True, slots=True)
class ScientificArmSpec:
    arm_index: int
    arm_id: str
    role: str
    moment_reduction: MomentReductionMode
    decoder_semantics: str
    posterior_reduction: PosteriorReductionMode
    decoder_reduction: DecoderReductionMode
    comparator_arm_id: str
    rms_reference_arm_id: str | None
    selectable: bool


SCIENTIFIC_ARMS = (
    ScientificArmSpec(
        0,
        "centered_fp32_conf",
        "scientific_candidate",
        MOMENT_CENTERED_FP32,
        "confidence",
        POSTERIOR_LEGACY,
        DECODER_LEGACY,
        "legacy_conf",
        None,
        True,
    ),
    ScientificArmSpec(
        1,
        "centered_fp32_mean_common_rms",
        "scientific_candidate",
        MOMENT_CENTERED_FP32,
        "posterior_mean_common_rms",
        POSTERIOR_LEGACY,
        DECODER_LEGACY,
        "legacy_mean_common_rms",
        "legacy_conf",
        True,
    ),
    ScientificArmSpec(
        2,
        "centered_fp64cast_conf",
        "precision_oracle",
        MOMENT_CENTERED_FP64,
        "confidence",
        POSTERIOR_LEGACY,
        DECODER_LEGACY,
        "legacy_conf",
        None,
        False,
    ),
    ScientificArmSpec(
        3,
        "centered_fp64cast_mean_common_rms",
        "precision_oracle",
        MOMENT_CENTERED_FP64,
        "posterior_mean_common_rms",
        POSTERIOR_LEGACY,
        DECODER_LEGACY,
        "legacy_mean_common_rms",
        "legacy_conf",
        False,
    ),
)
SCIENTIFIC_ARMS_BY_ID = MappingProxyType({spec.arm_id: spec for spec in SCIENTIFIC_ARMS})


@dataclass(frozen=True, slots=True)
class NumericalStabilityPolicy:
    protocol_id: str
    schema_version: str
    source_c3_run_id: str
    source_c3_manifest_sha256: str
    source_c3_run_manifest_sha256: str
    source_c3_git_head: str
    factorial_specs: tuple[FactorialSpec, ...]
    scientific_arms: tuple[ScientificArmSpec, ...]
    fixed_parameters: tuple[tuple[str, Any], ...]
    decisions: tuple[tuple[str, Any], ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


NUMERICAL_STABILITY_POLICY = NumericalStabilityPolicy(
    protocol_id=PROTOCOL_ID,
    schema_version=SCHEMA_VERSION,
    source_c3_run_id=SOURCE_C3_RUN_ID,
    source_c3_manifest_sha256=SOURCE_C3_MANIFEST_SHA256,
    source_c3_run_manifest_sha256=SOURCE_C3_RUN_MANIFEST_SHA256,
    source_c3_git_head=SOURCE_C3_GIT_HEAD,
    factorial_specs=FACTORIAL_SPECS,
    scientific_arms=SCIENTIFIC_ARMS,
    fixed_parameters=(
        ("raw_cost_dtype", "float32"),
        ("raw_cost_volume_per_case", 1),
        ("candidate_count", 27),
        ("candidate_radius", 1),
        ("mind_radius", 1),
        ("mind_dilation", 2),
        ("standardization_floor", STANDARDIZATION_FLOOR),
        ("centered_variance_ddof", 0),
        ("posterior_temperature", POSTERIOR_TEMPERATURE),
        ("residual_scale", RESIDUAL_SCALE),
        ("post_smoothing_passes", POST_SMOOTHING_PASSES),
        ("collar_width", COLLAR_WIDTH),
        ("work_eps", WORK_EPS),
        ("exact_claim_eps", EXACT_CLAIM_EPS),
        ("local_clip_sweeps", LOCAL_CLIP_SWEEPS),
        ("message_passing", "none"),
        ("execution_seed", 0),
        ("paths_profile", 3),
        ("test_115_authorized", False),
        ("labels_available_to_decision_workers", False),
        ("fp64_scope", "centered moments and z only; one cast to float32 before legacy posterior"),
        (
            "mean_decoder_rms_reference",
            "authenticated source C3 raw_conf_post1 requested residual reproduced by direct build_proposal",
        ),
        ("rms_matching_stage", "requested residual before direction-dependent local clipping"),
        (
            "oracle_faithful_scope",
            "preclip residual, stored postclip candidate and action, plus aggregate material/geometry/policy status",
        ),
        ("failed_vectorized_source_git_head", "e54d6bf4c026"),
        ("failed_vectorized_sentinel_reference", "F111 versus direct historical build_proposal at scale 2"),
        ("failed_vectorized_sentinel_gaps", tuple(SENTINEL_ALL_VECTORIZED_GAPS.items())),
        ("factorial_edges", FACTORIAL_EDGES),
        ("factorial_baseline", "independent F000 reduction path"),
        ("metric_ids", tuple(METRIC_SPECS)),
        ("primary_geometry_metric_id", MATHEMATICAL_SDLOGJ_CROP2),
        ("common_support_utility", "NCC7"),
        ("common_support_window", NCC7_WINDOW),
    ),
    decisions=(
        ("legacy_parity_atol", LEGACY_PARITY_ATOL),
        ("oracle_faithful_atol", ORACLE_FAITHFUL_ATOL),
        ("material_capacity_mean_dice_delta_min", CAPACITY_MEAN_DICE_DELTA_MIN),
        ("material_capacity_median_dice_delta", ">0"),
        ("material_capacity_ci_low_dice_delta_min_strict", CAPACITY_CI_LOW_DICE_DELTA_MIN),
        ("primary_policy_mean_dice_delta_min", POLICY_MEAN_DICE_DELTA_MIN),
        ("primary_policy_median_dice_delta", ">0"),
        ("primary_policy_ci_low_dice_delta", ">0"),
        ("geometry_noninferior_tolerance", GEOMETRY_NONINFERIOR_TOLERANCE),
        ("support_retention_min", SUPPORT_RETENTION_MIN),
        ("primary_ncc_improvement_min", PRIMARY_NCC_IMPROVEMENT_MIN),
        ("bootstrap_resamples", BOOTSTRAP_RESAMPLES),
        ("bootstrap_seed", BOOTSTRAP_SEED),
        ("bootstrap_confidence", BOOTSTRAP_CONFIDENCE),
        ("capacity_vs_legacy_mean_delta", ">0"),
        ("capacity_vs_legacy_median_delta", ">0"),
        ("capacity_vs_legacy_ci_low_delta", ">0"),
        ("primary_vs_legacy_mean_delta", ">0"),
        ("primary_vs_legacy_median_delta", ">=0"),
        ("primary_vs_legacy_ci_low_delta", ">=0"),
        ("winner_mean_tie_tolerance", WINNER_MEAN_TIE_TOLERANCE),
        (
            "decision_branch_precedence",
            (
                BRANCH_ADVANCE_C32,
                BRANCH_OPEN_C3B,
                BRANCH_OPEN_FP64,
                BRANCH_CLOSE_SINGLE_SCALE,
            ),
        ),
        ("capacity_geometry_reference", "source C3 c1_raw_conf_post1 candidate"),
        ("primary_geometry_reference", "source C3 c1_raw_conf_post1 returned"),
        ("fp64_oracle_selectable", False),
        ("postclip_realized_rms_must_be_reported", True),
    ),
)


def select_next_branch(
    summaries: list[dict[str, Any]],
    *,
    oracle_faithful_all_cases: bool,
) -> dict[str, Any]:
    """Select the preregistered post-gate branch without consulting test-115."""

    if not isinstance(oracle_faithful_all_cases, bool) or len(summaries) != len(SCIENTIFIC_ARMS):
        raise ValueError("branch selection requires one valid summary per frozen arm")
    normalized: list[tuple[ScientificArmSpec, dict[str, Any], float]] = []
    for spec, row in zip(SCIENTIFIC_ARMS, summaries, strict=True):
        capacity = row.get("capacity_vs_baseline") or {}
        mean = capacity.get("mean")
        if (
            row.get("arm_id") != spec.arm_id
            or row.get("arm_index") != spec.arm_index
            or row.get("role") != spec.role
            or row.get("selectable") is not spec.selectable
            or any(
                not isinstance(row.get(key), bool)
                for key in (
                    "material_capacity",
                    "viable_primary_policy",
                    "capacity_geometry_noninferior",
                )
            )
            or isinstance(mean, bool)
            or not isinstance(mean, (int, float))
            or not math.isfinite(float(mean))
        ):
            raise ValueError(f"invalid frozen arm summary for branch selection: {spec.arm_id}")
        normalized.append((spec, row, float(mean)))

    def choose(rows: list[tuple[ScientificArmSpec, dict[str, Any], float]]) -> str | None:
        if not rows:
            return None
        best_mean = max(item[2] for item in rows)
        tied = [item for item in rows if best_mean - item[2] <= WINNER_MEAN_TIE_TOLERANCE]
        return min(tied, key=lambda item: item[0].arm_index)[0].arm_id

    c32_material = [item for item in normalized if item[0].selectable and item[1]["material_capacity"]]
    c32_advance = [
        item for item in c32_material if item[1]["capacity_geometry_noninferior"] and item[1]["viable_primary_policy"]
    ]
    fp64_material = [item for item in normalized if item[0].role == "precision_oracle" and item[1]["material_capacity"]]
    if oracle_faithful_all_cases and c32_advance:
        branch_id = BRANCH_ADVANCE_C32
        parent = choose(c32_advance)
        reason = "faithful C32 arm is materially causal, geometrically noninferior, and policy-viable"
    elif oracle_faithful_all_cases and c32_material:
        branch_id = BRANCH_OPEN_C3B
        parent = choose(c32_material)
        reason = "faithful C32 arm is materially causal but geometry or returned-policy viability is unresolved"
    elif c32_material or fp64_material:
        branch_id = BRANCH_OPEN_FP64
        parent = choose(fp64_material or c32_material)
        reason = "material signal is precision-sensitive or appears only in the FP64 oracle"
    else:
        branch_id = BRANCH_CLOSE_SINGLE_SCALE
        parent = None
        reason = "no centered-moment arm establishes a material causal capacity signal"
    return {
        "branch_id": branch_id,
        "parent_arm_id": parent,
        "reason": reason,
        "oracle_faithful_all_cases": oracle_faithful_all_cases,
        "c32_material_arm_ids": [item[0].arm_id for item in c32_material],
        "c32_advance_eligible_arm_ids": [item[0].arm_id for item in c32_advance],
        "fp64_oracle_material_arm_ids": [item[0].arm_id for item in fp64_material],
        "winner_mean_tie_tolerance": WINNER_MEAN_TIE_TOLERANCE,
    }


def assess_arm_eligibility(
    *,
    selectable: bool,
    oracle_faithful_all_cases: bool,
    all_candidate_exact: bool,
    all_returned_exact: bool,
    all_support_defined: bool,
    primary_geometry_noninferior: bool,
    capacity_vs_baseline: dict[str, Any],
    primary_vs_baseline: dict[str, Any],
    capacity_vs_legacy: dict[str, Any],
    primary_vs_legacy: dict[str, Any],
) -> dict[str, bool]:
    """Apply the frozen practical and causal thresholds to one arm summary."""

    flags = (
        selectable,
        oracle_faithful_all_cases,
        all_candidate_exact,
        all_returned_exact,
        all_support_defined,
        primary_geometry_noninferior,
    )
    if any(not isinstance(value, bool) for value in flags):
        raise TypeError("eligibility flags must be boolean")

    def values(summary: dict[str, Any], label: str) -> tuple[float, float, float]:
        observed = tuple(summary.get(key) for key in ("mean", "median", "ci_low"))
        if any(
            isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value))
            for value in observed
        ):
            raise ValueError(f"{label} must contain finite mean, median, and ci_low")
        return float(observed[0]), float(observed[1]), float(observed[2])

    capacity_mean, capacity_median, capacity_ci_low = values(capacity_vs_baseline, "capacity_vs_baseline")
    primary_mean, primary_median, primary_ci_low = values(primary_vs_baseline, "primary_vs_baseline")
    causal_capacity_mean, causal_capacity_median, causal_capacity_ci_low = values(
        capacity_vs_legacy, "capacity_vs_legacy"
    )
    causal_primary_mean, causal_primary_median, causal_primary_ci_low = values(primary_vs_legacy, "primary_vs_legacy")
    material_vs_baseline = bool(
        all_candidate_exact
        and capacity_mean >= CAPACITY_MEAN_DICE_DELTA_MIN
        and capacity_median > 0.0
        and capacity_ci_low > CAPACITY_CI_LOW_DICE_DELTA_MIN
    )
    capacity_superior_to_legacy = bool(
        causal_capacity_mean > 0.0 and causal_capacity_median > 0.0 and causal_capacity_ci_low > 0.0
    )
    practical_primary_vs_baseline = bool(
        all_returned_exact
        and all_support_defined
        and primary_geometry_noninferior
        and primary_mean >= POLICY_MEAN_DICE_DELTA_MIN
        and primary_median > 0.0
        and primary_ci_low > 0.0
    )
    primary_superior_to_legacy = bool(
        causal_primary_mean > 0.0 and causal_primary_median >= 0.0 and causal_primary_ci_low >= 0.0
    )
    return {
        "material_capacity_vs_baseline": material_vs_baseline,
        "capacity_superior_to_legacy": capacity_superior_to_legacy,
        "material_capacity": material_vs_baseline and capacity_superior_to_legacy,
        "practical_primary_policy_vs_baseline": practical_primary_vs_baseline,
        "primary_policy_superior_to_legacy": primary_superior_to_legacy,
        "viable_primary_policy": bool(
            selectable and oracle_faithful_all_cases and practical_primary_vs_baseline and primary_superior_to_legacy
        ),
    }


def policy_sha256(policy: NumericalStabilityPolicy = NUMERICAL_STABILITY_POLICY) -> str:
    encoded = json.dumps(policy.to_dict(), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


NUMERICAL_STABILITY_POLICY_SHA256 = "ddf625a3c0c07806ce83fdf454cacfc82705615e06160bdc4438f16fbc67e159"


def assert_frozen_policy() -> None:
    actual = policy_sha256()
    if actual != NUMERICAL_STABILITY_POLICY_SHA256:
        raise RuntimeError(
            f"numerical-stability policy hash mismatch: declared={NUMERICAL_STABILITY_POLICY_SHA256}, actual={actual}"
        )


@dataclass(frozen=True)
class ReductionStudy:
    normalizations: dict[MomentReductionMode, StandardizedCostVolume]
    factorial_decoded: dict[str, torch.Tensor]
    factorial_residuals: dict[str, torch.Tensor]
    historical_requested: dict[str, torch.Tensor]
    scientific_requested: dict[str, torch.Tensor]
    normalization_rows: list[dict[str, Any]]
    factorial_cell_rows: list[dict[str, Any]]
    factorial_edge_rows: list[dict[str, Any]]
    scientific_rows: list[dict[str, Any]]
    oracle_faithful: bool


def _require_mask(mask: torch.Tensor, spatial: tuple[int, int, int], device: torch.device) -> None:
    if mask.shape != (1, 1, *spatial) or mask.dtype != torch.bool or mask.device != device:
        raise ValueError("geometry mask must be boolean [1,1,D,H,W] on the reduction device")
    if not bool(mask.any()):
        raise ValueError("geometry mask must not be empty")


def _quantiles(values: torch.Tensor) -> tuple[float, float]:
    selected = values.double().reshape(-1)
    if selected.numel() == 0 or not bool(torch.isfinite(selected).all()):
        raise ValueError("diagnostic quantiles require finite non-empty values")
    result = torch.quantile(selected, selected.new_tensor((0.95, 0.99)))
    return float(result[0].item()), float(result[1].item())


def scalar_map_error(
    candidate: torch.Tensor,
    reference: torch.Tensor,
    mask: torch.Tensor,
    *,
    relative_floor: float,
) -> dict[str, float]:
    if candidate.shape != reference.shape or candidate.shape != mask.shape:
        raise ValueError("scalar diagnostic maps and mask must have identical shapes")
    candidate64 = candidate.double()
    reference64 = reference.double()
    difference = (candidate64 - reference64).abs().masked_select(mask)
    reference_scale = reference64.abs().clamp_min(float(relative_floor)).masked_select(mask)
    if not bool(torch.isfinite(difference).all()) or not bool(torch.isfinite(reference_scale).all()):
        raise ValueError("scalar diagnostic maps must be finite inside geometry")
    relative = difference / reference_scale
    p95, p99 = _quantiles(difference)
    relative_p95, relative_p99 = _quantiles(relative)
    return {
        "mean_abs": float(difference.mean().item()),
        "rms": float(difference.square().mean().sqrt().item()),
        "p95_abs": p95,
        "p99_abs": p99,
        "max_abs": float(difference.max().item()),
        "relative_p95": relative_p95,
        "relative_p99": relative_p99,
        "relative_max": float(relative.max().item()),
    }


def candidate_map_error(
    candidate: torch.Tensor,
    reference: torch.Tensor,
    valid: torch.Tensor,
    geometry_mask: torch.Tensor,
    *,
    relative_floor: float,
) -> dict[str, float]:
    if candidate.shape != reference.shape or candidate.shape != valid.shape:
        raise ValueError("candidate diagnostics must share shape")
    active = valid & geometry_mask.expand_as(valid)
    if not bool(active.any()):
        raise ValueError("candidate diagnostics have empty active support")
    difference = torch.where(active, candidate.double() - reference.double(), 0.0)
    count = active.sum(dim=1, keepdim=True).double().clamp_min(1.0)
    voxel_rms = (difference.square().sum(dim=1, keepdim=True) / count).sqrt()
    reference_safe = torch.where(active, reference.double(), 0.0)
    reference_rms = (reference_safe.square().sum(dim=1, keepdim=True) / count).sqrt()
    selected_rms = voxel_rms.masked_select(geometry_mask)
    relative = selected_rms / reference_rms.clamp_min(float(relative_floor)).masked_select(geometry_mask)
    absolute = difference.abs().masked_select(active)
    p95, p99 = _quantiles(selected_rms)
    relative_p95, relative_p99 = _quantiles(relative)
    return {
        "global_rms": float(difference.square().masked_select(active).mean().sqrt().item()),
        "voxel_rms_mean": float(selected_rms.mean().item()),
        "voxel_rms_p95": p95,
        "voxel_rms_p99": p99,
        "max_abs": float(absolute.max().item()),
        "relative_voxel_rms_p95": relative_p95,
        "relative_voxel_rms_p99": relative_p99,
        "relative_voxel_rms_max": float(relative.max().item()),
    }


def field_difference(
    candidate: torch.Tensor,
    reference: torch.Tensor,
    geometry_mask: torch.Tensor,
) -> dict[str, float | int | None]:
    if candidate.shape != reference.shape or candidate.ndim != 5 or candidate.shape[1] != 3:
        raise ValueError("field diagnostics require matching [1,3,D,H,W] tensors")
    _require_mask(geometry_mask, tuple(candidate.shape[-3:]), candidate.device)
    active = geometry_mask.expand_as(candidate)
    if not bool(torch.isfinite(candidate.masked_select(active)).all()) or not bool(
        torch.isfinite(reference.masked_select(active)).all()
    ):
        raise ValueError("field diagnostics require finite values inside geometry")
    candidate64 = candidate.double()
    reference64 = reference.double()
    difference = candidate64 - reference64
    vector_error = difference.square().sum(dim=1, keepdim=True).sqrt().masked_select(geometry_mask)
    candidate_norm = candidate64.square().sum(dim=1, keepdim=True).sqrt()
    reference_norm = reference64.square().sum(dim=1, keepdim=True).sqrt()
    tolerance = torch.finfo(torch.float64).tiny
    candidate_nonzero = candidate_norm > tolerance
    reference_nonzero = reference_norm > tolerance
    both_nonzero = geometry_mask & candidate_nonzero & reference_nonzero
    both_zero = geometry_mask & ~candidate_nonzero & ~reference_nonzero
    one_zero = geometry_mask & (candidate_nonzero ^ reference_nonzero)
    cosine_mean: float | None = None
    cosine_min: float | None = None
    if bool(both_nonzero.any()):
        dot = (candidate64 * reference64).sum(dim=1, keepdim=True)
        cosine = dot / (candidate_norm * reference_norm).clamp_min(tolerance)
        selected_cosine = cosine.masked_select(both_nonzero)
        cosine_mean = float(selected_cosine.mean().item())
        cosine_min = float(selected_cosine.min().item())
    p95, p99 = _quantiles(vector_error)
    return {
        "max_abs": float(difference.abs().masked_select(active).max().item()),
        "vector_rms": float(vector_error.square().mean().sqrt().item()),
        "vector_error_p95": p95,
        "vector_error_p99": p99,
        "cosine_mean_both_nonzero": cosine_mean,
        "cosine_min_both_nonzero": cosine_min,
        "both_nonzero_voxels": int(both_nonzero.sum(dtype=torch.int64).item()),
        "both_zero_voxels": int(both_zero.sum(dtype=torch.int64).item()),
        "one_zero_voxels": int(one_zero.sum(dtype=torch.int64).item()),
        "candidate_rms": masked_vector_rms(candidate, geometry_mask),
        "reference_rms": masked_vector_rms(reference, geometry_mask),
    }


def _prefix(prefix: str, values: dict[str, Any]) -> dict[str, Any]:
    return {f"{prefix}_{key}": value for key, value in values.items()}


def _normalization_row(
    result: StandardizedCostVolume,
    oracle: StandardizedCostVolume,
    geometry_mask: torch.Tensor,
) -> dict[str, Any]:
    active_voxels = int(geometry_mask.sum(dtype=torch.int64).item())
    negative_count = int((result.variance_negative & geometry_mask).sum(dtype=torch.int64).item())
    floor_count = int((result.floor_hit & geometry_mask).sum(dtype=torch.int64).item())
    valid = result.valid
    active_candidates = valid & geometry_mask.expand_as(valid)
    count = result.valid_count.double().clamp_min(1.0)
    safe_z = torch.where(valid, result.standardized_costs.double(), 0.0)
    z_mean = safe_z.sum(dim=1, keepdim=True) / count
    z_rms = (safe_z.square().sum(dim=1, keepdim=True) / count).sqrt()
    masked_costs = torch.where(valid, result.standardized_costs, torch.full_like(result.standardized_costs, torch.inf))
    oracle_costs = torch.where(valid, oracle.standardized_costs, torch.full_like(oracle.standardized_costs, torch.inf))
    argmin = masked_costs.argmin(dim=1, keepdim=True)
    oracle_argmin = oracle_costs.argmin(dim=1, keepdim=True)
    minimum = masked_costs.amin(dim=1, keepdim=True)
    oracle_minimum = oracle_costs.amin(dim=1, keepdim=True)
    tie_set = valid & (masked_costs == minimum)
    oracle_tie_set = valid & (oracle_costs == oracle_minimum)
    tie_agreement = (tie_set == oracle_tie_set).all(dim=1, keepdim=True)
    z_rms_values = z_rms.masked_select(geometry_mask)
    z_rms_p95, z_rms_p99 = _quantiles(z_rms_values)
    row: dict[str, Any] = {
        "moment_reduction": result.standardization_mode,
        "statistics_dtype": str(result.cost_mean.dtype),
        "active_voxel_count": active_voxels,
        "active_candidate_count": int(active_candidates.sum(dtype=torch.int64).item()),
        "negative_variance_count": negative_count,
        "negative_variance_fraction": negative_count / active_voxels,
        "floor_hit_count": floor_count,
        "floor_hit_fraction": floor_count / active_voxels,
        "standardized_mean_abs_max": float(z_mean.abs().masked_select(geometry_mask).max().item()),
        "standardized_rms_mean": float(z_rms_values.mean().item()),
        "standardized_rms_p95": z_rms_p95,
        "standardized_rms_p99": z_rms_p99,
        "argmin_agreement_fraction_vs_fp64": float(
            (argmin == oracle_argmin).masked_select(geometry_mask).double().mean().item()
        ),
        "minimum_tie_set_agreement_fraction_vs_fp64": float(
            tie_agreement.masked_select(geometry_mask).double().mean().item()
        ),
    }
    row.update(
        _prefix(
            "mean_error_vs_fp64",
            scalar_map_error(result.cost_mean, oracle.cost_mean, geometry_mask, relative_floor=1e-12),
        )
    )
    row.update(
        _prefix(
            "std_error_vs_fp64",
            scalar_map_error(
                result.cost_std,
                oracle.cost_std,
                geometry_mask,
                relative_floor=STANDARDIZATION_FLOOR,
            ),
        )
    )
    row.update(
        _prefix(
            "z_error_vs_fp64",
            candidate_map_error(
                result.standardized_costs,
                oracle.standardized_costs,
                valid,
                geometry_mask,
                relative_floor=STANDARDIZATION_FLOOR,
            ),
        )
    )
    return row


def _posterior_row(posterior: Any, mask: torch.Tensor) -> dict[str, float]:
    peak = posterior.probabilities.amax(dim=1, keepdim=True)
    return {
        "entropy_mean": float(posterior.entropy.masked_select(mask).double().mean().item()),
        "normalized_entropy_mean": float(posterior.normalized_entropy.masked_select(mask).double().mean().item()),
        "peak_probability_mean": float(peak.masked_select(mask).double().mean().item()),
        "confidence_mean": float(posterior.confidence.masked_select(mask).double().mean().item()),
    }


def build_reduction_study(
    raw: RawMindCostVolume,
    geometry_mask: torch.Tensor,
    *,
    legacy_confidence_reference: torch.Tensor,
) -> ReductionStudy:
    """Run the frozen factorial and moments-only scientific branches in memory."""

    costs, valid = raw.costs, raw.valid
    _require_mask(geometry_mask, tuple(costs.shape[-3:]), costs.device)
    if costs.dtype != torch.float32:
        raise TypeError(f"the frozen numerical-stability study requires float32 raw costs, got {costs.dtype}")
    if raw.valid_count.shape != geometry_mask.shape or not torch.equal(raw.valid_count, valid.sum(dim=1, keepdim=True)):
        raise ValueError("raw cost-volume validity count is inconsistent")
    if bool((geometry_mask & (raw.valid_count == 0)).any()):
        raise ValueError("raw cost-volume has an active geometry voxel without a valid candidate")
    if (
        legacy_confidence_reference.shape != (1, 3, *costs.shape[-3:])
        or legacy_confidence_reference.dtype != costs.dtype
        or legacy_confidence_reference.device != costs.device
        or not bool(
            torch.isfinite(
                legacy_confidence_reference.masked_select(geometry_mask.expand_as(legacy_confidence_reference))
            ).all()
        )
    ):
        raise ValueError("legacy confidence reference must be finite float32 [1,3,D,H,W] on the reduction device")
    modes: tuple[MomentReductionMode, ...] = (
        MOMENT_LEGACY,
        MOMENT_VECTORIZED,
        MOMENT_CENTERED_FP32,
        MOMENT_CENTERED_FP64,
    )
    normalizations = {
        mode: standardize_candidate_costs(
            costs,
            valid,
            mode=mode,
            standardization_floor=STANDARDIZATION_FLOOR,
        )
        for mode in modes
    }
    oracle = normalizations[MOMENT_CENTERED_FP64]
    normalization_rows = [_normalization_row(normalizations[mode], oracle, geometry_mask) for mode in modes]

    posterior_cache: dict[tuple[MomentReductionMode, PosteriorReductionMode], Any] = {}
    factorial_decoded: dict[str, torch.Tensor] = {}
    factorial_residuals: dict[str, torch.Tensor] = {}
    factorial_cell_rows: list[dict[str, Any]] = []
    for spec in FACTORIAL_SPECS:
        key = (spec.moment_reduction, spec.posterior_reduction)
        if key not in posterior_cache:
            volume = normalizations[spec.moment_reduction]
            posterior_cache[key] = posterior_from_logits(
                -volume.standardized_costs,
                volume.valid,
                temperature=POSTERIOR_TEMPERATURE,
                reduction_mode=spec.posterior_reduction,
            )
        posterior = posterior_cache[key]
        decoded = decode_posterior(
            posterior,
            mode="confidence",
            reduction_mode=spec.decoder_reduction,
        )
        factorial_decoded[spec.cell_id] = decoded.displacement
        residual = postprocess_residual(
            decoded.displacement,
            scale=RESIDUAL_SCALE,
            post_smoothing_passes=POST_SMOOTHING_PASSES,
            collar_width=COLLAR_WIDTH,
        )
        factorial_residuals[spec.cell_id] = residual
        factorial_cell_rows.append(
            {
                **asdict(spec),
                **_posterior_row(posterior, geometry_mask),
                "raw_decoded_rms": masked_vector_rms(decoded.displacement, geometry_mask),
                "postprocessed_residual_rms": masked_vector_rms(residual, geometry_mask),
            }
        )

    legacy_residual = factorial_residuals["F000"]
    legacy_posterior = posterior_cache[(MOMENT_LEGACY, POSTERIOR_LEGACY)]
    legacy_mean = decode_posterior(
        legacy_posterior,
        mode="posterior_mean",
        reduction_mode=DECODER_LEGACY,
    )
    legacy_mean_post = postprocess_residual(
        legacy_mean.displacement,
        scale=RESIDUAL_SCALE,
        post_smoothing_passes=POST_SMOOTHING_PASSES,
        collar_width=COLLAR_WIDTH,
    )
    historical_requested = {
        "legacy_conf": legacy_confidence_reference,
        "legacy_mean_common_rms": match_postprocessed_rms(
            legacy_mean_post,
            legacy_confidence_reference,
            geometry_mask,
        ).displacement,
    }
    for row in factorial_cell_rows:
        row.update(
            _prefix(
                "difference_vs_f000",
                field_difference(factorial_residuals[row["cell_id"]], legacy_residual, geometry_mask),
            )
        )
    factorial_edge_rows: list[dict[str, Any]] = []
    for axis, source_id, target_id in FACTORIAL_EDGES:
        source_spec = FACTORIAL_BY_ID[source_id]
        target_spec = FACTORIAL_BY_ID[target_id]
        source_posterior = posterior_cache[(source_spec.moment_reduction, source_spec.posterior_reduction)]
        target_posterior = posterior_cache[(target_spec.moment_reduction, target_spec.posterior_reduction)]
        row: dict[str, Any] = {
            "axis": axis,
            "source_cell_id": source_id,
            "target_cell_id": target_id,
            **_prefix(
                "raw_decoder",
                field_difference(factorial_decoded[target_id], factorial_decoded[source_id], geometry_mask),
            ),
            **_prefix(
                "residual",
                field_difference(factorial_residuals[target_id], factorial_residuals[source_id], geometry_mask),
            ),
            **_prefix(
                "probability",
                candidate_map_error(
                    target_posterior.probabilities,
                    source_posterior.probabilities,
                    valid,
                    geometry_mask,
                    relative_floor=1e-12,
                ),
            ),
            **_prefix(
                "entropy",
                scalar_map_error(
                    target_posterior.entropy, source_posterior.entropy, geometry_mask, relative_floor=1e-12
                ),
            ),
            **_prefix(
                "confidence",
                scalar_map_error(
                    target_posterior.confidence,
                    source_posterior.confidence,
                    geometry_mask,
                    relative_floor=1e-12,
                ),
            ),
        }
        factorial_edge_rows.append(row)

    scientific_requested: dict[str, torch.Tensor] = {}
    scientific_rows: list[dict[str, Any]] = []
    for spec in SCIENTIFIC_ARMS:
        posterior_key = (spec.moment_reduction, spec.posterior_reduction)
        if posterior_key not in posterior_cache:
            volume = normalizations[spec.moment_reduction]
            posterior_cache[posterior_key] = posterior_from_logits(
                -volume.standardized_costs,
                volume.valid,
                temperature=POSTERIOR_TEMPERATURE,
                reduction_mode=spec.posterior_reduction,
            )
        posterior = posterior_cache[posterior_key]
        if spec.decoder_semantics == "confidence":
            decoder_mode = "confidence"
        elif spec.decoder_semantics == "posterior_mean_common_rms":
            decoder_mode = "posterior_mean"
        else:
            raise RuntimeError(f"unsupported frozen decoder semantics: {spec.decoder_semantics!r}")
        decoded = decode_posterior(
            posterior,
            mode=decoder_mode,
            reduction_mode=spec.decoder_reduction,
        )
        postprocessed = postprocess_residual(
            decoded.displacement,
            scale=RESIDUAL_SCALE,
            post_smoothing_passes=POST_SMOOTHING_PASSES,
            collar_width=COLLAR_WIDTH,
        )
        rms_scale_factor: float | None = None
        rms_source_before_matching: float | None = None
        rms_reference: float | None = None
        if spec.rms_reference_arm_id is not None:
            reference = historical_requested.get(spec.rms_reference_arm_id)
            if reference is None:
                raise RuntimeError(f"unknown frozen RMS reference arm: {spec.rms_reference_arm_id!r}")
            matched = match_postprocessed_rms(postprocessed, reference, geometry_mask)
            requested = matched.displacement
            rms_scale_factor = matched.scale_factor
            rms_source_before_matching = matched.source_rms
            rms_reference = matched.target_rms
        else:
            requested = postprocessed
        scientific_requested[spec.arm_id] = requested
        scientific_rows.append(
            {
                **asdict(spec),
                **_posterior_row(posterior, geometry_mask),
                "requested_rms": masked_vector_rms(requested, geometry_mask),
                "rms_source_before_matching": rms_source_before_matching,
                "rms_reference_requested_rms": rms_reference,
                "rms_scale_factor": rms_scale_factor,
            }
        )

    if tuple(scientific_requested) != tuple(spec.arm_id for spec in SCIENTIFIC_ARMS):
        raise RuntimeError("scientific execution did not materialize the frozen arm sequence")

    conf_difference = field_difference(
        scientific_requested["centered_fp32_conf"],
        scientific_requested["centered_fp64cast_conf"],
        geometry_mask,
    )
    mean_difference = field_difference(
        scientific_requested["centered_fp32_mean_common_rms"],
        scientific_requested["centered_fp64cast_mean_common_rms"],
        geometry_mask,
    )
    oracle_faithful = bool(
        float(conf_difference["max_abs"]) <= ORACLE_FAITHFUL_ATOL
        and float(mean_difference["max_abs"]) <= ORACLE_FAITHFUL_ATOL
    )
    for row in scientific_rows:
        arm_id = row["arm_id"]
        oracle_id = arm_id.replace("centered_fp32", "centered_fp64cast")
        row.update(
            _prefix(
                "difference_vs_fp64",
                field_difference(
                    scientific_requested[arm_id],
                    scientific_requested[oracle_id],
                    geometry_mask,
                ),
            )
        )
        row["centered_fp32_oracle_faithful_case"] = oracle_faithful

    return ReductionStudy(
        normalizations=normalizations,
        factorial_decoded=factorial_decoded,
        factorial_residuals=factorial_residuals,
        historical_requested=historical_requested,
        scientific_requested=scientific_requested,
        normalization_rows=normalization_rows,
        factorial_cell_rows=factorial_cell_rows,
        factorial_edge_rows=factorial_edge_rows,
        scientific_rows=scientific_rows,
        oracle_faithful=oracle_faithful,
    )


def selfcheck() -> dict[str, Any]:
    cell_ids = [spec.cell_id for spec in FACTORIAL_SPECS]
    checks = {
        "eight_unique_factorial_cells": len(cell_ids) == 8 and len(set(cell_ids)) == 8,
        "all_factorial_edges_present": len(FACTORIAL_EDGES) == 12 and len(set(FACTORIAL_EDGES)) == 12,
        "four_unique_scientific_arms": len(SCIENTIFIC_ARMS_BY_ID) == 4,
        "scientific_arms_freeze_legacy_downstream": all(
            spec.moment_reduction in (MOMENT_CENTERED_FP32, MOMENT_CENTERED_FP64)
            and spec.posterior_reduction == POSTERIOR_LEGACY
            and spec.decoder_reduction == DECODER_LEGACY
            for spec in SCIENTIFIC_ARMS
        ),
        "test_115_is_closed": dict(NUMERICAL_STABILITY_POLICY.fixed_parameters)["test_115_authorized"] is False,
        "policy_hash_is_frozen": policy_sha256() == NUMERICAL_STABILITY_POLICY_SHA256,
    }
    failed = [name for name, passed in checks.items() if not passed]
    return {
        "schema": f"ctcf-search-gate-numstab-selfcheck-{SCHEMA_VERSION}",
        "protocol_id": PROTOCOL_ID,
        "status": "PASS" if not failed else "FAIL",
        "checks": checks,
        "failed": failed,
        "policy_sha256": policy_sha256(),
    }
