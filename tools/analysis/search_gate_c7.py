from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from types import MappingProxyType
from typing import Any

from tools.analysis.search_gate_c3 import (
    NCC7_WINDOW,
    NCC_DENOMINATOR_EPS,
    PRIMARY_NCC_IMPROVEMENT_MIN,
    SUPPORT_RETENTION_MIN,
)
from tools.analysis.search_gate_c6 import PairedSummary, simultaneous_paired_summaries
from tools.analysis.search_gate_learned import (
    CORRMLP_FULL_STATE_KEY_COUNT,
    CORRMLP_IXI_LAST_CHECKPOINT_SHA256,
    CORRMLP_IXI_LAST_EPOCH,
    CORRMLP_X1_CHANNELS,
    CORRMLP_X1_CONV_PADDING_MARGIN,
    DEFAULT_MOMENT_REDUCTION,
)

PROTOCOL_ID = "CTCF-SEARCH-GATE-C7-V1"
SCHEMA_VERSION = "v1"
EXPECTED_CASE_COUNT = 58
DEVELOPMENT_DATASET_ID = "IXI_VALIDATION_58"
TEST_115_AUTHORIZED = False

SOURCE_C6_PROTOCOL_ID = "CTCF-SEARCH-GATE-C6-V2"
SOURCE_C6_POLICY_SHA256 = "d0337e60ac0c7d271ae506acd009b439091bc723afa142c81b250529e6c4807b"
SOURCE_C4_REFERENCE_ARM_ID = "c4_intensity_s2"
SOURCE_C6_CONTEXT_ARM_ID = "full421_a150"

DESCRIPTOR_MODEL_ID = "P11_CORRMLP_IXI"
DESCRIPTOR_CHECKPOINT_SHA256 = "9cafbf426bd8a86cf9bc7e2981fcf7399101af6177292c3e726fc5b56eefa170"
DESCRIPTOR_CHECKPOINT_DEFAULT = "results/P11_CORRMLP_IXI/ckpt/last.pth"
DESCRIPTOR_CHECKPOINT_BYTES = 50_703_205
DESCRIPTOR_CHECKPOINT_EPOCH = 99
DESCRIPTOR_STATE_KEY_COUNT = 386
FORBIDDEN_VALIDATION_SELECTED_CHECKPOINT_SHA256 = "cf00eac7fd62051c13dec006e7a1aa9c31c6a1777877c71e9a7f2b4d0ada27be"
DESCRIPTOR_ID = "CORRMLP_SHARED_ENCODER_X1_FP32"
DESCRIPTOR_CHANNELS = 8
DESCRIPTOR_CONV_PADDING_MARGIN = 2
DESCRIPTOR_EXTRACTION = "same_frozen_encoder_convblock1_called_separately_for_fixed_and_moving"
DESCRIPTOR_PREPROCESSING = "raw_dataset_volume_float32_no_ctcf_masked_zscore"
DESCRIPTOR_DTYPE = "float32_no_autocast_no_tf32"
ZERO_FIELD_LOCAL_COST_PARITY_ATOL = 1e-4
ZERO_FIELD_LOCAL_COST_PARITY_RTOL = 0.0
FLOAT32_BACKEND_CONTRACT = {
    "cuda_matmul_allow_tf32": False,
    "cudnn_allow_tf32": False,
    "float32_matmul_precision": "highest",
}
DESCRIPTOR_COST = "negative_mean_channel_product_under_ctcf_sampling"
HYBRID_FUSION = (
    "intersect_component_informative_support_then_equal_mean_of_separately_center_standardized_costs_"
    "then_center_standardize_and_exclude_final_floor"
)
COST_STANDARDIZATION = "centered_two_pass_fp32"

FACTORS = (4, 2, 1)
FINAL_AMPLITUDE = 1.5
WORK_EPS = 0.0011
EXACT_CLAIM_EPS = 0.001
STAGE_WORK_EPS_DECREMENT = 0.000025
COMMON_EVIDENCE_COLLAR = 7
STANDARDIZATION_FLOOR = 1e-6
IMAGE_NORMALIZATION_STD_FLOOR = 1e-6
PROPOSAL_MULTIPLIER = 1.0
POST_SMOOTHING_PASSES = 1
POSTERIOR_TEMPERATURE = 1.0
CENTRE_BETA = 0.0
REQUIRE_ALL_CANDIDATES_VALID = True
STAGE_LOCAL_CLIP_SWEEPS = 1
FINAL_LOCAL_CLIP_SWEEPS = 1
RMS_TARGET_SOURCE_ID = "source_c3_raw_conf_post1_requested"
STAGE_BUDGET = "equal_share_of_source_rms_then_final_net_rematch"

REFERENCE_ARM_ID = SOURCE_C4_REFERENCE_ARM_ID
SOURCE_CONTEXT_ARM_ID = SOURCE_C6_CONTEXT_ARM_ID
MATCHED_CONTROL_ARM_ID = "intensity_margin2_full421_a150"
SELECTABLE_ARM_IDS = (
    "corrmlp_last_x1_full421_a150",
    "corrmlp_last_x1_intensity_eq_full421_a150",
)
EVALUATION_LABEL_IDS = (
    1,
    2,
    3,
    5,
    6,
    7,
    8,
    9,
    10,
    11,
    12,
    13,
    14,
    15,
    16,
    18,
    20,
    21,
    22,
    23,
    25,
    26,
    27,
    28,
    29,
    30,
    31,
    32,
    34,
    36,
)
RISK_LABEL_IDS = (9, 29, 13)

CAPACITY_MEAN_VS_C4_MIN = 0.002
CAPACITY_CI_LOW_VS_C4_MIN_STRICT = 0.001
CAUSAL_MEAN_VS_INTENSITY_MIN = 0.0005
CAUSAL_CI_LOW_VS_INTENSITY_MIN_STRICT = 0.0
SAFE_SUBSTITUTION_CI_LOW_VS_INTENSITY_MIN_STRICT = -0.001
RETURNED_MEAN_VS_C4_MIN = 0.001
RETURNED_CI_LOW_VS_C4_MIN_STRICT = 0.0
SDLOGJ_CI_HIGH_VS_C4_MAX = 0.005
ALL_LABEL_CI_LOW_VS_C4_MIN_STRICT = -0.005
RISK_LABEL_CI_LOW_VS_C4_MIN_STRICT = -0.002
CLIP_RETENTION_MEDIAN_MIN = 0.95
CLIP_RETENTION_PER_CASE_MIN = 0.90
CLIP_RETENTION_GE_090_MIN_COUNT = 52

BRANCH_FREEZE = "FREEZE_C7_DESCRIPTOR_COST_BUNDLE_FOR_TRANSFER_MATRIX"
BRANCH_SELECTOR = "OPEN_ONE_BOUNDED_C7B_SELECTOR_ONLY"
BRANCH_MATCHED_CONTROL = "CLOSE_C7_DESCRIPTOR_COST_BUNDLE_CAUSAL_CLAIM_MATCHED_INTENSITY_EXPLAINS_CAPACITY"
BRANCH_SAFETY = "CLOSE_C7_DESCRIPTOR_COST_BUNDLE_FOR_SAFETY_RISK"
BRANCH_CLOSE = "CLOSE_CORRMLP_X1_NATIVE_PRODUCT_BUNDLE_KEEP_SEARCH_AS_SAFE_DIAGNOSTIC"
BRANCH_INVALID = "INVALID_C7_EVIDENCE"


@dataclass(frozen=True, slots=True)
class ArmSpec:
    arm_index: int
    arm_id: str
    role: str
    descriptor: str
    factors: tuple[int, ...]
    amplitude: float
    rewarp_between_stages: bool
    selectable: bool
    source_arm_id: str | None = None


ARM_SPECS = (
    ArmSpec(0, REFERENCE_ARM_ID, "FROZEN_REFERENCE", "intensity", (), 1.0, True, False, "intensity_s2"),
    ArmSpec(
        1,
        SOURCE_CONTEXT_ARM_ID,
        "FROZEN_C6_CONTEXT",
        "intensity",
        FACTORS,
        FINAL_AMPLITUDE,
        True,
        False,
        SOURCE_CONTEXT_ARM_ID,
    ),
    ArmSpec(
        2,
        MATCHED_CONTROL_ARM_ID,
        "MATCHED_INTENSITY_CONTROL",
        "intensity_common_support",
        FACTORS,
        FINAL_AMPLITUDE,
        True,
        False,
    ),
    ArmSpec(3, SELECTABLE_ARM_IDS[0], "SELECTABLE", "corrmlp_x1", FACTORS, FINAL_AMPLITUDE, True, True),
    ArmSpec(
        4,
        SELECTABLE_ARM_IDS[1],
        "SELECTABLE",
        "corrmlp_x1_plus_intensity",
        FACTORS,
        FINAL_AMPLITUDE,
        True,
        True,
    ),
)
ARM_SPECS_BY_ID: Mapping[str, ArmSpec] = MappingProxyType({row.arm_id: row for row in ARM_SPECS})


@dataclass(frozen=True, slots=True)
class ArmAssessment:
    arm_id: str
    descriptor_valid: bool
    capacity_material: bool
    attribution_passed: bool
    attribution_mode: str | None
    returned_material: bool
    geometry_passed: bool
    retention_passed: bool
    candidate_regional_passed: bool
    returned_regional_passed: bool
    promotion_eligible: bool


def assess_arm(
    arm_id: str,
    *,
    descriptor_valid: bool,
    capacity_vs_c4: PairedSummary,
    causal_vs_intensity: PairedSummary,
    returned_vs_c4: PairedSummary,
    sdlogj_vs_c4: PairedSummary,
    candidate_regional_vs_initial: Sequence[tuple[int, PairedSummary]],
    candidate_regional_vs_c4: Sequence[tuple[int, PairedSummary]],
    returned_regional_vs_initial: Sequence[tuple[int, PairedSummary]],
    returned_regional_vs_c4: Sequence[tuple[int, PairedSummary]],
    folds_all_zero: bool,
    clip_retention_median: float,
    clip_retention_ge_090_count: int,
    matched_control_safety_passed: bool,
) -> ArmAssessment:
    if arm_id not in SELECTABLE_ARM_IDS:
        raise ValueError(f"not a selectable C7 arm: {arm_id}")
    capacity = (
        capacity_vs_c4.mean >= CAPACITY_MEAN_VS_C4_MIN and capacity_vs_c4.ci_low > CAPACITY_CI_LOW_VS_C4_MIN_STRICT
    )
    capacity_attribution = (
        causal_vs_intensity.mean >= CAUSAL_MEAN_VS_INTENSITY_MIN
        and causal_vs_intensity.ci_low > CAUSAL_CI_LOW_VS_INTENSITY_MIN_STRICT
    )
    safe_substitution = (
        causal_vs_intensity.ci_low > SAFE_SUBSTITUTION_CI_LOW_VS_INTENSITY_MIN_STRICT
        and not matched_control_safety_passed
    )
    returned = (
        returned_vs_c4.mean >= RETURNED_MEAN_VS_C4_MIN and returned_vs_c4.ci_low > RETURNED_CI_LOW_VS_C4_MIN_STRICT
    )
    geometry = folds_all_zero and sdlogj_vs_c4.ci_high <= SDLOGJ_CI_HIGH_VS_C4_MAX
    retention = (
        clip_retention_median >= CLIP_RETENTION_MEDIAN_MIN
        and clip_retention_ge_090_count >= CLIP_RETENTION_GE_090_MIN_COUNT
    )

    def regional_passed(rows: Sequence[tuple[int, PairedSummary]]) -> bool:
        labels = tuple(label for label, _ in rows)
        return labels == EVALUATION_LABEL_IDS and all(
            summary.ci_low
            > (RISK_LABEL_CI_LOW_VS_C4_MIN_STRICT if label in RISK_LABEL_IDS else ALL_LABEL_CI_LOW_VS_C4_MIN_STRICT)
            for label, summary in rows
        )

    candidate_regional = regional_passed(candidate_regional_vs_initial) and regional_passed(candidate_regional_vs_c4)
    returned_regional = regional_passed(returned_regional_vs_initial) and regional_passed(returned_regional_vs_c4)
    attribution = capacity_attribution or safe_substitution
    attribution_mode = "CAPACITY_GAIN" if capacity_attribution else "SAFE_SUBSTITUTION" if safe_substitution else None
    eligible = (
        descriptor_valid
        and capacity
        and attribution
        and returned
        and geometry
        and retention
        and candidate_regional
        and returned_regional
    )
    return ArmAssessment(
        arm_id=arm_id,
        descriptor_valid=descriptor_valid,
        capacity_material=capacity,
        attribution_passed=attribution,
        attribution_mode=attribution_mode,
        returned_material=returned,
        geometry_passed=geometry,
        retention_passed=retention,
        candidate_regional_passed=candidate_regional,
        returned_regional_passed=returned_regional,
        promotion_eligible=eligible,
    )


def select_branch(
    assessments: Sequence[ArmAssessment],
    returned_means: Mapping[str, float],
    sdlogj_means: Mapping[str, float],
    *,
    integrity_passed: bool = True,
) -> dict[str, Any]:
    if (
        not integrity_passed
        or len(assessments) != len(SELECTABLE_ARM_IDS)
        or {row.arm_id for row in assessments} != set(SELECTABLE_ARM_IDS)
        or set(returned_means) != set(SELECTABLE_ARM_IDS)
        or set(sdlogj_means) != set(SELECTABLE_ARM_IDS)
    ):
        return {"branch": BRANCH_INVALID, "winner": None, "reason": "integrity or arm inventory failed"}
    if not all(row.descriptor_valid for row in assessments):
        return {"branch": BRANCH_INVALID, "winner": None, "reason": "descriptor authentication failed"}
    eligible = [row for row in assessments if row.promotion_eligible]
    if eligible:
        winner = max(
            eligible,
            key=lambda row: (
                returned_means[row.arm_id],
                -sdlogj_means[row.arm_id],
                -ARM_SPECS_BY_ID[row.arm_id].arm_index,
            ),
        )
        return {
            "branch": BRANCH_FREEZE,
            "winner": winner.arm_id,
            "reason": "all preregistered descriptor-cost-bundle, utility, geometry and regional conditions passed",
        }
    causal_capacity = [row for row in assessments if row.capacity_material and row.attribution_passed]
    selector_only = [
        row for row in causal_capacity if row.geometry_passed and row.retention_passed and row.candidate_regional_passed
    ]
    if selector_only and all(not row.returned_material or not row.returned_regional_passed for row in selector_only):
        return {
            "branch": BRANCH_SELECTOR,
            "winner": None,
            "reason": "material safe descriptor-cost-bundle capacity exists but the frozen NCC7 selector loses it",
        }
    if causal_capacity:
        return {
            "branch": BRANCH_SAFETY,
            "winner": None,
            "reason": "causal descriptor-cost-bundle capacity exists but geometry, retention or regional safety failed",
        }
    if any(row.capacity_material for row in assessments):
        return {
            "branch": BRANCH_MATCHED_CONTROL,
            "winner": None,
            "reason": "absolute capacity exists but not beyond the matched intensity construction",
        }
    return {
        "branch": BRANCH_CLOSE,
        "winner": None,
        "reason": (
            "the authenticated CorrMLP x1 plus native negative-product cost bundle has no material capacity "
            "in the frozen scaffold"
        ),
    }


def policy_dict() -> dict[str, Any]:
    return {
        "protocol_id": PROTOCOL_ID,
        "schema_version": SCHEMA_VERSION,
        "dataset": DEVELOPMENT_DATASET_ID,
        "evaluation_label_ids": list(EVALUATION_LABEL_IDS),
        "test_115_authorized": TEST_115_AUTHORIZED,
        "source": {
            "c6_protocol_id": SOURCE_C6_PROTOCOL_ID,
            "c6_policy_sha256": SOURCE_C6_POLICY_SHA256,
            "c4_reference_arm_id": SOURCE_C4_REFERENCE_ARM_ID,
            "c6_context_arm_id": SOURCE_C6_CONTEXT_ARM_ID,
        },
        "descriptor": {
            "model_id": DESCRIPTOR_MODEL_ID,
            "checkpoint_default": DESCRIPTOR_CHECKPOINT_DEFAULT,
            "checkpoint_sha256": DESCRIPTOR_CHECKPOINT_SHA256,
            "checkpoint_bytes": DESCRIPTOR_CHECKPOINT_BYTES,
            "checkpoint_epoch": DESCRIPTOR_CHECKPOINT_EPOCH,
            "state_key_count": DESCRIPTOR_STATE_KEY_COUNT,
            "checkpoint_selection": "fixed_epoch_99_endpoint_not_validation_dice_selected_best",
            "forbidden_validation_selected_checkpoint_sha256": FORBIDDEN_VALIDATION_SELECTED_CHECKPOINT_SHA256,
            "descriptor_id": DESCRIPTOR_ID,
            "channels": DESCRIPTOR_CHANNELS,
            "convolutional_padding_support_margin": DESCRIPTOR_CONV_PADDING_MARGIN,
            "support_contract": (
                "all arms use the same margin-2 validity rule; stage-1 support is common, later per-arm support "
                "is path-dependent through Psi; hybrid components share an exact per-stage intersection; "
                "InstanceNorm remains global"
            ),
            "extraction": DESCRIPTOR_EXTRACTION,
            "preprocessing": DESCRIPTOR_PREPROCESSING,
            "dtype": DESCRIPTOR_DTYPE,
            "float32_backend": dict(FLOAT32_BACKEND_CONTRACT),
            "cost": DESCRIPTOR_COST,
            "tested_unit": "corrmlp_x1_representation_plus_native_negative_product_cost_bundle",
            "sampling_semantics": (
                "ctcf_sample_at_psi; pilot bounds zero-field local-cost agreement with CorrMLP Correlation "
                "under the expected FP32 difference between normalized-coordinate trilinear sampling and "
                "native integer slicing; it does not claim native CorrMLP warp-path parity"
            ),
            "zero_field_local_cost_parity": {
                "absolute_tolerance": ZERO_FIELD_LOCAL_COST_PARITY_ATOL,
                "relative_tolerance": ZERO_FIELD_LOCAL_COST_PARITY_RTOL,
                "scope": "all_27_offsets_on_the_frozen_interior_support",
                "rationale": (
                    "CorrMLP Correlation uses integer tensor slices while C7 must use align_corners_false "
                    "grid_sample for subsequent noninteger Psi; normalized FP32 coordinates are not bit-exact "
                    "integer indices"
                ),
            },
            "cost_standardization": COST_STANDARDIZATION,
            "hybrid_fusion": HYBRID_FUSION,
            "weights_trainable": False,
            "initializer_weights_changed": False,
        },
        "arms": [asdict(row) for row in ARM_SPECS],
        "construction": {
            "factors": list(FACTORS),
            "rewarp_between_stages": True,
            "final_amplitude": FINAL_AMPLITUDE,
            "stage_budget": STAGE_BUDGET,
            "centre_beta": CENTRE_BETA,
            "posterior_temperature": POSTERIOR_TEMPERATURE,
            "proposal_multiplier": PROPOSAL_MULTIPLIER,
            "post_smoothing_passes": POST_SMOOTHING_PASSES,
            "standardization_floor": STANDARDIZATION_FLOOR,
            "image_normalization_std_floor": IMAGE_NORMALIZATION_STD_FLOOR,
            "matched_intensity_normalization": "per_arm_per_stage_dynamic_common_support_masked_zscore",
            "require_all_candidates_valid": REQUIRE_ALL_CANDIDATES_VALID,
            "work_eps": WORK_EPS,
            "stage_work_eps_decrement": STAGE_WORK_EPS_DECREMENT,
            "stage_work_eps": [round(WORK_EPS - index * STAGE_WORK_EPS_DECREMENT, 9) for index in range(3)],
            "stage_local_clip_sweeps": STAGE_LOCAL_CLIP_SWEEPS,
            "final_local_clip_sweeps": FINAL_LOCAL_CLIP_SWEEPS,
            "exact_claim_eps": EXACT_CLAIM_EPS,
            "collar": COMMON_EVIDENCE_COLLAR,
            "rms_target_source_id": RMS_TARGET_SOURCE_ID,
            "primary_utility_id": "COMMON_NCC7",
            "primary_ncc_window": NCC7_WINDOW,
            "ncc_denominator_eps": NCC_DENOMINATOR_EPS,
            "support_retention_min": SUPPORT_RETENTION_MIN,
            "primary_ncc_improvement_min": PRIMARY_NCC_IMPROVEMENT_MIN,
        },
        "thresholds": {
            "capacity_mean_vs_c4_min": CAPACITY_MEAN_VS_C4_MIN,
            "capacity_ci_low_vs_c4_min_strict": CAPACITY_CI_LOW_VS_C4_MIN_STRICT,
            "causal_mean_vs_intensity_min": CAUSAL_MEAN_VS_INTENSITY_MIN,
            "causal_ci_low_vs_intensity_min_strict": CAUSAL_CI_LOW_VS_INTENSITY_MIN_STRICT,
            "returned_mean_vs_c4_min": RETURNED_MEAN_VS_C4_MIN,
            "returned_ci_low_vs_c4_min_strict": RETURNED_CI_LOW_VS_C4_MIN_STRICT,
            "sdlogj_ci_high_vs_c4_max": SDLOGJ_CI_HIGH_VS_C4_MAX,
            "all_label_ci_low_vs_c4_min_strict": ALL_LABEL_CI_LOW_VS_C4_MIN_STRICT,
            "risk_label_ci_low_vs_c4_min_strict": RISK_LABEL_CI_LOW_VS_C4_MIN_STRICT,
            "risk_label_ids": list(RISK_LABEL_IDS),
            "safe_substitution_ci_low_vs_intensity_min_strict": SAFE_SUBSTITUTION_CI_LOW_VS_INTENSITY_MIN_STRICT,
            "clip_retention_median_min": CLIP_RETENTION_MEDIAN_MIN,
            "clip_retention_per_case_min": CLIP_RETENTION_PER_CASE_MIN,
            "clip_retention_ge_090_min_count": CLIP_RETENTION_GE_090_MIN_COUNT,
        },
        "statistics": {
            "resamples": 10_000,
            "seed": 0,
            "confidence": 0.95,
            "method": "paired_case_bootstrap_max_absolute_centered_mean_deviation",
        },
        "contrasts": {
            "capacity": "each_selectable_candidate_minus_c4_reference",
            "returned": "each_selectable_returned_minus_c4_reference",
            "descriptor_cost_bundle_substitution": (
                "pipeline_level_each_selectable_candidate_minus_matched_margin2_intensity_ssd; "
                "the validity rule is matched but later support is a downstream path-dependent effect"
            ),
            "hybrid_increment": "hybrid_candidate_minus_learned_candidate",
            "support_change_diagnostic": "matched_margin2_intensity_minus_historical_c6_full421",
            "initial": "each_computed_candidate_and_returned_minus_common_initial",
            "geometry": "matched_control_and_each_selectable_sdlogj_minus_c4_reference",
            "regional_families": (
                "candidate_vs_initial_and_c4_and_returned_vs_initial_and_c4; "
                "each family contains the matched control and both selectable arms"
            ),
        },
        "safety": {
            "clip_retention_per_case": "minimum_of_three_stage_retentions_and_final_retention",
            "matched_control_safety": (
                "exact_geometry_and_sdlogj_and_clip_retention_and_candidate_and_returned_regional_guards"
            ),
            "winner_tiebreak": "returned_dice_then_lower_returned_sdlogj_then_learned_only_simplicity",
        },
        "branches": {
            "winner": BRANCH_FREEZE,
            "selector_only": BRANCH_SELECTOR,
            "matched_control_explains": BRANCH_MATCHED_CONTROL,
            "safety_failure": BRANCH_SAFETY,
            "no_capacity": BRANCH_CLOSE,
            "invalid": BRANCH_INVALID,
        },
    }


def policy_sha256() -> str:
    payload = json.dumps(policy_dict(), sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


C7_POLICY_SHA256 = "daca934f080c0b5b46d428183914b6bbfc81d422322e74c4a65fb55e60cd95ea"


def assert_frozen_policy() -> None:
    if tuple(row.arm_id for row in ARM_SPECS if row.selectable) != SELECTABLE_ARM_IDS:
        raise RuntimeError("C7 selectable inventory disagrees with ARM_SPECS")
    if tuple(row.arm_index for row in ARM_SPECS) != tuple(range(len(ARM_SPECS))):
        raise RuntimeError("C7 arm indices are not contiguous")
    construction = policy_dict()["construction"]
    borrowed = {
        "primary_ncc_window": NCC7_WINDOW,
        "ncc_denominator_eps": NCC_DENOMINATOR_EPS,
        "support_retention_min": SUPPORT_RETENTION_MIN,
        "primary_ncc_improvement_min": PRIMARY_NCC_IMPROVEMENT_MIN,
    }
    if any(construction[key] != value for key, value in borrowed.items()):
        raise RuntimeError("C7 published NCC contract disagrees with its search_gate_c3 owner")
    descriptor = policy_dict()["descriptor"]
    learned_owner = {
        "checkpoint_sha256": CORRMLP_IXI_LAST_CHECKPOINT_SHA256,
        "checkpoint_epoch": CORRMLP_IXI_LAST_EPOCH,
        "state_key_count": CORRMLP_FULL_STATE_KEY_COUNT,
        "channels": CORRMLP_X1_CHANNELS,
        "convolutional_padding_support_margin": CORRMLP_X1_CONV_PADDING_MARGIN,
        "cost_standardization": DEFAULT_MOMENT_REDUCTION,
    }
    if any(descriptor[key] != value for key, value in learned_owner.items()):
        raise RuntimeError("C7 published descriptor contract disagrees with its learned-feature owner")
    observed = policy_sha256()
    if observed != C7_POLICY_SHA256:
        raise RuntimeError(f"C7 policy changed: {observed} != {C7_POLICY_SHA256}")


__all__ = [
    "ALL_LABEL_CI_LOW_VS_C4_MIN_STRICT",
    "ARM_SPECS",
    "BRANCH_CLOSE",
    "BRANCH_FREEZE",
    "BRANCH_INVALID",
    "BRANCH_MATCHED_CONTROL",
    "BRANCH_SAFETY",
    "BRANCH_SELECTOR",
    "C7_POLICY_SHA256",
    "CLIP_RETENTION_PER_CASE_MIN",
    "DESCRIPTOR_CHECKPOINT_SHA256",
    "DESCRIPTOR_CONV_PADDING_MARGIN",
    "EVALUATION_LABEL_IDS",
    "EXPECTED_CASE_COUNT",
    "FLOAT32_BACKEND_CONTRACT",
    "MATCHED_CONTROL_ARM_ID",
    "REFERENCE_ARM_ID",
    "RISK_LABEL_CI_LOW_VS_C4_MIN_STRICT",
    "RISK_LABEL_IDS",
    "SELECTABLE_ARM_IDS",
    "ZERO_FIELD_LOCAL_COST_PARITY_ATOL",
    "ZERO_FIELD_LOCAL_COST_PARITY_RTOL",
    "ArmAssessment",
    "ArmSpec",
    "assert_frozen_policy",
    "assess_arm",
    "policy_dict",
    "policy_sha256",
    "select_branch",
    "simultaneous_paired_summaries",
]
