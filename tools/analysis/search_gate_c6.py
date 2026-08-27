from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from types import MappingProxyType
from typing import Any

import numpy as np

from tools.analysis.search_gate_c3 import (
    NCC7_WINDOW,
    NCC_DENOMINATOR_EPS,
    PRIMARY_NCC_IMPROVEMENT_MIN,
    SUPPORT_RETENTION_MIN,
)

PROTOCOL_ID = "CTCF-SEARCH-GATE-C6-V1"
SCHEMA_VERSION = "v1"
EXPECTED_CASE_COUNT = 58
DEVELOPMENT_DATASET_ID = "IXI_VALIDATION_58"
TEST_115_AUTHORIZED = False

WORK_EPS = 0.0011
EXACT_CLAIM_EPS = 0.001
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
PYRAMID_FILTER = "recursive_half_scale_separable_binomial5_[1,4,6,4,1]/16_replicate"
PYRAMID_INTERPOLATION = "trilinear_align_corners_false"
PYRAMID_DISPLACEMENT_SCALE = "divide_on_projection_multiply_on_lift"
STAGE_BUDGET = "equal_share_of_source_rms_then_final_net_rematch"
BLURRED_CONTROL_SCOPE = (
    "the blurred control applies the explicit binomial anti-alias filter without decimation; the pyramid's own "
    "align_corners=False image decimation adds further image averaging, and its displacement lift is a separate "
    "interpolation of the recovered coarse vector rather than part of the image blur, so a pyramid-versus-blur "
    "contrast measures the whole scale-change pipeline and not decimation at an identical effective image blur"
)

FROZEN_CONSTRUCTION: Mapping[str, Any] = MappingProxyType(
    {
        "full_collar": COMMON_EVIDENCE_COLLAR,
        "work_eps": WORK_EPS,
        "standardization_floor": STANDARDIZATION_FLOOR,
        "image_std_floor": IMAGE_NORMALIZATION_STD_FLOOR,
        "proposal_multiplier": PROPOSAL_MULTIPLIER,
        "post_smoothing_passes": POST_SMOOTHING_PASSES,
        "posterior_temperature": POSTERIOR_TEMPERATURE,
        "centre_beta": CENTRE_BETA,
        "require_all_candidates_valid": REQUIRE_ALL_CANDIDATES_VALID,
        "stage_clip_sweeps": STAGE_LOCAL_CLIP_SWEEPS,
    }
)


PUBLISHED_CONSTRUCTION_KEYS: Mapping[str, str] = MappingProxyType(
    {
        "full_collar": "collar",
        "work_eps": "work_eps",
        "standardization_floor": "standardization_floor",
        "image_std_floor": "image_normalization_std_floor",
        "proposal_multiplier": "proposal_multiplier",
        "post_smoothing_passes": "post_smoothing_passes",
        "posterior_temperature": "posterior_temperature",
        "centre_beta": "centre_beta",
        "require_all_candidates_valid": "require_all_candidates_valid",
        "stage_clip_sweeps": "stage_local_clip_sweeps",
    }
)


def frozen_construction_kwargs() -> dict[str, Any]:
    """The exact build_pyramid_direction settings the frozen C6 policy owns."""

    return dict(FROZEN_CONSTRUCTION)


REFERENCE_ARM_ID = "c4_intensity_s2"
SELECTABLE_ARM_IDS = (
    "pyr21_a100",
    "pyr21_a150",
    "pyr421_a100",
    "pyr421_a150",
)
CONTROL_ARM_IDS = (
    "full21_a100",
    "full21_a150",
    "blur21_a100",
    "blur21_a150",
    "full421_a100",
    "full421_a150",
    "blur421_a100",
    "blur421_a150",
)
DIAGNOSTIC_ARM_IDS = ("pyr421_norewarp_a100",)
RISK_LABEL_IDS = (9, 29, 13)

CAPACITY_MEAN_VS_C4_MIN = 0.002
CAPACITY_CI_LOW_VS_C4_MIN_STRICT = 0.001
CAUSAL_MEAN_VS_CONTROL_MIN = 0.0005
CAUSAL_CI_LOW_VS_CONTROL_MIN_STRICT = 0.0
RETURNED_MEAN_VS_C4_MIN = 0.001
RETURNED_CI_LOW_VS_C4_MIN_STRICT = 0.0
SDLOGJ_CI_HIGH_VS_C4_MAX = 0.005
ALL_LABEL_CI_LOW_VS_C4_MIN_STRICT = -0.005
RISK_LABEL_CI_LOW_VS_C4_MIN_STRICT = -0.002

BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 0
BOOTSTRAP_CONFIDENCE = 0.95
BOOTSTRAP_METHOD_ID = "paired_case_bootstrap_max_absolute_centered_mean_deviation"

BRANCH_FREEZE = "FREEZE_C6_TRUE_PYRAMID_FOR_INDEPENDENT_CONFIRMATION"
BRANCH_NARROW_SAFETY = "OPEN_BOUNDED_C6B_PYRAMID_SAFETY_REPAIR"
BRANCH_CONTROLS_EXPLAIN = "CLOSE_TRUE_PYRAMID_CAUSAL_CLAIM_KEEP_SIMPLER_MULTISTAGE_CONTROL"
BRANCH_DROP_QUARTER = "FREEZE_C6_HALF_TO_FULL_DROP_QUARTER_LEVEL"
BRANCH_SIMPLIFY_FUSION = "OPEN_C6_NO_REWARP_SIMPLIFICATION_CHECK"
BRANCH_LEARNED = "CLOSE_HANDCRAFTED_INTENSITY_SEARCH_OPEN_LEARNED_OR_HYBRID_DESCRIPTOR"
BRANCH_INVALID = "INVALID_C6_EVIDENCE"


@dataclass(frozen=True, slots=True)
class ArmSpec:
    arm_index: int
    arm_id: str
    role: str
    family: str
    factors: tuple[int, ...]
    amplitude: float
    rewarp_between_levels: bool
    selectable: bool
    source_arm_id: str | None = None


ARM_SPECS = (
    ArmSpec(0, REFERENCE_ARM_ID, "FROZEN_REFERENCE", "source_c4", (), 1.0, True, False, "intensity_s2"),
    ArmSpec(1, "full21_a100", "MATCHED_CONTROL", "full_resolution", (2, 1), 1.0, True, False),
    ArmSpec(2, "full21_a150", "MATCHED_CONTROL", "full_resolution", (2, 1), 1.5, True, False),
    ArmSpec(3, "blur21_a100", "MATCHED_CONTROL", "blurred_full_resolution", (2, 1), 1.0, True, False),
    ArmSpec(4, "blur21_a150", "MATCHED_CONTROL", "blurred_full_resolution", (2, 1), 1.5, True, False),
    ArmSpec(5, "full421_a100", "MATCHED_CONTROL", "full_resolution", (4, 2, 1), 1.0, True, False),
    ArmSpec(6, "full421_a150", "MATCHED_CONTROL", "full_resolution", (4, 2, 1), 1.5, True, False),
    ArmSpec(7, "blur421_a100", "MATCHED_CONTROL", "blurred_full_resolution", (4, 2, 1), 1.0, True, False),
    ArmSpec(8, "blur421_a150", "MATCHED_CONTROL", "blurred_full_resolution", (4, 2, 1), 1.5, True, False),
    ArmSpec(9, "pyr21_a100", "SELECTABLE", "true_pyramid", (2, 1), 1.0, True, True),
    ArmSpec(10, "pyr21_a150", "SELECTABLE", "true_pyramid", (2, 1), 1.5, True, True),
    ArmSpec(11, "pyr421_a100", "SELECTABLE", "true_pyramid", (4, 2, 1), 1.0, True, True),
    ArmSpec(12, "pyr421_a150", "SELECTABLE", "true_pyramid", (4, 2, 1), 1.5, True, True),
    ArmSpec(13, "pyr421_norewarp_a100", "DIAGNOSTIC", "true_pyramid", (4, 2, 1), 1.0, False, False),
)
ARM_SPECS_BY_ID: Mapping[str, ArmSpec] = MappingProxyType({row.arm_id: row for row in ARM_SPECS})


def arm_ids_for_role(role: str) -> tuple[str, ...]:
    return tuple(row.arm_id for row in ARM_SPECS if row.role == role)


def matched_control_ids(arm_id: str) -> tuple[str, str]:
    spec = ARM_SPECS_BY_ID.get(arm_id)
    if spec is None or not spec.selectable:
        raise ValueError(f"not a selectable C6 arm: {arm_id}")
    suffix = "a100" if spec.amplitude == 1.0 else "a150"
    schedule = "21" if spec.factors == (2, 1) else "421"
    return f"full{schedule}_{suffix}", f"blur{schedule}_{suffix}"


@dataclass(frozen=True, slots=True)
class PairedSummary:
    family_id: str
    contrast_id: str
    n: int
    mean: float
    median: float
    ci_low: float
    ci_high: float
    improved: int
    worsened: int
    tied: int
    simultaneous_family_size: int
    bootstrap_resamples: int = BOOTSTRAP_RESAMPLES
    bootstrap_seed: int = BOOTSTRAP_SEED
    bootstrap_confidence: float = BOOTSTRAP_CONFIDENCE
    bootstrap_method: str = BOOTSTRAP_METHOD_ID


def simultaneous_paired_summaries(
    family_id: str,
    differences_by_contrast: Mapping[str, Sequence[float] | np.ndarray],
) -> dict[str, PairedSummary]:
    if not differences_by_contrast:
        raise ValueError("simultaneous family must not be empty")
    ids = tuple(differences_by_contrast)
    arrays = tuple(np.asarray(differences_by_contrast[key], dtype=np.float64) for key in ids)
    if any(row.shape != (EXPECTED_CASE_COUNT,) or not np.isfinite(row).all() for row in arrays):
        raise ValueError("every C6 contrast must contain exactly 58 finite paired values")
    matrix = np.stack(arrays)
    means = matrix.mean(axis=1)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    indices = rng.integers(0, EXPECTED_CASE_COUNT, size=(BOOTSTRAP_RESAMPLES, EXPECTED_CASE_COUNT))
    boot = matrix[:, indices].mean(axis=2)
    critical = float(np.quantile(np.abs(boot - means[:, None]).max(axis=0), BOOTSTRAP_CONFIDENCE, method="linear"))
    return {
        contrast_id: PairedSummary(
            family_id=family_id,
            contrast_id=contrast_id,
            n=EXPECTED_CASE_COUNT,
            mean=float(mean),
            median=float(np.median(values)),
            ci_low=float(mean - critical),
            ci_high=float(mean + critical),
            improved=int(np.count_nonzero(values > 0.0)),
            worsened=int(np.count_nonzero(values < 0.0)),
            tied=int(np.count_nonzero(values == 0.0)),
            simultaneous_family_size=len(ids),
        )
        for contrast_id, values, mean in zip(ids, arrays, means, strict=True)
    }


@dataclass(frozen=True, slots=True)
class ArmAssessment:
    arm_id: str
    capacity_material: bool
    causal_full_passed: bool
    causal_blur_passed: bool
    returned_material: bool
    geometry_passed: bool
    regional_passed: bool
    promotion_eligible: bool


def assess_arm(
    arm_id: str,
    *,
    capacity_vs_c4: PairedSummary,
    causal_vs_full: PairedSummary,
    causal_vs_blur: PairedSummary,
    returned_vs_c4: PairedSummary,
    sdlogj_vs_c4: PairedSummary,
    regional_vs_c4: Sequence[tuple[int, PairedSummary]],
    folds_all_zero: bool,
) -> ArmAssessment:
    if arm_id not in SELECTABLE_ARM_IDS:
        raise ValueError(f"not a selectable C6 arm: {arm_id}")
    capacity = (
        capacity_vs_c4.mean >= CAPACITY_MEAN_VS_C4_MIN and capacity_vs_c4.ci_low > CAPACITY_CI_LOW_VS_C4_MIN_STRICT
    )
    causal_full = (
        causal_vs_full.mean >= CAUSAL_MEAN_VS_CONTROL_MIN
        and causal_vs_full.ci_low > CAUSAL_CI_LOW_VS_CONTROL_MIN_STRICT
    )
    causal_blur = (
        causal_vs_blur.mean >= CAUSAL_MEAN_VS_CONTROL_MIN
        and causal_vs_blur.ci_low > CAUSAL_CI_LOW_VS_CONTROL_MIN_STRICT
    )
    returned = (
        returned_vs_c4.mean >= RETURNED_MEAN_VS_C4_MIN and returned_vs_c4.ci_low > RETURNED_CI_LOW_VS_C4_MIN_STRICT
    )
    geometry = folds_all_zero and sdlogj_vs_c4.ci_high <= SDLOGJ_CI_HIGH_VS_C4_MAX
    regional = all(
        summary.ci_low
        > (RISK_LABEL_CI_LOW_VS_C4_MIN_STRICT if label in RISK_LABEL_IDS else ALL_LABEL_CI_LOW_VS_C4_MIN_STRICT)
        for label, summary in regional_vs_c4
    )
    return ArmAssessment(
        arm_id=arm_id,
        capacity_material=capacity,
        causal_full_passed=causal_full,
        causal_blur_passed=causal_blur,
        returned_material=returned,
        geometry_passed=geometry,
        regional_passed=regional,
        promotion_eligible=capacity and causal_full and causal_blur and returned and geometry and regional,
    )


def select_branch(
    assessments: Sequence[ArmAssessment],
    capacity_means: Mapping[str, float],
    *,
    no_rewarp_vs_rewarp: PairedSummary | None,
    integrity_passed: bool = True,
) -> dict[str, Any]:
    if (
        not integrity_passed
        or len(assessments) != len(SELECTABLE_ARM_IDS)
        or {row.arm_id for row in assessments} != set(SELECTABLE_ARM_IDS)
    ):
        return {"branch": BRANCH_INVALID, "winner": None, "reason": "integrity or arm inventory failed"}
    eligible = [row for row in assessments if row.promotion_eligible]
    if eligible:
        winner = max(eligible, key=lambda row: (capacity_means[row.arm_id], -ARM_SPECS_BY_ID[row.arm_id].arm_index))
        if (
            winner.arm_id == "pyr421_a100"
            and no_rewarp_vs_rewarp is not None
            and no_rewarp_vs_rewarp.ci_low >= -CAUSAL_MEAN_VS_CONTROL_MIN
        ):
            return {
                "branch": BRANCH_SIMPLIFY_FUSION,
                "winner": winner.arm_id,
                "reason": "material pyramid signal exists and the no-rewarp diagnostic is practically noninferior",
            }
        branch = BRANCH_DROP_QUARTER if winner.arm_id.startswith("pyr21_") else BRANCH_FREEZE
        return {"branch": branch, "winner": winner.arm_id, "reason": "all preregistered promotion conditions passed"}
    material = [row for row in assessments if row.capacity_material]
    if material:
        if any(row.causal_full_passed and row.causal_blur_passed for row in material):
            return {
                "branch": BRANCH_NARROW_SAFETY,
                "winner": None,
                "reason": "causal capacity exists but safety or returned-policy evidence failed",
            }
        return {
            "branch": BRANCH_CONTROLS_EXPLAIN,
            "winner": None,
            "reason": "matched controls explain the apparent capacity",
        }
    return {"branch": BRANCH_LEARNED, "winner": None, "reason": "no material safe handcrafted pyramid signal"}


def policy_dict() -> dict[str, Any]:
    return {
        "protocol_id": PROTOCOL_ID,
        "schema_version": SCHEMA_VERSION,
        "dataset": DEVELOPMENT_DATASET_ID,
        "test_115_authorized": TEST_115_AUTHORIZED,
        "arms": [asdict(row) for row in ARM_SPECS],
        "construction": {
            "filter": PYRAMID_FILTER,
            "interpolation": PYRAMID_INTERPOLATION,
            "displacement_scaling": PYRAMID_DISPLACEMENT_SCALE,
            "stage_budget": STAGE_BUDGET,
            "rewarp": "moving image is sampled at the current provisional Psi before every finer level",
            "blurred_control_scope": BLURRED_CONTROL_SCOPE,
            "final_amplitudes": [1.0, 1.5],
            "centre_beta": CENTRE_BETA,
            "posterior_temperature": POSTERIOR_TEMPERATURE,
            "proposal_multiplier": PROPOSAL_MULTIPLIER,
            "post_smoothing_passes": POST_SMOOTHING_PASSES,
            "standardization_floor": STANDARDIZATION_FLOOR,
            "image_normalization_std_floor": IMAGE_NORMALIZATION_STD_FLOOR,
            "require_all_candidates_valid": REQUIRE_ALL_CANDIDATES_VALID,
            "work_eps": WORK_EPS,
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
            "causal_mean_vs_control_min": CAUSAL_MEAN_VS_CONTROL_MIN,
            "causal_ci_low_vs_control_min_strict": CAUSAL_CI_LOW_VS_CONTROL_MIN_STRICT,
            "returned_mean_vs_c4_min": RETURNED_MEAN_VS_C4_MIN,
            "returned_ci_low_vs_c4_min_strict": RETURNED_CI_LOW_VS_C4_MIN_STRICT,
            "sdlogj_ci_high_vs_c4_max": SDLOGJ_CI_HIGH_VS_C4_MAX,
            "all_label_ci_low_vs_c4_min_strict": ALL_LABEL_CI_LOW_VS_C4_MIN_STRICT,
            "risk_label_ci_low_vs_c4_min_strict": RISK_LABEL_CI_LOW_VS_C4_MIN_STRICT,
            "risk_label_ids": list(RISK_LABEL_IDS),
        },
        "statistics": {
            "resamples": BOOTSTRAP_RESAMPLES,
            "seed": BOOTSTRAP_SEED,
            "confidence": BOOTSTRAP_CONFIDENCE,
            "method": BOOTSTRAP_METHOD_ID,
        },
    }


def policy_sha256() -> str:
    payload = json.dumps(policy_dict(), sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


C6_POLICY_SHA256 = "72128551dab911171f628370704b0d219b0bb4f7d98aba571ceaede4dc8372b6"


def assert_frozen_policy() -> None:
    """Fail closed when the published contract, the arm inventory or a borrowed C3 value drifts."""

    if arm_ids_for_role("MATCHED_CONTROL") != CONTROL_ARM_IDS:
        raise RuntimeError("C6 control inventory disagrees with ARM_SPECS")
    if arm_ids_for_role("DIAGNOSTIC") != DIAGNOSTIC_ARM_IDS:
        raise RuntimeError("C6 diagnostic inventory disagrees with ARM_SPECS")
    if tuple(row.arm_id for row in ARM_SPECS if row.selectable) != SELECTABLE_ARM_IDS:
        raise RuntimeError("C6 selectable inventory disagrees with ARM_SPECS")
    construction = policy_dict()["construction"]
    borrowed = {
        "primary_ncc_window": NCC7_WINDOW,
        "ncc_denominator_eps": NCC_DENOMINATOR_EPS,
        "support_retention_min": SUPPORT_RETENTION_MIN,
        "primary_ncc_improvement_min": PRIMARY_NCC_IMPROVEMENT_MIN,
    }
    if any(construction[key] != value for key, value in borrowed.items()):
        raise RuntimeError("C6 published NCC contract disagrees with its search_gate_c3 owner")
    if any(construction[key] != FROZEN_CONSTRUCTION[builder] for builder, key in PUBLISHED_CONSTRUCTION_KEYS.items()):
        raise RuntimeError("C6 published construction disagrees with the settings passed to the builder")
    observed = policy_sha256()
    if observed != C6_POLICY_SHA256:
        raise RuntimeError(f"C6 policy changed: {observed} != {C6_POLICY_SHA256}")
