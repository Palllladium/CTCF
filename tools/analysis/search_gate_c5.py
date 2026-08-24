from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from types import MappingProxyType
from typing import Any

import numpy as np

PROTOCOL_ID = "CTCF-SEARCH-GATE-C5-V1"
SCHEMA_VERSION = "v1"
EXPECTED_CASE_COUNT = 58
TEST_115_AUTHORIZED = False
DEVELOPMENT_DATASET_ID = "IXI_VALIDATION_58"

DESCRIPTOR_ID = "ZSCORED_INTENSITY"
IMAGE_NORMALIZATION_MODE = "independent_masked_zscore"
IMAGE_NORMALIZATION_STD_FLOOR = 1e-6
CANDIDATE_COUNT = 27
STANDARDIZATION_MODE = "centered_two_pass_fp32"
STANDARDIZATION_FLOOR = 1e-6
DECODER_MODE = "posterior_mean"
POSTERIOR_TEMPERATURE = 1.0
POST_SMOOTHING_PASSES = 1
COMMON_EVIDENCE_COLLAR = 7
RMS_TARGET_SOURCE_ID = "source_c3_raw_conf_post1_requested"
RMS_TARGET_SOURCE_PATH_TEMPLATE = "source_historical/{case_id}/raw_conf_requested_field"
WORK_EPS = 0.0011
EXACT_CLAIM_EPS = 0.001
LOCAL_CLIP_SWEEPS = 1
NCC_WINDOW = 7
MIND_RADIUS = 1
MIND_DILATION = 2
AMPLITUDE_STAGE = "after_rms_match_before_local_clip"
CENTRE_PRIOR_FORMULA = "log_prior=-beta*sum(offset_zyx^2)/stride^2"
PRIMARY_REFERENCE_ARM_ID = "int_s2_a10_b0"
HISTORICAL_ANCHOR_ARM_IDS = ("int_s1_a10_b0", PRIMARY_REFERENCE_ARM_ID)

BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 0
BOOTSTRAP_CONFIDENCE = 0.95
BOOTSTRAP_METHOD_ID = "paired_case_bootstrap_max_absolute_centered_mean_deviation"
BOOTSTRAP_QUANTILE_METHOD = "linear"

CAPACITY_MEAN_DICE_DELTA_MIN = 0.002
CAPACITY_MEDIAN_DICE_DELTA_MIN_STRICT = 0.0
CAPACITY_CI_LOW_DICE_DELTA_MIN_STRICT = 0.001
INCREMENTAL_MEAN_DICE_DELTA_MIN = 0.0005
INCREMENTAL_CI_LOW_DICE_DELTA_MIN_STRICT = 0.0
SELECTOR_MEAN_DICE_DELTA_MIN = 0.001
SELECTOR_MEDIAN_DICE_DELTA_MIN_STRICT = 0.0
SELECTOR_CI_LOW_DICE_DELTA_MIN_STRICT = 0.0

UTILITY_SUPPORT_RETENTION_MIN = 0.99
UTILITY_IMPROVEMENT_MIN = 1e-6
AMPLITUDE_RETENTION_MEDIAN_MIN = 0.95
AMPLITUDE_RETENTION_CASE_MIN = 0.90
AMPLITUDE_RETENTION_CASE_COUNT_MIN = 52
FACTOR_BOUNDARY_MEAN_MIN = 0.0005
FACTOR_BOUNDARY_CI_LOW_MIN_STRICT = 0.0
REGIONAL_ALL_LABEL_CI_LOW_MIN_STRICT = -0.005
REGIONAL_RISK_LABEL_CI_LOW_MIN_STRICT = -0.002
REGIONAL_RISK_LABEL_IDS = (9, 29, 13)
REGIONAL_REPAIR_LABEL_IDS = (9, 29)
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
REGIONAL_REPAIR_MEAN_MIN = 0.002
REGIONAL_REPAIR_CI_LOW_MIN_STRICT = 0.0
RISK_REPAIR_AGGREGATE_CI_LOW_MIN_STRICT = -0.001
WINNER_MEAN_TIE_TOLERANCE = 1e-6

BRANCH_INVALID = "INVALID_C5_EVIDENCE"
BRANCH_CLIPPING = "OPEN_C5_CLIPPING_SATURATION"
BRANCH_CLOSE_NO_CAPACITY = "CLOSE_C5_NO_MATERIAL_CAPACITY"
BRANCH_FACTOR_BOUNDARY = "OPEN_C5_FACTOR_BOUNDARY"
BRANCH_GEOMETRY = "OPEN_C5_GEOMETRY"
BRANCH_REGION_RISK = "OPEN_C5_REGION_RISK"
BRANCH_UTILITY = "OPEN_C5_UTILITY"
BRANCH_FREEZE_SUPERIOR = "FREEZE_C5_SUPERIOR_FOR_INDEPENDENT_CONFIRMATION"
BRANCH_FREEZE_RISK_REPAIR = "FREEZE_C5_RISK_REPAIR_FOR_INDEPENDENT_CONFIRMATION"
BRANCH_RETAIN_REFERENCE = "RETAIN_C4_S2_CLOSE_C5"


def _offsets(stride: int) -> tuple[tuple[int, int, int], ...]:
    values = (-stride, 0, stride)
    return tuple((dz, dy, dx) for dz in values for dy in values for dx in values)


@dataclass(frozen=True, slots=True)
class ReachSpec:
    reach_id: str
    stride_voxels: int
    offsets_zyx: tuple[tuple[int, int, int], ...]
    pre_rms_multiplier: float


REACH_SPECS = tuple(
    ReachSpec(f"S{stride}", stride, _offsets(stride), 2.0 if stride == 1 else 1.0) for stride in range(1, 5)
)
REACH_SPECS_BY_ID: Mapping[str, ReachSpec] = MappingProxyType({spec.reach_id: spec for spec in REACH_SPECS})

AMPLITUDE_LEVELS = (("a05", 0.5), ("a10", 1.0), ("a20", 2.0))
BIAS_LEVELS = (("b0", 0, 0.0), ("b1", 1, math.log(2.0)), ("b2", 2, math.log(4.0)))


@dataclass(frozen=True, slots=True)
class ArmSpec:
    arm_index: int
    arm_id: str
    descriptor_id: str
    reach_id: str
    stride_voxels: int
    amplitude_token: str
    post_rms_amplitude: float
    bias_token: str
    bias_level: int
    centre_beta: float
    historical_anchor: bool
    selectable: bool


def _build_arms() -> tuple[ArmSpec, ...]:
    arms: list[ArmSpec] = []
    for reach in REACH_SPECS:
        for amplitude_token, amplitude in AMPLITUDE_LEVELS:
            for bias_token, bias_level, beta in BIAS_LEVELS:
                arm_id = f"int_s{reach.stride_voxels}_{amplitude_token}_{bias_token}"
                arms.append(
                    ArmSpec(
                        arm_index=len(arms),
                        arm_id=arm_id,
                        descriptor_id=DESCRIPTOR_ID,
                        reach_id=reach.reach_id,
                        stride_voxels=reach.stride_voxels,
                        amplitude_token=amplitude_token,
                        post_rms_amplitude=amplitude,
                        bias_token=bias_token,
                        bias_level=bias_level,
                        centre_beta=beta,
                        historical_anchor=arm_id in HISTORICAL_ANCHOR_ARM_IDS,
                        selectable=True,
                    )
                )
    return tuple(arms)


ARM_SPECS = _build_arms()
ARM_SPECS_BY_ID: Mapping[str, ArmSpec] = MappingProxyType({spec.arm_id: spec for spec in ARM_SPECS})
SELECTABLE_ARM_IDS = tuple(spec.arm_id for spec in ARM_SPECS)


@dataclass(frozen=True, slots=True)
class SelectorSpec:
    selector_index: int
    selector_id: str
    requires_ncc7_improvement: bool
    requires_mind_d2_improvement: bool
    geometry_delta_cap: float
    primary: bool


SELECTOR_SPECS = (
    SelectorSpec(0, "dual_g010", True, True, 0.010, True),
    SelectorSpec(1, "dual_g005", True, True, 0.005, False),
    SelectorSpec(2, "dual_g020", True, True, 0.020, False),
    SelectorSpec(3, "ncc_g010", True, False, 0.010, False),
    SelectorSpec(4, "mind_g010", False, True, 0.010, False),
)
SELECTOR_SPECS_BY_ID: Mapping[str, SelectorSpec] = MappingProxyType({spec.selector_id: spec for spec in SELECTOR_SPECS})
SELECTOR_IDS = tuple(spec.selector_id for spec in SELECTOR_SPECS)
PRIMARY_SELECTOR_ID = "dual_g010"


@dataclass(frozen=True, slots=True)
class ContrastSpec:
    contrast_index: int
    contrast_id: str
    family_id: str
    candidate_id: str
    reference_id: str
    diagnostic_only: bool


CAPACITY_FAMILY_ID = "capacity_vs_zero"
INCREMENTAL_FAMILY_ID = "capacity_vs_c4_intensity_s2"
MARGINAL_FAMILY_ID = "factor_adjacent_marginals"
INTERACTION_FAMILY_ID = "factor_trend_interactions"
SELECTOR_ZERO_FAMILY_ID = "selector_vs_zero"
SELECTOR_REFERENCE_FAMILY_ID = "selector_vs_c4_intensity_s2"
REGIONAL_ZERO_FAMILY_ID = "primary_selector_labels_vs_zero"
REGIONAL_REFERENCE_FAMILY_ID = "primary_selector_risk_labels_vs_c4_intensity_s2"


def _contrast_rows() -> tuple[ContrastSpec, ...]:
    rows: list[ContrastSpec] = []

    def add(
        contrast_id: str,
        family_id: str,
        candidate_id: str,
        reference_id: str,
        *,
        diagnostic_only: bool = False,
    ) -> None:
        rows.append(
            ContrastSpec(
                len(rows),
                contrast_id,
                family_id,
                candidate_id,
                reference_id,
                diagnostic_only,
            )
        )

    for arm_id in SELECTABLE_ARM_IDS:
        add(
            f"capacity::{arm_id}::vs_zero",
            CAPACITY_FAMILY_ID,
            arm_id,
            "zero_update_baseline",
        )
    for arm_id in SELECTABLE_ARM_IDS:
        if arm_id != PRIMARY_REFERENCE_ARM_ID:
            add(
                f"incremental::{arm_id}::vs_{PRIMARY_REFERENCE_ARM_ID}",
                INCREMENTAL_FAMILY_ID,
                arm_id,
                PRIMARY_REFERENCE_ARM_ID,
            )
    for contrast_id, candidate, reference in (
        ("reach_s2_vs_s1", "S2", "S1"),
        ("reach_s3_vs_s2", "S3", "S2"),
        ("reach_s4_vs_s3", "S4", "S3"),
        ("amplitude_a10_vs_a05", "a10", "a05"),
        ("amplitude_a20_vs_a10", "a20", "a10"),
        ("bias_b1_vs_b0", "b1", "b0"),
        ("bias_b2_vs_b1", "b2", "b1"),
    ):
        add(contrast_id, MARGINAL_FAMILY_ID, candidate, reference)
    for contrast_id in (
        "interaction_reach_amplitude",
        "interaction_reach_bias",
        "interaction_amplitude_bias",
        "interaction_reach_amplitude_bias",
    ):
        add(contrast_id, INTERACTION_FAMILY_ID, "factorial_endpoint", "factorial_endpoint", diagnostic_only=True)
    for selector_id in SELECTOR_IDS:
        add(
            f"selector::{selector_id}::vs_zero",
            SELECTOR_ZERO_FAMILY_ID,
            selector_id,
            "zero_update_baseline",
        )
    for selector_id in SELECTOR_IDS:
        add(
            f"selector::{selector_id}::vs_{PRIMARY_REFERENCE_ARM_ID}",
            SELECTOR_REFERENCE_FAMILY_ID,
            selector_id,
            PRIMARY_REFERENCE_ARM_ID,
        )
    return tuple(rows)


CONTRAST_SPECS = _contrast_rows()
CONTRAST_SPECS_BY_ID: Mapping[str, ContrastSpec] = MappingProxyType({spec.contrast_id: spec for spec in CONTRAST_SPECS})

REGIONAL_CONTRAST_SPECS = tuple(
    ContrastSpec(
        index,
        f"label::{label_id}::{PRIMARY_SELECTOR_ID}::vs_zero",
        REGIONAL_ZERO_FAMILY_ID,
        f"label_{label_id}",
        "zero_update_baseline",
        False,
    )
    for index, label_id in enumerate(EVALUATION_LABEL_IDS)
) + tuple(
    ContrastSpec(
        30 + index,
        f"label::{label_id}::{PRIMARY_SELECTOR_ID}::vs_{PRIMARY_REFERENCE_ARM_ID}",
        REGIONAL_REFERENCE_FAMILY_ID,
        f"label_{label_id}",
        PRIMARY_REFERENCE_ARM_ID,
        False,
    )
    for index, label_id in enumerate(REGIONAL_REPAIR_LABEL_IDS)
)

ALL_CONTRAST_SPECS = CONTRAST_SPECS + REGIONAL_CONTRAST_SPECS
if len({spec.contrast_id for spec in ALL_CONTRAST_SPECS}) != len(ALL_CONTRAST_SPECS):
    raise RuntimeError("C5 contrast IDs must be unique")

INFERENCE_FAMILY_IDS = (
    CAPACITY_FAMILY_ID,
    INCREMENTAL_FAMILY_ID,
    MARGINAL_FAMILY_ID,
    INTERACTION_FAMILY_ID,
    SELECTOR_ZERO_FAMILY_ID,
    SELECTOR_REFERENCE_FAMILY_ID,
    REGIONAL_ZERO_FAMILY_ID,
    REGIONAL_REFERENCE_FAMILY_ID,
)
CONTRAST_IDS_BY_FAMILY: Mapping[str, tuple[str, ...]] = MappingProxyType(
    {
        family_id: tuple(spec.contrast_id for spec in ALL_CONTRAST_SPECS if spec.family_id == family_id)
        for family_id in INFERENCE_FAMILY_IDS
    }
)
BOOTSTRAP_FAMILY_SIZES: Mapping[str, int] = MappingProxyType(
    {family_id: len(ids) for family_id, ids in CONTRAST_IDS_BY_FAMILY.items()}
)
BOOTSTRAP_MAIN_CONTRAST_COUNT = len(CONTRAST_SPECS)


@dataclass(frozen=True, slots=True)
class C5Policy:
    protocol_id: str
    schema_version: str
    development_dataset_id: str
    test_115_authorized: bool
    reaches: tuple[ReachSpec, ...]
    amplitudes: tuple[tuple[str, float], ...]
    biases: tuple[tuple[str, int, float], ...]
    arms: tuple[ArmSpec, ...]
    selectors: tuple[SelectorSpec, ...]
    contrasts: tuple[ContrastSpec, ...]
    regional_contrasts: tuple[ContrastSpec, ...]
    proposal_pipeline: tuple[tuple[str, Any], ...]
    thresholds: tuple[tuple[str, Any], ...]

    def to_dict(self) -> dict[str, Any]:
        return _json_compatible(asdict(self))


C5_POLICY = C5Policy(
    protocol_id=PROTOCOL_ID,
    schema_version=SCHEMA_VERSION,
    development_dataset_id=DEVELOPMENT_DATASET_ID,
    test_115_authorized=TEST_115_AUTHORIZED,
    reaches=REACH_SPECS,
    amplitudes=AMPLITUDE_LEVELS,
    biases=BIAS_LEVELS,
    arms=ARM_SPECS,
    selectors=SELECTOR_SPECS,
    contrasts=CONTRAST_SPECS,
    regional_contrasts=REGIONAL_CONTRAST_SPECS,
    proposal_pipeline=(
        ("descriptor", DESCRIPTOR_ID),
        ("image_normalization", IMAGE_NORMALIZATION_MODE),
        ("image_normalization_std_floor", IMAGE_NORMALIZATION_STD_FLOOR),
        ("candidate_count", CANDIDATE_COUNT),
        ("offset_order", "zyx lexicographic over negative, zero, positive stride"),
        ("candidate_standardization", STANDARDIZATION_MODE),
        ("candidate_standardization_floor", STANDARDIZATION_FLOOR),
        ("decoder", DECODER_MODE),
        ("posterior_temperature", POSTERIOR_TEMPERATURE),
        ("post_smoothing_passes", POST_SMOOTHING_PASSES),
        ("evidence_collar", COMMON_EVIDENCE_COLLAR),
        ("rms_target_source_id", RMS_TARGET_SOURCE_ID),
        ("rms_target_source_path_template", RMS_TARGET_SOURCE_PATH_TEMPLATE),
        ("work_eps", WORK_EPS),
        ("exact_claim_eps", EXACT_CLAIM_EPS),
        ("local_clip_sweeps", LOCAL_CLIP_SWEEPS),
        ("pre_rms_multiplier_by_reach", tuple((spec.reach_id, spec.pre_rms_multiplier) for spec in REACH_SPECS)),
        ("amplitude_stage", AMPLITUDE_STAGE),
        ("centre_prior_formula", CENTRE_PRIOR_FORMULA),
        ("all_candidates_materialized_before_labels", True),
        ("exact_certificate_required", True),
        ("global_selector_scope", "all_36_exact_candidates_per_case"),
        ("global_selector_amplitude_retention_min", AMPLITUDE_RETENTION_CASE_MIN),
        (
            "global_selector_rank",
            "max enabled MIND-D2 gain, max enabled NCC7 gain, min mathematical SDlogJ delta, lower alpha, lower stride, lower beta, arm index",
        ),
        ("selector_none_action", "return_zero_update_baseline"),
        ("diagnostic_utility_definitions", "support_contract_owned"),
        ("selector_ncc_window", NCC_WINDOW),
        ("selector_mind_radius", MIND_RADIUS),
        ("selector_mind_dilation", MIND_DILATION),
        (
            "factor_marginals",
            "adjacent level difference averaged equally over every cell of the other two factors",
        ),
        (
            "factor_interactions",
            "endpoint differences for reach x amplitude, reach x bias, amplitude x bias, and three-way; remaining factor equally averaged",
        ),
        ("test_115_authorized", TEST_115_AUTHORIZED),
    ),
    thresholds=(
        ("bootstrap_resamples", BOOTSTRAP_RESAMPLES),
        ("bootstrap_seed", BOOTSTRAP_SEED),
        ("bootstrap_confidence", BOOTSTRAP_CONFIDENCE),
        ("bootstrap_method", BOOTSTRAP_METHOD_ID),
        ("bootstrap_quantile_method", BOOTSTRAP_QUANTILE_METHOD),
        ("bootstrap_family_ids", INFERENCE_FAMILY_IDS),
        ("bootstrap_family_sizes", tuple(BOOTSTRAP_FAMILY_SIZES.items())),
        ("capacity_mean_min", CAPACITY_MEAN_DICE_DELTA_MIN),
        ("capacity_median", ">0"),
        ("capacity_ci_low", ">0.001"),
        ("incremental_mean_min", INCREMENTAL_MEAN_DICE_DELTA_MIN),
        ("incremental_ci_low", ">0"),
        ("selector_mean_min", SELECTOR_MEAN_DICE_DELTA_MIN),
        ("selector_median", ">0"),
        ("selector_ci_low", ">0"),
        ("utility_support_retention_min", UTILITY_SUPPORT_RETENTION_MIN),
        ("utility_improvement_min", UTILITY_IMPROVEMENT_MIN),
        ("amplitude_retention_median_min", AMPLITUDE_RETENTION_MEDIAN_MIN),
        ("amplitude_retention_case_min", AMPLITUDE_RETENTION_CASE_MIN),
        ("amplitude_retention_case_count_min", AMPLITUDE_RETENTION_CASE_COUNT_MIN),
        ("all_label_ci_low", ">-0.005"),
        ("risk_label_ids", REGIONAL_RISK_LABEL_IDS),
        ("risk_label_ci_low", ">-0.002"),
        ("repair_label_ids", REGIONAL_REPAIR_LABEL_IDS),
        ("repair_label_mean_min", REGIONAL_REPAIR_MEAN_MIN),
        ("repair_label_ci_low", ">0"),
        (
            "branch_priority",
            (
                BRANCH_INVALID,
                BRANCH_CLIPPING,
                BRANCH_CLOSE_NO_CAPACITY,
                BRANCH_FACTOR_BOUNDARY,
                BRANCH_GEOMETRY,
                BRANCH_FREEZE_SUPERIOR,
                BRANCH_FREEZE_RISK_REPAIR,
                BRANCH_REGION_RISK,
                BRANCH_UTILITY,
                BRANCH_RETAIN_REFERENCE,
            ),
        ),
        (
            "factor_boundary_requires",
            "best eligible arm is at S4, a20, or b2 and that factor's upper adjacent marginal mean>=0.0005, ci_low>0",
        ),
        ("risk_repair_aggregate_ci_low", ">-0.001"),
        (
            "risk_repair_safety",
            "all 30 label ci_low>-0.005 and label 13 ci_low>-0.002; only labels 9/29 receive the repair exception",
        ),
    ),
)


@dataclass(frozen=True, slots=True)
class C5DecisionPolicy:
    protocol_id: str
    schema_version: str
    development_dataset_id: str
    expected_case_count: int
    test_115_authorized: bool
    labels_accessible: bool
    reaches: tuple[ReachSpec, ...]
    amplitudes: tuple[tuple[str, float], ...]
    biases: tuple[tuple[str, int, float], ...]
    arms: tuple[ArmSpec, ...]
    selectors: tuple[SelectorSpec, ...]
    proposal_pipeline: tuple[tuple[str, Any], ...]
    selector_thresholds: tuple[tuple[str, Any], ...]

    def to_dict(self) -> dict[str, Any]:
        return _json_compatible(asdict(self))


C5_DECISION_POLICY = C5DecisionPolicy(
    protocol_id=PROTOCOL_ID,
    schema_version=SCHEMA_VERSION,
    development_dataset_id=DEVELOPMENT_DATASET_ID,
    expected_case_count=EXPECTED_CASE_COUNT,
    test_115_authorized=TEST_115_AUTHORIZED,
    labels_accessible=False,
    reaches=REACH_SPECS,
    amplitudes=AMPLITUDE_LEVELS,
    biases=BIAS_LEVELS,
    arms=ARM_SPECS,
    selectors=SELECTOR_SPECS,
    proposal_pipeline=C5_POLICY.proposal_pipeline,
    selector_thresholds=(
        ("utility_support_retention_min", UTILITY_SUPPORT_RETENTION_MIN),
        ("utility_improvement_min", UTILITY_IMPROVEMENT_MIN),
        ("amplitude_retention_case_min", AMPLITUDE_RETENTION_CASE_MIN),
        ("selector_none_action", "return_zero_update_baseline"),
    ),
)


def _json_compatible(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_compatible(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_compatible(item) for item in value]
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    return value


def canonical_policy_bytes(policy: C5Policy = C5_POLICY) -> bytes:
    return json.dumps(
        policy.to_dict(),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def policy_sha256(policy: C5Policy = C5_POLICY) -> str:
    return hashlib.sha256(canonical_policy_bytes(policy)).hexdigest()


def canonical_decision_policy_bytes(policy: C5DecisionPolicy = C5_DECISION_POLICY) -> bytes:
    return json.dumps(
        policy.to_dict(),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def decision_policy_sha256(policy: C5DecisionPolicy = C5_DECISION_POLICY) -> str:
    return hashlib.sha256(canonical_decision_policy_bytes(policy)).hexdigest()


def decision_policy_contract() -> dict[str, Any]:
    return C5_DECISION_POLICY.to_dict()


# Literal digest: changing any scientific choice requires a deliberate update.
C5_POLICY_SHA256 = "5612a190b5fe4adc581bdffe02fa5db762a0d5038df737a242abf593f1992097"
C5_DECISION_POLICY_SHA256 = "d32054e187cebffb5cf349912da92b7689a1911a0ff915276f51b8943f9a0af9"


def assert_frozen_policy() -> None:
    actual = policy_sha256()
    if actual != C5_POLICY_SHA256:
        raise RuntimeError(f"C5 policy hash mismatch: declared={C5_POLICY_SHA256}, actual={actual}")


def assert_frozen_decision_policy() -> None:
    actual = decision_policy_sha256()
    if actual != C5_DECISION_POLICY_SHA256:
        raise RuntimeError(f"C5 decision-policy hash mismatch: declared={C5_DECISION_POLICY_SHA256}, actual={actual}")


def apply_post_rms_amplitude(values: Sequence[float] | np.ndarray, arm_id: str) -> np.ndarray:
    spec = ARM_SPECS_BY_ID.get(arm_id)
    if spec is None:
        raise ValueError(f"unknown C5 arm: {arm_id}")
    array = np.asarray(values)
    if array.size == 0 or not np.issubdtype(array.dtype, np.number) or not np.isfinite(array).all():
        raise ValueError("rms_matched_values must be a non-empty finite numeric array")
    return array * spec.post_rms_amplitude


def centre_log_prior(arm_id: str) -> tuple[float, ...]:
    spec = ARM_SPECS_BY_ID.get(arm_id)
    if spec is None:
        raise ValueError(f"unknown C5 arm: {arm_id}")
    offsets = REACH_SPECS_BY_ID[spec.reach_id].offsets_zyx
    stride_squared = float(spec.stride_voxels**2)
    return tuple(-spec.centre_beta * sum(value * value for value in offset) / stride_squared for offset in offsets)


@dataclass(frozen=True, slots=True)
class CandidateSignals:
    arm_id: str
    exact_certified: bool
    support_retention: float
    amplitude_retention: float
    ncc7_improvement: float
    mind_d2_improvement: float
    mathematical_sdlogj_delta: float


@dataclass(frozen=True, slots=True)
class SelectorChoice:
    selector_id: str
    selected_arm_id: str | None
    action: str
    eligible_arm_ids: tuple[str, ...]


def _validate_candidate_signals(rows: Sequence[CandidateSignals]) -> None:
    if not isinstance(rows, Sequence):
        raise TypeError("candidate signals must be a sequence")
    if any(not isinstance(row, CandidateSignals) for row in rows):
        raise TypeError("candidate signals must contain CandidateSignals values")
    if tuple(row.arm_id for row in rows) != SELECTABLE_ARM_IDS:
        raise ValueError("selector requires all 36 candidates in frozen arm order")
    for row in rows:
        if not isinstance(row.exact_certified, bool):
            raise TypeError("exact_certified must be boolean")
        values = (
            row.support_retention,
            row.amplitude_retention,
            row.ncc7_improvement,
            row.mind_d2_improvement,
            row.mathematical_sdlogj_delta,
        )
        if any(not math.isfinite(float(value)) for value in values):
            raise ValueError(f"candidate signals are non-finite: {row.arm_id}")
        if not 0.0 <= float(row.support_retention) <= 1.0:
            raise ValueError(f"support retention is outside [0,1]: {row.arm_id}")
        if not 0.0 <= float(row.amplitude_retention) <= 1.0:
            raise ValueError(f"amplitude retention is outside [0,1]: {row.arm_id}")


def choose_global_candidate(rows: Sequence[CandidateSignals], selector_id: str) -> SelectorChoice:
    _validate_candidate_signals(rows)
    selector = SELECTOR_SPECS_BY_ID.get(selector_id)
    if selector is None:
        raise ValueError(f"unknown C5 selector: {selector_id}")

    eligible: list[CandidateSignals] = []
    for row in rows:
        if (
            not row.exact_certified
            or row.support_retention < UTILITY_SUPPORT_RETENTION_MIN
            or row.amplitude_retention < AMPLITUDE_RETENTION_CASE_MIN
        ):
            continue
        if selector.requires_ncc7_improvement and row.ncc7_improvement < UTILITY_IMPROVEMENT_MIN:
            continue
        if selector.requires_mind_d2_improvement and row.mind_d2_improvement < UTILITY_IMPROVEMENT_MIN:
            continue
        if row.mathematical_sdlogj_delta > selector.geometry_delta_cap:
            continue
        eligible.append(row)
    if not eligible:
        return SelectorChoice(selector_id, None, "RETURN_BASELINE", ())

    def rank(row: CandidateSignals) -> tuple[float, ...]:
        arm = ARM_SPECS_BY_ID[row.arm_id]
        return (
            -row.mind_d2_improvement if selector.requires_mind_d2_improvement else 0.0,
            -row.ncc7_improvement if selector.requires_ncc7_improvement else 0.0,
            row.mathematical_sdlogj_delta,
            arm.post_rms_amplitude,
            float(arm.stride_voxels),
            float(arm.bias_level),
            float(arm.arm_index),
        )

    selected = min(eligible, key=rank)
    return SelectorChoice(selector_id, selected.arm_id, "RETURN_CANDIDATE", tuple(row.arm_id for row in eligible))


def _finite_vector(values: Sequence[float] | np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size == 0 or not np.isfinite(array).all():
        raise ValueError(f"{name} must be a non-empty finite one-dimensional sequence")
    return array


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
    bootstrap_resamples: int
    bootstrap_seed: int
    bootstrap_confidence: float
    bootstrap_method: str
    simultaneous_family_size: int


def simultaneous_paired_summaries(
    family_id: str,
    differences: Mapping[str, Sequence[float] | np.ndarray],
) -> dict[str, PairedSummary]:
    expected_ids = CONTRAST_IDS_BY_FAMILY.get(family_id)
    if expected_ids is None:
        raise ValueError(f"unknown C5 inference family: {family_id}")
    if tuple(differences) != expected_ids:
        raise ValueError(f"C5 {family_id} bootstrap requires every contrast in frozen order")
    matrix = np.stack([_finite_vector(differences[contrast_id], contrast_id) for contrast_id in expected_ids])
    if matrix.shape != (len(expected_ids), EXPECTED_CASE_COUNT):
        raise ValueError("C5 simultaneous contrasts must each contain exactly 58 paired cases")

    generator = np.random.default_rng(BOOTSTRAP_SEED)
    indices = generator.integers(0, EXPECTED_CASE_COUNT, size=(BOOTSTRAP_RESAMPLES, EXPECTED_CASE_COUNT))
    observed = matrix.mean(axis=1)
    max_deviation = np.empty(BOOTSTRAP_RESAMPLES, dtype=np.float64)
    batch_size = 256
    for start in range(0, BOOTSTRAP_RESAMPLES, batch_size):
        stop = min(start + batch_size, BOOTSTRAP_RESAMPLES)
        boot = matrix[:, indices[start:stop]].mean(axis=2)
        max_deviation[start:stop] = np.abs(boot - observed[:, None]).max(axis=0)
    critical = float(np.quantile(max_deviation, BOOTSTRAP_CONFIDENCE, method=BOOTSTRAP_QUANTILE_METHOD))

    output: dict[str, PairedSummary] = {}
    for index, contrast_id in enumerate(expected_ids):
        values = matrix[index]
        output[contrast_id] = PairedSummary(
            family_id=family_id,
            contrast_id=contrast_id,
            n=EXPECTED_CASE_COUNT,
            mean=float(observed[index]),
            median=float(np.median(values)),
            ci_low=float(observed[index] - critical),
            ci_high=float(observed[index] + critical),
            improved=int((values > 0.0).sum()),
            worsened=int((values < 0.0).sum()),
            tied=int((values == 0.0).sum()),
            bootstrap_resamples=BOOTSTRAP_RESAMPLES,
            bootstrap_seed=BOOTSTRAP_SEED,
            bootstrap_confidence=BOOTSTRAP_CONFIDENCE,
            bootstrap_method=BOOTSTRAP_METHOD_ID,
            simultaneous_family_size=len(expected_ids),
        )
    return output


def _arm_matrix(capacity_deltas: Mapping[str, Sequence[float] | np.ndarray]) -> dict[str, np.ndarray]:
    if tuple(capacity_deltas) != SELECTABLE_ARM_IDS:
        raise ValueError("factor contrasts require all 36 arm deltas in frozen order")
    rows = {arm_id: _finite_vector(capacity_deltas[arm_id], arm_id) for arm_id in SELECTABLE_ARM_IDS}
    if any(row.size != EXPECTED_CASE_COUNT for row in rows.values()):
        raise ValueError("each C5 arm delta must contain exactly 58 paired cases")
    return rows


def factor_contrast_differences(
    capacity_deltas: Mapping[str, Sequence[float] | np.ndarray],
) -> dict[str, np.ndarray]:
    rows = _arm_matrix(capacity_deltas)

    def cell(reach: int, amplitude: str, bias: str) -> np.ndarray:
        return rows[f"int_s{reach}_{amplitude}_{bias}"]

    def average(values: Sequence[np.ndarray]) -> np.ndarray:
        return np.stack(values).mean(axis=0)

    output: dict[str, np.ndarray] = {}
    for contrast_id, high, low in (
        ("reach_s2_vs_s1", 2, 1),
        ("reach_s3_vs_s2", 3, 2),
        ("reach_s4_vs_s3", 4, 3),
    ):
        output[contrast_id] = average(
            [
                cell(high, amplitude, bias) - cell(low, amplitude, bias)
                for amplitude, _ in AMPLITUDE_LEVELS
                for bias, _, _ in BIAS_LEVELS
            ]
        )
    for contrast_id, high, low in (
        ("amplitude_a10_vs_a05", "a10", "a05"),
        ("amplitude_a20_vs_a10", "a20", "a10"),
    ):
        output[contrast_id] = average(
            [
                cell(reach.stride_voxels, high, bias) - cell(reach.stride_voxels, low, bias)
                for reach in REACH_SPECS
                for bias, _, _ in BIAS_LEVELS
            ]
        )
    for contrast_id, high, low in (("bias_b1_vs_b0", "b1", "b0"), ("bias_b2_vs_b1", "b2", "b1")):
        output[contrast_id] = average(
            [
                cell(reach.stride_voxels, amplitude, high) - cell(reach.stride_voxels, amplitude, low)
                for reach in REACH_SPECS
                for amplitude, _ in AMPLITUDE_LEVELS
            ]
        )

    output["interaction_reach_amplitude"] = average(
        [
            (cell(4, "a20", bias) - cell(4, "a05", bias)) - (cell(1, "a20", bias) - cell(1, "a05", bias))
            for bias, _, _ in BIAS_LEVELS
        ]
    )
    output["interaction_reach_bias"] = average(
        [
            (cell(4, amplitude, "b2") - cell(4, amplitude, "b0"))
            - (cell(1, amplitude, "b2") - cell(1, amplitude, "b0"))
            for amplitude, _ in AMPLITUDE_LEVELS
        ]
    )
    output["interaction_amplitude_bias"] = average(
        [
            (cell(reach.stride_voxels, "a20", "b2") - cell(reach.stride_voxels, "a20", "b0"))
            - (cell(reach.stride_voxels, "a05", "b2") - cell(reach.stride_voxels, "a05", "b0"))
            for reach in REACH_SPECS
        ]
    )
    high_bias = (cell(4, "a20", "b2") - cell(4, "a05", "b2")) - (cell(1, "a20", "b2") - cell(1, "a05", "b2"))
    low_bias = (cell(4, "a20", "b0") - cell(4, "a05", "b0")) - (cell(1, "a20", "b0") - cell(1, "a05", "b0"))
    output["interaction_reach_amplitude_bias"] = high_bias - low_bias
    expected = CONTRAST_IDS_BY_FAMILY[MARGINAL_FAMILY_ID] + CONTRAST_IDS_BY_FAMILY[INTERACTION_FAMILY_ID]
    if tuple(output) != expected:
        raise RuntimeError("C5 factor contrast implementation changed frozen order")
    return output


def _valid_summary(
    summary: PairedSummary,
    family_id: str,
    name: str,
    expected_contrast_id: str | None = None,
) -> None:
    if not isinstance(summary, PairedSummary):
        raise TypeError(f"{name} must be a PairedSummary")
    if summary.family_id != family_id:
        raise ValueError(f"{name} belongs to the wrong inference family")
    if expected_contrast_id is not None and summary.contrast_id != expected_contrast_id:
        raise ValueError(f"{name} belongs to the wrong contrast")
    if summary.contrast_id not in CONTRAST_IDS_BY_FAMILY[family_id]:
        raise ValueError(f"{name} has an unknown contrast ID")
    values = (summary.mean, summary.median, summary.ci_low, summary.ci_high)
    if any(not math.isfinite(float(value)) for value in values):
        raise ValueError(f"{name} contains a non-finite statistic")
    if summary.n != EXPECTED_CASE_COUNT or summary.improved + summary.worsened + summary.tied != summary.n:
        raise ValueError(f"{name} has invalid paired-case accounting")
    if any(not isinstance(value, int) or value < 0 for value in (summary.improved, summary.worsened, summary.tied)):
        raise ValueError(f"{name} has invalid outcome counts")
    if summary.ci_low > summary.mean or summary.mean > summary.ci_high:
        raise ValueError(f"{name} has an invalid confidence interval")
    if (
        summary.bootstrap_resamples != BOOTSTRAP_RESAMPLES
        or summary.bootstrap_seed != BOOTSTRAP_SEED
        or summary.bootstrap_confidence != BOOTSTRAP_CONFIDENCE
        or summary.bootstrap_method != BOOTSTRAP_METHOD_ID
        or summary.simultaneous_family_size != BOOTSTRAP_FAMILY_SIZES[family_id]
    ):
        raise ValueError(f"{name} does not use the frozen simultaneous inference contract")


def materially_strong_capacity(summary: PairedSummary) -> bool:
    _valid_summary(summary, CAPACITY_FAMILY_ID, "capacity_vs_zero")
    return (
        summary.mean >= CAPACITY_MEAN_DICE_DELTA_MIN
        and summary.median > CAPACITY_MEDIAN_DICE_DELTA_MIN_STRICT
        and summary.ci_low > CAPACITY_CI_LOW_DICE_DELTA_MIN_STRICT
    )


def materially_incremental(summary: PairedSummary) -> bool:
    _valid_summary(summary, INCREMENTAL_FAMILY_ID, "capacity_vs_c4_intensity_s2")
    return summary.mean >= INCREMENTAL_MEAN_DICE_DELTA_MIN and summary.ci_low > INCREMENTAL_CI_LOW_DICE_DELTA_MIN_STRICT


def materially_strong_selector(summary: PairedSummary) -> bool:
    _valid_summary(summary, SELECTOR_ZERO_FAMILY_ID, "selector_vs_zero")
    return (
        summary.mean >= SELECTOR_MEAN_DICE_DELTA_MIN
        and summary.median > SELECTOR_MEDIAN_DICE_DELTA_MIN_STRICT
        and summary.ci_low > SELECTOR_CI_LOW_DICE_DELTA_MIN_STRICT
    )


@dataclass(frozen=True, slots=True)
class ArmEvidence:
    arm_id: str
    capacity_vs_zero: PairedSummary
    incremental_vs_reference: PairedSummary | None
    all_work_units_complete: bool
    all_exact_certified: bool
    amplitude_retention_median: float
    amplitude_retention_cases_at_least_090: int


@dataclass(frozen=True, slots=True)
class ArmAssessment:
    arm_id: str
    material_capacity: bool
    incremental_over_reference: bool
    execution_complete: bool
    amplitude_interpretable: bool
    eligible: bool


def assess_arm(evidence: ArmEvidence) -> ArmAssessment:
    if not isinstance(evidence, ArmEvidence):
        raise TypeError("evidence must be ArmEvidence")
    if evidence.arm_id not in ARM_SPECS_BY_ID:
        raise ValueError(f"unknown C5 arm: {evidence.arm_id}")
    if not isinstance(evidence.all_work_units_complete, bool) or not isinstance(evidence.all_exact_certified, bool):
        raise TypeError("execution flags must be boolean")
    if not math.isfinite(evidence.amplitude_retention_median) or not 0.0 <= evidence.amplitude_retention_median <= 1.0:
        raise ValueError("amplitude retention median must lie in [0,1]")
    if (
        not isinstance(evidence.amplitude_retention_cases_at_least_090, int)
        or not 0 <= evidence.amplitude_retention_cases_at_least_090 <= EXPECTED_CASE_COUNT
    ):
        raise ValueError("amplitude retention case count is invalid")
    material = materially_strong_capacity(evidence.capacity_vs_zero)
    _valid_summary(
        evidence.capacity_vs_zero,
        CAPACITY_FAMILY_ID,
        f"{evidence.arm_id} capacity",
        f"capacity::{evidence.arm_id}::vs_zero",
    )
    if evidence.arm_id == PRIMARY_REFERENCE_ARM_ID:
        if evidence.incremental_vs_reference is not None:
            raise ValueError("C4 intensity-S2 reference must not carry a self-contrast")
        incremental = True
    else:
        if evidence.incremental_vs_reference is None:
            raise ValueError(f"missing C4 intensity-S2 contrast for {evidence.arm_id}")
        _valid_summary(
            evidence.incremental_vs_reference,
            INCREMENTAL_FAMILY_ID,
            f"{evidence.arm_id} incremental",
            f"incremental::{evidence.arm_id}::vs_{PRIMARY_REFERENCE_ARM_ID}",
        )
        incremental = materially_incremental(evidence.incremental_vs_reference)
    execution = evidence.all_work_units_complete and evidence.all_exact_certified
    amplitude = (
        evidence.amplitude_retention_median >= AMPLITUDE_RETENTION_MEDIAN_MIN
        and evidence.amplitude_retention_cases_at_least_090 >= AMPLITUDE_RETENTION_CASE_COUNT_MIN
    )
    return ArmAssessment(
        evidence.arm_id,
        material,
        incremental,
        execution,
        amplitude,
        material and incremental and execution and amplitude,
    )


@dataclass(frozen=True, slots=True)
class SelectorEvidence:
    selector_id: str
    vs_zero: PairedSummary
    vs_reference: PairedSummary
    all_choices_complete: bool
    all_choices_contract_valid: bool


@dataclass(frozen=True, slots=True)
class RegionalEvidence:
    vs_zero: tuple[PairedSummary, ...]
    risk_vs_reference: tuple[PairedSummary, ...]


def _validate_selector(evidence: SelectorEvidence) -> None:
    if not isinstance(evidence, SelectorEvidence):
        raise TypeError("selector evidence must contain SelectorEvidence values")
    if evidence.selector_id not in SELECTOR_SPECS_BY_ID:
        raise ValueError(f"unknown C5 selector evidence: {evidence.selector_id}")
    if not isinstance(evidence.all_choices_complete, bool) or not isinstance(evidence.all_choices_contract_valid, bool):
        raise TypeError("selector execution flags must be boolean")
    _valid_summary(
        evidence.vs_zero,
        SELECTOR_ZERO_FAMILY_ID,
        f"{evidence.selector_id} vs zero",
        f"selector::{evidence.selector_id}::vs_zero",
    )
    _valid_summary(
        evidence.vs_reference,
        SELECTOR_REFERENCE_FAMILY_ID,
        f"{evidence.selector_id} vs reference",
        f"selector::{evidence.selector_id}::vs_{PRIMARY_REFERENCE_ARM_ID}",
    )


def _validate_regional(evidence: RegionalEvidence) -> None:
    if not isinstance(evidence, RegionalEvidence):
        raise TypeError("regional evidence must be RegionalEvidence")
    if len(evidence.vs_zero) != 30 or len(evidence.risk_vs_reference) != len(REGIONAL_REPAIR_LABEL_IDS):
        raise ValueError("regional evidence must contain all 30 labels and both repair labels")
    for label_id, summary in zip(EVALUATION_LABEL_IDS, evidence.vs_zero, strict=True):
        _valid_summary(
            summary,
            REGIONAL_ZERO_FAMILY_ID,
            f"label {label_id} vs zero",
            f"label::{label_id}::{PRIMARY_SELECTOR_ID}::vs_zero",
        )
    for label_id, summary in zip(REGIONAL_REPAIR_LABEL_IDS, evidence.risk_vs_reference, strict=True):
        _valid_summary(
            summary,
            REGIONAL_REFERENCE_FAMILY_ID,
            f"label {label_id} vs reference",
            f"label::{label_id}::{PRIMARY_SELECTOR_ID}::vs_{PRIMARY_REFERENCE_ARM_ID}",
        )


def region_safe(evidence: RegionalEvidence) -> bool:
    _validate_regional(evidence)
    if any(summary.ci_low <= REGIONAL_ALL_LABEL_CI_LOW_MIN_STRICT for summary in evidence.vs_zero):
        return False
    return all(
        evidence.vs_zero[EVALUATION_LABEL_IDS.index(label_id)].ci_low > REGIONAL_RISK_LABEL_CI_LOW_MIN_STRICT
        for label_id in REGIONAL_RISK_LABEL_IDS
    )


def region_repaired(evidence: RegionalEvidence) -> bool:
    _validate_regional(evidence)
    return all(
        summary.mean >= REGIONAL_REPAIR_MEAN_MIN and summary.ci_low > REGIONAL_REPAIR_CI_LOW_MIN_STRICT
        for summary in evidence.risk_vs_reference
    )


def region_repair_safe(evidence: RegionalEvidence) -> bool:
    """Keep general safety and label-13 safety while allowing the 9/29 repair exception."""

    _validate_regional(evidence)
    if any(summary.ci_low <= REGIONAL_ALL_LABEL_CI_LOW_MIN_STRICT for summary in evidence.vs_zero):
        return False
    label_13 = evidence.vs_zero[EVALUATION_LABEL_IDS.index(13)]
    return label_13.ci_low > REGIONAL_RISK_LABEL_CI_LOW_MIN_STRICT


@dataclass(frozen=True, slots=True)
class BranchDecision:
    branch_id: str
    selected_arm_id: str | None
    selected_selector_id: str | None
    eligible_arm_ids: tuple[str, ...]
    material_arm_ids: tuple[str, ...]
    reason: str


def select_next_branch(
    arm_rows: Sequence[ArmEvidence],
    selector_rows: Sequence[SelectorEvidence],
    marginal_summaries: Sequence[PairedSummary],
    regional_evidence: RegionalEvidence,
    *,
    integrity_passed: bool,
) -> BranchDecision:
    if not isinstance(integrity_passed, bool):
        raise TypeError("integrity_passed must be boolean")
    if not integrity_passed:
        return BranchDecision(BRANCH_INVALID, None, None, (), (), "run-level integrity checks did not pass")
    if not isinstance(arm_rows, Sequence) or any(not isinstance(row, ArmEvidence) for row in arm_rows):
        raise TypeError("arm_rows must contain ArmEvidence values")
    if tuple(row.arm_id for row in arm_rows) != SELECTABLE_ARM_IDS:
        raise ValueError("branch selection requires all 36 arms in frozen order")
    if not isinstance(selector_rows, Sequence) or any(not isinstance(row, SelectorEvidence) for row in selector_rows):
        raise TypeError("selector_rows must contain SelectorEvidence values")
    if tuple(row.selector_id for row in selector_rows) != SELECTOR_IDS:
        raise ValueError("branch selection requires all five selectors in frozen order")
    if len(marginal_summaries) != 7:
        raise ValueError("branch selection requires all seven marginal summaries")
    for contrast_id, summary in zip(CONTRAST_IDS_BY_FAMILY[MARGINAL_FAMILY_ID], marginal_summaries, strict=True):
        _valid_summary(summary, MARGINAL_FAMILY_ID, contrast_id, contrast_id)
    for selector in selector_rows:
        _validate_selector(selector)
    _validate_regional(regional_evidence)

    if any(not row.all_choices_complete or not row.all_choices_contract_valid for row in selector_rows):
        return BranchDecision(
            BRANCH_INVALID,
            None,
            None,
            (),
            (),
            "one or more global selector outputs are incomplete or violate their contract",
        )

    assessed = tuple(assess_arm(row) for row in arm_rows)
    if any(not item.execution_complete for item in assessed):
        return BranchDecision(
            BRANCH_INVALID,
            None,
            None,
            (),
            (),
            "one or more arm outputs are incomplete or lack exact certification",
        )
    material = tuple(item.arm_id for item in assessed if item.material_capacity)
    eligible = tuple(item.arm_id for item in assessed if item.eligible)
    if not material:
        return BranchDecision(BRANCH_CLOSE_NO_CAPACITY, None, None, eligible, (), "no arm has material capacity")
    clipped_material = tuple(
        item.arm_id for item in assessed if item.material_capacity and not item.amplitude_interpretable
    )
    if clipped_material:
        return BranchDecision(
            BRANCH_CLIPPING,
            None,
            None,
            eligible,
            material,
            "a material arm did not retain its preregistered post-RMS amplitude",
        )

    if eligible:
        means = {row.arm_id: row.capacity_vs_zero.mean for row in arm_rows if row.arm_id in eligible}
        best_mean = max(means.values())
        best_arm = next(
            arm_id
            for arm_id in SELECTABLE_ARM_IDS
            if arm_id in means and best_mean - means[arm_id] <= WINNER_MEAN_TIE_TOLERANCE
        )
        marginal_by_id = dict(zip(CONTRAST_IDS_BY_FAMILY[MARGINAL_FAMILY_ID], marginal_summaries, strict=True))
        best_spec = ARM_SPECS_BY_ID[best_arm]
        positive_boundaries = []
        for at_upper_boundary, contrast_id in (
            (best_spec.stride_voxels == 4, "reach_s4_vs_s3"),
            (best_spec.post_rms_amplitude == 2.0, "amplitude_a20_vs_a10"),
            (best_spec.bias_level == 2, "bias_b2_vs_b1"),
        ):
            summary = marginal_by_id[contrast_id]
            if (
                at_upper_boundary
                and summary.mean >= FACTOR_BOUNDARY_MEAN_MIN
                and summary.ci_low > FACTOR_BOUNDARY_CI_LOW_MIN_STRICT
            ):
                positive_boundaries.append(contrast_id)
        if positive_boundaries:
            return BranchDecision(
                BRANCH_FACTOR_BOUNDARY,
                best_arm,
                None,
                eligible,
                material,
                "the best arm reaches an upper factor boundary whose adjacent marginal remains positive: "
                + ",".join(positive_boundaries),
            )

    by_selector = {row.selector_id: row for row in selector_rows}
    primary = by_selector[PRIMARY_SELECTOR_ID]
    primary_complete = primary.all_choices_complete and primary.all_choices_contract_valid
    primary_material = primary_complete and materially_strong_selector(primary.vs_zero)
    loose_material = (
        by_selector["dual_g020"].all_choices_complete
        and by_selector["dual_g020"].all_choices_contract_valid
        and materially_strong_selector(by_selector["dual_g020"].vs_zero)
    )
    if not primary_material and loose_material:
        return BranchDecision(
            BRANCH_GEOMETRY,
            None,
            PRIMARY_SELECTOR_ID,
            eligible,
            material,
            "the relaxed geometry cap retains material utility while the primary cap does not",
        )
    if not primary_material:
        return BranchDecision(
            BRANCH_UTILITY,
            None,
            PRIMARY_SELECTOR_ID,
            eligible,
            material,
            "material arm capacity exists but the primary label-free selector does not retain it",
        )

    safe = region_safe(regional_evidence)
    primary_superior = (
        primary.vs_reference.mean >= INCREMENTAL_MEAN_DICE_DELTA_MIN
        and primary.vs_reference.ci_low > INCREMENTAL_CI_LOW_DICE_DELTA_MIN_STRICT
    )
    if primary_superior and safe:
        return BranchDecision(
            BRANCH_FREEZE_SUPERIOR,
            None,
            PRIMARY_SELECTOR_ID,
            eligible,
            material,
            "the primary selector is material, superior to C4 intensity-S2, and region-safe",
        )
    if (
        primary.vs_reference.ci_low > RISK_REPAIR_AGGREGATE_CI_LOW_MIN_STRICT
        and region_repair_safe(regional_evidence)
        and region_repaired(regional_evidence)
    ):
        return BranchDecision(
            BRANCH_FREEZE_RISK_REPAIR,
            None,
            PRIMARY_SELECTOR_ID,
            eligible,
            material,
            "the primary selector is aggregate-noninferior and repairs labels 9 and 29",
        )
    if not safe:
        return BranchDecision(
            BRANCH_REGION_RISK,
            None,
            PRIMARY_SELECTOR_ID,
            eligible,
            material,
            "the primary selector is material but violates preregistered regional safeguards",
        )
    return BranchDecision(
        BRANCH_RETAIN_REFERENCE,
        PRIMARY_REFERENCE_ARM_ID,
        None,
        eligible,
        material,
        "no preregistered selector establishes superiority or the narrow risk-repair exception",
    )


assert_frozen_policy()
assert_frozen_decision_policy()
