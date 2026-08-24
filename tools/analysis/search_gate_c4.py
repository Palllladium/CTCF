from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from types import MappingProxyType
from typing import Any

import numpy as np

PROTOCOL_ID = "CTCF-SEARCH-GATE-C4-V1"
SCHEMA_VERSION = "v1"
EXPECTED_CASE_COUNT = 58
TEST_115_AUTHORIZED = False

MIND_RADIUS = 1
CANDIDATE_COUNT = 27
STANDARDIZATION_MODE = "centered_two_pass_fp32"
STANDARDIZATION_FLOOR = 1e-6
IMAGE_NORMALIZATION_STD_FLOOR = 1e-6
POSTERIOR_TEMPERATURE = 1.0
DECODER_MODE = "posterior_mean"
POST_SMOOTHING_PASSES = 1
COMMON_EVIDENCE_COLLAR = 7
WORK_EPS = 0.0011
EXACT_CLAIM_EPS = 0.001
LOCAL_CLIP_SWEEPS = 1
SCIENTIFIC_REFERENCE_ARM_ID = "mind_d2_s1"
RMS_TARGET_SOURCE_ID = "source_c3_raw_conf_post1_requested"
PRIMARY_UTILITY_ID = "COMMON_NCC7"
PRIMARY_NCC_WINDOW = 7
SUPPORT_RETENTION_MIN = 0.99
PRIMARY_NCC_IMPROVEMENT_MIN = 1e-6
BASELINE_DICE_PARITY_ATOL = 1e-8

BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 0
BOOTSTRAP_CONFIDENCE = 0.95
BOOTSTRAP_METHOD_ID = "paired_case_bootstrap_max_absolute_centered_mean_deviation"
BOOTSTRAP_QUANTILE_METHOD = "linear"
BOOTSTRAP_FAMILY_SIZE = 33
CAPACITY_MEAN_DICE_DELTA_MIN = 0.002
CAPACITY_MEDIAN_DICE_DELTA_MIN_STRICT = 0.0
CAPACITY_CI_LOW_DICE_DELTA_MIN_STRICT = 0.001
REFERENCE_MEAN_DICE_DELTA_MIN = 0.0005
REFERENCE_CI_LOW_DICE_DELTA_MIN_STRICT = 0.0
POLICY_MEAN_DICE_DELTA_MIN = 0.001
POLICY_MEDIAN_DICE_DELTA_MIN_STRICT = 0.0
POLICY_CI_LOW_DICE_DELTA_MIN_STRICT = 0.0
GEOMETRY_NONINFERIOR_TOLERANCE = 1e-6
WINNER_MEAN_TIE_TOLERANCE = 1e-6

BRANCH_ADVANCE = "ADVANCE_C4_ARM_TO_INDEPENDENT_CONFIRMATION"
BRANCH_GEOMETRY = "OPEN_C4_GEOMETRY_PROX_FOR_MATERIAL_CAPACITY"
BRANCH_UTILITY = "OPEN_C4_UTILITY_FOR_MATERIAL_CAPACITY"
BRANCH_CLOSE = "CLOSE_C4_FACTORIAL_WITHOUT_PROMOTION"


@dataclass(frozen=True, slots=True)
class DescriptorSpec:
    descriptor_id: str
    feature_family: str
    mind_radius: int | None
    mind_dilations: tuple[int, ...]
    channel_reduction: str
    fusion: str
    selectable: bool


DESCRIPTOR_SPECS = (
    DescriptorSpec("D1", "MIND_SSC", MIND_RADIUS, (1,), "mean_squared_difference", "none", True),
    DescriptorSpec("D2", "MIND_SSC", MIND_RADIUS, (2,), "mean_squared_difference", "none", True),
    DescriptorSpec("D4", "MIND_SSC", MIND_RADIUS, (4,), "mean_squared_difference", "none", True),
    DescriptorSpec(
        "F124",
        "MIND_SSC",
        MIND_RADIUS,
        (1, 2, 4),
        "mean_squared_difference",
        "equal_mean_of_separately_standardized_logits_then_centered_restandardize",
        True,
    ),
    DescriptorSpec(
        "F222",
        "MIND_SSC",
        MIND_RADIUS,
        (2, 2, 2),
        "mean_squared_difference",
        "equal_mean_of_separately_standardized_logits_then_centered_restandardize",
        False,
    ),
    DescriptorSpec(
        "INTENSITY",
        "ZSCORED_INTENSITY",
        None,
        (),
        "squared_difference",
        "none",
        False,
    ),
)
DESCRIPTOR_SPECS_BY_ID: Mapping[str, DescriptorSpec] = MappingProxyType(
    {spec.descriptor_id: spec for spec in DESCRIPTOR_SPECS}
)


def _offsets(stride: int) -> tuple[tuple[int, int, int], ...]:
    values = (-stride, 0, stride)
    return tuple((dz, dy, dx) for dz in values for dy in values for dx in values)


@dataclass(frozen=True, slots=True)
class SearchReachSpec:
    search_id: str
    offset_stride_voxels: int
    offsets_zyx: tuple[tuple[int, int, int], ...]
    pre_rms_multiplier: float


SEARCH_REACH_SPECS = (
    SearchReachSpec("S1", 1, _offsets(1), 2.0),
    SearchReachSpec("S2", 2, _offsets(2), 1.0),
)
SEARCH_REACH_SPECS_BY_ID: Mapping[str, SearchReachSpec] = MappingProxyType(
    {spec.search_id: spec for spec in SEARCH_REACH_SPECS}
)


@dataclass(frozen=True, slots=True)
class ArmSpec:
    arm_index: int
    arm_id: str
    role: str
    descriptor_id: str
    search_id: str
    standardization_mode: str
    decoder_mode: str
    post_smoothing_passes: int
    evidence_collar: int
    pre_rms_multiplier: float
    rms_target_source_id: str | None
    selectable: bool
    diagnostic_only: bool
    materialize_candidate: bool
    post_barrier_evaluation: bool


def _main_arm(index: int, arm_id: str, descriptor_id: str, search_id: str) -> ArmSpec:
    return ArmSpec(
        arm_index=index,
        arm_id=arm_id,
        role="scientific_reference" if arm_id == SCIENTIFIC_REFERENCE_ARM_ID else "scientific_candidate",
        descriptor_id=descriptor_id,
        search_id=search_id,
        standardization_mode=STANDARDIZATION_MODE,
        decoder_mode=DECODER_MODE,
        post_smoothing_passes=POST_SMOOTHING_PASSES,
        evidence_collar=COMMON_EVIDENCE_COLLAR,
        pre_rms_multiplier=SEARCH_REACH_SPECS_BY_ID[search_id].pre_rms_multiplier,
        rms_target_source_id=RMS_TARGET_SOURCE_ID,
        selectable=True,
        diagnostic_only=False,
        materialize_candidate=True,
        post_barrier_evaluation=True,
    )


MAIN_ARM_SPECS = (
    _main_arm(0, "mind_d1_s1", "D1", "S1"),
    _main_arm(1, "mind_d2_s1", "D2", "S1"),
    _main_arm(2, "mind_d4_s1", "D4", "S1"),
    _main_arm(3, "mind_f124_s1", "F124", "S1"),
    _main_arm(4, "mind_d1_s2", "D1", "S2"),
    _main_arm(5, "mind_d2_s2", "D2", "S2"),
    _main_arm(6, "mind_d4_s2", "D4", "S2"),
    _main_arm(7, "mind_f124_s2", "F124", "S2"),
)

DIAGNOSTIC_ARM_SPECS = (
    ArmSpec(
        8,
        "legacy_mind_d2_s1_collar4",
        "legacy_parity_diagnostic",
        "D2",
        "S1",
        "legacy_sequential_fp32",
        "posterior_expectation_times_confidence",
        1,
        4,
        2.0,
        None,
        False,
        True,
        False,
        False,
    ),
    ArmSpec(
        9,
        "mind_f222_s1",
        "fusion_idempotence_diagnostic",
        "F222",
        "S1",
        STANDARDIZATION_MODE,
        DECODER_MODE,
        POST_SMOOTHING_PASSES,
        COMMON_EVIDENCE_COLLAR,
        2.0,
        None,
        False,
        True,
        False,
        False,
    ),
    ArmSpec(
        10,
        "intensity_s1",
        "descriptor_specificity_diagnostic",
        "INTENSITY",
        "S1",
        STANDARDIZATION_MODE,
        DECODER_MODE,
        POST_SMOOTHING_PASSES,
        COMMON_EVIDENCE_COLLAR,
        2.0,
        RMS_TARGET_SOURCE_ID,
        False,
        True,
        True,
        True,
    ),
    ArmSpec(
        11,
        "intensity_s2",
        "descriptor_specificity_diagnostic",
        "INTENSITY",
        "S2",
        STANDARDIZATION_MODE,
        DECODER_MODE,
        POST_SMOOTHING_PASSES,
        COMMON_EVIDENCE_COLLAR,
        1.0,
        RMS_TARGET_SOURCE_ID,
        False,
        True,
        True,
        True,
    ),
)

ARM_SPECS = MAIN_ARM_SPECS + DIAGNOSTIC_ARM_SPECS
ARM_SPECS_BY_ID: Mapping[str, ArmSpec] = MappingProxyType({spec.arm_id: spec for spec in ARM_SPECS})
SELECTABLE_ARM_IDS = tuple(spec.arm_id for spec in MAIN_ARM_SPECS)
DIAGNOSTIC_ARM_IDS = tuple(spec.arm_id for spec in DIAGNOSTIC_ARM_SPECS)


@dataclass(frozen=True, slots=True)
class WorkUnitSpec:
    unit_index: int
    unit_id: str
    arm_id: str
    stage: str
    labels_accessible: bool
    materialize_candidate: bool
    post_barrier_evaluation: bool
    output: str


WORK_UNIT_SPECS = tuple(
    WorkUnitSpec(
        unit_index=spec.arm_index,
        unit_id=f"label_free_proposal::{spec.arm_id}",
        arm_id=spec.arm_id,
        stage="LABEL_FREE_DECISION",
        labels_accessible=False,
        materialize_candidate=spec.materialize_candidate,
        post_barrier_evaluation=spec.post_barrier_evaluation,
        output=(
            "materialized_fp32_candidate_and_exact_certificate"
            if spec.materialize_candidate
            else "label_free_parity_diagnostics_only"
        ),
    )
    for spec in ARM_SPECS
)


@dataclass(frozen=True, slots=True)
class ContrastSpec:
    contrast_id: str
    family: str
    candidate_arm_id: str
    reference_id: str
    required_for_promotion: bool
    post_barrier_only: bool


CAPACITY_CONTRASTS = tuple(
    ContrastSpec(
        f"capacity::{arm_id}::vs_zero_update_baseline",
        "capacity_vs_baseline",
        arm_id,
        "zero_update_baseline",
        True,
        True,
    )
    for arm_id in SELECTABLE_ARM_IDS
)
RETURNED_POLICY_CONTRASTS = tuple(
    ContrastSpec(
        f"returned_policy::{arm_id}::vs_zero_update_baseline",
        "returned_policy_vs_baseline",
        arm_id,
        "zero_update_baseline",
        True,
        True,
    )
    for arm_id in SELECTABLE_ARM_IDS
)
COMMON_REFERENCE_CONTRASTS = tuple(
    ContrastSpec(
        f"incremental::{arm_id}::vs_{SCIENTIFIC_REFERENCE_ARM_ID}",
        "incremental_vs_common_reference",
        arm_id,
        SCIENTIFIC_REFERENCE_ARM_ID,
        True,
        True,
    )
    for arm_id in SELECTABLE_ARM_IDS
    if arm_id != SCIENTIFIC_REFERENCE_ARM_ID
)
FACTORIAL_CONTRASTS = (
    *(
        ContrastSpec(
            f"descriptor::{candidate}::vs_{reference}",
            "descriptor_at_fixed_reach",
            candidate,
            reference,
            False,
            True,
        )
        for candidate, reference in (
            ("mind_d1_s1", "mind_d2_s1"),
            ("mind_d4_s1", "mind_d2_s1"),
            ("mind_f124_s1", "mind_d2_s1"),
            ("mind_d1_s2", "mind_d2_s2"),
            ("mind_d4_s2", "mind_d2_s2"),
            ("mind_f124_s2", "mind_d2_s2"),
        )
    ),
    *(
        ContrastSpec(
            f"reach::{candidate}::vs_{reference}",
            "search_reach_at_fixed_descriptor",
            candidate,
            reference,
            False,
            True,
        )
        for candidate, reference in (
            ("mind_d1_s2", "mind_d1_s1"),
            ("mind_d2_s2", "mind_d2_s1"),
            ("mind_d4_s2", "mind_d4_s1"),
            ("mind_f124_s2", "mind_f124_s1"),
        )
    ),
)
CONTRAST_SPECS = CAPACITY_CONTRASTS + RETURNED_POLICY_CONTRASTS + COMMON_REFERENCE_CONTRASTS + FACTORIAL_CONTRASTS
if len(CONTRAST_SPECS) != BOOTSTRAP_FAMILY_SIZE:
    raise RuntimeError("C4 simultaneous-bootstrap family size changed")


@dataclass(frozen=True, slots=True)
class C4Policy:
    protocol_id: str
    schema_version: str
    descriptors: tuple[DescriptorSpec, ...]
    search_reaches: tuple[SearchReachSpec, ...]
    arms: tuple[ArmSpec, ...]
    work_units: tuple[WorkUnitSpec, ...]
    contrasts: tuple[ContrastSpec, ...]
    proposal_pipeline: tuple[tuple[str, Any], ...]
    labels_and_barrier: tuple[tuple[str, Any], ...]
    thresholds: tuple[tuple[str, Any], ...]

    def to_dict(self) -> dict[str, Any]:
        return _json_compatible(asdict(self))


C4_POLICY = C4Policy(
    protocol_id=PROTOCOL_ID,
    schema_version=SCHEMA_VERSION,
    descriptors=DESCRIPTOR_SPECS,
    search_reaches=SEARCH_REACH_SPECS,
    arms=ARM_SPECS,
    work_units=WORK_UNIT_SPECS,
    contrasts=CONTRAST_SPECS,
    proposal_pipeline=(
        ("development_split_case_count", EXPECTED_CASE_COUNT),
        ("test_115_authorized", TEST_115_AUTHORIZED),
        ("candidate_offset_order", "zyx lexicographic over negative, zero, positive stride"),
        ("candidate_count", CANDIDATE_COUNT),
        ("raw_cost_dtype", "float32"),
        ("image_normalization", "independent_masked_zscore"),
        ("image_normalization_mask", "common collar7 geometry mask"),
        ("image_normalization_std_floor", IMAGE_NORMALIZATION_STD_FLOOR),
        ("candidate_standardization", STANDARDIZATION_MODE),
        ("candidate_standardization_ddof", 0),
        ("candidate_standardization_floor", STANDARDIZATION_FLOOR),
        ("fusion_input", "logits=-separately_standardized_costs"),
        ("fusion_equal_weights", True),
        ("fusion_restandardization", STANDARDIZATION_MODE),
        ("fusion_validity", "intersection; mismatched descriptor validity is an integrity failure"),
        ("posterior", "masked_softmax_over_valid_candidate_logits"),
        ("posterior_temperature", POSTERIOR_TEMPERATURE),
        ("decoder", DECODER_MODE),
        ("pre_rms_multiplier_owned_by_search_reach", True),
        ("post_smoothing_passes", POST_SMOOTHING_PASSES),
        ("common_evidence_collar", COMMON_EVIDENCE_COLLAR),
        ("descriptor_common_support", "nonempty intersection; retention is diagnostic only"),
        ("scientific_reference_arm_id", SCIENTIFIC_REFERENCE_ARM_ID),
        ("rms_target_source_id", RMS_TARGET_SOURCE_ID),
        (
            "rms_target_scope",
            "authenticated per-case source C3 raw_conf_post1 requested residual on collar7",
        ),
        (
            "rms_matching_stage",
            "after multiplier, smoothing, and collar; before local clipping",
        ),
        ("work_eps", WORK_EPS),
        ("local_clip_sweeps", LOCAL_CLIP_SWEEPS),
        ("save_reload_dtype", "float32"),
        ("exact_claim_eps", EXACT_CLAIM_EPS),
        ("exact_certificate_required_for_every_main_candidate", True),
        ("exact_certificate_required_for_every_materialized_candidate", True),
        ("primary_label_free_utility", PRIMARY_UTILITY_ID),
        ("primary_ncc_window", PRIMARY_NCC_WINDOW),
        ("primary_support_retention_min", SUPPORT_RETENTION_MIN),
        ("primary_ncc_improvement_min", PRIMARY_NCC_IMPROVEMENT_MIN),
        ("materialized_action", "ACCEPT iff exact and frozen NCC7 predicate pass; otherwise ROLLBACK"),
    ),
    labels_and_barrier=(
        ("decision_workers_have_label_paths", False),
        ("decision_workers_load_labels", False),
        ("decision_workers_have_test_115", False),
        (
            "barrier_input",
            "all materialized candidates, hashes, exact certificates, and nonmaterialized parity diagnostics",
        ),
        ("barrier_must_complete_before_labels", True),
        ("validation_labels_after_barrier", "capacity and preregistered aggregate branch only"),
        ("per_case_label_informed_acceptance", False),
        ("test_115_authorized", TEST_115_AUTHORIZED),
    ),
    thresholds=(
        ("bootstrap_resamples", BOOTSTRAP_RESAMPLES),
        ("bootstrap_seed", BOOTSTRAP_SEED),
        ("bootstrap_confidence", BOOTSTRAP_CONFIDENCE),
        ("bootstrap_method", BOOTSTRAP_METHOD_ID),
        ("bootstrap_quantile_method", BOOTSTRAP_QUANTILE_METHOD),
        ("bootstrap_simultaneous_family_size", BOOTSTRAP_FAMILY_SIZE),
        (
            "bootstrap_simultaneous_family_ids",
            tuple(spec.contrast_id for spec in CONTRAST_SPECS),
        ),
        ("baseline_dice_parity_atol", BASELINE_DICE_PARITY_ATOL),
        ("capacity_mean_dice_delta_min", CAPACITY_MEAN_DICE_DELTA_MIN),
        ("capacity_median_dice_delta", ">0"),
        ("capacity_ci_low_dice_delta", ">0.001"),
        ("incremental_vs_common_reference_mean_dice_delta_min", REFERENCE_MEAN_DICE_DELTA_MIN),
        ("incremental_vs_common_reference_ci_low_dice_delta", ">0"),
        ("returned_policy_mean_dice_delta_min", POLICY_MEAN_DICE_DELTA_MIN),
        ("returned_policy_median_dice_delta", ">0"),
        ("returned_policy_ci_low_dice_delta", ">0"),
        ("common_reference_incremental_contrast", "not_applicable"),
        (
            "geometry_noninferior_reference",
            "common mind_d2_s1 capacity candidate; zero-update deltas remain diagnostic",
        ),
        ("geometry_noninferior_tolerance", GEOMETRY_NONINFERIOR_TOLERANCE),
        ("winner_mean_tie_tolerance", WINNER_MEAN_TIE_TOLERANCE),
        ("winner_tie_break", "frozen main arm order"),
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


def canonical_policy_bytes(policy: C4Policy = C4_POLICY) -> bytes:
    encoded = json.dumps(
        policy.to_dict(),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return encoded.encode("utf-8")


def policy_sha256(policy: C4Policy = C4_POLICY) -> str:
    return hashlib.sha256(canonical_policy_bytes(policy)).hexdigest()


# Literal digest: changing any frozen scientific choice requires a deliberate update.
C4_POLICY_SHA256 = "97020963f07165a2297a08e1605db3c0f413fac2bddc76f945fdaed49e02c3b1"


def assert_frozen_policy() -> None:
    actual = policy_sha256()
    if actual != C4_POLICY_SHA256:
        raise RuntimeError(f"C4 policy hash mismatch: declared={C4_POLICY_SHA256}, actual={actual}")


@dataclass(frozen=True, slots=True)
class PairedSummary:
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


def _finite_vector(values: Sequence[float] | np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size == 0 or not np.isfinite(array).all():
        raise ValueError(f"{name} must be a non-empty finite one-dimensional sequence")
    return array


def paired_summary(
    candidate_or_differences: Sequence[float] | np.ndarray,
    baseline: Sequence[float] | np.ndarray | None = None,
) -> PairedSummary:
    candidate = _finite_vector(candidate_or_differences, "candidate_or_differences")
    if baseline is None:
        differences = candidate
    else:
        baseline_array = _finite_vector(baseline, "baseline")
        if baseline_array.shape != candidate.shape:
            raise ValueError("candidate and baseline must have the same number of paired cases")
        differences = candidate - baseline_array
    generator = np.random.default_rng(BOOTSTRAP_SEED)
    indices = generator.integers(0, differences.size, size=(BOOTSTRAP_RESAMPLES, differences.size))
    bootstrap_means = differences[indices].mean(axis=1)
    tail = (1.0 - BOOTSTRAP_CONFIDENCE) / 2.0
    ci_low, ci_high = np.quantile(bootstrap_means, (tail, 1.0 - tail))
    return PairedSummary(
        n=int(differences.size),
        mean=float(differences.mean()),
        median=float(np.median(differences)),
        ci_low=float(ci_low),
        ci_high=float(ci_high),
        improved=int((differences > 0.0).sum()),
        worsened=int((differences < 0.0).sum()),
        tied=int((differences == 0.0).sum()),
        bootstrap_resamples=BOOTSTRAP_RESAMPLES,
        bootstrap_seed=BOOTSTRAP_SEED,
        bootstrap_confidence=BOOTSTRAP_CONFIDENCE,
        bootstrap_method="pointwise_percentile_diagnostic",
        simultaneous_family_size=1,
    )


def simultaneous_paired_summaries(
    differences: Mapping[str, Sequence[float] | np.ndarray],
) -> dict[str, PairedSummary]:
    expected_ids = tuple(spec.contrast_id for spec in CONTRAST_SPECS)
    if tuple(differences) != expected_ids:
        raise ValueError("C4 simultaneous bootstrap requires all 33 preregistered contrasts in frozen order")
    matrix = np.stack([_finite_vector(differences[contrast_id], contrast_id) for contrast_id in expected_ids])
    if matrix.shape != (BOOTSTRAP_FAMILY_SIZE, EXPECTED_CASE_COUNT):
        raise ValueError("C4 simultaneous contrasts must each contain exactly 58 paired cases")
    generator = np.random.default_rng(BOOTSTRAP_SEED)
    indices = generator.integers(0, EXPECTED_CASE_COUNT, size=(BOOTSTRAP_RESAMPLES, EXPECTED_CASE_COUNT))
    observed = matrix.mean(axis=1)
    boot = matrix[:, indices].mean(axis=2)
    max_deviation = np.abs(boot - observed[:, None]).max(axis=0)
    critical = float(np.quantile(max_deviation, BOOTSTRAP_CONFIDENCE, method=BOOTSTRAP_QUANTILE_METHOD))
    output: dict[str, PairedSummary] = {}
    for index, contrast_id in enumerate(expected_ids):
        values = matrix[index]
        output[contrast_id] = PairedSummary(
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
            simultaneous_family_size=BOOTSTRAP_FAMILY_SIZE,
        )
    return output


@dataclass(frozen=True, slots=True)
class GeometryComparison:
    metric_id: str
    candidate_mean: float | None
    reference_mean: float | None
    lower_is_better: bool


@dataclass(frozen=True, slots=True)
class ArmEvidence:
    arm_id: str
    capacity_vs_baseline: PairedSummary
    incremental_vs_reference: PairedSummary | None
    policy_vs_baseline: PairedSummary
    geometry: tuple[GeometryComparison, ...]
    all_work_units_complete: bool
    all_exact_certified: bool


@dataclass(frozen=True, slots=True)
class ArmEligibility:
    arm_id: str
    material_capacity: bool
    incremental_over_reference: bool
    practical_policy: bool
    geometry_noninferior: bool
    execution_complete: bool
    eligible: bool


def _valid_summary(summary: PairedSummary, name: str) -> None:
    if not isinstance(summary, PairedSummary):
        raise TypeError(f"{name} must be a PairedSummary")
    values = (summary.mean, summary.median, summary.ci_low, summary.ci_high)
    if any(not math.isfinite(value) for value in values):
        raise ValueError(f"{name} contains a non-finite statistic")
    if summary.n != EXPECTED_CASE_COUNT:
        raise ValueError(f"{name} must contain exactly {EXPECTED_CASE_COUNT} paired cases")
    if (summary.improved + summary.worsened + summary.tied) != summary.n:
        raise ValueError(f"{name} outcome counts do not sum to n")
    frozen = (BOOTSTRAP_RESAMPLES, BOOTSTRAP_SEED, BOOTSTRAP_CONFIDENCE)
    observed = (summary.bootstrap_resamples, summary.bootstrap_seed, summary.bootstrap_confidence)
    if observed != frozen:
        raise ValueError(f"{name} does not use the frozen bootstrap contract")
    if summary.bootstrap_method != BOOTSTRAP_METHOD_ID or summary.simultaneous_family_size != BOOTSTRAP_FAMILY_SIZE:
        raise ValueError(f"{name} does not use the frozen simultaneous contrast family")


def materially_strong_capacity(summary: PairedSummary) -> bool:
    _valid_summary(summary, "capacity_vs_baseline")
    return (
        summary.mean >= CAPACITY_MEAN_DICE_DELTA_MIN
        and summary.median > CAPACITY_MEDIAN_DICE_DELTA_MIN_STRICT
        and summary.ci_low > CAPACITY_CI_LOW_DICE_DELTA_MIN_STRICT
    )


def materially_better_than_reference(summary: PairedSummary) -> bool:
    _valid_summary(summary, "incremental_vs_reference")
    return summary.mean >= REFERENCE_MEAN_DICE_DELTA_MIN and summary.ci_low > REFERENCE_CI_LOW_DICE_DELTA_MIN_STRICT


def materially_strong_policy(summary: PairedSummary) -> bool:
    _valid_summary(summary, "policy_vs_baseline")
    return (
        summary.mean >= POLICY_MEAN_DICE_DELTA_MIN
        and summary.median > POLICY_MEDIAN_DICE_DELTA_MIN_STRICT
        and summary.ci_low > POLICY_CI_LOW_DICE_DELTA_MIN_STRICT
    )


def geometry_noninferior(comparisons: Sequence[GeometryComparison]) -> bool:
    if not isinstance(comparisons, Sequence) or not comparisons:
        raise ValueError("at least one preregistered geometry comparison is required")
    for comparison in comparisons:
        if not isinstance(comparison, GeometryComparison):
            raise TypeError("geometry comparisons must contain GeometryComparison values")
        if not comparison.metric_id.strip():
            raise ValueError("geometry metric_id must be non-empty")
        if not comparison.lower_is_better:
            raise ValueError("C4 geometry comparisons must use explicitly lower-is-better metrics")
        if comparison.candidate_mean is None or comparison.reference_mean is None:
            return False
        candidate = float(comparison.candidate_mean)
        reference = float(comparison.reference_mean)
        if not math.isfinite(candidate) or not math.isfinite(reference):
            return False
        if candidate > reference + GEOMETRY_NONINFERIOR_TOLERANCE:
            return False
    return True


def assess_arm(evidence: ArmEvidence) -> ArmEligibility:
    if not isinstance(evidence, ArmEvidence):
        raise TypeError("evidence must be ArmEvidence")
    spec = ARM_SPECS_BY_ID.get(evidence.arm_id)
    if spec is None:
        raise ValueError(f"unknown C4 arm: {evidence.arm_id}")
    if not spec.selectable or spec.diagnostic_only:
        raise ValueError(f"diagnostic arm cannot enter branch selection: {evidence.arm_id}")
    if not isinstance(evidence.all_work_units_complete, bool) or not isinstance(evidence.all_exact_certified, bool):
        raise TypeError("execution flags must be boolean")
    material = materially_strong_capacity(evidence.capacity_vs_baseline)
    if evidence.arm_id == SCIENTIFIC_REFERENCE_ARM_ID:
        if evidence.incremental_vs_reference is not None:
            raise ValueError("the common reference must not carry a self-contrast")
        incremental = True
    else:
        if evidence.incremental_vs_reference is None:
            raise ValueError(f"missing common-reference contrast for {evidence.arm_id}")
        incremental = materially_better_than_reference(evidence.incremental_vs_reference)
    policy_ok = materially_strong_policy(evidence.policy_vs_baseline)
    geometry_ok = geometry_noninferior(evidence.geometry)
    execution_ok = evidence.all_work_units_complete and evidence.all_exact_certified
    return ArmEligibility(
        arm_id=evidence.arm_id,
        material_capacity=material,
        incremental_over_reference=incremental,
        practical_policy=policy_ok,
        geometry_noninferior=geometry_ok,
        execution_complete=execution_ok,
        eligible=material and incremental and policy_ok and geometry_ok and execution_ok,
    )


@dataclass(frozen=True, slots=True)
class BranchDecision:
    branch_id: str
    selected_arm_id: str | None
    eligible_arm_ids: tuple[str, ...]
    material_arm_ids: tuple[str, ...]
    incremental_arm_ids: tuple[str, ...]
    practical_policy_arm_ids: tuple[str, ...]
    geometry_noninferior_arm_ids: tuple[str, ...]
    reason: str


def select_next_branch(evidence_rows: Sequence[ArmEvidence]) -> BranchDecision:
    if not isinstance(evidence_rows, Sequence):
        raise TypeError("evidence_rows must be a sequence")
    observed_ids = tuple(row.arm_id for row in evidence_rows if isinstance(row, ArmEvidence))
    if len(observed_ids) != len(evidence_rows):
        raise TypeError("evidence_rows must contain only ArmEvidence values")
    if observed_ids != SELECTABLE_ARM_IDS:
        raise ValueError("branch selection requires all eight main arms in frozen order and no diagnostics")
    assessed = tuple(assess_arm(row) for row in evidence_rows)
    eligible = tuple(item.arm_id for item in assessed if item.eligible)
    material = tuple(item.arm_id for item in assessed if item.material_capacity)
    incremental = tuple(item.arm_id for item in assessed if item.incremental_over_reference)
    policy_ok = tuple(item.arm_id for item in assessed if item.practical_policy)
    geometry_ok = tuple(item.arm_id for item in assessed if item.geometry_noninferior)
    if not eligible:
        science_ready = tuple(
            item.arm_id
            for item in assessed
            if item.material_capacity and item.incremental_over_reference and item.execution_complete
        )
        geometry_ready = tuple(arm_id for arm_id in science_ready if arm_id in geometry_ok)
        if science_ready and not geometry_ready:
            branch = BRANCH_GEOMETRY
            reason = "material incremental capacity exists, but every such arm pays a preregistered geometry price"
        elif geometry_ready and not any(arm_id in policy_ok for arm_id in geometry_ready):
            branch = BRANCH_UTILITY
            reason = "material geometry-noninferior capacity exists, but the frozen NCC7 return policy loses it"
        else:
            branch = BRANCH_CLOSE
            reason = "no main arm passes the frozen capacity and reference requirements"
        return BranchDecision(
            branch,
            None,
            (),
            material,
            incremental,
            policy_ok,
            geometry_ok,
            reason,
        )

    scores = {row.arm_id: row.capacity_vs_baseline.mean for row in evidence_rows if row.arm_id in eligible}
    best = max(scores.values())
    tied = tuple(
        arm_id
        for arm_id in SELECTABLE_ARM_IDS
        if arm_id in scores and best - scores[arm_id] <= WINNER_MEAN_TIE_TOLERANCE
    )
    return BranchDecision(
        BRANCH_ADVANCE,
        tied[0],
        eligible,
        material,
        incremental,
        policy_ok,
        geometry_ok,
        "at least one main arm passes every frozen promotion requirement",
    )


assert_frozen_policy()
