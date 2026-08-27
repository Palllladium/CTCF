from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from types import MappingProxyType
from typing import Any

import numpy as np

from tools.analysis.search_gate_metrics import (
    DETERMINANT_NAMES,
    DETJ_DIAGNOSTICS,
    DIGITAL_DECOMPOSITION,
    MATHEMATICAL_SDLOGJ_CROP2,
    METRIC_SPECS,
)

PROTOCOL_ID = "CTCF-SEARCH-GATE-C5B-V3"
SCHEMA_VERSION = "v3"
EXPECTED_CASE_COUNT = 58
DEVELOPMENT_DATASET_ID = "IXI_VALIDATION_58"
TEST_115_AUTHORIZED = False

WORK_EPS = 0.0011
EXACT_CLAIM_EPS = 0.001
CLIP_CURRENT_FAST_BOUND_ROLE = "HARD_PRECONDITION_AT_WORK_EPS"
CLIP_OUTPUT_FAST_BOUND_ROLE = "FINITE_DIAGNOSTIC_EXACT_SAVED_FP32_CERTIFICATE_IS_AUTHORITATIVE"
DESCRIPTOR_ID = "ZSCORED_INTENSITY"
IMAGE_NORMALIZATION_MODE = "independent_masked_zscore"
IMAGE_NORMALIZATION_STD_FLOOR = 1e-6
STANDARDIZATION_MODE = "centered_two_pass_fp32"
STANDARDIZATION_FLOOR = 1e-6
DECODER_MODE = "posterior_mean"
POSTERIOR_TEMPERATURE = 1.0
STRIDE_VOXELS = 4
CANDIDATE_OFFSETS_ZYX = tuple(
    (dz, dy, dx)
    for dz in (-STRIDE_VOXELS, 0, STRIDE_VOXELS)
    for dy in (-STRIDE_VOXELS, 0, STRIDE_VOXELS)
    for dx in (-STRIDE_VOXELS, 0, STRIDE_VOXELS)
)
CANDIDATE_COUNT = len(CANDIDATE_OFFSETS_ZYX)
CENTRE_BETA = 0.0
PRE_RMS_MULTIPLIER = 1.0
POST_SMOOTHING_PASSES = 1
COMMON_EVIDENCE_COLLAR = 7
RMS_TARGET_SOURCE_ID = "source_c3_raw_conf_post1_requested"
AMPLITUDE_STAGE = "after_rms_match_before_local_clip"
BRIDGE_CONSTRUCTION = "recompute_from_authenticated_preclip_s4_direction"
POSTCLIP_INTERPOLATION_ALLOWED = False
OBSERVED_FOLD_COUNT_DEFINITION = "digital_corner_union_violation_witnesses"
CENTRAL_DETJ_INVALID_REQUIRED_ZERO = True
DIGITAL_CORNER_UNION_REQUIRED_ZERO = True
DIGITAL_TEN_UNION_ROLE = "DIAGNOSTIC_ONLY_NOT_A_TRILINEAR_FOLD_WITNESS"
SDLOGJ_METRIC_ID = MATHEMATICAL_SDLOGJ_CROP2
DICE_AGGREGATION = "unweighted_macro_mean_over_fixed_30_ixi_labels"
DICE_WARP_INTERPOLATION = "nearest"


@dataclass(frozen=True, slots=True)
class C5BGeometryDiagnostics:
    central_invalid_count: int
    central_detj_min: float
    corner_union_violation_count: int
    jstar_union_violation_count: int
    jstar_union_violation_fraction: float
    digital_ten_union_violation_count: int
    digital_ten_union_violation_fraction: float
    digital_ten_union_percent: float


@dataclass(frozen=True, slots=True)
class C5BClipOperatorDiagnostics:
    current_fast_cert_bound: float
    output_fast_cert_bound: float


def validate_c5b_clip_operator(
    operator: Any,
    *,
    expected_sweeps: int,
    label: str,
) -> C5BClipOperatorDiagnostics:
    """Validate operator identity while leaving the post-cast fast bound diagnostic-only."""
    if not isinstance(operator, Mapping):
        raise RuntimeError(f"C5b clip operator is absent: {label}")
    if (
        operator.get("operator") != "CERTIFIED_LOCAL_CLIP"
        or operator.get("sweeps") != expected_sweeps
        or not math.isclose(float(operator.get("work_eps", math.nan)), WORK_EPS, rel_tol=0.0, abs_tol=0.0)
    ):
        raise RuntimeError(f"C5b clip operator identity changed: {label}")
    current_bound = _geometry_finite(
        operator.get("current_fast_cert_bound"),
        f"C5b current fast certificate {label}",
    )
    output_bound = _geometry_finite(
        operator.get("output_fast_cert_bound"),
        f"C5b output fast certificate {label}",
    )
    if current_bound < WORK_EPS:
        raise RuntimeError(f"C5b current fast certificate is below work_eps: {label}")
    return C5BClipOperatorDiagnostics(current_bound, output_bound)


def _geometry_finite(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise RuntimeError(f"{label} must be a finite real scalar")
    return float(value)


def _geometry_count(value: Any, label: str) -> int:
    observed = _geometry_finite(value, label)
    if observed < 0.0 or not observed.is_integer():
        raise RuntimeError(f"{label} must be a non-negative integer")
    return int(observed)


def _fraction_count(value: Any, voxels: int, label: str) -> int:
    fraction = _geometry_finite(value, label)
    if not 0.0 <= fraction <= 1.0:
        raise RuntimeError(f"{label} must lie in [0,1]")
    count = round(fraction * voxels)
    if not math.isclose(fraction, count / voxels, rel_tol=0.0, abs_tol=1e-12):
        raise RuntimeError(f"{label} is not an exact voxel fraction")
    return count


def validate_c5b_geometry_bundle(geometry: Any, label: str) -> C5BGeometryDiagnostics:
    """Validate exact-field geometry while keeping J1*/J2* diagnostic-only."""
    if not isinstance(geometry, Mapping) or set(geometry) != set(METRIC_SPECS):
        raise RuntimeError(f"C5b geometry inventory changed: {label}")
    for metric_id in METRIC_SPECS:
        row = geometry[metric_id]
        if not isinstance(row, Mapping) or row.get("metric_id") != metric_id or row.get("status") != "OK":
            raise RuntimeError(f"C5b geometry metric identity changed: {label}/{metric_id}")
        if not isinstance(row.get("components"), Mapping):
            raise RuntimeError(f"C5b geometry components are absent: {label}/{metric_id}")
        if metric_id == DETJ_DIAGNOSTICS:
            if row.get("value") is not None:
                raise RuntimeError(f"C5b detJ diagnostics unexpectedly has a scalar value: {label}")
        else:
            _geometry_finite(row.get("value"), f"C5b geometry value {label}/{metric_id}")

    detj = geometry[DETJ_DIAGNOSTICS]["components"]
    detj_voxels = _geometry_count(detj.get("voxels"), f"C5b detJ voxels {label}")
    finite_count = _geometry_count(detj.get("finite_count"), f"C5b detJ finite count {label}")
    nonfinite = _geometry_count(detj.get("nonfinite_count"), f"C5b detJ nonfinite count {label}")
    nonpositive = _geometry_count(detj.get("nonpositive_count"), f"C5b detJ nonpositive count {label}")
    invalid = _geometry_count(detj.get("invalid_count"), f"C5b detJ invalid count {label}")
    detj_min = _geometry_finite(detj.get("detj_min"), f"C5b minimum detJ {label}")
    if (
        detj_voxels <= 0
        or finite_count + nonfinite != detj_voxels
        or nonpositive > finite_count
        or invalid != nonfinite + nonpositive
        or (CENTRAL_DETJ_INVALID_REQUIRED_ZERO and invalid != 0)
        or detj_min <= 0.0
    ):
        raise RuntimeError(f"C5b central Jacobian validity failed: {label}")
    for count_name, count in (
        ("nonfinite", nonfinite),
        ("nonpositive", nonpositive),
        ("invalid", invalid),
    ):
        fraction = _geometry_finite(detj.get(f"{count_name}_fraction"), f"C5b detJ {count_name} fraction {label}")
        if not math.isclose(fraction, count / detj_voxels, rel_tol=0.0, abs_tol=1e-12):
            raise RuntimeError(f"C5b detJ count/fraction arithmetic changed: {label}/{count_name}")
    for key, value in detj.items():
        observed = _geometry_finite(value, f"C5b detJ component {label}/{key}")
        if key.endswith("_fraction") and not 0.0 <= observed <= 1.0:
            raise RuntimeError(f"C5b detJ fraction is outside [0,1]: {label}/{key}")

    digital_row = geometry[DIGITAL_DECOMPOSITION]
    digital = digital_row["components"]
    expected_components = {
        "voxels",
        "corner_union_violation_fraction",
        "jstar_union_violation_fraction",
        "union_violation_count",
        "union_violation_fraction",
        "sum_of_component_fractions",
        *(
            key
            for name in DETERMINANT_NAMES
            for key in (
                f"{name}_min",
                f"{name}_nonfinite_count",
                f"{name}_violation_count",
                f"{name}_violation_fraction",
            )
        ),
    }
    if set(digital) != expected_components:
        raise RuntimeError(f"C5b digital component inventory changed: {label}")
    voxels = _geometry_count(digital["voxels"], f"C5b digital voxels {label}")
    union_count = _geometry_count(digital["union_violation_count"], f"C5b digital union count {label}")
    if voxels <= 0 or union_count > voxels:
        raise RuntimeError(f"C5b digital voxel accounting failed: {label}")
    for key, value in digital.items():
        observed = _geometry_finite(value, f"C5b digital component {label}/{key}")
        if key.endswith("_count"):
            count = _geometry_count(value, f"C5b digital count {label}/{key}")
            if count > voxels:
                raise RuntimeError(f"C5b digital count exceeds support: {label}/{key}")
        elif key.endswith("_fraction") and not 0.0 <= observed <= 1.0:
            raise RuntimeError(f"C5b digital fraction is outside [0,1]: {label}/{key}")
    component_counts: dict[str, int] = {}
    for name in DETERMINANT_NAMES:
        count = _geometry_count(digital[f"{name}_violation_count"], f"C5b digital count {label}/{name}")
        nonfinite_count = _geometry_count(
            digital[f"{name}_nonfinite_count"], f"C5b digital nonfinite count {label}/{name}"
        )
        fraction = _geometry_finite(digital[f"{name}_violation_fraction"], f"C5b digital fraction {label}/{name}")
        determinant_min = _geometry_finite(digital[f"{name}_min"], f"C5b digital minimum {label}/{name}")
        if (
            nonfinite_count > count
            or (count == 0 and determinant_min <= 0.0)
            or not math.isclose(fraction, count / voxels, rel_tol=0.0, abs_tol=1e-12)
        ):
            raise RuntimeError(f"C5b digital count/fraction arithmetic changed: {label}/{name}")
        component_counts[name] = count
    union_fraction = _geometry_finite(digital["union_violation_fraction"], f"C5b digital union fraction {label}")
    if not math.isclose(union_fraction, union_count / voxels, rel_tol=0.0, abs_tol=1e-12):
        raise RuntimeError(f"C5b digital union count/fraction arithmetic changed: {label}")
    corner_count = _fraction_count(
        digital["corner_union_violation_fraction"], voxels, f"C5b digital corner-union fraction {label}"
    )
    jstar_count = _fraction_count(
        digital["jstar_union_violation_fraction"], voxels, f"C5b digital Jstar-union fraction {label}"
    )
    corner_component_counts = [component_counts[name] for name in DETERMINANT_NAMES[:8]]
    jstar_component_counts = [component_counts[name] for name in DETERMINANT_NAMES[8:]]
    if not max(corner_component_counts) <= corner_count <= sum(corner_component_counts):
        raise RuntimeError(f"C5b digital corner-union decomposition changed: {label}")
    if not max(jstar_component_counts) <= jstar_count <= sum(jstar_component_counts):
        raise RuntimeError(f"C5b digital Jstar-union decomposition changed: {label}")
    if not max(corner_count, jstar_count) <= union_count <= corner_count + jstar_count:
        raise RuntimeError(f"C5b digital union decomposition changed: {label}")
    component_fraction_sum = math.fsum(
        _geometry_finite(digital[f"{name}_violation_fraction"], f"C5b digital fraction {label}/{name}")
        for name in DETERMINANT_NAMES
    )
    if not math.isclose(
        _geometry_finite(digital["sum_of_component_fractions"], f"C5b digital component sum {label}"),
        component_fraction_sum,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise RuntimeError(f"C5b digital component sum changed: {label}")
    digital_percent = _geometry_finite(digital_row["value"], f"C5b digital percentage {label}")
    if not math.isclose(digital_percent, 100.0 * union_fraction, rel_tol=0.0, abs_tol=1e-12):
        raise RuntimeError(f"C5b digital percentage arithmetic changed: {label}")
    if DIGITAL_CORNER_UNION_REQUIRED_ZERO and corner_count != 0:
        raise RuntimeError(f"C5b corner determinant exactness failed: {label}")
    return C5BGeometryDiagnostics(
        central_invalid_count=invalid,
        central_detj_min=detj_min,
        corner_union_violation_count=corner_count,
        jstar_union_violation_count=jstar_count,
        jstar_union_violation_fraction=float(digital["jstar_union_violation_fraction"]),
        digital_ten_union_violation_count=union_count,
        digital_ten_union_violation_fraction=union_fraction,
        digital_ten_union_percent=digital_percent,
    )


BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 0
BOOTSTRAP_CONFIDENCE = 0.95
BOOTSTRAP_METHOD_ID = "paired_case_bootstrap_max_absolute_centered_mean_deviation"
BOOTSTRAP_QUANTILE_METHOD = "linear"

AMPLITUDE_RETENTION_MEDIAN_MIN = 0.95
AMPLITUDE_RETENTION_CASE_MIN = 0.90
AMPLITUDE_RETENTION_CASE_COUNT_MIN = 52
DICE_VS_REFERENCE_MEAN_MIN = 0.0005
DICE_VS_REFERENCE_MEDIAN_MIN_STRICT = 0.0
DICE_VS_REFERENCE_CI_LOW_MIN_STRICT = 0.0
SDLOGJ_VS_REFERENCE_CI_HIGH_MAX = 0.005
REGIONAL_ALL_LABEL_CI_LOW_MIN_STRICT = -0.005
REGIONAL_RISK_LABEL_CI_LOW_MIN_STRICT = -0.002
REGIONAL_RISK_LABEL_IDS = (9, 29, 13)
REGIONAL_REPAIR_LABEL_IDS = (9, 29)
REGIONAL_REPAIR_MEAN_MIN = 0.002
REGIONAL_REPAIR_CI_LOW_MIN_STRICT = 0.0
RISK_REPAIR_AGGREGATE_CI_LOW_MIN_STRICT = -0.001
WINNER_MEAN_TIE_TOLERANCE = 1e-6
WINNER_TIE_BREAK_ORDER = (
    "maximum_dice_vs_reference_mean",
    "minimum_sdlogj_vs_reference_mean_within_dice_tolerance",
    "minimum_post_rms_amplitude_within_dice_tolerance",
    "frozen_selectable_arm_order_within_dice_tolerance",
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

REFERENCE_ARM_ID = "c4_int_s2_a10_b0"
ANCHOR_ARM_IDS = (
    REFERENCE_ARM_ID,
    "c5_int_s4_a10_b0_w1",
    "c5_int_s4_a20_b0_w1",
)
SELECTABLE_ARM_IDS = (
    "int_s4_a125_b0_w1",
    "int_s4_a150_b0_w1",
    "int_s4_a175_b0_w1",
)
DIAGNOSTIC_ARM_ID = "int_s4_a200_b0_w2"

BRANCH_INVALID = "INVALID_C5B_EVIDENCE"
BRANCH_CLOSE_CLIP = "CLOSE_C5B_NO_INTERPRETABLE_BRIDGE_OPEN_TRUE_PYRAMID"
BRANCH_CLOSE_NO_SUPERIORITY = "CLOSE_C5B_NO_DICE_SUPERIORITY_OPEN_TRUE_PYRAMID"
BRANCH_CLOSE_GEOMETRY = "CLOSE_C5B_GEOMETRY_LIMIT_OPEN_TRUE_PYRAMID"
BRANCH_CLOSE_REGIONAL = "CLOSE_C5B_REGIONAL_RISK_OPEN_TRUE_PYRAMID"
BRANCH_FREEZE_SUPERIOR = "FREEZE_C5B_SUPERIOR_FOR_INDEPENDENT_CONFIRMATION"
BRANCH_FREEZE_RISK_REPAIR = "FREEZE_C5B_RISK_REPAIR_FOR_INDEPENDENT_CONFIRMATION"


@dataclass(frozen=True, slots=True)
class ArmSpec:
    arm_index: int
    arm_id: str
    role: str
    source_arm_id: str | None
    stride_voxels: int
    post_rms_amplitude: float
    centre_beta: float
    local_clip_sweeps: int
    recompute_preclip_direction: bool
    selectable: bool


ARM_SPECS = (
    ArmSpec(0, REFERENCE_ARM_ID, "ANCHOR", "int_s2_a10_b0", 2, 1.0, 0.0, 1, False, False),
    ArmSpec(1, ANCHOR_ARM_IDS[1], "ANCHOR", "int_s4_a10_b0", 4, 1.0, 0.0, 1, True, False),
    ArmSpec(2, ANCHOR_ARM_IDS[2], "ANCHOR", "int_s4_a20_b0", 4, 2.0, 0.0, 1, True, False),
    ArmSpec(3, SELECTABLE_ARM_IDS[0], "SELECTABLE", None, 4, 1.25, 0.0, 1, True, True),
    ArmSpec(4, SELECTABLE_ARM_IDS[1], "SELECTABLE", None, 4, 1.50, 0.0, 1, True, True),
    ArmSpec(5, SELECTABLE_ARM_IDS[2], "SELECTABLE", None, 4, 1.75, 0.0, 1, True, True),
    ArmSpec(6, DIAGNOSTIC_ARM_ID, "DIAGNOSTIC", None, 4, 2.0, 0.0, 2, True, False),
)
ARM_SPECS_BY_ID: Mapping[str, ArmSpec] = MappingProxyType({spec.arm_id: spec for spec in ARM_SPECS})

DICE_VS_REFERENCE_FAMILY_ID = "c5b_selectable_dice_vs_c4"
SDLOGJ_VS_REFERENCE_FAMILY_ID = "c5b_selectable_sdlogj_vs_c4"
REGIONAL_ZERO_FAMILY_ID = "c5b_all_selectable_all_labels_vs_zero"
REGIONAL_REPAIR_FAMILY_ID = "c5b_all_selectable_risk_repair_vs_c4"


def regional_zero_family_id(arm_id: str) -> str:
    if arm_id not in SELECTABLE_ARM_IDS:
        raise ValueError(f"arm is not a selectable C5b bridge arm: {arm_id}")
    return REGIONAL_ZERO_FAMILY_ID


def regional_repair_family_id(arm_id: str) -> str:
    if arm_id not in SELECTABLE_ARM_IDS:
        raise ValueError(f"arm is not a selectable C5b bridge arm: {arm_id}")
    return REGIONAL_REPAIR_FAMILY_ID


def _dice_contrast_id(arm_id: str) -> str:
    return f"dice::{arm_id}::vs_{REFERENCE_ARM_ID}"


def _sdlogj_contrast_id(arm_id: str) -> str:
    return f"sdlogj::{arm_id}::vs_{REFERENCE_ARM_ID}"


def _regional_zero_contrast_id(arm_id: str, label_id: int) -> str:
    return f"label::{label_id}::{arm_id}::vs_zero"


def _regional_repair_contrast_id(arm_id: str, label_id: int) -> str:
    return f"label::{label_id}::{arm_id}::vs_{REFERENCE_ARM_ID}"


def _build_contrast_ids() -> Mapping[str, tuple[str, ...]]:
    rows: dict[str, tuple[str, ...]] = {
        DICE_VS_REFERENCE_FAMILY_ID: tuple(_dice_contrast_id(arm_id) for arm_id in SELECTABLE_ARM_IDS),
        SDLOGJ_VS_REFERENCE_FAMILY_ID: tuple(_sdlogj_contrast_id(arm_id) for arm_id in SELECTABLE_ARM_IDS),
    }
    rows[REGIONAL_ZERO_FAMILY_ID] = tuple(
        _regional_zero_contrast_id(arm_id, label_id)
        for arm_id in SELECTABLE_ARM_IDS
        for label_id in EVALUATION_LABEL_IDS
    )
    rows[REGIONAL_REPAIR_FAMILY_ID] = tuple(
        _regional_repair_contrast_id(arm_id, label_id)
        for arm_id in SELECTABLE_ARM_IDS
        for label_id in REGIONAL_REPAIR_LABEL_IDS
    )
    return MappingProxyType(rows)


CONTRAST_IDS_BY_FAMILY = _build_contrast_ids()
BOOTSTRAP_FAMILY_SIZES: Mapping[str, int] = MappingProxyType(
    {family_id: len(contrast_ids) for family_id, contrast_ids in CONTRAST_IDS_BY_FAMILY.items()}
)


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
    differences_by_contrast: Mapping[str, Sequence[float] | np.ndarray],
) -> tuple[PairedSummary, ...]:
    expected_ids = CONTRAST_IDS_BY_FAMILY.get(family_id)
    if expected_ids is None:
        raise ValueError(f"unknown C5b inference family: {family_id}")
    if tuple(differences_by_contrast) != expected_ids:
        raise ValueError(f"C5b {family_id} bootstrap requires every contrast in frozen order")
    arrays = tuple(np.asarray(differences_by_contrast[contrast_id], dtype=np.float64) for contrast_id in expected_ids)
    if any(array.shape != (EXPECTED_CASE_COUNT,) for array in arrays):
        raise ValueError("C5b simultaneous contrasts must each contain exactly 58 paired cases")
    if any(not np.isfinite(array).all() for array in arrays):
        raise ValueError("C5b simultaneous contrasts must be finite")

    matrix = np.stack(arrays, axis=0)
    means = matrix.mean(axis=1)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    indices = rng.integers(0, EXPECTED_CASE_COUNT, size=(BOOTSTRAP_RESAMPLES, EXPECTED_CASE_COUNT))
    bootstrap_means = matrix[:, indices].mean(axis=2)
    max_deviation = np.max(np.abs(bootstrap_means - means[:, None]), axis=0)
    critical = float(np.quantile(max_deviation, BOOTSTRAP_CONFIDENCE, method=BOOTSTRAP_QUANTILE_METHOD))

    summaries = []
    for contrast_id, values, mean in zip(expected_ids, arrays, means, strict=True):
        summaries.append(
            PairedSummary(
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
                bootstrap_resamples=BOOTSTRAP_RESAMPLES,
                bootstrap_seed=BOOTSTRAP_SEED,
                bootstrap_confidence=BOOTSTRAP_CONFIDENCE,
                bootstrap_method=BOOTSTRAP_METHOD_ID,
                simultaneous_family_size=len(expected_ids),
            )
        )
    return tuple(summaries)


@dataclass(frozen=True, slots=True)
class RegionalEvidence:
    vs_zero: tuple[PairedSummary, ...]
    repair_vs_reference: tuple[PairedSummary, ...]


@dataclass(frozen=True, slots=True)
class ArmEvidence:
    arm_id: str
    dice_vs_reference: PairedSummary
    sdlogj_vs_reference: PairedSummary
    regional: RegionalEvidence
    all_work_units_complete: bool
    all_exact_certified: bool
    observed_fold_count: int
    amplitude_retention_median: float
    amplitude_retention_cases_at_least_090: int


@dataclass(frozen=True, slots=True)
class ArmAssessment:
    arm_id: str
    integrity_passed: bool
    amplitude_interpretable: bool
    dice_superior: bool
    geometry_noninferior: bool
    region_safe: bool
    risk_repair: bool
    superior_success: bool
    risk_repair_success: bool


@dataclass(frozen=True, slots=True)
class BranchDecision:
    branch_id: str
    selected_arm_id: str | None
    interpretable_arm_ids: tuple[str, ...]
    dice_superior_arm_ids: tuple[str, ...]
    reason: str


def _require_selectable_arm(arm_id: str) -> ArmSpec:
    spec = ARM_SPECS_BY_ID.get(arm_id)
    if spec is None or not spec.selectable:
        raise ValueError(f"arm is not a selectable C5b bridge arm: {arm_id}")
    return spec


def _validate_summary(summary: PairedSummary, family_id: str, contrast_id: str) -> None:
    if not isinstance(summary, PairedSummary):
        raise TypeError("C5b inference evidence must contain PairedSummary values")
    if summary.family_id != family_id or summary.contrast_id != contrast_id:
        raise ValueError(f"invalid C5b summary identity: {contrast_id}")
    if summary.n != EXPECTED_CASE_COUNT:
        raise ValueError(f"C5b summary does not contain 58 cases: {contrast_id}")
    if not all(math.isfinite(value) for value in (summary.mean, summary.median, summary.ci_low, summary.ci_high)):
        raise ValueError(f"C5b summary contains a non-finite value: {contrast_id}")
    if summary.ci_low > summary.ci_high:
        raise ValueError(f"C5b summary interval is reversed: {contrast_id}")
    if summary.improved + summary.worsened + summary.tied != EXPECTED_CASE_COUNT:
        raise ValueError(f"C5b summary case accounting is invalid: {contrast_id}")
    if (
        summary.bootstrap_resamples != BOOTSTRAP_RESAMPLES
        or summary.bootstrap_seed != BOOTSTRAP_SEED
        or summary.bootstrap_confidence != BOOTSTRAP_CONFIDENCE
        or summary.bootstrap_method != BOOTSTRAP_METHOD_ID
        or summary.simultaneous_family_size != BOOTSTRAP_FAMILY_SIZES[family_id]
    ):
        raise ValueError(f"C5b summary does not use the frozen simultaneous inference contract: {contrast_id}")


def _validate_regional(arm_id: str, evidence: RegionalEvidence) -> None:
    if not isinstance(evidence, RegionalEvidence):
        raise TypeError("C5b regional evidence must be RegionalEvidence")
    if len(evidence.vs_zero) != len(EVALUATION_LABEL_IDS):
        raise ValueError("C5b regional evidence must contain all 30 labels")
    if len(evidence.repair_vs_reference) != len(REGIONAL_REPAIR_LABEL_IDS):
        raise ValueError("C5b repair evidence must contain labels 9 and 29")
    zero_family = regional_zero_family_id(arm_id)
    for label_id, summary in zip(EVALUATION_LABEL_IDS, evidence.vs_zero, strict=True):
        _validate_summary(summary, zero_family, _regional_zero_contrast_id(arm_id, label_id))
    repair_family = regional_repair_family_id(arm_id)
    for label_id, summary in zip(REGIONAL_REPAIR_LABEL_IDS, evidence.repair_vs_reference, strict=True):
        _validate_summary(summary, repair_family, _regional_repair_contrast_id(arm_id, label_id))


def _region_safe(evidence: RegionalEvidence) -> bool:
    if any(row.ci_low <= REGIONAL_ALL_LABEL_CI_LOW_MIN_STRICT for row in evidence.vs_zero):
        return False
    by_label = dict(zip(EVALUATION_LABEL_IDS, evidence.vs_zero, strict=True))
    return all(
        by_label[label_id].ci_low > REGIONAL_RISK_LABEL_CI_LOW_MIN_STRICT for label_id in REGIONAL_RISK_LABEL_IDS
    )


def _risk_repair_safe(evidence: RegionalEvidence) -> bool:
    if any(row.ci_low <= REGIONAL_ALL_LABEL_CI_LOW_MIN_STRICT for row in evidence.vs_zero):
        return False
    by_label = dict(zip(EVALUATION_LABEL_IDS, evidence.vs_zero, strict=True))
    if by_label[13].ci_low <= REGIONAL_RISK_LABEL_CI_LOW_MIN_STRICT:
        return False
    return all(
        row.mean >= REGIONAL_REPAIR_MEAN_MIN and row.ci_low > REGIONAL_REPAIR_CI_LOW_MIN_STRICT
        for row in evidence.repair_vs_reference
    )


def assess_arm(evidence: ArmEvidence) -> ArmAssessment:
    if not isinstance(evidence, ArmEvidence):
        raise TypeError("evidence must be ArmEvidence")
    _require_selectable_arm(evidence.arm_id)
    _validate_summary(
        evidence.dice_vs_reference,
        DICE_VS_REFERENCE_FAMILY_ID,
        _dice_contrast_id(evidence.arm_id),
    )
    _validate_summary(
        evidence.sdlogj_vs_reference,
        SDLOGJ_VS_REFERENCE_FAMILY_ID,
        _sdlogj_contrast_id(evidence.arm_id),
    )
    _validate_regional(evidence.arm_id, evidence.regional)
    if not isinstance(evidence.all_work_units_complete, bool) or not isinstance(evidence.all_exact_certified, bool):
        raise TypeError("C5b execution flags must be boolean")
    if not isinstance(evidence.observed_fold_count, int) or evidence.observed_fold_count < 0:
        raise ValueError("C5b observed fold count must be a non-negative integer")
    if not math.isfinite(evidence.amplitude_retention_median) or not 0.0 <= evidence.amplitude_retention_median <= 1.0:
        raise ValueError("C5b amplitude retention median must lie in [0,1]")
    if (
        not isinstance(evidence.amplitude_retention_cases_at_least_090, int)
        or not 0 <= evidence.amplitude_retention_cases_at_least_090 <= EXPECTED_CASE_COUNT
    ):
        raise ValueError("C5b amplitude retention case count is invalid")

    integrity = evidence.all_work_units_complete and evidence.all_exact_certified and evidence.observed_fold_count == 0
    amplitude = (
        evidence.amplitude_retention_median >= AMPLITUDE_RETENTION_MEDIAN_MIN
        and evidence.amplitude_retention_cases_at_least_090 >= AMPLITUDE_RETENTION_CASE_COUNT_MIN
    )
    dice = (
        evidence.dice_vs_reference.mean >= DICE_VS_REFERENCE_MEAN_MIN
        and evidence.dice_vs_reference.median > DICE_VS_REFERENCE_MEDIAN_MIN_STRICT
        and evidence.dice_vs_reference.ci_low > DICE_VS_REFERENCE_CI_LOW_MIN_STRICT
    )
    geometry = evidence.sdlogj_vs_reference.ci_high <= SDLOGJ_VS_REFERENCE_CI_HIGH_MAX
    regional = _region_safe(evidence.regional)
    repair = evidence.dice_vs_reference.ci_low > RISK_REPAIR_AGGREGATE_CI_LOW_MIN_STRICT and _risk_repair_safe(
        evidence.regional
    )
    base = integrity and amplitude and geometry
    return ArmAssessment(
        arm_id=evidence.arm_id,
        integrity_passed=integrity,
        amplitude_interpretable=amplitude,
        dice_superior=dice,
        geometry_noninferior=geometry,
        region_safe=regional,
        risk_repair=repair,
        superior_success=base and dice and regional,
        risk_repair_success=base and repair,
    )


def _winner(rows: Sequence[ArmEvidence], candidate_ids: Sequence[str]) -> str:
    by_id = {row.arm_id: row for row in rows}
    order = {arm_id: index for index, arm_id in enumerate(SELECTABLE_ARM_IDS)}
    best_mean = max(by_id[arm_id].dice_vs_reference.mean for arm_id in candidate_ids)
    tied = [
        arm_id
        for arm_id in candidate_ids
        if best_mean - by_id[arm_id].dice_vs_reference.mean <= WINNER_MEAN_TIE_TOLERANCE
    ]
    return min(
        tied,
        key=lambda arm_id: (
            by_id[arm_id].sdlogj_vs_reference.mean,
            ARM_SPECS_BY_ID[arm_id].post_rms_amplitude,
            order[arm_id],
        ),
    )


def select_next_branch(rows: Sequence[ArmEvidence], *, integrity_passed: bool = True) -> BranchDecision:
    if not isinstance(integrity_passed, bool):
        raise TypeError("integrity_passed must be boolean")
    if not integrity_passed:
        return BranchDecision(BRANCH_INVALID, None, (), (), "run-level integrity checks did not pass")
    if not isinstance(rows, Sequence) or any(not isinstance(row, ArmEvidence) for row in rows):
        raise TypeError("rows must contain ArmEvidence values")
    if tuple(row.arm_id for row in rows) != SELECTABLE_ARM_IDS:
        raise ValueError("C5b branch selection requires the three selectable arms in frozen order")

    assessed = tuple(assess_arm(row) for row in rows)
    if any(not row.integrity_passed for row in assessed):
        return BranchDecision(
            BRANCH_INVALID,
            None,
            (),
            (),
            "one or more selectable outputs are incomplete, uncertified, or have non-zero folds",
        )
    interpretable = tuple(row.arm_id for row in assessed if row.amplitude_interpretable)
    if not interpretable:
        return BranchDecision(
            BRANCH_CLOSE_CLIP,
            None,
            (),
            (),
            "none of the bounded bridge amplitudes retained its preregistered post-RMS amplitude",
        )
    superior = tuple(row.arm_id for row in assessed if row.amplitude_interpretable and row.dice_superior)
    superior_success = tuple(row.arm_id for row in assessed if row.superior_success)
    if superior_success:
        winner = _winner(rows, superior_success)
        return BranchDecision(
            BRANCH_FREEZE_SUPERIOR,
            winner,
            interpretable,
            superior,
            "a bounded bridge arm is Dice-superior to C4, geometry-noninferior, and region-safe",
        )
    repair_success = tuple(row.arm_id for row in assessed if row.risk_repair_success)
    if repair_success:
        winner = _winner(rows, repair_success)
        return BranchDecision(
            BRANCH_FREEZE_RISK_REPAIR,
            winner,
            interpretable,
            superior,
            "a bounded bridge arm is aggregate-noninferior and repairs labels 9 and 29",
        )
    if not superior:
        return BranchDecision(
            BRANCH_CLOSE_NO_SUPERIORITY,
            None,
            interpretable,
            (),
            "no amplitude-interpretable bridge arm is Dice-superior to C4",
        )
    geometry_viable = tuple(
        row.arm_id for row in assessed if row.amplitude_interpretable and row.dice_superior and row.geometry_noninferior
    )
    if not geometry_viable:
        return BranchDecision(
            BRANCH_CLOSE_GEOMETRY,
            None,
            interpretable,
            superior,
            "Dice-superior bridge capacity exists only beyond the frozen SDlogJ noninferiority margin",
        )
    return BranchDecision(
        BRANCH_CLOSE_REGIONAL,
        None,
        interpretable,
        superior,
        "a Dice-superior and geometry-noninferior bridge arm violates regional safeguards",
    )


def _policy_items() -> tuple[tuple[str, Any], ...]:
    return (
        ("protocol_id", PROTOCOL_ID),
        ("schema_version", SCHEMA_VERSION),
        ("development_dataset_id", DEVELOPMENT_DATASET_ID),
        ("expected_case_count", EXPECTED_CASE_COUNT),
        ("test_115_authorized", TEST_115_AUTHORIZED),
        ("work_eps", WORK_EPS),
        ("exact_claim_eps", EXACT_CLAIM_EPS),
        ("clip_current_fast_bound_role", CLIP_CURRENT_FAST_BOUND_ROLE),
        ("clip_output_fast_bound_role", CLIP_OUTPUT_FAST_BOUND_ROLE),
        ("descriptor_id", DESCRIPTOR_ID),
        ("image_normalization_mode", IMAGE_NORMALIZATION_MODE),
        ("image_normalization_std_floor", IMAGE_NORMALIZATION_STD_FLOOR),
        ("standardization_mode", STANDARDIZATION_MODE),
        ("standardization_floor", STANDARDIZATION_FLOOR),
        ("decoder_mode", DECODER_MODE),
        ("posterior_temperature", POSTERIOR_TEMPERATURE),
        ("stride_voxels", STRIDE_VOXELS),
        ("candidate_offsets_zyx", CANDIDATE_OFFSETS_ZYX),
        ("candidate_count", CANDIDATE_COUNT),
        ("centre_beta", CENTRE_BETA),
        ("pre_rms_multiplier", PRE_RMS_MULTIPLIER),
        ("post_smoothing_passes", POST_SMOOTHING_PASSES),
        ("common_evidence_collar", COMMON_EVIDENCE_COLLAR),
        ("rms_target_source_id", RMS_TARGET_SOURCE_ID),
        ("amplitude_stage", AMPLITUDE_STAGE),
        ("bridge_construction", BRIDGE_CONSTRUCTION),
        ("postclip_interpolation_allowed", POSTCLIP_INTERPOLATION_ALLOWED),
        ("observed_fold_count_definition", OBSERVED_FOLD_COUNT_DEFINITION),
        ("central_detj_invalid_required_zero", CENTRAL_DETJ_INVALID_REQUIRED_ZERO),
        ("digital_corner_union_required_zero", DIGITAL_CORNER_UNION_REQUIRED_ZERO),
        ("digital_ten_union_role", DIGITAL_TEN_UNION_ROLE),
        ("sdlogj_metric_id", SDLOGJ_METRIC_ID),
        ("dice_aggregation", DICE_AGGREGATION),
        ("dice_warp_interpolation", DICE_WARP_INTERPOLATION),
        ("arms", tuple(asdict(spec) for spec in ARM_SPECS)),
        ("bootstrap_resamples", BOOTSTRAP_RESAMPLES),
        ("bootstrap_seed", BOOTSTRAP_SEED),
        ("bootstrap_confidence", BOOTSTRAP_CONFIDENCE),
        ("bootstrap_method", BOOTSTRAP_METHOD_ID),
        ("bootstrap_quantile_method", BOOTSTRAP_QUANTILE_METHOD),
        ("contrast_ids_by_family", tuple(CONTRAST_IDS_BY_FAMILY.items())),
        ("amplitude_retention_median_min", AMPLITUDE_RETENTION_MEDIAN_MIN),
        ("amplitude_retention_case_min", AMPLITUDE_RETENTION_CASE_MIN),
        ("amplitude_retention_case_count_min", AMPLITUDE_RETENTION_CASE_COUNT_MIN),
        ("dice_vs_reference_mean_min", DICE_VS_REFERENCE_MEAN_MIN),
        ("dice_vs_reference_median_min_strict", DICE_VS_REFERENCE_MEDIAN_MIN_STRICT),
        ("dice_vs_reference_ci_low_min_strict", DICE_VS_REFERENCE_CI_LOW_MIN_STRICT),
        ("sdlogj_vs_reference_ci_high_max", SDLOGJ_VS_REFERENCE_CI_HIGH_MAX),
        ("regional_all_label_ci_low_min_strict", REGIONAL_ALL_LABEL_CI_LOW_MIN_STRICT),
        ("regional_risk_label_ci_low_min_strict", REGIONAL_RISK_LABEL_CI_LOW_MIN_STRICT),
        ("regional_risk_label_ids", REGIONAL_RISK_LABEL_IDS),
        ("regional_repair_label_ids", REGIONAL_REPAIR_LABEL_IDS),
        ("regional_repair_mean_min", REGIONAL_REPAIR_MEAN_MIN),
        ("regional_repair_ci_low_min_strict", REGIONAL_REPAIR_CI_LOW_MIN_STRICT),
        ("risk_repair_aggregate_ci_low_min_strict", RISK_REPAIR_AGGREGATE_CI_LOW_MIN_STRICT),
        ("winner_mean_tie_tolerance", WINNER_MEAN_TIE_TOLERANCE),
        ("winner_tie_break_order", WINNER_TIE_BREAK_ORDER),
        (
            "branch_priority",
            (
                BRANCH_INVALID,
                BRANCH_CLOSE_CLIP,
                BRANCH_FREEZE_SUPERIOR,
                BRANCH_FREEZE_RISK_REPAIR,
                BRANCH_CLOSE_NO_SUPERIORITY,
                BRANCH_CLOSE_GEOMETRY,
                BRANCH_CLOSE_REGIONAL,
            ),
        ),
    )


C5B_POLICY = _policy_items()


def canonical_policy_bytes() -> bytes:
    return json.dumps(dict(C5B_POLICY), sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("ascii")


def policy_sha256() -> str:
    return hashlib.sha256(canonical_policy_bytes()).hexdigest()


C5B_POLICY_SHA256 = "4fa7a24c75a0558ddb8255ed8115d6d50c0396d31eb0fb691a9ef4668eb5958a"


def assert_frozen_policy() -> None:
    if tuple(spec.arm_index for spec in ARM_SPECS) != tuple(range(len(ARM_SPECS))):
        raise RuntimeError("C5b arm indices are not contiguous")
    if tuple(spec.arm_id for spec in ARM_SPECS[:3]) != ANCHOR_ARM_IDS:
        raise RuntimeError("C5b anchors changed")
    if tuple(spec.arm_id for spec in ARM_SPECS if spec.selectable) != SELECTABLE_ARM_IDS:
        raise RuntimeError("C5b selectable arms changed")
    diagnostic = ARM_SPECS_BY_ID[DIAGNOSTIC_ARM_ID]
    if diagnostic.selectable or diagnostic.role != "DIAGNOSTIC":
        raise RuntimeError("C5b diagnostic arm became selectable")
    if any(spec.stride_voxels == STRIDE_VOXELS and not spec.recompute_preclip_direction for spec in ARM_SPECS):
        raise RuntimeError("C5b S4 arms no longer recompute the authenticated pre-clip direction")
    if POSTCLIP_INTERPOLATION_ALLOWED:
        raise RuntimeError("C5b must not interpolate materialized post-clip fields")
    if CANDIDATE_COUNT != 27 or CANDIDATE_OFFSETS_ZYX[13] != (0, 0, 0):
        raise RuntimeError("C5b sparse S4 candidate convention changed")
    if policy_sha256() != C5B_POLICY_SHA256:
        raise RuntimeError("C5b policy hash changed during import")


assert_frozen_policy()
