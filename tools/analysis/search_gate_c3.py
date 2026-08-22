from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from enum import Enum
from types import MappingProxyType
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

PROTOCOL_ID = "CTCF-SEARCH-GATE-C3A-V1"
SCHEMA_VERSION = "v1"

MIND_RADIUS = 1
MIND_DILATION = 2
CANDIDATE_RADIUS = 1
CANDIDATE_COUNT = 27
POSTERIOR_TEMPERATURE = 1.0
COLLAR_WIDTH = 4
WORK_EPS = 0.0011
EXACT_CLAIM_EPS = 0.001
LOCAL_CLIP_SWEEPS = 1
CONTROL_DECODER_PARITY_ATOL = 2e-6
C2_DICE_PARITY_ATOL = 1e-8
COST_STANDARDIZATION_FLOOR = 1e-6
IMAGE_ZSCORE_STD_FLOOR = 1e-6
NCC_DENOMINATOR_EPS = 1e-5

MESSAGE_PASSING_ID = "CTCF_LINEAR_ENTROPY_LOGIT_MP_V1"
MESSAGE_PASSING_PASSES = 1
MESSAGE_PASSING_AXIS_KERNEL = (0.25, 0.5, 0.25)

MIND_SUPPORT_WINDOW = 1
NCC7_WINDOW = 7
NCC9_WINDOW = 9
SUPPORT_RETENTION_MIN = 0.99
PRIMARY_NCC_IMPROVEMENT_MIN = 1e-6

BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 0
BOOTSTRAP_CONFIDENCE = 0.95
CAPACITY_MEAN_DICE_DELTA_MIN = 0.002
CAPACITY_CI_LOW_DICE_DELTA_MIN = 0.001
POLICY_MEAN_DICE_DELTA_MIN = 0.001
GEOMETRY_NONINFERIOR_TOLERANCE = 1e-6
WINNER_TIE_TOLERANCE = 1e-6
FLOAT32_PARITY_RTOL = 1e-6
FLOAT32_PARITY_ATOL = 1e-8


@dataclass(frozen=True, slots=True)
class ArmSpec:
    """One immutable C3a state specification.

    `selectable` is deliberately false for the baseline, historical/raw controls,
    and the unnormalised stress arm. It is a protocol property, not a runtime
    conclusion.
    """

    arm_index: int
    arm_id: str
    role: str
    cost: str
    scale: float
    message_passing: str
    decoder: str
    post_smooth_passes: int
    rms_reference_arm_id: str | None
    selectable: bool
    stress_only: bool = False


ARM_SPECS = (
    ArmSpec(0, "zero_update", "baseline", "none", 0.0, "none", "none", 0, None, False),
    ArmSpec(
        1,
        "c1_raw_conf_post1",
        "historical_control",
        "raw_standardized_mind_ssc",
        1.0,
        "none",
        "posterior_expectation_times_confidence",
        1,
        None,
        False,
    ),
    ArmSpec(
        2,
        "raw_conf_post1",
        "raw_control",
        "raw_standardized_mind_ssc",
        2.0,
        "none",
        "posterior_expectation_times_confidence",
        1,
        None,
        False,
    ),
    ArmSpec(
        3,
        "raw_conf_post2",
        "post_smoothing_control",
        "raw_standardized_mind_ssc",
        2.0,
        "none",
        "posterior_expectation_times_confidence",
        2,
        None,
        False,
    ),
    ArmSpec(
        4,
        "iso_mp_conf_post1",
        "candidate",
        "raw_standardized_mind_ssc",
        2.0,
        "isotropic_case_mean_entropy_mass",
        "posterior_expectation_times_confidence",
        1,
        None,
        True,
    ),
    ArmSpec(
        5,
        "adaptive_mp_conf_post1",
        "candidate",
        "raw_standardized_mind_ssc",
        2.0,
        "voxelwise_normalized_entropy_mass",
        "posterior_expectation_times_confidence",
        1,
        None,
        True,
    ),
    ArmSpec(
        6,
        "raw_mean_normmatched_post1",
        "candidate",
        "raw_standardized_mind_ssc",
        2.0,
        "none",
        "posterior_expectation_rms_matched",
        1,
        "raw_conf_post1",
        True,
    ),
    ArmSpec(
        7,
        "adaptive_mean_adaptref_normmatched_post1",
        "candidate",
        "raw_standardized_mind_ssc",
        2.0,
        "voxelwise_normalized_entropy_mass",
        "posterior_expectation_rms_matched",
        1,
        "adaptive_mp_conf_post1",
        True,
    ),
    ArmSpec(
        8,
        "adaptive_mean_rawref_normmatched_post1",
        "candidate",
        "raw_standardized_mind_ssc",
        2.0,
        "voxelwise_normalized_entropy_mass",
        "posterior_expectation_rms_matched",
        1,
        "raw_conf_post1",
        True,
    ),
    ArmSpec(
        9,
        "adaptive_mean_raw_post1",
        "stress_control",
        "raw_standardized_mind_ssc",
        2.0,
        "voxelwise_normalized_entropy_mass",
        "posterior_expectation_unnormalized",
        1,
        None,
        False,
        True,
    ),
)

ARM_SPECS_BY_ID: Mapping[str, ArmSpec] = MappingProxyType({spec.arm_id: spec for spec in ARM_SPECS})
SELECTABLE_ARM_IDS = tuple(spec.arm_id for spec in ARM_SPECS if spec.selectable)
WINNER_TIE_ORDER = (
    "raw_mean_normmatched_post1",
    "iso_mp_conf_post1",
    "adaptive_mp_conf_post1",
    "adaptive_mean_adaptref_normmatched_post1",
    "adaptive_mean_rawref_normmatched_post1",
)


@dataclass(frozen=True, slots=True)
class C3APolicy:
    protocol_id: str
    schema_version: str
    arms: tuple[ArmSpec, ...]
    fixed_parameters: tuple[tuple[str, Any], ...]
    proposal_order: tuple[str, ...]
    message_passing: tuple[tuple[str, Any], ...]
    rms_matching: tuple[tuple[str, Any], ...]
    diagnostics: tuple[tuple[str, Any], ...]
    support: tuple[tuple[str, Any], ...]
    primary_transaction: tuple[tuple[str, Any], ...]
    statistics: tuple[tuple[str, Any], ...]
    winner_tie_order: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        """Return a detached JSON-compatible representation for manifests."""
        return _json_compatible(asdict(self))


C3A_POLICY = C3APolicy(
    protocol_id=PROTOCOL_ID,
    schema_version=SCHEMA_VERSION,
    arms=ARM_SPECS,
    fixed_parameters=(
        ("steps", 1),
        ("mind_radius", MIND_RADIUS),
        ("mind_dilation", MIND_DILATION),
        ("candidate_radius", CANDIDATE_RADIUS),
        ("candidate_count", CANDIDATE_COUNT),
        ("posterior_temperature", POSTERIOR_TEMPERATURE),
        ("identity_collar_width", COLLAR_WIDTH),
        ("work_eps", WORK_EPS),
        ("exact_claim_eps", EXACT_CLAIM_EPS),
        ("local_clip_sweeps", LOCAL_CLIP_SWEEPS),
        ("legacy_new_decoder_parity_atol", CONTROL_DECODER_PARITY_ATOL),
        ("frozen_c2_dice_parity_atol", C2_DICE_PARITY_ATOL),
        ("cost_standardization_floor", COST_STANDARDIZATION_FLOOR),
        ("image_zscore_std_floor", IMAGE_ZSCORE_STD_FLOOR),
        ("ncc_denominator_eps", NCC_DENOMINATOR_EPS),
        ("materialized_dtype", "float32"),
        ("decision_worker_has_label_inputs", False),
        ("decision_worker_has_raw_container_inputs", False),
        ("os_level_data_isolation_claimed", False),
        ("test115_accessible", False),
    ),
    proposal_order=(
        "cost_construction",
        "optional_predecoder_message_passing",
        "decoder",
        "certified_local_clip_work_eps_0.0011_one_sweep",
        "fp32_save_reload",
        "exact_certificate_eps_0.001",
        "frozen_label_free_utility",
    ),
    message_passing=(
        ("method_id", MESSAGE_PASSING_ID),
        ("input", "ell=-z, where z is the per-offset standardized cost"),
        ("posterior", "q0=softmax(ell)=softmax(-z), temperature=1"),
        ("invalid_offsets", "masked before posterior normalization"),
        ("entropy", "h=H(q0)/log(K), with h=0 when K=1"),
        ("passes", MESSAGE_PASSING_PASSES),
        ("axis_kernel", MESSAGE_PASSING_AXIS_KERNEL),
        ("kernel_3d", "outer product of the axis kernel"),
        ("neighbours", "geometry-valid voxels only; renormalize by valid kernel mass"),
        ("outside", "invalid, never zero-valued evidence"),
        ("isotropic_mass", "case mean of h over the geometry mask"),
        ("adaptive_mass", "voxelwise h"),
        ("mean_mass_parity_rtol", FLOAT32_PARITY_RTOL),
        ("mean_mass_parity_atol", FLOAT32_PARITY_ATOL),
        ("update", "ell_out=(1-lambda)*ell+lambda*M(ell)"),
        ("rezstandardize", False),
    ),
    rms_matching=(
        ("domain", "geometry mask after scale, message passing, decoder, post-smoothing and collar"),
        ("norm", "sqrt(mean(sum_channel(residual**2)))"),
        ("raw_mean_reference", "raw_conf_post1"),
        ("adaptive_adaptref_mean_reference", "adaptive_mp_conf_post1"),
        ("adaptive_rawref_mean_reference", "raw_conf_post1"),
        ("zero_over_zero", "return the zero residual"),
        ("positive_reference_over_zero", "integrity failure"),
        ("verification_rtol", FLOAT32_PARITY_RTOL),
        ("verification_atol", FLOAT32_PARITY_ATOL),
    ),
    diagnostics=(
        ("posterior_id", "MASKED_27_OFFSET_LOGIT_POSTERIOR_GEOMETRY_V1"),
        ("cost_gap", "mean_G(top1_valid_logit-top2_valid_logit); raw logits=-standardized_cost"),
        ("posterior_peak", "mean_G(max_valid_probability)"),
        ("entropy", "mean_G entropy in natural-log units"),
        ("invalid_offset_fraction", "invalid offsets divided by |G|*27"),
        ("decoder_norms", "mean_G voxelwise L2 posterior-mean before/after confidence and their ratio"),
        ("roughness_id", "RMS_VECTOR_FIRST_DIFFERENCE_GEOMETRY_PAIRS_V1"),
        ("roughness", "RMS vector first difference over positive-axis pairs with both endpoints in G"),
        (
            "requested_state",
            "materialized FP32; exact certificate, residual RMS, MIND/NCC7/NCC9, fast bound, geometry, "
            "post-barrier Dice",
        ),
        (
            "postclip_state",
            "materialized FP32; exact certificate, residual RMS, MIND/NCC7/NCC9, fast bound, geometry, "
            "post-barrier Dice",
        ),
        ("clip", "residual norm retention and effective-alpha min/p50/p95/max"),
    ),
    support=(
        ("mind_window", MIND_SUPPORT_WINDOW),
        ("ncc7_window", NCC7_WINDOW),
        ("ncc9_window", NCC9_WINDOW),
        ("baseline", "G & erode(valid(Psi0), win//2)"),
        ("candidate", "G & erode(valid(Psi0) & valid(Psia), win//2)"),
        ("erosion_outside", False),
    ),
    primary_transaction=(
        ("exact_certificate_required", True),
        ("support_retention_min", SUPPORT_RETENTION_MIN),
        ("empty_common_support", "retention=0 and explicit byte-identical rollback"),
        ("utility", "common-support NCC7 loss"),
        ("improvement", "baseline_loss-candidate_loss"),
        ("improvement_min_absolute", PRIMARY_NCC_IMPROVEMENT_MIN),
        ("failure", "byte-identical rollback"),
    ),
    statistics=(
        ("bootstrap", "paired subject bootstrap percentile CI for the mean; development-descriptive only"),
        ("bootstrap_resamples", BOOTSTRAP_RESAMPLES),
        ("bootstrap_seed", BOOTSTRAP_SEED),
        ("bootstrap_confidence", BOOTSTRAP_CONFIDENCE),
        ("bootstrap_rng", "numpy.random.default_rng(seed), PCG64"),
        ("wins", "paired mean > 0 and paired median > 0 and percentile CI low > 0"),
        ("capacity_estimand", "accept-all exact post-clip candidate"),
        ("capacity_eligibility", "exact certificate on every one of the 58 candidates"),
        ("requested_dice", "post-barrier diagnostic only; never selectable"),
        ("primary_exact_eligibility", "all returned fields exact; failed candidates roll back to exact baseline"),
        ("primary_support_eligibility", "support diagnostics defined; per-case retention failure is rollback"),
        ("primary_geometry_estimand", "field returned by the frozen primary transaction policy"),
        ("metric_status_error", "fatal scientific-integrity failure"),
        ("material_capacity_mean_dice_delta_min", CAPACITY_MEAN_DICE_DELTA_MIN),
        ("material_capacity_median_dice_delta", ">0"),
        ("material_capacity_ci_low_dice_delta_min_strict", CAPACITY_CI_LOW_DICE_DELTA_MIN),
        ("viable_policy_mean_dice_delta_min", POLICY_MEAN_DICE_DELTA_MIN),
        ("viable_policy_median_dice_delta", ">0"),
        ("viable_policy_ci_low_dice_delta", ">0"),
        ("geometry_noninferior_tolerance", GEOMETRY_NONINFERIOR_TOLERANCE),
        ("winner_score", "primary-policy returned mean Dice"),
        ("winner_tie_tolerance", WINNER_TIE_TOLERANCE),
    ),
    winner_tie_order=WINNER_TIE_ORDER,
)


def _json_compatible(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Mapping):
        return {str(key): _json_compatible(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_compatible(item) for item in value]
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    return value


def policy_sha256(policy: C3APolicy = C3A_POLICY) -> str:
    encoded = json.dumps(policy.to_dict(), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


# Literal digest: an edit to the policy must deliberately update this value.
C3A_POLICY_SHA256 = "d9e0161b9abe84fafb0ab2bb7f0a6c3e583d061a51e2de63d18373f1fe51781c"


def assert_frozen_policy() -> None:
    actual = policy_sha256()
    if actual != C3A_POLICY_SHA256:
        raise RuntimeError(f"C3a policy hash mismatch: declared={C3A_POLICY_SHA256}, actual={actual}")


def _validate_mask(mask: torch.Tensor, name: str) -> None:
    if not isinstance(mask, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if mask.dtype is not torch.bool:
        raise ValueError(f"{name} must have dtype torch.bool, got {mask.dtype}")
    if mask.ndim != 5 or mask.shape[1] != 1 or min(mask.shape[2:]) < 1:
        raise ValueError(f"{name} must have shape [B,1,D,H,W], got {tuple(mask.shape)}")


def binary_erode_mask(mask: torch.Tensor, radius: int) -> torch.Tensor:
    """Binary 3-D erosion with a cubic structuring element and false outside padding."""
    _validate_mask(mask, "mask")
    if isinstance(radius, bool) or not isinstance(radius, int) or radius < 0:
        raise ValueError("radius must be a non-negative integer")
    if radius == 0:
        return mask.clone()
    kernel_size = 2 * radius + 1
    output = mask
    # A cubic erosion is separable. Three 1-D passes avoid a dense win**3
    # convolution on full-resolution MRI volumes.
    for kernel_shape, padding in (
        ((kernel_size, 1, 1), (0, 0, 0, 0, radius, radius)),
        ((1, kernel_size, 1), (0, 0, radius, radius, 0, 0)),
        ((1, 1, kernel_size), (radius, radius, 0, 0, 0, 0)),
    ):
        padded = F.pad(output.to(torch.float32), padding, mode="constant", value=0.0)
        kernel = torch.ones((1, 1, *kernel_shape), dtype=torch.float32, device=mask.device)
        output = F.conv3d(padded, kernel) == float(kernel_size)
    return output


@dataclass(frozen=True, slots=True)
class CommonSupport:
    utility_id: str
    window: int
    erosion_radius: int
    baseline_mask: torch.Tensor
    pair_mask: torch.Tensor
    baseline_count: int
    pair_count: int
    retention: float


def build_common_support(
    geometry_mask: torch.Tensor,
    baseline_valid_mask: torch.Tensor,
    candidate_valid_mask: torch.Tensor,
    *,
    window: int,
    utility_id: str,
) -> CommonSupport:
    """Build baseline and pairwise-common center masks for one odd utility window.

    The geometry mask itself is not eroded. Erosion protects the local window
    from source-boundary zero padding; the fixed identity collar already defines
    the permitted target centers.
    """
    for name, mask in (
        ("geometry_mask", geometry_mask),
        ("baseline_valid_mask", baseline_valid_mask),
        ("candidate_valid_mask", candidate_valid_mask),
    ):
        _validate_mask(mask, name)
    if geometry_mask.shape != baseline_valid_mask.shape or geometry_mask.shape != candidate_valid_mask.shape:
        raise ValueError("geometry, baseline-valid and candidate-valid masks must have identical shapes")
    if not isinstance(utility_id, str) or not utility_id.strip():
        raise ValueError("utility_id must be a non-empty string")
    if isinstance(window, bool) or not isinstance(window, int) or window < 1 or window % 2 == 0:
        raise ValueError("window must be a positive odd integer")

    radius = window // 2
    baseline = geometry_mask & binary_erode_mask(baseline_valid_mask, radius)
    pair = geometry_mask & binary_erode_mask(baseline_valid_mask & candidate_valid_mask, radius)
    baseline_count = int(baseline.sum(dtype=torch.int64).item())
    pair_count = int(pair.sum(dtype=torch.int64).item())
    if baseline_count == 0:
        raise ValueError(f"{utility_id} baseline common-support denominator is empty")
    if pair_count > baseline_count or bool((pair & ~baseline).any()):
        raise RuntimeError("pairwise common support is not a subset of baseline support")
    return CommonSupport(
        utility_id=utility_id,
        window=window,
        erosion_radius=radius,
        baseline_mask=baseline,
        pair_mask=pair,
        baseline_count=baseline_count,
        pair_count=pair_count,
        retention=pair_count / baseline_count,
    )


def build_c3a_supports(
    geometry_mask: torch.Tensor,
    baseline_valid_mask: torch.Tensor,
    candidate_valid_mask: torch.Tensor,
) -> dict[str, CommonSupport]:
    """Return the only three common-support definitions owned by C3a."""
    return {
        "mind": build_common_support(
            geometry_mask,
            baseline_valid_mask,
            candidate_valid_mask,
            window=MIND_SUPPORT_WINDOW,
            utility_id="COMMON_MIND_SSC",
        ),
        "ncc7": build_common_support(
            geometry_mask,
            baseline_valid_mask,
            candidate_valid_mask,
            window=NCC7_WINDOW,
            utility_id="COMMON_NCC7",
        ),
        "ncc9": build_common_support(
            geometry_mask,
            baseline_valid_mask,
            candidate_valid_mask,
            window=NCC9_WINDOW,
            utility_id="COMMON_NCC9",
        ),
    }


@dataclass(frozen=True, slots=True)
class PrimaryPolicyDecision:
    accept: bool
    rollback: bool
    reason: str
    exact_certified: bool
    support_retention: float
    ncc_improvement: float | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _finite_float(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise TypeError(f"{name} must be a real number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def primary_ncc_decision(
    *,
    exact_certified: bool,
    support_retention: float,
    baseline_ncc_loss: float | None,
    candidate_ncc_loss: float | None,
) -> PrimaryPolicyDecision:
    """Apply the sole C3a transaction rule; lower NCC loss is better."""
    if not isinstance(exact_certified, (bool, np.bool_)):
        raise TypeError("exact_certified must be boolean")
    retention = _finite_float(support_retention, "support_retention")
    if not 0.0 <= retention <= 1.0:
        raise ValueError("support_retention must be in [0, 1]")

    if not bool(exact_certified):
        reason = "ROLLBACK_EXACT_CERTIFICATE_FAILED"
        improvement = None
    elif retention < SUPPORT_RETENTION_MIN:
        reason = "ROLLBACK_COMMON_SUPPORT_RETENTION_BELOW_0.99"
        improvement = None
    elif baseline_ncc_loss is None or candidate_ncc_loss is None:
        raise ValueError("eligible common-support NCC7 losses must both be finite")
    else:
        baseline = _finite_float(baseline_ncc_loss, "baseline_ncc_loss")
        candidate = _finite_float(candidate_ncc_loss, "candidate_ncc_loss")
        improvement = baseline - candidate
        reason = (
            "ROLLBACK_COMMON_NCC7_IMPROVEMENT_BELOW_1E-6"
            if improvement < PRIMARY_NCC_IMPROVEMENT_MIN
            else "ACCEPT_EXACT_COMMON_NCC7_POLICY"
        )
    accept = reason.startswith("ACCEPT_")
    return PrimaryPolicyDecision(
        accept=accept,
        rollback=not accept,
        reason=reason,
        exact_certified=bool(exact_certified),
        support_retention=retention,
        ncc_improvement=improvement,
    )


def should_rollback(**kwargs: Any) -> bool:
    return primary_ncc_decision(**kwargs).rollback


class MetricStatus(str, Enum):
    OK = "OK"
    UNDEFINED_NONPOSITIVE = "UNDEFINED_NONPOSITIVE"
    ERROR = "ERROR"


JsonScalar = str | int | float | bool | None


def _validate_metric_name(name: str, label: str) -> str:
    if not isinstance(name, str) or not name.strip():
        raise ValueError(f"{label} must be a non-empty string")
    normalized = name.strip().lower().replace("-", "_")
    if normalized.replace("_", "") == "sdlogj":
        raise ValueError(f"Ambiguous metric name {name!r} is forbidden; use a convention-bearing metric id")
    return name


def _component_items(components: Mapping[str, Any] | None) -> tuple[tuple[str, JsonScalar], ...]:
    if components is None:
        return ()
    if not isinstance(components, Mapping):
        raise TypeError("metric components must be a mapping")
    output: list[tuple[str, JsonScalar]] = []
    for key, raw_value in sorted(components.items()):
        name = _validate_metric_name(key, "component name")
        value = _json_compatible(raw_value)
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError(f"metric component {name!r} is non-finite")
        if value is not None and not isinstance(value, (str, int, float, bool)):
            raise TypeError(f"metric component {name!r} is not a JSON scalar")
        output.append((name, value))
    return tuple(output)


@dataclass(frozen=True, slots=True)
class MetricEnvelope:
    metric_id: str
    status: MetricStatus
    value: float | None
    components: tuple[tuple[str, JsonScalar], ...] = ()
    error_type: str | None = None
    detail: str | None = None

    def __post_init__(self) -> None:
        _validate_metric_name(self.metric_id, "metric_id")
        if not isinstance(self.status, MetricStatus):
            raise TypeError("status must be a MetricStatus")
        if self.status is MetricStatus.OK:
            if self.value is not None:
                object.__setattr__(self, "value", _finite_float(self.value, f"{self.metric_id} value"))
            if self.error_type is not None or self.detail is not None:
                raise ValueError("an OK metric cannot carry error fields")
            object.__setattr__(self, "components", _component_items(dict(self.components)))
        elif self.value is not None or self.components:
            raise ValueError("a non-OK metric cannot carry a value or components")
        elif not isinstance(self.error_type, str) or not isinstance(self.detail, str):
            raise ValueError("a non-OK metric must carry serializable error_type and detail strings")

    @classmethod
    def ok(
        cls,
        metric_id: str,
        value: float | None,
        components: Mapping[str, Any] | None = None,
    ) -> MetricEnvelope:
        metric_id = _validate_metric_name(metric_id, "metric_id")
        numeric = None if value is None else _finite_float(value, f"{metric_id} value")
        return cls(metric_id, MetricStatus.OK, numeric, _component_items(components))

    @classmethod
    def undefined_nonpositive(cls, metric_id: str, error: BaseException) -> MetricEnvelope:
        metric_id = _validate_metric_name(metric_id, "metric_id")
        return cls(
            metric_id,
            MetricStatus.UNDEFINED_NONPOSITIVE,
            None,
            (),
            type(error).__name__,
            str(error),
        )

    @classmethod
    def error(cls, metric_id: str, error: BaseException) -> MetricEnvelope:
        metric_id = _validate_metric_name(metric_id, "metric_id")
        return cls(metric_id, MetricStatus.ERROR, None, (), type(error).__name__, str(error))

    def to_dict(self) -> dict[str, Any]:
        return {
            "metric_id": self.metric_id,
            "status": self.status.value,
            "value": self.value,
            "components": dict(self.components),
            "error_type": self.error_type,
            "detail": self.detail,
        }


def metric_envelope(metric_id: str, computation: Callable[[], Any]) -> MetricEnvelope:
    """Execute one metric in a fail-isolated, JSON-serializable envelope."""
    _validate_metric_name(metric_id, "metric_id")
    if not callable(computation):
        raise TypeError("computation must be callable")
    try:
        result = computation()
        value = getattr(result, "value", result)
        components = getattr(result, "components", None)
        return MetricEnvelope.ok(metric_id, value, components)
    except Exception as error:  # fail-isolation is the purpose of this boundary
        # Keep this protocol module independent from the geometry implementation.
        # TASK-0014 owns this deliberately named exception contract.
        if type(error).__name__ == "MetricFailClosedError":
            return MetricEnvelope.undefined_nonpositive(metric_id, error)
        return MetricEnvelope.error(metric_id, error)


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

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _one_dimensional_finite(values: Sequence[float] | np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size == 0:
        raise ValueError(f"{name} must be a non-empty one-dimensional sequence")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values")
    return array


def paired_summary(
    candidate_or_differences: Sequence[float] | np.ndarray,
    baseline: Sequence[float] | np.ndarray | None = None,
    *,
    bootstrap_resamples: int = BOOTSTRAP_RESAMPLES,
    bootstrap_seed: int = BOOTSTRAP_SEED,
    confidence: float = BOOTSTRAP_CONFIDENCE,
) -> PairedSummary:
    """Summarize paired candidate-minus-baseline values with a percentile CI."""
    candidate = _one_dimensional_finite(candidate_or_differences, "candidate_or_differences")
    if baseline is None:
        differences = candidate
    else:
        baseline_array = _one_dimensional_finite(baseline, "baseline")
        if baseline_array.shape != candidate.shape:
            raise ValueError("candidate and baseline must have the same number of paired cases")
        differences = candidate - baseline_array
    if isinstance(bootstrap_resamples, bool) or not isinstance(bootstrap_resamples, int) or bootstrap_resamples < 1:
        raise ValueError("bootstrap_resamples must be a positive integer")
    if isinstance(bootstrap_seed, bool) or not isinstance(bootstrap_seed, int) or bootstrap_seed < 0:
        raise ValueError("bootstrap_seed must be a non-negative integer")
    confidence_value = _finite_float(confidence, "confidence")
    if not 0.0 < confidence_value < 1.0:
        raise ValueError("confidence must be strictly between 0 and 1")

    generator = np.random.default_rng(bootstrap_seed)
    indices = generator.integers(0, differences.size, size=(bootstrap_resamples, differences.size))
    bootstrap_means = differences[indices].mean(axis=1)
    tail = (1.0 - confidence_value) / 2.0
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
        bootstrap_resamples=bootstrap_resamples,
        bootstrap_seed=bootstrap_seed,
        bootstrap_confidence=confidence_value,
    )


def wins(summary: PairedSummary) -> bool:
    if not isinstance(summary, PairedSummary):
        raise TypeError("wins expects a PairedSummary")
    return summary.mean > 0.0 and summary.median > 0.0 and summary.ci_low > 0.0


def materially_strong_capacity(summary: PairedSummary) -> bool:
    if not isinstance(summary, PairedSummary):
        raise TypeError("materially_strong_capacity expects a PairedSummary")
    return (
        summary.mean >= CAPACITY_MEAN_DICE_DELTA_MIN
        and summary.median > 0.0
        and summary.ci_low > CAPACITY_CI_LOW_DICE_DELTA_MIN
    )


def geometry_noninferior(
    candidate_mean_delta: float,
    c1_mean_delta: float,
    *,
    all_candidate_metrics_defined: bool,
) -> bool:
    if not isinstance(all_candidate_metrics_defined, (bool, np.bool_)):
        raise TypeError("all_candidate_metrics_defined must be boolean")
    candidate = _finite_float(candidate_mean_delta, "candidate_mean_delta")
    c1 = _finite_float(c1_mean_delta, "c1_mean_delta")
    return bool(all_candidate_metrics_defined) and candidate <= c1 + GEOMETRY_NONINFERIOR_TOLERANCE


def viable_primary_policy(
    summary: PairedSummary,
    *,
    all_returned_exact_certified: bool,
    all_support_diagnostics_defined: bool,
    geometry_is_noninferior: bool,
) -> bool:
    if not isinstance(summary, PairedSummary):
        raise TypeError("viable_primary_policy expects a PairedSummary")
    flags = (all_returned_exact_certified, all_support_diagnostics_defined, geometry_is_noninferior)
    if not all(isinstance(value, (bool, np.bool_)) for value in flags):
        raise TypeError("viability flags must be boolean")
    return (
        bool(all_returned_exact_certified)
        and bool(all_support_diagnostics_defined)
        and bool(geometry_is_noninferior)
        and summary.mean >= POLICY_MEAN_DICE_DELTA_MIN
        and summary.median > 0.0
        and summary.ci_low > 0.0
    )


@dataclass(frozen=True, slots=True)
class WinnerDecision:
    winner_arm_id: str | None
    eligible_arm_ids: tuple[str, ...]
    best_score: float | None
    tied_at_tolerance: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def winner_decision(
    primary_policy_scores: Mapping[str, float],
    viable: Mapping[str, bool],
) -> WinnerDecision:
    """Choose among selectable arms by score, resolving 1e-6 ties by frozen simplicity order."""
    if not isinstance(primary_policy_scores, Mapping) or not isinstance(viable, Mapping):
        raise TypeError("primary_policy_scores and viable must be mappings")
    unknown = (set(primary_policy_scores) | set(viable)) - set(ARM_SPECS_BY_ID)
    if unknown:
        raise ValueError(f"unknown C3a arm ids: {sorted(unknown)}")
    validated_scores: dict[str, float] = {}
    for arm_id, raw_score in primary_policy_scores.items():
        score = _finite_float(raw_score, f"score for {arm_id}")
        if not 0.0 <= score <= 1.0:
            raise ValueError(f"primary-policy mean Dice for {arm_id} must be in [0, 1]")
        validated_scores[arm_id] = score
    for arm_id, is_viable in viable.items():
        if not isinstance(is_viable, (bool, np.bool_)):
            raise TypeError(f"viability for {arm_id} must be boolean")
    nonselectable_true = sorted(
        arm_id for arm_id, value in viable.items() if value and arm_id not in SELECTABLE_ARM_IDS
    )
    if nonselectable_true:
        raise ValueError(f"non-selectable arms cannot be marked viable: {nonselectable_true}")

    eligible: list[str] = []
    scores: dict[str, float] = {}
    for arm_id in SELECTABLE_ARM_IDS:
        is_viable = viable.get(arm_id, False)
        if not bool(is_viable):
            continue
        if arm_id not in primary_policy_scores:
            raise ValueError(f"missing primary-policy score for viable arm {arm_id}")
        scores[arm_id] = validated_scores[arm_id]
        eligible.append(arm_id)

    if not eligible:
        return WinnerDecision(None, (), None, ())
    best_score = max(scores.values())
    tied = tuple(
        arm_id
        for arm_id in WINNER_TIE_ORDER
        if arm_id in scores and best_score - scores[arm_id] <= WINNER_TIE_TOLERANCE
    )
    winner = tied[0]
    return WinnerDecision(winner, tuple(eligible), best_score, tied)


def select_winner(primary_policy_scores: Mapping[str, float], viable: Mapping[str, bool]) -> str | None:
    return winner_decision(primary_policy_scores, viable).winner_arm_id


assert_frozen_policy()
