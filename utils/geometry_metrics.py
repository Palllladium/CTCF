"""The named Jacobian and inverse-consistency metrics, each frozen under its own metric id.

The ids are long on purpose: they spell out the scheme (central vs shifted vs digital), the
crop, the mask, and the ddof, because these choices are what the number means. Two ids in this
module can score the same field and disagree by orders of magnitude, so a value from here is
only quotable together with the id that produced it -- ``compute_metric`` refuses an id it does
not know rather than guessing a nearby one.

``MetricFailClosedError`` marks a metric that declines to report rather than reporting a
degraded value. A metric that does not certify a field says nothing about whether the field
folds; only a negative witness does that.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

import numpy as np
import torch

from utils.field import jacobian_det

MATHEMATICAL_SDLOGJ_FULL = "CTCF_MATHEMATICAL_SDLOGJ_CENTRAL_FULL_UNMASKED_DDOF0_FAILCLOSED_V1"
MATHEMATICAL_SDLOGJ_CROP2 = "CTCF_MATHEMATICAL_SDLOGJ_CENTRAL_CROP2_UNMASKED_DDOF0_FAILCLOSED_V1"
DIAGNOSTIC_SDLOGJ_POSITIVE_ONLY_CROP2 = "CTCF_DIAGNOSTIC_SDLOGJ_CENTRAL_CROP2_UNMASKED_DDOF0_POSITIVE_ONLY_V1"
LEARN2REG_SHIFTED_SDLOGJ = "LEARN2REG_UTSRMORPH_SHIFTED_SDLOGJ_FP32DERIV_CROP2_UNMASKED_DDOF0_BATCH1_VOXEL_UNITS_V1"
LEARN2REG_SHIFTED_SDLOGJ_MASKED = (
    "LEARN2REG_UTSRMORPH_SHIFTED_SDLOGJ_FP32DERIV_CROP2_MASKED_DDOF0_BATCH1_VOXEL_UNITS_V1"
)
LEGACY_SHIFTED_J = "CTCF_LEGACY_SHIFTED_J_FULL_DDOF1_V1"
DETJ_DIAGNOSTICS = "CTCF_DETJ_DIAGNOSTICS_CENTRAL_CROP2_UNMASKED_V1"
DIGITAL_DECOMPOSITION = "CTCF_DIGITAL_DECOMPOSITION_TEN_DETERMINANTS_INTERIOR_UNMASKED_V1"

MIN_SPATIAL_SIZE = 5
SHIFT = 3.0
CLAMP_LOW = 1e-9
CLAMP_HIGH = 1e9
DETJ_QUANTILES = (0.001, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 0.999)
COMPRESSION_THRESHOLDS = (0.1, 0.25, 0.5)
EXPANSION_THRESHOLDS = (2.0, 4.0, 10.0)

_AXIS_MODES = (
    ("+", "+", "+"), ("+", "+", "-"), ("+", "-", "+"), ("+", "-", "-"),
    ("-", "+", "+"), ("-", "+", "-"), ("-", "-", "+"), ("-", "-", "-"),
)  # fmt: skip
_STAR_OFFSETS = (
    ((-1, -1, 0), (-1, 0, -1), (0, -1, -1)),
    ((1, 1, 0), (0, 1, 1), (1, 0, 1)),
)
_CORNER_NAMES = tuple("corner_" + "".join("p" if mode == "+" else "m" for mode in modes) for modes in _AXIS_MODES)
DETERMINANT_NAMES = (*_CORNER_NAMES, "jstar1", "jstar2")

_CENTRAL_DETERMINANT = "det(I + grad u), central differences, zero derivative on the outer slice"
_VOXEL_UNITS = "voxel units, channel-first [1,3,D,H,W], channel c pairs with spatial axis c"


class MetricFailClosedError(ValueError):
    """A metric refused to report because its own preconditions failed on this field."""


@dataclass(frozen=True)
class MetricResult:
    metric_id: str
    value: float | None
    components: dict[str, Any]
    metadata: Any


def _spec(metric_id: str, **fields: Any) -> Any:
    base = {
        "metric_id": metric_id,
        "axis_order": "zyx",
        "units": _VOXEL_UNITS,
        "batch": "exactly one field per call",
        "min_spatial_size": MIN_SPATIAL_SIZE,
    }
    return MappingProxyType({**base, **fields})


METRIC_SPECS: Any = MappingProxyType(
    {
        MATHEMATICAL_SDLOGJ_FULL: _spec(
            MATHEMATICAL_SDLOGJ_FULL,
            value="population std of log(detJ) over the whole volume",
            crop=0,
            ddof=0,
            mask="none",
            clamp="none",
            determinant=_CENTRAL_DETERMINANT,
            dtype="float64",
            fail_policy="raise MetricFailClosedError on any non-finite or non-positive detJ",
        ),
        MATHEMATICAL_SDLOGJ_CROP2: _spec(
            MATHEMATICAL_SDLOGJ_CROP2,
            value="population std of log(detJ) over the 2-voxel-cropped interior",
            crop=2,
            ddof=0,
            mask="none",
            clamp="none",
            determinant=_CENTRAL_DETERMINANT,
            dtype="float64",
            fail_policy="raise MetricFailClosedError on any non-finite or non-positive detJ",
        ),
        DIAGNOSTIC_SDLOGJ_POSITIVE_ONLY_CROP2: _spec(
            DIAGNOSTIC_SDLOGJ_POSITIVE_ONLY_CROP2,
            value="population std of log(detJ) over the positive detJ voxels only",
            crop=2,
            ddof=0,
            mask="none",
            clamp="none",
            determinant=_CENTRAL_DETERMINANT,
            dtype="float64",
            fail_policy="drop non-finite and non-positive detJ, report kept_fraction and dropped_count",
            interpretation="conditional diagnostic; it is not the mathematical SDlogJ of the field",
        ),
        LEARN2REG_SHIFTED_SDLOGJ: _spec(
            LEARN2REG_SHIFTED_SDLOGJ,
            value="population std of log(clip(detJ + 3, 1e-9, 1e9))",
            crop=2,
            ddof=0,
            mask="none",
            clamp="clip(detJ + 3, 1e-9, 1e9) applied before the logarithm",
            determinant="det(I + grad u), central differences evaluated in float32, then float64 determinant",
            dtype="float32 displacement and derivatives; float64 identity, determinant, logarithm and reduction",
            fail_policy="none; the clamp keeps the value finite on a folded field",
            source="Learn2Reg evaluation/evaluation.py + evaluation/utils.py at 88475095e35442087a24439ff34ec09f54bc7dac",
            parity_scope="exact convention for CTCF's stored float32 displacement fields",
        ),
        LEARN2REG_SHIFTED_SDLOGJ_MASKED: _spec(
            LEARN2REG_SHIFTED_SDLOGJ_MASKED,
            value="population std of log(clip(detJ + 3, 1e-9, 1e9)) over the mask foreground",
            crop=2,
            ddof=0,
            mask="required binary {0,1} mask; foreground is cropped by the same 2 voxels",
            clamp="clip(detJ + 3, 1e-9, 1e9) applied before the logarithm",
            determinant="det(I + grad u), central differences evaluated in float32, then float64 determinant",
            dtype="float32 displacement and derivatives; float64 identity, determinant, logarithm and reduction",
            fail_policy="raise ValueError on a missing, mis-shaped or empty mask",
            source="optional masked branch of the same Learn2Reg evaluator",
        ),
        LEGACY_SHIFTED_J: _spec(
            LEGACY_SHIFTED_J,
            value="torch std of log(clamp(detJ + 3, 1e-9, 1e9)) over the whole volume",
            crop=0,
            ddof=1,
            mask="none",
            clamp="clamp(detJ + 3, 1e-9, 1e9) applied before the logarithm",
            determinant=_CENTRAL_DETERMINANT,
            dtype="float32",
            fail_policy="none; the clamp keeps the value finite on a folded field",
            source="compatibility with utils.field.logdet_std_from_flow; torch.std defaults to correction=1",
            reproducibility="bit-parity only within one device, torch build and backend; across CPU and CUDA "
            "compare within 1e-6 absolute at float32",
            runtime_provenance="torch version, input device and dtype are emitted in result components",
        ),
        DETJ_DIAGNOSTICS: _spec(
            DETJ_DIAGNOSTICS,
            value=None,
            crop=2,
            ddof=0,
            mask="none",
            clamp="none; the non-positive count is taken before the +3 shift and before any clamp",
            determinant=_CENTRAL_DETERMINANT,
            dtype="float64",
            fail_policy="non-finite and finite non-positive determinants are counted separately; finite-only "
            "summaries are NaN when no finite determinant exists",
            quantiles=DETJ_QUANTILES,
            compression_thresholds=COMPRESSION_THRESHOLDS,
            expansion_thresholds=EXPANSION_THRESHOLDS,
        ),
        DIGITAL_DECOMPOSITION: _spec(
            DIGITAL_DECOMPOSITION,
            value="percent of interior voxels violating at least one of the ten determinants",
            crop=1,
            ddof="not applicable",
            mask="none",
            clamp="none",
            determinant="ten one-sided determinants of Liu et al., IJCV 2024: eight corner tetrahedra plus J1*, J2*",
            dtype="float64",
            fail_policy="every non-finite determinant is a violation (fail-closed)",
            determinant_names=DETERMINANT_NAMES,
            aggregation="union of the ten violation maps, never the sum of the component fractions",
        ),
    }
)


def _validate_flow(flow: torch.Tensor) -> None:
    if flow.dim() != 5 or flow.shape[0] != 1 or flow.shape[1] != 3:
        raise ValueError(f"Expected a single displacement field of shape [1,3,D,H,W], got {tuple(flow.shape)}.")
    if min(flow.shape[2:]) < MIN_SPATIAL_SIZE:
        raise ValueError(f"Every spatial size must be at least {MIN_SPATIAL_SIZE}, got {tuple(flow.shape[2:])}.")


def _foreground(mask: torch.Tensor | None, spatial: tuple[int, ...]) -> np.ndarray:
    if mask is None:
        raise ValueError("This metric id requires an explicit mask.")
    squeezed = mask.detach()
    while squeezed.dim() > 3:
        if squeezed.shape[0] != 1:
            raise ValueError(f"Expected a single mask, got leading size {squeezed.shape[0]}.")
        squeezed = squeezed[0]
    if squeezed.dim() != 3 or tuple(squeezed.shape) != spatial:
        raise ValueError(f"Mask shape {tuple(squeezed.shape)} does not match the field spatial shape {spatial}.")
    values = squeezed.cpu().numpy()
    if not np.isfinite(values).all() or not np.isin(values, (0, 1)).all():
        raise ValueError("Mask must be finite and binary with values in {0,1}.")
    foreground = values.astype(bool, copy=False)
    if not foreground.any():
        raise ValueError("Mask foreground is empty.")
    return foreground


def central_jacobian_determinant(flow: torch.Tensor, crop: int = 0) -> np.ndarray:
    """detJ = det(I + grad u) in float64, cropped by `crop` voxels on every spatial side."""
    _validate_flow(flow)
    det = jacobian_det(flow.detach().to(torch.float64))[0, 0].cpu().numpy()
    return det if crop == 0 else det[crop:-crop, crop:-crop, crop:-crop]


def _mathematical_sdlogj(flow: torch.Tensor, crop: int) -> tuple[float, dict[str, float]]:
    detj = central_jacobian_determinant(flow, crop)
    finite = np.isfinite(detj)
    non_finite = int((~finite).sum())
    non_positive = int((finite & (detj <= 0.0)).sum())
    if non_finite or non_positive:
        raise MetricFailClosedError(
            f"log(detJ) is undefined on this field: {non_finite} non-finite and {non_positive} non-positive "
            f"of {detj.size} determinants at crop {crop}."
        )
    return float(np.log(detj).std(ddof=0)), {"voxels": float(detj.size)}


def _positive_only_sdlogj(flow: torch.Tensor) -> tuple[float, dict[str, float]]:
    detj = central_jacobian_determinant(flow, crop=2)
    keep = np.isfinite(detj) & (detj > 0.0)
    if keep.sum() < 1:
        raise MetricFailClosedError("No positive finite determinant remains; the diagnostic is undefined.")
    components = {
        "voxels": float(detj.size),
        "kept_count": float(keep.sum()),
        "kept_fraction": float(keep.mean()),
        "dropped_count": float(detj.size - keep.sum()),
    }
    return float(np.log(detj[keep]).std(ddof=0)), components


def learn2reg_jacobian_determinant(flow: torch.Tensor) -> np.ndarray:
    """Clean-room implementation of the frozen upstream FP32 derivative convention."""
    _validate_flow(flow)
    displacement = flow.detach().cpu().numpy()[0].astype(np.float32, copy=False)
    shape = tuple(size - 4 for size in displacement.shape[1:])
    gradient = np.empty((3, 3, *shape), dtype=np.float32)
    for axis in range(3):
        forward = [slice(2, -2), slice(2, -2), slice(2, -2)]
        backward = list(forward)
        forward[axis] = slice(3, -1)
        backward[axis] = slice(1, -3)
        gradient[:, axis] = np.float32(0.5) * (
            displacement[(slice(None), *forward)] - displacement[(slice(None), *backward)]
        )
    matrix = gradient + np.eye(3, dtype=np.float64).reshape(3, 3, 1, 1, 1)
    return (
        matrix[0, 0] * (matrix[1, 1] * matrix[2, 2] - matrix[1, 2] * matrix[2, 1])
        - matrix[1, 0] * (matrix[0, 1] * matrix[2, 2] - matrix[0, 2] * matrix[2, 1])
        + matrix[2, 0] * (matrix[0, 1] * matrix[1, 2] - matrix[0, 2] * matrix[1, 1])
    )


def _shifted_log_detj(flow: torch.Tensor) -> np.ndarray:
    detj = learn2reg_jacobian_determinant(flow)
    return np.log(np.clip(detj + SHIFT, CLAMP_LOW, CLAMP_HIGH))


def _learn2reg_sdlogj(flow: torch.Tensor) -> tuple[float, dict[str, float]]:
    shifted = _shifted_log_detj(flow)
    return float(shifted.std(ddof=0)), {"voxels": float(shifted.size)}


def _learn2reg_sdlogj_masked(flow: torch.Tensor, mask: torch.Tensor | None) -> tuple[float, dict[str, float]]:
    _validate_flow(flow)
    foreground = _foreground(mask, tuple(flow.shape[2:]))[2:-2, 2:-2, 2:-2]
    if not foreground.any():
        raise ValueError("Mask foreground is empty inside the 2-voxel-cropped interior.")
    shifted = _shifted_log_detj(flow)[foreground]
    return float(shifted.std(ddof=0)), {"voxels": float(shifted.size)}


def _legacy_shifted_j(flow: torch.Tensor) -> tuple[float, dict[str, Any]]:
    """Compatibility wrapper for utils.field.logdet_std_from_flow, restricted to one field."""
    _validate_flow(flow)
    shifted = torch.clamp(jacobian_det(flow.detach().float()) + SHIFT, min=CLAMP_LOW, max=CLAMP_HIGH)
    components: dict[str, Any] = {
        "voxels": float(shifted.numel()),
        "torch_version": torch.__version__,
        "input_device": str(flow.device),
        "input_dtype": str(flow.dtype),
        "evaluation_device": str(shifted.device),
    }
    return float(torch.std(torch.log(shifted)).item()), components


def _detj_diagnostics(flow: torch.Tensor) -> tuple[None, dict[str, float]]:
    detj = central_jacobian_determinant(flow, crop=2)
    finite_mask = np.isfinite(detj)
    finite = detj[finite_mask]
    positive = detj[finite_mask & (detj > 0.0)]
    nonpositive_mask = finite_mask & (detj <= 0.0)
    components: dict[str, float] = {
        "voxels": float(detj.size),
        "finite_count": float(finite.size),
        "detj_min": float(finite.min()) if finite.size else float("nan"),
        "detj_max": float(finite.max()) if finite.size else float("nan"),
        "nonfinite_count": float((~finite_mask).sum()),
        "nonfinite_fraction": float((~finite_mask).mean()),
        "nonpositive_count": float(nonpositive_mask.sum()),
        "nonpositive_fraction": float(nonpositive_mask.mean()),
        "invalid_count": float((~finite_mask).sum() + nonpositive_mask.sum()),
        "invalid_fraction": float(((~finite_mask) | nonpositive_mask).mean()),
    }
    for level in DETJ_QUANTILES:
        components[f"detj_quantile_{level}"] = float(np.quantile(finite, level)) if finite.size else float("nan")
    for threshold in COMPRESSION_THRESHOLDS:
        components[f"compression_fraction_detj_below_{threshold}"] = (
            float((finite < threshold).mean()) if finite.size else float("nan")
        )
    for threshold in EXPANSION_THRESHOLDS:
        components[f"expansion_fraction_detj_above_{threshold}"] = (
            float((finite > threshold).mean()) if finite.size else float("nan")
        )
    components["volume_distortion_energy_mean_squared_detj_minus_one"] = (
        float(np.square(finite - 1.0).mean()) if finite.size else float("nan")
    )
    components["volume_distortion_energy_mean_squared_log_detj_positive_only"] = float(
        np.square(np.log(positive)).mean() if positive.size else float("nan")
    )
    components["volume_distortion_energy_positive_fraction"] = float(positive.size / detj.size)
    return None, components


def _one_sided_difference(transform: np.ndarray, axis: int, mode: str) -> np.ndarray:
    size = transform.shape[axis + 1]
    index = np.arange(size)
    shifted = np.clip(index + 1, None, size - 1) if mode == "+" else np.clip(index - 1, 0, None)
    taken = np.take(transform, shifted, axis=axis + 1)
    return taken - transform if mode == "+" else transform - taken


def _offset_difference(transform: np.ndarray, offset: tuple[int, int, int]) -> np.ndarray:
    shifted = transform
    for axis, step in enumerate(offset):
        if step == 0:
            continue
        size = shifted.shape[axis + 1]
        shifted = np.take(shifted, np.clip(np.arange(size) + step, 0, size - 1), axis=axis + 1)
    return shifted - transform


def _determinant_of(first: np.ndarray, second: np.ndarray, third: np.ndarray) -> np.ndarray:
    return (
        first[0] * (second[1] * third[2] - second[2] * third[1])
        - first[1] * (second[0] * third[2] - second[2] * third[0])
        + first[2] * (second[0] * third[1] - second[1] * third[0])
    )


def _ten_determinants(flow: torch.Tensor) -> list[np.ndarray]:
    """The eight corner determinants and J1*, J2*, each on the [1:-1] interior, in float64."""
    _validate_flow(flow)
    displacement = flow.detach().to(torch.float64).cpu().numpy()[0]
    grid = np.stack(np.meshgrid(*[np.arange(n, dtype=np.float64) for n in displacement.shape[1:]], indexing="ij"))
    transform = displacement + grid
    interior = (slice(1, -1),) * 3
    determinants = [
        _determinant_of(*(_one_sided_difference(transform, axis, mode) for axis, mode in enumerate(modes)))[interior]
        for modes in _AXIS_MODES
    ]
    determinants.extend(
        _determinant_of(*(_offset_difference(transform, offset) for offset in offsets))[interior]
        for offsets in _STAR_OFFSETS
    )
    return determinants


def _digital_decomposition(flow: torch.Tensor) -> tuple[float, dict[str, float]]:
    determinants = _ten_determinants(flow)
    violations = [(~np.isfinite(determinant)) | (determinant <= 0.0) for determinant in determinants]
    components: dict[str, float] = {"voxels": float(determinants[0].size)}
    for name, determinant, violation in zip(DETERMINANT_NAMES, determinants, violations, strict=True):
        finite = determinant[np.isfinite(determinant)]
        components[f"{name}_min"] = float(finite.min()) if finite.size else float("nan")
        components[f"{name}_nonfinite_count"] = float((~np.isfinite(determinant)).sum())
        components[f"{name}_violation_count"] = float(violation.sum())
        components[f"{name}_violation_fraction"] = float(violation.mean())
    corner_union = np.logical_or.reduce(violations[:8])
    jstar_union = np.logical_or.reduce(violations[8:])
    union = corner_union | jstar_union
    components["corner_union_violation_fraction"] = float(corner_union.mean())
    components["jstar_union_violation_fraction"] = float(jstar_union.mean())
    components["union_violation_count"] = float(union.sum())
    components["union_violation_fraction"] = float(union.mean())
    components["sum_of_component_fractions"] = float(sum(float(v.mean()) for v in violations))
    return float(union.mean() * 100.0), components


def compute_metric(metric_id: str, flow: torch.Tensor, mask: torch.Tensor | None = None) -> MetricResult:
    """Evaluate one named metric on one displacement field. An unknown id is refused, never guessed."""
    if metric_id not in METRIC_SPECS:
        raise ValueError(f"Unknown metric id {metric_id!r}; choose one of {sorted(METRIC_SPECS)}.")
    _validate_flow(flow)
    masked_metric = metric_id == LEARN2REG_SHIFTED_SDLOGJ_MASKED
    if mask is not None and not masked_metric:
        raise ValueError(f"Metric {metric_id} is unmasked and refuses an unexpected mask.")
    if metric_id == MATHEMATICAL_SDLOGJ_FULL:
        value, components = _mathematical_sdlogj(flow, crop=0)
    elif metric_id == MATHEMATICAL_SDLOGJ_CROP2:
        value, components = _mathematical_sdlogj(flow, crop=2)
    elif metric_id == DIAGNOSTIC_SDLOGJ_POSITIVE_ONLY_CROP2:
        value, components = _positive_only_sdlogj(flow)
    elif metric_id == LEARN2REG_SHIFTED_SDLOGJ:
        value, components = _learn2reg_sdlogj(flow)
    elif metric_id == LEARN2REG_SHIFTED_SDLOGJ_MASKED:
        value, components = _learn2reg_sdlogj_masked(flow, mask)
    elif metric_id == LEGACY_SHIFTED_J:
        value, components = _legacy_shifted_j(flow)
    elif metric_id == DETJ_DIAGNOSTICS:
        value, components = _detj_diagnostics(flow)
    else:
        value, components = _digital_decomposition(flow)
    return MetricResult(metric_id=metric_id, value=value, components=components, metadata=METRIC_SPECS[metric_id])


def compute_bundle(flow: torch.Tensor, mask: torch.Tensor | None = None) -> dict[str, MetricResult]:
    """Every metric that this input supports; without a mask the masked ids are skipped.
    Fail-closed metrics propagate MetricFailClosedError rather than returning a substitute number.
    """
    return {
        metric_id: compute_metric(metric_id, flow, mask if metric_id == LEARN2REG_SHIFTED_SDLOGJ_MASKED else None)
        for metric_id, spec in METRIC_SPECS.items()
        if mask is not None or not str(spec["mask"]).startswith("required")
    }
