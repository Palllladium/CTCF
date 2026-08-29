from __future__ import annotations

import csv
import hashlib
import io
import math
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from datasets.OASIS100 import stage5_sha256_array
from tools.analysis.run_artifacts import atomic_write_text, sha256_file
from tools.analysis.search.metrics import (
    DETJ_DIAGNOSTICS,
    LEARN2REG_SHIFTED_SDLOGJ_MASKED,
    METRIC_SPECS,
    MetricFailClosedError,
    compute_metric,
)
from tools.analysis.search.pyramid import array_sha256
from tools.analysis.search.transaction import load_flow_npz, sample_at_psi
from tools.analysis.stage5.contracts import (
    BASE_SEEDS,
    EVALUATION_RECORD_SCHEMA,
    PLANNED_CONTRASTS,
    VARIANT_IDS,
    canonical_json_bytes,
    canonical_sha256,
    validate_decision_barrier,
    validate_file_record,
    validate_protocol_contract,
    validate_training_barrier,
)
from utils.dice import OASIS_VOI_LABELS, dice_per_label

EVALUATION_SCHEMA = "ctcf-stage5-returned-field-evaluation-v1"
PAIR_EVALUATION_SCHEMA = "ctcf-stage5-pair-evaluation-v1"
AGGREGATE_SCHEMA = "ctcf-stage5-paired-aggregate-v1"

DICE_MEAN_METRIC_ID = "OASIS_DICE_MEAN_LABELS_01_TO_35_PSI_NEAREST_ALIGN_FALSE_V1"
DICE_LABEL_METRIC_IDS = tuple(f"OASIS_DICE_LABEL_{label:02d}_PSI_NEAREST_ALIGN_FALSE_V1" for label in OASIS_VOI_LABELS)
INVERSE_COMPONENT_RMS_METRIC_ID = "CTCF_INVERSE_CONSISTENCY_BIDIRECTIONAL_FULL_COMPONENT_RMS_VOXEL_V1"
INVERSE_MAX_VECTOR_METRIC_ID = "CTCF_INVERSE_CONSISTENCY_BIDIRECTIONAL_FULL_MAX_VECTOR_L2_VOXEL_V1"

GEOMETRY_METRIC_IDS = tuple(METRIC_SPECS)
GEOMETRY_SCALAR_METRIC_IDS = tuple(metric_id for metric_id in GEOMETRY_METRIC_IDS if metric_id != DETJ_DIAGNOSTICS)
STAGE5_EVALUATION_METRIC_IDS = (
    DICE_MEAN_METRIC_ID,
    *DICE_LABEL_METRIC_IDS,
    *GEOMETRY_METRIC_IDS,
    INVERSE_COMPONENT_RMS_METRIC_ID,
    INVERSE_MAX_VECTOR_METRIC_ID,
)
EFFECT_METRIC_IDS = (
    DICE_MEAN_METRIC_ID,
    *DICE_LABEL_METRIC_IDS,
    *GEOMETRY_SCALAR_METRIC_IDS,
    INVERSE_COMPONENT_RMS_METRIC_ID,
    INVERSE_MAX_VECTOR_METRIC_ID,
)
PAIR_RUNTIME_METRIC_ID = "STAGE5_DECISION_PAIR_RUNTIME_SECONDS_SUM_V1"
PAIR_PEAK_MEMORY_METRIC_ID = "STAGE5_DECISION_PAIR_PEAK_MEMORY_BYTES_MAX_V1"
PAIR_REQUESTED_DELTA_RMS_METRIC_ID = "STAGE5_DECISION_PAIR_REQUESTED_DELTA_RMS_MEAN_V1"
PAIR_CANDIDATE_DELTA_RMS_METRIC_ID = "STAGE5_DECISION_PAIR_CANDIDATE_DELTA_RMS_MEAN_V1"
PAIR_RETURNED_DELTA_RMS_METRIC_ID = "STAGE5_DECISION_PAIR_RETURNED_DELTA_RMS_MEAN_V1"
PAIR_CANDIDATE_RETENTION_METRIC_ID = "STAGE5_DECISION_PAIR_CANDIDATE_RETAINED_RATIO_MEAN_V1"
PAIR_RETURNED_RETENTION_METRIC_ID = "STAGE5_DECISION_PAIR_RETURNED_RETAINED_RATIO_MEAN_V1"
PAIR_DIAGNOSTIC_IDS = (
    PAIR_RUNTIME_METRIC_ID,
    PAIR_PEAK_MEMORY_METRIC_ID,
    PAIR_REQUESTED_DELTA_RMS_METRIC_ID,
    PAIR_CANDIDATE_DELTA_RMS_METRIC_ID,
    PAIR_RETURNED_DELTA_RMS_METRIC_ID,
    PAIR_CANDIDATE_RETENTION_METRIC_ID,
    PAIR_RETURNED_RETENTION_METRIC_ID,
)
PAIRED_DIAGNOSTIC_EFFECT_IDS = (PAIR_RUNTIME_METRIC_ID, PAIR_PEAK_MEMORY_METRIC_ID)

WARP_CONVENTION = {
    "operator": "tools.analysis.search.transaction.sample_at_psi",
    "representation": "Psi source-index displacement in z-y-x voxel units",
    "interpolation": "nearest",
    "padding_mode": "zeros",
    "align_corners": False,
    "coordinate_normalization": "2 * (coordinate + 0.5) / size - 1",
}
BOOTSTRAP_DOMAIN = "CTCF-STAGE5-PAIRED-BLOCK-BOOTSTRAP-V1"
BOOTSTRAP_ALGORITHM = "NumPy Generator PCG64 multinomial counts; exactly n unordered pair-ID draws with replacement"
BOOTSTRAP_ITERATIONS = 10_000
BOOTSTRAP_CI_PERCENT = 95.0
BOOTSTRAP_PERCENTILE_METHOD = "linear"
SIMULTANEOUS_CI_METHOD = "max absolute centered paired-bootstrap error"
SIMULTANEOUS_FAMILY_PRIMARY = "STAGE5_PRIMARY_MEAN_DICE_11_PREREGISTERED_CONTRASTS_V1"
SIMULTANEOUS_FAMILY_REGIONAL = "STAGE5_REGIONAL_DICE_11_CONTRASTS_X_35_LABELS_V1"


@dataclass(frozen=True, slots=True)
class EvaluationContext:
    protocol: dict[str, Any]
    training_barrier: dict[str, Any]
    decision_barrier: dict[str, Any]
    cases: dict[str, dict[str, str]]
    pairs: dict[str, tuple[str, str]]
    decisions: dict[str, dict[str, Any]]
    protocol_sha256: str
    training_barrier_sha256: str
    decision_barrier_sha256: str

    @classmethod
    def from_barriers(
        cls,
        protocol: Mapping[str, Any],
        training_barrier: Mapping[str, Any],
        decision_barrier: Mapping[str, Any],
        case_inventory: Sequence[Mapping[str, Any]],
    ) -> EvaluationContext:
        frozen_protocol = dict(protocol)
        frozen_training = dict(training_barrier)
        frozen_decision = dict(decision_barrier)
        validate_protocol_contract(frozen_protocol)
        validate_training_barrier(frozen_training, frozen_protocol, require_complete=True)
        validate_decision_barrier(
            frozen_decision,
            frozen_protocol,
            frozen_training,
            require_complete=True,
        )
        if tuple(frozen_protocol["metric_ids"]) != STAGE5_EVALUATION_METRIC_IDS:
            raise RuntimeError("Stage 5 evaluation metric inventory is not the frozen full bundle")

        expected_case_ids = tuple(str(value) for value in frozen_protocol["directed_case_ids"])
        cases: dict[str, dict[str, str]] = {}
        grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
        required = {"case_id", "pair_id", "moving_subject_id", "fixed_subject_id"}
        for raw in case_inventory:
            if set(raw) != required or any(not isinstance(raw[key], str) or not raw[key] for key in required):
                raise RuntimeError("Stage 5 case inventory schema changed")
            item = {key: str(raw[key]) for key in required}
            case_id = item["case_id"]
            if case_id in cases:
                raise RuntimeError("Duplicate Stage 5 directed case")
            if item["moving_subject_id"] == item["fixed_subject_id"]:
                raise RuntimeError("Stage 5 directed case cannot register a subject to itself")
            cases[case_id] = item
            grouped[item["pair_id"]].append(item)
        if tuple(cases) != expected_case_ids:
            raise RuntimeError("Case inventory order differs from the frozen Stage 5 protocol")

        pairs: dict[str, tuple[str, str]] = {}
        for pair_id, members in grouped.items():
            if len(members) != 2:
                raise RuntimeError(f"Pair {pair_id} must contain exactly two directed cases")
            first, second = members
            if (
                first["moving_subject_id"] != second["fixed_subject_id"]
                or first["fixed_subject_id"] != second["moving_subject_id"]
            ):
                raise RuntimeError(f"Pair {pair_id} does not contain exact reverse directions")
            pairs[pair_id] = (first["case_id"], second["case_id"])

        decisions = {str(item["decision_id"]): dict(item) for item in frozen_decision["records"]}
        if len(decisions) != len(frozen_decision["records"]):
            raise RuntimeError("Duplicate decision ID survived the Stage 5 barrier")
        return cls(
            protocol=frozen_protocol,
            training_barrier=frozen_training,
            decision_barrier=frozen_decision,
            cases=cases,
            pairs=pairs,
            decisions=decisions,
            protocol_sha256=canonical_sha256(frozen_protocol),
            training_barrier_sha256=canonical_sha256(frozen_training),
            decision_barrier_sha256=canonical_sha256(frozen_decision),
        )


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(child) for child in value]
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, bool) or value is None or isinstance(value, (str, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise FloatingPointError("metric output contains a non-finite scalar")
        return value
    raise TypeError(f"Unsupported evaluation payload type: {type(value).__name__}")


def _validate_label(label: np.ndarray, name: str, shape: tuple[int, int, int]) -> np.ndarray:
    if not isinstance(label, np.ndarray) or label.dtype != np.uint8 or label.ndim != 3:
        raise TypeError(f"{name} must be a uint8 [D,H,W] OASIS segmentation")
    if tuple(label.shape) != shape:
        raise ValueError(f"{name} shape {tuple(label.shape)} differs from returned field shape {shape}")
    values = np.unique(label)
    if values.size and (int(values[0]) < 0 or int(values[-1]) > max(OASIS_VOI_LABELS)):
        raise ValueError(f"{name} contains labels outside frozen OASIS values 0..35")
    return np.ascontiguousarray(label)


def _load_verified_field(
    path: Path,
    decision: Mapping[str, Any],
    record_key: str,
    label: str,
) -> torch.Tensor:
    path = Path(path)
    if not path.exists() or not path.is_file() or path.is_symlink():
        raise FileNotFoundError(f"Missing regular {label} displacement: {path}")
    record = decision[record_key]
    if path.stat().st_size != record["bytes"] or sha256_file(path) != record["sha256"]:
        raise RuntimeError(f"{label.capitalize()} displacement bytes differ from the COMPLETE decision barrier")
    field = load_flow_npz(path)
    if array_sha256(field) != record["array_sha256"]:
        raise RuntimeError(f"{label.capitalize()} displacement array differs from the COMPLETE decision barrier")
    if not bool(torch.isfinite(field).all()):
        raise FloatingPointError(f"{label.capitalize()} displacement is non-finite")
    return field


def _metric_ok(metric_id: str, value: float, *, metadata: Mapping[str, Any] | None = None) -> dict[str, Any]:
    result: dict[str, Any] = {"metric_id": metric_id, "status": "OK", "value": float(value)}
    if metadata is not None:
        result["metadata"] = _json_safe(metadata)
    return result


def _metric_error(metric_id: str, error: BaseException, metadata: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "metric_id": metric_id,
        "status": "ERROR",
        "value": None,
        "error": {
            "error_type": type(error).__name__,
            "message": str(error),
        },
        "metadata": _json_safe(metadata),
    }


def compute_geometry_bundle(field: torch.Tensor, fixed_label: torch.Tensor) -> dict[str, dict[str, Any]]:
    mask = (fixed_label > 0).to(dtype=torch.uint8)
    bundle: dict[str, dict[str, Any]] = {}
    for metric_id in GEOMETRY_METRIC_IDS:
        spec = dict(METRIC_SPECS[metric_id])
        try:
            result = compute_metric(
                metric_id,
                field,
                mask if metric_id == LEARN2REG_SHIFTED_SDLOGJ_MASKED else None,
            )
            payload = {
                "metric_id": result.metric_id,
                "status": "OK",
                "value": result.value,
                "components": result.components,
                "metadata": dict(result.metadata),
            }
            bundle[metric_id] = _json_safe(payload)
        except (MetricFailClosedError, ValueError, FloatingPointError) as exc:
            bundle[metric_id] = _metric_error(metric_id, exc, spec)
    return bundle


def evaluate_returned_decision(
    context: EvaluationContext,
    decision_id: str,
    returned_field_path: Path,
    moving_label: np.ndarray,
    fixed_label: np.ndarray,
    *,
    requested_field_path: Path,
    candidate_field_path: Path,
    device: str | torch.device = "cpu",
) -> dict[str, Any]:
    """Score one decision's returned field, plus its requested and candidate stages.

    All three paths are required: each stage's diagnostics are published under that
    stage's field record, so substituting one field for another would label a metric
    with a digest it was not computed from.
    """
    try:
        decision = context.decisions[decision_id]
    except KeyError as exc:
        raise RuntimeError("Evaluation request is outside the COMPLETE decision barrier") from exc
    case = context.cases[decision["case_id"]]
    field_cpu = _load_verified_field(returned_field_path, decision, "returned_field", "returned")
    requested_cpu = _load_verified_field(requested_field_path, decision, "requested_field", "requested")
    candidate_cpu = _load_verified_field(candidate_field_path, decision, "candidate_field", "candidate")
    shape = tuple(int(value) for value in field_cpu.shape[-3:])
    moving = _validate_label(moving_label, "moving_label", shape)
    fixed = _validate_label(fixed_label, "fixed_label", shape)
    moving_sha = stage5_sha256_array(moving)
    fixed_sha = stage5_sha256_array(fixed)
    label_source_sha = canonical_sha256(
        {
            "case_id": case["case_id"],
            "moving_subject_id": case["moving_subject_id"],
            "moving_label_array_sha256": moving_sha,
            "fixed_subject_id": case["fixed_subject_id"],
            "fixed_label_array_sha256": fixed_sha,
        }
    )

    resolved_device = torch.device(device)
    field = field_cpu.to(resolved_device)
    moving_tensor = torch.from_numpy(moving).unsqueeze(0).unsqueeze(0).to(resolved_device)
    fixed_tensor = torch.from_numpy(fixed).unsqueeze(0).unsqueeze(0).to(resolved_device)
    with torch.inference_mode():
        warped = sample_at_psi(moving_tensor.float(), field, mode="nearest").long()
        per_label = dice_per_label(warped, fixed_tensor.long(), OASIS_VOI_LABELS)
        mean_dice = float(per_label.mean())
        geometry = compute_geometry_bundle(field, fixed_tensor)
    if not np.isfinite(per_label).all() or not math.isfinite(mean_dice):
        raise FloatingPointError("OASIS Dice produced a non-finite result")

    dice_metadata = {
        "labels": list(OASIS_VOI_LABELS),
        "statistic": "2*intersection/(prediction_count+target_count+1e-5)",
        "absent_in_both": 0.0,
        "warp": WARP_CONVENTION,
    }
    dice_labels = [
        {
            "label": label,
            **_metric_ok(metric_id, float(value), metadata=dice_metadata),
        }
        for label, metric_id, value in zip(OASIS_VOI_LABELS, DICE_LABEL_METRIC_IDS, per_label, strict=True)
    ]
    stage_diagnostics: dict[str, dict[str, Any]] = {}
    stage_fields = {
        "requested": (requested_cpu, decision["requested_field"]),
        "candidate": (candidate_cpu, decision["candidate_field"]),
        "returned": (field_cpu, decision["returned_field"]),
    }
    cached_stage_metrics: dict[str, tuple[float, dict[str, dict[str, Any]]]] = {
        str(decision["returned_field"]["array_sha256"]): (mean_dice, geometry)
    }
    with torch.inference_mode():
        for stage, (stage_cpu, record) in stage_fields.items():
            digest = str(record["array_sha256"])
            if digest not in cached_stage_metrics:
                stage_field = stage_cpu.to(resolved_device)
                stage_warped = sample_at_psi(moving_tensor.float(), stage_field, mode="nearest").long()
                stage_per_label = dice_per_label(stage_warped, fixed_tensor.long(), OASIS_VOI_LABELS)
                stage_mean = float(stage_per_label.mean())
                if not np.isfinite(stage_per_label).all() or not math.isfinite(stage_mean):
                    raise FloatingPointError(f"OASIS Dice produced a non-finite {stage} diagnostic")
                cached_stage_metrics[digest] = (
                    stage_mean,
                    compute_geometry_bundle(stage_field, fixed_tensor),
                )
            stage_mean, stage_geometry = cached_stage_metrics[digest]
            stage_diagnostics[stage] = {
                "field": dict(record),
                "mean_dice": _metric_ok(DICE_MEAN_METRIC_ID, stage_mean, metadata=dice_metadata),
                "geometry": stage_geometry,
            }
    runtime = {
        "torch_version": torch.__version__,
        "device": str(resolved_device),
        "field_dtype": str(field.dtype),
        "label_dtype": str(moving_tensor.dtype),
    }
    execution_sha = canonical_sha256(
        {
            "protocol_sha256": context.protocol_sha256,
            "training_barrier_sha256": context.training_barrier_sha256,
            "decision_barrier_sha256": context.decision_barrier_sha256,
            "decision_record_sha256": canonical_sha256(decision),
            "returned_field_sha256": decision["returned_field"]["sha256"],
            "requested_field_sha256": decision["requested_field"]["sha256"],
            "candidate_field_sha256": decision["candidate_field"]["sha256"],
            "moving_label_array_sha256": moving_sha,
            "fixed_label_array_sha256": fixed_sha,
            "metric_ids": list(STAGE5_EVALUATION_METRIC_IDS),
            "warp": WARP_CONVENTION,
            "runtime": runtime,
        }
    )
    payload = {
        "schema": EVALUATION_SCHEMA,
        "evaluation_id": decision_id,
        "decision_id": decision_id,
        "case_id": case["case_id"],
        "pair_id": case["pair_id"],
        "moving_subject_id": case["moving_subject_id"],
        "fixed_subject_id": case["fixed_subject_id"],
        "seed": decision["seed"],
        "variant_id": decision["variant_id"],
        "transaction_status": decision["transaction_status"],
        "decision_diagnostics": {
            "runtime_seconds": decision["runtime_seconds"],
            "peak_memory_bytes": decision["peak_memory_bytes"],
            "requested_delta_rms": decision["requested_delta_rms"],
            "candidate_delta_rms": decision["candidate_delta_rms"],
            "returned_delta_rms": decision["returned_delta_rms"],
            "candidate_retained_ratio": decision["candidate_retained_ratio"],
            "returned_retained_ratio": decision["returned_retained_ratio"],
        },
        "protocol_sha256": context.protocol_sha256,
        "training_barrier_sha256": context.training_barrier_sha256,
        "decision_barrier_sha256": context.decision_barrier_sha256,
        "decision_record_sha256": canonical_sha256(decision),
        "returned_field": dict(decision["returned_field"]),
        "requested_field": dict(decision["requested_field"]),
        "candidate_field": dict(decision["candidate_field"]),
        "labels": {
            "loaded_after_complete_decision_barrier": True,
            "heldout_test_accessed": False,
            "moving_label_array_sha256": moving_sha,
            "fixed_label_array_sha256": fixed_sha,
            "label_source_sha256": label_source_sha,
        },
        "warp": dict(WARP_CONVENTION),
        "metrics": {
            "mean_dice": _metric_ok(DICE_MEAN_METRIC_ID, mean_dice, metadata=dice_metadata),
            "per_label_dice": dice_labels,
            "geometry": geometry,
        },
        "field_stage_diagnostics": {
            "role": "PREREGISTERED_INTERPRETATION_DIAGNOSTIC_NOT_A_SELECTION_RULE",
            "regional_dice_included": False,
            "stages": stage_diagnostics,
        },
        "runtime": runtime,
        "execution_sha256": execution_sha,
    }
    canonical_json_bytes(payload)
    return payload


def build_evaluation_record(
    evaluation: Mapping[str, Any],
    metrics_file: Mapping[str, Any],
) -> dict[str, Any]:
    if evaluation.get("schema") != EVALUATION_SCHEMA:
        raise RuntimeError("Unknown Stage 5 evaluation payload")
    validate_file_record(metrics_file, label="metrics_file")
    return {
        "schema": EVALUATION_RECORD_SCHEMA,
        "evaluation_id": evaluation["evaluation_id"],
        "decision_id": evaluation["decision_id"],
        "decision_record_sha256": evaluation["decision_record_sha256"],
        "returned_field_sha256": evaluation["returned_field"]["sha256"],
        "label_source_sha256": evaluation["labels"]["label_source_sha256"],
        "metrics_file": dict(metrics_file),
        "labels_loaded_after_decision_barrier": True,
        "heldout_test_accessed": False,
        "execution_sha256": evaluation["execution_sha256"],
    }


def _inverse_metrics(first: torch.Tensor, second: torch.Tensor) -> dict[str, dict[str, Any]]:
    if first.shape != second.shape:
        raise ValueError("Reverse returned fields have different shapes")
    with torch.inference_mode():
        first_then_second = first + sample_at_psi(second, first)
        second_then_first = second + sample_at_psi(first, second)
        rms = torch.sqrt(0.5 * (first_then_second.square().mean() + second_then_first.square().mean())).item()
        first_max = torch.linalg.vector_norm(first_then_second, dim=1).amax().item()
        second_max = torch.linalg.vector_norm(second_then_first, dim=1).amax().item()
        maximum = max(first_max, second_max)
    metadata = {
        "composition": "first + sample_at_psi(second, first), evaluated in both orders",
        "warp": WARP_CONVENTION,
        "domain": "full grid",
    }
    return {
        INVERSE_COMPONENT_RMS_METRIC_ID: _metric_ok(
            INVERSE_COMPONENT_RMS_METRIC_ID,
            float(rms),
            metadata={**metadata, "statistic": "sqrt(mean of the two component-wise mean-squared residuals)"},
        ),
        INVERSE_MAX_VECTOR_METRIC_ID: _metric_ok(
            INVERSE_MAX_VECTOR_METRIC_ID,
            float(maximum),
            metadata={**metadata, "statistic": "maximum vector L2 residual across both composition orders"},
        ),
    }


def _direction_metric(evaluation: Mapping[str, Any], metric_id: str) -> Mapping[str, Any]:
    if metric_id == DICE_MEAN_METRIC_ID:
        return evaluation["metrics"]["mean_dice"]
    if metric_id in DICE_LABEL_METRIC_IDS:
        index = DICE_LABEL_METRIC_IDS.index(metric_id)
        return evaluation["metrics"]["per_label_dice"][index]
    return evaluation["metrics"]["geometry"][metric_id]


def build_pair_evaluation(
    context: EvaluationContext,
    first_evaluation: Mapping[str, Any],
    second_evaluation: Mapping[str, Any],
    first_returned_path: Path,
    second_returned_path: Path,
    *,
    device: str | torch.device = "cpu",
) -> dict[str, Any]:
    evaluations = sorted((dict(first_evaluation), dict(second_evaluation)), key=lambda item: item["case_id"])
    if any(item.get("schema") != EVALUATION_SCHEMA for item in evaluations):
        raise RuntimeError("Pair aggregation received an unknown evaluation schema")
    first, second = evaluations
    identity = (first["pair_id"], first["seed"], first["variant_id"])
    if (second["pair_id"], second["seed"], second["variant_id"]) != identity:
        raise RuntimeError("Pair aggregation mixed pair, seed, or variant")
    expected_cases = set(context.pairs[first["pair_id"]])
    if {first["case_id"], second["case_id"]} != expected_cases:
        raise RuntimeError("Inverse consistency requires both exact directions of one frozen pair")
    paths = {
        str(first_evaluation["decision_id"]): Path(first_returned_path),
        str(second_evaluation["decision_id"]): Path(second_returned_path),
    }
    resolved_device = torch.device(device)
    fields = [
        _load_verified_field(
            paths[item["decision_id"]],
            context.decisions[item["decision_id"]],
            "returned_field",
            "returned",
        ).to(resolved_device)
        for item in evaluations
    ]
    inverse = _inverse_metrics(fields[0], fields[1])

    scalar_metrics: dict[str, dict[str, Any]] = {}
    for metric_id in (DICE_MEAN_METRIC_ID, *DICE_LABEL_METRIC_IDS, *GEOMETRY_SCALAR_METRIC_IDS):
        direction_values = [_direction_metric(item, metric_id) for item in evaluations]
        errors = [item.get("error") for item in direction_values if item["status"] != "OK"]
        if errors:
            scalar_metrics[metric_id] = {
                "metric_id": metric_id,
                "status": "ERROR",
                "value": None,
                "error": {"error_type": "DIRECTION_METRIC_ERROR", "details": errors},
            }
        else:
            values = [item["value"] for item in direction_values]
            if any(value is None or not math.isfinite(float(value)) for value in values):
                raise FloatingPointError(f"Pair metric {metric_id} is non-scalar or non-finite")
            scalar_metrics[metric_id] = _metric_ok(metric_id, float(np.mean(values)))
    scalar_metrics.update(inverse)
    if tuple(scalar_metrics) != EFFECT_METRIC_IDS:
        raise RuntimeError("Pair scalar metric inventory changed")

    direction_diagnostics = [item["decision_diagnostics"] for item in evaluations]

    def mean_optional(key: str) -> float | None:
        values = [item[key] for item in direction_diagnostics]
        return None if any(value is None for value in values) else float(np.mean(values))

    statuses = [item["transaction_status"] for item in evaluations]
    pair_diagnostics = {
        PAIR_RUNTIME_METRIC_ID: float(sum(item["runtime_seconds"] for item in direction_diagnostics)),
        PAIR_PEAK_MEMORY_METRIC_ID: int(max(item["peak_memory_bytes"] for item in direction_diagnostics)),
        PAIR_REQUESTED_DELTA_RMS_METRIC_ID: mean_optional("requested_delta_rms"),
        PAIR_CANDIDATE_DELTA_RMS_METRIC_ID: mean_optional("candidate_delta_rms"),
        PAIR_RETURNED_DELTA_RMS_METRIC_ID: mean_optional("returned_delta_rms"),
        PAIR_CANDIDATE_RETENTION_METRIC_ID: mean_optional("candidate_retained_ratio"),
        PAIR_RETURNED_RETENTION_METRIC_ID: mean_optional("returned_retained_ratio"),
    }
    payload = {
        "schema": PAIR_EVALUATION_SCHEMA,
        "pair_id": first["pair_id"],
        "seed": first["seed"],
        "variant_id": first["variant_id"],
        "case_ids": [item["case_id"] for item in evaluations],
        "decision_ids": [item["decision_id"] for item in evaluations],
        "returned_field_sha256": [item["returned_field"]["sha256"] for item in evaluations],
        "transaction_statuses": statuses,
        "transaction_counts": dict(sorted(Counter(statuses).items())),
        "direction_diagnostics": direction_diagnostics,
        "pair_diagnostics": pair_diagnostics,
        "scalar_metrics": scalar_metrics,
        "inverse_consistency": inverse,
    }
    canonical_json_bytes(payload)
    return payload


def _bootstrap_seed(scope: str, variant_id: str) -> int:
    digest = hashlib.sha256(f"{BOOTSTRAP_DOMAIN}\0{scope}\0{variant_id}".encode()).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False)


def _bootstrap_matrix(
    values: np.ndarray,
    metric_ids: Sequence[str],
    *,
    scope: str,
    variant_id: str,
) -> dict[str, dict[str, Any]]:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] < 2 or values.shape[1] != len(metric_ids) or not np.isfinite(values).all():
        raise ValueError("Paired block bootstrap requires a finite [pair,metric] matrix with at least two pairs")
    rng = np.random.Generator(np.random.PCG64(_bootstrap_seed(scope, variant_id)))
    probabilities = np.full(values.shape[0], 1.0 / values.shape[0], dtype=np.float64)
    # Multinomial counts are exactly the counts induced by drawing n pair IDs with replacement.
    counts = rng.multinomial(values.shape[0], probabilities, size=BOOTSTRAP_ITERATIONS)
    sampled_means = counts @ values / float(values.shape[0])
    tail = (100.0 - BOOTSTRAP_CI_PERCENT) / 2.0
    low, high = np.percentile(
        sampled_means,
        (tail, 100.0 - tail),
        axis=0,
        method=BOOTSTRAP_PERCENTILE_METHOD,
    )
    return {
        metric_id: {
            "status": "OK",
            "n_unordered_pairs": int(values.shape[0]),
            "mean": float(values[:, index].mean()),
            "median": float(np.median(values[:, index])),
            "sample_std_ddof1": float(values[:, index].std(ddof=1)),
            "ci_low": float(low[index]),
            "ci_high": float(high[index]),
            "bootstrap": {
                "domain": BOOTSTRAP_DOMAIN,
                "algorithm": BOOTSTRAP_ALGORITHM,
                "iterations": BOOTSTRAP_ITERATIONS,
                "ci_percent": BOOTSTRAP_CI_PERCENT,
                "percentile_method": BOOTSTRAP_PERCENTILE_METHOD,
                "shared_pair_resamples_across_metrics": True,
                "seed": _bootstrap_seed(scope, variant_id),
            },
        }
        for index, metric_id in enumerate(metric_ids)
    }


def _simultaneous_family(
    values: np.ndarray,
    columns: Sequence[tuple[str, str]],
    pointwise: Mapping[tuple[str, str], Mapping[str, Any]],
    *,
    scope: str,
    family_id: str,
) -> tuple[dict[str, Any], dict[tuple[str, str], dict[str, Any]]]:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] < 2 or values.shape[1] != len(columns) or not np.isfinite(values).all():
        raise ValueError("Simultaneous paired intervals require a finite [pair,hypothesis] matrix")
    seed = _bootstrap_seed(f"simultaneous/{scope}/{family_id}", "ALL_CONTROLLERS")
    rng = np.random.Generator(np.random.PCG64(seed))
    probabilities = np.full(values.shape[0], 1.0 / values.shape[0], dtype=np.float64)
    counts = rng.multinomial(values.shape[0], probabilities, size=BOOTSTRAP_ITERATIONS)
    point = values.mean(axis=0)
    sampled = counts @ values / float(values.shape[0])
    max_error = np.abs(sampled - point).max(axis=1)
    critical = float(
        np.percentile(
            max_error,
            BOOTSTRAP_CI_PERCENT,
            method=BOOTSTRAP_PERCENTILE_METHOD,
        )
    )
    intervals: dict[tuple[str, str], dict[str, Any]] = {}
    for index, column in enumerate(columns):
        pointwise_summary = pointwise[column]
        # Expanding by the pointwise endpoints is conservative and makes the reported
        # simultaneous interval visibly no narrower even under finite bootstrap Monte Carlo error.
        low = min(float(point[index] - critical), float(pointwise_summary["ci_low"]))
        high = max(float(point[index] + critical), float(pointwise_summary["ci_high"]))
        intervals[column] = {
            "family_id": family_id,
            "method": SIMULTANEOUS_CI_METHOD,
            "ci_percent": BOOTSTRAP_CI_PERCENT,
            "point_estimate": float(point[index]),
            "ci_low": low,
            "ci_high": high,
            "familywise_critical_value": critical,
            "shared_pair_resample_seed": seed,
            "shared_pair_resamples_across_family": True,
        }
    family = {
        "family_id": family_id,
        "scope": scope,
        "status": "OK",
        "method": SIMULTANEOUS_CI_METHOD,
        "ci_percent": BOOTSTRAP_CI_PERCENT,
        "n_unordered_pairs": int(values.shape[0]),
        "n_hypotheses": int(values.shape[1]),
        "iterations": BOOTSTRAP_ITERATIONS,
        "percentile_method": BOOTSTRAP_PERCENTILE_METHOD,
        "shared_pair_resample_seed": seed,
        "familywise_critical_value": critical,
        "columns": [{"contrast_id": contrast_id, "metric_id": metric} for contrast_id, metric in columns],
    }
    return family, intervals


def _aggregate_decision_diagnostics(
    index: Mapping[tuple[str, int, str], Mapping[str, Any]],
    pair_ids: Sequence[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    absolute: list[dict[str, Any]] = []
    effects: list[dict[str, Any]] = []
    for variant_id in VARIANT_IDS:
        for scope, seeds in (
            *((f"seed_{seed}", (seed,)) for seed in BASE_SEEDS),
            ("seed_mean", BASE_SEEDS),
        ):
            vectors: dict[str, np.ndarray] = {}
            unavailable: dict[str, list[str]] = defaultdict(list)
            for diagnostic_id in PAIR_DIAGNOSTIC_IDS:
                values = np.empty(len(pair_ids), dtype=np.float64)
                for pair_index, pair_id in enumerate(pair_ids):
                    seed_values = [
                        index[(pair_id, seed, variant_id)]["pair_diagnostics"][diagnostic_id] for seed in seeds
                    ]
                    if any(value is None for value in seed_values):
                        unavailable[diagnostic_id].extend(f"{pair_id}/S{seed}" for seed in seeds)
                        values[pair_index] = np.nan
                    else:
                        values[pair_index] = float(np.mean(seed_values))
                vectors[diagnostic_id] = values
            available = [diagnostic_id for diagnostic_id in PAIR_DIAGNOSTIC_IDS if diagnostic_id not in unavailable]
            summaries = _bootstrap_matrix(
                np.column_stack([vectors[diagnostic_id] for diagnostic_id in available]),
                available,
                scope=f"diagnostic_absolute/{scope}",
                variant_id=variant_id,
            )
            for diagnostic_id in PAIR_DIAGNOSTIC_IDS:
                identity = {"variant_id": variant_id, "scope": scope, "diagnostic_id": diagnostic_id}
                if diagnostic_id in unavailable:
                    absolute.append(
                        {
                            **identity,
                            "status": "UNDEFINED",
                            "reason": "diagnostic is undefined for at least one frozen pair/seed",
                            "affected_pair_seed": sorted(set(unavailable[diagnostic_id])),
                        }
                    )
                else:
                    absolute.append({**identity, "summary": summaries[diagnostic_id]})
            if variant_id == "U0":
                continue
            candidate = np.column_stack([vectors[value] for value in PAIRED_DIAGNOSTIC_EFFECT_IDS])
            baseline = np.column_stack(
                [
                    np.asarray(
                        [
                            np.mean([index[(pair_id, seed, "U0")]["pair_diagnostics"][diagnostic_id] for seed in seeds])
                            for pair_id in pair_ids
                        ],
                        dtype=np.float64,
                    )
                    for diagnostic_id in PAIRED_DIAGNOSTIC_EFFECT_IDS
                ]
            )
            paired = _bootstrap_matrix(
                candidate - baseline,
                PAIRED_DIAGNOSTIC_EFFECT_IDS,
                scope=f"diagnostic_paired_effect/{scope}",
                variant_id=variant_id,
            )
            effects.extend(
                {
                    "variant_id": variant_id,
                    "reference_variant_id": "U0",
                    "scope": scope,
                    "diagnostic_id": diagnostic_id,
                    "effect": "variant_minus_U0",
                    "summary": paired[diagnostic_id],
                }
                for diagnostic_id in PAIRED_DIAGNOSTIC_EFFECT_IDS
            )
    return absolute, effects


def _aggregate_planned_contrasts(
    index: Mapping[tuple[str, int, str], Mapping[str, Any]],
    pair_ids: Sequence[str],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for contrast in PLANNED_CONTRASTS:
        for scope, seeds in (
            *((f"seed_{seed}", (seed,)) for seed in BASE_SEEDS),
            ("seed_mean", BASE_SEEDS),
        ):
            differences = np.empty((len(pair_ids), len(EFFECT_METRIC_IDS)), dtype=np.float64)
            unavailable: dict[str, list[str]] = defaultdict(list)
            for pair_index, pair_id in enumerate(pair_ids):
                for metric_index, metric_id in enumerate(EFFECT_METRIC_IDS):
                    seed_differences: list[float] = []
                    for seed in seeds:
                        candidate = index[(pair_id, seed, contrast.variant_id)]["scalar_metrics"][metric_id]
                        reference = index[(pair_id, seed, contrast.reference_variant_id)]["scalar_metrics"][metric_id]
                        if candidate["status"] != "OK" or reference["status"] != "OK":
                            unavailable[metric_id].append(f"{pair_id}/S{seed}")
                            continue
                        seed_differences.append(float(candidate["value"]) - float(reference["value"]))
                    differences[pair_index, metric_index] = (
                        float(np.mean(seed_differences)) if len(seed_differences) == len(seeds) else np.nan
                    )
            available = [metric_id for metric_id in EFFECT_METRIC_IDS if metric_id not in unavailable]
            available_indices = [EFFECT_METRIC_IDS.index(metric_id) for metric_id in available]
            summaries = _bootstrap_matrix(
                differences[:, available_indices],
                available,
                scope=f"planned_contrast/{scope}/{contrast.contrast_id}",
                variant_id=contrast.variant_id,
            )
            for metric_id in EFFECT_METRIC_IDS:
                identity = {
                    "contrast_id": contrast.contrast_id,
                    "variant_id": contrast.variant_id,
                    "reference_variant_id": contrast.reference_variant_id,
                    "scientific_question": contrast.scientific_question,
                    "scope": scope,
                    "metric_id": metric_id,
                    "effect": "variant_minus_reference",
                }
                if metric_id in unavailable:
                    records.append(
                        {
                            **identity,
                            "status": "ERROR",
                            "error": {
                                "error_type": "FAIL_CLOSED_METRIC_UNAVAILABLE",
                                "affected_pair_seed": sorted(set(unavailable[metric_id])),
                            },
                        }
                    )
                else:
                    records.append({**identity, "summary": summaries[metric_id]})
    return records


def _aggregate_variant_effects(
    index: Mapping[tuple[str, int, str], Mapping[str, Any]],
    pair_ids: Sequence[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Per-variant absolute summaries and their paired effect against U0."""
    absolute: list[dict[str, Any]] = []
    effects: list[dict[str, Any]] = []
    for variant_id in VARIANT_IDS:
        for scope, seeds in (
            *((f"seed_{seed}", (seed,)) for seed in BASE_SEEDS),
            ("seed_mean", BASE_SEEDS),
        ):
            variant_matrix = np.empty((len(pair_ids), len(EFFECT_METRIC_IDS)), dtype=np.float64)
            baseline_matrix = np.empty_like(variant_matrix)
            unavailable: dict[str, list[str]] = defaultdict(list)
            for pair_index, pair_id in enumerate(pair_ids):
                for metric_index, metric_id in enumerate(EFFECT_METRIC_IDS):
                    variant_seed_values: list[float] = []
                    baseline_seed_values: list[float] = []
                    for seed in seeds:
                        candidate = index[(pair_id, seed, variant_id)]["scalar_metrics"][metric_id]
                        baseline = index[(pair_id, seed, "U0")]["scalar_metrics"][metric_id]
                        if candidate["status"] != "OK" or baseline["status"] != "OK":
                            unavailable[metric_id].append(f"{pair_id}/S{seed}")
                            continue
                        variant_seed_values.append(float(candidate["value"]))
                        baseline_seed_values.append(float(baseline["value"]))
                    if len(variant_seed_values) == len(seeds):
                        variant_matrix[pair_index, metric_index] = float(np.mean(variant_seed_values))
                        baseline_matrix[pair_index, metric_index] = float(np.mean(baseline_seed_values))
                    else:
                        variant_matrix[pair_index, metric_index] = np.nan
                        baseline_matrix[pair_index, metric_index] = np.nan
            available_metric_ids = [metric_id for metric_id in EFFECT_METRIC_IDS if metric_id not in unavailable]
            available_indices = [EFFECT_METRIC_IDS.index(metric_id) for metric_id in available_metric_ids]
            absolute_summaries = _bootstrap_matrix(
                variant_matrix[:, available_indices],
                available_metric_ids,
                scope=f"absolute/{scope}",
                variant_id=variant_id,
            )
            effect_summaries = (
                _bootstrap_matrix(
                    variant_matrix[:, available_indices] - baseline_matrix[:, available_indices],
                    available_metric_ids,
                    scope=f"paired_effect/{scope}",
                    variant_id=variant_id,
                )
                if variant_id != "U0"
                else {}
            )
            for metric_id in EFFECT_METRIC_IDS:
                identity = {"variant_id": variant_id, "scope": scope, "metric_id": metric_id}
                if metric_id in unavailable:
                    error = {
                        **identity,
                        "status": "ERROR",
                        "error": {
                            "error_type": "FAIL_CLOSED_METRIC_UNAVAILABLE",
                            "affected_pair_seed": sorted(set(unavailable[metric_id])),
                        },
                    }
                    absolute.append(dict(error))
                    if variant_id != "U0":
                        effects.append(dict(error))
                    continue
                absolute.append(
                    {
                        **identity,
                        "summary": absolute_summaries[metric_id],
                    }
                )
                if variant_id != "U0":
                    effects.append(
                        {
                            **identity,
                            "reference_variant_id": "U0",
                            "effect": "variant_minus_U0",
                            "summary": effect_summaries[metric_id],
                        }
                    )
    return absolute, effects


def _transaction_census(
    index: Mapping[tuple[str, int, str], Mapping[str, Any]],
    pair_ids: Sequence[str],
) -> dict[str, dict[str, int]]:
    """How each variant's decisions were resolved, counted over the frozen matrix."""
    transaction_counts: dict[str, dict[str, int]] = {}
    for variant_id in VARIANT_IDS:
        statuses: list[str] = []
        for pair_id in pair_ids:
            for seed in BASE_SEEDS:
                statuses.extend(index[(pair_id, seed, variant_id)]["transaction_statuses"])
        transaction_counts[variant_id] = dict(sorted(Counter(statuses).items()))
    return transaction_counts


def aggregate_pair_effects(context: EvaluationContext, pair_evaluations: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    index: dict[tuple[str, int, str], dict[str, Any]] = {}
    for raw in pair_evaluations:
        item = dict(raw)
        if item.get("schema") != PAIR_EVALUATION_SCHEMA:
            raise RuntimeError("Unknown Stage 5 pair-evaluation schema")
        key = (str(item["pair_id"]), int(item["seed"]), str(item["variant_id"]))
        if key in index:
            raise RuntimeError("Duplicate Stage 5 pair evaluation")
        if tuple(item["scalar_metrics"]) != EFFECT_METRIC_IDS:
            raise RuntimeError("Pair scalar metric inventory changed")
        index[key] = item
    expected = {
        (pair_id, seed, variant_id) for pair_id in context.pairs for seed in BASE_SEEDS for variant_id in VARIANT_IDS
    }
    if set(index) != expected:
        raise RuntimeError(f"Pair evaluation inventory is incomplete: expected={len(expected)} observed={len(index)}")

    pair_ids = tuple(sorted(context.pairs))
    absolute, effects = _aggregate_variant_effects(index, pair_ids)
    planned_contrasts = _aggregate_planned_contrasts(index, pair_ids)
    contrast_index = {(item["scope"], item["contrast_id"], item["metric_id"]): item for item in planned_contrasts}
    simultaneous_families: list[dict[str, Any]] = []
    for scope, seeds in (
        *((f"seed_{seed}", (seed,)) for seed in BASE_SEEDS),
        ("seed_mean", BASE_SEEDS),
    ):
        for family_id, family_metrics in (
            (SIMULTANEOUS_FAMILY_PRIMARY, (DICE_MEAN_METRIC_ID,)),
            (SIMULTANEOUS_FAMILY_REGIONAL, DICE_LABEL_METRIC_IDS),
        ):
            columns = tuple(
                (contrast.contrast_id, metric_id) for contrast in PLANNED_CONTRASTS for metric_id in family_metrics
            )
            unavailable_columns = [
                column for column in columns if "summary" not in contrast_index[(scope, column[0], column[1])]
            ]
            if unavailable_columns:
                simultaneous_families.append(
                    {
                        "family_id": family_id,
                        "scope": scope,
                        "status": "ERROR",
                        "error": {
                            "error_type": "FAIL_CLOSED_FAMILY_MEMBER_UNAVAILABLE",
                            "columns": [
                                {"contrast_id": contrast_id, "metric_id": metric}
                                for contrast_id, metric in unavailable_columns
                            ],
                        },
                    }
                )
                continue
            values = np.empty((len(pair_ids), len(columns)), dtype=np.float64)
            for pair_index, pair_id in enumerate(pair_ids):
                for column_index, (contrast_id, metric_id) in enumerate(columns):
                    contrast = next(item for item in PLANNED_CONTRASTS if item.contrast_id == contrast_id)
                    seed_deltas = []
                    for seed in seeds:
                        candidate = index[(pair_id, seed, contrast.variant_id)]["scalar_metrics"][metric_id]
                        reference = index[(pair_id, seed, contrast.reference_variant_id)]["scalar_metrics"][metric_id]
                        seed_deltas.append(float(candidate["value"]) - float(reference["value"]))
                    values[pair_index, column_index] = float(np.mean(seed_deltas))
            pointwise = {column: contrast_index[(scope, column[0], column[1])]["summary"] for column in columns}
            family, intervals = _simultaneous_family(
                values,
                columns,
                pointwise,
                scope=scope,
                family_id=family_id,
            )
            simultaneous_families.append(family)
            for column, interval in intervals.items():
                contrast_index[(scope, column[0], column[1])]["simultaneous_ci"] = interval

    diagnostic_summaries, diagnostic_effects = _aggregate_decision_diagnostics(index, pair_ids)
    transaction_counts = _transaction_census(index, pair_ids)
    payload = {
        "schema": AGGREGATE_SCHEMA,
        "statistical_unit": "unordered_pair_id",
        "seed_handling": {
            "per_seed": list(BASE_SEEDS),
            "combined": "average the three seed-specific paired values within pair before pair bootstrap",
            "pair_seed_rows_are_not_treated_as_independent": True,
        },
        "no_label_selected_ranking": True,
        "no_success_threshold": True,
        "pair_ids": list(pair_ids),
        "absolute_summaries": absolute,
        "paired_effects_vs_u0": effects,
        "planned_contrasts": planned_contrasts,
        "simultaneous_families": simultaneous_families,
        "diagnostic_summaries": diagnostic_summaries,
        "paired_diagnostic_effects_vs_u0": diagnostic_effects,
        "transaction_counts": transaction_counts,
    }
    canonical_json_bytes(payload)
    return payload


def _render_csv(fieldnames: Sequence[str], rows: Sequence[Mapping[str, Any]]) -> str:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fieldnames, lineterminator="\n", extrasaction="raise")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue()


def decision_csv(evaluations: Sequence[Mapping[str, Any]]) -> str:
    rows = [
        {
            "evaluation_id": item["evaluation_id"],
            "case_id": item["case_id"],
            "pair_id": item["pair_id"],
            "seed": item["seed"],
            "variant_id": item["variant_id"],
            "transaction_status": item["transaction_status"],
            "runtime_seconds": item["decision_diagnostics"]["runtime_seconds"],
            "peak_memory_bytes": item["decision_diagnostics"]["peak_memory_bytes"],
            "requested_delta_rms": item["decision_diagnostics"]["requested_delta_rms"],
            "candidate_delta_rms": item["decision_diagnostics"]["candidate_delta_rms"],
            "returned_delta_rms": item["decision_diagnostics"]["returned_delta_rms"],
            "candidate_retained_ratio": item["decision_diagnostics"]["candidate_retained_ratio"],
            "returned_retained_ratio": item["decision_diagnostics"]["returned_retained_ratio"],
            "mean_dice": item["metrics"]["mean_dice"]["value"],
            "returned_field_sha256": item["returned_field"]["sha256"],
            "decision_record_sha256": item["decision_record_sha256"],
            "execution_sha256": item["execution_sha256"],
        }
        for item in sorted(evaluations, key=lambda value: value["evaluation_id"])
    ]
    return _render_csv(tuple(rows[0]) if rows else (), rows)


def per_label_csv(evaluations: Sequence[Mapping[str, Any]]) -> str:
    rows: list[dict[str, Any]] = []
    for item in sorted(evaluations, key=lambda value: value["evaluation_id"]):
        for metric in item["metrics"]["per_label_dice"]:
            rows.append(
                {
                    "evaluation_id": item["evaluation_id"],
                    "case_id": item["case_id"],
                    "pair_id": item["pair_id"],
                    "seed": item["seed"],
                    "variant_id": item["variant_id"],
                    "label": metric["label"],
                    "metric_id": metric["metric_id"],
                    "dice": metric["value"],
                }
            )
    return _render_csv(tuple(rows[0]) if rows else (), rows)


def geometry_csv(evaluations: Sequence[Mapping[str, Any]]) -> str:
    rows: list[dict[str, Any]] = []
    for item in sorted(evaluations, key=lambda value: value["evaluation_id"]):
        for metric_id in GEOMETRY_METRIC_IDS:
            metric = item["metrics"]["geometry"][metric_id]
            error = metric.get("error", {})
            rows.append(
                {
                    "evaluation_id": item["evaluation_id"],
                    "case_id": item["case_id"],
                    "pair_id": item["pair_id"],
                    "seed": item["seed"],
                    "variant_id": item["variant_id"],
                    "metric_id": metric_id,
                    "status": metric["status"],
                    "value": metric["value"],
                    "error_type": error.get("error_type", ""),
                    "error_message": error.get("message", ""),
                }
            )
    return _render_csv(tuple(rows[0]) if rows else (), rows)


def field_stage_diagnostics_csv(evaluations: Sequence[Mapping[str, Any]]) -> str:
    rows: list[dict[str, Any]] = []
    for item in sorted(evaluations, key=lambda value: value["evaluation_id"]):
        stages = item["field_stage_diagnostics"]["stages"]
        for stage in ("requested", "candidate", "returned"):
            metrics = {
                DICE_MEAN_METRIC_ID: stages[stage]["mean_dice"],
                **stages[stage]["geometry"],
            }
            for metric_id, metric in metrics.items():
                error = metric.get("error", {})
                rows.append(
                    {
                        "evaluation_id": item["evaluation_id"],
                        "case_id": item["case_id"],
                        "pair_id": item["pair_id"],
                        "seed": item["seed"],
                        "variant_id": item["variant_id"],
                        "transaction_status": item["transaction_status"],
                        "field_stage": stage,
                        "field_sha256": stages[stage]["field"]["sha256"],
                        "metric_id": metric_id,
                        "status": metric["status"],
                        "value": metric["value"],
                        "error_type": error.get("error_type", ""),
                        "error_message": error.get("message", ""),
                    }
                )
    return _render_csv(tuple(rows[0]) if rows else (), rows)


def pair_metric_csv(pair_evaluations: Sequence[Mapping[str, Any]]) -> str:
    rows: list[dict[str, Any]] = []
    for item in sorted(pair_evaluations, key=lambda value: (value["pair_id"], value["seed"], value["variant_id"])):
        for metric_id, metric in item["scalar_metrics"].items():
            error = metric.get("error", {})
            rows.append(
                {
                    "pair_id": item["pair_id"],
                    "seed": item["seed"],
                    "variant_id": item["variant_id"],
                    "metric_id": metric_id,
                    "status": metric["status"],
                    "value": metric["value"],
                    "error_type": error.get("error_type", ""),
                }
            )
    return _render_csv(tuple(rows[0]) if rows else (), rows)


def effect_csv(items: Sequence[Mapping[str, Any]]) -> str:
    rows: list[dict[str, Any]] = []
    for item in items:
        summary = item.get("summary", {})
        error = item.get("error", {})
        simultaneous = item.get("simultaneous_ci", {})
        rows.append(
            {
                "contrast_id": item.get("contrast_id", ""),
                "variant_id": item["variant_id"],
                "reference_variant_id": item.get("reference_variant_id", "U0"),
                "scientific_question": item.get("scientific_question", ""),
                "scope": item["scope"],
                "metric_id": item["metric_id"],
                "status": summary.get("status", item.get("status", "ERROR")),
                "n_unordered_pairs": summary.get("n_unordered_pairs", ""),
                "mean_effect": summary.get("mean", ""),
                "median_effect": summary.get("median", ""),
                "ci_low": summary.get("ci_low", ""),
                "ci_high": summary.get("ci_high", ""),
                "simultaneous_family_id": simultaneous.get("family_id", ""),
                "simultaneous_ci_low": simultaneous.get("ci_low", ""),
                "simultaneous_ci_high": simultaneous.get("ci_high", ""),
                "error_type": error.get("error_type", ""),
            }
        )
    return _render_csv(tuple(rows[0]) if rows else (), rows)


def diagnostic_csv(aggregate: Mapping[str, Any]) -> str:
    rows: list[dict[str, Any]] = []
    for kind, items in (
        ("absolute", aggregate["diagnostic_summaries"]),
        ("paired_effect_vs_U0", aggregate["paired_diagnostic_effects_vs_u0"]),
    ):
        for item in items:
            summary = item.get("summary", {})
            rows.append(
                {
                    "kind": kind,
                    "variant_id": item["variant_id"],
                    "reference_variant_id": item.get("reference_variant_id", ""),
                    "scope": item["scope"],
                    "diagnostic_id": item["diagnostic_id"],
                    "status": summary.get("status", item.get("status", "UNDEFINED")),
                    "n_unordered_pairs": summary.get("n_unordered_pairs", ""),
                    "mean": summary.get("mean", ""),
                    "median": summary.get("median", ""),
                    "ci_low": summary.get("ci_low", ""),
                    "ci_high": summary.get("ci_high", ""),
                }
            )
    return _render_csv(tuple(rows[0]) if rows else (), rows)


def _write_immutable_text(path: Path, text: str) -> None:
    if path.exists():
        if not path.is_file() or path.read_text(encoding="utf-8") != text:
            raise FileExistsError(f"Refusing to replace a different Stage 5 evaluation product: {path}")
        return
    atomic_write_text(path, text)


def write_decision_metrics(path: Path, evaluation: Mapping[str, Any]) -> str:
    payload = canonical_json_bytes(dict(evaluation)).decode("utf-8")
    _write_immutable_text(path, payload)
    return sha256_file(path)


def write_evaluation_products(
    output_root: Path,
    evaluations: Sequence[Mapping[str, Any]],
    pair_evaluations: Sequence[Mapping[str, Any]],
    aggregate: Mapping[str, Any],
) -> dict[str, str]:
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    bundle = {
        "schema": "ctcf-stage5-evaluation-products-v1",
        "evaluations": list(evaluations),
        "pair_evaluations": list(pair_evaluations),
        "aggregate": dict(aggregate),
    }
    products = {
        "evaluation_bundle.json": canonical_json_bytes(bundle).decode("utf-8"),
        "per_decision.csv": decision_csv(evaluations),
        "per_label.csv": per_label_csv(evaluations),
        "geometry_metrics.csv": geometry_csv(evaluations),
        "field_stage_diagnostics.csv": field_stage_diagnostics_csv(evaluations),
        "per_pair_metric.csv": pair_metric_csv(pair_evaluations),
        "paired_effects_vs_u0.csv": effect_csv(aggregate["paired_effects_vs_u0"]),
        "planned_contrasts.csv": effect_csv(aggregate["planned_contrasts"]),
        "decision_diagnostics.csv": diagnostic_csv(aggregate),
    }
    digests: dict[str, str] = {}
    for name, text in products.items():
        path = output_root / name
        _write_immutable_text(path, text)
        digests[name] = sha256_file(path)
    return digests


__all__ = [
    "AGGREGATE_SCHEMA",
    "BOOTSTRAP_ALGORITHM",
    "BOOTSTRAP_CI_PERCENT",
    "BOOTSTRAP_DOMAIN",
    "BOOTSTRAP_ITERATIONS",
    "DICE_LABEL_METRIC_IDS",
    "DICE_MEAN_METRIC_ID",
    "EFFECT_METRIC_IDS",
    "EVALUATION_SCHEMA",
    "GEOMETRY_METRIC_IDS",
    "INVERSE_COMPONENT_RMS_METRIC_ID",
    "INVERSE_MAX_VECTOR_METRIC_ID",
    "PAIR_DIAGNOSTIC_IDS",
    "PAIR_EVALUATION_SCHEMA",
    "SIMULTANEOUS_FAMILY_PRIMARY",
    "SIMULTANEOUS_FAMILY_REGIONAL",
    "STAGE5_EVALUATION_METRIC_IDS",
    "WARP_CONVENTION",
    "EvaluationContext",
    "aggregate_pair_effects",
    "build_evaluation_record",
    "build_pair_evaluation",
    "compute_geometry_bundle",
    "decision_csv",
    "diagnostic_csv",
    "effect_csv",
    "evaluate_returned_decision",
    "field_stage_diagnostics_csv",
    "geometry_csv",
    "pair_metric_csv",
    "per_label_csv",
    "write_decision_metrics",
    "write_evaluation_products",
]
