"""Small scalar training traces, separated by attempt from human-readable logs.

Only epochs authenticated by a checkpoint belong to the accepted trajectory.
Step records from an interrupted epoch remain useful evidence, not resumed work.
"""

from __future__ import annotations

import json
import math
import os
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from tools.analysis.run_artifacts import atomic_write_json, sha256_file


def snapshot_parameters(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    """Capture parameter bytes before AdamW without keeping a GPU-sized copy."""
    return {name: value.detach().to(device="cpu", copy=True) for name, value in model.named_parameters()}


def _norm_parts(value: torch.Tensor) -> tuple[float, float]:
    # Controller parameter tensors are small. CPU FP64 avoids FP16/FP32 sum
    # overflow and does not change training tensors, gradients or backend flags.
    flat = value.detach().to(device="cpu", dtype=torch.float64).reshape(-1)
    if not bool(torch.isfinite(flat).all()):
        raise FloatingPointError("non-finite controller telemetry tensor")
    if not flat.numel():
        return 0.0, 0.0
    return float(torch.linalg.vector_norm(flat)), float(flat.abs().max())


def parameter_telemetry(
    model: torch.nn.Module, *, before_update: Mapping[str, torch.Tensor] | None = None
) -> dict[str, Any]:
    """Observe weights, unscaled gradients and the actual optimizer displacement.

    Update/weight ratios use the PRE-update weight norm. Zero denominators are
    represented by null and an explicit status; no arbitrary epsilon is added.
    The caller must run this after optimizer.step when requesting update norms.
    """
    grouped: dict[str, dict[str, list[tuple[float, float]]]] = {}
    missing_gradients = []
    first_bias_name = None
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        current = parameter.detach().to(device="cpu", copy=True)
        previous = current if before_update is None else before_update[name].detach().cpu()
        if current.shape != previous.shape:
            raise ValueError(f"telemetry parameter shape changed: {name}")
        measurements = {"weight": _norm_parts(previous)}
        if parameter.grad is not None:
            measurements["gradient"] = _norm_parts(parameter.grad)
        else:
            missing_gradients.append(name)
        if before_update is not None:
            # Subtract in FP64: this measures the difference between stored FP32
            # endpoints, including AdamW weight decay and rounding of the update.
            measurements["update"] = _norm_parts(current.double() - previous.double())
        labels = ["global", name.split(".", 1)[0]]
        if first_bias_name is None and (name == "bias" or name.endswith(".bias")):
            first_bias_name = name
            labels.append("first_bias")
        for label in dict.fromkeys(labels):
            groups = grouped.setdefault(label, {})
            for kind, pair in measurements.items():
                groups.setdefault(kind, []).append(pair)

    stats = {}
    for label, groups in grouped.items():
        row: dict[str, Any] = {}
        for kind, parts in groups.items():
            row[f"{kind}_l2"] = math.hypot(*(item[0] for item in parts))
            row[f"{kind}_max_abs"] = max(item[1] for item in parts)
        if before_update is not None:
            norm = row["weight_l2"]
            row["update_to_weight_l2"] = row["update_l2"] / norm if norm > 0 else None
            row["update_to_weight_status"] = "DEFINED" if norm > 0 else "ZERO_WEIGHT_NORM"
        stats[label] = row
    return {
        "schema": "ctcf-stage5-parameter-telemetry-v1",
        "weight_reference": "BEFORE_UPDATE" if before_update is not None else "CURRENT",
        "gradient_reference": "UNSCALED_PARAMETER_GRADIENT",
        "first_bias_parameter": first_bias_name,
        "missing_gradient_parameters": missing_gradients,
        "groups": stats,
    }


def _scalars(value: Mapping[str, Any], prefix: str = ""):
    for name, item in value.items():
        path = f"{prefix}.{name}" if prefix else name
        if isinstance(item, Mapping):
            yield from _scalars(item, path)
        elif isinstance(item, (int, float)) and not isinstance(item, bool):
            number = float(item)
            if not math.isfinite(number):
                raise FloatingPointError(f"non-finite telemetry scalar: {path}")
            yield path, number


class TrainingTelemetry:
    """An append-only JSONL per process attempt; epochs are committed separately."""

    def __init__(self, output_root: Path, identity: Mapping[str, Any], *, start_epoch: int):
        attempt = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ") + f"_{os.getpid()}"
        self.root = output_root / "telemetry" / attempt
        self.root.mkdir(parents=True, exist_ok=False)
        self.steps_path = self.root / "steps.jsonl"
        self.epochs_path = self.root / "epochs.jsonl"
        self.steps_path.touch(exist_ok=False)
        self.epochs_path.touch(exist_ok=False)
        self._epochs: dict[int, dict[str, Any]] = {}
        atomic_write_json(
            self.root / "attempt.json",
            {
                "schema": "ctcf-stage5-training-telemetry-attempt-v1",
                "identity": dict(identity),
                "attempt": attempt,
                "resume_epoch_completed": start_epoch,
                "step_records_are_provisional_until_epoch_checkpoint": True,
                "partial_final_jsonl_line_after_process_interruption": "IGNORE_INCOMPLETE_LINE_ONLY",
            },
        )

    @staticmethod
    def _append(path: Path, record: Mapping[str, Any]) -> None:
        line = json.dumps(record, ensure_ascii=True, allow_nan=False, separators=(",", ":"))
        with path.open("a", encoding="utf-8", newline="\n") as stream:
            stream.write(line + "\n")
            stream.flush()

    def write_step(self, *, context: Mapping[str, Any], metrics: Mapping[str, Any], telemetry: Mapping[str, Any]):
        record = {"context": dict(context), "metrics": dict(metrics), "parameters": dict(telemetry)}
        self._append(self.steps_path, record)
        epoch = int(context["epoch_one_based"])
        summary = self._epochs.setdefault(epoch, {"steps": 0, "scalars": {}})
        summary["steps"] += 1
        for key, value in _scalars({"metrics": metrics, "parameters": telemetry.get("groups", {})}):
            accumulator = summary["scalars"].setdefault(
                key, {"count": 0, "mean": 0.0, "max": value, "max_at": dict(context)}
            )
            accumulator["count"] += 1
            accumulator["mean"] += (value - accumulator["mean"]) / accumulator["count"]
            if value > accumulator["max"]:
                accumulator["max"] = value
                accumulator["max_at"] = dict(context)

    def summarize_epoch(self, epoch: int) -> dict[str, Any]:
        summary = self._epochs.get(epoch)
        if summary is None:
            raise RuntimeError("cannot summarize an epoch without step telemetry")
        return {
            "attempt": self.root.name,
            "epoch_one_based": epoch,
            "steps": summary["steps"],
            "scalars": summary["scalars"],
        }

    def commit_epoch(self, epoch: int, checkpoint: Path, pair_schedule_sha256: str) -> None:
        summary = self.summarize_epoch(epoch)
        self._append(
            self.epochs_path,
            {
                "schema": "ctcf-stage5-telemetry-committed-epoch-v1",
                "summary": summary,
                "checkpoint": str(checkpoint.resolve()),
                "checkpoint_sha256_at_commit": sha256_file(checkpoint),
                "pair_schedule_sha256": pair_schedule_sha256,
            },
        )
        del self._epochs[epoch]
