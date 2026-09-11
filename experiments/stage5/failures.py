"""Preserve a failed controller update on the training server, without retrying it."""

import errno
import traceback
from collections.abc import Mapping
from dataclasses import fields, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from experiments.stage5.checkpoints import atomic_torch_save
from experiments.stage5.precision import controller_precision_contract
from tools.analysis.run_artifacts import atomic_write_json

# Only environmental I/O failures with a concrete errno can be acknowledged.
# Missing files, permission/configuration mistakes and untyped OSError messages
# require investigation rather than receiving a generic recovery bypass.
TECHNICAL_IO_ERRNOS = frozenset(
    getattr(errno, name)
    for name in (
        "ENOSPC",
        "EDQUOT",
        "EIO",
        "ESTALE",
        "ETIMEDOUT",
        "ECONNRESET",
        "ENETDOWN",
        "ENETUNREACH",
        "EHOSTUNREACH",
        "EMFILE",
        "ENFILE",
        "ENOBUFS",
    )
    if hasattr(errno, name)
)


def failure_exception_chain(error: BaseException) -> list[dict[str, Any]]:
    """Retain the cause of wrappers, including bootstrap failures caused by OOM."""
    records = []
    seen: set[int] = set()
    current: BaseException | None = error
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        records.append(
            {
                "type": type(current).__name__,
                "message": str(current),
                "numerical": isinstance(current, FloatingPointError),
                "resource": isinstance(current, (torch.OutOfMemoryError, MemoryError))
                or (
                    isinstance(current, RuntimeError)
                    and any(
                        marker in str(current).lower()
                        for marker in ("out of memory", "cudnn_status_alloc_failed", "cublas_status_alloc_failed")
                    )
                ),
                "io": isinstance(current, OSError),
                "errno": current.errno if isinstance(current, OSError) else None,
            }
        )
        cause = current.__cause__
        current = cause if cause is not None else (None if current.__suppress_context__ else current.__context__)
    return records


def classify_failure_chain(chain: list[dict[str, Any]]) -> tuple[str, bool]:
    """Return a conservative class and eligibility, never an automatic retry."""
    if any(item.get("numerical") is True for item in chain):
        return "NUMERICAL", False
    # Follow a wrapper to its actual cause. An OOM encountered while handling an
    # unexplained earlier error must not make that earlier error recoverable.
    cause = chain[-1] if chain else {}
    if cause.get("resource") is True:
        return "RESOURCE", True
    io_errors = [item for item in chain if item.get("io") is True]
    if io_errors:
        technical = cause.get("io") is True and all(item.get("errno") in TECHNICAL_IO_ERRNOS for item in io_errors)
        return "IO", technical
    return "INVARIANT_OR_UNKNOWN", False


def cpu_snapshot(value: Any) -> Any:
    """Copy small training state normally; copy full inputs only after a failure."""
    if isinstance(value, torch.Tensor):
        return value.detach().to(device="cpu", copy=True)
    if is_dataclass(value) and not isinstance(value, type):
        return {field.name: cpu_snapshot(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, Mapping):
        return {key: cpu_snapshot(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(cpu_snapshot(item) for item in value)
    if isinstance(value, list):
        return [cpu_snapshot(item) for item in value]
    return value


def require_finite_parameters(model: torch.nn.Module, *, gradients: bool) -> None:
    for name, parameter in model.named_parameters():
        value = parameter.grad if gradients else parameter
        if value is None:
            continue  # Some variant-specific heads are intentionally unused.
        if value.dtype != torch.float32:
            raise RuntimeError(f"Stage5 controller {'gradient' if gradients else 'parameter'} is not FP32: {name}")
        if not bool(torch.isfinite(value).all()):
            raise FloatingPointError(f"non-finite Stage5 controller {'gradient' if gradients else 'parameter'}: {name}")


def require_finite_optimizer(optimizer: torch.optim.Optimizer) -> None:
    for parameter_index, state in enumerate(optimizer.state.values()):
        for name, value in state.items():
            if isinstance(value, torch.Tensor) and not bool(torch.isfinite(value).all()):
                raise FloatingPointError(f"non-finite Stage5 controller optimizer state: {parameter_index}/{name}")


class ControllerFailureRecorder:
    def __init__(self, output_root: Path, identity: Mapping[str, Any]):
        self.root = output_root / "failures"
        self.identity = dict(identity)

    def require_no_previous_failure(self, *, resume: Path | None = None) -> list[dict[str, str]]:
        from tools.analysis.stage5.recovery import require_acknowledged_controller_failures

        return require_acknowledged_controller_failures(
            output_root=self.root.parent, identity=self.identity, resume=resume
        )

    def capture(self, *, error: Exception, state: dict[str, Any], step: Any) -> Path:
        failure_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
        directory = self.root / failure_id
        directory.mkdir(parents=True, exist_ok=False)
        manifest_path = directory / "failure.json"
        chain = failure_exception_chain(error)
        failure_kind, eligible = classify_failure_chain(chain)
        record = {
            "schema": "ctcf-stage5-controller-failure-v2",
            "failure_id": failure_id,
            "failure_kind": failure_kind,
            "technical_recovery_eligible": eligible,
            "exception_chain": chain,
            "identity": self.identity,
            "pair": state["pair"],
            "epoch_one_based": state["epoch_one_based"],
            "pair_index_one_based": state["pair_index_one_based"],
            "successful_updates_before_pair": state["successful_updates_before_pair"],
            "phase": state["phase"],
            "exception_type": type(error).__name__,
            "exception": str(error),
            "traceback": "".join(traceback.format_exception(type(error), error, error.__traceback__)),
            "precision": controller_precision_contract(),
            "prepared_inputs_available": state["inputs"] is not None,
            "automatic_retry_allowed": False,
            "training_checkpoint": False,
            "capture_status": "PENDING",
        }
        # A compact marker survives even if disk space prevents the heavy capture.
        atomic_write_json(manifest_path, record)
        try:
            if state["before_update"] is None:
                state["before_update"] = {
                    "model": cpu_snapshot(step.controller.state_dict()),
                    "optimizer": cpu_snapshot(step.optimizer.state_dict()),
                }
            payload = {
                **record,
                "schema": "ctcf-stage5-controller-failure-capture-v1",
                "capture_status": "COMPLETE",
                **cpu_snapshot(state),
                "gradients_at_failure": cpu_snapshot(
                    {name: parameter.grad for name, parameter in step.controller.named_parameters()}
                ),
                "model_at_failure": cpu_snapshot(step.controller.state_dict()),
                "optimizer_at_failure": cpu_snapshot(step.optimizer.state_dict()),
            }
            capture_path = directory / "capture.pth"
            digest = atomic_torch_save(capture_path, payload)
            record.update(
                capture_status="COMPLETE",
                capture_file={
                    "path": str(capture_path.resolve()),
                    "sha256": digest,
                    "bytes": capture_path.stat().st_size,
                },
            )
        except Exception as capture_error:
            record.update(capture_status="FAILED", capture_error=f"{type(capture_error).__name__}: {capture_error}")
        atomic_write_json(manifest_path, record)
        print(f"[STAGE5 CONTROLLER FAILURE] {manifest_path} capture_status={record['capture_status']}", flush=True)
        return manifest_path
