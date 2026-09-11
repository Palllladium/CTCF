"""Explicit recovery from an identified environmental controller failure.

Failure evidence is never edited or moved. An acknowledgement authorizes one
completed checkpoint, not a failed pair, a captured optimizer state, or a new
numerical policy. Subsequent checkpoints carry the acknowledgement digest.
"""

from __future__ import annotations

import math
import os
from collections.abc import Mapping
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from experiments.stage5.checkpoints import STAGE5_TRAINING_STATE_SCHEMA, state_dict_sha256
from experiments.stage5.config import ControllerTrainingConfig
from experiments.stage5.failures import classify_failure_chain
from experiments.stage5.losses import ControllerLossConfig
from tools.analysis.run_artifacts import sha256_file
from tools.analysis.stage5.contracts import CHECKPOINT_SELECTION_POLICY
from tools.analysis.stage5.primitives import (
    canonical_json_bytes,
    is_link_like,
    load_json_object,
    require_git_sha,
    require_plain_directory,
    require_sha256,
    resolve_inside_root,
)

ACKNOWLEDGEMENT_SCHEMA = "ctcf-stage5-controller-recovery-acknowledgement-v1"
_IDENTITY_KEYS = (
    "role",
    "variant",
    "seed",
    "git_head",
    "protocol_sha256",
    "data_contract_sha256",
    "training_contract_sha256",
    "base_checkpoint_sha256",
    "initial_controller_state_sha256",
    "source_contract_sha256",
    "config",
    "bootstrap_policy",
)


@contextmanager
def locked_stopped_stage5_run(*, protocol_path: Path):
    """Hold the runner's existing lock throughout acknowledgement on its server."""
    if protocol_path.name != "protocol.json" or protocol_path.parent.name != "protocol":
        raise ValueError("Recovery requires the run's protocol/protocol.json path")
    root = require_plain_directory(protocol_path.parent.parent, "Stage5 run root")
    resolve_inside_root(root, "protocol/protocol.json", label="run protocol")
    lock_path = resolve_inside_root(root, "stage5.lock", label="existing Stage5 runner lock")
    try:
        import fcntl
    except ImportError as error:
        raise RuntimeError("Controller recovery acknowledgement must run on the Linux training server") from error
    with lock_path.open("r+b") as stream:
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("Stage5 run is still active; stop its workers before acknowledging a failure") from error
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def _identity_binding(identity: Mapping[str, Any]) -> dict[str, Any]:
    missing = set(_IDENTITY_KEYS) - identity.keys()
    if missing:
        raise RuntimeError(f"Controller recovery identity is incomplete: {sorted(missing)}")
    binding = {key: identity[key] for key in _IDENTITY_KEYS}
    require_git_sha(binding["git_head"], "recovery Git HEAD")
    for key in _IDENTITY_KEYS:
        if key.endswith("sha256"):
            require_sha256(binding[key], f"recovery {key}")
    if binding["role"] != "CONTROLLER":
        raise RuntimeError("Only a production controller failure can be acknowledged")
    return binding


def _failure_record(root: Path, failure_id: str) -> tuple[Path, dict[str, Any]]:
    if (
        not isinstance(failure_id, str)
        or not failure_id
        or Path(failure_id).name != failure_id
        or failure_id in {".", ".."}
    ):
        raise ValueError("failure_id must name exactly one recorded failure directory")
    path = resolve_inside_root(root, f"failures/{failure_id}/failure.json", label="controller failure")
    record = load_json_object(path)
    if record.get("schema") != "ctcf-stage5-controller-failure-v2" or record.get("failure_id") != failure_id:
        raise RuntimeError("Failure has no validated technical classification; automatic retry is forbidden")
    chain = record.get("exception_chain")
    if not isinstance(chain, list) or not chain or any(not isinstance(item, dict) for item in chain):
        raise RuntimeError("Controller failure has no valid exception chain")
    kind, eligible = classify_failure_chain(chain)
    if kind != record.get("failure_kind") or eligible != record.get("technical_recovery_eligible"):
        raise RuntimeError("Controller failure classification changed")
    if not eligible or kind not in {"RESOURCE", "IO"}:
        raise RuntimeError(f"{kind} failure remains blocked; automatic retry is forbidden")
    return path, record


def _require_finite_state(value: Any) -> None:
    if isinstance(value, torch.Tensor):
        if not bool(torch.isfinite(value).all()):
            raise RuntimeError("Recovery checkpoint contains non-finite model or optimizer state")
    elif isinstance(value, float) and not math.isfinite(value):
        raise RuntimeError("Recovery checkpoint contains non-finite optimizer scalar state")
    elif isinstance(value, Mapping):
        for item in value.values():
            _require_finite_state(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _require_finite_state(item)


def _last_checkpoint(root: Path, identity: Mapping[str, Any]) -> tuple[str, dict[str, Any]]:
    """Read the generation runtime would adopt, without repairing/moving any file."""
    from experiments.stage5.runtime import _validate_runtime_checkpoint_metadata

    names = ("last.pth", ".last.pth.next", ".last.pth.previous")
    files = []
    sidecars = []
    for name in names:
        for suffix, target in (("", files), (".sha256.json", sidecars)):
            candidate = root / f"{name}{suffix}"
            if candidate.exists() or is_link_like(candidate):
                target.append(resolve_inside_root(root, candidate.name, label="recovery checkpoint"))
    records = []
    for path in sidecars:
        try:
            records.append(load_json_object(path))
        except (OSError, ValueError, RuntimeError):
            continue  # A different intact generation may still be recoverable.
    selected = None
    for path in files:
        digest = sha256_file(path)
        expected = {
            "schema": "ctcf-stage5-checkpoint-sha256-v1",
            "file_name": "last.pth",
            "bytes": path.stat().st_size,
            "sha256": digest,
        }
        if expected in records:
            selected = path, digest
            break
    if selected is None:
        raise RuntimeError(
            "No sidecar-authenticated completed controller epoch is available. "
            "Recovery cannot use capture.pth or silently restart from initial weights; "
            "a separately approved new run is required."
        )
    path, digest = selected
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or payload.get("schema") != STAGE5_TRAINING_STATE_SCHEMA:
        raise RuntimeError("Recovery checkpoint is not a Stage5 training checkpoint")
    expected_fields = {
        key: identity[key] for key in _IDENTITY_KEYS if key not in {"config", "bootstrap_policy", "variant"}
    }
    expected_fields["variant_id"] = identity["variant"]
    expected_fields["selection_policy"] = CHECKPOINT_SELECTION_POLICY
    if any(payload.get(key) != value for key, value in expected_fields.items()):
        raise RuntimeError("Recovery checkpoint identity differs from the recorded failure")
    epoch = payload.get("epoch_completed")
    if type(epoch) is not int or epoch < 1:
        raise RuntimeError("Recovery requires a completed controller epoch")
    model_state = payload.get("model_state")
    if not isinstance(model_state, dict) or state_dict_sha256(model_state) != payload.get("model_state_sha256"):
        raise RuntimeError("Recovery checkpoint model-state digest mismatch")
    if not isinstance(payload.get("optimizer_state"), dict) or not isinstance(payload.get("rng_state"), dict):
        raise RuntimeError("Recovery checkpoint lacks optimizer/RNG state")
    if set(payload["rng_state"]) != {"python", "numpy", "torch_cpu", "torch_cuda"}:
        raise RuntimeError("Recovery checkpoint RNG-state fields changed")
    _require_finite_state(model_state)
    _require_finite_state(payload["optimizer_state"])
    config_fields = dict(identity["config"])
    config_fields["loss"] = ControllerLossConfig(**config_fields["loss"])
    config = ControllerTrainingConfig(**config_fields)
    _validate_runtime_checkpoint_metadata(
        payload,
        role="CONTROLLER",
        variant=identity["variant"],
        seed=identity["seed"],
        config=config,
        expected_git_head=identity["git_head"],
        expected_base_checkpoint_sha256=identity["base_checkpoint_sha256"],
        expected_initial_controller_state_sha256=identity["initial_controller_state_sha256"],
        expected_source_contract_sha256=identity["source_contract_sha256"],
    )
    if epoch > config.fixed_epoch:
        raise RuntimeError("Recovery checkpoint exceeds the frozen controller endpoint")
    return digest, payload


def acknowledge_controller_failure(
    *,
    output_root: Path,
    failure_id: str,
    failure_sha256: str,
    checkpoint_sha256: str,
    reason: str,
    expected_git_head: str,
    expected_protocol_sha256: str,
) -> Path:
    """Acknowledge one inspected RESOURCE/technical IO failure; never launch work."""
    root = require_plain_directory(output_root, "controller output root")
    require_git_sha(expected_git_head, "expected Git HEAD", error=ValueError)
    require_sha256(expected_protocol_sha256, "expected protocol SHA-256", error=ValueError)
    require_sha256(failure_sha256, "failure SHA-256", error=ValueError)
    require_sha256(checkpoint_sha256, "checkpoint SHA-256", error=ValueError)
    if not isinstance(reason, str) or len(reason.strip()) < 5:
        raise ValueError("An explicit reason describing the inspected and resolved technical cause is required")
    failure_path, failure = _failure_record(root, failure_id)
    if sha256_file(failure_path) != failure_sha256:
        raise RuntimeError("Failure SHA-256 differs from the explicitly acknowledged evidence")
    identity = _identity_binding(failure["identity"])
    if identity["git_head"] != expected_git_head or identity["protocol_sha256"] != expected_protocol_sha256:
        raise RuntimeError("Acknowledgement HEAD/protocol differs from the failed run")
    digest, checkpoint = _last_checkpoint(root, identity)
    if digest != checkpoint_sha256:
        raise RuntimeError("Checkpoint SHA-256 differs from the current last valid epoch")
    if checkpoint["epoch_completed"] >= failure["epoch_one_based"]:
        raise RuntimeError("Acknowledgement checkpoint does not precede the failed epoch")
    record = {
        "schema": ACKNOWLEDGEMENT_SCHEMA,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "failure_id": failure_id,
        "failure_sha256": failure_sha256,
        "failure_kind": failure["failure_kind"],
        "identity": identity,
        "reason": reason.strip(),
        "action": "RESUME_FROM_COMPLETED_EPOCH",
        "automatic_retry_allowed": False,
        "checkpoint": {"logical_path": "last.pth", "sha256": digest, "epoch_completed": checkpoint["epoch_completed"]},
    }
    path = failure_path.parent / "acknowledgement.json"
    if is_link_like(path):
        raise RuntimeError("Acknowledgement must not be a link")
    # Exclusive creation prevents two approvals from replacing each other. A
    # partial write stays invalid and blocked; failure evidence is never removed.
    with path.open("xb") as stream:
        stream.write(canonical_json_bytes(record))
        stream.flush()
        os.fsync(stream.fileno())
    return path


def require_acknowledged_controller_failures(
    *,
    output_root: Path,
    identity: Mapping[str, Any],
    resume: Path | None = None,
) -> list[dict[str, str]]:
    """Return digest references that every subsequent controller checkpoint stores."""
    failure_root = output_root / "failures"
    if not failure_root.exists() and not is_link_like(failure_root):
        return []
    root = require_plain_directory(output_root, "controller output root")
    failure_root = require_plain_directory(failure_root, "controller failure root")
    directories = sorted(failure_root.iterdir())
    if not directories:
        return []
    if resume is not None and resume.absolute() != root / "last.pth":
        raise RuntimeError(
            "Recorded failures can only resume their own last.pth, never an external checkpoint or capture"
        )
    accepted = []
    current = None
    for directory in directories:
        require_plain_directory(directory, "controller failure directory")
        failure_path, failure = _failure_record(root, directory.name)
        acknowledgement = directory / "acknowledgement.json"
        if not acknowledgement.exists():
            raise RuntimeError(
                f"Recorded failure {directory.name} has no explicit acknowledgement; automatic retry is forbidden"
            )
        acknowledgement = resolve_inside_root(
            root, f"failures/{directory.name}/acknowledgement.json", label="failure acknowledgement"
        )
        ack = load_json_object(acknowledgement)
        expected = {
            "schema": ACKNOWLEDGEMENT_SCHEMA,
            "failure_id": directory.name,
            "failure_sha256": sha256_file(failure_path),
            "failure_kind": failure["failure_kind"],
            "identity": _identity_binding(identity),
            "action": "RESUME_FROM_COMPLETED_EPOCH",
            "automatic_retry_allowed": False,
        }
        if (
            any(ack.get(key) != value for key, value in expected.items())
            or _identity_binding(failure["identity"]) != expected["identity"]
        ):
            raise RuntimeError("Failure acknowledgement or current run identity changed")
        if not isinstance(ack.get("reason"), str) or len(ack["reason"].strip()) < 5:
            raise RuntimeError("Failure acknowledgement has no technical explanation")
        try:
            created = datetime.fromisoformat(ack["created_utc"])
            if created.tzinfo is None:
                raise ValueError("timezone missing")
        except (KeyError, TypeError, ValueError) as error:
            raise RuntimeError("Failure acknowledgement has no valid timestamp") from error
        approved = ack.get("checkpoint")
        if (
            not isinstance(approved, dict)
            or approved.get("logical_path") != "last.pth"
            or type(approved.get("epoch_completed")) is not int
            or approved["epoch_completed"] < 1
        ):
            raise RuntimeError("Failure acknowledgement has no completed checkpoint")
        require_sha256(approved.get("sha256"), "acknowledged checkpoint SHA-256")
        if approved["epoch_completed"] >= failure["epoch_one_based"]:
            raise RuntimeError("Acknowledged checkpoint does not precede the failed epoch")
        if current is None:
            current = _last_checkpoint(root, expected["identity"])
        digest, checkpoint = current
        reference = {"failure_id": directory.name, "acknowledgement_sha256": sha256_file(acknowledgement)}
        same_checkpoint = digest == approved["sha256"] and checkpoint["epoch_completed"] == approved["epoch_completed"]
        descendant = checkpoint["epoch_completed"] > approved["epoch_completed"] and reference in checkpoint.get(
            "recovery_acknowledgements", []
        )
        if not same_checkpoint and not descendant:
            raise RuntimeError(
                "Current checkpoint is neither the acknowledged epoch nor a checkpoint carrying its recovery reference"
            )
        accepted.append(reference)
    return accepted
