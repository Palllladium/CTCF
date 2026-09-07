"""Import completed U0 endpoints across the one bootstrap-only Stage-5 hotfix.

The legacy ``git_head``/``protocol_sha256`` fields become the target *binding*.
Actual training provenance stays in ``training_git_head`` and the authenticated
``u0_import_lineage`` object, also retained in a compact import manifest. No
training state, metric, or source artifact is changed by this operation.
"""

from __future__ import annotations

import copy
import hashlib
from pathlib import Path
from typing import Any

import torch

from experiments.stage5.checkpoints import STAGE5_TRAINING_STATE_SCHEMA, atomic_torch_save, state_dict_sha256
from experiments.stage5.config import STAGE5_SEEDS, U0TrainingConfig
from experiments.stage5.runtime import _validate_runtime_checkpoint_metadata
from tools.analysis.run_artifacts import sha256_file
from tools.analysis.stage5.contracts import CHECKPOINT_SELECTION_POLICY, validate_protocol_contract
from tools.analysis.stage5.primitives import (
    canonical_sha256,
    is_link_like,
    load_json_object,
    readable_json_bytes,
    require_int,
    require_regular_file,
    require_sha256,
    write_immutable_bytes,
    write_immutable_json,
)
from tools.analysis.stage5.protocol import bootstrap_parameters, u0_training_contract

SOURCE_GIT_HEAD = "68df5b24104272fd137106fb51959907502db2a9"
IMPORT_SCHEMA = "ctcf-stage5-completed-u0-import-v1"
OLD_REPAIR_ID = "CTCF_DIGITAL_THEN_TRILINEAR_COLLAR_REPAIR_V1"
NEW_REPAIR_ID = "CTCF_DIGITAL_THEN_TRILINEAR_COLLAR_REPAIR_V2"
_BINDING_FIELDS = frozenset({"git_head", "protocol_sha256", "training_git_head", "u0_import_lineage"})


def _plain_path(path: Path) -> Path:
    path = Path(path).absolute()
    if any(is_link_like(candidate) for candidate in (path, *path.parents)):
        raise RuntimeError(f"Stage5 U0 import must not traverse a link: {path}")
    return path.resolve()


def _load_protocol(path: Path) -> tuple[dict[str, Any], U0TrainingConfig]:
    require_regular_file(path, "protocol")
    protocol = load_json_object(path)
    validate_protocol_contract(protocol)
    contracts = {}
    for name in ("u0_training_contract", "controller_training_contract", "search_contract"):
        contract_path = _plain_path(path.with_name(f"{name}.json"))
        require_regular_file(contract_path, name)
        if sha256_file(contract_path) != protocol[f"{name}_sha256"]:
            raise RuntimeError(f"Stage5 U0 import {name} file digest mismatch")
        contracts[name] = load_json_object(contract_path)
    config = U0TrainingConfig(**contracts["u0_training_contract"]["config"])
    if contracts["u0_training_contract"] != u0_training_contract(config):
        raise RuntimeError("Stage5 U0 import training contract differs from the frozen implementation")
    if protocol["u0_fixed_epoch"] != config.fixed_epoch:
        raise RuntimeError("Stage5 U0 import protocol and training endpoint disagree")
    return protocol, config


def _validate_hotfix(source: dict[str, Any], target: dict[str, Any]) -> None:
    if source["git_head"] != SOURCE_GIT_HEAD or target["git_head"] == SOURCE_GIT_HEAD:
        raise RuntimeError("Stage5 U0 import only supports the named historical revision into a new revision")
    parameters = bootstrap_parameters()
    if (
        parameters.get("repair_operator_id") != NEW_REPAIR_ID
        or parameters["repair_parameters"].get("digital_residual_policy") != "DIAGNOSTIC_CONTINUE_TO_TRILINEAR"
    ):
        raise RuntimeError("Stage5 U0 import requires the expected V2 bootstrap implementation")
    if target["bootstrap"] != {"policy": "collar_repair", "parameters": parameters}:
        raise RuntimeError("Stage5 U0 import target bootstrap is not the exact V2 hotfix")
    expected_source = copy.deepcopy(target)
    expected_source["git_head"] = SOURCE_GIT_HEAD
    expected_source["bootstrap"]["parameters"]["repair_operator_id"] = OLD_REPAIR_ID
    del expected_source["bootstrap"]["parameters"]["repair_parameters"]["digital_residual_policy"]
    if source != expected_source:
        raise RuntimeError("Stage5 U0 import protocols differ beyond git_head and the exact bootstrap V1-to-V2 hotfix")


class _HashWriter:
    """Hash torch's serialization without retaining a second checkpoint in RAM."""

    def __init__(self) -> None:
        self.digest = hashlib.sha256()

    def write(self, data: bytes) -> int:
        self.digest.update(data)
        return len(data)

    def flush(self) -> None:
        pass


def _preserved_digest(payload: dict[str, Any]) -> str:
    writer = _HashWriter()
    torch.save({key: value for key, value in payload.items() if key not in _BINDING_FIELDS}, writer)
    return writer.digest.hexdigest()


def _load_endpoint(
    path: Path, *, seed: int, protocol: dict[str, Any], config: U0TrainingConfig
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Verify the live generation read-only; never invoke resume recovery on the source."""
    path = _plain_path(path)
    require_regular_file(path, "completed U0 checkpoint")
    sidecar = _plain_path(path.with_name(f"{path.name}.sha256.json"))
    require_regular_file(sidecar, "U0 checkpoint sidecar")
    digest = sha256_file(path)
    expected_sidecar = {
        "schema": "ctcf-stage5-checkpoint-sha256-v1",
        "file_name": path.name,
        "bytes": path.stat().st_size,
        "sha256": digest,
    }
    if load_json_object(sidecar) != expected_sidecar:
        raise RuntimeError(f"Stage5 U0 import checkpoint bytes or sidecar changed: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise RuntimeError("Stage5 U0 import requires a training-state object")
    for key in ("seed", "epoch_completed", "fixed_epoch"):
        require_int(payload.get(key), f"U0 checkpoint {key}")
    expected = {
        "schema": STAGE5_TRAINING_STATE_SCHEMA,
        "role": "U0",
        "variant_id": "U0",
        "seed": seed,
        "epoch_completed": config.fixed_epoch,
        "fixed_epoch": config.fixed_epoch,
        "selection_policy": CHECKPOINT_SELECTION_POLICY,
        "protocol_sha256": canonical_sha256(protocol),
        "data_contract_sha256": protocol["data_contract_sha256"],
        "training_contract_sha256": protocol["u0_training_contract_sha256"],
    }
    changed = {key: (payload.get(key), value) for key, value in expected.items() if payload.get(key) != value}
    if changed:
        raise RuntimeError(f"Stage5 U0 import completed-endpoint contract mismatch: {changed}")
    model_state = payload.get("model_state")
    if (
        not isinstance(model_state, dict)
        or not model_state
        or state_dict_sha256(model_state) != payload.get("model_state_sha256")
    ):
        raise RuntimeError("Stage5 U0 import model-state digest mismatch")
    for key in ("optimizer_state", "scaler_state", "rng_state"):
        if not isinstance(payload.get(key), dict):
            raise RuntimeError(f"Stage5 U0 import checkpoint is missing {key}")
    if set(payload["rng_state"]) != {"python", "numpy", "torch_cpu", "torch_cuda"}:
        raise RuntimeError("Stage5 U0 import RNG state is incomplete")
    metrics = _validate_runtime_checkpoint_metadata(
        payload,
        role="U0",
        variant="U0",
        seed=seed,
        config=config,
        expected_git_head=protocol["git_head"],
        expected_base_checkpoint_sha256=None,
        expected_initial_controller_state_sha256=None,
        expected_source_contract_sha256=None,
    )
    metrics_bytes = readable_json_bytes(metrics)
    if hashlib.sha256(metrics_bytes).hexdigest() != payload.get("metrics_sha256"):
        raise RuntimeError("Stage5 U0 import embedded metrics do not authenticate the metrics file")
    for row in metrics["epochs"]:
        if require_int(row.get("pairs"), "U0 epoch pairs") != 294:
            raise RuntimeError("Stage5 U0 import requires every full 294-pair training epoch")
        require_sha256(row.get("pair_schedule_sha256"), "U0 epoch pair schedule")
    metrics_path = _plain_path(path.with_name("metrics.json"))
    if metrics_path.exists():
        require_regular_file(metrics_path, "U0 metrics")
        if metrics_path.read_bytes() != metrics_bytes:
            raise RuntimeError("Stage5 U0 import external metrics disagree with the completed checkpoint")
    if sha256_file(path) != digest:
        raise RuntimeError("Stage5 U0 import source changed while it was being read")
    record = {
        "seed": seed,
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": digest,
        "model_state_sha256": payload["model_state_sha256"],
        "metrics_sha256": payload["metrics_sha256"],
        "metrics_payload_sha256": payload["metrics_payload_sha256"],
        "preserved_training_state_sha256": _preserved_digest(payload),
        "epoch_completed": payload["epoch_completed"],
    }
    return payload, record


def import_completed_u0(
    *,
    source_protocol: Path,
    target_protocol: Path,
    source_checkpoint_root: Path,
    target_checkpoint_root: Path,
    output_manifest: Path,
) -> dict[str, Any]:
    """Create new U0 copies, or verify an identical complete prior import.

    All three source endpoints must verify before any target checkpoint is written.
    A partially written target is rejected for inspection, never overwritten. Callers
    must hold the target run lock; the source run must no longer be writing U0.
    """
    source_protocol, target_protocol = map(_plain_path, (source_protocol, target_protocol))
    source_root, target_root = map(_plain_path, (source_checkpoint_root, target_checkpoint_root))
    output_manifest = _plain_path(output_manifest)
    if source_root == target_root or source_root in target_root.parents or target_root in source_root.parents:
        raise RuntimeError("Stage5 U0 import source and target checkpoint roots must be disjoint")
    if source_root in output_manifest.parents or output_manifest == source_protocol:
        raise RuntimeError("Stage5 U0 import manifest must not write into the source artifacts")
    source, source_config = _load_protocol(source_protocol)
    target, target_config = _load_protocol(target_protocol)
    _validate_hotfix(source, target)

    source_records = []
    for seed in STAGE5_SEEDS:
        source_path = source_root / "u0" / f"seed_{seed}" / "last.pth"
        payload, record = _load_endpoint(source_path, seed=seed, protocol=source, config=source_config)
        if "u0_import_lineage" in payload or "training_git_head" in payload:
            raise RuntimeError("Stage5 U0 recursive imports are not supported")
        source_records.append(record)
        del payload

    manifest = {
        "schema": IMPORT_SCHEMA,
        "status": "COMPLETE",
        "operation": "IMPORT_COMPLETED_U0_WITHOUT_TRAINING",
        "source_protocol": source,
        "source_protocol_path": str(source_protocol),
        "source_protocol_file_sha256": sha256_file(source_protocol),
        "source_protocol_sha256": canonical_sha256(source),
        "target_protocol_path": str(target_protocol),
        "target_protocol_file_sha256": sha256_file(target_protocol),
        "target_protocol_sha256": canonical_sha256(target),
        "training_git_head": source["git_head"],
        "target_binding_git_head": target["git_head"],
        "source_checkpoint_root": str(source_root),
        "target_checkpoint_root": str(target_root),
        "source_checkpoints": source_records,
    }
    existing = output_manifest.exists()
    if existing:
        require_regular_file(output_manifest, "U0 import manifest")
        previous = load_json_object(output_manifest)
        if {key: value for key, value in previous.items() if key != "target_checkpoints"} != manifest:
            raise RuntimeError("Stage5 U0 import manifest differs from this exact source and target")
    elif target_root.exists() and any(target_root.iterdir()):
        raise RuntimeError("Stage5 U0 import refuses a nonempty target without its complete import manifest")

    target_records = []
    for source_record in source_records:
        seed = source_record["seed"]
        source_path = Path(source_record["path"])
        payload, checked = _load_endpoint(source_path, seed=seed, protocol=source, config=source_config)
        if checked != source_record:
            raise RuntimeError("Stage5 U0 import source changed after preflight")
        lineage = {
            "schema": IMPORT_SCHEMA,
            "operation": "IMPORT_COMPLETED_U0_WITHOUT_TRAINING",
            "source_training_git_head": source["git_head"],
            "source_protocol": source,
            "source_protocol_sha256": canonical_sha256(source),
            "source_checkpoint": source_record,
            "target_binding_git_head": target["git_head"],
            "target_protocol_sha256": canonical_sha256(target),
            "legacy_git_head_field_semantics": "TARGET_BINDING_NOT_TRAINING_REVISION",
        }
        target_path = _plain_path(target_root / "u0" / f"seed_{seed}" / "last.pth")
        payload["git_head"] = target["git_head"]
        payload["protocol_sha256"] = canonical_sha256(target)
        payload["training_git_head"] = source["git_head"]
        payload["u0_import_lineage"] = lineage
        if not existing:
            if target_path.exists() or target_path.parent.exists():
                raise RuntimeError("Stage5 U0 import refuses a partially written target")
            digest = atomic_torch_save(target_path, payload)
            write_immutable_json(
                target_path.with_name("last.pth.sha256.json"),
                {
                    "schema": "ctcf-stage5-checkpoint-sha256-v1",
                    "file_name": "last.pth",
                    "bytes": target_path.stat().st_size,
                    "sha256": digest,
                },
            )
            write_immutable_bytes(
                target_path.with_name("metrics.json"), readable_json_bytes(payload["metrics_payload"])
            )
        del payload
        copied, target_record = _load_endpoint(target_path, seed=seed, protocol=target, config=target_config)
        if copied.get("u0_import_lineage") != lineage or copied.get("training_git_head") != source["git_head"]:
            raise RuntimeError("Stage5 U0 import target training lineage changed")
        if target_record["preserved_training_state_sha256"] != source_record["preserved_training_state_sha256"]:
            raise RuntimeError("Stage5 U0 import changed preserved model, optimizer, scaler, RNG, or metric state")
        if not target_path.with_name("metrics.json").is_file():
            raise RuntimeError("Stage5 U0 import target metrics file is missing")
        target_records.append(target_record)
        del copied
    manifest["target_checkpoints"] = target_records
    write_immutable_json(output_manifest, manifest)
    return manifest


__all__ = ["import_completed_u0"]
