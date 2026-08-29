from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

from experiments.stage5.checkpoints import STAGE5_TRAINING_STATE_SCHEMA, state_dict_sha256
from experiments.stage5.config import ControllerTrainingConfig, build_stage5_controller
from tools.analysis.run_artifacts import sha256_file
from tools.analysis.search.pyramid import array_sha256
from tools.analysis.search.transaction import load_flow_npz
from tools.analysis.stage5.contracts import (
    CHECKPOINT_SCHEMA,
    CHECKPOINT_SELECTION_POLICY,
    canonical_sha256,
    validate_checkpoint_metadata,
)


def _relative_file(root: Path, path: Path) -> tuple[Path, str]:
    root = root.resolve(strict=True)
    path = path.resolve(strict=True)
    if not root.is_dir() or not path.is_file():
        raise FileNotFoundError(path)
    try:
        relative = path.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"artifact escapes its declared root: {path}") from exc
    if path.is_symlink() or any(parent.is_symlink() for parent in path.parents if parent != root.parent):
        raise RuntimeError(f"Stage5 artifacts must not traverse symlinks: {path}")
    return path, relative.as_posix()


def file_record(root_id: str, root: Path, path: Path) -> dict[str, Any]:
    resolved, relative = _relative_file(root, path)
    return {
        "root_id": root_id,
        "relative_path": relative,
        "bytes": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
    }


def field_record(root_id: str, root: Path, path: Path) -> dict[str, Any]:
    record = file_record(root_id, root, path)
    record["array_sha256"] = array_sha256(load_flow_npz(path))
    return record


def save_reload_attestation(
    record: Mapping[str, Any],
    *,
    in_memory_array_sha256: str,
    reloaded_path: Path,
) -> dict[str, Any]:
    reloaded_sha = array_sha256(load_flow_npz(reloaded_path))
    if sha256_file(reloaded_path) != record["sha256"]:
        raise RuntimeError("Stage5 persisted field changed before attestation")
    return {
        "file_sha256": record["sha256"],
        "in_memory_array_sha256": in_memory_array_sha256,
        "reloaded_array_sha256": reloaded_sha,
        "reloaded_from_persisted_bytes": True,
    }


def load_canonical_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return payload


def checkpoint_metadata(
    *,
    checkpoint_id: str,
    checkpoint_path: Path,
    checkpoint_root: Path,
    metrics_path: Path,
    protocol: Mapping[str, Any],
) -> dict[str, Any]:
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or payload.get("schema") != STAGE5_TRAINING_STATE_SCHEMA:
        raise RuntimeError(f"invalid Stage5 checkpoint: {checkpoint_path}")
    state = payload.get("model_state")
    if not isinstance(state, dict) or state_dict_sha256(state) != payload.get("model_state_sha256"):
        raise RuntimeError("Stage5 checkpoint state digest mismatch")
    role = str(payload.get("role"))
    variant = str(payload.get("variant_id"))
    if role == "CONTROLLER":
        parameter_count = sum(int(value.numel()) for value in state.values())
        reference_count = sum(
            parameter.numel() for parameter in build_stage5_controller(ControllerTrainingConfig()).parameters()
        )
        if parameter_count != reference_count:
            raise RuntimeError("Stage5 controller checkpoint parameter count changed")
    elif role == "U0":
        parameter_count = 0
    else:
        raise RuntimeError("unknown Stage5 checkpoint role")
    metadata = {
        "schema": CHECKPOINT_SCHEMA,
        "checkpoint_id": checkpoint_id,
        "role": role,
        "variant_id": variant,
        "seed": int(payload["seed"]),
        "fixed_epoch": int(payload["fixed_epoch"]),
        "selection_policy": payload["selection_policy"],
        "git_head": payload["git_head"],
        "protocol_sha256": payload["protocol_sha256"],
        "data_contract_sha256": payload["data_contract_sha256"],
        "training_contract_sha256": payload["training_contract_sha256"],
        "checkpoint_file": file_record("checkpoint_root", checkpoint_root, checkpoint_path),
        "state_dict_sha256": payload["model_state_sha256"],
        "metrics_sha256": sha256_file(metrics_path),
        "base_checkpoint_sha256": payload.get("base_checkpoint_sha256"),
        "initial_controller_state_sha256": payload.get("initial_controller_state_sha256"),
        "source_contract_sha256": payload.get("source_contract_sha256"),
        "controller_parameter_count": parameter_count,
    }
    if metadata["selection_policy"] != CHECKPOINT_SELECTION_POLICY:
        raise RuntimeError("Stage5 checkpoint was selected by a forbidden policy")
    if payload.get("metrics_sha256") != metadata["metrics_sha256"]:
        raise RuntimeError("Stage5 checkpoint metrics digest mismatch")
    validate_checkpoint_metadata(metadata, protocol)
    return metadata


def execution_sha256(payload: Mapping[str, Any]) -> str:
    return canonical_sha256(payload)


__all__ = [
    "checkpoint_metadata",
    "execution_sha256",
    "field_record",
    "file_record",
    "load_canonical_json",
    "save_reload_attestation",
]
