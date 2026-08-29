from __future__ import annotations

import hashlib
import io
import os
import random
import tempfile
from contextlib import suppress
from pathlib import Path
from typing import Any

import numpy as np
import torch

from tools.analysis.run_artifacts import sha256_file
from tools.analysis.stage5.contracts import CHECKPOINT_SELECTION_POLICY

STAGE5_TRAINING_STATE_SCHEMA = "ctcf-stage5-training-state-v1"


def state_dict_sha256(state_dict: dict[str, torch.Tensor]) -> str:
    buffer = io.BytesIO()
    torch.save({key: value.detach().cpu().contiguous() for key, value in sorted(state_dict.items())}, buffer)
    return hashlib.sha256(buffer.getvalue()).hexdigest()


def capture_rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def restore_rng_state(state: dict[str, Any]) -> None:
    required = {"python", "numpy", "torch_cpu", "torch_cuda"}
    if set(state) != required:
        raise RuntimeError("Stage5 RNG-state fields changed")
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def atomic_torch_save(path: Path, payload: dict[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".part", dir=path.parent)
    os.close(fd)
    try:
        torch.save(payload, temporary_name)
        os.replace(temporary_name, path)
    except BaseException:
        with suppress(FileNotFoundError):
            os.unlink(temporary_name)
        raise
    return sha256_file(path)


def build_training_state(
    *,
    role: str,
    variant_id: str,
    seed: int,
    epoch_completed: int,
    fixed_epoch: int,
    git_head: str,
    protocol_sha256: str,
    data_contract_sha256: str,
    training_contract_sha256: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    pair_schedule_sha256: str,
    metrics_sha256: str,
    base_checkpoint_sha256: str | None = None,
    initial_controller_state_sha256: str | None = None,
    source_contract_sha256: str | None = None,
) -> dict[str, Any]:
    if role not in {"U0", "CONTROLLER"}:
        raise ValueError("Stage5 checkpoint role must be U0 or CONTROLLER")
    if epoch_completed < 0 or fixed_epoch < 1 or epoch_completed > fixed_epoch:
        raise ValueError("invalid Stage5 checkpoint epoch")
    model_state = model.state_dict()
    return {
        "schema": STAGE5_TRAINING_STATE_SCHEMA,
        "role": role,
        "variant_id": variant_id,
        "seed": seed,
        "epoch_completed": epoch_completed,
        "fixed_epoch": fixed_epoch,
        "selection_policy": CHECKPOINT_SELECTION_POLICY,
        "git_head": git_head,
        "protocol_sha256": protocol_sha256,
        "data_contract_sha256": data_contract_sha256,
        "training_contract_sha256": training_contract_sha256,
        "pair_schedule_sha256": pair_schedule_sha256,
        "metrics_sha256": metrics_sha256,
        "base_checkpoint_sha256": base_checkpoint_sha256,
        "initial_controller_state_sha256": initial_controller_state_sha256,
        "source_contract_sha256": source_contract_sha256,
        "model_state": model_state,
        "model_state_sha256": state_dict_sha256(model_state),
        "optimizer_state": optimizer.state_dict(),
        "scaler_state": scaler.state_dict(),
        "rng_state": capture_rng_state(),
    }


def load_training_state(
    path: Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None,
    scaler: torch.amp.GradScaler | None,
    expected_role: str,
    expected_variant: str,
    expected_seed: int,
    expected_protocol_sha256: str,
    expected_data_contract_sha256: str,
    expected_training_contract_sha256: str,
    restore_rng: bool,
) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or payload.get("schema") != STAGE5_TRAINING_STATE_SCHEMA:
        raise RuntimeError("invalid Stage5 training checkpoint schema")
    expected = {
        "role": expected_role,
        "variant_id": expected_variant,
        "seed": expected_seed,
        "protocol_sha256": expected_protocol_sha256,
        "data_contract_sha256": expected_data_contract_sha256,
        "training_contract_sha256": expected_training_contract_sha256,
        "selection_policy": CHECKPOINT_SELECTION_POLICY,
    }
    changed = {key: (payload.get(key), value) for key, value in expected.items() if payload.get(key) != value}
    if changed:
        raise RuntimeError(f"Stage5 checkpoint contract mismatch: {changed}")
    state = payload.get("model_state")
    if not isinstance(state, dict) or state_dict_sha256(state) != payload.get("model_state_sha256"):
        raise RuntimeError("Stage5 checkpoint model-state digest mismatch")
    incompatible = model.load_state_dict(state, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError("Stage5 strict checkpoint load reported incompatible keys")
    if optimizer is not None:
        optimizer.load_state_dict(payload["optimizer_state"])
    if scaler is not None:
        scaler.load_state_dict(payload["scaler_state"])
    if restore_rng:
        restore_rng_state(payload["rng_state"])
    return payload


__all__ = [
    "STAGE5_TRAINING_STATE_SCHEMA",
    "atomic_torch_save",
    "build_training_state",
    "capture_rng_state",
    "load_training_state",
    "restore_rng_state",
    "state_dict_sha256",
]
