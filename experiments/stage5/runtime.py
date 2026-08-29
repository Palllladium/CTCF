from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import random
import time
from collections.abc import Mapping
from contextlib import suppress
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch

from datasets.OASIS100 import Stage5OasisImageStore
from experiments.core.helpers import select_lr_policy
from experiments.stage5.checkpoints import (
    atomic_torch_save,
    build_training_state,
    load_training_state,
    state_dict_sha256,
)
from experiments.stage5.config import (
    ControllerTrainingConfig,
    U0TrainingConfig,
    build_stage5_controller,
    config_sha256,
    require_seed,
)
from experiments.stage5.features import build_stage5_features
from experiments.stage5.losses import controller_objective
from experiments.stage5.safety import (
    BOOTSTRAP_POLICIES,
    WORK_EPS,
    commit_controller_delta,
    construct_initial_field,
    prepare_initial_field,
)
from experiments.train_CTCF import Runner
from models.CTCF.controller import STAGE5_VARIANTS, Stage5SpatialController
from tools.analysis.run_artifacts import atomic_write_json, sha256_file
from tools.analysis.search.pyramid import array_sha256
from tools.analysis.search.transaction import load_flow_npz, phi_to_psi_displacement
from tools.analysis.stage5.contracts import CHECKPOINT_SELECTION_POLICY, canonical_sha256
from tools.analysis.stage5.primitives import (
    atomic_write_bytes,
    readable_json_bytes,
    require_git_sha,
    require_sha256,
)

PAIR_SCHEDULE_DOMAIN = "CTCF-STAGE5-PAIR-SCHEDULE-V1\0"
CONTROLLER_PAIR_DOMAIN = "CTCF-STAGE5-CONTROLLER-PAIR-V2\0"
_U0_FORBIDDEN_METRIC_KEYS = frozenset({"dice", "dice_tr", "segmentation", "label"})


def _execution_determinism_contract() -> dict[str, Any]:
    return {
        "python_numpy_torch_rng_seeded": True,
        "deterministic_algorithms": False,
        "cudnn_deterministic": True,
        "cudnn_benchmark": False,
        "known_limitation": "CUDA 3-D grid_sample backward has no strict deterministic implementation",
    }


def _require_cuda(device: torch.device, phase: str) -> None:
    if not isinstance(device, torch.device) or device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError(f"Stage5 {phase} requires a visible CUDA GPU; CPU execution is not supported")


def _checkpoint_sidecar_path(path: Path) -> Path:
    return path.with_name(f"{path.name}.sha256.json")


def _checkpoint_generations(path: Path) -> tuple[Path, Path, Path]:
    """The live checkpoint and the two crash-recovery generations beside it."""
    return (path, path.with_name(f".{path.name}.next"), path.with_name(f".{path.name}.previous"))


def _checkpoint_generation_paths(path: Path) -> tuple[Path, ...]:
    checkpoints = _checkpoint_generations(path)
    return (*checkpoints, *(_checkpoint_sidecar_path(candidate) for candidate in checkpoints))


def _checkpoint_sidecar_record(logical_path: Path, stored_path: Path, digest: str) -> dict[str, Any]:
    return {
        "schema": "ctcf-stage5-checkpoint-sha256-v1",
        "file_name": logical_path.name,
        "bytes": stored_path.stat().st_size,
        "sha256": digest,
    }


def _write_checkpoint_with_sidecar(path: Path, payload: dict[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or _checkpoint_sidecar_path(path).exists():
        _verify_checkpoint_sidecar(path)
    _, staging, backup = _checkpoint_generations(path)
    staging_sidecar = _checkpoint_sidecar_path(staging)
    backup_sidecar = _checkpoint_sidecar_path(backup)
    for stale in (staging, staging_sidecar, backup, backup_sidecar):
        with suppress(FileNotFoundError):
            stale.unlink()
    digest = atomic_torch_save(staging, payload)
    atomic_write_json(staging_sidecar, _checkpoint_sidecar_record(path, staging, digest))
    if path.exists():
        os.replace(path, backup)
        os.replace(_checkpoint_sidecar_path(path), backup_sidecar)
    os.replace(staging, path)
    os.replace(staging_sidecar, _checkpoint_sidecar_path(path))
    _verify_checkpoint_sidecar(path)
    for obsolete in (backup, backup_sidecar):
        with suppress(FileNotFoundError):
            obsolete.unlink()
    return digest


def _verify_checkpoint_sidecar(path: Path) -> str:
    path = path.resolve()
    checkpoint_candidates = _checkpoint_generations(path)
    sidecar_candidates = tuple(_checkpoint_sidecar_path(candidate) for candidate in checkpoint_candidates)
    records: list[tuple[Path, dict[str, Any]]] = []
    for sidecar in sidecar_candidates:
        if not sidecar.is_file():
            continue
        try:
            record = json.loads(sidecar.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(record, dict):
            records.append((sidecar, record))
    matches: list[tuple[int, Path, Path, dict[str, Any]]] = []
    for priority, checkpoint in enumerate(checkpoint_candidates):
        if not checkpoint.is_file():
            continue
        digest = sha256_file(checkpoint)
        expected = _checkpoint_sidecar_record(path, checkpoint, digest)
        for sidecar, record in records:
            if record == expected:
                matches.append((priority, checkpoint, sidecar, record))
    if not matches:
        raise RuntimeError(f"Stage5 checkpoint bytes or sidecar changed, with no recoverable generation: {path}")
    _, checkpoint, sidecar, record = min(matches, key=lambda item: item[0])
    if checkpoint != path:
        os.replace(checkpoint, path)
    canonical_sidecar = _checkpoint_sidecar_path(path)
    if sidecar != canonical_sidecar:
        os.replace(sidecar, canonical_sidecar)
    for candidate in (*checkpoint_candidates[1:], *sidecar_candidates[1:]):
        with suppress(FileNotFoundError):
            candidate.unlink()
    return str(record["sha256"])


def _validate_metrics_payload(
    payload: Any,
    *,
    role: str,
    variant: str,
    seed: int,
    epoch_completed: int,
    pair_schedule_sha256: str,
) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise RuntimeError("Stage5 checkpoint has no embedded metrics object")
    expected_schema = "ctcf-stage5-u0-metrics-v1" if role == "U0" else "ctcf-stage5-controller-metrics-v1"
    expected = {
        "schema": expected_schema,
        "role": role,
        "variant": variant,
        "seed": seed,
        "label_metrics_present": False,
        "selection_policy": CHECKPOINT_SELECTION_POLICY,
    }
    changed = {key: (payload.get(key), value) for key, value in expected.items() if payload.get(key) != value}
    if changed:
        raise RuntimeError(f"Stage5 embedded metrics contract mismatch: {changed}")
    rows = payload.get("epochs")
    if not isinstance(rows, list) or len(rows) != epoch_completed:
        raise RuntimeError("Stage5 embedded metrics do not match the checkpoint epoch")
    if [row.get("epoch") for row in rows if isinstance(row, dict)] != list(range(1, epoch_completed + 1)):
        raise RuntimeError("Stage5 embedded metric epochs are not contiguous")
    if epoch_completed and rows[-1].get("pair_schedule_sha256") != pair_schedule_sha256:
        raise RuntimeError("Stage5 checkpoint and embedded metrics disagree on their final schedule")
    if any(
        any(forbidden in str(key).lower() for forbidden in _U0_FORBIDDEN_METRIC_KEYS)
        for row in rows
        for key in row.get("metrics", {})
    ):
        raise RuntimeError("Stage5 label-derived metric names are forbidden before the decision barrier")
    return payload


def _attach_runtime_checkpoint_metadata(
    payload: dict[str, Any],
    *,
    config: U0TrainingConfig | ControllerTrainingConfig,
    metrics_payload: dict[str, Any],
) -> None:
    payload["training_config_sha256"] = config_sha256(config)
    payload["metrics_payload"] = metrics_payload
    payload["metrics_payload_sha256"] = canonical_sha256(metrics_payload)
    payload["execution_determinism"] = _execution_determinism_contract()


def _validate_runtime_checkpoint_metadata(
    payload: dict[str, Any],
    *,
    role: str,
    variant: str,
    seed: int,
    config: U0TrainingConfig | ControllerTrainingConfig,
    expected_git_head: str | None,
    expected_base_checkpoint_sha256: str | None,
    expected_initial_controller_state_sha256: str | None,
    expected_source_contract_sha256: str | None,
) -> dict[str, Any]:
    if payload.get("fixed_epoch") != config.fixed_epoch:
        raise RuntimeError("Stage5 checkpoint fixed endpoint changed")
    if payload.get("training_config_sha256") != config_sha256(config):
        raise RuntimeError("Stage5 checkpoint training configuration changed")
    if payload.get("execution_determinism") != _execution_determinism_contract():
        raise RuntimeError("Stage5 checkpoint execution-determinism contract changed")
    if expected_git_head is not None and payload.get("git_head") != expected_git_head:
        raise RuntimeError("Stage5 resume checkpoint belongs to another Git revision")
    if payload.get("base_checkpoint_sha256") != expected_base_checkpoint_sha256:
        raise RuntimeError("Stage5 controller checkpoint is bound to another U0 endpoint")
    if payload.get("initial_controller_state_sha256") != expected_initial_controller_state_sha256:
        raise RuntimeError("Stage5 controller checkpoint is bound to another common initial state")
    if payload.get("source_contract_sha256") != expected_source_contract_sha256:
        raise RuntimeError("Stage5 checkpoint is bound to another training-source contract")
    metrics = _validate_metrics_payload(
        payload.get("metrics_payload"),
        role=role,
        variant=variant,
        seed=seed,
        epoch_completed=int(payload.get("epoch_completed", -1)),
        pair_schedule_sha256=str(payload.get("pair_schedule_sha256", "")),
    )
    if payload.get("metrics_payload_sha256") != canonical_sha256(metrics):
        raise RuntimeError("Stage5 embedded metrics digest mismatch")
    return metrics


def _restore_authoritative_metrics(
    metrics_path: Path,
    payload: dict[str, Any],
    metrics: dict[str, Any],
) -> None:
    expected_file_sha = str(payload.get("metrics_sha256", ""))
    require_sha256(expected_file_sha, "checkpoint metrics digest", error=ValueError)
    expected_bytes = readable_json_bytes(metrics)
    if hashlib.sha256(expected_bytes).hexdigest() != expected_file_sha:
        raise RuntimeError("Stage5 checkpoint metrics digest does not match its embedded metrics")
    if not metrics_path.is_file() or sha256_file(metrics_path) != expected_file_sha:
        atomic_write_bytes(metrics_path, readable_json_bytes(metrics))
    if sha256_file(metrics_path) != expected_file_sha:
        raise RuntimeError("could not restore Stage5 metrics to the checkpoint-authoritative state")


@dataclass(frozen=True, slots=True)
class _RunIdentity:
    """Everything that distinguishes a U0 run from a controller run at checkpoint time.

    U0 and controller training share their resume protocol and their per-epoch commit
    byte-for-byte; only these bindings differ, so they are passed rather than restated.
    """

    role: str
    variant: str
    seed: int
    git_head: str
    protocol_sha256: str
    data_contract_sha256: str
    training_contract_sha256: str
    metrics_schema: str
    label: str
    base_checkpoint_sha256: str | None = None
    initial_controller_state_sha256: str | None = None
    source_contract_sha256: str | None = None

    def metrics_payload(self, epochs: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "schema": self.metrics_schema,
            "role": self.role,
            "variant": self.variant,
            "seed": self.seed,
            "label_metrics_present": False,
            "selection_policy": CHECKPOINT_SELECTION_POLICY,
            "epochs": epochs,
        }


def _adopt_existing_endpoint(
    output_root: Path,
    resume: Path | None,
    *,
    identity: _RunIdentity,
    config: U0TrainingConfig | ControllerTrainingConfig,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
) -> tuple[int, list[dict[str, Any]]]:
    """Restore an interrupted run in place, refusing an output tree it cannot account for.

    Returns the epoch to resume from and the checkpoint-authoritative metric rows. A
    ``start_epoch`` equal to the frozen endpoint means the run is already complete.
    """
    checkpoint_path = output_root / "last.pth"
    metrics_path = output_root / "metrics.json"
    if resume is None and any(path.exists() for path in _checkpoint_generation_paths(checkpoint_path)):
        _verify_checkpoint_sidecar(checkpoint_path)
        resume = checkpoint_path
    if resume is None:
        if any(path.exists() for path in (checkpoint_path, _checkpoint_sidecar_path(checkpoint_path), metrics_path)):
            raise FileExistsError(
                f"Stage5 {identity.label} output already exists; pass its checkpoint explicitly via resume"
            )
        return 0, []

    resume = resume.resolve()
    _verify_checkpoint_sidecar(resume)
    state = load_training_state(
        resume,
        model=model,
        optimizer=optimizer,
        scaler=scaler,
        expected_role=identity.role,
        expected_variant=identity.variant,
        expected_seed=identity.seed,
        expected_protocol_sha256=identity.protocol_sha256,
        expected_data_contract_sha256=identity.data_contract_sha256,
        expected_training_contract_sha256=identity.training_contract_sha256,
        restore_rng=True,
    )
    authoritative_metrics = _validate_runtime_checkpoint_metadata(
        state,
        role=identity.role,
        variant=identity.variant,
        seed=identity.seed,
        config=config,
        expected_git_head=identity.git_head,
        expected_base_checkpoint_sha256=identity.base_checkpoint_sha256,
        expected_initial_controller_state_sha256=identity.initial_controller_state_sha256,
        expected_source_contract_sha256=identity.source_contract_sha256,
    )
    start_epoch = int(state["epoch_completed"])
    if start_epoch > config.fixed_epoch:
        raise RuntimeError(f"Stage5 {identity.label} resume epoch exceeds the frozen endpoint")
    _restore_authoritative_metrics(metrics_path, state, authoritative_metrics)
    if start_epoch == config.fixed_epoch and checkpoint_path.resolve() != resume:
        raise RuntimeError(
            f"a completed Stage5 {identity.label} checkpoint must be resumed in its own output directory"
        )
    return start_epoch, list(authoritative_metrics["epochs"])


def _commit_epoch(
    output_root: Path,
    *,
    identity: _RunIdentity,
    config: U0TrainingConfig | ControllerTrainingConfig,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    epoch: int,
    row: dict[str, Any],
    epochs: list[dict[str, Any]],
) -> None:
    """Persist one completed epoch: metrics first, then the checkpoint that authenticates them."""
    metrics_path = output_root / "metrics.json"
    payload = identity.metrics_payload(epochs)
    atomic_write_bytes(metrics_path, readable_json_bytes(payload))
    state = build_training_state(
        role=identity.role,
        variant_id=identity.variant,
        seed=identity.seed,
        epoch_completed=epoch + 1,
        fixed_epoch=config.fixed_epoch,
        git_head=identity.git_head,
        protocol_sha256=identity.protocol_sha256,
        data_contract_sha256=identity.data_contract_sha256,
        training_contract_sha256=identity.training_contract_sha256,
        model=model,
        optimizer=optimizer,
        scaler=scaler,
        pair_schedule_sha256=row["pair_schedule_sha256"],
        metrics_sha256=sha256_file(metrics_path),
        base_checkpoint_sha256=identity.base_checkpoint_sha256,
        initial_controller_state_sha256=identity.initial_controller_state_sha256,
        source_contract_sha256=identity.source_contract_sha256,
    )
    _attach_runtime_checkpoint_metadata(state, config=config, metrics_payload=payload)
    _write_checkpoint_with_sidecar(output_root / "last.pth", state)


def _strict_scaler_step(
    scaler: torch.amp.GradScaler,
    optimizer: torch.optim.Optimizer,
    *,
    phase: str,
) -> None:
    scale_before = float(scaler.get_scale())
    scaler.step(optimizer)
    scaler.update()
    if float(scaler.get_scale()) < scale_before:
        raise FloatingPointError(f"non-finite {phase} gradients caused an AMP optimizer-step skip")


def _sanitize_u0_logs(logs: Mapping[str, Any]) -> dict[str, float]:
    forbidden_present = {key for key in logs if any(name in key.lower() for name in _U0_FORBIDDEN_METRIC_KEYS)}
    for key in forbidden_present:
        value = float(logs[key])
        if value != 0.0:
            raise RuntimeError(f"label-derived U0 metric is non-zero in label-free training: {key}")
    result: dict[str, float] = {}
    for key, raw in logs.items():
        if key in forbidden_present:
            continue
        value = float(raw)
        if not math.isfinite(value):
            raise FloatingPointError(f"non-finite U0 metric: {key}")
        result[key] = value
    return result


def _digest(domain: str, *parts: object) -> str:
    payload = domain + "\0".join(str(part) for part in parts)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _training_subjects(store: Stage5OasisImageStore) -> tuple[str, ...]:
    return tuple(item["subject_id"] for item in store.runtime.split["training"])


def epoch_pair_schedule(subject_ids: tuple[str, ...], *, seed: int, epoch: int) -> tuple[tuple[str, str], ...]:
    require_seed(seed)
    if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 0:
        raise ValueError("Stage5 pair-schedule epoch must be a non-negative integer")
    if len(subject_ids) < 2 or len(subject_ids) != len(set(subject_ids)):
        raise ValueError("pair schedule requires at least two unique subjects")
    moving = sorted(subject_ids, key=lambda value: _digest(PAIR_SCHEDULE_DOMAIN, seed, epoch, "moving", value))
    fixed = sorted(subject_ids, key=lambda value: _digest(PAIR_SCHEDULE_DOMAIN, seed, epoch, "fixed", value))
    if any(left == right for left, right in zip(moving, fixed, strict=True)):
        for shift in range(1, len(fixed)):
            rotated = fixed[shift:] + fixed[:shift]
            if all(left != right for left, right in zip(moving, rotated, strict=True)):
                fixed = rotated
                break
        else:
            raise RuntimeError("could not construct a deterministic Stage5 derangement")
    pairs = tuple(zip(moving, fixed, strict=True))
    if len({left for left, _ in pairs}) != len(subject_ids) or len({right for _, right in pairs}) != len(subject_ids):
        raise RuntimeError("Stage5 epoch pair schedule is not a bijection")
    return pairs


def controller_epoch_pairs(
    subject_ids: tuple[str, ...],
    *,
    seed: int,
    epoch: int,
) -> tuple[dict[str, str], ...]:
    require_seed(seed)
    if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 0:
        raise ValueError("Stage5 controller-pair epoch must be a non-negative integer")
    if len(subject_ids) % 2 or len(subject_ids) != len(set(subject_ids)):
        raise ValueError("controller training requires an even unique subject inventory")
    ordered = sorted(subject_ids, key=lambda value: _digest(CONTROLLER_PAIR_DOMAIN, seed, epoch, value))
    records: list[dict[str, str]] = []
    for index in range(0, len(ordered), 2):
        first, second = ordered[index], ordered[index + 1]
        pair_id = f"S5TRAIN-E{epoch + 1:03d}-P{index // 2 + 1:03d}"
        records.append({"pair_id": pair_id, "subject_a": first, "subject_b": second})
    return tuple(records)


def development_case_inventory(store: Stage5OasisImageStore) -> tuple[dict[str, str], ...]:
    return tuple({**item, "split": "development"} for item in store.runtime.pairs["cases"])


def _u0_args(config: U0TrainingConfig, seed: int) -> SimpleNamespace:
    return SimpleNamespace(
        ds="OASIS",
        time_steps=config.time_steps,
        config=config.config_key,
        use_checkpoint=1,
        lr=config.learning_rate,
        w_dice=0.0,
        reg_mode="diffusion",
        ema_decay=0.0,
        ema_lambda=0.0,
        w_reg_l1=None,
        w_reg_l2=None,
        w_reg_l3=None,
        l1_base_ch=None,
        l3_base_ch=None,
        l3_error_mode=None,
        l3_corr_mode=None,
        l3_iters=None,
        l3_unshared=None,
        l1_half_res=None,
        l2_full_res=None,
        l3_full_res=None,
        l3_svf=None,
        l3_num_heads=None,
        l3_ls_space=None,
        l3_ls_eps=None,
        schedule_max_epoch=config.fixed_epoch,
        max_epoch=config.fixed_epoch,
        disable_l1=0,
        disable_l3=0,
        l1_from_start=0,
        w_ncc=config.w_ncc,
        w_reg=config.w_reg,
        w_icon=config.w_icon,
        w_jac=config.w_jac,
        icon_mode="l1",
        jac_mode="central",
        tri_pen_mode="bernstein",
        tri_pen_reduce="mean",
        log_tri_gradnorm=0,
        dare_beta=1.0,
        elastic_mu=1.0,
        elastic_lam=1.0,
        seed=seed,
    )


def _tensor_image(store: Stage5OasisImageStore, subject_id: str, device: torch.device) -> torch.Tensor:
    array = store.load_image(subject_id)
    return torch.from_numpy(array).unsqueeze(0).unsqueeze(0).to(device=device, dtype=torch.float32)


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Strict deterministic algorithms are incompatible with the CUDA 3-D
    # grid_sample backward required by both CTCF and the controller objective.
    torch.use_deterministic_algorithms(False)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def train_u0(
    *,
    data_contract: Path,
    image_root: Path,
    output_root: Path,
    seed: int,
    device: torch.device,
    git_head: str,
    protocol_sha256: str,
    training_contract_sha256: str,
    config: U0TrainingConfig,
    resume: Path | None = None,
) -> Path:
    _require_cuda(device, "U0 training")
    require_seed(seed)
    require_git_sha(git_head, "git_head", error=ValueError)
    require_sha256(protocol_sha256, "protocol_sha256", error=ValueError)
    require_sha256(training_contract_sha256, "training_contract_sha256", error=ValueError)
    _seed_everything(seed)
    store = Stage5OasisImageStore(data_contract, image_root)
    args = _u0_args(config, seed)
    try:
        runner = Runner(args, device)
    except Exception as exc:
        raise RuntimeError(
            "Stage5 could not construct the frozen CTCF-CascadeA-Mamba U0 model; "
            "verify the H100 environment and Mamba dependencies"
        ) from exc
    scaler = torch.amp.GradScaler("cuda", enabled=True)
    output_root.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_root / "last.pth"
    identity = _RunIdentity(
        role="U0",
        variant="U0",
        seed=seed,
        git_head=git_head,
        protocol_sha256=protocol_sha256,
        data_contract_sha256=store.runtime.contract_sha256,
        training_contract_sha256=training_contract_sha256,
        metrics_schema="ctcf-stage5-u0-metrics-v1",
        label="U0",
    )
    start_epoch, metrics = _adopt_existing_endpoint(
        output_root,
        resume,
        identity=identity,
        config=config,
        model=runner.model,
        optimizer=runner.optimizer,
        scaler=scaler,
    )
    if start_epoch == config.fixed_epoch:
        return checkpoint_path

    subjects = _training_subjects(store)
    for epoch in range(start_epoch, config.fixed_epoch):
        epoch_started = time.perf_counter()
        runner.model.train()
        lr = select_lr_policy(runner, runner.optimizer, epoch, config.fixed_epoch, config.learning_rate)
        schedule = epoch_pair_schedule(subjects, seed=seed, epoch=epoch)
        totals: dict[str, float] = {}
        completed = 0
        for moving_id, fixed_id in schedule:
            moving = _tensor_image(store, moving_id, device)
            fixed = _tensor_image(store, fixed_id, device)
            runner.optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True):
                loss, logs = runner.train_step((moving, fixed), epoch)
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError(f"non-finite U0 loss at epoch {epoch}")
            scaler.scale(loss).backward()
            _strict_scaler_step(scaler, runner.optimizer, phase="U0")
            for key, value in _sanitize_u0_logs(logs).items():
                totals[key] = totals.get(key, 0.0) + value
            completed += 1
        if completed != len(subjects):
            raise RuntimeError("Stage5 U0 epoch did not consume its full deterministic bijection")
        row = {
            "epoch": epoch + 1,
            "learning_rate": lr,
            "pairs": completed,
            "pair_schedule_sha256": canonical_sha256(schedule),
            "metrics": {key: value / completed for key, value in sorted(totals.items())},
        }
        metrics.append(row)
        _commit_epoch(
            output_root,
            identity=identity,
            config=config,
            model=runner.model,
            optimizer=runner.optimizer,
            scaler=scaler,
            epoch=epoch,
            row=row,
            epochs=metrics,
        )
        metric_text = " ".join(f"{key}={value:.6g}" for key, value in row["metrics"].items())
        print(
            f"[STAGE5 U0 EPOCH] seed={seed} epoch={epoch + 1}/{config.fixed_epoch} "
            f"pairs={completed} elapsed_seconds={time.perf_counter() - epoch_started:.1f} {metric_text}",
            flush=True,
        )
    _verify_checkpoint_sidecar(checkpoint_path)
    return checkpoint_path


def load_frozen_u0(
    checkpoint: Path,
    *,
    seed: int,
    device: torch.device,
    protocol_sha256: str,
    data_contract_sha256: str,
    training_contract_sha256: str,
    config: U0TrainingConfig,
    expected_git_head: str | None = None,
) -> Runner:
    _require_cuda(device, "U0 inference")
    require_seed(seed)
    _verify_checkpoint_sidecar(checkpoint)
    args = _u0_args(config, seed)
    try:
        runner = Runner(args, device)
    except Exception as exc:
        raise RuntimeError("Stage5 could not reconstruct the frozen Mamba U0 model") from exc
    state = load_training_state(
        checkpoint,
        model=runner.model,
        optimizer=None,
        scaler=None,
        expected_role="U0",
        expected_variant="U0",
        expected_seed=seed,
        expected_protocol_sha256=protocol_sha256,
        expected_data_contract_sha256=data_contract_sha256,
        expected_training_contract_sha256=training_contract_sha256,
        restore_rng=False,
    )
    _validate_runtime_checkpoint_metadata(
        state,
        role="U0",
        variant="U0",
        seed=seed,
        config=config,
        expected_git_head=expected_git_head,
        expected_base_checkpoint_sha256=None,
        expected_initial_controller_state_sha256=None,
        expected_source_contract_sha256=None,
    )
    if int(state["epoch_completed"]) != config.fixed_epoch:
        raise RuntimeError("U0 checkpoint is not the frozen fixed-epoch endpoint")
    runner.model.eval()
    runner.model.requires_grad_(False)
    return runner


def _exact_report_passed(payload: Any, *, require_identity_boundary: bool) -> bool:
    if not isinstance(payload, dict) or payload.get("status") != "CERTIFIED" or payload.get("certified") is not True:
        return False
    return not (require_identity_boundary and payload.get("boundary_nonzero_count") != 0)


def _require_regular_source_file(path: Path, label: str) -> None:
    if not path.is_file() or path.is_symlink():
        raise FileNotFoundError(f"missing regular Stage5 {label}: {path}")


def _validate_identity_bootstrap_report(
    nested: Mapping[str, Any],
    *,
    image_shape: tuple[int, int, int],
    phi_array_sha256: str,
    psi_array_sha256: str,
) -> None:
    """The identity bootstrap must be the authoritative zero field, not merely a field that scored well."""
    if (
        nested.get("scientific_degradation") != "IDENTITY_BASELINE"
        or nested.get("digital_residual_percent") != 0.0
        or nested.get("trilinear_repair") is not None
    ):
        raise RuntimeError("Stage5 identity bootstrap report changed")
    identity_phi = torch.zeros((1, 3, *image_shape), dtype=torch.float32)
    identity_psi = phi_to_psi_displacement(identity_phi).float()
    if phi_array_sha256 != array_sha256(identity_phi) or psi_array_sha256 != array_sha256(identity_psi):
        raise RuntimeError("Stage5 identity bootstrap arrays are not the authoritative identity fields")


def _validate_collar_repair_bootstrap_report(nested: Mapping[str, Any]) -> None:
    """The collar-repair bootstrap must carry a certified, finite trilinear bound at the work epsilon."""
    repair = nested.get("trilinear_repair")
    bound = repair.get("cert_bound") if isinstance(repair, dict) else None
    certified_repair = isinstance(repair, dict) and repair.get("certified") is True
    usable_bound = not isinstance(bound, bool) and isinstance(bound, (int, float)) and math.isfinite(bound)
    if (
        nested.get("scientific_degradation") is not None
        or nested.get("digital_residual_percent") != 0.0
        or not certified_repair
        or not usable_bound
        or bound < WORK_EPS
    ):
        raise RuntimeError("Stage5 collar-repair bootstrap report changed")


def validate_certified_source_artifact(
    case_root: Path,
    *,
    seed: int,
    case: Mapping[str, str],
    u0_checkpoint_sha256: str,
    image_shape: tuple[int, int, int],
    bootstrap_policy: str,
) -> dict[str, Any]:
    """Authenticate one immutable source record without trusting its report alone."""
    seed = require_seed(seed)
    u0_checkpoint_sha256 = require_sha256(u0_checkpoint_sha256, "U0 checkpoint SHA-256", error=ValueError)
    if bootstrap_policy not in BOOTSTRAP_POLICIES:
        raise ValueError(f"unsupported Stage5 source bootstrap policy: {bootstrap_policy}")
    if len(image_shape) != 3 or any(
        isinstance(size, bool) or not isinstance(size, int) or size < 2 for size in image_shape
    ):
        raise ValueError("Stage5 source image shape must contain three integer dimensions >= 2")
    case_root = case_root.resolve()
    report_path = case_root / "initial_report.json"
    phi_path = case_root / "initial_phi.npz"
    psi_path = case_root / "initial_psi.npz"
    for path, label in ((report_path, "source report"), (phi_path, "Phi"), (psi_path, "Psi")):
        _require_regular_source_file(path, label)
    try:
        payload = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"invalid Stage5 source report: {report_path}") from exc
    expected_fields = {
        "schema",
        "seed",
        "case",
        "u0_checkpoint_sha256",
        "bootstrap_policy",
        "phi_sha256",
        "psi_sha256",
        "report",
    }
    if not isinstance(payload, dict) or set(payload) != expected_fields:
        raise RuntimeError(f"Stage5 source report fields changed: {report_path}")
    expected_values = {
        "schema": "ctcf-stage5-certified-source-v1",
        "seed": seed,
        "case": dict(case),
        "u0_checkpoint_sha256": u0_checkpoint_sha256,
    }
    changed = {key: (payload.get(key), value) for key, value in expected_values.items() if payload.get(key) != value}
    if changed:
        raise RuntimeError(f"Stage5 source report provenance mismatch: {changed}")
    policy = payload.get("bootstrap_policy")
    if policy not in BOOTSTRAP_POLICIES:
        raise RuntimeError("Stage5 source report has an unknown bootstrap policy")
    if policy != bootstrap_policy:
        raise RuntimeError("Stage5 source report belongs to another frozen bootstrap policy")
    actual_phi_sha = sha256_file(phi_path)
    actual_psi_sha = sha256_file(psi_path)
    if payload.get("phi_sha256") != actual_phi_sha or payload.get("psi_sha256") != actual_psi_sha:
        raise RuntimeError("Stage5 source field bytes changed after certification")
    nested = payload.get("report")
    if not isinstance(nested, dict):
        raise RuntimeError("Stage5 source certification report is missing")
    if (
        nested.get("policy") != policy
        or nested.get("phi_sha256") != actual_phi_sha
        or nested.get("psi_sha256") != actual_psi_sha
        or not _exact_report_passed(nested.get("phi_exact"), require_identity_boundary=True)
        or not _exact_report_passed(nested.get("psi_exact"), require_identity_boundary=False)
    ):
        raise RuntimeError("Stage5 source field has no valid exact-certification attestation")
    phi_array_sha256 = require_sha256(
        nested["phi_exact"].get("sha256"),
        "Stage5 source exact Phi array SHA-256",
    )
    psi_array_sha256 = require_sha256(
        nested["psi_exact"].get("sha256"),
        "Stage5 source exact Psi array SHA-256",
    )
    if policy == "identity":
        _validate_identity_bootstrap_report(
            nested,
            image_shape=image_shape,
            phi_array_sha256=phi_array_sha256,
            psi_array_sha256=psi_array_sha256,
        )
    else:
        _validate_collar_repair_bootstrap_report(nested)
    return {
        "case": dict(case),
        "seed": seed,
        "u0_checkpoint_sha256": u0_checkpoint_sha256,
        "bootstrap_policy": policy,
        "report_sha256": sha256_file(report_path),
        "phi_sha256": actual_phi_sha,
        "psi_sha256": actual_psi_sha,
        "phi_array_sha256": phi_array_sha256,
        "psi_array_sha256": psi_array_sha256,
    }


class _CertifiedSourceStore:
    def __init__(
        self,
        root: Path,
        *,
        seed: int,
        u0_checkpoint_sha256: str,
        image_shape: tuple[int, int, int],
        bootstrap_policy: str,
    ) -> None:
        self.root = root.resolve()
        self.seed = require_seed(seed)
        self.u0_checkpoint_sha256 = require_sha256(u0_checkpoint_sha256, "U0 checkpoint SHA-256", error=ValueError)
        self.image_shape = image_shape
        if bootstrap_policy not in BOOTSTRAP_POLICIES:
            raise ValueError(f"unsupported Stage5 source bootstrap policy: {bootstrap_policy}")
        self.bootstrap_policy = bootstrap_policy
        self._records: dict[str, dict[str, Any]] = {}
        self._fingerprints: dict[Path, tuple[int, int]] = {}

    def _case_root(self, case_id: str) -> Path:
        path = (self.root / f"seed_{self.seed}" / case_id).resolve()
        try:
            path.relative_to(self.root)
        except ValueError as exc:
            raise RuntimeError("Stage5 source-field path escapes its declared root") from exc
        return path

    def verify_case(self, case: Mapping[str, str]) -> dict[str, Any]:
        case_id = str(case["case_id"])
        case_root = self._case_root(case_id)
        report_path = case_root / "initial_report.json"
        phi_path = case_root / "initial_phi.npz"
        psi_path = case_root / "initial_psi.npz"
        record = validate_certified_source_artifact(
            case_root,
            seed=self.seed,
            case=case,
            u0_checkpoint_sha256=self.u0_checkpoint_sha256,
            image_shape=self.image_shape,
            bootstrap_policy=self.bootstrap_policy,
        )
        self._records[case_id] = record
        for path in (report_path, phi_path, psi_path):
            stat = path.stat()
            self._fingerprints[path] = (stat.st_size, stat.st_mtime_ns)
        return record

    def inventory_sha256(self, cases: tuple[dict[str, str], ...]) -> str:
        if not cases or len({case["case_id"] for case in cases}) != len(cases):
            raise RuntimeError("Stage5 source inventory requires unique ordered development cases")
        records = [self.verify_case(case) for case in cases]
        policies = {record["bootstrap_policy"] for record in records}
        if len(policies) != 1:
            raise RuntimeError("Stage5 source inventory mixes bootstrap policies")
        return canonical_sha256(
            {
                "schema": "ctcf-stage5-development-source-inventory-v1",
                "seed": self.seed,
                "u0_checkpoint_sha256": self.u0_checkpoint_sha256,
                "records": records,
            }
        )

    def load(self, case: Mapping[str, str], device: torch.device) -> torch.Tensor:
        case_id = str(case["case_id"])
        if case_id not in self._records:
            self.verify_case(case)
        case_root = self._case_root(case_id)
        for path in (case_root / "initial_report.json", case_root / "initial_phi.npz", case_root / "initial_psi.npz"):
            _require_regular_source_file(path, "source artifact")
            stat = path.stat()
            if self._fingerprints.get(path) != (stat.st_size, stat.st_mtime_ns):
                raise RuntimeError(f"Stage5 source artifact changed during controller training: {path}")
        psi = load_flow_npz(case_root / "initial_psi.npz")
        expected_shape = (1, 3, *self.image_shape)
        if psi.shape != expected_shape or psi.dtype != torch.float32 or not bool(torch.isfinite(psi).all()):
            raise RuntimeError(f"invalid certified Stage5 source field tensor for {case_id}")
        if array_sha256(psi) != self._records[case_id]["psi_array_sha256"]:
            raise RuntimeError(f"Stage5 source array differs from its exact certificate: {case_id}")
        if self._records[case_id]["bootstrap_policy"] == "identity":
            expected_psi = phi_to_psi_displacement(torch.zeros_like(psi)).float()
            if not torch.equal(psi, expected_psi):
                raise RuntimeError(f"Stage5 source differs from the authoritative identity Psi: {case_id}")
        return psi.to(device=device, dtype=torch.float32)


def materialize_source_fields(
    *,
    data_contract: Path,
    image_root: Path,
    output_root: Path,
    u0_checkpoint: Path,
    seed: int,
    device: torch.device,
    protocol_sha256: str,
    u0_training_contract_sha256: str,
    u0_config: U0TrainingConfig,
    bootstrap_policy: str,
    shard_index: int,
    num_shards: int,
) -> int:
    _require_cuda(device, "source-field materialization")
    require_seed(seed)
    require_sha256(protocol_sha256, "protocol_sha256", error=ValueError)
    require_sha256(u0_training_contract_sha256, "u0_training_contract_sha256", error=ValueError)
    store = Stage5OasisImageStore(data_contract, image_root)
    u0_checkpoint_sha256 = _verify_checkpoint_sidecar(u0_checkpoint)
    runner = load_frozen_u0(
        u0_checkpoint,
        seed=seed,
        device=device,
        protocol_sha256=protocol_sha256,
        data_contract_sha256=store.runtime.contract_sha256,
        training_contract_sha256=u0_training_contract_sha256,
        config=u0_config,
    )
    inventory = development_case_inventory(store)
    if (
        isinstance(shard_index, bool)
        or isinstance(num_shards, bool)
        or not isinstance(shard_index, int)
        or not isinstance(num_shards, int)
        or num_shards < 1
        or not 0 <= shard_index < num_shards
    ):
        raise ValueError("invalid source-field shard")
    source_store = _CertifiedSourceStore(
        output_root,
        seed=seed,
        u0_checkpoint_sha256=u0_checkpoint_sha256,
        image_shape=store.image_shape,
        bootstrap_policy=bootstrap_policy,
    )
    completed = 0
    for index, case in enumerate(inventory):
        if index % num_shards != shard_index:
            continue
        case_root = output_root / f"seed_{seed}" / case["case_id"]
        report_path = case_root / "initial_report.json"
        if report_path.exists():
            source_store.verify_case(case)
            completed += 1
            continue
        moving = _tensor_image(store, case["moving_subject_id"], device)
        fixed = _tensor_image(store, case["fixed_subject_id"], device)
        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True):
            _, raw_phi = runner.model(moving, fixed, alpha_l1=1.0, alpha_l3=1.0)
        artifact = prepare_initial_field(raw_phi.float(), case_root, policy=bootstrap_policy)
        atomic_write_json(
            report_path,
            {
                "schema": "ctcf-stage5-certified-source-v1",
                "seed": seed,
                "case": case,
                "u0_checkpoint_sha256": u0_checkpoint_sha256,
                "bootstrap_policy": bootstrap_policy,
                "phi_sha256": artifact.phi_sha256,
                "psi_sha256": artifact.psi_sha256,
                "report": artifact.report,
            },
        )
        source_store.verify_case(case)
        completed += 1
    return completed


def initialize_controller_state(*, seed: int, config: ControllerTrainingConfig, output_path: Path) -> str:
    require_seed(seed)
    if any(path.exists() for path in _checkpoint_generation_paths(output_path)):
        _verify_checkpoint_sidecar(output_path)
        controller = build_stage5_controller(config)
        return _load_initial_controller(output_path, controller, seed=seed, config=config)
    _seed_everything(seed)
    controller = build_stage5_controller(config)
    payload = {
        "schema": "ctcf-stage5-controller-initial-state-v1",
        "seed": seed,
        "config": asdict(config),
        "config_sha256": config_sha256(config),
        "state": controller.state_dict(),
        "state_sha256": state_dict_sha256(controller.state_dict()),
    }
    _write_checkpoint_with_sidecar(output_path, payload)
    return payload["state_sha256"]


def _load_initial_controller(
    path: Path,
    controller: Stage5SpatialController,
    *,
    seed: int,
    config: ControllerTrainingConfig,
) -> str:
    _verify_checkpoint_sidecar(path)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("schema") != "ctcf-stage5-controller-initial-state-v1" or payload.get("seed") != seed:
        raise RuntimeError("controller initial-state contract mismatch")
    expected_config = asdict(config)
    observed_config = payload.get("config")
    if observed_config != expected_config:
        raise RuntimeError("controller initial state was built for another frozen configuration")
    if payload.get("config_sha256") != canonical_sha256(observed_config):
        raise RuntimeError("controller initial-state configuration digest mismatch")
    if state_dict_sha256(payload["state"]) != payload.get("state_sha256"):
        raise RuntimeError("controller initial-state digest mismatch")
    controller.load_state_dict(payload["state"], strict=True)
    return str(payload["state_sha256"])


def _controller_training_tensors(features: Any) -> tuple[torch.Tensor, ...]:
    source = (
        features.controller_input,
        features.s2.proposal,
        features.s4.proposal,
        features.fixed_normalized,
        features.moving_normalized,
    )
    result = tuple(value.float().clone() for value in source)
    if any(value.dtype != torch.float32 or value.is_inference() for value in result):
        raise RuntimeError("Stage5 features did not materialize as ordinary FP32 training tensors")
    return result


@dataclass(frozen=True, slots=True)
class _ControllerStep:
    """The collaborators one controller training step needs, fixed for the whole run."""

    store: Stage5OasisImageStore
    base_runner: Runner
    controller: Stage5SpatialController
    optimizer: torch.optim.Optimizer
    scaler: torch.amp.GradScaler
    device: torch.device
    variant: str
    bootstrap_policy: str
    config: ControllerTrainingConfig


def _controller_pair_step(step: _ControllerStep, pair: Mapping[str, str], epoch: int) -> dict[str, float]:
    """Take one optimizer step on one unordered pair, seen in both directions."""
    moving_ab = _tensor_image(step.store, pair["subject_a"], step.device)
    fixed_ab = _tensor_image(step.store, pair["subject_b"], step.device)
    moving_ba = fixed_ab
    fixed_ba = moving_ab
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True):
        _, raw_ab = step.base_runner.model(moving_ab, fixed_ab, alpha_l1=1.0, alpha_l3=1.0)
        _, raw_ba = step.base_runner.model(moving_ba, fixed_ba, alpha_l1=1.0, alpha_l3=1.0)
    _, psi_ab, _ = construct_initial_field(raw_ab, policy=step.bootstrap_policy)
    _, psi_ba, _ = construct_initial_field(raw_ba, policy=step.bootstrap_policy)
    features_ab = build_stage5_features(fixed_ab, moving_ab, psi_ab)
    features_ba = build_stage5_features(fixed_ba, moving_ba, psi_ba)
    input_ab, s2_ab, s4_ab, fixed_norm_ab, moving_norm_ab = _controller_training_tensors(features_ab)
    input_ba, s2_ba, s4_ba, fixed_norm_ba, moving_norm_ba = _controller_training_tensors(features_ba)
    step.optimizer.zero_grad(set_to_none=True)
    # The controller runs under FP16 autocast while controller_objective re-enters FP32 inside.
    # That boundary is the numerical contract of the run; do not widen or move it.
    with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True):
        output_ab = step.controller(input_ab, step.variant, s2_proposal=s2_ab, s4_proposal=s4_ab)
        output_ba = step.controller(input_ba, step.variant, s2_proposal=s2_ba, s4_proposal=s4_ba)
        loss, logs = controller_objective(
            fixed_norm_ab,
            moving_norm_ab,
            fixed_norm_ba,
            moving_norm_ba,
            psi_ab,
            psi_ba,
            output_ab.requested_delta,
            output_ba.requested_delta,
            config=step.config.loss,
        )
    if not bool(torch.isfinite(loss)):
        raise FloatingPointError(f"non-finite Stage5 controller loss at epoch {epoch}")
    step.scaler.scale(loss).backward()
    _strict_scaler_step(step.scaler, step.optimizer, phase=f"controller {step.variant}")
    return logs


def train_controller(
    *,
    data_contract: Path,
    image_root: Path,
    output_root: Path,
    base_checkpoint: Path,
    initial_controller: Path,
    variant: str,
    seed: int,
    device: torch.device,
    git_head: str,
    protocol_sha256: str,
    u0_training_contract_sha256: str,
    training_contract_sha256: str,
    bootstrap_policy: str,
    config: ControllerTrainingConfig,
    resume: Path | None = None,
) -> Path:
    _require_cuda(device, "controller training")
    require_seed(seed)
    require_git_sha(git_head, "git_head", error=ValueError)
    require_sha256(protocol_sha256, "protocol_sha256", error=ValueError)
    require_sha256(u0_training_contract_sha256, "u0_training_contract_sha256", error=ValueError)
    require_sha256(training_contract_sha256, "training_contract_sha256", error=ValueError)
    if bootstrap_policy not in BOOTSTRAP_POLICIES:
        raise ValueError(f"unsupported Stage5 controller bootstrap policy: {bootstrap_policy}")
    if variant not in STAGE5_VARIANTS:
        raise ValueError(f"unknown Stage5 controller variant: {variant}")
    _seed_everything(seed)
    store = Stage5OasisImageStore(data_contract, image_root)
    controller = build_stage5_controller(config).to(device)
    initial_sha = _load_initial_controller(initial_controller, controller, seed=seed, config=config)
    optimizer = torch.optim.AdamW(
        controller.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=True)
    base_sha = _verify_checkpoint_sidecar(base_checkpoint)
    base_runner = load_frozen_u0(
        base_checkpoint,
        seed=seed,
        device=device,
        protocol_sha256=protocol_sha256,
        data_contract_sha256=store.runtime.contract_sha256,
        training_contract_sha256=u0_training_contract_sha256,
        config=U0TrainingConfig(),
    )
    base_runner.model.eval().requires_grad_(False)
    training_subjects = _training_subjects(store)
    source_contract_sha256 = canonical_sha256(
        {
            "schema": "ctcf-stage5-on-the-fly-training-source-v1",
            "u0_checkpoint_sha256": base_sha,
            "data_contract_sha256": store.runtime.contract_sha256,
            "protocol_sha256": protocol_sha256,
            "pair_domain": CONTROLLER_PAIR_DOMAIN,
            "bootstrap_policy": bootstrap_policy,
            "source_policy": "shared_frozen_bootstrap_construction_on_the_fly",
        }
    )
    output_root.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_root / "last.pth"
    identity = _RunIdentity(
        role="CONTROLLER",
        variant=variant,
        seed=seed,
        git_head=git_head,
        protocol_sha256=protocol_sha256,
        data_contract_sha256=store.runtime.contract_sha256,
        training_contract_sha256=training_contract_sha256,
        metrics_schema="ctcf-stage5-controller-metrics-v1",
        label="controller",
        base_checkpoint_sha256=base_sha,
        initial_controller_state_sha256=initial_sha,
        source_contract_sha256=source_contract_sha256,
    )
    start_epoch, metrics = _adopt_existing_endpoint(
        output_root,
        resume,
        identity=identity,
        config=config,
        model=controller,
        optimizer=optimizer,
        scaler=scaler,
    )
    if start_epoch == config.fixed_epoch:
        return checkpoint_path

    step = _ControllerStep(
        store=store,
        base_runner=base_runner,
        controller=controller,
        optimizer=optimizer,
        scaler=scaler,
        device=device,
        variant=variant,
        bootstrap_policy=bootstrap_policy,
        config=config,
    )
    for epoch in range(start_epoch, config.fixed_epoch):
        epoch_started = time.perf_counter()
        controller.train()
        ordered = controller_epoch_pairs(training_subjects, seed=seed, epoch=epoch)
        totals: dict[str, float] = {}
        completed = 0
        for pair in ordered:
            logs = _controller_pair_step(step, pair, epoch)
            for key, value in logs.items():
                value = float(value)
                if not math.isfinite(value):
                    raise FloatingPointError(f"non-finite Stage5 controller metric: {key}")
                totals[key] = totals.get(key, 0.0) + value
            completed += 1
        if completed != len(ordered) or completed * 2 != len(training_subjects):
            raise RuntimeError("Stage5 controller epoch did not consume one perfect matching")
        row = {
            "epoch": epoch + 1,
            "pairs": completed,
            "pair_schedule_sha256": canonical_sha256(ordered),
            "metrics": {key: value / completed for key, value in sorted(totals.items())},
        }
        metrics.append(row)
        _commit_epoch(
            output_root,
            identity=identity,
            config=config,
            model=controller,
            optimizer=optimizer,
            scaler=scaler,
            epoch=epoch,
            row=row,
            epochs=metrics,
        )
        metric_text = " ".join(f"{key}={value:.6g}" for key, value in row["metrics"].items())
        print(
            f"[STAGE5 CONTROLLER EPOCH] seed={seed} variant={variant} "
            f"epoch={epoch + 1}/{config.fixed_epoch} pairs={completed} "
            f"elapsed_seconds={time.perf_counter() - epoch_started:.1f} {metric_text}",
            flush=True,
        )
    _verify_checkpoint_sidecar(checkpoint_path)
    return checkpoint_path


def _cuda_measurement_start(device: torch.device) -> float:
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    return time.perf_counter()


def _cuda_measurement_stop(device: torch.device, started: float) -> dict[str, float | int]:
    torch.cuda.synchronize(device)
    return {
        "elapsed_seconds": time.perf_counter() - started,
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
    }


@dataclass(frozen=True, slots=True)
class _SmokeContext:
    """What one smoke variant needs from the U0 phase that set it up."""

    seed: int
    device: torch.device
    store: Stage5OasisImageStore
    output_root: Path
    git_head: str
    protocol_sha256: str
    controller_training_contract_sha256: str
    controller_config: ControllerTrainingConfig
    u0_checkpoint_sha256: str
    common_state: dict[str, Any]
    common_state_sha256: str
    source_contract_sha256: str
    pair_schedule_sha256: str
    source_psi_path: Path


def _smoke_controller_step(
    context: _SmokeContext,
    variant: str,
    fixed: torch.Tensor,
    moving: torch.Tensor,
    psi_ab: torch.Tensor,
    psi_ba: torch.Tensor,
) -> dict[str, Any]:
    """Run one variant end to end: real optimizer step, transaction, checkpoint round-trip.

    Every tensor allocated here dies with the frame, which is what a single-GPU smoke run
    needs between the two variants.
    """
    seed = context.seed
    device = context.device
    store = context.store
    output_root = context.output_root
    git_head = context.git_head
    protocol_sha256 = context.protocol_sha256
    controller_config = context.controller_config
    controller_training_contract_sha256 = context.controller_training_contract_sha256
    u0_checkpoint_sha256 = context.u0_checkpoint_sha256
    common_state = context.common_state
    common_state_sha256 = context.common_state_sha256
    source_contract_sha256 = context.source_contract_sha256
    u0_pair_schedule_sha256 = context.pair_schedule_sha256

    _seed_everything(seed)
    torch.cuda.empty_cache()
    started = _cuda_measurement_start(device)
    features_ab = build_stage5_features(fixed, moving, psi_ab)
    features_ba = build_stage5_features(moving, fixed, psi_ba)
    controller = build_stage5_controller(controller_config).to(device)
    controller.load_state_dict(common_state, strict=True)
    parameters_before = state_dict_sha256(dict(controller.named_parameters()))
    optimizer = torch.optim.AdamW(
        controller.parameters(),
        lr=controller_config.learning_rate,
        weight_decay=controller_config.weight_decay,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=True)
    optimizer.zero_grad(set_to_none=True)
    input_ab, s2_ab, s4_ab, fixed_norm_ab, moving_norm_ab = _controller_training_tensors(features_ab)
    input_ba, s2_ba, s4_ba, fixed_norm_ba, moving_norm_ba = _controller_training_tensors(features_ba)
    with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True):
        output_ab = controller(input_ab, variant, s2_proposal=s2_ab, s4_proposal=s4_ab)
        output_ba = controller(input_ba, variant, s2_proposal=s2_ba, s4_proposal=s4_ba)
        loss, logs = controller_objective(
            fixed_norm_ab,
            moving_norm_ab,
            fixed_norm_ba,
            moving_norm_ba,
            psi_ab,
            psi_ba,
            output_ab.requested_delta,
            output_ba.requested_delta,
            config=controller_config.loss,
        )
    if not bool(torch.isfinite(loss)):
        raise FloatingPointError(f"non-finite Stage5 {variant} smoke loss")
    scaler.scale(loss).backward()
    _strict_scaler_step(scaler, optimizer, phase=f"controller smoke {variant}")
    parameters_after = state_dict_sha256(dict(controller.named_parameters()))
    if parameters_after == parameters_before:
        raise RuntimeError(f"Stage5 {variant} smoke optimizer step did not change any trainable parameter")
    measurement = _cuda_measurement_stop(device, started)
    measurement["loss"] = float(loss.detach().item())
    measurement["metrics"] = {key: float(value) for key, value in logs.items()}
    controller.eval()
    decision_started = _cuda_measurement_start(device)
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True):
        post_step = controller(input_ab, variant, s2_proposal=s2_ab, s4_proposal=s4_ab)
    requested_delta_rms = float(post_step.requested_delta.float().square().mean().sqrt().item())
    if not math.isfinite(requested_delta_rms) or (variant == "F24P" and requested_delta_rms <= 0.0):
        raise RuntimeError(f"Stage5 {variant} smoke produced an invalid post-step requested delta")
    transaction = commit_controller_delta(
        context.source_psi_path,
        post_step.requested_delta,
        output_root / f"decision_{variant}",
    )
    decision_measurement = _cuda_measurement_stop(device, decision_started)
    measurement["post_step_transaction"] = {
        **decision_measurement,
        "status": transaction.status,
        "requested_array_sha256": transaction.requested_array_sha256,
        "candidate_array_sha256": transaction.candidate_array_sha256,
        "returned_array_sha256": transaction.returned_array_sha256,
        "candidate_exact_status": transaction.candidate_exact_report["status"],
        "returned_exact_status": transaction.returned_exact_report["status"],
        "rollback_byte_identical": transaction.rollback_byte_identical,
        "requested_delta_rms": requested_delta_rms,
        "parameters_before_sha256": parameters_before,
        "parameters_after_sha256": parameters_after,
    }
    controller_metrics_payload = {
        "schema": "ctcf-stage5-controller-metrics-v1",
        "role": "CONTROLLER",
        "variant": variant,
        "seed": seed,
        "label_metrics_present": False,
        "selection_policy": CHECKPOINT_SELECTION_POLICY,
        "epochs": [
            {
                "epoch": 1,
                "pairs": 1,
                "pair_schedule_sha256": u0_pair_schedule_sha256,
                "metrics": measurement["metrics"],
            }
        ],
    }
    controller_checkpoint_root = output_root / f"checkpoint_{variant}"
    controller_metrics_path = controller_checkpoint_root / "metrics.json"
    atomic_write_bytes(controller_metrics_path, readable_json_bytes(controller_metrics_payload))
    controller_checkpoint = controller_checkpoint_root / "last.pth"
    controller_state = build_training_state(
        role="CONTROLLER",
        variant_id=variant,
        seed=seed,
        epoch_completed=1,
        fixed_epoch=controller_config.fixed_epoch,
        git_head=git_head,
        protocol_sha256=protocol_sha256,
        data_contract_sha256=store.runtime.contract_sha256,
        training_contract_sha256=controller_training_contract_sha256,
        model=controller,
        optimizer=optimizer,
        scaler=scaler,
        pair_schedule_sha256=u0_pair_schedule_sha256,
        metrics_sha256=sha256_file(controller_metrics_path),
        base_checkpoint_sha256=u0_checkpoint_sha256,
        initial_controller_state_sha256=common_state_sha256,
        source_contract_sha256=source_contract_sha256,
    )
    _attach_runtime_checkpoint_metadata(
        controller_state,
        config=controller_config,
        metrics_payload=controller_metrics_payload,
    )
    controller_model_state_sha256 = str(controller_state["model_state_sha256"])
    controller_checkpoint_sha256 = _write_checkpoint_with_sidecar(controller_checkpoint, controller_state)
    reloaded_controller = build_stage5_controller(controller_config).to(device)
    reloaded_optimizer = torch.optim.AdamW(
        reloaded_controller.parameters(),
        lr=controller_config.learning_rate,
        weight_decay=controller_config.weight_decay,
    )
    reloaded_scaler = torch.amp.GradScaler("cuda", enabled=True)
    reloaded_controller_state = load_training_state(
        controller_checkpoint,
        model=reloaded_controller,
        optimizer=reloaded_optimizer,
        scaler=reloaded_scaler,
        expected_role="CONTROLLER",
        expected_variant=variant,
        expected_seed=seed,
        expected_protocol_sha256=protocol_sha256,
        expected_data_contract_sha256=store.runtime.contract_sha256,
        expected_training_contract_sha256=controller_training_contract_sha256,
        restore_rng=True,
    )
    _validate_runtime_checkpoint_metadata(
        reloaded_controller_state,
        role="CONTROLLER",
        variant=variant,
        seed=seed,
        config=controller_config,
        expected_git_head=git_head,
        expected_base_checkpoint_sha256=u0_checkpoint_sha256,
        expected_initial_controller_state_sha256=common_state_sha256,
        expected_source_contract_sha256=source_contract_sha256,
    )
    if state_dict_sha256(reloaded_controller.state_dict()) != controller_model_state_sha256:
        raise RuntimeError(f"Stage5 {variant} smoke checkpoint did not reload the exact controller state")
    measurement["checkpoint_sha256"] = controller_checkpoint_sha256
    measurement["reloaded_model_state_sha256"] = controller_model_state_sha256
    return measurement


def smoke_stage5_runtime(
    *,
    data_contract: Path,
    image_root: Path,
    output_root: Path,
    seed: int,
    device: torch.device,
    git_head: str,
    protocol_sha256: str,
    u0_training_contract_sha256: str,
    controller_training_contract_sha256: str,
    bootstrap_policy: str,
    u0_config: U0TrainingConfig,
    controller_config: ControllerTrainingConfig,
) -> Path:
    """Exercise real H100-only paths without producing an accepted checkpoint."""

    _require_cuda(device, "runtime smoke test")
    require_seed(seed)
    require_git_sha(git_head, "git_head", error=ValueError)
    require_sha256(protocol_sha256, "protocol_sha256", error=ValueError)
    require_sha256(u0_training_contract_sha256, "u0_training_contract_sha256", error=ValueError)
    require_sha256(controller_training_contract_sha256, "controller_training_contract_sha256", error=ValueError)
    if bootstrap_policy not in BOOTSTRAP_POLICIES:
        raise ValueError(f"unsupported Stage5 smoke bootstrap policy: {bootstrap_policy}")
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"Stage5 smoke output must be new or empty: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    _seed_everything(seed)
    store = Stage5OasisImageStore(data_contract, image_root)
    subjects = _training_subjects(store)
    moving_id, fixed_id = epoch_pair_schedule(subjects, seed=seed, epoch=0)[0]
    moving = _tensor_image(store, moving_id, device)
    fixed = _tensor_image(store, fixed_id, device)

    try:
        runner = Runner(_u0_args(u0_config, seed), device)
    except Exception as exc:
        raise RuntimeError("Stage5 H100 smoke test could not construct the frozen Mamba U0") from exc
    u0_scaler = torch.amp.GradScaler("cuda", enabled=True)
    u0_parameters_before = state_dict_sha256(dict(runner.model.named_parameters()))
    runner.model.train()
    runner.optimizer.zero_grad(set_to_none=True)
    started = _cuda_measurement_start(device)
    with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True):
        u0_loss, raw_logs = runner.train_step((moving, fixed), 0)
    if not bool(torch.isfinite(u0_loss)):
        raise FloatingPointError("non-finite Stage5 U0 smoke loss")
    u0_scaler.scale(u0_loss).backward()
    _strict_scaler_step(u0_scaler, runner.optimizer, phase="U0 smoke")
    u0_parameters_after = state_dict_sha256(dict(runner.model.named_parameters()))
    if u0_parameters_after == u0_parameters_before:
        raise RuntimeError("Stage5 U0 smoke optimizer step did not change any trainable parameter")
    u0_measurement = _cuda_measurement_stop(device, started)
    u0_measurement["loss"] = float(u0_loss.detach().item())
    u0_measurement["metrics"] = _sanitize_u0_logs(raw_logs)
    u0_measurement["parameters_before_sha256"] = u0_parameters_before
    u0_measurement["parameters_after_sha256"] = u0_parameters_after

    runner.model.eval()
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True):
        _, raw_ab = runner.model(moving, fixed, alpha_l1=1.0, alpha_l3=1.0)
        _, raw_ba = runner.model(fixed, moving, alpha_l1=1.0, alpha_l3=1.0)
    source_started = _cuda_measurement_start(device)
    source_ab = prepare_initial_field(raw_ab.float(), output_root / "source_ab", policy=bootstrap_policy)
    source_ba = prepare_initial_field(raw_ba.float(), output_root / "source_ba", policy=bootstrap_policy)
    source_measurement = _cuda_measurement_stop(device, source_started)

    u0_pair_schedule_sha256 = canonical_sha256(((moving_id, fixed_id),))
    u0_metrics_payload = {
        "schema": "ctcf-stage5-u0-metrics-v1",
        "role": "U0",
        "variant": "U0",
        "seed": seed,
        "label_metrics_present": False,
        "selection_policy": CHECKPOINT_SELECTION_POLICY,
        "epochs": [
            {
                "epoch": 1,
                "learning_rate": u0_config.learning_rate,
                "pairs": 1,
                "pair_schedule_sha256": u0_pair_schedule_sha256,
                "metrics": u0_measurement["metrics"],
            }
        ],
    }
    u0_checkpoint_root = output_root / "checkpoint_u0"
    u0_metrics_path = u0_checkpoint_root / "metrics.json"
    atomic_write_bytes(u0_metrics_path, readable_json_bytes(u0_metrics_payload))
    u0_checkpoint = u0_checkpoint_root / "last.pth"
    u0_state = build_training_state(
        role="U0",
        variant_id="U0",
        seed=seed,
        epoch_completed=1,
        fixed_epoch=u0_config.fixed_epoch,
        git_head=git_head,
        protocol_sha256=protocol_sha256,
        data_contract_sha256=store.runtime.contract_sha256,
        training_contract_sha256=u0_training_contract_sha256,
        model=runner.model,
        optimizer=runner.optimizer,
        scaler=u0_scaler,
        pair_schedule_sha256=u0_pair_schedule_sha256,
        metrics_sha256=sha256_file(u0_metrics_path),
    )
    _attach_runtime_checkpoint_metadata(u0_state, config=u0_config, metrics_payload=u0_metrics_payload)
    u0_model_state_sha256 = str(u0_state["model_state_sha256"])
    u0_checkpoint_sha256 = _write_checkpoint_with_sidecar(u0_checkpoint, u0_state)

    del raw_ab, raw_ba, raw_logs, u0_loss, u0_scaler, u0_state, runner
    torch.cuda.empty_cache()
    reloaded_runner = Runner(_u0_args(u0_config, seed), device)
    reloaded_scaler = torch.amp.GradScaler("cuda", enabled=True)
    reloaded_state = load_training_state(
        u0_checkpoint,
        model=reloaded_runner.model,
        optimizer=reloaded_runner.optimizer,
        scaler=reloaded_scaler,
        expected_role="U0",
        expected_variant="U0",
        expected_seed=seed,
        expected_protocol_sha256=protocol_sha256,
        expected_data_contract_sha256=store.runtime.contract_sha256,
        expected_training_contract_sha256=u0_training_contract_sha256,
        restore_rng=True,
    )
    _validate_runtime_checkpoint_metadata(
        reloaded_state,
        role="U0",
        variant="U0",
        seed=seed,
        config=u0_config,
        expected_git_head=git_head,
        expected_base_checkpoint_sha256=None,
        expected_initial_controller_state_sha256=None,
        expected_source_contract_sha256=None,
    )
    if state_dict_sha256(reloaded_runner.model.state_dict()) != u0_model_state_sha256:
        raise RuntimeError("Stage5 U0 smoke checkpoint did not reload the exact model state")
    u0_measurement["checkpoint_sha256"] = u0_checkpoint_sha256
    u0_measurement["reloaded_model_state_sha256"] = u0_model_state_sha256
    del reloaded_runner, reloaded_scaler, reloaded_state
    torch.cuda.empty_cache()
    psi_ab = load_flow_npz(source_ab.psi_path).to(device=device, dtype=torch.float32)
    psi_ba = load_flow_npz(source_ba.psi_path).to(device=device, dtype=torch.float32)
    _seed_everything(seed)
    template = build_stage5_controller(controller_config)
    common_state = copy.deepcopy(template.state_dict())
    common_state_sha256 = state_dict_sha256(common_state)
    del template

    source_contract_sha256 = canonical_sha256(
        {
            "source_ab_phi": source_ab.phi_sha256,
            "source_ab_psi": source_ab.psi_sha256,
            "source_ba_phi": source_ba.phi_sha256,
            "source_ba_psi": source_ba.psi_sha256,
        }
    )
    context = _SmokeContext(
        seed=seed,
        device=device,
        store=store,
        output_root=output_root,
        git_head=git_head,
        protocol_sha256=protocol_sha256,
        controller_training_contract_sha256=controller_training_contract_sha256,
        controller_config=controller_config,
        u0_checkpoint_sha256=u0_checkpoint_sha256,
        common_state=common_state,
        common_state_sha256=common_state_sha256,
        source_contract_sha256=source_contract_sha256,
        pair_schedule_sha256=u0_pair_schedule_sha256,
        source_psi_path=source_ab.psi_path,
    )
    controller_measurements = {
        variant: _smoke_controller_step(context, variant, fixed, moving, psi_ab, psi_ba) for variant in ("F24P", "A24P")
    }

    report_path = output_root / "smoke_report.json"
    atomic_write_json(
        report_path,
        {
            "schema": "ctcf-stage5-runtime-smoke-v1",
            "status": "PASS",
            "production_artifact": False,
            "accepted_production_checkpoint_written": False,
            "smoke_checkpoint_roundtrip": True,
            "git_head": git_head,
            "protocol_sha256": protocol_sha256,
            "data_contract_sha256": store.runtime.contract_sha256,
            "u0_training_contract_sha256": u0_training_contract_sha256,
            "controller_training_contract_sha256": controller_training_contract_sha256,
            "seed": seed,
            "pair": {"moving_subject_id": moving_id, "fixed_subject_id": fixed_id},
            "bootstrap_policy": bootstrap_policy,
            "u0_config_sha256": config_sha256(u0_config),
            "controller_config_sha256": config_sha256(controller_config),
            "common_controller_initial_state_sha256": common_state_sha256,
            "execution_determinism": _execution_determinism_contract(),
            "u0_step": u0_measurement,
            "source_materialization": {
                **source_measurement,
                "ab_phi_sha256": source_ab.phi_sha256,
                "ab_psi_sha256": source_ab.psi_sha256,
                "ba_phi_sha256": source_ba.phi_sha256,
                "ba_psi_sha256": source_ba.psi_sha256,
            },
            "controller_steps": controller_measurements,
        },
    )
    return report_path


__all__ = [
    "ControllerTrainingConfig",
    "U0TrainingConfig",
    "controller_epoch_pairs",
    "development_case_inventory",
    "epoch_pair_schedule",
    "initialize_controller_state",
    "load_frozen_u0",
    "materialize_source_fields",
    "smoke_stage5_runtime",
    "train_controller",
    "train_u0",
    "validate_certified_source_artifact",
]
