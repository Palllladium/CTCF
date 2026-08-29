from __future__ import annotations

import platform
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

from datasets.OASIS100 import Stage5OasisImageStore
from experiments.stage5.checkpoints import load_training_state
from experiments.stage5.config import ControllerTrainingConfig, build_stage5_controller
from experiments.stage5.features import build_stage5_features
from experiments.stage5.runtime import validate_certified_source_artifact
from experiments.stage5.safety import commit_controller_delta
from models.CTCF.controller import STAGE5_VARIANTS, Stage5SpatialController
from tools.analysis.run_artifacts import sha256_file
from tools.analysis.search.pyramid import array_sha256
from tools.analysis.search.transaction import load_flow_npz
from tools.analysis.stage5.artifacts import (
    checkpoint_metadata,
    execution_sha256,
    field_record,
    file_record,
    load_canonical_json,
    save_reload_attestation,
)
from tools.analysis.stage5.contracts import (
    BASE_SEEDS,
    CONTROLLER_VARIANT_IDS,
    DECISION_RECORD_SCHEMA,
    build_decision_barrier,
    build_training_barrier,
    canonical_sha256,
    validate_decision_barrier,
    validate_protocol_contract,
    validate_training_barrier,
    write_immutable_json,
)
from utils.cert_exact import certify_flow_exact


def checkpoint_path(checkpoint_root: Path, *, seed: int, variant: str) -> Path:
    if variant == "U0":
        return checkpoint_root / "u0" / f"seed_{seed}" / "last.pth"
    return checkpoint_root / "controllers" / f"seed_{seed}" / variant / "last.pth"


def collect_checkpoint_metadata(
    *,
    protocol_path: Path,
    checkpoint_root: Path,
) -> list[dict[str, Any]]:
    protocol = load_canonical_json(protocol_path)
    validate_protocol_contract(protocol)
    records: list[dict[str, Any]] = []
    for seed in BASE_SEEDS:
        for variant in ("U0", *CONTROLLER_VARIANT_IDS):
            path = checkpoint_path(checkpoint_root, seed=seed, variant=variant)
            records.append(
                checkpoint_metadata(
                    checkpoint_id=f"S5_S{seed}_{variant}",
                    checkpoint_path=path,
                    checkpoint_root=checkpoint_root,
                    metrics_path=path.with_name("metrics.json"),
                    protocol=protocol,
                )
            )
    return records


def freeze_training_barrier(
    *,
    protocol_path: Path,
    checkpoint_root: Path,
    output_path: Path,
) -> dict[str, Any]:
    protocol = load_canonical_json(protocol_path)
    records = collect_checkpoint_metadata(protocol_path=protocol_path, checkpoint_root=checkpoint_root)
    barrier = build_training_barrier(protocol, records)
    validate_training_barrier(barrier, protocol, require_complete=True)
    write_immutable_json(output_path, barrier)
    return barrier


def _checkpoint_index(training_barrier: Mapping[str, Any]) -> dict[tuple[int, str], Mapping[str, Any]]:
    return {(int(item["seed"]), str(item["variant_id"])): item for item in training_barrier["checkpoints"]}


def _source_artifact(
    source_root: Path,
    *,
    seed: int,
    case: Mapping[str, Any],
    expected_u0_sha256: str,
    bootstrap_policy: str,
    image_shape: tuple[int, int, int],
) -> tuple[Path, dict[str, Any], torch.Tensor]:
    source_root = source_root.resolve(strict=True)
    case_root = (source_root / f"seed_{seed}" / str(case["case_id"])).resolve(strict=True)
    try:
        case_root.relative_to(source_root)
    except ValueError as exc:
        raise RuntimeError("Stage5 source artifact escaped its declared root") from exc
    psi_path = case_root / "initial_psi.npz"
    verification = validate_certified_source_artifact(
        case_root,
        seed=seed,
        case=case,
        u0_checkpoint_sha256=expected_u0_sha256,
        image_shape=image_shape,
        bootstrap_policy=bootstrap_policy,
    )
    psi = load_flow_npz(psi_path)
    if verification["psi_array_sha256"] != array_sha256(psi):
        raise RuntimeError("Stage5 source array differs from its exact certificate")
    if psi.dtype != torch.float32 or psi.shape != (1, 3, *image_shape) or not bool(torch.isfinite(psi).all()):
        raise RuntimeError("Stage5 source array has an invalid shape, dtype, or value")
    return psi_path, verification, psi


def _load_controller(
    checkpoint: Path,
    *,
    metadata: Mapping[str, Any],
    protocol: Mapping[str, Any],
    device: torch.device,
    config: ControllerTrainingConfig,
) -> Stage5SpatialController:
    expected_checkpoint_sha = str(metadata["checkpoint_file"]["sha256"])
    if (
        not checkpoint.is_file()
        or checkpoint.is_symlink()
        or checkpoint.stat().st_size != int(metadata["checkpoint_file"]["bytes"])
        or sha256_file(checkpoint) != expected_checkpoint_sha
    ):
        raise RuntimeError("Stage5 controller checkpoint bytes differ from the frozen training barrier")
    controller = build_stage5_controller(config).to(device)
    state = load_training_state(
        checkpoint,
        model=controller,
        optimizer=None,
        scaler=None,
        expected_role="CONTROLLER",
        expected_variant=str(metadata["variant_id"]),
        expected_seed=int(metadata["seed"]),
        expected_protocol_sha256=canonical_sha256(protocol),
        expected_data_contract_sha256=str(protocol["data_contract_sha256"]),
        expected_training_contract_sha256=str(protocol["controller_training_contract_sha256"]),
        restore_rng=False,
    )
    if int(state["epoch_completed"]) != int(protocol["controller_fixed_epoch"]):
        raise RuntimeError("Stage5 controller is not the frozen endpoint")
    controller.eval().requires_grad_(False)
    return controller


def _case_images(
    store: Stage5OasisImageStore,
    case: Mapping[str, Any],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    moving = torch.from_numpy(store.load_image(str(case["moving_subject_id"]))).unsqueeze(0).unsqueeze(0)
    fixed = torch.from_numpy(store.load_image(str(case["fixed_subject_id"]))).unsqueeze(0).unsqueeze(0)
    return moving.to(device=device, dtype=torch.float32), fixed.to(device=device, dtype=torch.float32)


def _delta_rms(field: torch.Tensor, source: torch.Tensor) -> float:
    difference = field.detach().cpu().float() - source.detach().cpu().float()
    return float(torch.sqrt(difference.square().mean()).item())


def materialize_decisions(
    *,
    protocol_path: Path,
    training_barrier_path: Path,
    data_contract_path: Path,
    image_root: Path,
    checkpoint_root: Path,
    source_root: Path,
    decision_root: Path,
    seed: int,
    variant: str,
    shard_index: int,
    num_shards: int,
    device: torch.device,
    controller_config: ControllerTrainingConfig | None = None,
) -> int:
    if seed not in BASE_SEEDS or variant not in ("U0", *STAGE5_VARIANTS):
        raise ValueError("decision seed or variant is outside the frozen Stage5 matrix")
    if not 0 <= shard_index < num_shards:
        raise ValueError("invalid Stage5 decision shard")
    protocol = load_canonical_json(protocol_path)
    training = load_canonical_json(training_barrier_path)
    validate_training_barrier(training, protocol, require_complete=True)
    store = Stage5OasisImageStore(data_contract_path, image_root)
    if store.runtime.contract_sha256 != protocol["data_contract_sha256"]:
        raise RuntimeError("Stage5 data contract differs from the protocol")
    case_inventory = tuple(store.runtime.pairs["cases"])
    if [case["case_id"] for case in case_inventory] != protocol["directed_case_ids"]:
        raise RuntimeError("Stage5 decision cases differ from the protocol")
    checkpoints = _checkpoint_index(training)
    metadata = checkpoints[(seed, variant)]
    u0_metadata = checkpoints[(seed, "U0")]
    u0_sha = str(u0_metadata["checkpoint_file"]["sha256"])
    bootstrap_policy = str(protocol["bootstrap"]["policy"])
    controller = None
    degraded_identity = bootstrap_policy == "identity"
    if variant != "U0" and not degraded_identity:
        controller = _load_controller(
            checkpoint_path(checkpoint_root, seed=seed, variant=variant),
            metadata=metadata,
            protocol=protocol,
            device=device,
            config=controller_config or ControllerTrainingConfig(),
        )

    completed = 0
    for index, case in enumerate(case_inventory):
        if index % num_shards != shard_index:
            continue
        decision_id = f"{case['case_id']}__S{seed}__{variant}"
        record_path = decision_root / "records" / f"{decision_id}.json"
        if record_path.is_file():
            existing = load_canonical_json(record_path)
            build_decision_barrier(protocol, training, [existing])
            if existing.get("decision_id") != decision_id:
                raise RuntimeError("existing Stage5 decision has another identity")
            completed += 1
            continue
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
        started = time.perf_counter()
        source_path, _, source = _source_artifact(
            source_root,
            seed=seed,
            case=case,
            expected_u0_sha256=u0_sha,
            bootstrap_policy=bootstrap_policy,
            image_shape=store.image_shape,
        )
        source_record = field_record("source_field_root", source_root, source_path)
        source_array_sha = array_sha256(source)

        decision_case_root = decision_root / "fields" / f"seed_{seed}" / variant / str(case["case_id"])
        # A baseline row and a degraded-identity row both return the source bytes untouched,
        # so every field path below is the source path.
        source_only = variant == "U0" or degraded_identity
        requested_path = source_path if source_only else decision_case_root / "requested.npz"
        candidate_path = source_path if source_only else decision_case_root / "post_safety_candidate.npz"
        if source_only:
            requested_record = source_record
            candidate_record = source_record
            returned_record = source_record
            requested_array_sha = source_array_sha
            candidate_array_sha = source_array_sha
            returned_array_sha = source_array_sha
            candidate_exact = certify_flow_exact(source, eps="0.001")
            returned_exact = candidate_exact
            transaction_status = "CERTIFIED_DEGRADED_IDENTITY" if degraded_identity else "BASELINE_CERTIFIED"
            rollback_equal = False
            clip_report: Mapping[str, Any] | None = None
        else:
            assert controller is not None
            moving, fixed = _case_images(store, case, device)
            source_device = source.to(device=device, dtype=torch.float32)
            # The frozen S2/S4 feature contract is FP32.  Controller convolutions may
            # use AMP, but constructing the search posterior under autocast would make
            # deployment consume different inputs from controller training.
            with torch.inference_mode():
                features = build_stage5_features(fixed, moving, source_device)
            with (
                torch.inference_mode(),
                torch.autocast(
                    device_type="cuda",
                    dtype=torch.float16,
                    enabled=device.type == "cuda",
                ),
            ):
                output = controller(
                    features.controller_input,
                    variant,
                    s2_proposal=features.s2.proposal,
                    s4_proposal=features.s4.proposal,
                )
            transaction = commit_controller_delta(
                source_path,
                output.requested_delta.float(),
                decision_case_root,
            )
            requested_record = field_record("decision_output_root", decision_root, transaction.requested_path)
            candidate_record = field_record("decision_output_root", decision_root, transaction.candidate_path)
            returned_record = candidate_record if transaction.status == "ACCEPTED" else source_record
            requested_array_sha = transaction.requested_array_sha256
            candidate_array_sha = transaction.candidate_array_sha256
            returned_array_sha = transaction.returned_array_sha256
            candidate_exact = transaction.candidate_exact_report
            returned_exact = transaction.returned_exact_report
            transaction_status = transaction.status
            rollback_equal = transaction.rollback_byte_identical
            clip_report = transaction.clip_report

        requested_loaded = load_flow_npz(requested_path)
        candidate_loaded = load_flow_npz(candidate_path)
        returned_loaded = candidate_loaded if transaction_status == "ACCEPTED" else source
        requested_delta_rms = _delta_rms(requested_loaded, source)
        candidate_delta_rms = _delta_rms(candidate_loaded, source)
        returned_delta_rms = _delta_rms(returned_loaded, source)
        if requested_delta_rms == 0.0:
            candidate_retained_ratio = None
            returned_retained_ratio = None
        else:
            candidate_retained_ratio = candidate_delta_rms / requested_delta_rms
            returned_retained_ratio = returned_delta_rms / requested_delta_rms
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            peak_memory_bytes = int(torch.cuda.max_memory_allocated(device))
        else:
            peak_memory_bytes = 0
        runtime_seconds = time.perf_counter() - started

        exact_payload = {
            "schema": "ctcf-stage5-decision-exact-report-v1",
            "decision_id": decision_id,
            "source_field": source_record,
            "candidate_exact": candidate_exact,
            "returned_exact": returned_exact,
            "clip_report": clip_report,
            "execution": {
                "protocol_sha256": canonical_sha256(protocol),
                "training_barrier_sha256": canonical_sha256(training),
                "checkpoint_sha256": metadata["checkpoint_file"]["sha256"],
                "device_type": device.type,
                "torch_version": torch.__version__,
                "python_version": platform.python_version(),
                "labels_loaded": False,
            },
        }
        exact_path = decision_root / "exact_reports" / f"{decision_id}.json"
        write_immutable_json(exact_path, exact_payload)
        returned_path = candidate_path if transaction_status == "ACCEPTED" else source_path
        record = {
            "schema": DECISION_RECORD_SCHEMA,
            "decision_id": decision_id,
            "case_id": case["case_id"],
            "seed": seed,
            "variant_id": variant,
            "checkpoint_sha256": metadata["checkpoint_file"]["sha256"],
            "certified_source_field": source_record,
            "requested_field": requested_record,
            "candidate_field": candidate_record,
            "returned_field": returned_record,
            "requested_save_reload": save_reload_attestation(
                requested_record,
                in_memory_array_sha256=requested_array_sha,
                reloaded_path=requested_path,
            ),
            "candidate_save_reload": save_reload_attestation(
                candidate_record,
                in_memory_array_sha256=candidate_array_sha,
                reloaded_path=candidate_path,
            ),
            "returned_save_reload": save_reload_attestation(
                returned_record,
                in_memory_array_sha256=returned_array_sha,
                reloaded_path=returned_path,
            ),
            "exact_report": file_record("decision_output_root", decision_root, exact_path),
            "candidate_exact_status": candidate_exact["status"],
            "candidate_exact_certified": candidate_exact["certified"],
            "returned_exact_status": returned_exact["status"],
            "returned_certified": returned_exact["certified"],
            "transaction_status": transaction_status,
            "rollback_source_sha256_equal": rollback_equal,
            "runtime_seconds": runtime_seconds,
            "peak_memory_bytes": peak_memory_bytes,
            "requested_delta_rms": requested_delta_rms,
            "candidate_delta_rms": candidate_delta_rms,
            "returned_delta_rms": returned_delta_rms,
            "candidate_retained_ratio": candidate_retained_ratio,
            "returned_retained_ratio": returned_retained_ratio,
            "labels_loaded": False,
            "execution_sha256": execution_sha256(
                {
                    "environment": exact_payload["execution"],
                    "performance": {
                        "runtime_seconds": runtime_seconds,
                        "peak_memory_bytes": peak_memory_bytes,
                        "requested_delta_rms": requested_delta_rms,
                        "candidate_delta_rms": candidate_delta_rms,
                        "returned_delta_rms": returned_delta_rms,
                        "candidate_retained_ratio": candidate_retained_ratio,
                        "returned_retained_ratio": returned_retained_ratio,
                    },
                }
            ),
        }
        write_immutable_json(record_path, record)
        completed += 1
    return completed


def freeze_decision_barrier(
    *,
    protocol_path: Path,
    training_barrier_path: Path,
    decision_root: Path,
    output_path: Path,
) -> dict[str, Any]:
    protocol = load_canonical_json(protocol_path)
    training = load_canonical_json(training_barrier_path)
    records = [load_canonical_json(path) for path in sorted((decision_root / "records").glob("*.json"))]
    barrier = build_decision_barrier(protocol, training, records)
    validate_decision_barrier(barrier, protocol, training, require_complete=True)
    write_immutable_json(output_path, barrier)
    return barrier


__all__ = [
    "checkpoint_path",
    "collect_checkpoint_metadata",
    "freeze_decision_barrier",
    "freeze_training_barrier",
    "materialize_decisions",
]
