from __future__ import annotations

import argparse
import json
import math
import re
import shutil
import subprocess
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any

import torch

from datasets.OASIS100 import (
    STAGE5_LABEL_EVALUATION_AUTHORIZATION,
    STAGE5_PREPARE_AUTHORIZATION,
    Stage5OasisEvaluationLabelStore,
    load_stage5_runtime_contract,
    prepare_stage5_oasis_data,
)
from experiments.stage5.checkpoints import state_dict_sha256
from experiments.stage5.config import ControllerTrainingConfig, U0TrainingConfig, build_stage5_controller
from experiments.stage5.runtime import (
    initialize_controller_state,
    materialize_source_fields,
    smoke_stage5_runtime,
    train_controller,
    train_u0,
)
from models.CTCF.controller import STAGE5_VARIANTS
from tools.analysis.run_artifacts import atomic_write_text, sha256_file
from tools.analysis.search.pyramid import array_sha256
from tools.analysis.search.transaction import load_flow_npz
from tools.analysis.stage5.artifacts import file_record, load_canonical_json
from tools.analysis.stage5.contracts import (
    BASE_SEEDS,
    VARIANT_IDS,
    build_evaluation_barrier,
    canonical_json_bytes,
    canonical_sha256,
    validate_decision_barrier,
    validate_evaluation_barrier,
    validate_protocol_contract,
    validate_training_barrier,
    write_immutable_json,
)
from tools.analysis.stage5.evaluation import (
    EVALUATION_SCHEMA,
    EvaluationContext,
    aggregate_pair_effects,
    build_evaluation_record,
    build_pair_evaluation,
    decision_csv,
    diagnostic_csv,
    effect_csv,
    evaluate_returned_decision,
    field_stage_diagnostics_csv,
    geometry_csv,
    pair_metric_csv,
    per_label_csv,
    write_decision_metrics,
    write_evaluation_products,
)
from tools.analysis.stage5.pipeline import (
    checkpoint_path,
    freeze_decision_barrier,
    freeze_training_barrier,
    materialize_decisions,
)
from tools.analysis.stage5.primitives import (
    FileGeneration,
    file_generation,
    generation_cache_is_safe,
    is_link_like,
    require_git_sha,
    require_sha256,
)
from tools.analysis.stage5.protocol import prepare_protocol_bundle

RUN_ID_RE = re.compile(r"S5_[A-Z0-9]+_[0-9]{8}T[0-9]{6}Z_[0-9a-f]{12}")

DISK_TARGET_GIB = {
    "data": 16,
    "source": 230,
    "decision": 470,
    "full": 730,
}
DISK_RESERVE_GIB = 10
GIB = 1024**3
SMOKE_BARRIER_SCHEMA = "ctcf-stage5-runtime-smoke-barrier-v1"

# The compact root carries only text attestations. This is a suffix denylist, not a proof:
# it catches every format Stage 5 actually writes, and the manifest's contains_* flags are
# claims about those formats, not about arbitrary bytes.
HEAVY_SUFFIXES = frozenset(
    {".pth", ".pt", ".pt2", ".ckpt", ".bin", ".safetensors", ".h5", ".npz", ".npy", ".pkl", ".nii", ".gz", ".mgz"}
)


def _git(repo_root: Path, *arguments: str) -> str:
    result = subprocess.run(
        ("git", "-C", str(repo_root), *arguments),
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def assert_clean_exact_git(repo_root: Path, expected_git_head: str) -> str:
    repo_root = repo_root.resolve(strict=True)
    if not repo_root.is_dir() or not (repo_root / ".git").exists():
        raise RuntimeError(f"Stage5 repository root is invalid: {repo_root}")
    require_git_sha(expected_git_head, "--expected-git-head", error=ValueError)
    actual_head = _git(repo_root, "rev-parse", "HEAD")
    if actual_head != expected_git_head:
        raise RuntimeError(f"Stage5 Git HEAD mismatch: expected={expected_git_head} actual={actual_head}")
    status = _git(repo_root, "status", "--porcelain=v1", "--untracked-files=all")
    if status:
        raise RuntimeError(f"Stage5 refuses a dirty Git tree:\n{status}")
    return actual_head


def _read_protocol(path: Path, git_head: str) -> dict[str, Any]:
    protocol = load_canonical_json(path)
    validate_protocol_contract(protocol)
    if protocol["git_head"] != git_head:
        raise RuntimeError("Stage5 protocol was frozen on another Git HEAD")
    return protocol


def _protocol_context(args: argparse.Namespace) -> tuple[str, dict[str, Any]]:
    head = assert_clean_exact_git(args.repo_root, args.expected_git_head)
    protocol = _read_protocol(args.protocol, head)
    if hasattr(args, "smoke_barrier"):
        _validate_smoke_gate(args.smoke_barrier, args.smoke_report, protocol, head)
    return head, protocol


def _validate_smoke_report(report: Mapping[str, Any], protocol: Mapping[str, Any], git_head: str) -> None:
    if (
        report.get("schema") != "ctcf-stage5-runtime-smoke-v1"
        or report.get("status") != "PASS"
        or report.get("production_artifact") is not False
        or report.get("accepted_production_checkpoint_written") is not False
        or report.get("smoke_checkpoint_roundtrip") is not True
        or report.get("git_head") != git_head
        or report.get("protocol_sha256") != canonical_sha256(protocol)
        or report.get("data_contract_sha256") != protocol["data_contract_sha256"]
        or report.get("u0_training_contract_sha256") != protocol["u0_training_contract_sha256"]
        or report.get("controller_training_contract_sha256") != protocol["controller_training_contract_sha256"]
        or report.get("seed") != 0
        or report.get("bootstrap_policy") != protocol["bootstrap"]["policy"]
    ):
        raise RuntimeError("Stage5 H100 smoke report does not match the frozen protocol")
    u0 = report.get("u0_step")
    if not isinstance(u0, Mapping):
        raise RuntimeError("Stage5 H100 smoke report has no U0 optimizer step")
    for key in (
        "parameters_before_sha256",
        "parameters_after_sha256",
        "checkpoint_sha256",
        "reloaded_model_state_sha256",
    ):
        require_sha256(u0.get(key), f"Stage5 H100 smoke U0 digest {key}")
    if u0["parameters_before_sha256"] == u0["parameters_after_sha256"]:
        raise RuntimeError("Stage5 H100 smoke U0 optimizer step changed no trainable parameter")
    controllers = report.get("controller_steps")
    if not isinstance(controllers, Mapping) or set(controllers) != {"F24P", "A24P"}:
        raise RuntimeError("Stage5 H100 smoke controller inventory changed")
    for variant, raw in controllers.items():
        if not isinstance(raw, Mapping):
            raise RuntimeError(f"Stage5 H100 smoke controller report is invalid: {variant}")
        for key in (
            "checkpoint_sha256",
            "reloaded_model_state_sha256",
        ):
            require_sha256(raw.get(key), f"Stage5 H100 smoke controller digest {variant}/{key}")
        transaction = raw.get("post_step_transaction")
        if not isinstance(transaction, Mapping):
            raise RuntimeError(f"Stage5 H100 smoke has no post-step transaction: {variant}")
        for key in (
            "parameters_before_sha256",
            "parameters_after_sha256",
            "requested_array_sha256",
            "candidate_array_sha256",
            "returned_array_sha256",
        ):
            require_sha256(transaction.get(key), f"Stage5 H100 smoke transaction digest {variant}/{key}")
        if transaction["parameters_before_sha256"] == transaction["parameters_after_sha256"]:
            raise RuntimeError(f"Stage5 H100 smoke optimizer step changed no parameter: {variant}")
        delta_rms = transaction.get("requested_delta_rms")
        if (
            isinstance(delta_rms, bool)
            or not isinstance(delta_rms, (int, float))
            or not math.isfinite(float(delta_rms))
            or (variant == "F24P" and float(delta_rms) <= 0.0)
            or transaction.get("returned_exact_status") != "CERTIFIED"
            or transaction.get("status") not in {"ACCEPTED", "ROLLED_BACK"}
        ):
            raise RuntimeError(f"Stage5 H100 smoke post-step transaction is invalid: {variant}")


def _validate_smoke_gate(
    barrier_path: Path,
    report_path: Path,
    protocol: Mapping[str, Any],
    git_head: str,
) -> dict[str, Any]:
    barrier = load_canonical_json(barrier_path)
    report = load_canonical_json(report_path)
    _validate_smoke_report(report, protocol, git_head)
    expected = {
        "schema": SMOKE_BARRIER_SCHEMA,
        "status": "COMPLETE",
        "git_head": git_head,
        "protocol_sha256": canonical_sha256(protocol),
        "data_contract_sha256": protocol["data_contract_sha256"],
        "smoke_report_bytes": report_path.stat().st_size,
        "smoke_report_sha256": sha256_file(report_path),
    }
    if barrier != expected:
        raise RuntimeError("Stage5 H100 smoke barrier is invalid or belongs to another run")
    return barrier


def _resolve_artifact(record: Mapping[str, Any], roots: Mapping[str, Path]) -> Path:
    root_id = str(record.get("root_id", ""))
    if root_id not in roots:
        raise RuntimeError(f"Unknown Stage5 artifact root: {root_id}")
    declared_root = roots[root_id].absolute()
    if declared_root.is_symlink():
        raise RuntimeError("Stage5 artifact root must not be a symlink")
    root = declared_root.resolve(strict=True)
    relative = Path(str(record.get("relative_path", "")))
    if relative.is_absolute() or ".." in relative.parts:
        raise RuntimeError("Stage5 artifact has an unsafe relative path")
    unresolved = root / relative
    current = unresolved
    while current != root:
        if current.is_symlink():
            raise RuntimeError("Stage5 artifact path must not traverse symlinks")
        current = current.parent
    path = unresolved.resolve(strict=True)
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise RuntimeError("Stage5 artifact escaped its declared root") from exc
    if not path.is_file() or path.is_symlink():
        raise FileNotFoundError(path)
    generation = file_generation(path)
    if generation.size != int(record["bytes"]) or _stable_file_sha256(path, generation) != str(record["sha256"]):
        raise RuntimeError(f"Stage5 artifact bytes changed: {path}")
    return path


@lru_cache(maxsize=4096)
def _cached_file_sha256(generation: FileGeneration) -> str:
    return sha256_file(Path(generation.path))


@lru_cache(maxsize=4096)
def _cached_flow_array_sha256(generation: FileGeneration) -> str:
    return array_sha256(load_flow_npz(generation.path))


def _stable_generation_value(
    path: Path,
    generation: FileGeneration,
    cached_reader: Callable[[FileGeneration], str],
    direct_reader: Callable[[Path], str],
) -> str:
    """Read one value and reject a file that changed during that read.

    Linux ctime/inode metadata safely scopes reuse to one file generation. The
    project-supported Windows runtime exposes no equivalent mutation field, so
    local verification deliberately recomputes the value on every call.
    """

    value = cached_reader(generation) if generation_cache_is_safe() else direct_reader(path)
    if file_generation(path) != generation:
        raise RuntimeError(f"Stage5 artifact changed while it was being verified: {path}")
    return value


def _stable_file_sha256(path: Path, generation: FileGeneration) -> str:
    return _stable_generation_value(path, generation, _cached_file_sha256, sha256_file)


def _stable_flow_array_sha256(path: Path, generation: FileGeneration) -> str:
    return _stable_generation_value(
        path,
        generation,
        _cached_flow_array_sha256,
        lambda value: array_sha256(load_flow_npz(value)),
    )


def _verify_decision_artifacts(record: Mapping[str, Any], roots: Mapping[str, Path]) -> None:
    for name in ("certified_source_field", "requested_field", "candidate_field", "returned_field"):
        path = _resolve_artifact(record[name], roots)
        generation = file_generation(path)
        if _stable_flow_array_sha256(path, generation) != str(record[name]["array_sha256"]):
            raise RuntimeError(f"Stage5 {name} array differs from its record")
    exact_path = _resolve_artifact(record["exact_report"], roots)
    exact_bytes = exact_path.read_bytes()
    exact = json.loads(exact_bytes)
    if not isinstance(exact, dict) or canonical_json_bytes(exact) != exact_bytes:
        raise RuntimeError("Stage5 decision exact report is not canonical")
    performance = {
        key: record[key]
        for key in (
            "runtime_seconds",
            "peak_memory_bytes",
            "requested_delta_rms",
            "candidate_delta_rms",
            "returned_delta_rms",
            "candidate_retained_ratio",
            "returned_retained_ratio",
        )
    }
    if (
        exact.get("schema") != "ctcf-stage5-decision-exact-report-v1"
        or exact.get("decision_id") != record["decision_id"]
        or exact.get("source_field") != record["certified_source_field"]
        or exact.get("candidate_exact", {}).get("status") != record["candidate_exact_status"]
        or exact.get("candidate_exact", {}).get("certified") != record["candidate_exact_certified"]
        or exact.get("candidate_exact", {}).get("sha256") != record["candidate_field"]["array_sha256"]
        or exact.get("returned_exact", {}).get("status") != record["returned_exact_status"]
        or exact.get("returned_exact", {}).get("certified") != record["returned_certified"]
        or exact.get("returned_exact", {}).get("sha256") != record["returned_field"]["array_sha256"]
        or canonical_sha256({"environment": exact.get("execution"), "performance": performance})
        != record["execution_sha256"]
    ):
        raise RuntimeError("Stage5 decision exact report differs from its record")


def _verify_evaluation_record(record: Mapping[str, Any], evaluation_root: Path) -> dict[str, Any]:
    metrics_path = _resolve_artifact(record["metrics_file"], {"evaluation_output_root": evaluation_root})
    evaluation = load_canonical_json(metrics_path)
    actual_file = file_record("evaluation_output_root", evaluation_root, metrics_path)
    if build_evaluation_record(evaluation, actual_file) != dict(record):
        raise RuntimeError("Stage5 evaluation record differs from its persisted metrics")
    return evaluation


def _verify_initial_controller(path: Path, *, seed: int) -> None:
    sidecar = path.with_name(f"{path.name}.sha256.json")
    if not path.is_file() or not sidecar.is_file():
        raise FileNotFoundError("Stage5 controller initial state or sidecar is missing")
    record = json.loads(sidecar.read_text(encoding="utf-8"))
    expected_sidecar = {
        "schema": "ctcf-stage5-checkpoint-sha256-v1",
        "file_name": path.name,
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }
    if record != expected_sidecar:
        raise RuntimeError("Stage5 controller initial-state sidecar changed")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    config = ControllerTrainingConfig()
    expected_fields = {"schema", "seed", "config", "config_sha256", "state", "state_sha256"}
    if not isinstance(payload, dict) or set(payload) != expected_fields:
        raise RuntimeError("Stage5 controller initial-state schema changed")
    if (
        payload["schema"] != "ctcf-stage5-controller-initial-state-v1"
        or payload["seed"] != seed
        or payload["config"] != asdict(config)
        or payload["config_sha256"] != canonical_sha256(payload["config"])
        or payload["state_sha256"] != state_dict_sha256(payload["state"])
    ):
        raise RuntimeError("Stage5 controller initial-state provenance changed")
    model = build_stage5_controller(config)
    incompatible = model.load_state_dict(payload["state"], strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError("Stage5 controller initial state is incompatible")


def _tree_bytes(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def command_selfcheck(_: argparse.Namespace) -> int:
    if tuple(BASE_SEEDS) != (0, 1, 2) or tuple(VARIANT_IDS) != ("U0", *STAGE5_VARIANTS):
        raise RuntimeError("Stage5 seed or variant inventory changed")
    if len(VARIANT_IDS) != 9 or len(STAGE5_VARIANTS) != 8:
        raise RuntimeError("Stage5 matrix must contain one U0 and eight controllers")
    for name, gib in DISK_TARGET_GIB.items():
        if gib <= DISK_RESERVE_GIB:
            raise RuntimeError(f"invalid Stage5 {name} disk target")
    print(
        json.dumps(
            {
                "schema": "ctcf-stage5-orchestrator-selfcheck-v1",
                "status": "PASS",
                "base_seeds": list(BASE_SEEDS),
                "variants": list(VARIANT_IDS),
                "controller_variants": list(STAGE5_VARIANTS),
                "heldout_test_authorized": False,
                "dice_success_threshold_present": False,
            },
            sort_keys=True,
        )
    )
    return 0


def command_disk_preflight(args: argparse.Namespace) -> int:
    assert_clean_exact_git(args.repo_root, args.expected_git_head)
    root = args.target_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    target = DISK_TARGET_GIB[args.phase] * GIB
    occupied = _tree_bytes(root)
    free = shutil.disk_usage(root).free
    additional = max(0, target - occupied) + DISK_RESERVE_GIB * GIB
    payload = {
        "schema": "ctcf-stage5-disk-preflight-v1",
        "phase": args.phase,
        "target_bytes": target,
        "already_materialized_bytes": occupied,
        "required_additional_bytes_including_reserve": additional,
        "free_bytes": free,
        "status": "PASS" if free >= additional else "FAIL",
    }
    print(json.dumps(payload, sort_keys=True))
    if free < additional:
        raise RuntimeError(
            f"Insufficient disk for Stage5 {args.phase}: free={free / GIB:.1f} GiB, required={additional / GIB:.1f} GiB"
        )
    return 0


def command_prepare_data(args: argparse.Namespace) -> int:
    assert_clean_exact_git(args.repo_root, args.expected_git_head)
    result = prepare_stage5_oasis_data(
        args.oasis_all_root,
        args.manifest_root,
        args.image_root,
        privileged_prepare_authorization=STAGE5_PREPARE_AUTHORIZATION,
    )
    print(json.dumps({"data_contract": str(result.contract_path), "sha256": result.contract_sha256}))
    return 0


def command_prepare_protocol(args: argparse.Namespace) -> int:
    head = assert_clean_exact_git(args.repo_root, args.expected_git_head)
    bundle = prepare_protocol_bundle(
        git_head=head,
        data_contract_path=args.data_contract,
        output_root=args.output_root,
        u0_config=U0TrainingConfig(),
        controller_config=ControllerTrainingConfig(),
        bootstrap_policy="collar_repair",
    )
    print(json.dumps(bundle, sort_keys=True))
    return 0


def command_smoke(args: argparse.Namespace) -> int:
    head, protocol = _protocol_context(args)
    report = smoke_stage5_runtime(
        data_contract=args.data_contract,
        image_root=args.image_root,
        output_root=args.output_root,
        seed=0,
        device=torch.device(args.device),
        git_head=head,
        protocol_sha256=canonical_sha256(protocol),
        u0_training_contract_sha256=protocol["u0_training_contract_sha256"],
        controller_training_contract_sha256=protocol["controller_training_contract_sha256"],
        bootstrap_policy=protocol["bootstrap"]["policy"],
        u0_config=U0TrainingConfig(),
        controller_config=ControllerTrainingConfig(),
    )
    print(f"[STAGE5 H100 SMOKE] {report}")
    return 0


def command_freeze_smoke(args: argparse.Namespace) -> int:
    head, protocol = _protocol_context(args)
    report = load_canonical_json(args.smoke_report)
    _validate_smoke_report(report, protocol, head)
    barrier = {
        "schema": SMOKE_BARRIER_SCHEMA,
        "status": "COMPLETE",
        "git_head": head,
        "protocol_sha256": canonical_sha256(protocol),
        "data_contract_sha256": protocol["data_contract_sha256"],
        "smoke_report_bytes": args.smoke_report.stat().st_size,
        "smoke_report_sha256": sha256_file(args.smoke_report),
    }
    write_immutable_json(args.output, barrier)
    _validate_smoke_gate(args.output, args.smoke_report, protocol, head)
    print(f"[STAGE5 H100 SMOKE BARRIER] {canonical_sha256(barrier)}")
    return 0


def command_train_u0(args: argparse.Namespace) -> int:
    head, protocol = _protocol_context(args)
    output_root = args.checkpoint_root / "u0" / f"seed_{args.seed}"
    endpoint = output_root / "last.pth"
    result = train_u0(
        data_contract=args.data_contract,
        image_root=args.image_root,
        output_root=output_root,
        seed=args.seed,
        device=torch.device(args.device),
        git_head=head,
        protocol_sha256=canonical_sha256(protocol),
        training_contract_sha256=protocol["u0_training_contract_sha256"],
        config=U0TrainingConfig(),
        resume=endpoint if endpoint.is_file() else None,
    )
    print(f"[STAGE5 U0] {result}")
    return 0


def command_materialize_source(args: argparse.Namespace) -> int:
    _, protocol = _protocol_context(args)
    count = materialize_source_fields(
        data_contract=args.data_contract,
        image_root=args.image_root,
        output_root=args.source_root,
        u0_checkpoint=checkpoint_path(args.checkpoint_root, seed=args.seed, variant="U0"),
        seed=args.seed,
        device=torch.device(args.device),
        protocol_sha256=canonical_sha256(protocol),
        u0_training_contract_sha256=protocol["u0_training_contract_sha256"],
        u0_config=U0TrainingConfig(),
        bootstrap_policy=protocol["bootstrap"]["policy"],
        shard_index=args.shard_index,
        num_shards=args.num_shards,
    )
    print(f"[STAGE5 SOURCE] seed={args.seed} shard={args.shard_index}/{args.num_shards} count={count}")
    return 0


def command_init_controller(args: argparse.Namespace) -> int:
    _protocol_context(args)
    path = args.checkpoint_root / "controller_initial" / f"seed_{args.seed}" / "initial.pth"
    if path.is_file() and path.with_name(f"{path.name}.sha256.json").is_file():
        _verify_initial_controller(path, seed=args.seed)
        print(f"[STAGE5 CONTROLLER INITIAL RESUME] {path}")
        return 0
    digest = initialize_controller_state(seed=args.seed, config=ControllerTrainingConfig(), output_path=path)
    print(f"[STAGE5 CONTROLLER INITIAL] {path} {digest}")
    return 0


def command_train_controller(args: argparse.Namespace) -> int:
    head, protocol = _protocol_context(args)
    output_root = args.checkpoint_root / "controllers" / f"seed_{args.seed}" / args.variant
    endpoint = output_root / "last.pth"
    result = train_controller(
        data_contract=args.data_contract,
        image_root=args.image_root,
        output_root=output_root,
        base_checkpoint=checkpoint_path(args.checkpoint_root, seed=args.seed, variant="U0"),
        initial_controller=args.checkpoint_root / "controller_initial" / f"seed_{args.seed}" / "initial.pth",
        variant=args.variant,
        seed=args.seed,
        device=torch.device(args.device),
        git_head=head,
        protocol_sha256=canonical_sha256(protocol),
        u0_training_contract_sha256=protocol["u0_training_contract_sha256"],
        training_contract_sha256=protocol["controller_training_contract_sha256"],
        bootstrap_policy=protocol["bootstrap"]["policy"],
        config=ControllerTrainingConfig(),
        resume=endpoint if endpoint.is_file() else None,
    )
    print(f"[STAGE5 CONTROLLER] {result}")
    return 0


def command_freeze_training(args: argparse.Namespace) -> int:
    _protocol_context(args)
    barrier = freeze_training_barrier(
        protocol_path=args.protocol,
        checkpoint_root=args.checkpoint_root,
        output_path=args.output,
    )
    print(f"[STAGE5 TRAINING BARRIER] {canonical_sha256(barrier)}")
    return 0


def command_decide(args: argparse.Namespace) -> int:
    _protocol_context(args)
    roots = {"source_field_root": args.source_root, "decision_output_root": args.decision_root}
    records_root = args.decision_root / "records"
    if records_root.is_dir():
        for path in sorted(records_root.glob("*.json")):
            record = load_canonical_json(path)
            if record.get("seed") == args.seed and record.get("variant_id") == args.variant:
                _verify_decision_artifacts(record, roots)
    count = materialize_decisions(
        protocol_path=args.protocol,
        training_barrier_path=args.training_barrier,
        data_contract_path=args.data_contract,
        image_root=args.image_root,
        checkpoint_root=args.checkpoint_root,
        source_root=args.source_root,
        decision_root=args.decision_root,
        seed=args.seed,
        variant=args.variant,
        shard_index=args.shard_index,
        num_shards=args.num_shards,
        device=torch.device(args.device),
        controller_config=ControllerTrainingConfig(),
    )
    print(f"[STAGE5 DECISION] seed={args.seed} variant={args.variant} count={count}")
    return 0


def command_freeze_decision(args: argparse.Namespace) -> int:
    _protocol_context(args)
    roots = {"source_field_root": args.source_root, "decision_output_root": args.decision_root}
    for path in sorted((args.decision_root / "records").glob("*.json")):
        _verify_decision_artifacts(load_canonical_json(path), roots)
    barrier = freeze_decision_barrier(
        protocol_path=args.protocol,
        training_barrier_path=args.training_barrier,
        decision_root=args.decision_root,
        output_path=args.output,
    )
    print(f"[STAGE5 DECISION BARRIER] {canonical_sha256(barrier)}")
    return 0


def _evaluation_context(args: argparse.Namespace) -> tuple[EvaluationContext, Any]:
    _, protocol = _protocol_context(args)
    training = load_canonical_json(args.training_barrier)
    decision = load_canonical_json(args.decision_barrier)
    validate_training_barrier(training, protocol, require_complete=True)
    validate_decision_barrier(decision, protocol, training, require_complete=True)
    expected_sha = require_sha256(args.decision_barrier_sha256, "--decision-barrier-sha256", error=ValueError)
    if canonical_sha256(decision) != expected_sha:
        raise RuntimeError("Stage5 decision barrier differs from the operator-frozen SHA-256")
    runtime = load_stage5_runtime_contract(args.data_contract)
    context = EvaluationContext.from_barriers(protocol, training, decision, runtime.pairs["cases"])
    return context, runtime


def _accept_existing_evaluation(context: Any, decision_id: str, record_path: Path, evaluation_root: Path) -> None:
    """Re-authenticate an evaluation an earlier run of this shard already wrote."""
    record = load_canonical_json(record_path)
    build_evaluation_barrier(context.protocol, context.training_barrier, context.decision_barrier, [record])
    evaluation = _verify_evaluation_record(record, evaluation_root)
    if evaluation.get("decision_id") != decision_id:
        raise RuntimeError("Existing Stage5 evaluation has another identity")


def _recover_evaluation_record(
    context: Any,
    decision: Mapping[str, Any],
    decision_id: str,
    metrics_path: Path,
    record_path: Path,
    evaluation_root: Path,
) -> None:
    """Rebuild the record for metrics that survived a crash before their record was written."""
    evaluation = load_canonical_json(metrics_path)
    if (
        evaluation.get("schema") != EVALUATION_SCHEMA
        or evaluation.get("decision_id") != decision_id
        or evaluation.get("decision_record_sha256") != canonical_sha256(decision)
        or evaluation.get("protocol_sha256") != context.protocol_sha256
        or evaluation.get("training_barrier_sha256") != context.training_barrier_sha256
        or evaluation.get("decision_barrier_sha256") != context.decision_barrier_sha256
    ):
        raise RuntimeError(f"Orphan Stage5 evaluation metrics are not recoverable: {decision_id}")
    record = build_evaluation_record(
        evaluation,
        file_record("evaluation_output_root", evaluation_root, metrics_path),
    )
    build_evaluation_barrier(context.protocol, context.training_barrier, context.decision_barrier, [record])
    write_immutable_json(record_path, record)
    _verify_evaluation_record(record, evaluation_root)
    print(f"[STAGE5 EVALUATION RECOVERED] {decision_id}")


def command_evaluate(args: argparse.Namespace) -> int:
    context, _ = _evaluation_context(args)
    label_store = Stage5OasisEvaluationLabelStore(
        args.data_contract,
        args.oasis_all_root,
        protocol_path=args.protocol,
        training_barrier_path=args.training_barrier,
        decision_barrier_path=args.decision_barrier,
        label_evaluation_authorization=STAGE5_LABEL_EVALUATION_AUTHORIZATION,
    )
    roots = {
        "source_field_root": args.source_root,
        "decision_output_root": args.decision_root,
    }
    selected_cases = {
        case_id
        for index, case_id in enumerate(context.protocol["directed_case_ids"])
        if index % args.num_shards == args.shard_index
    }
    decisions = sorted(context.decisions.values(), key=lambda item: item["decision_id"])
    completed = 0
    cached_case_id: str | None = None
    cached_labels: tuple[Any, Any] | None = None
    for decision in decisions:
        if decision["case_id"] not in selected_cases:
            continue
        decision_id = str(decision["decision_id"])
        metrics_path = args.evaluation_root / "metrics" / f"{decision_id}.json"
        record_path = args.evaluation_root / "records" / f"{decision_id}.json"
        if metrics_path.is_file() and record_path.is_file():
            _accept_existing_evaluation(context, decision_id, record_path, args.evaluation_root)
            completed += 1
            continue
        if metrics_path.is_file() and not record_path.exists():
            _recover_evaluation_record(
                context,
                decision,
                decision_id,
                metrics_path,
                record_path,
                args.evaluation_root,
            )
            completed += 1
            continue
        if metrics_path.exists() or record_path.exists():
            raise RuntimeError(f"Partial Stage5 evaluation exists for {decision_id}")
        returned_path = _resolve_artifact(decision["returned_field"], roots)
        requested_path = _resolve_artifact(decision["requested_field"], roots)
        candidate_path = _resolve_artifact(decision["candidate_field"], roots)
        case_id = str(decision["case_id"])
        if cached_case_id != case_id:
            cached_labels = label_store.load_case_labels(case_id)
            cached_case_id = case_id
        assert cached_labels is not None
        moving_label, fixed_label = cached_labels
        evaluation = evaluate_returned_decision(
            context,
            decision_id,
            returned_path,
            moving_label,
            fixed_label,
            requested_field_path=requested_path,
            candidate_field_path=candidate_path,
            device=torch.device(args.device),
        )
        write_decision_metrics(metrics_path, evaluation)
        record = build_evaluation_record(
            evaluation,
            file_record("evaluation_output_root", args.evaluation_root, metrics_path),
        )
        write_immutable_json(record_path, record)
        completed += 1
    print(f"[STAGE5 EVALUATION] shard={args.shard_index}/{args.num_shards} count={completed}")
    return 0


def command_freeze_evaluation(args: argparse.Namespace) -> int:
    context, _ = _evaluation_context(args)
    records = [load_canonical_json(path) for path in sorted((args.evaluation_root / "records").glob("*.json"))]
    for record in records:
        _verify_evaluation_record(record, args.evaluation_root)
    barrier = build_evaluation_barrier(
        context.protocol,
        context.training_barrier,
        context.decision_barrier,
        records,
    )
    validate_evaluation_barrier(
        barrier,
        context.protocol,
        context.training_barrier,
        context.decision_barrier,
        require_complete=True,
    )
    write_immutable_json(args.output, barrier)
    print(f"[STAGE5 EVALUATION BARRIER] {canonical_sha256(barrier)}")
    return 0


def command_aggregate(args: argparse.Namespace) -> int:
    context, _ = _evaluation_context(args)
    evaluation_barrier = load_canonical_json(args.evaluation_barrier)
    validate_evaluation_barrier(
        evaluation_barrier,
        context.protocol,
        context.training_barrier,
        context.decision_barrier,
        require_complete=True,
    )
    for record in evaluation_barrier["records"]:
        _verify_evaluation_record(record, args.evaluation_root)
    evaluations = {
        decision_id: load_canonical_json(args.evaluation_root / "metrics" / f"{decision_id}.json")
        for decision_id in sorted(context.decisions)
    }
    roots = {
        "source_field_root": args.source_root,
        "decision_output_root": args.decision_root,
    }
    pairs: list[dict[str, Any]] = []
    for pair_id in sorted(context.pairs):
        case_ids = context.pairs[pair_id]
        for seed in BASE_SEEDS:
            for variant in VARIANT_IDS:
                decision_ids = [f"{case_id}__S{seed}__{variant}" for case_id in case_ids]
                pair_evaluation = build_pair_evaluation(
                    context,
                    evaluations[decision_ids[0]],
                    evaluations[decision_ids[1]],
                    _resolve_artifact(context.decisions[decision_ids[0]]["returned_field"], roots),
                    _resolve_artifact(context.decisions[decision_ids[1]]["returned_field"], roots),
                    device=torch.device(args.device),
                )
                pairs.append(pair_evaluation)
    aggregate = aggregate_pair_effects(context, pairs)
    products = write_evaluation_products(args.output_root, list(evaluations.values()), pairs, aggregate)
    print(json.dumps({"stage5_products": products}, sort_keys=True))
    return 0


def _validate_complete_compact_run(run_root: Path, git_head: str) -> None:
    data_root = run_root / "data_attestations"
    protocol_path = run_root / "protocol" / "protocol.json"
    training_path = run_root / "barriers" / "training_barrier.json"
    decision_path = run_root / "barriers" / "decision_barrier.json"
    evaluation_path = run_root / "barriers" / "evaluation_barrier.json"
    smoke_report_path = run_root / "smoke" / "smoke_report.json"
    smoke_barrier_path = run_root / "barriers" / "smoke_barrier.json"
    products_root = run_root / "evaluation" / "products"
    evaluation_root = run_root / "evaluation"

    protocol = _read_protocol(protocol_path, git_head)
    runtime = load_stage5_runtime_contract(data_root / "data_contract.json")
    if runtime.contract_sha256 != protocol["data_contract_sha256"]:
        raise RuntimeError("Stage5 compact data contract differs from the frozen protocol")
    _validate_smoke_gate(smoke_barrier_path, smoke_report_path, protocol, git_head)
    training = load_canonical_json(training_path)
    decision = load_canonical_json(decision_path)
    evaluation_barrier = load_canonical_json(evaluation_path)
    validate_training_barrier(training, protocol, require_complete=True)
    validate_decision_barrier(decision, protocol, training, require_complete=True)
    validate_evaluation_barrier(evaluation_barrier, protocol, training, decision, require_complete=True)
    context = EvaluationContext.from_barriers(protocol, training, decision, runtime.pairs["cases"])

    evaluations: list[dict[str, Any]] = []
    for record in evaluation_barrier["records"]:
        evaluations.append(_verify_evaluation_record(record, evaluation_root))
    evaluations.sort(key=lambda item: item["evaluation_id"])

    bundle_path = products_root / "evaluation_bundle.json"
    bundle = load_canonical_json(bundle_path)
    if (
        set(bundle) != {"schema", "evaluations", "pair_evaluations", "aggregate"}
        or bundle.get("schema") != "ctcf-stage5-evaluation-products-v1"
    ):
        raise RuntimeError("Stage5 evaluation bundle schema changed")
    if bundle["evaluations"] != evaluations:
        raise RuntimeError("Stage5 evaluation bundle differs from its immutable evaluation records")
    recomputed = aggregate_pair_effects(context, bundle["pair_evaluations"])
    if recomputed != bundle["aggregate"]:
        raise RuntimeError("Stage5 aggregate is not reproducible from its pair-level rows")

    expected_products = {
        "evaluation_bundle.json": canonical_json_bytes(bundle).decode("utf-8"),
        "per_decision.csv": decision_csv(evaluations),
        "per_label.csv": per_label_csv(evaluations),
        "geometry_metrics.csv": geometry_csv(evaluations),
        "field_stage_diagnostics.csv": field_stage_diagnostics_csv(evaluations),
        "per_pair_metric.csv": pair_metric_csv(bundle["pair_evaluations"]),
        "paired_effects_vs_u0.csv": effect_csv(recomputed["paired_effects_vs_u0"]),
        "planned_contrasts.csv": effect_csv(recomputed["planned_contrasts"]),
        "decision_diagnostics.csv": diagnostic_csv(recomputed),
    }
    for name, expected in expected_products.items():
        product = products_root / name
        if not product.is_file() or product.read_text(encoding="utf-8") != expected:
            raise RuntimeError(f"Stage5 compact product is missing or internally inconsistent: {product}")


def command_finalize(args: argparse.Namespace) -> int:
    head = assert_clean_exact_git(args.repo_root, args.expected_git_head)
    if RUN_ID_RE.fullmatch(args.run_id) is None:
        raise ValueError("Invalid Stage5 run ID")
    run_root = args.run_root.resolve(strict=True)
    linked = [path.relative_to(run_root).as_posix() for path in run_root.rglob("*") if is_link_like(path)]
    if linked:
        raise RuntimeError(f"Compact Stage5 root contains linked paths: {linked[:5]}")
    forbidden = [
        path.relative_to(run_root).as_posix()
        for path in run_root.rglob("*")
        if path.is_file() and path.suffix.lower() in HEAVY_SUFFIXES
    ]
    if forbidden:
        raise RuntimeError(f"Compact Stage5 root contains heavy or label-bearing files: {forbidden[:5]}")
    if args.status == "COMPLETE":
        required = (
            run_root / "data_attestations" / "source_inventory.json",
            run_root / "data_attestations" / "split_manifest.json",
            run_root / "data_attestations" / "pair_manifest.json",
            run_root / "data_attestations" / "data_contract.json",
            run_root / "protocol" / "protocol.json",
            run_root / "smoke" / "smoke_report.json",
            run_root / "barriers" / "smoke_barrier.json",
            run_root / "barriers" / "training_barrier.json",
            run_root / "barriers" / "decision_barrier.json",
            run_root / "barriers" / "evaluation_barrier.json",
            run_root / "evaluation" / "products" / "evaluation_bundle.json",
            run_root / "evaluation" / "products" / "per_decision.csv",
            run_root / "evaluation" / "products" / "per_label.csv",
            run_root / "evaluation" / "products" / "geometry_metrics.csv",
            run_root / "evaluation" / "products" / "field_stage_diagnostics.csv",
            run_root / "evaluation" / "products" / "per_pair_metric.csv",
            run_root / "evaluation" / "products" / "planned_contrasts.csv",
            run_root / "evaluation" / "products" / "paired_effects_vs_u0.csv",
            run_root / "evaluation" / "products" / "decision_diagnostics.csv",
        )
        missing = [str(path) for path in required if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"Cannot finalize COMPLETE Stage5 run: {missing}")
        _validate_complete_compact_run(run_root, head)
    manifest_dir = run_root / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = manifest_dir / f"{args.attempt_id}.json"
    outputs_path = manifest_dir / f"{args.attempt_id}.outputs.tsv"
    excluded = {manifest_path.resolve(), outputs_path.resolve()}
    rows = ["relative_path\tbytes\tsha256"]
    for path in sorted(item for item in run_root.rglob("*") if item.is_file()):
        if path.resolve() in excluded:
            continue
        rows.append(f"{path.relative_to(run_root).as_posix()}\t{path.stat().st_size}\t{sha256_file(path)}")
    atomic_write_text(outputs_path, "\n".join(rows) + "\n")
    payload = {
        "schema": "ctcf-stage5-compact-run-manifest-v1",
        "run_id": args.run_id,
        "attempt_id": args.attempt_id,
        "status": args.status,
        "exit_code": args.exit_code,
        "started_at_utc": args.started_at_utc,
        "completed_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "git_head": head,
        "tracked_tree_clean": True,
        "heldout_test_accessed": False,
        "heldout_test_member_payload_extracted": False,
        "heldout_test_decoded_or_evaluated": False,
        "data_source_scope": "LOCAL_OASIS_L2R_ALL_PICKLES",
        "network_identity_lookup_performed": False,
        "contains_checkpoint_bytes": False,
        "contains_field_bytes": False,
        "contains_label_bytes": False,
        "outputs_file": {
            "relative_path": outputs_path.relative_to(run_root).as_posix(),
            "sha256": sha256_file(outputs_path),
        },
        "remote_heavy_locator": args.remote_heavy_locator,
    }
    write_immutable_json(manifest_path, payload)
    print(f"[STAGE5 MANIFEST] {manifest_path}")
    return 0


def _add_git(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--expected-git-head", required=True)


def _add_protocol(parser: argparse.ArgumentParser) -> None:
    _add_git(parser)
    parser.add_argument("--protocol", type=Path, required=True)


def _add_data(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--data-contract", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)


def _add_smoke_gate(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--smoke-barrier", type=Path, required=True)
    parser.add_argument("--smoke-report", type=Path, required=True)


def _add_shard(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--num-shards", type=int, required=True)


def _add_evaluation_context(parser: argparse.ArgumentParser) -> None:
    _add_protocol(parser)
    parser.add_argument("--training-barrier", type=Path, required=True)
    parser.add_argument("--decision-barrier", type=Path, required=True)
    parser.add_argument("--decision-barrier-sha256", required=True)
    parser.add_argument("--data-contract", type=Path, required=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fail-closed Stage5 learned-controller orchestration.")
    actions = parser.add_subparsers(dest="action", required=True)

    def action(name: str, handler: Callable[[argparse.Namespace], int]) -> argparse.ArgumentParser:
        """Declare one subcommand and the function that runs it, in one place."""
        subparser = actions.add_parser(name)
        subparser.set_defaults(handler=handler)
        return subparser

    action("selfcheck", command_selfcheck)

    disk = action("disk-preflight", command_disk_preflight)
    _add_git(disk)
    disk.add_argument("--phase", choices=tuple(DISK_TARGET_GIB), required=True)
    disk.add_argument("--target-root", type=Path, required=True)

    prepare_data = action("prepare-data", command_prepare_data)
    _add_git(prepare_data)
    prepare_data.add_argument("--oasis-all-root", type=Path, required=True)
    prepare_data.add_argument("--manifest-root", type=Path, required=True)
    prepare_data.add_argument("--image-root", type=Path, required=True)

    prepare_protocol = action("prepare-protocol", command_prepare_protocol)
    _add_git(prepare_protocol)
    prepare_protocol.add_argument("--data-contract", type=Path, required=True)
    prepare_protocol.add_argument("--output-root", type=Path, required=True)

    smoke = action("smoke", command_smoke)
    _add_protocol(smoke)
    _add_data(smoke)
    smoke.add_argument("--device", required=True)
    smoke.add_argument("--output-root", type=Path, required=True)

    freeze_smoke = action("freeze-smoke", command_freeze_smoke)
    _add_protocol(freeze_smoke)
    freeze_smoke.add_argument("--smoke-report", type=Path, required=True)
    freeze_smoke.add_argument("--output", type=Path, required=True)

    u0 = action("train-u0", command_train_u0)
    _add_protocol(u0)
    _add_data(u0)
    _add_smoke_gate(u0)
    u0.add_argument("--device", required=True)
    u0.add_argument("--checkpoint-root", type=Path, required=True)
    u0.add_argument("--seed", type=int, choices=BASE_SEEDS, required=True)

    source = action("materialize-source", command_materialize_source)
    _add_protocol(source)
    _add_data(source)
    _add_smoke_gate(source)
    source.add_argument("--device", required=True)
    _add_shard(source)
    source.add_argument("--checkpoint-root", type=Path, required=True)
    source.add_argument("--source-root", type=Path, required=True)
    source.add_argument("--seed", type=int, choices=BASE_SEEDS, required=True)

    initial = action("init-controller", command_init_controller)
    _add_protocol(initial)
    initial.add_argument("--checkpoint-root", type=Path, required=True)
    initial.add_argument("--seed", type=int, choices=BASE_SEEDS, required=True)

    controller = action("train-controller", command_train_controller)
    _add_protocol(controller)
    _add_data(controller)
    _add_smoke_gate(controller)
    controller.add_argument("--device", required=True)
    controller.add_argument("--checkpoint-root", type=Path, required=True)
    controller.add_argument("--seed", type=int, choices=BASE_SEEDS, required=True)
    controller.add_argument("--variant", choices=STAGE5_VARIANTS, required=True)

    training = action("freeze-training", command_freeze_training)
    _add_protocol(training)
    _add_smoke_gate(training)
    training.add_argument("--checkpoint-root", type=Path, required=True)
    training.add_argument("--output", type=Path, required=True)

    decide = action("decide", command_decide)
    _add_protocol(decide)
    _add_data(decide)
    _add_smoke_gate(decide)
    decide.add_argument("--device", required=True)
    _add_shard(decide)
    decide.add_argument("--training-barrier", type=Path, required=True)
    decide.add_argument("--checkpoint-root", type=Path, required=True)
    decide.add_argument("--source-root", type=Path, required=True)
    decide.add_argument("--decision-root", type=Path, required=True)
    decide.add_argument("--seed", type=int, choices=BASE_SEEDS, required=True)
    decide.add_argument("--variant", choices=VARIANT_IDS, required=True)

    decision = action("freeze-decision", command_freeze_decision)
    _add_protocol(decision)
    _add_smoke_gate(decision)
    decision.add_argument("--training-barrier", type=Path, required=True)
    decision.add_argument("--source-root", type=Path, required=True)
    decision.add_argument("--decision-root", type=Path, required=True)
    decision.add_argument("--output", type=Path, required=True)

    evaluate = action("evaluate", command_evaluate)
    _add_evaluation_context(evaluate)
    _add_smoke_gate(evaluate)
    evaluate.add_argument("--device", required=True)
    _add_shard(evaluate)
    evaluate.add_argument("--oasis-all-root", type=Path, required=True)
    evaluate.add_argument("--source-root", type=Path, required=True)
    evaluate.add_argument("--decision-root", type=Path, required=True)
    evaluate.add_argument("--evaluation-root", type=Path, required=True)

    freeze_evaluation = action("freeze-evaluation", command_freeze_evaluation)
    _add_evaluation_context(freeze_evaluation)
    _add_smoke_gate(freeze_evaluation)
    freeze_evaluation.add_argument("--evaluation-root", type=Path, required=True)
    freeze_evaluation.add_argument("--output", type=Path, required=True)

    aggregate = action("aggregate", command_aggregate)
    _add_evaluation_context(aggregate)
    _add_smoke_gate(aggregate)
    aggregate.add_argument("--device", required=True)
    aggregate.add_argument("--evaluation-barrier", type=Path, required=True)
    aggregate.add_argument("--source-root", type=Path, required=True)
    aggregate.add_argument("--decision-root", type=Path, required=True)
    aggregate.add_argument("--evaluation-root", type=Path, required=True)
    aggregate.add_argument("--output-root", type=Path, required=True)

    finalize = action("finalize", command_finalize)
    _add_git(finalize)
    finalize.add_argument("--run-root", type=Path, required=True)
    finalize.add_argument("--run-id", required=True)
    finalize.add_argument("--attempt-id", required=True)
    finalize.add_argument("--status", choices=("COMPLETE", "PARTIAL", "FAILED"), required=True)
    finalize.add_argument("--exit-code", type=int, required=True)
    finalize.add_argument("--started-at-utc", required=True)
    finalize.add_argument("--remote-heavy-locator", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if hasattr(args, "num_shards") and not 0 <= args.shard_index < args.num_shards:
        raise ValueError("Stage5 shard must satisfy 0 <= shard_index < num_shards")
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
