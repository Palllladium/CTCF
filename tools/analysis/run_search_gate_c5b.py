from __future__ import annotations

import argparse
import csv
import io
import json
import math
import platform
import shutil
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch

from tools.analysis.run_artifacts import atomic_write_json, atomic_write_text, sha256_file
from tools.analysis.search_gate_c5b import (
    ARM_SPECS,
    C5B_POLICY_SHA256,
    PROTOCOL_ID,
    TEST_115_AUTHORIZED,
    assert_frozen_policy,
)
from tools.analysis.search_gate_c5b_contracts import (
    build_decision_barrier,
    build_evaluation_barrier,
    freeze_evaluation_contract,
    load_barrier,
    load_contracts,
    load_decision_contract_isolated,
    load_evaluation_barrier,
    load_evaluation_contract,
    prepare_contracts,
)
from tools.analysis.search_gate_c5b_workers import (
    finalize_c5b,
    run_decision_case,
    run_decision_worker,
    run_evaluation_worker,
)
from tools.analysis.search_gate_common import git, utc_now
from tools.analysis.search_gate_runtime import parse_physical_gpus
from utils import setup_device

DEFAULT_MIN_FREE_GIB = 50.0
RESUME_MIN_FREE_GIB = 5.0


def _runtime_signature() -> dict[str, Any]:
    import scipy

    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
    }


def _assert_clean_runtime(decision: Mapping[str, Any], stage: str) -> None:
    if git("rev-parse", "HEAD") != decision["git_head"] or git("status", "--porcelain=v1"):
        raise RuntimeError(f"C5b {stage} code differs from its clean prepared contract")
    observed = _runtime_signature()
    if observed != dict(decision["runtime_signature"]):
        raise RuntimeError(f"C5b {stage} runtime changed: {observed} != {dict(decision['runtime_signature'])}")


def _tree_bytes(root: Path) -> int:
    return sum(path.stat().st_size for path in root.rglob("*") if path.is_file()) if root.exists() else 0


def _validate_disk_budget(target: Path, minimum_free_gib: float) -> None:
    if not math.isfinite(minimum_free_gib) or minimum_free_gib < 0:
        raise ValueError("C5b disk budget must be finite and non-negative")
    target.parent.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(target.parent).free / 2**30
    retained = _tree_bytes(target) / 2**30
    if target.exists():
        if free < RESUME_MIN_FREE_GIB or free + retained < minimum_free_gib:
            raise RuntimeError(
                f"C5b resume lacks disk: free={free:.2f} GiB, retained={retained:.2f} GiB, required={minimum_free_gib:.2f} GiB"
            )
    elif free < minimum_free_gib:
        raise RuntimeError(f"C5b requires {minimum_free_gib:.2f} GiB free; found {free:.2f} GiB")


def _dataset_tsv(raw_inputs: Mapping[str, Mapping[str, Any]]) -> str:
    fields = ("dataset", "split", "case_id", "path", "bytes", "sha256", "mtime_utc")
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields, delimiter="\t", lineterminator="\n")
    writer.writeheader()
    for record in raw_inputs.values():
        writer.writerow({field: record.get(field, "") for field in fields})
    return stream.getvalue()


def _immutable_text(path: Path, value: str) -> None:
    if path.exists():
        if path.read_text(encoding="utf-8") != value:
            raise FileExistsError(f"Refusing to replace immutable C5b artifact: {path}")
    else:
        atomic_write_text(path, value)


def _load_pair(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any], str, str]:
    source_sha = str(args.source_contract_sha256)
    decision_sha = str(args.decision_contract_sha256)
    source, decision = load_contracts(args.run_root, source_sha, decision_sha)
    return source, decision, source_sha, decision_sha


def _load_decision(args: argparse.Namespace) -> tuple[dict[str, Any], str]:
    decision_sha = str(args.decision_contract_sha256)
    return load_decision_contract_isolated(args.run_root, decision_sha), decision_sha


def _load_evaluation_bundle(args: argparse.Namespace) -> tuple[Any, ...]:
    source, decision, source_sha, decision_sha = _load_pair(args)
    barrier_sha = str(args.barrier_sha256)
    barrier = load_barrier(args.run_root, barrier_sha, decision, decision_sha)
    evaluation_sha = str(args.evaluation_contract_sha256)
    evaluation = load_evaluation_contract(
        args.run_root,
        evaluation_sha,
        source,
        source_sha,
        decision,
        decision_sha,
        barrier_sha,
    )
    return source, decision, source_sha, decision_sha, barrier, barrier_sha, evaluation, evaluation_sha


def _execution(
    decision: Mapping[str, Any],
    *,
    phase: str,
    attempt_id: str,
    shard_index: int,
    physical_gpu: str,
    device: torch.device,
) -> dict[str, Any]:
    return {
        "phase": phase,
        "attempt_id": attempt_id,
        "shard_index": shard_index,
        "physical_gpu": physical_gpu,
        "host": platform.node(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(device),
        "seed": decision["seed"],
        "deterministic": True,
        "labels_loaded_to_device": phase == "evaluation",
    }


def selfcheck_stage(args: argparse.Namespace) -> int:
    assert_frozen_policy()
    report = {
        "schema": "ctcf-search-c5b-selfcheck-v1",
        "protocol_id": PROTOCOL_ID,
        "status": "PASS",
        "policy_sha256": C5B_POLICY_SHA256,
        "arm_ids": [row.arm_id for row in ARM_SPECS],
        "test_115_authorized": TEST_115_AUTHORIZED,
    }
    atomic_write_json(args.output, report)
    print(json.dumps(report, indent=2))
    return 0


def prepare_stage(args: argparse.Namespace) -> int:
    if git("status", "--porcelain=v1"):
        raise RuntimeError("C5b prepare requires a clean tracked and untracked tree")
    physical_gpus = parse_physical_gpus(
        args.physical_gpus,
        args.num_shards,
        "C5b requires exactly one unique non-negative physical GPU per shard",
    )
    run_root, heavy_root = args.run_root.resolve(), args.heavy_root.resolve()
    if run_root == heavy_root or run_root in heavy_root.parents or heavy_root in run_root.parents:
        raise RuntimeError("C5b compact and heavy roots must not overlap")
    _validate_disk_budget(heavy_root, args.min_free_gib)
    source, decision, source_sha, decision_sha = prepare_contracts(
        run_root=run_root,
        target_heavy_root=heavy_root,
        source_c5_dir=args.source_c5_dir,
        source_c5_heavy_root=args.source_c5_heavy_root,
        git_head=git("rev-parse", "HEAD"),
        runtime_signature=_runtime_signature(),
        physical_gpus=physical_gpus,
    )
    _immutable_text(run_root / "datasets.tsv", _dataset_tsv(source["raw_inputs"]))
    roots = decision["roots"]
    _immutable_text(
        run_root / "heavy_retention.txt",
        "".join(f"{key}={value}\n" for key, value in roots.items())
        + "retention=RETAIN_ALL_FOUR_ROOTS_UNTIL_EXPLICIT_OPERATOR_DECISION\npackaged=false\n",
    )
    print(
        json.dumps(
            {
                "source_contract_sha256": source_sha,
                "decision_contract_sha256": decision_sha,
                "n_cases": len(decision["case_ids"]),
                "new_heavy_field_count": 4 * len(decision["case_ids"]),
            }
        )
    )
    return 0


def decision_pilot_stage(args: argparse.Namespace) -> int:
    decision, decision_sha = _load_decision(args)
    _assert_clean_runtime(decision, "decision pilot")
    if args.num_shards != decision["num_shards"] or args.physical_gpu != decision["shard_to_physical_gpu"]["0"]:
        raise RuntimeError("C5b pilot settings differ from the frozen contract")
    device = setup_device(args.gpu, seed=decision["seed"], deterministic=True)
    if device.type != "cuda":
        raise RuntimeError("C5b decision pilot requires CUDA")
    case_id = decision["shards"]["0"][0]
    marker = run_decision_case(
        case_id=case_id,
        shard_index=0,
        physical_gpu=args.physical_gpu,
        run_root=args.run_root,
        decision=decision,
        decision_sha256=decision_sha,
        device=device,
        execution=_execution(
            decision,
            phase="decision",
            attempt_id=args.attempt_id,
            shard_index=0,
            physical_gpu=args.physical_gpu,
            device=device,
        ),
    )
    print(f"[C5B DECISION PILOT COMPLETE] case={case_id} marker={marker}")
    return 0


def decision_worker_stage(args: argparse.Namespace) -> int:
    decision, decision_sha = _load_decision(args)
    _assert_clean_runtime(decision, "decision worker")
    if args.num_shards != decision["num_shards"] or args.physical_gpu != decision["shard_to_physical_gpu"].get(
        str(args.shard_index)
    ):
        raise RuntimeError("C5b worker settings differ from the frozen contract")
    device = setup_device(args.gpu, seed=decision["seed"], deterministic=True)
    if device.type != "cuda":
        raise RuntimeError("C5b decision worker requires CUDA")
    marker = run_decision_worker(
        case_ids=decision["shards"][str(args.shard_index)],
        shard_index=args.shard_index,
        physical_gpu=args.physical_gpu,
        attempt_id=args.attempt_id,
        run_root=args.run_root,
        decision=decision,
        decision_sha256=decision_sha,
        device=device,
        execution=_execution(
            decision,
            phase="decision",
            attempt_id=args.attempt_id,
            shard_index=args.shard_index,
            physical_gpu=args.physical_gpu,
            device=device,
        ),
    )
    print(f"[C5B DECISION WORKER COMPLETE] {marker}")
    return 0


def decision_barrier_stage(args: argparse.Namespace) -> int:
    decision, decision_sha = _load_decision(args)
    _assert_clean_runtime(decision, "decision barrier")
    path = args.run_root.resolve() / "decision_barrier.json"
    if path.is_file():
        digest = sha256_file(path)
        load_barrier(args.run_root, digest, decision, decision_sha)
        print(f"[C5B DECISION BARRIER REUSED] {digest}")
        return 0
    _, digest = build_decision_barrier(
        run_root=args.run_root,
        decision=decision,
        decision_sha256=decision_sha,
        attempt_id=args.attempt_id,
        completed_at_utc=utc_now(),
    )
    print(f"[C5B DECISION BARRIER] {digest}")
    return 0


def freeze_evaluation_stage(args: argparse.Namespace) -> int:
    source, decision, source_sha, decision_sha = _load_pair(args)
    _assert_clean_runtime(decision, "evaluation freeze")
    barrier = load_barrier(args.run_root, args.barrier_sha256, decision, decision_sha)
    _, digest = freeze_evaluation_contract(
        run_root=args.run_root,
        source=source,
        source_sha256=source_sha,
        decision=decision,
        decision_sha256=decision_sha,
        barrier=barrier,
        barrier_sha256=args.barrier_sha256,
    )
    print(f"[C5B EVALUATION CONTRACT] {digest}")
    return 0


def _verify_raw(record: Mapping[str, Any]) -> None:
    path = Path(str(record.get("path", ""))).resolve()
    if (
        not path.is_file()
        or path.stat().st_size != int(record.get("bytes", -1))
        or sha256_file(path) != record.get("sha256")
    ):
        raise RuntimeError(f"C5b frozen raw input changed: {path}")


def evaluation_worker_stage(args: argparse.Namespace) -> int:
    _, decision, _, decision_sha, barrier, barrier_sha, evaluation, evaluation_sha = _load_evaluation_bundle(args)
    _assert_clean_runtime(decision, "evaluation worker")
    if args.num_shards != decision["num_shards"] or args.physical_gpu != decision["shard_to_physical_gpu"].get(
        str(args.shard_index)
    ):
        raise RuntimeError("C5b evaluation worker settings differ from the frozen contract")
    assigned = decision["shards"][str(args.shard_index)]
    for case_id in ("atlas", *assigned):
        _verify_raw(evaluation["raw_inputs"][case_id])
    from experiments.core.inference_metrics import metric_profile_for
    from experiments.core.inference_runtime import build_infer_dataset

    device = setup_device(args.gpu, seed=decision["seed"], deterministic=True)
    if device.type != "cuda":
        raise RuntimeError("C5b evaluation worker requires CUDA")
    dataset = build_infer_dataset(
        "IXI",
        [evaluation["raw_inputs"][case_id]["path"] for case_id in assigned],
        evaluation["raw_inputs"]["atlas"]["path"],
    )
    labels = tuple(metric_profile_for("IXI").labels)
    index_by_case = {case_id: index for index, case_id in enumerate(assigned)}
    marker = run_evaluation_worker(
        case_ids=assigned,
        dataset_item_for_case=lambda case_id: dataset[index_by_case[case_id]],
        labels=labels,
        shard_index=args.shard_index,
        physical_gpu=args.physical_gpu,
        attempt_id=args.attempt_id,
        run_root=args.run_root,
        decision=decision,
        decision_sha256=decision_sha,
        barrier=barrier,
        barrier_sha256=barrier_sha,
        evaluation=evaluation,
        evaluation_sha256=evaluation_sha,
        device=device,
        execution=_execution(
            decision,
            phase="evaluation",
            attempt_id=args.attempt_id,
            shard_index=args.shard_index,
            physical_gpu=args.physical_gpu,
            device=device,
        ),
    )
    print(f"[C5B EVALUATION WORKER COMPLETE] {marker}")
    return 0


def finalize_stage(args: argparse.Namespace) -> int:
    _, decision, _, decision_sha, barrier, barrier_sha, evaluation, evaluation_sha = _load_evaluation_bundle(args)
    _assert_clean_runtime(decision, "finalization")
    evaluation_barrier_path = args.run_root.resolve() / "evaluation_barrier.json"
    if evaluation_barrier_path.is_file():
        evaluation_barrier_sha = sha256_file(evaluation_barrier_path)
        evaluation_barrier = load_evaluation_barrier(
            args.run_root,
            evaluation_barrier_sha,
            decision,
            decision_sha,
            barrier,
            barrier_sha,
            evaluation,
            evaluation_sha,
        )
    else:
        evaluation_barrier, evaluation_barrier_sha = build_evaluation_barrier(
            run_root=args.run_root,
            decision=decision,
            decision_sha256=decision_sha,
            decision_barrier=barrier,
            decision_barrier_sha256=barrier_sha,
            evaluation=evaluation,
            evaluation_sha256=evaluation_sha,
            attempt_id=args.attempt_id,
            completed_at_utc=utc_now(),
        )
    artifacts = finalize_c5b(
        run_root=args.run_root,
        decision=decision,
        decision_sha256=decision_sha,
        barrier=barrier,
        barrier_sha256=barrier_sha,
        evaluation=evaluation,
        evaluation_sha256=evaluation_sha,
        evaluation_barrier=evaluation_barrier,
        evaluation_barrier_sha256=evaluation_barrier_sha,
    )
    print(json.dumps({"status": "COMPLETE", "artifacts": artifacts}, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the frozen C5b pre-clip amplitude bridge gate.")
    sub = parser.add_subparsers(dest="action", required=True)
    selfcheck = sub.add_parser("selfcheck")
    selfcheck.add_argument("--output", type=Path, required=True)
    prepare = sub.add_parser("prepare")
    prepare.add_argument("--run-root", type=Path, required=True)
    prepare.add_argument("--heavy-root", type=Path, required=True)
    prepare.add_argument("--source-c5-dir", type=Path, required=True)
    prepare.add_argument("--source-c5-heavy-root", type=Path, required=True)
    prepare.add_argument("--num-shards", type=int, required=True)
    prepare.add_argument("--physical-gpus", required=True)
    prepare.add_argument("--min-free-gib", type=float, default=DEFAULT_MIN_FREE_GIB)
    pilot = sub.add_parser("decision-pilot")
    pilot.add_argument("--run-root", type=Path, required=True)
    pilot.add_argument("--decision-contract-sha256", required=True)
    pilot.add_argument("--num-shards", type=int, required=True)
    pilot.add_argument("--gpu", type=int, default=0)
    pilot.add_argument("--physical-gpu", required=True)
    pilot.add_argument("--attempt-id", required=True)
    for action in ("decision-worker", "evaluation-worker"):
        worker = sub.add_parser(action)
        worker.add_argument("--run-root", type=Path, required=True)
        worker.add_argument("--decision-contract-sha256", required=True)
        worker.add_argument("--shard-index", type=int, required=True)
        worker.add_argument("--num-shards", type=int, required=True)
        worker.add_argument("--gpu", type=int, default=0)
        worker.add_argument("--physical-gpu", required=True)
        worker.add_argument("--attempt-id", required=True)
        if action == "evaluation-worker":
            worker.add_argument("--source-contract-sha256", required=True)
            worker.add_argument("--barrier-sha256", required=True)
            worker.add_argument("--evaluation-contract-sha256", required=True)
    barrier = sub.add_parser("decision-barrier")
    barrier.add_argument("--run-root", type=Path, required=True)
    barrier.add_argument("--decision-contract-sha256", required=True)
    barrier.add_argument("--attempt-id", required=True)
    freeze = sub.add_parser("freeze-evaluation")
    freeze.add_argument("--run-root", type=Path, required=True)
    freeze.add_argument("--source-contract-sha256", required=True)
    freeze.add_argument("--decision-contract-sha256", required=True)
    freeze.add_argument("--barrier-sha256", required=True)
    finalize = sub.add_parser("finalize")
    finalize.add_argument("--run-root", type=Path, required=True)
    finalize.add_argument("--source-contract-sha256", required=True)
    finalize.add_argument("--decision-contract-sha256", required=True)
    finalize.add_argument("--barrier-sha256", required=True)
    finalize.add_argument("--evaluation-contract-sha256", required=True)
    finalize.add_argument("--attempt-id", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    actions = {
        "selfcheck": selfcheck_stage,
        "prepare": prepare_stage,
        "decision-pilot": decision_pilot_stage,
        "decision-worker": decision_worker_stage,
        "decision-barrier": decision_barrier_stage,
        "freeze-evaluation": freeze_evaluation_stage,
        "evaluation-worker": evaluation_worker_stage,
        "finalize": finalize_stage,
    }
    return actions[args.action](args)


if __name__ == "__main__":
    raise SystemExit(main())
