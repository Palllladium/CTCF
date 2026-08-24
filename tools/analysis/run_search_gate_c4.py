from __future__ import annotations

import argparse
import csv
import io
import json
import math
import platform
import shutil
from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from tools.analysis.run_artifacts import atomic_write_json, atomic_write_text, sha256_file
from tools.analysis.search_gate_c4 import (
    ARM_SPECS,
    C4_POLICY,
    C4_POLICY_SHA256,
    COMMON_EVIDENCE_COLLAR,
    PRIMARY_NCC_IMPROVEMENT_MIN,
    PRIMARY_NCC_WINDOW,
    PRIMARY_UTILITY_ID,
    PROTOCOL_ID,
    SUPPORT_RETENTION_MIN,
    assert_frozen_policy,
)
from tools.analysis.search_gate_c4_contracts import (
    build_decision_barrier,
    canonical_offset_table,
    load_decision_barrier,
    load_decision_contract,
    load_decision_contract_isolated,
    load_source_contract,
    payload_sha256,
    prepare_contracts,
    validate_arm_specs,
    validate_offset_table,
    validate_support_contract,
    validate_worker_marker,
    write_decision_barrier,
)
from tools.analysis.search_gate_c4_workers import finalize_c4, run_decision_worker, run_evaluation_worker
from tools.analysis.search_gate_common import git, utc_now
from tools.analysis.search_gate_runtime import parse_physical_gpus
from utils import setup_device

ARM_SPECS_SHA256 = "40e44f732723008747bfba2575536c1f796c888bee1fab259b4f672a1f17c660"
OFFSET_TABLE_SHA256 = "96b983988102c14fbabb4f491274de8fd69e43064ea904853d4d7581c0343da2"
SUPPORT_CONTRACT_SHA256 = "0d740a8c439ac26a6e2ad459f00f4c129bb7072fd735f054cc1781288bb417de"
RESUME_MIN_FREE_GIB = 5.0
EXPECTED_SOURCE_SEED = 0


def arm_contract_rows() -> list[dict[str, Any]]:
    return [asdict(spec) for spec in ARM_SPECS]


def support_contract() -> dict[str, Any]:
    return {
        "support_id": "C4_COMMON_COLLAR7_NCC7_V1",
        "collar_width": COMMON_EVIDENCE_COLLAR,
        "mask_rule": "geometry & common-valid-support",
        "utility_retention_min": SUPPORT_RETENTION_MIN,
        "descriptor_retention_policy": "diagnostic_only_nonempty",
        "utility_id": PRIMARY_UTILITY_ID,
        "window": PRIMARY_NCC_WINDOW,
        "improvement_min": PRIMARY_NCC_IMPROVEMENT_MIN,
    }


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


def _assert_clean_code(expected_head: str, stage: str) -> None:
    observed_head = git("rev-parse", "HEAD")
    status = git("status", "--porcelain=v1")
    if observed_head != expected_head or status:
        raise RuntimeError(f"C4 {stage} code differs from its clean prepared contract")


def _assert_runtime(expected: Mapping[str, Any], stage: str) -> None:
    observed = _runtime_signature()
    if observed != dict(expected):
        raise RuntimeError(f"C4 {stage} runtime changed: {observed} != {dict(expected)}")


def _verify_file_record(record: Mapping[str, Any]) -> None:
    path = Path(str(record.get("path", ""))).resolve()
    if (
        not path.is_file()
        or path.stat().st_size != int(record.get("bytes", -1))
        or sha256_file(path) != record.get("sha256")
    ):
        raise RuntimeError(f"C4 frozen raw input changed or is missing: {path}")


def _tree_bytes(root: Path) -> int:
    if not root.exists():
        return 0
    return sum(path.stat().st_size for path in root.rglob("*") if path.is_file())


def _validate_disk_budget(target: Path, minimum_free_gib: float) -> None:
    if not math.isfinite(minimum_free_gib) or minimum_free_gib < 0:
        raise ValueError("C4 minimum free disk budget must be finite and non-negative")
    target.parent.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(target.parent).free / 2**30
    retained = _tree_bytes(target) / 2**30
    if target.exists():
        if free < RESUME_MIN_FREE_GIB or free + retained < minimum_free_gib:
            raise RuntimeError(
                "C4 resume lacks its frozen disk budget: "
                f"free={free:.2f} GiB, current_run={retained:.2f} GiB, required={minimum_free_gib:.2f} GiB"
            )
    elif free < minimum_free_gib:
        raise RuntimeError(f"C4 requires {minimum_free_gib:.2f} GiB free; found {free:.2f} GiB")


def _dataset_tsv(raw_inputs: Mapping[str, Mapping[str, Any]]) -> str:
    fields = ("dataset", "split", "case_id", "path", "bytes", "sha256", "mtime_utc")
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields, delimiter="\t", lineterminator="\n")
    writer.writeheader()
    for record in raw_inputs.values():
        writer.writerow({field: record.get(field, "") for field in fields})
    return stream.getvalue()


def _frozen_payloads() -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    policy = C4_POLICY.to_dict()
    arms = arm_contract_rows()
    offsets = canonical_offset_table()
    support = support_contract()
    observed = {
        "policy": payload_sha256(policy),
        "arms": payload_sha256(arms),
        "offsets": payload_sha256(offsets),
        "support": payload_sha256(support),
    }
    expected = {
        "policy": C4_POLICY_SHA256,
        "arms": ARM_SPECS_SHA256,
        "offsets": OFFSET_TABLE_SHA256,
        "support": SUPPORT_CONTRACT_SHA256,
    }
    if observed != expected:
        raise RuntimeError(f"C4 frozen payload hash mismatch: {observed} != {expected}")
    validate_arm_specs(arms)
    validate_offset_table(offsets)
    validate_support_contract(support)
    return policy, arms, offsets, support


def _load_contract_pair(
    run_root: Path, source_sha256: str, decision_sha256: str
) -> tuple[dict[str, Any], dict[str, Any], str, str]:
    source, source_sha = load_source_contract(run_root, source_sha256)
    decision, decision_sha = load_decision_contract(
        run_root,
        decision_sha256,
        source=source,
        expected_source_sha256=source_sha,
        expected_policy_sha256=C4_POLICY_SHA256,
        expected_arm_specs_sha256=ARM_SPECS_SHA256,
        expected_offset_table_sha256=OFFSET_TABLE_SHA256,
        expected_support_contract_sha256=SUPPORT_CONTRACT_SHA256,
    )
    return source, decision, source_sha, decision_sha


def selfcheck_stage(args: argparse.Namespace) -> int:
    assert_frozen_policy()
    policy, arms, offsets, support = _frozen_payloads()
    checks = {
        "protocol_is_c4_v1": PROTOCOL_ID == "CTCF-SEARCH-GATE-C4-V1",
        "eight_selectable_and_four_diagnostic_arms": len(arms) == 12
        and sum(bool(row["selectable"]) for row in arms) == 8,
        "two_explicit_27_offset_reaches": len(offsets) == 2 and all(len(row["offsets_zyx"]) == 27 for row in offsets),
        "common_collar_is_7": support["collar_width"] == 7,
        "label_free_utility_is_common_ncc7": support["utility_id"] == "COMMON_NCC7" and support["window"] == 7,
        "policy_hash_is_canonical": payload_sha256(policy) == C4_POLICY_SHA256,
        "test_115_is_not_authorized": policy["proposal_pipeline"][1] == ["test_115_authorized", False],
    }
    failed = [name for name, passed in checks.items() if not passed]
    payload = {
        "schema": "ctcf-search-c4-selfcheck-v1",
        "protocol_id": PROTOCOL_ID,
        "status": "PASS" if not failed else "FAIL",
        "checks": checks,
        "failed": failed,
        "hashes": {
            "policy": C4_POLICY_SHA256,
            "arm_specs": ARM_SPECS_SHA256,
            "offset_table": OFFSET_TABLE_SHA256,
            "support_contract": SUPPORT_CONTRACT_SHA256,
        },
    }
    atomic_write_json(args.output, payload)
    if failed:
        raise RuntimeError(f"C4 self-check failed: {failed}")
    print(json.dumps(payload, indent=2))
    return 0


def prepare_stage(args: argparse.Namespace) -> int:
    if git("status", "--porcelain=v1"):
        raise RuntimeError("C4 prepare requires a clean tracked and untracked tree")
    head = git("rev-parse", "HEAD")
    physical_gpus = parse_physical_gpus(
        args.physical_gpus,
        args.num_shards,
        "C4 requires exactly one unique non-negative physical GPU per shard",
    )
    run_root = args.run_root.resolve()
    heavy_root = args.heavy_root.resolve()
    if run_root == heavy_root or run_root in heavy_root.parents or heavy_root in run_root.parents:
        raise ValueError("C4 compact and heavy roots must not overlap")
    _validate_disk_budget(heavy_root, args.min_free_gib)
    policy, arms, offsets, support = _frozen_payloads()
    bundle = prepare_contracts(
        run_root=run_root,
        source_c3_dir=args.source_c3_dir,
        source_c3_heavy_root=args.source_c3_heavy_root,
        target_heavy_root=heavy_root,
        git_head=head,
        runtime_signature=_runtime_signature(),
        physical_gpus=physical_gpus,
        policy=policy,
        expected_policy_sha256=C4_POLICY_SHA256,
        arm_specs=arms,
        expected_arm_specs_sha256=ARM_SPECS_SHA256,
        offset_table=offsets,
        expected_offset_table_sha256=OFFSET_TABLE_SHA256,
        support_contract=support,
        expected_support_contract_sha256=SUPPORT_CONTRACT_SHA256,
    )
    if bundle.source["seed"] != EXPECTED_SOURCE_SEED or bundle.decision["seed"] != EXPECTED_SOURCE_SEED:
        raise RuntimeError(
            "C4 frozen C3 source seed changed: "
            f"source={bundle.source['seed']}, decision={bundle.decision['seed']}, "
            f"expected={EXPECTED_SOURCE_SEED}"
        )
    atomic_write_text(run_root / "datasets.tsv", _dataset_tsv(bundle.source["raw_inputs"]))
    print(
        json.dumps(
            {
                "source_contract_sha256": bundle.source_sha256,
                "decision_contract_sha256": bundle.decision_sha256,
                "n_cases": len(bundle.source["case_ids"]),
            }
        )
    )
    return 0


def decision_worker_stage(args: argparse.Namespace) -> int:
    decision, decision_sha = load_decision_contract_isolated(
        args.run_root,
        args.decision_contract_sha256,
        expected_source_sha256=args.source_contract_sha256,
        expected_policy_sha256=C4_POLICY_SHA256,
        expected_arm_specs_sha256=ARM_SPECS_SHA256,
        expected_offset_table_sha256=OFFSET_TABLE_SHA256,
        expected_support_contract_sha256=SUPPORT_CONTRACT_SHA256,
    )
    _assert_clean_code(decision["git_head"], "decision worker")
    _assert_runtime(decision["runtime_signature"], "decision worker")
    if args.num_shards != decision["num_shards"] or not 0 <= args.shard_index < args.num_shards:
        raise RuntimeError("C4 decision worker shard parameters changed")
    if args.physical_gpu != decision["shard_to_physical_gpu"][str(args.shard_index)]:
        raise RuntimeError("C4 decision worker physical GPU changed")
    assigned = decision["shards"][str(args.shard_index)]
    device = setup_device(args.gpu, seed=decision["seed"], deterministic=True)
    if device.type != "cuda":
        raise RuntimeError("C4 decision worker requires CUDA")
    execution = {
        "phase": "decision",
        "attempt_id": args.attempt_id,
        "shard_index": args.shard_index,
        "physical_gpu": args.physical_gpu,
        "host": platform.node(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(device),
        "seed": decision["seed"],
        "deterministic": True,
        "labels_loaded_to_device": False,
    }
    marker = run_decision_worker(
        case_ids=assigned,
        shard_index=args.shard_index,
        physical_gpu=args.physical_gpu,
        attempt_id=args.attempt_id,
        run_root=args.run_root,
        decision=decision,
        decision_sha256=decision_sha,
        device=device,
        execution=execution,
    )
    print(f"[C4 DECISION WORKER COMPLETE] {marker}")
    return 0


def decision_barrier_stage(args: argparse.Namespace) -> int:
    _, decision, _, decision_sha = _load_contract_pair(
        args.run_root, args.source_contract_sha256, args.decision_contract_sha256
    )
    _assert_clean_code(decision["git_head"], "decision barrier")
    _assert_runtime(decision["runtime_signature"], "decision barrier")
    barrier_path = args.run_root.resolve() / "decision_barrier.json"
    if barrier_path.is_file():
        barrier, digest = load_decision_barrier(
            args.run_root,
            sha256_file(barrier_path),
            decision_contract_sha256=decision_sha,
            case_ids=decision["case_ids"],
        )
        for case_id, expected in barrier["decision_case_sha256"].items():
            observed = sha256_file(args.run_root.resolve() / "cases" / case_id / "decision_complete.json")
            if observed != expected:
                raise RuntimeError(f"C4 frozen decision case changed before barrier reuse: {case_id}")
        print(f"[C4 DECISION BARRIER REUSED] {digest}")
        return 0
    worker_paths = [
        args.run_root.resolve() / "workers" / "decision" / "attempts" / args.attempt_id / f"worker_{index:02d}.json"
        for index in range(decision["num_shards"])
    ]
    barrier = build_decision_barrier(
        args.run_root,
        decision,
        decision_sha,
        attempt_id=args.attempt_id,
        worker_paths=worker_paths,
        verify_heavy_bytes=True,
        completed_at_utc=utc_now(),
    )
    digest = write_decision_barrier(args.run_root, barrier)
    print(f"[C4 DECISION BARRIER] {digest}")
    return 0


def evaluation_worker_stage(args: argparse.Namespace) -> int:
    source, decision, _, decision_sha = _load_contract_pair(
        args.run_root, args.source_contract_sha256, args.decision_contract_sha256
    )
    _assert_clean_code(decision["git_head"], "evaluation worker")
    _assert_runtime(decision["runtime_signature"], "evaluation worker")
    barrier, barrier_sha = load_decision_barrier(
        args.run_root,
        args.barrier_sha256,
        decision_contract_sha256=decision_sha,
        case_ids=decision["case_ids"],
    )
    if args.num_shards != decision["num_shards"] or not 0 <= args.shard_index < args.num_shards:
        raise RuntimeError("C4 evaluation worker shard parameters changed")
    if args.physical_gpu != decision["shard_to_physical_gpu"][str(args.shard_index)]:
        raise RuntimeError("C4 evaluation worker physical GPU changed")
    assigned = decision["shards"][str(args.shard_index)]
    for case_id in ("atlas", *assigned):
        _verify_file_record(source["raw_inputs"][case_id])
    from experiments.core.inference_metrics import metric_profile_for
    from experiments.core.inference_runtime import build_infer_dataset

    device = setup_device(args.gpu, seed=decision["seed"], deterministic=True)
    if device.type != "cuda":
        raise RuntimeError("C4 evaluation worker requires CUDA")
    dataset = build_infer_dataset(
        "IXI",
        [source["raw_inputs"][case_id]["path"] for case_id in assigned],
        source["raw_inputs"]["atlas"]["path"],
    )
    index_by_case = {case_id: index for index, case_id in enumerate(assigned)}
    labels = tuple(metric_profile_for("IXI").labels)
    execution = {
        "phase": "evaluation",
        "attempt_id": args.attempt_id,
        "shard_index": args.shard_index,
        "physical_gpu": args.physical_gpu,
        "host": platform.node(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(device),
        "seed": decision["seed"],
        "deterministic": True,
        "labels_loaded_after_barrier": True,
    }
    marker = run_evaluation_worker(
        case_ids=assigned,
        dataset_item_for_case=lambda case_id: dataset[index_by_case[case_id]],
        labels=labels,
        shard_index=args.shard_index,
        physical_gpu=args.physical_gpu,
        attempt_id=args.attempt_id,
        run_root=args.run_root,
        source=source,
        decision=decision,
        decision_sha256=decision_sha,
        barrier=barrier,
        barrier_sha256=barrier_sha,
        device=device,
        execution=execution,
    )
    print(f"[C4 EVALUATION WORKER COMPLETE] {marker}")
    return 0


def finalize_stage(args: argparse.Namespace) -> int:
    source, decision, _, decision_sha = _load_contract_pair(
        args.run_root, args.source_contract_sha256, args.decision_contract_sha256
    )
    _assert_clean_code(decision["git_head"], "finalization")
    _assert_runtime(decision["runtime_signature"], "finalization")
    barrier, barrier_sha = load_decision_barrier(
        args.run_root,
        args.barrier_sha256,
        decision_contract_sha256=decision_sha,
        case_ids=decision["case_ids"],
    )
    for shard_index in range(decision["num_shards"]):
        path = (
            args.run_root.resolve()
            / "workers"
            / "evaluation"
            / "attempts"
            / args.attempt_id
            / f"worker_{shard_index:02d}.json"
        )
        payload = json.loads(path.read_text(encoding="utf-8"))
        validate_worker_marker(
            payload,
            decision,
            decision_sha,
            phase="evaluation",
            shard_index=shard_index,
            attempt_id=args.attempt_id,
            barrier_sha256=barrier_sha,
        )
    artifacts = finalize_c4(
        run_root=args.run_root,
        source=source,
        decision=decision,
        decision_sha256=decision_sha,
        barrier=barrier,
        barrier_sha256=barrier_sha,
    )
    print(json.dumps({"status": "COMPLETE", "artifacts": artifacts}, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run frozen factorized C4 descriptor-scale and search-reach gate.")
    subparsers = parser.add_subparsers(dest="action", required=True)
    selfcheck = subparsers.add_parser("selfcheck")
    selfcheck.add_argument("--output", type=Path, required=True)

    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--run-root", type=Path, required=True)
    prepare.add_argument("--heavy-root", type=Path, required=True)
    prepare.add_argument("--source-c3-dir", type=Path, required=True)
    prepare.add_argument("--source-c3-heavy-root", type=Path, required=True)
    prepare.add_argument("--num-shards", type=int, required=True)
    prepare.add_argument("--physical-gpus", required=True)
    prepare.add_argument("--min-free-gib", type=float, default=80.0)

    for action in ("decision-worker", "evaluation-worker"):
        worker = subparsers.add_parser(action)
        worker.add_argument("--run-root", type=Path, required=True)
        worker.add_argument("--source-contract-sha256", required=True)
        worker.add_argument("--decision-contract-sha256", required=True)
        worker.add_argument("--shard-index", type=int, required=True)
        worker.add_argument("--num-shards", type=int, required=True)
        worker.add_argument("--gpu", type=int, default=0)
        worker.add_argument("--physical-gpu", required=True)
        worker.add_argument("--attempt-id", required=True)
        if action == "evaluation-worker":
            worker.add_argument("--barrier-sha256", required=True)

    barrier = subparsers.add_parser("decision-barrier")
    barrier.add_argument("--run-root", type=Path, required=True)
    barrier.add_argument("--source-contract-sha256", required=True)
    barrier.add_argument("--decision-contract-sha256", required=True)
    barrier.add_argument("--attempt-id", required=True)

    finalize = subparsers.add_parser("finalize")
    finalize.add_argument("--run-root", type=Path, required=True)
    finalize.add_argument("--source-contract-sha256", required=True)
    finalize.add_argument("--decision-contract-sha256", required=True)
    finalize.add_argument("--barrier-sha256", required=True)
    finalize.add_argument("--attempt-id", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    actions = {
        "selfcheck": selfcheck_stage,
        "prepare": prepare_stage,
        "decision-worker": decision_worker_stage,
        "decision-barrier": decision_barrier_stage,
        "evaluation-worker": evaluation_worker_stage,
        "finalize": finalize_stage,
    }
    return actions[args.action](args)


if __name__ == "__main__":
    raise SystemExit(main())
