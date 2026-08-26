from __future__ import annotations

import argparse
import copy
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
from tools.analysis.search_gate_c5 import (
    ALL_CONTRAST_SPECS,
    ARM_SPECS,
    BOOTSTRAP_CONFIDENCE,
    BOOTSTRAP_FAMILY_SIZES,
    BOOTSTRAP_METHOD_ID,
    BOOTSTRAP_QUANTILE_METHOD,
    BOOTSTRAP_RESAMPLES,
    BOOTSTRAP_SEED,
    C5_DECISION_POLICY_SHA256,
    C5_POLICY,
    C5_POLICY_SHA256,
    INFERENCE_FAMILY_IDS,
    PROTOCOL_ID,
    REACH_SPECS,
    SELECTOR_SPECS,
    TEST_115_AUTHORIZED,
    assert_frozen_decision_policy,
    assert_frozen_policy,
    decision_policy_contract,
)
from tools.analysis.search_gate_c5_contracts import (
    EVALUATION_TRANSITION_POST_BARRIER_RECOVERY,
    EVALUATION_TRANSITION_SAME_COMMIT,
    EXPECTED_SUPPORT_CONTRACT,
    SOURCE_C4_MANIFEST_SHA256,
    SOURCE_C4_RUN_ID,
    SOURCE_C4_RUN_MANIFEST_SHA256,
    build_decision_barrier,
    build_evaluation_contract,
    load_decision_barrier,
    load_decision_contract,
    load_decision_contract_isolated,
    load_evaluation_contract,
    load_source_contract,
    payload_sha256,
    prepare_contracts,
    validate_worker_marker,
    write_decision_barrier,
    write_evaluation_contract,
)
from tools.analysis.search_gate_c5_workers import (
    finalize_c5,
    run_decision_case,
    run_decision_worker,
    run_evaluation_worker,
)
from tools.analysis.search_gate_common import git, utc_now
from tools.analysis.search_gate_multiscale import offsets_for_stride
from tools.analysis.search_gate_runtime import parse_physical_gpus
from utils import setup_device

ARM_SPECS_SHA256 = "f51aec615420da160bf4664d02d86ed2f8399edb41283218a86ae2dc4aebb1f2"
SELECTOR_SPECS_SHA256 = "338851b844aa00f7a9e42e69b7aa9c001dc75935d65ef0373cbc5c64f24bc8a8"
OFFSET_TABLE_SHA256 = "ea02d92bc9de673e2146c9c22f735832dfeef512bfb27738180e05deef5f264b"
SUPPORT_CONTRACT_SHA256 = "f8d86720fb5523390bd739b947a4a762509cf86e93da35401607569ac3c01392"
CONTRAST_CONTRACT_SHA256 = "2fcf37fe0d6a8cab73905a245d7b464c55ff0f10b39b37e1337c8ea4cc4df065"
DEFAULT_MIN_FREE_GIB = 180.0
RESUME_MIN_FREE_GIB = 5.0
EXPECTED_SOURCE_SEED = 0


def arm_contract_rows() -> list[dict[str, Any]]:
    return [asdict(spec) for spec in ARM_SPECS]


def selector_contract_rows() -> list[dict[str, Any]]:
    return [asdict(spec) for spec in SELECTOR_SPECS]


def canonical_offset_table() -> list[dict[str, Any]]:
    return [
        {
            "reach_index": index,
            "reach_id": spec.reach_id,
            "stride_voxels": spec.stride_voxels,
            "offsets_zyx": [list(offset) for offset in offsets_for_stride(spec.stride_voxels)],
            "pre_rms_multiplier": spec.pre_rms_multiplier,
        }
        for index, spec in enumerate(REACH_SPECS)
    ]


def support_contract() -> dict[str, Any]:
    return copy.deepcopy(EXPECTED_SUPPORT_CONTRACT)


def contrast_contract() -> dict[str, Any]:
    return {
        "schema": "ctcf-search-c5-contrast-contract-v1",
        "all_contrasts": [asdict(spec) for spec in ALL_CONTRAST_SPECS],
        "family_ids": list(INFERENCE_FAMILY_IDS),
        "family_sizes": {family_id: int(BOOTSTRAP_FAMILY_SIZES[family_id]) for family_id in INFERENCE_FAMILY_IDS},
        "bootstrap": {
            "method_id": BOOTSTRAP_METHOD_ID,
            "resamples": BOOTSTRAP_RESAMPLES,
            "seed": BOOTSTRAP_SEED,
            "confidence": BOOTSTRAP_CONFIDENCE,
            "quantile_method": BOOTSTRAP_QUANTILE_METHOD,
            "unit": "paired_IXI_validation_subject",
            "simultaneous_scope": "within_each_declared_family",
        },
        "test_115_authorized": False,
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
        raise RuntimeError(f"C5 {stage} code differs from its clean prepared contract")


def _assert_runtime(expected: Mapping[str, Any], stage: str) -> None:
    observed = _runtime_signature()
    if observed != dict(expected):
        raise RuntimeError(f"C5 {stage} runtime changed: {observed} != {dict(expected)}")


def _verify_file_record(record: Mapping[str, Any]) -> None:
    path = Path(str(record.get("path", ""))).resolve()
    if (
        not path.is_file()
        or path.stat().st_size != int(record.get("bytes", -1))
        or sha256_file(path) != record.get("sha256")
    ):
        raise RuntimeError(f"C5 frozen raw input changed or is missing: {path}")


def _tree_bytes(root: Path) -> int:
    if not root.exists():
        return 0
    return sum(path.stat().st_size for path in root.rglob("*") if path.is_file())


def _validate_disk_budget(target: Path, minimum_free_gib: float) -> None:
    if not math.isfinite(minimum_free_gib) or minimum_free_gib < 0:
        raise ValueError("C5 minimum free disk budget must be finite and non-negative")
    target.parent.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(target.parent).free / 2**30
    retained = _tree_bytes(target) / 2**30
    if target.exists():
        if free < RESUME_MIN_FREE_GIB or free + retained < minimum_free_gib:
            raise RuntimeError(
                "C5 resume lacks its frozen disk budget: "
                f"free={free:.2f} GiB, current_run={retained:.2f} GiB, required={minimum_free_gib:.2f} GiB"
            )
    elif free < minimum_free_gib:
        raise RuntimeError(f"C5 requires {minimum_free_gib:.2f} GiB free; found {free:.2f} GiB")


def _immutable_text(path: Path, content: str) -> None:
    if path.exists():
        if path.read_text(encoding="utf-8") != content:
            raise FileExistsError(f"Refusing to replace immutable C5 artifact: {path}")
        return
    atomic_write_text(path, content)


def _dataset_tsv(raw_inputs: Mapping[str, Mapping[str, Any]]) -> str:
    fields = ("dataset", "split", "case_id", "path", "bytes", "sha256", "mtime_utc")
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields, delimiter="\t", lineterminator="\n")
    writer.writeheader()
    for record in raw_inputs.values():
        writer.writerow({field: record.get(field, "") for field in fields})
    return stream.getvalue()


def _retention_text(source: Mapping[str, Any]) -> str:
    roots = source["roots"]
    return "".join(
        (
            f"source_c3_heavy_root={roots['source_c3_heavy']}\n",
            f"source_c4_heavy_root={roots['source_c4_heavy']}\n",
            f"target_c5_heavy_root={roots['target_c5_heavy']}\n",
            "retention=RETAIN_ALL_THREE_ON_H100_UNTIL_EXPLICIT_OPERATOR_DECISION\n",
            "packaged=false\n",
        )
    )


def _frozen_payloads() -> dict[str, Any]:
    payloads = {
        "full_policy": C5_POLICY.to_dict(),
        "decision_policy": decision_policy_contract(),
        "arms": arm_contract_rows(),
        "selectors": selector_contract_rows(),
        "offsets": canonical_offset_table(),
        "support": support_contract(),
        "contrasts": contrast_contract(),
    }
    observed = {name: payload_sha256(value) for name, value in payloads.items()}
    expected = {
        "full_policy": C5_POLICY_SHA256,
        "decision_policy": C5_DECISION_POLICY_SHA256,
        "arms": ARM_SPECS_SHA256,
        "selectors": SELECTOR_SPECS_SHA256,
        "offsets": OFFSET_TABLE_SHA256,
        "support": SUPPORT_CONTRACT_SHA256,
        "contrasts": CONTRAST_CONTRACT_SHA256,
    }
    if observed != expected:
        raise RuntimeError(f"C5 frozen payload hash mismatch: {observed} != {expected}")
    if len(payloads["arms"]) != 36 or len(payloads["selectors"]) != 5 or len(payloads["offsets"]) != 4:
        raise RuntimeError("C5 frozen factorial inventory changed")
    if any(len(row["offsets_zyx"]) != 27 for row in payloads["offsets"]):
        raise RuntimeError("C5 reach table must contain 27 sparse offsets at every stride")
    if payloads["support"] != EXPECTED_SUPPORT_CONTRACT:
        raise RuntimeError("C5 support contract changed")
    return payloads


def _load_contract_pair(
    run_root: Path,
    source_sha256: str,
    decision_sha256: str,
) -> tuple[dict[str, Any], dict[str, Any], str, str]:
    source, source_sha = load_source_contract(run_root, source_sha256)
    decision, decision_sha = load_decision_contract(
        run_root,
        decision_sha256,
        source=source,
        expected_source_sha256=source_sha,
        expected_decision_policy_sha256=C5_DECISION_POLICY_SHA256,
        expected_arm_specs_sha256=ARM_SPECS_SHA256,
        expected_selector_specs_sha256=SELECTOR_SPECS_SHA256,
        expected_offset_table_sha256=OFFSET_TABLE_SHA256,
        expected_support_contract_sha256=SUPPORT_CONTRACT_SHA256,
        expected_contrast_contract_sha256=CONTRAST_CONTRACT_SHA256,
    )
    return source, decision, source_sha, decision_sha


def _load_isolated_decision(
    run_root: Path,
    source_sha256: str,
    decision_sha256: str,
) -> tuple[dict[str, Any], str]:
    return load_decision_contract_isolated(
        run_root,
        decision_sha256,
        expected_source_sha256=source_sha256,
        expected_decision_policy_sha256=C5_DECISION_POLICY_SHA256,
        expected_arm_specs_sha256=ARM_SPECS_SHA256,
        expected_selector_specs_sha256=SELECTOR_SPECS_SHA256,
        expected_offset_table_sha256=OFFSET_TABLE_SHA256,
        expected_support_contract_sha256=SUPPORT_CONTRACT_SHA256,
        expected_contrast_contract_sha256=CONTRAST_CONTRACT_SHA256,
    )


def _load_evaluation_bundle(
    args: argparse.Namespace,
) -> tuple[dict[str, Any], dict[str, Any], str, str, dict[str, Any], str, dict[str, Any], str]:
    isolated, isolated_sha = _load_isolated_decision(
        args.run_root,
        args.source_contract_sha256,
        args.decision_contract_sha256,
    )
    barrier, barrier_sha = load_decision_barrier(
        args.run_root,
        args.barrier_sha256,
        contract=isolated,
        decision_contract_sha256=isolated_sha,
        case_ids=isolated["case_ids"],
    )
    source, decision, source_sha, decision_sha = _load_contract_pair(
        args.run_root, args.source_contract_sha256, args.decision_contract_sha256
    )
    if decision_sha != isolated_sha or decision != isolated:
        raise RuntimeError("C5 full decision contract differs from its pre-barrier isolated load")
    evaluation, evaluation_sha = load_evaluation_contract(
        args.run_root,
        args.evaluation_contract_sha256,
        source=source,
        barrier=barrier,
        expected_source_sha256=source_sha,
        expected_decision_sha256=decision_sha,
        expected_barrier_sha256=barrier_sha,
    )
    return source, decision, source_sha, decision_sha, barrier, barrier_sha, evaluation, evaluation_sha


def _evaluation_code_transition(
    decision: Mapping[str, Any],
    *,
    resume_after_barrier: bool,
) -> tuple[str, dict[str, Any], str]:
    if resume_after_barrier:
        evaluation_git_head = git("rev-parse", "HEAD")
        _assert_clean_code(evaluation_git_head, "post-barrier recovery")
    else:
        evaluation_git_head = decision["git_head"]
        _assert_clean_code(evaluation_git_head, "evaluation")
    _assert_runtime(decision["runtime_signature"], "evaluation")
    transition = (
        EVALUATION_TRANSITION_SAME_COMMIT
        if evaluation_git_head == decision["git_head"]
        else EVALUATION_TRANSITION_POST_BARRIER_RECOVERY
    )
    if transition == EVALUATION_TRANSITION_POST_BARRIER_RECOVERY and not resume_after_barrier:
        raise RuntimeError("C5 evaluation code changed without an explicit post-barrier recovery")
    return evaluation_git_head, _runtime_signature(), transition


def selfcheck_stage(args: argparse.Namespace) -> int:
    assert_frozen_policy()
    assert_frozen_decision_policy()
    payloads = _frozen_payloads()
    contrasts = payloads["contrasts"]
    checks = {
        "protocol_is_c5_v1": PROTOCOL_ID == "CTCF-SEARCH-GATE-C5-V1",
        "factorial_has_36_unique_arms": len({row["arm_id"] for row in payloads["arms"]}) == 36,
        "five_global_selectors": len({row["selector_id"] for row in payloads["selectors"]}) == 5,
        "four_sparse_27_offset_reaches": len(payloads["offsets"]) == 4
        and all(len(row["offsets_zyx"]) == 27 for row in payloads["offsets"]),
        "support_is_exact_public_owner": payloads["support"] == EXPECTED_SUPPORT_CONTRACT,
        "eight_separate_inference_families": contrasts["family_ids"] == list(INFERENCE_FAMILY_IDS)
        and len(contrasts["family_ids"]) == 8,
        "exact_c4_source_is_pinned": SOURCE_C4_RUN_ID == "C4_DEVELOPMENT_20260824T161239Z_c69d12000176"
        and len(SOURCE_C4_MANIFEST_SHA256) == 64
        and len(SOURCE_C4_RUN_MANIFEST_SHA256) == 64,
        "all_payload_hashes_are_canonical": all(
            payload_sha256(payloads[name]) == digest
            for name, digest in {
                "full_policy": C5_POLICY_SHA256,
                "decision_policy": C5_DECISION_POLICY_SHA256,
                "arms": ARM_SPECS_SHA256,
                "selectors": SELECTOR_SPECS_SHA256,
                "offsets": OFFSET_TABLE_SHA256,
                "support": SUPPORT_CONTRACT_SHA256,
                "contrasts": CONTRAST_CONTRACT_SHA256,
            }.items()
        ),
        "test_115_is_not_authorized": TEST_115_AUTHORIZED is False
        and payloads["full_policy"]["test_115_authorized"] is False
        and contrasts["test_115_authorized"] is False,
    }
    failed = [name for name, passed in checks.items() if not passed]
    report = {
        "schema": "ctcf-search-c5-selfcheck-v1",
        "protocol_id": PROTOCOL_ID,
        "status": "PASS" if not failed else "FAIL",
        "checks": checks,
        "failed": failed,
        "hashes": {
            "full_policy": C5_POLICY_SHA256,
            "decision_policy": C5_DECISION_POLICY_SHA256,
            "arm_specs": ARM_SPECS_SHA256,
            "selector_specs": SELECTOR_SPECS_SHA256,
            "offset_table": OFFSET_TABLE_SHA256,
            "support_contract": SUPPORT_CONTRACT_SHA256,
            "contrast_contract": CONTRAST_CONTRACT_SHA256,
            "source_c4_manifest": SOURCE_C4_MANIFEST_SHA256,
            "source_c4_run_manifest": SOURCE_C4_RUN_MANIFEST_SHA256,
        },
    }
    atomic_write_json(args.output, report)
    if failed:
        raise RuntimeError(f"C5 self-check failed: {failed}")
    print(json.dumps(report, indent=2))
    return 0


def prepare_stage(args: argparse.Namespace) -> int:
    if git("status", "--porcelain=v1"):
        raise RuntimeError("C5 prepare requires a clean tracked and untracked tree")
    head = git("rev-parse", "HEAD")
    physical_gpus = parse_physical_gpus(
        args.physical_gpus,
        args.num_shards,
        "C5 requires exactly one unique non-negative physical GPU per shard",
    )
    run_root = args.run_root.resolve()
    heavy_root = args.heavy_root.resolve()
    if run_root == heavy_root or run_root in heavy_root.parents or heavy_root in run_root.parents:
        raise ValueError("C5 compact and heavy roots must not overlap")
    _validate_disk_budget(heavy_root, args.min_free_gib)
    payloads = _frozen_payloads()
    bundle = prepare_contracts(
        run_root=run_root,
        source_c4_dir=args.source_c4_dir,
        source_c4_heavy_root=args.source_c4_heavy_root,
        target_heavy_root=heavy_root,
        git_head=head,
        runtime_signature=_runtime_signature(),
        physical_gpus=physical_gpus,
        full_policy=payloads["full_policy"],
        expected_full_policy_sha256=C5_POLICY_SHA256,
        decision_policy=payloads["decision_policy"],
        expected_decision_policy_sha256=C5_DECISION_POLICY_SHA256,
        arm_specs=payloads["arms"],
        expected_arm_specs_sha256=ARM_SPECS_SHA256,
        selector_specs=payloads["selectors"],
        expected_selector_specs_sha256=SELECTOR_SPECS_SHA256,
        offset_table=payloads["offsets"],
        expected_offset_table_sha256=OFFSET_TABLE_SHA256,
        support_contract=payloads["support"],
        expected_support_contract_sha256=SUPPORT_CONTRACT_SHA256,
        contrast_contract=payloads["contrasts"],
        expected_contrast_contract_sha256=CONTRAST_CONTRACT_SHA256,
        verify_anchor_bytes=True,
    )
    if bundle.source["seed"] != EXPECTED_SOURCE_SEED or bundle.decision["seed"] != EXPECTED_SOURCE_SEED:
        raise RuntimeError(
            "C5 frozen C4 source seed changed: "
            f"source={bundle.source['seed']}, decision={bundle.decision['seed']}, expected={EXPECTED_SOURCE_SEED}"
        )
    _immutable_text(run_root / "datasets.tsv", _dataset_tsv(bundle.source["raw_inputs"]))
    _immutable_text(run_root / "heavy_retention.txt", _retention_text(bundle.source))
    print(
        json.dumps(
            {
                "source_contract_sha256": bundle.source_sha256,
                "decision_contract_sha256": bundle.decision_sha256,
                "n_cases": len(bundle.source["case_ids"]),
                "new_heavy_field_count": (len(ARM_SPECS) - 2) * len(bundle.source["case_ids"]),
            }
        )
    )
    return 0


def _decision_execution(
    decision: Mapping[str, Any],
    *,
    attempt_id: str,
    shard_index: int,
    physical_gpu: str,
    device: torch.device,
) -> dict[str, Any]:
    return {
        "phase": "decision",
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
        "labels_loaded_to_device": False,
    }


def decision_pilot_stage(args: argparse.Namespace) -> int:
    decision, decision_sha = _load_isolated_decision(
        args.run_root,
        args.source_contract_sha256,
        args.decision_contract_sha256,
    )
    _assert_clean_code(decision["git_head"], "decision pilot")
    _assert_runtime(decision["runtime_signature"], "decision pilot")
    if args.num_shards != decision["num_shards"] or not decision["shards"].get("0"):
        raise RuntimeError("C5 decision pilot requires the complete non-empty shard inventory")
    if args.physical_gpu != decision["shard_to_physical_gpu"]["0"]:
        raise RuntimeError("C5 decision pilot physical GPU changed")
    device = setup_device(args.gpu, seed=decision["seed"], deterministic=True)
    if device.type != "cuda":
        raise RuntimeError("C5 decision pilot requires CUDA")
    case_id = decision["shards"]["0"][0]
    marker = run_decision_case(
        case_id=case_id,
        shard_index=0,
        physical_gpu=args.physical_gpu,
        run_root=args.run_root,
        decision=decision,
        decision_sha256=decision_sha,
        device=device,
        execution=_decision_execution(
            decision,
            attempt_id=args.attempt_id,
            shard_index=0,
            physical_gpu=args.physical_gpu,
            device=device,
        ),
    )
    print(f"[C5 DECISION PILOT COMPLETE] case={case_id} marker={marker}")
    return 0


def decision_worker_stage(args: argparse.Namespace) -> int:
    decision, decision_sha = _load_isolated_decision(
        args.run_root,
        args.source_contract_sha256,
        args.decision_contract_sha256,
    )
    _assert_clean_code(decision["git_head"], "decision worker")
    _assert_runtime(decision["runtime_signature"], "decision worker")
    if args.num_shards != decision["num_shards"] or not 0 <= args.shard_index < args.num_shards:
        raise RuntimeError("C5 decision worker shard parameters changed")
    if args.physical_gpu != decision["shard_to_physical_gpu"][str(args.shard_index)]:
        raise RuntimeError("C5 decision worker physical GPU changed")
    assigned = decision["shards"][str(args.shard_index)]
    device = setup_device(args.gpu, seed=decision["seed"], deterministic=True)
    if device.type != "cuda":
        raise RuntimeError("C5 decision worker requires CUDA")
    execution = _decision_execution(
        decision,
        attempt_id=args.attempt_id,
        shard_index=args.shard_index,
        physical_gpu=args.physical_gpu,
        device=device,
    )
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
    print(f"[C5 DECISION WORKER COMPLETE] {marker}")
    return 0


def decision_barrier_stage(args: argparse.Namespace) -> int:
    decision, decision_sha = _load_isolated_decision(
        args.run_root,
        args.source_contract_sha256,
        args.decision_contract_sha256,
    )
    _assert_clean_code(decision["git_head"], "decision barrier")
    _assert_runtime(decision["runtime_signature"], "decision barrier")
    barrier_path = args.run_root.resolve() / "decision_barrier.json"
    if barrier_path.is_file():
        barrier, digest = load_decision_barrier(
            args.run_root,
            sha256_file(barrier_path),
            contract=decision,
            decision_contract_sha256=decision_sha,
            case_ids=decision["case_ids"],
        )
        for case_id, expected in barrier["decision_case_sha256"].items():
            observed = sha256_file(args.run_root.resolve() / "cases" / case_id / "decision_complete.json")
            if observed != expected:
                raise RuntimeError(f"C5 frozen decision case changed before barrier reuse: {case_id}")
        print(f"[C5 DECISION BARRIER REUSED] {digest}")
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
    print(f"[C5 DECISION BARRIER] {digest}")
    return 0


def freeze_evaluation_stage(args: argparse.Namespace) -> int:
    isolated, isolated_sha = _load_isolated_decision(
        args.run_root,
        args.source_contract_sha256,
        args.decision_contract_sha256,
    )
    resume_after_barrier = bool(getattr(args, "resume_after_barrier", False))
    evaluation_git_head, evaluation_runtime, transition = _evaluation_code_transition(
        isolated,
        resume_after_barrier=resume_after_barrier,
    )
    barrier, barrier_sha = load_decision_barrier(
        args.run_root,
        args.barrier_sha256,
        contract=isolated,
        decision_contract_sha256=isolated_sha,
        case_ids=isolated["case_ids"],
    )
    source, decision, source_sha, decision_sha = _load_contract_pair(
        args.run_root, args.source_contract_sha256, args.decision_contract_sha256
    )
    if decision_sha != isolated_sha or decision != isolated:
        raise RuntimeError("C5 decision contract changed while opening the post-barrier evaluation contract")
    evaluation = build_evaluation_contract(
        source,
        source_sha,
        decision_sha,
        barrier,
        barrier_sha,
        evaluation_git_head=evaluation_git_head,
        evaluation_runtime_signature=evaluation_runtime,
        code_transition=transition,
    )
    digest = write_evaluation_contract(args.run_root, evaluation)
    print(f"[C5 EVALUATION CONTRACT] {digest}")
    return 0


def recovery_preflight_stage(args: argparse.Namespace) -> int:
    if (args.run_root.resolve() / "evaluation_contract.json").exists():
        raise RuntimeError("C5 recovery preflight requires a run stopped before evaluation-contract creation")
    isolated, isolated_sha = _load_isolated_decision(
        args.run_root,
        args.source_contract_sha256,
        args.decision_contract_sha256,
    )
    physical_gpus = parse_physical_gpus(
        args.physical_gpus,
        isolated["num_shards"],
        "C5 recovery requires the frozen physical GPU inventory",
    )
    if physical_gpus != isolated["physical_gpus"]:
        raise RuntimeError("C5 recovery physical GPUs differ from the frozen decision contract")
    evaluation_git_head, evaluation_runtime, transition = _evaluation_code_transition(
        isolated,
        resume_after_barrier=True,
    )
    if transition != EVALUATION_TRANSITION_POST_BARRIER_RECOVERY:
        raise RuntimeError("C5 recovery preflight requires a distinct post-barrier consumer commit")
    barrier, barrier_sha = load_decision_barrier(
        args.run_root,
        args.barrier_sha256,
        contract=isolated,
        decision_contract_sha256=isolated_sha,
        case_ids=isolated["case_ids"],
    )
    source, decision, source_sha, decision_sha = _load_contract_pair(
        args.run_root,
        args.source_contract_sha256,
        args.decision_contract_sha256,
    )
    if decision_sha != isolated_sha or decision != isolated:
        raise RuntimeError("C5 recovery decision contract differs from its isolated preflight load")
    missing_heavy_roots = [
        f"{root_id}={path}" for root_id, value in source["roots"].items() if not (path := Path(value)).is_dir()
    ]
    if missing_heavy_roots:
        raise RuntimeError(f"C5 recovery heavy roots are missing: {missing_heavy_roots}")
    evaluation = build_evaluation_contract(
        source,
        source_sha,
        decision_sha,
        barrier,
        barrier_sha,
        evaluation_git_head=evaluation_git_head,
        evaluation_runtime_signature=evaluation_runtime,
        code_transition=transition,
    )
    print(
        json.dumps(
            {
                "status": "PASS",
                "mode": EVALUATION_TRANSITION_POST_BARRIER_RECOVERY,
                "decision_git_head": decision["git_head"],
                "evaluation_git_head": evaluation["evaluation_code"]["git_head"],
                "decision_barrier_sha256": barrier_sha,
                "cases": len(decision["case_ids"]),
                "physical_gpus": physical_gpus,
                "test_115_authorized": False,
            },
            indent=2,
        )
    )
    return 0


def evaluation_worker_stage(args: argparse.Namespace) -> int:
    _, decision, _, decision_sha, barrier, barrier_sha, evaluation, evaluation_sha = _load_evaluation_bundle(args)
    evaluation_code = evaluation["evaluation_code"]
    _assert_clean_code(evaluation_code["git_head"], "evaluation worker")
    _assert_runtime(evaluation_code["runtime_signature"], "evaluation worker")
    if args.num_shards != decision["num_shards"] or not 0 <= args.shard_index < args.num_shards:
        raise RuntimeError("C5 evaluation worker shard parameters changed")
    if args.physical_gpu != decision["shard_to_physical_gpu"][str(args.shard_index)]:
        raise RuntimeError("C5 evaluation worker physical GPU changed")
    assigned = decision["shards"][str(args.shard_index)]
    for case_id in ("atlas", *assigned):
        _verify_file_record(evaluation["raw_inputs"][case_id])

    from experiments.core.inference_metrics import metric_profile_for
    from experiments.core.inference_runtime import build_infer_dataset

    device = setup_device(args.gpu, seed=decision["seed"], deterministic=True)
    if device.type != "cuda":
        raise RuntimeError("C5 evaluation worker requires CUDA")
    dataset = build_infer_dataset(
        "IXI",
        [evaluation["raw_inputs"][case_id]["path"] for case_id in assigned],
        evaluation["raw_inputs"]["atlas"]["path"],
    )
    index_by_case = {case_id: index for index, case_id in enumerate(assigned)}
    labels = tuple(metric_profile_for("IXI").labels)
    if labels != tuple(evaluation["evaluation_label_ids"]):
        raise RuntimeError("C5 runtime IXI label order differs from the frozen evaluation contract")
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
        "evaluation_contract_sha256": evaluation_sha,
    }
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
        evaluation_contract=evaluation,
        evaluation_contract_sha256=evaluation_sha,
        device=device,
        execution=execution,
    )
    print(f"[C5 EVALUATION WORKER COMPLETE] {marker}")
    return 0


def finalize_stage(args: argparse.Namespace) -> int:
    _, decision, _, decision_sha, barrier, barrier_sha, evaluation, evaluation_sha = _load_evaluation_bundle(args)
    evaluation_code = evaluation["evaluation_code"]
    _assert_clean_code(evaluation_code["git_head"], "finalization")
    _assert_runtime(evaluation_code["runtime_signature"], "finalization")
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
            evaluation_contract_sha256=evaluation_sha,
        )
    artifacts = finalize_c5(
        run_root=args.run_root,
        decision=decision,
        decision_sha256=decision_sha,
        barrier=barrier,
        barrier_sha256=barrier_sha,
        evaluation_contract=evaluation,
        evaluation_contract_sha256=evaluation_sha,
    )
    print(json.dumps({"status": "COMPLETE", "artifacts": artifacts}, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the frozen C5 sparse-intensity factorial search gate.")
    subparsers = parser.add_subparsers(dest="action", required=True)
    selfcheck = subparsers.add_parser("selfcheck")
    selfcheck.add_argument("--output", type=Path, required=True)

    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--run-root", type=Path, required=True)
    prepare.add_argument("--heavy-root", type=Path, required=True)
    prepare.add_argument("--source-c4-dir", type=Path, required=True)
    prepare.add_argument("--source-c4-heavy-root", type=Path, required=True)
    prepare.add_argument("--num-shards", type=int, required=True)
    prepare.add_argument("--physical-gpus", required=True)
    prepare.add_argument("--min-free-gib", type=float, default=DEFAULT_MIN_FREE_GIB)

    pilot = subparsers.add_parser("decision-pilot")
    pilot.add_argument("--run-root", type=Path, required=True)
    pilot.add_argument("--source-contract-sha256", required=True)
    pilot.add_argument("--decision-contract-sha256", required=True)
    pilot.add_argument("--num-shards", type=int, required=True)
    pilot.add_argument("--gpu", type=int, default=0)
    pilot.add_argument("--physical-gpu", required=True)
    pilot.add_argument("--attempt-id", required=True)

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
            worker.add_argument("--evaluation-contract-sha256", required=True)

    barrier = subparsers.add_parser("decision-barrier")
    barrier.add_argument("--run-root", type=Path, required=True)
    barrier.add_argument("--source-contract-sha256", required=True)
    barrier.add_argument("--decision-contract-sha256", required=True)
    barrier.add_argument("--attempt-id", required=True)

    freeze = subparsers.add_parser("freeze-evaluation")
    freeze.add_argument("--run-root", type=Path, required=True)
    freeze.add_argument("--source-contract-sha256", required=True)
    freeze.add_argument("--decision-contract-sha256", required=True)
    freeze.add_argument("--barrier-sha256", required=True)
    freeze.add_argument("--resume-after-barrier", action="store_true")

    recovery = subparsers.add_parser("recovery-preflight")
    recovery.add_argument("--run-root", type=Path, required=True)
    recovery.add_argument("--source-contract-sha256", required=True)
    recovery.add_argument("--decision-contract-sha256", required=True)
    recovery.add_argument("--barrier-sha256", required=True)
    recovery.add_argument("--physical-gpus", required=True)

    finalize = subparsers.add_parser("finalize")
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
        "recovery-preflight": recovery_preflight_stage,
        "evaluation-worker": evaluation_worker_stage,
        "finalize": finalize_stage,
    }
    return actions[args.action](args)


if __name__ == "__main__":
    raise SystemExit(main())
