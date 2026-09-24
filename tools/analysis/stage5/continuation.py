"""Explicit post-training continuation without rewriting frozen training provenance."""

from __future__ import annotations

import subprocess
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from datasets.OASIS100 import Stage5OasisImageStore
from experiments.stage5.runtime import development_case_inventory, validate_certified_source_artifact
from tools.analysis.run_artifacts import sha256_file
from tools.analysis.stage5.artifacts import load_canonical_json
from tools.analysis.stage5.contracts import (
    BASE_SEEDS,
    build_training_barrier,
    canonical_sha256,
    validate_protocol_contract,
    validate_training_barrier,
    write_immutable_json,
)
from tools.analysis.stage5.pipeline import collect_checkpoint_metadata
from tools.analysis.stage5.primitives import require_git_sha

CONTINUATION_SCHEMA = "ctcf-stage5-evaluation-continuation-v1"
POST_TRAINING_ACTIONS = frozenset({"decide", "freeze-decision", "evaluate", "freeze-evaluation", "aggregate"})
ROOT_NAMES = ("checkpoint_root", "source_root", "decision_root", "evaluation_root", "image_root")
FILE_NAMES = ("protocol", "training_barrier", "data_contract")
# Reviewed post-training modules may change; model, training, data and search
# operator modules must remain byte-identical to the training revision. Changes
# within this allowlist still require review; it is not a semantic equivalence proof.
ORCHESTRATION_FILES = frozenset(
    {
        "tools/analysis/run_stage5.py",
        "tools/analysis/stage5/pipeline.py",
        "tools/analysis/stage5/continuation.py",
        "tools/analysis/stage5/controller_observations.py",
        "tools/analysis/stage5/evaluation.py",
    }
)


def verify_code_compatibility(repo_root: Path, training_head: str, execution_head: str) -> list[str]:
    for value in (training_head, execution_head):
        require_git_sha(value, "continuation Git HEAD")
    ancestry = subprocess.run(
        ["git", "-C", str(repo_root), "merge-base", "--is-ancestor", training_head, execution_head],
        capture_output=True,
        text=True,
    )
    if ancestry.returncode == 1:
        raise RuntimeError(f"Continuation HEAD {execution_head} does not descend from training HEAD {training_head}")
    if ancestry.returncode != 0:
        raise RuntimeError(f"Continuation cannot relate the two revisions: {ancestry.stderr.strip()}")
    changed = subprocess.run(
        [
            "git",
            "-C",
            str(repo_root),
            "diff",
            "--name-only",
            training_head,
            execution_head,
            "--",
            "datasets",
            "experiments",
            "models",
            "utils",
            "tools/analysis",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    forbidden = [
        path for path in changed if path not in ORCHESTRATION_FILES and not path.startswith("tools/analysis/tests/")
    ]
    if forbidden:
        raise RuntimeError(f"Continuation changes frozen numerical/data code: {forbidden}")
    return changed


def prepare_continuation(
    *,
    repo_root: Path,
    training_head: str,
    execution_head: str,
    paths: Mapping[str, Path],
    output: Path,
) -> dict[str, Any]:
    changed = verify_code_compatibility(repo_root, training_head, execution_head)
    protocol = load_canonical_json(paths["protocol"])
    validate_protocol_contract(protocol)
    if protocol["git_head"] != training_head:
        raise RuntimeError("Continuation training HEAD differs from the frozen protocol")
    training = load_canonical_json(paths["training_barrier"])
    validate_training_barrier(training, protocol, require_complete=True)
    actual = collect_checkpoint_metadata(protocol_path=paths["protocol"], checkpoint_root=paths["checkpoint_root"])
    if build_training_barrier(protocol, actual) != training:
        raise RuntimeError("Continuation checkpoint bytes or metadata differ from the training barrier")
    print(f"[STAGE5 CONTINUATION] verified {len(actual)} frozen checkpoints", flush=True)
    store = Stage5OasisImageStore(paths["data_contract"], paths["image_root"])
    if store.runtime.contract_sha256 != protocol["data_contract_sha256"]:
        raise RuntimeError("Continuation data differs from the frozen protocol")
    cases = development_case_inventory(store)
    if [case["case_id"] for case in cases] != protocol["directed_case_ids"]:
        raise RuntimeError("Continuation cases differ from the frozen protocol")
    u0 = {item["seed"]: item["checkpoint_file"]["sha256"] for item in actual if item["variant_id"] == "U0"}
    sources = []
    for seed in BASE_SEEDS:
        for case in cases:
            sources.append(
                validate_certified_source_artifact(
                    paths["source_root"] / f"seed_{seed}" / case["case_id"],
                    seed=seed,
                    case=case,
                    u0_checkpoint_sha256=u0[seed],
                    image_shape=store.image_shape,
                    bootstrap_policy=protocol["bootstrap"]["policy"],
                )
            )
            if len(sources) % 10 == 0:
                print(
                    f"[STAGE5 CONTINUATION] verified sources {len(sources)}/{len(cases) * len(BASE_SEEDS)}", flush=True
                )
    payload = {
        "schema": CONTINUATION_SCHEMA,
        "status": "VERIFIED",
        "training_git_head": training_head,
        "execution_git_head": execution_head,
        "changed_code_paths": changed,
        "paths": {name: str(paths[name].resolve()) for name in (*FILE_NAMES, *ROOT_NAMES)},
        "file_sha256": {name: sha256_file(paths[name]) for name in FILE_NAMES},
        "protocol_sha256": canonical_sha256(protocol),
        "training_barrier_sha256": canonical_sha256(training),
        "source_inventory": sources,
        "labels_loaded": False,
        "training_performed": False,
    }
    write_immutable_json(output, payload)
    return payload


def validate_continuation(
    path: Path,
    *,
    execution_head: str,
    protocol_path: Path,
    supplied_paths: Mapping[str, Path] | None = None,
) -> dict[str, Any]:
    """Check the immutable binding at every downstream CLI boundary.

    Checkpoints/sources are authenticated in prepare_continuation and again by
    their consumers; this check does not repeatedly hash the full heavy tree.
    """
    payload = load_canonical_json(path)
    if (
        payload.get("schema") != CONTINUATION_SCHEMA
        or payload.get("status") != "VERIFIED"
        or payload.get("execution_git_head") != execution_head
        or payload.get("labels_loaded") is not False
        or payload.get("training_performed") is not False
    ):
        raise RuntimeError("Invalid Stage5 continuation or execution HEAD")
    require_git_sha(payload.get("training_git_head"), "continuation training HEAD")
    protocol = load_canonical_json(protocol_path)
    validate_protocol_contract(protocol)
    if protocol.get("git_head") != payload["training_git_head"] or canonical_sha256(protocol) != payload.get(
        "protocol_sha256"
    ):
        raise RuntimeError("Continuation differs from the frozen protocol")
    bindings = {"protocol": protocol_path, **(supplied_paths or {})}
    for name, value in bindings.items():
        if name in (*FILE_NAMES, *ROOT_NAMES) and str(value.resolve()) != payload["paths"].get(name):
            raise RuntimeError(f"Continuation path changed: {name}")
    for name in FILE_NAMES:
        file_path = Path(payload["paths"][name])
        if sha256_file(file_path) != payload["file_sha256"][name]:
            raise RuntimeError(f"Continuation frozen file changed: {name}")
    training = load_canonical_json(Path(payload["paths"]["training_barrier"]))
    validate_training_barrier(training, protocol, require_complete=True)
    if canonical_sha256(training) != payload["training_barrier_sha256"]:
        raise RuntimeError("Continuation training barrier changed")
    return payload
