"""Authenticate unchanged decisions before reuse across evaluation revisions."""

from __future__ import annotations

import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from tools.analysis.stage5.artifacts import field_record
from tools.analysis.stage5.contracts import build_decision_barrier, canonical_json_bytes, canonical_sha256
from tools.analysis.stage5.work_margin import WORK_MARGIN_POLICY


def decision_digest(record: Mapping[str, Any], exact: Mapping[str, Any]) -> dict[str, str]:
    return {"record_sha256": canonical_sha256(record), "exact_sha256": canonical_sha256(exact)}


def execution_matches(record: Mapping[str, Any], exact: Mapping[str, Any], expected: Mapping[str, Any]) -> bool:
    binding = exact.get("execution", {})
    if all(binding.get(key) == expected.get(key) for key in ("execution_git_head", "continuation_sha256")):
        return True
    approved = expected.get("reusable_decisions", {}).get(record["decision_id"])
    return approved is not None and approved == decision_digest(record, exact)


def verify_reusable_decision(
    record: Mapping[str, Any],
    exact: Mapping[str, Any],
    *,
    roots: Mapping[str, Path],
    protocol: Mapping[str, Any],
    training: Mapping[str, Any],
) -> None:
    """Check journals as strictly as published records, without publishing either."""
    build_decision_barrier(protocol, training, [record])
    decision_id = record["decision_id"]
    source_root = roots["source_field_root"]
    expected_source = source_root / f"seed_{record['seed']}" / record["case_id"] / "initial_psi.npz"
    if field_record("source_field_root", source_root, expected_source) != record["certified_source_field"]:
        raise RuntimeError("Reusable Stage5 decision has another source")
    for name in ("certified_source_field", "requested_field", "candidate_field", "returned_field"):
        saved = record[name]
        root = roots[saved["root_id"]].resolve()
        path = (root / saved["relative_path"]).resolve()
        if not path.is_relative_to(root) or field_record(saved["root_id"], root, path) != saved:
            raise RuntimeError(f"Reusable Stage5 field changed: {name}")
    expected_exact = {
        "root_id": "decision_output_root",
        "relative_path": f"exact_reports/{decision_id}.json",
        "bytes": len(canonical_json_bytes(exact)),
        "sha256": canonical_sha256(exact),
    }
    execution = exact.get("execution", {})
    if (
        record["exact_report"] != expected_exact
        or exact.get("schema") not in {"ctcf-stage5-decision-exact-report-v2", "ctcf-stage5-decision-exact-report-v3"}
        or exact.get("decision_id") != decision_id
        or exact.get("source_field") != record["certified_source_field"]
        or execution.get("training_git_head") != protocol["git_head"]
        or execution.get("protocol_sha256") != canonical_sha256(protocol)
        or execution.get("training_barrier_sha256") != canonical_sha256(training)
        or execution.get("checkpoint_sha256") != record["checkpoint_sha256"]
        or execution.get("labels_loaded") is not False
    ):
        raise RuntimeError("Reusable Stage5 decision provenance differs")
    for stage in ("candidate", "returned"):
        certificate = exact.get(f"{stage}_exact", {})
        certified_key = "candidate_exact_certified" if stage == "candidate" else "returned_certified"
        if (
            certificate.get("sha256") != record[f"{stage}_field"]["array_sha256"]
            or certificate.get("status") != record[f"{stage}_exact_status"]
            or certificate.get("certified") != record[certified_key]
        ):
            raise RuntimeError("Reusable Stage5 decision certificate differs")
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
    if canonical_sha256({"environment": execution, "performance": performance}) != record["execution_sha256"]:
        raise RuntimeError("Reusable Stage5 execution digest differs")
    policy = execution.get("decision_safety_policy")
    if policy == WORK_MARGIN_POLICY:
        return
    if policy is not None:
        raise RuntimeError("Reusable Stage5 decision has an unknown safety policy")
    clip = exact.get("clip_report")
    if record["variant_id"] == "U0" or record["transaction_status"] == "CERTIFIED_DEGRADED_IDENTITY":
        if clip is not None:
            raise RuntimeError("Reusable Stage5 baseline unexpectedly has a clip report")
        return
    nominal = float(WORK_MARGIN_POLICY["nominal_work_epsilon"])
    bound = float((clip or {}).get("current_fast_cert_bound", float("nan")))
    if (
        not isinstance(clip, dict)
        or clip.get("operator") != "CERTIFIED_LOCAL_CLIP"
        or clip.get("work_eps") != nominal
        or not math.isfinite(bound)
        or bound < nominal
    ):
        raise RuntimeError("Legacy Stage5 decision is not on the unchanged nominal margin path")
