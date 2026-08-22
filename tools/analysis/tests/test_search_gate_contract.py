"""Behaviour-preserving contract for the C0/C1/C2 search gates.

Four independent things are pinned: gate ownership, frozen protocol hashes, golden
replay of the historical products, and negative controls over tampered inputs.

The golden replay needs no dataset, checkpoint or GPU: production finalize also
consumes JSON-loaded case markers, so replaying them is faithful. Everything under
`results/` is read-only; tampering happens on a copy in a temporary directory.

Set CTCF_REQUIRE_HISTORICAL_GOLDENS=1 to turn a missing historical product into a
failure instead of a skip.
"""

from __future__ import annotations

import argparse
import ast
import copy
import inspect
import io
import json
import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import torch

import tools.analysis.run_search_gate_c0 as c0
import tools.analysis.run_search_gate_c1 as c1
import tools.analysis.run_search_gate_c2 as c2
import tools.analysis.search_gate_common as common
import tools.analysis.search_gate_runtime as runtime
import tools.analysis.transactional_search as ts
from tools.analysis.run_artifacts import rows_to_csv, sha256_file

REPO_ROOT = Path(__file__).resolve().parents[3]
RESULTS = REPO_ROOT / "results"
BASELINE_COMMIT = "7e8fc5b9b17af3a71c068a4113ace75e17e527d6"

GATE_MODULES = {
    "tools.analysis.run_search_gate_c0": c0,
    "tools.analysis.run_search_gate_c1": c1,
    "tools.analysis.run_search_gate_c2": c2,
}

HISTORICAL_PRODUCTS = {
    "c0": RESULTS / "search_gate_c0/C0_20260820T071655Z_d2a9b06efec6",
    "c1_exploration": RESULTS / "search_gate_c1/C1_EXPLORATION_20260821T121202Z_9275171a67a3/exploration",
    "c1_confirmation": RESULTS / "search_gate_c1/C1_CONFIRMATION_20260821T150103Z_9275171a67a3/confirmation",
    "c1_superseded": RESULTS / "search_gate_c1/C1_EXPLORATION_20260821T085333Z_272beee8c6a4/exploration",
    "c2": RESULTS / "search_gate_c2/C2_DEVELOPMENT_20260821T183655Z_7e8fc5b9b17a",
}
REQUIRE_HISTORICAL = os.environ.get("CTCF_REQUIRE_HISTORICAL_GOLDENS") == "1"

C1_POLICY_SHA256 = "9dd1320776c38dc137346d03f40b173d23f373c6fef00b758e5a5d1ca2f9d9e6"
C2_POLICY_SHA256 = "01a021ee22bdc99e1c8148e94358f89e944c35fa05fefeea7044011fb3a79447"
C1_EXPLORATION_MANIFEST_SHA256 = "2e2084c66d33f165fb3a5995d05f535fb4b500e5d22072a4b811b1d388d94403"
C1_CONFIRMATION_MANIFEST_SHA256 = "7ccf7b13f32c67b6821aeea31d742ade450d26d9f2dde56b286495479255defa"
C2_MANIFEST_SHA256 = "99eebe51e8cc53a2d5156fe427bbf34acbb6ec67f39d509c80baff0427dba286"

C1_PER_CANDIDATE_PREFERRED = [
    "stage",
    "case_id",
    "candidate_id",
    "family",
    "feature",
    "orientation",
    "operator",
    "operator_status",
    "scale",
    "sweeps",
    "coefficient_index",
    "coefficient",
]
C2_STEP_PREFERRED = ["case_id", "trajectory_id", "step", "action", "reason"]
C2_BRANCH_PREFERRED = ["case_id", "trajectory_id", "accepted_steps", "terminal_action"]


def needs_product(name: str):
    """Skip only when the product is absent and required mode is off."""
    return unittest.skipIf(
        not HISTORICAL_PRODUCTS[name].is_dir() and not REQUIRE_HISTORICAL,
        f"historical product '{name}' absent; set CTCF_REQUIRE_HISTORICAL_GOLDENS=1 to fail instead of skip",
    )


def json_artifact(payload: Any) -> str:
    """Reproduce atomic_write_json byte-for-byte."""
    buffer = io.StringIO()
    json.dump(payload, buffer, ensure_ascii=False, indent=2, sort_keys=True)
    buffer.write("\n")
    return buffer.getvalue()


def module_imports(module: Any) -> list[str]:
    tree = ast.parse(Path(inspect.getfile(module)).read_text(encoding="utf-8"))
    names = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            names.append(node.module)
        elif isinstance(node, ast.Import):
            names.extend(alias.name for alias in node.names)
    return names


class GateOwnershipTest(unittest.TestCase):
    def test_no_gate_module_imports_another_gate_module(self) -> None:
        for name, module in GATE_MODULES.items():
            others = {other for other in GATE_MODULES if other != name}
            leaked = sorted(others.intersection(module_imports(module)))
            self.assertEqual([], leaked, f"{name} imports another gate entrypoint: {leaked}")

    def test_no_gate_module_reexports_another_gates_objects(self) -> None:
        for name, module in GATE_MODULES.items():
            others = {other for other in GATE_MODULES if other != name}
            leaked = sorted(
                attribute
                for attribute, value in vars(module).items()
                if (inspect.isfunction(value) or inspect.isclass(value)) and getattr(value, "__module__", "") in others
            )
            self.assertEqual([], leaked, f"{name} re-exports objects owned by another gate: {leaked}")

    def test_gate_entrypoints_never_reference_a_test_split(self) -> None:
        for name, module in GATE_MODULES.items():
            source = Path(inspect.getfile(module)).read_text(encoding="utf-8")
            self.assertNotIn("test_dir", source, f"{name} references a test split directory")


class FrozenProtocolTest(unittest.TestCase):
    def test_policy_payload_hashes_are_frozen(self) -> None:
        self.assertEqual(C1_POLICY_SHA256, common.payload_sha256(c1.CONFIRMATION_POLICY))
        self.assertEqual(C1_POLICY_SHA256, c1.CONFIRMATION_POLICY_SHA256)
        self.assertEqual(C2_POLICY_SHA256, common.payload_sha256(c2.C2_POLICY))
        self.assertEqual(C2_POLICY_SHA256, c2.C2_POLICY_SHA256)

    def test_protocol_identifiers_and_shared_constants(self) -> None:
        self.assertEqual("CTCF-SEARCH-GATE-C0-V1", c0.PROTOCOL_ID)
        self.assertEqual("CTCF-SEARCH-GATE-C1-V1", c1.PROTOCOL_ID)
        self.assertEqual("CTCF-SEARCH-GATE-C2-V1", c2.PROTOCOL_ID)
        self.assertEqual("CTCF-SEARCH-GATE-C1-V1", c2.C1_PROTOCOL_ID)
        self.assertEqual("CTCF-GATE-C0-V1|", common.PROTOCOL_SALT)
        self.assertEqual("CTCF-GATE-C0-V1-SALTED-IXI-VAL-58", common.SPLIT_PROTOCOL_ID)
        self.assertEqual(0.001, common.CLAIM_EPS)
        self.assertEqual(0.0011, common.WORK_EPS)
        self.assertEqual(4, common.COLLAR_WIDTH)
        self.assertEqual(6, common.TIME_STEPS)
        for module in (c0, c1, c2):
            for name in ("CLAIM_EPS", "COLLAR_WIDTH", "TIME_STEPS"):
                if hasattr(module, name):
                    self.assertIs(getattr(common, name), getattr(module, name), f"{module.__name__}.{name}")

    def test_branch_rule_and_trajectory_order_is_frozen(self) -> None:
        self.assertEqual(("mind_soft", "mind_hard", "intensity_soft", "mind_reversed"), c0.BRANCH_ORDER)
        self.assertEqual(("topology_only", "mind", "ncc9", "support_ncc9", "mind_and_ncc9"), c1.UTILITY_RULES)
        candidate_ids = c1._expected_candidate_ids("exploration")
        self.assertEqual(24, len(candidate_ids))
        self.assertEqual(24, len(set(candidate_ids)))
        self.assertEqual(tuple(2.0**-index for index in range(17)), c1.GLOBAL_COEFFICIENTS)
        self.assertEqual(4, len(c2.TRAJECTORIES))
        self.assertEqual(4, c2.MAX_STEPS)
        self.assertEqual((0.0011, 0.001075, 0.00105, 0.001025), c2.MARGIN_SCHEDULE)
        self.assertEqual(0.001, c2.MIN_MEAN_DICE_DELTA)
        self.assertEqual(19, len(common.IXI_DEVELOPMENT_CASES))

    def test_selfcheck_invariants_hold(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            for module, name in ((c1, "c1.json"), (c2, "c2.json")):
                output = Path(tmp) / name
                module.selfcheck_stage(argparse.Namespace(output=output))
                payload = json.loads(output.read_text(encoding="utf-8"))
                self.assertEqual("PASS", payload["status"], payload["failed"])


class ConfirmationIsPreregisteredTest(unittest.TestCase):
    """C1 confirmation replays one frozen candidate; it must never derive one automatically."""

    @staticmethod
    def _prepare_args(stage: str, manifest: str | None, sha: str | None) -> argparse.Namespace:
        return argparse.Namespace(
            stage=stage,
            num_shards=1,
            physical_gpus="0",
            explore_manifest=manifest,
            explore_manifest_sha256=sha,
        )

    def test_confirmation_without_a_frozen_manifest_is_refused(self) -> None:
        for manifest, sha in ((None, None), ("m.json", None), (None, "a" * 64)):
            with self.assertRaises(ValueError):
                c1.prepare_stage(self._prepare_args("confirmation", manifest, sha))

    def test_exploration_must_not_consume_an_exploration_manifest(self) -> None:
        with self.assertRaises(ValueError):
            c1.prepare_stage(self._prepare_args("exploration", "m.json", "a" * 64))

    def test_confirmation_candidate_set_is_a_single_frozen_id(self) -> None:
        self.assertEqual([c1.CONFIRMATION_SPEC["candidate_id"]], c1._expected_candidate_ids("confirmation"))
        self.assertEqual("mind_clip_s1_w1", c1.CONFIRMATION_SPEC["candidate_id"])
        self.assertEqual(c1.CONFIRMATION_SPEC["candidate_id"], c1.CONFIRMATION_POLICY["candidate_id"])
        self.assertIn(c1.CONFIRMATION_SPEC["candidate_id"], c1._expected_candidate_ids("exploration"))

    @needs_product("c1_confirmation")
    def test_a_confirmation_marker_carrying_another_candidate_is_rejected(self) -> None:
        stage_dir = HISTORICAL_PRODUCTS["c1_confirmation"]
        contract_path = stage_dir / "stage_contract.json"
        contract = json.loads(contract_path.read_text(encoding="utf-8"))
        contract_sha = sha256_file(contract_path)
        case_id = contract["case_ids"][0]
        marker = stage_dir / "cases" / case_id / "case_complete.json"
        payload = json.loads(marker.read_text(encoding="utf-8"))
        for other in ("mind_global_k00", "mind_clip_s1_w2"):
            tampered = copy.deepcopy(payload)
            tampered["rows"][0]["candidate_id"] = other
            with self.assertRaises(RuntimeError, msg=f"confirmation accepted candidate {other}"):
                c1._validate_case_payload(tampered, "confirmation", case_id, contract_sha, marker, contract)


@needs_product("c1_exploration")
class C1GoldenReplayTest(unittest.TestCase):
    product: ClassVar[str] = "c1_exploration"

    def _replay(self, stage_dir: Path, stage: str) -> dict[str, Any]:
        contract_path = stage_dir / "stage_contract.json"
        contract = json.loads(contract_path.read_text(encoding="utf-8"))
        contract_sha = sha256_file(contract_path)
        rows: list[dict[str, Any]] = []
        for case_id in contract["case_ids"]:
            path = stage_dir / "cases" / case_id / "case_complete.json"
            payload = json.loads(path.read_text(encoding="utf-8"))
            rows.extend(c1._validate_case_payload(payload, stage, case_id, contract_sha, path, contract))
        summary, rule_rows = c1._summarise(rows, stage, len(contract["case_ids"]))
        serializable = [{k: v for k, v in row.items() if k != "exact_report"} for row in rows]
        per_case = c1._per_case_rows(rows, stage)
        return {
            "contract": contract,
            "rows": rows,
            "summary": summary,
            "rule_rows": rule_rows,
            "per_case": per_case,
            "per_candidate.csv": rows_to_csv(c1._csv_fields(serializable, C1_PER_CANDIDATE_PREFERRED), serializable),
            "per_case.csv": rows_to_csv(list(per_case[0].keys()), per_case),
            "operator_rule_summary.csv": rows_to_csv(list(rule_rows[0].keys()), rule_rows),
            "summary.json": json_artifact(summary),
        }

    def _assert_artifacts(self, out: dict[str, Any], stage_dir: Path) -> None:
        for filename in ("per_candidate.csv", "per_case.csv", "operator_rule_summary.csv", "summary.json"):
            stored = (stage_dir / filename).read_text(encoding="utf-8")
            self.assertEqual(stored, out[filename], f"{filename} no longer reproduces the historical bytes")

    def test_exploration_reproduces_every_stored_artifact(self) -> None:
        stage_dir = HISTORICAL_PRODUCTS["c1_exploration"]
        out = self._replay(stage_dir, "exploration")
        self.assertEqual(
            (19, 456, 19, 120),
            (len(out["contract"]["case_ids"]), len(out["rows"]), len(out["per_case"]), len(out["rule_rows"])),
        )
        self.assertEqual("PASS", out["summary"]["execution_integrity_status"])
        self._assert_artifacts(out, stage_dir)

    @needs_product("c1_confirmation")
    def test_confirmation_reproduces_every_stored_artifact(self) -> None:
        stage_dir = HISTORICAL_PRODUCTS["c1_confirmation"]
        out = self._replay(stage_dir, "confirmation")
        self.assertEqual(
            (39, 39, 39, 5),
            (len(out["contract"]["case_ids"]), len(out["rows"]), len(out["per_case"]), len(out["rule_rows"])),
        )
        self._assert_artifacts(out, stage_dir)

    @needs_product("c1_confirmation")
    def test_historical_manifests_are_unchanged(self) -> None:
        self.assertEqual(
            C1_EXPLORATION_MANIFEST_SHA256, sha256_file(HISTORICAL_PRODUCTS["c1_exploration"] / "run_manifest.json")
        )
        self.assertEqual(
            C1_CONFIRMATION_MANIFEST_SHA256, sha256_file(HISTORICAL_PRODUCTS["c1_confirmation"] / "run_manifest.json")
        )

    def test_candidate_row_order_is_the_frozen_candidate_order(self) -> None:
        out = self._replay(HISTORICAL_PRODUCTS["c1_exploration"], "exploration")
        expected = c1._expected_candidate_ids("exploration")
        for index, case_id in enumerate(out["contract"]["case_ids"]):
            chunk = out["rows"][index * 24 : (index + 1) * 24]
            self.assertEqual([case_id] * 24, [row["case_id"] for row in chunk])
            self.assertEqual(expected, [row["candidate_id"] for row in chunk])


@needs_product("c1_superseded")
class ExpectedRejectionTest(unittest.TestCase):
    """Commit 9275171 added required geometry fields, so the earlier product must be refused."""

    product: ClassVar[str] = "c1_superseded"

    def test_superseded_exploration_product_is_rejected(self) -> None:
        stage_dir = HISTORICAL_PRODUCTS["c1_superseded"]
        contract_path = stage_dir / "stage_contract.json"
        contract = json.loads(contract_path.read_text(encoding="utf-8"))
        contract_sha = sha256_file(contract_path)
        case_id = contract["case_ids"][0]
        path = stage_dir / "cases" / case_id / "case_complete.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(contract_sha, payload["contract_sha256"])
        with self.assertRaises(RuntimeError):
            c1._validate_case_payload(payload, "exploration", case_id, contract_sha, path, contract)


@needs_product("c1_confirmation")
class C1WorkerReportTest(unittest.TestCase):
    """`_valid_worker_report` is the gate between a shard's output and finalization."""

    product: ClassVar[str] = "c1_confirmation"

    def setUp(self) -> None:
        stage_dir = HISTORICAL_PRODUCTS["c1_confirmation"]
        contract_path = stage_dir / "stage_contract.json"
        self.contract = json.loads(contract_path.read_text(encoding="utf-8"))
        self.contract_sha = sha256_file(contract_path)
        report_path = next(stage_dir.glob("workers/attempts/*/worker_00.json"))
        self.report = json.loads(report_path.read_text(encoding="utf-8"))
        self.attempt_id = self.report["attempt_id"]

    def _valid(self, report: dict[str, Any], shard_index: int = 0) -> bool:
        return c1._valid_worker_report(report, self.contract, self.contract_sha, shard_index, self.attempt_id)

    def test_unmodified_report_is_accepted(self) -> None:
        self.assertTrue(self._valid(copy.deepcopy(self.report)))
        self.assertGreaterEqual(len(self.report["computed_case_ids"]), 2)

    def test_wrong_physical_gpu_is_rejected(self) -> None:
        mapped = self.contract["shard_to_physical_gpu"]["0"]
        for wrong in ("99", str(int(mapped) + 1), ""):
            report = copy.deepcopy(self.report)
            report["execution"]["physical_gpu"] = wrong
            self.assertFalse(self._valid(report), f"physical_gpu={wrong!r} accepted")

    def test_non_strict_checkpoint_load_is_rejected(self) -> None:
        for value in (False, None, "True"):
            report = copy.deepcopy(self.report)
            report["checkpoint"]["strict"] = value
            self.assertFalse(self._valid(report), f"strict={value!r} accepted")

    def test_unexpected_or_missing_checkpoint_keys_are_rejected(self) -> None:
        report = copy.deepcopy(self.report)
        report["checkpoint"]["unexpected_keys"] = ["head.weight"]
        self.assertFalse(self._valid(report))
        report = copy.deepcopy(self.report)
        report["checkpoint"]["missing_keys"] = ["st_half.grid"]
        self.assertFalse(self._valid(report))

    def test_duplicate_case_is_rejected(self) -> None:
        report = copy.deepcopy(self.report)
        report["computed_case_ids"][1] = report["computed_case_ids"][0]
        self.assertFalse(self._valid(report))

    def test_missing_case_is_rejected(self) -> None:
        report = copy.deepcopy(self.report)
        report["computed_case_ids"].pop()
        self.assertFalse(self._valid(report))

    def test_reordered_cases_are_rejected(self) -> None:
        report = copy.deepcopy(self.report)
        report["computed_case_ids"] = list(reversed(report["computed_case_ids"]))
        self.assertFalse(self._valid(report))

    def test_extra_case_outside_the_shard_is_rejected(self) -> None:
        outside = next(c for c in self.contract["case_ids"] if c not in self.contract["shards"]["0"])
        report = copy.deepcopy(self.report)
        report["computed_case_ids"].append(outside)
        self.assertFalse(self._valid(report))

    def test_wrong_shard_index_and_contract_are_rejected(self) -> None:
        self.assertFalse(self._valid(copy.deepcopy(self.report), shard_index=1))
        report = copy.deepcopy(self.report)
        report["contract_sha256"] = "0" * 64
        self.assertFalse(self._valid(report))

    def test_incomplete_or_unloaded_worker_is_rejected(self) -> None:
        report = copy.deepcopy(self.report)
        report["status"] = "PARTIAL"
        self.assertFalse(self._valid(report))
        report = copy.deepcopy(self.report)
        report["execution"]["model_loaded"] = False
        self.assertFalse(self._valid(report))


@needs_product("c2")
class C2GoldenReplayTest(unittest.TestCase):
    product: ClassVar[str] = "c2"

    @classmethod
    def setUpClass(cls) -> None:
        root = HISTORICAL_PRODUCTS["c2"]
        cls.root = root
        cls.contract = json.loads((root / "c2_contract.json").read_text(encoding="utf-8"))
        cls.contract_sha = sha256_file(root / "c2_contract.json")
        cls.refs = {row["case_id"]: row for row in runtime.read_csv(root / "c1_reference.csv")}
        cls.by_input = {row["case_id"]: row for row in runtime.read_csv(root / "datasets.csv") if row["split"] == "val"}

    def _replay(self) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        step_rows: list[dict[str, Any]] = []
        branch_rows: list[dict[str, Any]] = []
        for case_id in self.contract["case_ids"]:
            path = self.root / "cases" / case_id / "case_complete.json"
            payload = json.loads(path.read_text(encoding="utf-8"))
            step_rows.extend(
                c2._validate_case(
                    payload, path, case_id, self.contract_sha, self.by_input[case_id], self.contract, self.refs[case_id]
                )
            )
            branch_rows.extend(payload["branch_rows"])
        return step_rows, branch_rows

    def test_reproduces_every_stored_artifact(self) -> None:
        step_rows, branch_rows = self._replay()
        self.assertEqual((58, 928, 232), (len(self.contract["case_ids"]), len(step_rows), len(branch_rows)))
        summary_rows, summary = c2._summary_rows(branch_rows, [self.refs[c] for c in self.contract["case_ids"]])
        self.assertEqual(4, len(summary_rows))
        produced = {
            "per_step.csv": rows_to_csv(c2._csv_fields(step_rows, C2_STEP_PREFERRED), step_rows),
            "per_branch.csv": rows_to_csv(c2._csv_fields(branch_rows, C2_BRANCH_PREFERRED), branch_rows),
            "trajectory_summary.csv": rows_to_csv(list(summary_rows[0].keys()), summary_rows),
            "summary.json": json_artifact(summary),
        }
        for filename, text in produced.items():
            stored = (self.root / filename).read_text(encoding="utf-8")
            self.assertEqual(stored, text, f"{filename} no longer reproduces the historical bytes")

    def test_every_trajectory_keeps_four_rows_even_after_stopping(self) -> None:
        step_rows, _ = self._replay()
        expected = [(spec["trajectory_id"], step) for spec in c2.TRAJECTORIES for step in range(1, 5)]
        for index, case_id in enumerate(self.contract["case_ids"]):
            chunk = step_rows[index * 16 : (index + 1) * 16]
            self.assertEqual(expected, [(row["trajectory_id"], row["step"]) for row in chunk])
            self.assertEqual([case_id] * 16, [row["case_id"] for row in chunk])

    def test_historical_manifest_is_unchanged(self) -> None:
        self.assertEqual(C2_MANIFEST_SHA256, sha256_file(self.root / "c2_manifest.json"))

    def test_decision_is_taken_before_labels(self) -> None:
        step_rows, _ = self._replay()
        self.assertTrue(all(row["labels_used_for_decision"] is False for row in step_rows))


class _C2CaseFixture(unittest.TestCase):
    """Shared setup for the C2 tamper suites; every mutation happens on a temp copy."""

    def setUp(self) -> None:
        self.root = HISTORICAL_PRODUCTS["c2"]
        self.contract = json.loads((self.root / "c2_contract.json").read_text(encoding="utf-8"))
        self.contract_sha = sha256_file(self.root / "c2_contract.json")
        self.refs = {row["case_id"]: row for row in runtime.read_csv(self.root / "c1_reference.csv")}
        self.by_input = {
            row["case_id"]: row for row in runtime.read_csv(self.root / "datasets.csv") if row["split"] == "val"
        }
        self.case_id = self.contract["case_ids"][0]
        self.marker = self.root / "cases" / self.case_id / "case_complete.json"
        self.payload = json.loads(self.marker.read_text(encoding="utf-8"))

    def validate(self, payload: dict[str, Any], **overrides: Any) -> list[dict[str, Any]]:
        kwargs: dict[str, Any] = {
            "marker": self.marker,
            "case_id": self.case_id,
            "contract_sha": self.contract_sha,
            "input_row": self.by_input[self.case_id],
            "contract": self.contract,
            "c1_row": self.refs[self.case_id],
        }
        kwargs.update(overrides)
        return c2._validate_case(payload, **kwargs)


@needs_product("c2")
class C2NegativeControlTest(_C2CaseFixture):
    product: ClassVar[str] = "c2"

    def test_unmodified_marker_is_accepted(self) -> None:
        self.assertEqual(16, len(self.validate(copy.deepcopy(self.payload))))

    def test_altered_contract_sha_is_rejected(self) -> None:
        with self.assertRaises(RuntimeError):
            self.validate(copy.deepcopy(self.payload), contract_sha="0" * 64)

    def test_altered_input_row_is_rejected(self) -> None:
        bad = dict(self.by_input[self.case_id])
        bad["sha256"] = "0" * 64
        with self.assertRaises(RuntimeError):
            self.validate(copy.deepcopy(self.payload), input_row=bad)

    def test_changed_c1_reference_is_rejected(self) -> None:
        bad = dict(self.refs[self.case_id])
        bad["baseline_dice"] = "0.5"
        with self.assertRaises(RuntimeError):
            self.validate(copy.deepcopy(self.payload), c1_row=bad)

    def test_wrong_physical_gpu_is_rejected(self) -> None:
        payload = copy.deepcopy(self.payload)
        payload["execution"]["physical_gpu"] = "99"
        with self.assertRaises(RuntimeError):
            self.validate(payload)

    def test_non_strict_checkpoint_load_is_rejected(self) -> None:
        for value in (False, None, "True"):
            payload = copy.deepcopy(self.payload)
            payload["execution"]["checkpoint_load_report"]["strict"] = value
            with self.assertRaises(RuntimeError, msg=f"strict={value!r} accepted"):
                self.validate(payload)

    def test_unexpected_checkpoint_keys_are_rejected(self) -> None:
        payload = copy.deepcopy(self.payload)
        payload["execution"]["checkpoint_load_report"]["unexpected_keys"] = ["head.weight"]
        with self.assertRaises(RuntimeError):
            self.validate(payload)

    def test_reordered_or_dropped_step_rows_are_rejected(self) -> None:
        payload = copy.deepcopy(self.payload)
        payload["rows"] = list(reversed(payload["rows"]))
        with self.assertRaises(RuntimeError):
            self.validate(payload)
        payload = copy.deepcopy(self.payload)
        del payload["rows"][7]
        with self.assertRaises(RuntimeError):
            self.validate(payload)

    def test_label_leaking_into_the_decision_is_rejected(self) -> None:
        payload = copy.deepcopy(self.payload)
        payload["rows"][0]["labels_used_for_decision"] = True
        with self.assertRaises(RuntimeError):
            self.validate(payload)

    def test_uncertified_returned_field_is_rejected(self) -> None:
        for status in ("ERROR", "INVALID_INPUT", "INCONCLUSIVE_RESOURCE_LIMIT", "NOT_CERTIFIED_BY_PREDICATE"):
            payload = copy.deepcopy(self.payload)
            payload["rows"][0]["returned_exact_status"] = status
            with self.assertRaises(RuntimeError, msg=f"status {status} passed C2"):
                self.validate(payload)

    def test_nonzero_trilinear_fold_bound_is_rejected(self) -> None:
        payload = copy.deepcopy(self.payload)
        payload["rows"][0]["returned_trilinear_fold_percent_upper_bound"] = 0.1
        with self.assertRaises(RuntimeError):
            self.validate(payload)

    def test_corrupted_complete_marker_is_rejected(self) -> None:
        for key, value in (("status", "PARTIAL"), ("schema", "ctcf-search-c2-case-v0"), ("case_id", "subject_999")):
            payload = copy.deepcopy(self.payload)
            payload[key] = value
            with self.assertRaises(RuntimeError, msg=f"{key}={value} passed C2"):
                self.validate(payload)

    def test_tampered_branch_rows_are_rejected(self) -> None:
        payload = copy.deepcopy(self.payload)
        payload["branch_rows"][0]["final_dice"] = 0.99
        with self.assertRaises(RuntimeError):
            self.validate(payload)

    def test_frozen_partition_is_exact_round_robin(self) -> None:
        self.assertEqual(
            runtime.round_robin_shards(self.contract["case_ids"], self.contract["num_shards"]),
            {key: list(value) for key, value in self.contract["shards"].items()},
        )
        flattened = runtime.flattened_shards(self.contract)
        self.assertEqual(58, len(flattened))
        self.assertEqual(sorted(self.contract["case_ids"]), sorted(flattened))


@needs_product("c2")
class C2DecisionSnapshotTamperTest(_C2CaseFixture):
    """The decision snapshot proves the decision was taken before labels reached the GPU."""

    product: ClassVar[str] = "c2"

    def setUp(self) -> None:
        super().setUp()
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.case_dir = Path(self._tmp.name) / self.case_id
        shutil.copytree(self.marker.parent, self.case_dir)
        self.marker = self.case_dir / "case_complete.json"
        self.decision_dir = self.case_dir / "decision"

    def _rewrite(self, name: str, payload: dict[str, Any]) -> str:
        path = self.decision_dir / name
        path.write_text(json_artifact(payload), encoding="utf-8", newline="\n")
        return sha256_file(path)

    def test_copy_without_tampering_is_accepted(self) -> None:
        self.assertEqual(16, len(self.validate(copy.deepcopy(self.payload))))

    def test_decision_inputs_hash_mismatch_is_rejected(self) -> None:
        payload = copy.deepcopy(self.payload)
        payload["decision_inputs_sha256"] = "0" * 64
        with self.assertRaises(RuntimeError):
            self.validate(payload)

    def test_decisions_hash_mismatch_is_rejected(self) -> None:
        payload = copy.deepcopy(self.payload)
        payload["decisions_sha256"] = "0" * 64
        with self.assertRaises(RuntimeError):
            self.validate(payload)

    def test_edited_decision_inputs_file_is_rejected(self) -> None:
        inputs = json.loads((self.decision_dir / "decision_inputs.json").read_text(encoding="utf-8"))
        inputs["rows"][0]["current_mind"] = 0.0
        self._rewrite("decision_inputs.json", inputs)
        with self.assertRaises(RuntimeError):
            self.validate(copy.deepcopy(self.payload))

    def test_label_flag_flipped_in_the_snapshot_is_rejected(self) -> None:
        for name, key in (("decision_inputs.json", "decision_inputs_sha256"), ("decisions.json", "decisions_sha256")):
            self.setUp()
            document = json.loads((self.decision_dir / name).read_text(encoding="utf-8"))
            document["labels_loaded_to_device"] = True
            payload = copy.deepcopy(self.payload)
            payload[key] = self._rewrite(name, document)
            if name == "decisions.json":
                payload["decisions_sha256"] = self._rewrite(name, document)
            with self.assertRaises(RuntimeError, msg=f"{name} accepted labels_loaded_to_device=True"):
                self.validate(payload)

    def test_label_field_restored_into_decision_inputs_is_rejected(self) -> None:
        inputs = json.loads((self.decision_dir / "decision_inputs.json").read_text(encoding="utf-8"))
        self.assertNotIn("returned_dice", inputs["rows"][0])
        inputs["rows"][0]["returned_dice"] = self.payload["rows"][0]["returned_dice"]
        payload = copy.deepcopy(self.payload)
        payload["decision_inputs_sha256"] = self._rewrite("decision_inputs.json", inputs)
        with self.assertRaises(RuntimeError):
            self.validate(payload)

    def test_decision_inputs_must_reconstruct_from_the_case_rows(self) -> None:
        inputs = json.loads((self.decision_dir / "decision_inputs.json").read_text(encoding="utf-8"))
        inputs["rows"].pop()
        payload = copy.deepcopy(self.payload)
        payload["decision_inputs_sha256"] = self._rewrite("decision_inputs.json", inputs)
        with self.assertRaises(RuntimeError):
            self.validate(payload)

    def test_altered_recorded_actions_are_rejected(self) -> None:
        decisions = json.loads((self.decision_dir / "decisions.json").read_text(encoding="utf-8"))
        decisions["actions"][0]["action"] = "ROLLBACK"
        payload = copy.deepcopy(self.payload)
        payload["decisions_sha256"] = self._rewrite("decisions.json", decisions)
        with self.assertRaises(RuntimeError):
            self.validate(payload)

    def test_decisions_pointing_at_another_input_snapshot_is_rejected(self) -> None:
        decisions = json.loads((self.decision_dir / "decisions.json").read_text(encoding="utf-8"))
        decisions["decision_inputs_sha256"] = "0" * 64
        payload = copy.deepcopy(self.payload)
        payload["decisions_sha256"] = self._rewrite("decisions.json", decisions)
        with self.assertRaises(RuntimeError):
            self.validate(payload)

    def test_missing_snapshot_file_is_rejected(self) -> None:
        (self.decision_dir / "decisions.json").unlink()
        with self.assertRaises(RuntimeError):
            self.validate(copy.deepcopy(self.payload))


@needs_product("c0")
class C0HistoricalAuditTest(unittest.TestCase):
    product: ClassVar[str] = "c0"

    def test_stage_cardinalities(self) -> None:
        for stage, n_cases, n_rows in (("smoke", 1, 4), ("development", 19, 76)):
            stage_dir = HISTORICAL_PRODUCTS["c0"] / stage
            contract = json.loads((stage_dir / "stage_contract.json").read_text(encoding="utf-8"))
            self.assertEqual(n_cases, len(contract["case_ids"]), stage)
            self.assertEqual(n_cases, len(list((stage_dir / "cases").iterdir())), stage)
            rows = runtime.read_csv(stage_dir / "per_case_branch.csv")
            self.assertEqual(n_rows, len(rows), stage)
            self.assertEqual(n_cases * len(c0.BRANCH_ORDER), len(rows), stage)

    def test_branch_order_within_each_case(self) -> None:
        for stage in ("smoke", "development"):
            rows = runtime.read_csv(HISTORICAL_PRODUCTS["c0"] / stage / "per_case_branch.csv")
            for start in range(0, len(rows), 4):
                chunk = rows[start : start + 4]
                self.assertEqual(list(c0.BRANCH_ORDER), [row["branch"] for row in chunk])
                self.assertEqual(1, len({row["case_id"] for row in chunk}))

    def test_development_cases_match_the_frozen_salted_prefix(self) -> None:
        contract = json.loads(
            (HISTORICAL_PRODUCTS["c0"] / "development" / "stage_contract.json").read_text(encoding="utf-8")
        )
        self.assertEqual(list(common.IXI_DEVELOPMENT_CASES), list(contract["case_ids"]))


GOLDEN_CLASSES = (
    C0HistoricalAuditTest,
    C1GoldenReplayTest,
    C1WorkerReportTest,
    C2DecisionSnapshotTamperTest,
    C2GoldenReplayTest,
    C2NegativeControlTest,
    ExpectedRejectionTest,
)


class HistoricalGoldenCoverageTest(unittest.TestCase):
    """A missing historical product must never read as a silent green."""

    def test_every_golden_class_declares_a_known_product(self) -> None:
        for cls in GOLDEN_CLASSES:
            self.assertIn(cls.product, HISTORICAL_PRODUCTS, cls.__name__)
        self.assertEqual(5, len(HISTORICAL_PRODUCTS))
        self.assertEqual(set(HISTORICAL_PRODUCTS), {cls.product for cls in GOLDEN_CLASSES})

    def test_executed_and_skipped_products_are_disjoint_and_total(self) -> None:
        products = {cls.product for cls in GOLDEN_CLASSES}
        executed = {name for name in products if HISTORICAL_PRODUCTS[name].is_dir()}
        skipped = products - executed
        self.assertEqual(set(), executed & skipped)
        self.assertEqual(products, executed | skipped)
        if REQUIRE_HISTORICAL:
            self.assertEqual(
                set(),
                skipped,
                f"CTCF_REQUIRE_HISTORICAL_GOLDENS=1 but these products are absent: {sorted(skipped)}",
            )

    def test_required_mode_is_reachable_from_the_environment(self) -> None:
        self.assertEqual(os.environ.get("CTCF_REQUIRE_HISTORICAL_GOLDENS") == "1", REQUIRE_HISTORICAL)


class RuntimeHelperTest(unittest.TestCase):
    def test_dataset_rows_records_identity_size_and_hash(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first = root / "subject_7.pkl"
            second = root / "p_OASIS_0002.pkl"
            atlas = root / "atlas.pkl"
            first.write_bytes(b"abc")
            second.write_bytes(b"de")
            atlas.write_bytes(b"f")
            rows = runtime.dataset_rows([str(first), str(second)], "IXI", "val", str(atlas))
            self.assertEqual(["subject_7", "OASIS_0002", "atlas"], [row["case_id"] for row in rows])
            self.assertEqual(["val", "val", "atlas"], [row["split"] for row in rows])
            self.assertEqual(["IXI"] * 3, [row["dataset"] for row in rows])
            self.assertEqual([3, 2, 1], [row["bytes"] for row in rows])
            self.assertEqual(sha256_file(first), rows[0]["sha256"])
            self.assertEqual(str(first.resolve()), rows[0]["path"])
            self.assertTrue(all(row["mtime_utc"].endswith("Z") for row in rows))

    def test_dataset_rows_without_an_atlas(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "subject_1.pkl"
            path.write_bytes(b"x")
            rows = runtime.dataset_rows([str(path)], "OASIS", "val", None)
            self.assertEqual(1, len(rows))
            self.assertEqual("val", rows[0]["split"])

    def test_save_reload_certify_certifies_the_stored_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "candidate.npz"
            flow = torch.zeros((1, 3, 5, 6, 7), dtype=torch.float32)
            stored, exact = runtime.save_reload_certify(flow, path, common.CLAIM_EPS)
            self.assertTrue(path.is_file())
            self.assertEqual("CERTIFIED", exact["status"])
            self.assertEqual(flow.shape, stored.shape)
            self.assertIsNot(flow, stored)
            self.assertTrue(torch.equal(flow, stored))

    def test_save_reload_certify_reports_a_folded_field_as_not_certified(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "folded.npz"
            flow = torch.zeros((1, 3, 5, 6, 7), dtype=torch.float32)
            flow[0, 0, 2, 2, 2] = -5.0
            _, exact = runtime.save_reload_certify(flow, path, common.CLAIM_EPS)
            self.assertNotEqual("CERTIFIED", exact["status"])

    def test_round_robin_and_flattened_order(self) -> None:
        case_ids = [f"s{index}" for index in range(7)]
        shards = runtime.round_robin_shards(case_ids, 3)
        self.assertEqual({"0": ["s0", "s3", "s6"], "1": ["s1", "s4"], "2": ["s2", "s5"]}, shards)
        contract = {"num_shards": 3, "shards": shards, "case_ids": case_ids}
        self.assertEqual(["s0", "s3", "s6", "s1", "s4", "s2", "s5"], runtime.flattened_shards(contract))
        self.assertEqual(0, runtime.expected_shard_for_case(contract, "s6"))
        self.assertEqual(2, runtime.expected_shard_for_case(contract, "s5"))

    def test_validate_shard_partition_accepts_only_the_frozen_order(self) -> None:
        case_ids = [f"s{index}" for index in range(6)]
        contract = {"num_shards": 2, "shards": runtime.round_robin_shards(case_ids, 2), "case_ids": case_ids}
        good = runtime.flattened_shards(contract)
        runtime.validate_shard_partition(contract, good, 6, "boom")
        for bad in (
            list(reversed(good)),
            [*good[:-1], good[0]],
            good[:-1],
            case_ids,
        ):
            with self.assertRaises(RuntimeError, msg=f"accepted {bad}"):
                runtime.validate_shard_partition(contract, bad, 6, "boom")

    def test_parse_physical_gpus_rejects_bad_lists(self) -> None:
        self.assertEqual(["2", "3"], runtime.parse_physical_gpus(" 2 , 3 ", 2, "boom"))
        for value, shards in (("2,3", 3), ("2,2", 2), ("2,x", 2), ("", 1)):
            with self.assertRaises(ValueError, msg=f"accepted {value!r}"):
                runtime.parse_physical_gpus(value, shards, "boom")

    def test_worker_marker_paths_follow_the_attempt_convention(self) -> None:
        root = Path("/run")
        marker, failure = runtime.worker_marker_paths(root, "A_1", 3)
        self.assertEqual(runtime.attempt_dir(root, "A_1") / "worker_03.json", marker)
        self.assertEqual(runtime.attempt_dir(root, "A_1") / "worker_03_failure.json", failure)


class PureHelperGoldenTest(unittest.TestCase):
    """Exact numeric golden for the helpers a refactor may relocate; values compared through repr()."""

    @classmethod
    def setUpClass(cls) -> None:
        shape = (16, 18, 20)
        gen = torch.Generator().manual_seed(12345)
        cls.fixed = torch.rand((1, 1, *shape), generator=gen, dtype=torch.float32)
        cls.moving = torch.rand((1, 1, *shape), generator=gen, dtype=torch.float32)
        cls.flow = torch.randn((1, 3, *shape), generator=gen, dtype=torch.float32) * 0.05
        cls.mask = ts.geometry_mask(shape, 4, torch.device("cpu"))
        cls.values = np.array([0.1, -0.2, 0.0, 0.35, -0.05, 0.9, 0.02, -0.4], dtype=np.float64)

    def test_case_identity_and_salted_order(self) -> None:
        self.assertEqual("subject_126", common.case_id_from_path("/data/IXI_data/Val/subject_126.pkl"))
        self.assertEqual("OASIS_0001", common.case_id_from_path("/x/p_OASIS_0001.pkl"))
        self.assertEqual(
            "28e8d4f70d749d99b166a394dd5b6a80457d7d4282105b9e8a395384a149b77d",
            common.salted_case_hash("subject_126"),
        )
        ordered = sorted(common.IXI_DEVELOPMENT_CASES, key=common.salted_case_hash)
        self.assertEqual(list(common.IXI_DEVELOPMENT_CASES), ordered)

    def test_canonical_payload_hashing(self) -> None:
        self.assertEqual(
            "2db667784f96b7c1b3d2279c6f45bb307169b7d162c84572df3d0893637346aa",
            common.text_sha256("CTCF-GATE"),
        )
        self.assertEqual(
            common.payload_sha256({"a": True, "b": [1, 2, {"c": 3.5}]}),
            common.payload_sha256({"b": [1, 2, {"c": 3.5}], "a": True}),
        )
        self.assertNotEqual(common.payload_sha256({"a": 1}), common.payload_sha256({"a": 1.0}))

    def test_finiteness_predicate(self) -> None:
        for value in (1, 1.0, np.float64(2.5)):
            self.assertTrue(common.is_finite_number(value), repr(value))
        for value in (True, False, float("nan"), float("inf"), -float("inf"), "1", None):
            self.assertFalse(common.is_finite_number(value), repr(value))
        with self.assertRaises(RuntimeError):
            common.require_finite({"a": float("nan"), "b": 1.0}, "L")

    def test_relative_improvement_tolerance(self) -> None:
        self.assertFalse(common.relative_improvement(2.0, 2.0)[2])
        self.assertTrue(common.relative_improvement(2.0, 2.0 - 4e-6)[2])
        self.assertFalse(common.relative_improvement(float("nan"), 1.0)[2])
        self.assertIsNone(common.relative_improvement(float("nan"), 1.0)[0])
        improvement, tolerance, improved = common.relative_improvement(2.0, 1.5)
        self.assertEqual(0.5, improvement)
        self.assertEqual(repr(2e-06), repr(tolerance))
        self.assertTrue(improved)

    def test_distribution_and_sign_summaries(self) -> None:
        sign = common.sign_summary(self.values)
        self.assertEqual((4, 3, 1), (sign["improved"], sign["worsened"], sign["unchanged"]))
        self.assertEqual(repr(0.09), repr(sign["mean"]))
        self.assertEqual((10_000, 0), (sign["mean_ci95"]["replicates"], sign["mean_ci95"]["seed"]))
        self.assertEqual(repr(-0.135), repr(sign["mean_ci95"]["low"]))
        self.assertEqual(repr(0.37124999999999997), repr(sign["mean_ci95"]["high"]))
        distribution = common.distribution_summary(self.values)
        self.assertEqual({"mean", "median", "min", "max"}, set(distribution))
        self.assertEqual((repr(0.9), repr(-0.4)), (repr(distribution["max"]), repr(distribution["min"])))
        with self.assertRaises(RuntimeError):
            common.sign_summary(np.array([np.nan, 1.0]))
        self.assertEqual("not_available", common.bootstrap_ci(np.array([], dtype=np.float64))["method"])

    def test_geometry_diagnostics_keep_legacy_behaviour(self) -> None:
        tiny = self.flow * 0.01
        certified = common.deformation_quality_metrics(tiny, exact_certified=True)
        uncertified = common.deformation_quality_metrics(tiny, exact_certified=False)
        self.assertEqual(repr(0.00014250155072659254), repr(certified["sdlogj"]))
        self.assertEqual(0.0, certified["trilinear_fold_percent_upper_bound"])
        self.assertEqual("ZERO_BY_EXACT_CERTIFICATE", certified["trilinear_fold_status"])
        self.assertIsNone(uncertified["trilinear_fold_percent_upper_bound"])
        self.assertEqual("NOT_ESTABLISHED", uncertified["trilinear_fold_status"])
        self.assertEqual(certified["sdlogj"], uncertified["sdlogj"])

    def test_utility_metrics_on_a_fixed_field(self) -> None:
        fixed_norm = ts.masked_zscore(self.fixed, self.mask)
        moving_norm = ts.masked_zscore(self.moving, self.mask)
        metrics = common.candidate_metrics(
            self.flow,
            fixed_norm,
            moving_norm,
            ts.mind_ssc(self.fixed),
            ts.mind_ssc(self.moving),
            self.mask,
            ts.proposal_support_weights(self.flow, self.mask),
        )
        self.assertEqual({"ncc9", "ncc7", "support_ncc9", "mind"}, set(metrics))
        self.assertEqual(
            repr(ts.ncc_loss_from_normalized(fixed_norm, moving_norm, self.flow, self.mask, win=9)),
            repr(metrics["ncc9"]),
        )

    def test_exact_checker_gap_classification(self) -> None:
        self.assertFalse(c1._exact_checker_gap({"exact_status": "CERTIFIED"}))
        self.assertFalse(c1._exact_checker_gap({"exact_status": "NOT_CERTIFIED_BY_PREDICATE"}))
        self.assertTrue(c1._exact_checker_gap({"exact_status": "INCONCLUSIVE_RESOURCE_LIMIT"}))

    def test_proposal_statistics_on_a_fixed_proposal(self) -> None:
        proposal = ts.build_proposal(
            self.fixed, self.moving, self.flow, self.mask, feature="mind", orientation="target_centered"
        )
        stats = common.proposal_statistics(proposal, self.flow, self.mask)
        self.assertEqual({"entropy_mean", "confidence_mean", "proposal_norm_mean", "proposal_norm_max"}, set(stats))
        magnitude = self.flow.square().sum(dim=1, keepdim=True).sqrt()
        self.assertEqual(
            repr(float(magnitude.masked_select(self.mask).double().mean().item())),
            repr(stats["proposal_norm_mean"]),
        )

    def test_offset_grid_and_step_panel_are_frozen(self) -> None:
        self.assertEqual(27, len(ts.OFFSETS))
        self.assertEqual(13, ts.ZERO_OFFSET_INDEX)
        self.assertEqual((0, 0, 0), ts.OFFSETS[13])
        self.assertEqual(tuple(2.0**-index for index in range(13)), ts.STEP_COEFFICIENTS)

    def test_collar_boundary_precondition_is_enforced(self) -> None:
        current = self.flow * 0.01
        interior = ts.smooth_proposal(self.flow) * 0.01 * self.mask
        candidate, report = ts.certified_local_clip_candidate(current, interior, self.mask)
        self.assertEqual(current.shape, candidate.shape)
        self.assertIn("operator", report)
        with self.assertRaises(RuntimeError):
            ts.certified_local_clip_candidate(current, ts.smooth_proposal(self.flow) * 0.01, self.mask)

    def test_decision_snapshot_drops_only_post_decision_label_metrics(self) -> None:
        row = {key: index for index, key in enumerate(sorted(c1.POST_DECISION_EVALUATION_FIELDS))}
        row.update({"case_id": "subject_1", "candidate_id": "mind_global_k00"})
        stripped = c1._decision_input_row(dict(row))
        for field in c1.POST_DECISION_EVALUATION_FIELDS:
            self.assertNotIn(field, stripped, f"{field} must not be a decision input")
        self.assertEqual("subject_1", stripped["case_id"])


CLI_PROBE = r"""
import argparse, contextlib, io, json, sys
sys.path.insert(0, sys.argv[1])
import tools.analysis.run_search_gate_c0 as c0
import tools.analysis.run_search_gate_c1 as c1
import tools.analysis.run_search_gate_c2 as c2

out = {}
for name, module in (("c1", c1), ("c2", c2)):
    parser = module.build_parser()
    out[name] = parser.format_help()
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            for sub_name, sub in action.choices.items():
                out[name + "." + sub_name] = sub.format_help()
buffer = io.StringIO()
sys.argv = ["run_search_gate_c0", "--help"]
with contextlib.redirect_stdout(buffer), contextlib.suppress(SystemExit):
    c0.parse_args()
out["c0"] = buffer.getvalue()
json.dump(out, sys.stdout, sort_keys=True)
"""


@unittest.skipUnless(shutil.which("git"), "git not available")
class BaselineDifferentialTest(unittest.TestCase):
    """Re-derive the CLI differential against the frozen baseline commit on every run."""

    def _cli_surface(self, tree: Path) -> dict[str, str]:
        env = dict(os.environ, PYTHONPATH=str(tree), COLUMNS="100", PYTHONHASHSEED="0")
        result = subprocess.run(
            [os.sys.executable, "-c", CLI_PROBE, str(tree)],
            capture_output=True,
            cwd=tree,
            env=env,
        )
        self.assertEqual(0, result.returncode, result.stderr.decode("utf-8", "replace"))
        return json.loads(result.stdout.decode("utf-8"))

    def test_cli_surface_matches_the_frozen_baseline(self) -> None:
        probe = subprocess.run(
            ["git", "cat-file", "-e", f"{BASELINE_COMMIT}^{{commit}}"], cwd=REPO_ROOT, capture_output=True
        )
        if probe.returncode != 0:
            self.skipTest(f"baseline commit {BASELINE_COMMIT[:12]} not in this clone")
        with tempfile.TemporaryDirectory() as tmp:
            baseline = Path(tmp) / "baseline"
            baseline.mkdir()
            archive = subprocess.run(
                ["git", "archive", "--format=tar", BASELINE_COMMIT], cwd=REPO_ROOT, capture_output=True
            )
            self.assertEqual(0, archive.returncode, archive.stderr.decode("utf-8", "replace"))
            extract = subprocess.run(["tar", "-x", "-C", str(baseline)], input=archive.stdout, capture_output=True)
            self.assertEqual(0, extract.returncode, extract.stderr.decode("utf-8", "replace"))
            expected = self._cli_surface(baseline)
        observed = self._cli_surface(REPO_ROOT)
        self.assertEqual(11, len(expected))
        self.assertEqual(sorted(expected), sorted(observed))
        for name in sorted(expected):
            self.assertEqual(expected[name], observed[name], f"CLI surface '{name}' drifted from the baseline")


SHELL_HELPER_PROBE = r"""
set -uo pipefail
source "$1"
fail=0
chk() { if [[ "$2" == "$3" ]]; then :; else echo "FAIL $1: got '$2' want '$3'"; fail=1; fi; }

sg_parse_gpu_list "2,3,4,5,6"; chk plain_list "${GPUS[*]}" "2 3 4 5 6"
sg_parse_gpu_list " 0 , 1 ,2 "; chk whitespace "${GPUS[*]}" "0 1 2"
sg_parse_gpu_list "1,1,2";      chk duplicates_left_to_caller "${GPUS[*]}" "1 1 2"

out=$( (sg_parse_gpu_list "1,x,3") 2>&1 ); rc=$?
chk noninteger_exit "$rc" "2"
chk noninteger_message "$out" "[FAIL] GPU_LIST must be a comma-separated list of non-negative integers"

for good in C1_EXPLORATION_20260821T121202Z_9275171a67a3 A_20260821T121202Z_1800469 a.b-c_1; do
  sg_is_safe_identifier "$good" || { echo "FAIL rejects $good"; fail=1; }
done
for bad in "a b" "a/b" "" 'a$b' "a;b"; do
  if sg_is_safe_identifier "$bad"; then echo "FAIL accepts '$bad'"; fail=1; fi
done

PKG="/data/x/C2.tar.gz"
chk remote_locator "$(sg_default_remote_locator "$PKG")" "H100_LOCAL_ARCHIVE=${PKG};H100_LOCAL_SIDECAR=${PKG}.sha256"
[[ "$(sg_utc_run_stamp)" =~ ^[0-9]{8}T[0-9]{6}Z$ ]] || { echo "FAIL run stamp"; fail=1; }
[[ "$(sg_utc_started_at)" =~ ^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z$ ]] || { echo "FAIL started_at"; fail=1; }
( PYTHONPATH="/pre"; sg_export_pythonpath; chk pythonpath_append "$PYTHONPATH" "/pre:$(pwd)" )
( unset PYTHONPATH; sg_export_pythonpath; chk pythonpath_empty "$PYTHONPATH" "$(pwd)" )
exit $fail
"""


class ShellRunnerContractTest(unittest.TestCase):
    RUNNERS: ClassVar[dict[str, Path]] = {
        "c0": REPO_ROOT / "tools/runners/eval/search_gate_c0.sh",
        "c1": REPO_ROOT / "tools/runners/eval/search_gate_c1.sh",
        "c2": REPO_ROOT / "tools/runners/eval/search_gate_c2.sh",
    }
    EXPECTED_VARIABLES: ClassVar[dict[str, set[str]]] = {
        "c0": {
            "MODE",
            "GPU",
            "PYBIN",
            "PATHS_PROFILE",
            "SEED",
            "OUT_ROOT",
            "RUN_ID",
            "OAS_CKPT",
            "IXI_CKPT",
            "REMOTE_LOCATOR",
        },
        "c1": {
            "MODE",
            "GPU_LIST",
            "PYBIN",
            "PATHS_PROFILE",
            "SEED",
            "OUT_ROOT",
            "IXI_CKPT",
            "REMOTE_LOCATOR",
            "KEEP_FIELDS",
            "EXPLORE_MANIFEST",
            "EXPLORE_MANIFEST_SHA256",
            "RUN_ID",
            "ATTEMPT_ID",
        },
        "c2": {
            "GPU_LIST",
            "PYBIN",
            "PATHS_PROFILE",
            "SEED",
            "OUT_ROOT",
            "IXI_CKPT",
            "REMOTE_LOCATOR",
            "C1_EXPLORE_MANIFEST",
            "C1_EXPLORE_SHA256",
            "C1_CONFIRM_MANIFEST",
            "C1_CONFIRM_SHA256",
            "RUN_ID",
            "ATTEMPT_ID",
        },
    }

    def test_every_documented_knob_is_still_overridable(self) -> None:
        for gate, path in self.RUNNERS.items():
            source = path.read_text(encoding="utf-8")
            for variable in self.EXPECTED_VARIABLES[gate]:
                self.assertIn(f"{variable}:-", source, f"{path.name} no longer honours ${variable}")

    def test_c2_pins_the_frozen_c1_manifests_by_hash(self) -> None:
        source = self.RUNNERS["c2"].read_text(encoding="utf-8")
        self.assertIn(C1_EXPLORATION_MANIFEST_SHA256, source)
        self.assertIn(C1_CONFIRMATION_MANIFEST_SHA256, source)

    def test_runners_refuse_to_start_from_a_dirty_tree(self) -> None:
        for path in self.RUNNERS.values():
            self.assertIn("git status --porcelain=v1", path.read_text(encoding="utf-8"), path.name)

    def test_every_runner_sources_the_shared_shell_module(self) -> None:
        for path in self.RUNNERS.values():
            self.assertIn("_search_gate_common.sh", path.read_text(encoding="utf-8"), path.name)

    def test_c0_stays_single_process_and_c1_c2_fan_out(self) -> None:
        self.assertNotIn("GPU_LIST", self.RUNNERS["c0"].read_text(encoding="utf-8"))
        for gate in ("c1", "c2"):
            self.assertIn("GPU_LIST", self.RUNNERS[gate].read_text(encoding="utf-8"))


@unittest.skipUnless(shutil.which("bash"), "bash not available")
class ShellHelperEquivalenceTest(unittest.TestCase):
    """The runners drive multi-hour GPU jobs, so the extracted helpers are exercised directly."""

    def test_helpers_reproduce_the_inline_behaviour(self) -> None:
        self.assertTrue((REPO_ROOT / "tools/runners/eval/_search_gate_common.sh").is_file())
        result = subprocess.run(
            ["bash", "-s", "tools/runners/eval/_search_gate_common.sh"],
            input=SHELL_HELPER_PROBE.encode("utf-8"),
            capture_output=True,
            cwd=REPO_ROOT,
        )
        self.assertEqual(0, result.returncode, (result.stdout + result.stderr).decode("utf-8", "replace"))

    def test_gate_sources_use_lf_endings(self) -> None:
        names = [
            "tools/runners/eval/_search_gate_common.sh",
            "tools/runners/eval/search_gate_c0.sh",
            "tools/runners/eval/search_gate_c1.sh",
            "tools/runners/eval/search_gate_c2.sh",
            "tools/analysis/run_search_gate_c0.py",
            "tools/analysis/run_search_gate_c1.py",
            "tools/analysis/run_search_gate_c2.py",
            "tools/analysis/search_gate_common.py",
            "tools/analysis/search_gate_runtime.py",
        ]
        crlf = bytes((13, 10))
        offenders = [name for name in names if crlf in (REPO_ROOT / name).read_bytes()]
        self.assertEqual([], offenders, f"CRLF line endings found in: {offenders}")

    def test_runners_pass_a_syntax_check(self) -> None:
        for name in ("_search_gate_common.sh", "search_gate_c0.sh", "search_gate_c1.sh", "search_gate_c2.sh"):
            result = subprocess.run(["bash", "-n", f"tools/runners/eval/{name}"], capture_output=True, cwd=REPO_ROOT)
            self.assertEqual(0, result.returncode, result.stderr.decode("utf-8", "replace"))


if __name__ == "__main__":
    unittest.main()
