"""Contract for the declarative search-history registry and its read-only verifier.

Three things are pinned: the registry stays well formed and free of machine-local
paths, a valid product tree still verifies after being copied elsewhere, and every
tampering listed in the acceptance protocol fails on its own named check rather than
on an accidental earlier one.

Nothing here needs a GPU, a dataset or a checkpoint. Everything under `results/` is
read-only; every mutation happens on a copy in a temporary directory.

Set CTCF_REQUIRE_HISTORICAL_GOLDENS=1 to turn a missing historical product into a
failure instead of a skip.
"""

from __future__ import annotations

import ast
import copy
import hashlib
import inspect
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from tools.analysis.search_history import verify as sh
from tools.analysis.search_history.__main__ import _locate, main

REPO_ROOT = Path(__file__).resolve().parents[3]
RESULTS = REPO_ROOT / "results"
PACKAGE = REPO_ROOT / "tools" / "analysis" / "search_history"
REQUIRE_HISTORICAL = os.environ.get("CTCF_REQUIRE_HISTORICAL_GOLDENS") == "1"

FREEZE_COMMIT = "f3119f7f2f93147ca1bb3aed7340db93b2bb5079"
C5_PRODUCER_HEAD = "242dde3281d22c573a4a64bc494c8d2e5ef597b2"
C5_CONSUMER_HEAD = "baafc3bcfed74ed3e471222390e3a604816ae4a3"
MIND_REFERENCE_COMMIT = "b229e52e44b114e2040a503334c92269750c16b2"

# Acceptance totals for the eleven canonical products; the parity witness is counted apart.
CANONICAL_INDEX_ROWS = 1795
CANONICAL_FILES = 1819
CANONICAL_BYTES = 161175357
PARITY_INDEX_ROWS = 212
PARITY_FILES = 214
PARITY_BYTES = 7898999

FORBIDDEN_IMPORT_PREFIXES = ("run_search_gate", "search_gate_c", "search_gate_common", "search_gate_runtime")
FORBIDDEN_IMPORT_ROOTS = ("torch", "numpy", "scipy", "pandas", "matplotlib")
HEAD_DEPENDENT_GIT_SUBCOMMANDS = ("status", "rev-parse", "symbolic-ref", "describe", "branch", "log", "diff")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, document: Any) -> None:
    path.write_bytes((json.dumps(document, indent=2, sort_keys=True) + "\n").encode("utf-8"))


def _entry(registry: Mapping[str, Any], product_id: str) -> dict[str, Any]:
    return sh.product_by_id(registry, product_id)  # type: ignore[return-value]


class SearchHistoryTestCase(unittest.TestCase):
    registry: dict[str, Any]

    @classmethod
    def setUpClass(cls) -> None:
        cls.registry = sh.load_registry()

    def product_root(self, product_id: str) -> Path:
        hint = _entry(self.registry, product_id)["relative_hints"][0]
        root = RESULTS / hint
        if not (root / "run_manifest.json").is_file():
            message = f"historical product {product_id} is absent at {hint}"
            if REQUIRE_HISTORICAL:
                self.fail(message)
            self.skipTest(message)
        return root

    def copy_product(self, product_id: str) -> Path:
        source = self.product_root(product_id)
        temporary = Path(tempfile.mkdtemp(prefix="ctcf_search_history_"))
        self.addCleanup(shutil.rmtree, temporary, ignore_errors=True)
        destination = temporary / "product"
        shutil.copytree(source, destination)
        return destination

    def working_registry(self) -> dict[str, Any]:
        return copy.deepcopy(self.registry)

    def reseal_envelope(self, root: Path, registry: dict[str, Any], product_id: str) -> None:
        """Re-derive the index and the manifest so an inner mutation reaches its own check."""
        index_path = root / "outputs.tsv"
        lines = index_path.read_text(encoding="utf-8").split("\n")
        rebuilt = [lines[0]]
        for line in lines[1:]:
            if line == "":
                continue
            relative = line.split("\t")[0]
            target = root / relative
            rebuilt.append(f"{relative}\t{target.stat().st_size}\t{_sha256(target)}")
        index_path.write_bytes(("\n".join(rebuilt) + "\n").encode("utf-8"))

        manifest_path = root / "run_manifest.json"
        manifest = _read_json(manifest_path)
        manifest["files"]["outputs_sha256"] = _sha256(index_path)
        _write_json(manifest_path, manifest)

        entry = _entry(registry, product_id)
        entry["manifest"]["sha256"] = _sha256(manifest_path)
        for item in entry["unindexed_files"]:
            item["sha256"] = _sha256(root / item["path"])

    def reseal_full(self, root: Path, registry: dict[str, Any], product_id: str) -> None:
        """Also refresh every pinned file hash, leaving only the semantic checks armed."""
        entry = _entry(registry, product_id)
        for group in ("assertions", "test_isolation", "source_links"):
            for item in entry[group]:
                if item["kind"] == "file_sha256":
                    item["sha256"] = _sha256(root / item["file"])
        self.reseal_envelope(root, registry, product_id)

    def reseal_chains(self, root: Path, registry: dict[str, Any], product_id: str) -> None:
        """Model a forger who also re-signs the contract chain, so only the semantics stay armed."""
        chains = [
            item for item in _entry(registry, product_id)["assertions"] if item["kind"] == "json_equals_file_sha256"
        ]
        for _pass in range(len(chains) + 1):
            changed = False
            for item in chains:
                path = root / item["file"]
                document = _read_json(path)
                node = document
                tokens = item["pointer"].split("/")[1:]
                for token in tokens[:-1]:
                    node = node[token]
                digest = _sha256(root / item["target_file"])
                if node[tokens[-1]] != digest:
                    node[tokens[-1]] = digest
                    _write_json(path, document)
                    changed = True
            if not changed:
                break
        self.reseal_full(root, registry, product_id)

    def assert_rejected(
        self,
        product_id: str,
        mutate: Callable[[Path, dict[str, Any]], None],
        expected_check: str,
        expected_category: str = "INVALID_PRODUCT",
    ) -> sh.VerifierError:
        root = self.copy_product(product_id)
        registry = self.working_registry()
        mutate(root, registry)
        with self.assertRaises(sh.VerifierError) as captured:
            sh.verify_product(registry, product_id, root, repo_root=REPO_ROOT)
        error = captured.exception
        self.assertEqual(error.check, expected_check, msg=f"unexpected check for {product_id}: {error}")
        self.assertEqual(error.category, expected_category, msg=str(error))
        return error


class RegistryShapeTest(SearchHistoryTestCase):
    def test_registry_declares_eleven_canonical_products_and_one_supporting_witness(self) -> None:
        roles = [entry["role"] for entry in self.registry["products"]]
        self.assertEqual(roles.count("CANONICAL"), 11)
        self.assertEqual(roles.count("SUPPORTING"), 1)
        self.assertEqual(_entry(self.registry, "SG-C2-SCIENCE")["role"], "CANONICAL")
        self.assertEqual(_entry(self.registry, "SG-C2-PARITY")["role"], "SUPPORTING")

    def test_registry_freeze_commit_is_the_declared_freeze(self) -> None:
        self.assertEqual(self.registry["freeze_commit"], FREEZE_COMMIT)

    def test_registry_stores_no_absolute_path_hostname_or_user_profile(self) -> None:
        text = (PACKAGE / "registry.v1.json").read_text(encoding="utf-8")
        for forbidden in ("C:", "c:", "\\\\", "/data/", "/home/", "Users", "g593-zd2", "mooncake"):
            self.assertNotIn(forbidden, text, msg=f"registry leaks {forbidden!r}")

    def test_registry_hint_for_c5b_keeps_the_doubled_nested_path(self) -> None:
        hint = _entry(self.registry, "SG-C5B")["relative_hints"][0]
        segments = hint.split("/")
        self.assertEqual(segments[0], "search_gate_c5b")
        self.assertEqual(segments[1], segments[2], msg="the C5b hint must carry the doubled run directory")

    def test_registry_pins_both_c5_heads_with_explicit_roles(self) -> None:
        heads = {head["role"]: head["git_head"] for head in _entry(self.registry, "SG-C5")["code_heads"]}
        self.assertEqual(heads["DECISION_PRODUCER"], C5_PRODUCER_HEAD)
        self.assertEqual(heads["EVALUATION_CONSUMER"], C5_CONSUMER_HEAD)
        self.assertEqual(heads["RUN_MANIFEST"], C5_CONSUMER_HEAD)

    def test_registry_treats_the_mind_commit_as_external_and_never_as_a_ctcf_commit(self) -> None:
        external = {item["commit"]: item for item in self.registry["external_reference_commits"]}
        self.assertIn(MIND_REFERENCE_COMMIT, external)
        self.assertEqual(external[MIND_REFERENCE_COMMIT]["repository"], "EXTERNAL_NOT_CTCF")
        for entry in self.registry["products"]:
            declared = {head["git_head"] for head in entry["code_heads"]}
            declared |= {item["git_head"] for item in entry["entrypoints"]}
            self.assertNotIn(MIND_REFERENCE_COMMIT, declared)

    def test_registry_prescribes_the_unindexed_allowance_of_each_product(self) -> None:
        for entry in self.registry["products"]:
            names = sorted(item["path"] for item in entry["unindexed_files"])
            if entry["id"] == "SG-C0":
                expected = [
                    "development/run_manifest.json",
                    "outputs.tsv",
                    "run_manifest.json",
                    "smoke/run_manifest.json",
                ]
            else:
                expected = ["outputs.tsv", "run_manifest.json"]
            self.assertEqual(names, expected, msg=entry["id"])

    def test_registry_schema_admits_a_later_failed_product_without_a_verifier_change(self) -> None:
        self.assertIn("FAILED", sh.PRODUCT_STATUSES)
        registry = self.working_registry()
        diagnostic = copy.deepcopy(_entry(registry, "SG-C0"))
        diagnostic.update(
            {
                "id": "SG-DIAGNOSTIC-FAILED",
                "run_id": "C0_00000000T000000Z_000000000000",
                "relative_hints": ["search_gate_c0/C0_00000000T000000Z_000000000000"],
            }
        )
        diagnostic["manifest"] = {**diagnostic["manifest"], "status": "FAILED", "exit_code": 1}
        registry["products"].append(diagnostic)
        sh._validate_products(registry["products"])


class RegistryRejectionTest(SearchHistoryTestCase):
    def _reject(self, mutate: Callable[[dict[str, Any]], None], expected_check: str) -> None:
        registry = self.working_registry()
        mutate(registry)
        with self.assertRaises(sh.InvalidRegistryError) as captured:
            sh._validate_products(registry["products"])
        self.assertEqual(captured.exception.check, expected_check)

    def _reject_entry(self, product_id: str, mutate: Callable[[dict[str, Any]], None], expected_check: str) -> None:
        registry = self.working_registry()
        mutate(_entry(registry, product_id))
        with self.assertRaises(sh.InvalidRegistryError) as captured:
            sh._validate_products(registry["products"])
        self.assertEqual(captured.exception.check, expected_check)

    def test_duplicate_product_id_is_rejected(self) -> None:
        self._reject(lambda r: r["products"].append(copy.deepcopy(_entry(r, "SG-C0"))), "registry_unique_id")

    def test_duplicate_run_id_is_rejected(self) -> None:
        def mutate(registry: dict[str, Any]) -> None:
            clone = copy.deepcopy(_entry(registry, "SG-C0"))
            clone["id"] = "SG-C0-CLONE"
            clone["relative_hints"] = ["search_gate_c0/other"]
            registry["products"].append(clone)

        self._reject(mutate, "registry_unique_run_id")

    def test_duplicate_relative_hint_is_rejected(self) -> None:
        def mutate(registry: dict[str, Any]) -> None:
            clone = copy.deepcopy(_entry(registry, "SG-C0"))
            clone["id"] = "SG-C0-CLONE"
            clone["run_id"] = "C0_OTHER"
            registry["products"].append(clone)

        self._reject(mutate, "registry_unique_hint")

    def test_source_link_naming_an_unknown_product_is_rejected(self) -> None:
        registry = self.working_registry()
        _entry(registry, "SG-C4")["source_links"][0]["source_product_id"] = "SG-NOWHERE"
        with self.assertRaises(sh.InvalidRegistryError) as captured:
            sh._validate_source_link_targets(registry["products"], {e["id"] for e in registry["products"]})
        self.assertEqual(captured.exception.check, "registry_source_link_id")

    def test_source_link_pointing_at_its_own_product_is_rejected(self) -> None:
        registry = self.working_registry()
        _entry(registry, "SG-C4")["source_links"][0]["source_product_id"] = "SG-C4"
        with self.assertRaises(sh.InvalidRegistryError) as captured:
            sh._validate_source_link_targets(registry["products"], {e["id"] for e in registry["products"]})
        self.assertEqual(captured.exception.check, "registry_source_link_id")

    def test_deleting_any_evidence_group_is_rejected(self) -> None:
        for group in ("code_heads", "entrypoints", "unindexed_files", *sh.EVIDENCE_GROUPS):
            self._reject_entry("SG-C7", lambda e, g=group: e.pop(g), "registry_required_key")

    def test_emptying_any_evidence_group_is_rejected(self) -> None:
        for group in ("code_heads", "entrypoints", "unindexed_files", *sh.EVIDENCE_GROUPS):
            self._reject_entry("SG-C7", lambda e, g=group: e.__setitem__(g, []), "registry_required_key")

    def test_an_undeclared_gap_cannot_be_smuggled_in_by_editing_evidence_gaps(self) -> None:
        # Emptying a group and declaring the gap is allowed only where the entry legitimately has none.
        self._reject_entry(
            "SG-C7",
            lambda e: (e.__setitem__("source_links", []), e.__setitem__("evidence_gaps", ["assertions"]))[0],
            "registry_schema",
        )

    def test_a_populated_group_declared_as_a_gap_is_rejected(self) -> None:
        self._reject_entry("SG-C7", lambda e: e["evidence_gaps"].append("source_links"), "registry_required_key")

    def test_c0_declares_its_two_genuine_evidence_gaps(self) -> None:
        self.assertEqual(_entry(self.registry, "SG-C0")["evidence_gaps"], ["test_isolation", "source_links"])
        self.assertEqual(_entry(self.registry, "SG-C1-EXPLORATION")["evidence_gaps"], ["source_links"])
        for product_id in ("SG-C3", "SG-C1-CONFIRMATION"):
            self.assertEqual(_entry(self.registry, product_id)["evidence_gaps"], [])

    def test_duplicate_code_head_role_is_rejected(self) -> None:
        self._reject_entry(
            "SG-C5",
            lambda e: e["code_heads"].append(copy.deepcopy(e["code_heads"][0])),
            "registry_unique_role",
        )

    def test_duplicate_entrypoint_is_rejected(self) -> None:
        self._reject_entry(
            "SG-C7",
            lambda e: e["entrypoints"].append(copy.deepcopy(e["entrypoints"][0])),
            "registry_unique_entrypoint",
        )

    def test_duplicate_assertion_check_name_is_rejected(self) -> None:
        self._reject_entry(
            "SG-C7",
            lambda e: e["assertions"].append(copy.deepcopy(e["assertions"][0])),
            "registry_unique_check",
        )

    def test_a_check_name_reused_across_two_groups_is_rejected(self) -> None:
        self._reject_entry(
            "SG-C7",
            lambda e: e["test_isolation"].append(copy.deepcopy(e["assertions"][0])),
            "registry_unique_check",
        )

    def _reject_frozen_gaps(self, mutate: Callable[[dict[str, Any]], None]) -> None:
        registry = self.working_registry()
        mutate(registry)
        with self.assertRaises(sh.InvalidRegistryError) as captured:
            sh._validate_frozen_evidence_gaps(registry["products"])
        self.assertEqual(captured.exception.check, "registry_frozen_evidence_gaps")

    def test_the_frozen_evidence_gap_map_is_exactly_the_two_declared_entries(self) -> None:
        self.assertEqual(
            dict(sh.FROZEN_EVIDENCE_GAPS),
            {"SG-C0": ("test_isolation", "source_links"), "SG-C1-EXPLORATION": ("source_links",)},
        )
        for entry in self.registry["products"]:
            expected = list(sh.FROZEN_EVIDENCE_GAPS.get(entry["id"], ()))
            self.assertEqual(entry["evidence_gaps"], expected, msg=entry["id"])
        gapped = [entry["id"] for entry in self.registry["products"] if entry["evidence_gaps"]]
        self.assertEqual(gapped, ["SG-C0", "SG-C1-EXPLORATION"])

    def test_opening_a_new_gap_on_another_product_is_rejected(self) -> None:
        # The structural pass accepts an empty group that declares its gap; the frozen map still refuses.
        def mutate(registry: dict[str, Any]) -> None:
            entry = _entry(registry, "SG-C7")
            entry["source_links"] = []
            entry["evidence_gaps"] = ["source_links"]

        registry = self.working_registry()
        mutate(registry)
        sh._validate_products(registry["products"])
        self._reject_frozen_gaps(mutate)

    def test_the_c7_gap_mutation_is_rejected_end_to_end_as_invalid_registry(self) -> None:
        staged = Path(tempfile.mkdtemp(prefix="ctcf_search_history_gap_"))
        self.addCleanup(shutil.rmtree, staged, ignore_errors=True)
        registry = self.working_registry()
        entry = _entry(registry, "SG-C7")
        entry["source_links"] = []
        entry["evidence_gaps"] = ["source_links"]
        path = staged / "registry.json"
        path.write_bytes(json.dumps(registry).encode("utf-8"))
        with self.assertRaises(sh.InvalidRegistryError) as captured:
            sh.load_registry(path)
        self.assertEqual(captured.exception.check, "registry_frozen_evidence_gaps")
        self.assertEqual(captured.exception.category, "INVALID_REGISTRY")

    def test_closing_a_genuine_gap_by_declaration_alone_is_rejected(self) -> None:
        self._reject_frozen_gaps(lambda r: _entry(r, "SG-C0").__setitem__("evidence_gaps", ["test_isolation"]))

    def test_reordering_the_frozen_gap_list_is_rejected(self) -> None:
        self._reject_frozen_gaps(
            lambda r: _entry(r, "SG-C0").__setitem__("evidence_gaps", ["source_links", "test_isolation"])
        )

    def test_a_frozen_gap_naming_an_absent_product_is_rejected(self) -> None:
        registry = self.working_registry()
        registry["products"] = [e for e in registry["products"] if e["id"] != "SG-C0"]
        with self.assertRaises(sh.InvalidRegistryError) as captured:
            sh._validate_frozen_evidence_gaps(registry["products"])
        self.assertEqual(captured.exception.check, "registry_frozen_evidence_gaps")

    def _reject_frozen_relation(self, mutate: Callable[[list[dict[str, Any]]], None]) -> None:
        registry = self.working_registry()
        mutate(registry["relations"])
        with self.assertRaises(sh.InvalidRegistryError) as captured:
            sh._validate_frozen_relations(registry["relations"])
        self.assertEqual(captured.exception.check, "registry_frozen_relations")

    def test_the_frozen_relation_contract_matches_the_registry(self) -> None:
        self.assertEqual(len(self.registry["relations"]), 1)
        relation = self.registry["relations"][0]
        self.assertEqual(relation["type"], sh.FROZEN_RELATION_TYPE)
        self.assertEqual(relation["from_id"], sh.FROZEN_RELATION_FROM_ID)
        self.assertEqual(relation["to_id"], sh.FROZEN_RELATION_TO_ID)
        self.assertEqual(tuple(relation["files"]), sh.FROZEN_RELATION_FILES)
        self.assertEqual(len(sh.FROZEN_RELATION_FILES), 9)

    def test_an_absent_relations_key_is_rejected(self) -> None:
        staged = Path(tempfile.mkdtemp(prefix="ctcf_search_history_rel_"))
        self.addCleanup(shutil.rmtree, staged, ignore_errors=True)
        registry = self.working_registry()
        del registry["relations"]
        path = staged / "registry.json"
        path.write_bytes(json.dumps(registry).encode("utf-8"))
        with self.assertRaises(sh.InvalidRegistryError) as captured:
            sh.load_registry(path)
        self.assertEqual(captured.exception.check, "registry_required_key")

    def test_an_empty_relation_list_is_rejected(self) -> None:
        self._reject_frozen_relation(lambda relations: relations.clear())

    def test_a_duplicated_relation_is_rejected(self) -> None:
        self._reject_frozen_relation(lambda relations: relations.append(copy.deepcopy(relations[0])))

    def test_a_wrong_relation_type_is_rejected(self) -> None:
        self._reject_frozen_relation(lambda relations: relations[0].__setitem__("type", "SUPERSEDES"))

    def test_a_reversed_relation_direction_is_rejected(self) -> None:
        def mutate(relations: list[dict[str, Any]]) -> None:
            relations[0]["from_id"], relations[0]["to_id"] = relations[0]["to_id"], relations[0]["from_id"]

        self._reject_frozen_relation(mutate)

    def test_promoting_the_parity_run_by_renaming_the_relation_side_is_rejected(self) -> None:
        self._reject_frozen_relation(lambda relations: relations[0].__setitem__("to_id", "SG-C3"))

    def test_a_removed_relation_file_is_rejected(self) -> None:
        self._reject_frozen_relation(lambda relations: relations[0]["files"].remove("summary.json"))

    def test_a_substituted_relation_file_is_rejected(self) -> None:
        def mutate(relations: list[dict[str, Any]]) -> None:
            relations[0]["files"][relations[0]["files"].index("per_step.csv")] = "per_case.csv"

        self._reject_frozen_relation(mutate)

    def test_an_added_relation_file_is_rejected(self) -> None:
        self._reject_frozen_relation(lambda relations: relations[0]["files"].append("per_case.csv"))

    def test_reordering_the_relation_files_is_rejected(self) -> None:
        self._reject_frozen_relation(lambda relations: relations[0]["files"].reverse())

    def test_duplicate_relation_file_is_rejected(self) -> None:
        registry = self.working_registry()
        registry["relations"][0]["files"].append(registry["relations"][0]["files"][0])
        with self.assertRaises(sh.InvalidRegistryError) as captured:
            sh._validate_relations(registry["relations"], {e["id"] for e in registry["products"]})
        self.assertEqual(captured.exception.check, "registry_unique_relation_file")

    def test_a_relation_that_relates_a_product_to_itself_is_rejected(self) -> None:
        registry = self.working_registry()
        registry["relations"][0]["from_id"] = registry["relations"][0]["to_id"]
        with self.assertRaises(sh.InvalidRegistryError) as captured:
            sh._validate_relations(registry["relations"], {e["id"] for e in registry["products"]})
        self.assertEqual(captured.exception.check, "registry_relation_id")

    def test_registry_rejects_a_duplicate_json_key(self) -> None:
        broken = Path(tempfile.mkdtemp(prefix="ctcf_search_history_"))
        self.addCleanup(shutil.rmtree, broken, ignore_errors=True)
        path = broken / "registry.json"
        path.write_bytes(b'{"schema": "a", "schema": "b"}')
        with self.assertRaises(sh.InvalidRegistryError) as captured:
            sh.load_registry(path)
        self.assertEqual(captured.exception.check, "registry_json")

    def test_registry_rejects_a_non_json_constant(self) -> None:
        broken = Path(tempfile.mkdtemp(prefix="ctcf_search_history_"))
        self.addCleanup(shutil.rmtree, broken, ignore_errors=True)
        path = broken / "registry.json"
        path.write_bytes(b'{"schema": NaN}')
        with self.assertRaises(sh.InvalidRegistryError) as captured:
            sh.load_registry(path)
        self.assertEqual(captured.exception.check, "registry_json")


class PositiveControlTest(SearchHistoryTestCase):
    def test_a_copy_of_a_valid_product_verifies_in_another_directory(self) -> None:
        root = self.copy_product("SG-C0")
        report = sh.verify_product(self.registry, "SG-C0", root, repo_root=REPO_ROOT)
        self.assertEqual(report["result"], "PASS")
        self.assertEqual(report["total_files"], 121)
        self.assertEqual(report["unindexed_files"], 4)
        self.assertEqual(report["verification_scope"], "COMPACT_PRODUCT_BYTES_AND_RECORDED_PROVENANCE")

    def test_every_success_limits_its_own_strength(self) -> None:
        root = self.copy_product("SG-C0")
        report = sh.verify_product(self.registry, "SG-C0", root, repo_root=REPO_ROOT)
        self.assertEqual(report["verification_scope"], sh.VERIFICATION_SCOPE)
        joined = " ".join(report["scope_exclusions"])
        for excluded in ("GPU", "heavy roots", "checkpoint", "scientific correctness"):
            self.assertIn(excluded, joined)

    def test_the_parity_witness_verifies_without_becoming_a_canonical_owner(self) -> None:
        root = self.copy_product("SG-C2-PARITY")
        report = sh.verify_product(self.registry, "SG-C2-PARITY", root, repo_root=REPO_ROOT)
        self.assertEqual(report["result"], "PASS")
        self.assertEqual(report["role"], "SUPPORTING")
        self.assertEqual(
            (report["indexed_files"], report["total_files"], report["total_bytes"]),
            (PARITY_INDEX_ROWS, PARITY_FILES, PARITY_BYTES),
        )

    def test_the_nine_payload_files_are_equal_across_both_c2_trees(self) -> None:
        roots = {
            "SG-C2-SCIENCE": self.product_root("SG-C2-SCIENCE"),
            "SG-C2-PARITY": self.product_root("SG-C2-PARITY"),
        }
        relation = self.registry["relations"][0]
        self.assertEqual(relation["type"], "EQUIVALENT_SCIENTIFIC_PAYLOAD_TO")
        self.assertEqual(len(relation["files"]), 9)
        report = sh.verify_relation(self.registry, relation, roots)
        self.assertEqual(report["result"], "PASS")
        self.assertEqual(len(report["files"]), 9)


class ProductRejectionTest(SearchHistoryTestCase):
    def test_absent_product_root(self) -> None:
        missing = Path(tempfile.mkdtemp(prefix="ctcf_search_history_")) / "nowhere"
        with self.assertRaises(sh.MissingProductError) as captured:
            sh.verify_product(self.registry, "SG-C0", missing, repo_root=REPO_ROOT)
        self.assertEqual(captured.exception.check, "product_root")

    def test_mutated_indexed_file(self) -> None:
        def mutate(root: Path, _registry: dict[str, Any]) -> None:
            target = root / "development/cases/subject_126/case_complete.json"
            payload = bytearray(target.read_bytes())
            payload[-2] = payload[-2] ^ 0x01  # size-preserving, so the digest check is the one that fires
            target.write_bytes(bytes(payload))

        self.assert_rejected("SG-C0", mutate, "indexed_file_sha256")

    def test_mutated_size_in_the_index(self) -> None:
        def mutate(root: Path, registry: dict[str, Any]) -> None:
            index = root / "outputs.tsv"
            lines = index.read_text(encoding="utf-8").split("\n")
            fields = lines[1].split("\t")
            lines[1] = "\t".join([fields[0], str(int(fields[1]) + 1), fields[2]])
            index.write_bytes("\n".join(lines).encode("utf-8"))
            self.reseal_envelope_without_index(root, registry, "SG-C0")

        self.assert_rejected("SG-C0", mutate, "indexed_file_bytes")

    def test_mutated_sha_in_the_index(self) -> None:
        def mutate(root: Path, registry: dict[str, Any]) -> None:
            index = root / "outputs.tsv"
            lines = index.read_text(encoding="utf-8").split("\n")
            fields = lines[1].split("\t")
            lines[1] = "\t".join([fields[0], fields[1], "0" * 64])
            index.write_bytes("\n".join(lines).encode("utf-8"))
            self.reseal_envelope_without_index(root, registry, "SG-C0")

        self.assert_rejected("SG-C0", mutate, "indexed_file_sha256")

    def reseal_envelope_without_index(self, root: Path, registry: dict[str, Any], product_id: str) -> None:
        """Re-authenticate a deliberately edited index so the per-file checks are reached."""
        manifest_path = root / "run_manifest.json"
        manifest = _read_json(manifest_path)
        manifest["files"]["outputs_sha256"] = _sha256(root / "outputs.tsv")
        _write_json(manifest_path, manifest)
        entry = _entry(registry, product_id)
        entry["manifest"]["sha256"] = _sha256(manifest_path)
        for item in entry["unindexed_files"]:
            item["sha256"] = _sha256(root / item["path"])

    def test_a_float_exit_code_is_not_accepted_as_the_integer_zero(self) -> None:
        def mutate(root: Path, registry: dict[str, Any]) -> None:
            path = root / "run_manifest.json"
            document = _read_json(path)
            document["exit_code"] = 0.0
            _write_json(path, document)
            self.reseal_envelope(root, registry, "SG-C0")

        error = self.assert_rejected("SG-C0", mutate, "manifest_exit_code")
        self.assertIn("0.0", error.detail)

    def test_a_boolean_exit_code_is_not_accepted_as_the_integer_zero(self) -> None:
        def mutate(root: Path, registry: dict[str, Any]) -> None:
            path = root / "run_manifest.json"
            document = _read_json(path)
            document["exit_code"] = False
            _write_json(path, document)
            self.reseal_envelope(root, registry, "SG-C0")

        self.assert_rejected("SG-C0", mutate, "manifest_exit_code")

    def test_a_product_root_that_is_a_reparse_point_is_rejected(self) -> None:
        root = self.copy_product("SG-C0")
        original = sh.os.lstat
        reparse = getattr(sh.stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)

        class _LinkLike:
            def __init__(self, borrowed: os.stat_result) -> None:
                self.st_mode = sh.stat.S_IFLNK | 0o777 if reparse == 0 else borrowed.st_mode
                self.st_file_attributes = reparse

        def patched(path: Any) -> Any:
            entry = original(path)
            return _LinkLike(entry) if Path(path) == root else entry

        sh.os.lstat = patched
        self.addCleanup(setattr, sh.os, "lstat", original)
        with self.assertRaises(sh.InvalidProductError) as captured:
            sh.verify_product(self.registry, "SG-C0", root, repo_root=REPO_ROOT)
        self.assertEqual(captured.exception.check, "product_root_is_plain")

    def test_mutated_manifest_hash(self) -> None:
        def mutate(_root: Path, registry: dict[str, Any]) -> None:
            _entry(registry, "SG-C0")["manifest"]["sha256"] = "0" * 64

        self.assert_rejected("SG-C0", mutate, "run_manifest_sha256")

    def test_mutated_run_id(self) -> None:
        def mutate(_root: Path, registry: dict[str, Any]) -> None:
            _entry(registry, "SG-C0")["run_id"] = "C0_WRONG"

        self.assert_rejected("SG-C0", mutate, "manifest_run_id")

    def test_mutated_code_head(self) -> None:
        def mutate(_root: Path, registry: dict[str, Any]) -> None:
            _entry(registry, "SG-C0")["manifest"]["code_git_head"] = "0" * 40

        self.assert_rejected("SG-C0", mutate, "manifest_code_git_head")

    def test_mutated_status(self) -> None:
        def mutate(_root: Path, registry: dict[str, Any]) -> None:
            _entry(registry, "SG-C0")["manifest"]["status"] = "FAILED"

        self.assert_rejected("SG-C0", mutate, "manifest_status")

    def test_mutated_exit_code(self) -> None:
        def mutate(_root: Path, registry: dict[str, Any]) -> None:
            _entry(registry, "SG-C0")["manifest"]["exit_code"] = 1

        self.assert_rejected("SG-C0", mutate, "manifest_exit_code")

    def test_duplicate_json_key_in_the_manifest(self) -> None:
        def mutate(root: Path, _registry: dict[str, Any]) -> None:
            path = root / "run_manifest.json"
            path.write_bytes(path.read_bytes().replace(b'"exit_code": 0,', b'"exit_code": 0,\n  "exit_code": 0,', 1))

        self.assert_rejected("SG-C0", mutate, "run_manifest_json")

    def test_non_json_constant_in_a_referenced_document(self) -> None:
        def mutate(root: Path, registry: dict[str, Any]) -> None:
            path = root / "selfcheck.json"
            document = _read_json(path)
            document["status"] = "PASS"
            path.write_bytes(json.dumps(document).replace('"PASS"', "NaN", 1).encode("utf-8"))
            self.reseal_full(root, registry, "SG-C0")

        self.assert_rejected("SG-C0", mutate, "referenced_json_strict")

    def test_repeated_path_in_the_index(self) -> None:
        def mutate(root: Path, registry: dict[str, Any]) -> None:
            index = root / "outputs.tsv"
            lines = index.read_text(encoding="utf-8").split("\n")
            index.write_bytes("\n".join([*lines[:2], lines[1], *lines[2:]]).encode("utf-8"))
            self.reseal_envelope_without_index(root, registry, "SG-C0")

        self.assert_rejected("SG-C0", mutate, "outputs_index_format")

    def test_repeated_header_in_the_index(self) -> None:
        def mutate(root: Path, registry: dict[str, Any]) -> None:
            index = root / "outputs.tsv"
            lines = index.read_text(encoding="utf-8").split("\n")
            index.write_bytes("\n".join([lines[0], lines[0], *lines[1:]]).encode("utf-8"))
            self.reseal_envelope_without_index(root, registry, "SG-C0")

        self.assert_rejected("SG-C0", mutate, "outputs_index_format")

    def _index_path_rejection(self, replacement: str) -> None:
        def mutate(root: Path, registry: dict[str, Any]) -> None:
            index = root / "outputs.tsv"
            lines = index.read_text(encoding="utf-8").split("\n")
            fields = lines[1].split("\t")
            lines[1] = "\t".join([replacement, fields[1], fields[2]])
            index.write_bytes("\n".join(lines).encode("utf-8"))
            self.reseal_envelope_without_index(root, registry, "SG-C0")

        self.assert_rejected("SG-C0", mutate, "outputs_index_format")

    def test_posix_absolute_path_in_the_index(self) -> None:
        self._index_path_rejection("/etc/passwd")

    def test_windows_drive_path_in_the_index(self) -> None:
        self._index_path_rejection("C:/Windows/system.ini")

    def test_unc_path_in_the_index(self) -> None:
        self._index_path_rejection("//host/share/file.json")

    def test_backslash_path_in_the_index(self) -> None:
        self._index_path_rejection("development\\cases\\x.json")

    def test_parent_escape_in_the_index(self) -> None:
        self._index_path_rejection("../outside.json")

    def test_casefold_collision_in_the_index(self) -> None:
        def mutate(root: Path, registry: dict[str, Any]) -> None:
            index = root / "outputs.tsv"
            lines = index.read_text(encoding="utf-8").split("\n")
            fields = lines[1].split("\t")
            collision = "\t".join([fields[0].upper(), fields[1], fields[2]])
            index.write_bytes("\n".join([*lines[:2], collision, *lines[2:]]).encode("utf-8"))
            self.reseal_envelope_without_index(root, registry, "SG-C0")

        self.assert_rejected("SG-C0", mutate, "outputs_index_format")

    def test_extra_unindexed_file(self) -> None:
        def mutate(root: Path, registry: dict[str, Any]) -> None:
            (root / "development" / "stray.json").write_bytes(b"{}\n")
            self.reseal_envelope(root, registry, "SG-C0")

        self.assert_rejected("SG-C0", mutate, "tree_closure_extra")

    def test_absent_indexed_file(self) -> None:
        def mutate(root: Path, _registry: dict[str, Any]) -> None:
            (root / "development/cases/subject_126/case_complete.json").unlink()

        self.assert_rejected("SG-C0", mutate, "tree_closure_missing")

    def test_mutated_selfcheck_hash(self) -> None:
        def mutate(root: Path, registry: dict[str, Any]) -> None:
            path = root / "c7_selfcheck.json"
            document = _read_json(path)
            document["candidate_moment_reduction"] = "tampered"
            _write_json(path, document)
            self.reseal_envelope(root, registry, "SG-C7")

        self.assert_rejected("SG-C7", mutate, "sha256::c7_selfcheck.json")

    def test_mutated_policy_hash(self) -> None:
        def mutate(root: Path, registry: dict[str, Any]) -> None:
            path = root / "c7_selfcheck.json"
            document = _read_json(path)
            document["policy_sha256"] = "0" * 64
            _write_json(path, document)
            self.reseal_full(root, registry, "SG-C7")

        self.assert_rejected("SG-C7", mutate, "equals::c7_selfcheck.json#policy_sha256")

    def test_broken_contract_chain(self) -> None:
        def mutate(root: Path, registry: dict[str, Any]) -> None:
            path = root / "decision_contract.json"
            document = _read_json(path)
            document["protocol_id"] = "TAMPERED"
            _write_json(path, document)
            self.reseal_full(root, registry, "SG-C7")

        self.assert_rejected(
            "SG-C7", mutate, "chain::decision_barrier.json#decision_contract_sha256->decision_contract.json"
        )

    def test_test_115_authorized_flipped_to_true(self) -> None:
        def mutate(root: Path, registry: dict[str, Any]) -> None:
            path = root / "c7_selfcheck.json"
            document = _read_json(path)
            document["test_115_authorized"] = True
            _write_json(path, document)
            self.reseal_full(root, registry, "SG-C7")

        self.assert_rejected("SG-C7", mutate, "equals::c7_selfcheck.json#test_115_authorized")

    def test_test_split_accessed_flipped_to_true(self) -> None:
        def mutate(root: Path, registry: dict[str, Any]) -> None:
            path = root / "evaluation_barrier.json"
            document = _read_json(path)
            document["test_split_accessed"] = True
            _write_json(path, document)
            self.reseal_full(root, registry, "SG-C7")

        self.assert_rejected("SG-C7", mutate, "equals::evaluation_barrier.json#test_split_accessed")

    def test_c5_losing_the_producer_head(self) -> None:
        def mutate(root: Path, registry: dict[str, Any]) -> None:
            path = root / "decision_contract.json"
            document = _read_json(path)
            document["git_head"] = C5_CONSUMER_HEAD
            _write_json(path, document)
            self.reseal_full(root, registry, "SG-C5")

        self.assert_rejected("SG-C5", mutate, "code_head_decision_producer")

    def test_c5_losing_the_consumer_head(self) -> None:
        def mutate(root: Path, registry: dict[str, Any]) -> None:
            path = root / "evaluation_contract.json"
            document = _read_json(path)
            document["evaluation_code"]["git_head"] = C5_PRODUCER_HEAD
            _write_json(path, document)
            self.reseal_full(root, registry, "SG-C5")

        self.assert_rejected("SG-C5", mutate, "code_head_evaluation_consumer")

    def test_broken_payload_equality_inside_a_single_c2_tree(self) -> None:
        def mutate(root: Path, registry: dict[str, Any]) -> None:
            path = root / "per_step.csv"
            path.write_bytes(path.read_bytes() + b"\n")
            self.reseal_envelope(root, registry, "SG-C2-PARITY")

        self.assert_rejected("SG-C2-PARITY", mutate, "sha256::per_step.csv")

    def test_broken_payload_equality_across_the_two_c2_trees(self) -> None:
        parity = self.copy_product("SG-C2-PARITY")
        target = parity / "trajectory_summary.csv"
        target.write_bytes(target.read_bytes() + b"\n")
        roots = {"SG-C2-PARITY": parity, "SG-C2-SCIENCE": self.product_root("SG-C2-SCIENCE")}
        with self.assertRaises(sh.InvalidProductError) as captured:
            sh.verify_relation(self.registry, self.registry["relations"][0], roots)
        self.assertEqual(captured.exception.check, "relation_equivalent_scientific_payload_to")
        self.assertIn("trajectory_summary.csv", captured.exception.detail)

    def test_a_relation_file_deleted_on_disk_is_rejected_at_verification_time(self) -> None:
        parity = self.copy_product("SG-C2-PARITY")
        (parity / "summary.json").unlink()
        roots = {"SG-C2-PARITY": parity, "SG-C2-SCIENCE": self.product_root("SG-C2-SCIENCE")}
        with self.assertRaises(sh.InvalidProductError) as captured:
            sh.verify_relation(self.registry, self.registry["relations"][0], roots)
        self.assertEqual(captured.exception.check, "relation_equivalent_scientific_payload_to")
        self.assertIn("summary.json", captured.exception.detail)

    def test_a_relation_whose_side_was_not_located_is_rejected(self) -> None:
        roots = {"SG-C2-PARITY": self.product_root("SG-C2-PARITY")}
        with self.assertRaises(sh.MissingProductError) as captured:
            sh.verify_relation(self.registry, self.registry["relations"][0], roots)
        self.assertEqual(captured.exception.check, "relation_equivalent_scientific_payload_to")

    def test_parity_product_substituted_for_the_scientific_c2(self) -> None:
        parity_root = self.product_root("SG-C2-PARITY")
        self.product_root("SG-C2-SCIENCE")
        with self.assertRaises(sh.InvalidProductError) as captured:
            sh.verify_product(self.registry, "SG-C2-SCIENCE", parity_root, repo_root=REPO_ROOT)
        self.assertEqual(captured.exception.check, "run_manifest_sha256")

    def test_c1_confirmation_losing_its_exploration_freeze_link(self) -> None:
        def mutate(root: Path, registry: dict[str, Any]) -> None:
            path = root / "confirmation/stage_contract.json"
            document = _read_json(path)
            document["exploration_freeze"]["summary_sha256"] = "0" * 64
            _write_json(path, document)
            self.reseal_chains(root, registry, "SG-C1-CONFIRMATION")

        self.assert_rejected(
            "SG-C1-CONFIRMATION",
            mutate,
            "source_link::SG-C1-EXPLORATION<-confirmation/stage_contract.json#exploration_freeze.summary_sha256",
        )

    def test_c3_losing_its_link_to_the_parity_c2_it_actually_consumed(self) -> None:
        def mutate(root: Path, registry: dict[str, Any]) -> None:
            path = root / "source_contract.json"
            document = _read_json(path)
            document["c2_source"]["manifest_sha256"] = "0" * 64
            _write_json(path, document)
            self.reseal_chains(root, registry, "SG-C3")

        self.assert_rejected(
            "SG-C3", mutate, "source_link::SG-C2-PARITY<-source_contract.json#c2_source.manifest_sha256"
        )

    def test_c3_source_link_names_the_parity_tree_and_not_the_scientific_one(self) -> None:
        links = _entry(self.registry, "SG-C3")["source_links"]
        self.assertEqual({item["source_product_id"] for item in links}, {"SG-C2-PARITY"})
        self.assertEqual({item["source_artifact"] for item in links}, {"c2_manifest.json", "c2_contract.json"})
        # The link is only discriminating because the two C2 trees really differ in these two files.
        for name in ("c2_manifest.json", "c2_contract.json"):
            science = _sha256(self.product_root("SG-C2-SCIENCE") / name)
            parity = _sha256(self.product_root("SG-C2-PARITY") / name)
            self.assertNotEqual(science, parity, msg=name)
        self.assertIn("SG-C2-SCIENCE remains the scientific owner", _entry(self.registry, "SG-C3")["summary"])

    def test_c3_links_resolve_against_the_parity_tree_and_fail_against_the_scientific_one(self) -> None:
        parity = self.product_root("SG-C2-PARITY")
        science = self.product_root("SG-C2-SCIENCE")
        entry = _entry(self.registry, "SG-C3")
        sh.verify_source_link_targets(self.registry, entry, {"SG-C2-PARITY": parity})
        with self.assertRaises(sh.InvalidProductError):
            sh.verify_source_link_targets(self.registry, entry, {"SG-C2-PARITY": science})

    def test_source_link_that_no_longer_matches_the_source_product(self) -> None:
        def mutate(root: Path, registry: dict[str, Any]) -> None:
            path = root / "decision_contract.json"
            document = _read_json(path)
            document["source_c3_run_manifest_sha256"] = "0" * 64
            _write_json(path, document)
            self.reseal_chains(root, registry, "SG-C4")

        self.assert_rejected(
            "SG-C4", mutate, "source_link::SG-C3<-decision_contract.json#source_c3_run_manifest_sha256"
        )


class GitProvenanceTest(SearchHistoryTestCase):
    def test_unknown_commit_is_rejected(self) -> None:
        registry = self.working_registry()
        _entry(registry, "SG-C0")["code_heads"][0]["git_head"] = "0" * 40
        root = self.product_root("SG-C0")
        with self.assertRaises(sh.VerifierError) as captured:
            sh.verify_product(registry, "SG-C0", root, repo_root=REPO_ROOT)
        self.assertEqual(captured.exception.check, "code_head_run_manifest")

    def test_unknown_commit_in_an_entrypoint_is_rejected(self) -> None:
        registry = self.working_registry()
        _entry(registry, "SG-C0")["entrypoints"][0]["git_head"] = "0" * 40
        root = self.product_root("SG-C0")
        with self.assertRaises(sh.MissingGitObjectError) as captured:
            sh.verify_product(registry, "SG-C0", root, repo_root=REPO_ROOT)
        self.assertEqual(captured.exception.check, "git_commit_present")

    def test_absent_historical_entrypoint_is_rejected(self) -> None:
        registry = self.working_registry()
        _entry(registry, "SG-C0")["entrypoints"][0]["path"] = "tools/analysis/run_search_gate_c9.py"
        root = self.product_root("SG-C0")
        with self.assertRaises(sh.MissingGitObjectError) as captured:
            sh.verify_product(registry, "SG-C0", root, repo_root=REPO_ROOT)
        self.assertEqual(captured.exception.check, "git_entrypoint_blob")

    def test_a_commit_outside_the_freeze_ancestry_is_rejected(self) -> None:
        stray = self._commit_outside_the_freeze_ancestry()
        registry = self.working_registry()
        _entry(registry, "SG-C0")["entrypoints"][0]["git_head"] = stray
        root = self.product_root("SG-C0")
        with self.assertRaises(sh.MissingGitObjectError) as captured:
            sh.verify_product(registry, "SG-C0", root, repo_root=REPO_ROOT)
        self.assertEqual(captured.exception.check, "git_commit_ancestor")

    def _commit_outside_the_freeze_ancestry(self) -> str:
        runner = sh._GitRunner(REPO_ROOT)
        listed = runner.run(["rev-list", "--all", "--max-count=600"])
        if listed.returncode != 0:
            self.skipTest("git rev-list is unavailable")
        for candidate in listed.stdout.split():
            if not runner.is_ancestor(candidate, FREEZE_COMMIT):
                return candidate
        self.skipTest("this clone holds no commit outside the freeze ancestry")
        raise AssertionError

    def test_the_historical_verdict_ignores_the_current_head_and_a_dirty_worktree(self) -> None:
        borrowed = self._borrowed_object_repository()
        root = self.product_root("SG-C5")
        native = sh.verify_product(self.registry, "SG-C5", root, repo_root=REPO_ROOT)
        detached = sh.verify_product(self.registry, "SG-C5", root, repo_root=borrowed)
        self.assertEqual(native["git"]["commits"], detached["git"]["commits"])
        self.assertEqual(native["git"]["entrypoints"], detached["git"]["entrypoints"])
        self.assertEqual(detached["result"], "PASS")

    def _borrowed_object_repository(self) -> Path:
        """A throwaway repository that borrows the objects but carries a different HEAD."""
        temporary = Path(tempfile.mkdtemp(prefix="ctcf_search_history_git_"))
        self.addCleanup(shutil.rmtree, temporary, ignore_errors=True)
        git_dir = temporary / ".git"
        (git_dir / "objects" / "info").mkdir(parents=True)
        (git_dir / "refs" / "heads").mkdir(parents=True)
        # Written as bytes: a CRLF translation here would become part of the alternates path.
        (git_dir / "config").write_bytes(b"[core]\n\trepositoryformatversion = 0\n")
        (git_dir / "HEAD").write_bytes(f"{C5_PRODUCER_HEAD}\n".encode("ascii"))
        alternates = (REPO_ROOT / ".git" / "objects").as_posix()
        (git_dir / "objects" / "info" / "alternates").write_bytes(f"{alternates}\n".encode())
        (temporary / "unrelated_dirty_file.txt").write_bytes(b"dirty\n")
        return temporary

    def test_the_verifier_never_calls_a_head_dependent_git_subcommand(self) -> None:
        tree = ast.parse((PACKAGE / "verify.py").read_text(encoding="utf-8"))
        invoked: set[str] = set()
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            target = node.func
            if isinstance(target, ast.Attribute) and target.attr == "run" and node.args:
                argument = node.args[0]
                if isinstance(argument, ast.List) and argument.elts and isinstance(argument.elts[0], ast.Constant):
                    invoked.add(str(argument.elts[0].value))
        self.assertTrue(invoked, msg="no git argument list was found to inspect")
        self.assertFalse(invoked & set(HEAD_DEPENDENT_GIT_SUBCOMMANDS), msg=f"head-dependent git call: {invoked}")

    def test_the_git_runner_drops_an_inherited_git_environment(self) -> None:
        runner = sh._GitRunner(REPO_ROOT)
        for inherited in ("GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE", "GIT_OBJECT_DIRECTORY"):
            self.assertNotIn(inherited, runner.environment)
        self.assertEqual(runner.environment["GIT_NO_REPLACE_OBJECTS"], "1")

    def test_the_git_runner_forbids_network_fetch_and_repository_locks(self) -> None:
        runner = sh._GitRunner(REPO_ROOT)
        self.assertEqual(runner.environment["GIT_NO_LAZY_FETCH"], "1")
        self.assertEqual(runner.environment["GIT_OPTIONAL_LOCKS"], "0")
        self.assertEqual(runner.environment["GIT_TERMINAL_PROMPT"], "0")

    def test_a_partial_clone_never_reaches_the_network_for_a_missing_object(self) -> None:
        runner = sh._GitRunner(REPO_ROOT)
        completed = runner.run(["cat-file", "-t", "0" * 40])
        self.assertNotEqual(completed.returncode, 0)
        self.assertNotIn("fetch", completed.stderr.lower())

    def test_a_hostile_git_environment_does_not_change_the_verdict(self) -> None:
        root = self.product_root("SG-C0")
        expected = sh.verify_product(self.registry, "SG-C0", root, repo_root=REPO_ROOT)
        poisoned = dict(os.environ)
        poisoned.update({"GIT_DIR": "nonexistent", "GIT_WORK_TREE": "nonexistent", "GIT_INDEX_FILE": "nonexistent"})
        saved = dict(os.environ)
        os.environ.update(poisoned)
        self.addCleanup(lambda: (os.environ.clear(), os.environ.update(saved)))
        actual = sh.verify_product(self.registry, "SG-C0", root, repo_root=REPO_ROOT)
        self.assertEqual(actual["git"], expected["git"])


class PathSafetyTest(SearchHistoryTestCase):
    def setUp(self) -> None:
        self.sandbox = Path(tempfile.mkdtemp(prefix="ctcf_search_history_path_"))
        self.addCleanup(shutil.rmtree, self.sandbox, ignore_errors=True)
        (self.sandbox / "inside.json").write_bytes(b"{}\n")

    def _try_symlink(self, link: Path, target: Path, *, directory: bool = False) -> None:
        try:
            link.symlink_to(target, target_is_directory=directory)
        except (OSError, NotImplementedError) as exc:
            self.skipTest(f"this platform will not create a symlink: {exc}")

    def test_a_symlink_that_escapes_the_root_is_rejected(self) -> None:
        outside = Path(tempfile.mkdtemp(prefix="ctcf_search_history_out_"))
        self.addCleanup(shutil.rmtree, outside, ignore_errors=True)
        (outside / "secret.json").write_bytes(b"{}\n")
        self._try_symlink(self.sandbox / "escape.json", outside / "secret.json")
        with self.assertRaises(sh.InvalidProductError) as captured:
            sh.resolve_inside_root(self.sandbox, "escape.json", check="probe", error=sh.InvalidProductError)
        self.assertIn("symlink or reparse point", captured.exception.detail)

    def test_a_symlink_that_stays_inside_the_root_is_still_rejected(self) -> None:
        self._try_symlink(self.sandbox / "alias.json", self.sandbox / "inside.json")
        with self.assertRaises(sh.InvalidProductError):
            sh.resolve_inside_root(self.sandbox, "alias.json", check="probe", error=sh.InvalidProductError)

    def test_the_tree_walk_rejects_a_symlink(self) -> None:
        self._try_symlink(self.sandbox / "alias.json", self.sandbox / "inside.json")
        with self.assertRaises(sh.InvalidProductError) as captured:
            sh.enumerate_tree(self.sandbox, check="tree_closure")
        self.assertEqual(captured.exception.check, "tree_closure")

    def _force_link_like(self, target: Path) -> None:
        """Present one path as a reparse point, so the rejection runs without the OS privilege."""
        original = sh.os.lstat
        reparse = getattr(sh.stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
        marked = sh.stat.S_IFLNK | 0o777 if reparse == 0 else None

        class _LinkLike:
            def __init__(self, borrowed: os.stat_result) -> None:
                self.st_mode = marked if marked is not None else borrowed.st_mode
                self.st_file_attributes = reparse
                self.st_size = borrowed.st_size
                self.st_mtime_ns = borrowed.st_mtime_ns

        def patched(path: Any) -> Any:
            entry = original(path)
            return _LinkLike(entry) if Path(path) == target else entry

        sh.os.lstat = patched
        self.addCleanup(setattr, sh.os, "lstat", original)

    def test_a_reparse_point_is_rejected_by_the_path_resolver(self) -> None:
        self._force_link_like(self.sandbox / "inside.json")
        with self.assertRaises(sh.InvalidProductError) as captured:
            sh.resolve_inside_root(self.sandbox, "inside.json", check="probe", error=sh.InvalidProductError)
        self.assertIn("symlink or reparse point", captured.exception.detail)

    def test_a_reparse_point_is_rejected_by_the_tree_walk(self) -> None:
        self._force_link_like(self.sandbox / "inside.json")
        with self.assertRaises(sh.InvalidProductError) as captured:
            sh.enumerate_tree(self.sandbox, check="tree_closure")
        self.assertIn("symlink or reparse point", captured.exception.detail)

    def test_link_detection_covers_the_posix_mode_and_the_windows_attribute(self) -> None:
        class _Entry:
            def __init__(self, mode: int, attributes: int) -> None:
                self.st_mode = mode
                self.st_file_attributes = attributes

        self.assertTrue(sh._is_link_like(_Entry(sh.stat.S_IFLNK | 0o777, 0)))
        self.assertFalse(sh._is_link_like(_Entry(sh.stat.S_IFREG | 0o644, 0)))
        reparse = getattr(sh.stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
        if reparse:
            self.assertTrue(sh._is_link_like(_Entry(sh.stat.S_IFREG | 0o644, reparse)))

    def test_a_directory_is_not_accepted_as_a_file(self) -> None:
        (self.sandbox / "folder").mkdir()
        with self.assertRaises(sh.InvalidProductError) as captured:
            sh.resolve_inside_root(self.sandbox, "folder", check="probe", error=sh.InvalidProductError)
        self.assertIn("not a regular file", captured.exception.detail)

    def test_the_path_grammar_rejects_every_hostile_form(self) -> None:
        hostile = [
            "/absolute",
            "C:/drive",
            "c:\\drive",
            "\\\\host\\share",
            "back\\slash",
            "..",
            "../escape",
            "a/../b",
            "./a",
            "a//b",
            "a/",
            "",
            "a\x00b",
            "with space",
            # Windows strips a trailing dot or space, so these alias an existing name.
            "foo.",
            "cases/foo./x.json",
            "NUL",
            "con.json",
            "cases/AUX/x.json",
            "LPT1.txt",
        ]
        for candidate in hostile:
            with self.assertRaises(sh.InvalidProductError, msg=candidate):
                sh.validate_relative_path(candidate, check="probe", error=sh.InvalidProductError)

    def test_the_path_grammar_accepts_the_shapes_the_products_use(self) -> None:
        for candidate in (
            "outputs.tsv",
            "cases/subject_1/decision_complete.json",
            "attempts/A_1_2/runner_contract.txt",
        ):
            sh.validate_relative_path(candidate, check="probe", error=sh.InvalidProductError)

    def test_a_file_that_changes_while_it_is_read_is_rejected(self) -> None:
        target = self.sandbox / "growing.bin"
        target.write_bytes(b"0" * (2 * sh.READ_CHUNK_BYTES))
        original = sh.os.fstat

        state = {"calls": 0}

        def unstable(descriptor: int) -> os.stat_result:
            state["calls"] += 1
            result = original(descriptor)
            if state["calls"] > 1:
                fields = list(result)
                fields[6] = fields[6] + 1
                return os.stat_result(fields)
            return result

        sh.os.fstat = unstable
        self.addCleanup(setattr, sh.os, "fstat", original)
        with self.assertRaises(sh.InvalidProductError) as captured:
            sh.hash_regular_file(target, check="probe", error=sh.InvalidProductError)
        self.assertIn("changed size or mtime", captured.exception.detail)


class IndexGrammarTest(SearchHistoryTestCase):
    def _index(self, body: str) -> dict[str, tuple[int, str]]:
        return sh.parse_outputs_index(body.encode("utf-8"), check="probe")

    def test_a_well_formed_index_is_accepted(self) -> None:
        parsed = self._index(f"{sh.OUTPUTS_HEADER}\na.json\t3\t{'a' * 64}\n")
        self.assertEqual(parsed, {"a.json": (3, "a" * 64)})

    def test_a_wrong_header_is_rejected(self) -> None:
        with self.assertRaises(sh.InvalidProductError):
            self._index("path\tbytes\tsha256\na.json\t3\t" + "a" * 64 + "\n")

    def test_a_negative_or_non_canonical_size_is_rejected(self) -> None:
        for size in ("-1", "+1", "007", " 3", "3 ", "", "٣"):
            with self.assertRaises(sh.InvalidProductError, msg=size):
                self._index(f"{sh.OUTPUTS_HEADER}\na.json\t{size}\t{'a' * 64}\n")

    def test_a_malformed_digest_is_rejected(self) -> None:
        for digest in ("A" * 64, "a" * 63, "a" * 65, "", "g" * 64):
            with self.assertRaises(sh.InvalidProductError, msg=digest):
                self._index(f"{sh.OUTPUTS_HEADER}\na.json\t3\t{digest}\n")

    def test_a_row_with_the_wrong_field_count_is_rejected(self) -> None:
        with self.assertRaises(sh.InvalidProductError):
            self._index(f"{sh.OUTPUTS_HEADER}\na.json\t3\n")

    def test_a_carriage_return_is_rejected(self) -> None:
        with self.assertRaises(sh.InvalidProductError):
            self._index(f"{sh.OUTPUTS_HEADER}\r\na.json\t3\t{'a' * 64}\r\n")


class PlacementTest(SearchHistoryTestCase):
    def _namespace(self, results_root: Path) -> Any:
        return results_root

    def test_zero_matches_and_several_declared_matches_are_distinguishable(self) -> None:
        empty = Path(tempfile.mkdtemp(prefix="ctcf_search_history_empty_"))
        self.addCleanup(shutil.rmtree, empty, ignore_errors=True)
        entry = _entry(self.registry, "SG-C0")
        self.assertEqual(_locate(self.registry, entry, empty)["status"], "NOT_FOUND")

        staged = Path(tempfile.mkdtemp(prefix="ctcf_search_history_two_"))
        self.addCleanup(shutil.rmtree, staged, ignore_errors=True)
        source = self.product_root("SG-C0")
        clone = copy.deepcopy(entry)
        clone["relative_hints"] = ["copy_a/product", "copy_b/product"]
        for hint in clone["relative_hints"]:
            shutil.copytree(source, staged / hint)
        self.assertEqual(_locate(self.registry, clone, staged)["status"], "AMBIGUOUS")

    def test_a_hint_that_holds_a_different_run_id_is_rejected(self) -> None:
        staged = Path(tempfile.mkdtemp(prefix="ctcf_search_history_swap_"))
        self.addCleanup(shutil.rmtree, staged, ignore_errors=True)
        entry = copy.deepcopy(_entry(self.registry, "SG-C0"))
        entry["relative_hints"] = ["here/product"]
        shutil.copytree(self.product_root("SG-C1-EXPLORATION"), staged / "here/product")
        with self.assertRaises(sh.MissingProductError) as captured:
            _locate(self.registry, entry, staged)
        self.assertEqual(captured.exception.check, "verify_known_run_id")

    def test_a_hint_holding_a_structurally_broken_manifest_is_rejected_strictly(self) -> None:
        staged = Path(tempfile.mkdtemp(prefix="ctcf_search_history_dup_"))
        self.addCleanup(shutil.rmtree, staged, ignore_errors=True)
        entry = copy.deepcopy(_entry(self.registry, "SG-C0"))
        entry["relative_hints"] = ["here/product"]
        destination = staged / "here/product"
        destination.mkdir(parents=True)
        (destination / "run_manifest.json").write_bytes(b'{"run_id": "a", "run_id": "b"}')
        with self.assertRaises(sh.InvalidProductError) as captured:
            _locate(self.registry, entry, staged)
        self.assertEqual(captured.exception.check, "verify_known_manifest")

    def test_a_results_root_that_is_a_reparse_point_is_rejected(self) -> None:
        staged = Path(tempfile.mkdtemp(prefix="ctcf_search_history_rp_"))
        self.addCleanup(shutil.rmtree, staged, ignore_errors=True)
        original = sh.os.lstat
        reparse = getattr(sh.stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)

        class _LinkLike:
            def __init__(self, borrowed: os.stat_result) -> None:
                self.st_mode = sh.stat.S_IFLNK | 0o777 if reparse == 0 else borrowed.st_mode
                self.st_file_attributes = reparse

        sh.os.lstat = lambda path: _LinkLike(original(path)) if Path(path) == staged else original(path)
        self.addCleanup(setattr, sh.os, "lstat", original)
        with self.assertRaises(sh.InvalidProductError) as captured:
            _locate(self.registry, _entry(self.registry, "SG-C0"), staged)
        self.assertEqual(captured.exception.check, "verify_known_results_root")

    def test_a_hint_whose_directory_is_a_reparse_point_is_rejected(self) -> None:
        staged = Path(tempfile.mkdtemp(prefix="ctcf_search_history_hint_"))
        self.addCleanup(shutil.rmtree, staged, ignore_errors=True)
        entry = copy.deepcopy(_entry(self.registry, "SG-C0"))
        entry["relative_hints"] = ["here/product"]
        destination = staged / "here/product"
        destination.mkdir(parents=True)
        (destination / "run_manifest.json").write_bytes(b"{}")
        original = sh.os.lstat
        reparse = getattr(sh.stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)

        class _LinkLike:
            def __init__(self, borrowed: os.stat_result) -> None:
                self.st_mode = sh.stat.S_IFLNK | 0o777 if reparse == 0 else borrowed.st_mode
                self.st_file_attributes = reparse

        sh.os.lstat = lambda path: _LinkLike(original(path)) if Path(path) == destination else original(path)
        self.addCleanup(setattr, sh.os, "lstat", original)
        with self.assertRaises(sh.InvalidProductError) as captured:
            _locate(self.registry, entry, staged)
        self.assertEqual(captured.exception.check, "verify_known_hint_path")

    def test_verify_known_locates_the_doubled_c5b_directory(self) -> None:
        entry = _entry(self.registry, "SG-C5B")
        placement = _locate(self.registry, entry, RESULTS)
        if placement["status"] == "NOT_FOUND" and not REQUIRE_HISTORICAL:
            self.skipTest("the C5b product is absent")
        self.assertEqual(placement["status"], "MATCHED")


class IsolationTest(SearchHistoryTestCase):
    def test_no_package_module_imports_a_historical_producer_or_a_heavy_dependency(self) -> None:
        for source in sorted(PACKAGE.glob("*.py")):
            tree = ast.parse(source.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    names = [alias.name for alias in node.names]
                elif isinstance(node, ast.ImportFrom):
                    names = [node.module or ""]
                else:
                    continue
                for name in names:
                    tail = name.rsplit(".", 1)[-1]
                    self.assertFalse(
                        tail.startswith(FORBIDDEN_IMPORT_PREFIXES),
                        msg=f"{source.name} imports the historical producer {name}",
                    )
                    self.assertNotIn(
                        name.split(".")[0],
                        FORBIDDEN_IMPORT_ROOTS,
                        msg=f"{source.name} imports the heavy dependency {name}",
                    )

    def test_importing_the_package_pulls_in_no_historical_producer_module(self) -> None:
        completed = subprocess.run(
            [
                sys.executable,
                "-c",
                "import sys; import tools.analysis.search_history as p; "
                "print(' '.join(sorted(m for m in sys.modules if 'search_gate' in m)))",
            ],
            capture_output=True,
            text=True,
            cwd=REPO_ROOT,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, msg=completed.stderr)
        self.assertEqual(completed.stdout.strip(), "")

    def test_the_verifier_reads_only_the_standard_library(self) -> None:
        tree = ast.parse((PACKAGE / "verify.py").read_text(encoding="utf-8"))
        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                imported.add(node.module.split(".")[0])
        self.assertTrue(imported <= set(sys.stdlib_module_names), msg=f"non-stdlib imports: {imported}")


class CommandLineTest(SearchHistoryTestCase):
    def _run(self, arguments: list[str]) -> tuple[int, dict[str, Any] | None, str]:
        completed = subprocess.run(
            [sys.executable, "-m", "tools.analysis.search_history", *arguments],
            capture_output=True,
            text=True,
            cwd=REPO_ROOT,
            check=False,
        )
        document = json.loads(completed.stdout) if completed.stdout.strip() else None
        return completed.returncode, document, completed.stderr

    def test_list_prints_one_deterministic_json_document(self) -> None:
        code, document, stderr = self._run(["list"])
        self.assertEqual(code, sh.EXIT_OK)
        self.assertEqual(stderr, "")
        self.assertEqual(document["canonical_count"], 11)
        self.assertEqual(document["supporting_count"], 1)
        again = self._run(["list"])[1]
        self.assertEqual(document, again)

    def test_verify_accepts_a_relocated_product(self) -> None:
        root = self.copy_product("SG-C0")
        code, document, _ = self._run(["verify", "--id", "SG-C0", "--product-root", str(root)])
        self.assertEqual(code, sh.EXIT_OK)
        self.assertEqual(document["result"], "PASS")

    def test_verify_reports_a_missing_product_with_its_own_exit_code(self) -> None:
        code, document, _ = self._run(["verify", "--id", "SG-C0", "--product-root", str(RESULTS / "nowhere")])
        self.assertEqual(code, sh.EXIT_MISSING_PRODUCT)
        self.assertEqual(document["category"], "MISSING_PRODUCT")

    def test_an_unusable_registry_reports_invalid_registry(self) -> None:
        broken = Path(tempfile.mkdtemp(prefix="ctcf_search_history_reg_"))
        self.addCleanup(shutil.rmtree, broken, ignore_errors=True)
        path = broken / "registry.json"
        path.write_bytes(b'{"schema": "wrong"}')
        code, document, stderr = self._run(["--registry", str(path), "list"])
        self.assertEqual(code, sh.EXIT_INVALID_REGISTRY)
        self.assertEqual(stderr, "")
        self.assertEqual(document["category"], "INVALID_REGISTRY")
        self.assertEqual(document["result"], "FAIL")

    def test_verify_known_requires_every_registered_product(self) -> None:
        empty = Path(tempfile.mkdtemp(prefix="ctcf_search_history_none_"))
        self.addCleanup(shutil.rmtree, empty, ignore_errors=True)
        for arguments in (
            ["verify-known", "--results-root", str(empty)],
            ["verify-known", "--results-root", str(empty), "--require-all"],
        ):
            code, document, _ = self._run(arguments)
            self.assertEqual(code, sh.EXIT_MISSING_PRODUCT, msg=str(arguments))
            self.assertEqual(document["result"], "FAIL")
            self.assertTrue(all(entry["placement"] == "NOT_FOUND" for entry in document["products"]))
            self.assertTrue(all(entry["result"] == "FAIL" for entry in document["products"]))

    def test_verify_known_accepts_the_whole_recorded_series(self) -> None:
        for product_id in (entry["id"] for entry in self.registry["products"]):
            self.product_root(product_id)
        code, document, stderr = self._run(["verify-known", "--results-root", str(RESULTS), "--require-all"])
        self.assertEqual(code, sh.EXIT_OK, msg=stderr)
        self.assertEqual(document["result"], "PASS")
        canonical, supporting = document["canonical"], document["supporting"]
        self.assertEqual((canonical["registered"], canonical["passed"]), (11, 11))
        self.assertEqual((supporting["registered"], supporting["passed"]), (1, 1))
        self.assertEqual(
            (canonical["index_rows"], canonical["files"], canonical["bytes"]),
            (CANONICAL_INDEX_ROWS, CANONICAL_FILES, CANONICAL_BYTES),
        )
        self.assertEqual(
            (supporting["index_rows"], supporting["files"], supporting["bytes"]),
            (PARITY_INDEX_ROWS, PARITY_FILES, PARITY_BYTES),
        )
        self.assertNotIn("SG-C2-PARITY", canonical["ids"])
        self.assertEqual(supporting["ids"], ["SG-C2-PARITY"])
        self.assertTrue(all(item["result"] == "PASS" for item in document["relations"]))
        self.assertTrue(all(item["result"] == "PASS" for item in document["source_links"]))

    def test_an_absent_registry_file_reports_invalid_registry(self) -> None:
        broken = Path(tempfile.mkdtemp(prefix="ctcf_search_history_int_"))
        self.addCleanup(shutil.rmtree, broken, ignore_errors=True)
        code, document, _ = self._run(["--registry", str(broken / "absent.json"), "list"])
        self.assertEqual(code, sh.EXIT_INVALID_REGISTRY)
        self.assertEqual(document["category"], "INVALID_REGISTRY")

    def test_an_unknown_product_id_reports_invalid_registry_as_one_json_document(self) -> None:
        code, document, stderr = self._run(["verify", "--id", "SG-NOWHERE", "--product-root", str(RESULTS)])
        self.assertEqual(code, sh.EXIT_INVALID_REGISTRY)
        self.assertEqual(stderr, "")
        self.assertEqual(document["category"], "INVALID_REGISTRY")
        self.assertEqual(document["check"], "registry_lookup")

    def test_an_unexpected_fault_maps_to_the_internal_error_code_and_still_prints_json(self) -> None:
        original = main.__globals__["load_registry"]

        def exploding(*_args: Any, **_kwargs: Any) -> Any:
            raise RuntimeError("synthetic fault")

        main.__globals__["load_registry"] = exploding
        self.addCleanup(main.__globals__.__setitem__, "load_registry", original)
        stream = io.StringIO()
        saved = sys.stdout
        sys.stdout = stream
        try:
            code = main(["list"])
        finally:
            sys.stdout = saved
        self.assertEqual(code, sh.EXIT_INTERNAL_ERROR)
        document = json.loads(stream.getvalue())
        self.assertEqual(document["category"], "INTERNAL_ERROR")
        self.assertEqual(document["check"], "unexpected_exception")
        self.assertIn("synthetic fault", document["detail"])

    def test_the_cli_offers_no_way_to_skip_the_git_provenance_pass(self) -> None:
        code, document, _ = self._run(["verify", "--id", "SG-C0", "--product-root", str(RESULTS), "--skip-git"])
        self.assertNotEqual(code, sh.EXIT_OK)
        self.assertIsNone(document, msg="argparse must reject the removed flag before any work happens")
        source = (PACKAGE / "__main__.py").read_text(encoding="utf-8")
        self.assertNotIn("skip-git", source)
        self.assertNotIn("skip_git", source)

    def test_every_cli_success_reports_a_checked_git_provenance(self) -> None:
        root = self.copy_product("SG-C0")
        _, document, _ = self._run(["verify", "--id", "SG-C0", "--product-root", str(root)])
        self.assertEqual(document["result"], "PASS")
        self.assertTrue(document["git"]["checked"])

    def test_verify_product_exposes_no_way_to_skip_the_git_pass(self) -> None:
        parameters = inspect.signature(sh.verify_product).parameters
        self.assertEqual(sorted(parameters), ["product_id", "product_root", "registry", "repo_root"])
        source = (PACKAGE / "verify.py").read_text(encoding="utf-8")
        for banned in ("check_git", "skip_git", "PASS_WITHOUT_GIT_PROVENANCE"):
            self.assertNotIn(banned, source, msg=f"{banned} is a bytes-only bypass and must not exist")
        with self.assertRaises(TypeError):
            sh.verify_product(self.registry, "SG-C0", RESULTS, check_git=False)

    def test_every_public_success_carries_a_checked_git_provenance(self) -> None:
        root = self.copy_product("SG-C0")
        report = sh.verify_product(self.registry, "SG-C0", root, repo_root=REPO_ROOT)
        self.assertEqual(report["result"], "PASS")
        self.assertIs(report["git"]["checked"], True)


class ReadOnlyTest(SearchHistoryTestCase):
    def test_verification_leaves_the_product_bytes_and_mtimes_untouched(self) -> None:
        root = self.product_root("SG-C0")
        before = {
            path.relative_to(root).as_posix(): (path.stat().st_size, path.stat().st_mtime_ns, _sha256(path))
            for path in sorted(root.rglob("*"))
            if path.is_file()
        }
        sh.verify_product(self.registry, "SG-C0", root, repo_root=REPO_ROOT)
        after = {
            path.relative_to(root).as_posix(): (path.stat().st_size, path.stat().st_mtime_ns, _sha256(path))
            for path in sorted(root.rglob("*"))
            if path.is_file()
        }
        self.assertEqual(before, after)


if __name__ == "__main__":
    unittest.main()
