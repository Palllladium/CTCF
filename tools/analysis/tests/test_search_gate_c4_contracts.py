from __future__ import annotations

import copy
import json
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch

import numpy as np

from tools.analysis import search_gate_c4 as policy_owner, search_gate_c4_contracts as c4
from tools.analysis.run_artifacts import sha256_file


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")


def _file_record(path: Path, *, case_id: str, split: str) -> dict[str, object]:
    return {
        "dataset": "IXI",
        "split": split,
        "case_id": case_id,
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "mtime_utc": "2026-08-24T00:00:00Z",
    }


def _image_record(path: Path, array: np.ndarray) -> dict[str, object]:
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "array_sha256": c4.array_sha256(array),
        "shape": list(array.shape),
        "dtype": str(array.dtype),
    }


def _field_record(path: Path, root: Path, array: np.ndarray) -> dict[str, str]:
    return {
        "relative_path": path.resolve().relative_to(root.resolve()).as_posix(),
        "npz_sha256": sha256_file(path),
        "array_sha256": c4.array_sha256(array),
    }


def _save_flow(path: Path, value: float) -> np.ndarray:
    array = np.full((1, 3, 3, 3, 3), value, dtype=np.float32)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, flow=array)
    return array


def _execution(contract: dict[str, object], phase: str, shard: int, attempt: str) -> dict[str, object]:
    runtime = contract["runtime_signature"]
    payload: dict[str, object] = {
        "phase": phase,
        "attempt_id": attempt,
        "shard_index": shard,
        "physical_gpu": contract["shard_to_physical_gpu"][str(shard)],
        "host": "fixture-host",
        "python": runtime["python"],
        "torch": runtime["torch"],
        "device": "cuda:0",
        "gpu_name": "fixture-H100",
        "seed": contract["seed"],
        "deterministic": True,
    }
    payload["labels_loaded_to_device" if phase == "decision" else "labels_loaded_after_barrier"] = phase == "evaluation"
    return payload


def _geometry_bundle() -> dict[str, object]:
    return {
        metric_id: {
            "status": "OK",
            "value": None if metric_id == c4.DETJ_DIAGNOSTICS else 0.0,
            **(
                {"components": {"corner_union_violation_fraction": 0.0}}
                if metric_id == c4.DIGITAL_DECOMPOSITION
                else (
                    {"components": {"detj_min": 1.0, "invalid_count": 0.0}} if metric_id == c4.DETJ_DIAGNOSTICS else {}
                )
            ),
        }
        for metric_id in c4.METRIC_SPECS
    }


class ContractFixture(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.heavy = self.root / "heavy"
        self.heavy.mkdir()
        self.case_ids = [f"subject_{index + 1000}" for index in range(c4.EXPECTED_CASES)]
        image_inputs = {
            case_id: {"array_sha256": c4.payload_sha256(["image", case_id])} for case_id in ["atlas", *self.case_ids]
        }
        source_initial = {
            case_id: {
                "field": {
                    "relative_path": f"source/{case_id}.npz",
                    "npz_sha256": c4.payload_sha256(["npz", case_id]),
                    "array_sha256": c4.payload_sha256(["array", case_id]),
                },
                "exact": {
                    "status": "CERTIFIED",
                    "certified": True,
                    "sha256": c4.payload_sha256(["array", case_id]),
                },
                "source_decision_case_sha256": c4.payload_sha256(["case", case_id]),
            }
            for case_id in self.case_ids
        }
        source_historical = {
            case_id: {
                "raw_conf_requested_field": source_initial[case_id]["field"],
                "source_decision_case_sha256": source_initial[case_id]["source_decision_case_sha256"],
            }
            for case_id in self.case_ids
        }
        raw_inputs = {
            case_id: {
                "dataset": "IXI",
                "split": "atlas" if case_id == "atlas" else "val",
                "case_id": case_id,
                "path": f"/data/{case_id}.pkl",
                "bytes": 1,
                "sha256": c4.payload_sha256(["raw", case_id]),
            }
            for case_id in ["atlas", *self.case_ids]
        }
        self.snapshot = {
            "source_c3": {
                "compact_directory": "/source/compact",
                "heavy_root": "/source/heavy",
                "run_id": c4.SOURCE_C3_RUN_ID,
                "git_head": c4.SOURCE_C3_GIT_HEAD,
                "manifest_sha256": c4.SOURCE_C3_MANIFEST_SHA256,
                "run_manifest_sha256": c4.SOURCE_C3_RUN_MANIFEST_SHA256,
                "source_contract_sha256": "1" * 64,
                "decision_contract_sha256": "2" * 64,
                "decision_barrier_sha256": "3" * 64,
            },
            "raw_inputs": raw_inputs,
            "image_inputs": image_inputs,
            "source_initial": source_initial,
            "source_historical": source_historical,
            "evaluation_baseline_dice": {case_id: 0.75 for case_id in self.case_ids},
            "case_ids": self.case_ids,
            "seed": 0,
            "runtime_signature": {"python": "3.10"},
        }
        self.source = c4.build_source_contract(
            self.snapshot,
            git_head="a" * 40,
            runtime_signature={"python": "3.10", "torch": "fixture"},
            target_heavy_root=self.heavy,
            physical_gpus=["2", "3", "4", "5", "6"],
        )
        self.source_sha = "b" * 64
        self.policy = {
            "protocol_id": c4.PROTOCOL_ID,
            "schema_version": c4.SCHEMA_VERSION,
            "scientific_reference_arm_id": c4.SCIENTIFIC_REFERENCE_ARM_ID,
        }
        self.arm_specs = [
            {
                "arm_index": index,
                "arm_id": arm_id,
                "role": (
                    "scientific_reference"
                    if arm_id == c4.SCIENTIFIC_REFERENCE_ARM_ID
                    else (
                        "scientific_candidate"
                        if arm_id in c4.SCIENTIFIC_ARM_IDS
                        else {
                            "legacy_mind_d2_s1_collar4": "legacy_parity_diagnostic",
                            "mind_f222_s1": "fusion_idempotence_diagnostic",
                            "intensity_s1": "descriptor_specificity_diagnostic",
                            "intensity_s2": "descriptor_specificity_diagnostic",
                        }[arm_id]
                    )
                ),
                "selectable": arm_id in c4.SCIENTIFIC_ARM_IDS,
                "diagnostic_only": arm_id in c4.DIAGNOSTIC_ARM_IDS,
                "materialize_candidate": arm_id not in {"legacy_mind_d2_s1_collar4", "mind_f222_s1"},
                "post_barrier_evaluation": arm_id not in {"legacy_mind_d2_s1_collar4", "mind_f222_s1"},
            }
            for index, arm_id in enumerate(c4.ALL_ARM_IDS)
        ]
        self.offsets = c4.canonical_offset_table()
        self.support = {
            "support_id": "C4_COMMON_COLLAR7_NCC7_V1",
            "collar_width": 7,
            "mask_rule": "geometry & common-valid-support",
            "utility_retention_min": 0.99,
            "descriptor_retention_policy": "diagnostic_only_nonempty",
            "utility_id": "COMMON_NCC7",
            "window": 7,
            "improvement_min": 1e-6,
        }
        self.hashes = {
            "policy": c4.payload_sha256(self.policy),
            "arms": c4.payload_sha256(self.arm_specs),
            "offsets": c4.payload_sha256(self.offsets),
            "support": c4.payload_sha256(self.support),
        }
        self.decision = self.build_decision()

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def build_decision(self) -> dict[str, object]:
        return c4.build_decision_contract(
            self.source,
            self.source_sha,
            policy=self.policy,
            expected_policy_sha256=self.hashes["policy"],
            arm_specs=self.arm_specs,
            expected_arm_specs_sha256=self.hashes["arms"],
            offset_table=self.offsets,
            expected_offset_table_sha256=self.hashes["offsets"],
            support_contract=self.support,
            expected_support_contract_sha256=self.hashes["support"],
        )

    def candidate_marker(self, case_id: str | None = None) -> dict[str, object]:
        case_id = case_id or self.case_ids[0]
        shard = next(index for index, cases in enumerate(self.decision["shards"].values()) if case_id in cases)
        arms: list[dict[str, object]] = []
        for spec in self.arm_specs:
            common: dict[str, object] = {
                "arm_index": spec["arm_index"],
                "arm_id": spec["arm_id"],
                "action": "ACCEPT" if spec["materialize_candidate"] else "DIAGNOSTIC_ONLY",
                "support_contract_sha256": self.hashes["support"],
            }
            if not spec["materialize_candidate"]:
                arms.append({**common, "diagnostics": {"status": "PASS"}})
                continue
            path = self.heavy / "cases" / case_id / f"{spec['arm_id']}.npz"
            array = _save_flow(path, float(spec["arm_index"]) / 100.0)
            field = _field_record(path, self.heavy, array)
            arms.append(
                {
                    **common,
                    "support": {
                        "support_id": self.support["support_id"],
                        "baseline_count": 100,
                        "pair_count": 100,
                        "retention": 1.0,
                    },
                    "utility": {
                        "utility_id": self.support["utility_id"],
                        "baseline_loss": 0.5,
                        "candidate_loss": 0.4,
                        "improvement": 0.1,
                    },
                    "candidate_field": field,
                    "persistence": {
                        "protocol_id": c4.PERSISTENCE_PROTOCOL_ID,
                        "saved_npz_sha256": field["npz_sha256"],
                        "reloaded_array_sha256": field["array_sha256"],
                    },
                    "exact": {
                        "status": "CERTIFIED",
                        "certified": True,
                        "sha256": field["array_sha256"],
                    },
                    "returned_field": field,
                    "rollback_to_source_initial": False,
                    "geometry": _geometry_bundle(),
                }
            )
        return {
            "schema": c4.DECISION_CASE_SCHEMA,
            "protocol_id": c4.PROTOCOL_ID,
            "status": "COMPLETE",
            "case_id": case_id,
            "shard_index": shard,
            "physical_gpu": self.decision["shard_to_physical_gpu"][str(shard)],
            "decision_contract_sha256": "c" * 64,
            "arm_specs_sha256": self.hashes["arms"],
            "offset_table_sha256": self.hashes["offsets"],
            "support_contract_sha256": self.hashes["support"],
            "source_image_array_sha256": self.decision["image_inputs"][case_id]["array_sha256"],
            "source_initial_array_sha256": self.decision["source_initial"][case_id]["field"]["array_sha256"],
            "labels_loaded_to_device": False,
            "test_split_accessed": False,
            "support": {"geometry_count": 100, "common_count": 100, "retention": 1.0},
            "baseline_geometry": _geometry_bundle(),
            "arms": arms,
            "execution": _execution(self.decision, "decision", shard, "A_fixture"),
        }


class FrozenProtocolTest(unittest.TestCase):
    def test_arm_ids_and_reference_are_explicit(self) -> None:
        self.assertEqual(len(c4.SCIENTIFIC_ARM_IDS), 8)
        self.assertEqual(len(c4.DIAGNOSTIC_ARM_IDS), 4)
        self.assertEqual(c4.SCIENTIFIC_REFERENCE_ARM_ID, "mind_d2_s1")
        self.assertEqual(len(set(c4.ALL_ARM_IDS)), 12)

    def test_offset_table_is_complete_ordered_and_hashable(self) -> None:
        offsets = c4.canonical_offset_table()
        c4.validate_offset_table(offsets)
        self.assertEqual([row["search_id"] for row in offsets], ["S1", "S2"])
        self.assertEqual(offsets[0]["offsets_zyx"][0], [-1, -1, -1])
        self.assertEqual(offsets[0]["offsets_zyx"][13], [0, 0, 0])
        self.assertEqual(offsets[1]["offsets_zyx"][-1], [2, 2, 2])
        self.assertRegex(c4.payload_sha256(offsets), r"^[0-9a-f]{64}$")

    def test_contract_layer_accepts_the_frozen_policy_owner_without_rehashing(self) -> None:
        specs = [asdict(spec) for spec in policy_owner.ARM_SPECS]
        c4.validate_arm_specs(specs)
        self.assertEqual(c4.payload_sha256(policy_owner.C4_POLICY.to_dict()), policy_owner.C4_POLICY_SHA256)
        self.assertEqual(
            [row["arm_id"] for row in specs if row["materialize_candidate"]],
            [*c4.SCIENTIFIC_ARM_IDS, "intensity_s1", "intensity_s2"],
        )
        self.assertEqual(
            [row["arm_id"] for row in specs if row["post_barrier_evaluation"]],
            [*c4.SCIENTIFIC_ARM_IDS, "intensity_s1", "intensity_s2"],
        )

    def test_common_support_and_label_free_utility_are_frozen(self) -> None:
        c4.validate_support_contract(
            {
                "support_id": "C4_COMMON_COLLAR7_NCC7_V1",
                "collar_width": 7,
                "mask_rule": "geometry & common-valid-support",
                "utility_retention_min": 0.99,
                "descriptor_retention_policy": "diagnostic_only_nonempty",
                "utility_id": "COMMON_NCC7",
                "window": 7,
                "improvement_min": 1e-6,
            }
        )
        self.assertEqual(c4.COMMON_EVIDENCE_COLLAR, 7)
        self.assertEqual(c4.PRIMARY_UTILITY_ID, "COMMON_NCC7")


class ContractHashAndTamperTest(ContractFixture):
    def validate_decision(self, payload: dict[str, object]) -> None:
        c4.validate_decision_contract(
            payload,
            source=self.source,
            expected_source_sha256=self.source_sha,
            expected_policy_sha256=self.hashes["policy"],
            expected_arm_specs_sha256=self.hashes["arms"],
            expected_offset_table_sha256=self.hashes["offsets"],
            expected_support_contract_sha256=self.hashes["support"],
        )

    def test_canonical_contract_round_trip_and_source_byte_tamper(self) -> None:
        run_root = self.root / "run"
        run_root.mkdir()
        source_path = run_root / "source_contract.json"
        source_sha = c4._write_immutable_json(source_path, self.source)
        loaded, actual = c4.load_source_contract(run_root, source_sha)
        self.assertEqual(loaded, self.source)
        self.assertEqual(actual, source_sha)

        source_path.write_text(source_path.read_text(encoding="utf-8") + " ", encoding="utf-8")
        with self.assertRaisesRegex(RuntimeError, "hash mismatch"):
            c4.load_source_contract(run_root, source_sha)

    def test_isolated_loader_never_deserializes_the_source_contract(self) -> None:
        run_root = self.root / "isolated"
        run_root.mkdir()
        source_path = run_root / "source_contract.json"
        source_sha = c4._write_immutable_json(source_path, self.source)
        decision = c4.build_decision_contract(
            self.source,
            source_sha,
            policy=self.policy,
            expected_policy_sha256=self.hashes["policy"],
            arm_specs=self.arm_specs,
            expected_arm_specs_sha256=self.hashes["arms"],
            offset_table=self.offsets,
            expected_offset_table_sha256=self.hashes["offsets"],
            support_contract=self.support,
            expected_support_contract_sha256=self.hashes["support"],
        )
        decision_path = run_root / "decision_contract.json"
        decision_sha = c4._write_immutable_json(decision_path, decision)
        original_load_json = c4._load_json

        def guarded_load_json(path: Path) -> dict[str, object]:
            if path.resolve() == source_path.resolve():
                self.fail("isolated decision loader deserialized source_contract.json")
            return original_load_json(path)

        with patch.object(c4, "_load_json", side_effect=guarded_load_json) as load_json:
            loaded, actual = c4.load_decision_contract_isolated(
                run_root,
                decision_sha,
                expected_source_sha256=source_sha,
                expected_policy_sha256=self.hashes["policy"],
                expected_arm_specs_sha256=self.hashes["arms"],
                expected_offset_table_sha256=self.hashes["offsets"],
                expected_support_contract_sha256=self.hashes["support"],
            )
        self.assertEqual(loaded, decision)
        self.assertEqual(actual, decision_sha)
        load_json.assert_called_once_with(decision_path.resolve())

        source_path.write_text(source_path.read_text(encoding="utf-8") + " ", encoding="utf-8")
        with (
            patch.object(c4, "_load_json", wraps=c4._load_json) as load_json,
            self.assertRaisesRegex(RuntimeError, "source contract byte hash mismatch"),
        ):
            c4.load_decision_contract_isolated(
                run_root,
                decision_sha,
                expected_source_sha256=source_sha,
                expected_policy_sha256=self.hashes["policy"],
                expected_arm_specs_sha256=self.hashes["arms"],
                expected_offset_table_sha256=self.hashes["offsets"],
                expected_support_contract_sha256=self.hashes["support"],
            )
        load_json.assert_not_called()

    def test_isolated_loader_rejects_semantically_invalid_decision_with_matching_byte_hash(self) -> None:
        run_root = self.root / "isolated_tamper"
        run_root.mkdir()
        source_sha = c4._write_immutable_json(run_root / "source_contract.json", self.source)
        decision = c4.build_decision_contract(
            self.source,
            source_sha,
            policy=self.policy,
            expected_policy_sha256=self.hashes["policy"],
            arm_specs=self.arm_specs,
            expected_arm_specs_sha256=self.hashes["arms"],
            offset_table=self.offsets,
            expected_offset_table_sha256=self.hashes["offsets"],
            support_contract=self.support,
            expected_support_contract_sha256=self.hashes["support"],
        )
        decision["labels_available_to_decision_workers"] = True
        decision_sha = c4._write_immutable_json(run_root / "decision_contract.json", decision)
        with self.assertRaisesRegex(RuntimeError, "Invalid or altered"):
            c4.load_decision_contract_isolated(
                run_root,
                decision_sha,
                expected_source_sha256=source_sha,
                expected_policy_sha256=self.hashes["policy"],
                expected_arm_specs_sha256=self.hashes["arms"],
                expected_offset_table_sha256=self.hashes["offsets"],
                expected_support_contract_sha256=self.hashes["support"],
            )

    def test_offset_policy_arm_and_support_tampering_are_rejected(self) -> None:
        mutations: dict[str, dict[str, object]] = {}
        offsets = copy.deepcopy(self.decision)
        offsets["offset_table"][0]["offsets_zyx"][0][-1] = 0
        mutations["offset"] = offsets
        arms = copy.deepcopy(self.decision)
        arms["arm_specs"][0]["arm_id"] = "foreign"
        mutations["arm"] = arms
        policy = copy.deepcopy(self.decision)
        policy["policy"]["scientific_reference_arm_id"] = "foreign"
        mutations["policy"] = policy
        support = copy.deepcopy(self.decision)
        support["support_contract"]["collar_width"] = 4
        mutations["support"] = support

        for label, payload in mutations.items():
            with self.subTest(label=label), self.assertRaises(RuntimeError):
                self.validate_decision(payload)

    def test_source_contract_cannot_authorize_test_115(self) -> None:
        payload = copy.deepcopy(self.source)
        payload["test_115_authorized"] = True
        with self.assertRaisesRegex(RuntimeError, "Invalid or altered"):
            c4.validate_source_contract(payload)

    def test_evaluation_baselines_stay_out_of_the_label_free_decision_contract(self) -> None:
        self.assertIn("evaluation_baseline_dice", self.source)
        self.assertNotIn("evaluation_baseline_dice", self.decision)
        serialized = json.dumps(self.decision, sort_keys=True).lower()
        self.assertNotIn('"dice', serialized)

        payload = copy.deepcopy(self.source)
        payload["source_historical"].pop(self.case_ids[0])
        with self.assertRaisesRegex(RuntimeError, "historical-field"):
            c4.validate_source_contract(payload)


class DecisionMarkerTamperTest(ContractFixture):
    def test_valid_marker_authenticates_saved_and_reloaded_heavy_bytes(self) -> None:
        marker = self.candidate_marker()
        c4.validate_decision_case_marker(marker, self.decision, "c" * 64)

    def test_foreign_case_wrong_shard_and_label_leak_are_rejected(self) -> None:
        base = self.candidate_marker()
        mutations = {
            "foreign_case": {**base, "case_id": "subject_foreign"},
            "wrong_shard": {**base, "shard_index": (base["shard_index"] + 1) % 5},
            "labels_true": {**base, "labels_loaded_to_device": True},
            "dice": {**base, "diagnostic_dice": 0.9},
            "label_payload": {**base, "moving_labels": [1, 2]},
        }
        for label, marker in mutations.items():
            with self.subTest(label=label), self.assertRaises(RuntimeError):
                c4.validate_decision_case_marker(marker, self.decision, "c" * 64, verify_heavy_bytes=False)
        nondeterministic = copy.deepcopy(base)
        nondeterministic["execution"]["deterministic"] = False
        with self.assertRaisesRegex(RuntimeError, "execution provenance"):
            c4.validate_decision_case_marker(
                nondeterministic,
                self.decision,
                "c" * 64,
                verify_heavy_bytes=False,
            )

    def test_candidate_file_and_persistence_tampering_are_rejected(self) -> None:
        marker = self.candidate_marker()
        first_path = self.heavy / marker["arms"][0]["candidate_field"]["relative_path"]
        first_path.write_bytes(first_path.read_bytes() + b"tamper")
        with self.assertRaisesRegex(RuntimeError, "missing or changed"):
            c4.validate_decision_case_marker(marker, self.decision, "c" * 64)

        marker = self.candidate_marker(self.case_ids[1])
        marker["arms"][0]["persistence"]["reloaded_array_sha256"] = "f" * 64
        with self.assertRaisesRegex(RuntimeError, "save/reload"):
            c4.validate_decision_case_marker(marker, self.decision, "c" * 64, verify_heavy_bytes=False)

        marker = self.candidate_marker(self.case_ids[2])
        marker["arms"][0]["exact"]["certified"] = False
        marker["arms"][0]["exact"]["status"] = "FAILED"
        with self.assertRaisesRegex(RuntimeError, "save/reload"):
            c4.validate_decision_case_marker(marker, self.decision, "c" * 64, verify_heavy_bytes=False)

        marker = self.candidate_marker(self.case_ids[3])
        marker["arms"][0]["geometry"].pop(next(iter(c4.METRIC_SPECS)))
        with self.assertRaisesRegex(RuntimeError, "geometry metric bundle"):
            c4.validate_decision_case_marker(marker, self.decision, "c" * 64, verify_heavy_bytes=False)

    def test_common_support_utility_and_transaction_action_are_recomputed(self) -> None:
        base = self.candidate_marker()
        wrong_action = copy.deepcopy(base)
        wrong_action["arms"][0]["action"] = "ROLLBACK"
        wrong_retention = copy.deepcopy(base)
        wrong_retention["arms"][0]["support"]["retention"] = 0.5
        wrong_arithmetic = copy.deepcopy(base)
        wrong_arithmetic["arms"][0]["utility"]["improvement"] = 0.2
        for label, marker in {
            "action": wrong_action,
            "retention": wrong_retention,
            "arithmetic": wrong_arithmetic,
        }.items():
            with self.subTest(label=label), self.assertRaises(RuntimeError):
                c4.validate_decision_case_marker(marker, self.decision, "c" * 64, verify_heavy_bytes=False)


class BarrierContractTest(ContractFixture):
    def worker(self, phase: str, shard: int, attempt: str, barrier_sha: str | None = None) -> dict[str, object]:
        payload: dict[str, object] = {
            "schema": c4.WORKER_SCHEMA,
            "protocol_id": c4.PROTOCOL_ID,
            "status": "COMPLETE",
            "phase": phase,
            "attempt_id": attempt,
            "shard_index": shard,
            "physical_gpu": self.decision["shard_to_physical_gpu"][str(shard)],
            "case_ids": self.decision["shards"][str(shard)],
            "decision_contract_sha256": "c" * 64,
            "test_split_accessed": False,
        }
        if phase == "decision":
            payload["labels_loaded_to_device"] = False
        else:
            payload["decision_barrier_sha256"] = barrier_sha
            payload["labels_loaded_after_barrier"] = True
        payload["execution"] = _execution(self.decision, phase, shard, attempt)
        return payload

    def test_worker_case_order_gpu_and_label_isolation_are_frozen(self) -> None:
        marker = self.worker("decision", 0, "A_fixture")
        c4.validate_worker_marker(
            marker,
            self.decision,
            "c" * 64,
            phase="decision",
            shard_index=0,
            attempt_id="A_fixture",
        )
        for change in (
            {"case_ids": list(reversed(marker["case_ids"]))},
            {"physical_gpu": "7"},
            {"labels_loaded_to_device": True},
        ):
            with self.assertRaises(RuntimeError):
                c4.validate_worker_marker(
                    {**marker, **change},
                    self.decision,
                    "c" * 64,
                    phase="decision",
                    shard_index=0,
                    attempt_id="A_fixture",
                )

    def test_post_barrier_evaluation_is_bound_to_immutable_barrier_hash(self) -> None:
        case_id = self.case_ids[0]
        barrier_sha = "d" * 64
        decision_case_sha = "e" * 64
        barrier = {
            "decision_case_sha256": {case_id: decision_case_sha},
        }
        evaluation = {
            "schema": c4.EVALUATION_CASE_SCHEMA,
            "protocol_id": c4.PROTOCOL_ID,
            "status": "COMPLETE",
            "case_id": case_id,
            "decision_contract_sha256": "c" * 64,
            "decision_barrier_sha256": barrier_sha,
            "decision_case_sha256": decision_case_sha,
            "labels_loaded_after_barrier": True,
            "test_split_accessed": False,
            "labels": [1, 2],
            "baseline_c3_parity_verified": True,
            "execution": _execution(self.decision, "evaluation", 0, "A_fixture"),
            "arms": [
                {
                    "arm_index": index,
                    "arm_id": arm_id,
                    "evaluated": self.arm_specs[index]["post_barrier_evaluation"],
                    **(
                        {
                            "baseline_dice": 0.7,
                            "capacity_candidate_dice": 0.71,
                            "capacity_dice_delta": 0.01,
                            "primary_returned_dice": 0.71,
                            "primary_dice_delta": 0.01,
                            "primary_action": "ACCEPT",
                            "per_label": [
                                {
                                    "label": label,
                                    "baseline_dice": 0.7,
                                    "candidate_dice": 0.71,
                                    "returned_dice": 0.71,
                                }
                                for label in (1, 2)
                            ],
                        }
                        if self.arm_specs[index]["post_barrier_evaluation"]
                        else {"primary_action": "DIAGNOSTIC_ONLY"}
                    ),
                }
                for index, arm_id in enumerate(c4.ALL_ARM_IDS)
            ],
        }
        c4.validate_evaluation_case_marker(evaluation, self.decision, "c" * 64, barrier, barrier_sha)

        for change in (
            {"decision_barrier_sha256": "f" * 64},
            {"decision_case_sha256": "0" * 64},
            {"labels_loaded_after_barrier": False},
        ):
            with self.assertRaises(RuntimeError):
                c4.validate_evaluation_case_marker(
                    {**evaluation, **change}, self.decision, "c" * 64, barrier, barrier_sha
                )
        bad_arithmetic = copy.deepcopy(evaluation)
        bad_arithmetic["arms"][0]["capacity_dice_delta"] = 0.02
        bad_execution = copy.deepcopy(evaluation)
        bad_execution["execution"]["deterministic"] = False
        for marker in (bad_arithmetic, bad_execution):
            with self.assertRaises(RuntimeError):
                c4.validate_evaluation_case_marker(marker, self.decision, "c" * 64, barrier, barrier_sha)


class FrozenC3AuthenticationTest(unittest.TestCase):
    def build_fixture(self, root: Path) -> tuple[Path, Path, str, str]:
        compact = root / "compact"
        heavy = root / "source_heavy"
        raw_root = root / "raw"
        compact.mkdir()
        heavy.mkdir()
        raw_root.mkdir()
        case_ids = ["subject_1", "subject_2"]

        raw_inputs: dict[str, dict[str, object]] = {}
        image_inputs: dict[str, dict[str, object]] = {}
        initial_records: dict[str, tuple[dict[str, str], dict[str, object], dict[str, str]]] = {}
        for case_id in ["atlas", *case_ids]:
            raw_path = raw_root / f"{case_id}.bin"
            raw_path.write_bytes(case_id.encode("utf-8"))
            raw_inputs[case_id] = _file_record(
                raw_path,
                case_id=case_id,
                split="atlas" if case_id == "atlas" else "val",
            )
            image = np.full((1, 3, 3, 3), len(case_id), dtype=np.float32)
            image_path = heavy / "images" / f"{case_id}.npy"
            image_path.parent.mkdir(exist_ok=True)
            np.save(image_path, image, allow_pickle=False)
            image_inputs[case_id] = _image_record(image_path, image)
            if case_id != "atlas":
                flow_path = heavy / "cases" / case_id / "initial.npz"
                flow = _save_flow(flow_path, 0.0)
                field = _field_record(flow_path, heavy, flow)
                historical_path = heavy / "cases" / case_id / "raw_conf.npz"
                historical = _save_flow(historical_path, 0.01)
                historical_field = _field_record(historical_path, heavy, historical)
                initial_records[case_id] = (
                    field,
                    {"status": "CERTIFIED", "certified": True, "sha256": field["array_sha256"]},
                    historical_field,
                )

        source = {
            "schema": "ctcf-search-c3a-source-contract-v1",
            "git_head": "1" * 40,
            "heavy_root": str(heavy.resolve()),
            "case_ids": case_ids,
            "raw_inputs": raw_inputs,
            "seed": 0,
            "runtime_signature": {"python": "fixture"},
            "ixi_test_split_accessed": False,
            "test_115_authorized": False,
        }
        _write_json(compact / "source_contract.json", source)
        source_sha = sha256_file(compact / "source_contract.json")
        decision = {
            "schema": "ctcf-search-c3a-decision-contract-v1",
            "git_head": "1" * 40,
            "heavy_root": str(heavy.resolve()),
            "case_ids": case_ids,
            "image_inputs": image_inputs,
            "source_contract_sha256": source_sha,
            "ixi_test_split_accessed": False,
            "test_115_authorized": False,
        }
        _write_json(compact / "decision_contract.json", decision)
        decision_sha = sha256_file(compact / "decision_contract.json")

        decision_hashes: dict[str, str] = {}
        evaluation_hashes: dict[str, str] = {}
        for case_id in case_ids:
            field, exact, historical_field = initial_records[case_id]
            decision_case = {
                "schema": "ctcf-search-c3a-decision-case-v1",
                "status": "COMPLETE",
                "case_id": case_id,
                "labels_loaded_to_device": False,
                "test_split_accessed": False,
                "initial": {"field": field, "report": {"psi_exact": exact}},
                "arms": [
                    {
                        "arm_id": "raw_conf_post1",
                        "requested_state": {"field": historical_field},
                    }
                ],
            }
            evaluation_case = {
                "schema": "ctcf-search-c3a-evaluation-case-v1",
                "status": "COMPLETE",
                "case_id": case_id,
                "labels_loaded_after_barrier": True,
                "test_split_accessed": False,
                "arms": [{"baseline_dice": 0.75}, {"baseline_dice": 0.75}],
            }
            decision_path = compact / "cases" / case_id / "decision_complete.json"
            evaluation_path = compact / "cases" / case_id / "evaluation_complete.json"
            _write_json(decision_path, decision_case)
            _write_json(evaluation_path, evaluation_case)
            decision_hashes[case_id] = sha256_file(decision_path)
            evaluation_hashes[case_id] = sha256_file(evaluation_path)

        barrier = {
            "schema": "ctcf-search-c3a-decision-barrier-v1",
            "decision_contract_sha256": decision_sha,
            "decision_workers_received_label_inputs": False,
            "test_split_accessed": False,
            "decision_case_sha256": decision_hashes,
        }
        _write_json(compact / "decision_barrier.json", barrier)
        barrier_sha = sha256_file(compact / "decision_barrier.json")
        required_files = {
            "datasets_sha256": "datasets.csv",
            "c2_decision_goldens_sha256": "c2_decision_goldens.json",
            "c2_evaluation_goldens_sha256": "c2_evaluation_goldens.json",
            "per_arm_sha256": "per_arm.csv",
            "arm_summary_sha256": "arm_summary.csv",
            "hypotheses_sha256": "hypotheses.json",
            "summary_sha256": "summary.json",
        }
        files = {
            "source_contract_sha256": source_sha,
            "decision_contract_sha256": decision_sha,
            "decision_barrier_sha256": barrier_sha,
        }
        for key, name in required_files.items():
            path = compact / name
            path.write_text(f"fixture:{name}\n", encoding="utf-8")
            files[key] = sha256_file(path)
        datasets_tsv = compact / "datasets.tsv"
        datasets_tsv.write_text("fixture\n", encoding="utf-8")

        manifest = {
            "schema": "ctcf-search-c3a-run-manifest-v1",
            "protocol_id": "CTCF-SEARCH-GATE-C3A-V1",
            "run_id": "fixture-c3",
            "status": "COMPLETE",
            "code": {"git_head": "1" * 40, "git_status": ""},
            "storage": {"heavy_root": str(heavy.resolve())},
            "summary": {
                "execution_integrity_status": "PASS",
                "n_cases": 2,
                "test_115_authorized": False,
                "test_split_accessed": False,
                "labels_used_for_decision": False,
            },
            "files": files,
            "decision_case_sha256": decision_hashes,
            "evaluation_case_sha256": evaluation_hashes,
        }
        _write_json(compact / "c3_manifest.json", manifest)
        run_manifest = {
            "schema": "ctcf-native-manifest-v1",
            "run_id": "fixture-c3",
            "status": "COMPLETE",
            "code": {"git_head": "1" * 40, "tracked_tree_clean_at_start": True},
            "files": {"datasets_sha256": sha256_file(datasets_tsv)},
            "exit_code": 0,
        }
        _write_json(compact / "run_manifest.json", run_manifest)
        return compact, heavy, sha256_file(compact / "c3_manifest.json"), sha256_file(compact / "run_manifest.json")

    def test_source_c3_is_authenticated_and_raw_byte_tamper_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            compact, heavy, manifest_sha, run_manifest_sha = self.build_fixture(Path(temporary))
            patches = (
                patch.object(c4, "EXPECTED_CASES", 2),
                patch.object(c4, "SOURCE_C3_RUN_ID", "fixture-c3"),
                patch.object(c4, "SOURCE_C3_GIT_HEAD", "1" * 40),
                patch.object(c4, "SOURCE_C3_MANIFEST_SHA256", manifest_sha),
                patch.object(c4, "SOURCE_C3_RUN_MANIFEST_SHA256", run_manifest_sha),
            )
            with patches[0], patches[1], patches[2], patches[3], patches[4]:
                snapshot = c4.authenticate_frozen_c3(compact, heavy)
                self.assertEqual(snapshot["case_ids"], ["subject_1", "subject_2"])
                raw_path = Path(snapshot["raw_inputs"]["subject_1"]["path"])
                raw_path.write_bytes(b"tampered")
                with self.assertRaisesRegex(RuntimeError, "bytes changed"):
                    c4.authenticate_frozen_c3(compact, heavy)


if __name__ == "__main__":
    unittest.main()
