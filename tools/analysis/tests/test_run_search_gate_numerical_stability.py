from __future__ import annotations

import argparse
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from tools.analysis import (
    run_search_gate_numerical_stability as runner,
    search_gate_numerical_stability as policy,
)
from tools.analysis.transactional_search import geometry_mask, save_flow_npz_atomic


class DatasetProjectionContractTest(unittest.TestCase):
    def test_csv_and_tsv_with_the_same_rows_are_semantically_equal(self) -> None:
        csv_text = "dataset,split,case_id,path\nIXI,val,subject_1,/data/subject_1.pkl\n"
        tsv_text = "dataset\tsplit\tcase_id\tpath\nIXI\tval\tsubject_1\t/data/subject_1.pkl\n"

        csv_projection = runner._table_rows(csv_text, delimiter=",", label="fixture.csv")
        tsv_projection = runner._table_rows(tsv_text, delimiter="\t", label="fixture.tsv")

        self.assertEqual(csv_projection, tsv_projection)

    def test_csv_and_tsv_with_different_rows_are_not_equivalent(self) -> None:
        csv_text = "dataset,split,case_id,path\nIXI,val,subject_1,/data/subject_1.pkl\n"
        tsv_text = "dataset\tsplit\tcase_id\tpath\nIXI\tval\tsubject_2\t/data/subject_2.pkl\n"

        csv_projection = runner._table_rows(csv_text, delimiter=",", label="fixture.csv")
        tsv_projection = runner._table_rows(tsv_text, delimiter="\t", label="fixture.tsv")

        self.assertNotEqual(csv_projection, tsv_projection)


class DecisionContractRoundTripTest(unittest.TestCase):
    @staticmethod
    def payload() -> dict[str, object]:
        return {
            "schema": f"ctcf-search-gate-numstab-decision-contract-{runner.SCHEMA_VERSION}",
            "protocol_id": runner.PROTOCOL_ID,
            "policy_sha256": runner.NUMERICAL_STABILITY_POLICY_SHA256,
            "policy": policy.NUMERICAL_STABILITY_POLICY.to_dict(),
            "source_c3_manifest_sha256": runner.SOURCE_C3_MANIFEST_SHA256,
            "source_c3_run_manifest_sha256": runner.SOURCE_C3_RUN_MANIFEST_SHA256,
            "decision_contract_contains_label_data": False,
            "decision_worker_uses_raw_containers": False,
            "labels_available_to_decision_workers": False,
            "ixi_test_split_accessed": False,
            "test_115_authorized": False,
        }

    def test_written_contract_survives_json_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / runner.DECISION_CONTRACT_NAME
            runner.atomic_write_json(path, self.payload())
            expected_sha = runner.sha256_file(path)

            loaded, actual_sha = runner._load_decision(root, expected_sha)

            self.assertEqual(actual_sha, expected_sha)
            self.assertTrue(
                runner._json_equivalent(
                    loaded["policy"],
                    policy.NUMERICAL_STABILITY_POLICY.to_dict(),
                )
            )

    def test_semantically_changed_policy_is_rejected_after_json_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / runner.DECISION_CONTRACT_NAME
            payload = self.payload()
            payload["policy"]["source_c3_git_head"] = "0" * 40
            runner.atomic_write_json(path, payload)

            with self.assertRaisesRegex(RuntimeError, "Invalid NUMSTAB decision contract"):
                runner._load_decision(root, runner.sha256_file(path))


class BarrierResumeContractTest(unittest.TestCase):
    def test_existing_barrier_is_reused_by_a_new_attempt(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            case_id = "subject_1"
            old_attempt = "A_old"
            new_attempt = "A_new"
            contract_sha = "a" * 64
            contract = {
                "git_head": "b" * 40,
                "num_shards": 1,
                "shards": {"0": [case_id]},
                "case_ids": [case_id],
            }
            decision_path = root / "cases" / case_id / "decision_complete.json"
            decision_path.parent.mkdir(parents=True)
            decision_path.write_text(json.dumps({"case_id": case_id}), encoding="utf-8")

            old_worker = root / "workers" / "decision" / "attempts" / old_attempt / "worker_00.json"
            new_worker = root / "workers" / "decision" / "attempts" / new_attempt / "worker_00.json"
            for path, attempt in ((old_worker, old_attempt), (new_worker, new_attempt)):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps({"attempt_id": attempt}), encoding="utf-8")

            barrier_path = root / runner.BARRIER_NAME
            barrier_payload = {
                "schema": f"ctcf-search-gate-numstab-decision-barrier-{runner.SCHEMA_VERSION}",
                "status": "COMPLETE",
                "protocol_id": runner.PROTOCOL_ID,
                "attempt_id": old_attempt,
                "decision_contract_sha256": contract_sha,
                "decision_workers_received_label_inputs": False,
                "test_split_accessed": False,
                "workers": [
                    {
                        "path": old_worker.relative_to(root).as_posix(),
                        "sha256": runner.sha256_file(old_worker),
                    }
                ],
                "decision_case_sha256": {case_id: runner.sha256_file(decision_path)},
                "completed_at_utc": "2026-08-23T00:00:00Z",
            }
            barrier_path.write_text(json.dumps(barrier_payload), encoding="utf-8")
            barrier_before = barrier_path.read_bytes()
            args = argparse.Namespace(
                run_root=root,
                decision_contract_sha256=contract_sha,
                attempt_id=new_attempt,
            )

            with (
                patch.object(runner, "_load_decision", return_value=(contract, contract_sha)),
                patch.object(runner, "_assert_clean_code"),
                patch.object(runner, "_validate_worker"),
                patch.object(runner, "validate_shard_partition"),
                patch.object(runner, "_validate_decision_case", return_value=[]),
            ):
                result = runner.barrier_stage(args)

            self.assertEqual(result, 0)
            self.assertEqual(barrier_path.read_bytes(), barrier_before)


class PostclipOracleContractTest(unittest.TestCase):
    def test_stored_candidate_and_action_must_both_match_the_fp64_oracle(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            heavy = Path(tmp)
            shape = (9, 9, 9)
            arms: list[dict[str, object]] = []
            for spec in policy.SCIENTIFIC_ARMS:
                path = heavy / f"{spec.arm_id}.npz"
                field = torch.zeros((1, 3, *shape), dtype=torch.float32)
                save_flow_npz_atomic(path, field)
                arms.append(
                    {
                        "arm_id": spec.arm_id,
                        "action": "ACCEPT",
                        "candidate": {
                            "field": runner._field_record(path, heavy, runner._sha256_array(field)),
                        },
                    }
                )
            mask = geometry_mask(shape, 4, torch.device("cpu"))

            faithful = runner._postclip_oracle_pairs(arms, heavy, mask)
            self.assertTrue(all(row["faithful"] for row in faithful))

            arms[-2]["action"] = "ROLLBACK"
            mismatched = runner._postclip_oracle_pairs(arms, heavy, mask)
            self.assertFalse(mismatched[0]["faithful"])
            self.assertFalse(mismatched[0]["action_equal"])


class EligibilityAndBranchContractTest(unittest.TestCase):
    @staticmethod
    def paired(mean: float, median: float | None = None, ci_low: float | None = None) -> dict[str, float]:
        return {
            "mean": mean,
            "median": mean if median is None else median,
            "ci_low": mean if ci_low is None else ci_low,
        }

    @staticmethod
    def summaries(**updates: dict[str, object]) -> list[dict[str, object]]:
        rows: list[dict[str, object]] = []
        for spec in policy.SCIENTIFIC_ARMS:
            row: dict[str, object] = {
                "arm_index": spec.arm_index,
                "arm_id": spec.arm_id,
                "role": spec.role,
                "selectable": spec.selectable,
                "material_capacity": False,
                "viable_primary_policy": False,
                "capacity_geometry_noninferior": False,
                "capacity_vs_baseline": {"mean": 0.0},
            }
            row.update(updates.get(spec.arm_id, {}))
            rows.append(row)
        return rows

    def test_comparator_worse_arm_is_neither_material_nor_viable(self) -> None:
        result = policy.assess_arm_eligibility(
            selectable=True,
            oracle_faithful_all_cases=True,
            all_candidate_exact=True,
            all_returned_exact=True,
            all_support_defined=True,
            primary_geometry_noninferior=True,
            capacity_vs_baseline=self.paired(0.003, 0.002, 0.0015),
            primary_vs_baseline=self.paired(0.0015, 0.001, 0.0005),
            capacity_vs_legacy=self.paired(-0.0002),
            primary_vs_legacy=self.paired(-0.0001),
        )

        self.assertTrue(result["material_capacity_vs_baseline"])
        self.assertTrue(result["practical_primary_policy_vs_baseline"])
        self.assertFalse(result["capacity_superior_to_legacy"])
        self.assertFalse(result["primary_policy_superior_to_legacy"])
        self.assertFalse(result["material_capacity"])
        self.assertFalse(result["viable_primary_policy"])

    def test_oracle_failure_prevents_c32_advance(self) -> None:
        arm_id = policy.SCIENTIFIC_ARMS[0].arm_id
        rows = self.summaries(
            **{
                arm_id: {
                    "material_capacity": True,
                    "viable_primary_policy": True,
                    "capacity_geometry_noninferior": True,
                    "capacity_vs_baseline": {"mean": 0.003},
                }
            }
        )

        branch = policy.select_next_branch(rows, oracle_faithful_all_cases=False)

        self.assertNotEqual(branch["branch_id"], policy.BRANCH_ADVANCE_C32)
        self.assertEqual(branch["branch_id"], policy.BRANCH_OPEN_FP64)

    def test_fp64_material_signal_is_nonviable_and_opens_precision_study(self) -> None:
        oracle_spec = next(spec for spec in policy.SCIENTIFIC_ARMS if spec.role == "precision_oracle")
        eligibility = policy.assess_arm_eligibility(
            selectable=oracle_spec.selectable,
            oracle_faithful_all_cases=True,
            all_candidate_exact=True,
            all_returned_exact=True,
            all_support_defined=True,
            primary_geometry_noninferior=True,
            capacity_vs_baseline=self.paired(0.003, 0.002, 0.0015),
            primary_vs_baseline=self.paired(0.0015, 0.001, 0.0005),
            capacity_vs_legacy=self.paired(0.001, 0.0008, 0.0002),
            primary_vs_legacy=self.paired(0.0005, 0.0003, 0.0001),
        )
        rows = self.summaries(
            **{
                oracle_spec.arm_id: {
                    "material_capacity": eligibility["material_capacity"],
                    "viable_primary_policy": eligibility["viable_primary_policy"],
                    "capacity_geometry_noninferior": True,
                    "capacity_vs_baseline": {"mean": 0.003},
                }
            }
        )

        branch = policy.select_next_branch(rows, oracle_faithful_all_cases=True)

        self.assertTrue(eligibility["material_capacity"])
        self.assertFalse(eligibility["viable_primary_policy"])
        self.assertEqual(branch["branch_id"], policy.BRANCH_OPEN_FP64)
        self.assertEqual(branch["parent_arm_id"], oracle_spec.arm_id)

    def test_faithful_material_viable_c32_arm_advances(self) -> None:
        arm_id = policy.SCIENTIFIC_ARMS[0].arm_id
        rows = self.summaries(
            **{
                arm_id: {
                    "material_capacity": True,
                    "viable_primary_policy": True,
                    "capacity_geometry_noninferior": True,
                    "capacity_vs_baseline": {"mean": 0.003},
                }
            }
        )

        branch = policy.select_next_branch(rows, oracle_faithful_all_cases=True)

        self.assertEqual(branch["branch_id"], policy.BRANCH_ADVANCE_C32)
        self.assertEqual(branch["parent_arm_id"], arm_id)

    def test_faithful_material_c32_with_policy_or_geometry_price_opens_c3b(self) -> None:
        arm_id = policy.SCIENTIFIC_ARMS[0].arm_id
        rows = self.summaries(
            **{
                arm_id: {
                    "material_capacity": True,
                    "viable_primary_policy": False,
                    "capacity_geometry_noninferior": False,
                    "capacity_vs_baseline": {"mean": 0.003},
                }
            }
        )

        branch = policy.select_next_branch(rows, oracle_faithful_all_cases=True)

        self.assertEqual(branch["branch_id"], policy.BRANCH_OPEN_C3B)
        self.assertEqual(branch["parent_arm_id"], arm_id)

    def test_absent_material_signal_closes_single_scale_radius1_mind(self) -> None:
        branch = policy.select_next_branch(self.summaries(), oracle_faithful_all_cases=True)

        self.assertEqual(branch["branch_id"], policy.BRANCH_CLOSE_SINGLE_SCALE)
        self.assertIsNone(branch["parent_arm_id"])

    def test_winner_tie_within_tolerance_prefers_lower_arm_index(self) -> None:
        first, second = policy.SCIENTIFIC_ARMS[:2]
        rows = self.summaries(
            **{
                first.arm_id: {
                    "material_capacity": True,
                    "viable_primary_policy": True,
                    "capacity_geometry_noninferior": True,
                    "capacity_vs_baseline": {"mean": 0.003},
                },
                second.arm_id: {
                    "material_capacity": True,
                    "viable_primary_policy": True,
                    "capacity_geometry_noninferior": True,
                    "capacity_vs_baseline": {"mean": 0.0030005},
                },
            }
        )

        branch = policy.select_next_branch(rows, oracle_faithful_all_cases=True)

        self.assertEqual(branch["branch_id"], policy.BRANCH_ADVANCE_C32)
        self.assertEqual(branch["parent_arm_id"], first.arm_id)


if __name__ == "__main__":
    unittest.main()
