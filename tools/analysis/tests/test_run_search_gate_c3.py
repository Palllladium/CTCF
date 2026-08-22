from __future__ import annotations

import json
import math
import os
import tempfile
import unittest
from pathlib import Path

import torch

from tools.analysis.run_artifacts import atomic_write_json, sha256_file
from tools.analysis.run_search_gate_c3 import (
    DECISION_CONTRACT_NAME,
    _build_requested_arms,
    _build_summaries,
    _contrast,
    _extract_c2_goldens,
    _load_decision_contract,
    _validate_arm_construction_invariants,
    _validate_c2_source,
    _validate_disk_budget,
    _validate_evaluation_case,
    _validate_support_utility,
    _validate_worker_report,
    build_parser,
    main,
)
from tools.analysis.search_gate_c3 import (
    ARM_SPECS,
    C3A_POLICY,
    C3A_POLICY_SHA256,
    FLOAT32_PARITY_ATOL,
    FLOAT32_PARITY_RTOL,
    PROTOCOL_ID,
)
from tools.analysis.search_gate_cost_volume import (
    POSTERIOR_DIAGNOSTICS_ID,
    RMS_FIRST_DIFFERENCE_ROUGHNESS_ID,
)
from tools.analysis.search_gate_metrics import DETJ_DIAGNOSTICS, DIGITAL_DECOMPOSITION, MATHEMATICAL_SDLOGJ_CROP2
from tools.analysis.transactional_search import geometry_mask, masked_zscore, mind_ssc

REPO_ROOT = Path(__file__).resolve().parents[3]
C2_PRODUCT = REPO_ROOT / "results/search_gate_c2/C2_DEVELOPMENT_20260822T111122Z_f31aaf8a39b9"
C2_MANIFEST_SHA256 = "3ece06588d6d4d1995f12b150a137e6d1e33bd9b020739279d99d7c8cfe2f6a9"
REQUIRE_HISTORICAL = os.environ.get("CTCF_REQUIRE_HISTORICAL_GOLDENS") == "1"

needs_c2 = unittest.skipIf(
    not C2_PRODUCT.is_dir() and not REQUIRE_HISTORICAL,
    "historical refactor-parity C2 product absent; set CTCF_REQUIRE_HISTORICAL_GOLDENS=1 to fail",
)


@needs_c2
class FrozenC2ProjectionTest(unittest.TestCase):
    def test_latest_complete_c2_projects_exact_control_goldens(self):
        manifest, contract = _validate_c2_source(C2_PRODUCT, C2_MANIFEST_SHA256)
        decision_goldens, evaluation_goldens = _extract_c2_goldens(C2_PRODUCT, manifest, contract)
        self.assertEqual(len(decision_goldens["case_ids"]), 58)
        self.assertEqual(set(decision_goldens["cases"]), set(decision_goldens["case_ids"]))
        self.assertEqual(
            decision_goldens["cases"]["subject_344"]["control_candidate_array_sha256"]["c1_raw_conf_post1"],
            "8f93c30f63084af9fe4b49623505090a57ebe2bea5b825ca160af9e6713f83c8",
        )
        self.assertEqual(decision_goldens["source_manifest_sha256"], C2_MANIFEST_SHA256)
        self.assertNotIn("dice", json.dumps(decision_goldens).lower())
        self.assertAlmostEqual(evaluation_goldens["cases"]["subject_344"]["baseline_dice"], 0.7779488799756565)

    def test_wrong_c2_manifest_hash_is_refused(self):
        with self.assertRaises(RuntimeError):
            _validate_c2_source(C2_PRODUCT, "0" * 64)


class DecisionContractIsolationTest(unittest.TestCase):
    def setUp(self):
        holder = tempfile.TemporaryDirectory()
        self.addCleanup(holder.cleanup)
        self.root = Path(holder.name)
        self.contract = {
            "schema": "ctcf-search-c3a-decision-contract-v1",
            "protocol_id": PROTOCOL_ID,
            "policy": C3A_POLICY.to_dict(),
            "policy_sha256": C3A_POLICY_SHA256,
            "decision_contract_contains_label_data": False,
            "decision_worker_uses_raw_containers": False,
            "os_level_data_isolation_claimed": False,
            "ixi_test_split_accessed": False,
            "test_115_authorized": False,
        }

    def write(self) -> str:
        path = self.root / DECISION_CONTRACT_NAME
        atomic_write_json(path, self.contract)
        return sha256_file(path)

    def test_minimal_typed_contract_is_accepted(self):
        digest = self.write()
        observed, actual = _load_decision_contract(self.root, digest)
        self.assertEqual(observed, self.contract)
        self.assertEqual(actual, digest)

    def test_label_authorization_and_raw_container_paths_are_refused(self):
        self.contract["decision_contract_contains_label_data"] = True
        with self.assertRaises(RuntimeError):
            _load_decision_contract(self.root, self.write())
        self.contract["decision_contract_contains_label_data"] = False
        self.contract["leak"] = "/data/subject.pkl"
        with self.assertRaises(RuntimeError):
            _load_decision_contract(self.root, self.write())

    def test_label_derived_c2_goldens_are_refused(self):
        self.contract["c2_decision_goldens"] = {"cases": {"subject": {"baseline_dice": 0.8}}}
        with self.assertRaises(RuntimeError):
            _load_decision_contract(self.root, self.write())


class FailClosedRuntimeHelperTest(unittest.TestCase):
    def test_empty_support_is_nullable_but_nonempty_support_is_not(self):
        supports = {
            "mind": {
                "utility_id": "COMMON_MIND_SSC",
                "window": 1,
                "baseline_count": 10,
                "pair_count": 0,
                "retention": 0.0,
            },
            "ncc7": {
                "utility_id": "COMMON_NCC7",
                "window": 7,
                "baseline_count": 10,
                "pair_count": 0,
                "retention": 0.0,
            },
            "ncc9": {
                "utility_id": "COMMON_NCC9",
                "window": 9,
                "baseline_count": 10,
                "pair_count": 0,
                "retention": 0.0,
            },
        }
        utility = {
            f"{key}_{side}_common": None for key in ("mind", "ncc7", "ncc9") for side in ("baseline", "candidate")
        }
        _validate_support_utility(supports, utility, label="empty-positive")
        utility["ncc7_baseline_common"] = 0.1
        with self.assertRaises(RuntimeError):
            _validate_support_utility(supports, utility, label="empty-finite-tamper")
        utility["ncc7_baseline_common"] = None
        supports["ncc7"].update(pair_count=10, retention=1.0)
        with self.assertRaises(RuntimeError):
            _validate_support_utility(supports, utility, label="nonempty-null-tamper")

    def test_resume_disk_budget_accounts_for_current_run_bytes(self):
        _validate_disk_budget(free_gib=20.0, heavy_gib=100.0, min_free_gib=120.0, is_resume=True)
        with self.assertRaises(RuntimeError):
            _validate_disk_budget(free_gib=4.0, heavy_gib=116.0, min_free_gib=120.0, is_resume=True)
        with self.assertRaises(RuntimeError):
            _validate_disk_budget(free_gib=20.0, heavy_gib=90.0, min_free_gib=120.0, is_resume=True)
        with self.assertRaises(RuntimeError):
            _validate_disk_budget(free_gib=119.0, heavy_gib=0.0, min_free_gib=120.0, is_resume=False)

    def test_reused_evaluation_attempt_does_not_claim_loading_labels(self):
        case_id = "subject_000"
        contract = {
            "shards": {"0": [case_id]},
            "shard_to_physical_gpu": {"0": 2},
            "runtime_signature": {"python": "frozen"},
        }
        report = {
            "schema": "ctcf-search-c3a-evaluation-worker-v1",
            "status": "COMPLETE",
            "phase": "evaluation",
            "attempt_id": "A_RESUME",
            "shard_index": 0,
            "physical_gpu": 2,
            "decision_contract_sha256": "1" * 64,
            "assigned_case_ids": [case_id],
            "computed_case_ids": [],
            "reused_case_ids": [case_id],
            "runtime_signature": {"python": "frozen"},
            "labels_loaded_this_attempt": False,
            "all_cases_have_postbarrier_evaluation_evidence": True,
        }
        _validate_worker_report(
            report,
            phase="evaluation",
            attempt_id="A_RESUME",
            shard=0,
            contract=contract,
            contract_sha="1" * 64,
        )
        report["labels_loaded_this_attempt"] = True
        with self.assertRaises(RuntimeError):
            _validate_worker_report(
                report,
                phase="evaluation",
                attempt_id="A_RESUME",
                shard=0,
                contract=contract,
                contract_sha="1" * 64,
            )


class ArmIntegrationTest(unittest.TestCase):
    def test_all_arms_are_built_and_rms_matches_the_frozen_references(self):
        shape = (11, 11, 11)
        generator = torch.Generator().manual_seed(441)
        moving = torch.randn((1, 1, *shape), generator=generator)
        fixed = torch.randn((1, 1, *shape), generator=generator)
        initial = torch.zeros((1, 3, *shape))
        mask = geometry_mask(shape, 4, torch.device("cpu"))
        fixed_mind = mind_ssc(masked_zscore(fixed, mask), radius=1, dilation=2)
        moving_mind = mind_ssc(masked_zscore(moving, mask), radius=1, dilation=2)

        requested, metadata = _build_requested_arms(
            fixed_image=fixed,
            moving_image=moving,
            initial=initial,
            mask=mask,
            fixed_mind=fixed_mind,
            moving_mind=moving_mind,
        )

        expected = {spec.arm_id for spec in ARM_SPECS}
        self.assertEqual(set(requested), expected)
        self.assertEqual(set(metadata), expected)
        self.assertEqual(int(torch.count_nonzero(requested["zero_update"])), 0)
        self.assertLessEqual(metadata["raw_conf_post1"]["independent_decoder_max_abs_difference"], 2e-6)
        for arm_id in (
            "raw_mean_normmatched_post1",
            "adaptive_mean_adaptref_normmatched_post1",
            "adaptive_mean_rawref_normmatched_post1",
        ):
            self.assertAlmostEqual(metadata[arm_id]["rms_matched"], metadata[arm_id]["rms_reference"], places=7)
        self.assertAlmostEqual(
            metadata["raw_mean_normmatched_post1"]["rms_reference"],
            metadata["adaptive_mean_rawref_normmatched_post1"]["rms_reference"],
            places=7,
        )
        for spec in ARM_SPECS:
            diagnostic = metadata[spec.arm_id]["posterior_diagnostics"]
            if spec.arm_id == "zero_update":
                self.assertIsNone(diagnostic)
            else:
                self.assertEqual(diagnostic["diagnostic_id"], POSTERIOR_DIAGNOSTICS_ID)
            roughness = metadata[spec.arm_id]["postprocessed_residual_roughness"]
            self.assertEqual(roughness["metric_id"], RMS_FIRST_DIFFERENCE_ROUGHNESS_ID)
            self.assertGreater(roughness["pair_count"], 0)


def metric_bundle(value: float) -> dict:
    return {
        MATHEMATICAL_SDLOGJ_CROP2: {
            "metric_id": MATHEMATICAL_SDLOGJ_CROP2,
            "status": "OK",
            "value": value,
            "components": {},
        },
        DIGITAL_DECOMPOSITION: {
            "metric_id": DIGITAL_DECOMPOSITION,
            "status": "OK",
            "value": 0.0,
            "components": {
                "corner_union_violation_fraction": 0.0,
                "jstar_union_violation_fraction": 0.0,
            },
        },
        DETJ_DIAGNOSTICS: {
            "metric_id": DETJ_DIAGNOSTICS,
            "status": "OK",
            "value": None,
            "components": {"nonfinite_count": 0.0, "nonpositive_count": 0.0},
        },
    }


class SummaryDecisionTest(unittest.TestCase):
    def test_capacity_claim_is_suppressed_when_any_candidate_is_not_exact(self):
        rows = {
            "first": [
                {
                    "requested_diagnostic_dice": 0.91,
                    "capacity_candidate_dice": 0.9,
                    "primary_returned_dice": 0.8,
                    "exact_certified": False,
                    "requested_exact_certified": False,
                }
            ],
            "second": [
                {
                    "requested_diagnostic_dice": 0.81,
                    "capacity_candidate_dice": 0.8,
                    "primary_returned_dice": 0.8,
                    "exact_certified": True,
                    "requested_exact_certified": True,
                }
            ],
        }

        result = _contrast(rows, "first", "second")

        self.assertFalse(result["capacity_exact_eligible"])
        self.assertFalse(result["capacity_wins"])

    def test_tied_viable_candidates_use_frozen_simplicity_order(self):
        decisions, evaluations = {}, {}
        for case_index in range(58):
            case_id = f"subject_{case_index:03d}"
            decision_arms, evaluation_arms = [], []
            for spec in ARM_SPECS:
                delta = 0.0005 if spec.arm_id == "c1_raw_conf_post1" else 0.0011
                if spec.arm_id == "zero_update":
                    delta = 0.0
                proposal = {"requested_rms": 0.1, "lambda_mean": 0.5}
                if spec.arm_id in (
                    "raw_mean_normmatched_post1",
                    "adaptive_mean_adaptref_normmatched_post1",
                    "adaptive_mean_rawref_normmatched_post1",
                ):
                    proposal.update({"rms_reference": 0.1, "rms_matched": 0.1})
                candidate_geometry = metric_bundle(0.0501)
                decision_arms.append(
                    {
                        "arm_index": spec.arm_index,
                        "arm_id": spec.arm_id,
                        "selectable": spec.selectable,
                        "stress_only": spec.stress_only,
                        "action": "ACCEPT" if delta else "ROLLBACK",
                        "reason": "synthetic",
                        "exact": {"status": "CERTIFIED"},
                        "exact_certified": True,
                        "candidate_field": {"array_sha256": f"{spec.arm_index:064x}"},
                        "returned_field": {"array_sha256": f"{spec.arm_index:064x}"},
                        "requested_state": {
                            "field": {"array_sha256": f"{spec.arm_index + 100:064x}"},
                            "exact": {"status": "CERTIFIED"},
                            "exact_certified": True,
                            "geometry": candidate_geometry,
                            "fast_cert_bound_work_eps": 0.0012,
                            "residual_rms_from_stored_state": 0.1,
                            "utility": {
                                "mind_baseline_common": 0.2,
                                "mind_candidate_common": 0.19,
                                "ncc7_baseline_common": -0.4,
                                "ncc7_candidate_common": -0.41,
                                "ncc9_baseline_common": -0.3,
                                "ncc9_candidate_common": -0.31,
                            },
                        },
                        "supports": {
                            key: {"baseline_count": 100, "pair_count": 100, "retention": 1.0}
                            for key in ("mind", "ncc7", "ncc9")
                        },
                        "utility": {
                            "mind_baseline_common": 0.2,
                            "mind_candidate_common": 0.19,
                            "ncc7_baseline_common": -0.4,
                            "ncc7_candidate_common": -0.41,
                            "ncc9_baseline_common": -0.3,
                            "ncc9_candidate_common": -0.31,
                        },
                        "proposal": proposal,
                        "operator": {
                            "output_fast_cert_bound": 0.0012,
                            "retained_norm_ratio": 1.0,
                            "effective_alpha_min": 1.0,
                            "effective_alpha_p50": 1.0,
                            "effective_alpha_p95": 1.0,
                            "effective_alpha_max": 1.0,
                        },
                        "candidate_geometry": candidate_geometry,
                        "postclip_residual_rms": 0.1,
                        "returned_residual_rms": 0.1 if delta else 0.0,
                    }
                )
                evaluation_arms.append(
                    {
                        "arm_index": spec.arm_index,
                        "arm_id": spec.arm_id,
                        "baseline_dice": 0.75,
                        "requested_diagnostic_dice": 0.75 + delta,
                        "requested_diagnostic_dice_delta": delta,
                        "capacity_candidate_dice": 0.75 + delta,
                        "capacity_dice_delta": delta,
                        "primary_returned_dice": 0.75 + delta,
                        "primary_dice_delta": delta,
                    }
                )
            decisions[case_id] = {
                "initial": {"geometry": metric_bundle(0.05)},
                "arms": decision_arms,
            }
            evaluations[case_id] = {"arms": evaluation_arms}

        flat, arms, hypotheses, summary, fatal = _build_summaries(decisions, evaluations)

        self.assertFalse(fatal)
        self.assertEqual(len(flat), 58 * 10)
        self.assertEqual(len(arms), 10)
        self.assertEqual(summary["winner_arm_id"], "raw_mean_normmatched_post1")
        self.assertEqual(summary["execution_integrity_status"], "PASS")
        self.assertFalse(hypotheses["test_115_authorized"])
        self.assertTrue(hypotheses["no_material_plus_0_002_capacity_closes_only_this_configuration"])
        zero = next(row for row in arms if row["arm_id"] == "zero_update")
        self.assertAlmostEqual(zero["candidate_geometry_delta_mean"], 0.0001)
        self.assertEqual(zero["primary_returned_geometry_delta_mean"], 0.0)

    def test_arm_construction_invariants_reject_rms_and_message_mass_tampering(self):
        arms = []
        for spec in ARM_SPECS:
            proposal = {"requested_rms": 0.1, "lambda_mean": 0.5}
            if spec.arm_id in (
                "raw_mean_normmatched_post1",
                "adaptive_mean_adaptref_normmatched_post1",
                "adaptive_mean_rawref_normmatched_post1",
            ):
                proposal.update(rms_reference=0.1, rms_matched=0.1)
            arms.append({"arm_id": spec.arm_id, "proposal": proposal})
        _validate_arm_construction_invariants(arms, label="positive")

        by_id = {row["arm_id"]: row for row in arms}
        by_id["raw_mean_normmatched_post1"]["proposal"]["rms_matched"] = 0.2
        with self.assertRaises(RuntimeError):
            _validate_arm_construction_invariants(arms, label="rms-tamper")
        by_id["raw_mean_normmatched_post1"]["proposal"]["rms_matched"] = 0.1
        by_id["adaptive_mp_conf_post1"]["proposal"]["lambda_mean"] = 0.6
        with self.assertRaises(RuntimeError):
            _validate_arm_construction_invariants(arms, label="mass-tamper")

    def test_float32_sized_rms_roundoff_is_not_misclassified(self):
        self.assertTrue(math.isclose(0.1 + 5e-8, 0.1, rel_tol=FLOAT32_PARITY_RTOL, abs_tol=FLOAT32_PARITY_ATOL))
        self.assertFalse(math.isclose(0.1 + 5e-6, 0.1, rel_tol=FLOAT32_PARITY_RTOL, abs_tol=FLOAT32_PARITY_ATOL))


class FrozenEvaluationBindingTest(unittest.TestCase):
    def test_action_and_baseline_are_bound_to_the_frozen_decision(self):
        from experiments.core.inference_metrics import metric_profile_for

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            case_id = "subject_000"
            case_root = root / "cases" / case_id
            decision_path = case_root / "decision_complete.json"
            decision_payload = {
                "arms": [
                    {"arm_index": spec.arm_index, "arm_id": spec.arm_id, "action": "ROLLBACK"} for spec in ARM_SPECS
                ]
            }
            atomic_write_json(decision_path, decision_payload)
            decision_case_sha = sha256_file(decision_path)
            baseline, candidate = 0.8, 0.81
            payload = {
                "schema": "ctcf-search-c3a-evaluation-case-v1",
                "status": "COMPLETE",
                "case_id": case_id,
                "decision_contract_sha256": "1" * 64,
                "decision_barrier_sha256": "2" * 64,
                "decision_case_sha256": decision_case_sha,
                "source_input_sha256": "3" * 64,
                "labels_loaded_after_barrier": True,
                "test_split_accessed": False,
                "labels": list(metric_profile_for("IXI").labels),
                "c2_dice_parity_verified": True,
                "arms": [
                    {
                        "arm_index": spec.arm_index,
                        "arm_id": spec.arm_id,
                        "baseline_dice": baseline,
                        "requested_diagnostic_dice": candidate,
                        "requested_diagnostic_dice_delta": candidate - baseline,
                        "capacity_candidate_dice": candidate,
                        "capacity_dice_delta": candidate - baseline,
                        "primary_returned_dice": baseline,
                        "primary_dice_delta": 0.0,
                        "primary_action": "ROLLBACK",
                    }
                    for spec in ARM_SPECS
                ],
                "execution": {
                    "phase": "evaluation",
                    "shard_index": 0,
                    "physical_gpu": 2,
                    "device": "cuda:0",
                    "deterministic": True,
                    "labels_loaded_after_barrier": True,
                },
            }
            contract = {
                "source_container_sha256": {case_id: "3" * 64},
                "num_shards": 1,
                "shards": {"0": [case_id]},
                "shard_to_physical_gpu": {"0": 2},
            }
            evaluation_goldens = {
                "cases": {
                    case_id: {
                        "baseline_dice": baseline,
                        "control_candidate_dice": {
                            arm_id: candidate
                            for arm_id in (
                                "c1_raw_conf_post1",
                                "raw_conf_post1",
                                "raw_conf_post2",
                            )
                        },
                    }
                }
            }
            barrier = {"decision_case_sha256": {case_id: decision_case_sha}}
            marker = case_root / "evaluation_complete.json"

            self.assertEqual(
                len(
                    _validate_evaluation_case(
                        payload,
                        marker,
                        case_id,
                        contract,
                        "1" * 64,
                        barrier,
                        "2" * 64,
                        evaluation_goldens,
                    )
                ),
                len(ARM_SPECS),
            )
            payload["arms"][1]["capacity_candidate_dice"] += 1e-5
            payload["arms"][1]["capacity_dice_delta"] += 1e-5
            with self.assertRaisesRegex(RuntimeError, "control Dice parity"):
                _validate_evaluation_case(
                    payload,
                    marker,
                    case_id,
                    contract,
                    "1" * 64,
                    barrier,
                    "2" * 64,
                    evaluation_goldens,
                )
            payload["arms"][1]["capacity_candidate_dice"] -= 1e-5
            payload["arms"][1]["capacity_dice_delta"] -= 1e-5
            payload["arms"][0]["primary_action"] = "ACCEPT"
            payload["arms"][0]["primary_returned_dice"] = candidate
            payload["arms"][0]["primary_dice_delta"] = candidate - baseline
            with self.assertRaisesRegex(RuntimeError, "Dice arithmetic"):
                _validate_evaluation_case(
                    payload,
                    marker,
                    case_id,
                    contract,
                    "1" * 64,
                    barrier,
                    "2" * 64,
                    evaluation_goldens,
                )


class CliSurfaceTest(unittest.TestCase):
    def test_exact_stage_surface(self):
        parser = build_parser()
        action = next(action for action in parser._actions if action.dest == "action")
        self.assertEqual(
            set(action.choices),
            {
                "selfcheck",
                "prepare",
                "extract-images",
                "decision-worker",
                "decision-barrier",
                "evaluation-worker",
                "finalize",
            },
        )

    def test_selfcheck_cli_writes_pass(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "selfcheck.json"
            self.assertEqual(main(["selfcheck", "--output", str(output)]), 0)
            self.assertEqual(json.loads(output.read_text(encoding="utf-8"))["status"], "PASS")


if __name__ == "__main__":
    unittest.main()
