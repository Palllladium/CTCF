from __future__ import annotations

import dataclasses
import json
import math
import unittest
from types import SimpleNamespace

import numpy as np
import torch

from tools.analysis.search_gate_c3 import (
    ARM_SPECS,
    BOOTSTRAP_RESAMPLES,
    BOOTSTRAP_SEED,
    C3A_POLICY,
    C3A_POLICY_SHA256,
    CAPACITY_CI_LOW_DICE_DELTA_MIN,
    CAPACITY_MEAN_DICE_DELTA_MIN,
    EXACT_CLAIM_EPS,
    GEOMETRY_NONINFERIOR_TOLERANCE,
    LOCAL_CLIP_SWEEPS,
    MESSAGE_PASSING_AXIS_KERNEL,
    MESSAGE_PASSING_ID,
    MESSAGE_PASSING_PASSES,
    NCC7_WINDOW,
    NCC9_WINDOW,
    POLICY_MEAN_DICE_DELTA_MIN,
    PRIMARY_NCC_IMPROVEMENT_MIN,
    SELECTABLE_ARM_IDS,
    SUPPORT_RETENTION_MIN,
    WINNER_TIE_ORDER,
    WINNER_TIE_TOLERANCE,
    WORK_EPS,
    MetricEnvelope,
    MetricStatus,
    binary_erode_mask,
    build_c3a_supports,
    build_common_support,
    geometry_noninferior,
    materially_strong_capacity,
    metric_envelope,
    paired_summary,
    policy_sha256,
    primary_ncc_decision,
    select_winner,
    should_rollback,
    viable_primary_policy,
    winner_decision,
    wins,
)

MATHEMATICAL_SDLOGJ_CROP2 = "CTCF_MATHEMATICAL_SDLOGJ_CENTRAL_CROP2_UNMASKED_DDOF0_FAILCLOSED_V1"


class MetricFailClosedError(ValueError):
    pass


class FrozenContractTest(unittest.TestCase):
    def test_exact_constants_and_ten_arm_matrix(self):
        self.assertEqual([arm.arm_index for arm in ARM_SPECS], list(range(10)))
        self.assertEqual(
            [arm.arm_id for arm in ARM_SPECS],
            [
                "zero_update",
                "c1_raw_conf_post1",
                "raw_conf_post1",
                "raw_conf_post2",
                "iso_mp_conf_post1",
                "adaptive_mp_conf_post1",
                "raw_mean_normmatched_post1",
                "adaptive_mean_adaptref_normmatched_post1",
                "adaptive_mean_rawref_normmatched_post1",
                "adaptive_mean_raw_post1",
            ],
        )
        self.assertEqual(WORK_EPS, 0.0011)
        self.assertEqual(EXACT_CLAIM_EPS, 0.001)
        self.assertEqual(LOCAL_CLIP_SWEEPS, 1)
        self.assertEqual(MESSAGE_PASSING_ID, "CTCF_LINEAR_ENTROPY_LOGIT_MP_V1")
        self.assertEqual(MESSAGE_PASSING_PASSES, 1)
        self.assertEqual(MESSAGE_PASSING_AXIS_KERNEL, (0.25, 0.5, 0.25))
        self.assertEqual((NCC7_WINDOW, NCC9_WINDOW), (7, 9))
        self.assertEqual(SUPPORT_RETENTION_MIN, 0.99)
        self.assertEqual(PRIMARY_NCC_IMPROVEMENT_MIN, 1e-6)
        self.assertEqual((BOOTSTRAP_RESAMPLES, BOOTSTRAP_SEED), (10_000, 0))
        self.assertEqual(CAPACITY_MEAN_DICE_DELTA_MIN, 0.002)
        self.assertEqual(CAPACITY_CI_LOW_DICE_DELTA_MIN, 0.001)
        self.assertEqual(POLICY_MEAN_DICE_DELTA_MIN, 0.001)
        self.assertEqual(GEOMETRY_NONINFERIOR_TOLERANCE, 1e-6)
        self.assertEqual(WINNER_TIE_TOLERANCE, 1e-6)

    def test_only_scientific_candidates_are_selectable(self):
        self.assertEqual(
            SELECTABLE_ARM_IDS,
            (
                "iso_mp_conf_post1",
                "adaptive_mp_conf_post1",
                "raw_mean_normmatched_post1",
                "adaptive_mean_adaptref_normmatched_post1",
                "adaptive_mean_rawref_normmatched_post1",
            ),
        )
        self.assertTrue(ARM_SPECS[-1].stress_only)
        self.assertFalse(ARM_SPECS[-1].selectable)
        self.assertEqual(ARM_SPECS[6].rms_reference_arm_id, "raw_conf_post1")
        self.assertEqual(ARM_SPECS[7].rms_reference_arm_id, "adaptive_mp_conf_post1")
        self.assertEqual(ARM_SPECS[8].rms_reference_arm_id, "raw_conf_post1")

    def test_policy_is_immutable_and_hash_is_literal(self):
        self.assertEqual(policy_sha256(C3A_POLICY), C3A_POLICY_SHA256)
        self.assertRegex(C3A_POLICY_SHA256, r"^[0-9a-f]{64}$")
        with self.assertRaises(dataclasses.FrozenInstanceError):
            C3A_POLICY.protocol_id = "mutated"
        with self.assertRaises(dataclasses.FrozenInstanceError):
            ARM_SPECS[0].arm_id = "mutated"
        json.dumps(C3A_POLICY.to_dict(), sort_keys=True)


class SupportMaskTest(unittest.TestCase):
    def test_binary_erosion_treats_outside_as_false(self):
        mask = torch.ones(1, 1, 7, 7, 7, dtype=torch.bool)
        eroded = binary_erode_mask(mask, 1)
        self.assertEqual(int(eroded.sum()), 5**3)
        self.assertTrue(bool(eroded[0, 0, 1:-1, 1:-1, 1:-1].all()))
        self.assertFalse(bool(eroded[0, 0, 0].any()))
        self.assertTrue(torch.equal(binary_erode_mask(mask, 0), mask))
        self.assertIsNot(binary_erode_mask(mask, 0), mask)

    def test_common_support_intersects_before_window_erosion(self):
        geometry = torch.ones(1, 1, 15, 15, 15, dtype=torch.bool)
        baseline_valid = torch.ones_like(geometry)
        candidate_valid = torch.ones_like(geometry)
        candidate_valid[0, 0, 7, 7, 7] = False
        support = build_common_support(
            geometry,
            baseline_valid,
            candidate_valid,
            window=7,
            utility_id="COMMON_NCC7",
        )
        self.assertEqual(support.baseline_count, 9**3)
        self.assertEqual(support.pair_count, 9**3 - 7**3)
        self.assertAlmostEqual(support.retention, (9**3 - 7**3) / 9**3)
        self.assertFalse(bool(support.pair_mask[0, 0, 7, 7, 7]))
        self.assertFalse(bool((support.pair_mask & ~support.baseline_mask).any()))

    def test_c3a_support_bundle_has_common_mind_ncc7_and_ncc9(self):
        geometry = torch.ones(1, 1, 19, 19, 19, dtype=torch.bool)
        baseline_valid = torch.ones_like(geometry)
        candidate_valid = baseline_valid.clone()
        candidate_valid[0, 0, 9, 9, 9] = False
        supports = build_c3a_supports(geometry, baseline_valid, candidate_valid)
        self.assertEqual(set(supports), {"mind", "ncc7", "ncc9"})
        self.assertEqual(supports["mind"].baseline_count, 19**3)
        self.assertEqual(supports["mind"].pair_count, 19**3 - 1)
        self.assertEqual(supports["ncc7"].erosion_radius, 3)
        self.assertEqual(supports["ncc9"].erosion_radius, 4)
        self.assertEqual(supports["ncc7"].baseline_count, 13**3)
        self.assertEqual(supports["ncc9"].baseline_count, 11**3)

    def test_support_negative_inputs_fail_loudly(self):
        mask = torch.ones(1, 1, 9, 9, 9, dtype=torch.bool)
        with self.assertRaises(ValueError):
            binary_erode_mask(mask.float(), 1)
        with self.assertRaises(ValueError):
            binary_erode_mask(mask, -1)
        with self.assertRaises(ValueError):
            build_common_support(mask, mask, mask, window=6, utility_id="NCC")
        with self.assertRaises(ValueError):
            build_common_support(mask, mask[..., :-1], mask, window=7, utility_id="NCC")
        empty = torch.zeros_like(mask)
        with self.assertRaises(ValueError):
            build_common_support(empty, mask, mask, window=7, utility_id="NCC")


class PrimaryPolicyTest(unittest.TestCase):
    def decision(self, **changes):
        values = {
            "exact_certified": True,
            "support_retention": 0.99,
            "baseline_ncc_loss": -0.5,
            "candidate_ncc_loss": -0.500001,
        }
        values.update(changes)
        return primary_ncc_decision(**values)

    def test_threshold_equalities_are_accepted(self):
        decision = self.decision()
        self.assertTrue(decision.accept)
        self.assertFalse(decision.rollback)
        self.assertAlmostEqual(decision.ncc_improvement, 1e-6, delta=1e-15)

    def test_every_failed_conjunct_rolls_back(self):
        self.assertTrue(self.decision(exact_certified=False).rollback)
        self.assertTrue(self.decision(support_retention=np.nextafter(0.99, 0.0)).rollback)
        self.assertTrue(self.decision(candidate_ncc_loss=-0.500000999).rollback)
        self.assertTrue(
            should_rollback(
                exact_certified=False,
                support_retention=1.0,
                baseline_ncc_loss=-0.5,
                candidate_ncc_loss=-0.6,
            )
        )

    def test_empty_common_support_and_failed_exactness_rollback_without_fake_utility(self):
        exact_failure = self.decision(
            exact_certified=False,
            baseline_ncc_loss=None,
            candidate_ncc_loss=None,
        )
        empty_support = self.decision(
            support_retention=0.0,
            baseline_ncc_loss=None,
            candidate_ncc_loss=None,
        )
        self.assertEqual(exact_failure.reason, "ROLLBACK_EXACT_CERTIFICATE_FAILED")
        self.assertEqual(empty_support.reason, "ROLLBACK_COMMON_SUPPORT_RETENTION_BELOW_0.99")
        self.assertIsNone(exact_failure.ncc_improvement)
        self.assertIsNone(empty_support.ncc_improvement)

    def test_eligible_support_requires_two_finite_losses(self):
        with self.assertRaises(ValueError):
            self.decision(baseline_ncc_loss=None, candidate_ncc_loss=None)
        with self.assertRaises(ValueError):
            self.decision(baseline_ncc_loss=None)

    def test_nonfinite_or_out_of_range_inputs_do_not_become_decisions(self):
        with self.assertRaises(ValueError):
            self.decision(support_retention=1.01)
        with self.assertRaises(ValueError):
            self.decision(candidate_ncc_loss=float("nan"))
        with self.assertRaises(TypeError):
            self.decision(exact_certified=1)


class MetricEnvelopeTest(unittest.TestCase):
    def test_ok_metric_is_json_serializable_and_convention_bearing(self):
        result = SimpleNamespace(value=0.123, components={"voxels": 125.0})
        envelope = metric_envelope(MATHEMATICAL_SDLOGJ_CROP2, lambda: result)
        self.assertEqual(envelope.status, MetricStatus.OK)
        self.assertEqual(envelope.value, 0.123)
        self.assertEqual(dict(envelope.components), {"voxels": 125.0})
        self.assertEqual(json.loads(json.dumps(envelope.to_dict()))["status"], "OK")

    def test_failclosed_and_generic_errors_remain_distinguishable(self):
        def undefined():
            raise MetricFailClosedError("detJ contains one non-positive value")

        def broken():
            raise RuntimeError("kernel failed")

        first = metric_envelope(MATHEMATICAL_SDLOGJ_CROP2, undefined)
        second = metric_envelope("CTCF_DETJ_DIAGNOSTICS_V1", broken)
        self.assertEqual(first.status, MetricStatus.UNDEFINED_NONPOSITIVE)
        self.assertEqual(first.error_type, "MetricFailClosedError")
        self.assertEqual(second.status, MetricStatus.ERROR)
        self.assertEqual(second.error_type, "RuntimeError")
        json.dumps([first.to_dict(), second.to_dict()])

    def test_nonfinite_result_is_an_error_not_an_ok_value(self):
        envelope = metric_envelope("EXPLICIT_METRIC_V1", lambda: float("nan"))
        self.assertEqual(envelope.status, MetricStatus.ERROR)
        self.assertIsNone(envelope.value)

    def test_ambiguous_metric_or_component_name_is_forbidden(self):
        with self.assertRaises(ValueError):
            MetricEnvelope.ok("sdlogj", 0.1)
        with self.assertRaises(ValueError):
            MetricEnvelope.ok("EXPLICIT_METRIC_V1", 0.1, {"sdlogj": 0.1})
        with self.assertRaises(ValueError):
            MetricEnvelope("sd_logj", MetricStatus.OK, 0.1)
        with self.assertRaises(ValueError):
            MetricEnvelope("EXPLICIT_METRIC_V1", MetricStatus.ERROR, 0.1, error_type="X", detail="bad")


class PairedDecisionTest(unittest.TestCase):
    def test_paired_summary_and_win_are_deterministic(self):
        differences = np.array([0.001, 0.002, 0.003, 0.004], dtype=np.float64)
        first = paired_summary(differences)
        second = paired_summary(differences)
        self.assertEqual(first, second)
        self.assertEqual(first.n, 4)
        self.assertAlmostEqual(first.mean, 0.0025)
        self.assertAlmostEqual(first.median, 0.0025)
        self.assertEqual((first.improved, first.worsened, first.tied), (4, 0, 0))
        self.assertTrue(wins(first))

    def test_candidate_and_baseline_form_matches_explicit_differences(self):
        baseline = np.array([0.7, 0.8, 0.9])
        candidate = baseline + np.array([0.001, 0.002, 0.003])
        direct = paired_summary(candidate - baseline, bootstrap_resamples=100, bootstrap_seed=7)
        paired = paired_summary(candidate, baseline, bootstrap_resamples=100, bootstrap_seed=7)
        self.assertEqual(direct, paired)

    def test_capacity_and_policy_thresholds_are_not_conflated(self):
        strong = paired_summary(np.full(12, 0.0021))
        policy_only = paired_summary(np.full(12, 0.0011))
        weak = paired_summary(np.full(12, 0.0009))
        self.assertTrue(materially_strong_capacity(strong))
        self.assertFalse(materially_strong_capacity(policy_only))
        self.assertTrue(
            viable_primary_policy(
                policy_only,
                all_returned_exact_certified=True,
                all_support_diagnostics_defined=True,
                geometry_is_noninferior=True,
            )
        )
        self.assertFalse(
            viable_primary_policy(
                weak,
                all_returned_exact_certified=True,
                all_support_diagnostics_defined=True,
                geometry_is_noninferior=True,
            )
        )
        self.assertFalse(
            viable_primary_policy(
                strong,
                all_returned_exact_certified=True,
                all_support_diagnostics_defined=False,
                geometry_is_noninferior=True,
            )
        )

    def test_geometry_noninferiority_has_explicit_direction_and_tolerance(self):
        self.assertTrue(geometry_noninferior(0.02 + 1e-6, 0.02, all_candidate_metrics_defined=True))
        self.assertFalse(geometry_noninferior(0.02 + 1.01e-6, 0.02, all_candidate_metrics_defined=True))
        self.assertFalse(geometry_noninferior(0.0, 0.0, all_candidate_metrics_defined=False))

    def test_winner_uses_frozen_tie_order_not_mapping_order(self):
        scores = {
            "adaptive_mean_adaptref_normmatched_post1": 0.801,
            "adaptive_mean_rawref_normmatched_post1": 0.8009997,
            "adaptive_mp_conf_post1": 0.8009995,
            "iso_mp_conf_post1": 0.8009994,
            "raw_mean_normmatched_post1": 0.8009991,
        }
        viable = {arm_id: True for arm_id in SELECTABLE_ARM_IDS}
        self.assertEqual(WINNER_TIE_ORDER[0], "raw_mean_normmatched_post1")
        self.assertEqual(select_winner(scores, viable), "raw_mean_normmatched_post1")
        decision = winner_decision(scores, viable)
        self.assertEqual(decision.tied_at_tolerance, WINNER_TIE_ORDER)
        scores["adaptive_mean_adaptref_normmatched_post1"] += 2e-6
        self.assertEqual(select_winner(scores, viable), "adaptive_mean_adaptref_normmatched_post1")

    def test_no_viable_arm_is_a_valid_no_winner_branch(self):
        decision = winner_decision({}, {})
        self.assertIsNone(decision.winner_arm_id)
        self.assertEqual(decision.eligible_arm_ids, ())

    def test_negative_statistical_and_winner_inputs_fail(self):
        with self.assertRaises(ValueError):
            paired_summary([])
        with self.assertRaises(ValueError):
            paired_summary([0.1, float("inf")])
        with self.assertRaises(ValueError):
            paired_summary([0.1], [0.1, 0.2])
        with self.assertRaises(ValueError):
            paired_summary([0.1], bootstrap_resamples=0)
        with self.assertRaises(TypeError):
            wins("not a summary")
        with self.assertRaises(ValueError):
            select_winner({"invented_arm": 0.9}, {"invented_arm": True})
        with self.assertRaises(ValueError):
            select_winner({"adaptive_mean_raw_post1": 0.9}, {"adaptive_mean_raw_post1": True})
        with self.assertRaises(ValueError):
            select_winner(
                {"iso_mp_conf_post1": math.nan},
                {"iso_mp_conf_post1": True},
            )
        with self.assertRaises(ValueError):
            select_winner(
                {"iso_mp_conf_post1": -0.1},
                {"iso_mp_conf_post1": True},
            )


if __name__ == "__main__":
    unittest.main()
