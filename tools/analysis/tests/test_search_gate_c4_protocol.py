from __future__ import annotations

import dataclasses
import json
import math
import unittest

import numpy as np

from tools.analysis.search_gate_c4 import (
    ARM_SPECS,
    BOOTSTRAP_CONFIDENCE,
    BOOTSTRAP_FAMILY_SIZE,
    BOOTSTRAP_METHOD_ID,
    BOOTSTRAP_RESAMPLES,
    BOOTSTRAP_SEED,
    BRANCH_ADVANCE,
    BRANCH_CLOSE,
    BRANCH_GEOMETRY,
    BRANCH_UTILITY,
    C4_POLICY,
    C4_POLICY_SHA256,
    CAPACITY_CI_LOW_DICE_DELTA_MIN_STRICT,
    CAPACITY_MEAN_DICE_DELTA_MIN,
    COMMON_EVIDENCE_COLLAR,
    CONTRAST_SPECS,
    DIAGNOSTIC_ARM_IDS,
    DIAGNOSTIC_ARM_SPECS,
    EXPECTED_CASE_COUNT,
    GEOMETRY_NONINFERIOR_TOLERANCE,
    MAIN_ARM_SPECS,
    POLICY_MEAN_DICE_DELTA_MIN,
    PROTOCOL_ID,
    REFERENCE_MEAN_DICE_DELTA_MIN,
    RMS_TARGET_SOURCE_ID,
    SCIENTIFIC_REFERENCE_ARM_ID,
    SEARCH_REACH_SPECS_BY_ID,
    SELECTABLE_ARM_IDS,
    TEST_115_AUTHORIZED,
    WORK_UNIT_SPECS,
    ArmEvidence,
    GeometryComparison,
    PairedSummary,
    assess_arm,
    canonical_policy_bytes,
    geometry_noninferior,
    materially_better_than_reference,
    materially_strong_capacity,
    materially_strong_policy,
    paired_summary,
    policy_sha256,
    select_next_branch,
)


def summary(mean: float, *, median: float | None = None, ci_low: float | None = None) -> PairedSummary:
    median_value = mean if median is None else median
    low = mean if ci_low is None else ci_low
    return PairedSummary(
        n=EXPECTED_CASE_COUNT,
        mean=mean,
        median=median_value,
        ci_low=low,
        ci_high=max(mean, low) + 0.0001,
        improved=EXPECTED_CASE_COUNT if median_value > 0 else 0,
        worsened=0 if median_value > 0 else EXPECTED_CASE_COUNT,
        tied=0,
        bootstrap_resamples=BOOTSTRAP_RESAMPLES,
        bootstrap_seed=BOOTSTRAP_SEED,
        bootstrap_confidence=BOOTSTRAP_CONFIDENCE,
        bootstrap_method=BOOTSTRAP_METHOD_ID,
        simultaneous_family_size=BOOTSTRAP_FAMILY_SIZE,
    )


def evidence(
    arm_id: str,
    *,
    capacity_mean: float = 0.003,
    capacity_low: float = 0.002,
    incremental_mean: float = 0.001,
    incremental_low: float = 0.0002,
    geometry_candidate: float = 0.3,
    geometry_reference: float = 0.3,
    policy_mean: float = 0.002,
    policy_low: float = 0.0002,
    complete: bool = True,
    exact: bool = True,
) -> ArmEvidence:
    incremental = None
    if arm_id != SCIENTIFIC_REFERENCE_ARM_ID:
        incremental = summary(incremental_mean, ci_low=incremental_low)
    return ArmEvidence(
        arm_id=arm_id,
        capacity_vs_baseline=summary(capacity_mean, ci_low=capacity_low),
        incremental_vs_reference=incremental,
        policy_vs_baseline=summary(policy_mean, ci_low=policy_low),
        geometry=(
            GeometryComparison(
                "CTCF_MATHEMATICAL_SDLOGJ_CENTRAL_CROP2_UNMASKED_DDOF0_FAILCLOSED_V1",
                geometry_candidate,
                geometry_reference,
                True,
            ),
        ),
        all_work_units_complete=complete,
        all_exact_certified=exact,
    )


class FrozenProtocolTest(unittest.TestCase):
    def test_main_and_diagnostic_ids_are_exact(self):
        self.assertEqual(PROTOCOL_ID, "CTCF-SEARCH-GATE-C4-V1")
        self.assertEqual(
            SELECTABLE_ARM_IDS,
            (
                "mind_d1_s1",
                "mind_d2_s1",
                "mind_d4_s1",
                "mind_f124_s1",
                "mind_d1_s2",
                "mind_d2_s2",
                "mind_d4_s2",
                "mind_f124_s2",
            ),
        )
        self.assertEqual(
            DIAGNOSTIC_ARM_IDS,
            (
                "legacy_mind_d2_s1_collar4",
                "mind_f222_s1",
                "intensity_s1",
                "intensity_s2",
            ),
        )
        self.assertEqual([arm.arm_index for arm in ARM_SPECS], list(range(12)))

    def test_factorial_axes_do_not_conflate_descriptor_and_reach(self):
        for arm in MAIN_ARM_SPECS:
            self.assertTrue(arm.selectable)
            self.assertFalse(arm.diagnostic_only)
            self.assertEqual(arm.evidence_collar, COMMON_EVIDENCE_COLLAR)
            self.assertEqual(arm.standardization_mode, "centered_two_pass_fp32")
            self.assertEqual(arm.decoder_mode, "posterior_mean")
            self.assertEqual(arm.rms_target_source_id, RMS_TARGET_SOURCE_ID)
        self.assertEqual(SEARCH_REACH_SPECS_BY_ID["S1"].pre_rms_multiplier, 2.0)
        self.assertEqual(SEARCH_REACH_SPECS_BY_ID["S2"].pre_rms_multiplier, 1.0)
        self.assertEqual(SEARCH_REACH_SPECS_BY_ID["S1"].offset_stride_voxels, 1)
        self.assertEqual(SEARCH_REACH_SPECS_BY_ID["S2"].offset_stride_voxels, 2)

    def test_candidate_offsets_are_27_unique_zyx_values_in_frozen_order(self):
        for search_id, stride in (("S1", 1), ("S2", 2)):
            offsets = SEARCH_REACH_SPECS_BY_ID[search_id].offsets_zyx
            self.assertEqual(len(offsets), 27)
            self.assertEqual(len(set(offsets)), 27)
            self.assertEqual(offsets[0], (-stride, -stride, -stride))
            self.assertEqual(offsets[13], (0, 0, 0))
            self.assertEqual(offsets[-1], (stride, stride, stride))

    def test_diagnostic_contracts_are_nonselectable_and_specific(self):
        for arm in DIAGNOSTIC_ARM_SPECS:
            self.assertFalse(arm.selectable)
            self.assertTrue(arm.diagnostic_only)
        legacy, f222, intensity_s1, intensity_s2 = DIAGNOSTIC_ARM_SPECS
        self.assertEqual(
            (legacy.standardization_mode, legacy.decoder_mode, legacy.evidence_collar),
            ("legacy_sequential_fp32", "posterior_expectation_times_confidence", 4),
        )
        self.assertEqual(f222.descriptor_id, "F222")
        self.assertIsNone(f222.rms_target_source_id)
        self.assertEqual((intensity_s1.descriptor_id, intensity_s2.descriptor_id), ("INTENSITY", "INTENSITY"))
        self.assertEqual(
            (intensity_s1.rms_target_source_id, intensity_s2.rms_target_source_id),
            (RMS_TARGET_SOURCE_ID, RMS_TARGET_SOURCE_ID),
        )
        self.assertEqual(
            [arm.materialize_candidate for arm in DIAGNOSTIC_ARM_SPECS],
            [False, False, True, True],
        )
        self.assertEqual(
            [arm.post_barrier_evaluation for arm in DIAGNOSTIC_ARM_SPECS],
            [False, False, True, True],
        )

    def test_work_units_are_explicit_and_label_free(self):
        self.assertEqual(len(WORK_UNIT_SPECS), len(ARM_SPECS))
        self.assertEqual(tuple(unit.arm_id for unit in WORK_UNIT_SPECS), tuple(arm.arm_id for arm in ARM_SPECS))
        self.assertEqual(len({unit.unit_id for unit in WORK_UNIT_SPECS}), len(WORK_UNIT_SPECS))
        self.assertTrue(all(unit.stage == "LABEL_FREE_DECISION" for unit in WORK_UNIT_SPECS))
        self.assertTrue(all(unit.labels_accessible is False for unit in WORK_UNIT_SPECS))
        self.assertEqual(
            [unit.materialize_candidate for unit in WORK_UNIT_SPECS],
            [True] * 8 + [False, False, True, True],
        )
        self.assertEqual(
            [unit.post_barrier_evaluation for unit in WORK_UNIT_SPECS],
            [True] * 8 + [False, False, True, True],
        )
        self.assertFalse(TEST_115_AUTHORIZED)

    def test_contrasts_cover_capacity_reference_descriptor_and_reach(self):
        by_family: dict[str, int] = {}
        for contrast in CONTRAST_SPECS:
            by_family[contrast.family] = by_family.get(contrast.family, 0) + 1
            self.assertTrue(contrast.post_barrier_only)
        self.assertEqual(
            by_family,
            {
                "capacity_vs_baseline": 8,
                "returned_policy_vs_baseline": 8,
                "incremental_vs_common_reference": 7,
                "descriptor_at_fixed_reach": 6,
                "search_reach_at_fixed_descriptor": 4,
            },
        )

    def test_policy_is_immutable_json_canonical_and_hash_frozen(self):
        self.assertEqual(policy_sha256(), C4_POLICY_SHA256)
        self.assertRegex(C4_POLICY_SHA256, r"^[0-9a-f]{64}$")
        self.assertEqual(canonical_policy_bytes(), canonical_policy_bytes(C4_POLICY))
        json.loads(canonical_policy_bytes())
        with self.assertRaises(dataclasses.FrozenInstanceError):
            C4_POLICY.protocol_id = "mutated"
        with self.assertRaises(dataclasses.FrozenInstanceError):
            MAIN_ARM_SPECS[0].arm_id = "mutated"


class PairedThresholdTest(unittest.TestCase):
    def test_paired_bootstrap_is_deterministic_and_uses_frozen_contract(self):
        baseline = np.linspace(0.7, 0.8, EXPECTED_CASE_COUNT)
        candidate = baseline + np.linspace(0.001, 0.004, EXPECTED_CASE_COUNT)
        first = paired_summary(candidate, baseline)
        second = paired_summary(candidate - baseline)
        self.assertEqual(first, second)
        self.assertEqual(first.bootstrap_resamples, 10_000)
        self.assertEqual(first.bootstrap_seed, 0)
        self.assertEqual(first.bootstrap_confidence, 0.95)

    def test_capacity_thresholds_use_required_strictness(self):
        exact = summary(
            CAPACITY_MEAN_DICE_DELTA_MIN,
            median=0.0001,
            ci_low=np.nextafter(CAPACITY_CI_LOW_DICE_DELTA_MIN_STRICT, math.inf),
        )
        self.assertTrue(materially_strong_capacity(exact))
        self.assertFalse(
            materially_strong_capacity(
                summary(CAPACITY_MEAN_DICE_DELTA_MIN, median=0.0001, ci_low=CAPACITY_CI_LOW_DICE_DELTA_MIN_STRICT)
            )
        )
        self.assertFalse(materially_strong_capacity(summary(CAPACITY_MEAN_DICE_DELTA_MIN, median=0.0, ci_low=0.0011)))

    def test_reference_threshold_has_mean_equality_and_strict_positive_lower_bound(self):
        self.assertTrue(
            materially_better_than_reference(summary(REFERENCE_MEAN_DICE_DELTA_MIN, ci_low=np.nextafter(0.0, math.inf)))
        )
        self.assertFalse(materially_better_than_reference(summary(REFERENCE_MEAN_DICE_DELTA_MIN, ci_low=0.0)))
        self.assertFalse(materially_better_than_reference(summary(0.000499, ci_low=0.0001)))

    def test_returned_policy_requires_practical_mean_and_strict_positive_lower_bound(self):
        self.assertTrue(materially_strong_policy(summary(POLICY_MEAN_DICE_DELTA_MIN, ci_low=0.0001)))
        self.assertFalse(materially_strong_policy(summary(POLICY_MEAN_DICE_DELTA_MIN, ci_low=0.0)))
        self.assertFalse(materially_strong_policy(summary(0.0009, ci_low=0.0001)))

    def test_malformed_summaries_fail_closed(self):
        wrong_n = dataclasses.replace(summary(0.003), n=57, improved=57)
        wrong_bootstrap = dataclasses.replace(summary(0.003), bootstrap_seed=1)
        nonfinite = dataclasses.replace(summary(0.003), mean=float("nan"))
        for value in (wrong_n, wrong_bootstrap, nonfinite):
            with self.assertRaises(ValueError):
                materially_strong_capacity(value)
        with self.assertRaises(ValueError):
            paired_summary([])
        with self.assertRaises(ValueError):
            paired_summary([0.1, float("inf")])


class GeometryTest(unittest.TestCase):
    def test_noninferiority_uses_lower_is_better_and_exact_tolerance(self):
        metric_id = "MATHEMATICAL_SDLOGJ"
        self.assertTrue(
            geometry_noninferior((GeometryComparison(metric_id, 0.3 + GEOMETRY_NONINFERIOR_TOLERANCE, 0.3, True),))
        )
        self.assertFalse(
            geometry_noninferior(
                (GeometryComparison(metric_id, 0.3 + 1.01 * GEOMETRY_NONINFERIOR_TOLERANCE, 0.3, True),)
            )
        )

    def test_undefined_geometry_is_not_silently_noninferior(self):
        self.assertFalse(geometry_noninferior((GeometryComparison("SDLOGJ", None, 0.3, True),)))
        self.assertFalse(geometry_noninferior((GeometryComparison("SDLOGJ", float("nan"), 0.3, True),)))
        with self.assertRaises(ValueError):
            geometry_noninferior(())
        with self.assertRaises(ValueError):
            geometry_noninferior((GeometryComparison("DICE", 0.8, 0.8, False),))


class BranchSelectionTest(unittest.TestCase):
    def all_rows(self, **changes: ArmEvidence) -> tuple[ArmEvidence, ...]:
        return tuple(changes.get(arm_id, evidence(arm_id)) for arm_id in SELECTABLE_ARM_IDS)

    def test_reference_needs_absolute_materiality_but_not_self_contrast(self):
        result = assess_arm(evidence(SCIENTIFIC_REFERENCE_ARM_ID))
        self.assertTrue(result.incremental_over_reference)
        self.assertTrue(result.eligible)
        invalid = dataclasses.replace(
            evidence(SCIENTIFIC_REFERENCE_ARM_ID),
            incremental_vs_reference=summary(0.0, ci_low=0.0),
        )
        with self.assertRaises(ValueError):
            assess_arm(invalid)

    def test_nonreference_requires_reference_contrast(self):
        missing = dataclasses.replace(evidence("mind_d1_s1"), incremental_vs_reference=None)
        with self.assertRaises(ValueError):
            assess_arm(missing)
        weak = evidence("mind_d1_s1", incremental_mean=0.0004, incremental_low=0.0001)
        self.assertFalse(assess_arm(weak).eligible)

    def test_exactness_completion_and_geometry_are_each_required(self):
        self.assertFalse(assess_arm(evidence("mind_d1_s1", exact=False)).eligible)
        self.assertFalse(assess_arm(evidence("mind_d1_s1", complete=False)).eligible)
        self.assertFalse(assess_arm(evidence("mind_d1_s1", policy_mean=0.0009, policy_low=0.0001)).eligible)
        self.assertFalse(
            assess_arm(
                evidence(
                    "mind_d1_s1",
                    geometry_candidate=0.3 + 2 * GEOMETRY_NONINFERIOR_TOLERANCE,
                )
            ).eligible
        )

    def test_diagnostics_can_never_be_assessed_or_promoted(self):
        diagnostic = dataclasses.replace(evidence("mind_d1_s1"), arm_id="intensity_s1")
        with self.assertRaises(ValueError):
            assess_arm(diagnostic)
        rows = list(self.all_rows())
        rows[-1] = diagnostic
        with self.assertRaises(ValueError):
            select_next_branch(rows)

    def test_selector_requires_all_main_arms_in_frozen_order(self):
        rows = self.all_rows()
        with self.assertRaises(ValueError):
            select_next_branch(rows[:-1])
        with self.assertRaises(ValueError):
            select_next_branch(tuple(reversed(rows)))

    def test_selector_advances_and_resolves_close_scores_by_frozen_order(self):
        rows = [evidence(arm_id, capacity_mean=0.003, capacity_low=0.002) for arm_id in SELECTABLE_ARM_IDS]
        rows[0] = evidence("mind_d1_s1", capacity_mean=0.0040000, capacity_low=0.002)
        rows[1] = evidence("mind_d2_s1", capacity_mean=0.0040005, capacity_low=0.002)
        decision = select_next_branch(rows)
        self.assertEqual(decision.branch_id, BRANCH_ADVANCE)
        self.assertEqual(decision.selected_arm_id, "mind_d1_s1")
        self.assertEqual(decision.eligible_arm_ids, SELECTABLE_ARM_IDS)

    def test_selector_closes_when_no_arm_is_material(self):
        rows = tuple(evidence(arm_id, capacity_mean=0.001, capacity_low=0.0005) for arm_id in SELECTABLE_ARM_IDS)
        decision = select_next_branch(rows)
        self.assertEqual(decision.branch_id, BRANCH_CLOSE)
        self.assertIsNone(decision.selected_arm_id)
        self.assertEqual(decision.eligible_arm_ids, ())

    def test_selector_opens_geometry_before_utility_when_capacity_is_material(self):
        rows = self.all_rows(
            mind_d1_s1=evidence(
                "mind_d1_s1",
                geometry_candidate=0.31,
                geometry_reference=0.3,
            )
        )
        rows = tuple(
            row
            if row.arm_id == "mind_d1_s1"
            else dataclasses.replace(row, capacity_vs_baseline=summary(0.001, ci_low=0.0005))
            for row in rows
        )
        self.assertEqual(select_next_branch(rows).branch_id, BRANCH_GEOMETRY)

    def test_selector_opens_utility_after_geometry_passes(self):
        rows = self.all_rows(mind_d1_s1=evidence("mind_d1_s1", policy_mean=0.0009, policy_low=0.0001))
        rows = tuple(
            row
            if row.arm_id == "mind_d1_s1"
            else dataclasses.replace(row, capacity_vs_baseline=summary(0.001, ci_low=0.0005))
            for row in rows
        )
        self.assertEqual(select_next_branch(rows).branch_id, BRANCH_UTILITY)


if __name__ == "__main__":
    unittest.main()
