from __future__ import annotations

import dataclasses
import unittest

import numpy as np

from tools.analysis.search_gate_c5b import (
    AMPLITUDE_RETENTION_CASE_COUNT_MIN,
    ANCHOR_ARM_IDS,
    ARM_SPECS,
    BOOTSTRAP_CONFIDENCE,
    BOOTSTRAP_FAMILY_SIZES,
    BOOTSTRAP_METHOD_ID,
    BOOTSTRAP_RESAMPLES,
    BOOTSTRAP_SEED,
    BRANCH_CLOSE_CLIP,
    BRANCH_CLOSE_GEOMETRY,
    BRANCH_CLOSE_NO_SUPERIORITY,
    BRANCH_CLOSE_REGIONAL,
    BRANCH_FREEZE_RISK_REPAIR,
    BRANCH_FREEZE_SUPERIOR,
    BRANCH_INVALID,
    C5B_POLICY,
    C5B_POLICY_SHA256,
    CANDIDATE_COUNT,
    CANDIDATE_OFFSETS_ZYX,
    COMMON_EVIDENCE_COLLAR,
    DECODER_MODE,
    DESCRIPTOR_ID,
    DIAGNOSTIC_ARM_ID,
    DICE_VS_REFERENCE_FAMILY_ID,
    EVALUATION_LABEL_IDS,
    EXPECTED_CASE_COUNT,
    IMAGE_NORMALIZATION_MODE,
    IMAGE_NORMALIZATION_STD_FLOOR,
    POSTCLIP_INTERPOLATION_ALLOWED,
    POSTERIOR_TEMPERATURE,
    PRE_RMS_MULTIPLIER,
    REFERENCE_ARM_ID,
    REGIONAL_REPAIR_LABEL_IDS,
    SDLOGJ_VS_REFERENCE_FAMILY_ID,
    SELECTABLE_ARM_IDS,
    STANDARDIZATION_FLOOR,
    STANDARDIZATION_MODE,
    TEST_115_AUTHORIZED,
    WINNER_MEAN_TIE_TOLERANCE,
    WINNER_TIE_BREAK_ORDER,
    ArmEvidence,
    PairedSummary,
    RegionalEvidence,
    assess_arm,
    canonical_policy_bytes,
    policy_sha256,
    regional_repair_family_id,
    regional_zero_family_id,
    select_next_branch,
    simultaneous_paired_summaries,
)


def summary(
    family_id: str,
    contrast_id: str,
    mean: float,
    *,
    median: float | None = None,
    ci_low: float | None = None,
    ci_high: float | None = None,
) -> PairedSummary:
    median_value = mean if median is None else median
    low = mean - 0.0001 if ci_low is None else ci_low
    high = mean + 0.0001 if ci_high is None else ci_high
    return PairedSummary(
        family_id=family_id,
        contrast_id=contrast_id,
        n=EXPECTED_CASE_COUNT,
        mean=mean,
        median=median_value,
        ci_low=low,
        ci_high=high,
        improved=EXPECTED_CASE_COUNT if median_value > 0.0 else 0,
        worsened=EXPECTED_CASE_COUNT if median_value < 0.0 else 0,
        tied=EXPECTED_CASE_COUNT if median_value == 0.0 else 0,
        bootstrap_resamples=BOOTSTRAP_RESAMPLES,
        bootstrap_seed=BOOTSTRAP_SEED,
        bootstrap_confidence=BOOTSTRAP_CONFIDENCE,
        bootstrap_method=BOOTSTRAP_METHOD_ID,
        simultaneous_family_size=BOOTSTRAP_FAMILY_SIZES[family_id],
    )


def regional(
    arm_id: str,
    *,
    zero_low: float = -0.001,
    risk_low: float | None = None,
    label_13_low: float | None = None,
    repair_mean: float = 0.003,
    repair_low: float = 0.001,
) -> RegionalEvidence:
    zero_family = regional_zero_family_id(arm_id)
    zero_rows = []
    for label_id in EVALUATION_LABEL_IDS:
        low = zero_low
        if risk_low is not None and label_id in (9, 29):
            low = risk_low
        if label_13_low is not None and label_id == 13:
            low = label_13_low
        zero_rows.append(
            summary(
                zero_family,
                f"label::{label_id}::{arm_id}::vs_zero",
                low + 0.001,
                ci_low=low,
                ci_high=low + 0.002,
            )
        )
    repair_family = regional_repair_family_id(arm_id)
    repair_rows = tuple(
        summary(
            repair_family,
            f"label::{label_id}::{arm_id}::vs_{REFERENCE_ARM_ID}",
            repair_mean,
            ci_low=repair_low,
        )
        for label_id in REGIONAL_REPAIR_LABEL_IDS
    )
    return RegionalEvidence(tuple(zero_rows), repair_rows)


def evidence(
    arm_id: str,
    *,
    dice_mean: float = 0.001,
    dice_median: float = 0.001,
    dice_low: float = 0.0002,
    sd_mean: float = 0.0,
    sd_high: float = 0.004,
    retention_median: float = 0.97,
    retention_cases: int = 55,
    complete: bool = True,
    exact: bool = True,
    folds: int = 0,
    regional_evidence: RegionalEvidence | None = None,
) -> ArmEvidence:
    return ArmEvidence(
        arm_id=arm_id,
        dice_vs_reference=summary(
            DICE_VS_REFERENCE_FAMILY_ID,
            f"dice::{arm_id}::vs_{REFERENCE_ARM_ID}",
            dice_mean,
            median=dice_median,
            ci_low=dice_low,
        ),
        sdlogj_vs_reference=summary(
            SDLOGJ_VS_REFERENCE_FAMILY_ID,
            f"sdlogj::{arm_id}::vs_{REFERENCE_ARM_ID}",
            sd_mean,
            ci_low=sd_mean - 0.001,
            ci_high=sd_high,
        ),
        regional=regional_evidence or regional(arm_id),
        all_work_units_complete=complete,
        all_exact_certified=exact,
        observed_fold_count=folds,
        amplitude_retention_median=retention_median,
        amplitude_retention_cases_at_least_090=retention_cases,
    )


class FrozenProtocolTest(unittest.TestCase):
    def test_exact_arms_and_roles_are_frozen(self):
        self.assertEqual(
            ANCHOR_ARM_IDS,
            ("c4_int_s2_a10_b0", "c5_int_s4_a10_b0_w1", "c5_int_s4_a20_b0_w1"),
        )
        self.assertEqual(
            SELECTABLE_ARM_IDS,
            ("int_s4_a125_b0_w1", "int_s4_a150_b0_w1", "int_s4_a175_b0_w1"),
        )
        self.assertEqual(DIAGNOSTIC_ARM_ID, "int_s4_a200_b0_w2")
        self.assertEqual([row.arm_index for row in ARM_SPECS], list(range(7)))
        self.assertEqual([row.post_rms_amplitude for row in ARM_SPECS[3:6]], [1.25, 1.5, 1.75])
        self.assertFalse(ARM_SPECS[0].recompute_preclip_direction)
        self.assertTrue(all(row.recompute_preclip_direction for row in ARM_SPECS[1:]))
        self.assertFalse(ARM_SPECS[-1].selectable)
        self.assertFalse(POSTCLIP_INTERPOLATION_ALLOWED)
        self.assertFalse(TEST_115_AUTHORIZED)

    def test_inherited_c5_mechanism_is_frozen(self):
        self.assertEqual(DESCRIPTOR_ID, "ZSCORED_INTENSITY")
        self.assertEqual(IMAGE_NORMALIZATION_MODE, "independent_masked_zscore")
        self.assertEqual(IMAGE_NORMALIZATION_STD_FLOOR, 1e-6)
        self.assertEqual(STANDARDIZATION_MODE, "centered_two_pass_fp32")
        self.assertEqual(STANDARDIZATION_FLOOR, 1e-6)
        self.assertEqual(DECODER_MODE, "posterior_mean")
        self.assertEqual(POSTERIOR_TEMPERATURE, 1.0)
        self.assertEqual(PRE_RMS_MULTIPLIER, 1.0)
        self.assertEqual(COMMON_EVIDENCE_COLLAR, 7)
        self.assertEqual(CANDIDATE_COUNT, 27)
        self.assertEqual(CANDIDATE_OFFSETS_ZYX[0], (-4, -4, -4))
        self.assertEqual(CANDIDATE_OFFSETS_ZYX[13], (0, 0, 0))
        self.assertEqual(CANDIDATE_OFFSETS_ZYX[-1], (4, 4, 4))
        self.assertEqual(BOOTSTRAP_FAMILY_SIZES[regional_zero_family_id(SELECTABLE_ARM_IDS[0])], 90)
        self.assertEqual(BOOTSTRAP_FAMILY_SIZES[regional_repair_family_id(SELECTABLE_ARM_IDS[0])], 6)

    def test_policy_hash_and_bytes_are_stable(self):
        self.assertEqual(policy_sha256(), C5B_POLICY_SHA256)
        self.assertEqual(policy_sha256(), "ff4efaf30d836b4a0bfda89f22b1c1715d86f5349bc5e4a0c2b706ce03ed1b1c")
        policy = dict(C5B_POLICY)
        self.assertEqual(policy["winner_mean_tie_tolerance"], WINNER_MEAN_TIE_TOLERANCE)
        self.assertEqual(policy["winner_tie_break_order"], WINNER_TIE_BREAK_ORDER)
        self.assertEqual(canonical_policy_bytes(), canonical_policy_bytes())

    def test_diagnostic_arm_cannot_be_assessed_or_enter_branch(self):
        with self.assertRaisesRegex(ValueError, "not a selectable"):
            assess_arm(evidence(DIAGNOSTIC_ARM_ID))
        rows = tuple(evidence(arm_id) for arm_id in SELECTABLE_ARM_IDS)
        diagnostic = dataclasses.replace(rows[-1], arm_id=DIAGNOSTIC_ARM_ID)
        with self.assertRaisesRegex(ValueError, "frozen order"):
            select_next_branch((*rows[:2], diagnostic))


class SimultaneousBootstrapTest(unittest.TestCase):
    def test_three_selectable_dice_contrasts_share_one_max_deviation(self):
        values = {
            f"dice::{arm_id}::vs_{REFERENCE_ARM_ID}": np.linspace(index, index + 0.01, EXPECTED_CASE_COUNT)
            for index, arm_id in enumerate(SELECTABLE_ARM_IDS)
        }
        rows = simultaneous_paired_summaries(DICE_VS_REFERENCE_FAMILY_ID, values)
        self.assertEqual(len(rows), 3)
        self.assertTrue(all(row.simultaneous_family_size == 3 for row in rows))
        widths = [row.ci_high - row.mean for row in rows]
        self.assertTrue(np.allclose(widths, widths[0], rtol=0.0, atol=1e-15))

    def test_bootstrap_rejects_reordering_and_wrong_case_count(self):
        ids = [f"dice::{arm_id}::vs_{REFERENCE_ARM_ID}" for arm_id in SELECTABLE_ARM_IDS]
        values = {contrast_id: np.ones(EXPECTED_CASE_COUNT) for contrast_id in ids}
        with self.assertRaisesRegex(ValueError, "frozen order"):
            simultaneous_paired_summaries(DICE_VS_REFERENCE_FAMILY_ID, dict(reversed(tuple(values.items()))))
        values[ids[0]] = np.ones(EXPECTED_CASE_COUNT - 1)
        with self.assertRaisesRegex(ValueError, "exactly 58"):
            simultaneous_paired_summaries(DICE_VS_REFERENCE_FAMILY_ID, values)


class AssessmentBoundaryTest(unittest.TestCase):
    def test_success_boundaries_are_inclusive_or_strict_as_frozen(self):
        row = evidence(
            SELECTABLE_ARM_IDS[0],
            dice_mean=0.0005,
            dice_median=np.nextafter(0.0, 1.0),
            dice_low=np.nextafter(0.0, 1.0),
            sd_high=0.005,
            retention_median=0.95,
            retention_cases=AMPLITUDE_RETENTION_CASE_COUNT_MIN,
        )
        assessed = assess_arm(row)
        self.assertTrue(assessed.superior_success)

    def test_zero_median_or_ci_low_is_not_superior(self):
        median_zero = assess_arm(evidence(SELECTABLE_ARM_IDS[0], dice_median=0.0))
        low_zero = assess_arm(evidence(SELECTABLE_ARM_IDS[0], dice_low=0.0))
        self.assertFalse(median_zero.dice_superior)
        self.assertFalse(low_zero.dice_superior)

    def test_sdlogj_upper_interval_above_margin_is_rejected(self):
        row = assess_arm(evidence(SELECTABLE_ARM_IDS[0], sd_high=np.nextafter(0.005, 1.0)))
        self.assertFalse(row.geometry_noninferior)

    def test_regional_thresholds_are_strict(self):
        arm_id = SELECTABLE_ARM_IDS[0]
        all_label_edge = assess_arm(evidence(arm_id, regional_evidence=regional(arm_id, zero_low=-0.005)))
        risk_edge = assess_arm(evidence(arm_id, regional_evidence=regional(arm_id, risk_low=-0.002)))
        self.assertFalse(all_label_edge.region_safe)
        self.assertFalse(risk_edge.region_safe)


class BranchTest(unittest.TestCase):
    def rows(self, **changes: ArmEvidence) -> tuple[ArmEvidence, ...]:
        return tuple(changes.get(arm_id, evidence(arm_id)) for arm_id in SELECTABLE_ARM_IDS)

    def test_integrity_failure_is_invalid(self):
        arm_id = SELECTABLE_ARM_IDS[0]
        for row in (
            evidence(arm_id, complete=False),
            evidence(arm_id, exact=False),
            evidence(arm_id, folds=1),
        ):
            with self.subTest(row=row):
                decision = select_next_branch(self.rows(**{arm_id: row}))
                self.assertEqual(decision.branch_id, BRANCH_INVALID)

    def test_no_interpretable_amplitude_closes_the_bridge(self):
        rows = tuple(evidence(arm_id, retention_median=0.94, retention_cases=51) for arm_id in SELECTABLE_ARM_IDS)
        self.assertEqual(select_next_branch(rows).branch_id, BRANCH_CLOSE_CLIP)

    def test_no_superiority_moves_to_true_pyramid(self):
        rows = tuple(
            evidence(
                arm_id,
                dice_mean=0.0004,
                dice_low=0.0001,
                regional_evidence=regional(arm_id, repair_mean=0.001, repair_low=-0.001),
            )
            for arm_id in SELECTABLE_ARM_IDS
        )
        self.assertEqual(select_next_branch(rows).branch_id, BRANCH_CLOSE_NO_SUPERIORITY)

    def test_geometry_limit_is_distinct(self):
        rows = tuple(
            evidence(
                arm_id,
                sd_high=0.006,
                regional_evidence=regional(arm_id, repair_mean=0.001, repair_low=-0.001),
            )
            for arm_id in SELECTABLE_ARM_IDS
        )
        self.assertEqual(select_next_branch(rows).branch_id, BRANCH_CLOSE_GEOMETRY)

    def test_regional_risk_is_distinct(self):
        rows = tuple(
            evidence(
                arm_id,
                regional_evidence=regional(arm_id, risk_low=-0.003, repair_mean=0.001, repair_low=-0.001),
            )
            for arm_id in SELECTABLE_ARM_IDS
        )
        self.assertEqual(select_next_branch(rows).branch_id, BRANCH_CLOSE_REGIONAL)

    def test_superior_safe_winner_is_frozen(self):
        first, second, third = SELECTABLE_ARM_IDS
        rows = self.rows(
            **{
                first: evidence(first, dice_mean=0.0010, sd_mean=0.001),
                second: evidence(second, dice_mean=0.0015, sd_mean=0.002),
                third: evidence(third, dice_mean=0.0012, sd_mean=0.000),
            }
        )
        decision = select_next_branch(rows)
        self.assertEqual(decision.branch_id, BRANCH_FREEZE_SUPERIOR)
        self.assertEqual(decision.selected_arm_id, second)

    def test_tied_winner_prefers_lower_sdlogj(self):
        first, second, third = SELECTABLE_ARM_IDS
        rows = self.rows(
            **{
                first: evidence(first, dice_mean=0.0010000, sd_mean=0.002),
                second: evidence(second, dice_mean=0.0010005, sd_mean=0.001),
                third: evidence(third, dice_mean=0.0008, sd_mean=0.0),
            }
        )
        self.assertEqual(select_next_branch(rows).selected_arm_id, second)

    def test_tied_winner_prefers_lower_amplitude_after_equal_sdlogj(self):
        first, second, third = SELECTABLE_ARM_IDS
        rows = self.rows(
            **{
                first: evidence(first, dice_mean=0.0010000, sd_mean=0.001),
                second: evidence(second, dice_mean=0.0010005, sd_mean=0.001),
                third: evidence(third, dice_mean=0.0008, sd_mean=0.0),
            }
        )
        self.assertEqual(select_next_branch(rows).selected_arm_id, first)

    def test_risk_repair_can_freeze_without_superiority(self):
        arm_id, *others = SELECTABLE_ARM_IDS
        repaired = evidence(
            arm_id,
            dice_mean=0.0002,
            dice_median=0.0002,
            dice_low=-0.0005,
            regional_evidence=regional(arm_id, risk_low=-0.003, repair_mean=0.003, repair_low=0.001),
        )
        changes = {arm_id: repaired}
        for other in others:
            changes[other] = evidence(
                other,
                dice_mean=0.0002,
                dice_median=0.0002,
                dice_low=-0.001,
                regional_evidence=regional(other, repair_mean=0.001, repair_low=-0.001),
            )
        decision = select_next_branch(self.rows(**changes))
        self.assertEqual(decision.branch_id, BRANCH_FREEZE_RISK_REPAIR)
        self.assertEqual(decision.selected_arm_id, arm_id)

    def test_risk_repair_requires_strict_aggregate_and_label13_safety(self):
        arm_id = SELECTABLE_ARM_IDS[0]
        aggregate_edge = evidence(
            arm_id,
            dice_mean=0.0,
            dice_median=0.0,
            dice_low=-0.001,
            regional_evidence=regional(arm_id, risk_low=-0.003),
        )
        label_13_edge = dataclasses.replace(
            aggregate_edge,
            dice_vs_reference=dataclasses.replace(aggregate_edge.dice_vs_reference, ci_low=-0.0009),
            regional=regional(arm_id, risk_low=-0.003, label_13_low=-0.002),
        )
        self.assertFalse(assess_arm(aggregate_edge).risk_repair_success)
        self.assertFalse(assess_arm(label_13_edge).risk_repair_success)


if __name__ == "__main__":
    unittest.main()
