from __future__ import annotations

import dataclasses
import json
import math
import unittest

import numpy as np

from tools.analysis.search_gate_c5 import (
    AMPLITUDE_RETENTION_CASE_COUNT_MIN,
    ARM_SPECS,
    BIAS_LEVELS,
    BOOTSTRAP_CONFIDENCE,
    BOOTSTRAP_FAMILY_SIZES,
    BOOTSTRAP_MAIN_CONTRAST_COUNT,
    BOOTSTRAP_METHOD_ID,
    BOOTSTRAP_RESAMPLES,
    BOOTSTRAP_SEED,
    BRANCH_CLIPPING,
    BRANCH_CLOSE_NO_CAPACITY,
    BRANCH_FACTOR_BOUNDARY,
    BRANCH_FREEZE_RISK_REPAIR,
    BRANCH_FREEZE_SUPERIOR,
    BRANCH_GEOMETRY,
    BRANCH_INVALID,
    BRANCH_REGION_RISK,
    BRANCH_RETAIN_REFERENCE,
    BRANCH_UTILITY,
    C5_DECISION_POLICY,
    C5_DECISION_POLICY_SHA256,
    C5_POLICY,
    C5_POLICY_SHA256,
    CAPACITY_FAMILY_ID,
    CONTRAST_IDS_BY_FAMILY,
    CONTRAST_SPECS,
    EVALUATION_LABEL_IDS,
    EXPECTED_CASE_COUNT,
    HISTORICAL_ANCHOR_ARM_IDS,
    INCREMENTAL_FAMILY_ID,
    INTERACTION_FAMILY_ID,
    MARGINAL_FAMILY_ID,
    PRIMARY_REFERENCE_ARM_ID,
    PRIMARY_SELECTOR_ID,
    PROTOCOL_ID,
    REACH_SPECS,
    REGIONAL_REFERENCE_FAMILY_ID,
    REGIONAL_ZERO_FAMILY_ID,
    SELECTABLE_ARM_IDS,
    SELECTOR_IDS,
    SELECTOR_REFERENCE_FAMILY_ID,
    SELECTOR_SPECS,
    SELECTOR_ZERO_FAMILY_ID,
    TEST_115_AUTHORIZED,
    ArmEvidence,
    CandidateSignals,
    PairedSummary,
    RegionalEvidence,
    SelectorEvidence,
    apply_post_rms_amplitude,
    assess_arm,
    canonical_decision_policy_bytes,
    canonical_policy_bytes,
    centre_log_prior,
    choose_global_candidate,
    decision_policy_contract,
    decision_policy_sha256,
    factor_contrast_differences,
    policy_sha256,
    region_repair_safe,
    region_repaired,
    region_safe,
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
) -> PairedSummary:
    median_value = mean if median is None else median
    low = mean if ci_low is None else ci_low
    return PairedSummary(
        family_id=family_id,
        contrast_id=contrast_id,
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
        simultaneous_family_size=BOOTSTRAP_FAMILY_SIZES[family_id],
    )


def arm_evidence(
    arm_id: str,
    *,
    capacity_mean: float = 0.003,
    capacity_low: float = 0.002,
    incremental_mean: float = 0.001,
    incremental_low: float = 0.0002,
    retention_median: float = 0.99,
    retention_cases: int = 58,
    complete: bool = True,
    exact: bool = True,
) -> ArmEvidence:
    incremental = None
    if arm_id != PRIMARY_REFERENCE_ARM_ID:
        incremental = summary(
            INCREMENTAL_FAMILY_ID,
            f"incremental::{arm_id}::vs_{PRIMARY_REFERENCE_ARM_ID}",
            incremental_mean,
            ci_low=incremental_low,
        )
    return ArmEvidence(
        arm_id=arm_id,
        capacity_vs_zero=summary(
            CAPACITY_FAMILY_ID,
            f"capacity::{arm_id}::vs_zero",
            capacity_mean,
            ci_low=capacity_low,
        ),
        incremental_vs_reference=incremental,
        all_work_units_complete=complete,
        all_exact_certified=exact,
        amplitude_retention_median=retention_median,
        amplitude_retention_cases_at_least_090=retention_cases,
    )


def selector_evidence(
    selector_id: str,
    *,
    zero_mean: float = 0.002,
    zero_low: float = 0.0005,
    reference_mean: float = 0.001,
    reference_low: float = 0.0002,
    complete: bool = True,
    valid: bool = True,
) -> SelectorEvidence:
    return SelectorEvidence(
        selector_id=selector_id,
        vs_zero=summary(
            SELECTOR_ZERO_FAMILY_ID,
            f"selector::{selector_id}::vs_zero",
            zero_mean,
            ci_low=zero_low,
        ),
        vs_reference=summary(
            SELECTOR_REFERENCE_FAMILY_ID,
            f"selector::{selector_id}::vs_{PRIMARY_REFERENCE_ARM_ID}",
            reference_mean,
            ci_low=reference_low,
        ),
        all_choices_complete=complete,
        all_choices_contract_valid=valid,
    )


def regional_evidence(
    *,
    zero_low: float = -0.001,
    risk_zero_low: float | None = None,
    label_13_zero_low: float | None = None,
    repair_mean: float = 0.003,
    repair_low: float = 0.001,
) -> RegionalEvidence:
    zero_rows = []
    for label_id in EVALUATION_LABEL_IDS:
        if risk_zero_low is not None and label_id in (9, 29):
            low = risk_zero_low
        elif label_13_zero_low is not None and label_id == 13:
            low = label_13_zero_low
        else:
            low = zero_low
        zero_rows.append(
            summary(
                REGIONAL_ZERO_FAMILY_ID,
                f"label::{label_id}::{PRIMARY_SELECTOR_ID}::vs_zero",
                max(low + 0.001, 0.0001),
                ci_low=low,
            )
        )
    reference_rows = tuple(
        summary(
            REGIONAL_REFERENCE_FAMILY_ID,
            f"label::{label_id}::{PRIMARY_SELECTOR_ID}::vs_{PRIMARY_REFERENCE_ARM_ID}",
            repair_mean,
            ci_low=repair_low,
        )
        for label_id in (9, 29)
    )
    return RegionalEvidence(tuple(zero_rows), reference_rows)


def marginal_summaries(*, boundary_positive: bool = False) -> tuple[PairedSummary, ...]:
    rows = []
    for contrast_id in CONTRAST_IDS_BY_FAMILY[MARGINAL_FAMILY_ID]:
        positive = boundary_positive and contrast_id == "reach_s4_vs_s3"
        rows.append(
            summary(
                MARGINAL_FAMILY_ID,
                contrast_id,
                0.001 if positive else 0.0,
                median=0.001 if positive else 0.0,
                ci_low=0.0002 if positive else -0.001,
            )
        )
    return tuple(rows)


class FrozenProtocolTest(unittest.TestCase):
    def test_axes_and_reach_major_arm_order_are_exact(self):
        self.assertEqual(PROTOCOL_ID, "CTCF-SEARCH-GATE-C5-V1")
        self.assertFalse(TEST_115_AUTHORIZED)
        self.assertEqual(len(ARM_SPECS), 36)
        self.assertEqual([arm.arm_index for arm in ARM_SPECS], list(range(36)))
        self.assertEqual(
            SELECTABLE_ARM_IDS[:10],
            (
                "int_s1_a05_b0",
                "int_s1_a05_b1",
                "int_s1_a05_b2",
                "int_s1_a10_b0",
                "int_s1_a10_b1",
                "int_s1_a10_b2",
                "int_s1_a20_b0",
                "int_s1_a20_b1",
                "int_s1_a20_b2",
                "int_s2_a05_b0",
            ),
        )
        self.assertEqual(SELECTABLE_ARM_IDS[-1], "int_s4_a20_b2")
        self.assertEqual(HISTORICAL_ANCHOR_ARM_IDS, ("int_s1_a10_b0", "int_s2_a10_b0"))
        self.assertEqual(PRIMARY_REFERENCE_ARM_ID, "int_s2_a10_b0")

    def test_offsets_and_bias_dose_are_frozen(self):
        for reach in REACH_SPECS:
            self.assertEqual(len(reach.offsets_zyx), 27)
            self.assertEqual(len(set(reach.offsets_zyx)), 27)
            self.assertEqual(reach.offsets_zyx[0], (-reach.stride_voxels,) * 3)
            self.assertEqual(reach.offsets_zyx[13], (0, 0, 0))
            self.assertEqual(reach.offsets_zyx[-1], (reach.stride_voxels,) * 3)
        self.assertEqual([reach.pre_rms_multiplier for reach in REACH_SPECS], [2.0, 1.0, 1.0, 1.0])
        self.assertEqual(BIAS_LEVELS[0][2], 0.0)
        self.assertEqual(BIAS_LEVELS[1][2], math.log(2.0))
        self.assertEqual(BIAS_LEVELS[2][2], math.log(4.0))
        prior = centre_log_prior("int_s4_a10_b1")
        self.assertEqual(prior[13], 0.0)
        self.assertAlmostEqual(prior[12], -math.log(2.0))
        self.assertAlmostEqual(prior[0], -3.0 * math.log(2.0))

    def test_amplitude_is_explicitly_post_rms_and_nonmutating(self):
        values = np.array([1.0, -2.0], dtype=np.float32)
        original = values.copy()
        scaled = apply_post_rms_amplitude(values, "int_s2_a20_b0")
        np.testing.assert_array_equal(scaled, np.array([2.0, -4.0], dtype=np.float32))
        np.testing.assert_array_equal(values, original)
        self.assertEqual(dict(C5_POLICY.proposal_pipeline)["amplitude_stage"], "after_rms_match_before_local_clip")
        with self.assertRaises(ValueError):
            apply_post_rms_amplitude([1.0], "missing")

    def test_five_global_selectors_are_exact(self):
        self.assertEqual(
            SELECTOR_IDS,
            ("dual_g010", "dual_g005", "dual_g020", "ncc_g010", "mind_g010"),
        )
        primary, tight, loose, ncc, mind = SELECTOR_SPECS
        self.assertTrue(primary.primary)
        self.assertEqual(
            (tight.geometry_delta_cap, primary.geometry_delta_cap, loose.geometry_delta_cap), (0.005, 0.010, 0.020)
        )
        self.assertEqual(
            [(row.requires_ncc7_improvement, row.requires_mind_d2_improvement) for row in (ncc, mind)],
            [(True, False), (False, True)],
        )

    def test_contrast_families_are_exact_unique_and_separate(self):
        self.assertEqual(BOOTSTRAP_MAIN_CONTRAST_COUNT, 92)
        self.assertEqual(
            dict(BOOTSTRAP_FAMILY_SIZES),
            {
                "capacity_vs_zero": 36,
                "capacity_vs_c4_intensity_s2": 35,
                "factor_adjacent_marginals": 7,
                "factor_trend_interactions": 4,
                "selector_vs_zero": 5,
                "selector_vs_c4_intensity_s2": 5,
                "primary_selector_labels_vs_zero": 30,
                "primary_selector_risk_labels_vs_c4_intensity_s2": 2,
            },
        )
        ids = [row.contrast_id for row in CONTRAST_SPECS]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(
            CONTRAST_IDS_BY_FAMILY[MARGINAL_FAMILY_ID],
            (
                "reach_s2_vs_s1",
                "reach_s3_vs_s2",
                "reach_s4_vs_s3",
                "amplitude_a10_vs_a05",
                "amplitude_a20_vs_a10",
                "bias_b1_vs_b0",
                "bias_b2_vs_b1",
            ),
        )
        self.assertEqual(
            EVALUATION_LABEL_IDS,
            (
                1,
                2,
                3,
                5,
                6,
                7,
                8,
                9,
                10,
                11,
                12,
                13,
                14,
                15,
                16,
                18,
                20,
                21,
                22,
                23,
                25,
                26,
                27,
                28,
                29,
                30,
                31,
                32,
                34,
                36,
            ),
        )

    def test_policy_hash_is_literal_canonical_and_immutable(self):
        self.assertEqual(policy_sha256(), C5_POLICY_SHA256)
        self.assertRegex(C5_POLICY_SHA256, r"^[0-9a-f]{64}$")
        json.loads(canonical_policy_bytes())
        with self.assertRaises(dataclasses.FrozenInstanceError):
            C5_POLICY.protocol_id = "mutated"
        with self.assertRaises(dataclasses.FrozenInstanceError):
            ARM_SPECS[0].arm_id = "mutated"

    def test_decision_policy_is_separately_hashed_and_contains_no_evaluation_metadata(self):
        self.assertEqual(decision_policy_sha256(), C5_DECISION_POLICY_SHA256)
        self.assertRegex(C5_DECISION_POLICY_SHA256, r"^[0-9a-f]{64}$")
        self.assertEqual(canonical_decision_policy_bytes(), canonical_decision_policy_bytes(C5_DECISION_POLICY))
        payload = decision_policy_contract()
        self.assertFalse(payload["labels_accessible"])
        encoded = json.dumps(payload, sort_keys=True)
        for forbidden in ("label::", "regional", "dice", "contrast"):
            self.assertNotIn(forbidden, encoded.lower())
        self.assertNotIn("thresholds", payload)
        with self.assertRaises(dataclasses.FrozenInstanceError):
            C5_DECISION_POLICY.protocol_id = "mutated"


class SelectorTest(unittest.TestCase):
    def rows(self) -> list[CandidateSignals]:
        return [CandidateSignals(arm_id, True, 1.0, 1.0, 0.01, 0.01, 0.0) for arm_id in SELECTABLE_ARM_IDS]

    def test_primary_ranking_is_lexicographic_and_not_dice_informed(self):
        rows = self.rows()
        first = dataclasses.replace(rows[0], mind_d2_improvement=0.02, ncc7_improvement=0.02)
        second = dataclasses.replace(rows[1], mind_d2_improvement=0.02, ncc7_improvement=0.03)
        rows[0], rows[1] = first, second
        choice = choose_global_candidate(rows, PRIMARY_SELECTOR_ID)
        self.assertEqual(choice.selected_arm_id, rows[1].arm_id)
        rows[0] = dataclasses.replace(first, mind_d2_improvement=0.021, ncc7_improvement=0.000002)
        self.assertEqual(choose_global_candidate(rows, PRIMARY_SELECTOR_ID).selected_arm_id, rows[0].arm_id)

    def test_tie_break_prefers_lower_amplitude_stride_bias_then_index(self):
        rows = self.rows()
        for index, row in enumerate(rows):
            rows[index] = dataclasses.replace(row, exact_certified=False)
        preferred = SELECTABLE_ARM_IDS.index("int_s1_a05_b0")
        other = SELECTABLE_ARM_IDS.index("int_s4_a20_b2")
        rows[preferred] = dataclasses.replace(rows[preferred], exact_certified=True)
        rows[other] = dataclasses.replace(rows[other], exact_certified=True)
        self.assertEqual(choose_global_candidate(rows, PRIMARY_SELECTOR_ID).selected_arm_id, "int_s1_a05_b0")

    def test_single_utility_selectors_do_not_use_the_disabled_utility_for_ranking(self):
        rows = self.rows()
        for index, row in enumerate(rows):
            rows[index] = dataclasses.replace(row, exact_certified=False)
        first, second = 0, 1
        rows[first] = dataclasses.replace(
            rows[first],
            exact_certified=True,
            ncc7_improvement=0.02,
            mind_d2_improvement=0.01,
        )
        rows[second] = dataclasses.replace(
            rows[second],
            exact_certified=True,
            ncc7_improvement=0.02,
            mind_d2_improvement=0.50,
        )
        self.assertEqual(choose_global_candidate(rows, "ncc_g010").selected_arm_id, rows[first].arm_id)

        rows[first] = dataclasses.replace(rows[first], ncc7_improvement=0.01, mind_d2_improvement=0.02)
        rows[second] = dataclasses.replace(rows[second], ncc7_improvement=0.50, mind_d2_improvement=0.02)
        self.assertEqual(choose_global_candidate(rows, "mind_g010").selected_arm_id, rows[first].arm_id)

    def test_each_predicate_excludes_and_empty_selector_returns_baseline(self):
        rows = self.rows()
        for index, row in enumerate(rows):
            rows[index] = dataclasses.replace(row, exact_certified=False)
        rows[0] = CandidateSignals(SELECTABLE_ARM_IDS[0], True, 0.989, 1.0, 0.1, 0.1, 0.0)
        choice = choose_global_candidate(rows, PRIMARY_SELECTOR_ID)
        self.assertEqual((choice.action, choice.selected_arm_id), ("RETURN_BASELINE", None))
        rows[0] = dataclasses.replace(rows[0], support_retention=1.0, mathematical_sdlogj_delta=0.011)
        self.assertIsNone(choose_global_candidate(rows, PRIMARY_SELECTOR_ID).selected_arm_id)
        self.assertEqual(choose_global_candidate(rows, "dual_g020").selected_arm_id, rows[0].arm_id)
        rows[0] = dataclasses.replace(rows[0], mathematical_sdlogj_delta=0.0, mind_d2_improvement=0.0)
        self.assertIsNone(choose_global_candidate(rows, PRIMARY_SELECTOR_ID).selected_arm_id)
        self.assertEqual(choose_global_candidate(rows, "ncc_g010").selected_arm_id, rows[0].arm_id)
        rows[0] = dataclasses.replace(rows[0], mind_d2_improvement=0.1, amplitude_retention=0.899)
        self.assertIsNone(choose_global_candidate(rows, "ncc_g010").selected_arm_id)

    def test_malformed_or_reordered_evidence_fails_closed(self):
        rows = self.rows()
        with self.assertRaises(ValueError):
            choose_global_candidate(rows[:-1], PRIMARY_SELECTOR_ID)
        with self.assertRaises(ValueError):
            choose_global_candidate(tuple(reversed(rows)), PRIMARY_SELECTOR_ID)
        rows[0] = dataclasses.replace(rows[0], support_retention=float("nan"))
        with self.assertRaises(ValueError):
            choose_global_candidate(rows, PRIMARY_SELECTOR_ID)


class FactorAndInferenceTest(unittest.TestCase):
    def test_additive_factorial_recovers_adjacent_effects_and_zero_interactions(self):
        rows: dict[str, np.ndarray] = {}
        for arm in ARM_SPECS:
            value = 0.1 * arm.stride_voxels + math.log2(arm.post_rms_amplitude) + 2.0 * arm.bias_level
            rows[arm.arm_id] = np.full(EXPECTED_CASE_COUNT, value)
        contrasts = factor_contrast_differences(rows)
        for contrast_id in ("reach_s2_vs_s1", "reach_s3_vs_s2", "reach_s4_vs_s3"):
            np.testing.assert_allclose(contrasts[contrast_id], 0.1)
        for contrast_id in ("amplitude_a10_vs_a05", "amplitude_a20_vs_a10"):
            np.testing.assert_allclose(contrasts[contrast_id], 1.0)
        for contrast_id in ("bias_b1_vs_b0", "bias_b2_vs_b1"):
            np.testing.assert_allclose(contrasts[contrast_id], 2.0)
        for contrast_id in CONTRAST_IDS_BY_FAMILY[INTERACTION_FAMILY_ID]:
            np.testing.assert_allclose(contrasts[contrast_id], 0.0, atol=1e-12)

    def test_factor_input_must_be_complete_and_ordered(self):
        rows = {arm_id: np.zeros(EXPECTED_CASE_COUNT) for arm_id in SELECTABLE_ARM_IDS}
        rows.pop(SELECTABLE_ARM_IDS[-1])
        with self.assertRaises(ValueError):
            factor_contrast_differences(rows)

    def test_simultaneous_bootstrap_requires_exact_family_and_is_deterministic(self):
        ids = CONTRAST_IDS_BY_FAMILY[INTERACTION_FAMILY_ID]
        values = {
            contrast_id: np.linspace(0.0, 0.001 * (index + 1), EXPECTED_CASE_COUNT)
            for index, contrast_id in enumerate(ids)
        }
        first = simultaneous_paired_summaries(INTERACTION_FAMILY_ID, values)
        second = simultaneous_paired_summaries(INTERACTION_FAMILY_ID, values)
        self.assertEqual(first, second)
        self.assertTrue(all(row.simultaneous_family_size == 4 for row in first.values()))
        with self.assertRaises(ValueError):
            simultaneous_paired_summaries(INTERACTION_FAMILY_ID, dict(reversed(tuple(values.items()))))


class BranchSelectionTest(unittest.TestCase):
    def arms(self, **changes: ArmEvidence) -> tuple[ArmEvidence, ...]:
        return tuple(changes.get(arm_id, arm_evidence(arm_id)) for arm_id in SELECTABLE_ARM_IDS)

    def selectors(self, **changes: SelectorEvidence) -> tuple[SelectorEvidence, ...]:
        return tuple(changes.get(selector_id, selector_evidence(selector_id)) for selector_id in SELECTOR_IDS)

    def decide(
        self,
        *,
        arms: tuple[ArmEvidence, ...] | None = None,
        selectors: tuple[SelectorEvidence, ...] | None = None,
        marginals: tuple[PairedSummary, ...] | None = None,
        regional: RegionalEvidence | None = None,
        integrity: bool = True,
    ):
        return select_next_branch(
            self.arms() if arms is None else arms,
            self.selectors() if selectors is None else selectors,
            marginal_summaries() if marginals is None else marginals,
            regional_evidence() if regional is None else regional,
            integrity_passed=integrity,
        )

    def test_invalid_and_clipping_have_highest_priority(self):
        self.assertEqual(self.decide(integrity=False).branch_id, BRANCH_INVALID)
        clipped = arm_evidence(
            SELECTABLE_ARM_IDS[0],
            retention_cases=AMPLITUDE_RETENTION_CASE_COUNT_MIN - 1,
        )
        self.assertEqual(self.decide(arms=self.arms(**{clipped.arm_id: clipped})).branch_id, BRANCH_CLIPPING)

    def test_nonmaterial_saturated_arm_does_not_mask_interpretable_winner(self):
        clipped = arm_evidence(
            SELECTABLE_ARM_IDS[0],
            capacity_mean=0.001,
            capacity_low=0.0001,
            retention_median=0.5,
            retention_cases=0,
        )
        self.assertEqual(
            self.decide(arms=self.arms(**{clipped.arm_id: clipped})).branch_id,
            BRANCH_FREEZE_SUPERIOR,
        )

    def test_incomplete_arm_or_selector_is_invalid_not_a_scientific_branch(self):
        incomplete_arm = arm_evidence(SELECTABLE_ARM_IDS[0], exact=False)
        self.assertEqual(
            self.decide(arms=self.arms(**{incomplete_arm.arm_id: incomplete_arm})).branch_id,
            BRANCH_INVALID,
        )
        invalid_selector = selector_evidence(PRIMARY_SELECTOR_ID, valid=False)
        self.assertEqual(
            self.decide(selectors=self.selectors(dual_g010=invalid_selector)).branch_id,
            BRANCH_INVALID,
        )

    def test_no_material_capacity_closes_sparse_grid(self):
        rows = tuple(arm_evidence(arm_id, capacity_mean=0.001, capacity_low=0.0001) for arm_id in SELECTABLE_ARM_IDS)
        self.assertEqual(self.decide(arms=rows).branch_id, BRANCH_CLOSE_NO_CAPACITY)

    def test_upper_corner_and_positive_adjacent_marginal_open_boundary(self):
        rows = []
        for arm_id in SELECTABLE_ARM_IDS:
            mean = 0.005 if arm_id == "int_s4_a20_b2" else 0.003
            rows.append(arm_evidence(arm_id, capacity_mean=mean))
        decision = self.decide(arms=tuple(rows), marginals=marginal_summaries(boundary_positive=True))
        self.assertEqual((decision.branch_id, decision.selected_arm_id), (BRANCH_FACTOR_BOUNDARY, "int_s4_a20_b2"))

    def test_each_factor_boundary_is_detected_without_requiring_the_three_way_corner(self):
        marginal_ids = CONTRAST_IDS_BY_FAMILY[MARGINAL_FAMILY_ID]
        for best_arm, positive_id in (
            ("int_s4_a10_b0", "reach_s4_vs_s3"),
            ("int_s2_a20_b0", "amplitude_a20_vs_a10"),
            ("int_s2_a10_b2", "bias_b2_vs_b1"),
        ):
            with self.subTest(best_arm=best_arm):
                rows = tuple(
                    arm_evidence(arm_id, capacity_mean=0.005 if arm_id == best_arm else 0.003)
                    for arm_id in SELECTABLE_ARM_IDS
                )
                marginals = tuple(
                    summary(
                        MARGINAL_FAMILY_ID,
                        contrast_id,
                        mean=0.0006 if contrast_id == positive_id else 0.0,
                        ci_low=0.0001 if contrast_id == positive_id else -0.0001,
                    )
                    for contrast_id in marginal_ids
                )
                decision = self.decide(arms=rows, marginals=marginals)
                self.assertEqual((decision.branch_id, decision.selected_arm_id), (BRANCH_FACTOR_BOUNDARY, best_arm))

    def test_positive_unrelated_boundary_marginal_does_not_open_boundary(self):
        best_arm = "int_s2_a10_b0"
        rows = tuple(
            arm_evidence(arm_id, capacity_mean=0.005 if arm_id == best_arm else 0.003) for arm_id in SELECTABLE_ARM_IDS
        )
        marginals = tuple(
            summary(
                MARGINAL_FAMILY_ID,
                contrast_id,
                mean=0.0006 if contrast_id == "reach_s4_vs_s3" else 0.0,
                ci_low=0.0001 if contrast_id == "reach_s4_vs_s3" else -0.0001,
            )
            for contrast_id in CONTRAST_IDS_BY_FAMILY[MARGINAL_FAMILY_ID]
        )
        self.assertNotEqual(self.decide(arms=rows, marginals=marginals).branch_id, BRANCH_FACTOR_BOUNDARY)

    def test_loose_geometry_success_opens_geometry_branch(self):
        primary = selector_evidence(PRIMARY_SELECTOR_ID, zero_mean=0.0005, zero_low=-0.0001)
        loose = selector_evidence("dual_g020", zero_mean=0.002, zero_low=0.0002)
        decision = self.decide(selectors=self.selectors(dual_g010=primary, dual_g020=loose))
        self.assertEqual(decision.branch_id, BRANCH_GEOMETRY)

    def test_primary_region_failure_opens_region_risk(self):
        decision = self.decide(
            regional=regional_evidence(
                risk_zero_low=-0.0021,
                repair_mean=0.001,
                repair_low=-0.0001,
            )
        )
        self.assertEqual(decision.branch_id, BRANCH_REGION_RISK)

    def test_primary_failure_without_geometry_signal_opens_utility(self):
        weak = selector_evidence(PRIMARY_SELECTOR_ID, zero_mean=0.0005, zero_low=-0.0001)
        loose = selector_evidence("dual_g020", zero_mean=0.0005, zero_low=-0.0001)
        decision = self.decide(selectors=self.selectors(dual_g010=weak, dual_g020=loose))
        self.assertEqual(decision.branch_id, BRANCH_UTILITY)

    def test_superior_primary_freezes_one_future_independent_confirmation(self):
        decision = self.decide()
        self.assertEqual(
            (decision.branch_id, decision.selected_selector_id), (BRANCH_FREEZE_SUPERIOR, PRIMARY_SELECTOR_ID)
        )

    def test_nonsuperior_but_safe_region_repair_uses_narrow_exception(self):
        primary = selector_evidence(PRIMARY_SELECTOR_ID, reference_mean=0.0, reference_low=-0.0005)
        decision = self.decide(
            selectors=self.selectors(dual_g010=primary),
            regional=regional_evidence(risk_zero_low=-0.003),
        )
        self.assertEqual(decision.branch_id, BRANCH_FREEZE_RISK_REPAIR)
        self.assertTrue(region_safe(regional_evidence()))
        self.assertTrue(region_repaired(regional_evidence()))
        self.assertTrue(region_repair_safe(regional_evidence(risk_zero_low=-0.003)))

    def test_risk_repair_cannot_bypass_general_or_label_13_safety(self):
        primary = selector_evidence(PRIMARY_SELECTOR_ID, reference_mean=0.0, reference_low=-0.0005)
        for regional in (
            regional_evidence(zero_low=-0.006),
            regional_evidence(risk_zero_low=-0.003, label_13_zero_low=-0.0021),
        ):
            decision = self.decide(selectors=self.selectors(dual_g010=primary), regional=regional)
            self.assertEqual(decision.branch_id, BRANCH_REGION_RISK)

    def test_no_superiority_or_repair_retains_c4_reference(self):
        primary = selector_evidence(PRIMARY_SELECTOR_ID, reference_mean=0.0, reference_low=-0.0005)
        decision = self.decide(
            selectors=self.selectors(dual_g010=primary),
            regional=regional_evidence(repair_mean=0.001, repair_low=-0.0001),
        )
        self.assertEqual(
            (decision.branch_id, decision.selected_arm_id), (BRANCH_RETAIN_REFERENCE, PRIMARY_REFERENCE_ARM_ID)
        )

    def test_malformed_order_contrast_or_self_reference_fails_closed(self):
        with self.assertRaises(ValueError):
            self.decide(arms=tuple(reversed(self.arms())))
        bad = dataclasses.replace(
            arm_evidence(PRIMARY_REFERENCE_ARM_ID),
            incremental_vs_reference=summary(
                INCREMENTAL_FAMILY_ID,
                f"incremental::{SELECTABLE_ARM_IDS[0]}::vs_{PRIMARY_REFERENCE_ARM_ID}",
                0.001,
            ),
        )
        with self.assertRaises(ValueError):
            assess_arm(bad)
        reordered = tuple(reversed(regional_evidence().vs_zero))
        with self.assertRaises(ValueError):
            self.decide(regional=dataclasses.replace(regional_evidence(), vs_zero=reordered))


if __name__ == "__main__":
    unittest.main()
