from __future__ import annotations

import math
import unittest
from dataclasses import replace

from tools.analysis.search_gate_c6 import PairedSummary
from tools.analysis.search_gate_c7 import (
    ALL_LABEL_CI_LOW_VS_C4_MIN_STRICT,
    ARM_SPECS,
    BRANCH_CLOSE,
    BRANCH_FREEZE,
    BRANCH_INVALID,
    BRANCH_MATCHED_CONTROL,
    BRANCH_SAFETY,
    BRANCH_SELECTOR,
    CAPACITY_CI_LOW_VS_C4_MIN_STRICT,
    CAPACITY_MEAN_VS_C4_MIN,
    CAUSAL_CI_LOW_VS_INTENSITY_MIN_STRICT,
    CAUSAL_MEAN_VS_INTENSITY_MIN,
    COST_STANDARDIZATION,
    DESCRIPTOR_CHECKPOINT_BYTES,
    DESCRIPTOR_CHECKPOINT_DEFAULT,
    DESCRIPTOR_CHECKPOINT_EPOCH,
    DESCRIPTOR_CHECKPOINT_SHA256,
    DESCRIPTOR_CONV_PADDING_MARGIN,
    DESCRIPTOR_STATE_KEY_COUNT,
    EVALUATION_LABEL_IDS,
    FORBIDDEN_VALIDATION_SELECTED_CHECKPOINT_SHA256,
    MATCHED_CONTROL_ARM_ID,
    RETURNED_CI_LOW_VS_C4_MIN_STRICT,
    RETURNED_MEAN_VS_C4_MIN,
    RISK_LABEL_CI_LOW_VS_C4_MIN_STRICT,
    RISK_LABEL_IDS,
    SDLOGJ_CI_HIGH_VS_C4_MAX,
    SELECTABLE_ARM_IDS,
    SOURCE_CONTEXT_ARM_ID,
    ArmAssessment,
    assert_frozen_policy,
    assess_arm,
    policy_dict,
    policy_sha256,
    select_branch,
)


def summary(mean: float, ci_low: float, ci_high: float | None = None) -> PairedSummary:
    return PairedSummary(
        family_id="synthetic",
        contrast_id="synthetic",
        n=58,
        mean=mean,
        median=mean,
        ci_low=ci_low,
        ci_high=mean if ci_high is None else ci_high,
        improved=58,
        worsened=0,
        tied=0,
        simultaneous_family_size=1,
    )


def above(value: float) -> float:
    return math.nextafter(value, math.inf)


def below(value: float) -> float:
    return math.nextafter(value, -math.inf)


def passing_regional() -> list[tuple[int, PairedSummary]]:
    return [
        (
            label,
            summary(
                0.0,
                above(
                    RISK_LABEL_CI_LOW_VS_C4_MIN_STRICT if label in RISK_LABEL_IDS else ALL_LABEL_CI_LOW_VS_C4_MIN_STRICT
                ),
            ),
        )
        for label in EVALUATION_LABEL_IDS
    ]


def assessment(arm_id: str = SELECTABLE_ARM_IDS[0], **overrides) -> ArmAssessment:
    inputs = {
        "descriptor_valid": True,
        "capacity_vs_c4": summary(CAPACITY_MEAN_VS_C4_MIN, above(CAPACITY_CI_LOW_VS_C4_MIN_STRICT)),
        "causal_vs_intensity": summary(
            CAUSAL_MEAN_VS_INTENSITY_MIN,
            above(CAUSAL_CI_LOW_VS_INTENSITY_MIN_STRICT),
        ),
        "returned_vs_c4": summary(RETURNED_MEAN_VS_C4_MIN, above(RETURNED_CI_LOW_VS_C4_MIN_STRICT)),
        "sdlogj_vs_c4": summary(-0.001, -0.002, SDLOGJ_CI_HIGH_VS_C4_MAX),
        "candidate_regional_vs_initial": passing_regional(),
        "candidate_regional_vs_c4": passing_regional(),
        "returned_regional_vs_initial": passing_regional(),
        "returned_regional_vs_c4": passing_regional(),
        "folds_all_zero": True,
        "clip_retention_median": 0.95,
        "clip_retention_ge_090_count": 52,
        "matched_control_safety_passed": True,
    }
    inputs.update(overrides)
    return assess_arm(arm_id, **inputs)


class C7PolicyTest(unittest.TestCase):
    def test_policy_is_frozen(self) -> None:
        self.assertEqual(policy_sha256(), "daca934f080c0b5b46d428183914b6bbfc81d422322e74c4a65fb55e60cd95ea")
        assert_frozen_policy()

    def test_arm_inventory_separates_historical_context_from_matched_control(self) -> None:
        self.assertEqual(tuple(row.arm_index for row in ARM_SPECS), tuple(range(5)))
        self.assertEqual(ARM_SPECS[1].arm_id, SOURCE_CONTEXT_ARM_ID)
        self.assertEqual(ARM_SPECS[1].role, "FROZEN_C6_CONTEXT")
        self.assertEqual(ARM_SPECS[2].arm_id, MATCHED_CONTROL_ARM_ID)
        self.assertEqual(ARM_SPECS[2].role, "MATCHED_INTENSITY_CONTROL")
        self.assertEqual(tuple(row.arm_id for row in ARM_SPECS if row.selectable), SELECTABLE_ARM_IDS)

    def test_descriptor_uses_fixed_endpoint_and_convolution_padding_support(self) -> None:
        self.assertTrue(DESCRIPTOR_CHECKPOINT_DEFAULT.endswith("last.pth"))
        self.assertEqual(
            DESCRIPTOR_CHECKPOINT_SHA256, "9cafbf426bd8a86cf9bc7e2981fcf7399101af6177292c3e726fc5b56eefa170"
        )
        self.assertEqual(DESCRIPTOR_CONV_PADDING_MARGIN, 2)
        self.assertEqual(DESCRIPTOR_CHECKPOINT_BYTES, 50_703_205)
        self.assertEqual(DESCRIPTOR_CHECKPOINT_EPOCH, 99)
        self.assertEqual(DESCRIPTOR_STATE_KEY_COUNT, 386)
        self.assertEqual(COST_STANDARDIZATION, "centered_two_pass_fp32")
        self.assertNotEqual(DESCRIPTOR_CHECKPOINT_SHA256, FORBIDDEN_VALIDATION_SELECTED_CHECKPOINT_SHA256)
        descriptor = policy_dict()["descriptor"]
        self.assertEqual(
            descriptor["checkpoint_selection"], "fixed_epoch_99_endpoint_not_validation_dice_selected_best"
        )
        self.assertIn("negative_mean_channel_product", descriptor["cost"])
        self.assertEqual(
            policy_dict()["safety"]["clip_retention_per_case"],
            "minimum_of_three_stage_retentions_and_final_retention",
        )

    def test_every_guard_is_required(self) -> None:
        self.assertTrue(assessment().promotion_eligible)
        failures = (
            {"descriptor_valid": False},
            {
                "capacity_vs_c4": summary(
                    below(CAPACITY_MEAN_VS_C4_MIN),
                    above(CAPACITY_CI_LOW_VS_C4_MIN_STRICT),
                )
            },
            {
                "causal_vs_intensity": summary(
                    CAUSAL_MEAN_VS_INTENSITY_MIN,
                    CAUSAL_CI_LOW_VS_INTENSITY_MIN_STRICT,
                )
            },
            {"returned_vs_c4": summary(below(RETURNED_MEAN_VS_C4_MIN), 0.01)},
            {"sdlogj_vs_c4": summary(0.0, -0.01, above(SDLOGJ_CI_HIGH_VS_C4_MAX))},
            {"candidate_regional_vs_c4": passing_regional()[:-1]},
            {
                "returned_regional_vs_initial": [
                    *passing_regional()[:-1],
                    passing_regional()[0],
                ]
            },
            {"folds_all_zero": False},
            {"clip_retention_median": below(0.95)},
            {"clip_retention_ge_090_count": 51},
        )
        for override in failures:
            with self.subTest(override=override):
                self.assertFalse(assessment(**override).promotion_eligible)

    def test_safe_substitution_is_only_available_when_the_matched_control_is_unsafe(self) -> None:
        noninferior = summary(0.0, above(-0.001))
        with_safe_control = assessment(
            causal_vs_intensity=noninferior,
            matched_control_safety_passed=True,
        )
        with_unsafe_control = assessment(
            causal_vs_intensity=noninferior,
            matched_control_safety_passed=False,
        )
        self.assertFalse(with_safe_control.attribution_passed)
        self.assertTrue(with_unsafe_control.attribution_passed)
        self.assertEqual(with_unsafe_control.attribution_mode, "SAFE_SUBSTITUTION")

    def test_regional_inventory_is_the_real_ixi_ontology_not_one_through_thirty(self) -> None:
        self.assertNotEqual(EVALUATION_LABEL_IDS, tuple(range(1, 31)))
        wrong_inventory = [(label, summary(0.0, 0.0)) for label in range(1, 31)]
        self.assertFalse(assessment(candidate_regional_vs_c4=wrong_inventory).candidate_regional_passed)

    def test_branch_table(self) -> None:
        passing = [assessment(arm_id) for arm_id in SELECTABLE_ARM_IDS]
        returned = {SELECTABLE_ARM_IDS[0]: 0.77, SELECTABLE_ARM_IDS[1]: 0.771}
        sdlogj = {SELECTABLE_ARM_IDS[0]: 0.30, SELECTABLE_ARM_IDS[1]: 0.31}
        self.assertEqual(select_branch(passing, returned, sdlogj)["branch"], BRANCH_FREEZE)
        self.assertEqual(select_branch(passing, returned, sdlogj)["winner"], SELECTABLE_ARM_IDS[1])

        selector = [replace(row, returned_material=False, promotion_eligible=False) for row in passing]
        self.assertEqual(select_branch(selector, returned, sdlogj)["branch"], BRANCH_SELECTOR)

        unsafe = [replace(row, geometry_passed=False, promotion_eligible=False) for row in passing]
        self.assertEqual(select_branch(unsafe, returned, sdlogj)["branch"], BRANCH_SAFETY)

        explained = [
            replace(row, attribution_passed=False, attribution_mode=None, promotion_eligible=False) for row in passing
        ]
        self.assertEqual(select_branch(explained, returned, sdlogj)["branch"], BRANCH_MATCHED_CONTROL)

        absent = [
            replace(
                row, capacity_material=False, attribution_passed=False, attribution_mode=None, promotion_eligible=False
            )
            for row in passing
        ]
        self.assertEqual(select_branch(absent, returned, sdlogj)["branch"], BRANCH_CLOSE)
        self.assertEqual(select_branch(absent[:-1], returned, sdlogj)["branch"], BRANCH_INVALID)


if __name__ == "__main__":
    unittest.main()
