from __future__ import annotations

import copy
import json
import math
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from typing import ClassVar

import numpy as np
import torch

import tools.analysis.search_gate_c6 as search_gate_c6
from tools.analysis.run_artifacts import atomic_write_json, sha256_file
from tools.analysis.run_search_gate_c6 import (
    DIRECTION_DIAGNOSTIC_FIELDS,
    _assert_decision_label_free,
    _direction_diagnostics,
    _load_barrier,
    _stage_initial_margin_preflight,
    build_decision_barrier,
    build_parser,
    run_decision_case,
    run_evaluation_case,
)
from tools.analysis.search_gate_c3 import (
    NCC7_WINDOW,
    NCC_DENOMINATOR_EPS,
    PRIMARY_NCC_IMPROVEMENT_MIN,
    SUPPORT_RETENTION_MIN,
)
from tools.analysis.search_gate_c6 import (
    ALL_LABEL_CI_LOW_VS_C4_MIN_STRICT,
    ARM_SPECS,
    BRANCH_CONTROLS_EXPLAIN,
    BRANCH_DROP_QUARTER,
    BRANCH_FREEZE,
    BRANCH_INVALID,
    BRANCH_LEARNED,
    BRANCH_NARROW_SAFETY,
    BRANCH_SIMPLIFY_FUSION,
    C6_POLICY_SHA256,
    CAPACITY_CI_LOW_VS_C4_MIN_STRICT,
    CAPACITY_MEAN_VS_C4_MIN,
    CAUSAL_CI_LOW_VS_CONTROL_MIN_STRICT,
    CAUSAL_MEAN_VS_CONTROL_MIN,
    CONTROL_ARM_IDS,
    DIAGNOSTIC_ARM_IDS,
    EXACT_CLAIM_EPS,
    EXPECTED_CASE_COUNT,
    FROZEN_CONSTRUCTION,
    PUBLISHED_CONSTRUCTION_KEYS,
    RETURNED_CI_LOW_VS_C4_MIN_STRICT,
    RETURNED_MEAN_VS_C4_MIN,
    RISK_LABEL_CI_LOW_VS_C4_MIN_STRICT,
    RISK_LABEL_IDS,
    SDLOGJ_CI_HIGH_VS_C4_MAX,
    SELECTABLE_ARM_IDS,
    WORK_EPS,
    ArmAssessment,
    PairedSummary,
    arm_ids_for_role,
    assert_frozen_policy,
    assess_arm,
    frozen_construction_kwargs,
    matched_control_ids,
    policy_dict,
    policy_sha256,
    select_branch,
    simultaneous_paired_summaries,
    stage_work_eps_schedule,
)
from tools.analysis.search_gate_pyramid import (
    array_sha256,
    binomial_blur3d,
    blurred_full_resolution_image,
    build_pyramid_direction,
    downsample_image,
    lift_level_vector,
    project_psi_to_level,
)
from utils import dice_per_label
from utils.field import boundary_nonzero_count, identity_collar

LABEL_IDS = tuple(range(1, 31))
PLAIN_LABEL_IDS = tuple(label for label in LABEL_IDS if label not in RISK_LABEL_IDS)
SELECTABLE_BY_ID = {row.arm_id: row.selectable for row in ARM_SPECS}


def summary(mean: float, ci_low: float, ci_high: float | None = None) -> PairedSummary:
    """A PairedSummary whose guard-relevant fields are set exactly, without resampling."""

    return PairedSummary(
        family_id="synthetic",
        contrast_id="synthetic",
        n=EXPECTED_CASE_COUNT,
        mean=mean,
        median=mean,
        ci_low=ci_low,
        ci_high=mean if ci_high is None else ci_high,
        improved=EXPECTED_CASE_COUNT,
        worsened=0,
        tied=0,
        simultaneous_family_size=1,
    )


def above(value: float) -> float:
    return math.nextafter(value, math.inf)


def below(value: float) -> float:
    return math.nextafter(value, -math.inf)


def passing_regional(labels=LABEL_IDS) -> list[tuple[int, PairedSummary]]:
    return [(label, summary(0.0, above(RISK_LABEL_CI_LOW_VS_C4_MIN_STRICT))) for label in labels]


def passing_inputs() -> dict:
    return {
        "capacity_vs_c4": summary(CAPACITY_MEAN_VS_C4_MIN, above(CAPACITY_CI_LOW_VS_C4_MIN_STRICT)),
        "causal_vs_full": summary(CAUSAL_MEAN_VS_CONTROL_MIN, above(CAUSAL_CI_LOW_VS_CONTROL_MIN_STRICT)),
        "causal_vs_blur": summary(CAUSAL_MEAN_VS_CONTROL_MIN, above(CAUSAL_CI_LOW_VS_CONTROL_MIN_STRICT)),
        "returned_vs_c4": summary(RETURNED_MEAN_VS_C4_MIN, above(RETURNED_CI_LOW_VS_C4_MIN_STRICT)),
        "sdlogj_vs_c4": summary(-0.001, -0.002, ci_high=SDLOGJ_CI_HIGH_VS_C4_MAX),
        "regional_vs_c4": passing_regional(),
        "folds_all_zero": True,
    }


def assess(**overrides) -> ArmAssessment:
    inputs = passing_inputs()
    inputs.update(overrides)
    return assess_arm(SELECTABLE_ARM_IDS[0], **inputs)


def eligible_assessments(**overrides) -> list[ArmAssessment]:
    template = ArmAssessment(
        arm_id="",
        capacity_material=True,
        causal_full_passed=True,
        causal_blur_passed=True,
        returned_material=True,
        geometry_passed=True,
        regional_passed=True,
        promotion_eligible=True,
    )
    return [replace(template, arm_id=arm_id, **overrides) for arm_id in SELECTABLE_ARM_IDS]


def capacity_means(**overrides) -> dict[str, float]:
    means = dict.fromkeys(SELECTABLE_ARM_IDS, 0.003)
    means.update(overrides)
    return means


class AssessArmGuardTest(unittest.TestCase):
    def test_every_guard_passes_exactly_at_its_frozen_boundary(self) -> None:
        result = assess()
        self.assertTrue(result.promotion_eligible)
        self.assertEqual(
            (
                result.capacity_material,
                result.causal_full_passed,
                result.causal_blur_passed,
                result.returned_material,
                result.geometry_passed,
                result.regional_passed,
            ),
            (True, True, True, True, True, True),
        )

    def test_capacity_needs_both_the_mean_and_a_strictly_higher_lower_bound(self) -> None:
        weak_mean = summary(below(CAPACITY_MEAN_VS_C4_MIN), above(CAPACITY_CI_LOW_VS_C4_MIN_STRICT))
        touching_bound = summary(CAPACITY_MEAN_VS_C4_MIN, CAPACITY_CI_LOW_VS_C4_MIN_STRICT)
        for candidate in (weak_mean, touching_bound):
            result = assess(capacity_vs_c4=candidate)
            self.assertFalse(result.capacity_material)
            self.assertFalse(result.promotion_eligible)

    def test_each_causal_control_is_required_on_its_own(self) -> None:
        failing = summary(CAUSAL_MEAN_VS_CONTROL_MIN, CAUSAL_CI_LOW_VS_CONTROL_MIN_STRICT)
        against_full = assess(causal_vs_full=failing)
        self.assertFalse(against_full.causal_full_passed)
        self.assertTrue(against_full.causal_blur_passed)
        self.assertFalse(against_full.promotion_eligible)
        against_blur = assess(causal_vs_blur=failing)
        self.assertTrue(against_blur.causal_full_passed)
        self.assertFalse(against_blur.causal_blur_passed)
        self.assertFalse(against_blur.promotion_eligible)

    def test_returned_policy_is_required_even_when_capacity_holds(self) -> None:
        result = assess(returned_vs_c4=summary(below(RETURNED_MEAN_VS_C4_MIN), 0.001))
        self.assertTrue(result.capacity_material)
        self.assertFalse(result.returned_material)
        self.assertFalse(result.promotion_eligible)

    def test_sdlogj_bounds_the_increase_and_never_demands_one(self) -> None:
        """Lower SDlogJ is better, so the guard caps ci_high and must accept a large improvement."""

        improved = assess(sdlogj_vs_c4=summary(-0.05, -0.06, ci_high=-0.04))
        self.assertTrue(improved.geometry_passed)
        at_bound = assess(sdlogj_vs_c4=summary(0.0, -0.01, ci_high=SDLOGJ_CI_HIGH_VS_C4_MAX))
        self.assertTrue(at_bound.geometry_passed)
        worsened = assess(sdlogj_vs_c4=summary(0.0, -0.01, ci_high=above(SDLOGJ_CI_HIGH_VS_C4_MAX)))
        self.assertFalse(worsened.geometry_passed)
        self.assertFalse(worsened.promotion_eligible)

    def test_geometry_also_requires_the_zero_corner_fold_contract(self) -> None:
        self.assertFalse(assess(folds_all_zero=False).geometry_passed)

    def test_regional_guard_reads_the_lower_bound_not_the_upper_one(self) -> None:
        harmed = [(label, summary(-0.02, -0.05, ci_high=0.01)) for label in LABEL_IDS]
        result = assess(regional_vs_c4=harmed)
        self.assertFalse(result.regional_passed)
        self.assertFalse(result.promotion_eligible)

    def test_risk_labels_use_the_tighter_bound_than_every_other_label(self) -> None:
        between = -0.003
        self.assertLess(between, RISK_LABEL_CI_LOW_VS_C4_MIN_STRICT)
        self.assertGreater(between, ALL_LABEL_CI_LOW_VS_C4_MIN_STRICT)
        plain_only = [(label, summary(0.0, between)) for label in PLAIN_LABEL_IDS]
        plain_only += [(label, summary(0.0, above(RISK_LABEL_CI_LOW_VS_C4_MIN_STRICT))) for label in RISK_LABEL_IDS]
        self.assertTrue(assess(regional_vs_c4=plain_only).regional_passed)
        for risk_label in RISK_LABEL_IDS:
            mixed = [
                (label, summary(0.0, between if label == risk_label else above(RISK_LABEL_CI_LOW_VS_C4_MIN_STRICT)))
                for label in LABEL_IDS
            ]
            self.assertFalse(assess(regional_vs_c4=mixed).regional_passed, risk_label)

    def test_a_single_harmed_plain_label_still_fails_the_global_safeguard(self) -> None:
        mixed = [
            (label, summary(0.0, below(ALL_LABEL_CI_LOW_VS_C4_MIN_STRICT) if label == 5 else 0.0))
            for label in LABEL_IDS
        ]
        self.assertFalse(assess(regional_vs_c4=mixed).regional_passed)

    def test_only_selectable_arms_can_be_assessed(self) -> None:
        for arm_id in (*CONTROL_ARM_IDS, *DIAGNOSTIC_ARM_IDS, "c4_intensity_s2", "unknown"):
            with self.assertRaises(ValueError):
                assess_arm(arm_id, **passing_inputs())


class SelectBranchTest(unittest.TestCase):
    def test_freeze_when_a_quarter_schedule_arm_passes_everything(self) -> None:
        result = select_branch(eligible_assessments(), capacity_means(pyr421_a100=0.01), no_rewarp_vs_rewarp=None)
        self.assertEqual(result["branch"], BRANCH_FREEZE)
        self.assertEqual(result["winner"], "pyr421_a100")

    def test_drop_quarter_when_the_half_schedule_arm_wins(self) -> None:
        result = select_branch(eligible_assessments(), capacity_means(pyr21_a150=0.01), no_rewarp_vs_rewarp=None)
        self.assertEqual(result["branch"], BRANCH_DROP_QUARTER)
        self.assertEqual(result["winner"], "pyr21_a150")

    def test_simplify_fusion_only_for_the_quarter_arm_with_a_noninferior_no_rewarp(self) -> None:
        noninferior = summary(0.0, -CAUSAL_MEAN_VS_CONTROL_MIN)
        inferior = summary(-0.01, below(-CAUSAL_MEAN_VS_CONTROL_MIN))
        winner_means = capacity_means(pyr421_a100=0.01)
        simplified = select_branch(eligible_assessments(), winner_means, no_rewarp_vs_rewarp=noninferior)
        self.assertEqual(simplified["branch"], BRANCH_SIMPLIFY_FUSION)
        self.assertEqual(simplified["winner"], "pyr421_a100")
        kept = select_branch(eligible_assessments(), winner_means, no_rewarp_vs_rewarp=inferior)
        self.assertEqual(kept["branch"], BRANCH_FREEZE)
        other_winner = select_branch(
            eligible_assessments(), capacity_means(pyr421_a150=0.01), no_rewarp_vs_rewarp=noninferior
        )
        self.assertEqual(other_winner["branch"], BRANCH_FREEZE)

    def test_narrow_safety_when_causal_capacity_survives_but_a_safeguard_fails(self) -> None:
        blocked = eligible_assessments(promotion_eligible=False, regional_passed=False)
        result = select_branch(blocked, capacity_means(), no_rewarp_vs_rewarp=None)
        self.assertEqual(result["branch"], BRANCH_NARROW_SAFETY)
        self.assertIsNone(result["winner"])

    def test_controls_explain_when_capacity_exists_without_causal_superiority(self) -> None:
        blocked = eligible_assessments(promotion_eligible=False, causal_full_passed=False, causal_blur_passed=False)
        result = select_branch(blocked, capacity_means(), no_rewarp_vs_rewarp=None)
        self.assertEqual(result["branch"], BRANCH_CONTROLS_EXPLAIN)
        self.assertIsNone(result["winner"])

    def test_learned_branch_when_no_arm_has_material_capacity(self) -> None:
        blocked = eligible_assessments(promotion_eligible=False, capacity_material=False)
        result = select_branch(blocked, capacity_means(), no_rewarp_vs_rewarp=None)
        self.assertEqual(result["branch"], BRANCH_LEARNED)
        self.assertIsNone(result["winner"])

    def test_invalid_on_failed_integrity_or_a_broken_arm_inventory(self) -> None:
        complete = eligible_assessments()
        means = capacity_means()
        cases = {
            "integrity": (complete, means, False),
            "missing": (complete[:-1], means, True),
            "duplicated": ([*complete, complete[0]], means, True),
            "foreign": ([*complete[:-1], replace(complete[-1], arm_id="blur21_a100")], means, True),
        }
        for name, (assessments, capacity, integrity) in cases.items():
            result = select_branch(assessments, capacity, no_rewarp_vs_rewarp=None, integrity_passed=integrity)
            self.assertEqual(result["branch"], BRANCH_INVALID, name)
            self.assertIsNone(result["winner"], name)

    def test_the_winner_is_taken_only_from_promotion_eligible_arms(self) -> None:
        assessments = eligible_assessments()
        assessments[2] = replace(assessments[2], promotion_eligible=False)
        result = select_branch(assessments, capacity_means(pyr421_a100=0.09, pyr21_a100=0.02), no_rewarp_vs_rewarp=None)
        self.assertEqual(result["winner"], "pyr21_a100")

    def test_the_no_rewarp_diagnostic_can_never_be_promoted(self) -> None:
        diagnostic = DIAGNOSTIC_ARM_IDS[0]
        self.assertNotIn(diagnostic, SELECTABLE_ARM_IDS)
        with self.assertRaises(ValueError):
            matched_control_ids(diagnostic)
        with self.assertRaises(ValueError):
            assess_arm(diagnostic, **passing_inputs())
        smuggled = [*eligible_assessments()[:-1], replace(eligible_assessments()[0], arm_id=diagnostic)]
        result = select_branch(smuggled, {**capacity_means(), diagnostic: 0.5}, no_rewarp_vs_rewarp=None)
        self.assertEqual(result["branch"], BRANCH_INVALID)


class FrozenContractTest(unittest.TestCase):
    def test_policy_and_arm_inventory_are_frozen(self) -> None:
        assert_frozen_policy()
        self.assertEqual(policy_sha256(), C6_POLICY_SHA256)
        self.assertEqual(len(ARM_SPECS), 14)
        self.assertEqual(len({row.arm_id for row in ARM_SPECS}), 14)
        self.assertEqual(tuple(row.arm_id for row in ARM_SPECS if row.selectable), SELECTABLE_ARM_IDS)

    def test_control_and_diagnostic_inventories_match_arm_specs(self) -> None:
        self.assertEqual(arm_ids_for_role("MATCHED_CONTROL"), CONTROL_ARM_IDS)
        self.assertEqual(arm_ids_for_role("DIAGNOSTIC"), DIAGNOSTIC_ARM_IDS)
        self.assertEqual(arm_ids_for_role("SELECTABLE"), SELECTABLE_ARM_IDS)
        self.assertEqual(arm_ids_for_role("FROZEN_REFERENCE"), ("c4_intensity_s2",))
        self.assertEqual(
            len(ARM_SPECS),
            len(CONTROL_ARM_IDS) + len(DIAGNOSTIC_ARM_IDS) + len(SELECTABLE_ARM_IDS) + 1,
        )
        for arm_id in DIAGNOSTIC_ARM_IDS:
            self.assertFalse(SELECTABLE_BY_ID[arm_id])
        for arm_id in CONTROL_ARM_IDS:
            self.assertFalse(SELECTABLE_BY_ID[arm_id])

    def test_published_construction_covers_every_field_changing_setting(self) -> None:
        construction = policy_dict()["construction"]
        required = {
            "standardization_floor",
            "image_normalization_std_floor",
            "proposal_multiplier",
            "post_smoothing_passes",
            "posterior_temperature",
            "centre_beta",
            "require_all_candidates_valid",
            "work_eps",
            "stage_work_eps_decrement",
            "stage_work_eps_by_depth",
            "exact_claim_eps",
            "stage_local_clip_sweeps",
            "final_local_clip_sweeps",
            "primary_ncc_window",
            "ncc_denominator_eps",
            "support_retention_min",
            "primary_ncc_improvement_min",
        }
        self.assertTrue(required.issubset(construction))

    def test_published_construction_equals_what_the_builder_receives(self) -> None:
        construction = policy_dict()["construction"]
        kwargs = frozen_construction_kwargs()
        self.assertEqual(set(kwargs), set(FROZEN_CONSTRUCTION))
        for builder_key, policy_key in PUBLISHED_CONSTRUCTION_KEYS.items():
            self.assertEqual(kwargs[builder_key], construction[policy_key], builder_key)
        self.assertEqual(set(PUBLISHED_CONSTRUCTION_KEYS), set(FROZEN_CONSTRUCTION))

    def test_borrowed_ncc_values_track_their_search_gate_c3_owner(self) -> None:
        construction = policy_dict()["construction"]
        self.assertEqual(construction["primary_ncc_window"], NCC7_WINDOW)
        self.assertEqual(construction["support_retention_min"], SUPPORT_RETENTION_MIN)
        self.assertEqual(construction["primary_ncc_improvement_min"], PRIMARY_NCC_IMPROVEMENT_MIN)
        self.assertEqual(construction["ncc_denominator_eps"], NCC_DENOMINATOR_EPS)
        self.assertFalse(hasattr(search_gate_c6, "PRIMARY_NCC_WINDOW"), "C6 must not shadow the NCC window owner")

    def test_stage_work_margins_descend_but_remain_above_the_exact_claim(self) -> None:
        self.assertEqual(stage_work_eps_schedule(2), (0.0011, 0.001075))
        self.assertEqual(stage_work_eps_schedule(3), (0.0011, 0.001075, 0.00105))
        self.assertGreater(stage_work_eps_schedule(3)[-1], EXACT_CLAIM_EPS)
        self.assertEqual(
            policy_dict()["construction"]["stage_work_eps_by_depth"]["3"], list(stage_work_eps_schedule(3))
        )
        failed_v1_continuation_bound = 0.001098990539567836
        self.assertLess(failed_v1_continuation_bound, stage_work_eps_schedule(3)[0])
        self.assertGreater(failed_v1_continuation_bound, stage_work_eps_schedule(3)[1])

    def test_initial_margin_preflight_covers_all_cases_and_fails_closed(self) -> None:
        case_ids = [f"subject_{index:03d}" for index in range(EXPECTED_CASE_COUNT)]
        snapshot = {
            "case_ids": case_ids,
            "source_initial": {
                case_id: {"exact": {"certified": True, "interval_lo_min": WORK_EPS + (index + 1) * 1e-9}}
                for index, case_id in enumerate(case_ids)
            },
        }
        profile = _stage_initial_margin_preflight(snapshot)
        self.assertEqual(profile["n_cases"], EXPECTED_CASE_COUNT)
        self.assertEqual(profile["below_required_count"], 0)
        self.assertEqual(profile["minimum_case_id"], case_ids[0])
        broken = copy.deepcopy(snapshot)
        broken["source_initial"][case_ids[-1]]["exact"]["interval_lo_min"] = WORK_EPS - 1e-9
        with self.assertRaisesRegex(RuntimeError, "do not support first-stage"):
            _stage_initial_margin_preflight(broken)

    def test_assert_frozen_policy_rejects_a_drifted_ncc_window(self) -> None:
        original = search_gate_c6.NCC7_WINDOW
        search_gate_c6.NCC7_WINDOW = original + 2
        try:
            with self.assertRaises(RuntimeError):
                assert_frozen_policy()
        finally:
            search_gate_c6.NCC7_WINDOW = original
        assert_frozen_policy()

    def test_assert_frozen_policy_rejects_a_drifted_inventory(self) -> None:
        original = search_gate_c6.CONTROL_ARM_IDS
        search_gate_c6.CONTROL_ARM_IDS = original[:-1]
        try:
            with self.assertRaises(RuntimeError):
                assert_frozen_policy()
        finally:
            search_gate_c6.CONTROL_ARM_IDS = original
        assert_frozen_policy()

    def test_each_selectable_has_schedule_and_amplitude_matched_controls(self) -> None:
        expected = {
            "pyr21_a100": ("full21_a100", "blur21_a100"),
            "pyr21_a150": ("full21_a150", "blur21_a150"),
            "pyr421_a100": ("full421_a100", "blur421_a100"),
            "pyr421_a150": ("full421_a150", "blur421_a150"),
        }
        self.assertEqual({key: matched_control_ids(key) for key in SELECTABLE_ARM_IDS}, expected)

    def test_simultaneous_interval_is_family_wide_and_deterministic(self) -> None:
        rows = {
            "a": np.linspace(-0.001, 0.003, EXPECTED_CASE_COUNT),
            "b": np.linspace(0.0, 0.004, EXPECTED_CASE_COUNT),
        }
        left = simultaneous_paired_summaries("synthetic", rows)
        right = simultaneous_paired_summaries("synthetic", rows)
        self.assertEqual(left, right)
        self.assertEqual(left["a"].simultaneous_family_size, 2)
        self.assertLess(left["a"].ci_low, left["a"].mean)
        self.assertGreater(left["a"].ci_high, left["a"].mean)

    def test_decision_contract_rejects_evaluation_data(self) -> None:
        safe = {"labels_loaded_to_device": False, "field": "cache.npy", "arms": []}
        _assert_decision_label_free(safe)
        for injected in (
            {"dice": 0.8},
            {"label_ids": [1, 2]},
            {"path": "/tmp/validation.pkl"},
            {"segmentation_path": "x"},
        ):
            payload = copy.deepcopy(safe)
            payload["injected"] = injected
            with self.assertRaises(RuntimeError):
                _assert_decision_label_free(payload)

    def test_decision_contract_allows_only_the_exact_frozen_policy_metadata(self) -> None:
        safe = {
            "policy": policy_dict(),
            "policy_sha256": C6_POLICY_SHA256,
            "labels_loaded_to_device": False,
            "labels_loaded": False,
        }
        _assert_decision_label_free(safe)
        _assert_decision_label_free(json.loads(json.dumps(safe)))

        tampered = copy.deepcopy(safe)
        tampered["policy"]["injected_label_values"] = [1, 2]
        with self.assertRaisesRegex(RuntimeError, "altered frozen policy"):
            _assert_decision_label_free(tampered)

    def test_cli_exposes_every_runner_stage(self) -> None:
        parser = build_parser()
        actions = next(action for action in parser._actions if action.dest == "action").choices
        self.assertEqual(
            set(actions),
            {
                "selfcheck",
                "prepare",
                "decision-pilot",
                "decision-worker",
                "decision-barrier",
                "freeze-evaluation",
                "evaluation-worker",
                "finalize",
            },
        )


def decision_row(arm_id: str = "pyr421_a100", **overrides) -> dict:
    row = {
        "arm_id": arm_id,
        "direction": {
            "reference_rms": 0.2,
            "pre_normalization_rms": 0.125,
            "normalized_rms": 0.2,
            "stages": [{"clip_retention": 0.9}, {"clip_retention": 0.5}, {"clip_retention": 0.7}],
        },
        "operator": {"retained_norm_ratio": 0.42},
    }
    row.update(overrides)
    return row


class DirectionDiagnosticsTest(unittest.TestCase):
    """These run only at the end of a full H100 attempt, so they are pinned here instead."""

    def test_gain_and_retention_are_derived_from_the_stored_direction(self) -> None:
        values = _direction_diagnostics(decision_row(), "case/arm")
        self.assertEqual(set(values), set(DIRECTION_DIAGNOSTIC_FIELDS))
        self.assertAlmostEqual(values["pre_normalization_rms"], 0.125)
        self.assertAlmostEqual(values["rematch_gain"], 1.6)
        self.assertAlmostEqual(values["normalized_rms"], 0.2)
        self.assertAlmostEqual(values["stage_clip_retention_min"], 0.5)
        self.assertAlmostEqual(values["stage_clip_retention_mean"], 0.7)
        self.assertAlmostEqual(values["final_clip_retained_norm_ratio"], 0.42)

    def test_the_frozen_c4_reference_reports_explicit_absence(self) -> None:
        values = _direction_diagnostics({"arm_id": "c4_intensity_s2"}, "case/reference")
        self.assertEqual(values, dict.fromkeys(DIRECTION_DIAGNOSTIC_FIELDS))

    def test_a_degenerate_or_missing_direction_fails_closed(self) -> None:
        for broken in (
            decision_row(direction={**decision_row()["direction"], "pre_normalization_rms": 0.0}),
            decision_row(direction={**decision_row()["direction"], "reference_rms": -1.0}),
            decision_row(direction={**decision_row()["direction"], "normalized_rms": float("nan")}),
            decision_row(direction={**decision_row()["direction"], "stages": []}),
            decision_row(operator={"retained_norm_ratio": None}),
        ):
            with self.assertRaises(RuntimeError):
                _direction_diagnostics(broken, "case/arm")


class PyramidMathTest(unittest.TestCase):
    def test_constant_image_survives_filter_and_decimation(self) -> None:
        image = torch.full((1, 1, 32, 40, 48), 3.25)
        before = image.clone()
        filtered = binomial_blur3d(image, passes=2)
        coarse = downsample_image(image, 4)
        self.assertTrue(torch.equal(image, before))
        self.assertTrue(torch.allclose(filtered, image, atol=0.0, rtol=0.0))
        self.assertEqual(tuple(coarse.shape), (1, 1, 8, 10, 12))
        self.assertTrue(torch.allclose(coarse, torch.full_like(coarse, 3.25), atol=0.0, rtol=0.0))

    def test_quarter_level_is_two_recursive_half_steps(self) -> None:
        generator = torch.Generator().manual_seed(3)
        image = torch.rand((1, 1, 32, 40, 48), generator=generator)
        direct_api = downsample_image(image, 4)
        explicit_recursive = downsample_image(downsample_image(image, 2), 2)
        self.assertTrue(torch.equal(direct_api, explicit_recursive))

    def test_quarter_projection_is_two_recursive_half_steps_on_a_non_zero_field(self) -> None:
        """A zero or constant field cannot separate a recursive projection from a single 1/4 resample."""

        generator = torch.Generator().manual_seed(17)
        field = torch.rand((1, 3, 32, 32, 32), generator=generator)
        direct_api = project_psi_to_level(field, 4)
        explicit_recursive = project_psi_to_level(project_psi_to_level(field, 2), 2)
        single_step = (
            torch.nn.functional.interpolate(field, size=(8, 8, 8), mode="trilinear", align_corners=False) / 4.0
        )
        self.assertTrue(torch.equal(direct_api, explicit_recursive))
        self.assertFalse(torch.allclose(direct_api, single_step, atol=1e-6))

    def test_constant_voxel_displacement_scales_on_projection_and_lift(self) -> None:
        field = torch.zeros((1, 3, 32, 40, 48))
        field[:, 0] = 4.0
        field[:, 1] = -2.0
        field[:, 2] = 1.0
        coarse = project_psi_to_level(field, 4)
        self.assertTrue(torch.allclose(coarse[:, 0], torch.ones_like(coarse[:, 0]), atol=1e-6))
        self.assertTrue(torch.allclose(coarse[:, 1], torch.full_like(coarse[:, 1], -0.5), atol=1e-6))
        lifted = lift_level_vector(coarse, field.shape[-3:], 4)
        self.assertTrue(torch.allclose(lifted, field, atol=1e-6))

    def test_projection_and_lift_preserve_a_linear_ramp_in_the_interior(self) -> None:
        """A constant field is blind to a half-voxel phase error; a ramp is not."""

        size = 32
        field = torch.zeros((1, 3, size, size, size), dtype=torch.float64)
        ramp = torch.arange(size, dtype=torch.float64)
        field[:, 0] = ramp.view(size, 1, 1)
        field[:, 2] = 0.25 * ramp.view(1, 1, size)
        for factor in (2, 4):
            lifted = lift_level_vector(project_psi_to_level(field, factor), field.shape[-3:], factor)
            interior = lifted[:, :, 12:-12, 12:-12, 12:-12] - field[:, :, 12:-12, 12:-12, 12:-12]
            self.assertLess(float(interior.abs().max()), 1e-9, factor)

    def test_blurred_full_grid_never_changes_resolution_or_input(self) -> None:
        generator = torch.Generator().manual_seed(7)
        image = torch.rand((1, 1, 24, 28, 32), generator=generator)
        before = image.clone()
        first = blurred_full_resolution_image(image, 4)
        second = blurred_full_resolution_image(image, 4)
        self.assertEqual(first.shape, image.shape)
        self.assertTrue(torch.equal(first, second))
        self.assertTrue(torch.equal(image, before))

    def test_quarter_blur_control_is_dilation_one_then_dilation_two(self) -> None:
        """The second control pass must match the half-grid spacing of the second pyramid level."""

        size = 33
        impulse = torch.zeros((1, 1, size, size, size), dtype=torch.float64)
        impulse[0, 0, size // 2, size // 2, size // 2] = 1.0
        control = blurred_full_resolution_image(impulse, 4)
        line = control[0, 0, :, size // 2, size // 2]
        taps = torch.nonzero(line > 0).flatten().tolist()
        self.assertEqual(min(taps), size // 2 - 6)
        self.assertEqual(max(taps), size // 2 + 6)
        self.assertTrue(torch.equal(blurred_full_resolution_image(impulse, 2), binomial_blur3d(impulse, passes=1)))
        self.assertFalse(torch.equal(control, binomial_blur3d(impulse, passes=2)))


class PyramidDirectionTest(unittest.TestCase):
    directions: ClassVar[dict[tuple, object]] = {}

    @classmethod
    def setUpClass(cls) -> None:
        size = 32
        generator = torch.Generator().manual_seed(11)
        cls.fixed = torch.rand((1, 1, size, size, size), generator=generator)
        cls.moving = torch.roll(cls.fixed, shifts=1, dims=-1)
        cls.initial = torch.zeros((1, 3, size, size, size))
        reference = torch.zeros_like(cls.initial)
        reference[:, 2] = 0.2
        cls.reference = identity_collar(reference, width=7)
        for spec in ARM_SPECS[1:]:
            key = (spec.family, spec.factors, spec.rewarp_between_levels)
            if key not in cls.directions:
                cls.directions[key] = build_pyramid_direction(
                    cls.fixed,
                    cls.moving,
                    cls.initial,
                    cls.reference,
                    family=spec.family,
                    factors=spec.factors,
                    rewarp_between_levels=spec.rewarp_between_levels,
                    **frozen_construction_kwargs(),
                )

    def test_every_frozen_configuration_builds(self) -> None:
        self.assertEqual(len(self.directions), 7)
        for key, direction in self.directions.items():
            self.assertTrue(torch.isfinite(direction.displacement).all(), key)
            self.assertEqual(boundary_nonzero_count(direction.displacement), 0, key)
            self.assertEqual(len(direction.stages), len(direction.factors), key)

    def test_every_arm_reaches_the_same_final_rms_budget(self) -> None:
        for key, direction in self.directions.items():
            self.assertAlmostEqual(direction.normalized_rms, direction.reference_rms, places=6, msg=str(key))

    def test_true_pyramid_searches_at_stride_one_and_controls_at_the_matching_factor(self) -> None:
        for (family, factors, _), direction in self.directions.items():
            strides = tuple(stage.stride_voxels for stage in direction.stages)
            shapes = tuple(stage.level_shape for stage in direction.stages)
            if family == "true_pyramid":
                self.assertEqual(strides, (1,) * len(factors), family)
                self.assertEqual(shapes, tuple((32 // factor,) * 3 for factor in factors), family)
            else:
                self.assertEqual(strides, tuple(factors), family)
                self.assertEqual(shapes, ((32, 32, 32),) * len(factors), family)

    def test_each_stage_requests_an_equal_share_of_the_source_rms(self) -> None:
        """The no-rewarp arm applies no clip, so its realized stage RMS is the requested share itself."""

        for (family, factors, rewarp), direction in self.directions.items():
            share = direction.reference_rms / len(factors)
            for stage in direction.stages:
                self.assertAlmostEqual(stage.requested_stage_rms, share, places=9, msg=str(family))
                self.assertLess(stage.realized_stage_rms, direction.reference_rms, str(family))
                if not rewarp:
                    self.assertAlmostEqual(stage.realized_stage_rms, share, places=5, msg=str(family))

    def test_rewarped_stages_publish_a_safe_descending_continuation_contract(self) -> None:
        for (family, factors, rewarp), direction in self.directions.items():
            if not rewarp:
                for stage in direction.stages:
                    self.assertIsNone(stage.clip_work_eps)
                    self.assertIsNone(stage.continuation_eps)
                    self.assertIsNone(stage.output_fast_cert_bound)
                continue
            schedule = stage_work_eps_schedule(len(factors))
            self.assertEqual(tuple(stage.clip_work_eps for stage in direction.stages), schedule, family)
            self.assertEqual(
                tuple(stage.continuation_eps for stage in direction.stages),
                (*schedule[1:], EXACT_CLAIM_EPS),
                family,
            )
            for stage in direction.stages:
                self.assertGreaterEqual(stage.output_fast_cert_bound, stage.continuation_eps, family)

    def test_rewarp_and_no_rewarp_produce_different_fields(self) -> None:
        rewarped = self.directions[("true_pyramid", (4, 2, 1), True)].displacement
        plain = self.directions[("true_pyramid", (4, 2, 1), False)].displacement
        self.assertFalse(torch.equal(rewarped, plain))
        relative = float((rewarped - plain).norm() / rewarped.norm())
        self.assertGreater(relative, 1e-6)

    def test_a_pyramid_arm_differs_from_both_of_its_matched_controls(self) -> None:
        for factors in ((2, 1), (4, 2, 1)):
            pyramid = self.directions[("true_pyramid", factors, True)].displacement
            for family in ("full_resolution", "blurred_full_resolution"):
                control = self.directions[(family, factors, True)].displacement
                self.assertFalse(torch.equal(pyramid, control), f"{family}{factors}")


class C6DecisionPilotSmokeTest(unittest.TestCase):
    def test_one_case_crosses_the_real_decision_pilot_boundary(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source_c3 = root / "source_c3"
            source_c4 = root / "source_c4"
            target_c6 = root / "target_c6"
            run_root = root / "run"
            for directory in (source_c3, source_c4, target_c6, run_root):
                directory.mkdir()

            size = 32
            case_id = "subject_smoke"
            generator = torch.Generator().manual_seed(29)
            atlas = torch.rand((1, size, size, size), generator=generator).numpy().astype(np.float32)
            fixed = np.roll(atlas, shift=1, axis=-1).copy()
            initial = torch.zeros((1, 3, size, size, size), dtype=torch.float32)
            reference = torch.zeros_like(initial)
            reference[:, 2] = 0.2
            reference = identity_collar(reference, width=7)

            def image_record(name: str, array: np.ndarray) -> dict:
                path = source_c3 / f"{name}.npy"
                np.save(path, array, allow_pickle=False)
                return {
                    "root_id": "source_c3_heavy",
                    "relative_path": path.relative_to(source_c3).as_posix(),
                    "sha256": sha256_file(path),
                    "array_sha256": array_sha256(torch.from_numpy(array)),
                    "shape": list(array.shape),
                }

            def flow_record(root_id: str, base: Path, name: str, field: torch.Tensor) -> dict:
                path = base / f"{name}.npz"
                np.savez_compressed(path, flow=field.numpy())
                return {
                    "root_id": root_id,
                    "relative_path": path.relative_to(base).as_posix(),
                    "npz_sha256": sha256_file(path),
                    "array_sha256": array_sha256(field),
                }

            decision = {
                "roots": {
                    "source_c3_heavy": str(source_c3),
                    "source_c4_heavy": str(source_c4),
                    "target_c6_heavy": str(target_c6),
                },
                "shards": {"0": [case_id]},
                "case_ids": [case_id],
                "shard_to_physical_gpu": {"0": "0"},
                "image_inputs": {"atlas": image_record("atlas", atlas), case_id: image_record(case_id, fixed)},
                "source_initial": {case_id: {"field": flow_record("source_c3_heavy", source_c3, "initial", initial)}},
                "source_historical": {
                    case_id: {
                        "raw_conf_requested_field": flow_record("source_c3_heavy", source_c3, "historical", reference)
                    }
                },
                "source_c4_anchors": {
                    case_id: {"intensity_s2": {"field": flow_record("source_c4_heavy", source_c4, "anchor", initial)}}
                },
            }
            decision_contract = run_root / "decision_contract.json"
            atomic_write_json(decision_contract, decision)
            decision_sha = sha256_file(decision_contract)
            marker = run_decision_case(
                case_id=case_id,
                shard_index=0,
                physical_gpu="0",
                run_root=run_root,
                decision=decision,
                decision_sha256=decision_sha,
                device=torch.device("cpu"),
                execution={"synthetic_smoke": True},
            )
            payload = json.loads(marker.read_text(encoding="utf-8"))
            self.assertEqual(payload["status"], "COMPLETE")
            self.assertEqual(len(payload["arms"]), len(ARM_SPECS))
            self.assertFalse(payload["labels_loaded_to_device"])
            self.assertTrue(all(row["exact"]["certified"] for row in payload["arms"][1:]))

            barrier_sha = build_decision_barrier(run_root, decision, decision_sha, "synthetic_attempt")
            barrier = _load_barrier(run_root, barrier_sha, decision_sha)
            moving_seg = torch.ones((1, size, size, size), dtype=torch.long)
            fixed_seg = moving_seg.clone()
            identity_dice = float(dice_per_label(moving_seg.unsqueeze(0), fixed_seg.unsqueeze(0), [1]).mean())
            evaluation = {
                "evaluation_label_ids": [1],
                "evaluation_baseline_dice": {case_id: identity_dice},
                "evaluation_c4_anchor_dice": {
                    case_id: {
                        "intensity_s2": {
                            "aggregate_dice": identity_dice,
                            "per_label": [{"label": 1, "dice": identity_dice}],
                        }
                    }
                },
            }
            evaluation_marker = run_evaluation_case(
                case_id=case_id,
                dataset_item=(torch.from_numpy(atlas), torch.from_numpy(fixed), moving_seg, fixed_seg),
                labels=[1],
                run_root=run_root,
                decision=decision,
                decision_sha=decision_sha,
                barrier=barrier,
                barrier_sha=barrier_sha,
                evaluation=evaluation,
                evaluation_sha="b" * 64,
                device=torch.device("cpu"),
                execution={"synthetic_smoke": True},
            )
            evaluated = json.loads(evaluation_marker.read_text(encoding="utf-8"))
            self.assertEqual(evaluated["status"], "COMPLETE")
            self.assertTrue(evaluated["labels_loaded_after_barrier"])
            self.assertEqual(len(evaluated["arms"]), len(ARM_SPECS))


if __name__ == "__main__":
    unittest.main()
