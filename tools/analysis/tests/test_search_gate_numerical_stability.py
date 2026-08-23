from __future__ import annotations

import unittest

import torch

from tools.analysis.search_gate_cost_volume import (
    RawMindCostVolume,
    decode_posterior,
    posterior_from_logits,
    standardize_candidate_costs,
)
from tools.analysis.search_gate_numerical_stability import (
    DECODER_LEGACY,
    DECODER_VECTORIZED,
    FACTORIAL_EDGES,
    FACTORIAL_SPECS,
    MOMENT_CENTERED_FP32,
    MOMENT_CENTERED_FP64,
    MOMENT_LEGACY,
    MOMENT_VECTORIZED,
    NUMERICAL_STABILITY_POLICY,
    NUMERICAL_STABILITY_POLICY_SHA256,
    POSTERIOR_LEGACY,
    POSTERIOR_VECTORIZED,
    SCIENTIFIC_ARMS,
    SENTINEL_ALL_VECTORIZED_GAPS,
    build_reduction_study,
    policy_sha256,
    selfcheck,
)
from tools.analysis.transactional_search import OFFSETS, geometry_mask


class CandidateMomentReductionTest(unittest.TestCase):
    def setUp(self) -> None:
        generator = torch.Generator().manual_seed(7)
        self.costs = 0.1 + 1e-5 * torch.randn((1, len(OFFSETS), 4, 5, 6), generator=generator)
        self.valid = torch.ones_like(self.costs, dtype=torch.bool)

    def reduce(self, mode: str):
        return standardize_candidate_costs(
            self.costs,
            self.valid,
            mode=mode,
            standardization_floor=1e-6,
        )

    def test_centered_fp32_keeps_legacy_mean_and_tracks_fp64_std(self) -> None:
        legacy = self.reduce(MOMENT_LEGACY)
        centered = self.reduce(MOMENT_CENTERED_FP32)
        oracle = self.reduce(MOMENT_CENTERED_FP64)

        torch.testing.assert_close(centered.cost_mean, legacy.cost_mean, atol=0, rtol=0)
        legacy_error = ((legacy.cost_std.double() - oracle.cost_std).abs() / oracle.cost_std).max()
        centered_error = ((centered.cost_std.double() - oracle.cost_std).abs() / oracle.cost_std).max()
        self.assertGreater(int(legacy.floor_hit.sum()), 0)
        self.assertEqual(int(centered.floor_hit.sum()), 0)
        self.assertLess(float(centered_error), 1e-4)
        self.assertLess(float(centered_error), float(legacy_error) * 1e-3)
        self.assertEqual(centered.cost_mean.dtype, torch.float32)
        self.assertEqual(oracle.cost_mean.dtype, torch.float64)
        self.assertEqual(oracle.standardized_costs.dtype, torch.float32)

    def test_vectorized_mode_matches_the_failed_e54_formula(self) -> None:
        result = self.reduce(MOMENT_VECTORIZED)
        count = self.valid.sum(dim=1, keepdim=True).to(self.costs.dtype)
        safe = torch.where(self.valid, self.costs, torch.zeros_like(self.costs))
        mean = safe.sum(dim=1, keepdim=True) / count
        variance = safe.square().sum(dim=1, keepdim=True) / count - mean.square()
        std = variance.clamp_min(0.0).sqrt().clamp_min(1e-6)

        torch.testing.assert_close(result.cost_mean, mean, atol=0, rtol=0)
        torch.testing.assert_close(result.cost_variance_unclamped, variance, atol=0, rtol=0)
        torch.testing.assert_close(result.cost_std, std, atol=0, rtol=0)

    def test_invalid_nonfinite_values_are_ignored_but_valid_ones_fail(self) -> None:
        costs = self.costs.clone()
        valid = self.valid.clone()
        valid[:, 0, 0, 0, 0] = False
        costs[:, 0, 0, 0, 0] = torch.nan
        for mode in (MOMENT_LEGACY, MOMENT_VECTORIZED, MOMENT_CENTERED_FP32, MOMENT_CENTERED_FP64):
            result = standardize_candidate_costs(costs, valid, mode=mode, standardization_floor=1e-6)
            self.assertEqual(float(result.standardized_costs[:, 0, 0, 0, 0]), 0.0)

        valid[:, 0, 0, 0, 0] = True
        for mode in (MOMENT_LEGACY, MOMENT_VECTORIZED, MOMENT_CENTERED_FP32, MOMENT_CENTERED_FP64):
            with self.subTest(mode=mode), self.assertRaisesRegex(ValueError, "valid candidate costs"):
                standardize_candidate_costs(costs, valid, mode=mode, standardization_floor=1e-6)

    def test_valid_nonfinite_values_and_float32_accumulator_overflow_fail_closed(self) -> None:
        for nonfinite in (torch.nan, torch.inf, -torch.inf):
            costs = self.costs.clone()
            costs[:, 0, 0, 0, 0] = nonfinite
            for mode in (MOMENT_LEGACY, MOMENT_VECTORIZED, MOMENT_CENTERED_FP32, MOMENT_CENTERED_FP64):
                with (
                    self.subTest(nonfinite=nonfinite, mode=mode),
                    self.assertRaisesRegex(ValueError, "valid candidate costs"),
                ):
                    standardize_candidate_costs(
                        costs,
                        self.valid,
                        mode=mode,
                        standardization_floor=1e-6,
                    )

        overflowing = torch.full_like(self.costs, torch.finfo(torch.float32).max)
        for mode in (MOMENT_LEGACY, MOMENT_VECTORIZED, MOMENT_CENTERED_FP32):
            with self.subTest(mode=mode), self.assertRaisesRegex(FloatingPointError, "became non-finite"):
                standardize_candidate_costs(
                    overflowing,
                    self.valid,
                    mode=mode,
                    standardization_floor=1e-6,
                )

        oracle = standardize_candidate_costs(
            overflowing,
            self.valid,
            mode=MOMENT_CENTERED_FP64,
            standardization_floor=1e-6,
        )
        self.assertTrue(bool(torch.isfinite(oracle.cost_mean).all()))
        self.assertTrue(bool(torch.isfinite(oracle.standardized_costs).all()))

    def test_constant_costs_use_floor_and_produce_zero_z(self) -> None:
        costs = torch.full((1, len(OFFSETS), 2, 3, 4), 0.25)
        valid = torch.ones_like(costs, dtype=torch.bool)
        for mode in (MOMENT_LEGACY, MOMENT_VECTORIZED, MOMENT_CENTERED_FP32, MOMENT_CENTERED_FP64):
            result = standardize_candidate_costs(costs, valid, mode=mode, standardization_floor=1e-6)
            torch.testing.assert_close(result.cost_std, torch.full_like(result.cost_std, 1e-6), atol=0, rtol=0)
            torch.testing.assert_close(result.standardized_costs, torch.zeros_like(costs), atol=0, rtol=0)
            self.assertTrue(bool(result.floor_hit.all()))

    def test_reductions_do_not_mutate_inputs_and_fail_on_bad_floor(self) -> None:
        costs_snapshot = self.costs.clone()
        valid_snapshot = self.valid.clone()
        for mode in (MOMENT_LEGACY, MOMENT_VECTORIZED, MOMENT_CENTERED_FP32, MOMENT_CENTERED_FP64):
            standardize_candidate_costs(self.costs, self.valid, mode=mode, standardization_floor=1e-6)
        torch.testing.assert_close(self.costs, costs_snapshot, atol=0, rtol=0)
        self.assertTrue(torch.equal(self.valid, valid_snapshot))
        for floor in (0.0, -1.0, float("nan"), float("inf")):
            with self.assertRaisesRegex(ValueError, "finite and positive"):
                standardize_candidate_costs(self.costs, self.valid, mode=MOMENT_LEGACY, standardization_floor=floor)


class DownstreamReductionTest(unittest.TestCase):
    def test_vectorized_posterior_matches_e54_expression(self) -> None:
        generator = torch.Generator().manual_seed(19)
        logits = torch.randn((1, len(OFFSETS), 3, 4, 5), generator=generator)
        valid = torch.rand(logits.shape, generator=generator) > 0.15
        valid[:, :2] = True
        posterior = posterior_from_logits(logits, valid, reduction_mode="vectorized_e54")
        maximum = torch.where(valid, logits, torch.full_like(logits, -torch.inf)).amax(dim=1, keepdim=True)
        weights = torch.where(valid, torch.exp(logits - maximum), torch.zeros_like(logits))
        probabilities = weights / weights.sum(dim=1, keepdim=True)
        log_probabilities = torch.where(
            probabilities > 0,
            probabilities.clamp_min(torch.finfo(logits.dtype).tiny).log(),
            torch.zeros_like(probabilities),
        )
        entropy = -(probabilities * log_probabilities).sum(dim=1, keepdim=True)

        torch.testing.assert_close(posterior.probabilities, probabilities, atol=0, rtol=0)
        torch.testing.assert_close(posterior.entropy, entropy, atol=0, rtol=0)
        self.assertEqual(posterior.reduction_mode, "vectorized_e54")

    def test_einsum_decoder_matches_the_named_e54_expression(self) -> None:
        logits = torch.randn((1, len(OFFSETS), 3, 4, 5), generator=torch.Generator().manual_seed(8))
        valid = torch.ones_like(logits, dtype=torch.bool)
        posterior = posterior_from_logits(logits, valid)
        decoded = decode_posterior(posterior, mode="posterior_mean", reduction_mode="einsum_e54")
        expected = torch.einsum("bkdwh,kc->bcdwh", posterior.probabilities, logits.new_tensor(OFFSETS))
        torch.testing.assert_close(decoded.posterior_mean, expected, atol=0, rtol=0)
        self.assertEqual(decoded.reduction_mode, "einsum_e54")


class FrozenNumericalStabilityProtocolTest(unittest.TestCase):
    def test_factorial_and_scientific_arms_are_identifiable(self) -> None:
        expected_cells = (
            ("F000", MOMENT_LEGACY, POSTERIOR_LEGACY, DECODER_LEGACY),
            ("F001", MOMENT_LEGACY, POSTERIOR_LEGACY, DECODER_VECTORIZED),
            ("F010", MOMENT_LEGACY, POSTERIOR_VECTORIZED, DECODER_LEGACY),
            ("F011", MOMENT_LEGACY, POSTERIOR_VECTORIZED, DECODER_VECTORIZED),
            ("F100", MOMENT_VECTORIZED, POSTERIOR_LEGACY, DECODER_LEGACY),
            ("F101", MOMENT_VECTORIZED, POSTERIOR_LEGACY, DECODER_VECTORIZED),
            ("F110", MOMENT_VECTORIZED, POSTERIOR_VECTORIZED, DECODER_LEGACY),
            ("F111", MOMENT_VECTORIZED, POSTERIOR_VECTORIZED, DECODER_VECTORIZED),
        )
        self.assertEqual(
            tuple(
                (spec.cell_id, spec.moment_reduction, spec.posterior_reduction, spec.decoder_reduction)
                for spec in FACTORIAL_SPECS
            ),
            expected_cells,
        )
        expected_edges = (
            ("moments", "F000", "F100"),
            ("moments", "F001", "F101"),
            ("moments", "F010", "F110"),
            ("moments", "F011", "F111"),
            ("posterior", "F000", "F010"),
            ("posterior", "F001", "F011"),
            ("posterior", "F100", "F110"),
            ("posterior", "F101", "F111"),
            ("decoder", "F000", "F001"),
            ("decoder", "F010", "F011"),
            ("decoder", "F100", "F101"),
            ("decoder", "F110", "F111"),
        )
        self.assertEqual(FACTORIAL_EDGES, expected_edges)
        bit_by_axis = {"moments": 1, "posterior": 2, "decoder": 3}
        for axis, source, target in FACTORIAL_EDGES:
            changed_bits = [index for index in range(1, 4) if source[index] != target[index]]
            self.assertEqual(changed_bits, [bit_by_axis[axis]])
            self.assertEqual(source[bit_by_axis[axis]], "0")
            self.assertEqual(target[bit_by_axis[axis]], "1")

        expected_arms = (
            (
                0,
                "centered_fp32_conf",
                "scientific_candidate",
                MOMENT_CENTERED_FP32,
                "confidence",
                POSTERIOR_LEGACY,
                DECODER_LEGACY,
                "legacy_conf",
                None,
                True,
            ),
            (
                1,
                "centered_fp32_mean_common_rms",
                "scientific_candidate",
                MOMENT_CENTERED_FP32,
                "posterior_mean_common_rms",
                POSTERIOR_LEGACY,
                DECODER_LEGACY,
                "legacy_mean_common_rms",
                "legacy_conf",
                True,
            ),
            (
                2,
                "centered_fp64cast_conf",
                "precision_oracle",
                MOMENT_CENTERED_FP64,
                "confidence",
                POSTERIOR_LEGACY,
                DECODER_LEGACY,
                "legacy_conf",
                None,
                False,
            ),
            (
                3,
                "centered_fp64cast_mean_common_rms",
                "precision_oracle",
                MOMENT_CENTERED_FP64,
                "posterior_mean_common_rms",
                POSTERIOR_LEGACY,
                DECODER_LEGACY,
                "legacy_mean_common_rms",
                "legacy_conf",
                False,
            ),
        )
        self.assertEqual(
            tuple(
                (
                    spec.arm_index,
                    spec.arm_id,
                    spec.role,
                    spec.moment_reduction,
                    spec.decoder_semantics,
                    spec.posterior_reduction,
                    spec.decoder_reduction,
                    spec.comparator_arm_id,
                    spec.rms_reference_arm_id,
                    spec.selectable,
                )
                for spec in SCIENTIFIC_ARMS
            ),
            expected_arms,
        )

    def test_failed_e54_sentinel_map_is_exactly_frozen_inside_the_policy(self) -> None:
        expected = {
            "subject_344": 0.473039687,
            "subject_136": 0.460612535,
            "subject_165": 0.500411987,
            "subject_475": 0.530803859,
            "subject_131": 0.438622355,
        }
        self.assertEqual(dict(SENTINEL_ALL_VECTORIZED_GAPS), expected)
        fixed = dict(NUMERICAL_STABILITY_POLICY.fixed_parameters)
        self.assertEqual(fixed["failed_vectorized_source_git_head"], "e54d6bf4c026")
        self.assertEqual(fixed["failed_vectorized_sentinel_gaps"], tuple(expected.items()))

    def test_policy_hash_and_selfcheck_are_frozen(self) -> None:
        self.assertEqual(policy_sha256(), NUMERICAL_STABILITY_POLICY_SHA256)
        report = selfcheck()
        self.assertEqual(report["status"], "PASS")
        self.assertEqual(report["failed"], [])

    def test_one_raw_volume_drives_the_complete_study(self) -> None:
        shape = (9, 10, 11)
        generator = torch.Generator().manual_seed(31)
        costs = 0.1 + 1e-5 * torch.randn((1, len(OFFSETS), *shape), generator=generator)
        valid = torch.ones_like(costs, dtype=torch.bool)
        raw = RawMindCostVolume(costs=costs, valid=valid, valid_count=valid.sum(dim=1, keepdim=True))
        mask = geometry_mask(shape, 4, costs.device)
        legacy_reference = torch.randn((1, 3, *shape), generator=generator) * 0.01
        snapshots = (costs.clone(), valid.clone())

        study = build_reduction_study(raw, mask, legacy_confidence_reference=legacy_reference)

        self.assertEqual(set(study.factorial_residuals), {spec.cell_id for spec in FACTORIAL_SPECS})
        self.assertEqual(set(study.historical_requested), {"legacy_conf", "legacy_mean_common_rms"})
        torch.testing.assert_close(study.historical_requested["legacy_conf"], legacy_reference, atol=0, rtol=0)
        reference_rms = legacy_reference.double().square().sum(dim=1, keepdim=True).masked_select(mask).mean().sqrt()
        matched_mean_rms = (
            study.historical_requested["legacy_mean_common_rms"]
            .double()
            .square()
            .sum(dim=1, keepdim=True)
            .masked_select(mask)
            .mean()
            .sqrt()
        )
        torch.testing.assert_close(matched_mean_rms, reference_rms, atol=1e-8, rtol=1e-8)
        self.assertEqual(set(study.scientific_requested), {spec.arm_id for spec in SCIENTIFIC_ARMS})
        self.assertEqual(len(study.normalization_rows), 4)
        self.assertEqual(len(study.factorial_cell_rows), 8)
        self.assertEqual(len(study.factorial_edge_rows), 12)
        self.assertEqual(
            [
                (
                    row["cell_id"],
                    row["moment_reduction"],
                    row["posterior_reduction"],
                    row["decoder_reduction"],
                )
                for row in study.factorial_cell_rows
            ],
            [
                (spec.cell_id, spec.moment_reduction, spec.posterior_reduction, spec.decoder_reduction)
                for spec in FACTORIAL_SPECS
            ],
        )
        self.assertEqual(
            [(row["axis"], row["source_cell_id"], row["target_cell_id"]) for row in study.factorial_edge_rows],
            list(FACTORIAL_EDGES),
        )
        self.assertEqual(
            [(row["arm_id"], row["moment_reduction"], row["decoder_semantics"]) for row in study.scientific_rows],
            [(spec.arm_id, spec.moment_reduction, spec.decoder_semantics) for spec in SCIENTIFIC_ARMS],
        )
        torch.testing.assert_close(costs, snapshots[0], atol=0, rtol=0)
        self.assertTrue(torch.equal(valid, snapshots[1]))

    def test_study_rejects_float64_raw_costs_before_any_scientific_branch(self) -> None:
        shape = (9, 9, 9)
        costs = torch.zeros((1, len(OFFSETS), *shape), dtype=torch.float64)
        valid = torch.ones_like(costs, dtype=torch.bool)
        raw = RawMindCostVolume(costs=costs, valid=valid, valid_count=valid.sum(dim=1, keepdim=True))
        mask = geometry_mask(shape, 4, costs.device)

        with self.assertRaisesRegex(TypeError, "requires float32 raw costs"):
            build_reduction_study(
                raw,
                mask,
                legacy_confidence_reference=torch.zeros((1, 3, *shape), dtype=torch.float32),
            )

    def test_study_rejects_active_geometry_without_any_valid_candidate(self) -> None:
        shape = (9, 9, 9)
        costs = torch.zeros((1, len(OFFSETS), *shape), dtype=torch.float32)
        valid = torch.ones_like(costs, dtype=torch.bool)
        mask = geometry_mask(shape, 4, costs.device)
        active = mask.nonzero(as_tuple=False)[0]
        valid[0, :, int(active[2]), int(active[3]), int(active[4])] = False
        raw = RawMindCostVolume(costs=costs, valid=valid, valid_count=valid.sum(dim=1, keepdim=True))

        with self.assertRaisesRegex(ValueError, "active geometry voxel without a valid candidate"):
            build_reduction_study(
                raw,
                mask,
                legacy_confidence_reference=torch.zeros((1, 3, *shape), dtype=torch.float32),
            )


if __name__ == "__main__":
    unittest.main()
