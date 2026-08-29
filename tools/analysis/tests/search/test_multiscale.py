from __future__ import annotations

import math
import unittest

import torch

from tools.analysis.search.multiscale import (
    C4_COMMON_COLLAR_WIDTH,
    OFFSETS_STRIDE1,
    OFFSETS_STRIDE2,
    OFFSETS_STRIDE3,
    OFFSETS_STRIDE4,
    DecodedProposal,
    PosteriorVolume,
    RawCostVolume,
    WorkEstimate,
    build_c4_common_support,
    build_mind_feature_pair,
    build_raw_intensity_cost_volume,
    build_raw_mind_cost_volume,
    build_raw_mind_cost_volume_from_features,
    centered_standardize,
    decode_posterior_mean,
    descriptor_support_margin,
    duplicate_fusion_diagnostic,
    fuse_standardized_costs,
    offsets_for_stride,
    posterior_from_standardized_costs,
    posterior_from_standardized_costs_with_prior,
    postprocess_and_match_rms,
    quadratic_center_log_prior,
    scale_agreement_diagnostics,
)


def raw_volume(
    costs: torch.Tensor,
    *,
    cost_id: str = "raw",
    valid: torch.Tensor | None = None,
    offsets: tuple[tuple[int, int, int], ...] = OFFSETS_STRIDE1,
) -> RawCostVolume:
    return RawCostVolume(
        cost_id=cost_id,
        costs=costs,
        valid=torch.ones_like(costs, dtype=torch.bool) if valid is None else valid,
        offsets=offsets,
        work=WorkEstimate(0, int(costs.numel()), int(costs.numel())),
    )


class OffsetAndSupportTest(unittest.TestCase):
    def test_offset_tables_are_explicit_lexicographic_zyx(self) -> None:
        self.assertEqual(len(OFFSETS_STRIDE1), 27)
        self.assertEqual(len(OFFSETS_STRIDE2), 27)
        self.assertEqual(OFFSETS_STRIDE1, tuple(sorted(OFFSETS_STRIDE1)))
        self.assertEqual(OFFSETS_STRIDE2, tuple(sorted(OFFSETS_STRIDE2)))
        self.assertEqual(OFFSETS_STRIDE3, tuple(sorted(OFFSETS_STRIDE3)))
        self.assertEqual(OFFSETS_STRIDE4, tuple(sorted(OFFSETS_STRIDE4)))
        self.assertEqual(OFFSETS_STRIDE1[13], (0, 0, 0))
        self.assertEqual(OFFSETS_STRIDE2[13], (0, 0, 0))
        self.assertEqual(OFFSETS_STRIDE2, tuple(tuple(2 * value for value in offset) for offset in OFFSETS_STRIDE1))
        self.assertEqual(OFFSETS_STRIDE3, tuple(tuple(3 * value for value in offset) for offset in OFFSETS_STRIDE1))
        self.assertEqual(OFFSETS_STRIDE4, tuple(tuple(4 * value for value in offset) for offset in OFFSETS_STRIDE1))
        self.assertIs(offsets_for_stride(1), OFFSETS_STRIDE1)
        self.assertIs(offsets_for_stride(2), OFFSETS_STRIDE2)
        self.assertIs(offsets_for_stride(3), OFFSETS_STRIDE3)
        self.assertIs(offsets_for_stride(4), OFFSETS_STRIDE4)
        with self.assertRaisesRegex(ValueError, "must be one of"):
            offsets_for_stride(5)

    def test_quadratic_center_prior_is_dimensionless_across_strides(self) -> None:
        beta = 0.75
        expected = tuple(-beta * sum(value * value for value in offset) for offset in OFFSETS_STRIDE1)

        for stride in (1, 2, 3, 4):
            self.assertEqual(quadratic_center_log_prior(offsets_for_stride(stride), beta=beta), expected)

        prior = quadratic_center_log_prior(OFFSETS_STRIDE4, beta=beta)
        self.assertEqual(prior[OFFSETS_STRIDE4.index((0, 0, 0))], 0.0)
        self.assertEqual(prior[OFFSETS_STRIDE4.index((4, 0, 0))], -beta)
        self.assertEqual(prior[OFFSETS_STRIDE4.index((4, 4, 0))], -2.0 * beta)
        self.assertEqual(prior[OFFSETS_STRIDE4.index((4, 4, 4))], -3.0 * beta)

    def test_quadratic_center_prior_rejects_invalid_beta(self) -> None:
        for beta in (-0.1, math.inf, math.nan):
            with self.assertRaisesRegex(ValueError, "finite and non-negative"):
                quadratic_center_log_prior(OFFSETS_STRIDE1, beta=beta)
        with self.assertRaisesRegex(TypeError, "real scalar"):
            quadratic_center_log_prior(OFFSETS_STRIDE1, beta=True)

    def test_collar7_is_exact_max_descriptor_plus_stride_support_at_identity(self) -> None:
        shape = (19, 20, 21)
        psi = torch.zeros((1, 3, *shape))
        support = build_c4_common_support(psi)

        self.assertEqual(descriptor_support_margin(radius=1, dilation=4), 5)
        self.assertEqual(C4_COMMON_COLLAR_WIDTH, 5 + 2)
        torch.testing.assert_close(support.common_mask, support.geometry_mask, atol=0, rtol=0)
        self.assertEqual(support.retention, 1.0)

    def test_common_support_accounts_for_psi_not_only_target_collar(self) -> None:
        shape = (19, 19, 19)
        psi = torch.zeros((1, 3, *shape))
        baseline = build_c4_common_support(psi)
        psi[:, 0] = 1.0
        shifted = build_c4_common_support(psi)

        self.assertEqual(baseline.common_count, 5**3)
        self.assertEqual(shifted.common_count, 4 * 5 * 5)
        self.assertLess(shifted.retention, 1.0)


class CostConstructionTest(unittest.TestCase):
    @staticmethod
    def translated_pair(
        offset: tuple[int, int, int], shape: tuple[int, int, int] = (21, 21, 21)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        generator = torch.Generator().manual_seed(81)
        moving = torch.zeros((1, 1, *shape))
        moving[:, :, 7:-7, 7:-7, 7:-7] = torch.randn((1, 1, 7, 7, 7), generator=generator)
        fixed = torch.roll(moving, shifts=tuple(-value for value in offset), dims=(2, 3, 4))
        return fixed, moving

    def assert_mind_recovers_shift(self, stride: int, offset: tuple[int, int, int]) -> None:
        fixed, moving = self.translated_pair(offset)
        psi = torch.zeros((1, 3, *fixed.shape[-3:]))
        geometry = torch.zeros((1, 1, *fixed.shape[-3:]), dtype=torch.bool)
        geometry[..., 10, 10, 10] = True
        raw = build_raw_mind_cost_volume(
            fixed,
            moving,
            psi,
            geometry,
            dilation=2,
            offsets=offsets_for_stride(stride),
        )
        observed = raw.offsets[int(raw.costs[0, :, 10, 10, 10].argmin().item())]
        self.assertEqual(observed, offset)

    def test_s1_mind_cost_recovers_known_translation(self) -> None:
        self.assert_mind_recovers_shift(1, (1, 0, -1))

    def test_s2_mind_cost_recovers_known_translation(self) -> None:
        self.assert_mind_recovers_shift(2, (2, 0, -2))

    def test_precomputed_mind_pair_matches_convenience_path_and_is_reusable(self) -> None:
        fixed, moving = self.translated_pair((1, 0, -1))
        psi = torch.zeros((1, 3, *fixed.shape[-3:]))
        geometry = torch.zeros((1, 1, *fixed.shape[-3:]), dtype=torch.bool)
        geometry[..., 10, 10, 10] = True
        features = build_mind_feature_pair(fixed, moving, dilation=2)

        reused = build_raw_mind_cost_volume_from_features(
            features,
            psi,
            geometry,
            offsets=OFFSETS_STRIDE1,
            cost_id="reused",
        )
        direct = build_raw_mind_cost_volume(
            fixed,
            moving,
            psi,
            geometry,
            dilation=2,
            offsets=OFFSETS_STRIDE1,
            cost_id="direct",
        )

        torch.testing.assert_close(reused.costs, direct.costs, atol=0, rtol=0)
        self.assertTrue(torch.equal(reused.valid, direct.valid))
        self.assertEqual(features.work.descriptor_evaluations, 2)
        self.assertEqual(reused.work.descriptor_evaluations, 0)
        self.assertEqual(direct.work.descriptor_evaluations, 2)

    def test_intensity_cost_is_target_centred_scalar_ssd(self) -> None:
        shape = (17, 17, 17)
        moving = torch.arange(math.prod(shape), dtype=torch.float32).reshape(1, 1, *shape)
        fixed = torch.roll(moving, shifts=(-1, 0, 1), dims=(2, 3, 4))
        psi = torch.zeros((1, 3, *shape))
        geometry = torch.zeros((1, 1, *shape), dtype=torch.bool)
        geometry[..., 8, 8, 8] = True

        raw = build_raw_intensity_cost_volume(
            fixed,
            moving,
            psi,
            geometry,
            offsets=OFFSETS_STRIDE1,
            cost_id="intensity_s1",
        )

        best = raw.offsets[int(raw.costs[0, :, 8, 8, 8].argmin().item())]
        self.assertEqual(best, (1, 0, -1))
        self.assertEqual(float(raw.costs[0, raw.offsets.index(best), 8, 8, 8]), 0.0)


class StandardizationAndFusionTest(unittest.TestCase):
    def setUp(self) -> None:
        generator = torch.Generator().manual_seed(15)
        self.base = torch.randn((1, 27, 2, 3, 4), generator=generator)

    def test_centered_standardization_has_zero_mean_and_unit_population_std(self) -> None:
        bank = centered_standardize(raw_volume(self.base))

        torch.testing.assert_close(bank.standardized_costs.mean(dim=1), torch.zeros((1, 2, 3, 4)), atol=2e-7, rtol=0)
        torch.testing.assert_close(
            bank.standardized_costs.square().mean(dim=1),
            torch.ones((1, 2, 3, 4)),
            atol=3e-7,
            rtol=0,
        )

    def test_f222_duplicate_fusion_is_idempotent(self) -> None:
        d2 = centered_standardize(raw_volume(self.base, cost_id="mind_d2_s1"))
        f222 = fuse_standardized_costs((d2, d2, d2), cost_id="mind_f222_s1")
        mask = torch.ones((1, 1, 2, 3, 4), dtype=torch.bool)

        diagnostic = duplicate_fusion_diagnostic(d2, f222, mask)

        self.assertLessEqual(diagnostic.max_abs_standardized_difference, 5e-7)
        self.assertEqual(diagnostic.argmin_agreement, 1.0)

    def test_f124_removes_positive_affine_scale_before_fusion(self) -> None:
        d1 = centered_standardize(raw_volume(self.base, cost_id="mind_d1"))
        d2 = centered_standardize(raw_volume(self.base * 10.0 + 7.0, cost_id="mind_d2"))
        d4 = centered_standardize(raw_volume(self.base * 0.1 - 3.0, cost_id="mind_d4"))
        fused = fuse_standardized_costs((d1, d2, d4), cost_id="mind_f124")

        torch.testing.assert_close(fused.standardized_costs, d1.standardized_costs, atol=8e-6, rtol=8e-6)
        torch.testing.assert_close(fused.standardized_costs.mean(dim=1), torch.zeros((1, 2, 3, 4)), atol=2e-7, rtol=0)
        torch.testing.assert_close(
            fused.standardized_costs.square().mean(dim=1),
            torch.ones((1, 2, 3, 4)),
            atol=4e-7,
            rtol=0,
        )

    def test_flat_and_nonfinite_valid_costs_fail_closed(self) -> None:
        with self.assertRaisesRegex(FloatingPointError, "flat or unsupported"):
            centered_standardize(raw_volume(torch.zeros_like(self.base), cost_id="flat"))

        nonfinite = self.base.clone()
        nonfinite[:, 0, 0, 0, 0] = torch.nan
        with self.assertRaisesRegex(ValueError, "valid candidate costs must be finite"):
            centered_standardize(raw_volume(nonfinite, cost_id="nan"))

    def test_nonfinite_invalid_storage_is_not_treated_as_evidence(self) -> None:
        costs = self.base.clone()
        valid = torch.ones_like(costs, dtype=torch.bool)
        costs[:, 0, 0, 0, 0] = torch.inf
        valid[:, 0, 0, 0, 0] = False

        observed = centered_standardize(raw_volume(costs, valid=valid))

        self.assertFalse(bool(observed.valid[:, 0, 0, 0, 0]))
        self.assertEqual(float(observed.standardized_costs[:, 0, 0, 0, 0]), 0.0)


class PosteriorAndDiagnosticsTest(unittest.TestCase):
    def test_beta_zero_is_bit_identical_to_unbiased_posterior(self) -> None:
        generator = torch.Generator().manual_seed(913)
        costs = torch.randn((1, 27, 2, 3, 4), generator=generator)
        valid = torch.rand((1, 27, 2, 3, 4), generator=generator) > 0.15
        bank = centered_standardize(raw_volume(costs, valid=valid))

        unbiased = posterior_from_standardized_costs(bank, temperature=0.7)
        beta_zero = posterior_from_standardized_costs_with_prior(bank, beta=0.0, temperature=0.7)

        self.assertTrue(torch.equal(beta_zero.probabilities, unbiased.probabilities))
        self.assertTrue(torch.equal(beta_zero.entropy, unbiased.entropy))
        self.assertTrue(torch.equal(beta_zero.confidence, unbiased.confidence))
        self.assertIs(beta_zero.valid, unbiased.valid)
        self.assertEqual(beta_zero.offsets, unbiased.offsets)
        self.assertEqual(beta_zero.temperature, unbiased.temperature)

    def test_quadratic_prior_prefers_centre_without_changing_validity(self) -> None:
        costs = torch.tensor(
            [sum(value * value for value in offset) for offset in OFFSETS_STRIDE4],
            dtype=torch.float32,
        ).view(1, 27, 1, 1, 1)
        valid = torch.ones_like(costs, dtype=torch.bool)
        invalid_index = OFFSETS_STRIDE4.index((-4, -4, -4))
        valid[:, invalid_index] = False
        bank = centered_standardize(raw_volume(costs, valid=valid, offsets=OFFSETS_STRIDE4))

        posterior = posterior_from_standardized_costs_with_prior(bank, beta=0.5)

        centre = posterior.probabilities[0, OFFSETS_STRIDE4.index((0, 0, 0)), 0, 0, 0]
        face = posterior.probabilities[0, OFFSETS_STRIDE4.index((4, 0, 0)), 0, 0, 0]
        corner = posterior.probabilities[0, OFFSETS_STRIDE4.index((4, 4, 4)), 0, 0, 0]
        self.assertGreater(float(centre), float(face))
        self.assertGreater(float(face), float(corner))
        self.assertEqual(float(posterior.probabilities[0, invalid_index, 0, 0, 0]), 0.0)
        self.assertIs(posterior.valid, bank.valid)
        self.assertTrue(torch.equal(posterior.valid, bank.valid))

    def test_prior_posterior_is_stride_invariant_for_identical_costs(self) -> None:
        generator = torch.Generator().manual_seed(303)
        costs = torch.randn((1, 27, 2, 1, 1), generator=generator)
        stride1 = centered_standardize(raw_volume(costs, offsets=OFFSETS_STRIDE1))
        stride4 = centered_standardize(raw_volume(costs, offsets=OFFSETS_STRIDE4))

        posterior1 = posterior_from_standardized_costs_with_prior(stride1, beta=0.4)
        posterior4 = posterior_from_standardized_costs_with_prior(stride4, beta=0.4)

        self.assertTrue(torch.equal(posterior1.probabilities, posterior4.probabilities))
        self.assertTrue(torch.equal(posterior1.entropy, posterior4.entropy))
        self.assertTrue(torch.equal(posterior1.confidence, posterior4.confidence))

    def test_decoder_uses_explicit_zyx_offset_magnitude_and_sign(self) -> None:
        shape = (2, 2, 2)
        probabilities = torch.zeros((1, 27, *shape))
        index = OFFSETS_STRIDE2.index((2, -2, 0))
        probabilities[:, index] = 1.0
        valid = torch.ones_like(probabilities, dtype=torch.bool)
        posterior = PosteriorVolume(
            cost_id="one_hot",
            probabilities=probabilities,
            entropy=torch.zeros((1, 1, *shape)),
            confidence=torch.ones((1, 1, *shape)),
            valid=valid,
            offsets=OFFSETS_STRIDE2,
            temperature=1.0,
        )

        decoded = decode_posterior_mean(posterior)

        self.assertTrue(bool((decoded.displacement[:, 0] == 2.0).all()))
        self.assertTrue(bool((decoded.displacement[:, 1] == -2.0).all()))
        self.assertTrue(bool((decoded.displacement[:, 2] == 0.0).all()))

    def test_identical_banks_have_zero_js_and_unit_cosines(self) -> None:
        generator = torch.Generator().manual_seed(44)
        costs = torch.randn((1, 27, 2, 3, 4), generator=generator)
        bank = centered_standardize(raw_volume(costs, cost_id="same"))
        posterior = posterior_from_standardized_costs(bank)
        residual = decode_posterior_mean(posterior).displacement
        mask = torch.ones((1, 1, 2, 3, 4), dtype=torch.bool)

        diagnostic = scale_agreement_diagnostics(bank, bank, posterior, posterior, residual, residual, mask)

        self.assertEqual(diagnostic.argmin_agreement, 1.0)
        self.assertAlmostEqual(diagnostic.posterior_js_divergence_mean, 0.0, places=8)
        self.assertAlmostEqual(diagnostic.posterior_cosine_mean, 1.0, places=6)
        self.assertAlmostEqual(diagnostic.residual_cosine_mean or 0.0, 1.0, places=6)

    def test_postprocess_wrapper_applies_multiplier_smoothing_collar_and_rms_match(self) -> None:
        shape = (17, 17, 17)
        generator = torch.Generator().manual_seed(72)
        residual = torch.randn((1, 3, *shape), generator=generator)
        reference = residual * 0.25
        mask = torch.zeros((1, 1, *shape), dtype=torch.bool)
        mask[:, :, 3:-3, 3:-3, 3:-3] = True

        result = postprocess_and_match_rms(
            DecodedProposal("proposal", residual, OFFSETS_STRIDE1),
            mask,
            proposal_multiplier=2.0,
            smoothing_passes=1,
            collar_width=3,
            rms_reference=reference,
        )

        self.assertAlmostEqual(result.output_rms, result.target_rms or -1.0, places=7)
        self.assertEqual(int(torch.count_nonzero(result.displacement[:, :, 0, :, :])), 0)
        self.assertEqual(int(torch.count_nonzero(result.displacement[:, :, -1, :, :])), 0)
        self.assertEqual(int(torch.count_nonzero(result.displacement[:, :, :, 0, :])), 0)
        self.assertEqual(int(torch.count_nonzero(result.displacement[:, :, :, :, -1])), 0)


if __name__ == "__main__":
    unittest.main()
