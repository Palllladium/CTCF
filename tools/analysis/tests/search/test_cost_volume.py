from __future__ import annotations

import math
import unittest
from dataclasses import replace

import torch
import torch.nn.functional as F

from tools.analysis.search.cost_volume import (
    POSTERIOR_DIAGNOSTICS_ID,
    RMS_FIRST_DIFFERENCE_ROUGHNESS_ID,
    _standardize_candidate_costs,
    apply_message_passing,
    build_standardized_mind_cost_volume,
    decode_posterior,
    masked_rms_first_difference_roughness,
    masked_separable_smooth_logits,
    masked_vector_rms,
    match_postprocessed_rms,
    posterior_diagnostics,
    posterior_from_logits,
    postprocess_residual,
    raw_posterior,
)
from tools.analysis.search.transaction import OFFSETS, ZERO_OFFSET_INDEX, geometry_mask, sample_at_psi


def candidate_tensors(shape: tuple[int, int, int] = (7, 8, 9)) -> tuple[torch.Tensor, torch.Tensor]:
    logits = torch.zeros((1, len(OFFSETS), *shape), dtype=torch.float32)
    valid = torch.ones_like(logits, dtype=torch.bool)
    return logits, valid


class CostVolumeConstructionTest(unittest.TestCase):
    def test_standardization_preserves_legacy_order_under_float32_cancellation(self) -> None:
        generator = torch.Generator().manual_seed(7)
        costs = 0.1 + 1e-5 * torch.randn((1, len(OFFSETS), 4, 5, 6), generator=generator)
        valid = torch.ones_like(costs, dtype=torch.bool)

        standardized, count, mean, std = _standardize_candidate_costs(
            costs,
            valid,
            standardization_floor=1e-6,
        )
        expected_sum = torch.zeros_like(mean)
        expected_square = torch.zeros_like(mean)
        for index in range(len(OFFSETS)):
            candidate = costs[:, index : index + 1]
            expected_sum += candidate
            expected_square += candidate.square()
        expected_mean = expected_sum / float(len(OFFSETS))
        expected_std = (
            (expected_square / float(len(OFFSETS)) - expected_mean.square()).clamp_min(0.0).sqrt().clamp_min(1e-6)
        )
        expected_standardized = (costs - expected_mean) / expected_std

        torch.testing.assert_close(mean, expected_mean, atol=0, rtol=0)
        torch.testing.assert_close(std, expected_std, atol=0, rtol=0)
        torch.testing.assert_close(standardized, expected_standardized, atol=0, rtol=0)
        self.assertTrue(bool((count == len(OFFSETS)).all()))

        vector_mean = costs.sum(dim=1, keepdim=True) / float(len(OFFSETS))
        vector_std = (
            (costs.square().sum(dim=1, keepdim=True) / float(len(OFFSETS)) - vector_mean.square())
            .clamp_min(0.0)
            .sqrt()
            .clamp_min(1e-6)
        )
        vector_standardized = (costs - vector_mean) / vector_std
        self.assertGreater(float((vector_standardized - standardized).abs().max()), 0.01)

    def test_known_shift_preserves_offset_order_and_additive_sign(self) -> None:
        shape = (9, 10, 11)
        generator = torch.Generator().manual_seed(77)
        moving = torch.randn((1, 12, *shape), generator=generator)
        psi = torch.zeros((1, 3, *shape))
        fixed = sample_at_psi(moving, psi, offset=(0, 0, 1))
        mask = geometry_mask(shape, 2, moving.device)

        volume = build_standardized_mind_cost_volume(fixed, moving, psi, mask)
        posterior = raw_posterior(volume)
        interior = (slice(None), slice(None), slice(3, -3), slice(3, -3), slice(3, -3))
        argmax = posterior.probabilities[interior].argmax(dim=1)
        expected_index = OFFSETS.index((0, 0, 1))

        self.assertEqual(ZERO_OFFSET_INDEX, 13)
        self.assertEqual(expected_index, 14)
        self.assertTrue(bool((argmax == expected_index).all()))
        decoded = decode_posterior(posterior, mode="posterior_mean").displacement
        mean_components = decoded[:, :, 3:-3, 3:-3, 3:-3].mean(dim=(0, 2, 3, 4))
        self.assertGreater(float(mean_components[2]), 0.0)
        self.assertLess(abs(float(mean_components[0])), 0.05)
        self.assertLess(abs(float(mean_components[1])), 0.05)

    def test_builder_does_not_mutate_features_field_or_mask(self) -> None:
        shape = (7, 8, 9)
        generator = torch.Generator().manual_seed(4)
        fixed = torch.randn((1, 12, *shape), generator=generator)
        moving = torch.randn((1, 12, *shape), generator=generator)
        psi = torch.zeros((1, 3, *shape))
        mask = geometry_mask(shape, 2, fixed.device)
        snapshots = tuple(value.clone() for value in (fixed, moving, psi, mask))

        result = build_standardized_mind_cost_volume(fixed, moving, psi, mask)

        for actual, expected in zip((fixed, moving, psi, mask), snapshots, strict=True):
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        self.assertEqual(result.standardized_costs.shape, (1, 27, *shape))
        self.assertEqual(result.valid.dtype, torch.bool)
        self.assertTrue(bool(torch.isfinite(result.standardized_costs).all()))

    def test_standardization_is_per_voxel_over_valid_candidates(self) -> None:
        shape = (7, 8, 9)
        generator = torch.Generator().manual_seed(18)
        fixed = torch.randn((1, 12, *shape), generator=generator)
        moving = torch.randn((1, 12, *shape), generator=generator)
        psi = torch.zeros((1, 3, *shape))
        mask = geometry_mask(shape, 2, fixed.device)

        result = build_standardized_mind_cost_volume(fixed, moving, psi, mask)
        centre = result.standardized_costs[0, :, 3, 3, 4]
        centre_valid = result.valid[0, :, 3, 3, 4]
        values = centre[centre_valid]
        self.assertAlmostEqual(float(values.mean()), 0.0, places=5)
        self.assertAlmostEqual(float(values.square().mean()), 1.0, places=4)
        self.assertTrue(bool((result.standardized_costs.masked_select(~result.valid) == 0.0).all()))


class PosteriorAndDecoderTest(unittest.TestCase):
    def test_posterior_and_decoder_preserve_legacy_candidate_reduction_order(self) -> None:
        generator = torch.Generator().manual_seed(123)
        logits = torch.randn((1, len(OFFSETS), 3, 4, 5), generator=generator)
        valid = torch.rand((1, len(OFFSETS), 3, 4, 5), generator=generator) > 0.1
        valid[:, :2] = True
        valid[:, ZERO_OFFSET_INDEX] = True

        maximum = torch.full((1, 1, 3, 4, 5), -torch.inf)
        for index in range(len(OFFSETS)):
            candidate = logits[:, index : index + 1]
            candidate_valid = valid[:, index : index + 1]
            maximum = torch.where(candidate_valid & (candidate > maximum), candidate, maximum)
        weights = torch.zeros_like(logits)
        normalizer = torch.zeros_like(maximum)
        weighted_shift_sum = torch.zeros_like(maximum)
        for index in range(len(OFFSETS)):
            candidate_valid = valid[:, index : index + 1]
            shifted = logits[:, index : index + 1] - maximum
            safe_shifted = torch.where(candidate_valid, shifted, torch.zeros_like(shifted))
            weight = torch.exp(safe_shifted) * candidate_valid
            weights[:, index : index + 1] = weight
            normalizer += weight
            weighted_shift_sum += weight * safe_shifted
        expected_probabilities = weights / normalizer
        expected_entropy = torch.log(normalizer) - weighted_shift_sum / normalizer
        count = valid.sum(dim=1, keepdim=True)
        expected_h = expected_entropy / torch.log(count.to(logits.dtype))
        expected_confidence = (1.0 - expected_h).clamp(0.0, 1.0)
        expected_mean = torch.zeros((1, 3, 3, 4, 5))
        for index, offset in enumerate(OFFSETS):
            expected_mean += expected_probabilities[:, index : index + 1] * logits.new_tensor(offset).view(
                1, 3, 1, 1, 1
            )

        posterior = posterior_from_logits(logits, valid)
        decoded = decode_posterior(posterior, mode="confidence")

        torch.testing.assert_close(posterior.probabilities, expected_probabilities, atol=0, rtol=0)
        torch.testing.assert_close(posterior.entropy, expected_entropy, atol=0, rtol=0)
        torch.testing.assert_close(posterior.confidence, expected_confidence, atol=0, rtol=0)
        torch.testing.assert_close(decoded.posterior_mean, expected_mean, atol=0, rtol=0)
        torch.testing.assert_close(
            decoded.displacement,
            expected_mean * expected_confidence,
            atol=0,
            rtol=0,
        )

    def test_flat_posterior_has_logk_entropy_and_zero_expectation(self) -> None:
        logits, valid = candidate_tensors()
        posterior = posterior_from_logits(logits, valid)

        torch.testing.assert_close(
            posterior.entropy,
            torch.full_like(posterior.entropy, math.log(len(OFFSETS))),
            atol=2e-6,
            rtol=0,
        )
        torch.testing.assert_close(posterior.normalized_entropy, torch.ones_like(posterior.entropy), atol=2e-6, rtol=0)
        torch.testing.assert_close(posterior.confidence, torch.zeros_like(posterior.entropy), atol=2e-6, rtol=0)
        for mode in ("confidence", "posterior_mean"):
            decoded = decode_posterior(posterior, mode=mode)
            torch.testing.assert_close(decoded.displacement, torch.zeros_like(decoded.displacement), atol=1e-7, rtol=0)

    def test_confidence_and_mean_decoders_differ_only_by_confidence(self) -> None:
        logits, valid = candidate_tensors((5, 5, 5))
        logits[:, OFFSETS.index((0, 0, 1))] = 2.0
        posterior = posterior_from_logits(logits, valid)
        mean = decode_posterior(posterior, mode="posterior_mean")
        confidence = decode_posterior(posterior, mode="confidence")

        torch.testing.assert_close(mean.posterior_mean, confidence.posterior_mean, atol=0, rtol=0)
        torch.testing.assert_close(
            confidence.displacement,
            mean.displacement * posterior.confidence,
            atol=0,
            rtol=0,
        )


class LabelFreeDiagnosticsTest(unittest.TestCase):
    def test_posterior_diagnostics_have_explicit_masked_denominators(self) -> None:
        shape = (3, 3, 3)
        logits = torch.zeros((1, 27, *shape))
        valid = torch.zeros_like(logits, dtype=torch.bool)
        mask = torch.zeros((1, 1, *shape), dtype=torch.bool)
        mask[0, 0, 1, 1, 1] = True
        indices = [OFFSETS.index(offset) for offset in ((0, 0, 1), (0, 1, 0), (1, 0, 0))]
        for index, value in zip(indices, (3.0, 1.0, 0.0), strict=True):
            valid[0, index, 1, 1, 1] = True
            logits[0, index, 1, 1, 1] = value
        posterior = posterior_from_logits(logits, valid)
        snapshots = (logits.clone(), valid.clone(), posterior.probabilities.clone(), mask.clone())

        diagnostics = posterior_diagnostics(logits, valid, posterior, mask)
        decoded = decode_posterior(posterior, mode="posterior_mean")
        mean_norm = float(decoded.posterior_mean[0, :, 1, 1, 1].double().norm().item())
        confidence = float(posterior.confidence[0, 0, 1, 1, 1].item())

        self.assertEqual(diagnostics.diagnostic_id, POSTERIOR_DIAGNOSTICS_ID)
        self.assertEqual(diagnostics.active_voxel_count, 1)
        self.assertEqual(diagnostics.candidate_count, 27)
        self.assertAlmostEqual(diagnostics.top1_top2_valid_logit_gap_mean, 2.0)
        self.assertAlmostEqual(
            diagnostics.posterior_peak_probability_mean,
            float(posterior.probabilities[0, :, 1, 1, 1].max().item()),
        )
        self.assertAlmostEqual(
            diagnostics.entropy_nats_mean,
            float(posterior.entropy[0, 0, 1, 1, 1].item()),
        )
        self.assertAlmostEqual(diagnostics.invalid_offset_fraction, 24.0 / 27.0)
        self.assertAlmostEqual(diagnostics.posterior_mean_l2_norm_mean, mean_norm)
        self.assertAlmostEqual(diagnostics.confidence_weighted_mean_l2_norm_mean, mean_norm * confidence)
        self.assertAlmostEqual(diagnostics.confidence_to_mean_l2_norm_ratio, confidence)
        for actual, expected in zip(
            (logits, valid, posterior.probabilities, mask),
            snapshots,
            strict=True,
        ):
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)

    def test_flat_symmetric_posterior_has_undefined_zero_over_zero_norm_ratio(self) -> None:
        logits, valid = candidate_tensors((3, 3, 3))
        mask = torch.ones((1, 1, 3, 3, 3), dtype=torch.bool)
        posterior = posterior_from_logits(logits, valid)

        diagnostics = posterior_diagnostics(logits, valid, posterior, mask)

        self.assertAlmostEqual(diagnostics.top1_top2_valid_logit_gap_mean, 0.0)
        self.assertAlmostEqual(diagnostics.posterior_peak_probability_mean, 1.0 / 27.0, places=7)
        self.assertAlmostEqual(diagnostics.entropy_nats_mean, math.log(27), places=6)
        self.assertEqual(diagnostics.invalid_offset_fraction, 0.0)
        self.assertEqual(diagnostics.posterior_mean_l2_norm_mean, 0.0)
        self.assertEqual(diagnostics.confidence_weighted_mean_l2_norm_mean, 0.0)
        self.assertIsNone(diagnostics.confidence_to_mean_l2_norm_ratio)

    def test_posterior_diagnostics_fail_closed_on_undefined_or_inconsistent_inputs(self) -> None:
        shape = (3, 3, 3)
        logits = torch.zeros((1, 27, *shape))
        valid = torch.zeros_like(logits, dtype=torch.bool)
        mask = torch.zeros((1, 1, *shape), dtype=torch.bool)
        mask[..., 1, 1, 1] = True
        valid[:, :2, 1, 1, 1] = True
        posterior = posterior_from_logits(logits, valid)

        one_valid = valid.clone()
        one_valid[:, 1, 1, 1, 1] = False
        one_posterior = posterior_from_logits(logits, one_valid)
        with self.assertRaisesRegex(ValueError, "at least two valid"):
            posterior_diagnostics(logits, one_valid, one_posterior, mask)

        nonfinite_logits = logits.clone()
        nonfinite_logits[:, 0, 1, 1, 1] = torch.nan
        with self.assertRaisesRegex(ValueError, "valid candidate logits"):
            posterior_diagnostics(nonfinite_logits, valid, posterior, mask)

        bad_probabilities = posterior.probabilities.clone()
        bad_probabilities[:, 2, 1, 1, 1] = 0.1
        with self.assertRaisesRegex(ValueError, "exactly zero"):
            posterior_diagnostics(
                logits,
                valid,
                replace(posterior, probabilities=bad_probabilities),
                mask,
            )

        with self.assertRaisesRegex(ValueError, "entropy is inconsistent"):
            posterior_diagnostics(
                logits,
                valid,
                replace(posterior, entropy=posterior.entropy + 0.1),
                mask,
            )

    def test_linear_field_has_known_rms_vector_first_differences(self) -> None:
        shape = (3, 4, 5)
        zz, yy, xx = torch.meshgrid(
            torch.arange(shape[0]),
            torch.arange(shape[1]),
            torch.arange(shape[2]),
            indexing="ij",
        )
        residual = torch.zeros((1, 3, *shape), dtype=torch.float32)
        residual[:, 0] = 2.0 * zz + 3.0 * yy + 4.0 * xx
        mask = torch.ones((1, 1, *shape), dtype=torch.bool)
        snapshot = residual.clone()

        result = masked_rms_first_difference_roughness(residual, mask)
        counts = (2 * 4 * 5, 3 * 3 * 5, 3 * 4 * 4)
        expected = math.sqrt((counts[0] * 2.0**2 + counts[1] * 3.0**2 + counts[2] * 4.0**2) / sum(counts))

        self.assertEqual(result.metric_id, RMS_FIRST_DIFFERENCE_ROUGHNESS_ID)
        self.assertEqual(result.axis_pair_counts_zyx, counts)
        self.assertEqual(result.pair_count, sum(counts))
        for observed, expected_axis in zip(
            result.axis_rms_vector_first_difference_zyx,
            (2.0, 3.0, 4.0),
            strict=True,
        ):
            self.assertAlmostEqual(observed, expected_axis)
        self.assertAlmostEqual(result.rms_vector_first_difference, expected)
        torch.testing.assert_close(residual, snapshot, atol=0, rtol=0)

    def test_constant_field_is_zero_and_masked_out_values_do_not_form_pairs(self) -> None:
        constant = torch.full((1, 3, 3, 4, 5), 7.0)
        full_mask = torch.ones((1, 1, 3, 4, 5), dtype=torch.bool)
        self.assertEqual(
            masked_rms_first_difference_roughness(constant, full_mask).rms_vector_first_difference,
            0.0,
        )

        residual = torch.full((1, 3, 3, 3, 3), torch.nan)
        mask = torch.zeros((1, 1, 3, 3, 3), dtype=torch.bool)
        mask[..., 1, 1, 1:3] = True
        residual[..., 1, 1, 1] = 1.0
        residual[..., 1, 1, 2] = 4.0

        result = masked_rms_first_difference_roughness(residual, mask)

        self.assertEqual(result.axis_pair_counts_zyx, (0, 0, 1))
        self.assertEqual(result.axis_rms_vector_first_difference_zyx[:2], (None, None))
        self.assertAlmostEqual(result.axis_rms_vector_first_difference_zyx[2], math.sqrt(27.0))
        self.assertAlmostEqual(result.rms_vector_first_difference, math.sqrt(27.0))

    def test_roughness_fails_closed_without_pairs_or_with_nonfinite_geometry_value(self) -> None:
        residual = torch.zeros((1, 3, 3, 3, 3))
        mask = torch.zeros((1, 1, 3, 3, 3), dtype=torch.bool)
        mask[..., 1, 1, 1] = True
        with self.assertRaisesRegex(ValueError, "at least one"):
            masked_rms_first_difference_roughness(residual, mask)

        mask[..., 1, 1, 2] = True
        residual[..., 1, 1, 1] = torch.inf
        with self.assertRaisesRegex(ValueError, "finite at every geometry"):
            masked_rms_first_difference_roughness(residual, mask)


class MessagePassingTest(unittest.TestCase):
    def test_none_is_out_of_place_exact_parity(self) -> None:
        generator = torch.Generator().manual_seed(9)
        logits = torch.randn((1, 27, 7, 8, 9), generator=generator)
        valid = torch.ones_like(logits, dtype=torch.bool)
        h = torch.rand((1, 1, 7, 8, 9), generator=generator)
        mask = torch.ones_like(h, dtype=torch.bool)
        snapshot = logits.clone()

        result = apply_message_passing(logits, valid, h, mask, mode="none")

        torch.testing.assert_close(result.logits, logits, atol=0, rtol=0)
        torch.testing.assert_close(logits, snapshot, atol=0, rtol=0)
        self.assertNotEqual(result.logits.data_ptr(), logits.data_ptr())
        self.assertEqual(result.lambda_mean, 0.0)

    def test_constant_entropy_makes_adaptive_equal_isotropic(self) -> None:
        generator = torch.Generator().manual_seed(10)
        logits = torch.randn((1, 27, 7, 8, 9), generator=generator)
        valid = torch.ones_like(logits, dtype=torch.bool)
        h = torch.full((1, 1, 7, 8, 9), 0.625)
        mask = torch.ones_like(h, dtype=torch.bool)

        isotropic = apply_message_passing(logits, valid, h, mask, mode="isotropic")
        adaptive = apply_message_passing(logits, valid, h, mask, mode="adaptive")

        torch.testing.assert_close(isotropic.logits, adaptive.logits, atol=0, rtol=0)
        torch.testing.assert_close(isotropic.lambda_map, adaptive.lambda_map, atol=0, rtol=0)
        self.assertAlmostEqual(isotropic.lambda_mean, 0.625)

    def test_isotropic_uses_mean_g_and_adaptive_uses_voxelwise_h(self) -> None:
        logits, valid = candidate_tensors((5, 5, 5))
        logits[:, :, 2, 2, 2] = 1.0
        h = torch.zeros((1, 1, 5, 5, 5))
        h[..., 1:4, 1:4, 1:4] = torch.linspace(0.0, 1.0, 27).reshape(1, 1, 3, 3, 3)
        mask = torch.zeros_like(h, dtype=torch.bool)
        mask[..., 1:4, 1:4, 1:4] = True

        isotropic = apply_message_passing(logits, valid, h, mask, mode="isotropic")
        adaptive = apply_message_passing(logits, valid, h, mask, mode="adaptive")

        self.assertAlmostEqual(isotropic.lambda_mean, 0.5, places=6)
        torch.testing.assert_close(
            isotropic.lambda_map.masked_select(mask),
            torch.full((27,), 0.5),
            atol=1e-7,
            rtol=0,
        )
        torch.testing.assert_close(adaptive.lambda_map.masked_select(mask), h.masked_select(mask), atol=0, rtol=0)

    def test_constant_logits_stay_constant(self) -> None:
        logits = torch.full((1, 27, 7, 8, 9), 3.25)
        valid = torch.ones_like(logits, dtype=torch.bool)
        valid[..., 0, :, :] = False
        valid[..., -1, :, :] = False
        smoothed = masked_separable_smooth_logits(logits, valid)

        torch.testing.assert_close(smoothed.masked_select(valid), torch.full_like(smoothed.masked_select(valid), 3.25))
        self.assertTrue(bool((smoothed.masked_select(~valid) == 0.0).all()))

    def test_separable_kernel_has_the_expected_impulse_response(self) -> None:
        logits = torch.zeros((1, 27, 5, 5, 5))
        valid = torch.ones_like(logits, dtype=torch.bool)
        logits[0, ZERO_OFFSET_INDEX, 2, 2, 2] = 1.0

        smoothed = masked_separable_smooth_logits(logits, valid)[0, ZERO_OFFSET_INDEX]

        self.assertAlmostEqual(float(smoothed[2, 2, 2]), 1.0 / 8.0)
        self.assertAlmostEqual(float(smoothed[1, 2, 2]), 1.0 / 16.0)
        self.assertAlmostEqual(float(smoothed[1, 1, 1]), 1.0 / 64.0)

    def test_invalid_neighbours_are_excluded_without_nan(self) -> None:
        logits = torch.zeros((1, 27, 5, 5, 5))
        valid = torch.ones_like(logits, dtype=torch.bool)
        valid[:, :, 2, 2, 1] = False
        logits[:, :, 2, 2, 1] = 1e20
        logits[:, :, 2, 2, 2] = 2.0
        snapshot = logits.clone()

        smoothed = masked_separable_smooth_logits(logits, valid)

        self.assertTrue(bool(torch.isfinite(smoothed).all()))
        self.assertLess(float(smoothed.masked_select(valid).abs().max()), 3.0)
        self.assertTrue(bool((smoothed.masked_select(~valid) == 0.0).all()))
        torch.testing.assert_close(logits, snapshot, atol=0, rtol=0)

    def test_masked_separable_pass_matches_one_full_3d_ratio(self) -> None:
        generator = torch.Generator().manual_seed(91)
        logits = torch.randn((1, 27, 5, 6, 7), generator=generator)
        valid = torch.rand((1, 27, 5, 6, 7), generator=generator) > 0.3
        valid[..., 2, 3, 3] = True
        vector = logits.new_tensor((1.0, 2.0, 1.0)) / 4.0
        kernel = torch.einsum("i,j,k->ijk", vector, vector, vector)
        weight = kernel.view(1, 1, 3, 3, 3).expand(27, 1, 3, 3, 3)
        numerator = F.conv3d(torch.where(valid, logits, 0.0), weight, padding=1, groups=27)
        denominator = F.conv3d(valid.to(logits.dtype), weight, padding=1, groups=27)
        expected = torch.where(valid & (denominator > 0.0), numerator / denominator.clamp_min(1e-12), 0.0)

        observed = masked_separable_smooth_logits(logits, valid)

        torch.testing.assert_close(observed, expected, atol=2e-7, rtol=2e-7)

    def test_no_restandardization_after_message_blend(self) -> None:
        generator = torch.Generator().manual_seed(12)
        logits = torch.randn((1, 27, 5, 5, 5), generator=generator) * 3.0 + 4.0
        valid = torch.ones_like(logits, dtype=torch.bool)
        h = torch.full((1, 1, 5, 5, 5), 0.25)
        mask = torch.ones_like(h, dtype=torch.bool)
        smooth = masked_separable_smooth_logits(logits, valid)

        observed = apply_message_passing(logits, valid, h, mask, mode="isotropic").logits
        expected = 0.75 * logits + 0.25 * smooth

        torch.testing.assert_close(observed, expected, atol=0, rtol=0)


class PostprocessAndRMSTest(unittest.TestCase):
    def test_postprocess_applies_an_exact_zero_collar(self) -> None:
        generator = torch.Generator().manual_seed(31)
        residual = torch.randn((1, 3, 9, 10, 11), generator=generator)
        snapshot = residual.clone()

        output = postprocess_residual(residual, scale=2.0, post_smoothing_passes=2, collar_width=2)

        self.assertEqual(int(torch.count_nonzero(output[..., 0, :, :])), 0)
        self.assertEqual(int(torch.count_nonzero(output[..., -1, :, :])), 0)
        self.assertEqual(int(torch.count_nonzero(output[..., :, 0, :])), 0)
        self.assertEqual(int(torch.count_nonzero(output[..., :, -1, :])), 0)
        self.assertEqual(int(torch.count_nonzero(output[..., :, :, 0])), 0)
        self.assertEqual(int(torch.count_nonzero(output[..., :, :, -1])), 0)
        torch.testing.assert_close(residual, snapshot, atol=0, rtol=0)

    def test_nonzero_rms_is_matched_after_postprocessing(self) -> None:
        shape = (9, 10, 11)
        generator = torch.Generator().manual_seed(32)
        raw_reference = torch.randn((1, 3, *shape), generator=generator)
        raw_residual = 4.0 * raw_reference
        mask = geometry_mask(shape, 2, raw_reference.device)
        reference = postprocess_residual(raw_reference, scale=2.0, post_smoothing_passes=2, collar_width=2)
        residual = postprocess_residual(raw_residual, scale=2.0, post_smoothing_passes=2, collar_width=2)
        reference_snapshot, residual_snapshot = reference.clone(), residual.clone()

        result = match_postprocessed_rms(residual, reference, mask)

        self.assertAlmostEqual(result.scale_factor, 0.25, places=6)
        self.assertAlmostEqual(result.matched_rms, result.target_rms, places=10)
        self.assertAlmostEqual(masked_vector_rms(result.displacement, mask), result.target_rms, places=10)
        torch.testing.assert_close(reference, reference_snapshot, atol=0, rtol=0)
        torch.testing.assert_close(residual, residual_snapshot, atol=0, rtol=0)

    def test_zero_rms_cases_fail_closed_or_remain_zero(self) -> None:
        shape = (7, 8, 9)
        mask = geometry_mask(shape, 2, torch.device("cpu"))
        zero = torch.zeros((1, 3, *shape))
        nonzero = zero.clone()
        nonzero[:, :, 3, 3, 4] = 1.0

        both_zero = match_postprocessed_rms(zero, zero, mask)
        self.assertEqual(both_zero.scale_factor, 1.0)
        self.assertEqual(both_zero.matched_rms, 0.0)

        target_zero = match_postprocessed_rms(nonzero, zero, mask)
        self.assertEqual(target_zero.scale_factor, 0.0)
        self.assertEqual(target_zero.matched_rms, 0.0)

        with self.assertRaisesRegex(ValueError, "zero residual"):
            match_postprocessed_rms(zero, nonzero, mask)


if __name__ == "__main__":
    unittest.main()
