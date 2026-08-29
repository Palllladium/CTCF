from __future__ import annotations

import unittest

import torch

from models.CTCF.controller import (
    STAGE5_CHANNEL_GROUPS,
    STAGE5_FREE_RESIDUAL_HEAD,
    STAGE5_INPUT_CHANNEL_COUNT,
    STAGE5_INPUT_CHANNELS,
    STAGE5_RESERVED_HEAD,
    STAGE5_S2_ATTENUATION_HEAD,
    STAGE5_S4_ATTENUATION_HEAD,
    STAGE5_VARIANT_GROUPS,
    STAGE5_VARIANTS,
    Stage5SpatialController,
    mask_stage5_features,
    stage5_variant_mask,
)
from tools.analysis.stage5.contracts import CONTEXT_CHANNELS, SEARCH_CHANNELS, VARIANT_BY_ID
from utils.field import identity_collar


def _features(seed: int = 7, shape: tuple[int, int, int] = (8, 10, 12)) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.randn((1, STAGE5_INPUT_CHANNEL_COUNT, *shape), generator=generator)


def _proposals(shape: tuple[int, int, int] = (8, 10, 12)) -> tuple[torch.Tensor, torch.Tensor]:
    s2 = torch.zeros((1, 3, *shape))
    s4 = torch.zeros_like(s2)
    s2[:, 0] = 1.0
    s4[:, 1] = 2.0
    return s2, s4


class Stage5LayoutTest(unittest.TestCase):
    def test_layout_is_explicit_unique_and_exactly_71_channels(self) -> None:
        self.assertEqual(STAGE5_INPUT_CHANNEL_COUNT, 71)
        self.assertEqual(len(STAGE5_INPUT_CHANNELS), 71)
        self.assertEqual(len(set(STAGE5_INPUT_CHANNELS)), 71)
        self.assertEqual(STAGE5_INPUT_CHANNELS, (*CONTEXT_CHANNELS, *SEARCH_CHANNELS))
        self.assertEqual(
            STAGE5_INPUT_CHANNELS[:5],
            (
                "fixed_image",
                "warped_moving_image",
                "certified_source_displacement_z",
                "certified_source_displacement_y",
                "certified_source_displacement_x",
            ),
        )
        self.assertEqual(STAGE5_INPUT_CHANNELS[8], "s2_posterior_00")
        self.assertEqual(STAGE5_INPUT_CHANNELS[34], "s2_posterior_26")
        self.assertEqual(
            STAGE5_INPUT_CHANNELS[35:38],
            (
                "s2_entropy",
                "s2_top2_gap",
                "s2_valid_support",
            ),
        )
        self.assertEqual(STAGE5_INPUT_CHANNELS[41], "s4_posterior_00")
        self.assertEqual(STAGE5_INPUT_CHANNELS[67], "s4_posterior_26")
        self.assertEqual(
            STAGE5_INPUT_CHANNELS[68:71],
            (
                "s4_entropy",
                "s4_top2_gap",
                "s4_valid_support",
            ),
        )

        covered: list[int] = []
        for group in STAGE5_CHANNEL_GROUPS.values():
            covered.extend(range(group.start, group.stop))
        self.assertEqual(covered, list(range(71)))

    def test_variant_masks_match_the_frozen_causal_matrix(self) -> None:
        self.assertEqual(tuple(STAGE5_VARIANT_GROUPS), STAGE5_VARIANTS)
        expected_counts = {
            "F0": 5,
            "F2V": 8,
            "F2S": 11,
            "F2P": 35,
            "F4P": 35,
            "F24P": 65,
            "A2P": 35,
            "A24P": 65,
        }
        for variant, expected in expected_counts.items():
            mask = stage5_variant_mask(variant)
            self.assertEqual(mask.shape, (1, 71, 1, 1, 1))
            self.assertEqual(int(mask.sum()), expected)
            self.assertTrue(bool(mask[:, STAGE5_CHANNEL_GROUPS["context"]].all()))
            active_names = tuple(
                name for index, name in enumerate(STAGE5_INPUT_CHANNELS) if bool(mask[0, index, 0, 0, 0])
            )
            contract = VARIANT_BY_ID[variant]
            self.assertEqual(active_names, (*CONTEXT_CHANNELS, *contract.active_search_channels))

        with self.assertRaisesRegex(ValueError, "unknown Stage-5"):
            stage5_variant_mask("UNKNOWN")

    def test_masking_writes_literal_positive_zeros_to_unavailable_channels(self) -> None:
        features = torch.full((1, 71, 3, 4, 5), -3.0)
        masked = mask_stage5_features(features, "F0")
        search = masked[:, 5:]
        self.assertEqual(int(torch.count_nonzero(search)), 0)
        self.assertFalse(bool(torch.signbit(search).any()))
        torch.testing.assert_close(masked[:, :5], features[:, :5], atol=0, rtol=0)

    def test_shapes_dtypes_and_nonfinite_inputs_fail_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "shape"):
            mask_stage5_features(torch.zeros((1, 70, 3, 4, 5)), "F0")
        with self.assertRaisesRegex(TypeError, "floating"):
            mask_stage5_features(torch.zeros((1, 71, 3, 4, 5), dtype=torch.int64), "F0")
        invalid = torch.zeros((1, 71, 3, 4, 5))
        invalid[:, 70, 0, 0, 0] = torch.nan
        with self.assertRaisesRegex(ValueError, "finite before variant masking"):
            mask_stage5_features(invalid, "F0")


class Stage5ControllerTest(unittest.TestCase):
    def test_every_variant_uses_identical_parameter_shapes_and_count(self) -> None:
        signatures = []
        counts = []
        for _variant in STAGE5_VARIANTS:
            model = Stage5SpatialController(width=4)
            signatures.append(tuple((name, tuple(value.shape)) for name, value in model.state_dict().items()))
            counts.append(sum(parameter.numel() for parameter in model.parameters()))
        self.assertTrue(all(signature == signatures[0] for signature in signatures))
        self.assertEqual(len(set(counts)), 1)

    def test_paired_seed_produces_bit_identical_initial_states(self) -> None:
        torch.manual_seed(41)
        first = Stage5SpatialController(width=4)
        torch.manual_seed(41)
        second = Stage5SpatialController(width=4)
        for key, value in first.state_dict().items():
            self.assertTrue(torch.equal(value, second.state_dict()[key]), key)

    def test_zero_head_returns_exact_baseline_for_every_variant(self) -> None:
        model = Stage5SpatialController(width=4, collar_width=2)
        self.assertEqual(int(torch.count_nonzero(model.head.weight)), 0)
        self.assertEqual(int(torch.count_nonzero(model.head.bias)), 0)
        features = _features()
        s2, s4 = _proposals()
        for variant in STAGE5_VARIANTS:
            kwargs = {}
            if variant in {"A2P", "A24P"}:
                kwargs["s2_proposal"] = s2
            if variant == "A24P":
                kwargs["s4_proposal"] = s4
            output = model(features, variant, **kwargs)
            self.assertEqual(int(torch.count_nonzero(output.raw_head)), 0, variant)
            self.assertEqual(int(torch.count_nonzero(output.requested_delta)), 0, variant)
            if output.alpha_s2 is not None:
                self.assertEqual(int(torch.count_nonzero(output.alpha_s2)), 0, variant)
            if output.alpha_s4 is not None:
                self.assertEqual(int(torch.count_nonzero(output.alpha_s4)), 0, variant)

    def test_six_head_channels_have_disjoint_fixed_semantics(self) -> None:
        self.assertEqual(STAGE5_FREE_RESIDUAL_HEAD, slice(0, 3))
        self.assertEqual(STAGE5_S2_ATTENUATION_HEAD, 3)
        self.assertEqual(STAGE5_S4_ATTENUATION_HEAD, 4)
        self.assertEqual(STAGE5_RESERVED_HEAD, 5)

        model = Stage5SpatialController(width=4, collar_width=2)
        with torch.no_grad():
            model.head.bias[STAGE5_RESERVED_HEAD] = 100.0
        output = model(_features(), "F0")
        self.assertEqual(int(torch.count_nonzero(output.raw_head[:, STAGE5_RESERVED_HEAD])), 0)
        output.requested_delta.sum().backward()
        self.assertIsNotNone(model.head.bias.grad)
        assert model.head.bias.grad is not None
        self.assertEqual(float(model.head.bias.grad[STAGE5_RESERVED_HEAD]), 0.0)

    def test_free_policy_is_bounded_tapered_and_does_not_read_search_scale(self) -> None:
        model = Stage5SpatialController(width=4, free_residual_limit_voxels=1.25, collar_width=2)
        with torch.no_grad():
            model.head.bias[:3].copy_(torch.tensor((0.5, -0.75, 2.0)))

        left = _features(seed=1)
        right = left.clone()
        right[:, 5:] = torch.randn_like(right[:, 5:]) * 1_000.0
        left_output = model(left, "F0").requested_delta
        right_output = model(right, "F0").requested_delta
        torch.testing.assert_close(left_output, right_output, atol=0, rtol=0)
        self.assertLessEqual(float(left_output.detach().abs().max()), 1.25)
        self.assertEqual(int(torch.count_nonzero(left_output[:, :, 0])), 0)
        self.assertEqual(int(torch.count_nonzero(left_output[:, :, -1])), 0)
        self.assertEqual(int(torch.count_nonzero(left_output[:, :, :, 0])), 0)
        self.assertEqual(int(torch.count_nonzero(left_output[:, :, :, -1])), 0)
        self.assertEqual(int(torch.count_nonzero(left_output[:, :, :, :, 0])), 0)
        self.assertEqual(int(torch.count_nonzero(left_output[:, :, :, :, -1])), 0)

    def test_a2_attenuates_the_full_resolution_proposal_without_stride_rescaling(self) -> None:
        model = Stage5SpatialController(width=4, collar_width=2)
        with torch.no_grad():
            model.head.bias[STAGE5_S2_ATTENUATION_HEAD] = 2.0
        features = _features()
        s2, _ = _proposals()
        output = model(features, "A2P", s2_proposal=s2)
        self.assertIsNotNone(output.alpha_s2)
        assert output.alpha_s2 is not None
        self.assertTrue(bool(((output.alpha_s2 >= 0.0) & (output.alpha_s2 <= 1.0)).all()))
        expected = identity_collar(s2, width=2)
        torch.testing.assert_close(output.requested_delta, expected, atol=0, rtol=0)

    def test_a24_is_a_convex_no_update_s2_s4_mixture(self) -> None:
        model = Stage5SpatialController(width=4, collar_width=2)
        with torch.no_grad():
            model.head.bias[STAGE5_S2_ATTENUATION_HEAD] = 2.0
            model.head.bias[STAGE5_S4_ATTENUATION_HEAD] = 2.0
        features = _features()
        s2, s4 = _proposals()
        output = model(features, "A24P", s2_proposal=s2, s4_proposal=s4)
        self.assertIsNotNone(output.alpha_s2)
        self.assertIsNotNone(output.alpha_s4)
        assert output.alpha_s2 is not None and output.alpha_s4 is not None
        self.assertTrue(bool((output.alpha_s2 >= 0.0).all()))
        self.assertTrue(bool((output.alpha_s4 >= 0.0).all()))
        self.assertTrue(bool((output.alpha_s2 + output.alpha_s4 <= 1.0).all()))
        expected = identity_collar(0.5 * s2 + 0.5 * s4, width=2)
        torch.testing.assert_close(output.requested_delta, expected, atol=0, rtol=0)

    def test_a_policies_require_finite_physical_proposals(self) -> None:
        model = Stage5SpatialController(width=4)
        features = _features()
        with self.assertRaisesRegex(ValueError, "s2_proposal is required"):
            model(features, "A2P")
        invalid, s4 = _proposals()
        invalid[:, :, 0, 0, 0] = torch.inf
        with self.assertRaisesRegex(ValueError, "s2_proposal must be finite"):
            model(features, "A24P", s2_proposal=invalid, s4_proposal=s4)

    def test_physical_output_policy_remains_fp32_under_autocast(self) -> None:
        model = Stage5SpatialController(width=4, collar_width=2)
        features = _features()
        s2, _ = _proposals()
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            output = model(features, "A2P", s2_proposal=s2)
        self.assertEqual(output.raw_head.dtype, torch.float32)
        self.assertEqual(output.requested_delta.dtype, torch.float32)

    def test_straight_through_attenuation_has_live_gradient_at_exact_zero(self) -> None:
        features = _features()
        s2, s4 = _proposals()
        for variant in ("A2P", "A24P"):
            model = Stage5SpatialController(width=4, collar_width=2)
            kwargs = {"s2_proposal": s2}
            if variant == "A24P":
                kwargs["s4_proposal"] = s4
            output = model(features, variant, **kwargs)
            self.assertEqual(int(torch.count_nonzero(output.requested_delta)), 0)
            output.requested_delta.sum().backward()
            self.assertIsNotNone(model.head.bias.grad)
            assert model.head.bias.grad is not None
            self.assertNotEqual(float(model.head.bias.grad[STAGE5_S2_ATTENUATION_HEAD]), 0.0)
            if variant == "A24P":
                self.assertNotEqual(float(model.head.bias.grad[STAGE5_S4_ATTENUATION_HEAD]), 0.0)


if __name__ == "__main__":
    unittest.main()
