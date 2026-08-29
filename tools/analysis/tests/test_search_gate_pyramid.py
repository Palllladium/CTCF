from __future__ import annotations

import unittest
from typing import ClassVar

import torch

from tools.analysis.search_gate_pyramid import (
    binomial_blur3d,
    blurred_full_resolution_image,
    build_pyramid_direction,
    downsample_image,
    lift_level_vector,
    project_psi_to_level,
)
from utils.field import boundary_nonzero_count, identity_collar

PYRAMID_KWARGS = {
    "full_collar": 7,
    "work_eps": 0.0011,
    "stage_work_eps_decrement": 0.000025,
    "exact_claim_eps": 0.001,
    "standardization_floor": 1e-6,
    "image_std_floor": 1e-6,
    "proposal_multiplier": 1.0,
    "post_smoothing_passes": 1,
    "posterior_temperature": 1.0,
    "centre_beta": 0.0,
    "require_all_candidates_valid": True,
    "stage_clip_sweeps": 1,
}

PYRAMID_CONFIGURATIONS = (
    ("full_resolution", (2, 1), True),
    ("blurred_full_resolution", (2, 1), True),
    ("true_pyramid", (2, 1), True),
    ("full_resolution", (4, 2, 1), True),
    ("blurred_full_resolution", (4, 2, 1), True),
    ("true_pyramid", (4, 2, 1), True),
    ("true_pyramid", (4, 2, 1), False),
)


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

    def test_quarter_blur_control_uses_unit_then_half_grid_spacing(self) -> None:
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
    directions: ClassVar[dict[tuple[str, tuple[int, ...], bool], object]] = {}

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
        for family, factors, rewarp in PYRAMID_CONFIGURATIONS:
            key = (family, factors, rewarp)
            cls.directions[key] = build_pyramid_direction(
                cls.fixed,
                cls.moving,
                cls.initial,
                cls.reference,
                family=family,
                factors=factors,
                rewarp_between_levels=rewarp,
                **PYRAMID_KWARGS,
            )

    def test_every_configuration_builds(self) -> None:
        self.assertEqual(len(self.directions), len(PYRAMID_CONFIGURATIONS))
        for key, direction in self.directions.items():
            self.assertTrue(torch.isfinite(direction.displacement).all(), key)
            self.assertEqual(boundary_nonzero_count(direction.displacement), 0, key)
            self.assertEqual(len(direction.stages), len(direction.factors), key)

    def test_every_configuration_reaches_the_same_final_rms_budget(self) -> None:
        for key, direction in self.directions.items():
            self.assertAlmostEqual(direction.normalized_rms, direction.reference_rms, places=6, msg=str(key))

    def test_true_pyramid_and_full_grid_controls_use_their_declared_grids(self) -> None:
        for (family, factors, _), direction in self.directions.items():
            strides = tuple(stage.stride_voxels for stage in direction.stages)
            shapes = tuple(stage.level_shape for stage in direction.stages)
            if family == "true_pyramid":
                self.assertEqual(strides, (1,) * len(factors), family)
                self.assertEqual(shapes, tuple((32 // factor,) * 3 for factor in factors), family)
            else:
                self.assertEqual(strides, factors, family)
                self.assertEqual(shapes, ((32, 32, 32),) * len(factors), family)

    def test_each_stage_requests_an_equal_share_of_the_source_rms(self) -> None:
        for (family, factors, rewarp), direction in self.directions.items():
            share = direction.reference_rms / len(factors)
            for stage in direction.stages:
                self.assertAlmostEqual(stage.requested_stage_rms, share, places=9, msg=family)
                self.assertLess(stage.realized_stage_rms, direction.reference_rms, family)
                if not rewarp:
                    self.assertAlmostEqual(stage.realized_stage_rms, share, places=5, msg=family)

    def test_rewarped_stages_publish_a_safe_descending_continuation_contract(self) -> None:
        for (family, factors, rewarp), direction in self.directions.items():
            if not rewarp:
                for stage in direction.stages:
                    self.assertIsNone(stage.clip_work_eps)
                    self.assertIsNone(stage.continuation_eps)
                    self.assertIsNone(stage.output_fast_cert_bound)
                continue
            schedule = tuple(
                round(PYRAMID_KWARGS["work_eps"] - index * PYRAMID_KWARGS["stage_work_eps_decrement"], 9)
                for index in range(len(factors))
            )
            self.assertEqual(tuple(stage.clip_work_eps for stage in direction.stages), schedule, family)
            self.assertEqual(
                tuple(stage.continuation_eps for stage in direction.stages),
                (*schedule[1:], PYRAMID_KWARGS["exact_claim_eps"]),
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

    def test_a_pyramid_differs_from_both_matched_full_grid_controls(self) -> None:
        for factors in ((2, 1), (4, 2, 1)):
            pyramid = self.directions[("true_pyramid", factors, True)].displacement
            for family in ("full_resolution", "blurred_full_resolution"):
                control = self.directions[(family, factors, True)].displacement
                self.assertFalse(torch.equal(pyramid, control), f"{family}{factors}")


if __name__ == "__main__":
    unittest.main()
