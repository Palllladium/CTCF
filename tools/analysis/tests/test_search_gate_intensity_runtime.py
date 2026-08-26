from __future__ import annotations

import unittest
from unittest.mock import patch

import torch

from tools.analysis.search_gate_intensity_runtime import (
    build_intensity_reach_bank,
    decode_intensity_direction,
    materialize_intensity_candidate,
)


class IntensityRuntimeTest(unittest.TestCase):
    @staticmethod
    def bank_inputs() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        shape = (13, 13, 13)
        generator = torch.Generator().manual_seed(17)
        moving = torch.randn((1, 1, *shape), generator=generator)
        fixed = torch.roll(moving, shifts=(-4, 0, 4), dims=(2, 3, 4))
        initial = torch.zeros((1, 3, *shape))
        generation_mask = torch.zeros((1, 1, *shape), dtype=torch.bool)
        generation_mask[..., 6, 6, 6] = True
        return fixed, moving, initial, generation_mask

    def test_build_and_decode_sparse_intensity_direction(self) -> None:
        fixed, moving, initial, generation_mask = self.bank_inputs()
        bank = build_intensity_reach_bank(
            fixed,
            moving,
            initial,
            generation_mask,
            reach_id="S4",
            cost_id="intensity_s4",
            stride_voxels=4,
            standardization_floor=1e-6,
        )
        direction = decode_intensity_direction(
            bank,
            direction_id="intensity_s4_b0",
            centre_beta=0.0,
            posterior_temperature=1.0,
        )

        self.assertEqual(bank.generation_count, 1)
        self.assertEqual(bank.raw_all_candidates_valid_count, 1)
        self.assertEqual(bank.standardized_informative_count, 1)
        self.assertEqual(bank.volume.offsets[13], (0, 0, 0))
        self.assertEqual(direction.direction_id, "intensity_s4_b0")
        self.assertEqual(direction.stride_voxels, 4)
        self.assertEqual(direction.decoded.displacement.shape, initial.shape)
        self.assertTrue(torch.isfinite(direction.decoded.displacement).all())

    def test_sweeps_change_only_one_clip_call_not_requested_displacement(self) -> None:
        fixed, moving, initial, generation_mask = self.bank_inputs()
        bank = build_intensity_reach_bank(
            fixed,
            moving,
            initial,
            generation_mask,
            reach_id="S4",
            cost_id="intensity_s4",
            stride_voxels=4,
            standardization_floor=1e-6,
        )
        direction = decode_intensity_direction(
            bank,
            direction_id="intensity_s4_b0",
            centre_beta=0.0,
            posterior_temperature=1.0,
        )
        mask = torch.zeros_like(generation_mask)
        mask[..., 2:-2, 2:-2, 2:-2] = True
        rms_reference = torch.zeros_like(initial)
        rms_reference[:, 0, 3:-3, 3:-3, 3:-3] = 0.1

        calls: list[tuple[torch.Tensor, int]] = []

        def clip_once(
            current: torch.Tensor,
            requested: torch.Tensor,
            observed_mask: torch.Tensor,
            *,
            work_eps: float,
            sweeps: int,
        ) -> tuple[torch.Tensor, dict[str, float | int | str]]:
            self.assertIs(observed_mask, mask)
            self.assertEqual(work_eps, 0.0011)
            calls.append((requested.clone(), sweeps))
            return current + requested, {
                "operator": "CERTIFIED_LOCAL_CLIP",
                "work_eps": work_eps,
                "sweeps": sweeps,
            }

        with patch(
            "tools.analysis.search_gate_intensity_runtime.certified_local_clip_candidate",
            side_effect=clip_once,
        ) as clip:
            sweep1 = materialize_intensity_candidate(
                direction,
                initial,
                rms_reference,
                mask,
                candidate_id="s4_a200_w1",
                pre_rms_multiplier=1.0,
                post_rms_amplitude=2.0,
                smoothing_passes=1,
                collar_width=2,
                work_eps=0.0011,
                sweeps=1,
            )
            sweep2 = materialize_intensity_candidate(
                direction,
                initial,
                rms_reference,
                mask,
                candidate_id="s4_a200_w2",
                pre_rms_multiplier=1.0,
                post_rms_amplitude=2.0,
                smoothing_passes=1,
                collar_width=2,
                work_eps=0.0011,
                sweeps=2,
            )

        self.assertEqual(clip.call_count, 2)
        self.assertEqual([sweeps for _, sweeps in calls], [1, 2])
        torch.testing.assert_close(calls[0][0], calls[1][0], atol=0, rtol=0)
        torch.testing.assert_close(sweep1.requested_displacement, sweep2.requested_displacement, atol=0, rtol=0)
        self.assertEqual(sweep1.clip_rms_retention, 1.0)
        self.assertEqual(sweep2.clip_rms_retention, 1.0)
        self.assertEqual(sweep2.operator["sweeps"], 2)

    def test_protocol_parameters_fail_closed(self) -> None:
        fixed, moving, initial, generation_mask = self.bank_inputs()
        with self.assertRaisesRegex(ValueError, "non-empty"):
            build_intensity_reach_bank(
                fixed,
                moving,
                initial,
                generation_mask,
                reach_id="",
                cost_id="intensity_s4",
                stride_voxels=4,
                standardization_floor=1e-6,
            )
        with self.assertRaisesRegex(ValueError, "one of"):
            build_intensity_reach_bank(
                fixed,
                moving,
                initial,
                generation_mask,
                reach_id="S5",
                cost_id="intensity_s5",
                stride_voxels=5,
                standardization_floor=1e-6,
            )


if __name__ == "__main__":
    unittest.main()
