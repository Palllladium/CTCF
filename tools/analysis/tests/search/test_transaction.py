from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch
import torch.nn.functional as F

from tools.analysis.run_artifacts import sha256_file
from tools.analysis.search.transaction import (
    OFFSETS,
    ZERO_OFFSET_INDEX,
    CandidateScreen,
    _local_ncc_map,
    build_proposal,
    certified_local_clip_candidate,
    commit_exact_candidate,
    field_change_statistics,
    geometry_mask,
    load_flow_npz,
    ncc_loss_from_normalized,
    phi_to_psi_displacement,
    proposal_support_weights,
    psi_to_phi_displacement,
    sample_at_psi,
    save_flow_npz_atomic,
    smooth_proposal,
    voxel_grid_like,
)


class TransactionalSearchTest(unittest.TestCase):
    def test_proposal_smoothing_is_reusable_immutable_and_pass_counted(self) -> None:
        generator = torch.Generator().manual_seed(101)
        proposal = torch.randn((1, 3, 7, 8, 9), generator=generator)
        snapshot = proposal.clone()
        once = smooth_proposal(proposal, passes=1)
        twice = smooth_proposal(proposal, passes=2)
        manual_twice = smooth_proposal(once, passes=1)
        torch.testing.assert_close(proposal, snapshot, atol=0, rtol=0)
        torch.testing.assert_close(twice, manual_twice, atol=0, rtol=0)
        self.assertFalse(torch.equal(once, twice))
        for invalid in (0, -1, True, 1.5):
            with self.assertRaises(ValueError):
                smooth_proposal(proposal, passes=invalid)  # type: ignore[arg-type]

    def test_offset_order_and_coordinate_sign(self) -> None:
        self.assertEqual(len(OFFSETS), 27)
        self.assertEqual(ZERO_OFFSET_INDEX, 13)
        self.assertEqual(OFFSETS[0], (-1, -1, -1))
        self.assertEqual(OFFSETS[-1], (1, 1, 1))

        shape = (7, 8, 9)
        psi = torch.zeros((1, 3, *shape), dtype=torch.float32)
        ramp = torch.arange(shape[-1], dtype=torch.float32).view(1, 1, 1, 1, -1).expand(1, 1, *shape)
        sampled = sample_at_psi(ramp, psi, offset=(0, 0, 1))
        torch.testing.assert_close(sampled[..., 2:-2], ramp[..., 2:-2] + 1.0, atol=1e-5, rtol=0)

    def test_phi_psi_roundtrip(self) -> None:
        generator = torch.Generator().manual_seed(7)
        phi = torch.randn((1, 3, 7, 8, 9), generator=generator) * 0.05
        recovered = psi_to_phi_displacement(phi_to_psi_displacement(phi))
        torch.testing.assert_close(recovered, phi, atol=2e-6, rtol=0)

    def test_voxel_grid_is_reused_for_equal_contracts(self) -> None:
        first = voxel_grid_like(torch.zeros((1, 3, 7, 8, 9)))
        snapshot = first.clone()
        second = voxel_grid_like(torch.ones((1, 3, 7, 8, 9)))
        self.assertEqual(first.data_ptr(), second.data_ptr())
        phi = torch.zeros((1, 3, 7, 8, 9))
        psi_to_phi_displacement(phi_to_psi_displacement(phi))
        sample_at_psi(torch.zeros((1, 1, 7, 8, 9)), phi)
        torch.testing.assert_close(first, snapshot, atol=0, rtol=0)

    def test_support_weighted_ncc_and_change_statistics_are_well_formed(self) -> None:
        shape = (7, 8, 9)
        generator = torch.Generator().manual_seed(91)
        fixed = torch.randn((1, 1, *shape), generator=generator)
        moving = fixed.clone()
        psi = torch.zeros((1, 3, *shape))
        mask = geometry_mask(shape, 2, fixed.device)
        proposal = torch.zeros_like(psi)
        proposal[:, 2, 3, 3, 3] = 0.25
        weights = proposal_support_weights(proposal, mask)
        loss = ncc_loss_from_normalized(fixed, moving, psi, mask, weights=weights)
        self.assertTrue(torch.isfinite(torch.tensor(loss)))

        output = psi + 0.5 * proposal
        stats = field_change_statistics(psi, proposal, output, mask)
        self.assertAlmostEqual(stats["effective_alpha_mean"], 0.5)
        self.assertAlmostEqual(stats["retained_norm_ratio"], 0.5)
        self.assertEqual(stats["orthogonal_residual_max"], 0.0)

    def test_local_clip_wrapper_checks_precondition_and_boundary(self) -> None:
        shape = (7, 8, 9)
        current = torch.zeros((1, 3, *shape))
        mask = geometry_mask(shape, 2, current.device)
        delta = torch.zeros_like(current)
        output, report = certified_local_clip_candidate(current, delta, mask, work_eps=0.0011)
        torch.testing.assert_close(output, current, atol=0, rtol=0)
        self.assertGreaterEqual(report["output_fast_cert_bound"], 0.0011)

        bad_boundary = delta.clone()
        bad_boundary[:, :, 0, 0, 0] = 0.1
        with self.assertRaisesRegex(RuntimeError, "boundary"):
            certified_local_clip_candidate(current, bad_boundary, mask, work_eps=0.0011)

        unsafe = current.clone()
        unsafe[:, 0, 3, 3, 3] = -4.0
        with self.assertRaisesRegex(RuntimeError, "precondition"):
            certified_local_clip_candidate(unsafe, torch.zeros_like(unsafe), mask, work_eps=0.0011)

    def test_batch_greater_than_one_fails_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "shape"):
            sample_at_psi(torch.zeros((2, 1, 7, 8, 9)), torch.zeros((2, 3, 7, 8, 9)))

    def test_identical_images_produce_a_negligible_soft_proposal(self) -> None:
        generator = torch.Generator().manual_seed(11)
        image = torch.randn((1, 1, 9, 10, 11), generator=generator)
        psi = torch.zeros((1, 3, 9, 10, 11))
        mask = geometry_mask((9, 10, 11), 2, image.device)
        proposal = build_proposal(
            image,
            image.clone(),
            psi,
            mask,
            feature="intensity",
            collar_width=2,
        )
        self.assertLessEqual(float(proposal.displacement.abs().max()), 5e-4)

    def test_reversed_cost_is_converted_to_the_additive_psi_sign(self) -> None:
        shape = (9, 10, 11)
        generator = torch.Generator().manual_seed(77)
        moving_feature = torch.randn((1, 12, *shape), generator=generator)
        psi = torch.zeros((1, 3, *shape))
        fixed_feature = sample_at_psi(moving_feature, psi, offset=(0, 0, 1))
        image = torch.zeros((1, 1, *shape))
        mask = geometry_mask(shape, 2, image.device)

        for orientation in ("target_centered", "reversed"):
            proposal = build_proposal(
                image,
                image,
                psi,
                mask,
                feature="mind",
                orientation=orientation,
                collar_width=2,
                fixed_feature_override=fixed_feature,
                moving_feature_override=moving_feature,
            )
            interior = proposal.hard_displacement[:, :, 3:-3, 3:-3, 3:-3]
            means = interior.mean(dim=(0, 2, 3, 4))
            torch.testing.assert_close(means, torch.tensor([0.0, 0.0, 1.0]), atol=0, rtol=0)

    def test_separable_ncc_matches_registered_full_box_filter(self) -> None:
        generator = torch.Generator().manual_seed(31)
        fixed = torch.randn((1, 1, 9, 10, 11), generator=generator)
        moving = torch.randn((1, 1, 9, 10, 11), generator=generator)
        win, eps = 9, 1e-5
        kernel = torch.ones((1, 1, win, win, win))
        count = float(win**3)
        fixed_sum = F.conv3d(fixed, kernel, padding=win // 2)
        moving_sum = F.conv3d(moving, kernel, padding=win // 2)
        fixed2_sum = F.conv3d(fixed.square(), kernel, padding=win // 2)
        moving2_sum = F.conv3d(moving.square(), kernel, padding=win // 2)
        product_sum = F.conv3d(fixed * moving, kernel, padding=win // 2)
        fixed_mean, moving_mean = fixed_sum / count, moving_sum / count
        cross = product_sum - moving_mean * fixed_sum - fixed_mean * moving_sum + fixed_mean * moving_mean * count
        fixed_var = (fixed2_sum - 2 * fixed_mean * fixed_sum + fixed_mean.square() * count).clamp_min(eps)
        moving_var = (moving2_sum - 2 * moving_mean * moving_sum + moving_mean.square() * count).clamp_min(eps)
        reference = -(cross.square() / (fixed_var * moving_var))
        torch.testing.assert_close(_local_ncc_map(fixed, moving), reference, atol=2e-8, rtol=2e-6)

    def test_rollback_preserves_original_npz_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            initial = root / "initial.npz"
            output = root / "output.npz"
            flow = torch.zeros((1, 3, 5, 6, 7), dtype=torch.float32)
            save_flow_npz_atomic(initial, flow)
            digest = sha256_file(initial)
            outcome = commit_exact_candidate(initial, output, eligible=[])
            self.assertEqual(outcome.status, "ROLLED_BACK")
            self.assertTrue(outcome.rollback_byte_identical)
            self.assertEqual(sha256_file(output), digest)
            torch.testing.assert_close(load_flow_npz(output), flow, atol=0, rtol=0)

    def test_candidate_is_checked_after_save_and_reload(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            initial = root / "initial.npz"
            output = root / "output.npz"
            flow = torch.zeros((1, 3, 5, 6, 7), dtype=torch.float32)
            save_flow_npz_atomic(initial, flow)
            screen = CandidateScreen(
                coefficient=1.0,
                utility=-1.0,
                improvement=0.1,
                tolerance=1e-6,
                cert_bound=1.0,
                utility_passed=True,
                fast_certificate_passed=True,
            )
            outcome = commit_exact_candidate(
                initial,
                output,
                eligible=[screen],
                initial_psi=flow,
                proposal=torch.zeros_like(flow),
            )
            self.assertEqual(outcome.status, "ACCEPTED")
            self.assertEqual(outcome.exact_report["status"], "CERTIFIED")


if __name__ == "__main__":
    unittest.main()
