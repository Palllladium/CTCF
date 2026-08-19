from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch
import torch.nn.functional as F

from tools.analysis.run_artifacts import sha256_file
from tools.analysis.run_search_gate_c0 import _run_case, _summarise
from tools.analysis.transactional_search import (
    OFFSETS,
    ZERO_OFFSET_INDEX,
    CandidateScreen,
    _local_ncc_map,
    build_proposal,
    commit_exact_candidate,
    geometry_mask,
    load_flow_npz,
    phi_to_psi_displacement,
    psi_to_phi_displacement,
    sample_at_psi,
    save_flow_npz_atomic,
    voxel_grid_like,
)


class TransactionalSearchTest(unittest.TestCase):
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
        second = voxel_grid_like(torch.ones((1, 3, 7, 8, 9)))
        self.assertEqual(first.data_ptr(), second.data_ptr())

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

    def test_no_update_is_not_an_execution_failure(self) -> None:
        rows = [
            {
                "branch": branch,
                "baseline_dice": 0.5,
                "accepted_dice": 0.5,
                "exact_status": "CERTIFIED",
                "action": "ROLLBACK",
                "rollback_byte_identical": True,
            }
            for branch in ("mind_soft", "mind_hard", "intensity_soft", "mind_reversed")
        ]
        summary = _summarise(rows, "smoke")
        self.assertEqual(summary["execution_integrity_status"], "PASS")
        self.assertEqual(summary["search_update_status"], "NO_ACCEPTED_UPDATE")

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

    def test_compact_case_pipeline_finishes_without_retaining_fields(self) -> None:
        class Dataset:
            def __getitem__(self, _: int):
                generator = torch.Generator().manual_seed(23)
                image = torch.randn((1, 9, 10, 11), generator=generator)
                segmentation = torch.ones((1, 9, 10, 11), dtype=torch.long)
                return image, image.clone(), segmentation, segmentation.clone()

        class Adapter:
            @staticmethod
            def forward(model, moving, fixed, amp=True):
                del model, fixed, amp
                return torch.zeros((1, 3, *moving.shape[-3:]), dtype=moving.dtype, device=moving.device)

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            input_path = root / "p_fixture.pkl"
            input_path.write_bytes(b"fixture")
            rows = _run_case(
                index=0,
                path=str(input_path),
                dataset=Dataset(),
                adapter=Adapter(),
                model=None,
                device=torch.device("cpu"),
                labels=(1,),
                stage_dir=root,
                keep_fields=False,
            )
            self.assertEqual(len(rows), 4)
            self.assertTrue(all(row["exact_status"] == "CERTIFIED" for row in rows))
            self.assertFalse((root / "cases" / "fixture" / "work").exists())
            self.assertTrue((root / "cases" / "fixture" / "case_complete.json").is_file())


if __name__ == "__main__":
    unittest.main()
