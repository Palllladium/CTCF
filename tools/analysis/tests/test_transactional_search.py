from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
import torch.nn.functional as F

from tools.analysis.run_artifacts import sha256_file
from tools.analysis.run_search_gate_c0 import _run_case, _summarise
from tools.analysis.run_search_gate_c1 import _build_local_candidate, _operator_rule_rows
from tools.analysis.run_search_gate_c2 import (
    MARGIN_SCHEDULE,
    TRAJECTORIES,
    _row_without_labels,
    _run_case as _run_c2_case,
    _summary_rows,
    _validate_case as _validate_c2_case,
)
from tools.analysis.search_gate_common import (
    CLAIM_EPS,
    WORK_EPS,
    deformation_quality_metrics as _deformation_quality_metrics,
)
from tools.analysis.transactional_search import (
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

    def test_c2_policy_has_fixed_val58_trajectories_and_margins(self) -> None:
        self.assertEqual(len(TRAJECTORIES), 4)
        self.assertEqual(len({item["trajectory_id"] for item in TRAJECTORIES}), 4)
        self.assertEqual(MARGIN_SCHEDULE, (0.0011, 0.001075, 0.00105, 0.001025))
        self.assertTrue(all(value > CLAIM_EPS for value in MARGIN_SCHEDULE))
        self.assertEqual(sum(item["sdlogj_relative_cap"] is not None for item in TRAJECTORIES), 1)

    def test_c2_decision_snapshot_excludes_only_postdecision_label_metrics(self) -> None:
        row = {
            "case_id": "subject_1",
            "action": "ACCEPT",
            "candidate_mind": 0.2,
            "returned_array_sha256": "a" * 64,
            "baseline_dice": 0.8,
            "returned_dice": 0.81,
            "returned_dice_delta": 0.01,
            "returned_sdlogj": 0.1,
            "returned_sdlogj_delta": 0.01,
            "returned_j_leq0_central_percent": 0.0,
            "returned_j_leq0_digital10_percent": 0.0,
            "returned_trilinear_fold_percent_upper_bound": 0.0,
        }
        frozen = _row_without_labels(row)
        self.assertEqual(frozen["candidate_mind"], 0.2)
        self.assertEqual(frozen["returned_array_sha256"], "a" * 64)
        self.assertNotIn("baseline_dice", frozen)
        self.assertNotIn("returned_sdlogj", frozen)

    def test_c2_gate_rejects_a_small_gain_even_when_better_than_c1(self) -> None:
        c1_rows = [
            {
                "case_id": f"case_{index}",
                "dice_delta": "0.0004",
                "sdlogj_delta": "0.0003",
                "final_digital10_percent": "0.002",
            }
            for index in range(58)
        ]
        branch_rows = []
        for trajectory in TRAJECTORIES:
            for index in range(58):
                branch_rows.append(
                    {
                        "case_id": f"case_{index}",
                        "trajectory_id": trajectory["trajectory_id"],
                        "accepted_steps": 1,
                        "final_exact_status": "CERTIFIED",
                        "final_dice": 0.8008,
                        "dice_delta": 0.0008,
                        "paired_dice_advantage_vs_c1": 0.0004,
                        "final_sdlogj": 0.1,
                        "sdlogj_delta": 0.0002,
                        "final_j_leq0_digital10_percent": 0.001,
                    }
                )
        summaries, summary = _summary_rows(branch_rows, c1_rows)
        self.assertTrue(all(not row["eligible_for_test_gate"] for row in summaries))
        self.assertEqual(summary["scientific_status"], "C2_NOT_PROMISING")
        self.assertIsNone(summary["selected_trajectory_id"])

    def test_c2_compact_transaction_reconstructs_before_resume(self) -> None:
        class Dataset:
            def __getitem__(self, _: int):
                generator = torch.Generator().manual_seed(4)
                image = torch.randn((1, 11, 12, 13), generator=generator)
                segmentation = torch.ones((1, 11, 12, 13), dtype=torch.long)
                return image, image.clone(), segmentation, segmentation.clone()

        class Adapter:
            @staticmethod
            def forward(model, moving, fixed, amp=True):
                del model, fixed, amp
                return torch.zeros((1, 3, *moving.shape[-3:]), dtype=moving.dtype)

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "run"
            source = Path(temporary) / "fixture.pkl"
            source.write_bytes(b"fixture")
            input_row = {
                "dataset": "IXI",
                "split": "val",
                "case_id": "fixture",
                "path": str(source),
                "bytes": str(source.stat().st_size),
                "sha256": sha256_file(source),
                "mtime_utc": "fixture",
            }
            c1_row = {
                "case_id": "fixture",
                "final_dice": "1.0",
                "sdlogj_delta": "0.0",
                "final_digital10_percent": "0.0",
            }
            execution = {
                "attempt_id": "fixture",
                "shard_index": 0,
                "physical_gpu": "0",
                "host": "fixture",
                "python": "fixture",
                "torch": "fixture",
                # The tensor computation is CPU-only; the string exercises the production provenance validator.
                "device": "cuda:0",
                "gpu_name": "fixture",
                "seed": 0,
                "deterministic": True,
                "checkpoint_sha256": "a" * 64,
                "checkpoint_load_report": {
                    "strict": True,
                    "missing_keys": [],
                    "allowed_missing_buffers": [],
                    "unexpected_keys": [],
                },
            }
            rows = _run_c2_case(
                index=0,
                input_row=input_row,
                c1_row=c1_row,
                dataset=Dataset(),
                adapter=Adapter(),
                model=None,
                device=torch.device("cpu"),
                labels=(1,),
                root=root,
                contract_sha="b" * 64,
                execution=execution,
            )
            marker = root / "cases" / "fixture" / "case_complete.json"
            payload = json.loads(marker.read_text(encoding="utf-8"))
            reconstructed = _validate_c2_case(
                payload,
                marker,
                "fixture",
                "b" * 64,
                input_row,
                c1_row=c1_row,
            )
            self.assertEqual(len(rows), 16)
            self.assertEqual(reconstructed, rows)
            self.assertFalse((marker.parent / "work").exists())

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

    def test_c1_float32_work_margin_loss_continues_to_claim_check(self) -> None:
        shape = (7, 8, 9)
        current = torch.zeros((1, 3, *shape))
        proposal = torch.zeros_like(current)
        mask = geometry_mask(shape, 2, current.device)
        output_bound = (WORK_EPS + CLAIM_EPS) / 2.0
        report = {
            "output_fast_cert_bound": output_bound,
            "alpha_min": 0.5,
            "alpha_mean": 0.5,
            "alpha_max": 0.5,
        }
        spec = {
            "operator": "certified_local_clip",
            "scale": 1.0,
            "sweeps": 1,
        }
        with patch(
            "tools.analysis.run_search_gate_c1.certified_local_clip_candidate",
            return_value=(current.clone(), report),
        ):
            requested, candidate, observed = _build_local_candidate(spec, current, proposal, mask)
        torch.testing.assert_close(requested, proposal, atol=0, rtol=0)
        torch.testing.assert_close(candidate, current, atol=0, rtol=0)
        self.assertEqual(observed["status"], "COMPLETE_BELOW_WORK_MARGIN")
        self.assertFalse(observed["retained_work_margin_after_float32"])
        self.assertTrue(observed["retained_claim_margin_after_float32"])

    def test_c1_geometry_metrics_separate_measured_counts_from_exact_guarantee(self) -> None:
        field = torch.zeros((1, 3, 7, 8, 9))
        certified = _deformation_quality_metrics(field, exact_certified=True)
        self.assertEqual(certified["j_leq0_central_percent"], 0.0)
        self.assertEqual(certified["j_leq0_digital10_percent"], 0.0)
        self.assertEqual(certified["trilinear_fold_percent_upper_bound"], 0.0)
        self.assertEqual(certified["trilinear_fold_status"], "ZERO_BY_EXACT_CERTIFICATE")

        unverified = _deformation_quality_metrics(field, exact_certified=False)
        self.assertIsNone(unverified["trilinear_fold_percent_upper_bound"])
        self.assertEqual(unverified["trilinear_fold_status"], "NOT_ESTABLISHED")

    def test_c1_rule_summary_reports_absolute_dice_sdlogj_and_folds(self) -> None:
        rows = []
        for accepted, baseline_dice, candidate_dice, baseline_sdlogj, candidate_sdlogj in (
            (True, 0.70, 0.71, 0.20, 0.19),
            (False, 0.80, 0.79, 0.30, 0.31),
        ):
            row = {
                "candidate_id": "candidate",
                "family": "local",
                "feature": "mind",
                "orientation": "target_centered",
                "operator": "certified_local_clip",
                "operator_status": "COMPLETE",
                "scale": 1.0,
                "sweeps": 1,
                "coefficient_index": None,
                "baseline_dice": baseline_dice,
                "candidate_dice": candidate_dice,
                "candidate_dice_delta": candidate_dice - baseline_dice,
                "baseline_sdlogj": baseline_sdlogj,
                "candidate_sdlogj": candidate_sdlogj,
                "candidate_sdlogj_delta": candidate_sdlogj - baseline_sdlogj,
                "baseline_j_leq0_central_percent": 0.0,
                "candidate_j_leq0_central_percent": 0.0,
                "baseline_j_leq0_digital10_percent": 0.0,
                "candidate_j_leq0_digital10_percent": 0.0,
            }
            for rule in ("topology_only", "mind", "ncc9", "support_ncc9", "mind_and_ncc9"):
                row[f"rule_{rule}"] = accepted
            rows.append(row)
        summary = next(row for row in _operator_rule_rows(rows) if row["rule"] == "mind")
        self.assertAlmostEqual(summary["returned_dice_mean"], (0.71 + 0.80) / 2)
        self.assertAlmostEqual(summary["returned_sdlogj_mean"], (0.19 + 0.30) / 2)
        self.assertEqual(summary["returned_j_leq0_digital10_percent_max"], 0.0)
        self.assertEqual(summary["returned_trilinear_fold_percent_upper_bound"], 0.0)

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
