from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from experiments.stage5.safety import commit_controller_delta
from tools.analysis.search.transaction import (
    certified_local_clip_candidate,
    geometry_mask,
    load_flow_npz,
    save_flow_npz_atomic,
)
from tools.analysis.stage5.work_margin import WORK_MARGIN_POLICY, margin_from_bounds, select_work_margin
from utils.cert_exact import certify_flow_exact
from utils.field import trilinear_cert_bound

EXACT = "tools.analysis.stage5.work_margin.certify_flow_exact"


class WorkMarginTest(unittest.TestCase):
    def setUp(self):
        # A certified source whose smallest Bernstein coefficient lies between the claim and 0.0011.
        self.source = torch.zeros(1, 3, 18, 18, 18)
        self.source[0, 0, 8, 8, 8] = -0.99895

    def test_source_between_claim_and_nominal_is_the_regression_case(self):
        self.assertTrue(certify_flow_exact(self.source, eps="0.001")["certified"])
        self.assertLess(trilinear_cert_bound(self.source), 0.0011)
        with self.assertRaisesRegex(RuntimeError, "precondition failed"):
            certified_local_clip_candidate(
                self.source, torch.zeros_like(self.source), geometry_mask((18, 18, 18), 7, self.source.device)
            )

    def test_near_claim_zero_and_nonzero_transactions_are_certified(self):
        margin = select_work_margin(self.source)
        self.assertGreater(margin["selected_work_eps"], 0.001)
        self.assertLessEqual(margin["selected_work_eps"], margin["source_exact_lower"])
        self.assertLessEqual(margin["selected_work_eps"], margin["source_fast_bound"])
        for amplitude in (0.0, -0.05):
            with self.subTest(amplitude=amplitude), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                path = root / "initial.npz"
                save_flow_npz_atomic(path, self.source)
                before = path.read_bytes()
                delta = torch.zeros_like(self.source)
                delta[0, 0, 8, 8, 8] = amplitude
                result = commit_controller_delta(path, delta, root / "decision")
                self.assertEqual(result.status, "ACCEPTED")
                self.assertTrue(result.returned_exact_report["certified"])
                self.assertEqual(result.clip_report["work_eps"], margin["selected_work_eps"])
                self.assertEqual(result.clip_report["margin_selected_work_eps"], margin["selected_work_eps"])
                self.assertEqual(path.read_bytes(), before)
                if amplitude == 0:
                    self.assertTrue(torch.equal(load_flow_npz(result.returned_path), self.source))

    def test_nominal_candidate_is_bit_identical_to_original_operator(self):
        source = torch.zeros_like(self.source)
        delta = source.clone()
        delta[0, 0, 8, 8, 8] = -2.0
        expected, _ = certified_local_clip_candidate(source, delta, geometry_mask((18, 18, 18), 7, source.device))
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            save_flow_npz_atomic(root / "initial.npz", source)
            with patch(EXACT) as exact:
                result = commit_controller_delta(root / "initial.npz", delta, root / "decision")
            exact.assert_not_called()
            self.assertEqual(result.clip_report["work_eps"], 0.0011)
            self.assertTrue(torch.equal(load_flow_npz(result.candidate_path), expected))

    def test_bad_candidate_still_rolls_back_under_reduced_margin(self):
        folded = self.source.clone()
        folded[0, 0, 8, 8, 8] = -2.0
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / "initial.npz"
            save_flow_npz_atomic(path, self.source)
            with patch("experiments.stage5.safety.certified_local_clip_candidate", return_value=(folded, {})):
                result = commit_controller_delta(path, torch.zeros_like(self.source), root / "decision")
            self.assertEqual(result.status, "ROLLED_BACK")
            self.assertTrue(result.rollback_byte_identical)
            self.assertTrue(result.returned_exact_report["certified"])

    def test_invalid_or_unproved_margin_is_rejected(self):
        for fast, lower in ((float("nan"), 1.0), (0.00105, float("nan")), (0.001, 0.0011), (0.00105, 0.001)):
            with self.subTest(fast=fast, lower=lower), self.assertRaises(RuntimeError):
                margin_from_bounds(fast, lower)
        unresolved = {"status": "UNRESOLVED", "certified": False}
        with patch(EXACT, return_value=unresolved), self.assertRaisesRegex(RuntimeError, "failed exact certification"):
            select_work_margin(self.source)
        at_claim = {"status": "CERTIFIED", "certified": True, "interval_lo_min": 0.001}
        with patch(EXACT, return_value=at_claim), self.assertRaisesRegex(RuntimeError, "no verified working margin"):
            select_work_margin(self.source)

    def test_policy_lowers_the_margin_only_as_far_as_the_source_requires(self):
        self.assertEqual(margin_from_bounds(0.0011), 0.0011)
        self.assertEqual(margin_from_bounds(1.0), 0.0011)
        self.assertEqual(margin_from_bounds(0.00105, 0.00104), 0.00104)
        self.assertEqual(margin_from_bounds(0.00104, 0.00105), 0.00104)
        self.assertEqual(WORK_MARGIN_POLICY["claim_epsilon"], "0.001")


if __name__ == "__main__":
    unittest.main()
