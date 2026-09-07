from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch
import torch.nn.functional as F

from experiments.stage5 import losses, runtime
from experiments.stage5.checkpoints import capture_rng_state
from tools.analysis import diagnose_stage5_amp as amp
from tools.analysis.stage5 import ncc_diagnostic as ncc
from utils import NCCVxm


def centered_oracle(first, second, width):
    windows = []
    for value in (first.double(), second.double()):
        value = F.pad(value, [width // 2] * 6)
        for dimension in (2, 3, 4):
            value = value.unfold(dimension, width, 1)
        windows.append(value - value.mean(dim=(-3, -2, -1), keepdim=True))
    x, y = windows
    cross = (x * y).sum(dim=(-3, -2, -1))
    vi = x.square().sum(dim=(-3, -2, -1)).clamp_min(1e-5)
    vj = y.square().sum(dim=(-3, -2, -1)).clamp_min(1e-5)
    return -(cross.square() / (vi * vj)).mean()


class NCCReferenceTest(unittest.TestCase):
    def test_reference_matches_independently_centered_windows_and_gradients(self):
        generator = torch.Generator().manual_seed(491)
        for offset, noise in ((0.0, 1.0), (-1.23, 0.0), (-1.23, 1e-4), (20.0, 0.1)):
            x = (torch.randn(1, 1, 9, 10, 11, generator=generator) * noise + offset).requires_grad_()
            y = torch.randn(x.shape, generator=generator) * noise + offset * 1.3
            actual = ncc.ReferenceNCC(win=(7, 7, 7))(x, y)
            expected = centered_oracle(x, y, 7)
            observed_gradient = torch.autograd.grad(actual, x)[0]
            expected_gradient = torch.autograd.grad(expected, x)[0]
            self.assertAlmostEqual(float(actual.detach()), float(expected.detach()), places=6)
            torch.testing.assert_close(observed_gradient, expected_gradient, rtol=2e-4, atol=1e-7)

    def test_box_sum_and_adjoint_match_full_convolution(self):
        generator = torch.Generator().manual_seed(814)
        x = torch.randn(1, 1, 5, 6, 7, dtype=torch.float64, generator=generator, requires_grad=True)
        weight = torch.randn(x.shape, dtype=torch.float64, generator=generator)
        kernel = torch.ones(1, 1, 3, 5, 7, dtype=x.dtype)
        actual = ncc._BoxSum.apply(x, (3, 5, 7))
        expected = F.conv3d(x, kernel, padding=(1, 2, 3))
        torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)
        torch.testing.assert_close(
            torch.autograd.grad((actual * weight).sum(), x)[0],
            torch.autograd.grad((expected * weight).sum(), x)[0],
            atol=1e-12,
            rtol=1e-12,
        )

    def test_constant_input_reproduces_legacy_defect_and_worst_window_explains_it(self):
        x = torch.full((1, 1, 24, 24, 24), -1.23)
        y = torch.full_like(x, -1.45)
        result = ncc.compare_ncc(x, y, (7, 7, 7))
        legacy = result["fp32_backend_default"]
        self.assertLess(legacy["loss"], -100)
        self.assertEqual(legacy["production_loss_absolute_difference"], 0)
        self.assertGreater(legacy["cc_above_one_plus_1e_6"], 0)
        self.assertGreater(legacy["worst_window"]["reported_cc"], 1)
        self.assertLessEqual(legacy["worst_window"]["centered_fp64_cc"], 1)
        self.assertTrue(-1 <= result["fp64_reference"]["loss"] <= 0)
        self.assertGreater(legacy["gradient_difference_from_fp64"]["max_abs"], 0)
        json.dumps(result, allow_nan=False)

    def test_tf32_context_is_scoped_even_on_exception(self):
        previous = (torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32)
        with self.assertRaisesRegex(RuntimeError, "intentional"), ncc.ieee_convolutions(True):
            self.assertFalse(torch.backends.cudnn.allow_tf32)
            self.assertFalse(torch.backends.cuda.matmul.allow_tf32)
            raise RuntimeError("intentional")
        self.assertEqual((torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32), previous)

    def test_reference_does_not_change_the_production_loss_class(self):
        self.assertIs(losses.NCCVxm, NCCVxm)
        with mock.patch.object(losses, "NCCVxm", ncc.ReferenceNCC):
            self.assertIs(losses.NCCVxm, ncc.ReferenceNCC)
        self.assertIs(losses.NCCVxm, NCCVxm)

    def test_reference_rejects_unsupported_inputs(self):
        x = torch.zeros(1, 1, 5, 5, 5)
        for window in ((2, 3, 3), (0, 3, 3), (3, 3)):
            with self.assertRaises(ValueError):
                ncc.ReferenceNCC(win=window)(x, x)

    def test_overflow_is_reported_as_json_null_and_counts(self):
        x = torch.full((1, 1, 5, 5, 5), 1e20)
        result = ncc.compare_ncc(x, x, (3, 3, 3))
        legacy = result["fp32_backend_default"]
        self.assertIsNone(legacy["loss"])
        self.assertGreater(legacy["cc"]["nonfinite"], 0)
        self.assertFalse(legacy["loss_within_mathematical_range_1e_6"])
        json.dumps(result, allow_nan=False)


class NCCReplayTest(unittest.TestCase):
    def test_reference_report_cannot_be_substituted(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "F0.json"
            path.write_text('{"status":"DIAGNOSTIC_COMPLETE"}', encoding="utf-8")
            args = SimpleNamespace(ncc_audit=True, reference_report=path, variant="F0")
            with self.assertRaisesRegex(RuntimeError, "exact reviewed"):
                amp.load_ncc_reference(args)
        self.assertIsNone(amp.load_ncc_reference(SimpleNamespace(ncc_audit=False)))

    def test_wrong_pair_is_rejected_before_ncc_measurement(self):
        args = SimpleNamespace()
        report = {"source_hashes": {}, "pair": {"pair_id": "wrong"}}
        reference = {"source_hashes": {}, "pair": {"pair_id": "right"}}
        with mock.patch.object(ncc, "audit_pair") as audit, self.assertRaisesRegex(RuntimeError, "pair"):
            amp.audit_ncc_failure(args, report, None, None, None, None, reference)
        audit.assert_not_called()

    def test_new_replay_records_historical_state_mismatch_without_hiding_it(self):
        report = {
            key: "same"
            for key in (
                "pair",
                "pair_index_one_based",
                "successful_in_memory_updates",
                "pair_schedule_sha256",
                "initial_controller_state_sha256",
                "protocol_sha256",
                "data_contract_sha256",
            )
        }
        reference = dict(report, failing_controller_state_sha256="old")
        report["source_hashes"] = reference["source_hashes"] = {}
        report["failing_controller_state_sha256"] = "new"
        with mock.patch.object(ncc, "audit_pair", return_value={"status": "NCC_AUDIT_COMPLETE"}):
            amp.audit_ncc_failure(SimpleNamespace(), report, None, None, None, None, reference)
        self.assertFalse(report["historical_controller_state_match"])

    def test_replaced_u0_with_consistent_sidecars_is_rejected_against_historical_report(self):
        report = {"source_hashes": {"u0/seed_0/last.pth": "replacement"}}
        reference = {"source_hashes": {"u0/seed_0/last.pth": "original"}}
        with mock.patch.object(ncc, "audit_pair") as audit, self.assertRaisesRegex(RuntimeError, "source bytes"):
            amp.audit_ncc_failure(SimpleNamespace(), report, None, None, None, None, reference)
        audit.assert_not_called()

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required for production controller")
    def test_real_controller_audit_keeps_weights_and_optimizer_unchanged(self):
        config = runtime.ControllerTrainingConfig()
        controller = runtime.build_stage5_controller(config).cuda()
        optimizer = torch.optim.AdamW(controller.parameters())
        features = torch.zeros(1, 71, 16, 16, 16, device="cuda")
        proposal = torch.zeros(1, 3, 16, 16, 16, device="cuda")
        image = torch.randn(1, 1, 16, 16, 16, device="cuda")
        inputs = runtime._ControllerPairInputs(
            proposal,
            proposal,
            (features, proposal, proposal, image, image),
            (features, proposal, proposal, image, image),
            0.0,
        )
        base_runner = SimpleNamespace(model=lambda first, _second, **_kwargs: (first, proposal))
        step = SimpleNamespace(
            controller=controller,
            optimizer=optimizer,
            variant="F0",
            config=config,
            device=torch.device("cuda"),
            store=None,
            base_runner=base_runner,
        )
        weights, opt = copy.deepcopy(controller.state_dict()), copy.deepcopy(optimizer.state_dict())
        snapshots = []
        with (
            mock.patch.object(optimizer, "step", side_effect=AssertionError("Must not update")),
            mock.patch.object(runtime, "_tensor_image", return_value=image),
        ):
            result = ncc.audit_pair(
                step,
                inputs,
                {"subject_a": "a", "subject_b": "b"},
                capture_rng_state(),
                probe_fn=amp.probe,
                save=lambda result: snapshots.append(result["status"]),
            )
        self.assertEqual(result["status"], "NCC_AUDIT_COMPLETE")
        self.assertFalse(result["training_validated"])
        self.assertEqual(len(result["controller_images"]), 2)
        self.assertEqual(len(result["u0_endpoint_seed0"]), 2)
        self.assertEqual(len(result["controller_probes"]), 4)
        self.assertTrue(all(p["status"] == "FINITE" for p in result["controller_probes"][1:]))
        self.assertEqual(optimizer.state_dict(), opt)
        for name, tensor in controller.state_dict().items():
            self.assertTrue(torch.equal(tensor, weights[name]))
        self.assertIs(losses.NCCVxm, NCCVxm)
        json.dumps(result, allow_nan=False)


if __name__ == "__main__":
    unittest.main()
