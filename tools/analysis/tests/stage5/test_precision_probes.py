from __future__ import annotations

import copy
import json
import unittest
from types import SimpleNamespace
from unittest import mock

import torch
import torch.nn.functional as F

from experiments.stage5 import losses, runtime
from experiments.stage5.checkpoints import capture_rng_state
from experiments.stage5.ncc import ControllerNCC, _BoxSum
from tools.analysis.stage5 import precision_probes as probes


class IndependentNCCTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_centered_reference_checks_values_and_both_input_gradients(self):
        generator = torch.Generator().manual_seed(2309)
        for offset, noise in ((0.0, 1.0), (-1.23, 0.0), (-1.23, 1e-4), (20.0, 0.1)):
            first = torch.randn(1, 1, 7, 8, 9, generator=generator) * noise + offset
            second = torch.randn(first.shape, generator=generator) * noise + offset * 1.3
            result = probes.audit_ncc_crop(first, second, (3, 5, 7))
            self.assertEqual(result["status"], "PASS", result)
            self.assertEqual(len(result["gradient_comparisons"]), 2)
            json.dumps(result, allow_nan=False)

    def test_independent_reference_detects_wrong_custom_adjoint(self):
        generator = torch.Generator().manual_seed(23)
        first = torch.randn(1, 1, 7, 7, 7, generator=generator)
        second = torch.randn(first.shape, generator=generator)
        original = _BoxSum.backward

        def wrong_backward(ctx, gradient):
            value, _ = original(ctx, gradient)
            return value * 1.1, None

        with mock.patch.object(_BoxSum, "backward", wrong_backward):
            result = probes.audit_ncc_crop(first, second, (3, 3, 3))
        self.assertEqual(result["status"], "FAIL")
        self.assertLess(result["value_absolute_error"], 2e-6)
        self.assertGreater(result["gradient_comparisons"][0]["relative_l2"], 0.09)

    def test_explicit_reference_cannot_allocate_full_volume_windows(self):
        value = torch.zeros(1, 1, 32, 32, 32)
        with self.assertRaisesRegex(ValueError, "small crops"):
            probes.centered_ncc_reference(value, value, (7, 7, 7))

    def test_gradient_screening_is_safe_near_zero_and_detects_wrong_direction(self):
        zero = probes.gradient_difference(torch.zeros(4), torch.zeros(4))
        self.assertTrue(zero["within_heuristic_tolerance"])
        self.assertIsNone(zero["relative_l2"])
        self.assertIsNone(zero["cosine"])
        wrong = probes.gradient_difference(-torch.ones(4), torch.ones(4))
        self.assertFalse(wrong["within_heuristic_tolerance"])
        self.assertEqual(wrong["cosine"], -1.0)
        bad = probes.gradient_difference(torch.tensor([float("inf")]), torch.ones(1))
        self.assertEqual(bad["status"], "NONFINITE")
        json.dumps([zero, wrong, bad], allow_nan=False)

    def test_tf32_scope_restores_flags_on_failure(self):
        original = probes.backend_flags()
        with self.assertRaisesRegex(RuntimeError, "sentinel"), probes.precision_context(True):
            self.assertFalse(torch.backends.cudnn.allow_tf32)
            self.assertFalse(torch.backends.cuda.matmul.allow_tf32)
            raise RuntimeError("sentinel")
        self.assertEqual(probes.backend_flags(), original)
        with probes.precision_context(False):
            self.assertEqual(probes.backend_flags(), original)


class IncompleteAuditTest(unittest.TestCase):
    def setUp(self):
        model = torch.nn.Linear(1, 1)
        self.step = SimpleNamespace(
            controller=model,
            optimizer=torch.optim.SGD(model.parameters(), lr=0.1),
            device=torch.device("cpu"),
            config=runtime.ControllerTrainingConfig(),
        )
        value = torch.zeros(1, 1, 3, 3, 3)
        self.inputs = SimpleNamespace(psi_ab=value, psi_ba=value, tensors_ab=(value,), tensors_ba=(value,))

    @staticmethod
    def _probe_result(*_args, fp32, scale, reference_status="FINITE"):
        return (
            {
                "mode": "fp32_strict" if fp32 else "fp16",
                "scale": scale,
                "status": reference_status if fp32 else "FINITE",
            },
            {},
            {},
            {},
        )

    def test_fp32_oom_before_ncc_capture_is_incomplete_not_bad_math(self):
        def reference_oom(*args, **kwargs):
            return self._probe_result(*args, **kwargs, reference_status="OOM")

        with (
            mock.patch.object(probes, "_run_probe", side_effect=reference_oom),
            mock.patch.object(probes, "audit_ncc_crop", return_value={"status": "PASS"}),
        ):
            result = probes.compare_pair(self.step, self.inputs)
        self.assertEqual(result["status"], "INCOMPLETE")
        self.assertEqual(result["ncc_audit"]["captured_calls"], 0)
        self.assertEqual(result["ncc_audit"]["status"], "ERROR")
        self.assertEqual(result["ncc_audit"]["error_kind"], "OOM")
        self.assertEqual(result["component_gradient_audit"]["status"], "ERROR")
        self.assertEqual(result["component_gradient_audit"]["error_kind"], "OOM")
        self.assertTrue(all(result["state_preserved"].values()))
        json.dumps(result, allow_nan=False)

    def test_crop_oom_after_finite_reference_does_not_become_numerical_failure(self):
        def captured(capture):
            value = self.inputs.psi_ab
            capture.calls = 2
            capture.crops = [({"direction_call": 1}, (value, value), (3, 3, 3))]
            return capture

        with (
            mock.patch.object(probes, "_run_probe", side_effect=self._probe_result),
            mock.patch.object(probes._NCCCapture, "__enter__", captured),
            mock.patch.object(probes._NCCCapture, "__exit__", return_value=False),
            mock.patch.object(probes, "audit_ncc_crop", side_effect=torch.OutOfMemoryError("crop workspace")),
            mock.patch.object(probes, "_component_audit", return_value={"status": "PASS"}),
        ):
            result = probes.compare_pair(self.step, self.inputs)
        self.assertEqual(result["status"], "INCOMPLETE")
        self.assertEqual(result["probes"][0]["status"], "FINITE")
        self.assertEqual(result["ncc_audit"]["status"], "ERROR")
        self.assertEqual(result["ncc_audit"]["error_kind"], "OOM")
        self.assertTrue(all(case["status"] == "OOM" for case in result["ncc_audit"]["cases"]))
        json.dumps(result, allow_nan=False)

    def test_only_measured_mismatch_is_fail_even_when_other_evidence_is_missing(self):
        result = probes._ncc_audit_outcome([{"status": "PASS"}], 0, "FINITE")
        self.assertEqual(result["status"], "ERROR")
        self.assertEqual(result["error_kind"], "INCOMPLETE")
        result = probes._ncc_audit_outcome([{"status": "FAIL"}, {"status": "OOM"}], 2, "FINITE")
        self.assertEqual(result["status"], "FAIL")
        self.assertFalse(result["complete"])


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required for actual controller probes")
class ControllerProbeTest(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(42)
        config = runtime.ControllerTrainingConfig(width=4)
        controller = runtime.build_stage5_controller(config).cuda()
        # Exercise backward through the backbone, not only the zero-initialized head.
        with torch.no_grad():
            controller.head.weight.normal_(std=0.002)
        optimizer = torch.optim.AdamW(controller.parameters())
        scaler = torch.amp.GradScaler("cuda", init_scale=65536.0)
        self.step = SimpleNamespace(
            controller=controller,
            optimizer=optimizer,
            scaler=scaler,
            variant="F0",
            config=config,
            device=torch.device("cuda"),
        )
        shape = (16, 16, 16)
        features = torch.randn(1, 71, *shape, device="cuda") * 0.1
        proposal = torch.zeros(1, 3, *shape, device="cuda")
        first = torch.randn(1, 1, *shape, device="cuda")
        second = first + torch.randn_like(first) * 0.05
        self.inputs = runtime._ControllerPairInputs(
            proposal,
            proposal,
            (features, proposal, proposal, first, second),
            (features, proposal, proposal, second, first),
            0.0,
        )

    def test_actual_pair_preserves_states_and_traces_functional_interpolation(self):
        step = self.step
        # Existing gradients and Adam moments must also survive the diagnostic.
        for parameter in step.controller.parameters():
            parameter.grad = torch.ones_like(parameter)
        step.optimizer.step()
        weights = copy.deepcopy(step.controller.state_dict())
        optimizer = copy.deepcopy(step.optimizer.state_dict())
        scaler = copy.deepcopy(step.scaler.state_dict())
        gradients = {name: parameter.grad.clone() for name, parameter in step.controller.named_parameters()}
        rng, flags = capture_rng_state(), probes.backend_flags()
        interpolate = F.interpolate
        snapshots = []
        with mock.patch.object(step.optimizer, "step", side_effect=AssertionError("no updates allowed")):
            result = probes.compare_pair(step, self.inputs, save=lambda report: snapshots.append(report["status"]))
        self.assertEqual(result["status"], "COMPLETE")
        self.assertEqual(len(result["probes"]), 6)
        self.assertEqual(result["probes"][0]["mode"], "fp32_strict")
        self.assertEqual(result["probes"][0]["status"], "FINITE")
        self.assertEqual(result["ncc_audit"]["status"], "PASS")
        self.assertEqual(result["component_gradient_audit"]["status"], "PASS")
        self.assertTrue(all(result["state_preserved"].values()))
        events = result["probes"][1]["trace"]
        self.assertTrue(
            any(
                event["tensor"] == "functional.interpolate#1.input" and event["stage"] == "scaled_backward"
                for event in events
            )
        )
        self.assertTrue(
            any(
                event["tensor"] == "functional.interpolate#1.input" and event["dtype"] == "torch.float16"
                for event in events
            )
        )
        self.assertTrue(any(event["tensor"] == "functional.interpolate#1.output" for event in events))
        self.assertTrue(any(event["tensor"].endswith(".requested_delta") for event in events))
        self.assertTrue(probes._state_equal(weights, step.controller.state_dict()))
        self.assertTrue(probes._state_equal(optimizer, step.optimizer.state_dict()))
        self.assertTrue(probes._state_equal(scaler, step.scaler.state_dict()))
        self.assertTrue(probes._state_equal(rng, capture_rng_state()))
        for name, parameter in step.controller.named_parameters():
            torch.testing.assert_close(parameter.grad, gradients[name], rtol=0, atol=0)
        self.assertEqual(probes.backend_flags(), flags)
        self.assertIs(F.interpolate, interpolate)
        self.assertIs(losses.ControllerNCC, ControllerNCC)
        self.assertGreaterEqual(len(snapshots), 7)
        self.assertFalse(result["training_validated"])
        json.dumps(result, allow_nan=False)

    def test_oom_and_math_errors_are_distinct_and_hooks_are_removed(self):
        original = F.interpolate
        for error, status in (
            (torch.OutOfMemoryError("synthetic OOM"), "OOM"),
            (FloatingPointError("synthetic invalid NCC"), "MATH_ERROR"),
        ):
            with mock.patch.object(runtime, "_controller_pair_loss", side_effect=error):
                result, *_ = probes._run_probe(self.step, self.inputs, fp32=True, scale=1.0)
            self.assertEqual(result["status"], status)
            self.assertIs(F.interpolate, original)
            json.dumps(result, allow_nan=False)

    def test_unexpected_weight_mutation_is_reported_and_restored(self):
        weights = copy.deepcopy(self.step.controller.state_dict())
        snapshots = []

        def mutate(*_args, **_kwargs):
            with torch.no_grad():
                next(self.step.controller.parameters()).add_(1)
            raise FloatingPointError("sentinel")

        with (
            mock.patch.object(runtime, "_controller_pair_loss", side_effect=mutate),
            self.assertRaisesRegex(RuntimeError, "mutated"),
        ):
            probes.compare_pair(self.step, self.inputs, save=lambda report: snapshots.append(report["status"]))
        self.assertTrue(probes._state_equal(weights, self.step.controller.state_dict()))
        self.assertEqual(snapshots[-1], "STATE_PRESERVATION_ERROR")


if __name__ == "__main__":
    unittest.main()
