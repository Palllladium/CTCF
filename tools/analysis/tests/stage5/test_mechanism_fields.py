from __future__ import annotations

import copy
import json
import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from experiments.stage5 import losses, runtime
from experiments.stage5.checkpoints import capture_rng_state
from tools.analysis.stage5 import mechanism_fields as fields
from tools.analysis.stage5.precision_probes import _state_equal, backend_flags


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required for field precision audit")
class FieldAuditTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def setUp(self):
        torch.manual_seed(2909)
        model = torch.nn.Linear(1, 1).cuda()
        self.step = SimpleNamespace(
            controller=model,
            device=torch.device("cuda"),
            variant="F0",
            config=runtime.ControllerTrainingConfig(),
        )
        value = torch.ones(1, 3, 4, 4, 4, device="cuda")
        self.inputs = runtime._ControllerPairInputs(value, value, (value,) * 5, (value,) * 5, 0.0)

    @staticmethod
    def _controlled_fields(_step, _inputs, *, autocast, tf32):
        value = torch.ones(1, 3, 4, 4, 4) * (2.0 if autocast else 1.0)
        return (value, value * 0.75), backend_flags()

    @staticmethod
    def _quadratic(*args, config):
        first, second = args[-2:]
        # Deliberately violate the production precision boundary to prove that
        # the diagnostic distinguishes it from a changed evaluation point.
        factor = 1.1 if torch.is_autocast_enabled("cuda") else 1.0
        total = factor * (first.square().mean() + second.square().mean())
        return total, {"loss": float(total.detach())}

    def test_separates_field_shift_from_same_point_arithmetic(self):
        with (
            mock.patch.object(fields, "_capture_deltas", side_effect=self._controlled_fields),
            mock.patch.object(losses, "controller_objective", side_effect=self._quadratic),
        ):
            report = fields.run_field_audit(self.step, self.inputs)
        self.assertEqual(report["status"], "COMPLETE")
        point = report["field_points"]["fp16_tf32_off"]
        self.assertAlmostEqual(point["evaluation_point_effect"]["forward"]["relative_l2"], 1.0, places=6)
        self.assertAlmostEqual(point["field_shift_vs_strict_controller"]["forward"]["relative_l2"], 1.0, places=6)
        arithmetic = point["evaluations"]["fp16_tf32_off"]["same_point_arithmetic_effect"]
        self.assertAlmostEqual(arithmetic["forward"]["relative_l2"], 0.1, places=6)
        self.assertEqual(point["repeat"]["same_point_repeat_difference"]["forward"]["absolute_l2"], 0.0)
        self.assertFalse(report["training_validated"])
        json.dumps(report, allow_nan=False)

    def test_restores_rng_backend_training_flags_weights_and_existing_gradients(self):
        controller = self.step.controller
        controller.train(False)
        for parameter in controller.parameters():
            parameter.grad = torch.randn_like(parameter)
        weights = copy.deepcopy(controller.state_dict())
        gradients = [parameter.grad.clone() for parameter in controller.parameters()]
        rng, flags = capture_rng_state(), backend_flags()

        def capture(*args, **kwargs):
            torch.rand(8, device="cuda")
            return self._controlled_fields(*args, **kwargs)

        with (
            mock.patch.object(fields, "_capture_deltas", side_effect=capture),
            mock.patch.object(losses, "controller_objective", side_effect=self._quadratic),
        ):
            report = fields.run_field_audit(self.step, self.inputs)
        self.assertTrue(all(report["state_preserved"].values()))
        self.assertTrue(_state_equal(weights, controller.state_dict()))
        self.assertTrue(_state_equal(rng, capture_rng_state()))
        self.assertEqual(flags, backend_flags())
        self.assertFalse(controller.training)
        for actual, expected in zip(controller.parameters(), gradients, strict=True):
            torch.testing.assert_close(actual.grad, expected, rtol=0, atol=0)

    def test_mutation_is_detected_and_restored(self):
        state = copy.deepcopy(self.step.controller.state_dict())

        def mutate(*args, **kwargs):
            with torch.no_grad():
                next(self.step.controller.parameters()).add_(1.0)
            return self._controlled_fields(*args, **kwargs)

        with (
            mock.patch.object(fields, "_capture_deltas", side_effect=mutate),
            mock.patch.object(losses, "controller_objective", side_effect=self._quadratic),
        ):
            report = fields.run_field_audit(self.step, self.inputs)
        self.assertEqual(report["status"], "STATE_PRESERVATION_ERROR")
        self.assertFalse(report["state_preserved"]["model_during_audit"])
        self.assertTrue(_state_equal(state, self.step.controller.state_dict()))

    def test_oom_is_incomplete_and_restores_backend_flags(self):
        previous = backend_flags()
        with (
            mock.patch.object(fields, "_capture_deltas", side_effect=self._controlled_fields),
            mock.patch.object(losses, "controller_objective", side_effect=RuntimeError("CUDA out of memory")),
        ):
            report = fields.run_field_audit(self.step, self.inputs)
        self.assertEqual(report["status"], "INCOMPLETE")
        self.assertEqual(report["field_points"]["fp32_tf32_off"]["evaluations"]["fp16_tf32_on"]["status"], "OOM")
        self.assertEqual(backend_flags(), previous)
        json.dumps(report, allow_nan=False)

    def test_save_exception_also_restores_rng(self):
        rng = capture_rng_state()

        def fail_save(_report):
            raise RuntimeError("save failure")

        def capture(*args, **kwargs):
            torch.rand(5, device="cuda")
            return self._controlled_fields(*args, **kwargs)

        with (
            mock.patch.object(fields, "_capture_deltas", side_effect=capture),
            self.assertRaisesRegex(RuntimeError, "save failure"),
        ):
            fields.run_field_audit(self.step, self.inputs, save=fail_save)
        self.assertTrue(_state_equal(rng, capture_rng_state()))

    def test_actual_controller_and_objective_full_matrix(self):
        from tools.analysis.stage5.mechanism_report import field_audit_errors

        config = runtime.ControllerTrainingConfig()
        self.step.controller = runtime.build_stage5_controller(config).cuda()
        self.step.controller.train()
        shape = (16, 16, 16)
        features = torch.randn(1, 71, *shape, device="cuda") * 0.1
        proposal = torch.zeros(1, 3, *shape, device="cuda")
        first = torch.randn(1, 1, *shape, device="cuda")
        second = first + 0.05 * torch.randn_like(first)
        inputs = runtime._ControllerPairInputs(
            proposal,
            proposal,
            (features, proposal, proposal, first, second),
            (features, proposal, proposal, second, first),
            0.0,
        )
        result = fields.run_field_audit(self.step, inputs)
        self.assertEqual(result["status"], "COMPLETE", result)
        self.assertEqual(field_audit_errors(result), [])
        self.assertEqual(len(result["controller_outputs"]), 4)
        self.assertEqual(len(result["field_points"]), 4)
        self.assertTrue(all(result["state_preserved"].values()))
        for point in result["field_points"].values():
            self.assertEqual(len(point["evaluations"]), 4)
            self.assertEqual(point["repeat"]["status"], "FINITE")
            for observation in point["evaluations"].values():
                self.assertEqual(observation["status"], "FINITE")
                self.assertFalse(observation["backend_flags"]["matmul_allow_tf32"])
                self.assertEqual(observation["backend_flags"]["float32_matmul_precision"], "highest")
        json.dumps(result, allow_nan=False)


if __name__ == "__main__":
    unittest.main()
