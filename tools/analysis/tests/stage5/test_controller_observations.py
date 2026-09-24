from __future__ import annotations

import copy
import json
import unittest

import torch

from models.CTCF.controller import Stage5ControllerOutput, Stage5SpatialController
from tools.analysis.stage5.controller_observations import observe_controller_output


class ControllerObservationsTest(unittest.TestCase):
    def _forward(self, model: Stage5SpatialController, variant: str) -> Stage5ControllerOutput:
        features = torch.zeros(1, 71, 6, 8, 10)
        s2 = torch.ones(1, 3, 6, 8, 10)
        s4 = 2 * torch.ones_like(s2)
        return model(features, variant, s2_proposal=s2, s4_proposal=s4)

    def test_real_forward_distinguishes_raw_saturation_from_normalized_gates(self) -> None:
        model = Stage5SpatialController(width=4, collar_width=1).eval()
        with torch.no_grad():
            model.head.bias[3] = 7.0
            model.head.bias[4] = 2.0
        with torch.inference_mode():
            output = self._forward(model, "A24P")
            before = output.requested_delta.clone()
            report = observe_controller_output(output, collar_width=model.collar_width)
        self.assertTrue(torch.equal(before, output.requested_delta))
        raw = report["raw_head"]["s2"]["full_volume"]
        self.assertEqual(raw["min"], 7.0)
        self.assertEqual(raw["unit_interval_counts"]["above_one"], 480)
        self.assertEqual(report["alpha"]["s2"]["full_volume"]["mean"], 0.5)
        self.assertEqual(report["alpha"]["s4"]["full_volume"]["mean"], 0.5)
        self.assertEqual(report["alpha_sum"]["full_volume"]["unit_interval_counts"]["at_one"], 480)
        interior = report["requested_delta"]["interior_without_collar"]
        self.assertEqual(interior["count"], 3 * 4 * 6 * 8)
        self.assertEqual(interior["mean"], 1.5)
        self.assertLess(report["requested_delta"]["full_volume"]["mean"], interior["mean"])
        json.dumps(report, allow_nan=False)

    def test_a2p_omits_inactive_s4_and_records_saturated_zero(self) -> None:
        model = Stage5SpatialController(width=4, collar_width=1).eval()
        with torch.no_grad():
            model.head.bias[3] = -4.0
            model.head.bias[4] = 20.0
            report = observe_controller_output(self._forward(model, "A2P"), collar_width=1)
        self.assertEqual(set(report["raw_head"]), {"s2"})
        self.assertEqual(set(report["alpha"]), {"s2"})
        self.assertIsNone(report["alpha_sum"])
        self.assertEqual(report["raw_head"]["s2"]["full_volume"]["unit_interval_counts"]["below_zero"], 480)
        self.assertEqual(report["alpha"]["s2"]["full_volume"]["unit_interval_counts"]["at_zero"], 480)
        self.assertEqual(report["requested_delta"]["full_volume"]["abs_max"], 0.0)

    def test_observation_preserves_model_state_rng_and_backward(self) -> None:
        model = Stage5SpatialController(width=4, collar_width=1).eval()
        with torch.no_grad():
            model.head.bias[3] = 0.25
            model.head.bias[4] = 0.5
        reference = copy.deepcopy(model)
        output = self._forward(model, "A24P")
        state_before = {name: value.clone() for name, value in model.state_dict().items()}
        rng_before = torch.get_rng_state().clone()
        observe_controller_output(output, collar_width=1)
        self.assertTrue(torch.equal(torch.get_rng_state(), rng_before))
        for name, value in model.state_dict().items():
            self.assertTrue(torch.equal(value, state_before[name]), name)
        output.requested_delta.square().sum().backward()
        self._forward(reference, "A24P").requested_delta.square().sum().backward()
        for (name, parameter), expected in zip(model.named_parameters(), reference.parameters(), strict=True):
            self.assertTrue(torch.equal(parameter.grad, expected.grad), name)

    def test_nonfinite_and_out_of_range_values_are_observed_without_repair(self) -> None:
        alpha = torch.tensor([float("nan"), float("inf"), -0.25, 0.0, 0.5, 1.0, 1.25, -float("inf")])
        alpha = alpha.reshape(1, 1, 2, 2, 2)
        raw = alpha.repeat(1, 6, 1, 1, 1)
        output = Stage5ControllerOutput("A2P", alpha.repeat(1, 3, 1, 1, 1), raw, alpha, None)
        report = observe_controller_output(output, collar_width=1)
        summary = report["alpha"]["s2"]["full_volume"]
        self.assertEqual(summary["nonfinite_count"], 3)
        self.assertEqual(summary["finite_count"], 5)
        self.assertEqual(summary["unit_interval_counts"]["below_zero"], 1)
        self.assertEqual(summary["unit_interval_counts"]["above_one"], 1)
        self.assertIsNone(report["alpha"]["s2"]["interior_without_collar"])
        self.assertEqual(float(output.alpha_s2.flatten()[6]), 1.25)
        json.dumps(report, allow_nan=False)

    def test_free_residual_policy_has_no_attenuation_observations(self) -> None:
        model = Stage5SpatialController(width=4, collar_width=1)
        self.assertIsNone(observe_controller_output(self._forward(model, "F2P"), collar_width=1))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_cuda_observation_preserves_output_and_reports_pre_observation_peak(self) -> None:
        model = Stage5SpatialController(width=4, collar_width=1).cuda().eval()
        with torch.inference_mode():
            model.head.bias[3] = 2.0
            model.head.bias[4] = 3.0
            features = torch.zeros(1, 71, 6, 8, 10, device="cuda")
            proposal = torch.ones(1, 3, 6, 8, 10, device="cuda")
            output = model(features, "A24P", s2_proposal=proposal, s4_proposal=proposal)
            before = output.requested_delta.clone()
            torch.cuda.synchronize()
            peak_before = torch.cuda.max_memory_allocated()
            report = observe_controller_output(output, collar_width=1)
        self.assertTrue(torch.equal(before, output.requested_delta))
        self.assertEqual(report["peak_memory_bytes_before_observation"], peak_before)
        self.assertEqual(report["alpha_sum"]["full_volume"]["mean"], 1.0)
        self.assertIn("excluded", report["timing_scope"])


if __name__ == "__main__":
    unittest.main()
