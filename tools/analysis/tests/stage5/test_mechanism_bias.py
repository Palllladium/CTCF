from __future__ import annotations

import copy
import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from experiments.stage5.checkpoints import capture_rng_state
from tools.analysis.stage5 import mechanism_bias as bias
from tools.analysis.stage5.precision_probes import _state_equal, backend_flags


class ReductionTest(unittest.TestCase):
    def test_preservation_accepts_absent_optional_optimizer_and_scaler(self):
        step = SimpleNamespace(controller=torch.nn.Linear(1, 1), optimizer=None, scaler=None)
        original = copy.deepcopy(step.controller.state_dict())
        with bias._preserve(step), torch.no_grad():
            step.controller.weight.add_(1)
        self.assertTrue(_state_equal(step.controller.state_dict(), original))

    def test_channel_sums_include_batch_and_chunk_boundaries(self):
        value = torch.arange(2 * 3 * 2 * 3 * 4, dtype=torch.float32).reshape(2, 3, 2, 3, 4) / 16
        with patch.object(bias, "CHUNK_ELEMENTS", 7):
            single, double = bias.channel_sum_reference(value)
        expected = value.double().sum((0, 2, 3, 4))
        self.assertTrue(torch.equal(double, expected))
        self.assertTrue(torch.equal(single.double(), expected))

    def test_exact_zero_and_subnormal_counts_are_separate_from_sample(self):
        value = torch.tensor([0, 2**-24, 2**-14, 1, float("inf")], dtype=torch.float16)
        with patch.object(bias, "CHUNK_ELEMENTS", 2):
            result = bias.gradient_stats(value, 2)
        self.assertEqual(result["zero"], 1)
        self.assertEqual(result["native_subnormal"], 1)
        self.assertEqual(result["nonfinite"], 1)
        self.assertEqual(result["unscaled_max_finite_abs"], 0.5)
        json.dumps(result, allow_nan=False)

    def test_sample_reports_values_lost_to_zero(self):
        result = bias._sample_comparison(torch.tensor([0.0, 1.0]), torch.tensor([1e-12, 1.0]))
        self.assertEqual(result["actual_zero_reference_nonzero"], 1)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required for autocast convolution boundary")
class BiasCudaTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def _step(self):
        model = torch.nn.Module()
        model.stem = torch.nn.Sequential(torch.nn.Conv3d(2, 2, 1))
        model.forward = lambda value: model.stem(value)
        model = model.cuda()
        with torch.no_grad():
            model.stem[0].weight.fill_(0.1)
            model.stem[0].bias.fill_(0.1)
        step = SimpleNamespace(
            controller=model,
            optimizer=torch.optim.Adam(model.parameters()),
            scaler=torch.amp.GradScaler("cuda"),
            device=torch.device("cuda"),
        )
        value = torch.ones(1, 2, 4, 4, 4, device="cuda")
        inputs = SimpleNamespace(psi_ab=value, psi_ba=value, tensors_ab=(value,), tensors_ba=(value,))
        return step, inputs

    def test_actual_half_bias_return_overflows_with_finite_upstream(self):
        step, _ = self._step()
        value = torch.ones(1, 2, 32, 32, 32, device="cuda")
        with bias._BoundaryTrace(step.controller, 1.0) as trace:
            with torch.autocast("cuda", dtype=torch.float16):
                first = step.controller(value)
            (first.float().sum() * 2).backward()
        record = trace.bias[0]
        self.assertEqual(record["status"], "CAPTURED")
        self.assertEqual(record["upstream"]["nonfinite"], 0)
        self.assertEqual(record["sum_fp64"], [65536.0, 65536.0])
        self.assertEqual(record["convolution_returned_bias"]["nonfinite"], 2)
        self.assertEqual(record["sum_fp64_then_half_nonfinite"], 2)
        self.assertEqual(record["sum_exceeds_half_max"], 2)
        self.assertEqual(len(step.controller.stem[0]._forward_hooks), 0)
        json.dumps(record, allow_nan=False)

    def test_scale1_zero_loss_and_full_state_restoration(self):
        step, inputs = self._step()
        for parameter in step.controller.parameters():
            parameter.grad = torch.full_like(parameter, 7)
        model_state = copy.deepcopy(step.controller.state_dict())
        optimizer_state = copy.deepcopy(step.optimizer.state_dict())
        scaler_state = copy.deepcopy(step.scaler.state_dict())
        rng = capture_rng_state()
        flags = backend_flags()

        def loss(current, prepared):
            first = current.controller(prepared.tensors_ab[0])
            second = current.controller(prepared.tensors_ba[0])
            total = (first.float().sum() + second.float().sum()) * 1e-10
            return total, {"loss": float(total.detach())}

        with patch.object(bias.runtime, "_controller_pair_loss", loss):
            report = bias.run_bias_audit(step, inputs)
        self.assertEqual(report["status"], "COMPLETE", report)
        comparison = report["comparisons"]["fp16_scale1_vs_fp16_scale32768"]
        self.assertEqual(comparison["parameters_all_elements"]["stem.0.bias"]["actual_zero_reference_nonzero"], 2)
        self.assertGreater(comparison["layer_samples"]["stem.0#1"]["actual_zero_reference_nonzero"], 0)
        self.assertEqual(report["probes"]["fp16_scale1"]["bias_directions"][0]["upstream"]["zero"], 128)
        self.assertTrue(_state_equal(model_state, step.controller.state_dict()))
        self.assertTrue(_state_equal(optimizer_state, step.optimizer.state_dict()))
        self.assertTrue(_state_equal(scaler_state, step.scaler.state_dict()))
        self.assertTrue(_state_equal(rng, capture_rng_state()))
        self.assertEqual(flags, backend_flags())
        for parameter in step.controller.parameters():
            self.assertTrue(bool((parameter.grad == 7).all()))
        json.dumps(report, allow_nan=False)

    def test_failure_and_save_exception_restore_hooks_rng_and_state(self):
        step, inputs = self._step()
        rng = capture_rng_state()
        model_state = copy.deepcopy(step.controller.state_dict())
        original_interpolate = bias.F.interpolate

        def fail(current, *_args, **_kwargs):
            torch.rand(2, device="cuda")
            with torch.no_grad():
                current.controller.stem[0].bias.add_(1)
            raise RuntimeError("sentinel")

        with patch.object(bias.runtime, "_controller_pair_loss", fail):
            report = bias.run_bias_audit(step, inputs)
            self.assertEqual(report["status"], "INCOMPLETE")
            with self.assertRaisesRegex(ValueError, "save failed"):
                bias.run_bias_audit(step, inputs, save=lambda _report: (_ for _ in ()).throw(ValueError("save failed")))
        self.assertTrue(_state_equal(model_state, step.controller.state_dict()))
        self.assertTrue(_state_equal(rng, capture_rng_state()))
        self.assertIs(original_interpolate, bias.F.interpolate)
        self.assertEqual(len(step.controller.stem[0]._forward_hooks), 0)

    def test_real_controller_captures_both_directions_and_forward_identity(self):
        from tools.analysis.stage5.mechanism_report import bias_audit_errors

        runtime = bias.runtime
        config = runtime.ControllerTrainingConfig(width=4)
        controller = runtime.build_stage5_controller(config).cuda()
        with torch.no_grad():
            controller.head.weight.normal_(std=0.002)
        step = SimpleNamespace(controller=controller, variant="F0", config=config, device=torch.device("cuda"))
        shape = (16, 16, 16)
        features = torch.randn(1, 71, *shape, device="cuda") * 0.1
        proposal = torch.zeros(1, 3, *shape, device="cuda")
        first = torch.randn(1, 1, *shape, device="cuda")
        second = first + torch.randn_like(first) * 0.05
        inputs = runtime._ControllerPairInputs(
            proposal,
            proposal,
            (features, proposal, proposal, first, second),
            (features, proposal, proposal, second, first),
            0.0,
        )
        report = bias.run_bias_audit(step, inputs)
        self.assertEqual(report["status"], "COMPLETE", report)
        self.assertEqual(bias_audit_errors(report, backend_flags()), [])
        for probe in report["probes"].values():
            self.assertEqual(len(probe["requested_delta_forward_hashes"]), 2)
            self.assertEqual({direction["direction"] for direction in probe["bias_directions"]}, {"ab", "ba"})
            self.assertIn("fp64_sum_of_direction_references", probe["bias_accumulation"])
        comparison = report["comparisons"]["fp16_scale1_vs_fp16_scale32768"]
        self.assertTrue(comparison["requested_delta_forward_bytes_equal"])
        self.assertIn("interpolate#1.input", comparison["layer_samples"])
        json.dumps(report, allow_nan=False)

    def test_optional_reference_mode_can_be_absent_from_comparison_plan(self):
        step, inputs = self._step()
        pair = ("fp16_scale1", "fp16_scale32768_tf32_off")
        with (
            bias.strict_fp32(),
            patch.object(bias, "BIAS_COMPARISONS", (*bias.BIAS_COMPARISONS, pair)),
            patch.object(bias.runtime, "_controller_pair_loss", side_effect=RuntimeError("sentinel")),
        ):
            report = bias.run_bias_audit(step, inputs)
        self.assertEqual(report["status"], "INCOMPLETE")
        self.assertNotIn(f"{pair[0]}_vs_{pair[1]}", report["comparisons"])


if __name__ == "__main__":
    unittest.main()
