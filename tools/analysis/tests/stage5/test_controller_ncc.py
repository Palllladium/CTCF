from __future__ import annotations

import unittest
from unittest import mock

import torch
import torch.nn.functional as F

from experiments.stage5 import losses, ncc, runtime
from experiments.stage5.config import ControllerTrainingConfig, U0TrainingConfig
from tools.analysis import diagnose_stage5_amp
from tools.analysis.stage5.ncc_diagnostic import ReferenceNCC
from tools.analysis.stage5.protocol import controller_training_contract, u0_training_contract
from utils import NCCVxm


def centered_ncc(first, second, width):
    patches = []
    for value in (first.double(), second.double()):
        value = F.pad(value, [width // 2] * 6)
        for dim in (2, 3, 4):
            value = value.unfold(dim, width, 1)
        patches.append(value - value.mean((-3, -2, -1), keepdim=True))
    x, y = patches
    cross = (x * y).sum((-3, -2, -1))
    vi = x.square().sum((-3, -2, -1)).clamp_min(1e-5)
    vj = y.square().sum((-3, -2, -1)).clamp_min(1e-5)
    return -(cross.square() / (vi * vj)).mean()


class ControllerNCCTest(unittest.TestCase):
    def test_matches_h100_reference_and_independent_centered_windows_in_value_and_gradient(self):
        generator = torch.Generator().manual_seed(947)
        for offset, noise in ((0.0, 1.0), (-1.23, 0.0), (-1.23, 1e-4), (20.0, 0.1)):
            x = (offset + noise * torch.randn(1, 1, 9, 10, 11, generator=generator)).requires_grad_()
            y = offset * 1.3 + noise * torch.randn(x.shape, generator=generator)
            actual = ncc.ControllerNCC(win=(7, 7, 7))(x, y)
            reference = ReferenceNCC(win=(7, 7, 7))(x, y)
            oracle = centered_ncc(x, y, 7)
            gradient = torch.autograd.grad(actual, x)[0]
            self.assertTrue(torch.equal(actual, reference))
            self.assertTrue(torch.equal(gradient, torch.autograd.grad(reference, x)[0]))
            self.assertAlmostEqual(float(actual.detach()), float(oracle.detach()), places=6)
            torch.testing.assert_close(gradient, torch.autograd.grad(oracle, x)[0], atol=1e-7, rtol=2e-4)

    def test_old_constant_input_failure_is_repaired(self):
        x = torch.full((1, 1, 24, 24, 24), -1.23, requires_grad=True)
        y = torch.full_like(x, -1.45)
        self.assertLess(float(NCCVxm(win=(7, 7, 7))(x, y).detach()), -100)
        actual = ncc.ControllerNCC(win=(7, 7, 7))(x, y)
        self.assertAlmostEqual(float(actual.detach()), -0.578125, places=6)
        actual.backward()
        self.assertTrue(bool(torch.isfinite(x.grad).all()))

    def test_checks_each_window_even_when_mean_would_be_valid(self):
        x = torch.zeros(1, 1, 5, 5, 5)
        cc = torch.zeros_like(x, dtype=torch.float64)
        cc.flatten()[0] = 2.0
        self.assertLess(float(cc.mean()), 1)
        with (
            mock.patch.object(ncc, "_squared_correlation", return_value=cc),
            self.assertRaisesRegex(FloatingPointError, "outside"),
        ):
            ncc.ControllerNCC(win=(3, 3, 3))(x, x)

    def test_nonfinite_and_negative_windows_are_rejected(self):
        x = torch.zeros(1, 1, 5, 5, 5)
        for value in (float("nan"), float("inf"), -0.01):
            cc = torch.full_like(x, value, dtype=torch.float64)
            with mock.patch.object(ncc, "_squared_correlation", return_value=cc), self.assertRaises(FloatingPointError):
                ncc.ControllerNCC(win=(3, 3, 3))(x, x)

    def test_tiny_roundoff_is_retained_not_clipped(self):
        x = torch.zeros(1, 1, 5, 5, 5)
        cc = torch.full_like(x, 1 + 0.5e-6, dtype=torch.float64)
        with mock.patch.object(ncc, "_squared_correlation", return_value=cc):
            loss = ncc.ControllerNCC(win=(3, 3, 3))(x, x)
        self.assertEqual(float(loss), float(-cc.mean().float()))
        self.assertLess(float(loss), -1)

    def test_extreme_offset_does_not_silently_misreport_fp64_cancellation(self):
        x = torch.full((1, 1, 13, 13, 13), 123456.7)
        y = torch.full_like(x, 160493.71)
        with self.assertRaises(FloatingPointError):
            ncc.ControllerNCC(win=(7, 7, 7))(x, y)

    def test_fp32_scalar_and_gradients_under_outer_autocast(self):
        x = torch.rand(1, 1, 9, 9, 9, requires_grad=True)
        y = torch.rand_like(x)
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            loss = ncc.ControllerNCC(win=(7, 7, 7))(x, y)
        loss.backward()
        self.assertEqual(loss.dtype, torch.float32)
        self.assertEqual(x.grad.dtype, torch.float32)

    def test_contract_keeps_fp64_ncc_and_u0_while_controllers_use_fp32(self):
        config = ControllerTrainingConfig()
        controller = controller_training_contract(config)
        u0 = u0_training_contract(U0TrainingConfig())
        self.assertEqual(controller["schema"], "ctcf-stage5-controller-training-contract-v5")
        self.assertEqual(controller["objective_numerics"], ncc.controller_ncc_contract())
        self.assertEqual(controller["objective_numerics"]["ncc_out_of_range_policy"], "RAISE_WITHOUT_CLIPPING")
        self.assertFalse(controller["precision"]["gradient_scaler"])
        self.assertEqual(u0["schema"], "ctcf-stage5-u0-training-contract-v2")
        self.assertNotIn("objective_numerics", u0)
        self.assertIs(losses.ControllerNCC, ncc.ControllerNCC)

    def test_historical_diagnostic_replays_legacy_ncc_and_restores_production(self):
        def replay(_args, _report):
            self.assertIs(losses.ControllerNCC, NCCVxm)
            raise RuntimeError("end of synthetic replay")

        with (
            mock.patch.object(diagnose_stage5_amp, "_diagnose_legacy_source", side_effect=replay),
            self.assertRaises(RuntimeError),
        ):
            diagnose_stage5_amp.diagnose(None, {})
        self.assertIs(losses.ControllerNCC, ncc.ControllerNCC)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required for controller AMP")
    def test_all_eight_controllers_take_a_real_strict_fp32_step(self):
        torch.manual_seed(91)
        config = ControllerTrainingConfig()
        shape = (16, 16, 16)
        features = torch.randn(1, 71, *shape, device="cuda")
        proposals = torch.randn(1, 3, *shape, device="cuda") * 0.1
        psi = torch.zeros_like(proposals)
        first = torch.full((1, 1, *shape), -1.23, device="cuda")
        second = torch.full_like(first, -1.45)
        first[:, :, 5:11, 5:11, 5:11] = torch.rand(1, 1, 6, 6, 6, device="cuda")
        second[:, :, 4:10, 5:11, 5:11] = torch.rand(1, 1, 6, 6, 6, device="cuda")
        inputs = runtime._ControllerPairInputs(
            psi,
            psi,
            (features, proposals, proposals, first, second),
            (features, proposals, proposals, second, first),
            0.0,
        )
        for variant in runtime.STAGE5_VARIANTS:
            controller = runtime.build_stage5_controller(config).cuda()
            optimizer = torch.optim.AdamW(controller.parameters(), lr=config.learning_rate)
            step = mock.Mock(
                controller=controller,
                variant=variant,
                config=config,
                optimizer=optimizer,
                scaler=None,
                device=torch.device("cuda"),
            )
            before = {name: value.detach().clone() for name, value in controller.named_parameters()}
            with mock.patch.object(runtime, "_prepare_controller_pair", return_value=inputs):
                metrics = runtime._controller_pair_step(step, {}, 0)
            self.assertTrue(-1.000001 <= metrics["ncc"] <= 0)
            self.assertTrue(any(not torch.equal(value, before[name]) for name, value in controller.named_parameters()))


if __name__ == "__main__":
    unittest.main()
