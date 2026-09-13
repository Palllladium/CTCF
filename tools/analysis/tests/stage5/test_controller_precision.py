from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from experiments.stage5 import runtime
from experiments.stage5.checkpoints import build_training_state, load_training_state, state_dict_sha256
from experiments.stage5.failures import ControllerFailureRecorder
from experiments.stage5.precision import (
    controller_precision,
    controller_precision_contract,
    precision_context,
    precision_mode_contract,
)
from tools.analysis.run_artifacts import sha256_file


class ControllerPrecisionTest(unittest.TestCase):
    def test_comparison_context_modes_are_explicit_and_nesting_restores_parent(self):
        device = torch.device("cpu")
        with controller_precision(device):
            for mode in ("fp32_strict", "tf32", "bf16"):
                with precision_context(device, mode):
                    contract = precision_mode_contract(mode)
                    self.assertEqual(torch.is_autocast_enabled("cpu"), contract["autocast"])
                    self.assertEqual(torch.backends.cudnn.allow_tf32, contract["cudnn_allow_tf32"])
                    self.assertEqual(torch.backends.cuda.matmul.allow_tf32, contract["matmul_allow_tf32"])
                    actual = torch.mm(torch.ones(2, 2), torch.ones(2, 2))
                    self.assertEqual(actual.dtype, torch.bfloat16 if mode == "bf16" else torch.float32)
                self.assertFalse(torch.is_autocast_enabled("cpu"))
                self.assertTrue(torch.backends.cudnn.allow_tf32)
                self.assertTrue(torch.backends.cuda.matmul.allow_tf32)
            with self.assertRaisesRegex(ValueError, "unknown"), precision_context(device, "automatic"):
                self.fail("unknown mode entered")

    def test_shared_loss_never_reenables_autocast_inside_production_context(self):
        observations = []

        def controller(*args, **kwargs):
            observations.append(torch.is_autocast_enabled("cpu"))
            return SimpleNamespace(requested_delta=torch.zeros(1))

        inputs = runtime._ControllerPairInputs(
            torch.zeros(1), torch.zeros(1), (torch.ones(1),) * 5, (torch.ones(1),) * 5, 0
        )
        step = SimpleNamespace(controller=controller, variant="F0", config=runtime.ControllerTrainingConfig())
        with (
            patch.object(runtime, "controller_objective", return_value=(torch.tensor(0.0), {})),
            torch.autocast("cpu", dtype=torch.bfloat16),
            controller_precision(torch.device("cpu")),
        ):
            runtime._controller_pair_loss(step, inputs)
        self.assertEqual(observations, [False, False])

    def test_medium_matmul_setting_is_restored_without_mixing_backend_apis(self):
        before = torch.get_float32_matmul_precision()
        try:
            torch.set_float32_matmul_precision("medium")
            with controller_precision(torch.device("cpu")):
                self.assertEqual(torch.get_float32_matmul_precision(), "high")
                self.assertTrue(torch.backends.cuda.matmul.allow_tf32)
            self.assertEqual(torch.get_float32_matmul_precision(), "medium")
        finally:
            torch.set_float32_matmul_precision(before)

    def test_context_enables_tf32_without_autocast_and_restores_on_exception(self):
        before = (torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32)
        with (
            self.assertRaisesRegex(RuntimeError, "test error"),
            torch.autocast("cpu", dtype=torch.bfloat16),
            controller_precision(torch.device("cpu")),
        ):
            self.assertFalse(torch.is_autocast_enabled("cpu"))
            self.assertTrue(torch.backends.cudnn.allow_tf32)
            self.assertTrue(torch.backends.cuda.matmul.allow_tf32)
            self.assertEqual(torch.mm(torch.ones(2, 2), torch.ones(2, 2)).dtype, torch.float32)
            raise RuntimeError("test error")
        self.assertEqual(before, (torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32))

    def _small_step(self):
        model = torch.nn.Linear(1, 1, bias=False)
        with torch.no_grad():
            model.weight.fill_(1.0)
        return SimpleNamespace(
            controller=model,
            optimizer=torch.optim.AdamW(model.parameters(), lr=0.1, weight_decay=0.0),
            scaler=None,
            config=runtime.ControllerTrainingConfig(),
            device=torch.device("cpu"),
            variant="F0",
        )

    def test_failure_capture_preserves_pair_inputs_rng_and_unmodified_update(self):
        step = self._small_step()
        before = state_dict_sha256(step.controller.state_dict())
        inputs = runtime._ControllerPairInputs(
            torch.zeros(1), torch.ones(1), (torch.ones(1),), (torch.zeros(1),), 0.125
        )
        pair = {"subject_a": "subject-a", "subject_b": "subject-b"}
        hook = step.controller.weight.register_hook(lambda gradient: gradient * float("inf"))
        with tempfile.TemporaryDirectory() as temporary:
            recorder = ControllerFailureRecorder(Path(temporary), {"seed": 1, "variant": "F0"})
            with (
                patch.object(runtime, "_prepare_controller_pair", return_value=inputs),
                patch.object(
                    runtime,
                    "_controller_pair_loss",
                    side_effect=lambda *a, **k: (step.controller.weight.square().sum(), {"loss": 1.0}),
                ),
                self.assertRaisesRegex(FloatingPointError, "gradient"),
            ):
                runtime._controller_pair_step(step, pair, 2, recorder=recorder, pair_index=7, successful_updates=301)
            hook.remove()
            manifest_path = next(Path(temporary).rglob("failure.json"))
            manifest = json.loads(manifest_path.read_text())
            capture_path = Path(manifest["capture_file"]["path"])
            self.assertEqual(sha256_file(capture_path), manifest["capture_file"]["sha256"])
            capture = torch.load(capture_path, weights_only=False)
            self.assertEqual(manifest["phase"], "backward")
            self.assertEqual(manifest["successful_updates_before_pair"], 301)
            self.assertEqual(manifest["pair_index_one_based"], 8)
            self.assertEqual(manifest["epoch_one_based"], 3)
            self.assertEqual(capture["pair"], pair)
            self.assertTrue(torch.equal(capture["inputs"]["psi_ba"], inputs.psi_ba))
            self.assertEqual(before, state_dict_sha256(capture["before_update"]["model"]))
            self.assertEqual(before, state_dict_sha256(step.controller.state_dict()))
            self.assertEqual(capture["before_update"]["optimizer"]["state"], {})
            self.assertEqual(set(capture["rng_before_controller"]), {"python", "numpy", "torch_cpu", "torch_cuda"})
            with self.assertRaisesRegex(RuntimeError, "automatic retry"):
                recorder.require_no_previous_failure()

    def test_optimizer_failure_preserves_pre_update_state(self):
        step = self._small_step()
        before = state_dict_sha256(step.controller.state_dict())

        def broken_update():
            with torch.no_grad():
                step.controller.weight.fill_(float("nan"))

        with tempfile.TemporaryDirectory() as temporary:
            recorder = ControllerFailureRecorder(Path(temporary), {})
            with (
                patch.object(runtime, "_prepare_controller_pair", return_value=SimpleNamespace(bootstrap_residual=0)),
                patch.object(
                    runtime,
                    "_controller_pair_loss",
                    side_effect=lambda *a, **k: (step.controller.weight.square().sum(), {"loss": 1.0}),
                ),
                patch.object(step.optimizer, "step", side_effect=broken_update),
                self.assertRaisesRegex(FloatingPointError, "parameter"),
            ):
                runtime._controller_pair_step(step, {}, 0, recorder=recorder)
            path = next(Path(temporary).rglob("capture.pth"))
            capture = torch.load(path, weights_only=False)
            self.assertEqual(capture["phase"], "optimizer_step")
            self.assertEqual(before, state_dict_sha256(capture["before_update"]["model"]))
            self.assertTrue(torch.isnan(capture["model_at_failure"]["weight"]).all())

    def test_controller_checkpoint_roundtrip_without_scaler(self):
        step = self._small_step()
        payload = build_training_state(
            role="CONTROLLER",
            variant_id="F0",
            seed=0,
            epoch_completed=1,
            fixed_epoch=100,
            git_head="a" * 40,
            protocol_sha256="b" * 64,
            data_contract_sha256="c" * 64,
            training_contract_sha256="d" * 64,
            model=step.controller,
            optimizer=step.optimizer,
            scaler=None,
            pair_schedule_sha256="e" * 64,
            metrics_sha256="f" * 64,
        )
        self.assertIsNone(payload["scaler_state"])
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "last.pth"
            torch.save(payload, path)
            loaded = load_training_state(
                path,
                model=step.controller,
                optimizer=step.optimizer,
                scaler=None,
                expected_role="CONTROLLER",
                expected_variant="F0",
                expected_seed=0,
                expected_protocol_sha256="b" * 64,
                expected_data_contract_sha256="c" * 64,
                expected_training_contract_sha256="d" * 64,
                restore_rng=True,
            )
            self.assertIsNone(loaded["scaler_state"])

    def test_capture_write_failure_keeps_original_error_and_blocks_retry(self):
        step = self._small_step()
        with tempfile.TemporaryDirectory() as temporary:
            recorder = ControllerFailureRecorder(Path(temporary), {})
            with (
                patch.object(runtime, "_prepare_controller_pair", side_effect=FloatingPointError("source failure")),
                patch("experiments.stage5.failures.atomic_torch_save", side_effect=OSError("disk full")),
                self.assertRaisesRegex(FloatingPointError, "source failure"),
            ):
                runtime._controller_pair_step(step, {}, 0, recorder=recorder)
            path = next(Path(temporary).rglob("failure.json"))
            manifest = json.loads(path.read_text())
            self.assertEqual(manifest["capture_status"], "FAILED")
            self.assertFalse(manifest["prepared_inputs_available"])
            self.assertIn("disk full", manifest["capture_error"])
            with self.assertRaisesRegex(RuntimeError, "automatic retry"):
                recorder.require_no_previous_failure()

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_comparison_modes_execute_real_controller_convolutions_and_finite_updates(self):
        from experiments.stage5.failures import require_finite_optimizer, require_finite_parameters
        from models.CTCF.controller import STAGE5_VARIANTS

        device = torch.device("cuda")
        config = runtime.ControllerTrainingConfig()
        controller = runtime.build_stage5_controller(config).to(device)
        features = torch.randn(1, 71, 16, 16, 16, device=device) * 0.01
        proposal = torch.zeros(1, 3, 16, 16, 16, device=device)
        first = torch.randn(1, 1, 16, 16, 16, device=device)
        inputs = runtime._ControllerPairInputs(
            proposal,
            proposal,
            (features, proposal, proposal, first, first.roll(1, -1)),
            (features, proposal, proposal, first.roll(1, -1), first),
            0.0,
        )
        optimizer = torch.optim.AdamW(controller.parameters(), lr=config.learning_rate)
        for mode in ("fp32_strict", "tf32", "bf16"):
            for variant in STAGE5_VARIANTS:
                observed = []
                hook = controller.stem[0].register_forward_hook(
                    lambda _m, _a, output, observed=observed: observed.append(output.dtype)
                )
                step = SimpleNamespace(controller=controller, optimizer=optimizer, variant=variant, config=config)
                try:
                    with precision_context(device, mode):
                        optimizer.zero_grad(set_to_none=True)
                        loss, _logs = runtime._controller_pair_loss(step, inputs)
                        self.assertEqual(loss.dtype, torch.float32)
                        loss.backward()
                        require_finite_parameters(controller, gradients=True)
                        optimizer.step()
                        require_finite_parameters(controller, gradients=False)
                        require_finite_optimizer(optimizer)
                finally:
                    hook.remove()
                expected = torch.bfloat16 if mode == "bf16" else torch.float32
                self.assertEqual(observed, [expected, expected])

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_real_bidirectional_controller_step_uses_tf32_forward_and_backward(self):
        device = torch.device("cuda")
        for variant in ("F0", "F2V", "F24P"):
            config = runtime.ControllerTrainingConfig()
            controller = runtime.build_stage5_controller(config).to(device)
            optimizer = torch.optim.AdamW(controller.parameters(), lr=config.learning_rate)
            features = torch.randn(1, 71, 16, 16, 16, device=device) * 0.01
            proposal = torch.zeros(1, 3, 16, 16, 16, device=device)
            first = torch.randn(1, 1, 16, 16, 16, device=device)
            second = first.roll(1, -1)
            inputs = runtime._ControllerPairInputs(
                proposal,
                proposal,
                (features, proposal, proposal, first, second),
                (features, proposal, proposal, second, first),
                0.0,
            )
            step = SimpleNamespace(
                controller=controller, optimizer=optimizer, scaler=None, config=config, device=device, variant=variant
            )
            observed = []

            def observe(_module, _args, output, observed=observed):
                observed.append(output.requested_delta.dtype)
                self.assertTrue(torch.backends.cudnn.allow_tf32)
                output.requested_delta.register_hook(check_gradient)

            def check_gradient(gradient):
                self.assertEqual(gradient.dtype, torch.float32)
                self.assertTrue(torch.backends.cudnn.allow_tf32)
                return gradient

            hook = controller.register_forward_hook(observe)
            before = state_dict_sha256(controller.state_dict())
            with patch.object(runtime, "_prepare_controller_pair", return_value=inputs):
                metrics = runtime._controller_pair_step(step, {}, 0)
            hook.remove()
            self.assertEqual(observed, [torch.float32, torch.float32])
            self.assertNotEqual(before, state_dict_sha256(controller.state_dict()))
            self.assertTrue(all(torch.isfinite(torch.tensor(value)) for value in metrics.values()))
            self.assertFalse(controller_precision_contract()["gradient_scaler"])


if __name__ == "__main__":
    unittest.main()
