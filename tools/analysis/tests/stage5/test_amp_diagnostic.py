from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch

from experiments.stage5 import runtime
from tools.analysis import diagnose_stage5_amp as diagnostic
from tools.analysis.run_artifacts import atomic_write_json, sha256_file


class DiagnosticProbeTest(unittest.TestCase):
    def setUp(self):
        self.model = torch.nn.Linear(1, 1, bias=False)
        self.model.weight.data.fill_(1.0)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=0.1)

    def loss_fn(self, fp32):
        output = self.model(torch.ones(1, 1))
        if not fp32:
            output = output.half().float()
        return (output * 2).sum(), {"value": 2.0}

    def test_scale_sensitive_backward_and_fp32_reference_without_updates(self):
        state = copy.deepcopy(self.model.state_dict())
        optimizer = copy.deepcopy(self.optimizer.state_dict())
        trials = []
        with mock.patch.object(self.optimizer, "step", side_effect=AssertionError("Must not update")):
            for scale, fp32 in ((65536.0, False), (1.0, False), (1.0, True)):
                trials.append(
                    diagnostic.probe(
                        self.model,
                        self.optimizer,
                        self.loss_fn,
                        device_type="cpu",
                        scale=scale,
                        fp32=fp32,
                    )
                )
        self.assertEqual([t["status"] for t in trials], ["NONFINITE_GRADIENTS", "FINITE", "FINITE"])
        self.assertIsNotNone(trials[0]["first_observed_nonfinite"])
        self.assertEqual(trials[0]["first_observed_nonfinite"]["stage"], "scaled_backward")
        self.assertTrue(torch.equal(self.model.weight, state["weight"]))
        self.assertEqual(self.optimizer.state_dict(), optimizer)
        self.assertEqual(len(self.model._forward_hooks), 0)
        self.assertEqual(diagnostic.classify_probes(trials), "SCALE_SENSITIVE_OVERFLOW_ON_THIS_PAIR")
        json.dumps(trials, allow_nan=False)

    def test_nonfinite_loss_is_not_misreported_as_gradient_overflow(self):
        result = diagnostic.probe(
            self.model,
            self.optimizer,
            lambda _fp32: (self.model(torch.ones(1, 1)).sum() * float("nan"), {"loss": float("nan")}),
            device_type="cpu",
            scale=1.0,
            fp32=False,
        )
        self.assertEqual(result["status"], "NONFINITE_LOSS")
        self.assertIsNone(result["loss"])
        json.dumps(result, allow_nan=False)

    def test_probe_exception_is_recorded_and_hooks_removed(self):
        def fail(_fp32):
            self.model(torch.ones(1, 1))
            raise FloatingPointError("bad head")

        result = diagnostic.probe(
            self.model,
            self.optimizer,
            fail,
            device_type="cpu",
            scale=1.0,
            fp32=False,
        )
        self.assertEqual(result["status"], "PROBE_EXCEPTION")
        self.assertIn("bad head", result["error"])
        self.assertEqual(len(self.model._forward_hooks), 0)

    def test_tensor_stats_are_json_safe(self):
        result = diagnostic.tensor_stats(torch.tensor([1.0, float("inf"), float("nan"), -3.0]))
        self.assertEqual(result["nonfinite"], 2)
        self.assertEqual(result["max_finite_abs"], 3)
        json.dumps(result, allow_nan=False)

    def test_classification_never_calls_one_finite_probe_a_training_success(self):
        trials = [{"status": "NONFINITE_GRADIENTS"}, {"status": "NONFINITE_GRADIENTS"}, {"status": "FINITE"}]
        self.assertEqual(diagnostic.classify_probes(trials), "FP16_PATH_FAILURE_NOT_RESOLVED_BY_TESTED_SCALES")
        trials[0]["status"] = "FINITE"
        self.assertEqual(diagnostic.classify_probes(trials), "FAILURE_NOT_REPRODUCED_IN_INSTRUMENTED_PROBE")


class ReadonlyCheckpointTest(unittest.TestCase):
    def test_valid_source_is_unchanged_and_recovery_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "initial.pth"
            path.write_bytes(b"checkpoint bytes")
            digest = sha256_file(path)
            atomic_write_json(
                runtime._checkpoint_sidecar_path(path),
                {
                    "file_name": path.name,
                    "bytes": path.stat().st_size,
                    "sha256": digest,
                },
            )
            self.assertEqual(diagnostic.require_readonly_checkpoint(path), digest)
            recovery = path.with_name(f".{path.name}.previous")
            recovery.write_bytes(b"retained recovery")
            with self.assertRaisesRegex(RuntimeError, "recovery required"):
                diagnostic.require_readonly_checkpoint(path)
            self.assertTrue(recovery.exists())
            self.assertEqual(sha256_file(path), digest)

    def test_corrupt_source_is_rejected_without_repair(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "initial.pth"
            path.write_bytes(b"changed")
            atomic_write_json(
                runtime._checkpoint_sidecar_path(path),
                {
                    "file_name": path.name,
                    "bytes": path.stat().st_size,
                    "sha256": "a" * 64,
                },
            )
            with self.assertRaisesRegex(RuntimeError, "mismatch"):
                diagnostic.require_readonly_checkpoint(path)
            self.assertEqual(path.read_bytes(), b"changed")


class ProductionStepParityTest(unittest.TestCase):
    def test_replay_finite_prefix_then_probe_same_failing_pair(self):
        model = torch.nn.Linear(1, 1, bias=False)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        step = SimpleNamespace(
            controller=model,
            optimizer=optimizer,
            scaler=torch.amp.GradScaler("cpu"),
            store=None,
            variant="F0",
            device=torch.device("cpu"),
        )
        pairs = ({"subject_a": "a", "subject_b": "b"}, {"subject_a": "c", "subject_b": "d"})

        def pair_loss(_step, pair):
            with torch.autocast("cpu", enabled=False):
                output = model(torch.ones(1, 1))
            if pair is pairs[1] and torch.is_autocast_enabled("cpu"):
                output = output.half().float()
            return output.sum() * 2, {}

        with tempfile.TemporaryDirectory() as temporary:
            args = SimpleNamespace(output=Path(temporary) / "report.json", variant="F0")
            report = {"source_hashes": {}}
            original_probe = diagnostic.probe

            def cpu_probe(*positional, **keywords):
                keywords["device_type"] = "cpu"
                return original_probe(*positional, **keywords)

            with (
                mock.patch.object(diagnostic, "prepare_step", return_value=step),
                mock.patch.object(runtime, "_training_subjects", return_value=()),
                mock.patch.object(runtime, "controller_epoch_pairs", return_value=pairs),
                mock.patch.object(runtime, "_prepare_controller_pair", side_effect=lambda _step, pair, epoch: pair),
                mock.patch.object(runtime, "_controller_pair_loss", side_effect=pair_loss),
                mock.patch.object(diagnostic, "probe", side_effect=cpu_probe),
            ):
                diagnostic.diagnose(args, report)
            self.assertEqual(report["successful_in_memory_updates"], 1)
            self.assertEqual(report["pair_index_one_based"], 2)
            self.assertEqual(report["pair"], pairs[1])
            self.assertEqual(len(report["probes"]), 6)
            self.assertEqual(report["interpretation"], "SCALE_SENSITIVE_OVERFLOW_ON_THIS_PAIR")
            self.assertTrue(report["source_bytes_unchanged"])

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required for real controller autocast")
    def test_real_controller_hooks_and_loss_in_both_precisions(self):
        controller = runtime.build_stage5_controller(runtime.ControllerTrainingConfig()).cuda()
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
        step = SimpleNamespace(controller=controller, variant="F0", config=runtime.ControllerTrainingConfig())
        for fp32 in (False, True):
            result = diagnostic.probe(
                controller,
                optimizer,
                lambda mode: runtime._controller_pair_loss(step, inputs),
                device_type="cuda",
                scale=1.0,
                fp32=fp32,
            )
            self.assertEqual(result["status"], "FINITE", result.get("error"))
            self.assertTrue(any("requested_delta" in event["tensor"] for event in result["tensor_events"]))
            self.assertTrue(any(event["stage"] == "scaled_backward" for event in result["tensor_events"]))

    def test_legacy_replay_keeps_amp_and_strict_optimizer_step(self):
        parameter = torch.nn.Parameter(torch.tensor(1.0))
        optimizer = torch.optim.SGD([parameter], lr=0.1)
        step = SimpleNamespace(
            optimizer=optimizer, scaler=torch.amp.GradScaler("cpu"), variant="F0", device=torch.device("cpu")
        )
        inputs = SimpleNamespace(bootstrap_residual=0.125)
        with (
            mock.patch.object(runtime, "_prepare_controller_pair", return_value=inputs),
            mock.patch.object(
                runtime, "_controller_pair_loss", return_value=(parameter.square(), {"ncc": 1.0})
            ) as loss,
        ):
            logs = runtime._legacy_controller_pair_step(step, {"subject_a": "a", "subject_b": "b"}, 0)
        loss.assert_called_once_with(step, inputs)
        self.assertAlmostEqual(float(parameter.detach()), 0.8)
        self.assertEqual(logs["bootstrap_digital_residual_percent"], 0.125)

    def test_source_protocol_binding_is_exact(self):
        self.assertEqual(diagnostic.SOURCE_HEAD, "458489f77fc6f7c792ba1411bb763f4bb06310c5")
        self.assertEqual(diagnostic.VARIANTS, ("F0", "F2V", "F2S", "F2P"))
        self.assertEqual(runtime.LegacyControllerTrainingConfig().amp_initial_scale, 65536)


if __name__ == "__main__":
    unittest.main()
