from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from experiments.stage5 import runtime
from experiments.stage5.checkpoints import state_dict_sha256
from experiments.stage5.telemetry import TrainingTelemetry, parameter_telemetry, snapshot_parameters


class TrainingTelemetryTests(unittest.TestCase):
    def test_actual_adamw_displacement_and_zero_norm_are_observed_without_mutation(self):
        model = torch.nn.Linear(2, 1)
        with torch.no_grad():
            model.weight.copy_(torch.tensor([[3.0, 4.0]]))
            model.bias.zero_()
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.2, weight_decay=0.1)
        model(torch.ones(1, 2)).sum().backward()
        before = snapshot_parameters(model)
        optimizer.step()
        after_sha = state_dict_sha256(model.state_dict())
        gradients = [parameter.grad.clone() for parameter in model.parameters()]
        record = parameter_telemetry(model, before_update=before)
        global_stats = record["groups"]["global"]
        actual = torch.cat(
            [(p.detach().double().cpu() - before[n].double()).flatten() for n, p in model.named_parameters()]
        )
        self.assertEqual(global_stats["weight_l2"], 5.0)
        self.assertAlmostEqual(global_stats["gradient_l2"], 3**0.5)
        self.assertAlmostEqual(global_stats["update_l2"], float(actual.norm()))
        self.assertAlmostEqual(global_stats["update_to_weight_l2"], float(actual.norm()) / 5.0)
        self.assertIsNone(record["groups"]["first_bias"]["update_to_weight_l2"])
        self.assertEqual(record["groups"]["first_bias"]["update_to_weight_status"], "ZERO_WEIGHT_NORM")
        self.assertEqual(after_sha, state_dict_sha256(model.state_dict()))
        for parameter, gradient in zip(model.parameters(), gradients, strict=True):
            self.assertTrue(torch.equal(parameter.grad, gradient))
        json.dumps(record, allow_nan=False)

    def test_norms_remain_finite_when_fp32_square_would_overflow(self):
        model = torch.nn.Linear(2, 1, bias=False)
        with torch.no_grad():
            model.weight.fill_(1e30)
        model.weight.grad = torch.full_like(model.weight, 1e30)
        record = parameter_telemetry(model)
        self.assertGreater(record["groups"]["global"]["gradient_l2"], 1e30)
        json.dumps(record, allow_nan=False)

    def test_attempts_preserve_provisional_steps_and_only_commit_after_checkpoint(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = TrainingTelemetry(root, {"seed": 0}, start_epoch=0)
            for index, value in enumerate((1.0, 100.0, 4.0)):
                first.write_step(
                    context={"epoch_one_based": 1, "pair_index_one_based": index + 1, "pair": {"a": str(index)}},
                    metrics={"loss": value},
                    telemetry={"groups": {"global": {"gradient_l2": value}}},
                )
            summary = first.summarize_epoch(1)
            gradient = summary["scalars"]["parameters.global.gradient_l2"]
            self.assertEqual(gradient["mean"], 35.0)
            self.assertEqual(gradient["max"], 100.0)
            self.assertEqual(gradient["max_at"]["pair_index_one_based"], 2)
            self.assertEqual(first.epochs_path.read_text(), "")
            checkpoint = root / "last.pth"
            checkpoint.write_bytes(b"accepted epoch checkpoint")
            first.commit_epoch(1, checkpoint, "a" * 64)
            committed = json.loads(first.epochs_path.read_text())
            self.assertEqual(committed["summary"]["steps"], 3)
            first.write_step(context={"epoch_one_based": 2}, metrics={"loss": 3.0}, telemetry={})
            second = TrainingTelemetry(root, {"seed": 0}, start_epoch=1)
            self.assertNotEqual(first.root, second.root)
            self.assertEqual(len(first.steps_path.read_text().splitlines()), 4)
            self.assertEqual(len(first.epochs_path.read_text().splitlines()), 1)
            self.assertEqual(second.steps_path.read_text(), "")

    def test_production_step_writes_post_update_telemetry_outside_loss_metrics(self):
        model = torch.nn.Linear(1, 1, bias=False)
        step = SimpleNamespace(
            controller=model,
            optimizer=torch.optim.AdamW(model.parameters(), lr=0.1),
            scaler=None,
            config=runtime.ControllerTrainingConfig(),
            device=torch.device("cpu"),
            variant="F0",
        )
        with tempfile.TemporaryDirectory() as temporary:
            telemetry = TrainingTelemetry(Path(temporary), {}, start_epoch=0)
            with (
                patch.object(runtime, "_prepare_controller_pair", return_value=SimpleNamespace(bootstrap_residual=0)),
                patch.object(
                    runtime,
                    "_controller_pair_loss",
                    side_effect=lambda *args: (model.weight.square().sum(), {"loss": 1.0}),
                ),
            ):
                logs = runtime._controller_pair_step(step, {"subject_a": "A"}, 0, telemetry=telemetry)
            self.assertEqual(set(logs), {"loss", "bootstrap_digital_residual_percent"})
            record = json.loads(telemetry.steps_path.read_text())
            self.assertGreater(record["parameters"]["groups"]["global"]["update_l2"], 0)
            self.assertEqual(record["context"]["successful_update_index"], 1)

    def test_completed_controller_epochs_bind_telemetry_and_recovery_lineage(self):
        model = torch.nn.Linear(1, 1, bias=False)
        store = SimpleNamespace(runtime=SimpleNamespace(contract_sha256="b" * 64))
        config = runtime.ControllerTrainingConfig()
        with patch("experiments.stage5.config.STAGE5_CONTROLLER_FIXED_EPOCH", 2):
            config = replace(config, fixed_epoch=2)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with (
                patch.object(runtime, "_require_cuda"),
                patch.object(runtime, "Stage5OasisImageStore", return_value=store),
                patch.object(runtime, "build_stage5_controller", return_value=model),
                patch.object(runtime, "_load_initial_controller", return_value="c" * 64),
                patch.object(runtime, "_verify_checkpoint_sidecar", return_value="d" * 64),
                patch.object(runtime, "load_frozen_u0", return_value=SimpleNamespace(model=torch.nn.Linear(1, 1))),
                patch.object(runtime, "_training_subjects", return_value=("A", "B", "C", "D")),
                patch.object(runtime.torch.cuda, "get_device_name", return_value="TEST CPU"),
                patch.object(runtime, "_prepare_controller_pair", return_value=SimpleNamespace(bootstrap_residual=0)),
                patch.object(
                    runtime,
                    "_controller_pair_loss",
                    side_effect=lambda *args: (model.weight.square().sum(), {"loss": 1.0}),
                ),
            ):
                checkpoint = runtime.train_controller(
                    data_contract=root / "data.json",
                    image_root=root / "images",
                    output_root=root / "output",
                    base_checkpoint=root / "base.pth",
                    initial_controller=root / "initial.pth",
                    variant="F0",
                    seed=0,
                    device=torch.device("cpu"),
                    git_head="a" * 40,
                    protocol_sha256="a" * 64,
                    u0_training_contract_sha256="a" * 64,
                    training_contract_sha256="a" * 64,
                    bootstrap_policy="collar_repair",
                    config=config,
                )
            payload = torch.load(checkpoint, weights_only=False)
            self.assertEqual(payload["recovery_acknowledgements"], [])
            self.assertEqual(payload["epoch_completed"], 2)
            rows = payload["metrics_payload"]["epochs"]
            self.assertEqual([row["telemetry"]["steps"] for row in rows], [2, 2])
            self.assertIn("parameters.global.gradient_l2", rows[-1]["telemetry"]["scalars"])
            steps = next((root / "output").rglob("steps.jsonl"))
            self.assertEqual(len(steps.read_text().splitlines()), 4)
            committed = steps.with_name("epochs.jsonl")
            self.assertEqual(len(committed.read_text().splitlines()), 2)


if __name__ == "__main__":
    unittest.main()
