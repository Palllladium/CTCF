from __future__ import annotations

import copy
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from experiments.stage5 import runtime
from experiments.stage5.checkpoints import build_training_state
from experiments.stage5.precision import controller_precision_contract, precision_context, precision_mode_contract
from tools.analysis.run_artifacts import sha256_file
from tools.analysis.stage5 import artifacts, pipeline
from tools.analysis.stage5.contracts import build_protocol_contract
from tools.analysis.stage5.primitives import canonical_sha256, readable_json_bytes
from tools.analysis.stage5.protocol import controller_training_contract


def flags():
    return torch.get_float32_matmul_precision(), torch.backends.cudnn.allow_tf32, torch.is_autocast_enabled("cpu")


class ProductionTf32BoundaryTest(unittest.TestCase):
    def test_production_uses_the_measured_tf32_contract(self):
        expected = precision_mode_contract("tf32")
        expected["schema"] = "ctcf-stage5-controller-precision-v2"
        self.assertEqual(controller_precision_contract(), expected)

    def test_success_and_failure_do_not_change_the_next_frozen_u0_preparation(self):
        model = torch.nn.Linear(1, 1)
        step = SimpleNamespace(
            controller=model,
            optimizer=torch.optim.AdamW(model.parameters()),
            scaler=None,
            config=runtime.ControllerTrainingConfig(),
            device=torch.device("cpu"),
            variant="F0",
        )
        preparations, updates = [], []

        def prepare(*args):
            preparations.append(flags())
            return SimpleNamespace(bootstrap_residual=0)

        def loss(*args):
            updates.append(flags())
            return model.weight.square().sum(), {"loss": 1.0}

        with (
            precision_context(step.device, "fp32_strict"),
            patch.object(runtime, "_prepare_controller_pair", side_effect=prepare),
            patch.object(runtime, "_controller_pair_loss", side_effect=loss),
        ):
            runtime._controller_pair_step(step, {}, 0)
            with (
                patch.object(step.optimizer, "step", side_effect=OSError("test disk failure")),
                self.assertRaisesRegex(OSError, "test disk failure"),
            ):
                runtime._controller_pair_step(step, {}, 0)
            runtime._controller_pair_step(step, {}, 0)
            self.assertEqual(flags(), ("highest", False, False))
        self.assertEqual(preparations, [("highest", False, False)] * 3)
        self.assertEqual(updates, [("high", True, False)] * 3)

    def test_decision_features_and_safety_remain_outside_tf32(self):
        observed = []
        tensor = torch.zeros(1)

        def features(*args):
            observed.append(("features", flags()))
            return SimpleNamespace(
                controller_input=tensor, s2=SimpleNamespace(proposal=tensor), s4=SimpleNamespace(proposal=tensor)
            )

        def forward(*args, **kwargs):
            observed.append(("controller", flags()))
            return SimpleNamespace(requested_delta=tensor)

        def transaction(*args):
            observed.append(("transaction", flags()))
            # Stop before file IO: this test concerns the real inference boundary.
            raise OSError("transaction reached")

        context = SimpleNamespace(controller=forward, store=None, device=torch.device("cpu"), variant="F0")
        with (
            precision_context(context.device, "fp32_strict"),
            patch.object(pipeline, "_case_images", return_value=(tensor, tensor)),
            patch.object(pipeline, "build_stage5_features", side_effect=features),
            patch.object(pipeline, "commit_controller_delta", side_effect=transaction),
            self.assertRaisesRegex(OSError, "transaction reached"),
        ):
            pipeline._controller_outcome(context, {}, tensor, Path("source.npz"), {}, Path("decision"))
        self.assertEqual(
            observed,
            [
                ("features", ("highest", False, False)),
                ("controller", ("high", True, False)),
                ("transaction", ("highest", False, False)),
            ],
        )


class ProductionTf32CheckpointTest(unittest.TestCase):
    def test_saved_precision_is_enforced_on_resume_barrier_and_decision(self):
        config = runtime.ControllerTrainingConfig()
        protocol = build_protocol_contract(
            git_head="a" * 40,
            data_contract_sha256="b" * 64,
            u0_training_contract_sha256="c" * 64,
            controller_training_contract_sha256=canonical_sha256(controller_training_contract(config)),
            search_contract_sha256="d" * 64,
            directed_case_ids=("case",),
            metric_ids=("metric",),
            u0_fixed_epoch=400,
            controller_fixed_epoch=100,
            bootstrap_policy="identity",
            bootstrap_parameters={},
        )
        model = torch.nn.Linear(2, 1)
        metrics = {
            "schema": "ctcf-stage5-controller-metrics-v1",
            "role": "CONTROLLER",
            "variant": "F0",
            "seed": 0,
            "label_metrics_present": False,
            "selection_policy": protocol["checkpoint_selection_policy"],
            "epochs": [
                {"epoch": epoch, "pairs": 1, "pair_schedule_sha256": "e" * 64, "metrics": {"loss": 0.5}}
                for epoch in range(1, 101)
            ],
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path, metrics_path = root / "last.pth", root / "metrics.json"
            metrics_path.write_bytes(readable_json_bytes(metrics))
            payload = build_training_state(
                role="CONTROLLER",
                variant_id="F0",
                seed=0,
                epoch_completed=100,
                fixed_epoch=100,
                git_head=protocol["git_head"],
                protocol_sha256=canonical_sha256(protocol),
                data_contract_sha256=protocol["data_contract_sha256"],
                training_contract_sha256=protocol["controller_training_contract_sha256"],
                model=model,
                optimizer=torch.optim.AdamW(model.parameters()),
                scaler=None,
                pair_schedule_sha256="e" * 64,
                metrics_sha256=sha256_file(metrics_path),
                base_checkpoint_sha256="f" * 64,
                initial_controller_state_sha256="1" * 64,
                source_contract_sha256="2" * 64,
            )
            runtime._attach_runtime_checkpoint_metadata(payload, config=config, metrics_payload=metrics)
            torch.save(payload, path)
            loaded = torch.load(path, weights_only=False)

            def resume(state):
                return runtime._validate_runtime_checkpoint_metadata(
                    state,
                    role="CONTROLLER",
                    variant="F0",
                    seed=0,
                    config=config,
                    expected_git_head=protocol["git_head"],
                    expected_base_checkpoint_sha256="f" * 64,
                    expected_initial_controller_state_sha256="1" * 64,
                    expected_source_contract_sha256="2" * 64,
                )

            def barrier():
                with patch.object(artifacts, "build_stage5_controller", return_value=torch.nn.Linear(2, 1)):
                    return artifacts.checkpoint_metadata(
                        checkpoint_id="s0_F0",
                        checkpoint_path=path,
                        checkpoint_root=root,
                        metrics_path=metrics_path,
                        protocol=protocol,
                    )

            def decision(metadata):
                with patch.object(pipeline, "build_stage5_controller", return_value=torch.nn.Linear(2, 1)):
                    return pipeline._load_controller(
                        path,
                        metadata=metadata,
                        protocol=protocol,
                        device=torch.device("cpu"),
                        config=config,
                    )

            self.assertEqual(loaded["controller_precision"]["mode"], "tf32")
            self.assertEqual(resume(loaded), metrics)
            metadata = barrier()
            self.assertFalse(decision(metadata).training)
            for mode in ("fp32_strict", "bf16", None):
                with self.subTest(mode=mode):
                    invalid = copy.deepcopy(loaded)
                    invalid["controller_precision"] = (
                        {**precision_mode_contract(mode), "schema": loaded["controller_precision"]["schema"]}
                        if mode
                        else None
                    )
                    torch.save(invalid, path)
                    with self.assertRaisesRegex(RuntimeError, "precision contract"):
                        resume(invalid)
                    with self.assertRaisesRegex(RuntimeError, "TF32"):
                        barrier()
                    # Authenticate the edited file so decision rejection is about precision, not file integrity.
                    edited_metadata = copy.deepcopy(metadata)
                    edited_metadata["checkpoint_file"].update(sha256=sha256_file(path), bytes=path.stat().st_size)
                    with self.assertRaisesRegex(RuntimeError, "TF32"):
                        decision(edited_metadata)


if __name__ == "__main__":
    unittest.main()
