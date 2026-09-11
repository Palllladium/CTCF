from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from experiments.stage5 import runtime
from experiments.stage5.checkpoints import state_dict_sha256
from tools.analysis import compare_stage5_precision as worker
from tools.analysis.stage5 import comparison_contract as contract
from tools.analysis.stage5.comparison_math import (
    ArithmeticObserver,
    advance,
    mapping_difference,
    restore,
    snapshot,
    tensor_difference,
)
from tools.analysis.stage5.comparison_summary import summarize_comparison_results


class ComparisonMathTest(unittest.TestCase):
    def test_zero_reference_is_explicit_and_nonfinite_is_not_accepted(self):
        result = tensor_difference(torch.ones(3), torch.zeros(3))
        self.assertTrue(result["reference_zero"])
        self.assertIsNone(result["relative_l2"])
        self.assertIsNone(result["cosine"])
        self.assertAlmostEqual(result["rms"], 1)
        self.assertNotIn("within_heuristic_tolerance", result)
        self.assertEqual(tensor_difference(torch.tensor([float("inf")]), torch.ones(1))["status"], "NONFINITE")

    def test_chunk_reduction_and_inventory_guards(self):
        actual = torch.ones(1_000_007)
        reference = torch.ones_like(actual) * 2
        result = tensor_difference(actual, reference)
        self.assertAlmostEqual(result["relative_l2"], 0.5)
        self.assertAlmostEqual(result["cosine"], 1)
        with self.assertRaisesRegex(ValueError, "inventories"):
            mapping_difference({"a": torch.ones(1)}, {"b": torch.ones(1)})
        with self.assertRaisesRegex(ValueError, "shapes"):
            tensor_difference(torch.ones(1), torch.ones(2))

    def test_shared_counts_follow_contract(self):
        self.assertEqual(contract.expected_updates("coverage"), len(contract.MODES) * contract.COVERAGE_UPDATES)
        self.assertEqual(contract.expected_updates("trajectory"), contract.EPOCHS * contract.TRAINING_SUBJECTS // 2)
        self.assertEqual(contract.expected_updates("paired"), len(contract.PAIRED_MODES))
        with self.assertRaises(ValueError):
            contract.expected_updates("trajectory", 292)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_same_state_probe_observes_fields_gradients_and_real_modes(self):
        device = torch.device("cuda:0")
        config = runtime.ControllerTrainingConfig()
        model = runtime.build_stage5_controller(config).to(device)
        step = SimpleNamespace(
            controller=model,
            optimizer=torch.optim.AdamW(model.parameters(), lr=1e-4),
            scaler=None,
            config=config,
            device=device,
            variant="F2P",
        )
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
        initial = snapshot(step)
        initial_hash = state_dict_sha256(initial["model"])
        for mode in contract.MODES:
            with self.subTest(mode=mode):
                restore(step, initial)
                self.assertEqual(state_dict_sha256(step.controller.state_dict()), initial_hash)
                with ArithmeticObserver(model, capture_fields=True) as observer:
                    logs = advance(step, inputs, mode)
                self.assertEqual(observer.validate(mode)["directional_forward_calls"], 2)
                self.assertEqual(set(observer.fields), {"forward", "reverse"})
                self.assertEqual(set(observer.field_gradients), {"forward", "reverse"})
                self.assertTrue(all(torch.isfinite(value).all() for value in observer.field_gradients.values()))
                self.assertIsInstance(logs["loss"], float)
                self.assertNotEqual(state_dict_sha256(step.controller.state_dict()), initial_hash)
        restore(step, initial)
        self.assertEqual(state_dict_sha256(step.controller.state_dict()), initial_hash)

    def test_nonfinite_gradient_rejects_update_before_optimizer(self):
        model = torch.nn.Linear(1, 1)
        step = SimpleNamespace(
            controller=model, optimizer=torch.optim.AdamW(model.parameters()), scaler=None, device=torch.device("cpu")
        )
        before = state_dict_sha256(model.state_dict())
        inputs = SimpleNamespace(bootstrap_residual=0)
        hook = model.weight.register_hook(lambda value: value * float("inf"))
        try:
            with (
                patch.object(
                    runtime,
                    "_controller_pair_loss",
                    side_effect=lambda *a: (model.weight.square().sum(), {"loss": 1.0}),
                ),
                self.assertRaisesRegex(FloatingPointError, "gradient"),
            ):
                advance(step, inputs, "fp32_strict")
        finally:
            hook.remove()
        self.assertEqual(state_dict_sha256(model.state_dict()), before)
        self.assertEqual(step.optimizer.state_dict()["state"], {})

    def test_failure_capture_names_its_actual_mode(self):
        model = torch.nn.Linear(1, 1)
        step = SimpleNamespace(controller=model, optimizer=torch.optim.AdamW(model.parameters()))
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            args = SimpleNamespace(heavy_root=root / "heavy", output_root=root / "compact", expected_git_head="a" * 40)
            args.output_root.mkdir()
            record = worker.failure_record(
                args, step, None, {"phase": "backward"}, FloatingPointError("bad"), {"mode": "bf16"}
            )
            self.assertEqual(record["precision"]["mode"], "bf16")
            self.assertEqual(record["capture_status"], "COMPLETE")
            payload = torch.load(record["capture_file"]["path"], weights_only=False)
            self.assertTrue(payload["production_resume_forbidden"])
            self.assertEqual(payload["capture_status"], "COMPLETE")
            self.assertNotIn("role", payload)

    def test_initialization_oom_is_an_observed_resource_failure(self):
        error = RuntimeError("Could not reconstruct U0")
        error.__cause__ = RuntimeError("CUDNN_STATUS_ALLOC_FAILED")
        sources = SimpleNamespace(new_step=lambda: (_ for _ in ()).throw(error))
        with (
            patch.object(torch.cuda, "reset_peak_memory_stats"),
            patch.object(torch.cuda, "max_memory_allocated", return_value=1234),
            patch.object(torch.cuda, "max_memory_reserved", return_value=2048),
            patch.object(torch.cuda, "empty_cache"),
        ):
            record = worker.train_case(SimpleNamespace(), sources, mode="fp32_strict")
        self.assertEqual(record["status"], "CANDIDATE_FAILURE")
        self.assertEqual(record["failure"]["phase"], "initialize")
        self.assertEqual(record["peak_allocated_bytes"], 1234)
        self.assertEqual(record["successful_updates"], 0)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_two_epoch_worker_separates_telemetry_and_diagnostic_endpoint(self):
        device = torch.device("cuda:0")
        config = runtime.ControllerTrainingConfig()
        controller = runtime.build_stage5_controller(config).to(device)
        step = SimpleNamespace(
            controller=controller,
            optimizer=torch.optim.AdamW(controller.parameters(), lr=1e-4),
            scaler=None,
            config=config,
            device=device,
            variant="F0",
        )
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
        pairs = [
            {"pair_id": "p0", "subject_a": "a", "subject_b": "b"},
            {"pair_id": "p1", "subject_a": "c", "subject_b": "d"},
        ]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            args = SimpleNamespace(
                output_root=root / "compact",
                heavy_root=root / "heavy",
                seed=0,
                variant="F0",
                expected_git_head="a" * 40,
            )
            args.output_root.mkdir()
            sources = SimpleNamespace(new_step=lambda: step, subjects=("a", "b", "c", "d"), protocol_sha="b" * 64)
            with (
                patch.object(contract, "expected_updates", return_value=4),
                patch.object(runtime, "controller_epoch_pairs", return_value=pairs),
                patch.object(runtime, "_prepare_controller_pair", return_value=inputs),
            ):
                record = worker.train_case(args, sources, mode="fp32_strict")
            self.assertEqual(record["status"], "COMPLETE")
            self.assertEqual(record["successful_updates"], 4)
            self.assertEqual(len(record["epochs"]), 2)
            rows = [json.loads(line) for line in (args.output_root / record["step_records"]).read_text().splitlines()]
            self.assertEqual(len(rows), 4)
            self.assertEqual(rows[-1]["epoch"], 2)
            self.assertIn("global", rows[-1]["telemetry"]["groups"])
            state = torch.load(record["diagnostic_state"]["path"], weights_only=False)
            self.assertTrue(state["production_resume_forbidden"])
            self.assertNotIn("role", state)
            self.assertEqual(state["successful_updates"], 4)


class ComparisonSummaryTest(unittest.TestCase):
    def make_record(self, root, mode, replicate, values):
        output = root / f"{mode}_{replicate}"
        output.mkdir()
        rows = [
            {"epoch": 1, "pair": {"pair_id": str(index)}, "metrics": {"loss": value}}
            for index, value in enumerate(values)
        ]
        (output / "steps.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
        return {
            "spec": {
                "job": {
                    "stage": "trajectory",
                    "variant": "F0",
                    "seed": 0,
                    "mode": mode,
                    "replicate": replicate,
                    "gpu": "3",
                }
            },
            "output_root": str(output),
            "result": {
                "status": "COMPLETE",
                "successful_updates": len(rows),
                "step_records": "steps.jsonl",
                "initial_model_sha256": "same",
                "full_step_seconds": [2.0],
                "compute_seconds": [1.0],
                "preparation_seconds": [1.0],
            },
        }

    def test_fp32_repeat_is_reported_without_a_quality_decision(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            records = [
                self.make_record(root, "fp32_strict", 0, [1, 2]),
                self.make_record(root, "fp32_strict", 1, [1.1, 1.9]),
                self.make_record(root, "bf16", 0, [1.2, 1.8]),
            ]
            result = summarize_comparison_results(records)
            self.assertFalse(result["automatic_mode_selection"])
            self.assertFalse(result["training_quality_assessed"])
            self.assertAlmostEqual(
                result["trajectories"][1]["trajectory_metric_difference_vs_strict"]["loss"]["mean_absolute"], 0.1
            )
            records[2]["spec"]["job"]["gpu"] = "2"
            with self.assertRaisesRegex(ValueError, "physical GPU"):
                summarize_comparison_results(records)

    def test_incomplete_trajectory_does_not_get_speed_ranking(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            records = [self.make_record(root, "fp32_strict", 0, [1]), self.make_record(root, "bf16", 0, [1])]
            records[1]["result"]["status"] = "CANDIDATE_FAILURE"
            result = summarize_comparison_results(records)
            self.assertNotIn("full_step_time_ratio_vs_strict", result["trajectories"][1])


if __name__ == "__main__":
    unittest.main()
