from __future__ import annotations

import dataclasses
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch

from experiments.stage5 import runtime
from experiments.stage5.checkpoints import capture_rng_state
from models.CTCF.controller import STAGE5_VARIANTS
from tools.analysis import diagnose_stage5_precision as diagnostic


def small_step():
    model = torch.nn.Linear(1, 1, bias=False)
    model.weight.data.fill_(1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    return runtime._ControllerStep(
        store=None,
        base_runner=None,
        controller=model,
        optimizer=optimizer,
        scaler=torch.amp.GradScaler("cpu", init_scale=65536),
        device=torch.device("cpu"),
        variant="F0",
        bootstrap_policy="collar_repair",
        config=None,
    )


class PrecisionUpdateTests(unittest.TestCase):
    def test_rejected_scaled_gradient_never_updates_optimizer_or_scaler(self):
        step = small_step()
        before = diagnostic.training_snapshot(step)

        def loss(_step, _inputs, *, diagnostic_fp32):
            value = _step.controller(torch.ones(1, 1))
            if not diagnostic_fp32:
                value = value.half().float()
            return value.sum() * 2, {"loss": float(value.detach().sum() * 2)}

        with (
            mock.patch.object(runtime, "_controller_pair_loss", side_effect=loss),
            mock.patch.object(step.optimizer, "step", side_effect=AssertionError("bad update accepted")),
            mock.patch.object(step.scaler, "update", side_effect=AssertionError("scale changed")),
            self.assertRaisesRegex(FloatingPointError, "before optimizer update"),
        ):
            diagnostic.advance_pair(step, None, fp32=False)
        self.assertTrue(torch.equal(step.controller.weight, before["model"]["weight"]))
        self.assertEqual(step.optimizer.state_dict(), before["optimizer"])
        self.assertEqual(step.scaler.state_dict(), before["scaler"])

    def test_fp32_has_no_gradient_scaling_and_performs_an_update(self):
        step = small_step()
        with (
            mock.patch.object(
                runtime,
                "_controller_pair_loss",
                side_effect=lambda state, _inputs, **_kwargs: (state.controller(torch.ones(1, 1)).sum() * 2, {}),
            ),
            mock.patch.object(step.scaler, "scale", side_effect=AssertionError("FP32 used scaling")),
        ):
            diagnostic.advance_pair(step, None, fp32=True)
        self.assertLess(float(step.controller.weight.detach()), 1)
        self.assertTrue(step.optimizer.state)

    def test_snapshot_restores_parameters_adam_state_scaler_and_rng(self):
        step = small_step()
        step.controller(torch.ones(1, 1)).sum().backward()
        step.optimizer.step()
        state = diagnostic.training_snapshot(step)
        expected_rng = capture_rng_state()["torch_cpu"]
        step.controller.weight.data.add_(3)
        next(iter(step.optimizer.state.values()))["exp_avg"].add_(5)
        torch.rand(4)
        diagnostic.restore_snapshot(step, state)
        self.assertTrue(torch.equal(step.controller.weight, state["model"]["weight"]))
        restored = step.optimizer.state_dict()["state"][0]
        for key, value in state["optimizer"]["state"][0].items():
            self.assertTrue(torch.equal(restored[key], value))
        self.assertTrue(torch.equal(torch.get_rng_state(), expected_rng))
        self.assertIsNone(step.controller.weight.grad)


class CoverageTests(unittest.TestCase):
    def test_all_variants_and_seeds_start_from_same_per_seed_initial_state(self):
        starts = []
        subjects = tuple(str(index) for index in range(16))
        pairs = [{"subject_a": subjects[index], "subject_b": subjects[index + 1]} for index in range(0, 16, 2)]
        update_counts = {}

        def advance(step, _inputs, *, fp32):
            self.assertTrue(fp32)
            if step.variant not in update_counts:
                update_counts[step.variant] = 0
            if update_counts[step.variant] % 8 == 0:
                starts.append((step.variant, float(step.controller.weight.detach())))
            step.controller.weight.data.add_(1)
            update_counts[step.variant] += 1
            return {"loss": -0.2}

        with (
            mock.patch.object(diagnostic, "prepare_step", side_effect=lambda *a, **k: small_step()),
            mock.patch.object(runtime, "controller_epoch_pairs", return_value=pairs),
            mock.patch.object(runtime, "_prepare_controller_pair", return_value=None),
            mock.patch.object(diagnostic, "advance_pair", side_effect=advance),
            mock.patch.object(torch.cuda, "empty_cache"),
        ):
            report = {}
            diagnostic.run_coverage(SimpleNamespace(), report, SimpleNamespace(subjects=subjects), lambda: None)
        cases = report["coverage"]["cases"]
        self.assertEqual(
            {(row["seed"], row["variant"]) for row in cases}, {(s, v) for s in (0, 1, 2) for v in STAGE5_VARIANTS}
        )
        self.assertEqual(len(cases), 24)
        self.assertTrue(all(row["status"] == "COMPLETE" and row["completed_updates"] == 8 for row in cases))
        self.assertEqual(len(starts), 24)
        self.assertTrue(all(value == 1 for _, value in starts))

    def test_step_collaborators_remain_frozen(self):
        step = small_step()
        with self.assertRaises(dataclasses.FrozenInstanceError):
            step.variant = "F2S"


class OutputAndLifecycleTests(unittest.TestCase):
    def test_outputs_cannot_overlap_sources_or_include_heavy_in_zip(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            args = SimpleNamespace(
                source_root=root / "source",
                checkpoint_root=root / "weights",
                image_root=root / "images",
                data_contract=root / "data" / "contract.json",
                output=root / "diagnostic" / "F0.json",
                capture_root=root / "heavy",
            )
            diagnostic._validate_output_roots(args)
            args.capture_root = args.output.parent / "heavy"
            with self.assertRaisesRegex(RuntimeError, "Heavy captures"):
                diagnostic._validate_output_roots(args)
            args.capture_root = root / "heavy"
            args.output = args.source_root / "accidental.json"
            with self.assertRaisesRegex(RuntimeError, "disjoint"):
                diagnostic._validate_output_roots(args)

    def test_independent_phases_continue_after_replay_error(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "compact" / "F0.json"
            arguments = [
                "diagnostic",
                "--expected-git-head",
                "a" * 40,
                "--source-root",
                str(root / "source"),
                "--checkpoint-root",
                str(root / "weights"),
                "--image-root",
                str(root / "images"),
                "--data-contract",
                str(root / "data" / "contract.json"),
                "--capture-root",
                str(root / "heavy"),
                "--job",
                "F0",
                "--output",
                str(output),
            ]
            with (
                mock.patch("sys.argv", arguments),
                mock.patch.object(diagnostic, "assert_clean_exact_git", return_value="a" * 40),
                mock.patch.object(runtime, "_require_cuda"),
                mock.patch.object(torch.cuda, "get_device_name", return_value="test GPU"),
                mock.patch.object(torch.cuda, "get_device_properties", return_value=SimpleNamespace(total_memory=123)),
                mock.patch.object(torch.cuda, "empty_cache"),
                mock.patch.object(diagnostic, "Sources"),
                mock.patch.object(diagnostic, "replay_failure", side_effect=RuntimeError("replay unavailable")),
                mock.patch.object(diagnostic, "run_trajectory") as trajectory,
                mock.patch.object(diagnostic, "benchmark") as benchmark,
            ):
                diagnostic.main()
            trajectory.assert_called_once()
            benchmark.assert_called_once()
            result = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(result["replay"]["status"], "INCOMPLETE")
            self.assertIn("replay unavailable", result["replay"]["failure"]["error"])
            self.assertIsNone(result["source_bytes_unchanged"])
            self.assertFalse(result["production_training_validated"])


if __name__ == "__main__":
    torch.set_num_threads(2)
    unittest.main()
