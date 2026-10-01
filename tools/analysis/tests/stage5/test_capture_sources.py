from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch

from experiments.stage5.checkpoints import state_dict_sha256
from tools.analysis.stage5 import capture_sources as worker, mechanism_contract as contract
from tools.analysis.stage5.precision_contract import SOURCE_HEAD


def small_step():
    model = torch.nn.Linear(1, 1, bias=False)
    model.weight.data.fill_(1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    return worker.runtime._ControllerStep(
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


class CaptureReaderTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.compact = self.root / "compact"
        self.compact.mkdir()
        self.step = small_step()
        state = worker.training_snapshot(self.step)
        context = {"variant": "F0", "epoch": 2, "pair_index": 23}
        heavy = self.root / "results/stage5_heavy" / contract.PRECISION_RUN / "F0/historical_failure"
        heavy.mkdir(parents=True)
        files = {
            "state.pth": {
                "schema": "ctcf-stage5-diagnostic-capture-v1",
                "context": context,
                "diagnostic_git_head": contract.PRECISION_HEAD,
                "source_git_head": SOURCE_HEAD,
                "production_resume_forbidden": True,
                "training_state": state,
                "bootstrap_residual": 0.0,
            }
        }
        for name in ["psi_ab", "psi_ba", *[f"tensors_{d}_{i}" for d in ("ab", "ba") for i in range(5)]]:
            files[f"{name}.pth"] = {"tensor": torch.ones(1, 3, 2, 2, 2)}
        records = []
        for name, content in files.items():
            path = heavy / name
            torch.save(content, path)
            records.append(
                {
                    "path": path.relative_to(self.root).as_posix(),
                    "sha256": worker.sha256_file(path),
                    "bytes": path.stat().st_size,
                }
            )
        capture = {"context": context, "records": records, "exact_prepared_inputs_saved": True}
        (heavy / "manifest.json").write_text(json.dumps(capture))
        for job in contract.JOBS:
            data = {
                "job": job,
                "status": "DIAGNOSTIC_COMPLETE",
                "diagnostic_git_head": contract.PRECISION_HEAD,
                "source_bytes_unchanged": True,
                "failure_reproduced": True,
                "heavy_captures": [capture],
                "comparison": {"initial_model_sha256": state_dict_sha256(state["model"])},
            }
            (self.compact / f"{job}.json").write_text(json.dumps(data))
        (self.compact / "git_head.txt").write_text(contract.PRECISION_HEAD)
        sums = "".join(f"{worker.sha256_file(p)}  {p.name}\n" for p in sorted(self.compact.iterdir()))
        (self.compact / "SHA256SUMS").write_text(sums)
        self.addCleanup(mock.patch.stopall)
        mock.patch.object(contract, "PRECISION_SUMS_SHA256", worker.sha256_file(self.compact / "SHA256SUMS")).start()
        self.heavy = heavy

    def test_verified_saved_inputs_load_without_u0_or_image_store(self):
        metadata = {}
        source = worker.CaptureSources(self.root, self.compact, metadata)
        with (
            mock.patch.object(
                worker, "build_stage5_controller", side_effect=lambda config: torch.nn.Linear(1, 1, bias=False)
            ),
            mock.patch.object(
                worker.runtime, "_stage5_grad_scaler", side_effect=lambda config: torch.amp.GradScaler("cpu")
            ),
        ):
            step, inputs = source.load_failure("F0", device=torch.device("cpu"))
        self.assertTrue(torch.equal(step.controller.weight, self.step.controller.weight))
        self.assertEqual(len(inputs.tensors_ab), 5)
        self.assertTrue(metadata["saved_failure"]["exact_input_bytes_verified"])
        source.verify_unchanged()
        self.assertTrue(metadata["mechanism_sources_unchanged"])

    def test_changed_capture_rejected_before_deserialization(self):
        source = worker.CaptureSources(self.root, self.compact, {})
        (self.heavy / "state.pth").write_bytes(b"corrupt")
        with (
            mock.patch.object(torch, "load", side_effect=AssertionError("unverified pickle loaded")),
            self.assertRaisesRegex(RuntimeError, "bytes differ"),
        ):
            source.load_failure("F0", device=torch.device("cpu"))

    def test_custom_capture_root_keeps_reviewed_hashes_and_original_record_prefix(self):
        relocated = self.root / "retained-elsewhere"
        shutil.copytree(self.heavy, relocated / "F0/historical_failure")
        (self.heavy / "state.pth").write_bytes(b"do not read original location")
        source = worker.CaptureSources(self.root, self.compact, {})
        with (
            mock.patch.object(
                worker, "build_stage5_controller", side_effect=lambda config: torch.nn.Linear(1, 1, bias=False)
            ),
            mock.patch.object(
                worker.runtime, "_stage5_grad_scaler", side_effect=lambda config: torch.amp.GradScaler("cpu")
            ),
        ):
            loaded, _ = source.load_failure("F0", device=torch.device("cpu"), capture_root=relocated)
        self.assertTrue(torch.equal(loaded.controller.weight, self.step.controller.weight))
        source.verify_unchanged()
        (relocated / "F0/historical_failure/state.pth").write_bytes(b"changed")
        with self.assertRaisesRegex(RuntimeError, "bytes differ"):
            source.load_failure("F0", device=torch.device("cpu"), capture_root=relocated)

    def test_compact_changes_and_path_escapes_rejected(self):
        (self.compact / "F0.json").write_text("{}")
        with self.assertRaisesRegex(RuntimeError, "member differs"):
            worker.CaptureSources(self.root, self.compact, {})
        with self.assertRaisesRegex(RuntimeError, "escapes"):
            worker.CaptureSources.contained(self.compact, "../outside")


class SnapshotTests(unittest.TestCase):
    def test_snapshot_restores_parameters_adam_state_scaler_and_rng(self):
        step = small_step()
        step.controller(torch.ones(1, 1)).sum().backward()
        step.optimizer.step()
        state = worker.training_snapshot(step)
        expected_rng = worker.capture_rng_state()["torch_cpu"]
        step.controller.weight.data.add_(3)
        next(iter(step.optimizer.state.values()))["exp_avg"].add_(5)
        torch.rand(4)
        worker.restore_snapshot(step, state)
        self.assertTrue(torch.equal(step.controller.weight, state["model"]["weight"]))
        restored = step.optimizer.state_dict()["state"][0]
        for key, value in state["optimizer"]["state"][0].items():
            self.assertTrue(torch.equal(restored[key], value))
        self.assertTrue(torch.equal(torch.get_rng_state(), expected_rng))
        self.assertIsNone(step.controller.weight.grad)
