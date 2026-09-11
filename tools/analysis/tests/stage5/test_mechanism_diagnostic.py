from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch

from experiments.stage5.checkpoints import state_dict_sha256
from tools.analysis import diagnose_stage5_mechanism as worker
from tools.analysis.stage5 import mechanism_contract as contract, mechanism_report as report
from tools.analysis.stage5.precision_contract import SOURCE_HEAD
from tools.analysis.tests.stage5.test_mechanism_report import complete_audits
from tools.analysis.tests.stage5.test_precision_diagnostic import small_step

HEAD = "b" * 40


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


class F2PReplayTests(unittest.TestCase):
    def test_observer_preserves_pre_skip_state_with_real_production_scaler(self):
        model = torch.nn.Module()
        model.stem = torch.nn.Sequential(torch.nn.Linear(1, 16))
        step = SimpleNamespace(
            controller=model,
            optimizer=torch.optim.AdamW(model.parameters()),
            scaler=torch.amp.GradScaler("cpu", init_scale=65536),
            variant="F2P",
            device=torch.device("cpu"),
        )
        before = worker.training_snapshot(step)

        def loss(current, inputs):
            value = current.controller.stem(torch.ones(1, 1)).half().float().sum() * 4
            return value, {"loss": float(value.detach())}

        with (
            mock.patch.object(
                worker.runtime, "_prepare_controller_pair", return_value=SimpleNamespace(bootstrap_residual=0.0)
            ),
            mock.patch.object(worker.runtime, "_controller_pair_loss", side_effect=loss),
            worker.ReplayObserver(step) as observer,
            self.assertRaises(FloatingPointError),
        ):
            worker.runtime._legacy_controller_pair_step(step, {}, 0)
        self.assertEqual(step.scaler.get_scale(), 32768)
        saved = observer.current["failure_state"]
        self.assertEqual(saved["scaler"]["scale"], 65536)
        self.assertEqual(state_dict_sha256(saved["model"]), state_dict_sha256(before["model"]))
        self.assertEqual(saved["optimizer"], before["optimizer"])
        worker.restore_snapshot(step, saved)
        self.assertEqual(step.scaler.get_scale(), 65536)

    def run_replay(self, *, fail):
        counter = []
        steps = []

        def make_step(*args, **kwargs):
            model = torch.nn.Module()
            model.stem = torch.nn.Sequential(torch.nn.Linear(1, 16))
            step = SimpleNamespace(
                controller=model,
                optimizer=torch.optim.AdamW(model.parameters()),
                scaler=torch.amp.GradScaler("cpu"),
                variant="F2P",
                device=torch.device("cpu"),
            )
            steps.append(step)
            return step

        def production_step(step, pair, epoch):
            worker.runtime._prepare_controller_pair(step, pair, epoch)
            step.optimizer.zero_grad(set_to_none=True)
            loss = step.controller.stem(torch.ones(1, 1)).sum()
            if fail == "forward":
                raise FloatingPointError("objective guard before backward")
            if fail == "observation":
                return {"loss": float(loss.detach())}
            if fail:
                loss = loss * torch.tensor(float("nan"))
            loss.backward()
            worker.runtime._strict_scaler_step(step.scaler, step.optimizer, phase="controller F2P")
            return {"loss": float(loss.detach())}

        def strict_step(scaler, optimizer, **kwargs):
            counter.append("production_scaler_called")
            if fail:
                raise FloatingPointError("nonfinite production skip")
            optimizer.step()

        pairs = [{"pair_id": str(i), "subject_a": "a", "subject_b": "b"} for i in range(2)]
        metadata = {}
        with (
            mock.patch.object(worker, "prepare_step", side_effect=make_step),
            mock.patch.object(worker.runtime, "controller_epoch_pairs", return_value=pairs),
            mock.patch.object(worker.runtime, "_prepare_controller_pair", return_value=SimpleNamespace()),
            mock.patch.object(worker.runtime, "_legacy_controller_pair_step", side_effect=production_step),
            mock.patch.object(worker.runtime, "_strict_scaler_step", side_effect=strict_step),
            mock.patch.object(worker, "_try_capture") as capture,
            mock.patch.object(worker, "run_audits") as audit,
        ):
            worker.replay_f2p(SimpleNamespace(), SimpleNamespace(subjects=("a", "b")), metadata, lambda: None)
        return metadata, counter, steps, capture, audit

    def test_two_bounded_production_replays_without_failure(self):
        metadata, calls, steps, capture, audit = self.run_replay(fail=False)
        self.assertEqual(len(calls), 4)
        self.assertEqual(len(steps), contract.F2P_ATTEMPTS)
        self.assertTrue(
            all(a["status"] == "NOT_REPRODUCED" and a["completed_updates"] == 2 for a in metadata["f2p_replay"])
        )
        capture.assert_not_called()
        audit.assert_not_called()

    def test_first_failure_stops_retries_and_runs_saved_state_audits(self):
        metadata, calls, steps, capture, audit = self.run_replay(fail=True)
        self.assertEqual(len(calls), 1)
        self.assertEqual(len(steps), 1)
        self.assertEqual(metadata["f2p_replay"][0]["status"], "FAILURE_CAPTURED")
        capture.assert_called_once()
        audit.assert_called_once()
        self.assertIn("stem.0.bias", metadata["f2p_replay"][0]["gradient_observation"]["bad_parameters"])

    def test_forward_math_guard_preserves_available_state_without_claiming_amp_reproduction(self):
        metadata, calls, _steps, capture, audit = self.run_replay(fail="forward")
        self.assertEqual(calls, [])
        attempt = metadata["f2p_replay"][0]
        self.assertEqual(attempt["status"], "FAILURE_CAPTURED")
        self.assertEqual(attempt["failure"]["stage"], "forward_or_backward")
        self.assertFalse(attempt["historical_gradient_failure_reproduced"])
        capture.assert_called_once()
        audit.assert_called_once()

    def test_missing_gradient_observation_stops_replay_as_incomplete(self):
        metadata, calls, steps, capture, audit = self.run_replay(fail="observation")
        self.assertEqual(calls, [])
        self.assertEqual(len(steps), 1)
        attempt = metadata["f2p_replay"][0]
        self.assertEqual(attempt["status"], "OTHER_FAILURE")
        self.assertEqual(attempt["completed_updates"], 0)
        self.assertIn("required gradient observation", attempt["failure"]["error"])
        capture.assert_not_called()
        audit.assert_not_called()


class MechanismReportTests(unittest.TestCase):
    def aggregate(self, *, attempts=None, audit_status="COMPLETE", source_ok=True):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for job in contract.JOBS:
                payload = {
                    "schema": contract.SCHEMA,
                    "job": job,
                    "diagnostic_git_head": HEAD,
                    "workload_contract": contract.workload_contract(),
                    "precision_package_verified": True,
                    "mechanism_sources_unchanged": source_ok,
                    "production_checkpoint_written": False,
                    "labels_accessed": False,
                    "status": "DIAGNOSTIC_COMPLETE",
                    **complete_audits(),
                }
                payload["environment"] = {"backend_flags": payload["bias_audit"]["original_backend_flags"]}
                payload["bias_audit"]["status"] = audit_status
                if job == "F2P":
                    payload["f2p_replay"] = (
                        attempts
                        if attempts is not None
                        else [
                            {
                                "attempt": i + 1,
                                "status": "NOT_REPRODUCED",
                                "completed_updates": contract.F2P_UPDATES,
                                "expected_pairs": contract.F2P_UPDATES,
                                "initial_model_sha256": "a" * 64,
                                "pair_schedule_sha256": "b" * 64,
                                "pairs": [{}] * contract.F2P_UPDATES,
                            }
                            for i in range(contract.F2P_ATTEMPTS)
                        ]
                    )
                (root / f"{job}.json").write_text(json.dumps(payload))
            return report.aggregate_reports(root, HEAD, 0)

    def test_nonreproduction_is_explicit_without_erasing_other_observations(self):
        result = self.aggregate()
        self.assertEqual(result["status"], "OBSERVATIONS_COMPLETE_REVIEW_REQUIRED")
        self.assertEqual(result["jobs"]["F2P"]["reproduction"], "NOT_REPRODUCED")
        self.assertFalse(result["production_restart_authorized"])
        self.assertIn("bias_audit", result["jobs"]["F0"]["audits"])

    def test_missing_audit_or_replay_or_source_proof_is_incomplete(self):
        for kwargs in ({"audit_status": "ERROR"}, {"attempts": []}, {"source_ok": False}):
            self.assertEqual(self.aggregate(**kwargs)["status"], "INCOMPLETE")

    def test_observed_failure_requires_exact_capture(self):
        attempt = {
            "attempt": 1,
            "status": "FAILURE_CAPTURED",
            "capture_saved": False,
            "completed_updates": 0,
            "pairs": [],
            "expected_pairs": contract.F2P_UPDATES,
            "initial_model_sha256": "a" * 64,
            "pair_schedule_sha256": "b" * 64,
        }
        self.assertEqual(self.aggregate(attempts=[attempt])["status"], "INCOMPLETE")
        attempt["capture_saved"] = True
        self.assertEqual(self.aggregate(attempts=[attempt])["status"], "OBSERVATIONS_COMPLETE_REVIEW_REQUIRED")

    def test_f2p_attempts_require_same_initial_model_schedule_and_complete_workload(self):
        for changed in ("initial_model_sha256", "pair_schedule_sha256", "expected_pairs"):
            attempts = [
                {
                    "attempt": i + 1,
                    "status": "NOT_REPRODUCED",
                    "completed_updates": contract.F2P_UPDATES,
                    "pairs": [{}] * contract.F2P_UPDATES,
                    "expected_pairs": contract.F2P_UPDATES,
                    "initial_model_sha256": "a" * 64,
                    "pair_schedule_sha256": "b" * 64,
                }
                for i in range(contract.F2P_ATTEMPTS)
            ]
            attempts[1][changed] = 1 if changed == "expected_pairs" else "c" * 64
            result = self.aggregate(attempts=attempts)
            self.assertEqual(result["status"], "INCOMPLETE", result)
            self.assertTrue(any("F2P" in error for error in result["errors"]))


if __name__ == "__main__":
    unittest.main()
