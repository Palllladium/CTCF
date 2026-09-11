from __future__ import annotations

import errno
import hashlib
import json
import os
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import torch

from experiments.stage5 import runtime
from experiments.stage5.checkpoints import build_training_state, state_dict_sha256
from experiments.stage5.failures import ControllerFailureRecorder, classify_failure_chain, failure_exception_chain
from tools.analysis.run_artifacts import sha256_file
from tools.analysis.stage5.primitives import readable_json_bytes
from tools.analysis.stage5.recovery import acknowledge_controller_failure, locked_stopped_stage5_run


class FailureClassificationTest(unittest.TestCase):
    def test_wrapped_bootstrap_oom_preserves_resource_cause(self):
        try:
            try:
                raise RuntimeError("CUDA out of memory. Tried to allocate a tensor")
            except RuntimeError as error:
                raise RuntimeError("bootstrap construction failed") from error
        except RuntimeError as error:
            chain = failure_exception_chain(error)
        self.assertEqual(len(chain), 2)
        self.assertEqual(classify_failure_chain(chain), ("RESOURCE", True))

    def test_numerical_evidence_takes_precedence_over_resource_wrapper(self):
        error = torch.OutOfMemoryError("capture also ran out of memory")
        error.__cause__ = FloatingPointError("non-finite objective")
        self.assertEqual(classify_failure_chain(failure_exception_chain(error)), ("NUMERICAL", False))

    def test_error_handling_oom_does_not_hide_an_unknown_original_error(self):
        error = torch.OutOfMemoryError("OOM while recording the invalid shape")
        error.__context__ = ValueError("invalid shape")
        self.assertEqual(classify_failure_chain(failure_exception_chain(error)), ("INVARIANT_OR_UNKNOWN", False))

    def test_only_identified_environmental_io_is_eligible(self):
        for error, expected in (
            (OSError(errno.ENOSPC, "disk full"), ("IO", True)),
            (OSError("unclassified IO"), ("IO", False)),
            (FileNotFoundError(errno.ENOENT, "missing input"), ("IO", False)),
            (RuntimeError("illegal CUDA memory access"), ("INVARIANT_OR_UNKNOWN", False)),
            (AssertionError("shape mismatch"), ("INVARIANT_OR_UNKNOWN", False)),
        ):
            with self.subTest(error=error):
                self.assertEqual(classify_failure_chain(failure_exception_chain(error)), expected)


class FailureRecoveryTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.config = runtime.ControllerTrainingConfig()
        self.run_identity = runtime._RunIdentity(
            role="CONTROLLER",
            variant="F0",
            seed=0,
            git_head="a" * 40,
            protocol_sha256="b" * 64,
            data_contract_sha256="c" * 64,
            training_contract_sha256="d" * 64,
            metrics_schema="ctcf-stage5-controller-metrics-v1",
            label="controller",
            base_checkpoint_sha256="e" * 64,
            initial_controller_state_sha256="f" * 64,
            source_contract_sha256="1" * 64,
        )
        self.identity = {**asdict(self.run_identity), "config": asdict(self.config), "bootstrap_policy": "identity"}
        self.recorder = ControllerFailureRecorder(self.root, self.identity)
        model = torch.nn.Linear(2, 1)
        self.step = SimpleNamespace(controller=model, optimizer=torch.optim.AdamW(model.parameters()))

    def checkpoint(self, epoch=1, references=None, mutate=None):
        epochs = [
            {"epoch": index, "pairs": 1, "pair_schedule_sha256": "2" * 64, "metrics": {"loss": 0.25}}
            for index in range(1, epoch + 1)
        ]
        metrics = self.run_identity.metrics_payload(epochs)
        payload = build_training_state(
            role="CONTROLLER",
            variant_id="F0",
            seed=0,
            epoch_completed=epoch,
            fixed_epoch=100,
            git_head=self.run_identity.git_head,
            protocol_sha256=self.run_identity.protocol_sha256,
            data_contract_sha256=self.run_identity.data_contract_sha256,
            training_contract_sha256=self.run_identity.training_contract_sha256,
            model=self.step.controller,
            optimizer=self.step.optimizer,
            scaler=None,
            pair_schedule_sha256="2" * 64,
            metrics_sha256=hashlib.sha256(readable_json_bytes(metrics)).hexdigest(),
            base_checkpoint_sha256=self.run_identity.base_checkpoint_sha256,
            initial_controller_state_sha256=self.run_identity.initial_controller_state_sha256,
            source_contract_sha256=self.run_identity.source_contract_sha256,
        )
        runtime._attach_runtime_checkpoint_metadata(payload, config=self.config, metrics_payload=metrics)
        if references is not None:
            payload["recovery_acknowledgements"] = references
        if mutate is not None:
            mutate(payload)
        runtime._write_checkpoint_with_sidecar(self.root / "last.pth", payload)
        return payload

    def failure(self, error=None, epoch=2):
        return self.recorder.capture(
            error=error if error is not None else torch.OutOfMemoryError("GPU out of memory"),
            state={
                "pair": {"subject_a": "a", "subject_b": "b"},
                "epoch_one_based": epoch,
                "pair_index_one_based": 2,
                "successful_updates_before_pair": 5,
                "phase": "prepare_inputs",
                "inputs": None,
                "before_update": None,
            },
            step=self.step,
        )

    def acknowledge(self, failure, **overrides):
        kwargs = {
            "output_root": self.root,
            "failure_id": failure.parent.name,
            "failure_sha256": sha256_file(failure),
            "checkpoint_sha256": sha256_file(self.root / "last.pth") if (self.root / "last.pth").exists() else "0" * 64,
            "reason": "Conflicting external GPU process stopped; resources are available.",
            "expected_git_head": self.run_identity.git_head,
            "expected_protocol_sha256": self.run_identity.protocol_sha256,
        }
        kwargs.update(overrides)
        return acknowledge_controller_failure(**kwargs)

    def test_empty_tree_needs_no_acknowledgement(self):
        self.assertEqual(self.recorder.require_no_previous_failure(), [])

    def test_recovery_lock_cannot_be_redirected_to_an_unrelated_layout(self):
        with (
            self.assertRaisesRegex(ValueError, "protocol/protocol.json"),
            locked_stopped_stage5_run(protocol_path=self.root / "elsewhere.json"),
        ):
            self.fail("An unrelated path must not supply a run lock")

    @unittest.skipIf(os.name == "nt", "runner flock is a Linux server contract")
    def test_active_runner_lock_blocks_acknowledgement_and_is_preserved(self):
        protocol = self.root / "protocol" / "protocol.json"
        protocol.parent.mkdir()
        protocol.write_text("{}")
        lock = self.root / "stage5.lock"
        lock.write_text("existing runner lock")
        with (
            locked_stopped_stage5_run(protocol_path=protocol),
            self.assertRaisesRegex(RuntimeError, "still active"),
            locked_stopped_stage5_run(protocol_path=protocol),
        ):
            self.fail("An active runner must block acknowledgement")
        with locked_stopped_stage5_run(protocol_path=protocol):
            self.assertEqual(lock.read_text(), "existing runner lock")

    def test_acknowledgement_is_explicit_immutable_and_preserves_evidence(self):
        self.checkpoint()
        failure = self.failure()
        evidence = {path.name: sha256_file(path) for path in failure.parent.iterdir()}
        with self.assertRaisesRegex(RuntimeError, "no explicit acknowledgement"):
            self.recorder.require_no_previous_failure()
        ack = self.acknowledge(failure)
        self.assertEqual({path.name: sha256_file(path) for path in failure.parent.iterdir() if path != ack}, evidence)
        refs = self.recorder.require_no_previous_failure()
        self.assertEqual(refs, [{"failure_id": failure.parent.name, "acknowledgement_sha256": sha256_file(ack)}])
        with self.assertRaises(FileExistsError):
            self.acknowledge(failure, reason="A different explanation must not overwrite the original.")

    def test_numerical_unknown_and_untyped_io_cannot_be_acknowledged(self):
        self.checkpoint()
        for error in (FloatingPointError("nan"), RuntimeError("invalid shape"), OSError("disk issue")):
            with self.subTest(error=error):
                failure = self.failure(error)
                with self.assertRaisesRegex(RuntimeError, "remains blocked"):
                    self.acknowledge(failure)

    def test_disk_full_errno_can_be_acknowledged_without_erasing_failure(self):
        self.checkpoint()
        failure = self.failure(OSError(errno.ENOSPC, "disk full"))
        self.acknowledge(failure, reason="Disk quota extended; the failed snapshot is retained.")
        self.assertEqual(len(self.recorder.require_no_previous_failure()), 1)

    def test_first_epoch_requires_separately_approved_new_run(self):
        failure = self.failure(epoch=1)
        with self.assertRaisesRegex(RuntimeError, "separately approved new run"):
            self.acknowledge(failure)
        self.assertFalse((failure.parent / "acknowledgement.json").exists())

    def test_exact_failure_and_checkpoint_digests_are_required(self):
        self.checkpoint()
        failure = self.failure()
        for kwargs in ({"failure_sha256": "0" * 64}, {"checkpoint_sha256": "0" * 64}):
            with self.subTest(kwargs=kwargs), self.assertRaisesRegex(RuntimeError, "SHA-256"):
                self.acknowledge(failure, **kwargs)

    def test_head_and_protocol_cannot_change_for_technical_resume(self):
        self.checkpoint()
        failure = self.failure()
        for kwargs in ({"expected_git_head": "f" * 40}, {"expected_protocol_sha256": "e" * 64}):
            with self.subTest(kwargs=kwargs), self.assertRaisesRegex(RuntimeError, "HEAD/protocol"):
                self.acknowledge(failure, **kwargs)
        self.acknowledge(failure)
        recorder = ControllerFailureRecorder(self.root, {**self.identity, "git_head": "f" * 40})
        with self.assertRaisesRegex(RuntimeError, "identity changed"):
            recorder.require_no_previous_failure()

    def test_gpu_number_or_device_label_is_not_a_scientific_identity_change(self):
        self.checkpoint()
        failure = self.failure()
        self.acknowledge(failure)
        recorder = ControllerFailureRecorder(self.root, {**self.identity, "device_name": "same architecture, GPU 3"})
        self.assertEqual(len(recorder.require_no_previous_failure()), 1)

    def test_checkpoint_identity_and_nonfinite_state_are_rejected(self):
        for change in (lambda payload: payload.update(seed=1), self._nonfinite_model):
            with self.subTest(change=change):
                self.checkpoint(mutate=change)
                failure = self.failure()
                with self.assertRaisesRegex(RuntimeError, "identity differs|non-finite"):
                    self.acknowledge(failure)

    def _nonfinite_model(self, payload):
        payload["model_state"]["weight"] = torch.full_like(payload["model_state"]["weight"], float("nan"))
        payload["model_state_sha256"] = state_dict_sha256(payload["model_state"])

    def test_checkpoint_cannot_be_replaced_by_capture(self):
        self.checkpoint(mutate=lambda payload: payload.update(schema="ctcf-stage5-controller-failure-capture-v1"))
        failure = self.failure()
        with self.assertRaisesRegex(RuntimeError, "not a Stage5 training checkpoint"):
            self.acknowledge(failure)

    def test_recoverable_checkpoint_generation_is_verified_without_moving_files(self):
        self.checkpoint()
        checkpoint_bytes = (self.root / "last.pth").read_bytes()
        sidecar_bytes = (self.root / "last.pth.sha256.json").read_bytes()
        self.checkpoint(epoch=2)
        (self.root / ".last.pth.previous").write_bytes(checkpoint_bytes)
        (self.root / ".last.pth.previous.sha256.json").write_bytes(sidecar_bytes)
        (self.root / "last.pth").write_bytes(b"interrupted write")
        before = {path.name: sha256_file(path) for path in self.root.iterdir() if path.is_file()}
        failure = self.failure(epoch=3)
        self.acknowledge(failure, checkpoint_sha256=hashlib.sha256(checkpoint_bytes).hexdigest())
        self.assertEqual(len(self.recorder.require_no_previous_failure()), 1)
        after = {path.name: sha256_file(path) for path in self.root.iterdir() if path.is_file()}
        self.assertEqual(after, before)

    def test_external_resume_checkpoint_is_forbidden_after_failure(self):
        self.checkpoint()
        failure = self.failure()
        self.acknowledge(failure)
        with self.assertRaisesRegex(RuntimeError, "never an external checkpoint or capture"):
            self.recorder.require_no_previous_failure(resume=failure.parent / "capture.pth")

    def test_new_checkpoint_requires_embedded_recovery_reference(self):
        self.checkpoint()
        failure = self.failure()
        self.acknowledge(failure)
        refs = self.recorder.require_no_previous_failure()
        self.checkpoint(epoch=2)
        with self.assertRaisesRegex(RuntimeError, "neither the acknowledged epoch"):
            self.recorder.require_no_previous_failure()
        self.checkpoint(epoch=2, references=refs)
        self.assertEqual(self.recorder.require_no_previous_failure(), refs)
        self.checkpoint(epoch=100, references=refs)
        self.assertEqual(self.recorder.require_no_previous_failure(), refs)

    def test_older_checkpoint_or_changed_ack_cannot_reuse_permission(self):
        self.checkpoint()
        failure = self.failure()
        ack_path = self.acknowledge(failure)
        refs = self.recorder.require_no_previous_failure()
        self.checkpoint(epoch=2, references=refs)
        ack = json.loads(ack_path.read_text())
        ack["reason"] = "Changed explanation after the resumed checkpoint was committed."
        ack_path.write_text(json.dumps(ack), encoding="utf-8")
        with self.assertRaisesRegex(RuntimeError, "neither the acknowledged epoch"):
            self.recorder.require_no_previous_failure()

    def test_later_failure_needs_its_own_acknowledgement(self):
        self.checkpoint()
        first = self.failure()
        self.acknowledge(first)
        first_refs = self.recorder.require_no_previous_failure()
        self.checkpoint(epoch=2, references=first_refs)
        second = self.failure(epoch=3)
        with self.assertRaisesRegex(RuntimeError, "no explicit acknowledgement"):
            self.recorder.require_no_previous_failure()
        self.acknowledge(second)
        all_refs = self.recorder.require_no_previous_failure()
        self.assertEqual(len(all_refs), 2)
        self.checkpoint(epoch=3, references=all_refs)
        self.assertEqual(self.recorder.require_no_previous_failure(), all_refs)

    def test_path_traversal_and_linked_acknowledgement_are_rejected(self):
        self.checkpoint()
        failure = self.failure()
        with self.assertRaises((ValueError, RuntimeError)):
            self.acknowledge(failure, failure_id="../other")
        destination = self.root / "outside.json"
        destination.write_text("{}")
        try:
            (failure.parent / "acknowledgement.json").symlink_to(destination)
        except OSError:
            self.skipTest("symlink creation is unavailable")
        with self.assertRaisesRegex(RuntimeError, "link"):
            self.acknowledge(failure)
        self.assertEqual(destination.read_text(), "{}")


if __name__ == "__main__":
    unittest.main()
