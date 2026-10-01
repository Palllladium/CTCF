"""Verified historical failure captures used by the retained precision comparison.

Run bindings deliberately remain frozen: these are regression inputs, not
production checkpoints. No replay scheduler or training-data reader lives here.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import torch

from experiments.stage5 import runtime
from experiments.stage5.checkpoints import capture_rng_state, restore_rng_state, state_dict_sha256
from experiments.stage5.config import LegacyControllerTrainingConfig, build_stage5_controller
from experiments.stage5.failures import cpu_snapshot
from tools.analysis.run_artifacts import sha256_file
from tools.analysis.stage5 import mechanism_contract as contract
from tools.analysis.stage5.precision_contract import SOURCE_HEAD


def training_snapshot(step):
    return {
        "model": cpu_snapshot(step.controller.state_dict()),
        "optimizer": cpu_snapshot(step.optimizer.state_dict()),
        "scaler": copy.deepcopy(step.scaler.state_dict()),
        "rng": capture_rng_state(),
    }


def restore_snapshot(step, state):
    step.controller.load_state_dict(state["model"], strict=True)
    step.optimizer.load_state_dict(copy.deepcopy(state["optimizer"]))
    step.scaler.load_state_dict(copy.deepcopy(state["scaler"]))
    step.optimizer.zero_grad(set_to_none=True)
    restore_rng_state(state["rng"])


class CaptureSources:
    """Validate the pinned compact package before trusting any serialized state."""

    def __init__(self, repo, compact, report):
        self.repo, self.compact, self.report = repo.resolve(), compact.resolve(), report
        self.hashes = report.setdefault("mechanism_source_hashes", {})
        sums = self.compact / "SHA256SUMS"
        if self.remember(sums) != contract.PRECISION_SUMS_SHA256:
            raise RuntimeError("Unexpected precision package SHA256SUMS")
        for line in sums.read_text(encoding="utf-8").splitlines():
            expected, relative = line.split("  ", 1)
            path = self.contained(self.compact, relative)
            if self.remember(path) != expected:
                raise RuntimeError(f"Precision package member differs: {relative}")
        if (self.compact / "git_head.txt").read_text().strip() != contract.PRECISION_HEAD:
            raise RuntimeError("Unexpected precision diagnostic revision")
        self.workers = {job: json.loads((self.compact / f"{job}.json").read_text()) for job in contract.JOBS}
        for job, worker in self.workers.items():
            if (
                worker.get("job") != job
                or worker.get("diagnostic_git_head") != contract.PRECISION_HEAD
                or worker.get("status") != "DIAGNOSTIC_COMPLETE"
                or worker.get("source_bytes_unchanged") is not True
            ):
                raise RuntimeError(f"Invalid reviewed worker: {job}")
        report["precision_package_verified"] = True

    @staticmethod
    def contained(root, relative):
        candidate = root / relative
        resolved = candidate.resolve()
        if not resolved.is_relative_to(root.resolve()) or candidate.is_symlink():
            raise RuntimeError("Source member escapes its root or is a symlink")
        return resolved

    def remember(self, path):
        if not path.is_file() or path.is_symlink():
            raise RuntimeError(f"Expected regular source file: {path}")
        digest = sha256_file(path)
        if self.hashes.setdefault(str(path), digest) != digest:
            raise RuntimeError(f"Source changed while being read: {path}")
        return digest

    def load_failure(self, job, *, device, capture_root: Path | None = None):
        previous = self.workers[job]
        captures = previous.get("heavy_captures", [])
        if previous.get("failure_reproduced") is not True or len(captures) != 1:
            raise RuntimeError("Expected exactly one reviewed failure capture")
        capture = captures[0]
        expected_root = self.repo / "results/stage5_heavy" / contract.PRECISION_RUN / job / "historical_failure"
        actual_root = expected_root if capture_root is None else capture_root.resolve() / job / "historical_failure"
        members = {}
        for record in capture["records"]:
            path = self.contained(self.repo, record["path"])
            if path.parent != expected_root.resolve() or path.name in members:
                raise RuntimeError("Unexpected or duplicate captured tensor path")
            path = self.contained(actual_root, path.name)
            if path.stat().st_size != record["bytes"] or self.remember(path) != record["sha256"]:
                raise RuntimeError(f"Saved failure bytes differ: {path}")
            members[path.name] = path
        expected = {"state.pth", "psi_ab.pth", "psi_ba.pth"}
        expected.update(f"tensors_{direction}_{index}.pth" for direction in ("ab", "ba") for index in range(5))
        if set(members) != expected or capture.get("exact_prepared_inputs_saved") is not True:
            raise RuntimeError("Failure capture is incomplete")
        manifest_path = self.contained(actual_root, "manifest.json")
        self.remember(manifest_path)
        if json.loads(manifest_path.read_text()) != capture:
            raise RuntimeError("Failure manifest differs from reviewed compact record")
        state = torch.load(members["state.pth"], map_location="cpu", weights_only=False)
        if (
            state.get("schema") != "ctcf-stage5-diagnostic-capture-v1"
            or state.get("diagnostic_git_head") != contract.PRECISION_HEAD
            or state.get("source_git_head") != SOURCE_HEAD
            or state.get("context") != capture["context"]
            or state.get("production_resume_forbidden") is not True
        ):
            raise RuntimeError("Saved state has unexpected provenance")
        training = state["training_state"]
        if state_dict_sha256(training["model"]) != previous["comparison"]["initial_model_sha256"]:
            raise RuntimeError("Saved model differs from the reviewed same-state probe")
        config = LegacyControllerTrainingConfig()
        controller = build_stage5_controller(config).to(device).train()
        optimizer = torch.optim.AdamW(
            controller.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
        )
        step = SimpleNamespace(
            controller=controller,
            optimizer=optimizer,
            scaler=runtime._stage5_grad_scaler(config),
            device=device,
            variant=job,
            config=config,
        )
        restore_snapshot(step, training)

        def tensor(name):
            value = torch.load(members[f"{name}.pth"], map_location="cpu", weights_only=True)["tensor"]
            if value.dtype != torch.float32 or not bool(torch.isfinite(value).all()):
                raise RuntimeError(f"Unexpected captured tensor: {name}")
            return value.to(device)

        inputs = runtime._ControllerPairInputs(
            psi_ab=tensor("psi_ab"),
            psi_ba=tensor("psi_ba"),
            tensors_ab=tuple(tensor(f"tensors_ab_{i}") for i in range(5)),
            tensors_ba=tuple(tensor(f"tensors_ba_{i}") for i in range(5)),
            bootstrap_residual=state["bootstrap_residual"],
        )
        self.report["saved_failure"] = {
            "context": capture["context"],
            "model_sha256": state_dict_sha256(training["model"]),
            "exact_input_bytes_verified": True,
        }
        return step, inputs

    def verify_unchanged(self):
        for name, expected in self.hashes.items():
            if sha256_file(Path(name)) != expected:
                raise RuntimeError(f"Mechanism source changed: {name}")
        self.report["mechanism_sources_unchanged"] = True
