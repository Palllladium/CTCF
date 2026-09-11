"""Investigate saved AMP failures and bounded F2P replay without production changes."""

from __future__ import annotations

import argparse
import gc
import json
import platform
import subprocess
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from experiments.stage5 import runtime
from experiments.stage5.checkpoints import capture_rng_state, state_dict_sha256
from experiments.stage5.config import (
    LegacyControllerTrainingConfig,
    build_stage5_controller,
)
from tools.analysis.diagnose_stage5_precision import (
    Sources,
    _try_capture,
    prepare_step,
    restore_snapshot,
    training_snapshot,
)
from tools.analysis.run_artifacts import atomic_write_json, sha256_file
from tools.analysis.run_stage5 import assert_clean_exact_git
from tools.analysis.stage5 import mechanism_contract as contract
from tools.analysis.stage5.mechanism_bias import run_bias_audit
from tools.analysis.stage5.mechanism_fields import run_field_audit
from tools.analysis.stage5.precision_contract import SOURCE_HEAD, error_status
from tools.analysis.stage5.precision_probes import backend_flags, tensor_stats
from tools.analysis.stage5.primitives import canonical_sha256


def utc_now():
    return datetime.now(timezone.utc).isoformat()


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


def run_audits(step, inputs, report, save):
    state = training_snapshot(step)
    for name, function in (("bias_audit", run_bias_audit), ("field_audit", run_field_audit)):
        restore_snapshot(step, state)
        try:

            def progress(value, section=name):
                report[section] = value
                save()

            report[name] = function(step, inputs, save=progress)
        except Exception as exc:
            report[name] = {
                "status": "ERROR",
                "kind": error_status(exc),
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
        finally:
            restore_snapshot(step, state)
            save()
            gc.collect()
            torch.cuda.empty_cache()


class ReplayObserver:
    """Observe the historical AMP path without changing its optimizer/scaler."""

    def __init__(self, step):
        self.step = step
        self.current = {}
        self.original_prepare = runtime._prepare_controller_pair
        self.original_update = runtime._strict_scaler_step

    def prepare(self, *values, **keywords):
        inputs = self.original_prepare(*values, **keywords)
        self.current.update(inputs=inputs, pre_loss_rng=capture_rng_state(), stage="forward_or_backward")
        return inputs

    def capture_state(self):
        state = training_snapshot(self.step)
        state["rng"] = self.current["pre_loss_rng"]
        self.current["failure_state"] = state

    def update(self, scaler, optimizer, *, phase):
        gradients = {name: parameter.grad for name, parameter in self.step.controller.named_parameters()}
        bad = [name for name, value in gradients.items() if value is not None and not bool(torch.isfinite(value).all())]
        missing = [name for name, value in gradients.items() if value is None]
        self.current["gradient_observation"] = {
            "scale_before": scaler.get_scale(),
            "bad_parameters": bad,
            "missing_parameters": missing,
            "stem_bias_scaled": tensor_stats(gradients["stem.0.bias"])
            if gradients["stem.0.bias"] is not None
            else None,
        }
        if bad:
            self.capture_state()
        self.current["stage"] = "optimizer"
        # Preserve the historical AMP behavior, including its fail-fast skip.
        return self.original_update(scaler, optimizer, phase=phase)

    def __enter__(self):
        self.prepare_patch = patch.object(runtime, "_prepare_controller_pair", self.prepare)
        self.update_patch = patch.object(runtime, "_strict_scaler_step", self.update)
        self.prepare_patch.start()
        self.update_patch.start()
        return self

    def __exit__(self, *_args):
        self.update_patch.stop()
        self.prepare_patch.stop()


def replay_f2p(args, sources, report, save):
    """Replay the historical AMP step, observing gradients before its scaler step."""
    attempts = report["f2p_replay"] = []
    for attempt in range(1, contract.F2P_ATTEMPTS + 1):
        step = prepare_step(sources, variant="F2P", seed=0)
        pairs = runtime.controller_epoch_pairs(sources.subjects, seed=0, epoch=0)
        record = {
            "attempt": attempt,
            "status": "RUNNING",
            "expected_pairs": len(pairs),
            "pair_schedule_sha256": canonical_sha256(pairs),
            "completed_updates": 0,
            "pairs": [],
            "initial_model_sha256": state_dict_sha256(step.controller.state_dict()),
        }
        attempts.append(record)
        observer = ReplayObserver(step)
        current = observer.current
        with observer:
            for index, pair in enumerate(pairs):
                current.clear()
                context = {"variant": "F2P", "seed": 0, "epoch": 1, "pair_index": index + 1, "pair": pair}
                record["last_pair"] = context
                print(
                    f"[MECHANISM F2P] attempt={attempt}/{contract.F2P_ATTEMPTS} pair={index + 1}/{len(pairs)}",
                    flush=True,
                )
                save()
                try:
                    metrics = runtime._legacy_controller_pair_step(step, pair, 0)
                    if not current.get("gradient_observation"):
                        raise RuntimeError("Legacy AMP step returned without the required gradient observation")
                except Exception as exc:
                    if error_status(exc) == "MATH_ERROR" and current.get("stage") == "forward_or_backward":
                        observer.capture_state()
                    reproduced = bool(current.get("gradient_observation", {}).get("bad_parameters"))
                    record.update(
                        status="FAILURE_CAPTURED" if "failure_state" in current else "OTHER_FAILURE",
                        failure={
                            "kind": error_status(exc),
                            "error": str(exc),
                            "stage": current.get("stage", "prepare"),
                        },
                        historical_gradient_failure_reproduced=reproduced,
                    )
                    record["gradient_observation"] = current.get("gradient_observation")
                    break
                record["pairs"].append({**context, "metrics": metrics, **current["gradient_observation"]})
                record["completed_updates"] += 1
            else:
                record["status"] = "NOT_REPRODUCED"
        save()
        if "failure_state" in current:
            restore_snapshot(step, current["failure_state"])
            _try_capture(
                args,
                report,
                step,
                current["inputs"],
                current["failure_state"],
                record["last_pair"],
                name=f"attempt_{attempt}_failure",
            )
            run_audits(step, current["inputs"], report, save)
            record["capture_saved"] = bool(report.get("heavy_captures"))
            save()
            return
        if record["status"] != "NOT_REPRODUCED":
            return
        current.clear()
        del observer, step
        gc.collect()
        torch.cuda.empty_cache()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "repo-root",
        "precision-root",
        "source-root",
        "checkpoint-root",
        "data-contract",
        "image-root",
        "output",
        "capture-root",
    ):
        parser.add_argument(f"--{name}", required=True, type=Path)
    parser.add_argument("--expected-git-head", required=True)
    parser.add_argument("--job", choices=contract.JOBS, required=True)
    args = parser.parse_args(argv)
    assert_clean_exact_git(args.repo_root, args.expected_git_head)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8"):
        pass
    report = {
        "schema": contract.SCHEMA,
        "job": args.job,
        "status": "RUNNING",
        "started_at_utc": utc_now(),
        "diagnostic_git_head": args.expected_git_head,
        "workload_contract": contract.workload_contract(),
        "source_git_head": SOURCE_HEAD,
        "production_checkpoint_written": False,
        "labels_accessed": False,
        "production_training_validated": False,
        "mechanism_sources_unchanged": None,
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
            "platform": platform.platform(),
            "startup_backend_flags": backend_flags(),
        },
    }

    def save():
        atomic_write_json(args.output, report)

    captures = sources = None
    code = 0
    try:
        runtime._require_cuda(torch.device("cuda:0"), "mechanism diagnostic")
        runtime._seed_everything(0)
        report["environment"]["backend_flags"] = backend_flags()
        report["environment"]["gpu"] = torch.cuda.get_device_name(0)
        report["environment"]["execution_determinism"] = runtime._execution_determinism_contract()
        freeze = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True, timeout=60)
        report["environment"]["pip_freeze"] = freeze.stdout.splitlines()
        report["environment"]["pip_freeze_exit_code"] = freeze.returncode
        captures = CaptureSources(args.repo_root, args.precision_root, report)
        if args.job == "F2P":
            sources = Sources(args, report)
            replay_f2p(args, sources, report, save)
        else:
            step, inputs = captures.load_failure(args.job, device=torch.device("cuda:0"))
            run_audits(step, inputs, report, save)
        report["status"] = "DIAGNOSTIC_COMPLETE"
    except Exception as exc:
        report.update(
            status="ERROR", failure={"kind": error_status(exc), "error": str(exc)}, traceback=traceback.format_exc()
        )
        code = 1
    finally:
        for source in (captures, sources):
            if source is not None:
                try:
                    source.verify_unchanged()
                except Exception as exc:
                    report.update(status="ERROR", mechanism_sources_unchanged=False, source_integrity_error=str(exc))
                    code = 1
        report["completed_at_utc"] = utc_now()
        save()
    print(f"[MECHANISM DIAG RESULT] {args.job} {report['status']} {args.output}", flush=True)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
