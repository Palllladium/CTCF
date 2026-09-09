"""Diagnose the reviewed post-NCC-fix failures without modifying training runs.

The historical FP16 replay, same-state numerical probes, fresh FP32 trajectory,
and timing measurements are separate experiments. No diagnostic state is a
production checkpoint and no development image or label is evaluated.
"""

from __future__ import annotations

import argparse
import copy
import csv
import gc
import platform
import shutil
import time
import traceback
from dataclasses import fields, replace
from datetime import datetime, timezone
from pathlib import Path

import torch

from datasets.OASIS100 import Stage5OasisImageStore
from experiments.stage5 import runtime
from experiments.stage5.checkpoints import (
    atomic_torch_save,
    capture_rng_state,
    load_training_state,
    restore_rng_state,
)
from experiments.stage5.config import ControllerTrainingConfig, U0TrainingConfig, build_stage5_controller
from tools.analysis.diagnose_stage5_amp import require_readonly_checkpoint
from tools.analysis.run_artifacts import atomic_write_json, sha256_file
from tools.analysis.run_stage5 import assert_clean_exact_git
from tools.analysis.stage5 import precision_contract as contract
from tools.analysis.stage5.artifacts import load_canonical_json
from tools.analysis.stage5.contracts import canonical_sha256, validate_protocol_contract
from tools.analysis.stage5.precision_contract import JOBS, SOURCE_HEAD, SOURCE_RUN, error_status
from tools.analysis.stage5.precision_probes import compare_pair, precision_context

SOURCE_PROTOCOL_SHA = "77956f111bc8410b7817b732b7fd2afdfe200ab4060bb9d8ec4fd64334ec6de8"
SOURCE_MANIFEST = "A_20260907T203312Z_2417900"
SOURCE_MANIFEST_SHA = "ac92165be46a8c17f3460e7631cf0b3794fef28e1e1c46ae6e494fe3479f76ec"


class NonfiniteGradientError(FloatingPointError):
    """The reviewed failure class: backward overflow before an optimizer update."""


def _cpu_copy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _cpu_copy(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_cpu_copy(item) for item in value)
    if isinstance(value, list):
        return [_cpu_copy(item) for item in value]
    return copy.deepcopy(value)


def training_snapshot(step):
    return {
        "model": _cpu_copy(step.controller.state_dict()),
        "optimizer": _cpu_copy(step.optimizer.state_dict()),
        "scaler": copy.deepcopy(step.scaler.state_dict()),
        "rng": capture_rng_state(),
    }


def restore_snapshot(step, state):
    step.controller.load_state_dict(state["model"], strict=True)
    step.optimizer.load_state_dict(copy.deepcopy(state["optimizer"]))
    step.scaler.load_state_dict(copy.deepcopy(state["scaler"]))
    step.optimizer.zero_grad(set_to_none=True)
    restore_rng_state(state["rng"])


def _failure(exc, *, stage):
    status = error_status(exc)
    kind = "NUMERICAL" if status == "MATH_ERROR" else status
    return {
        "kind": kind,
        "stage": stage,
        "mechanism": "NONFINITE_GRADIENT" if isinstance(exc, NonfiniteGradientError) else "OTHER",
        "error": f"{type(exc).__name__}: {exc}",
    }


def advance_pair(step, inputs, *, fp32):
    """One diagnostic update; reject bad gradients before any optimizer mutation."""
    step.optimizer.zero_grad(set_to_none=True)
    with precision_context(fp32):
        loss, logs = runtime._controller_pair_loss(step, inputs, diagnostic_fp32=fp32)
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError("non-finite controller objective")
        if fp32:
            loss.backward()
        else:
            step.scaler.scale(loss).backward()
        gradients = [p.grad for p in step.controller.parameters() if p.requires_grad]
        if any(value is None for value in gradients):
            raise RuntimeError("Missing controller parameter gradient")
        if any(not bool(torch.isfinite(value).all()) for value in gradients):
            raise NonfiniteGradientError("non-finite controller gradients before optimizer update")
        if fp32:
            step.optimizer.step()
        else:
            runtime._strict_scaler_step(step.scaler, step.optimizer, phase=f"precision replay {step.variant}")
    if any(not bool(torch.isfinite(value).all()) for value in step.controller.state_dict().values()):
        raise FloatingPointError("optimizer produced non-finite controller state")
    for state in step.optimizer.state.values():
        if any(isinstance(value, torch.Tensor) and not bool(torch.isfinite(value).all()) for value in state.values()):
            raise FloatingPointError("optimizer produced non-finite internal state")
    return logs


class Sources:
    """Bindings to the immutable compact package and its attested heavy states."""

    def __init__(self, args, report):
        self.args = args
        self.report = report
        self.hashes = report.setdefault("source_hashes", {})
        manifest = args.source_root / "manifests" / f"{SOURCE_MANIFEST}.json"
        if self.remember(manifest) != SOURCE_MANIFEST_SHA:
            raise RuntimeError("Expected the reviewed final FAILED manifest")
        payload = load_canonical_json(manifest)
        if payload.get("status") != "FAILED" or payload.get("git_head") != SOURCE_HEAD:
            raise RuntimeError("Unexpected source attempt status or revision")
        index = args.source_root / payload["outputs_file"]["relative_path"]
        if self.remember(index) != payload["outputs_file"]["sha256"]:
            raise RuntimeError("Source output index digest mismatch")
        with index.open(encoding="utf-8", newline="") as stream:
            for row in csv.DictReader(stream, delimiter="\t"):
                path = self.source_file(row["relative_path"])
                if self.remember(path) != row["sha256"] or path.stat().st_size != int(row["bytes"]):
                    raise RuntimeError(f"Source artifact mismatch: {path}")
        self.protocol = load_canonical_json(args.source_root / "protocol" / "protocol.json")
        validate_protocol_contract(self.protocol)
        if canonical_sha256(self.protocol) != SOURCE_PROTOCOL_SHA or self.protocol["git_head"] != SOURCE_HEAD:
            raise RuntimeError("Diagnostic requires the reviewed post-fix source protocol")
        for name in ("u0", "controller"):
            path = args.source_root / "protocol" / f"{name}_training_contract.json"
            if canonical_sha256(load_canonical_json(path)) != self.protocol[f"{name}_training_contract_sha256"]:
                raise RuntimeError(f"Source {name} contract mismatch")
        self.store = Stage5OasisImageStore(args.data_contract, args.image_root)
        if self.store.runtime.contract_sha256 != self.protocol["data_contract_sha256"]:
            raise RuntimeError("Image-only cache belongs to a different data contract")
        self.remember(args.data_contract)
        self.subjects = runtime._training_subjects(self.store)
        if len(self.subjects) != contract.TRAINING_SUBJECTS:
            raise RuntimeError(f"Expected exactly the frozen {contract.TRAINING_SUBJECTS} training subjects")
        report.update(
            protocol_sha256=SOURCE_PROTOCOL_SHA,
            data_contract_sha256=self.store.runtime.contract_sha256,
            source_manifest_verified=True,
        )

    def source_file(self, relative):
        path = self.args.source_root / relative
        if not path.resolve().is_relative_to(self.args.source_root.resolve()):
            raise RuntimeError("Source member escapes its root")
        return path

    def remember(self, path):
        if not path.is_file() or path.is_symlink():
            raise RuntimeError(f"Expected regular source artifact: {path}")
        digest = sha256_file(path)
        previous = self.hashes.setdefault(str(path), digest)
        if previous != digest:
            raise RuntimeError(f"Source changed while being read: {path}")
        return digest

    def checkpoint(self, relative):
        path = self.args.checkpoint_root / relative
        digest = require_readonly_checkpoint(path)
        attestation = self.source_file(f"training_attestations/{relative}.sha256.json")
        frozen = load_canonical_json(attestation)
        if frozen.get("sha256") != digest or frozen.get("bytes") != path.stat().st_size:
            raise RuntimeError(f"Heavy checkpoint differs from source attestation: {path}")
        self.remember(path)
        self.remember(runtime._checkpoint_sidecar_path(path))
        return path, digest

    def verify_unchanged(self):
        for name, digest in self.hashes.items():
            if sha256_file(Path(name)) != digest:
                raise RuntimeError(f"Source artifact changed during diagnosis: {name}")
        self.report["source_bytes_unchanged"] = True


def prepare_step(sources, *, variant, seed, resume_f0=False):
    runtime._seed_everything(seed)
    config = ControllerTrainingConfig()
    device = torch.device("cuda:0")
    base, base_sha = sources.checkpoint(f"u0/seed_{seed}/last.pth")
    initial, _ = sources.checkpoint(f"controller_initial/seed_{seed}/initial.pth")
    controller = build_stage5_controller(config).to(device)
    initial_sha = runtime._load_initial_controller(initial, controller, seed=seed, config=config)
    optimizer = torch.optim.AdamW(controller.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scaler = runtime._stage5_grad_scaler(config)
    protocol = sources.protocol
    base_runner = runtime.load_frozen_u0(
        base,
        seed=seed,
        device=device,
        protocol_sha256=SOURCE_PROTOCOL_SHA,
        data_contract_sha256=sources.store.runtime.contract_sha256,
        training_contract_sha256=protocol["u0_training_contract_sha256"],
        config=U0TrainingConfig(),
        expected_git_head=SOURCE_HEAD,
    )
    controller.train()
    step = runtime._ControllerStep(
        store=sources.store,
        base_runner=base_runner,
        controller=controller,
        optimizer=optimizer,
        scaler=scaler,
        device=device,
        variant=variant,
        bootstrap_policy=protocol["bootstrap"]["policy"],
        config=config,
    )
    if resume_f0:
        if variant != "F0" or seed != 0:
            raise RuntimeError("Only the reviewed F0 seed0 epoch1 state may be resumed")
        path, _ = sources.checkpoint("controllers/seed_0/F0/last.pth")
        state = load_training_state(
            path,
            model=controller,
            optimizer=optimizer,
            scaler=scaler,
            expected_role="CONTROLLER",
            expected_variant="F0",
            expected_seed=0,
            expected_protocol_sha256=SOURCE_PROTOCOL_SHA,
            expected_data_contract_sha256=sources.store.runtime.contract_sha256,
            expected_training_contract_sha256=protocol["controller_training_contract_sha256"],
            restore_rng=True,
        )
        source_contract = canonical_sha256(
            {
                "schema": "ctcf-stage5-on-the-fly-training-source-v1",
                "u0_checkpoint_sha256": base_sha,
                "data_contract_sha256": sources.store.runtime.contract_sha256,
                "protocol_sha256": SOURCE_PROTOCOL_SHA,
                "pair_domain": runtime.CONTROLLER_PAIR_DOMAIN,
                "bootstrap_policy": protocol["bootstrap"]["policy"],
                "source_policy": "shared_frozen_bootstrap_construction_on_the_fly",
            }
        )
        metrics = runtime._validate_runtime_checkpoint_metadata(
            state,
            role="CONTROLLER",
            variant="F0",
            seed=0,
            config=config,
            expected_git_head=SOURCE_HEAD,
            expected_base_checkpoint_sha256=base_sha,
            expected_initial_controller_state_sha256=initial_sha,
            expected_source_contract_sha256=source_contract,
        )
        metrics_path = sources.source_file("training_attestations/controllers/seed_0/F0/metrics.json")
        schedule = runtime.controller_epoch_pairs(sources.subjects, seed=0, epoch=0)
        if (
            state["epoch_completed"] != 1
            or metrics != load_canonical_json(metrics_path)
            or state["metrics_sha256"] != sha256_file(metrics_path)
            or state["pair_schedule_sha256"] != canonical_sha256(schedule)
        ):
            raise RuntimeError("F0 epoch1 checkpoint does not match reviewed metrics and schedule")
    return step


def capture_failure(args, report, step, inputs, state, context, *, name):
    """Keep one exact prepared failure on the server; compact ZIP gets only hashes."""
    destination = args.capture_root / args.job / name
    destination.mkdir(parents=True, exist_ok=False)
    tensors = {}
    if inputs is not None:
        for field in fields(inputs):
            value = getattr(inputs, field.name)
            if isinstance(value, torch.Tensor):
                tensors[field.name] = value
            elif isinstance(value, tuple):
                tensors.update({f"{field.name}_{index}": item for index, item in enumerate(value)})
    expected_bytes = sum(value.numel() * value.element_size() for value in tensors.values())
    if shutil.disk_usage(destination).free < expected_bytes + 512 * 1024**2:
        raise OSError("Insufficient disk for the exact failure capture; source artifacts are untouched")
    records = []
    state_path = destination / "state.pth"
    payload = {
        "schema": "ctcf-stage5-diagnostic-capture-v1",
        "diagnostic_only": True,
        "production_resume_forbidden": True,
        "context": context,
        "training_state": state,
        "bootstrap_residual": None if inputs is None else inputs.bootstrap_residual,
        "source_git_head": SOURCE_HEAD,
        "diagnostic_git_head": report["diagnostic_git_head"],
    }
    digest = atomic_torch_save(state_path, payload)
    records.append({"path": str(state_path), "bytes": state_path.stat().st_size, "sha256": digest})
    # Copy/save one tensor at a time instead of duplicating the full feature set in RAM.
    for label, tensor in tensors.items():
        path = destination / f"{label}.pth"
        digest = atomic_torch_save(path, {"tensor": tensor.detach().cpu()})
        records.append({"path": str(path), "bytes": path.stat().st_size, "sha256": digest})
    manifest = {"context": context, "records": records, "exact_prepared_inputs_saved": inputs is not None}
    atomic_write_json(destination / "manifest.json", manifest)
    report.setdefault("heavy_captures", []).append(manifest)


def _try_capture(args, report, step, inputs, state, context, *, name):
    try:
        capture_failure(args, report, step, inputs, state, context, name=name)
    except (OSError, RuntimeError) as exc:
        report.setdefault("capture_errors", []).append({"name": name, "error": str(exc)})


def _sync(device):
    torch.cuda.synchronize(device)


def replay_failure(args, report, sources, save):
    variant = args.job
    epoch = 1 if variant == "F0" else 0
    step = prepare_step(sources, variant=variant, seed=0, resume_f0=variant == "F0")
    pairs = runtime.controller_epoch_pairs(sources.subjects, seed=0, epoch=epoch)
    replay = report["replay"] = {
        "status": "RUNNING",
        "epoch": epoch + 1,
        "pair_schedule_sha256": canonical_sha256(pairs),
        "expected_pairs": len(pairs),
        "completed_updates": 0,
        "metrics": [],
    }
    report["failure_reproduced"] = False
    last_inputs = None
    for index, pair in enumerate(pairs):
        context = {"variant": variant, "seed": 0, "epoch": epoch + 1, "pair_index": index + 1, "pair": pair}
        replay["current"] = context
        save()
        print(f"[PRECISION REPLAY] {variant} epoch={epoch + 1} pair={index + 1}/{len(pairs)}", flush=True)
        inputs = runtime._prepare_controller_pair(step, pair, epoch)
        state = training_snapshot(step)
        failure = None
        try:
            logs = advance_pair(step, inputs, fp32=False)
        except (RuntimeError, FloatingPointError) as exc:
            failure = _failure(exc, stage="historical_fp16_step")
        if failure is not None:
            report["failure_reproduced"] = failure["mechanism"] == "NONFINITE_GRADIENT"
            replay.update(
                status="FAILURE_REPRODUCED" if report["failure_reproduced"] else "DIFFERENT_FAILURE",
                failure=failure,
            )
            restore_snapshot(step, state)
            _try_capture(args, report, step, inputs, state, context, name="historical_failure")
            save()
            last_inputs = inputs
            break
        replay["completed_updates"] += 1
        replay["metrics"].append({**context, "values": logs})
        if index + 1 == len(pairs):
            # An unreproduced source failure must still get FP32/FP16 comparisons.
            last_inputs = inputs
        else:
            del inputs
    else:
        replay["status"] = "NOT_REPRODUCED"
    replay["last_pair"] = replay.pop("current")
    save()
    if last_inputs is None:
        raise RuntimeError("Replay produced no prepared inputs to compare")

    def save_comparison(value):
        report["comparison"] = value
        save()

    report["comparison"] = compare_pair(step, last_inputs, save=save_comparison)
    save()
    del last_inputs, step
    gc.collect()
    torch.cuda.empty_cache()


def run_trajectory(args, report, sources, save):
    step = prepare_step(sources, variant=args.job, seed=0)
    record = report["fp32_trajectory"] = {
        "status": "RUNNING",
        "initialization": "COMMON_INITIAL_CONTROLLER_FROM_ZERO",
        "epochs": contract.FP32_EPOCHS,
        "expected_updates": contract.fp32_updates(),
        "completed_updates": 0,
        "metrics": [],
        "full_step_seconds": [],
        "peak_allocated_bytes": 0,
    }
    failed = False
    for epoch in range(contract.FP32_EPOCHS):
        pairs = runtime.controller_epoch_pairs(sources.subjects, seed=0, epoch=epoch)
        for index, pair in enumerate(pairs):
            context = {"variant": args.job, "seed": 0, "epoch": epoch + 1, "pair_index": index + 1, "pair": pair}
            record["current"] = context
            save()
            state = training_snapshot(step)
            inputs = None
            _sync(step.device)
            torch.cuda.reset_peak_memory_stats(step.device)
            started = time.perf_counter()
            failure = None
            stage = "input_preparation"
            try:
                inputs = runtime._prepare_controller_pair(step, pair, epoch)
                state["rng"] = capture_rng_state()
                stage = "fp32_update"
                logs = advance_pair(step, inputs, fp32=True)
                _sync(step.device)
            except (RuntimeError, FloatingPointError) as exc:
                failure = _failure(exc, stage=stage)
            if failure is not None:
                record.update(status="FAILED", failure=failure)
                restore_snapshot(step, state)
                _try_capture(args, report, step, inputs, state, context, name="fp32_failure")
                save()
                if inputs is not None and failure["kind"] == "NUMERICAL":
                    record["failure_comparison"] = compare_pair(step, inputs)
                failed = True
                break
            record["full_step_seconds"].append(time.perf_counter() - started)
            record["peak_allocated_bytes"] = max(
                record["peak_allocated_bytes"], torch.cuda.max_memory_allocated(step.device)
            )
            record["metrics"].append({**context, "values": logs})
            record["completed_updates"] += 1
            del inputs
            print(f"[PRECISION FP32] {args.job} epoch={epoch + 1} pair={index + 1}/{len(pairs)}", flush=True)
        if failed:
            break
    if not failed:
        record["status"] = "COMPLETE"
    record["last_pair"] = record.pop("current")
    save()
    del step
    gc.collect()
    torch.cuda.empty_cache()


def benchmark(args, report, sources, save):
    step = prepare_step(sources, variant=args.job, seed=0)
    initial = training_snapshot(step)
    pair = runtime.controller_epoch_pairs(sources.subjects, seed=0, epoch=0)[0]
    record = report["benchmark"] = {
        "status": "RUNNING",
        "paired_inputs": True,
        "timing_kind": "SHARED_PREPARATION_PLUS_MEASURED_MODE_STEP",
        "warmup_iterations": 1,
        "repeats": contract.BENCHMARK_REPEATS,
        "memory_kind": "PHASE_PEAKS_AND_INCREMENT_ABOVE_LIVE_MODE_INPUTS",
        "preparation_seconds": [],
        "preparation_peak_allocated_bytes": [],
        "fp16": {
            "completed_steps": 0,
            "full_step_seconds": [],
            "compute_seconds": [],
            "peak_allocated_bytes": 0,
            "mode_peak_allocated_bytes": [],
            "mode_start_allocated_bytes": [],
            "mode_incremental_peak_bytes": [],
            "gradient_status": [],
        },
        "fp32_strict": {
            "completed_steps": 0,
            "full_step_seconds": [],
            "compute_seconds": [],
            "peak_allocated_bytes": 0,
            "mode_peak_allocated_bytes": [],
            "mode_start_allocated_bytes": [],
            "mode_incremental_peak_bytes": [],
            "gradient_status": [],
        },
    }
    for iteration in range(contract.BENCHMARK_REPEATS + 1):
        restore_snapshot(step, initial)
        _sync(step.device)
        torch.cuda.reset_peak_memory_stats(step.device)
        started = time.perf_counter()
        inputs = runtime._prepare_controller_pair(step, pair, epoch=0)
        _sync(step.device)
        preparation_seconds = time.perf_counter() - started
        preparation_peak = torch.cuda.max_memory_allocated(step.device)
        if iteration:
            record["preparation_seconds"].append(preparation_seconds)
            record["preparation_peak_allocated_bytes"].append(preparation_peak)
        prepared_rng = capture_rng_state()
        # Alternate order to reduce systematic warm-cache / contention bias.
        modes = (False, True) if iteration % 2 == 0 else (True, False)
        for fp32 in modes:
            mode = "fp32_strict" if fp32 else "fp16"
            restore_snapshot(step, initial)
            restore_rng_state(prepared_rng)
            _sync(step.device)
            torch.cuda.reset_peak_memory_stats(step.device)
            mode_start_bytes = torch.cuda.memory_allocated(step.device)
            started = time.perf_counter()
            status = "FINITE"
            try:
                advance_pair(step, inputs, fp32=fp32)
            except FloatingPointError:
                # This is timing only; a rejected update is never accepted as training.
                status = "NONFINITE_GRADIENT_OR_STATE"
            _sync(step.device)
            elapsed = time.perf_counter() - started
            if iteration:
                measured = record[mode]
                measured["full_step_seconds"].append(preparation_seconds + elapsed)
                measured["compute_seconds"].append(elapsed)
                measured["gradient_status"].append(status)
                measured["completed_steps"] += 1
                mode_peak_bytes = torch.cuda.max_memory_allocated(step.device)
                measured["mode_peak_allocated_bytes"].append(mode_peak_bytes)
                measured["mode_start_allocated_bytes"].append(mode_start_bytes)
                measured["mode_incremental_peak_bytes"].append(max(0, mode_peak_bytes - mode_start_bytes))
                measured["peak_allocated_bytes"] = max(
                    measured["peak_allocated_bytes"], preparation_peak, mode_peak_bytes
                )
        del inputs
        save()
    record["status"] = "COMPLETE"
    record["limitation"] = (
        "Short same-initial-state estimate; rejected FP16 update times are not successful training. Concurrent GPU load may affect timings."
    )
    save()
    del step
    gc.collect()
    torch.cuda.empty_cache()


def run_coverage(args, report, sources, save):
    coverage = report["coverage"] = {
        "expected_cases": len(contract.coverage_cases()),
        "expected_updates_per_case": contract.COVERAGE_UPDATES,
        "cases": [],
    }
    for seed in contract.SEEDS:
        # Reuse the immutable U0 model for the seed; reset every controller/optimizer/RNG.
        step = prepare_step(sources, variant="F0", seed=seed)
        initial = training_snapshot(step)
        pairs = runtime.controller_epoch_pairs(sources.subjects, seed=seed, epoch=0)[: contract.COVERAGE_UPDATES]
        for variant in contract.VARIANTS:
            restore_snapshot(step, initial)
            step = replace(step, variant=variant)
            record = {
                "seed": seed,
                "variant": variant,
                "status": "RUNNING",
                "expected_updates": contract.COVERAGE_UPDATES,
                "completed_updates": 0,
                "metrics": [],
            }
            coverage["cases"].append(record)
            for index, pair in enumerate(pairs):
                record["current_pair"] = pair
                save()
                state = training_snapshot(step)
                inputs = None
                stage = "input_preparation"
                failure = None
                try:
                    inputs = runtime._prepare_controller_pair(step, pair, epoch=0)
                    state["rng"] = capture_rng_state()
                    stage = "fp32_update"
                    logs = advance_pair(step, inputs, fp32=True)
                except (RuntimeError, FloatingPointError) as exc:
                    failure = _failure(exc, stage=stage)
                if failure is not None:
                    record.update(status="FAILED", failure=failure)
                    restore_snapshot(step, state)
                    if "first_failure_capture_attempted" not in coverage:
                        coverage["first_failure_capture_attempted"] = True
                        context = {"seed": seed, "variant": variant, "pair_index": index + 1, "pair": pair}
                        _try_capture(args, report, step, inputs, state, context, name="coverage_failure")
                        if inputs is not None and failure["kind"] == "NUMERICAL":
                            record["failure_comparison"] = compare_pair(step, inputs)
                    del inputs
                    break
                record["metrics"].append({"pair_index": index + 1, "pair": pair, "values": logs})
                record["completed_updates"] += 1
                del inputs
            if record["completed_updates"] == contract.COVERAGE_UPDATES:
                record["status"] = "COMPLETE"
            record["last_pair"] = record.pop("current_pair")
            print(f"[PRECISION COVERAGE] seed={seed} variant={variant} {record['status']}", flush=True)
            save()
            step.optimizer.zero_grad(set_to_none=True)
            gc.collect()
            torch.cuda.empty_cache()
        del step
        gc.collect()
        torch.cuda.empty_cache()


def _validate_output_roots(args):
    protected = (args.source_root, args.checkpoint_root, args.image_root, args.data_contract.parent)
    for target in (args.output.parent, args.capture_root):
        for source in protected:
            if target.resolve().is_relative_to(source.resolve()) or source.resolve().is_relative_to(target.resolve()):
                raise RuntimeError("Diagnostic output and source directories must be disjoint")
    if args.capture_root.resolve().is_relative_to(args.output.parent.resolve()):
        raise RuntimeError("Heavy captures must not enter the compact ZIP directory")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--expected-git-head", required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--data-contract", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--job", choices=JOBS, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--capture-root", type=Path, required=True)
    args = parser.parse_args()
    head = assert_clean_exact_git(args.repo_root, args.expected_git_head)
    _validate_output_roots(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8"):
        pass
    report = {
        "schema": contract.WORKER_SCHEMA,
        "job": args.job,
        "source_run": SOURCE_RUN,
        "source_git_head": SOURCE_HEAD,
        "diagnostic_git_head": head,
        "diagnostic_only": True,
        "labels_accessed": False,
        "production_checkpoint_written": False,
        "source_bytes_unchanged": None,
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "STARTING",
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "python": platform.python_version(),
        "limits": {"fp32_epochs": contract.FP32_EPOCHS, "coverage_updates": contract.COVERAGE_UPDATES},
        "workload_contract": contract.workload_contract(),
        "production_training_validated": False,
    }

    def save():
        atomic_write_json(args.output, report)

    sources = None
    exit_code = 0
    try:
        runtime._require_cuda(torch.device("cuda:0"), "precision diagnostic")
        report["gpu"] = torch.cuda.get_device_name(0)
        report["gpu_total_memory_bytes"] = torch.cuda.get_device_properties(0).total_memory
        report["execution_determinism"] = runtime._execution_determinism_contract()
        sources = Sources(args, report)
        if args.job == "coverage":
            run_coverage(args, report, sources, save)
        else:
            # Keep collecting independent evidence after a phase-local error.
            # A failed replay/probe does not explain away the fresh FP32 trajectory.
            for key, function in (
                ("replay", replay_failure),
                ("fp32_trajectory", run_trajectory),
                ("benchmark", benchmark),
            ):
                try:
                    function(args, report, sources, save)
                except Exception as exc:
                    record = report.setdefault(key, {})
                    failure = _failure(exc, stage=key)
                    record.setdefault("failure", failure)
                    record["phase_exception"] = failure
                    if record.get("status") != "FAILED":
                        record["status"] = "INCOMPLETE"
                    report.setdefault("phase_errors", []).append({"phase": key, "traceback": traceback.format_exc()})
                    save()
                gc.collect()
                torch.cuda.empty_cache()
        report["status"] = "DIAGNOSTIC_COMPLETE"
    except Exception as exc:
        report.update(status="EXCEPTION", failure=_failure(exc, stage="worker"), traceback=traceback.format_exc())
        exit_code = 1
    finally:
        records = [report.get(key, {}) for key in ("replay", "fp32_trajectory")]
        records.extend(report.get("coverage", {}).get("cases", []))
        for record in records:
            for key in ("current", "current_pair"):
                if key in record:
                    record["last_pair"] = record.pop(key)
        if sources is not None:
            try:
                sources.verify_unchanged()
            except Exception as exc:
                report.update(status="EXCEPTION", source_bytes_unchanged=False, source_integrity_error=str(exc))
                exit_code = 1
        report["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
        save()
    print(f"[PRECISION DIAG RESULT] {args.job} {report['status']} {args.output}", flush=True)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
