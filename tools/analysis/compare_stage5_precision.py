"""Bounded image-only comparison workers, isolated from production checkpoints."""

from __future__ import annotations

import argparse
import gc
import json
import os
import platform
import time
import traceback
from contextlib import nullcontext
from datetime import datetime, timezone
from pathlib import Path

import torch

from datasets.OASIS100 import Stage5OasisImageStore
from experiments.stage5 import runtime
from experiments.stage5.checkpoints import atomic_torch_save, capture_rng_state, state_dict_sha256
from experiments.stage5.config import ControllerTrainingConfig, U0TrainingConfig, build_stage5_controller
from experiments.stage5.failures import classify_failure_chain, cpu_snapshot, failure_exception_chain
from experiments.stage5.ncc import controller_ncc_contract
from experiments.stage5.precision import precision_mode_contract
from experiments.stage5.telemetry import parameter_telemetry
from tools.analysis.run_artifacts import atomic_write_json, sha256_file
from tools.analysis.run_stage5 import _protocol_context
from tools.analysis.stage5 import comparison_contract as contract
from tools.analysis.stage5.comparison_math import (
    ArithmeticObserver,
    advance,
    mapping_difference,
    restore,
    snapshot,
    tensor_difference,
)
from tools.analysis.stage5.primitives import canonical_sha256, is_link_like


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def mode_supported(mode):
    return mode != "bf16" or torch.cuda.is_bf16_supported(including_emulation=False)


def error_status(exc):
    kind, _ = classify_failure_chain(failure_exception_chain(exc))
    return {"RESOURCE": "OOM", "NUMERICAL": "MATH_ERROR"}.get(kind, "ERROR")


class Sources:
    def __init__(self, args, report):
        self.args, self.report = args, report
        self.head, self.protocol = _protocol_context(args)
        self.protocol_sha = canonical_sha256(self.protocol)
        self.hashes = report.setdefault("source_hashes", {})
        self.remember(args.protocol)
        self.remember(args.data_contract)
        self.store = Stage5OasisImageStore(args.data_contract, args.image_root)
        if self.store.runtime.contract_sha256 != self.protocol["data_contract_sha256"]:
            raise RuntimeError("Comparison data contract differs from protocol")
        self.subjects = runtime._training_subjects(self.store)
        contract.expected_updates(args.stage, len(self.subjects))
        self.base_runner = None
        self.device = torch.device("cuda:0")
        report.update(protocol_sha256=self.protocol_sha, data_contract_sha256=self.store.runtime.contract_sha256)

    def remember(self, path):
        if not path.is_file() or is_link_like(path):
            raise RuntimeError(f"Expected a regular comparison source: {path}")
        digest = sha256_file(path)
        if self.hashes.setdefault(str(path.resolve()), digest) != digest:
            raise RuntimeError(f"Comparison source changed: {path}")
        return digest

    def new_step(self):
        args = self.args
        runtime._seed_everything(args.seed)
        config = ControllerTrainingConfig()
        model = build_stage5_controller(config).to(self.device).train()
        initial = args.checkpoint_root / "controller_initial" / f"seed_{args.seed}" / "initial.pth"
        self.remember(initial)
        self.remember(runtime._checkpoint_sidecar_path(initial))
        runtime._load_initial_controller(initial, model, seed=args.seed, config=config)
        if self.base_runner is None:
            base = args.checkpoint_root / "u0" / f"seed_{args.seed}" / "last.pth"
            self.remember(base)
            self.remember(runtime._checkpoint_sidecar_path(base))
            self.base_runner = runtime.load_frozen_u0(
                base,
                seed=args.seed,
                device=self.device,
                protocol_sha256=self.protocol_sha,
                data_contract_sha256=self.store.runtime.contract_sha256,
                training_contract_sha256=self.protocol["u0_training_contract_sha256"],
                config=U0TrainingConfig(),
                expected_git_head=self.head,
            )
        step = runtime._ControllerStep(
            store=self.store,
            base_runner=self.base_runner,
            controller=model,
            optimizer=torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay),
            scaler=None,
            device=self.device,
            variant=args.variant,
            bootstrap_policy=self.protocol["bootstrap"]["policy"],
            config=config,
        )
        # Construction may consume RNG; every fresh diagnostic arm starts here.
        runtime._seed_everything(args.seed)
        return step

    def verify(self):
        for name, digest in self.hashes.items():
            if sha256_file(Path(name)) != digest:
                raise RuntimeError(f"Comparison modified a source: {name}")
        self.report["sources_unchanged"] = True


def failure_record(args, step, inputs, state, exc, context):
    """Preserve mode-specific diagnostics; never emit a production checkpoint."""
    name = f"{context['mode']}_{context.get('replicate', 0)}_{time.time_ns()}"
    directory = args.heavy_root / "failures" / name
    directory.mkdir(parents=True, exist_ok=False)
    record = {
        "kind": error_status(exc),
        "exception_type": type(exc).__name__,
        "message": str(exc),
        "traceback": traceback.format_exc(),
        "context": context,
        "phase": state.get("phase", "prepare_inputs"),
        "production_resume_forbidden": True,
        "precision": precision_mode_contract(context["mode"]),
        "capture_status": "PENDING",
    }
    compact = args.output_root / f"failure_{name}.json"
    atomic_write_json(compact, record)
    try:
        payload = {
            **record,
            "schema": "ctcf-stage5-comparison-capture-v1",
            "capture_status": "COMPLETE",
            "git_head": args.expected_git_head,
            "state": cpu_snapshot(state),
            "inputs": cpu_snapshot(inputs),
            "model_at_failure": cpu_snapshot(step.controller.state_dict()),
            "optimizer_at_failure": cpu_snapshot(step.optimizer.state_dict()),
            "gradients_at_failure": cpu_snapshot({n: p.grad for n, p in step.controller.named_parameters()}),
        }
        path = directory / "capture.pth"
        digest = atomic_torch_save(path, payload)
        record.update(
            capture_status="COMPLETE",
            capture_file={"path": str(path.resolve()), "sha256": digest, "bytes": path.stat().st_size},
        )
    except Exception as capture_error:
        record.update(capture_status="FAILED", capture_error=f"{type(capture_error).__name__}: {capture_error}")
    atomic_write_json(compact, record)
    return record


def new_case(mode, replicate=0):
    return {
        "mode": mode,
        "replicate": replicate,
        "precision": precision_mode_contract(mode),
        "status": "RUNNING",
        "successful_updates": 0,
        "full_step_seconds": [],
        "compute_seconds": [],
        "preparation_seconds": [],
        "telemetry_seconds": [],
        "peak_allocated_bytes": 0,
        "peak_reserved_bytes": 0,
        "epochs": [],
        "timing_kind": "UNHOOKED_STEP_WITH_PREUPDATE_SNAPSHOT_PLUS_TELEMETRY_EXCLUDING_REPORT_IO",
        "memory_kind": "CUDA_ALLOCATOR_PEAKS_PER_WORKER_EXCLUDING_NON_PYTORCH_ALLOCATIONS",
    }


def observe_memory(record, *, failure=False):
    allocated = torch.cuda.max_memory_allocated()
    reserved = torch.cuda.max_memory_reserved()
    record["peak_allocated_bytes"] = max(record["peak_allocated_bytes"], allocated)
    record["peak_reserved_bytes"] = max(record["peak_reserved_bytes"], reserved)
    if failure:
        record["failed_step_memory"] = {"peak_allocated_bytes": allocated, "peak_reserved_bytes": reserved}


def wait_start_gate(args):
    if args.ready_file is None and args.start_gate is None:
        return
    if args.ready_file is None or args.start_gate is None:
        raise ValueError("Throughput requires both ready file and start gate")
    args.ready_file.parent.mkdir(parents=True, exist_ok=True)
    with args.ready_file.open("x", encoding="utf-8") as stream:
        json.dump({"pid": os.getpid(), "ready_at_utc": utc_now()}, stream)
    deadline = time.monotonic() + args.gate_timeout_seconds
    while not args.start_gate.is_file():
        if time.monotonic() >= deadline:
            raise TimeoutError("Comparison start gate timed out")
        time.sleep(0.1)


def train_case(args, sources, *, mode, replicate=0, updates=None, use_gate=False):
    record = new_case(mode, replicate)
    record["expected_updates"] = updates if updates is not None else contract.expected_updates("trajectory")
    if not mode_supported(mode):
        record["status"] = "UNSUPPORTED"
        return record
    torch.cuda.reset_peak_memory_stats()
    try:
        step = sources.new_step()
    except Exception as exc:
        if error_status(exc) != "OOM":
            raise
        observe_memory(record, failure=True)
        record.update(
            status="CANDIDATE_FAILURE",
            failure={
                "kind": "OOM",
                "phase": "initialize",
                "message": str(exc),
                "traceback": traceback.format_exc(),
                "capture_status": "NOT_AVAILABLE_INITIALIZATION_INCOMPLETE",
                "production_resume_forbidden": True,
            },
        )
        gc.collect()
        torch.cuda.empty_cache()
        return record
    observe_memory(record)
    record["initial_model_sha256"] = state_dict_sha256(step.controller.state_dict())
    log_path = args.output_root / f"steps_{mode}_{replicate}.jsonl"
    record["step_records"] = log_path.name
    if use_gate:
        wait_start_gate(args)
    record["training_started_at_utc"] = utc_now()
    training_started = time.perf_counter()
    try:
        with log_path.open("x", encoding="utf-8", newline="\n") as log:
            for epoch in range(contract.EPOCHS):
                pairs = runtime.controller_epoch_pairs(sources.subjects, seed=args.seed, epoch=epoch)
                epoch_rows = []
                for index, pair in enumerate(pairs):
                    context = {
                        "mode": mode,
                        "replicate": replicate,
                        "epoch": epoch + 1,
                        "pair_index": index + 1,
                        "pair": pair,
                    }
                    inputs = None
                    state = {"phase": "prepare_inputs", "rng_before_pair": capture_rng_state()}
                    torch.cuda.synchronize(step.device)
                    torch.cuda.reset_peak_memory_stats(step.device)
                    started = time.perf_counter()
                    try:
                        inputs = runtime._prepare_controller_pair(step, pair, epoch)
                        state["rng_before_controller"] = capture_rng_state()
                        torch.cuda.synchronize(step.device)
                        prepared = time.perf_counter()
                        observer = ArithmeticObserver(step.controller) if record["successful_updates"] == 0 else None
                        with observer if observer is not None else nullcontext():
                            metrics = advance(step, inputs, mode, failure_state=state)
                        if observer is not None:
                            record["observed_arithmetic"] = observer.validate(mode)
                            record["dtype_observer_update_indices_zero_based"] = [0]
                        torch.cuda.synchronize(step.device)
                        computed = time.perf_counter()
                        telemetry = parameter_telemetry(step.controller, before_update=state["before_update"]["model"])
                        finished = time.perf_counter()
                    except Exception as exc:
                        observe_memory(record, failure=True)
                        record["failure"] = failure_record(args, step, inputs, state, exc, context)
                        record["status"] = (
                            "CANDIDATE_FAILURE" if error_status(exc) in {"OOM", "MATH_ERROR"} else "INCOMPLETE"
                        )
                        return record
                    record["successful_updates"] += 1
                    record["preparation_seconds"].append(prepared - started)
                    record["compute_seconds"].append(computed - prepared)
                    record["telemetry_seconds"].append(finished - computed)
                    record["full_step_seconds"].append(finished - started)
                    record["peak_allocated_bytes"] = max(
                        record["peak_allocated_bytes"], torch.cuda.max_memory_allocated(step.device)
                    )
                    record["peak_reserved_bytes"] = max(
                        record["peak_reserved_bytes"], torch.cuda.max_memory_reserved(step.device)
                    )
                    row = {**context, "metrics": metrics, "telemetry": telemetry}
                    log.write(json.dumps(row, allow_nan=False, separators=(",", ":")) + "\n")
                    log.flush()
                    epoch_rows.append(metrics)
                    del inputs
                    if (index + 1) % max(1, len(pairs) // 10) == 0:
                        print(
                            f"[COMPARISON PROGRESS] {args.variant} {mode}/{replicate} epoch={epoch + 1} pair={index + 1}/{len(pairs)}",
                            flush=True,
                        )
                    if record["successful_updates"] == record["expected_updates"]:
                        break
                record["epochs"].append(
                    {
                        "epoch": epoch + 1,
                        "updates": len(epoch_rows),
                        "pair_schedule_sha256": canonical_sha256(pairs),
                        "metrics": {
                            key: sum(row[key] for row in epoch_rows) / len(epoch_rows) for key in epoch_rows[0]
                        },
                    }
                )
                if record["successful_updates"] == record["expected_updates"]:
                    break
        if record["successful_updates"] != record["expected_updates"]:
            raise RuntimeError("Comparison trajectory did not complete its declared update count")
        record["training_finished_at_utc"] = utc_now()
        record["training_wall_seconds"] = time.perf_counter() - training_started
        record["status"] = "COMPLETE"
        record["final_model_sha256"] = state_dict_sha256(step.controller.state_dict())
        if updates is None:
            endpoint = args.heavy_root / f"{mode}_{replicate}_diagnostic_state.pth"
            digest = atomic_torch_save(
                endpoint,
                {
                    "schema": "ctcf-stage5-comparison-state-v1",
                    "production_resume_forbidden": True,
                    "git_head": args.expected_git_head,
                    "protocol_sha256": sources.protocol_sha,
                    "mode": mode,
                    "variant": args.variant,
                    "seed": args.seed,
                    "replicate": replicate,
                    "successful_updates": record["successful_updates"],
                    "state": snapshot(step),
                },
            )
            record["diagnostic_state"] = {
                "path": str(endpoint.resolve()),
                "sha256": digest,
                "bytes": endpoint.stat().st_size,
            }
        return record
    finally:
        if "training_finished_at_utc" not in record:
            record["training_finished_at_utc"] = utc_now()
            record["training_wall_seconds"] = time.perf_counter() - training_started
        del step
        gc.collect()
        torch.cuda.empty_cache()


def paired_case(args, sources, report):
    if args.variant in ("F0", "F2V"):
        from tools.analysis.diagnose_stage5_mechanism import CaptureSources

        captures = CaptureSources(args.repo_root, args.precision_source_root, report)
        step, inputs = captures.load_failure(args.variant, device=sources.device, capture_root=args.capture_root)
        step.scaler = None
        step.config = ControllerTrainingConfig()
        verify = captures.verify_unchanged
        report["paired_state_origin"] = "HASH_VERIFIED_SAVED_FAILURE"
    else:
        step = sources.new_step()
        pairs = runtime.controller_epoch_pairs(sources.subjects, seed=args.seed, epoch=0)
        # Zero-initialized output heads hide backward sensitivity at the first pair.
        for pair in pairs[: contract.COVERAGE_UPDATES]:
            inputs = runtime._prepare_controller_pair(step, pair, 0)
            advance(step, inputs, "fp32_strict")
            del inputs
        inputs = runtime._prepare_controller_pair(step, pairs[contract.COVERAGE_UPDATES], 0)
        verify = sources.verify
        report["paired_state_origin"] = f"FRESH_INITIAL_AFTER_{contract.COVERAGE_UPDATES}_STRICT_UPDATES"
    initial = snapshot(step)
    report["initial_model_sha256"] = state_dict_sha256(initial["model"])
    report["same_prepared_input_objects"] = True
    cases, observations = [], {}
    report["cases"] = cases
    for mode, replicate in contract.PAIRED_MODES:
        case = {"mode": mode, "replicate": replicate, "precision": precision_mode_contract(mode), "status": "RUNNING"}
        cases.append(case)
        if not mode_supported(mode):
            case["status"] = "UNSUPPORTED"
            continue
        restore(step, initial)
        failure_state = {"rng_before_controller": initial["rng"]}
        try:
            with ArithmeticObserver(step.controller, capture_fields=True) as observer:
                case["metrics"] = advance(step, inputs, mode, failure_state=failure_state)
            case["observed_arithmetic"] = observer.validate(mode)
            observation = {
                "fields": observer.fields,
                "field_gradients": observer.field_gradients,
                "parameter_gradients": cpu_snapshot(
                    {n: p.grad for n, p in step.controller.named_parameters() if p.grad is not None}
                ),
                "parameter_updates": {
                    n: p.detach().cpu().double() - initial["model"][n].double()
                    for n, p in step.controller.named_parameters()
                },
            }
            observations[(mode, replicate)] = observation
            case["status"] = "COMPLETE"
        except Exception as exc:
            case["failure"] = failure_record(
                args, step, inputs, failure_state, exc, {"mode": mode, "replicate": replicate}
            )
            case["status"] = "CANDIDATE_FAILURE" if error_status(exc) in {"OOM", "MATH_ERROR"} else "INCOMPLETE"
        reference = observations.get(("fp32_strict", 0))
        actual = observations.get((mode, replicate))
        if reference is not None and actual is not None:
            case["difference_vs_strict"] = {
                name: {
                    direction: tensor_difference(actual[name][direction], reference[name][direction])
                    for direction in ("forward", "reverse")
                }
                for name in ("fields", "field_gradients")
            }
            for name in ("parameter_gradients", "parameter_updates"):
                case["difference_vs_strict"][name] = mapping_difference(actual[name], reference[name])
        if (mode, replicate) != ("fp32_strict", 0):
            observations.pop((mode, replicate), None)
        atomic_write_json(args.output_root / "paired_progress.json", {"cases": cases})
        gc.collect()
        torch.cuda.empty_cache()
    # Separate warm, unhooked same-state timings; no tensor comparison in timer.
    report["benchmark"] = benchmark_pair(step, inputs, initial)
    restore(step, initial)
    verify()
    return cases


def benchmark_pair(step, inputs, initial):
    result = {
        mode: {"compute_seconds": [], "incremental_peak_bytes": [], "status": "COMPLETE"} for mode in contract.MODES
    }
    for iteration in range(contract.BENCHMARK_REPEATS + 1):
        modes = contract.MODES if iteration % 2 == 0 else tuple(reversed(contract.MODES))
        for mode in modes:
            record = result[mode]
            if not mode_supported(mode):
                record["status"] = "UNSUPPORTED"
                continue
            if record["status"] != "COMPLETE":
                continue
            restore(step, initial)
            torch.cuda.synchronize(step.device)
            torch.cuda.reset_peak_memory_stats(step.device)
            baseline = torch.cuda.memory_allocated(step.device)
            started = time.perf_counter()
            try:
                advance(step, inputs, mode)
                torch.cuda.synchronize(step.device)
            except Exception as exc:
                record.update(
                    status="CANDIDATE_FAILURE" if error_status(exc) in {"OOM", "MATH_ERROR"} else "INCOMPLETE",
                    error=str(exc),
                    kind=error_status(exc),
                )
                continue
            elapsed = time.perf_counter() - started
            if iteration:
                record["compute_seconds"].append(elapsed)
                record["incremental_peak_bytes"].append(max(0, torch.cuda.max_memory_allocated(step.device) - baseline))
    return {
        "warmup_iterations_per_mode": 1,
        "repeats": contract.BENCHMARK_REPEATS,
        "shared_preparation_excluded": True,
        "cases": result,
    }


def run_worker(args):
    if args.stage in {"paired", "trajectory"} and (args.seed != 0 or args.variant not in contract.TRAJECTORY_VARIANTS):
        raise ValueError("Paired probes and trajectories use the declared seed-0 variants")
    if args.stage == "throughput" and (args.seed != 0 or args.variant != "F0"):
        raise ValueError("Throughput uses matched F0 seed-0 workers")
    for path in (args.output_root, args.heavy_root):
        if is_link_like(path):
            raise ValueError("Comparison output must not be a link")
        path.mkdir(parents=True, exist_ok=True)
    if args.heavy_root.resolve().is_relative_to(args.output_root.resolve()):
        raise ValueError("Heavy comparison output must be outside compact output")
    destination = args.output_root / "result.json"
    if destination.exists():
        raise FileExistsError("Comparison worker never overwrites an existing result")
    report = {
        "schema": contract.SCHEMA,
        "stage": args.stage,
        "variant": args.variant,
        "seed": args.seed,
        "mode": args.mode,
        "replicate": args.replicate,
        "git_head": args.expected_git_head,
        "run_id": args.run_id,
        "status": "RUNNING",
        "started_at_utc": utc_now(),
        "workload_contract": contract.workload_contract(),
        "expected_updates": contract.expected_updates(args.stage),
        "successful_updates": 0,
        "labels_accessed": False,
        "production_checkpoint_written": False,
        "automatic_mode_selection": False,
        "ncc_contract": controller_ncc_contract(),
    }
    atomic_write_json(destination, report)
    try:
        runtime._require_cuda(torch.device("cuda:0"), "precision comparison")
        report["environment"] = {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
            "device": torch.cuda.get_device_name(0),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "bf16_native_supported": mode_supported("bf16"),
            "execution_determinism": runtime._execution_determinism_contract(),
        }
        sources = Sources(args, report)
        if args.stage == "paired":
            cases = paired_case(args, sources, report)
            report["successful_updates"] = sum(case["status"] == "COMPLETE" for case in cases)
        elif args.stage == "coverage":
            cases = report["cases"] = []
            for mode in contract.MODES:
                cases.append(train_case(args, sources, mode=mode, updates=contract.COVERAGE_UPDATES))
                atomic_write_json(destination, report)
        else:
            cases = [
                train_case(args, sources, mode=args.mode, replicate=args.replicate, use_gate=args.stage == "throughput")
            ]
            report.update({key: value for key, value in cases[0].items() if key not in {"status", "mode", "replicate"}})
        report["cases"] = cases
        if args.stage != "paired":
            report["successful_updates"] = sum(case["successful_updates"] for case in cases)
        statuses = {case["status"] for case in cases}
        if "benchmark" in report:
            statuses.update(case["status"] for case in report["benchmark"]["cases"].values())
        report["status"] = (
            "INCOMPLETE"
            if "INCOMPLETE" in statuses
            else "CANDIDATE_FAILURE"
            if "CANDIDATE_FAILURE" in statuses
            else "UNSUPPORTED"
            if "UNSUPPORTED" in statuses
            else "COMPLETE"
        )
        sources.verify()
    except Exception as exc:
        report.update(
            status="INCOMPLETE",
            error={"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()},
        )
    report["finished_at_utc"] = utc_now()
    atomic_write_json(destination, report)
    print(
        f"[COMPARISON RESULT] {args.stage} {args.variant} {args.mode}/{args.replicate} {report['status']} {destination}",
        flush=True,
    )
    return 1 if report["status"] == "INCOMPLETE" else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("worker",))
    parser.add_argument("--stage", required=True, choices=("paired", "coverage", "trajectory", "throughput"))
    parser.add_argument("--variant", required=True, choices=contract.VARIANTS)
    parser.add_argument("--mode", choices=contract.MODES, default="fp32_strict")
    parser.add_argument("--replicate", type=int, choices=(0, 1), default=0)
    parser.add_argument("--seed", type=int, choices=contract.SEEDS, default=0)
    for name in (
        "protocol",
        "data-contract",
        "image-root",
        "checkpoint-root",
        "output-root",
        "heavy-root",
        "repo-root",
        "precision-source-root",
        "capture-root",
    ):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--expected-git-head", required=True)
    parser.add_argument("--ready-file", type=Path)
    parser.add_argument("--start-gate", type=Path)
    parser.add_argument("--gate-timeout-seconds", type=float, default=900)
    return run_worker(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
