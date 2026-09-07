"""Replay the failed Stage-5 controller epoch without writing training artifacts.

Only training image-only caches are opened. Source checkpoints are read-only.
The finite prefix is replayed in memory; probes on the first failing pair never
update weights. FP32 affects only the controller, not U0/bootstrap/features.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import traceback
from collections import defaultdict
from dataclasses import fields, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import torch

from datasets.OASIS100 import Stage5OasisImageStore
from experiments.stage5 import runtime
from experiments.stage5.checkpoints import capture_rng_state, restore_rng_state, state_dict_sha256
from experiments.stage5.config import ControllerTrainingConfig, U0TrainingConfig, build_stage5_controller
from tools.analysis.run_artifacts import atomic_write_json, sha256_file
from tools.analysis.run_stage5 import assert_clean_exact_git
from tools.analysis.stage5.artifacts import load_canonical_json
from tools.analysis.stage5.contracts import canonical_sha256, validate_protocol_contract
from utils import NCCVxm

SOURCE_HEAD = "458489f77fc6f7c792ba1411bb763f4bb06310c5"
SOURCE_PROTOCOL_SHA = "36e4881fb618103ab2d67c29872ae6f3ae155b285f6c7807d7971b631ee66c88"
PROBE_SCALES = (65536.0, 32768.0, 8192.0, 1024.0, 1.0)
VARIANTS = ("F0", "F2V", "F2S", "F2P")
NCC_REFERENCE_HASHES = {
    "F0": "fec30d4bfceea476edbab3bafe308fec8b0cd7bb4fb0b059129c143392b3404f",
    "F2V": "75ce7cba2a542f6710731ba4501e6fd54bf1bba05b44fc83e45070db91101c2d",
    "F2S": "0b3b587878a4808dcea35288f23d29676cabfa84a0cccd66e1c6ce660c46ab84",
    "F2P": "f54cb5fdc4125d57fc4c3879cfea27687549cac05826c713874278949c793e70",
}


def tensor_stats(tensor):
    value = tensor.detach()
    finite = torch.isfinite(value)
    count = int(finite.sum().item())
    # Avoid NaN/Infinity JSON and avoid an FP16 reduction overflowing itself.
    maximum = float(value.float().abs().masked_fill(~finite, 0).max().item()) if value.numel() else 0.0
    return {
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "elements": value.numel(),
        "nonfinite": value.numel() - count,
        "max_finite_abs": maximum,
    }


def _output_tensors(output, prefix=""):
    if isinstance(output, torch.Tensor):
        yield prefix, output
    elif is_dataclass(output):
        for field in fields(output):
            yield from _output_tensors(getattr(output, field.name), f"{prefix}.{field.name}")
    elif isinstance(output, (tuple, list)):
        for index, item in enumerate(output):
            yield from _output_tensors(item, f"{prefix}.{index}")


class GradientTrace:
    """Record module outputs and their scaled backward gradients in observed order."""

    def __init__(self, model):
        self.model = model
        self.events = []
        self.handles = []
        self.calls = defaultdict(int)

    def __enter__(self):
        for name, module in self.model.named_modules():
            if name and any(module.children()):
                continue
            self.handles.append(module.register_forward_hook(self._hook(name or "controller")))
        return self

    def _hook(self, name):
        def observe(_module, _args, output):
            self.calls[name] += 1
            call = self.calls[name]
            for suffix, tensor in _output_tensors(output):
                label = f"{name}#{call}{suffix}"
                self.events.append({"stage": "forward", "tensor": label, **tensor_stats(tensor)})
                if tensor.requires_grad:
                    self.handles.append(tensor.register_hook(self._gradient(label)))

        return observe

    def _gradient(self, label):
        def observe(gradient):
            self.events.append({"stage": "scaled_backward", "tensor": label, **tensor_stats(gradient)})

        return observe

    def __exit__(self, *_args):
        for handle in self.handles:
            handle.remove()


def probe(model, optimizer, loss_fn, *, device_type, scale, fp32):
    """Measure forward/backward only. Never calls optimizer.step or scaler.step."""
    optimizer.zero_grad(set_to_none=True)
    scaler = torch.amp.GradScaler(device_type, init_scale=scale, growth_interval=1_000_000)
    result = {"mode": "controller_fp32" if fp32 else "controller_fp16", "scale": scale}
    with GradientTrace(model) as trace:
        try:
            loss, logs = loss_fn(fp32)
            finite_loss = bool(torch.isfinite(loss))
            result["loss"] = float(loss.detach()) if finite_loss else None
            result["metrics"] = {k: float(v) if math.isfinite(float(v)) else None for k, v in logs.items()}
            if not finite_loss:
                result["status"] = "NONFINITE_LOSS"
            else:
                scaler.scale(loss).backward()
                result["scaled_parameter_gradients"] = {
                    name: tensor_stats(p.grad) for name, p in model.named_parameters() if p.grad is not None
                }
                result["missing_parameter_gradients"] = [
                    name for name, p in model.named_parameters() if p.requires_grad and p.grad is None
                ]
                if not result["scaled_parameter_gradients"]:
                    raise RuntimeError("No controller gradients reached any parameter")
                scaler.unscale_(optimizer)
                result["unscaled_parameter_gradients"] = {
                    name: tensor_stats(p.grad) for name, p in model.named_parameters() if p.grad is not None
                }
                bad = any(v["nonfinite"] for v in result["unscaled_parameter_gradients"].values())
                result["status"] = "NONFINITE_GRADIENTS" if bad else "FINITE"
        except (RuntimeError, FloatingPointError) as exc:
            if isinstance(exc, torch.cuda.OutOfMemoryError):
                raise
            result["status"] = "PROBE_EXCEPTION"
            result["error"] = f"{type(exc).__name__}: {exc}"
            result["traceback"] = traceback.format_exc()
    result["tensor_events"] = trace.events
    result["first_observed_nonfinite"] = next((e for e in trace.events if e["nonfinite"]), None)
    optimizer.zero_grad(set_to_none=True)
    return result


def require_readonly_checkpoint(path):
    """Reject recovery generations so the production loader cannot rename/delete them."""
    if not path.is_file() or path.is_symlink():
        raise RuntimeError(f"Missing regular checkpoint: {path}")
    for candidate in runtime._checkpoint_generation_paths(path):
        if candidate not in (path, runtime._checkpoint_sidecar_path(path)) and candidate.exists():
            raise RuntimeError(f"Checkpoint recovery required separately; diagnostic refuses to modify {candidate}")
    sidecar = runtime._checkpoint_sidecar_path(path)
    record = json.loads(sidecar.read_text(encoding="utf-8"))
    digest = sha256_file(path)
    if record["sha256"] != digest or record["bytes"] != path.stat().st_size or record["file_name"] != path.name:
        raise RuntimeError(f"Checkpoint/sidecar mismatch: {path}")
    return digest


def prepare_step(args, report):
    runtime._require_cuda(torch.device("cuda:0"), "AMP diagnostic")
    protocol_path = args.source_root / "protocol" / "protocol.json"
    protocol = load_canonical_json(protocol_path)
    validate_protocol_contract(protocol)
    if protocol["git_head"] != SOURCE_HEAD or canonical_sha256(protocol) != SOURCE_PROTOCOL_SHA:
        raise RuntimeError("Diagnostic only supports the reviewed failed bootstrap-V2 run")
    # Verify the training contract as well as its binding in the frozen protocol.
    for name in ("u0", "controller"):
        contract = load_canonical_json(args.source_root / "protocol" / f"{name}_training_contract.json")
        if canonical_sha256(contract) != protocol[f"{name}_training_contract_sha256"]:
            raise RuntimeError(f"{name} training contract mismatch")
    initial = args.checkpoint_root / "controller_initial" / "seed_0" / "initial.pth"
    base = args.checkpoint_root / "u0" / "seed_0" / "last.pth"
    source_hashes = {str(path): require_readonly_checkpoint(path) for path in (base, initial)}
    for checkpoint in (base, initial):
        sidecar = runtime._checkpoint_sidecar_path(checkpoint)
        attestation = args.source_root / "training_attestations" / sidecar.relative_to(args.checkpoint_root)
        frozen = load_canonical_json(attestation)
        if frozen.get("sha256") != source_hashes[str(checkpoint)]:
            raise RuntimeError(f"Checkpoint differs from the failed run's compact attestation: {checkpoint}")
        for path in (sidecar, attestation):
            source_hashes[str(path)] = sha256_file(path)
    source_hashes[str(protocol_path)] = sha256_file(protocol_path)
    runtime._seed_everything(0)
    device = torch.device("cuda:0")
    config = ControllerTrainingConfig()
    store = Stage5OasisImageStore(args.data_contract, args.image_root)
    if store.runtime.contract_sha256 != protocol["data_contract_sha256"]:
        raise RuntimeError("Image-only data contract differs from the failed run")
    controller = build_stage5_controller(config).to(device)
    initial_sha = runtime._load_initial_controller(initial, controller, seed=0, config=config)
    optimizer = torch.optim.AdamW(controller.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scaler = runtime._stage5_grad_scaler(config)
    base_runner = runtime.load_frozen_u0(
        base,
        seed=0,
        device=device,
        protocol_sha256=canonical_sha256(protocol),
        data_contract_sha256=store.runtime.contract_sha256,
        training_contract_sha256=protocol["u0_training_contract_sha256"],
        config=U0TrainingConfig(),
        expected_git_head=SOURCE_HEAD,
    )
    controller.train()
    report.update(
        {
            "source_hashes": source_hashes,
            "protocol_sha256": canonical_sha256(protocol),
            "initial_controller_state_sha256": initial_sha,
            "data_contract_sha256": store.runtime.contract_sha256,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "python": platform.python_version(),
            "gpu": torch.cuda.get_device_name(device),
            "determinism": runtime._execution_determinism_contract(),
        }
    )
    return runtime._ControllerStep(
        store=store,
        base_runner=base_runner,
        controller=controller,
        optimizer=optimizer,
        scaler=scaler,
        device=device,
        variant=args.variant,
        bootstrap_policy=protocol["bootstrap"]["policy"],
        config=config,
    )


def classify_probes(probes):
    baseline = probes[0]["status"]
    lower = any(p["status"] == "FINITE" for p in probes[1:-1])
    reference = probes[-1]["status"]
    if baseline == "FINITE":
        return "FAILURE_NOT_REPRODUCED_IN_INSTRUMENTED_PROBE"
    if baseline == "NONFINITE_GRADIENTS" and lower:
        return "SCALE_SENSITIVE_OVERFLOW_ON_THIS_PAIR"
    if reference == "FINITE":
        return "FP16_PATH_FAILURE_NOT_RESOLVED_BY_TESTED_SCALES"
    return "FAILURE_PERSISTS_IN_FP32_OR_PROBE_ERROR"


def load_ncc_reference(args):
    if not getattr(args, "ncc_audit", False):
        return None
    if args.reference_report is None or sha256_file(args.reference_report) != NCC_REFERENCE_HASHES[args.variant]:
        raise RuntimeError("NCC audit requires the exact reviewed H100 AMP report for this variant")
    return json.loads(args.reference_report.read_text(encoding="utf-8"))


def audit_ncc_failure(args, report, step, inputs, pair, rng, reference):
    from tools.analysis.stage5.ncc_diagnostic import audit_pair

    for name, digest in reference["source_hashes"].items():
        if report["source_hashes"].get(name) != digest:
            raise RuntimeError(f"NCC audit source bytes differ from the reviewed failure: {name}")
    for key in (
        "pair",
        "pair_index_one_based",
        "successful_in_memory_updates",
        "pair_schedule_sha256",
        "initial_controller_state_sha256",
        "protocol_sha256",
        "data_contract_sha256",
    ):
        if report[key] != reference[key]:
            raise RuntimeError(f"NCC audit replay differs from the reviewed failure: {key}")
    # CUDA grid_sample backward is not promised bitwise deterministic. Distinguish
    # historical replay equality from exact tensor/state reuse within this attempt.
    report["historical_controller_state_match"] = (
        report["failing_controller_state_sha256"] == reference["failing_controller_state_sha256"]
    )
    report["reference_controller_state_sha256"] = reference["failing_controller_state_sha256"]

    def save(result):
        report["ncc_audit"] = result
        atomic_write_json(args.output, report)
        print(f"[NCC DIAG PROGRESS] {args.variant} {result['status']}", flush=True)

    report["ncc_audit"] = audit_pair(step, inputs, pair, rng, probe_fn=probe, save=save)
    report["interpretation"] = "INSPECT_NCC_VALUES_AND_GRADIENTS_NOT_A_TRAINING_SUCCESS"


def diagnose(args, report):
    # These tools replay the fixed historical source protocol, not the current
    # production objective. Keep its NCC explicit after the production fix.
    with patch("experiments.stage5.losses.ControllerNCC", NCCVxm):
        _diagnose_legacy_source(args, report)


def _diagnose_legacy_source(args, report):
    reference = load_ncc_reference(args)
    step = prepare_step(args, report)
    pairs = runtime.controller_epoch_pairs(runtime._training_subjects(step.store), seed=0, epoch=0)
    report["pair_schedule_sha256"] = canonical_sha256(pairs)
    report["max_pairs"] = len(pairs)
    report["successful_in_memory_updates"] = 0
    report["status"] = "REPLAYING_FIRST_EPOCH"
    report["replay_metrics"] = []
    if reference is not None:
        report["reference_report_sha256"] = sha256_file(args.reference_report)
        report["source_hashes"][str(args.reference_report)] = report["reference_report_sha256"]
        pairs = pairs[: reference["pair_index_one_based"]]
        report["max_pairs"] = len(pairs)
    for index, pair in enumerate(pairs):
        report["pair_index_one_based"] = index + 1
        report["pair"] = pair
        atomic_write_json(args.output, report)
        print(f"[AMP DIAG PAIR] {args.variant} {index + 1}/{len(pairs)} {pair}", flush=True)
        inputs = runtime._prepare_controller_pair(step, pair, epoch=0)
        rng = capture_rng_state()
        step.optimizer.zero_grad(set_to_none=True)
        try:
            loss, _logs = runtime._controller_pair_loss(step, inputs)
            report["replay_metrics"].append({"pair": pair, "metrics": _logs})
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("non-finite controller loss during diagnostic replay")
            step.scaler.scale(loss).backward()
            runtime._strict_scaler_step(step.scaler, step.optimizer, phase=f"diagnostic replay {args.variant}")
        except FloatingPointError as exc:
            report["reproduced_error"] = str(exc)
            # Release the failed autograd graph before probing identical prepared inputs.
            if "loss" in locals():
                del loss
            step.optimizer.zero_grad(set_to_none=True)
            before = state_dict_sha256(step.controller.state_dict())
            report["failing_controller_state_sha256"] = before
            if reference is not None:
                audit_ncc_failure(args, report, step, inputs, pair, rng, reference)
                report["status"] = "NCC_AUDIT_COMPLETE"
                break
            report["probes"] = []
            for scale, fp32 in [*((s, False) for s in PROBE_SCALES), (1.0, True)]:
                restore_rng_state(rng)
                trial = probe(
                    step.controller,
                    step.optimizer,
                    lambda use_fp32, pair_inputs=inputs: runtime._controller_pair_loss(
                        step, pair_inputs, diagnostic_fp32=use_fp32
                    ),
                    device_type="cuda",
                    scale=scale,
                    fp32=fp32,
                )
                if state_dict_sha256(step.controller.state_dict()) != before:
                    raise RuntimeError("Diagnostic probe changed controller state") from exc
                report["probes"].append(trial)
                atomic_write_json(args.output, report)
                print(f"[AMP DIAG PROBE] {args.variant} {trial['mode']} scale={scale:g} {trial['status']}", flush=True)
            report["interpretation"] = classify_probes(report["probes"])
            report["status"] = "DIAGNOSTIC_COMPLETE"
            break
        report["successful_in_memory_updates"] += 1
        del loss, inputs
    else:
        report["status"] = "NOT_REPRODUCED_WITHIN_REVIEWED_PREFIX" if reference else "NOT_REPRODUCED_WITHIN_FIRST_EPOCH"
    for name, digest in report["source_hashes"].items():
        if sha256_file(Path(name)) != digest:
            raise RuntimeError(f"Source artifact changed during diagnostic: {name}")
    report["source_bytes_unchanged"] = True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--expected-git-head", required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--data-contract", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--variant", choices=VARIANTS, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ncc-audit", action="store_true", help="Audit NCC on the reviewed failing pair, no AMP sweep")
    parser.add_argument("--reference-report", type=Path, help="Reviewed prior H100 JSON for this variant")
    args = parser.parse_args()
    head = assert_clean_exact_git(args.repo_root, args.expected_git_head)
    args.output = args.output.resolve()
    protected = (args.source_root, args.checkpoint_root, args.image_root, args.data_contract.parent)
    if any(args.output.is_relative_to(p.resolve()) for p in protected):
        raise RuntimeError("Diagnostic output must not be inside any source directory")
    # Refuse overwrite before any expensive work. Parent belongs to a new diagnostic attempt.
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8"):
        pass
    report = {
        "schema": "ctcf-stage5-ncc-diagnostic-v1" if args.ncc_audit else "ctcf-stage5-amp-diagnostic-v1",
        "diagnostic_git_head": head,
        "source_git_head": SOURCE_HEAD,
        "variant": args.variant,
        "seed": 0,
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "diagnostic_only": True,
        "checkpoint_written": False,
        "labels_accessed": False,
        "development_images_accessed": False,
        "heldout_test_accessed": False,
        "probe_scales": list(PROBE_SCALES),
        "fp32_reference_controller_only": True,
        "status": "INITIALIZING",
    }
    if args.ncc_audit:
        report.pop("fp32_reference_controller_only")
        report.update(probe_scales=[65536.0, 1.0], fp64_reference_ncc_only=True)
    code = 0
    try:
        diagnose(args, report)
    except Exception as exc:
        report.update(status="DIAGNOSTIC_ERROR", error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc())
        code = 1
    finally:
        report["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
        atomic_write_json(args.output, report)
        print(f"[AMP DIAG RESULT] {args.variant} {report['status']} {args.output}", flush=True)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
