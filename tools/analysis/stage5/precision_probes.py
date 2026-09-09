"""Read-only numerical probes for the current Stage5 controller objective.

This module never updates an optimizer or substitutes the production NCC. The
FP32 comparison is a diagnostic reference, not a production precision policy.
Hooked timings are deliberately not presented as training benchmarks.
"""

from __future__ import annotations

import copy
import math
import time
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import fields, is_dataclass
from itertools import product
from unittest.mock import patch

import numpy as np
import torch
import torch.nn.functional as F

from experiments.stage5 import losses, runtime
from experiments.stage5.checkpoints import capture_rng_state, restore_rng_state, state_dict_sha256
from experiments.stage5.ncc import NCC_VARIANCE_FLOOR, ControllerNCC, controller_ncc_contract
from tools.analysis.stage5.precision_contract import PROBE_SCALES, error_status as _error_status

COMPARISON_TOLERANCES = {
    "purpose": "diagnostic screening, not a convergence or scientific-quality guarantee",
    "mixed_precision_gradient": {"relative": 0.05, "absolute": 1e-7, "cosine_minimum": 0.999},
    "ncc_crop": {
        "value_absolute": 2e-6,
        "gradient": {"relative": 2e-4, "absolute": 1e-7, "cosine_minimum": 0.0},
    },
    "component_gradient_sum": {"relative": 2e-4, "absolute": 1e-7, "cosine_minimum": 0.999},
}


def _finite_float(value):
    result = float(value)
    return result if math.isfinite(result) else None


def tensor_stats(tensor):
    value = tensor.detach()
    # Hooks must not allocate a full-volume FP64 feature map merely to log it.
    # Reduce flat chunks of at most one million elements to cap workspaces.
    flat = value.reshape(-1)
    nonfinite, maximum, squared_l2 = 0, 0.0, 0.0
    for chunk in flat.split(1_000_000):
        if not chunk.numel():
            continue
        finite = torch.isfinite(chunk)
        safe = chunk.double().masked_fill(~finite, 0)
        nonfinite += int((~finite).sum())
        maximum = max(maximum, float(safe.abs().max()))
        squared_l2 += float(safe.square().sum())
    return {
        "dtype": str(value.dtype),
        "shape": list(value.shape),
        "elements": value.numel(),
        "nonfinite": nonfinite,
        "max_finite_abs": maximum,
        "finite_l2": _finite_float(math.sqrt(squared_l2)),
    }


def gradient_difference(actual, reference, *, relative=None, absolute=None, cosine_minimum=None):
    """Report all errors; the mixed absolute/relative test is safe near zero."""
    defaults = COMPARISON_TOLERANCES["mixed_precision_gradient"]
    relative = defaults["relative"] if relative is None else relative
    absolute = defaults["absolute"] if absolute is None else absolute
    cosine_minimum = defaults["cosine_minimum"] if cosine_minimum is None else cosine_minimum
    tolerances = {"relative": relative, "absolute": absolute, "cosine_minimum": cosine_minimum}
    actual, reference = actual.detach().double().cpu(), reference.detach().double().cpu()
    if actual.shape != reference.shape:
        raise ValueError("gradient comparison shape mismatch")
    if not bool(torch.isfinite(actual).all() and torch.isfinite(reference).all()):
        return {"status": "NONFINITE", "within_heuristic_tolerance": False, "tolerances": tolerances}
    difference = actual - reference
    error, norm = float(difference.norm()), float(reference.norm())
    observed_norm = float(actual.norm())
    cosine = None
    if norm > absolute and observed_norm > absolute:
        cosine = max(-1.0, min(1.0, float((actual.flatten() @ reference.flatten()) / (norm * observed_norm))))
    close = error <= absolute + relative * norm
    return {
        "status": "FINITE",
        "tolerances": tolerances,
        "reference_l2": norm,
        "actual_l2": observed_norm,
        "absolute_l2": error,
        "relative_l2": error / norm if norm > absolute else None,
        "reference_near_zero": norm <= absolute,
        "max_abs": float(difference.abs().max()) if difference.numel() else 0.0,
        "cosine": cosine,
        "within_heuristic_tolerance": close and (cosine is None or cosine >= cosine_minimum),
    }


def backend_flags():
    return {
        "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
        "matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
    }


@contextmanager
def strict_fp32(enabled=True):
    """Scope IEEE FP32 settings; keep default backend flags for FP16 probes."""
    previous = backend_flags()
    try:
        if enabled:
            torch.set_float32_matmul_precision("highest")
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
        yield
    finally:
        torch.set_float32_matmul_precision(previous["float32_matmul_precision"])
        torch.backends.cuda.matmul.allow_tf32 = previous["matmul_allow_tf32"]
        torch.backends.cudnn.allow_tf32 = previous["cudnn_allow_tf32"]


def precision_context(fp32):
    return strict_fp32(fp32)


def centered_ncc_reference(first, second, window):
    """Independent FP64 explicit windows; bounded crops only, never full volumes.

    Zero padding participates in the window mean and variance, exactly as in
    production. Directly centered samples avoid subtraction of raw moments.
    """
    if first.shape != second.shape or first.ndim != 5 or first.shape[1] != 1:
        raise ValueError("reference needs matching scalar 3D crops")
    if len(window) != 3 or any(width < 1 or width % 2 != 1 for width in window):
        raise ValueError("reference needs three positive odd window widths")
    if first.numel() * math.prod(window) > 2_000_000:
        raise ValueError("explicit NCC reference is restricted to small crops")
    centered = []
    for value in (first, second):
        padded = F.pad(value.double(), [w // 2 for w in reversed(window) for _ in range(2)])
        for dimension, width in enumerate(window, start=2):
            padded = padded.unfold(dimension, width, 1)
        centered.append(padded - padded.mean(dim=(-3, -2, -1), keepdim=True))
    first_centered, second_centered = centered
    cross = (first_centered * second_centered).sum(dim=(-3, -2, -1))
    var_first = first_centered.square().sum(dim=(-3, -2, -1)).clamp_min(NCC_VARIANCE_FLOOR)
    var_second = second_centered.square().sum(dim=(-3, -2, -1)).clamp_min(NCC_VARIANCE_FLOOR)
    return -(cross.square() / (var_first * var_second)).mean()


def audit_ncc_crop(first, second, window, *, device="cpu"):
    tolerances = COMPARISON_TOLERANCES["ncc_crop"]
    first = first.detach().to(device=device, dtype=torch.float32).clone().requires_grad_()
    second = second.detach().to(device=device, dtype=torch.float32).clone().requires_grad_()
    actual = ControllerNCC(win=window)(first, second)
    reference_first = first.detach().cpu().clone().requires_grad_()
    reference_second = second.detach().cpu().clone().requires_grad_()
    expected = centered_ncc_reference(reference_first, reference_second, window)
    actual_gradients = torch.autograd.grad(actual, (first, second))
    expected_gradients = torch.autograd.grad(expected, (reference_first, reference_second))
    comparisons = [
        gradient_difference(a, r, **tolerances["gradient"])
        for a, r in zip(actual_gradients, expected_gradients, strict=True)
    ]
    value_error = abs(float(actual.detach()) - float(expected.detach()))
    passed = value_error <= tolerances["value_absolute"] and all(
        item["within_heuristic_tolerance"] for item in comparisons
    )
    return {
        "status": "PASS" if passed else "FAIL",
        "tolerances": copy.deepcopy(tolerances),
        "production_device": str(device),
        "shape": list(first.shape),
        "production_value": _finite_float(actual.detach()),
        "centered_reference_value": _finite_float(expected.detach()),
        "value_absolute_error": value_error,
        "gradient_comparisons": comparisons,
    }


def _selected_crops(first, second, window):
    # A bounded deterministic 3x3x3 grid also selects a real low-variance crop.
    widths = tuple(min(size, width + 2) for size, width in zip(first.shape[-3:], window, strict=True))
    starts = [
        sorted({0, (size - width) // 2, size - width}) for size, width in zip(first.shape[-3:], widths, strict=True)
    ]
    candidates = []
    for start in product(*starts):
        slices = tuple(slice(s, s + w) for s, w in zip(start, widths, strict=True))
        pair = tuple(
            value[(slice(0, 1), slice(0, 1), *slices)].detach().float().cpu().clone() for value in (first, second)
        )
        variance = sum(float(value.double().var(unbiased=False)) for value in pair)
        candidates.append((start, pair, variance))
    selected = {
        0,
        len(candidates) // 2,
        len(candidates) - 1,
        min(range(len(candidates)), key=lambda i: candidates[i][2]),
    }
    for index in sorted(selected):
        start, pair, variance = candidates[index]
        yield (
            {
                "origin": list(start),
                "variance_sum": variance,
                "selection": "deterministic_grid_including_lowest_variance",
            },
            pair,
        )


class _NCCCapture:
    def __init__(self):
        self.crops = []
        self.calls = 0

    def __enter__(self):
        original = ControllerNCC.forward

        def observe(module, first, second):
            self.calls += 1
            for metadata, pair in _selected_crops(first, second, module.window):
                self.crops.append(({"direction_call": self.calls, **metadata}, pair, module.window))
            return original(module, first, second)

        self.patch = patch.object(ControllerNCC, "forward", observe)
        self.patch.start()
        return self

    def __exit__(self, *_args):
        self.patch.stop()


def _output_tensors(output, label=""):
    if isinstance(output, torch.Tensor):
        yield label, output
    elif is_dataclass(output):
        for field in fields(output):
            yield from _output_tensors(getattr(output, field.name), f"{label}.{field.name}")
    elif isinstance(output, (list, tuple)):
        for index, value in enumerate(output):
            yield from _output_tensors(value, f"{label}.{index}")


class GradientTrace:
    """Observe tensors without altering gradients; include functional upsampling.

    The first *observed* bad tensor locates a boundary, not necessarily the root
    cause. CUDA backward scheduling and hook coverage constrain that inference.
    """

    def __init__(self, model, scale):
        self.model, self.scale = model, scale
        self.events, self.handles = [], []
        self.calls = defaultdict(int)
        self.delta_values, self.delta_gradients = {}, {}

    def _observe(self, label, tensor):
        self.events.append({"stage": "forward", "tensor": label, **tensor_stats(tensor)})
        if label.endswith(".requested_delta"):
            self.delta_values[label] = tensor.detach().float().cpu().clone()
        if not tensor.requires_grad:
            return

        def backward(gradient):
            self.events.append({"stage": "scaled_backward", "tensor": label, **tensor_stats(gradient)})
            if label.endswith(".requested_delta"):
                self.delta_gradients[label] = gradient.detach().float().cpu() / self.scale

        self.handles.append(tensor.register_hook(backward))

    def __enter__(self):
        def module_hook(name):
            def observe(_module, _args, output):
                self.calls[name] += 1
                for suffix, value in _output_tensors(output):
                    self._observe(f"{name}#{self.calls[name]}{suffix}", value)

            return observe

        for name, module in self.model.named_modules():
            if name and any(module.children()):
                continue
            self.handles.append(module.register_forward_hook(module_hook(name or "controller")))
        original = F.interpolate

        def interpolate(value, *args, **kwargs):
            self.calls["functional.interpolate"] += 1
            label = f"functional.interpolate#{self.calls['functional.interpolate']}"
            self._observe(f"{label}.input", value)
            output = original(value, *args, **kwargs)
            self._observe(f"{label}.output", output)
            return output

        self.patch = patch.object(F, "interpolate", interpolate)
        self.patch.start()
        return self

    def __exit__(self, *_args):
        self.patch.stop()
        for handle in self.handles:
            handle.remove()


def _run_probe(step, inputs, *, fp32, scale):
    step.optimizer.zero_grad(set_to_none=True)
    result = {"mode": "fp32_strict" if fp32 else "fp16", "scale": scale}
    started = time.monotonic()
    gradients = {}
    with strict_fp32(fp32), GradientTrace(step.controller, scale) as trace:
        result["backend_flags"] = backend_flags()
        try:
            loss, logs = runtime._controller_pair_loss(step, inputs, diagnostic_fp32=fp32)
            result["metrics"] = {name: _finite_float(value) for name, value in logs.items()}
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("non-finite objective")
            # Scaling is explicit so no GradScaler or optimizer state is touched.
            (loss * scale).backward()
            gradients = {
                name: parameter.grad.detach().float().cpu() / scale
                for name, parameter in step.controller.named_parameters()
                if parameter.grad is not None
            }
            result["missing_parameter_gradients"] = [
                name
                for name, parameter in step.controller.named_parameters()
                if parameter.requires_grad and parameter.grad is None
            ]
            result["parameter_gradient_stats"] = {name: tensor_stats(value) for name, value in gradients.items()}
            nonfinite = any(item["nonfinite"] for item in result["parameter_gradient_stats"].values())
            nonfinite = nonfinite or any(event["nonfinite"] for event in trace.events)
            result["status"] = "NONFINITE_GRADIENT" if nonfinite else "FINITE"
            if result["missing_parameter_gradients"]:
                result["status"] = "MISSING_GRADIENT"
        except Exception as exc:
            result.update(status=_error_status(exc), exception_type=type(exc).__name__, error=str(exc))
        finally:
            result["trace"] = trace.events
            result["first_observed_nonfinite"] = next((event for event in trace.events if event["nonfinite"]), None)
            result["instrumented_seconds_not_a_benchmark"] = time.monotonic() - started
    return result, gradients, trace.delta_values, trace.delta_gradients


def _component_audit(step, inputs, deltas):
    if len(deltas) != 2:
        return {"status": "ERROR", "error": "strict FP32 probe did not produce both requested deltas"}
    delta_ab, delta_ba = (value.to(step.device).requires_grad_() for value in deltas.values())
    captured = {}
    original = losses._require_finite

    def capture(name, value):
        captured[name] = value
        return original(name, value)

    with strict_fp32(), patch.object(losses, "_require_finite", capture):
        total, logs = losses.controller_objective(
            inputs.tensors_ab[3],
            inputs.tensors_ab[4],
            inputs.tensors_ba[3],
            inputs.tensors_ba[4],
            inputs.psi_ab,
            inputs.psi_ba,
            delta_ab,
            delta_ba,
            config=step.config.loss,
        )
        total_gradients = torch.autograd.grad(total, (delta_ab, delta_ba), retain_graph=True)
        weighted = [torch.zeros_like(value) for value in total_gradients]
        terms = {}
        weights = {
            "ncc": step.config.loss.ncc_weight,
            "diffusion": step.config.loss.diffusion_weight,
            "inverse_consistency": step.config.loss.inverse_consistency_weight,
            "magnitude": step.config.loss.magnitude_weight,
        }
        for index, (name, weight) in enumerate(weights.items()):
            gradients = torch.autograd.grad(captured[name], (delta_ab, delta_ba), retain_graph=index < 3)
            terms[name] = {
                "weight": weight,
                "value": logs[name],
                "delta_gradients": [tensor_stats(value) for value in gradients],
            }
            for target, gradient in zip(weighted, gradients, strict=True):
                target.add_(gradient, alpha=weight)
        comparisons = [
            gradient_difference(a, b, **COMPARISON_TOLERANCES["component_gradient_sum"])
            for a, b in zip(weighted, total_gradients, strict=True)
        ]
    finite = all(stats["nonfinite"] == 0 for term in terms.values() for stats in term["delta_gradients"])
    return {
        "status": "PASS" if finite and all(item["within_heuristic_tolerance"] for item in comparisons) else "FAIL",
        "tolerances": copy.deepcopy(COMPARISON_TOLERANCES["component_gradient_sum"]),
        "terms": terms,
        "weighted_gradient_sum_vs_total": comparisons,
        "scope": "component localization and gradient linearity on identical FP32 deltas; not an independent derivative oracle for every operator",
    }


def _state_equal(first, second):
    if isinstance(first, torch.Tensor):
        return isinstance(second, torch.Tensor) and torch.equal(first.cpu(), second.cpu())
    if isinstance(first, np.ndarray):
        return isinstance(second, np.ndarray) and np.array_equal(first, second)
    if isinstance(first, dict):
        return (
            isinstance(second, dict)
            and first.keys() == second.keys()
            and all(_state_equal(first[key], second[key]) for key in first)
        )
    if isinstance(first, (list, tuple)):
        return (
            isinstance(second, type(first))
            and len(first) == len(second)
            and all(_state_equal(a, b) for a, b in zip(first, second, strict=True))
        )
    return first == second


def _compare_named(actual, reference):
    if not reference or actual.keys() != reference.keys():
        return {"status": "UNAVAILABLE", "within_heuristic_tolerance": False}
    return gradient_difference(
        torch.cat([actual[name].flatten() for name in sorted(actual)]),
        torch.cat([reference[name].flatten() for name in sorted(reference)]),
        **COMPARISON_TOLERANCES["mixed_precision_gradient"],
    )


def _ncc_audit_outcome(cases, captured_calls, reference_status):
    """Preserve measured mismatches and operator guards despite missing evidence."""
    errors = [case["status"] for case in cases if case["status"] not in {"PASS", "FAIL", "MATH_ERROR"}]
    if reference_status in {"OOM", "ERROR", "MATH_ERROR"}:
        errors.append(reference_status)
    if captured_calls != 2:
        errors.append("INCOMPLETE")
    if any(case["status"] in {"FAIL", "MATH_ERROR"} for case in cases):
        # A measured mismatch remains evidence even when another crop is absent.
        status = "FAIL"
    else:
        status = "ERROR" if errors else "PASS"
    result = {"status": status, "complete": not errors}
    if errors:
        result["error_kind"] = "OOM" if "OOM" in errors else errors[0]
        result["incomplete_reasons"] = errors
    return result


def compare_pair(step, inputs, *, save=None):
    """Compare identical prepared inputs and starting state without any update.

    ``save`` receives incremental JSON-safe reports. State is restored even on
    an exception. CUDA algorithms may remain nondeterministic; this does not
    claim bitwise replay of the historical trajectory.
    """
    rng = capture_rng_state()
    model_state = copy.deepcopy(step.controller.state_dict())
    optimizer_state = copy.deepcopy(step.optimizer.state_dict())
    scaler_state = copy.deepcopy(step.scaler.state_dict()) if getattr(step, "scaler", None) is not None else None
    previous_grads = {
        name: None if parameter.grad is None else parameter.grad.detach().clone()
        for name, parameter in step.controller.named_parameters()
    }
    input_tensors = (inputs.psi_ab, inputs.psi_ba, *inputs.tensors_ab, *inputs.tensors_ba)
    input_versions = [value._version for value in input_tensors]
    model_hash = state_dict_sha256(model_state)
    report = {
        "status": "RUNNING",
        "training_validated": False,
        "probes": [],
        "ncc_contract": controller_ncc_contract(),
        "initial_model_sha256": model_hash,
        "comparison_tolerances": copy.deepcopy(COMPARISON_TOLERANCES),
        "ncc_audit": {"status": "PENDING"},
        "component_gradient_audit": {"status": "PENDING"},
    }
    reference_gradients, reference_deltas, reference_delta_gradients = {}, {}, {}
    capture = _NCCCapture()
    try:
        for fp32, scale in [(True, 1.0), *((False, scale) for scale in PROBE_SCALES)]:
            restore_rng_state(rng)
            if fp32:
                with capture:
                    probe, gradients, deltas, delta_gradients = _run_probe(step, inputs, fp32=True, scale=scale)
                reference_gradients, reference_deltas, reference_delta_gradients = gradients, deltas, delta_gradients
            else:
                probe, gradients, _deltas, delta_gradients = _run_probe(step, inputs, fp32=False, scale=scale)
            probe["parameter_gradient_comparison"] = _compare_named(gradients, reference_gradients)
            probe["requested_delta_gradient_comparison"] = _compare_named(delta_gradients, reference_delta_gradients)
            probe["requested_delta_gradient_comparison_by_direction"] = {
                name: gradient_difference(
                    value, reference_delta_gradients[name], **COMPARISON_TOLERANCES["mixed_precision_gradient"]
                )
                for name, value in delta_gradients.items()
                if name in reference_delta_gradients
            }
            report["probes"].append(probe)
            if not _state_equal(model_state, step.controller.state_dict()) or not _state_equal(
                optimizer_state, step.optimizer.state_dict()
            ):
                raise RuntimeError("diagnostic probe unexpectedly mutated model or optimizer state")
            if save:
                save(report)
            step.optimizer.zero_grad(set_to_none=True)
            if probe["status"] == "OOM" and torch.cuda.is_available():
                torch.cuda.empty_cache()
        cases = []
        for metadata, pair, window in capture.crops:
            try:
                cases.append({**metadata, **audit_ncc_crop(*pair, window, device=step.device)})
            except Exception as exc:
                cases.append({**metadata, "status": _error_status(exc), "error": str(exc)})
        # Include controlled low-variance inputs even when no real crop is flat.
        window = (step.config.loss.ncc_window,) * 3
        base = torch.full((1, 1, 9, 9, 9), -1.23)
        perturbation = torch.linspace(-1e-4, 1e-4, base.numel()).reshape(base.shape)
        for name, first, second in (
            ("constant", base, base * 1.1),
            ("low_variance", base + perturbation, base * 1.1 - perturbation),
        ):
            try:
                cases.append({"selection": name, **audit_ncc_crop(first, second, window, device=step.device)})
            except Exception as exc:
                cases.append({"selection": name, "status": _error_status(exc), "error": str(exc)})
        report["ncc_audit"] = {
            **_ncc_audit_outcome(cases, capture.calls, report["probes"][0]["status"]),
            "captured_calls": capture.calls,
            "cases": cases,
            "scope": "production-device NCC versus independent centered CPU FP64 value/gradient on captured FP32 crop bytes, with crop-local zero padding; not full-volume backward equivalence",
        }
        if save:
            save(report)
        try:
            report["component_gradient_audit"] = _component_audit(step, inputs, reference_deltas)
        except Exception as exc:
            report["component_gradient_audit"] = {
                "status": "ERROR",
                "error_kind": _error_status(exc),
                "error": str(exc),
            }
        component_audit = report["component_gradient_audit"]
        if component_audit["status"] == "ERROR" and len(reference_deltas) != 2:
            component_audit["error_kind"] = report["probes"][0]["status"]
        completed = (
            report["ncc_audit"]["complete"]
            and component_audit["status"] != "ERROR"
            and all(probe["status"] not in {"OOM", "ERROR"} for probe in report["probes"])
        )
        report["status"] = "COMPLETE" if completed else "INCOMPLETE"
    finally:
        report["state_preserved"] = {
            "model": _state_equal(model_state, step.controller.state_dict()),
            "optimizer": _state_equal(optimizer_state, step.optimizer.state_dict()),
            "scaler": scaler_state is None or _state_equal(scaler_state, step.scaler.state_dict()),
            "prepared_input_versions": input_versions == [value._version for value in input_tensors],
        }
        step.controller.load_state_dict(model_state, strict=True)
        step.optimizer.load_state_dict(optimizer_state)
        if scaler_state is not None:
            step.scaler.load_state_dict(scaler_state)
        for name, parameter in step.controller.named_parameters():
            parameter.grad = previous_grads[name]
        restore_rng_state(rng)
        report["state_preserved"]["rng_restored"] = _state_equal(rng, capture_rng_state())
        if not all(report["state_preserved"].values()):
            report["status"] = "STATE_PRESERVATION_ERROR"
        if save:
            save(report)
    if report["status"] == "STATE_PRESERVATION_ERROR":
        raise RuntimeError("precision diagnostic state preservation failed")
    return report


__all__ = ["compare_pair", "precision_context", "strict_fp32"]
