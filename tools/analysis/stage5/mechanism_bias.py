"""Read-only probes at convolution bias and loss-scaling boundaries.

Full-volume gradients are summarized in bounded chunks. Only small deterministic
samples and channel sums survive a hook. A convolution autograd-node hook sees
the returned bias derivative, not the internal CUDA accumulator or kernel.
"""

from __future__ import annotations

import copy
import hashlib
import math
from collections import defaultdict
from contextlib import contextmanager
from unittest.mock import patch

import torch
import torch.nn.functional as F

from experiments.stage5 import runtime
from experiments.stage5.checkpoints import capture_rng_state, restore_rng_state
from tools.analysis.stage5.mechanism_contract import BIAS_COMPARISONS, bias_modes
from tools.analysis.stage5.precision_contract import error_status
from tools.analysis.stage5.precision_probes import (
    _output_tensors,
    _state_equal,
    backend_flags,
    gradient_difference,
    strict_fp32,
)

CHUNK_ELEMENTS = 262_144
SAMPLE_ELEMENTS = 8192


def _numbers(tensor):
    return [float(value) if math.isfinite(float(value)) else None for value in tensor.detach().cpu().flatten()]


def gradient_stats(value, scale=1.0):
    """Exact counts, bounded FP64 reductions, and unscaled finite norms."""
    flat = value.detach().reshape(-1)
    zero = nonfinite = subnormal = 0
    maximum = square = 0.0
    minimum = None
    tiny = torch.finfo(value.dtype).tiny
    for chunk in flat.split(CHUNK_ELEMENTS):
        finite = torch.isfinite(chunk)
        zero += int((chunk == 0).sum())
        nonfinite += int((~finite).sum())
        subnormal += int(((chunk.abs() < tiny) & (chunk != 0) & finite).sum())
        safe = chunk.double().masked_fill(~finite, 0).abs() / scale
        if safe.numel():
            maximum = max(maximum, float(safe.max()))
            square += float(safe.square().sum())
            positive = safe[safe > 0]
            if positive.numel():
                found = float(positive.min())
                minimum = found if minimum is None else min(minimum, found)
    return {
        "dtype": str(value.dtype),
        "shape": list(value.shape),
        "elements": value.numel(),
        "zero": zero,
        "nonfinite": nonfinite,
        "native_subnormal": subnormal,
        "unscaled_max_finite_abs": maximum,
        "unscaled_min_nonzero_finite_abs": minimum,
        "unscaled_finite_l2": math.sqrt(square),
    }


def channel_sum_reference(gradient):
    """Reduce N,D,H,W for each channel without a full-volume FP64 copy."""
    if gradient.ndim != 5:
        raise ValueError("bias audit requires an NCDHW convolution output gradient")
    sum32 = torch.zeros(gradient.shape[1], dtype=torch.float32, device=gradient.device)
    sum64 = torch.zeros(gradient.shape[1], dtype=torch.float64, device=gradient.device)
    for batch in gradient.detach():
        for channel, plane in enumerate(batch):
            for chunk in plane.reshape(-1).split(CHUNK_ELEMENTS):
                sum32[channel] += chunk.sum(dtype=torch.float32)
                sum64[channel] += chunk.sum(dtype=torch.float64)
    return sum32.cpu(), sum64.cpu()


def _sample(value, scale):
    flat = value.detach().reshape(-1)
    stride = max(1, math.ceil(flat.numel() / SAMPLE_ELEMENTS))
    return flat[::stride].float().cpu().clone() / scale


def _sample_comparison(actual, reference):
    if actual.shape != reference.shape:
        return {"status": "SHAPE_MISMATCH"}
    result = gradient_difference(actual, reference, absolute=1e-30)
    result.update(
        sample_elements=actual.numel(),
        actual_zero_reference_nonzero=int(((actual == 0) & (reference != 0) & torch.isfinite(reference)).sum()),
        actual_nonzero_reference_zero=int(((actual != 0) & (reference == 0) & torch.isfinite(actual)).sum()),
    )
    return result


class _BoundaryTrace:
    def __init__(self, model, scale):
        self.model, self.scale = model, scale
        self.handles, self.events, self.bias = [], [], []
        self.samples = {}
        self.forward_hashes = {}
        self.calls = defaultdict(int)

    def _observe(self, label, tensor):
        if label.endswith(".requested_delta"):
            digest = hashlib.sha256()
            for chunk in tensor.detach().reshape(-1).split(CHUNK_ELEMENTS):
                digest.update(chunk.contiguous().cpu().numpy().tobytes())
            self.forward_hashes[label] = {
                "dtype": str(tensor.dtype),
                "shape": list(tensor.shape),
                "sha256": digest.hexdigest(),
            }
        if not tensor.requires_grad:
            return

        def backward(gradient):
            self.events.append({"tensor": label, **gradient_stats(gradient, self.scale)})
            self.samples[label] = _sample(gradient, self.scale)

        self.handles.append(tensor.register_hook(backward))

    def _stem(self, output, direction):
        record = {"direction": direction, "node": type(output.grad_fn).__name__, "status": "PENDING"}
        self.bias.append(record)

        def upstream(gradient):
            sum32, sum64 = channel_sum_reference(gradient)
            record.update(
                upstream=gradient_stats(gradient, self.scale),
                sum_fp32=_numbers(sum32),
                sum_fp64=_numbers(sum64),
                sum_fp64_unscaled=_numbers(sum64 / self.scale),
                fp32_sum_vs_fp64=gradient_difference(sum32, sum64, absolute=1e-30),
                sum_fp64_then_half=_numbers(sum64.half()),
                sum_fp64_then_half_nonfinite=int((~torch.isfinite(sum64.half())).sum()),
                sum_exceeds_half_max=int((sum64.abs() > torch.finfo(torch.float16).max).sum()),
            )
            # Direct reduction requests a half result but does not establish its
            # internal accumulator precision. Compare it to cast-after-FP64.
            direct = gradient.detach().sum(dim=(0, 2, 3, 4), dtype=torch.float16)
            record["direct_half_result_sum"] = _numbers(direct)
            record["direct_half_result_sum_nonfinite"] = int((~torch.isfinite(direct)).sum())

        self.handles.append(output.register_hook(upstream))
        if type(output.grad_fn).__name__ != "ConvolutionBackward0":
            record.update(status="UNSUPPORTED_NODE", error="cannot identify bias slot safely")
            return

        def node_backward(gradient_inputs, _gradient_outputs):
            if len(gradient_inputs) != 3 or gradient_inputs[2] is None:
                record.update(status="MISSING_BIAS_GRADIENT")
                return
            actual = gradient_inputs[2].detach()
            record.update(
                status="CAPTURED",
                convolution_returned_bias=gradient_stats(actual, self.scale),
                convolution_returned_bias_values=_numbers(actual),
                convolution_returned_bias_unscaled=_numbers(actual.double() / self.scale),
            )
            expected = record.get("sum_fp64")
            if expected is not None and all(value is not None for value in expected):
                reference = torch.tensor(expected, dtype=torch.float64)
                record["returned_bias_vs_fp64_sum"] = gradient_difference(actual, reference, absolute=1e-30)

        self.handles.append(output.grad_fn.register_hook(node_backward))

    def __enter__(self):
        def observe(name):
            def hook(_module, _args, output):
                self.calls[name] += 1
                label = f"{name}#{self.calls[name]}"
                for suffix, value in _output_tensors(output):
                    self._observe(label + suffix, value)
                if name == "stem.0":
                    self._stem(output, "ab" if self.calls[name] == 1 else "ba")

            return hook

        for name, module in self.model.named_modules():
            if name and any(module.children()):
                continue
            self.handles.append(module.register_forward_hook(observe(name or "controller")))
        original = F.interpolate

        def interpolate(value, *args, **kwargs):
            self.calls["interpolate"] += 1
            label = f"interpolate#{self.calls['interpolate']}"
            self._observe(label + ".input", value)
            output = original(value, *args, **kwargs)
            self._observe(label + ".output", output)
            return output

        self.patch = patch.object(F, "interpolate", interpolate)
        self.patch.start()
        return self

    def __exit__(self, *_args):
        self.patch.stop()
        for handle in self.handles:
            handle.remove()


@contextmanager
def _preserve(step):
    model = copy.deepcopy(step.controller.state_dict())
    grads = {
        name: None if p.grad is None else p.grad.detach().clone() for name, p in step.controller.named_parameters()
    }
    rng = capture_rng_state()
    modes = {name: module.training for name, module in step.controller.named_modules()}
    optimizer = copy.deepcopy(step.optimizer.state_dict()) if getattr(step, "optimizer", None) is not None else None
    scaler = copy.deepcopy(step.scaler.state_dict()) if getattr(step, "scaler", None) is not None else None
    try:
        yield rng, model
    finally:
        step.controller.load_state_dict(model, strict=True)
        for name, parameter in step.controller.named_parameters():
            parameter.grad = grads[name]
        for name, module in step.controller.named_modules():
            module.training = modes[name]
        if optimizer is not None:
            step.optimizer.load_state_dict(optimizer)
        if scaler is not None:
            step.scaler.load_state_dict(scaler)
        restore_rng_state(rng)


def _probe(step, inputs, *, fp32, scale, disable_tf32):
    step.controller.zero_grad(set_to_none=True)
    record = {"mode": "fp32" if fp32 else "fp16", "scale": scale}
    parameters = {}
    with strict_fp32(disable_tf32), _BoundaryTrace(step.controller, scale) as trace:
        record["backend_flags"] = backend_flags()
        try:
            loss, logs = runtime._controller_pair_loss(step, inputs, diagnostic_fp32=fp32)
            record["metrics"] = {
                key: float(value) if math.isfinite(float(value)) else None for key, value in logs.items()
            }
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("non-finite objective")
            (loss * scale).backward()
            parameters = {
                name: parameter.grad.detach().float().cpu().clone() / scale
                for name, parameter in step.controller.named_parameters()
                if parameter.grad is not None
            }
            record["parameters"] = {name: gradient_stats(value) for name, value in parameters.items()}
            bias = dict(step.controller.named_parameters())["stem.0.bias"].grad
            record["accumulated_parameter_bias_scaled"] = None if bias is None else _numbers(bias)
            record["accumulated_parameter_bias_unscaled"] = None if bias is None else _numbers(bias.double() / scale)
            record["missing_parameters"] = [
                name for name, p in step.controller.named_parameters() if p.requires_grad and p.grad is None
            ]
            record["status"] = "CAPTURED"
        except Exception as exc:
            record.update(status=error_status(exc), exception_type=type(exc).__name__, error=str(exc))
        record["layer_gradients"] = trace.events
        record["bias_directions"] = trace.bias
        record["requested_delta_forward_hashes"] = trace.forward_hashes
        _accumulated_bias_comparison(record)
    return record, parameters, trace.samples


def _accumulated_bias_comparison(record):
    directions = record["bias_directions"]
    if len(directions) != 2:
        return
    sums = [direction.get("sum_fp64", []) for direction in directions]
    returned = [direction.get("convolution_returned_bias_values", []) for direction in directions]
    actual = record.get("accumulated_parameter_bias_scaled")
    if not all(sums) or any(value is None for row in sums for value in row):
        return
    sums = torch.tensor(sums, dtype=torch.float64)
    comparison = {
        "fp64_sum_of_direction_references": _numbers(sums.sum(0)),
        "each_reference_cast_half_then_accumulate_fp32": _numbers(sums.half().float().sum(0)),
        "each_reference_cast_half_then_accumulate_half": _numbers(sums.half().sum(0, dtype=torch.float16)),
        "scope": "cast/sum order candidates; these do not identify the internal accumulation implementation",
    }
    if actual is not None and all(value is not None for value in actual):
        comparison["actual_vs_fp64_direction_sum"] = gradient_difference(
            torch.tensor(actual, dtype=torch.float64), sums.sum(0), absolute=1e-30
        )
    if all(returned) and all(value is not None for row in returned for value in row):
        returned = torch.tensor(returned, dtype=torch.float64)
        comparison["returned_directions_accumulate_fp32"] = _numbers(returned.float().sum(0))
        comparison["returned_directions_accumulate_half"] = _numbers(returned.half().sum(0, dtype=torch.float16))
    record["bias_accumulation"] = comparison


def run_bias_audit(step, inputs, *, save=None):
    """Inspect exact saved inputs/state; never perform an optimizer update.

    Complete means measurements captured, not that gradients are finite or that
    a production precision regime has been validated.
    """
    first = dict(step.controller.named_modules()).get("stem.0")
    if not isinstance(first, torch.nn.Conv3d) or first.bias is None:
        raise ValueError("bias audit requires controller stem.0 Conv3d with bias")
    original_flags = backend_flags()
    inputs_all = (inputs.psi_ab, inputs.psi_ba, *inputs.tensors_ab, *inputs.tensors_ba)
    versions = [value._version for value in inputs_all]
    report = {
        "status": "RUNNING",
        "training_validated": False,
        "probes": {},
        "original_backend_flags": original_flags,
        "scope": "convolution backward return, cast-back and parameter accumulation boundaries; internal CUDA accumulator/kernel remains unobserved",
        "sampling": {
            "maximum_elements_per_tensor": SAMPLE_ELEMENTS,
            "selection": "deterministic flattened stride",
            "counts_and_channel_sums": "all_elements",
        },
        "interpretation": "Layer zero counts are exact; cross-mode zero loss is sampled. Different forward precision can change gradients. Scale1 versus scale32768 holds forward precision/backend fixed; CUDA nondeterminism may remain.",
    }
    plan = bias_modes(original_flags)
    stored = {}
    with _preserve(step) as (rng, model):
        for name, fp32, scale, disable_tf32 in plan:
            step.controller.load_state_dict(model, strict=True)
            restore_rng_state(rng)
            record, parameters, samples = _probe(step, inputs, fp32=fp32, scale=scale, disable_tf32=disable_tf32)
            report["probes"][name] = record
            stored[name] = (parameters, samples)
            if save:
                save(report)
            step.controller.zero_grad(set_to_none=True)
            if record["status"] == "OOM" and torch.cuda.is_available():
                torch.cuda.empty_cache()
        comparisons = {}
        for actual_name, reference_name in BIAS_COMPARISONS:
            if actual_name not in stored or reference_name not in stored:
                continue
            actual_parameters, actual_samples = stored[actual_name]
            reference_parameters, reference_samples = stored[reference_name]
            comparisons[f"{actual_name}_vs_{reference_name}"] = {
                "requested_delta_forward_bytes_equal": (
                    report["probes"][actual_name]["requested_delta_forward_hashes"]
                    == report["probes"][reference_name]["requested_delta_forward_hashes"]
                    if report["probes"][actual_name]["requested_delta_forward_hashes"]
                    and report["probes"][reference_name]["requested_delta_forward_hashes"]
                    else None
                ),
                "parameters_all_elements": {
                    name: _sample_comparison(actual_parameters[name], value)
                    for name, value in reference_parameters.items()
                    if name in actual_parameters
                },
                "layer_samples": {
                    name: _sample_comparison(actual_samples[name], value)
                    for name, value in reference_samples.items()
                    if name in actual_samples
                },
                "missing_layer_samples": sorted(reference_samples.keys() - actual_samples.keys()),
            }
        report["comparisons"] = comparisons
        complete = all(
            record["status"] == "CAPTURED"
            and not record["missing_parameters"]
            and len(record["bias_directions"]) == 2
            and all(direction["status"] == "CAPTURED" for direction in record["bias_directions"])
            for record in report["probes"].values()
        )
        report["status"] = "COMPLETE" if complete else "INCOMPLETE"
    report["state_restored"] = {
        "rng": _state_equal(rng, capture_rng_state()),
        "backend": original_flags == backend_flags(),
        "prepared_input_versions": versions == [value._version for value in inputs_all],
        "model": _state_equal(model, step.controller.state_dict()),
    }
    if not all(report["state_restored"].values()):
        report["status"] = "STATE_PRESERVATION_ERROR"
    if save:
        save(report)
    return report
