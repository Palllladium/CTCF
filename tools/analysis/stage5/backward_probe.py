"""Non-updating forward/backward probe for saved numerical regressions.

The richer pair comparison is in precision_probes; this small callback-based
probe also accepts synthetic CPU failures without a Stage5 data store.
"""

from __future__ import annotations

import math
import traceback
from collections import defaultdict
from dataclasses import fields, is_dataclass

import torch


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
            with torch.autocast(device_type=device_type, dtype=torch.float16, enabled=not fp32):
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
