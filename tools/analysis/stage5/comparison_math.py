"""Numerical observations outside timing; no heuristic quality acceptance gate."""

from __future__ import annotations

import math
from contextlib import AbstractContextManager

import torch

from experiments.stage5.checkpoints import capture_rng_state, restore_rng_state
from experiments.stage5.failures import cpu_snapshot, require_finite_optimizer, require_finite_parameters
from experiments.stage5.precision import precision_context, precision_mode_contract


def snapshot(step):
    return {
        "model": cpu_snapshot(step.controller.state_dict()),
        "optimizer": cpu_snapshot(step.optimizer.state_dict()),
        "rng": capture_rng_state(),
    }


def restore(step, state):
    step.controller.load_state_dict(state["model"], strict=True)
    step.optimizer.load_state_dict(cpu_snapshot(state["optimizer"]))
    step.optimizer.zero_grad(set_to_none=True)
    restore_rng_state(state["rng"])


def tensor_difference(actual, reference):
    """Chunked FP64 reductions on CPU, with explicit zero-reference semantics."""
    if actual.shape != reference.shape:
        raise ValueError("Comparison tensor shapes differ")
    actual = actual.detach().cpu().reshape(-1)
    reference = reference.detach().cpu().reshape(-1)
    error2 = reference2 = actual2 = dot = maximum = 0.0
    for first, second in zip(actual.split(1_000_000), reference.split(1_000_000), strict=True):
        first, second = first.double(), second.double()
        if not bool(torch.isfinite(first).all() and torch.isfinite(second).all()):
            return {"status": "NONFINITE"}
        difference = first - second
        error2 += float(difference.square().sum())
        reference2 += float(second.square().sum())
        actual2 += float(first.square().sum())
        dot += float((first * second).sum())
        if difference.numel():
            maximum = max(maximum, float(difference.abs().max()))
    absolute = math.sqrt(error2)
    norm, actual_norm = math.sqrt(reference2), math.sqrt(actual2)
    return {
        "status": "FINITE",
        "absolute_l2": absolute,
        "reference_l2": norm,
        "actual_l2": actual_norm,
        "relative_l2": absolute / norm if norm else None,
        "reference_zero": norm == 0,
        "rms": absolute / math.sqrt(actual.numel()) if actual.numel() else 0.0,
        "max_abs": maximum,
        "cosine": max(-1.0, min(1.0, dot / (norm * actual_norm))) if norm and actual_norm else None,
    }


def mapping_difference(actual, reference):
    if set(actual) != set(reference):
        raise ValueError("Compared parameter inventories differ")
    per_tensor = {name: tensor_difference(actual[name], reference[name]) for name in sorted(actual)}
    # Parameter vectors are small; full-volume fields use tensor_difference directly.
    joined_actual = torch.cat([actual[name].detach().cpu().reshape(-1) for name in sorted(actual)])
    joined_reference = torch.cat([reference[name].detach().cpu().reshape(-1) for name in sorted(reference)])
    return {"global": tensor_difference(joined_actual, joined_reference), "per_tensor": per_tensor}


class ArithmeticObserver(AbstractContextManager):
    """Observe actual convolution dtypes and optional field/field-gradient bytes."""

    def __init__(self, model, *, capture_fields=False):
        self.model = model
        self.capture_fields = capture_fields
        self.handles = []
        self.convolution_dtypes = set()
        self.fields, self.field_gradients = {}, {}
        self.forward_calls = 0

    def __enter__(self):
        def convolution(_module, _inputs, output):
            self.convolution_dtypes.add(str(output.dtype))

        def controller(_module, _inputs, output):
            direction = ("forward", "reverse")[self.forward_calls]
            self.forward_calls += 1
            if self.capture_fields:
                value = output.requested_delta
                self.fields[direction] = value.detach().float().cpu().clone()

                def gradient_hook(gradient):
                    self.field_gradients[direction] = gradient.detach().float().cpu().clone()

                value.register_hook(gradient_hook)

        for module in self.model.modules():
            if isinstance(module, torch.nn.Conv3d):
                self.handles.append(module.register_forward_hook(convolution))
        self.handles.append(self.model.register_forward_hook(controller))
        return self

    def __exit__(self, *_args):
        for handle in self.handles:
            handle.remove()

    def validate(self, mode):
        expected = "torch.bfloat16" if mode == "bf16" else "torch.float32"
        if self.forward_calls != 2 or self.convolution_dtypes != {expected}:
            raise RuntimeError(f"Controller arithmetic differs from {mode}: {self.convolution_dtypes}")
        return {"convolution_output_dtypes": sorted(self.convolution_dtypes), "directional_forward_calls": 2}


def advance(step, inputs, mode, *, failure_state=None):
    """One diagnostic update with the production numerical checks and no scaler."""
    from experiments.stage5 import runtime

    if step.scaler is not None:
        raise ValueError("Comparison never uses GradScaler")
    state = failure_state if failure_state is not None else {}
    step.optimizer.zero_grad(set_to_none=True)
    with precision_context(step.device, mode):
        state["phase"] = "forward"
        require_finite_parameters(step.controller, gradients=False)
        loss, logs = runtime._controller_pair_loss(step, inputs)
        logs["bootstrap_digital_residual_percent"] = inputs.bootstrap_residual
        if not bool(torch.isfinite(loss)) or any(not math.isfinite(float(value)) for value in logs.values()):
            raise FloatingPointError("Non-finite comparison objective or metric")
        state["phase"] = "backward"
        loss.backward()
        require_finite_parameters(step.controller, gradients=True)
        # Match the production capture policy, including its small CPU snapshot cost.
        state["before_update"] = {
            "model": cpu_snapshot(step.controller.state_dict()),
            "optimizer": cpu_snapshot(step.optimizer.state_dict()),
        }
        state["phase"] = "optimizer_step"
        step.optimizer.step()
        require_finite_parameters(step.controller, gradients=False)
        require_finite_optimizer(step.optimizer)
        mode_contract = precision_mode_contract(mode)
        if torch.is_autocast_enabled("cuda") != mode_contract["autocast"]:
            raise RuntimeError("Unexpected autocast state during comparison")
        if bool(torch.backends.cudnn.allow_tf32) != mode_contract["cudnn_allow_tf32"]:
            raise RuntimeError("Unexpected cuDNN TF32 state during comparison")
        if bool(torch.backends.cuda.matmul.allow_tf32) != mode_contract["matmul_allow_tf32"]:
            raise RuntimeError("Unexpected matmul TF32 state during comparison")
    return logs
