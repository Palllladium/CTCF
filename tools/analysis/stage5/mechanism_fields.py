"""Separate evaluation-point shifts from arithmetic changes in field gradients.

No optimizer step or production operator replacement is performed. Comparisons
are observations on one saved pair, not a derivative oracle or training policy.
"""

from __future__ import annotations

import copy
from contextlib import contextmanager

import torch

from experiments.stage5 import losses
from experiments.stage5.checkpoints import capture_rng_state, restore_rng_state, state_dict_sha256
from tools.analysis.stage5.mechanism_contract import FIELD_MODES as MODES
from tools.analysis.stage5.precision_contract import error_status
from tools.analysis.stage5.precision_probes import (
    _finite_float,
    _state_equal,
    backend_flags,
    gradient_difference,
    tensor_stats,
)

FIELD_TOLERANCES = {"relative": 2e-4, "absolute": 1e-7, "cosine_minimum": 0.999}
DIRECTIONS = ("forward", "reverse")


@contextmanager
def _arithmetic(*, autocast, tf32, device):
    previous = backend_flags()
    try:
        torch.set_float32_matmul_precision("highest")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = tf32
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=autocast):
            yield
    finally:
        torch.set_float32_matmul_precision(previous["float32_matmul_precision"])
        torch.backends.cuda.matmul.allow_tf32 = previous["matmul_allow_tf32"]
        torch.backends.cudnn.allow_tf32 = previous["cudnn_allow_tf32"]


def _pair_comparison(actual, reference):
    return {
        direction: gradient_difference(first, second, **FIELD_TOLERANCES)
        for direction, first, second in zip(DIRECTIONS, actual, reference, strict=True)
    }


def _capture_deltas(step, inputs, *, autocast, tf32):
    fields = []
    with torch.no_grad(), _arithmetic(autocast=autocast, tf32=tf32, device=step.device):
        flags = backend_flags()
        for tensors in (inputs.tensors_ab, inputs.tensors_ba):
            output = step.controller(tensors[0], step.variant, s2_proposal=tensors[1], s4_proposal=tensors[2])
            fields.append(output.requested_delta.detach().float().cpu().clone())
    return tuple(fields), flags


def _objective_gradients(step, inputs, fields, *, autocast, tf32):
    leaves = tuple(field.to(device=step.device, dtype=torch.float32).clone().requires_grad_() for field in fields)
    with _arithmetic(autocast=autocast, tf32=tf32, device=step.device):
        flags = backend_flags()
        total, logs = losses.controller_objective(
            inputs.tensors_ab[3],
            inputs.tensors_ab[4],
            inputs.tensors_ba[3],
            inputs.tensors_ba[4],
            inputs.psi_ab,
            inputs.psi_ba,
            *leaves,
            config=step.config.loss,
        )
        gradients = torch.autograd.grad(total, leaves)
    gradients = tuple(value.detach().float().cpu() for value in gradients)
    stats = {direction: tensor_stats(value) for direction, value in zip(DIRECTIONS, gradients, strict=True)}
    finite = bool(torch.isfinite(total)) and all(value["nonfinite"] == 0 for value in stats.values())
    return {
        "status": "FINITE" if finite else "NONFINITE",
        "backend_flags": flags,
        "outer_autocast": autocast,
        "leaf_dtype": "torch.float32",
        "metrics": {name: _finite_float(value) for name, value in logs.items()},
        "gradient_stats": stats,
    }, gradients


def _failure(exc):
    return {"status": error_status(exc), "exception_type": type(exc).__name__, "error": str(exc)}


def run_field_audit(step, inputs, *, save=None):
    """Return JSON-safe, incremental observations while preserving caller state.

    ``save(report)`` can write the report after each completed evaluation. TF32
    flags indicate permission, not proof a particular kernel used TF32. The
    production objective owns its inner autocast-disabled region; outer FP16 is
    varied to verify that boundary on identical leaf values. CUDA repeat noise
    is recorded, never subtracted from errors or declared an exact bound.
    """
    device = torch.device(step.device)
    if device.type != "cuda":
        raise ValueError("field precision audit requires a CUDA device")
    rng = capture_rng_state()
    original_flags = backend_flags()
    state = {name: value.detach().cpu().clone() for name, value in step.controller.state_dict().items()}
    train_flags = {name: module.training for name, module in step.controller.named_modules()}
    gradients_before = {
        name: None if parameter.grad is None else parameter.grad.detach().cpu().clone()
        for name, parameter in step.controller.named_parameters()
    }
    input_tensors = (inputs.psi_ab, inputs.psi_ba, *inputs.tensors_ab, *inputs.tensors_ba)
    input_versions = [value._version for value in input_tensors]
    model_unchanged = True
    train_flags_unchanged = True
    report = {
        "status": "RUNNING",
        "training_validated": False,
        "initial_model_sha256": state_dict_sha256(state),
        "tolerances": copy.deepcopy(FIELD_TOLERANCES),
        "tolerance_scope": "descriptive diagnostic screening only; not an acceptance rule for training",
        "controller_outputs": {},
        "field_points": {},
        "interpretation": {
            "evaluation_point_effect": "Different controller fields evaluated with the same strict objective arithmetic; a difference is not evidence of an incorrect derivative.",
            "same_point_arithmetic_effect": "Identical FP32 leaf bytes, with only outer autocast and cuDNN TF32 permission varied.",
            "repeatability": "One strict repeat per field point measures observed repeat noise, not a rigorous nondeterminism bound.",
            "tf32": "Only cuDNN TF32 permission is varied; matrix multiplication TF32 is always disabled, so a matmul-TF32 effect cannot be isolated. Permission does not prove a kernel selected TF32.",
            "independent_derivative_oracle": "NOT_PERFORMED: piecewise trilinear warps can change derivative at sampling cell boundaries; this audit isolates causes of disagreement without claiming a global derivative proof.",
        },
    }
    fields_by_mode = {}
    common_gradients = None

    def emit():
        if save is not None:
            save(report)

    try:
        for name, autocast, tf32 in MODES:
            restore_rng_state(rng)
            step.controller.load_state_dict(state, strict=True)
            for module_name, module in step.controller.named_modules():
                module.training = train_flags[module_name]
            try:
                fields, flags = _capture_deltas(step, inputs, autocast=autocast, tf32=tf32)
                fields_by_mode[name] = fields
                stats = {direction: tensor_stats(value) for direction, value in zip(DIRECTIONS, fields, strict=True)}
                report["controller_outputs"][name] = {
                    "status": "FINITE" if all(value["nonfinite"] == 0 for value in stats.values()) else "NONFINITE",
                    "backend_flags": flags,
                    "autocast": autocast,
                    "field_stats": stats,
                }
            except Exception as exc:
                report["controller_outputs"][name] = _failure(exc)
            model_unchanged &= _state_equal(state, step.controller.state_dict())
            train_flags_unchanged &= train_flags == {
                module_name: module.training for module_name, module in step.controller.named_modules()
            }
            emit()

        common_fields = fields_by_mode.get(MODES[0][0])
        for point_name, fields in fields_by_mode.items():
            point = {"evaluations": {}, "repeat": {"status": "PENDING"}}
            report["field_points"][point_name] = point
            if common_fields is not None:
                point["field_shift_vs_strict_controller"] = _pair_comparison(fields, common_fields)
            point_gradients = None
            for mode_name, autocast, tf32 in MODES:
                restore_rng_state(rng)
                try:
                    observation, gradients = _objective_gradients(step, inputs, fields, autocast=autocast, tf32=tf32)
                    if mode_name == MODES[0][0]:
                        point_gradients = gradients
                        if point_name == MODES[0][0]:
                            common_gradients = gradients
                        if common_gradients is not None:
                            point["evaluation_point_effect"] = _pair_comparison(gradients, common_gradients)
                    if point_gradients is not None:
                        observation["same_point_arithmetic_effect"] = _pair_comparison(gradients, point_gradients)
                    point["evaluations"][mode_name] = observation
                    del gradients
                except Exception as exc:
                    point["evaluations"][mode_name] = _failure(exc)
                emit()
            restore_rng_state(rng)
            try:
                repeat, gradients = _objective_gradients(step, inputs, fields, autocast=False, tf32=False)
                if point_gradients is not None:
                    repeat["same_point_repeat_difference"] = _pair_comparison(gradients, point_gradients)
                point["repeat"] = repeat
                del gradients
            except Exception as exc:
                point["repeat"] = _failure(exc)
            emit()
        statuses = [value["status"] for value in report["controller_outputs"].values()]
        for point in report["field_points"].values():
            statuses.extend(value["status"] for value in point["evaluations"].values())
            statuses.append(point["repeat"]["status"])
        complete = len(fields_by_mode) == len(MODES) and all(
            value in {"FINITE", "NONFINITE", "MATH_ERROR"} for value in statuses
        )
        report["status"] = "COMPLETE" if complete else "INCOMPLETE"
    finally:
        preserved = {
            "model_during_audit": model_unchanged and _state_equal(state, step.controller.state_dict()),
            "training_flags_during_audit": train_flags_unchanged,
            "input_versions": input_versions == [value._version for value in input_tensors],
            "parameter_gradients": all(
                _state_equal(gradients_before[name], parameter.grad)
                for name, parameter in step.controller.named_parameters()
            ),
            "backend_flags": original_flags == backend_flags(),
        }
        step.controller.load_state_dict(state, strict=True)
        for name, module in step.controller.named_modules():
            module.training = train_flags[name]
        for name, parameter in step.controller.named_parameters():
            previous = gradients_before[name]
            parameter.grad = None if previous is None else previous.to(parameter.device)
        restore_rng_state(rng)
        preserved["rng_restored"] = _state_equal(rng, capture_rng_state())
        preserved["training_flags_restored"] = train_flags == {
            name: module.training for name, module in step.controller.named_modules()
        }
        report["state_preserved"] = preserved
        if not all(preserved.values()):
            report["status"] = "STATE_PRESERVATION_ERROR"
        emit()
    return report
