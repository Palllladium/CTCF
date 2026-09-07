"""Diagnostic-only NCC moments and FP64 reference; never imported by training."""

from __future__ import annotations

from contextlib import contextmanager
from functools import partial
from math import isfinite
from unittest.mock import patch

import torch
import torch.nn.functional as F

from experiments.stage5 import losses, runtime
from experiments.stage5.checkpoints import restore_rng_state, state_dict_sha256
from utils import NCCVxm


def _separable_sum(value, window):
    for dim, width in enumerate(window, start=2):
        padding = [0] * 6
        offset = 2 * (4 - dim)
        padding[offset : offset + 2] = [width // 2] * 2
        padded = F.pad(value, padding)
        result = torch.zeros_like(value)
        for shift in range(width):
            result.add_(padded.narrow(dim, shift, value.shape[dim]))
        value = result
    return value


class _BoxSum(torch.autograd.Function):
    """Symmetric zero-padded box operator; its adjoint is the same operator."""

    @staticmethod
    def forward(ctx, value, window):
        ctx.window = window
        return _separable_sum(value, window)

    @staticmethod
    def backward(ctx, gradient):
        return _BoxSum.apply(gradient, ctx.window), None


def ncc_moments(first, second, window, *, reference):
    """Keep legacy formula/epsilon; change only accumulation for the reference."""
    if first.ndim != 5 or first.shape[1] != 1 or first.shape != second.shape:
        raise ValueError("NCC diagnostic needs matching scalar 3D volumes")
    if len(window) != 3 or any(w <= 0 or w % 2 != 1 for w in window):
        raise ValueError("NCC diagnostic needs three positive odd window widths")
    if reference:
        first, second = first.double(), second.double()

        def box(value):
            return _BoxSum.apply(value, window)
    else:
        first, second = first.float(), second.float()
        kernel = torch.ones((1, 1, *window), device=first.device, dtype=first.dtype)
        box = partial(F.conv3d, weight=kernel, padding=tuple(w // 2 for w in window))
    first_sum, second_sum = box(first), box(second)
    first_squared, second_squared, product = box(first * first), box(second * second), box(first * second)
    size = float(window[0] * window[1] * window[2])
    mean_first, mean_second = first_sum / size, second_sum / size
    cross = product - mean_second * first_sum - mean_first * second_sum + mean_first * mean_second * size
    var_first = first_squared - 2 * mean_first * first_sum + mean_first * mean_first * size
    var_second = second_squared - 2 * mean_second * second_sum + mean_second * mean_second * size
    return {
        "sum_first": first_sum,
        "sum_second": second_sum,
        "sum_squared_first": first_squared,
        "sum_squared_second": second_squared,
        "sum_product": product,
        "cross": cross,
        "raw_variance_first": var_first,
        "raw_variance_second": var_second,
    }


class ReferenceNCC(torch.nn.Module):
    """FP64 moments on unchanged FP32 image bytes, FP32 scalar for the objective."""

    def __init__(self, win, eps=1e-5):
        super().__init__()
        self.window, self.eps = tuple(win), eps

    def forward(self, first, second):
        with torch.autocast(device_type=first.device.type, enabled=False):
            moments = ncc_moments(first, second, self.window, reference=True)
            denominator = moments["raw_variance_first"].clamp_min(self.eps)
            denominator = denominator * moments["raw_variance_second"].clamp_min(self.eps)
            return -(moments["cross"].square() / denominator).mean().float()


@contextmanager
def ieee_convolutions(enabled):
    """Change TF32 only during NCC measurements, and restore on every exit."""
    previous = (torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32)
    try:
        if enabled:
            torch.backends.cudnn.allow_tf32 = False
            torch.backends.cuda.matmul.allow_tf32 = False
        yield
    finally:
        torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32 = previous


def distribution(value):
    value = value.detach()
    finite = torch.isfinite(value)
    selected = value[finite].double()
    return {
        "elements": value.numel(),
        "nonfinite": int((~finite).sum()),
        "min": float(selected.min()) if selected.numel() else None,
        "max": float(selected.max()) if selected.numel() else None,
        "mean": float(selected.mean()) if selected.numel() else None,
        "negative": int((value < 0).sum()),
    }


def finite_float(value):
    result = float(value)
    return result if isfinite(result) else None


def scalar_difference(first, second):
    return abs(first - second) if first is not None and second is not None else None


def _difference(first, second):
    difference = first.double() - second.double()
    finite = torch.isfinite(difference)
    safe = difference.masked_fill(~finite, 0)
    return {
        "nonfinite": int((~finite).sum()),
        "max_abs": float(safe.abs().max()),
        "rms": float(safe.square().mean().sqrt()),
    }


def centered_worst_window(first, second, cc, window):
    """Independent explicit patch centering at the largest reported correlation."""
    flat = int(torch.nan_to_num(cc.detach(), nan=-float("inf")).argmax())
    index = []
    for size in reversed(cc.shape):
        index.append(flat % size)
        flat //= size
    index = list(reversed(index))
    padding = [width // 2 for width in reversed(window) for _ in range(2)]
    slices = tuple(slice(i, i + width) for i, width in zip(index[-3:], window, strict=True))
    patches = []
    # Only these tiny patches are promoted; no full-volume unfold allocation.
    for value in (first, second):
        padded = F.pad(value.detach(), padding)
        patches.append(padded[index[0], index[1]][slices].double())
    x, y = patches
    centered_x, centered_y = x - x.mean(), y - y.mean()
    vi, vj = centered_x.square().sum(), centered_y.square().sum()
    cross = (centered_x * centered_y).sum()
    return {
        "index_bcdhw": index,
        "reported_cc": finite_float(cc.detach()[tuple(index)]),
        "first_patch": distribution(x),
        "second_patch": distribution(y),
        "centered_fp64_cross": float(cross),
        "centered_fp64_variance_first": float(vi),
        "centered_fp64_variance_second": float(vj),
        "centered_fp64_cc": float(cross.square() / (vi.clamp_min(1e-5) * vj.clamp_min(1e-5))),
    }


def inspect_ncc(first, second, window, *, reference=False, disable_tf32=False):
    """Measure the same warped tensors; no repeated registration or normalisation."""
    epsilon = 1e-5
    x = first.detach().float().clone().requires_grad_()
    y = second.detach().float()
    with torch.enable_grad(), torch.autocast(device_type=x.device.type, enabled=False), ieee_convolutions(disable_tf32):
        moments = ncc_moments(x, y, window, reference=reference)
        raw_i, raw_j = moments["raw_variance_first"], moments["raw_variance_second"]
        cc = moments["cross"].square() / (raw_i.clamp_min(epsilon) * raw_j.clamp_min(epsilon))
        loss = -cc.mean()
        gradient = torch.autograd.grad(loss, x)[0].detach()
        observed = finite_float(loss.detach())
        result = {
            "loss": observed,
            "loss_within_mathematical_range_1e_6": observed is not None and -1.000001 <= observed <= 0.000001,
            "moments": {name: distribution(value) for name, value in moments.items()},
            "cc": distribution(cc),
            "cc_above_one_plus_1e_6": int((cc > 1.000001).sum()),
            "variance_first_below_epsilon": int((raw_i < epsilon).sum()),
            "variance_second_below_epsilon": int((raw_j < epsilon).sum()),
            "gradient_wrt_warped_fp32": distribution(gradient),
            "gradient_max_abs": finite_float(gradient.abs().max()),
            "worst_window": centered_worst_window(x, y, cc, window),
        }
        if not reference:
            with torch.no_grad():
                actual = finite_float(NCCVxm(win=window)(x, y))
            result["production_loss"] = actual
            difference = scalar_difference(observed, actual)
            result["production_loss_absolute_difference"] = difference
            if difference is not None and difference > 1e-6 * max(1.0, abs(actual)):
                raise RuntimeError("Diagnostic moment formula differs from production NCC")
    return result, gradient


def compare_ncc(first, second, window):
    report = {"first": distribution(first), "second": distribution(second), "window": list(window)}
    reference, ref_gradient = inspect_ncc(first, second, window, reference=True)
    report["fp64_reference"] = reference
    for mode, disable_tf32 in (("fp32_backend_default", False), ("fp32_tf32_disabled", True)):
        measured, gradient = inspect_ncc(first, second, window, disable_tf32=disable_tf32)
        measured["gradient_difference_from_fp64"] = _difference(gradient, ref_gradient)
        measured["loss_absolute_difference_from_fp64"] = scalar_difference(measured["loss"], reference["loss"])
        report[mode] = measured
        del gradient
    return report


def capture_controller_images(step, inputs):
    images = []

    class CaptureNCC(NCCVxm):
        def forward(self, first, second):
            images.append((first.detach().clone(), second.detach().clone()))
            return super().forward(first, second)

    with torch.no_grad(), patch.object(losses, "NCCVxm", CaptureNCC):
        _, metrics = runtime._controller_pair_loss(step, inputs)
    if len(images) != 2:
        raise RuntimeError("Expected exactly two directed NCC calls")
    return images, metrics


def audit_pair(step, inputs, pair, rng, *, probe_fn, save):
    """Audit one known failing state; only the preceding replay updates weights."""
    result = {
        "status": "RUNNING",
        "backend": {
            "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
            "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
            "cudnn_version": torch.backends.cudnn.version(),
        },
        "controller_images": {},
        "controller_probes": [],
        "reference_contract": "FP64 separable box moments; same epsilon/window/warp/normalisation; FP32 scalar",
    }
    initial_state = state_dict_sha256(step.controller.state_dict())
    restore_rng_state(rng)
    images, metrics = capture_controller_images(step, inputs)
    result["captured_metrics"] = metrics
    save(result)
    for direction, (warped, fixed) in zip(("forward", "reverse"), images, strict=True):
        result["controller_images"][direction] = compare_ncc(warped, fixed, (step.config.loss.ncc_window,) * 3)
        save(result)
    del images
    # This changes NCC only inside this diagnostic process. Every other objective
    # term and the frozen U0/bootstrap/features remain the production operations.
    for use_reference, fp32, scale in (
        (False, False, 65536.0),
        (True, False, 65536.0),
        (True, False, 1.0),
        (True, True, 1.0),
    ):
        restore_rng_state(rng)
        ncc_class = ReferenceNCC if use_reference else NCCVxm
        with patch.object(losses, "NCCVxm", ncc_class):
            trial = probe_fn(
                step.controller,
                step.optimizer,
                lambda mode: runtime._controller_pair_loss(step, inputs, diagnostic_fp32=mode),
                device_type=step.device.type,
                scale=scale,
                fp32=fp32,
            )
        trial["ncc_mode"] = "fp64_reference" if use_reference else "production_fp32"
        result["controller_probes"].append(trial)
        if state_dict_sha256(step.controller.state_dict()) != initial_state:
            raise RuntimeError("NCC audit changed controller state")
        save(result)
    # Inspect the U0 endpoint on this train pair using its raw inputs and window.
    # This is not a replay of its 400 training epochs or other seeds.
    result["u0_endpoint_seed0"] = {}
    moving = runtime._tensor_image(step.store, pair["subject_a"], step.device)
    fixed = runtime._tensor_image(step.store, pair["subject_b"], step.device)
    for direction, first, second in (("forward", moving, fixed), ("reverse", fixed, moving)):
        with torch.no_grad(), torch.autocast(device_type=step.device.type, dtype=torch.float16):
            warped, _flow = step.base_runner.model(first, second, alpha_l1=1.0, alpha_l3=1.0)
        result["u0_endpoint_seed0"][direction] = compare_ncc(warped.float(), second, (9, 9, 9))
        del warped, _flow
        save(result)
    result["status"] = "NCC_AUDIT_COMPLETE"
    result["training_validated"] = False
    return result
