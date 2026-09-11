"""Explicit controller arithmetic; frozen U0 and feature preparation run outside."""

from contextlib import contextmanager

import torch

PRECISION_MODES = ("fp32_strict", "tf32", "bf16")


def precision_mode_contract(mode: str) -> dict:
    if mode not in PRECISION_MODES:
        raise ValueError(f"unknown controller precision mode: {mode!r}")
    return {
        "schema": "ctcf-stage5-comparison-precision-v1",
        "mode": mode,
        "parameter_dtype": "float32",
        "autocast": mode == "bf16",
        "autocast_dtype": "bfloat16" if mode == "bf16" else None,
        "cudnn_allow_tf32": mode == "tf32",
        "matmul_allow_tf32": mode == "tf32",
        "float32_matmul_precision": "high" if mode == "tf32" else "highest",
        "gradient_scaler": False,
        "objective_numerics": "SEE_CONTROLLER_NCC_CONTRACT",
        "frozen_u0_and_features": "OUTSIDE_CONTROLLER_CONTEXT_UNCHANGED",
    }


def controller_precision_contract() -> dict:
    return {
        "schema": "ctcf-stage5-controller-precision-v1",
        "parameter_dtype": "float32",
        "autocast": False,
        "cudnn_allow_tf32": False,
        "matmul_allow_tf32": False,
        "gradient_scaler": False,
        "objective_numerics": "SEE_CONTROLLER_NCC_CONTRACT",
    }


@contextmanager
def precision_context(device: torch.device, mode: str = "fp32_strict"):
    """Scope forward/backward/update arithmetic and restore every changed setting.

    NCC retains its separately specified FP64 moments. Each runner worker is a
    separate process: backend flags must not be changed concurrently by threads.
    """
    contract = precision_mode_contract(mode)
    previous_matmul = torch.get_float32_matmul_precision()
    previous_cudnn = torch.backends.cudnn.allow_tf32
    try:
        torch.set_float32_matmul_precision(contract["float32_matmul_precision"])
        torch.backends.cudnn.allow_tf32 = contract["cudnn_allow_tf32"]
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=contract["autocast"]):
            yield
    finally:
        torch.backends.cudnn.allow_tf32 = previous_cudnn
        torch.set_float32_matmul_precision(previous_matmul)


@contextmanager
def controller_precision(device: torch.device):
    """Production remains strict FP32 regardless of comparison modes."""
    with precision_context(device, "fp32_strict"):
        yield
