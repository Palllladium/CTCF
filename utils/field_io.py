"""Exact float32 persistence of displacement fields consumed by the certificate verifier."""

from __future__ import annotations

import os
import tempfile
from contextlib import suppress
from pathlib import Path

import numpy as np
import torch


def _require_field(field: torch.Tensor, name: str) -> None:
    if field.dim() != 5 or field.shape[0] != 1 or field.shape[1] != 3:
        raise ValueError(f"{name} must have shape [1,3,D,H,W], got {tuple(field.shape)}")


def save_flow_npz_atomic(path: Path, flow: torch.Tensor) -> None:
    """Atomically persist the exact float32 flow object consumed by the verifier."""
    _require_field(flow, "flow")
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".npz", dir=path.parent)
    os.close(fd)
    try:
        array = flow.detach().cpu().contiguous().numpy().astype(np.float32, copy=False)
        np.savez_compressed(temporary_name, flow=array)
        os.replace(temporary_name, path)
    except BaseException:
        with suppress(FileNotFoundError):
            os.unlink(temporary_name)
        raise


def load_flow_npz(path: Path) -> torch.Tensor:
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != {"flow"}:
            raise ValueError(f"{path}: expected the sole key 'flow', got {archive.files}")
        array = archive["flow"]
    if array.dtype != np.float32:
        raise TypeError(f"{path}: stored flow must be float32, got {array.dtype}")
    tensor = torch.from_numpy(np.ascontiguousarray(array))
    if tensor.dim() == 4:
        tensor = tensor.unsqueeze(0)
    _require_field(tensor, str(path))
    return tensor
