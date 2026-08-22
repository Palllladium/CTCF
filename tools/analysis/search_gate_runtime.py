from __future__ import annotations

import csv
import os
import shutil
import tempfile
from contextlib import suppress
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from tools.analysis.run_artifacts import sha256_file
from tools.analysis.search_gate_common import case_id_from_path
from tools.analysis.transactional_search import load_flow_npz, save_flow_npz_atomic
from utils.cert_exact import certify_flow_exact


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def dataset_rows(files: list[str], dataset: str, split: str, atlas: str | None) -> list[dict[str, Any]]:
    """Observe every frozen input once: identity, size, hash and mtime at prepare time."""
    rows: list[dict[str, Any]] = []
    for value in [*files, *([atlas] if atlas else [])]:
        path = Path(value).resolve()
        stat = path.stat()
        is_atlas = atlas is not None and value == atlas
        rows.append(
            {
                "dataset": dataset,
                "split": "atlas" if is_atlas else split,
                "case_id": "atlas" if is_atlas else case_id_from_path(value),
                "path": str(path),
                "bytes": stat.st_size,
                "sha256": sha256_file(path),
                "mtime_utc": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat().replace("+00:00", "Z"),
            }
        )
    return rows


def parse_physical_gpus(value: str, num_shards: int, message: str) -> list[str]:
    """One unique non-negative integer per shard; `message` keeps each gate's CLI error text."""
    physical_gpus = [item.strip() for item in value.split(",")]
    if (
        len(physical_gpus) != num_shards
        or any(not item.isdigit() for item in physical_gpus)
        or len(set(physical_gpus)) != len(physical_gpus)
    ):
        raise ValueError(message)
    return physical_gpus


def round_robin_shards(case_ids: list[str], num_shards: int) -> dict[str, list[str]]:
    """Assign by position, so the partition is a pure function of the case order."""
    return {
        str(index): [value for position, value in enumerate(case_ids) if position % num_shards == index]
        for index in range(num_shards)
    }


def shard_gpu_map(physical_gpus: list[str]) -> dict[str, str]:
    return {str(index): value for index, value in enumerate(physical_gpus)}


def expected_shard_for_case(contract: dict[str, Any], value: str) -> int:
    return next(index for index in range(contract["num_shards"]) if value in contract["shards"][str(index)])


def flattened_shards(contract: dict[str, Any]) -> list[str]:
    """Contract cases in shard order: shard 0 in full, then shard 1, and so on."""
    return [value for index in range(contract["num_shards"]) for value in contract["shards"][str(index)]]


def validate_shard_partition(contract: dict[str, Any], observed: list[str], expected_total: int, message: str) -> None:
    """Refuse a worker set that dropped, duplicated or reordered the frozen partition.

    Order is compared against `flattened_shards`, not `case_ids`: round-robin interleaves,
    so the two orders differ by construction.
    """
    if (
        observed != flattened_shards(contract)
        or len(observed) != expected_total
        or sorted(observed) != sorted(contract["case_ids"])
    ):
        raise RuntimeError(message)


def attempt_dir(root: Path, attempt_id: str) -> Path:
    return root / "workers" / "attempts" / attempt_id


def worker_marker_paths(root: Path, attempt_id: str, shard_index: int) -> tuple[Path, Path]:
    directory = attempt_dir(root, attempt_id)
    return (
        directory / f"worker_{shard_index:02d}.json",
        directory / f"worker_{shard_index:02d}_failure.json",
    )


def atomic_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{destination.name}.rollback.", dir=destination.parent)
    os.close(fd)
    try:
        shutil.copyfile(source, temporary)
        os.replace(temporary, destination)
    finally:
        with suppress(FileNotFoundError):
            os.unlink(temporary)


def save_reload_certify(candidate: torch.Tensor, path: Path, eps: float) -> tuple[torch.Tensor, dict[str, Any]]:
    """Persist, read back, then certify the stored bytes; callers own how a failure is reported."""
    save_flow_npz_atomic(path, candidate.float())
    stored = load_flow_npz(path)
    exact = certify_flow_exact(stored, eps=str(eps))
    return stored, exact
