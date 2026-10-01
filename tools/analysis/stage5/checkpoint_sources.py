"""Read-only checkpoint validation before a loader with recovery capabilities."""

from __future__ import annotations

import json

from experiments.stage5 import runtime
from tools.analysis.run_artifacts import sha256_file


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
