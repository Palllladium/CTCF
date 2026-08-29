"""One owner for the operations every Stage-5 module performs on artifacts.

Canonical bytes, immutable writes, reparse-point refusal, root containment and
out-of-contract scalar rejection previously had two or three independent
implementations each. They live here so that a change to any of them is a change
to all of them, and so that two modules cannot silently disagree about what
"canonical" or "inside its root" means.

The scalar guards take the exception class as a keyword because Stage 5 makes a
real distinction: a caller passing a bad argument raises ``ValueError``, a stored
artifact failing its contract raises ``RuntimeError``.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import stat
import tempfile
from collections.abc import Mapping
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from tools.analysis.run_artifacts import sha256_file

SHA256_RE = re.compile(r"[0-9a-f]{64}")
GIT_SHA_RE = re.compile(r"[0-9a-f]{40}")

_REPARSE_POINT = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)


@dataclass(frozen=True, slots=True)
class FileGeneration:
    """Filesystem identity used only to reuse a digest within one unchanged generation."""

    path: str
    device: int
    inode: int
    size: int
    mtime_ns: int
    ctime_ns: int


def file_generation(path: Path) -> FileGeneration:
    info = path.stat()
    return FileGeneration(
        path=str(path),
        device=int(info.st_dev),
        inode=int(info.st_ino),
        size=int(info.st_size),
        mtime_ns=int(info.st_mtime_ns),
        ctime_ns=int(info.st_ctime_ns),
    )


def generation_cache_is_safe() -> bool:
    """Whether ctime identifies content rewrites on this platform.

    Windows exposes creation time as ``st_ctime`` in the Python versions used by
    this project, so a writer can restore every available generation field with
    ``utime``. On that platform verification must not reuse content digests.
    """

    return os.name != "nt"


def canonical_json_bytes(payload: Any) -> bytes:
    """Compact canonical UTF-8 JSON: the form every contract digest is taken over."""
    text = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return (text + "\n").encode("utf-8")


def readable_json_bytes(payload: Any) -> bytes:
    """Indented canonical UTF-8 JSON: the form data attestations and metrics are stored in.

    Never interchangeable with :func:`canonical_json_bytes` — the two produce different
    bytes, and therefore different digests, for the same object. A file is authenticated
    with the function that wrote it and with no other.
    """
    text = json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
    return (text + "\n").encode("utf-8")


def canonical_sha256(payload: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def is_link_like(path: Path) -> bool:
    """True for POSIX symlinks and for Windows junctions and other reparse points."""
    if path.is_symlink():
        return True
    try:
        attributes = getattr(path.lstat(), "st_file_attributes", 0)
    except OSError:
        return False
    return bool(attributes & _REPARSE_POINT)


def require_regular_file(path: Path, label: str, *, error: type[Exception] = RuntimeError) -> Path:
    if not path.is_file() or is_link_like(path):
        raise error(f"missing or linked {label}: {path}")
    return path


def require_plain_directory(
    path: Path,
    label: str,
    *,
    create: bool = False,
    error: type[Exception] = RuntimeError,
) -> Path:
    path = Path(path)
    if create:
        path.mkdir(parents=True, exist_ok=True)
    if not path.is_dir() or is_link_like(path):
        raise error(f"{label} must be a regular directory: {path}")
    return path.resolve()


def relative_posix(
    value: Any,
    label: str,
    *,
    suffix: str | None = None,
    error: type[Exception] = RuntimeError,
) -> PurePosixPath:
    """Accept only a normalised, relative, separator-safe POSIX path."""
    if not isinstance(value, str) or not value or "\\" in value:
        raise error(f"invalid {label}: {value!r}")
    path = PurePosixPath(value)
    if path.is_absolute() or path.as_posix() != value or any(part in {"", ".", ".."} for part in path.parts):
        raise error(f"unsafe {label}: {value!r}")
    if suffix is not None and path.suffix != suffix:
        raise error(f"{label} must end in {suffix}: {value!r}")
    return path


def resolve_inside_root(
    root: Path,
    relative_value: str,
    *,
    label: str,
    suffix: str | None = None,
    must_exist: bool = True,
    error: type[Exception] = RuntimeError,
) -> Path:
    """Join a declared relative path to a root, refusing escapes and traversed links."""
    relative = relative_posix(relative_value, label, suffix=suffix, error=error)
    candidate = root.joinpath(*relative.parts)
    if must_exist:
        require_regular_file(candidate, label, error=error)
        current = candidate
        while current != root:
            if is_link_like(current):
                raise error(f"{label} must not traverse a link: {candidate}")
            current = current.parent
    try:
        candidate.resolve(strict=False).relative_to(root)
    except ValueError as exc:
        raise error(f"{label} escapes its root: {relative_value}") from exc
    return candidate


def load_json_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"Expected a JSON object: {path}")
    return payload


def atomic_write_bytes(path: Path, payload: bytes) -> None:
    """Write through a temporary file, fsync it, then rename it into place."""
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(handle, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        with suppress(FileNotFoundError):
            os.unlink(temporary)
        raise


def write_immutable_bytes(
    path: Path,
    payload: bytes,
    *,
    label: str = "Stage 5 artifact",
    error: type[Exception] = FileExistsError,
) -> str:
    """Create one file or verify that its bytes are already exactly this payload."""
    if path.exists() or is_link_like(path):
        if not path.is_file() or is_link_like(path) or path.read_bytes() != payload:
            raise error(f"Refusing to replace a different immutable {label}: {path}")
    else:
        atomic_write_bytes(path, payload)
    return sha256_file(path)


def write_immutable_json(path: Path, payload: Mapping[str, Any]) -> str:
    """Create one canonical JSON object or verify its already-identical bytes."""
    return write_immutable_bytes(path, canonical_json_bytes(dict(payload)))


def require_exact_fields(
    payload: Mapping[str, Any],
    fields: frozenset[str],
    label: str,
    *,
    error: type[Exception] = RuntimeError,
) -> None:
    observed = set(payload)
    if observed != fields:
        raise error(f"{label} fields changed; missing={sorted(fields - observed)} extra={sorted(observed - fields)}")


def require_nonempty(value: Any, label: str, *, error: type[Exception] = RuntimeError) -> str:
    if not isinstance(value, str) or not value:
        raise error(f"{label} must be a non-empty string")
    return value


def require_sha256(value: Any, label: str, *, error: type[Exception] = RuntimeError) -> str:
    if not isinstance(value, str) or SHA256_RE.fullmatch(value) is None:
        raise error(f"{label} must be a lowercase SHA-256")
    return value


def require_git_sha(value: Any, label: str, *, error: type[Exception] = RuntimeError) -> str:
    if not isinstance(value, str) or GIT_SHA_RE.fullmatch(value) is None:
        raise error(f"{label} must be a full lowercase Git SHA")
    return value


def require_int(
    value: Any,
    label: str,
    *,
    minimum: int = 0,
    error: type[Exception] = RuntimeError,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise error(f"{label} must be an integer >= {minimum}")
    return value


def require_finite(
    value: Any,
    label: str,
    *,
    minimum: float | None = None,
    allow_none: bool = False,
    error: type[Exception] = RuntimeError,
) -> float | None:
    """Reject bools, non-numbers, NaN and infinities, then an optional lower bound.

    ``bool`` is excluded explicitly: it is an ``int`` subclass, so ``True`` would
    otherwise satisfy every numeric contract in Stage 5.
    """
    if value is None and allow_none:
        return None
    bound = "a finite number" if minimum is None else f"a finite number >= {minimum}"
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise error(f"{label} must be {bound}")
    parsed = float(value)
    if not math.isfinite(parsed) or (minimum is not None and parsed < minimum):
        raise error(f"{label} must be {bound}")
    return parsed


def require_positive_decimal(value: Any, label: str, *, error: type[Exception] = RuntimeError) -> float:
    """Accept only an explicit decimal string, so the stored epsilon is never a float repr."""
    if not isinstance(value, str) or not value:
        raise error(f"{label} must be an explicit positive decimal string")
    try:
        parsed = float(value)
    except ValueError as exc:
        raise error(f"{label} must be an explicit positive decimal string") from exc
    result = require_finite(parsed, label, error=error)
    if result is None or result <= 0.0:
        raise error(f"{label} must be positive and finite")
    return result


__all__ = [
    "GIT_SHA_RE",
    "SHA256_RE",
    "FileGeneration",
    "atomic_write_bytes",
    "canonical_json_bytes",
    "canonical_sha256",
    "file_generation",
    "generation_cache_is_safe",
    "is_link_like",
    "load_json_object",
    "readable_json_bytes",
    "relative_posix",
    "require_exact_fields",
    "require_finite",
    "require_git_sha",
    "require_int",
    "require_nonempty",
    "require_plain_directory",
    "require_positive_decimal",
    "require_regular_file",
    "require_sha256",
    "resolve_inside_root",
    "write_immutable_bytes",
    "write_immutable_json",
]
