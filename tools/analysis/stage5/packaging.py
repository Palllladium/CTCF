"""Package a quiescent Stage5 compact run without copying retained tensor bytes."""

from __future__ import annotations

import hashlib
import re
import zipfile
from pathlib import Path

from tools.analysis.run_artifacts import sha256_file
from tools.analysis.stage5.primitives import file_generation, is_link_like

# Every tensor, image and archive format written by Stage5 is rejected. These
# suffix checks describe known formats, not a content classifier for arbitrary bytes.
FORBIDDEN_SUFFIXES = frozenset(
    {
        ".pth",
        ".pt",
        ".pt2",
        ".ckpt",
        ".bin",
        ".safetensors",
        ".h5",
        ".npz",
        ".npy",
        ".pkl",
        ".nii",
        ".gz",
        ".mgz",
        ".zip",
        ".tar",
        ".tgz",
        ".bz2",
        ".xz",
        ".7z",
    }
)


def package_run_zip(run_root: Path, export_root: Path, *, root_name: str, archive_stem: str) -> tuple[Path, str]:
    """Create a new single-root ZIP, internal SHA256SUMS, and an external sidecar.

    SHA256SUMS exists only inside this attempt's ZIP. The live compact run can
    therefore gain further attempts without retaining a stale hash manifest.
    Call only after all workers have stopped and the attempt has been finalized.
    """
    for name in (root_name, archive_stem):
        if re.fullmatch(r"[A-Za-z0-9_-]+", name) is None:
            raise ValueError(f"Unsafe Stage5 archive name: {name!r}")
    if is_link_like(run_root) or not run_root.is_dir():
        raise ValueError("Stage5 compact root must be a real directory")
    run_root = run_root.resolve(strict=True)
    export_root = export_root.resolve()
    if export_root.is_relative_to(run_root):
        raise ValueError("Stage5 export directory must be outside the compact root")
    files: list[tuple[Path, str]] = []
    for path in sorted(run_root.rglob("*")):
        relative = path.relative_to(run_root).as_posix()
        if is_link_like(path):
            raise ValueError(f"Linked path is forbidden in Stage5 ZIP: {relative}")
        if path.is_dir():
            continue
        if any(character in relative for character in "\r\n\t\\"):
            raise ValueError(f"Ambiguous Stage5 ZIP path: {relative!r}")
        if not path.is_file() or path.suffix.lower() in FORBIDDEN_SUFFIXES:
            raise ValueError(f"Heavy, archived or non-regular file is forbidden in Stage5 ZIP: {relative}")
        if relative == "SHA256SUMS":
            raise FileExistsError("The compact root contains an unexpected pre-existing SHA256SUMS")
        files.append((path, relative))

    export_root.mkdir(parents=True, exist_ok=True)
    archive = export_root / f"{archive_stem}.zip"
    sidecar = archive.with_suffix(".zip.sha256")
    temporary = archive.with_name(f".{archive.name}.part")
    if any(path.exists() for path in (archive, sidecar, temporary)):
        raise FileExistsError("Stage5 ZIP, checksum or incomplete package already exists")
    hashes: list[str] = []
    generations = {}
    with zipfile.ZipFile(temporary, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as package:
        for path, relative in files:
            generations[path] = file_generation(path)
            digest = hashlib.sha256()
            # Hash the bytes actually written, avoiding a hash/write race.
            with path.open("rb") as source, package.open(f"{root_name}/{relative}", "w", force_zip64=True) as target:
                for block in iter(lambda: source.read(1024 * 1024), b""):
                    digest.update(block)
                    target.write(block)
            hashes.append(f"{digest.hexdigest()}  {relative}\n")
        package.writestr(f"{root_name}/SHA256SUMS", "".join(hashes))
    if any(file_generation(path) != generation for path, generation in generations.items()):
        raise RuntimeError("Stage5 compact files changed during packaging; incomplete .part retained")
    with zipfile.ZipFile(temporary) as package:
        bad_member = package.testzip()
        if bad_member is not None:
            raise RuntimeError(f"Stage5 ZIP verification failed: {bad_member}")
    digest = sha256_file(temporary)
    temporary.rename(archive)
    with sidecar.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(f"{digest}  {archive.name}\n")
    return archive, digest
