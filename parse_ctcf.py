#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
parse.py — selective, safe project dumper (ALLOW-LIST)

This parser exports:
  ✅ models/CTCF/**   (all files under that folder)
  ✅ experiments/OASIS/train_CTCF.py

and nothing else.

It writes ONE output .txt containing:
  1) a compact tree (only allowed paths)
  2) concatenated file contents with clear separators

Safety:
  - Never crashes on encoding/binary files
  - Per-file and total size limits with truncation
  - Skips huge files automatically (or truncates)
"""

from __future__ import annotations

import argparse
import os
import sys
import mimetypes
from pathlib import Path
from typing import Iterable, List, Tuple, Optional


# -----------------------------
# ALLOW LIST (edit if needed)
# -----------------------------
ALLOWED_DIRS = {
    Path("models") / "CTCF",
    # Path("experiments"),
    # Path("utils"),
    # Path("tools"),
    # Path("datasets")
}

# Default output filename
DEFAULT_OUT = "proj.txt"

EXCLUDE_DIR_NAMES = {
    "__pycache__",
    ".git",
    ".idea",
    ".vscode",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    "build",
    "dist",
}

EXCLUDE_FILE_SUFFIXES = {
    ".pyc",
    ".pyo",
    ".pyd",
}


# -----------------------------
# Helpers
# -----------------------------
def norm_rel(p: Path) -> Path:
    """Normalize to a clean relative path with forward components."""
    return Path(*p.parts)


def is_allowed(abs_path: Path, project_root: Path) -> bool:
    """
    Returns True if abs_path is exactly one of ALLOWED_FILES (relative),
    or is inside one of ALLOWED_DIRS (relative).
    """
    try:
        rel = norm_rel(abs_path.relative_to(project_root))
    except Exception:
        return False

    for d in ALLOWED_DIRS:
        d = norm_rel(d)
        if rel == d or d in rel.parents:
            return True

    return False


def safe_read_text(
    path: Path,
    max_bytes: int,
) -> Tuple[str, bool]:
    """
    Read file content safely.
    Returns (text, truncated_flag).
    - Tries UTF-8 with errors=replace; fallback to latin-1 if needed.
    - If file is likely binary, returns a placeholder line.
    """
    try:
        size = path.stat().st_size
    except Exception:
        return "[[UNREADABLE: stat() failed]]\n", False

    # Fast binary heuristic: if it has null bytes in first chunk
    try:
        with path.open("rb") as f:
            head = f.read(min(8192, size if size > 0 else 0))
            if b"\x00" in head:
                return "[[BINARY FILE: skipped]]\n", False
    except Exception:
        return "[[UNREADABLE: open() failed]]\n", False

    truncated = False
    to_read = size
    if max_bytes >= 0 and size > max_bytes:
        to_read = max_bytes
        truncated = True

    raw = b""
    try:
        with path.open("rb") as f:
            raw = f.read(to_read)
    except Exception:
        return "[[UNREADABLE: read() failed]]\n", False

    # Decode safely
    try:
        text = raw.decode("utf-8", errors="replace")
    except Exception:
        text = raw.decode("latin-1", errors="replace")

    if truncated:
        text += "\n[[TRUNCATED: file exceeds per-file limit]]\n"

    return text, truncated


def iter_allowed_files(project_root: Path) -> List[Path]:
    """
    Enumerate all allowed files under ALLOWED_DIRS.
    Returned paths are absolute, sorted by relative path.
    Skips cache/build folders like __pycache__.
    """
    out: List[Path] = []

    root_resolved = project_root.resolve()

    def is_excluded(p: Path) -> bool:
        # Skip anything that has excluded dir in its path parts
        parts = set(p.parts)
        if parts & EXCLUDE_DIR_NAMES:
            return True
        # Skip common compiled/binary python artifacts
        if p.suffix.lower() in EXCLUDE_FILE_SUFFIXES:
            return True
        return False

    for d in sorted(ALLOWED_DIRS):
        abs_d = (root_resolved / d).resolve()
        if abs_d.is_dir():
            for p in abs_d.rglob("*"):
                if not p.is_file():
                    continue
                if is_excluded(p):
                    continue
                out.append(p)
        else:
            pass

    # Deduplicate + sort by relative path
    uniq = {}
    for p in out:
        try:
            rel = p.resolve().relative_to(root_resolved)
        except Exception:
            continue
        rel_key = str(rel).replace("\\", "/")
        uniq[rel_key] = p.resolve()

    keys = sorted(uniq.keys())
    return [uniq[k] for k in keys]


def build_tree(project_root: Path, files_abs: List[Path]) -> str:
    """
    Build a minimal tree only containing the allowed files.
    """
    # Collect all path components
    rels = []
    for p in files_abs:
        rel = p.relative_to(project_root).as_posix()
        rels.append(rel)

    # Build nested dict-like structure (using dicts)
    tree = {}
    for rel in rels:
        parts = rel.split("/")
        cur = tree
        for i, part in enumerate(parts):
            cur = cur.setdefault(part, {})

    def render(node: dict, prefix: str = "") -> List[str]:
        lines = []
        keys = sorted(node.keys())
        for i, k in enumerate(keys):
            last = (i == len(keys) - 1)
            branch = "└── " if last else "├── "
            lines.append(prefix + branch + k)
            child = node[k]
            if isinstance(child, dict) and child:
                ext = "    " if last else "│   "
                lines.extend(render(child, prefix + ext))
        return lines

    return "\n".join(render(tree)) + ("\n" if tree else "")


def write_dump(
    project_root: Path,
    out_path: Path,
    files_abs: List[Path],
    max_file_bytes: int,
    max_total_bytes: int,
) -> None:
    total_written = 0

    def w(s: str) -> None:
        nonlocal total_written
        if max_total_bytes >= 0 and total_written >= max_total_bytes:
            return
        if max_total_bytes >= 0:
            # Write only remaining capacity
            remain = max_total_bytes - total_written
            if remain <= 0:
                return
            b = s.encode("utf-8", errors="replace")
            if len(b) > remain:
                b = b[:remain]
                s2 = b.decode("utf-8", errors="replace")
                out_f.write(s2)
                total_written += len(b)
                return
        out_f.write(s)
        total_written += len(s.encode("utf-8", errors="replace"))

    with out_path.open("w", encoding="utf-8", newline="\n") as out_f:
        w(f"PROJECT ROOT: {project_root.as_posix()}\n")
        w("ALLOW-LIST:\n")
        for d in sorted(ALLOWED_DIRS):
            w(f"  DIR : {d.as_posix()}\n")

        w("\n" + "=" * 80 + "\n")
        w("TREE (allowed only)\n")
        w("=" * 80 + "\n")
        w(build_tree(project_root, files_abs))

        w("\n" + "=" * 80 + "\n")
        w("CONTENTS\n")
        w("=" * 80 + "\n\n")

        for abs_path in files_abs:
            if max_total_bytes >= 0 and total_written >= max_total_bytes:
                w("\n[[STOPPED: total output limit reached]]\n")
                break

            rel = abs_path.relative_to(project_root).as_posix()
            try:
                size = abs_path.stat().st_size
            except Exception:
                size = -1

            w("\n" + "#" * 80 + "\n")
            w(f"# FILE: {rel}\n")
            if size >= 0:
                w(f"# SIZE: {size} bytes\n")
            w("#" * 80 + "\n\n")

            text, _ = safe_read_text(abs_path, max_bytes=max_file_bytes)
            w(text)

        w("\n")


# -----------------------------
# CLI
# -----------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Export only models/CTCF/** and experiments/OASIS/train_CTCF.py into one txt."
    )
    p.add_argument(
        "--root",
        type=str,
        default=None,
        help="Project root. Default: directory where this script lives.",
    )
    p.add_argument(
        "--out",
        type=str,
        default=DEFAULT_OUT,
        help=f"Output txt file (default: {DEFAULT_OUT})",
    )
    p.add_argument(
        "--max_file_kb",
        type=int,
        default=512,
        help="Max KB per file in output (-1 = no limit). Default: 512KB.",
    )
    p.add_argument(
        "--max_total_mb",
        type=int,
        default=50,
        help="Max total MB of output (-1 = no limit). Default: 50MB.",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()

    script_dir = Path(__file__).resolve().parent
    project_root = Path(args.root).resolve() if args.root else script_dir

    out_path = Path(args.out).resolve()
    if not out_path.is_absolute():
        out_path = (project_root / out_path).resolve()

    max_file_bytes = -1 if args.max_file_kb < 0 else int(args.max_file_kb) * 1024
    max_total_bytes = -1 if args.max_total_mb < 0 else int(args.max_total_mb) * 1024 * 1024

    files_abs = iter_allowed_files(project_root)
    if not files_abs:
        print("[ERROR] No allowed files found. Check ALLOWED_DIRS/ALLOWED_FILES and --root.", file=sys.stderr)
        return 2

    print(f"[INFO] Project root: {project_root}")
    print(f"[INFO] Allowed files: {len(files_abs)}")
    print(f"[INFO] Output: {out_path}")
    print(f"[INFO] Limits: per-file={args.max_file_kb}KB, total={args.max_total_mb}MB")

    try:
        write_dump(
            project_root=project_root,
            out_path=out_path,
            files_abs=files_abs,
            max_file_bytes=max_file_bytes,
            max_total_bytes=max_total_bytes,
        )
    except Exception as e:
        print(f"[ERROR] Failed: {e}", file=sys.stderr)
        return 1

    print("[OK] Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())