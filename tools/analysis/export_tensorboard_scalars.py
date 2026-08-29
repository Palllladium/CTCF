#!/usr/bin/env python3
"""Export legacy TensorBoard scalars without retaining heavy image payloads.

The exporter is deliberately fail-closed.  It accepts only TensorBoard event
files containing ``file_version`` records and summary values of type
``simple_value`` or ``image``.  Scalar values are exported losslessly (their
IEEE-754 bits are recorded); image payloads are counted but never copied.

Run this script with the project's ``oasis-ctcf`` Python environment, which
provides TensorBoard::

    python tools/analysis/export_tensorboard_scalars.py
    python tools/analysis/export_tensorboard_scalars.py --check
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import importlib.metadata
import io
import json
import math
import os
import shutil
import struct
import sys
import tempfile
import uuid
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "CTCF_TENSORBOARD_SCALAR_EXPORT_V1"
DEFAULT_OUTPUT = "logs/TENSORBOARD_EXPORT_20260820"
DEFAULT_INVENTORY = "logs/tensorboard_inventory_20260820.csv"
EVENT_GLOB = "events.out.tfevents*"
READ_CHUNK_BYTES = 4 * 1024 * 1024

SCALAR_FIELDS = (
    "event_file_id",
    "event_file_ordinal",
    "event_record_index",
    "summary_value_index",
    "source_relative_path",
    "tag",
    "step",
    "wall_time_utc",
    "wall_time_ieee754_hex",
    "value_float32",
    "value_ieee754_hex",
)

TAG_SUMMARY_FIELDS = (
    "tag",
    "scalar_points",
    "source_file_count",
    "run_directory_count",
    "min_step",
    "max_step",
    "first_wall_time_utc",
    "last_wall_time_utc",
    "finite_points",
    "nonfinite_points",
    "min_value_float32",
    "max_value_float32",
)

INVENTORY_FIELDS = (
    "event_file_id",
    "source_relative_path",
    "bytes",
    "sha256",
    "event_records",
    "scalar_points",
    "scalar_sequence_sha256",
    "image_points",
    "image_encoded_bytes",
    "scalar_tags_json",
    "image_tags_json",
    "first_wall_time_utc",
    "last_wall_time_utc",
    "min_step",
    "max_step",
    "sibling_logfile_relative_path",
    "sibling_logfile_bytes",
    "sibling_logfile_sha256",
    "sibling_summary_paths_json",
    "sibling_per_case_paths_json",
    "parse_status",
    "retention_class",
    "retention_candidate",
    "deletion_authorization",
    "deletion_status",
    "notes",
)


class ExportError(RuntimeError):
    """Raised when a lossless export invariant is not satisfied."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(READ_CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _relative_posix(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError as exc:
        raise ExportError(f"Path is outside the repository root: {path}") from exc


def _display_relative(path: Path, root: Path) -> str:
    return Path(os.path.relpath(path.resolve(), root.resolve())).as_posix()


def _resolve_from_root(root: Path, value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _wall_time_utc(value: float) -> str:
    if not math.isfinite(value):
        raise ExportError(f"Non-finite wall_time is not supported: {value!r}")
    return datetime.fromtimestamp(value, timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def _float32_hex(value: float) -> str:
    return struct.pack(">f", value).hex()


def _float64_hex(value: float) -> str:
    return struct.pack(">d", value).hex()


def _float32_decimal(value: float) -> str:
    # Nine significant digits are sufficient for round-tripping every binary32.
    return format(struct.unpack(">f", struct.pack(">f", value))[0], ".9g")


def _canonical_scalar_digest_update(
    digest: Any,
    record_index: int,
    value_index: int,
    tag: str,
    step: int,
    wall_time_hex: str,
    value_hex: str,
) -> None:
    payload = [record_index, value_index, tag, step, wall_time_hex, value_hex]
    digest.update(
        (json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n").encode(
            "utf-8",
        ),
    )


def _same_file_state(left: os.stat_result, right: os.stat_result) -> bool:
    return left.st_size == right.st_size and left.st_mtime_ns == right.st_mtime_ns


def _open_deterministic_gzip_text(path: Path):
    raw = path.open("wb")
    compressed = gzip.GzipFile(
        filename="",
        mode="wb",
        compresslevel=9,
        fileobj=raw,
        mtime=0,
    )
    return raw, compressed, io.TextIOWrapper(compressed, encoding="utf-8", newline="")


def _close_gzip_text(raw: Any, compressed: Any, text: Any) -> None:
    try:
        text.flush()
        text.detach()
        compressed.close()
    finally:
        raw.close()


def _json_list(values: Iterable[str]) -> str:
    return json.dumps(sorted(set(values)), ensure_ascii=False, separators=(",", ":"))


def _sibling_metadata(source: Path, repo_root: Path) -> dict[str, Any]:
    logfile = source.parent / "logfile.log"
    if logfile.exists():
        if not logfile.is_file() or logfile.is_symlink():
            raise ExportError(f"Unsafe sibling logfile: {logfile}")
        logfile_path = _relative_posix(logfile, repo_root)
        logfile_bytes: int | str = logfile.stat().st_size
        logfile_sha = _sha256_file(logfile)
    else:
        logfile_path = ""
        logfile_bytes = ""
        logfile_sha = ""

    siblings = sorted(path for path in source.parent.iterdir() if path.is_file())
    summaries = [
        _relative_posix(path, repo_root)
        for path in siblings
        if "summary" in path.name.lower() and not path.name.startswith("events.out.tfevents")
    ]
    per_case = [
        _relative_posix(path, repo_root)
        for path in siblings
        if "per_case" in path.name.lower() and not path.name.startswith("events.out.tfevents")
    ]
    return {
        "sibling_logfile_relative_path": logfile_path,
        "sibling_logfile_bytes": logfile_bytes,
        "sibling_logfile_sha256": logfile_sha,
        "sibling_summary_paths_json": _json_list(summaries),
        "sibling_per_case_paths_json": _json_list(per_case),
    }


def _fail_closed_raw_events(source: Path, relative_path: str) -> Iterable[bytes]:
    """Yield every TFRecord and reject truncated or corrupt tails."""
    try:
        from tensorboard.backend.event_processing.event_file_loader import (
            _make_tf_record_iterator,
        )
        from tensorboard.compat import tf
    except ModuleNotFoundError as exc:  # pragma: no cover - depends on caller environment
        raise ExportError(
            "TensorBoard is required; run with the oasis-ctcf Python environment",
        ) from exc

    records_read = 0
    iterator = _make_tf_record_iterator(str(source))
    while True:
        try:
            raw_event = next(iterator)
        except StopIteration:
            return
        except tf.errors.DataLossError as exc:
            raise ExportError(
                f"Corrupt or truncated TensorBoard TFRecord after {records_read} records: {relative_path}",
            ) from exc
        records_read += 1
        yield raw_event


def _scan_source(task: tuple[str, str, int, str | None]) -> dict[str, Any]:
    """Hash and stream one event file; optionally write a headerless gzip member."""
    source_text, repo_root_text, ordinal, fragment_text = task
    source = Path(source_text)
    repo_root = Path(repo_root_text)

    try:
        from tensorboard.compat.proto import event_pb2
    except ModuleNotFoundError as exc:  # pragma: no cover - depends on caller environment
        raise ExportError(
            "TensorBoard is required; run with the oasis-ctcf Python environment",
        ) from exc

    if source.is_symlink() or not source.is_file():
        raise ExportError(f"Event source must be a regular non-symlink file: {source}")

    state_before = source.stat()
    source_sha = _sha256_file(source)
    state_after_hash = source.stat()
    if not _same_file_state(state_before, state_after_hash):
        raise ExportError(f"Source changed while hashing: {source}")

    event_file_id = f"TB-SHA256-{source_sha}"
    relative_path = _relative_posix(source, repo_root)
    fragment = Path(fragment_text) if fragment_text else None

    raw_handle = compressed_handle = text_handle = None
    writer = None
    if fragment is not None:
        fragment.parent.mkdir(parents=True, exist_ok=True)
        raw_handle, compressed_handle, text_handle = _open_deterministic_gzip_text(fragment)
        writer = csv.writer(text_handle, lineterminator="\n")

    event_records = 0
    scalar_points = 0
    image_points = 0
    image_encoded_bytes = 0
    scalar_tags: Counter[str] = Counter()
    image_tags: Counter[str] = Counter()
    tag_stats: dict[str, dict[str, Any]] = {}
    scalar_digest = hashlib.sha256()
    first_wall_time: float | None = None
    last_wall_time: float | None = None
    min_step: int | None = None
    max_step: int | None = None

    try:
        for record_index, raw_event in enumerate(_fail_closed_raw_events(source, relative_path)):
            event = event_pb2.Event.FromString(raw_event)
            event_records += 1
            wall_time = float(event.wall_time)
            wall_time_utc = _wall_time_utc(wall_time)
            wall_time_hex = _float64_hex(wall_time)
            step = int(event.step)
            first_wall_time = wall_time if first_wall_time is None else min(first_wall_time, wall_time)
            last_wall_time = wall_time if last_wall_time is None else max(last_wall_time, wall_time)
            min_step = step if min_step is None else min(min_step, step)
            max_step = step if max_step is None else max(max_step, step)

            event_kind = event.WhichOneof("what")
            if event_kind == "file_version":
                continue
            if event_kind != "summary":
                raise ExportError(
                    f"Unsupported TensorBoard event kind {event_kind!r} in {relative_path}",
                )

            for value_index, value in enumerate(event.summary.value):
                value_kind = value.WhichOneof("value")
                if value_kind == "simple_value":
                    numeric = float(value.simple_value)
                    value_hex = _float32_hex(numeric)
                    value_decimal = _float32_decimal(numeric)
                    scalar_points += 1
                    scalar_tags[value.tag] += 1
                    _canonical_scalar_digest_update(
                        scalar_digest,
                        record_index,
                        value_index,
                        value.tag,
                        step,
                        wall_time_hex,
                        value_hex,
                    )
                    if writer is not None:
                        writer.writerow(
                            (
                                event_file_id,
                                ordinal,
                                record_index,
                                value_index,
                                relative_path,
                                value.tag,
                                step,
                                wall_time_utc,
                                wall_time_hex,
                                value_decimal,
                                value_hex,
                            ),
                        )

                    stats = tag_stats.setdefault(
                        value.tag,
                        {
                            "count": 0,
                            "min_step": None,
                            "max_step": None,
                            "first_wall_time": None,
                            "last_wall_time": None,
                            "finite_points": 0,
                            "nonfinite_points": 0,
                            "min_value": None,
                            "max_value": None,
                        },
                    )
                    stats["count"] += 1
                    stats["min_step"] = step if stats["min_step"] is None else min(stats["min_step"], step)
                    stats["max_step"] = step if stats["max_step"] is None else max(stats["max_step"], step)
                    stats["first_wall_time"] = (
                        wall_time if stats["first_wall_time"] is None else min(stats["first_wall_time"], wall_time)
                    )
                    stats["last_wall_time"] = (
                        wall_time if stats["last_wall_time"] is None else max(stats["last_wall_time"], wall_time)
                    )
                    if math.isfinite(numeric):
                        stats["finite_points"] += 1
                        stats["min_value"] = numeric if stats["min_value"] is None else min(stats["min_value"], numeric)
                        stats["max_value"] = numeric if stats["max_value"] is None else max(stats["max_value"], numeric)
                    else:
                        stats["nonfinite_points"] += 1
                elif value_kind == "image":
                    image_points += 1
                    image_tags[value.tag] += 1
                    image_encoded_bytes += len(value.image.encoded_image_string)
                else:
                    raise ExportError(
                        f"Unsupported summary value kind {value_kind!r} in {relative_path}",
                    )
    finally:
        if text_handle is not None:
            _close_gzip_text(raw_handle, compressed_handle, text_handle)

    state_after_parse = source.stat()
    if not _same_file_state(state_after_hash, state_after_parse):
        raise ExportError(f"Source changed while parsing: {source}")
    if event_records == 0:
        raise ExportError(f"No readable TensorBoard records: {source}")

    sibling = _sibling_metadata(source, repo_root)
    return {
        "event_file_ordinal": ordinal,
        "event_file_id": event_file_id,
        "source_relative_path": relative_path,
        "bytes": state_before.st_size,
        "sha256": source_sha,
        "event_records": event_records,
        "scalar_points": scalar_points,
        "scalar_sequence_sha256": scalar_digest.hexdigest(),
        "image_points": image_points,
        "image_encoded_bytes": image_encoded_bytes,
        "scalar_tags": dict(sorted(scalar_tags.items())),
        "image_tags": dict(sorted(image_tags.items())),
        "tag_stats": tag_stats,
        "first_wall_time": first_wall_time,
        "last_wall_time": last_wall_time,
        "min_step": min_step,
        "max_step": max_step,
        **sibling,
    }


def _render_csv(fields: Sequence[str], rows: Iterable[Mapping[str, Any]]) -> bytes:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fields, lineterminator="\n", extrasaction="raise")
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue().encode("utf-8")


def _inventory_rows(results: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in sorted(results, key=lambda value: int(value["event_file_ordinal"])):
        rows.append(
            {
                "event_file_id": item["event_file_id"],
                "source_relative_path": item["source_relative_path"],
                "bytes": item["bytes"],
                "sha256": item["sha256"],
                "event_records": item["event_records"],
                "scalar_points": item["scalar_points"],
                "scalar_sequence_sha256": item["scalar_sequence_sha256"],
                "image_points": item["image_points"],
                "image_encoded_bytes": item["image_encoded_bytes"],
                "scalar_tags_json": json.dumps(
                    item["scalar_tags"], ensure_ascii=False, sort_keys=True, separators=(",", ":")
                ),
                "image_tags_json": json.dumps(
                    item["image_tags"], ensure_ascii=False, sort_keys=True, separators=(",", ":")
                ),
                "first_wall_time_utc": _wall_time_utc(float(item["first_wall_time"])),
                "last_wall_time_utc": _wall_time_utc(float(item["last_wall_time"])),
                "min_step": item["min_step"],
                "max_step": item["max_step"],
                "sibling_logfile_relative_path": item["sibling_logfile_relative_path"],
                "sibling_logfile_bytes": item["sibling_logfile_bytes"],
                "sibling_logfile_sha256": item["sibling_logfile_sha256"],
                "sibling_summary_paths_json": item["sibling_summary_paths_json"],
                "sibling_per_case_paths_json": item["sibling_per_case_paths_json"],
                "parse_status": "PASS",
                "retention_class": "HEAVY_REGENERABLE",
                "retention_candidate": "YES_AFTER_VALIDATED_EXPORT_AND_EXPLICIT_APPROVAL",
                "deletion_authorization": "NOT_GRANTED",
                "deletion_status": "PRESENT",
                "notes": "Scalars exported losslessly; image payloads counted but not exported.",
            },
        )
    return rows


def _tag_summary_rows(results: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    aggregate: dict[str, dict[str, Any]] = {}
    for item in results:
        source = str(item["source_relative_path"])
        run_dir = str(Path(source).parent.as_posix())
        for tag, local in item["tag_stats"].items():
            stats = aggregate.setdefault(
                tag,
                {
                    "scalar_points": 0,
                    "source_files": set(),
                    "run_directories": set(),
                    "min_step": None,
                    "max_step": None,
                    "first_wall_time": None,
                    "last_wall_time": None,
                    "finite_points": 0,
                    "nonfinite_points": 0,
                    "min_value": None,
                    "max_value": None,
                },
            )
            stats["scalar_points"] += int(local["count"])
            stats["source_files"].add(source)
            stats["run_directories"].add(run_dir)
            for key in ("min_step", "first_wall_time", "min_value"):
                value = local[key]
                if value is not None:
                    stats[key] = value if stats[key] is None else min(stats[key], value)
            for key in ("max_step", "last_wall_time", "max_value"):
                value = local[key]
                if value is not None:
                    stats[key] = value if stats[key] is None else max(stats[key], value)
            stats["finite_points"] += int(local["finite_points"])
            stats["nonfinite_points"] += int(local["nonfinite_points"])

    rows = []
    for tag, stats in sorted(aggregate.items()):
        rows.append(
            {
                "tag": tag,
                "scalar_points": stats["scalar_points"],
                "source_file_count": len(stats["source_files"]),
                "run_directory_count": len(stats["run_directories"]),
                "min_step": stats["min_step"],
                "max_step": stats["max_step"],
                "first_wall_time_utc": _wall_time_utc(float(stats["first_wall_time"])),
                "last_wall_time_utc": _wall_time_utc(float(stats["last_wall_time"])),
                "finite_points": stats["finite_points"],
                "nonfinite_points": stats["nonfinite_points"],
                "min_value_float32": (
                    "" if stats["min_value"] is None else _float32_decimal(float(stats["min_value"]))
                ),
                "max_value_float32": (
                    "" if stats["max_value"] is None else _float32_decimal(float(stats["max_value"]))
                ),
            },
        )
    return rows


def _source_index_digest(results: Sequence[Mapping[str, Any]]) -> str:
    rows = [
        [
            item["source_relative_path"],
            item["bytes"],
            item["sha256"],
            item["scalar_sequence_sha256"],
        ]
        for item in sorted(results, key=lambda value: str(value["source_relative_path"]))
    ]
    payload = json.dumps(rows, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return _sha256_bytes(payload)


def _discover_sources(repo_root: Path, events_root: Path) -> list[Path]:
    if not events_root.is_dir():
        raise ExportError(f"Events root does not exist: {events_root}")
    _relative_posix(events_root, repo_root)
    sources = sorted(
        (path for path in events_root.rglob(EVENT_GLOB) if path.is_file()),
        key=lambda path: _relative_posix(path, repo_root),
    )
    if not sources:
        raise ExportError(f"No event files matching {EVENT_GLOB!r} below {events_root}")
    for source in sources:
        if source.is_symlink():
            raise ExportError(f"Symlink event sources are not accepted: {source}")
    return sources


def _run_scans(
    sources: Sequence[Path],
    repo_root: Path,
    workers: int,
    fragment_dir: Path | None,
) -> list[dict[str, Any]]:
    tasks = []
    for ordinal, source in enumerate(sources):
        fragment = None
        if fragment_dir is not None:
            fragment = str(fragment_dir / f"{ordinal:04d}.csv.gz")
        tasks.append((str(source), str(repo_root), ordinal, fragment))

    results: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(_scan_source, task): task[2] for task in tasks}
        for completed, future in enumerate(as_completed(futures), start=1):
            result = future.result()
            results.append(result)
            print(
                f"[{completed}/{len(tasks)}] {result['source_relative_path']} "
                f"scalars={result['scalar_points']} images={result['image_points']}",
                flush=True,
            )
    results.sort(key=lambda value: int(value["event_file_ordinal"]))
    return results


def _write_header_member(path: Path) -> None:
    raw, compressed, text = _open_deterministic_gzip_text(path)
    try:
        csv.writer(text, lineterminator="\n").writerow(SCALAR_FIELDS)
    finally:
        _close_gzip_text(raw, compressed, text)


def _assemble_scalar_gzip(stage: Path, source_count: int) -> Path:
    fragments = stage / ".fragments"
    header = fragments / "header.csv.gz"
    _write_header_member(header)
    output = stage / "scalars.csv.gz"
    with output.open("wb") as target:
        for component in [header, *(fragments / f"{index:04d}.csv.gz" for index in range(source_count))]:
            if not component.is_file():
                raise ExportError(f"Missing scalar fragment: {component}")
            with component.open("rb") as source:
                shutil.copyfileobj(source, target, length=READ_CHUNK_BYTES)
        target.flush()
        os.fsync(target.fileno())
    shutil.rmtree(fragments)
    return output


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_text = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(temporary_text)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _validate_scalar_csv(path: Path, results: Sequence[Mapping[str, Any]]) -> int:
    expected = {str(item["source_relative_path"]): item for item in results}
    seen_paths: list[str] = []
    current_path: str | None = None
    current_digest = hashlib.sha256()
    current_count = 0
    total_rows = 0
    previous_key: tuple[int, int, int] | None = None

    def finish_source() -> None:
        nonlocal current_path, current_digest, current_count
        if current_path is None:
            return
        item = expected.get(current_path)
        if item is None:
            raise ExportError(f"Scalar CSV references an unknown source: {current_path}")
        if current_count != int(item["scalar_points"]):
            raise ExportError(
                f"Scalar count mismatch for {current_path}: {current_count} != {item['scalar_points']}",
            )
        if current_digest.hexdigest() != item["scalar_sequence_sha256"]:
            raise ExportError(f"Scalar sequence digest mismatch for {current_path}")
        seen_paths.append(current_path)

    with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != SCALAR_FIELDS:
            raise ExportError(f"Unexpected scalar CSV columns: {reader.fieldnames}")
        for row in reader:
            source_path = row["source_relative_path"]
            if source_path != current_path:
                finish_source()
                if source_path in seen_paths:
                    raise ExportError(f"Non-contiguous scalar source block: {source_path}")
                current_path = source_path
                current_digest = hashlib.sha256()
                current_count = 0
                previous_key = None

            item = expected.get(source_path)
            if item is None:
                raise ExportError(f"Unknown scalar source path: {source_path}")
            if row["event_file_id"] != item["event_file_id"]:
                raise ExportError(f"event_file_id mismatch for {source_path}")
            ordinal = int(row["event_file_ordinal"])
            record_index = int(row["event_record_index"])
            value_index = int(row["summary_value_index"])
            if ordinal != int(item["event_file_ordinal"]):
                raise ExportError(f"event_file_ordinal mismatch for {source_path}")
            key = (ordinal, record_index, value_index)
            if previous_key is not None and key <= previous_key:
                raise ExportError(f"Scalar row order is not strictly increasing at {key}")
            previous_key = key

            wall_time = struct.unpack(">d", bytes.fromhex(row["wall_time_ieee754_hex"]))[0]
            if row["wall_time_utc"] != _wall_time_utc(wall_time):
                raise ExportError(f"wall_time representation mismatch for {source_path} at {key}")
            value_bytes = bytes.fromhex(row["value_ieee754_hex"])
            if len(value_bytes) != 4:
                raise ExportError(f"Invalid float32 hex for {source_path} at {key}")
            exact_value = struct.unpack(">f", value_bytes)[0]
            decimal_value = float(row["value_float32"])
            if math.isnan(exact_value):
                if not math.isnan(decimal_value):
                    raise ExportError(f"NaN decimal mismatch for {source_path} at {key}")
            elif struct.pack(">f", decimal_value) != value_bytes:
                raise ExportError(f"float32 decimal mismatch for {source_path} at {key}")

            _canonical_scalar_digest_update(
                current_digest,
                record_index,
                value_index,
                row["tag"],
                int(row["step"]),
                row["wall_time_ieee754_hex"],
                row["value_ieee754_hex"],
            )
            current_count += 1
            total_rows += 1
    finish_source()

    expected_paths = [str(item["source_relative_path"]) for item in results if int(item["scalar_points"]) > 0]
    if seen_paths != expected_paths:
        raise ExportError("Scalar source block order/completeness mismatch")
    return total_rows


def _manifest_payload(
    repo_root: Path,
    events_root: Path,
    output_dir: Path,
    inventory_path: Path,
    results: Sequence[Mapping[str, Any]],
    scalars_path: Path,
    tag_summary_path: Path,
    inventory_bytes: bytes,
    scalar_rows: int,
) -> dict[str, Any]:
    total_source_bytes = sum(int(item["bytes"]) for item in results)
    total_images = sum(int(item["image_points"]) for item in results)
    total_image_bytes = sum(int(item["image_encoded_bytes"]) for item in results)
    source_hashes = Counter(str(item["sha256"]) for item in results)
    run_dirs = {str(Path(str(item["source_relative_path"])).parent.as_posix()) for item in results}
    script_path = Path(__file__).resolve()
    return {
        "schema_version": SCHEMA_VERSION,
        "build_status": "COMPLETE",
        "determinism": {
            "source_order": "source_relative_path ascending",
            "record_order": "event record index then summary value index",
            "gzip_mtime": 0,
            "csv_encoding": "UTF-8",
            "csv_line_ending": "LF",
        },
        "tool": {
            "relative_path": _relative_posix(script_path, repo_root),
            "sha256": _sha256_file(script_path),
            "python_version": sys.version.split()[0],
            "tensorboard_version": importlib.metadata.version("tensorboard"),
        },
        "source": {
            "events_root": _relative_posix(events_root, repo_root),
            "event_file_count": len(results),
            "run_directory_count": len(run_dirs),
            "total_bytes": total_source_bytes,
            "exact_duplicate_file_count": sum(count - 1 for count in source_hashes.values() if count > 1),
            "source_index_sha256": _source_index_digest(results),
        },
        "products": {
            "scalars.csv.gz": {
                "relative_path": _relative_posix(output_dir / scalars_path.name, repo_root),
                "bytes": scalars_path.stat().st_size,
                "sha256": _sha256_file(scalars_path),
                "rows": scalar_rows,
            },
            "tag_summary.csv": {
                "relative_path": _relative_posix(output_dir / tag_summary_path.name, repo_root),
                "bytes": tag_summary_path.stat().st_size,
                "sha256": _sha256_file(tag_summary_path),
                "rows": len(_tag_summary_rows(results)),
            },
            "external_inventory": {
                "relative_path": _display_relative(inventory_path, repo_root),
                "bytes": len(inventory_bytes),
                "sha256": _sha256_bytes(inventory_bytes),
                "rows": len(results),
            },
        },
        "boundary": {
            "exported_summary_value_kinds": ["simple_value"],
            "scalar_points_exported": scalar_rows,
            "image_payloads_exported": False,
            "image_points_not_exported": total_images,
            "image_encoded_bytes_not_exported": total_image_bytes,
            "reason": (
                "TensorBoard images are HEAVY_REGENERABLE diagnostic previews; "
                "this package preserves their per-source counts/tags/byte totals, not pixels."
            ),
        },
        "retention": {
            "source_class": "HEAVY_REGENERABLE",
            "source_files_are_candidates": True,
            "deletion_authorization": "NOT_GRANTED",
            "deletion_status": "PRESENT",
            "required_before_deletion": (
                "Successful --check followed by explicit user authorization for the exact inventory rows."
            ),
        },
        "validation": {
            "source_parse_status": "PASS",
            "unsupported_event_kinds": 0,
            "unsupported_summary_value_kinds": 0,
            "multi_file_overlaps_collapsed": False,
            "scalar_sequence_digests_recorded": True,
        },
        "output_root": _relative_posix(output_dir, repo_root),
    }


def _export(args: argparse.Namespace) -> None:
    repo_root = args.repo_root
    events_root = args.events_root
    output_dir = args.output_dir
    inventory_path = args.inventory

    if output_dir.exists():
        raise ExportError(f"Output directory already exists; refusing to overwrite: {output_dir}")
    if inventory_path.exists():
        raise ExportError(f"Inventory already exists; refusing to overwrite: {inventory_path}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    inventory_path.parent.mkdir(parents=True, exist_ok=True)
    sources = _discover_sources(repo_root, events_root)

    stage = output_dir.parent / f".{output_dir.name}.tmp-{uuid.uuid4().hex}"
    stage.mkdir(parents=False, exist_ok=False)
    published = False
    try:
        fragment_dir = stage / ".fragments"
        fragment_dir.mkdir()
        results = _run_scans(sources, repo_root, args.workers, fragment_dir)
        scalars_path = _assemble_scalar_gzip(stage, len(results))

        inventory_bytes = _render_csv(INVENTORY_FIELDS, _inventory_rows(results))
        tag_summary_path = stage / "tag_summary.csv"
        tag_summary_bytes = _render_csv(TAG_SUMMARY_FIELDS, _tag_summary_rows(results))
        tag_summary_path.write_bytes(tag_summary_bytes)

        scalar_rows = _validate_scalar_csv(scalars_path, results)
        expected_scalar_rows = sum(int(item["scalar_points"]) for item in results)
        if scalar_rows != expected_scalar_rows:
            raise ExportError(f"Scalar row mismatch: {scalar_rows} != {expected_scalar_rows}")

        manifest = _manifest_payload(
            repo_root,
            events_root,
            output_dir,
            inventory_path,
            results,
            scalars_path,
            tag_summary_path,
            inventory_bytes,
            scalar_rows,
        )
        manifest_bytes = (json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode("utf-8")
        (stage / "manifest.json").write_bytes(manifest_bytes)

        os.replace(stage, output_dir)
        published = True
        _atomic_write_bytes(inventory_path, inventory_bytes)
    except Exception:
        if stage.exists():
            shutil.rmtree(stage)
        if published and output_dir.exists() and not inventory_path.exists():
            shutil.rmtree(output_dir)
        raise

    print(
        json.dumps(
            {
                "state": "COMPLETE",
                "event_files": len(sources),
                "scalar_rows": scalar_rows,
                "output_dir": str(output_dir),
                "inventory": str(inventory_path),
            },
            ensure_ascii=False,
            sort_keys=True,
        ),
    )


def _load_manifest(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ExportError(f"Cannot read manifest {path}: {exc}") from exc
    if value.get("schema_version") != SCHEMA_VERSION or value.get("build_status") != "COMPLETE":
        raise ExportError("Manifest schema/build_status mismatch")
    return value


def _assert_product(path: Path, metadata: Mapping[str, Any]) -> None:
    if not path.is_file():
        raise ExportError(f"Missing product: {path}")
    if path.stat().st_size != int(metadata["bytes"]):
        raise ExportError(f"Product size mismatch: {path}")
    if _sha256_file(path) != metadata["sha256"]:
        raise ExportError(f"Product SHA-256 mismatch: {path}")


def _check(args: argparse.Namespace) -> None:
    repo_root = args.repo_root
    output_dir = args.output_dir
    inventory_path = args.inventory
    manifest = _load_manifest(output_dir / "manifest.json")

    scalars_path = output_dir / "scalars.csv.gz"
    tag_summary_path = output_dir / "tag_summary.csv"
    _assert_product(scalars_path, manifest["products"]["scalars.csv.gz"])
    _assert_product(tag_summary_path, manifest["products"]["tag_summary.csv"])
    _assert_product(inventory_path, manifest["products"]["external_inventory"])

    if _sha256_file(Path(__file__).resolve()) != manifest["tool"]["sha256"]:
        raise ExportError("Exporter script SHA-256 differs from the manifest")

    sources = _discover_sources(repo_root, args.events_root)
    results = _run_scans(sources, repo_root, args.workers, fragment_dir=None)
    if _source_index_digest(results) != manifest["source"]["source_index_sha256"]:
        raise ExportError("Source index/SHA-256 digest mismatch")

    regenerated_inventory = _render_csv(INVENTORY_FIELDS, _inventory_rows(results))
    if regenerated_inventory != inventory_path.read_bytes():
        raise ExportError("Inventory does not match the current sources")
    regenerated_tag_summary = _render_csv(TAG_SUMMARY_FIELDS, _tag_summary_rows(results))
    if regenerated_tag_summary != tag_summary_path.read_bytes():
        raise ExportError("Tag summary does not match the current sources")

    scalar_rows = _validate_scalar_csv(scalars_path, results)
    if scalar_rows != int(manifest["products"]["scalars.csv.gz"]["rows"]):
        raise ExportError("Scalar row count differs from the manifest")
    if scalar_rows != sum(int(item["scalar_points"]) for item in results):
        raise ExportError("Scalar row count differs from the event sources")

    total_images = sum(int(item["image_points"]) for item in results)
    total_image_bytes = sum(int(item["image_encoded_bytes"]) for item in results)
    if total_images != int(manifest["boundary"]["image_points_not_exported"]):
        raise ExportError("Image point boundary count mismatch")
    if total_image_bytes != int(manifest["boundary"]["image_encoded_bytes_not_exported"]):
        raise ExportError("Image byte boundary count mismatch")
    if manifest["retention"]["deletion_authorization"] != "NOT_GRANTED":
        raise ExportError("Deletion authorization must remain NOT_GRANTED")

    print(
        json.dumps(
            {
                "state": "PASS",
                "event_files": len(results),
                "scalar_rows": scalar_rows,
                "image_points_not_exported": total_images,
                "source_hashes": "PASS",
                "products": "PASS",
                "deletion_authorization": "NOT_GRANTED",
            },
            ensure_ascii=False,
            sort_keys=True,
        ),
    )


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", default=".", help="CTCF repository root")
    parser.add_argument("--events-root", default="logs", help="Event search root, relative to repo")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT, help="New deterministic export directory")
    parser.add_argument("--inventory", default=DEFAULT_INVENTORY, help="External per-source inventory CSV")
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--check", action="store_true", help="Verify products and all current sources")
    args = parser.parse_args(argv)
    if args.workers < 1:
        parser.error("--workers must be positive")
    args.repo_root = Path(args.repo_root).resolve()
    args.events_root = _resolve_from_root(args.repo_root, args.events_root)
    args.output_dir = _resolve_from_root(args.repo_root, args.output_dir)
    args.inventory = _resolve_from_root(args.repo_root, args.inventory)
    return args


def main(argv: Sequence[str] | None = None) -> int:
    try:
        importlib.metadata.version("tensorboard")
    except importlib.metadata.PackageNotFoundError:
        print(
            "ERROR: TensorBoard is unavailable; use the oasis-ctcf Python environment.",
            file=sys.stderr,
        )
        return 2
    try:
        args = _parse_args(argv)
        if args.check:
            _check(args)
        else:
            _export(args)
    except (ExportError, OSError, ValueError, csv.Error) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
