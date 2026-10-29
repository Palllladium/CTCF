from __future__ import annotations

import argparse
import csv
import glob
import hashlib
import io
import json
import os
import platform
import tempfile
import zipfile
from collections.abc import Iterable
from contextlib import suppress
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REGENERATED_LEGACY_BUFFERS = frozenset({"st_half.grid"})


@dataclass(frozen=True)
class CheckpointCompatibility:
    allowed_missing_buffers: tuple[str, ...]
    disallowed_missing_keys: tuple[str, ...]
    unexpected_keys: tuple[str, ...]

    @property
    def compatible(self) -> bool:
        return not self.disallowed_missing_keys and not self.unexpected_keys


def classify_checkpoint_incompatibilities(
    missing_keys: Iterable[str],
    unexpected_keys: Iterable[str],
    model_buffer_names: Iterable[str],
) -> CheckpointCompatibility:
    """Allow only explicitly known, model-owned deterministic legacy buffers."""
    missing = set(missing_keys)
    unexpected = set(unexpected_keys)
    buffers = set(model_buffer_names)
    allowed = missing & REGENERATED_LEGACY_BUFFERS & buffers
    return CheckpointCompatibility(
        allowed_missing_buffers=tuple(sorted(allowed)),
        disallowed_missing_keys=tuple(sorted(missing - allowed)),
        unexpected_keys=tuple(sorted(unexpected)),
    )


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(text)
        os.replace(tmp_name, path)
    except BaseException:
        with suppress(FileNotFoundError):
            os.unlink(tmp_name)
        raise


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    buffer = io.StringIO()
    json.dump(payload, buffer, ensure_ascii=False, indent=2, sort_keys=True)
    buffer.write("\n")
    atomic_write_text(path, buffer.getvalue())


def rows_to_tsv(fieldnames: list[str], rows: Iterable[dict[str, Any]]) -> str:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fieldnames, delimiter="\t", lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue()


def rows_to_csv(fieldnames: list[str], rows: Iterable[dict[str, Any]]) -> str:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue()


def write_dataset_manifest(paths_profile: int, dataset_splits: list[str], output: Path) -> int:
    from experiments.core.path_profiles import get_dataset_paths

    rows: list[dict[str, Any]] = []
    for value in dataset_splits:
        try:
            dataset, split = value.split(":", 1)
        except ValueError as exc:
            raise ValueError(f"Invalid dataset split '{value}'; expected DATASET:val or DATASET:test") from exc
        dataset = dataset.upper()
        split = split.lower()
        if split not in {"val", "test"}:
            raise ValueError(f"Unsupported split '{split}' for {dataset}")

        paths = get_dataset_paths(paths_profile, dataset)
        key = "test_dir" if split == "test" else "val_dir"
        if key not in paths:
            raise ValueError(f"Path profile {paths_profile} has no {key} for {dataset}")
        root = paths[key]
        files = sorted(glob.glob(os.path.join(root, "*.pkl")))
        if not files:
            raise FileNotFoundError(f"No .pkl files for {dataset}/{split}: {root}")
        for name in files:
            stat = os.stat(name)
            rows.append(
                {
                    "dataset": dataset,
                    "split": split,
                    "path": name,
                    "bytes": stat.st_size,
                    "mtime_utc": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat().replace("+00:00", "Z"),
                }
            )

        atlas = paths.get("atlas_path")
        if atlas:
            stat = os.stat(atlas)
            rows.append(
                {
                    "dataset": dataset,
                    "split": "atlas",
                    "path": atlas,
                    "bytes": stat.st_size,
                    "mtime_utc": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat().replace("+00:00", "Z"),
                }
            )

    fields = ["dataset", "split", "path", "bytes", "mtime_utc"]
    atomic_write_text(output, rows_to_tsv(fields, rows))
    return len(rows)


def validate_result_directory(datasets: Path, result_dir: Path, dataset: str, split: str) -> int:
    required = [result_dir / "per_case.csv", result_dir / "summary.csv", result_dir / "summary.json"]
    missing = [str(path) for path in required if not path.is_file() or path.stat().st_size <= 0]
    if missing:
        raise FileNotFoundError(f"Missing required result file(s): {', '.join(missing)}")

    with datasets.open(encoding="utf-8", newline="") as stream:
        expected = sum(
            1
            for row in csv.DictReader(stream, delimiter="\t")
            if row["dataset"] == dataset.upper() and row["split"] == split.lower()
        )
    with (result_dir / "per_case.csv").open(encoding="utf-8", newline="") as stream:
        observed = sum(1 for _ in csv.DictReader(stream))
    with (result_dir / "summary.json").open(encoding="utf-8") as stream:
        summary_n = int(json.load(stream)["n_cases"])
    if expected <= 0 or observed != expected or summary_n != expected:
        raise ValueError(
            f"Case-count mismatch for {dataset}/{split}: expected={expected}, per_case={observed}, summary={summary_n}"
        )
    return expected


def aggregate_summaries(run_root: Path, patterns: list[str], expected_count: int, output: Path) -> int:
    summaries = sorted({path for pattern in patterns for path in run_root.glob(pattern)})
    if len(summaries) != expected_count:
        raise ValueError(f"Expected {expected_count} summary files, found {len(summaries)}")

    fields = [
        "suite",
        "run",
        "metric",
        "mean",
        "std",
        "sem",
        "ci95",
        "min",
        "max",
        "n_cases",
        "model",
        "ckpt_path",
        "test_dir",
    ]
    rows: list[dict[str, Any]] = []
    for path in summaries:
        data = json.loads(path.read_text(encoding="utf-8"))
        relative = path.relative_to(run_root)
        suite, run = relative.parts[0], relative.parts[1]
        for metric, values in sorted(data["metrics"].items()):
            rows.append(
                {
                    "suite": suite,
                    "run": run,
                    "metric": metric,
                    "n_cases": data["n_cases"],
                    "model": data["model"],
                    "ckpt_path": data["ckpt_path"],
                    "test_dir": data["test_dir"],
                    **values,
                }
            )
    atomic_write_text(output, rows_to_csv(fields, rows))
    return len(summaries)


def write_output_index(run_root: Path, output: Path, excluded_paths: set[str]) -> int:
    rows = []
    for path in sorted(p for p in run_root.rglob("*") if p.is_file()):
        relative = path.relative_to(run_root).as_posix()
        if relative in excluded_paths:
            continue
        rows.append(
            {
                "relative_path": relative,
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    atomic_write_text(output, rows_to_tsv(["relative_path", "bytes", "sha256"], rows))
    return len(rows)


def finalize_run(args: argparse.Namespace) -> Path:
    run_root = args.run_root.resolve()
    strict_checkpoint_load = getattr(args, "strict_checkpoint_load", True)
    if not isinstance(strict_checkpoint_load, bool):
        raise TypeError("strict_checkpoint_load must be boolean")
    required = {
        "commands": run_root / "commands.sh",
        "datasets": run_root / "datasets.tsv",
        "environment": run_root / "environment.txt",
        "git_status": run_root / "git_status.txt",
    }
    for label, path in required.items():
        if not path.is_file():
            raise FileNotFoundError(f"Missing {label} file: {path}")

    outputs = run_root / "outputs.tsv"
    manifest_path = run_root / "run_manifest.json"
    # Exclude only the two self-referential root files. Nested stage manifests are
    # independent outputs and must remain covered by the top-level index.
    write_output_index(run_root, outputs, {outputs.relative_to(run_root).as_posix(), manifest_path.name})

    preflights = []
    for path in sorted((run_root / "preflight").glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        preflights.append(
            {
                "file": path.relative_to(run_root).as_posix(),
                "status": data.get("status"),
                "checkpoint": data.get("checkpoint"),
                "sha256": data.get("sha256"),
                "ctcf_config": data.get("ctcf_config"),
                "time_steps": data.get("time_steps"),
                "ctcf_l3_svf": data.get("ctcf_l3_svf"),
                "load": data.get("load"),
            }
        )
    if args.status == "COMPLETE":
        if len(preflights) != args.expected_preflights:
            raise ValueError(f"Expected {args.expected_preflights} preflight reports, found {len(preflights)}")
        expected_strict_checkpoint_load = bool(preflights)
        if strict_checkpoint_load is not expected_strict_checkpoint_load:
            raise ValueError(
                "A COMPLETE run must report strict_checkpoint_load=true exactly when checkpoint preflights are present"
            )
        for item in preflights:
            load = item.get("load") or {}
            missing = set(load.get("missing_keys") or [])
            allowed_missing = set(load.get("allowed_missing_buffers") or [])
            if (
                item["status"] != "PASS"
                or load.get("strict") is not True
                or missing != allowed_missing
                or not allowed_missing.issubset(REGENERATED_LEGACY_BUFFERS)
                or load.get("unexpected_keys")
            ):
                raise ValueError("A COMPLETE run requires successful strict checkpoint preflights")
    elif len(preflights) > args.expected_preflights:
        raise ValueError(f"Expected at most {args.expected_preflights} preflight reports, found {len(preflights)}")

    manifest = {
        "schema": "ctcf-native-manifest-v1",
        "run_id": args.run_id,
        "status": args.status,
        "exit_code": args.exit_code,
        "started_at_utc": args.started_at,
        "completed_at_utc": args.completed_at,
        "code": {
            "git_head": args.git_head,
            "branch": args.branch,
            "tracked_tree_clean_at_start": True,
        },
        "execution": {
            "host": platform.node(),
            "gpu_index": args.gpu_index,
            "mode": args.mode,
            "paths_profile": args.paths_profile,
            "seed": args.seed,
            "deterministic": True,
            "strict_checkpoint_load": strict_checkpoint_load,
            "time_steps": args.time_steps,
        },
        "checkpoints": preflights,
        "files": {
            "commands_sha256": sha256_file(required["commands"]),
            "datasets_sha256": sha256_file(required["datasets"]),
            "environment_sha256": sha256_file(required["environment"]),
            "git_status_sha256": sha256_file(required["git_status"]),
            "outputs_sha256": sha256_file(outputs),
        },
        "storage": {
            "checkpoint_bytes_in_package": False,
            "flow_bytes_in_package": False,
            "remote_locator": args.remote_locator,
        },
    }
    atomic_write_json(manifest_path, manifest)
    return manifest_path


def package_run(
    run_root: Path,
    prefix: Path,
    exit_code: int,
    required_steps: list[str],
    git_head: str,
    label: str,
    include_roots: list[Path] | None = None,
) -> str:
    """Package compact diagnostics, including failed/interrupted runs, without weights or field payloads."""
    codes = {path.stem: int(path.read_text().strip()) for path in (run_root / "exit_codes").glob("*.txt")}
    complete = exit_code == 0 and all(codes.get(step) == 0 for step in required_steps)
    status = "COMPLETE" if complete else "PARTIAL" if exit_code in (130, 143) else "FAILED"
    archive = Path(f"{prefix}__{status}.zip")
    archive.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(
        run_root / "package_manifest.json",
        {
            "status": status,
            "label": label,
            "required_steps": required_steps,
            "head": git_head,
            "exit_codes": codes,
            "runner_exit_code": exit_code,
            "payload": "arrays, displacements and checkpoints retained on the server",
            "included_compact_roots": [str(p) for p in include_roots or []],
        },
    )
    temporary = archive.with_suffix(".zip.tmp")
    try:
        with zipfile.ZipFile(temporary, "w", zipfile.ZIP_DEFLATED) as handle:
            sums = []
            roots = [(run_root, ""), *[(p, f"external/{i}_{p.name}/") for i, p in enumerate(include_roots or [])]]
            for root, folder in roots:
                for path in sorted(root.rglob("*")):
                    if not path.is_file() or path.suffix not in (".json", ".csv", ".txt", ".log", ".py", ".md"):
                        continue
                    name = folder + path.relative_to(root).as_posix()
                    digest = hashlib.sha256()
                    with path.open("rb") as source, handle.open(f"{archive.stem}/{name}", "w") as target:
                        while chunk := source.read(8 * 1024 * 1024):
                            target.write(chunk)
                            digest.update(chunk)
                    sums.append(f"{digest.hexdigest()}  {name}")
            handle.writestr(f"{archive.stem}/SHA256SUMS", "\n".join(sums) + "\n")
        temporary.replace(archive)
    finally:
        temporary.unlink(missing_ok=True)
    digest = sha256_file(archive)
    sidecar = Path(str(archive) + ".sha256")
    atomic_write_text(sidecar, f"{digest}  {archive.name}\n")
    print(f"[{label} PACKAGE] {archive.resolve()}")
    print(f"[{label} SIDECAR] {sidecar.resolve()}")
    print(f"{digest}  {archive.name}")
    print(f"[{label} HEAVY RETAINED] {run_root}")
    return status


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Reusable manifests and validation for CTCF runs.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    package = subparsers.add_parser("package")
    package.add_argument("--run-root", type=Path, required=True)
    package.add_argument("--prefix", type=Path, required=True)
    package.add_argument("--exit-code", type=int, required=True)
    package.add_argument("--required-steps", nargs="+", required=True)
    package.add_argument("--git-head", required=True)
    package.add_argument("--label", default="RUN")
    package.add_argument("--include-root", type=Path, action="append", default=[])

    dataset = subparsers.add_parser("dataset-manifest")
    dataset.add_argument("--paths-profile", type=int, required=True)
    dataset.add_argument("--dataset-split", action="append", required=True)
    dataset.add_argument("--output", type=Path, required=True)

    validate = subparsers.add_parser("validate-result")
    validate.add_argument("--datasets", type=Path, required=True)
    validate.add_argument("--result-dir", type=Path, required=True)
    validate.add_argument("--dataset", required=True)
    validate.add_argument("--split", choices=["val", "test"], required=True)

    aggregate = subparsers.add_parser("aggregate")
    aggregate.add_argument("--run-root", type=Path, required=True)
    aggregate.add_argument("--summary-glob", action="append", required=True)
    aggregate.add_argument("--expected-count", type=int, required=True)
    aggregate.add_argument("--output", type=Path, required=True)

    finalize = subparsers.add_parser("finalize")
    finalize.add_argument("--run-root", type=Path, required=True)
    finalize.add_argument("--run-id", required=True)
    finalize.add_argument("--status", choices=["COMPLETE", "FAILED"], required=True)
    finalize.add_argument("--exit-code", type=int, required=True)
    finalize.add_argument("--started-at", required=True)
    finalize.add_argument("--completed-at", required=True)
    finalize.add_argument("--git-head", required=True)
    finalize.add_argument("--branch", required=True)
    finalize.add_argument("--gpu-index", type=int, required=True)
    finalize.add_argument("--mode", required=True)
    finalize.add_argument("--paths-profile", type=int, required=True)
    finalize.add_argument("--seed", type=int, default=0)
    finalize.add_argument("--time-steps", type=int, required=True)
    finalize.add_argument("--expected-preflights", type=int, required=True)
    finalize.add_argument(
        "--strict-checkpoint-load",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether this run loaded a checkpoint under the strict preflight contract.",
    )
    finalize.add_argument("--remote-locator", default="PENDING_UPLOAD")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.command == "package":
        status = package_run(
            args.run_root,
            args.prefix,
            args.exit_code,
            args.required_steps,
            args.git_head,
            args.label,
            args.include_root,
        )
        return 0 if status == "COMPLETE" else 1
    if args.command == "dataset-manifest":
        count = write_dataset_manifest(args.paths_profile, args.dataset_split, args.output)
        print(f"[DATASET MANIFEST] {args.output} ({count} rows)")
    elif args.command == "validate-result":
        count = validate_result_directory(args.datasets, args.result_dir, args.dataset, args.split)
        print(f"[RESULT VALID] {args.result_dir} ({count} cases)")
    elif args.command == "aggregate":
        count = aggregate_summaries(args.run_root, args.summary_glob, args.expected_count, args.output)
        print(f"[AGGREGATE] {args.output} ({count} summaries)")
    elif args.command == "finalize":
        print(f"[MANIFEST] {finalize_run(args)}")
    else:
        raise AssertionError(f"Unhandled command: {args.command}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
