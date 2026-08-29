from __future__ import annotations

import csv
import json
import math
import os
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter

from utils import AverageMeter, compute_fig

ITER_LOG_KEYS: tuple[tuple[str, str], ...] = (
    ("ncc", "ncc"),
    ("reg", "reg"),
    ("icon", "icon"),
    ("jac", "jac"),
    ("dice_tr", "dice_tr"),
)

METRICS_SCHEMA_VERSION = "CTCF_EPOCH_METRICS_V1"
METRICS_FIELDS: tuple[str, ...] = (
    "schema_version",
    "experiment",
    "dataset",
    "history_start_epoch",
    "epoch",
    "learning_rate",
    "train_iterations",
    "loss_meters_json",
    "perf_epoch_time_sec",
    "perf_iter_time_ms",
    "perf_peak_gpu_mem_gib",
    "val_dice",
    "val_jac_nonpositive_percent",
    "val_ndv_percent",
    "val_sdlogj",
    "is_best",
    "best_val_dice",
)
LOSS_METER_FIELDS = ("avg", "count", "last", "std", "sum")


@dataclass(frozen=True)
class PreparedEpochMetrics:
    path: Path
    history_start_epoch: int


def _exact_float(value: Any, label: str) -> str:
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError(f"Non-finite epoch metric {label}: {numeric!r}")
    return repr(numeric)


def _optional_exact_float(value: Any | None, label: str) -> str:
    return "" if value is None else _exact_float(value, label)


def _loss_meters_json(meters: Mapping[str, AverageMeter]) -> str:
    payload: dict[str, dict[str, float | int]] = {}
    for name, meter in sorted(meters.items()):
        if not isinstance(name, str) or not name:
            raise ValueError(f"Loss meter names must be non-empty strings, got {name!r}")
        count = int(meter.count)
        if count < 0:
            raise ValueError(f"Loss meter {name!r} has a negative count: {count}")
        values = {
            "avg": float(meter.avg),
            "count": count,
            "last": float(meter.val),
            "std": float(meter.std),
            "sum": float(meter.sum),
        }
        for field, value in values.items():
            if field != "count" and not math.isfinite(float(value)):
                raise ValueError(f"Non-finite loss meter {name}.{field}: {value!r}")
        payload[name] = values
    return json.dumps(payload, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":"))


def _atomic_write_metrics(path: Path, rows: Sequence[Mapping[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_text = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary = Path(temporary_text)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(
                stream,
                fieldnames=METRICS_FIELDS,
                lineterminator="\n",
                extrasaction="raise",
            )
            writer.writeheader()
            writer.writerows(rows)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _validate_loss_meters(raw: str, row_number: int) -> None:
    try:
        meters = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid loss_meters_json at metrics row {row_number}") from exc
    if not isinstance(meters, dict):
        raise ValueError(f"loss_meters_json must be an object at metrics row {row_number}")
    for name, values in meters.items():
        if not isinstance(name, str) or not name or not isinstance(values, dict):
            raise ValueError(f"Invalid loss meter entry at metrics row {row_number}: {name!r}")
        if tuple(sorted(values)) != tuple(sorted(LOSS_METER_FIELDS)):
            raise ValueError(f"Invalid fields for loss meter {name!r} at metrics row {row_number}")
        count = values["count"]
        if not isinstance(count, int) or isinstance(count, bool) or count < 0:
            raise ValueError(f"Invalid count for loss meter {name!r} at metrics row {row_number}")
        for field in ("avg", "last", "std", "sum"):
            value = values[field]
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                raise ValueError(f"Invalid {field} for loss meter {name!r} at metrics row {row_number}")


def _read_metrics(path: Path, experiment: str, dataset: str) -> list[dict[str, str]]:
    if path.is_symlink():
        raise ValueError(f"metrics.csv must not be a symlink: {path}")
    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if tuple(reader.fieldnames or ()) != METRICS_FIELDS:
            raise ValueError(f"Unexpected metrics.csv schema in {path}: {reader.fieldnames}")
        rows = list(reader)

    history_start_epoch: int | None = None
    previous_epoch: int | None = None
    for row_number, row in enumerate(rows, start=2):
        if None in row or any(value is None for value in row.values()):
            raise ValueError(f"Malformed metrics.csv row {row_number} in {path}")
        if row["schema_version"] != METRICS_SCHEMA_VERSION:
            raise ValueError(f"Unexpected metrics schema version at row {row_number}")
        if row["experiment"] != experiment or row["dataset"] != dataset:
            raise ValueError(
                f"metrics.csv context mismatch at row {row_number}: "
                f"{row['experiment']!r}/{row['dataset']!r} != {experiment!r}/{dataset!r}",
            )
        try:
            row_history_start = int(row["history_start_epoch"])
            epoch = int(row["epoch"])
            iterations = int(row["train_iterations"])
            is_best = int(row["is_best"])
        except ValueError as exc:
            raise ValueError(f"Invalid integer field at metrics row {row_number}") from exc
        if str(row_history_start) != row["history_start_epoch"] or row_history_start < 0:
            raise ValueError(
                f"Invalid history_start_epoch at metrics row {row_number}: {row['history_start_epoch']!r}",
            )
        if history_start_epoch is None:
            history_start_epoch = row_history_start
            if epoch != history_start_epoch:
                raise ValueError(
                    f"First metrics epoch {epoch} differs from history_start_epoch {history_start_epoch}",
                )
        elif row_history_start != history_start_epoch:
            raise ValueError(f"history_start_epoch changed at metrics row {row_number}")
        if str(epoch) != row["epoch"] or epoch < 0:
            raise ValueError(f"Invalid epoch at metrics row {row_number}: {row['epoch']!r}")
        if str(iterations) != row["train_iterations"] or iterations < 0:
            raise ValueError(f"Invalid train_iterations at metrics row {row_number}")
        if is_best not in (0, 1) or str(is_best) != row["is_best"]:
            raise ValueError(f"Invalid is_best at metrics row {row_number}")
        if previous_epoch is not None and epoch != previous_epoch + 1:
            raise ValueError(
                f"Non-contiguous metrics epochs at row {row_number}: {previous_epoch} -> {epoch}",
            )
        previous_epoch = epoch

        required_floats = (
            "learning_rate",
            "perf_epoch_time_sec",
            "perf_iter_time_ms",
            "val_dice",
            "val_jac_nonpositive_percent",
            "best_val_dice",
        )
        optional_floats = ("perf_peak_gpu_mem_gib", "val_ndv_percent", "val_sdlogj")
        for field in required_floats:
            try:
                value = float(row[field])
            except ValueError as exc:
                raise ValueError(f"Invalid {field} at metrics row {row_number}") from exc
            if not math.isfinite(value):
                raise ValueError(f"Non-finite {field} at metrics row {row_number}")
        for field in optional_floats:
            if not row[field]:
                continue
            try:
                value = float(row[field])
            except ValueError as exc:
                raise ValueError(f"Invalid {field} at metrics row {row_number}") from exc
            if not math.isfinite(value):
                raise ValueError(f"Non-finite {field} at metrics row {row_number}")
        _validate_loss_meters(row["loss_meters_json"], row_number)
    return rows


def prepare_metrics_csv(
    path: str | os.PathLike[str],
    *,
    experiment: str,
    dataset: str,
    epoch_start: int,
) -> PreparedEpochMetrics:
    """Prepare an atomic, resume-safe canonical epoch journal."""
    metrics_path = Path(path)
    epoch_start = int(epoch_start)
    if epoch_start < 0:
        raise ValueError(f"epoch_start must be non-negative, got {epoch_start}")
    if not metrics_path.exists():
        _atomic_write_metrics(metrics_path, [])
        return PreparedEpochMetrics(metrics_path, epoch_start)

    rows = _read_metrics(metrics_path, experiment, dataset)
    if epoch_start == 0:
        if rows:
            raise ValueError(
                f"Refusing to mix a fresh run with existing epoch metrics in {metrics_path}",
            )
        return PreparedEpochMetrics(metrics_path, 0)

    committed = [row for row in rows if int(row["epoch"]) < epoch_start]
    history_start_epoch = int(rows[0]["history_start_epoch"]) if rows else epoch_start
    if epoch_start < history_start_epoch:
        raise ValueError(
            f"Checkpoint epoch {epoch_start} precedes metrics history_start_epoch {history_start_epoch}",
        )
    if epoch_start > history_start_epoch and (not committed or int(committed[-1]["epoch"]) != epoch_start - 1):
        raise ValueError(
            "metrics.csv ends before resumed checkpoint: "
            f"last={committed[-1]['epoch'] if committed else 'NONE'} expected={epoch_start - 1}",
        )
    if len(committed) != len(rows):
        _atomic_write_metrics(metrics_path, committed)
    return PreparedEpochMetrics(metrics_path, history_start_epoch)


def append_epoch_metrics(
    journal: PreparedEpochMetrics,
    *,
    experiment: str,
    dataset: str,
    epoch: int,
    learning_rate: float,
    train_iterations: int,
    meters: Mapping[str, AverageMeter],
    perf_epoch_time_sec: float,
    perf_iter_time_ms: float,
    perf_peak_gpu_mem_gib: float | None,
    val_dice: float,
    val_jac_nonpositive_percent: float,
    val_ndv_percent: float | None,
    val_sdlogj: float | None,
    is_best: bool,
    best_val_dice: float,
) -> None:
    """Atomically append one complete epoch without rounding any numeric value."""
    if not isinstance(journal, PreparedEpochMetrics):
        raise TypeError("append_epoch_metrics requires the handle returned by prepare_metrics_csv")
    metrics_path = journal.path
    if not metrics_path.is_file():
        raise ValueError(f"metrics.csv was not prepared: {metrics_path}")
    rows = _read_metrics(metrics_path, experiment, dataset)
    epoch = int(epoch)
    if epoch < 0:
        raise ValueError(f"epoch must be non-negative, got {epoch}")
    history_start_epoch = int(rows[0]["history_start_epoch"]) if rows else journal.history_start_epoch
    if history_start_epoch != journal.history_start_epoch:
        raise ValueError(
            "Prepared metrics history boundary changed: "
            f"prepared={journal.history_start_epoch} observed={history_start_epoch}",
        )
    expected_epoch = int(rows[-1]["epoch"]) + 1 if rows else history_start_epoch
    if epoch != expected_epoch:
        raise ValueError(
            "metrics.csv append is not contiguous: "
            f"last={rows[-1]['epoch'] if rows else 'NONE'} next={epoch} expected={expected_epoch}",
        )

    iterations = int(train_iterations)
    if iterations < 0:
        raise ValueError(f"train_iterations must be non-negative, got {iterations}")
    row = {
        "schema_version": METRICS_SCHEMA_VERSION,
        "experiment": experiment,
        "dataset": dataset,
        "history_start_epoch": str(history_start_epoch),
        "epoch": str(epoch),
        "learning_rate": _exact_float(learning_rate, "learning_rate"),
        "train_iterations": str(iterations),
        "loss_meters_json": _loss_meters_json(meters),
        "perf_epoch_time_sec": _exact_float(perf_epoch_time_sec, "perf_epoch_time_sec"),
        "perf_iter_time_ms": _exact_float(perf_iter_time_ms, "perf_iter_time_ms"),
        "perf_peak_gpu_mem_gib": _optional_exact_float(
            perf_peak_gpu_mem_gib,
            "perf_peak_gpu_mem_gib",
        ),
        "val_dice": _exact_float(val_dice, "val_dice"),
        "val_jac_nonpositive_percent": _exact_float(
            val_jac_nonpositive_percent,
            "val_jac_nonpositive_percent",
        ),
        "val_ndv_percent": _optional_exact_float(val_ndv_percent, "val_ndv_percent"),
        "val_sdlogj": _optional_exact_float(val_sdlogj, "val_sdlogj"),
        "is_best": "1" if is_best else "0",
        "best_val_dice": _exact_float(best_val_dice, "best_val_dice"),
    }
    _atomic_write_metrics(metrics_path, [*rows, row])


def write_tb_images(writer: SummaryWriter, last_vis: dict, epoch: int) -> None:
    """Write segmentation and grid previews to TensorBoard."""
    if not last_vis:
        return

    def_out = last_vis.get("def_seg")
    def_grid = last_vis.get("def_grid")
    x_vis = last_vis.get("x_seg")
    y_vis = last_vis.get("y_seg")
    if def_out is None or def_grid is None or x_vis is None or y_vis is None:
        return

    plt.switch_backend("agg")
    figures = {
        "Grid": compute_fig(def_grid),
        "Input": compute_fig(x_vis),
        "Ground truth": compute_fig(y_vis),
        "Prediction": compute_fig(def_out),
    }
    for tag, fig in figures.items():
        writer.add_figure(tag, fig, epoch)
        plt.close(fig)


def format_iter_log(
    meters: dict[str, AverageMeter],
    it: int,
    train_total: int,
    lr_now: float,
) -> str:
    main = "all" if "all" in meters else next(iter(meters.keys()))
    parts = [f"Iter {it:4d} / {train_total:4d} | {main}(avg)={meters[main].avg:.4f}"]

    extras = []
    for key, label in ITER_LOG_KEYS:
        if key in meters:
            extras.append(f"{label}={meters[key].val:.4f}")
    if extras:
        parts.append("| " + " ".join(extras))

    parts.append(f"| lr={lr_now:.3e}")
    return " ".join(parts)


def format_metric_suffix(ndvp: float | None, sdlogj: float | None) -> str:
    parts = []
    if ndvp is not None:
        parts.append(f" | ndv%={ndvp:.2f}")
    if sdlogj is not None:
        parts.append(f" | sdlogj={sdlogj:.4f}")
    return "".join(parts)


def format_train_suffix(meters: dict[str, AverageMeter]) -> str:
    if "alpha_l1" not in meters:
        return ""
    suffix = f" | a1={meters['alpha_l1'].val:.3f} a3={meters['alpha_l3'].val:.3f} w={meters['warm'].val:.3f}"
    if "jac_active_frac" in meters:  # trilinear penalty diagnostics (runaway watch)
        suffix += f" jac_af={meters['jac_active_frac'].avg:.2e} jac_raw={meters['jac_raw'].avg:.3e}"
    if "jac_gradnorm" in meters:
        suffix += f" jac_gn={meters['jac_gradnorm'].avg:.3e}"
    return suffix
