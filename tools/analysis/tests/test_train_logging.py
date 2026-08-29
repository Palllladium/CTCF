from __future__ import annotations

import argparse
import csv
import json
import math
import tempfile
import unittest
from contextlib import redirect_stderr
from io import StringIO
from pathlib import Path

from experiments.core.cli_common import add_common_args
from experiments.core.train_logging import (
    METRICS_FIELDS,
    METRICS_SCHEMA_VERSION,
    PreparedEpochMetrics,
    append_epoch_metrics,
    prepare_metrics_csv,
)
from utils import AverageMeter


def _meter(*values: float) -> AverageMeter:
    meter = AverageMeter()
    for value in values:
        meter.update(value)
    return meter


def _append(
    journal: PreparedEpochMetrics,
    epoch: int,
    *,
    val_dice: float | None = None,
    dataset: str = "IXI",
) -> None:
    append_epoch_metrics(
        journal,
        experiment="TEST_RUN",
        dataset=dataset,
        epoch=epoch,
        learning_rate=math.nextafter(1e-4, math.inf),
        train_iterations=7,
        meters={"all": _meter(0.1 + epoch, 0.2 + epoch), "reg": _meter(0.3 + epoch)},
        perf_epoch_time_sec=1.25 + epoch,
        perf_iter_time_ms=2.5 + epoch,
        perf_peak_gpu_mem_gib=None,
        val_dice=(0.5 + epoch / 100) if val_dice is None else val_dice,
        val_jac_nonpositive_percent=0.0,
        val_ndv_percent=0.0,
        val_sdlogj=None,
        is_best=True,
        best_val_dice=(0.5 + epoch / 100) if val_dice is None else val_dice,
    )


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if tuple(reader.fieldnames or ()) != METRICS_FIELDS:
            raise AssertionError(reader.fieldnames)
        return list(reader)


class EpochMetricsCsvTest(unittest.TestCase):
    def test_exact_values_and_variable_loss_meters_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "metrics.csv"
            journal = prepare_metrics_csv(path, experiment="TEST_RUN", dataset="IXI", epoch_start=0)
            learning_rate = math.nextafter(1e-4, math.inf)
            unusual = math.nextafter(0.1, math.inf)
            meters = {"zeta": _meter(unusual, -unusual), "alpha": _meter(math.pi)}

            append_epoch_metrics(
                journal,
                experiment="TEST_RUN",
                dataset="IXI",
                epoch=0,
                learning_rate=learning_rate,
                train_iterations=2,
                meters=meters,
                perf_epoch_time_sec=math.nextafter(12.5, math.inf),
                perf_iter_time_ms=math.nextafter(3.25, math.inf),
                perf_peak_gpu_mem_gib=None,
                val_dice=math.nextafter(0.75, math.inf),
                val_jac_nonpositive_percent=0.0,
                val_ndv_percent=math.nextafter(0.5, math.inf),
                val_sdlogj=None,
                is_best=True,
                best_val_dice=math.nextafter(0.75, math.inf),
            )

            row = _rows(path)[0]
            self.assertEqual(row["schema_version"], METRICS_SCHEMA_VERSION)
            self.assertEqual(row["history_start_epoch"], "0")
            self.assertEqual(float(row["learning_rate"]), learning_rate)
            self.assertEqual(row["perf_peak_gpu_mem_gib"], "")
            self.assertEqual(row["val_sdlogj"], "")
            payload = json.loads(row["loss_meters_json"])
            self.assertEqual(list(payload), ["alpha", "zeta"])
            for name, meter in meters.items():
                self.assertEqual(payload[name]["avg"], meter.avg)
                self.assertEqual(payload[name]["count"], meter.count)
                self.assertEqual(payload[name]["last"], meter.val)
                self.assertEqual(payload[name]["std"], meter.std)
                self.assertEqual(payload[name]["sum"], meter.sum)
            self.assertEqual(list(path.parent.glob(".metrics.csv.*.tmp")), [])

    def test_resume_discards_only_tail_not_backed_by_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "metrics.csv"
            journal = prepare_metrics_csv(path, experiment="TEST_RUN", dataset="IXI", epoch_start=0)
            for epoch in range(3):
                _append(journal, epoch)

            journal = prepare_metrics_csv(path, experiment="TEST_RUN", dataset="IXI", epoch_start=2)
            self.assertEqual([row["epoch"] for row in _rows(path)], ["0", "1"])

            _append(journal, 2, val_dice=0.9876543210123456)
            rows = _rows(path)
            self.assertEqual([row["epoch"] for row in rows], ["0", "1", "2"])
            self.assertEqual(float(rows[-1]["val_dice"]), 0.9876543210123456)

    def test_resume_rejects_checkpoint_ahead_of_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "metrics.csv"
            journal = prepare_metrics_csv(path, experiment="TEST_RUN", dataset="IXI", epoch_start=0)
            _append(journal, 0)
            _append(journal, 1)

            with self.assertRaisesRegex(ValueError, "ends before resumed checkpoint"):
                prepare_metrics_csv(path, experiment="TEST_RUN", dataset="IXI", epoch_start=3)

    def test_legacy_resume_records_missing_history_boundary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            missing = Path(tmp) / "missing" / "metrics.csv"
            journal = prepare_metrics_csv(missing, experiment="TEST_RUN", dataset="IXI", epoch_start=4)
            _append(journal, 4)
            self.assertEqual(_rows(missing)[0]["history_start_epoch"], "4")

            empty = Path(tmp) / "empty" / "metrics.csv"
            prepare_metrics_csv(empty, experiment="TEST_RUN", dataset="IXI", epoch_start=0)
            journal = prepare_metrics_csv(empty, experiment="TEST_RUN", dataset="IXI", epoch_start=7)
            _append(journal, 7)
            _append(journal, 8)
            journal = prepare_metrics_csv(empty, experiment="TEST_RUN", dataset="IXI", epoch_start=9)
            self.assertEqual({row["history_start_epoch"] for row in _rows(empty)}, {"7"})

    def test_fresh_run_and_context_mismatch_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "metrics.csv"
            journal = prepare_metrics_csv(path, experiment="TEST_RUN", dataset="IXI", epoch_start=0)
            _append(journal, 0)

            with self.assertRaisesRegex(ValueError, "fresh run"):
                prepare_metrics_csv(path, experiment="TEST_RUN", dataset="IXI", epoch_start=0)
            with self.assertRaisesRegex(ValueError, "context mismatch"):
                _append(journal, 1, dataset="OASIS")
            with self.assertRaisesRegex(ValueError, "not contiguous"):
                _append(journal, 2)

    def test_default_training_cli_does_not_enable_tensorboard(self) -> None:
        parser = add_common_args(argparse.ArgumentParser())
        args = parser.parse_args([])
        self.assertEqual(args.use_tb, 0)
        self.assertEqual(args.tb_images_every, 0)
        with redirect_stderr(StringIO()), self.assertRaises(SystemExit):
            parser.parse_args(["--tb_images_every", "-1"])


if __name__ == "__main__":
    unittest.main()
