from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from tensorboard.compat.proto import event_pb2, summary_pb2
from tensorboard.summary.writer.event_file_writer import EventFileWriter

from tools.analysis.export_tensorboard_scalars import ExportError, _parse_args, _scan_source


class TensorBoardScalarExportTest(unittest.TestCase):
    def test_default_inventory_stays_inside_the_repository(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            repo_root = Path(temporary).resolve()
            args = _parse_args(["--repo-root", str(repo_root)])

            self.assertEqual(repo_root / "logs" / "tensorboard_inventory_20260820.csv", args.inventory)

    def test_scan_rejects_truncated_tfrecord_tail(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            repo_root = Path(temporary)
            log_dir = repo_root / "logs" / "run"
            writer = EventFileWriter(str(log_dir))
            writer.add_event(
                event_pb2.Event(
                    wall_time=1.25,
                    step=3,
                    summary=summary_pb2.Summary(
                        value=[summary_pb2.Summary.Value(tag="metric", simple_value=0.5)],
                    ),
                ),
            )
            writer.close()

            source = next(log_dir.glob("events.out.tfevents*"))
            complete = _scan_source((str(source), str(repo_root), 0, None))
            self.assertEqual(complete["event_records"], 2)
            self.assertEqual(complete["scalar_points"], 1)

            source.write_bytes(source.read_bytes()[:-1])
            with self.assertRaisesRegex(ExportError, "Corrupt or truncated TensorBoard TFRecord"):
                _scan_source((str(source), str(repo_root), 0, None))


if __name__ == "__main__":
    unittest.main()
