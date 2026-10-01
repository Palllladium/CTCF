from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from experiments.stage5 import runtime
from tools.analysis import compare_stage5_precision as comparison
from tools.analysis.run_artifacts import atomic_write_json, sha256_file
from tools.analysis.stage5 import checkpoint_sources as diagnostic


class ReadonlyCheckpointTest(unittest.TestCase):
    def test_valid_source_is_unchanged_and_recovery_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "initial.pth"
            path.write_bytes(b"checkpoint bytes")
            digest = sha256_file(path)
            atomic_write_json(
                runtime._checkpoint_sidecar_path(path),
                {
                    "file_name": path.name,
                    "bytes": path.stat().st_size,
                    "sha256": digest,
                },
            )
            self.assertEqual(diagnostic.require_readonly_checkpoint(path), digest)
            recovery = path.with_name(f".{path.name}.previous")
            recovery.write_bytes(b"retained recovery")
            with self.assertRaisesRegex(RuntimeError, "recovery required"):
                diagnostic.require_readonly_checkpoint(path)
            self.assertTrue(recovery.exists())
            self.assertEqual(sha256_file(path), digest)

    def test_corrupt_source_is_rejected_without_repair(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "initial.pth"
            path.write_bytes(b"changed")
            atomic_write_json(
                runtime._checkpoint_sidecar_path(path),
                {
                    "file_name": path.name,
                    "bytes": path.stat().st_size,
                    "sha256": "a" * 64,
                },
            )
            with self.assertRaisesRegex(RuntimeError, "mismatch"):
                diagnostic.require_readonly_checkpoint(path)
            self.assertEqual(path.read_bytes(), b"changed")


class ComparisonSourceTests(unittest.TestCase):
    def test_recovery_generation_is_rejected_before_a_mutating_loader(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            initial = root / "controller_initial/seed_0/initial.pth"
            initial.parent.mkdir(parents=True)
            initial.write_bytes(b"checkpoint")
            initial.with_name(".initial.pth.previous").write_bytes(b"recovery")
            source = comparison.Sources.__new__(comparison.Sources)
            source.args = SimpleNamespace(seed=0, checkpoint_root=root)
            source.device = torch.device("cpu")
            with (
                patch.object(comparison, "build_stage5_controller", return_value=torch.nn.Linear(1, 1)),
                patch.object(runtime, "_load_initial_controller", side_effect=AssertionError("loader mutated source")),
                self.assertRaisesRegex(RuntimeError, "recovery required"),
            ):
                source.new_step()
            self.assertTrue(initial.with_name(".initial.pth.previous").exists())
