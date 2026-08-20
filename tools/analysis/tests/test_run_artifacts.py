from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tools.analysis.run_artifacts import (
    aggregate_summaries,
    classify_checkpoint_incompatibilities,
    finalize_run,
    validate_result_directory,
    write_dataset_manifest,
    write_output_index,
)


class RunArtifactsTest(unittest.TestCase):
    def test_output_index_keeps_nested_manifests(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            nested = root / "stage" / "run_manifest.json"
            nested.parent.mkdir()
            nested.write_text("nested\n", encoding="utf-8")
            (root / "run_manifest.json").write_text("root\n", encoding="utf-8")
            output = root / "outputs.tsv"

            count = write_output_index(root, output, {"outputs.tsv", "run_manifest.json"})

            self.assertEqual(count, 1)
            with output.open(encoding="utf-8", newline="") as stream:
                rows = list(csv.DictReader(stream, delimiter="\t"))
            self.assertEqual([row["relative_path"] for row in rows], ["stage/run_manifest.json"])

    def test_only_known_model_owned_legacy_buffer_is_allowed(self) -> None:
        compatible = classify_checkpoint_incompatibilities(
            ["st_half.grid"],
            [],
            ["st_full.grid", "st_half.grid"],
        )
        self.assertTrue(compatible.compatible)
        self.assertEqual(compatible.allowed_missing_buffers, ("st_half.grid",))

        unknown = classify_checkpoint_incompatibilities(
            ["st_half.grid", "level2.weight"],
            [],
            ["st_full.grid", "st_half.grid"],
        )
        self.assertFalse(unknown.compatible)
        self.assertEqual(unknown.disallowed_missing_keys, ("level2.weight",))

        not_a_buffer = classify_checkpoint_incompatibilities(
            ["st_half.grid"],
            [],
            ["st_full.grid"],
        )
        self.assertFalse(not_a_buffer.compatible)

        unexpected = classify_checkpoint_incompatibilities(
            [],
            ["obsolete.weight"],
            ["st_full.grid", "st_half.grid"],
        )
        self.assertFalse(unexpected.compatible)

    def test_command_line_modules_are_importable_from_repo_root(self) -> None:
        repo_root = Path(__file__).resolve().parents[3]
        for module in ("tools.analysis.run_artifacts", "tools.analysis.checkpoint_preflight"):
            result = subprocess.run(
                [sys.executable, "-m", module, "--help"],
                cwd=repo_root,
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)

    def test_dataset_manifest_and_result_validation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data = root / "data"
            data.mkdir()
            for name in ("case,1.pkl", "case2.pkl"):
                (data / name).write_bytes(b"fixture")

            datasets = root / "datasets.tsv"
            with patch(
                "experiments.core.path_profiles.get_dataset_paths",
                return_value={"val_dir": str(data)},
            ):
                write_dataset_manifest(3, ["OASIS:val"], datasets)

            result = root / "result"
            result.mkdir()
            (result / "per_case.csv").write_text("case_id,dice_mean\na,0.1\nb,0.2\n", encoding="utf-8")
            (result / "summary.csv").write_text("metric,mean\ndice_mean,0.15\n", encoding="utf-8")
            (result / "summary.json").write_text(json.dumps({"n_cases": 2}), encoding="utf-8")
            self.assertEqual(validate_result_directory(datasets, result, "OASIS", "val"), 2)

            with datasets.open(encoding="utf-8", newline="") as stream:
                paths = [row["path"] for row in csv.DictReader(stream, delimiter="\t")]
            self.assertIn(str(data / "case,1.pkl"), paths)

    def test_aggregate_quotes_csv_values(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            summary = root / "suite" / "run" / "summary.json"
            summary.parent.mkdir(parents=True)
            summary.write_text(
                json.dumps(
                    {
                        "model": "ctcf",
                        "ckpt_path": "path,with,commas.pth",
                        "test_dir": "/data/test",
                        "n_cases": 1,
                        "metrics": {
                            "dice_mean": {"mean": 0.5, "std": 0.0, "sem": 0.0, "ci95": 0.0, "min": 0.5, "max": 0.5}
                        },
                    }
                ),
                encoding="utf-8",
            )
            output = root / "aggregate.csv"
            aggregate_summaries(root, ["suite/*/summary.json"], 1, output)
            with output.open(encoding="utf-8", newline="") as stream:
                row = next(csv.DictReader(stream))
            self.assertEqual(row["ckpt_path"], "path,with,commas.pth")

    def test_complete_finalization_requires_strict_preflight(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "preflight").mkdir()
            for name in ("commands.sh", "datasets.tsv", "environment.txt", "git_status.txt"):
                (root / name).write_text("fixture\n", encoding="utf-8")
            preflight = root / "preflight" / "checkpoint.json"
            preflight.write_text(
                json.dumps(
                    {
                        "status": "PASS",
                        "checkpoint": "best.pth",
                        "sha256": "0" * 64,
                        "ctcf_config": "CTCF-CascadeA",
                        "time_steps": 6,
                        "ctcf_l3_svf": None,
                        "load": {"strict": True, "missing_keys": [], "unexpected_keys": []},
                    }
                ),
                encoding="utf-8",
            )
            args = argparse.Namespace(
                run_root=root,
                run_id="test",
                status="COMPLETE",
                exit_code=0,
                started_at="2026-01-01T00:00:00Z",
                completed_at="2026-01-01T00:01:00Z",
                git_head="a" * 40,
                branch="test",
                gpu_index=0,
                mode="test",
                paths_profile=3,
                seed=0,
                time_steps=6,
                expected_preflights=1,
                remote_locator="PENDING_UPLOAD",
            )
            manifest = finalize_run(args)
            self.assertEqual(json.loads(manifest.read_text(encoding="utf-8"))["status"], "COMPLETE")

            data = json.loads(preflight.read_text(encoding="utf-8"))
            data["load"]["missing_keys"] = ["missing.weight"]
            preflight.write_text(json.dumps(data), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "strict checkpoint preflights"):
                finalize_run(args)

            data["load"]["missing_keys"] = ["st_half.grid"]
            data["load"]["allowed_missing_buffers"] = ["st_half.grid"]
            preflight.write_text(json.dumps(data), encoding="utf-8")
            manifest = finalize_run(args)
            self.assertEqual(json.loads(manifest.read_text(encoding="utf-8"))["status"], "COMPLETE")

            data["load"]["missing_keys"] = ["missing.weight"]
            data["load"]["allowed_missing_buffers"] = ["missing.weight"]
            preflight.write_text(json.dumps(data), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "strict checkpoint preflights"):
                finalize_run(args)


if __name__ == "__main__":
    unittest.main()
