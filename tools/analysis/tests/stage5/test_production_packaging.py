from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

from tools.analysis import run_stage5
from tools.analysis.stage5.packaging import package_run_zip

RUN_ID = "S5_DEVELOPMENT_20260911T000000Z_aaaaaaaaaaaa"


class ProductionPackagingTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / RUN_ID
        self.root.mkdir()
        self.exports = Path(self.temp.name) / "exports"
        self.failure = self.root / "training_attestations/seed_0/F0/failures/step_7/failure.json"
        self.failure.parent.mkdir(parents=True)
        self.failure.write_text('{"status":"FAILED","heavy_capture":"capture.pth"}\n', encoding="utf-8")
        (self.root / "stage5.lock").touch()

    def package(self, attempt="A_first"):
        return package_run_zip(self.root, self.exports, root_name=RUN_ID, archive_stem=f"{RUN_ID}__{attempt}__FAILED")

    def verify(self, archive, digest):
        self.assertEqual(hashlib.sha256(archive.read_bytes()).hexdigest(), digest)
        self.assertEqual(archive.with_suffix(".zip.sha256").read_text(), f"{digest}  {archive.name}\n")
        with zipfile.ZipFile(archive) as package:
            self.assertIsNone(package.testzip())
            prefix = f"{RUN_ID}/"
            self.assertEqual({name.split("/")[0] for name in package.namelist()}, {RUN_ID})
            lines = package.read(prefix + "SHA256SUMS").decode("utf-8").splitlines()
            records = dict(line.split("  ", 1)[::-1] for line in lines)
            self.assertEqual(set(records), {name.removeprefix(prefix) for name in package.namelist()} - {"SHA256SUMS"})
            for name, expected in records.items():
                self.assertEqual(hashlib.sha256(package.read(prefix + name)).hexdigest(), expected)
            self.assertIn(prefix + self.failure.relative_to(self.root).as_posix(), package.namelist())
            self.assertFalse(any(name.endswith((".pth", ".tar", ".gz", ".zip")) for name in package.namelist()))

    def test_failed_attempt_contains_compact_failure_and_verified_hashes(self):
        self.verify(*self.package())
        self.assertFalse((self.root / "SHA256SUMS").exists())

    def test_later_attempt_gets_new_hashes_and_preserves_first_archive(self):
        first, first_digest = self.package()
        (self.root / "new_attempt.log").write_text("new attempt\n", encoding="utf-8")
        second, second_digest = self.package("A_second")
        self.verify(first, first_digest)
        self.verify(second, second_digest)
        self.assertNotEqual(first_digest, second_digest)
        with zipfile.ZipFile(first) as package:
            self.assertNotIn(f"{RUN_ID}/new_attempt.log", package.namelist())
        with self.assertRaises(FileExistsError):
            self.package()

    def test_heavy_captures_and_nested_archives_are_refused(self):
        for suffix in (".pth", ".npz", ".tar", ".gz", ".zip"):
            with self.subTest(suffix=suffix):
                path = self.root / f"forbidden{suffix}"
                path.write_bytes(b"fixture")
                with self.assertRaisesRegex(ValueError, "forbidden"):
                    self.package()
                path.unlink()
        self.assertFalse(self.exports.exists())

    def test_export_inside_run_and_unsafe_archive_names_are_refused(self):
        with self.assertRaisesRegex(ValueError, "outside"):
            package_run_zip(self.root, self.root / "exports", root_name=RUN_ID, archive_stem="A_bad")
        with self.assertRaisesRegex(ValueError, "Unsafe"):
            package_run_zip(self.root, self.exports, root_name="../escape", archive_stem="A_bad")

    def test_cli_requires_matching_finalized_attempt(self):
        manifests = self.root / "manifests"
        manifests.mkdir()
        manifest = {"run_id": RUN_ID, "attempt_id": "A_test", "status": "FAILED", "git_head": "a" * 40}
        (manifests / "A_test.json").write_text(json.dumps(manifest), encoding="utf-8")
        args = argparse.Namespace(
            repo_root=self.root,
            expected_git_head="a" * 40,
            run_root=self.root,
            run_id=RUN_ID,
            attempt_id="A_test",
            status="COMPLETE",
            export_root=self.exports,
        )
        with patch.object(run_stage5, "assert_clean_exact_git", return_value="a" * 40):
            with self.assertRaisesRegex(RuntimeError, "finalized attempt"):
                run_stage5.command_package(args)
            args.status = "FAILED"
            self.assertEqual(run_stage5.command_package(args), 0)
        self.assertEqual(len(list(self.exports.glob("*.zip"))), 1)


if __name__ == "__main__":
    unittest.main()
