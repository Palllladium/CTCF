from __future__ import annotations

import argparse
import copy
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from tools.analysis import run_stage5
from tools.analysis.run_artifacts import sha256_file
from tools.analysis.stage5 import continuation
from tools.analysis.stage5.contracts import BASE_SEEDS, build_training_barrier, write_immutable_json
from tools.analysis.tests.stage5 import test_runtime
from tools.analysis.tests.stage5.test_contracts import _all_checkpoints, _protocol


class ContinuationBindingTest(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.paths = {name: self.root / name for name in (*continuation.FILE_NAMES, *continuation.ROOT_NAMES)}
        for name in continuation.ROOT_NAMES:
            self.paths[name].mkdir()
        self.protocol = _protocol()
        self.checkpoints = _all_checkpoints(self.protocol)
        self.training = build_training_barrier(self.protocol, self.checkpoints)
        write_immutable_json(self.paths["protocol"], self.protocol)
        write_immutable_json(self.paths["training_barrier"], self.training)
        self.paths["data_contract"].write_text("{}\n", encoding="utf-8")
        self.cases = [
            {"case_id": "pair00_ab", "moving_subject_id": "A", "fixed_subject_id": "B"},
            {"case_id": "pair00_ba", "moving_subject_id": "B", "fixed_subject_id": "A"},
        ]
        self.store = SimpleNamespace(
            runtime=SimpleNamespace(pairs={"cases": self.cases}, contract_sha256=self.protocol["data_contract_sha256"]),
            image_shape=(16, 16, 16),
        )
        self.output = self.root / "continuation.json"
        self.training_head = self.protocol["git_head"]
        self.execution_head = "b" * 40

    def prepare(self, *, source_validator=None, actual_checkpoints=None):
        if source_validator is None:

            def source_validator(path, **kwargs):
                path.mkdir(parents=True, exist_ok=True)
                write_immutable_json(path / "initial_report.json", {"report": {"psi_exact": {"interval_lo_min": 1.0}}})
                return {"case": kwargs["case"], "seed": kwargs["seed"]}

        with (
            patch.object(continuation, "verify_code_compatibility", return_value=[]) as code,
            patch.object(
                continuation,
                "collect_checkpoint_metadata",
                return_value=self.checkpoints if actual_checkpoints is None else actual_checkpoints,
            ) as checkpoints,
            patch.object(continuation, "Stage5OasisImageStore", return_value=self.store),
            patch.object(continuation, "validate_certified_source_artifact", side_effect=source_validator) as sources,
        ):
            result = continuation.prepare_continuation(
                repo_root=self.root,
                training_head=self.training_head,
                execution_head=self.execution_head,
                paths=self.paths,
                output=self.output,
            )
        code.assert_called_once_with(self.root, self.training_head, self.execution_head)
        checkpoints.assert_called_once_with(
            protocol_path=self.paths["protocol"], checkpoint_root=self.paths["checkpoint_root"]
        )
        self.assertEqual(sources.call_count, len(BASE_SEEDS) * len(self.cases))
        self.assertTrue(all(call.kwargs["case"]["split"] == "development" for call in sources.call_args_list))
        return result

    def validate(self, **kwargs):
        return continuation.validate_continuation(
            self.output,
            execution_head=self.execution_head,
            protocol_path=self.paths["protocol"],
            **kwargs,
        )

    def test_verified_binding_is_immutable_and_preserves_frozen_files(self) -> None:
        before = {name: sha256_file(self.paths[name]) for name in continuation.FILE_NAMES}
        payload = self.prepare()
        self.assertEqual(self.validate(supplied_paths=self.paths), payload)
        self.assertEqual(self.prepare(), payload)
        self.assertEqual({name: sha256_file(self.paths[name]) for name in continuation.FILE_NAMES}, before)
        self.assertEqual(payload["training_git_head"], self.training_head)
        self.assertEqual(payload["execution_git_head"], self.execution_head)
        self.assertFalse(payload["training_performed"])
        self.assertFalse(payload["labels_loaded"])

    def test_preparation_rejects_checkpoint_metadata_drift(self) -> None:
        changed = copy.deepcopy(self.checkpoints)
        changed[-1]["checkpoint_file"]["sha256"] = "c" * 64
        with self.assertRaisesRegex(RuntimeError, "checkpoint bytes or metadata"):
            self.prepare(actual_checkpoints=changed)
        self.assertFalse(self.output.exists())

    def test_preparation_rejects_training_head_data_and_case_drift(self) -> None:
        original_head = self.training_head
        self.training_head = "c" * 40
        with self.assertRaisesRegex(RuntimeError, "training HEAD"):
            self.prepare()
        self.training_head = original_head
        self.store.runtime.contract_sha256 = "c" * 64
        with self.assertRaisesRegex(RuntimeError, "data differs"):
            self.prepare()
        self.store.runtime.contract_sha256 = self.protocol["data_contract_sha256"]
        self.cases.reverse()
        with self.assertRaisesRegex(RuntimeError, "cases differ"):
            self.prepare()
        self.assertFalse(self.output.exists())

    def test_real_source_inventory_rejects_changed_field_bytes(self) -> None:
        validator = continuation.validate_certified_source_artifact
        u0_sha = {
            item["seed"]: item["checkpoint_file"]["sha256"] for item in self.checkpoints if item["variant_id"] == "U0"
        }
        for seed in BASE_SEEDS:
            for case in self.cases:
                test_runtime.CertifiedSourceTest._create_source(
                    self.paths["source_root"],
                    seed=seed,
                    case={**case, "split": "development"},
                    base_sha=u0_sha[seed],
                )
        payload = self.prepare(source_validator=validator)
        self.assertEqual(len(payload["source_inventory"]), 6)
        source = self.paths["source_root"] / "seed_0" / self.cases[0]["case_id"] / "initial_psi.npz"
        with source.open("ab") as stream:
            stream.write(b"changed")
        with self.assertRaisesRegex(RuntimeError, "bytes changed"):
            self.prepare(source_validator=validator)

    def test_validation_rejects_execution_head_and_root_substitution(self) -> None:
        self.prepare()
        with self.assertRaisesRegex(RuntimeError, "execution HEAD"):
            continuation.validate_continuation(
                self.output, execution_head="c" * 40, protocol_path=self.paths["protocol"]
            )
        for name in (*continuation.ROOT_NAMES, "data_contract", "training_barrier"):
            with self.subTest(name=name), self.assertRaisesRegex(RuntimeError, "path changed"):
                self.validate(supplied_paths={name: self.root / "substitute"})

    def test_validation_rejects_frozen_file_byte_changes(self) -> None:
        self.prepare()
        for name in continuation.FILE_NAMES:
            path = self.paths[name]
            original = path.read_bytes()
            path.write_bytes(original + b" ")
            with self.subTest(name=name), self.assertRaisesRegex(RuntimeError, "frozen file changed"):
                self.validate()
            path.write_bytes(original)

    def test_protocol_context_requires_explicit_continuation_after_head_change(self) -> None:
        self.prepare()
        args = argparse.Namespace(
            action="decide", repo_root=self.root, expected_git_head=self.execution_head, **self.paths
        )
        with patch.object(run_stage5, "assert_clean_exact_git", return_value=self.execution_head):
            with self.assertRaisesRegex(RuntimeError, "another Git HEAD"):
                run_stage5._protocol_context(args)
            args.continuation = self.output
            self.assertEqual(run_stage5._protocol_context(args), (self.execution_head, self.protocol))
            args.action = "train-controller"
            with self.assertRaisesRegex(RuntimeError, "restricted to post-training"):
                run_stage5._protocol_context(args)

    def test_continuation_does_not_bypass_exact_head_guard(self) -> None:
        self.prepare()
        args = argparse.Namespace(
            action="decide",
            repo_root=self.root,
            expected_git_head=self.execution_head,
            continuation=self.output,
            **self.paths,
        )
        with (
            patch.object(run_stage5, "assert_clean_exact_git", side_effect=RuntimeError("Git HEAD mismatch")),
            patch.object(run_stage5, "validate_continuation") as validation,
            self.assertRaisesRegex(RuntimeError, "Git HEAD mismatch"),
        ):
            run_stage5._protocol_context(args)
        validation.assert_not_called()

    def test_training_cli_does_not_expose_continuation(self) -> None:
        parser = run_stage5.build_parser()
        surfaces = next(action for action in parser._actions if action.dest == "action").choices
        allowed = continuation.POST_TRAINING_ACTIONS | {"finalize"}
        for name, surface in surfaces.items():
            options = {option for action in surface._actions for option in action.option_strings}
            with self.subTest(action=name):
                self.assertEqual("--continuation" in options, name in allowed)


class ContinuationCodeCompatibilityTest(unittest.TestCase):
    def test_safety_exception_preserves_bootstrap_globals_and_interface(self):
        original = "WORK_EPS = 0.0011\ndef construct_initial_field(x):\n return x\ndef commit_controller_delta(a, b, c):\n return a\n"
        fixed = original.replace(
            " return a",
            " from tools.analysis.stage5.work_margin import select_work_margin\n return select_work_margin(a)",
        )
        for modified, accepted in (
            (fixed, True),
            (fixed.replace("0.0011", "0.00101"), False),
            (fixed.replace("return x", "return x + 1"), False),
            (fixed.replace("(a, b, c)", "(a, b, c, d)"), False),
        ):
            outputs = [SimpleNamespace(stdout=original), SimpleNamespace(stdout=modified)]
            with self.subTest(accepted=accepted), patch.object(continuation.subprocess, "run", side_effect=outputs):
                if accepted:
                    continuation._verify_frozen_safety(Path("."), "a" * 40, "b" * 40)
                else:
                    with self.assertRaisesRegex(RuntimeError, "frozen safety"):
                        continuation._verify_frozen_safety(Path("."), "a" * 40, "b" * 40)

    def test_only_orchestration_changes_are_accepted_and_exact_head_remains_clean(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)

            def git(*args):
                return subprocess.run(
                    ["git", "-C", str(root), *args], check=True, capture_output=True, text=True
                ).stdout.strip()

            git("init", "--quiet")
            git("config", "user.name", "Stage5 test")
            git("config", "user.email", "stage5-test@example.invalid")
            math = root / "utils" / "operator.py"
            orchestration = root / "tools" / "analysis" / "stage5" / "pipeline.py"
            math.parent.mkdir(parents=True)
            orchestration.parent.mkdir(parents=True)
            math.write_text("VALUE = 1\n", encoding="utf-8")
            orchestration.write_text("# original\n", encoding="utf-8")
            git("add", ".")
            git("-c", "commit.gpgsign=false", "commit", "--quiet", "-m", "initial")
            training_head = git("rev-parse", "HEAD")
            orchestration.write_text("# corrected orchestration\n", encoding="utf-8")
            git("add", ".")
            git("-c", "commit.gpgsign=false", "commit", "--quiet", "-m", "orchestration")
            execution_head = git("rev-parse", "HEAD")
            self.assertEqual(
                continuation.verify_code_compatibility(root, training_head, execution_head),
                ["tools/analysis/stage5/pipeline.py"],
            )
            self.assertEqual(run_stage5.assert_clean_exact_git(root, execution_head), execution_head)
            with self.assertRaisesRegex(RuntimeError, "HEAD mismatch"):
                run_stage5.assert_clean_exact_git(root, training_head)
            math.write_text("VALUE = 2\n", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "dirty Git tree"):
                run_stage5.assert_clean_exact_git(root, execution_head)
            git("add", ".")
            git("-c", "commit.gpgsign=false", "commit", "--quiet", "-m", "changed math")
            changed_head = git("rev-parse", "HEAD")
            with self.assertRaisesRegex(RuntimeError, "frozen numerical/data code"):
                continuation.verify_code_compatibility(root, training_head, changed_head)
            with self.assertRaisesRegex(RuntimeError, "does not descend from training HEAD"):
                continuation.verify_code_compatibility(root, changed_head, training_head)
            with self.assertRaisesRegex(RuntimeError, "cannot relate the two revisions"):
                continuation.verify_code_compatibility(root, training_head, "0" * 40)


if __name__ == "__main__":
    unittest.main()
