from __future__ import annotations

import argparse
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from tools.analysis import run_search_gate_c5 as runner
from tools.analysis.search_gate_c5 import (
    C5_DECISION_POLICY_SHA256,
    C5_POLICY_SHA256,
    INFERENCE_FAMILY_IDS,
)
from tools.analysis.search_gate_c5_contracts import EXPECTED_SUPPORT_CONTRACT, payload_sha256


class FrozenRunnerPayloadTest(unittest.TestCase):
    def test_every_runner_owned_payload_matches_its_literal_hash(self) -> None:
        payloads = runner._frozen_payloads()
        expected = {
            "full_policy": C5_POLICY_SHA256,
            "decision_policy": C5_DECISION_POLICY_SHA256,
            "arms": runner.ARM_SPECS_SHA256,
            "selectors": runner.SELECTOR_SPECS_SHA256,
            "offsets": runner.OFFSET_TABLE_SHA256,
            "support": runner.SUPPORT_CONTRACT_SHA256,
            "contrasts": runner.CONTRAST_CONTRACT_SHA256,
        }
        self.assertEqual({name: payload_sha256(value) for name, value in payloads.items()}, expected)
        self.assertEqual([row["arm_index"] for row in payloads["arms"]], list(range(36)))
        self.assertEqual([row["selector_index"] for row in payloads["selectors"]], list(range(5)))
        self.assertEqual([row["stride_voxels"] for row in payloads["offsets"]], [1, 2, 3, 4])
        self.assertTrue(all(len(row["offsets_zyx"]) == 27 for row in payloads["offsets"]))
        self.assertEqual(payloads["support"], EXPECTED_SUPPORT_CONTRACT)
        self.assertEqual(payloads["contrasts"]["family_ids"], list(INFERENCE_FAMILY_IDS))

    def test_selfcheck_is_machine_readable_and_test_115_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "selfcheck.json"
            self.assertEqual(runner.selfcheck_stage(argparse.Namespace(output=output)), 0)
            payload = json.loads(output.read_text(encoding="utf-8"))
        self.assertEqual(payload["status"], "PASS")
        self.assertTrue(payload["checks"]["test_115_is_not_authorized"])
        self.assertTrue(payload["checks"]["exact_c4_source_is_pinned"])
        self.assertEqual(payload["hashes"]["full_policy"], C5_POLICY_SHA256)
        self.assertEqual(payload["hashes"]["contrast_contract"], runner.CONTRAST_CONTRACT_SHA256)

    def test_dataset_inventory_is_tab_separated_and_ordered(self) -> None:
        records = {
            "atlas": {
                "dataset": "IXI",
                "split": "atlas",
                "case_id": "atlas",
                "path": "/atlas.pkl",
                "bytes": 1,
                "sha256": "a" * 64,
                "mtime_utc": "2026-08-24T00:00:00Z",
            },
            "subject_1": {
                "dataset": "IXI",
                "split": "val",
                "case_id": "subject_1",
                "path": "/subject_1.pkl",
                "bytes": 2,
                "sha256": "b" * 64,
                "mtime_utc": "2026-08-24T00:00:01Z",
            },
        }
        text = runner._dataset_tsv(records)
        self.assertEqual(text.splitlines()[0], "dataset\tsplit\tcase_id\tpath\tbytes\tsha256\tmtime_utc")
        self.assertEqual([line.split("\t")[2] for line in text.splitlines()[1:]], ["atlas", "subject_1"])


class DiskBudgetTest(unittest.TestCase):
    def test_new_run_requires_180_gib_and_resume_keeps_a_5_gib_floor(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "heavy" / "run"
            with (
                patch.object(runner.shutil, "disk_usage", return_value=SimpleNamespace(free=179 * 2**30)),
                self.assertRaisesRegex(RuntimeError, "requires 180.00 GiB"),
            ):
                runner._validate_disk_budget(target, runner.DEFAULT_MIN_FREE_GIB)

            target.mkdir(parents=True)
            with (
                patch.object(runner.shutil, "disk_usage", return_value=SimpleNamespace(free=10 * 2**30)),
                patch.object(runner, "_tree_bytes", return_value=170 * 2**30),
            ):
                runner._validate_disk_budget(target, runner.DEFAULT_MIN_FREE_GIB)
            with (
                patch.object(runner.shutil, "disk_usage", return_value=SimpleNamespace(free=4 * 2**30)),
                patch.object(runner, "_tree_bytes", return_value=176 * 2**30),
                self.assertRaisesRegex(RuntimeError, "resume lacks"),
            ):
                runner._validate_disk_budget(target, runner.DEFAULT_MIN_FREE_GIB)

    def test_invalid_disk_budget_is_refused(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "run"
            for value in (-1.0, float("nan"), float("inf")):
                with self.subTest(value=value), self.assertRaises(ValueError):
                    runner._validate_disk_budget(target, value)


class DecisionProcessIsolationTest(unittest.TestCase):
    def test_decision_stage_loads_only_the_label_free_contract(self) -> None:
        decision = {
            "git_head": "a" * 40,
            "runtime_signature": {"runtime": "frozen"},
            "num_shards": 1,
            "shard_to_physical_gpu": {"0": "2"},
            "shards": {"0": ["subject_1"]},
            "seed": 0,
        }
        args = argparse.Namespace(
            run_root=Path("run"),
            source_contract_sha256="b" * 64,
            decision_contract_sha256="c" * 64,
            shard_index=0,
            num_shards=1,
            gpu=0,
            physical_gpu="2",
            attempt_id="attempt",
        )
        with (
            patch.object(runner, "load_source_contract", side_effect=AssertionError("source contract was loaded")),
            patch.object(runner, "load_evaluation_contract", side_effect=AssertionError("evaluation was loaded")),
            patch.object(runner, "load_decision_contract_isolated", return_value=(decision, "c" * 64)) as isolated,
            patch.object(runner, "_assert_clean_code"),
            patch.object(runner, "_assert_runtime"),
            patch.object(runner, "setup_device", return_value=torch.device("cuda")),
            patch.object(torch.cuda, "get_device_name", return_value="H100"),
            patch.object(runner, "run_decision_worker", return_value=Path("worker.json")) as worker,
        ):
            self.assertEqual(runner.decision_worker_stage(args), 0)
        isolated.assert_called_once()
        call = worker.call_args.kwargs
        self.assertNotIn("source", call)
        self.assertNotIn("evaluation_contract", call)
        self.assertIs(call["decision"], decision)
        self.assertFalse(call["execution"]["labels_loaded_to_device"])

    def test_decision_barrier_does_not_open_the_label_bearing_source_contract(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_root = Path(directory)
            decision = {
                "git_head": "a" * 40,
                "runtime_signature": {"runtime": "frozen"},
                "num_shards": 1,
                "case_ids": ["subject_1"],
            }
            args = argparse.Namespace(
                run_root=run_root,
                source_contract_sha256="b" * 64,
                decision_contract_sha256="c" * 64,
                attempt_id="attempt",
            )
            barrier = {"schema": "barrier"}
            with (
                patch.object(runner, "_load_contract_pair", side_effect=AssertionError("source contract was loaded")),
                patch.object(runner, "_load_isolated_decision", return_value=(decision, "c" * 64)),
                patch.object(runner, "_assert_clean_code"),
                patch.object(runner, "_assert_runtime"),
                patch.object(runner, "build_decision_barrier", return_value=barrier) as build,
                patch.object(runner, "write_decision_barrier", return_value="d" * 64),
            ):
                self.assertEqual(runner.decision_barrier_stage(args), 0)
            self.assertEqual(build.call_args.kwargs["verify_heavy_bytes"], True)


class PostBarrierEvaluationTest(unittest.TestCase):
    def test_evaluation_contract_is_created_only_from_a_loaded_barrier(self) -> None:
        source = {"case_ids": ["subject_1"]}
        decision = {"git_head": "a" * 40, "runtime_signature": {"runtime": "frozen"}, "case_ids": ["subject_1"]}
        barrier = {"schema": "barrier"}
        evaluation = {"schema": "evaluation"}
        args = argparse.Namespace(
            run_root=Path("run"),
            source_contract_sha256="1" * 64,
            decision_contract_sha256="2" * 64,
            barrier_sha256="3" * 64,
        )
        order: list[str] = []

        def load_barrier(*args, **kwargs):
            order.append("barrier")
            return barrier, "3" * 64

        def load_full(*args, **kwargs):
            order.append("source")
            return source, decision, "1" * 64, "2" * 64

        with (
            patch.object(runner, "_load_isolated_decision", return_value=(decision, "2" * 64)),
            patch.object(runner, "_load_contract_pair", side_effect=load_full),
            patch.object(runner, "_assert_clean_code"),
            patch.object(runner, "_assert_runtime"),
            patch.object(runner, "load_decision_barrier", side_effect=load_barrier) as barrier_loader,
            patch.object(runner, "build_evaluation_contract", return_value=evaluation) as build,
            patch.object(runner, "write_evaluation_contract", return_value="4" * 64) as write,
        ):
            self.assertEqual(runner.freeze_evaluation_stage(args), 0)
        barrier_loader.assert_called_once()
        self.assertEqual(order, ["barrier", "source"])
        build.assert_called_once_with(source, "1" * 64, "2" * 64, barrier, "3" * 64)
        write.assert_called_once_with(Path("run"), evaluation)

    def test_cli_requires_evaluation_hash_for_label_stage_and_finalizer(self) -> None:
        parser = runner.build_parser()
        actions = next(action for action in parser._actions if action.dest == "action")
        self.assertEqual(
            set(actions.choices),
            {
                "selfcheck",
                "prepare",
                "decision-worker",
                "decision-barrier",
                "freeze-evaluation",
                "evaluation-worker",
                "finalize",
            },
        )
        prepare = parser.parse_args(
            [
                "prepare",
                "--run-root",
                "run",
                "--heavy-root",
                "heavy",
                "--source-c4-dir",
                "c4",
                "--source-c4-heavy-root",
                "c4-heavy",
                "--num-shards",
                "5",
                "--physical-gpus",
                "2,3,4,5,6",
            ]
        )
        self.assertEqual(prepare.min_free_gib, 180.0)


class ShellContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.path = Path("tools/runners/eval/search_gate_c5.sh")
        cls.raw = cls.path.read_bytes()
        cls.text = cls.raw.decode("utf-8")

    def test_shell_is_lf_and_keeps_operator_log_outside_git(self) -> None:
        self.assertNotIn(b"\r\n", self.raw)
        self.assertIn("/tmp/search_gate_c5.log", self.text)
        self.assertIn("Test-115 was not accessed", self.text)
        self.assertNotIn("subject_115", self.text)

    def test_shell_authenticates_exact_c4_and_retains_all_heavy_roots(self) -> None:
        self.assertIn("C4_DEVELOPMENT_20260824T161239Z_c69d12000176", self.text)
        self.assertIn("SOURCE_C4_DIR", self.text)
        self.assertIn("SOURCE_C4_HEAVY_ROOT", self.text)
        self.assertIn("Retain all three", self.text)
        self.assertNotIn("SOURCE_C3_DIR", self.text)

    def test_shell_freezes_evaluation_after_barrier_and_before_labels(self) -> None:
        order = [
            self.text.index("decision-barrier"),
            self.text.index("freeze-evaluation"),
            self.text.index("evaluation-worker"),
            self.text.rindex('"$PYBIN" -m tools.analysis.run_search_gate_c5 finalize'),
        ]
        self.assertEqual(order, sorted(order))
        self.assertIn('--evaluation-contract-sha256 "$EVALUATION_CONTRACT_SHA256"', self.text)

    def test_shell_uses_five_default_gpus_and_defensible_disk_budget(self) -> None:
        self.assertIn('GPU_LIST="${GPU_LIST:-2,3,4,5,6}"', self.text)
        self.assertIn('MIN_FREE_GIB="${MIN_FREE_GIB:-180}"', self.text)

    def test_shell_packages_only_compact_and_declares_no_checkpoints(self) -> None:
        self.assertIn('tar -czf "$package" -C "$OUT_ROOT" "$RUN_ID"', self.text)
        self.assertNotIn('-C "$HEAVY_OUT_ROOT"', self.text)
        self.assertIn("--expected-preflights 0 --no-strict-checkpoint-load", self.text)


if __name__ == "__main__":
    unittest.main()
