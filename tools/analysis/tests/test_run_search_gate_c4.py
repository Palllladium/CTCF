from __future__ import annotations

import argparse
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from tools.analysis import run_search_gate_c4 as runner
from tools.analysis.search_gate_c4 import C4_POLICY_SHA256
from tools.analysis.search_gate_c4_contracts import payload_sha256


class FrozenRunnerPayloadTest(unittest.TestCase):
    def test_all_runner_owned_payloads_match_literal_hashes(self) -> None:
        policy, arms, offsets, support = runner._frozen_payloads()
        self.assertEqual(payload_sha256(policy), C4_POLICY_SHA256)
        self.assertEqual(payload_sha256(arms), runner.ARM_SPECS_SHA256)
        self.assertEqual(payload_sha256(offsets), runner.OFFSET_TABLE_SHA256)
        self.assertEqual(payload_sha256(support), runner.SUPPORT_CONTRACT_SHA256)
        self.assertEqual([row["arm_index"] for row in arms], list(range(12)))
        self.assertEqual(support["collar_width"], 7)
        self.assertEqual(support["utility_id"], "COMMON_NCC7")

    def test_selfcheck_is_machine_readable_and_test_115_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "selfcheck.json"
            runner.selfcheck_stage(argparse.Namespace(output=output))
            payload = json.loads(output.read_text(encoding="utf-8"))
        self.assertEqual(payload["status"], "PASS")
        self.assertTrue(payload["checks"]["test_115_is_not_authorized"])
        self.assertEqual(payload["hashes"]["policy"], C4_POLICY_SHA256)

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


class DecisionProcessIsolationTest(unittest.TestCase):
    def test_decision_stage_never_loads_or_passes_source_contract(self) -> None:
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
            patch.object(runner, "load_source_contract", side_effect=AssertionError("source JSON was loaded")),
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
        self.assertIs(call["decision"], decision)
        self.assertFalse(call["execution"]["labels_loaded_to_device"])


class ShellContractTest(unittest.TestCase):
    def test_shell_keeps_logs_outside_git_and_test_split_closed(self) -> None:
        path = Path("tools/runners/eval/search_gate_c4.sh")
        text = path.read_text(encoding="utf-8")
        self.assertIn("/tmp/search_gate_c4.log", text)
        self.assertIn("Test-115 was not accessed", text)
        self.assertIn("SOURCE_C3_HEAVY_ROOT", text)
        self.assertIn("Neither heavy root was packaged or deleted", text)
        self.assertNotIn("subject_115", text)

    def test_shell_does_not_expose_a_false_seed_override(self) -> None:
        text = Path("tools/runners/eval/search_gate_c4.sh").read_text(encoding="utf-8")
        self.assertIn("SEED=0", text)
        self.assertNotIn('SEED="${SEED:-0}"', text)
        self.assertNotIn(" SEED=%q ", text)


if __name__ == "__main__":
    unittest.main()
