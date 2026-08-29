from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from tools.analysis import run_stage5
from tools.analysis.search.transaction import save_flow_npz_atomic
from tools.analysis.stage5.artifacts import field_record, file_record
from tools.analysis.stage5.contracts import canonical_sha256, write_immutable_json
from utils.cert_exact import certify_flow_exact


class ParserContractTest(unittest.TestCase):
    def test_expected_actions_are_present(self) -> None:
        parser = run_stage5.build_parser()
        subparsers = next(action for action in parser._actions if action.dest == "action")
        self.assertEqual(
            set(subparsers.choices),
            {
                "selfcheck",
                "disk-preflight",
                "prepare-data",
                "prepare-protocol",
                "smoke",
                "freeze-smoke",
                "train-u0",
                "materialize-source",
                "init-controller",
                "train-controller",
                "freeze-training",
                "decide",
                "freeze-decision",
                "evaluate",
                "freeze-evaluation",
                "aggregate",
                "finalize",
            },
        )

    def test_no_git_bypass_or_scientific_override_is_exposed(self) -> None:
        parser = run_stage5.build_parser()
        subparsers = next(action for action in parser._actions if action.dest == "action")
        exposed_options = {
            option
            for surface in (parser, *subparsers.choices.values())
            for action in surface._actions
            for option in action.option_strings
        }
        for forbidden in (
            "--skip-git",
            "--fixed-epoch",
            "--learning-rate",
            "--loss-weight",
            "--bootstrap-policy",
            "--dice-threshold",
            "--heldout-test",
        ):
            self.assertNotIn(forbidden, exposed_options)

    def test_invalid_shard_fails_before_dispatch(self) -> None:
        argv = [
            "materialize-source",
            "--repo-root",
            ".",
            "--expected-git-head",
            "a" * 40,
            "--protocol",
            "protocol.json",
            "--data-contract",
            "data.json",
            "--image-root",
            "images",
            "--smoke-barrier",
            "smoke-barrier.json",
            "--smoke-report",
            "smoke-report.json",
            "--device",
            "cuda:0",
            "--shard-index",
            "3",
            "--num-shards",
            "3",
            "--checkpoint-root",
            "checkpoints",
            "--source-root",
            "sources",
            "--seed",
            "0",
        ]
        with self.assertRaisesRegex(ValueError, "0 <= shard_index"):
            run_stage5.main(argv)


class GitBarrierTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        (self.root / ".git").mkdir()
        self.head = "a" * 40

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_clean_exact_head_passes(self) -> None:
        with patch.object(run_stage5, "_git", side_effect=(self.head, "")):
            self.assertEqual(run_stage5.assert_clean_exact_git(self.root, self.head), self.head)

    def test_wrong_head_fails(self) -> None:
        with (
            patch.object(run_stage5, "_git", return_value="b" * 40),
            self.assertRaisesRegex(RuntimeError, "Git HEAD mismatch"),
        ):
            run_stage5.assert_clean_exact_git(self.root, self.head)

    def test_dirty_tree_fails_and_has_no_bypass(self) -> None:
        with (
            patch.object(run_stage5, "_git", side_effect=(self.head, "?? unexpected.py")),
            self.assertRaisesRegex(RuntimeError, "dirty Git tree"),
        ):
            run_stage5.assert_clean_exact_git(self.root, self.head)


class SmokeBarrierTest(unittest.TestCase):
    def setUp(self) -> None:
        self.protocol = {
            "data_contract_sha256": "a" * 64,
            "u0_training_contract_sha256": "b" * 64,
            "controller_training_contract_sha256": "c" * 64,
            "bootstrap": {"policy": "collar_repair"},
        }
        transaction = {
            "status": "ACCEPTED",
            "returned_exact_status": "CERTIFIED",
            "requested_delta_rms": 0.1,
            "parameters_before_sha256": "d" * 64,
            "parameters_after_sha256": "e" * 64,
            "requested_array_sha256": "f" * 64,
            "candidate_array_sha256": "1" * 64,
            "returned_array_sha256": "2" * 64,
        }
        controller = {
            "checkpoint_sha256": "3" * 64,
            "reloaded_model_state_sha256": "4" * 64,
            "post_step_transaction": transaction,
        }
        self.report = {
            "schema": "ctcf-stage5-runtime-smoke-v1",
            "status": "PASS",
            "production_artifact": False,
            "accepted_production_checkpoint_written": False,
            "smoke_checkpoint_roundtrip": True,
            "git_head": "9" * 40,
            "protocol_sha256": canonical_sha256(self.protocol),
            "data_contract_sha256": "a" * 64,
            "u0_training_contract_sha256": "b" * 64,
            "controller_training_contract_sha256": "c" * 64,
            "seed": 0,
            "bootstrap_policy": "collar_repair",
            "u0_step": {
                "parameters_before_sha256": "5" * 64,
                "parameters_after_sha256": "6" * 64,
                "checkpoint_sha256": "7" * 64,
                "reloaded_model_state_sha256": "8" * 64,
            },
            "controller_steps": {"F24P": copy.deepcopy(controller), "A24P": copy.deepcopy(controller)},
        }

    def test_real_optimizer_and_transaction_attestations_are_required(self) -> None:
        run_stage5._validate_smoke_report(self.report, self.protocol, "9" * 40)
        altered = copy.deepcopy(self.report)
        altered["controller_steps"]["F24P"]["post_step_transaction"]["parameters_after_sha256"] = "d" * 64
        with self.assertRaisesRegex(RuntimeError, "changed no parameter"):
            run_stage5._validate_smoke_report(altered, self.protocol, "9" * 40)

    def test_report_is_bound_to_the_frozen_protocol(self) -> None:
        altered = copy.deepcopy(self.report)
        altered["protocol_sha256"] = "0" * 64
        with self.assertRaisesRegex(RuntimeError, "frozen protocol"):
            run_stage5._validate_smoke_report(altered, self.protocol, "9" * 40)


class ResumeAndArtifactTest(unittest.TestCase):
    def test_u0_existing_endpoint_is_passed_as_authoritative_resume(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            endpoint = root / "u0" / "seed_0" / "last.pth"
            endpoint.parent.mkdir(parents=True)
            endpoint.write_bytes(b"endpoint")
            args = argparse.Namespace(
                repo_root=root,
                expected_git_head="a" * 40,
                protocol=root / "protocol.json",
                data_contract=root / "data.json",
                image_root=root / "images",
                checkpoint_root=root,
                seed=0,
                device="cuda:0",
            )
            protocol = {
                "u0_training_contract_sha256": "b" * 64,
            }
            with (
                patch.object(run_stage5, "_protocol_context", return_value=("a" * 40, protocol)),
                patch.object(run_stage5, "canonical_sha256", return_value="c" * 64),
                patch.object(run_stage5, "train_u0", return_value=endpoint) as train,
            ):
                self.assertEqual(run_stage5.command_train_u0(args), 0)
            self.assertEqual(train.call_args.kwargs["resume"], endpoint)

    def test_existing_controller_initial_state_is_not_overwritten(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "controller_initial" / "seed_1" / "initial.pth"
            path.parent.mkdir(parents=True)
            path.write_bytes(b"state")
            path.with_name("initial.pth.sha256.json").write_text("{}", encoding="utf-8")
            args = argparse.Namespace(checkpoint_root=root, seed=1)
            with (
                patch.object(run_stage5, "_protocol_context", return_value=("a" * 40, {})),
                patch.object(run_stage5, "_verify_initial_controller"),
                patch.object(run_stage5, "initialize_controller_state") as initialize,
            ):
                self.assertEqual(run_stage5.command_init_controller(args), 0)
            initialize.assert_not_called()

    def test_artifact_path_escape_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            root.mkdir(exist_ok=True)
            record = {"root_id": "source_field_root", "relative_path": "../escape.npz"}
            with self.assertRaisesRegex(RuntimeError, "unsafe relative path"):
                run_stage5._resolve_artifact(record, {"source_field_root": root})

    def test_artifact_rewrite_with_restored_size_and_mtime_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "artifact.bin"
            original = b"A" * 64
            path.write_bytes(original)
            info = path.stat()
            record = {
                "root_id": "source_field_root",
                "relative_path": path.name,
                "bytes": len(original),
                "sha256": hashlib.sha256(original).hexdigest(),
            }
            run_stage5._resolve_artifact(record, {"source_field_root": root})
            path.write_bytes(b"B" * len(original))
            os.utime(path, ns=(info.st_atime_ns, info.st_mtime_ns))
            self.assertEqual(path.stat().st_size, info.st_size)
            self.assertEqual(path.stat().st_mtime_ns, info.st_mtime_ns)
            with self.assertRaisesRegex(RuntimeError, "bytes changed"):
                run_stage5._resolve_artifact(record, {"source_field_root": root})

    def test_exact_report_is_bound_to_the_persisted_candidate_array(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_root = root / "source"
            decision_root = root / "decision"
            source_root.mkdir()
            decision_root.mkdir()
            source_path = source_root / "source.npz"
            candidate_path = decision_root / "candidate.npz"
            flow = torch.zeros(1, 3, 5, 6, 7, dtype=torch.float32)
            save_flow_npz_atomic(source_path, flow)
            save_flow_npz_atomic(candidate_path, flow)
            source = field_record("source_field_root", source_root, source_path)
            candidate = field_record("decision_output_root", decision_root, candidate_path)
            performance = {
                "runtime_seconds": 1.0,
                "peak_memory_bytes": 1,
                "requested_delta_rms": 0.0,
                "candidate_delta_rms": 0.0,
                "returned_delta_rms": 0.0,
                "candidate_retained_ratio": None,
                "returned_retained_ratio": None,
            }
            execution = {"labels_loaded": False}
            certificate = certify_flow_exact(flow, eps="0.001")
            exact = {
                "schema": "ctcf-stage5-decision-exact-report-v1",
                "decision_id": "case__S0__F0",
                "source_field": source,
                "candidate_exact": certificate,
                "returned_exact": certificate,
                "clip_report": None,
                "execution": execution,
            }
            exact_path = decision_root / "exact.json"
            write_immutable_json(exact_path, exact)
            record = {
                "decision_id": exact["decision_id"],
                "certified_source_field": source,
                "requested_field": copy.deepcopy(candidate),
                "candidate_field": copy.deepcopy(candidate),
                "returned_field": copy.deepcopy(candidate),
                "exact_report": file_record("decision_output_root", decision_root, exact_path),
                "candidate_exact_status": certificate["status"],
                "candidate_exact_certified": certificate["certified"],
                "returned_exact_status": certificate["status"],
                "returned_certified": certificate["certified"],
                **performance,
                "execution_sha256": canonical_sha256({"environment": execution, "performance": performance}),
            }
            roots = {
                "source_field_root": source_root,
                "decision_output_root": decision_root,
            }
            run_stage5._verify_decision_artifacts(record, roots)

            altered = copy.deepcopy(record)
            altered["candidate_field"]["array_sha256"] = "f" * 64
            with self.assertRaisesRegex(RuntimeError, "candidate_field array differs"):
                run_stage5._verify_decision_artifacts(altered, roots)

            wrong_certificate = copy.deepcopy(certificate)
            wrong_certificate["sha256"] = "e" * 64
            exact["candidate_exact"] = wrong_certificate
            second_exact_path = decision_root / "wrong_exact.json"
            write_immutable_json(second_exact_path, exact)
            altered = copy.deepcopy(record)
            altered["exact_report"] = file_record(
                "decision_output_root",
                decision_root,
                second_exact_path,
            )
            with self.assertRaisesRegex(RuntimeError, "exact report differs"):
                run_stage5._verify_decision_artifacts(altered, roots)


class PackagingGuardTest(unittest.TestCase):
    @staticmethod
    def _args(root: Path) -> argparse.Namespace:
        return argparse.Namespace(
            repo_root=root,
            expected_git_head="a" * 40,
            run_root=root,
            run_id="S5_DEVELOPMENT_20260829T000000Z_aaaaaaaaaaaa",
            attempt_id="A_test",
            status="FAILED",
            exit_code=1,
            started_at_utc="2026-08-29T00:00:00Z",
            remote_heavy_locator="PENDING_UPLOAD",
        )

    def test_complete_requires_all_barriers_and_products(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = argparse.Namespace(
                repo_root=root,
                expected_git_head="a" * 40,
                run_root=root,
                run_id="S5_DEVELOPMENT_20260829T000000Z_aaaaaaaaaaaa",
                attempt_id="A_test",
                status="COMPLETE",
                exit_code=0,
                started_at_utc="2026-08-29T00:00:00Z",
                remote_heavy_locator="PENDING_UPLOAD",
            )
            with (
                patch.object(run_stage5, "assert_clean_exact_git", return_value="a" * 40),
                self.assertRaisesRegex(FileNotFoundError, "Cannot finalize COMPLETE"),
            ):
                run_stage5.command_finalize(args)

    def test_compact_root_rejects_heavy_or_label_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "leak.npz").write_bytes(b"heavy")
            args = argparse.Namespace(
                repo_root=root,
                expected_git_head="a" * 40,
                run_root=root,
                run_id="S5_DEVELOPMENT_20260829T000000Z_aaaaaaaaaaaa",
                attempt_id="A_test",
                status="FAILED",
                exit_code=1,
                started_at_utc="2026-08-29T00:00:00Z",
                remote_heavy_locator="PENDING_UPLOAD",
            )
            with (
                patch.object(run_stage5, "assert_clean_exact_git", return_value="a" * 40),
                self.assertRaisesRegex(RuntimeError, "heavy or label-bearing"),
            ):
                run_stage5.command_finalize(args)

    def test_manifest_describes_only_the_local_data_source(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch.object(run_stage5, "assert_clean_exact_git", return_value="a" * 40):
                run_stage5.command_finalize(self._args(root))
            manifest = json.loads((root / "manifests" / "A_test.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["data_source_scope"], "LOCAL_OASIS_L2R_ALL_PICKLES")
            self.assertIs(manifest["network_identity_lookup_performed"], False)
            self.assertNotIn("identity_transport_stopped_after_all394", manifest)

    def test_compact_root_rejects_links_when_the_platform_can_create_them(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "target.txt"
            target.write_text("target", encoding="utf-8")
            link = root / "link.txt"
            try:
                link.symlink_to(target)
            except OSError:
                self.skipTest("the local Windows account cannot create symlinks")
            with (
                patch.object(run_stage5, "assert_clean_exact_git", return_value="a" * 40),
                self.assertRaisesRegex(RuntimeError, "linked paths"),
            ):
                run_stage5.command_finalize(self._args(root))


class ShellContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.source = Path("tools/runners/train/stage5.sh").read_text(encoding="utf-8")

    def test_u0_seeds_come_from_the_seed_array_not_the_gpu_slot(self) -> None:
        self.assertIn("GPU_LIST:-0,1,2,3,4,5,6,7", self.source)
        self.assertIn("readonly -a SEEDS=(0 1 2)", self.source)
        self.assertIn("source_s${seed}_${shard}of${count}", self.source)
        block = self.source.split("train_u0_phase()", 1)[1].split("materialize_source_phase()", 1)[0]
        self.assertIn('for slot in "${!SEEDS[@]}"', block)
        self.assertIn('seed="${SEEDS[$slot]}"', block)
        self.assertIn('--seed "$seed"', block)
        # The slot index must not double as the seed: they coincide only for (0 1 2).
        self.assertNotIn("for slot in 0 1 2", self.source)
        self.assertNotIn('--seed "$slot"', self.source)

    def test_gpu_range_and_sharding_are_derived_from_the_seed_array(self) -> None:
        self.assertIn("${#GPUS[@]} -lt ${#SEEDS[@]}", self.source)
        self.assertIn("${#GPUS[@]} -gt 8", self.source)
        self.assertIn('seed_count="${#SEEDS[@]}"', self.source)
        self.assertIn("base=$((gpu_count / seed_count))", self.source)
        self.assertIn("remainder=$((gpu_count % seed_count))", self.source)
        self.assertNotIn("gpu_count / 3", self.source)

    def test_controller_matrix_uses_three_seed_waves(self) -> None:
        self.assertIn('for seed in "${SEEDS[@]}"', self.source)
        self.assertIn("start += ${#GPUS[@]}", self.source)
        self.assertIn('for slot in "${!GPUS[@]}"', self.source)
        self.assertIn("decision_queue", self.source)
        self.assertIn("physical_slot=$(((slot + seed) % ${#GPUS[@]}))", self.source)

    def test_controller_training_does_not_depend_on_persisted_training_fields(self) -> None:
        block = self.source.split("train_controller_phase()", 1)[1].split("decision_worker()", 1)[0]
        self.assertNotIn("--source-root", block)
        all_block = self.source.split("  all)\n", 1)[1]
        self.assertLess(all_block.index("train_controller_phase"), all_block.index("materialize_source_phase"))

    def test_logs_never_default_to_repository_root(self) -> None:
        self.assertIn('LOG_ROOT="$COMPACT_ROOT/logs/$ATTEMPT_ID"', self.source)
        self.assertIn('echo "[START] $log_file"', self.source)
        self.assertNotIn("> stage5.log", self.source)

    def test_dependencies_fail_before_any_long_gpu_phase(self) -> None:
        self.assertIn("dependency_preflight", self.source)
        self.assertIn("import mamba_ssm, numpy, torch", self.source)
        self.assertLess(self.source.index("dependency_preflight\n"), self.source.index("capture_provenance\n"))

    def test_heldout_test_is_never_a_configurable_input(self) -> None:
        self.assertNotIn("TEST20_ROOT", self.source)
        self.assertNotIn("--heldout", self.source)
        self.assertIn("Test20 is forbidden", self.source)

    def test_local_all_is_the_only_data_input_and_no_network_identity_step_exists(self) -> None:
        self.assertIn('OASIS_ALL_ROOT="${OASIS_ALL_ROOT:', self.source)
        self.assertNotIn("verify-oasis-identity", self.source)
        self.assertNotIn("neurite", self.source.lower())
        self.assertNotIn("nibabel", self.source.lower())
        self.assertIn(
            "for name in data_contract.json source_inventory.json split_manifest.json pair_manifest.json", self.source
        )
        self.assertIn('"$COMPACT_ROOT/data_attestations"', self.source)

    def test_complete_requires_preregistered_and_diagnostic_products(self) -> None:
        self.assertIn("evaluation_products_complete", self.source)
        self.assertIn("planned_contrasts.csv", self.source)
        self.assertIn("paired_effects_vs_u0.csv", self.source)
        self.assertIn("decision_diagnostics.csv", self.source)


if __name__ == "__main__":
    unittest.main()
