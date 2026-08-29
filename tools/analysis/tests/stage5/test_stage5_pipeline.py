from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from experiments.stage5.features import build_stage5_features
from experiments.stage5.runtime import ControllerTrainingConfig, U0TrainingConfig
from experiments.stage5.safety import commit_controller_delta, construct_initial_field, prepare_initial_field
from models.CTCF.controller import STAGE5_INPUT_CHANNEL_COUNT
from tools.analysis.search.transaction import load_flow_npz, save_flow_npz_atomic
from tools.analysis.stage5.artifacts import field_record, save_reload_attestation
from tools.analysis.stage5.protocol import controller_training_contract, search_contract, u0_training_contract


class Stage5FrozenSubcontractTest(unittest.TestCase):
    def test_training_contracts_are_label_free_fixed_endpoints(self) -> None:
        u0 = u0_training_contract(U0TrainingConfig())
        controller = controller_training_contract(ControllerTrainingConfig())
        self.assertEqual(u0["fixed_endpoint_epoch"], 500)
        self.assertEqual(controller["fixed_endpoint_epoch"], 100)
        self.assertFalse(u0["labels_reachable"])
        self.assertFalse(controller["labels_reachable"])
        self.assertFalse(u0["best_checkpoint_written"])
        self.assertFalse(controller["best_checkpoint_written"])
        self.assertFalse(u0["development_access_during_training"])
        self.assertFalse(controller["development_access_during_training"])
        self.assertIn("on_the_fly", controller["source_field_policy"])
        self.assertIn("epoch-specific deterministic perfect matching", controller["pair_schedule"])
        self.assertNotIn("dice", str(u0).lower())
        self.assertNotIn("dice", str(controller).lower())

    def test_search_contract_freezes_units_and_reserved_head(self) -> None:
        contract = search_contract()
        self.assertEqual(contract["input_channel_count"], STAGE5_INPUT_CHANNEL_COUNT)
        self.assertEqual(contract["vector_input_units"], "stride-normalized")
        self.assertEqual(contract["physical_proposal_units"], "full-resolution voxels")
        self.assertEqual(contract["reserved_head_semantics"], "literal zero, unused")
        self.assertTrue(contract["safety"]["rollback_is_source_byte_identity"])

    def test_search_feature_contract_disables_outer_autocast(self) -> None:
        shape = (18, 18, 18)
        fixed = torch.linspace(0.0, 1.0, 18**3, dtype=torch.float32).reshape(1, 1, *shape)
        moving = fixed.flip(-1).contiguous()
        psi = torch.zeros((1, 3, *shape), dtype=torch.float32)
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            features = build_stage5_features(fixed, moving, psi)
        tensors = (
            features.controller_input,
            features.fixed_normalized,
            features.moving_normalized,
            features.s2.posterior,
            features.s2.proposal,
            features.s4.posterior,
            features.s4.proposal,
        )
        self.assertTrue(all(value.dtype == torch.float32 for value in tensors))


class Stage5TransactionTest(unittest.TestCase):
    def test_persisted_source_uses_the_same_bootstrap_construction_as_training(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            raw = torch.zeros((1, 3, 18, 18, 18), dtype=torch.float32)
            raw[:, 0, 8:10, 8:10, 8:10] = 0.1
            phi, psi, _ = construct_initial_field(raw, policy="collar_repair")
            artifact = prepare_initial_field(raw, Path(temporary), policy="collar_repair")
            self.assertTrue(torch.equal(phi, load_flow_npz(artifact.phi_path)))
            self.assertTrue(torch.equal(psi, load_flow_npz(artifact.psi_path)))

    def test_zero_delta_is_saved_reloaded_and_accepted_without_return_copy(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source" / "initial_psi.npz"
            source.parent.mkdir()
            identity = torch.zeros((1, 3, 18, 18, 18), dtype=torch.float32)
            save_flow_npz_atomic(source, identity)
            transaction = commit_controller_delta(source, torch.zeros_like(identity), root / "decision")
            self.assertEqual(transaction.status, "ACCEPTED")
            self.assertEqual(transaction.returned_path, transaction.candidate_path)
            self.assertEqual(transaction.candidate_array_sha256, transaction.returned_array_sha256)
            self.assertEqual(transaction.candidate_exact_report["status"], "CERTIFIED")
            self.assertEqual(transaction.returned_exact_report["status"], "CERTIFIED")
            self.assertFalse((root / "decision" / "returned.npz").exists())

            record = field_record("source", root, source)
            attestation = save_reload_attestation(
                record,
                in_memory_array_sha256=record["array_sha256"],
                reloaded_path=source,
            )
            self.assertEqual(attestation["in_memory_array_sha256"], attestation["reloaded_array_sha256"])

    def test_failed_candidate_returns_the_exact_source_path(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "initial_psi.npz"
            identity = torch.zeros((1, 3, 18, 18, 18), dtype=torch.float32)
            save_flow_npz_atomic(source, identity)
            folded = identity.clone()
            folded[:, 0, 8] = 20.0
            clip_report = {"status": "synthetic-fold"}
            with patch(
                "experiments.stage5.safety.certified_local_clip_candidate",
                return_value=(folded, clip_report),
            ):
                transaction = commit_controller_delta(source, folded, root / "decision")
            self.assertEqual(transaction.status, "ROLLED_BACK")
            self.assertEqual(transaction.returned_path, source.resolve())
            self.assertTrue(transaction.rollback_byte_identical)
            self.assertFalse(transaction.candidate_exact_report["certified"])
            self.assertTrue(transaction.returned_exact_report["certified"])
            self.assertEqual(transaction.requested_path.parent, root / "decision")


if __name__ == "__main__":
    unittest.main()
