from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from experiments.stage5 import runtime
from tools.analysis.run_artifacts import sha256_file
from tools.analysis.stage5 import pipeline
from tools.analysis.stage5.artifacts import load_canonical_json
from tools.analysis.stage5.contracts import (
    build_decision_barrier,
    build_training_barrier,
    canonical_sha256,
    validate_decision_barrier,
    write_immutable_json,
)
from tools.analysis.stage5.protocol import bootstrap_parameters
from tools.analysis.tests.stage5.test_contracts import _all_checkpoints, _protocol


class Stage5SourceDecisionHandoffTest(unittest.TestCase):
    def test_real_producer_is_consumed_by_decisions_without_weakening_provenance(self) -> None:
        # Only the external image/U0 inputs and CUDA requirement are replaced.
        # Source repair, stored certificates, case validation and decision writing
        # all use production code, including the consumer's own case inventory.
        protocol = _protocol(bootstrap_policy="collar_repair", bootstrap_parameters=bootstrap_parameters())
        training = build_training_barrier(protocol, _all_checkpoints(protocol))
        u0_sha = next(
            item["checkpoint_file"]["sha256"]
            for item in training["checkpoints"]
            if item["seed"] == 0 and item["variant_id"] == "U0"
        )
        cases = [
            {"case_id": "pair00_ab", "moving_subject_id": "A", "fixed_subject_id": "B"},
            {"case_id": "pair00_ba", "moving_subject_id": "B", "fixed_subject_id": "A"},
        ]
        shape = (18, 18, 18)
        store = SimpleNamespace(
            runtime=SimpleNamespace(pairs={"cases": cases}, contract_sha256=protocol["data_contract_sha256"]),
            image_shape=shape,
            load_image=lambda _: np.zeros(shape, dtype=np.float32),
        )
        raw = torch.zeros((1, 3, *shape), dtype=torch.float32)
        raw[:, 0, 8:10, 8:10, 8:10] = 0.1
        runner = SimpleNamespace(model=lambda moving, fixed, **kwargs: (moving, raw))
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            protocol_path = root / "protocol.json"
            training_path = root / "training.json"
            source_root = root / "sources"
            decision_root = root / "decisions"
            write_immutable_json(protocol_path, protocol)
            write_immutable_json(training_path, training)
            with (
                patch.object(runtime, "_require_cuda"),
                patch.object(runtime, "Stage5OasisImageStore", return_value=store),
                patch.object(runtime, "_verify_checkpoint_sidecar", return_value=u0_sha),
                patch.object(runtime, "load_frozen_u0", return_value=runner),
            ):
                count = runtime.materialize_source_fields(
                    data_contract=root / "unused-data-contract",
                    image_root=root / "unused-images",
                    output_root=source_root,
                    u0_checkpoint=root / "unused-u0.pth",
                    seed=0,
                    device=torch.device("cpu"),
                    protocol_sha256=canonical_sha256(protocol),
                    u0_training_contract_sha256=protocol["u0_training_contract_sha256"],
                    u0_config=runtime.U0TrainingConfig(),
                    bootstrap_policy="collar_repair",
                    shard_index=0,
                    num_shards=1,
                )
            self.assertEqual(count, len(cases))
            source_hashes = {path: sha256_file(path) for path in source_root.rglob("*") if path.is_file()}
            decision_args = {
                "protocol_path": protocol_path,
                "training_barrier_path": training_path,
                "data_contract_path": root / "unused-data-contract",
                "image_root": root / "unused-images",
                "checkpoint_root": root / "unused-checkpoints",
                "source_root": source_root,
                "decision_root": decision_root,
                "seed": 0,
                "variant": "U0",
                "shard_index": 0,
                "num_shards": 1,
                "device": torch.device("cpu"),
            }
            with patch.object(pipeline, "Stage5OasisImageStore", return_value=store):
                self.assertEqual(pipeline.materialize_decisions(**decision_args), len(cases))
                decision_hashes = {path: sha256_file(path) for path in decision_root.rglob("*") if path.is_file()}
                self.assertEqual(pipeline.materialize_decisions(**decision_args), len(cases))
            self.assertTrue(all(sha256_file(path) == digest for path, digest in source_hashes.items()))
            self.assertTrue(all(sha256_file(path) == digest for path, digest in decision_hashes.items()))
            records = [load_canonical_json(path) for path in sorted((decision_root / "records").glob("*.json"))]
            barrier = build_decision_barrier(protocol, training, records)
            validate_decision_barrier(barrier, protocol, training, require_complete=False)
            self.assertEqual(len(records), len(cases))
            for record in records:
                self.assertEqual(record["transaction_status"], "BASELINE_CERTIFIED")
                self.assertTrue(record["returned_certified"])
                self.assertFalse(record["labels_loaded"])
                self.assertEqual(record["returned_field"], record["certified_source_field"])

            development_case = runtime.development_case_inventory(store)[0]
            for name, case in (
                ("missing_split", cases[0]),
                ("wrong_subject", {**development_case, "moving_subject_id": "C"}),
                ("wrong_split", {**development_case, "split": "training"}),
            ):
                with self.subTest(name=name), self.assertRaisesRegex(RuntimeError, "provenance mismatch"):
                    pipeline._source_artifact(
                        source_root,
                        seed=0,
                        case=case,
                        expected_u0_sha256=u0_sha,
                        bootstrap_policy="collar_repair",
                        image_shape=shape,
                    )


if __name__ == "__main__":
    unittest.main()
