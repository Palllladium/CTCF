from __future__ import annotations

import copy
import hashlib
import random
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch

from experiments.stage5.checkpoints import atomic_torch_save, build_training_state, load_training_state
from experiments.stage5.config import U0TrainingConfig
from experiments.stage5.runtime import _attach_runtime_checkpoint_metadata
from tools.analysis.run_artifacts import atomic_write_json, sha256_file
from tools.analysis.stage5.artifacts import checkpoint_metadata
from tools.analysis.stage5.contracts import CHECKPOINT_SELECTION_POLICY, build_protocol_contract
from tools.analysis.stage5.primitives import canonical_sha256, readable_json_bytes, write_immutable_json
from tools.analysis.stage5.protocol import bootstrap_parameters, u0_training_contract
from tools.analysis.stage5.u0_import import SOURCE_GIT_HEAD, import_completed_u0


class CompletedU0ImportTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.config = U0TrainingConfig()
        self.source_root = self.root / "source" / "checkpoints"
        self.target_root = self.root / "target" / "checkpoints"
        self.source_protocol_path = self.root / "source" / "protocol" / "protocol.json"
        self.target_protocol_path = self.root / "target" / "protocol" / "protocol.json"
        self.manifest_path = self.root / "target" / "u0_import.json"
        self.source = self._write_protocol(self.source_protocol_path, SOURCE_GIT_HEAD, old=True)
        self.target = self._write_protocol(self.target_protocol_path, "d" * 40, old=False)
        for seed in (0, 1, 2):
            self._write_source(seed)

    def _write_protocol(self, path: Path, head: str, *, old: bool) -> dict:
        # The import verifies all three sibling files; their content is not used
        # to instantiate a controller or load any images in this CPU-only test.
        contracts = {
            "u0_training_contract": u0_training_contract(self.config),
            "controller_training_contract": {"schema": "synthetic-controller-contract"},
            "search_contract": {"schema": "synthetic-search-contract"},
        }
        digests = {}
        for name, payload in contracts.items():
            contract_path = path.with_name(f"{name}.json")
            write_immutable_json(contract_path, payload)
            digests[f"{name}_sha256"] = sha256_file(contract_path)
        parameters = bootstrap_parameters()
        if old:
            parameters["repair_operator_id"] = "CTCF_DIGITAL_THEN_TRILINEAR_COLLAR_REPAIR_V1"
            parameters["repair_parameters"].pop("digital_residual_policy")
        protocol = build_protocol_contract(
            git_head=head,
            data_contract_sha256="a" * 64,
            **digests,
            directed_case_ids=("pair_forward", "pair_reverse"),
            metric_ids=("SYNTHETIC_METRIC",),
            u0_fixed_epoch=400,
            controller_fixed_epoch=100,
            bootstrap_policy="collar_repair",
            bootstrap_parameters=parameters,
        )
        write_immutable_json(path, protocol)
        return protocol

    def _checkpoint(self, seed: int, *, target: bool = False) -> Path:
        root = self.target_root if target else self.source_root
        return root / "u0" / f"seed_{seed}" / "last.pth"

    @staticmethod
    def _seal(path: Path, payload: dict, *, write_metrics: bool = True) -> None:
        digest = atomic_torch_save(path, payload)
        atomic_write_json(
            path.with_name("last.pth.sha256.json"),
            {
                "schema": "ctcf-stage5-checkpoint-sha256-v1",
                "file_name": "last.pth",
                "bytes": path.stat().st_size,
                "sha256": digest,
            },
        )
        if write_metrics:
            path.with_name("metrics.json").write_bytes(readable_json_bytes(payload["metrics_payload"]))

    def _write_source(self, seed: int) -> None:
        model = torch.nn.Linear(2, 2)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.config.learning_rate)
        model(torch.ones((1, 2))).sum().backward()
        optimizer.step()
        metrics = {
            "schema": "ctcf-stage5-u0-metrics-v1",
            "role": "U0",
            "variant": "U0",
            "seed": seed,
            "label_metrics_present": False,
            "selection_policy": CHECKPOINT_SELECTION_POLICY,
            "epochs": [
                {
                    "epoch": epoch,
                    "pairs": 294,
                    "learning_rate": 0.0001,
                    "pair_schedule_sha256": canonical_sha256([seed, epoch]),
                    "metrics": {"ncc": 0.5},
                }
                for epoch in range(1, 401)
            ],
        }
        rng = {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch_cpu": torch.get_rng_state(),
            "torch_cuda": [],
        }
        with mock.patch("experiments.stage5.checkpoints.capture_rng_state", return_value=rng):
            payload = build_training_state(
                role="U0",
                variant_id="U0",
                seed=seed,
                epoch_completed=400,
                fixed_epoch=400,
                git_head=SOURCE_GIT_HEAD,
                protocol_sha256=canonical_sha256(self.source),
                data_contract_sha256=self.source["data_contract_sha256"],
                training_contract_sha256=self.source["u0_training_contract_sha256"],
                model=model,
                optimizer=optimizer,
                scaler=torch.amp.GradScaler("cpu", enabled=False),
                pair_schedule_sha256=metrics["epochs"][-1]["pair_schedule_sha256"],
                metrics_sha256=hashlib.sha256(readable_json_bytes(metrics)).hexdigest(),
            )
        _attach_runtime_checkpoint_metadata(payload, config=self.config, metrics_payload=metrics)
        self._seal(self._checkpoint(seed), payload)

    def _import(self) -> dict:
        return import_completed_u0(
            source_protocol=self.source_protocol_path,
            target_protocol=self.target_protocol_path,
            source_checkpoint_root=self.source_root,
            target_checkpoint_root=self.target_root,
            output_manifest=self.manifest_path,
        )

    def _mutate_source(self, key: str, value) -> None:
        path = self._checkpoint(1)
        payload = torch.load(path, weights_only=False)
        payload[key] = value
        self._seal(path, payload)

    def test_import_preserves_training_state_and_source_bytes_and_is_idempotent(self) -> None:
        before = {str(path): sha256_file(path) for path in (self.root / "source").rglob("*") if path.is_file()}
        result = self._import()
        self.assertEqual(result["operation"], "IMPORT_COMPLETED_U0_WITHOUT_TRAINING")
        self.assertEqual(result["training_git_head"], SOURCE_GIT_HEAD)
        for seed in (0, 1, 2):
            source = torch.load(self._checkpoint(seed), weights_only=False)
            target = torch.load(self._checkpoint(seed, target=True), weights_only=False)
            self.assertEqual(target["git_head"], self.target["git_head"])
            self.assertEqual(target["training_git_head"], SOURCE_GIT_HEAD)
            self.assertEqual(
                target["u0_import_lineage"]["source_checkpoint"]["sha256"], sha256_file(self._checkpoint(seed))
            )
            self.assertEqual(target["metrics_payload"], source["metrics_payload"])
            self.assertEqual(target["model_state_sha256"], source["model_state_sha256"])
            self.assertTrue(torch.equal(target["model_state"]["weight"], source["model_state"]["weight"]))
            self.assertTrue(
                torch.equal(
                    target["optimizer_state"]["state"][0]["exp_avg"], source["optimizer_state"]["state"][0]["exp_avg"]
                )
            )
            self.assertTrue(np.array_equal(target["rng_state"]["numpy"][1], source["rng_state"]["numpy"][1]))
            self.assertEqual(
                result["source_checkpoints"][seed]["preserved_training_state_sha256"],
                result["target_checkpoints"][seed]["preserved_training_state_sha256"],
            )
        self.assertEqual(result, self._import())
        after = {str(path): sha256_file(path) for path in (self.root / "source").rglob("*") if path.is_file()}
        self.assertEqual(before, after)

    def test_imported_endpoint_loads_under_target_contract_and_enters_training_inventory(self) -> None:
        self._import()
        path = self._checkpoint(0, target=True)
        state = load_training_state(
            path,
            model=torch.nn.Linear(2, 2),
            optimizer=None,
            scaler=None,
            expected_role="U0",
            expected_variant="U0",
            expected_seed=0,
            expected_protocol_sha256=canonical_sha256(self.target),
            expected_data_contract_sha256=self.target["data_contract_sha256"],
            expected_training_contract_sha256=self.target["u0_training_contract_sha256"],
            restore_rng=False,
        )
        self.assertEqual(state["epoch_completed"], 400)
        metadata = checkpoint_metadata(
            checkpoint_id="S5_S0_U0",
            checkpoint_path=path,
            checkpoint_root=self.target_root,
            metrics_path=path.with_name("metrics.json"),
            protocol=self.target,
        )
        self.assertEqual(metadata["checkpoint_file"]["sha256"], sha256_file(path))

    def test_missing_source_metrics_are_restored_only_in_target(self) -> None:
        source_metrics = self._checkpoint(1).with_name("metrics.json")
        source_metrics.unlink()
        self._import()
        self.assertFalse(source_metrics.exists())
        self.assertTrue(self._checkpoint(1, target=True).with_name("metrics.json").is_file())

    def test_partial_endpoint_is_rejected_before_any_target_write(self) -> None:
        self._mutate_source("epoch_completed", 399)
        with self.assertRaisesRegex(RuntimeError, "completed-endpoint"):
            self._import()
        self.assertFalse(self.target_root.exists())

    def test_checkpoint_byte_corruption_is_rejected(self) -> None:
        with self._checkpoint(2).open("ab") as stream:
            stream.write(b"corruption")
        with self.assertRaisesRegex(RuntimeError, "bytes or sidecar"):
            self._import()
        self.assertFalse(self.target_root.exists())

    def test_resealed_changed_model_digest_is_rejected(self) -> None:
        self._mutate_source("model_state_sha256", "f" * 64)
        with self.assertRaisesRegex(RuntimeError, "model-state digest"):
            self._import()

    def test_resealed_changed_runtime_config_is_rejected(self) -> None:
        self._mutate_source("training_config_sha256", "f" * 64)
        with self.assertRaisesRegex(RuntimeError, "training configuration"):
            self._import()

    def test_resealed_changed_embedded_metrics_is_rejected(self) -> None:
        self._mutate_source("metrics_payload_sha256", "f" * 64)
        with self.assertRaisesRegex(RuntimeError, "embedded metrics digest"):
            self._import()

    def test_changed_external_metrics_are_rejected_without_rewriting_source(self) -> None:
        path = self._checkpoint(1).with_name("metrics.json")
        path.write_bytes(b"{}\n")
        with self.assertRaisesRegex(RuntimeError, "external metrics"):
            self._import()
        self.assertEqual(path.read_bytes(), b"{}\n")

    def test_protocol_differences_outside_bootstrap_hotfix_are_rejected(self) -> None:
        changed = copy.deepcopy(self.target)
        changed["data_contract_sha256"] = "f" * 64
        atomic_write_json(self.target_protocol_path, changed)
        with self.assertRaisesRegex(RuntimeError, "beyond git_head"):
            self._import()

    def test_referenced_training_contract_corruption_is_rejected(self) -> None:
        self.source_protocol_path.with_name("u0_training_contract.json").write_text("{}", encoding="utf-8")
        with self.assertRaisesRegex(RuntimeError, "file digest mismatch"):
            self._import()

    def test_recursive_import_is_rejected(self) -> None:
        self._mutate_source("training_git_head", SOURCE_GIT_HEAD)
        with self.assertRaisesRegex(RuntimeError, "recursive imports"):
            self._import()

    def test_partial_target_without_manifest_is_rejected(self) -> None:
        self.target_root.mkdir(parents=True)
        (self.target_root / "unclaimed.pth").write_bytes(b"partial")
        with self.assertRaisesRegex(RuntimeError, "nonempty target"):
            self._import()

    def test_imported_optimizer_cannot_change_even_with_a_new_valid_sidecar(self) -> None:
        self._import()
        path = self._checkpoint(1, target=True)
        payload = torch.load(path, weights_only=False)
        payload["optimizer_state"]["state"][0]["exp_avg"].add_(1)
        self._seal(path, payload)
        with self.assertRaisesRegex(RuntimeError, "preserved model, optimizer"):
            self._import()

    def test_existing_import_refuses_missing_target_metrics(self) -> None:
        self._import()
        self._checkpoint(1, target=True).with_name("metrics.json").unlink()
        with self.assertRaisesRegex(RuntimeError, "target metrics file is missing"):
            self._import()

    def test_existing_import_refuses_changed_lineage(self) -> None:
        self._import()
        path = self._checkpoint(1, target=True)
        payload = torch.load(path, weights_only=False)
        payload["u0_import_lineage"]["source_checkpoint"]["sha256"] = "f" * 64
        self._seal(path, payload)
        with self.assertRaisesRegex(RuntimeError, "training lineage"):
            self._import()


if __name__ == "__main__":
    unittest.main()
