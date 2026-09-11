from __future__ import annotations

import unittest
from dataclasses import asdict

from experiments.stage5.config import ControllerTrainingConfig, LegacyControllerTrainingConfig, U0TrainingConfig
from experiments.stage5.precision import controller_precision_contract
from tools.analysis.stage5.primitives import canonical_sha256
from tools.analysis.stage5.protocol import (
    controller_training_contract,
    legacy_controller_training_contract,
    u0_training_contract,
)


class Stage5TrainingPrecisionContractTest(unittest.TestCase):
    def test_frozen_u0_contract_and_historical_controller_bytes_are_preserved(self) -> None:
        # These hashes identify the documents in the original 68df5b run package.
        self.assertEqual(
            canonical_sha256(u0_training_contract(U0TrainingConfig())),
            "5e94e5a0ca66fc82d184f23b7cdbbc92691a3b0ebece9002b61f7369c4d2bfe4",
        )
        self.assertEqual(
            canonical_sha256(legacy_controller_training_contract(LegacyControllerTrainingConfig(), version=2)),
            "fc4d00f404ab8e7d4f2f0873b7aeda6ae889530ea495cc975d8077d609f7155d",
        )
        self.assertEqual(
            canonical_sha256(legacy_controller_training_contract(LegacyControllerTrainingConfig())),
            "4e5b21a792f6f5c1638a2d1334445de18a54ffa96de96a302116b14990d9b37b",
        )

    def test_precision_transition_does_not_change_controller_architecture_or_objective(self) -> None:
        old = legacy_controller_training_contract(LegacyControllerTrainingConfig())
        current = controller_training_contract(ControllerTrainingConfig())
        self.assertEqual(current.pop("schema"), "ctcf-stage5-controller-training-contract-v4")
        self.assertEqual(current.pop("precision"), controller_precision_contract())
        self.assertEqual(
            current.pop("numerical_failure_policy"),
            "FAIL_CLOSED_CAPTURE_STATE_NO_SKIPPED_OR_RETRIED_UPDATES",
        )
        self.assertEqual(
            current.pop("technical_recovery_policy"),
            "EXPLICIT_ACK_BOUND_TO_FAILURE_AND_LAST_COMPLETED_EPOCH",
        )
        self.assertEqual(current.pop("telemetry_schema"), "ctcf-stage5-parameter-telemetry-v1")
        old.pop("schema")
        old.pop("amp_overflow_policy")
        old["config"].pop("amp_initial_scale")
        old["config"].pop("amp_growth_interval")
        self.assertEqual(current, old)

    def test_production_contract_cannot_accept_legacy_amp_configuration(self) -> None:
        self.assertNotIn("amp_initial_scale", asdict(ControllerTrainingConfig()))
        with self.assertRaisesRegex(ValueError, "strict FP32"):
            controller_training_contract(LegacyControllerTrainingConfig())
        with self.assertRaisesRegex(ValueError, "frozen AMP configuration"):
            legacy_controller_training_contract(ControllerTrainingConfig())
        with self.assertRaisesRegex(ValueError, "AMP initial scale"):
            LegacyControllerTrainingConfig(amp_initial_scale=32768)


if __name__ == "__main__":
    unittest.main()
