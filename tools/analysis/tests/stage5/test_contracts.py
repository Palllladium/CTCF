from __future__ import annotations

import copy
import hashlib
import tempfile
import unittest
from pathlib import Path

from tools.analysis.stage5.contracts import (
    BASE_SEEDS,
    CHECKPOINT_SCHEMA,
    CHECKPOINT_SELECTION_POLICY,
    CONTROLLER_VARIANT_IDS,
    DECISION_RECORD_SCHEMA,
    EVALUATION_RECORD_SCHEMA,
    PLANNED_CONTRASTS,
    SEARCH_CHANNELS,
    VARIANT_IDS,
    VARIANT_SPECS,
    build_decision_barrier,
    build_evaluation_barrier,
    build_protocol_contract,
    build_training_barrier,
    canonical_json_bytes,
    canonical_sha256,
    load_json_object,
    validate_checkpoint_metadata,
    validate_decision_barrier,
    validate_evaluation_barrier,
    validate_protocol_contract,
    validate_root_access,
    validate_training_barrier,
    write_immutable_json,
)


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _file_record(
    value: str,
    *,
    field: bool = False,
    root_id: str | None = None,
) -> dict[str, object]:
    record: dict[str, object] = {
        "root_id": root_id or ("source_field_root" if field else "evaluation_output_root"),
        "relative_path": f"objects/{value}.{'npz' if field else 'json'}",
        "bytes": 128,
        "sha256": _digest(f"file:{value}"),
    }
    if field:
        record["array_sha256"] = _digest(f"array:{value}")
    return record


def _save_reload(field: dict[str, object]) -> dict[str, object]:
    return {
        "file_sha256": field["sha256"],
        "in_memory_array_sha256": field["array_sha256"],
        "reloaded_array_sha256": field["array_sha256"],
        "reloaded_from_persisted_bytes": True,
    }


def _protocol(*, bootstrap_policy: str = "identity", bootstrap_parameters=None):
    if bootstrap_parameters is None:
        bootstrap_parameters = {}
    return build_protocol_contract(
        git_head="a" * 40,
        data_contract_sha256=_digest("data"),
        u0_training_contract_sha256=_digest("u0-training"),
        controller_training_contract_sha256=_digest("controller-training"),
        search_contract_sha256=_digest("search"),
        directed_case_ids=("pair00_ab", "pair00_ba"),
        metric_ids=("OASIS_DICE_1_TO_35_V1", "CTCF_MATHEMATICAL_SDLOGJ_CROP2_V1"),
        u0_fixed_epoch=120,
        controller_fixed_epoch=40,
        bootstrap_policy=bootstrap_policy,
        bootstrap_parameters=bootstrap_parameters,
    )


def _checkpoint(
    protocol,
    *,
    seed: int,
    variant: str,
    base_sha: str | None = None,
) -> dict[str, object]:
    controller = variant != "U0"
    return {
        "schema": CHECKPOINT_SCHEMA,
        "checkpoint_id": f"S5_S{seed}_{variant}",
        "role": "CONTROLLER" if controller else "U0",
        "variant_id": variant,
        "seed": seed,
        "fixed_epoch": protocol["controller_fixed_epoch"] if controller else protocol["u0_fixed_epoch"],
        "selection_policy": CHECKPOINT_SELECTION_POLICY,
        "git_head": protocol["git_head"],
        "protocol_sha256": canonical_sha256(protocol),
        "data_contract_sha256": protocol["data_contract_sha256"],
        "training_contract_sha256": protocol[
            "controller_training_contract_sha256" if controller else "u0_training_contract_sha256"
        ],
        "checkpoint_file": _file_record(
            f"checkpoint-s{seed}-{variant}",
            root_id="checkpoint_root",
        ),
        "state_dict_sha256": _digest(f"state-s{seed}-{variant}"),
        "metrics_sha256": _digest(f"metrics-s{seed}-{variant}"),
        "base_checkpoint_sha256": base_sha if controller else None,
        "initial_controller_state_sha256": _digest(f"initial-controller-s{seed}") if controller else None,
        "source_contract_sha256": _digest(f"source-contract-s{seed}") if controller else None,
        "controller_parameter_count": 12345 if controller else 0,
    }


def _all_checkpoints(protocol) -> list[dict[str, object]]:
    result = []
    bases = {}
    for seed in BASE_SEEDS:
        base = _checkpoint(protocol, seed=seed, variant="U0")
        bases[seed] = base
        result.append(base)
    for seed in BASE_SEEDS:
        for variant in CONTROLLER_VARIANT_IDS:
            result.append(
                _checkpoint(
                    protocol,
                    seed=seed,
                    variant=variant,
                    base_sha=bases[seed]["checkpoint_file"]["sha256"],
                )
            )
    return result


def _decision_record(
    protocol,
    checkpoints,
    *,
    case_id: str,
    seed: int,
    variant: str,
    source: dict[str, object],
) -> dict[str, object]:
    checkpoint = checkpoints[(seed, variant)]
    if variant == "U0":
        requested = source
        candidate = source
        returned = copy.deepcopy(source)
        returned["relative_path"] = f"objects/returned-{case_id}-s{seed}-{variant}.npz"
        transaction_status = "BASELINE_CERTIFIED"
    else:
        requested = _file_record(
            f"requested-{case_id}-s{seed}-{variant}",
            field=True,
            root_id="decision_output_root",
        )
        candidate = _file_record(
            f"candidate-{case_id}-s{seed}-{variant}",
            field=True,
            root_id="decision_output_root",
        )
        returned = copy.deepcopy(candidate)
        returned["relative_path"] = f"objects/returned-{case_id}-s{seed}-{variant}.npz"
        transaction_status = "ACCEPTED"
    decision_id = f"{case_id}__S{seed}__{variant}"
    return {
        "schema": DECISION_RECORD_SCHEMA,
        "decision_id": decision_id,
        "case_id": case_id,
        "seed": seed,
        "variant_id": variant,
        "checkpoint_sha256": checkpoint["checkpoint_file"]["sha256"],
        "certified_source_field": source,
        "requested_field": requested,
        "candidate_field": candidate,
        "returned_field": returned,
        "requested_save_reload": _save_reload(requested),
        "candidate_save_reload": _save_reload(candidate),
        "returned_save_reload": _save_reload(returned),
        "exact_report": _file_record(f"exact-{decision_id}", root_id="decision_output_root"),
        "candidate_exact_status": "CERTIFIED",
        "candidate_exact_certified": True,
        "returned_exact_status": "CERTIFIED",
        "returned_certified": True,
        "transaction_status": transaction_status,
        "rollback_source_sha256_equal": False,
        "runtime_seconds": 1.25,
        "peak_memory_bytes": 1024,
        "requested_delta_rms": 0.0 if variant == "U0" else 1.0,
        "candidate_delta_rms": 0.0 if variant == "U0" else 0.8,
        "returned_delta_rms": 0.0 if variant == "U0" else 0.8,
        "candidate_retained_ratio": None if variant == "U0" else 0.8,
        "returned_retained_ratio": None if variant == "U0" else 0.8,
        "labels_loaded": False,
        "execution_sha256": _digest(f"execution-{decision_id}"),
    }


def _all_decisions(protocol, training_barrier) -> list[dict[str, object]]:
    checkpoints = {(row["seed"], row["variant_id"]): row for row in training_barrier["checkpoints"]}
    records = []
    for case_id in protocol["directed_case_ids"]:
        for seed in BASE_SEEDS:
            source = _file_record(
                f"source-{case_id}-s{seed}",
                field=True,
                root_id="source_field_root",
            )
            for variant in VARIANT_IDS:
                records.append(
                    _decision_record(
                        protocol,
                        checkpoints,
                        case_id=case_id,
                        seed=seed,
                        variant=variant,
                        source=source,
                    )
                )
    return records


def _evaluation_record(decision) -> dict[str, object]:
    decision_id = decision["decision_id"]
    return {
        "schema": EVALUATION_RECORD_SCHEMA,
        "evaluation_id": decision_id,
        "decision_id": decision_id,
        "decision_record_sha256": canonical_sha256(decision),
        "returned_field_sha256": decision["returned_field"]["sha256"],
        "label_source_sha256": _digest(f"labels-{decision['case_id']}"),
        "metrics_file": _file_record(f"metrics-{decision_id}", root_id="evaluation_output_root"),
        "labels_loaded_after_decision_barrier": True,
        "heldout_test_accessed": False,
        "execution_sha256": _digest(f"eval-execution-{decision_id}"),
    }


class CanonicalContractTest(unittest.TestCase):
    def test_canonical_json_is_order_independent_and_immutable(self) -> None:
        left = {"b": [2, 3], "a": 1}
        right = {"a": 1, "b": [2, 3]}
        self.assertEqual(canonical_json_bytes(left), canonical_json_bytes(right))
        self.assertEqual(canonical_sha256(left), canonical_sha256(right))
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "contract.json"
            digest = write_immutable_json(path, left)
            self.assertEqual(digest, write_immutable_json(path, right))
            self.assertEqual(load_json_object(path), right)
            with self.assertRaisesRegex(FileExistsError, "immutable"):
                write_immutable_json(path, {"a": 2})

    def test_non_finite_json_is_refused(self) -> None:
        with self.assertRaises(ValueError):
            canonical_json_bytes({"value": float("nan")})


class ProtocolTest(unittest.TestCase):
    def test_variant_and_channel_inventory_is_frozen_without_dice_threshold(self) -> None:
        protocol = _protocol()
        validate_protocol_contract(protocol)
        self.assertEqual([row["variant_id"] for row in protocol["variants"]], list(VARIANT_IDS))
        self.assertEqual(len(protocol["search_channels"]), 66)
        self.assertEqual(len(protocol["context_channels"]) + len(protocol["search_channels"]), 71)
        self.assertEqual(len(SEARCH_CHANNELS), len(set(SEARCH_CHANNELS)))
        self.assertEqual(protocol["expected_inventory"]["u0_checkpoints"], 3)
        self.assertEqual(protocol["expected_inventory"]["controller_checkpoints"], 24)
        self.assertEqual(
            [row["contrast_id"] for row in protocol["planned_contrasts"]],
            [contrast.contrast_id for contrast in PLANNED_CONTRASTS],
        )
        self.assertNotIn("dice_success_threshold", canonical_json_bytes(protocol).decode("utf-8"))
        self.assertEqual({spec.controller_family for spec in VARIANT_SPECS}, {"NONE", "FREE_RESIDUAL", "ATTENUATION"})

    def test_collar_repair_requires_all_frozen_parameters(self) -> None:
        parameters = {
            "collar_width": 4,
            "work_epsilon_decimal": "0.0011",
            "claim_epsilon_decimal": "0.001",
            "repair_operator_id": "CTCF_FIXED_COLLAR_REPAIR_V1",
            "repair_parameters": {"max_iterations": 100, "step": "0.01"},
        }
        protocol = _protocol(bootstrap_policy="collar_repair", bootstrap_parameters=parameters)
        self.assertEqual(protocol["bootstrap"]["parameters"], parameters)
        broken = copy.deepcopy(parameters)
        broken.pop("repair_operator_id")
        with self.assertRaisesRegex(RuntimeError, "freeze exactly"):
            _protocol(bootstrap_policy="collar_repair", bootstrap_parameters=broken)

    def test_identity_refuses_hidden_repair_parameters(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "must not carry"):
            _protocol(bootstrap_policy="identity", bootstrap_parameters={"collar_width": 4})
        with self.assertRaisesRegex(RuntimeError, "bootstrap_policy"):
            _protocol(bootstrap_policy="automatic")

    def test_loaded_protocol_still_validates(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "protocol.json"
            write_immutable_json(path, _protocol())
            validate_protocol_contract(load_json_object(path))

    def test_label_roots_are_phase_separated_by_exact_schema(self) -> None:
        validate_root_access(
            "training",
            {
                "image_root": "images",
                "checkpoint_root": "checkpoints",
                "source_field_root": "sources",
                "heavy_output_root": "heavy",
            },
        )
        validate_root_access(
            "evaluation",
            {
                "image_root": "images",
                "label_root": "labels",
                "source_field_root": "sources",
                "decision_output_root": "decisions",
                "evaluation_output_root": "evaluation",
            },
        )
        with self.assertRaisesRegex(RuntimeError, "extra=.*label_root"):
            validate_root_access(
                "decision",
                {
                    "image_root": "images",
                    "checkpoint_root": "checkpoints",
                    "source_field_root": "sources",
                    "decision_output_root": "decisions",
                    "label_root": "labels",
                },
            )
        with self.assertRaisesRegex(RuntimeError, "missing=.*label_root"):
            validate_root_access(
                "evaluation",
                {
                    "image_root": "images",
                    "source_field_root": "sources",
                    "decision_output_root": "decisions",
                    "evaluation_output_root": "evaluation",
                },
            )


class CheckpointAndTrainingBarrierTest(unittest.TestCase):
    def setUp(self) -> None:
        self.protocol = _protocol()
        self.checkpoints = _all_checkpoints(self.protocol)

    def test_complete_checkpoint_inventory_builds_computed_barrier(self) -> None:
        barrier = build_training_barrier(self.protocol, self.checkpoints)
        validate_training_barrier(barrier, self.protocol)
        self.assertEqual(barrier["status"], "COMPLETE")
        self.assertEqual(barrier["observed_checkpoints"], 27)

    def test_missing_checkpoint_is_incomplete_not_complete(self) -> None:
        barrier = build_training_barrier(self.protocol, self.checkpoints[:-1])
        self.assertEqual(barrier["status"], "INCOMPLETE")
        validate_training_barrier(barrier, self.protocol, require_complete=False)
        with self.assertRaisesRegex(RuntimeError, "incomplete"):
            validate_training_barrier(barrier, self.protocol)

    def test_validation_selected_checkpoint_is_rejected(self) -> None:
        broken = copy.deepcopy(self.checkpoints[0])
        broken["selection_policy"] = "BEST_DICE"
        with self.assertRaisesRegex(RuntimeError, "frozen endpoint"):
            validate_checkpoint_metadata(broken, self.protocol)

    def test_controller_must_reference_matching_u0(self) -> None:
        broken = copy.deepcopy(self.checkpoints)
        controller = next(row for row in broken if row["variant_id"] != "U0")
        controller["base_checkpoint_sha256"] = _digest("another-base")
        with self.assertRaisesRegex(RuntimeError, "another U0"):
            build_training_barrier(self.protocol, broken)

    def test_controller_capacity_and_initial_state_must_be_paired(self) -> None:
        broken = copy.deepcopy(self.checkpoints)
        controller = next(row for row in broken if row["seed"] == 0 and row["variant_id"] == "F2V")
        controller["controller_parameter_count"] += 1
        with self.assertRaisesRegex(RuntimeError, "identical parameter count"):
            build_training_barrier(self.protocol, broken)

        broken = copy.deepcopy(self.checkpoints)
        controller = next(row for row in broken if row["seed"] == 0 and row["variant_id"] == "F2V")
        controller["initial_controller_state_sha256"] = _digest("different-initial-state")
        with self.assertRaisesRegex(RuntimeError, "initial states differ"):
            build_training_barrier(self.protocol, broken)

        broken = copy.deepcopy(self.checkpoints)
        controller = next(row for row in broken if row["seed"] == 0 and row["variant_id"] == "F2V")
        controller["source_contract_sha256"] = _digest("different-source-contract")
        with self.assertRaisesRegex(RuntimeError, "source contracts differ"):
            build_training_barrier(self.protocol, broken)

    def test_unknown_checkpoint_field_fails_closed(self) -> None:
        broken = copy.deepcopy(self.checkpoints[0])
        broken["best_dice"] = 0.9
        with self.assertRaisesRegex(RuntimeError, "extra=.*best_dice"):
            validate_checkpoint_metadata(broken, self.protocol)

    def test_checkpoint_root_substitution_is_rejected(self) -> None:
        broken = copy.deepcopy(self.checkpoints[0])
        broken["checkpoint_file"]["root_id"] = "source_field_root"
        with self.assertRaisesRegex(RuntimeError, "wrong Stage 5 root role"):
            validate_checkpoint_metadata(broken, self.protocol)


class DecisionBarrierTest(unittest.TestCase):
    def setUp(self) -> None:
        self.protocol = _protocol()
        self.training = build_training_barrier(self.protocol, _all_checkpoints(self.protocol))
        self.records = _all_decisions(self.protocol, self.training)

    def test_complete_decision_inventory_is_label_free_and_computed(self) -> None:
        barrier = build_decision_barrier(self.protocol, self.training, self.records)
        validate_decision_barrier(barrier, self.protocol, self.training)
        self.assertEqual(barrier["status"], "COMPLETE")
        self.assertEqual(barrier["observed_decisions"], 54)
        self.assertFalse(barrier["labels_loaded"])

    def test_missing_decision_is_incomplete(self) -> None:
        barrier = build_decision_barrier(self.protocol, self.training, self.records[:-1])
        self.assertEqual(barrier["status"], "INCOMPLETE")
        validate_decision_barrier(barrier, self.protocol, self.training, require_complete=False)
        with self.assertRaisesRegex(RuntimeError, "incomplete"):
            validate_decision_barrier(barrier, self.protocol, self.training)

    def test_label_access_before_barrier_is_rejected(self) -> None:
        broken = copy.deepcopy(self.records)
        broken[0]["labels_loaded"] = True
        with self.assertRaisesRegex(RuntimeError, "accessed labels"):
            build_decision_barrier(self.protocol, self.training, broken)

    def test_rollback_must_have_source_sha_equality(self) -> None:
        broken = copy.deepcopy(self.records)
        record = next(row for row in broken if row["variant_id"] != "U0")
        record["transaction_status"] = "ROLLED_BACK"
        record["candidate_exact_status"] = "REJECTED"
        record["candidate_exact_certified"] = False
        record["rollback_source_sha256_equal"] = True
        record["returned_delta_rms"] = 0.0
        record["returned_retained_ratio"] = 0.0
        returned = copy.deepcopy(record["certified_source_field"])
        returned["relative_path"] = f"objects/rollback-{record['decision_id']}.npz"
        record["returned_field"] = returned
        record["returned_save_reload"] = _save_reload(returned)
        build_decision_barrier(self.protocol, self.training, broken)

        record["returned_field"]["sha256"] = _digest("not-source-bytes")
        record["returned_save_reload"] = _save_reload(record["returned_field"])
        with self.assertRaisesRegex(RuntimeError, "equal SHA-256"):
            build_decision_barrier(self.protocol, self.training, broken)

    def test_uncertified_return_is_rejected(self) -> None:
        broken = copy.deepcopy(self.records)
        broken[0]["returned_exact_status"] = "INCONCLUSIVE_RESOURCE_LIMIT"
        broken[0]["returned_certified"] = False
        with self.assertRaisesRegex(RuntimeError, "exactly certified after reload"):
            build_decision_barrier(self.protocol, self.training, broken)

    def test_candidate_status_and_save_reload_attestations_fail_closed(self) -> None:
        broken = copy.deepcopy(self.records)
        record = next(row for row in broken if row["variant_id"] != "U0")
        record["candidate_exact_status"] = "INCONCLUSIVE_RESOURCE_LIMIT"
        with self.assertRaisesRegex(RuntimeError, "status and certified flag disagree"):
            build_decision_barrier(self.protocol, self.training, broken)

        broken = copy.deepcopy(self.records)
        broken[0]["requested_save_reload"]["reloaded_array_sha256"] = _digest("changed-on-reload")
        with self.assertRaisesRegex(RuntimeError, "changed across save and reload"):
            build_decision_barrier(self.protocol, self.training, broken)

        broken = copy.deepcopy(self.records)
        broken[0]["candidate_save_reload"]["reloaded_from_persisted_bytes"] = False
        with self.assertRaisesRegex(RuntimeError, "reload from persisted bytes"):
            build_decision_barrier(self.protocol, self.training, broken)

        broken = copy.deepcopy(self.records)
        controller = next(row for row in broken if row["variant_id"] != "U0")
        controller["returned_retained_ratio"] = 0.2
        with self.assertRaisesRegex(RuntimeError, "arithmetically inconsistent"):
            build_decision_barrier(self.protocol, self.training, broken)

    def test_barrier_tamper_is_detected(self) -> None:
        barrier = build_decision_barrier(self.protocol, self.training, self.records)
        broken = copy.deepcopy(barrier)
        broken["decision_record_sha256"][self.records[0]["decision_id"]] = _digest("tampered")
        with self.assertRaisesRegex(RuntimeError, "computed status changed"):
            validate_decision_barrier(broken, self.protocol, self.training)

    def test_source_and_decision_root_substitutions_are_rejected(self) -> None:
        broken = copy.deepcopy(self.records)
        broken[0]["certified_source_field"]["root_id"] = "decision_output_root"
        with self.assertRaisesRegex(RuntimeError, "source field.*wrong.*root role"):
            build_decision_barrier(self.protocol, self.training, broken)

        broken = copy.deepcopy(self.records)
        accepted = next(row for row in broken if row["transaction_status"] == "ACCEPTED")
        accepted["returned_field"]["root_id"] = "source_field_root"
        with self.assertRaisesRegex(RuntimeError, "return the exactly certified candidate bytes"):
            build_decision_barrier(self.protocol, self.training, broken)


class EvaluationBarrierTest(unittest.TestCase):
    def setUp(self) -> None:
        self.protocol = _protocol()
        self.training = build_training_barrier(self.protocol, _all_checkpoints(self.protocol))
        decisions = _all_decisions(self.protocol, self.training)
        self.decision = build_decision_barrier(self.protocol, self.training, decisions)
        self.records = [_evaluation_record(row) for row in self.decision["records"]]

    def test_complete_evaluation_requires_post_barrier_labels(self) -> None:
        barrier = build_evaluation_barrier(self.protocol, self.training, self.decision, self.records)
        validate_evaluation_barrier(barrier, self.protocol, self.training, self.decision)
        self.assertEqual(barrier["status"], "COMPLETE")
        self.assertTrue(barrier["labels_loaded_after_decision_barrier"])

    def test_missing_evaluation_is_incomplete(self) -> None:
        barrier = build_evaluation_barrier(self.protocol, self.training, self.decision, self.records[:-1])
        self.assertEqual(barrier["status"], "INCOMPLETE")
        validate_evaluation_barrier(
            barrier,
            self.protocol,
            self.training,
            self.decision,
            require_complete=False,
        )
        with self.assertRaisesRegex(RuntimeError, "incomplete"):
            validate_evaluation_barrier(barrier, self.protocol, self.training, self.decision)

    def test_pre_barrier_label_claim_and_heldout_test_are_rejected(self) -> None:
        broken = copy.deepcopy(self.records)
        broken[0]["labels_loaded_after_decision_barrier"] = False
        with self.assertRaisesRegex(RuntimeError, "post-barrier"):
            build_evaluation_barrier(self.protocol, self.training, self.decision, broken)

        broken = copy.deepcopy(self.records)
        broken[0]["heldout_test_accessed"] = True
        with self.assertRaisesRegex(RuntimeError, "unauthorized held-out"):
            build_evaluation_barrier(self.protocol, self.training, self.decision, broken)

    def test_evaluation_cannot_substitute_another_returned_field(self) -> None:
        broken = copy.deepcopy(self.records)
        broken[0]["returned_field_sha256"] = _digest("another-field")
        with self.assertRaisesRegex(RuntimeError, "another returned field"):
            build_evaluation_barrier(self.protocol, self.training, self.decision, broken)

    def test_decision_barrier_tamper_invalidates_evaluation(self) -> None:
        barrier = build_evaluation_barrier(self.protocol, self.training, self.decision, self.records)
        tampered = copy.deepcopy(self.decision)
        tampered["status"] = "INCOMPLETE"
        with self.assertRaises(RuntimeError):
            validate_evaluation_barrier(barrier, self.protocol, self.training, tampered)

    def test_evaluation_root_substitution_is_rejected(self) -> None:
        broken = copy.deepcopy(self.records)
        broken[0]["metrics_file"]["root_id"] = "decision_output_root"
        with self.assertRaisesRegex(RuntimeError, "wrong Stage 5 root role"):
            build_evaluation_barrier(self.protocol, self.training, self.decision, broken)


if __name__ == "__main__":
    unittest.main()
