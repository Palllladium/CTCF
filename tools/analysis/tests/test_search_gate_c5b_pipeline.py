from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from tools.analysis.run_artifacts import atomic_write_json, sha256_file
from tools.analysis.search_gate_c5b import (
    ANCHOR_ARM_IDS,
    ARM_SPECS,
    BRANCH_FREEZE_SUPERIOR,
    C5B_POLICY_SHA256,
    DIAGNOSTIC_ARM_ID,
    EVALUATION_LABEL_IDS,
    EXPECTED_CASE_COUNT,
    PROTOCOL_ID,
    REFERENCE_ARM_ID,
    SELECTABLE_ARM_IDS,
)
from tools.analysis.search_gate_c5b_contracts import (
    DECISION_CASE_SCHEMA,
    DECISION_SCHEMA,
    EVALUATION_CASE_SCHEMA,
    EVALUATION_CONTRACT_SCHEMA,
    WORKER_SCHEMA,
    build_evaluation_barrier,
    load_decision_contract_isolated,
    load_evaluation_barrier,
    validate_decision_case_marker,
    validate_evaluation_case_marker,
    validate_worker_marker,
    verify_field_record,
)
from tools.analysis.search_gate_c5b_workers import finalize_c5b
from tools.analysis.search_gate_metrics import DETJ_DIAGNOSTICS, DIGITAL_DECOMPOSITION, METRIC_SPECS

SHA_A = "a" * 64
SHA_B = "b" * 64


def geometry(sdlogj: float = 0.3) -> dict[str, dict[str, object]]:
    rows: dict[str, dict[str, object]] = {}
    for metric_id in METRIC_SPECS:
        rows[metric_id] = {
            "metric_id": metric_id,
            "status": "OK",
            "value": sdlogj,
            "components": {},
            "detail": None,
            "error_type": None,
        }
    rows[DETJ_DIAGNOSTICS]["value"] = None
    rows[DETJ_DIAGNOSTICS]["components"] = {
        "voxels": 100.0,
        "detj_min": 0.1,
        "detj_max": 2.0,
        "nonfinite_count": 0.0,
        "nonfinite_fraction": 0.0,
        "nonpositive_count": 0.0,
        "nonpositive_fraction": 0.0,
        "invalid_count": 0.0,
        "invalid_fraction": 0.0,
    }
    rows[DIGITAL_DECOMPOSITION]["value"] = 0.0
    rows[DIGITAL_DECOMPOSITION]["components"] = {
        "voxels": 100.0,
        "corner_union_violation_fraction": 0.0,
        "jstar_union_violation_fraction": 0.0,
        "union_violation_count": 0.0,
        "union_violation_fraction": 0.0,
        "sum_of_component_fractions": 0.0,
        **{
            f"corner_{token}_violation_count": 0.0 for token in ("mmm", "mmp", "mpm", "mpp", "pmm", "pmp", "ppm", "ppp")
        },
    }
    return rows


def field(root_id: str, arm_id: str) -> dict[str, str]:
    return {
        "root_id": root_id,
        "relative_path": f"cases/subject_1/arms/{arm_id}.npz",
        "npz_sha256": SHA_A,
        "array_sha256": SHA_B,
    }


def decision_contract() -> dict[str, object]:
    source_anchors = {
        "c4_reference_s2_a10_b0": {"field": field("source_c4_heavy", REFERENCE_ARM_ID)},
        "c5_s4_a10_b0_sweep1": {"field": field("source_c5_heavy", ANCHOR_ARM_IDS[1])},
        "c5_s4_a20_b0_sweep1": {"field": field("source_c5_heavy", ANCHOR_ARM_IDS[2])},
    }
    return {
        "schema": DECISION_SCHEMA,
        "protocol_id": PROTOCOL_ID,
        "status": "FROZEN_LABEL_FREE",
        "policy_sha256": C5B_POLICY_SHA256,
        "source_contract_sha256": SHA_A,
        "case_ids": ["subject_1"],
        "shards": {"0": ["subject_1"]},
        "shard_to_physical_gpu": {"0": "2"},
        "source_anchors": {"subject_1": source_anchors},
        "roots": {
            "source_c3_heavy": "/c3",
            "source_c4_heavy": "/c4",
            "source_c5_heavy": "/c5",
            "target_c5b_heavy": "/c5b",
        },
        "test_115_authorized": False,
        "test_split_accessed": False,
    }


def decision_marker() -> dict[str, object]:
    direction_sha, post_sha = "c" * 64, "d" * 64
    rows = []
    for spec in ARM_SPECS:
        owner = (
            "source_c4_heavy"
            if spec.arm_id == REFERENCE_ARM_ID
            else "source_c5_heavy"
            if spec.arm_id in ANCHOR_ARM_IDS[1:]
            else "target_c5b_heavy"
        )
        proposal = {}
        parity = {"array_byte_identical": True} if spec.arm_id == REFERENCE_ARM_ID else None
        if spec.stride_voxels == 4:
            proposal = {
                "preclip_direction_array_sha256": direction_sha,
                "postprocessed_direction_array_sha256": post_sha,
                "amplitude_stage": "after_rms_match_before_local_clip",
                "post_rms_amplitude": spec.post_rms_amplitude,
                "local_clip_sweeps": spec.local_clip_sweeps,
                "rms_source": 1.0,
                "rms_target": 1.0,
                "rms_matched": 1.0,
                "rms_scale_factor": 1.0,
                "rms_requested": spec.post_rms_amplitude,
                "rms_realized": spec.post_rms_amplitude * 0.97,
                "clip_rms_retention": 0.97,
                "clip_cosine": 1.0,
                "operator": {
                    "operator": "CERTIFIED_LOCAL_CLIP",
                    "work_eps": 0.0011,
                    "sweeps": spec.local_clip_sweeps,
                    "current_fast_cert_bound": 0.0012,
                    "output_fast_cert_bound": 0.0012,
                },
            }
        if spec.arm_id in ANCHOR_ARM_IDS[1:]:
            parity = {
                "array_byte_identical": True,
                "source_array_sha256": SHA_B,
                "replayed_array_sha256": SHA_B,
            }
        rows.append(
            {
                "arm_index": spec.arm_index,
                "arm_id": spec.arm_id,
                "role": spec.role,
                "selectable": spec.selectable,
                "post_rms_amplitude": spec.post_rms_amplitude,
                "local_clip_sweeps": spec.local_clip_sweeps,
                "proposal": proposal,
                "source_parity": parity,
                "candidate_field": field(owner, spec.arm_id),
                "exact": {
                    "status": "CERTIFIED",
                    "certified": True,
                    "sha256": SHA_B,
                    "epsilon_decimal": "0.001",
                },
                "observed_fold_count": 0,
                "geometry": geometry(),
            }
        )
    return {
        "schema": DECISION_CASE_SCHEMA,
        "protocol_id": PROTOCOL_ID,
        "status": "COMPLETE",
        "strict": True,
        "case_id": "subject_1",
        "shard_index": 0,
        "physical_gpu": "2",
        "decision_contract_sha256": SHA_B,
        "labels_loaded_to_device": False,
        "test_split_accessed": False,
        "s4_preclip_direction_array_sha256": direction_sha,
        "s4_postprocessed_array_sha256": post_sha,
        "arms": rows,
        "execution": {
            "phase": "decision",
            "attempt_id": "A_TEST",
            "shard_index": 0,
            "physical_gpu": "2",
            "labels_loaded_to_device": False,
            "deterministic": True,
        },
    }


class IsolatedDecisionLoaderTest(unittest.TestCase):
    def test_decision_loader_hashes_but_does_not_parse_label_bearing_source(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source_contract.json"
            source.write_text("this is deliberately not JSON", encoding="utf-8")
            source_sha = sha256_file(source)
            decision = {
                "schema": DECISION_SCHEMA,
                "protocol_id": PROTOCOL_ID,
                "status": "FROZEN_LABEL_FREE",
                "policy_sha256": C5B_POLICY_SHA256,
                "source_contract_sha256": source_sha,
                "case_ids": [],
                "test_115_authorized": False,
                "test_split_accessed": False,
            }
            atomic_write_json(root / "decision_contract.json", decision)
            source.unlink()
            observed = load_decision_contract_isolated(root, sha256_file(root / "decision_contract.json"))
            self.assertEqual(observed, decision)


class DecisionMarkerContractTest(unittest.TestCase):
    def test_exact_marker_passes_and_mapping_order_is_irrelevant(self) -> None:
        contract, marker = decision_contract(), decision_marker()
        validate_decision_case_marker(marker, contract, SHA_B, verify_heavy_bytes=False)
        reordered = dict(reversed(tuple(marker.items())))
        validate_decision_case_marker(reordered, contract, SHA_B, verify_heavy_bytes=False)

    def test_arm_semantics_and_shared_direction_fail_closed(self) -> None:
        contract = decision_contract()
        mutations = []
        wrong_amplitude = decision_marker()
        wrong_amplitude["arms"][3]["post_rms_amplitude"] = 1.4
        mutations.append(wrong_amplitude)
        wrong_direction = decision_marker()
        wrong_direction["arms"][4]["proposal"]["preclip_direction_array_sha256"] = "e" * 64
        mutations.append(wrong_direction)
        wrong_sweeps = decision_marker()
        wrong_sweeps["arms"][-1]["proposal"]["operator"]["sweeps"] = 1
        mutations.append(wrong_sweeps)
        wrong_owner = decision_marker()
        wrong_owner["arms"][3]["candidate_field"]["root_id"] = "source_c5_heavy"
        mutations.append(wrong_owner)
        wrong_source = decision_marker()
        wrong_source["arms"][1]["candidate_field"]["relative_path"] = "different.npz"
        mutations.append(wrong_source)
        label_access = decision_marker()
        label_access["labels_loaded_to_device"] = True
        mutations.append(label_access)
        for mutation in mutations:
            with self.subTest(mutation=mutation), self.assertRaises(RuntimeError):
                validate_decision_case_marker(mutation, contract, SHA_B, verify_heavy_bytes=False)

    def test_corrupt_metric_envelopes_and_fold_witness_fail_closed(self) -> None:
        contract = decision_contract()
        wrong_id = decision_marker()
        wrong_id["arms"][0]["geometry"][DETJ_DIAGNOSTICS]["metric_id"] = "WRONG"
        nonfinite = decision_marker()
        metric_id = next(metric_id for metric_id in METRIC_SPECS if metric_id != DETJ_DIAGNOSTICS)
        nonfinite["arms"][0]["geometry"][metric_id]["value"] = float("nan")
        negative_count = decision_marker()
        negative_count["arms"][0]["geometry"][DETJ_DIAGNOSTICS]["components"]["invalid_count"] = -1.0
        wrong_witness = decision_marker()
        wrong_witness["arms"][0]["observed_fold_count"] = 1
        for mutation in (wrong_id, nonfinite, negative_count, wrong_witness):
            with self.subTest(mutation=mutation), self.assertRaises(RuntimeError):
                validate_decision_case_marker(mutation, contract, SHA_B, verify_heavy_bytes=False)

    def test_field_record_recomputes_the_array_hash(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "field.npz"
            np.savez_compressed(path, flow=np.zeros((1, 3, 5, 5, 5), dtype=np.float32))
            record = {
                "root_id": "root",
                "relative_path": "field.npz",
                "npz_sha256": sha256_file(path),
                "array_sha256": "0" * 64,
            }
            with self.assertRaisesRegex(RuntimeError, "array changed"):
                verify_field_record(record, {"root": str(root)}, "fixture")


class EvaluationMarkerContractTest(unittest.TestCase):
    def payloads(self) -> tuple[dict[str, object], ...]:
        contract = decision_contract()
        frozen = {
            "baseline_dice": 0.75,
            "baseline_per_label": [{"label": label, "dice": 0.75} for label in EVALUATION_LABEL_IDS],
            "anchors": {
                arm_id: {
                    "candidate_dice": 0.76,
                    "per_label": [{"label": label, "candidate_dice": 0.76} for label in EVALUATION_LABEL_IDS],
                }
                for arm_id in ANCHOR_ARM_IDS
            },
        }
        evaluation = {
            "schema": EVALUATION_CONTRACT_SCHEMA,
            "frozen_evaluation": {"subject_1": frozen},
        }
        rows = []
        for spec in ARM_SPECS:
            candidate = 0.76 if spec.arm_id in ANCHOR_ARM_IDS else 0.761
            rows.append(
                {
                    "arm_id": spec.arm_id,
                    "arm_index": spec.arm_index,
                    "baseline_dice": 0.75,
                    "candidate_dice": candidate,
                    "dice_delta": candidate - 0.75,
                    "source_evaluation_parity_verified": True if spec.arm_id in ANCHOR_ARM_IDS else None,
                    "per_label": [
                        {
                            "label": label,
                            "baseline_dice": 0.75,
                            "candidate_dice": candidate,
                            "dice_delta": candidate - 0.75,
                        }
                        for label in EVALUATION_LABEL_IDS
                    ],
                }
            )
        payload = {
            "schema": EVALUATION_CASE_SCHEMA,
            "protocol_id": PROTOCOL_ID,
            "status": "COMPLETE",
            "strict": True,
            "case_id": "subject_1",
            "decision_contract_sha256": SHA_B,
            "decision_barrier_sha256": SHA_A,
            "evaluation_contract_sha256": "c" * 64,
            "decision_case_sha256": "d" * 64,
            "labels_loaded_after_barrier": True,
            "test_split_accessed": False,
            "labels": list(EVALUATION_LABEL_IDS),
            "arms": rows,
            "execution": {
                "phase": "evaluation",
                "attempt_id": "A_TEST",
                "shard_index": 0,
                "physical_gpu": "2",
                "labels_loaded_to_device": True,
                "deterministic": True,
            },
        }
        barrier = {"decision_case_sha256": {"subject_1": "d" * 64}}
        return payload, contract, barrier, evaluation

    def test_evaluation_arithmetic_and_source_parity_are_checked(self) -> None:
        payload, contract, barrier, evaluation = self.payloads()
        validate_evaluation_case_marker(payload, contract, SHA_B, barrier, SHA_A, evaluation, "c" * 64)
        payload["arms"][3]["dice_delta"] += 0.001
        with self.assertRaisesRegex(RuntimeError, "arithmetic"):
            validate_evaluation_case_marker(payload, contract, SHA_B, barrier, SHA_A, evaluation, "c" * 64)

    def test_evaluation_dice_range_and_anchor_parity_fail_closed(self) -> None:
        payload, contract, barrier, evaluation = self.payloads()
        row = payload["arms"][3]
        row["candidate_dice"] = 2.0
        row["dice_delta"] = 1.25
        for item in row["per_label"]:
            item["candidate_dice"] = 2.0
            item["dice_delta"] = 1.25
        with self.assertRaisesRegex(RuntimeError, "outside"):
            validate_evaluation_case_marker(payload, contract, SHA_B, barrier, SHA_A, evaluation, "c" * 64)

        payload, contract, barrier, evaluation = self.payloads()
        payload["arms"][0]["source_evaluation_parity_verified"] = False
        with self.assertRaisesRegex(RuntimeError, "parity"):
            validate_evaluation_case_marker(payload, contract, SHA_B, barrier, SHA_A, evaluation, "c" * 64)


class WorkerAndBarrierContractTest(unittest.TestCase):
    def test_decision_worker_nested_label_access_fails_closed(self) -> None:
        contract = decision_contract()
        marker = {
            "schema": WORKER_SCHEMA,
            "protocol_id": PROTOCOL_ID,
            "status": "COMPLETE",
            "strict": True,
            "phase": "decision",
            "attempt_id": "A_TEST",
            "shard_index": 0,
            "physical_gpu": "2",
            "case_ids": ["subject_1"],
            "case_sha256": {"subject_1": SHA_A},
            "decision_contract_sha256": SHA_B,
            "labels_loaded": False,
            "test_split_accessed": False,
            "execution": {
                "phase": "decision",
                "attempt_id": "A_TEST",
                "shard_index": 0,
                "physical_gpu": "2",
                "labels_loaded_to_device": False,
                "deterministic": True,
            },
        }
        validate_worker_marker(
            marker,
            contract,
            SHA_B,
            phase="decision",
            shard_index=0,
            attempt_id="A_TEST",
        )
        marker["execution"]["labels_loaded_to_device"] = True
        with self.assertRaises(RuntimeError):
            validate_worker_marker(
                marker,
                contract,
                SHA_B,
                phase="decision",
                shard_index=0,
                attempt_id="A_TEST",
            )

    def test_evaluation_barrier_preserves_case_order_outside_hash_mappings(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            case_ids = ["subject_2", "subject_1", "subject_4", "subject_3"]
            decision = {
                "case_ids": case_ids,
                "num_shards": 2,
                "shards": {"0": ["subject_2", "subject_4"], "1": ["subject_1", "subject_3"]},
            }
            attempt_id = "A_TEST"
            for case_id in case_ids:
                atomic_write_json(root / "cases" / case_id / "evaluation_complete.json", {"case_id": case_id})
            for shard_index in range(2):
                assigned = decision["shards"][str(shard_index)]
                atomic_write_json(
                    root / "workers" / "evaluation" / "attempts" / attempt_id / f"worker_{shard_index:02d}.json",
                    {
                        "case_sha256": {
                            case_id: sha256_file(root / "cases" / case_id / "evaluation_complete.json")
                            for case_id in assigned
                        }
                    },
                )
            with (
                patch("tools.analysis.search_gate_c5b_contracts.validate_worker_marker"),
                patch("tools.analysis.search_gate_c5b_contracts.validate_evaluation_case_marker"),
            ):
                barrier, digest = build_evaluation_barrier(
                    run_root=root,
                    decision=decision,
                    decision_sha256=SHA_A,
                    decision_barrier={},
                    decision_barrier_sha256=SHA_B,
                    evaluation={},
                    evaluation_sha256="c" * 64,
                    attempt_id=attempt_id,
                    completed_at_utc="2026-08-26T00:00:00Z",
                )
                loaded = load_evaluation_barrier(
                    root,
                    digest,
                    decision,
                    SHA_A,
                    {},
                    SHA_B,
                    {},
                    "c" * 64,
                )
            self.assertEqual(barrier["case_ids"], case_ids)
            self.assertEqual(loaded["case_ids"], case_ids)


class FinalizerWiringTest(unittest.TestCase):
    def test_full_58_case_wiring_selects_only_a_preregistered_arm(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            case_ids = [f"subject_{index:03d}" for index in range(EXPECTED_CASE_COUNT)]
            decision = {"case_ids": case_ids}
            barrier = {"decision_case_sha256": {}}
            arm_dice = {
                REFERENCE_ARM_ID: 0.7600,
                ANCHOR_ARM_IDS[1]: 0.7590,
                ANCHOR_ARM_IDS[2]: 0.7630,
                SELECTABLE_ARM_IDS[0]: 0.7610,
                SELECTABLE_ARM_IDS[1]: 0.7603,
                SELECTABLE_ARM_IDS[2]: 0.7595,
                DIAGNOSTIC_ARM_ID: 0.7700,
            }
            for case_id in case_ids:
                decision_rows = []
                evaluation_rows = []
                for spec in ARM_SPECS:
                    proposal = {"clip_rms_retention": 0.97} if spec.arm_id != REFERENCE_ARM_ID else {}
                    decision_rows.append(
                        {
                            "arm_id": spec.arm_id,
                            "role": spec.role,
                            "selectable": spec.selectable,
                            "proposal": proposal,
                            "exact": {"certified": True},
                            "observed_fold_count": 0,
                            "geometry": geometry(0.300 if spec.arm_id == REFERENCE_ARM_ID else 0.301),
                            "candidate_field": {"root_id": "synthetic", "relative_path": spec.arm_id},
                        }
                    )
                    candidate = arm_dice[spec.arm_id]
                    evaluation_rows.append(
                        {
                            "arm_id": spec.arm_id,
                            "baseline_dice": 0.7500,
                            "candidate_dice": candidate,
                            "dice_delta": candidate - 0.7500,
                            "per_label": [
                                {
                                    "label": label,
                                    "baseline_dice": 0.7500,
                                    "candidate_dice": candidate,
                                    "dice_delta": candidate - 0.7500,
                                }
                                for label in EVALUATION_LABEL_IDS
                            ],
                        }
                    )
                dpath = root / "cases" / case_id / "decision_complete.json"
                epath = root / "cases" / case_id / "evaluation_complete.json"
                atomic_write_json(dpath, {"arms": decision_rows, "resource": {"wall_sec": 1.0}})
                atomic_write_json(epath, {"arms": evaluation_rows})
                barrier["decision_case_sha256"][case_id] = sha256_file(dpath)
            for name in ("source_contract.json", "decision_contract.json"):
                atomic_write_json(root / name, {"synthetic": True})
            with (
                patch("tools.analysis.search_gate_c5b_workers.validate_decision_case_marker"),
                patch("tools.analysis.search_gate_c5b_workers.validate_evaluation_case_marker"),
            ):
                finalize_c5b(
                    run_root=root,
                    decision=decision,
                    decision_sha256=SHA_B,
                    barrier=barrier,
                    barrier_sha256=SHA_A,
                    evaluation={},
                    evaluation_sha256="c" * 64,
                    evaluation_barrier={
                        "evaluation_case_sha256": {
                            case_id: sha256_file(root / "cases" / case_id / "evaluation_complete.json")
                            for case_id in case_ids
                        }
                    },
                    evaluation_barrier_sha256="e" * 64,
                )
            branch = json.loads((root / "next_branch.json").read_text(encoding="utf-8"))
            self.assertEqual(branch["branch_id"], BRANCH_FREEZE_SUPERIOR)
            self.assertEqual(branch["selected_arm_id"], SELECTABLE_ARM_IDS[0])
            self.assertNotEqual(branch["selected_arm_id"], DIAGNOSTIC_ARM_ID)
            summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
            self.assertAlmostEqual(summary["reference_c4"]["dice_mean"], 0.76)
            self.assertEqual(len(summary["arms"]), 7)


class RunnerSourceTest(unittest.TestCase):
    def test_shell_has_pilot_tmp_hint_and_unique_failed_package(self) -> None:
        text = Path("tools/runners/eval/search_gate_c5b.sh").read_text(encoding="utf-8")
        self.assertIn("/tmp/search_gate_c5b.log", text)
        self.assertLess(text.index("decision-pilot"), text.index("decision-worker"))
        self.assertIn("${RUN_ID}__${ATTEMPT_ID}__FAILED", text)
        self.assertIn("Test-115 was not accessed", text)


if __name__ == "__main__":
    unittest.main()
