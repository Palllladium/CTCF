from __future__ import annotations

import copy
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path

from tools.analysis.search_gate_c5 import (
    ARM_SPECS,
    C5_DECISION_POLICY_SHA256,
    C5_POLICY,
    C5_POLICY_SHA256,
    SELECTOR_SPECS,
    CandidateSignals,
    choose_global_candidate,
    decision_policy_contract,
)
from tools.analysis.search_gate_c5_contracts import (
    BARRIER_SCHEMA,
    DECISION_CASE_SCHEMA,
    EVALUATION_CASE_SCHEMA,
    EVALUATION_LABEL_IDS,
    EXPECTED_SUPPORT_CONTRACT,
    PROTOCOL_ID,
    SOURCE_C4_ANCHOR_IDS,
    SOURCE_C4_GIT_HEAD,
    SOURCE_C4_MANIFEST_SHA256,
    SOURCE_C4_RUN_ID,
    SOURCE_C4_RUN_MANIFEST_SHA256,
    build_decision_contract,
    build_evaluation_contract,
    build_source_contract,
    payload_sha256,
    validate_decision_case_marker,
    validate_evaluation_case_marker,
    validate_worker_marker,
    verify_rooted_record,
)
from tools.analysis.search_gate_metrics import MATHEMATICAL_SDLOGJ_CROP2

SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64
GIT_SHA = "d" * 40


def _field(root_id: str, relative_path: str, sha: str = SHA_A) -> dict[str, object]:
    return {
        "root_id": root_id,
        "relative_path": relative_path,
        "npz_sha256": sha,
        "array_sha256": sha,
    }


def _geometry(value: float) -> dict[str, object]:
    return {
        MATHEMATICAL_SDLOGJ_CROP2: {
            "metric_id": MATHEMATICAL_SDLOGJ_CROP2,
            "status": "OK",
            "value": value,
        }
    }


class C5ContractFixture(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        base = Path(self.temporary.name)
        self.c3 = base / "c3"
        self.c4 = base / "c4"
        self.target = base / "c5"
        for root in (self.c3, self.c4, self.target):
            root.mkdir()
        self.case_ids = [f"subject_{index:03d}" for index in range(58)]
        image_inputs = {"atlas": _field("source_c3_heavy", "images/atlas.npy")}
        image_inputs.update({case_id: _field("source_c3_heavy", f"images/{case_id}.npy") for case_id in self.case_ids})
        source_initial = {
            case_id: {
                "field": _field("source_c3_heavy", f"initial/{case_id}.npz"),
                "exact": {"status": "CERTIFIED", "certified": True, "sha256": SHA_A},
            }
            for case_id in self.case_ids
        }
        source_historical = {
            case_id: {
                "raw_conf_requested_field": _field("source_c3_heavy", f"requested/{case_id}.npz"),
                "source_decision_case_sha256": SHA_B,
            }
            for case_id in self.case_ids
        }
        anchors = {
            case_id: {
                anchor_id: {
                    "field": _field("source_c4_heavy", f"cases/{case_id}/{anchor_id}.npz", SHA_B),
                    "source_decision_case_sha256": SHA_B,
                    "exact_array_sha256": SHA_B,
                }
                for anchor_id in SOURCE_C4_ANCHOR_IDS
            }
            for case_id in self.case_ids
        }
        per_label = [{"label": label, "dice": 0.5} for label in EVALUATION_LABEL_IDS]
        snapshot = {
            "source_c4": {
                "compact_directory": str(base / "compact"),
                "heavy_root": str(self.c4),
                "run_id": SOURCE_C4_RUN_ID,
                "git_head": SOURCE_C4_GIT_HEAD,
                "manifest_sha256": SOURCE_C4_MANIFEST_SHA256,
                "run_manifest_sha256": SOURCE_C4_RUN_MANIFEST_SHA256,
                "source_contract_sha256": SHA_A,
                "decision_contract_sha256": SHA_B,
                "decision_barrier_sha256": SHA_C,
            },
            "source_c3_heavy_root": str(self.c3),
            "raw_inputs": {
                "atlas": {"case_id": "atlas", "split": "atlas", "sha256": SHA_A},
                **{case_id: {"case_id": case_id, "split": "val", "sha256": SHA_A} for case_id in self.case_ids},
            },
            "image_inputs": image_inputs,
            "source_initial": source_initial,
            "source_historical": source_historical,
            "source_c4_anchors": anchors,
            "baseline_geometry": {case_id: _geometry(0.3) for case_id in self.case_ids},
            "evaluation_baseline_dice": {case_id: 0.75 for case_id in self.case_ids},
            "evaluation_baseline_per_label": {case_id: per_label for case_id in self.case_ids},
            "evaluation_c4_anchor_dice": {
                case_id: {
                    anchor_id: {
                        "aggregate_dice": 0.76 if anchor_id == "intensity_s1" else 0.77,
                        "per_label": per_label,
                        "source_evaluation_case_sha256": SHA_C,
                    }
                    for anchor_id in SOURCE_C4_ANCHOR_IDS
                }
                for case_id in self.case_ids
            },
            "evaluation_label_ids": list(EVALUATION_LABEL_IDS),
            "case_ids": self.case_ids,
            "seed": 0,
            "runtime_signature": {"torch": "synthetic"},
        }
        self.policy = C5_POLICY.to_dict()
        self.contrasts = {"families": ["synthetic"]}
        self.source = build_source_contract(
            snapshot,
            git_head=GIT_SHA,
            runtime_signature=snapshot["runtime_signature"],
            target_heavy_root=self.target,
            physical_gpus=("2", "3", "4", "5", "6"),
            full_policy=self.policy,
            expected_full_policy_sha256=C5_POLICY_SHA256,
            contrast_contract=self.contrasts,
            expected_contrast_contract_sha256=payload_sha256(self.contrasts),
        )
        self.source_sha = payload_sha256(self.source)
        self.arms = [asdict(spec) for spec in ARM_SPECS]
        self.selectors = [asdict(spec) for spec in SELECTOR_SPECS]
        self.offsets = [{"reach_id": f"S{value}", "offsets": []} for value in range(1, 5)]
        self.support = copy.deepcopy(EXPECTED_SUPPORT_CONTRACT)
        self.decision = build_decision_contract(
            self.source,
            self.source_sha,
            decision_policy=decision_policy_contract(),
            expected_decision_policy_sha256=C5_DECISION_POLICY_SHA256,
            arm_specs=self.arms,
            expected_arm_specs_sha256=payload_sha256(self.arms),
            selector_specs=self.selectors,
            expected_selector_specs_sha256=payload_sha256(self.selectors),
            offset_table=self.offsets,
            expected_offset_table_sha256=payload_sha256(self.offsets),
            support_contract=self.support,
            expected_support_contract_sha256=payload_sha256(self.support),
        )
        self.decision_sha = payload_sha256(self.decision)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def decision_case(self, case_id: str | None = None) -> dict[str, object]:
        case_id = case_id or self.case_ids[0]
        baseline = self.decision["baseline_geometry"][case_id]
        arms: list[dict[str, object]] = []
        signals = []
        source_map = {"int_s1_a10_b0": "intensity_s1", "int_s2_a10_b0": "intensity_s2"}
        for spec in self.arms:
            arm_id = spec["arm_id"]
            source_id = source_map.get(arm_id)
            field = (
                self.decision["source_c4_anchors"][case_id][source_id]["field"]
                if source_id is not None
                else _field("target_c5_heavy", f"cases/{case_id}/{arm_id}.npz")
            )
            row = {
                "arm_index": spec["arm_index"],
                "arm_id": arm_id,
                "descriptor_id": spec["descriptor_id"],
                "reach_id": spec["reach_id"],
                "stride_voxels": spec["stride_voxels"],
                "post_rms_amplitude": spec["post_rms_amplitude"],
                "centre_beta": spec["centre_beta"],
                "historical_anchor": spec["historical_anchor"],
                "selectable": spec["selectable"],
                "candidate_field": field,
                "persistence": {
                    "owner": field["root_id"],
                    "saved_npz_sha256": field["npz_sha256"],
                    "reloaded_array_sha256": field["array_sha256"],
                    "source_anchor_reused": spec["historical_anchor"],
                },
                "exact": {"status": "CERTIFIED", "certified": True, "sha256": field["array_sha256"]},
                "geometry": _geometry(0.305),
                "mathematical_sdlogj_delta": 0.005,
                "support": {
                    "retention": 1.0,
                    "ncc7": {"baseline_count": 100, "pair_count": 100, "retention": 1.0},
                    "mind_d2": {"baseline_count": 90, "pair_count": 90, "retention": 1.0},
                },
                "utilities": {
                    "ncc7": {
                        "utility_id": "COMMON_NCC7",
                        "baseline_count": 100,
                        "pair_count": 100,
                        "retention": 1.0,
                        "baseline_loss": 1.0,
                        "candidate_loss": 0.99,
                        "improvement": 0.01,
                    },
                    "mind_d2": {
                        "utility_id": "COMMON_MIND_D2",
                        "baseline_count": 90,
                        "pair_count": 90,
                        "retention": 1.0,
                        "baseline_loss": 1.0,
                        "candidate_loss": 0.98,
                        "improvement": 0.02,
                    },
                },
                "proposal": {
                    "reach_id": spec["reach_id"],
                    "stride_voxels": spec["stride_voxels"],
                    "centre_beta": spec["centre_beta"],
                    "amplitude_stage": "after_rms_match_before_local_clip",
                    "pre_rms_multiplier": 2.0 if spec["reach_id"] == "S1" else 1.0,
                    "post_rms_amplitude": spec["post_rms_amplitude"],
                    "smoothing_passes": 1,
                    "collar_width": 7,
                    "rms_target_source_id": "source_c3_raw_conf_post1_requested",
                    "rms_target": 0.1,
                    "rms_requested": 0.1 * spec["post_rms_amplitude"],
                    "rms_realized": 0.09 * spec["post_rms_amplitude"],
                    "clip_rms_retention_raw": 0.9,
                    "clip_rms_retention": 0.9,
                    "clip_cosine": 1.0,
                },
            }
            arms.append(row)
            signals.append(CandidateSignals(arm_id, True, 1.0, 0.9, 0.01, 0.02, 0.005))
        selectors = []
        by_arm = {row["arm_id"]: row for row in arms}
        for spec in self.selectors:
            choice = choose_global_candidate(signals, spec["selector_id"])
            selectors.append(
                {
                    "selector_index": spec["selector_index"],
                    "selector_id": spec["selector_id"],
                    "action": choice.action,
                    "selected_arm_id": choice.selected_arm_id,
                    "eligible_arm_ids": list(choice.eligible_arm_ids),
                    "returned_field": by_arm[choice.selected_arm_id]["candidate_field"]
                    if choice.selected_arm_id
                    else None,
                    "rollback_to_source_initial": choice.selected_arm_id is None,
                }
            )
        return {
            "schema": DECISION_CASE_SCHEMA,
            "protocol_id": PROTOCOL_ID,
            "status": "COMPLETE",
            "case_id": case_id,
            "decision_contract_sha256": self.decision_sha,
            "shard_index": 0,
            "physical_gpu": "2",
            "arm_specs_sha256": self.decision["arm_specs_sha256"],
            "selector_specs_sha256": self.decision["selector_specs_sha256"],
            "labels_loaded_to_device": False,
            "test_split_accessed": False,
            "source_image_array_sha256": SHA_A,
            "source_initial_array_sha256": SHA_A,
            "source_rms_reference": {
                "field": self.decision["source_historical"][case_id]["raw_conf_requested_field"],
                "source_decision_case_sha256": SHA_B,
                "residual_rms": 0.1,
            },
            "generation_support": {
                "support_id": "C4_COMMON_DESCRIPTOR_SUPPORT_RETAINED_V1",
                "geometry_count": 100,
                "common_count": 90,
                "retention": 0.9,
                "reach": [
                    {
                        "reach_id": f"S{stride}",
                        "stride_voxels": stride,
                        "generation_count": 90,
                        "all_candidates_valid_count": 90,
                        "all_candidates_valid": True,
                    }
                    for stride in range(1, 5)
                ],
            },
            "historical_anchor_parity": [
                {
                    "arm_id": c5_id,
                    "source_anchor_id": c4_id,
                    "source_array_sha256": SHA_B,
                    "candidate_array_sha256": SHA_B,
                    "array_byte_identical": True,
                }
                for c5_id, c4_id in source_map.items()
            ],
            "baseline_geometry": baseline,
            "arms": arms,
            "selectors": selectors,
        }


class LabelIsolationAndRootTest(C5ContractFixture):
    def test_decision_projection_excludes_all_evaluation_values(self) -> None:
        serialized = str(self.decision).lower()
        self.assertNotIn("evaluation_baseline_dice", self.decision)
        self.assertNotIn("evaluation_c4_anchor_dice", self.decision)
        self.assertNotIn("raw_inputs", self.decision)
        self.assertNotIn("segmentation", serialized)

    def test_root_escape_and_wrong_owner_are_rejected(self) -> None:
        roots = {key: Path(value) for key, value in self.source["roots"].items()}
        record = _field("target_c5_heavy", "../escape.npz")
        with self.assertRaises(RuntimeError):
            verify_rooted_record(record, roots, verify_bytes=False)
        marker = self.decision_case()
        marker["arms"][0]["candidate_field"]["root_id"] = "source_c4_heavy"
        with self.assertRaises(RuntimeError):
            validate_decision_case_marker(marker, self.decision, self.decision_sha, verify_heavy_bytes=False)


class DecisionTamperTest(C5ContractFixture):
    def test_valid_marker_and_source_owned_historical_anchors(self) -> None:
        marker = self.decision_case()
        validate_decision_case_marker(marker, self.decision, self.decision_sha, verify_heavy_bytes=False)
        by_id = {row["arm_id"]: row for row in marker["arms"]}
        self.assertEqual(by_id["int_s1_a10_b0"]["candidate_field"]["root_id"], "source_c4_heavy")
        self.assertEqual(by_id["int_s2_a10_b0"]["candidate_field"]["root_id"], "source_c4_heavy")

    def test_geometry_amplitude_and_selector_tampering_are_rejected(self) -> None:
        for mutate in (
            lambda marker: marker["arms"][0].__setitem__("mathematical_sdlogj_delta", 0.004),
            lambda marker: marker["arms"][0]["proposal"].__setitem__("rms_requested", 0.2),
            lambda marker: marker["arms"][0]["support"].__setitem__("retention", 0.999),
            lambda marker: marker["arms"][0]["utilities"]["ncc7"].__setitem__("improvement", 0.02),
            lambda marker: marker["selectors"][0].__setitem__("selected_arm_id", marker["arms"][-1]["arm_id"]),
        ):
            marker = self.decision_case()
            mutate(marker)
            with self.assertRaises(RuntimeError):
                validate_decision_case_marker(marker, self.decision, self.decision_sha, verify_heavy_bytes=False)

    def test_factor_and_case_provenance_tampering_are_rejected(self) -> None:
        for mutate in (
            lambda marker: marker["arms"][0].__setitem__("centre_beta", 123.0),
            lambda marker: marker["arms"][0]["proposal"].__setitem__("stride_voxels", 4),
            lambda marker: marker["arms"][0]["proposal"].__setitem__("pre_rms_multiplier", 1.0),
            lambda marker: marker["source_rms_reference"].__setitem__("source_decision_case_sha256", SHA_A),
            lambda marker: marker["generation_support"]["reach"][2].__setitem__("all_candidates_valid_count", 89),
            lambda marker: marker["baseline_geometry"][MATHEMATICAL_SDLOGJ_CROP2].__setitem__("value", 0.31),
            lambda marker: marker.__setitem__("execution", {"opaque": "hidden_segmentation.pkl"}),
        ):
            marker = self.decision_case()
            mutate(marker)
            with self.assertRaises(RuntimeError):
                validate_decision_case_marker(marker, self.decision, self.decision_sha, verify_heavy_bytes=False)


class EvaluationTamperTest(C5ContractFixture):
    def test_evaluation_worker_is_bound_to_both_barrier_and_evaluation_contract(self) -> None:
        marker = {
            "schema": "ctcf-search-c5-worker-v1",
            "protocol_id": PROTOCOL_ID,
            "status": "COMPLETE",
            "phase": "evaluation",
            "attempt_id": "attempt",
            "shard_index": 0,
            "physical_gpu": "2",
            "case_ids": self.decision["shards"]["0"],
            "decision_contract_sha256": self.decision_sha,
            "decision_barrier_sha256": SHA_B,
            "evaluation_contract_sha256": SHA_C,
            "labels_loaded": True,
            "test_split_accessed": False,
        }
        validate_worker_marker(
            marker,
            self.decision,
            self.decision_sha,
            phase="evaluation",
            shard_index=0,
            attempt_id="attempt",
            barrier_sha256=SHA_B,
            evaluation_contract_sha256=SHA_C,
        )
        marker["evaluation_contract_sha256"] = SHA_A
        with self.assertRaises(RuntimeError):
            validate_worker_marker(
                marker,
                self.decision,
                self.decision_sha,
                phase="evaluation",
                shard_index=0,
                attempt_id="attempt",
                barrier_sha256=SHA_B,
                evaluation_contract_sha256=SHA_C,
            )

    def _evaluation_fixture(self) -> tuple[dict[str, object], dict[str, object], dict[str, object], str, str]:
        case_id = self.case_ids[0]
        decision_case = self.decision_case(case_id)
        decision_case_sha = SHA_C
        barrier = {
            "schema": BARRIER_SCHEMA,
            "protocol_id": PROTOCOL_ID,
            "status": "COMPLETE",
            "decision_contract_sha256": self.decision_sha,
            "decision_workers_received_label_inputs": False,
            "test_split_accessed": False,
            "decision_case_sha256": {value: (SHA_C if value == case_id else SHA_A) for value in self.case_ids},
        }
        barrier_sha = SHA_B
        evaluation_contract = build_evaluation_contract(
            self.source,
            self.source_sha,
            self.decision_sha,
            barrier,
            barrier_sha,
        )
        evaluation_sha = payload_sha256(evaluation_contract)
        baseline_labels = self.source["evaluation_baseline_per_label"][case_id]
        baseline_rows = [
            {"label": row["label"], "baseline_dice": row["dice"], "candidate_dice": row["dice"], "dice_delta": 0.0}
            for row in baseline_labels
        ]
        source_map = {"int_s1_a10_b0": "intensity_s1", "int_s2_a10_b0": "intensity_s2"}
        arms = []
        arm_values: dict[str, tuple[float, list[dict[str, object]]]] = {}
        for spec in self.arms:
            source_id = source_map.get(spec["arm_id"])
            if source_id is None:
                candidate = 0.75
                per_label = copy.deepcopy(baseline_rows)
                parity = False
            else:
                frozen = self.source["evaluation_c4_anchor_dice"][case_id][source_id]
                candidate = frozen["aggregate_dice"]
                per_label = [
                    {
                        "label": row["label"],
                        "baseline_dice": baseline_labels[index]["dice"],
                        "candidate_dice": row["dice"],
                        "dice_delta": row["dice"] - baseline_labels[index]["dice"],
                    }
                    for index, row in enumerate(frozen["per_label"])
                ]
                parity = True
            arm_values[spec["arm_id"]] = (candidate, per_label)
            row = {
                "arm_index": spec["arm_index"],
                "arm_id": spec["arm_id"],
                "baseline_dice": 0.75,
                "candidate_dice": candidate,
                "capacity_dice_delta": candidate - 0.75,
                "per_label": per_label,
            }
            if parity:
                row["historical_c4_dice_parity_verified"] = True
            arms.append(row)
        selectors = []
        decision_selectors = {row["selector_id"]: row for row in decision_case["selectors"]}
        for spec in self.selectors:
            selected = decision_selectors[spec["selector_id"]]["selected_arm_id"]
            candidate, per_label = (0.75, baseline_rows) if selected is None else arm_values[selected]
            selectors.append(
                {
                    "selector_index": spec["selector_index"],
                    "selector_id": spec["selector_id"],
                    "action": decision_selectors[spec["selector_id"]]["action"],
                    "selected_arm_id": selected,
                    "returned_dice": candidate,
                    "dice_delta": candidate - 0.75,
                    "per_label": [
                        {
                            "label": row["label"],
                            "baseline_dice": row["baseline_dice"],
                            "returned_dice": row["candidate_dice"],
                            "dice_delta": row["dice_delta"],
                        }
                        for row in per_label
                    ],
                }
            )
        evaluation = {
            "schema": EVALUATION_CASE_SCHEMA,
            "protocol_id": PROTOCOL_ID,
            "status": "COMPLETE",
            "case_id": case_id,
            "decision_contract_sha256": self.decision_sha,
            "decision_barrier_sha256": barrier_sha,
            "evaluation_contract_sha256": evaluation_sha,
            "decision_case_sha256": decision_case_sha,
            "labels_loaded_after_barrier": True,
            "test_split_accessed": False,
            "labels": list(EVALUATION_LABEL_IDS),
            "arms": arms,
            "selectors": selectors,
        }
        return evaluation, decision_case, barrier, barrier_sha, evaluation_sha

    def test_valid_evaluation_and_anchor_dice_tamper(self) -> None:
        evaluation, decision_case, barrier, barrier_sha, evaluation_sha = self._evaluation_fixture()
        evaluation_contract = build_evaluation_contract(
            self.source, self.source_sha, self.decision_sha, barrier, barrier_sha
        )
        validate_evaluation_case_marker(
            evaluation,
            self.decision,
            self.decision_sha,
            barrier,
            barrier_sha,
            evaluation_contract,
            evaluation_sha,
            decision_case,
            SHA_C,
        )
        anchor = next(row for row in evaluation["arms"] if row["arm_id"] == "int_s2_a10_b0")
        anchor["candidate_dice"] += 1e-4
        anchor["capacity_dice_delta"] += 1e-4
        with self.assertRaises(RuntimeError):
            validate_evaluation_case_marker(
                evaluation,
                self.decision,
                self.decision_sha,
                barrier,
                barrier_sha,
                evaluation_contract,
                evaluation_sha,
                decision_case,
                SHA_C,
            )

    def test_evaluation_barrier_and_label_order_tampering_are_rejected(self) -> None:
        for mutate in (
            lambda value: value.__setitem__("decision_barrier_sha256", SHA_A),
            lambda value: value["labels"].reverse(),
            lambda value: value["selectors"][0].__setitem__("selected_arm_id", None),
        ):
            evaluation, decision_case, barrier, barrier_sha, evaluation_sha = self._evaluation_fixture()
            evaluation_contract = build_evaluation_contract(
                self.source, self.source_sha, self.decision_sha, barrier, barrier_sha
            )
            mutate(evaluation)
            with self.assertRaises(RuntimeError):
                validate_evaluation_case_marker(
                    evaluation,
                    self.decision,
                    self.decision_sha,
                    barrier,
                    barrier_sha,
                    evaluation_contract,
                    evaluation_sha,
                    decision_case,
                    SHA_C,
                )


if __name__ == "__main__":
    unittest.main()
