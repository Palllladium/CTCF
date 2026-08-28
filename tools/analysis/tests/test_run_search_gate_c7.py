from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch

import tools.analysis.search_gate_c7_source as c7_source
from tools.analysis.run_artifacts import sha256_file
from tools.analysis.run_search_gate_c7 import (
    _centered_for_posterior,
    _configure_strict_fp32_backend,
    _decision_policy,
    _intensity_raw_on_support,
    _load_barrier,
    _padding_support_mask,
    _validate_decision_case_marker,
    _validate_evaluation_case_marker,
    build_decision_barrier,
    build_parser,
)
from tools.analysis.search_gate_c3 import primary_ncc_decision
from tools.analysis.search_gate_c7 import (
    ARM_SPECS,
    C7_POLICY_SHA256,
    DESCRIPTOR_CHECKPOINT_SHA256,
    MATCHED_CONTROL_ARM_ID,
    PROTOCOL_ID,
    REFERENCE_ARM_ID,
    SOURCE_CONTEXT_ARM_ID,
)
from tools.analysis.search_gate_c7_source import (
    BARRIER_SCHEMA,
    C6_MANIFEST_SHA256,
    DECISION_CASE_SCHEMA,
    EVALUATION_CASE_SCHEMA,
    assert_decision_payload_label_free,
    authenticate_c6_source,
)
from tools.analysis.search_gate_learned import (
    RawCandidateCostVolume,
    build_raw_corrmlp_x1_cost_volume,
    corrmlp_x1_offsets,
    equal_standardized_intensity_hybrid,
)

STRICT_FP32_RUNTIME = {
    "cuda_matmul_allow_tf32": False,
    "cudnn_allow_tf32": False,
    "float32_matmul_precision": "highest",
}


class DecisionIsolationTest(unittest.TestCase):
    def test_decision_policy_is_label_free(self) -> None:
        policy = _decision_policy()
        self.assertNotIn("thresholds", policy)
        self.assertNotIn("statistics", policy)
        assert_decision_payload_label_free(policy)

    def test_evaluation_key_is_rejected(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "evaluation data"):
            assert_decision_payload_label_free({"label_ids": [1, 2]})
        with self.assertRaisesRegex(RuntimeError, "evaluation data"):
            assert_decision_payload_label_free({"evaluation_rows": {}})

    def test_false_protected_flags_are_accepted_but_true_is_rejected(self) -> None:
        assert_decision_payload_label_free(
            {
                "labels_loaded_to_device": False,
                "labels_loaded": False,
                "test_split_accessed": False,
                "test_115_authorized": False,
            }
        )
        with self.assertRaisesRegex(RuntimeError, "protected flag"):
            assert_decision_payload_label_free({"labels_loaded_to_device": True})


class ParserContractTest(unittest.TestCase):
    def test_all_staged_actions_are_exposed(self) -> None:
        parser = build_parser()
        action = next(item for item in parser._actions if item.dest == "action")
        self.assertEqual(
            set(action.choices),
            {
                "selfcheck",
                "prepare",
                "decision-pilot",
                "decision-worker",
                "decision-barrier",
                "freeze-evaluation",
                "evaluation-worker",
                "finalize",
            },
        )

    def test_policy_uses_unambiguous_margin_name(self) -> None:
        self.assertEqual(MATCHED_CONTROL_ARM_ID, "intensity_margin2_full421_a150")
        self.assertNotIn("rf2", json.dumps(_decision_policy()).lower())
        self.assertEqual(len(C7_POLICY_SHA256), 64)

    def test_backend_contract_disables_tf32(self) -> None:
        prior_matmul = torch.backends.cuda.matmul.allow_tf32
        prior_cudnn = torch.backends.cudnn.allow_tf32
        prior_precision = torch.get_float32_matmul_precision()
        try:
            _configure_strict_fp32_backend()
            self.assertIs(torch.backends.cuda.matmul.allow_tf32, False)
            self.assertIs(torch.backends.cudnn.allow_tf32, False)
            self.assertEqual(torch.get_float32_matmul_precision(), "highest")
        finally:
            torch.backends.cuda.matmul.allow_tf32 = prior_matmul
            torch.backends.cudnn.allow_tf32 = prior_cudnn
            torch.set_float32_matmul_precision(prior_precision)


class C6AuthenticationTest(unittest.TestCase):
    def test_frozen_local_manifest_hash_when_present(self) -> None:
        path = Path("results/search_gate_c6/C6_DEVELOPMENT_20260827T211512Z_c0a59d1c04af/c6_manifest.json")
        if not path.is_file():
            self.skipTest("historical C6 compact product is absent")
        self.assertEqual(sha256_file(path), C6_MANIFEST_SHA256)

    def test_altered_c6_manifest_is_rejected_before_projection(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "c6_manifest.json").write_text("{}\n", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "authenticated C6 artifact"):
                authenticate_c6_source(root, root / "c3", root / "c4", root / "c6")

    def test_pre_barrier_projection_contains_no_evaluation_data(self) -> None:
        root = Path("results/search_gate_c6/C6_DEVELOPMENT_20260827T211512Z_c0a59d1c04af")
        if not root.is_dir():
            self.skipTest("historical C6 compact product is absent")
        decision = json.loads((root / "decision_contract.json").read_text(encoding="utf-8"))
        owner_roots = {key: Path(value) for key, value in decision["roots"].items()}
        with mock.patch.object(c7_source, "_require_file", wraps=c7_source._require_file) as require_file:
            snapshot = authenticate_c6_source(
                root,
                owner_roots["source_c3_heavy"],
                owner_roots["source_c4_heavy"],
                owner_roots["target_c6_heavy"],
                verify_heavy_bytes=False,
            )
        opened_relatives = [str(call.args[1]) for call in require_file.call_args_list]
        self.assertFalse(any("evaluation" in value for value in opened_relatives))
        source = {
            "decision_policy": _decision_policy(),
            "authenticated_c6": snapshot,
            "test_115_authorized": False,
            "test_split_accessed": False,
        }
        assert_decision_payload_label_free(source)

        def keys(value: object) -> set[str]:
            if isinstance(value, dict):
                return {str(key).lower() for key in value} | set().union(*(keys(child) for child in value.values()))
            if isinstance(value, list):
                return set().union(*(keys(child) for child in value)) if value else set()
            return set()

        observed = keys(source)
        self.assertFalse(any("label" in key for key in observed))
        self.assertFalse(any("dice" in key for key in observed))
        self.assertFalse(any("evaluation" in key and key != "evaluation_device" for key in observed))


class DecisionBarrierTest(unittest.TestCase):
    def _fixture(self, root: Path) -> tuple[dict[str, object], str, Path]:
        decision = {
            "case_ids": ["case_a"],
            "image_inputs": {"subject_344": {"shape": [1, 9, 9, 9]}},
        }
        decision_path = root / "decision_contract.json"
        decision_path.write_text(json.dumps(decision), encoding="utf-8")
        decision_sha = sha256_file(decision_path)
        case_path = root / "cases" / "case_a" / "decision_complete.json"
        case_path.parent.mkdir(parents=True)
        execution = {
            "attempt_id": "attempt",
            "shard_index": 0,
            "physical_gpu": "0",
            "host": "host",
            "device": "cuda:0",
            "gpu_name": "gpu",
        }
        case_path.write_text(json.dumps({"status": "COMPLETE", "execution": execution}), encoding="utf-8")
        pilot = {
            "schema": "ctcf-search-c7-descriptor-pilot-v1",
            "protocol_id": PROTOCOL_ID,
            "status": "PASS",
            "strict": True,
            "case_id": "subject_344",
            "decision_contract_sha256": decision_sha,
            "checkpoint_sha256": DESCRIPTOR_CHECKPOINT_SHA256,
            "checkpoint_epoch": 99,
            "state_key_count": 386,
            "feature_shape": [1, 8, 9, 9, 9],
            "feature_dtype": "torch.float32",
            "feature_array_sha256": "f" * 64,
            "feature_nonconstant": True,
            "feature_deterministic": True,
            "feature_requires_grad": False,
            "zero_field_local_cost_parity_max_abs": 0.0,
            "zero_field_local_cost_parity_support_count": 1,
            "zero_field_local_cost_sign_mapping": "negative_runner_cost_equals_positive_upstream_correlation",
            "zero_field_local_cost_offset_order_mapping": "identity_lexicographic_zyx_stride1",
            "sampling_scope": "zero_field_local_cost_only",
            "full_native_decoder_sampling_parity_claimed": False,
            "target_centred_nonconstant_field_semantics": "intentionally_not_native_neighbor_warp_semantics",
            "labels_loaded_to_device": False,
            "test_115_authorized": False,
            "test_split_accessed": False,
        }
        (root / "descriptor_pilot.json").write_text(json.dumps(pilot), encoding="utf-8")
        return decision, decision_sha, case_path

    def test_barrier_uses_canonical_false_flag_and_detects_case_tamper(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            decision, decision_sha, case_path = self._fixture(root)
            with mock.patch(
                "tools.analysis.run_search_gate_c7._validate_decision_case_marker",
                return_value={
                    "status": "COMPLETE",
                    "execution": {
                        "attempt_id": "attempt",
                        "shard_index": 0,
                        "physical_gpu": "0",
                        "host": "host",
                        "device": "cuda:0",
                        "gpu_name": "gpu",
                    },
                },
            ):
                digest = build_decision_barrier(root, decision, decision_sha, "attempt")
            barrier = json.loads((root / "decision_barrier.json").read_text(encoding="utf-8"))
            self.assertEqual(barrier["schema"], BARRIER_SCHEMA)
            self.assertIs(barrier["labels_loaded_to_device"], False)
            self.assertNotIn("labels_loaded_before_barrier", barrier)
            self.assertEqual(barrier["descriptor_pilot_sha256"], sha256_file(root / "descriptor_pilot.json"))
            _load_barrier(root, digest, decision_sha)
            case_path.write_text('{"status":"ALTERED"}\n', encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "snapshot changed"):
                _load_barrier(root, digest, decision_sha)

    def test_barrier_rejects_descriptor_pilot_tamper(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            decision, decision_sha, _ = self._fixture(root)
            with mock.patch(
                "tools.analysis.run_search_gate_c7._validate_decision_case_marker",
                return_value={
                    "execution": {
                        "attempt_id": "attempt",
                        "shard_index": 0,
                        "physical_gpu": "0",
                        "host": "host",
                        "device": "cuda:0",
                        "gpu_name": "gpu",
                    }
                },
            ):
                digest = build_decision_barrier(root, decision, decision_sha, "attempt")
            pilot = json.loads((root / "descriptor_pilot.json").read_text(encoding="utf-8"))
            pilot["full_native_decoder_sampling_parity_claimed"] = True
            (root / "descriptor_pilot.json").write_text(json.dumps(pilot), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "pilot"):
                _load_barrier(root, digest, decision_sha)

    def test_barrier_rejects_old_descriptor_pilot_claim_before_binding(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            decision, decision_sha, _ = self._fixture(root)
            pilot_path = root / "descriptor_pilot.json"
            pilot = json.loads(pilot_path.read_text(encoding="utf-8"))
            pilot["zero_field_local_cost_offset_order_mapping"] = "unverified_native_order"
            pilot_path.write_text(json.dumps(pilot), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "pilot"):
                build_decision_barrier(root, decision, decision_sha, "attempt")


class DecisionMarkerSemanticsTest(unittest.TestCase):
    @staticmethod
    def _fixture(root: Path) -> tuple[Path, dict[str, object], str, dict[str, object]]:
        decision_sha = "d" * 64
        support = {"utility_id": "COMMON_NCC7", "baseline_count": 10, "pair_count": 10, "retention": 1.0}
        utility = {"baseline_loss": -1.0, "candidate_loss": -1.1, "improvement": 0.1}
        inferred = primary_ncc_decision(
            exact_certified=True,
            support_retention=1.0,
            baseline_ncc_loss=-1.0,
            candidate_ncc_loss=-1.1,
        ).to_dict()
        action = "ACCEPT" if inferred["accept"] else "ROLLBACK"
        source_sha = "s" * 64
        source_records: dict[str, dict[str, object]] = {}
        rows = []
        for spec in ARM_SPECS:
            root_id = (
                "source_c4_heavy"
                if spec.arm_id == REFERENCE_ARM_ID
                else "source_c6_heavy"
                if spec.arm_id == SOURCE_CONTEXT_ARM_ID
                else "target_c7_heavy"
            )
            field = {
                "root_id": root_id,
                "relative_path": f"{spec.arm_id}.npz",
                "npz_sha256": "n" * 64,
                "array_sha256": "a" * 64,
            }
            base = {
                "arm_index": spec.arm_index,
                "arm_id": spec.arm_id,
                "role": spec.role,
                "descriptor": spec.descriptor,
                "selectable": spec.selectable,
                "factors": list(spec.factors),
                "amplitude": spec.amplitude,
                "rewarp_between_levels": spec.rewarp_between_stages,
                "source_arm_id": spec.source_arm_id,
                "candidate_field": field,
                "geometry": {},
            }
            if spec.arm_id == REFERENCE_ARM_ID:
                base.update(
                    action="REFERENCE",
                    source_c6_decision_case_sha256=source_sha,
                    support=None,
                    utility=None,
                )
                source_records["reference"] = {"field": field, "geometry": {}}
            elif spec.arm_id == SOURCE_CONTEXT_ARM_ID:
                base.update(
                    action="ACCEPT",
                    source_c6_decision_case_sha256=source_sha,
                    support=support,
                    utility=utility,
                )
                source_records["context"] = {
                    "field": field,
                    "geometry": {},
                    "action": "ACCEPT",
                    "support": support,
                    "utility": utility,
                }
            else:
                base.update(
                    action=action,
                    reason=inferred["reason"],
                    exact={"status": "CERTIFIED", "certified": True, "sha256": "a" * 64},
                    direction={
                        "family": "full_resolution",
                        "factors": list(spec.factors),
                        "rewarp_between_levels": spec.rewarp_between_stages,
                    },
                    support=support,
                    utility=utility,
                )
            rows.append(base)
        execution = {
            "phase": "decision",
            "attempt_id": "attempt",
            "shard_index": 0,
            "physical_gpu": "0",
            "host": "host",
            "python": "3.10",
            "torch": "2.0",
            **STRICT_FP32_RUNTIME,
            "device": "cuda:0",
            "gpu_name": "gpu",
            "seed": 0,
            "deterministic": True,
            "labels_loaded_to_device": False,
        }
        payload = {
            "schema": DECISION_CASE_SCHEMA,
            "protocol_id": PROTOCOL_ID,
            "status": "COMPLETE",
            "strict": True,
            "case_id": "case_a",
            "shard_index": 0,
            "physical_gpu": "0",
            "decision_contract_sha256": decision_sha,
            "descriptor_checkpoint_sha256": DESCRIPTOR_CHECKPOINT_SHA256,
            "descriptor_epoch": 99,
            "descriptor_state_key_count": 386,
            "labels_loaded_to_device": False,
            "test_115_authorized": False,
            "test_split_accessed": False,
            "arms": rows,
            "resource": {"wall_sec": 1.0, "peak_cuda_bytes": 1},
            "execution": execution,
        }
        decision = json.loads(
            json.dumps(
                {
                    "shards": {"0": ["case_a"]},
                    "shard_to_physical_gpu": {"0": "0"},
                    "seed": 0,
                    "runtime_signature": {"python": "3.10", "torch": "2.0", **STRICT_FP32_RUNTIME},
                    "source_c6_decision_case_sha256": {"case_a": source_sha},
                    "source_c4_anchors": {"case_a": source_records["reference"]},
                    "source_c6_context": {"case_a": source_records["context"]},
                }
            )
        )
        path = root / "decision_complete.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path, decision, decision_sha, payload

    def _validate(self, path: Path, decision: dict[str, object], decision_sha: str) -> None:
        with (
            mock.patch("tools.analysis.run_search_gate_c7.verify_record"),
            mock.patch("tools.analysis.run_search_gate_c7._require_exact_geometry"),
        ):
            _validate_decision_case_marker(path, decision, decision_sha, "case_a")

    def test_valid_marker_passes(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            path, decision, decision_sha, _ = self._fixture(Path(temp))
            self._validate(path, decision, decision_sha)

    def test_computed_action_cannot_be_resealed_against_utility(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            path, decision, decision_sha, payload = self._fixture(Path(temp))
            payload["arms"][2]["action"] = "ROLLBACK"
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "inconsistent with frozen utility"):
                self._validate(path, decision, decision_sha)

    def test_arm_spec_and_source_records_are_frozen(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            path, decision, decision_sha, payload = self._fixture(Path(temp))
            payload["arms"][2]["descriptor"] = "altered"
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "arm identity"):
                self._validate(path, decision, decision_sha)
            payload["arms"][2]["descriptor"] = ARM_SPECS[2].descriptor
            payload["arms"][0]["candidate_field"]["array_sha256"] = "x" * 64
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "reference record"):
                self._validate(path, decision, decision_sha)


class EvaluationMarkerSemanticsTest(unittest.TestCase):
    labels = (1, 2)

    @staticmethod
    def _runtime_decision() -> dict[str, object]:
        return {
            "shards": {"0": ["case_a"]},
            "shard_to_physical_gpu": {"0": "0"},
            "seed": 0,
            "runtime_signature": {"python": "3.10", "torch": "2.0", **STRICT_FP32_RUNTIME},
        }

    @staticmethod
    def _decision_payload(actions: dict[str, str]) -> dict[str, object]:
        return {
            "schema": DECISION_CASE_SCHEMA,
            "protocol_id": PROTOCOL_ID,
            "status": "COMPLETE",
            "strict": True,
            "arms": [
                {"arm_index": spec.arm_index, "arm_id": spec.arm_id, "action": actions[spec.arm_id]}
                for spec in ARM_SPECS
            ],
        }

    def _evaluation_payload(self, actions: dict[str, str], decision_sha: str) -> dict[str, object]:
        rows = []
        for spec in ARM_SPECS:
            baseline = [0.2, 0.4]
            candidate = [0.3 + spec.arm_index * 0.01, 0.5 + spec.arm_index * 0.01]
            returned = candidate if actions[spec.arm_id] in {"REFERENCE", "ACCEPT"} else baseline
            baseline_mean = sum(baseline) / 2
            candidate_mean = sum(candidate) / 2
            returned_mean = sum(returned) / 2
            rows.append(
                {
                    "arm_index": spec.arm_index,
                    "arm_id": spec.arm_id,
                    "action": actions[spec.arm_id],
                    "baseline_dice": baseline_mean,
                    "candidate_dice": candidate_mean,
                    "capacity_delta_vs_initial": candidate_mean - baseline_mean,
                    "returned_dice": returned_mean,
                    "returned_delta_vs_initial": returned_mean - baseline_mean,
                    "source_parity_verified": (
                        True if spec.arm_id in {REFERENCE_ARM_ID, SOURCE_CONTEXT_ARM_ID} else None
                    ),
                    "per_label": [
                        {
                            "label": label,
                            "baseline_dice": base,
                            "candidate_dice": cand,
                            "returned_dice": ret,
                        }
                        for label, base, cand, ret in zip(self.labels, baseline, candidate, returned, strict=True)
                    ],
                }
            )
        return {
            "schema": EVALUATION_CASE_SCHEMA,
            "protocol_id": PROTOCOL_ID,
            "status": "COMPLETE",
            "strict": True,
            "case_id": "case_a",
            "decision_contract_sha256": "d" * 64,
            "decision_barrier_sha256": "b" * 64,
            "evaluation_contract_sha256": "e" * 64,
            "decision_case_sha256": decision_sha,
            "shard_index": 0,
            "physical_gpu": "0",
            "labels_loaded_after_barrier": True,
            "test_115_authorized": False,
            "test_split_accessed": False,
            "labels": list(self.labels),
            "arms": rows,
            "execution": {
                "phase": "evaluation",
                "attempt_id": "attempt",
                "shard_index": 0,
                "physical_gpu": "0",
                "host": "host",
                "python": "3.10",
                "torch": "2.0",
                **STRICT_FP32_RUNTIME,
                "device": "cuda:0",
                "gpu_name": "gpu",
                "seed": 0,
                "deterministic": True,
                "labels_loaded_to_device": True,
            },
        }

    def _write_fixture(self, root: Path) -> tuple[Path, str, dict[str, object], dict[str, object], dict[str, object]]:
        actions = {spec.arm_id: "ACCEPT" for spec in ARM_SPECS}
        actions[REFERENCE_ARM_ID] = "REFERENCE"
        actions[SOURCE_CONTEXT_ARM_ID] = "ROLLBACK"
        decision = self._decision_payload(actions)
        decision_path = root / "cases" / "case_a" / "decision_complete.json"
        decision_path.parent.mkdir(parents=True)
        decision_path.write_text(json.dumps(decision), encoding="utf-8")
        decision_sha = sha256_file(decision_path)
        payload = self._evaluation_payload(actions, decision_sha)
        path = decision_path.with_name("evaluation_complete.json")
        path.write_text(json.dumps(payload), encoding="utf-8")
        baseline = payload["arms"][0]["per_label"]
        evaluation = {
            "evaluation_baseline_per_label": {
                "case_a": [{"label": row["label"], "dice": row["baseline_dice"]} for row in baseline]
            },
            "source_c6_evaluation_rows": {
                "case_a": {
                    "reference": json.loads(json.dumps(payload["arms"][0])),
                    "context": json.loads(json.dumps(payload["arms"][1])),
                }
            },
        }
        return path, decision_sha, payload, self._runtime_decision(), evaluation

    def _validate(
        self,
        path: Path,
        decision_sha: str,
        decision: dict[str, object],
        evaluation: dict[str, object],
    ) -> None:
        _validate_evaluation_case_marker(
            path,
            decision=decision,
            evaluation=evaluation,
            case_id="case_a",
            decision_sha="d" * 64,
            barrier_sha="b" * 64,
            evaluation_sha="e" * 64,
            decision_case_sha=decision_sha,
            labels=self.labels,
        )

    def test_valid_marker_passes(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            path, decision_sha, _, decision, evaluation = self._write_fixture(Path(temp))
            self._validate(path, decision_sha, decision, evaluation)

    def test_action_inconsistent_return_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            path, decision_sha, payload, decision, evaluation = self._write_fixture(Path(temp))
            payload["arms"][1]["returned_dice"] = payload["arms"][1]["candidate_dice"]
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "arithmetic or action semantics"):
                self._validate(path, decision_sha, decision, evaluation)

    def test_action_drift_from_decision_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            path, decision_sha, payload, decision, evaluation = self._write_fixture(Path(temp))
            payload["arms"][2]["action"] = "ROLLBACK"
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "arm identity"):
                self._validate(path, decision_sha, decision, evaluation)

    def test_per_label_order_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            path, decision_sha, payload, decision, evaluation = self._write_fixture(Path(temp))
            payload["arms"][0]["per_label"].reverse()
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "per-label order"):
                self._validate(path, decision_sha, decision, evaluation)

    def test_sealed_source_parity_claim_is_recomputed(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            path, decision_sha, payload, decision, evaluation = self._write_fixture(Path(temp))
            payload["arms"][0]["candidate_dice"] += 0.01
            payload["arms"][0]["capacity_delta_vs_initial"] += 0.01
            payload["arms"][0]["returned_dice"] += 0.01
            payload["arms"][0]["returned_delta_vs_initial"] += 0.01
            payload["arms"][0]["per_label"][0]["candidate_dice"] += 0.01
            payload["arms"][0]["per_label"][1]["candidate_dice"] += 0.01
            payload["arms"][0]["per_label"][0]["returned_dice"] += 0.01
            payload["arms"][0]["per_label"][1]["returned_dice"] += 0.01
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "frozen C6 evaluation parity"):
                self._validate(path, decision_sha, decision, evaluation)


class CostBundleIntegrationTest(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(7)
        self.fixed_features = torch.randn(1, 8, 17, 17, 17)
        self.moving_features = torch.randn(1, 8, 17, 17, 17)
        self.fixed_image = torch.randn(1, 1, 17, 17, 17)
        self.moving_image = torch.randn(1, 1, 17, 17, 17)
        self.current = torch.zeros(1, 3, 17, 17, 17)
        self.full_mask = torch.ones(1, 1, 17, 17, 17, dtype=torch.bool)

    def test_stride_order_and_matched_support_are_exact(self) -> None:
        for stride in (4, 2, 1):
            support = _padding_support_mask(self.current, self.full_mask, stride)
            learned = build_raw_corrmlp_x1_cost_volume(
                self.fixed_features,
                self.moving_features,
                self.current,
                support,
                stride_voxels=stride,
                padding_margin=2,
                require_all_candidates_valid=True,
            )
            intensity = _intensity_raw_on_support(
                self.fixed_image,
                self.moving_image,
                self.current,
                support,
                stride,
            )
            self.assertEqual(learned.offsets, corrmlp_x1_offsets(stride))
            self.assertEqual(learned.offsets, intensity.offsets)
            self.assertTrue(torch.equal(learned.valid, intensity.valid))
            self.assertEqual(int(learned.valid_count.masked_select(support).min()), 27)
            self.assertTrue(
                torch.equal(
                    learned.valid_count.masked_select(~support),
                    torch.zeros_like(learned.valid_count.masked_select(~support)),
                )
            )

    def test_hybrid_centering_preserves_joint_support_and_source_ids(self) -> None:
        support = _padding_support_mask(self.current, self.full_mask, 1)
        learned = build_raw_corrmlp_x1_cost_volume(
            self.fixed_features,
            self.moving_features,
            self.current,
            support,
            stride_voxels=1,
            padding_margin=2,
            require_all_candidates_valid=True,
        )
        intensity = _intensity_raw_on_support(
            self.fixed_image,
            self.moving_image,
            self.current,
            support,
            1,
        )
        hybrid = equal_standardized_intensity_hybrid(learned, intensity)
        hybrid_raw = RawCandidateCostVolume(
            cost_id=hybrid.cost_id,
            costs=hybrid.standardized_costs,
            valid=hybrid.valid,
            valid_count=hybrid.valid_count,
            offsets=hybrid.offsets,
        )
        centered = _centered_for_posterior(
            hybrid_raw,
            hybrid=hybrid,
            source_ids=(learned.cost_id, intensity.cost_id),
        )
        self.assertTrue(torch.equal(centered.valid, hybrid.valid))
        self.assertEqual(centered.source_ids, (learned.cost_id, intensity.cost_id))


class ShellContractTest(unittest.TestCase):
    def test_shell_is_lf_only_and_packages_both_outcomes(self) -> None:
        source = Path("tools/runners/eval/search_gate_c7.sh").read_bytes()
        self.assertNotIn(b"\r\n", source)
        text = source.decode("utf-8")
        for token in (
            "GPU_LIST",
            "decision-pilot",
            "decision-barrier",
            "freeze-evaluation",
            "evaluation-worker",
            "__FAILED",
            "--strict-checkpoint-load",
            "Test-115 was not accessed",
        ):
            self.assertIn(token, text)

    def test_runner_sources_have_no_rf2_claim(self) -> None:
        paths = (
            Path("tools/analysis/run_search_gate_c7.py"),
            Path("tools/analysis/search_gate_c7_source.py"),
            Path("tools/runners/eval/search_gate_c7.sh"),
        )
        self.assertNotIn("rf2", "\n".join(path.read_text(encoding="utf-8").lower() for path in paths))


class HashSanityTest(unittest.TestCase):
    def test_frozen_digests_are_real_sha256(self) -> None:
        for value in (C7_POLICY_SHA256, DESCRIPTOR_CHECKPOINT_SHA256, C6_MANIFEST_SHA256):
            self.assertEqual(len(value), 64)
            self.assertEqual(hashlib.sha256(bytes.fromhex(value)).digest_size, 32)


if __name__ == "__main__":
    unittest.main()
