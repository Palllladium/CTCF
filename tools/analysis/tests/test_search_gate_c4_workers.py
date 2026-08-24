from __future__ import annotations

import copy
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from tools.analysis.run_artifacts import atomic_write_json, sha256_file
from tools.analysis.run_search_gate_c4 import arm_contract_rows, support_contract
from tools.analysis.search_gate_c4 import (
    ARM_SPECS_BY_ID,
    BOOTSTRAP_CONFIDENCE,
    BOOTSTRAP_METHOD_ID,
    BOOTSTRAP_RESAMPLES,
    BOOTSTRAP_SEED,
    BRANCH_ADVANCE,
    BRANCH_CLOSE,
    BRANCH_GEOMETRY,
    BRANCH_UTILITY,
    CONTRAST_SPECS,
    EXPECTED_CASE_COUNT,
    POLICY_MEAN_DICE_DELTA_MIN,
    PRIMARY_NCC_IMPROVEMENT_MIN,
    PRIMARY_NCC_WINDOW,
    PRIMARY_UTILITY_ID,
    SCIENTIFIC_REFERENCE_ARM_ID,
    SELECTABLE_ARM_IDS,
    ArmEvidence,
    GeometryComparison,
    PairedSummary,
)
from tools.analysis.search_gate_c4_contracts import (
    array_sha256,
    payload_sha256,
    validate_decision_case_marker,
)
from tools.analysis.search_gate_c4_workers import (
    _assert_decision_label_free,
    _field_record,
    _materialize_arm,
    _resolve_field,
    _scientific_reference_geometry_mean,
    _source_c3_baseline,
    _source_c3_raw_reference,
    _utility_and_action,
    classify_next_branch,
    finalize_c4,
    run_decision_case,
    run_evaluation_case,
    simultaneous_paired_summaries,
)
from tools.analysis.transactional_search import (
    build_proposal,
    geometry_mask,
    masked_zscore,
    mind_ssc,
    sample_at_psi,
    save_flow_npz_atomic,
)
from utils import dice_per_label


def frozen_summary(mean: float, *, low: float | None = None, median: float | None = None) -> PairedSummary:
    median_value = mean if median is None else median
    low_value = mean if low is None else low
    return PairedSummary(
        n=EXPECTED_CASE_COUNT,
        mean=mean,
        median=median_value,
        ci_low=low_value,
        ci_high=max(mean, low_value) + 0.0001,
        improved=EXPECTED_CASE_COUNT if median_value > 0 else 0,
        worsened=EXPECTED_CASE_COUNT if median_value < 0 else 0,
        tied=EXPECTED_CASE_COUNT if median_value == 0 else 0,
        bootstrap_resamples=BOOTSTRAP_RESAMPLES,
        bootstrap_seed=BOOTSTRAP_SEED,
        bootstrap_confidence=BOOTSTRAP_CONFIDENCE,
        bootstrap_method=BOOTSTRAP_METHOD_ID,
        simultaneous_family_size=len(CONTRAST_SPECS),
    )


def arm_evidence(
    arm_id: str,
    *,
    capacity: float = 0.003,
    incremental: float = 0.001,
    policy: float = 0.002,
    candidate_geometry: float = 0.3,
    reference_geometry: float = 0.3,
) -> ArmEvidence:
    return ArmEvidence(
        arm_id=arm_id,
        capacity_vs_baseline=frozen_summary(capacity, low=capacity - 0.0005),
        incremental_vs_reference=(
            None if arm_id == SCIENTIFIC_REFERENCE_ARM_ID else frozen_summary(incremental, low=incremental / 2)
        ),
        policy_vs_baseline=frozen_summary(policy, low=policy / 2),
        geometry=(
            GeometryComparison(
                "CTCF_MATHEMATICAL_SDLOGJ_CENTRAL_CROP2_UNMASKED_DDOF0_FAILCLOSED_V1",
                candidate_geometry,
                reference_geometry,
                True,
            ),
        ),
        all_work_units_complete=True,
        all_exact_certified=True,
    )


class SimultaneousBootstrapTest(unittest.TestCase):
    @staticmethod
    def family() -> dict[str, np.ndarray]:
        output = {spec.contrast_id: np.zeros(EXPECTED_CASE_COUNT, dtype=np.float64) for spec in CONTRAST_SPECS}
        output[CONTRAST_SPECS[0].contrast_id] = np.linspace(0.001, 0.004, EXPECTED_CASE_COUNT)
        output[CONTRAST_SPECS[1].contrast_id] = np.linspace(-0.002, 0.003, EXPECTED_CASE_COUNT)
        return output

    def test_max_stat_bootstrap_is_deterministic_and_simultaneous(self) -> None:
        inputs = self.family()
        first_id, second_id = (CONTRAST_SPECS[index].contrast_id for index in (0, 1))

        observed = simultaneous_paired_summaries(inputs)
        repeated = simultaneous_paired_summaries(inputs)

        self.assertEqual(observed, repeated)
        self.assertEqual(tuple(observed), tuple(spec.contrast_id for spec in CONTRAST_SPECS))
        self.assertAlmostEqual(observed[first_id].mean, float(inputs[first_id].mean()))
        self.assertAlmostEqual(
            observed[first_id].ci_high - observed[first_id].mean,
            observed[second_id].ci_high - observed[second_id].mean,
        )
        self.assertEqual(observed[first_id].bootstrap_resamples, 10_000)
        self.assertEqual(observed[first_id].bootstrap_seed, 0)
        self.assertEqual(observed[first_id].bootstrap_method, BOOTSTRAP_METHOD_ID)
        self.assertEqual(observed[first_id].simultaneous_family_size, len(CONTRAST_SPECS))

    def test_bootstrap_rejects_missing_nonfinite_and_unpaired_inputs(self) -> None:
        with self.assertRaises(ValueError):
            simultaneous_paired_summaries({})
        short = self.family()
        short[CONTRAST_SPECS[0].contrast_id] = np.zeros(EXPECTED_CASE_COUNT - 1)
        with self.assertRaises(ValueError):
            simultaneous_paired_summaries(short)
        invalid = self.family()
        invalid[CONTRAST_SPECS[0].contrast_id][2] = np.nan
        with self.assertRaises(ValueError):
            simultaneous_paired_summaries(invalid)


class BranchClassificationTest(unittest.TestCase):
    @staticmethod
    def classify(rows: tuple[ArmEvidence, ...]) -> dict[str, object]:
        return classify_next_branch(rows, {row.arm_id: row.policy_vs_baseline for row in rows})

    def test_all_four_preregistered_branches_are_reachable(self) -> None:
        viable = tuple(arm_evidence(arm_id) for arm_id in SELECTABLE_ARM_IDS)
        self.assertEqual(self.classify(viable)["branch_id"], BRANCH_ADVANCE)

        weak = tuple(arm_evidence(arm_id, capacity=0.0005) for arm_id in SELECTABLE_ARM_IDS)
        self.assertEqual(self.classify(weak)["branch_id"], BRANCH_CLOSE)

        bad_geometry = tuple(
            arm_evidence(arm_id, candidate_geometry=0.31, reference_geometry=0.3) for arm_id in SELECTABLE_ARM_IDS
        )
        self.assertEqual(self.classify(bad_geometry)["branch_id"], BRANCH_GEOMETRY)

        weak_policy = tuple(
            arm_evidence(arm_id, policy=POLICY_MEAN_DICE_DELTA_MIN / 10) for arm_id in SELECTABLE_ARM_IDS
        )
        self.assertEqual(self.classify(weak_policy)["branch_id"], BRANCH_UTILITY)

    def test_classifier_refuses_a_second_policy_story(self) -> None:
        rows = tuple(arm_evidence(arm_id) for arm_id in SELECTABLE_ARM_IDS)
        policies = {row.arm_id: row.policy_vs_baseline for row in rows}
        policies[rows[0].arm_id] = replace(policies[rows[0].arm_id], mean=0.0)
        with self.assertRaisesRegex(ValueError, "disagree"):
            classify_next_branch(rows, policies)

    def test_geometry_reference_is_the_common_mind_arm_not_zero_update(self) -> None:
        rows = {
            SCIENTIFIC_REFERENCE_ARM_ID: [
                {"baseline_geometry": 0.1, "candidate_geometry": 0.3 + index * 1e-5}
                for index in range(EXPECTED_CASE_COUNT)
            ]
        }
        expected = float(np.mean([row["candidate_geometry"] for row in rows[SCIENTIFIC_REFERENCE_ARM_ID]]))

        observed = _scientific_reference_geometry_mean(rows)

        self.assertAlmostEqual(observed, expected)
        self.assertNotAlmostEqual(observed, 0.1)


class DecisionIsolationTest(unittest.TestCase):
    def test_label_free_payload_accepts_geometry_and_hashes(self) -> None:
        _assert_decision_label_free(
            {
                "labels_loaded_to_device": False,
                "geometry": {"metric_id": "MATHEMATICAL_SDLOGJ", "value": 0.3},
                "source_image_array_sha256": "a" * 64,
            }
        )

    def test_dice_key_segmentation_text_and_raw_container_are_rejected(self) -> None:
        payloads = (
            {"candidate_dice": 0.8},
            {"note": "segmentation was used"},
            {"source": "/data/subject.pkl"},
        )
        for payload in payloads:
            with self.subTest(payload=payload), self.assertRaisesRegex(RuntimeError, "leaked"):
                _assert_decision_label_free(payload)


class PersistenceTest(unittest.TestCase):
    def test_field_resolver_detects_npz_and_array_tampering(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "cases" / "subject" / "candidate.npz"
            field = torch.zeros((1, 3, 4, 5, 6), dtype=torch.float32)
            save_flow_npz_atomic(path, field)
            record = _field_record(path, root, array_sha256(field.numpy()))

            self.assertEqual(_resolve_field(root, record), path.resolve())

            bad_array = copy.deepcopy(record)
            bad_array["array_sha256"] = "0" * 64
            with self.assertRaisesRegex(RuntimeError, "array changed"):
                _resolve_field(root, bad_array)

            bad_npz = copy.deepcopy(record)
            bad_npz["npz_sha256"] = "0" * 64
            with self.assertRaisesRegex(RuntimeError, "bytes changed"):
                _resolve_field(root, bad_npz)

    def test_field_resolver_rejects_traversal(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            record = {
                "relative_path": "../escape.npz",
                "npz_sha256": "0" * 64,
                "array_sha256": "0" * 64,
            }
            with self.assertRaisesRegex(RuntimeError, "traversal"):
                _resolve_field(Path(directory), record)

    def test_historical_inputs_are_consumed_from_authenticated_contract_records(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            initial = torch.zeros((1, 3, 4, 5, 6), dtype=torch.float32)
            historical = torch.full_like(initial, 0.25)
            path = root / "history" / "raw_conf.npz"
            save_flow_npz_atomic(path, historical)
            record = _field_record(path, root, array_sha256(historical.numpy()))
            decision = {
                "source_c3_heavy_root": str(root),
                "source_historical": {
                    "subject_fixture": {
                        "raw_conf_requested_field": record,
                        "source_decision_case_sha256": "a" * 64,
                    }
                },
            }
            residual, provenance = _source_c3_raw_reference(decision, "subject_fixture", initial, torch.device("cpu"))

            torch.testing.assert_close(residual, historical, atol=0, rtol=0)
            self.assertEqual(provenance["source_decision_case_sha256"], "a" * 64)
            self.assertEqual(
                _source_c3_baseline({"evaluation_baseline_dice": {"subject_fixture": 0.75}}, "subject_fixture"),
                0.75,
            )


class FrozenUtilityTest(unittest.TestCase):
    def test_identical_candidate_rolls_back_under_exact_frozen_ncc7(self) -> None:
        shape = (17, 17, 17)
        generator = torch.Generator().manual_seed(42)
        fixed = torch.randn((1, 1, *shape), generator=generator)
        moving = fixed.clone()
        initial = torch.zeros((1, 3, *shape))
        mask = torch.zeros((1, 1, *shape), dtype=torch.bool)
        mask[:, :, 4:-4, 4:-4, 4:-4] = True
        support_contract = {
            "support_id": "C4_COMMON_COLLAR7_NCC7_V1",
            "utility_id": PRIMARY_UTILITY_ID,
            "utility_window": PRIMARY_NCC_WINDOW,
            "utility_retention_min": 0.99,
            "descriptor_retention_policy": "diagnostic_only_nonempty",
            "improvement_min": PRIMARY_NCC_IMPROVEMENT_MIN,
        }

        support, utility, action = _utility_and_action(
            fixed,
            moving,
            initial,
            initial,
            mask,
            exact_certified=True,
            support_contract=support_contract,
        )

        self.assertEqual(support["support_id"], support_contract["support_id"])
        self.assertEqual(support["retention"], 1.0)
        self.assertEqual(utility["utility_id"], PRIMARY_UTILITY_ID)
        self.assertAlmostEqual(utility["improvement"], 0.0)
        self.assertFalse(action["accept"])
        self.assertTrue(action["rollback"])

    def test_utility_rejects_an_unfrozen_identity(self) -> None:
        shape = (17, 17, 17)
        image = torch.zeros((1, 1, *shape))
        field = torch.zeros((1, 3, *shape))
        mask = torch.zeros((1, 1, *shape), dtype=torch.bool)
        mask[:, :, 4:-4, 4:-4, 4:-4] = True
        support = {
            "support_id": "wrong",
            "utility_id": "NOT_NCC7",
            "utility_window": PRIMARY_NCC_WINDOW,
            "utility_retention_min": 0.99,
            "descriptor_retention_policy": "diagnostic_only_nonempty",
            "improvement_min": PRIMARY_NCC_IMPROVEMENT_MIN,
        }
        with self.assertRaisesRegex(RuntimeError, "frozen NCC7"):
            _utility_and_action(
                image,
                image,
                field,
                field,
                mask,
                exact_certified=True,
                support_contract=support,
            )


class ExactFailureBoundaryTest(unittest.TestCase):
    def test_failed_save_reload_certificate_stops_before_geometry_and_utility(self) -> None:
        shape = (9, 9, 9)
        field = torch.zeros((1, 3, *shape))
        mask = torch.ones((1, 1, *shape), dtype=torch.bool)
        post = SimpleNamespace(
            displacement=field,
            proposal_multiplier=2.0,
            smoothing_passes=1,
            collar_width=7,
            rms_target_source_id="source",
            source_rms=0.0,
            target_rms=0.0,
            output_rms=0.0,
            rms_scale_factor=1.0,
        )
        exact = {"status": "FAILED", "certified": False}
        support = {
            "support_id": "C4_COMMON_COLLAR7_NCC7_V1",
            "utility_id": PRIMARY_UTILITY_ID,
            "window": PRIMARY_NCC_WINDOW,
            "utility_retention_min": 0.99,
            "descriptor_retention_policy": "diagnostic_only_nonempty",
            "improvement_min": PRIMARY_NCC_IMPROVEMENT_MIN,
        }
        with (
            tempfile.TemporaryDirectory() as directory,
            patch("tools.analysis.search_gate_c4_workers.postprocess_and_match_rms", return_value=post),
            patch(
                "tools.analysis.search_gate_c4_workers.certified_local_clip_candidate",
                return_value=(field, {"operator": "fixture"}),
            ),
            patch("tools.analysis.search_gate_c4_workers.save_reload_certify", return_value=(field, exact)),
            patch(
                "tools.analysis.search_gate_c4_workers._geometry_bundle",
                side_effect=AssertionError("geometry must not run"),
            ),
            patch(
                "tools.analysis.search_gate_c4_workers._utility_and_action",
                side_effect=AssertionError("utility must not run"),
            ),
            self.assertRaisesRegex(RuntimeError, "exact certificate failed"),
        ):
            _materialize_arm(
                case_id="subject_fixture",
                arm=ARM_SPECS_BY_ID["mind_d1_s1"],
                decoded=SimpleNamespace(displacement=field),
                initial=field,
                source_rms_reference=field,
                mask=mask,
                fixed_norm=torch.zeros((1, 1, *shape)),
                moving_norm=torch.zeros((1, 1, *shape)),
                heavy_root=Path(directory),
                support_contract=support,
                support_contract_sha256="a" * 64,
                proposal_diagnostics={},
            )


class SyntheticPipelineSmokeTest(unittest.TestCase):
    def test_real_decision_evaluation_and_58_case_finalizer(self) -> None:
        case_ids = [f"subject_{index:03d}" for index in range(EXPECTED_CASE_COUNT)]
        first_case = case_ids[0]
        shape = (17, 17, 17)
        device = torch.device("cpu")
        generator = np.random.default_rng(7)
        atlas = generator.normal(size=(1, *shape)).astype(np.float32)
        case = generator.normal(size=(1, *shape)).astype(np.float32)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_heavy = root / "source_heavy"
            target_heavy = root / "target_heavy"
            run_root = root / "run"
            source_heavy.mkdir()
            target_heavy.mkdir()
            run_root.mkdir()

            image_records: dict[str, dict[str, object]] = {}
            for identity, array in (("atlas", atlas), ("case", case)):
                path = source_heavy / f"{identity}.npy"
                np.save(path, array)
                image_records[identity] = {
                    "path": str(path.resolve()),
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                    "shape": list(array.shape),
                    "array_sha256": array_sha256(array),
                }
            decision_images = {
                "atlas": image_records["atlas"],
                **{case_id: image_records["case"] for case_id in case_ids},
            }

            initial = torch.zeros((1, 3, *shape), dtype=torch.float32)
            initial_path = source_heavy / "initial.npz"
            save_flow_npz_atomic(initial_path, initial)
            initial_record = _field_record(initial_path, source_heavy, array_sha256(initial.numpy()))

            moving = torch.from_numpy(atlas).unsqueeze(0)
            fixed = torch.from_numpy(case).unsqueeze(0)
            mask4 = geometry_mask(shape, 4, device)
            fixed_norm = masked_zscore(fixed, mask4)
            moving_norm = masked_zscore(moving, mask4)
            fixed_mind = mind_ssc(fixed_norm, radius=1, dilation=2)
            moving_mind = mind_ssc(moving_norm, radius=1, dilation=2)
            legacy = build_proposal(
                fixed,
                moving,
                initial,
                mask4,
                feature="mind",
                orientation="target_centered",
                collar_width=4,
                mind_radius=1,
                mind_dilation=2,
                fixed_feature_override=fixed_mind,
                moving_feature_override=moving_mind,
            )
            historical = (initial + 2.0 * legacy.displacement).float()
            historical_path = source_heavy / "raw_conf_post1.npz"
            save_flow_npz_atomic(historical_path, historical)
            historical_record = _field_record(
                historical_path,
                source_heavy,
                array_sha256(historical.numpy()),
            )

            arms = arm_contract_rows()
            support = support_contract()
            decision = {
                "case_ids": case_ids,
                "shards": {"0": case_ids},
                "shard_to_physical_gpu": {"0": "2"},
                "seed": 0,
                "runtime_signature": {"python": "fixture", "torch": "fixture"},
                "image_inputs": decision_images,
                "source_c3_heavy_root": str(source_heavy.resolve()),
                "heavy_root": str(target_heavy.resolve()),
                "source_initial": {case_id: {"field": initial_record} for case_id in case_ids},
                "source_historical": {
                    case_id: {
                        "raw_conf_requested_field": historical_record,
                        "source_decision_case_sha256": "a" * 64,
                    }
                    for case_id in case_ids
                },
                "arm_specs": arms,
                "arm_specs_sha256": payload_sha256(arms),
                "offset_table_sha256": "b" * 64,
                "support_contract": support,
                "support_contract_sha256": payload_sha256(support),
            }
            decision_sha = "c" * 64
            decision_execution = {
                "phase": "decision",
                "attempt_id": "A_fixture",
                "shard_index": 0,
                "physical_gpu": "2",
                "seed": 0,
                "deterministic": True,
                "python": "fixture",
                "torch": "fixture",
                "device": "cuda:0",
                "host": "fixture",
                "gpu_name": "fixture-H100",
                "labels_loaded_to_device": False,
            }
            first_decision_path = run_decision_case(
                case_id=first_case,
                shard_index=0,
                physical_gpu="2",
                run_root=run_root,
                decision=decision,
                decision_sha256=decision_sha,
                device=device,
                execution=decision_execution,
            )
            first_decision = json.loads(first_decision_path.read_text(encoding="utf-8"))
            reordered = copy.deepcopy(first_decision)
            reordered["arms"] = list(reversed(reordered["arms"]))
            with self.assertRaisesRegex(RuntimeError, "arm order"):
                validate_decision_case_marker(reordered, decision, decision_sha, verify_heavy_bytes=True)

            decision_hashes = {first_case: sha256_file(first_decision_path)}
            for case_id in case_ids[1:]:
                payload = copy.deepcopy(first_decision)
                payload["case_id"] = case_id
                path = run_root / "cases" / case_id / "decision_complete.json"
                atomic_write_json(path, payload)
                decision_hashes[case_id] = sha256_file(path)
            barrier = {"decision_case_sha256": decision_hashes}
            barrier_sha = "d" * 64

            labels = (1, 2)
            moving_seg = torch.from_numpy(generator.integers(1, 3, size=(1, *shape), dtype=np.int16))
            fixed_seg = torch.from_numpy(generator.integers(1, 3, size=(1, *shape), dtype=np.int16))
            baseline_per_label = dice_per_label(
                sample_at_psi(moving_seg.unsqueeze(0).float(), initial, mode="nearest").long(),
                fixed_seg.unsqueeze(0).long(),
                labels,
            )
            baseline = float(baseline_per_label.mean())
            source = {
                "evaluation_baseline_dice": {case_id: baseline for case_id in case_ids},
                "source_c3": {"run_id": "fixture-c3"},
            }
            evaluation_execution = {
                "phase": "evaluation",
                "attempt_id": "A_fixture",
                "shard_index": 0,
                "physical_gpu": "2",
                "seed": 0,
                "deterministic": True,
                "python": "fixture",
                "torch": "fixture",
                "device": "cuda:0",
                "host": "fixture",
                "gpu_name": "fixture-H100",
                "labels_loaded_after_barrier": True,
            }
            first_evaluation_path = run_evaluation_case(
                case_id=first_case,
                dataset_item=(torch.from_numpy(atlas), torch.from_numpy(case), moving_seg, fixed_seg),
                labels=labels,
                run_root=run_root,
                source=source,
                decision=decision,
                decision_sha256=decision_sha,
                barrier=barrier,
                barrier_sha256=barrier_sha,
                device=device,
                execution=evaluation_execution,
            )
            first_evaluation = json.loads(first_evaluation_path.read_text(encoding="utf-8"))
            for case_id in case_ids[1:]:
                payload = copy.deepcopy(first_evaluation)
                payload["case_id"] = case_id
                payload["decision_case_sha256"] = decision_hashes[case_id]
                atomic_write_json(run_root / "cases" / case_id / "evaluation_complete.json", payload)

            artifacts = finalize_c4(
                run_root=run_root,
                source=source,
                decision=decision,
                decision_sha256=decision_sha,
                barrier=barrier,
                barrier_sha256=barrier_sha,
            )
            summary = json.loads((run_root / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["n_cases"], EXPECTED_CASE_COUNT)
            self.assertEqual(summary["status"], "COMPLETE")
            self.assertIn("c4_manifest", artifacts)
            self.assertEqual(len(list(target_heavy.rglob("*.npz"))), 10)


if __name__ == "__main__":
    unittest.main()
