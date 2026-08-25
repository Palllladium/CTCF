from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from tools.analysis import search_gate_c5_workers as c5_workers
from tools.analysis.run_artifacts import sha256_file
from tools.analysis.search_gate_c5 import (
    ALL_CONTRAST_SPECS,
    ARM_SPECS,
    ARM_SPECS_BY_ID,
    CAPACITY_FAMILY_ID,
    CONTRAST_IDS_BY_FAMILY,
    CONTRAST_SPECS,
    EVALUATION_LABEL_IDS,
    HISTORICAL_ANCHOR_ARM_IDS,
    INCREMENTAL_FAMILY_ID,
    INFERENCE_FAMILY_IDS,
    PRIMARY_REFERENCE_ARM_ID,
    REGIONAL_REPAIR_LABEL_IDS,
    SELECTABLE_ARM_IDS,
    SELECTOR_IDS,
    SELECTOR_SPECS,
    SELECTOR_ZERO_FAMILY_ID,
)
from tools.analysis.search_gate_c5_contracts import EXPECTED_SUPPORT_CONTRACT, array_sha256
from tools.analysis.search_gate_c5_workers import (
    ReachBankResult,
    _apply_amplitude_and_clip,
    _assert_decision_label_free,
    _assert_exact_geometry,
    _assert_unique_field_records,
    _build_reach_bank,
    _central_difference_zyx,
    _contrast_vectors,
    _decode_direction,
    _geometry_bundle,
    _materialize_arm,
    _ngf_diagnostic,
    _ngf_reference,
    _persist_candidate,
    _publish_validated_marker,
    _reach_support_record,
    _selector_rows,
    _verify_baseline_geometry,
)
from tools.analysis.search_gate_cost_volume import masked_vector_rms
from tools.analysis.search_gate_metrics import DETJ_DIAGNOSTICS, MATHEMATICAL_SDLOGJ_CROP2
from tools.analysis.search_gate_multiscale import DecodedProposal
from tools.analysis.transactional_search import save_flow_npz_atomic


def _arm_delta(case_index: int, arm_index: int) -> float:
    return (arm_index + 1) * 1e-5 + (case_index - 28.5) * 1e-7


def _label_rows(baseline: float, observed: float, value_key: str) -> list[dict[str, float | int]]:
    return [
        {
            "label": label,
            "baseline_dice": baseline,
            value_key: observed,
            "dice_delta": observed - baseline,
        }
        for label in EVALUATION_LABEL_IDS
    ]


def _synthetic_case(case_id: str, case_index: int) -> tuple[dict[str, object], dict[str, object]]:
    baseline = 0.70 + case_index * 1e-5
    arm_dice: dict[str, float] = {}
    decision_arms = []
    evaluation_arms = []
    for spec in ARM_SPECS:
        delta = _arm_delta(case_index, spec.arm_index)
        candidate = baseline + delta
        arm_dice[spec.arm_id] = candidate
        decision_arms.append(
            {
                "arm_index": spec.arm_index,
                "arm_id": spec.arm_id,
                "reach_id": spec.reach_id,
                "stride_voxels": spec.stride_voxels,
                "post_rms_amplitude": spec.post_rms_amplitude,
                "centre_beta": spec.centre_beta,
                "historical_anchor": spec.historical_anchor,
                "exact": {"certified": True},
                "candidate_field": {
                    "root_id": "target_c5_heavy",
                    "relative_path": f"cases/{case_id}/{spec.arm_id}.npz",
                },
                "proposal": {"clip_rms_retention": 0.99, "clip_cosine": 1.0},
                "geometry": {
                    MATHEMATICAL_SDLOGJ_CROP2: {
                        "metric_id": MATHEMATICAL_SDLOGJ_CROP2,
                        "status": "OK",
                        "value": 0.30 + spec.arm_index * 1e-4,
                    }
                },
                "mathematical_sdlogj_delta": spec.arm_index * 1e-4,
                "support": {"retention": 1.0},
                "utilities": {key: {"improvement": delta} for key in ("ncc5", "ncc7", "ncc9", "mind_d2", "ngf")},
            }
        )
        evaluation_arms.append(
            {
                "arm_index": spec.arm_index,
                "arm_id": spec.arm_id,
                "baseline_dice": baseline,
                "candidate_dice": candidate,
                "capacity_dice_delta": delta,
                "historical_c4_dice_parity_verified": spec.historical_anchor,
                "per_label": _label_rows(baseline, candidate, "candidate_dice"),
            }
        )

    decision_selectors = []
    evaluation_selectors = []
    for spec in SELECTOR_SPECS:
        selected = SELECTABLE_ARM_IDS[spec.selector_index]
        returned = arm_dice[selected]
        decision_selectors.append(
            {
                "selector_index": spec.selector_index,
                "selector_id": spec.selector_id,
                "action": "RETURN_CANDIDATE",
                "selected_arm_id": selected,
                "eligible_arm_ids": [selected],
            }
        )
        evaluation_selectors.append(
            {
                "selector_index": spec.selector_index,
                "selector_id": spec.selector_id,
                "action": "RETURN_CANDIDATE",
                "selected_arm_id": selected,
                "baseline_dice": baseline,
                "returned_dice": returned,
                "dice_delta": returned - baseline,
                "per_label": _label_rows(baseline, returned, "returned_dice"),
            }
        )
    return (
        {
            "arms": decision_arms,
            "selectors": decision_selectors,
            "resources": {"elapsed_sec": float(case_index + 1), "peak_gpu_bytes": 1_000_000 + case_index},
        },
        {"arms": evaluation_arms, "selectors": evaluation_selectors},
    )


def _csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def selector_arm_rows() -> list[dict[str, object]]:
    return [
        {
            "arm_id": arm_id,
            "exact": {"certified": True},
            "support": {"retention": 1.0},
            "proposal": {"clip_rms_retention": 1.0},
            "utilities": {
                "ncc7": {"improvement": -1.0},
                "mind_d2": {"improvement": -1.0},
            },
            "mathematical_sdlogj_delta": 0.0,
            "candidate_field": {
                "root_id": "target_c5_heavy",
                "relative_path": f"{arm_id}.npz",
            },
        }
        for arm_id in SELECTABLE_ARM_IDS
    ]


class BankAndReachTest(unittest.TestCase):
    def test_four_raw_banks_produce_twelve_posterior_directions(self) -> None:
        shape = (13, 13, 13)
        generator = torch.Generator().manual_seed(51)
        fixed = torch.randn((1, 1, *shape), generator=generator)
        moving = torch.randn((1, 1, *shape), generator=generator)
        initial = torch.zeros((1, 3, *shape))
        mask = torch.zeros((1, 1, *shape), dtype=torch.bool)
        mask[:, :, 4:-4, 4:-4, 4:-4] = True

        banks = [_build_reach_bank(fixed, moving, initial, mask, stride) for stride in range(1, 5)]
        directions = [
            _decode_direction(bank.volume, beta)[1] for bank in banks for beta in (0.0, np.log(2.0), np.log(4.0))
        ]

        self.assertEqual(len(banks), 4)
        self.assertEqual(len(directions), 12)
        self.assertTrue(all(tuple(direction.displacement.shape) == (1, 3, *shape) for direction in directions))

    def test_reach_support_distinguishes_raw_validity_from_informative_standardization(self) -> None:
        bank = ReachBankResult(
            volume=SimpleNamespace(),
            elapsed_sec=1.0,
            generation_count=100,
            raw_all_candidates_valid_count=100,
            standardized_informative_count=41,
        )
        record = _reach_support_record(SimpleNamespace(reach_id="S1", stride_voxels=1), bank)

        self.assertEqual(record["all_candidates_valid_count"], 100)
        self.assertTrue(record["all_candidates_valid"])
        self.assertEqual(record["standardized_informative_count"], 41)
        self.assertEqual(record["standardized_informative_fraction"], 0.41)

    def test_s4_fails_closed_when_frozen_generation_support_is_not_valid(self) -> None:
        shape = (9, 9, 9)
        image = torch.zeros((1, 1, *shape))
        initial = torch.zeros((1, 3, *shape))
        mask = torch.ones((1, 1, *shape), dtype=torch.bool)
        with self.assertRaisesRegex(RuntimeError, "S4 is not fully valid"):
            _build_reach_bank(image, image, initial, mask, 4)


class MaterializationOrderTest(unittest.TestCase):
    def test_component_only_detj_diagnostics_are_validated_without_a_scalar_value(self) -> None:
        shape = (7, 7, 7)
        field = torch.zeros((1, 3, *shape))
        mask = torch.ones((1, 1, *shape), dtype=torch.bool)
        geometry = _geometry_bundle(field, mask)
        diagnostics = geometry[DETJ_DIAGNOSTICS]

        self.assertEqual(diagnostics["status"], "OK")
        self.assertIsNone(diagnostics["value"])
        self.assertEqual(diagnostics["components"]["invalid_count"], 0.0)
        _assert_exact_geometry(geometry, label="identity")

        diagnostics["components"]["invalid_count"] = 1.0
        with self.assertRaisesRegex(RuntimeError, "component-only detJ diagnostics"):
            _assert_exact_geometry(geometry, label="tampered")

    def test_baseline_parity_compares_component_only_detj_diagnostics(self) -> None:
        shape = (7, 7, 7)
        field = torch.zeros((1, 3, *shape))
        mask = torch.ones((1, 1, *shape), dtype=torch.bool)
        observed = _geometry_bundle(field, mask)
        expected = _geometry_bundle(field.clone(), mask)

        _verify_baseline_geometry(observed, expected, case_id="identity")

        expected[DETJ_DIAGNOSTICS]["components"]["detj_min"] = 0.5
        with self.assertRaisesRegex(RuntimeError, "baseline geometry differs from frozen C4"):
            _verify_baseline_geometry(observed, expected, case_id="tampered")

    def test_amplitude_is_applied_after_rms_match_and_before_clip(self) -> None:
        shape = (5, 5, 5)
        initial = torch.zeros((1, 3, *shape))
        mask = torch.ones((1, 1, *shape), dtype=torch.bool)
        matched = torch.ones_like(initial)
        target_rms = masked_vector_rms(matched, mask)
        post = SimpleNamespace(
            displacement=matched,
            target_rms=target_rms,
            source_rms=target_rms,
            output_rms=target_rms,
            rms_scale_factor=1.0,
            smoothing_passes=1,
            collar_width=7,
        )
        observed: dict[str, torch.Tensor] = {}

        def clip(current: torch.Tensor, requested: torch.Tensor, *_: object, **__: object):
            observed["requested"] = requested.clone()
            return current + requested, {"operator": "fixture"}

        arm = ARM_SPECS_BY_ID["int_s2_a05_b0"]
        decoded = DecodedProposal("fixture", matched, tuple((0, 0, 0) for _ in range(27)))
        with (
            patch("tools.analysis.search_gate_c5_workers.postprocess_and_match_rms", return_value=post),
            patch("tools.analysis.search_gate_c5_workers.certified_local_clip_candidate", side_effect=clip),
        ):
            candidate, proposal, _ = _apply_amplitude_and_clip(decoded, arm, initial, matched, mask)

        torch.testing.assert_close(observed["requested"], matched * 0.5, atol=0, rtol=0)
        torch.testing.assert_close(candidate, matched * 0.5, atol=0, rtol=0)
        self.assertEqual(proposal["post_rms_amplitude"], 0.5)
        self.assertEqual(proposal["clip_rms_retention"], 1.0)

    def test_posterior_diagnostics_use_generation_not_rms_mask(self) -> None:
        shape = (3, 3, 3)
        field = torch.zeros((1, 3, *shape))
        rms_mask = torch.ones((1, 1, *shape), dtype=torch.bool)
        generation_mask = torch.zeros_like(rms_mask)
        generation_mask[..., 1, 1, 1] = True
        geometry = {
            "CTCF_MATHEMATICAL_SDLOGJ_CENTRAL_CROP2_UNMASKED_DDOF0_FAILCLOSED_V1": {
                "metric_id": "CTCF_MATHEMATICAL_SDLOGJ_CENTRAL_CROP2_UNMASKED_DDOF0_FAILCLOSED_V1",
                "status": "OK",
                "value": 0.0,
            }
        }
        arm = ARM_SPECS_BY_ID["int_s2_a05_b0"]
        with (
            patch(
                "tools.analysis.search_gate_c5_workers._apply_amplitude_and_clip",
                return_value=(field, {"clip_rms_retention": 1.0}, {}),
            ),
            patch(
                "tools.analysis.search_gate_c5_workers._persist_candidate",
                return_value=(
                    {"root_id": "target_c5_heavy", "relative_path": "x", "npz_sha256": "a", "array_sha256": "b"},
                    {"certified": True},
                    None,
                ),
            ),
            patch("tools.analysis.search_gate_c5_workers._geometry_bundle", return_value=geometry),
            patch("tools.analysis.search_gate_c5_workers._assert_exact_geometry"),
            patch(
                "tools.analysis.search_gate_c5_workers._utility_bundle",
                return_value=({"retention": 1.0}, {"ncc7": {}, "mind_d2": {}}),
            ),
            patch(
                "tools.analysis.search_gate_c5_workers._assert_support_contract", return_value=EXPECTED_SUPPORT_CONTRACT
            ),
            patch("tools.analysis.search_gate_c5_workers._metric_value", return_value=0.0),
            patch("tools.analysis.search_gate_c5_workers._posterior_record", return_value={}) as posterior,
        ):
            _materialize_arm(
                case_id="fixture",
                arm=arm,
                decoded=SimpleNamespace(),
                volume=SimpleNamespace(),
                posterior=SimpleNamespace(),
                bank_elapsed=0.0,
                initial=field,
                rms_reference=field,
                mask=rms_mask,
                generation_mask=generation_mask,
                fixed_norm=field[:, :1],
                moving_norm=field[:, :1],
                fixed_mind=field,
                moving_mind=field,
                ngf_reference=SimpleNamespace(),
                baseline_geometry=geometry,
                decision={},
            )
        self.assertIs(posterior.call_args.args[2], generation_mask)


class PersistenceAndSelectorTest(unittest.TestCase):
    def test_failed_validation_does_not_publish_a_complete_marker(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            marker = Path(directory) / "case" / "decision_complete.json"

            def reject(_: object) -> None:
                raise RuntimeError("invalid marker")

            with self.assertRaisesRegex(RuntimeError, "invalid marker"):
                _publish_validated_marker(marker, {"status": "COMPLETE"}, reject)
            self.assertFalse(marker.exists())

    def test_decision_worker_reuses_the_validated_pilot_case(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            marker = root / "cases" / "subject_001" / "decision_complete.json"
            marker.parent.mkdir(parents=True)
            marker.write_text("{}", encoding="utf-8")
            decision = {
                "shards": {"0": ["subject_001"]},
                "shard_to_physical_gpu": {"0": "2"},
            }
            with (
                patch.object(c5_workers, "validate_decision_case_marker") as validate_case,
                patch.object(c5_workers, "run_decision_case") as run_case,
                patch.object(c5_workers, "validate_worker_marker"),
            ):
                c5_workers.run_decision_worker(
                    case_ids=["subject_001"],
                    shard_index=0,
                    physical_gpu="2",
                    attempt_id="attempt",
                    run_root=root,
                    decision=decision,
                    decision_sha256="a" * 64,
                    device=torch.device("cpu"),
                )
            validate_case.assert_called_once()
            run_case.assert_not_called()

    def test_historical_anchor_mismatch_fails_before_duplication(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            c3 = root / "c3"
            c4 = root / "c4"
            c5 = root / "c5"
            for path in (c3, c4, c5):
                path.mkdir()
            stored = torch.zeros((1, 3, 3, 3, 3))
            anchor_path = c4 / "anchor.npz"
            save_flow_npz_atomic(anchor_path, stored)
            field = {
                "root_id": "source_c4_heavy",
                "relative_path": "anchor.npz",
                "npz_sha256": __import__("hashlib").sha256(anchor_path.read_bytes()).hexdigest(),
                "array_sha256": array_sha256(stored.numpy()),
            }
            decision = {
                "roots": {
                    "source_c3_heavy": str(c3),
                    "source_c4_heavy": str(c4),
                    "target_c5_heavy": str(c5),
                },
                "source_c4_anchors": {"fixture": {"intensity_s1": {"field": field}, "intensity_s2": {"field": field}}},
            }
            with self.assertRaisesRegex(RuntimeError, "differs from frozen C4 anchor"):
                _persist_candidate(
                    case_id="fixture",
                    arm=ARM_SPECS_BY_ID[HISTORICAL_ANCHOR_ARM_IDS[0]],
                    candidate=torch.ones_like(stored),
                    decision=decision,
                )
            self.assertEqual(list(c5.rglob("*.npz")), [])

    def test_duplicate_materialized_field_owner_is_rejected(self) -> None:
        field = {"root_id": "target_c5_heavy", "relative_path": "same.npz"}
        with self.assertRaisesRegex(RuntimeError, "duplicate"):
            _assert_unique_field_records([{"candidate_field": field}, {"candidate_field": field}])

    def test_selector_action_is_recomputed_from_label_free_signals(self) -> None:
        rows = selector_arm_rows()
        self.assertTrue(all(row["action"] == "RETURN_BASELINE" for row in _selector_rows(rows)))
        rows[0]["utilities"]["ncc7"]["improvement"] = 1.0  # type: ignore[index]
        rows[0]["utilities"]["mind_d2"]["improvement"] = 1.0  # type: ignore[index]
        selected = _selector_rows(rows)
        self.assertEqual(selected[0]["selected_arm_id"], SELECTABLE_ARM_IDS[0])
        self.assertEqual(selected[0]["returned_field"], rows[0]["candidate_field"])

    def test_label_or_decision_mutation_is_not_silent(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "leaked"):
            _assert_decision_label_free({"candidate_dice": 0.8})
        rows = selector_arm_rows()
        rows[0]["proposal"]["clip_rms_retention"] = 1.1  # type: ignore[index]
        with self.assertRaises(ValueError):
            _selector_rows(rows)


class NGFTest(unittest.TestCase):
    def test_gradient_axes_and_identity_similarity_are_explicit(self) -> None:
        z = torch.arange(7, dtype=torch.float32).view(1, 1, 7, 1, 1)
        image = z.expand(1, 1, 7, 7, 7).contiguous()
        gradient = _central_difference_zyx(image)
        self.assertEqual(float(gradient[0, 0, 3, 3, 3]), 1.0)
        self.assertEqual(float(gradient[0, 1, 3, 3, 3]), 0.0)
        self.assertEqual(float(gradient[0, 2, 3, 3, 3]), 0.0)

        initial = torch.zeros((1, 3, 7, 7, 7))
        mask = torch.ones((1, 1, 7, 7, 7), dtype=torch.bool)
        reference = _ngf_reference(image, image, initial, mask)
        diagnostic = _ngf_diagnostic(reference, image, initial)
        self.assertAlmostEqual(diagnostic["baseline_similarity"], 1.0)
        self.assertAlmostEqual(diagnostic["candidate_similarity"], 1.0)
        self.assertAlmostEqual(diagnostic["improvement"], 0.0)

    def test_eta_fails_closed_for_constant_images(self) -> None:
        image = torch.zeros((1, 1, 7, 7, 7))
        initial = torch.zeros((1, 3, 7, 7, 7))
        mask = torch.ones((1, 1, 7, 7, 7), dtype=torch.bool)
        with self.assertRaisesRegex(RuntimeError, "eta"):
            _ngf_reference(image, image, initial, mask)


class FinalizerArithmeticTest(unittest.TestCase):
    def test_all_eight_families_use_the_declared_references(self) -> None:
        n = 58
        capacity = {arm_id: np.full(n, index / 10_000.0) for index, arm_id in enumerate(SELECTABLE_ARM_IDS)}
        baseline = np.full(n, 0.75)
        candidates = {arm_id: baseline + values for arm_id, values in capacity.items()}
        selector_delta = {
            selector_id: np.full(n, (index + 1) / 1000.0) for index, selector_id in enumerate(SELECTOR_IDS)
        }
        selectors = {selector_id: baseline + values for selector_id, values in selector_delta.items()}
        label_zero = {label_id: np.full(n, label_id / 100_000.0) for label_id in EVALUATION_LABEL_IDS}
        label_reference = {label_id: np.full(n, -label_id / 100_000.0) for label_id in REGIONAL_REPAIR_LABEL_IDS}

        vectors = _contrast_vectors(capacity, candidates, selector_delta, selectors, label_zero, label_reference)

        self.assertEqual(float(vectors[CAPACITY_FAMILY_ID][f"capacity::{SELECTABLE_ARM_IDS[3]}::vs_zero"][0]), 0.0003)
        reference_index = SELECTABLE_ARM_IDS.index(PRIMARY_REFERENCE_ARM_ID)
        incremental_id = f"incremental::{SELECTABLE_ARM_IDS[-1]}::vs_{PRIMARY_REFERENCE_ARM_ID}"
        self.assertAlmostEqual(
            float(vectors[INCREMENTAL_FAMILY_ID][incremental_id][0]),
            (len(SELECTABLE_ARM_IDS) - 1 - reference_index) / 10_000.0,
        )
        self.assertEqual(
            float(vectors[SELECTOR_ZERO_FAMILY_ID][f"selector::{SELECTOR_IDS[0]}::vs_zero"][0]),
            0.001,
        )

    def test_finalizer_rejects_missing_or_reordered_arm_vectors(self) -> None:
        n = 58
        capacity = {arm_id: np.zeros(n) for arm_id in SELECTABLE_ARM_IDS[:-1]}
        candidates = {arm_id: np.zeros(n) for arm_id in SELECTABLE_ARM_IDS}
        selectors = {selector_id: np.zeros(n) for selector_id in SELECTOR_IDS}
        labels = {label_id: np.zeros(n) for label_id in EVALUATION_LABEL_IDS}
        repairs = {label_id: np.zeros(n) for label_id in REGIONAL_REPAIR_LABEL_IDS}
        with self.assertRaisesRegex(ValueError, "all arm vectors"):
            _contrast_vectors(capacity, candidates, selectors, selectors, labels, repairs)


class FullShapeFinalizerTest(unittest.TestCase):
    def test_full_58_by_36_by_5_finalizer_inventory_and_hashes(self) -> None:
        case_ids = [f"subject_{index:03d}" for index in range(58)]
        frozen_sha = "a" * 64
        decision = {
            "case_ids": case_ids,
            "source_contract_sha256": frozen_sha,
            "full_policy_sha256": frozen_sha,
            "decision_policy_sha256": frozen_sha,
            "arm_specs_sha256": frozen_sha,
            "selector_specs_sha256": frozen_sha,
            "offset_table_sha256": frozen_sha,
            "support_contract_sha256": frozen_sha,
            "contrast_contract_sha256": frozen_sha,
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            barrier: dict[str, object] = {"decision_case_sha256": {}}
            for case_index, case_id in enumerate(case_ids):
                case_root = root / "cases" / case_id
                case_root.mkdir(parents=True)
                decision_case, evaluation_case = _synthetic_case(case_id, case_index)
                decision_path = case_root / "decision_complete.json"
                evaluation_path = case_root / "evaluation_complete.json"
                decision_path.write_text(
                    json.dumps(decision_case, sort_keys=True, separators=(",", ":")), encoding="utf-8"
                )
                evaluation_path.write_text(
                    json.dumps(evaluation_case, sort_keys=True, separators=(",", ":")), encoding="utf-8"
                )
                barrier["decision_case_sha256"][case_id] = sha256_file(decision_path)  # type: ignore[index]

            with (
                patch.object(c5_workers, "validate_decision_case_marker"),
                patch.object(c5_workers, "validate_evaluation_case_marker"),
            ):
                artifacts = c5_workers.finalize_c5(
                    run_root=root,
                    decision=decision,
                    decision_sha256="b" * 64,
                    barrier=barrier,
                    barrier_sha256="c" * 64,
                    evaluation_contract={},
                    evaluation_contract_sha256="d" * 64,
                )

            expected_artifacts = {
                "per_arm",
                "per_selector",
                "per_arm_label_dice",
                "per_selector_label_dice",
                "diagnostic_utilities",
                "arm_summary",
                "selector_summary",
                "preregistered_contrasts",
                "resource_summary",
                "hypotheses",
                "summary",
                "next_branch",
                "c5_manifest",
            }
            self.assertEqual(set(artifacts), expected_artifacts)
            self.assertEqual(len(_csv_rows(root / "per_arm.csv")), 58 * 36)
            self.assertEqual(len(_csv_rows(root / "per_selector.csv")), 58 * 5)
            self.assertEqual(len(_csv_rows(root / "per_arm_label_dice.csv")), 58 * 36 * 30)
            self.assertEqual(len(_csv_rows(root / "per_selector_label_dice.csv")), 58 * 5 * 30)
            self.assertEqual(len(_csv_rows(root / "diagnostic_utilities.csv")), 58 * 36 * 5)
            self.assertEqual(len(_csv_rows(root / "arm_summary.csv")), 36)
            self.assertEqual(len(_csv_rows(root / "selector_summary.csv")), 5)
            self.assertEqual(len(_csv_rows(root / "resource_summary.csv")), 58)
            self.assertEqual(len(_csv_rows(root / "preregistered_contrasts.csv")), 124)

            contrast_rows = _csv_rows(root / "preregistered_contrasts.csv")
            self.assertEqual(
                [row["contrast_id"] for row in contrast_rows],
                [spec.contrast_id for spec in ALL_CONTRAST_SPECS],
            )
            self.assertEqual(len(CONTRAST_SPECS), 92)
            self.assertEqual(len(ALL_CONTRAST_SPECS), 124)
            self.assertEqual(
                {family: len(CONTRAST_IDS_BY_FAMILY[family]) for family in INFERENCE_FAMILY_IDS},
                {
                    "capacity_vs_zero": 36,
                    "capacity_vs_c4_intensity_s2": 35,
                    "factor_adjacent_marginals": 7,
                    "factor_trend_interactions": 4,
                    "selector_vs_zero": 5,
                    "selector_vs_c4_intensity_s2": 5,
                    "primary_selector_labels_vs_zero": 30,
                    "primary_selector_risk_labels_vs_c4_intensity_s2": 2,
                },
            )
            per_arm = _csv_rows(root / "per_arm.csv")
            self.assertEqual(
                [(row["case_id"], row["arm_id"]) for row in per_arm],
                [(case_id, arm_id) for case_id in case_ids for arm_id in SELECTABLE_ARM_IDS],
            )
            per_selector = _csv_rows(root / "per_selector.csv")
            self.assertEqual(
                [(row["case_id"], row["selector_id"]) for row in per_selector],
                [(case_id, selector_id) for case_id in case_ids for selector_id in SELECTOR_IDS],
            )

            summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
            manifest = json.loads((root / "c5_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(
                summary["design_counts"],
                {
                    "arms": 36,
                    "selectors": 5,
                    "inference_families": 8,
                    "cases": 58,
                    "c4_anchor_references_reused": 116,
                    "new_c5_heavy_fields": 1972,
                },
            )
            self.assertFalse(summary["test_115_authorized"])
            self.assertFalse(summary["test_split_accessed"])
            self.assertFalse(summary["labels_used_for_decision"])
            self.assertEqual(manifest["decision_case_sha256"], barrier["decision_case_sha256"])
            self.assertEqual(len(manifest["evaluation_case_sha256"]), 58)
            self.assertEqual(manifest["next_branch"], summary["next_branch"])
            self.assertEqual(set(manifest["files"]), expected_artifacts - {"c5_manifest"})
            suffix = {
                "hypotheses": ".json",
                "summary": ".json",
                "next_branch": ".json",
                "c5_manifest": ".json",
            }
            for name, digest in artifacts.items():
                path = root / f"{name}{suffix.get(name, '.csv')}"
                self.assertEqual(sha256_file(path), digest)


if __name__ == "__main__":
    unittest.main()
