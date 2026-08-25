from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from tools.analysis.search_gate_c5 import (
    ARM_SPECS_BY_ID,
    CAPACITY_FAMILY_ID,
    EVALUATION_LABEL_IDS,
    HISTORICAL_ANCHOR_ARM_IDS,
    INCREMENTAL_FAMILY_ID,
    PRIMARY_REFERENCE_ARM_ID,
    REGIONAL_REPAIR_LABEL_IDS,
    SELECTABLE_ARM_IDS,
    SELECTOR_IDS,
    SELECTOR_ZERO_FAMILY_ID,
)
from tools.analysis.search_gate_c5_contracts import EXPECTED_SUPPORT_CONTRACT, array_sha256
from tools.analysis.search_gate_c5_workers import (
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
    _selector_rows,
    _verify_baseline_geometry,
)
from tools.analysis.search_gate_cost_volume import masked_vector_rms
from tools.analysis.search_gate_metrics import DETJ_DIAGNOSTICS
from tools.analysis.search_gate_multiscale import DecodedProposal
from tools.analysis.transactional_search import save_flow_npz_atomic


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

        banks = [_build_reach_bank(fixed, moving, initial, mask, stride)[0] for stride in range(1, 5)]
        directions = [_decode_direction(bank, beta)[1] for bank in banks for beta in (0.0, np.log(2.0), np.log(4.0))]

        self.assertEqual(len(banks), 4)
        self.assertEqual(len(directions), 12)
        self.assertTrue(all(tuple(direction.displacement.shape) == (1, 3, *shape) for direction in directions))

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


if __name__ == "__main__":
    unittest.main()
