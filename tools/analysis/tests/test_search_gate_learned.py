from __future__ import annotations

import hashlib
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch
import torch.nn.functional as F

from models.CorrMLP.wrapper import CorrMLPSolo
from tools.analysis.search_gate_learned import (
    CORRMLP_FULL_STATE_KEY_COUNT,
    CORRMLP_IXI_LAST_EPOCH,
    CORRMLP_X1_CHANNELS,
    CORRMLP_X1_CONV_PADDING_MARGIN,
    DEFAULT_MOMENT_REDUCTION,
    build_raw_corrmlp_x1_cost_volume,
    corrmlp_x1_offsets,
    equal_standardized_intensity_hybrid,
    extract_corrmlp_x1,
    load_frozen_corrmlp_x1,
    raw_candidate_cost_volume,
    valid_corrmlp_x1_sample_mask,
)
from tools.analysis.transactional_search import OFFSETS, ZERO_OFFSET_INDEX, geometry_mask, sample_at_psi


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def unit_features(shape: tuple[int, int, int], seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return F.normalize(torch.randn((1, CORRMLP_X1_CHANNELS, *shape), generator=generator), dim=1)


class CorrMLPCheckpointContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.tempdir = tempfile.TemporaryDirectory()
        cls.checkpoint_path = Path(cls.tempdir.name) / "last.pth"
        torch.manual_seed(91)
        source = CorrMLPSolo(enc_channels=8, dec_channels=16, use_checkpoint=True)
        cls.state = source.state_dict()
        torch.save(
            {
                "epoch": CORRMLP_IXI_LAST_EPOCH,
                "state_dict": cls.state,
                "optimizer": {},
                "best_dsc": 0.0,
                "scaler": {},
            },
            cls.checkpoint_path,
        )
        cls.checkpoint_sha256 = file_sha256(cls.checkpoint_path)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.tempdir.cleanup()

    def test_full_wrapper_load_is_strict_frozen_and_buffer_free(self) -> None:
        loaded = load_frozen_corrmlp_x1(
            self.checkpoint_path,
            expected_sha256=self.checkpoint_sha256,
        )

        self.assertEqual(loaded.epoch, CORRMLP_IXI_LAST_EPOCH)
        self.assertEqual(loaded.state_key_count, CORRMLP_FULL_STATE_KEY_COUNT)
        self.assertEqual(loaded.checkpoint_sha256, self.checkpoint_sha256)
        self.assertFalse(loaded.model.training)
        self.assertFalse(any(parameter.requires_grad for parameter in loaded.model.parameters()))
        self.assertEqual(tuple(loaded.model.named_buffers()), ())
        self.assertEqual(set(loaded.model.state_dict()), set(self.state))

    def test_sha_mismatch_fails_before_deserialization(self) -> None:
        with (
            mock.patch("tools.analysis.search_gate_learned.torch.load") as load,
            self.assertRaisesRegex(RuntimeError, "SHA-256 mismatch"),
        ):
            load_frozen_corrmlp_x1(self.checkpoint_path, expected_sha256="0" * 64)
        load.assert_not_called()

    def test_wrong_epoch_is_rejected_even_when_bytes_are_authenticated(self) -> None:
        checkpoint = {"epoch": CORRMLP_IXI_LAST_EPOCH - 1, "state_dict": self.state}
        with (
            mock.patch("tools.analysis.search_gate_learned.torch.load", return_value=checkpoint),
            self.assertRaisesRegex(RuntimeError, "epoch mismatch"),
        ):
            load_frozen_corrmlp_x1(
                self.checkpoint_path,
                expected_sha256=self.checkpoint_sha256,
            )

    def test_partial_state_is_rejected_instead_of_loading_non_strictly(self) -> None:
        partial = dict(self.state)
        partial.pop(next(iter(partial)))
        checkpoint = {"epoch": CORRMLP_IXI_LAST_EPOCH, "state_dict": partial}
        with (
            mock.patch("tools.analysis.search_gate_learned.torch.load", return_value=checkpoint),
            self.assertRaisesRegex(RuntimeError, "state key count mismatch"),
        ):
            load_frozen_corrmlp_x1(
                self.checkpoint_path,
                expected_sha256=self.checkpoint_sha256,
            )

    def test_data_parallel_prefix_is_rejected_instead_of_being_stripped(self) -> None:
        altered = dict(self.state)
        first = next(iter(altered))
        altered[f"module.{first}"] = altered.pop(first)
        checkpoint = {"epoch": CORRMLP_IXI_LAST_EPOCH, "state_dict": altered}
        with (
            mock.patch("tools.analysis.search_gate_learned.torch.load", return_value=checkpoint),
            self.assertRaisesRegex(RuntimeError, "required 'net.' prefix"),
        ):
            load_frozen_corrmlp_x1(
                self.checkpoint_path,
                expected_sha256=self.checkpoint_sha256,
            )

    def test_x1_extraction_is_unary_full_resolution_fp32_and_deterministic(self) -> None:
        loaded = load_frozen_corrmlp_x1(
            self.checkpoint_path,
            expected_sha256=self.checkpoint_sha256,
        )
        image = torch.randn((1, 1, 16, 16, 16), generator=torch.Generator().manual_seed(6))

        first = extract_corrmlp_x1(loaded.model, image)
        second = extract_corrmlp_x1(loaded.model, image.clone())

        self.assertEqual(first.shape, (1, CORRMLP_X1_CHANNELS, 16, 16, 16))
        self.assertEqual(first.dtype, torch.float32)
        self.assertFalse(first.requires_grad)
        torch.testing.assert_close(first, second, atol=0, rtol=0)

    def test_extraction_rejects_a_trainable_or_training_model(self) -> None:
        loaded = load_frozen_corrmlp_x1(
            self.checkpoint_path,
            expected_sha256=self.checkpoint_sha256,
        )
        image = torch.zeros((1, 1, 16, 16, 16))
        loaded.model.train()
        with self.assertRaisesRegex(RuntimeError, "frozen and in eval mode"):
            extract_corrmlp_x1(loaded.model, image)


class CorrMLPCostVolumeTest(unittest.TestCase):
    def test_identity_features_select_the_zero_offset_deterministically(self) -> None:
        shape = (21, 22, 23)
        features = unit_features(shape, seed=5)
        psi = torch.zeros((1, 3, *shape))
        support = geometry_mask(shape, 7, features.device)

        first = build_raw_corrmlp_x1_cost_volume(features, features.clone(), psi, support, stride_voxels=4)
        second = build_raw_corrmlp_x1_cost_volume(features, features.clone(), psi, support, stride_voxels=4)

        self.assertEqual(ZERO_OFFSET_INDEX, OFFSETS.index((0, 0, 0)))
        best = first.costs.masked_fill(~first.valid, torch.inf).argmin(dim=1, keepdim=True)
        self.assertTrue(bool((best.masked_select(support) == ZERO_OFFSET_INDEX).all()))
        torch.testing.assert_close(first.costs, second.costs, atol=0, rtol=0)
        self.assertTrue(torch.equal(first.valid, second.valid))

    def test_known_positive_x_shift_preserves_scaled_offset_sign_and_order(self) -> None:
        shape = (21, 22, 23)
        moving = unit_features(shape, seed=77)
        psi = torch.zeros((1, 3, *shape))
        for stride in (2, 4):
            with self.subTest(stride=stride):
                physical_offset = (0, 0, stride)
                fixed = sample_at_psi(moving, psi, offset=physical_offset)
                support = geometry_mask(shape, 7, moving.device)

                volume = build_raw_corrmlp_x1_cost_volume(
                    fixed,
                    moving,
                    psi,
                    support,
                    stride_voxels=stride,
                )
                expected = corrmlp_x1_offsets(stride).index(physical_offset)
                best = volume.costs.masked_fill(~volume.valid, torch.inf).argmin(dim=1, keepdim=True)

                self.assertEqual(expected, 14)
                self.assertEqual(volume.offsets[expected], physical_offset)
                self.assertTrue(bool((best.masked_select(support) == expected).all()))

    def test_convolution_padding_margin_excludes_boundary_features(self) -> None:
        shape = (9, 10, 11)
        psi = torch.zeros((1, 3, *shape))
        valid = valid_corrmlp_x1_sample_mask(psi, (0, 0, 0), stride_voxels=1)
        margin = CORRMLP_X1_CONV_PADDING_MARGIN

        expected = torch.zeros_like(valid)
        expected[:, :, margin:-margin, margin:-margin, margin:-margin] = True
        self.assertTrue(torch.equal(valid, expected))

    def test_candidate_validity_uses_shifted_source_coordinates(self) -> None:
        shape = (9, 10, 11)
        psi = torch.zeros((1, 3, *shape))
        centre = valid_corrmlp_x1_sample_mask(psi, (0, 0, 0), stride_voxels=1)
        positive_x = valid_corrmlp_x1_sample_mask(psi, (0, 0, 1), stride_voxels=1)

        self.assertTrue(bool(centre[0, 0, 4, 4, -3]))
        self.assertFalse(bool(positive_x[0, 0, 4, 4, -3]))

    def test_builder_rejects_a_geometry_mask_inside_the_encoder_padding_band(self) -> None:
        shape = (9, 10, 11)
        fixed = unit_features(shape, seed=1)
        moving = unit_features(shape, seed=2)
        psi = torch.zeros((1, 3, *shape))
        unsafe = geometry_mask(shape, 1, fixed.device)

        with self.assertRaisesRegex(ValueError, "contaminated by encoder padding"):
            build_raw_corrmlp_x1_cost_volume(fixed, moving, psi, unsafe, stride_voxels=1)


class EqualHybridTest(unittest.TestCase):
    def test_components_are_standardized_separately_before_equal_fusion(self) -> None:
        shape = (3, 4, 5)
        generator = torch.Generator().manual_seed(44)
        learned_costs = torch.randn((1, len(OFFSETS), *shape), generator=generator)
        intensity_base = torch.randn((1, len(OFFSETS), *shape), generator=generator)
        intensity_costs = 100.0 * intensity_base + 37.0
        valid = torch.ones_like(learned_costs, dtype=torch.bool)
        offsets = corrmlp_x1_offsets(4)
        learned = raw_candidate_cost_volume("learned", learned_costs, valid, offsets=offsets)
        intensity = raw_candidate_cost_volume("intensity", intensity_costs, valid.clone(), offsets=offsets)
        intensity_reference = raw_candidate_cost_volume(
            "intensity-reference", intensity_base, valid.clone(), offsets=offsets
        )

        fused = equal_standardized_intensity_hybrid(learned, intensity)
        reference = equal_standardized_intensity_hybrid(learned, intensity_reference)
        equal_mean = 0.5 * fused.learned.standardized_costs + 0.5 * fused.intensity.standardized_costs

        torch.testing.assert_close(fused.standardized_costs, fused.fusion.standardized_costs, atol=0, rtol=0)
        self.assertEqual(DEFAULT_MOMENT_REDUCTION, "centered_two_pass_fp32")
        self.assertEqual(fused.learned.standardization_mode, DEFAULT_MOMENT_REDUCTION)
        self.assertEqual(fused.intensity.standardization_mode, DEFAULT_MOMENT_REDUCTION)
        self.assertEqual(fused.fusion.standardization_mode, DEFAULT_MOMENT_REDUCTION)
        torch.testing.assert_close(
            fused.standardized_costs,
            reference.standardized_costs,
            atol=3e-6,
            rtol=0,
        )
        self.assertGreater(float((fused.standardized_costs - equal_mean).abs().max()), 0.0)
        means = fused.standardized_costs.mean(dim=1)
        variances = fused.standardized_costs.square().mean(dim=1)
        torch.testing.assert_close(means, torch.zeros_like(means), atol=1e-6, rtol=0)
        torch.testing.assert_close(variances, torch.ones_like(variances), atol=2e-5, rtol=0)

    def test_hybrid_rejects_even_one_support_bit_of_mismatch(self) -> None:
        costs = torch.zeros((1, len(OFFSETS), 3, 4, 5))
        valid = torch.ones_like(costs, dtype=torch.bool)
        other_valid = valid.clone()
        other_valid[0, 0, 0, 0, 0] = False
        offsets = corrmlp_x1_offsets(2)
        learned = raw_candidate_cost_volume("learned", costs, valid, offsets=offsets)
        intensity = raw_candidate_cost_volume("intensity", costs.clone(), other_valid, offsets=offsets)

        with self.assertRaisesRegex(RuntimeError, "exact same candidate-support mask"):
            equal_standardized_intensity_hybrid(learned, intensity)

    def test_hybrid_rejects_tampered_valid_count(self) -> None:
        costs = torch.randn((1, len(OFFSETS), 3, 4, 5), generator=torch.Generator().manual_seed(4))
        valid = torch.ones_like(costs, dtype=torch.bool)
        learned = raw_candidate_cost_volume("learned", costs, valid, offsets=corrmlp_x1_offsets(1))
        object.__setattr__(learned, "valid_count", torch.zeros_like(learned.valid_count))

        with self.assertRaisesRegex(RuntimeError, "valid_count"):
            equal_standardized_intensity_hybrid(learned, learned)

    def test_hybrid_rejects_either_flat_component_instead_of_becoming_the_other_component(self) -> None:
        shape = (3, 4, 5)
        valid = torch.ones((1, len(OFFSETS), *shape), dtype=torch.bool)
        flat = raw_candidate_cost_volume(
            "flat",
            torch.full(valid.shape, 17.0),
            valid,
            offsets=corrmlp_x1_offsets(1),
        )
        informative = raw_candidate_cost_volume(
            "informative",
            torch.randn(valid.shape, generator=torch.Generator().manual_seed(92)),
            valid.clone(),
            offsets=corrmlp_x1_offsets(1),
        )

        for learned, intensity in ((flat, informative), (informative, flat)):
            with (
                self.subTest(flat_component=learned.cost_id),
                self.assertRaisesRegex(FloatingPointError, "no jointly informative component support"),
            ):
                equal_standardized_intensity_hybrid(learned, intensity)

    def test_hybrid_support_is_the_exact_joint_component_informative_intersection(self) -> None:
        shape = (3, 4, 5)
        generator = torch.Generator().manual_seed(105)
        valid = torch.ones((1, len(OFFSETS), *shape), dtype=torch.bool)
        learned_costs = torch.randn(valid.shape, generator=generator)
        intensity_costs = torch.randn(valid.shape, generator=generator)
        learned_costs[:, :, 0, 0, 0] = 3.0
        intensity_costs[:, :, 1, 2, 3] = -4.0
        learned = raw_candidate_cost_volume("learned", learned_costs, valid, offsets=corrmlp_x1_offsets(1))
        intensity = raw_candidate_cost_volume(
            "intensity", intensity_costs, valid.clone(), offsets=corrmlp_x1_offsets(1)
        )

        fused = equal_standardized_intensity_hybrid(learned, intensity)
        expected = valid.clone()
        expected[:, :, 0, 0, 0] = False
        expected[:, :, 1, 2, 3] = False

        self.assertTrue(torch.equal(fused.valid, expected))
        self.assertTrue(torch.equal(fused.valid_count, expected.sum(dim=1, keepdim=True)))
        self.assertTrue(bool((fused.standardized_costs.masked_select(~expected) == 0.0).all()))


if __name__ == "__main__":
    unittest.main()
