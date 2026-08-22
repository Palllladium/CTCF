from __future__ import annotations

import math
import sys
import tempfile
import unittest
from dataclasses import FrozenInstanceError
from pathlib import Path
from types import ModuleType

import numpy as np
import torch

from tools.analysis import search_gate_metrics
from tools.analysis.search_gate_metrics import (
    DETJ_DIAGNOSTICS,
    DIAGNOSTIC_SDLOGJ_POSITIVE_ONLY_CROP2,
    DIGITAL_DECOMPOSITION,
    LEARN2REG_SHIFTED_SDLOGJ,
    LEARN2REG_SHIFTED_SDLOGJ_MASKED,
    LEGACY_SHIFTED_J,
    MATHEMATICAL_SDLOGJ_CROP2,
    MATHEMATICAL_SDLOGJ_FULL,
    METRIC_SPECS,
    MetricFailClosedError,
    central_jacobian_determinant,
    compute_bundle,
    compute_metric,
    learn2reg_jacobian_determinant,
)
from tools.analysis.transactional_search import load_flow_npz, save_flow_npz_atomic
from utils.field import _digital_determinants, digital_fold_percent, logdet_std_from_flow

SHAPE = (9, 10, 11)


def coordinates(shape: tuple[int, int, int]) -> np.ndarray:
    """[3,D,H,W] voxel coordinates in the (z, y, x) channel order of the project."""
    grids = np.meshgrid(*[np.arange(n, dtype=np.float64) for n in shape], indexing="ij")
    return np.stack(grids, axis=0)


def affine_flow(matrix: np.ndarray, shape: tuple[int, int, int] = SHAPE) -> torch.Tensor:
    """Displacement of the global affine map T(p) = matrix @ p, so detJ == det(matrix)."""
    displacement = np.einsum("cj,jzyx->czyx", matrix - np.eye(3), coordinates(shape))
    return torch.from_numpy(displacement[None].astype(np.float32))


def polynomial_flow(shape: tuple[int, int, int] = SHAPE, amplitude: float = 0.05) -> torch.Tensor:
    z, y, x = coordinates(shape)
    displacement = np.stack(
        [
            amplitude * (0.30 * z + 0.011 * y * y - 0.007 * x * z),
            amplitude * (-0.20 * y + 0.009 * z * x + 0.004 * y * y),
            amplitude * (0.17 * x - 0.013 * z * y + 0.006 * x * x),
        ],
        axis=0,
    )
    return torch.from_numpy(displacement[None].astype(np.float32))


def random_flow(seed: int, shape: tuple[int, int, int] = SHAPE, scale: float = 0.15) -> torch.Tensor:
    generator = np.random.default_rng(seed)
    return torch.from_numpy((scale * generator.standard_normal((1, 3, *shape))).astype(np.float32))


def cleanroom_official_detj(disp: np.ndarray) -> np.ndarray:
    """Independent pointwise oracle for the pinned FP32-derivative convention."""
    field = np.asarray(disp)[0].astype(np.float32, copy=False)
    output = np.empty(tuple(size - 4 for size in field.shape[1:]), dtype=np.float64)
    for z in range(2, field.shape[1] - 2):
        for y in range(2, field.shape[2] - 2):
            for x in range(2, field.shape[3] - 2):
                point = [z, y, x]
                matrix = np.eye(3, dtype=np.float64)
                for axis in range(3):
                    before, after = list(point), list(point)
                    before[axis] -= 1
                    after[axis] += 1
                    derivative = np.float32(0.5) * (
                        field[:, after[0], after[1], after[2]] - field[:, before[0], before[1], before[2]]
                    )
                    matrix[:, axis] += derivative.astype(np.float64)
                output[z - 2, y - 2, x - 2] = np.linalg.det(matrix)
    return output


def cleanroom_official_sdlogj(disp: np.ndarray, mask: np.ndarray | None = None) -> float:
    log_jac_det = np.log(np.clip(cleanroom_official_detj(disp) + 3.0, 1e-9, 1e9))
    if mask is None:
        return float(log_jac_det.std(ddof=0))
    return float(log_jac_det[mask[2:-2, 2:-2, 2:-2].astype(bool)].std(ddof=0))


def oracle_central_detj(flow: torch.Tensor, crop: int) -> np.ndarray:
    """Independent central-difference detJ: per-axis loop, zero derivative on the outer slice."""
    displacement = flow.detach().cpu().numpy().astype(np.float64)[0]
    gradient = np.zeros((3, 3, *displacement.shape[1:]), dtype=np.float64)
    for axis in range(3):
        forward = [slice(None)] * 3
        backward = [slice(None)] * 3
        target = [slice(None)] * 3
        forward[axis] = slice(2, None)
        backward[axis] = slice(0, -2)
        target[axis] = slice(1, -1)
        for c in range(3):
            gradient[(c, axis, *target)] = 0.5 * (displacement[(c, *forward)] - displacement[(c, *backward)])
    matrix = gradient + np.eye(3).reshape(3, 3, 1, 1, 1)
    det = np.linalg.det(np.moveaxis(matrix, (0, 1), (-2, -1)))
    if crop == 0:
        return det
    return det[crop:-crop, crop:-crop, crop:-crop]


class OracleAgreementTest(unittest.TestCase):
    def test_identity_field_is_rigid(self):
        flow = torch.zeros(1, 3, *SHAPE)
        self.assertTrue(np.allclose(central_jacobian_determinant(flow, crop=2), 1.0))
        for metric_id in (MATHEMATICAL_SDLOGJ_FULL, MATHEMATICAL_SDLOGJ_CROP2, LEARN2REG_SHIFTED_SDLOGJ):
            self.assertEqual(compute_metric(metric_id, flow).value, 0.0)
        self.assertEqual(compute_metric(DIGITAL_DECOMPOSITION, flow).value, 0.0)

    def test_legacy_identity_matches_the_frozen_function(self):
        flow = torch.zeros(1, 3, *SHAPE)
        legacy = compute_metric(LEGACY_SHIFTED_J, flow).value
        self.assertEqual(legacy, float(logdet_std_from_flow(flow)))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
    def test_legacy_cpu_cuda_difference_stays_within_declared_tolerance(self):
        flow = random_flow(101)
        cpu = compute_metric(LEGACY_SHIFTED_J, flow).value
        cuda = compute_metric(LEGACY_SHIFTED_J, flow.cuda()).value
        self.assertLessEqual(abs(cpu - cuda), 1e-6)

    def test_affine_determinant_matches_closed_form(self):
        matrix = np.array([[1.10, 0.30, -0.08], [0.05, 0.95, -0.20], [0.15, 0.02, 1.07]])
        det = central_jacobian_determinant(affine_flow(matrix), crop=2)
        self.assertTrue(np.allclose(det, np.linalg.det(matrix), atol=1e-5))

    def test_channel_axis_pairing_is_zyx(self):
        matrix = np.array([[1.10, 0.30, -0.08], [0.05, 0.95, -0.20], [0.15, 0.02, 1.07]])
        flow = affine_flow(matrix)
        permuted = flow[:, [1, 2, 0]].contiguous()
        self.assertNotAlmostEqual(
            float(central_jacobian_determinant(flow, crop=2).mean()),
            float(central_jacobian_determinant(permuted, crop=2).mean()),
            places=3,
        )

    def test_polynomial_field_matches_independent_oracle(self):
        flow = polynomial_flow()
        for crop in (0, 2):
            self.assertTrue(
                np.allclose(central_jacobian_determinant(flow, crop=crop), oracle_central_detj(flow, crop), atol=1e-12)
            )

    def test_learn2reg_parity_on_random_fields(self):
        for seed in range(5):
            flow = random_flow(seed)
            expected = cleanroom_official_sdlogj(flow.numpy())
            self.assertAlmostEqual(compute_metric(LEARN2REG_SHIFTED_SDLOGJ, flow).value, expected, delta=1e-12)
            self.assertTrue(np.allclose(learn2reg_jacobian_determinant(flow), cleanroom_official_detj(flow.numpy())))

    def test_learn2reg_masked_parity(self):
        flow = random_flow(11)
        mask = np.zeros(SHAPE, dtype=np.int64)
        mask[2:-2, 3:-2, 2:-3] = 1
        value = compute_metric(LEARN2REG_SHIFTED_SDLOGJ_MASKED, flow, torch.from_numpy(mask)).value
        self.assertAlmostEqual(value, cleanroom_official_sdlogj(flow.numpy(), mask), delta=1e-12)

    def test_legacy_parity_with_frozen_function(self):
        for seed in range(3):
            flow = random_flow(seed + 20)
            self.assertEqual(compute_metric(LEGACY_SHIFTED_J, flow).value, float(logdet_std_from_flow(flow)))

    def test_parity_after_fp32_save_reload(self):
        flow = polynomial_flow()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "flow.npz"
            save_flow_npz_atomic(path, flow)
            reloaded = load_flow_npz(path)
        for metric_id in (MATHEMATICAL_SDLOGJ_CROP2, LEARN2REG_SHIFTED_SDLOGJ, LEGACY_SHIFTED_J):
            self.assertEqual(compute_metric(metric_id, flow).value, compute_metric(metric_id, reloaded).value)


class FailClosedTest(unittest.TestCase):
    def folded_flow(self) -> torch.Tensor:
        flow = polynomial_flow().clone()
        flow[0, 0, 4, 5, 5] = -6.0
        flow[0, 0, 6, 5, 5] = 6.0
        return flow

    def test_mathematical_metric_refuses_non_positive_determinant(self):
        flow = self.folded_flow()
        self.assertLess(float(central_jacobian_determinant(flow, crop=2).min()), 0.0)
        for metric_id in (MATHEMATICAL_SDLOGJ_FULL, MATHEMATICAL_SDLOGJ_CROP2):
            with self.assertRaises(MetricFailClosedError):
                compute_metric(metric_id, flow)

    def test_mathematical_metric_refuses_non_finite_determinant(self):
        flow = polynomial_flow().clone()
        flow[0, 1, 5, 5, 5] = float("inf")
        with self.assertRaises(MetricFailClosedError):
            compute_metric(MATHEMATICAL_SDLOGJ_CROP2, flow)

    def test_positive_only_diagnostic_reports_what_it_dropped(self):
        result = compute_metric(DIAGNOSTIC_SDLOGJ_POSITIVE_ONLY_CROP2, self.folded_flow())
        self.assertGreater(result.components["dropped_count"], 0.0)
        self.assertLess(result.components["kept_fraction"], 1.0)
        self.assertTrue(math.isfinite(result.value))

    def test_positive_only_is_defined_on_one_voxel_crop(self):
        result = compute_metric(DIAGNOSTIC_SDLOGJ_POSITIVE_ONLY_CROP2, torch.zeros(1, 3, 5, 5, 5))
        self.assertEqual(result.value, 0.0)
        self.assertEqual(result.components["kept_count"], 1.0)

    def test_shifted_metrics_still_report_on_a_folded_field(self):
        flow = self.folded_flow()
        self.assertTrue(math.isfinite(compute_metric(LEARN2REG_SHIFTED_SDLOGJ, flow).value))
        self.assertTrue(math.isfinite(compute_metric(LEGACY_SHIFTED_J, flow).value))

    def test_clamp_limits_are_reached_on_both_sides(self):
        amplitude = 1.0e4
        extreme = polynomial_flow().clone()
        extreme[0, 0, 4, 5, 5], extreme[0, 0, 6, 5, 5] = -amplitude, amplitude
        extreme[0, 1, 5, 4, 5], extreme[0, 1, 5, 6, 5] = -amplitude, amplitude
        extreme[0, 2, 5, 5, 4], extreme[0, 2, 5, 5, 6] = -amplitude, amplitude
        detj = learn2reg_jacobian_determinant(extreme)
        self.assertLess(float(detj.min()) + 3.0, 1e-9)
        self.assertGreater(float(detj.max()) + 3.0, 1e9)
        expected = float(np.log(np.clip(detj + 3.0, 1e-9, 1e9)).std())
        self.assertAlmostEqual(compute_metric(LEARN2REG_SHIFTED_SDLOGJ, extreme).value, expected, delta=1e-9)


class CropAndConventionTest(unittest.TestCase):
    def shell_perturbed(self, shell: int) -> torch.Tensor:
        flow = polynomial_flow().clone()
        flow[0, :, shell, :, :] += 0.4
        return flow

    def test_crop2_ignores_the_first_boundary_shell(self):
        base = compute_metric(MATHEMATICAL_SDLOGJ_CROP2, polynomial_flow()).value
        self.assertEqual(compute_metric(MATHEMATICAL_SDLOGJ_CROP2, self.shell_perturbed(0)).value, base)

    def test_crop2_sees_the_second_boundary_shell(self):
        base = compute_metric(MATHEMATICAL_SDLOGJ_CROP2, polynomial_flow()).value
        self.assertNotAlmostEqual(compute_metric(MATHEMATICAL_SDLOGJ_CROP2, self.shell_perturbed(1)).value, base)

    def test_full_crop_sees_the_first_boundary_shell(self):
        base = compute_metric(MATHEMATICAL_SDLOGJ_FULL, polynomial_flow()).value
        self.assertNotAlmostEqual(compute_metric(MATHEMATICAL_SDLOGJ_FULL, self.shell_perturbed(0)).value, base)

    def test_full_and_crop2_are_different_metrics(self):
        flow = polynomial_flow()
        self.assertNotAlmostEqual(
            compute_metric(MATHEMATICAL_SDLOGJ_FULL, flow).value,
            compute_metric(MATHEMATICAL_SDLOGJ_CROP2, flow).value,
        )

    def test_removing_the_shift_changes_the_learn2reg_value(self):
        flow = random_flow(7)
        detj = central_jacobian_determinant(flow, crop=2)
        unshifted = float(np.log(np.clip(detj, 1e-9, 1e9)).std())
        self.assertNotAlmostEqual(compute_metric(LEARN2REG_SHIFTED_SDLOGJ, flow).value, unshifted)

    def test_ddof_choice_is_visible_on_a_small_volume(self):
        flow = random_flow(3, shape=(5, 5, 5))
        detj = central_jacobian_determinant(flow, crop=0)
        shifted = np.log(np.clip(detj + 3.0, 1e-9, 1e9))
        legacy = compute_metric(LEGACY_SHIFTED_J, flow).value
        self.assertAlmostEqual(legacy, float(shifted.std(ddof=1)), delta=1e-6)
        population = float(shifted.std(ddof=0))
        self.assertAlmostEqual(legacy / population, math.sqrt(shifted.size / (shifted.size - 1)), delta=1e-6)
        self.assertGreater(abs(legacy - population) / population, 1e-3)

    def test_declared_fp32_derivative_chain_is_stable(self):
        flow = random_flow(9)
        value = compute_metric(LEARN2REG_SHIFTED_SDLOGJ, flow).value
        self.assertAlmostEqual(value, cleanroom_official_sdlogj(flow.numpy()), delta=1e-12)
        self.assertEqual(compute_metric(LEARN2REG_SHIFTED_SDLOGJ, flow.double()).value, value)
        promoted_first = float(np.log(np.clip(central_jacobian_determinant(flow, crop=2) + 3.0, 1e-9, 1e9)).std(ddof=0))
        self.assertNotEqual(value, promoted_first)


class DigitalDecompositionTest(unittest.TestCase):
    def folded_flow(self) -> torch.Tensor:
        flow = polynomial_flow(amplitude=0.6).clone()
        flow[0, 0, 4, 5, 5] = -3.0
        flow[0, 1, 5, 4, 5] = 2.5
        flow[0, 2, 5, 5, 4] = -2.0
        return flow

    def test_union_matches_frozen_digital_fold_percent(self):
        for flow in (polynomial_flow(), self.folded_flow(), random_flow(31, scale=0.9)):
            result = compute_metric(DIGITAL_DECOMPOSITION, flow)
            self.assertAlmostEqual(result.value, float(digital_fold_percent(flow.double()).item()), delta=1e-12)

    def test_float32_frozen_call_agrees_where_signs_are_separated(self):
        for flow in (polynomial_flow(), self.folded_flow()):
            result = compute_metric(DIGITAL_DECOMPOSITION, flow)
            self.assertAlmostEqual(result.value, float(digital_fold_percent(flow).item()), delta=1e-6)

    def test_component_minima_match_the_frozen_determinant_maps(self):
        flow = self.folded_flow()
        result = compute_metric(DIGITAL_DECOMPOSITION, flow)
        expected = [float(det.min().item()) for det in _digital_determinants(flow.double())]
        observed = [result.components[f"{name}_min"] for name in result.metadata["determinant_names"]]
        for got, want in zip(observed, expected, strict=True):
            self.assertAlmostEqual(got, want, delta=1e-9)

    def test_union_is_not_the_sum_of_component_fractions(self):
        result = compute_metric(DIGITAL_DECOMPOSITION, self.folded_flow())
        names = result.metadata["determinant_names"]
        total = sum(result.components[f"{name}_violation_fraction"] for name in names)
        union = result.components["union_violation_fraction"]
        self.assertGreater(total, union)
        self.assertGreater(union, 0.0)

    def test_corner_and_jstar_counts_stay_separate(self):
        result = compute_metric(DIGITAL_DECOMPOSITION, self.folded_flow())
        corner = result.components["corner_union_violation_fraction"]
        jstar = result.components["jstar_union_violation_fraction"]
        union = result.components["union_violation_fraction"]
        self.assertGreaterEqual(union, corner)
        self.assertGreaterEqual(union, jstar)
        self.assertNotAlmostEqual(corner + jstar, union)

    def test_nonfinite_determinants_are_violations(self):
        flow = torch.zeros(1, 3, *SHAPE)
        flow[0, 0, 4, 5, 5] = float("nan")
        result = compute_metric(DIGITAL_DECOMPOSITION, flow)
        nonfinite = sum(result.components[f"{name}_nonfinite_count"] for name in result.metadata["determinant_names"])
        self.assertGreater(nonfinite, 0.0)
        self.assertGreater(result.value, 0.0)


class DetjDiagnosticsTest(unittest.TestCase):
    def test_diagnostics_agree_with_direct_numpy(self):
        flow = random_flow(5, scale=0.4)
        result = compute_metric(DETJ_DIAGNOSTICS, flow)
        detj = oracle_central_detj(flow, crop=2)
        self.assertAlmostEqual(result.components["detj_min"], float(detj.min()), delta=1e-9)
        self.assertAlmostEqual(result.components["detj_max"], float(detj.max()), delta=1e-9)
        self.assertAlmostEqual(result.components["detj_quantile_0.5"], float(np.quantile(detj, 0.5)), delta=1e-9)
        self.assertAlmostEqual(result.components["nonpositive_fraction"], float((detj <= 0.0).mean()), delta=1e-12)

    def test_non_positive_count_precedes_the_shift_and_clamp(self):
        flow = random_flow(13, scale=1.2)
        result = compute_metric(DETJ_DIAGNOSTICS, flow)
        detj = oracle_central_detj(flow, crop=2)
        self.assertGreater(result.components["nonpositive_count"], 0.0)
        self.assertEqual(result.components["nonpositive_count"], float((detj <= 0.0).sum()))
        self.assertEqual(float((np.clip(detj + 3.0, 1e-9, 1e9) <= 0.0).sum()), 0.0)

    def test_tails_and_energy_are_named_explicitly(self):
        result = compute_metric(DETJ_DIAGNOSTICS, random_flow(17, scale=0.5))
        for key in (
            "compression_fraction_detj_below_0.5",
            "expansion_fraction_detj_above_2.0",
            "volume_distortion_energy_mean_squared_detj_minus_one",
            "volume_distortion_energy_mean_squared_log_detj_positive_only",
            "volume_distortion_energy_positive_fraction",
        ):
            self.assertIn(key, result.components)

    def test_nonfinite_and_nonpositive_counts_are_disjoint(self):
        flow = torch.zeros(1, 3, *SHAPE)
        flow[0, 0, 4, 5, 5] = float("nan")
        result = compute_metric(DETJ_DIAGNOSTICS, flow)
        self.assertGreater(result.components["nonfinite_count"], 0.0)
        self.assertEqual(result.components["nonpositive_count"], 0.0)
        self.assertEqual(
            result.components["invalid_count"],
            result.components["nonfinite_count"] + result.components["nonpositive_count"],
        )

    def test_all_nonfinite_field_is_described_without_crashing(self):
        result = compute_metric(DETJ_DIAGNOSTICS, torch.full((1, 3, *SHAPE), float("nan")))
        self.assertEqual(result.components["finite_count"], 0.0)
        self.assertTrue(math.isnan(result.components["detj_min"]))


class MutationSensitivityTest(unittest.TestCase):
    """Each convention named in a metric id must be load-bearing: rewrite it in a private copy of
    the module and the number has to move. Without this the oracles above could be passing for a
    reason unrelated to the convention they claim to pin.
    """

    def mutated(self, old: str, new: str):
        source = Path(search_gate_metrics.__file__).read_text(encoding="utf-8")
        self.assertEqual(source.count(old), 1, f"mutation anchor is not unique: {old!r}")
        name = "mutated_search_gate_metrics"
        module = ModuleType(name)
        sys.modules[name] = module
        self.addCleanup(sys.modules.pop, name, None)
        exec(compile(source.replace(old, new), "<mutated>", "exec"), module.__dict__)
        return module

    def assert_moves(self, module, metric_id: str, flow: torch.Tensor) -> None:
        self.assertNotAlmostEqual(module.compute_metric(metric_id, flow).value, compute_metric(metric_id, flow).value)

    def test_removing_the_shift_moves_the_learn2reg_value(self):
        self.assert_moves(self.mutated("SHIFT = 3.0", "SHIFT = 0.0"), LEARN2REG_SHIFTED_SDLOGJ, random_flow(41))

    def test_changing_the_crop_moves_the_learn2reg_value(self):
        anchor = "detj = learn2reg_jacobian_determinant(flow)"
        module = self.mutated(anchor, "detj = central_jacobian_determinant(flow, crop=1)")
        self.assert_moves(module, LEARN2REG_SHIFTED_SDLOGJ, random_flow(43))

    def test_changing_the_ddof_moves_the_legacy_value(self):
        module = self.mutated("torch.std(torch.log(shifted))", "torch.std(torch.log(shifted), correction=0)")
        self.assert_moves(module, LEGACY_SHIFTED_J, random_flow(45, shape=(5, 5, 5)))

    def test_permuting_the_coordinate_axes_moves_the_digital_value(self):
        module = self.mutated("transform = displacement + grid", "transform = displacement + grid[::-1]")
        self.assert_moves(module, DIGITAL_DECOMPOSITION, DigitalDecompositionTest().folded_flow())

    def test_axis_to_mode_pairing_alone_cannot_move_the_corner_union(self):
        """The eight corner determinants enumerate every sign combination, so permuting which axis
        takes which one-sided mode maps the family onto itself. The axis convention is therefore
        pinned by the coordinate grid and by test_channel_axis_pairing_is_zyx, not by this loop.
        """
        module = self.mutated(
            "_one_sided_difference(transform, axis, mode) for axis, mode in enumerate(modes)",
            "_one_sided_difference(transform, (axis + 1) % 3, mode) for axis, mode in enumerate(modes)",
        )
        flow = DigitalDecompositionTest().folded_flow()
        self.assertEqual(
            module.compute_metric(DIGITAL_DECOMPOSITION, flow).value,
            compute_metric(DIGITAL_DECOMPOSITION, flow).value,
        )

    def test_dropping_jstar_from_the_union_moves_the_digital_value(self):
        module = self.mutated("union = corner_union | jstar_union", "union = corner_union")
        self.assert_moves(module, DIGITAL_DECOMPOSITION, DigitalDecompositionTest().folded_flow())


class ContractTest(unittest.TestCase):
    def test_every_metric_declares_its_conventions(self):
        required = {"metric_id", "crop", "ddof", "mask", "clamp", "determinant", "dtype", "axis_order", "units"}
        for metric_id, spec in METRIC_SPECS.items():
            self.assertEqual(spec["metric_id"], metric_id)
            self.assertLessEqual(required, set(spec))

    def test_no_metric_is_called_sdlogj(self):
        for metric_id in METRIC_SPECS:
            self.assertNotEqual(metric_id.lower(), "sdlogj")
        for result in compute_bundle(polynomial_flow()).values():
            for key in result.components:
                self.assertNotEqual(key.lower(), "sdlogj")

    def test_unknown_or_implicit_metric_id_is_refused(self):
        flow = polynomial_flow()
        for metric_id in ("sdlogj", "", "CTCF_MATHEMATICAL_SDLOGJ_CENTRAL_CROP2_UNMASKED_DDOF0_FAILCLOSED_V2"):
            with self.assertRaises(ValueError):
                compute_metric(metric_id, flow)

    def test_result_carries_its_own_id_and_is_frozen(self):
        result = compute_metric(MATHEMATICAL_SDLOGJ_CROP2, polynomial_flow())
        self.assertEqual(result.metric_id, MATHEMATICAL_SDLOGJ_CROP2)
        with self.assertRaises(FrozenInstanceError):
            result.metric_id = "other"

    def test_batch_greater_than_one_is_refused(self):
        flow = torch.zeros(2, 3, *SHAPE)
        for metric_id in METRIC_SPECS:
            with self.assertRaises(ValueError):
                compute_metric(metric_id, flow, torch.ones(2, 1, *SHAPE))

    def test_small_spatial_size_is_refused(self):
        for size in (2, 3, 4):
            with self.assertRaises(ValueError):
                compute_metric(MATHEMATICAL_SDLOGJ_CROP2, torch.zeros(1, 3, size, size, size))

    def test_wrong_channel_count_is_refused(self):
        with self.assertRaises(ValueError):
            compute_metric(MATHEMATICAL_SDLOGJ_CROP2, torch.zeros(1, 2, *SHAPE))

    def test_mask_negative_controls(self):
        flow = random_flow(23)
        with self.assertRaises(ValueError):
            compute_metric(LEARN2REG_SHIFTED_SDLOGJ_MASKED, flow, torch.zeros(1, 1, *SHAPE))
        with self.assertRaises(ValueError):
            compute_metric(LEARN2REG_SHIFTED_SDLOGJ_MASKED, flow)
        with self.assertRaises(ValueError):
            compute_metric(LEARN2REG_SHIFTED_SDLOGJ_MASKED, flow, torch.ones(1, 1, 5, 5, 5))
        nonbinary = torch.ones(1, 1, *SHAPE)
        nonbinary[:, :, 4, 4, 4] = 0.5
        with self.assertRaises(ValueError):
            compute_metric(LEARN2REG_SHIFTED_SDLOGJ_MASKED, flow, nonbinary)
        full = torch.ones(1, 1, *SHAPE)
        self.assertAlmostEqual(
            compute_metric(LEARN2REG_SHIFTED_SDLOGJ_MASKED, flow, full).value,
            compute_metric(LEARN2REG_SHIFTED_SDLOGJ, flow).value,
            delta=1e-12,
        )

    def test_unmasked_metrics_refuse_an_unexpected_mask(self):
        flow = random_flow(29)
        mask = torch.zeros(1, 1, *SHAPE)
        mask[:, :, 3:-3, 3:-3, 3:-3] = 1
        with self.assertRaises(ValueError):
            compute_metric(LEARN2REG_SHIFTED_SDLOGJ, flow, mask)

    def test_bundle_covers_every_metric(self):
        bundle = compute_bundle(polynomial_flow(), torch.ones(1, 1, *SHAPE))
        self.assertEqual(set(bundle), set(METRIC_SPECS))
        for metric_id, result in bundle.items():
            self.assertEqual(result.metric_id, metric_id)


if __name__ == "__main__":
    unittest.main()
