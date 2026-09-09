from __future__ import annotations

import copy
import unittest

from tools.analysis.stage5 import mechanism_contract as contract, mechanism_report as report


def _stats():
    return {"elements": 2, "nonfinite": 0, "dtype": "torch.float32", "shape": [2]}


def _comparison():
    return {
        "status": "FINITE",
        "within_heuristic_tolerance": True,
        "tolerances": {"relative": 2e-4, "absolute": 1e-7, "cosine_minimum": 0.999},
        "reference_l2": 1.0,
        "actual_l2": 1.0,
        "absolute_l2": 0.0,
        "max_abs": 0.0,
    }


def _directions():
    return {name: _comparison() for name in ("forward", "reverse")}


def _field_observation(autocast=False, tf32=False, repeat=False):
    return {
        "status": "FINITE",
        "leaf_dtype": "torch.float32",
        "outer_autocast": autocast,
        "backend_flags": {"cudnn_allow_tf32": tf32, "matmul_allow_tf32": False, "float32_matmul_precision": "highest"},
        "gradient_stats": {name: _stats() for name in ("forward", "reverse")},
        "same_point_repeat_difference" if repeat else "same_point_arithmetic_effect": _directions(),
    }


def complete_field_audit():
    return {
        "status": "COMPLETE",
        "state_preserved": {
            name: True
            for name in (
                "model_during_audit",
                "training_flags_during_audit",
                "input_versions",
                "parameter_gradients",
                "backend_flags",
                "rng_restored",
                "training_flags_restored",
            )
        },
        "controller_outputs": {
            name: {
                "status": "FINITE",
                "autocast": autocast,
                "backend_flags": {
                    "cudnn_allow_tf32": tf32,
                    "matmul_allow_tf32": False,
                    "float32_matmul_precision": "highest",
                },
                "field_stats": {d: _stats() for d in ("forward", "reverse")},
            }
            for name, autocast, tf32 in contract.FIELD_MODES
        },
        "field_points": {
            name: {
                "evaluations": {
                    mode: _field_observation(autocast, tf32) for mode, autocast, tf32 in contract.FIELD_MODES
                },
                "repeat": _field_observation(repeat=True),
                "evaluation_point_effect": _directions(),
                "field_shift_vs_strict_controller": _directions(),
            }
            for name, _, _ in contract.FIELD_MODES
        },
    }


def complete_bias_audit(flags=None):
    flags = {
        "cudnn_allow_tf32": False,
        "matmul_allow_tf32": False,
        "float32_matmul_precision": "highest",
        "cudnn_benchmark": False,
        "cudnn_deterministic": True,
        "deterministic_algorithms": False,
        **(flags or {}),
    }
    return {
        "status": "COMPLETE",
        "original_backend_flags": copy.deepcopy(flags),
        "state_restored": {name: True for name in ("rng", "backend", "prepared_input_versions", "model")},
        "comparisons": {
            f"{actual}_vs_{reference}": {
                "parameters_all_elements": {"stem.0.bias": _comparison()},
                "layer_samples": {"stem.0#1": _comparison()},
                "missing_layer_samples": [],
            }
            for actual, reference in contract.BIAS_COMPARISONS
            if {actual, reference}.issubset({mode[0] for mode in contract.bias_modes(flags)})
        },
        "probes": {
            name: {
                "status": "CAPTURED",
                "mode": "fp32" if fp32 else "fp16",
                "scale": scale,
                "backend_flags": {
                    **flags,
                    **(
                        {"cudnn_allow_tf32": False, "matmul_allow_tf32": False, "float32_matmul_precision": "highest"}
                        if disable_tf32
                        else {}
                    ),
                },
                "parameters": {"stem.0.bias": _stats()},
                "missing_parameters": [],
                "layer_gradients": [_stats()],
                "bias_directions": [
                    {
                        "direction": d,
                        "status": "CAPTURED",
                        "node": "ConvolutionBackward0",
                        "upstream": _stats(),
                        "convolution_returned_bias": _stats(),
                        "sum_fp32": [1.0, 2.0],
                        "sum_fp64": [1.0, 2.0],
                        "convolution_returned_bias_values": [1.0, 2.0],
                        "returned_bias_vs_fp64_sum": _comparison(),
                    }
                    for d in ("ab", "ba")
                ],
            }
            for name, fp32, scale, disable_tf32 in contract.bias_modes(flags)
        },
    }


def complete_audits(flags=None):
    return {"field_audit": complete_field_audit(), "bias_audit": complete_bias_audit(flags)}


class MechanismStructureTests(unittest.TestCase):
    def test_complete_matrices_are_accepted(self):
        self.assertEqual(report.field_audit_errors(complete_field_audit()), [])
        audit = complete_bias_audit()
        self.assertEqual(report.bias_audit_errors(audit, audit["original_backend_flags"]), [])

    def test_worker_baseline_and_requested_probe_flags_are_cross_checked(self):
        audit = complete_bias_audit()
        worker_flags = dict(audit["original_backend_flags"], cudnn_allow_tf32=True)
        self.assertTrue(any("baseline differs" in error for error in report.bias_audit_errors(audit, worker_flags)))
        self.assertTrue(any("flags missing" in error for error in report.bias_audit_errors(audit, {})))
        audit["probes"]["fp32_strict"]["backend_flags"]["matmul_allow_tf32"] = True
        self.assertTrue(
            any(
                "probe arithmetic flags" in error
                for error in report.bias_audit_errors(audit, audit["original_backend_flags"])
            )
        )

    def test_complete_label_alone_is_insufficient(self):
        self.assertTrue(report.field_audit_errors({"status": "COMPLETE"}))
        self.assertTrue(report.bias_audit_errors({"status": "COMPLETE"}, {}))

    def test_missing_preservation_key_or_false_value_rejected(self):
        for missing in (True, False):
            audit = complete_field_audit()
            if missing:
                del audit["state_preserved"]["rng_restored"]
            else:
                audit["state_preserved"]["rng_restored"] = False
            self.assertTrue(report.field_audit_errors(audit))
        audit = complete_bias_audit()
        audit["state_restored"] = {}
        self.assertTrue(report.bias_audit_errors(audit, audit["original_backend_flags"]))

    def test_missing_field_mode_repeat_comparison_and_flags_rejected(self):
        mutations = (
            lambda a: a["controller_outputs"].pop("fp16_tf32_on"),
            lambda a: a["field_points"]["fp32_tf32_off"].pop("repeat"),
            lambda a: a["field_points"]["fp32_tf32_off"]["evaluations"].pop("fp16_tf32_off"),
            lambda a: a["field_points"]["fp32_tf32_off"]["evaluations"]["fp16_tf32_off"].pop(
                "same_point_arithmetic_effect"
            ),
            lambda a: a["field_points"]["fp32_tf32_off"]["evaluations"]["fp16_tf32_off"].pop("backend_flags"),
        )
        for mutate in mutations:
            audit = complete_field_audit()
            mutate(audit)
            self.assertTrue(report.field_audit_errors(audit))

    def test_oom_is_incomplete_but_math_error_remains_observation(self):
        audit = complete_field_audit()
        point = audit["field_points"]["fp16_tf32_on"]["evaluations"]
        point["fp16_tf32_on"] = {"status": "OOM", "error": "allocation failed"}
        self.assertTrue(report.field_audit_errors(audit))
        point["fp16_tf32_on"] = {"status": "MATH_ERROR", "error": "NCC guard"}
        self.assertEqual(report.field_audit_errors(audit), [])

    def test_finite_disagreement_does_not_make_observations_incomplete(self):
        audit = complete_field_audit()
        comparison = audit["field_points"]["fp16_tf32_on"]["evaluation_point_effect"]["forward"]
        comparison.update(within_heuristic_tolerance=False, absolute_l2=10.0)
        self.assertEqual(report.field_audit_errors(audit), [])

    def test_tf32_bias_mode_is_required_from_actual_audit_baseline(self):
        audit = complete_bias_audit({"cudnn_allow_tf32": True, "matmul_allow_tf32": False})
        self.assertEqual(report.bias_audit_errors(audit, audit["original_backend_flags"]), [])
        del audit["probes"][contract.BIAS_TF32_OFF_MODE[0]]
        self.assertTrue(report.bias_audit_errors(audit, audit["original_backend_flags"]))

    def test_bias_node_reductions_and_both_directions_required(self):
        for key in ("node", "sum_fp64", "returned_bias_vs_fp64_sum"):
            audit = complete_bias_audit()
            del audit["probes"][contract.BIAS_MODES[0][0]]["bias_directions"][0][key]
            self.assertTrue(report.bias_audit_errors(audit, audit["original_backend_flags"]))
        audit = complete_bias_audit()
        audit["probes"][contract.BIAS_MODES[0][0]]["bias_directions"].pop()
        self.assertTrue(report.bias_audit_errors(audit, audit["original_backend_flags"]))

    def test_bias_cross_scale_comparisons_required(self):
        audit = complete_bias_audit()
        audit.pop("comparisons")
        self.assertTrue(report.bias_audit_errors(audit, audit["original_backend_flags"]))

    def test_nonfinite_bias_return_is_preserved_as_observation(self):
        audit = complete_bias_audit()
        direction = audit["probes"][contract.BIAS_MODES[0][0]]["bias_directions"][0]
        direction["convolution_returned_bias"]["nonfinite"] = 1
        direction["convolution_returned_bias_values"][0] = None
        direction["returned_bias_vs_fp64_sum"] = {
            "status": "NONFINITE",
            "within_heuristic_tolerance": False,
            "tolerances": {"absolute": 1e-30},
        }
        self.assertEqual(report.bias_audit_errors(audit, audit["original_backend_flags"]), [])


if __name__ == "__main__":
    unittest.main()
