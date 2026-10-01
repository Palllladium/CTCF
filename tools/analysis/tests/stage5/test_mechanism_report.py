from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path

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


HEAD = "b" * 40


class MechanismReportTests(unittest.TestCase):
    def aggregate(self, *, attempts=None, audit_status="COMPLETE", source_ok=True):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for job in contract.JOBS:
                payload = {
                    "schema": contract.SCHEMA,
                    "job": job,
                    "diagnostic_git_head": HEAD,
                    "workload_contract": contract.workload_contract(),
                    "precision_package_verified": True,
                    "mechanism_sources_unchanged": source_ok,
                    "production_checkpoint_written": False,
                    "labels_accessed": False,
                    "status": "DIAGNOSTIC_COMPLETE",
                    **complete_audits(),
                }
                payload["environment"] = {"backend_flags": payload["bias_audit"]["original_backend_flags"]}
                payload["bias_audit"]["status"] = audit_status
                if job == "F2P":
                    payload["f2p_replay"] = (
                        attempts
                        if attempts is not None
                        else [
                            {
                                "attempt": i + 1,
                                "status": "NOT_REPRODUCED",
                                "completed_updates": contract.F2P_UPDATES,
                                "expected_pairs": contract.F2P_UPDATES,
                                "initial_model_sha256": "a" * 64,
                                "pair_schedule_sha256": "b" * 64,
                                "pairs": [{}] * contract.F2P_UPDATES,
                            }
                            for i in range(contract.F2P_ATTEMPTS)
                        ]
                    )
                (root / f"{job}.json").write_text(json.dumps(payload))
            return report.aggregate_reports(root, HEAD, 0)

    def test_nonreproduction_is_explicit_without_erasing_other_observations(self):
        result = self.aggregate()
        self.assertEqual(result["status"], "OBSERVATIONS_COMPLETE_REVIEW_REQUIRED")
        self.assertEqual(result["jobs"]["F2P"]["reproduction"], "NOT_REPRODUCED")
        self.assertFalse(result["production_restart_authorized"])
        self.assertIn("bias_audit", result["jobs"]["F0"]["audits"])

    def test_missing_audit_or_replay_or_source_proof_is_incomplete(self):
        for kwargs in ({"audit_status": "ERROR"}, {"attempts": []}, {"source_ok": False}):
            self.assertEqual(self.aggregate(**kwargs)["status"], "INCOMPLETE")

    def test_observed_failure_requires_exact_capture(self):
        attempt = {
            "attempt": 1,
            "status": "FAILURE_CAPTURED",
            "capture_saved": False,
            "completed_updates": 0,
            "pairs": [],
            "expected_pairs": contract.F2P_UPDATES,
            "initial_model_sha256": "a" * 64,
            "pair_schedule_sha256": "b" * 64,
        }
        self.assertEqual(self.aggregate(attempts=[attempt])["status"], "INCOMPLETE")
        attempt["capture_saved"] = True
        self.assertEqual(self.aggregate(attempts=[attempt])["status"], "OBSERVATIONS_COMPLETE_REVIEW_REQUIRED")

    def test_f2p_attempts_require_same_initial_model_schedule_and_complete_workload(self):
        for changed in ("initial_model_sha256", "pair_schedule_sha256", "expected_pairs"):
            attempts = [
                {
                    "attempt": i + 1,
                    "status": "NOT_REPRODUCED",
                    "completed_updates": contract.F2P_UPDATES,
                    "pairs": [{}] * contract.F2P_UPDATES,
                    "expected_pairs": contract.F2P_UPDATES,
                    "initial_model_sha256": "a" * 64,
                    "pair_schedule_sha256": "b" * 64,
                }
                for i in range(contract.F2P_ATTEMPTS)
            ]
            attempts[1][changed] = 1 if changed == "expected_pairs" else "c" * 64
            result = self.aggregate(attempts=attempts)
            self.assertEqual(result["status"], "INCOMPLETE", result)
            self.assertTrue(any("F2P" in error for error in result["errors"]))


if __name__ == "__main__":
    unittest.main()
