from __future__ import annotations

import copy
import hashlib
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from tools.analysis.run_artifacts import sha256_file
from tools.analysis.search.metrics import MATHEMATICAL_SDLOGJ_CROP2, MATHEMATICAL_SDLOGJ_FULL
from tools.analysis.search.pyramid import array_sha256
from tools.analysis.search.transaction import save_flow_npz_atomic
from tools.analysis.stage5.contracts import (
    BASE_SEEDS,
    CHECKPOINT_SCHEMA,
    CHECKPOINT_SELECTION_POLICY,
    CONTROLLER_VARIANT_IDS,
    DECISION_RECORD_SCHEMA,
    VARIANT_IDS,
    build_decision_barrier,
    build_protocol_contract,
    build_training_barrier,
    canonical_sha256,
)
from tools.analysis.stage5.evaluation import (
    DICE_MEAN_METRIC_ID,
    EFFECT_METRIC_IDS,
    EVALUATION_SCHEMA,
    INVERSE_COMPONENT_RMS_METRIC_ID,
    SIMULTANEOUS_FAMILY_PRIMARY,
    SIMULTANEOUS_FAMILY_REGIONAL,
    STAGE5_EVALUATION_METRIC_IDS,
    EvaluationContext,
    aggregate_pair_effects,
    build_evaluation_record,
    build_pair_evaluation,
    compute_geometry_bundle,
    evaluate_returned_decision,
    write_decision_metrics,
    write_evaluation_products,
)


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _dummy_file(value: str, *, root_id: str) -> dict[str, object]:
    return {
        "root_id": root_id,
        "relative_path": f"objects/{value}.json",
        "bytes": 128,
        "sha256": _digest(f"file:{value}"),
    }


def _field_record(root: Path, path: Path) -> dict[str, object]:
    return {
        "root_id": "decision_output_root",
        "relative_path": path.relative_to(root).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "array_sha256": array_sha256(torch.from_numpy(np.load(path, allow_pickle=False)["flow"])),
    }


def _save_reload(field: dict[str, object]) -> dict[str, object]:
    return {
        "file_sha256": field["sha256"],
        "in_memory_array_sha256": field["array_sha256"],
        "reloaded_array_sha256": field["array_sha256"],
        "reloaded_from_persisted_bytes": True,
    }


def _case_inventory() -> list[dict[str, str]]:
    return [
        {
            "case_id": "S5PAIR-001-D0",
            "pair_id": "S5PAIR-001",
            "moving_subject_id": "A",
            "fixed_subject_id": "B",
        },
        {
            "case_id": "S5PAIR-001-D1",
            "pair_id": "S5PAIR-001",
            "moving_subject_id": "B",
            "fixed_subject_id": "A",
        },
        {
            "case_id": "S5PAIR-002-D0",
            "pair_id": "S5PAIR-002",
            "moving_subject_id": "C",
            "fixed_subject_id": "D",
        },
        {
            "case_id": "S5PAIR-002-D1",
            "pair_id": "S5PAIR-002",
            "moving_subject_id": "D",
            "fixed_subject_id": "C",
        },
    ]


def _protocol() -> dict[str, object]:
    return build_protocol_contract(
        git_head="a" * 40,
        data_contract_sha256=_digest("data"),
        u0_training_contract_sha256=_digest("u0-training"),
        controller_training_contract_sha256=_digest("controller-training"),
        search_contract_sha256=_digest("search"),
        directed_case_ids=tuple(item["case_id"] for item in _case_inventory()),
        metric_ids=STAGE5_EVALUATION_METRIC_IDS,
        u0_fixed_epoch=500,
        controller_fixed_epoch=100,
        bootstrap_policy="identity",
        bootstrap_parameters={},
    )


def _checkpoint(protocol: dict[str, object], seed: int, variant: str, base_sha: str | None) -> dict[str, object]:
    controller = variant != "U0"
    return {
        "schema": CHECKPOINT_SCHEMA,
        "checkpoint_id": f"S5-S{seed}-{variant}",
        "role": "CONTROLLER" if controller else "U0",
        "variant_id": variant,
        "seed": seed,
        "fixed_epoch": protocol["controller_fixed_epoch"] if controller else protocol["u0_fixed_epoch"],
        "selection_policy": CHECKPOINT_SELECTION_POLICY,
        "git_head": protocol["git_head"],
        "protocol_sha256": canonical_sha256(protocol),
        "data_contract_sha256": protocol["data_contract_sha256"],
        "training_contract_sha256": protocol[
            "controller_training_contract_sha256" if controller else "u0_training_contract_sha256"
        ],
        "checkpoint_file": _dummy_file(
            f"checkpoint-S{seed}-{variant}",
            root_id="checkpoint_root",
        ),
        "state_dict_sha256": _digest(f"state-S{seed}-{variant}"),
        "metrics_sha256": _digest(f"metrics-S{seed}-{variant}"),
        "base_checkpoint_sha256": base_sha if controller else None,
        "initial_controller_state_sha256": _digest(f"initial-S{seed}") if controller else None,
        "source_contract_sha256": _digest(f"source-S{seed}") if controller else None,
        "controller_parameter_count": 100 if controller else 0,
    }


def _training_barrier(protocol: dict[str, object]) -> dict[str, object]:
    checkpoints: list[dict[str, object]] = []
    bases: dict[int, dict[str, object]] = {}
    for seed in BASE_SEEDS:
        base = _checkpoint(protocol, seed, "U0", None)
        bases[seed] = base
        checkpoints.append(base)
    for seed in BASE_SEEDS:
        for variant in CONTROLLER_VARIANT_IDS:
            checkpoints.append(_checkpoint(protocol, seed, variant, str(bases[seed]["checkpoint_file"]["sha256"])))
    return build_training_barrier(protocol, checkpoints)


def _decision(
    case_id: str,
    seed: int,
    variant: str,
    checkpoint: dict[str, object],
    field: dict[str, object],
) -> dict[str, object]:
    status = "BASELINE_CERTIFIED" if variant == "U0" else "ACCEPTED"
    decision_id = f"{case_id}__S{seed}__{variant}"
    source = copy.deepcopy(field)
    source["root_id"] = "source_field_root"
    returned = copy.deepcopy(field)
    returned["root_id"] = "source_field_root" if variant == "U0" else "decision_output_root"
    return {
        "schema": DECISION_RECORD_SCHEMA,
        "decision_id": decision_id,
        "case_id": case_id,
        "seed": seed,
        "variant_id": variant,
        "checkpoint_sha256": checkpoint["checkpoint_file"]["sha256"],
        "certified_source_field": source,
        "requested_field": copy.deepcopy(returned),
        "candidate_field": copy.deepcopy(returned),
        "returned_field": returned,
        "requested_save_reload": _save_reload(returned),
        "candidate_save_reload": _save_reload(returned),
        "returned_save_reload": _save_reload(returned),
        "exact_report": _dummy_file(
            f"exact-{decision_id}",
            root_id="decision_output_root",
        ),
        "candidate_exact_status": "CERTIFIED",
        "candidate_exact_certified": True,
        "returned_exact_status": "CERTIFIED",
        "returned_certified": True,
        "transaction_status": status,
        "rollback_source_sha256_equal": False,
        "runtime_seconds": 1.0 + 0.1 * seed,
        "peak_memory_bytes": 1024 + seed,
        "requested_delta_rms": 0.0,
        "candidate_delta_rms": 0.0,
        "returned_delta_rms": 0.0,
        "candidate_retained_ratio": None,
        "returned_retained_ratio": None,
        "labels_loaded": False,
        "execution_sha256": _digest(f"execution-{decision_id}"),
    }


def _context(root: Path, path: Path) -> EvaluationContext:
    protocol = _protocol()
    training = _training_barrier(protocol)
    checkpoints = {(item["seed"], item["variant_id"]): item for item in training["checkpoints"]}
    record = _field_record(root, path)
    decisions = [
        _decision(case["case_id"], seed, variant, checkpoints[(seed, variant)], record)
        for case in _case_inventory()
        for seed in BASE_SEEDS
        for variant in VARIANT_IDS
    ]
    barrier = build_decision_barrier(protocol, training, decisions)
    return EvaluationContext.from_barriers(protocol, training, barrier, _case_inventory())


def _labels(shape: tuple[int, int, int]) -> np.ndarray:
    label = np.zeros(shape, dtype=np.uint8)
    label[2:5, 2:5, 2:5] = 1
    label[3:5, 3:5, 3:5] = 2
    return label


class ReturnedFieldEvaluationTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.field_path = self.root / "returned.npz"
        save_flow_npz_atomic(self.field_path, torch.zeros(1, 3, 7, 7, 7))
        self.context = _context(self.root, self.field_path)

    def tearDown(self):
        self.temporary.cleanup()

    def _evaluate(self, case: str, seed: int = 0, variant: str = "U0") -> dict[str, object]:
        label = _labels((7, 7, 7))
        return evaluate_returned_decision(
            self.context,
            f"{case}__S{seed}__{variant}",
            self.field_path,
            label,
            label,
            requested_field_path=self.field_path,
            candidate_field_path=self.field_path,
        )

    def test_identity_psi_uses_nearest_voxel_center_warp_and_full_metrics(self):
        result = self._evaluate("S5PAIR-001-D0")
        self.assertEqual(len(STAGE5_EVALUATION_METRIC_IDS), 46)
        self.assertEqual(len(EFFECT_METRIC_IDS), 45)
        self.assertEqual(result["schema"], EVALUATION_SCHEMA)
        self.assertFalse(result["warp"]["align_corners"])
        self.assertEqual(result["warp"]["operator"], "tools.analysis.search.transaction.sample_at_psi")
        self.assertAlmostEqual(result["metrics"]["mean_dice"]["value"], 2.0 / 35.0, places=7)
        self.assertEqual(len(result["metrics"]["per_label_dice"]), 35)
        self.assertEqual(tuple(result["metrics"]["geometry"]), STAGE5_EVALUATION_METRIC_IDS[36:-2])
        self.assertEqual(result, self._evaluate("S5PAIR-001-D0"))

    def test_barrier_and_returned_bytes_are_fail_closed(self):
        altered = copy.deepcopy(self.context.decision_barrier)
        altered["status"] = "INCOMPLETE"
        with self.assertRaises(RuntimeError):
            EvaluationContext.from_barriers(
                self.context.protocol,
                self.context.training_barrier,
                altered,
                _case_inventory(),
            )
        self.field_path.write_bytes(self.field_path.read_bytes() + b"tamper")
        with self.assertRaisesRegex(RuntimeError, "bytes differ"):
            self._evaluate("S5PAIR-001-D0")

    def test_labels_outside_frozen_oasis_inventory_are_rejected(self):
        label = _labels((7, 7, 7))
        label[0, 0, 0] = 36
        with self.assertRaisesRegex(ValueError, "outside frozen OASIS"):
            evaluate_returned_decision(
                self.context,
                "S5PAIR-001-D0__S0__U0",
                self.field_path,
                label,
                _labels((7, 7, 7)),
                requested_field_path=self.field_path,
                candidate_field_path=self.field_path,
            )

    def test_fail_closed_geometry_error_is_structured_not_numeric(self):
        zz = torch.arange(7, dtype=torch.float32).view(1, 1, 7, 1, 1).expand(1, 1, 7, 7, 7)
        folded = torch.cat((-2.0 * zz, torch.zeros_like(zz), torch.zeros_like(zz)), dim=1)
        fixed = torch.from_numpy(_labels((7, 7, 7))).unsqueeze(0).unsqueeze(0)
        bundle = compute_geometry_bundle(folded, fixed)
        for metric_id in (MATHEMATICAL_SDLOGJ_FULL, MATHEMATICAL_SDLOGJ_CROP2):
            self.assertEqual(bundle[metric_id]["status"], "ERROR")
            self.assertIsNone(bundle[metric_id]["value"])
            self.assertEqual(bundle[metric_id]["error"]["error_type"], "MetricFailClosedError")

    def test_inverse_consistency_requires_two_exact_reverse_directions(self):
        forward = self._evaluate("S5PAIR-001-D0")
        reverse = self._evaluate("S5PAIR-001-D1")
        pair = build_pair_evaluation(
            self.context,
            forward,
            reverse,
            self.field_path,
            self.field_path,
        )
        self.assertEqual(pair["scalar_metrics"][INVERSE_COMPONENT_RMS_METRIC_ID]["value"], 0.0)
        with self.assertRaisesRegex(RuntimeError, "both exact directions"):
            build_pair_evaluation(
                self.context,
                forward,
                copy.deepcopy(forward),
                self.field_path,
                self.field_path,
            )

    def test_evaluation_record_binds_metrics_file_and_decision(self):
        result = self._evaluate("S5PAIR-001-D0")
        metrics_path = self.root / "metrics.json"
        digest = write_decision_metrics(metrics_path, result)
        metrics_file = {
            "root_id": "evaluation_output_root",
            "relative_path": metrics_path.name,
            "bytes": metrics_path.stat().st_size,
            "sha256": digest,
        }
        record = build_evaluation_record(result, metrics_file)
        self.assertEqual(record["decision_record_sha256"], result["decision_record_sha256"])
        self.assertEqual(record["returned_field_sha256"], result["returned_field"]["sha256"])


class PairedAggregationTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.field_path = self.root / "returned.npz"
        save_flow_npz_atomic(self.field_path, torch.zeros(1, 3, 7, 7, 7))
        self.context = _context(self.root, self.field_path)
        label = _labels((7, 7, 7))
        forward = evaluate_returned_decision(
            self.context,
            "S5PAIR-001-D0__S0__U0",
            self.field_path,
            label,
            label,
            requested_field_path=self.field_path,
            candidate_field_path=self.field_path,
        )
        reverse = evaluate_returned_decision(
            self.context,
            "S5PAIR-001-D1__S0__U0",
            self.field_path,
            label,
            label,
            requested_field_path=self.field_path,
            candidate_field_path=self.field_path,
        )
        self.template = build_pair_evaluation(
            self.context,
            forward,
            reverse,
            self.field_path,
            self.field_path,
        )
        self.forward_evaluation = forward

    def tearDown(self):
        self.temporary.cleanup()

    def _rows(self) -> list[dict[str, object]]:
        rows: list[dict[str, object]] = []
        for pair_index, pair_id in enumerate(sorted(self.context.pairs)):
            for seed in BASE_SEEDS:
                for variant_index, variant in enumerate(VARIANT_IDS):
                    row = copy.deepcopy(self.template)
                    row["pair_id"] = pair_id
                    row["seed"] = seed
                    row["variant_id"] = variant
                    row["case_ids"] = list(self.context.pairs[pair_id])
                    delta = 0.001 * variant_index + 0.0002 * variant_index * pair_index
                    for metric_index, metric_id in enumerate(EFFECT_METRIC_IDS):
                        row["scalar_metrics"][metric_id]["value"] = (
                            0.1 + 0.01 * metric_index + 0.0001 * pair_index + delta
                        )
                    row["pair_diagnostics"]["STAGE5_DECISION_PAIR_RUNTIME_SECONDS_SUM_V1"] = 2.0 + 0.1 * variant_index
                    row["pair_diagnostics"]["STAGE5_DECISION_PAIR_PEAK_MEMORY_BYTES_MAX_V1"] = 1024 + 10 * variant_index
                    if variant != "U0":
                        row["transaction_statuses"] = ["ACCEPTED", "ACCEPTED"]
                        row["transaction_counts"] = {"ACCEPTED": 2}
                        row["pair_diagnostics"]["STAGE5_DECISION_PAIR_REQUESTED_DELTA_RMS_MEAN_V1"] = 1.0
                        row["pair_diagnostics"]["STAGE5_DECISION_PAIR_CANDIDATE_DELTA_RMS_MEAN_V1"] = 0.8
                        row["pair_diagnostics"]["STAGE5_DECISION_PAIR_RETURNED_DELTA_RMS_MEAN_V1"] = 0.7
                        row["pair_diagnostics"]["STAGE5_DECISION_PAIR_CANDIDATE_RETAINED_RATIO_MEAN_V1"] = 0.8
                        row["pair_diagnostics"]["STAGE5_DECISION_PAIR_RETURNED_RETAINED_RATIO_MEAN_V1"] = 0.7
                    rows.append(row)
        return rows

    def test_bootstrap_is_pair_blocked_seed_aware_deterministic_and_has_no_threshold(self):
        rows = self._rows()
        first = aggregate_pair_effects(self.context, rows)
        second = aggregate_pair_effects(self.context, rows)
        self.assertEqual(first, second)
        self.assertTrue(first["no_success_threshold"])
        self.assertTrue(first["seed_handling"]["pair_seed_rows_are_not_treated_as_independent"])
        selected = next(
            item
            for item in first["planned_contrasts"]
            if item["contrast_id"] == "F0_MINUS_U0"
            and item["scope"] == "seed_mean"
            and item["metric_id"] == DICE_MEAN_METRIC_ID
        )
        self.assertEqual(selected["summary"]["n_unordered_pairs"], 2)
        self.assertAlmostEqual(selected["summary"]["mean"], 0.0011, places=12)
        self.assertGreater(selected["summary"]["ci_high"], selected["summary"]["ci_low"])
        simultaneous = selected["simultaneous_ci"]
        self.assertEqual(simultaneous["family_id"], SIMULTANEOUS_FAMILY_PRIMARY)
        self.assertLessEqual(simultaneous["ci_low"], selected["summary"]["ci_low"])
        self.assertGreaterEqual(simultaneous["ci_high"], selected["summary"]["ci_high"])
        primary_family = next(
            item
            for item in first["simultaneous_families"]
            if item["family_id"] == SIMULTANEOUS_FAMILY_PRIMARY and item["scope"] == "seed_mean"
        )
        regional_family = next(
            item
            for item in first["simultaneous_families"]
            if item["family_id"] == SIMULTANEOUS_FAMILY_REGIONAL and item["scope"] == "seed_mean"
        )
        self.assertEqual(primary_family["n_hypotheses"], 11)
        self.assertEqual(regional_family["n_hypotheses"], 11 * 35)
        shared_primary_seeds = {
            item["simultaneous_ci"]["shared_pair_resample_seed"]
            for item in first["planned_contrasts"]
            if item["scope"] == "seed_mean" and item["metric_id"] == DICE_MEAN_METRIC_ID
        }
        self.assertEqual(shared_primary_seeds, {primary_family["shared_pair_resample_seed"]})
        for item in first["planned_contrasts"]:
            if "simultaneous_ci" not in item:
                continue
            self.assertLessEqual(item["simultaneous_ci"]["ci_low"], item["summary"]["ci_low"])
            self.assertGreaterEqual(item["simultaneous_ci"]["ci_high"], item["summary"]["ci_high"])
        runtime_effect = next(
            item
            for item in first["paired_diagnostic_effects_vs_u0"]
            if item["variant_id"] == "F0"
            and item["scope"] == "seed_mean"
            and item["diagnostic_id"] == "STAGE5_DECISION_PAIR_RUNTIME_SECONDS_SUM_V1"
        )
        self.assertAlmostEqual(runtime_effect["summary"]["mean"], 0.1, places=12)
        u0_retention = next(
            item
            for item in first["diagnostic_summaries"]
            if item["variant_id"] == "U0"
            and item["scope"] == "seed_mean"
            and item["diagnostic_id"] == "STAGE5_DECISION_PAIR_RETURNED_RETAINED_RATIO_MEAN_V1"
        )
        self.assertEqual(u0_retention["status"], "UNDEFINED")
        self.assertEqual(first["transaction_counts"]["U0"], {"BASELINE_CERTIFIED": 12})
        self.assertEqual(first["transaction_counts"]["F0"], {"ACCEPTED": 12})

    def test_metric_error_is_not_dropped_from_paired_effect(self):
        rows = self._rows()
        target = next(
            item
            for item in rows
            if item["pair_id"] == "S5PAIR-001" and item["seed"] == 0 and item["variant_id"] == "F2V"
        )
        target["scalar_metrics"][DICE_MEAN_METRIC_ID] = {
            "metric_id": DICE_MEAN_METRIC_ID,
            "status": "ERROR",
            "value": None,
            "error": {"error_type": "MUTATION"},
        }
        aggregate = aggregate_pair_effects(self.context, rows)
        selected = next(
            item
            for item in aggregate["paired_effects_vs_u0"]
            if item["variant_id"] == "F2V" and item["scope"] == "seed_0" and item["metric_id"] == DICE_MEAN_METRIC_ID
        )
        self.assertEqual(selected["status"], "ERROR")
        self.assertIn("S5PAIR-001/S0", selected["error"]["affected_pair_seed"])
        failed_family = next(
            item
            for item in aggregate["simultaneous_families"]
            if item["family_id"] == SIMULTANEOUS_FAMILY_PRIMARY and item["scope"] == "seed_0"
        )
        self.assertEqual(failed_family["status"], "ERROR")

    def test_products_are_immutable(self):
        rows = self._rows()
        aggregate = aggregate_pair_effects(self.context, rows)
        evaluation = self.forward_evaluation
        output = self.root / "products"
        digests = write_evaluation_products(output, [evaluation], rows, aggregate)
        self.assertEqual(
            set(digests),
            {
                "evaluation_bundle.json",
                "per_decision.csv",
                "per_label.csv",
                "geometry_metrics.csv",
                "field_stage_diagnostics.csv",
                "per_pair_metric.csv",
                "paired_effects_vs_u0.csv",
                "planned_contrasts.csv",
                "decision_diagnostics.csv",
            },
        )
        write_evaluation_products(output, [evaluation], rows, aggregate)
        (output / "per_decision.csv").write_text("tampered\n", encoding="utf-8")
        with self.assertRaises(FileExistsError):
            write_evaluation_products(output, [evaluation], rows, aggregate)


if __name__ == "__main__":
    unittest.main()
