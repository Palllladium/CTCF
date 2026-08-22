from __future__ import annotations

import csv
import io
import json
import math
import os
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from tools.analysis.analyze_search_gate_c2 import (
    ANALYSIS_KIND,
    ANALYSIS_SCHEMA,
    REQUIRED_STEP_COLUMNS,
    SIGN_RULES,
    analyze,
    main,
)
from tools.analysis.run_artifacts import sha256_file

REPO_ROOT = Path(__file__).resolve().parents[3]
C2_PRODUCT = REPO_ROOT / "results/search_gate_c2/C2_DEVELOPMENT_20260821T183655Z_7e8fc5b9b17a"
REQUIRE_HISTORICAL = os.environ.get("CTCF_REQUIRE_HISTORICAL_GOLDENS") == "1"

needs_product = unittest.skipIf(
    not C2_PRODUCT.is_dir() and not REQUIRE_HISTORICAL,
    "historical C2 product absent; set CTCF_REQUIRE_HISTORICAL_GOLDENS=1 to fail instead of skip",
)


def branch_step(analysis: dict, trajectory_id: str, step: int) -> dict:
    return next(
        row for row in analysis["branch_steps"] if row["trajectory_id"] == trajectory_id and row["step"] == step
    )


def sign_rule(analysis: dict, trajectory_id: str, step: int, rule: str) -> dict:
    return next(
        row
        for row in analysis["sign_rules"]
        if row["trajectory_id"] == trajectory_id and row["step"] == step and row["rule"] == rule
    )


def synthetic_product(directory: Path) -> Path:
    """Two cases, one branch, two steps: enough to exercise every aggregation without the GPU run."""
    directory.mkdir(parents=True, exist_ok=True)
    rows = []
    for case_index, case_id in enumerate(("IXI_A", "IXI_B")):
        baseline = 0.70 + 0.01 * case_index
        entering = baseline
        for step in (1, 2):
            accepted = not (case_index == 1 and step == 2)
            candidate = entering + 0.002
            returned = candidate if accepted else entering
            rows.append(
                {
                    "case_id": case_id,
                    "trajectory_id": "mind_s1_sm1",
                    "step": step,
                    "action": "ACCEPT" if accepted else "ROLLBACK",
                    "reason": "EXACT_MIND_AND_GEOMETRY_POLICY_PASS",
                    "baseline_dice": baseline,
                    "candidate_dice": candidate,
                    "candidate_dice_delta": candidate - baseline,
                    "candidate_exact_status": "CERTIFIED",
                    "candidate_ncc7": -0.40 - 0.001 * step,
                    "candidate_ncc9": -0.30 - 0.001 * case_index,
                    "candidate_sdlogj": 0.06 + 0.001 * step,
                    "current_ncc7": -0.40,
                    "current_ncc9": -0.30,
                    "current_sdlogj": 0.06,
                    "labels_used_for_decision": "False",
                    "mind_improved": "True",
                    "proposal_built": "True",
                    "proposal_confidence_mean": 0.05 + 0.001 * step,
                    "proposal_entropy_mean": 3.10 + 0.01 * step,
                    "returned_dice": returned,
                    "returned_dice_delta": returned - baseline,
                    "returned_j_leq0_digital10_percent": 0.01 * step,
                    "returned_sdlogj": 0.06 + 0.0005 * step,
                    "scale": 1.0,
                    "sdlogj_cap_limit": "",
                    "sdlogj_cap_passed": "",
                    "sdlogj_cap_relative": "",
                    "smoothing_passes": 1,
                    "work_eps": 0.0011 - 0.000025 * (step - 1),
                }
            )
            entering = returned
    with (directory / "per_step.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=sorted(REQUIRED_STEP_COLUMNS))
        writer.writeheader()
        writer.writerows(rows)
    (directory / "summary.json").write_text(
        json.dumps(
            {
                "protocol_id": "CTCF-SEARCH-GATE-C2-V1",
                "schema": "ctcf-search-c2-summary-v1",
                "n_cases": 2,
                "n_step_rows": len(rows),
                "n_trajectories": 1,
                "scientific_status": "C2_NOT_PROMISING",
                "selected_trajectory_id": None,
                "test_115_authorized": False,
                "test_split_accessed": False,
                "labels_used_for_transaction_decision": False,
            }
        ),
        encoding="utf-8",
    )
    return directory


def rewrite_summary(directory: Path, **changes) -> None:
    path = directory / "summary.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload.update(changes)
    path.write_text(json.dumps(payload), encoding="utf-8")


def rewrite_rows(directory: Path, mutate) -> None:
    path = directory / "per_step.csv"
    with path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
        fields = list(rows[0])
    mutate(rows)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


class SyntheticProductTest(unittest.TestCase):
    def setUp(self):
        holder = tempfile.TemporaryDirectory()
        self.addCleanup(holder.cleanup)
        self.directory = Path(holder.name)
        self.product = synthetic_product(self.directory / "c2")

    def test_cardinalities_and_actions(self):
        analysis = analyze(self.product)
        first = branch_step(analysis, "mind_s1_sm1", 1)
        second = branch_step(analysis, "mind_s1_sm1", 2)
        self.assertEqual(first["cases"], 2)
        self.assertEqual(first["actions"], {"ACCEPT": 2})
        self.assertEqual(second["actions"], {"ACCEPT": 1, "ROLLBACK": 1})
        self.assertEqual(second["proposals_built"], 2)

    def test_absolute_dice_delta_and_incremental_gain(self):
        analysis = analyze(self.product)
        first = branch_step(analysis, "mind_s1_sm1", 1)
        second = branch_step(analysis, "mind_s1_sm1", 2)
        self.assertAlmostEqual(first["returned_dice_mean"], 0.7070, places=12)
        self.assertAlmostEqual(first["incremental_gain_vs_previous_step"], 0.002, places=12)
        self.assertAlmostEqual(second["returned_dice_mean"], 0.7080, places=12)
        self.assertAlmostEqual(second["incremental_gain_vs_previous_step"], 0.001, places=12)
        self.assertAlmostEqual(second["returned_dice_delta_mean"], 0.003, places=12)

    def test_work_margin_confound_is_stated(self):
        analysis = analyze(self.product)
        confound = analysis["work_margin_confound"]
        self.assertTrue(confound["work_eps_changes_with_step"])
        self.assertTrue(confound["monotone_decreasing"])
        self.assertIn("not a clean causal ablation", confound["statement"])

    def test_sign_rules_cover_the_frozen_list(self):
        analysis = analyze(self.product)
        rules = {row["rule"] for row in analysis["sign_rules"]}
        self.assertEqual(rules, set(SIGN_RULES))
        accept_all = sign_rule(analysis, "mind_s1_sm1", 1, "accept_all")
        self.assertEqual(accept_all["accepted"], 2)
        self.assertEqual(accept_all["dice_improved"], 2)
        self.assertAlmostEqual(accept_all["mean_returned_dice_delta"], 0.002, places=12)
        ncc7 = sign_rule(analysis, "mind_s1_sm1", 1, "ncc7_improves")
        self.assertEqual(ncc7["accepted"], 2)
        ncc9 = sign_rule(analysis, "mind_s1_sm1", 1, "ncc9_improves")
        self.assertEqual(ncc9["accepted"], 1)
        self.assertEqual(sign_rule(analysis, "mind_s1_sm1", 1, "ncc7_and_ncc9_improve")["accepted"], 1)
        self.assertEqual(sign_rule(analysis, "mind_s1_sm1", 1, "ncc7_or_ncc9_improve")["accepted"], 2)
        self.assertEqual(accept_all["estimand"], "ONE_STEP_COUNTERFACTUAL_FROM_THE_OBSERVED_ENTERING_STATE")
        self.assertFalse(accept_all["selection_authorized"])

    def test_diagnostic_marker_and_closed_test_split(self):
        analysis = analyze(self.product)
        self.assertEqual(analysis["analysis_kind"], ANALYSIS_KIND)
        self.assertEqual(analysis["schema"], ANALYSIS_SCHEMA)
        self.assertFalse(analysis["test_115_authorized"])
        self.assertFalse(analysis["test_split_accessed"])

    def test_input_is_not_modified(self):
        before = {path.name: sha256_file(path) for path in sorted(self.product.iterdir())}
        output = self.directory / "out"
        main(["--c2_dir", str(self.product), "--output", str(output)])
        after = {path.name: sha256_file(path) for path in sorted(self.product.iterdir())}
        self.assertEqual(before, after)
        self.assertEqual(sorted(before), ["per_step.csv", "summary.json"])

    def test_output_writes_json_and_csv(self):
        output = self.directory / "out"
        main(["--c2_dir", str(self.product), "--output", str(output)])
        written = sorted(path.name for path in output.iterdir())
        self.assertEqual(written, ["analysis.json", "branch_steps.csv", "sign_rules.csv", "smoothing_contrast.csv"])
        payload = json.loads((output / "analysis.json").read_text(encoding="utf-8"))
        self.assertEqual(payload["analysis_kind"], ANALYSIS_KIND)
        with (output / "sign_rules.csv").open(encoding="utf-8", newline="") as stream:
            self.assertEqual(len(list(csv.DictReader(stream))), len(SIGN_RULES) * 2)

    def test_rerun_replaces_the_output_in_place(self):
        output = self.directory / "out"
        main(["--c2_dir", str(self.product), "--output", str(output)])
        first = sha256_file(output / "analysis.json")
        main(["--c2_dir", str(self.product), "--output", str(output)])
        self.assertEqual(sha256_file(output / "analysis.json"), first)
        self.assertFalse([path for path in output.iterdir() if path.name.startswith(".")])

    def test_output_inside_the_product_is_refused(self):
        with self.assertRaises(ValueError):
            main(["--c2_dir", str(self.product), "--output", str(self.product / "analysis")])
        with self.assertRaises(ValueError):
            main(["--c2_dir", str(self.product), "--output", str(self.product)])

    def test_without_output_the_json_goes_to_stdout(self):
        stream = io.StringIO()
        with redirect_stdout(stream):
            main(["--c2_dir", str(self.product)])
        self.assertEqual(json.loads(stream.getvalue())["analysis_kind"], ANALYSIS_KIND)

    def test_missing_column_is_refused(self):
        broken = self.directory / "broken"
        synthetic_product(broken)
        path = broken / "per_step.csv"
        with path.open(encoding="utf-8", newline="") as stream:
            rows = list(csv.DictReader(stream))
        fields = [name for name in sorted(REQUIRED_STEP_COLUMNS) if name != "candidate_ncc7"]
        with path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
        with self.assertRaises(ValueError):
            analyze(broken)

    def test_source_authorization_flags_must_be_real_false_booleans(self):
        for key, value in (
            ("test_115_authorized", True),
            ("test_split_accessed", True),
            ("labels_used_for_transaction_decision", True),
            ("labels_used_for_transaction_decision", "False"),
        ):
            broken = synthetic_product(self.directory / f"broken_{key}_{value}")
            rewrite_summary(broken, **{key: value})
            with self.assertRaises(ValueError):
                analyze(broken)

    def test_row_level_label_use_is_refused(self):
        broken = synthetic_product(self.directory / "labels")
        rewrite_rows(broken, lambda rows: rows[0].__setitem__("labels_used_for_decision", "True"))
        with self.assertRaises(ValueError):
            analyze(broken)

    def test_duplicate_or_incomplete_row_grid_is_refused(self):
        duplicate = synthetic_product(self.directory / "duplicate")
        rewrite_rows(duplicate, lambda rows: rows.append(dict(rows[0])))
        rewrite_summary(duplicate, n_step_rows=5)
        with self.assertRaises(ValueError):
            analyze(duplicate)

        incomplete = synthetic_product(self.directory / "incomplete")
        rewrite_rows(incomplete, lambda rows: rows.pop())
        rewrite_summary(incomplete, n_step_rows=3)
        with self.assertRaises(ValueError):
            analyze(incomplete)

    def test_summary_counts_and_baseline_constancy_are_verified(self):
        broken_count = synthetic_product(self.directory / "count")
        rewrite_summary(broken_count, n_cases=3)
        with self.assertRaises(ValueError):
            analyze(broken_count)

        broken_baseline = synthetic_product(self.directory / "baseline")
        rewrite_rows(broken_baseline, lambda rows: rows[1].__setitem__("baseline_dice", "0.123"))
        with self.assertRaises(ValueError):
            analyze(broken_baseline)

    def test_summary_schema_and_protocol_are_verified(self):
        for key, value in (("schema", "other"), ("protocol_id", "other")):
            broken = synthetic_product(self.directory / f"broken_{key}")
            rewrite_summary(broken, **{key: value})
            with self.assertRaises(ValueError):
                analyze(broken)

    def test_manifest_hashes_are_verified_when_manifest_is_present(self):
        summary = json.loads((self.product / "summary.json").read_text(encoding="utf-8"))
        manifest = {
            "schema": "ctcf-search-c2-run-manifest-v1",
            "protocol_id": "CTCF-SEARCH-GATE-C2-V1",
            "status": "COMPLETE",
            "summary": summary,
            "files": {
                "per_step_sha256": sha256_file(self.product / "per_step.csv"),
                "summary_sha256": sha256_file(self.product / "summary.json"),
            },
        }
        (self.product / "c2_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        self.assertEqual(analyze(self.product)["source"]["n_cases"], 2)
        manifest["files"]["per_step_sha256"] = "0" * 64
        (self.product / "c2_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        with self.assertRaises(ValueError):
            analyze(self.product)


@needs_product
class HistoricalProductTest(unittest.TestCase):
    """Every expectation here is a number published in EV-SEARCH-003, recomputed from the primary CSV."""

    @classmethod
    def setUpClass(cls):
        cls.analysis = analyze(C2_PRODUCT)

    def test_source_inventory(self):
        source = self.analysis["source"]
        self.assertEqual(source["protocol_id"], "CTCF-SEARCH-GATE-C2-V1")
        self.assertEqual(source["n_cases"], 58)
        self.assertEqual(source["n_step_rows"], 928)
        self.assertEqual(source["n_trajectories"], 4)
        self.assertEqual(
            {entry["name"] for entry in source["inputs"]},
            {"per_step.csv", "per_branch.csv", "trajectory_summary.csv", "summary.json", "c2_manifest.json"},
        )

    def test_pooled_baseline(self):
        self.assertAlmostEqual(self.analysis["baseline"]["dice_mean"], 0.7614069834506391, places=12)
        self.assertAlmostEqual(self.analysis["baseline"]["dice_median"], 0.7668482348839223, places=12)

    def test_actions_per_branch(self):
        for trajectory_id in ("mind_s1_sm1", "mind_s2_sm1", "mind_s2_sm2"):
            for step in (1, 2, 3, 4):
                self.assertEqual(branch_step(self.analysis, trajectory_id, step)["actions"], {"ACCEPT": 58})
        self.assertEqual(branch_step(self.analysis, "mind_s2_sm2_sdlogj_cap", 1)["actions"], {"ROLLBACK": 58})
        for step in (2, 3, 4):
            self.assertEqual(branch_step(self.analysis, "mind_s2_sm2_sdlogj_cap", step)["actions"], {"STOPPED": 58})

    def test_absolute_dice_after_each_step(self):
        expected = [0.7622189119, 0.7627187771, 0.7629627193, 0.7629808897]
        for step, want in enumerate(expected, start=1):
            got = branch_step(self.analysis, "mind_s2_sm2", step)["returned_dice_mean"]
            self.assertAlmostEqual(got, want, places=10)

    def test_fourth_step_incremental_gain(self):
        fourth = branch_step(self.analysis, "mind_s2_sm2", 4)
        self.assertAlmostEqual(fourth["incremental_gain_vs_previous_step"], 0.0000181704, places=10)
        self.assertAlmostEqual(fourth["returned_dice_delta_mean"], 0.001573906271496121, places=12)

    def test_work_margin_moves_with_the_step_index(self):
        confound = self.analysis["work_margin_confound"]
        self.assertTrue(confound["work_eps_changes_with_step"])
        self.assertTrue(confound["monotone_decreasing"])
        self.assertEqual(confound["work_eps_by_step"]["mind_s2_sm2"], [0.0011, 0.001075, 0.00105, 0.001025])

    def test_smoothing_contrast(self):
        contrast = self.analysis["smoothing_contrast"]
        self.assertEqual(contrast["trajectories"], ["mind_s2_sm2", "mind_s2_sm1"])
        self.assertAlmostEqual(contrast["final"]["dice_difference_mean"], 0.0006475907, places=10)
        self.assertEqual(contrast["final"]["first_better_cases"], 56)
        self.assertEqual(contrast["final"]["cases"], 58)
        first_step = contrast["per_step"][0]
        self.assertAlmostEqual(first_step["dice_difference_mean"], -0.0000064537, places=10)
        self.assertEqual(first_step["first_better_sdlogj_cases"], 58)
        self.assertEqual(first_step["first_better_digital10_cases"], 58)

    def test_proposal_information(self):
        info = self.analysis["proposal_information"]
        self.assertEqual(info["proposals_built"], 754)
        self.assertAlmostEqual(info["entropy_mean_nats"], 3.1411410549, places=10)
        self.assertAlmostEqual(info["max_entropy_nats"], math.log(27), places=12)
        self.assertAlmostEqual(info["confidence_mean"], 0.0469288799, places=10)
        self.assertAlmostEqual(info["entropy_ratio"], 3.1411410549 / math.log(27), places=9)

    def test_sign_rules_on_the_first_s2_sm2_step(self):
        expected = {
            "accept_all": (58, 55, 3, 0.0008119285),
            "ncc7_improves": (45, 44, 1, 0.0006281127),
            "ncc9_improves": (25, 25, 0, 0.0003315977),
            "ncc7_and_ncc9_improve": (20, 20, 0, 0.0002417703),
            "ncc7_or_ncc9_improve": (50, 49, 1, 0.0007179401),
        }
        for rule, (accepted, improved, worsened, delta) in expected.items():
            row = sign_rule(self.analysis, "mind_s2_sm2", 1, rule)
            self.assertEqual(
                (row["accepted"], row["dice_improved"], row["dice_worsened"]), (accepted, improved, worsened)
            )
            self.assertAlmostEqual(row["mean_returned_dice_delta"], delta, places=10)

    def test_sdlogj_cap_rolled_back_every_first_candidate(self):
        cap = self.analysis["sdlogj_cap"]
        self.assertEqual(cap["trajectory_id"], "mind_s2_sm2_sdlogj_cap")
        self.assertEqual(cap["rolled_back_cases"], 58)
        self.assertEqual(cap["mind_and_certificate_passed_cases"], 58)
        self.assertAlmostEqual(cap["relative_growth_percent_min"], 0.8684, places=4)
        self.assertAlmostEqual(cap["relative_growth_percent_max"], 1.8604, places=4)

    def test_labels_were_not_used_for_any_decision(self):
        self.assertFalse(self.analysis["source"]["labels_used_for_transaction_decision"])
        self.assertFalse(self.analysis["test_115_authorized"])

    def test_analysis_does_not_touch_the_product(self):
        before = {path.name: sha256_file(path) for path in sorted(C2_PRODUCT.glob("*.csv"))}
        with tempfile.TemporaryDirectory() as directory:
            main(["--c2_dir", str(C2_PRODUCT), "--output", str(Path(directory) / "out")])
        after = {path.name: sha256_file(path) for path in sorted(C2_PRODUCT.glob("*.csv"))}
        self.assertEqual(before, after)


if __name__ == "__main__":
    unittest.main()
