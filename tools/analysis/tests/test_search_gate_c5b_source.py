from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path

import numpy as np

from tools.analysis import search_gate_c5b_source as source_auth
from tools.analysis.search_gate_c5b import SCHEMA_VERSION, validate_c5b_geometry_bundle

HISTORICAL_PRODUCT = Path("results/search_gate_c5/C5_DEVELOPMENT_20260825T175112Z_242dde3281d2")
REQUIRE_HISTORICAL = os.environ.get("CTCF_REQUIRE_HISTORICAL_GOLDENS") == "1"


def _recorded_c5_heavy_root(compact: Path) -> Path:
    source = json.loads((compact / "source_contract.json").read_text(encoding="utf-8"))
    return Path(source["roots"]["target_c5_heavy"])


def _append_byte(path: Path) -> None:
    path.write_bytes(path.read_bytes() + b"\n")


class HistoricalGoldenAvailabilityTest(unittest.TestCase):
    def test_required_mode_never_silently_skips_the_successful_c5_product(self) -> None:
        if REQUIRE_HISTORICAL:
            self.assertTrue(
                HISTORICAL_PRODUCT.is_dir(),
                f"CTCF_REQUIRE_HISTORICAL_GOLDENS=1 but C5 is absent: {HISTORICAL_PRODUCT}",
            )


@unittest.skipUnless(
    HISTORICAL_PRODUCT.is_dir(),
    "successful compact C5 product is absent; set CTCF_REQUIRE_HISTORICAL_GOLDENS=1 to fail",
)
class SuccessfulC5GoldenTest(unittest.TestCase):
    def authenticate(self, root: Path = HISTORICAL_PRODUCT) -> dict[str, object]:
        return source_auth.authenticate_c5_source(
            root,
            _recorded_c5_heavy_root(root),
            verify_heavy_bytes=False,
        )

    def test_exact_successful_product_builds_the_minimal_decision_projection(self) -> None:
        projection = self.authenticate()
        self.assertEqual(projection["schema"], f"ctcf-search-c5b-decision-source-{SCHEMA_VERSION}")
        self.assertEqual(len(projection["case_ids"]), 58)
        self.assertEqual(projection["case_ids"][0], "subject_344")
        self.assertEqual(
            set(projection["image_inputs"]),
            {"atlas", *projection["case_ids"]},
        )
        self.assertEqual(set(projection["source_initial"]), set(projection["case_ids"]))
        self.assertEqual(set(projection["source_rms"]), set(projection["case_ids"]))
        self.assertEqual(set(projection["source_anchors"]), set(projection["case_ids"]))
        self.assertEqual(
            set(projection["roots"]),
            {"source_c3_heavy", "source_c4_heavy", "source_c5_heavy"},
        )
        self.assertIs(projection["test_115_authorized"], False)
        self.assertIs(projection["test_split_accessed"], False)
        preflight = projection["anchor_geometry_preflight"]
        self.assertEqual(preflight["validated_anchor_count"], 3 * 58)
        self.assertEqual(preflight["central_invalid_count"], 0)
        self.assertEqual(preflight["corner_union_violation_count"], 0)
        self.assertGreater(preflight["digital_ten_nonzero_anchor_count"], 0)
        operator_preflight = projection["anchor_clip_operator_preflight"]
        self.assertEqual(operator_preflight["validated_anchor_count"], 3 * 58)
        self.assertEqual(operator_preflight["current_fast_below_work_eps_count"], 0)
        self.assertEqual(operator_preflight["output_fast_below_work_eps_count"], 3 * 58)
        self.assertEqual(operator_preflight["output_fast_below_exact_claim_eps_count"], 0)
        self.assertAlmostEqual(operator_preflight["output_fast_cert_bound_min"], 0.0010979427340438719)
        self.assertAlmostEqual(operator_preflight["output_fast_cert_bound_max"], 0.0010994449669079753)
        source_auth.assert_c5b_decision_projection_is_label_free(projection)
        serialized = json.dumps(projection, sort_keys=True).lower()
        self.assertNotIn(str(HISTORICAL_PRODUCT.resolve()).lower(), serialized)
        self.assertNotIn(".pkl", serialized)

    def test_projection_carries_exact_endpoint_byte_anchors_without_scores(self) -> None:
        projection = self.authenticate()
        anchors = projection["source_anchors"]["subject_105"]
        self.assertEqual(
            anchors["c5_s4_a10_b0_sweep1"]["field"]["array_sha256"],
            "129e332978c3e6262d962404ca22cd1d4dfbfad77a751583faa4a5bf218800db",
        )
        self.assertEqual(
            anchors["c5_s4_a20_b0_sweep1"]["field"]["array_sha256"],
            "7a1b169d95df776f7f37724e5f3dd6e1999b58129323a0b4c533f6d5d4839e00",
        )
        self.assertEqual(
            anchors["c4_reference_s2_a10_b0"]["field"]["root_id"],
            "source_c4_heavy",
        )
        self.assertEqual(
            anchors["c5_s4_a10_b0_sweep1"]["field"]["root_id"],
            "source_c5_heavy",
        )
        self.assertEqual(
            {name: value["clip_sweeps"] for name, value in anchors.items()},
            {name: 1 for name in anchors},
        )

    def test_subject_344_real_geometry_separates_exact_corners_from_jstar_diagnostic(self) -> None:
        marker = json.loads(
            (HISTORICAL_PRODUCT / "cases/subject_344/decision_complete.json").read_text(encoding="utf-8")
        )
        expected = {
            "int_s2_a10_b0": 414,
            "int_s4_a10_b0": 454,
            "int_s4_a20_b0": 1152,
        }
        for arm_id, union_count in expected.items():
            with self.subTest(arm_id=arm_id):
                arm = next(row for row in marker["arms"] if row["arm_id"] == arm_id)
                observed = validate_c5b_geometry_bundle(arm["geometry"], f"subject_344/{arm_id}")
                self.assertEqual(observed.corner_union_violation_count, 0)
                self.assertEqual(observed.digital_ten_union_violation_count, union_count)
                self.assertGreater(observed.jstar_union_violation_fraction, 0.0)

    def test_wrong_explicit_c5_heavy_root_is_rejected_even_without_byte_verification(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "Explicit C5 heavy root differs"):
            source_auth.authenticate_c5_source(
                HISTORICAL_PRODUCT,
                Path("definitely-not-the-frozen-c5-root"),
                verify_heavy_bytes=False,
            )

    def test_compact_byte_mutations_and_unlisted_files_fail_closed(self) -> None:
        mutations = (
            "c5_manifest.json",
            "run_manifest.json",
            "arm_summary.csv",
            "cases/subject_105/decision_complete.json",
            "cases/subject_105/evaluation_complete.json",
            "outputs.tsv",
        )
        for relative in mutations:
            with self.subTest(relative=relative), tempfile.TemporaryDirectory() as directory:
                copied = Path(directory) / "c5"
                shutil.copytree(HISTORICAL_PRODUCT, copied)
                _append_byte(copied / relative)
                with self.assertRaises(RuntimeError):
                    self.authenticate(copied)
        with tempfile.TemporaryDirectory() as directory:
            copied = Path(directory) / "c5"
            shutil.copytree(HISTORICAL_PRODUCT, copied)
            (copied / "unlisted.txt").write_text("unexpected", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "missing or unexpected files"):
                self.authenticate(copied)


class ProjectionAndRootUnitTest(unittest.TestCase):
    def test_projection_leak_checker_rejects_disguised_metadata_and_raw_containers(self) -> None:
        valid = {"source": {"field": {"relative_path": "cases/one.npz"}}}
        source_auth.assert_c5b_decision_projection_is_label_free(valid)
        for mutation in (
            {"evaluation_path": "/tmp/result"},
            {"per_label_values": [1.0]},
            {"score": {"dice": 0.8}},
            {"source": "/data/subject.pkl"},
            {"compact_directory": "/data/c5"},
        ):
            with self.subTest(mutation=mutation), self.assertRaises(RuntimeError):
                source_auth.assert_c5b_decision_projection_is_label_free(mutation)

    def test_root_validator_requires_distinct_existing_nonoverlapping_roots(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            roots = {}
            for name in ("c3", "c4", "c5"):
                path = base / name
                path.mkdir()
                roots[name] = path
            recorded = {
                "source_c3_heavy": str(roots["c3"]),
                "source_c4_heavy": str(roots["c4"]),
                "target_c5_heavy": str(roots["c5"]),
            }
            observed = source_auth._validate_roots(recorded, roots["c5"], require_exists=True)
            self.assertEqual(set(observed), {"source_c3_heavy", "source_c4_heavy", "source_c5_heavy"})

            missing = deepcopy(recorded)
            missing["source_c3_heavy"] = str(base / "absent")
            with self.assertRaisesRegex(RuntimeError, "roots are absent"):
                source_auth._validate_roots(missing, roots["c5"], require_exists=True)

            nested = deepcopy(recorded)
            nested["source_c4_heavy"] = str(roots["c3"] / "nested")
            (roots["c3"] / "nested").mkdir()
            with self.assertRaisesRegex(RuntimeError, "must not overlap"):
                source_auth._validate_roots(nested, roots["c5"], require_exists=True)

    def test_heavy_record_verifies_file_and_array_bytes_and_rejects_escape(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            array = np.arange(24, dtype=np.float32).reshape(1, 3, 2, 2, 2)
            path = root / "case.npy"
            np.save(path, array, allow_pickle=False)
            record = {
                "root_id": "source_c3_heavy",
                "relative_path": "case.npy",
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "array_sha256": hashlib.sha256(np.ascontiguousarray(array).tobytes(order="C")).hexdigest(),
                "bytes": path.stat().st_size,
            }
            roots = {"source_c3_heavy": root}
            source_auth._verify_heavy_record(record, roots, "synthetic image")
            changed = deepcopy(record)
            changed["array_sha256"] = "0" * 64
            with self.assertRaisesRegex(RuntimeError, "array bytes changed"):
                source_auth._verify_heavy_record(changed, roots, "synthetic image")
            escaped = deepcopy(record)
            escaped["relative_path"] = "../case.npy"
            with self.assertRaisesRegex(RuntimeError, "escapes"):
                source_auth._verify_heavy_record(escaped, roots, "synthetic image")

    def test_authenticator_does_not_import_c5_implementation_modules(self) -> None:
        text = Path(source_auth.__file__).read_text(encoding="utf-8")
        self.assertNotIn("from tools.analysis.search_gate_c5 import", text)
        self.assertNotIn("import tools.analysis.search_gate_c5\n", text)
        self.assertNotIn("search_gate_c5_contracts", text)
        self.assertNotIn("search_gate_c5_workers", text)


if __name__ == "__main__":
    unittest.main()
