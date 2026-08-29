from __future__ import annotations

import json
import os
import pickle
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from datasets.OASIS100 import (
    STAGE5_LABEL_EVALUATION_AUTHORIZATION,
    STAGE5_PREPARE_AUTHORIZATION,
    Stage5DataError,
    Stage5OasisDirectedCaseDataset,
    Stage5OasisEvaluationLabelStore,
    Stage5OasisImageDataset,
    Stage5OasisImageStore,
    build_stage5_split_manifests,
    load_stage5_runtime_contract,
    prepare_stage5_oasis_data,
    stage5_sha256_array,
)
from tools.analysis.run_artifacts import sha256_file
from tools.analysis.stage5.contracts import (
    build_decision_barrier,
    build_protocol_contract,
    build_training_barrier,
    canonical_json_bytes as transaction_json_bytes,
)
from tools.analysis.stage5.primitives import readable_json_bytes
from tools.analysis.tests.stage5.test_contracts import _all_checkpoints, _all_decisions, _digest


class Stage5DataTestCase(unittest.TestCase):
    image_shape = (2, 3, 4)

    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.source = self.root / "All"
        self.manifests = self.root / "manifests"
        self.cache = self.root / "image_only"
        self.source.mkdir()
        self.expected: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        for numeric_id in range(1, 7):
            image = np.arange(np.prod(self.image_shape), dtype=np.float32).reshape(self.image_shape)
            image = np.ascontiguousarray(image / 100.0 + numeric_id)
            label = np.full(self.image_shape, numeric_id, dtype=np.uint8)
            self.expected[numeric_id] = (image, label)
            with (self.source / f"p_{numeric_id:04d}.pkl").open("wb") as stream:
                pickle.dump((image, label), stream, protocol=4)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def prepare(self, manifests: Path | None = None, cache: Path | None = None):
        return prepare_stage5_oasis_data(
            self.source,
            manifests or self.manifests,
            cache or self.cache,
            privileged_prepare_authorization=STAGE5_PREPARE_AUTHORIZATION,
            expected_subjects=6,
            train_count=4,
            dev_count=2,
            image_shape=self.image_shape,
        )

    @staticmethod
    def read_json(path: Path) -> dict:
        return json.loads(path.read_bytes())

    @staticmethod
    def write_json(path: Path, document: dict) -> None:
        path.write_bytes(readable_json_bytes(document))

    def rewrite_inventory(self, mutate) -> None:
        inventory_path = self.manifests / "source_inventory.json"
        inventory = self.read_json(inventory_path)
        mutate(inventory)
        self.write_json(inventory_path, inventory)
        contract_path = self.manifests / "data_contract.json"
        contract = self.read_json(contract_path)
        contract["files"]["source_inventory"]["sha256"] = sha256_file(inventory_path)
        self.write_json(contract_path, contract)

    def label_barriers(self, *, complete: bool = True) -> tuple[Path, Path, Path]:
        runtime = load_stage5_runtime_contract(self.manifests / "data_contract.json")
        protocol = build_protocol_contract(
            git_head="a" * 40,
            data_contract_sha256=runtime.contract_sha256,
            u0_training_contract_sha256=_digest("u0-training"),
            controller_training_contract_sha256=_digest("controller-training"),
            search_contract_sha256=_digest("search"),
            directed_case_ids=tuple(item["case_id"] for item in runtime.pairs["cases"]),
            metric_ids=("OASIS_DICE_1_TO_35_V1",),
            u0_fixed_epoch=2,
            controller_fixed_epoch=2,
            bootstrap_policy="identity",
            bootstrap_parameters={},
        )
        training = build_training_barrier(protocol, _all_checkpoints(protocol))
        decisions = _all_decisions(protocol, training)
        decision = build_decision_barrier(protocol, training, decisions if complete else decisions[:-1])
        paths = (
            self.root / "protocol.json",
            self.root / "training_barrier.json",
            self.root / "decision_barrier.json",
        )
        for path, document in zip(paths, (protocol, training, decision), strict=True):
            path.write_bytes(transaction_json_bytes(document))
        return paths

    def open_labels(
        self,
        barriers: tuple[Path, Path, Path],
        *,
        authorization: str = STAGE5_LABEL_EVALUATION_AUTHORIZATION,
    ) -> Stage5OasisEvaluationLabelStore:
        protocol, training, decision = barriers
        return Stage5OasisEvaluationLabelStore(
            self.manifests / "data_contract.json",
            self.source,
            protocol_path=protocol,
            training_barrier_path=training,
            decision_barrier_path=decision,
            label_evaluation_authorization=authorization,
        )


class PreparationTest(Stage5DataTestCase):
    def test_prepare_requires_privileged_local_conversion(self) -> None:
        with self.assertRaisesRegex(Stage5DataError, "privileged authorization"):
            prepare_stage5_oasis_data(
                self.source,
                self.manifests,
                self.cache,
                privileged_prepare_authorization="",
                expected_subjects=6,
                train_count=4,
                dev_count=2,
                image_shape=self.image_shape,
            )

    def test_prepare_is_local_deterministic_and_image_only(self) -> None:
        first = self.prepare()
        frozen = {path.name: path.read_bytes() for path in self.manifests.glob("*.json")}
        second = self.prepare()
        self.assertEqual(first.contract_sha256, second.contract_sha256)
        self.assertEqual(first.contract_sha256, sha256_file(first.contract_path))
        self.assertEqual(frozen, {path.name: path.read_bytes() for path in self.manifests.glob("*.json")})
        self.assertEqual(sorted(path.suffix for path in self.cache.iterdir()), [".npy"] * 6)
        for numeric_id, (image, _) in self.expected.items():
            cached = np.load(self.cache / f"p_{numeric_id:04d}.npy", allow_pickle=False)
            self.assertEqual(cached.dtype, np.float32)
            np.testing.assert_array_equal(cached, image)

    def test_inventory_authenticates_local_pickle_image_label_and_cache(self) -> None:
        self.prepare()
        inventory = self.read_json(self.manifests / "source_inventory.json")
        self.assertEqual(inventory["source_scope"], "LOCAL_OASIS_L2R_ALL_PICKLES")
        self.assertEqual(inventory["subject_count"], 6)
        for record in inventory["subjects"]:
            numeric_id = record["numeric_id"]
            image, label = self.expected[numeric_id]
            self.assertEqual(record["subject_id"], f"OASIS_ALL_P{numeric_id:04d}")
            source = self.source / record["source"]["relative_path"]
            cache = self.cache / record["cache"]["relative_path"]
            self.assertEqual(record["source"]["sha256"], sha256_file(source))
            self.assertEqual(record["image_array_sha256"], stage5_sha256_array(image))
            self.assertEqual(record["label_array_sha256"], stage5_sha256_array(label))
            self.assertEqual(record["cache"]["sha256"], sha256_file(cache))

    def test_prepare_rejects_incomplete_malformed_and_duplicate_local_data(self) -> None:
        with (self.source / "p_0001.pkl").open("wb") as stream:
            pickle.dump((self.expected[1][0],), stream, protocol=4)
        with self.assertRaisesRegex(Stage5DataError, "exactly"):
            self.prepare()

        (self.source / "p_0001.pkl").unlink()
        with self.assertRaisesRegex(Stage5DataError, "expected 6"):
            self.prepare()

        image, label = self.expected[2]
        with (self.source / "p_0001.pkl").open("wb") as stream:
            pickle.dump((image, label), stream, protocol=4)
        with self.assertRaisesRegex(Stage5DataError, "duplicate"):
            self.prepare()

    def test_prepare_never_replaces_a_different_cache(self) -> None:
        self.cache.mkdir()
        target = self.cache / "p_0001.npy"
        target.write_bytes(b"corrupt")
        before = target.read_bytes()
        with self.assertRaisesRegex(Stage5DataError, "image cache"):
            self.prepare()
        self.assertEqual(target.read_bytes(), before)


class SplitAndRuntimeTest(Stage5DataTestCase):
    def test_default_split_has_294_train_100_dev_and_100_directions(self) -> None:
        subjects = [f"OASIS_ALL_P{index:04d}" for index in range(1, 395)]
        split, pairs = build_stage5_split_manifests(subjects)
        self.assertEqual(split["counts"], {"all": 394, "development": 100, "training": 294})
        self.assertEqual(pairs["counts"], {"directed_cases": 100, "unordered_pairs": 50})
        self.assertEqual((split, pairs), build_stage5_split_manifests(tuple(reversed(subjects))))

    def test_development_pairs_are_disjoint_and_bidirectional(self) -> None:
        self.prepare()
        runtime = load_stage5_runtime_contract(self.manifests / "data_contract.json")
        development = {row["subject_id"] for row in runtime.split["development"]}
        training = {row["subject_id"] for row in runtime.split["training"]}
        self.assertFalse(development & training)
        seen: list[str] = []
        for pair in runtime.pairs["pairs"]:
            first, second = (runtime.cases[case_id] for case_id in pair["case_ids"])
            self.assertEqual(first["moving_subject_id"], second["fixed_subject_id"])
            self.assertEqual(first["fixed_subject_id"], second["moving_subject_id"])
            seen.extend((pair["subject_a"], pair["subject_b"]))
        self.assertEqual(set(seen), development)
        self.assertEqual(len(seen), len(set(seen)))

    def test_image_store_and_datasets_never_open_pickle_sources(self) -> None:
        self.prepare()
        training = Stage5OasisImageDataset(self.manifests / "data_contract.json", self.cache, split="training")
        image, subject_id = training[0]
        self.assertEqual(len(training), 4)
        self.assertEqual(image.dtype, torch.float32)
        self.assertEqual(tuple(image.shape), (1, *self.image_shape))
        self.assertIn(subject_id, training.subject_ids)

        cases = Stage5OasisDirectedCaseDataset(self.manifests / "data_contract.json", self.cache)
        moving0, fixed0, _, moving_id0, fixed_id0 = cases[0]
        moving1, fixed1, _, moving_id1, fixed_id1 = cases[1]
        self.assertEqual((moving_id0, fixed_id0), (fixed_id1, moving_id1))
        np.testing.assert_array_equal(moving0.numpy(), fixed1.numpy())
        np.testing.assert_array_equal(fixed0.numpy(), moving1.numpy())

    def test_runtime_rejects_cache_and_manifest_drift(self) -> None:
        self.prepare()
        target = next(self.cache.glob("*.npy"))
        target.write_bytes(target.read_bytes() + b"drift")
        subject_id = f"OASIS_ALL_P{int(target.stem.split('_')[1]):04d}"
        with self.assertRaisesRegex(Stage5DataError, "drift"):
            Stage5OasisImageStore(self.manifests / "data_contract.json", self.cache).load_image(subject_id)

        self.manifests = self.root / "manifests_fresh"
        self.cache = self.root / "cache_fresh"
        self.prepare()
        self.rewrite_inventory(lambda inventory: inventory["subjects"][0]["cache"].update(relative_path="../x.npy"))
        with self.assertRaisesRegex(Stage5DataError, "unsafe"):
            Stage5OasisImageStore(self.manifests / "data_contract.json", self.cache)

    def test_runtime_rejects_same_size_cache_rewrite_with_restored_mtime(self) -> None:
        self.prepare()
        store = Stage5OasisImageStore(self.manifests / "data_contract.json", self.cache)
        subject_id = next(iter(store.runtime.subjects))
        record = store.runtime.subjects[subject_id]
        target = self.cache / record["cache"]["relative_path"]
        store.load_image(subject_id)
        info = target.stat()
        replacement = np.full(self.image_shape, 99.0, dtype=np.float32)
        with target.open("wb") as stream:
            np.save(stream, replacement, allow_pickle=False)
        os.utime(target, ns=(info.st_atime_ns, info.st_mtime_ns))
        self.assertEqual(target.stat().st_size, info.st_size)
        self.assertEqual(target.stat().st_mtime_ns, info.st_mtime_ns)
        with self.assertRaisesRegex(Stage5DataError, "content changed"):
            store.load_image(subject_id)

    def test_noncanonical_contract_is_rejected(self) -> None:
        self.prepare()
        path = self.manifests / "data_contract.json"
        path.write_bytes(path.read_bytes() + b" ")
        with self.assertRaisesRegex(Stage5DataError, "not canonical JSON"):
            load_stage5_runtime_contract(path)


class LabelBarrierTest(Stage5DataTestCase):
    def test_labels_open_only_for_frozen_development_after_complete_decisions(self) -> None:
        self.prepare()
        barriers = self.label_barriers()
        with self.assertRaisesRegex(Stage5DataError, "explicit evaluator authorization"):
            self.open_labels(barriers, authorization="")

        labels = self.open_labels(barriers)
        dev_id = next(iter(labels.development_ids))
        numeric_id = int(dev_id[-4:])
        np.testing.assert_array_equal(labels.load_label(dev_id), self.expected[numeric_id][1])
        train_id = next(row["subject_id"] for row in labels.runtime.split["training"])
        with self.assertRaisesRegex(Stage5DataError, "only open frozen development labels"):
            labels.load_label(train_id)

    def test_incomplete_decision_barrier_and_changed_source_are_rejected(self) -> None:
        self.prepare()
        with self.assertRaisesRegex(Stage5DataError, "invalid or incomplete"):
            self.open_labels(self.label_barriers(complete=False))

        labels = self.open_labels(self.label_barriers())
        dev_id = next(iter(labels.development_ids))
        source = self.source / labels.runtime.subjects[dev_id]["source"]["relative_path"]
        source.write_bytes(source.read_bytes() + b"drift")
        with self.assertRaisesRegex(Stage5DataError, "changed"):
            labels.load_label(dev_id)


if __name__ == "__main__":
    unittest.main()
