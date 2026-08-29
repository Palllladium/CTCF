"""The frozen local OASIS 294/100 data protocol used by Stage 5."""

from __future__ import annotations

import hashlib
import json
import os
import pickle
import re
import tempfile
from collections.abc import Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from tools.analysis.run_artifacts import sha256_file
from tools.analysis.stage5.contracts import (
    canonical_json_bytes,
    canonical_sha256,
    validate_decision_barrier,
    validate_protocol_contract,
    validate_training_barrier,
)
from tools.analysis.stage5.primitives import (
    FileGeneration,
    file_generation,
    generation_cache_is_safe,
    is_link_like,
    readable_json_bytes,
    relative_posix,
    require_plain_directory,
    resolve_inside_root,
    write_immutable_bytes,
)

# Stage 5 consumes the existing local Learn2Reg/OASIS ``All`` pickles. A
# privileged one-time preparation separates images from segmentations so that
# training workers cannot open label-bearing files.
STAGE5_DATA_PROTOCOL_ID = "CTCF-STAGE5-OASIS-DATA-V2"
STAGE5_SOURCE_INVENTORY_SCHEMA = "ctcf-stage5-oasis-source-inventory-v2"
STAGE5_SPLIT_MANIFEST_SCHEMA = "ctcf-stage5-oasis-split-v2"
STAGE5_PAIR_MANIFEST_SCHEMA = "ctcf-stage5-oasis-pairs-v2"
STAGE5_DATA_CONTRACT_SCHEMA = "ctcf-stage5-oasis-data-contract-v2"
STAGE5_PREPARE_AUTHORIZATION = "CTCF_STAGE5_PRIVILEGED_PREPARE_V2"
STAGE5_LABEL_EVALUATION_AUTHORIZATION = "CTCF_STAGE5_LABEL_EVALUATION_ONLY_V2"

STAGE5_SPLIT_DOMAIN = "CTCF-S5-OASIS-SPLIT-V2\0"
STAGE5_PAIR_DOMAIN = "CTCF-S5-OASIS-PAIR-V2\0"
STAGE5_DIRECTION_DOMAIN = "CTCF-S5-OASIS-DIRECTION-V2\0"

STAGE5_DEFAULT_IMAGE_SHAPE = (160, 192, 224)
STAGE5_DEFAULT_SUBJECT_COUNT = 394
STAGE5_DEFAULT_TRAIN_COUNT = 294
STAGE5_DEFAULT_DEV_COUNT = 100

_STAGE5_SOURCE_NAME = re.compile(r"p_(\d{4})\.pkl")
_STAGE5_SHA256 = re.compile(r"[0-9a-f]{64}")


class Stage5DataError(RuntimeError):
    pass


@dataclass(frozen=True)
class PreparedStage5Data:
    contract_path: Path
    contract_sha256: str
    source_inventory_path: Path
    split_manifest_path: Path
    pair_manifest_path: Path
    cache_root: Path


@dataclass(frozen=True)
class Stage5RuntimeContract:
    contract_path: Path
    contract_sha256: str
    contract: dict[str, Any]
    inventory: dict[str, Any]
    split: dict[str, Any]
    pairs: dict[str, Any]
    subjects: Mapping[str, dict[str, Any]]
    cases: Mapping[str, dict[str, Any]]


def stage5_sha256_array(array: np.ndarray) -> str:
    return hashlib.sha256(memoryview(np.ascontiguousarray(array))).hexdigest()


def _stage5_digest(domain: str, value: str) -> str:
    return hashlib.sha256((domain + value).encode("utf-8")).hexdigest()


def _stage5_atomic_save_npy(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(handle, "wb") as stream:
            np.save(stream, array, allow_pickle=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        with suppress(FileNotFoundError):
            os.unlink(temporary)
        raise


def _stage5_read_json(path: Path, *, schema: str, label: str) -> dict[str, Any]:
    if not path.is_file() or is_link_like(path):
        raise Stage5DataError(f"missing or linked {label}: {path}")
    payload = path.read_bytes()
    try:
        document = json.loads(payload, parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise Stage5DataError(f"invalid JSON in {label}: {path}") from exc
    if not isinstance(document, dict) or document.get("schema") != schema:
        raise Stage5DataError(f"invalid {label} schema: {path}")
    if readable_json_bytes(document) != payload:
        raise Stage5DataError(f"{label} is not canonical JSON: {path}")
    return document


def _stage5_source_arrays(
    source_path: Path,
    payload: bytes,
    *,
    image_shape: tuple[int, int, int],
) -> tuple[np.ndarray, np.ndarray]:
    try:
        value = pickle.loads(payload)
    except Exception as exc:
        raise Stage5DataError(f"could not deserialize OASIS source: {source_path}") from exc
    if not isinstance(value, tuple) or len(value) != 2:
        raise Stage5DataError(f"OASIS source must contain exactly (image, segmentation): {source_path}")
    image, label = value
    if not isinstance(image, np.ndarray) or image.shape != image_shape or image.dtype != np.float32:
        raise Stage5DataError(f"invalid image contract in {source_path}")
    if not isinstance(label, np.ndarray) or label.shape != image_shape or label.dtype != np.uint8:
        raise Stage5DataError(f"invalid segmentation contract in {source_path}")
    if not bool(np.isfinite(image).all()):
        raise Stage5DataError(f"non-finite image values in {source_path}")
    return np.ascontiguousarray(image), np.ascontiguousarray(label)


def _stage5_cache_record(
    path: Path,
    *,
    expected_shape: tuple[int, int, int],
    expected_array_sha256: str,
) -> dict[str, Any]:
    if not path.is_file() or is_link_like(path):
        raise Stage5DataError(f"missing or linked image cache: {path}")
    generation = file_generation(path)
    try:
        image = np.load(path, allow_pickle=False)
    except (OSError, ValueError) as exc:
        raise Stage5DataError(f"invalid image cache: {path}") from exc
    if (
        not isinstance(image, np.ndarray)
        or image.shape != expected_shape
        or image.dtype != np.float32
        or not image.flags.c_contiguous
        or image.dtype.hasobject
        or not bool(np.isfinite(image).all())
        or stage5_sha256_array(image) != expected_array_sha256
    ):
        raise Stage5DataError(f"image cache content changed: {path}")
    digest = sha256_file(path)
    if file_generation(path) != generation:
        raise Stage5DataError(f"image cache changed while it was being verified: {path}")
    return {"bytes": generation.size, "sha256": digest}


def build_stage5_split_manifests(
    subject_ids: Sequence[str],
    *,
    train_count: int = STAGE5_DEFAULT_TRAIN_COUNT,
    dev_count: int = STAGE5_DEFAULT_DEV_COUNT,
) -> tuple[dict[str, Any], dict[str, Any]]:
    subjects = list(subject_ids)
    if len(subjects) != len(set(subjects)):
        raise Stage5DataError("Stage 5 subject IDs must be unique")
    if train_count <= 0 or dev_count <= 0 or train_count + dev_count != len(subjects) or dev_count % 2:
        raise Stage5DataError("Stage 5 train/dev counts must cover all subjects and dev must be even")
    ranked = sorted((_stage5_digest(STAGE5_SPLIT_DOMAIN, subject), subject) for subject in subjects)
    dev_ranked = ranked[:dev_count]
    train_ranked = ranked[dev_count:]
    split = {
        "schema": STAGE5_SPLIT_MANIFEST_SCHEMA,
        "protocol_id": STAGE5_DATA_PROTOCOL_ID,
        "algorithm": {"development_selection": "lowest_sha256_rank", "split_domain": STAGE5_SPLIT_DOMAIN},
        "counts": {"all": len(subjects), "development": dev_count, "training": train_count},
        "development": [{"rank_sha256": rank, "subject_id": subject} for rank, subject in dev_ranked],
        "training": [{"rank_sha256": rank, "subject_id": subject} for rank, subject in train_ranked],
    }
    paired = sorted((_stage5_digest(STAGE5_PAIR_DOMAIN, subject), subject) for _, subject in dev_ranked)
    pair_rows: list[dict[str, Any]] = []
    cases: list[dict[str, Any]] = []
    for offset in range(0, len(paired), 2):
        subject_a, subject_b = sorted((paired[offset][1], paired[offset + 1][1]))
        pair_id = f"S5PAIR-{offset // 2 + 1:03d}"
        direction = _stage5_digest(STAGE5_DIRECTION_DOMAIN, f"{subject_a}\0{subject_b}")
        first, second = (subject_a, subject_b) if int(direction[-1], 16) % 2 == 0 else (subject_b, subject_a)
        directed = (
            {"case_id": f"{pair_id}-D0", "moving_subject_id": first, "fixed_subject_id": second},
            {"case_id": f"{pair_id}-D1", "moving_subject_id": second, "fixed_subject_id": first},
        )
        pair_rows.append(
            {
                "pair_id": pair_id,
                "subject_a": subject_a,
                "subject_b": subject_b,
                "direction_sha256": direction,
                "case_ids": [item["case_id"] for item in directed],
            }
        )
        cases.extend(directed)
    pair_manifest = {
        "schema": STAGE5_PAIR_MANIFEST_SCHEMA,
        "protocol_id": STAGE5_DATA_PROTOCOL_ID,
        "algorithm": {
            "pairing": "adjacent_after_sha256_rank",
            "pair_domain": STAGE5_PAIR_DOMAIN,
            "direction_domain": STAGE5_DIRECTION_DOMAIN,
            "both_directions": True,
        },
        "counts": {"unordered_pairs": len(pair_rows), "directed_cases": len(cases)},
        "pairs": pair_rows,
        "cases": cases,
    }
    return split, pair_manifest


def prepare_stage5_oasis_data(
    source_root: Path,
    output_root: Path,
    cache_root: Path,
    *,
    privileged_prepare_authorization: str,
    expected_subjects: int = STAGE5_DEFAULT_SUBJECT_COUNT,
    train_count: int = STAGE5_DEFAULT_TRAIN_COUNT,
    dev_count: int = STAGE5_DEFAULT_DEV_COUNT,
    image_shape: tuple[int, int, int] = STAGE5_DEFAULT_IMAGE_SHAPE,
) -> PreparedStage5Data:
    """Freeze the local OASIS All inventory and create an image-only cache."""
    if privileged_prepare_authorization != STAGE5_PREPARE_AUTHORIZATION:
        raise Stage5DataError("Stage 5 data preparation requires explicit privileged authorization")
    source_root = require_plain_directory(source_root, "OASIS All root", create=False, error=Stage5DataError)
    output_root = require_plain_directory(output_root, "Stage 5 data root", create=True, error=Stage5DataError)
    cache_root = require_plain_directory(cache_root, "Stage 5 image cache root", create=True, error=Stage5DataError)
    if len(image_shape) != 3 or any(
        isinstance(value, bool) or not isinstance(value, int) or value < 2 for value in image_shape
    ):
        raise Stage5DataError("Stage 5 image shape must contain three dimensions >= 2")
    candidates = sorted(source_root.glob("p_*.pkl"))
    if len(candidates) != expected_subjects:
        raise Stage5DataError(f"expected {expected_subjects} OASIS All files, found {len(candidates)}")
    records: list[dict[str, Any]] = []
    numeric_ids: set[int] = set()
    for source_path in candidates:
        match = _STAGE5_SOURCE_NAME.fullmatch(source_path.name)
        if match is None or is_link_like(source_path):
            raise Stage5DataError(f"invalid OASIS All source file: {source_path}")
        numeric_id = int(match.group(1))
        if numeric_id in numeric_ids:
            raise Stage5DataError(f"duplicate OASIS numeric ID: {numeric_id}")
        numeric_ids.add(numeric_id)
        payload = source_path.read_bytes()
        image, label = _stage5_source_arrays(source_path, payload, image_shape=image_shape)
        image_sha = stage5_sha256_array(image)
        cache_path = cache_root / f"p_{numeric_id:04d}.npy"
        if not cache_path.exists():
            _stage5_atomic_save_npy(cache_path, image)
        cache = _stage5_cache_record(cache_path, expected_shape=image_shape, expected_array_sha256=image_sha)
        cache["relative_path"] = cache_path.name
        records.append(
            {
                "subject_id": f"OASIS_ALL_P{numeric_id:04d}",
                "numeric_id": numeric_id,
                "source": {
                    "relative_path": source_path.name,
                    "bytes": len(payload),
                    "sha256": hashlib.sha256(payload).hexdigest(),
                },
                "cache": cache,
                "image_array_sha256": image_sha,
                "label_array_sha256": stage5_sha256_array(label),
            }
        )
    records.sort(key=lambda item: item["numeric_id"])
    if len({item["source"]["sha256"] for item in records}) != len(records):
        raise Stage5DataError("OASIS All contains duplicate source payloads")
    if len({item["image_array_sha256"] for item in records}) != len(records):
        raise Stage5DataError("OASIS All contains duplicate image arrays")
    inventory = {
        "schema": STAGE5_SOURCE_INVENTORY_SCHEMA,
        "protocol_id": STAGE5_DATA_PROTOCOL_ID,
        "source_scope": "LOCAL_OASIS_L2R_ALL_PICKLES",
        "source_contract": {
            "format": "pickle_tuple_image_segmentation",
            "shape": list(image_shape),
            "dtype_image": "float32",
            "dtype_segmentation": "uint8",
        },
        "cache_contract": {
            "format": "npy_single_array",
            "shape": list(image_shape),
            "dtype": "float32",
            "allow_pickle": False,
        },
        "subject_count": len(records),
        "subjects": records,
    }
    split, pairs = build_stage5_split_manifests(
        [item["subject_id"] for item in records], train_count=train_count, dev_count=dev_count
    )
    inventory_path = output_root / "source_inventory.json"
    split_path = output_root / "split_manifest.json"
    pair_path = output_root / "pair_manifest.json"
    write_immutable_bytes(inventory_path, readable_json_bytes(inventory), error=Stage5DataError)
    write_immutable_bytes(split_path, readable_json_bytes(split), error=Stage5DataError)
    write_immutable_bytes(pair_path, readable_json_bytes(pairs), error=Stage5DataError)
    contract = {
        "schema": STAGE5_DATA_CONTRACT_SCHEMA,
        "protocol_id": STAGE5_DATA_PROTOCOL_ID,
        "counts": {
            "source_subjects": len(records),
            "training_subjects": train_count,
            "development_subjects": dev_count,
            "unordered_development_pairs": dev_count // 2,
            "directed_development_cases": dev_count,
        },
        "files": {
            "source_inventory": {"path": inventory_path.name, "sha256": sha256_file(inventory_path)},
            "split_manifest": {"path": split_path.name, "sha256": sha256_file(split_path)},
            "pair_manifest": {"path": pair_path.name, "sha256": sha256_file(pair_path)},
        },
    }
    contract_path = output_root / "data_contract.json"
    write_immutable_bytes(contract_path, readable_json_bytes(contract), error=Stage5DataError)
    loaded = load_stage5_runtime_contract(contract_path)
    return PreparedStage5Data(
        contract_path=contract_path.resolve(),
        contract_sha256=loaded.contract_sha256,
        source_inventory_path=inventory_path.resolve(),
        split_manifest_path=split_path.resolve(),
        pair_manifest_path=pair_path.resolve(),
        cache_root=cache_root,
    )


def _stage5_contract_child(contract_path: Path, reference: Any, *, schema: str, label: str) -> dict[str, Any]:
    if not isinstance(reference, Mapping) or set(reference) != {"path", "sha256"}:
        raise Stage5DataError(f"invalid {label} reference")
    digest = reference.get("sha256")
    if not isinstance(digest, str) or _STAGE5_SHA256.fullmatch(digest) is None:
        raise Stage5DataError(f"invalid {label} SHA-256")
    root = require_plain_directory(
        contract_path.parent, "Stage 5 data contract root", create=False, error=Stage5DataError
    )
    path = resolve_inside_root(root, str(reference.get("path")), label=label, suffix=".json", error=Stage5DataError)
    if sha256_file(path) != digest:
        raise Stage5DataError(f"{label} SHA-256 drift")
    return _stage5_read_json(path, schema=schema, label=label)


def _validate_contract_envelope(contract: Mapping[str, Any]) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    """Check the top-level data contract and return its counts and file references."""
    if set(contract) != {"schema", "protocol_id", "counts", "files"}:
        raise Stage5DataError("invalid Stage 5 data contract fields")
    if contract.get("protocol_id") != STAGE5_DATA_PROTOCOL_ID:
        raise Stage5DataError("unexpected Stage 5 data protocol")
    counts = contract.get("counts")
    files = contract.get("files")
    expected_count_fields = {
        "source_subjects",
        "training_subjects",
        "development_subjects",
        "unordered_development_pairs",
        "directed_development_cases",
    }
    if not isinstance(counts, Mapping) or set(counts) != expected_count_fields:
        raise Stage5DataError("invalid Stage 5 data counts")
    if any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in counts.values()):
        raise Stage5DataError("Stage 5 data counts must be positive integers")
    if not isinstance(files, Mapping) or set(files) != {"source_inventory", "split_manifest", "pair_manifest"}:
        raise Stage5DataError("invalid Stage 5 data file references")
    return counts, files


def load_stage5_runtime_contract(contract_path: Path) -> Stage5RuntimeContract:
    """Authenticate the data contract and both manifests, then rebuild them from the inventory."""
    contract_path = Path(contract_path)
    contract = _stage5_read_json(contract_path, schema=STAGE5_DATA_CONTRACT_SCHEMA, label="Stage 5 data contract")
    counts, files = _validate_contract_envelope(contract)
    inventory = _stage5_contract_child(
        contract_path, files["source_inventory"], schema=STAGE5_SOURCE_INVENTORY_SCHEMA, label="source inventory"
    )
    split = _stage5_contract_child(
        contract_path, files["split_manifest"], schema=STAGE5_SPLIT_MANIFEST_SCHEMA, label="split manifest"
    )
    pairs = _stage5_contract_child(
        contract_path, files["pair_manifest"], schema=STAGE5_PAIR_MANIFEST_SCHEMA, label="pair manifest"
    )
    if any(document.get("protocol_id") != STAGE5_DATA_PROTOCOL_ID for document in (inventory, split, pairs)):
        raise Stage5DataError("Stage 5 child manifests belong to another protocol")
    subjects = _validate_subject_records(_validate_inventory_contract(inventory))
    if len(subjects) != counts["source_subjects"]:
        raise Stage5DataError("Stage 5 source count differs from its inventory")
    expected_split, expected_pairs = build_stage5_split_manifests(
        list(subjects), train_count=counts["training_subjects"], dev_count=counts["development_subjects"]
    )
    if split != expected_split or pairs != expected_pairs:
        raise Stage5DataError("Stage 5 split/pair manifests do not reconstruct from local inventory")
    cases = {item["case_id"]: dict(item) for item in pairs["cases"]}
    if len(cases) != counts["directed_development_cases"]:
        raise Stage5DataError("invalid Stage 5 directed-case inventory")
    return Stage5RuntimeContract(
        contract_path=contract_path.resolve(),
        contract_sha256=sha256_file(contract_path),
        contract=contract,
        inventory=inventory,
        split=split,
        pairs=pairs,
        subjects=subjects,
        cases=cases,
    )


def _validate_inventory_contract(inventory: Mapping[str, Any]) -> list[Any]:
    """Check the source inventory's own fields and its frozen source/cache formats."""
    if set(inventory) != {
        "schema",
        "protocol_id",
        "source_scope",
        "source_contract",
        "cache_contract",
        "subject_count",
        "subjects",
    }:
        raise Stage5DataError("invalid Stage 5 source inventory fields")
    records = inventory.get("subjects")
    if not isinstance(records, list) or inventory.get("subject_count") != len(records):
        raise Stage5DataError("invalid Stage 5 source inventory")
    source_contract = inventory.get("source_contract")
    cache_contract = inventory.get("cache_contract")
    if (
        inventory.get("source_scope") != "LOCAL_OASIS_L2R_ALL_PICKLES"
        or not isinstance(source_contract, Mapping)
        or set(source_contract) != {"format", "shape", "dtype_image", "dtype_segmentation"}
        or source_contract.get("format") != "pickle_tuple_image_segmentation"
        or source_contract.get("dtype_image") != "float32"
        or source_contract.get("dtype_segmentation") != "uint8"
        or not isinstance(cache_contract, Mapping)
        or set(cache_contract) != {"format", "shape", "dtype", "allow_pickle"}
        or cache_contract.get("format") != "npy_single_array"
        or cache_contract.get("dtype") != "float32"
        or cache_contract.get("allow_pickle") is not False
        or cache_contract.get("shape") != source_contract.get("shape")
    ):
        raise Stage5DataError("invalid Stage 5 source/cache contract")
    shape = source_contract.get("shape")
    if (
        not isinstance(shape, list)
        or len(shape) != 3
        or any(not isinstance(value, int) or value < 2 for value in shape)
    ):
        raise Stage5DataError("invalid Stage 5 image shape")
    return records


def _validate_subject_records(records: Sequence[Any]) -> dict[str, dict[str, Any]]:
    """Check every subject row against the frozen naming and digest contract."""
    subjects: dict[str, dict[str, Any]] = {}
    numeric_ids: set[int] = set()
    for record in records:
        if not isinstance(record, Mapping) or set(record) != {
            "subject_id",
            "numeric_id",
            "source",
            "cache",
            "image_array_sha256",
            "label_array_sha256",
        }:
            raise Stage5DataError("invalid Stage 5 subject record")
        subject_id = record.get("subject_id")
        numeric_id = record.get("numeric_id")
        source = record.get("source")
        cache = record.get("cache")
        if (
            not isinstance(subject_id, str)
            or not re.fullmatch(r"OASIS_ALL_P\d{4}", subject_id)
            or isinstance(numeric_id, bool)
            or not isinstance(numeric_id, int)
            or subject_id != f"OASIS_ALL_P{numeric_id:04d}"
            or numeric_id in numeric_ids
            or not isinstance(source, Mapping)
            or set(source) != {"relative_path", "bytes", "sha256"}
            or not isinstance(cache, Mapping)
            or set(cache) != {"relative_path", "bytes", "sha256"}
        ):
            raise Stage5DataError("malformed or duplicate Stage 5 subject")
        relative_posix(str(source.get("relative_path")), "source path", suffix=".pkl", error=Stage5DataError)
        relative_posix(str(cache.get("relative_path")), "cache path", suffix=".npy", error=Stage5DataError)
        if source["relative_path"] != f"p_{numeric_id:04d}.pkl" or cache["relative_path"] != f"p_{numeric_id:04d}.npy":
            raise Stage5DataError("Stage 5 source/cache path differs from numeric ID")
        for file_record in (source, cache):
            if (
                isinstance(file_record.get("bytes"), bool)
                or not isinstance(file_record.get("bytes"), int)
                or file_record["bytes"] <= 0
                or not isinstance(file_record.get("sha256"), str)
                or _STAGE5_SHA256.fullmatch(file_record["sha256"]) is None
            ):
                raise Stage5DataError("invalid Stage 5 file record")
        for key in ("image_array_sha256", "label_array_sha256"):
            if not isinstance(record.get(key), str) or _STAGE5_SHA256.fullmatch(record[key]) is None:
                raise Stage5DataError(f"invalid {key}")
        subjects[subject_id] = dict(record)
        numeric_ids.add(numeric_id)
    return subjects


class Stage5OasisImageStore:
    """Lazy, label-free access to the prepared image cache."""

    def __init__(self, contract_path: Path, cache_root: Path):
        self.runtime = load_stage5_runtime_contract(contract_path)
        self.cache_root = require_plain_directory(
            cache_root, "Stage 5 image cache root", create=False, error=Stage5DataError
        )
        self.image_shape = tuple(int(value) for value in self.runtime.inventory["cache_contract"]["shape"])
        self._verified: dict[str, FileGeneration] = {}

    def load_image(self, subject_id: str) -> np.ndarray:
        try:
            record = self.runtime.subjects[subject_id]
        except KeyError as exc:
            raise Stage5DataError(f"unknown Stage 5 subject: {subject_id}") from exc
        path = resolve_inside_root(
            self.cache_root,
            record["cache"]["relative_path"],
            label="image cache",
            suffix=".npy",
            error=Stage5DataError,
        )
        generation = file_generation(path)
        try:
            image = np.load(path, allow_pickle=False)
        except (OSError, ValueError) as exc:
            raise Stage5DataError(f"invalid image cache: {path}") from exc
        verify_content = not generation_cache_is_safe() or self._verified.get(subject_id) != generation
        if verify_content:
            if (
                not isinstance(image, np.ndarray)
                or image.shape != self.image_shape
                or image.dtype != np.float32
                or not image.flags.c_contiguous
                or image.dtype.hasobject
                or not bool(np.isfinite(image).all())
                or stage5_sha256_array(image) != record["image_array_sha256"]
            ):
                raise Stage5DataError(f"image cache content changed: {path}")
            observed = {"bytes": generation.size, "sha256": sha256_file(path)}
            if observed != {"bytes": record["cache"]["bytes"], "sha256": record["cache"]["sha256"]}:
                raise Stage5DataError(f"image cache file SHA-256 or size drift: {path}")
        if file_generation(path) != generation:
            raise Stage5DataError(f"image cache changed while it was being loaded: {path}")
        self._verified[subject_id] = generation
        return np.array(image, dtype=np.float32, order="C", copy=True)


class Stage5OasisImageDataset(Dataset):
    def __init__(self, contract_path: Path, cache_root: Path, *, split: str):
        self.store = Stage5OasisImageStore(contract_path, cache_root)
        if split not in {"training", "development"}:
            raise Stage5DataError("Stage5OasisImageDataset split must be training or development")
        self.subject_ids = tuple(item["subject_id"] for item in self.store.runtime.split[split])

    def __len__(self) -> int:
        return len(self.subject_ids)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, str]:
        subject_id = self.subject_ids[index]
        return torch.from_numpy(self.store.load_image(subject_id)).unsqueeze(0), subject_id


class Stage5OasisDirectedCaseDataset(Dataset):
    def __init__(self, contract_path: Path, cache_root: Path):
        self.store = Stage5OasisImageStore(contract_path, cache_root)
        self.cases = tuple(self.store.runtime.pairs["cases"])

    def __len__(self) -> int:
        return len(self.cases)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, str, str, str]:
        case = self.cases[index]
        moving_id = case["moving_subject_id"]
        fixed_id = case["fixed_subject_id"]
        moving = torch.from_numpy(self.store.load_image(moving_id)).unsqueeze(0)
        fixed = torch.from_numpy(self.store.load_image(fixed_id)).unsqueeze(0)
        return moving, fixed, case["case_id"], moving_id, fixed_id


def _stage5_transaction_json(path: Path, label: str) -> dict[str, Any]:
    payload = path.read_bytes()
    try:
        document = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise Stage5DataError(f"invalid {label}: {path}") from exc
    if not isinstance(document, dict) or canonical_json_bytes(document) != payload:
        raise Stage5DataError(f"{label} is not a canonical Stage 5 artifact")
    return document


class Stage5OasisEvaluationLabelStore:
    """Open only frozen development labels after training and decision barriers."""

    def __init__(
        self,
        contract_path: Path,
        source_root: Path,
        *,
        protocol_path: Path,
        training_barrier_path: Path,
        decision_barrier_path: Path,
        label_evaluation_authorization: str,
    ):
        if label_evaluation_authorization != STAGE5_LABEL_EVALUATION_AUTHORIZATION:
            raise Stage5DataError("opening Stage 5 labels requires explicit evaluator authorization")
        self.runtime = load_stage5_runtime_contract(contract_path)
        protocol = _stage5_transaction_json(protocol_path, "Stage 5 protocol")
        training = _stage5_transaction_json(training_barrier_path, "Stage 5 training barrier")
        decision = _stage5_transaction_json(decision_barrier_path, "Stage 5 decision barrier")
        try:
            validate_protocol_contract(protocol)
            validate_training_barrier(training, protocol, require_complete=True)
            validate_decision_barrier(decision, protocol, training, require_complete=True)
        except (KeyError, RuntimeError, TypeError, ValueError) as exc:
            raise Stage5DataError("Stage 5 label-access barriers are invalid or incomplete") from exc
        expected_cases = [item["case_id"] for item in self.runtime.pairs["cases"]]
        if (
            protocol["data_contract_sha256"] != self.runtime.contract_sha256
            or protocol["directed_case_ids"] != expected_cases
        ):
            raise Stage5DataError("Stage 5 protocol is bound to another data inventory")
        self.protocol_sha256 = canonical_sha256(protocol)
        self.training_barrier_sha256 = canonical_sha256(training)
        self.decision_barrier_sha256 = canonical_sha256(decision)
        self.source_root = require_plain_directory(
            source_root, "OASIS label-bearing source root", create=False, error=Stage5DataError
        )
        self.development_ids = frozenset(item["subject_id"] for item in self.runtime.split["development"])

    def load_label(self, subject_id: str) -> np.ndarray:
        if subject_id not in self.development_ids:
            raise Stage5DataError("Stage 5 evaluator may only open frozen development labels")
        record = self.runtime.subjects[subject_id]
        path = resolve_inside_root(
            self.source_root,
            record["source"]["relative_path"],
            label="label-bearing OASIS source",
            suffix=".pkl",
            error=Stage5DataError,
        )
        payload = path.read_bytes()
        if (
            len(payload) != record["source"]["bytes"]
            or hashlib.sha256(payload).hexdigest() != record["source"]["sha256"]
        ):
            raise Stage5DataError(f"label-bearing source changed: {path}")
        shape = tuple(int(value) for value in self.runtime.inventory["source_contract"]["shape"])
        image, label = _stage5_source_arrays(path, payload, image_shape=shape)
        if stage5_sha256_array(image) != record["image_array_sha256"]:
            raise Stage5DataError(f"source image changed: {path}")
        if stage5_sha256_array(label) != record["label_array_sha256"]:
            raise Stage5DataError(f"source segmentation changed: {path}")
        return np.array(label, dtype=np.uint8, order="C", copy=True)

    def load_case_labels(self, case_id: str) -> tuple[np.ndarray, np.ndarray]:
        try:
            case = self.runtime.cases[case_id]
        except KeyError as exc:
            raise Stage5DataError(f"unknown Stage 5 development case: {case_id}") from exc
        return self.load_label(case["moving_subject_id"]), self.load_label(case["fixed_subject_id"])


__all__ = [
    "STAGE5_DATA_CONTRACT_SCHEMA",
    "STAGE5_DATA_PROTOCOL_ID",
    "STAGE5_DEFAULT_DEV_COUNT",
    "STAGE5_DEFAULT_IMAGE_SHAPE",
    "STAGE5_DEFAULT_SUBJECT_COUNT",
    "STAGE5_DEFAULT_TRAIN_COUNT",
    "STAGE5_LABEL_EVALUATION_AUTHORIZATION",
    "STAGE5_PAIR_MANIFEST_SCHEMA",
    "STAGE5_PREPARE_AUTHORIZATION",
    "STAGE5_SOURCE_INVENTORY_SCHEMA",
    "STAGE5_SPLIT_MANIFEST_SCHEMA",
    "PreparedStage5Data",
    "Stage5DataError",
    "Stage5OasisDirectedCaseDataset",
    "Stage5OasisEvaluationLabelStore",
    "Stage5OasisImageDataset",
    "Stage5OasisImageStore",
    "Stage5RuntimeContract",
    "build_stage5_split_manifests",
    "load_stage5_runtime_contract",
    "prepare_stage5_oasis_data",
    "stage5_sha256_array",
]
