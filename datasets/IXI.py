"""IXI pair datasets: a fixed atlas against each subject, intensity only, no labels."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

from tools.analysis.run_artifacts import sha256_file
from utils import SegNorm, pkload
from utils.common import validate_subject_pairs


class IXIBrainDataset(Dataset):
    """IXI atlas-to-subject training pairs (intensity only)."""

    def __init__(self, data_path, atlas_path, transforms):
        self.paths = list(data_path)
        self.transforms = transforms
        self.atlas_x, _ = pkload(atlas_path)

    def __getitem__(self, index):
        y, _ = pkload(self.paths[index])
        x = self.atlas_x
        x, y = x[None, ...], y[None, ...]
        x, y = self.transforms([x, y])
        x = torch.from_numpy(np.ascontiguousarray(x)).float()
        y = torch.from_numpy(np.ascontiguousarray(y)).float()
        return x, y

    def __len__(self):
        return len(self.paths)


class IXIBrainInferDataset(Dataset):
    """IXI atlas-to-subject validation/test pairs (intensity + segmentation)."""

    def __init__(self, data_path, atlas_path, transforms):
        self.paths = list(data_path)
        self.transforms = transforms
        self.atlas_x, self.atlas_seg = pkload(atlas_path)

    def __getitem__(self, index):
        y, y_seg = pkload(self.paths[index])
        x, x_seg = self.atlas_x, self.atlas_seg
        x, y = x[None, ...], y[None, ...]
        x_seg, y_seg = x_seg[None, ...], y_seg[None, ...]
        x, x_seg = self.transforms([x, x_seg])
        y, y_seg = self.transforms([y, y_seg])

        x = torch.from_numpy(np.ascontiguousarray(x)).float()
        y = torch.from_numpy(np.ascontiguousarray(y)).float()
        x_seg = torch.from_numpy(np.ascontiguousarray(x_seg)).long()
        y_seg = torch.from_numpy(np.ascontiguousarray(y_seg)).long()
        return x, y, x_seg, y_seg

    def __len__(self):
        return len(self.paths)


def read_paired_manifest(
    path: str | Path, *, require_review: bool = True, verify_bytes: bool = False, require_t2: bool = True
) -> dict:
    """Read the prepared Train/Val cohort and its frozen bidirectional validation pairs."""
    path = Path(path)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if require_review and not manifest.get("reviewed", False):
        raise ValueError("Review the IXI alignment previews and finalise the prepared manifest before training")
    groups = [manifest[key] for key in ("training", "validation")]
    ids = [[row["id"] for row in group] for group in groups]
    original_ids = [row["ixi_id"] for group in groups for row in group]
    if (
        any(len(group) < 2 or len(group) != len(set(group)) for group in ids)
        or set(ids[0]) & set(ids[1])
        or len(original_ids) != len(set(original_ids))
    ):
        raise ValueError("Paired IXI requires distinct Train/Val subjects without shared original IXI IDs")
    for key, group, expected_split in zip(("training", "validation"), groups, ("Train", "Val"), strict=True):
        if len(group) != manifest["expected_counts"][key] or any(row["split"] != expected_split for row in group):
            raise ValueError(f"Prepared IXI {key} differs from its frozen cohort or contains the wrong split")
        for row in group:
            for field in ("t1", "t2"):
                value = Path(row[field])
                row[field] = str(value if value.is_absolute() else path.parent / value)
                if field == "t2" and not require_t2:
                    continue
                if not Path(row[field]).is_file():
                    raise FileNotFoundError(row[field])
                if verify_bytes and sha256_file(Path(row[field])) != row[f"{field}_sha256"]:
                    raise ValueError(f"Prepared {field} bytes changed for {row['id']}")
    pairs = [tuple(pair) for pair in manifest["validation_pairs"]]
    validate_subject_pairs(pairs, ids[1])
    manifest["fingerprint"] = hashlib.sha256(path.read_bytes()).hexdigest()
    return manifest


class IXIPairSampler(Sampler):
    """One moving occurrence per subject/epoch; independent deterministic fixed-partner sampling."""

    def __init__(self, size: int, seed: int, fingerprint: str):
        self.size, self.seed, self.fingerprint, self.epoch = size, seed, fingerprint, 0
        if size < 2:
            raise ValueError("Subject-pair sampling needs at least two subjects")

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        for moving in torch.randperm(self.size, generator=generator).tolist():
            fixed = int(torch.randint(self.size - 1, (), generator=generator))
            yield moving, fixed + int(fixed >= moving)

    def __len__(self):
        return self.size


class IXIPairedDataset(Dataset):
    """Subject pairs with separate model inputs, training-only T2 observations and evaluation labels."""

    def __init__(self, manifest: dict, arm: str, *, validation: bool = False):
        if arm not in {"b0", "b1", "b2"}:
            raise ValueError(f"Unknown IXI contrast arm {arm}")
        self.rows = manifest["validation" if validation else "training"]
        self.arm, self.validation = arm, validation
        self.shape = tuple(manifest["shape"])
        index = {row["id"]: i for i, row in enumerate(self.rows)}
        self.pairs = [(index[a], index[b]) for a, b in manifest["validation_pairs"]] if validation else None

    def _subject(self, index):
        row = self.rows[index]
        t1, labels = pkload(row["t1"])
        needs_t2 = self.arm == "b2" or (self.arm == "b1" and not self.validation)
        t2 = np.load(row["t2"], allow_pickle=False) if needs_t2 else None
        if t1.shape != self.shape or (t2 is not None and t2.shape != self.shape):
            raise ValueError(f"Prepared image shape changed for {row['id']}")
        inputs = np.stack((t1, t2 if self.arm == "b2" else t1)).astype(np.float32)
        if self.validation:
            _, labels = SegNorm()([t1, labels])
            side = torch.from_numpy(np.ascontiguousarray(labels[None], dtype=np.int16)).long()
        else:
            side = torch.from_numpy(np.ascontiguousarray((t2 if t2 is not None else t1)[None], dtype=np.float32))
        return torch.from_numpy(inputs), side

    def __getitem__(self, index):
        moving, fixed = self.pairs[index] if self.validation else index
        x, x_side = self._subject(moving)
        y, y_side = self._subject(fixed)
        return x, y, x_side, y_side

    def __len__(self):
        return len(self.pairs) if self.validation else len(self.rows)
