"""Prepared LPBA40 T1 pairs with evaluation-only anatomical labels."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from tools.analysis.run_artifacts import sha256_file


def read_lpba_manifest(path: Path | str, *, verify_bytes: bool = False) -> dict:
    path = Path(path)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    ids = [row["id"] for row in manifest["subjects"]]
    if len(set(ids)) != len(ids) or not manifest["pairs"]:
        raise ValueError("LPBA40 subjects must be unique and the frozen pair list nonempty")
    pairs = [tuple(pair) for pair in manifest["pairs"]]
    if len(pairs) != len(set(pairs)) or any(a == b or a not in ids or b not in ids for a, b in pairs):
        raise ValueError("LPBA40 pairs contain duplicates, self pairs or unknown subjects")
    for row in manifest["subjects"]:
        file = Path(row["file"])
        if file.is_absolute() or not (path.parent / file).resolve().is_relative_to(path.parent.resolve()):
            raise ValueError("LPBA40 files must be relative to the manifest directory")
        row["file"] = str(path.parent / file)
        if not Path(row["file"]).is_file():
            raise FileNotFoundError(row["file"])
        if verify_bytes and sha256_file(Path(row["file"])) != row["sha256"]:
            raise ValueError(f"LPBA40 payload changed for {row['id']}")
    manifest["fingerprint"] = sha256_file(path)
    return manifest


def load_lpba_subject(row: dict) -> tuple[torch.Tensor, torch.Tensor]:
    with np.load(row["file"], allow_pickle=False) as data:
        image, labels = data["image"], data["labels"]
    if (
        image.ndim != 3
        or labels.shape != image.shape
        or not np.isfinite(image).all()
        or not np.issubdtype(labels.dtype, np.integer)
        or labels.min() < 0
    ):
        raise ValueError(f"Invalid LPBA40 image/labels for {row['id']}")
    return torch.from_numpy(np.ascontiguousarray(image[None])).float(), torch.from_numpy(labels[None].copy()).long()


class LPBAPairDataset(Dataset):
    def __init__(self, manifest: dict):
        self.rows = {row["id"]: row for row in manifest["subjects"]}
        self.pairs, self.shape = manifest["pairs"], tuple(manifest["shape"])

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        moving, fixed = self.pairs[index]
        x, x_seg = load_lpba_subject(self.rows[moving])
        y, y_seg = load_lpba_subject(self.rows[fixed])
        if tuple(x.shape[1:]) != self.shape or x.shape != y.shape:
            raise ValueError("LPBA40 image shape differs from its manifest")
        return x, y, x_seg, y_seg, f"{moving}__{fixed}"
