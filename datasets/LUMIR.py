from __future__ import annotations

import json
import random
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from torch.utils.data import Dataset

LUMIR_SHAPE = (160, 224, 192)
LUMIR_INTENSITY_MAX = 255.0
LUMIR_T1_MODALITY = "0000"


def lumir_name_parts(image_path: str) -> tuple[str, str]:
    """(subject, modality) of an official image path, e.g. ./imagesVal/LUMIRMRI_3455_0000.nii.gz -> (3455, 0000)."""
    parts = Path(image_path).name.removesuffix(".nii.gz").split("_")
    if len(parts) != 3 or parts[0] != "LUMIRMRI" or not image_path.endswith(".nii.gz"):
        raise ValueError(f"not an official LUMIR image path: {image_path}")
    return parts[1], parts[2]


def lumir_disp_name(fixed_path: str, moving_path: str) -> str:
    """File stem the official evaluator looks up: short form for one modality or the 0000/0001 pair."""
    fixed_subject, fixed_modality = lumir_name_parts(fixed_path)
    moving_subject, moving_modality = lumir_name_parts(moving_path)
    if fixed_modality == moving_modality or (fixed_modality == "0000" and moving_modality == "0001"):
        return f"disp_{fixed_subject}_{moving_subject}"
    return f"disp_{fixed_subject}_{fixed_modality}_{moving_subject}_{moving_modality}"


def read_lumir_json(json_path: str | Path) -> tuple[list[str], list[tuple[str, str]]]:
    """Training image paths and official (fixed, moving) validation pairs, relative as in the dataset JSON."""
    data = json.loads(Path(json_path).read_text(encoding="utf-8"))
    training = [entry["image"] for entry in data["training"]]
    pairs = [(entry["fixed"], entry["moving"]) for entry in data["validation"]]
    if len(training) != data["numTraining"] or len(pairs) != data["numValidation"]:
        raise ValueError(f"{json_path}: entry counts disagree with numTraining/numValidation")
    train_subjects = {lumir_name_parts(path)[0] for path in training}
    val_subjects = {lumir_name_parts(path)[0] for pair in pairs for path in pair}
    if train_subjects & val_subjects:
        raise ValueError(f"{json_path}: validation pairs reuse training subjects")
    return training, pairs


def load_lumir_volume(path: Path, shape: tuple[int, int, int] = LUMIR_SHAPE) -> np.ndarray:
    """Float32 volume in stored voxel order (no reorientation): the axes the official evaluator indexes."""
    data = nib.load(path).get_fdata(dtype=np.float32)
    if data.shape != shape:
        raise ValueError(f"{path}: shape {data.shape}, expected {shape}")
    return data


def labelled_t1_pairs(pairs: list[tuple[str, str]], label_dir: Path) -> list[tuple[str, str]]:
    """Official pairs whose two images are T1 (modality 0000) and have a label map named like the image."""
    return [
        pair
        for pair in pairs
        if all(lumir_name_parts(path)[1] == LUMIR_T1_MODALITY for path in pair)
        and all((label_dir / Path(path).name).is_file() for path in pair)
    ]


class LUMIRTrainDataset(Dataset):
    """Each training image as moving with a random other training image as fixed; (x, y) scaled to [0, 1]."""

    def __init__(self, root: str | Path, image_paths: list[str], shape: tuple[int, int, int] = LUMIR_SHAPE):
        self.root = Path(root)
        self.paths = list(image_paths)
        self.shape = tuple(shape)
        if len(self.paths) < 2:
            raise RuntimeError("LUMIRTrainDataset requires at least 2 images.")

    def __getitem__(self, index):
        partner = random.randrange(len(self.paths) - 1)
        if partner >= index:
            partner += 1
        x = load_lumir_volume(self.root / self.paths[index], self.shape) / LUMIR_INTENSITY_MAX
        y = load_lumir_volume(self.root / self.paths[partner], self.shape) / LUMIR_INTENSITY_MAX
        return torch.from_numpy(x[None]), torch.from_numpy(y[None])

    def __len__(self):
        return len(self.paths)


class LUMIRValDataset(Dataset):
    """Official (fixed, moving) pairs as (x=moving, y=fixed, x_seg, y_seg); labels are label_dir/<image name>."""

    def __init__(
        self,
        root: str | Path,
        pairs: list[tuple[str, str]],
        label_dir: str | Path,
        shape: tuple[int, int, int] = LUMIR_SHAPE,
    ):
        self.root = Path(root)
        self.pairs = list(pairs)
        self.label_dir = Path(label_dir)
        self.shape = tuple(shape)
        missing = [path for pair in self.pairs for path in pair if not (self.label_dir / Path(path).name).is_file()]
        if missing:
            raise FileNotFoundError(f"{len(missing)} label maps missing in {self.label_dir}, first {missing[0]}")

    def _labels(self, image_path: str) -> torch.Tensor:
        labels = load_lumir_volume(self.label_dir / Path(image_path).name, self.shape)
        return torch.from_numpy(np.rint(labels).astype(np.int16)[None])

    def __getitem__(self, index):
        fixed, moving = self.pairs[index]
        x = load_lumir_volume(self.root / moving, self.shape) / LUMIR_INTENSITY_MAX
        y = load_lumir_volume(self.root / fixed, self.shape) / LUMIR_INTENSITY_MAX
        return torch.from_numpy(x[None]), torch.from_numpy(y[None]), self._labels(moving), self._labels(fixed)

    def __len__(self):
        return len(self.pairs)
