from __future__ import annotations

import glob
import json
import os
from functools import partial
from pathlib import Path

import numpy as np
import torch.distributed as dist
from torch.utils.data import DataLoader
from torchvision import transforms

from datasets import IXI, LUMIR, OASIS
from experiments.core.path_profiles import resolve_dataset_dirs, resolve_lumir_paths
from utils import NumpyType, RandomFlip, SegNorm


def _find_split_files(train_dir: str, val_dir: str, tag: str) -> tuple[list[str], list[str]]:
    tr_files = glob.glob(os.path.join(train_dir, "*.pkl"))
    va_files = glob.glob(os.path.join(val_dir, "*.pkl"))

    if not tr_files:
        raise RuntimeError(f"{tag}: no *.pkl in Train dir = {train_dir}")
    if not va_files:
        raise RuntimeError(f"{tag}: no *.pkl in Validation dir = {val_dir}")
    return tr_files, va_files


def _build_loaders(
    train_ds,
    val_ds,
    batch_size: int,
    num_workers: int,
    val_bs: int,
    drop_last_train: bool,
    drop_last_val: bool,
) -> tuple[DataLoader, DataLoader]:
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=drop_last_train,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=val_bs,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=drop_last_val,
    )
    return train_loader, val_loader


def oasis_loaders(
    args,
    train_cls,
    val_cls,
    val_bs: int = 1,
    drop_last_train: bool = False,
    drop_last_val: bool = True,
) -> tuple[DataLoader, DataLoader]:
    """Build OASIS train/val loaders for a pair of dataset classes."""
    train_dir, val_dir, _ = resolve_dataset_dirs(args)
    if getattr(args, "oasis_split", ""):
        tr_files, dev = oasis_subject_paths(train_dir, args.oasis_split)
        va_files = oasis_validation_pairs(dev, Path(args.oasis_split).with_name("pair_manifest.json"))
        print(f"OASIS subject split: Train={len(tr_files)}, Dev={len(dev)}, Dev pairs={len(va_files)}")
    else:
        tr_files, va_files = _find_split_files(train_dir, val_dir, tag="OASIS")

    tfm = transforms.Compose([NumpyType((np.float32, np.int16))])
    if getattr(args, "distill_teacher", "none") != "none":
        tr = train_cls(tr_files, transforms=tfm, return_ids=True)
    else:
        tr = train_cls(tr_files, transforms=tfm)
    va = val_cls(va_files, transforms=tfm)
    return _build_loaders(
        train_ds=tr,
        val_ds=va,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        val_bs=val_bs,
        drop_last_train=drop_last_train,
        drop_last_val=drop_last_val,
    )


def oasis_subject_paths(root: str | Path, manifest: str | Path) -> tuple[list[str], list[str]]:
    split = json.loads(Path(manifest).read_text(encoding="utf-8"))
    groups = [[item["subject_id"] for item in split[key]] for key in ("training", "development")]
    if any(len(ids) < 2 or len(ids) != len(set(ids)) for ids in groups) or set(groups[0]) & set(groups[1]):
        raise ValueError("OASIS Train/Dev must contain distinct, disjoint subjects")
    paths = [[str(Path(root) / f"p_{name.rsplit('_P', 1)[1]}.pkl") for name in ids] for ids in groups]
    if any(not Path(path).is_file() for group in paths for path in group):
        raise FileNotFoundError("OASIS split references missing subject files")
    return paths[0], paths[1]


def oasis_validation_pairs(development: list[str], manifest: str | Path) -> list[tuple[str, str]]:
    records = json.loads(Path(manifest).read_text(encoding="utf-8"))["cases"]
    subjects = {f"OASIS_ALL_P{Path(path).stem[2:]}": path for path in development}
    pairs = [(row["moving_subject_id"], row["fixed_subject_id"]) for row in records]
    if (
        len(pairs) != len(subjects)
        or len(set(pairs)) != len(pairs)
        or {a for a, _ in pairs} != subjects.keys()
        or {b for _, b in pairs} != subjects.keys()
        or any(a == b or (b, a) not in pairs for a, b in pairs)
    ):
        raise ValueError("Dev manifest must pair its subjects once in both directions, without Train subjects")
    return [(subjects[moving], subjects[fixed]) for moving, fixed in pairs]


def ixi_loaders(
    args,
    train_cls,
    val_cls,
    val_bs: int = 1,
    drop_last_train: bool = True,
    drop_last_val: bool = True,
    flip_axes: tuple[int, ...] = (1, 2, 3),
) -> tuple[DataLoader, DataLoader]:
    """Build IXI train/val loaders for a pair of dataset classes."""
    train_dir, val_dir, atlas_path = resolve_dataset_dirs(args, require_atlas=True)
    tr_files, va_files = _find_split_files(train_dir, val_dir, tag="IXI")

    train_tfm = transforms.Compose([RandomFlip(flip_axes), NumpyType((np.float32, np.float32))])
    val_tfm = transforms.Compose([SegNorm(), NumpyType((np.float32, np.int16))])
    tr = train_cls(tr_files, atlas_path, transforms=train_tfm)
    va = val_cls(va_files, atlas_path, transforms=val_tfm)
    return _build_loaders(
        train_ds=tr,
        val_ds=va,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        val_bs=val_bs,
        drop_last_train=drop_last_train,
        drop_last_val=drop_last_val,
    )


def lumir_loaders(args, val_bs: int = 1) -> tuple[DataLoader, DataLoader]:
    """Build LUMIR loaders: the official training list; the labelled T1 pairs of the official validation list."""
    paths = resolve_lumir_paths(args)
    training, _ = LUMIR.read_lumir_json(paths["train_json"])
    _, official_pairs = LUMIR.read_lumir_json(paths["val_json"])
    pairs = LUMIR.labelled_t1_pairs(official_pairs, paths["label_dir"])
    if not pairs:
        raise RuntimeError(f"LUMIR: no official validation pair has two T1 label maps in {paths['label_dir']}")
    print(
        f"LUMIR: {len(training)} training images; {len(pairs)}/{len(official_pairs)} validation pairs are T1 with labels"
    )
    train = LUMIR.LUMIRTrainDataset(paths["train_root"], training)
    if getattr(args, "global_batch_size", 0):
        rank, world = (dist.get_rank(), dist.get_world_size()) if dist.is_initialized() else (0, 1)
        sampler = LUMIR.LUMIRPairSampler(
            train,
            global_batch=args.global_batch_size,
            updates=args.max_train_iters,
            rank=rank,
            world=world,
            seed=args.seed,
        )
        return (
            DataLoader(
                train,
                batch_size=args.batch_size,
                sampler=sampler,
                num_workers=args.num_workers,
                pin_memory=True,
                drop_last=True,
            ),
            DataLoader(
                LUMIR.LUMIRValDataset(paths["val_root"], pairs, paths["label_dir"]),
                batch_size=val_bs,
                num_workers=args.num_workers,
                pin_memory=True,
            ),
        )
    return _build_loaders(
        train_ds=train,
        val_ds=LUMIR.LUMIRValDataset(paths["val_root"], pairs, paths["label_dir"]),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        val_bs=val_bs,
        drop_last_train=False,
        drop_last_val=False,
    )


def loaders_baseline(args, ixi_flip_axes: tuple[int, ...] = (1, 2, 3)) -> tuple[DataLoader, DataLoader]:
    """Dispatch baseline data-loader factory by dataset name."""
    match args.ds:
        case "OASIS":
            return oasis_loaders(
                args,
                train_cls=OASIS.OASISBrainDataset,
                val_cls=OASIS.OASISBrainInferDataset,
                val_bs=1,
                drop_last_train=False,
                drop_last_val=True,
            )
        case "IXI":
            return ixi_loaders(
                args,
                train_cls=IXI.IXIBrainDataset,
                val_cls=IXI.IXIBrainInferDataset,
                val_bs=1,
                drop_last_train=True,
                drop_last_val=True,
                flip_axes=ixi_flip_axes,
            )
        case "LUMIR":
            return lumir_loaders(args)
        case _:
            raise ValueError(f"Unsupported dataset = '{args.ds}' for baseline loaders.")


def ixi_flip_axes_for(ds) -> tuple[int, ...]:
    """RandomFlip axes for [C, D, H, W] arrays.

    IXI returns (0,) = channel axis (size 1) → no effective spatial flip (atlas pairs are
    kept in canonical orientation); other datasets flip all three spatial axes (1, 2, 3).
    """
    if str(ds).upper() == "IXI":
        return (0,)
    return (1, 2, 3)


def baseline_loader_builder(args):
    """Return the baseline loader factory used by solo model trainers."""
    return partial(loaders_baseline, ixi_flip_axes=ixi_flip_axes_for(args.ds))
