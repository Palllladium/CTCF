from __future__ import annotations

import glob
import os
from functools import partial

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
