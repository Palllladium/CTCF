from __future__ import annotations

import os
from pathlib import Path

# LUMIR layout under "root": the LUMIR 2024 training images and JSON; the public LUMIR 2025 validation images,
# JSON and coarse label maps in "val_root" (the 2025 validation set differs from the 2024 one).
_LUMIR_LAYOUT = {
    "train_json": "LUMIR_dataset.json",
    "val_root": "LUMIR25",
    "val_json": "LUMIR25_dataset.json",
    "label_dir": "simplelabelsVal",
}

PATHS: dict[int, dict[str, dict[str, str]]] = {
    1: {
        "OASIS": {
            "train_dir": "C:/Users/user/Documents/Education/MasterWork/datasets/OASIS_L2R_2021_task03/All",
            "val_dir": "C:/Users/user/Documents/Education/MasterWork/datasets/OASIS_L2R_2021_task03/Test",
        },
        "IXI": {
            "train_dir": "C:/Users/user/Documents/Education/MasterWork/datasets/IXI_data/Train",
            "val_dir": "C:/Users/user/Documents/Education/MasterWork/datasets/IXI_data/Val",
            "test_dir": "C:/Users/user/Documents/Education/MasterWork/datasets/IXI_data/Test",
            "atlas_path": "C:/Users/user/Documents/Education/MasterWork/datasets/IXI_data/atlas.pkl",
        },
        "LUMIR": {"root": "C:/Users/user/Documents/Education/MasterWork/datasets/LUMIR", **_LUMIR_LAYOUT},
    },
    2: {
        "OASIS": {
            "train_dir": "/home/roman/P/OASIS_L2R_2021_task03/All",
            "val_dir": "/home/roman/P/OASIS_L2R_2021_task03/Test",
        },
        "IXI": {
            "train_dir": "/home/roman/P/IXI_data/Train",
            "val_dir": "/home/roman/P/IXI_data/Val",
            "test_dir": "/home/roman/P/IXI_data/Test",
            "atlas_path": "/home/roman/P/IXI_data/atlas.pkl",
        },
        "LUMIR": {"root": "/home/roman/P/LUMIR", **_LUMIR_LAYOUT},
    },
    3: {
        "OASIS": {
            "train_dir": os.environ.get("CTCF_DATA_DIR", "/data") + "/OASIS_L2R_2021_task03/All",
            "val_dir": os.environ.get("CTCF_DATA_DIR", "/data") + "/OASIS_L2R_2021_task03/Test",
        },
        "IXI": {
            "train_dir": os.environ.get("CTCF_DATA_DIR", "/data") + "/IXI_data/Train",
            "val_dir": os.environ.get("CTCF_DATA_DIR", "/data") + "/IXI_data/Val",
            "test_dir": os.environ.get("CTCF_DATA_DIR", "/data") + "/IXI_data/Test",
            "atlas_path": os.environ.get("CTCF_DATA_DIR", "/data") + "/IXI_data/atlas.pkl",
        },
        "LUMIR": {"root": os.environ.get("CTCF_DATA_DIR", "/data") + "/LUMIR", **_LUMIR_LAYOUT},
    },
}


def get_dataset_paths(paths_id: int, ds: str) -> dict[str, str]:
    """Return the configured paths for a dataset/profile pair."""
    return PATHS[paths_id][ds.upper()]


def resolve_dataset_dirs(args, require_atlas: bool = False) -> tuple[str, str, str | None]:
    """Return (train_dir, val_dir, atlas_path) and raise if anything is missing."""
    paths = get_dataset_paths(args.paths, args.ds)
    train_dir = paths.get("train_dir", "")
    val_dir = paths.get("val_dir", "")

    if not os.path.isdir(train_dir):
        raise RuntimeError(f"Train dir not found: {train_dir}")
    if not os.path.isdir(val_dir):
        raise RuntimeError(f"Validation dir not found: {val_dir}")

    atlas_path: str | None = None
    if require_atlas:
        atlas_path = str(paths.get("atlas_path", "")).rstrip("/\\")
        if not atlas_path or not os.path.exists(atlas_path):
            raise RuntimeError(f"Atlas path not found: {atlas_path}")

    return train_dir, val_dir, atlas_path


def resolve_lumir_paths(args, required: tuple[str, ...] | None = None) -> dict[str, Path]:
    """Resolve LUMIR paths, checking only the entries needed by the caller."""
    paths = get_dataset_paths(args.paths, "LUMIR")
    root = Path(paths["root"])
    val_root = root / paths["val_root"]
    resolved = {
        "train_root": root,
        "train_json": root / paths["train_json"],
        "val_root": val_root,
        "val_json": val_root / paths["val_json"],
        "label_dir": val_root / paths["label_dir"],
    }
    for name in resolved if required is None else required:
        path = resolved[name]
        if not path.exists():
            raise RuntimeError(f"LUMIR {name} not found: {path}")
    return resolved


def print_experiment_header(args, ds_key: str) -> None:
    print(f">>> Experiment: {args.exp} | ds={args.ds} | paths={args.paths}")
    if ds_key == "LUMIR":
        for name, path in resolve_lumir_paths(args).items():
            print(f"    {name:<10} = {path}")
        return
    if ds_key not in ("OASIS", "IXI"):
        return

    paths = get_dataset_paths(args.paths, ds_key)
    print(f"    Train dir = {paths['train_dir']}")
    print(f"    Val dir   = {paths['val_dir']}")
    if ds_key == "IXI":
        atlas_path = str(paths["atlas_path"]).rstrip("/\\")
        print(f"    Atlas     = {atlas_path}")
