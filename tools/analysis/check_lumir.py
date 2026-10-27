"""Check the official LUMIR input layout and selected CUDA environment before training."""

import argparse
import importlib
import json
import traceback
from pathlib import Path

import nibabel as nib
import numpy as np
import scipy
import torch
import torch.distributed as dist

from datasets.LUMIR import LUMIR_SHAPE, labelled_t1_pairs, load_lumir_volume, lumir_name_parts, read_lumir_json
from experiments.core.path_profiles import resolve_lumir_paths
from experiments.core.precision import configure_precision
from models.CTCF.configs import CONFIGS
from tools.analysis.run_artifacts import sha256_file


def check_data(paths, pairs_source, expected_coarse_pairs=18) -> dict:
    training, validation24 = read_lumir_json(paths["train_json"])
    training25, validation25 = read_lumir_json(paths["val_json"])
    if set(training25) != set(training) or len(set(training)) != len(training):
        raise ValueError("2024/2025 training lists differ; verify the intended official training protocol")
    if {lumir_name_parts(p)[0] for p in training} & {lumir_name_parts(p)[0] for pair in validation25 for p in pair}:
        raise ValueError("LUMIR25 validation subjects overlap the training list")
    coarse = labelled_t1_pairs(validation25, paths["label_dir"])
    if len(coarse) != expected_coarse_pairs:
        raise ValueError(f"expected {expected_coarse_pairs} public coarse-labelled T1 pairs, found {len(coarse)}")
    images = {paths["train_root"] / p for p in training}
    images.update(paths["val_root"] / p for pair in validation25 for p in pair)
    if pairs_source == "lumir24":
        images.update(paths["train_root"] / p for pair in validation24 for p in pair)
    for path in sorted(images):
        image = nib.load(path)
        if image.shape != LUMIR_SHAPE or not np.isfinite(image.affine).all():
            raise ValueError(f"invalid shape/affine: {path}: {image.shape}")
    for name in {p for pair in coarse for p in pair}:
        image = nib.load(paths["val_root"] / name)
        label_path = paths["label_dir"] / Path(name).name
        label = nib.load(label_path)
        if label.shape != image.shape or not np.array_equal(label.affine, image.affine):
            raise ValueError(f"label/image grid differs: {label_path}")
    sample = load_lumir_volume(paths["train_root"] / training[0], shape=LUMIR_SHAPE)
    if not np.isfinite(sample).all() or sample.max() <= sample.min():
        raise ValueError("the first training image is nonfinite or constant")
    return {
        "training_images": len(training),
        "coarse_validation_pairs": len(coarse),
        "export_pairs": len(validation24 if pairs_source == "lumir24" else validation25),
        "headers_checked": len(images),
        "shape": LUMIR_SHAPE,
        "train_json_sha256": sha256_file(paths["train_json"]),
        "val_json_sha256": sha256_file(paths["val_json"]),
        "scope": "all required headers, coarse-label grids and one full training image; not a full-payload checksum",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paths", type=int, default=3)
    parser.add_argument("--config", choices=CONFIGS, required=True)
    parser.add_argument("--precision", choices=("fp16", "bf16", "fp32", "tf32"), required=True)
    parser.add_argument("--pairs", choices=("lumir24", "lumir25"), required=True)
    parser.add_argument("--world-size", type=int, default=1)
    parser.add_argument("--data-only", action="store_true", help="Check files without requiring a CUDA node.")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report = {"config": args.config, "precision": args.precision, "world_size": args.world_size}
    try:
        report["data"] = check_data(resolve_lumir_paths(args), args.pairs)
        report["versions"] = {
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "nibabel": nib.__version__,
            "scipy": scipy.__version__,
        }
        if not args.data_only:
            if not torch.cuda.is_available() or torch.cuda.device_count() != args.world_size:
                raise RuntimeError("the selected CUDA devices do not match world_size")
            if args.world_size > 1 and not dist.is_nccl_available():
                raise RuntimeError("NCCL is unavailable in this PyTorch build")
            report["gpus"] = []
            for index in range(args.world_size):
                configure_precision(args.precision, torch.device("cuda", index))
                properties = torch.cuda.get_device_properties(index)
                report["gpus"].append({"name": properties.name, "total_memory": properties.total_memory})
            if CONFIGS[args.config].backbone in ("mamba", "vmamba"):
                report["versions"]["mamba_ssm"] = importlib.import_module("mamba_ssm").__version__
                importlib.import_module("selective_scan_cuda")
        report["status"] = "DATA_CHECKED" if args.data_only else "READY_FOR_PROFILE"
    except Exception as error:
        report.update(status="ERROR", error=f"{type(error).__name__}: {error}")
        traceback.print_exc()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"[LUMIR CHECK] {report['status']} {args.out}")
    return int(report["status"] == "ERROR")


if __name__ == "__main__":
    raise SystemExit(main())
