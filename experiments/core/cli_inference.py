from __future__ import annotations

import argparse

from experiments.core.cli_ctcf import add_ctcf_override_args

MODEL_CHOICES = [
    "tm-dca",
    "utsrmorph",
    "ctcf",
    "voxelmorph",
    "lkunet",
    "efficientmorph",
    "mambamorph",
    "vmambamorph",
    "corrmlp",
    "sacb",
]


def add_inference_args(p: argparse.ArgumentParser) -> None:
    add_inference_run_args(p)
    add_inference_output_args(p)
    add_inference_model_config_args(p)
    add_ctcf_override_args(p, prefix="ctcf_")


def add_inference_run_args(p: argparse.ArgumentParser) -> None:
    group = p.add_argument_group("inference run")
    group.add_argument(
        "--model",
        required=True,
        choices=MODEL_CHOICES,
        help="Model family to run.",
    )
    group.add_argument(
        "--ckpt",
        required=True,
        help="Path to model checkpoint (.pth).",
    )
    group.add_argument(
        "--strict_ckpt",
        type=int,
        choices=[0, 1],
        default=1,
        help="Strict checkpoint key matching (1) or tolerant load (0).",
    )
    group.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for reproducibility.",
    )
    group.add_argument(
        "--deterministic",
        action="store_true",
        help="Enable deterministic backend settings.",
    )
    group.add_argument(
        "--print_every",
        type=int,
        default=1,
        help="Console print period (cases).",
    )


def add_inference_output_args(p: argparse.ArgumentParser) -> None:
    group = p.add_argument_group("inference outputs")
    group.add_argument(
        "--save_flow",
        action="store_true",
        help="Save flow fields to compressed .npz files.",
    )
    group.add_argument(
        "--save_pngs",
        action="store_true",
        help="Save per-case preview PNGs.",
    )
    group.add_argument(
        "--png_limit",
        type=int,
        default=5,
        help="Max PNG count (-1 for all cases).",
    )
    group.add_argument(
        "--out_dir",
        type=str,
        default=None,
        help="Override output directory (default: results/infer/<DS>/<model>/<ckpt_stem>).",
    )
    group.add_argument(
        "--use_test",
        action="store_true",
        help="Use test_dir instead of val_dir (for IXI final evaluation).",
    )
    group.add_argument(
        "--hd95",
        action="store_true",
        help="Compute HD95 in addition to core protocol metrics.",
    )


def add_inference_model_config_args(p: argparse.ArgumentParser) -> None:
    group = p.add_argument_group("model configs")
    group.add_argument(
        "--time_steps",
        type=int,
        default=12,
        help="Integration steps for velocity-based models.",
    )
    group.add_argument(
        "--tm_config",
        type=str,
        default="TransMorph-3-LVL",
        help="TransMorph-DCA config key.",
    )
    group.add_argument(
        "--utsr_config",
        type=str,
        default="UTSRMorph-Large",
        help="UTSRMorph config key.",
    )
    group.add_argument(
        "--ctcf_config",
        type=str,
        default="CTCF-CascadeA",
        help="CTCF config key.",
    )
    group.add_argument(
        "--vxm_config",
        type=str,
        default="VxmDense",
        help="VoxelMorph config key.",
    )
    group.add_argument(
        "--lku_config",
        type=str,
        default="LKU-8",
        help="LKU-Net config key.",
    )
    group.add_argument(
        "--em_config",
        type=str,
        default="EfficientMorph_2x3_2_hires",
        help="EfficientMorph config key.",
    )
    group.add_argument(
        "--mamba_config",
        type=str,
        default="MambaMorph",
        help="MambaMorph config key.",
    )
    group.add_argument(
        "--mamba_diffeo",
        type=int,
        choices=[0, 1],
        default=1,
        help="Use diffeomorphic MambaMorph variant.",
    )
    group.add_argument(
        "--vmamba_config",
        type=str,
        default="VMambaMorph",
        help="VMambaMorph config key.",
    )
