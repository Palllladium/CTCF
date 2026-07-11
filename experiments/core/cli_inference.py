from __future__ import annotations

import argparse

from experiments.core.cli_ctcf import add_ctcf_override_args
from utils.tto import TTO_MODES, TTO_SCHEDULES

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
    add_tto_args(p)
    add_ctcf_override_args(p, prefix="ctcf_")


def add_tto_args(p: argparse.ArgumentParser) -> None:
    group = p.add_argument_group("test-time optimisation")
    group.add_argument(
        "--tto_mode",
        type=str,
        choices=list(TTO_MODES),
        default="none",
        help="Per-pair field refinement: none | disp (dense residual) | svf (diffeomorphic) | inr.",
    )
    group.add_argument(
        "--tto_steps",
        type=int,
        default=200,
        help="Adam steps per pair.",
    )
    group.add_argument(
        "--tto_lr",
        type=float,
        default=0.01,
        help="Adam learning rate (voxel units).",
    )
    group.add_argument(
        "--tto_w_reg",
        type=float,
        default=1.0,
        help="Diffusion weight; keep at the training value so losses stay comparable.",
    )
    group.add_argument(
        "--tto_w_jac",
        type=float,
        default=0.005,
        help="Negative-Jacobian penalty weight.",
    )
    group.add_argument(
        "--tto_jac_eps",
        type=float,
        default=0.0,
        help="Overcorrection margin: penalise detJ < eps, not just detJ < 0.",
    )
    group.add_argument(
        "--tto_lr_schedule",
        type=str,
        choices=list(TTO_SCHEDULES),
        default="cosine",
        help="LR schedule over the TTO steps.",
    )
    group.add_argument(
        "--tto_mask",
        type=int,
        choices=[0, 1],
        default=0,
        help="Restrict the TTO loss to the brain (KAN-IDIR does this). Changes the loss scale.",
    )
    group.add_argument(
        "--tto_kan_degree",
        type=int,
        default=28,
        help="Max Chebyshev degree for --tto_mode kan.",
    )
    group.add_argument(
        "--tto_kan_k",
        type=int,
        default=12,
        help="Active random degrees per layer for --tto_mode randkan (drawn from 1..84).",
    )
    group.add_argument(
        "--tto_trace",
        type=int,
        nargs="*",
        default=None,
        help="Also score the field at these intermediate steps (one run yields the whole curve).",
    )


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
        "--img_size",
        type=int,
        nargs=3,
        default=(160, 192, 224),
        help="Input volume size D H W (needed by models that fix shape at build, e.g. SACB).",
    )
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
