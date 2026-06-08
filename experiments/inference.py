from __future__ import annotations

import argparse

from experiments.core.cli_args import add_common_args
from experiments.core.inference_runtime import InferRunner


def _add_run_args(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--model",
        required=True,
        choices=["tm-dca", "utsrmorph", "ctcf", "voxelmorph", "lkunet", "efficientmorph", "mambamorph", "vmambamorph"],
        help="Model family to run.",
    )
    p.add_argument(
        "--ckpt",
        required=True,
        help="Path to model checkpoint (.pth).",
    )
    p.add_argument(
        "--strict_ckpt",
        type=int,
        choices=[0, 1],
        default=1,
        help="Strict checkpoint key matching (1) or tolerant load (0).",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for reproducibility.",
    )
    p.add_argument(
        "--deterministic",
        action="store_true",
        help="Enable deterministic backend settings.",
    )
    p.add_argument(
        "--print_every",
        type=int,
        default=1,
        help="Console print period (cases).",
    )


def _add_output_args(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--save_flow",
        action="store_true",
        help="Save flow fields to compressed .npz files.",
    )
    p.add_argument(
        "--save_pngs",
        action="store_true",
        help="Save per-case preview PNGs.",
    )
    p.add_argument(
        "--png_limit",
        type=int,
        default=5,
        help="Max PNG count (-1 for all cases).",
    )
    p.add_argument(
        "--out_dir",
        type=str,
        default=None,
        help="Override output directory (default: results/infer/<DS>/<model>/<ckpt_stem>).",
    )
    p.add_argument(
        "--use_test",
        action="store_true",
        help="Use test_dir instead of val_dir (for IXI final evaluation).",
    )
    p.add_argument(
        "--hd95",
        action="store_true",
        help="Compute HD95 in addition to core protocol metrics.",
    )


def _add_model_config_args(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--time_steps",
        type=int,
        default=12,
        help="Integration steps for velocity-based models.",
    )
    p.add_argument(
        "--tm_config",
        type=str,
        default="TransMorph-3-LVL",
        help="TransMorph-DCA config key.",
    )
    p.add_argument(
        "--utsr_config",
        type=str,
        default="UTSRMorph-Large",
        help="UTSRMorph config key.",
    )
    p.add_argument(
        "--ctcf_config",
        type=str,
        default="CTCF-CascadeA",
        help="CTCF config key.",
    )
    p.add_argument(
        "--ctcf_l3_iters",
        type=int,
        default=None,
        help="Override CTCF Level-3 refinement iterations.",
    )
    p.add_argument(
        "--ctcf_l3_unshared",
        type=int,
        choices=[0, 1],
        default=None,
        help="Override CTCF unshared Level-3 iter modules.",
    )
    p.add_argument(
        "--ctcf_l1_half_res",
        type=int,
        choices=[0, 1],
        default=None,
        help="Override CTCF Level-1 half-res mode.",
    )
    p.add_argument(
        "--ctcf_l2_full_res",
        type=int,
        choices=[0, 1],
        default=None,
        help="Override CTCF Level-2 full-res mode.",
    )
    p.add_argument(
        "--ctcf_l3_full_res",
        type=int,
        choices=[0, 1],
        default=None,
        help="Override CTCF Level-3 full-res mode.",
    )
    p.add_argument(
        "--ctcf_l3_svf",
        type=int,
        choices=[0, 1],
        default=None,
        help="Override CTCF Level-3 SVF integration mode.",
    )
    p.add_argument(
        "--vxm_config",
        type=str,
        default="VxmDense",
        help="VoxelMorph config key.",
    )
    p.add_argument(
        "--lku_config",
        type=str,
        default="LKU-8",
        help="LKU-Net config key.",
    )
    p.add_argument(
        "--em_config",
        type=str,
        default="EfficientMorph_2x3_2_hires",
        help="EfficientMorph config key.",
    )
    p.add_argument(
        "--mamba_config",
        type=str,
        default="MambaMorph",
        help="MambaMorph config key.",
    )
    p.add_argument(
        "--mamba_diffeo",
        type=int,
        choices=[0, 1],
        default=1,
        help="Use diffeomorphic MambaMorph variant.",
    )
    p.add_argument(
        "--vmamba_config",
        type=str,
        default="VMambaMorph",
        help="VMambaMorph config key.",
    )


def parse_args():
    p = argparse.ArgumentParser()
    add_common_args(p, mode="infer")
    _add_run_args(p)
    _add_output_args(p)
    _add_model_config_args(p)
    return p.parse_args()


def main():
    args = parse_args()
    InferRunner(args).run()


if __name__ == "__main__":
    main()
