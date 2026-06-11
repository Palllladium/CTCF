from __future__ import annotations

import argparse

from experiments.core.cli_common import optional_bool
from models.CTCF.configs import CONFIGS

_INT_OVERRIDES = ("l1_base_ch", "l3_base_ch", "l3_iters", "l3_num_heads")
_BOOL_OVERRIDES = ("l3_unshared", "l1_half_res", "l2_full_res", "l3_full_res", "l3_svf")
_ERROR_MODE_CHOICES = ("absdiff", "gradmag", "ncc")

_OVERRIDE_HELP = {
    "l1_base_ch": "L1 coarse-net base channels (default: config value).",
    "l3_base_ch": "L3 refiner base channels (default: config value).",
    "l3_iters": "Number of L3 refinement iterations (default: config value, typically 1).",
    "l3_num_heads": "M1 multi-head L3: parallel flow heads with per-voxel routing (default: config value).",
    "l3_error_mode": "L3 error-map mode.",
    "l3_unshared": "use separate L3 weights per iteration (requires l3_iters>1).",
    "l1_half_res": "run L1 at half-res instead of quarter-res.",
    "l2_full_res": "run L2 at full-res.",
    "l3_full_res": "run L3 at full-res.",
    "l3_svf": "integrate L3 delta as SVF via scaling-and-squaring.",
}

CTCF_OVERRIDE_KEYS = (*_INT_OVERRIDES, "l3_error_mode", *_BOOL_OVERRIDES)


def add_ctcf_train_args(p: argparse.ArgumentParser) -> None:
    add_ctcf_model_args(p)
    add_ctcf_loss_args(p)
    add_ctcf_schedule_args(p)
    add_ctcf_override_args(p)
    add_ctcf_mechanism_args(p)
    add_ctcf_reg_mode_args(p)
    add_ctcf_synth_args(p)


def add_ctcf_model_args(p: argparse.ArgumentParser) -> None:
    group = p.add_argument_group("CTCF model")
    group.add_argument(
        "--config",
        type=str,
        default="CTCF-CascadeA",
        choices=list(CONFIGS.keys()),
        help="Model config key.",
    )
    group.add_argument(
        "--time_steps",
        type=int,
        default=6,
        help="Number of velocity integration steps.",
    )
    group.add_argument(
        "--schedule_max_epoch",
        type=int,
        default=0,
        help="If >0, use this epoch horizon for the CTCF stage schedule, independent of --max_epoch.",
    )
    group.add_argument(
        "--use_checkpoint",
        type=int,
        choices=[0, 1],
        default=1,
        help="Enable gradient checkpointing in Swin blocks.",
    )


def add_ctcf_loss_args(p: argparse.ArgumentParser) -> None:
    group = p.add_argument_group("CTCF losses")
    group.add_argument(
        "--w_ncc",
        type=float,
        default=1.0,
        help="NCC similarity loss weight.",
    )
    group.add_argument(
        "--w_reg",
        type=float,
        default=None,
        help="Flow regularization loss weight (auto: IXI=4.0, others=1.0).",
    )
    group.add_argument(
        "--w_icon",
        type=float,
        default=0.05,
        help="ICON loss base weight.",
    )
    group.add_argument(
        "--w_jac",
        type=float,
        default=0.005,
        help="Negative Jacobian penalty base weight.",
    )
    group.add_argument(
        "--icon_mode",
        type=str,
        choices=["l1", "l2"],
        default="l1",
        help="ICON loss norm: l1 (default) or l2.",
    )


def add_ctcf_schedule_args(p: argparse.ArgumentParser) -> None:
    group = p.add_argument_group("CTCF schedule")
    group.add_argument(
        "--l1_from_start",
        type=int,
        choices=[0, 1],
        default=0,
        help="If 1, alpha_l1=1.0 from epoch 0 (skip schedule).",
    )
    group.add_argument(
        "--disable_l1",
        type=int,
        choices=[0, 1],
        default=0,
        help="If 1, force alpha_l1=0 (disable Level 1 coarse flow).",
    )
    group.add_argument(
        "--disable_l3",
        type=int,
        choices=[0, 1],
        default=0,
        help="If 1, force alpha_l3=0 (disable Level 3 refiner).",
    )


def add_ctcf_override_args(p: argparse.ArgumentParser, prefix: str = "") -> None:
    group = p.add_argument_group("CTCF architecture overrides")
    for name in _INT_OVERRIDES:
        group.add_argument(
            f"--{prefix}{name}",
            type=int,
            default=None,
            help=_OVERRIDE_HELP[name],
        )
    group.add_argument(
        f"--{prefix}l3_error_mode",
        type=str,
        choices=list(_ERROR_MODE_CHOICES),
        default=None,
        help=_OVERRIDE_HELP["l3_error_mode"],
    )
    for name in _BOOL_OVERRIDES:
        group.add_argument(
            f"--{prefix}{name}",
            type=int,
            choices=[0, 1],
            default=None,
            help=f"Override config (0/1): {_OVERRIDE_HELP[name]}",
        )


def add_ctcf_mechanism_args(p: argparse.ArgumentParser) -> None:
    group = p.add_argument_group("CTCF mechanisms")
    group.add_argument(
        "--ema_decay",
        type=float,
        default=0.0,
        help="M2 EMA self-distillation: teacher decay rate (e.g., 0.999). 0 = disabled.",
    )
    group.add_argument(
        "--ema_lambda",
        type=float,
        default=0.0,
        help="M2 EMA self-distillation: weight on student-teacher flow L1 consistency loss. 0 = disabled.",
    )
    group.add_argument(
        "--w_reg_l1",
        type=float,
        default=None,
        help=(
            "M3 cascade-aware reg: diffusion weight on phi_l1. If any of w_reg_l1/l2/l3 is set, replaces uniform w_reg."
        ),
    )
    group.add_argument(
        "--w_reg_l2",
        type=float,
        default=None,
        help="M3 cascade-aware reg: diffusion weight on phi_l2_residual.",
    )
    group.add_argument(
        "--w_reg_l3",
        type=float,
        default=None,
        help="M3 cascade-aware reg: diffusion weight on delta_l3.",
    )


def add_ctcf_reg_mode_args(p: argparse.ArgumentParser) -> None:
    group = p.add_argument_group("CTCF regularization modes")
    group.add_argument(
        "--reg_mode",
        type=str,
        choices=["diffusion", "dare", "elastic"],
        default="diffusion",
        help="Regularization mode: diffusion (default Grad3d), dare (DARE-minimal), elastic (Navier-Cauchy).",
    )
    group.add_argument(
        "--dare_beta",
        type=float,
        default=1.0,
        help="DARE beta: controls adaptive weighting strength.",
    )
    group.add_argument(
        "--elastic_mu",
        type=float,
        default=1.0,
        help="ElasticMorph mu (shear modulus).",
    )
    group.add_argument(
        "--elastic_lam",
        type=float,
        default=1.0,
        help="ElasticMorph lambda (Lame first parameter).",
    )


def add_ctcf_synth_args(p: argparse.ArgumentParser) -> None:
    group = p.add_argument_group("CTCF synthetic data")
    group.add_argument(
        "--synth_train_samples",
        type=int,
        default=256,
        help="Number of synthetic training pairs.",
    )
    group.add_argument(
        "--synth_val_samples",
        type=int,
        default=32,
        help="Number of synthetic validation pairs.",
    )
    group.add_argument(
        "--synth_num_labels",
        type=int,
        default=36,
        help="Number of synthetic segmentation labels.",
    )
    group.add_argument(
        "--synth_vol_size",
        type=int,
        nargs=3,
        default=(96, 96, 96),
        help="Synthetic volume size D H W (each must be divisible by 32).",
    )
    group.add_argument(
        "--synth_flow_max_disp",
        type=float,
        default=6.0,
        help="Max synthetic displacement amplitude in voxels.",
    )
    group.add_argument(
        "--synth_seed",
        type=int,
        default=123,
        help="Base seed for synthetic pair generation.",
    )


def ctcf_overrides_from_args(args: argparse.Namespace, prefix: str = "") -> dict:
    overrides = {name: getattr(args, f"{prefix}{name}") for name in _INT_OVERRIDES}
    overrides["l3_error_mode"] = getattr(args, f"{prefix}l3_error_mode")
    for name in _BOOL_OVERRIDES:
        overrides[name] = optional_bool(getattr(args, f"{prefix}{name}"))
    return overrides
