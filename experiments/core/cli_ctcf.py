from __future__ import annotations

import argparse

from experiments.core.cli_common import optional_bool
from experiments.core.precision import add_precision_arg
from models.CTCF.configs import CONFIGS

_INT_OVERRIDES = ("l1_base_ch", "l3_base_ch", "l3_iters", "l3_num_heads")
_BOOL_OVERRIDES = ("l3_unshared", "l1_half_res", "l2_full_res", "l3_full_res", "l3_svf")
_ERROR_MODE_CHOICES = ("absdiff", "gradmag", "ncc")
_CORR_MODE_CHOICES = ("none", "hadamard", "corr", "corr_feat")

_OVERRIDE_HELP = {
    "l1_base_ch": "L1 coarse-net base channels (default: config value).",
    "l3_base_ch": "L3 refiner base channels (default: config value).",
    "l3_iters": "Number of L3 refinement iterations (default: config value, typically 1).",
    "l3_num_heads": "M1 multi-head L3: parallel flow heads with per-voxel routing (default: config value).",
    "l3_error_mode": "L3 error-map mode.",
    "l3_corr_mode": "L3 correspondence cost-volume injected before the flow head (none/hadamard/corr/corr_feat).",
    "l3_unshared": "use separate L3 weights per iteration (requires l3_iters>1).",
    "l1_half_res": "run L1 at half-res instead of quarter-res.",
    "l2_full_res": "run L2 at full-res.",
    "l3_full_res": "run L3 at full-res.",
    "l3_svf": "integrate L3 delta as SVF via scaling-and-squaring.",
    "l3_ls_space": "INFERENCE-ONLY certified line-search: clip each L3 update to the largest "
    "trilinear-diffeomorphic fraction. off (default) | vel (integrate t*delta per probe) | disp "
    "(integrate once, scale the increment).",
    "l3_ls_eps": "certificate margin for --*l3_ls_space (tri_cert_bound >= eps at each L3 step).",
}

CTCF_OVERRIDE_KEYS = (
    *_INT_OVERRIDES,
    "l3_error_mode",
    "l3_corr_mode",
    "l3_ls_space",
    "l3_ls_eps",
    *_BOOL_OVERRIDES,
)


def add_ctcf_train_args(p: argparse.ArgumentParser) -> None:
    add_precision_arg(p)
    p.add_argument(
        "--global_batch_size",
        type=int,
        default=0,
        help="LUMIR global pairs per optimizer update, including accumulation; required with torchrun. "
        "Positive values make max_train_iters the update budget per epoch (0: one padded full pass).",
    )
    p.add_argument(
        "--perf_warmup_updates",
        type=int,
        default=0,
        help="Exclude this many initial updates per epoch from synchronized throughput timing.",
    )
    add_ctcf_model_args(p)
    add_ctcf_loss_args(p)
    add_ctcf_schedule_args(p)
    add_ctcf_override_args(p)
    add_ctcf_mechanism_args(p)
    add_ctcf_reg_mode_args(p)
    add_ctcf_synth_args(p)
    add_ctcf_auxiliary_args(p)


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
        help="Jacobian penalty base weight (ramped by the warm-up schedule).",
    )
    group.add_argument(
        "--jac_mode",
        type=str,
        choices=["central", "digital", "trilinear"],
        default="central",
        help="Jacobian penalty: central (legacy detJ, inert on SVF fields), digital (hinge on the ten "
        "Liu-et-al. determinants), or trilinear (hinge on the deployed grid_sample warp — the criterion "
        "we certify). Default central keeps every prior run reproducible.",
    )
    group.add_argument(
        "--tri_pen_mode",
        type=str,
        choices=["bernstein", "sampled"],
        default="bernstein",
        help="For --jac_mode trilinear: bernstein (sound 27-coefficient hinge) or sampled (cheaper "
        "interior-lattice proxy).",
    )
    group.add_argument(
        "--tri_pen_reduce",
        type=str,
        choices=["mean", "active"],
        default="mean",
        help="For --jac_mode trilinear: reduce the per-cell hinge by 'mean' (over all cells) or 'active' "
        "(over only violating cells). active concentrates the gradient where the field actually folds, since "
        "folds are sparse and mean smears them into a near-zero signal.",
    )
    group.add_argument(
        "--log_tri_gradnorm",
        type=int,
        choices=[0, 1],
        default=0,
        help="For --jac_mode trilinear: also log the L2 norm of the penalty's gradient w.r.t. the field "
        "(a decoupled extra backward). Off by default; enable for the scale-matched TRI_ACTIVE sweep to "
        "watch the runaway (active-reduce amplifies the gradient by ~1/active_frac as folds get sparse).",
    )
    group.add_argument(
        "--w_dice",
        type=float,
        default=0.0,
        help="Weakly-supervised soft-Dice loss weight on warped segmentations. 0 = off (unsupervised, "
        "the default for every prior run); ~1.0 is the standard weakly-supervised setting.",
    )
    group.add_argument(
        "--anatomy_loss",
        type=str,
        choices=["none", "gmm", "latent", "foundation", "learned"],
        default="none",
        help="Label-free anatomical target. gmm (P1): soft-Dice between warped and fixed soft tissue classes "
        "of a 3-component intensity mixture fitted on training images. latent (P2b): a co-trained 16-class "
        "parcellation kept consistent under the warp, informative and invariant to intensity changes. "
        "foundation (P2c): cosine similarity of frozen anatomix features (pip install anatomix first). "
        "learned (round 2): 1 - cosine of a frozen feature net (--feature_ckpt; experiments.train_features, tag rounds-final).",
    )
    group.add_argument(
        "--feature_ckpt",
        default="",
        help="--anatomy_loss learned: checkpoint written by experiments.train_features (git tag rounds-final).",
    )
    group.add_argument(
        "--feature_mode",
        type=str,
        choices=["warped_image", "warped_features"],
        default="warped_image",
        help="--anatomy_loss learned: features of the warped image, or the moving image's features warped.",
    )
    group.add_argument(
        "--w_anat",
        type=float,
        default=1.0,
        help="Weight of the --anatomy_loss term.",
    )
    group.add_argument(
        "--ncc_regions",
        type=str,
        choices=["none", "labels"],
        default="none",
        help="labels (round-1 oracle diagnostic): the local-NCC map is averaged with equal weight per labelled "
        "structure of the fixed image instead of per voxel. The labels only re-weight the similarity and never "
        "enter as a Dice target, yet they are supervision: the run is an upper-bound probe, not a label-free arm.",
    )
    group.add_argument(
        "--certified_steps",
        type=int,
        default=0,
        help="Pilot P3: sweeps of the certified local clip from identity, applied with its exact derivative "
        "to one random crop of each output; the soft-Dice of the clipped crop joins the loss. Inference "
        "deploys the same operator with --tto_clip_from_identity 1 --tto_clip_sweeps <same>. 0 = off.",
    )
    group.add_argument(
        "--cert_patch",
        type=int,
        default=64,
        help="Even crop edge for --certified_steps; Dice is scored >= 8*sweeps voxels inside its border.",
    )
    group.add_argument(
        "--w_cert",
        type=float,
        default=1.0,
        help="Weight of the certified-crop soft-Dice term (--certified_steps > 0).",
    )
    group.add_argument(
        "--init_ckpt",
        default="",
        help="Initialise model weights only (strict load) and train from epoch 0 with a fresh "
        "optimizer; unlike --resume it restores no epoch, optimizer or scaler state.",
    )
    group.add_argument(
        "--finetune",
        type=int,
        choices=[0, 1],
        default=0,
        help="Keep L1, L3 and the ICON/Jacobian warm-up fully on from epoch 0 (for continuing a trained "
        "cascade, where the warm-up ramp would switch its levels off).",
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
    group.add_argument(
        f"--{prefix}l3_corr_mode",
        type=str,
        choices=list(_CORR_MODE_CHOICES),
        default=None,
        help=_OVERRIDE_HELP["l3_corr_mode"],
    )
    group.add_argument(
        f"--{prefix}l3_ls_space",
        type=str,
        choices=["off", "vel", "disp"],
        default=None,
        help=_OVERRIDE_HELP["l3_ls_space"],
    )
    group.add_argument(
        f"--{prefix}l3_ls_eps",
        type=float,
        default=None,
        help=_OVERRIDE_HELP["l3_ls_eps"],
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


def add_ctcf_auxiliary_args(p: argparse.ArgumentParser) -> None:
    group = p.add_argument_group("CTCF auxiliary objectives")
    group.add_argument(
        "--sim",
        choices=["ncc", "mse"],
        default="ncc",
        help="Image similarity of the base loss.",
    )
    group.add_argument(
        "--w_mse",
        type=float,
        default=1.0,
        help="Weight of the MSE similarity (--sim mse).",
    )
    group.add_argument(
        "--known_warp_every",
        type=int,
        default=0,
        help="Every k-th training step is a known-warp synthetic pair (experiments.dispatch1.known_warp); 0 disables.",
    )
    group.add_argument(
        "--known_warp_target",
        choices=["point", "set", "structure", "deadzone"],
        default="point",
        help="Target of the anchors: exact point, admissible set, structure-tensor weight or dead zone.",
    )
    group.add_argument(
        "--known_warp_gamma",
        type=float,
        default=0.0,
        help="Weight of the anchor loss (from --aux_calibrate).",
    )
    group.add_argument(
        "--known_warp_seed",
        type=int,
        default=0,
        help="Seed of the synthetic stream, separate from pairs.",
    )
    group.add_argument(
        "--known_warp_anchors",
        type=int,
        default=2048,
        help="Anchors per direction and pair.",
    )
    group.add_argument(
        "--known_warp_stats",
        default="",
        help="Thresholds JSON from experiments.dispatch1.eval_known_warp --stats.",
    )
    group.add_argument(
        "--distill_teacher",
        choices=["none", "bridge_minimax", "bridge_proxy", "random_bridge", "direct", "recursion"],
        default="none",
        help="Teacher field from a frozen copy of the starting weights (experiments.dispatch1.distill_teacher).",
    )
    group.add_argument(
        "--distill_pairs",
        choices=["eligible", "all"],
        default="eligible",
        help="Pairs taught.",
    )
    group.add_argument(
        "--distill_beta",
        type=float,
        default=0.0,
        help="Weight of the distillation loss; zero disables the teacher.",
    )
    group.add_argument(
        "--distill_seed",
        type=int,
        default=0,
        help="Seed of the random-bridge teacher.",
    )
    group.add_argument(
        "--energy_table",
        default="",
        help="NPZ from experiments.dispatch1.energy_table.",
    )
    group.add_argument(
        "--proxy_values",
        default="",
        help="proxy_values.json from experiments.dispatch1.eval_bridge.",
    )
    group.add_argument(
        "--aux_calibrate",
        default="",
        help="Write the weight of the enabled auxiliary objective (gamma, beta or w_mse) to this JSON and exit.",
    )


def ctcf_overrides_from_args(args: argparse.Namespace, prefix: str = "") -> dict:
    overrides = {name: getattr(args, f"{prefix}{name}") for name in _INT_OVERRIDES}
    overrides["l3_error_mode"] = getattr(args, f"{prefix}l3_error_mode")
    overrides["l3_corr_mode"] = getattr(args, f"{prefix}l3_corr_mode")
    overrides["l3_ls_space"] = getattr(args, f"{prefix}l3_ls_space")
    overrides["l3_ls_eps"] = getattr(args, f"{prefix}l3_ls_eps")
    for name in _BOOL_OVERRIDES:
        overrides[name] = optional_bool(getattr(args, f"{prefix}{name}"))
    return overrides
