from __future__ import annotations

import argparse
import copy
import glob
import json
import math
import os
import random
import statistics
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import optim

from datasets.LUMIR import LUMIR_SHAPE
from datasets.synthetic import build_synth_loaders
from experiments.core.cli_common import add_common_args
from experiments.core.cli_ctcf import add_ctcf_train_args, ctcf_overrides_from_args
from experiments.core.data_loaders import baseline_loader_builder
from experiments.core.inference_runtime import load_checkpoint_state
from experiments.core.model_adapters import get_model_adapter, volume_window_size
from experiments.core.path_profiles import resolve_dataset_dirs
from experiments.core.precision import configure_precision, precision_autocast
from experiments.core.train_runtime import TrainContext, run_train
from experiments.dispatch1.distill_teacher import DistillTeacher
from experiments.dispatch1.known_warp import KnownWarpSource, anchor_loss, effective_points
from utils import (
    DareDiffusion,
    RegisterModel,
    ctcf_schedule,
    dice_val,
    digital_fold_penalty,
    elastic_loss,
    icon_loss,
    neg_jacobian_penalty,
    pkload,
    setup_device,
    soft_dice_loss,
    trilinear_fold_penalty,
)
from utils.anatomy import (
    LatentParcellation,
    anatomix_features,
    augment_intensity,
    effective_classes,
    fit_tissue_gmm,
    information_loss,
    load_anatomix,
    tissue_posteriors,
    with_background,
)
from utils.feature_similarity import LearnedSimilarity, load_feature_net
from utils.field import differentiable_local_clip
from utils.losses import local_ncc_map, region_mean

# One-hot channel count covering OASIS (labels 0-35) and IXI (0-36); soft_dice_loss drops
# absent/background channels, so an over-wide count is harmless.
_SEG_NUM_CLASSES = 37
# Training images the tissue mixture is fitted on (sorted, first N): fixed before any pilot run.
_GMM_FIT_FILES = 40
# Clip margin above the published certificate eps 0.001, as in the inference repair chain.
_CERT_WORK_EPS = 0.0011
# P2b: epochs in which the parcellation learns alone before its consistency term reaches the flow.
_LATENT_SOLO_EPOCHS = 5
_AUX_CALIBRATION_PAIRS = 8
_AUX_CALIBRATION_SEED = 2026
_CHARBONNIER_EPS = 0.01  # voxels


def _gradient_ratio(base, aux, tensors) -> float:
    """Ratio of gradient norms, counting unused tensors as zero."""
    tensors = list(tensors)
    norms = []
    for index, loss in enumerate((base, aux)):
        grads = torch.autograd.grad(loss, tensors, retain_graph=index == 0, allow_unused=True)
        norms.append(math.sqrt(sum(float((g.float() ** 2).sum()) for g in grads if g is not None)))
    if any(not math.isfinite(n) or n <= 0.0 for n in norms):
        raise RuntimeError(f"calibration requires finite nonzero gradients of both losses, got {norms}")
    return norms[0] / norms[1]


def certified_crop_warp(flow, moving, sweeps, size, origin):
    """Warp ``moving`` by the certified-clipped ``flow`` on the even-aligned crop at ``origin``.

    Only the crop interior >= 8*sweeps voxels from its border is returned, where the crop clip equals the
    full-volume clip; sampling is normalised exactly as utils.spatial.SpatialTransformer does. Returns the
    warped interior and its absolute (z, y, x) slices.
    """
    margin = 8 * sweeps
    dims = flow.shape[2:]
    crop = flow[:, :, origin[0] : origin[0] + size, origin[1] : origin[1] + size, origin[2] : origin[2] + size]
    clipped = differentiable_local_clip(crop.float(), eps=_CERT_WORK_EPS, sweeps=sweeps)
    inner = clipped[:, :, margin : size - margin, margin : size - margin, margin : size - margin]
    start = [o + margin for o in origin]
    n = size - 2 * margin
    axes = [torch.arange(lo, lo + n, device=flow.device, dtype=inner.dtype) for lo in start]
    grid = torch.stack(torch.meshgrid(*axes, indexing="ij"))[None]
    locs = torch.stack([2.0 * ((grid[:, i] + inner[:, i]) / (dims[i] - 1) - 0.5) for i in range(3)], dim=-1)
    warped = F.grid_sample(moving, locs[..., [2, 1, 0]], align_corners=False, mode="bilinear")
    return warped, tuple(slice(lo, lo + n) for lo in start)


class Runner:
    """CTCF trainer with legacy cascade warm-up schedule and bidirectional losses."""

    def __init__(self, args, device):
        self.args = args
        self.device = device
        self.is_synth = args.ds == "SYNTH"
        self.adapter = get_model_adapter("ctcf")

        img_size = None
        if self.is_synth:
            img_size = tuple(args.synth_vol_size)
        elif args.ds == "LUMIR":
            img_size = LUMIR_SHAPE
        dwin_size = None if img_size is None else self._adapt_swin_windows(img_size)

        self.model = self.adapter.build(
            time_steps=args.time_steps,
            config_key=args.config,
            use_checkpoint=bool(args.use_checkpoint),
            img_size=img_size,
            dwin_size=dwin_size,
            **ctcf_overrides_from_args(args),
        ).to(device)
        if args.init_ckpt:
            if args.resume:
                raise ValueError("--init_ckpt starts a new run from given weights; do not combine it with --resume.")
            report = load_checkpoint_state(self.model, args.init_ckpt, strict=True)
            print(f">>> Initialised weights from {args.init_ckpt}: {report}")
        self._init_auxiliary_objectives(args, device)

        self.optimizer = optim.Adam(self.model.parameters(), lr=args.lr, amsgrad=True)
        self.img_size = tuple(self.model.img_size_full)
        self.ctx = TrainContext(device, vol_size=self.img_size, ncc_win=(9, 9, 9))
        self.reg_nearest = RegisterModel(self.img_size, mode="nearest").to(device) if self.is_synth else None
        # Differentiable (bilinear) warp of one-hot labels for the weakly-supervised Dice loss.
        if args.certified_steps > 0:
            margin = 8 * args.certified_steps
            if args.w_dice <= 0.0:
                raise ValueError("--certified_steps scores the clipped crop against labels; it needs --w_dice > 0.")
            if args.cert_patch % 2 or args.cert_patch <= 2 * margin:
                raise ValueError(f"--cert_patch must be even and larger than 2*8*sweeps = {2 * margin}.")
        uses_soft_targets = args.w_dice > 0.0 or args.anatomy_loss != "none"
        self.reg_bilin = RegisterModel(self.img_size, mode="bilinear").to(device) if uses_soft_targets else None
        self.tissue_gmm = None
        if args.anatomy_loss == "gmm":
            train_dir, _, _ = resolve_dataset_dirs(args)
            fit_files = sorted(glob.glob(os.path.join(train_dir, "*.pkl")))[:_GMM_FIT_FILES]
            self.tissue_gmm = fit_tissue_gmm(pkload(path)[0] for path in fit_files)
            print(
                f">>> Tissue GMM on {len(fit_files)} training images: means={self.tissue_gmm.means.round(4)} "
                f"stds={self.tissue_gmm.variances.round(6) ** 0.5} weights={self.tissue_gmm.weights.round(4)}"
            )
        self.parcellation = None
        if args.anatomy_loss == "latent":
            self.parcellation = LatentParcellation().to(device)
            self.optimizer.add_param_group({"params": self.parcellation.parameters()})
        self.feature_net = load_anatomix(device) if args.anatomy_loss == "foundation" else None
        self.learned_similarity = None
        if args.anatomy_loss == "learned":
            if not args.feature_ckpt:
                raise ValueError(
                    "--anatomy_loss learned needs --feature_ckpt from experiments.train_features (git tag rounds-final)."
                )
            self.learned_similarity = LearnedSimilarity(load_feature_net(args.feature_ckpt, device), args.feature_mode)
        self.forward_flow = self._forward_flow
        self.lr_policy = "ctcf"
        self._val_alpha_l1 = 0.0 if args.disable_l1 else 1.0
        self._val_alpha_l3 = 0.0 if args.disable_l3 else 1.0

        self.reg_mode = args.reg_mode
        if self.reg_mode == "dare":
            self.dare_reg = DareDiffusion(beta=args.dare_beta)
        elif self.reg_mode == "elastic":
            self._elastic_mu = args.elastic_mu
            self._elastic_lam = args.elastic_lam

        # M2: EMA self-distillation
        self.ema_decay = args.ema_decay
        self.ema_lambda = args.ema_lambda
        self.use_ema = self.ema_decay > 0.0 and self.ema_lambda > 0.0
        if self.use_ema:
            self.teacher = copy.deepcopy(self.model)
            for p in self.teacher.parameters():
                p.requires_grad = False
            self.teacher.eval()
        else:
            self.teacher = None

        # M3: cascade-aware (per-level) regularization
        w_l1 = args.w_reg_l1
        w_l2 = args.w_reg_l2
        w_l3 = args.w_reg_l3
        self.use_cascade_reg = any(v is not None for v in (w_l1, w_l2, w_l3))
        if self.use_cascade_reg:
            self.w_reg_l1 = w_l1 if w_l1 is not None else 1.0
            self.w_reg_l2 = w_l2 if w_l2 is not None else 1.0
            self.w_reg_l3 = w_l3 if w_l3 is not None else 1.0
            if self.reg_mode != "diffusion":
                raise ValueError("Cascade-aware reg (w_reg_l1/l2/l3) requires --reg_mode diffusion.")

    @staticmethod
    def _adapt_swin_windows(img):
        window_size = volume_window_size(img)
        print(f"Volume cfg: img_size={img}, window_size={window_size}, dwin_size={window_size}")
        return window_size

    @torch.no_grad()
    def _forward_flow(self, x, y):
        with self._amp():
            _, flow = self.model(x, y, alpha_l1=self._val_alpha_l1, alpha_l3=self._val_alpha_l3)
        return flow.float()

    def _ema_update(self):
        """M2: update the EMA teacher towards current student weights.

        Called at the start of train_step, so it reflects the student after the previous
        optimizer step (one-step lag, negligible for slow decay).
        """
        if not self.use_ema or self.teacher is None:
            return
        with torch.no_grad():
            for p_t, p_s in zip(self.teacher.parameters(), self.model.parameters(), strict=True):
                p_t.data.mul_(self.ema_decay).add_(p_s.data, alpha=1.0 - self.ema_decay)
            for b_t, b_s in zip(self.teacher.buffers(), self.model.buffers(), strict=True):
                b_t.data.copy_(b_s.data)

    def _certified_crop_dice(self, flow, moving_oh, fixed_oh):
        size = self.args.cert_patch
        origin = [2 * random.randrange((n - size) // 2 + 1) for n in flow.shape[2:]]
        warped, block = certified_crop_warp(flow, moving_oh, self.args.certified_steps, size, origin)
        return soft_dice_loss(warped, fixed_oh[(slice(None), slice(None), *block)])

    def _anatomy_loss(self, x, y, def_xy, def_yx, flow_xy, flow_yx, epoch):
        """The label-free anatomical term selected by --anatomy_loss, symmetric in the two directions."""
        args = self.args
        if args.anatomy_loss == "learned":
            with torch.autocast(device_type="cuda", enabled=False):
                sim = self.learned_similarity
                loss = 0.5 * (sim(x, def_xy, y, flow_xy) + sim(y, def_yx, x, flow_yx)) * args.w_anat
            return loss, {}

        if args.anatomy_loss == "gmm":
            x_tissue = tissue_posteriors(x, self.tissue_gmm)
            y_tissue = tissue_posteriors(y, self.tissue_gmm)
            warp_xy = self.reg_bilin((x_tissue, flow_xy))
            warp_yx = self.reg_bilin((y_tissue, flow_yx))
            return 0.5 * (soft_dice_loss(warp_xy, y_tissue) + soft_dice_loss(warp_yx, x_tissue)) * args.w_anat, {}

        if args.anatomy_loss == "latent":
            x_probs = torch.softmax(self.parcellation(x), dim=1)
            y_probs = torch.softmax(self.parcellation(y), dim=1)
            x_classes, y_classes = with_background(x_probs, x), with_background(y_probs, y)
            coupled = epoch >= _LATENT_SOLO_EPOCHS
            to_y = flow_xy if coupled else flow_xy.detach()
            to_x = flow_yx if coupled else flow_yx.detach()
            consistency = 0.5 * (
                soft_dice_loss(self.reg_bilin((x_classes, to_y)), y_classes)
                + soft_dice_loss(self.reg_bilin((y_classes, to_x)), x_classes)
            )
            information = 0.5 * (information_loss(x_probs, x) + information_loss(y_probs, y))
            x_augmented = torch.softmax(self.parcellation(augment_intensity(x)), dim=1)
            invariance = soft_dice_loss(with_background(x_augmented, x), x_classes.detach())
            loss = (consistency + information + invariance) * args.w_anat
            return loss, {"k_eff": effective_classes(x_probs.detach(), x)}

        if args.anatomy_loss == "foundation":
            with torch.no_grad():
                x_feat = anatomix_features(self.feature_net, x)
                y_feat = anatomix_features(self.feature_net, y)
            cos_xy = (F.normalize(self.reg_bilin((x_feat, flow_xy)), dim=1) * y_feat).sum(dim=1)
            cos_yx = (F.normalize(self.reg_bilin((y_feat, flow_yx)), dim=1) * x_feat).sum(dim=1)
            loss = 0.5 * ((1.0 - cos_xy)[y[:, 0] > 0].mean() + (1.0 - cos_yx)[x[:, 0] > 0].mean()) * args.w_anat
            return loss, {}

        return torch.zeros((), device=self.device, dtype=flow_xy.dtype), {}

    def _init_auxiliary_objectives(self, args, device):
        """Known-warp synthetic steps or a distillation teacher (a zero --distill_beta disables the teacher)."""
        self._step = 0
        self.known_warp, self.known_warp_radius, self.distill = None, 0.0, None
        if args.known_warp_every < 0 or not math.isfinite(args.distill_beta) or args.distill_beta < 0.0:
            raise ValueError("--known_warp_every and --distill_beta must be non-negative and finite.")
        if args.sim == "mse" and args.ncc_regions == "labels":
            raise ValueError("--sim mse replaces the NCC; it cannot be combined with --ncc_regions labels.")
        teaching = args.distill_teacher != "none" and (args.aux_calibrate or args.distill_beta > 0.0)
        if not (args.known_warp_every or teaching):
            return
        if args.known_warp_every and args.distill_teacher != "none":
            raise ValueError("known-warp steps and distillation are separate objectives.")
        if args.batch_size != 1:
            raise ValueError("known-warp steps and distillation teachers work on single pairs: --batch_size 1.")
        if teaching:
            if not args.init_ckpt:
                raise ValueError("--distill_teacher teaches with a frozen copy of the starting weights: --init_ckpt.")
            self.distill = DistillTeacher(
                self.model,
                args.distill_teacher,
                args.distill_pairs,
                device,
                self._train_files(),
                table=args.energy_table,
                proxy=args.proxy_values,
                seed=args.distill_seed,
            )
            return
        if args.w_dice > 0.0 or self.is_synth or args.anatomy_loss != "none":
            raise ValueError("known-warp steps are label-free: no --w_dice, --ds SYNTH or --anatomy_loss.")
        if not args.aux_calibrate and not (math.isfinite(args.known_warp_gamma) and args.known_warp_gamma > 0.0):
            raise ValueError("--known_warp_every needs --known_warp_gamma from an --aux_calibrate run.")
        stats = json.loads(Path(args.known_warp_stats).read_text(encoding="utf-8")) if args.known_warp_stats else None
        if args.known_warp_target == "deadzone":
            if not stats:
                raise ValueError("the dead-zone target needs --known_warp_stats")
            self.known_warp_radius = float(stats["deadzone_radius"])
        self.known_warp = KnownWarpSource(
            self._train_files(),
            device,
            seed=args.known_warp_seed,
            anchors=args.known_warp_anchors,
            target=args.known_warp_target,
            stats=stats,
        )

    def _train_files(self) -> list[str]:
        train_dir, _, _ = resolve_dataset_dirs(self.args)
        return sorted(glob.glob(os.path.join(train_dir, "*.pkl")))

    def _load_image(self, path) -> torch.Tensor:
        return torch.from_numpy(pkload(path)[0].copy()).to(self.device, torch.float32)[None, None]

    def train_step(self, batch, epoch):
        args = self.args
        self._ema_update()
        self._step += 1
        if self.known_warp is not None and self._step % args.known_warp_every == 0:
            pair = self.known_warp.sample()
            loss, logs, flow_xy, flow_yx = self._pair_loss(pair.moving, pair.fixed, None, None, epoch)
            aux, aux_logs = self._known_warp_loss(pair, flow_xy, flow_yx)
            weighted = args.known_warp_gamma * aux
            total = loss + weighted
            logs = {f"syn_{k}": v for k, v in logs.items()} | aux_logs
            logs.update(base=float(loss.detach()), kw_weighted=float(weighted.detach()), all=float(total.detach()))
            return total, logs
        x, y, x_seg, y_seg, ids = self._unpack(batch)
        loss, logs, flow_xy, flow_yx = self._pair_loss(x, y, x_seg, y_seg, epoch)
        if self.distill is not None:
            aux, aux_logs = self._distill_loss(x, y, ids, flow_xy, flow_yx)
            weighted = args.distill_beta * aux
            logs["base"] = float(loss.detach())
            loss = loss + weighted
            logs.update(aux_logs)
            logs.update(dist_weighted=float(weighted.detach()), all=float(loss.detach()))
        return loss, logs

    def _unpack(self, batch):
        args = self.args
        ids = (batch[4][0], batch[5][0]) if len(batch) == 6 else None
        if self.is_synth or args.w_dice > 0.0 or args.ncc_regions == "labels":
            if len(batch) < 4:
                raise RuntimeError("--w_dice/--ncc_regions need a dataset that provides segmentations.")
            x_seg, y_seg = batch[2].to(self.device).long(), batch[3].to(self.device).long()
        else:
            x_seg = y_seg = None
        return batch[0].to(self.device).float(), batch[1].to(self.device).float(), x_seg, y_seg, ids

    def _pair_loss(self, x, y, x_seg, y_seg, epoch):
        args, ctx = self.args, self.ctx

        schedule_max_epoch = args.schedule_max_epoch if args.schedule_max_epoch > 0 else args.max_epoch
        alpha_l1, alpha_l3, warm = ctcf_schedule(epoch=epoch, max_epoch=schedule_max_epoch)
        if args.l1_from_start:
            alpha_l1 = 1.0
        if args.finetune:
            alpha_l1 = alpha_l3 = warm = 1.0
        if args.disable_l1:
            alpha_l1 = 0.0
        if args.disable_l3:
            alpha_l3 = 0.0
        self._val_alpha_l1 = alpha_l1
        self._val_alpha_l3 = alpha_l3
        W_icon = args.w_icon * warm
        W_jac = args.w_jac * warm

        if self.use_cascade_reg:
            def_xy, flow_xy, breakdown_xy = self.model(
                x, y, alpha_l1=alpha_l1, alpha_l3=alpha_l3, return_breakdown=True
            )
            def_yx, flow_yx, breakdown_yx = self.model(
                y, x, alpha_l1=alpha_l1, alpha_l3=alpha_l3, return_breakdown=True
            )
        else:
            def_xy, flow_xy = self.model(x, y, alpha_l1=alpha_l1, alpha_l3=alpha_l3)
            def_yx, flow_yx = self.model(y, x, alpha_l1=alpha_l1, alpha_l3=alpha_l3)

        with torch.autocast(device_type="cuda", enabled=False):
            if args.sim == "mse":
                L_ncc = (
                    0.5 * (F.mse_loss(def_xy.float(), y.float()) + F.mse_loss(def_yx.float(), x.float())) * args.w_mse
                )
            elif args.ncc_regions == "labels":
                # The control's exact per-voxel map (same operator, argument order, window and eps): only the
                # averaging changes, from every voxel to every labelled structure of the fixed image.
                win, eps = ctx.ncc.win, ctx.ncc.eps
                per_structure_xy = region_mean(local_ncc_map(def_xy.float(), y.float(), win, eps), y_seg)
                per_structure_yx = region_mean(local_ncc_map(def_yx.float(), x.float(), win, eps), x_seg)
                L_ncc = -0.5 * (per_structure_xy + per_structure_yx) * args.w_ncc
            else:
                L_ncc = 0.5 * (ctx.ncc(def_xy.float(), y.float()) + ctx.ncc(def_yx.float(), x.float())) * args.w_ncc

        L_icon = icon_loss(flow_xy, flow_yx, mode=args.icon_mode) * W_icon
        tri_stats = None
        if args.jac_mode == "trilinear":
            p_xy, s_xy = trilinear_fold_penalty(
                flow_xy, mode=args.tri_pen_mode, reduce=args.tri_pen_reduce, return_stats=True
            )
            p_yx, s_yx = trilinear_fold_penalty(
                flow_yx, mode=args.tri_pen_mode, reduce=args.tri_pen_reduce, return_stats=True
            )
            raw_pen = 0.5 * (p_xy + p_yx)
            L_jac = raw_pen * W_jac
            tri_stats = {
                "jac_active_frac": 0.5 * (s_xy["active_frac"] + s_yx["active_frac"]),
                "jac_raw": float(raw_pen.item()),
            }
        else:
            jac_pen = digital_fold_penalty if args.jac_mode == "digital" else neg_jacobian_penalty
            L_jac = 0.5 * (jac_pen(flow_xy) + jac_pen(flow_yx)) * W_jac

        L_dice = torch.zeros((), device=self.device, dtype=flow_xy.dtype)
        if args.w_dice > 0.0 and x_seg is not None:
            x_oh = F.one_hot(x_seg.squeeze(1), _SEG_NUM_CLASSES).permute(0, 4, 1, 2, 3).float()
            y_oh = F.one_hot(y_seg.squeeze(1), _SEG_NUM_CLASSES).permute(0, 4, 1, 2, 3).float()
            warp_xy = self.reg_bilin((x_oh, flow_xy))
            warp_yx = self.reg_bilin((y_oh, flow_yx))
            L_dice = 0.5 * (soft_dice_loss(warp_xy, y_oh) + soft_dice_loss(warp_yx, x_oh)) * args.w_dice

        L_cert = torch.zeros((), device=self.device, dtype=flow_xy.dtype)
        if args.certified_steps > 0:
            # The Bernstein coefficients go through einsum, which autocast would run in float16.
            with torch.autocast(device_type="cuda", enabled=False):
                L_cert = (
                    0.5
                    * (self._certified_crop_dice(flow_xy, x_oh, y_oh) + self._certified_crop_dice(flow_yx, y_oh, x_oh))
                    * args.w_cert
                )

        L_anat, anat_logs = self._anatomy_loss(x, y, def_xy, def_yx, flow_xy, flow_yx, epoch)

        if self.use_cascade_reg:
            level_weights = (
                ("phi_l1", self.w_reg_l1),
                ("phi_l2_residual", self.w_reg_l2),
                ("delta_l3", self.w_reg_l3),
            )
            L_reg = torch.zeros((), device=self.device, dtype=flow_xy.dtype)
            for key, weight in level_weights:
                f_xy, f_yx = breakdown_xy.get(key), breakdown_yx.get(key)
                if f_xy is None or f_yx is None or weight == 0.0:
                    continue
                L_reg = L_reg + 0.5 * weight * (ctx.reg(f_xy) + ctx.reg(f_yx))
            L_reg = L_reg * args.w_reg
        elif self.reg_mode == "dare":
            L_reg = 0.5 * (self.dare_reg(flow_xy) + self.dare_reg(flow_yx)) * args.w_reg
        elif self.reg_mode == "elastic":
            mu = self._elastic_mu
            lam = self._elastic_lam
            L_reg = 0.5 * (elastic_loss(flow_xy, mu, lam) + elastic_loss(flow_yx, mu, lam)) * args.w_reg
        else:
            L_reg = 0.5 * (ctx.reg(flow_xy) + ctx.reg(flow_yx)) * args.w_reg

        L_ema = torch.zeros((), device=self.device, dtype=flow_xy.dtype)
        if self.use_ema and self.teacher is not None:
            with torch.no_grad():
                _, flow_xy_t = self.teacher(x, y, alpha_l1=alpha_l1, alpha_l3=alpha_l3)
                _, flow_yx_t = self.teacher(y, x, alpha_l1=alpha_l1, alpha_l3=alpha_l3)
            L_ema = (
                0.5
                * ((flow_xy - flow_xy_t.detach()).abs().mean() + (flow_yx - flow_yx_t.detach()).abs().mean())
                * self.ema_lambda
                * warm
            )

        loss = L_ncc + L_icon + L_reg + L_jac + L_ema + L_dice + L_anat + L_cert

        logs = {
            "all": loss.item(),
            "ncc": L_ncc.item(),
            "reg": L_reg.item(),
            "icon": L_icon.item(),
            "jac": L_jac.item(),
            "dice": L_dice.item(),
            "anat": L_anat.item(),
            "cert": L_cert.item(),
            **anat_logs,
            "alpha_l1": alpha_l1,
            "alpha_l3": alpha_l3,
            "warm": warm,
        }
        if tri_stats is not None:
            logs.update(tri_stats)
            if args.log_tri_gradnorm:
                # Decoupled from the main graph: recompute the penalty on the detached field and take the
                # gradient w.r.t. the field only. Measures how hard the penalty pushes the deformation
                # (the runaway signal), without perturbing the model's backward or the AMP scaler.
                gn = 0.0
                for f in (flow_xy, flow_yx):
                    fd = f.detach().clone().requires_grad_(True)
                    pen = trilinear_fold_penalty(fd, mode=args.tri_pen_mode, reduce=args.tri_pen_reduce)
                    (g,) = torch.autograd.grad(pen, fd)
                    gn += 0.5 * float(g.norm().item())
                logs["jac_gradnorm"] = gn

        if self.use_ema:
            logs["ema"] = float(L_ema.item())

        if self.is_synth:
            with torch.no_grad():
                def_seg_xy = self.reg_nearest((x_seg.float(), flow_xy.detach().float()))
                logs["dice_tr"] = float(dice_val(def_seg_xy.long(), y_seg, num_clus=args.synth_num_labels).item())

        return loss, logs, flow_xy, flow_yx

    def _known_warp_loss(self, pair, flow_xy, flow_yx):
        kind, radius = self.args.known_warp_target, self.known_warp_radius
        with torch.autocast(device_type="cuda", enabled=False):
            aux = 0.5 * (anchor_loss(flow_xy, pair.fm, kind, radius) + anchor_loss(flow_yx, pair.mf, kind, radius))
        with torch.no_grad():
            psi = effective_points(flow_xy, pair.fm.anchors)
            logs = {
                "kw": aux.detach().item(),
                "kw_epe": float(torch.linalg.vector_norm(psi - pair.fm.points, dim=-1).mean()),
                "kw_valid": pair.valid_fraction,
            }
            if kind == "set":
                inf = float("inf")
                mask = pair.fm.member_mask
                gap = torch.linalg.vector_norm(psi[:, None] - pair.fm.members, dim=-1).masked_fill(~mask, inf)
                still = torch.linalg.vector_norm(pair.fm.members - pair.fm.anchors[:, None].float(), dim=-1)
                logs["kw_set_size"] = float(mask.sum(1).float().mean())
                logs["kw_near_member"] = float((gap.min(1).values <= 0.5).float().mean())
                logs["kw_identity_admissible"] = float(
                    (still.masked_fill(~mask, inf).min(1).values <= 0.5).float().mean()
                )
        return aux, logs

    def _distill_targets(self, x, y, ids):
        if ids is None:
            raise RuntimeError("--distill_teacher needs subject stems from the dataset")
        moving_id, fixed_id = ids
        return (
            self.distill.target(x, y, moving_id, fixed_id),
            self.distill.target(y, x, fixed_id, moving_id),
        )

    def _distill_loss(self, x, y, ids, flow_xy, flow_yx, targets=None):
        """Bidirectional Charbonnier distance inside each fixed brain."""
        targets = self._distill_targets(x, y, ids) if targets is None else targets
        terms, taught = [], 0
        for flow, fixed, target in zip((flow_xy, flow_yx), (y, x), targets, strict=True):
            if target is None:
                terms.append(torch.zeros((), device=self.device))
                continue
            taught += 1
            with torch.autocast(device_type="cuda", enabled=False):
                diff = flow.float() - target
                charbonnier = torch.sqrt((diff**2).sum(1) + _CHARBONNIER_EPS**2)[0]
                terms.append(charbonnier[fixed[0, 0] > 0].mean())
        aux = 0.5 * (terms[0] + terms[1])
        return aux, {"dist": aux.detach().item(), "dist_taught": taught / 2}

    def _amp(self):
        return precision_autocast(self.args.precision, self.device)

    def _random_pair(self, files, rng):
        i, j = rng.sample(range(len(files)), 2)
        return self._load_image(files[i]), self._load_image(files[j]), (Path(files[i]).stem, Path(files[j]).stem)

    def _known_warp_ratios(self, epoch) -> tuple[list[float], dict]:
        """Base over anchor-loss gradient norms w.r.t. both flows, on synthetic pairs."""
        args, ratios = self.args, []
        source = KnownWarpSource(
            self._train_files(),
            self.device,
            seed=args.known_warp_seed + _AUX_CALIBRATION_SEED,
            anchors=args.known_warp_anchors,
            target=args.known_warp_target,
            stats=self.known_warp.stats,
        )
        for _ in range(_AUX_CALIBRATION_PAIRS):
            pair = source.sample()
            with self._amp():
                base, _, flow_xy, flow_yx = self._pair_loss(pair.moving, pair.fixed, None, None, epoch)
                aux, _ = self._known_warp_loss(pair, flow_xy, flow_yx)
            ratios.append(_gradient_ratio(base, aux, (flow_xy, flow_yx)))
        return ratios, {}

    def _distill_ratios(self, epoch) -> tuple[list[float], dict]:
        """Base over distillation gradient norms w.r.t. both flows, on real pairs the teacher teaches."""
        files, rng, ratios = self._train_files(), random.Random(_AUX_CALIBRATION_SEED), []
        examined = taught = 0
        while len(ratios) < _AUX_CALIBRATION_PAIRS and examined < 50 * _AUX_CALIBRATION_PAIRS:
            x, y, ids = self._random_pair(files, rng)
            targets = self._distill_targets(x, y, ids)
            examined += 1
            taught += sum(target is not None for target in targets)
            if all(target is None for target in targets):
                continue
            with self._amp():
                base, _, flow_xy, flow_yx = self._pair_loss(x, y, None, None, epoch)
                aux, _ = self._distill_loss(x, y, ids, flow_xy, flow_yx, targets=targets)
            ratios.append(_gradient_ratio(base, aux, (flow_xy, flow_yx)))
        exposure = taught / (2 * examined)
        if exposure < 0.05:
            print(f"WARNING: the teacher teaches only {exposure:.1%} of the directions it is shown", flush=True)
        return ratios, {"examined_pairs": examined, "taught_direction_fraction": exposure}

    def _mse_ratios(self) -> tuple[list[float], dict]:
        """NCC over MSE gradient norms w.r.t. the trainable parameters."""
        files, rng, ratios = self._train_files(), random.Random(_AUX_CALIBRATION_SEED), []
        params = [p for p in self.model.parameters() if p.requires_grad]
        for _ in range(_AUX_CALIBRATION_PAIRS):
            x, y, _ = self._random_pair(files, rng)
            with self._amp():
                def_xy, _ = self.model(x, y, alpha_l1=1.0, alpha_l3=1.0)
                def_yx, _ = self.model(y, x, alpha_l1=1.0, alpha_l3=1.0)
            ncc = 0.5 * (self.ctx.ncc(def_xy.float(), y) + self.ctx.ncc(def_yx.float(), x))
            mse = 0.5 * (F.mse_loss(def_xy.float(), y) + F.mse_loss(def_yx.float(), x))
            ratios.append(_gradient_ratio(ncc, mse, params))
        return ratios, {}

    def calibrate_aux(self, out_path: str) -> dict:
        """The weight of the enabled auxiliary objective: the median ratio of base to auxiliary gradient norms at the
        starting weights, every cascade level on. Written to ``out_path``; no training follows."""
        epoch = self.args.max_epoch  # past the cascade schedule, as with --finetune
        if self.known_warp is not None:
            key, (ratios, details) = "known_warp_gamma", self._known_warp_ratios(epoch)
        elif self.distill is not None:
            key, (ratios, details) = "distill_beta", self._distill_ratios(epoch)
        elif self.args.sim == "mse":
            key, (ratios, details) = "w_mse", self._mse_ratios()
        else:
            raise ValueError("--aux_calibrate needs --known_warp_every, --distill_teacher or --sim mse.")
        if len(ratios) < _AUX_CALIBRATION_PAIRS:
            raise RuntimeError(f"only {len(ratios)} calibration pairs were taught; the policy is too selective")
        result = {key: statistics.median(ratios), "ratios": ratios, "init_ckpt": self.args.init_ckpt, **details}
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        Path(out_path).write_text(json.dumps(result, indent=1), encoding="utf-8")
        print(f">>> {key} = {result[key]:.6g} from {len(ratios)} pairs -> {out_path}", flush=True)
        return result


def parse_args():
    p = argparse.ArgumentParser()
    add_common_args(p, include_synth=True, include_lumir=True)
    p.set_defaults(exp="CTCF")
    add_ctcf_train_args(p)
    return p.parse_args()


def main():
    args = parse_args()
    if args.w_reg is None:
        args.w_reg = 4.0 if args.ds.upper() == "IXI" else 1.0
    build_loaders = build_synth_loaders if args.ds == "SYNTH" else baseline_loader_builder(args)
    device = setup_device(gpu_id=args.gpu, seed=args.seed, deterministic=False)
    configure_precision(args.precision, device)
    runner = Runner(args, device)
    if args.aux_calibrate:
        runner.calibrate_aux(args.aux_calibrate)
        return
    run_train(args=args, runner=runner, build_loaders=build_loaders)


if __name__ == "__main__":
    main()
