from __future__ import annotations

import argparse
import copy
import glob
import os
import random

import torch
import torch.nn.functional as F
from torch import optim

from datasets.synthetic import build_synth_loaders
from experiments.core.cli_common import add_common_args
from experiments.core.cli_ctcf import add_ctcf_train_args, ctcf_overrides_from_args
from experiments.core.data_loaders import baseline_loader_builder
from experiments.core.inference_runtime import load_checkpoint_state
from experiments.core.model_adapters import get_model_adapter
from experiments.core.path_profiles import resolve_dataset_dirs
from experiments.core.train_runtime import TrainContext, run_train
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
from utils.field import differentiable_local_clip

# One-hot channel count covering OASIS (labels 0-35) and IXI (0-36); soft_dice_loss drops
# absent/background channels, so an over-wide count is harmless.
_SEG_NUM_CLASSES = 37
# Training images the tissue mixture is fitted on (sorted, first N): fixed before any pilot run.
_GMM_FIT_FILES = 40
# Clip margin above the published certificate eps 0.001, as in the inference repair chain.
_CERT_WORK_EPS = 0.0011
# P2b: epochs in which the parcellation learns alone before its consistency term reaches the flow.
_LATENT_SOLO_EPOCHS = 5


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

        synth_img_size = None
        synth_dwin = None
        if self.is_synth:
            synth_img_size = tuple(args.synth_vol_size)
            synth_dwin = self._adapt_swin_windows(synth_img_size)

        self.model = self.adapter.build(
            time_steps=args.time_steps,
            config_key=args.config,
            use_checkpoint=bool(args.use_checkpoint),
            synth_img_size=synth_img_size,
            synth_dwin=synth_dwin,
            **ctcf_overrides_from_args(args),
        ).to(device)
        if args.init_ckpt:
            if args.resume:
                raise ValueError("--init_ckpt starts a new run from given weights; do not combine it with --resume.")
            report = load_checkpoint_state(self.model, args.init_ckpt, strict=True)
            print(f">>> Initialised weights from {args.init_ckpt}: {report}")

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
        if any(v % 32 != 0 for v in img):
            raise ValueError(f"synth_vol_size={img} must be divisible by 32.")
        window_size = tuple(max(2, v // 32) for v in img)
        print(f"Synth cfg: img_size={img}, window_size={window_size}, dwin_size={window_size}")
        return window_size

    @torch.no_grad()
    def _forward_flow(self, x, y):
        use_amp = torch.cuda.is_available()
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
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

    def _anatomy_loss(self, x, y, flow_xy, flow_yx, epoch):
        """The label-free anatomical term selected by --anatomy_loss, symmetric in the two directions."""
        args = self.args
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

    def train_step(self, batch, epoch):
        args, ctx = self.args, self.ctx
        self._ema_update()

        if self.is_synth or args.w_dice > 0.0:
            if len(batch) < 4:
                raise RuntimeError("--w_dice > 0 needs a dataset that provides segmentations (x_seg, y_seg).")
            x, y, x_seg, y_seg = batch[0], batch[1], batch[2], batch[3]
            x_seg, y_seg = x_seg.to(self.device).long(), y_seg.to(self.device).long()
        else:
            x, y = batch[0], batch[1]
            x_seg = y_seg = None

        x, y = x.to(self.device).float(), y.to(self.device).float()

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

        L_anat, anat_logs = self._anatomy_loss(x, y, flow_xy, flow_yx, epoch)

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

        return loss, logs


def parse_args():
    p = argparse.ArgumentParser()
    add_common_args(p, include_synth=True)
    p.set_defaults(exp="CTCF")
    add_ctcf_train_args(p)
    return p.parse_args()


def main():
    args = parse_args()
    if args.w_reg is None:
        args.w_reg = 4.0 if args.ds.upper() == "IXI" else 1.0
    build_loaders = build_synth_loaders if args.ds == "SYNTH" else baseline_loader_builder(args)
    device = setup_device(gpu_id=args.gpu, seed=args.seed, deterministic=False)
    runner = Runner(args, device)
    run_train(args=args, runner=runner, build_loaders=build_loaders)


if __name__ == "__main__":
    main()
