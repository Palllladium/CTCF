from __future__ import annotations

import argparse
import copy

import torch
from torch import optim

from datasets.synthetic import build_synth_loaders
from experiments.core.cli_args import add_common_args, optional_bool
from experiments.core.data_loaders import baseline_loader_builder
from experiments.core.model_adapters import get_model_adapter
from experiments.core.train_runtime import TrainContext, run_train
from models.CTCF.configs import CONFIGS
from utils import (
    DareDiffusion,
    RegisterModel,
    ctcf_schedule,
    dice_val,
    elastic_loss,
    icon_loss,
    neg_jacobian_penalty,
    setup_device,
)


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
            l1_base_ch=args.l1_base_ch,
            l3_base_ch=args.l3_base_ch,
            l3_error_mode=args.l3_error_mode,
            l3_iters=args.l3_iters,
            l3_unshared=optional_bool(args.l3_unshared),
            l1_half_res=optional_bool(args.l1_half_res),
            l2_full_res=optional_bool(args.l2_full_res),
            l3_full_res=optional_bool(args.l3_full_res),
            l3_svf=optional_bool(args.l3_svf),
            l3_num_heads=args.l3_num_heads,
        ).to(device)

        self.optimizer = optim.Adam(self.model.parameters(), lr=args.lr, amsgrad=True)
        self.img_size = tuple(self.model.img_size_full)
        self.ctx = TrainContext(device, vol_size=self.img_size, ncc_win=(9, 9, 9))
        self.reg_nearest = RegisterModel(self.img_size, mode="nearest").to(device) if self.is_synth else None
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

    def train_step(self, batch, epoch):
        args, ctx = self.args, self.ctx
        self._ema_update()

        if self.is_synth:
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
        L_jac = 0.5 * (neg_jacobian_penalty(flow_xy) + neg_jacobian_penalty(flow_yx)) * W_jac

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

        loss = L_ncc + L_icon + L_reg + L_jac + L_ema

        logs = {
            "all": loss.item(),
            "ncc": L_ncc.item(),
            "reg": L_reg.item(),
            "icon": L_icon.item(),
            "jac": L_jac.item(),
            "alpha_l1": alpha_l1,
            "alpha_l3": alpha_l3,
            "warm": warm,
        }
        if self.use_ema:
            logs["ema"] = float(L_ema.item())

        if self.is_synth:
            with torch.no_grad():
                def_seg_xy = self.reg_nearest((x_seg.float(), flow_xy.detach().float()))
                logs["dice_tr"] = float(dice_val(def_seg_xy.long(), y_seg, num_clus=args.synth_num_labels).item())

        return loss, logs


def _add_model_args(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--config",
        type=str,
        default="CTCF-CascadeA",
        choices=list(CONFIGS.keys()),
        help="Model config key.",
    )
    p.add_argument(
        "--time_steps",
        type=int,
        default=6,
        help="Number of velocity integration steps.",
    )
    p.add_argument(
        "--schedule_max_epoch",
        type=int,
        default=0,
        help="If >0, uses this epoch horizon for CTCF stage schedule (alpha/warm), independent of --max_epoch.",
    )
    p.add_argument(
        "--use_checkpoint",
        type=int,
        choices=[0, 1],
        default=1,
        help="Enable gradient checkpointing in Swin blocks.",
    )


def _add_loss_args(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--w_ncc",
        type=float,
        default=1.0,
        help="NCC similarity loss weight.",
    )
    p.add_argument(
        "--w_reg",
        type=float,
        default=None,
        help="Flow regularization loss weight (auto: IXI=4.0, others=1.0).",
    )
    p.add_argument(
        "--w_icon",
        type=float,
        default=0.05,
        help="ICON loss base weight.",
    )
    p.add_argument(
        "--w_jac",
        type=float,
        default=0.005,
        help="Negative Jacobian penalty base weight.",
    )
    p.add_argument(
        "--icon_mode",
        type=str,
        choices=["l1", "l2"],
        default="l1",
        help="ICON loss norm: l1 (default) or l2.",
    )


def _add_cascade_args(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--l1_from_start",
        type=int,
        choices=[0, 1],
        default=0,
        help="If 1, alpha_l1=1.0 from epoch 0 (skip schedule).",
    )
    p.add_argument(
        "--disable_l1",
        type=int,
        choices=[0, 1],
        default=0,
        help="If 1, force alpha_l1=0 (disable Level 1 coarse flow).",
    )
    p.add_argument(
        "--disable_l3",
        type=int,
        choices=[0, 1],
        default=0,
        help="If 1, force alpha_l3=0 (disable Level 3 refiner).",
    )
    p.add_argument(
        "--l1_base_ch",
        type=int,
        default=None,
        help="L1 coarse net base channels (default: config value, typically 16).",
    )
    p.add_argument(
        "--l3_base_ch",
        type=int,
        default=None,
        help="L3 refiner base channels (default: config value, typically 16).",
    )
    p.add_argument(
        "--l3_error_mode",
        type=str,
        choices=["absdiff", "gradmag", "ncc"],
        default=None,
        help="L3 error map mode.",
    )
    p.add_argument(
        "--l3_iters",
        type=int,
        default=None,
        help="Number of L3 refinement iterations (default: 1).",
    )
    p.add_argument(
        "--l3_unshared",
        type=int,
        choices=[0, 1],
        default=None,
        help="Override config: use separate L3 weights per iteration (requires l3_iters>1).",
    )
    p.add_argument(
        "--l1_half_res",
        type=int,
        choices=[0, 1],
        default=None,
        help="Override config: run L1 at half-res instead of quarter-res.",
    )
    p.add_argument(
        "--l2_full_res",
        type=int,
        choices=[0, 1],
        default=None,
        help="Override config: run L2 at full-res.",
    )
    p.add_argument(
        "--l3_full_res",
        type=int,
        choices=[0, 1],
        default=None,
        help="Override config: run L3 at full-res.",
    )
    p.add_argument(
        "--l3_svf",
        type=int,
        choices=[0, 1],
        default=None,
        help="Override config: integrate L3 delta as SVF via scaling-and-squaring.",
    )
    p.add_argument(
        "--l3_num_heads",
        type=int,
        default=None,
        help="M1 Multi-head L3: number of parallel flow heads with per-voxel learned routing (default: config value, normally 1 = single-head).",
    )


def _add_mechanism_args(p: argparse.ArgumentParser) -> None:
    # M2: EMA self-distillation
    p.add_argument(
        "--ema_decay",
        type=float,
        default=0.0,
        help="M2 EMA self-distillation: teacher decay rate (e.g., 0.999). 0 = disabled.",
    )
    p.add_argument(
        "--ema_lambda",
        type=float,
        default=0.0,
        help="M2 EMA self-distillation: weight on student-teacher flow L1 consistency loss. 0 = disabled.",
    )

    # M3: cascade-aware regularization
    p.add_argument(
        "--w_reg_l1",
        type=float,
        default=None,
        help="M3 cascade-aware reg: diffusion weight on the raw L1 flow (phi_l1). If any of w_reg_l1/l2/l3 set, replaces uniform w_reg.",
    )
    p.add_argument(
        "--w_reg_l2",
        type=float,
        default=None,
        help="M3 cascade-aware reg: diffusion weight on the L2 residual flow (phi_l2_residual; post-L1-init removed).",
    )
    p.add_argument(
        "--w_reg_l3",
        type=float,
        default=None,
        help="M3 cascade-aware reg: diffusion weight on the mean L3 delta (delta_l3; raw, before SVF).",
    )


def _add_reg_mode_args(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--reg_mode",
        type=str,
        choices=["diffusion", "dare", "elastic"],
        default="diffusion",
        help="Regularization mode: diffusion (default Grad3d), dare (DARE-minimal), elastic (Navier-Cauchy).",
    )
    p.add_argument(
        "--dare_beta",
        type=float,
        default=1.0,
        help="DARE beta: controls adaptive weighting strength.",
    )
    p.add_argument(
        "--elastic_mu",
        type=float,
        default=1.0,
        help="ElasticMorph mu (shear modulus).",
    )
    p.add_argument(
        "--elastic_lam",
        type=float,
        default=1.0,
        help="ElasticMorph lambda (Lame first parameter).",
    )


def _add_synth_args(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--synth_train_samples",
        type=int,
        default=256,
        help="Number of synthetic training pairs.",
    )
    p.add_argument(
        "--synth_val_samples",
        type=int,
        default=32,
        help="Number of synthetic validation pairs.",
    )
    p.add_argument(
        "--synth_num_labels",
        type=int,
        default=36,
        help="Number of synthetic segmentation labels.",
    )
    p.add_argument(
        "--synth_vol_size",
        type=int,
        nargs=3,
        default=(96, 96, 96),
        help="Synthetic volume size D H W (each must be divisible by 32).",
    )
    p.add_argument(
        "--synth_flow_max_disp",
        type=float,
        default=6.0,
        help="Max synthetic displacement amplitude in voxels.",
    )
    p.add_argument(
        "--synth_seed",
        type=int,
        default=123,
        help="Base seed for synthetic pair generation.",
    )


def parse_args():
    p = argparse.ArgumentParser()
    add_common_args(p, include_synth=True)
    p.set_defaults(exp="CTCF")
    _add_model_args(p)
    _add_loss_args(p)
    _add_cascade_args(p)
    _add_mechanism_args(p)
    _add_reg_mode_args(p)
    _add_synth_args(p)
    return p.parse_args()


def main():
    args = parse_args()
    if args.w_reg is None:
        args.w_reg = 4.0 if args.ds.upper() == "IXI" else 1.0
    build_loaders = build_synth_loaders if args.ds == "SYNTH" else baseline_loader_builder(args)
    device = setup_device(gpu_id=args.gpu, seed=0, deterministic=False)
    runner = Runner(args, device)
    run_train(args=args, runner=runner, build_loaders=build_loaders)


if __name__ == "__main__":
    main()
