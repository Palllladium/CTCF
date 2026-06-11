from __future__ import annotations

import argparse
import copy

import torch
from torch import optim

from datasets.synthetic import build_synth_loaders
from experiments.core.cli_common import add_common_args
from experiments.core.cli_ctcf import add_ctcf_train_args, ctcf_overrides_from_args
from experiments.core.data_loaders import baseline_loader_builder
from experiments.core.model_adapters import get_model_adapter
from experiments.core.train_runtime import TrainContext, run_train
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
            **ctcf_overrides_from_args(args),
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
    device = setup_device(gpu_id=args.gpu, seed=0, deterministic=False)
    runner = Runner(args, device)
    run_train(args=args, runner=runner, build_loaders=build_loaders)


if __name__ == "__main__":
    main()
