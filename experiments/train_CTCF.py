import argparse
from functools import partial

import torch
from torch import optim

from datasets.synthetic import build_synth_loaders
from experiments.core.model_adapters import get_model_adapter
from experiments.core.train_rules import apply_ctcf_dataset_defaults
from experiments.core.train_runtime import Ctx, add_common_args, loaders_baseline, run_train
from utils import RegisterModel, ctcf_schedule, dice_val, icon_loss, neg_jacobian_penalty, setup_device


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
            synth_img_size = tuple(int(v) for v in args.synth_vol_size)
            synth_dwin = self._adapt_swin_windows(synth_img_size)

        self.model = self.adapter.build(
            time_steps=int(args.time_steps),
            config_key=args.config,
            use_checkpoint=bool(int(args.use_checkpoint)),
            synth_img_size=synth_img_size,
            synth_dwin=synth_dwin,
            l1_base_ch=getattr(args, 'l1_base_ch', None),
            l3_base_ch=getattr(args, 'l3_base_ch', None),
            l3_error_mode=getattr(args, 'l3_error_mode', None),
            prealign_encoder=True if int(getattr(args, 'prealign_encoder', 0)) else None,
        ).to(device)

        self.optimizer = optim.Adam(self.model.parameters(), lr=args.lr, amsgrad=True)
        self.img_size = tuple(int(v) for v in self.model.img_size_full)
        self.ctx = Ctx(device, vol_size=self.img_size, ncc_win=(9, 9, 9))
        self.reg_nearest = RegisterModel(self.img_size, mode="nearest").to(device) if self.is_synth else None
        self.forward_flow = self._forward_flow
        self.lr_policy = "ctcf"
        self._val_alpha_l1 = 0.0 if int(getattr(args, 'disable_l1', 0)) else 1.0
        self._val_alpha_l3 = 0.0 if int(getattr(args, 'disable_l3', 0)) else 1.0


    @staticmethod
    def _adapt_swin_windows(img):
        img = tuple(int(v) for v in img)
        if any(v % 32 != 0 for v in img):
            raise ValueError(f"synth_vol_size={img} must be divisible by 32.")
        ws = tuple(max(2, v // 32) for v in img)
        print(f"Synth cfg: img_size={img}, window_size={ws}, dwin_size={ws}")
        return ws


    @torch.no_grad()
    def _forward_flow(self, x, y):
        use_amp = torch.cuda.is_available()
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
            _, flow = self.model(x, y, return_all=False,
                                alpha_l1=self._val_alpha_l1,
                                alpha_l3=self._val_alpha_l3)
        return flow.float()


    def train_step(self, batch, epoch):
        args, ctx = self.args, self.ctx

        if self.is_synth:
            x, y, x_seg, y_seg = batch[0], batch[1], batch[2], batch[3]
            x_seg, y_seg = x_seg.to(self.device).long(), y_seg.to(self.device).long()
        else:
            x, y = (batch[0], batch[1]) if args.ds == "OASIS" else batch
            x_seg = y_seg = None

        x, y = x.to(self.device).float(), y.to(self.device).float()

        schedule_max_epoch = int(args.schedule_max_epoch) if int(getattr(args, "schedule_max_epoch", 0)) > 0 else int(args.max_epoch)
        alpha_l1, alpha_l3, warm = ctcf_schedule(epoch=int(epoch), max_epoch=schedule_max_epoch)
        if args.l1_from_start:
            alpha_l1 = 1.0
        if args.disable_l1:
            alpha_l1 = 0.0
        if args.disable_l3:
            alpha_l3 = 0.0
        self._val_alpha_l1 = alpha_l1
        self._val_alpha_l3 = alpha_l3
        W_icon = float(args.w_icon) * warm
        W_jac = float(args.w_jac) * warm

        def_xy, flow_xy = self.model(x, y, return_all=False, alpha_l1=alpha_l1, alpha_l3=alpha_l3)
        def_yx, flow_yx = self.model(y, x, return_all=False, alpha_l1=alpha_l1, alpha_l3=alpha_l3)

        with torch.autocast(device_type="cuda", enabled=False):
            L_ncc = 0.5 * (ctx.ncc(def_xy.float(), y.float()) + ctx.ncc(def_yx.float(), x.float())) * args.w_ncc

        L_icon = icon_loss(flow_xy, flow_yx, mode=args.icon_mode) * W_icon
        L_reg = 0.5 * (ctx.reg(flow_xy) + ctx.reg(flow_yx)) * args.w_reg
        L_jac = 0.5 * (neg_jacobian_penalty(flow_xy) + neg_jacobian_penalty(flow_yx)) * W_jac
        loss = L_ncc + L_icon + L_reg + L_jac

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

        if self.is_synth:
            with torch.no_grad():
                def_seg_xy = self.reg_nearest((x_seg.float(), flow_xy.detach().float()))
                logs["dice_tr"] = float(dice_val(def_seg_xy.long(), y_seg, num_clus=int(args.synth_num_labels)).item())

        return loss, logs


def parse_args():
    p = argparse.ArgumentParser()
    add_common_args(p, include_synth=True)
    p.set_defaults(exp="CTCF")

    p.add_argument("--config", type=str, default="CTCF-CascadeA", help="Model config key.")
    p.add_argument("--time_steps", type=int, default=6, help="Number of velocity integration steps.")
    p.add_argument("--schedule_max_epoch", type=int, default=0, help="If >0, uses this epoch horizon for CTCF stage schedule (alpha/warm), independent of --max_epoch.")
    p.add_argument("--use_checkpoint", type=int, choices=[0, 1], default=1, help="Enable gradient checkpointing in Swin blocks.")

    p.add_argument("--w_ncc", type=float, default=1.0, help="NCC similarity loss weight.")
    p.add_argument("--w_reg", type=float, default=None, help="Flow regularization loss weight (auto: IXI=4.0, others=1.0).")
    p.add_argument("--w_icon", type=float, default=0.05, help="ICON loss base weight.")
    p.add_argument("--w_jac", type=float, default=0.005, help="Negative Jacobian penalty base weight.")
    p.add_argument("--icon_mode", type=str, choices=["l1", "l2"], default="l1", help="ICON loss norm: l1 (default) or l2.")
    p.add_argument("--l1_from_start", type=int, choices=[0, 1], default=0, help="If 1, alpha_l1=1.0 from epoch 0 (skip schedule).")
    p.add_argument("--disable_l1", type=int, choices=[0, 1], default=0, help="If 1, force alpha_l1=0 (disable Level 1 coarse flow).")
    p.add_argument("--disable_l3", type=int, choices=[0, 1], default=0, help="If 1, force alpha_l3=0 (disable Level 3 refiner).")
    p.add_argument("--l1_base_ch", type=int, default=None, help="L1 coarse net base channels (default: config value, typically 16).")
    p.add_argument("--l3_base_ch", type=int, default=None, help="L3 refiner base channels (default: config value, typically 16).")
    p.add_argument("--l3_error_mode", type=str, choices=["absdiff", "gradmag", "ncc"], default=None, help="L3 error map mode.")
    p.add_argument("--prealign_encoder", type=int, choices=[0, 1], default=0, help="If 1, L2 encoder sees L1-warped mov instead of raw mov.")

    p.add_argument("--synth_train_samples", type=int, default=256, help="Number of synthetic training pairs.")
    p.add_argument("--synth_val_samples", type=int, default=32, help="Number of synthetic validation pairs.")
    p.add_argument("--synth_num_labels", type=int, default=36, help="Number of synthetic segmentation labels.")
    p.add_argument("--synth_vol_size", type=int, nargs=3, default=(96, 96, 96), help="Synthetic volume size D H W (each must be divisible by 32).")
    p.add_argument("--synth_flow_max_disp", type=float, default=6.0, help="Max synthetic displacement amplitude in voxels.")
    p.add_argument("--synth_seed", type=int, default=123, help="Base seed for synthetic pair generation.")
    return p.parse_args()


def main():
    args = parse_args()
    apply_ctcf_dataset_defaults(args)

    # IXI: flip axis 0 only (depth), matching UTSRMorph original protocol.
    # OASIS / other: flip all 3 axes.
    ixi_flip = (0,) if args.ds == "IXI" else (1, 2, 3)
    build_loaders = build_synth_loaders if args.ds == "SYNTH" else partial(loaders_baseline, ixi_flip_axes=ixi_flip)
    device = setup_device(gpu_id=int(args.gpu), seed=0, deterministic=False)
    runner = Runner(args, device)
    run_train(args=args, runner=runner, build_loaders=build_loaders)


if __name__ == "__main__":
    main()
