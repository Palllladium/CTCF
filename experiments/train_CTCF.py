import argparse

import torch
from torch import optim

from datasets.synthetic import build_synth_loaders
from experiments.core.model_adapters import get_model_adapter
from experiments.core.train_rules import CtcfTrainRules, apply_ctcf_dataset_defaults
from experiments.core.train_runtime import Ctx, add_common_args, loaders_baseline, run_train
from models.CTCF.blocks import upsample_flow
from utils import RegisterModel, dice_val, icon_loss, neg_jacobian_penalty, setup_device


class Runner:
    """CTCF trainer with deterministic schedules and jacobian governor (no controller phases)."""
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
        ).to(device)

        self.optimizer = optim.Adam(self.model.parameters(), lr=args.lr, amsgrad=True)
        self.img_size = tuple(int(v) for v in self.model.img_size_full)
        self.ctx = Ctx(device, vol_size=self.img_size, ncc_win=(9, 9, 9))
        self.reg_nearest = RegisterModel(self.img_size, mode="nearest").to(device) if self.is_synth else None
        self.forward_flow = self._forward_flow
        self.lr_policy = "ctcf"
        self.rules = CtcfTrainRules(ds=args.ds, max_epoch=args.max_epoch)


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
            _, flow = self.model(x, y, return_all=False, alpha_l1=1.0)
        return flow.float()


    def on_val_end(self, epoch: int, _val_dice: float, val_jac_percent: float):
        del epoch, _val_dice
        self.rules.update_from_val(val_jac_percent)


    def train_step(self, batch, epoch):
        args, ctx = self.args, self.ctx
        self.rules.on_epoch_start(int(epoch))

        if self.is_synth:
            x, y, x_seg, y_seg = batch[0], batch[1], batch[2], batch[3]
            x_seg, y_seg = x_seg.to(self.device).long(), y_seg.to(self.device).long()
        else:
            x, y = (batch[0], batch[1]) if args.ds == "OASIS" else batch
            x_seg = y_seg = None

        x, y = x.to(self.device).float(), y.to(self.device).float()

        W_icon = float(args.w_icon) * float(self.rules.w_icon_mul)
        W_cyc = float(args.w_cyc) * float(self.rules.w_cyc_mul)
        W_jac = float(args.w_jac) * float(self.rules.w_jac_mul)

        def_xy, flow_xy, aux_xy = self.model(x, y, return_all=True, alpha_l1=1.0)
        def_yx, flow_yx, aux_yx = self.model(y, x, return_all=True, alpha_l1=1.0)

        with torch.autocast(device_type="cuda", enabled=False):
            L_ncc = 0.5 * (ctx.ncc(def_xy.float(), y.float()) + ctx.ncc(def_yx.float(), x.float())) * args.w_ncc

        L_icon = icon_loss(flow_xy, flow_yx) * W_icon
        L_reg = 0.5 * (ctx.reg(flow_xy) + ctx.reg(flow_yx)) * args.w_reg
        L_jac = 0.5 * (neg_jacobian_penalty(flow_xy, crop=1) + neg_jacobian_penalty(flow_yx, crop=1)) * W_jac
        L_cyc = ((ctx.reg_model((def_xy, flow_yx)) - x).abs().mean() + (ctx.reg_model((def_yx, flow_xy)) - y).abs().mean()) * W_cyc
        L_main = L_ncc + L_icon + L_reg + L_jac + L_cyc

        L_reg_l1 = x.new_tensor(0.0)
        if "flow_quarter" in aux_xy and "flow_quarter" in aux_yx:
            flow_l1_xy = upsample_flow(aux_xy["flow_quarter"], scale_factor=4)
            flow_l1_yx = upsample_flow(aux_yx["flow_quarter"], scale_factor=4)
            L_reg_l1 = 0.5 * (ctx.reg(flow_l1_xy) + ctx.reg(flow_l1_yx)) * args.w_reg

        L_reg_l2 = x.new_tensor(0.0)
        if "flow_half_l2" in aux_xy and "flow_half_l2" in aux_yx:
            flow_l2_xy = upsample_flow(aux_xy["flow_half_l2"], scale_factor=2)
            flow_l2_yx = upsample_flow(aux_yx["flow_half_l2"], scale_factor=2)
            L_reg_l2 = 0.5 * (ctx.reg(flow_l2_xy) + ctx.reg(flow_l2_yx)) * args.w_reg

        lam_l1, lam_l2 = self.rules.aux_lambdas(int(epoch))
        L_aux = L_reg_l1 * lam_l1 + L_reg_l2 * lam_l2
        loss = L_main + L_aux

        logs = {
            "all": loss.item(),
            "main": L_main.item(),
            "ncc": L_ncc.item(),
            "reg": L_reg.item(),
            "icon": L_icon.item(),
            "cyc": L_cyc.item(),
            "jac": L_jac.item(),
            "aux": L_aux.item(),
            "aux_l1": L_reg_l1.item(),
            "aux_l2": L_reg_l2.item(),
            "lam_l1": lam_l1,
            "lam_l2": lam_l2,
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
    p.add_argument("--time_steps", type=int, default=12, help="Number of velocity integration steps.")
    p.add_argument("--use_checkpoint", type=int, choices=[0, 1], default=1, help="Enable gradient checkpointing in Swin blocks.")

    p.add_argument("--w_ncc", type=float, default=1.0, help="NCC similarity loss weight.")
    p.add_argument("--w_reg", type=float, default=None, help="Flow regularization loss weight (auto: IXI=4.0, others=1.0).")
    p.add_argument("--w_icon", type=float, default=0.05, help="ICON loss base weight.")
    p.add_argument("--w_cyc", type=float, default=0.02, help="Cycle consistency loss base weight.")
    p.add_argument("--w_jac", type=float, default=0.005, help="Negative Jacobian penalty base weight.")

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

    build_loaders = build_synth_loaders if args.ds == "SYNTH" else loaders_baseline
    device = setup_device(gpu_id=int(args.gpu), seed=0, deterministic=False)
    runner = Runner(args, device)
    run_train(args=args, runner=runner, build_loaders=build_loaders)


if __name__ == "__main__":
    main()
