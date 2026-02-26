import argparse

import torch
from torch import optim

from datasets.synthetic import build_synth_loaders
from experiments.core.train_runtime import Ctx, add_common_args, loaders_baseline, run_train
from experiments.core.model_adapters import get_model_adapter
from models.CTCF.controller import CTCFController, CTCFControllerCfg
from utils import RegisterModel, dice_val, icon_loss, neg_jacobian_penalty, setup_device

PHASE_TO_ID = {"S0": 0.0, "S1": 1.0, "S2": 2.0, "S3": 3.0}


class Runner:
    def __init__(self, args, device):
        self.args = args
        self.device = device
        self.is_synth = args.ds == "SYNTH"
        self.adapter = get_model_adapter("ctcf")

        synth_img_size = None
        synth_dwin = None
        use_level1 = None
        use_level3 = None

        if self.is_synth:
            synth_img_size = tuple(int(v) for v in args.synth_vol_size)
            synth_dwin = self._adapt_swin_windows(synth_img_size)
            use_level1 = bool(int(args.use_level1))
            use_level3 = bool(int(args.use_level3))

        self.model = self.adapter.build(
            time_steps=int(args.time_steps),
            config_key=args.config,
            use_level1=use_level1,
            use_level3=use_level3,
            use_checkpoint=bool(int(args.use_checkpoint)),
            synth_img_size=synth_img_size,
            synth_dwin=synth_dwin,
        ).to(device)
        
        if self.is_synth:
            print(
                f"Synth architecture: use_level1={self.model.use_level1}, "
                f"use_level2={self.model.use_level2}, use_level3={self.model.use_level3}"
            )

        self.optimizer = optim.Adam(self.model.parameters(), lr=args.lr, amsgrad=True)
        self.img_size = tuple(int(v) for v in self.model.img_size_full)
        self.ctx = Ctx(device, vol_size=self.img_size, ncc_win=(9, 9, 9))
        self.reg_nearest = RegisterModel(self.img_size, mode="nearest").to(device) if self.is_synth else None

        ctrl_cfg = CTCFControllerCfg()
        self.ctx.ctcf_ctrl = CTCFController(ctrl_cfg)
        self.forward_flow = self._forward_flow


    @staticmethod
    def _adapt_swin_windows(img):
        """Compute Swin window sizes from full-resolution synthetic volume size."""
        img = tuple(int(v) for v in img)
        if any(v % 32 != 0 for v in img):
            raise ValueError(
                f"synth_vol_size={img} must be divisible by 32. Example: 96 96 96 or 160 192 224."
            )
        ws = tuple(max(2, v // 32) for v in img)
        print(f"Synth cfg: img_size={img}, window_size={ws}, dwin_size={ws}")
        return ws


    @torch.no_grad()
    def _forward_flow(self, x, y):
        a3 = float(self.ctx.ctcf_ctrl.get().alpha_l3)
        use_amp = torch.cuda.is_available()
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
            _, flow = self.model(x, y, return_all=False, alpha_l1=1.0, alpha_l3=a3)
        return flow.float()


    def train_step(self, batch, epoch):
        """Compute bidirectional CTCF losses for one optimization step."""
        args, ctx = self.args, self.ctx

        if self.is_synth:
            x, y, x_seg, y_seg = batch[0], batch[1], batch[2], batch[3]
            x_seg, y_seg = x_seg.to(self.device).long(), y_seg.to(self.device).long()
        else:
            x, y = (batch[0], batch[1]) if args.ds == "OASIS" else batch
            x_seg = y_seg = None

        x, y = x.to(self.device).float(), y.to(self.device).float()

        ctrl = ctx.ctcf_ctrl
        w = ctrl.get()
        a3 = float(w.alpha_l3)

        wI, wC, wJ = float(w.w_icon_mul), float(w.w_cyc_mul), float(w.w_jac_mul)
        W_icon = float(args.w_icon) * wI
        W_cyc = float(args.w_cyc) * wC
        W_jac = float(args.w_jac) * wJ

        def_xy, flow_xy = self.model(x, y, return_all=False, alpha_l1=1.0, alpha_l3=a3)
        def_yx, flow_yx = self.model(y, x, return_all=False, alpha_l1=1.0, alpha_l3=a3)

        with torch.autocast(device_type="cuda", enabled=False):
            L_ncc = 0.5 * (ctx.ncc(def_xy.float(), y.float()) + ctx.ncc(def_yx.float(), x.float())) * args.w_ncc

        L_icon = icon_loss(flow_xy, flow_yx) * W_icon
        L_reg = 0.5 * (ctx.reg(flow_xy) + ctx.reg(flow_yx)) * args.w_reg
        L_jac = 0.5 * (neg_jacobian_penalty(flow_xy) + neg_jacobian_penalty(flow_yx)) * W_jac
        L_cyc = (
            (ctx.reg_model((def_xy, flow_yx)) - x).abs().mean()
            + (ctx.reg_model((def_yx, flow_xy)) - y).abs().mean()
        ) * W_cyc
        loss = L_ncc + L_icon + L_reg + L_jac + L_cyc

        phase_id = float(PHASE_TO_ID.get(ctrl.phase, -1.0))
        logs = {
            "all": loss.item(),
            "ncc": L_ncc.item(),
            "reg": L_reg.item(),
            "icon": L_icon.item(),
            "cyc": L_cyc.item(),
            "jac": L_jac.item(),
            "phase": phase_id,
            "a3": a3,
            "wI": wI,
            "wC": wC,
            "wJ": wJ,
            "W_icon": W_icon,
            "W_cyc": W_cyc,
            "W_jac": W_jac,
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
    p.add_argument("--w_reg", type=float, default=1.0, help="Flow regularization loss weight.")
    p.add_argument("--w_icon", type=float, default=0.05, help="ICON loss base weight (multiplied by controller knob).")
    p.add_argument("--w_cyc", type=float, default=0.02, help="Cycle consistency loss base weight (multiplied by controller knob).")
    p.add_argument("--w_jac", type=float, default=0.005, help="Negative Jacobian penalty base weight (multiplied by controller knob).")

    p.add_argument("--synth_train_samples", type=int, default=256, help="Number of synthetic training pairs.")
    p.add_argument("--synth_val_samples", type=int, default=32, help="Number of synthetic validation pairs.")
    p.add_argument("--synth_num_labels", type=int, default=36, help="Number of synthetic segmentation labels.")
    p.add_argument("--synth_vol_size", type=int, nargs=3, default=(96, 96, 96), help="Synthetic volume size D H W (each must be divisible by 32).")
    p.add_argument("--synth_flow_max_disp", type=float, default=6.0, help="Max synthetic displacement amplitude in voxels.")
    p.add_argument("--synth_seed", type=int, default=123, help="Base seed for synthetic pair generation.")
    p.add_argument("--use_level1", type=int, choices=[0, 1], default=1, help="Enable Level-1 coarse path (SYNTH mode only).")
    p.add_argument("--use_level3", type=int, choices=[0, 1], default=1, help="Enable Level-3 refiner path (SYNTH mode only).")
    return p.parse_args()


def main():
    args = parse_args()
    build_loaders = build_synth_loaders if args.ds == "SYNTH" else loaders_baseline
    device = setup_device(gpu_id=int(args.gpu), seed=0, deterministic=False)
    runner = Runner(args, device)
    run_train(args=args, runner=runner, build_loaders=build_loaders)


if __name__ == "__main__":
    main()
