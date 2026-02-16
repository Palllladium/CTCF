import argparse
import torch
from torch import optim

from models.CTCF.model import CTCF_CascadeA, CONFIGS as CONFIGS_CTCF
from models.CTCF.controller import CTCFController, CTCFControllerCfg
from datasets.synthetic import build_synth_loaders
from utils import icon_loss, neg_jacobian_penalty, setup_device, register_model, dice_val
from experiments.engine import add_common_args, run_train, make_ctx


class Runner:
    def __init__(self, args, device):
        self.args, self.device = args, device

        cfg = CONFIGS_CTCF[args.config]
        cfg.time_steps = int(args.time_steps)
        cfg.img_size = tuple(int(v) for v in args.synth_vol_size)
        cfg.use_level1 = bool(int(args.use_level1))
        cfg.use_level3 = bool(int(args.use_level3))
        self._adapt_swin_windows(cfg)

        self.model = CTCF_CascadeA(cfg).to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=args.lr, amsgrad=True)
        self.ctx = make_ctx(device, vol_size=cfg.img_size, ncc_win=(9, 9, 9))
        self.reg_nearest = register_model(tuple(cfg.img_size), mode="nearest").to(device)
        print(f"Synth architecture: use_level1={cfg.use_level1}, use_level2={cfg.use_level2}, use_level3={cfg.use_level3}")

        ctrl_cfg = CTCFControllerCfg(
            fold_soft=float(args.fold_soft),
            fold_hard=float(args.fold_hard),
            alpha_l3_start=float(args.alpha_l3_start),
        )
        self.ctx.ctcf_ctrl = CTCFController(ctrl_cfg)

    @staticmethod
    def _adapt_swin_windows(cfg):
        """
        Make Swin windows compatible with synthetic volume.

        For this model, level-2 works at half resolution and uses patch_size=4,
        so token resolution is full_size / 8.
        We set window_size to token/4 (stage-3 resolution), i.e. full_size/32.
        """
        img = tuple(int(v) for v in cfg.img_size)
        if any(v % 32 != 0 for v in img):
            raise ValueError(
                f"synth_vol_size={img} must be divisible by 32 for this architecture. "
                f"Example: 96 96 96 or 160 192 224."
            )

        ws = tuple(max(2, v // 32) for v in img)
        cfg.window_size = ws
        cfg.dwin_size = ws
        print(f"Synth cfg: img_size={cfg.img_size}, window_size={cfg.window_size}, dwin_size={cfg.dwin_size}")

    @torch.no_grad()
    def forward_flow(self, x, y):
        ctrl = getattr(self.ctx, "ctcf_ctrl", None)
        w = ctrl.get()
        a1, a3 = float(w.alpha_l1), float(w.alpha_l3)

        use_amp = torch.cuda.is_available()
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
            _, flow = self.model(x, y, return_all=False, alpha_l1=a1, alpha_l3=a3)
        return flow

    def train_step(self, batch, epoch):
        args, ctx = self.args, self.ctx
        x, y, x_seg, y_seg = batch[0], batch[1], batch[2], batch[3]
        x, y = x.to(self.device).float(), y.to(self.device).float()
        x_seg, y_seg = x_seg.to(self.device).long(), y_seg.to(self.device).long()

        ctrl = self.ctx.ctcf_ctrl
        w = ctrl.get()
        a1, a3 = float(w.alpha_l1), float(w.alpha_l3)

        wI = float(w.w_icon_mul)
        wC = float(w.w_cyc_mul)
        wJ = float(w.w_jac_mul)

        W_icon = float(args.w_icon) * wI
        W_cyc = float(args.w_cyc) * wC
        W_jac = float(args.w_jac) * wJ

        def_xy, flow_xy = self.model(x, y, return_all=False, alpha_l1=a1, alpha_l3=a3)
        def_yx, flow_yx = self.model(y, x, return_all=False, alpha_l1=a1, alpha_l3=a3)

        with torch.autocast(device_type="cuda", enabled=False):
            L_ncc = 0.5 * (ctx.ncc(def_xy.float(), y.float()) + ctx.ncc(def_yx.float(), x.float())) * args.w_ncc

        L_icon = icon_loss(flow_xy, flow_yx) * W_icon
        L_reg = 0.5 * (ctx.reg(flow_xy) + ctx.reg(flow_yx)) * args.w_reg
        L_jac = 0.5 * (neg_jacobian_penalty(flow_xy) + neg_jacobian_penalty(flow_yx)) * W_jac
        L_cyc = (
            (ctx.reg_model((def_xy, flow_yx)) - x).abs().mean()
            + (ctx.reg_model((def_yx, flow_xy)) - y).abs().mean()
        ) * W_cyc

        with torch.no_grad():
            def_seg_xy = self.reg_nearest((x_seg.float(), flow_xy.detach().float()))
            dice_tr = dice_val(def_seg_xy.long(), y_seg, num_clus=int(args.synth_num_labels))

        loss = L_ncc + L_icon + L_reg + L_jac + L_cyc
        logs = {
            "all": loss.item(),
            "ncc": L_ncc.item(),
            "reg": L_reg.item(),
            "icon": L_icon.item(),
            "cyc": L_cyc.item(),
            "jac": L_jac.item(),
            "dice_tr": float(dice_tr.item()),
            "phase": float({"S0": 0, "S1": 1, "S2": 2, "S3": 3}.get(ctrl.phase, -1)),
            "a3": a3,
            "wI": wI,
            "wC": wC,
            "wJ": wJ,
            "W_icon": W_icon,
            "W_cyc": W_cyc,
            "W_jac": W_jac,
        }
        return loss, logs


def parse_args():
    p = argparse.ArgumentParser()
    add_common_args(p)
    p.set_defaults(
        exp="CTCF_SYNTH",
        ds="OASIS",
        num_workers=0,
        batch_size=1,
        max_epoch=80,
        tb_images_every=0,
        max_train_iters=64,
        max_val_batches=8,
    )

    p.add_argument("--config", type=str, default="CTCF-CascadeA")
    p.add_argument("--w_ncc", type=float, default=1.0)
    p.add_argument("--w_reg", type=float, default=1.0)
    p.add_argument("--w_icon", type=float, default=0.05)
    p.add_argument("--w_cyc", type=float, default=0.02)
    p.add_argument("--w_jac", type=float, default=0.005)
    p.add_argument("--fold_soft", type=float, default=5.0)
    p.add_argument("--fold_hard", type=float, default=10.0)
    p.add_argument("--alpha_l3_start", type=float, default=0.10)
    p.add_argument("--time_steps", type=int, default=12)

    p.add_argument("--synth_train_samples", type=int, default=256)
    p.add_argument("--synth_val_samples", type=int, default=32)
    p.add_argument("--synth_num_labels", type=int, default=36)
    p.add_argument("--synth_vol_size", type=int, nargs=3, default=(96, 96, 96))
    p.add_argument("--synth_flow_max_disp", type=float, default=6.0)
    p.add_argument("--synth_seed", type=int, default=123)
    p.add_argument("--use_level1", type=int, choices=[0, 1], default=1, help="Enable/disable L1 coarse branch.")
    p.add_argument("--use_level3", type=int, choices=[0, 1], default=1, help="Enable/disable L3 refinement branch.")
    p.add_argument("--max_train_iters", type=int, default=64, help="Limit train iterations per epoch; <=0 disables limit.")
    p.add_argument("--max_val_batches", type=int, default=8, help="Limit validation batches per epoch; <=0 disables limit.")
    return p.parse_args()


def main():
    args = parse_args()
    device = setup_device(gpu_id=int(args.gpu), seed=0, deterministic=False)
    runner = Runner(args, device)
    run_train(args=args, runner=runner, build_loaders=build_synth_loaders)


if __name__ == "__main__":
    main()