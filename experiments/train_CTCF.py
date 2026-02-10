import argparse, torch
from torch import optim

from models.CTCF.model import CTCF_CascadeA, CONFIGS as CONFIGS_CTCF
from models.CTCF.controller import CTCFController, CTCFControllerCfg
from utils import icon_loss, neg_jacobian_penalty, setup_device
from experiments.engine import add_common_args, apply_paths, run_train, make_ctx, loaders_baseline


class Runner:
    def __init__(self, args, device):
        self.args, self.device = args, device

        cfg = CONFIGS_CTCF[args.config] 
        cfg.time_steps = int(args.time_steps)

        self.model = CTCF_CascadeA(cfg).to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=args.lr, amsgrad=True)
        self.ctx = make_ctx(device, vol_size=cfg.img_size, ncc_win=(9, 9, 9))

        ctrl_cfg = CTCFControllerCfg(
            fold_soft=float(args.fold_soft),
            fold_hard=float(args.fold_hard),
            alpha_l3_start=float(args.alpha_l3_start),
        )
        self.ctx.ctcf_ctrl = CTCFController(ctrl_cfg)


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
        x, y = (batch[0], batch[1]) if args.ds == "OASIS" else batch
        x, y = x.to(self.device).float(), y.to(self.device).float()

        ctrl = self.ctx.ctcf_ctrl
        w = ctrl.get()
        a1, a3 = float(w.alpha_l1), float(w.alpha_l3)
        
        wI = float(w.w_icon_mul)
        wC = float(w.w_cyc_mul)
        wJ = float(w.w_jac_mul)

        W_icon = float(args.w_icon) * wI
        W_cyc  = float(args.w_cyc)  * wC
        W_jac  = float(args.w_jac)  * wJ

        def_xy, flow_xy = self.model(x, y, return_all=False, alpha_l1=a1, alpha_l3=a3)
        def_yx, flow_yx = self.model(y, x, return_all=False, alpha_l1=a1, alpha_l3=a3)

        with torch.autocast(device_type="cuda", enabled=False):
            L_ncc = 0.5 * (ctx.ncc(def_xy.float(), y.float()) + ctx.ncc(def_yx.float(), x.float())) * args.w_ncc

        L_icon = icon_loss(flow_xy, flow_yx) * W_icon
        L_reg = 0.5 * (ctx.reg(flow_xy) + ctx.reg(flow_yx)) * args.w_reg
        L_jac = 0.5 * (neg_jacobian_penalty(flow_xy) + neg_jacobian_penalty(flow_yx)) * W_jac
        L_cyc = ((ctx.reg_model((def_xy, flow_yx)) - x).abs().mean() + (ctx.reg_model((def_yx, flow_xy)) - y).abs().mean()) * W_cyc

        loss = L_ncc + L_icon + L_reg + L_jac + L_cyc
        logs = {
            "all":  loss.item(),
            "ncc":  L_ncc.item(),
            "reg":  L_reg.item(),
            "icon": L_icon.item(),
            "cyc":  L_cyc.item(),
            "jac":  L_jac.item(),

            "phase": float({"S0": 0, "S1": 1, "S2": 2, "S3": 3}.get(ctrl.phase, -1)),
            "a3":    a3,
            "wI":    wI, "wC": wC, "wJ": wJ,
            "W_icon": W_icon, "W_cyc": W_cyc, "W_jac": W_jac,
        }
        return loss, logs


def parse_args():
    p = argparse.ArgumentParser()
    add_common_args(p)
    p.set_defaults(exp="CTCF")
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
    return p.parse_args()


def main():
    args = parse_args()
    apply_paths(args)
    device = setup_device(gpu_id=int(args.gpu), seed=0, deterministic=False)
    runner = Runner(args, device) # addressing own class
    run_train(args=args, runner=runner, build_loaders=loaders_baseline)


if __name__ == "__main__":
    main()