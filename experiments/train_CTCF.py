import argparse, torch
from torch import optim

from models.CTCF.model import CTCF_CascadeA, CONFIGS as CONFIGS_CTCF
from utils import icon_loss, neg_jacobian_penalty, ctcf_schedule, adjust_lr_ctcf_schedule
from experiments.engine import add_engine_args, apply_paths, run_train, make_ctx, loaders_baseline


@torch.no_grad()
def forward_flow_fn(model, x, y, device, args, ctx):
    use_amp = torch.cuda.is_available()
    with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
        _, flow = model(x, y)
    return flow


def train_step(model, batch, device, args, epoch, ctx):
    if args.ds == "OASIS": x, y, *_ = batch
    else: x, y = batch
    x, y = x.to(device).float(), y.to(device).float()

    a1, a3, warm = ctcf_schedule(epoch, args.max_epoch)
    W_icon, W_cyc, W_jac = args.w_icon * warm, args.w_cyc * warm, args.w_jac * warm

    def_xy, flow_xy = model(x, y, return_all=False, alpha_l1=a1, alpha_l3=a3)
    def_yx, flow_yx = model(y, x, return_all=False, alpha_l1=a1, alpha_l3=a3)

    with torch.autocast(device_type="cuda", enabled=False):
        L_ncc = 0.5 * (ctx.ncc(def_xy.float(), y.float()) + ctx.ncc(def_yx.float(), x.float())) * args.w_ncc

    L_icon = icon_loss(flow_xy, flow_yx) * W_icon
    L_reg = 0.5 * (ctx.reg(flow_xy) + ctx.reg(flow_yx)) * args.w_reg
    L_jac = 0.5 * (neg_jacobian_penalty(flow_xy) + neg_jacobian_penalty(flow_yx)) * W_jac
    L_cyc = ((ctx.reg_model((def_xy, flow_yx)) - x).abs().mean() + (ctx.reg_model((def_yx, flow_xy)) - y).abs().mean()) * W_cyc

    loss = L_ncc + L_icon + L_reg + L_jac + L_cyc
    return loss, {"all": loss.item(), "ncc": L_ncc.item(), "reg": L_reg.item(), "icon": L_icon.item(), "cyc": L_cyc.item(), "jac": L_jac.item()}


def build_model(device, args):
    cfg = CONFIGS_CTCF[args.ctcf_config]; cfg.time_steps = int(args.time_steps)
    model = CTCF_CascadeA(cfg).to(device)
    opt = optim.Adam(model.parameters(), lr=args.lr, amsgrad=True)
    vol_size = cfg.img_size if args.ds == "OASIS" else args.vol_size
    ctx = make_ctx(device, vol_size=vol_size, ncc_win=(9, 9, 9))
    return model, opt, ctx


def lr_step(optimizer, epoch, args):
    return adjust_lr_ctcf_schedule(optimizer, epoch, args.max_epoch, args.lr)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ds", choices=["OASIS", "IXI"], default="OASIS")
    add_engine_args(p, dataset="IXI")
    p.set_defaults(exp="CTCF")
    p.add_argument("--ctcf_config", type=str, default="CTCF-CascadeA")
    p.add_argument("--w_ncc", type=float, default=1.0)
    p.add_argument("--w_reg", type=float, default=1.0)
    p.add_argument("--w_icon", type=float, default=0.05)
    p.add_argument("--w_cyc", type=float, default=0.02)
    p.add_argument("--w_jac", type=float, default=0.005)
    p.add_argument("--time_steps", type=int, default=12)
    p.add_argument("--vol_size", type=int, nargs=3, default=[160, 192, 224])
    return p.parse_args()


def main():
    args = parse_args()
    apply_paths(args, dataset=args.ds)
    if not args.exp: args.exp = "CTCF" if args.ds == "OASIS" else "CTCF_IXI"
    run_train(dataset=args.ds, args=args, build_model=build_model, build_loaders=loaders_baseline, train_step=train_step, forward_flow_fn=forward_flow_fn, lr_step=lr_step)


if __name__ == "__main__":
    main()