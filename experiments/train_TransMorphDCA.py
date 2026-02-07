import argparse, torch
from torch import optim
import torch.nn.functional as F

from experiments.engine import add_engine_args, apply_paths, run_train, make_ctx, loaders_baseline, forward_flow_halfres
from models.TransMorph_DCA.model import TransMorphCascadeAd, CONFIGS as CONFIGS_TM


@torch.no_grad()
def forward_flow_fn(model, x, y, device, args, ctx):
    return forward_flow_halfres(model, x, y, pool=2, amp=True)


def train_step(model, batch, device, args, epoch, ctx):
    if args.ds == "OASIS": x, y, *_ = batch
    else: x, y = batch
    x, y = x.to(device).float(), y.to(device).float()

    flow = forward_flow_halfres(model, x, y, pool=2, amp=True)
    def_x = ctx.reg_model((x, flow))

    with torch.autocast(device_type="cuda", enabled=False):
        L_ncc = ctx.ncc(def_x.float(), y.float()) * args.w_ncc
    L_reg = ctx.reg(flow) * args.w_reg

    loss = L_ncc + L_reg
    return loss, {"all": loss.item(), "ncc": L_ncc.item(), "reg": L_reg.item()}


def build_model(device, args):
    D, H, W = args.vol_size
    cfg = CONFIGS_TM["TransMorph-3-LVL"]
    cfg.img_size = (D // 2, H // 2, W // 2)
    cfg.dwin_kernel_size = tuple(args.dwin)
    cfg.window_size = (D // 32, H // 32, W // 32)

    model = TransMorphCascadeAd(cfg, int(args.time_steps)).to(device)
    opt = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0, amsgrad=True)
    ctx = make_ctx(device, vol_size=args.vol_size, ncc_win=(9, 9, 9))
    return model, opt, ctx


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ds", choices=["OASIS", "IXI"], default="OASIS")
    add_engine_args(p, dataset="IXI")
    p.set_defaults(exp="TM_DCA")
    p.add_argument("--w_ncc", type=float, default=1.0)
    p.add_argument("--w_reg", type=float, default=1.0)
    p.add_argument("--time_steps", type=int, default=12)
    p.add_argument("--vol_size", type=int, nargs=3, default=[160, 192, 224])
    p.add_argument("--dwin", type=int, nargs=3, default=[7, 5, 3])
    return p.parse_args()


def main():
    args = parse_args()
    apply_paths(args, dataset=args.ds)
    if not args.exp: args.exp = "TM_DCA_OASIS" if args.ds == "OASIS" else "TM_DCA_IXI"
    run_train(dataset=args.ds, args=args, build_model=build_model, build_loaders=loaders_baseline, train_step=train_step, forward_flow_fn=forward_flow_fn)


if __name__ == "__main__":
    main()