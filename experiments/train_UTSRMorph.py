import argparse, torch
from torch import optim

from experiments.engine import add_engine_args, apply_paths, run_train, make_ctx, oasis_loaders, ixi_loaders
from datasets import OASIS as oasis_ds
from datasets import IXI as ixi_ds

from models.UTSRMorph.model import CONFIGS as CONFIGS_UM, UTSRMorph


@torch.no_grad()
def forward_flow_fn(model, x, y, device, args, ctx):
    inp = torch.cat((x, y), dim=1)
    use_amp = torch.cuda.is_available()
    with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
        _, flow = model(inp)
    return flow


def train_step(model, batch, device, args, epoch, ctx):
    if args.ds == "OASIS": x, y, *_ = batch
    else: x, y = batch
    x, y = x.to(device).float(), y.to(device).float()

    inp = torch.cat((x, y), dim=1)
    use_amp = torch.cuda.is_available()
    with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
        def_x, flow = model(inp)

    with torch.autocast(device_type="cuda", enabled=False):
        L_ncc = ctx.ncc(def_x.float(), y.float()) * args.w_ncc
    L_reg = ctx.reg(flow) * args.w_reg

    loss = L_ncc + L_reg
    return loss, {"all": loss.item(), "ncc": L_ncc.item(), "reg": L_reg.item()}


def build_model(device, args):
    cfg = CONFIGS_UM["UTSRMorph-Large"]
    model = UTSRMorph(cfg).to(device)
    opt = optim.Adam(model.parameters(), lr=args.lr, amsgrad=True)
    ctx = make_ctx(device, vol_size=args.vol_size, ncc_win=(9, 9, 9))
    return model, opt, ctx


def build_loaders(args):
    if args.ds == "OASIS":
        return oasis_loaders(args, train_cls=oasis_ds.OASISBrainDataset, val_cls=oasis_ds.OASISBrainInferDataset, val_bs=1, drop_last_train=False, drop_last_val=True)
    return ixi_loaders(args, train_cls=ixi_ds.IXIBrainDataset, val_cls=ixi_ds.IXIBrainInferDataset, val_bs=1, drop_last_train=True, drop_last_val=True)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ds", choices=["OASIS", "IXI"], default="OASIS")
    add_engine_args(p, dataset="IXI")
    p.set_defaults(exp="UTSRMorph")
    p.add_argument("--w_ncc", type=float, default=1.0)
    p.add_argument("--w_reg", type=float, default=1.0)
    p.add_argument("--vol_size", type=int, nargs=3, default=[160, 192, 224])
    return p.parse_args()


def main():
    args = parse_args()
    apply_paths(args, dataset=args.ds)
    if not args.exp: args.exp = "UTSRMorph_OASIS" if args.ds == "OASIS" else "UTSRMorph_IXI"
    run_train(dataset=args.ds, args=args, build_model=build_model, build_loaders=build_loaders, train_step=train_step, forward_flow_fn=forward_flow_fn)


if __name__ == "__main__":
    main()