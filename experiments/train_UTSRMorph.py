import argparse, torch
from torch import optim

from utils import setup_device
from models.UTSRMorph.model import CONFIGS as CONFIGS_UM, UTSRMorph
from experiments.engine import add_common_args, apply_paths, run_train, make_ctx, loaders_baseline


class Runner:
    def __init__(self, args, device):
        self.args, self.device = args, device
        
        cfg = CONFIGS_UM[args.config]
        self.model = UTSRMorph(cfg).to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=args.lr, amsgrad=True)
        self.ctx = make_ctx(device, vol_size=cfg.img_size, ncc_win=(9, 9, 9))


    @torch.no_grad()
    def forward_flow(self, x, y):
        inp = torch.cat((x, y), dim=1)
        use_amp = torch.cuda.is_available()
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
            _, flow = self.model(inp)
        return flow


    def train_step(self, batch, epoch):
        args, ctx = self.args, self.ctx
        x, y = (batch[0], batch[1]) if args.ds == "OASIS" else batch
        x, y = x.to(self.device).float(), y.to(self.device).float()

        inp = torch.cat((x, y), dim=1)
        use_amp = torch.cuda.is_available()
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
            def_x, flow = self.model(inp)

        with torch.autocast(device_type="cuda", enabled=False):
            L_ncc = ctx.ncc(def_x.float(), y.float()) * args.w_ncc
        L_reg = ctx.reg(flow) * args.w_reg

        loss = L_ncc + L_reg
        return loss, {"all": loss.item(), "ncc": L_ncc.item(), "reg": L_reg.item()}


def parse_args():
    p = argparse.ArgumentParser()
    add_common_args(p)
    p.set_defaults(exp="UTSRMorph")
    p.add_argument("--config", type=str, default="UTSRMorph-Large")
    p.add_argument("--w_ncc", type=float, default=1.0)
    p.add_argument("--w_reg", type=float, default=1.0)
    return p.parse_args()


def main():
    args = parse_args()
    apply_paths(args)
    device = setup_device(gpu_id=int(args.gpu), seed=0, deterministic=False)
    runner = Runner(args, device) # addressing own class
    run_train(args=args, runner=runner, build_loaders=loaders_baseline)


if __name__ == "__main__":
    main()