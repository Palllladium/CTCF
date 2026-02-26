import argparse

import torch
from torch import optim

from utils import setup_device
from experiments.core.train_runtime import Ctx, add_common_args, run_train, loaders_baseline
from experiments.core.model_adapters import get_model_adapter


class Runner:
    def __init__(self, args, device):
        self.args, self.device = args, device
        self.adapter = get_model_adapter("utsrmorph")

        self.model = self.adapter.build(config_key=args.config).to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=args.lr, amsgrad=True)
        self.ctx = Ctx(device, vol_size=self.model.cfg.img_size, ncc_win=(9, 9, 9))
        self.forward_flow = lambda x, y: self.adapter.forward(self.model, x, y, amp=True)


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
    
    p.add_argument("--config", type=str, default="UTSRMorph-Large", help="Model config key.")
    p.add_argument("--w_ncc", type=float, default=1.0, help="NCC similarity loss weight.")
    p.add_argument("--w_reg", type=float, default=1.0, help="Flow regularization loss weight.")
    return p.parse_args()


def main():
    args = parse_args()
    device = setup_device(gpu_id=int(args.gpu), seed=0, deterministic=False)
    runner = Runner(args, device)
    run_train(args=args, runner=runner, build_loaders=loaders_baseline)


if __name__ == "__main__":
    main()
