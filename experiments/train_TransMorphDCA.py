import argparse

import torch
from torch import optim

from utils import setup_device
from experiments.core.train_runtime import Ctx, add_common_args, run_train, loaders_baseline
from experiments.core.model_adapters import get_model_adapter


class Runner:
    def __init__(self, args, device):
        self.args, self.device = args, device
        self.adapter = get_model_adapter("tm-dca")

        self.model = self.adapter.build(time_steps=int(args.time_steps), config_key=args.config).to(device)
        half = tuple(int(v) for v in self.model.img_size)
        full = tuple(s * 2 for s in half)

        self.optimizer = optim.AdamW(self.model.parameters(), lr=args.lr, weight_decay=0, amsgrad=True)
        self.ctx = Ctx(device, vol_size=full, ncc_win=(9, 9, 9))
        self.forward_flow = lambda x, y: self.adapter.forward(self.model, x, y, amp=True, pool=2)


    def train_step(self, batch, epoch):
        args, ctx = self.args, self.ctx
        x, y = (batch[0], batch[1]) if args.ds == "OASIS" else batch
        x, y = x.to(self.device).float(), y.to(self.device).float()

        flow = self.adapter.forward(self.model, x, y, amp=True, pool=2)
        def_x = ctx.reg_model((x, flow))

        with torch.autocast(device_type="cuda", enabled=False):
            L_ncc = ctx.ncc(def_x.float(), y.float()) * args.w_ncc
        L_reg = ctx.reg(flow) * args.w_reg

        loss = L_ncc + L_reg
        return loss, {"all": loss.item(), "ncc": L_ncc.item(), "reg": L_reg.item()}


def parse_args():
    p = argparse.ArgumentParser()
    add_common_args(p)
    p.set_defaults(exp="TM_DCA")
    
    p.add_argument("--config", type=str, default="TransMorph-3-LVL", help="Model config key.")
    p.add_argument("--w_ncc", type=float, default=1.0, help="NCC similarity loss weight.")
    p.add_argument("--w_reg", type=float, default=1.0, help="Flow regularization loss weight.")
    p.add_argument("--time_steps", type=int, default=12, help="Number of velocity integration steps.")
    return p.parse_args()


def main():
    args = parse_args()
    device = setup_device(gpu_id=int(args.gpu), seed=0, deterministic=False)
    runner = Runner(args, device)
    run_train(args=args, runner=runner, build_loaders=loaders_baseline)


if __name__ == "__main__":
    main()
