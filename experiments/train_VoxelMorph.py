import argparse
from functools import partial

import torch
from torch import optim

from utils import setup_device
from experiments.core.train_runtime import Ctx, add_common_args, run_train, loaders_baseline
from models.VoxelMorph.wrapper import VxmDense
from models.VoxelMorph.configs import CONFIGS


IMG_SIZE = (160, 192, 224)


class Runner:
    def __init__(self, args, device):
        self.args, self.device = args, device

        cfg = CONFIGS[args.config]
        self.model = VxmDense(
            vol_size=IMG_SIZE,
            enc_nf=cfg["enc_nf"],
            dec_nf=cfg["dec_nf"],
            int_steps=cfg["int_steps"],
        ).to(device)

        n_params = sum(p.numel() for p in self.model.parameters())
        print(f"VxmDense [{args.config}] params: {n_params:,} ({n_params/1e6:.3f}M)")

        self.optimizer = optim.Adam(self.model.parameters(), lr=args.lr, amsgrad=True)
        self.ctx = Ctx(device, vol_size=IMG_SIZE, ncc_win=(9, 9, 9))
        self.forward_flow = self._forward_flow


    @torch.no_grad()
    def _forward_flow(self, x, y):
        _, flow = self.model(x, y)
        return flow


    def train_step(self, batch, epoch):
        args, ctx = self.args, self.ctx
        x, y = (batch[0], batch[1]) if args.ds == "OASIS" else batch
        x, y = x.to(self.device).float(), y.to(self.device).float()

        def_x, flow = self.model(x, y)

        with torch.autocast(device_type="cuda", enabled=False):
            L_ncc = ctx.ncc(def_x.float(), y.float()) * args.w_ncc
        L_reg = ctx.reg(flow) * args.w_reg
        loss = L_ncc + L_reg

        return loss, {"all": loss.item(), "ncc": L_ncc.item(), "reg": L_reg.item()}


def parse_args():
    p = argparse.ArgumentParser()
    add_common_args(p)
    p.set_defaults(exp="VoxelMorph")

    p.add_argument("--config", type=str, default="VxmDense",
                    choices=list(CONFIGS.keys()), help="VoxelMorph config key.")
    p.add_argument("--w_ncc", type=float, default=1.0, help="NCC similarity loss weight.")
    p.add_argument("--w_reg", type=float, default=1.0, help="Flow regularization loss weight.")
    return p.parse_args()


def main():
    args = parse_args()
    device = setup_device(gpu_id=int(args.gpu), seed=0, deterministic=False)
    runner = Runner(args, device)
    ixi_flip = (0,) if args.ds == "IXI" else (1, 2, 3)
    run_train(args=args, runner=runner, build_loaders=partial(loaders_baseline, ixi_flip_axes=ixi_flip))


if __name__ == "__main__":
    main()
