from __future__ import annotations

import argparse

import torch
from torch import optim

from experiments.core.cli_args import add_common_args
from experiments.core.data_loaders import baseline_loader_builder
from experiments.core.train_runtime import TrainContext, run_train
from models.MambaMorph.configs import CONFIGS
from models.MambaMorph.wrapper import MambaMorphSolo
from utils import setup_device


class Runner:
    def __init__(self, args, device):
        self.args, self.device = args, device
        self.img_size = tuple(args.img_size)
        diffeo = bool(args.diffeo)

        self.model = MambaMorphSolo(
            config_key=args.config,
            diffeomorphic=diffeo,
            img_size=self.img_size,
        ).to(device)
        n_params = sum(p.numel() for p in self.model.parameters())
        print(f"MambaMorphSolo [{args.config}, diffeo={diffeo}] params: {n_params:,} ({n_params / 1e6:.3f}M)")

        self.optimizer = optim.Adam(self.model.parameters(), lr=args.lr, amsgrad=True)
        self.ctx = TrainContext(device, vol_size=self.img_size, ncc_win=(9, 9, 9))
        self.forward_flow = self._forward_flow

    @torch.no_grad()
    def _forward_flow(self, x, y):
        _, flow = self.model(x, y)
        return flow

    def train_step(self, batch, epoch):
        args, ctx = self.args, self.ctx
        x, y = batch[0], batch[1]
        x, y = x.to(self.device).float(), y.to(self.device).float()

        out = self.model.model(x, y)
        warped = out["moved_vol"]
        reg_target = out["preint_flow"]

        with torch.autocast(device_type="cuda", enabled=False):
            L_ncc = ctx.ncc(y.float(), warped.float()) * args.w_ncc
        L_reg = ctx.reg(reg_target) * args.w_reg
        loss = L_ncc + L_reg

        return loss, {"all": loss.item(), "ncc": L_ncc.item(), "reg": L_reg.item()}


def parse_args():
    p = argparse.ArgumentParser()
    add_common_args(p)
    p.set_defaults(exp="MambaMorph")

    p.add_argument(
        "--config",
        type=str,
        default="MambaMorph",
        choices=list(CONFIGS.keys()),
        help="MambaMorph config key.",
    )
    p.add_argument(
        "--diffeo",
        type=int,
        choices=[0, 1],
        default=1,
        help=("1 = diffeomorphic (VecInt-integrated SVF, paper default); 0 = direct displacement (MambaMorphOri)."),
    )
    p.add_argument(
        "--w_ncc",
        type=float,
        default=1.0,
        help="NCC similarity loss weight.",
    )
    p.add_argument(
        "--w_reg",
        type=float,
        default=1.0,
        help="Diffusion regularization weight (applied to velocity).",
    )
    return p.parse_args()


def main():
    args = parse_args()
    device = setup_device(gpu_id=args.gpu, seed=0, deterministic=False)
    runner = Runner(args, device)
    run_train(args=args, runner=runner, build_loaders=baseline_loader_builder(args))


if __name__ == "__main__":
    main()
