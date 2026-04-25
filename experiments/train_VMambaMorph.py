"""
Standalone VMambaMorph training (Wang et al., 2024; arxiv 2404.05105).

Loss configuration mirrors MambaMorph (NCC + diffusion regularization on the
pre-integration velocity field). The architecture difference between the
two is the encoder block (single-direction Mamba scan vs 4-direction SS2D
cross-scan); the solo training pipeline is identical.

Requires `mamba_ssm` (provides `selective_scan_fn`) and `einops`.
See tools/install_mamba.sh for installation.

Usage:
    python -m experiments.train_VMambaMorph --ds OASIS --2 --max_epoch 100
"""
import argparse
from functools import partial

import torch
from torch import optim

from utils import setup_device, Grad3d
from experiments.core.train_runtime import Ctx, add_common_args, run_train, loaders_baseline
from models.VMambaMorph.wrapper import VMambaMorphSolo
from models.VMambaMorph.configs import CONFIGS


IMG_SIZE = (160, 192, 224)


class Runner:
    def __init__(self, args, device):
        self.args, self.device = args, device

        self.model = VMambaMorphSolo(config_key=args.config).to(device)
        n_params = sum(p.numel() for p in self.model.parameters())
        print(f"VMambaMorphSolo [{args.config}] params: {n_params:,} ({n_params/1e6:.3f}M)")

        self.optimizer = optim.Adam(self.model.parameters(), lr=args.lr, amsgrad=True)
        self.ctx = Ctx(device, vol_size=IMG_SIZE, ncc_win=(9, 9, 9))
        self.forward_flow = self._forward_flow
        self._reg_l2 = Grad3d(penalty="l2")

    @torch.no_grad()
    def _forward_flow(self, x, y):
        _, flow = self.model(x, y)
        return flow

    def train_step(self, batch, epoch):
        args, ctx = self.args, self.ctx
        x, y = (batch[0], batch[1]) if args.ds == "OASIS" else batch
        x, y = x.to(self.device).float(), y.to(self.device).float()

        out = self.model.model(x, y)
        warped = out["moved_vol"]
        reg_target = out["preint_flow"]

        with torch.autocast(device_type="cuda", enabled=False):
            L_ncc = ctx.ncc(y.float(), warped.float()) * args.w_ncc
        L_reg = self._reg_l2(reg_target) * args.w_reg
        loss = L_ncc + L_reg

        return loss, {"all": loss.item(), "ncc": L_ncc.item(), "reg": L_reg.item()}


def parse_args():
    p = argparse.ArgumentParser()
    add_common_args(p)
    p.set_defaults(exp="VMambaMorph")

    p.add_argument("--config", type=str, default="VMambaMorph",
                    choices=list(CONFIGS.keys()), help="VMambaMorph config key.")
    p.add_argument("--w_ncc", type=float, default=1.0, help="NCC similarity loss weight.")
    p.add_argument("--w_reg", type=float, default=1.0,
                    help="Diffusion regularization weight (applied to velocity).")
    return p.parse_args()


def main():
    args = parse_args()
    device = setup_device(gpu_id=int(args.gpu), seed=0, deterministic=False)
    runner = Runner(args, device)
    ixi_flip = (0,) if args.ds == "IXI" else (1, 2, 3)
    run_train(args=args, runner=runner, build_loaders=partial(loaders_baseline, ixi_flip_axes=ixi_flip))


if __name__ == "__main__":
    main()
