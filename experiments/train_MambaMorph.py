"""
Standalone MambaMorph training (Guo et al., 2024; arxiv 2401.13934).

Loss configuration follows the upstream MambaMorph script for unsupervised
brain MRI registration:
    L = w_ncc * NCC(warped, fix) + w_reg * Grad3d_l2(velocity)

The smoothness regularizer is applied to the PRE-INTEGRATION velocity field
(`preint_flow`) when running the diffeomorphic variant, consistent with how
SVF-style models are typically trained. We retain our project's NCCVxm
(uniform window=9) for the similarity term, matching the rest of our
training infrastructure.

Requires `mamba_ssm` to be importable (run tools/install_mamba.sh first).

Usage:
    python -m experiments.train_MambaMorph --ds OASIS --2 --max_epoch 100
"""
import argparse
from functools import partial

import torch
from torch import optim

from utils import setup_device, Grad3d
from experiments.core.train_runtime import Ctx, add_common_args, run_train, loaders_baseline
from models.MambaMorph.wrapper import MambaMorphSolo
from models.MambaMorph.configs import CONFIGS


IMG_SIZE = (160, 192, 224)


class Runner:
    def __init__(self, args, device):
        self.args, self.device = args, device

        self.model = MambaMorphSolo(
            config_key=args.config,
            diffeomorphic=bool(int(args.diffeo)),
        ).to(device)
        n_params = sum(p.numel() for p in self.model.parameters())
        print(f"MambaMorphSolo [{args.config}, diffeo={bool(int(args.diffeo))}] "
              f"params: {n_params:,} ({n_params/1e6:.3f}M)")

        self.optimizer = optim.Adam(self.model.parameters(), lr=args.lr, amsgrad=True)
        self.ctx = Ctx(device, vol_size=IMG_SIZE, ncc_win=(9, 9, 9))
        self.forward_flow = self._forward_flow
        self._reg_l2 = Grad3d(penalty="l2")

    @torch.no_grad()
    def _forward_flow(self, x, y):
        # Returns the integrated (positive) displacement field used for
        # downstream metrics (Dice/Jacobian).
        warped, flow = self.model(x, y)
        return flow

    def train_step(self, batch, epoch):
        args, ctx = self.args, self.ctx
        x, y = (batch[0], batch[1]) if args.ds == "OASIS" else batch
        x, y = x.to(self.device).float(), y.to(self.device).float()

        # Run the underlying model directly so we can regularize the
        # PRE-integration velocity / displacement (`preint_flow`) instead of
        # the integrated `pos_flow`. Smoothness on the velocity field is the
        # standard SVF training recipe and matches the upstream MambaMorph code.
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
    p.set_defaults(exp="MambaMorph")

    p.add_argument("--config", type=str, default="MambaMorph",
                    choices=list(CONFIGS.keys()), help="MambaMorph config key.")
    p.add_argument("--diffeo", type=int, choices=[0, 1], default=1,
                    help="1 = diffeomorphic (VecInt-integrated SVF, paper default); "
                    "0 = direct displacement (MambaMorphOri).")
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
