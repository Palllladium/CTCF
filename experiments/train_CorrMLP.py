"""Train CorrMLP (Meng et al., CVPR 2024) as a standalone baseline on our OASIS/IXI split.

Verifies the unsupervised gap reported in memory/competitor_landscape_2026.md (~0.871 OASIS)
on our exact data and protocol. Loss = NCC(win=9) + diffusion, weights [1, 1], Adam 1e-4 —
identical to CorrMLP's native recipe and to our baseline loss. GPL-3.0 upstream code lives in
models/CorrMLP/networks_upstream.py (git-ignored); this trainer is our harness glue.
"""

from __future__ import annotations

import argparse

import torch
from torch import optim

from experiments.core.cli_common import add_common_args
from experiments.core.data_loaders import baseline_loader_builder
from experiments.core.train_runtime import TrainContext, run_train
from models.CorrMLP.wrapper import CorrMLPSolo
from utils import setup_device


class Runner:
    def __init__(self, args, device):
        self.args, self.device = args, device
        self.img_size = tuple(args.img_size)
        self.model = CorrMLPSolo(
            enc_channels=args.enc_channels,
            dec_channels=args.dec_channels,
            use_checkpoint=bool(args.use_checkpoint),
        ).to(device)

        n_params = sum(p.numel() for p in self.model.parameters())
        print(f"CorrMLP [enc={args.enc_channels} dec={args.dec_channels}] params: {n_params:,} ({n_params / 1e6:.3f}M)")

        self.optimizer = optim.Adam(self.model.parameters(), lr=args.lr)
        self.ctx = TrainContext(device, vol_size=self.img_size, ncc_win=(9, 9, 9))
        self.forward_flow = self._forward_flow

    @torch.no_grad()
    def _forward_flow(self, x, y):
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=torch.cuda.is_available()):
            _, flow = self.model(x, y)
        return flow.float()

    def train_step(self, batch, epoch):
        args, ctx = self.args, self.ctx
        x, y = batch[0].to(self.device).float(), batch[1].to(self.device).float()

        def_x, flow = self.model(x, y)

        with torch.autocast(device_type="cuda", enabled=False):
            L_ncc = ctx.ncc(def_x.float(), y.float()) * args.w_ncc
        L_reg = ctx.reg(flow) * args.w_reg
        loss = L_ncc + L_reg

        return loss, {"all": loss.item(), "ncc": L_ncc.item(), "reg": L_reg.item()}


def parse_args():
    p = argparse.ArgumentParser()
    add_common_args(p)
    p.set_defaults(exp="CorrMLP")

    p.add_argument("--enc_channels", type=int, default=8, help="CorrMLP encoder base channels.")
    p.add_argument("--dec_channels", type=int, default=16, help="CorrMLP decoder base channels.")
    p.add_argument(
        "--use_checkpoint",
        type=int,
        choices=[0, 1],
        default=1,
        help="Gradient checkpointing inside CorrMLP (recommended at full-res).",
    )
    p.add_argument("--w_ncc", type=float, default=1.0, help="NCC similarity loss weight.")
    p.add_argument(
        "--w_reg",
        type=float,
        default=1.0,
        help="Diffusion regularization weight (CorrMLP native = 1.0 for both OASIS and IXI).",
    )
    return p.parse_args()


def main():
    args = parse_args()
    device = setup_device(gpu_id=args.gpu, seed=0, deterministic=False)
    runner = Runner(args, device)
    run_train(args=args, runner=runner, build_loaders=baseline_loader_builder(args))


if __name__ == "__main__":
    main()
