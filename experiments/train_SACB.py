"""Train SACB-Net (Cheng et al., CVPR 2025) as a standalone baseline on our OASIS/IXI split.

Verifies its reported IXI 0.769 (vs our 0.7635) on our exact data. SACB's flow is voxel-units
(TransMorph-style ST), so it runs through our Runner/val like VoxelMorph/CorrMLP. Loss = NCC(win 9)
+ diffusion, weights [1, 0.3] (SACB native), Adam 1e-4. Verbatim upstream net lives in models/SACB/
(SACB1/model/nn_util/utils); this trainer is our harness glue.

Requires CUDA + deps: einops kmeans_gpu timm monai pystrum scipy (SACB1 hardcodes `.cuda()`),
so there is no CPU smoke path — sanity-check val Dice (>0.7) on the first GPU run.
"""

from __future__ import annotations

import argparse

import torch
from torch import optim

from experiments.core.cli_common import add_common_args
from experiments.core.data_loaders import baseline_loader_builder
from experiments.core.train_runtime import TrainContext, run_train
from models.SACB.wrapper import SACBSolo
from utils import setup_device


class Runner:
    def __init__(self, args, device):
        self.args, self.device = args, device
        self.img_size = tuple(args.img_size)
        self.model = SACBSolo(
            img_size=self.img_size,
            num_k=args.num_k,
            ch_scale=args.ch_scale,
        ).to(device)

        n_params = sum(p.numel() for p in self.model.parameters())
        print(f"SACB-Net [num_k={args.num_k} ch_scale={args.ch_scale}] params: {n_params:,} ({n_params / 1e6:.3f}M)")

        self.optimizer = optim.Adam(self.model.parameters(), lr=args.lr)
        self.ctx = TrainContext(device, vol_size=self.img_size, ncc_win=(9, 9, 9))
        self.forward_flow = self._forward_flow

    @torch.no_grad()
    def _forward_flow(self, x, y):
        _, flow = self.model(x, y)
        return flow.float()

    def train_step(self, batch, epoch):
        args, ctx = self.args, self.ctx
        x, y = batch[0].to(self.device).float(), batch[1].to(self.device).float()

        # run_train wraps train_step in fp16 autocast, but SACB's kmeans_gpu clustering breaks
        # under fp16 (Half/Float dtype mismatch in kmeans.fit_predict's index_put). Force fp32.
        with torch.autocast(device_type="cuda", enabled=False):
            def_x, flow = self.model(x, y)

            L_ncc = ctx.ncc(def_x.float(), y.float()) * args.w_ncc
            L_reg = ctx.reg(flow) * args.w_reg
            loss = L_ncc + L_reg

        return loss, {"all": loss.item(), "ncc": L_ncc.item(), "reg": L_reg.item()}


def parse_args():
    p = argparse.ArgumentParser()
    add_common_args(p)
    p.set_defaults(exp="SACB")

    p.add_argument("--num_k", type=int, default=7, help="SACB clusters per block (SACB native = 7).")
    p.add_argument("--ch_scale", type=int, default=4, help="SACB encoder channel scale.")
    p.add_argument("--w_ncc", type=float, default=1.0, help="NCC similarity loss weight.")
    p.add_argument(
        "--w_reg",
        type=float,
        default=0.3,
        help="Diffusion regularization weight (SACB native = 0.3 on its voxel-units flow).",
    )
    return p.parse_args()


def main():
    args = parse_args()
    device = setup_device(gpu_id=args.gpu, seed=0, deterministic=False)
    runner = Runner(args, device)
    run_train(args=args, runner=runner, build_loaders=baseline_loader_builder(args))


if __name__ == "__main__":
    main()
