from __future__ import annotations

import argparse
from math import exp

import torch
import torch.nn.functional as F
from torch import optim

from experiments.core.cli_args import add_common_args
from experiments.core.data_loaders import baseline_loader_builder
from experiments.core.train_runtime import TrainContext, run_train
from models.EfficientMorph.configs import CONFIGS
from models.EfficientMorph.wrapper import EfficientMorphSolo
from utils import setup_device


def _build_ncc_gauss(win: int = 9, device: torch.device | str = "cuda"):
    """Gaussian-windowed local NCC loss (1 - mean correlation), as used by EfficientMorph."""
    sigma = 1.5
    gauss = torch.tensor([exp(-((x - win // 2) ** 2) / (2 * sigma**2)) for x in range(win)])
    gauss = gauss / gauss.sum()
    _1d = gauss.unsqueeze(1)
    _2d = _1d @ _1d.t()
    _3d = (_1d @ _2d.reshape(1, -1)).reshape(win, win, win).float()
    filt = _3d.unsqueeze(0).unsqueeze(0).contiguous().to(device)
    pad = win // 2

    def _ncc(y_true, y_pred):
        Ii, Ji = y_true, y_pred
        mu1 = F.conv3d(Ii, filt, padding=pad)
        mu2 = F.conv3d(Ji, filt, padding=pad)
        mu1_sq, mu2_sq = mu1.pow(2), mu2.pow(2)
        mu1_mu2 = mu1 * mu2
        sigma1_sq = F.conv3d(Ii * Ii, filt, padding=pad) - mu1_sq
        sigma2_sq = F.conv3d(Ji * Ji, filt, padding=pad) - mu2_sq
        sigma12 = F.conv3d(Ii * Ji, filt, padding=pad) - mu1_mu2
        cc = (sigma12 * sigma12 + 1e-5) / (sigma1_sq * sigma2_sq + 1e-5)
        return 1.0 - torch.mean(cc)

    return _ncc


class Runner:
    def __init__(self, args, device):
        self.args, self.device = args, device
        self.img_size = tuple(args.img_size)

        self.model = EfficientMorphSolo(config_key=args.config, img_size=self.img_size).to(device)

        n_params = sum(p.numel() for p in self.model.parameters())
        print(f"EfficientMorphSolo [{args.config}] params: {n_params:,} ({n_params / 1e6:.3f}M)")

        self.optimizer = optim.Adam(self.model.parameters(), lr=args.lr, amsgrad=True)
        self.ctx = TrainContext(device, vol_size=self.img_size, ncc_win=(9, 9, 9))
        self.forward_flow = self._forward_flow
        self._ncc_gauss = _build_ncc_gauss(win=9, device=device)

    @torch.no_grad()
    def _forward_flow(self, x, y):
        _, flow = self.model(x, y)
        return flow

    def train_step(self, batch, epoch):
        args, ctx = self.args, self.ctx
        x, y = batch[0], batch[1]
        x, y = x.to(self.device).float(), y.to(self.device).float()

        warped, flow = self.model(x, y)

        with torch.autocast(device_type="cuda", enabled=False):
            L_ncc = self._ncc_gauss(y.float(), warped.float()) * args.w_ncc
        L_reg = ctx.reg(flow) * args.w_reg
        loss = L_ncc + L_reg

        return loss, {"all": loss.item(), "ncc": L_ncc.item(), "reg": L_reg.item()}


def parse_args():
    p = argparse.ArgumentParser()
    add_common_args(p)
    p.set_defaults(exp="EfficientMorph")

    p.add_argument(
        "--config",
        type=str,
        default="EfficientMorph_2x3_2_hires",
        choices=list(CONFIGS.keys()),
        help=(
            "EfficientMorph config key. Light (embed_dim=24): EfficientMorph_{2x3,1x1}_{2,4}. "
            "Hires (embed_dim=96, paper main): EfficientMorph_{2x3,1x1}_2_hires."
        ),
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
        help="Diffusion regularization weight.",
    )
    return p.parse_args()


def main():
    args = parse_args()
    device = setup_device(gpu_id=args.gpu, seed=0, deterministic=False)
    runner = Runner(args, device)
    run_train(args=args, runner=runner, build_loaders=baseline_loader_builder(args))


if __name__ == "__main__":
    main()
