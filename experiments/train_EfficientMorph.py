"""
Standalone EfficientMorph training (Bin Aziz et al., WACV 2025).

Loss configuration (UNSUPERVISED, matching the paper's similarity + smoothness terms
but EXCLUDING the optional supervised Dice loss --seg_loss in the original script):
    L = w_ncc * NCC_gauss(warped, fix) + w_reg * Grad3d_l2(flow)

NCC_gauss is EfficientMorph's native Gaussian-windowed NCC (window=9, sigma=1.5),
distinct from VoxelMorph-style uniform-window NCC. We retain it for fidelity to
the published baseline.

Training infrastructure (LR schedule, optimizer, batch size, epochs) follows our
unified per-backbone protocol via experiments/core/train_runtime.py for clean
cross-backbone comparison.

Usage:
    python -m experiments.train_EfficientMorph --ds OASIS --1 --config EfficientMorph_2x3_2 --max_epoch 100
"""
import argparse
from functools import partial

import torch
from torch import optim

from utils import setup_device, Grad3d
from experiments.core.train_runtime import Ctx, add_common_args, run_train, loaders_baseline
from models.EfficientMorph.wrapper import EfficientMorphSolo
from models.EfficientMorph.configs import (
    get_EM_2x3_2_config, get_EM_1x1_2_config,
    get_EM_2x3_4_config, get_EM_1x1_4_config,
)


# Available config keys (mirrors the dict in models/EfficientMorph/model.py).
CONFIG_KEYS = ["EfficientMorph_2x3_2", "EfficientMorph_1x1_2",
               "EfficientMorph_2x3_4", "EfficientMorph_1x1_4",
               "EfficientMorph_2x3_2_hires", "EfficientMorph_1x1_2_hires"]


def _build_ncc_gauss(win: int = 9, device: str = "cuda"):
    """EfficientMorph's Gaussian-windowed NCC, but with explicit device.

    The original NCC_gauss class hardcodes `.to('cuda')` in __init__, which
    breaks for non-CUDA devices. We construct it inline here on the supplied
    device.
    """
    from math import exp
    import torch.nn.functional as F

    sigma = 1.5
    gauss = torch.tensor([exp(-(x - win // 2) ** 2 / (2 * sigma ** 2)) for x in range(win)])
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


IMG_SIZE = (160, 192, 224)


class Runner:
    def __init__(self, args, device):
        self.args, self.device = args, device

        self.model = EfficientMorphSolo(config_key=args.config).to(device)

        n_params = sum(p.numel() for p in self.model.parameters())
        print(f"EfficientMorphSolo [{args.config}] params: {n_params:,} ({n_params/1e6:.3f}M)")

        self.optimizer = optim.Adam(self.model.parameters(), lr=args.lr, amsgrad=True)
        self.ctx = Ctx(device, vol_size=IMG_SIZE, ncc_win=(9, 9, 9))
        self.forward_flow = self._forward_flow

        self._ncc_gauss = _build_ncc_gauss(win=9, device=str(device))
        self._reg_l2 = Grad3d(penalty="l2")

    @torch.no_grad()
    def _forward_flow(self, x, y):
        _, flow = self.model(x, y)
        return flow

    def train_step(self, batch, epoch):
        args = self.args
        x, y = (batch[0], batch[1]) if args.ds == "OASIS" else batch
        x, y = x.to(self.device).float(), y.to(self.device).float()

        warped, flow = self.model(x, y)

        with torch.autocast(device_type="cuda", enabled=False):
            L_ncc = self._ncc_gauss(y.float(), warped.float()) * args.w_ncc
        L_reg = self._reg_l2(flow) * args.w_reg
        loss = L_ncc + L_reg

        return loss, {"all": loss.item(), "ncc": L_ncc.item(), "reg": L_reg.item()}


def parse_args():
    p = argparse.ArgumentParser()
    add_common_args(p)
    p.set_defaults(exp="EfficientMorph")

    p.add_argument("--config", type=str, default="EfficientMorph_2x3_2_hires",
                    choices=CONFIG_KEYS, help="EfficientMorph config key. "
                    "Light (embed_dim=24): EfficientMorph_{2x3,1x1}_{2,4}. "
                    "Hires (embed_dim=96, paper main): EfficientMorph_{2x3,1x1}_2_hires.")
    p.add_argument("--w_ncc", type=float, default=1.0, help="NCC similarity loss weight.")
    p.add_argument("--w_reg", type=float, default=1.0, help="Diffusion regularization weight.")
    return p.parse_args()


def main():
    args = parse_args()
    device = setup_device(gpu_id=int(args.gpu), seed=0, deterministic=False)
    runner = Runner(args, device)
    ixi_flip = (0,) if args.ds == "IXI" else (1, 2, 3)
    run_train(args=args, runner=runner, build_loaders=partial(loaders_baseline, ixi_flip_axes=ixi_flip))


if __name__ == "__main__":
    main()
