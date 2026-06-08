import argparse

import torch
from torch import optim

from utils import setup_device
from experiments.core.cli_args import add_common_args
from experiments.core.data_loaders import baseline_loader_builder
from experiments.core.train_runtime import TrainContext, run_train
from models.LKUNet.wrapper import LkuNetSolo
from models.LKUNet.configs import CONFIGS
from models.LKUNet.model import smoothloss, mse_loss, sad_loss


def _ncc_to_loss_fn(ctx_ncc):
    """Adapt utils.NCCVxm (window=9) into a (true, pred) signature."""
    def _fn(y_true, y_pred):
        return ctx_ncc(y_true, y_pred)
    return _fn


class Runner:
    def __init__(self, args, device):
        self.args, self.device = args, device
        self.img_size = tuple(int(v) for v in args.img_size)

        cfg = CONFIGS[args.config]
        self.model = LkuNetSolo(
            vol_size=self.img_size,
            in_channel=int(cfg.in_channel),
            n_classes=int(cfg.n_classes),
            start_channel=int(cfg.start_channel),
        ).to(device)

        n_params = sum(p.numel() for p in self.model.parameters())
        print(f"LkuNetSolo [{args.config}] params: {n_params:,} ({n_params/1e6:.3f}M)")

        self.optimizer = optim.Adam(self.model.parameters(), lr=args.lr, amsgrad=True)
        self.ctx = TrainContext(device, vol_size=self.img_size, ncc_win=(9, 9, 9))
        self.forward_flow = self._forward_flow

        d, h, w = self.img_size
        scale = torch.tensor(
            [(d - 1) / 2.0, (h - 1) / 2.0, (w - 1) / 2.0],
            device=device,
            dtype=torch.float32,
        ).view(1, 3, 1, 1, 1)
        self._flow_scale = scale

        sim_key = str(args.sim).lower()
        if sim_key == "mse": self._sim_fn = mse_loss
        elif sim_key == "sad": self._sim_fn = sad_loss
        elif sim_key == "ncc": self._sim_fn = _ncc_to_loss_fn(self.ctx.ncc)
        else: raise ValueError(f"Unknown --sim '{args.sim}'. Use one of: mse, sad, ncc.")
        self._sim_key = sim_key


    @torch.no_grad()
    def _forward_flow(self, x, y):
        """Return voxel-unit flow; LKU-Net predicts normalized grid flow."""
        flow_normalized = self.model.unet(x, y)
        return flow_normalized * self._flow_scale


    def train_step(self, batch, epoch):
        args, ctx = self.args, self.ctx
        x, y = (batch[0], batch[1]) if args.ds == "OASIS" else batch
        x, y = x.to(self.device).float(), y.to(self.device).float()

        warped, flow = self.model(x, y)

        with torch.autocast(device_type="cuda", enabled=False):
            L_sim = self._sim_fn(y.float(), warped.float()) * args.w_sim
        L_reg = smoothloss(flow) * args.w_reg
        loss = L_sim + L_reg

        return loss, {
            "all": loss.item(),
            "ncc": L_sim.item() if self._sim_key == "ncc" else 0.0,
            "sim": L_sim.item(),
            "reg": L_reg.item(),
        }


def parse_args():
    p = argparse.ArgumentParser()
    add_common_args(p)
    p.set_defaults(exp="LKUNet")

    p.add_argument("--config", type=str, default="LKU-8",
                    choices=list(CONFIGS.keys()), help="LKU-Net config key. "
                    "LKU-4=paper main (0.5M); LKU-8=lightweight baseline (2M); "
                    "LKU-16=mid (8M); LKU-32=L2R-2021 submission (33M).")
    p.add_argument("--sim", type=str, default="mse", choices=["mse", "sad", "ncc"], help="Similarity loss (paper default: mse).")
    p.add_argument("--w_sim", type=float, default=1.0, help="Similarity loss weight.")
    p.add_argument("--w_reg", type=float, default=0.01, help="Smoothness loss weight (paper default for OASIS: 0.01).")
    return p.parse_args()


def main():
    args = parse_args()
    device = setup_device(gpu_id=int(args.gpu), seed=0, deterministic=False)
    runner = Runner(args, device)
    run_train(args=args, runner=runner, build_loaders=baseline_loader_builder(args))


if __name__ == "__main__":
    main()
