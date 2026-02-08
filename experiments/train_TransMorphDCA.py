import argparse, torch
from torch import optim

from utils import setup_device
from models.TransMorph_DCA.model import TransMorphCascadeAd, CONFIGS as CONFIGS_TM
from experiments.engine import add_common_args, apply_paths, run_train, make_ctx, loaders_baseline, forward_flow_halfres


class Runner:
    def __init__(self, args, device):
        self.args, self.device = args, device

        cfg = CONFIGS_TM[args.config] 
        cfg.time_steps = int(args.time_steps)
        half = tuple(cfg.img_size)
        full = tuple(s * 2 for s in half)

        self.model = TransMorphCascadeAd(cfg, int(args.time_steps)).to(device)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=args.lr, weight_decay=0, amsgrad=True)
        self.ctx = make_ctx(device, vol_size=full, ncc_win=(9, 9, 9))


    @torch.no_grad()
    def forward_flow(self, x, y):
        return forward_flow_halfres(self.model, x, y, pool=2, amp=True)


    def train_step(self, batch, epoch):
        args, ctx = self.args, self.ctx
        x, y = (batch[0], batch[1]) if args.ds == "OASIS" else batch
        x, y = x.to(self.device).float(), y.to(self.device).float()

        flow = forward_flow_halfres(self.model, x, y, pool=2, amp=True)
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
    p.add_argument("--config", type=str, default="TransMorph-3-LVL")
    p.add_argument("--w_ncc", type=float, default=1.0)
    p.add_argument("--w_reg", type=float, default=1.0)
    p.add_argument("--time_steps", type=int, default=12)
    return p.parse_args()


def main():
    args = parse_args()
    apply_paths(args)
    device = setup_device(gpu_id=int(args.gpu), seed=0, deterministic=False)
    runner = Runner(args, device) # addressing own class
    run_train(args=args, runner=runner, build_loaders=loaders_baseline)


if __name__ == "__main__":
    main()