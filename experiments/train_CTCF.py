import argparse
from functools import partial

import torch
from torch import optim

from datasets.synthetic import build_synth_loaders
from experiments.core.model_adapters import get_model_adapter
from experiments.core.train_rules import apply_ctcf_dataset_defaults
from experiments.core.train_runtime import Ctx, add_common_args, loaders_baseline, run_train
from utils import RegisterModel, ctcf_schedule, dice_val, icon_loss, neg_jacobian_penalty, setup_device, DareDiffusion, elastic_loss


class Runner:
    """CTCF trainer with legacy cascade warm-up schedule and bidirectional losses."""
    def __init__(self, args, device):
        self.args = args
        self.device = device
        self.is_synth = args.ds == "SYNTH"
        self.adapter = get_model_adapter("ctcf")

        synth_img_size = None
        synth_dwin = None

        if self.is_synth:
            synth_img_size = tuple(int(v) for v in args.synth_vol_size)
            synth_dwin = self._adapt_swin_windows(synth_img_size)

        self.model = self.adapter.build(
            time_steps=int(args.time_steps),
            config_key=args.config,
            use_checkpoint=bool(int(args.use_checkpoint)),
            synth_img_size=synth_img_size,
            synth_dwin=synth_dwin,
            l1_base_ch=getattr(args, 'l1_base_ch', None),
            l3_base_ch=getattr(args, 'l3_base_ch', None),
            l3_error_mode=getattr(args, 'l3_error_mode', None),
            prealign_encoder=True if int(getattr(args, 'prealign_encoder', 0)) else None,
            # GEN2 (architectural)
            l3_iters=getattr(args, 'l3_iters', None),
            l3_full_res=True if int(getattr(args, 'l3_full_res', 0)) else None,
            learned_upsample=True if int(getattr(args, 'learned_upsample', 0)) else None,
            l2_l3_skip=True if int(getattr(args, 'l2_l3_skip', 0)) else None,
            l1_half_res=True if int(getattr(args, 'l1_half_res', 0)) else None,
            l2_full_res=True if int(getattr(args, 'l2_full_res', 0)) else None,
            l1_l2_skip=True if int(getattr(args, 'l1_l2_skip', 0)) else None,
            l3_compose=True if int(getattr(args, 'l3_compose', 0)) else None,
            l3_svf=True if int(getattr(args, 'l3_svf', 0)) else None,
            # GEN2.5 (capacity)
            l3_cab=True if int(getattr(args, 'l3_cab', 0)) else None,
            l3_context_blocks=int(args.l3_context) if int(getattr(args, 'l3_context', 0)) > 0 else None,
            l3_gate=True if int(getattr(args, 'l3_gate', 0)) else None,
            l3_unshared=True if int(getattr(args, 'l3_unshared', 0)) else None,
            l1_cab=True if int(getattr(args, 'l1_cab', 0)) else None,
        ).to(device)

        self.optimizer = optim.Adam(self.model.parameters(), lr=args.lr, amsgrad=True)
        self.img_size = tuple(int(v) for v in self.model.img_size_full)
        self.ctx = Ctx(device, vol_size=self.img_size, ncc_win=(9, 9, 9))
        self.reg_nearest = RegisterModel(self.img_size, mode="nearest").to(device) if self.is_synth else None
        self.forward_flow = self._forward_flow
        self.lr_policy = "ctcf"
        self._val_alpha_l1 = 0.0 if int(getattr(args, 'disable_l1', 0)) else 1.0
        self._val_alpha_l3 = 0.0 if int(getattr(args, 'disable_l3', 0)) else 1.0

        # Phase 2 losses
        self.l2_full_res = bool(int(getattr(args, 'l2_full_res', 0)))
        self.reg_mode = str(getattr(args, 'reg_mode', 'diffusion'))
        if self.reg_mode == 'dare':
            self.dare_reg = DareDiffusion(beta=float(getattr(args, 'dare_beta', 1.0)))
        elif self.reg_mode == 'elastic':
            self._elastic_mu = float(getattr(args, 'elastic_mu', 1.0))
            self._elastic_lam = float(getattr(args, 'elastic_lam', 1.0))
        self.w_aux = float(getattr(args, 'w_aux', 0.0))
        self.w_reg_l1 = float(getattr(args, 'w_reg_l1', 0.0))
        self.w_reg_l3 = float(getattr(args, 'w_reg_l3', 0.0))
        self._need_aux = self.w_aux > 0 or self.w_reg_l1 > 0 or self.w_reg_l3 > 0


    @staticmethod
    def _adapt_swin_windows(img):
        img = tuple(int(v) for v in img)
        if any(v % 32 != 0 for v in img):
            raise ValueError(f"synth_vol_size={img} must be divisible by 32.")
        ws = tuple(max(2, v // 32) for v in img)
        print(f"Synth cfg: img_size={img}, window_size={ws}, dwin_size={ws}")
        return ws


    @torch.no_grad()
    def _forward_flow(self, x, y):
        use_amp = torch.cuda.is_available()
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
            _, flow = self.model(x, y, return_all=False,
                                alpha_l1=self._val_alpha_l1,
                                alpha_l3=self._val_alpha_l3)
        return flow.float()


    def train_step(self, batch, epoch):
        args, ctx = self.args, self.ctx

        if self.is_synth:
            x, y, x_seg, y_seg = batch[0], batch[1], batch[2], batch[3]
            x_seg, y_seg = x_seg.to(self.device).long(), y_seg.to(self.device).long()
        else:
            x, y = (batch[0], batch[1]) if args.ds == "OASIS" else batch
            x_seg = y_seg = None

        x, y = x.to(self.device).float(), y.to(self.device).float()

        schedule_max_epoch = int(args.schedule_max_epoch) if int(getattr(args, "schedule_max_epoch", 0)) > 0 else int(args.max_epoch)
        alpha_l1, alpha_l3, warm = ctcf_schedule(epoch=int(epoch), max_epoch=schedule_max_epoch)
        if args.l1_from_start:
            alpha_l1 = 1.0
        if args.disable_l1:
            alpha_l1 = 0.0
        if args.disable_l3:
            alpha_l3 = 0.0
        self._val_alpha_l1 = alpha_l1
        self._val_alpha_l3 = alpha_l3
        W_icon = float(args.w_icon) * warm
        W_jac = float(args.w_jac) * warm

        need_all = self._need_aux
        out_xy = self.model(x, y, return_all=need_all, alpha_l1=alpha_l1, alpha_l3=alpha_l3)
        out_yx = self.model(y, x, return_all=need_all, alpha_l1=alpha_l1, alpha_l3=alpha_l3)

        if need_all:
            def_xy, flow_xy, aux_xy = out_xy
            def_yx, flow_yx, aux_yx = out_yx
        else:
            def_xy, flow_xy = out_xy
            def_yx, flow_yx = out_yx
            aux_xy = aux_yx = {}

        with torch.autocast(device_type="cuda", enabled=False):
            L_ncc = 0.5 * (ctx.ncc(def_xy.float(), y.float()) + ctx.ncc(def_yx.float(), x.float())) * args.w_ncc

        L_icon = icon_loss(flow_xy, flow_yx, mode=args.icon_mode) * W_icon
        L_jac = 0.5 * (neg_jacobian_penalty(flow_xy) + neg_jacobian_penalty(flow_yx)) * W_jac

        # Regularization (switchable)
        if self.reg_mode == 'dare':
            L_reg = 0.5 * (self.dare_reg(flow_xy) + self.dare_reg(flow_yx)) * args.w_reg
        elif self.reg_mode == 'elastic':
            L_reg = 0.5 * (elastic_loss(flow_xy, self._elastic_mu, self._elastic_lam) +
                           elastic_loss(flow_yx, self._elastic_mu, self._elastic_lam)) * args.w_reg
        else:
            L_reg = 0.5 * (ctx.reg(flow_xy) + ctx.reg(flow_yx)) * args.w_reg

        loss = L_ncc + L_icon + L_reg + L_jac

        # Cascade-aware per-level regularization
        L_reg_l1 = L_reg_l3 = torch.tensor(0.0)
        if self.w_reg_l1 > 0 and "flow_half_init" in aux_xy:
            L_reg_l1 = 0.5 * (ctx.reg(aux_xy["flow_half_init"]) + ctx.reg(aux_yx["flow_half_init"])) * self.w_reg_l1
            loss = loss + L_reg_l1
        if self.w_reg_l3 > 0 and "flow_half_ref" in aux_xy:
            L_reg_l3 = 0.5 * (ctx.reg(aux_xy["flow_half_ref"]) + ctx.reg(aux_yx["flow_half_ref"])) * self.w_reg_l3
            loss = loss + L_reg_l3

        # Aux loss: NCC on L2 intermediate output
        L_aux = torch.tensor(0.0)
        if self.w_aux > 0 and "def_half_l2" in aux_xy:
            if self.l2_full_res:
                aux_tgt_xy, aux_tgt_yx = y, x  # L2 output already full-res
            else:
                aux_tgt_xy = torch.nn.functional.interpolate(y, scale_factor=0.5, mode="trilinear", align_corners=False)
                aux_tgt_yx = torch.nn.functional.interpolate(x, scale_factor=0.5, mode="trilinear", align_corners=False)
            with torch.autocast(device_type="cuda", enabled=False):
                L_aux = 0.5 * (ctx.ncc(aux_xy["def_half_l2"].float(), aux_tgt_xy.float()) +
                               ctx.ncc(aux_yx["def_half_l2"].float(), aux_tgt_yx.float())) * self.w_aux
            loss = loss + L_aux

        logs = {
            "all": loss.item(),
            "ncc": L_ncc.item(),
            "reg": L_reg.item(),
            "icon": L_icon.item(),
            "jac": L_jac.item(),
            "alpha_l1": alpha_l1,
            "alpha_l3": alpha_l3,
            "warm": warm,
        }
        if self.w_aux > 0: logs["aux"] = L_aux.item()
        if self.w_reg_l1 > 0: logs["reg_l1"] = L_reg_l1.item()
        if self.w_reg_l3 > 0: logs["reg_l3"] = L_reg_l3.item()

        if self.is_synth:
            with torch.no_grad():
                def_seg_xy = self.reg_nearest((x_seg.float(), flow_xy.detach().float()))
                logs["dice_tr"] = float(dice_val(def_seg_xy.long(), y_seg, num_clus=int(args.synth_num_labels)).item())

        return loss, logs


def parse_args():
    p = argparse.ArgumentParser()
    add_common_args(p, include_synth=True)
    p.set_defaults(exp="CTCF")

    p.add_argument("--config", type=str, default="CTCF-CascadeA", help="Model config key.")
    p.add_argument("--time_steps", type=int, default=6, help="Number of velocity integration steps.")
    p.add_argument("--schedule_max_epoch", type=int, default=0, help="If >0, uses this epoch horizon for CTCF stage schedule (alpha/warm), independent of --max_epoch.")
    p.add_argument("--use_checkpoint", type=int, choices=[0, 1], default=1, help="Enable gradient checkpointing in Swin blocks.")

    p.add_argument("--w_ncc", type=float, default=1.0, help="NCC similarity loss weight.")
    p.add_argument("--w_reg", type=float, default=None, help="Flow regularization loss weight (auto: IXI=4.0, others=1.0).")
    p.add_argument("--w_icon", type=float, default=0.05, help="ICON loss base weight.")
    p.add_argument("--w_jac", type=float, default=0.005, help="Negative Jacobian penalty base weight.")
    p.add_argument("--icon_mode", type=str, choices=["l1", "l2"], default="l1", help="ICON loss norm: l1 (default) or l2.")
    p.add_argument("--l1_from_start", type=int, choices=[0, 1], default=0, help="If 1, alpha_l1=1.0 from epoch 0 (skip schedule).")
    p.add_argument("--disable_l1", type=int, choices=[0, 1], default=0, help="If 1, force alpha_l1=0 (disable Level 1 coarse flow).")
    p.add_argument("--disable_l3", type=int, choices=[0, 1], default=0, help="If 1, force alpha_l3=0 (disable Level 3 refiner).")
    p.add_argument("--l1_base_ch", type=int, default=None, help="L1 coarse net base channels (default: config value, typically 16).")
    p.add_argument("--l3_base_ch", type=int, default=None, help="L3 refiner base channels (default: config value, typically 16).")
    p.add_argument("--l3_error_mode", type=str, choices=["absdiff", "gradmag", "ncc"], default=None, help="L3 error map mode.")
    p.add_argument("--prealign_encoder", type=int, choices=[0, 1], default=0, help="If 1, L2 encoder sees L1-warped mov instead of raw mov.")

    # GEN2 enhancements (architectural)
    p.add_argument("--l3_iters", type=int, default=None, help="Number of L3 refinement iterations (default: 1).")
    p.add_argument("--l3_full_res", type=int, choices=[0, 1], default=0, help="If 1, run L3 at full-res instead of half-res.")
    p.add_argument("--learned_upsample", type=int, choices=[0, 1], default=0, help="If 1, use learned flow upsampling instead of trilinear.")
    p.add_argument("--l2_l3_skip", type=int, choices=[0, 1], default=0, help="If 1, pass L2 decoder features to L3.")
    p.add_argument("--l1_half_res", type=int, choices=[0, 1], default=0, help="If 1, run L1 at half-res instead of quarter-res.")
    p.add_argument("--l2_full_res", type=int, choices=[0, 1], default=0, help="If 1, run L2 at full-res (for lightweight backbones like VxM).")
    p.add_argument("--l1_l2_skip", type=int, choices=[0, 1], default=0, help="If 1, pass L1 features to L2 conv-skip path.")
    p.add_argument("--l3_compose", type=int, choices=[0, 1], default=0, help="If 1, use flow composition in L3 instead of addition.")
    p.add_argument("--l3_svf", type=int, choices=[0, 1], default=0, help="If 1, integrate L3 delta as SVF via scaling-and-squaring (diffeomorphic L3).")

    # GEN2.5 enhancements (capacity)
    p.add_argument("--l3_cab", type=int, choices=[0, 1], default=0, help="If 1, add channel attention (CAB) to L3 decoder.")
    p.add_argument("--l3_context", type=int, default=0, help="Number of ResidualContext3D blocks in L3 bottleneck (0=none).")
    p.add_argument("--l3_gate", type=int, choices=[0, 1], default=0, help="If 1, spatial RefineGate3D on L3 delta output.")
    p.add_argument("--l3_unshared", type=int, choices=[0, 1], default=0, help="If 1, use separate L3 weights per iteration (requires l3_iters>1).")
    p.add_argument("--l1_cab", type=int, choices=[0, 1], default=0, help="If 1, add channel attention (CAB) to L1 decoder.")

    # Phase 2: loss innovations
    p.add_argument("--reg_mode", type=str, choices=["diffusion", "dare", "elastic"], default="diffusion",
                    help="Regularization mode: diffusion (default Grad3d), dare (DARE-minimal), elastic (Navier-Cauchy).")
    p.add_argument("--dare_beta", type=float, default=1.0, help="DARE beta: controls adaptive weighting strength.")
    p.add_argument("--elastic_mu", type=float, default=1.0, help="ElasticMorph mu (shear modulus).")
    p.add_argument("--elastic_lam", type=float, default=1.0, help="ElasticMorph lambda (Lame first parameter).")
    p.add_argument("--w_aux", type=float, default=0.0, help="Auxiliary NCC loss on L2 intermediate output (0=disabled).")
    p.add_argument("--w_reg_l1", type=float, default=0.0, help="Extra diffusion reg on L1 flow (cascade-aware, 0=disabled).")
    p.add_argument("--w_reg_l3", type=float, default=0.0, help="Extra diffusion reg on L3 delta (cascade-aware, 0=disabled).")

    p.add_argument("--synth_train_samples", type=int, default=256, help="Number of synthetic training pairs.")
    p.add_argument("--synth_val_samples", type=int, default=32, help="Number of synthetic validation pairs.")
    p.add_argument("--synth_num_labels", type=int, default=36, help="Number of synthetic segmentation labels.")
    p.add_argument("--synth_vol_size", type=int, nargs=3, default=(96, 96, 96), help="Synthetic volume size D H W (each must be divisible by 32).")
    p.add_argument("--synth_flow_max_disp", type=float, default=6.0, help="Max synthetic displacement amplitude in voxels.")
    p.add_argument("--synth_seed", type=int, default=123, help="Base seed for synthetic pair generation.")
    return p.parse_args()


def main():
    args = parse_args()
    apply_ctcf_dataset_defaults(args)

    # IXI: flip axis 0 only (depth), matching UTSRMorph original protocol.
    # OASIS / other: flip all 3 axes.
    ixi_flip = (0,) if args.ds == "IXI" else (1, 2, 3)
    build_loaders = build_synth_loaders if args.ds == "SYNTH" else partial(loaders_baseline, ixi_flip_axes=ixi_flip)
    device = setup_device(gpu_id=int(args.gpu), seed=0, deterministic=False)
    runner = Runner(args, device)
    run_train(args=args, runner=runner, build_loaders=build_loaders)


if __name__ == "__main__":
    main()
