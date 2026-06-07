import argparse
import copy

import torch
from torch import optim

from datasets.synthetic import build_synth_loaders
from experiments.core.model_adapters import get_model_adapter
from experiments.core.train_runtime import Ctx, add_common_args, baseline_loader_builder, run_train
from utils import RegisterModel, ctcf_schedule, dice_val, icon_loss, neg_jacobian_penalty, setup_device, DareDiffusion, elastic_loss
from models.CTCF.configs import CONFIGS


def _optional_bool(value):
    if value is None:
        return None
    return bool(int(value))


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
            l3_iters=getattr(args, 'l3_iters', None),
            l3_unshared=_optional_bool(getattr(args, 'l3_unshared', None)),
            l1_half_res=_optional_bool(getattr(args, 'l1_half_res', None)),
            l2_full_res=_optional_bool(getattr(args, 'l2_full_res', None)),
            l3_full_res=_optional_bool(getattr(args, 'l3_full_res', None)),
            l3_svf=_optional_bool(getattr(args, 'l3_svf', None)),
            l3_num_heads=getattr(args, 'l3_num_heads', None),
        ).to(device)

        self.optimizer = optim.Adam(self.model.parameters(), lr=args.lr, amsgrad=True)
        self.img_size = tuple(int(v) for v in self.model.img_size_full)
        self.ctx = Ctx(device, vol_size=self.img_size, ncc_win=(9, 9, 9))
        self.reg_nearest = RegisterModel(self.img_size, mode="nearest").to(device) if self.is_synth else None
        self.forward_flow = self._forward_flow
        self.lr_policy = "ctcf"
        self._val_alpha_l1 = 0.0 if int(getattr(args, 'disable_l1', 0)) else 1.0
        self._val_alpha_l3 = 0.0 if int(getattr(args, 'disable_l3', 0)) else 1.0

        self.reg_mode = str(getattr(args, 'reg_mode', 'diffusion'))
        if self.reg_mode == 'dare':
            self.dare_reg = DareDiffusion(beta=float(getattr(args, 'dare_beta', 1.0)))
        elif self.reg_mode == 'elastic':
            self._elastic_mu = float(getattr(args, 'elastic_mu', 1.0))
            self._elastic_lam = float(getattr(args, 'elastic_lam', 1.0))

        # M2: EMA self-distillation
        self.ema_decay = float(getattr(args, 'ema_decay', 0.0) or 0.0)
        self.ema_lambda = float(getattr(args, 'ema_lambda', 0.0) or 0.0)
        self.use_ema = self.ema_decay > 0.0 and self.ema_lambda > 0.0
        if self.use_ema:
            self.teacher = copy.deepcopy(self.model)
            for p in self.teacher.parameters():
                p.requires_grad = False
            self.teacher.eval()
        else:
            self.teacher = None

        # M3: cascade-aware (per-level) regularization
        w_l1 = getattr(args, 'w_reg_l1', None)
        w_l2 = getattr(args, 'w_reg_l2', None)
        w_l3 = getattr(args, 'w_reg_l3', None)
        self.use_cascade_reg = any(v is not None for v in (w_l1, w_l2, w_l3))
        if self.use_cascade_reg:
            self.w_reg_l1 = float(w_l1) if w_l1 is not None else 1.0
            self.w_reg_l2 = float(w_l2) if w_l2 is not None else 1.0
            self.w_reg_l3 = float(w_l3) if w_l3 is not None else 1.0
            if self.reg_mode != 'diffusion':
                raise ValueError("Cascade-aware reg (w_reg_l1/l2/l3) requires --reg_mode diffusion.")


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
            _, flow = self.model(x, y,
                                alpha_l1=self._val_alpha_l1,
                                alpha_l3=self._val_alpha_l3)
        return flow.float()


    def _ema_update(self):
        """M2: update EMA teacher towards current student weights.
        Called at the start of each train_step, so the update reflects the
        student state after the *previous* optimizer.step() (one-step lag,
        irrelevant for slow decay ~0.999).
        """
        if not self.use_ema or self.teacher is None:
            return
        with torch.no_grad():
            for p_t, p_s in zip(self.teacher.parameters(), self.model.parameters()):
                p_t.data.mul_(self.ema_decay).add_(p_s.data, alpha=1.0 - self.ema_decay)
            for b_t, b_s in zip(self.teacher.buffers(), self.model.buffers()):
                b_t.data.copy_(b_s.data)


    def train_step(self, batch, epoch):
        args, ctx = self.args, self.ctx

        # M2: refresh EMA teacher from last-step student weights (no-op if disabled).
        self._ema_update()

        if self.is_synth:
            x, y, x_seg, y_seg = batch[0], batch[1], batch[2], batch[3]
            x_seg, y_seg = x_seg.to(self.device).long(), y_seg.to(self.device).long()
        else:
            x, y = (batch[0], batch[1]) if args.ds == "OASIS" else batch
            x_seg = y_seg = None

        x, y = x.to(self.device).float(), y.to(self.device).float()

        schedule_max_epoch = int(args.schedule_max_epoch) if int(getattr(args, "schedule_max_epoch", 0)) > 0 else int(args.max_epoch)
        alpha_l1, alpha_l3, warm = ctcf_schedule(epoch=int(epoch), max_epoch=schedule_max_epoch)
        if args.l1_from_start: alpha_l1 = 1.0
        if args.disable_l1: alpha_l1 = 0.0
        if args.disable_l3: alpha_l3 = 0.0
        self._val_alpha_l1 = alpha_l1
        self._val_alpha_l3 = alpha_l3
        W_icon = float(args.w_icon) * warm
        W_jac = float(args.w_jac) * warm

        if self.use_cascade_reg:
            def_xy, flow_xy, bd_xy = self.model(x, y, alpha_l1=alpha_l1, alpha_l3=alpha_l3, return_breakdown=True)
            def_yx, flow_yx, bd_yx = self.model(y, x, alpha_l1=alpha_l1, alpha_l3=alpha_l3, return_breakdown=True)
        else:
            def_xy, flow_xy = self.model(x, y, alpha_l1=alpha_l1, alpha_l3=alpha_l3)
            def_yx, flow_yx = self.model(y, x, alpha_l1=alpha_l1, alpha_l3=alpha_l3)

        with torch.autocast(device_type="cuda", enabled=False):
            L_ncc = 0.5 * (ctx.ncc(def_xy.float(), y.float()) + ctx.ncc(def_yx.float(), x.float())) * args.w_ncc

        L_icon = icon_loss(flow_xy, flow_yx, mode=args.icon_mode) * W_icon
        L_jac = 0.5 * (neg_jacobian_penalty(flow_xy) + neg_jacobian_penalty(flow_yx)) * W_jac

        # M3: cascade-aware regularisation overrides flat L_reg when enabled.
        if self.use_cascade_reg:
            level_weights = (
                ("phi_l1", self.w_reg_l1),
                ("phi_l2_residual", self.w_reg_l2),
                ("delta_l3", self.w_reg_l3),
            )
            L_reg = torch.zeros((), device=self.device, dtype=flow_xy.dtype)
            for key, w in level_weights:
                f_xy, f_yx = bd_xy.get(key), bd_yx.get(key)
                if f_xy is None or f_yx is None or w == 0.0:
                    continue
                L_reg = L_reg + 0.5 * w * (ctx.reg(f_xy) + ctx.reg(f_yx))
            L_reg = L_reg * args.w_reg
        elif self.reg_mode == 'dare':
            L_reg = 0.5 * (self.dare_reg(flow_xy) + self.dare_reg(flow_yx)) * args.w_reg
        elif self.reg_mode == 'elastic':
            mu = self._elastic_mu
            lam = self._elastic_lam
            L_reg = 0.5 * (elastic_loss(flow_xy, mu, lam) + elastic_loss(flow_yx, mu, lam)) * args.w_reg
        else:
            L_reg = 0.5 * (ctx.reg(flow_xy) + ctx.reg(flow_yx)) * args.w_reg

        # M2: EMA consistency on flow (student vs teacher), warm-ramped together with ICON/Jac.
        L_ema = torch.zeros((), device=self.device, dtype=flow_xy.dtype)
        if self.use_ema and self.teacher is not None:
            with torch.no_grad():
                _, flow_xy_t = self.teacher(x, y, alpha_l1=alpha_l1, alpha_l3=alpha_l3)
                _, flow_yx_t = self.teacher(y, x, alpha_l1=alpha_l1, alpha_l3=alpha_l3)
            L_ema = 0.5 * (
                (flow_xy - flow_xy_t.detach()).abs().mean() +
                (flow_yx - flow_yx_t.detach()).abs().mean()
            ) * self.ema_lambda * warm

        loss = L_ncc + L_icon + L_reg + L_jac + L_ema

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
        if self.use_ema:
            logs["ema"] = float(L_ema.item())

        if self.is_synth:
            with torch.no_grad():
                def_seg_xy = self.reg_nearest((x_seg.float(), flow_xy.detach().float()))
                logs["dice_tr"] = float(dice_val(def_seg_xy.long(), y_seg, num_clus=int(args.synth_num_labels)).item())

        return loss, logs


def parse_args():
    p = argparse.ArgumentParser()
    add_common_args(p, include_synth=True)
    p.set_defaults(exp="CTCF")

    p.add_argument("--config", type=str, default="CTCF-CascadeA", choices=list(CONFIGS.keys()), help="Model config key.")
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

    p.add_argument("--l3_iters", type=int, default=None, help="Number of L3 refinement iterations (default: 1).")
    p.add_argument("--l3_unshared", type=int, choices=[0, 1], default=None, help="Override config: use separate L3 weights per iteration (requires l3_iters>1).")
    p.add_argument("--l1_half_res", type=int, choices=[0, 1], default=None, help="Override config: run L1 at half-res instead of quarter-res.")
    p.add_argument("--l2_full_res", type=int, choices=[0, 1], default=None, help="Override config: run L2 at full-res.")
    p.add_argument("--l3_full_res", type=int, choices=[0, 1], default=None, help="Override config: run L3 at full-res.")
    p.add_argument("--l3_svf", type=int, choices=[0, 1], default=None, help="Override config: integrate L3 delta as SVF via scaling-and-squaring.")
    p.add_argument("--l3_num_heads", type=int, default=None, help="M1 Multi-head L3: number of parallel flow heads with per-voxel learned routing (default: 1 = standard single-head).")

    # M2: EMA self-distillation
    p.add_argument("--ema_decay", type=float, default=0.0, help="M2 EMA self-distillation: teacher decay rate (e.g., 0.999). 0 = disabled.")
    p.add_argument("--ema_lambda", type=float, default=0.0, help="M2 EMA self-distillation: weight on student-teacher flow L1 consistency loss. 0 = disabled.")

    # M3: cascade-aware regularization
    p.add_argument("--w_reg_l1", type=float, default=None, help="M3 cascade-aware reg: diffusion weight on raw L1 flow. If any of w_reg_l1/l2/l3 set, replaces uniform w_reg.")
    p.add_argument("--w_reg_l2", type=float, default=None, help="M3 cascade-aware reg: diffusion weight on raw L2 flow.")
    p.add_argument("--w_reg_l3", type=float, default=None, help="M3 cascade-aware reg: diffusion weight on mean L3 delta (raw, before SVF).")

    p.add_argument("--reg_mode", type=str, choices=["diffusion", "dare", "elastic"], default="diffusion",
                    help="Regularization mode: diffusion (default Grad3d), dare (DARE-minimal), elastic (Navier-Cauchy).")
    p.add_argument("--dare_beta", type=float, default=1.0, help="DARE beta: controls adaptive weighting strength.")
    p.add_argument("--elastic_mu", type=float, default=1.0, help="ElasticMorph mu (shear modulus).")
    p.add_argument("--elastic_lam", type=float, default=1.0, help="ElasticMorph lambda (Lame first parameter).")

    p.add_argument("--synth_train_samples", type=int, default=256, help="Number of synthetic training pairs.")
    p.add_argument("--synth_val_samples", type=int, default=32, help="Number of synthetic validation pairs.")
    p.add_argument("--synth_num_labels", type=int, default=36, help="Number of synthetic segmentation labels.")
    p.add_argument("--synth_vol_size", type=int, nargs=3, default=(96, 96, 96), help="Synthetic volume size D H W (each must be divisible by 32).")
    p.add_argument("--synth_flow_max_disp", type=float, default=6.0, help="Max synthetic displacement amplitude in voxels.")
    p.add_argument("--synth_seed", type=int, default=123, help="Base seed for synthetic pair generation.")
    return p.parse_args()


def main():
    args = parse_args()
    if args.w_reg is None:
        args.w_reg = 4.0 if args.ds.upper() == "IXI" else 1.0
    build_loaders = build_synth_loaders if args.ds == "SYNTH" else baseline_loader_builder(args)
    device = setup_device(gpu_id=int(args.gpu), seed=0, deterministic=False)
    runner = Runner(args, device)
    run_train(args=args, runner=runner, build_loaders=build_loaders)


if __name__ == "__main__":
    main()
