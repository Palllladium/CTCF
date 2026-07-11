from __future__ import annotations

import torch
import torch.nn as nn

from models.CTCF.blocks import resize_3d, upsample_flow
from models.CTCF.stages import CoarseFlowNetQuarter, CTCFDCACoreHalf, FlowRefiner3D
from models.TransMorph_DCA.model import SpatialTransformer
from utils.field import compose_flows, integrate_svf


def _build_level2(config, backbone: str) -> nn.Module:
    """Dispatch L2 backbone construction. Wrappers are lazy-imported per branch."""
    if backbone == "swin-dca":
        return CTCFDCACoreHalf(config, time_steps=config.time_steps)
    if backbone == "vxm":
        from models.VoxelMorph.wrapper import VxmCascadeL2

        return VxmCascadeL2(config)
    if backbone == "lku":
        from models.LKUNet.wrapper import LkuNetCascadeL2

        return LkuNetCascadeL2(config)
    if backbone == "mamba":
        from models.MambaMorph.wrapper import MambaMorphCascadeL2

        return MambaMorphCascadeL2(config)
    if backbone == "vmamba":
        from models.VMambaMorph.wrapper import VMambaMorphCascadeL2

        return VMambaMorphCascadeL2(config)
    if backbone == "effm":
        from models.EfficientMorph.wrapper import EffMorphCascadeL2

        return EffMorphCascadeL2(config)
    raise ValueError(f"Unknown backbone: {backbone}")


class CTCFCascadeA(nn.Module):
    """Three-level coarse-to-fine cascade. Composes L1 -> L2 -> L3 into a full-resolution warp.

    When called with `return_breakdown=True`, forward returns an extra dict with per-level
    contributions at their native resolution, intended as the level-isolated signal for
    cascade-aware per-level regularisation:
        * phi_l1 — raw L1 output (None if L1 disabled or alpha_l1 = 0);
        * phi_l2_residual — L2 contribution alone (flow_l2 - flow_l2_init); equals raw flow_l2
          when L1 did not inject anything;
        * delta_l3 — mean of raw L3 head outputs across iterations, BEFORE SVF integration
          (None if L3 disabled or alpha_l3 = 0).
    """

    def __init__(self, config):
        super().__init__()

        self.img_size_full: tuple[int, int, int] = tuple(config.img_size)
        self.use_level1 = config.use_level1
        self.use_level2 = config.use_level2
        self.use_level3 = config.use_level3
        self.backbone = config.backbone

        self.l3_iters = config.l3_iters
        self.l3_svf = config.l3_svf
        self.l1_half_res = config.l1_half_res
        self.l2_full_res = config.l2_full_res
        self.l3_full_res = config.l3_full_res
        self.l3_unshared = config.l3_unshared and self.l3_iters > 1

        self._validate_flags()

        l3_kwargs = {
            "base_ch": config.level3_base_ch,
            "error_mode": config.level3_error_mode,
            "num_heads": config.level3_num_heads,
            "corr_mode": config.level3_corr_mode,
        }

        self.level1 = None
        if self.use_level1:
            self.level1 = CoarseFlowNetQuarter(base_ch=config.level1_base_ch)

        self.level2 = None
        if self.use_level2:
            self.level2 = _build_level2(config, self.backbone)

        self.level3 = None
        if self.use_level3:
            self.level3 = FlowRefiner3D(**l3_kwargs)

        self.level3_extra = None
        if self.l3_unshared and self.use_level3:
            extras = [FlowRefiner3D(**l3_kwargs) for _ in range(self.l3_iters - 1)]
            self.level3_extra = nn.ModuleList(extras)

        self.st_full = SpatialTransformer(self.img_size_full)
        img_size_half = tuple(s // 2 for s in self.img_size_full)
        self.st_half = SpatialTransformer(img_size_half)

    def _validate_flags(self) -> None:
        """Reject configuration combinations the cascade cannot honour."""
        if self.backbone == "swin-dca" and self.l2_full_res:
            raise ValueError(
                "Swin-DCA L2 (CTCFDCACoreHalf) operates on a half-resolution grid by construction; "
                "l2_full_res=True is incompatible with backbone='swin-dca'.",
            )
        if self.l3_iters < 1:
            raise ValueError(f"l3_iters must be >= 1, got {self.l3_iters}.")
        if not self.use_level2:
            raise ValueError("CTCFCascadeA requires use_level2=True; L1/L3 alone are not enough.")

    def _get_l3(self, it: int) -> FlowRefiner3D:
        """Return the L3 module to use at iteration `it`."""
        if it == 0 or self.level3_extra is None:
            return self.level3
        return self.level3_extra[it - 1]

    def _integrate_svf(
        self,
        vel: torch.Tensor,
        st: SpatialTransformer,
        steps: int = 7,
    ) -> torch.Tensor:
        """Integrate a stationary velocity field via scaling-and-squaring."""
        return integrate_svf(vel, st, steps=steps)

    def _apply_l3_delta(
        self,
        delta: torch.Tensor,
        flow_cur: torch.Tensor,
        st: SpatialTransformer,
    ) -> torch.Tensor:
        """Merge L3 residual into the running flow (SVF + composition, or direct addition)."""
        if self.l3_svf:
            delta = self._integrate_svf(delta, st)
            return compose_flows(delta, flow_cur)
        return flow_cur + delta

    def _to_full_flow(
        self,
        flow: torch.Tensor,
        already_full_res: bool,
    ) -> torch.Tensor:
        """Return `flow` at full resolution, upsampling from half-res if necessary."""
        if already_full_res:
            return flow
        return upsample_flow(flow, scale_factor=2)

    def _run_level1(
        self,
        mov_full: torch.Tensor,
        fix_full: torch.Tensor,
        mov_half: torch.Tensor,
        fix_half: torch.Tensor,
        alpha_l1: float,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """Run L1 if enabled. Returns (flow_half_init, phi_l1_raw); both None if skipped."""
        if self.level1 is None or alpha_l1 <= 0.0:
            return None, None

        if self.l1_half_res:
            phi_l1_raw = self.level1(mov_half, fix_half)
            flow_half_init = phi_l1_raw * alpha_l1
            return flow_half_init, phi_l1_raw

        mov_quarter = resize_3d(mov_full, scale_factor=0.25)
        fix_quarter = resize_3d(fix_full, scale_factor=0.25)
        phi_l1_raw = self.level1(mov_quarter, fix_quarter)
        flow_half_init = upsample_flow(phi_l1_raw, scale_factor=2) * alpha_l1
        return flow_half_init, phi_l1_raw

    def _prepare_level2_inputs(
        self,
        mov_full: torch.Tensor,
        fix_full: torch.Tensor,
        mov_half: torch.Tensor,
        fix_half: torch.Tensor,
        flow_half_init: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """Select L2 working resolution and adapt the L1 init flow to it."""
        if not self.l2_full_res:
            return mov_half, fix_half, flow_half_init

        flow_l2_init = None
        if flow_half_init is not None:
            flow_l2_init = upsample_flow(flow_half_init, scale_factor=2)
        return mov_full, fix_full, flow_l2_init

    def _prepare_l3_frame(
        self,
        mov_full: torch.Tensor,
        fix_full: torch.Tensor,
        mov_half: torch.Tensor,
        fix_half: torch.Tensor,
        def_l2: torch.Tensor,
        flow_l2: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, SpatialTransformer, bool]:
        """Pick the resolution at which L3 iterates and prepare matching inputs.

        Returns (def_cur, flow_cur, mov_ref, fix_ref, st, is_full_res) — the inputs to
        `_run_l3_loop` and a flag telling the caller whether the result needs upsampling.
        """
        run_full_res = self.l3_full_res or self.l2_full_res

        if not run_full_res:
            return def_l2, flow_l2, mov_half, fix_half, self.st_half, False

        if self.l2_full_res:
            return def_l2, flow_l2, mov_full, fix_full, self.st_full, True

        flow_cur = upsample_flow(flow_l2, scale_factor=2)
        def_cur = self.st_full(mov_full, flow_cur)
        return def_cur, flow_cur, mov_full, fix_full, self.st_full, True

    def _run_l3_loop(
        self,
        def_cur: torch.Tensor,
        flow_cur: torch.Tensor,
        mov_ref: torch.Tensor,
        fix_ref: torch.Tensor,
        st: SpatialTransformer,
        alpha_l3: float,
        track_breakdown: bool,
    ) -> tuple[torch.Tensor, torch.Tensor | None, int]:
        """Iterate L3 refinement `l3_iters` times at a single resolution."""
        delta_accum: torch.Tensor | None = None
        count = 0

        for it in range(self.l3_iters):
            delta_raw = self._get_l3(it)(def_cur, fix_ref, flow_cur)

            if track_breakdown:
                if delta_accum is None:
                    delta_accum = delta_raw
                else:
                    delta_accum = delta_accum + delta_raw
                count += 1

            flow_cur = self._apply_l3_delta(delta_raw * alpha_l3, flow_cur, st)
            if it < self.l3_iters - 1:
                def_cur = st(mov_ref, flow_cur)

        return flow_cur, delta_accum, count

    def _finalize(
        self,
        def_full: torch.Tensor,
        flow_full: torch.Tensor,
        breakdown: dict[str, torch.Tensor | None] | None,
        phi_l1_raw: torch.Tensor | None,
        flow_l2: torch.Tensor,
        flow_l2_init: torch.Tensor | None,
        delta_l3_accum: torch.Tensor | None,
        l3_count: int,
    ):
        """Pack the breakdown dict (if requested) and return the cascade output."""
        if breakdown is None:
            return def_full, flow_full

        phi_l2_residual = flow_l2
        if flow_l2_init is not None:
            phi_l2_residual = flow_l2 - flow_l2_init

        delta_l3 = None
        if delta_l3_accum is not None:
            delta_l3 = delta_l3_accum / max(1, l3_count)

        breakdown.update(
            phi_l1=phi_l1_raw,
            phi_l2_residual=phi_l2_residual,
            delta_l3=delta_l3,
        )

        return def_full, flow_full, breakdown

    def forward(
        self,
        mov_full: torch.Tensor,
        fix_full: torch.Tensor,
        alpha_l1: float = 1.0,
        alpha_l3: float = 1.0,
        return_breakdown: bool = False,
    ):
        """Run the cascade. Returns `(def_full, flow_full)`, or `(..., breakdown)` when requested."""
        breakdown: dict[str, torch.Tensor | None] | None = {} if return_breakdown else None

        mov_half = resize_3d(mov_full, scale_factor=0.5)
        fix_half = resize_3d(fix_full, scale_factor=0.5)

        flow_half_init, phi_l1_raw = self._run_level1(
            mov_full=mov_full,
            fix_full=fix_full,
            mov_half=mov_half,
            fix_half=fix_half,
            alpha_l1=alpha_l1,
        )

        l2_mov, l2_fix, flow_l2_init = self._prepare_level2_inputs(
            mov_full=mov_full,
            fix_full=fix_full,
            mov_half=mov_half,
            fix_half=fix_half,
            flow_half_init=flow_half_init,
        )

        def_l2, flow_l2 = self.level2(
            l2_mov,
            l2_fix,
            init_flow=flow_l2_init,
            return_all_flows=False,
        )

        delta_l3_accum: torch.Tensor | None = None
        l3_count = 0

        if self.level3 is not None and alpha_l3 > 0.0:
            def_cur, flow_cur, mov_ref, fix_ref, st, is_full = self._prepare_l3_frame(
                mov_full=mov_full,
                fix_full=fix_full,
                mov_half=mov_half,
                fix_half=fix_half,
                def_l2=def_l2,
                flow_l2=flow_l2,
            )
            flow_cur, delta_l3_accum, l3_count = self._run_l3_loop(
                def_cur=def_cur,
                flow_cur=flow_cur,
                mov_ref=mov_ref,
                fix_ref=fix_ref,
                st=st,
                alpha_l3=alpha_l3,
                track_breakdown=breakdown is not None,
            )
            flow_full = self._to_full_flow(flow_cur, already_full_res=is_full)
        else:
            flow_full = self._to_full_flow(flow_l2, already_full_res=self.l2_full_res)

        def_full = self.st_full(mov_full, flow_full)

        return self._finalize(
            def_full=def_full,
            flow_full=flow_full,
            breakdown=breakdown,
            phi_l1_raw=phi_l1_raw,
            flow_l2=flow_l2,
            flow_l2_init=flow_l2_init,
            delta_l3_accum=delta_l3_accum,
            l3_count=l3_count,
        )
