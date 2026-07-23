from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from utils.coord_nets import ChebyKANResidual, RandChebyKANResidual, SirenResidual
from utils.field import (
    compose_flows,
    digital_fold_penalty,
    digital_penalty_and_folds,
    integrate_svf,
    jacobian_penalty_and_folds,
    neg_jacobian_penalty,
)
from utils.losses import Grad3d, NCCVxm
from utils.spatial import SpatialTransformer

TTO_MODES = ("none", "disp", "svf", "inr", "kan", "randkan")
TTO_SCHEDULES = ("cosine", "onecycle", "exp", "const")
TTO_STOP_MODES = ("fixed", "topology", "plateau", "both")
TTO_JAC_MODES = ("central", "digital")
_DENSE_MODES = ("disp", "svf")


@dataclass
class TTOConfig:
    """Per-pair refinement of the deformation field with the network weights frozen.

    `mode` selects what is optimised: the displacement itself (disp), a stationary velocity field
    integrated into one (svf), or a coordinate network producing the residual (inr/kan/randkan).
    `stop_mode` may end a run before `steps`, which is always the ceiling.
    `jac_mode` selects the penalty the topology term acts on: 'central' is the legacy
    central-difference detJ (inert once min detJ>0, i.e. on every SVF field we have); 'digital'
    is the hinge on the ten Liu-et-al. determinants, whose gradient is non-zero exactly on the
    voxels that fold digitally. `topo_mask` restricts that penalty to the brain interior.

    Plus:
    - every trained checkpoint depends on the eps=0 knife-edge form
    - 'use_mask' rescales similarity against the regulariser; w_reg would need re-tuning
    - 'inr_chunks' unchunked, a coord net over every voxel needs ~23 GB of activations
    """

    mode: str = "svf"
    steps: int = 200
    lr: float = 0.01
    w_reg: float = 1.0
    w_jac: float = 0.005
    jac_mode: str = "central"
    jac_eps: float = 0.0
    topo_mask: bool = False
    ncc_win: int = 9
    svf_int_steps: int = 7
    lr_schedule: str = "cosine"
    use_mask: bool = False
    inr_hidden: int = 128
    inr_layers: int = 3
    kan_degree: int = 28
    kan_k: int = 12
    kan_layers: tuple[int, ...] = (3, 70, 70, 3)
    inr_chunks: int = 16
    snapshot_at: tuple[int, ...] = field(default_factory=tuple)

    stop_mode: str = "fixed"
    fold_k: float = 1.25
    fold_delta: float = 0.01  # percentage points
    fold_check_every: int = 10
    plateau_window: int = 50
    plateau_rel: float = 0.02

    def __post_init__(self):
        if self.mode not in TTO_MODES:
            raise ValueError(f"Unknown TTO mode '{self.mode}'; expected one of {TTO_MODES}.")
        if self.jac_mode not in TTO_JAC_MODES:
            raise ValueError(f"Unknown TTO jac_mode '{self.jac_mode}'; expected one of {TTO_JAC_MODES}.")
        if self.lr_schedule not in TTO_SCHEDULES:
            raise ValueError(f"Unknown TTO schedule '{self.lr_schedule}'; expected one of {TTO_SCHEDULES}.")
        if self.stop_mode not in TTO_STOP_MODES:
            raise ValueError(f"Unknown TTO stop mode '{self.stop_mode}'; expected one of {TTO_STOP_MODES}.")
        if self.steps < 0:
            raise ValueError(f"TTO steps must be >= 0, got {self.steps}.")
        self.snapshot_at = tuple(sorted({s for s in self.snapshot_at if 0 < s <= self.steps}))

    @property
    def enabled(self) -> bool:
        return self.mode != "none" and self.steps > 0

    @property
    def guards_topology(self) -> bool:
        return self.stop_mode in ("topology", "both")

    @property
    def guards_plateau(self) -> bool:
        return self.stop_mode in ("plateau", "both")

    def slug(self) -> str:
        """Output-directory identifier, e.g. 'tto_svf_s200' or 'tto_svf_s400_topology_digital'."""
        base = f"tto_{self.mode}_s{self.steps}"
        if self.stop_mode != "fixed":
            base = f"{base}_{self.stop_mode}"
        if self.jac_mode != "central":
            base = f"{base}_{self.jac_mode}"
        return base


@dataclass
class TTOResult:
    flow: torch.Tensor
    snapshots: dict[int, torch.Tensor]
    steps_run: int
    stop_reason: str = "fixed"
    fold_budget: float = 0.0
    folds_start: float = 0.0
    folds_end: float = 0.0


def _normalised_coords(shape: tuple[int, int, int], device, dtype) -> torch.Tensor:
    """Flattened [N,3] voxel coordinates mapped to [-1,1] per axis."""
    axes = [torch.linspace(-1.0, 1.0, s, device=device, dtype=dtype) for s in shape]
    grid = torch.stack(torch.meshgrid(*axes, indexing="ij"), dim=-1)
    return grid.reshape(-1, 3)


def _build_coord_net(cfg: TTOConfig) -> nn.Module:
    """Coordinate network for the residual modes; all are zero at initialisation."""
    layers = list(cfg.kan_layers)
    if cfg.mode == "inr":
        return SirenResidual(hidden=cfg.inr_hidden, layers=cfg.inr_layers)
    if cfg.mode == "kan":
        return ChebyKANResidual(layers=layers, degree=cfg.kan_degree)
    return RandChebyKANResidual(layers=layers, degree=84, k=cfg.kan_k)


def _build_scheduler(opt: torch.optim.Optimizer, cfg: TTOConfig):
    """Learning-rate schedule over the TTO steps; None for a constant rate."""
    if cfg.lr_schedule == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.steps)
    if cfg.lr_schedule == "onecycle":
        return torch.optim.lr_scheduler.OneCycleLR(
            opt,
            max_lr=cfg.lr,
            total_steps=cfg.steps,
            div_factor=1,
            final_div_factor=10,
            pct_start=0.5,
        )
    if cfg.lr_schedule == "exp":
        return torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 0.025 ** min(s / max(cfg.steps, 1), 1.0))
    return None


class _FieldParameterisation(nn.Module):
    """Maps the free parameters to a full-resolution flow, given the cascade's flow0.
    Every mode yields exactly flow0 at initialisation, so step 0 reproduces the cascade.
    """

    def __init__(self, cfg: TTOConfig, flow0: torch.Tensor, st: SpatialTransformer):
        super().__init__()
        self.cfg = cfg
        self.st = st
        self.shape = tuple(flow0.shape[2:])
        self.register_buffer("flow0", flow0.detach(), persistent=False)

        if cfg.mode in _DENSE_MODES:
            self.theta = nn.Parameter(torch.zeros_like(flow0))
            self.net = None
            self.coords = None
            return

        self.theta = None
        self.net = _build_coord_net(cfg).to(flow0.device)
        self.register_buffer("coords", _normalised_coords(self.shape, flow0.device, flow0.dtype), persistent=False)

    def _coord_residual(self) -> torch.Tensor:
        chunks = torch.chunk(self.coords, self.cfg.inr_chunks, dim=0)
        if torch.is_grad_enabled():
            parts = [checkpoint(self.net, c, use_reentrant=False) for c in chunks]
        else:
            parts = [self.net(c) for c in chunks]
        return torch.cat(parts, dim=0).reshape(1, *self.shape, 3).permute(0, 4, 1, 2, 3)

    def forward(self) -> torch.Tensor:
        if self.cfg.mode == "disp":
            return self.flow0 + self.theta
        if self.cfg.mode == "svf":
            # model.py::_merge_l3's chain verbatim: this repo carries two align_corners conventions.
            delta = integrate_svf(self.theta, self.st, steps=self.cfg.svf_int_steps)
            return compose_flows(delta, self.flow0)
        return self.flow0 + self._coord_residual()


def _is_plateau(sim_hist: list[float], cfg: TTOConfig) -> bool:
    """True once the similarity term's recent gain is a negligible share of its total gain."""
    if len(sim_hist) <= cfg.plateau_window:
        return False
    total_gain = sim_hist[0] - sim_hist[-1]
    window_gain = sim_hist[-1 - cfg.plateau_window] - sim_hist[-1]
    return total_gain > 0.0 and window_gain < cfg.plateau_rel * total_gain


def refine_flow(
    flow0: torch.Tensor,
    moving: torch.Tensor,
    fixed: torch.Tensor,
    cfg: TTOConfig,
    st_bilinear: SpatialTransformer,
    mask: torch.Tensor | None = None,
) -> TTOResult:
    """Optimise the deformation field for one pair, starting from the cascade's flow0.
    `st_bilinear` must be the warp the evaluation path uses, so the loss scores what the metrics will.
    Iteration `step` evaluates the field after `step` updates, so iteration 0 is flow0 itself.
    """
    if not cfg.enabled:
        return TTOResult(flow=flow0, snapshots={}, steps_run=0)

    param = _FieldParameterisation(cfg, flow0, st_bilinear).to(flow0.device)
    opt = torch.optim.Adam(param.parameters(), lr=cfg.lr)
    sched = _build_scheduler(opt, cfg)

    ncc = NCCVxm(win=[cfg.ncc_win] * 3)
    reg = Grad3d(penalty="l2")
    loss_mask = mask if cfg.use_mask else None
    topo_mask = mask if cfg.topo_mask else None
    every = max(cfg.fold_check_every, 1)

    snapshots: dict[int, torch.Tensor] = {}
    sim_hist: list[float] = []
    admissible, steps_run = flow0, 0
    budget = folds_start = folds_end = 0.0
    stop_reason = "fixed"

    for step in range(cfg.steps + 1):
        flow = param()
        warped = st_bilinear(moving, flow)
        sim = ncc(fixed, warped, mask=loss_mask)
        loss = sim + cfg.w_reg * reg(flow)

        # The strict fold count is expensive; pay for it only on the scheduled steps. It is needed
        # to guard topology, and to report folds while the digital penalty is what shapes the field.
        on_schedule = step % every == 0 or step == cfg.steps
        measure = on_schedule and (cfg.guards_topology or cfg.jac_mode == "digital")
        folds = 0.0
        if cfg.jac_mode == "digital":
            if measure:
                pen, folds = digital_penalty_and_folds(flow, mask=topo_mask, eps=cfg.jac_eps)
            elif cfg.w_jac > 0.0:
                pen = digital_fold_penalty(flow, mask=topo_mask, eps=cfg.jac_eps)
            else:
                pen = None
        elif measure:
            pen, folds = jacobian_penalty_and_folds(flow, mask=loss_mask, eps=cfg.jac_eps)
        elif cfg.w_jac > 0.0:
            pen = neg_jacobian_penalty(flow, mask=loss_mask, eps=cfg.jac_eps)
        else:
            pen = None
        if pen is not None and cfg.w_jac > 0.0:
            loss = loss + cfg.w_jac * pen

        if step == 0 and cfg.guards_topology:
            folds_start = folds
            budget = max(cfg.fold_k * folds, folds + cfg.fold_delta)

        if cfg.guards_topology and measure and folds > budget:
            stop_reason = "topology"
            break

        if measure or not cfg.guards_topology:
            # With a guard, only a field whose topology was just verified may be returned.
            admissible, steps_run = flow.detach(), step
        if measure:
            folds_end = folds
        if step in cfg.snapshot_at:
            snapshots[step] = flow.detach()

        sim_hist.append(float(sim.detach()))
        if cfg.guards_plateau and _is_plateau(sim_hist, cfg):
            stop_reason = "plateau"
            break

        if step == cfg.steps:
            break

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if sched is not None:
            sched.step()

    return TTOResult(
        flow=admissible,
        snapshots=snapshots,
        steps_run=steps_run,
        stop_reason=stop_reason,
        fold_budget=budget,
        folds_start=folds_start,
        folds_end=folds_end,
    )
