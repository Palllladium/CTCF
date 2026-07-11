from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from utils.coord_nets import ChebyKANResidual, RandChebyKANResidual, SirenResidual
from utils.field import compose_flows, integrate_svf, neg_jacobian_penalty
from utils.losses import Grad3d, NCCVxm
from utils.spatial import SpatialTransformer

TTO_MODES = ("none", "disp", "svf", "inr", "kan", "randkan")
TTO_SCHEDULES = ("cosine", "onecycle", "exp", "const")
_DENSE_MODES = ("disp", "svf")


@dataclass
class TTOConfig:
    """Per-pair refinement of the deformation field with the network weights frozen.

    `mode` selects what is optimised: the displacement itself (disp), a stationary velocity field
    integrated into one (svf), or a coordinate network producing the residual (inr/kan/randkan).
    """

    mode: str = "svf"
    steps: int = 200
    lr: float = 0.01
    w_reg: float = 1.0
    w_jac: float = 0.005
    # eps > 0 penalises detJ below eps rather than only below 0; every trained checkpoint and
    # reported number depends on the eps=0 form, so it stays the default.
    jac_eps: float = 0.0
    ncc_win: int = 9
    svf_int_steps: int = 7
    lr_schedule: str = "cosine"
    # Masking rescales the similarity term against the regulariser, so w_reg would need re-tuning.
    use_mask: bool = False
    inr_hidden: int = 128
    inr_layers: int = 3
    kan_degree: int = 28
    kan_k: int = 12
    kan_layers: tuple[int, ...] = (3, 70, 70, 3)
    # A coordinate net evaluated on all ~6.9M voxels at once needs ~23 GB of activations; chunking
    # the forward and recomputing chunks in backward trades ~30% time for that memory.
    inr_chunks: int = 16
    snapshot_at: tuple[int, ...] = field(default_factory=tuple)

    def __post_init__(self):
        if self.mode not in TTO_MODES:
            raise ValueError(f"Unknown TTO mode '{self.mode}'; expected one of {TTO_MODES}.")
        if self.lr_schedule not in TTO_SCHEDULES:
            raise ValueError(f"Unknown TTO schedule '{self.lr_schedule}'; expected one of {TTO_SCHEDULES}.")
        if self.steps < 0:
            raise ValueError(f"TTO steps must be >= 0, got {self.steps}.")
        self.snapshot_at = tuple(sorted({s for s in self.snapshot_at if 0 < s <= self.steps}))

    @property
    def enabled(self) -> bool:
        return self.mode != "none" and self.steps > 0

    def slug(self) -> str:
        """Output-directory identifier, e.g. 'tto_svf_s200'."""
        return f"tto_{self.mode}_s{self.steps}"


@dataclass
class TTOResult:
    flow: torch.Tensor
    snapshots: dict[int, torch.Tensor]
    steps_run: int


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
            # Same integrate-then-compose chain the cascade uses in model.py::_merge_l3; exp(v) is
            # a diffeomorphism, so the refinement cannot fold the field by construction.
            delta = integrate_svf(self.theta, self.st, steps=self.cfg.svf_int_steps)
            return compose_flows(delta, self.flow0)
        return self.flow0 + self._coord_residual()


def refine_flow(
    flow0: torch.Tensor,
    moving: torch.Tensor,
    fixed: torch.Tensor,
    cfg: TTOConfig,
    st_bilinear: SpatialTransformer,
    mask: torch.Tensor | None = None,
) -> TTOResult:
    """Optimise the deformation field for one pair, starting from the cascade's flow0.

    `st_bilinear` must be the warp used by the evaluation path, so the loss scores the same
    deformation the metrics will. `mask` is consulted only when `cfg.use_mask` is set.
    """
    if not cfg.enabled:
        return TTOResult(flow=flow0, snapshots={}, steps_run=0)

    param = _FieldParameterisation(cfg, flow0, st_bilinear).to(flow0.device)
    opt = torch.optim.Adam(param.parameters(), lr=cfg.lr)
    sched = _build_scheduler(opt, cfg)

    ncc = NCCVxm(win=[cfg.ncc_win] * 3)
    reg = Grad3d(penalty="l2")
    loss_mask = mask if cfg.use_mask else None
    snapshots: dict[int, torch.Tensor] = {}

    for step in range(1, cfg.steps + 1):
        opt.zero_grad(set_to_none=True)
        flow = param()
        warped = st_bilinear(moving, flow)

        loss = ncc(fixed, warped, mask=loss_mask) + cfg.w_reg * reg(flow)
        if cfg.w_jac > 0.0:
            loss = loss + cfg.w_jac * neg_jacobian_penalty(flow, mask=loss_mask, eps=cfg.jac_eps)

        loss.backward()
        opt.step()
        if sched is not None:
            sched.step()

        if step in cfg.snapshot_at:
            with torch.no_grad():
                snapshots[step] = param().detach()

    with torch.no_grad():
        final = param().detach()
    return TTOResult(flow=final, snapshots=snapshots, steps_run=cfg.steps)
