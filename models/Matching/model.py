"""T1 registration with a frozen parent, learned descriptors and field refinement."""

from __future__ import annotations

import torch
from torch import nn

from utils.field import legacy_sampling_displacement

from .sampling import flow_inside_fraction, sample_voxels, voxel_grid
from .solver import matching_energy, refine


@torch.no_grad()
def base_flow(model, moving, fixed, precision="bf16"):
    channels = model.cfg.image_channels
    x, y = [value[:, :1].expand(-1, channels, -1, -1, -1) for value in (moving, fixed)]
    with torch.autocast(
        device_type=moving.device.type,
        dtype=torch.bfloat16 if precision == "bf16" else torch.float16,
        enabled=moving.is_cuda and precision in ("fp16", "bf16"),
    ):
        flow = model(x, y, alpha_l1=1.0, alpha_l3=1.0)[1]
    return legacy_sampling_displacement(flow.float())


@torch.no_grad()
def overlap_support(moving, fixed, initial):
    positions = voxel_grid(initial.shape[2:], device=initial.device) + initial
    support = (fixed > 0) & (sample_voxels(moving, positions) > 0)
    if not bool(support.any()):
        raise ValueError("T1 overlap support is empty at the initial field")
    return support.float()


class Matching(nn.Module):
    def __init__(self, encoder, solver_config, *, base=None, base_precision="bf16"):
        super().__init__()
        self.encoder = encoder
        self.base = base
        self.base_precision = base_precision
        self.solver_config = dict(solver_config)
        if self.base is not None:
            self.base.requires_grad_(False).eval()

    def train(self, mode=True):
        super().train(mode)
        if self.base is not None:
            self.base.eval()
        return self

    def initial_field(self, moving, fixed):
        if self.base is None:
            raise ValueError("Matching requires a parent model or an explicit initial field")
        return base_flow(self.base, moving, fixed, self.base_precision)

    def forward(
        self,
        moving,
        fixed,
        initial=None,
        *,
        mask=None,
        features=None,
        steps=None,
        create_graph=None,
        return_diagnostics=False,
    ):
        """Refine a native-grid field; optional features are ordered (moving, fixed)."""
        if initial is None:
            initial = self.initial_field(moving, fixed)
        if create_graph is None:
            create_graph = self.training and torch.is_grad_enabled()
        with torch.set_grad_enabled(create_graph):
            fm, ff = (self.encoder(moving), self.encoder(fixed)) if features is None else features
        if mask is None:
            # Freeze overlap at the initial field for all solver iterates.
            mask = overlap_support(moving, fixed, initial)
        config = self.solver_config if steps is None else self.solver_config | {"steps": steps}
        end = refine(ff, fm, initial, mask=mask, create_graph=create_graph, **config)
        if not return_diagnostics:
            return end
        energy_args = {key: config[key] for key in ("reg_weight", "outside_weight")}
        with torch.no_grad():
            diagnostics = {
                "inside_fraction": float(flow_inside_fraction(end)),
                "energy_before": float(matching_energy(ff, fm, initial, mask=mask, **energy_args)),
                "energy_after": float(matching_energy(ff, fm, end, mask=mask, **energy_args)),
                "descriptor_shape": list(ff.shape[2:]),
                "field_shape": list(initial.shape[2:]),
            }
        return end, diagnostics
