"""Coarse-to-fine matching with deferred (D3) or early (E3) hypothesis reduction.

D3 carries up to three modes, queries 27 neighbours per mode and uses local NMS
plus two spatial compatibility passes. E3 keeps one mode and queries 81 offsets.
Both use the pinned im2grid encoder and tied positional projections and return canonical voxel displacements (z,y,x).
Modes propagate by nearest-parent displacement; their indices are not interpolated.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

try:
    from models.Im2grid.vfa_port.model import Im2gridDecoder, Im2gridEncoder
except ModuleNotFoundError:
    Im2gridDecoder, Im2gridEncoder = nn.Module, None

MODES = ("D3", "E3")
NEG = -1e9  # stands in for -inf in log-masses so that sums stay finite; masked again before every softmax


def window_offsets(kind: str) -> torch.Tensor:
    """27: the 3x3x3 cube; 81: the 81 integer offsets nearest the origin (Euclidean, ties by lexicographic order)."""
    r = torch.arange(-3, 4, dtype=torch.float32)
    cube = torch.stack(torch.meshgrid(r, r, r, indexing="ij"), dim=-1).reshape(-1, 3)
    if kind == "27":
        return cube[(cube.abs() <= 1).all(-1)]
    order = sorted(range(len(cube)), key=lambda i: (float((cube[i] ** 2).sum()), tuple(cube[i].tolist())))
    return cube[order[:81]]


def sample_at(volume: torch.Tensor, coords: torch.Tensor, padding_mode="zeros") -> torch.Tensor:
    """Trilinear values of volume (B, C, D, H, W) at index coordinates (B, ..., 3) -> (B, C, ...)."""
    shape = volume.shape[2:]
    scale = torch.tensor([2.0 / (n - 1) for n in shape], device=coords.device, dtype=coords.dtype)
    grid = (coords * scale - 1.0)[..., [2, 1, 0]]
    flat = grid.reshape(grid.shape[0], -1, 1, 1, 3)
    # Keep fractional coordinates in FP32 even when the encoder runs under AMP.
    with torch.autocast(device_type=volume.device.type, enabled=False):
        out = F.grid_sample(
            volume.float(), flat.float(), mode="bilinear", align_corners=True, padding_mode=padding_mode
        )
    return out.reshape(volume.shape[0], volume.shape[1], *coords.shape[1:-1])


def identity_coords(shape, device) -> torch.Tensor:
    axes = [torch.arange(n, device=device, dtype=torch.float32) for n in shape]
    return torch.stack(torch.meshgrid(*axes, indexing="ij"), dim=-1)  # (D, H, W, 3)


def selected_displacement(coords, logmass):
    best = logmass.argmax(dim=1, keepdim=True)
    chosen = coords.gather(1, best[..., None].expand(-1, -1, *coords.shape[2:5], 3))[:, 0]
    return (chosen - identity_coords(chosen.shape[1:4], chosen.device)).permute(0, 4, 1, 2, 3)


class MultiModeDecoder(Im2gridDecoder):
    """The im2grid decoder with early or deferred reduction of correspondence modes."""

    def __init__(self, mode, levels, start, max_channels, matching_channels, hypotheses, compat_sigma, compat_weight):
        if Im2gridEncoder is None:
            raise RuntimeError("im2grid source is missing: python -m models.Im2grid.fetch")
        super().__init__(levels, matching_channels, start, max_channels, skip={0})
        self.mode, self.levels = mode, levels
        self.k = hypotheses if mode == "D3" else 1
        self.compat_sigma, self.compat_weight = compat_sigma, compat_weight
        self.register_buffer("window", window_offsets("27" if mode == "D3" else "81"), persistent=False)
        self.offsets = tuple(tuple(int(value) for value in offset) for offset in self.window.tolist())
        self.radius = 1 if mode == "D3" else 2
        self.diagnostics = {}

    def _level(self, level, fixed_feat, moving_feat, coords, logmass, lineage=None):
        """One matching level. coords (B, K, D, H, W, 3), logmass (B, K, D, H, W) -> the same shapes."""
        b, k = coords.shape[:2]
        spatial = coords.shape[2:5]
        window = self.window.to(coords.device)
        w = len(window)
        parent_maps = coords.permute(0, 1, 5, 2, 3, 4).reshape(b * k, 3, *spatial)
        local_positions = identity_coords(spatial, coords.device)[None, None] + window.view(1, w, 1, 1, 1, 3)
        parent_positions = sample_at(parent_maps, local_positions.expand(b * k, -1, *spatial, 3), "border")
        positions = parent_positions.permute(0, 2, 3, 4, 5, 1).reshape(b, k * w, *spatial, 3)

        warped = sample_at(moving_feat, coords, "border").permute(0, 2, 1, 3, 4, 5)
        warped = warped.reshape(b * k, moving_feat.shape[1], *spatial)
        padded = F.pad(warped, (self.radius,) * 6)
        moving_pe = self.get_positional_embedding(moving_feat, padding=self.radius).repeat_interleave(k, 0)
        keys = self.positional_embedding[level](padded, moving_pe)
        query = self.positional_embedding[level](fixed_feat, self.get_positional_embedding(fixed_feat))
        features = torch.stack(
            [
                keys[
                    ...,
                    self.radius + int(z) : self.radius + int(z) + spatial[0],
                    self.radius + int(y) : self.radius + int(y) + spatial[1],
                    self.radius + int(x) : self.radius + int(x) + spatial[2],
                ]
                for z, y, x in self.offsets
            ],
            dim=2,
        ).reshape(b, k, keys.shape[1], w, *spatial)
        query_tokens = query.permute(0, 2, 3, 4, 1)[:, None, ..., None, :]
        key_tokens = features.permute(0, 1, 4, 5, 6, 3, 2)
        logits = torch.matmul(query_tokens, key_tokens.transpose(-1, -2))
        logits = logits.squeeze(-2).permute(0, 1, 5, 2, 3, 4).float()
        scores = (logmass[:, :, None] + logits).reshape(b, k * w, *spatial)
        local_positions = local_positions.expand(b, -1, *spatial, 3)

        new_coords, new_mass, new_lineage = [], [], []
        available = scores.clone()
        represented = torch.zeros_like(scores, dtype=torch.bool)
        for i in range(self.k):
            if i == 1:  # every further hypothesis must be a peak of its parent's window, else its slot stays absent
                available = available.masked_fill(~self._peaks(logits), NEG)
            best = available.argmax(dim=1, keepdim=True)  # (B, 1, D, H, W)
            best_score = available.gather(1, best)
            pos = positions.gather(1, best[..., None].expand(-1, -1, *spatial, 3))
            parent = torch.div(best, w, rounding_mode="floor")
            parent_logits = logits.gather(1, parent[:, :, None].expand(-1, -1, w, *spatial))[:, 0]
            local_peak = local_positions.gather(1, (best % w)[..., None].expand(-1, -1, *spatial, 3))
            near = (local_positions - local_peak).abs().amax(-1) <= 1.0
            masked = parent_logits.masked_fill(~near, NEG)
            weights = torch.softmax(masked, dim=1)
            local_mean = (weights[..., None] * local_positions).sum(1)
            composed = sample_at(parent_maps, local_mean.repeat_interleave(k, 0), "border")
            composed = composed.permute(0, 2, 3, 4, 1).reshape(b, k, *spatial, 3)
            refined = composed.gather(1, parent[..., None].expand(-1, -1, *spatial, 3))[:, 0]
            mass = logmass.gather(1, parent)[:, 0] + torch.logsumexp(masked, dim=1)
            active = best_score[:, 0] > NEG / 2
            new_coords.append(refined)
            new_mass.append(torch.where(active, mass, torch.full_like(mass, NEG)))
            if lineage is not None:
                ancestor = lineage.gather(1, parent)[:, 0]
                new_lineage.append(torch.where(active, ancestor, torch.full_like(ancestor, -1)))
            suppress = (positions - pos).abs().amax(-1) <= 1.0
            represented |= suppress & active[:, None]
            available = available.masked_fill(suppress, NEG)
        for i in range(1, self.k):
            for j in range(i):
                duplicate = (new_coords[i] - new_coords[j]).square().sum(-1) < 0.5**2
                duplicate &= (new_mass[i] > NEG / 2) & (new_mass[j] > NEG / 2)
                fraction = torch.sigmoid(new_mass[i] - new_mass[j])[..., None]
                merged = new_coords[j] + fraction * (new_coords[i] - new_coords[j])
                new_coords[j] = torch.where(duplicate[..., None], merged, new_coords[j])
                if lineage is not None:
                    choose_i = duplicate & (new_mass[i] > new_mass[j])
                    new_lineage[j] = torch.where(choose_i, new_lineage[i], new_lineage[j])
                    new_lineage[i] = new_lineage[i].masked_fill(duplicate, -1)
                new_mass[j] = torch.where(duplicate, torch.logaddexp(new_mass[j], new_mass[i]), new_mass[j])
                new_mass[i] = new_mass[i].masked_fill(duplicate, NEG)
        coords = torch.stack(new_coords, 1)
        logmass = torch.stack(new_mass, 1)
        if self.k > 1:
            logmass = self._compatibility(coords, logmass)
        logmass = torch.log_softmax(logmass, dim=1).masked_fill(logmass <= NEG / 2, NEG)
        if not self.training:
            with torch.no_grad():  # candidates no kept mode represents; voxels holding two modes of mass > 0.1
                total = torch.logsumexp(scores, dim=1)
                remaining = scores.masked_fill(represented, NEG)
                self.diagnostics[f"dropped_mass_l{level}"] = float((torch.logsumexp(remaining, 1) - total).exp().mean())
                live = (torch.softmax(logmass, dim=1) > 0.1).sum(1) >= 2
                self.diagnostics[f"live_modes_l{level}"] = float(live.float().mean())
        return coords, logmass, torch.stack(new_lineage, 1) if lineage is not None else None

    @torch.no_grad()
    def _lineage_diagnostics(self, full_displacement, chosen_origin, coarse_best, coarse_cells):
        switches = eligible = 0
        jumps = []
        for spatial_axis in (1, 2, 3):
            shared_cell = coarse_cells.diff(dim=spatial_axis) == 0
            switches += int(((chosen_origin.diff(dim=spatial_axis) != 0) & shared_cell).sum())
            eligible += int(shared_cell.sum())
            jumps.append(full_displacement.diff(dim=spatial_axis + 1).norm(dim=1).flatten())
        jumps = torch.cat(jumps)
        self.diagnostics.update(
            coarse_origin_switch_fraction=float((chosen_origin != coarse_best).float().mean()),
            mode_switch_fraction_within_coarse_cell=switches / eligible if eligible else 0.0,
            mode_switch_eligible_edges=float(eligible),
            displacement_jump_mean_voxels=float(jumps.mean()),
            displacement_jump_p95_voxels=float(torch.quantile(jumps, 0.95)),
            displacement_jump_max_voxels=float(jumps.max()),
        )

    @staticmethod
    @torch.no_grad()
    def _peaks(logits: torch.Tensor) -> torch.Tensor:
        """Local maxima in each parent's 3x3x3 window, excluding flat regions."""
        b, k, w = logits.shape[:3]
        spatial = logits.shape[3:]
        cube = logits.permute(0, 1, 3, 4, 5, 2).reshape(-1, 1, 3, 3, 3)
        high = F.max_pool3d(cube, 3, stride=1, padding=1)
        low = -F.max_pool3d(-cube, 3, stride=1, padding=1)
        peak = (cube >= high) & (cube > low)
        return peak.reshape(b, k, *spatial, w).permute(0, 1, 5, 2, 3, 4).reshape(b, k * w, *spatial)

    def _compatibility(self, coords, logmass, passes: int = 2):
        """Mean-field reweighting by the agreement of displacements with the 6 neighbours' hypotheses."""
        disp = coords - identity_coords(coords.shape[2:5], coords.device)
        for _ in range(passes):
            mass = torch.log_softmax(logmass, dim=1)
            messages = []
            count = torch.zeros_like(logmass[:, :1])
            for axis in (2, 3, 4):
                for shift in (1, -1):
                    nb_disp = torch.roll(disp, shifts=shift, dims=axis)
                    nb_mass = torch.roll(mass, shifts=shift, dims=axis)
                    dist2 = ((disp[:, :, None] - nb_disp[:, None]) ** 2).sum(-1)  # (B, K, K, D, H, W)
                    message = torch.logsumexp(nb_mass[:, None] - dist2 / (2 * self.compat_sigma**2), dim=2)
                    valid = torch.ones_like(count)
                    boundary = [slice(None)] * count.ndim
                    boundary[axis] = 0 if shift == 1 else -1
                    valid[tuple(boundary)] = 0
                    messages.append(message * valid)
                    count = count + valid
            update = torch.stack(messages).sum(0) / count.clamp_min(1)
            logmass = torch.where(logmass > NEG / 2, logmass + self.compat_weight * update, logmass)
        return logmass

    def forward(self, f_feats, m_feats):
        self.diagnostics.clear()
        top = f_feats[-1]
        b, spatial = top.shape[0], top.shape[2:]
        coords = identity_coords(spatial, top.device)[None, None].expand(b, self.k, *spatial, 3).clone()
        logmass = torch.full((b, self.k, *spatial), NEG, device=top.device)
        logmass[:, 0] = 0.0
        states, lineage, coarse_best, coarse_cells = [], None, None, None
        for level in range(self.levels, 0, -1):
            coords, logmass, lineage = self._level(level, f_feats[level], m_feats[level], coords, logmass, lineage)
            if not self.training and level == self.levels:
                lineage = torch.arange(self.k, device=top.device).view(1, self.k, 1, 1, 1)
                lineage = lineage.expand(b, self.k, *spatial)
                coarse_best = logmass.argmax(1)
                coarse_cells = (
                    torch.arange(top[0, 0].numel(), device=top.device).reshape(1, *spatial).expand(b, *spatial)
                )
            states.append((level, coords, logmass))
            if level > 1:
                shape = coords.shape[2:5]
                disp = coords - identity_coords(coords.shape[2:5], coords.device)
                scale = coords.new_tensor([(2 * n - 1) / (n - 1) for n in shape])
                disp = disp.repeat_interleave(2, 2).repeat_interleave(2, 3).repeat_interleave(2, 4) * scale
                logmass = logmass.repeat_interleave(2, 2).repeat_interleave(2, 3).repeat_interleave(2, 4)
                coords = disp + identity_coords(disp.shape[2:5], disp.device)
                if lineage is not None:
                    lineage = lineage.repeat_interleave(2, 2).repeat_interleave(2, 3).repeat_interleave(2, 4)
                    coarse_best = coarse_best.repeat_interleave(2, 1).repeat_interleave(2, 2).repeat_interleave(2, 3)
                    coarse_cells = coarse_cells.repeat_interleave(2, 1).repeat_interleave(2, 2).repeat_interleave(2, 3)
        half = selected_displacement(coords, logmass)
        scale = half.new_tensor([(2 * n - 1) / (n - 1) for n in half.shape[2:]]).view(1, 3, 1, 1, 1)
        flow = F.interpolate(half, scale_factor=2, mode="trilinear", align_corners=True) * scale
        if lineage is not None:
            chosen_origin = lineage.gather(1, logmass.argmax(1, keepdim=True))[:, 0]
            self._lineage_diagnostics(half * scale, chosen_origin, coarse_best, coarse_cells)
        return flow, states


class MultiMatch(nn.Module):
    def __init__(
        self,
        mode="D3",
        levels=4,
        start=8,
        max_channels=64,
        matching_channels=8,
        hypotheses=3,
        compat_sigma=1.0,
        compat_weight=1.0,
    ):
        super().__init__()
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        if levels < 1 or hypotheses < 1 or compat_sigma <= 0 or matching_channels < 6:
            raise ValueError("positive levels, hypotheses, sigma and at least six matching channels are required")
        if Im2gridEncoder is None:
            raise RuntimeError("im2grid source is missing: python -m models.Im2grid.fetch")
        self.levels = levels
        self.encoder = Im2gridEncoder(1, levels, start, max_channels)
        self.decoder = MultiModeDecoder(
            mode, levels, start, max_channels, matching_channels, hypotheses, compat_sigma, compat_weight
        )
        self.diagnostics = self.decoder.diagnostics

    def forward(self, moving, fixed):
        factor = 2**self.levels
        if moving.shape != fixed.shape or any(n < 2 * factor or n % factor for n in moving.shape[2:]):
            raise ValueError(f"matching images need equal shapes, axes divisible by {factor} and at least {2 * factor}")
        return self.decoder(self.encoder(fixed), self.encoder(moving))


def level_mismatch(states, fixed: torch.Tensor, moving: torch.Tensor) -> torch.Tensor:
    """Sum over matched levels of the expected squared intensity mismatch under each voxel's hypothesis masses."""
    total = fixed.new_zeros(())
    for level, coords, logmass in states:
        f_l = F.avg_pool3d(fixed, 2**level) if level else fixed
        m_l = F.avg_pool3d(moving, 2**level) if level else moving
        b, k = coords.shape[:2]
        values = sample_at(m_l, coords.reshape(b, k, *coords.shape[2:5], 3))[:, 0]  # (B, K, D, H, W)
        p = torch.softmax(logmass, dim=1)
        total = total + (p * (values - f_l) ** 2).sum(1).mean()
    return total


def level_smoothness(states) -> torch.Tensor:
    """Diffusion of the selected displacement at each matched level."""
    terms = []
    for _, coords, logmass in states:
        flow = selected_displacement(coords, logmass)
        terms.append(sum(flow.diff(dim=axis).square().mean() for axis in (2, 3, 4)) / 3)
    return torch.stack(terms).sum()
