"""A population atlas with nested unconditional, linear and nonlinear Haar velocity priors."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from experiments.population.geometry import (
    HaarLevel,
    HaarPyramid,
    canonical_flow,
    coefficient_bands,
    haar_forward,
    haar_inverse,
    pad_velocity,
)


@dataclass
class PopulationState:
    coefficients: HaarPyramid
    velocity: torch.Tensor
    flow: torch.Tensor
    kl: torch.Tensor
    log_prior: torch.Tensor
    log_q: torch.Tensor


def _normal_log_prob(value: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    value, mean, std = value.double(), mean.double(), std.double()
    terms = -0.5 * ((value - mean) / std).square() - std.log() - 0.5 * math.log(2 * math.pi)
    return terms.flatten(1).sum(dim=1, dtype=torch.float64)


def _normal_kl(q_mean: torch.Tensor, q_std: torch.Tensor, p_mean: torch.Tensor, p_std: torch.Tensor) -> torch.Tensor:
    q_mean, q_std, p_mean, p_std = q_mean.double(), q_std.double(), p_mean.double(), p_std.double()
    log_variance_ratio = 2 * (q_std / p_std).log()
    terms = 0.5 * (log_variance_ratio.expm1() - log_variance_ratio + ((q_mean - p_mean) / p_std).square())
    return terms.flatten(1).sum(dim=1, dtype=torch.float64)


class _MeanHead(nn.Module):
    def __init__(self, inputs: int, width: int, *, output_bias: bool = True):
        super().__init__()
        self.hidden = nn.Sequential(nn.Conv3d(inputs, width, 3, padding=1), nn.SiLU())
        self.output = nn.Conv3d(width, 3, 3, padding=1, bias=output_bias)
        nn.init.zeros_(self.output.weight)
        if self.output.bias is not None:
            nn.init.zeros_(self.output.bias)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.output(self.hidden(inputs))


class _PosteriorHead(nn.Module):
    def __init__(self, inputs: int, width: int, bands: int, scale_logit: float):
        super().__init__()
        self.hidden = nn.Sequential(nn.Conv3d(inputs, width, 3, padding=1), nn.SiLU())
        self.mean = nn.Conv3d(width, 3, 3, padding=1)
        self.scale = nn.Linear(width, 3 * bands)
        nn.init.zeros_(self.mean.weight)
        nn.init.zeros_(self.mean.bias)
        nn.init.zeros_(self.scale.weight)
        nn.init.constant_(self.scale.bias, scale_logit)

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.hidden(inputs)
        return self.mean(hidden), self.scale(hidden.mean(dim=(2, 3, 4)))


class PopulationAtlas(nn.Module):
    """Nested priors: A0=mu(T), A_lin=mu(T)+Lc, A1=mu(T)+Lc+g(T,c)-g(T,0).

    Posterior standard deviations are spatially constant within each band/component but depend on
    the image and coarser samples. The reported deterministic path is a sequence of conditional means,
    not the joint posterior mean. Neither SVF integration nor these priors certify the resulting map.
    Function-class nesting does not imply an ordering of heldout Dice or finite-budget optimization.
    """

    def __init__(
        self,
        template: torch.Tensor,
        *,
        levels: int,
        collar: int,
        width: int,
        conditional: bool | None = None,
        arm: str | None = None,
        scale_min: float,
        scale_max: float,
        prior_std: float | Sequence[float],
        posterior_std: float | Sequence[float],
        integration_steps: int = 7,
    ):
        super().__init__()
        if arm is None:
            if conditional is None:
                raise ValueError("choose arm A0, A_lin or A1")
            arm = "A1" if conditional else "A0"
        if arm not in ("A0", "A_lin", "A1"):
            raise ValueError("arm must be A0, A_lin or A1")
        if conditional is not None and bool(conditional) != (arm != "A0"):
            raise ValueError("conditional alias disagrees with arm")
        if template.ndim != 5 or template.shape[:2] != (1, 1) or template.dtype != torch.float32:
            raise ValueError("template must have shape [1,1,D,H,W] and dtype float32")
        if not torch.isfinite(template).all() or template.std() <= 0:
            raise ValueError("template must have finite, nonzero contrast")
        if type(levels) is not int or levels < 0:
            raise ValueError("levels must be a nonnegative integer")
        if type(collar) is not int or collar < 1 or min(template.shape[2:]) - 2 * collar < 2 or width < 1:
            raise ValueError("positive collar must leave at least two interior voxels; width must be positive")
        prior_stds = (float(prior_std),) * (levels + 1) if isinstance(prior_std, (float, int)) else tuple(prior_std)
        posterior_stds = (
            (float(posterior_std),) * (levels + 1) if isinstance(posterior_std, (float, int)) else tuple(posterior_std)
        )
        if len(prior_stds) == 1:
            prior_stds = prior_stds * (levels + 1)
        if len(posterior_stds) == 1:
            posterior_stds = posterior_stds * (levels + 1)
        if len(prior_stds) != levels + 1 or len(posterior_stds) != levels + 1:
            raise ValueError("coefficient scales must give one value or levels+1 coarse-to-fine stage values")
        scales = (*prior_stds, *posterior_stds)
        if not all(math.isfinite(s) for s in (scale_min, scale_max, *scales)):
            raise ValueError("all coefficient scales must be finite")
        if not 0 < scale_min < min(scales) <= max(scales) < scale_max:
            raise ValueError("initial scales must lie strictly between positive scale_min and scale_max")
        if type(integration_steps) is not int or integration_steps < 0:
            raise ValueError("integration_steps must be a nonnegative integer")
        self.levels, self.collar = levels, collar
        self.arm = arm
        self.conditional = arm != "A0"
        self.scale_min, self.scale_max = scale_min, scale_max
        self.integration_steps = integration_steps
        self.raw_atlas = nn.Parameter(template.detach().clone())
        self.interior_shape = tuple(n - 2 * collar for n in template.shape[2:])
        layout = haar_forward(torch.zeros((1, 3, *self.interior_shape), device=template.device), levels)
        self.level_shapes = tuple(level.shape for level in layout.levels)
        self.band_shapes = tuple(tuple(band.shape[2:]) for band in coefficient_bands(layout))
        self.stage_shapes = (self.band_shapes[0], *reversed(self.level_shapes))
        initial_scales = (prior_stds[0], *(value for std in prior_stds[1:] for value in [std] * 7))
        self.prior_scale_logits = nn.Parameter(
            torch.tensor([self._scale_logit(std) for std in initial_scales])[:, None].expand(-1, 3).clone()
        )
        self.prior_heads = nn.ModuleList([_MeanHead(1, width) for _ in range(levels)])
        # All arms allocate in the same order, preserving identical shared initialization for a given seed.
        self.linear_heads = nn.ModuleList([nn.Conv3d(3, 3, 5, padding=2, bias=False) for _ in range(levels)])
        for head in self.linear_heads:
            nn.init.zeros_(head.weight)
        self.residual_heads = nn.ModuleList([_MeanHead(4, width, output_bias=False) for _ in range(levels)])
        if arm == "A0":
            self.linear_heads.requires_grad_(False)
        if arm != "A1":
            self.residual_heads.requires_grad_(False)
        self.encoder = nn.Sequential(
            nn.Conv3d(2, width, 3, padding=1), nn.SiLU(), nn.Conv3d(width, width, 3, padding=1), nn.SiLU()
        )
        self.posterior_heads = nn.ModuleList(
            [
                _PosteriorHead(width + 3, width, bands, self._scale_logit(std))
                for bands, std in zip((1, *([7] * levels)), posterior_stds, strict=True)
            ]
        )
        self.to(device=template.device)
        self.project_atlas_()

    def _scale_logit(self, std: float) -> float:
        fraction = (std - self.scale_min) / (self.scale_max - self.scale_min)
        return math.log(fraction / (1 - fraction))

    def _scale(self, logits: torch.Tensor) -> torch.Tensor:
        return self.scale_min + (self.scale_max - self.scale_min) * logits.sigmoid()

    def atlas(self) -> torch.Tensor:
        centered = self.raw_atlas - self.raw_atlas.mean()
        return centered / centered.square().mean().sqrt().clamp_min(torch.finfo(centered.dtype).eps)

    @torch.no_grad()
    def project_atlas_(self) -> None:
        self.raw_atlas.copy_(self.atlas())

    def _crop(self, values: torch.Tensor) -> torch.Tensor:
        c = self.collar
        return values[:, :, c:-c, c:-c, c:-c]

    def _empty_bands(self, batch: int) -> list[torch.Tensor]:
        return [self.raw_atlas.new_zeros(batch, 3, *shape) for shape in self.band_shapes]

    def with_bands(self, bands: tuple[torch.Tensor, ...] | list[torch.Tensor]) -> HaarPyramid:
        if len(bands) != len(self.band_shapes):
            raise ValueError("coefficient count does not match the model Haar levels")
        for band, shape in zip(bands, self.band_shapes, strict=True):
            if (
                band.shape != (bands[0].shape[0], 3, *shape)
                or band.dtype != self.raw_atlas.dtype
                or band.device != self.raw_atlas.device
            ):
                raise ValueError("coefficients must match the model Haar shapes, dtype and device")
        records = []
        for fine_index, shape in enumerate(self.level_shapes):
            offset = 1 + 7 * (self.levels - 1 - fine_index)
            records.append(HaarLevel(shape, tuple(bands[offset : offset + 7])))
        return HaarPyramid(bands[0], tuple(records))

    def materialize(self, coefficients: HaarPyramid) -> tuple[torch.Tensor, torch.Tensor]:
        coefficients = self.with_bands(coefficient_bands(coefficients))
        velocity = pad_velocity(haar_inverse(coefficients), self.collar)
        return velocity, canonical_flow(velocity, steps=self.integration_steps)

    def _context(self, bands: list[torch.Tensor], shape: tuple[int, ...]) -> torch.Tensor:
        # Zero detail bands remain in the reconstruction until their stage is sampled.
        return F.adaptive_avg_pool3d(haar_inverse(self.with_bands(bands)), shape)

    def _prior_step(
        self, stage: int, bands: list[torch.Tensor]
    ) -> tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]:
        batch = bands[0].shape[0]
        if stage == 0:
            means = (torch.zeros_like(bands[0]),)
            scales = (self._scale(self.prior_scale_logits[0])[None, :, None, None, None],)
            return means, scales
        shape = self.stage_shapes[stage]
        template = F.adaptive_avg_pool3d(self._crop(self.atlas()), shape).expand(batch, -1, -1, -1, -1)
        prediction = self.prior_heads[stage - 1](template)
        if self.arm != "A0":
            context = self._context(bands, shape)
            prediction = prediction + self.linear_heads[stage - 1](context)
            if self.arm == "A1":
                head = self.residual_heads[stage - 1]
                residual = head(torch.cat((template, context), dim=1)) - head(
                    torch.cat((template, torch.zeros_like(context)), dim=1)
                )
                prediction = prediction + residual
        means = haar_forward(prediction, 1).levels[0].details
        offset = 1 + 7 * (stage - 1)
        scales = tuple(self._scale(self.prior_scale_logits[offset + i])[None, :, None, None, None] for i in range(7))
        return means, scales

    def _posterior_step(
        self, stage: int, features: torch.Tensor, bands: list[torch.Tensor]
    ) -> tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]:
        shape = self.stage_shapes[stage]
        context = self._context(bands, shape)
        inputs = torch.cat((F.adaptive_avg_pool3d(features, shape), context), dim=1)
        prediction, logits = self.posterior_heads[stage](inputs)
        means = (prediction,) if stage == 0 else haar_forward(prediction, 1).levels[0].details
        scales = self._scale(logits.reshape(logits.shape[0], len(means), 3))
        return means, tuple(scales[:, i, :, None, None, None] for i in range(len(means)))

    def sample(self, observed: torch.Tensor, *, stochastic: bool = True) -> PopulationState:
        if observed.ndim != 5 or observed.shape[1:] != self.raw_atlas.shape[1:] or observed.dtype != torch.float32:
            raise ValueError("observed must be float32 [B,1,D,H,W] on the template grid")
        with torch.autocast(device_type=observed.device.type, enabled=False):
            atlas = self.atlas().expand(observed.shape[0], -1, -1, -1, -1)
            features = self.encoder(torch.cat((self._crop(observed), self._crop(atlas)), dim=1))
            bands = self._empty_bands(observed.shape[0])
            kl = observed.new_zeros(observed.shape[0], dtype=torch.float64)
            log_prior, log_q = torch.zeros_like(kl), torch.zeros_like(kl)
            offset = 0
            for stage in range(self.levels + 1):
                means, scales = self._posterior_step(stage, features, bands)
                prior_means, prior_scales = self._prior_step(stage, bands)
                new_bands = []
                for mean, std, p_mean, p_std in zip(means, scales, prior_means, prior_scales, strict=True):
                    value = mean + std * torch.randn_like(mean) if stochastic else mean
                    kl = kl + _normal_kl(mean, std, p_mean, p_std)
                    log_prior = log_prior + _normal_log_prob(value, p_mean, p_std)
                    log_q = log_q + _normal_log_prob(value, mean, std)
                    new_bands.append(value)
                bands[offset : offset + len(means)] = new_bands
                offset += len(means)
            coefficients = self.with_bands(bands)
            velocity, flow = self.materialize(coefficients)
            return PopulationState(coefficients, velocity, flow, kl, log_prior, log_q)

    def posterior_mean(self, observed: torch.Tensor) -> HaarPyramid:
        """Return the autoregressive conditional-mean path, not the joint posterior expectation or mode."""
        return self.sample(observed, stochastic=False).coefficients

    def log_prior(self, coefficients: HaarPyramid) -> torch.Tensor:
        values = coefficient_bands(self.with_bands(coefficient_bands(coefficients)))
        prefix = self._empty_bands(values[0].shape[0])
        result = values[0].new_zeros(values[0].shape[0], dtype=torch.float64)
        with torch.autocast(device_type=values[0].device.type, enabled=False):
            offset = 0
            for stage in range(self.levels + 1):
                means, scales = self._prior_step(stage, prefix)
                selected = values[offset : offset + len(means)]
                for value, mean, std in zip(selected, means, scales, strict=True):
                    result = result + _normal_log_prob(value, mean, std)
                prefix[offset : offset + len(means)] = selected
                offset += len(means)
        return result

    def weight_penalty(self) -> torch.Tensor:
        """Squared norm for a Gaussian hyperprior on generative network weights and scale logits."""
        return sum(parameter.square().sum(dtype=torch.float64) for parameter in self.prior_parameters())

    def prior_parameters(self):
        """Generative parameters frozen during the shared atlas/posterior warmup."""
        yield self.prior_scale_logits
        yield from self.prior_heads.parameters()
        if self.arm != "A0":
            yield from self.linear_heads.parameters()
        if self.arm == "A1":
            yield from self.residual_heads.parameters()
