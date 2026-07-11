from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


class SirenResidual(nn.Module):
    """Coordinate MLP with sine activations mapping [-1,1]^3 to a voxel-unit displacement."""

    def __init__(self, hidden: int = 128, layers: int = 3, w0: float = 30.0):
        super().__init__()
        self.w0 = w0
        dims = [3] + [hidden] * layers
        self.linears = nn.ModuleList(nn.Linear(dims[i], dims[i + 1]) for i in range(layers))
        self.head = nn.Linear(hidden, 3)

        with torch.no_grad():
            self.linears[0].weight.uniform_(-1.0 / 3, 1.0 / 3)
            for lin in self.linears[1:]:
                bound = math.sqrt(6.0 / lin.in_features) / w0
                lin.weight.uniform_(-bound, bound)
            # Zero head: the residual is exactly 0 at step 0, so TTO starts from the cascade's flow.
            self.head.weight.zero_()
            self.head.bias.zero_()

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        h = coords
        for i, lin in enumerate(self.linears):
            h = lin(h)
            h = torch.sin(self.w0 * h) if i == 0 else torch.sin(h)
        return self.head(h)


# Chebyshev-KAN layers adapted from KAN-IDIR (MIT, (c) 2025 Lomonosov MSU, Laboratory of
# Mathematical Methods of Image Processing): https://github.com/anac0der/KAN-IDIR
class _ChebyKANLinear(nn.Module):
    """Learned linear map over Chebyshev features T_d(tanh(x)), plus a SiLU skip path."""

    def __init__(self, in_dim: int, out_dim: int, degrees: torch.Tensor, coeff_std_div: float = 1.0):
        super().__init__()
        self.out_dim = out_dim
        n_basis = degrees.numel()

        self.cheby_coeffs = nn.Parameter(torch.empty(in_dim, out_dim, n_basis))
        nn.init.normal_(self.cheby_coeffs, mean=0.0, std=1.0 / (in_dim * n_basis * coeff_std_div))
        self.base_weight = nn.Parameter(torch.empty(out_dim, in_dim))
        nn.init.normal_(self.base_weight, mean=0.0, std=1.0 / in_dim)

        self.base_activation: nn.Module = nn.SiLU()
        self.register_buffer("degrees", degrees.view(1, 1, -1).float(), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y_base = F.linear(self.base_activation(x), self.base_weight)
        # arccos' gradient diverges at +-1, so keep tanh's output strictly inside the interval.
        t = torch.clamp(torch.tanh(x), -1 + 1e-6, 1 - 1e-6)
        t = torch.cos(t.acos()[..., None] * self.degrees)
        return torch.einsum("bid,iod->bo", t, self.cheby_coeffs) + y_base


class _ChebyKANBase(nn.Module):
    """Stack of Chebyshev-KAN layers mapping a coordinate to a voxel-unit displacement."""

    def __init__(
        self,
        layers: list[int],
        degrees_per_layer: list[torch.Tensor],
        mult: float,
        coeff_std_div: float,
    ):
        super().__init__()
        self.mult = mult
        self.layers = nn.ModuleList(
            _ChebyKANLinear(i, o, d, coeff_std_div)
            for i, o, d in zip(layers[:-1], layers[1:], degrees_per_layer, strict=True)
        )
        # Coordinates already lie in [-1,1]; squashing them again would waste the first layer.
        self.layers[0].base_activation = nn.Identity()

        with torch.no_grad():
            self.layers[-1].cheby_coeffs.zero_()
            self.layers[-1].base_weight.zero_()

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        x = coords
        for layer in self.layers:
            x = layer(x)
        return x * self.mult


class ChebyKANResidual(_ChebyKANBase):
    """Chebyshev-KAN residual using every degree from 0 to `degree` in every layer."""

    def __init__(self, layers: list[int] | None = None, degree: int = 28, mult: float = 0.2):
        layers = layers or [3, 70, 70, 3]
        degrees = [torch.arange(degree + 1) for _ in range(len(layers) - 1)]
        super().__init__(layers, degrees, mult, coeff_std_div=1.0)


class RandChebyKANResidual(_ChebyKANBase):
    """Chebyshev-KAN residual keeping degree 0 plus `k` degrees drawn once from 1..`degree`.

    The sparse draw reaches high-degree basis functions without paying for the degrees below them.
    """

    def __init__(
        self,
        layers: list[int] | None = None,
        degree: int = 84,
        k: int = 12,
        mult: float = 0.2,
        seed: int | None = 0,
    ):
        layers = layers or [3, 70, 70, 3]
        rng = np.random.default_rng(seed)
        degrees = []
        for _ in range(len(layers) - 1):
            picked = np.sort(rng.choice(np.arange(1, degree + 1), size=k, replace=False))
            degrees.append(torch.from_numpy(np.concatenate(([0], picked))).long())
        super().__init__(layers, degrees, mult, coeff_std_div=5.0)
