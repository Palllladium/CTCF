"""Native-resolution descriptors with a residual pyramid and fine-scale skip."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint


def blur_resize(image, size):
    if tuple(image.shape[2:]) == tuple(size):
        return image
    kernel = image.new_tensor([0.25, 0.5, 0.25])
    channels = image.shape[1]
    for axis in range(3):
        shape = [1, 1, 1]
        shape[axis] = 3
        weight = kernel.reshape(1, 1, *shape).expand(channels, 1, *shape)
        padding = [0] * 6
        padding[2 * (2 - axis) : 2 * (2 - axis) + 2] = [1, 1]
        image = F.conv3d(F.pad(image, padding, mode="replicate"), weight, groups=channels)
    return F.interpolate(image, size=tuple(size), mode="trilinear", align_corners=True)


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        groups = math.gcd(channels, 8)
        self.layers = nn.Sequential(
            nn.Conv3d(channels, channels, 3, padding=1, bias=False),
            nn.GroupNorm(groups, channels),
            nn.GELU(),
            nn.Conv3d(channels, channels, 3, padding=1, bias=False),
            nn.GroupNorm(groups, channels),
        )

    def forward(self, value):
        return F.gelu(value + self.layers(value))


class FeatureEncoder(nn.Module):
    def __init__(self, input_channels=1, width=16, channels=16, stride=1, *, checkpoint_blocks=True, precision="bf16"):
        super().__init__()
        if stride not in (1, 2) or min(input_channels, width, channels) < 1:
            raise ValueError("Positive channel counts and descriptor stride 1 or 2 are required")
        if precision not in ("fp32", "bf16"):
            raise ValueError("Matching convolutions support explicit fp32 or bf16 precision")
        self.stride, self.precision, self.checkpoint_blocks = stride, precision, checkpoint_blocks
        widths = (width, 2 * width, 4 * width, min(6 * width, 128))
        self.stem = nn.Sequential(
            nn.Conv3d(input_channels, 8, 3, padding=1, bias=False),
            nn.GroupNorm(4, 8),
            nn.GELU(),
            nn.Conv3d(8, 8, 3, padding=1, bias=False),
            nn.GroupNorm(4, 8),
            nn.GELU(),
        )
        self.transitions = nn.ModuleList(
            nn.Conv3d(a, b, 1, bias=False) for a, b in zip((8, *widths[:-1]), widths, strict=True)
        )
        self.stages = nn.ModuleList(nn.Sequential(ResidualBlock(c), ResidualBlock(c)) for c in widths)
        self.laterals = nn.ModuleList(nn.Conv3d(c, channels, 1) for c in widths)
        self.fusions = nn.ModuleList(ResidualBlock(channels) for _ in range(3))
        self.native_lateral = nn.Conv3d(8, channels, 1, bias=False)
        self.native_norm = nn.GroupNorm(math.gcd(channels, 8), channels)
        self.head = nn.Conv3d(channels, channels, 1)

    def _run(self, layer, value):
        if self.checkpoint_blocks and self.training and torch.is_grad_enabled():
            return checkpoint(layer, value, use_reentrant=False)
        return layer(value)

    def forward(self, image):
        if image.ndim != 5 or min(image.shape[2:]) < 16:
            raise ValueError("Encoder inputs must be B,C,D,H,W with each spatial size at least sixteen")
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=image.is_cuda and self.precision == "bf16"):
            fine = self._run(self.stem, image)
            value, levels = fine, []
            for projection, stage in zip(self.transitions, self.stages, strict=True):
                value = blur_resize(value, tuple(max(1, n // 2) for n in value.shape[2:]))
                value = self._run(stage, projection(value))
                levels.append(value)
            value = self.laterals[-1](levels[-1])
            for index in (2, 1, 0):
                value = F.interpolate(value, size=levels[index].shape[2:], mode="trilinear", align_corners=True)
                value = self._run(self.fusions[index], value + self.laterals[index](levels[index]))
            value = F.interpolate(value, size=image.shape[2:], mode="trilinear", align_corners=True)
            value = self.head(F.gelu(self.native_norm(value + self.native_lateral(fine))))
        if value.dtype in (torch.float16, torch.bfloat16):
            value = value.float()
        if self.stride == 2:
            value = blur_resize(value, tuple(n // 2 for n in image.shape[2:]))
        return F.normalize(value, dim=1, eps=1e-6)
