"""Label-free anatomical targets: soft tissue classes of a 1-D intensity Gaussian mixture.

The mixture is fitted once on foreground voxels of training images (background is exactly 0
in the Learn2Reg OASIS volumes) and its components are ordered by mean intensity, so in T1
they read CSF < GM < WM. No segmentation enters the fit or the posteriors.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
import torch


@dataclass(frozen=True)
class TissueGMM:
    means: np.ndarray
    variances: np.ndarray
    weights: np.ndarray


def fit_tissue_gmm(
    volumes: Iterable[np.ndarray],
    k: int = 3,
    iters: int = 100,
    samples_per_volume: int = 200_000,
    seed: int = 0,
) -> TissueGMM:
    """EM fit of a k-component 1-D mixture to a fixed random subsample of foreground voxels."""
    rng = np.random.default_rng(seed)
    samples = []
    for volume in volumes:
        foreground = np.asarray(volume, dtype=np.float64)[np.asarray(volume) > 0]
        take = min(samples_per_volume, foreground.size)
        samples.append(rng.choice(foreground, size=take, replace=False))
    x = np.concatenate(samples)[:, None]
    if x.size < k:
        raise ValueError("Too few foreground voxels to fit the tissue mixture.")

    means = np.quantile(x, np.linspace(0.15, 0.85, k))
    variances = np.full(k, x.var() / k)
    weights = np.full(k, 1.0 / k)
    for _ in range(iters):
        log_p = -0.5 * ((x - means) ** 2 / variances + np.log(2.0 * math.pi * variances)) + np.log(weights)
        log_p -= log_p.max(axis=1, keepdims=True)
        resp = np.exp(log_p)
        resp /= resp.sum(axis=1, keepdims=True)
        mass = resp.sum(axis=0)
        weights = mass / x.shape[0]
        means = (resp * x).sum(axis=0) / mass
        variances = (resp * (x - means) ** 2).sum(axis=0) / mass + 1e-8

    order = np.argsort(means)
    return TissueGMM(means=means[order], variances=variances[order], weights=weights[order])


def tissue_posteriors(image: torch.Tensor, gmm: TissueGMM) -> torch.Tensor:
    """[B,1,D,H,W] image -> [B,1+k,D,H,W]: channel 0 is background, then soft tissue classes."""
    shape = (1, -1, 1, 1, 1)
    means, variances, weights = (
        torch.as_tensor(values, dtype=image.dtype, device=image.device).view(shape)
        for values in (gmm.means, gmm.variances, gmm.weights)
    )
    log_p = -0.5 * ((image - means) ** 2 / variances + torch.log(2.0 * math.pi * variances)) + torch.log(weights)
    foreground = (image > 0).to(image.dtype)
    return torch.cat([1.0 - foreground, torch.softmax(log_p, dim=1) * foreground], dim=1)


def with_background(probs: torch.Tensor, image: torch.Tensor) -> torch.Tensor:
    """[B,K,...] soft classes -> [B,1+K,...]: channel 0 is the image background (exactly 0)."""
    foreground = (image > 0).to(probs.dtype)
    return torch.cat([1.0 - foreground, probs * foreground], dim=1)


def information_loss(probs: torch.Tensor, image: torch.Tensor) -> torch.Tensor:
    """Mean per-voxel entropy minus entropy of the mean assignment over the foreground, / log K.

    -1 means confident voxels spread evenly over all K classes; 0 means no information (one class or
    uniform guesses). Minimising it keeps a self-taught parcellation from collapsing.
    """
    k = probs.shape[1]
    p = probs.permute(0, 2, 3, 4, 1)[(image[:, 0] > 0)].clamp_min(1e-8)  # [N_fg, K]
    voxel_entropy = -(p * p.log()).sum(dim=1).mean()
    mean = p.mean(dim=0)
    usage_entropy = -(mean * mean.log()).sum()
    return (voxel_entropy - usage_entropy) / math.log(k)


def effective_classes(probs: torch.Tensor, image: torch.Tensor) -> float:
    """exp(entropy of the mean foreground assignment): how many classes are actually in use."""
    mean = probs.permute(0, 2, 3, 4, 1)[(image[:, 0] > 0)].mean(dim=0).clamp_min(1e-8)
    return float(torch.exp(-(mean * mean.log()).sum()))


class LatentParcellation(torch.nn.Module):
    """Small U-Net at half resolution that assigns each voxel to K self-taught classes (pilot P2b)."""

    def __init__(self, classes: int = 16, width: int = 16):
        super().__init__()

        def block(cin: int, cout: int) -> torch.nn.Sequential:
            return torch.nn.Sequential(
                torch.nn.Conv3d(cin, cout, 3, padding=1),
                torch.nn.LeakyReLU(0.2),
                torch.nn.Conv3d(cout, cout, 3, padding=1),
                torch.nn.LeakyReLU(0.2),
            )

        self.enc1, self.enc2, self.enc3 = block(1, width), block(width, 2 * width), block(2 * width, 4 * width)
        self.dec2, self.dec1 = block(6 * width, 2 * width), block(3 * width, width)
        self.head = torch.nn.Conv3d(width, classes, 1)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        half = torch.nn.functional.avg_pool3d(image, 2)
        e1 = self.enc1(half)
        e2 = self.enc2(torch.nn.functional.max_pool3d(e1, 2))
        e3 = self.enc3(torch.nn.functional.max_pool3d(e2, 2))
        d2 = self.dec2(torch.cat([torch.nn.functional.interpolate(e3, size=e2.shape[2:]), e2], dim=1))
        d1 = self.dec1(torch.cat([torch.nn.functional.interpolate(d2, size=e1.shape[2:]), e1], dim=1))
        logits = self.head(d1)
        return torch.nn.functional.interpolate(logits, size=image.shape[2:], mode="trilinear", align_corners=False)


def augment_intensity(image: torch.Tensor, generator: torch.Generator | None = None) -> torch.Tensor:
    """Random gamma in [0.7, 1.4] and Gaussian noise (sigma 0.02) on the foreground only."""
    gamma = 0.7 + 0.7 * torch.rand((), generator=generator, device="cpu").item()
    noise = 0.02 * torch.randn(image.shape, generator=generator, device="cpu").to(image.device, image.dtype)
    foreground = image > 0
    return torch.where(foreground, (image.clamp_min(0) ** gamma + noise).clamp_min(1e-4), image)


# Pinned anatomix (MIT, ICLR 2025): github.com/neel-dey/anatomix @ 4c86aeeca5e4bf712d9e5775a01fadb37b30c13a,
# weights neeldey/anatomix/anatomix.pth on the HuggingFace Hub.
_ANATOMIX_REPO, _ANATOMIX_WEIGHTS = "neeldey/anatomix", "anatomix.pth"


def load_anatomix(device: torch.device) -> torch.nn.Module:
    """The frozen 6M-parameter anatomix U-Net (16 feature channels), in eval mode (pilot P2c)."""
    try:
        from anatomix.model.network import Unet
        from huggingface_hub import hf_hub_download
    except ImportError as error:
        raise ImportError(
            "--anatomy_loss foundation needs anatomix and huggingface_hub: pip install --no-deps "
            '"git+https://github.com/neel-dey/anatomix.git@4c86aeeca5e4bf712d9e5775a01fadb37b30c13a"'
        ) from error
    net = Unet(dimension=3, input_nc=1, output_nc=16, num_downs=4, ngf=16)
    state = torch.load(hf_hub_download(_ANATOMIX_REPO, _ANATOMIX_WEIGHTS), map_location="cpu")
    net.load_state_dict({key.replace("_orig_mod.", ""): value for key, value in state.items()}, strict=True)
    return net.to(device).eval().requires_grad_(False)


def anatomix_features(net: torch.nn.Module, image: torch.Tensor) -> torch.Tensor:
    """Unit-norm (over channels) anatomix features of a [B,1,D,H,W] image rescaled to [0, 1]."""
    scale = image.amax(dim=(1, 2, 3, 4), keepdim=True).clamp_min(1e-8)
    return torch.nn.functional.normalize(net(image / scale), dim=1)
