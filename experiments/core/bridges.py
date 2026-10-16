"""Bridges through training subjects for a frozen CTCF parent: selection, composition and the parent itself.

A bridge F <- S <- M is compose_flows(legacy_sampling_displacement(phi_FS), phi_SM), again a stored flow of the
legacy sampler. The minimax rule picks the bank subject whose harder leg has the least energy, the mean field norm
inside the fixed brain: a heuristic that eases the hardest step, not an anatomical distance.
"""

from __future__ import annotations

import math
import random

import numpy as np
import torch
from scipy import ndimage

from experiments.core.inference_runtime import load_checkpoint_state
from experiments.core.model_adapters import get_model_adapter
from utils.field import compose_flows, legacy_sampling_displacement
from utils.spatial import SpatialTransformer

SUB_BANK_SIZE = 64
BANK_SEED = 0
PROXY_FIT_FILES = 20  # training images pooled for the CSF threshold of the ventricle proxy
PROXY_ERODE = 3


def sub_bank(stems, size: int = SUB_BANK_SIZE, seed: int = BANK_SEED) -> list[str]:
    stems = sorted(stems)
    return sorted(random.Random(seed).sample(stems, min(size, len(stems))))


def energy(flow: torch.Tensor, fixed: torch.Tensor) -> float:
    value = float(torch.linalg.vector_norm(flow[0].float(), dim=0)[fixed[0, 0] > 0].mean())
    if not math.isfinite(value):
        raise ValueError("bridge energy is non-finite; check the field and the fixed brain mask")
    return value


def compose_bridge(flow_fs: torch.Tensor, flow_sm: torch.Tensor) -> torch.Tensor:
    """F <- M through S from the stored flows F <- S and S <- M."""
    return compose_flows(legacy_sampling_displacement(flow_fs), flow_sm)


def otsu_three_class(values: np.ndarray, bins: int = 256) -> tuple[float, float]:
    """The two thresholds of an exhaustive three-class Otsu split over histogram bins."""
    hist, edges = np.histogram(values, bins=bins)
    p = hist / hist.sum()
    centers = 0.5 * (edges[1:] + edges[:-1])
    w, m = np.cumsum(p), np.cumsum(p * centers)
    total = m[-1]
    best, best_ij = -1.0, (1, 2)
    for i in range(bins - 2):
        j = np.arange(i + 1, bins - 1)
        w0, m0 = w[i], m[i]
        w1, m1 = w[j] - w0, m[j] - m0
        w2, m2 = 1.0 - w[j], total - m[j]
        var = m0**2 / max(w0, 1e-12) + m1**2 / np.maximum(w1, 1e-12) + m2**2 / np.maximum(w2, 1e-12)
        var = np.where((w1 > 0) & (w2 > 0) & (w0 > 0), var, -1.0)
        k = int(np.argmax(var))
        if var[k] > best:
            best, best_ij = float(var[k]), (i, int(j[k]))
    return float(edges[best_ij[0] + 1]), float(edges[best_ij[1] + 1])


def csf_threshold(images) -> float:
    pooled = np.concatenate([img[img > 0] for img in images])
    return otsu_three_class(pooled)[0]


def ventricle_proxy(img: np.ndarray, threshold: float) -> int:
    """Label-free ventricle size: dark voxels of the eroded brain inside the central half of every axis."""
    brain = ndimage.binary_erosion(img > 0, iterations=PROXY_ERODE)
    box = tuple(slice(n // 4, 3 * n // 4) for n in img.shape)
    return int(((img < threshold) & brain)[box].sum())


def nearest_proxy(values: dict, fixed: str, moving: str, bank) -> str:
    """The bank subject whose proxy is nearest the geometric mean of the pair's proxies, the pair excluded."""
    target = 0.5 * (math.log(max(values[fixed], 1)) + math.log(max(values[moving], 1)))
    candidates = [s for s in bank if s not in (fixed, moving)]
    return min(candidates, key=lambda s: abs(math.log(max(values[s], 1)) - target))


class EnergyTable:
    """E(fixed subject, moving bank subject) and E(fixed bank subject, moving subject) for every training subject."""

    def __init__(self, path):
        with np.load(path, allow_pickle=False) as data:
            self.subjects = [str(s) for s in data["subjects"]]
            self.bank = [str(s) for s in data["bank"]]
            self.to_bank = torch.from_numpy(data["to_bank"]).float()  # (subjects, bank)
            self.from_bank = torch.from_numpy(data["from_bank"]).float()  # (bank, subjects)
        if len(set(self.subjects)) != len(self.subjects) or len(set(self.bank)) != len(self.bank):
            raise ValueError("bridge energy table contains duplicate subjects")
        shape = (len(self.subjects), len(self.bank))
        if self.to_bank.shape != shape or self.from_bank.shape != shape[::-1]:
            raise ValueError("bridge energy table dimensions do not match its subject lists")
        if not torch.isfinite(self.to_bank).all() or not torch.isfinite(self.from_bank).all():
            raise ValueError("bridge energy table contains non-finite values")
        self.row = {s: i for i, s in enumerate(self.subjects)}

    def best(self, fixed: str, moving: str) -> tuple[str, float]:
        """The minimax bridge of F <- M over the bank and its bottleneck energy."""
        scores = torch.maximum(self.to_bank[self.row[fixed]], self.from_bank[:, self.row[moving]]).clone()
        for k, s in enumerate(self.bank):
            if s in (fixed, moving):
                scores[k] = float("inf")
        k = int(torch.argmin(scores))
        if not torch.isfinite(scores[k]):
            raise ValueError(f"no bridge subject remains for {fixed!r}, {moving!r}")
        return self.bank[k], float(scores[k])


class Parent:
    """A frozen CTCF model with the inference forward (every level, AMP) and the legacy bilinear warp."""

    def __init__(self, model: torch.nn.Module, device, load_report=None):
        self.model, self.device, self.load_report = model.eval(), device, load_report
        for p in self.model.parameters():
            p.requires_grad_(False)
        self._warp = None

    @classmethod
    def load(cls, ckpt: str, config: str, l3_svf: bool, time_steps: int, device) -> Parent:
        """Strict load; only the known deterministic buffer may be regenerated."""
        model = get_model_adapter("ctcf").build(time_steps=time_steps, config_key=config, l3_svf=l3_svf).to(device)
        return cls(model, device, load_checkpoint_state(model, ckpt, strict=True))

    @torch.no_grad()
    def flow(self, moving: torch.Tensor, fixed: torch.Tensor) -> torch.Tensor:
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=moving.is_cuda):
            _, flow = self.model(moving.float(), fixed.float(), alpha_l1=1.0, alpha_l3=1.0)
        return flow.float()

    @torch.no_grad()
    def warp(self, image: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
        if self._warp is None:
            self._warp = SpatialTransformer(tuple(flow.shape[2:]), mode="bilinear").to(self.device)
        return self._warp(image.float(), flow)
