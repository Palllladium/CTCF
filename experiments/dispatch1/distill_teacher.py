"""Teacher fields of a frozen copy of the starting weights: direct, a second pass, or a bridge through a training
subject chosen by minimax, by the ventricle proxy or at random. Under the eligible-pairs policy a pair is taught only
when the minimax bridge eases its harder leg below the direct field's energy, whatever the teacher kind."""

from __future__ import annotations

import copy
import json
import random
from pathlib import Path

import numpy as np
import torch

from experiments.dispatch1.bridges import EnergyTable, Parent, compose_bridge, energy, nearest_proxy
from utils import pkload
from utils.field import compose_flows, legacy_sampling_displacement

KINDS = ("bridge_minimax", "bridge_proxy", "random_bridge", "direct", "recursion")
PAIRS = ("eligible", "all")


class DistillTeacher:
    def __init__(self, model, kind: str, pairs: str, device, train_files, table: str = "", proxy: str = "", seed=0):
        if kind not in KINDS or pairs not in PAIRS:
            raise ValueError(f"teacher kind must be one of {KINDS} and pairs one of {PAIRS}")
        if (kind == "bridge_minimax" or pairs == "eligible") and not table:
            raise ValueError("the minimax teacher and the eligible-pairs policy need --energy_table")
        if kind == "bridge_proxy" and not proxy:
            raise ValueError("the proxy teacher needs --proxy_values from experiments.dispatch1.eval_bridge")
        self.parent = Parent(copy.deepcopy(model), device)
        self.kind, self.pairs = kind, pairs
        self.table = EnergyTable(table) if table else None
        self.proxy = json.loads(Path(proxy).read_text(encoding="utf-8"))["values"] if proxy else None
        self.files = {Path(f).stem: f for f in train_files}
        self.bank = self.table.bank if self.table else sorted(self.files)
        self.rng = random.Random(seed)
        self._images: dict[str, torch.Tensor] = {}

    def _image(self, subject: str) -> torch.Tensor:
        if subject not in self._images:  # kept on the host: the whole bank on the GPU would grow every worker
            img = pkload(self.files[subject])[0]
            self._images[subject] = torch.from_numpy(np.ascontiguousarray(img)).float()[None, None]
        return self._images[subject].to(self.parent.device)

    def _bridge_subject(self, fixed: str, moving: str) -> str:
        if self.kind == "bridge_minimax":
            return self.table.best(fixed, moving)[0]
        if self.kind == "bridge_proxy":
            return nearest_proxy(self.proxy, fixed, moving, self.bank)
        return self.rng.choice([s for s in self.bank if s not in (fixed, moving)])

    @torch.no_grad()
    def target(self, moving, fixed, moving_subject: str, fixed_subject: str) -> torch.Tensor | None:
        """The teacher field F <- M, or None for a pair the policy does not teach. Warps and compositions run in
        float32 whatever the caller's autocast, as in experiments.dispatch1.eval_bridge."""
        with torch.autocast(device_type=moving.device.type, enabled=False):
            direct = None
            if self.pairs == "eligible" or self.kind in ("direct", "recursion"):
                direct = self.parent.flow(moving, fixed)
            if self.pairs == "eligible" and self.table.best(fixed_subject, moving_subject)[1] >= energy(direct, fixed):
                return None
            if self.kind == "direct":
                return direct
            if self.kind == "recursion":
                second = self.parent.flow(self.parent.warp(moving, direct), fixed)
                return compose_flows(legacy_sampling_displacement(second), direct)
            via = self._image(self._bridge_subject(fixed_subject, moving_subject))
            return compose_bridge(self.parent.flow(via, fixed), self.parent.flow(moving, via))
