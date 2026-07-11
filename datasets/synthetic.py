from __future__ import annotations

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from utils import RegisterModel


def _trilinear_to(t: torch.Tensor, size: tuple[int, int, int]) -> torch.Tensor:
    return F.interpolate(t, size=size, mode="trilinear", align_corners=False)


class SyntheticCTCFDataset(Dataset):
    """Deterministic synthetic registration pairs matching the OASIS output contract."""

    def __init__(
        self,
        num_samples: int,
        vol_size: tuple[int, int, int] = (96, 96, 96),
        num_labels: int = 36,
        flow_max_disp: float = 6.0,
        seed: int = 0,
    ):
        self.num_samples = num_samples
        self.vol_size = vol_size
        self.num_labels = num_labels
        self.flow_max_disp = flow_max_disp
        self.seed = seed

        self._st_lin = RegisterModel(img_size=vol_size, mode="bilinear")
        self._st_near = RegisterModel(img_size=vol_size, mode="nearest")

    def __len__(self):
        return self.num_samples

    def _low_res_shape(self) -> tuple[int, int, int]:
        d, h, w = self.vol_size
        return max(6, d // 8), max(6, h // 8), max(6, w // 8)

    def _make_seg(self, g: torch.Generator) -> torch.Tensor:
        lr = self._low_res_shape()
        logits = torch.randn((1, self.num_labels, *lr), generator=g) * 0.8

        for c in range(self.num_labels):
            zz, yy, xx = (torch.randint(low=0, high=s, size=(1,), generator=g).item() for s in lr)
            logits[0, c, zz, yy, xx] += 6.0

        return _trilinear_to(logits, self.vol_size).argmax(dim=1, keepdim=True).long()

    def _make_img_from_seg(self, seg: torch.Tensor, g: torch.Generator) -> torch.Tensor:
        lr = self._low_res_shape()

        lut = torch.linspace(start=0.0, end=1.0, steps=self.num_labels)
        lut = lut[torch.randperm(self.num_labels, generator=g)]

        img = lut[seg.squeeze(0).squeeze(0)].unsqueeze(0).unsqueeze(0)
        lf_noise = _trilinear_to(torch.randn((1, 1, *lr), generator=g), self.vol_size)
        return torch.clamp(img + 0.08 * lf_noise, min=0.0, max=1.0)

    def _make_flow(self, g: torch.Generator) -> torch.Tensor:
        lr = self._low_res_shape()

        flow = _trilinear_to(torch.randn((1, 3, *lr), generator=g), self.vol_size)
        flow = flow / (flow.abs().amax() + 1e-6)

        amp = 0.35 + 0.65 * torch.rand(1, generator=g).item()
        return flow * (self.flow_max_disp * amp)

    def __getitem__(self, index):
        g = torch.Generator().manual_seed(self.seed + index)

        x_seg = self._make_seg(g)
        x = self._make_img_from_seg(x_seg, g)
        flow = self._make_flow(g)

        y = self._st_lin((x, flow))
        y_seg = self._st_near((x_seg.float(), flow)).long()

        return x[0], y[0], x_seg[0], y_seg[0]


def build_synth_loaders(args):
    train_ds = SyntheticCTCFDataset(
        num_samples=args.synth_train_samples,
        vol_size=tuple(args.synth_vol_size),
        num_labels=args.synth_num_labels,
        flow_max_disp=args.synth_flow_max_disp,
        seed=args.synth_seed,
    )
    val_ds = SyntheticCTCFDataset(
        num_samples=args.synth_val_samples,
        vol_size=tuple(args.synth_vol_size),
        num_labels=args.synth_num_labels,
        flow_max_disp=args.synth_flow_max_disp,
        seed=args.synth_seed + 100_000,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    return train_loader, val_loader
