import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from utils import register_model


class SyntheticCTCFDataset(Dataset):
    """
    Deterministic synthetic registration pairs for fast ablation runs.

    Output contract matches OASIS datasets:
      x, y, x_seg, y_seg with shapes [1, D, H, W].
    """

    def __init__(
        self,
        *,
        num_samples: int,
        vol_size=(96, 96, 96),
        num_labels: int = 36,
        flow_max_disp: float = 6.0,
        seed: int = 0,
    ):
        self.num_samples = int(num_samples)
        self.vol_size = tuple(int(v) for v in vol_size)
        self.num_labels = int(num_labels)
        self.flow_max_disp = float(flow_max_disp)
        self.seed = int(seed)

        self._st_lin = register_model(self.vol_size, mode="bilinear")
        self._st_near = register_model(self.vol_size, mode="nearest")

    def __len__(self):
        return self.num_samples

    def _make_seg(self, g: torch.Generator) -> torch.Tensor:
        d, h, w = self.vol_size
        lr = (max(6, d // 8), max(6, h // 8), max(6, w // 8))

        logits = torch.randn((1, self.num_labels, *lr), generator=g) * 0.8

        # Force each class to appear at least locally.
        for c in range(self.num_labels):
            zz = int(torch.randint(0, lr[0], (1,), generator=g).item())
            yy = int(torch.randint(0, lr[1], (1,), generator=g).item())
            xx = int(torch.randint(0, lr[2], (1,), generator=g).item())
            logits[0, c, zz, yy, xx] += 6.0

        logits = F.interpolate(logits, size=self.vol_size, mode="trilinear", align_corners=False)
        seg = logits.argmax(dim=1, keepdim=True).long()  # [1,1,D,H,W]
        return seg

    def _make_img_from_seg(self, seg: torch.Tensor, g: torch.Generator) -> torch.Tensor:
        d, h, w = self.vol_size
        lr = (max(6, d // 8), max(6, h // 8), max(6, w // 8))

        lut = torch.linspace(0.0, 1.0, self.num_labels)
        lut = lut[torch.randperm(self.num_labels, generator=g)]

        img = lut[seg.squeeze(0).squeeze(0)].unsqueeze(0).unsqueeze(0).float()
        lf_noise = torch.randn((1, 1, *lr), generator=g)
        lf_noise = F.interpolate(lf_noise, size=self.vol_size, mode="trilinear", align_corners=False)
        img = torch.clamp(img + 0.08 * lf_noise, 0.0, 1.0)
        return img

    def _make_flow(self, g: torch.Generator) -> torch.Tensor:
        d, h, w = self.vol_size
        lr = (max(6, d // 8), max(6, h // 8), max(6, w // 8))

        flow = torch.randn((1, 3, *lr), generator=g)
        flow = F.interpolate(flow, size=self.vol_size, mode="trilinear", align_corners=False)
        flow = flow / (flow.abs().amax() + 1e-6)

        amp = 0.35 + 0.65 * float(torch.rand(1, generator=g).item())
        flow = flow * (self.flow_max_disp * amp)
        return flow.float()

    def __getitem__(self, index):
        g = torch.Generator().manual_seed(self.seed + int(index))

        x_seg = self._make_seg(g)                      # [1,1,D,H,W]
        x = self._make_img_from_seg(x_seg, g)          # [1,1,D,H,W]
        flow = self._make_flow(g)                      # [1,3,D,H,W]

        y = self._st_lin((x, flow))                    # [1,1,D,H,W]
        y_seg = self._st_near((x_seg.float(), flow)).long()

        x = x[0].float()
        y = y[0].float()
        x_seg = x_seg[0].long()
        y_seg = y_seg[0].long()
        return x, y, x_seg, y_seg


def build_synth_loaders(args):
    train_ds = SyntheticCTCFDataset(
        num_samples=int(args.synth_train_samples),
        vol_size=tuple(args.synth_vol_size),
        num_labels=int(args.synth_num_labels),
        flow_max_disp=float(args.synth_flow_max_disp),
        seed=int(args.synth_seed),
    )
    val_ds = SyntheticCTCFDataset(
        num_samples=int(args.synth_val_samples),
        vol_size=tuple(args.synth_vol_size),
        num_labels=int(args.synth_num_labels),
        flow_max_disp=float(args.synth_flow_max_disp),
        seed=int(args.synth_seed) + 100_000,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=int(args.batch_size),
        shuffle=True,
        num_workers=int(args.num_workers),
        pin_memory=True,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=int(args.num_workers),
        pin_memory=True,
        drop_last=False,
    )
    return train_loader, val_loader
