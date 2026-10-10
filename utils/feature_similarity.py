from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


def _conv_block(cin: int, cout: int, dropout: float = 0.0) -> nn.Sequential:
    layers: list[nn.Module] = [
        nn.Conv3d(cin, cout, kernel_size=3, padding=1),
        nn.LeakyReLU(0.2),
        nn.Conv3d(cout, cout, kernel_size=3, padding=1),
        nn.LeakyReLU(0.2),
    ]
    if dropout > 0.0:
        layers.append(nn.Dropout3d(p=dropout))
    return nn.Sequential(*layers)


def _resize(t: torch.Tensor, size: tuple[int, ...]) -> torch.Tensor:
    return F.interpolate(t, size=size, mode="trilinear", align_corners=False)


class ContrastiveFeatureNet(nn.Module):
    """3D U-Net run at half resolution: [B,1,D,H,W] image -> unit-norm [B,C,D/2,H/2,W/2] embedding.

    Its encoder matches FeatureAutoencoder's (widths 32/64/128); trained by experiments.train_features with an
    InfoNCE loss over voxels matched by a known deformation.
    """

    def __init__(self, channels: int = 16, widths: tuple[int, int, int] = (32, 64, 128)):
        super().__init__()
        w1, w2, w3 = widths
        self.enc1 = _conv_block(1, w1)
        self.enc2 = _conv_block(w1, w2)
        self.enc3 = _conv_block(w2, w3)
        self.dec2 = _conv_block(w3 + w2, w2)
        self.dec1 = _conv_block(w2 + w1, w1)
        self.head = nn.Conv3d(w1, channels, kernel_size=1)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(F.avg_pool3d(image, kernel_size=2))
        e2 = self.enc2(F.max_pool3d(e1, kernel_size=2))
        e3 = self.enc3(F.max_pool3d(e2, kernel_size=2))
        d2 = self.dec2(torch.cat([_resize(e3, e2.shape[2:]), e2], dim=1))
        d1 = self.dec1(torch.cat([_resize(d2, e1.shape[2:]), e1], dim=1))
        return F.normalize(self.head(d1), dim=1)

    def feature_maps(self, image: torch.Tensor) -> list[torch.Tensor]:
        return [self(image)]


class FeatureAutoencoder(nn.Module):
    """DeepSim-style autoencoder (Czolbe et al., MIDL 2021) without skip connections, run at half resolution.

    Widths follow DeepSim's 3D setting (32/64/128); the three encoder stage outputs are the features.
    Trained by experiments.train_features to reconstruct the half-resolution image.
    """

    def __init__(self, widths: tuple[int, int, int] = (32, 64, 128), dropout: float = 0.1):
        super().__init__()
        w1, w2, w3 = widths
        self.enc1 = _conv_block(1, w1, dropout)
        self.enc2 = _conv_block(w1, w2, dropout)
        self.enc3 = _conv_block(w2, w3, dropout)
        self.dec2 = _conv_block(w3, w2)
        self.dec1 = _conv_block(w2, w1)
        self.head = nn.Conv3d(w1, 1, kernel_size=1)

    def encode(self, image: torch.Tensor) -> list[torch.Tensor]:
        e1 = self.enc1(F.avg_pool3d(image, kernel_size=2))
        e2 = self.enc2(F.max_pool3d(e1, kernel_size=2))
        e3 = self.enc3(F.max_pool3d(e2, kernel_size=2))
        return [e1, e2, e3]

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        e1, e2, e3 = self.encode(image)
        d2 = self.dec2(_resize(e3, e2.shape[2:]))
        d1 = self.dec1(_resize(d2, e1.shape[2:]))
        return torch.sigmoid(self.head(d1))

    def feature_maps(self, image: torch.Tensor) -> list[torch.Tensor]:
        return self.encode(image)


FEATURE_NETS = {"contrastive": ContrastiveFeatureNet, "autoencoder": FeatureAutoencoder}


def save_feature_net(path: str, net: nn.Module, kind: str, config: dict, meta: dict) -> None:
    torch.save({"kind": kind, "config": config, "state_dict": net.state_dict(), "meta": meta}, path)


def load_feature_net(path: str, device: torch.device) -> nn.Module:
    """A trained feature net from save_feature_net, frozen and in eval mode."""
    payload = torch.load(path, map_location="cpu", weights_only=False)
    net = FEATURE_NETS[payload["kind"]](**payload["config"])
    net.load_state_dict(payload["state_dict"], strict=True)
    return net.to(device).eval().requires_grad_(False)


def index_grid(points: torch.Tensor, shape: tuple[int, ...]) -> torch.Tensor:
    """Index coordinates (z, y, x) -> grid_sample's [-1, 1] (x, y, z) at align_corners=True."""
    scale = torch.tensor([n - 1 for n in shape], device=points.device, dtype=points.dtype)
    return (2.0 * points / scale - 1.0)[..., [2, 1, 0]]


def sample_at(volume: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
    """[1,C,D,H,W] sampled trilinearly at index points [..., 3] -> [..., C], zero outside."""
    flat = points.reshape(1, -1, 1, 1, 3)
    values = F.grid_sample(volume, index_grid(flat, volume.shape[2:]), mode="bilinear", align_corners=True)
    return values[0, :, :, 0, 0].transpose(0, 1).reshape(*points.shape[:-1], volume.shape[1])


def level_points(points: torch.Tensor, shape: tuple[int, ...], level_shape: tuple[int, ...]) -> torch.Tensor:
    """Full-resolution index points -> index points of a level whose voxel j covers f voxels per axis and so sits
    at full-resolution f*j + (f-1)/2 (average/max pooling by 2, as in the feature nets)."""
    factor = torch.tensor([n / m for n, m in zip(shape, level_shape, strict=True)], device=points.device)
    return (points - (factor - 1) / 2) / factor


def warped_level_maps(moving_maps: list[torch.Tensor], flow: torch.Tensor) -> list[torch.Tensor]:
    """E(I) o phi on each level's own grid. Level voxel j reads the moving features at the point that
    utils.spatial.SpatialTransformer samples for its full-resolution centre p: (p + u(p)) * N / (N - 1) - 0.5."""
    shape = tuple(flow.shape[2:])
    n = torch.tensor(shape, device=flow.device, dtype=flow.dtype)
    warped = []
    for fmap in moving_maps:
        level = tuple(fmap.shape[2:])
        axes = [
            torch.arange(m, device=flow.device, dtype=flow.dtype) * (s / m) + (s / m - 1) / 2
            for s, m in zip(shape, level, strict=True)
        ]
        centres = torch.stack(torch.meshgrid(*axes, indexing="ij"), dim=-1)
        u = sample_at(flow, centres)
        sampled = (centres + u) * n / (n - 1) - 0.5
        coords = index_grid(level_points(sampled, shape, level), level)[None]
        warped.append(F.grid_sample(fmap, coords, mode="bilinear", align_corners=True))
    return warped


def feature_dissimilarity(
    moving_maps: list[torch.Tensor],
    fixed_maps: list[torch.Tensor],
    foreground: torch.Tensor,
) -> torch.Tensor:
    """Mean over levels of the mean (1 - cosine) over the fixed image's foreground at that level."""
    total = torch.zeros((), device=foreground.device)
    for moving, fixed in zip(moving_maps, fixed_maps, strict=True):
        mask = F.interpolate(foreground.float(), size=fixed.shape[2:], mode="area")[:, 0] > 0.5
        cos = (F.normalize(moving.float(), dim=1) * F.normalize(fixed.float(), dim=1)).sum(dim=1)
        total = total + (1.0 - cos)[mask].mean()
    return total / len(fixed_maps)


class LearnedSimilarity:
    """Registration term: feature dissimilarity between the warped moving image and the fixed image.

    Both modes compare on the feature levels' own grids with the same mask and averaging. "warped_image" takes
    features of the warped image, E(I o phi); "warped_features" samples the moving image's features through the
    same warp, E(I) o phi. The feature net is frozen; the gradient reaches the field through the warp.
    """

    def __init__(self, net: nn.Module, mode: str):
        if mode not in ("warped_image", "warped_features"):
            raise ValueError(f"Unknown feature mode {mode!r}.")
        self.net, self.mode = net, mode

    def __call__(
        self,
        moving: torch.Tensor,
        warped_moving: torch.Tensor | None,
        fixed: torch.Tensor,
        flow: torch.Tensor,
    ) -> torch.Tensor:
        with torch.no_grad():
            fixed_maps = self.net.feature_maps(fixed.float())
        if self.mode == "warped_image":
            moving_maps = self.net.feature_maps(warped_moving.float())
        else:
            with torch.no_grad():
                features = self.net.feature_maps(moving.float())
            moving_maps = warped_level_maps(features, flow.float())
        return feature_dissimilarity(moving_maps, fixed_maps, fixed > 0)
