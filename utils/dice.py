from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np
import torch
from torch import nn

OASIS_VOI_LABELS = tuple(range(1, 36))
_IXI_MISSING_LABELS = frozenset({4, 17, 19, 24, 33, 35})
IXI_VOI_LABELS = tuple(i for i in range(1, 37) if i not in _IXI_MISSING_LABELS)


def dice_val(y_pred: torch.Tensor, y_true: torch.Tensor, num_clus: int) -> torch.Tensor:
    """Mean multi-class Dice over one-hot tensors."""
    y_pred = nn.functional.one_hot(y_pred, num_classes=num_clus)
    y_pred = torch.squeeze(y_pred, dim=1).permute(0, 4, 1, 2, 3).contiguous()
    y_true = nn.functional.one_hot(y_true, num_classes=num_clus)
    y_true = torch.squeeze(y_true, dim=1).permute(0, 4, 1, 2, 3).contiguous()
    intersection = (y_pred * y_true).sum(dim=[2, 3, 4])
    union = y_pred.sum(dim=[2, 3, 4]) + y_true.sum(dim=[2, 3, 4])
    dsc = (2.0 * intersection) / (union + 1e-5)
    return torch.mean(torch.mean(dsc, dim=1))


def dice_val_subset(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    labels: Sequence[int],
) -> torch.Tensor:
    """Mean Dice over a fixed label subset on [B,1,D,H,W] tensors."""
    vals = []
    for lbl in labels:
        pred_i = y_pred == lbl
        true_i = y_true == lbl
        inter = (pred_i & true_i).sum().float()
        union = pred_i.sum().float() + true_i.sum().float()
        vals.append((2.0 * inter) / (union + 1e-5))
    return torch.stack(vals).mean()


def dice_per_label(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    labels: Sequence[int] = OASIS_VOI_LABELS,
) -> np.ndarray:
    """Per-label Dice for a single [B=1,C=1,D,H,W] segmentation pair."""
    pred = y_pred.detach().cpu().numpy()[0, 0]
    true = y_true.detach().cpu().numpy()[0, 0]
    out = np.zeros((len(labels),), dtype=np.float64)
    for i, lbl in enumerate(labels):
        p = pred == lbl
        t = true == lbl
        inter = np.sum(p & t)
        union = np.sum(p) + np.sum(t)
        out[i] = (2.0 * inter) / (union + 1e-5)
    return out


def hd95_mean_labels(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    labels: Iterable[int] = OASIS_VOI_LABELS,
    spacing: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> float:
    """Mean robust Hausdorff(95%) across given labels for [B=1,C=1,D,H,W]."""
    from surface_distance import compute_robust_hausdorff, compute_surface_distances

    pred = y_pred.detach().cpu().numpy()[0, 0]
    true = y_true.detach().cpu().numpy()[0, 0]
    vals = []
    for lbl in labels:
        p = pred == lbl
        t = true == lbl
        if p.sum() == 0 or t.sum() == 0:
            vals.append(0.0)
            continue
        sd = compute_surface_distances(t, p, spacing)
        vals.append(float(compute_robust_hausdorff(sd, percent=95.0)))
    return float(np.mean(vals))
