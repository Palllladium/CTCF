from typing import Iterable, Sequence

import numpy as np
import torch
from torch import nn


OASIS_VOI_LABELS = tuple(range(1, 36))
IXI_VOI_LABELS = (1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 18, 20, 21, 22, 23, 25, 26, 27, 28, 29, 30, 31, 32, 34, 36)


def dice_val(y_pred: torch.Tensor, y_true: torch.Tensor, num_clus: int) -> torch.Tensor:
    """Compute mean multi-class Dice via one-hot tensors."""
    y_pred = nn.functional.one_hot(y_pred, num_classes=num_clus)
    y_pred = torch.squeeze(y_pred, 1).permute(0, 4, 1, 2, 3).contiguous()
    y_true = nn.functional.one_hot(y_true, num_classes=num_clus)
    y_true = torch.squeeze(y_true, 1).permute(0, 4, 1, 2, 3).contiguous()
    intersection = (y_pred * y_true).sum(dim=[2, 3, 4])
    union = y_pred.sum(dim=[2, 3, 4]) + y_true.sum(dim=[2, 3, 4])
    dsc = (2.0 * intersection) / (union + 1e-5)
    return torch.mean(torch.mean(dsc, dim=1))


def dice_val_subset(y_pred: torch.Tensor, y_true: torch.Tensor, labels: Sequence[int]) -> torch.Tensor:
    """Compute mean Dice over a fixed label subset on [B,1,D,H,W] tensors."""
    vals = []
    for lbl in labels:
        pred_i = y_pred == int(lbl)
        true_i = y_true == int(lbl)
        inter = (pred_i & true_i).sum().float()
        union = pred_i.sum().float() + true_i.sum().float()
        vals.append((2.0 * inter) / (union + 1e-5))
    return torch.stack(vals).mean()


def dice_per_label(y_pred: torch.Tensor, y_true: torch.Tensor, labels: Sequence[int] = OASIS_VOI_LABELS) -> np.ndarray:
    """Compute per-label Dice for a single [B=1,C=1,D,H,W] segmentation pair."""
    pred = y_pred.detach().cpu().numpy()[0, 0]
    true = y_true.detach().cpu().numpy()[0, 0]
    out = np.zeros((len(labels),), dtype=np.float64)
    for i, lbl in enumerate(labels):
        p = pred == int(lbl)
        t = true == int(lbl)
        inter = np.sum(p & t)
        union = np.sum(p) + np.sum(t)
        out[i] = (2.0 * inter) / (union + 1e-5)
    return out


def hd95_mean_labels(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    labels: Iterable[int] = OASIS_VOI_LABELS,
    spacing=(1.0, 1.0, 1.0),
) -> float:
    """Compute mean robust Hausdorff(95%) across given labels for [B=1,C=1,D,H,W]."""
    from surface_distance import compute_robust_hausdorff, compute_surface_distances

    pred = y_pred.detach().cpu().numpy()[0, 0]
    true = y_true.detach().cpu().numpy()[0, 0]
    vals = []
    for lbl in labels:
        p = pred == int(lbl)
        t = true == int(lbl)
        if p.sum() == 0 or t.sum() == 0:
            vals.append(0.0)
        else:
            sd = compute_surface_distances(t, p, spacing)
            vals.append(float(compute_robust_hausdorff(sd, 95.0)))
    return float(np.mean(vals))
