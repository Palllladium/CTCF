import numpy as np
import torch
from torch import nn
from scipy.ndimage import gaussian_filter


def dice_val(y_pred: torch.Tensor, y_true: torch.Tensor, num_clus: int) -> torch.Tensor:
    y_pred = nn.functional.one_hot(y_pred, num_classes=num_clus)
    y_pred = torch.squeeze(y_pred, 1).permute(0, 4, 1, 2, 3).contiguous()
    y_true = nn.functional.one_hot(y_true, num_classes=num_clus)
    y_true = torch.squeeze(y_true, 1).permute(0, 4, 1, 2, 3).contiguous()

    intersection = (y_pred * y_true).sum(dim=[2, 3, 4])
    union = y_pred.sum(dim=[2, 3, 4]) + y_true.sum(dim=[2, 3, 4])
    dsc = (2.0 * intersection) / (union + 1e-5)
    return torch.mean(torch.mean(dsc, dim=1))


def dice_val_VOI(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    VOI_lbls = list(range(1, 36))  # 1..35
    pred = y_pred.detach().cpu().numpy()[0, 0, ...]
    true = y_true.detach().cpu().numpy()[0, 0, ...]
    dscs = []
    for i in VOI_lbls:
        pred_i = (pred == i)
        true_i = (true == i)
        inter = np.sum(pred_i & true_i)
        union = np.sum(pred_i) + np.sum(true_i)
        dscs.append((2.0 * inter) / (union + 1e-5))
    return torch.tensor(float(np.mean(dscs)), device=y_pred.device)


def dice_val_substruct(y_pred: torch.Tensor, y_true: torch.Tensor, std_idx: int) -> str:
    with torch.no_grad():
        y_pred_oh = nn.functional.one_hot(y_pred, num_classes=46)
        y_pred_oh = torch.squeeze(y_pred_oh, 1).permute(0, 4, 1, 2, 3).contiguous()
        y_true_oh = nn.functional.one_hot(y_true, num_classes=46)
        y_true_oh = torch.squeeze(y_true_oh, 1).permute(0, 4, 1, 2, 3).contiguous()

    yp = y_pred_oh.detach().cpu().numpy()
    yt = y_true_oh.detach().cpu().numpy()

    line = f"p_{std_idx}"
    for i in range(46):
        pred_clus = yp[0, i, ...]
        true_clus = yt[0, i, ...]
        inter = (pred_clus * true_clus).sum()
        union = pred_clus.sum() + true_clus.sum()
        dsc = (2.0 * inter) / (union + 1e-5)
        line += "," + str(dsc)
    return line


def dice(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    inter = float(np.sum(y_pred * y_true))
    union = float(np.sum(y_pred) + np.sum(y_true))
    return (2.0 * inter) / (union + 1e-5)


def smooth_seg(binary_img: np.ndarray, sigma: float = 1.5, thresh: float = 0.4) -> np.ndarray:
    sm = gaussian_filter(binary_img.astype(np.float32), sigma=sigma)
    return (sm > thresh)