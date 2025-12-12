import pickle
import numpy as np
import torch
import torch.nn.functional as F


def pkload(fname: str):
    with open(fname, "rb") as f:
        return pickle.load(f)


class AverageMeter:
    """Computes and stores the average and current value."""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0
        self.vals = []
        self.std = 0.0

    def update(self, val, n: int = 1):
        val = float(val)
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / max(1, self.count)
        self.vals.append(val)
        self.std = float(np.std(self.vals)) if len(self.vals) > 1 else 0.0


def pad_image(img: torch.Tensor, target_size):
    """
    Pad 3D image tensor [B,C,D,H,W] to at least target_size (D,H,W) with zeros.
    """
    rows_to_pad = max(target_size[0] - img.shape[2], 0)
    cols_to_pad = max(target_size[1] - img.shape[3], 0)
    slcs_to_pad = max(target_size[2] - img.shape[4], 0)
    return F.pad(img, (0, slcs_to_pad, 0, cols_to_pad, 0, rows_to_pad), "constant", 0)


def write2csv(line: str, name: str):
    with open(name + ".csv", "a", encoding="utf-8") as f:
        f.write(line)
        f.write("\n")