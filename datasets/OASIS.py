import random

import numpy as np
import torch
from torch.utils.data import Dataset

from utils import pkload


class OASISBrainDataset(Dataset):
    def __init__(self, data_path, transforms):
        self.paths = list(data_path)
        self.transforms = transforms
        if len(self.paths) < 2:
            raise RuntimeError("OASISBrainDataset requires at least 2 samples.")


    def __getitem__(self, index):
        src_path = self.paths[index]
        j = random.randrange(len(self.paths) - 1)
        if j >= index:
            j += 1
        tar_path = self.paths[j]

        src = pkload(src_path)
        x, x_seg = src if len(src) == 2 else (src[0], src[2])
        tar = pkload(tar_path)
        y, y_seg = tar if len(tar) == 2 else (tar[0], tar[2])

        x, y = x[None, ...], y[None, ...]
        x_seg, y_seg = x_seg[None, ...], y_seg[None, ...]
        x, x_seg = self.transforms([x, x_seg])
        y, y_seg = self.transforms([y, y_seg])
        x = torch.from_numpy(np.ascontiguousarray(x)).float()
        y = torch.from_numpy(np.ascontiguousarray(y)).float()
        x_seg = torch.from_numpy(np.ascontiguousarray(x_seg)).long()
        y_seg = torch.from_numpy(np.ascontiguousarray(y_seg)).long()
        return x, y, x_seg, y_seg


    def __len__(self):
        return len(self.paths)


class OASISBrainInferDataset(Dataset):
    def __init__(self, data_path, transforms):
        self.paths = list(data_path)
        self.transforms = transforms


    def __getitem__(self, index):
        x, y, x_seg, y_seg = pkload(self.paths[index])
        x, y = x[None, ...], y[None, ...]
        x_seg, y_seg = x_seg[None, ...], y_seg[None, ...]
        x, x_seg = self.transforms([x, x_seg])
        y, y_seg = self.transforms([y, y_seg])
        x = torch.from_numpy(np.ascontiguousarray(x)).float()
        y = torch.from_numpy(np.ascontiguousarray(y)).float()
        x_seg = torch.from_numpy(np.ascontiguousarray(x_seg)).long()
        y_seg = torch.from_numpy(np.ascontiguousarray(y_seg)).long()
        return x, y, x_seg, y_seg


    def __len__(self):
        return len(self.paths)
