import numpy as np
import torch
from torch.utils.data import Dataset

from utils import pkload


class IXIBrainDataset(Dataset):
    def __init__(self, data_path, atlas_path, transforms):
        self.paths = list(data_path)
        self.transforms = transforms
        self.atlas_x, _ = pkload(atlas_path)


    def __getitem__(self, index):
        y, _ = pkload(self.paths[index])
        x = self.atlas_x
        x, y = x[None, ...], y[None, ...]
        x, y = self.transforms([x, y])
        x = torch.from_numpy(np.ascontiguousarray(x)).float()
        y = torch.from_numpy(np.ascontiguousarray(y)).float()
        return x, y


    def __len__(self):
        return len(self.paths)


class IXIBrainInferDataset(Dataset):
    def __init__(self, data_path, atlas_path, transforms):
        self.paths = list(data_path)
        self.transforms = transforms
        self.atlas_x, self.atlas_seg = pkload(atlas_path)


    def __getitem__(self, index):
        y, y_seg = pkload(self.paths[index])
        x, x_seg = self.atlas_x, self.atlas_seg
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
