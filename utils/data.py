import random
import numpy as np
import torch
import re

M = 2 ** 32 - 1
_shape = (240, 240, 155)
_zero = torch.tensor([0])


def init_fn(worker: int):
    seed = torch.LongTensor(1).random_().item()
    seed = (seed + worker) % M
    np.random.seed(seed)
    random.seed(seed)


def add_mask(x: torch.Tensor, mask: torch.Tensor, dim: int = 1) -> torch.Tensor:
    mask = mask.unsqueeze(dim)
    shape = list(x.shape)
    shape[dim] += 21
    new_x = x.new_zeros(*shape)
    new_x = new_x.scatter_(dim, mask, 1.0)
    s = [slice(None)] * len(shape)
    s[dim] = slice(21, None)
    new_x[s] = x
    return new_x


def sample(x: np.ndarray, size: int) -> torch.Tensor:
    i = random.sample(range(x.shape[0]), size)
    return torch.tensor(x[i], dtype=torch.int16)


def get_all_coords(stride: int) -> torch.Tensor:
    return torch.tensor(
        np.stack(
            [v.reshape(-1) for v in np.meshgrid(
                *[stride // 2 + np.arange(0, s, stride) for s in _shape],
                indexing="ij"
            )],
            -1
        ),
        dtype=torch.int16
    )


def gen_feats() -> np.ndarray:
    x, y, z = _shape
    feats = np.stack(np.meshgrid(np.arange(x), np.arange(y), np.arange(z), indexing="ij"), -1).astype("float32")
    shape = np.array([x, y, z])
    feats -= shape / 2.0
    feats /= shape
    return feats


def process_label(label_info_path: str = "label_info.txt"):
    seg_table = [
        0, 2, 3, 4, 5, 7, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18, 24, 26,
        28, 30, 31, 41, 42, 43, 44, 46, 47, 49, 50, 51, 52, 53, 54, 58,
        60, 62, 63, 72, 77, 80, 85, 251, 252, 253, 254, 255
    ]
    with open(label_info_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    label_dict = {}
    seg_i = 0
    for seg_label in seg_table:
        for line in lines:
            parts = re.sub(" +", " ", line).split(" ")
            try:
                int(parts[0])
            except Exception:
                continue
            if int(parts[0]) == seg_label:
                label_dict[seg_i] = parts[1]
        seg_i += 1
    return label_dict