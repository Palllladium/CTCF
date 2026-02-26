import pickle

import numpy as np


def pkload(fname: str):
    with open(fname, "rb") as f:
        return pickle.load(f)


class AverageMeter:
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
