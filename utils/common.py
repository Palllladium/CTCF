from __future__ import annotations

import pickle

import numpy as np


def pkload(fname: str):
    with open(fname, "rb") as f:
        return pickle.load(f)


def validate_subject_pairs(pairs, subjects, name="Validation") -> None:
    subjects = set(subjects)
    if (
        len(pairs) != len(subjects)
        or len(set(pairs)) != len(pairs)
        or {a for a, _ in pairs} != subjects
        or {b for _, b in pairs} != subjects
        or any(a == b or (b, a) not in pairs for a, b in pairs)
    ):
        raise ValueError(f"{name} must pair each subject once in both directions without foreign subjects")


class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0
        self.vals: list[float] = []
        self.std = 0.0

    def update(self, val: float, n: int = 1):
        val = float(val)
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / max(1, self.count)
        self.vals.append(val)
        self.std = float(np.std(self.vals)) if len(self.vals) > 1 else 0.0
