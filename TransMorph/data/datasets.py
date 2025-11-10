import random
import numpy as np
import torch
from torch.utils.data import Dataset
from .data_utils import pkload

_SKIPPED_CACHE = set()  # чтобы не спамить одинаковыми путями в лог

def _is_pair_ok(obj):
    """Принимаем только (x, x_seg) или (x, y, x_seg, y_seg)."""
    if isinstance(obj, tuple):
        return (len(obj) == 2) or (len(obj) >= 4)
    return False

def _unpack_x_seg(obj, path):
    """
    Строгое извлечение x и x_seg. Бросаем исключение, если структура не та.
    Возвращаем np.ndarray формата [1, D, H, W] для обоих.
    """
    if not isinstance(obj, tuple):
        raise ValueError(f"Unexpected PKL at {path}: not a tuple")

    if len(obj) >= 4:
        x, x_seg = obj[0], obj[2]
    elif len(obj) == 2:
        x, x_seg = obj[0], obj[1]
    else:
        raise ValueError(f"Unexpected PKL at {path}: tuple len={len(obj)}")

    x     = np.asarray(x)
    x_seg = np.asarray(x_seg)

    # ожидаем [D,H,W] -> [1,D,H,W]
    if x.ndim == 4 and x.shape[0] == 1: x = x[0]
    if x_seg.ndim == 4 and x_seg.shape[0] == 1: x_seg = x_seg[0]
    if x.ndim != 3 or x_seg.ndim != 3:
        raise ValueError(f"Bad ndim in {path}: x.ndim={x.ndim}, x_seg.ndim={x_seg.ndim}")

    x     = x[None, ...]                           # [1,D,H,W]
    x_seg = x_seg.astype(np.int16, copy=False)[None, ...]
    return x, x_seg

def _safe_load_pair(path):
    """
    Пытаемся строго распаковать PKL. Если структура не подходит — кидаем пометку в лог (один раз) и возвращаем None.
    """
    try:
        obj = pkload(path)
        if not _is_pair_ok(obj):
            if path not in _SKIPPED_CACHE:
                print(f"[SKIP] Bad PKL structure (expected (x,x_seg) or (x,y,x_seg,y_seg)): {path}")
                _SKIPPED_CACHE.add(path)
            return None
        return _unpack_x_seg(obj, path)
    except Exception as e:
        if path not in _SKIPPED_CACHE:
            print(f"[SKIP] Failed to read {path}: {type(e).__name__}: {e}")
            _SKIPPED_CACHE.add(path)
        return None

def _pick_other_index(cur_idx, n):
    """Случайный другой индекс (не cur_idx)."""
    if n < 2:
        raise RuntimeError("Нужно >=2 файлов для формирования пары (x,y)")
    j = random.randrange(0, n-1)
    if j >= cur_idx:
        j += 1
    return j

class OASISBrainDataset(Dataset):
    """
    Робастный датасет: принимает только корректные PKL. Плохие файлы пропускаем, обучение не останавливаем.
    """
    def __init__(self, data_path, transforms):
        # отфильтруем плохие файлы заранее
        paths = list(data_path)
        good = []
        for p in paths:
            obj = None
            try:
                obj = pkload(p)
            except Exception as e:
                print(f"[SKIP] Failed to open {p}: {e}")
                continue
            if _is_pair_ok(obj):
                good.append(p)
            else:
                print(f"[SKIP] Bad PKL structure (expected (x,x_seg) or (x,y,x_seg,y_seg)): {p}")

        if len(good) < 2:
            raise RuntimeError(f"Too few valid PKLs: {len(good)}. "
                               f"Нужно >=2 для формирования пар. Проверь данные.")
        self.paths = good
        self.transforms = transforms

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        n = len(self.paths)
        src_path = self.paths[index]

        max_tries = 10
        for _ in range(max_tries):
            tar_idx = _pick_other_index(index, n)
            tar_path = self.paths[tar_idx]

            pair_x  = _safe_load_pair(src_path)
            pair_y  = _safe_load_pair(tar_path)
            if (pair_x is not None) and (pair_y is not None):
                x, x_seg = pair_x
                y, y_seg = pair_y
                break
        else:
            raise RuntimeError(f"Failed to form a valid pair after {max_tries} tries. "
                               f"src={src_path}")

        x, x_seg = self.transforms([x, x_seg])
        y, y_seg = self.transforms([y, y_seg])

        x     = torch.from_numpy(np.ascontiguousarray(x))
        y     = torch.from_numpy(np.ascontiguousarray(y))
        x_seg = torch.from_numpy(np.ascontiguousarray(x_seg))
        y_seg = torch.from_numpy(np.ascontiguousarray(y_seg))
        return x, y, x_seg, y_seg


class OASISBrainInferDataset(Dataset):
    """
    Инференс-датасет: ждём корректные PKL; можно ослабить, если нужно.
    """
    def __init__(self, data_path, transforms):
        self.paths = list(data_path)
        self.transforms = transforms
        if len(self.paths) == 0:
            raise RuntimeError("Empty infer path list")

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        obj = pkload(path)
        if not _is_pair_ok(obj):
            raise ValueError(f"Bad PKL for infer: {path}")
        # (x,y,x_seg,y_seg) или (x,x_seg)
        if len(obj) >= 4:
            x, y, x_seg, y_seg = obj[0], obj[1], obj[2], obj[3]
        else:
            x, x_seg = obj[0], obj[1]
            y, y_seg = x, x_seg

        def _ensure_b1dhw(a):
            a = np.asarray(a)
            if a.ndim == 4 and a.shape[0] == 1: a = a[0]
            if a.ndim != 3:
                raise ValueError(f"Infer: expected [D,H,W], got shape {a.shape} at {path}")
            return a[None, ...]
        x = _ensure_b1dhw(x)
        y = _ensure_b1dhw(y)
        x_seg = _ensure_b1dhw(x_seg).astype(np.int16, copy=False)
        y_seg = _ensure_b1dhw(y_seg).astype(np.int16, copy=False)

        x, x_seg = self.transforms([x, x_seg])
        y, y_seg = self.transforms([y, y_seg])

        x     = torch.from_numpy(np.ascontiguousarray(x))
        y     = torch.from_numpy(np.ascontiguousarray(y))
        x_seg = torch.from_numpy(np.ascontiguousarray(x_seg))
        y_seg = torch.from_numpy(np.ascontiguousarray(y_seg))
        return x, y, x_seg, y_seg
