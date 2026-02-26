from collections.abc import Sequence

import numpy as np


class RandomFlip:
    def __init__(self, axis=(1, 2, 3)):
        self.axis = tuple(axis)


    def _sample(self):
        return tuple(bool(np.random.randint(0, 2)) for _ in self.axis)


    def _apply(self, img, flags):
        out = img
        for do_flip, ax in zip(flags, self.axis):
            if do_flip:
                out = np.flip(out, axis=ax)
        return out


    def __call__(self, img):
        flags = self._sample()
        if isinstance(img, Sequence):
            return [self._apply(x, flags) for x in img]
        return self._apply(img, flags)


class SegNorm:
    seg_table = np.array(
        [
            0, 2, 3, 4, 5, 7, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18, 24, 26,
            28, 30, 31, 41, 42, 43, 44, 46, 47, 49, 50, 51, 52, 53, 54, 58, 60, 62,
            63, 72, 77, 80, 85, 251, 252, 253, 254, 255,
        ],
        dtype=np.int16,
    )


    def _map_seg(self, seg):
        out = np.zeros_like(seg)
        for i, lbl in enumerate(self.seg_table):
            out[seg == lbl] = i
        return out


    def __call__(self, img):
        if isinstance(img, Sequence):
            out = []
            for k, x in enumerate(img):
                out.append(self._map_seg(x) if k > 0 else x)
            return out
        return self._map_seg(img)


class NumpyType:
    def __init__(self, types):
        self.types = tuple(types)


    def __call__(self, img):
        if isinstance(img, Sequence):
            out = []
            for k, x in enumerate(img):
                t = self.types[min(k, len(self.types) - 1)]
                out.append(x.astype(t))
            return out
        return img.astype(self.types[0])
