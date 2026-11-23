import numpy as np

from ...registry import TRANSFORMS
from ..layout import join_blocks, present, split_blocks


@TRANSFORMS.register("resample")
class Resample:
    """Linear-index resampling to a fixed number of frames."""

    def __init__(self, length: int = 60):
        self.length = length

    def __call__(self, x, rng):
        if len(x) == 0:
            return np.zeros((self.length, x.shape[1]), np.float32)
        idx = np.linspace(0, len(x) - 1, self.length).round().astype(int)
        return x[idx]


@TRANSFORMS.register("fill_missing_hands")
class FillMissingHands:
    """Forward-fill (then back-fill) frames where a hand was not detected — the same
    imputation the live app uses, now applied in training too."""

    def __call__(self, x, rng):
        left, right, pose = (b.copy() for b in split_blocks(x))
        for block in (left, right):
            mask = present(block)
            if not mask.any():
                continue
            idx = np.where(mask, np.arange(len(mask)), -1)
            idx = np.maximum.accumulate(idx)
            first = np.argmax(mask)
            idx[idx < 0] = first
            block[:] = block[idx]
        return join_blocks(left, right, pose)


@TRANSFORMS.register("trim_inactive")
class TrimInactive:
    """Drop leading/trailing frames where no hand is visible (keeps `margin` frames)."""

    def __init__(self, margin: int = 3):
        self.margin = margin

    def __call__(self, x, rng):
        left, right, _ = split_blocks(x)
        active = np.where(present(left) | present(right))[0]
        if len(active) == 0:
            return x
        lo, hi = max(0, active[0] - self.margin), min(len(x), active[-1] + 1 + self.margin)
        return x[lo:hi]
