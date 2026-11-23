"""
Train-time augmentations. Fixes over FYP v1 (scripts/step3_augment.py):
  * undetected (all-zero) blocks are never altered — v1's flip turned them into x=1.0
  * mirroring swaps left/right hands and left/right pose landmarks
  * no clipping to [0, 1] — v1 destroyed negative z values
  * applied on the fly with a new draw every epoch instead of fixed files on disk
"""

import numpy as np

from ...registry import TRANSFORMS
from ..layout import POSE_LR_PAIRS, join_blocks, present, split_blocks


@TRANSFORMS.register("mirror")
class Mirror:
    """Horizontal mirror for raw MediaPipe coordinates (x in [0, 1])."""

    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, x, rng):
        if rng.random() >= self.p:
            return x
        left, right, pose = (b.copy() for b in split_blocks(x))
        for block in (left, right, pose):
            m = present(block)
            block[m, :, 0] = 1.0 - block[m, :, 0]
        perm = np.arange(pose.shape[1])
        for a, b in POSE_LR_PAIRS:
            perm[a], perm[b] = b, a
        return join_blocks(right, left, pose[:, perm])


@TRANSFORMS.register("time_warp")
class TimeWarp:
    """Random smooth speed change: resample along a monotone warped time axis."""

    def __init__(self, p: float = 0.5, strength: float = 0.2, knots: int = 4):
        self.p, self.strength, self.knots = p, strength, knots

    def __call__(self, x, rng):
        T = len(x)
        if rng.random() >= self.p or T < 4:
            return x
        speed = np.clip(rng.normal(1.0, self.strength, self.knots + 1), 0.3, 3.0)
        speed = np.interp(np.linspace(0, self.knots, T), np.arange(self.knots + 1), speed)
        t = np.cumsum(speed)
        t = (t - t[0]) / (t[-1] - t[0]) * (T - 1)
        return x[np.clip(t.round().astype(int), 0, T - 1)]


@TRANSFORMS.register("speed")
class Speed:
    """Global speed change by a random factor (clip gets shorter/longer)."""

    def __init__(self, p: float = 0.5, low: float = 0.8, high: float = 1.25):
        self.p, self.low, self.high = p, low, high

    def __call__(self, x, rng):
        if rng.random() >= self.p or len(x) < 2:
            return x
        n = max(2, int(round(len(x) / rng.uniform(self.low, self.high))))
        return x[np.linspace(0, len(x) - 1, n).round().astype(int)]


@TRANSFORMS.register("affine")
class Affine:
    """Random scale / rotation / shift of x,y. Use after normalize_body (origin = shoulders)."""

    def __init__(self, p: float = 0.8, scale: float = 0.1, rotate_deg: float = 10.0, shift: float = 0.05):
        self.p, self.scale, self.rot, self.shift = p, scale, np.deg2rad(rotate_deg), shift

    def __call__(self, x, rng):
        if rng.random() >= self.p:
            return x
        s = rng.uniform(1 - self.scale, 1 + self.scale)
        a = rng.uniform(-self.rot, self.rot)
        R = s * np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]], np.float32)
        t = rng.uniform(-self.shift, self.shift, 2).astype(np.float32)
        left, right, pose = (b.copy() for b in split_blocks(x))
        for block in (left, right, pose):
            m = present(block)
            block[m, :, :2] = block[m, :, :2] @ R.T + t
            block[m, :, 2] *= s
        return join_blocks(left, right, pose)


@TRANSFORMS.register("gaussian_noise")
class GaussianNoise:
    """Small jitter on detected landmarks only (visibility untouched)."""

    def __init__(self, p: float = 0.5, sigma: float = 0.01):
        self.p, self.sigma = p, sigma

    def __call__(self, x, rng):
        if rng.random() >= self.p:
            return x
        left, right, pose = (b.copy() for b in split_blocks(x))
        for block in (left, right, pose):
            m = present(block)
            block[m, :, :3] += rng.normal(0, self.sigma, block[m, :, :3].shape).astype(np.float32)
        return join_blocks(left, right, pose)


@TRANSFORMS.register("hand_dropout")
class HandDropout:
    """Blank one hand for a short random span (simulates detection loss)."""

    def __init__(self, p: float = 0.3, max_frac: float = 0.2):
        self.p, self.max_frac = p, max_frac

    def __call__(self, x, rng):
        if rng.random() >= self.p or len(x) < 2:
            return x
        x = x.copy()
        span = max(1, int(len(x) * rng.uniform(0, self.max_frac)))
        start = rng.integers(0, len(x) - span + 1)
        cols = slice(0, 63) if rng.random() < 0.5 else slice(63, 126)
        x[start:start + span, cols] = 0.0
        return x
