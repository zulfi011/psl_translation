"""
Transforms: (T, F) numpy array → (T', F') array. Configured as an ordered list:

    transforms:
      - {name: mirror, train_only: true, params: {p: 0.5}}
      - {name: fill_missing_hands}
      - {name: normalize_body}
      - {name: resample, params: {length: 60}}

Entries with train_only are random augmentations applied to training samples
only, re-drawn every epoch (on the fly, never saved to disk).
"""

import numpy as np

from ...registry import TRANSFORMS
from . import augment, normalize, temporal  # noqa: F401  (registers transforms)


class Pipeline:
    def __init__(self, spec: list[dict], train: bool, seed: int | None = None):
        self.steps = [TRANSFORMS.build(s["name"], **s.get("params", {}))
                      for s in spec if train or not s.get("train_only", False)]
        self.rng = np.random.default_rng(seed)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        for step in self.steps:
            x = step(x, self.rng)
        return x.astype(np.float32)
