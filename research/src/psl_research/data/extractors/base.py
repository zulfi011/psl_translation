from pathlib import Path

import numpy as np


class Extractor:
    """Turns one video into a (T, n_features) float32 array (full length — no resampling)."""

    name: str = "base"
    n_features: int = 0

    def extract(self, video_path: Path) -> np.ndarray:
        raise NotImplementedError

    def close(self):
        pass
