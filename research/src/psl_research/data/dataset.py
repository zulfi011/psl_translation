"""PyTorch dataset over extracted landmarks (data/interim/<extractor>/<sign>/<clip>.npy)."""

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .. import paths
from .transforms import Pipeline


class LandmarkDataset(Dataset):
    def __init__(self, clip_ids: list[str], label_map: dict[str, int], extractor: str,
                 transforms: list[dict], train: bool, seed: int | None = None,
                 interim_dir: Path | None = None):
        root = (interim_dir or paths.INTERIM) / extractor
        self.items = [(np.load(root / f"{cid}.npy"), label_map[cid.split("/")[0]]) for cid in clip_ids]
        self.clip_ids = clip_ids
        self.pipeline = Pipeline(transforms, train=train, seed=seed)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        x, y = self.items[i]
        return torch.from_numpy(self.pipeline(x)), y

    def feature_dim(self) -> int:
        return self.pipeline(self.items[0][0]).shape[1]
