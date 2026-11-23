"""
dataset.py
===========
PyTorch Dataset for loading pre-extracted .npy landmark files.
Used by step4_train.py and step5_evaluate.py.
Not run directly.

Each .npy file is a (60, 258) float32 array.
Labels come from the sub-folder names, sorted alphabetically → integer index.
"""

import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset


class SignDataset(Dataset):
    """
    Loads all .npy files from a split folder (train / val / test).

    Parameters
    ----------
    root      : str  – path to the split folder, e.g. "data/train"
    label_map : dict – optional pre-built {"label": index} dict.
                       If None, it is built from the sub-folder names.
    """

    def __init__(self, root, label_map=None):
        self.data      = []
        self.labels    = []

        # build label map from sorted sub-folder names if not provided
        if label_map is None:
            label_names    = sorted(os.listdir(root))
            self.label_map = {name: i for i, name in enumerate(label_names)}
        else:
            self.label_map = label_map

        # load every .npy file
        for label, idx in self.label_map.items():
            label_path = os.path.join(root, label)
            if not os.path.isdir(label_path):
                continue   # skip if this class has no folder in this split

            for fname in os.listdir(label_path):
                if not fname.endswith(".npy"):
                    continue
                arr = np.load(os.path.join(label_path, fname))
                self.data.append(arr)
                self.labels.append(idx)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # returns tensor (60, 258) and integer label
        x = torch.tensor(self.data[idx], dtype=torch.float32)
        y = self.labels[idx]
        return x, y


def load_label_map(path=None):
    """Helper to load the label map saved by step2_split_dataset.py"""
    if path is None:
        from .config import LABELS_JSON
        path = LABELS_JSON
    with open(path) as f:
        return json.load(f)
