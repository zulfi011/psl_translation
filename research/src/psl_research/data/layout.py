"""
Landmark feature layout for the MediaPipe Holistic extractor (258 values / frame).

  [0   – 62 ]  left hand   21 pts × (x, y, z)
  [63  – 125]  right hand  21 pts × (x, y, z)
  [126 – 257]  pose        33 pts × (x, y, z, visibility)

A block that MediaPipe did not detect in a frame is all zeros.
"""

import numpy as np

N_HAND, N_POSE = 21, 33
LH = slice(0, 63)
RH = slice(63, 126)
POSE = slice(126, 258)
N_FEATURES = 258

# MediaPipe pose landmark pairs (left, right) — swapped when mirroring
POSE_LR_PAIRS = [(1, 4), (2, 5), (3, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16),
                 (17, 18), (19, 20), (21, 22), (23, 24), (25, 26), (27, 28), (29, 30), (31, 32)]
L_SHOULDER, R_SHOULDER = 11, 12


def split_blocks(x: np.ndarray):
    """(T, 258) → views: left (T,21,3), right (T,21,3), pose (T,33,4)."""
    return (x[:, LH].reshape(-1, N_HAND, 3),
            x[:, RH].reshape(-1, N_HAND, 3),
            x[:, POSE].reshape(-1, N_POSE, 4))


def join_blocks(left, right, pose) -> np.ndarray:
    T = left.shape[0]
    return np.concatenate([left.reshape(T, -1), right.reshape(T, -1), pose.reshape(T, -1)], axis=1)


def present(block: np.ndarray) -> np.ndarray:
    """(T, N, C) → (T,) bool: was this block detected in each frame?"""
    return np.any(block != 0, axis=(1, 2))
