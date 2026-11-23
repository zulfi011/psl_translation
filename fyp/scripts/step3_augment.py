"""
step3_augment.py
=================
PURPOSE : Apply data augmentation to TRAINING data only to reach ~250 clips
          per sign.  Val and test sets are never touched.
RUN     : python scripts/step3_augment.py

Techniques applied (all operate on the (60, 258) landmark array):
  1. Gaussian noise      – adds tiny random noise to x,y,z values
  2. Time warping        – randomly stretches/compresses the sequence in time
  3. Spatial scaling     – scales x,y coordinates by a random factor near 1.0
  4. Horizontal flip     – mirrors x coordinates (simulates mirrored signing)
  5. Frame dropout       – randomly zeros out 1-3 frames (simulates occlusion)

Each original clip can produce multiple augmented copies.
Augmented files are saved alongside originals in data/split/train/<label>/.

FEATURE LAYOUT REMINDER  (needed for correct augmentation):
  [0   – 62 ]   Left  hand x,y,z  (21 pts)
  [63  – 125]   Right hand x,y,z  (21 pts)
  [126 – 257]   Pose  x,y,z,vis   (33 pts)  ← visibility channel NOT flipped
"""

import os
import numpy as np
import random

# ── paths ──────────────────────────────────────────────────────────────────────
from psl_translator.config import TRAIN_DIR

# ── target clips per sign after augmentation ──────────────────────────────────
TARGET_PER_CLASS = 250

# ── feature layout constants ──────────────────────────────────────────────────
SEQUENCE_LEN    = 60
TOTAL_FEATURES  = 258

# hand x-coordinate column indices (used for horizontal flip)
# Left hand:  cols 0,3,6,...,60   (every 3rd starting at 0, for 21 pts)
# Right hand: cols 63,66,...,123
LEFT_X_COLS  = list(range(0,   63, 3))   # 21 x-coords of left hand
RIGHT_X_COLS = list(range(63, 126, 3))   # 21 x-coords of right hand
# Pose x-coords: cols 126, 130, 134, ...  (every 4th, 33 pts)
POSE_X_COLS  = list(range(126, 258, 4))

SEED = 42
random.seed(SEED)
np.random.seed(SEED)


# ── augmentation functions ─────────────────────────────────────────────────────

def gaussian_noise(seq, sigma=0.005):
    """
    Add small Gaussian noise to all x,y,z values.
    Simulates natural hand tremor and sensor noise.
    sigma=0.005 is subtle enough not to distort the sign.
    """
    noise        = np.random.normal(0, sigma, seq.shape).astype(np.float32)
    augmented    = seq + noise
    # clip to [0,1] so coordinates stay in the MediaPipe normalised range
    augmented    = np.clip(augmented, 0.0, 1.0)
    return augmented


def time_warp(seq, sigma=0.2):
    """
    Randomly stretch or compress the sequence along the time axis.
    A smooth random warp curve is generated, then the sequence is
    re-sampled at the warped time points and resized back to SEQUENCE_LEN.
    Simulates fast and slow signers.
    """
    T       = len(seq)
    # generate a smooth warp path using cumulative random steps
    warp    = np.random.normal(loc=1.0, scale=sigma, size=T)
    warp    = np.clip(warp, 0.2, 2.0)            # prevent negative/huge warps
    warp    = np.cumsum(warp)
    warp    = warp / warp[-1]                     # normalise to [0, 1]
    warp    = (warp * (T - 1)).astype(int)
    warp    = np.clip(warp, 0, T - 1)
    warped  = seq[warp]                           # re-index frames
    # re-sample back to original length
    indices = np.linspace(0, len(warped) - 1, T, dtype=int)
    return warped[indices]


def spatial_scale(seq, scale_range=(0.85, 1.15)):
    """
    Multiply all x and y coordinates by a random scale factor.
    Simulates the signer standing closer or farther from the camera.
    z and visibility values are not changed.
    """
    scale     = random.uniform(*scale_range)
    augmented = seq.copy()

    # scale x-coords of left hand, right hand, and pose
    for col in LEFT_X_COLS + RIGHT_X_COLS + POSE_X_COLS:
        augmented[:, col] *= scale
    # scale y-coords (each x-col + 1 is the y-col)
    y_cols = [c + 1 for c in LEFT_X_COLS + RIGHT_X_COLS + POSE_X_COLS]
    for col in y_cols:
        augmented[:, col] *= scale

    return np.clip(augmented, 0.0, 1.0).astype(np.float32)


def horizontal_flip(seq):
    """
    Mirror x-coordinates around the centre (x → 1 - x).
    Simulates a mirrored view of the signer.
    Left/right hand roles are NOT swapped here because in PSL
    most signs are performed bilaterally or with the dominant hand
    and mirroring is a valid natural variation.
    """
    augmented = seq.copy()
    for col in LEFT_X_COLS + RIGHT_X_COLS + POSE_X_COLS:
        augmented[:, col] = 1.0 - augmented[:, col]
    return augmented.astype(np.float32)


def frame_dropout(seq, n_drop=2):
    """
    Randomly zero out `n_drop` frames.
    Simulates momentary occlusion (hand passes in front of body, etc.)
    """
    augmented  = seq.copy()
    drop_idx   = random.sample(range(len(seq)), min(n_drop, len(seq)))
    augmented[drop_idx] = 0.0
    return augmented.astype(np.float32)


# ── list of all augmentation functions ────────────────────────────────────────
AUGMENTATIONS = [
    ("noise",    gaussian_noise),
    ("timewarp", time_warp),
    ("scale",    spatial_scale),
    ("hflip",    horizontal_flip),
    ("fdrop",    frame_dropout),
]


# ── main augmentation loop ─────────────────────────────────────────────────────
if __name__ == "__main__":
    if not TRAIN_DIR.is_dir():
        raise SystemExit(f"{TRAIN_DIR} not found — run step2_split_dataset.py first.")
    sign_labels = sorted(d for d in os.listdir(TRAIN_DIR) if (TRAIN_DIR / d).is_dir())
    print(f"Augmenting training data for {len(sign_labels)} signs ...\n")

    for label in sign_labels:
        label_dir = os.path.join(TRAIN_DIR, label)

        # collect original files only (not previously augmented ones)
        originals = sorted([
            f for f in os.listdir(label_dir)
            if f.endswith(".npy") and "_aug_" not in f
        ])
        n_orig = len(originals)
        print(f"[{label}]  {n_orig} original clips", end="")

        if n_orig == 0:
            print("  — skipped (no originals found)")
            continue

        aug_count = 0   # how many augmented files written so far

        # keep cycling through originals until we reach the target
        orig_cycle = 0
        while n_orig + aug_count < TARGET_PER_CLASS:
            # pick next original (cycle round-robin)
            fname    = originals[orig_cycle % n_orig]
            orig_cycle += 1

            seq = np.load(os.path.join(label_dir, fname)).astype(np.float32)

            # pick a random augmentation technique
            aug_name, aug_fn = random.choice(AUGMENTATIONS)

            # occasionally combine two augmentations (25 % chance)
            if random.random() < 0.25:
                aug_name2, aug_fn2 = random.choice(AUGMENTATIONS)
                aug_seq = aug_fn2(aug_fn(seq))
                combo_name = f"{aug_name}_{aug_name2}"
            else:
                aug_seq    = aug_fn(seq)
                combo_name = aug_name

            # build a unique filename so we never overwrite
            base      = fname.replace(".npy", "")
            save_name = f"{base}_aug_{combo_name}_{aug_count:04d}.npy"
            np.save(os.path.join(label_dir, save_name), aug_seq)
            aug_count += 1

        total = n_orig + aug_count
        print(f"  →  +{aug_count} augmented  =  {total} total")

    print("\nStep 3 complete. Training data augmented.")
