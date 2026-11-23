"""
step2_split_dataset.py
=======================
PURPOSE : Split extracted .npy landmark files into train / val / test sets.
          Person-aware split:
            • Persons 1-4  →  train  (80 %)
            • Person 5     →  val / test  (split 50/50 → 10 % val, 10 % test)

RUN     : python scripts/step2_split_dataset.py

INPUT
-----
data/landmarks/
    eat/    1 (1).npy  1 (2).npy … 5 (n).npy
    drink/  ...

OUTPUT
------
data/
    split/
        train/  eat/ drink/ ...
        val/    eat/ drink/ ...
        test/   eat/ drink/ ...
models/
    labels.json   ← {"eat": 0, "drink": 1, ...}
"""

import os
import re
import json
import shutil
import random
import numpy as np

# ── paths ──────────────────────────────────────────────────────────────────────
from psl_translator.config import LANDMARKS_DIR, TRAIN_DIR, VAL_DIR, TEST_DIR, LABELS_JSON

# ── reproducibility ────────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# ── person assignment ──────────────────────────────────────────────────────────
TRAIN_PERSONS   = {1, 2, 3, 4}   # persons whose data goes entirely to train
HOLDOUT_PERSON  = 5               # person whose data is split into val / test
# 50 / 50 on the holdout person → val ≈ 10 %, test ≈ 10 % of total

def get_person_id(filename: str) -> int | None:
    """
    Extract the leading person number from filenames like:
        '1 (1).npy'  → 1
        '2 (10).npy' → 2
        '5 (3).npy'  → 5
    Returns None if the pattern is not matched.
    """
    match = re.match(r'^(\d+)\s*\(', filename)
    # print(match)
    return int(match.group(1)) if match else None

if not LANDMARKS_DIR.is_dir() or not any(d.is_dir() for d in LANDMARKS_DIR.iterdir()):
    raise SystemExit(f"No landmark folders in {LANDMARKS_DIR} — run step1_extract_landmarks.py first.")

# ── build label map ────────────────────────────────────────────────────────────
sign_labels = sorted(
    entry for entry in os.listdir(LANDMARKS_DIR)
    if os.path.isdir(os.path.join(LANDMARKS_DIR, entry))
)
label_map = {label: i for i, label in enumerate(sign_labels)}

print(f"Classes ({len(sign_labels)}): {sign_labels}")
print(f"Label map: {label_map}\n")

totals = {"train": 0, "val": 0, "test": 0}

# ── split per class ────────────────────────────────────────────────────────────
for label in sign_labels:
    src_dir = os.path.join(LANDMARKS_DIR, label)
    all_files = sorted(f for f in os.listdir(src_dir) if f.endswith(".npy"))

    train_files   = []
    holdout_files = []
    skipped       = []

    for fname in all_files:
        pid = get_person_id(fname)
        if pid is None:
            skipped.append(fname)
        elif pid in TRAIN_PERSONS:
            train_files.append(fname)
        elif pid == HOLDOUT_PERSON:
            holdout_files.append(fname)
        else:
            skipped.append(fname)   # unexpected person id — ignored

    if skipped:
        print(f"  ⚠  {label}: {len(skipped)} file(s) skipped "
              f"(unrecognised pattern): {skipped[:5]}")

    # shuffle holdout then split 50 / 50 into val / test
    random.shuffle(holdout_files)
    n_val       = len(holdout_files) // 2
    val_files   = holdout_files[:n_val]
    test_files  = holdout_files[n_val:]

    splits = {
        "train": train_files,
        "val":   val_files,
        "test":  test_files,
    }

    for split_name, split_files in splits.items():
        dest_dir = os.path.join(
            {"train": TRAIN_DIR, "val": VAL_DIR, "test": TEST_DIR}[split_name],
            label
        )
        os.makedirs(dest_dir, exist_ok=True)

        for fname in split_files:
            shutil.copy(
                os.path.join(src_dir, fname),
                os.path.join(dest_dir, fname),
            )

        totals[split_name] += len(split_files)
        print(f"  {label:20s}  {split_name:5s}  {len(split_files):3d} files")

# ── save label map ─────────────────────────────────────────────────────────────
os.makedirs(os.path.dirname(LABELS_JSON), exist_ok=True)
with open(LABELS_JSON, "w") as f:
    json.dump(label_map, f, indent=2)

print(f"\nSplit complete:")
print(f"  Train : {totals['train']} samples  (persons 1–4)")
print(f"  Val   : {totals['val']}   samples  (person 5 — first half)")
print(f"  Test  : {totals['test']}  samples  (person 5 — second half)")
print(f"Labels saved to: {LABELS_JSON}")