"""Step 2 — run a landmark extractor over every clip in the manifest.

    python scripts/extract_landmarks.py --extractor mediapipe_holistic_v1

Output: data/interim/<extractor>/<sign>/<clip>.npy  (full-length, not resampled).
Already-extracted clips are skipped, so it can be stopped and resumed.
"""
import argparse

import numpy as np
from tqdm import tqdm

from psl_research import paths
from psl_research.data import extractors  # noqa: F401  (register extractors)
from psl_research.data.manifest import load_manifest
from psl_research.registry import EXTRACTORS

ap = argparse.ArgumentParser()
ap.add_argument("--extractor", default="mediapipe_holistic_v1", choices=EXTRACTORS.names())
args = ap.parse_args()

ext = EXTRACTORS.build(args.extractor)
out_root = paths.INTERIM / args.extractor
for row in tqdm(load_manifest(), desc=args.extractor):
    out = out_root / f"{row['clip_id']}.npy"
    if out.exists():
        continue
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(out, ext.extract(paths.RAW / row["path"]))
ext.close()
print(f"→ {out_root}")
