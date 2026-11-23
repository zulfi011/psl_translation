"""
step1_extract_landmarks.py
===========================
PURPOSE : Read every raw video clip and save a (T, 258) landmark array as .npy
RUN     : python scripts/step1_extract_landmarks.py

INPUT FOLDER STRUCTURE
----------------------
raw_data/
    eat/
        clip1.mp4
        clip2.mp4
        ...  (50 clips)
    drink/
        ...
    (one sub-folder per sign)

OUTPUT FOLDER STRUCTURE
-----------------------
data/landmarks/
    eat/
        clip1.mp4.npy   ← shape (60, 258)   fixed to 60 frames
    drink/
        ...

FEATURE LAYOUT PER FRAME  (258 values total)
--------------------------------------------
  [0   – 62 ]   Left  hand : 21 pts × 3 (x, y, z)
  [63  – 125]   Right hand : 21 pts × 3 (x, y, z)
  [126 – 257]   Body pose  : 33 pts × 4 (x, y, z, visibility)
  Face landmarks from Holistic are intentionally IGNORED.
"""

import os
import cv2
import numpy as np

from psl_translator.config import RAW_DIR, LANDMARKS_DIR as SAVE_DIR, SEQUENCE_LEN
from psl_translator.features import create_holistic, frame_to_features, fix_length

holistic = create_holistic()


def extract_video(video_path, show=False):
    """
    Extract raw landmark frames from a single video.
    Returns np.ndarray shape (T, 258) where T = number of video frames.
    """
    cap      = cv2.VideoCapture(str(video_path))
    sequence = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(rgb)

        sequence.append(frame_to_features(results))

        if show:
            cv2.imshow("Extracting", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break

    cap.release()
    if show:
        cv2.destroyAllWindows()

    return np.array(sequence, dtype=np.float32)   # (T, 258)


# ── main extraction loop ───────────────────────────────────────────────────────
if __name__ == "__main__":
    if not RAW_DIR.is_dir() or not any(d.is_dir() for d in RAW_DIR.iterdir()):
        raise SystemExit(f"No raw videos found in {RAW_DIR}. See data/README.md.")

    sign_labels = sorted(d for d in os.listdir(RAW_DIR) if (RAW_DIR / d).is_dir())
    print(f"Signs found: {sign_labels}\n")

    for label in sign_labels:
        label_raw  = os.path.join(RAW_DIR, label)
        label_save = os.path.join(SAVE_DIR, label)
        os.makedirs(label_save, exist_ok=True)

        clips = sorted(c for c in os.listdir(label_raw) if not c.startswith("."))
        print(f"[{label}]  {len(clips)} clips")

        for clip in clips:
            save_path = os.path.join(label_save, clip + ".npy")

            if os.path.exists(save_path):
                print(f"  Skipping : {clip}  (already extracted)")
                continue

            video_path = os.path.join(label_raw, clip)
            print(f"  Processing: {clip}", end=" ... ", flush=True)

            raw_seq   = extract_video(video_path, show=False)
            fixed_seq = fix_length(raw_seq, SEQUENCE_LEN)   # (60, 258)

            np.save(save_path, fixed_seq)
            print(f"saved  shape={fixed_seq.shape}")

    print("\nStep 1 complete. All landmarks saved to:", SAVE_DIR)
