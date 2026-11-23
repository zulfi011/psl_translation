"""
features.py
===========
MediaPipe Holistic feature extraction shared by every stage of the pipeline
(landmark extraction, real-time recognition and the Flask API).

FEATURE LAYOUT PER FRAME  (258 values total)
--------------------------------------------
  [0   – 62 ]   Left  hand : 21 pts × 3 (x, y, z)
  [63  – 125]   Right hand : 21 pts × 3 (x, y, z)
  [126 – 257]   Body pose  : 33 pts × 4 (x, y, z, visibility)
  Face landmarks from Holistic are intentionally IGNORED.
"""

import numpy as np

LEFT_HAND_SIZE  = 63    # 21 landmarks × 3
RIGHT_HAND_SIZE = 63    # 21 landmarks × 3
POSE_SIZE       = 132   # 33 landmarks × 4  (x, y, z, visibility)
TOTAL_FEATURES  = 258   # 63 + 63 + 132


def create_holistic():
    """Create a MediaPipe Holistic detector with the settings used in training."""
    import mediapipe as mp

    return mp.solutions.holistic.Holistic(
        static_image_mode=False,       # video / tracking mode
        model_complexity=1,
        smooth_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )


def frame_to_features(results):
    """
    Convert one Holistic result into a flat list of 258 floats.
    Missing detections → zero-filled section.
    """
    # left hand  (63 values)
    if results.left_hand_landmarks:
        left = []
        for lm in results.left_hand_landmarks.landmark:
            left.extend([lm.x, lm.y, lm.z])
    else:
        left = [0.0] * LEFT_HAND_SIZE

    # right hand  (63 values)
    if results.right_hand_landmarks:
        right = []
        for lm in results.right_hand_landmarks.landmark:
            right.extend([lm.x, lm.y, lm.z])
    else:
        right = [0.0] * RIGHT_HAND_SIZE

    # body pose  (132 values)  — no face mesh
    if results.pose_landmarks:
        pose = []
        for lm in results.pose_landmarks.landmark:
            pose.extend([lm.x, lm.y, lm.z, lm.visibility])
    else:
        pose = [0.0] * POSE_SIZE

    return left + right + pose   # 258 values


def fill_zero_frames(sequence):
    """
    Replace zero-hand frames with the last valid frame.
    Prevents zeros from corrupting the sequence at inference time.
    """
    sequence   = sequence.copy()
    last_valid = None

    for i in range(len(sequence)):
        hand_part = sequence[i, :126]  # first 126 = both hands
        if np.any(hand_part != 0):
            last_valid = sequence[i].copy()
        elif last_valid is not None:
            sequence[i] = last_valid
        # if no hand seen yet, leave as zeros

    return sequence


def fix_length(sequence, target=60):
    """
    Resize a (T, 258) sequence to exactly ``target`` frames using linear index
    sub-sampling (works for both longer and shorter clips).
    """
    T = len(sequence)
    if T == 0:
        return np.zeros((target, TOTAL_FEATURES), dtype=np.float32)

    indices = np.linspace(0, T - 1, target, dtype=int)
    return np.asarray(sequence, dtype=np.float32)[indices]
