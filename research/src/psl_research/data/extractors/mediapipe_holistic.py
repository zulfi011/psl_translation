"""
MediaPipe Holistic, hands + pose (face ignored), 258 features per frame.
Identical features to the FYP v1 model, but the full clip is stored: resampling
to a fixed length is now a configurable transform, not baked into the data.
"""

from pathlib import Path

import numpy as np

from ...registry import EXTRACTORS
from ..layout import N_FEATURES
from .base import Extractor


@EXTRACTORS.register("mediapipe_holistic_v1")
class MediaPipeHolisticV1(Extractor):
    name = "mediapipe_holistic_v1"
    n_features = N_FEATURES

    def __init__(self, model_complexity: int = 1, min_detection_confidence: float = 0.5,
                 min_tracking_confidence: float = 0.5):
        import mediapipe as mp
        self._mp = mp
        self._kw = dict(static_image_mode=False, model_complexity=model_complexity, smooth_landmarks=True,
                        min_detection_confidence=min_detection_confidence,
                        min_tracking_confidence=min_tracking_confidence)

    @staticmethod
    def frame_features(res) -> list[float]:
        def pts(lms, n, with_vis=False):
            if not lms:
                return [0.0] * (n * (4 if with_vis else 3))
            out = []
            for lm in lms.landmark:
                out.extend([lm.x, lm.y, lm.z, lm.visibility] if with_vis else [lm.x, lm.y, lm.z])
            return out
        return (pts(res.left_hand_landmarks, 21) + pts(res.right_hand_landmarks, 21)
                + pts(res.pose_landmarks, 33, with_vis=True))

    def extract(self, video_path: Path) -> np.ndarray:
        import cv2
        # fresh tracker per clip so tracking state never leaks between videos
        with self._mp.solutions.holistic.Holistic(**self._kw) as holistic:
            cap = cv2.VideoCapture(str(video_path))
            frames = []
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                res = holistic.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                frames.append(self.frame_features(res))
            cap.release()
        return np.asarray(frames, dtype=np.float32).reshape(-1, self.n_features)
